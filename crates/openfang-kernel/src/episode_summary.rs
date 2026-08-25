//! ANAI-220: episode **close → summary**, the second daemon-owned model call.
//!
//! # Why this exists
//!
//! Episodes have closed with a null title and a null summary since the day
//! they landed: nothing has ever promoted a batch of turns into durable
//! knowledge. The corpus reflects it — six deliberate notes fleet-wide in four
//! months, because hand-curation is a thing nobody does. This is the pass that
//! writes the summary the agent was never going to write itself.
//!
//! # The three invariants
//!
//! 1. **The close never waits on a provider.** The close commits first with a
//!    null summary (ANAI-219's sweep, or an explicit `memory_episode_close`);
//!    this runs afterwards, on its own tick, out of transaction. A provider
//!    outage costs a summary, never a close. That is the exact inverse of the
//!    gatekeeper's fail-closed policy, and the reason `background_complete`
//!    reports rather than decides.
//! 2. **No answer means no summary — and no retry.** Every failure variant
//!    leaves `summary` NULL, logs once, and moves on. Episodes age out of
//!    [`SUMMARY_LOOKBACK_MINUTES`] and stay null; recovering them is a
//!    backfill's job, deliberately not this one's.
//! 3. **Re-running is a no-op.** `set_summary` writes only into a NULL, and
//!    `has_summary_row` guards the recallable row. Both halves must hold for a
//!    later backfill to be safe to point at this same code path.
//!
//! # Its own task, not the sweep's
//!
//! This runs on a task gated on `consolidation.enabled`, adjacent to the idle
//! sweep rather than inside it. Folding it into the sweep would mean
//! `episode_idle_timeout_minutes = 0` silently also disabling summarisation —
//! and explicit closes, which need no timer at all, would never be summarised
//! on a fleet that leaves the timer off (which is the default). Two unrelated
//! behaviours behind one knob is the coupling ANAI-219 already refused once.

use crate::kernel::OpenFangKernel;
use chrono::{Duration, Utc};
use openfang_memory::episode::{
    Episode, EPISODE_ID_KEY, MAX_MATERIAL_ROWS, MIN_MATERIAL_ROWS, SUMMARY_KIND,
    SUMMARY_LOOKBACK_MINUTES,
};
use openfang_runtime::background_llm::{
    BackgroundFailure, BackgroundLlmOutcome, BackgroundLlmRequest, BackgroundPurpose,
};
use openfang_types::memory::MemorySource;
use std::collections::HashMap;
use tracing::{info, warn};

/// Tick cadence for the summariser task, in seconds.
///
/// Matched to the idle sweep's on purpose: summarisation happens *on close*, so
/// the close cadence already is the summary cadence, and `[memory.consolidation]`
/// deliberately carries no interval key of its own (ANAI-226). A tick that
/// finds nothing does one indexed `COUNT` against `episodes` and returns.
pub(crate) const CONSOLIDATION_TICK_SECS: u64 = 60;

/// Ticks an open breaker waits before this call site tries **one** probe call.
///
/// The breaker in `BackgroundLlmState` never self-recovers — by design, since
/// the gatekeeper wants a wedged judge to stay wedged until an operator looks
/// at it. A once-a-minute background sweep wants the opposite: a provider that
/// was down for five minutes should not cost summarisation until the next
/// daemon restart. So recovery lives here, at the call site, as the module
/// header of `background_llm` says it should.
///
/// **Caveat (ANAI-228, not mine to fix):** if the *driver* failed to construct
/// on the very first call, the `OnceLock` has cached that failure for the life
/// of the process and no probe can rescue it. This probe recovers provider
/// errors and timeouts, not a poisoned driver cell.
///
/// **Coupled to [`SUMMARY_LOOKBACK_MINUTES`].** At [`CONSOLIDATION_TICK_SECS`]
/// this is ~30 minutes, and two failed probe intervals exceed the 60-minute
/// lookback — so a provider outage of about an hour orphans that hour's
/// episodes to a future backfill even after the probe recovers. That boundary
/// is accepted, not accidental; it stops being accepted if these two constants
/// drift apart. Move them together.
const PROBE_AFTER_TICKS: u32 = 30;

/// Per-material-row character ceiling fed to the model.
///
/// `max_tokens` bounds what comes back; this bounds what goes out, so one
/// pathological row cannot turn a deliberately cheap call into an expensive
/// one.
const MAX_ROW_CHARS: usize = 1_200;

const SYSTEM_PROMPT: &str = "\
You summarise one AI agent's work episode for that agent's own long-term memory.

Write only from the material given. Do not infer, advise, speculate, or add \
context you were not shown. If the material is thin, say less — a short honest \
summary is correct; an invented one is not.

Respond in exactly this shape:

TITLE: <at most eight words>
<the summary: at most 120 words, plain prose, no bullet points, no preamble>";

/// Live state for the summariser task: everything the loop must remember
/// between ticks. Currently just the half-open probe's clock.
pub(crate) struct EpisodeSummarizer {
    /// Ticks observed since the breaker was last seen open without a probe.
    ticks_since_probe: u32,
}

impl EpisodeSummarizer {
    pub(crate) fn new() -> Self {
        Self {
            ticks_since_probe: 0,
        }
    }
}

/// One tick's tally, kept separate from the loop so the log line is assembled
/// in one place and cannot drift from what actually happened.
#[derive(Default)]
struct Tally {
    summarized: u32,
    /// Skipped for having less than [`MIN_MATERIAL_ROWS`] linked rows.
    no_material: u32,
    /// The model was called and produced nothing usable.
    failed: u32,
}

impl OpenFangKernel {
    /// One summariser tick. Called from the task spawned in `kernel.rs`.
    pub(crate) async fn consolidate_closed_episodes(&self, state: &mut EpisodeSummarizer) {
        let cfg = &self.config.memory.consolidation;
        if !cfg.enabled {
            return;
        }

        // Bounded by close recency, NOT by "every null summary ever". The
        // unbounded form of this query is the backfill; see
        // `SUMMARY_LOOKBACK_MINUTES`.
        let cutoff = Utc::now() - Duration::minutes(SUMMARY_LOOKBACK_MINUTES);
        let (candidates, pending) = match self
            .memory
            .episodes_awaiting_summary_async(cutoff, cfg.max_per_tick as usize)
            .await
        {
            Ok(found) => found,
            Err(e) => {
                warn!(target: "openfang::consolidation", error = %e,
                      "Could not read episodes awaiting summary");
                return;
            }
        };
        if candidates.is_empty() {
            return;
        }

        // Half-open probe. An open breaker normally means "do not call", but a
        // once-a-minute sweep that respects that forever loses summarisation
        // until restart over a provider blip.
        let mut probing = false;
        if self
            .background_llm
            .circuit_open(BackgroundPurpose::Consolidation, cfg.failure_threshold)
        {
            state.ticks_since_probe = state.ticks_since_probe.saturating_add(1);
            if state.ticks_since_probe < PROBE_AFTER_TICKS {
                return;
            }
            state.ticks_since_probe = 0;
            probing = true;
            // Clearing the count is what lets exactly one call through. A
            // failed probe slams it straight back open below rather than
            // spending `failure_threshold` more calls to re-trip.
            self.background_llm
                .note_success(BackgroundPurpose::Consolidation);
            info!(target: "openfang::consolidation",
                  "Consolidation breaker half-open: probing with one episode");
        } else {
            state.ticks_since_probe = 0;
        }

        let mut tally = Tally::default();
        for ep in &candidates {
            let material = match self
                .memory
                .episode_material_async(ep.id, MAX_MATERIAL_ROWS)
                .await
            {
                Ok(rows) => rows,
                Err(e) => {
                    warn!(target: "openfang::consolidation", episode = %ep.id, error = %e,
                          "Could not read episode material");
                    continue;
                }
            };

            // The floor is on stored rows, never on `turn_count`.
            //
            // The "82 turns / 6 rows" divergence this comment used to cite was
            // a measurement artifact, not a capture bug (ANAI-229): it counted
            // `json_extract(metadata, '$.episode_id')`, but since the v13-era
            // column lift `semantic.rs` REMOVES that key from the JSON and
            // stores it in the `memories.episode_id` column, rehydrating it
            // only on read. Counted on the column, the live fleet runs ~1:1
            // (61/61, 54/54, 14/14). Do not re-derive this from the JSON.
            //
            // The floor stays, because the two quantities still cannot be
            // assumed equal: `ensure_open` bumps the turn before the write, so
            // a failed or in-flight `remember` leaves a gap (one live episode
            // sits at 90/88), `material` excludes soft-deleted rows, and a
            // turn that stores nothing is a turn with nothing to summarise.
            // Trusting `turn_count` would spend a model call on a polished null.
            if material.len() < MIN_MATERIAL_ROWS {
                tally.no_material += 1;
                continue;
            }

            let call = BackgroundLlmRequest {
                purpose: BackgroundPurpose::Consolidation,
                provider: cfg.provider.clone(),
                // Verbatim to the driver — there is no empty-means-default
                // fallback for `model`, only for `provider`.
                model: cfg.model.clone(),
                system: Some(SYSTEM_PROMPT.to_string()),
                user: summary_prompt(ep, &material),
                max_tokens: cfg.max_tokens,
                timeout_secs: cfg.timeout_secs,
                failure_threshold: cfg.failure_threshold,
            };

            let parsed = match self.background_complete(&call).await {
                BackgroundLlmOutcome::Answered(text) => match parse_summary(&text) {
                    Some(parsed) => Some(parsed),
                    None => {
                        warn!(target: "openfang::consolidation", episode = %ep.id,
                              raw = %openfang_types::truncate_str(&text, 120),
                              "Summariser returned nothing usable — leaving the summary null");
                        None
                    }
                },
                BackgroundLlmOutcome::Failed(BackgroundFailure::CircuitOpen) => {
                    // Tripped mid-batch by an earlier episode. Stop: the rest
                    // of this batch would pay latency for answers it will not
                    // get, and they are still pending next tick.
                    break;
                }
                BackgroundLlmOutcome::Failed(f) => {
                    warn!(target: "openfang::consolidation", episode = %ep.id, failure = %f,
                          "Summary call failed — leaving the summary null");
                    None
                }
            };

            let Some((title, summary)) = parsed else {
                tally.failed += 1;
                self.record_summary_failure(cfg.failure_threshold, probing);
                // One failure is enough for one tick. Serialised on purpose
                // (concurrency 1), and hammering a sick provider eight times a
                // minute is how a hiccup becomes an incident.
                break;
            };

            self.background_llm
                .note_success(BackgroundPurpose::Consolidation);
            probing = false;

            // Recallable row FIRST, episode column second. If the process dies
            // between them the episode is still selected next tick and
            // `has_summary_row` stops the row being written twice — the
            // degradation is structural rather than a thing the caller has to
            // remember. The other order would strand a summarised episode with
            // nothing in the corpus, invisible and never retried.
            self.write_summary_row(ep, title.as_deref(), &summary).await;

            match self
                .memory
                .set_episode_summary_async(ep.id, title, summary)
                .await
            {
                Ok(true) => tally.summarized += 1,
                // Something wrote a summary between the select and here — an
                // explicit close by the agent itself, most likely. Its author
                // was there and this one was not; leave it alone.
                //
                // Known shape, not a surprise (alpha, ANAI-220 review): the
                // corpus row above is already written, so the episode ends up
                // carrying *their* summary while recall surfaces *ours*. Two
                // summaries of one episode, disagreeing. Narrow — it needs an
                // explicit close inside a single tick's select-to-write gap —
                // and the alternative orderings are worse: column-first strands
                // summarised episodes with nothing recallable, and holding a
                // lock across a model call puts the provider on the close path.
                Ok(false) => {}
                Err(e) => {
                    warn!(target: "openfang::consolidation", episode = %ep.id, error = %e,
                          "Could not store episode summary");
                }
            }
        }

        // `max_per_tick` deferred work is logged, never silently dropped: a cap
        // that reports nothing reads as "we did them all". Anything still
        // pending — clipped by the cap, skipped for thin material, or left by a
        // failure — is named here.
        let deferred = pending
            .saturating_sub(tally.summarized as usize)
            .saturating_sub(tally.no_material as usize);
        //
        // `no_material` deliberately does NOT fire this line on its own: a thin
        // episode re-selects every tick for the life of the lookback window, so
        // announcing it would `info!` ~60 times about work that costs nothing.
        // It is still reported as a field whenever the line does fire.
        if tally.summarized > 0 || deferred > 0 {
            info!(
                target: "openfang::consolidation",
                summarized = tally.summarized,
                deferred,
                skipped_no_material = tally.no_material,
                failed = tally.failed,
                "consolidation: {} summarized, {deferred} deferred to next tick",
                tally.summarized
            );
        }
    }

    /// Record a failed summary call against the breaker, logging exactly once —
    /// on the call that trips it, not every 60 seconds thereafter.
    fn record_summary_failure(&self, threshold: u32, probing: bool) {
        if probing {
            // A failed probe goes straight back to open. Walking it up one
            // failure at a time would spend `threshold` more unattended calls
            // against a provider that just told us it is not there.
            let mut failures = 0;
            while failures < threshold {
                failures = self
                    .background_llm
                    .note_failure(BackgroundPurpose::Consolidation);
            }
            warn!(target: "openfang::consolidation",
                  "Consolidation breaker probe failed — breaker re-opened");
            return;
        }
        let failures = self
            .background_llm
            .note_failure(BackgroundPurpose::Consolidation);
        if failures == threshold {
            warn!(
                target: "openfang::consolidation",
                failures,
                probe_after_ticks = PROBE_AFTER_TICKS,
                "Consolidation circuit breaker OPEN — episodes will close with null summaries"
            );
        }
    }

    /// Write the summary into the recallable corpus as one embedded row.
    ///
    /// This is the point of the whole slice. An `episodes.summary` column is
    /// invisible to `memory_recall`; a `memories` row is not, and it is what
    /// turns a closed episode into something a future turn can actually find.
    async fn write_summary_row(&self, ep: &Episode, title: Option<&str>, summary: &str) {
        match self.memory.episode_has_summary_row_async(ep.id).await {
            Ok(true) => return,
            Ok(false) => {}
            Err(e) => {
                warn!(target: "openfang::consolidation", episode = %ep.id, error = %e,
                      "Could not check for an existing summary row — skipping the write");
                return;
            }
        }

        let content = match title {
            Some(t) => format!("{t}\n\n{summary}"),
            None => summary.to_string(),
        };

        let embedding = match self.embedding_driver {
            Some(ref driver) => match driver.embed_one(&content).await {
                Ok(vec) => Some(vec),
                Err(e) => {
                    // Store unembedded rather than lose it: the row stays
                    // findable by text search and `update_embedding` can
                    // backfill the vector later.
                    warn!(target: "openfang::consolidation", error = %e,
                          "Summary embedding failed; storing without a vector");
                    None
                }
            },
            None => None,
        };

        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert(
            crate::kernel::MEMORY_KIND_KEY.to_string(),
            serde_json::Value::String(SUMMARY_KIND.to_string()),
        );
        metadata.insert(
            EPISODE_ID_KEY.to_string(),
            serde_json::Value::String(ep.id.to_string()),
        );

        if let Err(e) = self
            .memory
            .remember_with_embedding_async(
                ep.agent_id,
                &content,
                // Inference, not Observation: nobody saw this: it is derived
                // from rows the daemon already held. `memory_note` is the
                // Observation case — an agent recording something itself.
                MemorySource::Inference,
                crate::kernel::MEMORY_NOTE_SCOPE,
                metadata,
                embedding.as_deref(),
            )
            .await
        {
            warn!(target: "openfang::consolidation", episode = %ep.id, error = %e,
                  "Could not store the episode summary row");
        }
    }
}

/// Build the user-role body for one summary call.
fn summary_prompt(ep: &Episode, material: &[String]) -> String {
    let mut out = String::new();
    out.push_str("Episode material, oldest first.\n\n");
    out.push_str(&format!(
        "Opened: {}\nLast activity: {}\nTurns: {}\nStored rows shown: {}\n\n",
        ep.opened_at.to_rfc3339(),
        ep.last_activity_at.to_rfc3339(),
        ep.turn_count,
        material.len(),
    ));
    for (i, row) in material.iter().enumerate() {
        out.push_str(&format!(
            "--- {} ---\n{}\n",
            i + 1,
            openfang_types::truncate_str(row, MAX_ROW_CHARS)
        ));
    }
    out
}

/// Parse the summariser's answer into `(title, summary)`.
///
/// Forgiving about the title, strict about the body: a title is a nicety, the
/// summary is the product. `None` means "nothing usable", which the caller
/// treats as a failure — and a failure means the episode keeps its null
/// summary, never a fabricated one.
fn parse_summary(raw: &str) -> Option<(Option<String>, String)> {
    let text = raw.trim();
    let mut title = None;
    let mut body = text;

    if let Some(rest) = text.strip_prefix("TITLE:") {
        let (line, remainder) = rest.split_once('\n').unwrap_or((rest, ""));
        let candidate = line.trim();
        if !candidate.is_empty() {
            title = Some(openfang_types::truncate_str(candidate, 120).to_string());
        }
        body = remainder.trim();
    }

    if body.is_empty() {
        return None;
    }
    Some((title, body.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_well_formed_answer_splits_into_title_and_body() {
        let (title, body) =
            parse_summary("TITLE: landed the summariser\nWired episode close to a summary.")
                .unwrap();
        assert_eq!(title.as_deref(), Some("landed the summariser"));
        assert_eq!(body, "Wired episode close to a summary.");
    }

    /// The title is optional. A model that ignores the format but says
    /// something useful still produced a summary, and throwing it away would
    /// cost a real call to enforce a cosmetic contract.
    #[test]
    fn a_missing_title_is_not_a_failure() {
        let (title, body) = parse_summary("  Wired episode close to a summary.  ").unwrap();
        assert_eq!(title, None);
        assert_eq!(body, "Wired episode close to a summary.");
    }

    /// A title with no body is NOT a summary. Accepting it would write a
    /// headline into memory as though it were the content.
    #[test]
    fn a_title_with_no_body_is_a_failure() {
        assert!(parse_summary("TITLE: something happened").is_none());
        assert!(parse_summary("TITLE: something happened\n   \n").is_none());
    }

    /// An empty or whitespace answer is a failure, not an empty summary. The
    /// difference matters: a failure leaves NULL and the null is honest.
    #[test]
    fn an_empty_answer_is_a_failure() {
        assert!(parse_summary("").is_none());
        assert!(parse_summary("   \n\t ").is_none());
    }

    /// A model talked into an essay for a title cannot put an essay in the
    /// title column.
    #[test]
    fn a_runaway_title_is_truncated() {
        let long = "x".repeat(400);
        let (title, _) = parse_summary(&format!("TITLE: {long}\nbody")).unwrap();
        assert!(title.unwrap().chars().count() <= 120);
    }
}
