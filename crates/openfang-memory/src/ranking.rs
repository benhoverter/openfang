//! Kind-aware recall ranking (ANAI-232 shadow mode, ANAI-233 live weights).
//!
//! Vector recall today re-ranks its candidate window on pure cosine
//! similarity: a 20-line distilled episode summary competes head-to-head with
//! raw transcript sediment that outnumbers it ~1,760:1, and loses. The fix
//! (ANAI-233) is a per-kind multiplier on the cosine score, the same shape as
//! the `kind = 'fact'` boost that already exists on the *text-search* branch
//! and therefore only fires when the embedding driver is down.
//!
//! # What shadow mode answered
//!
//! The measurement came first and it came back unanimous. In one log window,
//! `"weighted ranking differs"` fired **11 times** against **0** `"no change"`
//! lines — every logged recall differed — and the direction never once
//! reversed: every `entered_kinds` was `summary` or `fact`, every `left_kinds`
//! was `turn`. On the deepest corpus in the fleet (130 candidates) the shipped
//! ranking surfaced `baseline_summaries=0`; the weights surfaced three. That
//! is ANAI-230's failure quantified — 216 turns out-vote 11 summaries on raw
//! cosine every time.
//!
//! So the weights go live, behind [`RecallWeights::enabled`]. What shadow mode
//! cannot say, and this module still does not claim, is whether the swap
//! produced a *better answer*: it proves composition changed, not that quality
//! improved. That judgment is qualitative and belongs to the operator reading
//! the results, which is precisely why the values are config rather than
//! constants.
//!
//! [`shadow_delta`] keeps computing the weighted-vs-cosine diff either way,
//! and [`log_shadow_delta`] emits one structured line per vector recall under
//! the `shadow_rank` target. Once the weights are live the diff has not
//! changed meaning, only sign: it now describes what the *shipped* ranking did
//! relative to pure cosine rather than what it declined to do. The line
//! carries `weights_live` so a log reader never has to guess which.
//!
//! Read the log with:
//! ```text
//! rg 'shadow_rank' ~/.openfang/logs/*.log
//! ```
//!
//! **Now configurable, under `[recall]`** — see [`RecallWeights`]. Shadow mode
//! deliberately had no knob, because a knob would have let someone retune the
//! thing being measured mid-measurement. That objection expires the moment the
//! weights have consequences, and an off-switch starts being worth its wiring.
//!
//! **No memory content is logged**, only ids, kinds and scores. The recall
//! corpus is other people's transcripts.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use tracing::info;

/// Compiled default multiplier for `summary` rows.
///
/// **1.25, raised from shadow mode's 1.15.** The timid value was chosen to
/// re-order rows that were already close; the log says close was not the
/// problem. `baseline_summaries=0` on a 130-candidate window is a summary
/// corpus that never reaches the cut line at all, not one that arrives just
/// under it. This is still a nudge and not an override —
/// [`weights_do_not_rescue_an_irrelevant_summary`] pins that, and if it ever
/// needs relaxing the weight is too high.
///
/// [`weights_do_not_rescue_an_irrelevant_summary`]: self::tests
pub const DEFAULT_WEIGHT_SUMMARY: f32 = 1.25;

/// Compiled default multiplier for `fact` rows.
///
/// **1.0 — neutral, lowered from shadow mode's 1.25, and this is an
/// inversion**: facts used to outrank summaries and now they do not.
///
/// The original reasoning was sound when it was written: a fact is one durable
/// claim someone deliberately asserted, bounded at one row per
/// `(agent_id, scope, claim_key)`, so it cannot flood a result set. What has
/// changed is that facts no longer need recall to reach the prompt. They
/// arrive through the rehydration pack (ANAI-247), through `memory_fact` reads
/// carrying staleness markers (ANAI-259), and through the managed `MEMORY.md`
/// block (ANAI-212). Boosting them *here* as well spends the context window
/// twice on the same handful of claims, and it does so by displacing the one
/// kind that has no other door — summaries.
///
/// Neutral rather than demoted on purpose: see [`MIN_WEIGHT`].
pub const DEFAULT_WEIGHT_FACT: f32 = 1.0;

/// Everything else, `turn` included: unweighted.
///
/// This is the identity element on purpose. Weighting is expressed as
/// promotion of the distilled kinds, never as demotion of transcript — a
/// multiplier below 1.0 on `turn` would be indistinguishable in the log from
/// a boost on everything else, and would quietly change the meaning of a
/// score that other code compares against a threshold.
pub const WEIGHT_DEFAULT: f32 = 1.0;

/// Floor for a configured weight: **1.0, i.e. neutral**.
///
/// Config can promote a kind; it cannot demote one. That is the module's
/// doctrine (see [`WEIGHT_DEFAULT`]) expressed as a validation rule rather
/// than left as a comment, because the doctrine is load-bearing: with every
/// weight `>= 1.0`, a weighted score is only ever raised relative to cosine,
/// so the sentinel ordering and any threshold comparison downstream keep their
/// meaning. It also means "turn this kind down" is not expressible, which is
/// correct — the way to make one kind matter less is to make the others matter
/// more, and the log can tell those apart.
pub const MIN_WEIGHT: f32 = 1.0;

/// Ceiling for a configured weight.
///
/// 3.0 is far above anything the evidence supports; it exists to catch a
/// misplaced decimal point (`12.5` for `1.25`), which would stop being a
/// ranking policy and start being a filter that returns summaries and nothing
/// else regardless of the query.
pub const MAX_WEIGHT: f32 = 3.0;

/// The fleet's per-kind recall weights, and whether they are live.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallWeights {
    /// Apply the weights to what recall actually returns.
    ///
    /// `false` is the compiled default, so this module lands inert and the
    /// operator flips it on a second bounce — the two-bounce rule that
    /// separates "the code is in the binary" from "the behaviour changed", so
    /// a regression has one candidate cause instead of two. Shadow logging
    /// runs either way; disabling the weights does not blind the instrument.
    pub enabled: bool,
    /// Multiplier for `summary` rows. See [`DEFAULT_WEIGHT_SUMMARY`].
    pub summary: f32,
    /// Multiplier for `fact` rows. See [`DEFAULT_WEIGHT_FACT`].
    pub fact: f32,
}

impl Default for RecallWeights {
    fn default() -> Self {
        Self {
            enabled: false,
            summary: DEFAULT_WEIGHT_SUMMARY,
            fact: DEFAULT_WEIGHT_FACT,
        }
    }
}

/// Reject weights that are not multipliers.
///
/// Pure, so the rules are testable without touching the process-globals every
/// other test in this binary reads through — the same split
/// `validate_working_set_ratio` uses, for the same reason.
pub fn validate_weights(w: RecallWeights) -> Result<(), String> {
    for (name, v) in [("summary", w.summary), ("fact", w.fact)] {
        if !v.is_finite() {
            return Err(format!(
                "recall weight {name} must be a finite number, got {v}"
            ));
        }
        if v < MIN_WEIGHT {
            return Err(format!(
                "recall weight {name} ({v}) is below the {MIN_WEIGHT} floor: weighting \
                 promotes a kind, it never demotes one — raise the other weights instead"
            ));
        }
        if v > MAX_WEIGHT {
            return Err(format!(
                "recall weight {name} ({v}) is above the {MAX_WEIGHT} ceiling: at that \
                 multiplier the kind is a filter, not a preference (misplaced decimal?)"
            ));
        }
    }
    Ok(())
}

// `0` is the nothing-installed sentinel: `0.0f32` has an all-zero bit pattern
// and is refused by `validate_weights` anyway, so it can never be a real
// value. `ENABLED` needs no sentinel — its compiled default *is* `false`.
static WEIGHT_SUMMARY_BITS: AtomicU32 = AtomicU32::new(0);
static WEIGHT_FACT_BITS: AtomicU32 = AtomicU32::new(0);
static WEIGHTS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Install the fleet weights at boot. Refuses the value, not the boot.
///
/// Same trade as `install_working_set_ratio` (ANAI-260) and
/// [`crate::staleness::install_policy`] (ANAI-259): a mistyped tuning number
/// logs an error and leaves the compiled defaults live, rather than taking a
/// hundred agents offline at 3am over a config typo.
///
/// Note what a refusal keeps: the compiled defaults, `enabled` included. A
/// bad weight therefore leaves recall on pure cosine rather than half-applying
/// an operator's intent.
pub fn install_weights(w: RecallWeights) -> Result<(), String> {
    validate_weights(w)?;
    WEIGHT_SUMMARY_BITS.store(w.summary.to_bits(), Ordering::Relaxed);
    WEIGHT_FACT_BITS.store(w.fact.to_bits(), Ordering::Relaxed);
    WEIGHTS_ENABLED.store(w.enabled, Ordering::Relaxed);
    Ok(())
}

fn load(cell: &AtomicU32, fallback: f32) -> f32 {
    match cell.load(Ordering::Relaxed) {
        0 => fallback,
        bits => f32::from_bits(bits),
    }
}

/// The live weights: whatever was installed, else the compiled defaults.
pub fn weights() -> RecallWeights {
    RecallWeights {
        enabled: WEIGHTS_ENABLED.load(Ordering::Relaxed),
        summary: load(&WEIGHT_SUMMARY_BITS, DEFAULT_WEIGHT_SUMMARY),
        fact: load(&WEIGHT_FACT_BITS, DEFAULT_WEIGHT_FACT),
    }
}

/// Whether recall should rank on weighted scores rather than raw cosine.
///
/// Read by the recall path to decide what ships. Everything else in this
/// module — [`shadow_delta`], [`weight_for_kind`] — ignores it, because the
/// measurement has to keep answering "what would the weights do" while they
/// are off.
pub fn weights_enabled() -> bool {
    WEIGHTS_ENABLED.load(Ordering::Relaxed)
}

/// Multiplier for a row's `kind`, under the live weights.
///
/// `None` — the ~46k pre-v13 rows that carry no discriminator at all — takes
/// the default, same as an unrecognised kind. A future kind must opt *in* to
/// promotion; inheriting it silently is how a ranking policy stops being a
/// policy.
pub fn weight_for_kind(kind: Option<&str>) -> f32 {
    let w = weights();
    match kind {
        Some(crate::episode::SUMMARY_KIND) => w.summary,
        Some(crate::fact::KIND_FACT) => w.fact,
        _ => WEIGHT_DEFAULT,
    }
}

/// A candidate's score under the weights.
///
/// The one place the sentinel rule lives, so the shadow diff and the shipped
/// sort cannot disagree about it. Rows with a negative similarity are the
/// no-embedding sentinel and are left unweighted: multiplying a sentinel by
/// 1.25 makes it *more* negative, which would "demote" a row for the crime of
/// being promoted. Skipping them keeps the sentinel meaning "last" under both
/// rankings.
pub fn weighted_score(similarity: f32, kind: Option<&str>) -> f32 {
    if similarity < 0.0 {
        similarity
    } else {
        similarity * weight_for_kind(kind)
    }
}

/// One candidate row, reduced to what ranking needs and nothing more.
#[derive(Debug, Clone)]
pub struct ShadowCandidate {
    /// Row id, for correlating a log line back to a memory.
    pub id: String,
    /// `kind` column value; `None` for pre-v13 rows.
    pub kind: Option<String>,
    /// Cosine similarity against the query embedding, as the shipped ranking
    /// computed it. Rows with no embedding arrive here as a negative sentinel.
    pub similarity: f32,
}

/// What the weights would have changed, had they been live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowDelta {
    /// Size of the candidate window that was re-ranked.
    pub candidates: usize,
    /// Number of rows actually returned (`limit`, or fewer).
    pub returned: usize,
    /// Ids that the weighted ranking would have added to the returned set.
    pub entered: Vec<String>,
    /// Ids that the weighted ranking would have dropped from it.
    pub left: Vec<String>,
    /// Kinds of the rows in `entered`, in the same order. Answers "did this
    /// actually promote a summary, or just shuffle turns?" without a join.
    pub entered_kinds: Vec<String>,
    /// Kinds of the rows in `left`. Answers "what did it cost?"
    pub left_kinds: Vec<String>,
    /// True when the weighted ranking changes the order of the returned set
    /// even though its membership is identical. Position matters: the recall
    /// block is read top-down and the per-kind budget (ANAI-231) is spent in
    /// order.
    pub reordered: bool,
    /// Count of summary rows in the shipped top-`limit`.
    pub baseline_summaries: usize,
    /// Count of summary rows in the weighted top-`limit`.
    pub weighted_summaries: usize,
}

impl ShadowDelta {
    /// Did the weights change anything at all? Used to keep the log quiet on
    /// the (expected, common) no-op case.
    pub fn is_noop(&self) -> bool {
        self.entered.is_empty() && self.left.is_empty() && !self.reordered
    }
}

/// Rank `candidates` both ways and diff the top-`limit`.
///
/// The baseline reproduces the shipped sort exactly: descending similarity,
/// **stable**, so ties keep the SQL candidate-window order. The weighted pass
/// differs in one respect only — the sort key is `similarity ×
/// weight_for_kind(kind)`.
///
/// "Baseline" stays pure cosine even once the weights are live, so the diff
/// keeps measuring the same thing across the switch. What changes is which
/// side of it shipped — [`log_shadow_delta`] records that as `weights_live`.
///
/// The no-embedding sentinel is handled in [`weighted_score`].
pub fn shadow_delta(candidates: &[ShadowCandidate], limit: usize) -> ShadowDelta {
    let mut baseline: Vec<usize> = (0..candidates.len()).collect();
    baseline.sort_by(|&a, &b| {
        candidates[b]
            .similarity
            .partial_cmp(&candidates[a].similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let score = |c: &ShadowCandidate| -> f32 { weighted_score(c.similarity, c.kind.as_deref()) };
    let mut weighted: Vec<usize> = (0..candidates.len()).collect();
    weighted.sort_by(|&a, &b| {
        score(&candidates[b])
            .partial_cmp(&score(&candidates[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = limit.min(candidates.len());
    let base_top = &baseline[..n];
    let weighted_top = &weighted[..n];

    let entered_idx: Vec<usize> = weighted_top
        .iter()
        .copied()
        .filter(|i| !base_top.contains(i))
        .collect();
    let left_idx: Vec<usize> = base_top
        .iter()
        .copied()
        .filter(|i| !weighted_top.contains(i))
        .collect();

    let is_summary =
        |i: &usize| candidates[*i].kind.as_deref() == Some(crate::episode::SUMMARY_KIND);
    let kind_label = |i: &usize| {
        candidates[*i]
            .kind
            .clone()
            .unwrap_or_else(|| "<none>".to_string())
    };

    ShadowDelta {
        candidates: candidates.len(),
        returned: n,
        entered: entered_idx
            .iter()
            .map(|i| candidates[*i].id.clone())
            .collect(),
        entered_kinds: entered_idx.iter().map(kind_label).collect(),
        left: left_idx.iter().map(|i| candidates[*i].id.clone()).collect(),
        left_kinds: left_idx.iter().map(kind_label).collect(),
        reordered: entered_idx.is_empty() && left_idx.is_empty() && base_top != weighted_top,
        baseline_summaries: base_top.iter().filter(|i| is_summary(i)).count(),
        weighted_summaries: weighted_top.iter().filter(|i| is_summary(i)).count(),
    }
}

/// Emit one line per vector recall under the `shadow_rank` target.
///
/// No-ops are logged too, and at `info` like everything else here — **the
/// no-op count is the denominator**. "Weights changed the top-5 nine times"
/// means nothing without "out of how many recalls", and a shadow run that
/// only records the interesting cases cannot distinguish "the weights rarely
/// fire" from "the instrumentation never ran". Those have opposite
/// consequences for ANAI-233.
///
/// `debug` would have been the tidier level and would also have made the
/// denominator invisible: the fleet default is `info` (no `log_level` in
/// `config.toml`), so a debug line is a line nobody reads. Volume is one
/// entry per vector recall — roughly one per agent turn — which is noise the
/// log can carry for 48 hours.
///
/// Every line carries `weights_live`. Without it the log is ambiguous the
/// moment ANAI-233's switch is flipped: the same "entered / left" fields mean
/// "what recall declined to do" when the weights are off and "what recall
/// did" when they are on, and a reader comparing two days of logs has no way
/// to tell which they are holding.
pub fn log_shadow_delta(agent_id: &str, candidates: &[ShadowCandidate], limit: usize) {
    let delta = shadow_delta(candidates, limit);
    let live = weights_enabled();
    if delta.is_noop() {
        info!(
            target: "shadow_rank",
            agent = agent_id,
            candidates = delta.candidates,
            returned = delta.returned,
            summaries = delta.baseline_summaries,
            weights_live = live,
            "shadow rank: no change"
        );
        return;
    }
    // Two call sites rather than one with a computed message: `info!` takes a
    // literal format string, and the two phrasings are worth keeping distinct
    // anyway — they are what an operator greps for.
    macro_rules! differs {
        ($msg:literal) => {
            info!(
                target: "shadow_rank",
                agent = agent_id,
                candidates = delta.candidates,
                returned = delta.returned,
                entered = ?delta.entered,
                entered_kinds = ?delta.entered_kinds,
                left = ?delta.left,
                left_kinds = ?delta.left_kinds,
                reordered = delta.reordered,
                baseline_summaries = delta.baseline_summaries,
                weighted_summaries = delta.weighted_summaries,
                weights_live = live,
                $msg
            )
        };
    }
    if live {
        differs!("recall rank: weights changed the returned set");
    } else {
        differs!("shadow rank: weighted ranking differs");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: &str, kind: Option<&str>, sim: f32) -> ShadowCandidate {
        ShadowCandidate {
            id: id.to_string(),
            kind: kind.map(|k| k.to_string()),
            similarity: sim,
        }
    }

    /// The kind strings this module weights must be the strings the rest of
    /// the crate writes. A rename elsewhere has to fail here loudly rather
    /// than silently reverting every summary to the default weight — the
    /// same guard ANAI-231 put on the budget table, for the same reason.
    #[test]
    fn kind_spellings_match_the_constants() {
        // Asserted against the *defaults*, not against `weights()`, so this
        // stays a spelling test: comparing the function to the table it reads
        // would pass no matter which branch the spelling took. No test in this
        // module installs weights, so the defaults are what is live here.
        assert_eq!(
            weight_for_kind(Some(crate::episode::SUMMARY_KIND)),
            DEFAULT_WEIGHT_SUMMARY
        );
        assert_ne!(
            weight_for_kind(Some(crate::episode::SUMMARY_KIND)),
            WEIGHT_DEFAULT,
            "a misspelled summary kind would silently fall through to the default"
        );
        assert_eq!(
            weight_for_kind(Some(crate::fact::KIND_FACT)),
            DEFAULT_WEIGHT_FACT
        );
        assert_eq!(
            weight_for_kind(Some(crate::semantic::KIND_TURN)),
            WEIGHT_DEFAULT
        );
    }

    #[test]
    fn unknown_and_absent_kinds_take_the_default() {
        assert_eq!(weight_for_kind(None), WEIGHT_DEFAULT);
        assert_eq!(weight_for_kind(Some("some-future-kind")), WEIGHT_DEFAULT);
    }

    /// The inversion ANAI-233 landed, pinned as a claim rather than left to
    /// be inferred from two float literals: summaries now outrank facts, and
    /// facts are neutral because they reach the prompt through three other
    /// doors (ANAI-247 packs, ANAI-259 reads, ANAI-212 MEMORY.md).
    #[test]
    fn defaults_promote_summaries_over_facts() {
        let d = RecallWeights::default();
        assert!(
            d.summary > d.fact,
            "summary must outrank fact: recall is the only door a summary has"
        );
        assert_eq!(d.fact, WEIGHT_DEFAULT, "facts are neutral, not demoted");
        assert!(d.summary > WEIGHT_DEFAULT);
    }

    /// Lands inert. The behaviour change is a second bounce, not this one.
    #[test]
    fn weights_are_off_by_default() {
        assert!(!RecallWeights::default().enabled);
    }

    /// The `[recall]` defaults in `openfang-types` are a hand-written mirror
    /// of the constants above — unavoidably, since `openfang-types` cannot
    /// depend on this crate — so the mirror is pinned here, where both are
    /// visible. This is the ANAI-260 step-1 lesson applied prospectively: a
    /// constant with two homes drifts, and the drift is silent.
    #[test]
    fn config_defaults_mirror_the_compiled_defaults() {
        let cfg = openfang_types::config::RecallConfig::default();
        let compiled = RecallWeights::default();
        assert_eq!(cfg.kind_weights_enabled, compiled.enabled);
        assert_eq!(cfg.summary_weight as f32, compiled.summary);
        assert_eq!(cfg.fact_weight as f32, compiled.fact);
        // And the mirrored values must survive the install validator, or the
        // fleet would boot on a refused config every time.
        assert!(validate_weights(RecallWeights {
            enabled: cfg.kind_weights_enabled,
            summary: cfg.summary_weight as f32,
            fact: cfg.fact_weight as f32,
        })
        .is_ok());
    }

    #[test]
    fn validation_refuses_demotion_and_absurd_promotion() {
        assert!(validate_weights(RecallWeights::default()).is_ok());
        assert!(validate_weights(RecallWeights {
            summary: 3.0,
            fact: 1.0,
            ..Default::default()
        })
        .is_ok());

        // Below neutral: demotion is not expressible by config.
        let e = validate_weights(RecallWeights {
            fact: 0.5,
            ..Default::default()
        })
        .unwrap_err();
        assert!(e.contains("fact"), "{e}");
        assert!(e.contains("promotes"), "{e}");

        // Misplaced decimal point.
        assert!(validate_weights(RecallWeights {
            summary: 12.5,
            ..Default::default()
        })
        .is_err());
        assert!(validate_weights(RecallWeights {
            summary: f32::NAN,
            ..Default::default()
        })
        .is_err());
    }

    /// A weight of exactly 1.0 must leave the ranking alone, not merely leave
    /// it *nearly* alone — this is what makes `fact = 1.0` an opt-out rather
    /// than a rounding accident.
    #[test]
    fn neutral_weight_is_the_identity() {
        assert_eq!(weighted_score(0.61, Some("turn")), 0.61);
        assert_eq!(weighted_score(-1.0, Some("summary")), -1.0);
    }

    /// The point of the whole exercise: a summary just below the cut line is
    /// promoted past a turn just above it.
    #[test]
    fn summary_enters_the_top_set() {
        let cands = vec![
            c("t1", Some("turn"), 0.80),
            c("t2", Some("turn"), 0.75),
            c("s1", Some("summary"), 0.70),
        ];
        let d = shadow_delta(&cands, 2);
        assert_eq!(d.entered, vec!["s1"]);
        assert_eq!(d.entered_kinds, vec!["summary"]);
        assert_eq!(d.left, vec!["t2"]);
        assert_eq!(d.baseline_summaries, 0);
        assert_eq!(d.weighted_summaries, 1);
        assert!(!d.is_noop());
    }

    /// 1.15 is a nudge, not an override: a summary that genuinely does not
    /// match must stay out. If this test ever needs relaxing, the weight is
    /// too high.
    #[test]
    fn weights_do_not_rescue_an_irrelevant_summary() {
        let cands = vec![
            c("t1", Some("turn"), 0.90),
            c("t2", Some("turn"), 0.85),
            c("s1", Some("summary"), 0.20),
        ];
        let d = shadow_delta(&cands, 2);
        assert!(d.entered.is_empty());
        assert!(d.is_noop());
    }

    /// Same membership, different order, still worth logging — the recall
    /// block is read top-down and ANAI-231's budget is spent in order.
    #[test]
    fn reorder_within_the_top_set_is_reported() {
        let cands = vec![c("t1", Some("turn"), 0.80), c("s1", Some("summary"), 0.75)];
        let d = shadow_delta(&cands, 2);
        assert!(d.entered.is_empty());
        assert!(d.left.is_empty());
        assert!(d.reordered);
        assert!(!d.is_noop());
    }

    /// The no-embedding sentinel must not be "promoted" into being more
    /// negative than an unweighted one — that would reorder rows that are all
    /// equally unrankable.
    #[test]
    fn negative_sentinel_is_not_weighted() {
        let cands = vec![
            c("s_noemb", Some("summary"), -1.0),
            c("t_noemb", Some("turn"), -1.0),
            c("t1", Some("turn"), 0.10),
        ];
        let d = shadow_delta(&cands, 3);
        // All three are returned, so membership cannot change; the sentinel
        // pair must also keep candidate-window order.
        assert!(d.entered.is_empty());
        assert!(d.left.is_empty());
        assert!(!d.reordered);
    }

    #[test]
    fn limit_larger_than_candidates_is_clamped() {
        let cands = vec![c("s1", Some("summary"), 0.5)];
        let d = shadow_delta(&cands, 5);
        assert_eq!(d.returned, 1);
        assert_eq!(d.candidates, 1);
        assert!(d.is_noop());
    }

    #[test]
    fn empty_candidate_window_is_a_noop() {
        let d = shadow_delta(&[], 5);
        assert_eq!(d.returned, 0);
        assert!(d.is_noop());
    }

    /// The inversion, at the ranking layer. This test previously asserted the
    /// opposite — `fact_outweighs_summary` — and its reversal *is* ANAI-233's
    /// behaviour change, so it is edited rather than deleted.
    #[test]
    fn summary_outweighs_fact() {
        let cands = vec![
            c("s1", Some("summary"), 0.60),
            c("f1", Some("fact"), 0.60),
            c("t1", Some("turn"), 0.61),
        ];
        let d = shadow_delta(&cands, 1);
        assert_eq!(d.entered, vec!["s1"]);
        assert_eq!(d.left, vec!["t1"]);

        // And a fact no longer displaces a turn on its own: neutral means
        // neutral, so the top slot is decided by cosine alone between them.
        let cands = vec![c("f1", Some("fact"), 0.60), c("t1", Some("turn"), 0.61)];
        let d = shadow_delta(&cands, 1);
        assert!(
            d.is_noop(),
            "a neutral fact must not reorder against a turn"
        );
    }
}
