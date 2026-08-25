//! Kind-aware recall ranking — **shadow mode** (ANAI-232).
//!
//! Vector recall today re-ranks its candidate window on pure cosine
//! similarity: a 20-line distilled episode summary competes head-to-head with
//! raw transcript sediment that outnumbers it ~1,760:1, and loses. The fix
//! (ANAI-233) is a per-kind multiplier on the cosine score, the same shape as
//! the `kind = 'fact'` boost that already exists on the *text-search* branch
//! and therefore only fires when the embedding driver is down.
//!
//! This module is the measurement that has to come first. Nothing here
//! changes what recall returns. [`shadow_delta`] computes the ranking the
//! weights *would* have produced, diffs it against the ranking that actually
//! shipped, and [`log_shadow_delta`] emits one structured line per vector
//! recall under the `shadow_rank` target. After ~48h of real traffic the log
//! answers the question I refuse to answer by feel: does a 1.15× nudge move
//! summaries into the top-5 at all, and what does it displace when it does?
//!
//! Read the log with:
//! ```text
//! rg 'shadow_rank' ~/.openfang/logs/*.log
//! ```
//!
//! **Deliberately not configurable.** The weights below are constants, not
//! `config.toml` keys. Shadow mode has no behaviour to tune — a knob here
//! would only let someone retune the thing being measured mid-measurement.
//! The config plumbing lands with ANAI-233, where the weights start having
//! consequences and an off-switch starts being worth its wiring.
//!
//! **No memory content is logged**, only ids, kinds and scores. The recall
//! corpus is other people's transcripts.

use tracing::info;

/// Candidate multiplier for `summary` rows.
///
/// Starting point, not a conclusion — the whole purpose of shadow mode is to
/// find out whether this number is right. 1.15 is deliberately timid: it
/// re-orders rows that were already close, and cannot drag a genuinely
/// irrelevant summary past a well-matched turn. If the log shows summaries
/// still never reaching the top-5, the answer is a bigger number, and we will
/// have the evidence to justify it rather than the vibe.
pub const WEIGHT_SUMMARY: f32 = 1.15;

/// Candidate multiplier for `fact` rows.
///
/// Higher than [`WEIGHT_SUMMARY`] because a fact is a single durable claim
/// that someone deliberately asserted, and because the live population is
/// bounded at one row per `(agent_id, scope, claim_key)` by the v14 partial
/// unique index — facts cannot flood a result set the way summaries
/// eventually could.
pub const WEIGHT_FACT: f32 = 1.25;

/// Everything else, `turn` included: unweighted.
///
/// This is the identity element on purpose. Weighting is expressed as
/// promotion of the distilled kinds, never as demotion of transcript — a
/// multiplier below 1.0 on `turn` would be indistinguishable in the log from
/// a boost on everything else, and would quietly change the meaning of a
/// score that other code compares against a threshold.
pub const WEIGHT_DEFAULT: f32 = 1.0;

/// Multiplier for a row's `kind`.
///
/// `None` — the ~46k pre-v13 rows that carry no discriminator at all — takes
/// the default, same as an unrecognised kind. A future kind must opt *in* to
/// promotion; inheriting it silently is how a ranking policy stops being a
/// policy.
pub fn weight_for_kind(kind: Option<&str>) -> f32 {
    match kind {
        Some(crate::episode::SUMMARY_KIND) => WEIGHT_SUMMARY,
        Some(crate::fact::KIND_FACT) => WEIGHT_FACT,
        _ => WEIGHT_DEFAULT,
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
/// Rows with a negative similarity are the no-embedding sentinel. They are
/// left unweighted: multiplying a sentinel by 1.25 makes it *more* negative,
/// which would "demote" a row for the crime of being promoted. Skipping them
/// keeps the sentinel meaning "last" under both rankings.
pub fn shadow_delta(candidates: &[ShadowCandidate], limit: usize) -> ShadowDelta {
    let mut baseline: Vec<usize> = (0..candidates.len()).collect();
    baseline.sort_by(|&a, &b| {
        candidates[b]
            .similarity
            .partial_cmp(&candidates[a].similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let weighted_score = |c: &ShadowCandidate| -> f32 {
        if c.similarity < 0.0 {
            c.similarity
        } else {
            c.similarity * weight_for_kind(c.kind.as_deref())
        }
    };
    let mut weighted: Vec<usize> = (0..candidates.len()).collect();
    weighted.sort_by(|&a, &b| {
        weighted_score(&candidates[b])
            .partial_cmp(&weighted_score(&candidates[a]))
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
        entered: entered_idx.iter().map(|i| candidates[*i].id.clone()).collect(),
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
pub fn log_shadow_delta(agent_id: &str, candidates: &[ShadowCandidate], limit: usize) {
    let delta = shadow_delta(candidates, limit);
    if delta.is_noop() {
        info!(
            target: "shadow_rank",
            agent = agent_id,
            candidates = delta.candidates,
            returned = delta.returned,
            summaries = delta.baseline_summaries,
            "shadow rank: no change"
        );
        return;
    }
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
        "shadow rank: weighted ranking differs"
    );
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
        assert_eq!(weight_for_kind(Some(crate::episode::SUMMARY_KIND)), WEIGHT_SUMMARY);
        assert_eq!(weight_for_kind(Some(crate::fact::KIND_FACT)), WEIGHT_FACT);
        assert_eq!(weight_for_kind(Some(crate::semantic::KIND_TURN)), WEIGHT_DEFAULT);
    }

    #[test]
    fn unknown_and_absent_kinds_take_the_default() {
        assert_eq!(weight_for_kind(None), WEIGHT_DEFAULT);
        assert_eq!(weight_for_kind(Some("some-future-kind")), WEIGHT_DEFAULT);
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
        let cands = vec![
            c("t1", Some("turn"), 0.80),
            c("s1", Some("summary"), 0.75),
        ];
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

    /// Facts outrank summaries at equal similarity, per the constants.
    #[test]
    fn fact_outweighs_summary() {
        let cands = vec![
            c("s1", Some("summary"), 0.60),
            c("f1", Some("fact"), 0.60),
            c("t1", Some("turn"), 0.61),
        ];
        let d = shadow_delta(&cands, 1);
        assert_eq!(d.entered, vec!["f1"]);
        assert_eq!(d.left, vec!["t1"]);
    }
}
