//! ANAI-245 / ANAI-242. The history safety valve: measurement, then policy.
//!
//! This module owns the one place where OpenFang throws conversation away
//! unconditionally — the trim that runs immediately before the LLM call in
//! both agent loops. ANAI-245 added the measurement; ANAI-242 replaced the
//! policy the measurement was pointed at.
//!
//! # What was wrong
//!
//! The valve capped the per-turn prompt at **20 messages**, a count with no
//! token arithmetic behind it. Three consequences, in increasing order of
//! badness:
//!
//! 1. On a 200k-token window, twenty messages is routinely under 1% of the
//!    budget. Conversation was destroyed to relieve pressure that did not
//!    exist.
//! 2. The compactor (30 messages / 70% of window) and the overflow recovery
//!    pipeline (70% / 90%) are the *designed* responses to context pressure,
//!    and both are more careful than a `drain()` — the compactor summarises
//!    what it removes. A cap of 20 sits below both, so on the per-turn vector
//!    the crude path ran first and the careful paths saw a prompt already
//!    cut down.
//! 3. Worst: the compactor's output — the summary of everything previously
//!    removed — is injected at **index 0**, and the valve drains **from the
//!    front**. Every trim on a compacted session deleted the summary of the
//!    conversation it had just paid an LLM call to preserve.
//!
//! # The policy now
//!
//! - An **explicit** `max_history_messages` in the manifest still means
//!   exactly what it said: a hard message count. Operators set 4-6 on
//!   short-lived workers deliberately, to keep `agent_send` results focused.
//!   That is intent, not a safety valve, and it is honoured verbatim.
//! - With no override, the valve is **token-aware**: it fires only when the
//!   estimated prompt crosses [`TOKEN_TRIM_RATIO`] of the model's real
//!   context window, and then removes only enough to get back under
//!   [`TOKEN_RELEASE_RATIO`] — hysteresis, so a session hovering at the
//!   threshold is not trimmed on every single turn.
//! - An [`ABSURDITY_CEILING`] on raw message count survives regardless. A
//!   runaway tool loop can emit hundreds of tiny messages that never trip a
//!   token threshold but do trip per-request message limits at the provider.
//!   Loops are real; the guard is cheap.
//! - The canonical-context message at index 0 is **preserved across the
//!   drain**. It is compaction output — the single densest message in the
//!   prompt — and dropping it first was precisely backwards.
//!
//! Nothing here removes the compactor or the overflow pipeline. The point is
//! to stop preempting them: this valve now fires at 85%, above the
//! compactor's 70% trigger, so the careful path gets first refusal.
//!
//! # The two vectors
//!
//! The most confusing thing about context handling in this codebase is that
//! two different message vectors are capped by two different mechanisms:
//!
//! - `session.messages` — the durable history. The compactor triggers on
//!   this one and it grows unbounded across turns.
//! - the per-turn `messages` copy — rebuilt from the session every turn,
//!   plus the canonical-context message and the `<turn_context>` envelope.
//!   The valve caps *this* one, and the cut never writes back.
//!
//! `session_messages` and `message_count` are both logged so the field data
//! shows that divergence rather than requiring the reader to know it already.

use openfang_types::message::Message;
use openfang_types::tool::ToolDefinition;
use std::sync::atomic::{AtomicUsize, Ordering};

/// ANAI-245 follow-up. Window utilisation at or above which a *quiet* turn —
/// one where the valve did not fire — is still logged at `info!`.
///
/// Instrumentation that only exists at `debug!` is instrumentation that is
/// not there: the deployed `RUST_LOG` filter is a module-path list, and a
/// custom target falls through it to the global default. The interesting
/// baseline — a session climbing toward the compactor's 70% trigger — must
/// survive an operator filter nobody remembers to update.
pub const NOTABLE_WINDOW_PCT: u32 = 40;

/// One quiet turn in every `N` below [`NOTABLE_WINDOW_PCT`] is promoted to
/// `info!`, so a wholly uneventful fleet still leaves a trickle of evidence
/// that the axis is alive and where idle sessions actually sit.
///
/// Sampled rather than unconditional: quiet turns are the overwhelming
/// majority, and one line per agent per turn at `info!` is log spam.
pub const QUIET_SAMPLE_EVERY: usize = 20;

/// Counts quiet turns process-wide for [`QUIET_SAMPLE_EVERY`].
///
/// Deliberately global rather than per-agent: the point is a bounded trickle
/// of baseline lines in the daemon log, not a per-agent guarantee.
static QUIET_TURNS: AtomicUsize = AtomicUsize::new(0);
use tracing::{debug, info};

/// Tracing target for every line this module emits.
///
/// Named so the whole axis isolates or silences independently of the rest of
/// the runtime: `RUST_LOG=context_pressure=info` to watch it alone,
/// `context_pressure=off` to mute it.
pub const TARGET: &str = "context_pressure";

/// Ratio of the context window at which the compactor's token trigger fires.
/// Mirrors `CompactionConfig::token_threshold_ratio`.
///
/// ANAI-244: the overflow pipeline's first stage used to fire here too, which
/// is why it preempted everything. It now enters at
/// `context_overflow::RECOVERY_ENTRY_RATIO` (0.92).
const SMART_PATH_RATIO: f64 = 0.70;

/// ANAI-242. Fraction of the context window at which the token-aware valve
/// fires.
///
/// Deliberately **above** [`SMART_PATH_RATIO`]: the compactor should get the
/// first attempt at relieving pressure, because it summarises what it
/// removes and this does not. Deliberately **below**
/// `context_overflow::RECOVERY_ENTRY_RATIO` (0.92), so the valve is still a
/// valve — it acts before the emergency path has to. ANAI-244 made that
/// second half true; it was aspiration until then.
pub const TOKEN_TRIM_RATIO: f64 = 0.85;

/// Fraction of the window the valve trims back *to* once it fires.
///
/// The gap between this and [`TOKEN_TRIM_RATIO`] is hysteresis. Trimming
/// only to the trigger point would leave the next turn one message from
/// firing again, and the valve would nibble the front of the conversation
/// every single turn.
pub const TOKEN_RELEASE_RATIO: f64 = 0.75;

/// ANAI-242. Hard ceiling on raw message count when no manifest override is
/// set, independent of token pressure.
///
/// A runaway tool loop emits many tiny messages: hundreds of them can sit
/// far under any token threshold while still breaking provider per-request
/// message limits. Token-awareness is the right default policy; it is not a
/// reason to remove the count backstop.
pub const ABSURDITY_CEILING: usize = 200;

/// Messages always left intact at the tail, whatever the arithmetic says.
///
/// Without a floor, a tiny context window plus a single enormous tool result
/// could compute a drain that removes the user's actual question.
pub const MIN_KEEP_RECENT: usize = 4;

/// Why the valve fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimReason {
    /// It did not.
    None,
    /// The manifest set an explicit message cap and the prompt exceeded it.
    ExplicitCap,
    /// No override, but raw message count crossed [`ABSURDITY_CEILING`].
    AbsurdityCeiling,
    /// No override, and estimated tokens crossed [`TOKEN_TRIM_RATIO`] of the
    /// real context window.
    TokenPressure,
}

impl TrimReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrimReason::None => "none",
            TrimReason::ExplicitCap => "explicit_cap",
            TrimReason::AbsurdityCeiling => "absurdity_ceiling",
            TrimReason::TokenPressure => "token_pressure",
        }
    }
}

/// What the valve intends to do, as data. Applying it is a separate step so
/// the decision is testable without a live prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimPlan {
    /// First index to remove. `1` when a canonical-context message is being
    /// preserved at index 0, else `0`.
    pub drain_from: usize,
    /// How many messages to remove starting at `drain_from`.
    pub drain_count: usize,
    /// Why.
    pub reason: TrimReason,
}

impl TrimPlan {
    /// The no-op plan.
    pub fn none() -> Self {
        Self {
            drain_from: 0,
            drain_count: 0,
            reason: TrimReason::None,
        }
    }

    pub fn fires(&self) -> bool {
        self.drain_count > 0
    }

    /// Apply the plan in place. Returns how many messages were removed.
    ///
    /// The caller is still responsible for re-validating tool_use/tool_result
    /// pairing afterwards — a drain can split a pair across the cut boundary,
    /// which strict providers reject.
    pub fn apply(&self, messages: &mut Vec<Message>) -> usize {
        if !self.fires() {
            return 0;
        }
        let end = (self.drain_from + self.drain_count).min(messages.len());
        if end <= self.drain_from {
            return 0;
        }
        messages.drain(self.drain_from..end);
        end - self.drain_from
    }
}

/// A measurement of context pressure taken immediately before the valve runs.
/// Pure data — constructing one has no side effects.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureObservation {
    /// Messages in the per-turn vector, before any trim.
    pub message_count: usize,
    /// Messages in the durable session history. Diverges from
    /// `message_count` because the trim never writes back.
    pub session_messages: usize,
    /// Estimated prompt size (chars/4 heuristic, the same one the compactor
    /// and the overflow recovery pipeline use — so the numbers are
    /// comparable across all three).
    pub estimated_tokens: usize,
    /// The model's real context window, or the 200k fallback.
    pub context_window: usize,
    /// `estimated_tokens` as a percentage of `context_window`, saturating.
    /// The headline number: a valve firing at 3% was a gratuitous cut.
    pub window_used_pct: u32,
    /// The manifest's explicit cap, if any.
    pub explicit_cap: Option<usize>,
    /// Was a canonical-context message injected at index 0 this turn?
    pub canonical_context_present: bool,
    /// Would the compactor's token trigger have fired on this prompt
    /// (70% of the window)?
    pub over_compactor_token_threshold: bool,
    /// Would the overflow recovery pipeline's first stage have fired
    /// (also 70%)? Same threshold today; kept distinct because ANAI-244 may
    /// move one and not the other.
    pub over_overflow_threshold: bool,
    /// What the valve decided to do about all of the above.
    pub plan: TrimPlan,
}

impl PressureObservation {
    /// Did the safety valve fire this turn?
    pub fn trimmed(&self) -> bool {
        self.plan.fires()
    }

    /// Is the canonical-context message being destroyed by this trim?
    ///
    /// Post-ANAI-242 this must be `false` on every turn — the plan never
    /// starts its drain at index 0 when the message is present. It is kept
    /// as a live field rather than deleted precisely so the logs keep
    /// asserting that, instead of the invariant living only in a test.
    pub fn canonical_context_dropped(&self) -> bool {
        self.canonical_context_present && self.plan.fires() && self.plan.drain_from == 0
    }

    /// The pathology ANAI-242 removed, still measured: the valve fired while
    /// neither smart path would have. Under the old count cap this was the
    /// common case; it should now only ever be true for an explicit
    /// operator-set cap or a runaway-loop ceiling hit.
    pub fn preempted_smart_paths(&self) -> bool {
        self.trimmed() && !self.over_compactor_token_threshold && !self.over_overflow_threshold
    }
}

/// Estimated token cost of one message, matching `estimate_token_count`'s
/// per-message arithmetic (text length plus framing overhead, chars/4).
fn message_tokens(msg: &Message) -> usize {
    (msg.content.text_length() + 16) / 4
}

/// Decide what the valve should do, and measure why.
///
/// `messages` must be the per-turn vector in its final pre-trim state —
/// after canonical-context injection and after `inject_turn_context` — so
/// the estimate reflects what would actually be sent.
///
/// The token estimate is not free (it serializes every tool schema), but the
/// same estimate is already computed once per loop iteration by
/// `recover_from_overflow`, so this adds one pass to a cost the turn was
/// paying regardless.
#[allow(clippy::too_many_arguments)]
pub fn observe(
    messages: &[Message],
    session_messages: usize,
    system_prompt: &str,
    tools: &[ToolDefinition],
    context_window: usize,
    explicit_cap: Option<usize>,
    canonical_context_present: bool,
) -> PressureObservation {
    let estimated_tokens =
        crate::compactor::estimate_token_count(messages, Some(system_prompt), Some(tools));

    let window_used_pct = if context_window > 0 {
        ((estimated_tokens as f64 / context_window as f64) * 100.0).round() as u32
    } else {
        0
    };

    let smart_threshold = (context_window as f64 * SMART_PATH_RATIO) as usize;
    let over_threshold = estimated_tokens > smart_threshold;

    let plan = plan_trim(
        messages,
        estimated_tokens,
        context_window,
        explicit_cap,
        canonical_context_present,
    );

    PressureObservation {
        message_count: messages.len(),
        session_messages,
        estimated_tokens,
        context_window,
        window_used_pct,
        explicit_cap,
        canonical_context_present,
        over_compactor_token_threshold: over_threshold,
        over_overflow_threshold: over_threshold,
        plan,
    }
}

/// The ANAI-242 policy itself, isolated from measurement and logging.
pub fn plan_trim(
    messages: &[Message],
    estimated_tokens: usize,
    context_window: usize,
    explicit_cap: Option<usize>,
    canonical_context_present: bool,
) -> TrimPlan {
    let len = messages.len();
    // Never drain index 0 when it holds compaction output. Everything below
    // computes counts against the drainable region only.
    let floor = usize::from(canonical_context_present && len > 0);
    // The tail the valve will not touch, plus the preserved head.
    let untouchable = floor + MIN_KEEP_RECENT;
    if len <= untouchable {
        return TrimPlan::none();
    }
    let max_drain = len - untouchable;

    // 1. Explicit operator intent. A worker agent pinned to 6 messages means
    //    it; this is a focus knob, not a safety valve, and it is honoured
    //    whatever the token arithmetic says.
    if let Some(cap) = explicit_cap {
        if len > cap {
            let want = len - cap;
            return TrimPlan {
                drain_from: floor,
                drain_count: want.min(max_drain),
                reason: TrimReason::ExplicitCap,
            };
        }
        return TrimPlan::none();
    }

    // 2. Runaway-loop backstop, count-based and window-independent.
    if len > ABSURDITY_CEILING {
        let want = len - ABSURDITY_CEILING;
        return TrimPlan {
            drain_from: floor,
            drain_count: want.min(max_drain),
            reason: TrimReason::AbsurdityCeiling,
        };
    }

    // 3. Token pressure. Fires above the compactor's trigger, releases to a
    //    lower mark so the valve does not nibble every turn.
    if context_window == 0 {
        return TrimPlan::none();
    }
    let trigger = (context_window as f64 * TOKEN_TRIM_RATIO) as usize;
    if estimated_tokens <= trigger {
        return TrimPlan::none();
    }
    let release = (context_window as f64 * TOKEN_RELEASE_RATIO) as usize;

    // Walk the drainable region from the front, shedding messages until the
    // estimate falls under the release mark or the tail floor is reached.
    let mut shed = 0usize;
    let mut remaining = estimated_tokens;
    let mut count = 0usize;
    for msg in messages.iter().skip(floor).take(max_drain) {
        if remaining <= release {
            break;
        }
        shed += message_tokens(msg);
        remaining = estimated_tokens.saturating_sub(shed);
        count += 1;
    }

    if count == 0 {
        return TrimPlan::none();
    }
    TrimPlan {
        drain_from: floor,
        drain_count: count,
        reason: TrimReason::TokenPressure,
    }
}

/// Provenance label for a kernel-resolved context window (ANAI-253).
///
/// `Some` means the model catalog answered; `None` means it did not and the
/// caller substituted [`DEFAULT_CONTEXT_WINDOW`](crate::agent_loop::DEFAULT_CONTEXT_WINDOW).
pub fn window_source(resolved: Option<usize>) -> &'static str {
    if resolved.is_some() {
        "catalog"
    } else {
        "fallback"
    }
}

/// Emit the observation.
///
/// Three tiers, chosen so the axis stays legible under the filter that is
/// actually deployed rather than the one the author had in mind:
///
/// - the valve fired — `info!`. The event under study.
/// - quiet, but utilisation is at or above [`NOTABLE_WINDOW_PCT`] — `info!`.
///   The approach to the event is what tells us whether 0.85/0.75 are the
///   right numbers.
/// - quiet and unremarkable — `debug!`, except one turn in
///   [`QUIET_SAMPLE_EVERY`], promoted to `info!` so the axis proves it is
///   alive even on an idle fleet.
///
/// `window_source` records where `obs.context_window` came from — `catalog`
/// when the kernel resolved a real entry, `fallback` when it could not and
/// [`DEFAULT_CONTEXT_WINDOW`](crate::agent_loop::DEFAULT_CONTEXT_WINDOW) was
/// substituted. ANAI-253: a hit and a miss used to produce byte-identical
/// lines, so no observation could be audited for whether the window it
/// measured against was real.
pub fn log(agent: &str, obs: &PressureObservation, streaming: bool, window_source: &str) {
    if obs.trimmed() {
        info!(
            target: TARGET,
            agent = %agent,
            streaming,
            message_count = obs.message_count,
            session_messages = obs.session_messages,
            estimated_tokens = obs.estimated_tokens,
            context_window = obs.context_window,
            window_source,
            window_used_pct = obs.window_used_pct,
            explicit_cap = ?obs.explicit_cap,
            trim_count = obs.plan.drain_count,
            trim_from = obs.plan.drain_from,
            reason = obs.plan.reason.as_str(),
            canonical_context_present = obs.canonical_context_present,
            canonical_context_dropped = obs.canonical_context_dropped(),
            over_compactor_token_threshold = obs.over_compactor_token_threshold,
            over_overflow_threshold = obs.over_overflow_threshold,
            preempted_smart_paths = obs.preempted_smart_paths(),
            "context pressure: safety valve firing"
        );
    } else if quiet_turn_is_loud(obs.window_used_pct) {
        info!(
            target: TARGET,
            agent = %agent,
            streaming,
            message_count = obs.message_count,
            session_messages = obs.session_messages,
            estimated_tokens = obs.estimated_tokens,
            context_window = obs.context_window,
            window_source,
            window_source,
            window_used_pct = obs.window_used_pct,
            explicit_cap = ?obs.explicit_cap,
            canonical_context_present = obs.canonical_context_present,
            over_compactor_token_threshold = obs.over_compactor_token_threshold,
            "context pressure: under cap"
        );
    } else {
        debug!(
            target: TARGET,
            agent = %agent,
            streaming,
            message_count = obs.message_count,
            session_messages = obs.session_messages,
            estimated_tokens = obs.estimated_tokens,
            context_window = obs.context_window,
            window_used_pct = obs.window_used_pct,
            explicit_cap = ?obs.explicit_cap,
            canonical_context_present = obs.canonical_context_present,
            over_compactor_token_threshold = obs.over_compactor_token_threshold,
            "context pressure: under cap"
        );
    }
}

/// Should this quiet turn be emitted at `info!`?
///
/// True when utilisation is notable, or when the process-wide quiet-turn
/// counter lands on a sample boundary. Advances that counter as a side
/// effect, so call it exactly once per quiet turn.
fn quiet_turn_is_loud(window_used_pct: u32) -> bool {
    if window_used_pct >= NOTABLE_WINDOW_PCT {
        return true;
    }
    let n = QUIET_TURNS.fetch_add(1, Ordering::Relaxed);
    n.is_multiple_of(QUIET_SAMPLE_EVERY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(n: usize, chars: usize) -> Vec<Message> {
        (0..n).map(|_| Message::user("x".repeat(chars))).collect()
    }

    // --- window provenance (ANAI-253) ---------------------------------

    #[test]
    fn a_resolved_window_is_labelled_catalog_and_an_unresolved_one_fallback() {
        // The whole point of the field: before it, a catalog hit and a
        // catalog miss produced byte-identical `context_pressure` lines, so
        // `context_window=200000` could not be read as evidence of anything.
        // `None` is the kernel saying it could not answer, not a window.
        assert_eq!(window_source(Some(32_000)), "catalog");
        assert_eq!(window_source(Some(200_000)), "catalog");
        assert_eq!(window_source(None), "fallback");
    }

    // --- measurement (ANAI-245) ---------------------------------------

    #[test]
    fn no_trim_under_pressure_threshold() {
        let obs = observe(&msgs(10, 100), 10, "sys", &[], 200_000, None, false);
        assert!(!obs.trimmed());
        assert!(!obs.preempted_smart_paths());
        assert!(!obs.canonical_context_dropped());
    }

    /// The old pathology, as a regression test. Twenty-five short messages
    /// used to trigger a five-message amputation; under ANAI-242 a prompt
    /// this small is simply not a problem.
    #[test]
    fn twenty_five_short_messages_is_no_longer_a_trim() {
        let obs = observe(&msgs(25, 100), 40, "sys", &[], 200_000, None, false);
        assert!(!obs.trimmed());
        assert!(obs.window_used_pct < 1);
        assert_eq!(obs.plan.reason, TrimReason::None);
    }

    #[test]
    fn zero_window_does_not_divide_by_zero() {
        let obs = observe(&msgs(5, 100), 5, "sys", &[], 0, None, false);
        assert_eq!(obs.window_used_pct, 0);
        assert!(!obs.trimmed());
    }

    #[test]
    fn session_count_is_recorded_separately() {
        let obs = observe(&msgs(21, 100), 340, "sys", &[], 200_000, None, false);
        assert_eq!(obs.message_count, 21);
        assert_eq!(obs.session_messages, 340);
    }

    // --- policy (ANAI-242) --------------------------------------------

    #[test]
    fn explicit_cap_is_honoured_verbatim() {
        // A focused worker pinned to 6 messages, nowhere near token pressure.
        let obs = observe(&msgs(20, 100), 20, "sys", &[], 200_000, Some(6), false);
        assert!(obs.trimmed());
        assert_eq!(obs.plan.reason, TrimReason::ExplicitCap);
        assert_eq!(obs.plan.drain_count, 14);
        assert_eq!(obs.plan.drain_from, 0);
    }

    #[test]
    fn explicit_cap_under_limit_does_not_fire() {
        let obs = observe(&msgs(5, 100), 5, "sys", &[], 200_000, Some(6), false);
        assert!(!obs.trimmed());
    }

    #[test]
    fn absurdity_ceiling_catches_a_runaway_loop() {
        // 300 tiny messages: ~30 tokens each, nowhere near 85% of 200k.
        let obs = observe(&msgs(300, 100), 300, "sys", &[], 200_000, None, false);
        assert!(obs.trimmed());
        assert_eq!(obs.plan.reason, TrimReason::AbsurdityCeiling);
        assert_eq!(obs.plan.drain_count, 100);
        assert!(!obs.over_compactor_token_threshold);
    }

    #[test]
    fn token_pressure_fires_above_the_trim_ratio() {
        // 20 messages x 40k chars = 800k chars = ~200k tokens on a 200k
        // window: far above 85%.
        let obs = observe(&msgs(20, 40_000), 20, "sys", &[], 200_000, None, false);
        assert!(obs.trimmed());
        assert_eq!(obs.plan.reason, TrimReason::TokenPressure);
        assert!(obs.over_compactor_token_threshold);
        assert!(!obs.preempted_smart_paths());
    }

    /// Hysteresis: the trim releases below the trigger, so a session sitting
    /// just over the line is not nibbled on every subsequent turn.
    #[test]
    fn token_trim_releases_below_the_trigger() {
        let mut m = msgs(40, 20_000); // ~200k tokens
                                      // Drive the plan with the REAL estimate, exactly as `observe` does:
                                      // the hysteresis arithmetic is only sound if the number the planner
                                      // sheds against is the number the prompt actually costs.
        let before = crate::compactor::estimate_token_count(&m, Some("sys"), Some(&[]));
        let plan = plan_trim(&m, before, 200_000, None, false);
        assert!(plan.fires());
        plan.apply(&mut m);
        let after = crate::compactor::estimate_token_count(&m, Some("sys"), Some(&[]));
        assert!(after <= (200_000.0 * TOKEN_RELEASE_RATIO) as usize);
        // And the trimmed prompt must not immediately re-trigger.
        assert!(!plan_trim(&m, after, 200_000, None, false).fires());
    }

    /// The bug ANAI-242 exists to kill: compaction output at index 0 used to
    /// be the first thing deleted. It must now survive every trim.
    #[test]
    fn canonical_context_survives_a_token_trim() {
        let obs = observe(&msgs(20, 40_000), 20, "sys", &[], 200_000, None, true);
        assert!(obs.trimmed());
        assert_eq!(obs.plan.drain_from, 1);
        assert!(!obs.canonical_context_dropped());
    }

    #[test]
    fn canonical_context_survives_an_explicit_cap_trim() {
        let obs = observe(&msgs(20, 100), 20, "sys", &[], 200_000, Some(6), true);
        assert!(obs.trimmed());
        assert_eq!(obs.plan.drain_from, 1);
        assert!(!obs.canonical_context_dropped());
    }

    #[test]
    fn apply_removes_exactly_the_planned_window() {
        let mut m: Vec<Message> = (0..10).map(|i| Message::user(format!("m{i}"))).collect();
        let plan = TrimPlan {
            drain_from: 1,
            drain_count: 4,
            reason: TrimReason::ExplicitCap,
        };
        assert_eq!(plan.apply(&mut m), 4);
        assert_eq!(m.len(), 6);
        assert_eq!(m[0].content.text_length(), "m0".len());
        assert_eq!(m[1].content.text_length(), "m5".len());
    }

    /// A tiny window plus a huge prompt must not eat the user's actual
    /// question. The tail floor holds.
    #[test]
    fn min_keep_recent_is_a_hard_floor() {
        let m = msgs(6, 100_000);
        let plan = plan_trim(&m, 150_000, 8_000, None, true);
        assert!(plan.drain_count <= 6 - 1 - MIN_KEEP_RECENT);
    }

    #[test]
    fn very_short_histories_are_never_trimmed() {
        for n in 0..=MIN_KEEP_RECENT {
            let m = msgs(n, 100_000);
            assert!(!plan_trim(&m, 999_999, 8_000, Some(1), false).fires());
        }
    }

    /// The valve now fires ABOVE the compactor's trigger, not below it, so
    /// the careful path gets first refusal.
    #[test]
    fn valve_trigger_sits_above_the_compactor_trigger() {
        const _: () = assert!(TOKEN_TRIM_RATIO > SMART_PATH_RATIO);
        const _: () = assert!(TOKEN_RELEASE_RATIO < TOKEN_TRIM_RATIO);
        // 75% of the window: over the compactor's 70%, under the valve's 85%.
        let obs = observe(&msgs(30, 20_000), 30, "sys", &[], 200_000, None, false);
        assert!(obs.over_compactor_token_threshold);
        assert!(!obs.trimmed());
    }

    /// A small-window model reaches its thresholds at a prompt a 200k model
    /// shrugs at. The policy must use the window it was handed.
    #[test]
    fn threshold_tracks_the_real_window() {
        let m = msgs(10, 4_000); // ~10k tokens
        let big = observe(&m, 10, "sys", &[], 200_000, None, false);
        let small = observe(&m, 10, "sys", &[], 8_000, None, false);
        assert!(!big.trimmed());
        assert!(small.trimmed());
        assert_eq!(small.plan.reason, TrimReason::TokenPressure);
    }

    // --- log tiering (ANAI-245 follow-up) ------------------------------

    /// A session climbing toward the compactor's trigger must reach `info!`
    /// without consuming a sample slot — that is the datum the 0.85/0.75
    /// numbers will be judged against.
    #[test]
    fn notable_utilisation_is_always_loud() {
        for pct in [NOTABLE_WINDOW_PCT, NOTABLE_WINDOW_PCT + 1, 99, 100] {
            assert!(quiet_turn_is_loud(pct));
        }
    }

    /// Unremarkable quiet turns are sampled, not silent and not spam:
    /// exactly one in [`QUIET_SAMPLE_EVERY`], whatever the counter's phase.
    #[test]
    fn unremarkable_quiet_turns_are_sampled() {
        let rounds = 3;
        let loud = (0..QUIET_SAMPLE_EVERY * rounds)
            .filter(|_| quiet_turn_is_loud(1))
            .count();
        assert_eq!(loud, rounds);
    }
}
