//! ANAI-245. Context-pressure instrumentation for the history safety valve.
//!
//! Log-only. This module measures the state of the world at the one place
//! where OpenFang throws conversation away unconditionally — the
//! `messages.drain(..trim_count)` safety valve in both agent loops — and
//! emits a single structured line per turn. It changes no behaviour.
//!
//! It exists because the ANAI-240 epic rests on a claim that had never been
//! measured: *most sessions are amputated at 20 messages for no reason.* The
//! preemption half of that claim is arithmetic and can be read off the source
//! — but "for no reason" is an empirical question about how much of the
//! context window was actually in use at the moment of the cut, and that
//! needs field data. ANAI-242 changes the policy; this ships first so the
//! logs bracket the change and the new threshold is picked from evidence
//! instead of taste.
//!
//! # The two vectors
//!
//! The single most confusing thing about context handling here is that two
//! different message vectors are capped by two different mechanisms:
//!
//! - `session.messages` — the durable history. The compactor triggers on
//!   this one (`threshold: 30` messages, or 70% of the window in tokens).
//!   It grows unbounded across turns.
//! - the per-turn `messages` copy — built fresh from the session each turn,
//!   plus the canonical-context message and the `<turn_context>` envelope.
//!   The safety valve caps *this* one at 20 and the cap does not persist.
//!
//! `session_messages` and `message_count` are both recorded so the field data
//! shows the divergence rather than requiring the reader to already know it.
//!
//! # The canonical-context hazard
//!
//! The compactor's output — the summary of everything that came before —
//! reaches the LLM as a message inserted at **index 0**. The safety valve
//! drains **from the front**. So whenever the valve fires on a session that
//! has been compacted, the first thing deleted is the summary of everything
//! that was previously deleted. `canonical_context_dropped` is the field that
//! catches that; it is the single most damning number this module can print,
//! and ANAI-242 exists in large part to drive it to zero.

use openfang_types::message::Message;
use openfang_types::tool::ToolDefinition;
use tracing::{debug, info};

/// Tracing target for every line this module emits.
///
/// Named so the whole axis isolates or silences independently of the rest of
/// the runtime: `RUST_LOG=context_pressure=info` to watch it alone,
/// `context_pressure=off` to mute it.
pub const TARGET: &str = "context_pressure";

/// A measurement of context pressure taken immediately before the safety
/// valve runs. Pure data — constructing one has no side effects.
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
    /// This is the headline number: if the valve fires at 3%, the cut was
    /// gratuitous.
    pub window_used_pct: u32,
    /// The effective cap — manifest override or the runtime default.
    pub max_history: usize,
    /// How many messages the valve is about to delete. Zero means it did not
    /// fire this turn.
    pub trim_count: usize,
    /// Was a canonical-context message injected at index 0 this turn?
    pub canonical_context_present: bool,
    /// Is the valve about to delete it? Compaction output, discarded.
    pub canonical_context_dropped: bool,
    /// Would the compactor's token trigger have fired on this prompt
    /// (70% of the window)? Recorded to show how far below its own trigger
    /// the valve operates.
    pub over_compactor_token_threshold: bool,
    /// Would the overflow recovery pipeline's first stage have fired
    /// (also 70%)? Same threshold today; kept as a distinct field because
    /// ANAI-244 may move one and not the other.
    pub over_overflow_threshold: bool,
}

impl PressureObservation {
    /// Did the safety valve fire this turn?
    pub fn trimmed(&self) -> bool {
        self.trim_count > 0
    }

    /// The claim ANAI-240 is built on, evaluated per turn: the valve fired
    /// while neither smart path would have. A turn where this is true is a
    /// turn where conversation was destroyed purely because a message
    /// *count* was exceeded, with the window nowhere near full.
    pub fn preempted_smart_paths(&self) -> bool {
        self.trimmed() && !self.over_compactor_token_threshold && !self.over_overflow_threshold
    }
}

/// Ratio of the context window at which the compactor's token trigger and
/// the overflow pipeline's first stage both fire. Mirrors
/// `CompactionConfig::token_threshold_ratio` and the `0.70` in
/// `context_overflow::recover_from_overflow`.
const SMART_PATH_RATIO: f64 = 0.70;

/// Measure context pressure at the safety valve.
///
/// `messages` must be the per-turn vector in its final pre-trim state — after
/// canonical-context injection and after `inject_turn_context` — so the
/// estimate reflects what would actually be sent.
///
/// The token estimate is not free (it serializes every tool schema), but the
/// same estimate is already computed once per loop iteration by
/// `recover_from_overflow`, so this adds one pass to a cost the turn was
/// paying regardless.
pub fn observe(
    messages: &[Message],
    session_messages: usize,
    system_prompt: &str,
    tools: &[ToolDefinition],
    context_window: usize,
    max_history: usize,
    canonical_context_present: bool,
) -> PressureObservation {
    let estimated_tokens =
        crate::compactor::estimate_token_count(messages, Some(system_prompt), Some(tools));

    let window_used_pct = if context_window > 0 {
        ((estimated_tokens as f64 / context_window as f64) * 100.0).round() as u32
    } else {
        0
    };

    let trim_count = messages.len().saturating_sub(max_history);
    let smart_threshold = (context_window as f64 * SMART_PATH_RATIO) as usize;
    let over_threshold = estimated_tokens > smart_threshold;

    PressureObservation {
        message_count: messages.len(),
        session_messages,
        estimated_tokens,
        context_window,
        window_used_pct,
        max_history,
        trim_count,
        canonical_context_present,
        // The drain takes from the front, so any trim at all takes index 0.
        canonical_context_dropped: canonical_context_present && trim_count > 0,
        over_compactor_token_threshold: over_threshold,
        over_overflow_threshold: over_threshold,
        // NOTE: both fields read the same threshold today by construction.
        // See the doc comment — they are separate so ANAI-244 can move one.
    }
}

/// Emit the observation.
///
/// `info!` when the valve fires (that is the event under study), `debug!`
/// otherwise so a quiet turn still leaves a baseline datum for anyone who
/// turns the level up, without spamming production logs.
pub fn log(agent: &str, obs: &PressureObservation, streaming: bool) {
    if obs.trimmed() {
        info!(
            target: TARGET,
            agent = %agent,
            streaming,
            message_count = obs.message_count,
            session_messages = obs.session_messages,
            estimated_tokens = obs.estimated_tokens,
            context_window = obs.context_window,
            window_used_pct = obs.window_used_pct,
            max_history = obs.max_history,
            trim_count = obs.trim_count,
            canonical_context_present = obs.canonical_context_present,
            canonical_context_dropped = obs.canonical_context_dropped,
            over_compactor_token_threshold = obs.over_compactor_token_threshold,
            over_overflow_threshold = obs.over_overflow_threshold,
            preempted_smart_paths = obs.preempted_smart_paths(),
            "context pressure: safety valve firing"
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
            max_history = obs.max_history,
            canonical_context_present = obs.canonical_context_present,
            over_compactor_token_threshold = obs.over_compactor_token_threshold,
            "context pressure: under cap"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(n: usize, chars: usize) -> Vec<Message> {
        (0..n).map(|_| Message::user("x".repeat(chars))).collect()
    }

    #[test]
    fn no_trim_under_cap() {
        let obs = observe(&msgs(10, 100), 10, "sys", &[], 200_000, 20, false);
        assert_eq!(obs.trim_count, 0);
        assert!(!obs.trimmed());
        assert!(!obs.preempted_smart_paths());
        assert!(!obs.canonical_context_dropped);
    }

    #[test]
    fn trim_count_is_the_overage() {
        let obs = observe(&msgs(25, 100), 25, "sys", &[], 200_000, 20, false);
        assert_eq!(obs.trim_count, 5);
        assert!(obs.trimmed());
    }

    /// The ANAI-240 thesis, as a test: 25 short messages blow the count cap
    /// while using a rounding error of the window.
    #[test]
    fn small_prompt_over_count_cap_preempts_both_smart_paths() {
        let obs = observe(&msgs(25, 100), 40, "sys", &[], 200_000, 20, false);
        assert!(obs.trimmed());
        assert!(!obs.over_compactor_token_threshold);
        assert!(!obs.over_overflow_threshold);
        assert!(obs.preempted_smart_paths());
        assert!(obs.window_used_pct < 1);
    }

    /// A genuinely full window is NOT a preemption — the smart paths would
    /// have fired too, so the valve is not the reason context was lost.
    #[test]
    fn large_prompt_does_not_count_as_preemption() {
        // 25 messages x 40k chars = 1M chars = ~250k tokens, over 70% of 200k.
        let obs = observe(&msgs(25, 40_000), 25, "sys", &[], 200_000, 20, false);
        assert!(obs.trimmed());
        assert!(obs.over_compactor_token_threshold);
        assert!(obs.over_overflow_threshold);
        assert!(!obs.preempted_smart_paths());
    }

    /// The hazard this module exists to surface: a compacted session whose
    /// summary sits at index 0 loses it to the very first drain.
    #[test]
    fn canonical_context_is_dropped_by_any_trim() {
        let obs = observe(&msgs(21, 100), 21, "sys", &[], 200_000, 20, true);
        assert_eq!(obs.trim_count, 1);
        assert!(obs.canonical_context_present);
        assert!(obs.canonical_context_dropped);
    }

    #[test]
    fn canonical_context_survives_when_valve_does_not_fire() {
        let obs = observe(&msgs(20, 100), 20, "sys", &[], 200_000, 20, true);
        assert_eq!(obs.trim_count, 0);
        assert!(obs.canonical_context_present);
        assert!(!obs.canonical_context_dropped);
    }

    /// A small-window model reaches 70% at a prompt a 200k model shrugs at.
    /// The observation must use the window it was handed, not the default.
    #[test]
    fn threshold_tracks_the_real_window() {
        let m = msgs(10, 4_000); // ~10k tokens
        let big = observe(&m, 10, "sys", &[], 200_000, 20, false);
        let small = observe(&m, 10, "sys", &[], 8_000, 20, false);
        assert!(!big.over_compactor_token_threshold);
        assert!(small.over_compactor_token_threshold);
        assert!(small.window_used_pct > big.window_used_pct);
    }

    #[test]
    fn zero_window_does_not_divide_by_zero() {
        let obs = observe(&msgs(5, 100), 5, "sys", &[], 0, 20, false);
        assert_eq!(obs.window_used_pct, 0);
    }

    /// The two vectors are recorded independently — the trim never writes
    /// back to the session, so the session can be far larger.
    #[test]
    fn session_count_is_recorded_separately() {
        let obs = observe(&msgs(21, 100), 340, "sys", &[], 200_000, 20, false);
        assert_eq!(obs.message_count, 21);
        assert_eq!(obs.session_messages, 340);
    }
}
