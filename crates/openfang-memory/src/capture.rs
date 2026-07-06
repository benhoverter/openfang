//! Episodic capture policy: decides whether an agent turn is persisted to
//! episodic memory or dropped.
//!
//! This is the memory subsystem's **write-side gate**. It consumes the typed
//! turn provenance ([`TurnTrigger`]) and the side-effect summary
//! ([`TurnEffects`]) produced upstream in the agent loop and renders a single
//! verdict ([`CaptureDecision`]).
//!
//! The policy lives here, in `openfang-memory`, because "what gets persisted to
//! memory" is a memory-subsystem concern (ANAI-80). Its inputs deliberately stay
//! in `openfang-types`: [`TurnTrigger`] is threaded through the kernel send
//! funnel and [`TurnEffects`]/`READ_ONLY_TOOLS` are shared with non-memory
//! consumers (e.g. `AgentMode::filter_tools`), so dragging them in here would
//! invert the dependency direction.

use openfang_types::turn::{TurnEffects, TurnTrigger};

/// Verdict of the capture predicate ([`should_capture_turn`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureDecision {
    /// Write the episodic row as normal.
    Keep,
    /// Skip the episodic write: a heartbeat turn that did nothing durable.
    Drop,
}

impl CaptureDecision {
    /// True for [`CaptureDecision::Drop`].
    pub fn is_drop(&self) -> bool {
        matches!(self, CaptureDecision::Drop)
    }
}

/// The conservative tick/heartbeat drop predicate (ANAI-76).
///
/// ```text
/// drop  <=>  (trigger == Heartbeat)  &&  (turn is inert)  &&  (observer is live)
/// ```
///
/// Pure, total, panic-free, allocation-free. Only [`TurnTrigger::Heartbeat`] is
/// eligible to drop -- `Cron`/`Proactive` carry scheduled/triggered intent and
/// are always kept, as is every human (`User`) or peer-agent (`AgentCall`) turn.
/// Inertness is judged purely by side-effect provenance ([`TurnEffects`]), never
/// by reply content. Biases hard toward keep: anything not provably a heartbeat
/// with zero durable side-effects is captured.
///
/// ANAI-77 (c) safety valve -- `observer_live`: the inert verdict is only
/// trustworthy when the driver actually feeds executed tools into the agent
/// loop (so [`TurnEffects::observe_tool`] fires). Subprocess CLI drivers
/// (`claude-code`, `qwen-code`) run tools inside their own subprocess and
/// return only final text, so the loop sees `tool_calls: []` and `is_inert()`
/// is *vacuously* true -- it cannot distinguish "did nothing" from "did work we
/// could not see". Gating the drop on a live observer makes the enforce flip
/// safe by construction: an observer-blind driver's inert heartbeats are always
/// kept, never dropped.
///
/// `observer_live` is supplied by the driver itself, via the
/// `CompletionResponse.observer_live` capability bit (ANAI-77x) — there is no
/// static per-provider allowlist. Native/HTTP drivers report `true` invariantly
/// (tool calls surface in-band); subprocess drivers report `true` only with a
/// live out-of-band observer this spawn (Claude Code + wired PreToolUse hook),
/// and `false` otherwise.
pub fn should_capture_turn(
    trigger: TurnTrigger,
    effects: &TurnEffects,
    observer_live: bool,
) -> CaptureDecision {
    if trigger == TurnTrigger::Heartbeat && effects.is_inert() && observer_live {
        CaptureDecision::Drop
    } else {
        CaptureDecision::Keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_only_inert_heartbeats() {
        let inert = TurnEffects::new();
        let mut active = TurnEffects::new();
        active.observe_tool("agent_send");

        // The one droppable case: heartbeat with no durable side-effect, on a
        // driver whose observer is live.
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &inert, true),
            CaptureDecision::Drop
        );
        // Heartbeat that did real work is kept.
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &active, true),
            CaptureDecision::Keep
        );
        // Every non-heartbeat trigger is kept even when inert.
        for trig in [
            TurnTrigger::User,
            TurnTrigger::AgentCall,
            TurnTrigger::Cron,
            TurnTrigger::Proactive,
        ] {
            assert_eq!(
                should_capture_turn(trig, &inert, true),
                CaptureDecision::Keep,
                "{trig} must never be dropped"
            );
        }
    }

    #[test]
    fn observer_blind_hard_keeps_inert_heartbeats() {
        // ANAI-77 (c): with a blind observer, an inert heartbeat is NOT
        // droppable -- inertness cannot be trusted, so the row is kept. This is
        // the safety property the enforce flip relies on.
        let inert = TurnEffects::new();
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &inert, false),
            CaptureDecision::Keep,
            "observer-blind inert heartbeat must be kept, never dropped"
        );
        // Same driver, an active heartbeat is kept too (unchanged).
        let mut active = TurnEffects::new();
        active.observe_tool("agent_send");
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &active, false),
            CaptureDecision::Keep
        );
    }

    #[test]
    fn is_drop_only_for_drop() {
        assert!(CaptureDecision::Drop.is_drop());
        assert!(!CaptureDecision::Keep.is_drop());
    }
}
