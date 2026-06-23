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
/// drop  ⇔  (trigger == Heartbeat)  ∧  (turn is inert)
/// ```
///
/// Pure, total, panic-free, allocation-free. Only [`TurnTrigger::Heartbeat`] is
/// eligible to drop — `Cron`/`Proactive` carry scheduled/triggered intent and
/// are always kept, as is every human (`User`) or peer-agent (`AgentCall`) turn.
/// Inertness is judged purely by side-effect provenance ([`TurnEffects`]), never
/// by reply content. Biases hard toward keep: anything not provably a heartbeat
/// with zero durable side-effects is captured.
pub fn should_capture_turn(trigger: TurnTrigger, effects: &TurnEffects) -> CaptureDecision {
    if trigger == TurnTrigger::Heartbeat && effects.is_inert() {
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

        // The one droppable case: heartbeat with no durable side-effect.
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &inert),
            CaptureDecision::Drop
        );
        // Heartbeat that did real work is kept.
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &active),
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
                should_capture_turn(trig, &inert),
                CaptureDecision::Keep,
                "{trig} must never be dropped"
            );
        }
    }

    #[test]
    fn is_drop_only_for_drop() {
        assert!(CaptureDecision::Drop.is_drop());
        assert!(!CaptureDecision::Keep.is_drop());
    }
}
