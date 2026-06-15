//! Provenance of an agent turn: what caused the agent loop to run.
//!
//! `TurnTrigger` is threaded from every message-entry path through the kernel
//! funnel (`send_message_with_handle_and_blocks`) into the agent loop, where it
//! is captured into episodic memory metadata under the `"trigger"` key.
//!
//! This is the typed discriminator that replaces brittle content-sniffing
//! (e.g. matching `[AUTONOMOUS TICK]` / `NO_REPLY` strings) for telling
//! human-driven turns apart from autonomous ones. See ANAI-84.

use serde::{Deserialize, Serialize};

/// What caused an agent turn to run.
///
/// Threaded as a required parameter through the kernel send funnel so that
/// every entry path is accounted for by the compiler — there is no defaulted
/// fail-open path that would silently mislabel an autonomous turn as `User`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnTrigger {
    /// A human user sent this message (channels, REST/WebSocket API, CLI, web UI).
    /// The thin `send_message*` wrappers pass this variant explicitly.
    User,
    /// A peer agent invoked this one (inter-agent tools / `agent_send`).
    AgentCall,
    /// Continuous-mode self-prompt (`[AUTONOMOUS TICK]`).
    Heartbeat,
    /// Periodic schedule or named cron job
    /// (`[SCHEDULED TICK]` / `CronAction::AgentTurn`).
    Cron,
    /// A registered proactive trigger/condition fired (`[PROACTIVE ALERT]`).
    Proactive,
}

impl TurnTrigger {
    /// Stable lowercase label used as the episodic metadata value under
    /// `metadata["trigger"]`. Kept in sync with the serde representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            TurnTrigger::User => "user",
            TurnTrigger::AgentCall => "agent_call",
            TurnTrigger::Heartbeat => "heartbeat",
            TurnTrigger::Cron => "cron",
            TurnTrigger::Proactive => "proactive",
        }
    }

    /// True when the turn was machine-initiated on a timer or event rather than
    /// driven by a human (or a peer agent acting on a human's behalf).
    ///
    /// `Heartbeat`, `Cron`, and `Proactive` are autonomous. `User` is not, and
    /// `AgentCall` is deliberately treated as non-autonomous: it carries real
    /// content delegated from a peer, which for prune/accounting purposes
    /// behaves like an external message, not a silent self-tick.
    pub fn is_autonomous(&self) -> bool {
        matches!(
            self,
            TurnTrigger::Heartbeat | TurnTrigger::Cron | TurnTrigger::Proactive
        )
    }
}

impl std::fmt::Display for TurnTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_matches_serde() {
        // as_str() must equal the snake_case serde tag so the metadata label
        // and any serialized form never drift.
        for t in [
            TurnTrigger::User,
            TurnTrigger::AgentCall,
            TurnTrigger::Heartbeat,
            TurnTrigger::Cron,
            TurnTrigger::Proactive,
        ] {
            let json = serde_json::to_string(&t).unwrap();
            assert_eq!(json, format!("\"{}\"", t.as_str()));
        }
    }

    #[test]
    fn autonomy_partition() {
        assert!(!TurnTrigger::User.is_autonomous());
        assert!(!TurnTrigger::AgentCall.is_autonomous());
        assert!(TurnTrigger::Heartbeat.is_autonomous());
        assert!(TurnTrigger::Cron.is_autonomous());
        assert!(TurnTrigger::Proactive.is_autonomous());
    }

    #[test]
    fn round_trips_through_serde() {
        let t = TurnTrigger::Cron;
        let json = serde_json::to_string(&t).unwrap();
        let back: TurnTrigger = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }
}
