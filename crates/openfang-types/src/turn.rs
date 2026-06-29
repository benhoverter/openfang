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

/// Tool names that perform **no durable side-effect** (read-only).
///
/// Single source of truth: `AgentMode::filter_tools` (Assist mode) references
/// this same list, so the two can never drift. The conservative bias of the
/// drop predicate (ANAI-76) keys off it: any tool **not** listed here is
/// treated as side-effecting, so an unrecognized tool keeps the turn.
pub const READ_ONLY_TOOLS: &[&str] = &[
    "file_read",
    "file_list",
    "memory_recall",
    "web_fetch",
    "web_search",
    "agent_list",
];

/// True if executing `tool_name` leaves no durable side-effect.
///
/// Conservative by construction: only the explicit `READ_ONLY_TOOLS` allowlist
/// returns `true`; everything else (including unknown tools) is side-effecting.
pub fn tool_is_read_only(tool_name: &str) -> bool {
    READ_ONLY_TOOLS.contains(&tool_name)
}

/// Side-effect summary of a single agent turn — the second input to the drop
/// predicate alongside [`TurnTrigger`].
///
/// Accumulated over the turn's tool executions in the agent loop (ANAI-77 calls
/// [`TurnEffects::observe_tool`] once per executed tool). "Inert" is judged from
/// this summary — by side-effect provenance, never by reply content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TurnEffects {
    /// Set once any state-mutating / outbound tool ran (anything outside
    /// [`READ_ONLY_TOOLS`]). Read-only tool calls never set this.
    ran_side_effecting_tool: bool,
    /// Count of all executed tools observed this turn (read-only and
    /// side-effecting alike). ANAI-77 field-confirmation notch: stamped onto
    /// heartbeat episodic rows so the shadow window can prove the
    /// tool-observation pipeline actually runs on the heartbeat path even when
    /// no heartbeat turn ever runs a side-effecting tool.
    tools_observed: u32,
    /// Count of executed tools that perform a durable side-effect (anything
    /// outside [`READ_ONLY_TOOLS`], including unknown tools). Read-only tool
    /// calls never increment this. Zero count means the turn is inert.
    side_effecting_tools: u32,
}

impl TurnEffects {
    /// A fresh, empty summary: no tools observed yet (inert until proven
    /// otherwise).
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one executed tool by name, updating the side-effect summary.
    /// Conservative: unknown tools count as side-effecting.
    pub fn observe_tool(&mut self, tool_name: &str) {
        self.tools_observed = self.tools_observed.saturating_add(1);
        if !tool_is_read_only(tool_name) {
            self.side_effecting_tools = self.side_effecting_tools.saturating_add(1);
        }
    }

    /// True when the turn produced **no durable side-effect** (no mutating /
    /// outbound tool ran). A turn that ran zero tools, or only read-only tools,
    /// is inert by this measure.
    pub fn is_inert(&self) -> bool {
        self.side_effecting_tools == 0
    }

    /// Count of side-effecting tools executed this turn. Content-free signal
    /// for the ANAI-77 shadow confirmation notch: on a `would_drop` heartbeat
    /// row this must be 0 -- the safety property the enforce-gate relies on.
    pub fn side_effecting_tool_count(&self) -> u32 {
        self.side_effecting_tools
    }

    /// Count of all tools observed this turn (read-only + side-effecting).
    /// Pairs with [`Self::side_effecting_tool_count`] in the ANAI-77 notch to
    /// show the observation pipeline ran even when the side-effect count is 0.
    pub fn tools_observed_count(&self) -> u32 {
        self.tools_observed
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

    #[test]
    fn read_only_classifier_matches_allowlist() {
        for t in READ_ONLY_TOOLS {
            assert!(tool_is_read_only(t), "{t} should be read-only");
        }
        // Side-effecting / outbound tools are not read-only.
        for t in [
            "memory_store",
            "agent_send",
            "channel_send",
            "file_write",
            "apply_patch",
            "shell_exec",
        ] {
            assert!(!tool_is_read_only(t), "{t} must not be read-only");
        }
        // Conservative bias: an unknown tool is treated as side-effecting.
        assert!(!tool_is_read_only("some_future_tool"));
    }

    #[test]
    fn effects_inert_until_side_effecting_tool() {
        let mut e = TurnEffects::new();
        assert!(e.is_inert(), "fresh summary is inert");
        e.observe_tool("file_read");
        e.observe_tool("memory_recall");
        assert!(e.is_inert(), "read-only tools keep the turn inert");
        e.observe_tool("memory_store");
        assert!(
            !e.is_inert(),
            "a side-effecting tool makes the turn non-inert"
        );
    }

    #[test]
    fn unknown_tool_makes_turn_non_inert() {
        let mut e = TurnEffects::new();
        e.observe_tool("some_future_tool");
        assert!(
            !e.is_inert(),
            "unknown tools are conservatively side-effecting"
        );
    }

    #[test]
    fn effects_count_tools_for_notch() {
        // ANAI-77 notch: counts back the would_drop verdict with content-free
        // numbers. Read-only tools advance tools_observed but not the
        // side-effecting count; the latter is the safety signal.
        let mut e = TurnEffects::new();
        assert_eq!(e.tools_observed_count(), 0);
        assert_eq!(e.side_effecting_tool_count(), 0);

        e.observe_tool("file_read");
        e.observe_tool("memory_recall");
        assert_eq!(e.tools_observed_count(), 2);
        assert_eq!(
            e.side_effecting_tool_count(),
            0,
            "read-only tools are observed but not side-effecting"
        );
        assert!(e.is_inert());

        e.observe_tool("agent_send");
        e.observe_tool("memory_store");
        assert_eq!(e.tools_observed_count(), 4);
        assert_eq!(e.side_effecting_tool_count(), 2);
        assert!(!e.is_inert());
    }
}
