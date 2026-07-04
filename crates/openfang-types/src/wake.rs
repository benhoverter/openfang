//! Cross-agent wake provenance: the call-chain contract carried in a wake
//! task's payload.
//!
//! Nothing in the runtime tracks cross-agent call lineage today. The intra-loop
//! `LoopGuard` (openfang-runtime/src/loop_guard.rs) catches same-tool spam and
//! ping-pong *within a single loop*, but it is rebuilt fresh per loop and is
//! therefore blind to cross-agent `A -> B -> A -> B` wake cycles (ANAI-100
//! spike).
//!
//! [`WakeLineage`] is the payload-carried contract that closes that gap. It is
//! the single source of truth for three security requirements on the
//! `agent_send_async` wake substrate:
//!
//! * **Cycle detection (req 4):** a wake whose target already appears in the
//!   chain is a cycle and must be refused — see [`WakeLineage::would_cycle`].
//! * **Depth bound (req 9):** depth is `agents.len()` — **one** counter, derived
//!   from the chain, never a second free-running integer that can disagree.
//!   The lineage-stamping entrypoint seeds depth from here so a raw wake cannot
//!   reset the guard every hop. See [`WakeLineage::depth`] /
//!   [`WakeLineage::exceeds_depth`].
//! * **Per-tree budget (req 10):** the **root** (`agents[0]`) pays for the whole
//!   subtree. A per-caller cap keyed on `created_by` is reset every hop
//!   (`created_by` is fresh each wake), so N-per-hop silently chains to N^k
//!   "legal" wakes. Keying the budget on [`WakeLineage::root`] defeats that.
//!
//! ## Enforcement status — READ THIS
//!
//! All three properties are now enforced cross-hop:
//!
//! * **req 4 (cycle) & req 9 (depth):** enforced across agents since ANAI-110.
//!   `run_woken_agent_loop` scopes the inbound lineage into a task-local that
//!   `tool_agent_send_async` reads (`resolve_wake_base_lineage`), so a wake
//!   extends the REAL `root -> ... -> this` chain instead of re-rooting at the
//!   sender each hop. A multi-hop ring `A -> B -> A` is caught and depth accrues
//!   across agents (5th hop trips [`DEFAULT_MAX_WAKE_DEPTH`]).
//! * **req 10 (per-tree budget):** implemented since ANAI-111. The producer
//!   charges each wake to its [`WakeLineage::root`] via a per-root sliding-window
//!   budget (`wake_tree_admit` in openfang-runtime/src/tool_runner.rs), which
//!   catches fan-out amplification that cycle/depth cannot. A coarse
//!   process-global ceiling (`wake_emit_admit`) is retained on top as a
//!   fleet-wide aggregate backstop. Both are tunable via the `[agent_wake]`
//!   config section (see [`crate::agent_wake`]).
//!
//! Origin turns (channel / cron / API — no inbound lineage) root the chain at
//! the sender, so only self-wake is a cycle for them; that is correct, not a
//! gap — such a turn has no ancestry to inherit.
//!
//! The chain is ordered **root -> ... -> current**: `agents[0]` began the chain
//! and the last element is the agent that emitted the current wake.

use serde::{Deserialize, Serialize};

use crate::turn::TurnTrigger;

/// Default maximum wake-chain depth before a wake is refused (ANAI security
/// default). Enforced against [`WakeLineage::depth`] at the wake entrypoint.
pub const DEFAULT_MAX_WAKE_DEPTH: usize = 5;

/// Reserved task-queue title/`task_type` prefix that marks a queued task as an
/// `agent_send_async` wake. Single source of truth shared by three sites:
///
/// * the **producer** (`agent_send_async` tool) stamps the task title with it,
/// * the **wake-consumer** claims only tasks bearing it (`task_claim_wake`), and
/// * ordinary `task_claim` **excludes** it, so a wake never gets pulled as a
///   regular collaboration task (and a regular task never runs as a wake).
///
/// Keeping the literal here means a future rename can't silently desync the
/// producer from the consumer.
pub const WAKE_TASK_PREFIX: &str = "wake:";

/// The ordered cross-agent call chain for a wake, carried in the task payload.
///
/// Stored root-first: `agents[0]` is the lineage root, the last element is the
/// most recent caller. An empty chain denotes "no wake ancestry yet" (the
/// origin turn that will *begin* a chain when it dispatches).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WakeLineage {
    /// Stable agent-id strings, ordered root -> ... -> current caller.
    agents: Vec<String>,
}

impl WakeLineage {
    /// An empty lineage (no wake ancestry). The origin turn dispatching its
    /// first wake starts from here.
    pub fn empty() -> Self {
        Self { agents: Vec::new() }
    }

    /// Begin a chain rooted at `root` (the agent that started the wake tree).
    pub fn root_at(root: impl Into<String>) -> Self {
        Self {
            agents: vec![root.into()],
        }
    }

    /// Build directly from an ordered (root-first) agent-id list.
    pub fn from_agents(agents: Vec<String>) -> Self {
        Self { agents }
    }

    /// The lineage root — the agent that began the chain. Per-tree budget keys
    /// on this (req 10). `None` only for an empty lineage.
    pub fn root(&self) -> Option<&str> {
        self.agents.first().map(String::as_str)
    }

    /// The most recent caller (the agent that emitted the current wake).
    pub fn current(&self) -> Option<&str> {
        self.agents.last().map(String::as_str)
    }

    /// Chain depth — the single source of truth for the depth bound (req 9).
    pub fn depth(&self) -> usize {
        self.agents.len()
    }

    /// True when the chain is empty (no wake ancestry).
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// The ordered (root-first) chain.
    pub fn as_slice(&self) -> &[String] {
        &self.agents
    }

    /// True when `agent` already appears anywhere in the chain. Used both for
    /// cycle detection and to answer "is this agent an ancestor of itself?".
    pub fn contains(&self, agent: &str) -> bool {
        self.agents.iter().any(|a| a == agent)
    }

    /// True when extending the chain to `next` would form a cycle
    /// (`A -> B -> A`). The wake entrypoint must refuse such a wake (req 4).
    pub fn would_cycle(&self, next: &str) -> bool {
        self.contains(next)
    }

    /// True when the chain depth is at or beyond `max` — i.e. a further wake
    /// would exceed the bound. Compared against [`Self::depth`] (req 9).
    pub fn exceeds_depth(&self, max: usize) -> bool {
        self.agents.len() >= max
    }

    /// Return a new lineage extended by `next`, leaving `self` untouched.
    ///
    /// Callers that must not introduce a cycle should gate on
    /// [`Self::would_cycle`] first; this method does **not** itself refuse a
    /// repeat — it is the pure mechanism, the entrypoint owns the policy.
    pub fn extended(&self, next: impl Into<String>) -> Self {
        let mut agents = self.agents.clone();
        agents.push(next.into());
        Self { agents }
    }
}

/// The full payload carried in an `agent_send_async` wake task.
///
/// This is the structured wrapper that serializes into the task-queue payload
/// BLOB (the opaque `payload: &[u8]` carrier at the `task_post` surface). The
/// wake-consumer decodes it with [`WakeEnvelope::from_payload`], reconstructs
/// the [`TurnTrigger`], and re-enters the kernel send funnel.
///
/// The `trigger` is carried as a typed [`TurnTrigger`] rather than a bare
/// string: serde round-trips the enum in its `snake_case` form as part of the
/// one JSON blob, so the consumer recovers the typed provenance directly — no
/// separate label-to-variant parse, and therefore no second source of truth to
/// drift from the enum definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WakeEnvelope {
    /// Agent-id of the wake target (the agent whose loop will run).
    pub target: String,
    /// Agent-id of the sender dispatching this wake.
    pub sender: String,
    /// The message content delivered to `target`.
    pub message: String,
    /// Cross-agent call lineage — the single source of truth for cycle
    /// detection (req 4), depth bound (req 9), and per-tree budget (req 10).
    /// Threaded cross-hop since ANAI-110, so all three enforce against real
    /// ancestry (see the module-level "Enforcement status" note); the per-tree
    /// budget keys on [`WakeLineage::root`] (ANAI-111).
    pub lineage: WakeLineage,
    /// Provenance to stamp on the woken turn. For timer/reconciliation wakes
    /// this is [`TurnTrigger::Cron`]; for genuine delegated peer content it is
    /// [`TurnTrigger::AgentCall`]. Never [`TurnTrigger::Heartbeat`] — that would
    /// make the woken turn eligible for the capture-drop predicate.
    pub trigger: TurnTrigger,
    /// Optional originating route (e.g. channel id) for replies/approval
    /// prompts raised by the woken agent. `None` means a prompt raised mid-wake
    /// has nowhere to route back — a documented latent gap, not an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

impl WakeEnvelope {
    /// Serialize to the opaque payload BLOB stored at the `task_post` surface.
    pub fn to_payload(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Decode an envelope from a task payload BLOB.
    pub fn from_payload(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_no_root_or_depth() {
        let l = WakeLineage::empty();
        assert!(l.is_empty());
        assert_eq!(l.depth(), 0);
        assert_eq!(l.root(), None);
        assert_eq!(l.current(), None);
    }

    #[test]
    fn root_at_seeds_single_element_chain() {
        let l = WakeLineage::root_at("orchestrator");
        assert_eq!(l.depth(), 1);
        assert_eq!(l.root(), Some("orchestrator"));
        assert_eq!(l.current(), Some("orchestrator"));
    }

    #[test]
    fn extended_is_immutable_and_root_is_stable() {
        let root = WakeLineage::root_at("orchestrator");
        let hop1 = root.extended("worker-a");
        let hop2 = hop1.extended("worker-b");

        // original untouched
        assert_eq!(root.depth(), 1);
        // root stays the budget owner across every hop (req 10)
        assert_eq!(hop2.root(), Some("orchestrator"));
        assert_eq!(hop2.current(), Some("worker-b"));
        assert_eq!(hop2.depth(), 3);
        assert_eq!(hop2.as_slice(), &["orchestrator", "worker-a", "worker-b"]);
    }

    #[test]
    fn cycle_detection_catches_repeat_target() {
        let chain = WakeLineage::root_at("a").extended("b");
        // A -> B -> A is a cycle
        assert!(chain.would_cycle("a"));
        assert!(chain.would_cycle("b"));
        // A -> B -> C is fine
        assert!(!chain.would_cycle("c"));
    }

    #[test]
    fn depth_bound_is_inclusive_at_max() {
        let chain = WakeLineage::from_agents(
            vec!["a", "b", "c", "d", "e"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        assert_eq!(chain.depth(), DEFAULT_MAX_WAKE_DEPTH);
        // at the bound, a further wake must be refused
        assert!(chain.exceeds_depth(DEFAULT_MAX_WAKE_DEPTH));
        // one shorter is still under the bound
        let shorter = WakeLineage::from_agents(
            vec!["a", "b", "c", "d"].into_iter().map(String::from).collect(),
        );
        assert!(!shorter.exceeds_depth(DEFAULT_MAX_WAKE_DEPTH));
    }

    #[test]
    fn round_trips_through_json() {
        let chain = WakeLineage::root_at("a").extended("b").extended("c");
        let json = serde_json::to_string(&chain).unwrap();
        let back: WakeLineage = serde_json::from_str(&json).unwrap();
        assert_eq!(chain, back);
    }

    #[test]
    fn envelope_round_trips_through_payload() {
        let env = WakeEnvelope {
            target: "worker-b".into(),
            sender: "orchestrator".into(),
            message: "do the thing — with an em dash \u{2014} inside".into(),
            lineage: WakeLineage::root_at("orchestrator").extended("worker-a"),
            trigger: TurnTrigger::AgentCall,
            origin: Some("channel:1086446153098342510".into()),
        };
        let payload = env.to_payload().unwrap();
        let back = WakeEnvelope::from_payload(&payload).unwrap();
        assert_eq!(env, back);
        // typed trigger survives the round trip — no label-to-variant parse.
        assert_eq!(back.trigger, TurnTrigger::AgentCall);
        assert_eq!(back.lineage.root(), Some("orchestrator"));
        assert_eq!(back.lineage.current(), Some("worker-a"));
    }

    #[test]
    fn envelope_origin_is_optional_and_omitted_when_none() {
        let env = WakeEnvelope {
            target: "worker-b".into(),
            sender: "orchestrator".into(),
            message: "tick".into(),
            lineage: WakeLineage::empty(),
            trigger: TurnTrigger::Cron,
            origin: None,
        };
        let json = serde_json::to_string(&env).unwrap();
        // origin is skipped on the wire when absent...
        assert!(!json.contains("origin"), "None origin must be omitted: {json}");
        // ...and a payload with no origin field decodes back to None.
        let back = WakeEnvelope::from_payload(json.as_bytes()).unwrap();
        assert_eq!(back.origin, None);
        assert_eq!(back.trigger, TurnTrigger::Cron);
    }
}
