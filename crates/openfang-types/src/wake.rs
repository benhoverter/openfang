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
    /// True when this wake is a **terminal reply** produced by `agent_reply_async`
    /// (ANAI-122), i.e. leg 3 of the four-step round-trip (fleet -> origin).
    ///
    /// A reply is the *completion* of a correlation, not an origination: the
    /// wake-consumer mints **no** reply-right (kernel `reply_rights` registry,
    /// ANAI-122) for a turn woken by a reply, so origin's leg-4 turn cannot
    /// reply-bounce back into the tree —
    /// it can only surface (leg 4, `channel_send`) or do fresh work. That one
    /// bit is the structural guarantee that replaces the cycle guard on the
    /// terminal edge (the reply targets an ancestor, which `would_cycle` would
    /// otherwise refuse — see ANAI-122 spike). Omitted on the wire when false,
    /// so every pre-ANAI-122 wake decodes back to a non-reply unchanged.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_reply: bool,
    /// Optional surfacing route for the terminal reply (ANAI-123/124).
    ///
    /// When an origin dispatches `agent_send_async` with `surface_to`, this
    /// value rides the WHOLE round-trip: on the outbound leg (1->2) it is baked
    /// into the callee's one-shot reply-right token; on the reply leg (3->4)
    /// `agent_reply_async` copies it back onto the reply envelope; and on
    /// origin's leg-4 woken turn the wake-consumer emits exactly one
    /// `channel_send(surface_to, reply_text)` so the delegated answer reaches a
    /// human channel instead of dying in origin's silent woken turn (the
    /// 2026-07-04 live-test failure this closes).
    ///
    /// Encoded as `"<channel>:<recipient>"` (e.g.
    /// `"discord:1086446153098342510"`) — the same (adapter, recipient) pair
    /// the `channel_send` tool takes. `None` means "no surfacing"; the reply
    /// still reaches origin as a woken turn, it just is not auto-posted.
    /// Omitted on the wire when absent, so a pre-ANAI-123 wake decodes back to
    /// no-surfacing unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_to: Option<String>,
    /// ANAI-199: *how* this reply came to exist. Meaningful only when
    /// [`Self::is_reply`] is set; ignored (and omitted on the wire) otherwise.
    ///
    /// The initiator's leg-4 turn must be able to tell a real answer from a
    /// kernel-synthesized stand-in, because the two demand different behaviour:
    /// an [`ReplyKind::Explicit`] reply is the callee's considered answer, while
    /// a synthetic one reports that the correlation was closed WITHOUT one. An
    /// orchestrator that cannot distinguish them treats "the target never ran"
    /// as a contract-satisfying result — precisely the silent-failure class this
    /// stack exists to close.
    ///
    /// Defaults to [`ReplyKind::Explicit`] and is skipped on the wire in that
    /// case, so every pre-ANAI-199 payload decodes back unchanged.
    #[serde(default, skip_serializing_if = "ReplyKind::is_explicit")]
    pub reply_kind: ReplyKind,
}

/// Provenance of a terminal reply (ANAI-199): who produced it, and therefore
/// how much it can be trusted to mean.
///
/// Only [`ReplyKind::Explicit`] is produced by an agent. Every other variant is
/// minted by the daemon to discharge an outstanding reply debt the callee did
/// not (or could not) pay itself, so that `agent_send_async` has a reply
/// guarantee rather than a reply *hope*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplyKind {
    /// The callee called `agent_reply_async` itself. The body is its answer.
    #[default]
    Explicit,
    /// The callee's turn ended without ever calling `agent_reply_async`; the
    /// daemon closed the correlation using the turn's final text (ANAI-198).
    AutoClose,
    /// The wake never produced an answer — undeliverable, refused before
    /// dispatch, or the agent loop errored mid-turn (ANAI-199). The body says
    /// which, and whether side effects may exist.
    Error,
    /// The sender's deadline elapsed and the callee's turn was aborted
    /// (ANAI-201). Side effects up to the abort point may exist.
    Timeout,
}

impl ReplyKind {
    /// serde `skip_serializing_if` predicate — see [`WakeEnvelope::reply_kind`].
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit)
    }

    /// True for every daemon-minted kind, i.e. "no agent authored this body".
    /// The bit an initiator should branch on before treating a reply as work
    /// product.
    pub fn is_synthetic(&self) -> bool {
        !self.is_explicit()
    }

    /// Stable lowercase label for logs, audit lines, and the prompt-visible
    /// header on a synthesized reply.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::AutoClose => "auto_close",
            Self::Error => "error",
            Self::Timeout => "timeout",
        }
    }
}

/// serde `skip_serializing_if` predicate: skip a `bool` field when it is false,
/// so the flag is absent on the wire unless explicitly set (keeps the common
/// non-reply wake payload byte-identical to the pre-ANAI-122 shape).
fn is_false(b: &bool) -> bool {
    !*b
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
            vec!["a", "b", "c", "d"]
                .into_iter()
                .map(String::from)
                .collect(),
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
            is_reply: false,
            surface_to: None,
            reply_kind: ReplyKind::default(),
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
            is_reply: false,
            surface_to: None,
            reply_kind: ReplyKind::default(),
        };
        let json = serde_json::to_string(&env).unwrap();
        // origin is skipped on the wire when absent...
        assert!(
            !json.contains("origin"),
            "None origin must be omitted: {json}"
        );
        // is_reply is likewise skipped when false — the default wake shape.
        assert!(
            !json.contains("is_reply"),
            "false is_reply must be omitted: {json}"
        );
        // ...and a payload with no origin field decodes back to None.
        let back = WakeEnvelope::from_payload(json.as_bytes()).unwrap();
        assert_eq!(back.origin, None);
        assert_eq!(back.trigger, TurnTrigger::Cron);
        assert!(!back.is_reply, "absent is_reply must decode to false");
        // surface_to is likewise absent on the wire when None.
        assert!(
            !json.contains("surface_to"),
            "None surface_to must be omitted: {json}"
        );
        assert_eq!(
            back.surface_to, None,
            "absent surface_to must decode to None"
        );
        // ANAI-199: the default reply-kind is likewise absent on the wire, so a
        // pre-ANAI-199 payload is byte-identical to a post-ANAI-199 one.
        assert!(
            !json.contains("reply_kind"),
            "explicit reply_kind must be omitted: {json}"
        );
        assert_eq!(
            back.reply_kind,
            ReplyKind::Explicit,
            "absent reply_kind must decode to Explicit"
        );
    }

    #[test]
    fn envelope_reply_flag_round_trips_and_is_present_when_set() {
        // ANAI-122: a terminal reply wake (leg 3) carries is_reply = true and a
        // fresh single-element lineage rooted at the replier, so the consumer
        // grants no further reply-right and origin's leg-4 turn is a leaf.
        let env = WakeEnvelope {
            target: "orchestrator".into(),
            sender: "worker-b".into(),
            message: "here is the result you asked for".into(),
            lineage: WakeLineage::root_at("worker-b"),
            trigger: TurnTrigger::AgentCall,
            origin: None,
            is_reply: true,
            surface_to: Some("discord:1086446153098342510".into()),
            reply_kind: ReplyKind::default(),
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.contains("is_reply"),
            "true is_reply must serialize: {json}"
        );
        let back = WakeEnvelope::from_payload(json.as_bytes()).unwrap();
        assert_eq!(env, back);
        assert!(back.is_reply);
        // ANAI-123: the surfacing route round-trips on the terminal reply leg,
        // so origin's leg-4 turn knows where to post the delegated answer.
        assert_eq!(
            back.surface_to.as_deref(),
            Some("discord:1086446153098342510")
        );
        // The reply roots a fresh chain at the replier: depth 1, so origin's
        // leg-4 turn starts clean and cannot be over-deep.
        assert_eq!(back.lineage.depth(), 1);
        assert_eq!(back.lineage.root(), Some("worker-b"));
    }

    /// ANAI-199: a daemon-synthesized reply must be distinguishable on the wire.
    /// If `reply_kind` did not round-trip, an initiator would read "the target
    /// was never reachable" as the target's considered answer.
    #[test]
    fn synthetic_reply_kind_round_trips_and_is_distinguishable() {
        for kind in [ReplyKind::AutoClose, ReplyKind::Error, ReplyKind::Timeout] {
            let env = WakeEnvelope {
                target: "orchestrator".into(),
                sender: "worker-b".into(),
                message: "the kernel is answering on the callee's behalf".into(),
                lineage: WakeLineage::root_at("worker-b"),
                trigger: TurnTrigger::AgentCall,
                origin: None,
                is_reply: true,
                surface_to: None,
                reply_kind: kind,
            };
            let json = serde_json::to_string(&env).unwrap();
            assert!(
                json.contains(kind.label()),
                "{kind:?} must serialize as its snake_case label: {json}"
            );
            let back = WakeEnvelope::from_payload(json.as_bytes()).unwrap();
            assert_eq!(env, back);
            assert!(
                back.reply_kind.is_synthetic(),
                "{kind:?} must report as synthetic so an initiator does not \
                 mistake it for the callee's own answer"
            );
        }
        assert!(!ReplyKind::Explicit.is_synthetic());
    }
}
