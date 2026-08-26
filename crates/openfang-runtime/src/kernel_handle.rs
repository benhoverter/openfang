//! Trait abstraction for kernel operations needed by the agent runtime.
//!
//! This trait allows `openfang-runtime` to call back into the kernel for
//! inter-agent operations (spawn, send, list, kill) without creating
//! a circular dependency. The kernel implements this trait and passes
//! it into the agent loop.

use async_trait::async_trait;
use std::sync::Arc;

use crate::bridge_auth::TokenIssuer;

/// ANAI-165: key prefix that routes a memory operation to the deliberate
/// cross-agent namespace instead of the caller's own.
///
/// A prefix rather than a `scope` parameter on purpose. The scope has to
/// survive every transport the memory tools already cross — the WASM host
/// shim, the MCP bridge's tool declarations, the bridge-IPC arg bundle, and
/// the Claude Code driver's tool schema — and a new parameter would have to be
/// threaded (and could be dropped) at each one. A prefix travels inside the
/// key that all of them already carry, and lands in the audit log verbatim.
pub const SHARED_KEY_PREFIX: &str = "shared:";

/// Agent info returned by list and discovery operations.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub state: String,
    pub model_provider: String,
    pub model_name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub tools: Vec<String>,
}

/// One tier-3 fact write, as it crosses the handle boundary (ANAI-204).
///
/// A struct rather than six positional parameters: the write already carries
/// two optional strings and an optional float, and a call site that has to
/// count `None`s to line them up is a defect waiting for the next field. The
/// fields are plain `String`s so the trait keeps its standing rule of not
/// dragging `openfang-memory`'s types across the boundary — the vocabulary
/// types (`FactScope`, `ClaimKey`, `FactStatus`) are parsed on the far side,
/// where the store that enforces them lives.
#[derive(Debug, Clone)]
pub struct FactWriteRequest {
    /// `agent` / `project` / `user`. Validated kernel-side.
    pub scope: String,
    /// What the claim is about. `None` is legal only for `agent` scope, which
    /// derives it from the caller.
    pub scope_ref: Option<String>,
    /// The slot name, e.g. `repo.trunk_model`.
    pub claim_key: String,
    /// The claim itself, in prose.
    pub claim: String,
    /// `open` or `settled`. `None` means settled.
    pub status: Option<String>,
    /// 0.0..=1.0. `None` means 1.0.
    pub confidence: Option<f64>,
}

/// Handle to kernel operations, passed into the agent loop so agents
/// can interact with each other via tools.
#[allow(clippy::too_many_arguments)]
#[async_trait]
pub trait KernelHandle: Send + Sync {
    /// Spawn a new agent from a TOML manifest string.
    /// `parent_id` is the UUID string of the spawning agent (for lineage tracking).
    /// Returns (agent_id, agent_name) on success.
    async fn spawn_agent(
        &self,
        manifest_toml: &str,
        parent_id: Option<&str>,
    ) -> Result<(String, String), String>;

    /// Send a message to another agent and get the response.
    async fn send_to_agent(&self, agent_id: &str, message: &str) -> Result<String, String>;

    /// ANAI-147: sender-attributed counterpart to [`Self::send_to_agent`].
    ///
    /// `sender_agent_id` is the CALLING agent's UUID, threaded into the send
    /// funnel's sender slot so the target's turn carries kernel-attested
    /// agent-to-agent attribution instead of arriving unattributed (and being
    /// pinned on whichever human last spoke in the target's session). Callers
    /// hand-writing a `[From: X]` prefix into the message body were papering
    /// over exactly this gap — and body text is the one thing an agent is told
    /// not to trust for identity.
    ///
    /// Defaults to the unattributed call so non-kernel implementors (test
    /// doubles, WASM host shims) need no change.
    async fn send_to_agent_from(
        &self,
        agent_id: &str,
        message: &str,
        sender_agent_id: Option<&str>,
    ) -> Result<String, String> {
        let _ = sender_agent_id;
        self.send_to_agent(agent_id, message).await
    }

    /// List all running agents.
    fn list_agents(&self) -> Vec<AgentInfo>;

    /// Kill an agent by ID.
    fn kill_agent(&self, agent_id: &str) -> Result<(), String>;

    /// Activate (wake up) an inactive agent by ID, flipping its state to Running.
    /// Used by orchestrator agents to dispatch work to currently inactive agents
    /// (Suspended, Crashed, or never-started). Terminated agents cannot be revived.
    /// Returns the agent's name on success.
    fn activate_agent(&self, agent_id: &str) -> Result<String, String> {
        let _ = agent_id;
        Err("Agent activation not available".to_string())
    }

    /// ANAI-165: store a value in the CALLER's own memory namespace.
    ///
    /// `caller_agent_id` is the calling agent's UUID or registered name. It is
    /// required, not optional-in-practice: before ANAI-165 every agent in the
    /// fleet wrote into one hardcoded namespace (`shared_memory_agent_id()`),
    /// which made provenance unanswerable at the source — 879 rows whose
    /// authors are unrecoverable. A `None` caller must therefore FAIL, never
    /// fall back to the shared bucket: a silent fallback would recreate the
    /// exact bug this fixes, one unattributed call at a time.
    ///
    /// Deliberate cross-agent state stays available through a `shared:` key
    /// prefix, resolved by the implementor. The prefix rides into the audit
    /// log verbatim, so "who wrote to shared" is answerable from the key
    /// alone — sharing is an explicit, visible act rather than the default.
    fn memory_store(
        &self,
        caller_agent_id: Option<&str>,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), String>;

    /// ANAI-165: recall a value from the CALLER's own memory namespace.
    ///
    /// Same contract as [`Self::memory_store`], including the `shared:` prefix
    /// and the fail-closed treatment of a `None` caller. Reads are scoped for
    /// the same reason writes are: an agent that can read every other agent's
    /// keys by guessing them has no namespace at all.
    fn memory_recall(
        &self,
        caller_agent_id: Option<&str>,
        key: &str,
    ) -> Result<Option<serde_json::Value>, String>;

    /// ANAI-194: close the CALLER's open episode (ADR 0001 §2.2, ADR 0002 §2.2).
    ///
    /// `reason` is the wire spelling of a `CloseReason`. Returns the closed
    /// episode's id, or `None` when the caller had nothing open — a second
    /// "wrap this up" is a no-op, not a failure, because the agent cannot see
    /// the episode table and should not be punished for asking twice.
    ///
    /// Scoped to the caller with no `shared:` escape. Episodes are strictly
    /// per-agent: there is no coherent meaning to closing someone else's, and
    /// a cross-agent close would silently truncate another agent's
    /// consolidation input.
    fn memory_episode_close(
        &self,
        caller_agent_id: Option<&str>,
        reason: &str,
        title: Option<&str>,
        summary: Option<&str>,
    ) -> Result<Option<String>, String>;

    /// ANAI-246: ask for the CALLER's conversation context to be reset at the
    /// end of the current turn — fresh session, canonical re-anchored, the
    /// compacted summary kept.
    ///
    /// Deferred by design. The tool that calls this runs mid-turn, while the
    /// agent loop holds a live `Session` it re-persists several more times
    /// before the turn ends; an immediate clear would be silently overwritten.
    /// So this records intent and the kernel applies it once the loop returns.
    ///
    /// Scoped to the caller with no `shared:` escape, for the same reason
    /// `memory_episode_close` is: resetting someone else's context is
    /// amnesia inflicted on an agent that did not ask for it.
    fn request_context_reset(&self, caller_agent_id: Option<&str>) -> Result<(), String>;

    /// ANAI-194: the CALLER's memory status — open episode, turns captured
    /// into it, and the idle countdown.
    ///
    /// Returned as JSON rather than a struct so the trait does not drag the
    /// `openfang-memory` episode types across the handle boundary, which every
    /// other method on here deliberately avoids.
    fn memory_status(&self, caller_agent_id: Option<&str>) -> Result<serde_json::Value, String>;

    /// ANAI-204 (ADR 0001 §2.3): write a claim into its tier-3 slot,
    /// superseding whatever was there.
    ///
    /// Returns the outcome as JSON — `created` / `affirmed` / `superseded`,
    /// the resolved slot address, and on a supersession the claim that was
    /// displaced. The caller usually wants that distinction: "I already
    /// believed this" and "I have changed my mind" are different things to
    /// report, and the tool layer cannot tell them apart from the outside
    /// without re-reading the row it just wrote.
    ///
    /// Async because the write opens a transaction and, where an embedding
    /// driver is configured, embeds the claim first.
    ///
    /// Defaulted to an error for the same reason [`Self::memory_search`] is:
    /// a non-kernel implementor that silently accepted the write and dropped
    /// it would be indistinguishable from a slot that never took.
    async fn memory_fact_write(
        &self,
        _caller_agent_id: Option<&str>,
        _request: FactWriteRequest,
    ) -> Result<serde_json::Value, String> {
        Err("memory_fact is not available on this kernel handle".to_string())
    }

    /// ANAI-204: the live claim in a slot, if any.
    ///
    /// Addressed by subject (`scope`, `scope_ref`, `claim_key`), never by
    /// author — see the module docs on `openfang_memory::fact`. `scope_ref`
    /// is `None` only for `agent` scope, which derives it from the caller.
    fn memory_fact_get(
        &self,
        _caller_agent_id: Option<&str>,
        _scope: &str,
        _scope_ref: Option<&str>,
        _claim_key: &str,
    ) -> Result<serde_json::Value, String> {
        Err("memory_fact is not available on this kernel handle".to_string())
    }

    /// ANAI-204: every claim that has occupied a slot, newest first.
    ///
    /// The audit path. Never part of automatic recall (§2.3.2) — a superseded
    /// claim reaching the prompt is the failure the whole tier is built to
    /// make unrepresentable, so reaching history takes an explicit call.
    fn memory_fact_history(
        &self,
        _caller_agent_id: Option<&str>,
        _scope: &str,
        _scope_ref: Option<&str>,
        _claim_key: &str,
        _limit: usize,
    ) -> Result<serde_json::Value, String> {
        Err("memory_history is not available on this kernel handle".to_string())
    }

    /// ANAI-166 (ADR 0002 §2.2/§2.3): retrieval over the CALLER's episodic
    /// memory — semantic when an embedding driver is configured, LIKE matching
    /// when it is not.
    ///
    /// Named `memory_search` rather than `memory_recall` because the existing
    /// `memory_recall` on this trait is the exact-key KV read and keeps that
    /// contract; the *tool* named `memory_recall` fans out to both. Note that
    /// `tool_compat` already maps a `memory_search` tool name onto
    /// `memory_recall`, so this name must never be advertised as a tool.
    ///
    /// Async because embedding the query is a network call on most providers.
    /// Returns JSON for the same reason [`Self::memory_status`] does: the trait
    /// must not drag `openfang-memory`'s fragment types across the boundary.
    ///
    /// Defaulted so non-kernel implementors (test doubles, the WASM host shim,
    /// whose `host_functions.rs` recall stays key-only by design) need no
    /// change. A default that errors is correct here: silently returning zero
    /// hits would be indistinguishable from "you have no such memory", which is
    /// exactly the failure mode §1.1 of the ADR is about.
    async fn memory_search(
        &self,
        caller_agent_id: Option<&str>,
        query: &str,
        scope: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<serde_json::Value, String> {
        let _ = (caller_agent_id, query, scope, kind, limit);
        Err("Memory search is not available on this kernel handle".to_string())
    }

    /// ANAI-166 (ADR 0002 §2.2): append an unstructured note to the CALLER's
    /// memory, attributed to the caller's currently open episode.
    ///
    /// The cheap write. It takes no key and no controlled vocabulary on
    /// purpose — a note the agent had to classify before writing is a note it
    /// will not write. Keyed, expensive writes are `memory_fact` (stage 3).
    ///
    /// Scoped to the caller with no `shared:` escape: a note is by definition
    /// unreviewed material, and cross-agent unreviewed writes are how the
    /// pre-ANAI-165 shared bucket became unattributable.
    ///
    /// Returns the new fragment's id.
    async fn memory_note(
        &self,
        caller_agent_id: Option<&str>,
        text: &str,
        tags: &[String],
    ) -> Result<String, String> {
        let _ = (caller_agent_id, text, tags);
        Err("Memory notes are not available on this kernel handle".to_string())
    }

    /// Find agents by query (matches on name substring, tag, or tool name; case-insensitive).
    fn find_agents(&self, query: &str) -> Vec<AgentInfo>;

    /// Post a task to the shared task queue. Returns the task ID.
    async fn task_post(
        &self,
        title: &str,
        description: &str,
        assigned_to: Option<&str>,
        created_by: Option<&str>,
        payload: &[u8],
    ) -> Result<String, String>;

    /// Privileged enqueue for an `agent_send_async` wake. Distinct from
    /// [`Self::task_post`] because the wake-queue title namespace
    /// (`WAKE_TASK_PREFIX`) is a trust boundary: the kernel wake-consumer
    /// dispatches anything in it, so it must be writable ONLY through the
    /// capability-gated `agent_send_async` producer — never the ordinary
    /// `task_post` tool. The real kernel overrides this to call the substrate's
    /// privileged `task_post_wake`; the default delegates to `task_post` so
    /// mock/test handles keep working.
    async fn wake_post(
        &self,
        title: &str,
        description: &str,
        assigned_to: Option<&str>,
        created_by: Option<&str>,
        payload: &[u8],
    ) -> Result<String, String> {
        self.task_post(title, description, assigned_to, created_by, payload)
            .await
    }

    /// ANAI-147: wake-queue depth for one caller — `(pending, in_flight)`.
    ///
    /// Backs the honesty fix on `agent_send_async`'s result. A returned task id
    /// reads as "this will run", but a caller at its per-caller in-flight cap
    /// has its wake sit `pending` behind the cap — indefinitely, if a slot is
    /// leaked. The producer reports the depth so a stuck queue is visible in
    /// the tool result instead of being diagnosed by A/B probe a day later.
    ///
    /// Defaults to `(0, 0)` so mock/test handles keep working: the depth line
    /// is diagnostic, never load-bearing for dispatch.
    async fn wake_queue_depth(&self, _created_by: &str) -> Result<(usize, usize), String> {
        Ok((0, 0))
    }

    /// ANAI-122: consume this agent's one-shot terminal reply-right for the
    /// current woken turn, if the kernel minted one at wake-dispatch.
    ///
    /// Replaces the `WAKE_REPLY_RIGHT` task-local, which the process/IPC
    /// boundary severed for every subprocess-driven agent (e.g. the Claude Code
    /// driver): that agent calls its tools on the bridge-IPC handler task, NOT
    /// the kernel wake-dispatch task the task-local lived on, so `try_with`
    /// fail-closed for nearly the whole fleet. This lookup rides the kernel
    /// handle instead — which `agent_reply_async` already holds via
    /// `require_kernel` — so native (in-process) and subprocess (IPC) drivers
    /// read the reply-right identically.
    ///
    /// The real kernel REMOVES the entry (consume-on-read), so a second call in
    /// the same turn — or any later turn — finds `None`, which is what keeps the
    /// reply strictly one-shot. The default returns `None`, keeping mock/test
    /// handles (and any turn the kernel minted no right for) inert exactly like
    /// an origin turn.
    fn take_reply_right(&self, agent_id: &str) -> Option<crate::tool_runner::ReplyRight> {
        let _ = agent_id;
        None
    }

    /// ANAI-125: resolve `agent_name`'s channel binding into a `surface_to`
    /// route (`"<channel>:<recipient>"`) so an async wake that omits an
    /// explicit route can default to the ORIGINATOR's own home channel — the
    /// common case where a delegated reply belongs back where the originator
    /// lives. `None` when the agent has no channel/peer binding (an unbound
    /// worker), which preserves the pure fire-and-forget wake.
    ///
    /// Name-keyed to mirror the binding table (the router treats
    /// `AgentBinding::agent` as a name key) and the sibling prose summary that
    /// feeds `PromptContext::channel_binding`. The default returns `None`,
    /// keeping mock/test handles inert exactly like a bindingless agent.
    fn channel_binding_route(&self, agent_name: &str) -> Option<String> {
        let _ = agent_name;
        None
    }

    /// The effective tool names an agent would be offered on its next turn —
    /// builtins after `capabilities.tools`, profile, allow/blocklist and
    /// `exec_policy` filtering.
    ///
    /// ## Why (ANAI-210, failure class B)
    ///
    /// An `agent_send_async` to a target that structurally CANNOT do the work
    /// — the observed case was `sleep 60` sent to an agent with no
    /// `shell_exec` — is not a failure the reply guarantee can improve on. The
    /// target answers "I can't", or stays silent and burns the sender's whole
    /// deadline, to establish a fact that was knowable before the wake was ever
    /// enqueued. This exposes that fact so the send can fail fast instead.
    ///
    /// `agent_id` may be a UUID or a name, matching `agent_send_async`'s own
    /// target resolution. `None` means "cannot determine" — an unknown agent,
    /// or a handle that does not implement this — and callers MUST treat it as
    /// *no evidence of absence* rather than as an empty tool set, or a mock
    /// handle would start refusing every send. The default returns `None`, so
    /// pre-flight degrades to exactly today's behaviour.
    ///
    /// Advisory by construction: it is a snapshot read of the registry, and a
    /// manifest edit or per-turn mode filter between the check and the target's
    /// turn can still move the set. A false "present" simply degrades to
    /// today's behaviour; a false "missing" cannot happen, because a tool named
    /// here is one the resolver actually produced.
    ///
    /// Deliberately NOT [`AgentInfo::tools`], which is the raw
    /// `capabilities.tools` declaration: that list is EMPTY for an agent with
    /// unrestricted access (empty or `"*"` means "all tools"), so reading it as
    /// the effective set would refuse every send to the least restricted agents
    /// in the fleet. It also predates profile, allow/blocklist and `exec_policy`
    /// filtering, so it drifts from what the agent is actually offered.
    fn agent_tool_names(&self, agent_id: &str) -> Option<Vec<String>> {
        let _ = agent_id;
        None
    }

    /// Claim the next available task (optionally filtered by assignee). Returns task JSON or None.
    async fn task_claim(&self, agent_id: &str) -> Result<Option<serde_json::Value>, String>;

    /// Mark a task as completed with a result string.
    async fn task_complete(&self, task_id: &str, result: &str) -> Result<(), String>;

    /// List tasks, optionally filtered by status.
    async fn task_list(&self, status: Option<&str>) -> Result<Vec<serde_json::Value>, String>;

    /// Publish a custom event that can trigger proactive agents.
    async fn publish_event(
        &self,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<(), String>;

    /// Add an entity to the knowledge graph.
    async fn knowledge_add_entity(
        &self,
        entity: openfang_types::memory::Entity,
    ) -> Result<String, String>;

    /// Add a relation to the knowledge graph.
    async fn knowledge_add_relation(
        &self,
        relation: openfang_types::memory::Relation,
    ) -> Result<String, String>;

    /// Query the knowledge graph with a pattern.
    async fn knowledge_query(
        &self,
        pattern: openfang_types::memory::GraphPattern,
    ) -> Result<Vec<openfang_types::memory::GraphMatch>, String>;

    /// Create a cron job for the calling agent.
    async fn cron_create(
        &self,
        agent_id: &str,
        job_json: serde_json::Value,
    ) -> Result<String, String> {
        let _ = (agent_id, job_json);
        Err("Cron scheduler not available".to_string())
    }

    /// List cron jobs for the calling agent.
    async fn cron_list(&self, agent_id: &str) -> Result<Vec<serde_json::Value>, String> {
        let _ = agent_id;
        Err("Cron scheduler not available".to_string())
    }

    /// Cancel a cron job by ID.
    async fn cron_cancel(&self, job_id: &str) -> Result<(), String> {
        let _ = job_id;
        Err("Cron scheduler not available".to_string())
    }

    /// Check if a tool requires approval based on current policy.
    fn requires_approval(&self, tool_name: &str) -> bool {
        let _ = tool_name;
        false
    }

    /// ANAI-154: layer-3.5 judgement on ONE gated `shell_exec` invocation.
    ///
    /// Answers only "does the operator need to see this?" — never "is this
    /// binary permitted?", which `agent.toml` already decided deterministically
    /// two layers up. The kernel's impl is a single-shot
    /// `LlmDriver::complete()` against a pinned model, in the same shape as
    /// `compactor::compact_session`: not an agent, not a turn, no session, no
    /// re-entrancy.
    ///
    /// The default is a fail-closed `Escalate` carrying `JudgeOutcome::Inert`
    /// — today's behaviour, every command reaching a human — so every test
    /// double and WASM host shim compiles untouched and the *absence* of a
    /// gatekeeper can never be mistaken for a suppression. Same pattern as
    /// `send_to_agent_from` and `wake_post`.
    ///
    /// ANAI-189: returns `GateReview`, not a bare verdict. A timeout and a
    /// considered escalation are the same `GateVerdict` and must not be the
    /// same audit row; the outcome discriminant is the only thing that tells
    /// them apart, and it has to cross the trait boundary because the kernel
    /// is where the timeout is observed and the runtime is where the row is
    /// written.
    async fn gatekeeper_review(
        &self,
        req: &openfang_types::gatekeeper::GateRequest,
    ) -> openfang_types::gatekeeper::GateReview {
        let _ = req;
        openfang_types::gatekeeper::GateReview::failed(
            openfang_types::gatekeeper::JudgeOutcome::Inert,
        )
    }

    /// ANAI-187: is the gatekeeper in shadow mode?
    ///
    /// Shadow is a *runtime* concern, not a kernel one: the judge still runs
    /// and its verdict is still real, so `gatekeeper_review` must keep
    /// returning the honest answer. What changes is whether the runtime acts
    /// on it. Collapsing the two in the kernel would make the audit row say
    /// `escalate` for a command the judge wanted to suppress, which destroys
    /// the only data shadow mode exists to collect.
    ///
    /// Defaults to `false` so every existing test double and host shim
    /// compiles untouched, and so the *absence* of an answer means "act on
    /// verdicts", matching pre-ANAI-187 behaviour exactly.
    fn gatekeeper_shadow(&self) -> bool {
        false
    }

    /// ANAI-186: append one gatekeeper verdict to the Merkle audit chain.
    ///
    /// The gatekeeper's `tracing::info!` is not a ledger. The daemon's stderr
    /// is a plain append-only file with no rotation configured (verified on
    /// the reference macOS/launchd deployment: no `newsyslog.d` entry, tens of
    /// MB and growing), it is freely editable by anything that can write the
    /// file, and it can be filtered to nothing by a `RUST_LOG` edit made for
    /// an unrelated subsystem. A suppressed command is one no operator will
    /// ever see, so the only honest record of it is the hash-chained,
    /// sqlite-backed trail.
    ///
    /// `command` is the VERBATIM `shell_exec` string, never truncated, for the
    /// same reason `ApprovalRequest::command` is (ANAI-151): a record whose
    /// dangerous tail was cut is not a record. `metadata` carries the decision
    /// context (floor flags, whether the model was consulted, latency).
    ///
    /// Synchronous and infallible by design — a gate must never block or fail
    /// on its own bookkeeping. The default is a no-op: a host with no audit
    /// log loses the row, but must never lose the verdict.
    fn audit_gatekeeper_verdict(
        &self,
        agent_id: &str,
        command: &str,
        metadata: &str,
        outcome: &str,
    ) {
        let _ = (agent_id, command, metadata, outcome);
    }

    /// ANAI-241: append what the HUMAN did with a gated command.
    ///
    /// `audit_gatekeeper_verdict` records the judge's opinion at review time.
    /// It cannot record the operator's, because the operator has not decided
    /// yet — the row is sealed into the hash chain seconds to minutes before
    /// the click. So the disposition is its own row, correlated by the
    /// `gk=<uuid>` token both `metadata` strings carry.
    ///
    /// This is the instrument the `enabled = true` flip needs and does not
    /// have. In shadow every command prompts, so "judge said suppress" and
    /// "human approved" are both true of the same command and the chain is
    /// unambiguous. Post-flip they diverge, and nothing today distinguishes a
    /// command a human approved from one the judge suppressed and the human
    /// never saw. That difference IS the safety boundary.
    ///
    /// Synchronous, infallible, no-op by default — same contract as the
    /// verdict row. A gate must never block or fail on its own bookkeeping.
    fn audit_gatekeeper_disposition(
        &self,
        agent_id: &str,
        command: &str,
        metadata: &str,
        disposition: &str,
    ) {
        let _ = (agent_id, command, metadata, disposition);
    }

    /// Request approval for a tool execution. Blocks until approved/denied/timed out.
    /// Returns the verbatim `ApprovalDecision`. It deliberately does NOT
    /// collapse to a bool: "Ben said no", "nobody was looking", and "your
    /// queue is full" are three different facts, and only the first is
    /// terminal (ANAI-153). Callers wanting the old semantics should ask
    /// `decision == ApprovalDecision::Approved` explicitly; callers that
    /// report failure upward must distinguish `ApprovalDecision::is_retryable`.
    ///
    /// `command` carries the verbatim `shell_exec` command string. Like
    /// `cache_binary`, it is captured at the gate where structured tool input
    /// still exists — render sites downstream must never re-derive it from the
    /// truncated `action_summary` (ANAI-151). `None` for non-shell tools.
    ///
    /// `gatekeeper_note` carries the gate's one-line opinion for rendering on
    /// the prompt itself (ANAI-188). A separate parameter rather than a prefix
    /// on `action_summary` on purpose: `action_summary` is agent-controlled,
    /// and an annotation the operator reads as the machine's verdict must not
    /// share a channel with text the requesting agent can author. `None` when
    /// the gate is inert or the tool is not gateable.
    async fn request_approval(
        &self,
        agent_id: &str,
        tool_name: &str,
        action_summary: &str,
        origin: Option<&openfang_types::approval::ApprovalOrigin>,
        cache_binary: Option<&str>,
        command: Option<&str>,
        gatekeeper_note: Option<&str>,
    ) -> Result<openfang_types::approval::ApprovalDecision, String> {
        let _ = (agent_id, tool_name, action_summary, origin, cache_binary);
        let _ = (command, gatekeeper_note);
        Ok(openfang_types::approval::ApprovalDecision::Approved) // Default: auto-approve
    }

    /// List available Hands and their activation status.
    async fn hand_list(&self) -> Result<Vec<serde_json::Value>, String> {
        Err("Hands system not available".to_string())
    }

    /// Install a Hand from TOML content.
    async fn hand_install(
        &self,
        toml_content: &str,
        skill_content: &str,
    ) -> Result<serde_json::Value, String> {
        let _ = (toml_content, skill_content);
        Err("Hands system not available".to_string())
    }

    /// Activate a Hand — spawns a specialized autonomous agent.
    async fn hand_activate(
        &self,
        hand_id: &str,
        config: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        let _ = (hand_id, config);
        Err("Hands system not available".to_string())
    }

    /// Check the status and dashboard metrics of an active Hand.
    async fn hand_status(&self, hand_id: &str) -> Result<serde_json::Value, String> {
        let _ = hand_id;
        Err("Hands system not available".to_string())
    }

    /// Deactivate a running Hand and stop its agent.
    async fn hand_deactivate(&self, instance_id: &str) -> Result<(), String> {
        let _ = instance_id;
        Err("Hands system not available".to_string())
    }

    /// List discovered external A2A agents as (name, url) pairs.
    fn list_a2a_agents(&self) -> Vec<(String, String)> {
        vec![]
    }

    /// Get the URL of a discovered external A2A agent by name.
    fn get_a2a_agent_url(&self, name: &str) -> Option<String> {
        let _ = name;
        None
    }

    /// Send a message to a user on a named channel adapter (e.g., "email", "telegram").
    /// When `thread_id` is provided, the message is sent as a thread reply.
    /// Returns a confirmation string on success.
    /// Get the default recipient for a channel (e.g. default_chat_id for Telegram).
    async fn get_channel_default_recipient(&self, channel: &str) -> Option<String> {
        let _ = channel;
        None
    }

    async fn send_channel_message(
        &self,
        channel: &str,
        recipient: &str,
        message: &str,
        thread_id: Option<&str>,
        workspace_root: Option<&std::path::Path>,
    ) -> Result<String, String> {
        let _ = (channel, recipient, message, thread_id, workspace_root);
        Err("Channel send not available".to_string())
    }

    /// Send media content (image/file) to a user on a named channel adapter.
    /// `media_type` is "image" or "file", `media_url` is the URL, `caption` is optional text.
    /// When `thread_id` is provided, the media is sent as a thread reply.
    async fn send_channel_media(
        &self,
        channel: &str,
        recipient: &str,
        media_type: &str,
        media_url: &str,
        caption: Option<&str>,
        filename: Option<&str>,
        thread_id: Option<&str>,
    ) -> Result<String, String> {
        let _ = (
            channel, recipient, media_type, media_url, caption, filename, thread_id,
        );
        Err("Channel media send not available".to_string())
    }

    /// Send a local file (raw bytes) to a user on a named channel adapter.
    /// Used by the `channel_send` tool when `file_path` is provided.
    /// When `thread_id` is provided, the file is sent as a thread reply.
    async fn send_channel_file_data(
        &self,
        channel: &str,
        recipient: &str,
        data: Vec<u8>,
        filename: &str,
        mime_type: &str,
        thread_id: Option<&str>,
    ) -> Result<String, String> {
        let _ = (channel, recipient, data, filename, mime_type, thread_id);
        Err("Channel file data send not available".to_string())
    }

    /// Refresh an agent's last_active timestamp without changing any other state.
    /// Called by the agent loop before long LLM calls to prevent heartbeat false-positives.
    fn touch_agent(&self, agent_id: &str) {
        let _ = agent_id;
    }

    /// Return the daemon's bridge `TokenIssuer`, if one has been wired.
    ///
    /// The agent loop calls this when constructing fallback drivers so they
    /// can participate in the hardened bridge handshake. The default returns
    /// `None`, keeping mock/test impls (e.g. `FakeKernelHandle`) on the legacy
    /// UUID path; `OpenFangKernel` overrides it to expose its authority.
    fn token_issuer(&self) -> Option<Arc<dyn TokenIssuer>> {
        None
    }

    /// Spawn an agent with capability inheritance enforcement.
    /// `parent_caps` are the parent's granted capabilities. The kernel MUST verify
    /// that every capability in the child manifest is covered by `parent_caps`.
    async fn spawn_agent_checked(
        &self,
        manifest_toml: &str,
        parent_id: Option<&str>,
        parent_caps: &[openfang_types::capability::Capability],
    ) -> Result<(String, String), String> {
        // Default: delegate to spawn_agent (no enforcement)
        // The kernel MUST override this with real enforcement
        let _ = parent_caps;
        self.spawn_agent(manifest_toml, parent_id).await
    }
}
