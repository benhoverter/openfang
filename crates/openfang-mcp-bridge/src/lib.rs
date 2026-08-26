//! # openfang-mcp-bridge
//!
//! MCP (Model Context Protocol) bridge for OpenFang Agent OS.
//!
//! ## What this crate is
//!
//! A protocol adapter that exposes OpenFang's tool surface to Claude Code
//! subprocesses (and other MCP clients) over stdio. One MCP server instance
//! per parent agent, scoped to that agent's identity and the capabilities
//! declared in its `agent.toml`.
//!
//! ## What this crate is NOT
//!
//! - It does NOT depend on `openfang-runtime`, `openfang-kernel`, or
//!   `openfang-memory` directly. The bridge consumes a narrow
//!   [`ToolDispatcher`] trait that the runtime exposes; the runtime owns
//!   identity, the kernel owns dispatch, the memory subsystem stays untouched.
//! - It does NOT define OpenFang's tool surface beyond a small built-in slice
//!   (see [`built_in_tools`]). The schemas declared here mirror
//!   `openfang_runtime::tool_runner::builtin_tool_definitions()` for the
//!   four ANAI-30 allowlisted tools — kept in lockstep deliberately.
//!
//! ## Project status
//!
//! ANAI-30 step 3. The bridge now:
//! - Defines the [`ToolDispatcher`] seam trait — the runtime (or, in the real
//!   topology, the daemon-bound IPC client) implements it.
//! - Registers the four-tool ANAI-30 surface (`file_read`, `file_list`,
//!   `agent_list`, `channel_send`) and translates `tools/call` into
//!   [`ToolDispatcher::call`].
//! - Filters its advertised tool list against [`ToolDispatcher::allowed_tools`]
//!   so an agent never sees tools its capabilities don't permit.
//!
//! Identity is bound at construction time. The IPC client implementation
//! lives in `main.rs`. ANAI-31 will replace the in-band-agent-id stub with
//! token-derived identity.

pub mod protocol;

use std::sync::Arc;

use rmcp::{model::*, service::RequestContext, ErrorData as McpError, ServerHandler};

use crate::protocol::UpstreamToolDef;

/// Narrow seam between the bridge and the OpenFang runtime.
///
/// The runtime (or, in the real topology, an IPC-backed adapter that talks
/// to the daemon) implements this trait and hands an `Arc<dyn ToolDispatcher>`
/// to the bridge at startup, scoped to a specific agent identity. The bridge
/// translates incoming MCP `tools/call` requests into [`ToolDispatcher::call`]
/// invocations.
///
/// **Identity is bound at construction time, not per-call.** A bridge instance
/// only ever speaks for one agent. This is the security invariant tracked by
/// ANAI-31.
#[async_trait::async_trait]
pub trait ToolDispatcher: Send + Sync {
    /// Identity of the agent this dispatcher is bound to. Used for audit
    /// logging and for cross-checking against `agent.toml` capabilities.
    fn agent_id(&self) -> &str;

    /// List of tool names this dispatcher will accept, derived from
    /// `agent.toml` capabilities. The bridge filters its advertised tool list
    /// against this set.
    ///
    /// For ANAI-30 this is the static four-tool slice; ANAI-31+ will derive
    /// it from `agent.toml`.
    fn allowed_tools(&self) -> Vec<String>;

    /// Forwarded upstream MCP tools the daemon told us this agent may
    /// invoke (from `agent.toml mcp_servers`, resolved server-side at
    /// list-upstream time). Default: none.
    ///
    /// Names are already namespaced (`mcp_{server}_{tool}`) and are
    /// advertised by the bridge in addition to the built-in surface
    /// without going through [`Self::allowed_tools`] — that field gates
    /// the OpenFang built-in slice; upstream gating is handled
    /// server-side by the daemon against the agent's MCP allowlist.
    fn upstream_tools(&self) -> Vec<UpstreamToolDef> {
        Vec::new()
    }

    /// Projected `file_convert` `options` sub-schema (ANAI-131), if the
    /// dispatcher can supply one. The bridge injects it into the advertised
    /// `file_convert` tool schema so its option surface matches what the
    /// dispatcher accepts. Default: `None` — the tool is still advertised,
    /// just without a projected `options` property. The runtime-free bridge
    /// never computes this itself; the IPC-backed dispatcher receives it from
    /// the daemon at handshake time (see `HelloAck::Ok`).
    fn convert_options_schema(&self) -> Option<serde_json::Value> {
        None
    }

    /// Invoke a tool by name with a JSON argument blob. The dispatcher is
    /// responsible for capability enforcement; the bridge MUST NOT assume the
    /// caller is trusted.
    async fn call(
        &self,
        tool_name: &str,
        args: serde_json::Value,
    ) -> Result<DispatchOk, ToolDispatchError>;
}

/// Successful dispatch outcome. Maps onto MCP's `CallToolResult` shape:
/// `content` becomes a single text content block, `is_error` becomes the
/// `isError` flag.
///
/// Note the distinction from [`ToolDispatchError`]: a tool that ran but
/// reported a failure to the model is `Ok(DispatchOk { is_error: true })`.
/// `Err(_)` means dispatch itself failed (unknown tool, not permitted,
/// transport error) — the bridge surfaces those as MCP errors instead.
#[derive(Debug, Clone)]
pub struct DispatchOk {
    pub content: String,
    pub is_error: bool,
}

/// Errors a [`ToolDispatcher`] can return. Bridge maps these to MCP errors.
#[derive(Debug, thiserror::Error)]
pub enum ToolDispatchError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("tool '{0}' not permitted for this agent")]
    NotPermitted(String),
    #[error("invalid arguments for tool '{tool}': {reason}")]
    InvalidArgs { tool: String, reason: String },
    #[error("tool execution failed: {0}")]
    Execution(#[from] anyhow::Error),
}

/// Built-in tool definitions advertised by the bridge in `tools/list`.
///
/// **These schemas mirror the equivalent entries in
/// `openfang_runtime::tool_runner::builtin_tool_definitions()`.** They are
/// duplicated here rather than imported because the bridge crate is
/// runtime-free by design (see crate-level docs). If the runtime's schemas
/// drift, update both sides.
///
/// Current surface:
/// - `file_read`, `file_list` — workspace-scoped, no kernel dependency
/// - `agent_list` — exercises `KernelHandle::list_agents`
/// - `channel_send` — exercises `KernelHandle::send_channel_message`,
///   one of the OpenFang-only capabilities a bare CC subprocess lacks
/// - `agent_send` — inter-agent messaging; first tool added past the
///   ANAI-30 slice. Per-agent gating via `OPENFANG_BRIDGE_ALLOWED`
///   (sourced from each agent's `agent.toml` capabilities) decides
///   whether any given bridge instance actually advertises it.
///
/// Default tool allowlist used by `main.rs` when [`OPENFANG_BRIDGE_ALLOWED`]
/// is unset (legacy/dev path). Lives in the library — not in `main.rs` — so
/// the bridge_ipc drift-catcher tests in `openfang-api` can assert the
/// surface relationships between this set, [`built_in_tools`], and the
/// daemon-side `bridge_ipc::ALLOWED_TOOLS`.
///
/// **Safe-by-default policy (S7-06 / S4-02):** the default set deliberately
/// *excludes* the privileged agent-lifecycle tools enumerated by
/// [`PRIVILEGED_DEFAULT_DENY`] — `agent_spawn`, `agent_kill`,
/// `agent_activate`. These remain in [`built_in_tools`] and in
/// `bridge_ipc::ALLOWED_TOOLS` (daemon dispatch) so that agents whose
/// `agent.toml` grants them can still invoke them: the runtime threads
/// the manifest-derived allowlist through `OPENFANG_BRIDGE_ALLOWED`
/// (see `agent_loop.rs`'s `allowed_tools` plumbing). Only the
/// no-env-var fallback — used in tests, dev shells, and any caller
/// who forgot to set the var — is downgraded to the non-privileged
/// subset.
///
/// [`OPENFANG_BRIDGE_ALLOWED`]: ../../openfang_mcp_bridge/index.html
pub const DEFAULT_ALLOWED: &[&str] = &[
    "file_read",
    "file_list",
    "file_write",
    "create_directory",
    "web_fetch",
    "agent_list",
    "channel_send",
    "agent_send",
    // ANAI-122: default-safe. In DEFAULT_ALLOWED (not PRIVILEGED_DEFAULT_DENY)
    // because being advertised grants nothing — the tool is inert without a
    // one-shot reply-right token the wake-consumer mints for a woken turn.
    "agent_reply_async",
    "memory_store",
    "memory_recall",
    "agent_find",
    "shell_exec",
    "web_search",
    "apply_patch",
    "file_convert",
    // ANAI-194. Default-safe: both are scoped to the calling agent's own
    // episodes with no cross-agent escape, and neither can delete a memory row.
    "memory_episode_close",
    "memory_status",
    // ANAI-166. Default-safe on the same test: scoped to the caller's own
    // rows, and a note is append-only — there is no agent-facing verb that
    // removes or rewrites one (ADR 0002 §2.6).
    "memory_note",
    // ANAI-204. These do NOT pass the "scoped to the caller's own rows" test —
    // a `project`- or `user`-scoped slot belongs to its subject, so one agent
    // can overwrite a claim another agent wrote. That is the design (see the
    // v14 index amendment), not an oversight, and it is why they are listed
    // here with a reason rather than by analogy to `memory_note`.
    //
    // Default-safe on a different test: no write destroys anything. A
    // supersession copies the outgoing claim into `fact_history` in the same
    // transaction, so the worst a confused agent can do is make a slot say
    // something wrong *and leave a signed record of what it used to say*. There
    // is no agent-facing verb that deletes a fact or a history row. `global`
    // scope, the one case where a bad write would render into ~70 prompts, is
    // refused at the writer.
    "memory_fact",
    "memory_history",
];

/// Agent-lifecycle tools that are dispatchable by the daemon and advertised
/// by the bridge, **but excluded from [`DEFAULT_ALLOWED`]** so the
/// no-env-var fallback path cannot reach them.
///
/// Closes S7-06 / S4-02 (bridge-side): in the legacy/dev path
/// (`OPENFANG_BRIDGE_ALLOWED` unset) an agent's bridge would otherwise be
/// able to spawn / kill / activate sibling agents regardless of its
/// manifest. Production callers thread `OPENFANG_BRIDGE_ALLOWED` from the
/// manifest-derived `available_tools`, so opted-in agents are unaffected;
/// only the fallback is narrowed.
///
/// Drift-pin: the `bridge_ipc::allowlist_*` tests assert that every entry
/// here is **present** in `ALLOWED_TOOLS` and `built_in_tools()` and
/// **absent** from `DEFAULT_ALLOWED`.
pub const PRIVILEGED_DEFAULT_DENY: &[&str] = &[
    "agent_spawn",
    "agent_kill",
    "agent_activate",
    "agent_send_async",
];

pub fn built_in_tools() -> Vec<Tool> {
    use serde_json::json;

    fn obj(v: serde_json::Value) -> std::sync::Arc<serde_json::Map<String, serde_json::Value>> {
        match v {
            serde_json::Value::Object(m) => std::sync::Arc::new(m),
            _ => std::sync::Arc::new(serde_json::Map::new()),
        }
    }

    vec![
        Tool::new(
            "file_read",
            "Read the contents of a file. Paths are relative to the agent workspace.",
            obj(json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to read" }
                },
                "required": ["path"]
            })),
        ),
        Tool::new(
            "file_list",
            "List files in a directory. Paths are relative to the agent workspace.",
            obj(json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The directory path to list" }
                },
                "required": ["path"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `file_write`. Workspace-
        // scoped via the daemon-side `FS_SANDBOXED_TOOLS` gate in `bridge_ipc`.
        Tool::new(
            "file_write",
            "Write content to a file. Paths are relative to the agent workspace.",
            obj(json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to write to" },
                    "content": { "type": "string", "description": "The content to write" }
                },
                "required": ["path", "content"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `create_directory`.
        // Workspace-scoped via the daemon-side `FS_SANDBOXED_TOOLS` gate.
        // Idempotent: succeeds if the directory already exists; creates
        // intermediate parents as needed.
        Tool::new(
            "create_directory",
            "Create a directory (and any missing parent directories) at the given path. \
             Paths are relative to the agent workspace. Idempotent: succeeds if the \
             directory already exists.",
            obj(json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The directory path to create" }
                },
                "required": ["path"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `web_fetch`. No FS touch,
        // so it does not appear in `FS_SANDBOXED_TOOLS`; SSRF protection lives
        // in the runtime implementation.
        Tool::new(
            "web_fetch",
            "Fetch a URL with SSRF protection. Supports GET/POST/PUT/PATCH/DELETE. \
             For GET, HTML is converted to Markdown. For other methods, returns raw \
             response body.",
            obj(json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch (http/https only)" },
                    "method": { "type": "string", "enum": ["GET","POST","PUT","PATCH","DELETE"], "description": "HTTP method (default: GET)" },
                    "headers": { "type": "object", "description": "Custom HTTP headers as key-value pairs" },
                    "body": { "type": "string", "description": "Request body for POST/PUT/PATCH" }
                },
                "required": ["url"]
            })),
        ),
        Tool::new(
            "agent_list",
            "List all currently running agents with their IDs, names, states, and models.",
            obj(json!({
                "type": "object",
                "properties": {}
            })),
        ),
        Tool::new(
            "channel_send",
            "Send a message to a user on a configured channel (email, telegram, slack, \
             discord, etc). For email: recipient is the email address; optionally set \
             subject. Use thread_id to reply in a specific thread/topic. Use \
             `attachments` (array of file paths, workspace-relative preferred) to \
             attach files alongside the message; paths resolve against the agent \
             workspace and the same allow-root rules apply as for inline \
             `<openfang:attach path=\"...\"/>` directives.",
            obj(json!({
                "type": "object",
                "properties": {
                    "channel": { "type": "string", "description": "Channel adapter name (e.g., 'email', 'telegram', 'slack', 'discord')" },
                    "recipient": { "type": "string", "description": "Platform-specific recipient identifier (email address, user ID, etc.)" },
                    "subject": { "type": "string", "description": "Optional subject line (used for email; ignored for other channels)" },
                    "message": { "type": "string", "description": "The message body to send" },
                    "attachments": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional file paths to attach. Workspace-relative paths are preferred (resolved against the agent's workspace root); absolute paths are also accepted. Same security gating as inline `<openfang:attach path=\"...\"/>` directives."
                    },
                    "thread_id": { "type": "string", "description": "Thread/topic ID to reply in" }
                },
                "required": ["channel", "recipient"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_send`. Kept in
        // sync with that schema by hand; the bridge crate is runtime-free
        // by design and can't import the source. Per-agent gating via
        // `OPENFANG_BRIDGE_ALLOWED` decides whether this tool is actually
        // advertised + dispatchable for any given bridge instance.
        Tool::new(
            "agent_send",
            "Send a message to another agent and receive their response. \
             Accepts UUID or agent name. Use agent_find first to discover agents.",
            obj(json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The target agent's UUID or name" },
                    "message": { "type": "string", "description": "The message to send to the agent" }
                },
                "required": ["agent_id", "message"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_send_async`.
        // Privileged (fire-and-forget cross-agent wake). Advertised but
        // excluded from `DEFAULT_ALLOWED` (see `PRIVILEGED_DEFAULT_DENY`);
        // reaches a bridge only via manifest-derived `OPENFANG_BRIDGE_ALLOWED`.
        Tool::new(
            "agent_send_async",
            "Wake another agent asynchronously (fire-and-forget). Queues the \
             message for the target and returns immediately — the caller does NOT \
             block on the target's loop and receives NO inline reply. Use this \
             instead of agent_send when you want to hand off work without waiting, \
             or to avoid the head-of-line blocking of a synchronous A->B call. \
             Accepts UUID or agent name. Optionally pass surface_to \
             (\"<channel>:<recipient>\") to have the target's eventual \
             agent_reply_async answer auto-posted to that channel. \
             You are GUARANTEED exactly one reply per call: if the target \
             answers, you get its answer; if it cannot or does not, the daemon \
             closes the correlation itself and tells you why. Pass timeout_secs \
             to bound how long that takes. Pass requires_tools to refuse the send \
             outright when the target lacks a tool the work needs, instead of \
             spending the whole deadline discovering it.",
            obj(json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The target agent's UUID or name to wake" },
                    "message": { "type": "string", "description": "The message delivered to the target when it runs" },
                    "surface_to": { "type": "string", "description": "Optional channel route, formatted \"<channel>:<recipient>\" (e.g. \"discord:1086446153098342510\"). When set, the target's one-shot agent_reply_async answer is auto-posted to this channel by the daemon. Omit for a pure fire-and-forget wake with no surfacing." },
                    "timeout_secs": { "type": "integer", "description": "Optional deadline in seconds. You are guaranteed a reply within roughly this long: if the target has not answered by then, its turn is ABORTED and the daemon sends you a timeout reply instead. Set it to how long you actually expect the work to take, with headroom — an over-tight value kills legitimate work, and partial side effects from the aborted turn may persist. Clamped into the operator's configured band; omit to accept the configured default." }
                    ,"requires_tools": { "type": "array", "items": { "type": "string" }, "description": "Optional pre-flight: tool names the target MUST have for this work (e.g. [\"shell_exec\"]). Checked BEFORE the wake is enqueued — if any is missing the call fails immediately, names what is missing, mints NO correlation and consumes no deadline, so nothing was sent and re-sending a corrected request is safe. Use it whenever the request depends on a specific capability; without it you spend the full deadline learning the target could never have done it. Omit for no check." }
                },
                "required": ["agent_id", "message"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_reply_async`.
        // NON-privileged and default-safe: it sits in `DEFAULT_ALLOWED`
        // (unlike its origination sibling) precisely because advertising it
        // grants nothing — it is INERT without a one-shot reply-right token
        // that only the wake-consumer can mint into task-local scope for a
        // woken turn (ANAI-122). Takes no target: the initiator is fixed by
        // the token, so it can never originate a wake to an arbitrary agent.
        Tool::new(
            "agent_reply_async",
            "Send a ONE-SHOT terminal reply to the agent that woke you via \
             agent_send_async (fire-and-forget). Valid ONLY inside a turn that \
             another agent woke asynchronously — it answers that initiator and \
             no one else, so it takes no target. The reply is terminal: the \
             initiator receives your message as its own woken turn and surfaces \
             or continues, but cannot bounce back. Outside a woken turn, or \
             after you have already replied once this turn, the call refuses.",
            obj(json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The reply delivered to your initiator when it runs" }
                },
                "required": ["message"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_spawn`. High-
        // capability tool (creates new agents). Gated per-agent via
        // agent.toml; daemon-side Gate 2 enforces.
        Tool::new(
            "agent_spawn",
            "Spawn a new agent from a TOML manifest. Returns the new agent's ID and name.",
            obj(json!({
                "type": "object",
                "properties": {
                    "manifest_toml": {
                        "type": "string",
                        "description": "The agent manifest in TOML format (must include name, module, [model], and [capabilities])"
                    }
                },
                "required": ["manifest_toml"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_kill`. High-
        // capability tool (terminates another agent). Gated per-agent.
        Tool::new(
            "agent_kill",
            "Kill (terminate) another agent by its ID.",
            obj(json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The agent's UUID to kill" }
                },
                "required": ["agent_id"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_activate`. Wakes
        // a Suspended/Crashed/Created agent. Terminated agents cannot be
        // revived.
        Tool::new(
            "agent_activate",
            "Activate (wake up) an inactive agent so it can receive messages \
             and process events. Use this when agent_list shows an agent in a \
             Suspended, Crashed, or Created state and you want to delegate work \
             to it via agent_send. Terminated agents cannot be revived.",
            obj(json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The target agent's UUID or human-readable name"
                    }
                },
                "required": ["agent_id"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `memory_store`. Kernel-
        // managed shared memory; no FS sandbox needed (kernel scopes writes).
        Tool::new(
            "memory_store",
            "Store a value in YOUR OWN memory namespace, private to you. Prefix the key with 'shared:' to write to the cross-agent namespace instead (e.g. 'shared:release_freeze') - use that only for state other agents genuinely need to read.",
            obj(json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "The storage key. Prefix with 'shared:' for cross-agent state." },
                    "value": { "type": "string", "description": "The value to store (JSON-encode objects/arrays, or pass a plain string)" }
                },
                "required": ["key", "value"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `memory_recall`. Read-only
        // companion to memory_store.
        //
        // ANAI-166: `query` was added here in the SAME commit that added it to
        // the runtime definition. ANAI-126 is the standing example of what a
        // one-sided param addition costs — every subprocess agent goes on
        // seeing the old schema and can never call the new shape, with no
        // error anywhere to say so. `built_in_tools_surface` asserts names
        // only, so `bridge_memory_recall_advertises_query` below is the guard
        // that actually catches this.
        Tool::new(
            "memory_recall",
            "Search YOUR OWN memory for relevant context: pass 'query' with what you are looking for, in words. Pass 'key' instead for an exact stored key ('shared:' prefix reads the cross-agent namespace). Exactly one of the two.",
            obj(json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What you are looking for, in plain words. Searches by meaning when embeddings are configured, by text match otherwise." },
                    "key": { "type": "string", "description": "Exact storage key, for values written with memory_store. Prefix with 'shared:' for cross-agent state." },
                    "scope": { "type": "string", "description": "Optional: restrict to one memory scope, e.g. 'episodic'." },
                    "kind": { "type": "string", "description": "Optional: restrict to one kind of memory, e.g. 'note'." },
                    "limit": { "type": "integer", "description": "Maximum results to return (default 5, maximum 25)." }
                },
                "required": []
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `agent_find`. Read-only
        // discovery; pairs with agent_send.
        Tool::new(
            "agent_find",
            "Discover agents by name, tag, tool, or description. Use to find specialists before delegating work.",
            obj(json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (matches agent name, tags, tools, description)" }
                },
                "required": ["query"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `shell_exec`. Daemon-side
        // `bridge_ipc` enforces workspace cwd sandbox + `exec_policy` (Full /
        // Allowlist / Denylist / None) from the calling agent's agent.toml.
        // The bridge advertises the tool unconditionally; per-agent capability
        // gating via `OPENFANG_BRIDGE_ALLOWED` decides whether any given bridge
        // instance actually exposes it, and Gate 2 in `bridge_ipc` rejects
        // commands that fall outside the agent's exec_policy.
        Tool::new(
            "shell_exec",
            "Execute a shell command and return its output. Runs in the agent's \
             workspace directory; commands are subject to the agent's exec_policy \
             (Full / Allowlist / Denylist / None) as declared in agent.toml.",
            obj(json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to execute" },
                    "timeout_seconds": { "type": "integer", "description": "Timeout in seconds (default: 30)" }
                },
                "required": ["command"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `apply_patch`. Workspace-
        // scoped via the daemon-side `FS_SANDBOXED_TOOLS` gate in `bridge_ipc`
        // — `tool_apply_patch` resolves every patch-embedded path against
        // `workspace_root`, so the no-workspace fail-closed gate is critical
        // (same sibling-leak surface as `file_write`).
        //
        // Why this is bridged: serves as a surgical-edit alternative to
        // whole-file `file_write` rewrites for CC subprocesses that lack
        // CC's native `Edit` tool. A native `string_edit` follow-up may
        // replace this as the primary edit ergonomic; for now, apply_patch
        // is the closest thing we have to Edit's emit-cost profile.
        Tool::new(
            "apply_patch",
            "Apply a multi-hunk diff patch to add, update, move, or delete files. \
             Use this for targeted edits instead of full file overwrites. Paths in \
             the patch are resolved relative to the agent workspace.",
            obj(json!({
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The patch in *** Begin Patch / *** End Patch format. Use *** Add File:, *** Update File:, *** Delete File: markers. Hunks use @@ headers with space (context), - (remove), + (add) prefixed lines."
                    }
                },
                "required": ["patch"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `web_search`. Pure-net,
        // no FS sandbox. Multi-provider (Tavily → Brave → Perplexity → DDG
        // fallback chain) configured via the daemon's `WebToolsContext`.
        Tool::new(
            "web_search",
            "Search the web using multiple providers (Tavily, Brave, Perplexity, \
             DuckDuckGo) with automatic fallback. Returns structured results with \
             titles, URLs, and snippets.",
            obj(json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" },
                    "max_results": { "type": "integer", "description": "Maximum number of results to return (default: 5, max: 20)" }
                },
                "required": ["query"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` -> `file_convert`. Workspace-
        // scoped: the runtime requires a `workspace_root` and resolves the
        // `input`/`output` paths against it; the daemon-side `FS_SANDBOXED_TOOLS`
        // gate fails the call closed when no workspace is registered. The tool
        // dispatch + recipe security core live on the file-convert leaf; this
        // entry is the cross-leaf advertise surface the bridge owes it.
        Tool::new(
            "file_convert",
            "Convert a workspace file from one format to another using an \
             allowlisted recipe table (e.g. Markdown to PDF). The source format \
             is inferred from the input file extension; the target format is the \
             'format' argument. Only conversions defined in the recipe manifest \
             are permitted. Paths are relative to the agent workspace.",
            obj(json!({
                "type": "object",
                "properties": {
                    "format": { "type": "string", "description": "Target format / output extension, e.g. \"pdf\"" },
                    "input": { "type": "string", "description": "Workspace-relative path to the source file. Its extension determines the source format." },
                    "output": { "type": "string", "description": "Optional workspace-relative output path. If omitted, the input path with the target extension is used." },
                    "preset": { "type": "string", "description": "Optional render preset selecting size/scale, e.g. \"mobile\", \"tablet\", \"desktop\", \"wide\". Must be one offered by the target recipe; omit to use the recipe's default preset. Ignored by recipes that define no presets." }
                },
                "required": ["format", "input"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` -> `memory_episode_close` /
        // `memory_status` (ANAI-194). Schemas must match the runtime
        // definitions verbatim — drift here is silent, since the bridge is
        // what a subprocess agent actually sees.
        Tool::new(
            "memory_episode_close",
            "Close the current episode - the stretch of turns your recent work \
             is grouped into - and label it. Call this when a piece of work is \
             finished, before you move to something unrelated. A new episode \
             opens on your next turn. Harmless to call when nothing is open. \
             Pass reset_context to also start the next episode with a clean \
             conversation window, and prime_for to have that fresh window \
             opened with what durable memory knows about the project you are \
             moving to.",
            obj(json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "Short label for the work that just finished, e.g. \"git trunk cutover\"" },
                    "summary": { "type": "string", "description": "Optional few-sentence wrap-up of what happened and what was decided. It is kept as a note on this episode and fed to the summariser as material; the episode's own summary is always synthesized afterwards, never taken from here." },
                    "reason": { "type": "string", "enum": ["explicit"], "description": "Why the episode is closing. Only 'explicit' is available to agents; timer closes are the system's." },
                    "reset_context": { "type": "boolean", "description": "Default false. When true, your conversation window is cleared at the END of this turn so the next episode starts fresh. Your durable memory is untouched and the running summary of earlier work is kept - you will not forget what happened, you stop re-reading it verbatim. Only set this when the work really is finished; doing it mid-task discards the detail you still need." },
                    "prime_for": { "type": "string", "description": "Optional project slug, e.g. \"openfang-fork\". Only meaningful with reset_context. The next episode opens with a short briefing assembled from durable memory for that project - your recently closed episodes and what the fleet currently believes about it - instead of you having to ask for it. Omitting it clears any previous priming." }
                },
                "required": ["title"]
            })),
        ),
        Tool::new(
            "memory_status",
            "Report the state of your own memory: which episode is open, how \
             many turns it has captured, how long it has been idle, and when it \
             will close on its own. Use it to notice you have drifted onto \
             unrelated work.",
            obj(json!({
                "type": "object",
                "properties": {}
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `memory_note` (ANAI-166).
        // Tail-appended, like every other bridge tool: `built_in_tools_surface`
        // asserts an exact ordered vec, so a mid-list insert would shift every
        // later entry and make an unrelated diff look like the cause.
        Tool::new(
            "memory_note",
            "Jot something down in your own memory, in your own words - a decision, an observation, something worth keeping. Cheap and unstructured; no key needed. It is attached to your current episode.",
            obj(json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "What to remember, in plain words." },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional short labels to help find this later." }
                },
                "required": ["text"]
            })),
        ),
        // Mirrors `openfang_runtime::tool_runner` → `memory_fact` /
        // `memory_history` (ANAI-204). Descriptions and schemas are copied
        // verbatim from the runtime definitions; Invariant C
        // (`openfang_api::bridge_ipc::tests::advertised_tool_schemas_match_runtime`)
        // asserts property-set equality, so a parameter added on one side and
        // not the other fails the api suite rather than going quietly
        // unreachable for every subprocess agent. Tail-appended, like every
        // other bridge tool.
        Tool::new(
            "memory_fact",
            "Read or write one durable claim slot - a named box holding the CURRENT truth about something, overwritten in place when it changes. Pass 'claim' to write; omit it to read what is already there. Keys are 'namespace.slot', e.g. 'repo.trunk_model' or 'project.tttb.promotion_status'; the namespaces are agent, build, deploy, delivery, memory, project, repo, tool and user. Store state that gets updated, not events that happened - a ticket id or a date in the key means it belongs in memory_note instead. Read a slot before you write it: prefer a key that already exists over minting a near-duplicate.",
            obj(json!({
                "type": "object",
                "properties": {
                    "scope": { "type": "string", "enum": ["agent", "project", "user"], "description": "Whose truth this is: 'agent' (about you), 'project', or 'user'." },
                    "scope_ref": { "type": "string", "description": "What the claim is about - the project or user slug, e.g. \"openfang-fork\". Required for 'project' and 'user'; ignored for 'agent', which is always you." },
                    "key": { "type": "string", "description": "The slot name, 'namespace.slot', e.g. \"repo.trunk_model\". Up to 7 dot-separated segments." },
                    "claim": { "type": "string", "description": "The claim itself, in plain words. Omit to READ the slot instead of writing it." },
                    "status": { "type": "string", "enum": ["open", "settled"], "description": "'settled' (default) for a stable belief; 'open' for an unfinished loop." },
                    "confidence": { "type": "number", "description": "How sure you are, 0.0 to 1.0. Defaults to 1.0." }
                },
                "required": ["scope", "key"]
            })),
        ),
        Tool::new(
            "memory_history",
            "Show every claim that has occupied a slot, newest first - what was believed, who asserted it, and when it stopped being true. The audit path for a fact whose current value looks wrong or surprising. Superseded claims never appear in ordinary recall, so this is the only way to see one.",
            obj(json!({
                "type": "object",
                "properties": {
                    "scope": { "type": "string", "enum": ["agent", "project", "user"], "description": "The slot's scope, same value you would pass to memory_fact." },
                    "scope_ref": { "type": "string", "description": "What the claim is about. Required for 'project' and 'user'; ignored for 'agent'." },
                    "key": { "type": "string", "description": "The slot name, e.g. \"repo.trunk_model\"." },
                    "limit": { "type": "integer", "description": "Maximum versions to return (default 5, maximum 20)." }
                },
                "required": ["scope", "key"]
            })),
        ),
    ]
}

/// Inject a projected `options` sub-schema into the `file_convert` tool's
/// input schema (ANAI-131). The base [`built_in_tools`] entry ships without an
/// `options` property; the dispatcher supplies the live projection (computed
/// daemon-side from the recipe manifest) so the advertised surface matches
/// what the dispatcher will accept. Clone-on-write: only the
/// `properties.options` slot changes; every other field is preserved.
fn inject_convert_options(mut tool: Tool, options_schema: serde_json::Value) -> Tool {
    let mut schema = (*tool.input_schema).clone();
    if let Some(serde_json::Value::Object(props)) = schema.get_mut("properties") {
        props.insert("options".to_string(), options_schema);
    }
    tool.input_schema = Arc::new(schema);
    tool
}

/// The MCP server handler — wraps a [`ToolDispatcher`] and serves the
/// four-tool ANAI-30 surface over MCP.
///
/// Filtering: `tools/list` advertises only tools that appear in *both*
/// [`built_in_tools`] *and* [`ToolDispatcher::allowed_tools`]. `tools/call`
/// double-checks before dispatch — defense in depth, since the dispatcher
/// itself enforces permissions too.
#[derive(Clone)]
pub struct Bridge {
    dispatcher: Arc<dyn ToolDispatcher>,
}

impl Bridge {
    pub fn new(dispatcher: Arc<dyn ToolDispatcher>) -> Self {
        Self { dispatcher }
    }

    /// Tools the bridge will both advertise and accept calls for, given the
    /// dispatcher's allowed set, plus any daemon-forwarded upstream MCP tools.
    fn permitted_tools(&self) -> Vec<Tool> {
        let allowed = self.dispatcher.allowed_tools();
        let convert_options = self.dispatcher.convert_options_schema();
        let mut tools: Vec<Tool> = built_in_tools()
            .into_iter()
            .filter(|t| allowed.iter().any(|a| a.as_str() == t.name.as_ref()))
            .map(|t| {
                if t.name.as_ref() == "file_convert" {
                    if let Some(opts) = convert_options.clone() {
                        return inject_convert_options(t, opts);
                    }
                }
                t
            })
            .collect();
        // Append upstream MCP tools. NOT gated by `allowed_tools()` —
        // server-side `agent.toml mcp_servers` gating already happened
        // when the daemon answered `ListUpstream`. Name collisions with
        // built-ins are refused at MCP discovery in the daemon, so we
        // trust the names here.
        for def in self.dispatcher.upstream_tools() {
            tools.push(upstream_def_to_tool(def));
        }
        tools
    }

    /// True if `name` is among the advertised upstream MCP tools.
    fn is_advertised_upstream(&self, name: &str) -> bool {
        self.dispatcher
            .upstream_tools()
            .into_iter()
            .any(|t| t.name == name)
    }
}

/// Convert a daemon-forwarded upstream MCP tool definition into the
/// rmcp `Tool` shape advertised on the MCP wire. Description defaults
/// to an empty string when absent; input schema is taken verbatim, or
/// substituted with an empty object if the upstream sent a non-object.
fn upstream_def_to_tool(def: UpstreamToolDef) -> Tool {
    let schema_map = match def.input_schema {
        serde_json::Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    Tool::new(
        def.name,
        def.description.unwrap_or_default(),
        std::sync::Arc::new(schema_map),
    )
}

impl ServerHandler for Bridge {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(
                "OpenFang MCP bridge. Exposes OpenFang's tool surface to MCP clients, \
                 scoped to a single parent agent's identity and capabilities. \
                 Per-agent gating via OPENFANG_BRIDGE_ALLOWED narrows the advertised \
                 set to the calling agent's agent.toml capabilities."
                    .to_string(),
            )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(self.permitted_tools()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let tool_name = request.name.as_ref();
        let args = request
            .arguments
            .map(serde_json::Value::Object)
            .unwrap_or(serde_json::Value::Null);

        // Defense-in-depth: re-check before crossing the seam. The
        // dispatcher will enforce again; that's intentional.
        //
        // Two paths:
        // - Built-in tools: must appear in `allowed_tools()` (the
        //   `OPENFANG_BRIDGE_ALLOWED` / `DEFAULT_ALLOWED` gate).
        // - Upstream `mcp_*` tools: must have been advertised by
        //   the daemon at list-upstream time. The daemon also
        //   enforces `agent.toml mcp_servers` server-side on
        //   dispatch.
        let allowed = self.dispatcher.allowed_tools();
        let is_builtin_allowed = allowed.iter().any(|a| a == tool_name);
        let is_advertised_upstream =
            tool_name.starts_with("mcp_") && self.is_advertised_upstream(tool_name);
        if !is_builtin_allowed && !is_advertised_upstream {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "tool '{tool_name}' not permitted for this agent"
            ))]));
        }

        match self.dispatcher.call(tool_name, args).await {
            Ok(DispatchOk { content, is_error }) => {
                let blocks = vec![Content::text(content)];
                Ok(if is_error {
                    CallToolResult::error(blocks)
                } else {
                    CallToolResult::success(blocks)
                })
            }
            Err(ToolDispatchError::UnknownTool(name)) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "unknown tool: {name}"
                ))]))
            }
            Err(ToolDispatchError::NotPermitted(name)) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "not permitted: {name}"
                ))]))
            }
            Err(ToolDispatchError::InvalidArgs { tool, reason }) => Ok(CallToolResult::error(
                vec![Content::text(format!("invalid args for {tool}: {reason}"))],
            )),
            Err(ToolDispatchError::Execution(e)) => Ok(CallToolResult::error(vec![Content::text(
                format!("tool execution failed: {e}"),
            )])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubDispatcher {
        agent: String,
        allowed: Vec<String>,
        upstream: Vec<UpstreamToolDef>,
        convert_options: Option<serde_json::Value>,
        canned: DispatchOk,
    }

    impl StubDispatcher {
        fn new(agent: &str, allowed: Vec<String>, canned: DispatchOk) -> Self {
            Self {
                agent: agent.into(),
                allowed,
                upstream: Vec::new(),
                convert_options: None,
                canned,
            }
        }
    }

    #[async_trait::async_trait]
    impl ToolDispatcher for StubDispatcher {
        fn agent_id(&self) -> &str {
            &self.agent
        }
        fn allowed_tools(&self) -> Vec<String> {
            self.allowed.clone()
        }
        fn upstream_tools(&self) -> Vec<UpstreamToolDef> {
            self.upstream.clone()
        }
        fn convert_options_schema(&self) -> Option<serde_json::Value> {
            self.convert_options.clone()
        }
        async fn call(
            &self,
            _tool_name: &str,
            _args: serde_json::Value,
        ) -> Result<DispatchOk, ToolDispatchError> {
            Ok(self.canned.clone())
        }
    }

    #[test]
    fn built_in_tools_surface() {
        let tools = built_in_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(
            names,
            vec![
                "file_read",
                "file_list",
                "file_write",
                "create_directory",
                "web_fetch",
                "agent_list",
                "channel_send",
                "agent_send",
                "agent_send_async",
                "agent_reply_async",
                "agent_spawn",
                "agent_kill",
                "agent_activate",
                "memory_store",
                "memory_recall",
                "agent_find",
                "shell_exec",
                "apply_patch",
                "web_search",
                "file_convert",
                "memory_episode_close",
                "memory_status",
                "memory_note",
                "memory_fact",
                "memory_history",
            ],
            "surface drift — update both this test and the runtime tool_runner \
             schema when adding or removing built-in bridge tools"
        );
    }

    /// ANAI-196: the *field-presence* half of this guard is gone. It now lives
    /// in `openfang_api::bridge_ipc::tests::advertised_tool_schemas_match_runtime`
    /// (Invariant C), which asserts property-set equality against the runtime
    /// for every mirrored tool rather than naming one field at a time. Two
    /// escapes (`surface_to`, `timeout_secs`) proved the per-field shape does
    /// not work; do not reintroduce it here.
    ///
    /// What remains is the part Invariant C deliberately cannot express: an
    /// *intentional divergence rule*. `memory_recall` takes exactly one of
    /// `query`/`key`, enforced in the handler, so `required` must stay empty.
    #[test]
    fn bridge_memory_recall_requires_neither_query_nor_key() {
        let tools = built_in_tools();
        let recall = tools
            .iter()
            .find(|t| t.name.as_ref() == "memory_recall")
            .expect("memory_recall must be advertised");

        // Additive-only rule: `key` must never become required again, or the
        // search shape becomes uncallable over the bridge.
        let required = recall
            .input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .expect("memory_recall schema must declare `required`");
        assert!(
            required.is_empty(),
            "memory_recall takes exactly one of `query`/`key`, enforced in the \
             handler; `required` must stay empty because provider support for \
             anyOf/oneOf is uneven and the bridge re-serializes this schema"
        );
    }

    #[test]
    fn file_convert_advertises_injected_options_schema() {
        // ANAI-131: a dispatcher-supplied projection is injected into the
        // advertised file_convert tool as its `options` property.
        let opts = serde_json::json!({
            "type": "object",
            "properties": {
                "orientation": { "type": "string", "enum": ["portrait", "landscape"] }
            },
            "additionalProperties": false
        });
        let stub = StubDispatcher {
            agent: "a".into(),
            allowed: vec!["file_convert".into()],
            upstream: Vec::new(),
            convert_options: Some(opts),
            canned: DispatchOk {
                content: String::new(),
                is_error: false,
            },
        };
        let bridge = Bridge::new(Arc::new(stub));
        let fc = bridge
            .permitted_tools()
            .into_iter()
            .find(|t| t.name.as_ref() == "file_convert")
            .expect("file_convert must be advertised");
        let has_orientation = fc
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
            .and_then(|p| p.get("options"))
            .and_then(|o| o.get("properties"))
            .and_then(|p| p.as_object())
            .is_some_and(|p| p.contains_key("orientation"));
        assert!(
            has_orientation,
            "injected options schema must carry the projected orientation property"
        );
    }

    #[test]
    fn file_convert_without_dispatcher_options_omits_options_property() {
        // Default dispatcher (convert_options_schema() == None) advertises
        // file_convert with no injected options property — still callable.
        let stub = StubDispatcher::new(
            "a",
            vec!["file_convert".into()],
            DispatchOk {
                content: String::new(),
                is_error: false,
            },
        );
        let bridge = Bridge::new(Arc::new(stub));
        let fc = bridge
            .permitted_tools()
            .into_iter()
            .find(|t| t.name.as_ref() == "file_convert")
            .expect("file_convert must be advertised");
        let has_options = fc
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
            .is_some_and(|p| p.contains_key("options"));
        assert!(
            !has_options,
            "no dispatcher options -> no injected options property"
        );
    }

    #[test]
    fn permitted_tools_intersects_with_dispatcher_allowed() {
        let bridge = Bridge::new(Arc::new(StubDispatcher::new(
            "a",
            // Dispatcher permits only file_read of the built-in slice;
            // not_a_real_tool is unknown to the bridge and must be ignored.
            vec!["file_read".into(), "not_a_real_tool".into()],
            DispatchOk {
                content: String::new(),
                is_error: false,
            },
        )));
        let names: Vec<String> = bridge
            .permitted_tools()
            .into_iter()
            .map(|t| t.name.into_owned())
            .collect();
        assert_eq!(names, vec!["file_read".to_string()]);
    }

    #[test]
    fn permitted_tools_appends_upstream_after_builtins() {
        let mut stub = StubDispatcher::new(
            "a",
            vec!["file_read".into()],
            DispatchOk {
                content: String::new(),
                is_error: false,
            },
        );
        stub.upstream = vec![
            UpstreamToolDef {
                name: "mcp_linear_getteams".into(),
                server: "linear".into(),
                description: Some("List teams".into()),
                input_schema: serde_json::json!({"type":"object"}),
            },
            UpstreamToolDef {
                name: "mcp_notion_search".into(),
                server: "notion".into(),
                description: None,
                input_schema: serde_json::json!({}),
            },
        ];
        let bridge = Bridge::new(Arc::new(stub));
        let names: Vec<String> = bridge
            .permitted_tools()
            .into_iter()
            .map(|t| t.name.into_owned())
            .collect();
        // Built-in first, then upstream, preserving order.
        assert_eq!(
            names,
            vec![
                "file_read".to_string(),
                "mcp_linear_getteams".to_string(),
                "mcp_notion_search".to_string(),
            ],
        );
    }

    #[test]
    fn is_advertised_upstream_matches_only_advertised_names() {
        let mut stub = StubDispatcher::new(
            "a",
            vec![],
            DispatchOk {
                content: String::new(),
                is_error: false,
            },
        );
        stub.upstream = vec![UpstreamToolDef {
            name: "mcp_linear_getteams".into(),
            server: "linear".into(),
            description: None,
            input_schema: serde_json::json!({}),
        }];
        let bridge = Bridge::new(Arc::new(stub));
        assert!(bridge.is_advertised_upstream("mcp_linear_getteams"));
        assert!(!bridge.is_advertised_upstream("mcp_linear_unknown"));
        assert!(!bridge.is_advertised_upstream("file_read"));
    }
}
