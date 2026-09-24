//! Daemon-side IPC server for the MCP bridge.
//!
//! ## Topology
//!
//! The bridge runs as a *grandchild* of the daemon:
//!
//! ```text
//! daemon (this process)
//!   └── claude            (CC subprocess, one per prompt)
//!         └── openfang-mcp-bridge   (CC spawns this from --mcp-config)
//!               └── ───── unix socket ─────► daemon (BridgeIpcServer)
//! ```
//!
//! Tools that need [`KernelHandle`](openfang_runtime::kernel_handle::KernelHandle)
//! (e.g. `agent_list`, `channel_send`) cannot run inside the bridge process;
//! it doesn't hold the kernel. The bridge forwards each MCP `tools/call`
//! over a unix-domain socket back here, where we dispatch into
//! `openfang_runtime::tool_runner::execute_tool` and ship the result back.
//!
//! ## Status — ANAI-30 step 2
//!
//! This module currently:
//! - Listens on `<home_dir>/run/bridge.sock`.
//! - Accepts the protocol [`Hello`](openfang_mcp_bridge::protocol::Hello)
//!   handshake (any non-empty token; real auth in ANAI-31).
//! - Decodes [`CallRequest`](openfang_mcp_bridge::protocol::CallRequest)
//!   frames, enforces the four-tool allowlist
//!   ([`ALLOWED_TOOLS`]: `file_read`, `file_list`, `agent_list`,
//!   `channel_send`), and dispatches into
//!   [`openfang_runtime::tool_runner::execute_tool`] with the kernel-bound
//!   context bundle. The shape mirrors the HTTP `/mcp` endpoint in
//!   `routes.rs` so the two execution paths stay in lockstep.
//!
//! Identity (`caller_agent_id`) is currently taken at face value from the
//! [`CallRequest::agent_id`] field. ANAI-31 replaces this with
//! token-derived identity bound at daemon-spawn time. Per-agent
//! capability gating (replacing the static [`ALLOWED_TOOLS`] allowlist
//! with `agent.toml` lookups) lands in the same ticket.

use crate::bridge_auth::BridgeAuthority;
use openfang_kernel::OpenFangKernel;
use openfang_mcp_bridge::protocol::{
    codec, CallRequest, CallResponse, CallResult, Frame, Hello, HelloAck, ListUpstreamRequest,
    UpstreamListResponse, UpstreamListResult, UpstreamToolDef, WireImage, MAX_FRAME_BYTES,
    PROTOCOL_VERSION, SOCKET_RELATIVE_PATH,
};
use openfang_runtime::mcp::{extract_mcp_server_from_known, is_mcp_tool};
use openfang_types::agent::AgentId;
use openfang_types::bridge_auth::Token;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;
use tracing::{debug, error, info, warn};

/// Tools the bridge IPC server is willing to dispatch. Anything outside
/// this set is rejected at the protocol layer (i.e. it never reaches
/// `execute_tool`). This is the daemon-side ceiling on the bridge surface
/// — the bridge subprocess's `built_in_tools` and per-spawn
/// `OPENFANG_BRIDGE_ALLOWED` are the layers that narrow it further per
/// agent (sourced from `agent.toml`).
///
/// Coverage exercises the full diversity of tool dependencies:
/// - `file_read` / `file_list` — workspace-scoped, no kernel needed
/// - `agent_list` — requires [`KernelHandle::list_agents`]
/// - `channel_send` — requires [`KernelHandle::send_channel_message`],
///   one of the OpenFang-only capabilities a CC subprocess wouldn't
///   otherwise have
/// - `agent_send` — inter-agent messaging via the kernel
pub const ALLOWED_TOOLS: &[&str] = &[
    "file_read",
    // ANAI-292: same tier as file_read by construction. A grep over a file
    // exposes a strict subset of what a read already returns, and every
    // candidate path re-enters the same resolver, so granting one and denying
    // the other protects nothing.
    "file_grep",
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
    "memory_store",
    "memory_recall",
    "agent_activate",
    "agent_find",
    "shell_exec",
    "web_search",
    "apply_patch",
    "file_convert",
    "memory_episode_close",
    "memory_status",
    "memory_note",
    // ANAI-204. Tier-3 claim slots: `memory_fact` reads and writes one keyed
    // slot, `memory_history` reads the supersession trail behind it.
    "memory_fact",
    "memory_history",
    // Browser automation, read-only subset. `browser_ctx` was already
    // threaded into `execute_tool` below; these five names are what let a
    // subprocess agent reach it.
    "browser_navigate",
    "browser_read_page",
    "browser_wait",
    "browser_scroll",
    "browser_close",
    // Browser automation, page-driving subset. Same dispatch path; these are
    // classified into `openfang_mcp_bridge::PRIVILEGED_DEFAULT_DENY`, so they
    // reach an agent only through a manifest-derived
    // `OPENFANG_BRIDGE_ALLOWED` grant and never from the no-env-var fallback.
    "browser_click",
    "browser_type",
    "browser_screenshot",
    "browser_run_js",
    "browser_back",
];

/// Subset of [`ALLOWED_TOOLS`] that operates on the agent's workspace
/// filesystem. These tools MUST be invoked with a sandbox-scoping
/// `workspace_root` — see the sandbox check in [`dispatch_call`].
///
/// History: prior to D-fix, the bridge passed `workspace_root: None` to
/// `execute_tool`, which fell through `resolve_file_path`'s "legacy"
/// branch and resolved paths against the daemon CWD (`~/.openfang`). That
/// let any agent with `file_read`/`file_list` advertised on its surface
/// read every sibling workspace plus `secrets.env` and the GCP service-
/// account JSON sitting at the openfang root. The fix below scopes every
/// FS call to the *authenticated* agent's workspace and refuses the call
/// outright when no workspace is registered.
/// `shell_exec` is included here because the command runs with
/// `current_dir(workspace_root)` (tool_runner.rs:1704-1707). Without a
/// registered workspace the shell would default to the daemon CWD
/// (`~/.openfang`), where `secrets.env` and the GCP service-account JSON
/// live — same sibling-leak surface the file tools had pre-D-fix. Refusing
/// the call when no workspace is registered keeps that closed.
/// `apply_patch` is included for the same reason: `tool_apply_patch`
/// resolves every patch-embedded path (Add / Update / Delete) against
/// `workspace_root`. Without a registered workspace, those paths fall
/// through to the daemon CWD and an attacker-crafted patch could touch
/// `secrets.env` or any sibling workspace. Fail-closed gate.
// Canonical FS-sandbox gate lives in `openfang_runtime::tool_runner` so
// the IPC and HTTP `/mcp` surfaces consult one source. Re-exported under
// the original module path to keep existing call sites (and tests at
// :1024,1035 pre-unification) compiling unchanged.
pub use openfang_runtime::tool_runner::FS_SANDBOXED_TOOLS;

/// Daemon-version string sent in [`HelloAck::Ok`].
fn daemon_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Resolve the bridge socket path under `home_dir`. Ensures the parent
/// directory exists.
pub fn socket_path(home_dir: &std::path::Path) -> std::io::Result<PathBuf> {
    let path = home_dir.join(SOCKET_RELATIVE_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(path)
}

/// Handle to a running bridge IPC server. Drop / call [`BridgeIpcServer::shutdown`]
/// to stop accepting connections and remove the socket file.
pub struct BridgeIpcServer {
    socket_path: PathBuf,
    shutdown: Arc<Notify>,
}

impl BridgeIpcServer {
    /// Start the IPC listener. Returns once the socket is bound; the accept
    /// loop runs in a detached tokio task until shutdown is signaled.
    ///
    /// `authority` is the daemon's [`BridgeAuthority`], cloned into each
    /// accepted connection so the handshake can resolve presented tokens to
    /// the [`AgentId`] they were issued for. See [`authenticate_hello`].
    pub async fn start(
        kernel: Arc<OpenFangKernel>,
        authority: Arc<BridgeAuthority>,
    ) -> std::io::Result<Self> {
        let socket_path = socket_path(&kernel.config.home_dir)?;

        // Remove any stale socket from a prior unclean shutdown. UnixListener
        // refuses to bind if the path exists, even if no one's listening.
        if socket_path.exists() {
            warn!(path = %socket_path.display(), "removing stale bridge socket");
            let _ = std::fs::remove_file(&socket_path);
        }

        let listener = UnixListener::bind(&socket_path)?;
        // Restrict to user-only — the socket is loopback to ourselves; no
        // reason for any other uid to connect.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(&socket_path) {
                let mut perms = meta.permissions();
                perms.set_mode(0o600);
                let _ = std::fs::set_permissions(&socket_path, perms);
            }
        }

        info!(path = %socket_path.display(), "bridge IPC server listening");

        let shutdown = Arc::new(Notify::new());
        let accept_shutdown = shutdown.clone();
        let _accept_kernel = kernel.clone();
        let accept_authority = authority.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = accept_shutdown.notified() => {
                        debug!("bridge IPC: accept loop shutting down");
                        break;
                    }
                    res = listener.accept() => {
                        match res {
                            Ok((stream, _addr)) => {
                                info!("bridge IPC: accepted connection");
                                let conn_kernel = _accept_kernel.clone();
                                let conn_authority = accept_authority.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = handle_connection(stream, conn_kernel, conn_authority).await {
                                        debug!(error = %e, "bridge IPC connection ended with error");
                                    }
                                });
                            }
                            Err(e) => {
                                error!(error = %e, "bridge IPC accept failed");
                                // Brief backoff to avoid spinning on a persistent error.
                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            socket_path,
            shutdown,
        })
    }

    /// Path to the unix socket the bridge listens on. Used by the daemon
    /// to publish `OPENFANG_BRIDGE_SOCKET` for subprocess drivers (Claude
    /// Code, etc.) so they can wire CC's `--mcp-config` to point bridges
    /// back here.
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    /// Signal the accept loop to stop and remove the socket file.
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }
}

impl Drop for BridgeIpcServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Resolved identity for a connected bridge after a successful handshake.
///
/// - `agent_id == Some(_)` is the **hardened path**: the bridge presented a
///   well-formed token that the [`BridgeAuthority`] resolved to a live spawn.
///   `dispatch_call` will substitute this id for the (untrusted)
///   `CallRequest::agent_id` field on every subsequent call on the
///   connection.
/// - `agent_id == None` is the **legacy path**: the token is non-empty but
///   not in 64-hex form, so it can't be a daemon-issued token. We accept it
///   for back-compat with non-unix builds and any caller that constructs a
///   kernel via `boot_with_config` (no issuer) — tests, desktop embeds, CLI
///   one-shots. Phase E closed the daemon-side boot ordering loophole, so on
///   a unix daemon every spawn site now sees an issuer and well-formed hex
///   tokens are the norm. In this legacy mode the bridge's claimed `agent_id`
///   is taken at face value — same trust model as ANAI-30. A future strict
///   mode can reject this arm outright once all supported deployments are on
///   the hardened path.
///
/// `token_fingerprint` is the first 32 bits of the resolved token, suitable
/// for log correlation. `None` on the legacy path.
#[derive(Debug)]
struct HandshakeIdentity {
    agent_id: Option<AgentId>,
    token_fingerprint: Option<String>,
}

/// Placeholder rendered in place of an agent name we could not resolve.
///
/// Attribution only — never feeds authorization.
const UNKNOWN_AGENT_NAME: &str = "<unknown>";

/// Resolve an agent's manifest name for log attribution.
///
/// Returns [`UNKNOWN_AGENT_NAME`] when the id has no registry entry (dead
/// spawn, spoofed id). Names resolved here are *display only*: the registry
/// entry stays the source of truth for capabilities, and a name resolved from
/// an unauthenticated (legacy-lane) id is rendered `<claimed:...>` by the
/// caller so a spoofable name never reads as an attributed one.
///
/// Emitted as its own `agent_name=` field rather than a parenthetical inside
/// `agent=`: a space inside a `tracing` field value terminates the value for
/// every `key=value` reader, including grep.
fn agent_name_for_log(kernel: &Arc<OpenFangKernel>, id: AgentId) -> String {
    kernel
        .registry
        .get(id)
        .map(|e| e.manifest.name.clone())
        .unwrap_or_else(|| UNKNOWN_AGENT_NAME.to_string())
}

/// Handle a single bridge connection: Hello/HelloAck handshake, then a loop
/// of CallRequest → CallResponse frames until the peer closes.
async fn handle_connection(
    mut stream: UnixStream,
    kernel: Arc<OpenFangKernel>,
    authority: Arc<BridgeAuthority>,
) -> std::io::Result<()> {
    let (read_half, mut write_half) = stream.split();
    let mut read_half = tokio::io::BufReader::new(read_half);

    // --- Handshake ---
    let hello = match codec::read_frame(&mut read_half).await? {
        Frame::Hello(h) => h,
        other => {
            warn!(?other, "bridge IPC: first frame was not Hello, closing");
            return Ok(());
        }
    };

    let identity = match authenticate_hello(&hello, &authority) {
        Ok(id) => id,
        Err(reason) => {
            let ack = Frame::HelloAck(HelloAck::Rejected {
                reason: reason.clone(),
            });
            let _ = codec::write_frame(&mut write_half, &ack).await;
            warn!(reason, "bridge IPC: rejected handshake");
            return Ok(());
        }
    };

    let ack = Frame::HelloAck(HelloAck::Ok {
        daemon_version: daemon_version(),
        // ANAI-131: hand the projected file_convert `options` schema to the
        // runtime-free bridge so its tools/list advertises the same option
        // surface the dispatcher accepts, without importing the runtime.
        convert_options_schema: Some(openfang_runtime::convert::file_convert_options_schema()),
    });
    codec::write_frame(&mut write_half, &ack).await?;
    let authed_display = identity
        .agent_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "<legacy-unauthenticated>".to_string());
    let fingerprint_display = identity
        .token_fingerprint
        .clone()
        .unwrap_or_else(|| "<legacy>".to_string());
    // The authenticated identity is fixed for the life of the socket, so
    // resolve its name once here rather than per request. The legacy lane has
    // no authenticated identity; its name is derived per call from the
    // self-claimed id and marked `<claimed:...>`.
    let authed_name: Option<String> = identity.agent_id.map(|id| agent_name_for_log(&kernel, id));
    let authed_name_display = authed_name
        .clone()
        .unwrap_or_else(|| "<legacy-unauthenticated>".to_string());
    info!(
        bridge_version = %hello.bridge_version,
        token_fingerprint = %fingerprint_display,
        authenticated_agent = %authed_display,
        authenticated_agent_name = %authed_name_display,
        "bridge IPC: handshake complete"
    );

    // --- Request loop ---
    loop {
        let frame = match codec::read_frame(&mut read_half).await {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                debug!("bridge IPC: peer closed");
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        match frame {
            Frame::Call(call) => {
                let call_agent_name = match &authed_name {
                    Some(name) => name.clone(),
                    None => match call.agent_id.parse::<AgentId>() {
                        Ok(claimed) => {
                            format!("<claimed:{}>", agent_name_for_log(&kernel, claimed))
                        }
                        Err(_) => UNKNOWN_AGENT_NAME.to_string(),
                    },
                };
                info!(
                    request_id = call.request_id,
                    tool = %call.tool_name,
                    agent = %call.agent_id,
                    agent_name = %call_agent_name,
                    "bridge IPC: dispatching call"
                );

                let result = dispatch_call(&call, &kernel, identity.agent_id.as_ref()).await;
                // Frame floor: every result, native or upstream, passes
                // through here. Oversized frames are refused by the codec and
                // the refusal closes the connection, so clamping must happen
                // before the frame is built — not inside any one tool.
                let (result, clamped_from) = clamp_result_to_frame(result);
                if let Some(original_bytes) = clamped_from {
                    warn!(
                        request_id = call.request_id,
                        tool = %call.tool_name,
                        agent = %call.agent_id,
                        agent_name = %call_agent_name,
                        original_bytes,
                        budget = FRAME_PAYLOAD_BUDGET,
                        "bridge IPC: truncated oversized tool result to fit the frame"
                    );
                }
                let result_kind = match &result {
                    CallResult::Ok {
                        is_error: false, ..
                    } => "ok",
                    CallResult::Ok { is_error: true, .. } => "tool_error",
                    CallResult::Rich {
                        is_error: false, ..
                    } => "ok",
                    CallResult::Rich { is_error: true, .. } => "tool_error",
                    CallResult::Error { .. } => "dispatch_error",
                };
                info!(
                    request_id = call.request_id,
                    tool = %call.tool_name,
                    agent = %call.agent_id,
                    agent_name = %call_agent_name,
                    outcome = result_kind,
                    "bridge IPC: call complete"
                );
                let response = Frame::Response(CallResponse {
                    request_id: call.request_id,
                    result,
                });
                codec::write_frame(&mut write_half, &response).await?;
            }
            Frame::ListUpstream(req) => {
                info!(
                    request_id = req.request_id,
                    agent = %authed_display,
                    agent_name = %authed_name_display,
                    "bridge IPC: dispatching list_upstream"
                );
                let response =
                    handle_list_upstream(&req, &kernel, identity.agent_id.as_ref()).await;
                let outcome = match &response.result {
                    UpstreamListResult::Ok { tools } => {
                        format!("ok({} tools)", tools.len())
                    }
                    UpstreamListResult::Error { .. } => "error".to_string(),
                };
                info!(
                    request_id = response.request_id,
                    agent = %authed_display,
                    agent_name = %authed_name_display,
                    outcome = %outcome,
                    "bridge IPC: list_upstream complete"
                );
                codec::write_frame(&mut write_half, &Frame::UpstreamList(response)).await?;
            }
            other => {
                warn!(?other, "bridge IPC: unexpected frame in request loop");
                continue;
            }
        }
    }
}

/// Headroom reserved inside [`MAX_FRAME_BYTES`] for everything in a response
/// frame that is *not* the result payload: the JSON envelope (`request_id`,
/// result tag, `is_error`), plus room for the truncation marker itself.
/// Deliberately generous — the cost of over-reserving is a slightly shorter
/// result, and the cost of under-reserving is a closed connection.
const FRAME_ENVELOPE_RESERVE: usize = 16 * 1024;

/// Payload budget for a single response frame, measured in JSON-*escaped*
/// bytes (see [`json_escaped_cost`]).
const FRAME_PAYLOAD_BUDGET: usize = MAX_FRAME_BYTES.saturating_sub(FRAME_ENVELOPE_RESERVE);

/// Bytes a single `char` occupies inside a serialized JSON string literal.
///
/// Mirrors `serde_json`'s escaping exactly: the seven short escapes cost two
/// bytes, any other C0 control byte becomes a six-byte `\u00XX`, and
/// everything else — including all non-ASCII — is emitted verbatim at its
/// UTF-8 length. Measuring the *escaped* cost is the point: a megabyte of
/// quotes doubles on the wire, so a budget applied to the raw string would
/// still overflow the frame.
fn json_escaped_cost(ch: char) -> usize {
    match ch {
        '"' | '\\' | '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}' => 2,
        c if (c as u32) < 0x20 => 6,
        c => c.len_utf8(),
    }
}

/// Longest prefix of `s` whose JSON-escaped form fits in `budget` bytes.
///
/// Cuts on a `char` boundary by construction (it walks `char_indices`), so the
/// result is always valid UTF-8 and never splits a multi-byte codepoint.
fn truncate_to_json_budget(s: &str, budget: usize) -> &str {
    let mut cost = 0usize;
    for (idx, ch) in s.char_indices() {
        let next = cost + json_escaped_cost(ch);
        if next > budget {
            return &s[..idx];
        }
        cost = next;
    }
    s
}

/// Clamp a dispatch result so its response frame cannot exceed
/// [`MAX_FRAME_BYTES`].
///
/// ## Why this exists
///
/// `codec::write_frame` *refuses* an oversized frame, and the request loop
/// propagates that error — which closes the connection. So before this, a
/// single tool returning more than a mebibyte did not merely fail that call:
/// it tore down the bridge for the rest of the turn, with no marker and no
/// diagnosis the model could act on. `file_read` has no cap of its own (it is
/// `read_to_string`), so one large file was enough.
///
/// The upstream-MCP path already truncated its own results at this boundary,
/// which is precisely why the hole was easy to miss: half the traffic was
/// protected and the protocol docs claimed *all* of it was. This moves the
/// clamp to the one place every result passes through, and the upstream path
/// now inherits it instead of hand-rolling it.
///
/// ## Semantics
///
/// Truncation sets `is_error = true` and appends a marker naming the original
/// size, the budget, and the remedy. Flagging it is deliberate: a truncated
/// result is not a complete answer, and the failure mode we are buying our way
/// out of all month is the one where incomplete output reads as finished. A
/// marker at the tail can be skimmed past; an error flag cannot.
///
/// Returns the (possibly clamped) result and the original payload size when it
/// was clamped, so the caller can log it with full request context.
fn clamp_result_to_frame(result: CallResult) -> (CallResult, Option<usize>) {
    match result {
        CallResult::Ok { content, is_error } => {
            let kept = truncate_to_json_budget(&content, FRAME_PAYLOAD_BUDGET);
            if kept.len() == content.len() {
                return (CallResult::Ok { content, is_error }, None);
            }
            let original = content.len();
            let mut clamped = kept.to_string();
            clamped.push_str(&format!(
                "\n\n[openfang: tool result truncated — {original} bytes exceeds the \
                 {MAX_FRAME_BYTES}-byte bridge frame limit. {kept_len} bytes are shown \
                 above; the rest was NOT returned. Re-request a bounded slice of this \
                 data rather than retrying the same call, which will truncate \
                 identically.]",
                kept_len = kept.len()
            ));
            (
                CallResult::Ok {
                    content: clamped,
                    is_error: true,
                },
                Some(original),
            )
        }
        CallResult::Error { message } => {
            let kept = truncate_to_json_budget(&message, FRAME_PAYLOAD_BUDGET);
            if kept.len() == message.len() {
                return (CallResult::Error { message }, None);
            }
            let original = message.len();
            let mut clamped = kept.to_string();
            clamped.push_str("\n\n[openfang: error message truncated to fit the bridge frame]");
            (CallResult::Error { message: clamped }, Some(original))
        }
        CallResult::Rich {
            content,
            is_error,
            images,
        } => clamp_rich_result(content, is_error, images),
    }
}

/// Per-image JSON overhead on top of the base64 itself: the object braces,
/// both key names, quotes and separators. Base64 contains no characters JSON
/// escapes, so the data costs exactly its length.
const WIRE_IMAGE_OVERHEAD: usize = 64;

/// Frame clamp for [`CallResult::Rich`] (ANAI-297). Images are all-or-nothing.
///
/// Text can be truncated with a marker and still mean something; a base64
/// payload cut short decodes to a corrupt image that the model receives as a
/// successful read. So when the whole result will not fit, every image is
/// dropped and the result becomes an error that says so and why — never a
/// partial image, and never a text-only success that silently lost its
/// pixels.
fn clamp_rich_result(
    content: String,
    is_error: bool,
    images: Vec<WireImage>,
) -> (CallResult, Option<usize>) {
    let text_cost: usize = content.chars().map(json_escaped_cost).sum();
    let image_cost: usize = images
        .iter()
        .map(|i| i.data_base64.len() + i.mime_type.len() + WIRE_IMAGE_OVERHEAD)
        .sum();
    let total = text_cost + image_cost;
    if total <= FRAME_PAYLOAD_BUDGET {
        return (
            CallResult::Rich {
                content,
                is_error,
                images,
            },
            None,
        );
    }
    let kept = truncate_to_json_budget(&content, FRAME_PAYLOAD_BUDGET / 2);
    let message = format!(
        "{kept}\n\n[openfang: {n} image(s) NOT returned — the result is {total} bytes \
         encoded, over the {budget}-byte bridge frame budget. Images are never sent \
         partially, because a truncated image decodes as corrupt yet reads as success. \
         Nothing above this line is a view of the image.]",
        n = images.len(),
        budget = FRAME_PAYLOAD_BUDGET,
    );
    (
        CallResult::Ok {
            content: message,
            is_error: true,
        },
        Some(total),
    )
}

/// Dispatch a single bridge tool call to the runtime.
///
/// Enforces the [`ALLOWED_TOOLS`] allowlist before invoking
/// [`openfang_runtime::tool_runner::execute_tool`]. The argument bundle
/// mirrors the HTTP `/mcp` endpoint in `routes.rs` — keep them in sync;
/// they share semantics intentionally.
///
/// Returns:
/// - [`CallResult::Error`] for protocol-layer rejections (unknown tool,
///   not on the allowlist).
/// - [`CallResult::Ok`] for anything `execute_tool` returned, with
///   `is_error` propagated. A tool that ran but returned an error to
///   the model is `Ok { is_error: true }`, **not** `Error` — the latter
///   means the bridge couldn't even attempt dispatch.
async fn dispatch_call(
    call: &CallRequest,
    kernel: &Arc<OpenFangKernel>,
    authenticated_agent_id: Option<&AgentId>,
) -> CallResult {
    // --- Early branch: upstream MCP tool (mcp_*) ---------------------------
    // Upstream MCP tools are not in `ALLOWED_TOOLS` and have a separate
    // dispatch path: per-agent server allowlist (agent.toml `mcp_servers`,
    // default-deny on empty), hardened-token lane only, and direct dispatch
    // into `kernel.mcp_connections` rather than `execute_tool`. See
    // `dispatch_upstream_mcp_call` for the gates.
    if is_mcp_tool(&call.tool_name) {
        return dispatch_upstream_mcp_call(call, kernel, authenticated_agent_id).await;
    }

    // --- Gate 1: static bridge-surface allowlist ----------------------------
    // The hard ceiling on what the bridge will ever dispatch. Independent of
    // any agent's per-agent surface; an unknown tool never reaches identity
    // resolution.
    if !ALLOWED_TOOLS.iter().any(|t| *t == call.tool_name) {
        return CallResult::Error {
            message: format!(
                "tool '{}' not in bridge allowlist (permitted: {:?})",
                call.tool_name, ALLOWED_TOOLS
            ),
        };
    }

    // --- Identity resolution (fail-closed) ---------------------------------
    // Hardened path: handshake-bound AgentId from BridgeAuthority. Legacy
    // path: parse the bridge's self-claimed `call.agent_id`. Either way we
    // require a *registered* AgentId before proceeding — string identifiers
    // and unknown agents never feed authorization. Closes the ANAI-30
    // "trust the claimed string" loophole: a parseable but unregistered id
    // (random UUID-shaped value) now rejects instead of falling through.
    let resolved_agent_id: AgentId = match authenticated_agent_id {
        Some(authed) => {
            let authed_str = authed.to_string();
            if authed_str != call.agent_id {
                warn!(
                    request_id = call.request_id,
                    tool = %call.tool_name,
                    claimed = %call.agent_id,
                    authenticated = %authed_str,
                    "bridge IPC: claimed agent_id disagrees with authenticated identity; \
                     using authenticated identity"
                );
            }
            *authed
        }
        None => match call.agent_id.parse::<AgentId>() {
            Ok(aid) => aid,
            Err(_) => {
                warn!(
                    request_id = call.request_id,
                    tool = %call.tool_name,
                    claimed = %call.agent_id,
                    "bridge IPC: rejecting call — legacy lane and claimed agent_id \
                     does not parse as AgentId"
                );
                return CallResult::Error {
                    message: "unresolvable agent identity for bridge call".to_string(),
                };
            }
        },
    };
    let resolved_agent_id_string = resolved_agent_id.to_string();

    // Registry entry is the source of truth for capabilities and workspace.
    // Missing entry → fail closed (spoofed AgentId, dead spawn, etc.).
    let entry = match kernel.registry.get(resolved_agent_id) {
        Some(e) => e,
        None => {
            warn!(
                request_id = call.request_id,
                tool = %call.tool_name,
                agent = %resolved_agent_id_string,
                agent_name = %UNKNOWN_AGENT_NAME,
                "bridge IPC: rejecting call — no registry entry for resolved agent"
            );
            return CallResult::Error {
                message: format!(
                    "agent '{resolved_agent_id_string}' has no registry entry; refusing call"
                ),
            };
        }
    };

    // Attribution only. `<claimed:...>` marks a name resolved from a
    // self-claimed (legacy-lane) id, which is spoofable - the same reason the
    // legacy lane's claimed *id* never feeds authorization.
    let agent_name_log = if authenticated_agent_id.is_some() {
        entry.manifest.name.clone()
    } else {
        format!("<claimed:{}>", entry.manifest.name)
    };

    // --- Workspace-aware skill snapshot ------------------------------------
    // Mirrors the agent_loop pattern (kernel.rs:2192-2210): bundled + global
    // + workspace skills, in that override order. We reuse this snapshot for
    // BOTH the per-agent permission gate below AND `execute_tool` later — so
    // the permission decision and the runtime see the same tool universe.
    let workspace_path: Option<PathBuf> = entry.manifest.workspace.clone();
    let skill_snapshot = {
        let mut snapshot = kernel
            .skill_registry
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot();
        if let Some(ref workspace) = workspace_path {
            let ws_skills = workspace.join("skills");
            if ws_skills.exists() {
                if let Err(e) = snapshot.load_workspace_skills(&ws_skills) {
                    warn!(
                        agent = %resolved_agent_id_string,
                        agent_name = %agent_name_log,
                        error = %e,
                        "bridge IPC: failed to load workspace skills for permission gate"
                    );
                }
            }
        }
        snapshot
    };

    // --- Gate 2: per-agent execute-time permission gate (ANAI C) -----------
    // Belt-and-suspenders with the advertise-time `OPENFANG_BRIDGE_ALLOWED`
    // env var the bridge subprocess was spawned with. Uses the same kernel
    // resolver agent_loop uses to build the env (kernel.rs:2214) against
    // the same registry entry — so the two gates can't drift. Any tool not
    // in this agent's resolved surface (capabilities.tools narrowed by
    // profile, allowlist, blocklist, skills, mcp_servers, mode filter) is
    // rejected here with a logged trace, even if it survived gate 1.
    //
    // Runs *before* the workspace sandbox gate so a denied call never
    // touches the filesystem lookup.
    let permitted: Vec<openfang_types::tool::ToolDefinition> = {
        let resolved =
            kernel.available_tools_with_registry(resolved_agent_id, Some(&skill_snapshot));
        entry.mode.filter_tools(resolved)
    };
    if !permitted.iter().any(|t| t.name == call.tool_name) {
        warn!(
            request_id = call.request_id,
            tool = %call.tool_name,
            agent = %resolved_agent_id_string,
            agent_name = %agent_name_log,
            mode = ?entry.mode,
            permitted_count = permitted.len(),
            "bridge IPC: rejecting tool not in agent's permitted set"
        );
        return CallResult::Error {
            message: format!("tool '{}' not permitted for this agent", call.tool_name),
        };
    }

    // Build the kernel handle. Cloning the Arc is cheap; the cast to
    // `dyn KernelHandle` is the same upcast the HTTP /mcp endpoint
    // performs.
    let kernel_handle: Arc<dyn openfang_runtime::kernel_handle::KernelHandle> =
        kernel.clone() as Arc<dyn openfang_runtime::kernel_handle::KernelHandle>;

    // execute_tool also enforces an allowlist via its `allowed_tools`
    // parameter; passing our four-tool set makes the runtime's check
    // belt-and-suspenders with ours. If the two ever drift, the runtime's
    // is authoritative — it sits closer to the actual tool implementations.
    let allowed_tools_owned: Vec<String> = ALLOWED_TOOLS.iter().map(|s| (*s).to_string()).collect();

    // --- Gate 3: workspace sandbox for filesystem tools (D-fix) ------------
    // Fail-closed for filesystem tools without a workspace. The runtime's
    // `resolve_file_path` falls through to a path-traversal-only check
    // when `workspace_root` is `None`, which is insufficient — absolute
    // paths bypass it and relative paths resolve against the daemon CWD.
    // Refusing the call here keeps the leak closed even if some future
    // call path forgets to pass workspace_root.
    if FS_SANDBOXED_TOOLS.contains(&call.tool_name.as_str()) && workspace_path.is_none() {
        warn!(
            request_id = call.request_id,
            tool = %call.tool_name,
            agent = %resolved_agent_id_string,
            agent_name = %agent_name_log,
            "bridge IPC: refusing filesystem tool — no workspace registered for agent"
        );
        return CallResult::Error {
            message: format!(
                "tool '{}' requires an agent workspace, but no workspace is registered \
                 for agent '{}' — refusing to fall back to an unscoped filesystem view",
                call.tool_name, resolved_agent_id_string
            ),
        };
    }
    let workspace_root_arg: Option<&Path> = workspace_path.as_deref();

    // shell_exec needs both `exec_policy` (allowlist / full / deny decision)
    // and `allowed_env_vars` (hand-granted env passthrough). Resolution is
    // shared with the HTTP `/mcp` path via `AgentExecContext` so the two
    // surfaces apply identical scoping — see S3-01 in the bridge-v2 audit.
    // Every other bridge tool ignores both, so this is cheap to compute
    // unconditionally.
    let exec_ctx =
        openfang_runtime::agent_tool_context::AgentExecContext::from_manifest(&entry.manifest);
    let effective_exec_policy = exec_ctx.exec_policy_ref();
    let allowed_env_arg: Option<&[String]> = exec_ctx.allowed_env();

    // Piece 3 (ANAI-82): the bridge IPC tool call runs on a separate task from
    // the agent's run loop, so origin isn't on this stack. Resolve it from the
    // kernel's per-run stash, keyed by the already-authenticated agent id.
    let bridge_origin = kernel
        .active_run_origins
        .get(&resolved_agent_id)
        .map(|r| r.clone());

    let result = openfang_runtime::tool_runner::execute_tool(
        &format!("bridge-{}", call.request_id),
        &call.tool_name,
        &call.args,
        Some(&kernel_handle),
        Some(&allowed_tools_owned),
        Some(resolved_agent_id_string.as_str()),
        Some(&skill_snapshot),
        Some(&kernel.mcp_connections),
        Some(&kernel.web_ctx),
        Some(&kernel.browser_ctx),
        allowed_env_arg,
        workspace_root_arg, // scoped to the authenticated agent's workspace; gated above
        Some(&kernel.media_engine),
        effective_exec_policy,
        entry.manifest.file_policy.as_ref(), // F6: agent's resolved policy (was None — silent tier downgrade on bridge)
        if kernel.config.tts.enabled {
            Some(&kernel.tts_engine)
        } else {
            None
        },
        if kernel.config.docker.enabled {
            Some(&kernel.config.docker)
        } else {
            None
        },
        Some(&*kernel.process_manager),
        // Piece 3 (ANAI-82): in-flight run's origin (targeting/audit only;
        // authz already enforced off the authenticated agent id above).
        bridge_origin.as_ref(),
    )
    .await;

    CallResult::Ok {
        content: result.content,
        is_error: result.is_error,
    }
}

/// Dispatch an upstream MCP tool call (`mcp_{server}_{tool}`) over the
/// kernel's `mcp_connections` registry.
///
/// Distinct from [`dispatch_call`]'s built-in tool path:
///
/// - **Hardened-token lane only.** Refuses the legacy
///   self-claimed-`agent_id` lane. Upstream MCP servers can carry
///   secrets (OAuth tokens, page contents, Linear issue bodies) and
///   the legacy lane has no daemon-issued identity to authorize
///   against — fail closed. The hardened lane is the only entry to
///   this surface for v1.
///
/// - **Per-agent server allowlist with default-deny.** Reads
///   `entry.manifest.mcp_servers` from the registry and rejects any
///   tool whose server prefix is not on the list. **Empty list means
///   no servers allowed** for the bridge path — this is a deliberate
///   semantic departure from the in-process MCP path
///   (`tool_runner.rs`), which historically treated `[]` as
///   "all servers". v1 ships the new semantic for the bridge only;
///   convergence is tracked as follow-up. See design doc §5.4.
///
/// - **Direct dispatch into `kernel.mcp_connections`.** Bypasses
///   `execute_tool` entirely; the runtime's MCP routing already does
///   what we need, and routing through `execute_tool` would require
///   adding every namespaced tool to its allowlist parameter.
///
/// Truncation: results larger than the response frame budget are
/// returned with `is_error=true` and an explicit truncation marker
/// rather than silently dropped. Per design doc §6 mitigation table.
async fn dispatch_upstream_mcp_call(
    call: &CallRequest,
    kernel: &Arc<OpenFangKernel>,
    authenticated_agent_id: Option<&AgentId>,
) -> CallResult {
    // Refuse the legacy lane outright. Upstream MCP forwarding is
    // hardened-path-only in v1.
    let authed = match authenticated_agent_id {
        Some(a) => a,
        None => {
            warn!(
                request_id = call.request_id,
                tool = %call.tool_name,
                claimed = %call.agent_id,
                "bridge IPC: refusing upstream MCP call on legacy token lane"
            );
            return CallResult::Error {
                message: "upstream MCP tools require a daemon-issued (hex) auth token;                           legacy lane refused"
                    .to_string(),
            };
        }
    };

    let resolved_agent_id = *authed;
    let resolved_agent_id_string = resolved_agent_id.to_string();

    // Log a mismatch but trust the authenticated identity, consistent
    // with the built-in path in `dispatch_call`.
    if resolved_agent_id_string != call.agent_id {
        warn!(
            request_id = call.request_id,
            tool = %call.tool_name,
            claimed = %call.agent_id,
            authenticated = %resolved_agent_id_string,
            "bridge IPC (upstream MCP): claimed agent_id disagrees with authenticated identity;              using authenticated identity"
        );
    }

    let entry = match kernel.registry.get(resolved_agent_id) {
        Some(e) => e,
        None => {
            warn!(
                request_id = call.request_id,
                tool = %call.tool_name,
                agent = %resolved_agent_id_string,
                agent_name = %UNKNOWN_AGENT_NAME,
                "bridge IPC (upstream MCP): no registry entry for resolved agent"
            );
            return CallResult::Error {
                message: format!(
                    "agent '{resolved_agent_id_string}' has no registry entry;                      refusing upstream MCP call"
                ),
            };
        }
    };

    // Hardened lane only (asserted above), so the name is authenticated.
    let agent_name_log = entry.manifest.name.clone();

    // Default-deny: empty `mcp_servers` → no upstream tools allowed.
    // This is the bridge-path semantic; the in-process path's
    // `[] = all` convention is left undisturbed for now (see design
    // doc §5.4).
    if entry.manifest.mcp_servers.is_empty() {
        warn!(
            request_id = call.request_id,
            tool = %call.tool_name,
            agent = %resolved_agent_id_string,
            agent_name = %agent_name_log,
            "bridge IPC (upstream MCP): agent has no mcp_servers allowlist; refusing"
        );
        return CallResult::Error {
            message: format!(
                "agent '{resolved_agent_id_string}' has no MCP servers allowlisted                  (set `mcp_servers` in agent.toml)"
            ),
        };
    }

    // Match the tool's server prefix against the allowlist. Use the
    // `from_known` helper so server names containing hyphens
    // (normalized to underscores in tool names) match correctly.
    let allowlist: Vec<&str> = entry
        .manifest
        .mcp_servers
        .iter()
        .map(|s| s.as_str())
        .collect();
    let server_name = match extract_mcp_server_from_known(&call.tool_name, &allowlist) {
        Some(name) => name.to_string(),
        None => {
            warn!(
                request_id = call.request_id,
                tool = %call.tool_name,
                agent = %resolved_agent_id_string,
                agent_name = %agent_name_log,
                allowlist = ?entry.manifest.mcp_servers,
                "bridge IPC (upstream MCP): tool's server prefix not in agent allowlist"
            );
            return CallResult::Error {
                message: format!(
                    "upstream MCP tool '{}' is not allowlisted for agent '{}'                      (allowed servers: {:?})",
                    call.tool_name, resolved_agent_id_string, entry.manifest.mcp_servers
                ),
            };
        }
    };

    // Dispatch into the kernel's MCP registry. Hold the lock just
    // long enough to issue the call; rmcp's `call_tool` awaits a
    // network round-trip, so this *does* serialize concurrent calls
    // against the same connection set. That matches the in-process
    // path in `tool_runner.rs` and is acceptable for v1 — Linear /
    // Notion calls are not hot-path. Revisit when latency complaints
    // arrive.
    let result_text = {
        let mut conns = kernel.mcp_connections.lock().await;
        let conn = conns.iter_mut().find(|c| c.name() == server_name);
        let conn = match conn {
            Some(c) => c,
            None => {
                warn!(
                    request_id = call.request_id,
                    tool = %call.tool_name,
                    agent = %resolved_agent_id_string,
                    agent_name = %agent_name_log,
                    server = %server_name,
                    "bridge IPC (upstream MCP): server allowlisted but not connected"
                );
                return CallResult::Error {
                    message: format!(
                        "MCP server '{server_name}' is allowlisted for agent                          '{resolved_agent_id_string}' but not currently connected"
                    ),
                };
            }
        };
        conn.call_tool(&call.tool_name, &call.args).await
    };

    match result_text {
        Ok(content) => {
            // No truncation here: `clamp_result_to_frame` in the request loop
            // is the single frame floor for every result, native or upstream.
            // The bespoke copy that used to live at this spot measured the RAW
            // byte length and then took `budget / 4` chars to stay safe — which
            // both over-trimmed ASCII by ~4x and, more importantly, left the
            // native tool path with no clamp at all.
            CallResult::Ok {
                content,
                is_error: false,
            }
        }
        Err(e) => {
            // Distinguish runtime tool errors from dispatch failures
            // the same way the built-in path does: a tool that ran
            // and reported error is `Ok { is_error: true }`; we
            // can't easily tell the difference from rmcp's surface
            // here, so we surface as `Ok { is_error: true }` to let
            // the model see the message rather than killing the
            // call frame.
            warn!(
                request_id = call.request_id,
                tool = %call.tool_name,
                agent = %resolved_agent_id_string,
                agent_name = %agent_name_log,
                error = %e,
                "bridge IPC (upstream MCP): tool call failed"
            );
            CallResult::Ok {
                content: format!("upstream MCP tool call failed: {e}"),
                is_error: true,
            }
        }
    }
}

/// Handle a `ListUpstream` request: enumerate the upstream MCP tools the
/// authenticated agent is allowed to invoke.
///
/// Same security model as [`dispatch_upstream_mcp_call`]:
/// - Hardened-token lane only.
/// - Per-agent `mcp_servers` allowlist with default-deny.
/// - Empty allowlist → empty tool list (not an error; the bridge will
///   simply advertise no upstream tools to Claude Code).
async fn handle_list_upstream(
    req: &ListUpstreamRequest,
    kernel: &Arc<OpenFangKernel>,
    authenticated_agent_id: Option<&AgentId>,
) -> UpstreamListResponse {
    let authed = match authenticated_agent_id {
        Some(a) => a,
        None => {
            warn!(
                request_id = req.request_id,
                "bridge IPC: refusing list_upstream on legacy token lane"
            );
            return UpstreamListResponse {
                request_id: req.request_id,
                result: UpstreamListResult::Error {
                    message: "upstream MCP listing requires a daemon-issued (hex) auth token;                               legacy lane refused"
                        .to_string(),
                },
            };
        }
    };

    let resolved_agent_id = *authed;
    let resolved_agent_id_string = resolved_agent_id.to_string();

    let entry = match kernel.registry.get(resolved_agent_id) {
        Some(e) => e,
        None => {
            warn!(
                request_id = req.request_id,
                agent = %resolved_agent_id_string,
                agent_name = %UNKNOWN_AGENT_NAME,
                "bridge IPC: list_upstream — no registry entry for resolved agent"
            );
            return UpstreamListResponse {
                request_id: req.request_id,
                result: UpstreamListResult::Error {
                    message: format!("agent '{resolved_agent_id_string}' has no registry entry"),
                },
            };
        }
    };

    // Empty allowlist → empty list (advertise nothing). This is the
    // natural representation of `mcp_servers = []` with the new
    // default-deny semantic.
    if entry.manifest.mcp_servers.is_empty() {
        return UpstreamListResponse {
            request_id: req.request_id,
            result: UpstreamListResult::Ok { tools: Vec::new() },
        };
    }

    let allowlist: std::collections::HashSet<&str> = entry
        .manifest
        .mcp_servers
        .iter()
        .map(|s| s.as_str())
        .collect();

    let conns = kernel.mcp_connections.lock().await;
    let mut tools: Vec<UpstreamToolDef> = Vec::new();
    for conn in conns.iter() {
        let server_name = conn.name();
        if !allowlist.contains(server_name) {
            continue;
        }
        for tool in conn.tools() {
            tools.push(UpstreamToolDef {
                name: tool.name.clone(),
                server: server_name.to_string(),
                description: if tool.description.is_empty() {
                    None
                } else {
                    Some(tool.description.clone())
                },
                input_schema: tool.input_schema.clone(),
            });
        }
    }

    UpstreamListResponse {
        request_id: req.request_id,
        result: UpstreamListResult::Ok { tools },
    }
}

/// Authenticate the bridge's Hello against the daemon's [`BridgeAuthority`].
///
/// Decision tree:
/// - Version mismatch → `Err` (existing rejection).
/// - Empty/whitespace token → `Err` (existing rejection).
/// - Token parses as 64-hex (`Token::from_hex`) → resolve via authority:
///   - `Some(agent_id)` → `Ok(HandshakeIdentity { agent_id: Some(_), ... })`
///     — hardened path; this is a daemon-issued, live token.
///   - `None` → `Err` — well-formed hex that the authority never issued.
///     This is the attacker / replay / stale-token rejection.
/// - Token is non-empty but not 64-hex → `Ok(HandshakeIdentity { agent_id:
///   None, .. })` — legacy back-compat lane. Logs `debug!` so the operator
///   can see how many legacy handshakes are still happening.
fn authenticate_hello(
    hello: &Hello,
    authority: &BridgeAuthority,
) -> Result<HandshakeIdentity, String> {
    if hello.protocol_version != PROTOCOL_VERSION {
        return Err(format!(
            "protocol version mismatch: bridge={} daemon={}",
            hello.protocol_version, PROTOCOL_VERSION
        ));
    }
    let presented = hello.token.trim();
    if presented.is_empty() {
        return Err("empty auth token".to_string());
    }

    match Token::from_hex(presented) {
        Ok(token) => {
            let fingerprint = token.fingerprint();
            match authority.resolve(&token) {
                Some(agent_id) => Ok(HandshakeIdentity {
                    agent_id: Some(agent_id),
                    token_fingerprint: Some(fingerprint),
                }),
                None => Err(format!(
                    "unknown bridge token (fingerprint={fingerprint}); \
                     the daemon never issued this token or its spawn has terminated"
                )),
            }
        }
        Err(_) => {
            // Non-hex tokens are still accepted for back-compat with drivers
            // built before TokenIssuer wiring reached every spawn site. The
            // bridge's claimed agent_id is taken at face value in this mode.
            debug!(
                "bridge IPC: legacy-format auth token (not 64-hex); \
                 falling back to self-claimed agent_id"
            );
            Ok(HandshakeIdentity {
                agent_id: None,
                token_fingerprint: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_mcp_bridge::protocol::{CallRequest, CallResult};
    use openfang_runtime::bridge_auth::TokenIssuer;
    use tokio::io::BufReader;
    use tokio::net::UnixStream as ClientStream;

    // ---- frame floor (clamp_result_to_frame) --------------------------------

    /// Serialized size of the response frame this result would produce. The
    /// assertions below measure THIS, not the payload length, because the
    /// codec's refusal is against the encoded frame.
    fn encoded_frame_len(result: CallResult) -> usize {
        let frame = Frame::Response(CallResponse {
            request_id: 7,
            result,
        });
        serde_json::to_vec(&frame).unwrap().len()
    }

    #[test]
    fn a_result_within_budget_is_returned_untouched() {
        let content = "x".repeat(4096);
        let (out, clamped) = clamp_result_to_frame(CallResult::Ok {
            content: content.clone(),
            is_error: false,
        });
        assert!(clamped.is_none(), "small result must not be clamped");
        match out {
            CallResult::Ok {
                content: got,
                is_error,
            } => {
                assert_eq!(got, content, "payload must be byte-identical");
                assert!(!is_error, "clamping must not invent an error flag");
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn an_oversized_result_is_clamped_to_a_writable_frame() {
        // 4 MiB — four times the frame limit. Before the floor existed this
        // reached `write_frame`, which refused it, and the refusal propagated
        // out of the request loop and CLOSED THE CONNECTION.
        let content = "y".repeat(4 * 1024 * 1024);
        let original = content.len();
        let (out, clamped) = clamp_result_to_frame(CallResult::Ok {
            content,
            is_error: false,
        });
        assert_eq!(
            clamped,
            Some(original),
            "caller must learn the original size"
        );
        let CallResult::Ok {
            content: got,
            is_error,
        } = out.clone()
        else {
            panic!("expected Ok");
        };
        assert!(is_error, "an incomplete result must carry the error flag");
        assert!(
            got.contains("truncated"),
            "the marker must say so in words: {}",
            &got[got.len().saturating_sub(300)..]
        );
        assert!(
            got.contains(&original.to_string()),
            "the marker must name the original size"
        );
        // The load-bearing assertion: the frame the codec would write FITS.
        let len = encoded_frame_len(out);
        assert!(
            len <= MAX_FRAME_BYTES,
            "clamped frame is {len} bytes, over the {MAX_FRAME_BYTES} limit"
        );
    }

    #[test]
    fn a_payload_of_pure_escapes_still_fits() {
        // Regression guard on measuring ESCAPED cost. A budget applied to raw
        // byte length would pass ~1 MiB of quotes through, and each one
        // doubles on the wire, so the encoded frame would be ~2 MiB and the
        // codec would refuse it — the exact failure the clamp exists to stop.
        let content = "\"".repeat(2 * 1024 * 1024);
        let (out, clamped) = clamp_result_to_frame(CallResult::Ok {
            content,
            is_error: false,
        });
        assert!(clamped.is_some());
        let len = encoded_frame_len(out);
        assert!(
            len <= MAX_FRAME_BYTES,
            "escape-heavy clamped frame is {len} bytes, over the limit"
        );
    }

    #[test]
    fn clamping_never_splits_a_codepoint() {
        // A multi-byte char straddling the budget boundary must not be cut in
        // half: the payload has to stay valid UTF-8 or serialization itself
        // becomes the failure.
        let content = "\u{1f600}".repeat(1024 * 1024);
        let (out, clamped) = clamp_result_to_frame(CallResult::Ok {
            content,
            is_error: false,
        });
        assert!(clamped.is_some());
        let CallResult::Ok { content: got, .. } = out.clone() else {
            panic!("expected Ok");
        };
        // `String` cannot hold invalid UTF-8, so the real check is that every
        // emoji survived whole rather than as a partial sequence.
        let body = got.split("\n\n[openfang:").next().unwrap();
        assert_eq!(
            body.len() % 4,
            0,
            "body must be a whole number of 4-byte codepoints"
        );
        assert!(body.chars().all(|c| c == '\u{1f600}'));
        assert!(encoded_frame_len(out) <= MAX_FRAME_BYTES);
    }

    #[test]
    fn an_oversized_dispatch_error_is_clamped_too() {
        let message = "z".repeat(3 * 1024 * 1024);
        let (out, clamped) = clamp_result_to_frame(CallResult::Error { message });
        assert!(clamped.is_some());
        match &out {
            CallResult::Error { message } => assert!(message.contains("truncated")),
            other => panic!("expected Error, got {other:?}"),
        }
        assert!(encoded_frame_len(out) <= MAX_FRAME_BYTES);
    }

    #[test]
    fn an_existing_tool_error_flag_survives_clamping() {
        let (out, _) = clamp_result_to_frame(CallResult::Ok {
            content: "boom".to_string(),
            is_error: true,
        });
        match out {
            CallResult::Ok { is_error, .. } => assert!(is_error),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    fn wire_png(len: usize) -> WireImage {
        WireImage {
            mime_type: "image/png".to_string(),
            data_base64: "A".repeat(len),
        }
    }

    // ANAI-297: an image result that fits passes through untouched, image and
    // all, and the frame the codec would write is within the limit.
    #[test]
    fn a_rich_result_that_fits_passes_through_whole() {
        let img = wire_png(900 * 1024);
        let (out, clamped) = clamp_result_to_frame(CallResult::Rich {
            content: "header".to_string(),
            is_error: false,
            images: vec![img.clone()],
        });
        assert_eq!(clamped, None);
        match &out {
            CallResult::Rich {
                content,
                is_error,
                images,
            } => {
                assert_eq!(content, "header");
                assert!(!is_error);
                assert_eq!(images, &vec![img]);
            }
            other => panic!("expected Rich, got {other:?}"),
        }
        assert!(encoded_frame_len(out) <= MAX_FRAME_BYTES);
    }

    // The load-bearing ANAI-297 clamp test. An image that does not fit is
    // dropped WHOLE: no partial base64 survives, the result is an error, and
    // the message says the image was not returned. A text-style truncation
    // here would hand the model a corrupt image flagged as a successful read.
    #[test]
    fn an_oversized_rich_result_drops_images_whole_and_errors() {
        let (out, clamped) = clamp_result_to_frame(CallResult::Rich {
            content: "header".to_string(),
            is_error: false,
            images: vec![wire_png(2 * 1024 * 1024)],
        });
        assert!(clamped.is_some(), "caller must learn the original size");
        match &out {
            CallResult::Ok { content, is_error } => {
                assert!(*is_error, "a dropped image must never read as success");
                assert!(content.contains("NOT returned"), "{content}");
                assert!(
                    !content.contains("AAAA"),
                    "no fragment of the image data may leak into the text"
                );
            }
            other => panic!("expected a text-only error result, got {other:?}"),
        }
        assert!(encoded_frame_len(out) <= MAX_FRAME_BYTES);
    }

    // Two images that each fit but together do not are refused together:
    // the budget is the whole frame, not per image.
    #[test]
    fn images_are_budgeted_as_a_set_not_individually() {
        let (out, clamped) = clamp_result_to_frame(CallResult::Rich {
            content: String::new(),
            is_error: false,
            images: vec![wire_png(600 * 1024), wire_png(600 * 1024)],
        });
        assert!(clamped.is_some());
        assert!(matches!(out, CallResult::Ok { is_error: true, .. }));
    }

    /// End-to-end wire-shape test: bind a listener at a tempfile path,
    /// connect, do the handshake, send two CallRequests:
    ///   1. A non-allowlisted tool — expect `CallResult::Error` from the
    ///      step-2 allowlist check.
    ///   2. An allowlisted tool — expect a canned `CallResult::Ok` from
    ///      the test twin (the real handler would dispatch into
    ///      `execute_tool`; we can't synthesize an `OpenFangKernel` here).
    ///
    /// What this test guarantees:
    /// - The Hello/HelloAck handshake stays correct.
    /// - The allowlist gate fires *before* dispatch (no kernel touched).
    /// - The wire framing for `CallResponse::Ok` and `CallResponse::Error`
    ///   round-trips cleanly.
    ///
    /// What this test does NOT cover (intentionally — needs a real kernel):
    /// - That `execute_tool` is invoked with the right argument bundle.
    /// - That tool results are correctly mapped to `CallResult::Ok`.
    /// Those land as integration tests once the daemon side spawns the
    /// bridge for real (ANAI-31).
    #[tokio::test]
    async fn ipc_handshake_and_allowlist_gate() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = tmp.path().join("bridge.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        // Spin up a real authority and issue a token for an agent. The twin
        // resolves the handshake token through it, exercising the hardened
        // auth path end-to-end (handshake → resolve → AgentId binding).
        let authority = BridgeAuthority::new();
        let agent_id = AgentId::new();
        let guard = authority.issue(agent_id);
        let presented_token = guard.token().to_hex();

        let server_authority = authority.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_connection_test_twin(stream, server_authority)
                .await
                .unwrap();
        });

        let mut client = ClientStream::connect(&sock).await.unwrap();
        let (cr, mut cw) = client.split();
        let mut cr = BufReader::new(cr);

        // Handshake — real hex token resolves to `agent_id` via authority.
        let hello = Frame::Hello(Hello {
            protocol_version: PROTOCOL_VERSION,
            token: presented_token,
            bridge_version: "test".into(),
        });
        codec::write_frame(&mut cw, &hello).await.unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::HelloAck(HelloAck::Ok { .. }) => {}
            other => panic!("expected HelloAck::Ok, got {other:?}"),
        }

        // 1. Non-allowlisted tool → allowlist Error.
        codec::write_frame(
            &mut cw,
            &Frame::Call(CallRequest {
                request_id: 1,
                agent_id: "test-agent".into(),
                tool_name: "definitely_not_a_real_tool".into(), // deliberately not on the list
                args: serde_json::json!({"cmd": "rm -rf /"}),
            }),
        )
        .await
        .unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::Response(CallResponse {
                request_id: 1,
                result: CallResult::Error { message },
            }) => {
                assert!(
                    message.contains("not in bridge allowlist"),
                    "expected allowlist rejection, got: {message}"
                );
            }
            other => panic!("unexpected response to disallowed tool: {other:?}"),
        }

        // 2. Allowlisted tool → twin returns canned Ok.
        codec::write_frame(
            &mut cw,
            &Frame::Call(CallRequest {
                request_id: 2,
                agent_id: "test-agent".into(),
                tool_name: "file_read".into(),
                args: serde_json::json!({"path": "x"}),
            }),
        )
        .await
        .unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::Response(CallResponse {
                request_id: 2,
                result: CallResult::Ok { is_error, .. },
            }) => {
                // Twin canned response is a non-error Ok; the real handler
                // would set `is_error` from `execute_tool`'s ToolResult.
                assert!(!is_error);
            }
            other => panic!("unexpected response to allowed tool: {other:?}"),
        }

        drop(client);
        server.await.unwrap();

        // Token guard outlives the twin's reads. Drop now so the spawn
        // table empties before we exit the test (sanity check on lifetimes).
        drop(guard);
        assert_eq!(authority.live_spawn_count(), 0);
    }

    /// Test-only twin of [`handle_connection`].
    ///
    /// Mirrors the production handler's *wire* behavior (handshake +
    /// request loop + allowlist gate) but stubs the runtime dispatch
    /// because we can't synthesize an `OpenFangKernel` in unit tests.
    /// If the production handler's wire shape diverges, update this twin.
    async fn handle_connection_test_twin(
        mut stream: UnixStream,
        authority: Arc<BridgeAuthority>,
    ) -> std::io::Result<()> {
        let (read_half, mut write_half) = stream.split();
        let mut read_half = BufReader::new(read_half);

        let hello = match codec::read_frame(&mut read_half).await? {
            Frame::Hello(h) => h,
            _ => return Ok(()),
        };
        if let Err(reason) = authenticate_hello(&hello, &authority) {
            let _ = codec::write_frame(
                &mut write_half,
                &Frame::HelloAck(HelloAck::Rejected { reason }),
            )
            .await;
            return Ok(());
        }
        codec::write_frame(
            &mut write_half,
            &Frame::HelloAck(HelloAck::Ok {
                daemon_version: daemon_version(),
                convert_options_schema: None,
            }),
        )
        .await?;

        loop {
            let frame = match codec::read_frame(&mut read_half).await {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(e),
            };
            match frame {
                Frame::Call(call) => {
                    // Mirror production allowlist + mcp_* early-branch logic.
                    let result = if openfang_runtime::mcp::is_mcp_tool(&call.tool_name) {
                        // Twin can't reach real mcp_connections; canned
                        // upstream-style ok lets list+invoke round-trip in tests.
                        CallResult::Ok {
                            content: format!(
                                "[test-twin canned upstream ok for {}]",
                                call.tool_name
                            ),
                            is_error: false,
                        }
                    } else if !ALLOWED_TOOLS.iter().any(|t| *t == call.tool_name) {
                        CallResult::Error {
                            message: format!(
                                "tool '{}' not in bridge allowlist (permitted: {:?})",
                                call.tool_name, ALLOWED_TOOLS
                            ),
                        }
                    } else {
                        // Canned Ok stand-in for `execute_tool` — kernel-free tests
                        // can't exercise the real dispatch path.
                        CallResult::Ok {
                            content: format!("[test-twin canned ok for {}]", call.tool_name),
                            is_error: false,
                        }
                    };

                    codec::write_frame(
                        &mut write_half,
                        &Frame::Response(CallResponse {
                            request_id: call.request_id,
                            result,
                        }),
                    )
                    .await?;
                }
                Frame::ListUpstream(req) => {
                    // Canned upstream-tools list. Real handler walks
                    // `kernel.mcp_connections`; twin returns a fixed
                    // shape so wire round-trip can be exercised.
                    let response = UpstreamListResponse {
                        request_id: req.request_id,
                        result: UpstreamListResult::Ok {
                            tools: vec![UpstreamToolDef {
                                name: "mcp_twinsrv_ping".to_string(),
                                server: "twinsrv".to_string(),
                                description: Some("canned twin tool".to_string()),
                                input_schema: serde_json::json!({"type": "object"}),
                            }],
                        },
                    };
                    codec::write_frame(&mut write_half, &Frame::UpstreamList(response)).await?;
                }
                _ => continue,
            }
        }
    }

    #[tokio::test]
    async fn ipc_list_upstream_roundtrip() {
        // End-to-end: handshake, send ListUpstream, expect UpstreamList
        // response with the twin's canned tool. Locks the wire shape of
        // the new variants and ensures the request loop dispatches them
        // alongside CallRequest without breaking the existing path.
        let tmp = tempfile::tempdir().unwrap();
        let sock = tmp.path().join("bridge.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let authority = BridgeAuthority::new();
        let agent_id = AgentId::new();
        let guard = authority.issue(agent_id);
        let presented_token = guard.token().to_hex();

        let server_authority = authority.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_connection_test_twin(stream, server_authority)
                .await
                .unwrap();
        });

        let mut client = ClientStream::connect(&sock).await.unwrap();
        let (cr, mut cw) = client.split();
        let mut cr = BufReader::new(cr);

        let hello = Frame::Hello(Hello {
            protocol_version: PROTOCOL_VERSION,
            token: presented_token,
            bridge_version: "test".into(),
        });
        codec::write_frame(&mut cw, &hello).await.unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::HelloAck(HelloAck::Ok { .. }) => {}
            other => panic!("expected HelloAck::Ok, got {other:?}"),
        }

        codec::write_frame(
            &mut cw,
            &Frame::ListUpstream(ListUpstreamRequest { request_id: 42 }),
        )
        .await
        .unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::UpstreamList(UpstreamListResponse {
                request_id: 42,
                result: UpstreamListResult::Ok { tools },
            }) => {
                assert_eq!(tools.len(), 1, "twin advertises one canned tool");
                assert_eq!(tools[0].name, "mcp_twinsrv_ping");
                assert_eq!(tools[0].server, "twinsrv");
            }
            other => panic!("unexpected response to ListUpstream: {other:?}"),
        }

        // Confirm the request loop still handles a Call after a
        // ListUpstream (no state corruption between message kinds).
        codec::write_frame(
            &mut cw,
            &Frame::Call(CallRequest {
                request_id: 43,
                agent_id: agent_id.to_string(),
                tool_name: "file_read".into(),
                args: serde_json::json!({"path": "x"}),
            }),
        )
        .await
        .unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::Response(CallResponse {
                request_id: 43,
                result: CallResult::Ok {
                    is_error: false, ..
                },
            }) => {}
            other => panic!("unexpected response after ListUpstream: {other:?}"),
        }

        drop(client);
        server.await.unwrap();
        drop(guard);
        assert_eq!(authority.live_spawn_count(), 0);
    }

    #[tokio::test]
    async fn ipc_mcp_call_through_twin_returns_canned_ok() {
        // The twin's mcp_* branch returns a canned ok rather than the
        // allowlist error, exercising the production early-branch shape:
        // mcp_* tools bypass the static ALLOWED_TOOLS gate.
        let tmp = tempfile::tempdir().unwrap();
        let sock = tmp.path().join("bridge.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let authority = BridgeAuthority::new();
        let agent_id = AgentId::new();
        let guard = authority.issue(agent_id);
        let presented_token = guard.token().to_hex();

        let server_authority = authority.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_connection_test_twin(stream, server_authority)
                .await
                .unwrap();
        });

        let mut client = ClientStream::connect(&sock).await.unwrap();
        let (cr, mut cw) = client.split();
        let mut cr = BufReader::new(cr);

        let hello = Frame::Hello(Hello {
            protocol_version: PROTOCOL_VERSION,
            token: presented_token,
            bridge_version: "test".into(),
        });
        codec::write_frame(&mut cw, &hello).await.unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::HelloAck(HelloAck::Ok { .. }) => {}
            other => panic!("expected HelloAck::Ok, got {other:?}"),
        }

        codec::write_frame(
            &mut cw,
            &Frame::Call(CallRequest {
                request_id: 7,
                agent_id: agent_id.to_string(),
                tool_name: "mcp_linear_getteams".into(),
                args: serde_json::json!({}),
            }),
        )
        .await
        .unwrap();
        match codec::read_frame(&mut cr).await.unwrap() {
            Frame::Response(CallResponse {
                request_id: 7,
                result:
                    CallResult::Ok {
                        content,
                        is_error: false,
                    },
            }) => {
                assert!(content.contains("mcp_linear_getteams"));
            }
            other => panic!("unexpected response to mcp_* call: {other:?}"),
        }

        drop(client);
        server.await.unwrap();
        drop(guard);
    }

    #[test]
    fn authenticate_hello_rejects_version_mismatch() {
        let authority = BridgeAuthority::new();
        let h = Hello {
            protocol_version: 999,
            token: "x".into(),
            bridge_version: "t".into(),
        };
        assert!(authenticate_hello(&h, &authority).is_err());
    }

    #[test]
    fn authenticate_hello_rejects_empty_token() {
        let authority = BridgeAuthority::new();
        let h = Hello {
            protocol_version: PROTOCOL_VERSION,
            token: "".into(),
            bridge_version: "t".into(),
        };
        assert!(authenticate_hello(&h, &authority).is_err());
    }

    #[test]
    fn authenticate_hello_resolves_authority_token() {
        // Hardened path: hex-encoded daemon-issued token → AgentId.
        let authority = BridgeAuthority::new();
        let agent_id = AgentId::new();
        let guard = authority.issue(agent_id);
        let h = Hello {
            protocol_version: PROTOCOL_VERSION,
            token: guard.token().to_hex(),
            bridge_version: "t".into(),
        };
        let identity = authenticate_hello(&h, &authority).expect("hardened path should succeed");
        assert_eq!(identity.agent_id, Some(agent_id));
        assert_eq!(identity.token_fingerprint, Some(guard.fingerprint()));
    }

    #[test]
    fn authenticate_hello_rejects_unknown_hex_token() {
        // Well-formed 64-hex token the authority never issued — attacker /
        // replay / stale-spawn case. Must be rejected outright; no legacy
        // fallback for well-formed-but-unknown tokens.
        let authority = BridgeAuthority::new();
        let stranger = Token::generate();
        let h = Hello {
            protocol_version: PROTOCOL_VERSION,
            token: stranger.to_hex(),
            bridge_version: "t".into(),
        };
        let err = authenticate_hello(&h, &authority).expect_err("unknown hex must reject");
        assert!(
            err.contains("unknown bridge token"),
            "expected unknown-token rejection, got: {err}"
        );
    }

    #[test]
    fn allowlist_contains_web_search() {
        // 13d: native CC `WebSearch` is denied by the 13a deny set; restore
        // it through the bridge so researcher agents (medical, business)
        // keep their primary research surface. Zero new plumbing — kernel
        // `web_ctx` is already passed to `execute_tool` at this call site.
        assert!(
            ALLOWED_TOOLS.contains(&"web_search"),
            "web_search must be on the bridge allowlist post-13a"
        );
    }

    #[test]
    fn allowlist_contains_shell_exec() {
        // 13b: shell_exec reachable through the bridge so CC subprocesses
        // operating under the 13a native-deny set still have a path to the
        // shell (gated, sandboxed, exec_policy-enforced). Locking this in by
        // name so a refactor of `ALLOWED_TOOLS` doesn't silently drop it.
        assert!(
            ALLOWED_TOOLS.contains(&"shell_exec"),
            "shell_exec must be on the bridge allowlist post-13a"
        );
    }

    #[test]
    fn allowlist_contains_apply_patch() {
        // 13e: apply_patch reachable through the bridge as a surgical-edit
        // alternative to whole-file `file_write` rewrites. Mitigates the
        // token + drift cost of the missing CC `Edit` tool while we wait on a
        // native `string_edit` follow-up. Name-locked so a refactor can't
        // silently drop it.
        assert!(
            ALLOWED_TOOLS.contains(&"apply_patch"),
            "apply_patch must be on the bridge allowlist post-13e"
        );
    }

    /// **Drift-catcher: surface correspondence + safe-by-default subset.**
    ///
    /// Three lists describe the bridge tool surface, with two distinct
    /// relationships:
    ///
    /// 1. `openfang_api::bridge_ipc::ALLOWED_TOOLS` — daemon-side dispatch
    ///    allowlist (the call-time gate).
    /// 2. `openfang_mcp_bridge::built_in_tools()` — MCP advertise surface
    ///    (what CC actually sees in `tools/list`).
    /// 3. `openfang_mcp_bridge::DEFAULT_ALLOWED` — bridge-process default
    ///    when `OPENFANG_BRIDGE_ALLOWED` is unset (legacy/dev fallback).
    ///
    /// **Invariant A (equality):** `ALLOWED_TOOLS == built_in_tools()`.
    /// Lesson from 13b/13d: a tool can be daemon-dispatchable but invisible
    /// to CC because someone forgot to add it to `built_in_tools()`.
    ///
    /// **Invariant B (safe-by-default subset, S7-06 / S4-02):**
    /// `DEFAULT_ALLOWED ⊂ ALLOWED_TOOLS`, with the deliberate exclusion of
    /// `PRIVILEGED_DEFAULT_DENY` — the agent-lifecycle verbs
    /// (`agent_spawn`, `agent_kill`, `agent_activate`, `agent_send_async`)
    /// and the page-driving browser verbs (`browser_click`, `browser_type`,
    /// `browser_screenshot`, `browser_run_js`, `browser_back`). The runtime
    /// threads the manifest-derived allowlist through
    /// `OPENFANG_BRIDGE_ALLOWED`, so opted-in agents still reach these
    /// tools; only the no-env-var fallback is narrowed.
    ///
    /// If you're here because this test failed: a bridge tool add or remove
    /// must touch `crates/openfang-api/src/bridge_ipc.rs` (`ALLOWED_TOOLS`)
    /// and `crates/openfang-mcp-bridge/src/lib.rs` (`built_in_tools` and
    /// usually `DEFAULT_ALLOWED`). For privileged additions, append to
    /// `PRIVILEGED_DEFAULT_DENY` and **not** to `DEFAULT_ALLOWED`.
    #[test]
    fn allowlist_surface_correspondence() {
        use openfang_mcp_bridge::{built_in_tools, DEFAULT_ALLOWED, PRIVILEGED_DEFAULT_DENY};
        use std::collections::BTreeSet;

        let daemon_set: BTreeSet<&str> = ALLOWED_TOOLS.iter().copied().collect();
        let advertise_set: BTreeSet<String> = built_in_tools()
            .iter()
            .map(|t| t.name.as_ref().to_string())
            .collect();
        let advertise_borrowed: BTreeSet<&str> = advertise_set.iter().map(|s| s.as_str()).collect();
        let default_set: BTreeSet<&str> = DEFAULT_ALLOWED.iter().copied().collect();
        let privileged_set: BTreeSet<&str> = PRIVILEGED_DEFAULT_DENY.iter().copied().collect();

        // Invariant A — daemon dispatch and MCP advertise must agree.
        assert_eq!(
            daemon_set,
            advertise_borrowed,
            "drift: ALLOWED_TOOLS (daemon dispatch) ≠ built_in_tools() (MCP advertise). \
             daemon-only: {:?}, advertise-only: {:?}",
            daemon_set
                .difference(&advertise_borrowed)
                .collect::<Vec<_>>(),
            advertise_borrowed
                .difference(&daemon_set)
                .collect::<Vec<_>>(),
        );

        // Invariant B.1 — every default-tool is daemon-dispatchable.
        let default_extras: Vec<&&str> = default_set.difference(&daemon_set).collect();
        assert!(
            default_extras.is_empty(),
            "drift: DEFAULT_ALLOWED contains tools missing from ALLOWED_TOOLS: {:?}",
            default_extras,
        );

        // Invariant B.2 — every privileged tool is daemon-dispatchable.
        let privileged_extras: Vec<&&str> = privileged_set.difference(&daemon_set).collect();
        assert!(
            privileged_extras.is_empty(),
            "drift: PRIVILEGED_DEFAULT_DENY contains tools missing from ALLOWED_TOOLS: {:?}",
            privileged_extras,
        );

        // Invariant B.3 — privileged tools must NOT be in the default
        // fallback. This is the S7-06 / S4-02 pin: if a future commit
        // accidentally re-adds `agent_spawn` etc. to DEFAULT_ALLOWED, the
        // legacy/dev path silently re-grants lifecycle control.
        let leaked: Vec<&&str> = privileged_set.intersection(&default_set).collect();
        assert!(
            leaked.is_empty(),
            "S7-06/S4-02 regression: PRIVILEGED_DEFAULT_DENY entries leaked into \
             DEFAULT_ALLOWED: {:?}. Privileged agent-lifecycle tools must reach the \
             bridge only via manifest-driven OPENFANG_BRIDGE_ALLOWED.",
            leaked,
        );

        // Invariant B.4 — the privileged set and the default set together
        // cover the daemon surface. Catches "added a new tool to
        // ALLOWED_TOOLS/built_in_tools but forgot to classify it as
        // default-safe or privileged-deny".
        let mut union: BTreeSet<&str> = default_set.clone();
        union.extend(privileged_set.iter().copied());
        let unclassified: Vec<&&str> = daemon_set.difference(&union).collect();
        assert!(
            unclassified.is_empty(),
            "drift: ALLOWED_TOOLS entries unclassified (neither in DEFAULT_ALLOWED \
             nor PRIVILEGED_DEFAULT_DENY): {:?}",
            unclassified,
        );
    }

    /// **Invariant C (per-tool schema correspondence).**
    ///
    /// Invariant A proves the two surfaces agree on *tool names*. It says
    /// nothing about the *shape* of each tool, and that gap has now bitten
    /// twice: `surface_to` (ANAI-126) and `timeout_secs` (ANAI-196) were
    /// both added to `openfang_runtime::tool_runner::builtin_tool_definitions()`
    /// and forgotten in `openfang_mcp_bridge::built_in_tools()`, so every
    /// bridge-backed agent was structurally unable to pass a parameter the
    /// runtime accepts.
    ///
    /// Both times the response was a hand-written, field-by-field
    /// `contains_key("surface_to")` guard living in the bridge crate. That
    /// shape cannot work: it only catches the *one* field someone thought to
    /// name, and the bridge crate has no dependency on the runtime, so it can
    /// only ever compare a literal against another literal. This test lives
    /// in `openfang-api` — the one crate that depends on both — and asserts
    /// **set equality of `properties` keys and of `required`** for every
    /// mirrored tool. A new field on either side fails here by construction,
    /// with no per-field maintenance.
    ///
    /// If you're here because this test failed: the two schemas for the named
    /// tool disagree. Mirror the field into
    /// `crates/openfang-mcp-bridge/src/lib.rs` (`built_in_tools`), copying the
    /// description verbatim from `tool_runner.rs`. Do **not** add a new
    /// single-field assertion — this test already covers it.
    ///
    /// Deliberately *not* asserted: description prose and per-property
    /// details (`type`, `enum`, `description`). Those drift benignly and
    /// pinning them would make this test a rewrite-tax on every copy edit.
    /// The failure mode we care about is a parameter that is *unreachable*
    /// through the bridge, which is exactly a key-set difference.
    ///
    /// Known, deliberate divergences live in `MIRROR_EXCEPTIONS` below, each
    /// with a reason. The list is self-cleaning: an exception that is no
    /// longer divergent fails the test, so a stale waiver cannot silently
    /// blind the check.
    /// `(tool, property, why)` — deliberate divergences, exempt from the
    /// equality assertion.
    const MIRROR_EXCEPTIONS: &[(&str, &str, &str)] = &[
        (
            "file_convert",
            "options",
            "ANAI-131: the static bridge entry ships WITHOUT `options` by \
                 design. The dispatcher injects a live projection computed \
                 daemon-side from the recipe manifest \
                 (`inject_convert_options`), so the advertised surface matches \
                 the recipes actually installed. A static mirror here would be \
                 wrong, not merely redundant. Covered by \
                 `openfang_mcp_bridge::tests::file_convert_advertises_injected_options_schema`.",
        ),
        (
            "channel_send",
            "image_url",
            "ANAI-196 finding: unmirrored, pending review — see below.",
        ),
        (
            "channel_send",
            "file_url",
            "ANAI-196 finding: unmirrored, pending review — see below.",
        ),
        (
            "channel_send",
            "file_path",
            "ANAI-196 finding: unmirrored, pending review. `file_path` \
                 makes `tool_channel_send` read local disk \
                 (`resolve_file_path` + `fs::read`) and ship the bytes to an \
                 arbitrary channel recipient, yet `channel_send` is \
                 deliberately absent from `FS_SANDBOXED_TOOLS` — whose doc \
                 comment asserts it does not touch the filesystem. It does. \
                 An agent with no registered workspace therefore gets the \
                 unscoped `resolve_file_path` fallback. Mirroring these four \
                 properties would make that path discoverable to every bridge \
                 agent, so it waits on the sandbox decision rather than riding \
                 along with a schema fix.",
        ),
        (
            "channel_send",
            "filename",
            "ANAI-196 finding: unmirrored, pending review — see `file_path`.",
        ),
    ];

    #[test]
    fn advertised_tool_schemas_match_runtime() {
        use openfang_mcp_bridge::built_in_tools;
        use openfang_runtime::tool_runner::builtin_tool_definitions;
        use std::collections::{BTreeMap, BTreeSet};

        fn key_set(schema: &serde_json::Value, field: &str) -> BTreeSet<String> {
            match field {
                "properties" => schema
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default(),
                _ => schema
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        }

        let runtime: BTreeMap<String, serde_json::Value> = builtin_tool_definitions()
            .into_iter()
            .map(|d| (d.name, d.input_schema))
            .collect();

        let mut drift: Vec<String> = Vec::new();
        // Exceptions actually exercised this run; anything declared but never
        // hit is stale and fails below.
        let mut waivers_used: BTreeSet<(String, String)> = BTreeSet::new();

        for tool in built_in_tools() {
            let name = tool.name.as_ref().to_string();
            let Some(runtime_schema) = runtime.get(&name) else {
                drift.push(format!(
                    "{name}: advertised by the bridge but absent from \
                     builtin_tool_definitions() — nothing to mirror against"
                ));
                continue;
            };
            let bridge_schema = serde_json::Value::Object(tool.input_schema.as_ref().clone());

            for field in ["properties", "required"] {
                let bridge_keys = key_set(&bridge_schema, field);
                let runtime_keys = key_set(runtime_schema, field);
                let missing: Vec<&String> = runtime_keys
                    .difference(&bridge_keys)
                    .filter(|k| {
                        let waived = field == "properties"
                            && MIRROR_EXCEPTIONS
                                .iter()
                                .any(|(t, p, _)| *t == name && *p == k.as_str());
                        if waived {
                            waivers_used.insert((name.clone(), (*k).clone()));
                        }
                        !waived
                    })
                    .collect();
                let extra: Vec<&String> = bridge_keys.difference(&runtime_keys).collect();
                if !missing.is_empty() || !extra.is_empty() {
                    drift.push(format!(
                        "{name}.{field}: missing from bridge {missing:?}, \
                         extra on bridge {extra:?}"
                    ));
                }
            }
        }

        let declared: BTreeSet<(String, String)> = MIRROR_EXCEPTIONS
            .iter()
            .map(|(t, p, _)| ((*t).to_string(), (*p).to_string()))
            .collect();
        let stale: Vec<&(String, String)> = declared.difference(&waivers_used).collect();
        assert!(
            stale.is_empty(),
            "stale MIRROR_EXCEPTIONS entries — these no longer diverge, so the \
             waiver is blinding the check for nothing. Delete them: {stale:?}",
        );

        assert!(
            drift.is_empty(),
            "schema drift between openfang_runtime::tool_runner::builtin_tool_definitions() \
             and openfang_mcp_bridge::built_in_tools():\n  {}",
            drift.join("\n  "),
        );
    }

    /// Pins the tool-surface cardinality. Bumps are expected when a new
    /// bridge tool lands — update intentionally, in lockstep with the sets
    /// exercised by [`allowlist_surface_correspondence`].
    ///
    /// Privileged tools (S7-06 / S4-02) are absent from `DEFAULT_ALLOWED`
    /// by design, so its cardinality lags `ALLOWED_TOOLS` by exactly
    /// `PRIVILEGED_DEFAULT_DENY.len()`.
    #[test]
    fn allowlist_cardinality_pin() {
        use openfang_mcp_bridge::{built_in_tools, DEFAULT_ALLOWED, PRIVILEGED_DEFAULT_DENY};
        // ANAI-166: 22 -> 23 (`memory_note`).
        // ANAI-204: 23 -> 25 (`memory_fact`, `memory_history`).
        // Browser: 25 -> 30 (navigate / read_page / wait / scroll / close).
        // Browser, page-driving: 30 -> 35 (click / type / screenshot /
        // run_js / back). All five are privileged-deny, so `DEFAULT_ALLOWED`
        // is unchanged at 26 and `PRIVILEGED_DEFAULT_DENY` goes 4 -> 9.
        // ANAI-292: 35 -> 36 (`file_grep`). NOT privileged-deny — it is
        // granted alongside `file_read`, so `DEFAULT_ALLOWED` moves with it,
        // 26 -> 27, and `PRIVILEGED_DEFAULT_DENY` stays at 9.
        assert_eq!(ALLOWED_TOOLS.len(), 36, "ALLOWED_TOOLS surface cardinality");
        assert_eq!(
            built_in_tools().len(),
            36,
            "built_in_tools() advertise surface cardinality"
        );
        assert_eq!(
            PRIVILEGED_DEFAULT_DENY.len(),
            9,
            "PRIVILEGED_DEFAULT_DENY cardinality (agent lifecycle x4 + browser click/type/screenshot/run_js/back)"
        );
        assert_eq!(
            DEFAULT_ALLOWED.len(),
            27,
            "DEFAULT_ALLOWED bridge-default cardinality (36 − 9 privileged)"
        );
    }

    /// S7-06 / S4-02 pin — explicit, name-level. The set-level test above
    /// catches drift generically; this one names the three tools so a
    /// grep for `agent_spawn` lands on the regression guard.
    #[test]
    fn privileged_lifecycle_tools_excluded_from_default() {
        use openfang_mcp_bridge::{DEFAULT_ALLOWED, PRIVILEGED_DEFAULT_DENY};
        for tool in [
            "agent_spawn",
            "agent_kill",
            "agent_activate",
            "agent_send_async",
        ] {
            assert!(
                PRIVILEGED_DEFAULT_DENY.contains(&tool),
                "{tool} must be in PRIVILEGED_DEFAULT_DENY",
            );
            assert!(
                !DEFAULT_ALLOWED.contains(&tool),
                "S7-06/S4-02 regression: {tool} re-introduced into DEFAULT_ALLOWED. \
                 Privileged agent-lifecycle tools must only reach the bridge via \
                 manifest-derived OPENFANG_BRIDGE_ALLOWED.",
            );
        }
    }

    /// Name-level pin for the page-driving browser verbs. The set-level test
    /// above catches drift generically; this one names the five so a grep for
    /// `browser_run_js` lands on the regression guard.
    ///
    /// The rule: reading a rendered page is default-safe, *driving* one is
    /// not. `run_js` in particular executes arbitrary JavaScript in a live
    /// Chrome session, which SSRF-checking on `browser_navigate` does not
    /// cover. These must be granted per-agent in `agent.toml` or not at all.
    #[test]
    fn page_driving_browser_tools_excluded_from_default() {
        use openfang_mcp_bridge::{DEFAULT_ALLOWED, PRIVILEGED_DEFAULT_DENY};
        for tool in [
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "browser_run_js",
            "browser_back",
        ] {
            assert!(
                ALLOWED_TOOLS.contains(&tool),
                "{tool} must be daemon-dispatchable",
            );
            assert!(
                PRIVILEGED_DEFAULT_DENY.contains(&tool),
                "{tool} must be in PRIVILEGED_DEFAULT_DENY",
            );
            assert!(
                !DEFAULT_ALLOWED.contains(&tool),
                "regression: {tool} re-introduced into DEFAULT_ALLOWED. \
                 Page-driving browser verbs must only reach the bridge via \
                 manifest-derived OPENFANG_BRIDGE_ALLOWED.",
            );
        }
    }

    #[test]
    fn apply_patch_is_workspace_sandboxed() {
        // tool_apply_patch resolves every patch-embedded path against
        // workspace_root. Without sandbox membership a no-workspace agent
        // could ship a patch whose Add/Update/Delete targets fall through to
        // the daemon CWD (`~/.openfang`) — sibling-workspace + secrets leak.
        // Fail-closed gate.
        assert!(
            FS_SANDBOXED_TOOLS.contains(&"apply_patch"),
            "apply_patch must require a registered workspace"
        );
    }

    #[test]
    fn shell_exec_is_workspace_sandboxed() {
        // shell_exec uses workspace_root as cwd (tool_runner.rs:1704-1707).
        // Without sandbox membership, a no-workspace agent would shell out
        // in `~/.openfang` and see secrets.env. Fail-closed gate.
        assert!(
            FS_SANDBOXED_TOOLS.contains(&"shell_exec"),
            "shell_exec must require a registered workspace"
        );
    }

    /// Belt-and-braces: every tool the FS sandbox gates must also be on
    /// the daemon-dispatch allowlist. Catches "added an FS tool to
    /// `FS_SANDBOXED_TOOLS` without wiring it onto the bridge surface"
    /// (or the inverse: exposed a new FS tool without sandboxing it).
    #[test]
    fn fs_sandboxed_tools_subset_of_allowed_tools() {
        use std::collections::BTreeSet;
        let allowed: BTreeSet<&str> = ALLOWED_TOOLS.iter().copied().collect();
        let sandboxed: BTreeSet<&str> = FS_SANDBOXED_TOOLS.iter().copied().collect();
        let extras: Vec<&&str> = sandboxed.difference(&allowed).collect();
        assert!(
            extras.is_empty(),
            "FS_SANDBOXED_TOOLS contains tools missing from ALLOWED_TOOLS: {:?}",
            extras
        );
    }

    #[test]
    fn authenticate_hello_accepts_legacy_non_hex_token() {
        // Back-compat lane: a non-hex non-empty token (e.g. legacy UUID)
        // resolves to `agent_id: None`, signaling the dispatcher to fall
        // back to the bridge's self-claimed agent_id. Closes when every
        // spawn site issues real tokens.
        let authority = BridgeAuthority::new();
        let h = Hello {
            protocol_version: PROTOCOL_VERSION,
            token: "550e8400-e29b-41d4-a716-446655440000".into(),
            bridge_version: "t".into(),
        };
        let identity = authenticate_hello(&h, &authority).expect("legacy path should succeed");
        assert!(
            identity.agent_id.is_none(),
            "legacy path must not bind agent_id"
        );
        assert!(
            identity.token_fingerprint.is_none(),
            "legacy path has no fingerprint"
        );
    }

    /// Drift detection: `RESERVED_BUILTIN_NAMES` (the collision-check list
    /// used at MCP discovery in `openfang-runtime`) MUST equal
    /// `ALLOWED_TOOLS` (this crate). If a new built-in tool is added to
    /// `ALLOWED_TOOLS` and `built_in_tools()` but the reservation list
    /// drifts, an upstream MCP server could shadow the new built-in.
    ///
    /// Owner contract: any addition to `ALLOWED_TOOLS` must also be added
    /// to `openfang_runtime::mcp::RESERVED_BUILTIN_NAMES`.
    #[test]
    fn reserved_builtin_names_matches_allowed_tools() {
        use openfang_runtime::mcp::RESERVED_BUILTIN_NAMES;
        use std::collections::BTreeSet;
        let allowed: BTreeSet<&str> = ALLOWED_TOOLS.iter().copied().collect();
        let reserved: BTreeSet<&str> = RESERVED_BUILTIN_NAMES.iter().copied().collect();
        assert_eq!(
            allowed, reserved,
            "drift: openfang_runtime::mcp::RESERVED_BUILTIN_NAMES ≠ ALLOWED_TOOLS. \
            Built-ins added to ALLOWED_TOOLS must also be added to \
            RESERVED_BUILTIN_NAMES so upstream MCP servers cannot shadow them."
        );
    }
    /// Drift catcher: no built-in tool may use the `mcp_` prefix.
    ///
    /// The dispatch gate `is_mcp_tool(name) = name.starts_with("mcp_")`
    /// in `openfang_runtime::mcp` short-circuits BEFORE the static
    /// `ALLOWED_TOOLS` check at the top of `dispatch_tool_call`. If a
    /// future built-in were named `mcp_*`, calls to it would route to
    /// `dispatch_upstream_mcp_call` instead of the built-in's real
    /// handler. Structurally impossible today; this test locks the
    /// invariant so a future addition doesn't silently subvert dispatch.
    #[test]
    fn no_builtin_uses_mcp_prefix() {
        for name in ALLOWED_TOOLS {
            assert!(
                !name.starts_with("mcp_"),
                "built-in '{name}' uses 'mcp_' prefix; conflicts with \
                 is_mcp_tool dispatch gate in openfang_runtime::mcp"
            );
        }
    }
}
