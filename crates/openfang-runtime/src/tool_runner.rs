//! Built-in tool execution.
//!
//! Provides filesystem, web, shell, and inter-agent tools. Agent tools
//! (agent_send, agent_spawn, etc.) require a KernelHandle to be passed in.

use crate::kernel_handle::KernelHandle;
use crate::mcp;
use crate::web_search::{parse_ddg_results, WebToolsContext};
use openfang_skills::registry::SkillRegistry;
use openfang_types::taint::{TaintLabel, TaintSink, TaintedValue};
use openfang_types::tool::{ToolDefinition, ToolResult};
use openfang_types::tool_compat::normalize_tool_name;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Maximum inter-agent call depth to prevent infinite recursion (A->B->C->...).
const MAX_AGENT_CALL_DEPTH: u32 = 5;

/// Tools whose invocation **must** be scoped to the calling agent's
/// `workspace_root`. Canonical source of truth, consumed by both bridge
/// surfaces (IPC + HTTP `/mcp`) to decide whether a given call's path
/// arguments require workspace-rewriting and whether the call should
/// fail-closed when no workspace is registered.
///
/// History: this list was duplicated in `openfang_api::bridge_ipc` and
/// `openfang_api::routes`, drifted (5 vs 4 tools), and missed
/// `create_directory` on the IPC side + `shell_exec`/`apply_patch` on the
/// HTTP side — both real sandbox gaps. Unified here so additions land in
/// one place.
///
/// Membership criteria: the tool touches the filesystem via a path arg
/// (or, for `shell_exec`, uses `workspace_root` as cwd). Tools that do
/// not touch the FS (`web_fetch`, `agent_*`, `memory_*`, `web_search`,
/// `channel_send`) are intentionally absent.
pub const FS_SANDBOXED_TOOLS: &[&str] = &[
    "file_read",
    "file_grep",
    "image_read",
    "file_list",
    "file_write",
    "create_directory",
    "shell_exec",
    "apply_patch",
    "file_convert",
];

/// Check if a tool name refers to a shell execution tool.
///
/// Used to determine whether exec_policy settings should bypass the approval gate.
/// SECURITY (#919): `process_start` is also a shell execution path — it spawns
/// arbitrary subprocesses via the persistent process manager. It must be gated
/// by the same approval rules as `shell_exec`.
pub(crate) fn is_shell_tool(name: &str) -> bool {
    matches!(name, "shell_exec" | "process_start")
}

/// Render the agent-facing message for a non-approved approval outcome
/// (ANAI-153).
///
/// The three negative outcomes are three different facts and the agent must be
/// able to act differently on each:
///
/// * `Denied` is terminal. A human read the command and refused. Retrying is
///   prompt-spam and the message says so explicitly.
/// * `TimedOut` means nobody answered. The command was never judged. This is
///   the one that burned unattended fan-outs: the agent read "denied",
///   apologised, and abandoned work no human had ever looked at.
/// * `Backpressure` means the request never reached a human at all because the
///   agent's own pending queue was full. Retryable, and likely to succeed once
///   the queue drains.
///
/// `Approved` never reaches here; it is rendered defensively rather than
/// panicking, because a wrong string is better than a killed tool call.
fn approval_outcome_message(
    tool_name: &str,
    decision: openfang_types::approval::ApprovalDecision,
) -> String {
    use openfang_types::approval::ApprovalDecision;
    match decision {
        ApprovalDecision::Approved => {
            format!("Execution of '{tool_name}' was approved.")
        }
        ApprovalDecision::Denied => format!(
            "Execution denied: '{tool_name}' requires human approval and a human explicitly \
             denied it. This is final. Do not retry the same operation; ask the operator \
             what they would prefer instead. The operation was not performed."
        ),
        ApprovalDecision::TimedOut => format!(
            "Approval timed out: '{tool_name}' requires human approval and no human responded \
             before the request expired. This is NOT a denial. Nobody judged the command. \
             The operation was not performed and may be retried when someone is available; \
             if this is an unattended run, report the pending approval rather than \
             abandoning the task."
        ),
        ApprovalDecision::Backpressure => format!(
            "Approval queue full: '{tool_name}' requires human approval but this agent already \
             has the maximum number of approval requests pending, so the request was never \
             surfaced to a human. This is NOT a denial. The operation was not performed and \
             may be retried once the pending requests resolve."
        ),
        // ANAI-186: record-only variants. `request_approval` never returns
        // these — they exist to label rows in the approvals feed — but a
        // wrong string still beats a killed tool call.
        ApprovalDecision::GatekeeperSuppressed | ApprovalDecision::GatekeeperDenied => format!(
            "Execution of '{tool_name}' was resolved by the approval gatekeeper without \
             human review."
        ),
    }
}

/// Extract argv[0] (the binary) from a `shell_exec` command for approval
/// caching's "Approve Similar" scope. Skips leading `VAR=value` environment
/// assignments, then returns the first whitespace-delimited token. Returns
/// `None` if nothing remains. shell_exec's input filter already bans shell
/// metacharacters, so the first token is an unambiguous binary name.
fn extract_cache_binary(command: &str) -> Option<String> {
    for tok in command.split_whitespace() {
        if let Some(eq) = tok.find('=') {
            let name = &tok[..eq];
            let looks_like_env = !name.is_empty()
                && name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if looks_like_env {
                continue;
            }
        }
        return Some(tok.to_string());
    }
    None
}

/// Check if a shell command should be blocked by taint tracking.
///
/// Layer 1: Shell metacharacter injection (backticks, `$(`, `${`, etc.)
/// Layer 2: Heuristic patterns for injected external data (piped curl, base64, eval)
///
/// This implements the TaintSink::shell_exec() policy from SOTA 2.
fn check_taint_shell_exec(command: &str) -> Option<String> {
    // Layer 1: Block shell metacharacters that enable command injection.
    // Uses the same validator as subprocess_sandbox and docker_sandbox.
    if let Some(reason) = crate::subprocess_sandbox::contains_shell_metacharacters(command) {
        return Some(format!("Shell metacharacter injection blocked: {reason}"));
    }

    // Layer 2: Heuristic patterns for injected external URLs / base64 payloads
    let suspicious_patterns = ["curl ", "wget ", "| sh", "| bash", "base64 -d", "eval "];
    for pattern in &suspicious_patterns {
        if command.contains(pattern) {
            let mut labels = HashSet::new();
            labels.insert(TaintLabel::ExternalNetwork);
            let tainted = TaintedValue::new(command, labels, "llm_tool_call");
            if let Err(violation) = tainted.check_sink(&TaintSink::shell_exec()) {
                warn!(command = crate::str_utils::safe_truncate_str(command, 80), %violation, "Shell taint check failed");
                return Some(violation.to_string());
            }
        }
    }
    None
}

/// Check if a URL should be blocked by taint tracking before network fetch.
///
/// Blocks URLs that appear to contain API keys, tokens, or other secrets
/// in query parameters (potential data exfiltration). Implements TaintSink::net_fetch().
fn check_taint_net_fetch(url: &str) -> Option<String> {
    let exfil_patterns = [
        "api_key=",
        "apikey=",
        "token=",
        "secret=",
        "password=",
        "Authorization:",
    ];
    for pattern in &exfil_patterns {
        if url.to_lowercase().contains(&pattern.to_lowercase()) {
            let mut labels = HashSet::new();
            labels.insert(TaintLabel::Secret);
            let tainted = TaintedValue::new(url, labels, "llm_tool_call");
            if let Err(violation) = tainted.check_sink(&TaintSink::net_fetch()) {
                warn!(url = crate::str_utils::safe_truncate_str(url, 80), %violation, "Net fetch taint check failed");
                return Some(violation.to_string());
            }
        }
    }
    None
}

tokio::task_local! {
    /// Tracks the current inter-agent call depth within a task.
    static AGENT_CALL_DEPTH: std::cell::Cell<u32>;
    /// Canvas max HTML size in bytes (set from kernel config at loop start).
    pub static CANVAS_MAX_BYTES: usize;
    /// Inbound wake lineage for the current woken turn (ANAI-110).
    ///
    /// Set by `Kernel::run_woken_agent_loop` around the woken send, scoped to
    /// the claimed [`WakeEnvelope`](openfang_types::wake::WakeEnvelope)'s chain
    /// (whose `current` is THIS agent). Read by `tool_agent_send_async` so a
    /// nested wake extends the REAL root->...->this chain instead of re-rooting
    /// at the sender every hop — the threading that makes cross-hop cycle
    /// (req 4) and depth (req 9) enforce for real. Absent on origin turns
    /// (channel / cron / API), which have no wake ancestry.
    pub static WAKE_LINEAGE: openfang_types::wake::WakeLineage;
}

/// Get the current inter-agent call depth from the task-local context.
/// Returns 0 if called outside an agent task.
pub fn current_agent_depth() -> u32 {
    AGENT_CALL_DEPTH.try_with(|d| d.get()).unwrap_or(0)
}

/// Resolve the base wake lineage for a new `agent_send_async` (ANAI-110).
///
/// Inside a woken turn, `Kernel::run_woken_agent_loop` scopes [`WAKE_LINEAGE`]
/// to the inbound chain — whose `current` is already this agent — so the new
/// wake extends the REAL `root -> ... -> this` chain and cross-hop cycle
/// (req 4) / depth (req 9) enforce against actual ancestry. On an origin turn
/// (channel / cron / API) the task-local is unset and there is no ancestry, so
/// the chain is rooted at `sender`, exactly as v1.
fn resolve_wake_base_lineage(sender: &str) -> openfang_types::wake::WakeLineage {
    WAKE_LINEAGE
        .try_with(|l| l.clone())
        .unwrap_or_else(|_| openfang_types::wake::WakeLineage::root_at(sender))
}

/// Execute a tool by name with the given input, returning a ToolResult.
///
/// The optional `kernel` handle enables inter-agent tools. If `None`,
/// agent tools will return an error indicating the kernel is not available.
///
/// `allowed_tools` enforces capability-based security: if provided, only
/// tools in the list may execute. This prevents an LLM from hallucinating
/// tool names outside the agent's capability grants.
#[allow(clippy::too_many_arguments)]
pub async fn execute_tool(
    tool_use_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    allowed_tools: Option<&[String]>,
    caller_agent_id: Option<&str>,
    skill_registry: Option<&SkillRegistry>,
    mcp_connections: Option<&tokio::sync::Mutex<Vec<mcp::McpConnection>>>,
    web_ctx: Option<&WebToolsContext>,
    browser_ctx: Option<&crate::browser::BrowserManager>,
    allowed_env_vars: Option<&[String]>,
    workspace_root: Option<&Path>,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
    exec_policy: Option<&openfang_types::config::ExecPolicy>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    tts_engine: Option<&crate::tts::TtsEngine>,
    docker_config: Option<&openfang_types::config::DockerSandboxConfig>,
    process_manager: Option<&crate::process_manager::ProcessManager>,
    origin: Option<&openfang_types::approval::ApprovalOrigin>,
) -> ToolResult {
    // Normalize the tool name through compat mappings so LLM-hallucinated aliases
    // (e.g. "fs-write" → "file_write") resolve to the canonical OpenFang name.
    let tool_name = normalize_tool_name(tool_name);

    // Capability enforcement: reject tools not in the allowed list
    if let Some(allowed) = allowed_tools {
        if !allowed.iter().any(|t| t == tool_name) {
            warn!(tool_name, "Capability denied: tool not in allowed list");
            return ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: format!(
                    "Permission denied: agent does not have capability to use tool '{tool_name}'"
                ),
                is_error: true,
            };
        }
    }

    // Approval gate: check if this tool requires human approval before execution.
    //
    // When exec_policy.mode = "full" (or allowlist with allowed_commands = ["*"]),
    // the user has explicitly opted into unrestricted shell access. In that case,
    // shell_exec should bypass the approval gate — requiring approval for commands
    // the user already whitelisted is contradictory (GitHub issue #772).
    let exec_policy_bypasses_approval = is_shell_tool(tool_name)
        && exec_policy.is_some_and(|p| {
            p.mode == openfang_types::config::ExecSecurityMode::Full
                || (p.mode == openfang_types::config::ExecSecurityMode::Allowlist
                    && p.allowed_commands.iter().any(|c| c == "*"))
        });

    if exec_policy_bypasses_approval {
        debug!(
            tool_name,
            "Approval bypassed: exec_policy grants unrestricted shell access"
        );
    }

    // #772 follow-up: in allowlist mode, suppress the approval prompt when
    // every base command is pre-approved via safe_bins or trusted_commands.
    // Uses the SAME extraction as the allowlist wall (fail-closed): an
    // unapproved, empty, or unparseable command yields None and falls through
    // to the prompt, where the wall re-validates. Logged, never surfaced to
    // Discord.
    let auto_approved = (!exec_policy_bypasses_approval && is_shell_tool(tool_name))
        .then_some(exec_policy)
        .flatten()
        .and_then(|p| {
            input
                .get("command")
                .and_then(|v| v.as_str())
                .map(|c| (p, c))
        })
        .and_then(|(p, c)| crate::subprocess_sandbox::command_approval_report(c, p));

    if let Some(report) = &auto_approved {
        let bases: Vec<String> = report
            .iter()
            .map(|b| format!("{}={:?}", b.base, b.via))
            .collect();
        info!(
            tool_name,
            bases = %bases.join(","),
            "Approval auto-granted: all command bases pre-approved (safe_bins/trusted_commands)"
        );
    }

    // SECURITY: For shell_exec, enforce the exec allowlist BEFORE the approval
    // gate. A command the allowlist wall will reject must never surface an
    // approval prompt or populate the approve-similar cache — otherwise the
    // operator sees an "Approved · cached" stamp for a command the very next
    // layer hard-denies (the `whoami` incident, 2026-06-17). Full mode is
    // unaffected: validate_command_allowlist returns Ok(()) immediately in Full,
    // so control falls through and the no-prompt guarantee still comes solely
    // from `exec_policy_bypasses_approval` above (which this block does not
    // touch). The canonical wall in the shell_exec match arm is retained as
    // idempotent defense-in-depth.
    if tool_name == "shell_exec" {
        if let Some(policy) = exec_policy {
            let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if let Err(reason) =
                crate::subprocess_sandbox::validate_command_allowlist(command, policy)
            {
                let reason = reason.trim_end_matches('.');
                return ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: format!(
                        "shell_exec blocked: {reason}. Current exec_policy.mode = '{:?}'. \
                         To allow shell commands, set exec_policy.mode = 'full' in the agent manifest or config.toml.",
                        policy.mode
                    ),
                    is_error: true,
                };
            }
        }
    }

    if let Some(kh) = kernel {
        if !exec_policy_bypasses_approval
            && auto_approved.is_none()
            && kh.requires_approval(tool_name)
        {
            let agent_id_str = caller_agent_id.unwrap_or("unknown");

            // ★ ANAI-154, layer 3.5: the gatekeeper sits between the
            // Approve-Similar cache and the human prompt. Everything that
            // reaches here has already cleared the hard-deny floor and the
            // allowlist wall, and has at least one base that comes only from
            // `allowed_commands` — otherwise `auto_approved` would be `Some`.
            //
            // The gate is inert unless the operator turned it on: the kernel's
            // impl returns `Escalate` when `[gatekeeper] enabled = false`, and
            // the trait default returns `Escalate` for every non-kernel handle.
            let gate_decision = crate::gatekeeper::review(
                kh,
                agent_id_str,
                tool_name,
                input,
                exec_policy,
                workspace_root,
                // ANAI-190: shell bypasses `file_policy` entirely — only
                // `file_write` and `apply_patch` route through `tier_for`. The
                // gate is therefore the one place that can ask whether a
                // command reaches past what the agent's own file tools grant.
                file_policy,
            )
            .await;

            // ANAI-241: the token joining this review to whatever the human
            // does next. `None` means the gate never ran (disabled, non-shell,
            // no exec policy) — the disposition row is still written, and says
            // so, because "approved with no judge involved" is a real fact.
            let gate_corr = gate_decision.as_ref().map(|d| d.correlation_id.clone());
            let gate_outcome = gate_decision.map(|d| d.outcome);

            // What the disposition rows below identify the command by.
            // Verbatim, same as the verdict row — a record whose dangerous
            // tail was cut is not a record (ANAI-151).
            let disposition_subject = input
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or(tool_name)
                .to_string();

            // `Deny` never prompts. The judge may only NARROW — it cannot
            // resurrect anything the allowlist wall already rejected — so a
            // deny here is strictly a refusal to spend the operator's attention
            // on something it considers plainly hostile.
            if let Some(crate::gatekeeper::GateOutcome::Deny(reason)) = &gate_outcome {
                // Recorded before the early return so the disposition table is
                // a complete census of gated commands on its own. Without this
                // row, "what fraction of gated commands reached a human" needs
                // a join against the verdict table to find the ones that never
                // could — and a denominator you have to remember to widen is a
                // denominator that gets reported wrong.
                crate::gatekeeper::record_disposition(
                    kh,
                    agent_id_str,
                    tool_name,
                    &disposition_subject,
                    gate_corr.as_deref(),
                    crate::gatekeeper::HitlDisposition::Decided(
                        openfang_types::approval::ApprovalDecision::GatekeeperDenied,
                    ),
                    0,
                );
                return ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: reason.clone(),
                    is_error: true,
                };
            }

            // Suppression skips the ENTIRE block below — including the
            // construction of `cache_binary`. That is structural, not
            // incidental: a gatekeeper suppression must never seed the
            // Approve-Similar cache, which blankets a whole binary for up to
            // `cache_max_uses` invocations. One judgement, one invocation.
            let suppressed = matches!(gate_outcome, Some(crate::gatekeeper::GateOutcome::Suppress));
            if suppressed {
                debug!(
                    tool_name,
                    "Approval suppressed by gatekeeper; executing without prompt"
                );
                // `wait_ms = 0` is not a placeholder — it is the measurement.
                // A suppression's whole value is the operator-latency it did
                // not spend, and this row is where that sits next to the
                // escalations that did spend it.
                //
                // Structurally unreachable while `shadow = true`: shadow
                // rewrites every verdict to `Escalate` before the caller sees
                // it. The first `gatekeeper_suppressed` row in the chain is
                // therefore the flip becoming real, with a timestamp.
                crate::gatekeeper::record_disposition(
                    kh,
                    agent_id_str,
                    tool_name,
                    &disposition_subject,
                    gate_corr.as_deref(),
                    crate::gatekeeper::HitlDisposition::Decided(
                        openfang_types::approval::ApprovalDecision::GatekeeperSuppressed,
                    ),
                    0,
                );
            }

            if !suppressed {
                let input_str = input.to_string();
                let summary = format!(
                    "{}: {}",
                    tool_name,
                    openfang_types::truncate_str(&input_str, 200)
                );
                // The judge declined to decide — say so on the prompt. An
                // operator who knows WHY the machine handed this back reads it
                // differently from one facing an undifferentiated Critical
                // banner, which is the fatigue this whole epic is about.
                //
                // ANAI-188: this rides its OWN field to the render site, not a
                // suffix on `summary`. Two reasons, and the second is the one
                // that mattered in the field. (1) `summary` is agent-controlled
                // — an annotation the operator reads as the machine's verdict
                // must not share a channel with text the requesting agent can
                // author. (2) `render_approval_body` prefers `command` over
                // `action_summary` for shell_exec (ANAI-151), so a suffix on
                // `summary` never reached the prompt at all: it surfaced only on
                // the post-approval resolution edit, which renders the summary.
                // The operator decided blind and read the judge's opinion after
                // clicking — the exact inversion of what annotating is for.
                let gatekeeper_note = match &gate_outcome {
                    Some(crate::gatekeeper::GateOutcome::Escalate(note)) => Some(note.clone()),
                    _ => None,
                };
                // Approve-Similar cache key source: argv[0] of a shell_exec
                // command, extracted once here where structured input still
                // exists (never re-parsed from the mangled action_summary).
                //
                // Same reasoning, same place, for the whole command string: the
                // operator decides from `command`, not from `summary` above, which
                // is serialized JSON cut at 200 bytes — precisely dropping the
                // argument tail that carries the risk (ANAI-151). Bounded to
                // MAX_COMMAND_LEN so a hostile agent cannot use the prompt as an
                // unbounded write into every downstream render surface.
                let raw_command = if tool_name == "shell_exec" {
                    input.get("command").and_then(|v| v.as_str())
                } else {
                    None
                };
                let cache_binary = raw_command.and_then(extract_cache_binary);
                let command = raw_command.map(|c| {
                    c.chars()
                        .take(openfang_types::approval::MAX_COMMAND_LEN)
                        .collect::<String>()
                });
                // ANAI-241: latency-to-human. Started here rather than at the
                // top of the gate because this is the only span the operator
                // is actually in — the judge's own latency is already recorded
                // separately on the verdict row, and folding the two together
                // would make model slowness indistinguishable from a human at
                // lunch.
                let prompt_started = std::time::Instant::now();
                let approval = kh
                    .request_approval(
                        agent_id_str,
                        tool_name,
                        &summary,
                        origin,
                        cache_binary.as_deref(),
                        command.as_deref(),
                        gatekeeper_note.as_deref(),
                    )
                    .await;
                let wait_ms = prompt_started.elapsed().as_millis();

                // One call, every arm, before any early return. The whole
                // point of this row is that it exists for commands whose
                // handling ends badly; writing it per-arm invites exactly the
                // omission that leaves a hole shaped like the interesting case.
                crate::gatekeeper::record_disposition(
                    kh,
                    agent_id_str,
                    tool_name,
                    &disposition_subject,
                    gate_corr.as_deref(),
                    match &approval {
                        Ok(d) => crate::gatekeeper::HitlDisposition::Decided(*d),
                        Err(_) => crate::gatekeeper::HitlDisposition::Error,
                    },
                    wait_ms,
                );

                match approval {
                    Ok(openfang_types::approval::ApprovalDecision::Approved) => {
                        debug!(tool_name, "Approval granted, proceeding with execution");
                    }
                    Ok(decision) => {
                        // ANAI-153: do not flatten these into one "denied". The
                        // outcome token is what makes a wave of timeouts (nobody
                        // watching, adapter down) greppable apart from a wave of
                        // real refusals, and `retryable` is what tells the agent
                        // whether abandoning the task is the correct response.
                        warn!(
                            tool_name,
                            outcome = decision.as_log_token(),
                            retryable = decision.is_retryable(),
                            "Approval not granted, blocking tool execution"
                        );
                        return ToolResult {
                            tool_use_id: tool_use_id.to_string(),
                            content: approval_outcome_message(tool_name, decision),
                            is_error: true,
                        };
                    }
                    Err(e) => {
                        warn!(tool_name, error = %e, "Approval system error");
                        return ToolResult {
                            tool_use_id: tool_use_id.to_string(),
                            content: format!("Approval system error: {e}"),
                            is_error: true,
                        };
                    }
                }
            }
        }
    }

    // file_policy prompt-tier pre-pass (D1-a / Q1=B): for single-path fs tools
    // whose resolved path falls under a `prompt` rule, route through the same
    // approval primitive the shell path uses. Approve => proceed (the path
    // helper treats an already-approved prompt as write); deny / timeout /
    // no-kernel => fail closed. apply_patch is multi-path and is governed
    // per-target in its helper (prompt-tier fails closed there for v1).
    let mut prevalidated_path: Option<PathBuf> = None;
    if let Some(fp) = file_policy {
        if fp.is_active() {
            if let Some((_needs_write, raw_path)) = fs_tool_single_path(tool_name, input) {
                if let Some(root) = workspace_root {
                    if let (Ok(canon), Ok(canon_root)) = (
                        crate::workspace_sandbox::sandbox_floor(raw_path, root),
                        root.canonicalize(),
                    ) {
                        if fp.tier_for(&canon, &canon_root)
                            == openfang_types::config::FileAccessTier::Prompt
                        {
                            let outcome: Option<openfang_types::approval::ApprovalDecision> =
                                match kernel {
                                    Some(kh) => {
                                        let summary = format!(
                                            "{} -> {} (file_policy prompt tier)",
                                            tool_name,
                                            canon.display()
                                        );
                                        kh.request_approval(
                                            caller_agent_id.unwrap_or("unknown"),
                                            tool_name,
                                            &summary,
                                            None,
                                            None,
                                            // Not a shell_exec path: there is no
                                            // command string to show (ANAI-151).
                                            None,
                                            // ...and the gatekeeper only sits on
                                            // the shell path, so there is no
                                            // verdict to annotate (ANAI-188).
                                            None,
                                        )
                                        .await
                                        .ok()
                                    }
                                    None => None,
                                };
                            // Fail closed: anything that is not an explicit
                            // Approved blocks. ANAI-153 changes only what the
                            // agent is TOLD about why, never whether it runs.
                            if outcome != Some(openfang_types::approval::ApprovalDecision::Approved)
                            {
                                warn!(
                                    tool_name,
                                    outcome =
                                        outcome.map(|d| d.as_log_token()).unwrap_or("unavailable"),
                                    retryable = outcome.map(|d| d.is_retryable()).unwrap_or(false),
                                    "file_policy prompt tier not approved, blocking tool execution"
                                );
                                return ToolResult {
                                    tool_use_id: tool_use_id.to_string(),
                                    content: match outcome {
                                        Some(d) => format!(
                                            "{} (target is a prompt-tier path under file_policy)",
                                            approval_outcome_message(tool_name, d)
                                        ),
                                        None => format!(
                                            "Execution blocked: '{tool_name}' targets a prompt-tier \
                                             path but the approval system was unavailable. This is \
                                             NOT a denial; no human judged the request. The \
                                             operation was not performed."
                                        ),
                                    },
                                    is_error: true,
                                };
                            }
                        }
                        // F5: remember the canonical path validated (and prompt-
                        // approved) here so the tool can assert its own I/O
                        // target matches — a swap during the approval window
                        // makes them differ and fails closed.
                        prevalidated_path = Some(canon);
                    }
                }
            }
        }
    }

    debug!(tool_name, "Executing tool");
    let result = match tool_name {
        // Filesystem tools
        "file_read" => {
            tool_file_read(
                input,
                workspace_root,
                file_policy,
                prevalidated_path.as_deref(),
            )
            .await
        }
        "file_write" => {
            tool_file_write(
                input,
                workspace_root,
                file_policy,
                prevalidated_path.as_deref(),
                caller_agent_id,
            )
            .await
        }
        "file_list" => {
            tool_file_list(
                input,
                workspace_root,
                file_policy,
                prevalidated_path.as_deref(),
            )
            .await
        }
        "file_grep" => {
            tool_file_grep(
                input,
                workspace_root,
                file_policy,
                prevalidated_path.as_deref(),
            )
            .await
        }
        // ANAI-297. The limit is operator policy from `[media]`; with no
        // media engine in scope (tests, bare call paths) the compiled default
        // applies, which is also what an unconfigured host gets.
        "image_read" => {
            let limit = media_engine
                .map(|e| e.config().effective_image_read_max_bytes())
                .unwrap_or_else(|| {
                    openfang_types::media::MediaConfig::default().effective_image_read_max_bytes()
                });
            if let Some(requested) = limit.clamped_from {
                warn!(
                    requested,
                    enforced = limit.max_bytes,
                    "image_read: [media] image_read_max_bytes is outside the valid band; clamped"
                );
            }
            tool_image_read(
                input,
                workspace_root,
                file_policy,
                prevalidated_path.as_deref(),
                limit,
            )
            .await
        }
        "create_directory" => {
            tool_create_directory(
                input,
                workspace_root,
                file_policy,
                prevalidated_path.as_deref(),
            )
            .await
        }
        "apply_patch" => {
            tool_apply_patch(input, workspace_root, file_policy, caller_agent_id).await
        }

        // File conversion tool (recipe-driven, allowlisted formats)
        "file_convert" => tool_file_convert(input, workspace_root, file_policy).await,

        // Web tools (upgraded: multi-provider search, SSRF-protected fetch)
        "web_fetch" => {
            // Taint check: block URLs containing secrets/PII from being exfiltrated
            let url = input["url"].as_str().unwrap_or("");
            if let Some(violation) = check_taint_net_fetch(url) {
                return ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: format!("Taint violation: {violation}"),
                    is_error: true,
                };
            }
            let method = input["method"].as_str().unwrap_or("GET");
            let headers = input.get("headers").and_then(|v| v.as_object());
            let body = input["body"].as_str();
            if let Some(ctx) = web_ctx {
                ctx.fetch
                    .fetch_with_options(url, method, headers, body)
                    .await
            } else {
                tool_web_fetch_legacy(input).await
            }
        }
        "web_search" => {
            if let Some(ctx) = web_ctx {
                let query = input["query"].as_str().unwrap_or("");
                let max_results = input["max_results"].as_u64().unwrap_or(5) as usize;
                ctx.search.search(query, max_results).await
            } else {
                tool_web_search_legacy(input).await
            }
        }

        // Shell tool — metacharacter check + exec policy + taint check
        "shell_exec" => {
            let command = input["command"].as_str().unwrap_or("");

            // SECURITY: Always check for shell metacharacters, even in Full mode.
            // These enable command injection regardless of exec policy.
            if let Some(reason) = crate::subprocess_sandbox::contains_shell_metacharacters(command)
            {
                return ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: format!(
                        "shell_exec blocked: command contains {reason}. \
                         Shell metacharacters are never allowed."
                    ),
                    is_error: true,
                };
            }

            // Exec policy enforcement (allowlist / deny / full)
            if let Some(policy) = exec_policy {
                if let Err(reason) =
                    crate::subprocess_sandbox::validate_command_allowlist(command, policy)
                {
                    let reason = reason.trim_end_matches('.');
                    return ToolResult {
                        tool_use_id: tool_use_id.to_string(),
                        content: format!(
                            "shell_exec blocked: {reason}. Current exec_policy.mode = '{:?}'. \
                             To allow shell commands, set exec_policy.mode = 'full' in the agent manifest or config.toml.",
                            policy.mode
                        ),
                        is_error: true,
                    };
                }
            }
            // Skip heuristic taint patterns for Full exec policy (e.g. hand agents that need curl)
            let is_full_exec = exec_policy
                .is_some_and(|p| p.mode == openfang_types::config::ExecSecurityMode::Full);
            if !is_full_exec {
                if let Some(violation) = check_taint_shell_exec(command) {
                    return ToolResult {
                        tool_use_id: tool_use_id.to_string(),
                        content: format!("Taint violation: {violation}"),
                        is_error: true,
                    };
                }
            }
            // A shell command is opaque: we cannot know which files it wrote.
            // So snapshot the workspace's context files, run it, then diff.
            // Reconcile runs on the error path too -- a command that fails
            // part-way can still have rewritten SOUL.md. Auditing never gates
            // the command and never changes its result.
            let context_snapshot = crate::context_audit::snapshot_workspace(workspace_root).await;
            let shell_result = tool_shell_exec(
                input,
                allowed_env_vars.unwrap_or(&[]),
                workspace_root,
                exec_policy,
            )
            .await;
            crate::context_audit::reconcile_workspace(
                caller_agent_id,
                "shell_exec",
                context_snapshot,
            )
            .await;
            shell_result
        }

        // Inter-agent tools (require kernel handle)
        "agent_send" => tool_agent_send(input, kernel, caller_agent_id).await,
        "agent_send_async" => tool_agent_send_async(input, kernel, caller_agent_id).await,
        "agent_reply_async" => tool_agent_reply_async(input, kernel, caller_agent_id).await,
        "agent_spawn" => tool_agent_spawn(input, kernel, caller_agent_id).await,
        "agent_list" => tool_agent_list(kernel),
        "agent_kill" => tool_agent_kill(input, kernel),
        "agent_activate" => tool_agent_activate(input, kernel),

        // Memory tools (agent-scoped; `shared:` prefix opts into cross-agent)
        "memory_store" => tool_memory_store(input, kernel, caller_agent_id),
        "memory_recall" => tool_memory_recall(input, kernel, caller_agent_id).await,
        "memory_note" => tool_memory_note(input, kernel, caller_agent_id).await,
        "memory_episode_close" => tool_memory_episode_close(input, kernel, caller_agent_id).await,
        "memory_status" => tool_memory_status(kernel, caller_agent_id),
        "memory_fact" => tool_memory_fact(input, kernel, caller_agent_id).await,
        "memory_history" => tool_memory_history(input, kernel, caller_agent_id),

        // Collaboration tools
        "agent_find" => tool_agent_find(input, kernel),
        "task_post" => tool_task_post(input, kernel, caller_agent_id).await,
        "task_claim" => tool_task_claim(kernel, caller_agent_id).await,
        "task_complete" => tool_task_complete(input, kernel).await,
        "task_list" => tool_task_list(input, kernel).await,
        "event_publish" => tool_event_publish(input, kernel).await,

        // Scheduling tools
        "schedule_create" => tool_schedule_create(input, kernel, caller_agent_id).await,
        "schedule_list" => tool_schedule_list(kernel, caller_agent_id).await,
        "schedule_delete" => tool_schedule_delete(input, kernel).await,

        // Knowledge graph tools
        "knowledge_add_entity" => tool_knowledge_add_entity(input, kernel).await,
        "knowledge_add_relation" => tool_knowledge_add_relation(input, kernel).await,
        "knowledge_query" => tool_knowledge_query(input, kernel).await,

        // Image analysis tool
        "image_analyze" => tool_image_analyze(input).await,

        // Media understanding tools
        "media_describe" => tool_media_describe(input, media_engine).await,
        "media_transcribe" => tool_media_transcribe(input, media_engine).await,

        // Image generation tool
        "image_generate" => tool_image_generate(input, workspace_root, media_engine).await,

        // TTS/STT tools
        "text_to_speech" => tool_text_to_speech(input, tts_engine, workspace_root).await,
        "speech_to_text" => tool_speech_to_text(input, media_engine, workspace_root).await,

        // Docker sandbox tool
        "docker_exec" => {
            tool_docker_exec(input, docker_config, workspace_root, caller_agent_id).await
        }

        // Location tool
        "location_get" => tool_location_get().await,

        // System time tool
        "system_time" => Ok(tool_system_time()),

        // Cron scheduling tools
        "cron_create" => tool_cron_create(input, kernel, caller_agent_id).await,
        "cron_list" => tool_cron_list(kernel, caller_agent_id).await,
        "cron_cancel" => tool_cron_cancel(input, kernel).await,

        // Channel send tool (proactive outbound messaging)
        "channel_send" => tool_channel_send(input, kernel, workspace_root).await,

        // Persistent process tools
        "process_start" => {
            tool_process_start(input, process_manager, caller_agent_id, exec_policy).await
        }
        "process_poll" => tool_process_poll(input, process_manager).await,
        "process_write" => tool_process_write(input, process_manager).await,
        "process_kill" => tool_process_kill(input, process_manager).await,
        "process_list" => tool_process_list(process_manager, caller_agent_id).await,

        // Hand tools (curated autonomous capability packages)
        "hand_list" => tool_hand_list(kernel).await,
        "hand_activate" => tool_hand_activate(input, kernel).await,
        "hand_status" => tool_hand_status(input, kernel).await,
        "hand_deactivate" => tool_hand_deactivate(input, kernel).await,

        // A2A outbound tools (cross-instance agent communication)
        "a2a_discover" => tool_a2a_discover(input).await,
        "a2a_send" => tool_a2a_send(input, kernel).await,

        // Browser automation tools
        "browser_navigate" => {
            let url = input["url"].as_str().unwrap_or("");
            if let Some(violation) = check_taint_net_fetch(url) {
                return ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: format!("Taint violation: {violation}"),
                    is_error: true,
                };
            }
            match browser_ctx {
                Some(mgr) => {
                    let aid = caller_agent_id.unwrap_or("default");
                    crate::browser::tool_browser_navigate(input, mgr, aid).await
                }
                None => Err(
                    "Browser tools not available. Ensure Chrome/Chromium is installed.".to_string(),
                ),
            }
        }
        "browser_click" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_click(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_type" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_type(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_screenshot" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_screenshot(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_read_page" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_read_page(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_close" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_close(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_scroll" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_scroll(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_wait" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_wait(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_run_js" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_run_js(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },
        "browser_back" => match browser_ctx {
            Some(mgr) => {
                let aid = caller_agent_id.unwrap_or("default");
                crate::browser::tool_browser_back(input, mgr, aid).await
            }
            None => {
                Err("Browser tools not available. Ensure Chrome/Chromium is installed.".to_string())
            }
        },

        // Skill introspection tools (issue #1038)
        "skill_list" => tool_skill_list(skill_registry),
        "skill_describe" => tool_skill_describe(input, skill_registry),
        "skill_execute" => tool_skill_execute(input, skill_registry).await,

        // Canvas / A2UI tool
        "canvas_present" => tool_canvas_present(input, workspace_root).await,

        other => {
            // Fallback 1: MCP tools (mcp_{server}_{tool} prefix)
            if mcp::is_mcp_tool(other) {
                if let Some(mcp_conns) = mcp_connections {
                    let mut conns = mcp_conns.lock().await;
                    let known_names: Vec<String> =
                        conns.iter().map(|c| c.name().to_string()).collect();
                    let known_refs: Vec<&str> = known_names.iter().map(|s| s.as_str()).collect();
                    if let Some(server_name) =
                        mcp::extract_mcp_server_from_known(other, &known_refs)
                    {
                        if let Some(conn) = conns.iter_mut().find(|c| c.name() == server_name) {
                            debug!(
                                tool = other,
                                server = server_name,
                                "Dispatching to MCP server"
                            );
                            match conn.call_tool(other, input).await {
                                Ok(content) => Ok(content),
                                Err(e) => Err(format!("MCP tool call failed: {e}")),
                            }
                        } else {
                            Err(format!("MCP server '{server_name}' not connected"))
                        }
                    } else {
                        Err(format!("Invalid MCP tool name: {other}"))
                    }
                } else {
                    Err(format!("MCP not available for tool: {other}"))
                }
            }
            // Fallback 2: Skill registry tool providers
            else if let Some(registry) = skill_registry {
                if let Some(skill) = registry.find_tool_provider(other) {
                    debug!(tool = other, skill = %skill.manifest.skill.name, "Dispatching to skill");
                    match openfang_skills::loader::execute_skill_tool(
                        &skill.manifest,
                        &skill.path,
                        other,
                        input,
                    )
                    .await
                    {
                        Ok(skill_result) => {
                            let content = serde_json::to_string(&skill_result.output)
                                .unwrap_or_else(|_| skill_result.output.to_string());
                            if skill_result.is_error {
                                Err(content)
                            } else {
                                Ok(content)
                            }
                        }
                        Err(e) => Err(format!("Skill execution failed: {e}")),
                    }
                } else {
                    Err(format!("Unknown tool: {other}"))
                }
            } else {
                Err(format!("Unknown tool: {other}"))
            }
        }
    };

    match result {
        Ok(content) => ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content,
            is_error: false,
        },
        Err(err) => ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content: format!("Error: {err}"),
            is_error: true,
        },
    }
}

/// The `memory_note` doctrine, in one place (ANAI-270).
///
/// Duplicated byte-for-byte in `openfang-mcp-bridge` (the seam is one-way, see
/// `EPISODE_CLOSE_DESCRIPTION`) and pinned equal by
/// `openfang-api/tests/memory_note_doctrine_drift_test.rs`.
///
/// Step 0 of ANAI-270 read the 40 closest same-author note pairs: 20 were full
/// replacements, and of those 8 were duplicates written minutes apart, 8 were
/// status updates that should have been facts, and only 4 were genuine
/// corrections. So the description carries three rules, in that order of
/// yield: a moving value is a fact; a note you can see and are replacing is
/// named in `supersedes`; and replacement is whole-note (Ben's ruling,
/// 2026-09-23) — a partial correction restates everything still true.
pub const MEMORY_NOTE_DESCRIPTION: &str = "Jot something down in your own memory, in your own words - an observation, a lesson, a decision and why, something that will still be true next week. Cheap and unstructured; no key needed. It is attached to your current episode. If what you are writing has a CURRENT VALUE that will move - a status, an owner, a commit, a count, a progress marker like 'step 1 done, next is step 2' - use memory_fact instead: rewriting a fact slot replaces the old value, while a second note leaves the stale one surfacing beside it. If this note corrects or replaces one of your own notes that you can see, name it in 'supersedes' so the old one stops surfacing. Replacement is whole-note: if only part of the old note is wrong, this note must carry the corrected part AND everything from the old note that is still true, because the old note stops surfacing in full. A note written without 'supersedes' answers with your closest existing notes - if one of them already says this, merge them as the reply describes.";

/// The `supersedes` parameter's description (ANAI-270). Pinned equal to the
/// bridge's copy by the same drift test as `MEMORY_NOTE_DESCRIPTION`.
pub const MEMORY_NOTE_SUPERSEDES_DESCRIPTION: &str = "Optional: ids of your own notes that this note corrects or replaces, e.g. [\"1a2b3c4d\"] - the id is shown in a recalled note's tag, [note · 2d · id:1a2b3c4d]. Those notes stop surfacing in recall. Replacement is whole-note: if only part of an old note was wrong, this note must carry the corrected part AND everything from the old note that is still true.";

/// The `memory_episode_close` doctrine, in one place (ANAI-283).
///
/// Three surfaces teach an agent when to close: this description, the
/// `## Memory` bullet in `prompt_builder`, and the bridge's copy in
/// `openfang-mcp-bridge` — which is what a SUBPROCESS agent actually reads.
/// They disagreed until now, and the model resolves a disagreement toward the
/// one attached to the button. The bridge cannot depend on this crate (the
/// seam is one-way on purpose), so it still carries its own literal; the
/// equality is pinned instead by a cross-crate test in `openfang-api`, the one
/// crate that sees both.
///
/// The trigger is deliberately a TOPIC SHIFT and not completion. ANAI-248 shipped
/// the completion wording to 53 manifests and the fleet produced 1 voluntary
/// close in three weeks against 210 timer closes: "is this finished?" asks the
/// agent to notice an ABSENCE, mid-task, on a turn whose job is something else.
/// "Does this message name work other than what I have been doing?" is readable
/// off two things the agent is already holding — so the cue moved into the
/// incoming turn, and the check moved to the top of it.
///
/// The bias against firing moved with it rather than being deleted. It now sits
/// on `reset_context`, which is the half that can actually cost something: a
/// close in a slightly odd place is a mislabelled boundary, while a reset on a
/// live thread is a window. So: close liberally, reset conservatively.
pub const EPISODE_CLOSE_DESCRIPTION: &str = "Close the current episode - the stretch of turns your recent work is grouped into - and label it. Check this BEFORE you start work on a turn, not after: if the incoming message moves you to different work - another project, another repo, another person's business, or an explicit \"let's switch to\" - the previous episode is over, and closing it first is what puts the new work in the new episode instead of the old one. Work reaching its end is also a close: a ticket landed, a question answered, a decision made. NOT a topic change: a question about what you just did, a digression that returns, a new ticket in the same project, or \"also, can you\". A long gap since the last message plus a different subject is two signals, not one - treat it as a change. Close on a plausible shift; a missed close is not free, because the boundary then lands hours late on the idle timeout, in the middle of the next topic. It is reset_context that deserves the caution, not the close: when you are unsure the old thread is finished, close WITHOUT reset_context - you keep your window and still get the boundary. Never reset mid-task, nor while something is unverified or a question to the operator is outstanding. Name the reason - \"topic-switch\" when the subject changed, \"explicit\" when the work finished or the operator asked. A new episode opens on your next turn. Harmless to call when nothing is open. Pass reset_context to also start the next episode with a clean conversation window, and prime_for to have that fresh window opened with what durable memory knows about the project you are moving to.";

/// Advertised description for `file_read`, shared with the bridge's
/// `built_in_tools()` so the two surfaces cannot drift (ANAI-291).
///
/// Says the two things a caller cannot discover by trying: that a large file
/// comes back as a head plus a manifest rather than whole, and that the
/// line-denominated `offset`/`limit` take exactly the numbers `file_grep`
/// hands back. The units agreeing across both tools is the whole reason they
/// are one design.
pub const FILE_READ_DESCRIPTION: &str = "Read the contents of a file. Paths are relative to the agent workspace. Use offset and limit to read a bounded window instead of the whole file: both are LINE numbers, 1-based, and they take the line numbers file_grep returns verbatim. A file too large to return whole comes back as its first 200 lines plus a manifest stating the total line count, the file's sha256, and the exact call that returns the next slice - so a large read is never a silent truncation, but it is also not the file. For anything big, searching with file_grep and then reading the range it points at costs far less context than paging through.";

/// Advertised description for `file_grep` (ANAI-292). Shared with the bridge's
/// copy and pinned equal by a cross-crate test, for the same reason
/// `FILE_READ_DESCRIPTION` is.
pub const FILE_GREP_DESCRIPTION: &str = "Search a file, or recursively a directory, for a regular expression and get back the matching LINE NUMBERS with their text. Paths are relative to the agent workspace. The line numbers are 1-based and can be passed straight to file_read's offset, which is the point: for anything large, grep for the anchor and then read that range, instead of pulling a whole file into context. Exposes strictly less than file_read already does, and resolves every path through the same policy, so it reaches nothing file_read would refuse. Every bound that bites is disclosed in the result - the match cap, the file cap, skipped binaries, and the build/VCS directories not descended into - because a silent cap reads as an absence of matches.";

/// Advertised description for `image_read` (ANAI-297). Shared with the
/// bridge's copy and pinned equal by a cross-crate test, like the two above.
///
/// States what a caller cannot discover by trying: that the format is decided
/// by the bytes, that the size limit refuses rather than truncates, and that
/// SVG belongs to `file_read`.
pub const IMAGE_READ_DESCRIPTION: &str = "Look at an image file. Returns the image itself (PNG, JPEG, GIF or WebP) as an image you can see, plus one line of text giving its type, size and sha256. Paths are relative to the agent workspace and resolve through the same file policy as file_read, so it reaches exactly the files file_read reaches. The type is decided by the file's bytes, not its name. A file over the operator's size limit ([media] image_read_max_bytes, default 3,750,000 bytes) is refused whole, never truncated. SVG is text: read it with file_read.";

/// `image_read`'s advertised argument schema (ANAI-297). Duplicated in the
/// bridge and pinned equal by a cross-crate test in `openfang-api`.
pub fn image_read_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "The image file to read" }
        },
        "required": ["path"]
    })
}

/// `file_grep`'s advertised argument schema. Duplicated in the bridge (the
/// crate seam is one-way) and pinned equal by a cross-crate test in
/// `openfang-api`.
pub fn file_grep_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "File to search, or directory to search recursively" },
            "pattern": { "type": "string", "description": "Regular expression to search for" },
            "ignore_case": { "type": "boolean", "description": "Match case-insensitively. Default false." },
            "context": { "type": "integer", "description": "Lines of surrounding context to include per match, 0-20. Default 0. Context lines are marked with '-' and matches with ':'." },
            "max_matches": { "type": "integer", "description": "Stop after this many matches. Default 100, ceiling 2000. Hitting it is disclosed in the result and is NOT a total." },
            "max_files": { "type": "integer", "description": "Stop enumerating after this many files in a directory search. Default 400, ceiling 5000." },
            "include": { "type": "string", "description": "Filename glob limiting which files are searched, e.g. \"*.rs\". Only '*' is a wildcard; everything else matches literally." }
        },
        "required": ["path", "pattern"]
    })
}

/// Get definitions for all built-in tools.
pub fn builtin_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // --- Filesystem tools ---
        ToolDefinition {
            name: "file_read".to_string(),
            description: FILE_READ_DESCRIPTION.to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to read" },
                    "offset": { "type": "integer", "description": "1-based line number to start at. Omit to start at line 1." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return. Omit to read to the end of the file." }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "file_write".to_string(),
            description: "Write content to a file. Paths are relative to the agent workspace.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to write to" },
                    "content": { "type": "string", "description": "The content to write" }
                },
                "required": ["path", "content"]
            }),
        },
        ToolDefinition {
            name: "file_list".to_string(),
            description: "List files in a directory. Paths are relative to the agent workspace.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The directory path to list" }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "file_grep".to_string(),
            description: FILE_GREP_DESCRIPTION.to_string(),
            input_schema: file_grep_input_schema(),
        },
        ToolDefinition {
            name: "image_read".to_string(),
            description: IMAGE_READ_DESCRIPTION.to_string(),
            input_schema: image_read_input_schema(),
        },
        ToolDefinition {
            name: "create_directory".to_string(),
            description: "Create a directory (and any missing parent directories) at the given path. Paths are relative to the agent workspace. Idempotent: succeeds if the directory already exists.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The directory path to create" }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "apply_patch".to_string(),
            description: "Apply a multi-hunk diff patch to add, update, move, or delete files. Use this for targeted edits instead of full file overwrites.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The patch in *** Begin Patch / *** End Patch format. Use *** Add File:, *** Update File:, *** Delete File: markers. Hunks use @@ headers with space (context), - (remove), + (add) prefixed lines."
                    }
                },
                "required": ["patch"]
            }),
        },
        ToolDefinition {
            name: "file_convert".to_string(),
            description: "Convert a file from one format to another using an allowlisted recipe table (e.g. Markdown to PDF). The source format is inferred from the input file extension; the target format is the 'format' argument. Only conversions defined in the recipe manifest are permitted. Paths are relative to the agent workspace; absolute paths follow the same file_policy as file_read (the input needs read access, the output needs write access). Returns the output path, not its content: file_read it afterwards.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "format": { "type": "string", "description": "Target format / output extension, e.g. \"pdf\"" },
                    "input": { "type": "string", "description": "Path to the source file, relative to the workspace or absolute. It must be readable under your file_policy, exactly as for file_read. Its extension determines the source format." },
                    "output": { "type": "string", "description": "Optional output path, relative to the workspace or absolute. It must be writable under your file_policy. If omitted, the input path with the target extension is used, so for an input you can only read, pass an output you can write." },
                    "preset": { "type": "string", "description": "Optional render preset selecting size/scale, e.g. \"mobile\", \"tablet\", \"desktop\", \"wide\". Must be one offered by the target recipe; omit to use the recipe's default preset. Ignored by recipes that define no presets." },
                    "options": file_convert_options_schema()
                },
                "required": ["format", "input"]
            }),
        },
        // --- Web tools ---
        ToolDefinition {
            name: "web_fetch".to_string(),
            description: "Fetch a URL with SSRF protection. Supports GET/POST/PUT/PATCH/DELETE. For GET, HTML is converted to Markdown. For other methods, returns raw response body.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch (http/https only)" },
                    "method": { "type": "string", "enum": ["GET","POST","PUT","PATCH","DELETE"], "description": "HTTP method (default: GET)" },
                    "headers": { "type": "object", "description": "Custom HTTP headers as key-value pairs" },
                    "body": { "type": "string", "description": "Request body for POST/PUT/PATCH" }
                },
                "required": ["url"]
            }),
        },
        ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web using multiple providers (Tavily, Brave, Perplexity, DuckDuckGo) with automatic fallback. Returns structured results with titles, URLs, and snippets.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" },
                    "max_results": { "type": "integer", "description": "Maximum number of results to return (default: 5, max: 20)" }
                },
                "required": ["query"]
            }),
        },
        // --- Shell tool ---
        ToolDefinition {
            name: "shell_exec".to_string(),
            description: "Execute a shell command and return its output.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to execute" },
                    "timeout_seconds": { "type": "integer", "description": "Timeout in seconds (default: 30)" }
                },
                "required": ["command"]
            }),
        },
        // --- Inter-agent tools ---
        ToolDefinition {
            name: "agent_send".to_string(),
            description: "Send a message to another agent and receive their response. Accepts UUID or agent name. Use agent_find first to discover agents.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The target agent's UUID or name" },
                    "message": { "type": "string", "description": "The message to send to the agent" }
                },
                "required": ["agent_id", "message"]
            }),
        },
        ToolDefinition {
            name: "agent_send_async".to_string(),
            description: "Wake another agent asynchronously (fire-and-forget). Queues the \
                          message for the target and returns immediately — the caller does NOT \
                          block on the target's loop and receives NO inline reply. Use this \
                          instead of agent_send when you want to hand off work without waiting, \
                          or to avoid the head-of-line blocking of a synchronous A->B call. \
                          Accepts UUID or agent name."
                .to_string()
                + " Optionally pass surface_to (\"<channel>:<recipient>\") to have the target's \
                   eventual agent_reply_async answer auto-posted to that channel."
                + " You are GUARANTEED exactly one reply per call: if the target answers, you \
                   get its answer; if it cannot or does not, the daemon closes the correlation \
                   itself and tells you why. Pass timeout_secs to bound how long that takes."
                + " Pass requires_tools to refuse the send outright when the target lacks a \
                   tool the work needs, instead of spending the whole deadline discovering it.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The target agent's UUID or name to wake" },
                    "message": { "type": "string", "description": "The message delivered to the target when it runs" },
                    "surface_to": { "type": "string", "description": "Optional channel route, formatted \"<channel>:<recipient>\" (e.g. \"discord:1086446153098342510\"). When set, the target's one-shot agent_reply_async answer is auto-posted to this channel by the daemon. Omit for a pure fire-and-forget wake with no surfacing." },
                    "timeout_secs": { "type": "integer", "description": "Optional deadline in seconds. You are guaranteed a reply within roughly this long: if the target has not answered by then, its turn is ABORTED and the daemon sends you a timeout reply instead. Set it to how long you actually expect the work to take, with headroom — an over-tight value kills legitimate work, and partial side effects from the aborted turn may persist. Clamped into the operator's configured band; omit to accept the configured default." }
                    ,"requires_tools": { "type": "array", "items": { "type": "string" }, "description": "Optional pre-flight: tool names the target MUST have for this work (e.g. [\"shell_exec\"]). Checked BEFORE the wake is enqueued — if any is missing the call fails immediately, names what is missing, mints NO correlation and consumes no deadline, so nothing was sent and re-sending a corrected request is safe. Use it whenever the request depends on a specific capability; without it you spend the full deadline learning the target could never have done it. Omit for no check." }
                },
                "required": ["agent_id", "message"]
            }),
        },
        ToolDefinition {
            name: "agent_reply_async".to_string(),
            description: "Send a ONE-SHOT terminal reply to the agent that woke you via \
                          agent_send_async (fire-and-forget). Valid ONLY inside a turn that \
                          another agent woke asynchronously — it answers that initiator and no \
                          one else, so it takes no target (you cannot choose who to reply to). \
                          The reply is terminal: the initiator receives your message as its own \
                          woken turn and surfaces or continues, but cannot bounce back. Outside \
                          a woken turn, or after you have already replied once this turn, the \
                          call refuses. Use this to complete a delegated async round-trip."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The reply delivered to your initiator when it runs" }
                },
                "required": ["message"]
            }),
        },
        ToolDefinition {
            name: "agent_spawn".to_string(),
            description: "Spawn a new agent from a TOML manifest. Returns the new agent's ID and name.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "manifest_toml": {
                        "type": "string",
                        "description": "The agent manifest in TOML format (must include name, module, [model], and [capabilities])"
                    }
                },
                "required": ["manifest_toml"]
            }),
        },
        ToolDefinition {
            name: "agent_list".to_string(),
            description: "List all currently running agents with their IDs, names, states, and models.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "agent_kill".to_string(),
            description: "Kill (terminate) another agent by its ID.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The agent's UUID to kill" }
                },
                "required": ["agent_id"]
            }),
        },
        ToolDefinition {
            name: "agent_activate".to_string(),
            description: "Activate (wake up) an inactive agent so it can receive messages \
                          and process events. Use this when agent_list shows an agent in a \
                          Suspended, Crashed, or Created state and you want to delegate work \
                          to it via agent_send. Terminated agents cannot be revived."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The target agent's UUID or human-readable name"
                    }
                },
                "required": ["agent_id"]
            }),
        },
        // --- Memory tools (ANAI-165: agent-scoped by default) ---
        ToolDefinition {
            name: "memory_store".to_string(),
            description: "Store a value in YOUR OWN memory namespace, private to you. Prefix the key with 'shared:' to write to the cross-agent namespace instead (e.g. 'shared:release_freeze') - use that only for state other agents genuinely need to read.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "The storage key. Prefix with 'shared:' for cross-agent state." },
                    "value": { "type": "string", "description": "The value to store (JSON-encode objects/arrays, or pass a plain string)" }
                },
                "required": ["key", "value"]
            }),
        },
        ToolDefinition {
            name: "memory_recall".to_string(),
            description: "Search YOUR OWN memory for relevant context: pass 'query' with what you are looking for, in words. Pass 'key' instead for an exact stored key ('shared:' prefix reads the cross-agent namespace). Exactly one of the two.".to_string(),
            // ANAI-166: schema evolution is ADDITIVE ONLY. `key` is never
            // removed and never repurposed; `required` drops to empty because
            // `anyOf` support is uneven across providers and the MCP bridge
            // re-serializes this schema. Exactly-one-of is therefore enforced
            // in the handler, with an error string that tells the caller what
            // to do. The bridge's copy of this schema
            // (`openfang-mcp-bridge/src/lib.rs`) must change in the same
            // commit — ANAI-126 is what happens when it does not.
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What you are looking for, in plain words. Searches by meaning when embeddings are configured, by text match otherwise." },
                    "key": { "type": "string", "description": "Exact storage key, for values written with memory_store. Prefix with 'shared:' for cross-agent state." },
                    "scope": { "type": "string", "description": "Optional: restrict to one memory scope, e.g. 'episodic'." },
                    "kind": { "type": "string", "description": "Optional: restrict to one kind of memory, e.g. 'note'." },
                    "limit": { "type": "integer", "description": "Maximum results to return (default 5, maximum 25)." }
                },
                "required": []
            }),
        },
        ToolDefinition {
            name: "memory_note".to_string(),
            description: MEMORY_NOTE_DESCRIPTION.to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "What to remember, in plain words." },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional short labels to help find this later." },
                    "supersedes": { "type": "array", "items": { "type": "string" }, "description": MEMORY_NOTE_SUPERSEDES_DESCRIPTION }
                },
                "required": ["text"]
            }),
        },
        // --- Episode tools (ANAI-194, ADR 0002 2.2) ---
        //
        // Grouped with the other memory tools rather than tail-appended: this
        // list is category-ordered and its drift test asserts membership, not
        // position. The FLAT bridge lists (`built_in_tools`, `ALLOWED_TOOLS`,
        // `DEFAULT_ALLOWED`) are tail-appended instead, because the bridge
        // surface test compares an exact ordered vec and a mid-list insert
        // there skews indices silently.
        ToolDefinition {
            name: "memory_episode_close".to_string(),
            description: EPISODE_CLOSE_DESCRIPTION.to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "Short label for the thread that just ended, e.g. \"git trunk cutover\"" },
                    "summary": { "type": "string", "description": "Optional few-sentence wrap-up of what happened and what was decided. It is kept as a note on this episode and fed to the summariser as material; the episode's own summary is always synthesized afterwards, never taken from here." },
                    "reason": { "type": "string", "enum": ["topic-switch", "explicit"], "description": "Why the episode is closing. Use 'topic-switch' when the incoming message moved you to different work, and 'explicit' when the work itself finished or the operator asked for a wrap-up. Defaults to 'explicit'. Timer closes are the system's and are not available to you. Name it honestly: this is the only record of whether the boundary came from a cue or from completion." },
                    "reset_context": { "type": "boolean", "description": "Default false. When true, your conversation window is cleared at the END of this turn so the next episode starts fresh. Your durable memory is untouched and the running summary of earlier work is kept - you will not forget what happened, you stop re-reading it verbatim. Only set this when the work really is finished; doing it mid-task discards the detail you still need. If you are weighing it up, the answer is no. Refused outright while you have an approval request outstanding to the operator." },
                    "prime_for": { "type": "string", "description": "Optional project slug, e.g. \"openfang\". Only meaningful with reset_context. The next episode opens with a short briefing assembled from durable memory for that project - your recently closed episodes and what the fleet currently believes about it - instead of you having to ask for it. Use dots to name a sub-project, \"openfang.memory\": the briefing then carries the sub-project's claims AND everything the parent knows, so being more specific never costs you facts. This is the project's slug, not your own agent name. Omitting it clears any previous priming." }
                },
                "required": ["title"]
            }),
        },
        ToolDefinition {
            name: "memory_status".to_string(),
            description: "Report the state of your own memory: which episode is open, how many turns it has captured, how long it has been idle, and when it will close on its own. Also lists every open claim slot you already hold - check those before writing a memory_fact, because correcting a slot you own supersedes it and keeps the history, while inventing a near-duplicate key splits one claim into two. Use it to notice you have drifted onto unrelated work.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        // --- Fact tools (ANAI-204, ADR 0001 2.3) ---
        //
        // Grouped with the other memory tools for the same reason the episode
        // tools are: this list is category-ordered and its drift test asserts
        // membership, not position. The FLAT bridge lists are tail-appended
        // instead, because the bridge surface test compares an exact ordered
        // vec.
        ToolDefinition {
            name: "memory_fact".to_string(),
            description: "Read or write one durable claim slot - a named box holding the CURRENT truth about something, overwritten in place when it changes. Pass 'claim' to write; omit it to read what is already there. Keys are 'namespace.slot', e.g. 'repo.trunk_model' or 'project.tttb.promotion_status'; the namespaces are agent, build, deploy, delivery, memory, project, repo, tool and user. Store state that gets updated, not events that happened - a ticket id or a date in the key means it belongs in memory_note instead. Read a slot before you write it: prefer a key that already exists over minting a near-duplicate. When you write, say how fast the claim rots with 'persistence_class' - a claim that goes unchecked past its class is surfaced later marked 'verify', so future readers ask instead of assert.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "scope": { "type": "string", "enum": ["agent", "project", "user"], "description": "Whose truth this is: 'agent' (about you), 'project', or 'user'." },
                    "scope_ref": { "type": "string", "description": "What the claim is about - the project or user slug, e.g. \"openfang\", or \"openfang.memory\" for a sub-project. Dots nest: a reader primed for \"openfang.memory\" also sees \"openfang\"'s claims, and the more specific slot wins where both hold the same key. File a claim at the level it is true of, and use the project's slug, not your own agent name. Required for 'project' and 'user'; ignored for 'agent', which is always you." },
                    "key": { "type": "string", "description": "The slot name, 'namespace.slot', e.g. \"repo.trunk_model\". Up to 7 dot-separated segments." },
                    "claim": { "type": "string", "description": "The claim itself, in plain words. Omit to READ the slot instead of writing it." },
                    "status": { "type": "string", "enum": ["open", "settled"], "description": "'settled' (default) for a stable belief; 'open' for an unfinished loop." },
                    "confidence": { "type": "number", "description": "How sure you are, 0.0 to 1.0. Defaults to 1.0." },
                    "persistence_class": { "type": "string", "enum": ["permanent", "stable", "active", "volatile"], "description": "How fast this claim rots, so a reader knows when to re-check it. 'permanent' never goes stale (a name, a table's name); 'stable' is good for months (architecture, ownership); 'active' is the default and is doubted after about a week; 'volatile' is doubted within a day (deploy state, a commit hash). Writing your own re-verification command into the claim text makes the marker actionable. If a claim rots in hours it is probably an event, not a slot - use memory_note." }
                },
                "required": ["scope", "key"]
            }),
        },
        ToolDefinition {
            name: "memory_history".to_string(),
            description: "Show every claim that has occupied a slot, newest first - what was believed, who asserted it, and when it stopped being true. The audit path for a fact whose current value looks wrong or surprising. Superseded claims never appear in ordinary recall, so this is the only way to see one.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "scope": { "type": "string", "enum": ["agent", "project", "user"], "description": "The slot's scope, same value you would pass to memory_fact." },
                    "scope_ref": { "type": "string", "description": "What the claim is about - the exact slug the claim was filed under, e.g. \"openfang\" or \"openfang.memory\". History is per-slot, so a parent's slug does not show a child's versions. Required for 'project' and 'user'; ignored for 'agent'." },
                    "key": { "type": "string", "description": "The slot name, e.g. \"repo.trunk_model\"." },
                    "limit": { "type": "integer", "description": "Maximum versions to return (default 5, maximum 20)." }
                },
                "required": ["scope", "key"]
            }),
        },
        // --- Collaboration tools ---
        ToolDefinition {
            name: "agent_find".to_string(),
            description: "Discover agents by name, tag, tool, or description. Use to find specialists before delegating work.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (matches agent name, tags, tools, description)" }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "task_post".to_string(),
            description: "Post a task to the shared task queue for another agent to pick up.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "Short task title" },
                    "description": { "type": "string", "description": "Detailed task description" },
                    "assigned_to": { "type": "string", "description": "Agent name or ID to assign the task to (optional)" }
                },
                "required": ["title", "description"]
            }),
        },
        ToolDefinition {
            name: "task_claim".to_string(),
            description: "Claim the next available task from the task queue assigned to you or unassigned.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "task_complete".to_string(),
            description: "Mark a previously claimed task as completed with a result.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": { "type": "string", "description": "The task ID to complete" },
                    "result": { "type": "string", "description": "The result or outcome of the task" }
                },
                "required": ["task_id", "result"]
            }),
        },
        ToolDefinition {
            name: "task_list".to_string(),
            description: "List tasks in the shared queue, optionally filtered by status (pending, in_progress, completed) or narrowed to a single task_id. Use task_id with the id returned by agent_send_async to check whether that wake actually ran.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "status": { "type": "string", "description": "Filter by status: pending, in_progress, completed (optional)" },
                    "task_id": { "type": "string", "description": "Look up ONE task by id (optional). Pass the id returned by agent_send_async to see whether that wake is still pending, in flight, or completed (and with what result)." }
                }
            }),
        },
        ToolDefinition {
            name: "event_publish".to_string(),
            description: "Publish a custom event that can trigger proactive agents. Use to broadcast signals to the agent fleet.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "event_type": { "type": "string", "description": "Type identifier for the event (e.g., 'code_review_requested')" },
                    "payload": { "type": "object", "description": "JSON payload data for the event" }
                },
                "required": ["event_type"]
            }),
        },
        // --- Scheduling tools ---
        ToolDefinition {
            name: "schedule_create".to_string(),
            description: "Schedule a recurring task using natural language or cron syntax. Examples: 'every 5 minutes', 'daily at 9am', 'weekdays at 6pm', '0 */5 * * *'.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": { "type": "string", "description": "What this schedule does (e.g., 'Check for new emails')" },
                    "schedule": { "type": "string", "description": "Natural language or cron expression (e.g., 'every 5 minutes', 'daily at 9am', '0 */5 * * *')" },
                    "agent": { "type": "string", "description": "Agent name or ID to run this task (optional, defaults to self)" }
                },
                "required": ["description", "schedule"]
            }),
        },
        ToolDefinition {
            name: "schedule_list".to_string(),
            description: "List all scheduled tasks with their IDs, descriptions, schedules, and next run times.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "schedule_delete".to_string(),
            description: "Remove a scheduled task by its ID.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The schedule ID to remove" }
                },
                "required": ["id"]
            }),
        },
        // --- Knowledge graph tools ---
        ToolDefinition {
            name: "knowledge_add_entity".to_string(),
            description: "Add an entity to the knowledge graph. Entities represent people, organizations, projects, concepts, locations, tools, etc.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Display name of the entity" },
                    "entity_type": { "type": "string", "description": "Type: person, organization, project, concept, event, location, document, tool, or a custom type" },
                    "properties": { "type": "object", "description": "Arbitrary key-value properties (optional)" }
                },
                "required": ["name", "entity_type"]
            }),
        },
        ToolDefinition {
            name: "knowledge_add_relation".to_string(),
            description: "Add a relation between two entities in the knowledge graph.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string", "description": "Source entity ID or name" },
                    "relation": { "type": "string", "description": "Relation type: works_at, knows_about, related_to, depends_on, owned_by, created_by, located_in, part_of, uses, produces, or a custom type" },
                    "target": { "type": "string", "description": "Target entity ID or name" },
                    "confidence": { "type": "number", "description": "Confidence score 0.0-1.0 (default: 1.0)" },
                    "properties": { "type": "object", "description": "Arbitrary key-value properties (optional)" }
                },
                "required": ["source", "relation", "target"]
            }),
        },
        ToolDefinition {
            name: "knowledge_query".to_string(),
            description: "Query the knowledge graph. Filter by source entity, relation type, and/or target entity. Returns matching entity-relation-entity triples.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string", "description": "Filter by source entity name or ID (optional)" },
                    "relation": { "type": "string", "description": "Filter by relation type (optional)" },
                    "target": { "type": "string", "description": "Filter by target entity name or ID (optional)" },
                    "max_depth": { "type": "integer", "description": "Maximum traversal depth (default: 1)" }
                }
            }),
        },
        // --- Image analysis tool ---
        ToolDefinition {
            name: "image_analyze".to_string(),
            description: "Analyze an image file — returns format, dimensions, file size, and a base64 preview. For vision-model analysis, include a prompt.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the image file" },
                    "prompt": { "type": "string", "description": "Optional prompt for vision analysis (e.g., 'Describe what you see')" }
                },
                "required": ["path"]
            }),
        },
        // --- Location tool ---
        ToolDefinition {
            name: "location_get".to_string(),
            description: "Get approximate geographic location based on IP address. Returns city, country, coordinates, and timezone.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        // --- Browser automation tools ---
        ToolDefinition {
            name: "browser_navigate".to_string(),
            description: "Navigate a browser to a URL. Returns the page title and readable content as markdown. Opens a persistent browser session.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to navigate to (http/https only)" }
                },
                "required": ["url"]
            }),
        },
        ToolDefinition {
            name: "browser_click".to_string(),
            description: "Click an element on the current browser page by CSS selector or visible text. Returns the resulting page state.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "CSS selector (e.g., '#submit-btn', '.add-to-cart') or visible text to click" }
                },
                "required": ["selector"]
            }),
        },
        ToolDefinition {
            name: "browser_type".to_string(),
            description: "Type text into an input field on the current browser page.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "CSS selector for the input field (e.g., 'input[name=\"email\"]', '#search-box')" },
                    "text": { "type": "string", "description": "The text to type into the field" }
                },
                "required": ["selector", "text"]
            }),
        },
        ToolDefinition {
            name: "browser_screenshot".to_string(),
            description: "Take a screenshot of the current browser page. Returns a base64-encoded PNG image.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "browser_read_page".to_string(),
            description: "Read the current browser page content as structured markdown. Use after clicking or navigating to see the updated page.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "browser_close".to_string(),
            description: "Close the browser session. The browser will also auto-close when the agent loop ends.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "browser_scroll".to_string(),
            description: "Scroll the browser page. Use this to see content below the fold or navigate long pages.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "direction": { "type": "string", "description": "Scroll direction: 'up', 'down', 'left', 'right' (default: 'down')" },
                    "amount": { "type": "integer", "description": "Pixels to scroll (default: 600)" }
                }
            }),
        },
        ToolDefinition {
            name: "browser_wait".to_string(),
            description: "Wait for a CSS selector to appear on the page. Useful for dynamic content that loads asynchronously.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "CSS selector to wait for" },
                    "timeout_ms": { "type": "integer", "description": "Max wait time in milliseconds (default: 5000, max: 30000)" }
                },
                "required": ["selector"]
            }),
        },
        ToolDefinition {
            name: "browser_run_js".to_string(),
            description: "Run JavaScript on the current browser page and return the result. For advanced interactions that other browser tools cannot handle.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": { "type": "string", "description": "JavaScript expression to run in the page context" }
                },
                "required": ["expression"]
            }),
        },
        ToolDefinition {
            name: "browser_back".to_string(),
            description: "Go back to the previous page in browser history.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        // --- Media understanding tools ---
        ToolDefinition {
            name: "media_describe".to_string(),
            description: "Describe an image using a vision-capable LLM. Auto-selects the best available provider (Anthropic, OpenAI, or Gemini). Returns a text description of the image content.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the image file (relative to workspace)" },
                    "prompt": { "type": "string", "description": "Optional prompt to guide the description (e.g., 'Extract all text from this image')" }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "media_transcribe".to_string(),
            description: "Transcribe audio to text using speech-to-text. Auto-selects the best available provider (Groq Whisper or OpenAI Whisper). Returns the transcript.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the audio file (relative to workspace). Supported: mp3, wav, ogg, flac, m4a, webm." },
                    "language": { "type": "string", "description": "Optional ISO-639-1 language code (e.g., 'en', 'es', 'ja')" }
                },
                "required": ["path"]
            }),
        },
        // --- Image generation tool ---
        ToolDefinition {
            name: "image_generate".to_string(),
            description: "Generate images from a text prompt using DALL-E 3, DALL-E 2, or GPT-Image-1. Requires OPENAI_API_KEY. Generated images are saved to the workspace output/ directory.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Text description of the image to generate (max 4000 chars)" },
                    "model": { "type": "string", "description": "Model to use: 'dall-e-3' (default), 'dall-e-2', or 'gpt-image-1'" },
                    "size": { "type": "string", "description": "Image size: '1024x1024' (default), '1024x1792', '1792x1024', '256x256', '512x512'" },
                    "quality": { "type": "string", "description": "Quality: 'hd' (default for dall-e-3) or 'standard'" },
                    "count": { "type": "integer", "description": "Number of images to generate (1-4, default: 1). DALL-E 3 only supports 1." }
                },
                "required": ["prompt"]
            }),
        },
        // --- Cron scheduling tools ---
        ToolDefinition {
            name: "cron_create".to_string(),
            description: "Create a scheduled/cron job. Supports one-shot (at), recurring (every N seconds), and cron expressions. Max 50 jobs per agent.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Job name (max 128 chars, alphanumeric + spaces/hyphens/underscores)" },
                    "schedule": {
                        "type": "object",
                        "description": "Schedule: {\"kind\":\"at\",\"at\":\"2025-01-01T00:00:00Z\"} or {\"kind\":\"every\",\"every_secs\":300} or {\"kind\":\"cron\",\"expr\":\"0 */6 * * *\"}"
                    },
                    "action": {
                        "type": "object",
                        "description": "Action: {\"kind\":\"system_event\",\"text\":\"...\"} or {\"kind\":\"agent_turn\",\"message\":\"...\",\"timeout_secs\":300}"
                    },
                    "delivery": {
                        "type": "object",
                        "description": "Delivery target: {\"kind\":\"none\"} or {\"kind\":\"channel\",\"channel\":\"telegram\"} or {\"kind\":\"last_channel\"}"
                    },
                    "one_shot": { "type": "boolean", "description": "If true, auto-delete after execution. Default: false" }
                },
                "required": ["name", "schedule", "action"]
            }),
        },
        ToolDefinition {
            name: "cron_list".to_string(),
            description: "List all scheduled/cron jobs for the current agent.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "cron_cancel".to_string(),
            description: "Cancel a scheduled/cron job by its ID.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "job_id": { "type": "string", "description": "The UUID of the cron job to cancel" }
                },
                "required": ["job_id"]
            }),
        },
        // --- Channel send tool (proactive outbound messaging) ---
        ToolDefinition {
            name: "channel_send".to_string(),
            description: "Send a message or media to a user on a configured channel (email, telegram, slack, etc). For email: recipient is the email address; optionally set subject. For media: set image_url, file_url, or file_path to send an image or file instead of (or alongside) text. Use `attachments` to send one or more local files alongside the message (workspace-relative paths preferred). Use thread_id to reply in a specific thread/topic.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel": { "type": "string", "description": "Channel adapter name (e.g., 'email', 'telegram', 'slack', 'discord')" },
                    "recipient": { "type": "string", "description": "Platform-specific recipient identifier (email address, user ID, etc.)" },
                    "subject": { "type": "string", "description": "Optional subject line (used for email; ignored for other channels)" },
                    "message": { "type": "string", "description": "The message body to send (required for text, optional caption for media)" },
                    "image_url": { "type": "string", "description": "URL of an image to send (supported on Telegram, Discord, Slack)" },
                    "file_url": { "type": "string", "description": "URL of a file to send as attachment" },
                    "file_path": { "type": "string", "description": "Local file path to send as attachment (reads from disk; use instead of file_url for local files)" },
                    "filename": { "type": "string", "description": "Filename for file attachments (defaults to the basename of file_path, or 'file')" },
                    "attachments": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of file paths to attach alongside the message. Workspace-relative paths are preferred (resolved against the agent's workspace root); absolute paths are also accepted. Routes through the outbound attachment parser with the same `allow_roots` security gating as inline `<openfang:attach path=\"...\"/>` directives. Composes with inline directives in `message`."
                    },
                    "thread_id": { "type": "string", "description": "Thread/topic ID to reply in (e.g., Telegram message_thread_id, Slack thread_ts)" }
                },
                "required": ["channel", "recipient"]
            }),
        },
        // --- Hand tools (curated autonomous capability packages) ---
        ToolDefinition {
            name: "hand_list".to_string(),
            description: "List available Hands (curated autonomous packages) and their activation status.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "hand_activate".to_string(),
            description: "Activate a Hand — spawns a specialized autonomous agent with curated tools and skills.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "hand_id": { "type": "string", "description": "The ID of the hand to activate (e.g. 'researcher', 'clip', 'browser')" },
                    "config": { "type": "object", "description": "Optional configuration overrides for the hand's settings" }
                },
                "required": ["hand_id"]
            }),
        },
        ToolDefinition {
            name: "hand_status".to_string(),
            description: "Check the status and metrics of an active Hand.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "hand_id": { "type": "string", "description": "The ID of the hand to check status for" }
                },
                "required": ["hand_id"]
            }),
        },
        ToolDefinition {
            name: "hand_deactivate".to_string(),
            description: "Deactivate a running Hand and stop its agent.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "instance_id": { "type": "string", "description": "The UUID of the hand instance to deactivate" }
                },
                "required": ["instance_id"]
            }),
        },
        // --- A2A outbound tools ---
        ToolDefinition {
            name: "a2a_discover".to_string(),
            description: "Discover an external A2A agent by fetching its agent card from a URL. Returns the agent's name, description, skills, and supported protocols.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Base URL of the remote OpenFang/A2A-compatible agent (e.g., 'https://agent.example.com')" }
                },
                "required": ["url"]
            }),
        },
        ToolDefinition {
            name: "a2a_send".to_string(),
            description: "Send a task/message to an external A2A agent and get the response. Use agent_name to send to a previously discovered agent, or agent_url for direct addressing.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The task/message to send to the remote agent" },
                    "agent_url": { "type": "string", "description": "Direct URL of the remote agent's A2A endpoint" },
                    "agent_name": { "type": "string", "description": "Name of a previously discovered A2A agent (looked up from kernel)" },
                    "session_id": { "type": "string", "description": "Optional session ID for multi-turn conversations" }
                },
                "required": ["message"]
            }),
        },
        // --- TTS/STT tools ---
        ToolDefinition {
            name: "text_to_speech".to_string(),
            description: "Convert text to speech audio. Auto-selects OpenAI or ElevenLabs. Saves audio to workspace output/ directory.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to convert to speech (max 4096 chars)" },
                    "voice": { "type": "string", "description": "Voice name: 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer' (default: 'alloy')" },
                    "format": { "type": "string", "description": "Output format: 'mp3', 'opus', 'aac', 'flac' (default: 'mp3')" }
                },
                "required": ["text"]
            }),
        },
        ToolDefinition {
            name: "speech_to_text".to_string(),
            description: "Transcribe audio to text using speech-to-text. Auto-selects Groq Whisper or OpenAI Whisper. Supported formats: mp3, wav, ogg, flac, m4a, webm.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the audio file (relative to workspace)" },
                    "language": { "type": "string", "description": "Optional ISO-639-1 language code (e.g., 'en', 'es', 'ja')" }
                },
                "required": ["path"]
            }),
        },
        // --- Docker sandbox tool ---
        ToolDefinition {
            name: "docker_exec".to_string(),
            description: "Execute a command inside a Docker container sandbox. Provides OS-level isolation with resource limits, network isolation, and capability dropping. Requires Docker to be installed and docker.enabled=true.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to execute inside the container" }
                },
                "required": ["command"]
            }),
        },
        // --- Persistent process tools ---
        ToolDefinition {
            name: "process_start".to_string(),
            description: "Start a long-running process (REPL, server, watcher). Returns a process_id for subsequent poll/write/kill operations. Max 5 processes per agent.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The executable to run (e.g. 'python', 'node', 'npm')" },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Command-line arguments (e.g. ['-i'] for interactive Python)"
                    }
                },
                "required": ["command"]
            }),
        },
        ToolDefinition {
            name: "process_poll".to_string(),
            description: "Read accumulated stdout/stderr from a running process. Non-blocking: returns whatever output has buffered since the last poll.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "process_id": { "type": "string", "description": "The process ID returned by process_start" }
                },
                "required": ["process_id"]
            }),
        },
        ToolDefinition {
            name: "process_write".to_string(),
            description: "Write data to a running process's stdin. A newline is appended automatically if not present.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "process_id": { "type": "string", "description": "The process ID returned by process_start" },
                    "data": { "type": "string", "description": "The data to write to stdin" }
                },
                "required": ["process_id", "data"]
            }),
        },
        ToolDefinition {
            name: "process_kill".to_string(),
            description: "Terminate a running process and clean up its resources.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "process_id": { "type": "string", "description": "The process ID returned by process_start" }
                },
                "required": ["process_id"]
            }),
        },
        ToolDefinition {
            name: "process_list".to_string(),
            description: "List all running processes for the current agent, including their IDs, commands, uptime, and alive status.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        // --- System time tool ---
        ToolDefinition {
            name: "system_time".to_string(),
            description: "Get the current date, time, and timezone. Returns ISO 8601 timestamp, Unix epoch seconds, and timezone info.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        // --- Canvas / A2UI tool ---
        ToolDefinition {
            name: "canvas_present".to_string(),
            description: "Present an interactive HTML canvas to the user. The HTML is sanitized (no scripts, no event handlers) and saved to the workspace. The dashboard will render it in a panel. Use for rich data visualizations, formatted reports, or interactive UI.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "html": { "type": "string", "description": "The HTML content to present. Must not contain <script> tags, event handlers, or javascript: URLs." },
                    "title": { "type": "string", "description": "Optional title for the canvas panel" }
                },
                "required": ["html"]
            }),
        },
        // --- Skill introspection tools (issue #1038) ---
        // These let the agent discover and read installed skills without
        // touching the filesystem. Global skills live at ~/.openfang/skills/
        // which is outside the workspace sandbox — file_read cannot reach them.
        ToolDefinition {
            name: "skill_list".to_string(),
            description: "List all installed skills available to this agent. Returns name, version, description, runtime type, and provided tool names. Use this instead of file_list on the skills directory.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "skill_describe".to_string(),
            description: "Read the full description (SKILL.md body / prompt context) of an installed skill by name. Use this instead of file_read on a skill's SKILL.md file — global skills live outside the workspace sandbox and cannot be read with file_read.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The skill name (as returned by skill_list)" }
                },
                "required": ["name"]
            }),
        },
        ToolDefinition {
            name: "skill_execute".to_string(),
            description: "Execute a tool provided by an installed skill. For code-runtime skills (Python/Node/Shell) this invokes the underlying script. For prompt-only skills this returns the skill's instruction body so the agent can follow it using built-in tools.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "skill": { "type": "string", "description": "The skill name (as returned by skill_list)" },
                    "tool": { "type": "string", "description": "Optional name of a tool the skill provides. Omit to invoke the skill's default behavior (returns SKILL.md body for prompt-only skills)." },
                    "input": { "type": "object", "description": "Optional JSON input for the skill tool" }
                },
                "required": ["skill"]
            }),
        },
    ]
}

// ---------------------------------------------------------------------------
// Filesystem tools
// ---------------------------------------------------------------------------

/// SECURITY: Reject path traversal attempts. Forbids `..` components in file paths.
fn validate_path(path: &str) -> Result<&str, String> {
    for component in std::path::Path::new(path).components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err("Path traversal denied: '..' components are forbidden".to_string());
        }
    }
    Ok(path)
}

/// For single-path filesystem tools, return `(needs_write, path)` so the
/// file_policy prompt pre-pass can evaluate the target's tier. Multi-path
/// tools (apply_patch) return None and are governed per-target in their helper.
fn fs_tool_single_path<'a>(
    tool_name: &str,
    input: &'a serde_json::Value,
) -> Option<(bool, &'a str)> {
    let needs_write = match tool_name {
        "file_read" | "file_list" | "file_grep" | "image_read" => false,
        "file_write" | "create_directory" => true,
        _ => return None,
    };
    input["path"].as_str().map(|p| (needs_write, p))
}

/// Resolve a file path through the workspace sandbox (if available) or legacy
/// validation. When a `file_policy` is active it governs the resolved path via
/// tier evaluation; otherwise the legacy workspace clamp applies. `needs_write`
/// distinguishes read verbs from mutating verbs for the `read` tier. The prompt
/// tier is pre-approved by execute_tool's pre-pass for single-path tools, so
/// `prompt_preapproved = true` here.
fn resolve_file_path(
    raw_path: &str,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    needs_write: bool,
) -> Result<PathBuf, String> {
    if let Some(root) = workspace_root {
        crate::workspace_sandbox::resolve_with_policy(
            raw_path,
            root,
            file_policy,
            needs_write,
            true,
        )
    } else {
        let _ = validate_path(raw_path)?;
        Ok(PathBuf::from(raw_path))
    }
}

/// Byte size at which an unranged `file_read` stops returning the whole file
/// and returns a head plus a manifest instead (ANAI-291).
///
/// Chosen against the two real ceilings this sits under: the native driver's
/// per-tool-result cap (30% of the context window at two chars per token —
/// 120,000 chars on a 200k window) and the bridge's 1 MiB frame. 128 KiB is
/// roughly 32k tokens, which is already a large share of a turn's context to
/// spend on one call, and it leaves both ceilings a wide margin.
///
/// Deliberately a constant and not yet an operator knob: `tool_file_read` does
/// not receive the config, and threading it through `execute_tool` would touch
/// every call site. If these numbers bite in ops, that plumbing is the fix.
const FILE_READ_WHOLE_LIMIT_BYTES: u64 = 128 * 1024;

/// Lines of head returned with the manifest when a file is too large to return
/// whole. Enough to identify a file and find its structure; not enough to
/// pretend it is the file.
const FILE_READ_HEAD_LINES: usize = 200;

/// Byte ceiling on an **explicit** range. Higher than
/// [`FILE_READ_WHOLE_LIMIT_BYTES`] because the caller stated a bound and is
/// entitled to more trust than a caller who stated none — but still bounded,
/// so `limit: 1000000` degrades to a marked truncation here rather than
/// travelling on to be clamped anonymously at the bridge frame.
const FILE_READ_SLICE_LIMIT_BYTES: usize = 256 * 1024;

/// A line-denominated window read out of a file, plus what it took to get it.
struct FileWindow {
    /// The window's bytes, newlines preserved exactly as on disk.
    bytes: Vec<u8>,
    /// 1-based line number of the first line included (0 when none were).
    first_line: usize,
    /// 1-based line number of the last line included (0 when none were).
    last_line: usize,
    /// Total lines in the whole file — the denominator, so a slice is never
    /// mistaken for the file.
    total_lines: usize,
    /// True when the window hit [`FILE_READ_SLICE_LIMIT_BYTES`] and stopped
    /// short of the requested line count.
    byte_capped: bool,
    /// sha256 of the entire file, not of the window. Identifies *which* file a
    /// slice came from across calls.
    sha256: String,
}

/// Read a 1-based line window out of `path` without holding the whole file.
///
/// Scans to EOF regardless of the window, because `total_lines` and the file
/// hash are the two things that make a partial read honest, and both require
/// seeing every byte. Only the window itself is retained.
///
/// Splits on `\n` with [`tokio::io::AsyncBufReadExt::read_until`] rather than
/// `lines()`: `lines()` strips the terminator and a trailing `\r`, which would
/// silently rewrite a CRLF file's contents on the way out.
async fn read_line_window(
    path: &Path,
    start: usize,
    max_lines: Option<usize>,
) -> Result<FileWindow, String> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncBufReadExt;

    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| format!("Failed to read file: {e}"))?;
    let mut reader = tokio::io::BufReader::new(file);
    let mut hasher = Sha256::new();

    let end_exclusive = max_lines.map(|n| start.saturating_add(n));
    let mut buf: Vec<u8> = Vec::new();
    let mut window: Vec<u8> = Vec::new();
    let mut line_no: usize = 0;
    let mut first_line: usize = 0;
    let mut last_line: usize = 0;
    let mut byte_capped = false;

    loop {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .await
            .map_err(|e| format!("Failed to read file: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf);
        line_no += 1;

        let in_window = line_no >= start && end_exclusive.is_none_or(|e| line_no < e);
        if in_window && !byte_capped {
            if window.len() + buf.len() > FILE_READ_SLICE_LIMIT_BYTES {
                // Stop collecting, keep scanning: the denominator still has to
                // be true.
                byte_capped = true;
            } else {
                if first_line == 0 {
                    first_line = line_no;
                }
                last_line = line_no;
                window.extend_from_slice(&buf);
            }
        }
    }

    Ok(FileWindow {
        bytes: window,
        first_line,
        last_line,
        total_lines: line_no,
        byte_capped,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

/// Leading bytes handed to [`sniff_binary_format`]. Taken with a `min`, not a
/// range index: `slice.get(..64)` yields `None` for anything SHORTER than 64
/// bytes, which silently costs the signpost on exactly the small files where
/// a signature is most of the file.
fn leading_bytes(bytes: &[u8]) -> &[u8] {
    &bytes[..bytes.len().min(64)]
}

/// Identify a few common binary formats from their leading bytes, so a
/// `file_read` that fails on UTF-8 can name what the file actually is and the
/// call that would work.
///
/// This is the signpost half of the "polymorphic file_read" question, and
/// deliberately not the router half: `file_read` never silently substitutes
/// converted text for the bytes on disk. A read that returns something other
/// than what is on disk makes every downstream assumption — diffing, hashing,
/// "I read it so I can patch it" — quietly wrong, with no way for the caller
/// to tell. So it points at `file_convert` and stops.
fn sniff_binary_format(head: &[u8]) -> Option<&'static str> {
    const SIGNATURES: &[(&[u8], &str)] = &[
        (b"%PDF-", "PDF"),
        (b"\x89PNG\r\n\x1a\n", "PNG"),
        (b"\xff\xd8\xff", "JPEG"),
        (b"GIF8", "GIF"),
        (b"PK\x03\x04", "ZIP or OOXML (docx/xlsx/pptx)"),
        (b"\x7fELF", "ELF executable"),
        (b"\xca\xfe\xba\xbe", "Mach-O universal binary"),
        (b"SQLite format 3\0", "SQLite database"),
        (b"\x1f\x8b", "gzip"),
    ];
    // WebP is `RIFF<size>WEBP`: the discriminating bytes sit after a length
    // field, so it cannot be a plain prefix row above.
    if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        return Some("WebP");
    }
    SIGNATURES
        .iter()
        .find(|(sig, _)| head.starts_with(sig))
        .map(|(_, name)| *name)
}

/// Turn a UTF-8 failure into a signpost naming the format and the call that
/// would work, instead of a bare decode error.
fn utf8_read_error(raw_path: &str, head: &[u8]) -> String {
    match sniff_binary_format(head) {
        Some("PDF") => format!(
            "'{raw_path}' is not valid UTF-8 text: it is a PDF. file_read returns bytes \
             as-is and will not extract for you. Convert it first, then read the result: \
             file_convert(to=\"txt\", path=\"{raw_path}\") returns a path you can file_read. \
             file_convert follows the same file_policy as file_read, so any PDF you can read \
             you can convert; pass an 'output' you can write if this location is read-only."
        ),
        Some(fmt @ ("PNG" | "JPEG" | "GIF" | "WebP")) => format!(
            "'{raw_path}' is not valid UTF-8 text: it is a {fmt} image. file_read only \
             returns text. To look at it, call image_read(path=\"{raw_path}\"), which \
             returns the image itself."
        ),
        Some(fmt) => format!(
            "'{raw_path}' is not valid UTF-8 text: it looks like {fmt}. file_read returns \
             the bytes on disk and does not transcode. Check file_convert for a recipe \
             that turns this format into text."
        ),
        None => format!(
            "'{raw_path}' is not valid UTF-8 text and its leading bytes match no format \
             openfang recognises. file_read only returns text; there is nothing here it \
             can hand back."
        ),
    }
}

/// Parse an optional 1-based line argument, liberally but not silently.
///
/// Accepts a JSON integer or a numeric string (models send both). Rejects
/// zero with an explicit note that lines are 1-based: silently treating 0 as 1
/// would hide a real off-by-one in the caller, and silently treating it as
/// "line zero" would shift every subsequent slice.
fn parse_line_arg(input: &serde_json::Value, key: &str) -> Result<Option<usize>, String> {
    let raw = &input[key];
    if raw.is_null() {
        return Ok(None);
    }
    let value = if let Some(n) = raw.as_u64() {
        n
    } else if let Some(s) = raw.as_str() {
        let s = s.trim();
        if s.is_empty() {
            return Ok(None);
        }
        s.parse::<u64>()
            .map_err(|_| format!("'{key}' must be a positive whole number of lines, got {s:?}"))?
    } else {
        return Err(format!(
            "'{key}' must be a positive whole number of lines, got {raw}"
        ));
    };
    if value == 0 {
        return Err(format!(
            "'{key}' is 0, but line numbers are 1-based: the first line of a file is \
             line 1. Pass 1 to start at the beginning."
        ));
    }
    Ok(Some(value as usize))
}

// ---------------------------------------------------------------------------
// image_read (ANAI-297)
// ---------------------------------------------------------------------------

/// One image a tool hands back to the model beside its text result
/// (ANAI-297). `data_base64` is standard base64 of the file's bytes exactly as
/// on disk; `mime_type` comes from the file's leading bytes, never its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolImage {
    /// `image/png`, `image/jpeg`, `image/gif` or `image/webp`.
    pub mime_type: String,
    /// Standard, padded base64 of the raw file.
    pub data_base64: String,
}

tokio::task_local! {
    /// Where an image-producing tool deposits its images (ANAI-297).
    ///
    /// `ToolResult` is text-only and is built in well over a hundred places,
    /// so images travel beside it rather than inside it. Only a caller that
    /// can actually deliver images to the model scopes this: today, the bridge
    /// IPC dispatcher, via [`with_image_sink`]. Every other path (native
    /// drivers, the HTTP `/mcp` route) leaves it unset, and `image_read`
    /// refuses there rather than returning a text-only "success" that quietly
    /// lost its pixels.
    static IMAGE_SINK: std::cell::RefCell<Vec<ToolImage>>;
}

/// Run `fut` (normally an [`execute_tool`] call) with an image sink in scope,
/// and return its output together with any images a tool deposited.
///
/// Scoping the sink is a promise that the caller delivers what lands in it.
/// Do not wrap a call whose images you would drop.
pub async fn with_image_sink<T, F>(fut: F) -> (T, Vec<ToolImage>)
where
    F: std::future::Future<Output = T>,
{
    IMAGE_SINK
        .scope(std::cell::RefCell::new(Vec::new()), async move {
            let out = fut.await;
            let images = IMAGE_SINK.with(|sink| std::mem::take(&mut *sink.borrow_mut()));
            (out, images)
        })
        .await
}

/// Identify an image format `image_read` returns, from the file's leading
/// bytes. These four are the formats the model accepts; anything else is
/// refused by name rather than forwarded to fail at the provider.
fn sniff_image_mime(head: &[u8]) -> Option<&'static str> {
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if head.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// Signpost for a file `image_read` will not return: what it looks like, and
/// the tool that would work.
fn image_read_format_error(raw_path: &str, head: &[u8]) -> String {
    let start = head
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(head.len());
    let text_head = &head[start..];
    if text_head.starts_with(b"<svg") || text_head.starts_with(b"<?xml") {
        return format!(
            "'{raw_path}' looks like SVG or other XML, which is text, not a raster \
             image. image_read returns PNG, JPEG, GIF or WebP only. Read it with \
             file_read. Nothing was returned."
        );
    }
    match sniff_binary_format(head) {
        Some(fmt) => format!(
            "'{raw_path}' looks like {fmt}, which is not a format image_read returns \
             (PNG, JPEG, GIF, WebP). Nothing was returned."
        ),
        None => format!(
            "'{raw_path}' does not start with the signature of PNG, JPEG, GIF or WebP, \
             the only formats image_read returns. The type is decided by the file's \
             bytes, not its name. Nothing was returned."
        ),
    }
}

/// Refuse an image over the operator's limit. Whole-file only: a cut-off
/// image decodes as corrupt yet reads as a successful read.
fn check_image_size(
    raw_path: &str,
    size: u64,
    limit: openfang_types::media::ImageReadLimit,
) -> Result<(), String> {
    if size > limit.max_bytes {
        return Err(format!(
            "'{raw_path}' is {size} bytes, over image_read's limit of {} bytes \
             ([media] image_read_max_bytes in config.toml). Nothing was returned: \
             images are never truncated, because a cut-off image decodes as corrupt \
             yet reads as a successful read. Downscale or recompress it and read \
             the smaller copy.",
            limit.max_bytes
        ));
    }
    Ok(())
}

/// `image_read` (ANAI-297): return an image file to the model as an image.
///
/// Path resolution, tiering and prompt-tier approval are `file_read`'s,
/// unchanged: the same `resolve_file_path(.., needs_write = false)`, the same
/// pre-pass in [`execute_tool`], the same prevalidated-path check. So it
/// reaches exactly the files `file_read` reaches, which is the admission
/// condition for its companion grant in the kernel.
///
/// The image is deposited in [`IMAGE_SINK`] only on success, and only after
/// every check has passed, so an error never carries a picture.
async fn tool_image_read(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    prevalidated: Option<&Path>,
    limit: openfang_types::media::ImageReadLimit,
) -> Result<String, String> {
    use base64::Engine;
    use sha2::{Digest, Sha256};

    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let resolved = resolve_file_path(raw_path, workspace_root, file_policy, false)?;
    crate::workspace_sandbox::assert_prevalidated(&resolved, prevalidated)?;

    let meta = tokio::fs::metadata(&resolved)
        .await
        .map_err(|e| format!("Failed to read file: {e}"))?;
    if meta.is_dir() {
        return Err(format!(
            "'{raw_path}' is a directory, not a file. Use file_list to enumerate it."
        ));
    }
    if meta.len() == 0 {
        return Err(format!(
            "'{raw_path}' is empty (0 bytes), so there is no image to return."
        ));
    }
    check_image_size(raw_path, meta.len(), limit)?;

    // Refuse before reading when nothing downstream can carry the image.
    if IMAGE_SINK.try_with(|_| ()).is_err() {
        return Err(format!(
            "image_read cannot return '{raw_path}' here: images are only carried by \
             the Claude Code bridge, and this call arrived through a text-only path. \
             Nothing was returned. This is a limit of the calling path, not of the file."
        ));
    }

    let bytes = tokio::fs::read(&resolved)
        .await
        .map_err(|e| format!("Failed to read file: {e}"))?;
    // Again on the bytes actually read: the file can grow between stat and read.
    check_image_size(raw_path, bytes.len() as u64, limit)?;
    let mime = sniff_image_mime(leading_bytes(&bytes))
        .ok_or_else(|| image_read_format_error(raw_path, leading_bytes(&bytes)))?;

    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let image = ToolImage {
        mime_type: mime.to_string(),
        data_base64: base64::engine::general_purpose::STANDARD.encode(&bytes),
    };
    IMAGE_SINK
        .try_with(|sink| sink.borrow_mut().push(image))
        .map_err(|_| "image_read lost its image sink mid-call; nothing was returned".to_string())?;

    Ok(format!(
        "[openfang image_read: '{raw_path}' | {mime} | {} bytes | sha256 {sha256}]\n\
         The image follows as a separate image block.",
        bytes.len()
    ))
}

async fn tool_file_read(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    prevalidated: Option<&Path>,
) -> Result<String, String> {
    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let resolved = resolve_file_path(raw_path, workspace_root, file_policy, false)?;
    crate::workspace_sandbox::assert_prevalidated(&resolved, prevalidated)?;

    let offset = parse_line_arg(input, "offset")?;
    let limit = parse_line_arg(input, "limit")?;

    let meta = tokio::fs::metadata(&resolved)
        .await
        .map_err(|e| format!("Failed to read file: {e}"))?;
    if meta.is_dir() {
        return Err(format!(
            "'{raw_path}' is a directory, not a file. Use file_list to enumerate it."
        ));
    }
    let total_bytes = meta.len();

    // Unranged read of a file that comfortably fits: byte-identical to the
    // behaviour before ranges existed, header and all (there isn't one). This
    // path is load-bearing for backward compatibility — annotating every read
    // would break every caller that hashes, diffs, or round-trips content.
    if offset.is_none() && limit.is_none() && total_bytes <= FILE_READ_WHOLE_LIMIT_BYTES {
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| format!("Failed to read file: {e}"))?;
        return String::from_utf8(bytes)
            .map_err(|e| utf8_read_error(raw_path, leading_bytes(e.as_bytes())));
    }

    let ranged = offset.is_some() || limit.is_some();
    let start = offset.unwrap_or(1);
    let max_lines = if ranged {
        limit
    } else {
        Some(FILE_READ_HEAD_LINES)
    };

    let window = read_line_window(&resolved, start, max_lines).await?;
    let body = String::from_utf8(window.bytes)
        .map_err(|e| utf8_read_error(raw_path, leading_bytes(e.as_bytes())))?;

    // An empty window is a refusal, not an empty success. Returning "" for an
    // offset past EOF is indistinguishable from an empty file, and the caller
    // would carry on believing it had read something.
    if window.first_line == 0 {
        if window.total_lines == 0 {
            return Err(format!(
                "'{raw_path}' is empty (0 lines, {total_bytes} bytes), so there is no \
                 line {start} to return."
            ));
        }
        return Err(format!(
            "offset {start} is past the end of '{raw_path}', which has {} lines. \
             Nothing was returned. Pass an offset between 1 and {}.",
            window.total_lines, window.total_lines
        ));
    }

    let mut out = String::new();
    if ranged {
        out.push_str(&format!(
            "[openfang file_read: '{}' lines {}-{} of {} | {} bytes total]\n",
            raw_path, window.first_line, window.last_line, window.total_lines, total_bytes
        ));
        if window.byte_capped {
            out.push_str(&format!(
                "[openfang file_read: this slice hit the {} byte ceiling and stopped at \
                 line {} — fewer lines than you asked for. Continue with offset {}.]\n",
                FILE_READ_SLICE_LIMIT_BYTES,
                window.last_line,
                window.last_line + 1
            ));
        }
    } else {
        // Head + manifest. The point of the manifest is that the dead end
        // becomes a signpost: it says what the file is, proves which file it
        // is, and names the exact call that gets the rest.
        out.push_str(&format!(
            "[openfang file_read: '{}' is {} bytes / {} lines — over the {} byte limit \
             for returning a file whole, so this is the HEAD ONLY.\n\
             Showing lines {}-{} of {}. sha256={}\n\
             Next slice:    file_read(path=\"{}\", offset={}, limit={})\n\
             Search first:  file_grep(path=\"{}\", pattern=\"...\") returns line numbers \
             you can pass straight to offset — usually the right move, since a large \
             read is nearly always a search wearing a read's clothes.]\n",
            raw_path,
            total_bytes,
            window.total_lines,
            FILE_READ_WHOLE_LIMIT_BYTES,
            window.first_line,
            window.last_line,
            window.total_lines,
            window.sha256,
            raw_path,
            window.last_line + 1,
            FILE_READ_HEAD_LINES,
            raw_path,
        ));
    }
    out.push_str(&body);
    Ok(out)
}

/// Default cap on reported matches. A pattern that matches every line would
/// otherwise reinvent exactly the context problem `file_grep` exists to solve.
const FILE_GREP_DEFAULT_MAX_MATCHES: usize = 100;

/// Hard ceiling on `max_matches`, so a caller cannot opt out of the bound.
const FILE_GREP_MAX_MATCHES_CEILING: usize = 2000;

/// Default cap on files visited in a directory search.
const FILE_GREP_DEFAULT_MAX_FILES: usize = 400;

/// Hard ceiling on `max_files`.
const FILE_GREP_MAX_FILES_CEILING: usize = 5000;

/// Longest reported match line before it is cut. One minified bundle line can
/// be megabytes; the line NUMBER is the useful part of a hit, and the caller
/// can `file_read` the range to see it whole.
const FILE_GREP_MAX_LINE_CHARS: usize = 512;

/// Files bigger than this are skipped during a directory walk and disclosed in
/// the trailer. Searching a multi-gigabyte artifact to find nothing is not
/// what the caller meant, and doing it silently is worse.
const FILE_GREP_MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// Directory names skipped by default during a recursive search: build output
/// and VCS internals, which are large, generated, and essentially never the
/// thing being looked for. Disclosed in the result trailer rather than applied
/// silently, because a skip nobody is told about is indistinguishable from an
/// absence of matches.
const FILE_GREP_SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
];

/// Translate a filename glob into a regex anchored to the whole file name.
///
/// Only `*` is meaningful, because `*.rs` is what a caller actually writes and
/// a half-implemented glob dialect is worse than a stated one. Everything else
/// is matched literally.
fn filename_glob_to_regex(glob: &str) -> Result<regex_lite::Regex, String> {
    let mut pattern = String::from("^");
    for ch in glob.chars() {
        if ch == '*' {
            pattern.push_str(".*");
        } else {
            pattern.push_str(&regex_lite::escape(&ch.to_string()));
        }
    }
    pattern.push('$');
    regex_lite::Regex::new(&pattern)
        .map_err(|e| format!("'include' is not a usable filename glob ({glob:?}): {e}"))
}

/// Parse an optional positive-integer argument, clamping to a stated ceiling
/// and disclosing the clamp rather than silently honouring the caller's number.
fn parse_capped_arg(
    input: &serde_json::Value,
    key: &str,
    default: usize,
    ceiling: usize,
) -> Result<(usize, Option<usize>), String> {
    let Some(requested) = parse_line_arg(input, key)? else {
        return Ok((default, None));
    };
    if requested > ceiling {
        Ok((ceiling, Some(requested)))
    } else {
        Ok((requested, None))
    }
}

/// One reported hit.
struct GrepHit {
    line_no: usize,
    /// `true` for the matching line, `false` for a context line.
    is_match: bool,
    text: String,
}

/// Search one file's lines, appending hits. Returns `Err` for a file that
/// should be counted as skipped rather than failed.
enum FileOutcome {
    Searched,
    SkippedNonText,
    SkippedTooLarge,
}

/// `file_grep`: pattern search returning LINE NUMBERS.
///
/// ## Why this is a tool and not a shell-out
///
/// Most of the fleet has no `shell_exec` at all, and the agents that do run
/// under an allowlist whose deny floor refuses `grep -i` outright (any `-i`
/// flag trips it) and blocks pipes and redirects. So "just use grep" is not
/// available to the callers who most need to avoid pulling a whole file into
/// context. An argv-literal tool hands every agent the retrieval half without
/// handing anyone a shell.
///
/// ## Why it shares `file_read`'s grant
///
/// A grep over a file exposes a strict SUBSET of what a read already returns,
/// so denying it while granting `file_read` protects nothing. That equivalence
/// has to be enforced and not merely asserted: every candidate path — not just
/// the top-level argument — goes through the same `resolve_file_path` that
/// `file_read` uses, so a subdirectory in a deny tier is refused here exactly
/// as it would be there. Routing recursion around the resolver would make this
/// a read-tier bypass regardless of which allowlist it sits in.
async fn tool_file_grep(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    prevalidated: Option<&Path>,
) -> Result<String, String> {
    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let pattern = input["pattern"]
        .as_str()
        .ok_or("Missing 'pattern' parameter")?;
    if pattern.is_empty() {
        return Err("'pattern' is empty, which matches every line. State what \
                    you are looking for."
            .to_string());
    }
    let ignore_case = input["ignore_case"].as_bool().unwrap_or(false);
    let context = parse_line_arg(input, "context")?.unwrap_or(0).min(20);
    let (max_matches, matches_clamped) = parse_capped_arg(
        input,
        "max_matches",
        FILE_GREP_DEFAULT_MAX_MATCHES,
        FILE_GREP_MAX_MATCHES_CEILING,
    )?;
    let (max_files, files_clamped) = parse_capped_arg(
        input,
        "max_files",
        FILE_GREP_DEFAULT_MAX_FILES,
        FILE_GREP_MAX_FILES_CEILING,
    )?;
    let include = match input["include"].as_str() {
        Some(g) if !g.trim().is_empty() => Some(filename_glob_to_regex(g.trim())?),
        _ => None,
    };

    let effective = if ignore_case {
        format!("(?i){pattern}")
    } else {
        pattern.to_string()
    };
    let re = regex_lite::Regex::new(&effective)
        .map_err(|e| format!("'pattern' is not a valid regular expression ({pattern:?}): {e}"))?;

    let root = resolve_file_path(raw_path, workspace_root, file_policy, false)?;
    crate::workspace_sandbox::assert_prevalidated(&root, prevalidated)?;
    let root_meta = tokio::fs::metadata(&root)
        .await
        .map_err(|e| format!("Failed to search '{raw_path}': {e}"))?;

    // Candidate files. A single-file search skips the walk entirely.
    let mut candidates: Vec<PathBuf> = Vec::new();
    let mut dirs_skipped: Vec<String> = Vec::new();
    let mut walk_capped = false;
    if root_meta.is_dir() {
        let mut queue = vec![root.clone()];
        while let Some(dir) = queue.pop() {
            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(e) => e,
                // An unreadable subdirectory is not a reason to fail the whole
                // search; it is a reason to say so, which the trailer does.
                Err(_) => continue,
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                if candidates.len() >= max_files {
                    walk_capped = true;
                    break;
                }
                let path = entry.path();
                // symlink_metadata, NOT metadata: following a directory
                // symlink can walk straight out of the resolved tier, and can
                // also loop.
                let Ok(meta) = tokio::fs::symlink_metadata(&path).await else {
                    continue;
                };
                if meta.file_type().is_symlink() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if meta.is_dir() {
                    if FILE_GREP_SKIP_DIRS.contains(&name.as_str()) {
                        if !dirs_skipped.contains(&name) {
                            dirs_skipped.push(name);
                        }
                        continue;
                    }
                    queue.push(path);
                    continue;
                }
                if let Some(inc) = &include {
                    if !inc.is_match(&name) {
                        continue;
                    }
                }
                candidates.push(path);
            }
            if walk_capped {
                break;
            }
        }
        candidates.sort();
    } else {
        candidates.push(root.clone());
    }

    let mut out_files: Vec<(String, Vec<GrepHit>)> = Vec::new();
    let mut total_matches = 0usize;
    let mut files_searched = 0usize;
    let mut skipped_non_text = 0usize;
    let mut skipped_too_large = 0usize;
    let mut match_cap_hit = false;

    for path in &candidates {
        if total_matches >= max_matches {
            match_cap_hit = true;
            break;
        }
        // Every candidate re-enters the resolver, so grep can never reach a
        // path file_read would refuse.
        let display = path.to_string_lossy().to_string();
        if resolve_file_path(&display, workspace_root, file_policy, false).is_err() {
            continue;
        }
        let outcome = search_one_file(
            path,
            &re,
            context,
            max_matches - total_matches,
            &mut out_files,
            &mut total_matches,
            &mut match_cap_hit,
            &display,
        )
        .await;
        match outcome {
            FileOutcome::Searched => files_searched += 1,
            FileOutcome::SkippedNonText => skipped_non_text += 1,
            FileOutcome::SkippedTooLarge => skipped_too_large += 1,
        }
    }

    // --- Render -------------------------------------------------------------
    let mut out = String::new();
    out.push_str(&format!(
        "[openfang file_grep: {} match(es) in {} file(s), {} file(s) searched under '{}' for /{}/{}]\n",
        total_matches,
        out_files.len(),
        files_searched,
        raw_path,
        pattern,
        if ignore_case { " (case-insensitive)" } else { "" }
    ));
    out.push_str(
        "[line numbers are 1-based and can be passed straight to \
         file_read(offset=...)]\n",
    );

    for (display, hits) in &out_files {
        out.push_str(&format!("\n{display}\n"));
        let mut previous: Option<usize> = None;
        for hit in hits {
            if let Some(prev) = previous {
                if hit.line_no > prev + 1 {
                    out.push_str("  --\n");
                }
            }
            // ':' for a match, '-' for context — the grep convention, so the
            // two are never confused for each other.
            let sep = if hit.is_match { ':' } else { '-' };
            out.push_str(&format!("{:>7}{} {}\n", hit.line_no, sep, hit.text));
            previous = Some(hit.line_no);
        }
    }

    if total_matches == 0 {
        out.push_str("\nNo matches.\n");
    }

    // --- Disclose every bound that bit ---------------------------------------
    // A cap applied silently reads as "there was nothing more", which is the
    // failure this whole line of work has been removing.
    if match_cap_hit {
        out.push_str(&format!(
            "\n[openfang file_grep: stopped at the {max_matches}-match cap. The count above \
             is what was found before stopping and is NOT a total -- there may be more \
             matches that were never looked for. Narrow the pattern, or raise max_matches \
             (ceiling {FILE_GREP_MAX_MATCHES_CEILING}).]\n"
        ));
    }
    if walk_capped {
        out.push_str(&format!(
            "[openfang file_grep: stopped enumerating at the {max_files}-file cap, so \
             some files under '{raw_path}' were never searched. Narrow with include, or \
             raise max_files (ceiling {FILE_GREP_MAX_FILES_CEILING}).]\n"
        ));
    }
    if skipped_non_text > 0 {
        out.push_str(&format!(
            "[openfang file_grep: skipped {skipped_non_text} non-text file(s).]\n"
        ));
    }
    if skipped_too_large > 0 {
        out.push_str(&format!(
            "[openfang file_grep: skipped {skipped_too_large} file(s) over \
             {FILE_GREP_MAX_FILE_BYTES} bytes.]\n"
        ));
    }
    if !dirs_skipped.is_empty() {
        dirs_skipped.sort();
        out.push_str(&format!(
            "[openfang file_grep: did not descend into {} (build output / VCS \
             internals are skipped by default).]\n",
            dirs_skipped.join(", ")
        ));
    }
    if let Some(req) = matches_clamped {
        out.push_str(&format!(
            "[openfang file_grep: you requested max_matches={req}; CLAMPED to \
             {max_matches}.]\n"
        ));
    }
    if let Some(req) = files_clamped {
        out.push_str(&format!(
            "[openfang file_grep: you requested max_files={req}; CLAMPED to \
             {max_files}.]\n"
        ));
    }

    Ok(out)
}

/// Stream one file, collecting matches (and their context) into `out_files`.
#[allow(clippy::too_many_arguments)]
async fn search_one_file(
    path: &Path,
    re: &regex_lite::Regex,
    context: usize,
    budget: usize,
    out_files: &mut Vec<(String, Vec<GrepHit>)>,
    total_matches: &mut usize,
    cap_hit: &mut bool,
    display: &str,
) -> FileOutcome {
    use tokio::io::AsyncBufReadExt;

    let Ok(meta) = tokio::fs::metadata(path).await else {
        return FileOutcome::SkippedNonText;
    };
    if meta.len() > FILE_GREP_MAX_FILE_BYTES {
        return FileOutcome::SkippedTooLarge;
    }
    let Ok(file) = tokio::fs::File::open(path).await else {
        return FileOutcome::SkippedNonText;
    };

    let mut reader = tokio::io::BufReader::new(file);
    let mut raw: Vec<u8> = Vec::new();
    let mut line_no = 0usize;
    let mut hits: Vec<GrepHit> = Vec::new();
    let mut before: std::collections::VecDeque<(usize, String)> = std::collections::VecDeque::new();
    let mut after_remaining = 0usize;
    let mut found = 0usize;

    loop {
        raw.clear();
        match reader.read_until(b'\n', &mut raw).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => return FileOutcome::SkippedNonText,
        }
        // A non-UTF8 file is skipped whole rather than emitting garbage for
        // the lines that happen to decode.
        let Ok(text) = std::str::from_utf8(&raw) else {
            return FileOutcome::SkippedNonText;
        };
        line_no += 1;
        let trimmed = text.trim_end_matches(['\n', '\r']);
        let shown = clip_line(trimmed);

        if re.is_match(trimmed) {
            for (n, t) in before.drain(..) {
                hits.push(GrepHit {
                    line_no: n,
                    is_match: false,
                    text: t,
                });
            }
            hits.push(GrepHit {
                line_no,
                is_match: true,
                text: shown,
            });
            found += 1;
            after_remaining = context;
            if found >= budget {
                // The budget stopped the scan mid-file, so there may be more
                // matches below that were never looked for. Setting this HERE
                // is the fix for a real defect: the outer loop only tested the
                // running total at the TOP of its next iteration, so a
                // single-file search that hit the cap disclosed nothing at all
                // and its truncated hit list read as the complete answer.
                //
                // When the last match happens to be the last line, this
                // over-discloses. That is the safe direction: the scan did
                // stop at the cap, and a warning that was not needed costs a
                // line, where a cap that stayed quiet costs the answer.
                *cap_hit = true;
                break;
            }
        } else if after_remaining > 0 {
            hits.push(GrepHit {
                line_no,
                is_match: false,
                text: shown,
            });
            after_remaining -= 1;
        } else if context > 0 {
            before.push_back((line_no, shown));
            if before.len() > context {
                before.pop_front();
            }
        }
    }

    if !hits.is_empty() {
        *total_matches += found;
        out_files.push((display.to_string(), hits));
    }
    FileOutcome::Searched
}

/// Clip an over-long line, saying so. The line number is the useful part of a
/// hit; a caller who wants the whole line can `file_read` the range.
fn clip_line(line: &str) -> String {
    if line.chars().count() <= FILE_GREP_MAX_LINE_CHARS {
        return line.to_string();
    }
    let kept: String = line.chars().take(FILE_GREP_MAX_LINE_CHARS).collect();
    format!("{kept} […line clipped]")
}

async fn tool_file_write(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    prevalidated: Option<&Path>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let resolved = resolve_file_path(raw_path, workspace_root, file_policy, true)?;
    crate::workspace_sandbox::assert_prevalidated(&resolved, prevalidated)?;
    let content = input["content"]
        .as_str()
        .ok_or("Missing 'content' parameter")?;
    if let Some(parent) = resolved.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create directories: {e}"))?;
    }
    // ANAI-149 D2: snapshot the prior content so the audit record can carry a
    // diff. Only for context files -- an ordinary write pays one filename
    // comparison and no extra I/O. Auditing never gates the write.
    let audited = crate::context_audit::is_audited(&resolved);
    let before = if audited {
        crate::context_audit::capture_before(&resolved).await
    } else {
        None
    };
    tokio::fs::write(&resolved, content)
        .await
        .map_err(|e| format!("Failed to write file: {e}"))?;
    if audited {
        crate::context_audit::record_write(
            caller_agent_id,
            "file_write",
            &resolved,
            before.as_deref(),
            Some(content),
        )
        .await;
    }
    Ok(format!(
        "Successfully wrote {} bytes to {}",
        content.len(),
        resolved.display()
    ))
}

/// Resolve a directory path for creation. Unlike `resolve_file_path`, this walks
/// up the path to find the nearest existing ancestor, canonicalizes that, and
/// re-appends the missing segments. This lets `create_directory` accept nested
/// paths like `a/b/c/d` even when none of `a`, `b`, `c` exist yet.
fn resolve_directory_path_for_create(
    raw_path: &str,
    workspace_root: Option<&Path>,
) -> Result<PathBuf, String> {
    // Reject `..` components regardless of workspace.
    let _ = validate_path(raw_path)?;

    let Some(root) = workspace_root else {
        return Ok(PathBuf::from(raw_path));
    };

    let path = Path::new(raw_path);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };

    let canon_root = root
        .canonicalize()
        .map_err(|e| format!("Failed to resolve workspace root: {e}"))?;

    // Walk up to find the nearest existing ancestor, canonicalize it, then
    // re-append the missing tail.
    let mut existing: PathBuf = candidate.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        let parent = match existing.parent() {
            Some(p) => p.to_path_buf(),
            None => return Err("Invalid path: no existing ancestor".to_string()),
        };
        let name = match existing.file_name() {
            Some(n) => n.to_os_string(),
            None => return Err("Invalid path: no filename component".to_string()),
        };
        tail.push(name);
        existing = parent;
    }

    let canon_existing = existing
        .canonicalize()
        .map_err(|e| format!("Failed to resolve ancestor directory: {e}"))?;

    let mut resolved = canon_existing;
    for segment in tail.into_iter().rev() {
        resolved.push(segment);
    }

    if !resolved.starts_with(&canon_root) {
        return Err(format!(
            "Access denied: path '{raw_path}' resolves outside workspace"
        ));
    }

    Ok(resolved)
}

async fn tool_create_directory(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    prevalidated: Option<&Path>,
) -> Result<String, String> {
    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    if raw_path.is_empty() {
        return Err("'path' parameter is empty".to_string());
    }
    let resolved = resolve_directory_path_for_create(raw_path, workspace_root)?;
    crate::workspace_sandbox::assert_prevalidated(&resolved, prevalidated)?;
    // Sensitive-path floor parity: resolve_directory_path_for_create has its own
    // clamp and does not run sandbox_floor, so apply the sensitive-path deny here.
    if let Some(reason) = crate::workspace_sandbox::is_sensitive_openfang_path(&resolved)
        .or_else(|| crate::workspace_sandbox::is_sensitive_home_path(&resolved))
    {
        return Err(format!(
            "Access denied: path '{}' resolves to a protected resource ({reason}). \
             These paths are never accessible to agents.",
            resolved.display()
        ));
    }
    // file_policy governs directory creation as a write verb (prompt-tier is
    // pre-approved by execute_tool's pre-pass for this single-path tool).
    // F4: enforce via is_active() (so a disabled per-agent override cannot
    // escape an enabled global floor) and fail closed when an active policy has
    // no workspace root to evaluate against — previously this path fell open.
    if let Some(fp) = file_policy {
        if fp.is_active() {
            let Some(root) = workspace_root else {
                return Err("Access denied: create_directory requires a workspace root when a file_policy is active".to_string());
            };
            let canon_root = root
                .canonicalize()
                .map_err(|e| format!("Failed to resolve workspace root: {e}"))?;
            match fp.tier_for(&resolved, &canon_root) {
                openfang_types::config::FileAccessTier::Write
                | openfang_types::config::FileAccessTier::Prompt => {}
                openfang_types::config::FileAccessTier::Read => {
                    return Err(format!(
                        "Access denied by file_policy: '{}' is read-only",
                        resolved.display()
                    ));
                }
                openfang_types::config::FileAccessTier::Deny => {
                    return Err(format!(
                        "Access denied by file_policy: '{}' (deny tier)",
                        resolved.display()
                    ));
                }
            }
        }
    }
    tokio::fs::create_dir_all(&resolved)
        .await
        .map_err(|e| format!("Failed to create directory: {e}"))?;
    Ok(format!("Created directory {}", resolved.display()))
}

async fn tool_file_list(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    prevalidated: Option<&Path>,
) -> Result<String, String> {
    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let resolved = resolve_file_path(raw_path, workspace_root, file_policy, false)?;
    crate::workspace_sandbox::assert_prevalidated(&resolved, prevalidated)?;
    let mut entries = tokio::fs::read_dir(&resolved)
        .await
        .map_err(|e| format!("Failed to list directory: {e}"))?;
    let mut files = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| format!("Failed to read entry: {e}"))?
    {
        let name = entry.file_name().to_string_lossy().to_string();
        let metadata = entry.metadata().await;
        let suffix = match metadata {
            Ok(m) if m.is_dir() => "/",
            _ => "",
        };
        files.push(format!("{name}{suffix}"));
    }
    files.sort();
    Ok(files.join("\n"))
}

// ---------------------------------------------------------------------------
// File conversion tool
// ---------------------------------------------------------------------------

/// The `BAD_PATH` message for a `file_convert` path the resolver refused
/// (ANAI-300). The resolver's own reason goes first and verbatim; the tail says
/// what the caller can do about it, because a bare "rejected" reads as a
/// lockout and gives the agent nothing to try.
fn convert_path_rejection(
    role: &str,
    raw: &str,
    reason: &str,
    policy_active: bool,
    output_defaulted: bool,
) -> String {
    let reason = reason.trim_end().trim_end_matches('.');
    let hint = if reason.contains("requires approval") {
        " file_convert cannot raise an approval prompt, so a prompt-tier path is refused \
         rather than asked about. Copy the file somewhere you can read (or write) without \
         approval, or ask the operator for a file_policy rule covering it."
    } else if output_defaulted && role == "output" {
        " No 'output' was given, so file_convert defaulted to writing next to the input, \
         and you cannot write there. Pass 'output' with a path you can write, such as a \
         file in your workspace."
    } else if !policy_active && reason.contains("outside workspace") {
        " This agent has no active file_policy, so file_convert, like file_read, is \
         limited to its workspace. Copy the file in first, or ask the operator for a \
         file_policy rule covering it."
    } else if reason.contains("read-only") {
        " Choose an output path you can write."
    } else {
        ""
    };
    format!("{role} path '{raw}' rejected: {reason}.{hint}")
}

/// `file_convert` — render a workspace file to another format via an
/// allowlisted recipe table (see [`crate::convert`]).
///
/// `file_convert` dispatcher (ANAI-67..71): the load-bearing security seam.
/// Resolves the request against the allowlisted recipe table (fail-closed),
/// resolves both paths under the caller's `file_policy` exactly as `file_read`
/// (input) and `file_write` (output) do (ANAI-300), substitutes the argv template
/// with resolved paths only, pins the launcher to an absolute file, preflights
/// the recipe's external `needs`, then spawns via an argv array (never a shell
/// string) with a guaranteed PATH. Every conversion outcome — success and the
/// four failure codes (UNKNOWN_FORMAT, BAD_PATH, MISSING_DEP, CONVERT_FAILED) —
/// is returned as the structured envelope from `convert_ok` / `convert_err`.
async fn tool_file_convert(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
) -> Result<String, String> {
    tool_file_convert_with_policy_in(
        input,
        workspace_root,
        file_policy,
        &crate::convert::openfang_home_dir(),
    )
    .await
}

/// Inner `file_convert` dispatcher with an injectable OpenFang home directory,
/// so hermetic tests can supply their own `scripts/` + `convert/recipes.toml`
/// without mutating process-global env. Production calls this via the wrapper
/// above with the real resolved home.
#[cfg(test)]
async fn tool_file_convert_in(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    home: &Path,
) -> Result<String, String> {
    tool_file_convert_with_policy_in(input, workspace_root, None, home).await
}

/// [`tool_file_convert_in`] with the caller's `file_policy` (ANAI-300). `None`
/// or a disabled policy keeps the legacy workspace clamp, byte for byte.
async fn tool_file_convert_with_policy_in(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    home: &Path,
) -> Result<String, String> {
    // Request-shape validation (a malformed call, not a conversion outcome):
    // these stay plain Err. Everything from the recipe lookup onward speaks the
    // structured envelope so callers can branch on `ok` + `error.code`.
    let to = input["format"]
        .as_str()
        .ok_or("Missing 'format' parameter (target format, e.g. \"pdf\")")?;
    let input_path = input["input"]
        .as_str()
        .ok_or("Missing 'input' parameter (path to the source file)")?;
    if input_path.trim().is_empty() {
        return Err("'input' parameter is empty".to_string());
    }

    // Derive the source format from the input file extension (pure string op).
    let from = match Path::new(input_path).extension().and_then(|e| e.to_str()) {
        Some(ext) => ext,
        None => {
            return Ok(convert_err(
                to,
                "UNKNOWN_FORMAT",
                &format!(
                    "input '{input_path}' has no file extension; cannot determine source format"
                ),
            ));
        }
    };

    // §5.2 allowlist (fail-closed): resolve the (from, to) pair against the
    // recipe table. An unknown pair is UNKNOWN_FORMAT, never a silent fallback,
    // and is rejected before any filesystem access or spawn.
    let recipes = crate::convert::load_recipes(home)
        .map_err(|e| format!("Failed to load conversion recipes: {e}"))?;
    let recipe = match recipes.lookup(from, to) {
        Some(r) => r,
        None => {
            let supported = recipes
                .recipes()
                .iter()
                .map(|r| format!("{}->{}", r.from, r.to))
                .collect::<Vec<_>>()
                .join(", ");
            return Ok(convert_err(
                to,
                "UNKNOWN_FORMAT",
                &format!("no conversion recipe for '{from}'->'{to}'. Supported: {supported}"),
            ));
        }
    };

    // Default the output next to the input, with the recipe's output extension.
    let output_path = input["output"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            Path::new(input_path)
                .with_extension(&recipe.out_ext)
                .to_string_lossy()
                .into_owned()
        });

    // Resolve the render preset (if any) into a trusted var map. The caller
    // supplies only a preset *key*; the dimension/scale strings that reach argv
    // are manifest-authored. Fail-closed on an unknown or unexpected preset,
    // mirroring the UNKNOWN_FORMAT pattern above (no spawn, no filesystem
    // access yet).
    let requested_preset = input["preset"].as_str().filter(|s| !s.trim().is_empty());
    let empty_vars = std::collections::BTreeMap::new();
    let preset_vars: &std::collections::BTreeMap<String, String> = if recipe.presets.is_empty() {
        if requested_preset.is_some() {
            return Ok(convert_err(
                to,
                "UNKNOWN_PRESET",
                &format!("conversion '{from}'->'{to}' takes no presets"),
            ));
        }
        &empty_vars
    } else {
        let chosen = match requested_preset.or(recipe.default_preset.as_deref()) {
            Some(c) => c,
            None => {
                return Ok(convert_err(
                    to,
                    "UNKNOWN_PRESET",
                    &format!(
                        "conversion '{from}'->'{to}' requires a preset but none was given and no default is set"
                    ),
                ));
            }
        };
        match recipe.presets.get(chosen) {
            Some(vars) => vars,
            None => {
                let offered = recipe
                    .presets
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                return Ok(convert_err(
                    to,
                    "UNKNOWN_PRESET",
                    &format!("unknown preset '{chosen}' for '{from}'->'{to}'. Offered: {offered}"),
                ));
            }
        }
    };

    // §5.1 path validation (ANAI-300): the SAME resolver file_read and
    // file_write use. The input is resolved as a read, so anything file_read
    // may open, file_convert may convert; the output is resolved as a write, so
    // it lands only where the agent may write. The floor (`..`-reject,
    // canonicalize, symlink resolution, sensitive-path deny) runs first either
    // way. With no active policy this is the legacy workspace clamp.
    //
    // `prompt_preapproved = false`: file_convert has two paths and is not in
    // execute_tool's single-path approval pre-pass, so a prompt-tier path was
    // never put to a human. Refuse it by name rather than treat it as approved.
    let root = workspace_root.ok_or("file_convert requires a workspace root")?;
    let policy_active = file_policy.is_some_and(|fp| fp.is_active());
    let resolved_input = match crate::workspace_sandbox::resolve_with_policy(
        input_path,
        root,
        file_policy,
        false,
        false,
    ) {
        Ok(p) => p,
        Err(e) => {
            return Ok(convert_err(
                to,
                "BAD_PATH",
                &convert_path_rejection("input", input_path, &e, policy_active, false),
            ))
        }
    };
    let output_defaulted = input["output"].as_str().is_none();
    let resolved_output = match crate::workspace_sandbox::resolve_with_policy(
        &output_path,
        root,
        file_policy,
        true,
        false,
    ) {
        Ok(p) => p,
        Err(e) => {
            return Ok(convert_err(
                to,
                "BAD_PATH",
                &convert_path_rejection(
                    "output",
                    &output_path,
                    &e,
                    policy_active,
                    output_defaulted,
                ),
            ))
        }
    };

    // §6 (ANAI-131) option resolution: build the substitution var map from the
    // recipe's declared option defaults, overlay validated caller-supplied
    // option values, then merge the selected preset vars. Caller option VALUES
    // are the first caller strings to reach argv (see `ArgvTokens.vars` and the
    // convert.rs module docs); they are enum/type-checked here and always land
    // as single argv literals (never a shell string). Defaults fill every
    // option, so the fixed-length argv always fully resolves.
    let mut merged_vars: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for (name, decl) in &recipe.options {
        merged_vars.insert(name.clone(), decl.default.clone());
    }
    match input.get("options") {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::Object(obj)) => {
            for (key, val) in obj {
                let decl = match recipe.options.get(key) {
                    Some(d) => d,
                    None => {
                        let valid = if recipe.options.is_empty() {
                            "(this conversion accepts no options)".to_string()
                        } else {
                            recipe
                                .options
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        };
                        return Ok(convert_err(
                            to,
                            "UNKNOWN_OPTION",
                            &format!("unknown option '{key}' for '{from}'->'{to}'. Valid: {valid}"),
                        ));
                    }
                };
                // Accept strings and naturally-stringifiable scalars so a caller
                // may pass e.g. embed_images: true or "true" interchangeably.
                let sval = match val {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => {
                        return Ok(convert_err(
                            to,
                            "INVALID_OPTION",
                            &format!(
                                "option '{key}' for '{from}'->'{to}' must be a string, boolean, or number"
                            ),
                        ));
                    }
                };
                if matches!(decl.kind, crate::convert::OptionKind::Enum)
                    && !decl.values.contains(&sval)
                {
                    return Ok(convert_err(
                        to,
                        "INVALID_OPTION",
                        &format!(
                            "option '{key}'='{sval}' not allowed for '{from}'->'{to}'. Valid: {}",
                            decl.values.join(", ")
                        ),
                    ));
                }
                merged_vars.insert(key.clone(), sval);
            }
        }
        Some(_) => {
            return Err("'options' parameter must be an object".to_string());
        }
    }
    // Preset and option namespaces are disjoint (guaranteed at load time), so
    // merging preset vars cannot clobber a resolved option.
    for (k, v) in preset_vars {
        merged_vars.insert(k.clone(), v.clone());
    }

    // §5.3 argv-template substitution: tokens -> resolved literal paths only.
    let scripts_dir = home.join("scripts");
    let argv = match recipe.resolve_argv(&crate::convert::ArgvTokens {
        script_dir: &scripts_dir,
        input: &resolved_input,
        output: &resolved_output,
        vars: &merged_vars,
    }) {
        Ok(a) => a,
        // A bad token is a manifest-authoring defect, not a conversion outcome.
        Err(e) => return Err(format!("recipe argv template invalid: {e}")),
    };

    // §5.5 absolute-binary resolution: the launcher (argv[0]) must be an
    // absolute path. We never trust the daemon's bare PATH to find it.
    let program = argv[0].clone();
    crate::subprocess_sandbox::validate_executable_path(&program)?;
    let program_path = Path::new(&program);
    if !program_path.is_absolute() {
        return Err(format!(
            "recipe launcher '{program}' did not resolve to an absolute path \
             (argv[0] must begin with the {{script}} token or an absolute path)"
        ));
    }

    // Compose the child PATH once: a guaranteed tool-dir prefix ahead of the
    // daemon's own PATH. Used for BOTH the preflight and the spawn, so what we
    // verify is exactly what the child resolves against.
    let child_path = convert_path_with_prefix();

    // §5.4 call-time preflight (the §4.3 drift backstop): the launcher and every
    // external `need` must be resolvable BEFORE we spawn. A missing binary is
    // MISSING_DEP — never a partial invocation.
    if !program_path.is_file() {
        return Ok(convert_err(
            to,
            "MISSING_DEP",
            &format!(
                "file_convert {from}->{to} launcher '{program}' not found. Recipe present, launcher missing."
            ),
        ));
    }
    for need in &recipe.needs {
        if find_on_path(need, &child_path).is_none() {
            return Ok(convert_err(
                to,
                "MISSING_DEP",
                &format!(
                    "file_convert {from}->{to} needs '{need}', not found on PATH. Recipe present, binary missing."
                ),
            ));
        }
    }
    // ANAI-287 `needs_files`: a plain existence check, because `needs` cannot
    // see a missing library. `find_on_path` searches PATH for an EXECUTABLE, so
    // a script-backed recipe whose interpreter is present but whose module is
    // gone passes `needs` and then dies at import. This closes that gap before
    // the spawn instead of after it.
    for need_file in &recipe.needs_files {
        if !Path::new(need_file).exists() {
            return Ok(convert_err(
                to,
                "MISSING_DEP",
                &format!(
                    "file_convert {from}->{to} needs the file '{need_file}', which does not exist. \
                     Recipe present, dependency missing. (This is a `needs_files` entry: an \
                     existence check for something PATH cannot find, such as an installed module.)"
                ),
            ));
        }
    }

    // ANAI-286: resolve the spawn deadline. Value from the recipe (a property of
    // the conversion), policy from `[convert]` in config.toml (a property of the
    // operator). Neither is per-agent. Before this existed the spawn was awaited
    // unconditionally, so a hung launcher hung the calling turn with it.
    let convert_cfg = crate::convert::load_convert_config(home);
    let (timeout_secs, clamped_from) = convert_cfg.resolve_timeout(recipe.timeout_secs);
    if let Some(requested) = clamped_from {
        tracing::warn!(
            from = %from,
            to = %to,
            requested_secs = requested,
            enforced_secs = timeout_secs,
            "ANAI-286: recipe timeout_secs exceeds [convert] max_timeout_secs; CLAMPED"
        );
    }

    // Spawn via argv array -- no shell, so substituted paths cannot inject shell
    // syntax. Env is cleared then repopulated with the daemon's safe vars, then
    // PATH is replaced with the guaranteed-prefixed PATH we just preflighted.
    let mut cmd = tokio::process::Command::new(program_path);
    cmd.args(&argv[1..]);
    cmd.current_dir(root);
    crate::subprocess_sandbox::sandbox_command(&mut cmd, &[]);
    cmd.env("PATH", &child_path);
    cmd.stdin(std::process::Stdio::null());
    // Load-bearing for the timeout below: dropping the `output()` future on
    // elapse must actually KILL the child. Without this the deadline would
    // return an error to the caller while leaving the runaway process alive and
    // still holding its output file open -- a worse state than the hang.
    cmd.kill_on_drop(true);

    let proc = match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        cmd.output(),
    )
    .await
    {
        Ok(result) => {
            result.map_err(|e| format!("failed to spawn conversion launcher '{program}': {e}"))?
        }
        Err(_) => {
            let asked = match clamped_from {
                Some(requested) => format!(
                    " (the recipe requested {requested}s; CLAMPED into the operator's \
                     [convert] max_timeout_secs)"
                ),
                None => String::new(),
            };
            return Ok(convert_err(
                to,
                "CONVERT_TIMEOUT",
                &format!(
                    "file_convert {from}->{to} exceeded its {timeout_secs}s deadline and was \
                     killed{asked}. No output is trustworthy: a partially-written file may exist \
                     at the output path. Raise `timeout_secs` on the recipe, or convert less \
                     input."
                ),
            ));
        }
    };

    if !proc.status.success() {
        // ANAI-287 reserved exit code 3 = "a dependency I need is not
        // importable". Only the interpreter that runs the conversion can
        // truthfully answer "is this module present", so the launcher answers it
        // and we classify from the code. MISSING_DEP and CONVERT_FAILED want
        // different retries: one is "install something", the other is "the input
        // or the options were wrong".
        if proc.status.code() == Some(3) {
            let stderr = String::from_utf8_lossy(&proc.stderr);
            return Ok(convert_err(
                to,
                "MISSING_DEP",
                &format!(
                    "file_convert {from}->{to} launcher reported a missing dependency \
                     (reserved exit code 3): {}",
                    openfang_types::truncate_str(stderr.trim(), 600)
                ),
            ));
        }
        let stderr = String::from_utf8_lossy(&proc.stderr);
        let detail = stderr.trim();
        let detail = if detail.is_empty() {
            String::from_utf8_lossy(&proc.stdout).trim().to_string()
        } else {
            detail.to_string()
        };
        return Ok(convert_err(
            to,
            "CONVERT_FAILED",
            &format!(
                "conversion command exited with {}: {}",
                proc.status,
                openfang_types::truncate_str(&detail, 600)
            ),
        ));
    }

    if !resolved_output.exists() {
        return Ok(convert_err(
            to,
            "CONVERT_FAILED",
            "conversion command reported success but produced no output file",
        ));
    }

    Ok(convert_ok(to, &output_path))
}

/// Build the `file_convert` success envelope:
/// `{ "ok": true, "format": "<fmt>", "output_path": "<path>" }`. The path is the
/// workspace-relative form the caller passed (or the derived default) — the path
/// a follow-up `file_read` can use directly.
fn convert_ok(format: &str, output_path: &str) -> String {
    serde_json::json!({
        "ok": true,
        "format": format,
        "output_path": output_path
    })
    .to_string()
}

/// Build the structured `file_convert` error envelope (§4.3):
/// `{ "ok": false, "format": "<fmt>", "error": { "code": "<CODE>", "message": "<msg>" } }`.
/// `code` is one of UNKNOWN_FORMAT, BAD_PATH, MISSING_DEP, CONVERT_FAILED,
/// UNKNOWN_PRESET, UNKNOWN_OPTION, INVALID_OPTION (ANAI-131), CONVERT_TIMEOUT
/// (ANAI-286). CONVERT_TIMEOUT is deliberately distinct from CONVERT_FAILED:
/// "it was still running" and "it exited nonzero" call for different retries.
fn convert_err(format: &str, code: &str, message: &str) -> String {
    serde_json::json!({
        "ok": false,
        "format": format,
        "error": { "code": code, "message": message }
    })
    .to_string()
}

/// Build the projected `options` sub-schema for the advertised `file_convert`
/// tool (ANAI-131) from the live recipe set, via the shared
/// `convert::project_options_schema` helper (the single source of truth both
/// schema mirrors use). On a recipe-load failure this advertises a permissive
/// object rather than panicking the tool list — the dispatcher still fail-closes
/// on every option at call time, so keeping the tool advertised (options
/// temporarily undiscoverable) preserves the core md->pdf path even under a
/// broken custom manifest.
fn file_convert_options_schema() -> serde_json::Value {
    // Delegates to the canonical projection in `convert` so the runtime schema
    // mirror and the MCP bridge (via the daemon handshake) render one and the
    // same schema — no drift between the two advertise surfaces.
    crate::convert::file_convert_options_schema()
}

/// Resolve a bare binary name against a composed PATH, returning the first
/// matching executable. Close enough to `command -v` for a preflight backstop:
/// on Unix it requires an executable bit; elsewhere an existing regular file.
/// A name that already contains a path separator is checked as-is, not searched.
fn find_on_path(bin: &str, path: &std::ffi::OsStr) -> Option<PathBuf> {
    if bin.contains('/') || bin.contains('\\') {
        let p = PathBuf::from(bin);
        return if is_executable_file(&p) {
            Some(p)
        } else {
            None
        };
    }
    for dir in std::env::split_paths(path) {
        let candidate = dir.join(bin);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Whether `path` is a regular file that is executable (Unix checks the exec
/// bit; other platforms fall back to file existence).
fn is_executable_file(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(md) if md.is_file() => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                md.permissions().mode() & 0o111 != 0
            }
            #[cfg(not(unix))]
            {
                true
            }
        }
        _ => false,
    }
}
/// Compose the child PATH for a conversion spawn: a guaranteed prefix of common
/// tool directories (which launchd may omit) ahead of the daemon's own PATH,
/// de-duplicated and joined with the platform separator. This is the ANAI-67
/// "inject a guaranteed PATH prefix" half of the hybrid launcher/PATH decision.
fn convert_path_with_prefix() -> std::ffi::OsString {
    let guaranteed = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"];
    let mut dirs: Vec<PathBuf> = guaranteed.iter().map(|d| PathBuf::from(*d)).collect();
    if let Some(existing) = std::env::var_os("PATH") {
        for p in std::env::split_paths(&existing) {
            if !dirs.contains(&p) {
                dirs.push(p);
            }
        }
    }
    std::env::join_paths(dirs).unwrap_or_else(|_| std::env::var_os("PATH").unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Patch tool
// ---------------------------------------------------------------------------

async fn tool_apply_patch(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let patch_str = input["patch"].as_str().ok_or("Missing 'patch' parameter")?;
    let root = workspace_root.ok_or("apply_patch requires a workspace root")?;
    let ops = crate::apply_patch::parse_patch(patch_str)?;
    let result = crate::apply_patch::apply_patch(&ops, root, file_policy, caller_agent_id).await;
    if result.is_ok() {
        Ok(result.summary())
    } else {
        Err(format!(
            "Patch partially applied: {}. Errors: {}",
            result.summary(),
            result.errors.join("; ")
        ))
    }
}

// ---------------------------------------------------------------------------
// Web tools
// ---------------------------------------------------------------------------

/// Legacy web fetch (no SSRF protection, no readability). Used when WebToolsContext is unavailable.
async fn tool_web_fetch_legacy(input: &serde_json::Value) -> Result<String, String> {
    let url = input["url"].as_str().ok_or("Missing 'url' parameter")?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;
    let status = resp.status();
    // Reject responses larger than 10MB to prevent memory exhaustion
    if let Some(len) = resp.content_length() {
        if len > 10 * 1024 * 1024 {
            return Err(format!("Response too large: {len} bytes (max 10MB)"));
        }
    }
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))?;
    let max_len = 50_000;
    let truncated = if body.len() > max_len {
        format!(
            "{}... [truncated, {} total bytes]",
            crate::str_utils::safe_truncate_str(&body, max_len),
            body.len()
        )
    } else {
        body
    };
    Ok(format!("HTTP {status}\n\n{truncated}"))
}

/// Legacy web search via DuckDuckGo HTML only. Used when WebToolsContext is unavailable.
async fn tool_web_search_legacy(input: &serde_json::Value) -> Result<String, String> {
    let query = input["query"].as_str().ok_or("Missing 'query' parameter")?;
    let max_results = input["max_results"].as_u64().unwrap_or(5) as usize;

    debug!(query, "Executing web search via DuckDuckGo HTML");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let resp = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("User-Agent", "Mozilla/5.0 (compatible; OpenFangAgent/0.1)")
        .send()
        .await
        .map_err(|e| format!("Search request failed: {e}"))?;

    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read search response: {e}"))?;

    // Parse DuckDuckGo HTML results
    let results = parse_ddg_results(&body, max_results);

    if results.is_empty() {
        return Ok(format!("No results found for '{query}'."));
    }

    let mut output = format!("Search results for '{query}':\n\n");
    for (i, (title, url, snippet)) in results.iter().enumerate() {
        output.push_str(&format!(
            "{}. {}\n   URL: {}\n   {}\n\n",
            i + 1,
            title,
            url,
            snippet
        ));
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Shell tool
// ---------------------------------------------------------------------------

async fn tool_shell_exec(
    input: &serde_json::Value,
    allowed_env: &[String],
    workspace_root: Option<&Path>,
    exec_policy: Option<&openfang_types::config::ExecPolicy>,
) -> Result<String, String> {
    let command = input["command"]
        .as_str()
        .ok_or("Missing 'command' parameter")?;
    // Use LLM-specified timeout, or fall back to exec policy timeout, or default 30s
    let policy_timeout = exec_policy.map(|p| p.timeout_secs).unwrap_or(30);
    let timeout_secs = input["timeout_seconds"].as_u64().unwrap_or(policy_timeout);

    // SECURITY: Determine execution strategy based on exec policy.
    //
    // In Allowlist mode (default): Use direct execution via shlex argv splitting.
    // This avoids invoking a shell interpreter, which eliminates an entire class
    // of injection attacks (encoding tricks, $IFS, glob expansion, etc.).
    //
    // In Full mode: User explicitly opted into unrestricted shell access,
    // so we use sh -c / cmd /C as before.
    let use_direct_exec = exec_policy
        .map(|p| p.mode == openfang_types::config::ExecSecurityMode::Allowlist)
        .unwrap_or(true); // Default to safe mode

    let mut cmd = if use_direct_exec {
        // SAFE PATH: Split command into argv using POSIX shell lexer rules,
        // then execute the binary directly — no shell interpreter involved.
        let argv = shlex::split(command).ok_or_else(|| {
            "Command contains unmatched quotes or invalid shell syntax".to_string()
        })?;
        if argv.is_empty() {
            return Err("Empty command after parsing".to_string());
        }
        let mut c = tokio::process::Command::new(&argv[0]);
        if argv.len() > 1 {
            c.args(&argv[1..]);
        }
        c
    } else {
        // UNSAFE PATH: Full mode — user explicitly opted in to shell interpretation.
        // Shell resolution: prefer sh (Git Bash/MSYS2) on Windows.
        #[cfg(windows)]
        let git_sh: Option<&str> = {
            const SH_PATHS: &[&str] = &[
                "C:\\Program Files\\Git\\usr\\bin\\sh.exe",
                "C:\\Program Files (x86)\\Git\\usr\\bin\\sh.exe",
            ];
            SH_PATHS
                .iter()
                .copied()
                .find(|p| std::path::Path::new(p).exists())
        };
        let (shell, shell_arg) = if cfg!(windows) {
            #[cfg(windows)]
            {
                if let Some(sh) = git_sh {
                    (sh, "-c")
                } else {
                    ("cmd", "/C")
                }
            }
            #[cfg(not(windows))]
            {
                ("sh", "-c")
            }
        } else {
            ("sh", "-c")
        };
        let mut c = tokio::process::Command::new(shell);
        c.arg(shell_arg).arg(command);
        c
    };

    // Set working directory to agent workspace so files are created there
    if let Some(ws) = workspace_root {
        cmd.current_dir(ws);
    }

    // SECURITY: Isolate environment to prevent credential leakage.
    // Hand settings may grant access to specific provider API keys.
    //
    // Operators can also forward additional vars via
    // `exec_policy.shell_env_passthrough` (issue #1169). This is the path
    // Docker users hit: their container env (TZ, GOG_*, etc.) is present
    // in PID 1 but `env_clear()` strips it. Listing names (or `"*"`) here
    // re-adds them to the child.
    let policy_env_passthrough: &[String] = exec_policy
        .map(|p| p.shell_env_passthrough.as_slice())
        .unwrap_or(&[]);
    let merged_env =
        crate::subprocess_sandbox::merge_env_passthrough(allowed_env, policy_env_passthrough);
    crate::subprocess_sandbox::sandbox_command(&mut cmd, &merged_env);

    // Ensure UTF-8 output on Windows
    #[cfg(windows)]
    cmd.env("PYTHONIOENCODING", "utf-8");

    // Prevent child from inheriting stdin (avoids blocking on Windows)
    cmd.stdin(std::process::Stdio::null());

    // Kill the child if this future is dropped (notably on the timeout below) so a
    // hung command can't outlive its tool call as an orphaned process. Covers the
    // direct child only — a command that backgrounds its own grandchildren is not
    // reaped here (would need a process group; out of scope).
    cmd.kill_on_drop(true);

    let result =
        tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output()).await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            // Truncate very long outputs to prevent memory issues
            let max_output = 100_000;
            let mut stdout_str = if stdout.len() > max_output {
                format!(
                    "{}...\n[truncated, {} total bytes]",
                    crate::str_utils::safe_truncate_str(&stdout, max_output),
                    stdout.len()
                )
            } else {
                stdout.to_string()
            };
            let stderr_str = if stderr.len() > max_output {
                format!(
                    "{}...\n[truncated, {} total bytes]",
                    crate::str_utils::safe_truncate_str(&stderr, max_output),
                    stderr.len()
                )
            } else {
                stderr.to_string()
            };

            if exit_code == 0 && stdout_str.is_empty() {
                stdout_str = "Command executed successfully".to_string();
            }

            Ok(format!(
                "Exit code: {exit_code}\n\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"
            ))
        }
        Ok(Err(e)) => Err(format!("Failed to execute command: {e}")),
        Err(_) => Err(format!(
            "Command timed out after {timeout_secs}s (process killed)"
        )),
    }
}

// ---------------------------------------------------------------------------
// Inter-agent tools
// ---------------------------------------------------------------------------

fn require_kernel(
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<&Arc<dyn KernelHandle>, String> {
    kernel.ok_or_else(|| {
        "Kernel handle not available. Inter-agent tools require a running kernel.".to_string()
    })
}

async fn tool_agent_send(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id = input["agent_id"]
        .as_str()
        .ok_or("Missing 'agent_id' parameter")?;
    let message = input["message"]
        .as_str()
        .ok_or("Missing 'message' parameter")?;

    // Check + increment inter-agent call depth
    let current_depth = AGENT_CALL_DEPTH.try_with(|d| d.get()).unwrap_or(0);
    if current_depth >= MAX_AGENT_CALL_DEPTH {
        return Err(format!(
            "Inter-agent call depth exceeded (max {}). \
             A->B->C chain is too deep. Use the task queue instead.",
            MAX_AGENT_CALL_DEPTH
        ));
    }

    AGENT_CALL_DEPTH
        .scope(std::cell::Cell::new(current_depth + 1), async {
            // ANAI-147: attribute the send to the CALLING agent so the target's
            // §9.1 says "peer agent <name>" instead of nothing. Unlike the
            // hand-written `[From: X]` body prefix this convention has leaned on,
            // it is kernel-attested and cannot be forged by message text.
            kh.send_to_agent_from(agent_id, message, caller_agent_id)
                .await
        })
        .await
}

// Wake-emission rate backstops (ANAI-111).
//
// Two sliding-window counters guard two DIFFERENT amplification modes; both
// are tunable via the `[agent_wake]` config section / `OPENFANG_AGENT_WAKE_*`
// env vars (see `openfang_types::agent_wake`) and both self-GC once
// emissions go quiet:
//
// * `wake_tree_admit` — the per-tree budget (req 10), keyed on the lineage
//   root. Since lineage threading landed (ANAI-110) the cross-hop cycle
//   (req 4) and depth (req 9) bounds enforce for real — but neither catches
//   fan-out: a tree that never repeats an agent and never exceeds the depth
//   bound can still emit `F^k` wakes. Charging the root for its whole
//   subtree's rate closes that hole.
// * `wake_emit_admit` — the coarse aggregate ceiling across ALL trees. N
//   trees each under their own per-tree budget still sum to `N * budget` of
//   fleet-wide load; this bounds that aggregate. It only ever refuses, never
//   permits — harmless defense-in-depth the per-tree budget does not subsume.

/// Process-global sliding window backing [`wake_emit_admit`]. Hoisted to module
/// scope (rather than a fn-local `static`) so the test-only
/// [`reset_wake_emit_window`] can drain it: the aggregate ceiling is a shared
/// mutable global, and the ceiling test must start from a known-empty window
/// even when a parallel test drives the real emit path (ANAI-122 — the reply-
/// right refactor added tests that exercise the real emit path the ceiling test
/// had assumed no other test touched).
static WAKE_EMIT_WINDOW: std::sync::OnceLock<
    std::sync::Mutex<std::collections::VecDeque<std::time::Instant>>,
> = std::sync::OnceLock::new();

/// Record-and-test the process-global aggregate wake-emission window.
///
/// Returns `true` if admitted (and stamps it into the window), `false` if the
/// trailing window is already at
/// [`emit_max`](openfang_types::agent_wake::emit_max) capacity — in which case
/// the caller must refuse the wake.
fn wake_emit_admit() -> bool {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    let slot = WAKE_EMIT_WINDOW.get_or_init(|| Mutex::new(VecDeque::new()));
    let window = Duration::from_secs(openfang_types::agent_wake::window_secs());
    let cap = openfang_types::agent_wake::emit_max();
    let now = Instant::now();
    // Recover rather than propagate: a poisoned lock means a prior caller
    // panicked mid-update; the queue itself is still a valid bound to enforce.
    let mut q = match slot.lock() {
        Ok(q) => q,
        Err(poisoned) => poisoned.into_inner(),
    };
    // Evict timestamps that have aged out of the trailing window.
    while let Some(&front) = q.front() {
        if now.duration_since(front) >= window {
            q.pop_front();
        } else {
            break;
        }
    }
    if q.len() >= cap {
        return false;
    }
    q.push_back(now);
    true
}

/// Test-only: drain the process-global emit window so a ceiling test can start
/// from empty regardless of what parallel emit-path tests have stamped. Pair it
/// with [`WAKE_EMIT_TEST_GUARD`] so no concurrent emit re-pollutes mid-test.
#[cfg(test)]
fn reset_wake_emit_window() {
    if let Some(slot) = WAKE_EMIT_WINDOW.get() {
        let mut q = slot.lock().unwrap_or_else(|p| p.into_inner());
        q.clear();
    }
}

/// Test-only serialization guard. The aggregate emit ceiling is a shared mutable
/// global, so the ceiling test and any test that drives the real emit path
/// (`tool_agent_reply_async`) must not interleave. Hold it across the emit-
/// touching critical section. Introduced by ANAI-122: the reply-right refactor's
/// tests newly exercise the real emit path, which the sole-consumer ceiling test
/// had assumed nothing else touched.
#[cfg(test)]
static WAKE_EMIT_TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Record-and-test the per-tree wake-emission window for one lineage `root`
/// (req 10). Returns `true` if this tree is under its
/// [`tree_budget_max`](openfang_types::agent_wake::tree_budget_max) for the
/// trailing window (and stamps the emission), `false` if the root's window is
/// already at capacity — in which case the caller must refuse the wake.
///
/// Keyed on the stable lineage root, so every hop of a tree charges the SAME
/// budget no matter how wide the fan-out. A root's key is dropped once its
/// window empties, so the map self-GCs and long-quiet trees leave no residue.
fn wake_tree_admit(root: &str) -> bool {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    static WINDOWS: OnceLock<Mutex<HashMap<String, VecDeque<Instant>>>> = OnceLock::new();
    let slot = WINDOWS.get_or_init(|| Mutex::new(HashMap::new()));
    let window = Duration::from_secs(openfang_types::agent_wake::window_secs());
    let cap = openfang_types::agent_wake::tree_budget_max();
    let now = Instant::now();
    let mut map = match slot.lock() {
        Ok(m) => m,
        Err(poisoned) => poisoned.into_inner(),
    };
    let q = map.entry(root.to_string()).or_default();
    // Evict timestamps that have aged out of the trailing window.
    while let Some(&front) = q.front() {
        if now.duration_since(front) >= window {
            q.pop_front();
        } else {
            break;
        }
    }
    if q.len() >= cap {
        // At capacity: leave the (non-empty) window in place for the next call.
        return false;
    }
    q.push_back(now);
    // Self-GC: an emptied window would only be created again on the next
    // emission, so never leave a dead root's empty deque behind. (After a
    // push the current root is non-empty, so this only reaps OTHER roots that
    // aged fully out — bounded, and keeps the map from growing without limit.)
    map.retain(|_, q| !q.is_empty());
    true
}

/// (continued) Queue an asynchronous wake for another agent (fire-and-forget).
///
/// Unlike [`tool_agent_send`], this does **not** block on the target's loop. It
/// frames a [`WakeEnvelope`](openfang_types::wake::WakeEnvelope) — target,
/// sender, message, lineage, typed `TurnTrigger` — into the task-queue payload
/// and returns as soon as the wake is enqueued. The kernel's wake-consumer
/// claims the task on its own task and re-enters the send funnel for the
/// target, so the caller never holds the target's per-agent lock. That
/// eliminates the `A -> B -> A` head-of-line block the synchronous path has.
///
/// Capability: gated by the standard `capabilities.tools` allowlist — the
/// boolean cap enforced in [`execute_tool`]. An agent without
/// `agent_send_async` in its declared tools is rejected before reaching here,
/// so no second gate is needed.
async fn tool_agent_send_async(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    use openfang_types::turn::TurnTrigger;
    use openfang_types::wake::{WakeEnvelope, DEFAULT_MAX_WAKE_DEPTH, WAKE_TASK_PREFIX};

    let kh = require_kernel(kernel)?;
    let target_raw = input["agent_id"]
        .as_str()
        .ok_or("Missing 'agent_id' parameter")?;
    let message = input["message"]
        .as_str()
        .ok_or("Missing 'message' parameter")?;
    let sender = caller_agent_id.ok_or("agent_send_async requires a caller identity (sender)")?;

    // ANAI-123: optional surfacing route. When set, it rides the whole
    // round-trip and origin's leg-4 turn auto-posts the delegated answer to
    // this channel (see `WakeEnvelope::surface_to`). Inert data on this leg —
    // it only takes effect if the callee replies. Encoded "<channel>:<recip>".
    let surface_to = input["surface_to"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    // ANAI-201: the sender's deadline for this correlation. This is what makes
    // the reply guarantee *bounded* rather than merely eventual — the
    // wake-consumer races the callee's whole turn against it and, on elapse,
    // aborts the turn and mints a `Timeout` reply.
    //
    // Clamped HERE, once, and the result stamped into the durable envelope, so
    // the queue row is the single source of truth for this correlation's
    // deadline. Deriving it at enforcement time instead would let a config edit
    // silently move the deadline of an already-dispatched send.
    //
    // A non-integer / negative value is treated as "omitted" rather than
    // rejected: refusing the whole send over a malformed optional knob would
    // trade a bounded reply for no reply at all, which is backwards.
    let requested_timeout = input["timeout_secs"].as_u64();
    let timeout_secs = openfang_types::async_reply::clamp_timeout_secs(requested_timeout);
    // Recorded only when the clamp actually rewrote the request, so the common
    // in-band case stays absent on the wire. Two consumers: the `Timeout` body
    // (so an orchestrator is told its deadline was moved rather than inferring
    // it was honored) and the requested-vs-enforced distribution these knobs
    // are meant to be tuned from once real data exists.
    let requested_timeout_secs = requested_timeout.filter(|r| *r != timeout_secs);

    // Canonicalize the target to a stable agent-id BEFORE the cycle check.
    // `sender` is always a UUID (the caller's agent id); a caller may address
    // `target` by NAME. would_cycle is a plain string compare, so without
    // canonicalization a self-wake-by-name (name != uuid) slips the guard and
    // run_woken_agent_loop resolves the name straight back to the same agent.
    // Resolving to the registered id here makes self-wake/cycle detection sound
    // and validates the target exists before anything is enqueued.
    // One agent snapshot serves BOTH target resolution and (below) the sender's
    // own-binding lookup for the ANAI-125 surface default.
    let agents = kh.list_agents();
    let target = agents
        .iter()
        .find(|a| a.id == target_raw || a.name == target_raw)
        .map(|a| a.id.clone())
        .ok_or_else(|| format!("Async wake target not found: {target_raw}"))?;

    // ANAI-210 (failure class B): refuse a wake the target structurally cannot
    // serve, BEFORE anything durable exists.
    //
    // The reply guarantee bounds how long a sender waits; it cannot make the
    // answer useful. Sending `sleep 60` to an agent with no `shell_exec` (the
    // observed 2026-08-23 case) costs the sender its entire deadline to learn a
    // static fact about the target's manifest. This turns that into an
    // immediate, side-effect-free refusal.
    //
    // Placement is the point: ahead of the surface default, the lineage/cycle
    // and depth checks, both wake budgets, and `wake_post`. Nothing has been
    // charged and no correlation has been minted, so the refusal leaves the
    // system byte-identical to never having called — which is what makes
    // re-sending a corrected request safe.
    //
    // Fail-OPEN by design: `agent_tool_names` returning `None` means "cannot
    // determine" (unresolvable agent, or a handle that does not implement the
    // lookup), and is treated as no evidence rather than as an empty tool set.
    // A pre-flight that invents refusals is worse than one that occasionally
    // lets a doomed send through to the deadline it would have hit anyway.
    // Omitting `requires_tools` skips the check entirely, so every existing
    // caller is unaffected.
    let required_tools: Vec<String> = input["requires_tools"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if !required_tools.is_empty() {
        if let Some(available) = kh.agent_tool_names(&target) {
            let missing: Vec<&str> = required_tools
                .iter()
                .filter(|req| !available.iter().any(|have| have == *req))
                .map(String::as_str)
                .collect();
            if !missing.is_empty() {
                warn!(
                    target = %target,
                    sender = %sender,
                    missing = ?missing,
                    "ANAI-210: async wake refused at pre-flight — target lacks required tools"
                );
                return Err(format!(
                    "Async wake NOT sent: '{target_raw}' lacks required tool(s): {}. \
                     No correlation was minted, no deadline consumed, and no side effects \
                     exist — this is a pre-flight refusal, not a failed delivery. Re-send \
                     to a target that has them, or drop the requirement if the target can \
                     do the work another way.",
                    missing.join(", ")
                ));
            }
        } else {
            info!(
                target = %target,
                "agent_send_async: requires_tools pre-flight skipped — target tool set \
                 unavailable, proceeding (ANAI-210 fails open)"
            );
        }
    }

    // ANAI-125: default the surfacing route to the ORIGINATOR's OWN channel
    // binding when the caller omitted `surface_to`. This makes the common case
    // — "the delegated reply comes back to my home channel" — the default,
    // instead of requiring every caller to pass `surface_to` by hand (leg 4
    // silently no-op'd precisely because an unbound-in-turn originate had
    // nothing to stamp). An EXPLICIT route always wins; a bindingless sender
    // still resolves to `None`, preserving a pure fire-and-forget wake. The
    // route rides the round-trip identically to an explicit one from here on.
    let surface_to = surface_to.or_else(|| {
        let sender_name = agents.iter().find(|a| a.id == sender).map(|a| a.name.as_str())?;
        let route = kh.channel_binding_route(sender_name)?;
        info!(
            sender = %sender,
            surface_to = %route,
            "agent_send_async: no explicit surface_to — defaulting to originator's own channel binding (ANAI-125)"
        );
        Some(route)
    });

    // Lineage (ANAI-110): thread the REAL inbound wake chain when this call is
    // itself running inside a woken turn (see `resolve_wake_base_lineage`). The
    // inbound chain's `current` is already THIS agent, so extending it below
    // yields the true root->...->this->target chain — cross-hop cycle (req 4)
    // and depth (req 9) now enforce for real, not just self-wake / single-hop.
    // On an origin turn (no task-local) it roots at the sender, exactly as v1.
    let lineage = resolve_wake_base_lineage(sender);
    if lineage.would_cycle(&target) {
        return Err(format!(
            "Refusing async wake: '{target}' is already in the call chain (cycle)."
        ));
    }
    let next_lineage = lineage.extended(target.as_str());
    if next_lineage.exceeds_depth(DEFAULT_MAX_WAKE_DEPTH) {
        return Err(format!(
            "Refusing async wake: chain depth would exceed {DEFAULT_MAX_WAKE_DEPTH}."
        ));
    }

    // Per-tree budget (ANAI-111, req 10): charge this wake to its lineage root.
    // The cycle (req 4) and depth (req 9) checks above now enforce cross-hop
    // for real (ANAI-110 threaded the inbound chain), but neither catches
    // *fan-out* — a tree that never repeats an agent and never exceeds the
    // depth bound can still emit F^k wakes. Keying a sliding-window budget on
    // the stable root makes the root pay for its whole subtree's emission rate.
    // `root()` is always Some here (the chain is non-empty after `extended`);
    // on the impossible None we skip only the per-tree check, never the global.
    if let Some(root) = next_lineage.root() {
        if !wake_tree_admit(root) {
            return Err(format!(
                "Refusing async wake: per-tree wake budget exceeded for root \
                 '{root}' ({} wakes / {}s window). One wake tree is fanning out \
                 too fast; the budget clears once its emissions go quiet.",
                openfang_types::agent_wake::tree_budget_max(),
                openfang_types::agent_wake::window_secs()
            ));
        }
    }

    // Aggregate cross-tree ceiling (ANAI-111): a coarse fleet-wide backstop on
    // top of the per-tree budget. N distinct trees each under their own budget
    // still sum to N x budget of total load; this bounds that aggregate. It
    // only ever refuses, never permits — defense-in-depth. See `wake_emit_admit`.
    if !wake_emit_admit() {
        return Err(format!(
            "Refusing async wake: aggregate wake-emission ceiling exceeded \
             ({} wakes / {}s window across all trees). It clears once emissions \
             go quiet.",
            openfang_types::agent_wake::emit_max(),
            openfang_types::agent_wake::window_secs()
        ));
    }

    let envelope = WakeEnvelope {
        target: target.clone(),
        sender: sender.to_string(),
        message: message.to_string(),
        lineage: next_lineage,
        // Genuine delegated peer content → AgentCall. Survives capture; never
        // Heartbeat (which would make the woken turn capture-droppable).
        trigger: TurnTrigger::AgentCall,
        // origin threading is a documented follow-up (audit finding #3): a wake
        // that raises an approval prompt has no inbound route yet.
        origin: None,
        // Origination, not a reply — the consumer WILL grant this target a
        // one-shot reply-right (ANAI-122). Only `agent_reply_async` sets true.
        is_reply: false,
        // ANAI-123: surfacing route rides outbound so the callee's reply-right
        // token inherits it (minted in `run_woken_agent_loop`). None on the
        // wire when unset.
        surface_to: surface_to.clone(),
        // Not a reply at all, so the kind is inert here (ANAI-199).
        reply_kind: openfang_types::wake::ReplyKind::Explicit,
        // ANAI-201: the clamped deadline rides the durable payload; see above.
        timeout_secs: Some(timeout_secs),
        requested_timeout_secs,
    };

    if let Some(route) = surface_to.as_deref() {
        info!(
            target = %target,
            sender = %sender,
            surface_to = %route,
            "agent_send_async: surfacing route set — callee's reply will auto-post here (ANAI-123)"
        );
    }

    let payload = envelope
        .to_payload()
        .map_err(|e| format!("Failed to serialize wake envelope: {e}"))?;

    // Reserved title prefix marks this as a wake task so the kernel's
    // wake-consumer claims it (and ordinary task_claim skips it). Enqueue via
    // the PRIVILEGED wake_post path — ordinary task_post rejects this prefix, so
    // a forged wake cannot enter the queue through the generic task tool.
    let title = format!("{WAKE_TASK_PREFIX}{target}");
    let task_id = kh
        .wake_post(&title, message, Some(&target), Some(sender), &payload)
        .await?;

    // ANAI-147: report the queue this wake just joined. "Queued" alone reads as
    // "will run", but a caller at its per-caller in-flight cap has its wake sit
    // `pending` behind that cap — the failure mode that cost a day of A/B
    // probing to distinguish from a dropped message. Depth is diagnostic only:
    // a lookup failure must never fail an enqueue that already succeeded.
    let depth_note = match kh.wake_queue_depth(sender).await {
        Ok((pending, in_flight)) => format!(
            " Queue for this sender: {pending} pending, {in_flight} in flight. \
             Check outcome with task_list (task_id={task_id})."
        ),
        Err(_) => String::new(),
    };

    // ANAI-201: state the enforced deadline, and say so explicitly when it is
    // NOT the number the caller passed. An orchestrator that asked for 30s and
    // silently got 60s would mis-plan every downstream step on a deadline it
    // never agreed to; the clamp is policy, but hiding it is a bug.
    let deadline_note = match requested_timeout_secs {
        Some(asked) => format!(
            " You will receive exactly one reply for this correlation. Deadline: {timeout_secs}s \
             (you requested {asked}s; CLAMPED into the operator's configured band). If the \
             target has not replied by then its turn is aborted and you get a timeout reply."
        ),
        None => format!(
            " You will receive exactly one reply for this correlation. Deadline: {timeout_secs}s; \
             if the target has not replied by then its turn is aborted and you get a timeout \
             reply."
        ),
    };

    Ok(format!(
        "Async wake queued for '{target}' (task {task_id}). \
         The target runs on its own; no reply is returned inline.{deadline_note}{depth_note}"
    ))
}

/// Send a **one-shot terminal reply** to the agent that woke this turn (ANAI-122,
/// leg 3 of the four-step round-trip: fleet -> origin).
///
/// Unlike [`tool_agent_send_async`] this is NOT an origination and takes no
/// target: the reply target is fixed by the [`ReplyRight`] token that the
/// wake-consumer minted into task-local scope for this turn. The tool is
/// advertised fleet-wide but **inert without that token** — outside a woken
/// turn (or inside a turn that was itself woken by a reply) there is no token
/// and the call refuses. The token authorizes exactly one reply to the
/// initiator and is consumed here, so it cannot be used to originate a wake to
/// anyone else or to fan out.
///
/// Terminal-edge rules (why the normal wake guards do not apply):
/// * **Cycle (req 4):** the reply targets an ancestor (the initiator), which
///   [`WakeLineage::would_cycle`](openfang_types::wake::WakeLineage) would
///   refuse. The one-shot token replaces the cycle guard on this edge — no ring
///   can form because the reply grants no further reply-right.
/// * **Depth (req 9):** the reply roots a FRESH single-element lineage at the
///   replier, so depth is 1 — it can never be over-deep.
/// * **Per-tree budget (req 10):** deliberately NOT charged. It keys on the
///   initiator's root, which the initiator's own fan-out may have exhausted;
///   gagging a legitimate terminal answer for the initiator's spending buys no
///   safety, since the reply is one-shot and cannot itself fan out.
/// * **Aggregate ceiling:** deliberately NOT enforced either, as of ANAI-200.
///   It guards amplification, and a reply cannot amplify — it is 1:1 with a
///   wake the ceiling already charged, and terminal. Worse, the gate used to
///   sit AFTER the token was consumed, so a trip destroyed the debt without
///   emitting anything and the kernel's turn-end auto-close (ANAI-198) saw
///   nothing owed: a ceiling trip turned a real answer into permanent silence,
///   exactly under the load that makes the guarantee matter. See the inline
///   note at the former gate site and `Kernel::emit_synthetic_reply`.
async fn tool_agent_reply_async(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    use openfang_types::turn::TurnTrigger;
    use openfang_types::wake::{WakeEnvelope, WakeLineage, WAKE_TASK_PREFIX};

    let kh = require_kernel(kernel)?;
    let sender = caller_agent_id.ok_or("agent_reply_async requires a caller identity (sender)")?;
    let message = input["message"]
        .as_str()
        .ok_or("Missing 'message' parameter")?;

    // One-shot reply-right, read from the KERNEL-HELD registry via the handle we
    // already hold. This REPLACES the `WAKE_REPLY_RIGHT` task-local, which the
    // process/IPC boundary severed: a subprocess-driven agent (e.g. Claude Code)
    // calls this tool on the bridge-IPC handler task, NOT the kernel
    // wake-dispatch task the task-local lived on, so `try_with` always
    // fail-closed for out-of-process agents — i.e. nearly the whole fleet. The
    // kernel handle crosses both drivers, so this works for native and
    // subprocess agents alike. `take_reply_right` CONSUMES the entry
    // (consume-on-read), so a second call this turn — or on any later turn —
    // finds `None` and refuses, keeping the reply strictly one-shot. Absent for:
    // an origin turn (channel/cron/API — never minted), a reply-woken turn
    // (terminal — no right minted), or an already-used right. In every case the
    // tool is inert without a live token — the default-safe property that lets
    // it live in DEFAULT_ALLOWED without a manifest grant.
    let right = kh.take_reply_right(sender).ok_or_else(|| {
        "agent_reply_async is only valid once inside a turn woken by another \
         agent's async wake: no one-shot reply-right is in scope here (an origin \
         channel/cron/API turn, an already-used right, or a reply-woken terminal \
         turn)."
            .to_string()
    })?;

    // The reply target is FIXED by the token (the initiator that woke us). The
    // tool takes no `agent_id`, so a woken agent can only ever answer its own
    // initiator — it cannot address anyone else. That is the whole safety story.
    let target = right.reply_to().to_string();

    // ANAI-123: inherit the surfacing route the origin stamped on the inbound
    // wake (baked into the token at mint time). The callee never sees or
    // chooses it — it just rides through so origin's leg-4 turn knows where to
    // auto-post. `None` when the origin dispatched without `surface_to`.
    let surface_to = right.surface_to().map(str::to_string);

    // ANAI-200: NO rate gate on this path — neither the per-tree budget nor the
    // aggregate ceiling. Both guard *amplification*, and a reply cannot amplify:
    // the one-shot right was minted by an already-admitted wake and consumed
    // above, so this emission is 1:1 bounded by a wake the gates already
    // charged, and `is_reply = true` makes it terminal (the consumer mints no
    // further right, so it cannot bounce).
    //
    // The ceiling used to be checked HERE, after `take_reply_right` — which
    // meant a refusal ate the token and emitted nothing, so the turn-end
    // sweep in `run_woken_agent_loop` saw the debt already settled and never
    // auto-closed (ANAI-198). A ceiling trip converted a real answer into
    // permanent silence for that correlation: precisely the failure this stack
    // exists to kill, firing exactly when the fleet is busy enough to need the
    // guarantee. Reordering the check ahead of the take would only downgrade
    // answers to `AutoClose` fallbacks — and the daemon's synthesized replies
    // are ungated anyway, so the ceiling would suppress the good message while
    // admitting the worse one. Dropping it is the correct fix, and it matches
    // the reasoning already documented on `Kernel::emit_synthetic_reply`.

    // Terminal edge: root a FRESH single-element lineage at the replier. This is
    // the *completion* of a correlation, not an extension of the inbound chain;
    // extending would form the real cycle [origin,...,callee,origin] that
    // `would_cycle` rightly refuses. `is_reply = true` tells the consumer to
    // grant NO further reply-right, so the initiator's leg-4 turn is a leaf: it
    // surfaces (channel_send) or does fresh work, but cannot reply-bounce.
    let envelope = WakeEnvelope {
        target: target.clone(),
        sender: sender.to_string(),
        message: message.to_string(),
        lineage: WakeLineage::root_at(sender),
        trigger: TurnTrigger::AgentCall,
        origin: None,
        is_reply: true,
        // ANAI-123: carry the surfacing route back to origin (leg 3->4).
        surface_to: surface_to.clone(),
        // ANAI-199: an agent authored this body. Every other kind is minted by
        // the daemon on the callee's behalf, and only the daemon may set them —
        // this is the one construction site that is allowed to say `Explicit`.
        reply_kind: openfang_types::wake::ReplyKind::Explicit,
        // ANAI-201: a reply-woken turn mints no reply-right, so there is no
        // debt on leg 4 for a deadline to bound the payment of. Leave it unset
        // and let dispatch apply the configured default: leg 4 is still a real
        // turn that can hang, and bounding it keeps the queue row from sitting
        // `in_progress` until the far coarser stale-wake reaper notices.
        timeout_secs: None,
        requested_timeout_secs: None,
    };

    debug!(
        target = %target,
        sender = %sender,
        surface_to = surface_to.as_deref().unwrap_or("<none>"),
        correlation = right.correlation(),
        "agent_reply_async: queuing terminal reply (ANAI-122/123)"
    );

    let payload = envelope
        .to_payload()
        .map_err(|e| format!("Failed to serialize reply envelope: {e}"))?;

    // Same reserved-prefix + privileged wake_post path as origination: the
    // consumer claims it as a wake, ordinary task_claim skips it, and a forged
    // reply cannot enter through the generic task tool.
    let title = format!("{WAKE_TASK_PREFIX}{target}");
    let task_id = kh
        .wake_post(&title, message, Some(&target), Some(sender), &payload)
        .await?;

    Ok(format!(
        "Async reply queued for initiator '{target}' (task {task_id}, \
         correlation {}). Terminal — no further reply-right is granted, so the \
         initiator surfaces or continues but cannot bounce back.",
        right.correlation()
    ))
}

async fn tool_agent_spawn(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    parent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let manifest_toml = input["manifest_toml"]
        .as_str()
        .ok_or("Missing 'manifest_toml' parameter")?;
    let (id, name) = kh.spawn_agent(manifest_toml, parent_id).await?;
    Ok(format!(
        "Agent spawned successfully.\n  ID: {id}\n  Name: {name}"
    ))
}

fn tool_agent_list(kernel: Option<&Arc<dyn KernelHandle>>) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agents = kh.list_agents();
    if agents.is_empty() {
        return Ok("No agents currently running.".to_string());
    }
    let mut output = format!("Running agents ({}):\n", agents.len());
    for a in &agents {
        output.push_str(&format!(
            "  - {} (id: {}, state: {}, model: {}:{})\n",
            a.name, a.id, a.state, a.model_provider, a.model_name
        ));
    }
    Ok(output)
}

fn tool_agent_kill(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id = input["agent_id"]
        .as_str()
        .ok_or("Missing 'agent_id' parameter")?;
    kh.kill_agent(agent_id)?;
    Ok(format!("Agent {agent_id} killed successfully."))
}

fn tool_agent_activate(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id = input["agent_id"]
        .as_str()
        .ok_or("Missing 'agent_id' parameter")?;
    let name = kh.activate_agent(agent_id)?;
    Ok(format!(
        "Agent '{name}' activated. It is now Running and ready to receive messages."
    ))
}

// ---------------------------------------------------------------------------
// Memory tools (ANAI-165: agent-scoped by default)
// ---------------------------------------------------------------------------

fn tool_memory_store(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let key = input["key"].as_str().ok_or("Missing 'key' parameter")?;
    let value = input.get("value").ok_or("Missing 'value' parameter")?;
    kh.memory_store(caller_agent_id, key, value.clone())?;
    match key.strip_prefix(crate::kernel_handle::SHARED_KEY_PREFIX) {
        Some(bare) => Ok(format!(
            "Stored value under key '{bare}' in SHARED memory (visible to all agents)."
        )),
        None => Ok(format!(
            "Stored value under key '{key}' in your own memory."
        )),
    }
}

/// Default and ceiling for `memory_recall`'s `limit`.
///
/// The ceiling is re-clamped kernel-side as well; this copy exists so the
/// value reaching the handle is already sane and so a non-kernel implementor
/// cannot be handed 10_000.
const MEMORY_RECALL_DEFAULT_LIMIT: usize = 5;
const MEMORY_RECALL_MAX_LIMIT: usize = 25;

/// ANAI-166: exact-key lookup, unchanged from the pre-search `memory_recall`
/// including the ANAI-165 shared-namespace fallback and its label.
fn memory_recall_by_key(
    kh: &Arc<dyn KernelHandle>,
    caller_agent_id: Option<&str>,
    key: &str,
) -> Result<String, String> {
    if let Some(val) = kh.memory_recall(caller_agent_id, key)? {
        return Ok(serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string()));
    }

    // ANAI-165 migration path. Every pre-scoping write from every agent landed
    // in the one shared namespace, so a bare miss here is ambiguous: the key
    // may simply predate scoping. Retry against shared and, on a hit, SAY SO —
    // an unlabelled fallback would quietly hand this agent another agent's
    // value and read exactly like its own memory. Store never falls back.
    if !key.starts_with(crate::kernel_handle::SHARED_KEY_PREFIX) {
        let shared_key = format!("{}{key}", crate::kernel_handle::SHARED_KEY_PREFIX);
        if let Some(val) = kh.memory_recall(caller_agent_id, &shared_key)? {
            let body = serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string());
            return Ok(format!(
                "[not in your own memory — found in the SHARED namespace, which may have been \
                 written by another agent; re-store it under '{key}' if it is yours]\n{body}"
            ));
        }
    }
    Ok(format!("No value found for key '{key}'."))
}

/// ANAI-166 (ADR 0002 §2.2/§2.3): the single read.
///
/// Two shapes behind one name. `key` is the pre-existing exact lookup and
/// keeps its exact behaviour; `query` is retrieval over the caller's episodic
/// memory. Merging them into one tool rather than adding `memory_search` is
/// principle 2 — one door, so the scope and (in stage 3) superseded filters
/// are enforced in exactly one place.
///
/// Exactly-one-of is enforced here rather than in the JSON schema because the
/// schema crosses the MCP bridge and several provider tool formats, and
/// `anyOf`/`oneOf` survive that trip unevenly. A handler-side check with a
/// usable error message is worth more than a constraint that silently
/// evaporates in re-serialization.
async fn tool_memory_recall(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let key = input["key"]
        .as_str()
        .map(str::trim)
        .filter(|k| !k.is_empty());
    let query = input["query"]
        .as_str()
        .map(str::trim)
        .filter(|q| !q.is_empty());

    match (key, query) {
        (Some(key), None) => memory_recall_by_key(kh, caller_agent_id, key),
        (None, Some(query)) => {
            let limit = input["limit"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(MEMORY_RECALL_DEFAULT_LIMIT)
                .clamp(1, MEMORY_RECALL_MAX_LIMIT);
            let scope = input["scope"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let kind = input["kind"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty());

            let found = kh
                .memory_search(caller_agent_id, query, scope, kind, limit)
                .await?;
            Ok(format_recall_hits(query, &found))
        }
        (Some(_), Some(_)) => Err(
            "Pass either 'query' (to search by meaning) or 'key' (to look up an exact stored \
             key), not both. If you want a keyed value, use 'key'; otherwise describe what you \
             are looking for in 'query'."
                .to_string(),
        ),
        (None, None) => Err(
            "Missing parameter: pass 'query' to search your memory, or 'key' to look up an exact \
             stored key."
                .to_string(),
        ),
    }
}

/// Render search hits for a model to read.
///
/// Plain text rather than raw JSON: this is prompt content, and a pretty
/// JSON array of fragments spends context on punctuation. The empty case says
/// which search mode ran, because "nothing found" from a degraded text search
/// means something different from "nothing found" from a semantic one.
fn format_recall_hits(query: &str, found: &serde_json::Value) -> String {
    let mode = found.get("mode").and_then(|m| m.as_str()).unwrap_or("text");
    let results = found
        .get("results")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();

    if results.is_empty() {
        return match mode {
            "semantic" => format!("No memories found for '{query}'."),
            _ => format!(
                "No memories found for '{query}'. (Text matching only — no embedding provider is \
                 configured, so this searched for the literal words, not the meaning.)"
            ),
        };
    }

    let mut out = format!(
        "{} memor{} for '{query}' ({mode} search):\n",
        results.len(),
        if results.len() == 1 { "y" } else { "ies" }
    );
    for hit in &results {
        let content = hit.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let created = hit
            .get("created_at")
            .and_then(|c| c.as_str())
            .unwrap_or("?");
        let kind = hit.get("kind").and_then(|k| k.as_str()).unwrap_or("turn");
        // ANAI-270: a note carries its short id — the handle `memory_note`'s
        // `supersedes` takes. Other kinds cannot be superseded that way, so
        // their ids would be noise.
        let id = match (kind, hit.get("id").and_then(|i| i.as_str())) {
            ("note", Some(id)) => format!(" · id:{}", openfang_memory::semantic::short_note_id(id)),
            _ => String::new(),
        };
        out.push_str(&format!("\n[{created} · {kind}{id}]\n{content}\n"));
    }
    out
}

/// ANAI-166 (ADR 0002 §2.2): the cheap write.
async fn tool_memory_note(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let text = input["text"]
        .as_str()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or("Missing 'text' parameter — a note with no content is not a note")?;
    let tags: Vec<String> = input["tags"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let supersedes = note_supersedes(input);
    if supersedes.is_empty() {
        let outcome = kh
            .memory_note_with_neighbours(caller_agent_id, text, &tags)
            .await?;
        return Ok(render_note_with_neighbours(&outcome));
    }

    let outcome = kh
        .memory_note_superseding(caller_agent_id, text, &tags, &supersedes)
        .await?;
    Ok(render_note_supersession(&outcome))
}

/// Pull `supersedes` out of a `memory_note` call (ANAI-270).
///
/// Takes a list, and forgives a bare string: a model naming one note will
/// sometimes pass `"1a2b3c4d"` rather than `["1a2b3c4d"]`, and refusing that
/// over punctuation would make the agent retry a call whose intent is plain.
fn note_supersedes(input: &serde_json::Value) -> Vec<String> {
    let raw: Vec<&str> = match &input["supersedes"] {
        serde_json::Value::String(s) => vec![s.as_str()],
        serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
        _ => Vec::new(),
    };
    raw.into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Render a plain note write and, when there are any, the author's existing
/// notes closest to it (ANAI-270 step 3).
///
/// The note is already written when this is read, so the repair it offers is
/// one more write that retires BOTH the old note and this one — the single
/// path that leaves exactly one current note without a separate link tool.
/// Advisory throughout: similar is not replaces, and the message says so.
fn render_note_with_neighbours(outcome: &serde_json::Value) -> String {
    use openfang_memory::semantic::{short_note_id, NOTE_DUPLICATE_SCORE};
    let id = short_note_id(outcome["id"].as_str().unwrap_or("?")).to_string();
    let mut out = format!(
        "Noted (id:{id}). It is attached to your current episode and will surface in \
         memory_recall."
    );
    let neighbours = outcome["neighbours"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if neighbours.is_empty() {
        return out;
    }

    out.push_str("\n\nYour existing notes closest to this one:");
    for n in &neighbours {
        let score = n["score"].as_f64().unwrap_or(0.0);
        let flag = if score >= f64::from(NOTE_DUPLICATE_SCORE) {
            " LIKELY DUPLICATE —"
        } else {
            ""
        };
        out.push_str(&format!(
            "\n  - id:{} ({score:.2}, {} chars) —{flag} \"{}\"",
            short_note_id(n["id"].as_str().unwrap_or("?")),
            n["chars"].as_u64().unwrap_or(0),
            n["preview"].as_str().unwrap_or(""),
        ));
    }
    let first = short_note_id(neighbours[0]["id"].as_str().unwrap_or("?")).to_string();
    out.push_str(&format!(
        "\n\nThis is a similarity guess, not a verdict — close notes can both be true. \
         If this note restates or corrects one of them, merge them: call memory_note \
         once more with the combined text and supersedes: [\"{first}\", \"{id}\"] (the \
         old id and this one). Both retire and one current note remains. Carry forward \
         everything from either that is still true. If they are different claims, do \
         nothing."
    ));
    out
}

/// Below this fraction of the largest note it replaces, a successor is
/// called out as possibly incomplete (ANAI-270).
///
/// Supersession is whole-note: a note that fixes one wrong line must carry
/// every line that is still true, because the old text leaves recall. A
/// successor under half the size of what it replaced is the cheap tell that
/// it did not. Advisory only — a shorter note is sometimes the right answer,
/// so this warns and never blocks.
const NOTE_SHRINK_WARN_RATIO: f64 = 0.5;

/// Render the outcome of a superseding note write for the model to read.
fn render_note_supersession(outcome: &serde_json::Value) -> String {
    use openfang_memory::semantic::short_note_id;
    let id = outcome["id"].as_str().unwrap_or("?");
    let chars = outcome["chars"].as_u64().unwrap_or(0);
    let superseded = outcome["superseded"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let raced = outcome["raced"].as_array().cloned().unwrap_or_default();

    let mut out = format!("Noted (id:{}, {chars} chars).", short_note_id(id));
    if !superseded.is_empty() {
        let listed: Vec<String> = superseded
            .iter()
            .map(|s| {
                format!(
                    "id:{} ({} chars)",
                    short_note_id(s["id"].as_str().unwrap_or("?")),
                    s["chars"].as_u64().unwrap_or(0)
                )
            })
            .collect();
        out.push_str(&format!(
            " It supersedes {}: {} no longer surface{} in recall.",
            listed.join(", "),
            if superseded.len() == 1 {
                "that note"
            } else {
                "those notes"
            },
            if superseded.len() == 1 { "s" } else { "" },
        ));
        let largest = superseded
            .iter()
            .filter_map(|s| s["chars"].as_u64())
            .max()
            .unwrap_or(0);
        if largest > 0 && (chars as f64) < (largest as f64) * NOTE_SHRINK_WARN_RATIO {
            out.push_str(&format!(
                "\nCHECK: this note is much shorter than what it replaces ({chars} vs {largest} \
                 chars). If only part of the old note was wrong, the new one must carry the \
                 corrected part AND everything from the old note that is still true — the old \
                 text no longer surfaces in recall. If you dropped something, write a fuller \
                 note with supersedes: [\"{}\"]. If the shorter note is complete, ignore this.",
                short_note_id(id)
            ));
        }
    }
    if !raced.is_empty() {
        let ids: Vec<&str> = raced
            .iter()
            .filter_map(|r| r.as_str())
            .map(short_note_id)
            .collect();
        out.push_str(&format!(
            "\nNot superseded: {} — another write retired {} first. Your note is saved; \
             recall for the current version if it matters.",
            ids.join(", "),
            if ids.len() == 1 { "it" } else { "them" },
        ));
    }
    out
}

// --- Tier-3 fact tools (ANAI-204, ADR 0001 §2.3) --------------------------

/// Default number of superseded versions `memory_history` returns.
///
/// Small on purpose. The common question is "what did this used to say", not
/// "give me the whole life of this slot"; a caller who wants more asks, and
/// the kernel clamps the ask.
const MEMORY_HISTORY_DEFAULT_LIMIT: usize = 5;

/// Statuses an agent may name on a fact write.
///
/// Checked here as well as kernel-side so the rejection can say what the legal
/// values are. The kernel's parse is the enforcement point; this one exists so
/// a typo comes back as a menu rather than as a store error.
const AGENT_FACT_STATUSES: &[&str] = &["open", "settled"];

/// Pull the shared `(scope, scope_ref, key)` slot address out of a tool call.
fn fact_address(input: &serde_json::Value) -> Result<(&str, Option<&str>, &str), String> {
    let scope = input["scope"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or(
            "Missing 'scope' parameter — whose truth this is. Use one of: agent, project, user",
        )?;
    let key = input["key"]
        .as_str()
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .ok_or("Missing 'key' parameter — the slot name, e.g. \"repo.trunk_model\"")?;
    let scope_ref = input["scope_ref"]
        .as_str()
        .map(str::trim)
        .filter(|r| !r.is_empty());
    Ok((scope, scope_ref, key))
}

async fn tool_memory_fact(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let (scope, scope_ref, key) = fact_address(input)?;
    let claim = input["claim"]
        .as_str()
        .map(str::trim)
        .filter(|c| !c.is_empty());

    // No claim means read. Deliberately not an error — reading a slot before
    // writing it is the behaviour §2.3.3 wants, since a caller who has seen
    // the existing key space selects from it instead of minting a
    // near-duplicate. But a caller who MEANT to write and left `claim` off
    // would otherwise get a success message for a call that changed nothing,
    // so the read says so in as many words.
    let Some(claim) = claim else {
        let payload = kh.memory_fact_get(caller_agent_id, scope, scope_ref, key)?;
        return Ok(render_fact_read(&payload));
    };

    if let Some(status) = input["status"].as_str().map(str::trim) {
        if !status.is_empty() && !AGENT_FACT_STATUSES.contains(&status) {
            return Err(format!(
                "Unknown status '{status}'. Use one of: {}",
                AGENT_FACT_STATUSES.join(", ")
            ));
        }
    }

    let request = crate::kernel_handle::FactWriteRequest {
        scope: scope.to_string(),
        scope_ref: scope_ref.map(str::to_string),
        claim_key: key.to_string(),
        claim: claim.to_string(),
        status: input["status"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        confidence: input["confidence"].as_f64(),
        persistence_class: input["persistence_class"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    };

    let payload = kh.memory_fact_write(caller_agent_id, request).await?;
    Ok(render_fact_write(&payload))
}

fn tool_memory_history(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let (scope, scope_ref, key) = fact_address(input)?;
    let limit = input["limit"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(MEMORY_HISTORY_DEFAULT_LIMIT);

    let payload = kh.memory_fact_history(caller_agent_id, scope, scope_ref, key, limit)?;
    Ok(render_fact_history(&payload))
}

/// `scope/scope_ref key` — the slot address as one readable token.
fn fact_slot_label(payload: &serde_json::Value) -> String {
    format!(
        "{}/{} {}",
        payload["scope"].as_str().unwrap_or("?"),
        payload["scope_ref"].as_str().unwrap_or("?"),
        payload["claim_key"].as_str().unwrap_or("?"),
    )
}

fn render_fact_read(payload: &serde_json::Value) -> String {
    let slot = fact_slot_label(payload);
    let Some(fact) = payload.get("fact").filter(|f| !f.is_null()) else {
        // ANAI-266: an empty child slot with a live ancestor is not an empty
        // subject. Saying "empty" here while a rehydration pack hands over the
        // parent's claim is two answers to one question, and it is the answer
        // that invites a duplicate write into the child scope.
        if let Some(inherited) = payload.get("inherited").filter(|f| !f.is_null()) {
            let from = inherited["from_scope_ref"].as_str().unwrap_or("?");
            return format!(
                "{slot}\n  (no claim at this exact slot — inherited from \
                 project/{from})\n{}\nRead only: no 'claim' was given, so nothing was \
                 written. Writing here would CREATE a new slot at this address rather \
                 than replace the one you just read, and it would shadow project/{from} \
                 for every reader of this sub-project. Correct the claim at \
                 project/{from} instead, unless it is genuinely different at this level.",
                render_fact_body(inherited),
            );
        }
        return format!(
            "{slot}\n  (empty — no claim occupies this slot)\n\nRead only: no 'claim' was \
             given, so nothing was written. Pass 'claim' to fill the slot."
        );
    };

    format!(
        "{slot}\n{}\nRead only: no 'claim' was given, so nothing was written. Pass \
         'claim' to replace this.",
        render_fact_body(fact),
    )
}

/// The claim and its provenance, indented, one line apiece.
///
/// Shared by the exact read and the inherited read (ANAI-266) so an inherited
/// claim cannot quietly lose its staleness marker — a fact the reader did not
/// address, shown with more confidence than the one they did, is the failure
/// mode worth engineering against here.
fn render_fact_body(fact: &serde_json::Value) -> String {
    let mut out = format!(
        "  {}\n",
        fact["claim"].as_str().unwrap_or("(unreadable claim)")
    );
    if let Some(status) = fact["status"].as_str() {
        out.push_str(&format!("  status: {status}"));
        if let Some(confidence) = fact["confidence"].as_f64() {
            out.push_str(&format!("   confidence: {confidence}"));
        }
        out.push('\n');
    }
    if let Some(author) = fact["authored_by"].as_str() {
        out.push_str(&format!("  asserted by: {author}\n"));
    }
    if let Some(created) = fact["created_at"].as_str() {
        out.push_str(&format!("  believed since: {created}\n"));
    }
    if let Some(affirmed) = fact["last_affirmed_at"].as_str() {
        out.push_str(&format!("  last re-affirmed: {affirmed}\n"));
    }
    // ANAI-259. The class is stated on every read, not just a stale one, so a
    // reader can see a claim is *deliberately* permanent rather than merely
    // unmarked. The warning is a separate line because it is the part that
    // should change what the reader does next.
    if let Some(class) = fact["persistence_class"].as_str() {
        out.push_str(&format!("  persistence: {class}\n"));
    }
    if fact["should_verify"].as_bool().unwrap_or(false) {
        let age = match fact["age_days"].as_f64() {
            Some(d) => format!("{} days", d as i64),
            None => "an unknown length of time".to_string(),
        };
        out.push_str(&format!(
            "  VERIFY: unchecked for {age}, longer than a '{}' claim should go. This is \
             still what we believe — treat it as a question to confirm, not a fact to \
             assert, and re-write the slot once you have checked it.\n",
            fact["persistence_class"].as_str().unwrap_or("active")
        ));
    }
    out
}

/// ANAI-277: the addresses that already exist where a new slot was just
/// minted.
///
/// Appended to the *created* message only, because that is the only outcome
/// where the agent still has a choice to make. A supersession means the right
/// address was already selected, and repeating the neighbourhood there would
/// train the agent to skim past the one message that carries a decision.
///
/// Advisory throughout. The write has already committed and nothing here can
/// or should undo it: a refusal at this point loses a claim the agent took the
/// trouble to record, which is a far worse failure than a duplicate key.
fn render_slot_neighbourhood(payload: &serde_json::Value) -> String {
    let addresses: Vec<&str> = payload["existing_slots"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let shadowed = payload["shadows_ancestor_slot"].as_str();
    if addresses.is_empty() && shadowed.is_none() {
        return String::new();
    }

    let mut out = String::new();

    // ANAI-281 leads, because it is the only part of this hint that is not a
    // guess. The ranked list below asks the agent to recognise something; this
    // states a fact about the corpus it can act on without judging anything.
    if let Some(ancestor) = shadowed {
        out.push_str(&format!(
            "\nSHADOWS AN EXISTING SLOT: `{ancestor}` already holds a live claim under this \
             exact key, at a project ABOVE the one you just wrote to. Nothing was lost, but \
             one claim now lives in two slots, and a lineage read returns yours and hides \
             that one. If the claim belongs to the parent project, rewrite `{ancestor}` \
             instead and let this slot go; if it is genuinely narrower than the parent's, \
             this is correct and you can ignore the warning.\n"
        ));
    }

    if !addresses.is_empty() {
        out.push_str(
            "\nAddresses that already exist here. If one of them is this same claim under \
             another name, write THAT key from now on — rewriting a slot supersedes it and \
             keeps the history, while a second key splits one claim in two and surfaces both:\n",
        );
        for address in &addresses {
            out.push_str(&format!("  - {address}\n"));
        }

        let total = payload["existing_slots_total"].as_u64().unwrap_or(0) as usize;
        if total > addresses.len() {
            // Say what was dropped rather than letting a capped list read as
            // the whole address space — the omitted tail is where an old
            // settled slot is most likely to be hiding.
            out.push_str(&format!(
                "  ({} more not shown; memory_status lists your open slots)\n",
                total - addresses.len()
            ));
        }
    }

    if let Some(near) = payload["likely_duplicate_of"].as_str() {
        out.push_str(&format!(
            "NEAR-DUPLICATE: `{near}` is a close spelling of the key you just minted. This is \
             a lexical guess, not a judgement about meaning — if the two are the same claim, \
             make `{near}` the one you keep writing.\n"
        ));
    }

    out
}

fn render_fact_write(payload: &serde_json::Value) -> String {
    let slot = fact_slot_label(payload);
    match payload["outcome"].as_str().unwrap_or("") {
        "created" => format!(
            "Created {slot}. This slot was empty; it now holds your claim.{}",
            render_slot_neighbourhood(payload)
        ),
        // Named distinctly from "created" on purpose: an agent that cannot
        // tell "I already believed this" from "I have changed my mind" will
        // report a no-op as news.
        "affirmed" => format!(
            "Affirmed {slot}. The claim was already exactly this, so nothing changed except \
             the last-affirmed timestamp — no history entry was written."
        ),
        "superseded" => match payload["previous_claim"].as_str() {
            Some(previous) => format!(
                "Superseded {slot}. The previous claim was: {previous}\nIt is now in \
                 fact_history and will never surface in recall; memory_history is the only \
                 way back to it."
            ),
            None => format!(
                "Superseded {slot}. The previous claim moved to fact_history and will never \
                 surface in recall."
            ),
        },
        other => format!("Wrote {slot} (outcome: {other})."),
    }
}

fn render_fact_history(payload: &serde_json::Value) -> String {
    let slot = fact_slot_label(payload);
    let entries = payload["entries"].as_array().cloned().unwrap_or_default();
    if entries.is_empty() {
        return format!(
            "{slot}\n  No superseded versions. Either the slot has never been overwritten, \
             or it has never been written at all — memory_fact without a 'claim' tells you \
             which."
        );
    }

    let mut out = format!("{slot} — {} superseded version(s):\n", entries.len());
    for entry in &entries {
        out.push_str(&format!(
            "  - {}\n",
            entry["claim"].as_str().unwrap_or("(unreadable claim)")
        ));
        let author = entry["authored_by"].as_str().unwrap_or("unknown");
        let held = entry["created_at"].as_str().unwrap_or("?");
        let ended = entry["superseded_at"].as_str().unwrap_or("?");
        out.push_str(&format!(
            "      asserted by {author}, believed {held} until {ended}\n"
        ));
    }
    out
}

/// Close reasons an AGENT may name (ANAI-194 / ANAI-283, ADR 0002 §2.2, §2.6).
///
/// `topic-switch` was held back by §2.6 — "deferred until the judgment it
/// depends on exists". ANAI-283 is that judgment: the close doctrine now fires
/// on a shift named by the incoming message rather than on the agent's private
/// sense of completion, which is a cue an agent can actually report. It is
/// opened here in the same change that ships the wording, because otherwise
/// every close still writes `explicit` and the census cannot tell a boundary
/// that came from the new cue from one that came from the old one — we would
/// ship the trigger and lose the only measurement of whether it fired right.
///
/// No migration: `CloseReason::TopicSwitch` already round-trips and
/// `episodes.close_reason` carries no CHECK.
///
/// Still narrower than the enum. `timer` stays the system's to write — an agent
/// claiming one would date the boundary wrong and make the idle gap
/// unfalsifiable — and `abandoned` is by definition what nobody was around to
/// say, so an agent emitting it is a contradiction rather than a report.
const AGENT_CLOSE_REASONS: &[&str] = &["topic-switch", "explicit"];

/// ANAI-252: an agent-authored wrap-up is *material*, never the summary.
///
/// Writing it into `episodes.summary` used to suppress consolidation outright.
/// The summariser's selector is `closed_at IS NOT NULL AND summary IS NULL`
/// (`EpisodeStore::awaiting_summary`), so a filled column made the episode
/// ineligible: the close succeeded, the agent's text sat in a column nothing
/// queries at recall time, and the corpus got **no embedded row at all**. The
/// agent that wrote the most careful wrap-up got the least recallable memory,
/// and it looked like the feature working.
///
/// So the text lands as a note on the still-open episode instead. That keeps
/// it (embedded, recallable, attached to the right episode), leaves `summary`
/// null so the episode stays a consolidation candidate, and — because
/// `episode_material` reads `memories` — feeds it to the summariser as input.
/// Invariant: every closed episode yields exactly one embedded summary row.
async fn tool_memory_episode_close(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let title = input["title"]
        .as_str()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or("Missing 'title' parameter — an episode closed without a label is a boundary nobody can find again")?;
    let wrapup = input["summary"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let reason = input["reason"].as_str().unwrap_or("explicit");
    if !AGENT_CLOSE_REASONS.contains(&reason) {
        return Err(format!(
            "Close reason '{reason}' is not available to agents. Use one of: {}",
            AGENT_CLOSE_REASONS.join(", ")
        ));
    }
    let reset_context = input["reset_context"].as_bool().unwrap_or(false);

    // ANAI-248: the self-amputation guard, in code rather than prose.
    //
    // Prompt language telling an agent "never close mid-task" is a
    // probabilistic mitigation of a failure that is silent when it happens.
    // A pending approval is the one machine-readable "I am blocked on a
    // human" we have: the agent asked, the operator has not answered, and
    // clearing the window now means the answer lands in an agent that no
    // longer remembers the question.
    //
    // Scoped to `reset_context` deliberately. A close *without* a reset is
    // bookkeeping — it labels a boundary and costs nothing — and refusing it
    // would be the tool overriding the agent's judgment on a harmless call.
    // The reset is the destructive half, so the reset is what is guarded.
    if reset_context && kh.has_pending_operator_question(caller_agent_id) {
        return Err(
            "You have an approval request outstanding to the operator, so this is mid-task: \
             clearing your window now would drop the context the answer belongs to. Nothing \
             was closed. Wait for the answer, finish the work, then close."
                .to_string(),
        );
    }

    // ANAI-247. Validated here, at the edge, rather than swallowed downstream:
    // a mistyped slug would otherwise prime the next episode for a project
    // that does not exist and produce a silently empty pack — the agent would
    // conclude its memory was empty when it was merely misaddressed.
    //
    // Refused *before* the close so the agent can simply retry the whole call.
    let prime_for = match input["prime_for"].as_str().map(str::trim) {
        Some("") | None => None,
        Some(slug) => {
            openfang_types::agent::validate_project_slug(slug).map_err(|e| {
                format!("prime_for is not a usable project slug: {e}. Nothing was closed.")
            })?;
            if !reset_context {
                return Err(
                    "prime_for only means something with reset_context: true — the briefing is \
                     what the fresh window opens with. Nothing was closed."
                        .to_string(),
                );
            }
            // ANAI-264. Membership, checked at the same edge as shape and for
            // the same reason: refused before the close, so the agent can
            // retry the whole call rather than discover a half-done boundary.
            //
            // This became checkable only with dotted slugs. Against flat
            // slugs the rule would have rejected the legitimate case — an
            // agent declaring `openfang` priming for the corner of it it is
            // actually working on — since nothing could tell "more specific
            // than my declaration" from "a different project". Segment
            // coverage tells them apart, which is why the guard I withdrew in
            // design lands here now.
            if let Some(refusal) = kh.project_membership_error(caller_agent_id, slug) {
                return Err(format!("prime_for: {refusal} Nothing was closed."));
            }
            Some(slug.to_string())
        }
    };

    // Before the close, so it attaches to the episode being closed rather than
    // to the one that opens next. Failure aborts the close rather than
    // dropping the wrap-up silently — closing is idempotent, so a retry costs
    // nothing, whereas a swallowed note is exactly the silent amnesia this
    // path exists to prevent.
    if let Some(text) = wrapup {
        kh.memory_note(caller_agent_id, text, &["episode-wrapup".to_string()])
            .await
            .map_err(|e| {
                format!(
                    "Could not record the wrap-up note, so nothing was closed: {e}. \
                     Retry — closing is idempotent."
                )
            })?;
    }

    // `None`, always: see this function's doc comment. The summary column is
    // the daemon consolidator's to fill.
    let closed = kh.memory_episode_close(caller_agent_id, reason, Some(title), None)?;

    // ANAI-246. Requested AFTER the close, so a close that fails takes the
    // reset down with it rather than clearing the window of an episode still
    // open. Honoured whether or not an episode was open: "clear my window" is
    // a coherent ask on its own, and the reset itself is idempotent.
    //
    // A reset failure does NOT fail the tool. The close has already committed
    // and the wrap-up note is already written; returning `Err` here would tell
    // the agent to retry a close that succeeded, duplicating the note. Say so
    // in the text instead — the agent can see it and so can the log.
    let reset_note = if reset_context {
        match kh.request_context_reset(caller_agent_id, prime_for.as_deref()) {
            Ok(()) => match prime_for.as_deref() {
                Some(slug) => {
                    // ANAI-264: say what the briefing actually resolved.
                    //
                    // The pack is assembled a turn later, and by then the only
                    // entity that could recognise a mistyped slug — the agent
                    // that typed it — has been reset and is not consulted. A
                    // slug naming no project renders a pack that looks healthy
                    // with its "what is true" half silently empty; that is the
                    // 2026-08-26 bug, and it went unnoticed because nothing
                    // ever reported the number.
                    //
                    // Advisory, never blocking: an unfamiliar slug may simply
                    // be a project no one has written a fact about yet. We
                    // report, the agent judges.
                    let resolved = match kh.rehydration_preview(caller_agent_id, slug) {
                        Some((episodes, 0)) => format!(
                            " That briefing resolves {episodes} closed episode(s) and no facts \
                             about {slug} — which usually means the slug is misaddressed rather \
                             than the project unknown. Check the spelling before you rely on it."
                        ),
                        Some((episodes, facts)) => format!(
                            " That briefing resolves {episodes} closed episode(s) and {facts} \
                             live fact(s) about {slug}."
                        ),
                        // "Cannot say" is not "nothing found": stay silent
                        // rather than report a zero we did not measure.
                        None => String::new(),
                    };
                    format!(
                        " Your conversation window will be cleared when this turn ends; the \
                         running summary of earlier work is kept, and the next episode opens \
                         with a briefing on {slug}.{resolved}"
                    )
                }
                None => " Your conversation window will be cleared when this turn ends; the running summary of earlier work is kept.".to_string(),
            },
            Err(_) => " Note: the context reset could not be scheduled, so your conversation window is unchanged. The close itself succeeded — do not repeat it.".to_string(),
        }
    } else {
        String::new()
    };

    match closed {
        Some(id) => Ok(format!(
            "Closed episode {id} as '{title}'. The next captured turn starts a new one.{reset_note}"
        )),
        // Not an error: the agent cannot see the episode table, so a double
        // wrap-up is a reasonable thing to do and a failure here would read as
        // "your memory is broken" when nothing is.
        None => Ok(format!(
            "No episode was open, so nothing was closed.{reset_note}"
        )),
    }
}

fn tool_memory_status(
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let status = kh.memory_status(caller_agent_id)?;

    let mut out = String::new();
    match status.get("episode").and_then(|e| e.as_object()) {
        Some(ep) => {
            let id = ep.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let turns = ep.get("turn_count").and_then(|v| v.as_u64()).unwrap_or(0);
            let opened = ep.get("opened_at").and_then(|v| v.as_str()).unwrap_or("?");
            out.push_str(&format!(
                "Open episode {id}\n  opened: {opened}\n  turns captured: {turns}\n"
            ));
            let idle = status.get("idle_minutes").and_then(|v| v.as_i64());
            if let Some(m) = idle {
                out.push_str(&format!("  idle: {m} min\n"));
            }
            match status
                .get("minutes_until_timer_close")
                .and_then(|v| v.as_i64())
            {
                Some(m) => out.push_str(&format!("  closes on timer in: {m} min\n")),
                None => out.push_str("  closes on timer in: never (timer disabled)\n"),
            }
        }
        // An agent with no open episode is the normal state before its first
        // captured turn, not a fault. Say so plainly rather than printing an
        // empty panel the agent has to interpret.
        None => out.push_str("No episode is open. The next captured turn opens one.\n"),
    }

    if let Some(recent) = status.get("recent_episodes").and_then(|v| v.as_array()) {
        if !recent.is_empty() {
            out.push_str("Recently closed:\n");
            for ep in recent {
                let title = ep
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(untitled)");
                let reason = ep
                    .get("close_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let turns = ep.get("turn_count").and_then(|v| v.as_u64()).unwrap_or(0);
                out.push_str(&format!("  - {title} ({turns} turns, closed: {reason})\n"));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Collaboration tools
// ---------------------------------------------------------------------------

fn tool_agent_find(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let query = input["query"].as_str().ok_or("Missing 'query' parameter")?;
    let agents = kh.find_agents(query);
    if agents.is_empty() {
        return Ok(format!("No agents found matching '{query}'."));
    }
    let result: Vec<serde_json::Value> = agents
        .iter()
        .map(|a| {
            serde_json::json!({
                "id": a.id,
                "name": a.name,
                "state": a.state,
                "description": a.description,
                "tags": a.tags,
                "tools": a.tools,
                "model": format!("{}:{}", a.model_provider, a.model_name),
            })
        })
        .collect();
    serde_json::to_string_pretty(&result).map_err(|e| format!("Serialize error: {e}"))
}

async fn tool_task_post(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let title = input["title"].as_str().ok_or("Missing 'title' parameter")?;
    let description = input["description"]
        .as_str()
        .ok_or("Missing 'description' parameter")?;
    let assigned_to = input["assigned_to"].as_str();
    let payload = match input["payload"].as_str() {
        Some(s) if !s.is_empty() => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(s)
                .map_err(|e| format!("Invalid base64 'payload': {e}"))?
        }
        _ => Vec::new(),
    };
    let task_id = kh
        .task_post(title, description, assigned_to, caller_agent_id, &payload)
        .await?;
    Ok(format!("Task created with ID: {task_id}"))
}

async fn tool_task_claim(
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id = caller_agent_id.unwrap_or("");
    match kh.task_claim(agent_id).await? {
        Some(task) => {
            serde_json::to_string_pretty(&task).map_err(|e| format!("Serialize error: {e}"))
        }
        None => Ok("No tasks available.".to_string()),
    }
}

async fn tool_task_complete(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let task_id = input["task_id"]
        .as_str()
        .ok_or("Missing 'task_id' parameter")?;
    let result = input["result"]
        .as_str()
        .ok_or("Missing 'result' parameter")?;
    kh.task_complete(task_id, result).await?;
    Ok(format!("Task {task_id} marked as completed."))
}

async fn tool_task_list(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let status = input["status"].as_str();
    let mut tasks = kh.task_list(status).await?;
    // ANAI-147: narrow to one id. The substrate query is status-keyed, so the
    // filter lands here rather than widening the KernelHandle contract for a
    // diagnostic read. Without this there is no way to ask "did the wake I
    // enqueued actually run?" — the gap that made a stalled queue look exactly
    // like a dropped message.
    if let Some(task_id) = input["task_id"].as_str() {
        tasks.retain(|t| t["id"].as_str() == Some(task_id));
        if tasks.is_empty() {
            return Ok(format!("No task found with id {task_id}."));
        }
    }
    if tasks.is_empty() {
        return Ok("No tasks found.".to_string());
    }
    serde_json::to_string_pretty(&tasks).map_err(|e| format!("Serialize error: {e}"))
}

async fn tool_event_publish(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let event_type = input["event_type"]
        .as_str()
        .ok_or("Missing 'event_type' parameter")?;
    let payload = input
        .get("payload")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    kh.publish_event(event_type, payload).await?;
    Ok(format!("Event '{event_type}' published successfully."))
}

// ---------------------------------------------------------------------------
// Knowledge graph tools
// ---------------------------------------------------------------------------

fn parse_entity_type(s: &str) -> openfang_types::memory::EntityType {
    use openfang_types::memory::EntityType;
    match s.to_lowercase().as_str() {
        "person" => EntityType::Person,
        "organization" | "org" => EntityType::Organization,
        "project" => EntityType::Project,
        "concept" => EntityType::Concept,
        "event" => EntityType::Event,
        "location" => EntityType::Location,
        "document" | "doc" => EntityType::Document,
        "tool" => EntityType::Tool,
        other => EntityType::Custom(other.to_string()),
    }
}

fn parse_relation_type(s: &str) -> openfang_types::memory::RelationType {
    use openfang_types::memory::RelationType;
    match s.to_lowercase().as_str() {
        "works_at" | "worksat" => RelationType::WorksAt,
        "knows_about" | "knowsabout" | "knows" => RelationType::KnowsAbout,
        "related_to" | "relatedto" | "related" => RelationType::RelatedTo,
        "depends_on" | "dependson" | "depends" => RelationType::DependsOn,
        "owned_by" | "ownedby" => RelationType::OwnedBy,
        "created_by" | "createdby" => RelationType::CreatedBy,
        "located_in" | "locatedin" => RelationType::LocatedIn,
        "part_of" | "partof" => RelationType::PartOf,
        "uses" => RelationType::Uses,
        "produces" => RelationType::Produces,
        other => RelationType::Custom(other.to_string()),
    }
}

async fn tool_knowledge_add_entity(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let name = input["name"].as_str().ok_or("Missing 'name' parameter")?;
    let entity_type_str = input["entity_type"]
        .as_str()
        .ok_or("Missing 'entity_type' parameter")?;
    let properties = input
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let entity = openfang_types::memory::Entity {
        id: String::new(), // kernel/store assigns a real ID
        entity_type: parse_entity_type(entity_type_str),
        name: name.to_string(),
        properties,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    let id = kh.knowledge_add_entity(entity).await?;
    Ok(format!("Entity '{name}' added with ID: {id}"))
}

async fn tool_knowledge_add_relation(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let source = input["source"]
        .as_str()
        .ok_or("Missing 'source' parameter")?;
    let relation_str = input["relation"]
        .as_str()
        .ok_or("Missing 'relation' parameter")?;
    let target = input["target"]
        .as_str()
        .ok_or("Missing 'target' parameter")?;
    let confidence = input["confidence"].as_f64().unwrap_or(1.0) as f32;
    let properties = input
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let relation = openfang_types::memory::Relation {
        source: source.to_string(),
        relation: parse_relation_type(relation_str),
        target: target.to_string(),
        properties,
        confidence,
        created_at: chrono::Utc::now(),
    };

    let id = kh.knowledge_add_relation(relation).await?;
    Ok(format!(
        "Relation '{source}' --[{relation_str}]--> '{target}' added with ID: {id}"
    ))
}

async fn tool_knowledge_query(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let source = input["source"].as_str().map(|s| s.to_string());
    let target = input["target"].as_str().map(|s| s.to_string());
    let relation = input["relation"].as_str().map(parse_relation_type);
    let max_depth = input["max_depth"].as_u64().unwrap_or(1) as u32;

    let pattern = openfang_types::memory::GraphPattern {
        source,
        relation,
        target,
        max_depth,
    };

    let matches = kh.knowledge_query(pattern).await?;
    if matches.is_empty() {
        return Ok("No matching knowledge graph entries found.".to_string());
    }

    let mut output = format!("Found {} match(es):\n", matches.len());
    for m in &matches {
        output.push_str(&format!(
            "\n  {} ({:?}) --[{:?} ({:.0}%)]--> {} ({:?})",
            m.source.name,
            m.source.entity_type,
            m.relation.relation,
            m.relation.confidence * 100.0,
            m.target.name,
            m.target.entity_type,
        ));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Scheduling tools
// ---------------------------------------------------------------------------

/// Parse a natural language schedule into a cron expression.
fn parse_schedule_to_cron(input: &str) -> Result<String, String> {
    let input = input.trim().to_lowercase();

    // If it already looks like a cron expression (5 space-separated fields), pass through
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.len() == 5
        && parts
            .iter()
            .all(|p| p.chars().all(|c| c.is_ascii_digit() || "*/,-".contains(c)))
    {
        return Ok(input);
    }

    // Natural language patterns
    if let Some(rest) = input.strip_prefix("every ") {
        if rest == "minute" || rest == "1 minute" {
            return Ok("* * * * *".to_string());
        }
        if let Some(mins) = rest.strip_suffix(" minutes") {
            let n: u32 = mins
                .trim()
                .parse()
                .map_err(|_| format!("Invalid number in '{input}'"))?;
            if n == 0 || n > 59 {
                return Err(format!("Minutes must be 1-59, got {n}"));
            }
            return Ok(format!("*/{n} * * * *"));
        }
        if rest == "hour" || rest == "1 hour" {
            return Ok("0 * * * *".to_string());
        }
        if let Some(hrs) = rest.strip_suffix(" hours") {
            let n: u32 = hrs
                .trim()
                .parse()
                .map_err(|_| format!("Invalid number in '{input}'"))?;
            if n == 0 || n > 23 {
                return Err(format!("Hours must be 1-23, got {n}"));
            }
            return Ok(format!("0 */{n} * * *"));
        }
        if rest == "day" || rest == "1 day" {
            return Ok("0 0 * * *".to_string());
        }
        if rest == "week" || rest == "1 week" {
            return Ok("0 0 * * 0".to_string());
        }
    }

    // "daily at Xam/pm"
    if let Some(time_str) = input.strip_prefix("daily at ") {
        let hour = parse_time_to_hour(time_str)?;
        return Ok(format!("0 {hour} * * *"));
    }

    // "weekdays at Xam/pm"
    if let Some(time_str) = input.strip_prefix("weekdays at ") {
        let hour = parse_time_to_hour(time_str)?;
        return Ok(format!("0 {hour} * * 1-5"));
    }

    // "weekends at Xam/pm"
    if let Some(time_str) = input.strip_prefix("weekends at ") {
        let hour = parse_time_to_hour(time_str)?;
        return Ok(format!("0 {hour} * * 0,6"));
    }

    // "hourly" / "daily" / "weekly" / "monthly"
    match input.as_str() {
        "hourly" => return Ok("0 * * * *".to_string()),
        "daily" => return Ok("0 0 * * *".to_string()),
        "weekly" => return Ok("0 0 * * 0".to_string()),
        "monthly" => return Ok("0 0 1 * *".to_string()),
        _ => {}
    }

    Err(format!(
        "Could not parse schedule '{input}'. Try: 'every 5 minutes', 'daily at 9am', 'weekdays at 6pm', or a cron expression like '0 */5 * * *'"
    ))
}

/// Parse a time string like "9am", "6pm", "14:00", "9:30am" into an hour (0-23).
fn parse_time_to_hour(s: &str) -> Result<u32, String> {
    let s = s.trim().to_lowercase();

    // Handle "9am", "6pm", "12pm", "12am"
    if let Some(h) = s.strip_suffix("am") {
        let hour: u32 = h.trim().parse().map_err(|_| format!("Invalid time: {s}"))?;
        return match hour {
            12 => Ok(0),
            1..=11 => Ok(hour),
            _ => Err(format!("Invalid hour: {hour}")),
        };
    }
    if let Some(h) = s.strip_suffix("pm") {
        let hour: u32 = h.trim().parse().map_err(|_| format!("Invalid time: {s}"))?;
        return match hour {
            12 => Ok(12),
            1..=11 => Ok(hour + 12),
            _ => Err(format!("Invalid hour: {hour}")),
        };
    }

    // Handle "14:00" or "9:30"
    if let Some((h, _m)) = s.split_once(':') {
        let hour: u32 = h.trim().parse().map_err(|_| format!("Invalid time: {s}"))?;
        if hour > 23 {
            return Err(format!("Hour must be 0-23, got {hour}"));
        }
        return Ok(hour);
    }

    // Plain number
    let hour: u32 = s.parse().map_err(|_| format!("Invalid time: {s}"))?;
    if hour > 23 {
        return Err(format!("Hour must be 0-23, got {hour}"));
    }
    Ok(hour)
}

/// Sanitize a description into a valid `CronJob.name` (alphanumeric +
/// space/hyphen/underscore, 1..=128 chars).
fn sanitize_schedule_name(description: &str) -> String {
    let filtered: String = description
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = filtered.trim();
    if trimmed.is_empty() {
        return "scheduled-task".to_string();
    }
    trimmed.chars().take(128).collect()
}

/// Resolve the `agent` field of `schedule_create` into an agent UUID string
/// suitable for `KernelHandle::cron_create`.
///
/// - Empty / "self" → caller's agent ID.
/// - Valid UUID → passed through.
/// - Non-empty name → looked up via `find_agents`; an exact name match wins,
///   a single fuzzy match is accepted, ambiguity is an error.
fn resolve_schedule_target(
    kh: &Arc<dyn KernelHandle>,
    agent: &str,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let a = agent.trim();
    if a.is_empty() || a.eq_ignore_ascii_case("self") {
        return caller_agent_id.map(|s| s.to_string()).ok_or_else(|| {
            "No caller agent available; specify 'agent' to target an agent by name or UUID"
                .to_string()
        });
    }
    if uuid::Uuid::parse_str(a).is_ok() {
        return Ok(a.to_string());
    }
    let matches = kh.find_agents(a);
    if let Some(m) = matches.iter().find(|m| m.name == a) {
        return Ok(m.id.clone());
    }
    match matches.len() {
        0 => Err(format!("Agent '{a}' not found")),
        1 => Ok(matches[0].id.clone()),
        n => Err(format!(
            "Agent name '{a}' is ambiguous ({n} matches). Pass the agent UUID."
        )),
    }
}

async fn tool_schedule_create(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let description = input["description"]
        .as_str()
        .ok_or("Missing 'description' parameter")?;
    let schedule_str = input["schedule"]
        .as_str()
        .ok_or("Missing 'schedule' parameter")?;
    let agent_input = input["agent"].as_str().unwrap_or("");

    let cron_expr = parse_schedule_to_cron(schedule_str)?;
    let target_agent_id = resolve_schedule_target(kh, agent_input, caller_agent_id)?;
    let name = sanitize_schedule_name(description);

    let job_json = serde_json::json!({
        "name": name,
        "schedule": { "kind": "cron", "expr": cron_expr, "tz": null },
        "action": {
            "kind": "agent_turn",
            "message": description,
            "model_override": null,
            "timeout_secs": null,
        },
        "delivery": { "kind": "none" },
        "one_shot": false,
    });

    let resp = kh.cron_create(&target_agent_id, job_json).await?;
    // Kernel returns JSON `{ "job_id": "...", "status": "created" }`.
    let job_id = serde_json::from_str::<serde_json::Value>(&resp)
        .ok()
        .and_then(|v| v["job_id"].as_str().map(str::to_string))
        .unwrap_or_else(|| resp.clone());

    let agent_display = if agent_input.trim().is_empty() {
        "(self)".to_string()
    } else {
        agent_input.to_string()
    };
    Ok(format!(
        "Schedule created:\n  ID: {job_id}\n  Description: {description}\n  Cron: {cron_expr}\n  Original: {schedule_str}\n  Agent: {agent_display}"
    ))
}

async fn tool_schedule_list(
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id =
        caller_agent_id.ok_or("Agent ID required for schedule_list (no caller context)")?;

    let jobs = kh.cron_list(agent_id).await?;
    if jobs.is_empty() {
        return Ok("No scheduled tasks.".to_string());
    }

    let mut output = format!("Scheduled tasks ({}):\n\n", jobs.len());
    for job in &jobs {
        let enabled = job["enabled"].as_bool().unwrap_or(true);
        let status = if enabled { "active" } else { "paused" };
        let id = job["id"].as_str().unwrap_or("?");
        let schedule_display = match job["schedule"]["kind"].as_str() {
            Some("cron") => job["schedule"]["expr"].as_str().unwrap_or("?").to_string(),
            Some("every") => format!(
                "every {}s",
                job["schedule"]["every_secs"].as_u64().unwrap_or(0)
            ),
            Some("at") => job["schedule"]["at"].as_str().unwrap_or("?").to_string(),
            _ => "?".to_string(),
        };
        let description = job["action"]["message"]
            .as_str()
            .or_else(|| job["action"]["text"].as_str())
            .unwrap_or_else(|| job["name"].as_str().unwrap_or("?"));
        let created = job["created_at"].as_str().unwrap_or("?");
        let agent = job["agent_id"].as_str().unwrap_or("(self)");
        output.push_str(&format!(
            "  [{status}] {id} — {description}\n    Cron: {schedule_display} | Agent: {agent}\n    Created: {created}\n\n"
        ));
    }

    Ok(output)
}

async fn tool_schedule_delete(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let id = input["id"].as_str().ok_or("Missing 'id' parameter")?;
    kh.cron_cancel(id)
        .await
        .map_err(|e| format!("Schedule '{id}' not found: {e}"))?;
    Ok(format!("Schedule '{id}' deleted."))
}

// ---------------------------------------------------------------------------
// Cron scheduling tools (delegated to kernel via KernelHandle trait)
// ---------------------------------------------------------------------------

async fn tool_cron_create(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id = caller_agent_id.ok_or("Agent ID required for cron_create")?;
    kh.cron_create(agent_id, input.clone()).await
}

async fn tool_cron_list(
    kernel: Option<&Arc<dyn KernelHandle>>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let agent_id = caller_agent_id.ok_or("Agent ID required for cron_list")?;
    let jobs = kh.cron_list(agent_id).await?;
    serde_json::to_string_pretty(&jobs).map_err(|e| format!("Failed to serialize cron jobs: {e}"))
}

async fn tool_cron_cancel(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let job_id = input["job_id"]
        .as_str()
        .ok_or("Missing 'job_id' parameter")?;
    kh.cron_cancel(job_id).await?;
    Ok(format!("Cron job '{job_id}' cancelled."))
}

// ---------------------------------------------------------------------------
// Channel send tool (proactive outbound messaging via configured adapters)
// ---------------------------------------------------------------------------

async fn tool_channel_send(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
    workspace_root: Option<&Path>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;

    let channel = input["channel"]
        .as_str()
        .ok_or("Missing 'channel' parameter")?
        .trim()
        .to_lowercase();
    let recipient_input = input["recipient"]
        .as_str()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // If recipient is empty, resolve from channel's default_chat_id config.
    let recipient = if recipient_input.is_empty() {
        let default_id = kh.get_channel_default_recipient(&channel).await;
        match default_id {
            Some(id) => id,
            None => {
                return Err(format!(
                "Missing 'recipient' parameter. Set default_chat_id in [channels.{channel}] config \
                 or pass recipient explicitly."
            ))
            }
        }
    } else {
        recipient_input
    };
    let recipient = recipient.as_str();

    let thread_id = input["thread_id"].as_str().filter(|s| !s.is_empty());

    // Check for media content (image_url, file_url, or file_path)
    let image_url = input["image_url"].as_str().filter(|s| !s.is_empty());
    let file_url = input["file_url"].as_str().filter(|s| !s.is_empty());
    let file_path = input["file_path"].as_str().filter(|s| !s.is_empty());

    if let Some(url) = image_url {
        let caption = input["message"].as_str().filter(|s| !s.is_empty());
        return kh
            .send_channel_media(&channel, recipient, "image", url, caption, None, thread_id)
            .await;
    }

    if let Some(url) = file_url {
        let caption = input["message"].as_str().filter(|s| !s.is_empty());
        let filename = input["filename"].as_str();
        return kh
            .send_channel_media(
                &channel, recipient, "file", url, caption, filename, thread_id,
            )
            .await;
    }

    // Local file attachment: read from disk and send as FileData
    if let Some(raw_path) = file_path {
        let resolved = resolve_file_path(raw_path, workspace_root, None, false)?;
        let data = tokio::fs::read(&resolved)
            .await
            .map_err(|e| format!("Failed to read file '{}': {e}", resolved.display()))?;

        // Derive filename from the path if not explicitly provided
        let filename = input["filename"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                resolved
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file")
                    .to_string()
            });

        // Determine MIME type from extension
        let ext = resolved
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let mime_type = match ext.as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "svg" => "image/svg+xml",
            "pdf" => "application/pdf",
            "txt" => "text/plain",
            "csv" => "text/csv",
            "json" => "application/json",
            "xml" => "application/xml",
            "zip" => "application/zip",
            "gz" | "gzip" => "application/gzip",
            "tar" => "application/x-tar",
            "mp3" => "audio/mpeg",
            "wav" => "audio/wav",
            "mp4" => "video/mp4",
            "doc" => "application/msword",
            "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls" => "application/vnd.ms-excel",
            "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            _ => "application/octet-stream",
        };

        return kh
            .send_channel_file_data(&channel, recipient, data, &filename, mime_type, thread_id)
            .await;
    }

    // Text + optional `attachments: string[]` (ANAI-53). The attachments
    // array is sugar over the inline `<openfang:attach .../>` directive
    // mechanism — we synthesize directives here, prepend them to the
    // message body, and let `kernel::send_channel_message` →
    // `bridge::send_parsed` → `outbound_attach::parse` handle resolution,
    // workspace allow-root gating, and multipart construction. Composes
    // with any inline directives the agent already wrote into `message`.
    let message_raw = input["message"].as_str().unwrap_or("");
    let attachments_raw = input.get("attachments");
    let has_attachments = attachments_raw
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    if message_raw.is_empty() && !has_attachments {
        return Err(
            "Missing 'message' parameter (required for text-only sends; \
             pass non-empty 'attachments' to send files without a body)"
                .to_string(),
        );
    }

    // For email channels, validate email format and prepend subject
    let final_message = if channel == "email" {
        if !recipient.contains('@') || !recipient.contains('.') {
            return Err(format!("Invalid email address: '{recipient}'"));
        }
        if let Some(subject) = input["subject"].as_str() {
            if !subject.is_empty() {
                format!("Subject: {subject}\n\n{message_raw}")
            } else {
                message_raw.to_string()
            }
        } else {
            message_raw.to_string()
        }
    } else {
        message_raw.to_string()
    };

    let final_message = synthesize_attach_directives(&final_message, attachments_raw)?;

    kh.send_channel_message(
        &channel,
        recipient,
        &final_message,
        thread_id,
        workspace_root,
    )
    .await
}

/// Prepend synthesized `<openfang:attach path="…"/>` directives from an
/// `attachments` JSON array to the message body.
///
/// The outbound parser downstream extracts the directives, resolves each
/// path against the calling agent's `workspace_root`, applies the same
/// `allow_roots` security gating used by inline directives, and dispatches
/// multipart payloads on the wire. This helper is pure sugar over that
/// mechanism — no security policy here, just text synthesis.
///
/// Behavior:
/// - `None` or `Null` `attachments` → returns `message` unchanged.
/// - `attachments` present but not a JSON array → error (caller passed
///   wrong shape).
/// - Empty array → returns `message` unchanged (explicit no-op).
/// - Each element must be a non-empty string. Path characters that would
///   break the directive boundary (`"`, `<`, `>`, `\n`, `\r`) are
///   rejected outright rather than silently escaped — these are
///   pathological in filenames and silent escaping would obscure the
///   failure mode.
/// - Synthesized directives are prepended to `message` (one per line,
///   followed by the original body). Composes additively with any inline
///   directives the caller already embedded in `message`.
///
/// Path resolution and authorisation happen entirely downstream in
/// `outbound_attach::resolve_directive` — this helper does not touch the
/// filesystem.
fn synthesize_attach_directives(
    message: &str,
    attachments: Option<&serde_json::Value>,
) -> Result<String, String> {
    let arr = match attachments {
        Some(v) if v.is_null() => return Ok(message.to_string()),
        Some(v) => v
            .as_array()
            .ok_or("'attachments' must be an array of file path strings")?,
        None => return Ok(message.to_string()),
    };
    if arr.is_empty() {
        return Ok(message.to_string());
    }

    let mut directives = String::new();
    for (i, item) in arr.iter().enumerate() {
        let path = item
            .as_str()
            .ok_or_else(|| format!("'attachments[{i}]' must be a string"))?;
        if path.is_empty() {
            return Err(format!("'attachments[{i}]' is an empty string"));
        }
        if let Some(bad) = path
            .chars()
            .find(|c| matches!(c, '"' | '<' | '>' | '\n' | '\r'))
        {
            return Err(format!(
                "'attachments[{i}]' contains character {bad:?} which would break \
                 the directive boundary; use an inline `<openfang:attach .../>` \
                 directive or the `file_path` parameter instead"
            ));
        }
        directives.push_str("<openfang:attach path=\"");
        directives.push_str(path);
        directives.push_str("\"/>\n");
    }

    if message.is_empty() {
        // Trim the trailing newline so an attachments-only send doesn't
        // emit a phantom blank line through the formatter.
        Ok(directives.trim_end().to_string())
    } else {
        directives.push_str(message);
        Ok(directives)
    }
}

// ---------------------------------------------------------------------------
// Hand tools (delegated to kernel via KernelHandle trait)
// ---------------------------------------------------------------------------

async fn tool_hand_list(kernel: Option<&Arc<dyn KernelHandle>>) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let hands = kh.hand_list().await?;

    if hands.is_empty() {
        return Ok(
            "No Hands available. Install hands to enable curated autonomous packages.".to_string(),
        );
    }

    let mut lines = vec!["Available Hands:".to_string(), String::new()];
    for h in &hands {
        let icon = h["icon"].as_str().unwrap_or("");
        let name = h["name"].as_str().unwrap_or("?");
        let id = h["id"].as_str().unwrap_or("?");
        let status = h["status"].as_str().unwrap_or("unknown");
        let desc = h["description"].as_str().unwrap_or("");

        let status_marker = match status {
            "Active" => "[ACTIVE]",
            "Paused" => "[PAUSED]",
            _ => "[available]",
        };

        lines.push(format!("{} {} ({}) {}", icon, name, id, status_marker));
        if !desc.is_empty() {
            lines.push(format!("  {}", desc));
        }
        if let Some(iid) = h["instance_id"].as_str() {
            lines.push(format!("  Instance: {}", iid));
        }
        lines.push(String::new());
    }

    Ok(lines.join("\n"))
}

async fn tool_hand_activate(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let hand_id = input["hand_id"]
        .as_str()
        .ok_or("Missing 'hand_id' parameter")?;
    let config: std::collections::HashMap<String, serde_json::Value> =
        if let Some(obj) = input["config"].as_object() {
            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            std::collections::HashMap::new()
        };

    let result = kh.hand_activate(hand_id, config).await?;

    let instance_id = result["instance_id"].as_str().unwrap_or("?");
    let agent_name = result["agent_name"].as_str().unwrap_or("?");
    let status = result["status"].as_str().unwrap_or("?");

    Ok(format!(
        "Hand '{}' activated!\n  Instance: {}\n  Agent: {} ({})",
        hand_id, instance_id, agent_name, status
    ))
}

async fn tool_hand_status(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let hand_id = input["hand_id"]
        .as_str()
        .ok_or("Missing 'hand_id' parameter")?;

    let result = kh.hand_status(hand_id).await?;

    let icon = result["icon"].as_str().unwrap_or("");
    let name = result["name"].as_str().unwrap_or(hand_id);
    let status = result["status"].as_str().unwrap_or("unknown");
    let instance_id = result["instance_id"].as_str().unwrap_or("?");
    let agent_name = result["agent_name"].as_str().unwrap_or("?");
    let activated = result["activated_at"].as_str().unwrap_or("?");

    Ok(format!(
        "{} {} — {}\n  Instance: {}\n  Agent: {}\n  Activated: {}",
        icon, name, status, instance_id, agent_name, activated
    ))
}

async fn tool_hand_deactivate(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let instance_id = input["instance_id"]
        .as_str()
        .ok_or("Missing 'instance_id' parameter")?;
    kh.hand_deactivate(instance_id).await?;
    Ok(format!("Hand instance '{}' deactivated.", instance_id))
}

// ---------------------------------------------------------------------------
// A2A outbound tools (cross-instance agent communication)
// ---------------------------------------------------------------------------

/// Discover an external A2A agent by fetching its agent card.
async fn tool_a2a_discover(input: &serde_json::Value) -> Result<String, String> {
    let url = input["url"].as_str().ok_or("Missing 'url' parameter")?;

    // SSRF protection: block private/metadata IPs
    if crate::web_fetch::check_ssrf(url, &[]).is_err() {
        return Err("SSRF blocked: URL resolves to a private or metadata address".to_string());
    }

    let client = crate::a2a::A2aClient::new();
    let card = client.discover(url).await?;

    serde_json::to_string_pretty(&card).map_err(|e| format!("Serialization error: {e}"))
}

/// Send a task to an external A2A agent.
async fn tool_a2a_send(
    input: &serde_json::Value,
    kernel: Option<&Arc<dyn KernelHandle>>,
) -> Result<String, String> {
    let kh = require_kernel(kernel)?;
    let message = input["message"]
        .as_str()
        .ok_or("Missing 'message' parameter")?;

    // Resolve agent URL: either directly provided or looked up by name
    let url = if let Some(url) = input["agent_url"].as_str() {
        // SSRF protection
        if crate::web_fetch::check_ssrf(url, &[]).is_err() {
            return Err("SSRF blocked: URL resolves to a private or metadata address".to_string());
        }
        url.to_string()
    } else if let Some(name) = input["agent_name"].as_str() {
        kh.get_a2a_agent_url(name)
            .ok_or_else(|| format!("No known A2A agent with name '{name}'. Use a2a_discover first or provide agent_url directly."))?
    } else {
        return Err("Missing 'agent_url' or 'agent_name' parameter".to_string());
    };

    let session_id = input["session_id"].as_str();
    let client = crate::a2a::A2aClient::new();
    let task = client.send_task(&url, message, session_id).await?;

    serde_json::to_string_pretty(&task).map_err(|e| format!("Serialization error: {e}"))
}

// ---------------------------------------------------------------------------
// Image analysis tool
// ---------------------------------------------------------------------------

async fn tool_image_analyze(input: &serde_json::Value) -> Result<String, String> {
    let path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let prompt = input["prompt"].as_str().unwrap_or("");

    let data = tokio::fs::read(path)
        .await
        .map_err(|e| format!("Failed to read image '{path}': {e}"))?;

    let file_size = data.len();

    // Detect image format from magic bytes
    let format = detect_image_format(&data);

    // Extract dimensions for common formats
    let dimensions = extract_image_dimensions(&data, &format);

    // Base64-encode (truncate for very large images in the response)
    let base64_preview = if file_size <= 512 * 1024 {
        // Under 512KB — include full base64
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(&data)
    } else {
        // Over 512KB — include first 64KB preview
        use base64::Engine;
        let preview_bytes = &data[..64 * 1024];
        format!(
            "{}... [truncated, {} total bytes]",
            base64::engine::general_purpose::STANDARD.encode(preview_bytes),
            file_size
        )
    };

    let mut result = serde_json::json!({
        "path": path,
        "format": format,
        "file_size_bytes": file_size,
        "file_size_human": format_file_size(file_size),
    });

    if let Some((w, h)) = dimensions {
        result["width"] = serde_json::json!(w);
        result["height"] = serde_json::json!(h);
    }

    if !prompt.is_empty() {
        result["prompt"] = serde_json::json!(prompt);
        result["note"] = serde_json::json!(
            "Vision analysis requires a vision-capable LLM. The base64 data is included for downstream processing."
        );
    }

    result["base64_preview"] = serde_json::json!(base64_preview);

    serde_json::to_string_pretty(&result).map_err(|e| format!("Serialize error: {e}"))
}

/// Detect image format from magic bytes.
fn detect_image_format(data: &[u8]) -> String {
    if data.len() < 4 {
        return "unknown".to_string();
    }
    if data.starts_with(b"\x89PNG") {
        "png".to_string()
    } else if data.starts_with(b"\xFF\xD8\xFF") {
        "jpeg".to_string()
    } else if data.starts_with(b"GIF8") {
        "gif".to_string()
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        "webp".to_string()
    } else if data.starts_with(b"BM") {
        "bmp".to_string()
    } else if data.starts_with(b"\x00\x00\x01\x00") {
        "ico".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Extract image dimensions from common formats.
fn extract_image_dimensions(data: &[u8], format: &str) -> Option<(u32, u32)> {
    match format {
        "png" => {
            // PNG: IHDR chunk starts at byte 16, width at 16-19, height at 20-23
            if data.len() >= 24 {
                let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
                let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
                Some((w, h))
            } else {
                None
            }
        }
        "gif" => {
            // GIF: width at bytes 6-7, height at bytes 8-9 (little-endian)
            if data.len() >= 10 {
                let w = u16::from_le_bytes([data[6], data[7]]) as u32;
                let h = u16::from_le_bytes([data[8], data[9]]) as u32;
                Some((w, h))
            } else {
                None
            }
        }
        "bmp" => {
            // BMP: width at bytes 18-21, height at bytes 22-25 (little-endian)
            if data.len() >= 26 {
                let w = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);
                let h = u32::from_le_bytes([data[22], data[23], data[24], data[25]]);
                Some((w, h))
            } else {
                None
            }
        }
        "jpeg" => {
            // JPEG: scan for SOF0 marker (0xFF 0xC0) to find dimensions
            extract_jpeg_dimensions(data)
        }
        _ => None,
    }
}

/// Extract JPEG dimensions by scanning for SOF markers.
fn extract_jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2; // Skip SOI marker
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        // SOF0-SOF3 markers contain dimensions
        if (0xC0..=0xC3).contains(&marker) && i + 9 < data.len() {
            let h = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32;
            let w = u16::from_be_bytes([data[i + 7], data[i + 8]]) as u32;
            return Some((w, h));
        }
        if i + 3 < data.len() {
            let seg_len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            i += 2 + seg_len;
        } else {
            break;
        }
    }
    None
}

/// Format file size in human-readable form.
fn format_file_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// ---------------------------------------------------------------------------
// Location tool
// ---------------------------------------------------------------------------

async fn tool_location_get() -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    // Use ip-api.com (free, no API key, JSON response)
    let resp = client
        .get("https://ip-api.com/json/?fields=status,message,country,regionName,city,zip,lat,lon,timezone,isp,query")
        .header("User-Agent", "OpenFang/0.1")
        .send()
        .await
        .map_err(|e| format!("Location request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Location API returned {}", resp.status()));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse location response: {e}"))?;

    if body["status"].as_str() != Some("success") {
        let msg = body["message"].as_str().unwrap_or("Unknown error");
        return Err(format!("Location lookup failed: {msg}"));
    }

    let result = serde_json::json!({
        "lat": body["lat"],
        "lon": body["lon"],
        "city": body["city"],
        "region": body["regionName"],
        "country": body["country"],
        "zip": body["zip"],
        "timezone": body["timezone"],
        "isp": body["isp"],
        "ip": body["query"],
    });

    serde_json::to_string_pretty(&result).map_err(|e| format!("Serialize error: {e}"))
}

// ---------------------------------------------------------------------------
// System time tool
// ---------------------------------------------------------------------------

/// Return current date, time, timezone, and Unix epoch.
fn tool_system_time() -> String {
    let now_utc = chrono::Utc::now();
    let now_local = chrono::Local::now();
    let result = serde_json::json!({
        "utc": now_utc.to_rfc3339(),
        "local": now_local.to_rfc3339(),
        "unix_epoch": now_utc.timestamp(),
        "timezone": now_local.format("%Z").to_string(),
        "utc_offset": now_local.format("%:z").to_string(),
        "date": now_local.format("%Y-%m-%d").to_string(),
        "time": now_local.format("%H:%M:%S").to_string(),
        "day_of_week": now_local.format("%A").to_string(),
    });
    serde_json::to_string_pretty(&result).unwrap_or_else(|_| now_utc.to_rfc3339())
}

// ---------------------------------------------------------------------------
// Media understanding tools
// ---------------------------------------------------------------------------

/// Describe an image using a vision-capable LLM provider.
async fn tool_media_describe(
    input: &serde_json::Value,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
) -> Result<String, String> {
    use base64::Engine;
    let engine = media_engine.ok_or("Media engine not available. Check media configuration.")?;
    let path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let _ = validate_path(path)?;

    // Read image file
    let data = tokio::fs::read(path)
        .await
        .map_err(|e| format!("Failed to read image file: {e}"))?;

    // Detect MIME type from extension
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        _ => return Err(format!("Unsupported image format: .{ext}")),
    };

    let attachment = openfang_types::media::MediaAttachment {
        media_type: openfang_types::media::MediaType::Image,
        mime_type: mime.to_string(),
        source: openfang_types::media::MediaSource::Base64 {
            data: base64::engine::general_purpose::STANDARD.encode(&data),
            mime_type: mime.to_string(),
        },
        size_bytes: data.len() as u64,
    };

    let understanding = engine.describe_image(&attachment).await?;
    serde_json::to_string_pretty(&understanding).map_err(|e| format!("Serialize error: {e}"))
}

/// Transcribe audio to text using speech-to-text.
async fn tool_media_transcribe(
    input: &serde_json::Value,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
) -> Result<String, String> {
    use base64::Engine;
    let engine = media_engine.ok_or("Media engine not available. Check media configuration.")?;
    let path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let _ = validate_path(path)?;

    // Read audio file
    let data = tokio::fs::read(path)
        .await
        .map_err(|e| format!("Failed to read audio file: {e}"))?;

    // Detect MIME type from extension
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mime = match ext.as_str() {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "webm" => "audio/webm",
        _ => return Err(format!("Unsupported audio format: .{ext}")),
    };

    let attachment = openfang_types::media::MediaAttachment {
        media_type: openfang_types::media::MediaType::Audio,
        mime_type: mime.to_string(),
        source: openfang_types::media::MediaSource::Base64 {
            data: base64::engine::general_purpose::STANDARD.encode(&data),
            mime_type: mime.to_string(),
        },
        size_bytes: data.len() as u64,
    };

    let understanding = engine.transcribe_audio(&attachment).await?;
    serde_json::to_string_pretty(&understanding).map_err(|e| format!("Serialize error: {e}"))
}

// ---------------------------------------------------------------------------
// Image generation tool
// ---------------------------------------------------------------------------

/// Generate images from a text prompt.
async fn tool_image_generate(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
) -> Result<String, String> {
    let prompt = input["prompt"]
        .as_str()
        .ok_or("Missing 'prompt' parameter")?;

    let model_str = input["model"].as_str().unwrap_or("dall-e-3");
    let model = match model_str {
        "dall-e-3" | "dalle3" | "dalle-3" => openfang_types::media::ImageGenModel::DallE3,
        "dall-e-2" | "dalle2" | "dalle-2" => openfang_types::media::ImageGenModel::DallE2,
        "gpt-image-1" | "gpt_image_1" => openfang_types::media::ImageGenModel::GptImage1,
        _ => {
            return Err(format!(
                "Unknown image model: {model_str}. Use 'dall-e-3', 'dall-e-2', or 'gpt-image-1'."
            ))
        }
    };

    let size = input["size"].as_str().unwrap_or("1024x1024").to_string();
    let quality = input["quality"].as_str().unwrap_or("hd").to_string();
    let count = input["count"].as_u64().unwrap_or(1).min(4) as u8;

    let request = openfang_types::media::ImageGenRequest {
        prompt: prompt.to_string(),
        model,
        size,
        quality,
        count,
    };

    // Closes #1051: route to a local OpenAI-compatible image generation
    // service when `media.image_gen_base_url` is set.
    let base_url_override = media_engine.and_then(|e| e.config().image_gen_base_url.as_deref());
    let result = crate::image_gen::generate_image(&request, base_url_override).await?;

    // Save images to workspace if available
    let saved_paths = if let Some(workspace) = workspace_root {
        match crate::image_gen::save_images_to_workspace(&result, workspace) {
            Ok(paths) => paths,
            Err(e) => {
                warn!("Failed to save images to workspace: {e}");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // Also save to the uploads temp dir so the web UI can serve them via
    // GET /api/uploads/{file_id}.  Each image gets a UUID filename.
    let mut image_urls: Vec<String> = Vec::new();
    {
        use base64::Engine;
        let upload_dir = std::env::temp_dir().join("openfang_uploads");
        let _ = std::fs::create_dir_all(&upload_dir);
        for img in &result.images {
            let file_id = uuid::Uuid::new_v4().to_string();
            if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(&img.data_base64)
            {
                let path = upload_dir.join(&file_id);
                if std::fs::write(&path, &decoded).is_ok() {
                    image_urls.push(format!("/api/uploads/{file_id}"));
                }
            }
        }
    }

    // Build response — include image_urls so the dashboard can render <img> tags
    let response = serde_json::json!({
        "model": result.model,
        "images_generated": result.images.len(),
        "saved_to": saved_paths,
        "revised_prompt": result.revised_prompt,
        "image_urls": image_urls,
    });

    serde_json::to_string_pretty(&response).map_err(|e| format!("Serialize error: {e}"))
}

// ---------------------------------------------------------------------------
// TTS / STT tools
// ---------------------------------------------------------------------------

async fn tool_text_to_speech(
    input: &serde_json::Value,
    tts_engine: Option<&crate::tts::TtsEngine>,
    workspace_root: Option<&Path>,
) -> Result<String, String> {
    let engine =
        tts_engine.ok_or("TTS engine not available. Ensure tts.enabled=true in config.")?;
    let text = input["text"].as_str().ok_or("Missing 'text' parameter")?;
    let voice = input["voice"].as_str();
    let format = input["format"].as_str();

    let result = engine.synthesize(text, voice, format).await?;

    // Save audio to workspace
    let saved_path = if let Some(workspace) = workspace_root {
        let output_dir = workspace.join("output");
        tokio::fs::create_dir_all(&output_dir)
            .await
            .map_err(|e| format!("Failed to create output dir: {e}"))?;

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("tts_{timestamp}.{}", result.format);
        let path = output_dir.join(&filename);

        tokio::fs::write(&path, &result.audio_data)
            .await
            .map_err(|e| format!("Failed to write audio file: {e}"))?;

        Some(path.display().to_string())
    } else {
        None
    };

    let response = serde_json::json!({
        "saved_to": saved_path,
        "format": result.format,
        "provider": result.provider,
        "duration_estimate_ms": result.duration_estimate_ms,
        "size_bytes": result.audio_data.len(),
    });

    serde_json::to_string_pretty(&response).map_err(|e| format!("Serialize error: {e}"))
}

async fn tool_speech_to_text(
    input: &serde_json::Value,
    media_engine: Option<&crate::media_understanding::MediaEngine>,
    workspace_root: Option<&Path>,
) -> Result<String, String> {
    let engine = media_engine.ok_or("Media engine not available for speech-to-text")?;
    let raw_path = input["path"].as_str().ok_or("Missing 'path' parameter")?;
    let _language = input["language"].as_str();

    let resolved = resolve_file_path(raw_path, workspace_root, None, false)?;

    // Read the audio file
    let data = tokio::fs::read(&resolved)
        .await
        .map_err(|e| format!("Failed to read audio file: {e}"))?;

    // Determine MIME type from extension
    let ext = resolved
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp3");
    let mime_type = match ext {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "webm" => "audio/webm",
        _ => "audio/mpeg",
    };

    use openfang_types::media::{MediaAttachment, MediaSource, MediaType};
    let attachment = MediaAttachment {
        media_type: MediaType::Audio,
        mime_type: mime_type.to_string(),
        source: MediaSource::Base64 {
            data: {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&data)
            },
            mime_type: mime_type.to_string(),
        },
        size_bytes: data.len() as u64,
    };

    let understanding = engine.transcribe_audio(&attachment).await?;

    let response = serde_json::json!({
        "transcript": understanding.description,
        "provider": understanding.provider,
        "model": understanding.model,
    });

    serde_json::to_string_pretty(&response).map_err(|e| format!("Serialize error: {e}"))
}

// ---------------------------------------------------------------------------
// Docker sandbox tool
// ---------------------------------------------------------------------------

async fn tool_docker_exec(
    input: &serde_json::Value,
    docker_config: Option<&openfang_types::config::DockerSandboxConfig>,
    workspace_root: Option<&Path>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let config = docker_config.ok_or("Docker sandbox not configured")?;

    if !config.enabled {
        return Err("Docker sandbox is disabled. Set docker.enabled=true in config.".into());
    }

    let command = input["command"]
        .as_str()
        .ok_or("Missing 'command' parameter")?;

    let workspace = workspace_root.ok_or("Docker exec requires a workspace directory")?;
    let agent_id = caller_agent_id.unwrap_or("default");

    // Check Docker availability
    if !crate::docker_sandbox::is_docker_available().await {
        return Err(
            "Docker is not available on this system. Install Docker to use docker_exec.".into(),
        );
    }

    // Create sandbox container
    let container = crate::docker_sandbox::create_sandbox(config, agent_id, workspace).await?;

    // Execute command with timeout
    let timeout = std::time::Duration::from_secs(config.timeout_secs);
    let result = crate::docker_sandbox::exec_in_sandbox(&container, command, timeout).await;

    // Always destroy the container after execution
    if let Err(e) = crate::docker_sandbox::destroy_sandbox(&container).await {
        warn!("Failed to destroy Docker sandbox: {e}");
    }

    let exec_result = result?;

    let response = serde_json::json!({
        "exit_code": exec_result.exit_code,
        "stdout": exec_result.stdout,
        "stderr": exec_result.stderr,
        "container_id": container.container_id,
    });

    serde_json::to_string_pretty(&response).map_err(|e| format!("Serialize error: {e}"))
}

// ---------------------------------------------------------------------------
// Persistent process tools
// ---------------------------------------------------------------------------

/// Start a long-running process (REPL, server, watcher).
///
/// SECURITY (#919): process_start previously spawned subprocesses with NO
/// exec policy enforcement, allowing an LLM in Allowlist mode to bypass
/// allowed_commands entirely. For example, process_start with command="rm"
/// args=["/some/file"] would delete the file even though "rm" was not
/// in the allowlist. This function now performs the same checks as
/// shell_exec: metacharacter rejection plus exec_policy validation.
async fn tool_process_start(
    input: &serde_json::Value,
    pm: Option<&crate::process_manager::ProcessManager>,
    caller_agent_id: Option<&str>,
    exec_policy: Option<&openfang_types::config::ExecPolicy>,
) -> Result<String, String> {
    let pm = pm.ok_or("Process manager not available")?;
    let agent_id = caller_agent_id.unwrap_or("default");
    let command = input["command"]
        .as_str()
        .ok_or("Missing 'command' parameter")?;
    let args: Vec<String> = input["args"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // SECURITY: Reject shell metacharacters in the command name itself.
    // The command field must be a single binary token.
    if let Some(reason) = crate::subprocess_sandbox::contains_shell_metacharacters(command) {
        return Err(format!(
            "process_start blocked: command contains {reason}. \
             Shell metacharacters are never allowed in the command field."
        ));
    }
    // Also reject metacharacters anywhere in the arguments. While direct
    // spawn does not interpret these, blocking them prevents an LLM from
    // smuggling a chained command past the allowlist via an argument.
    for arg in &args {
        if let Some(reason) = crate::subprocess_sandbox::contains_shell_metacharacters(arg) {
            return Err(format!(
                "process_start blocked: argument contains {reason}. \
                 Shell metacharacters are not allowed in process arguments."
            ));
        }
    }

    // SECURITY (#919): Enforce exec policy against the base command. The
    // shared validate_command_allowlist handles Deny / Full / Allowlist and
    // falls through to allow commands listed in safe_bins or allowed_commands.
    if let Some(policy) = exec_policy {
        if let Err(reason) = crate::subprocess_sandbox::validate_command_allowlist(command, policy)
        {
            let reason = reason.trim_end_matches('.');
            return Err(format!(
                "process_start blocked: {reason}. Current exec_policy.mode = '{:?}'. \
                 To allow this command, add it to exec_policy.allowed_commands or \
                 set exec_policy.mode = 'full'.",
                policy.mode
            ));
        }
    }

    let proc_id = pm.start(agent_id, command, &args).await?;
    Ok(serde_json::json!({
        "process_id": proc_id,
        "status": "started"
    })
    .to_string())
}

/// Read accumulated stdout/stderr from a process (non-blocking drain).
async fn tool_process_poll(
    input: &serde_json::Value,
    pm: Option<&crate::process_manager::ProcessManager>,
) -> Result<String, String> {
    let pm = pm.ok_or("Process manager not available")?;
    let proc_id = input["process_id"]
        .as_str()
        .ok_or("Missing 'process_id' parameter")?;
    let (stdout, stderr) = pm.read(proc_id).await?;
    Ok(serde_json::json!({
        "stdout": stdout,
        "stderr": stderr,
    })
    .to_string())
}

/// Write data to a process's stdin.
async fn tool_process_write(
    input: &serde_json::Value,
    pm: Option<&crate::process_manager::ProcessManager>,
) -> Result<String, String> {
    let pm = pm.ok_or("Process manager not available")?;
    let proc_id = input["process_id"]
        .as_str()
        .ok_or("Missing 'process_id' parameter")?;
    let data = input["data"].as_str().ok_or("Missing 'data' parameter")?;
    // Always append newline if not present (common expectation for REPLs)
    let data = if data.ends_with('\n') {
        data.to_string()
    } else {
        format!("{data}\n")
    };
    pm.write(proc_id, &data).await?;
    Ok(r#"{"status": "written"}"#.to_string())
}

/// Terminate a process.
async fn tool_process_kill(
    input: &serde_json::Value,
    pm: Option<&crate::process_manager::ProcessManager>,
) -> Result<String, String> {
    let pm = pm.ok_or("Process manager not available")?;
    let proc_id = input["process_id"]
        .as_str()
        .ok_or("Missing 'process_id' parameter")?;
    pm.kill(proc_id).await?;
    Ok(r#"{"status": "killed"}"#.to_string())
}

/// List processes for the current agent.
async fn tool_process_list(
    pm: Option<&crate::process_manager::ProcessManager>,
    caller_agent_id: Option<&str>,
) -> Result<String, String> {
    let pm = pm.ok_or("Process manager not available")?;
    let agent_id = caller_agent_id.unwrap_or("default");
    let procs = pm.list(agent_id);
    let list: Vec<serde_json::Value> = procs
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": p.id,
                "command": p.command,
                "alive": p.alive,
                "uptime_secs": p.uptime_secs,
            })
        })
        .collect();
    Ok(serde_json::Value::Array(list).to_string())
}

// ---------------------------------------------------------------------------
// Canvas / A2UI tool
// ---------------------------------------------------------------------------

/// Sanitize HTML for canvas presentation.
///
/// SECURITY: Strips dangerous elements and attributes to prevent XSS:
/// - Rejects <script>, <iframe>, <object>, <embed>, <applet> tags
/// - Strips all on* event attributes (onclick, onload, onerror, etc.)
/// - Strips javascript:, data:text/html, vbscript: URLs
/// - Enforces size limit
pub fn sanitize_canvas_html(html: &str, max_bytes: usize) -> Result<String, String> {
    if html.is_empty() {
        return Err("Empty HTML content".to_string());
    }
    if html.len() > max_bytes {
        return Err(format!(
            "HTML too large: {} bytes (max {})",
            html.len(),
            max_bytes
        ));
    }

    let lower = html.to_lowercase();

    // Reject dangerous tags
    let dangerous_tags = [
        "<script", "</script", "<iframe", "</iframe", "<object", "</object", "<embed", "<applet",
        "</applet",
    ];
    for tag in &dangerous_tags {
        if lower.contains(tag) {
            return Err(format!("Forbidden HTML tag detected: {tag}"));
        }
    }

    // Reject event handler attributes (on*)
    // Match patterns like: onclick=, onload=, onerror=, onmouseover=, etc.
    static EVENT_PATTERN: std::sync::LazyLock<regex_lite::Regex> =
        std::sync::LazyLock::new(|| regex_lite::Regex::new(r"(?i)\bon[a-z]+\s*=").unwrap());
    if EVENT_PATTERN.is_match(html) {
        return Err(
            "Forbidden event handler attribute detected (on* attributes are not allowed)"
                .to_string(),
        );
    }

    // Reject dangerous URL schemes
    let dangerous_schemes = ["javascript:", "vbscript:", "data:text/html"];
    for scheme in &dangerous_schemes {
        if lower.contains(scheme) {
            return Err(format!("Forbidden URL scheme detected: {scheme}"));
        }
    }

    Ok(html.to_string())
}

/// Canvas presentation tool handler.
async fn tool_canvas_present(
    input: &serde_json::Value,
    workspace_root: Option<&Path>,
) -> Result<String, String> {
    let html = input["html"].as_str().ok_or("Missing 'html' parameter")?;
    let title = input["title"].as_str().unwrap_or("Canvas");

    // Use configured max from task-local (set by agent_loop from KernelConfig), or default 512KB.
    let max_bytes = CANVAS_MAX_BYTES.try_with(|v| *v).unwrap_or(512 * 1024);
    let sanitized = sanitize_canvas_html(html, max_bytes)?;

    // Generate canvas ID
    let canvas_id = uuid::Uuid::new_v4().to_string();

    // Save to workspace output directory
    let output_dir = if let Some(root) = workspace_root {
        root.join("output")
    } else {
        PathBuf::from("output")
    };
    let _ = tokio::fs::create_dir_all(&output_dir).await;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!(
        "canvas_{timestamp}_{}.html",
        crate::str_utils::safe_truncate_str(&canvas_id, 8)
    );
    let filepath = output_dir.join(&filename);

    // Write the full HTML document
    let full_html = format!(
        "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"><title>{title}</title></head>\n<body>\n{sanitized}\n</body>\n</html>"
    );
    tokio::fs::write(&filepath, &full_html)
        .await
        .map_err(|e| format!("Failed to save canvas: {e}"))?;

    let response = serde_json::json!({
        "canvas_id": canvas_id,
        "title": title,
        "saved_to": filepath.to_string_lossy(),
        "size_bytes": full_html.len(),
    });

    serde_json::to_string_pretty(&response).map_err(|e| format!("Serialize error: {e}"))
}

// ---------------------------------------------------------------------------
// Skill introspection tools (issue #1038)
//
// Global skills live at ~/.openfang/skills/ which is outside the agent
// workspace sandbox. Without these tools the LLM falls back to file_read /
// shell_exec to inspect SKILL.md files — which fail with path-resolution
// errors because file_read is workspace-scoped. These tools surface the
// already-loaded skill registry directly to the agent.
// ---------------------------------------------------------------------------

/// List all skills available to this agent, with their provided tool names.
fn tool_skill_list(skill_registry: Option<&SkillRegistry>) -> Result<String, String> {
    let registry = match skill_registry {
        Some(r) => r,
        None => return Ok("No skill registry available.".to_string()),
    };
    let skills = registry.list();
    if skills.is_empty() {
        return Ok("No skills installed. Install skills via the dashboard or `openfang skill install <name>`.".to_string());
    }
    let entries: Vec<serde_json::Value> = skills
        .iter()
        .map(|s| {
            let tool_names: Vec<String> = s
                .manifest
                .tools
                .provided
                .iter()
                .map(|t| t.name.clone())
                .collect();
            serde_json::json!({
                "name": s.manifest.skill.name,
                "version": s.manifest.skill.version,
                "description": s.manifest.skill.description,
                "runtime": format!("{:?}", s.manifest.runtime.runtime_type),
                "enabled": s.enabled,
                "tools": tool_names,
                "has_prompt_context": s.manifest.prompt_context.as_ref().is_some_and(|c| !c.is_empty()),
            })
        })
        .collect();
    serde_json::to_string_pretty(&serde_json::json!({
        "count": entries.len(),
        "skills": entries,
    }))
    .map_err(|e| format!("Serialize error: {e}"))
}

/// Return the full description (SKILL.md body) of a named skill.
fn tool_skill_describe(
    input: &serde_json::Value,
    skill_registry: Option<&SkillRegistry>,
) -> Result<String, String> {
    let name = input["name"]
        .as_str()
        .ok_or("Missing 'name' parameter")?
        .trim();
    let registry = skill_registry.ok_or("No skill registry available")?;
    let skill = registry.get(name).ok_or_else(|| {
        format!("Skill '{name}' not found. Use skill_list to see installed skills.")
    })?;
    let body = skill
        .manifest
        .prompt_context
        .clone()
        .unwrap_or_else(|| "(No prompt context body — this skill provides executable tools only. Use skill_execute or call its tools directly.)".to_string());
    let tool_names: Vec<String> = skill
        .manifest
        .tools
        .provided
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let response = serde_json::json!({
        "name": skill.manifest.skill.name,
        "version": skill.manifest.skill.version,
        "description": skill.manifest.skill.description,
        "runtime": format!("{:?}", skill.manifest.runtime.runtime_type),
        "tools": tool_names,
        "body": body,
    });
    serde_json::to_string_pretty(&response).map_err(|e| format!("Serialize error: {e}"))
}

/// Execute a skill's tool, or for prompt-only skills return the description body.
async fn tool_skill_execute(
    input: &serde_json::Value,
    skill_registry: Option<&SkillRegistry>,
) -> Result<String, String> {
    let skill_name = input["skill"]
        .as_str()
        .ok_or("Missing 'skill' parameter")?
        .trim();
    let registry = skill_registry.ok_or("No skill registry available")?;
    let skill = registry.get(skill_name).ok_or_else(|| {
        format!("Skill '{skill_name}' not found. Use skill_list to see installed skills.")
    })?;

    // If no tool name was given, default behavior depends on runtime.
    // For prompt-only skills, return the SKILL.md body (most useful response
    // for issue #1038's daily-journal style skills).
    let tool_name = input["tool"].as_str().map(|s| s.trim());
    let tool_input = input.get("input").cloned().unwrap_or(serde_json::json!({}));

    let resolved_tool = match tool_name {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => {
            // No tool specified — return SKILL.md body so the agent can act on it.
            if let Some(ref body) = skill.manifest.prompt_context {
                if !body.is_empty() {
                    let response = serde_json::json!({
                        "skill": skill.manifest.skill.name,
                        "mode": "prompt_context",
                        "body": body,
                        "note": "This is a prompt-only skill. Follow the instructions in 'body' using your built-in tools.",
                    });
                    return serde_json::to_string_pretty(&response)
                        .map_err(|e| format!("Serialize error: {e}"));
                }
            }
            // Fall through: pick the first provided tool if any
            skill
                .manifest
                .tools
                .provided
                .first()
                .map(|t| t.name.clone())
                .ok_or_else(|| {
                    format!("Skill '{skill_name}' provides no tools and has no prompt body.")
                })?
        }
    };

    match openfang_skills::loader::execute_skill_tool(
        &skill.manifest,
        &skill.path,
        &resolved_tool,
        &tool_input,
    )
    .await
    {
        Ok(result) => {
            let content = serde_json::to_string_pretty(&serde_json::json!({
                "skill": skill.manifest.skill.name,
                "tool": resolved_tool,
                "output": result.output,
                "is_error": result.is_error,
            }))
            .unwrap_or_else(|_| result.output.to_string());
            if result.is_error {
                Err(content)
            } else {
                Ok(content)
            }
        }
        Err(e) => Err(format!("Skill execution failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ANAI-291: file_read ranges, head+manifest, binary signpost --------

    /// Call `tool_file_read` against a real file with no workspace and no
    /// policy, which is the configuration `resolve_file_path` treats as
    /// "absolute paths permitted" — the read path itself is what is under
    /// test here, not the resolver.
    async fn read(path: &Path, args: serde_json::Value) -> Result<String, String> {
        let mut input = args;
        input["path"] = serde_json::json!(path.to_str().unwrap());
        tool_file_read(&input, None, None, None).await
    }

    fn write_lines(dir: &Path, name: &str, count: usize) -> PathBuf {
        let path = dir.join(name);
        let body: String = (1..=count).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&path, body).unwrap();
        path
    }

    #[tokio::test]
    async fn a_small_unranged_read_is_byte_identical_to_the_file() {
        // Backward compatibility is the load-bearing property here. Every
        // existing caller that hashes, diffs, or round-trips content through
        // file_read breaks the moment an unranged read of a normal file grows
        // a header.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.txt");
        let content = "alpha\nbravo\ncharlie\n";
        std::fs::write(&path, content).unwrap();

        let got = read(&path, serde_json::json!({})).await.unwrap();
        assert_eq!(got, content, "no header, no annotation, no rewriting");
    }

    #[tokio::test]
    async fn a_range_returns_exactly_those_lines_with_a_denominator() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_lines(dir.path(), "many.txt", 500);

        let got = read(&path, serde_json::json!({ "offset": 10, "limit": 3 }))
            .await
            .unwrap();

        let (header, body) = got.split_once('\n').unwrap();
        assert!(
            header.contains("lines 10-12 of 500"),
            "the header must state the window AND the total, so a slice is \
             never mistaken for the file: {header}"
        );
        assert_eq!(body, "line 10\nline 11\nline 12\n");
    }

    #[tokio::test]
    async fn a_limit_without_an_offset_starts_at_line_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_lines(dir.path(), "many.txt", 50);
        let got = read(&path, serde_json::json!({ "limit": 2 }))
            .await
            .unwrap();
        assert!(got.contains("lines 1-2 of 50"));
        assert!(got.ends_with("line 1\nline 2\n"));
    }

    #[tokio::test]
    async fn an_offset_without_a_limit_runs_to_the_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_lines(dir.path(), "many.txt", 5);
        let got = read(&path, serde_json::json!({ "offset": 4 }))
            .await
            .unwrap();
        assert!(got.contains("lines 4-5 of 5"));
        assert!(got.ends_with("line 4\nline 5\n"));
    }

    #[tokio::test]
    async fn a_large_unranged_read_returns_a_head_and_a_manifest() {
        // Each line is ~1 KiB, so 400 lines clears the 128 KiB whole-file
        // limit without depending on the exact constant.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let filler = "x".repeat(1000);
        let body: String = (1..=400).map(|i| format!("{i} {filler}\n")).collect();
        std::fs::write(&path, &body).unwrap();

        let got = read(&path, serde_json::json!({})).await.unwrap();

        assert!(got.len() < body.len(), "the whole file must not come back");
        // The manifest's job: say what this is, prove which file it is, and
        // name the call that gets the rest.
        assert!(got.contains("HEAD ONLY"), "must not read as the whole file");
        assert!(got.contains("400 lines"), "must state the true denominator");
        assert!(got.contains("sha256="), "must identify the file");
        assert!(
            got.contains("offset=201"),
            "must name the NEXT call concretely, not describe it: {}",
            &got[..600.min(got.len())]
        );
        assert!(
            got.contains("file_grep"),
            "must point at search, which is what a large read usually wants"
        );
        // And the head is really the head.
        assert!(got.contains("\n1 xxx"), "line 1 present");
        assert!(got.contains("\n200 xxx"), "line 200 present");
        assert!(!got.contains("\n201 xxx"), "line 201 must NOT be present");
    }

    #[tokio::test]
    async fn an_explicit_range_is_honoured_past_the_whole_file_limit() {
        // The head+manifest substitution must apply ONLY to a caller who
        // stated no bound. A caller who asked for lines 300-302 of a large
        // file gets lines 300-302, not a head.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let filler = "y".repeat(1000);
        let body: String = (1..=400).map(|i| format!("{i} {filler}\n")).collect();
        std::fs::write(&path, body).unwrap();

        let got = read(&path, serde_json::json!({ "offset": 300, "limit": 3 }))
            .await
            .unwrap();
        assert!(got.contains("lines 300-302 of 400"));
        assert!(!got.contains("HEAD ONLY"));
        assert!(got.contains("\n300 yyy"));
    }

    #[tokio::test]
    async fn an_offset_past_the_end_is_a_refusal_not_an_empty_success() {
        // Returning "" here is indistinguishable from an empty file, and the
        // caller would carry on believing it had read something. This is the
        // silent-success class, so it is an error with the real line count.
        let dir = tempfile::tempdir().unwrap();
        let path = write_lines(dir.path(), "short.txt", 12);

        let err = read(&path, serde_json::json!({ "offset": 5000 }))
            .await
            .unwrap_err();
        assert!(err.contains("past the end"), "{err}");
        assert!(err.contains("12 lines"), "must name the real length: {err}");
        assert!(err.contains("Nothing was returned"), "{err}");
    }

    #[tokio::test]
    async fn a_zero_offset_is_refused_as_a_one_based_mistake() {
        // Coercing 0 to 1 would hide a real off-by-one in the caller; treating
        // it as "line zero" would shift every subsequent slice by one.
        let dir = tempfile::tempdir().unwrap();
        let path = write_lines(dir.path(), "f.txt", 3);
        let err = read(&path, serde_json::json!({ "offset": 0 }))
            .await
            .unwrap_err();
        assert!(err.contains("1-based"), "{err}");
    }

    #[tokio::test]
    async fn a_numeric_string_range_is_accepted_and_junk_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_lines(dir.path(), "f.txt", 9);

        // Models send "2" as readily as 2. Accepting it is liberal; accepting
        // "two" silently as 0 or 1 would not be.
        let got = read(&path, serde_json::json!({ "offset": "2", "limit": "2" }))
            .await
            .unwrap();
        assert!(got.contains("lines 2-3 of 9"));

        let err = read(&path, serde_json::json!({ "limit": "lots" }))
            .await
            .unwrap_err();
        assert!(err.contains("whole number of lines"), "{err}");
    }

    #[tokio::test]
    async fn crlf_line_endings_survive_a_ranged_read() {
        // `BufReader::lines()` strips the terminator and a trailing '\r',
        // which would silently rewrite the file's bytes on the way out. The
        // window is assembled with read_until precisely to avoid that.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crlf.txt");
        std::fs::write(&path, "one\r\ntwo\r\nthree\r\n").unwrap();

        let got = read(&path, serde_json::json!({ "offset": 2, "limit": 1 }))
            .await
            .unwrap();
        assert!(
            got.ends_with("two\r\n"),
            "the carriage return must still be there: {got:?}"
        );
    }

    #[tokio::test]
    async fn a_file_with_no_trailing_newline_is_not_given_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonl.txt");
        std::fs::write(&path, "a\nb").unwrap();

        let got = read(&path, serde_json::json!({ "offset": 2, "limit": 1 }))
            .await
            .unwrap();
        assert!(got.ends_with("b"), "{got:?}");
        assert!(got.contains("lines 2-2 of 2"));
    }

    #[tokio::test]
    async fn a_pdf_names_itself_and_the_call_that_would_work() {
        // The signpost, not a router: file_read never substitutes converted
        // text for the bytes on disk. It says what the file is and stops.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doc.pdf");
        let mut bytes = b"%PDF-1.7\n".to_vec();
        bytes.extend_from_slice(&[0xff, 0xfe, 0x00, 0x01]);
        std::fs::write(&path, bytes).unwrap();

        let err = read(&path, serde_json::json!({})).await.unwrap_err();
        assert!(err.contains("PDF"), "must name the format: {err}");
        assert!(
            err.contains("file_convert"),
            "must name the tool that can: {err}"
        );
        assert!(
            err.contains("to=\"txt\""),
            "must name the concrete call, not gesture at one: {err}"
        );
    }

    #[tokio::test]
    async fn an_unrecognised_binary_says_so_without_guessing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, [0xff, 0xfe, 0xfd, 0xfc, 0x00]).unwrap();

        let err = read(&path, serde_json::json!({})).await.unwrap_err();
        assert!(err.contains("no format"), "{err}");
        assert!(!err.contains("PDF"));
    }

    #[tokio::test]
    async fn a_directory_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let err = read(dir.path(), serde_json::json!({})).await.unwrap_err();
        assert!(err.contains("is a directory"), "{err}");
        assert!(err.contains("file_list"), "must name the right tool: {err}");
    }

    // ---- ANAI-292: file_grep ------------------------------------------------

    async fn grep(path: &Path, args: serde_json::Value) -> Result<String, String> {
        let mut input = args;
        input["path"] = serde_json::json!(path.to_str().unwrap());
        tool_file_grep(&input, None, None, None).await
    }

    #[tokio::test]
    async fn a_hit_reports_a_line_number_that_file_read_can_consume() {
        // The whole design: grep's output unit and file_read's input unit are
        // the same, so a hit pastes straight into a range with no conversion.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        std::fs::write(&path, "alpha\nbravo\nNEEDLE here\ndelta\n").unwrap();

        let got = grep(&path, serde_json::json!({ "pattern": "NEEDLE" }))
            .await
            .unwrap();
        assert!(got.contains("1 match(es)"), "{got}");
        assert!(got.contains("3: NEEDLE here"), "{got}");

        // And the number is right: reading that offset returns that line.
        let read_back = read(&path, serde_json::json!({ "offset": 3, "limit": 1 }))
            .await
            .unwrap();
        assert!(read_back.ends_with("NEEDLE here\n"), "{read_back:?}");
    }

    #[tokio::test]
    async fn context_lines_are_marked_differently_from_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "a\nb\nHIT\nd\ne\n").unwrap();

        let got = grep(&path, serde_json::json!({ "pattern": "HIT", "context": 1 }))
            .await
            .unwrap();
        // ':' for the match, '-' for context: grep's convention, so a context
        // line is never mistaken for a hit.
        assert!(got.contains("2- b"), "{got}");
        assert!(got.contains("3: HIT"), "{got}");
        assert!(got.contains("4- d"), "{got}");
        // context = 1, so line 1 is out of the window. Matched on the full
        // rendered form, because the header legitimately contains "1-based".
        assert!(
            !got.contains("1- a"),
            "context is 1, so line 1 is out: {got}"
        );
        assert!(!got.contains("5- e"), "and so is line 5: {got}");
    }

    #[tokio::test]
    async fn the_match_cap_is_disclosed_and_not_reported_as_a_total() {
        // The dangerous version of this says "20 matches" when it stopped
        // looking at 20. That is a fabricated total, and it reads as complete.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("many.txt");
        let body: String = (1..=500).map(|i| format!("hit {i}\n")).collect();
        std::fs::write(&path, body).unwrap();

        let got = grep(
            &path,
            serde_json::json!({ "pattern": "hit", "max_matches": 5 }),
        )
        .await
        .unwrap();
        assert!(got.contains("stopped at the 5-match cap"), "{got}");
        assert!(
            got.contains("NOT a total"),
            "the cap must refuse to be read as a total: {got}"
        );
    }

    #[tokio::test]
    async fn an_over_ceiling_cap_is_clamped_and_the_clamp_is_disclosed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "hit\n").unwrap();

        let got = grep(
            &path,
            serde_json::json!({ "pattern": "hit", "max_matches": 999_999 }),
        )
        .await
        .unwrap();
        assert!(got.contains("you requested max_matches=999999"), "{got}");
        assert!(got.contains("CLAMPED"), "{got}");
    }

    #[tokio::test]
    async fn no_matches_says_so_rather_than_returning_a_bare_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "nothing to see\n").unwrap();

        let got = grep(&path, serde_json::json!({ "pattern": "zzz" }))
            .await
            .unwrap();
        assert!(got.contains("No matches."), "{got}");
        assert!(got.contains("0 match(es)"), "{got}");
    }

    #[tokio::test]
    async fn a_directory_search_recurses_and_include_narrows_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn needle() {}\n").unwrap();
        std::fs::write(dir.path().join("sub/b.rs"), "// needle\n").unwrap();
        std::fs::write(dir.path().join("c.txt"), "needle in txt\n").unwrap();

        let all = grep(dir.path(), serde_json::json!({ "pattern": "needle" }))
            .await
            .unwrap();
        assert!(all.contains("3 match(es)"), "{all}");

        let only_rs = grep(
            dir.path(),
            serde_json::json!({ "pattern": "needle", "include": "*.rs" }),
        )
        .await
        .unwrap();
        assert!(only_rs.contains("2 match(es)"), "{only_rs}");
        assert!(!only_rs.contains("c.txt"), "{only_rs}");
    }

    #[tokio::test]
    async fn skipped_build_directories_are_disclosed_not_silently_dropped() {
        // A skip nobody is told about is indistinguishable from an absence of
        // matches, which is the whole failure class.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/gen.rs"), "needle\n").unwrap();
        std::fs::write(dir.path().join("real.rs"), "needle\n").unwrap();

        let got = grep(dir.path(), serde_json::json!({ "pattern": "needle" }))
            .await
            .unwrap();
        assert!(got.contains("1 match(es)"), "{got}");
        assert!(
            got.contains("did not descend into target"),
            "the skip must be stated: {got}"
        );
    }

    #[tokio::test]
    async fn a_binary_file_is_skipped_whole_and_counted() {
        // Not "skip the lines that fail to decode" — a half-decoded binary
        // yields plausible garbage.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("t.txt"), "needle\n").unwrap();
        std::fs::write(
            dir.path().join("b.bin"),
            [0xff, 0xfe, b'n', b'e', b'e', b'd', b'l', b'e', 0x00],
        )
        .unwrap();

        let got = grep(dir.path(), serde_json::json!({ "pattern": "needle" }))
            .await
            .unwrap();
        assert!(got.contains("1 match(es)"), "{got}");
        assert!(got.contains("skipped 1 non-text file"), "{got}");
    }

    #[tokio::test]
    async fn ignore_case_is_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "Needle\n").unwrap();

        let sensitive = grep(&path, serde_json::json!({ "pattern": "needle" }))
            .await
            .unwrap();
        assert!(sensitive.contains("0 match(es)"), "{sensitive}");

        let insensitive = grep(
            &path,
            serde_json::json!({ "pattern": "needle", "ignore_case": true }),
        )
        .await
        .unwrap();
        assert!(insensitive.contains("1 match(es)"), "{insensitive}");
        assert!(insensitive.contains("case-insensitive"), "{insensitive}");
    }

    #[tokio::test]
    async fn a_bad_pattern_is_refused_by_name_not_treated_as_a_literal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "x\n").unwrap();

        let err = grep(&path, serde_json::json!({ "pattern": "a(" }))
            .await
            .unwrap_err();
        assert!(err.contains("not a valid regular expression"), "{err}");

        // An empty pattern matches every line, which is never what was meant.
        let err = grep(&path, serde_json::json!({ "pattern": "" }))
            .await
            .unwrap_err();
        assert!(err.contains("matches every line"), "{err}");
    }

    #[tokio::test]
    async fn an_enormous_match_line_is_clipped_with_a_marker() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("min.js");
        let long = format!("var needle={};\n", "a".repeat(5000));
        std::fs::write(&path, long).unwrap();

        let got = grep(&path, serde_json::json!({ "pattern": "needle" }))
            .await
            .unwrap();
        assert!(got.contains("line clipped"), "{got}");
        assert!(got.len() < 3000, "one minified line must not be the result");
    }

    #[tokio::test]
    async fn a_directory_symlink_is_not_followed_out_of_the_tree() {
        // Following one walks straight out of the resolved tier, and can loop.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "needle\n").unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "nothing\n").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();

        let got = grep(dir.path(), serde_json::json!({ "pattern": "needle" }))
            .await
            .unwrap();
        assert!(got.contains("0 match(es)"), "symlink was followed: {got}");
    }

    #[tokio::test]
    async fn an_oversized_slice_is_capped_and_says_where_to_resume() {
        // A caller who asks for more than the slice ceiling gets a marked
        // truncation here, with a resume point — rather than travelling on to
        // be clamped anonymously at the bridge frame.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.txt");
        let filler = "z".repeat(1000);
        let body: String = (1..=600).map(|i| format!("{i} {filler}\n")).collect();
        std::fs::write(&path, body).unwrap();

        let got = read(&path, serde_json::json!({ "offset": 1, "limit": 600 }))
            .await
            .unwrap();
        assert!(
            got.contains("byte ceiling"),
            "the cap must announce itself: {}",
            &got[..400.min(got.len())]
        );
        assert!(got.contains("Continue with offset"));
        assert!(
            got.contains("of 600"),
            "the denominator stays true even when the window is short"
        );
    }

    // ANAI-110: lineage threading via the WAKE_LINEAGE task-local. These prove
    // `resolve_wake_base_lineage` reads the scoped inbound chain (the real read
    // path a woken turn takes) and falls back to root-at-sender otherwise.

    #[tokio::test]
    async fn wake_lineage_absent_roots_at_sender() {
        // Origin turn: no task-local scoped. Base = [sender]; only self-wake
        // is a cycle — identical to v1 behavior for channel / cron / API turns.
        let base = resolve_wake_base_lineage("agent-a");
        assert_eq!(base.as_slice(), &["agent-a"]);
        assert!(
            base.would_cycle("agent-a"),
            "self-wake must still be a cycle"
        );
        assert!(!base.would_cycle("agent-b"));
    }

    #[tokio::test]
    async fn wake_lineage_present_extends_real_inbound_chain() {
        // Woken turn A -> B: run_woken_agent_loop scoped the inbound chain
        // [a, b] whose `current` (b) is this agent, which is also `sender`.
        let inbound = openfang_types::wake::WakeLineage::from_agents(vec!["a".into(), "b".into()]);
        WAKE_LINEAGE
            .scope(inbound, async {
                let base = resolve_wake_base_lineage("b");
                // The REAL chain is used, NOT re-rooted at the sender.
                assert_eq!(base.as_slice(), &["a", "b"]);
                // Cross-hop cycle now caught: B waking A (A -> B -> A) is refused.
                assert!(base.would_cycle("a"), "cross-hop A->B->A must be a cycle");
                assert!(base.would_cycle("b"), "self-wake must be a cycle");
                // A genuine onward hop to a fresh agent is still allowed.
                assert!(!base.would_cycle("c"));
            })
            .await;
    }

    #[tokio::test]
    async fn wake_lineage_present_enforces_depth_across_hops() {
        // A four-deep inbound chain: extending it by one more hop reaches the
        // bound, so the producer's `exceeds_depth` check refuses the wake —
        // depth now accrues across agents instead of resetting to 1 each hop.
        let inbound = openfang_types::wake::WakeLineage::from_agents(
            vec!["a", "b", "c", "d"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        WAKE_LINEAGE
            .scope(inbound, async {
                let base = resolve_wake_base_lineage("d");
                assert_eq!(base.depth(), 4);
                let next = base.extended("e");
                assert!(
                    next.exceeds_depth(openfang_types::wake::DEFAULT_MAX_WAKE_DEPTH),
                    "5th hop must trip the depth bound"
                );
            })
            .await;
    }

    // ANAI-122: the one-shot reply-right. These prove the whole authority model
    // for `agent_reply_async` through the KERNEL-HANDLE registry (not the old
    // `WAKE_REPLY_RIGHT` task-local, which the process/IPC boundary severed for
    // subprocess agents): the tool is inert unless the kernel minted a right for
    // this agent, the right is consumed on first use (one-shot), and a turn with
    // no minted right — origin, already-used, or reply-woken/terminal — refuses.
    // Crucially these drive the tool with NO task-local set, i.e. the exact
    // out-of-process path all three prior smokes slipped through.

    #[test]
    fn reply_right_token_names_exactly_one_target() {
        // The token fixes the lawful reply target (the initiator) and carries
        // the inbound correlation id — the tool never chooses a target itself.
        let right = ReplyRight::new(
            "origin-agent",
            "wake-task-42",
            Some("discord:1086446153098342510".to_string()),
        );
        // ANAI-123: the surfacing route rides in the token so the reply inherits
        // it without the callee ever choosing a target/route.
        assert_eq!(right.surface_to(), Some("discord:1086446153098342510"));
        assert_eq!(right.reply_to(), "origin-agent");
        assert_eq!(right.correlation(), "wake-task-42");
    }

    #[tokio::test]
    async fn reply_right_consumed_one_shot_via_kernel_handle() {
        // THE regression that closes the gap all three smokes slipped through:
        // drive the real `tool_agent_reply_async` with NO task-local set —
        // exactly the out-of-process bridge-IPC handler task — and prove the
        // right is served from the kernel-handle registry, then consumed
        // one-shot.
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> =
            Arc::new(FakeKernelHandle::new().with_reply_right(
                "callee-agent",
                ReplyRight::new("origin-agent", "corr-1", None),
            ));
        let input = serde_json::json!({ "message": "here is my answer" });

        // ANAI-200: the reply path no longer touches the process-global emit
        // window at all, so this test needs neither the shared guard nor a
        // window drain — it cannot pollute the ceiling test, nor be polluted by
        // it. (Pre-ANAI-200 it stamped the window and had to serialize.)
        let first = tool_agent_reply_async(&input, Some(&handle), Some("callee-agent")).await;
        first.expect("a seeded right must produce a queued reply");

        // Second call, same turn: the right was consumed on first read, so the
        // registry is empty and the tool refuses. This is the one-shot property,
        // now enforced by the kernel handle rather than a task-local `Cell`.
        let second = tool_agent_reply_async(&input, Some(&handle), Some("callee-agent")).await;
        let err = second.expect_err("second reply in the same turn must be refused");
        assert!(
            err.contains("reply-right"),
            "second call must refuse at the reply-right gate; got: {err}"
        );
    }

    // The guard is held for the whole test on purpose: it exists to serialize
    // tests that mutate the process-global wake-emit window, so releasing it
    // before the first await is exactly the thing it is there to prevent.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn explicit_reply_survives_an_exhausted_aggregate_ceiling() {
        // ANAI-200 regression. The aggregate ceiling used to be checked AFTER
        // `take_reply_right` consumed the one-shot token: a ceiling trip ate the
        // debt and emitted nothing, so the kernel's turn-end sweep saw the
        // correlation as already settled and never auto-closed it (ANAI-198).
        // Net effect: under fleet-wide load — the one condition where the reply
        // guarantee earns its keep — a real answer became permanent silence.
        //
        // A reply cannot amplify (1:1 with an already-admitted wake, terminal by
        // `is_reply`), so the correct fix was to drop the gate, not reorder it.
        // Saturate the window and assert the reply still goes out.
        let _guard = WAKE_EMIT_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset_wake_emit_window();
        for _ in 0..openfang_types::agent_wake::emit_max() {
            assert!(wake_emit_admit(), "priming the window must admit");
        }
        assert!(
            !wake_emit_admit(),
            "precondition: the aggregate ceiling must now be refusing"
        );

        let handle: Arc<dyn crate::kernel_handle::KernelHandle> =
            Arc::new(FakeKernelHandle::new().with_reply_right(
                "callee-agent",
                ReplyRight::new("origin-agent", "corr-ceiling", None),
            ));
        let input = serde_json::json!({ "message": "answer under load" });

        let out = tool_agent_reply_async(&input, Some(&handle), Some("callee-agent"))
            .await
            .expect("a reply must not be refused by the aggregate ceiling");
        assert!(
            out.contains("corr-ceiling"),
            "the queued reply must name its correlation; got: {out}"
        );
    }

    // ---- ANAI-210: `requires_tools` pre-flight on agent_send_async ---------
    //
    // Failure class B: a wake the target structurally cannot serve. The reply
    // guarantee bounds the wait but cannot make the answer useful, so the whole
    // point of these tests is that the refusal happens BEFORE anything durable
    // exists — no correlation, no queue row, no deadline consumed.

    #[tokio::test]
    async fn requires_tools_refuses_a_target_missing_the_tool() {
        // The observed 2026-08-23 case, in miniature: `sleep 60` addressed to an
        // agent with no `shell_exec`. Without the pre-flight the sender burns its
        // entire deadline to learn a static fact about the target's manifest.
        let fake = Arc::new(FakeKernelHandle::new().with_agent("no-shell", &["file_read"]));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({
            "agent_id": "no-shell",
            "message": "run `sleep 60` and tell me when it finishes",
            "requires_tools": ["shell_exec"],
        });

        let err = tool_agent_send_async(&input, Some(&handle), Some("origin-agent"))
            .await
            .expect_err("a target missing a required tool must be refused");

        assert!(
            err.contains("shell_exec"),
            "the refusal must NAME the missing tool so the caller can correct it; got: {err}"
        );
        assert!(
            err.contains("NOT sent"),
            "the refusal must say nothing was sent, not merely that it failed; got: {err}"
        );
        // The property that makes a corrected re-send safe: no queue row exists.
        assert!(
            fake.wake_posts.lock().unwrap().is_empty(),
            "a pre-flight refusal must enqueue no wake at all"
        );
    }

    // Reaches the process-global emit window, so it must serialize against the
    // tests that deliberately saturate it — otherwise a satisfied pre-flight
    // fails on the CEILING error and reads as a pre-flight bug. Held across the
    // awaits on purpose: that is what the guard is for.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn requires_tools_proceeds_when_the_target_has_them_all() {
        let _guard = WAKE_EMIT_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset_wake_emit_window();
        // The check must be a filter, not a wall: a satisfied requirement is
        // invisible and the send behaves exactly as it did before ANAI-210.
        let fake =
            Arc::new(FakeKernelHandle::new().with_agent("has-shell", &["shell_exec", "file_read"]));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({
            "agent_id": "has-shell",
            "message": "run the smoke",
            "requires_tools": ["shell_exec"],
        });

        let out = tool_agent_send_async(&input, Some(&handle), Some("origin-has"))
            .await
            .expect("a target holding every required tool must not be refused");
        assert!(
            out.contains("Async wake queued"),
            "the satisfied path must reach the ordinary enqueue; got: {out}"
        );
        assert_eq!(
            fake.wake_posts.lock().unwrap().len(),
            1,
            "exactly one wake must be enqueued on the satisfied path"
        );
    }

    // Reaches the process-global emit window, so it must serialize against the
    // tests that deliberately saturate it — otherwise a satisfied pre-flight
    // fails on the CEILING error and reads as a pre-flight bug. Held across the
    // awaits on purpose: that is what the guard is for.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn requires_tools_fails_open_when_the_tool_set_is_unknown() {
        let _guard = WAKE_EMIT_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset_wake_emit_window();
        // `agent_tool_names` -> None means "cannot determine", NOT "no tools".
        // Reading it as an empty set would make every mock handle — and every
        // agent the registry cannot resolve tools for — refuse every send. A
        // pre-flight that invents refusals is strictly worse than one that
        // occasionally lets a doomed send through to the deadline it would have
        // hit anyway, so the unknown case degrades to pre-ANAI-210 behaviour.
        let fake = Arc::new(FakeKernelHandle::new().with_opaque_agent("opaque"));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({
            "agent_id": "opaque",
            "message": "do the thing",
            "requires_tools": ["shell_exec"],
        });

        let out = tool_agent_send_async(&input, Some(&handle), Some("origin-opaque"))
            .await
            .expect("an undeterminable tool set must fail OPEN, not refuse");
        assert!(
            out.contains("Async wake queued"),
            "fail-open must reach the ordinary enqueue; got: {out}"
        );
    }

    // Reaches the process-global emit window, so it must serialize against the
    // tests that deliberately saturate it — otherwise a satisfied pre-flight
    // fails on the CEILING error and reads as a pre-flight bug. Held across the
    // awaits on purpose: that is what the guard is for.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn an_absent_or_blank_requires_tools_is_a_no_op() {
        let _guard = WAKE_EMIT_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset_wake_emit_window();
        // Backward compatibility, stated as a test: every pre-ANAI-210 caller
        // omits the field, and must be unaffected even against a target with an
        // EMPTY effective tool set — the input that would trip a naive check.
        // Whitespace-only entries are filtered for the same reason: a caller
        // that passes `[""]` meant "no requirement", not "require the tool
        // named empty string", and refusing there would be a pure regression.
        let fake = Arc::new(FakeKernelHandle::new().with_agent("toolless", &[]));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();

        for requires in [
            serde_json::json!(null),
            serde_json::json!([]),
            serde_json::json!(["   ", ""]),
        ] {
            let input = serde_json::json!({
                "agent_id": "toolless",
                "message": "legacy caller",
                "requires_tools": requires,
            });
            let out = tool_agent_send_async(&input, Some(&handle), Some("origin-legacy"))
                .await
                .expect("an absent or blank requirement must never refuse");
            assert!(
                out.contains("Async wake queued"),
                "no-op path must enqueue as before; got: {out}"
            );
        }
    }

    #[tokio::test]
    async fn the_preflight_names_every_missing_tool_not_just_the_first() {
        // A caller that has to re-send once per missing tool learns the target's
        // shape one deadline at a time. Report the whole gap in one refusal.
        let fake = Arc::new(FakeKernelHandle::new().with_agent("partial", &["shell_exec"]));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({
            "agent_id": "partial",
            "message": "read a file and browse",
            "requires_tools": ["shell_exec", "file_read", "web_fetch"],
        });

        let err = tool_agent_send_async(&input, Some(&handle), Some("origin-partial"))
            .await
            .expect_err("a partially-equipped target must still be refused");
        assert!(
            err.contains("file_read") && err.contains("web_fetch"),
            "both missing tools must be named; got: {err}"
        );
        assert!(
            !err.contains("shell_exec"),
            "the tool the target DOES have must not appear in the missing list; got: {err}"
        );
    }

    // Held for the whole test on purpose — it serializes the process-global
    // wake-emit window, so releasing it before the first await defeats it.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn the_preflight_refuses_ahead_of_the_aggregate_ceiling() {
        // Ordering is the feature. The refusal sits ahead of the surface
        // default, lineage/cycle, depth, BOTH wake budgets and `wake_post`, so a
        // doomed send costs the fleet's emission budget nothing. Saturating the
        // aggregate ceiling and still getting the TOOLS error — not the ceiling
        // error — is the observable proof that the check runs first.
        let _guard = WAKE_EMIT_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset_wake_emit_window();
        for _ in 0..openfang_types::agent_wake::emit_max() {
            assert!(wake_emit_admit(), "priming the window must admit");
        }
        assert!(
            !wake_emit_admit(),
            "precondition: the aggregate ceiling must now be refusing"
        );

        let fake = Arc::new(FakeKernelHandle::new().with_agent("no-shell-2", &["file_read"]));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({
            "agent_id": "no-shell-2",
            "message": "run `sleep 60`",
            "requires_tools": ["shell_exec"],
        });

        let err = tool_agent_send_async(&input, Some(&handle), Some("origin-ceiling-2"))
            .await
            .expect_err("the send must be refused");
        assert!(
            err.contains("shell_exec"),
            "the pre-flight must win the race against the ceiling; got: {err}"
        );
        assert!(
            !err.contains("ceiling"),
            "reaching the ceiling check means the pre-flight ran too late; got: {err}"
        );
        reset_wake_emit_window();
    }

    #[tokio::test]
    async fn reply_right_absent_without_a_minted_right() {
        // No right in the registry (origin / channel / cron / API turn, OR a
        // reply-woken terminal turn — the kernel mints nothing for either). The
        // kernel-handle lookup returns `None`, so the tool is INERT and refuses.
        // This is the default-safe property that lets it live in DEFAULT_ALLOWED
        // without a manifest grant.
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let input = serde_json::json!({ "message": "no right here" });
        let result = tool_agent_reply_async(&input, Some(&handle), Some("some-agent")).await;
        let err = result.expect_err("without a minted right the tool must refuse");
        assert!(
            err.contains("reply-right"),
            "refusal must name the missing reply-right; got: {err}"
        );
    }

    #[test]
    fn wake_emit_admit_ceiling_trips_then_holds() {
        // The first `emit_max` emissions in a single window are admitted; the
        // next is refused. All calls land inside the trailing window, so none
        // age out — this exercises the ceiling, not eviction. (Uses the
        // process-global window.) ANAI-122: this static is shared with any test
        // that drives the real emit path, so serialize behind the guard and
        // drain the window first — otherwise a parallel `tool_agent_reply_async`
        // emit steals a slot and the `0..cap` loop trips one short.
        let _guard = WAKE_EMIT_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset_wake_emit_window();
        let cap = openfang_types::agent_wake::emit_max();
        for i in 0..cap {
            assert!(wake_emit_admit(), "emission {i} should be admitted");
        }
        assert!(
            !wake_emit_admit(),
            "emission past the aggregate ceiling must be refused"
        );
    }

    #[test]
    fn wake_tree_admit_budget_trips_per_root() {
        // ANAI-111: a single tree (one lineage root) fanning out is capped at
        // `tree_budget_max` per window, independent of any other root. Uses a
        // unique root so parallel tests can't cross-contaminate the per-root map.
        let budget = openfang_types::agent_wake::tree_budget_max();
        let root = "root-fanout-tree-under-test";
        for i in 0..budget {
            assert!(
                wake_tree_admit(root),
                "tree emission {i} should be admitted"
            );
        }
        // Over budget for this root -> refused...
        assert!(
            !wake_tree_admit(root),
            "emission past the per-tree budget must be refused"
        );
        // ...while a DIFFERENT root is unaffected (the budget is per-tree, not
        // global): its first emission still admits.
        assert!(
            wake_tree_admit("a-different-root-entirely"),
            "a distinct tree must have its own independent budget"
        );
    }

    #[test]
    fn extract_cache_binary_basic() {
        assert_eq!(extract_cache_binary("grep -r foo ."), Some("grep".into()));
        assert_eq!(extract_cache_binary("rm -rf /tmp/x"), Some("rm".into()));
        assert_eq!(extract_cache_binary("/bin/ls"), Some("/bin/ls".into()));
    }

    #[test]
    fn extract_cache_binary_skips_leading_env_assignments() {
        assert_eq!(
            extract_cache_binary("RUST_LOG=debug cargo build"),
            Some("cargo".into())
        );
        assert_eq!(
            extract_cache_binary("A=1 B=2 ./run.sh"),
            Some("./run.sh".into())
        );
    }

    #[test]
    fn extract_cache_binary_edge_cases() {
        assert_eq!(extract_cache_binary(""), None);
        assert_eq!(extract_cache_binary("   "), None);
        // all-env produces no binary token
        assert_eq!(extract_cache_binary("FOO=bar"), None);
        // a token whose pre-'=' part is not a valid var name is treated as argv0
        assert_eq!(extract_cache_binary("1bad=x cmd"), Some("1bad=x".into()));
    }

    #[test]
    fn test_builtin_tool_definitions() {
        let tools = builtin_tool_definitions();
        assert!(
            tools.len() >= 39,
            "Expected at least 39 tools, got {}",
            tools.len()
        );
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        // Original 12
        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"file_write"));
        assert!(names.contains(&"file_list"));
        assert!(names.contains(&"create_directory"));
        assert!(names.contains(&"shell_exec"));
        assert!(names.contains(&"agent_send"));
        assert!(names.contains(&"agent_send_async"));
        assert!(names.contains(&"agent_spawn"));
        assert!(names.contains(&"agent_list"));
        assert!(names.contains(&"agent_kill"));
        // Issue #890 — wake up inactive agents
        assert!(names.contains(&"agent_activate"));
        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_recall"));
        // 6 collaboration tools
        assert!(names.contains(&"agent_find"));
        assert!(names.contains(&"task_post"));
        assert!(names.contains(&"task_claim"));
        assert!(names.contains(&"task_complete"));
        assert!(names.contains(&"task_list"));
        assert!(names.contains(&"event_publish"));
        // 5 new Phase 3 tools
        assert!(names.contains(&"schedule_create"));
        assert!(names.contains(&"schedule_list"));
        assert!(names.contains(&"schedule_delete"));
        assert!(names.contains(&"image_analyze"));
        assert!(names.contains(&"location_get"));
        assert!(names.contains(&"system_time"));
        // 6 browser tools
        assert!(names.contains(&"browser_navigate"));
        assert!(names.contains(&"browser_click"));
        assert!(names.contains(&"browser_type"));
        assert!(names.contains(&"browser_screenshot"));
        assert!(names.contains(&"browser_read_page"));
        assert!(names.contains(&"browser_close"));
        assert!(names.contains(&"browser_scroll"));
        assert!(names.contains(&"browser_wait"));
        assert!(names.contains(&"browser_run_js"));
        assert!(names.contains(&"browser_back"));
        // 3 media/image generation tools
        assert!(names.contains(&"media_describe"));
        assert!(names.contains(&"media_transcribe"));
        assert!(names.contains(&"image_generate"));
        // 3 cron tools
        assert!(names.contains(&"cron_create"));
        assert!(names.contains(&"cron_list"));
        assert!(names.contains(&"cron_cancel"));
        // 1 channel send tool
        assert!(names.contains(&"channel_send"));
        // 4 hand tools
        assert!(names.contains(&"hand_list"));
        assert!(names.contains(&"hand_activate"));
        assert!(names.contains(&"hand_status"));
        assert!(names.contains(&"hand_deactivate"));
        // 3 voice/docker tools
        assert!(names.contains(&"text_to_speech"));
        assert!(names.contains(&"speech_to_text"));
        assert!(names.contains(&"docker_exec"));
        // Canvas tool
        assert!(names.contains(&"canvas_present"));
        // 3 skill introspection tools (issue #1038)
        assert!(names.contains(&"skill_list"));
        assert!(names.contains(&"skill_describe"));
        assert!(names.contains(&"skill_execute"));
    }

    /// Issue #1038: skill_list, skill_describe, skill_execute work without
    /// touching the filesystem so global skills (outside the workspace
    /// sandbox) are reachable by the agent.
    #[tokio::test]
    async fn test_skill_tools_no_filesystem_access() {
        use openfang_skills::registry::SkillRegistry;
        use tempfile::TempDir;

        // Build a skills directory containing one prompt-only SKILL.md skill
        // (mirroring the user's daily-journal scenario from #1038).
        let global_dir = TempDir::new().unwrap();
        let skill_dir = global_dir.path().join("daily-journal");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: daily-journal\ndescription: Keep a daily journal\n---\n\
             # Daily Journal\n\nWrite one paragraph per day about what you learned.",
        )
        .unwrap();

        let mut registry = SkillRegistry::new(global_dir.path().to_path_buf());
        registry.load_all().unwrap();
        assert_eq!(registry.count(), 1);

        // skill_list returns the global skill without any filesystem call
        let list_out = tool_skill_list(Some(&registry)).unwrap();
        assert!(list_out.contains("daily-journal"));
        assert!(list_out.contains("Keep a daily journal"));

        // skill_describe returns the SKILL.md body — no file_read needed
        let desc_out = tool_skill_describe(
            &serde_json::json!({ "name": "daily-journal" }),
            Some(&registry),
        )
        .unwrap();
        assert!(desc_out.contains("Daily Journal"));
        assert!(desc_out.contains("Write one paragraph"));

        // skill_execute on a prompt-only skill returns the body in 'prompt_context' mode
        let exec_out = tool_skill_execute(
            &serde_json::json!({ "skill": "daily-journal" }),
            Some(&registry),
        )
        .await
        .unwrap();
        assert!(exec_out.contains("prompt_context"));
        assert!(exec_out.contains("Daily Journal"));

        // skill_describe on a missing skill returns a helpful error
        let missing = tool_skill_describe(
            &serde_json::json!({ "name": "no-such-skill" }),
            Some(&registry),
        );
        assert!(missing.is_err());
        assert!(missing.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_collaboration_tool_schemas() {
        let tools = builtin_tool_definitions();
        let collab_tools = [
            "agent_find",
            "task_post",
            "task_claim",
            "task_complete",
            "task_list",
            "event_publish",
        ];
        for name in &collab_tools {
            let tool = tools
                .iter()
                .find(|t| t.name == *name)
                .unwrap_or_else(|| panic!("Tool '{}' not found", name));
            // Verify each has a valid JSON schema
            assert!(
                tool.input_schema.is_object(),
                "Tool '{}' schema should be an object",
                name
            );
            assert_eq!(
                tool.input_schema["type"], "object",
                "Tool '{}' should have type=object",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_file_read_missing() {
        let bad_path = std::env::temp_dir()
            .join("openfang_test_nonexistent_99999")
            .join("file.txt");
        let result = execute_tool(
            "test-id",
            "file_read",
            &serde_json::json!({"path": bad_path.to_str().unwrap()}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(
            result.is_error,
            "Expected error but got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_file_read_path_traversal_blocked() {
        let result = execute_tool(
            "test-id",
            "file_read",
            &serde_json::json!({"path": "../../etc/passwd"}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("traversal"));
    }

    #[tokio::test]
    async fn test_file_write_path_traversal_blocked() {
        let result = execute_tool(
            "test-id",
            "file_write",
            &serde_json::json!({"path": "../../../tmp/evil.txt", "content": "pwned"}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("traversal"));
    }

    #[tokio::test]
    async fn test_create_directory_path_traversal_blocked() {
        let result = execute_tool(
            "test-id",
            "create_directory",
            &serde_json::json!({"path": "../../etc/evil"}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("traversal"));
    }

    #[tokio::test]
    async fn test_create_directory_creates_nested() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let result = tool_create_directory(
            &serde_json::json!({"path": "a/b/c"}),
            Some(root),
            None,
            None,
        )
        .await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        let expected = root.join("a").join("b").join("c");
        assert!(
            expected.is_dir(),
            "Expected directory to exist: {}",
            expected.display()
        );
    }

    #[tokio::test]
    async fn test_create_directory_idempotent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        // First create
        let r1 = tool_create_directory(
            &serde_json::json!({"path": "data/logs"}),
            Some(root),
            None,
            None,
        )
        .await;
        assert!(r1.is_ok());
        // Second create on existing dir should also succeed
        let r2 = tool_create_directory(
            &serde_json::json!({"path": "data/logs"}),
            Some(root),
            None,
            None,
        )
        .await;
        assert!(r2.is_ok(), "Expected idempotent success, got: {:?}", r2);
    }

    #[tokio::test]
    async fn test_create_directory_f4_denied_tier_blocks() {
        use openfang_types::config::{FileAccessTier, FilePolicy, FileRule};
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let policy = FilePolicy::new(
            true,
            FileAccessTier::Write,
            vec![FileRule {
                path: "secret".to_string(),
                tier: FileAccessTier::Deny,
            }],
        );
        let r = tool_create_directory(
            &serde_json::json!({"path": "secret/sub"}),
            Some(root),
            Some(&policy),
            None,
        )
        .await;
        assert!(r.is_err(), "deny-tier directory must be refused: {:?}", r);
        assert!(!root.join("secret").join("sub").exists());
    }

    #[tokio::test]
    async fn test_create_directory_f4_active_policy_no_root_fails_closed() {
        use openfang_types::config::{FileAccessTier, FilePolicy};
        let policy = FilePolicy::new(true, FileAccessTier::Write, vec![]);
        let r = tool_create_directory(
            &serde_json::json!({"path": "anywhere"}),
            None,
            Some(&policy),
            None,
        )
        .await;
        assert!(
            r.is_err(),
            "active policy with no workspace root must fail closed: {:?}",
            r
        );
        assert!(r.unwrap_err().contains("workspace root"));
    }

    #[tokio::test]
    async fn test_create_directory_missing_path_param() {
        let result = tool_create_directory(&serde_json::json!({}), None, None, None).await;
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("Missing 'path'"), "got: {msg}");
    }

    #[tokio::test]
    async fn test_create_directory_dispatch_via_execute_tool() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let result = execute_tool(
            "test-id",
            "create_directory",
            &serde_json::json!({"path": "nested/folder"}),
            None,                 // kernel
            None,                 // allowed_tools
            None,                 // caller_agent_id
            None,                 // skill_registry
            None,                 // mcp_connections
            None,                 // web_ctx
            None,                 // browser_ctx
            None,                 // allowed_env_vars
            Some(root.as_path()), // workspace_root
            None,                 // media_engine
            None,                 // exec_policy
            None,                 // file_policy
            None,                 // tts_engine
            None,                 // docker_config
            None,                 // process_manager
            None,                 // origin
        )
        .await;
        assert!(
            !result.is_error,
            "Expected success, got: {}",
            result.content
        );
        assert!(root.join("nested").join("folder").is_dir());
    }

    #[tokio::test]
    async fn test_file_list_path_traversal_blocked() {
        let result = execute_tool(
            "test-id",
            "file_list",
            &serde_json::json!({"path": "/foo/../../etc"}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("traversal"));
    }

    #[tokio::test]
    async fn test_web_search() {
        let result = execute_tool(
            "test-id",
            "web_search",
            &serde_json::json!({"query": "rust programming"}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        // web_search now attempts a real fetch; may succeed or fail depending on network
        assert!(!result.tool_use_id.is_empty());
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let result = execute_tool(
            "test-id",
            "nonexistent_tool",
            &serde_json::json!({}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_agent_tools_without_kernel() {
        let result = execute_tool(
            "test-id",
            "agent_list",
            &serde_json::json!({}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("Kernel handle not available"));
    }

    #[tokio::test]
    async fn test_capability_enforcement_denied() {
        let allowed = vec!["file_read".to_string(), "file_list".to_string()];
        let result = execute_tool(
            "test-id",
            "shell_exec",
            &serde_json::json!({"command": "ls"}),
            None,
            Some(&allowed),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("Permission denied"));
    }

    #[tokio::test]
    async fn test_capability_enforcement_allowed() {
        let allowed = vec!["file_read".to_string()];
        // Use a cross-platform nonexistent path
        let bad_path = std::env::temp_dir()
            .join("openfang_test_nonexistent_12345")
            .join("file.txt");
        let result = execute_tool(
            "test-id",
            "file_read",
            &serde_json::json!({"path": bad_path.to_str().unwrap()}),
            None,
            Some(&allowed),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        // Should fail for file-not-found, NOT for permission denied
        assert!(
            result.is_error,
            "Expected error but got: {}",
            result.content
        );
        assert!(
            result.content.contains("Failed to read")
                || result.content.contains("not found")
                || result.content.contains("No such file"),
            "Unexpected error: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_capability_enforcement_aliased_tool_name() {
        // Agent has "file_write" in allowed tools, but LLM calls "fs-write".
        // After normalization, this should pass the capability check.
        let allowed = vec![
            "file_read".to_string(),
            "file_write".to_string(),
            "file_list".to_string(),
            "shell_exec".to_string(),
        ];
        let result = execute_tool(
            "test-id",
            "fs-write", // LLM-hallucinated alias
            &serde_json::json!({"path": "/nonexistent/file.txt", "content": "hello"}),
            None,
            Some(&allowed),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        // Should NOT be the capability-enforcement "Permission denied" — it should
        // normalize to file_write and pass the capability check.  It may still fail
        // for filesystem reasons (e.g. OS "Permission denied (os error 13)"), so we
        // check specifically for the capability-gate message.
        assert!(
            !result.content.contains("Permission denied: agent"),
            "fs-write should normalize to file_write and pass capability check, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_capability_enforcement_aliased_denied() {
        // Agent does NOT have file_write, and LLM calls "fs-write" — should be denied.
        let allowed = vec!["file_read".to_string()];
        let result = execute_tool(
            "test-id",
            "fs-write",
            &serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            None,
            Some(&allowed),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(
            result.content.contains("Permission denied"),
            "fs-write should normalize to file_write which is not in allowed list"
        );
    }

    // --- Schedule parser tests ---
    #[test]
    fn test_parse_schedule_every_minutes() {
        assert_eq!(
            parse_schedule_to_cron("every 5 minutes").unwrap(),
            "*/5 * * * *"
        );
        assert_eq!(
            parse_schedule_to_cron("every 1 minute").unwrap(),
            "* * * * *"
        );
        assert_eq!(parse_schedule_to_cron("every minute").unwrap(), "* * * * *");
        assert_eq!(
            parse_schedule_to_cron("every 30 minutes").unwrap(),
            "*/30 * * * *"
        );
    }

    #[test]
    fn test_parse_schedule_every_hours() {
        assert_eq!(parse_schedule_to_cron("every hour").unwrap(), "0 * * * *");
        assert_eq!(parse_schedule_to_cron("every 1 hour").unwrap(), "0 * * * *");
        assert_eq!(
            parse_schedule_to_cron("every 2 hours").unwrap(),
            "0 */2 * * *"
        );
    }

    #[test]
    fn test_parse_schedule_daily() {
        assert_eq!(parse_schedule_to_cron("daily at 9am").unwrap(), "0 9 * * *");
        assert_eq!(
            parse_schedule_to_cron("daily at 6pm").unwrap(),
            "0 18 * * *"
        );
        assert_eq!(
            parse_schedule_to_cron("daily at 12am").unwrap(),
            "0 0 * * *"
        );
        assert_eq!(
            parse_schedule_to_cron("daily at 12pm").unwrap(),
            "0 12 * * *"
        );
    }

    #[test]
    fn test_parse_schedule_weekdays() {
        assert_eq!(
            parse_schedule_to_cron("weekdays at 9am").unwrap(),
            "0 9 * * 1-5"
        );
        assert_eq!(
            parse_schedule_to_cron("weekends at 10am").unwrap(),
            "0 10 * * 0,6"
        );
    }

    #[test]
    fn test_parse_schedule_shorthand() {
        assert_eq!(parse_schedule_to_cron("hourly").unwrap(), "0 * * * *");
        assert_eq!(parse_schedule_to_cron("daily").unwrap(), "0 0 * * *");
        assert_eq!(parse_schedule_to_cron("weekly").unwrap(), "0 0 * * 0");
        assert_eq!(parse_schedule_to_cron("monthly").unwrap(), "0 0 1 * *");
    }

    #[test]
    fn test_parse_schedule_cron_passthrough() {
        assert_eq!(
            parse_schedule_to_cron("0 */5 * * *").unwrap(),
            "0 */5 * * *"
        );
        assert_eq!(
            parse_schedule_to_cron("30 9 * * 1-5").unwrap(),
            "30 9 * * 1-5"
        );
    }

    #[test]
    fn test_parse_schedule_invalid() {
        assert!(parse_schedule_to_cron("whenever I feel like it").is_err());
        assert!(parse_schedule_to_cron("every 0 minutes").is_err());
    }

    // --- Image format detection tests ---
    #[test]
    fn test_detect_image_format_png() {
        let data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10";
        assert_eq!(detect_image_format(data), "png");
    }

    #[test]
    fn test_detect_image_format_jpeg() {
        let data = b"\xFF\xD8\xFF\xE0\x00\x10JFIF";
        assert_eq!(detect_image_format(data), "jpeg");
    }

    #[test]
    fn test_detect_image_format_gif() {
        let data = b"GIF89a\x10\x00\x10\x00";
        assert_eq!(detect_image_format(data), "gif");
    }

    #[test]
    fn test_detect_image_format_bmp() {
        let data = b"BM\x00\x00\x00\x00";
        assert_eq!(detect_image_format(data), "bmp");
    }

    #[test]
    fn test_detect_image_format_unknown() {
        let data = b"\x00\x00\x00\x00";
        assert_eq!(detect_image_format(data), "unknown");
    }

    #[test]
    fn test_extract_png_dimensions() {
        // Minimal PNG header: signature (8) + IHDR length (4) + "IHDR" (4) + width (4) + height (4)
        let mut data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]; // signature
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D]); // IHDR length
        data.extend_from_slice(b"IHDR"); // chunk type
        data.extend_from_slice(&640u32.to_be_bytes()); // width
        data.extend_from_slice(&480u32.to_be_bytes()); // height
        assert_eq!(extract_image_dimensions(&data, "png"), Some((640, 480)));
    }

    #[test]
    fn test_extract_gif_dimensions() {
        let mut data = b"GIF89a".to_vec();
        data.extend_from_slice(&320u16.to_le_bytes()); // width
        data.extend_from_slice(&240u16.to_le_bytes()); // height
        assert_eq!(extract_image_dimensions(&data, "gif"), Some((320, 240)));
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(500), "500 B");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(2 * 1024 * 1024), "2.0 MB");
    }

    #[tokio::test]
    async fn test_image_analyze_missing_file() {
        let result = execute_tool(
            "test-id",
            "image_analyze",
            &serde_json::json!({"path": "/nonexistent/image.png"}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("Failed to read"));
    }

    #[test]
    fn test_depth_limit_constant() {
        assert_eq!(MAX_AGENT_CALL_DEPTH, 5);
    }

    #[test]
    fn test_depth_limit_first_call_succeeds() {
        // Default depth is 0, which is < MAX_AGENT_CALL_DEPTH
        let default_depth = AGENT_CALL_DEPTH.try_with(|d| d.get()).unwrap_or(0);
        assert!(default_depth < MAX_AGENT_CALL_DEPTH);
    }

    #[test]
    fn test_task_local_compiles() {
        // Verify task_local macro works — just ensure the type exists
        let cell = std::cell::Cell::new(0u32);
        assert_eq!(cell.get(), 0);
    }

    #[tokio::test]
    async fn test_schedule_tools_without_kernel() {
        let result = execute_tool(
            "test-id",
            "schedule_list",
            &serde_json::json!({}),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        )
        .await;
        assert!(result.is_error);
        assert!(result.content.contains("Kernel handle not available"));
    }

    // ─── Canvas / A2UI tests ────────────────────────────────────────

    #[test]
    fn test_sanitize_canvas_basic_html() {
        let html = "<h1>Hello World</h1><p>This is a test.</p>";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), html);
    }

    #[test]
    fn test_sanitize_canvas_rejects_script() {
        let html = "<div><script>alert('xss')</script></div>";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("script"));
    }

    #[test]
    fn test_sanitize_canvas_rejects_iframe() {
        let html = "<iframe src='https://evil.com'></iframe>";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("iframe"));
    }

    #[test]
    fn test_sanitize_canvas_rejects_event_handler() {
        let html = "<div onclick=\"alert('xss')\">click me</div>";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("event handler"));
    }

    #[test]
    fn test_sanitize_canvas_rejects_onload() {
        let html = "<img src='x' onerror = \"alert(1)\">";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_canvas_rejects_javascript_url() {
        let html = "<a href=\"javascript:alert('xss')\">click</a>";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("javascript:"));
    }

    #[test]
    fn test_sanitize_canvas_rejects_data_html() {
        let html = "<a href=\"data:text/html,<script>alert(1)</script>\">x</a>";
        let result = sanitize_canvas_html(html, 512 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_canvas_rejects_empty() {
        let result = sanitize_canvas_html("", 512 * 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Empty"));
    }

    #[test]
    fn test_sanitize_canvas_size_limit() {
        let html = "x".repeat(1024);
        let result = sanitize_canvas_html(&html, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[tokio::test]
    async fn test_canvas_present_tool() {
        let input = serde_json::json!({
            "html": "<h1>Test Canvas</h1><p>Hello world</p>",
            "title": "Test"
        });
        let tmp = std::env::temp_dir().join("openfang_canvas_test");
        let _ = std::fs::create_dir_all(&tmp);
        let result = tool_canvas_present(&input, Some(tmp.as_path())).await;
        assert!(result.is_ok());
        let output: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(output["canvas_id"].is_string());
        assert_eq!(output["title"], "Test");
        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Regression: GitHub issue #919 — rm bypass via process_start ──────
    //
    // Before the fix, an LLM in Allowlist mode could call process_start
    // with command="rm" and args=["/some/file"] to delete files even though
    // "rm" was not in exec_policy.allowed_commands. tool_process_start
    // spawned the subprocess directly without ever consulting exec_policy.
    //
    // These tests pin down the new contract:
    //   1. process_start with a non-allowlisted binary returns Err.
    //   2. The Err message identifies allowlist rejection (so callers and
    //      logs can distinguish it from a generic spawn failure).
    //   3. process_start with an allowlisted binary still works.
    //   4. is_shell_tool() now reports process_start as a shell tool so
    //      the approval-gate path treats it the same as shell_exec.

    #[tokio::test]
    async fn test_issue_919_process_start_rm_blocked_in_allowlist() {
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let pm = crate::process_manager::ProcessManager::new(5);
        let policy = ExecPolicy {
            mode: ExecSecurityMode::Allowlist,
            allowed_commands: vec!["ls".to_string(), "echo".to_string()],
            ..ExecPolicy::default()
        };
        let input = serde_json::json!({
            "command": "rm",
            "args": ["/tmp/openfang_test_should_not_be_deleted.txt"],
        });

        let result = tool_process_start(&input, Some(&pm), Some("test-agent"), Some(&policy)).await;

        assert!(
            result.is_err(),
            "process_start must reject 'rm' when not in allowlist (issue #919). Got: {:?}",
            result
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("not in the exec allowlist"),
            "Error must indicate allowlist rejection, got: {err}"
        );
        assert!(
            err.contains("process_start blocked"),
            "Error must identify process_start as the blocking tool, got: {err}"
        );
        assert_eq!(
            pm.count(),
            0,
            "No process must have been spawned when allowlist rejects the command"
        );
    }

    #[tokio::test]
    async fn test_issue_919_process_start_metachar_in_command_blocked() {
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let pm = crate::process_manager::ProcessManager::new(5);
        let policy = ExecPolicy {
            mode: ExecSecurityMode::Full,
            ..ExecPolicy::default()
        };
        // Even in Full mode, smuggling shell metacharacters into the command
        // field must be rejected — process_start does direct exec, not shell.
        let input = serde_json::json!({
            "command": "rm; cat /etc/passwd",
            "args": [],
        });
        let result = tool_process_start(&input, Some(&pm), Some("test-agent"), Some(&policy)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("metacharacter") || pm.count() == 0);
    }

    #[tokio::test]
    async fn test_issue_919_process_start_metachar_in_arg_blocked() {
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let pm = crate::process_manager::ProcessManager::new(5);
        let policy = ExecPolicy {
            mode: ExecSecurityMode::Allowlist,
            allowed_commands: vec!["echo".to_string()],
            ..ExecPolicy::default()
        };
        // Smuggling a chained command via an argument: echo "$(rm -rf /)"
        let input = serde_json::json!({
            "command": "echo",
            "args": ["$(rm -rf /)"],
        });
        let result = tool_process_start(&input, Some(&pm), Some("test-agent"), Some(&policy)).await;
        assert!(
            result.is_err(),
            "process_start must reject metacharacters in args"
        );
        assert_eq!(pm.count(), 0);
    }

    #[tokio::test]
    async fn test_issue_919_process_start_deny_mode_blocks_everything() {
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let pm = crate::process_manager::ProcessManager::new(5);
        let policy = ExecPolicy {
            mode: ExecSecurityMode::Deny,
            ..ExecPolicy::default()
        };
        let input = serde_json::json!({
            "command": "echo",
            "args": ["hello"],
        });
        let result = tool_process_start(&input, Some(&pm), Some("test-agent"), Some(&policy)).await;
        assert!(result.is_err(), "Deny mode must block process_start");
        assert!(result.unwrap_err().to_lowercase().contains("disabled"));
        assert_eq!(pm.count(), 0);
    }

    #[test]
    fn test_issue_919_is_shell_tool_includes_process_start() {
        // process_start must be treated as a shell tool by the approval gate
        // so #772 (full-mode approval bypass) and #919 (allowlist enforcement)
        // both apply consistently.
        assert!(is_shell_tool("shell_exec"));
        assert!(is_shell_tool("process_start"));
        assert!(!is_shell_tool("file_read"));
        assert!(!is_shell_tool("web_fetch"));
    }

    // ----------------------------------------------------------------------
    // Issue #1069: schedule_* tools route through the kernel cron scheduler
    // ----------------------------------------------------------------------

    #[test]
    fn test_sanitize_schedule_name_strips_punctuation() {
        // Colons, commas, dots, and other punctuation are replaced with '-'.
        let out = sanitize_schedule_name("Remind me: file report, please.");
        assert!(!out.contains(':'));
        assert!(!out.contains(','));
        assert!(!out.contains('.'));
        // Spaces, hyphens, and underscores survive.
        assert!(out
            .chars()
            .all(|c| c.is_alphanumeric() || c == ' ' || c == '-' || c == '_'));
        assert!(!out.is_empty());
    }

    #[test]
    fn test_sanitize_schedule_name_empty_fallback() {
        assert_eq!(sanitize_schedule_name(""), "scheduled-task");
        assert_eq!(sanitize_schedule_name("   "), "scheduled-task");
    }

    #[test]
    fn test_sanitize_schedule_name_caps_length() {
        let long = "a".repeat(500);
        let out = sanitize_schedule_name(&long);
        assert!(out.chars().count() <= 128);
    }

    // Minimal in-memory KernelHandle used to verify schedule_* tool wiring.
    // Records every cron_* call so tests can assert what the tool pushed into
    // the kernel, without booting a real OpenFangKernel.
    struct FakeKernelHandle {
        created: std::sync::Mutex<Vec<(String, serde_json::Value)>>,
        cancelled: std::sync::Mutex<Vec<String>>,
        jobs: std::sync::Mutex<Vec<serde_json::Value>>,
        // Flips true if the approval gate ever called request_approval. Used to
        // prove the allowlist wall short-circuits BEFORE the gate for shell_exec.
        approval_requested: std::sync::atomic::AtomicBool,
        // ANAI-122: kernel-held reply-right registry stand-in. `take_reply_right`
        // consumes from here, mirroring the real kernel's `reply_rights` DashMap,
        // so a test can drive the reply tool through the KERNEL-HANDLE lookup
        // with NO task-local set — the exact out-of-process (bridge-IPC) path the
        // three prior smokes slipped through.
        reply_rights: std::sync::Mutex<std::collections::HashMap<String, ReplyRight>>,
        // Records every `wake_post` (assigned_to target, message) so a test can
        // assert the terminal reply was queued to the right initiator.
        wake_posts: std::sync::Mutex<Vec<(String, String)>>,
        // ANAI-151: records the `command` the gate handed to request_approval,
        // so a test can prove the operator surface receives the VERBATIM
        // command and not the 200-byte serialized-JSON summary.
        approval_command: std::sync::Mutex<Option<String>>,
        // ANAI-188: recorded separately from `approval_summary` on purpose —
        // the whole point of the fix is that these are two channels.
        approval_note: std::sync::Mutex<Option<String>>,
        approval_summary: std::sync::Mutex<Option<String>>,
        // ANAI-153: the decision the gate should return. Defaults to Approved
        // so every pre-existing test keeps its behaviour; a test that wants to
        // exercise the negative outcomes overrides it.
        approval_decision: std::sync::Mutex<openfang_types::approval::ApprovalDecision>,
        // ANAI-154: the verdict the gatekeeper should return. `None` keeps the
        // trait default (Escalate), which is what every pre-existing test wants
        // — the gate must be invisible unless a test asks for it.
        gate_verdict: std::sync::Mutex<Option<openfang_types::gatekeeper::GateVerdict>>,
        // Flips true if `gatekeeper_review` was consulted at all.
        gate_consulted: std::sync::atomic::AtomicBool,
        // ANAI-186: every (agent_id, command, metadata, outcome) handed to the
        // audit sink, so a test can prove one verdict writes exactly one row
        // and that the row carries the VERBATIM command.
        gate_audits: std::sync::Mutex<Vec<(String, String, String, String)>>,
        gate_dispositions: std::sync::Mutex<Vec<(String, String, String, String)>>,
        // ANAI-187: shadow mode. Defaults false, matching the trait default,
        // so every pre-existing test still acts on verdicts.
        gate_shadow: std::sync::atomic::AtomicBool,
        // ANAI-165: a stand-in namespace store, keyed by the VERBATIM key the
        // tool handed down (so `shared:` prefixes are visible to assertions),
        // plus the caller recorded on every call. The real scoping decision is
        // the kernel's; what these prove is that the tool layer threads an
        // identity at all and labels a shared-namespace hit.
        memory: std::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>,
        memory_calls: std::sync::Mutex<Vec<(Option<String>, String)>>,
        // ANAI-194: (caller, reason, title, summary) per episode-close call.
        #[allow(clippy::type_complexity)]
        episode_closes:
            std::sync::Mutex<Vec<(Option<String>, String, Option<String>, Option<String>)>>,
        // The id the next close returns. `None` models "nothing was open".
        open_episode: std::sync::Mutex<Option<String>>,
        // ANAI-264: what a pack primed for the requested slug would resolve,
        // as `(closed episodes, project facts)`. `None` is the trait default
        // — "cannot say" — and is what every pre-existing test sees, so the
        // preview stays invisible unless a test asks for it.
        rehydration_preview: std::sync::Mutex<Option<(usize, usize)>>,
        // ANAI-264: the membership refusal the kernel would return for the
        // requested `prime_for` slug. `None` is the trait default — "no
        // objection" — so every pre-existing close test is untouched.
        membership_error: std::sync::Mutex<Option<String>>,
        status: std::sync::Mutex<serde_json::Value>,
        // ANAI-166: every (caller, query, scope, kind, limit) handed to
        // `memory_search`, and the canned payload it returns. The tool layer's
        // job is to route and to render; the ranking is the kernel's, so what
        // these prove is that the right shape reached the handle.
        #[allow(clippy::type_complexity)]
        searches: std::sync::Mutex<
            Vec<(
                Option<String>,
                String,
                Option<String>,
                Option<String>,
                usize,
            )>,
        >,
        search_result: std::sync::Mutex<serde_json::Value>,
        // (caller, text, tags) per `memory_note`.
        #[allow(clippy::type_complexity)]
        notes: std::sync::Mutex<Vec<(Option<String>, String, Vec<String>)>>,
        // ANAI-204: a stand-in slot table keyed by the RESOLVED address
        // `scope/scope_ref/claim_key`, plus the claims each slot has displaced.
        // The vocabulary and the uniqueness constraint are the store's job and
        // are tested there; what these prove is that the tool layer addresses a
        // slot by subject, threads a caller, and tells created/affirmed/
        // superseded apart in what it renders.
        facts: std::sync::Mutex<std::collections::HashMap<String, String>>,
        fact_history: std::sync::Mutex<std::collections::HashMap<String, Vec<serde_json::Value>>>,
        // ANAI-210: a stand-in registry for the `requires_tools` pre-flight.
        // `agents` feeds target resolution (`list_agents`); `agent_tools` is the
        // EFFECTIVE tool set the kernel would resolve for that agent, which is
        // deliberately a different thing from `AgentInfo::tools` (the raw
        // manifest declaration). An id absent from `agent_tools` models the
        // "cannot determine" answer the pre-flight must fail open on.
        agents: std::sync::Mutex<Vec<crate::kernel_handle::AgentInfo>>,
        agent_tools: std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>,
        // ANAI-246: callers that asked for a deferred context reset, plus a
        // switch to make that request fail, since the tool is required to keep
        // the close committed when it does.
        /// (caller, prime_for) — ANAI-247 records both, so a test can assert
        /// the slug reached the kernel and not merely that a reset did.
        #[allow(clippy::type_complexity)]
        context_resets: std::sync::Mutex<Vec<(Option<String>, Option<String>)>>,
        context_reset_fails: std::sync::atomic::AtomicBool,
        // ANAI-248: pretend the operator has been asked something and has not
        // answered yet.
        pending_operator_question: std::sync::atomic::AtomicBool,
    }

    impl FakeKernelHandle {
        fn new() -> Self {
            Self {
                created: std::sync::Mutex::new(Vec::new()),
                cancelled: std::sync::Mutex::new(Vec::new()),
                jobs: std::sync::Mutex::new(Vec::new()),
                approval_requested: std::sync::atomic::AtomicBool::new(false),
                reply_rights: std::sync::Mutex::new(std::collections::HashMap::new()),
                wake_posts: std::sync::Mutex::new(Vec::new()),
                approval_command: std::sync::Mutex::new(None),
                approval_note: std::sync::Mutex::new(None),
                approval_summary: std::sync::Mutex::new(None),
                approval_decision: std::sync::Mutex::new(
                    openfang_types::approval::ApprovalDecision::Approved,
                ),
                gate_verdict: std::sync::Mutex::new(None),
                gate_consulted: std::sync::atomic::AtomicBool::new(false),
                gate_audits: std::sync::Mutex::new(Vec::new()),
                gate_dispositions: std::sync::Mutex::new(Vec::new()),
                gate_shadow: std::sync::atomic::AtomicBool::new(false),
                memory: std::sync::Mutex::new(std::collections::HashMap::new()),
                memory_calls: std::sync::Mutex::new(Vec::new()),
                episode_closes: std::sync::Mutex::new(Vec::new()),
                open_episode: std::sync::Mutex::new(None),
                rehydration_preview: std::sync::Mutex::new(None),
                membership_error: std::sync::Mutex::new(None),
                status: std::sync::Mutex::new(serde_json::json!({})),
                searches: std::sync::Mutex::new(Vec::new()),
                search_result: std::sync::Mutex::new(
                    serde_json::json!({"mode": "semantic", "count": 0, "results": []}),
                ),
                notes: std::sync::Mutex::new(Vec::new()),
                facts: std::sync::Mutex::new(std::collections::HashMap::new()),
                fact_history: std::sync::Mutex::new(std::collections::HashMap::new()),
                agents: std::sync::Mutex::new(Vec::new()),
                agent_tools: std::sync::Mutex::new(std::collections::HashMap::new()),
                context_resets: std::sync::Mutex::new(Vec::new()),
                context_reset_fails: std::sync::atomic::AtomicBool::new(false),
                pending_operator_question: std::sync::atomic::AtomicBool::new(false),
            }
        }

        // ANAI-210: register a resolvable target whose effective tool set is
        // known. Use `with_opaque_agent` for one whose set is not.
        fn with_agent(self, name: &str, tools: &[&str]) -> Self {
            self.agent_tools.lock().unwrap().insert(
                name.to_string(),
                tools.iter().map(|t| t.to_string()).collect(),
            );
            self.with_opaque_agent(name)
        }

        // ANAI-210: a target that resolves but whose tool set the handle cannot
        // report — the `None` branch the pre-flight fails open on.
        fn with_opaque_agent(self, name: &str) -> Self {
            self.agents
                .lock()
                .unwrap()
                .push(crate::kernel_handle::AgentInfo {
                    id: name.to_string(),
                    name: name.to_string(),
                    state: "Running".into(),
                    model_provider: "test".into(),
                    model_name: "test".into(),
                    description: String::new(),
                    tags: Vec::new(),
                    // Intentionally empty: the raw manifest declaration is NOT
                    // what the pre-flight reads, and leaving it empty here
                    // proves the check consults `agent_tool_names` instead.
                    tools: Vec::new(),
                });
            self
        }

        // ANAI-166: canned search payload, in the kernel's wire shape.
        fn with_search_result(self, result: serde_json::Value) -> Self {
            *self.search_result.lock().unwrap() = result;
            self
        }

        // ANAI-194: pretend an episode is open, so `memory_episode_close`
        // returns an id instead of the nothing-was-open path.
        fn with_open_episode(self, id: &str) -> Self {
            *self.open_episode.lock().unwrap() = Some(id.to_string());
            self
        }

        // ANAI-264: pretend the primed slug resolves this much.
        fn with_rehydration_preview(self, episodes: usize, facts: usize) -> Self {
            *self.rehydration_preview.lock().unwrap() = Some((episodes, facts));
            self
        }

        // ANAI-264: pretend the caller is not a member of the primed project.
        fn with_membership_error(self, refusal: &str) -> Self {
            *self.membership_error.lock().unwrap() = Some(refusal.to_string());
            self
        }

        fn with_status(self, status: serde_json::Value) -> Self {
            *self.status.lock().unwrap() = status;
            self
        }

        // ANAI-154: stand in for a judge that returns a fixed verdict.
        fn with_gate_verdict(self, v: openfang_types::gatekeeper::GateVerdict) -> Self {
            *self.gate_verdict.lock().unwrap() = Some(v);
            self
        }

        // ANAI-187: put the gate in shadow mode.
        fn with_gate_shadow(self) -> Self {
            self.gate_shadow
                .store(true, std::sync::atomic::Ordering::SeqCst);
            self
        }

        // ANAI-153: drive the gate to a specific non-approved outcome.
        fn with_approval_decision(
            self,
            decision: openfang_types::approval::ApprovalDecision,
        ) -> Self {
            *self.approval_decision.lock().unwrap() = decision;
            self
        }

        fn with_job(self, job: serde_json::Value) -> Self {
            self.jobs.lock().unwrap().push(job);
            self
        }

        // Seed a one-shot reply-right for `agent_id`, as the kernel does at
        // wake-dispatch for an origination turn.
        fn with_reply_right(self, agent_id: &str, right: ReplyRight) -> Self {
            self.reply_rights
                .lock()
                .unwrap()
                .insert(agent_id.to_string(), right);
            self
        }
    }

    // -----------------------------------------------------------------------
    // ANAI-165: the memory tools carry the caller's identity
    // -----------------------------------------------------------------------

    #[test]
    fn memory_store_threads_the_caller_identity_down() {
        let fake = Arc::new(FakeKernelHandle::new());
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_store(
            &serde_json::json!({"key": "note", "value": "hi"}),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        assert!(out.contains("your own memory"), "unexpected result: {out}");
        let calls = fake.memory_calls.lock().unwrap();
        assert_eq!(
            calls.as_slice(),
            [(Some("agent-x".to_string()), "note".to_string())],
            "the kernel must receive the caller, not None"
        );
    }

    #[test]
    fn memory_store_labels_a_shared_write_as_shared() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_store(
            &serde_json::json!({"key": "shared:freeze", "value": "on"}),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        assert!(out.contains("SHARED"), "a shared write must say so: {out}");
    }

    #[tokio::test]
    async fn memory_recall_prefers_the_agents_own_namespace() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        tool_memory_store(
            &serde_json::json!({"key": "note", "value": "mine"}),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        tool_memory_store(
            &serde_json::json!({"key": "shared:note", "value": "theirs"}),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        let out = tool_memory_recall(
            &serde_json::json!({"key": "note"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("mine"), "own value must win: {out}");
        assert!(!out.contains("theirs"));
        assert!(!out.contains("SHARED namespace"));
    }

    // --- ANAI-194: episode tools -------------------------------------------

    /// ANAI-252: the title reaches the kernel; the wrap-up does NOT reach the
    /// summary column, because filling it makes the episode invisible to
    /// `awaiting_summary` and the corpus never gets an embedded row.
    #[tokio::test]
    async fn episode_close_routes_the_wrapup_to_a_note_not_the_summary_column() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "git trunk cutover", "summary": "retired the octopus"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("ep-1"), "{out}");
        let calls = fake.episode_closes.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "explicit", "reason defaults to explicit");
        assert_eq!(calls[0].2.as_deref(), Some("git trunk cutover"));
        assert_eq!(
            calls[0].3, None,
            "a filled summary column suppresses consolidation - it must stay null"
        );
        drop(calls);

        let notes = fake.notes.lock().unwrap();
        assert_eq!(notes.len(), 1, "the wrap-up must survive as a note");
        assert_eq!(notes[0].1, "retired the octopus");
        assert_eq!(notes[0].2, vec!["episode-wrapup".to_string()]);
    }

    /// No wrap-up, no note. The close is still a close.
    #[tokio::test]
    async fn episode_close_without_a_wrapup_writes_no_note() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        tool_memory_episode_close(
            &serde_json::json!({"title": "git trunk cutover"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert_eq!(fake.episode_closes.lock().unwrap().len(), 1);
        assert!(fake.notes.lock().unwrap().is_empty());
    }

    // --- ANAI-246: reset_context -------------------------------------------

    /// Absent `reset_context`, nothing is scheduled. The old two-argument
    /// close must keep behaving exactly as it did.
    #[tokio::test]
    async fn episode_close_does_not_reset_context_unless_asked() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "done"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(
            fake.context_resets.lock().unwrap().is_empty(),
            "a plain close must never touch the conversation window"
        );
        assert!(!out.contains("cleared"), "{out}");
    }

    /// `reset_context: true` schedules the reset for the CALLER, and says so.
    #[tokio::test]
    async fn episode_close_with_reset_context_schedules_it_for_the_caller() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "done", "reset_context": true}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        let resets = fake.context_resets.lock().unwrap();
        assert_eq!(resets.len(), 1);
        assert_eq!(resets[0].0.as_deref(), Some("agent-x"));
        assert_eq!(resets[0].1, None, "an unprimed close must carry no slug");
        assert!(out.contains("ep-1"), "{out}");
        assert!(
            out.contains("when this turn ends"),
            "the agent must be told the reset is deferred, not immediate: {out}"
        );
    }

    /// A refused reason must take the reset down with it. Otherwise an agent
    /// could clear its window on a close the kernel rejected.
    #[tokio::test]
    async fn a_refused_close_never_schedules_a_reset() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        tool_memory_episode_close(
            &serde_json::json!({"title": "t", "reason": "timer", "reset_context": true}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap_err();
        assert!(fake.context_resets.lock().unwrap().is_empty());
    }

    /// The close has already committed and the wrap-up note is already
    /// written, so a failed reset must NOT return `Err` — that would tell the
    /// agent to retry a close that succeeded and duplicate the note. It has to
    /// be visible in the text instead.
    #[tokio::test]
    async fn a_failed_reset_keeps_the_close_and_says_so() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        fake.context_reset_fails
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "done", "summary": "wrap", "reset_context": true}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("ep-1"), "the close still happened: {out}");
        assert!(
            out.contains("could not be scheduled") && out.contains("do not repeat it"),
            "a silent failure here is the amnesia this epic exists to prevent: {out}"
        );
        assert_eq!(
            fake.notes.lock().unwrap().len(),
            1,
            "exactly one wrap-up note, never two"
        );
    }

    // --- ANAI-248: self-amputation guard -----------------------------------

    /// The whole point: an agent blocked on an operator answer must not be
    /// able to clear the window that answer belongs to. Refused BEFORE the
    /// wrap-up note and before the close, so nothing partial survives.
    #[tokio::test]
    async fn a_pending_operator_question_refuses_the_reset() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        fake.pending_operator_question
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let err = tool_memory_episode_close(
            &serde_json::json!({
                "title": "done",
                "summary": "wrap",
                "reset_context": true
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap_err();

        assert!(err.contains("Nothing was closed"), "{err}");
        assert!(
            fake.episode_closes.lock().unwrap().is_empty(),
            "the close must not have happened"
        );
        assert!(
            fake.notes.lock().unwrap().is_empty(),
            "the wrap-up note must not have been written either — a refusal \
             that still writes is a half-close"
        );
        assert!(fake.context_resets.lock().unwrap().is_empty());
    }

    /// The guard is scoped to the destructive half. A close without a reset
    /// is bookkeeping — labelling a boundary costs nothing — and refusing it
    /// would be the tool overriding the agent on a harmless call.
    #[tokio::test]
    async fn a_pending_operator_question_still_allows_a_plain_close() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        fake.pending_operator_question
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "done"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("ep-1"), "{out}");
        assert_eq!(fake.episode_closes.lock().unwrap().len(), 1);
    }

    /// Negative control: with nothing outstanding the reset goes through, so
    /// the test above cannot pass for the wrong reason.
    #[tokio::test]
    async fn no_pending_question_leaves_the_reset_alone() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        tool_memory_episode_close(
            &serde_json::json!({"title": "done", "reset_context": true}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert_eq!(fake.context_resets.lock().unwrap().len(), 1);
    }

    // --- ANAI-247: prime_for -----------------------------------------------

    /// The slug must reach the kernel, and the agent must be told what its
    /// fresh window will open with.
    #[tokio::test]
    async fn prime_for_rides_along_with_the_reset() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({
                "title": "epic 240",
                "reset_context": true,
                "prime_for": "openfang-fork"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        let resets = fake.context_resets.lock().unwrap();
        assert_eq!(resets.len(), 1);
        assert_eq!(resets[0].1.as_deref(), Some("openfang-fork"));
        assert!(out.contains("briefing on openfang-fork"), "{out}");
    }

    /// A briefing is what the *fresh* window opens with, so priming without a
    /// reset is a request that cannot be honoured. Refuse it loudly rather
    /// than closing the episode and silently ignoring half the call — the
    /// agent would believe it was primed and never find out otherwise.
    #[tokio::test]
    async fn prime_for_without_reset_context_is_refused_before_the_close() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let err = tool_memory_episode_close(
            &serde_json::json!({"title": "done", "prime_for": "openfang-fork"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap_err();
        assert!(err.contains("reset_context"), "{err}");
        assert!(
            fake.episode_closes.lock().unwrap().is_empty(),
            "refused before the close, so the whole call can simply be retried"
        );
    }

    /// A mistyped slug would prime for a project that does not exist and
    /// produce an empty pack — the agent would conclude its memory was empty
    /// when it was merely misaddressed. Catch it at the edge.
    #[tokio::test]
    async fn a_malformed_slug_is_refused_and_nothing_is_closed() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let err = tool_memory_episode_close(
            &serde_json::json!({
                "title": "done",
                "reset_context": true,
                "prime_for": "OpenFang Fork"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap_err();
        assert!(err.contains("not a usable project slug"), "{err}");
        assert!(fake.episode_closes.lock().unwrap().is_empty());
        assert!(fake.context_resets.lock().unwrap().is_empty());
    }

    // --- ANAI-264: say what the prime resolved ------------------------------

    /// The 2026-08-26 bug: a well-formed slug that names no project passes
    /// validation, renders a pack from episodes alone, and reads as healthy.
    /// The close is the last moment the agent that typed the slug is still
    /// listening, so the zero has to be reported here or nowhere.
    #[tokio::test]
    async fn a_prime_that_resolves_no_facts_says_so() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_open_episode("ep-1")
                .with_rehydration_preview(3, 0),
        );
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({
                "title": "epic 240",
                "reset_context": true,
                "prime_for": "openfang-fork"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        assert!(out.contains("no facts about openfang-fork"), "{out}");
        assert!(out.contains("misaddressed"), "{out}");
        // Advisory, not blocking: the close and the reset still happened.
        assert_eq!(fake.episode_closes.lock().unwrap().len(), 1);
        assert_eq!(fake.context_resets.lock().unwrap().len(), 1);
    }

    /// Negative control: a prime that resolves something reports the counts
    /// and does NOT cry misaddressed, so the test above cannot pass by the
    /// warning being unconditional.
    #[tokio::test]
    async fn a_prime_that_resolves_facts_reports_the_counts() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_open_episode("ep-1")
                .with_rehydration_preview(2, 4),
        );
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({
                "title": "epic 240",
                "reset_context": true,
                "prime_for": "openfang"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        assert!(out.contains("2 closed episode(s)"), "{out}");
        assert!(out.contains("4 live fact(s) about openfang"), "{out}");
        assert!(!out.contains("misaddressed"), "{out}");
    }

    /// "Cannot say" is not "nothing found". A handle that cannot preview must
    /// leave the close text alone rather than report a zero it never
    /// measured — an invented zero would send an agent hunting a typo in a
    /// slug that was correct.
    #[tokio::test]
    async fn an_unavailable_preview_reports_nothing_rather_than_zero() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({
                "title": "epic 240",
                "reset_context": true,
                "prime_for": "openfang"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        assert!(out.contains("briefing on openfang"), "{out}");
        assert!(!out.contains("resolves"), "{out}");
        assert!(!out.contains("misaddressed"), "{out}");
    }

    /// ANAI-264 step 3. A prime for a project the caller is not a member of is
    /// refused BEFORE the close, so the agent can retry the whole call rather
    /// than find itself past a half-drawn boundary: episode closed, window
    /// intact, wrap-up already written.
    #[tokio::test]
    async fn a_prime_outside_declared_membership_is_refused_before_the_close() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_open_episode("ep-1")
                .with_membership_error(
                    "agent 'x' is not a member of project 'tttb' — it declares: openfang.",
                ),
        );
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let err = tool_memory_episode_close(
            &serde_json::json!({
                "title": "epic 240",
                "reset_context": true,
                "prime_for": "tttb"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .expect_err("a non-member prime must be refused");

        // The refusal names what the agent could have meant — the useful half.
        assert!(err.contains("it declares: openfang"), "{err}");
        assert!(err.contains("Nothing was closed"), "{err}");
        // And nothing was: not the close, not the reset.
        assert!(fake.episode_closes.lock().unwrap().is_empty());
        assert!(fake.context_resets.lock().unwrap().is_empty());
    }

    /// Negative control for the guard: with no membership objection the close
    /// proceeds, so the test above cannot pass by the gate being unconditional.
    #[tokio::test]
    async fn a_prime_within_declared_membership_closes_normally() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({
                "title": "epic 240",
                "reset_context": true,
                "prime_for": "openfang.memory"
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        assert!(out.contains("briefing on openfang.memory"), "{out}");
        assert_eq!(
            fake.context_resets.lock().unwrap()[0].1.as_deref(),
            Some("openfang.memory")
        );
    }

    /// An empty or whitespace `prime_for` is an omission, not an error — and
    /// must not trip the reset_context guard.
    #[tokio::test]
    async fn a_blank_prime_for_is_treated_as_absent() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        tool_memory_episode_close(
            &serde_json::json!({"title": "done", "prime_for": "   "}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .expect("a blank slug is an omission, not a refusal");
        assert!(fake.context_resets.lock().unwrap().is_empty());
    }

    /// ANAI-248: the tool description IS the fleet's prompt language for this
    /// decision — it ships in the binary, reaches every agent that has the
    /// tool, and needs no manifest edit. Its three load-bearing claims are
    /// pinned here so a later tidy-up cannot quietly delete the bias against
    /// firing: close on COMPLETION, never mid-task, and when unsure DON'T.
    #[test]
    fn the_close_tool_teaches_the_boundary_discipline() {
        let def = builtin_tool_definitions()
            .into_iter()
            .find(|d| d.name == "memory_episode_close")
            .expect("the tool exists");

        // ANAI-283 rewrites what ANAI-248 pinned here, deliberately and in the
        // open. The old assertions were `"FINISHED"`, `"Never mid-task"` and
        // `"do not close"` + `"idle timeout"` — the bias AGAINST firing, pinned
        // so a later tidy-up could not quietly delete it. It was not deleted;
        // it was moved onto `reset_context`, which is the half that can cost
        // something, and the assertions moved with it.
        assert!(
            def.description.contains("BEFORE you start work on a turn"),
            "the check is placed ahead of the work, not after it: {}",
            def.description
        );
        assert!(
            def.description.contains("moves you to different work"),
            "the trigger is a shift named by the incoming message, not a private \
             sense of completion: {}",
            def.description
        );
        assert!(
            def.description.contains("NOT a topic change"),
            "a topic trigger is far more firing-prone than a completion trigger, \
             so the negative space is part of the doctrine: {}",
            def.description
        );
        assert!(
            def.description.contains("WITHOUT reset_context"),
            "the tie-break survives, reassigned: close anyway, hold the window: {}",
            def.description
        );
        assert!(
            def.description.contains("Never reset mid-task"),
            "the prohibition has to stay explicit, and it is the RESET it now \
             prohibits: {}",
            def.description
        );
        assert!(
            def.description.contains("topic-switch") && def.description.contains("explicit"),
            "the agent is told which reason to name, or the census cannot tell \
             a cue-driven boundary from a completion one: {}",
            def.description
        );

        let reset = def.input_schema["properties"]["reset_context"]["description"]
            .as_str()
            .expect("reset_context is documented");
        assert!(
            reset.contains("the answer is no"),
            "the destructive parameter carries the tie-break too: {reset}"
        );
        assert!(
            reset.contains("Refused"),
            "the agent is told the guard exists rather than discovering it: {reset}"
        );
    }

    /// ANAI-283: the reason vocabulary the schema advertises and the one the
    /// handler accepts are the same list, and neither offers the system's.
    ///
    /// These drifted apart trivially before — the schema is a JSON literal and
    /// the gate is a Rust slice — and the failure mode is an agent picking a
    /// value off the enum and being refused by the tool that offered it.
    #[test]
    fn the_advertised_close_reasons_are_exactly_the_accepted_ones() {
        let def = builtin_tool_definitions()
            .into_iter()
            .find(|d| d.name == "memory_episode_close")
            .expect("the tool exists");
        let advertised: Vec<&str> = def.input_schema["properties"]["reason"]["enum"]
            .as_array()
            .expect("the reason is an enum")
            .iter()
            .map(|v| v.as_str().expect("string variants"))
            .collect();

        assert_eq!(
            advertised, AGENT_CLOSE_REASONS,
            "the schema offers what the handler accepts, in the same order"
        );
        assert!(
            advertised.contains(&"topic-switch"),
            "ADR 0002 §2.6 deferred this until the judgment existed; ANAI-283 is it"
        );
        for system_only in ["timer", "abandoned"] {
            assert!(
                !advertised.contains(&system_only),
                "'{system_only}' stays the system's: an agent claiming it would \
                 date the boundary wrong or report an absence it cannot witness"
            );
            assert!(
                openfang_memory::episode::CloseReason::parse(system_only).is_some(),
                "…but it is still a real reason the DB round-trips — the narrowing \
                 is a caller rule, not a missing variant"
            );
        }
    }

    /// The agent cannot see the episodes table, so asking twice is reasonable.
    /// A hard error would read as "your memory is broken" when nothing is.
    #[tokio::test]
    async fn episode_close_with_nothing_open_is_not_an_error() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "whatever"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("No episode was open"), "{out}");
    }

    /// `timer` is the system's to write and `abandoned` is by definition what
    /// nobody was around to say. Neither may be claimed by an agent, or the
    /// close reason stops being evidence.
    ///
    /// `topic-switch` used to be in this list, deferred by ADR 0002 §2.6 until
    /// the judgment it depends on existed. ANAI-283 is that judgment, so it
    /// moved to the test below.
    #[tokio::test]
    async fn episode_close_refuses_system_and_deferred_reasons() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        for reason in ["timer", "abandoned", "nonsense"] {
            let err = tool_memory_episode_close(
                &serde_json::json!({"title": "t", "reason": reason}),
                Some(&kh),
                Some("agent-x"),
            )
            .await
            .unwrap_err();
            assert!(
                err.contains("not available to agents") || err.contains(reason),
                "{err}"
            );
        }
        assert!(
            fake.episode_closes.lock().unwrap().is_empty(),
            "a refused reason must never reach the kernel"
        );
    }

    /// ANAI-283: the new cue is only measurable if the agent's report of it
    /// reaches the row. An accepted `topic-switch` that quietly persisted as
    /// `explicit` would leave the census unable to tell the doctrine working
    /// from the doctrine ignored — which is the whole reason the reason exists.
    #[tokio::test]
    async fn episode_close_forwards_a_topic_switch_verbatim() {
        let fake = Arc::new(FakeKernelHandle::new().with_open_episode("ep-1"));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_episode_close(
            &serde_json::json!({"title": "ANAI-283 close doctrine", "reason": "topic-switch"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .expect("topic-switch is available to agents now");
        assert!(out.contains("ep-1"), "{out}");

        let calls = fake.episode_closes.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].1, "topic-switch",
            "the reason reaches the kernel as the agent named it"
        );
    }

    #[tokio::test]
    async fn episode_close_requires_a_usable_title() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        for input in [serde_json::json!({}), serde_json::json!({"title": "   "})] {
            assert!(
                tool_memory_episode_close(&input, Some(&kh), Some("agent-x"))
                    .await
                    .is_err()
            );
        }
    }

    #[test]
    fn status_renders_the_open_episode_and_countdown() {
        let fake = Arc::new(FakeKernelHandle::new().with_status(serde_json::json!({
            "episode": {
                "id": "ep-1",
                "opened_at": "2026-08-19T10:00:00+00:00",
                "turn_count": 7,
                "title": null,
                "close_reason": null,
            },
            "idle_minutes": 12,
            "idle_timeout_minutes": 120,
            "minutes_until_timer_close": 108,
            "recent_episodes": [
                {"id": "ep-0", "title": "ADR 0002", "turn_count": 4, "close_reason": "explicit"}
            ],
        })));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake;
        let out = tool_memory_status(Some(&kh), Some("agent-x")).unwrap();
        assert!(out.contains("Open episode ep-1"), "{out}");
        assert!(out.contains("turns captured: 7"), "{out}");
        assert!(out.contains("closes on timer in: 108 min"), "{out}");
        assert!(out.contains("ADR 0002"), "{out}");
    }

    /// No open episode is the normal pre-first-turn state, not a fault.
    #[test]
    fn status_says_plainly_when_nothing_is_open() {
        let fake = Arc::new(FakeKernelHandle::new().with_status(serde_json::json!({
            "episode": null,
            "idle_timeout_minutes": 120,
            "recent_episodes": [],
        })));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake;
        let out = tool_memory_status(Some(&kh), Some("agent-x")).unwrap();
        assert!(out.contains("No episode is open"), "{out}");
    }

    #[test]
    fn status_reports_a_disabled_timer_rather_than_omitting_it() {
        let fake = Arc::new(FakeKernelHandle::new().with_status(serde_json::json!({
            "episode": {"id": "ep-1", "opened_at": "2026-08-19T10:00:00+00:00", "turn_count": 1},
            "idle_timeout_minutes": 0,
            "minutes_until_timer_close": null,
            "recent_episodes": [],
        })));
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake;
        let out = tool_memory_status(Some(&kh), Some("agent-x")).unwrap();
        assert!(out.contains("never (timer disabled)"), "{out}");
    }

    #[test]
    fn episode_tools_are_advertised() {
        let defs = builtin_tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"memory_episode_close"));
        assert!(names.contains(&"memory_status"));
    }

    // --- ANAI-204: tier-3 fact tools ---------------------------------------

    #[test]
    fn fact_tools_are_advertised_with_the_slot_address() {
        let defs = builtin_tool_definitions();
        for name in ["memory_fact", "memory_history"] {
            let def = defs
                .iter()
                .find(|d| d.name == name)
                .unwrap_or_else(|| panic!("{name} must be declared"));
            let props = def
                .input_schema
                .get("properties")
                .and_then(|p| p.as_object())
                .unwrap_or_else(|| panic!("{name} must declare properties"));
            // The slot address is the whole contract. A tool that advertises
            // `key` without `scope`/`scope_ref` invites the caller to address a
            // slot by name alone, which is the pre-amendment mistake in a
            // different costume.
            for param in ["scope", "scope_ref", "key"] {
                assert!(props.contains_key(param), "{name} is missing {param}");
            }
            let required: Vec<&str> = def
                .input_schema
                .get("required")
                .and_then(|r| r.as_array())
                .expect("required must be present")
                .iter()
                .filter_map(|v| v.as_str())
                .collect();
            assert_eq!(
                required,
                vec!["scope", "key"],
                "{name}: scope and key are the minimum addressable slot"
            );
        }
    }

    // --- ANAI-277: the write-time slot neighbourhood -----------------------

    #[test]
    fn a_created_slot_reports_the_addresses_around_it() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "created",
            "scope": "project",
            "scope_ref": "kimiya",
            "claim_key": "project.kimiya.matrix_baseline_recording",
            "existing_slots": [
                "kimiya/project.kimiya.matrix_baseline_state",
                "kimiya/repo.inference_vendor",
            ],
            "existing_slots_total": 2,
            "likely_duplicate_of": "kimiya/project.kimiya.matrix_baseline_state",
        }));
        assert!(out.contains("Created"), "{out}");
        assert!(
            out.contains("kimiya/project.kimiya.matrix_baseline_state"),
            "the near address must be named: {out}"
        );
        assert!(
            out.contains("kimiya/repo.inference_vendor"),
            "the full neighbourhood is shown, not just the ranked guess: {out}"
        );
        assert!(out.contains("NEAR-DUPLICATE"), "{out}");
    }

    /// The guess is lexical and says so. An advisory that presented itself as
    /// a judgement about meaning would be trusted past what it can support,
    /// and the cost of a wrong merge is two claims and their histories.
    #[test]
    fn the_near_duplicate_call_is_marked_as_a_guess() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "created",
            "scope": "agent",
            "scope_ref": "agent-x",
            "claim_key": "repo.trunk_heads",
            "existing_slots": ["repo.trunk_head"],
            "existing_slots_total": 1,
            "likely_duplicate_of": "repo.trunk_head",
        }));
        assert!(
            out.contains("lexical guess, not a judgement about meaning"),
            "{out}"
        );
    }

    // --- ANAI-281: the shadowed ancestor slot ------------------------------

    /// The measured case: `repo.trunk_head` minted at `openfang.memory` while
    /// `openfang` already held it. Decidable, so it is stated rather than
    /// ranked — and it leads, because everything else in the hint is a guess.
    #[test]
    fn a_shadowed_ancestor_slot_is_named_before_the_ranked_list() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "created",
            "scope": "project",
            "scope_ref": "openfang.memory",
            "claim_key": "repo.trunk_head",
            "existing_slots": ["openfang/repo.trunk_head", "openfang/deploy.live_binary"],
            "existing_slots_total": 2,
            "likely_duplicate_of": serde_json::Value::Null,
            "shadows_ancestor_slot": "openfang/repo.trunk_head",
        }));
        assert!(out.contains("SHADOWS AN EXISTING SLOT"), "{out}");
        let shadow = out.find("SHADOWS AN EXISTING SLOT").unwrap();
        let list = out.find("Addresses that already exist").unwrap();
        assert!(shadow < list, "the decidable warning must lead: {out}");
        // Both readings are legitimate and the message says so — a warning
        // that permitted only one would push a correctly narrower claim into
        // the parent's slot.
        assert!(out.contains("rewrite `openfang/repo.trunk_head`"), "{out}");
        assert!(out.contains("this is correct and you can ignore"), "{out}");
    }

    /// A shadow with nothing else around it must still render: the
    /// neighbourhood's emptiness is not evidence that the slot was free.
    #[test]
    fn a_shadow_survives_an_empty_neighbourhood() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "created",
            "scope": "project",
            "scope_ref": "openfang.memory",
            "claim_key": "repo.trunk_head",
            "existing_slots": [],
            "existing_slots_total": 0,
            "shadows_ancestor_slot": "openfang/repo.trunk_head",
        }));
        assert!(out.contains("SHADOWS AN EXISTING SLOT"), "{out}");
        assert!(!out.contains("Addresses that already exist"), "{out}");
    }

    /// A supersession carries neither half, for the ANAI-277 reason: the
    /// address was already chosen.
    #[test]
    fn a_supersession_carries_no_shadow_warning() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "superseded",
            "scope": "project",
            "scope_ref": "openfang.memory",
            "claim_key": "repo.trunk_head",
            "previous_claim": "main @ deadbee",
            "shadows_ancestor_slot": "openfang/repo.trunk_head",
        }));
        assert!(out.contains("Superseded"), "{out}");
        assert!(!out.contains("SHADOWS"), "{out}");
    }

    /// A capped list that read as the whole address space would be worse than
    /// no list: the omitted tail is where an old settled slot hides.
    #[test]
    fn a_truncated_neighbourhood_says_what_it_dropped() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "created",
            "scope": "agent",
            "scope_ref": "agent-x",
            "claim_key": "memory.new",
            "existing_slots": ["a.one", "b.two"],
            "existing_slots_total": 9,
            "likely_duplicate_of": serde_json::Value::Null,
        }));
        assert!(out.contains("(7 more not shown"), "{out}");
        assert!(!out.contains("NEAR-DUPLICATE"), "{out}");
    }

    /// A supersession means the right address was already chosen. Repeating
    /// the neighbourhood there trains the agent to skim the one message that
    /// carries a decision.
    #[test]
    fn a_supersession_carries_no_neighbourhood() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "superseded",
            "scope": "agent",
            "scope_ref": "agent-x",
            "claim_key": "repo.trunk_head",
            "previous_claim": "main @ deadbee",
            "existing_slots": ["memory.other"],
            "existing_slots_total": 1,
        }));
        assert!(out.contains("Superseded"), "{out}");
        assert!(!out.contains("memory.other"), "{out}");
        assert!(!out.contains("Addresses that already exist"), "{out}");
    }

    /// The first slot an agent ever writes has no neighbourhood, and a header
    /// introducing an empty list reads as a malfunction.
    #[test]
    fn an_empty_neighbourhood_is_silent() {
        let out = render_fact_write(&serde_json::json!({
            "outcome": "created",
            "scope": "agent",
            "scope_ref": "agent-x",
            "claim_key": "memory.first",
            "existing_slots": [],
            "existing_slots_total": 0,
        }));
        assert_eq!(
            out,
            "Created agent/agent-x memory.first. This slot was empty; it now holds your claim."
        );
    }

    #[tokio::test]
    async fn fact_write_distinguishes_created_affirmed_and_superseded() {
        // The distinction is the whole reason the handle returns an outcome
        // rather than unit: an agent that renders "affirmed" as "created" is
        // reporting a no-op as news, and one that renders "superseded" the same
        // way has silently changed its mind without saying so.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());

        let created = tool_memory_fact(
            &serde_json::json!({
                "scope": "project",
                "scope_ref": "openfang-fork",
                "key": "repo.trunk_model",
                "claim": "main is the trunk of record",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(created.starts_with("Created"), "{created}");
        assert!(
            created.contains("project/openfang-fork repo.trunk_model"),
            "{created}"
        );

        let affirmed = tool_memory_fact(
            &serde_json::json!({
                "scope": "project",
                "scope_ref": "openfang-fork",
                "key": "repo.trunk_model",
                "claim": "main is the trunk of record",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(affirmed.starts_with("Affirmed"), "{affirmed}");
        assert!(
            affirmed.contains("no history entry"),
            "an affirmation must say it wrote no history: {affirmed}"
        );

        let superseded = tool_memory_fact(
            &serde_json::json!({
                "scope": "project",
                "scope_ref": "openfang-fork",
                "key": "repo.trunk_model",
                "claim": "upstream/main is the trunk of record",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(superseded.starts_with("Superseded"), "{superseded}");
        assert!(
            superseded.contains("main is the trunk of record"),
            "a supersession must name what it displaced: {superseded}"
        );
    }

    /// ANAI-266. An empty child slot with a live ancestor must say so, name the
    /// source, and keep the ancestor's staleness marker — and it must still
    /// warn that writing here forks a new slot rather than replacing the one
    /// just read.
    #[test]
    fn inherited_read_names_its_source() {
        let payload = serde_json::json!({
            "scope": "project",
            "scope_ref": "openfang.memory",
            "claim_key": "repo.trunk_head",
            "fact": serde_json::Value::Null,
            "inherited": {
                "from_scope_ref": "openfang",
                "claim": "main @ 2de3b31",
                "status": "settled",
                "persistence_class": "volatile",
                "age_days": 3.0,
                "should_verify": true,
            },
        });
        let rendered = render_fact_read(&payload);
        assert!(
            rendered.contains("no claim at this exact slot"),
            "the exact answer survives: {rendered}"
        );
        assert!(
            rendered.contains("inherited from project/openfang"),
            "an inherited claim without its source is worse than none: {rendered}"
        );
        assert!(rendered.contains("main @ 2de3b31"), "{rendered}");
        assert!(
            rendered.contains("VERIFY:"),
            "a claim the reader did not address must not be shown with MORE \
             confidence than one they did: {rendered}"
        );
        assert!(
            rendered.contains("CREATE a new slot"),
            "the fork hazard is the reason this is not just an inherited read: {rendered}"
        );

        // Nothing inherited, nothing to say: the old wording stands.
        let bare = render_fact_read(&serde_json::json!({
            "scope": "project",
            "scope_ref": "openfang.memory",
            "claim_key": "repo.trunk_head",
            "fact": serde_json::Value::Null,
            "inherited": serde_json::Value::Null,
        }));
        assert!(
            bare.contains("empty — no claim occupies this slot"),
            "{bare}"
        );
    }

    #[tokio::test]
    async fn fact_read_without_a_claim_says_it_wrote_nothing() {
        // (ANAI-266 renderer cases live in `inherited_read_names_its_source`.)

        // A caller who meant to write and omitted `claim` gets a read. That is
        // the right behaviour — reading before writing is what keeps the key
        // space from growing near-duplicates — but it must not read as success.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());

        let empty = tool_memory_fact(
            &serde_json::json!({"scope": "agent", "key": "agent.working_style"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(empty.contains("empty"), "{empty}");
        assert!(
            empty.contains("nothing was written"),
            "a claimless call must disclose that it wrote nothing: {empty}"
        );

        tool_memory_fact(
            &serde_json::json!({
                "scope": "agent",
                "key": "agent.working_style",
                "claim": "small reviewable diffs",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        let filled = tool_memory_fact(
            &serde_json::json!({"scope": "agent", "key": "agent.working_style"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(filled.contains("small reviewable diffs"), "{filled}");
        assert!(
            filled.contains("nothing was written"),
            "the disclosure must survive a non-empty slot too: {filled}"
        );
    }

    #[tokio::test]
    async fn fact_write_derives_the_agent_slot_ref_from_the_caller() {
        // ANAI-165 from inside tier 3: an `agent`-scoped slot is addressed by
        // the caller's own id, and a supplied ref must not be able to point the
        // write at a sibling's slot space. The store is the enforcement point;
        // this asserts the tool does not invent a ref of its own on the way.
        let fake = Arc::new(FakeKernelHandle::new());
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_fact(
            &serde_json::json!({
                "scope": "agent",
                "key": "agent.working_style",
                "claim": "reads before writing",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("agent/agent-x agent.working_style"), "{out}");
    }

    #[tokio::test]
    async fn fact_write_refuses_a_status_outside_the_vocabulary() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let err = tool_memory_fact(
            &serde_json::json!({
                "scope": "agent",
                "key": "agent.working_style",
                "claim": "anything",
                "status": "superseded",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap_err();
        // Named specifically: `superseded` is the one status a caller is most
        // likely to reach for and the one the design says cannot exist, because
        // a superseded claim has left `memories` entirely.
        assert!(
            err.contains("open, settled"),
            "the rejection must offer the menu: {err}"
        );
    }

    #[tokio::test]
    async fn fact_tools_refuse_a_half_addressed_slot() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        for input in [
            serde_json::json!({"key": "repo.trunk_model"}),
            serde_json::json!({"scope": "project"}),
            serde_json::json!({"scope": "  ", "key": "repo.trunk_model"}),
        ] {
            assert!(
                tool_memory_fact(&input, Some(&kh), Some("agent-x"))
                    .await
                    .is_err(),
                "memory_fact accepted a half-addressed slot: {input}"
            );
            assert!(
                tool_memory_history(&input, Some(&kh), Some("agent-x")).is_err(),
                "memory_history accepted a half-addressed slot: {input}"
            );
        }
    }

    #[tokio::test]
    async fn history_reports_an_empty_trail_without_implying_an_empty_slot() {
        // "No history" and "no fact" are different states and a caller that
        // conflates them concludes the slot is empty when it is merely
        // unchanged. The empty rendering has to say which question it answered.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_history(
            &serde_json::json!({
                "scope": "project",
                "scope_ref": "openfang-fork",
                "key": "repo.trunk_model",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        assert!(out.contains("No superseded versions"), "{out}");
        assert!(
            out.contains("memory_fact"),
            "the empty trail must point at the tool that answers the other question: {out}"
        );
    }

    #[tokio::test]
    async fn history_renders_each_displaced_claim_with_its_author() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        for claim in ["local-main is the trunk", "main is the trunk"] {
            tool_memory_fact(
                &serde_json::json!({
                    "scope": "project",
                    "scope_ref": "openfang-fork",
                    "key": "repo.trunk_model",
                    "claim": claim,
                }),
                Some(&kh),
                Some("agent-x"),
            )
            .await
            .unwrap();
        }

        let out = tool_memory_history(
            &serde_json::json!({
                "scope": "project",
                "scope_ref": "openfang-fork",
                "key": "repo.trunk_model",
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        assert!(out.contains("local-main is the trunk"), "{out}");
        assert!(
            out.contains("asserted by agent-x"),
            "provenance must reach the render, not just the row: {out}"
        );
    }

    #[tokio::test]
    async fn memory_recall_falls_back_to_shared_but_says_so() {
        // Pre-ANAI-165 rows all live in the shared namespace. Recall may serve
        // them, but it must never present another agent's value as this
        // agent's own memory.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        tool_memory_store(
            &serde_json::json!({"key": "shared:legacy", "value": "old"}),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        let out = tool_memory_recall(
            &serde_json::json!({"key": "legacy"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(
            out.contains("old"),
            "legacy value should still be reachable: {out}"
        );
        assert!(
            out.contains("SHARED namespace"),
            "the fallback must be labelled, not silent: {out}"
        );
    }

    #[tokio::test]
    async fn memory_recall_misses_cleanly_when_nothing_exists() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_recall(
            &serde_json::json!({"key": "nope"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("No value found"), "unexpected result: {out}");
    }

    // --- ANAI-166: retrieval + notes ---------------------------------------

    #[tokio::test]
    async fn memory_recall_routes_a_query_to_search_not_the_kv_store() {
        // The whole point of stage 2. Before this, `query` had nowhere to go
        // and the tool answered every question with an exact-key miss.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(
            FakeKernelHandle::new().with_search_result(serde_json::json!({
                "mode": "semantic",
                "count": 1,
                "results": [{
                    "id": "m1",
                    "content": "we cut over to the trunk model on 2026-06-22",
                    "kind": "note",
                    "created_at": "2026-06-22T10:00:00Z",
                }],
            })),
        );

        let out = tool_memory_recall(
            &serde_json::json!({"query": "when did we change the git model", "kind": "note", "limit": 3}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        assert!(out.contains("trunk model"), "hit not rendered: {out}");
        assert!(out.contains("semantic"), "search mode not reported: {out}");

        // The KV path must NOT have been consulted — a query is not a key.
        assert!(
            !out.contains("No value found"),
            "query fell through to exact-key lookup: {out}"
        );
    }

    #[tokio::test]
    async fn memory_recall_threads_caller_scope_kind_and_a_clamped_limit() {
        let fake = Arc::new(FakeKernelHandle::new());
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        tool_memory_recall(
            &serde_json::json!({
                "query": "episodes",
                "scope": "episodic",
                "kind": "note",
                // Absurd on purpose: an unclamped limit is a context-window
                // exhaustion primitive aimed at the caller's own turn.
                "limit": 10_000,
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();

        let calls = fake.searches.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (caller, query, scope, kind, limit) = calls[0].clone();
        assert_eq!(caller.as_deref(), Some("agent-x"));
        assert_eq!(query, "episodes");
        assert_eq!(scope.as_deref(), Some("episodic"));
        assert_eq!(kind.as_deref(), Some("note"));
        assert_eq!(limit, MEMORY_RECALL_MAX_LIMIT, "limit must be clamped");
    }

    #[tokio::test]
    async fn memory_recall_refuses_both_query_and_key() {
        // Enforced in the handler because the JSON schema cannot carry it —
        // `required: []` is all that survives the bridge's re-serialization.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let err = tool_memory_recall(
            &serde_json::json!({"query": "anything", "key": "note"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap_err();
        assert!(err.contains("not both"), "unhelpful error: {err}");
    }

    #[tokio::test]
    async fn memory_recall_refuses_neither_query_nor_key() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let err = tool_memory_recall(&serde_json::json!({}), Some(&kh), Some("agent-x"))
            .await
            .unwrap_err();
        assert!(err.contains("'query'"), "unhelpful error: {err}");
        assert!(err.contains("'key'"), "unhelpful error: {err}");
    }

    #[tokio::test]
    async fn memory_recall_keeps_the_exact_key_path_intact() {
        // Additive-only: the pre-stage-2 shape must behave exactly as before,
        // shared-namespace label included.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        tool_memory_store(
            &serde_json::json!({"key": "shared:legacy", "value": "old"}),
            Some(&kh),
            Some("agent-x"),
        )
        .unwrap();
        let out = tool_memory_recall(
            &serde_json::json!({"key": "legacy"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("old"));
        assert!(out.contains("SHARED namespace"));
    }

    #[tokio::test]
    async fn empty_search_says_when_it_was_only_text_matching() {
        // "Nothing found" from a degraded text search means something very
        // different from "nothing found" from a semantic one, and an agent
        // that cannot tell them apart will conclude its memory is empty.
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(
            FakeKernelHandle::new().with_search_result(serde_json::json!({
                "mode": "text",
                "count": 0,
                "results": [],
            })),
        );
        let out = tool_memory_recall(
            &serde_json::json!({"query": "anything at all"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("No memories found"));
        assert!(
            out.contains("Text matching only"),
            "degraded mode must be disclosed: {out}"
        );
    }

    #[tokio::test]
    async fn memory_note_threads_text_tags_and_caller() {
        let fake = Arc::new(FakeKernelHandle::new());
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_note(
            &serde_json::json!({
                "text": "  the partial unique index is what makes this safe  ",
                "tags": ["episodes", "  ", "ddl"],
            }),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("Noted"));

        let notes = fake.notes.lock().unwrap();
        assert_eq!(notes.len(), 1);
        let (caller, text, tags) = notes[0].clone();
        assert_eq!(caller.as_deref(), Some("agent-x"));
        assert_eq!(text, "the partial unique index is what makes this safe");
        assert_eq!(
            tags,
            vec!["episodes".to_string(), "ddl".to_string()],
            "blank tags must be dropped, not stored"
        );
    }

    #[tokio::test]
    async fn memory_note_refuses_an_empty_note() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        for input in [serde_json::json!({}), serde_json::json!({"text": "   "})] {
            assert!(
                tool_memory_note(&input, Some(&kh), Some("agent-x"))
                    .await
                    .is_err(),
                "empty note should be refused: {input}"
            );
        }
    }

    // --- ANAI-270: note supersession on the tool surface ---------------------

    /// A plain note answers with its short id, so the agent holds the handle
    /// a later correction will need.
    #[tokio::test]
    async fn a_plain_note_reports_its_id() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_note(
            &serde_json::json!({"text": "a thing"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.starts_with("Noted (id:note-1)"), "{out}");
    }

    /// No neighbours: the reply is exactly the plain acknowledgement — no
    /// empty heading, nothing to skim.
    #[test]
    fn a_note_with_no_neighbours_renders_no_hint() {
        let out = render_note_with_neighbours(&serde_json::json!({
            "id": "0f0e0d0c-aaaa", "neighbours": []
        }));
        assert_eq!(
            out,
            "Noted (id:0f0e0d0c). It is attached to your current episode and will surface \
             in memory_recall."
        );
    }

    /// ANAI-270 step 3: neighbours list with id, score, size and preview; a
    /// near-identical one is flagged; the merge recipe names BOTH ids so one
    /// call leaves exactly one current note; and the reply says it is a
    /// guess, not a verdict.
    #[test]
    fn a_note_with_neighbours_lists_them_and_offers_the_merge() {
        let out = render_note_with_neighbours(&serde_json::json!({
            "id": "0f0e0d0c-aaaa",
            "neighbours": [
                {"id": "1a2b3c4d-full", "chars": 812, "score": 0.93, "preview": "step 1 done"},
                {"id": "5e6f7a8b-full", "chars": 400, "score": 0.86, "preview": "mod design"},
            ],
        }));
        assert!(out.starts_with("Noted (id:0f0e0d0c)."), "{out}");
        assert!(
            out.contains("id:1a2b3c4d (0.93, 812 chars) — LIKELY DUPLICATE — \"step 1 done\""),
            "{out}"
        );
        assert!(
            out.contains("id:5e6f7a8b (0.86, 400 chars) — \"mod design\""),
            "{out}"
        );
        assert!(
            !out.contains("5e6f7a8b (0.86, 400 chars) — LIKELY"),
            "{out}"
        );
        assert!(
            out.contains("supersedes: [\"1a2b3c4d\", \"0f0e0d0c\"]"),
            "the merge must retire the new note too: {out}"
        );
        assert!(out.contains("not a verdict"), "{out}");
        assert!(out.contains("still true"), "{out}");
    }

    /// The fake handle lacks the lookup, so the trait default must still
    /// write the note — the hint is optional, the write is not.
    #[tokio::test]
    async fn the_default_neighbour_path_still_writes_the_note() {
        let fake = Arc::new(FakeKernelHandle::new());
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_memory_note(
            &serde_json::json!({"text": "a thing"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(!out.contains("closest"), "{out}");
        assert_eq!(fake.notes.lock().unwrap().len(), 1);
    }

    #[test]
    fn supersedes_takes_a_list_or_a_bare_string_and_drops_blanks() {
        let got = |v: serde_json::Value| note_supersedes(&serde_json::json!({ "supersedes": v }));
        assert_eq!(got(serde_json::json!(["a", " b ", "", 7])), vec!["a", "b"]);
        assert_eq!(got(serde_json::json!("1a2b3c4d")), vec!["1a2b3c4d"]);
        assert!(got(serde_json::json!(null)).is_empty());
        assert!(note_supersedes(&serde_json::json!({})).is_empty());
    }

    /// The reply names what left recall, with both sizes — and a successor
    /// far smaller than its predecessor gets the whole-note reminder.
    #[tokio::test]
    async fn a_superseding_note_reports_sizes_and_flags_a_shrink() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_note(
            &serde_json::json!({"text": "short", "supersedes": ["1a2b3c4d-full-id"]}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(out.contains("Noted (id:0f0e0d0c, 5 chars)"), "{out}");
        assert!(out.contains("supersedes id:1a2b3c4d (1000 chars)"), "{out}");
        assert!(out.contains("no longer surfaces in recall"), "{out}");
        assert!(out.contains("CHECK:"), "a 5-char successor to 1000: {out}");
        assert!(out.contains("still true"), "{out}");
    }

    #[tokio::test]
    async fn a_full_size_successor_is_not_nagged() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let text = "x".repeat(600);
        let out = tool_memory_note(
            &serde_json::json!({"text": text, "supersedes": "1a2b3c4d"}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(
            !out.contains("CHECK:"),
            "600 of 1000 is over the ratio: {out}"
        );
    }

    /// A link lost to a concurrent write is reported, never claimed.
    #[tokio::test]
    async fn a_raced_link_is_reported_not_claimed() {
        let kh: Arc<dyn crate::kernel_handle::KernelHandle> = Arc::new(FakeKernelHandle::new());
        let out = tool_memory_note(
            &serde_json::json!({"text": "x".repeat(2000), "supersedes": ["raced"]}),
            Some(&kh),
            Some("agent-x"),
        )
        .await
        .unwrap();
        assert!(!out.contains("It supersedes"), "{out}");
        assert!(out.contains("Not superseded: raced"), "{out}");
    }

    /// Runtime half of the schema pin; Invariant C holds the bridge to it.
    #[test]
    fn memory_note_declares_supersedes_as_an_optional_list() {
        let defs = builtin_tool_definitions();
        let note = defs.iter().find(|d| d.name == "memory_note").unwrap();
        let prop = &note.input_schema["properties"]["supersedes"];
        assert_eq!(prop["type"], "array");
        let desc = prop["description"].as_str().unwrap();
        assert!(desc.contains("whole-note"), "{desc}");
        assert!(desc.contains("still true"), "{desc}");
        assert_eq!(note.input_schema["required"], serde_json::json!(["text"]));
    }

    /// Only notes show an id in recall output — the one kind `supersedes`
    /// accepts.
    #[test]
    fn recall_hits_show_an_id_on_notes_only() {
        let found = serde_json::json!({
            "mode": "semantic",
            "results": [
                {"id": "1a2b3c4d-0000", "kind": "note", "content": "n", "created_at": "t"},
                {"id": "9f9f9f9f-0000", "kind": "turn", "content": "t", "created_at": "t"},
            ]
        });
        let out = format_recall_hits("q", &found);
        assert!(out.contains("[t · note · id:1a2b3c4d]"), "{out}");
        assert!(out.contains("[t · turn]"), "{out}");
        assert!(!out.contains("9f9f9f9f"), "{out}");
    }

    #[test]
    fn memory_note_is_declared_and_recall_advertises_query() {
        let defs = builtin_tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"memory_note"));

        // Runtime half of the ANAI-126-style guard; the bridge half lives in
        // `openfang-mcp-bridge`. Both must move together or subprocess agents
        // see a different tool than in-process ones.
        let recall = defs
            .iter()
            .find(|d| d.name == "memory_recall")
            .expect("memory_recall must be declared");
        let props = recall
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
            .expect("memory_recall must declare properties");
        for param in ["query", "key", "scope", "kind", "limit"] {
            assert!(props.contains_key(param), "missing param: {param}");
        }
        assert!(
            recall
                .input_schema
                .get("required")
                .and_then(|r| r.as_array())
                .expect("required must be present")
                .is_empty(),
            "`key` must not be required again — exactly-one-of is a handler rule"
        );
    }
    #[async_trait::async_trait]
    impl crate::kernel_handle::KernelHandle for FakeKernelHandle {
        async fn spawn_agent(
            &self,
            _manifest_toml: &str,
            _parent_id: Option<&str>,
        ) -> Result<(String, String), String> {
            Err("not used".into())
        }
        async fn send_to_agent(&self, _agent_id: &str, _message: &str) -> Result<String, String> {
            Err("not used".into())
        }
        fn list_agents(&self) -> Vec<crate::kernel_handle::AgentInfo> {
            self.agents.lock().unwrap().clone()
        }
        // ANAI-210: the effective tool set, keyed by the id/name the fake
        // registered. `None` for an unregistered agent models the real kernel's
        // "unresolvable" answer, which the pre-flight must treat as no evidence.
        fn agent_tool_names(&self, agent_id: &str) -> Option<Vec<String>> {
            self.agent_tools.lock().unwrap().get(agent_id).cloned()
        }
        fn kill_agent(&self, _agent_id: &str) -> Result<(), String> {
            Ok(())
        }
        fn memory_store(
            &self,
            caller: Option<&str>,
            key: &str,
            value: serde_json::Value,
        ) -> Result<(), String> {
            self.memory_calls
                .lock()
                .unwrap()
                .push((caller.map(|c| c.to_string()), key.to_string()));
            self.memory.lock().unwrap().insert(key.to_string(), value);
            Ok(())
        }
        fn memory_recall(
            &self,
            caller: Option<&str>,
            key: &str,
        ) -> Result<Option<serde_json::Value>, String> {
            self.memory_calls
                .lock()
                .unwrap()
                .push((caller.map(|c| c.to_string()), key.to_string()));
            Ok(self.memory.lock().unwrap().get(key).cloned())
        }
        fn memory_episode_close(
            &self,
            caller: Option<&str>,
            reason: &str,
            title: Option<&str>,
            summary: Option<&str>,
        ) -> Result<Option<String>, String> {
            self.episode_closes.lock().unwrap().push((
                caller.map(|c| c.to_string()),
                reason.to_string(),
                title.map(|t| t.to_string()),
                summary.map(|s| s.to_string()),
            ));
            Ok(self.open_episode.lock().unwrap().take())
        }

        fn rehydration_preview(
            &self,
            _caller: Option<&str>,
            _slug: &str,
        ) -> Option<(usize, usize)> {
            *self.rehydration_preview.lock().unwrap()
        }

        fn project_membership_error(&self, _caller: Option<&str>, _slug: &str) -> Option<String> {
            self.membership_error.lock().unwrap().clone()
        }

        fn request_context_reset(
            &self,
            caller: Option<&str>,
            prime_for: Option<&str>,
        ) -> Result<(), String> {
            if self
                .context_reset_fails
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err("reset unavailable".to_string());
            }
            self.context_resets.lock().unwrap().push((
                caller.map(|c| c.to_string()),
                prime_for.map(|p| p.to_string()),
            ));
            Ok(())
        }
        fn has_pending_operator_question(&self, _caller: Option<&str>) -> bool {
            self.pending_operator_question
                .load(std::sync::atomic::Ordering::SeqCst)
        }
        fn memory_status(&self, _caller: Option<&str>) -> Result<serde_json::Value, String> {
            Ok(self.status.lock().unwrap().clone())
        }

        async fn memory_fact_write(
            &self,
            caller: Option<&str>,
            request: crate::kernel_handle::FactWriteRequest,
        ) -> Result<serde_json::Value, String> {
            let scope_ref = request
                .scope_ref
                .clone()
                .unwrap_or_else(|| caller.unwrap_or("unattributed").to_string());
            let addr = format!("{}/{}/{}", request.scope, scope_ref, request.claim_key);
            let previous = self
                .facts
                .lock()
                .unwrap()
                .insert(addr.clone(), request.claim.clone());
            let (outcome, previous_claim) = match previous {
                None => ("created", None),
                Some(p) if p == request.claim => ("affirmed", None),
                Some(p) => ("superseded", Some(p)),
            };
            if let Some(ref displaced) = previous_claim {
                self.fact_history
                    .lock()
                    .unwrap()
                    .entry(addr)
                    .or_default()
                    .insert(
                        0,
                        serde_json::json!({
                            "claim": displaced,
                            "authored_by": caller,
                            "created_at": "2026-08-21T00:00:00+00:00",
                            "superseded_at": "2026-08-21T01:00:00+00:00",
                        }),
                    );
            }
            Ok(serde_json::json!({
                "outcome": outcome,
                "id": "00000000-0000-0000-0000-000000000001",
                "scope": request.scope,
                "scope_ref": scope_ref,
                "claim_key": request.claim_key,
                "previous_claim": previous_claim,
            }))
        }

        fn memory_fact_get(
            &self,
            caller: Option<&str>,
            scope: &str,
            scope_ref: Option<&str>,
            claim_key: &str,
        ) -> Result<serde_json::Value, String> {
            let scope_ref = scope_ref
                .map(str::to_string)
                .unwrap_or_else(|| caller.unwrap_or("unattributed").to_string());
            let addr = format!("{scope}/{scope_ref}/{claim_key}");
            let claim = self.facts.lock().unwrap().get(&addr).cloned();
            Ok(serde_json::json!({
                "scope": scope,
                "scope_ref": scope_ref,
                "claim_key": claim_key,
                "fact": claim.map(|c| serde_json::json!({
                    "claim": c,
                    "status": "settled",
                    "confidence": 1.0,
                    "authored_by": caller,
                })),
            }))
        }

        fn memory_fact_history(
            &self,
            caller: Option<&str>,
            scope: &str,
            scope_ref: Option<&str>,
            claim_key: &str,
            limit: usize,
        ) -> Result<serde_json::Value, String> {
            let scope_ref = scope_ref
                .map(str::to_string)
                .unwrap_or_else(|| caller.unwrap_or("unattributed").to_string());
            let addr = format!("{scope}/{scope_ref}/{claim_key}");
            let mut entries = self
                .fact_history
                .lock()
                .unwrap()
                .get(&addr)
                .cloned()
                .unwrap_or_default();
            entries.truncate(limit);
            Ok(serde_json::json!({
                "scope": scope,
                "scope_ref": scope_ref,
                "claim_key": claim_key,
                "count": entries.len(),
                "entries": entries,
            }))
        }
        async fn memory_search(
            &self,
            caller: Option<&str>,
            query: &str,
            scope: Option<&str>,
            kind: Option<&str>,
            limit: usize,
        ) -> Result<serde_json::Value, String> {
            self.searches.lock().unwrap().push((
                caller.map(|c| c.to_string()),
                query.to_string(),
                scope.map(|s| s.to_string()),
                kind.map(|k| k.to_string()),
                limit,
            ));
            Ok(self.search_result.lock().unwrap().clone())
        }
        async fn memory_note(
            &self,
            caller: Option<&str>,
            text: &str,
            tags: &[String],
        ) -> Result<String, String> {
            self.notes.lock().unwrap().push((
                caller.map(|c| c.to_string()),
                text.to_string(),
                tags.to_vec(),
            ));
            Ok("note-1".to_string())
        }
        // ANAI-270: echo the named notes back as superseded, each 1000 chars,
        // except a ref of "raced" which comes back as lost to a race.
        async fn memory_note_superseding(
            &self,
            caller: Option<&str>,
            text: &str,
            tags: &[String],
            supersedes: &[String],
        ) -> Result<serde_json::Value, String> {
            self.memory_note(caller, text, tags).await?;
            let (raced, done): (Vec<&String>, Vec<&String>) =
                supersedes.iter().partition(|s| s.as_str() == "raced");
            Ok(serde_json::json!({
                "id": "0f0e0d0c-aaaa-bbbb-cccc-000000000001",
                "chars": text.chars().count(),
                "superseded": done.iter().map(|s| serde_json::json!({"id": s, "chars": 1000})).collect::<Vec<_>>(),
                "raced": raced,
            }))
        }
        fn find_agents(&self, _query: &str) -> Vec<crate::kernel_handle::AgentInfo> {
            vec![]
        }
        async fn task_post(
            &self,
            _title: &str,
            _description: &str,
            _assigned_to: Option<&str>,
            _created_by: Option<&str>,
            _payload: &[u8],
        ) -> Result<String, String> {
            Err("not used".into())
        }
        // ANAI-122: record the terminal-reply enqueue and return a stub task id,
        // so the reply tool's happy path completes end-to-end in-test.
        async fn wake_post(
            &self,
            _title: &str,
            description: &str,
            assigned_to: Option<&str>,
            _created_by: Option<&str>,
            _payload: &[u8],
        ) -> Result<String, String> {
            self.wake_posts.lock().unwrap().push((
                assigned_to.unwrap_or("").to_string(),
                description.to_string(),
            ));
            Ok("wake-task-stub".to_string())
        }
        // ANAI-122: consume the seeded reply-right, mirroring the real kernel's
        // consume-on-read `DashMap::remove`.
        fn take_reply_right(&self, agent_id: &str) -> Option<ReplyRight> {
            self.reply_rights.lock().unwrap().remove(agent_id)
        }
        async fn task_claim(&self, _agent_id: &str) -> Result<Option<serde_json::Value>, String> {
            Ok(None)
        }
        async fn task_complete(&self, _task_id: &str, _result: &str) -> Result<(), String> {
            Ok(())
        }
        async fn task_list(&self, _status: Option<&str>) -> Result<Vec<serde_json::Value>, String> {
            Ok(vec![])
        }
        async fn publish_event(
            &self,
            _event_type: &str,
            _payload: serde_json::Value,
        ) -> Result<(), String> {
            Ok(())
        }
        async fn knowledge_add_entity(
            &self,
            _entity: openfang_types::memory::Entity,
        ) -> Result<String, String> {
            Err("not used".into())
        }
        async fn knowledge_add_relation(
            &self,
            _relation: openfang_types::memory::Relation,
        ) -> Result<String, String> {
            Err("not used".into())
        }
        async fn knowledge_query(
            &self,
            _pattern: openfang_types::memory::GraphPattern,
        ) -> Result<Vec<openfang_types::memory::GraphMatch>, String> {
            Ok(vec![])
        }

        async fn cron_create(
            &self,
            agent_id: &str,
            job_json: serde_json::Value,
        ) -> Result<String, String> {
            let id = format!("job-{}", self.created.lock().unwrap().len());
            self.created
                .lock()
                .unwrap()
                .push((agent_id.to_string(), job_json.clone()));
            // Mirror what the real kernel returns (see cron_create in
            // openfang-kernel): `{ "job_id": "...", "status": "created" }`.
            let resp = serde_json::json!({ "job_id": id, "status": "created" });
            Ok(resp.to_string())
        }

        async fn cron_list(&self, _agent_id: &str) -> Result<Vec<serde_json::Value>, String> {
            Ok(self.jobs.lock().unwrap().clone())
        }

        async fn cron_cancel(&self, job_id: &str) -> Result<(), String> {
            self.cancelled.lock().unwrap().push(job_id.to_string());
            Ok(())
        }

        // Mark this kernel as "approval is configured" so the gate at
        // execute_tool would fire for any tool — unless something short-circuits
        // ahead of it.
        fn requires_approval(&self, _tool_name: &str) -> bool {
            true
        }

        async fn request_approval(
            &self,
            _agent_id: &str,
            _tool_name: &str,
            action_summary: &str,
            _origin: Option<&openfang_types::approval::ApprovalOrigin>,
            _cache_binary: Option<&str>,
            command: Option<&str>,
            gatekeeper_note: Option<&str>,
        ) -> Result<openfang_types::approval::ApprovalDecision, String> {
            self.approval_requested
                .store(true, std::sync::atomic::Ordering::SeqCst);
            *self.approval_command.lock().unwrap() = command.map(str::to_string);
            *self.approval_note.lock().unwrap() = gatekeeper_note.map(str::to_string);
            *self.approval_summary.lock().unwrap() = Some(action_summary.to_string());
            Ok(*self.approval_decision.lock().unwrap())
        }

        async fn gatekeeper_review(
            &self,
            _req: &openfang_types::gatekeeper::GateRequest,
        ) -> openfang_types::gatekeeper::GateReview {
            self.gate_consulted
                .store(true, std::sync::atomic::Ordering::SeqCst);
            // ANAI-189: a configured verdict stands in for a judge that
            // actually answered; the unset default stands in for a gate that
            // was never wired up at all, which is `Inert` and NOT a
            // consultation. Collapsing the two here would rebuild the exact
            // conflation this issue removed.
            match *self.gate_verdict.lock().unwrap() {
                Some(v) => openfang_types::gatekeeper::GateReview::answered(v),
                None => openfang_types::gatekeeper::GateReview::failed(
                    openfang_types::gatekeeper::JudgeOutcome::Inert,
                ),
            }
        }

        fn gatekeeper_shadow(&self) -> bool {
            self.gate_shadow.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn audit_gatekeeper_verdict(
            &self,
            agent_id: &str,
            command: &str,
            metadata: &str,
            outcome: &str,
        ) {
            self.gate_audits.lock().unwrap().push((
                agent_id.to_string(),
                command.to_string(),
                metadata.to_string(),
                outcome.to_string(),
            ));
        }

        fn audit_gatekeeper_disposition(
            &self,
            agent_id: &str,
            command: &str,
            metadata: &str,
            disposition: &str,
        ) {
            self.gate_dispositions.lock().unwrap().push((
                agent_id.to_string(),
                command.to_string(),
                metadata.to_string(),
                disposition.to_string(),
            ));
        }
    }

    #[tokio::test]
    async fn test_schedule_create_routes_to_cron_scheduler() {
        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let caller = "11111111-1111-1111-1111-111111111111";

        let input = serde_json::json!({
            "description": "Daily report",
            "schedule": "daily at 9am",
            "agent": "self",
        });

        let out = tool_schedule_create(&input, Some(&handle), Some(caller))
            .await
            .expect("tool_schedule_create should succeed with a valid schedule");

        // User-facing response shape is preserved.
        assert!(out.starts_with("Schedule created:"));
        assert!(out.contains("Daily report"));
        assert!(out.contains("Cron: "));

        // The fake kernel received a cron_create for the caller agent with a
        // well-formed job_json. This is the whole point of #1069: the tool
        // must call into the cron scheduler, not just write to shared memory.
        let created = fake.created.lock().unwrap();
        assert_eq!(created.len(), 1, "cron_create must be called exactly once");
        assert_eq!(created[0].0, caller, "target agent must be the caller");
        let job = &created[0].1;
        assert_eq!(job["schedule"]["kind"], "cron");
        assert_eq!(job["action"]["kind"], "agent_turn");
        assert_eq!(job["action"]["message"], "Daily report");
        assert!(job["schedule"]["expr"].is_string());
        assert_eq!(job["one_shot"], false);
    }

    /// The allowlist wall must run BEFORE the approval gate for shell_exec.
    /// A non-allowlisted command in Allowlist mode must be hard-denied without
    /// ever firing the approval gate — no prompt, no approve-similar cache
    /// population, and therefore no misleading "Approved · cached" stamp for a
    /// command the next layer rejects (the `whoami` incident, 2026-06-17).
    #[tokio::test]
    async fn test_allowlist_wall_precedes_approval_gate_for_shell_exec() {
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = ExecPolicy {
            mode: ExecSecurityMode::Allowlist,
            allowed_commands: vec!["grep".to_string()],
            ..ExecPolicy::default()
        };
        let input = serde_json::json!({ "command": "whoami" });

        let result = execute_tool(
            "test-id",
            "shell_exec",
            &input,
            Some(&handle),
            None, // allowed_tools (capability check skipped)
            Some("test-agent"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,          // media_engine
            Some(&policy), // exec_policy
            None,          // file_policy
            None,          // tts_engine
            None,          // docker_config
            None,          // process_manager
            None,          // origin
        )
        .await;

        assert!(
            result.is_error,
            "non-allowlisted shell_exec must be blocked, got: {}",
            result.content
        );
        assert!(
            result.content.contains("not in the exec allowlist"),
            "block must come from the allowlist wall, got: {}",
            result.content
        );
        assert!(
            !fake
                .approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "approval gate must NOT fire for a command the allowlist wall rejects \
             (no prompt, no cache, no misleading Approved stamp)"
        );
    }

    // ---------------------------------------------------------------------
    // ANAI-154: the gatekeeper (layer 3.5), driven end-to-end through
    // `execute_tool` against a spy judge.
    // ---------------------------------------------------------------------

    fn gate_policy(bin: &str) -> openfang_types::config::ExecPolicy {
        openfang_types::config::ExecPolicy {
            mode: openfang_types::config::ExecSecurityMode::Allowlist,
            allowed_commands: vec![bin.to_string()],
            ..openfang_types::config::ExecPolicy::default()
        }
    }

    async fn run_gated(
        handle: &Arc<dyn crate::kernel_handle::KernelHandle>,
        policy: &openfang_types::config::ExecPolicy,
        command: &str,
    ) -> ToolResult {
        let input = serde_json::json!({ "command": command });
        execute_tool(
            "test-id",
            "shell_exec",
            &input,
            Some(handle),
            None,
            Some("test-agent"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(policy),
            None,
            None,
            None,
            None,
            None,
        )
        .await
    }

    /// A `Suppress` executes with NO prompt — and, critically, without ever
    /// entering the block that builds `cache_binary`. Per-command, never
    /// per-pattern: one suppression must not become fifty via Approve-Similar.
    #[tokio::test]
    async fn test_gatekeeper_suppress_skips_the_prompt_entirely() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Suppress),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("grep");

        let result = run_gated(&handle, &policy, "grep --version").await;

        assert!(
            !result.is_error,
            "suppressed command should execute, got: {}",
            result.content
        );
        assert!(
            !fake
                .approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "a gatekeeper Suppress must not reach request_approval at all"
        );
        assert!(
            fake.approval_command.lock().unwrap().is_none(),
            "a Suppress must never populate the Approve-Similar cache path"
        );
    }

    /// ANAI-186: every verdict leaves exactly one durable row, and the row
    /// carries the FULL command.
    ///
    /// The suppressed and denied rows are the whole point — those commands are
    /// never shown to an operator, so an absent or truncated row here is not a
    /// logging gap, it is the only evidence of the decision going missing. The
    /// `escalate` row matters too: without a denominator, a suppression count
    /// is a number with no scale.
    #[tokio::test]
    async fn test_every_gatekeeper_verdict_writes_one_durable_row() {
        use openfang_types::gatekeeper::GateVerdict;

        // Long enough to trip the 200- and 512-char truncations that exist on
        // every neighbouring string in this path.
        let tail = "x".repeat(600);
        for (verdict, token) in [
            (GateVerdict::Suppress, "suppress"),
            (GateVerdict::Escalate, "escalate"),
            (GateVerdict::Deny, "deny"),
        ] {
            let fake = Arc::new(FakeKernelHandle::new().with_gate_verdict(verdict));
            let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
            let policy = gate_policy("grep");
            let command = format!("grep --version {tail}");

            let _ = run_gated(&handle, &policy, &command).await;

            let rows = fake.gate_audits.lock().unwrap().clone();
            assert_eq!(rows.len(), 1, "{token}: expected exactly one audit row");
            let (agent, cmd, metadata, outcome) = &rows[0];
            assert_eq!(agent, "test-agent");
            assert_eq!(outcome, token, "the row must name the verdict it recorded");
            assert_eq!(
                cmd, &command,
                "{token}: the audit row must carry the verbatim, untruncated command"
            );
            assert!(
                metadata.contains("consulted_model="),
                "{token}: the row must say whether a model was in the decision"
            );
        }
    }

    /// ANAI-241: every gated command leaves exactly one disposition row, and
    /// that row joins to its verdict row.
    ///
    /// The verdict row says what the judge wanted. Until now nothing said what
    /// happened next, so post-flip "the human approved this" and "the judge
    /// suppressed it and the human never saw it" were indistinguishable in the
    /// chain — which is the entire safety boundary the `enabled = true` flip
    /// moves. `human_decided` is the bit that answers it and is asserted here
    /// per-arm, because a `false` that should be `true` reads as a quiet
    /// operator rather than as a missing human.
    ///
    /// The `gk=` join is asserted rather than assumed: without it the only key
    /// is (agent, command), which is ambiguous exactly where it matters most —
    /// a build wrapper invoked five times in a minute is five identical rows.
    #[tokio::test]
    async fn test_every_gated_command_writes_one_correlated_disposition_row() {
        use openfang_types::approval::ApprovalDecision;
        use openfang_types::gatekeeper::GateVerdict;

        let tail = "y".repeat(600);
        for (verdict, decision, token, human) in [
            (
                GateVerdict::Suppress,
                ApprovalDecision::Approved,
                "gatekeeper_suppressed",
                false,
            ),
            (
                GateVerdict::Deny,
                ApprovalDecision::Approved,
                "gatekeeper_denied",
                false,
            ),
            (
                GateVerdict::Escalate,
                ApprovalDecision::Approved,
                "approved",
                true,
            ),
            (
                GateVerdict::Escalate,
                ApprovalDecision::Denied,
                "denied",
                true,
            ),
            (
                GateVerdict::Escalate,
                ApprovalDecision::TimedOut,
                "timed_out",
                false,
            ),
        ] {
            let fake = Arc::new(
                FakeKernelHandle::new()
                    .with_gate_verdict(verdict)
                    .with_approval_decision(decision),
            );
            let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
            let policy = gate_policy("grep");
            let command = format!("grep --version {tail}");

            let _ = run_gated(&handle, &policy, &command).await;

            let rows = fake.gate_dispositions.lock().unwrap().clone();
            assert_eq!(
                rows.len(),
                1,
                "{token}: expected exactly one disposition row"
            );
            let (agent, cmd, metadata, disposition) = &rows[0];
            assert_eq!(agent, "test-agent");
            assert_eq!(disposition, token);
            assert_eq!(
                cmd, &command,
                "{token}: the disposition row must carry the verbatim, untruncated command"
            );
            assert!(
                metadata.contains(&format!("human_decided={human}")),
                "{token}: expected human_decided={human} in {metadata}"
            );

            // The join key must be present on BOTH rows and identical.
            let gk_of = |m: &str| {
                m.split_whitespace()
                    .find_map(|f| f.strip_prefix("gk="))
                    .map(str::to_string)
                    .unwrap_or_default()
            };
            let verdict_gk = gk_of(&fake.gate_audits.lock().unwrap()[0].2);
            let disposition_gk = gk_of(metadata);
            assert!(
                !verdict_gk.is_empty() && verdict_gk != "none",
                "{token}: the verdict row must carry a correlation id"
            );
            assert_eq!(
                verdict_gk, disposition_gk,
                "{token}: the disposition row must join to its verdict row"
            );
        }
    }

    /// ANAI-240: the deterministic sheet verdict is recorded on every row.
    ///
    /// `PathFactSheet::suppress_eligible` had zero production call sites until
    /// this landed — the stricter of the two mechanisms was never asked, so the
    /// corpus could not say how often the judge suppressed something no
    /// deterministic rule would have granted. `grep --version` names no path at
    /// all, so the sheet is empty and therefore ineligible; a `Suppress`
    /// verdict on it is exactly the disagreement the field exists to count.
    #[tokio::test]
    async fn test_deterministic_eligibility_is_recorded_alongside_the_verdict() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Suppress),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let _ = run_gated(&handle, &gate_policy("grep"), "grep --version").await;
        let metadata = fake.gate_audits.lock().unwrap()[0].2.clone();
        assert!(
            metadata.contains("det=ineligible"),
            "an empty sheet is not deterministic-eligible: {metadata}"
        );
        assert!(
            metadata.contains("det_disagree=true"),
            "a model suppression the sheet refuses must be countable: {metadata}"
        );

        // The other side: an escalate is never a disagreement, whatever the
        // sheet says. The judge is allowed to be the stricter mechanism.
        let strict = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Escalate),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = strict.clone();
        let _ = run_gated(&handle, &gate_policy("grep"), "grep --version").await;
        let metadata = strict.gate_audits.lock().unwrap()[0].2.clone();
        assert!(
            metadata.contains("det_disagree=false"),
            "an escalate must never count as a disagreement: {metadata}"
        );
    }

    /// ANAI-189: `consulted_model` must describe the MODEL, not the floor.
    ///
    /// The bug this pins: `consulted` was a structural constant on the
    /// not-floor branch, so every row where the floor missed claimed a model
    /// consultation — including timeouts, provider errors, and an inert gate.
    /// A timed-out `escalate` counted as a considered one, which biases the
    /// exact escalate-rate the `enabled = true` flip is decided on.
    #[tokio::test]
    async fn test_floor_short_circuit_records_no_consultation() {
        // `curl` trips the network-binary floor, so the judge is deliberately
        // never billed. `--version` keeps the executed command harmless.
        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("cp");

        // ANAI-206 commit 6: `network` is a fact the judge weighs now, not a
        // bypass, so this pins the property on a predicate that is still hard.
        // A write to the judge's own policy file is not a question the judge
        // can be asked. Source and destination are the same path, so the
        // command is inert even if it were ever executed.
        let _ = run_gated(
            &handle,
            &policy,
            "cp ~/.openfang/gatekeeper.md ~/.openfang/gatekeeper.md",
        )
        .await;

        assert!(
            !fake
                .gate_consulted
                .load(std::sync::atomic::Ordering::SeqCst),
            "a floor hit must not reach the judge at all"
        );
        let rows = fake.gate_audits.lock().unwrap().clone();
        assert_eq!(rows.len(), 1);
        let metadata = &rows[0].2;
        assert!(
            metadata.contains("consulted_model=false"),
            "floor short-circuit claimed a consultation: {metadata}"
        );
        assert!(
            metadata.contains("judge=floor"),
            "the row must name WHY no model answered: {metadata}"
        );
    }

    /// The other half: a judge that genuinely answered is recorded as such,
    /// and is distinguishable from every fail-closed path that produces the
    /// same `escalate` token.
    #[tokio::test]
    async fn test_answered_and_inert_are_distinguishable_rows() {
        // A configured verdict stands in for a live judge.
        let answered = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Escalate),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = answered.clone();
        let _ = run_gated(&handle, &gate_policy("grep"), "grep --version").await;
        let metadata = answered.gate_audits.lock().unwrap()[0].2.clone();
        assert!(
            metadata.contains("consulted_model=true") && metadata.contains("judge=answered"),
            "a real judgement must record as one: {metadata}"
        );

        // No configured verdict = no gate wired up = inert. Same `escalate`
        // outcome token, and it must NOT read as a consultation.
        let inert = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = inert.clone();
        let _ = run_gated(&handle, &gate_policy("grep"), "grep --version").await;
        let metadata = inert.gate_audits.lock().unwrap()[0].2.clone();
        assert!(
            metadata.contains("consulted_model=false") && metadata.contains("judge=inert"),
            "an unanswered escalate must not read as a judgement: {metadata}"
        );
    }

    /// ANAI-187: shadow mode. The judge is consulted, its verdict is recorded
    /// honestly, and the command prompts anyway.
    ///
    /// This is the test that makes the flip decision checkable. If the row
    /// said `escalate` — the thing that actually happened — the corpus would
    /// contain zero would-have-suppressed rows and a week of shadow traffic
    /// would tell the operator nothing at all.
    #[tokio::test]
    async fn test_shadow_records_the_verdict_and_prompts_anyway() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Suppress)
                .with_gate_shadow(),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("grep");

        let result = run_gated(&handle, &policy, "grep --version").await;

        assert!(!result.is_error, "shadow must not change the outcome");
        assert!(
            fake.gate_consulted
                .load(std::sync::atomic::Ordering::SeqCst),
            "shadow mode exists to consult the judge; not consulting it \
             produces no data"
        );
        assert!(
            fake.approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "a shadow Suppress must still reach the human"
        );

        let rows = fake.gate_audits.lock().unwrap().clone();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].3, "shadow_suppress",
            "the row must record what the judge said, prefixed so it can never \
             be read as something that happened"
        );
    }

    /// ANAI-187: a shadow Deny is an escalation too. Shadow means the judge
    /// cannot block, not merely that it cannot suppress — a judge that can
    /// still refuse execution is being tested in production.
    #[tokio::test]
    async fn test_shadow_deny_does_not_block() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Deny)
                .with_gate_shadow(),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("grep");

        let result = run_gated(&handle, &policy, "grep --version").await;

        assert!(
            !result.is_error,
            "a shadow Deny must not block execution, got: {}",
            result.content
        );
        assert!(
            fake.approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "a shadow Deny falls through to the human like everything else"
        );
        assert_eq!(fake.gate_audits.lock().unwrap()[0].3, "shadow_deny");
    }

    /// A `Deny` refuses without ever showing a prompt, and says so legibly.
    #[tokio::test]
    async fn test_gatekeeper_deny_blocks_without_prompting() {
        // ANAI-185(a): counters are process-global statics, so assert the
        // delta this call produces rather than an absolute — the test binary
        // runs these in parallel and an absolute would be a flake generator.
        let before = crate::gatekeeper::counters();
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Deny),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("grep");

        let result = run_gated(&handle, &policy, "grep --version").await;

        assert!(result.is_error, "a gatekeeper Deny must block execution");
        assert!(
            result.content.contains("gatekeeper"),
            "the agent must be told which layer refused, got: {}",
            result.content
        );
        assert!(
            !fake
                .approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "a Deny must never surface a prompt"
        );
        // ANAI-185(a). The whole point of the ticket: a deny that prompts
        // nobody must still increment something a human can read.
        let after = crate::gatekeeper::counters();
        assert!(
            after.deny > before.deny,
            "an enforced Deny must be metered; before={before:?} after={after:?}"
        );
    }

    /// An `Escalate` falls through to the human prompt unchanged, with the
    /// machine's reason attached so the operator knows why it was handed back.
    #[tokio::test]
    async fn test_gatekeeper_escalate_falls_through_to_the_prompt() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Escalate),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("grep");

        let result = run_gated(&handle, &policy, "grep --version").await;

        assert!(!result.is_error, "approved after escalation should execute");
        assert!(
            fake.approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "an Escalate must reach the human prompt"
        );
        assert_eq!(
            fake.approval_command.lock().unwrap().as_deref(),
            Some("grep --version"),
            "ANAI-151's verbatim command must survive the gatekeeper detour"
        );

        // ANAI-188. The reason must ride its own parameter to the render site.
        // Folded into `action_summary` it never reached the prompt at all:
        // `render_approval_body` prefers `command` for shell_exec (ANAI-151),
        // so the annotation surfaced only on the post-approval resolution edit.
        let note = fake.approval_note.lock().unwrap().clone();
        assert!(
            note.as_deref().is_some_and(|n| n.contains("gatekeeper")),
            "the escalation reason must reach request_approval as its own \
             argument, got {note:?}"
        );
        let summary = fake.approval_summary.lock().unwrap().clone();
        assert!(
            !summary
                .as_deref()
                .unwrap_or_default()
                .contains("gatekeeper"),
            "the note must NOT be folded into the agent-controlled summary — \
             separate channels is the whole fix, got {summary:?}"
        );
    }

    /// The deterministic floor is a CEILING on the judge's authority. A command
    /// that trips a Rust predicate escalates even when the judge says
    /// `Suppress` — and the model is not consulted at all, so there is no call
    /// for a poisoned command string to influence.
    ///
    /// This is the single most important test in ANAI-154: it is what makes
    /// "the model can only narrow" a property of the code rather than a claim
    /// in a design doc.
    #[tokio::test]
    async fn test_floor_overrides_a_suppress_verdict_without_consulting_the_model() {
        let fake = Arc::new(
            FakeKernelHandle::new()
                .with_gate_verdict(openfang_types::gatekeeper::GateVerdict::Suppress)
                // Deny at the human gate so the network command never runs even
                // though the escalation path is exercised.
                .with_approval_decision(openfang_types::approval::ApprovalDecision::Denied),
        );
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("cp");

        // ANAI-206 commit 6: pinned on `policy_self_modification` rather than
        // `network`, which is now a fact the judge weighs. The property under
        // test is unchanged — a hard floor short-circuits before the model
        // call. Source and destination are the same path, so the command is
        // inert regardless.
        let result = run_gated(
            &handle,
            &policy,
            "cp ~/.openfang/gatekeeper.md ~/.openfang/gatekeeper.md",
        )
        .await;

        assert!(
            !fake
                .gate_consulted
                .load(std::sync::atomic::Ordering::SeqCst),
            "a floor hit must short-circuit BEFORE the model call — no LLM, no injection surface"
        );
        assert!(
            fake.approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "a floor hit must escalate to a human regardless of what the judge would have said"
        );
        assert!(result.is_error, "the human denied it");
    }

    /// The trait default is `Escalate`, so a handle with no gatekeeper — every
    /// test double, every WASM host shim, and the kernel itself when
    /// `[gatekeeper] enabled = false` — behaves exactly as it did before this
    /// change. The absence of a judge can never be read as a suppression.
    #[tokio::test]
    async fn test_absent_gatekeeper_preserves_pre_anai154_behaviour() {
        let fake = Arc::new(FakeKernelHandle::new()); // no with_gate_verdict
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = gate_policy("grep");

        let result = run_gated(&handle, &policy, "grep --version").await;

        assert!(!result.is_error);
        assert!(
            fake.approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "with no judge configured, every gated command must still reach the human"
        );
    }

    /// Positive control for the wall-before-gate ordering. An *allowlisted*
    /// command in Allowlist mode must CLEAR the wall, reach the approval gate,
    /// and fire it. This makes the negative test above a true negative: it
    /// proves the spy's `request_approval` is wired as the `KernelHandle`
    /// trait impl (not a dead inherent method) and that it *does* record a
    /// call when the gate is actually reached. Without this, an un-fired spy
    /// in the negative test could be a false pass (e.g. a mis-wired spy that
    /// never records regardless of gate firing).
    #[tokio::test]
    async fn test_allowlisted_command_reaches_approval_gate_for_shell_exec() {
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let policy = ExecPolicy {
            mode: ExecSecurityMode::Allowlist,
            allowed_commands: vec!["grep".to_string()],
            ..ExecPolicy::default()
        };
        // `grep` IS allowlisted, so it clears the wall and reaches the gate.
        // `--version` exits 0 immediately, so the post-approval execution is
        // fast and side-effect-free.
        let input = serde_json::json!({ "command": "grep --version" });

        let result = execute_tool(
            "test-id",
            "shell_exec",
            &input,
            Some(&handle),
            None, // allowed_tools (capability check skipped)
            Some("test-agent"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,          // media_engine
            Some(&policy), // exec_policy
            None,          // file_policy
            None,          // tts_engine
            None,          // docker_config
            None,          // process_manager
            None,          // origin
        )
        .await;

        // The gate MUST have fired — this is the positive control.
        assert!(
            fake.approval_requested
                .load(std::sync::atomic::Ordering::SeqCst),
            "approval gate MUST fire for an allowlisted command that clears the \
             wall — proves the spy records gate calls and the negative test is \
             a true negative"
        );
        // ANAI-151: and it must hand the gate the VERBATIM command, not the
        // serialized-JSON summary. If this ever regresses to `{"command":"…`
        // the render sites are back to displaying escaped, tail-cut JSON.
        assert_eq!(
            fake.approval_command.lock().unwrap().as_deref(),
            Some("grep --version"),
            "the gate must carry the raw command string to the operator surface"
        );
        // And the wall did NOT block it (independent of the command's exit
        // code, so this assertion is robust across grep implementations).
        assert!(
            !result.content.contains("not in the exec allowlist"),
            "an allowlisted command must clear the wall, got: {}",
            result.content
        );
    }

    /// ANAI-153. The acceptance criterion, exercised end to end through
    /// `execute_tool`: an agent whose approval request times out must receive
    /// something it can tell apart from a refusal.
    ///
    /// Before this change both outcomes rendered the same sentence
    /// ("was denied or timed out"), so an unattended run concluded it had been
    /// refused and abandoned work no human had ever looked at.
    #[tokio::test]
    async fn test_timeout_and_backpressure_are_distinguishable_from_denial() {
        use openfang_types::approval::ApprovalDecision;
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        async fn run_with(decision: ApprovalDecision) -> ToolResult {
            let fake = Arc::new(FakeKernelHandle::new().with_approval_decision(decision));
            let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
            let policy = ExecPolicy {
                mode: ExecSecurityMode::Allowlist,
                allowed_commands: vec!["grep".to_string()],
                ..ExecPolicy::default()
            };
            // Clears the allowlist wall, so the gate is genuinely reached.
            let input = serde_json::json!({ "command": "grep --version" });
            execute_tool(
                "test-id",
                "shell_exec",
                &input,
                Some(&handle),
                None,
                Some("test-agent"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&policy),
                None,
                None,
                None,
                None,
                None,
            )
            .await
        }

        let denied = run_with(ApprovalDecision::Denied).await;
        let timed_out = run_with(ApprovalDecision::TimedOut).await;
        let backpressure = run_with(ApprovalDecision::Backpressure).await;

        // All three still BLOCK. This change never opens a gate.
        for r in [&denied, &timed_out, &backpressure] {
            assert!(r.is_error, "every non-approved outcome must block");
        }

        // And all three say something different.
        assert_ne!(
            denied.content, timed_out.content,
            "a timeout must not read as a denial"
        );
        assert_ne!(
            denied.content, backpressure.content,
            "queue backpressure must not read as a denial"
        );
        assert_ne!(timed_out.content, backpressure.content);

        // The distinction has to be legible to a model, not just unequal.
        assert!(
            timed_out.content.contains("NOT a denial"),
            "timeout text must tell the agent it was not refused, got: {}",
            timed_out.content
        );
        assert!(
            backpressure.content.contains("NOT a denial"),
            "backpressure text must tell the agent it was not refused, got: {}",
            backpressure.content
        );
        assert!(
            !denied.content.contains("NOT a denial"),
            "a real denial must not disclaim itself, got: {}",
            denied.content
        );
        assert!(
            denied.content.contains("Do not retry"),
            "a real denial is terminal and must say so, got: {}",
            denied.content
        );
    }

    #[tokio::test]
    async fn test_schedule_create_rejects_missing_description() {
        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({ "schedule": "every hour" });
        let err = tool_schedule_create(&input, Some(&handle), Some("aaa"))
            .await
            .unwrap_err();
        assert!(err.contains("description"));
    }

    #[tokio::test]
    async fn test_schedule_list_reads_from_cron_scheduler() {
        let job = serde_json::json!({
            "id": "cron-1",
            "name": "demo",
            "enabled": true,
            "schedule": { "kind": "cron", "expr": "0 9 * * *" },
            "action": { "kind": "agent_turn", "message": "hello" },
            "created_at": "2026-01-01T00:00:00Z",
            "agent_id": "aaa",
        });
        let fake = Arc::new(FakeKernelHandle::new().with_job(job));
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();

        let out = tool_schedule_list(Some(&handle), Some("aaa"))
            .await
            .expect("schedule_list should succeed");
        assert!(out.contains("Scheduled tasks (1)"));
        assert!(out.contains("0 9 * * *"));
        assert!(out.contains("hello"));
    }

    #[tokio::test]
    async fn test_schedule_list_empty() {
        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let out = tool_schedule_list(Some(&handle), Some("aaa"))
            .await
            .unwrap();
        assert_eq!(out, "No scheduled tasks.");
    }

    #[tokio::test]
    async fn test_schedule_delete_routes_to_cron_cancel() {
        let fake = Arc::new(FakeKernelHandle::new());
        let handle: Arc<dyn crate::kernel_handle::KernelHandle> = fake.clone();
        let input = serde_json::json!({ "id": "abc-123" });
        let out = tool_schedule_delete(&input, Some(&handle)).await.unwrap();
        assert!(out.contains("abc-123"));
        let cancelled = fake.cancelled.lock().unwrap();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0], "abc-123");
    }

    #[tokio::test]
    async fn test_schedule_tools_require_kernel() {
        // Without a kernel handle, the new tools must fail loudly rather than
        // silently writing to the old shared-memory key.
        let err = tool_schedule_create(
            &serde_json::json!({"description": "x", "schedule": "every hour"}),
            None,
            Some("aaa"),
        )
        .await
        .unwrap_err();
        assert!(err.to_lowercase().contains("kernel"));

        let err = tool_schedule_list(None, Some("aaa")).await.unwrap_err();
        assert!(err.to_lowercase().contains("kernel"));

        let err = tool_schedule_delete(&serde_json::json!({"id": "x"}), None)
            .await
            .unwrap_err();
        assert!(err.to_lowercase().contains("kernel"));
    }

    // -----------------------------------------------------------------
    // ANAI-53: synthesize_attach_directives — pure helper tests
    // -----------------------------------------------------------------

    #[test]
    fn synth_attach_none_is_noop() {
        let out = super::synthesize_attach_directives("hello", None).unwrap();
        assert_eq!(out, "hello");
    }

    #[test]
    fn synth_attach_null_is_noop() {
        let v = serde_json::Value::Null;
        let out = super::synthesize_attach_directives("hello", Some(&v)).unwrap();
        assert_eq!(out, "hello");
    }

    #[test]
    fn synth_attach_empty_array_is_noop() {
        let v = serde_json::json!([]);
        let out = super::synthesize_attach_directives("hello", Some(&v)).unwrap();
        assert_eq!(out, "hello");
    }

    #[test]
    fn synth_attach_single_relative_path() {
        let v = serde_json::json!(["report.pdf"]);
        let out = super::synthesize_attach_directives("see attached", Some(&v)).unwrap();
        assert_eq!(out, "<openfang:attach path=\"report.pdf\"/>\nsee attached");
    }

    #[test]
    fn synth_attach_single_absolute_path() {
        let v = serde_json::json!(["/tmp/file.png"]);
        let out = super::synthesize_attach_directives("", Some(&v)).unwrap();
        // Empty message → trailing newline trimmed.
        assert_eq!(out, "<openfang:attach path=\"/tmp/file.png\"/>");
    }

    #[test]
    fn synth_attach_multiple_paths_preserve_order() {
        let v = serde_json::json!(["a.pdf", "b.png", "c.opus"]);
        let out = super::synthesize_attach_directives("caption", Some(&v)).unwrap();
        assert_eq!(
            out,
            "<openfang:attach path=\"a.pdf\"/>\n<openfang:attach path=\"b.png\"/>\n<openfang:attach path=\"c.opus\"/>\ncaption"
        );
    }

    #[test]
    fn synth_attach_composes_with_inline_directive() {
        // Caller wrote one inline directive, then also passed an
        // attachments param. Both must survive; downstream parser sees
        // two directives in the final message.
        let v = serde_json::json!(["b.pdf"]);
        let msg = "<openfang:attach path=\"a.pdf\"/>\nhere are two files";
        let out = super::synthesize_attach_directives(msg, Some(&v)).unwrap();
        assert_eq!(
            out,
            "<openfang:attach path=\"b.pdf\"/>\n<openfang:attach path=\"a.pdf\"/>\nhere are two files"
        );
        // And the parser regex finds both.
        let re = regex_lite::Regex::new(r#"<openfang:attach\s+([^>]*?)/>"#).unwrap();
        assert_eq!(re.find_iter(&out).count(), 2);
    }

    #[test]
    fn synth_attach_rejects_non_array() {
        let v = serde_json::json!("not-an-array");
        let err = super::synthesize_attach_directives("hi", Some(&v)).unwrap_err();
        assert!(err.contains("must be an array"), "err = {err}");
    }

    #[test]
    fn synth_attach_rejects_non_string_element() {
        let v = serde_json::json!(["ok.pdf", 42]);
        let err = super::synthesize_attach_directives("hi", Some(&v)).unwrap_err();
        assert!(err.contains("attachments[1]"), "err = {err}");
        assert!(err.contains("must be a string"), "err = {err}");
    }

    #[test]
    fn synth_attach_rejects_empty_path_element() {
        let v = serde_json::json!(["ok.pdf", ""]);
        let err = super::synthesize_attach_directives("hi", Some(&v)).unwrap_err();
        assert!(err.contains("attachments[1]"), "err = {err}");
        assert!(err.contains("empty"), "err = {err}");
    }

    #[test]
    fn synth_attach_rejects_quote_in_path() {
        let v = serde_json::json!(["weird\".pdf"]);
        let err = super::synthesize_attach_directives("hi", Some(&v)).unwrap_err();
        assert!(err.contains("directive boundary"), "err = {err}");
    }

    #[test]
    fn synth_attach_rejects_newline_in_path() {
        let v = serde_json::json!(["line1\nline2.pdf"]);
        let err = super::synthesize_attach_directives("hi", Some(&v)).unwrap_err();
        assert!(err.contains("directive boundary"), "err = {err}");
    }

    #[test]
    fn synth_attach_rejects_angle_bracket_in_path() {
        let v = serde_json::json!(["foo<bar>.pdf"]);
        let err = super::synthesize_attach_directives("hi", Some(&v)).unwrap_err();
        assert!(err.contains("directive boundary"), "err = {err}");
    }

    #[test]
    fn synth_attach_traversal_path_still_synthesized() {
        // Synthesis is pure — no path policy here. The downstream
        // outbound_attach parser is what rejects traversal/escape paths
        // (covered by its own test suite). This test pins down that the
        // helper does NOT pre-filter — security is downstream.
        let v = serde_json::json!(["../../../etc/hosts"]);
        let out = super::synthesize_attach_directives("hi", Some(&v)).unwrap();
        assert!(out.contains("<openfang:attach path=\"../../../etc/hosts\"/>"));
        // Defence in depth lives in outbound_attach::resolve_directive,
        // which canonicalises and applies allow_roots — see its tests.
    }

    #[test]
    fn test_trusted_commands_auto_approve_parity_shell_and_process_start() {
        // Parity: both shell_exec and process_start carry the command in
        // input["command"], and the gate uses command_approval_report for both.
        use openfang_types::config::{ExecPolicy, ExecSecurityMode};

        let policy = ExecPolicy {
            mode: ExecSecurityMode::Allowlist,
            trusted_commands: vec!["git".to_string()],
            ..ExecPolicy::default()
        };

        // shell_exec-shaped input.
        let shell_input = serde_json::json!({ "command": "git status" });
        let cmd = shell_input.get("command").and_then(|v| v.as_str()).unwrap();
        assert!(
            crate::subprocess_sandbox::command_approval_report(cmd, &policy).is_some(),
            "git must auto-approve via trusted_commands for shell_exec"
        );

        // process_start-shaped input (same "command" key — is_shell_tool covers both).
        let ps_input = serde_json::json!({ "command": "git", "args": ["status"] });
        let ps_cmd = ps_input.get("command").and_then(|v| v.as_str()).unwrap();
        assert!(
            crate::subprocess_sandbox::command_approval_report(ps_cmd, &policy).is_some(),
            "git must auto-approve via trusted_commands for process_start"
        );

        // A non-trusted command must NOT auto-approve (prompt still fires).
        let deny_input = serde_json::json!({ "command": "curl https://x" });
        let deny_cmd = deny_input.get("command").and_then(|v| v.as_str()).unwrap();
        assert!(
            crate::subprocess_sandbox::command_approval_report(deny_cmd, &policy).is_none(),
            "curl must not auto-approve"
        );
    }
}

#[cfg(test)]
mod convert_dispatch_tests {
    use super::*;

    #[cfg(unix)]
    use std::fs;
    #[cfg(unix)]
    use tempfile::TempDir;

    /// Build a hermetic OpenFang home: `scripts/stub.sh` + `convert/recipes.toml`
    /// defining md->txt. The stub is a 2-arg `INPUT OUTPUT` copier that also
    /// records its raw argv to `<output>.argv`, so a test can prove a
    /// metachar-laden path arrived as ONE literal argument (no shell expansion).
    #[cfg(unix)]
    fn hermetic_home(needs: &str) -> TempDir {
        use std::os::unix::fs::PermissionsExt;
        let home = TempDir::new().unwrap();
        let scripts = home.path().join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        let stub = scripts.join("stub.sh");
        fs::write(
            &stub,
            "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > \"$2.argv\"\ncp \"$1\" \"$2\"\n",
        )
        .unwrap();
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
        let convert_dir = home.path().join("convert");
        fs::create_dir_all(&convert_dir).unwrap();
        let needs_line = if needs.is_empty() {
            String::new()
        } else {
            format!("needs = [\"{needs}\"]\n")
        };
        fs::write(
            convert_dir.join("recipes.toml"),
            format!(
                "[[recipe]]\nfrom = \"md\"\nto = \"txt\"\nargv = [\"{{script}}/stub.sh\", \"{{input}}\", \"{{output}}\"]\n{needs_line}out_ext = \"txt\"\n"
            ),
        )
        .unwrap();
        home
    }

    #[cfg(unix)]
    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_happy_path_md_to_txt() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "# hi").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(true), "envelope: {out}");
        assert_eq!(v["format"], "txt");
        assert_eq!(v["output_path"], "note.txt");
        assert!(ws.path().join("note.txt").is_file());
    }

    // --- ANAI-300: file_convert resolves paths under file_policy ---------

    /// Enabled policy, default Deny, with the given absolute rules. Paths are
    /// canonicalized so macOS's /var -> /private/var symlink cannot make a rule
    /// miss the canonical path the resolver evaluates.
    #[cfg(unix)]
    fn convert_policy(
        enabled: bool,
        rules: Vec<(&Path, openfang_types::config::FileAccessTier)>,
    ) -> openfang_types::config::FilePolicy {
        openfang_types::config::FilePolicy::new(
            enabled,
            openfang_types::config::FileAccessTier::Deny,
            rules
                .into_iter()
                .map(|(p, tier)| openfang_types::config::FileRule {
                    path: p.canonicalize().unwrap().to_string_lossy().into_owned(),
                    tier,
                })
                .collect(),
        )
    }

    #[cfg(unix)]
    async fn convert_with(
        input: serde_json::Value,
        ws: &Path,
        policy: Option<&openfang_types::config::FilePolicy>,
        home: &Path,
    ) -> serde_json::Value {
        let out = tool_file_convert_with_policy_in(&input, Some(ws), policy, home)
            .await
            .unwrap();
        parse(&out)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_read_tier_input_outside_workspace_converts() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let src = outside.path().join("note.md");
        fs::write(&src, "# from outside").unwrap();
        let fp = convert_policy(true, vec![(ws.path(), T::Write), (outside.path(), T::Read)]);
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": src.to_str().unwrap(), "output": "note.txt" }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["ok"], serde_json::json!(true), "envelope: {v}");
        assert_eq!(
            fs::read_to_string(ws.path().join("note.txt")).unwrap(),
            "# from outside"
        );
    }

    /// A read-tier input with no `output` defaults next to the input, which is
    /// not writable. It must refuse and name the `output` argument, and it must
    /// NOT quietly relocate the output somewhere else.
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_read_tier_default_output_refused_with_signpost() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let src = outside.path().join("note.md");
        fs::write(&src, "x").unwrap();
        let fp = convert_policy(true, vec![(ws.path(), T::Write), (outside.path(), T::Read)]);
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": src.to_str().unwrap() }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("read-only"), "{msg}");
        assert!(msg.contains("Pass 'output'"), "{msg}");
        assert!(!outside.path().join("note.txt").exists());
        assert!(!ws.path().join("note.txt").exists(), "must not relocate");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_deny_tier_input_refused() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let src = outside.path().join("note.md");
        fs::write(&src, "x").unwrap();
        // No rule for `outside`: the default tier (Deny) governs it.
        let fp = convert_policy(true, vec![(ws.path(), T::Write)]);
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": src.to_str().unwrap(), "output": "note.txt" }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("deny tier"),
            "{v}"
        );
        assert!(!ws.path().join("note.txt").exists());
    }

    /// file_convert is outside execute_tool's approval pre-pass, so a prompt-tier
    /// path was never put to a human. It must refuse, and say why.
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_prompt_tier_refused_by_name() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let src = outside.path().join("note.md");
        fs::write(&src, "x").unwrap();
        let fp = convert_policy(
            true,
            vec![(ws.path(), T::Write), (outside.path(), T::Prompt)],
        );
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": src.to_str().unwrap(), "output": "note.txt" }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("cannot raise an approval prompt"),
            "{v}"
        );
        assert!(!ws.path().join("note.txt").exists());
    }

    /// The output is resolved as a WRITE: an explicit output into a read-tier
    /// directory is refused and nothing is written there.
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_output_into_read_tier_refused() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let ro = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let fp = convert_policy(true, vec![(ws.path(), T::Write), (ro.path(), T::Read)]);
        let dest = ro.path().join("note.txt");
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": "note.md", "output": dest.to_str().unwrap() }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.starts_with("output path"), "{msg}");
        assert!(msg.contains("Choose an output path you can write"), "{msg}");
        assert!(!dest.exists());
    }

    /// Output into a write-tier directory outside the workspace is allowed.
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_output_into_write_tier_outside_workspace() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let rw = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "body").unwrap();
        let fp = convert_policy(true, vec![(ws.path(), T::Write), (rw.path(), T::Write)]);
        let dest = rw.path().join("note.txt");
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": "note.md", "output": dest.to_str().unwrap() }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["ok"], serde_json::json!(true), "envelope: {v}");
        assert_eq!(fs::read_to_string(&dest).unwrap(), "body");
    }

    /// No policy: legacy clamp, but the refusal now says what to do instead of
    /// a bare "rejected" (the missing signpost from 2026-09-16).
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_no_policy_outside_workspace_signposts() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let src = outside.path().join("note.md");
        fs::write(&src, "x").unwrap();
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": src.to_str().unwrap(), "output": "note.txt" }),
            ws.path(),
            None,
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.starts_with("input path"), "{msg}");
        assert!(msg.contains("no active file_policy"), "{msg}");
        assert!(msg.contains("Copy the file in"), "{msg}");
    }

    /// A present-but-disabled policy is inert: its rules must not widen access,
    /// exactly as for file_read.
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_disabled_policy_keeps_workspace_clamp() {
        use openfang_types::config::FileAccessTier as T;
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let src = outside.path().join("note.md");
        fs::write(&src, "x").unwrap();
        let fp = convert_policy(false, vec![(outside.path(), T::Write)]);
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": src.to_str().unwrap(), "output": "note.txt" }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        assert!(!ws.path().join("note.txt").exists());
    }

    /// The sensitive-path floor runs under an active policy too: a Write rule
    /// over a directory cannot expose a protected file inside it.
    #[cfg(unix)]
    #[tokio::test]
    async fn convert_policy_cannot_open_mcp_auth() {
        use openfang_types::config::FileAccessTier as T;
        let Some(h) = std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .and_then(|h| h.canonicalize().ok())
        else {
            return;
        };
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let fp = convert_policy(true, vec![(ws.path(), T::Write), (h.as_path(), T::Write)]);
        let target = h.join(".mcp-auth").join("probe.md");
        let v = convert_with(
            serde_json::json!({ "format": "txt", "input": target.to_str().unwrap(), "output": "p.txt" }),
            ws.path(),
            Some(&fp),
            home.path(),
        )
        .await;
        assert_eq!(v["error"]["code"], "BAD_PATH", "envelope: {v}");
        assert!(!ws.path().join("p.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_input_traversal_blocked() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "../escape.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(parse(&out)["error"]["code"], "BAD_PATH", "envelope: {out}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_output_traversal_blocked() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "output": "../escape.txt" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(parse(&out)["error"]["code"], "BAD_PATH", "envelope: {out}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_absolute_input_rejected() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "/tmp/outside-abs.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(parse(&out)["error"]["code"], "BAD_PATH", "envelope: {out}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_absolute_output_rejected() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "output": "/tmp/escape.txt" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(parse(&out)["error"]["code"], "BAD_PATH", "envelope: {out}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_unknown_format_rejected() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "pdf", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(false));
        assert_eq!(v["error"]["code"], "UNKNOWN_FORMAT", "envelope: {out}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_missing_dep_no_partial_run() {
        let home = hermetic_home("totally-absent-binary-xyz");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(false));
        assert_eq!(v["error"]["code"], "MISSING_DEP", "envelope: {out}");
        // No partial run: the stub never executed, so no output file exists.
        assert!(!ws.path().join("note.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_injection_reaches_recipe_as_literal_arg() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        // A filename loaded with shell metacharacters. If ANY layer evaluated it
        // through a shell, `pwned`/`pwned2` would be created.
        let evil = "evil; $(touch pwned) `touch pwned2`.md";
        fs::write(ws.path().join(evil), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": evil }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(true), "envelope: {out}");
        // No shell expansion occurred anywhere in the pipeline.
        assert!(!ws.path().join("pwned").exists(), "injection executed!");
        assert!(!ws.path().join("pwned2").exists(), "injection executed!");
        // The stub received the metachar path as ONE literal argv entry.
        let expected_out = Path::new(evil).with_extension("txt");
        assert!(ws.path().join(&expected_out).is_file());
        let mut argv_log = ws.path().join(&expected_out).into_os_string();
        argv_log.push(".argv");
        let logged = fs::read_to_string(&argv_log).unwrap();
        assert!(
            logged.lines().any(|l| l.ends_with(evil)),
            "metachar path not a single literal arg; argv log: {logged}"
        );
    }

    #[test]
    fn convert_ok_envelope_shape() {
        let s = convert_ok("pdf", "out/doc.pdf");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], serde_json::json!(true));
        assert_eq!(v["format"], "pdf");
        assert_eq!(v["output_path"], "out/doc.pdf");
        assert!(v.get("error").is_none());
    }

    #[test]
    fn convert_err_envelope_shape() {
        let s = convert_err("pdf", "MISSING_DEP", "needs 'typst'");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], serde_json::json!(false));
        assert_eq!(v["format"], "pdf");
        assert_eq!(v["error"]["code"], "MISSING_DEP");
        assert_eq!(v["error"]["message"], "needs 'typst'");
        assert!(v.get("output_path").is_none());
    }

    #[cfg(unix)]
    fn hermetic_home_presets() -> TempDir {
        use std::os::unix::fs::PermissionsExt;
        let home = TempDir::new().unwrap();
        let scripts = home.path().join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        let stub = scripts.join("stub.sh");
        fs::write(
            &stub,
            "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > \"$2.argv\"\ncp \"$1\" \"$2\"\n",
        )
        .unwrap();
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
        let convert_dir = home.path().join("convert");
        fs::create_dir_all(&convert_dir).unwrap();
        fs::write(
            convert_dir.join("recipes.toml"),
            "[[recipe]]\nfrom = \"md\"\nto = \"txt\"\nargv = [\"{script}/stub.sh\", \"{input}\", \"{output}\", \"{viewport}\"]\nout_ext = \"txt\"\ndefault_preset = \"desktop\"\n\n[recipe.presets.mobile]\nviewport = \"MOBILEVP\"\n\n[recipe.presets.desktop]\nviewport = \"DESKTOPVP\"\n",
        )
        .unwrap();
        home
    }

    #[cfg(unix)]
    fn read_argv_log(ws: &Path, out_rel: &str) -> String {
        let mut p = ws.join(out_rel).into_os_string();
        p.push(".argv");
        fs::read_to_string(&p).unwrap()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_preset_selects_named_vars() {
        let home = hermetic_home_presets();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "preset": "mobile" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["ok"],
            serde_json::json!(true),
            "envelope: {out}"
        );
        let log = read_argv_log(ws.path(), "note.txt");
        assert!(log.lines().any(|l| l == "MOBILEVP"), "argv log: {log}");
        assert!(!log.lines().any(|l| l == "DESKTOPVP"), "argv log: {log}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_preset_omitted_uses_default() {
        let home = hermetic_home_presets();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["ok"],
            serde_json::json!(true),
            "envelope: {out}"
        );
        let log = read_argv_log(ws.path(), "note.txt");
        assert!(log.lines().any(|l| l == "DESKTOPVP"), "argv log: {log}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_unknown_preset_rejected() {
        let home = hermetic_home_presets();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "preset": "phone" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(false));
        assert_eq!(v["error"]["code"], "UNKNOWN_PRESET", "envelope: {out}");
        assert!(!ws.path().join("note.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_preset_on_presetless_recipe_rejected() {
        let home = hermetic_home("");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "preset": "mobile" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["error"]["code"],
            "UNKNOWN_PRESET",
            "envelope: {out}"
        );
    }

    // ------------------------------------------------------------------
    // ANAI-131: caller-supplied option resolution
    // ------------------------------------------------------------------

    /// Hermetic home whose md->txt recipe declares an enum option
    /// (`orientation`, default portrait), a string option (`font_body`,
    /// default ""), AND a preset (`viewport`) — so a single test can prove
    /// options and presets resolve into the same argv without collision.
    #[cfg(unix)]
    fn hermetic_home_options() -> TempDir {
        use std::os::unix::fs::PermissionsExt;
        let home = TempDir::new().unwrap();
        let scripts = home.path().join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        let stub = scripts.join("stub.sh");
        fs::write(
            &stub,
            "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > \"$2.argv\"\ncp \"$1\" \"$2\"\n",
        )
        .unwrap();
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
        let convert_dir = home.path().join("convert");
        fs::create_dir_all(&convert_dir).unwrap();
        fs::write(
            convert_dir.join("recipes.toml"),
            "[[recipe]]\nfrom = \"md\"\nto = \"txt\"\nargv = [\"{script}/stub.sh\", \"{input}\", \"{output}\", \"--orient\", \"{orientation}\", \"--font\", \"{font_body}\", \"--vp\", \"{viewport}\"]\nout_ext = \"txt\"\ndefault_preset = \"desktop\"\n\n[recipe.options.orientation]\ntype = \"enum\"\nvalues = [\"portrait\", \"landscape\"]\ndefault = \"portrait\"\ndesc = \"Page orientation\"\n\n[recipe.options.font_body]\ntype = \"string\"\ndefault = \"\"\ndesc = \"Body font\"\n\n[recipe.presets.mobile]\nviewport = \"MOBILEVP\"\n\n[recipe.presets.desktop]\nviewport = \"DESKTOPVP\"\n",
        )
        .unwrap();
        home
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_unknown_option_rejected() {
        let home = hermetic_home_options();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "options": { "bogus": "1" } }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(false));
        assert_eq!(v["error"]["code"], "UNKNOWN_OPTION", "envelope: {out}");
        // The message teaches: it lists the valid option names.
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("orientation"), "message: {msg}");
        assert!(!ws.path().join("note.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_invalid_enum_option_rejected() {
        let home = hermetic_home_options();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md", "options": { "orientation": "sideways" } }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(false));
        assert_eq!(v["error"]["code"], "INVALID_OPTION", "envelope: {out}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("portrait") && msg.contains("landscape"),
            "message: {msg}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_omitted_options_apply_defaults() {
        let home = hermetic_home_options();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["ok"],
            serde_json::json!(true),
            "envelope: {out}"
        );
        let log = read_argv_log(ws.path(), "note.txt");
        // Enum default applied; preset default applied; no token survives.
        assert!(log.lines().any(|l| l == "portrait"), "argv log: {log}");
        assert!(log.lines().any(|l| l == "DESKTOPVP"), "argv log: {log}");
        assert!(!log.contains('{'), "unresolved token in argv log: {log}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_options_and_preset_resolve_together() {
        let home = hermetic_home_options();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({
                "format": "txt",
                "input": "note.md",
                "preset": "mobile",
                "options": { "orientation": "landscape", "font_body": "Iosevka" }
            }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["ok"],
            serde_json::json!(true),
            "envelope: {out}"
        );
        let log = read_argv_log(ws.path(), "note.txt");
        // Caller option value overrides the default; string option flows through;
        // the chosen preset var lands in the same argv.
        assert!(log.lines().any(|l| l == "landscape"), "argv log: {log}");
        assert!(log.lines().any(|l| l == "Iosevka"), "argv log: {log}");
        assert!(log.lines().any(|l| l == "MOBILEVP"), "argv log: {log}");
        assert!(
            !log.lines().any(|l| l == "portrait"),
            "default leaked: {log}"
        );
    }

    // ------------------------------------------------------------------
    // ANAI-286 (spawn timeout) / ANAI-287 (dependency checking)
    // ------------------------------------------------------------------

    /// Hermetic home with a caller-supplied stub body and recipe extras, so a
    /// test can pin the launcher's exit code / runtime and the recipe's
    /// `timeout_secs` / `needs_files` independently.
    #[cfg(unix)]
    fn hermetic_home_custom(stub_body: &str, recipe_extra: &str, config_body: &str) -> TempDir {
        use std::os::unix::fs::PermissionsExt;
        let home = TempDir::new().unwrap();
        let scripts = home.path().join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        let stub = scripts.join("stub.sh");
        fs::write(&stub, stub_body).unwrap();
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
        let convert_dir = home.path().join("convert");
        fs::create_dir_all(&convert_dir).unwrap();
        fs::write(
            convert_dir.join("recipes.toml"),
            format!(
                "[[recipe]]\nfrom = \"md\"\nto = \"txt\"\nargv = [\"{{script}}/stub.sh\", \"{{input}}\", \"{{output}}\"]\nout_ext = \"txt\"\n{recipe_extra}"
            ),
        )
        .unwrap();
        if !config_body.is_empty() {
            fs::write(home.path().join("config.toml"), config_body).unwrap();
        }
        home
    }

    #[cfg(unix)]
    const SLEEPY_STUB: &str = "#!/usr/bin/env bash\nsleep 30\ncp \"$1\" \"$2\"\n";

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_timeout_kills_a_hanging_launcher() {
        // Before ANAI-286 this awaited `cmd.output()` unconditionally, so this
        // test would hang for 30s and then PASS as a successful conversion.
        let home = hermetic_home_custom(SLEEPY_STUB, "timeout_secs = 1\n", "");
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["ok"], serde_json::json!(false), "envelope: {out}");
        assert_eq!(v["error"]["code"], "CONVERT_TIMEOUT", "envelope: {out}");
        // Distinct from CONVERT_FAILED on purpose: the retries differ.
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("1s deadline"), "message: {msg}");
        // The launcher never reached its `cp`, so no output was produced.
        assert!(!ws.path().join("note.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_timeout_clamped_by_operator_policy_discloses_both_numbers() {
        // Recipe asks 9s; the operator ceiling is 1s. The refusal must state
        // what was requested AND what was enforced -- a silently shortened
        // conversion is indistinguishable from one that simply ran long.
        let home = hermetic_home_custom(
            SLEEPY_STUB,
            "timeout_secs = 9\n",
            "[convert]\nmax_timeout_secs = 1\n",
        );
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["error"]["code"], "CONVERT_TIMEOUT", "envelope: {out}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("1s deadline"), "enforced missing: {msg}");
        assert!(msg.contains("requested 9s"), "requested missing: {msg}");
        assert!(msg.contains("CLAMPED"), "clamp undisclosed: {msg}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_within_its_deadline_is_untouched() {
        // The satisfied case must be invisible: a recipe that finishes inside
        // its bound behaves exactly as it did before the deadline existed.
        let home = hermetic_home_custom(
            "#!/usr/bin/env bash\ncp \"$1\" \"$2\"\n",
            "timeout_secs = 30\n",
            "",
        );
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["ok"],
            serde_json::json!(true),
            "envelope: {out}"
        );
        assert!(ws.path().join("note.txt").is_file());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_exit_code_3_is_missing_dep_not_convert_failed() {
        // ANAI-287 option 1: only the interpreter that runs the conversion can
        // truthfully answer "is this module importable", so it answers with a
        // reserved code and we classify from that.
        let home = hermetic_home_custom(
            "#!/usr/bin/env bash\necho \"No module named 'pdfplumber'\" >&2\nexit 3\n",
            "",
            "",
        );
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["error"]["code"], "MISSING_DEP", "envelope: {out}");
        // The launcher's own diagnosis is carried through, not swallowed.
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("pdfplumber"), "message: {msg}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_other_nonzero_exit_is_still_convert_failed() {
        // Negative control for the test above: 3 is reserved, everything else
        // keeps its old classification.
        let home = hermetic_home_custom(
            "#!/usr/bin/env bash\necho 'bad input' >&2\nexit 1\n",
            "",
            "",
        );
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["error"]["code"],
            "CONVERT_FAILED",
            "envelope: {out}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_missing_needs_file_refuses_before_any_spawn() {
        // The stub WOULD produce output. The absent output file is the
        // observable proof that the preflight ran ahead of the spawn, rather
        // than the refusal being asserted by a comment.
        let home = hermetic_home_custom(
            "#!/usr/bin/env bash\ncp \"$1\" \"$2\"\n",
            "needs_files = [\"/definitely/not/here/pdfplumber/__init__.py\"]\n",
            "",
        );
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        let v = parse(&out);
        assert_eq!(v["error"]["code"], "MISSING_DEP", "envelope: {out}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("pdfplumber"), "message names nothing: {msg}");
        assert!(!ws.path().join("note.txt").exists(), "launcher was spawned");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn convert_satisfied_needs_file_is_invisible() {
        let home = hermetic_home_custom("#!/usr/bin/env bash\ncp \"$1\" \"$2\"\n", "", "");
        // Point needs_files at the stub itself: guaranteed to exist, and proves
        // a satisfied check changes nothing about the happy path.
        let marker = home.path().join("scripts").join("stub.sh");
        let convert_dir = home.path().join("convert");
        fs::write(
            convert_dir.join("recipes.toml"),
            format!(
                "[[recipe]]\nfrom = \"md\"\nto = \"txt\"\nargv = [\"{{script}}/stub.sh\", \"{{input}}\", \"{{output}}\"]\nout_ext = \"txt\"\nneeds_files = [\"{}\"]\n",
                marker.display()
            ),
        )
        .unwrap();
        let ws = TempDir::new().unwrap();
        fs::write(ws.path().join("note.md"), "x").unwrap();
        let out = tool_file_convert_in(
            &serde_json::json!({ "format": "txt", "input": "note.md" }),
            Some(ws.path()),
            home.path(),
        )
        .await
        .unwrap();
        assert_eq!(
            parse(&out)["ok"],
            serde_json::json!(true),
            "envelope: {out}"
        );
        assert!(ws.path().join("note.txt").is_file());
    }

    #[test]
    fn find_on_path_locates_executable() {
        #[cfg(unix)]
        {
            let path = std::ffi::OsString::from("/usr/bin:/bin");
            assert!(find_on_path("sh", &path).is_some());
            assert!(find_on_path("definitely-not-a-real-binary-xyz", &path).is_none());
        }
    }
}
/// One-shot authority minted by the wake-consumer (`run_woken_agent_loop`) for
/// the lifetime of a single woken turn, authorizing exactly ONE terminal reply
/// back to the agent that initiated the wake (ANAI-122, leg 3 of the four-step
/// round-trip: fleet -> origin).
///
/// This is the entire capability model for `agent_reply_async`. The tool is
/// advertised fleet-wide (it sits in `DEFAULT_ALLOWED`, not
/// `PRIVILEGED_DEFAULT_DENY`), but is INERT without this token in task-local
/// scope. The token names exactly one lawful target — the initiator, captured
/// as [`WakeEnvelope::sender`](openfang_types::wake::WakeEnvelope) — and is
/// consumed on first use. So a woken agent can answer its initiator once and
/// cannot originate a wake to anyone else: no standing grant, no new fan-out
/// surface, and the aggregate emission ceiling still bounds the total. Not
/// minted for a turn that was itself woken by a reply, which makes the reply
/// strictly terminal and closes the reply-bounce (A replies to B, B's turn
/// gets no right, so it cannot reply back to A).
#[derive(Debug, Clone)]
pub struct ReplyRight {
    reply_to: String,
    correlation: String,
    /// Surfacing route carried inbound on the originating wake (ANAI-123).
    /// Baked in at mint time so `agent_reply_async` copies it onto the terminal
    /// reply envelope without the callee ever seeing or choosing it — origin's
    /// leg-4 turn then auto-posts there. `None` when the origin dispatched with
    /// no `surface_to`.
    surface_to: Option<String>,
}

impl ReplyRight {
    /// Mint a reply-right authorizing one reply to `reply_to` (the initiator),
    /// tagged with the inbound wake's `correlation` id (its task id) for audit
    /// and provenance.
    ///
    /// `surface_to` is the inbound wake's surfacing route (ANAI-123), carried
    /// so the terminal reply inherits it — see [`Self::surface_to`].
    pub fn new(
        reply_to: impl Into<String>,
        correlation: impl Into<String>,
        surface_to: Option<String>,
    ) -> Self {
        Self {
            reply_to: reply_to.into(),
            correlation: correlation.into(),
            surface_to,
        }
    }

    /// The single lawful reply target — the initiator that woke this turn.
    pub fn reply_to(&self) -> &str {
        &self.reply_to
    }

    /// The inbound wake's correlation id (its task id).
    pub fn correlation(&self) -> &str {
        &self.correlation
    }

    /// The inbound surfacing route (ANAI-123), if the origin dispatched one.
    /// Copied onto the terminal reply envelope so origin's leg-4 woken turn
    /// emits exactly one `channel_send(surface_to, ...)`.
    pub fn surface_to(&self) -> Option<&str> {
        self.surface_to.as_deref()
    }
}

// ---------------------------------------------------------------------------
// image_read (ANAI-297)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod image_read_tests {
    use super::*;
    use base64::Engine;
    use openfang_types::media::ImageReadLimit;

    const PNG_SIG: &[u8] = b"\x89PNG\r\n\x1a\n";

    fn limit(max_bytes: u64) -> ImageReadLimit {
        ImageReadLimit {
            max_bytes,
            clamped_from: None,
        }
    }

    fn with_sig(sig: &[u8], len: usize) -> Vec<u8> {
        let mut v = sig.to_vec();
        v.resize(len.max(sig.len()), 0xAB);
        v
    }

    fn webp(len: usize) -> Vec<u8> {
        let mut v = b"RIFF\x00\x00\x00\x00WEBPVP8 ".to_vec();
        v.resize(len.max(v.len()), 0x11);
        v
    }

    async fn read_in_sink(
        root: &Path,
        path: &str,
        max_bytes: u64,
    ) -> (Result<String, String>, Vec<ToolImage>) {
        with_image_sink(tool_image_read(
            &serde_json::json!({ "path": path }),
            Some(root),
            None,
            None,
            limit(max_bytes),
        ))
        .await
    }

    #[tokio::test]
    async fn returns_the_bytes_on_disk_as_exactly_one_image() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = with_sig(PNG_SIG, 300);
        std::fs::write(dir.path().join("a.png"), &bytes).unwrap();

        let (res, images) = read_in_sink(dir.path(), "a.png", 1_000_000).await;
        let text = res.expect("a PNG inside the workspace must read");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type, "image/png");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&images[0].data_base64)
            .unwrap();
        assert_eq!(
            decoded, bytes,
            "the image must be the file's bytes, unaltered"
        );
        assert!(text.contains("image/png"), "{text}");
        assert!(text.contains("300 bytes"), "{text}");
        assert!(text.contains("sha256 "), "{text}");
    }

    /// The name is a claim; the bytes are the fact. A mislabelled file must be
    /// sent with its real type, or the provider rejects it after the upload.
    #[tokio::test]
    async fn the_type_comes_from_the_bytes_not_the_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("really-jpeg.png"),
            with_sig(b"\xff\xd8\xff\xe0", 64),
        )
        .unwrap();
        std::fs::write(dir.path().join("really-webp.jpg"), webp(64)).unwrap();
        std::fs::write(dir.path().join("really-gif.bin"), with_sig(b"GIF89a", 64)).unwrap();

        for (path, mime) in [
            ("really-jpeg.png", "image/jpeg"),
            ("really-webp.jpg", "image/webp"),
            ("really-gif.bin", "image/gif"),
        ] {
            let (res, images) = read_in_sink(dir.path(), path, 1_000_000).await;
            res.unwrap_or_else(|e| panic!("{path}: {e}"));
            assert_eq!(images[0].mime_type, mime, "{path}");
        }
    }

    /// The load-bearing refusal. Outside the bridge nothing can carry the
    /// image, and a text-only success would read as "I looked at it".
    #[tokio::test]
    async fn refuses_on_a_text_only_path_instead_of_succeeding_without_the_image() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), with_sig(PNG_SIG, 64)).unwrap();

        let err = tool_image_read(
            &serde_json::json!({ "path": "a.png" }),
            Some(dir.path()),
            None,
            None,
            limit(1_000_000),
        )
        .await
        .expect_err("no sink in scope: must refuse, not succeed");
        assert!(err.contains("text-only path"), "{err}");
        assert!(err.contains("Nothing was returned"), "{err}");
    }

    #[tokio::test]
    async fn over_the_limit_is_refused_whole_and_at_the_limit_passes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("big.png"), with_sig(PNG_SIG, 301)).unwrap();
        std::fs::write(dir.path().join("fits.png"), with_sig(PNG_SIG, 300)).unwrap();

        let (res, images) = read_in_sink(dir.path(), "big.png", 300).await;
        let err = res.expect_err("one byte over must refuse");
        assert!(err.contains("301 bytes"), "{err}");
        assert!(
            err.contains("image_read_max_bytes"),
            "names the setting: {err}"
        );
        assert!(images.is_empty(), "a refused read must deposit nothing");

        let (res, images) = read_in_sink(dir.path(), "fits.png", 300).await;
        res.expect("exactly at the limit must pass");
        assert_eq!(images.len(), 1);
    }

    #[tokio::test]
    async fn non_images_are_refused_by_name_and_deposit_nothing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("logo.svg"),
            b"  <svg xmlns=\"http://www.w3.org/2000/svg\"></svg>",
        )
        .unwrap();
        std::fs::write(dir.path().join("doc.png"), b"%PDF-1.7 not an image").unwrap();
        std::fs::write(dir.path().join("notes.png"), b"just some text").unwrap();

        let cases = [
            ("logo.svg", "file_read"),
            ("doc.png", "PDF"),
            ("notes.png", "not its name"),
        ];
        for (path, needle) in cases {
            let (res, images) = read_in_sink(dir.path(), path, 1_000_000).await;
            let err = res.expect_err(path);
            assert!(err.contains(needle), "{path}: {err}");
            assert!(images.is_empty(), "{path}: deposited an image on refusal");
        }
    }

    #[tokio::test]
    async fn empty_files_and_directories_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.png"), b"").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let (res, _) = read_in_sink(dir.path(), "empty.png", 1_000_000).await;
        assert!(res.unwrap_err().contains("empty"));
        let (res, _) = read_in_sink(dir.path(), "sub", 1_000_000).await;
        assert!(res.unwrap_err().contains("directory"));
    }

    /// The companion-grant admission condition, as behaviour: image_read
    /// reaches nothing file_read would refuse, because it asks the same
    /// resolver the same question.
    #[tokio::test]
    async fn reaches_nothing_file_read_would_refuse() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("secret.png");
        std::fs::write(&foreign, with_sig(PNG_SIG, 64)).unwrap();
        let foreign = foreign.to_str().unwrap().to_string();

        for path in [foreign.as_str(), "../secret.png"] {
            let read = tool_file_read(
                &serde_json::json!({ "path": path }),
                Some(workspace.path()),
                None,
                None,
            )
            .await;
            assert!(read.is_err(), "precondition: file_read refuses {path}");

            let (res, images) = read_in_sink(workspace.path(), path, 1_000_000).await;
            assert!(res.is_err(), "image_read must refuse {path} too");
            assert!(images.is_empty(), "{path}: deposited an image on refusal");
        }
    }

    /// End to end through `execute_tool`: the dispatch arm exists, the
    /// compiled-default limit applies with no media engine, and file_read on
    /// the same file points at image_read instead of dead-ending.
    #[tokio::test]
    async fn execute_tool_dispatches_it_and_file_read_signposts_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("shot.png"), with_sig(PNG_SIG, 128)).unwrap();

        let (result, images) = with_image_sink(execute_tool(
            "t",
            "image_read",
            &serde_json::json!({ "path": "shot.png" }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(dir.path()),
            None, // media_engine
            None, // exec_policy
            None, // file_policy
            None, // tts_engine
            None, // docker_config
            None, // process_manager
            None, // origin
        ))
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(images.len(), 1);

        let read = execute_tool(
            "t",
            "file_read",
            &serde_json::json!({ "path": "shot.png" }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(dir.path()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await;
        assert!(read.is_error);
        assert!(
            read.content.contains("image_read(path=\"shot.png\")"),
            "file_read on an image must name the call that works: {}",
            read.content
        );
    }

    #[test]
    fn webp_is_recognised_by_the_file_read_signpost_too() {
        assert_eq!(sniff_binary_format(&webp(16)), Some("WebP"));
        assert_eq!(sniff_image_mime(&webp(16)), Some("image/webp"));
        // RIFF alone is WAV/AVI territory, not an image.
        let wav = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        assert_eq!(sniff_image_mime(wav), None);
    }
}
