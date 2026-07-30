//! Turn-watchdog contract shared across crates (ANAI-109).
//!
//! A turn can wedge in two distinct ways and they want opposite recovery:
//!
//! * **Provider stall** — the LLM endpoint accepted the request but stopped
//!   making progress (hung socket, overloaded backend that never closes the
//!   stream). Re-issuing against the same endpoint just re-stalls, so this is
//!   **terminal**: the agent loop returns an `Err` (it does *not* self-inject a
//!   tool error and re-infer), and workflow `ErrorMode::Retry` must refuse to
//!   re-run the step.
//!
//! Because the error surfaces across crate boundaries as a flattened `String`
//! (`OpenFangError::LlmDriver(String)` → workflow's `Result<_, String>` →
//! channel `sanitize_agent_error`), the discriminator travels as a stable
//! substring marker rather than a typed variant. This const is the single
//! source of truth for that marker so producers and matchers cannot drift.
//!
//! Producers: `openfang-runtime::agent_loop` (LLM-call timeout wrappers).
//! Matchers:  `openfang-kernel::workflow` (retry gate),
//!            `openfang-channels::bridge` (`sanitize_agent_error`).

/// Stable marker embedded in the error string when the turn watchdog aborts an
/// LLM call because the provider stopped responding. Matched (case-insensitively
/// where callers lowercase) to gate retry/re-inference and to render a clean
/// user-facing message. Treat as a wire contract — changing it requires updating
/// every matcher listed in this module's docs.
pub const PROVIDER_STALL_MARKER: &str = "provider unresponsive (watchdog)";

/// Default ceiling, in seconds, for a single LLM call (`complete`/`stream`)
/// before the turn watchdog aborts it as a provider stall. A single completion
/// should land well inside a few minutes even on a heavy model, so this is a
/// tight-ish backstop, not a latency SLA. Operators tune it via the
/// `[watchdog] llm_call_timeout_secs` config knob or the
/// `OPENFANG_LLM_CALL_TIMEOUT_SECS` env var; read through [`llm_call_timeout_secs`].
pub const LLM_CALL_TIMEOUT_SECS: u64 = 240;

/// Default idle window, in seconds, for a *streaming* LLM call: the maximum
/// gap between two consecutive stream events (text deltas, tool events, the
/// terminal usage event) before the turn watchdog treats the stream as a dead
/// provider and aborts it.
///
/// Unlike [`LLM_CALL_TIMEOUT_SECS`] — a *total* wall-clock ceiling that cannot
/// tell a slow-but-alive turn from a hung one — this resets on every event, so
/// a long turn that keeps streaming survives indefinitely while a genuinely
/// silent socket dies within one idle window. With it in place the absolute
/// ceiling can be raised for long turns without re-opening the
/// "hang for the whole ceiling" hole (ANAI-114). Tune via `[watchdog]
/// stream_idle_timeout_secs` or `OPENFANG_STREAM_IDLE_TIMEOUT_SECS`; read
/// through [`stream_idle_timeout_secs`].
pub const STREAM_IDLE_TIMEOUT_SECS: u64 = 180;

/// Default ceiling, in seconds, for establishing the TCP/TLS connection to a
/// provider. Bounds the connect phase only — never the response body — so it is
/// safe for long-lived streaming responses.
pub const PROVIDER_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Default ceiling, in seconds, for a single MCP tool call before the per-tool
/// execution timeout aborts it. MCP calls — especially remote SSE transports
/// like `mcp-remote` — can wedge differently than local tools, so they get a
/// knob distinct from the generic tool ceiling. Tune via the
/// `[watchdog] mcp_tool_timeout_secs` config knob or `OPENFANG_MCP_TIMEOUT_SECS`;
/// read through [`mcp_tool_timeout_secs`]. `0` disables the bound entirely.
pub const MCP_TOOL_TIMEOUT_SECS: u64 = 120;

/// Lower bound for the resolved LLM-call ceiling. Guards against a fat-fingered
/// `0` or single-digit value silently turning the backstop into a turn-killer.
const LLM_CALL_TIMEOUT_FLOOR_SECS: u64 = 30;

/// Lower bound for the resolved stream idle window. Must comfortably exceed the
/// longest legitimate gap between stream events — notably an in-subprocess MCP
/// tool call, which emits no stdout events while it runs — so the idle watchdog
/// never false-kills a live-but-quiet turn. Guards against a fat-fingered tiny
/// value (ANAI-114).
const STREAM_IDLE_TIMEOUT_FLOOR_SECS: u64 = 30;

/// Operator-configured watchdog ceilings, sourced from the `[watchdog]` config
/// section and installed once at kernel boot via [`install_timeouts`].
#[derive(Clone, Copy, Debug)]
pub struct WatchdogTimeouts {
    /// Seconds for a single LLM `complete`/`stream` call.
    pub llm_call_timeout_secs: u64,
    /// Seconds for a single MCP tool call (`0` = unbounded).
    pub mcp_tool_timeout_secs: u64,
    /// Seconds of stream silence (no events) before a streaming call is aborted
    /// as a provider stall. Resets on every event — see
    /// [`STREAM_IDLE_TIMEOUT_SECS`].
    pub stream_idle_timeout_secs: u64,
}

static INSTALLED: std::sync::OnceLock<WatchdogTimeouts> = std::sync::OnceLock::new();

/// Install operator-configured watchdog ceilings. First writer wins; later
/// calls are ignored (the kernel installs exactly once at boot). Per-call env
/// overrides still take precedence over whatever is installed here.
pub fn install_timeouts(t: WatchdogTimeouts) {
    let _ = INSTALLED.set(t);
}

fn env_u64(var: &str) -> Option<u64> {
    std::env::var(var).ok().and_then(|s| s.trim().parse().ok())
}

/// Resolved ceiling, in seconds, for a single LLM `complete`/`stream` call.
///
/// Precedence: `OPENFANG_LLM_CALL_TIMEOUT_SECS` env var > installed `[watchdog]`
/// config > compiled default ([`LLM_CALL_TIMEOUT_SECS`]). Clamped up to
/// [`LLM_CALL_TIMEOUT_FLOOR_SECS`] so the backstop can never be tuned into a
/// turn-killer. Always on — there is no disable path; raise the number for slow
/// local inference (vLLM on old GPUs).
pub fn llm_call_timeout_secs() -> u64 {
    env_u64("OPENFANG_LLM_CALL_TIMEOUT_SECS")
        .or_else(|| INSTALLED.get().map(|t| t.llm_call_timeout_secs))
        .unwrap_or(LLM_CALL_TIMEOUT_SECS)
        .max(LLM_CALL_TIMEOUT_FLOOR_SECS)
}

/// Resolved ceiling, in seconds, for a single MCP tool call.
///
/// Precedence: `OPENFANG_MCP_TIMEOUT_SECS` env var > installed `[watchdog]`
/// config > compiled default ([`MCP_TOOL_TIMEOUT_SECS`]). A value of `0` means
/// "no timeout"; the caller is responsible for honoring that, matching the
/// generic tool-timeout opt-out (issue #1125).
pub fn mcp_tool_timeout_secs() -> u64 {
    env_u64("OPENFANG_MCP_TIMEOUT_SECS")
        .or_else(|| INSTALLED.get().map(|t| t.mcp_tool_timeout_secs))
        .unwrap_or(MCP_TOOL_TIMEOUT_SECS)
}

/// Resolved idle window, in seconds, between events on a streaming LLM call
/// before the turn watchdog aborts it as a provider stall.
///
/// Precedence: `OPENFANG_STREAM_IDLE_TIMEOUT_SECS` env var > installed
/// `[watchdog]` config > compiled default ([`STREAM_IDLE_TIMEOUT_SECS`]).
/// Clamped up to [`STREAM_IDLE_TIMEOUT_FLOOR_SECS`] so a tiny value can never
/// turn the idle watchdog into a false-kill on a live-but-quiet turn.
pub fn stream_idle_timeout_secs() -> u64 {
    env_u64("OPENFANG_STREAM_IDLE_TIMEOUT_SECS")
        .or_else(|| INSTALLED.get().map(|t| t.stream_idle_timeout_secs))
        .unwrap_or(STREAM_IDLE_TIMEOUT_SECS)
        .max(STREAM_IDLE_TIMEOUT_FLOOR_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All resolver env cases in one test to avoid env-var races across threads.
    /// `INSTALLED` is never set in this crate's test binary (only the kernel
    /// installs it), so the config layer falls through to the compiled defaults.
    #[test]
    fn test_timeout_resolvers() {
        // LLM: default, floor clamp, explicit override, unparseable fallback.
        std::env::remove_var("OPENFANG_LLM_CALL_TIMEOUT_SECS");
        assert_eq!(llm_call_timeout_secs(), 240);
        std::env::set_var("OPENFANG_LLM_CALL_TIMEOUT_SECS", "300");
        assert_eq!(llm_call_timeout_secs(), 300);
        std::env::set_var("OPENFANG_LLM_CALL_TIMEOUT_SECS", "5"); // below floor
        assert_eq!(llm_call_timeout_secs(), LLM_CALL_TIMEOUT_FLOOR_SECS);
        std::env::set_var("OPENFANG_LLM_CALL_TIMEOUT_SECS", "0"); // would-be disable
        assert_eq!(llm_call_timeout_secs(), LLM_CALL_TIMEOUT_FLOOR_SECS);
        std::env::set_var("OPENFANG_LLM_CALL_TIMEOUT_SECS", "junk");
        assert_eq!(llm_call_timeout_secs(), 240);
        std::env::remove_var("OPENFANG_LLM_CALL_TIMEOUT_SECS");

        // MCP: default, 0 = unbounded (preserved verbatim), explicit override.
        std::env::remove_var("OPENFANG_MCP_TIMEOUT_SECS");
        assert_eq!(mcp_tool_timeout_secs(), 120);
        std::env::set_var("OPENFANG_MCP_TIMEOUT_SECS", "0");
        assert_eq!(mcp_tool_timeout_secs(), 0);
        std::env::set_var("OPENFANG_MCP_TIMEOUT_SECS", "90");
        assert_eq!(mcp_tool_timeout_secs(), 90);
        std::env::remove_var("OPENFANG_MCP_TIMEOUT_SECS");

        // Stream idle: default, explicit override, floor clamp, unparseable.
        std::env::remove_var("OPENFANG_STREAM_IDLE_TIMEOUT_SECS");
        assert_eq!(stream_idle_timeout_secs(), 180);
        std::env::set_var("OPENFANG_STREAM_IDLE_TIMEOUT_SECS", "300");
        assert_eq!(stream_idle_timeout_secs(), 300);
        std::env::set_var("OPENFANG_STREAM_IDLE_TIMEOUT_SECS", "5"); // below floor
        assert_eq!(stream_idle_timeout_secs(), STREAM_IDLE_TIMEOUT_FLOOR_SECS);
        std::env::set_var("OPENFANG_STREAM_IDLE_TIMEOUT_SECS", "junk");
        assert_eq!(stream_idle_timeout_secs(), 180);
        std::env::remove_var("OPENFANG_STREAM_IDLE_TIMEOUT_SECS");
    }

    #[test]
    fn test_compiled_defaults() {
        assert_eq!(LLM_CALL_TIMEOUT_SECS, 240);
        assert_eq!(MCP_TOOL_TIMEOUT_SECS, 120);
        assert_eq!(STREAM_IDLE_TIMEOUT_SECS, 180);
    }
}
