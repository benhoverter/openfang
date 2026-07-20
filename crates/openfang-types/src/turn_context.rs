//! Per-turn context envelope resolver (ANAI-128).
//!
//! The `[turn_context]` config section is global (not per-agent). The kernel
//! installs it once at boot via [`install`]; the runtime's agent loop reads the
//! resolved values through [`enabled`], [`roster_enabled`], and [`roster_limit`]
//! without threading a config handle through `run_agent_loop`'s signature (which
//! would ripple across ~15 call/test sites). Mirrors the `[watchdog]` install
//! pattern in [`crate::watchdog`].
//!
//! Per-value env overrides sit above the installed config so a smoke test or an
//! emergency kill can flip behavior without editing `config.toml`:
//!   * `OPENFANG_TURN_CONTEXT` — `0`/`off`/`false`/`no` force-disables; any
//!     other non-empty value force-enables.
//!   * `OPENFANG_TURN_CONTEXT_ROSTER` — same truthiness, overrides the roster.

use crate::config::TurnContextConfig;

static INSTALLED: std::sync::OnceLock<TurnContextConfig> = std::sync::OnceLock::new();

/// Install operator-configured turn-context settings. First writer wins; later
/// calls are ignored (the kernel installs exactly once at boot).
pub fn install(cfg: TurnContextConfig) {
    let _ = INSTALLED.set(cfg);
}

/// Parse a truthy/falsey env var. Returns `None` when unset or empty so the
/// caller falls through to the installed config / compiled default.
fn env_bool(var: &str) -> Option<bool> {
    let raw = std::env::var(var).ok()?;
    let v = raw.trim();
    if v.is_empty() {
        return None;
    }
    Some(!matches!(
        v.to_ascii_lowercase().as_str(),
        "0" | "off" | "false" | "no"
    ))
}

/// Whether the per-turn context envelope is injected.
///
/// Precedence: `OPENFANG_TURN_CONTEXT` env var > installed `[turn_context]`
/// config > compiled default (`true`).
pub fn enabled() -> bool {
    env_bool("OPENFANG_TURN_CONTEXT")
        .or_else(|| INSTALLED.get().map(|c| c.enabled))
        .unwrap_or(true)
}

/// Whether the multi-actor `recently_present` roster line is rendered.
///
/// Precedence: `OPENFANG_TURN_CONTEXT_ROSTER` env var > installed config >
/// compiled default (`false`).
pub fn roster_enabled() -> bool {
    env_bool("OPENFANG_TURN_CONTEXT_ROSTER")
        .or_else(|| INSTALLED.get().map(|c| c.roster))
        .unwrap_or(false)
}

/// Max actors listed in the roster line. Installed config > compiled default
/// (5). Clamped to at least 1 so a fat-fingered `0` cannot silently blank the
/// roster while it is enabled.
pub fn roster_limit() -> usize {
    INSTALLED.get().map(|c| c.roster_limit).unwrap_or(5).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INSTALLED is never set in this crate's test binary (only the kernel
    /// installs it), so resolvers fall through to compiled defaults unless an
    /// env override is present. Run env cases in one test to avoid races.
    #[test]
    fn test_resolvers() {
        std::env::remove_var("OPENFANG_TURN_CONTEXT");
        std::env::remove_var("OPENFANG_TURN_CONTEXT_ROSTER");
        assert!(enabled(), "default enabled = true");
        assert!(!roster_enabled(), "default roster = false");
        assert_eq!(roster_limit(), 5);

        std::env::set_var("OPENFANG_TURN_CONTEXT", "off");
        assert!(!enabled());
        std::env::set_var("OPENFANG_TURN_CONTEXT", "1");
        assert!(enabled());
        std::env::remove_var("OPENFANG_TURN_CONTEXT");

        std::env::set_var("OPENFANG_TURN_CONTEXT_ROSTER", "true");
        assert!(roster_enabled());
        std::env::remove_var("OPENFANG_TURN_CONTEXT_ROSTER");
    }
}
