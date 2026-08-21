//! Async-reply deadline limits (ANAI-201): the tunable bounds on the
//! sender-supplied `timeout` that governs when the kernel pays an outstanding
//! reply debt.
//!
//! ## Why a sender-supplied deadline at all
//!
//! ANAI-197/199/198/200 make the reply-right a *debt the kernel pays* on every
//! path where kernel code still runs at turn end. That yields an **eventual**
//! reply. It does not yield a **bounded** one: a wedged subprocess or a hung
//! model call runs no turn-end code, so the debt is never discharged and the
//! initiator waits forever — the original symptom, relocated.
//!
//! An orchestrator needs the stronger contract: "this should take at most ten
//! minutes, and I want to be *certain* something comes back by then." That is
//! what `timeout` buys. The deadline is enforced by aborting the callee's turn
//! and minting a [`ReplyKind::Timeout`](crate::wake::ReplyKind::Timeout) reply.
//!
//! ## Why the bounds are configurable rather than compiled
//!
//! We have no empirical distribution of woken-turn durations yet, so any
//! compiled default is a guess. These knobs exist so the numbers can be tuned
//! from `config.toml` once the correlation rows have accumulated a real
//! distribution (each row records both the *requested* and the *clamped*
//! timeout, precisely so that distribution is free to collect).
//!
//! ## Why clamping happens at SEND, not at sweep
//!
//! [`clamp_timeout_secs`] is called once, in `agent_send_async`, and the
//! clamped value is stamped into the durable [`WakeEnvelope`]. The row is then
//! the single source of truth for that correlation's deadline. Re-deriving the
//! clamp at enforcement time would let a `config.toml` edit silently move the
//! deadline of an in-flight send — an orchestrator's contract would change
//! under it, mid-flight, with no signal.
//!
//! ## Tuning (mirrors the `[agent_wake]` knob, ANAI-111)
//!
//! Each limit resolves with fixed precedence: env var > installed
//! `[async_reply]` config > compiled default. `[async_reply]` installs once at
//! kernel boot via [`install_limits`] (first-writer-wins `OnceLock`); the env
//! override is read live per call but fixed at process launch.

/// Default deadline, in seconds, applied when the sender omits `timeout_secs`.
///
/// 15 minutes: comfortably above a long research/refactor turn, comfortably
/// below "an orchestrator has silently stalled". A guess, and known to be one —
/// see the module note on why these are config knobs.
///
/// Tune via `[async_reply] default_timeout_secs` or
/// `OPENFANG_ASYNC_REPLY_DEFAULT_TIMEOUT_SECS`; read through
/// [`default_timeout_secs`].
pub const DEFAULT_TIMEOUT_SECS: u64 = 900;

/// Floor for a sender-supplied deadline, in seconds.
///
/// An optimistic caller with a 5-second estimate would otherwise shred every
/// legitimate turn it dispatches, and the resulting fleet behaviour reads as
/// "agents are flaky" rather than "someone set a bad number". 60s is the
/// shortest deadline under which a real model turn has any chance to finish.
///
/// Tune via `[async_reply] min_timeout_secs` or
/// `OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS`; read through [`min_timeout_secs`].
pub const MIN_TIMEOUT_SECS: u64 = 60;

/// Ceiling for a sender-supplied deadline, in seconds.
///
/// Bounds how long one correlation may occupy a per-caller in-flight slot, and
/// caps the blast radius of a caller that passes an absurd value to opt out of
/// the guarantee entirely. 60 minutes matches the stale-wake reaper's default
/// cutoff ([`WAKE_STALE_SECS`](crate::agent_wake::WAKE_STALE_SECS)), so the two
/// backstops agree on the longest tolerable in-flight lifetime.
///
/// Tune via `[async_reply] max_timeout_secs` or
/// `OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS`; read through [`max_timeout_secs`].
pub const MAX_TIMEOUT_SECS: u64 = 3600;

/// Absolute floor applied to every resolved knob. A `0` deadline would abort
/// every turn before it began (or, for the ceiling, refuse every deadline), so
/// a fat-fingered value is clamped up rather than honored.
const LIMIT_FLOOR: u64 = 1;

/// Operator-configured async-reply deadline limits, sourced from the
/// `[async_reply]` config section and installed once at kernel boot via
/// [`install_limits`].
#[derive(Clone, Copy, Debug)]
pub struct AsyncReplyLimits {
    /// Deadline applied when the sender omits `timeout_secs`.
    pub default_timeout_secs: u64,
    /// Lower bound on a sender-supplied deadline.
    pub min_timeout_secs: u64,
    /// Upper bound on a sender-supplied deadline.
    pub max_timeout_secs: u64,
}

static INSTALLED: std::sync::OnceLock<AsyncReplyLimits> = std::sync::OnceLock::new();

/// Install operator-configured async-reply limits. First writer wins; later
/// calls are ignored (the kernel installs exactly once at boot). Per-call env
/// overrides still take precedence over whatever is installed here.
pub fn install_limits(l: AsyncReplyLimits) {
    let _ = INSTALLED.set(l);
}

fn env_u64(var: &str) -> Option<u64> {
    std::env::var(var).ok().and_then(|s| s.trim().parse().ok())
}

/// Resolved default deadline, in seconds, for a send that omits `timeout_secs`.
///
/// Precedence: `OPENFANG_ASYNC_REPLY_DEFAULT_TIMEOUT_SECS` env var > installed
/// `[async_reply]` config > compiled default ([`DEFAULT_TIMEOUT_SECS`]).
pub fn default_timeout_secs() -> u64 {
    env_u64("OPENFANG_ASYNC_REPLY_DEFAULT_TIMEOUT_SECS")
        .or_else(|| INSTALLED.get().map(|l| l.default_timeout_secs))
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
        .max(LIMIT_FLOOR)
}

/// Resolved lower bound on a sender-supplied deadline, in seconds.
///
/// Precedence: `OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS` env var > installed
/// `[async_reply]` config > compiled default ([`MIN_TIMEOUT_SECS`]).
pub fn min_timeout_secs() -> u64 {
    env_u64("OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS")
        .or_else(|| INSTALLED.get().map(|l| l.min_timeout_secs))
        .unwrap_or(MIN_TIMEOUT_SECS)
        .max(LIMIT_FLOOR)
}

/// Resolved upper bound on a sender-supplied deadline, in seconds.
///
/// Precedence: `OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS` env var > installed
/// `[async_reply]` config > compiled default ([`MAX_TIMEOUT_SECS`]).
pub fn max_timeout_secs() -> u64 {
    env_u64("OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS")
        .or_else(|| INSTALLED.get().map(|l| l.max_timeout_secs))
        .unwrap_or(MAX_TIMEOUT_SECS)
        .max(LIMIT_FLOOR)
}

/// Resolve a sender's requested deadline into the value that will actually be
/// enforced, and stamped into the wake envelope.
///
/// * `None` (sender omitted `timeout`) resolves to [`default_timeout_secs`].
/// * `Some(n)` is clamped into `[min_timeout_secs, max_timeout_secs]`.
///
/// A misordered configuration (`min > max`) resolves to the ceiling rather than
/// panicking on an inverted `clamp` range: the operator gets a usable deadline
/// and the misconfiguration is visible in the returned value, not in a crash.
///
/// The default is deliberately NOT clamped into the min/max band — an operator
/// who sets `default_timeout_secs` outside their own band has expressed an
/// explicit intent for the omitted case, and silently rewriting it would hide
/// the misconfiguration.
pub fn clamp_timeout_secs(requested: Option<u64>) -> u64 {
    let Some(n) = requested else {
        return default_timeout_secs();
    };
    let min = min_timeout_secs();
    let max = max_timeout_secs();
    if min > max {
        return max;
    }
    n.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All resolver + clamp cases in ONE test: env vars are process-global, so
    /// splitting these across `#[test]` fns races under the default multi-thread
    /// harness. `INSTALLED` is never set in this crate's test binary (only the
    /// kernel installs it), so the config layer falls through to the defaults.
    #[test]
    fn test_timeout_resolvers_and_clamp() {
        for v in [
            "OPENFANG_ASYNC_REPLY_DEFAULT_TIMEOUT_SECS",
            "OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS",
            "OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS",
        ] {
            std::env::remove_var(v);
        }

        // Compiled defaults.
        assert_eq!(default_timeout_secs(), 900);
        assert_eq!(min_timeout_secs(), 60);
        assert_eq!(max_timeout_secs(), 3600);

        // Omitted timeout takes the default, NOT the floor.
        assert_eq!(clamp_timeout_secs(None), 900);
        // In-band request is honored verbatim — the common case must not move.
        assert_eq!(clamp_timeout_secs(Some(600)), 600);
        // Below floor clamps up: the optimistic-orchestrator footgun.
        assert_eq!(clamp_timeout_secs(Some(5)), 60);
        // Above ceiling clamps down: no opting out of the guarantee.
        assert_eq!(clamp_timeout_secs(Some(86_400)), 3600);
        // Exact bounds are inclusive.
        assert_eq!(clamp_timeout_secs(Some(60)), 60);
        assert_eq!(clamp_timeout_secs(Some(3600)), 3600);

        // Env override wins over the compiled default.
        std::env::set_var("OPENFANG_ASYNC_REPLY_DEFAULT_TIMEOUT_SECS", "300");
        std::env::set_var("OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS", "30");
        std::env::set_var("OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS", "120");
        assert_eq!(clamp_timeout_secs(None), 300);
        assert_eq!(clamp_timeout_secs(Some(10)), 30);
        assert_eq!(clamp_timeout_secs(Some(999)), 120);

        // Zero is a footgun, not an instruction: floored, never honored.
        std::env::set_var("OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS", "0");
        assert_eq!(min_timeout_secs(), 1);
        std::env::set_var("OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS", "0");
        assert_eq!(max_timeout_secs(), 1);

        // Misordered band (min > max) resolves to the ceiling, never panics —
        // `u64::clamp` would panic on an inverted range.
        std::env::set_var("OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS", "600");
        std::env::set_var("OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS", "60");
        assert_eq!(clamp_timeout_secs(Some(300)), 60);

        // Unparseable falls through to the compiled default.
        std::env::set_var("OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS", "junk");
        assert_eq!(min_timeout_secs(), 60);

        for v in [
            "OPENFANG_ASYNC_REPLY_DEFAULT_TIMEOUT_SECS",
            "OPENFANG_ASYNC_REPLY_MIN_TIMEOUT_SECS",
            "OPENFANG_ASYNC_REPLY_MAX_TIMEOUT_SECS",
        ] {
            std::env::remove_var(v);
        }
    }
}
