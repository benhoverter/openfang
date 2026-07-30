//! Agent-wake emission limits (ANAI-111): the tunable ceilings that bound
//! `agent_send_async` amplification.
//!
//! Three limits guard three DIFFERENT failure modes; they are deliberately kept
//! together but are not redundant:
//!
//! * **Per-tree budget (`tree_budget_max`, req 10):** a sliding-window cap on
//!   the wakes emitted by ONE wake tree, keyed on the lineage root
//!   ([`WakeLineage::root`](crate::wake::WakeLineage::root)). Cross-hop cycle
//!   (req 4) and depth (req 9) — enforced for real since lineage threading
//!   landed (ANAI-110) — still miss *fan-out amplification*: a tree that never
//!   repeats an agent and never exceeds the depth bound can still emit `F^k`
//!   wakes (each hop wakes `F` distinct peers). Keying a budget on the root
//!   makes the root pay for its whole subtree's emission rate, closing that
//!   hole.
//! * **Aggregate ceiling (`emit_max`):** a process-global sliding-window cap on
//!   wake emissions across ALL trees. N trees each under their own per-tree
//!   budget still sum to `N * tree_budget_max` of fleet-wide load; this coarse
//!   backstop bounds that aggregate. It only ever refuses, never permits —
//!   harmless defense-in-depth that the per-tree budget does not subsume.
//! * **Max in-flight (`max_inflight`):** a concurrency cap (semaphore permits)
//!   on simultaneously-running woken agent loops in the kernel's wake-consumer.
//!   Distinct from the rate limits above: it bounds *concurrent dispatch*, not
//!   *emission rate*.
//! * **Per-caller in-flight cap (`per_caller_max`, ANAI-104):** a concurrency
//!   cap on the woken loops attributable to ONE caller (`created_by`), enforced
//!   at wake-claim. Async removed the synchronous path's accidental
//!   self-throttle (a blocking `agent_send` tied up the caller's own turn);
//!   this restores per-caller backpressure so one buggy/compromised orchestrator
//!   cannot dispatch unbounded concurrent runs. Where `max_inflight` bounds the
//!   FLEET's concurrent dispatch, this bounds any SINGLE caller's slice of it.
//!
//! Both sliding counters share one `window_secs` — they measure the same
//! per-window rate, at different granularities (one tree vs. the whole fleet).
//!
//! ## Tuning (mirrors the `[watchdog]` knob, ANAI-109)
//!
//! Each limit resolves with fixed precedence: env var > installed
//! `[agent_wake]` config > compiled default, then clamped up to a small floor
//! so a fat-fingered `0` cannot deadlock all wakes (or, for `max_inflight`,
//! stall the wake-consumer). `[agent_wake]` config installs once at kernel boot
//! via [`install_limits`] (first-writer-wins `OnceLock`); the env override is
//! read live per call but fixed at process launch. Net effect: edit
//! `config.toml` -> restart daemon -> new limits. No recompile, no rebuild.

/// Default aggregate ceiling: max wake emissions across ALL trees within one
/// [`window_secs`] window before the coarse cross-tree backstop refuses. Tune
/// via `[agent_wake] emit_max` or `OPENFANG_AGENT_WAKE_EMIT_MAX`; read through
/// [`emit_max`].
pub const WAKE_EMIT_MAX: usize = 120;

/// Default per-tree budget: max wakes one wake tree (keyed on its lineage root)
/// may emit within one [`window_secs`] window before the per-tree backstop
/// refuses. A third of [`WAKE_EMIT_MAX`] — greenfield; discover the real number
/// in ops. Tune via `[agent_wake] tree_budget_max` or
/// `OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX`; read through [`tree_budget_max`].
pub const WAKE_TREE_BUDGET_MAX: usize = 40;

/// Default sliding-window width, in seconds, shared by the aggregate ceiling
/// and the per-tree budget. Tune via `[agent_wake] window_secs` or
/// `OPENFANG_AGENT_WAKE_WINDOW_SECS`; read through [`window_secs`].
pub const WAKE_WINDOW_SECS: u64 = 60;

/// Default concurrency cap: max simultaneously in-flight woken agent loops in
/// the kernel's wake-consumer. Each dispatch runs a FULL agent loop, so this
/// bounds concurrency/memory amplification. Tune via `[agent_wake] max_inflight`
/// or `OPENFANG_AGENT_WAKE_MAX_INFLIGHT`; read through [`max_inflight`].
pub const MAX_INFLIGHT_WAKES: usize = 8;

/// Default per-caller in-flight cap (ANAI-104): max simultaneously in-flight
/// woken agent loops attributable to ONE caller (`created_by`), enforced at
/// wake-claim. A single caller therefore claims at most this many concurrent
/// runs even when the fleet-wide [`MAX_INFLIGHT_WAKES`] budget has more slots
/// free — so no one caller monopolizes dispatch (default is half of
/// `MAX_INFLIGHT_WAKES`, leaving room for a second caller). Tune via
/// `[agent_wake] per_caller_max` or `OPENFANG_AGENT_WAKE_PER_CALLER_MAX`; read
/// through [`per_caller_max`].
pub const WAKE_PER_CALLER_MAX: usize = 4;

/// Default stale-claim timeout, in seconds (ANAI-147 defect 2): how long a wake
/// may sit `in_progress` before the kernel's stale-wake reaper concludes its
/// dispatcher is dead and fails it closed, freeing the caller's
/// [`WAKE_PER_CALLER_MAX`] slot.
///
/// The per-caller cap is a *concurrency* limit whose slots are released only by
/// `task_complete`. A dispatcher that dies without completing (process restart,
/// panicked detached task, wedged agent loop) leaks its slot permanently, and
/// `per_caller_max` such leaks wedge that caller's queue forever. The boot
/// reaper handles the restart case; this timeout handles every other one.
///
/// One hour: far above any legitimate woken turn (the `[watchdog]` ceilings bound
/// a real turn to minutes), so a live loop is never reaped out from under itself.
/// Tune via `[agent_wake] stale_wake_secs` or `OPENFANG_AGENT_WAKE_STALE_SECS`;
/// read through [`stale_wake_secs`].
pub const WAKE_STALE_SECS: u64 = 3600;

/// Floor for every resolved limit. `0` for any of these is a footgun — a zero
/// ceiling/budget refuses all wakes, a zero window degenerates eviction, a zero
/// permit count stalls the wake-consumer forever — so a fat-fingered value is
/// clamped up to `1` rather than honored.
const LIMIT_FLOOR: u64 = 1;

/// Operator-configured agent-wake limits, sourced from the `[agent_wake]`
/// config section and installed once at kernel boot via [`install_limits`].
#[derive(Clone, Copy, Debug)]
pub struct AgentWakeLimits {
    /// Aggregate cross-tree emission ceiling per window.
    pub emit_max: usize,
    /// Per-tree (per-lineage-root) emission budget per window.
    pub tree_budget_max: usize,
    /// Sliding-window width in seconds, shared by both counters above.
    pub window_secs: u64,
    /// Max concurrently in-flight woken agent loops.
    pub max_inflight: usize,
    /// Max concurrently in-flight woken agent loops attributable to one caller.
    pub per_caller_max: usize,
    /// Seconds a wake may sit `in_progress` before the stale-claim reaper
    /// fails it closed.
    pub stale_wake_secs: u64,
}

static INSTALLED: std::sync::OnceLock<AgentWakeLimits> = std::sync::OnceLock::new();

/// Install operator-configured agent-wake limits. First writer wins; later
/// calls are ignored (the kernel installs exactly once at boot). Per-call env
/// overrides still take precedence over whatever is installed here.
pub fn install_limits(l: AgentWakeLimits) {
    let _ = INSTALLED.set(l);
}

fn env_u64(var: &str) -> Option<u64> {
    std::env::var(var).ok().and_then(|s| s.trim().parse().ok())
}

fn env_usize(var: &str) -> Option<usize> {
    std::env::var(var).ok().and_then(|s| s.trim().parse().ok())
}

/// Resolved aggregate cross-tree wake-emission ceiling per [`window_secs`].
///
/// Precedence: `OPENFANG_AGENT_WAKE_EMIT_MAX` env var > installed
/// `[agent_wake]` config > compiled default ([`WAKE_EMIT_MAX`]). Clamped up to
/// a floor of `1` so it can never be tuned into "refuse every wake".
pub fn emit_max() -> usize {
    env_usize("OPENFANG_AGENT_WAKE_EMIT_MAX")
        .or_else(|| INSTALLED.get().map(|l| l.emit_max))
        .unwrap_or(WAKE_EMIT_MAX)
        .max(LIMIT_FLOOR as usize)
}

/// Resolved per-tree (per-lineage-root) wake-emission budget per
/// [`window_secs`].
///
/// Precedence: `OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX` env var > installed
/// `[agent_wake]` config > compiled default ([`WAKE_TREE_BUDGET_MAX`]). Clamped
/// up to a floor of `1`.
pub fn tree_budget_max() -> usize {
    env_usize("OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX")
        .or_else(|| INSTALLED.get().map(|l| l.tree_budget_max))
        .unwrap_or(WAKE_TREE_BUDGET_MAX)
        .max(LIMIT_FLOOR as usize)
}

/// Resolved sliding-window width, in seconds, shared by the aggregate ceiling
/// and the per-tree budget.
///
/// Precedence: `OPENFANG_AGENT_WAKE_WINDOW_SECS` env var > installed
/// `[agent_wake]` config > compiled default ([`WAKE_WINDOW_SECS`]). Clamped up
/// to a floor of `1` so eviction never degenerates on a zero window.
pub fn window_secs() -> u64 {
    env_u64("OPENFANG_AGENT_WAKE_WINDOW_SECS")
        .or_else(|| INSTALLED.get().map(|l| l.window_secs))
        .unwrap_or(WAKE_WINDOW_SECS)
        .max(LIMIT_FLOOR)
}

/// Resolved concurrency cap on simultaneously in-flight woken agent loops.
///
/// Precedence: `OPENFANG_AGENT_WAKE_MAX_INFLIGHT` env var > installed
/// `[agent_wake]` config > compiled default ([`MAX_INFLIGHT_WAKES`]). Clamped
/// up to a floor of `1` so the wake-consumer always has at least one permit.
pub fn max_inflight() -> usize {
    env_usize("OPENFANG_AGENT_WAKE_MAX_INFLIGHT")
        .or_else(|| INSTALLED.get().map(|l| l.max_inflight))
        .unwrap_or(MAX_INFLIGHT_WAKES)
        .max(LIMIT_FLOOR as usize)
}

/// Resolved per-caller in-flight cap on woken agent loops (ANAI-104).
///
/// Precedence: `OPENFANG_AGENT_WAKE_PER_CALLER_MAX` env var > installed
/// `[agent_wake]` config > compiled default ([`WAKE_PER_CALLER_MAX`]). Clamped
/// up to a floor of `1` so a caller can always make forward progress (a `0`
/// would refuse every wake a caller enqueues).
pub fn per_caller_max() -> usize {
    env_usize("OPENFANG_AGENT_WAKE_PER_CALLER_MAX")
        .or_else(|| INSTALLED.get().map(|l| l.per_caller_max))
        .unwrap_or(WAKE_PER_CALLER_MAX)
        .max(LIMIT_FLOOR as usize)
}

/// Resolved stale-claim timeout for in-flight wakes, in seconds (ANAI-147).
///
/// Precedence: `OPENFANG_AGENT_WAKE_STALE_SECS` env var > installed
/// `[agent_wake]` config > compiled default ([`WAKE_STALE_SECS`]). Clamped up to
/// a floor of `1` — note that a very low value will reap *live* loops, so this
/// floor is a footgun guard, not a recommendation.
pub fn stale_wake_secs() -> u64 {
    env_u64("OPENFANG_AGENT_WAKE_STALE_SECS")
        .or_else(|| INSTALLED.get().map(|l| l.stale_wake_secs))
        .unwrap_or(WAKE_STALE_SECS)
        .max(LIMIT_FLOOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All resolver env cases in one test to avoid env-var races across threads.
    /// `INSTALLED` is never set in this crate's test binary (only the kernel
    /// installs it), so the config layer falls through to the compiled defaults.
    #[test]
    fn test_limit_resolvers() {
        // emit_max: default, explicit override, floor clamp, unparseable.
        std::env::remove_var("OPENFANG_AGENT_WAKE_EMIT_MAX");
        assert_eq!(emit_max(), 120);
        std::env::set_var("OPENFANG_AGENT_WAKE_EMIT_MAX", "200");
        assert_eq!(emit_max(), 200);
        std::env::set_var("OPENFANG_AGENT_WAKE_EMIT_MAX", "0"); // would-be deadlock
        assert_eq!(emit_max(), 1);
        std::env::set_var("OPENFANG_AGENT_WAKE_EMIT_MAX", "junk");
        assert_eq!(emit_max(), 120);
        std::env::remove_var("OPENFANG_AGENT_WAKE_EMIT_MAX");

        // tree_budget_max: default, override, floor.
        std::env::remove_var("OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX");
        assert_eq!(tree_budget_max(), 40);
        std::env::set_var("OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX", "80");
        assert_eq!(tree_budget_max(), 80);
        std::env::set_var("OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX", "0");
        assert_eq!(tree_budget_max(), 1);
        std::env::remove_var("OPENFANG_AGENT_WAKE_TREE_BUDGET_MAX");

        // window_secs: default, override, floor.
        std::env::remove_var("OPENFANG_AGENT_WAKE_WINDOW_SECS");
        assert_eq!(window_secs(), 60);
        std::env::set_var("OPENFANG_AGENT_WAKE_WINDOW_SECS", "30");
        assert_eq!(window_secs(), 30);
        std::env::set_var("OPENFANG_AGENT_WAKE_WINDOW_SECS", "0");
        assert_eq!(window_secs(), 1);
        std::env::remove_var("OPENFANG_AGENT_WAKE_WINDOW_SECS");

        // max_inflight: default, override, floor.
        std::env::remove_var("OPENFANG_AGENT_WAKE_MAX_INFLIGHT");
        assert_eq!(max_inflight(), 8);
        std::env::set_var("OPENFANG_AGENT_WAKE_MAX_INFLIGHT", "16");
        assert_eq!(max_inflight(), 16);
        std::env::set_var("OPENFANG_AGENT_WAKE_MAX_INFLIGHT", "0"); // would-be stall
        assert_eq!(max_inflight(), 1);
        std::env::remove_var("OPENFANG_AGENT_WAKE_MAX_INFLIGHT");

        // per_caller_max: default, override, floor.
        std::env::remove_var("OPENFANG_AGENT_WAKE_PER_CALLER_MAX");
        assert_eq!(per_caller_max(), 4);
        std::env::set_var("OPENFANG_AGENT_WAKE_PER_CALLER_MAX", "12");
        assert_eq!(per_caller_max(), 12);
        std::env::set_var("OPENFANG_AGENT_WAKE_PER_CALLER_MAX", "0"); // would-be refuse-all
        assert_eq!(per_caller_max(), 1);
        std::env::remove_var("OPENFANG_AGENT_WAKE_PER_CALLER_MAX");
        assert_eq!(WAKE_PER_CALLER_MAX, 4);

        // stale_wake_secs: default, override, floor.
        std::env::remove_var("OPENFANG_AGENT_WAKE_STALE_SECS");
        assert_eq!(stale_wake_secs(), 3600);
        std::env::set_var("OPENFANG_AGENT_WAKE_STALE_SECS", "900");
        assert_eq!(stale_wake_secs(), 900);
        std::env::set_var("OPENFANG_AGENT_WAKE_STALE_SECS", "0");
        assert_eq!(stale_wake_secs(), 1);
        std::env::remove_var("OPENFANG_AGENT_WAKE_STALE_SECS");
    }

    #[test]
    fn test_compiled_defaults() {
        assert_eq!(WAKE_EMIT_MAX, 120);
        assert_eq!(WAKE_TREE_BUDGET_MAX, 40);
        assert_eq!(WAKE_WINDOW_SECS, 60);
        assert_eq!(MAX_INFLIGHT_WAKES, 8);
        assert_eq!(WAKE_PER_CALLER_MAX, 4);
        assert_eq!(WAKE_STALE_SECS, 3600);
    }
}
