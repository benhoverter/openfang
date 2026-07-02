//! Episodic capture policy: decides whether an agent turn is persisted to
//! episodic memory or dropped.
//!
//! This is the memory subsystem's **write-side gate**. It consumes the typed
//! turn provenance ([`TurnTrigger`]) and the side-effect summary
//! ([`TurnEffects`]) produced upstream in the agent loop and renders a single
//! verdict ([`CaptureDecision`]).
//!
//! The policy lives here, in `openfang-memory`, because "what gets persisted to
//! memory" is a memory-subsystem concern (ANAI-80). Its inputs deliberately stay
//! in `openfang-types`: [`TurnTrigger`] is threaded through the kernel send
//! funnel and [`TurnEffects`]/`READ_ONLY_TOOLS` are shared with non-memory
//! consumers (e.g. `AgentMode::filter_tools`), so dragging them in here would
//! invert the dependency direction.

use openfang_types::turn::{TurnEffects, TurnTrigger};

/// Verdict of the capture predicate ([`should_capture_turn`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureDecision {
    /// Write the episodic row as normal.
    Keep,
    /// Skip the episodic write: a heartbeat turn that did nothing durable.
    Drop,
}

impl CaptureDecision {
    /// True for [`CaptureDecision::Drop`].
    pub fn is_drop(&self) -> bool {
        matches!(self, CaptureDecision::Drop)
    }
}

/// The conservative tick/heartbeat drop predicate (ANAI-76).
///
/// ```text
/// drop  <=>  (trigger == Heartbeat)  &&  (turn is inert)  &&  (observer is live)
/// ```
///
/// Pure, total, panic-free, allocation-free. Only [`TurnTrigger::Heartbeat`] is
/// eligible to drop -- `Cron`/`Proactive` carry scheduled/triggered intent and
/// are always kept, as is every human (`User`) or peer-agent (`AgentCall`) turn.
/// Inertness is judged purely by side-effect provenance ([`TurnEffects`]), never
/// by reply content. Biases hard toward keep: anything not provably a heartbeat
/// with zero durable side-effects is captured.
///
/// ANAI-77 (c) safety valve -- `observer_live`: the inert verdict is only
/// trustworthy when the driver actually feeds executed tools into the agent
/// loop (so [`TurnEffects::observe_tool`] fires). Subprocess CLI drivers
/// (`claude-code`, `qwen-code`) run tools inside their own subprocess and
/// return only final text, so the loop sees `tool_calls: []` and `is_inert()`
/// is *vacuously* true -- it cannot distinguish "did nothing" from "did work we
/// could not see". Gating the drop on a live observer makes the enforce flip
/// safe by construction: an observer-blind driver's inert heartbeats are always
/// kept, never dropped. See [`observer_is_live`].
pub fn should_capture_turn(
    trigger: TurnTrigger,
    effects: &TurnEffects,
    observer_live: bool,
) -> CaptureDecision {
    if trigger == TurnTrigger::Heartbeat && effects.is_inert() && observer_live {
        CaptureDecision::Drop
    } else {
        CaptureDecision::Keep
    }
}

/// Providers whose driver surfaces executed tool calls back into the OpenFang
/// agent loop, so [`TurnEffects::observe_tool`] actually runs and the inert
/// verdict (and the ANAI-77 `side_effecting_tools` / `tools_observed` counts)
/// can be trusted.
///
/// This is an **allowlist**, and the bias is deliberate: a provider is treated
/// as observer-live ONLY if it appears here. Anything not listed -- most
/// importantly the subprocess CLI drivers `claude-code` and `qwen-code`, which
/// hardcode `tool_calls: Vec::new()` because tools execute inside the
/// subprocess and only final text round-trips back -- is treated as
/// observer-blind and is therefore never eligible for an inert-heartbeat drop.
/// Unknown provider => keep, matching the drop predicate's conservative bias.
/// (Note: `codex`, `github-copilot`/`copilot`, and `kimi_coding` reuse the
/// HTTP OpenAI/Anthropic drivers, which DO parse tool_use into `tool_calls`,
/// so they are observer-live and listed here.)
///
/// INTERIM: the durable form of this signal is a per-driver capability on the
/// driver trait, landing with the CC tool-execution -> `TurnEffects` work. Until
/// that exists, the provider name is the only discriminator the capture policy
/// can key off without reaching into the driver layer. To make a native
/// provider's inert heartbeats droppable, add it here.
const OBSERVER_LIVE_PROVIDERS: &[&str] = &[
    "anthropic",
    "openai",
    "openrouter",
    "gemini",
    "google",
    "vertex",
    "vertex-ai",
    "google-vertex",
    "azure",
    "azure-openai",
    "bedrock",
    "codex",
    "openai-codex",
    "github-copilot",
    "copilot",
    "kimi_coding",
    "groq",
    "mistral",
    "deepseek",
    "xai",
    "grok",
    "together",
    "fireworks",
    "perplexity",
    "cohere",
    "nvidia",
    "chutes",
    "venice",
    "deepinfra",
    "ollama",
    "lmstudio",
    "vllm",
    "lemonade",
    "minimax",
    "zhipu",
    "zai",
    "qwen",
    "moonshot",
];

/// True when `provider`'s driver feeds executed tool calls back into the agent
/// loop, so the ANAI-77 side-effect observer is trustworthy for its turns.
///
/// Case-insensitive. Conservative by construction: an unrecognized provider
/// returns `false` (observer-blind => its inert heartbeats are hard-kept). The
/// subprocess CLI drivers (`claude-code`, `qwen-code`) are deliberately absent
/// from [`OBSERVER_LIVE_PROVIDERS`] and therefore return `false`.
pub fn observer_is_live(provider: &str) -> bool {
    let p = provider.trim().to_ascii_lowercase();
    OBSERVER_LIVE_PROVIDERS.contains(&p.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_only_inert_heartbeats() {
        let inert = TurnEffects::new();
        let mut active = TurnEffects::new();
        active.observe_tool("agent_send");

        // The one droppable case: heartbeat with no durable side-effect, on a
        // driver whose observer is live.
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &inert, true),
            CaptureDecision::Drop
        );
        // Heartbeat that did real work is kept.
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &active, true),
            CaptureDecision::Keep
        );
        // Every non-heartbeat trigger is kept even when inert.
        for trig in [
            TurnTrigger::User,
            TurnTrigger::AgentCall,
            TurnTrigger::Cron,
            TurnTrigger::Proactive,
        ] {
            assert_eq!(
                should_capture_turn(trig, &inert, true),
                CaptureDecision::Keep,
                "{trig} must never be dropped"
            );
        }
    }

    #[test]
    fn observer_blind_hard_keeps_inert_heartbeats() {
        // ANAI-77 (c): with a blind observer, an inert heartbeat is NOT
        // droppable -- inertness cannot be trusted, so the row is kept. This is
        // the safety property the enforce flip relies on.
        let inert = TurnEffects::new();
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &inert, false),
            CaptureDecision::Keep,
            "observer-blind inert heartbeat must be kept, never dropped"
        );
        // Same driver, an active heartbeat is kept too (unchanged).
        let mut active = TurnEffects::new();
        active.observe_tool("agent_send");
        assert_eq!(
            should_capture_turn(TurnTrigger::Heartbeat, &active, false),
            CaptureDecision::Keep
        );
    }

    #[test]
    fn observer_liveness_by_provider() {
        // Native / HTTP drivers surface tool_calls into the loop -> live.
        for p in [
            "anthropic",
            "openai",
            "openrouter",
            "codex",
            "github-copilot",
            "copilot",
            "kimi_coding",
            "gemini",
            "bedrock",
        ] {
            assert!(observer_is_live(p), "{p} should be observer-live");
        }
        // Subprocess CLI drivers execute tools out of view -> blind.
        for p in ["claude-code", "qwen-code"] {
            assert!(!observer_is_live(p), "{p} must be observer-blind");
        }
        // Unknown provider is conservatively blind (keep, never drop).
        assert!(!observer_is_live("some-future-provider"));
        // Classification is case-insensitive and trims.
        assert!(observer_is_live("  Anthropic  "));
        assert!(!observer_is_live("Claude-Code"));
    }

    #[test]
    fn is_drop_only_for_drop() {
        assert!(CaptureDecision::Drop.is_drop());
        assert!(!CaptureDecision::Keep.is_drop());
    }
}
