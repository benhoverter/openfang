//! ANAI-225: types for a **daemon-owned, single-shot model call** made with no
//! agent turn in flight.
//!
//! The shape is the gatekeeper judge's (ANAI-154), generalised: one
//! [`crate::llm_driver::LlmDriver::complete`] against a pinned model, no
//! session, no tools, no caller attribution, a hard timeout and a circuit
//! breaker. The kernel owns the invocation; this module owns the vocabulary so
//! callers outside the kernel crate can name a request and read an outcome
//! without depending on the kernel.
//!
//! # Why the caller decides what a non-answer means
//!
//! The gatekeeper fails **closed**: no answer means escalate to a human, and
//! [`openfang_types::gatekeeper::GateReview::failed`] enforces that by
//! construction. Episode consolidation (ANAI-220) needs the exact opposite —
//! no answer means leave `summary` null, move on, never retry — because a
//! close that waits on a provider is a close that a provider outage can lose.
//!
//! Both are correct for their purpose, so this primitive refuses to pick.
//! [`BackgroundLlmOutcome`] is a *typed report*, never a decision: it says what
//! happened, and the call site maps that to policy. Baking either policy in
//! here would make one of the two callers wrong.
//!
//! Nothing in this module performs I/O.

use std::fmt;

/// What a background call is *for*. One variant per daemon-side consumer.
///
/// Purpose is not decoration: it keys the driver cache and the circuit breaker,
/// so a wedged judge cannot disable episode summarisation and vice versa. Two
/// unrelated subsystems sharing one breaker is how a single provider blip takes
/// out a feature nobody was touching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackgroundPurpose {
    /// ANAI-154: the approval gatekeeper's judge.
    Gatekeeper,
    /// ANAI-220: episode close → summary. Not yet wired; the slot exists so the
    /// breaker and driver cache are per-purpose from the first commit rather
    /// than retrofitted once a second caller appears.
    Consolidation,
}

impl BackgroundPurpose {
    /// Every purpose, in slot order. `COUNT` and `slot()` derive from this, so
    /// adding a variant is a one-line change plus the `slot()` arm the
    /// exhaustive `match` will demand.
    pub const ALL: [BackgroundPurpose; 2] = [Self::Gatekeeper, Self::Consolidation];

    /// Number of distinct purposes — the width of the kernel's slot array.
    pub const COUNT: usize = Self::ALL.len();

    /// Stable index into the kernel's per-purpose state array.
    pub const fn slot(self) -> usize {
        match self {
            Self::Gatekeeper => 0,
            Self::Consolidation => 1,
        }
    }

    /// Stable log/audit token. Do not change these casually — they show up in
    /// operator greps.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gatekeeper => "gatekeeper",
            Self::Consolidation => "consolidation",
        }
    }
}

impl fmt::Display for BackgroundPurpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One daemon-side model call.
///
/// Every knob is explicit rather than read from config inside the invoker:
/// each purpose owns its own config block (`[gatekeeper]`, and ANAI-226's
/// `[memory.consolidation]`), and the invoker must not have to know which.
#[derive(Debug, Clone)]
pub struct BackgroundLlmRequest {
    /// Keys the driver cache and the circuit breaker.
    pub purpose: BackgroundPurpose,
    /// Provider id. Empty means "fall back to `default_model.provider`".
    pub provider: String,
    /// Pinned model id. Deliberately not the calling agent's model.
    pub model: String,
    /// System prompt, if the purpose has one.
    pub system: Option<String>,
    /// User-role prompt body.
    pub user: String,
    /// Output ceiling. Small ceilings are themselves a defence: there is no
    /// room for the model to be talked into an essay.
    pub max_tokens: u32,
    /// Hard wall on the call. Exceeding it yields [`BackgroundFailure::TimedOut`].
    pub timeout_secs: u64,
    /// Consecutive recorded failures that open this purpose's breaker.
    pub failure_threshold: u32,
}

/// Why a background call produced no text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundFailure {
    /// This purpose's breaker is open; no call was attempted.
    CircuitOpen,
    /// Driver construction failed, or the driver returned an error.
    ProviderError,
    /// The call exceeded `timeout_secs`.
    TimedOut,
}

impl BackgroundFailure {
    /// Stable log/audit token.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CircuitOpen => "circuit_open",
            Self::ProviderError => "provider_error",
            Self::TimedOut => "timed_out",
        }
    }
}

impl fmt::Display for BackgroundFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The result of one background call: a report, never a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundLlmOutcome {
    /// The model answered. The text is raw and **unvalidated** — parsing it,
    /// and deciding what an unparseable answer costs, belongs to the caller.
    Answered(String),
    /// No answer. The caller decides what that means for its own invariants.
    Failed(BackgroundFailure),
}

impl BackgroundLlmOutcome {
    /// The answer text, if there was one.
    pub fn answered(&self) -> Option<&str> {
        match self {
            Self::Answered(text) => Some(text.as_str()),
            Self::Failed(_) => None,
        }
    }

    /// The failure, if there was one.
    pub const fn failure(&self) -> Option<BackgroundFailure> {
        match self {
            Self::Answered(_) => None,
            Self::Failed(f) => Some(*f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot array is indexed by `slot()`; a duplicate or out-of-range index
    /// would silently make two purposes share one breaker, which is the exact
    /// coupling this type exists to prevent.
    #[test]
    fn slots_are_unique_and_in_range() {
        let mut seen = vec![false; BackgroundPurpose::COUNT];
        for p in BackgroundPurpose::ALL {
            let i = p.slot();
            assert!(i < BackgroundPurpose::COUNT, "{p} slot {i} out of range");
            assert!(!seen[i], "{p} shares slot {i} with another purpose");
            seen[i] = true;
        }
        assert!(seen.into_iter().all(|s| s), "a slot has no purpose");
    }

    #[test]
    fn outcome_accessors_are_exclusive() {
        let ok = BackgroundLlmOutcome::Answered("SUPPRESS".into());
        assert_eq!(ok.answered(), Some("SUPPRESS"));
        assert_eq!(ok.failure(), None);

        let bad = BackgroundLlmOutcome::Failed(BackgroundFailure::TimedOut);
        assert_eq!(bad.answered(), None);
        assert_eq!(bad.failure(), Some(BackgroundFailure::TimedOut));
        assert_eq!(bad.failure().unwrap().as_str(), "timed_out");
    }
}
