//! ANAI-225: per-purpose state for daemon-owned model calls.
//!
//! One driver cache and one circuit breaker **per [`BackgroundPurpose`]**. The
//! invocation itself lives on `OpenFangKernel::background_complete`; this
//! module holds only the state it keys into, so the state can be reasoned
//! about (and tested) without a kernel.
//!
//! # Accounting is the caller's, on purpose
//!
//! [`BackgroundLlmState`] exposes `note_success` / `note_failure` rather than
//! incrementing the breaker itself. That is not laziness: the gatekeeper counts
//! an *unparseable* answer as a failure, and only a successfully parsed verdict
//! clears its breaker — facts the transport layer cannot see. Whoever knows
//! whether the answer was usable is the one who records it.

use openfang_runtime::background_llm::BackgroundPurpose;
use openfang_runtime::llm_driver::LlmDriver;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use crate::kernel::OpenFangKernel;
use openfang_runtime::background_llm::{
    BackgroundFailure, BackgroundLlmOutcome, BackgroundLlmRequest,
};
use openfang_runtime::drivers;
use openfang_runtime::llm_driver::{CompletionRequest, DriverConfig};
use tracing::warn;

/// One purpose's slice of state.
#[derive(Default)]
struct Slot {
    /// Lazily built on first use. `Some(None)` means construction failed.
    ///
    /// NOTE (carried forward from ANAI-154, not introduced here): a `OnceLock`
    /// caches init *failure* permanently — a provider blip on the very first
    /// call disables this purpose for the life of the process. Preserved
    /// verbatim so this refactor changes no behaviour; fixing it is its own
    /// ticket and its own telemetry read.
    driver: OnceLock<Option<Arc<dyn LlmDriver>>>,
    /// Consecutive recorded failures.
    failures: AtomicU32,
}

/// Driver caches and circuit breakers for every [`BackgroundPurpose`].
pub struct BackgroundLlmState {
    slots: [Slot; BackgroundPurpose::COUNT],
}

impl Default for BackgroundLlmState {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundLlmState {
    pub fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| Slot::default()),
        }
    }

    fn slot(&self, purpose: BackgroundPurpose) -> &Slot {
        &self.slots[purpose.slot()]
    }

    /// The driver cell for `purpose`, for the kernel to `get_or_init`.
    pub(crate) fn driver_cell(
        &self,
        purpose: BackgroundPurpose,
    ) -> &OnceLock<Option<Arc<dyn LlmDriver>>> {
        &self.slot(purpose).driver
    }

    /// Consecutive failures recorded for `purpose`.
    pub fn failures(&self, purpose: BackgroundPurpose) -> u32 {
        self.slot(purpose).failures.load(Ordering::Relaxed)
    }

    /// Is `purpose`'s breaker open at `threshold`?
    pub fn circuit_open(&self, purpose: BackgroundPurpose, threshold: u32) -> bool {
        self.failures(purpose) >= threshold
    }

    /// Record a usable answer: clears the breaker for `purpose`.
    pub fn note_success(&self, purpose: BackgroundPurpose) {
        self.slot(purpose).failures.store(0, Ordering::Relaxed);
    }

    /// Record a failure. Returns the new consecutive-failure count so the
    /// caller can log exactly once, on the tick that trips the breaker.
    pub fn note_failure(&self, purpose: BackgroundPurpose) -> u32 {
        self.slot(purpose).failures.fetch_add(1, Ordering::Relaxed) + 1
    }
}

impl OpenFangKernel {
    /// ANAI-225: one daemon-owned model call, with no agent turn in flight.
    ///
    /// Lifted verbatim from `gatekeeper_review`'s private path (ANAI-154), whose
    /// properties are the reason this exists as a primitive rather than as a
    /// pattern people re-type:
    ///
    /// - **No agenthood.** No session, no history, no tools, no re-entrancy.
    ///   An agent loop would hand the callee context it must not have.
    /// - **No caller identity.** `caller_agent_id: None`, `allowed_tools: None`.
    ///   The call is the daemon's, not any agent's, and must not inherit an
    ///   agent's identity or tool surface. This is a capability boundary, not
    ///   bookkeeping: a background task that inherits an agent's tools is a
    ///   background task with shell access.
    /// - **Pinned model.** From the purpose's own config block, never the
    ///   calling agent's model.
    /// - **Hard timeout and a per-purpose breaker**, checked before the call so
    ///   a wedged purpose stops paying latency for an answer it won't use.
    ///
    /// Returns a typed report. It records **no** success or failure itself —
    /// see this module's header; the caller knows whether the answer was
    /// usable and calls `note_success` / `note_failure` accordingly.
    pub async fn background_complete(&self, req: &BackgroundLlmRequest) -> BackgroundLlmOutcome {
        let purpose = req.purpose;

        if self
            .background_llm
            .circuit_open(purpose, req.failure_threshold)
        {
            return BackgroundLlmOutcome::Failed(BackgroundFailure::CircuitOpen);
        }

        let driver = self.background_llm.driver_cell(purpose).get_or_init(|| {
            let provider = if req.provider.is_empty() {
                self.config.default_model.provider.clone()
            } else {
                req.provider.clone()
            };
            let env_var = self.config.resolve_api_key_env(&provider);
            let driver_config = DriverConfig {
                provider: provider.clone(),
                api_key: self.resolve_credential(&env_var),
                base_url: self.lookup_provider_url(&provider),
                skip_permissions: true,
                subprocess_timeout_secs: None,
            };
            match drivers::create_driver(&driver_config, self.token_issuer()) {
                Ok(d) => Some(d),
                Err(e) => {
                    warn!(
                        target: "openfang::background_llm",
                        purpose = %purpose,
                        provider = %provider,
                        error = %e,
                        "Background LLM driver init failed — every call for this purpose will fail"
                    );
                    None
                }
            }
        });
        let Some(driver) = driver.as_ref() else {
            return BackgroundLlmOutcome::Failed(BackgroundFailure::ProviderError);
        };

        let request = CompletionRequest {
            model: req.model.clone(),
            messages: vec![openfang_types::message::Message {
                role: openfang_types::message::Role::User,
                content: openfang_types::message::MessageContent::Blocks(vec![
                    openfang_types::message::ContentBlock::Text {
                        text: req.user.clone(),
                        provider_metadata: None,
                    },
                ]),
                ..Default::default()
            }],
            tools: vec![],
            max_tokens: req.max_tokens,
            temperature: 0.0,
            system: req.system.clone(),
            thinking: None,
            // No caller attribution: this call is the daemon's, and must not
            // inherit any agent's identity or tools.
            caller_agent_id: None,
            allowed_tools: None,
        };

        let call = tokio::time::timeout(
            std::time::Duration::from_secs(req.timeout_secs),
            driver.complete(request),
        )
        .await;

        match call {
            Ok(Ok(response)) => BackgroundLlmOutcome::Answered(response.text()),
            Ok(Err(e)) => {
                warn!(
                    target: "openfang::background_llm",
                    purpose = %purpose,
                    error = %e,
                    "Background LLM call failed"
                );
                BackgroundLlmOutcome::Failed(BackgroundFailure::ProviderError)
            }
            Err(_elapsed) => {
                warn!(
                    target: "openfang::background_llm",
                    purpose = %purpose,
                    timeout_secs = req.timeout_secs,
                    "Background LLM call timed out"
                );
                BackgroundLlmOutcome::Failed(BackgroundFailure::TimedOut)
            }
        }
    }

    /// Has `purpose`'s driver already been built (successfully or not)?
    ///
    /// Exists so a caller can emit its own one-shot "subsystem is live" line on
    /// the call that actually constructs the driver — the gatekeeper's
    /// `Approval gatekeeper enabled …` line is an operator-facing contract and
    /// is not the invoker's to rename.
    pub(crate) fn background_driver_built(&self, purpose: BackgroundPurpose) -> bool {
        self.background_llm.driver_cell(purpose).get().is_some()
    }

    /// Was `purpose`'s driver built *successfully*?
    pub(crate) fn background_driver_ready(&self, purpose: BackgroundPurpose) -> bool {
        matches!(
            self.background_llm.driver_cell(purpose).get(),
            Some(Some(_))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of per-purpose slots: a wedged judge must not disable
    /// episode summarisation.
    #[test]
    fn breakers_do_not_bleed_across_purposes() {
        let state = BackgroundLlmState::new();
        for _ in 0..3 {
            state.note_failure(BackgroundPurpose::Gatekeeper);
        }
        assert!(state.circuit_open(BackgroundPurpose::Gatekeeper, 3));
        assert_eq!(state.failures(BackgroundPurpose::Consolidation), 0);
        assert!(!state.circuit_open(BackgroundPurpose::Consolidation, 3));
    }

    /// Only a recorded success clears the count — matching the gatekeeper's
    /// pre-refactor semantics, where nothing but a parsed verdict resets it.
    #[test]
    fn success_clears_and_failures_accumulate() {
        let state = BackgroundLlmState::new();
        assert_eq!(state.note_failure(BackgroundPurpose::Gatekeeper), 1);
        assert_eq!(state.note_failure(BackgroundPurpose::Gatekeeper), 2);
        state.note_success(BackgroundPurpose::Gatekeeper);
        assert_eq!(state.failures(BackgroundPurpose::Gatekeeper), 0);
        assert_eq!(state.note_failure(BackgroundPurpose::Gatekeeper), 1);
    }
}
