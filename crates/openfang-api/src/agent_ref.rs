//! Agent handle resolution — accept either a UUID or a registry name.
//!
//! Historically every route that took an agent in its path did this:
//!
//! ```ignore
//! let agent_id: AgentId = match id.parse() {
//!     Ok(id) => id,
//!     Err(_) => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Invalid agent ID"}))),
//! };
//! ```
//!
//! Two problems. Names were rejected outright even though the CLI's own help
//! text advertises "Agent name or ID", and *every* failure — malformed handle,
//! unknown agent — collapsed into one 400 with one opaque string.
//!
//! This module is the single resolution point. Order is UUID-first, then the
//! registry name index: an agent named like a UUID can never shadow a real ID.
//!
//! Error split (ANAI-180's transparency requirement, landing early here):
//!   * malformed handle — empty, oversized, or holding characters no agent
//!     name may hold → **400**
//!   * well-formed handle, no such agent → **404**
//!
//! The 404 is a deliberate, narrow contract change: callers that used to see
//! 400 for an unknown-but-well-formed UUID now see 404. Accepting names is
//! strictly additive; this half is not. See ANAI-174.

use axum::http::StatusCode;
use axum::Json;
use openfang_kernel::registry::AgentRegistry;
use openfang_types::agent::AgentId;

/// Longest handle we will even look at. Matches the name cap enforced by
/// `patch_agent_config` (routes.rs) so the two can't disagree.
pub const MAX_HANDLE_LEN: usize = 256;

/// Why a handle could not be turned into an `AgentId`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentRefError {
    /// The handle is not a UUID and could not be a legal agent name either.
    Malformed(String),
    /// The handle is well-formed, but no agent answers to it.
    NotFound(String),
}

impl AgentRefError {
    /// HTTP status this error maps to.
    pub fn status(&self) -> StatusCode {
        match self {
            AgentRefError::Malformed(_) => StatusCode::BAD_REQUEST,
            AgentRefError::NotFound(_) => StatusCode::NOT_FOUND,
        }
    }

    /// Human-readable message, safe to return to the caller.
    pub fn message(&self) -> &str {
        match self {
            AgentRefError::Malformed(m) => m,
            AgentRefError::NotFound(m) => m,
        }
    }

    /// Ready-to-return axum response tuple.
    pub fn into_response(self) -> (StatusCode, Json<serde_json::Value>) {
        let status = self.status();
        (status, Json(serde_json::json!({ "error": self.message() })))
    }
}

/// What a handle *looks* like, before we ask the registry anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentRef<'a> {
    /// Parsed cleanly as a UUID.
    Id(AgentId),
    /// Plausible agent name.
    Name(&'a str),
}

/// Purely syntactic classification. No registry, no I/O — this is the half
/// that decides 400-vs-maybe, and it is exhaustively unit-tested.
pub fn classify(handle: &str) -> Result<AgentRef<'_>, AgentRefError> {
    if handle.is_empty() {
        return Err(AgentRefError::Malformed(
            "Agent handle is empty; expected a UUID or an agent name".to_string(),
        ));
    }
    if handle.len() > MAX_HANDLE_LEN {
        return Err(AgentRefError::Malformed(format!(
            "Agent handle exceeds max length ({MAX_HANDLE_LEN} chars)"
        )));
    }

    // UUID first, always. An agent named "0000...-...." cannot shadow a real ID.
    if let Ok(u) = handle.parse::<uuid::Uuid>() {
        return Ok(AgentRef::Id(AgentId(u)));
    }

    // Not a UUID, so it has to be a name. Reject anything that could not be
    // one — control characters and path separators are the dangerous shapes,
    // and surrounding whitespace is almost always a caller bug.
    if handle.chars().any(|c| c.is_control()) {
        return Err(AgentRefError::Malformed(
            "Agent handle contains control characters".to_string(),
        ));
    }
    if handle.contains('/') || handle.contains('\\') {
        return Err(AgentRefError::Malformed(
            "Agent handle contains a path separator".to_string(),
        ));
    }
    if handle.trim() != handle || handle.trim().is_empty() {
        return Err(AgentRefError::Malformed(
            "Agent handle has leading or trailing whitespace".to_string(),
        ));
    }

    Ok(AgentRef::Name(handle))
}

/// The two registry questions resolution needs. A trait so the resolution
/// logic can be tested without standing up a kernel.
pub trait AgentLookup {
    /// ID bound to this exact name, if any.
    fn lookup_name(&self, name: &str) -> Option<AgentId>;
    /// Whether this ID is currently registered.
    fn has_id(&self, id: AgentId) -> bool;
}

impl AgentLookup for AgentRegistry {
    fn lookup_name(&self, name: &str) -> Option<AgentId> {
        self.id_for_name(name)
    }
    fn has_id(&self, id: AgentId) -> bool {
        self.get(id).is_some()
    }
}

/// Resolve a handle against any lookup source.
pub fn resolve_with<L: AgentLookup + ?Sized>(
    lookup: &L,
    handle: &str,
) -> Result<AgentId, AgentRefError> {
    match classify(handle)? {
        AgentRef::Id(id) => {
            if lookup.has_id(id) {
                Ok(id)
            } else {
                Err(AgentRefError::NotFound(format!("No such agent: {handle}")))
            }
        }
        AgentRef::Name(name) => lookup
            .lookup_name(name)
            .ok_or_else(|| AgentRefError::NotFound(format!("No such agent: {handle}"))),
    }
}

/// Route-facing entry point: resolve a handle, or hand back a response tuple
/// the caller can return directly.
///
/// ```ignore
/// let agent_id = match resolve_agent_ref(&state, &id) {
///     Ok(id) => id,
///     Err(resp) => return resp,
/// };
/// ```
pub fn resolve_agent_ref(
    state: &crate::routes::AppState,
    handle: &str,
) -> Result<AgentId, (StatusCode, Json<serde_json::Value>)> {
    resolve_with(&state.kernel.registry, handle).map_err(|e| e.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Registry stand-in: name → id. An id is "live" iff some name maps to it.
    struct FakeRegistry {
        names: HashMap<String, AgentId>,
    }

    impl FakeRegistry {
        fn with(pairs: &[(&str, AgentId)]) -> Self {
            Self {
                names: pairs
                    .iter()
                    .map(|(n, id)| ((*n).to_string(), *id))
                    .collect(),
            }
        }
    }

    impl AgentLookup for FakeRegistry {
        fn lookup_name(&self, name: &str) -> Option<AgentId> {
            self.names.get(name).copied()
        }
        fn has_id(&self, id: AgentId) -> bool {
            self.names.values().any(|v| *v == id)
        }
    }

    fn id(n: u128) -> AgentId {
        AgentId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn uuid_handle_classifies_as_id() {
        let u = id(42);
        assert_eq!(classify(&u.0.to_string()).unwrap(), AgentRef::Id(u));
    }

    #[test]
    fn plain_name_classifies_as_name() {
        assert_eq!(
            classify("researcher").unwrap(),
            AgentRef::Name("researcher")
        );
    }

    #[test]
    fn empty_handle_is_malformed() {
        assert!(matches!(classify(""), Err(AgentRefError::Malformed(_))));
        assert_eq!(classify("").unwrap_err().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn oversized_handle_is_malformed() {
        let long = "a".repeat(MAX_HANDLE_LEN + 1);
        assert!(matches!(classify(&long), Err(AgentRefError::Malformed(_))));
    }

    #[test]
    fn path_separators_and_control_chars_are_malformed() {
        for bad in ["../etc/passwd", "a/b", "a\\b", "na\nme", "tab\there"] {
            assert!(
                matches!(classify(bad), Err(AgentRefError::Malformed(_))),
                "expected {bad:?} to be malformed"
            );
        }
    }

    #[test]
    fn surrounding_whitespace_is_malformed() {
        assert!(matches!(
            classify(" researcher"),
            Err(AgentRefError::Malformed(_))
        ));
        assert!(matches!(
            classify("researcher "),
            Err(AgentRefError::Malformed(_))
        ));
        assert!(matches!(classify("   "), Err(AgentRefError::Malformed(_))));
    }

    #[test]
    fn resolves_by_name() {
        let reg = FakeRegistry::with(&[("researcher", id(1))]);
        assert_eq!(resolve_with(&reg, "researcher").unwrap(), id(1));
    }

    #[test]
    fn resolves_by_uuid() {
        let reg = FakeRegistry::with(&[("researcher", id(1))]);
        assert_eq!(resolve_with(&reg, &id(1).0.to_string()).unwrap(), id(1));
    }

    #[test]
    fn uuid_wins_over_a_name_that_looks_like_one() {
        // An agent perversely *named* the textual form of another agent's ID.
        let looks_like = id(1).0.to_string();
        let reg = FakeRegistry::with(&[(looks_like.as_str(), id(2))]);
        // UUID parse happens first, so we resolve toward agent 1 — not the
        // impostor. Agent 1 isn't registered here, so this surfaces as
        // NotFound rather than silently handing back agent 2.
        assert!(matches!(
            resolve_with(&reg, &looks_like),
            Err(AgentRefError::NotFound(_))
        ));
    }

    #[test]
    fn unknown_name_is_not_found_404() {
        let reg = FakeRegistry::with(&[("researcher", id(1))]);
        let err = resolve_with(&reg, "ghost").unwrap_err();
        assert_eq!(err.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn unknown_uuid_is_not_found_404_not_400() {
        // This is the contract change: pre-ANAI-174 this was a 400.
        let reg = FakeRegistry::with(&[("researcher", id(1))]);
        let err = resolve_with(&reg, &id(999).0.to_string()).unwrap_err();
        assert_eq!(err.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn malformed_stays_400() {
        let reg = FakeRegistry::with(&[("researcher", id(1))]);
        let err = resolve_with(&reg, "../../etc/passwd").unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }
}
