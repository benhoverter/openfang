//! The tier-3 controlled vocabulary: `claim_key` grammar and fact `scope`
//! (ADR 0001 §2.3.2, §2.3.3 mitigation 1).
//!
//! # Why this exists at all
//!
//! §2.3.3 is explicit that the keyed-slot design relocates its failure rather
//! than removing it. The uniqueness constraint makes *stale-beside-current*
//! unrepresentable, but it cannot stop **two live rows under two different
//! keys saying contradictory things** — a dedup miss. The first and most
//! trusted mitigation the ADR names is the controlled vocabulary:
//!
//! > Consolidation *selects* a key from the existing key space and may only
//! > mint a new one when nothing fits. Far narrower than "invent a name."
//!
//! A grammar cannot make a model pick the *right* key. What it can do is make
//! the key space small enough that "does one of these already fit?" is a
//! question with a short answer — and refuse the shapes that guarantee a miss.
//!
//! # What the real key space looks like today
//!
//! This module's namespace list is not invented. ADR 0001 §5.4 asked for a
//! read-only pass over real content before the constraint ships; the closest
//! existing analogue to a claim-key space is `kv_store`, whose keys have been
//! free text since day one. Of 1,332 keys:
//!
//! - **203** are ticket-scoped (`anai-196-bridge-fix-state`, `anrd87-g7-...`)
//! - **12** are date-prefixed (`2026-06-19-mem-heartbeat-reroot-d8963ef`)
//! - **216** contain a dot, but nearly all of those are one key,
//!   `delivery.last_channel`, written many times
//!
//! The pattern is unambiguous. Every key a *model* minted is an episode label
//! — a thing that happened, stamped with when it happened. The one key the
//! *system* mints, `delivery.last_channel`, is a durable slot in exactly the
//! `namespace.slot` shape the ADR specifies. So the grammar below is drawn
//! from the observed good case and deliberately rejects the observed bad one:
//! a date or a ticket id cannot be a namespace, because a claim about *state*
//! does not belong to the moment it was written.
//!
//! # The namespace list is closed on purpose
//!
//! Slot names are open within a namespace — a new claim about git can name
//! itself. Namespaces are a hardcoded list, so minting one is a code change
//! and therefore a review, which is the "periodic key-space review" of §2.3.3
//! mitigation 3 paid for up front instead of deferred to a curation job that
//! may never run. Six is a starting point, not a ceiling; growing it is meant
//! to be easy but *visible*.

use openfang_types::error::{OpenFangError, OpenFangResult};

/// The closed namespace list — the first segment of every claim key.
///
/// Grounded, not invented: `git`, `memory` and `project` are the ADR's own
/// worked examples (§2.3.2); `delivery` is the one durable dotted key the
/// system already mints; `agent` and `user` mirror the scope tiers and the
/// `users/` workspace subdir that curated memory already organises around.
///
/// Adding an entry is a deliberate, reviewable act. See the module docs.
pub const NAMESPACES: &[&str] = &["agent", "delivery", "git", "memory", "project", "user"];

/// Namespaces that take a middle qualifier segment, e.g.
/// `project.<slug>.owner`.
///
/// Three-segment keys are allowed only here. Left open to every namespace,
/// `git.<anything>.status` would reintroduce the free-text key space this
/// module exists to close.
pub const QUALIFIED_NAMESPACES: &[&str] = &["project", "user"];

/// Longest a single segment may be.
const MAX_SEGMENT: usize = 48;

/// Longest a whole key may be.
const MAX_KEY: usize = 96;

/// Which tier of the world a fact is about (ADR 0001 §2.3.2).
///
/// This is the second component of the slot key, so it is *load-bearing for
/// uniqueness*: the same claim key under two scopes is two live rows. A typo'd
/// free-text scope would therefore silently open a second slot for the same
/// claim — precisely the §2.3.3 dedup miss — which is why this is a closed
/// enum and not a `String`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactScope {
    /// About the agent itself.
    Agent,
    /// About a project. Usually paired with a qualified key.
    Project,
    /// About a human.
    User,
    /// True for everyone, everywhere. See [`FactScope::is_shipped`].
    Global,
}

impl FactScope {
    /// The stored form, landing in `memories.scope`.
    pub fn as_str(self) -> &'static str {
        match self {
            FactScope::Agent => "agent",
            FactScope::Project => "project",
            FactScope::User => "user",
            FactScope::Global => "global",
        }
    }

    /// Every variant, for error messages and tests.
    pub fn all() -> [FactScope; 4] {
        [
            FactScope::Agent,
            FactScope::Project,
            FactScope::User,
            FactScope::Global,
        ]
    }

    /// Whether writing this scope is actually in scope for the shipped system.
    ///
    /// `global` is **not**. ADR 0001 §5.2 leaves it an open question — "does a
    /// `scope = global` fact render into every agent's prompt, and who is
    /// allowed to write one?" — and notes it is the same threat surface
    /// ANAI-165 closed and should not be reopened casually.
    ///
    /// The variant exists so the stored form is parseable and so the open
    /// question is represented in the type rather than forgotten. The writer
    /// refuses it, which turns an unanswered design question into a loud
    /// failure instead of a capability that shipped because nobody said no.
    pub fn is_shipped(self) -> bool {
        !matches!(self, FactScope::Global)
    }

    /// Parse a scope, rejecting anything outside the vocabulary.
    ///
    /// Note that this rejects `episodic`, which is the scope on all 13.5k
    /// existing rows. That is correct and not a migration hazard: the tier-3
    /// vocabulary governs `kind = 'fact'` rows only, and no fact rows existed
    /// before v14.
    pub fn parse(s: &str) -> OpenFangResult<Self> {
        match s {
            "agent" => Ok(FactScope::Agent),
            "project" => Ok(FactScope::Project),
            "user" => Ok(FactScope::User),
            "global" => Ok(FactScope::Global),
            other => Err(OpenFangError::InvalidInput(format!(
                "invalid fact scope {other:?}; expected one of: agent, project, user, global"
            ))),
        }
    }
}

impl std::fmt::Display for FactScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A validated claim key.
///
/// Constructing one is the only way to assert a key is in the vocabulary, so
/// a function taking `ClaimKey` cannot be handed free text by accident.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimKey(String);

impl ClaimKey {
    /// The stored form.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The namespace — first segment, guaranteed to be in [`NAMESPACES`].
    pub fn namespace(&self) -> &str {
        self.0.split('.').next().unwrap_or_default()
    }

    /// Validate a key against the grammar.
    ///
    /// ```text
    /// key       := namespace "." slot
    ///            | qualified_namespace "." qualifier "." slot
    /// namespace := one of NAMESPACES
    /// qualifier := [a-z0-9][a-z0-9_-]*      (slugs may hyphenate)
    /// slot      := [a-z][a-z0-9_]*          (slots may not)
    /// ```
    ///
    /// Errors are written for the model that chose the key, not for a log:
    /// each one says what was wrong *and* what the legal space is, because a
    /// rejection the caller cannot act on just becomes a retry loop.
    pub fn parse(raw: &str) -> OpenFangResult<Self> {
        let key = raw.trim();

        if key.is_empty() {
            return Err(reject(raw, "a claim key may not be empty"));
        }
        if key.len() > MAX_KEY {
            return Err(reject(
                raw,
                &format!("a claim key may be at most {MAX_KEY} characters"),
            ));
        }
        if key != raw {
            return Err(reject(
                raw,
                "a claim key may not have leading or trailing whitespace",
            ));
        }

        let segments: Vec<&str> = key.split('.').collect();
        if segments.iter().any(|s| s.is_empty()) {
            return Err(reject(
                raw,
                "a claim key may not have an empty segment (no leading, trailing or doubled '.')",
            ));
        }

        let (namespace, qualifier, slot) = match segments.as_slice() {
            [ns, slot] => (*ns, None, *slot),
            [ns, qual, slot] => (*ns, Some(*qual), *slot),
            [_] => {
                return Err(reject(
                    raw,
                    "a claim key needs a namespace: write 'namespace.slot', e.g. 'git.trunk_model'",
                ));
            }
            _ => {
                return Err(reject(
                    raw,
                    "a claim key may have at most three segments (namespace.qualifier.slot)",
                ));
            }
        };

        if !NAMESPACES.contains(&namespace) {
            return Err(reject(
                raw,
                &format!(
                    "{namespace:?} is not a known namespace; expected one of: {}",
                    NAMESPACES.join(", ")
                ),
            ));
        }

        if let Some(qual) = qualifier {
            if !QUALIFIED_NAMESPACES.contains(&namespace) {
                return Err(reject(
                    raw,
                    &format!(
                        "the {namespace:?} namespace does not take a middle qualifier; \
                         write '{namespace}.slot'. Qualifiers are for: {}",
                        QUALIFIED_NAMESPACES.join(", ")
                    ),
                ));
            }
            check_qualifier(raw, qual)?;
        }

        check_slot(raw, slot)?;

        Ok(ClaimKey(key.to_string()))
    }
}

impl std::fmt::Display for ClaimKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The grammar, quoted back to whoever got it wrong.
const GRAMMAR_HINT: &str = "Claim keys name a durable slot, not an event: \
'namespace.slot' (e.g. 'git.trunk_model', 'memory.sweep_status') or \
'project.<slug>.slot' for project- and user-qualified claims. Lowercase, \
words joined by '_'. A key must not encode a date or a ticket id — those \
describe when something was written, not what is true.";

/// Build a rejection that a model can act on.
fn reject(raw: &str, why: &str) -> OpenFangError {
    OpenFangError::InvalidInput(format!("rejected claim key {raw:?}: {why}. {GRAMMAR_HINT}"))
}

/// Slot names are strict: they are the part a model most wants to improvise.
fn check_slot(raw: &str, slot: &str) -> OpenFangResult<()> {
    if slot.len() > MAX_SEGMENT {
        return Err(reject(
            raw,
            &format!("segment {slot:?} exceeds {MAX_SEGMENT} characters"),
        ));
    }
    let mut chars = slot.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => {
            return Err(reject(
                raw,
                &format!("segment {slot:?} must start with a lowercase letter"),
            ));
        }
    }
    if let Some(bad) = chars.find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_'))
    {
        return Err(reject(
            raw,
            &format!("segment {slot:?} contains {bad:?}; slot names allow only a-z, 0-9 and '_'"),
        ));
    }
    Ok(())
}

/// Qualifiers are slugs of real names, so they may hyphenate — `of-mem-fact`
/// is the actual name of a thing, and forcing it to `of_mem_fact` would invite
/// two spellings of one slot.
fn check_qualifier(raw: &str, qual: &str) -> OpenFangResult<()> {
    if qual.len() > MAX_SEGMENT {
        return Err(reject(
            raw,
            &format!("segment {qual:?} exceeds {MAX_SEGMENT} characters"),
        ));
    }
    let mut chars = qual.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => {
            return Err(reject(
                raw,
                &format!("segment {qual:?} must start with a lowercase letter or digit"),
            ));
        }
    }
    if let Some(bad) =
        chars.find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_' || *c == '-'))
    {
        return Err(reject(
            raw,
            &format!(
                "segment {qual:?} contains {bad:?}; qualifiers allow only a-z, 0-9, '_' and '-'"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adr_examples_are_legal() {
        for key in [
            "git.trunk_model",
            "memory.sweep_status",
            "project.openfang-fork.owner",
            "delivery.last_channel",
        ] {
            ClaimKey::parse(key).unwrap_or_else(|e| panic!("{key} should parse: {e}"));
        }
    }

    /// The empirical case from the module docs: this is what the free-text key
    /// space actually filled up with, and none of it may become a claim key.
    #[test]
    fn observed_kv_store_shapes_are_rejected() {
        for key in [
            "2026-06-19-mem-heartbeat-reroot-d8963ef",
            "anai-196-bridge-fix-state",
            "anrd87-g7-tier1-lock-locked",
            "anai204_progress",
            "references/openfang-test-instance/skill-v0.2.1",
            "kimiya/contracts/matcher-output-line-v1.1.0",
        ] {
            assert!(
                ClaimKey::parse(key).is_err(),
                "{key} is episode-shaped and must not be a claim key"
            );
        }
    }

    #[test]
    fn namespace_must_be_known() {
        let err = ClaimKey::parse("weather.today").unwrap_err().to_string();
        assert!(err.contains("not a known namespace"), "{err}");
        // The rejection names the legal space, so the caller can retry once
        // rather than guessing.
        assert!(err.contains("git"), "{err}");
    }

    #[test]
    fn bare_slot_is_rejected_with_a_worked_example() {
        let err = ClaimKey::parse("trunk_model").unwrap_err().to_string();
        assert!(err.contains("needs a namespace"), "{err}");
        assert!(err.contains("git.trunk_model"), "{err}");
    }

    #[test]
    fn qualifier_only_where_allowed() {
        assert!(ClaimKey::parse("user.ben.timezone").is_ok());
        let err = ClaimKey::parse("git.openfang.trunk_model")
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not take a middle qualifier"), "{err}");
    }

    #[test]
    fn four_segments_rejected() {
        let err = ClaimKey::parse("project.a.b.c").unwrap_err().to_string();
        assert!(err.contains("at most three segments"), "{err}");
    }

    #[test]
    fn slot_may_not_hyphenate_but_qualifier_may() {
        assert!(ClaimKey::parse("project.of-mem-fact.owner").is_ok());
        assert!(ClaimKey::parse("git.trunk-model").is_err());
    }

    #[test]
    fn case_and_whitespace_rejected() {
        assert!(ClaimKey::parse("git.Trunk_Model").is_err());
        assert!(ClaimKey::parse("Git.trunk_model").is_err());
        assert!(ClaimKey::parse(" git.trunk_model").is_err());
        assert!(ClaimKey::parse("git.trunk model").is_err());
    }

    #[test]
    fn empty_segments_rejected() {
        for key in ["", ".", "git.", ".trunk_model", "git..trunk_model"] {
            assert!(ClaimKey::parse(key).is_err(), "{key:?} should be rejected");
        }
    }

    #[test]
    fn slot_may_not_start_with_a_digit() {
        // Guards the date-shaped key at segment level, not just at namespace
        // level: `memory.2026_sweep` is the same mistake wearing a legal
        // namespace.
        assert!(ClaimKey::parse("memory.2026_sweep").is_err());
        assert!(ClaimKey::parse("memory.sweep_2026").is_ok());
    }

    #[test]
    fn length_bounded() {
        let long_slot = "a".repeat(MAX_SEGMENT + 1);
        assert!(ClaimKey::parse(&format!("git.{long_slot}")).is_err());
        let long_key = "a".repeat(MAX_KEY);
        assert!(ClaimKey::parse(&format!("git.{long_key}")).is_err());
    }

    #[test]
    fn namespace_accessor_matches_first_segment() {
        let key = ClaimKey::parse("project.openfang-fork.owner").unwrap();
        assert_eq!(key.namespace(), "project");
        assert!(NAMESPACES.contains(&key.namespace()));
    }

    #[test]
    fn scope_round_trips() {
        for scope in FactScope::all() {
            assert_eq!(FactScope::parse(scope.as_str()).unwrap(), scope);
        }
    }

    #[test]
    fn legacy_episodic_scope_is_not_a_fact_scope() {
        // All 13.5k pre-v14 rows carry scope='episodic'. It is not a tier-3
        // scope and must not become one by accident.
        assert!(FactScope::parse("episodic").is_err());
    }

    #[test]
    fn unknown_scope_names_the_vocabulary() {
        let err = FactScope::parse("porject").unwrap_err().to_string();
        assert!(err.contains("agent, project, user, global"), "{err}");
    }

    #[test]
    fn global_scope_is_parseable_but_not_shipped() {
        // ADR 0001 §5.2 is unanswered. Parse it so stored data is readable;
        // refuse to write it so the question cannot be settled by accident.
        assert_eq!(FactScope::parse("global").unwrap(), FactScope::Global);
        assert!(!FactScope::Global.is_shipped());
        for scope in [FactScope::Agent, FactScope::Project, FactScope::User] {
            assert!(scope.is_shipped());
        }
    }
}
