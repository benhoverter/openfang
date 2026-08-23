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
//! # A grammar that only refuses pushes callers back to event keys
//!
//! The empirical finding above has a catch, and `tttb-ben` supplied it while
//! reviewing this module: the key he would have minted unprompted
//! (`feature-extract-v2-25row-batch-state`) is an episode label, and refusing
//! it is correct — but the *thing he wanted to store* was real durable state,
//! "where am I in a multi-step promotion". If nothing legal holds in-flight
//! process state, a refusal just sends the caller back to inventing an event
//! key with a different spelling.
//!
//! So [`GRAMMAR_HINT`] offers the shape rather than only naming the rule:
//! one slot per process, overwritten as it advances
//! (`project.tttb.promotion_status = "blocked-on-zod"`), not one key per
//! attempt.
//!
//! # The namespace list is closed; depth is not
//!
//! Slot names are open within a namespace, and a key may carry up to five
//! qualifier segments between the namespace and the slot. Namespaces are a
//! hardcoded list, so minting one is a code change and therefore a review —
//! the "periodic key-space review" of §2.3.3 mitigation 3 paid for up front
//! instead of deferred to a curation job that may never run.
//!
//! Depth is what makes a *short* namespace list survivable. `kimiya-alpha`'s
//! argument, which is why the two-or-three-segment cap was lifted: her claims
//! all want `project.<project>.<subject>.<slot>`, and squashing the subject
//! into the qualifier (`project.kimiya-spike13.corpus_size`) loses the ability
//! to ask "everything believed about Kimiya". One deep subtree beats four
//! shallow trees, because with four trees a caller has to remember which tree
//! a fact lives in and will guess wrong.
//!
//! The nine namespaces below were reviewed by five agents against work they
//! actually had in flight (2026-08-21):
//!
//! - `git` became **`repo.<name>.<slot>`** — there are three repos, so
//!   `git.default_branch` is ambiguous the moment a second one exists.
//! - **`build`**, **`deploy`** and **`tool`** are separate because they churn
//!   on different clocks: which commit the fleet is running, how it is
//!   supervised, and what a given tool does are three different re-read
//!   cadences.
//! - `policy`, `hazard`, `decision` and `security` were **considered and
//!   cut** (Ben, 2026-08-21). Policies live in `RULES.md` and are injected;
//!   hazards are Linear tickets; a reversed decision is context rather than a
//!   correction, so it is append-shaped and belongs in tier 1, not a slot;
//!   `security.*` needs a daemon-side writer that does not exist yet, and a
//!   security namespace an agent can self-author reads authoritative while
//!   being a lie about its own guardrails.

use openfang_types::error::{OpenFangError, OpenFangResult};

/// The closed namespace list — the first segment of every claim key.
///
/// Grounded, not invented. `memory` and `project` are the ADR's own worked
/// examples (§2.3.2); `delivery` is the one durable dotted key the system
/// already mints; `agent` and `user` mirror the scope tiers and the `users/`
/// workspace subdir that curated memory already organises around; `repo`,
/// `build`, `deploy` and `tool` each came from an agent naming a claim it had
/// re-derived by hand more than once.
///
/// Adding an entry is a deliberate, reviewable act. See the module docs.
pub const NAMESPACES: &[&str] = &[
    "agent", "build", "deploy", "delivery", "memory", "project", "repo", "tool", "user",
];

/// Longest a single segment may be.
const MAX_SEGMENT: usize = 48;

/// Longest a whole key may be.
///
/// 128 bytes, matching the key cap `openfang-security` asked for: long enough
/// for a full-depth key, short enough that a key cannot smuggle prose into the
/// render path.
const MAX_KEY: usize = 128;

/// Most segments a key may have, namespace and slot included.
///
/// Seven, not five. `tttb-ben`'s real keys are already four deep and
/// `project.tttb.feature_parsed.segments.role_set` is a legitimate five — a
/// cap his second-shallowest example already touches is not a cap, it is a
/// tripwire.
const MAX_SEGMENTS: usize = 7;

/// Which tier of the world a fact is about (ADR 0001 §2.3.2).
///
/// Load-bearing for uniqueness together with `scope_ref`: the same claim key
/// under two scopes is two slots. A typo'd free-text scope would silently open
/// a second slot for the same claim — precisely the §2.3.3 dedup miss — which
/// is why this is a closed enum and not a `String`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactScope {
    /// About the agent itself. `scope_ref` is the agent's own id.
    Agent,
    /// About a project. `scope_ref` is the project slug, and the row belongs
    /// to the project rather than to whoever wrote it.
    Project,
    /// About a human. `scope_ref` is the user slug.
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
    /// `global` is **not**, by Ben's call of 2026-08-21: "global gets ignored
    /// for this epic." The reasoning survived a five-agent review — a
    /// fleet-wide claim renders into ~70 prompts forever, and every cap
    /// proposed to bound that (rate, count, provenance) is satisfied by the
    /// single well-chosen row that is the actual attack. The control is human
    /// promotion, which is a mechanism nobody has built yet.
    ///
    /// The variant exists so the stored form stays parseable and so the open
    /// question is represented in the type rather than forgotten. The writer
    /// refuses it, which turns an unanswered design question into a loud
    /// failure instead of a capability that shipped because nobody said no.
    pub fn is_shipped(self) -> bool {
        !matches!(self, FactScope::Global)
    }

    /// Whether this scope's `scope_ref` is supplied by the caller.
    ///
    /// `agent` derives it from the writer's own id — accepting a caller-chosen
    /// agent ref would let one agent write facts into another's slot space,
    /// which is the ANAI-165 boundary reopened from inside tier 3.
    pub fn takes_caller_ref(self) -> bool {
        !matches!(self, FactScope::Agent)
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

/// Validate the `scope_ref` half of a slot address.
///
/// `scope_ref` is who or what the fact is *about* — an agent id, a project
/// slug, a user slug. It is half the uniqueness key, so the same slug spelled
/// two ways is two slots holding contradictory claims, which is the §2.3.3
/// dedup miss arriving through the address rather than through the key. Same
/// slug rules as a qualifier segment, and never empty: SQLite treats NULLs as
/// distinct in a unique index, so an absent ref would silently disable the
/// constraint it is part of.
pub fn check_scope_ref(scope: FactScope, raw: &str) -> OpenFangResult<()> {
    if raw.is_empty() {
        return Err(OpenFangError::InvalidInput(format!(
            "a {scope} fact needs a scope_ref naming what it is about \
             (the project slug, e.g. \"openfang-fork\"); without one the slot has no owner \
             and the uniqueness constraint cannot hold"
        )));
    }
    if raw.len() > MAX_SEGMENT {
        return Err(OpenFangError::InvalidInput(format!(
            "scope_ref {raw:?} exceeds {MAX_SEGMENT} characters"
        )));
    }
    let mut chars = raw.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => {
            return Err(OpenFangError::InvalidInput(format!(
                "scope_ref {raw:?} must start with a lowercase letter or digit"
            )));
        }
    }
    if let Some(bad) =
        chars.find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_' || *c == '-'))
    {
        return Err(OpenFangError::InvalidInput(format!(
            "scope_ref {raw:?} contains {bad:?}; allowed: a-z, 0-9, '_' and '-'"
        )));
    }
    Ok(())
}

/// Resolve the subject half of a slot address.
///
/// One function, called by the write path and by every read path, because a
/// reader that resolves `scope_ref` differently from the writer addresses a
/// *different slot* and reports the claim missing. That reads as data loss,
/// and it is a two-line drift to introduce if each path does its own
/// resolution.
///
/// `agent` scope derives its ref from the caller's own id and ignores anything
/// supplied: honouring a caller-chosen agent ref would let one agent write
/// into another's slot space, which is the ANAI-165 boundary reopened from
/// inside tier 3. Every other scope demands one, because SQLite treats NULLs
/// as distinct inside a unique index and an absent ref would silently switch
/// off the constraint it is part of.
pub fn resolve_scope_ref(
    scope: FactScope,
    agent_id: &str,
    given: Option<&str>,
) -> OpenFangResult<String> {
    if !scope.takes_caller_ref() {
        return Ok(agent_id.to_string());
    }
    match given {
        Some(given) => {
            check_scope_ref(scope, given)?;
            Ok(given.to_string())
        }
        None => Err(OpenFangError::InvalidInput(format!(
            "a {scope}-scoped fact needs a scope_ref naming what it is about \
             (the project or user slug). Without one the slot has no subject, so \
             the claim would be filed under whoever happened to write it."
        ))),
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
    /// key       := namespace ("." qualifier)* "." slot     (2..=7 segments)
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
        if segments.len() < 2 {
            return Err(reject(
                raw,
                "a claim key needs a namespace: write 'namespace.slot', e.g. 'repo.trunk_model'",
            ));
        }
        if segments.len() > MAX_SEGMENTS {
            return Err(reject(
                raw,
                &format!("a claim key may have at most {MAX_SEGMENTS} segments"),
            ));
        }

        let namespace = segments[0];
        if !NAMESPACES.contains(&namespace) {
            return Err(reject(
                raw,
                &format!(
                    "{namespace:?} is not a known namespace; expected one of: {}",
                    NAMESPACES.join(", ")
                ),
            ));
        }

        let last = segments.len() - 1;
        for qual in &segments[1..last] {
            check_qualifier(raw, qual)?;
        }
        check_slot(raw, segments[last])?;

        Ok(ClaimKey(key.to_string()))
    }
}

impl std::fmt::Display for ClaimKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The grammar, quoted back to whoever got it wrong.
///
/// The last sentence is doing work the rest of the module cannot: it names the
/// shape a caller should reach for when what they actually have is in-flight
/// process state. Refusal alone sends them back to minting event keys.
pub const GRAMMAR_HINT: &str = "Claim keys name a durable slot, not an event: \
'namespace.slot' (e.g. 'repo.trunk_model', 'memory.sweep_status'), with up to five \
qualifier segments in between for project- and repo-scoped claims \
('project.openfang-fork.memory.owner'). Lowercase, words joined by '_'; qualifiers may \
hyphenate, slots may not. A key must not encode a date or a ticket id — those describe \
when something was written, not what is true. If what you have is progress through a \
multi-step task, that IS a durable slot: name the process once and overwrite it as it \
advances ('project.<slug>.promotion_status' = \"blocked-on-zod\"), rather than minting a \
new key per attempt.";

/// Build a rejection that a model can act on.
fn reject(raw: &str, why: &str) -> OpenFangError {
    OpenFangError::InvalidInput(format!("rejected claim key {raw:?}: {why}. {GRAMMAR_HINT}"))
}

/// The `n` existing keys closest to `target` by edit distance.
///
/// Used to shape a rejection into a short menu instead of a dump of the key
/// space. `openfang-security`'s catch, and it is a real one: rejection text
/// lands in the transcript, and the transcript is memory-captured — so an
/// error that echoes 25 keys seeds tier-1 memory with a tier-3 enumeration
/// every time a model fat-fingers a key. Three near misses keep all of the
/// utility (the caller is trying to *select*, and a near miss is what they
/// meant) while dropping the enumeration to nothing.
pub fn nearest_keys(candidates: &[String], target: &str, n: usize) -> Vec<String> {
    let mut scored: Vec<(usize, &String)> = candidates
        .iter()
        .map(|c| (edit_distance(c, target), c))
        .collect();
    // Distance first, then lexical, so the suggestion list is deterministic —
    // a rejection that reorders between identical calls reads as nondeterminism
    // to whoever is debugging it.
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(n).map(|(_, c)| c.clone()).collect()
}

/// Levenshtein distance, two-row form.
///
/// Local rather than a dependency: it runs on a bounded candidate list in a
/// path that has already failed, and a new crate in the memory subsystem costs
/// more review than forty lines of textbook DP.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
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
            "repo.trunk_model",
            "memory.sweep_status",
            "project.openfang-fork.owner",
            "delivery.last_channel",
        ] {
            ClaimKey::parse(key).unwrap_or_else(|e| panic!("{key} should parse: {e}"));
        }
    }

    /// The keys five agents named as things they had re-derived by hand. If
    /// the namespace list ever stops covering these, it has drifted from the
    /// review that produced it.
    #[test]
    fn reviewed_agent_keys_are_legal() {
        for key in [
            "repo.openfang.trunk_model",
            "deploy.supervisor",
            "deploy.cron-eviction.last_run",
            "build.fleet.running_sha",
            "tool.memory_fact.status",
            "project.tttb.feature_parsed.promotion_status",
            "project.tttb.feature_parsed.segments.role_set",
            "user.ben.timezone",
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
        assert!(err.contains("repo"), "{err}");
    }

    /// `git` was renamed to `repo` in the 2026-08-21 review: three repos exist,
    /// so `git.default_branch` is ambiguous the moment a second one does.
    #[test]
    fn git_namespace_was_replaced_by_repo() {
        assert!(ClaimKey::parse("git.trunk_model").is_err());
        assert!(ClaimKey::parse("repo.trunk_model").is_ok());
    }

    /// The four namespaces cut on 2026-08-21 must stay cut, or the tier
    /// quietly grows a push surface nobody built the render rules for.
    #[test]
    fn cut_namespaces_stay_cut() {
        for key in [
            "policy.no_force_push",
            "hazard.bridge_timeout",
            "decision.dice_union",
            "security.gatekeeper_enabled",
        ] {
            assert!(
                ClaimKey::parse(key).is_err(),
                "{key} names a namespace that was considered and cut"
            );
        }
    }

    #[test]
    fn bare_slot_is_rejected_with_a_worked_example() {
        let err = ClaimKey::parse("trunk_model").unwrap_err().to_string();
        assert!(err.contains("needs a namespace"), "{err}");
        assert!(err.contains("repo.trunk_model"), "{err}");
    }

    /// A refusal has to offer the shape for in-flight state, or the caller
    /// goes back to minting an event key with a different spelling.
    #[test]
    fn rejection_offers_the_process_state_shape() {
        let err = ClaimKey::parse("feature-extract-v2-25row-batch-state")
            .unwrap_err()
            .to_string();
        assert!(err.contains("promotion_status"), "{err}");
        assert!(err.contains("overwrite it as it advances"), "{err}");
    }

    /// Depth is what lets a nine-namespace list absorb categories nobody
    /// anticipated — one deep subtree beats four shallow trees.
    #[test]
    fn qualifiers_are_allowed_in_any_namespace_and_at_depth() {
        assert!(ClaimKey::parse("repo.openfang.trunk_model").is_ok());
        assert!(ClaimKey::parse("project.a.b.c").is_ok());
        assert!(ClaimKey::parse("project.a.b.c.d.e.f").is_ok(), "seven deep");
    }

    #[test]
    fn depth_is_capped() {
        let err = ClaimKey::parse("project.a.b.c.d.e.f.g")
            .unwrap_err()
            .to_string();
        assert!(err.contains("at most 7 segments"), "{err}");
    }

    #[test]
    fn slot_may_not_hyphenate_but_qualifier_may() {
        assert!(ClaimKey::parse("project.of-mem-fact.owner").is_ok());
        assert!(ClaimKey::parse("repo.trunk-model").is_err());
    }

    #[test]
    fn case_and_whitespace_rejected() {
        assert!(ClaimKey::parse("repo.Trunk_Model").is_err());
        assert!(ClaimKey::parse("Repo.trunk_model").is_err());
        assert!(ClaimKey::parse(" repo.trunk_model").is_err());
        assert!(ClaimKey::parse("repo.trunk model").is_err());
    }

    #[test]
    fn empty_segments_rejected() {
        for key in ["", ".", "repo.", ".trunk_model", "repo..trunk_model"] {
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
        assert!(ClaimKey::parse(&format!("repo.{long_slot}")).is_err());
        let long_key = "a".repeat(MAX_KEY);
        assert!(ClaimKey::parse(&format!("repo.{long_key}")).is_err());
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
        // Ben, 2026-08-21: global is out for this epic. Parse it so stored
        // data stays readable; refuse to write it so the question cannot be
        // settled by accident.
        assert_eq!(FactScope::parse("global").unwrap(), FactScope::Global);
        assert!(!FactScope::Global.is_shipped());
        for scope in [FactScope::Agent, FactScope::Project, FactScope::User] {
            assert!(scope.is_shipped());
        }
    }

    /// An agent may not choose the `scope_ref` of an `agent`-scoped fact:
    /// that would be one agent writing into another's slot space, which is the
    /// ANAI-165 boundary reopened from inside tier 3.
    #[test]
    fn only_agent_scope_derives_its_own_ref() {
        assert!(!FactScope::Agent.takes_caller_ref());
        for scope in [FactScope::Project, FactScope::User, FactScope::Global] {
            assert!(scope.takes_caller_ref());
        }
    }

    #[test]
    fn scope_ref_rules_match_qualifier_rules() {
        assert!(check_scope_ref(FactScope::Project, "openfang-fork").is_ok());
        assert!(check_scope_ref(FactScope::Project, "tttb").is_ok());
        // Empty is the dangerous one: SQLite treats NULLs and would treat an
        // absent ref as distinct, silently disabling the constraint.
        assert!(check_scope_ref(FactScope::Project, "").is_err());
        assert!(check_scope_ref(FactScope::Project, "OpenFang").is_err());
        assert!(check_scope_ref(FactScope::Project, "of fork").is_err());
    }

    #[test]
    fn nearest_keys_ranks_by_edit_distance_and_is_deterministic() {
        let keys: Vec<String> = [
            "repo.trunk_model",
            "memory.sweep_status",
            "delivery.last_channel",
            "repo.trunk_mode",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let near = nearest_keys(&keys, "repo.trunk_modl", 3);
        assert_eq!(near.len(), 3);
        assert_eq!(near[0], "repo.trunk_mode");
        assert_eq!(near[1], "repo.trunk_model");
        assert_eq!(near, nearest_keys(&keys, "repo.trunk_modl", 3));
    }

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }
}
