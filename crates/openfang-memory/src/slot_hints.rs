//! Write-time slot hints (ANAI-277).
//!
//! The supersession machinery — one live row per `(scope, scope_ref,
//! claim_key)`, displaced claims preserved in `fact_history` — had never fired
//! once as of 2026-09-15. Not because it was broken: because agents mint a new
//! key instead of rewriting a slot. `kimiya-alpha` held 9 facts under 9
//! distinct keys with zero second versions, including three spellings of one
//! claim, twice:
//!
//! ```text
//! project.kimiya.matrix_baselines → matrix_baseline_state → matrix_baseline_recording
//! repo.inference_vendor → repo.pinned_model → project.kimiya.inference_pin
//! ```
//!
//! `memory_fact`'s description already said *"prefer a key that already exists
//! over minting a near-duplicate"* — an instruction asking for recognition
//! while supplying nothing to recognise against. Two rounds of reworded prompt
//! (ANAI-267 D1, ANAI-274 D1) moved note adoption and left slot proliferation
//! untouched, which is the evidence that the missing input is data and not
//! prose.
//!
//! So this module answers one question at the only moment it is actionable:
//! **when a write mints a NEW slot, which addresses does this agent already
//! own?** The ranking below is a convenience on that list, not a gate. It is
//! lexical and therefore blind to synonymy — it catches the `matrix_baseline*`
//! chain and *cannot* catch `inference_vendor` vs `pinned_model`. That is
//! precisely why the full list is returned and merely ordered: the agent holds
//! the semantics, the tool holds the addresses, and guessing on the agent's
//! behalf is the one thing that could destroy a claim.

use std::collections::BTreeSet;

/// How many neighbouring addresses a write-time hint carries by default.
///
/// Small on purpose. The hint rides a success message the agent reads in
/// passing; a list long enough to need scanning is a list that gets skipped.
pub const NEIGHBOUR_CAP: usize = 8;

/// At or above this score the top neighbour is named as a likely duplicate
/// rather than merely listed first.
pub const LIKELY_DUPLICATE_SCORE: f64 = 0.5;

/// The claim-key half of a slot address.
///
/// Addresses arrive in [`crate::memory_md::display_key`] spelling — bare for
/// agent scope, `slug/key` for project scope — because status, managed block
/// and this hint must all spell one slot one way. Similarity is computed on
/// the key alone; the display slug is addressing, not name.
fn claim_key_of(address: &str) -> &str {
    match address.rsplit_once('/') {
        Some((_, key)) => key,
        None => address,
    }
}

/// The discriminating tail of a claim key.
///
/// Keys are `namespace.slot` with optional qualifiers between
/// (`project.kimiya.matrix_baseline_state`), and the leading segments are
/// *address*, not name: comparing them would score every slot in a project as
/// half a match for every other, which fires the warning on almost every write
/// a project-scoped agent makes. Comparing tails also keeps the cross-scope
/// case — `repo.trunk_head` against `project.kimiya.trunk_head` is a real
/// duplicate and the measured `inference_vendor` drift crossed scopes exactly
/// that way.
fn slot_of(key: &str) -> &str {
    match key.rsplit_once('.') {
        Some((_, slot)) => slot,
        None => key,
    }
}

/// Split a slot name into comparable tokens.
///
/// Lowercased, split on the separators the key vocabulary actually uses, and
/// de-pluralised crudely — `matrix_baselines` and `matrix_baseline_state` are
/// the same subject spelled twice, and a comparison that missed that would
/// miss half of the measured evidence for this ticket.
fn tokens(key: &str) -> BTreeSet<String> {
    key.split(|c: char| c == '.' || c == '_' || c == '-' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            let lower = t.to_lowercase();
            match lower.strip_suffix('s') {
                Some(stem) if stem.chars().count() >= 3 => stem.to_string(),
                _ => lower,
            }
        })
        .collect()
}

/// Lexical similarity of two claim keys, in `0.0..=1.0`.
///
/// Jaccard over the token sets, lifted when one key's tokens are wholly
/// contained in the other's. Containment is the shape slot drift actually
/// takes — a key grows a qualifier (`matrix_baseline` →
/// `matrix_baseline_state`) rather than being rewritten — and raw Jaccard
/// punishes exactly that case for the length difference it is diagnostic of.
pub fn similarity(a: &str, b: &str) -> f64 {
    let (ta, tb) = (tokens(slot_of(a)), tokens(slot_of(b)));
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let shared = ta.intersection(&tb).count();
    if shared == 0 {
        return 0.0;
    }
    let union = ta.union(&tb).count();
    let jaccard = shared as f64 / union as f64;
    if shared == ta.len().min(tb.len()) {
        jaccard.max(0.75)
    } else {
        jaccard
    }
}

/// The addresses an agent already owns, ordered so the best guess reads first.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RankedNeighbours {
    /// Up to `cap` addresses, most similar to the new key first.
    pub addresses: Vec<String>,
    /// How many were available before the cap, so a truncated list can say so.
    pub total: usize,
    /// The top address, when it scored at or above
    /// [`LIKELY_DUPLICATE_SCORE`]. Advisory: the write already happened.
    pub likely_duplicate: Option<String>,
}

/// Rank `existing` addresses by similarity to `candidate_key`.
///
/// The candidate's own address is expected to be absent — the caller runs this
/// only on a *created* outcome, where the slot is new by definition — but an
/// exact key match is dropped anyway rather than reported as its own
/// duplicate.
///
/// Ordering is score descending, then address ascending. The second key is
/// what makes the output deterministic: most of a real corpus scores zero
/// against any given new key, and a hint that reshuffled its own tail between
/// two identical calls would read as new information.
pub fn rank_neighbours(candidate_key: &str, existing: &[String], cap: usize) -> RankedNeighbours {
    let mut scored: Vec<(f64, &String)> = existing
        .iter()
        .filter(|addr| claim_key_of(addr) != candidate_key)
        .map(|addr| (similarity(candidate_key, claim_key_of(addr)), addr))
        .collect();

    scored.sort_by(|(sa, aa), (sb, ab)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| aa.cmp(ab))
    });

    let total = scored.len();
    let likely_duplicate = scored
        .first()
        .filter(|(score, _)| *score >= LIKELY_DUPLICATE_SCORE)
        .map(|(_, addr)| (*addr).clone());

    RankedNeighbours {
        addresses: scored
            .into_iter()
            .take(cap)
            .map(|(_, addr)| addr.clone())
            .collect(),
        total,
        likely_duplicate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addrs(keys: &[&str]) -> Vec<String> {
        keys.iter().map(|k| (*k).to_string()).collect()
    }

    #[test]
    fn the_measured_matrix_baseline_chain_is_caught() {
        // kimiya-alpha's real chain, the half a lexical comparison can see.
        let ranked = rank_neighbours(
            "project.kimiya.matrix_baseline_recording",
            &addrs(&[
                "kimiya/project.kimiya.matrix_baselines",
                "kimiya/project.kimiya.matrix_baseline_state",
                "kimiya/repo.inference_vendor",
            ]),
            NEIGHBOUR_CAP,
        );
        // The bare plural is a subset of the new key, which is the strongest
        // signal available — a key that grew a qualifier is the exact shape
        // this drift takes — so it leads, with the sibling spelling behind it.
        assert_eq!(
            ranked.likely_duplicate.as_deref(),
            Some("kimiya/project.kimiya.matrix_baselines"),
            "the nearest spelling of the same subject was not named"
        );
        assert_eq!(
            ranked.addresses[..2],
            addrs(&[
                "kimiya/project.kimiya.matrix_baselines",
                "kimiya/project.kimiya.matrix_baseline_state",
            ])[..],
            "both spellings of the subject must outrank an unrelated key"
        );
        assert_eq!(ranked.addresses[2], "kimiya/repo.inference_vendor");
    }

    #[test]
    fn the_synonym_chain_is_listed_but_not_claimed() {
        // The other measured chain. `inference_vendor` and `pinned_model` are
        // one claim in two vocabularies; nothing lexical can know that. The
        // contract is that the address is still SHOWN, unranked, so the agent
        // can recognise it — and that we do not assert a duplicate we cannot
        // support.
        let ranked = rank_neighbours(
            "inference_pin",
            &addrs(&["repo.pinned_model", "repo.trunk_head"]),
            NEIGHBOUR_CAP,
        );
        assert_eq!(ranked.likely_duplicate, None);
        assert_eq!(ranked.addresses.len(), 2);
        assert!(ranked.addresses.contains(&"repo.pinned_model".to_string()));
    }

    #[test]
    fn plurals_are_the_same_subject() {
        assert!(similarity("matrix_baselines", "matrix_baseline") >= LIKELY_DUPLICATE_SCORE);
    }

    #[test]
    fn a_shared_namespace_alone_is_not_a_duplicate() {
        // Both live under `repo.`; nothing else is shared. Scoring these as
        // near-duplicates would fire the warning on almost every write.
        assert!(similarity("repo.trunk_head", "repo.inference_vendor") < LIKELY_DUPLICATE_SCORE);
    }

    /// The leading segments of a key are an address, not a name. Scoring them
    /// would make every slot in a project half a match for every other and
    /// fire the warning on almost every write a project-scoped agent makes.
    #[test]
    fn a_shared_project_address_is_not_a_duplicate() {
        assert_eq!(
            similarity(
                "project.kimiya.swap_schema",
                "project.kimiya.matrix_baselines"
            ),
            0.0
        );
    }

    /// The measured `repo.inference_vendor` → `project.kimiya.inference_pin`
    /// drift crossed scopes. Comparing tails is what keeps that visible.
    #[test]
    fn one_slot_name_under_two_addresses_is_a_duplicate() {
        assert!(
            similarity("repo.trunk_head", "project.openfang.trunk_head") >= LIKELY_DUPLICATE_SCORE
        );
    }

    #[test]
    fn a_key_is_not_its_own_neighbour() {
        let ranked = rank_neighbours(
            "repo.trunk_head",
            &addrs(&["repo.trunk_head", "openfang/repo.trunk_head"]),
            NEIGHBOUR_CAP,
        );
        assert!(ranked.addresses.is_empty());
        assert_eq!(ranked.total, 0);
        assert_eq!(ranked.likely_duplicate, None);
    }

    #[test]
    fn the_cap_truncates_but_the_total_does_not() {
        let existing = addrs(&["a.one", "b.two", "c.three", "d.four"]);
        let ranked = rank_neighbours("z.new", &existing, 2);
        assert_eq!(ranked.addresses.len(), 2);
        assert_eq!(ranked.total, 4, "the count must survive the truncation");
    }

    #[test]
    fn ordering_is_deterministic_across_identical_calls() {
        let existing = addrs(&["b.zero", "a.zero", "c.zero"]);
        let first = rank_neighbours("q.new", &existing, NEIGHBOUR_CAP);
        let second = rank_neighbours("q.new", &existing, NEIGHBOUR_CAP);
        assert_eq!(first, second);
        assert_eq!(first.addresses, addrs(&["a.zero", "b.zero", "c.zero"]));
    }

    #[test]
    fn an_empty_corpus_hints_nothing() {
        let ranked = rank_neighbours("repo.trunk_head", &[], NEIGHBOUR_CAP);
        assert_eq!(ranked, RankedNeighbours::default());
    }
}
