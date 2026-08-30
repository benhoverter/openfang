//! MEMORY.md managed-block rendering and splicing (ANAI-168 Layer 1,
//! re-pointed at tier 3 by ANAI-212).
//!
//! MEMORY.md has two writers that must never clobber each other:
//!
//! * the **agent**, writing free prose via `file_write` (judgment, conventions,
//!   corrections — things that have no key), and
//! * the **deterministic sweep**, writing a fenced *managed block* rendered
//!   from the agent's own tier-3 claim slots.
//!
//! The managed block is delimited by [`MANAGED_BEGIN`] / [`MANAGED_END`]. The
//! sweep only ever rewrites the region between those markers; every byte outside
//! them is preserved verbatim. The block stores no new state — it is a *view* of
//! the claim store, regenerable from scratch and safe to delete by hand.
//!
//! Failure policy is deliberately loud rather than clever: if the markers are
//! malformed (see [`SpliceError`]), the sweep refuses to write at all instead of
//! guessing where the block ought to go. A file that a human or an agent has
//! mangled is not a file we overwrite silently.
//!
//! # Why the source is tier 3, and why it is open loops only (ANAI-212)
//!
//! The block used to render `kv_store`, which was the only structured memory
//! that existed when ANAI-168 shipped. That left two sources of truth about
//! durable state — a hand-authored MEMORY.md and a claim store — and only one
//! of them had a clock. The drift was not hypothetical: this agent carried
//! `main @ 036347c, schema v14` in prose for days after both had moved.
//!
//! ADR 0002 §2.5 answers it in one line: **render open loops and task state,
//! not the whole fact set.** Settled claims are *retrieved* — `memory_fact`
//! reads them by key, recall finds them by meaning — and pasting them into
//! every prompt spends the context budget on background nobody asked for. What
//! a cold reader cannot reconstruct is the unfinished business, so that is what
//! the block carries.
//!
//! Two properties fall out of the source swap rather than being enforced:
//!
//! * **System keys can never render.** Tier 5 renders from tier 3 only, so
//!   `delivery.last_channel` and its neighbours in `kv_store` are excluded
//!   structurally — there is no denylist to keep in sync.
//! * **A superseded claim can never render.** Supersession moves the old text
//!   to `fact_history`, and this reads live slots, so the block cannot show a
//!   claim the store no longer believes.

use chrono::{DateTime, Utc};

use crate::fact::{Fact, FactStatus};
use crate::staleness::Staleness;

/// Opening marker for the sweep-managed region.
pub const MANAGED_BEGIN: &str = "<!-- openfang:managed:begin -->";
/// Closing marker for the sweep-managed region.
pub const MANAGED_END: &str = "<!-- openfang:managed:end -->";

/// Maximum characters the rendered managed block may occupy.
///
/// `BUDGET_MEMORY` in the prompt builder is 8000 chars (ANAI-167); half is
/// reserved for the block so hand-written prose always has room to survive.
pub const BLOCK_BUDGET_CHARS: usize = 4000;

/// Maximum characters rendered for a single claim before elision.
pub const VALUE_CAP_CHARS: usize = 240;

/// Why a splice was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpliceError {
    /// A begin marker with no matching end marker after it.
    UnterminatedBlock,
    /// An end marker appearing before any begin marker.
    OrphanedEnd,
    /// More than one begin marker — ambiguous which region is managed.
    DuplicateBegin,
}

impl std::fmt::Display for SpliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedBlock => {
                write!(f, "managed-block begin marker has no matching end marker")
            }
            Self::OrphanedEnd => write!(f, "managed-block end marker precedes any begin marker"),
            Self::DuplicateBegin => write!(f, "more than one managed-block begin marker"),
        }
    }
}

impl std::error::Error for SpliceError {}

/// Keep only the claims the block is allowed to render (ANAI-212).
///
/// Applied by the caller *before* the render so the sweep's "nothing to say"
/// check and the block itself agree about what "nothing" means — a sweep that
/// counted settled claims would rewrite a file to say the same thing it already
/// said, and an agent whose only claims are settled would never be skipped.
pub fn open_loops(facts: Vec<Fact>) -> Vec<Fact> {
    facts
        .into_iter()
        .filter(|f| matches!(f.status, FactStatus::Open))
        .collect()
}

/// How a slot is addressed in the block.
///
/// Agent-scope slots render bare: their `scope_ref` is the agent's own UUID,
/// which is noise in the agent's own file. Project slots carry their slug,
/// because the same `claim_key` under two scopes is two different slots and a
/// block that hid the difference would report one of them as the other.
fn display_key(fact: &Fact) -> String {
    if fact.scope == "agent" {
        fact.claim_key.clone()
    } else {
        format!("{}/{}", fact.scope_ref, fact.claim_key)
    }
}

/// The age trailer, in the same shape the rehydration pack uses.
///
/// Deliberately identical wording: a claim that says `verify` in a briefing and
/// something else in MEMORY.md would read as two different facts about the same
/// slot. One convention, stated twice.
fn trailer(fact: &Fact, now: DateTime<Utc>) -> String {
    match fact.staleness_at(now) {
        Staleness::Fresh => format!("_[{}]_", fact.persistence_class),
        Staleness::ShouldVerify { age_days } if age_days < 0 => {
            format!("_[{} · age unknown · verify]_", fact.persistence_class)
        }
        Staleness::ShouldVerify { age_days } => format!(
            "_[{} · last verified {age_days}d ago · verify]_",
            fact.persistence_class
        ),
    }
}

/// Flatten a claim to one line and cap it at [`VALUE_CAP_CHARS`].
fn render_claim(claim: &str) -> String {
    let flat = claim.split_whitespace().collect::<Vec<_>>().join(" ");
    cap(&flat, VALUE_CAP_CHARS)
}

/// Truncate on a char boundary, appending an ellipsis when anything was cut.
fn cap(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", kept.trim_end())
}

/// Render the managed block — markers included — from open claim slots.
///
/// `facts` must already be filtered by [`open_loops`] and ordered the way the
/// caller wants them displayed. Entries are emitted until
/// [`BLOCK_BUDGET_CHARS`] would be exceeded; any remainder is reported in a
/// visible footer rather than dropped silently.
///
/// `now` is injected rather than read from the clock so the staleness trailers
/// are testable and so one sweep dates every line by the same instant.
///
/// The output is deterministic: identical input renders byte-identical output,
/// so a sweep that changes nothing rewrites nothing.
pub fn render_managed_block(facts: &[Fact], now: DateTime<Utc>) -> String {
    let header = "_Auto-generated from this agent's **open** claim slots (`memory_fact` with \
                  `status: open`). Settled claims are not listed here — read them by key with \
                  `memory_fact`, or find them with `memory_recall`. Correct a claim by writing \
                  the slot, not by editing this block: edits inside it are overwritten by the \
                  next sweep. Durable prose belongs below it._";

    let mut body = String::new();
    let mut rendered = 0usize;

    for fact in facts {
        let line = format!(
            "- `{}` — {} {}\n",
            display_key(fact),
            render_claim(&fact.claim),
            trailer(fact, now),
        );
        // Reserve room for the footer we may still need to append.
        if body.chars().count() + line.chars().count() > BLOCK_BUDGET_CHARS {
            break;
        }
        body.push_str(&line);
        rendered += 1;
    }

    let omitted = facts.len().saturating_sub(rendered);
    let footer = if omitted > 0 {
        format!(
            "\n_[… {omitted} more open slot(s) omitted: managed block is at its \
             {BLOCK_BUDGET_CHARS}-char budget. Use `memory_recall` for these.]_\n"
        )
    } else {
        String::new()
    };

    if rendered == 0 && omitted == 0 {
        return format!("{MANAGED_BEGIN}\n{header}\n\n_No open claim slots._\n{MANAGED_END}");
    }

    format!("{MANAGED_BEGIN}\n{header}\n\n{body}{footer}{MANAGED_END}")
}

/// Replace the managed region of `existing` with `block`, preserving every byte
/// outside the markers.
///
/// If no managed region is present the block is appended to the end of the file
/// (with a blank-line separator). If the markers are malformed the splice is
/// refused — see [`SpliceError`].
///
/// Splicing is idempotent: `splice(splice(f, b), b) == splice(f, b)`.
pub fn splice_managed_block(existing: &str, block: &str) -> Result<String, SpliceError> {
    let begins: Vec<usize> = existing
        .match_indices(MANAGED_BEGIN)
        .map(|(i, _)| i)
        .collect();
    if begins.len() > 1 {
        return Err(SpliceError::DuplicateBegin);
    }

    let Some(&begin) = begins.first() else {
        // No begin marker. An end marker on its own means someone truncated the
        // file mid-block; refuse rather than append a second, nested region.
        if existing.contains(MANAGED_END) {
            return Err(SpliceError::OrphanedEnd);
        }
        return Ok(append_block(existing, block));
    };

    let after_begin = &existing[begin + MANAGED_BEGIN.len()..];
    let Some(rel_end) = after_begin.find(MANAGED_END) else {
        return Err(SpliceError::UnterminatedBlock);
    };
    if existing[..begin].contains(MANAGED_END) {
        return Err(SpliceError::OrphanedEnd);
    }

    let end = begin + MANAGED_BEGIN.len() + rel_end + MANAGED_END.len();
    let mut out = String::with_capacity(existing.len() + block.len());
    out.push_str(&existing[..begin]);
    out.push_str(block);
    out.push_str(&existing[end..]);
    Ok(out)
}

/// Append a fresh managed block to a file that has none.
fn append_block(existing: &str, block: &str) -> String {
    if existing.trim().is_empty() {
        return format!("{block}\n");
    }
    let sep = if existing.ends_with("\n\n") {
        ""
    } else if existing.ends_with('\n') {
        "\n"
    } else {
        "\n\n"
    };
    format!("{existing}{sep}{block}\n")
}

/// Extract the slot addresses currently listed inside `text`'s managed region.
///
/// Used by the dry-run planner to report which slots a sweep would add or drop
/// without diffing whole files. Parsing is deliberately literal — it reads the
/// exact `- \`key\` — ` shape [`render_managed_block`] emits and ignores
/// anything else, so hand-written prose (inside or outside the block) never
/// shows up as a phantom key. Returns an empty vec when there is no block, and
/// never errors: this is reporting, not a write path.
pub fn managed_block_keys(text: &str) -> Vec<String> {
    let Some(begin) = text.find(MANAGED_BEGIN) else {
        return Vec::new();
    };
    let after = &text[begin + MANAGED_BEGIN.len()..];
    let region = match after.find(MANAGED_END) {
        Some(end) => &after[..end],
        None => after,
    };

    region
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("- `")?;
            let (key, tail) = rest.split_once('`')?;
            // Only count entries in the rendered shape; a stray backticked
            // word in prose is not a slot.
            tail.trim_start().starts_with('—').then(|| key.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staleness::PersistenceClass;
    use openfang_types::memory::MemoryId;
    use std::collections::HashMap;
    use uuid::Uuid;

    const NOW: &str = "2026-08-30T00:00:00Z";

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(NOW)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn slot(scope: &str, scope_ref: &str, key: &str, claim: &str) -> Fact {
        Fact {
            id: MemoryId(Uuid::new_v4()),
            authored_by: None,
            scope: scope.to_string(),
            scope_ref: scope_ref.to_string(),
            claim_key: key.to_string(),
            claim: claim.to_string(),
            status: FactStatus::Open,
            confidence: 1.0,
            episode_id: None,
            created_at: "2026-08-29T12:00:00Z".to_string(),
            last_affirmed_at: None,
            persistence_class: PersistenceClass::Stable,
            metadata: HashMap::new(),
        }
    }

    fn agent_slot(key: &str, claim: &str) -> Fact {
        slot("agent", &Uuid::new_v4().to_string(), key, claim)
    }

    #[test]
    fn renders_open_slots_between_markers() {
        let block =
            render_managed_block(&[agent_slot("build.forge_cmd", "cargo xtask forge")], now());
        assert!(block.starts_with(MANAGED_BEGIN));
        assert!(block.ends_with(MANAGED_END));
        assert!(block.contains("`build.forge_cmd` — cargo xtask forge"));
    }

    #[test]
    fn agent_scope_does_not_render_its_uuid() {
        let fact = agent_slot("memory.fact_tool_status", "live");
        let block = render_managed_block(std::slice::from_ref(&fact), now());
        assert!(!block.contains(&fact.scope_ref));
    }

    #[test]
    fn project_scope_renders_its_slug() {
        let block = render_managed_block(
            &[slot(
                "project",
                "openfang",
                "repo.trunk_head",
                "main @ 3d6cd3c",
            )],
            now(),
        );
        assert!(block.contains("`openfang/repo.trunk_head` — main @ 3d6cd3c"));
    }

    #[test]
    fn settled_claims_are_filtered_before_rendering() {
        let mut settled = agent_slot("k", "v");
        settled.status = FactStatus::Settled;
        let open = agent_slot("open.loop", "unfinished");
        let kept = open_loops(vec![settled, open]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].claim_key, "open.loop");
    }

    #[test]
    fn a_stale_claim_carries_a_verify_marker() {
        let mut fact = slot("project", "openfang", "deploy.live_binary", "pid 70484");
        fact.persistence_class = PersistenceClass::Volatile;
        fact.created_at = "2026-08-20T00:00:00Z".to_string();
        let block = render_managed_block(&[fact], now());
        assert!(block.contains("· verify]"));
        assert!(block.contains("last verified 10d ago"));
    }

    #[test]
    fn a_fresh_claim_carries_its_class_and_no_marker() {
        let block = render_managed_block(&[agent_slot("k", "v")], now());
        assert!(block.contains("_[stable]_"));
        assert!(!block.contains("verify]"));
    }

    #[test]
    fn an_undateable_claim_is_flagged_rather_than_called_fresh() {
        let mut fact = agent_slot("k", "v");
        fact.created_at = "not-a-timestamp".to_string();
        let block = render_managed_block(&[fact], now());
        assert!(block.contains("age unknown · verify"));
    }

    #[test]
    fn empty_slot_set_still_renders_a_valid_block() {
        let block = render_managed_block(&[], now());
        assert!(block.contains("No open claim slots"));
        // Must still be spliceable, so the next sweep can find its own markers.
        let spliced = splice_managed_block("", &block).unwrap();
        assert!(spliced.contains(MANAGED_BEGIN));
    }

    #[test]
    fn multiline_claims_are_flattened_to_one_line() {
        let block = render_managed_block(&[agent_slot("note", "line one\nline two")], now());
        assert!(block.contains("`note` — line one line two"));
        assert_eq!(block.matches("`note`").count(), 1);
    }

    #[test]
    fn oversized_claims_are_capped() {
        let long = "x".repeat(VALUE_CAP_CHARS * 2);
        let block = render_managed_block(&[agent_slot("big", &long)], now());
        assert!(block.contains('…'));
        assert!(!block.contains(&"x".repeat(VALUE_CAP_CHARS + 1)));
    }

    #[test]
    fn block_respects_budget_and_reports_omissions() {
        let facts: Vec<Fact> = (0..100)
            .map(|i| agent_slot(&format!("key_{i:03}"), &"v".repeat(VALUE_CAP_CHARS)))
            .collect();
        let block = render_managed_block(&facts, now());
        assert!(block.chars().count() < BLOCK_BUDGET_CHARS + 700);
        assert!(block.contains("more open slot(s) omitted"));
    }

    #[test]
    fn render_is_deterministic() {
        let facts = vec![agent_slot("a", "1"), slot("project", "openfang", "b", "2")];
        assert_eq!(
            render_managed_block(&facts, now()),
            render_managed_block(&facts, now())
        );
    }

    #[test]
    fn append_when_no_markers_present() {
        let existing = "# Long-Term Memory\n\nHand-written prose.\n";
        let block = render_managed_block(&[agent_slot("k", "v")], now());
        let out = splice_managed_block(existing, &block).unwrap();
        assert!(out.starts_with(existing));
        assert!(out.contains(MANAGED_BEGIN));
    }

    #[test]
    fn prose_outside_markers_is_byte_preserved() {
        let before = "# Long-Term Memory\n\nBen prefers small diffs.\n\n";
        let after = "\n\n## Notes\nFORGE transform layer is Erik's.\n";
        let existing = format!("{before}{MANAGED_BEGIN}\nstale\n{MANAGED_END}{after}");
        let block = render_managed_block(&[agent_slot("k", "v")], now());
        let out = splice_managed_block(&existing, &block).unwrap();
        assert!(out.starts_with(before));
        assert!(out.ends_with(after));
        assert!(!out.contains("stale"));
    }

    #[test]
    fn splice_is_idempotent() {
        let block = render_managed_block(&[agent_slot("k", "v")], now());
        let once = splice_managed_block("# Memory\n\nprose\n", &block).unwrap();
        let twice = splice_managed_block(&once, &block).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn refuses_unterminated_block() {
        let existing = format!("prose\n{MANAGED_BEGIN}\nhalf a block\n");
        let block = render_managed_block(&[], now());
        assert_eq!(
            splice_managed_block(&existing, &block),
            Err(SpliceError::UnterminatedBlock)
        );
    }

    #[test]
    fn refuses_orphaned_end_marker() {
        let existing = format!("prose\n{MANAGED_END}\nmore\n");
        let block = render_managed_block(&[], now());
        assert_eq!(
            splice_managed_block(&existing, &block),
            Err(SpliceError::OrphanedEnd)
        );
    }

    #[test]
    fn refuses_duplicate_begin_markers() {
        let existing =
            format!("{MANAGED_BEGIN}\na\n{MANAGED_END}\n{MANAGED_BEGIN}\nb\n{MANAGED_END}\n");
        let block = render_managed_block(&[], now());
        assert_eq!(
            splice_managed_block(&existing, &block),
            Err(SpliceError::DuplicateBegin)
        );
    }

    #[test]
    fn empty_file_gets_a_clean_block() {
        let block = render_managed_block(&[agent_slot("k", "v")], now());
        let out = splice_managed_block("", &block).unwrap();
        assert_eq!(out, format!("{block}\n"));
    }

    #[test]
    fn block_keys_round_trip_from_a_rendered_block() {
        let block = render_managed_block(
            &[
                agent_slot("a_key", "v"),
                slot("project", "openfang", "b_key", "v"),
            ],
            now(),
        );
        assert_eq!(managed_block_keys(&block), vec!["a_key", "openfang/b_key"]);
    }

    #[test]
    fn block_keys_ignores_prose_outside_and_inside_the_markers() {
        let block = render_managed_block(&[agent_slot("real_key", "v")], now());
        let file = format!(
            "# Memory\n\n- `not_a_fact` is prose above the block\n\n{block}\n\n\
             - `also_prose` below the block\n"
        );
        assert_eq!(managed_block_keys(&file), vec!["real_key"]);
    }

    #[test]
    fn block_keys_is_empty_when_no_block_present() {
        assert!(managed_block_keys("# Memory\n\njust prose\n").is_empty());
    }
}
