//! MEMORY.md managed-block rendering and splicing (ANAI-168, Layer 1).
//!
//! MEMORY.md has two writers that must never clobber each other:
//!
//! * the **agent**, writing free prose via `file_write` (judgment, conventions,
//!   corrections — things that have no key), and
//! * the **deterministic sweep**, writing a fenced *managed block* rendered from
//!   the agent's own KV namespace.
//!
//! The managed block is delimited by [`MANAGED_BEGIN`] / [`MANAGED_END`]. The
//! sweep only ever rewrites the region between those markers; every byte outside
//! them is preserved verbatim. The block stores no new state — it is a *view* of
//! `kv_store`, regenerable from scratch and safe to delete by hand.
//!
//! Failure policy is deliberately loud rather than clever: if the markers are
//! malformed (see [`SpliceError`]), the sweep refuses to write at all instead of
//! guessing where the block ought to go. A file that a human or an agent has
//! mangled is not a file we overwrite silently.

use serde_json::Value;

/// Opening marker for the sweep-managed region.
pub const MANAGED_BEGIN: &str = "<!-- openfang:managed:begin -->";
/// Closing marker for the sweep-managed region.
pub const MANAGED_END: &str = "<!-- openfang:managed:end -->";

/// Maximum characters the rendered managed block may occupy.
///
/// `BUDGET_MEMORY` in the prompt builder is 8000 chars (ANAI-167); half is
/// reserved for the block so hand-written prose always has room to survive.
pub const BLOCK_BUDGET_CHARS: usize = 4000;

/// Maximum characters rendered for a single KV value before elision.
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

/// One KV row eligible for the managed block.
#[derive(Debug, Clone)]
pub struct KvFact {
    /// The key, as the agent stored it.
    pub key: String,
    /// The stored value.
    pub value: Value,
    /// RFC3339 write timestamp from `kv_store.updated_at`.
    pub updated_at: String,
}

/// Render a value for display: bare strings unquoted, everything else compact
/// JSON. Newlines are flattened so one fact stays one list item, and the result
/// is capped at [`VALUE_CAP_CHARS`].
fn render_value(value: &Value) -> String {
    let raw = match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let flat = raw.split_whitespace().collect::<Vec<_>>().join(" ");
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

/// Date portion of an RFC3339 stamp, for display. Falls back to the whole
/// string if it does not look like a timestamp.
fn short_date(updated_at: &str) -> &str {
    updated_at.split('T').next().unwrap_or(updated_at)
}

/// Render the managed block — markers included — from ranked facts.
///
/// `facts` must already be in the order the caller wants them displayed
/// (the sweep ranks by write recency; see `StructuredStore::list_kv_ranked`).
/// Entries are emitted until [`BLOCK_BUDGET_CHARS`] would be exceeded; any
/// remainder is reported in a visible footer rather than dropped silently.
///
/// The output is deterministic: identical input renders byte-identical output,
/// so a sweep that changes nothing rewrites nothing.
pub fn render_managed_block(facts: &[KvFact]) -> String {
    let header = "_Auto-generated from this agent's own memory_store keys. \
                  Edits inside this block are overwritten by the next sweep; \
                  write durable prose below it instead._";

    let mut body = String::new();
    let mut rendered = 0usize;

    for fact in facts {
        let line = format!(
            "- `{}` — {} _(updated {})_\n",
            fact.key,
            render_value(&fact.value),
            short_date(&fact.updated_at),
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
            "\n_[… {omitted} more key(s) omitted: managed block is at its \
             {BLOCK_BUDGET_CHARS}-char budget. Use `memory_recall` for these.]_\n"
        )
    } else {
        String::new()
    };

    if rendered == 0 && omitted == 0 {
        return format!("{MANAGED_BEGIN}\n{header}\n\n_No stored facts yet._\n{MANAGED_END}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fact(key: &str, value: Value) -> KvFact {
        KvFact {
            key: key.to_string(),
            value,
            updated_at: "2026-08-18T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn renders_facts_between_markers() {
        let block = render_managed_block(&[fact("forge_build_cmd", json!("cargo xtask forge"))]);
        assert!(block.starts_with(MANAGED_BEGIN));
        assert!(block.ends_with(MANAGED_END));
        assert!(block.contains("`forge_build_cmd` — cargo xtask forge _(updated 2026-08-18)_"));
    }

    #[test]
    fn empty_namespace_still_renders_a_valid_block() {
        let block = render_managed_block(&[]);
        assert!(block.contains("No stored facts yet"));
        // Must still be spliceable, so the next sweep can find its own markers.
        let spliced = splice_managed_block("", &block).unwrap();
        assert!(spliced.contains(MANAGED_BEGIN));
    }

    #[test]
    fn non_string_values_render_as_compact_json() {
        let block = render_managed_block(&[fact("limits", json!({"max": 3}))]);
        assert!(block.contains(r#"`limits` — {"max":3}"#));
    }

    #[test]
    fn multiline_values_are_flattened_to_one_line() {
        let block = render_managed_block(&[fact("note", json!("line one\nline two"))]);
        assert!(block.contains("`note` — line one line two"));
        assert_eq!(block.matches("`note`").count(), 1);
    }

    #[test]
    fn oversized_values_are_capped() {
        let long = "x".repeat(VALUE_CAP_CHARS * 2);
        let block = render_managed_block(&[fact("big", json!(long))]);
        assert!(block.contains('…'));
        assert!(!block.contains(&"x".repeat(VALUE_CAP_CHARS + 1)));
    }

    #[test]
    fn block_respects_budget_and_reports_omissions() {
        // Each entry is ~260 chars, so well over the budget in aggregate.
        let facts: Vec<KvFact> = (0..100)
            .map(|i| fact(&format!("key_{i:03}"), json!("v".repeat(VALUE_CAP_CHARS))))
            .collect();
        let block = render_managed_block(&facts);
        assert!(block.chars().count() < BLOCK_BUDGET_CHARS + 500);
        assert!(block.contains("more key(s) omitted"));
    }

    #[test]
    fn append_when_no_markers_present() {
        let existing = "# Long-Term Memory\n\nHand-written prose.\n";
        let block = render_managed_block(&[fact("k", json!("v"))]);
        let out = splice_managed_block(existing, &block).unwrap();
        assert!(out.starts_with(existing));
        assert!(out.contains(MANAGED_BEGIN));
    }

    #[test]
    fn prose_outside_markers_is_byte_preserved() {
        let before = "# Long-Term Memory\n\nBen prefers small diffs.\n\n";
        let after = "\n\n## Notes\nFORGE transform layer is Erik's.\n";
        let existing = format!("{before}{MANAGED_BEGIN}\nstale\n{MANAGED_END}{after}");
        let block = render_managed_block(&[fact("k", json!("v"))]);
        let out = splice_managed_block(&existing, &block).unwrap();
        assert!(out.starts_with(before));
        assert!(out.ends_with(after));
        assert!(!out.contains("stale"));
    }

    #[test]
    fn splice_is_idempotent() {
        let block = render_managed_block(&[fact("k", json!("v"))]);
        let once = splice_managed_block("# Memory\n\nprose\n", &block).unwrap();
        let twice = splice_managed_block(&once, &block).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn refuses_unterminated_block() {
        let existing = format!("prose\n{MANAGED_BEGIN}\nhalf a block\n");
        let block = render_managed_block(&[]);
        assert_eq!(
            splice_managed_block(&existing, &block),
            Err(SpliceError::UnterminatedBlock)
        );
    }

    #[test]
    fn refuses_orphaned_end_marker() {
        let existing = format!("prose\n{MANAGED_END}\nmore\n");
        let block = render_managed_block(&[]);
        assert_eq!(
            splice_managed_block(&existing, &block),
            Err(SpliceError::OrphanedEnd)
        );
    }

    #[test]
    fn refuses_duplicate_begin_markers() {
        let existing =
            format!("{MANAGED_BEGIN}\na\n{MANAGED_END}\n{MANAGED_BEGIN}\nb\n{MANAGED_END}\n");
        let block = render_managed_block(&[]);
        assert_eq!(
            splice_managed_block(&existing, &block),
            Err(SpliceError::DuplicateBegin)
        );
    }

    #[test]
    fn empty_file_gets_a_clean_block() {
        let block = render_managed_block(&[fact("k", json!("v"))]);
        let out = splice_managed_block("", &block).unwrap();
        assert_eq!(out, format!("{block}\n"));
    }
}
