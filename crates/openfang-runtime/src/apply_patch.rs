//! Multi-hunk diff-based file patching.
//!
//! Implements a structured patch format similar to unified diffs, allowing
//! targeted edits without full file overwrites. Supports adding, updating
//! (including move/rename), and deleting files with multi-hunk precision.
//!
//! Patch format:
//! ```text
//! *** Begin Patch
//! *** Add File: path/to/new.rs
//! +line1
//! +line2
//! *** Update File: path/to/existing.rs
//! @@ fn enclosing_function
//!  unchanged_line
//! -old_line
//! +new_line
//!  unchanged_line
//! *** Delete File: path/to/old.rs
//! *** End Patch
//! ```
//!
//! How a hunk is placed (ANAI-298):
//! - The anchor is the WHOLE hunk: leading context, `-` lines and trailing
//!   context, matched as one contiguous block.
//! - Hunks are searched forward, each starting where the previous one ended,
//!   so they must be listed in file order.
//! - Text after `@@` names a line at or above the change (e.g. the enclosing
//!   `fn` signature); the search starts at that line. A bare `@@` or a
//!   unified-diff range (`@@ -12,5 +12,6 @@`) carries no scope.
//! - Exact matches are tried first, then matches ignoring trailing
//!   whitespace. More than one match is refused, never guessed.
//! - Every hunk in every file is resolved before anything is written; one
//!   failure rejects the whole patch.
//! - Untouched lines keep their own line ending; inserted lines take the
//!   file's dominant one.
//!
//! When a hunk matches more than one place (ANAI-298 B), nothing is written
//! and the result is a `needs_input` question, not an error: every ambiguous
//! hunk is listed with its candidate locations, plus a `state_token` that
//! fingerprints the patch and every target file. A retry carrying that token
//! and one `{file, hunk, at_line}` choice per listed hunk is applied at the
//! chosen places. A token that no longer matches (a file or the patch
//! changed) voids its choices, and the question is asked again.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Most match locations quoted in one error message, and most candidates
/// offered for one ambiguous hunk.
const MAX_LISTED: usize = 10;
/// Most ambiguous hunks described in one needs_input response.
const MAX_DECISIONS: usize = 10;
/// Lines shown on each side of a candidate, outside the hunk's own anchor.
const SHOWN_CONTEXT: usize = 2;
/// Longest line quoted in a candidate, in chars.
const MAX_LINE_CHARS: usize = 160;
/// How far above a candidate to look for its enclosing line.
const WITHIN_SCAN: usize = 2000;
/// Largest patch echoed back verbatim in the retry arguments. Keeps the whole
/// response far below the bridge's 1 MiB frame clamp, which would otherwise
/// truncate the JSON and flip the result to an error.
const MAX_ECHO_PATCH: usize = 64 * 1024;

/// A single operation in a patch.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchOp {
    /// Add a new file with the given content.
    AddFile { path: String, content: String },
    /// Update an existing file, optionally moving/renaming it.
    UpdateFile {
        path: String,
        move_to: Option<String>,
        hunks: Vec<Hunk>,
    },
    /// Delete an existing file.
    DeleteFile { path: String },
}

/// A single hunk within a file update — describes one contiguous change region.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hunk {
    /// Lines of unchanged context before the change (for anchoring).
    pub context_before: Vec<String>,
    /// Old lines to be removed (without `-` prefix).
    pub old_lines: Vec<String>,
    /// New lines to be inserted (without `+` prefix).
    pub new_lines: Vec<String>,
    /// Lines of unchanged context after the change (for anchoring).
    pub context_after: Vec<String>,
    /// Text from the `@@` header naming a line at or above the change; the
    /// search for this hunk starts at the first line containing it. `None`
    /// for a bare `@@`, a unified-diff range, or the second and later regions
    /// of a split hunk.
    pub scope: Option<String>,
}

/// Result of applying a patch.
#[derive(Debug, Default)]
pub struct PatchResult {
    /// Number of files added.
    pub files_added: u32,
    /// Number of files updated.
    pub files_updated: u32,
    /// Number of files deleted.
    pub files_deleted: u32,
    /// Number of files moved/renamed.
    pub files_moved: u32,
    /// Errors encountered during application.
    pub errors: Vec<String>,
    /// ANAI-298 B: set when hunks matched several places and need the
    /// caller's choice. Nothing was written.
    pub needs_input: Option<NeedsInput>,
    /// The call carried a `state_token` that no longer matched, so its
    /// choices were ignored.
    pub stale_selection: bool,
}

impl PatchResult {
    /// Returns true if the patch was applied in full.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty() && self.needs_input.is_none()
    }

    /// Summary string for tool output.
    pub fn summary(&self) -> String {
        if let Some(ni) = &self.needs_input {
            return format!(
                "No changes applied — {} needed, nothing was written",
                plural(ni.decisions.len(), "decision")
            );
        }
        let mut parts = Vec::new();
        if self.files_added > 0 {
            parts.push(format!("{} added", self.files_added));
        }
        if self.files_updated > 0 {
            parts.push(format!("{} updated", self.files_updated));
        }
        if self.files_deleted > 0 {
            parts.push(format!("{} deleted", self.files_deleted));
        }
        if self.files_moved > 0 {
            parts.push(format!("{} moved", self.files_moved));
        }
        if parts.is_empty() && !self.errors.is_empty() {
            // ANAI-298: planning refuses the whole patch before any write, so
            // say so plainly — the caller's wrapper text may not.
            return format!(
                "No changes applied — patch rejected, nothing was written ({} {})",
                self.errors.len(),
                if self.errors.len() == 1 {
                    "error"
                } else {
                    "errors"
                }
            );
        }
        if !self.errors.is_empty() {
            parts.push(format!("{} errors", self.errors.len()));
        }
        if parts.is_empty() {
            "No changes applied".to_string()
        } else {
            parts.join(", ")
        }
    }
}

fn plural(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// ANAI-298 B: the caller's pick for one hunk that matched several places.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    /// File path exactly as written in the patch's `*** Update File:` line.
    pub file: String,
    /// 1-based hunk number within that file.
    pub hunk: usize,
    /// 1-based line, in the file as it is on disk, where the chosen match starts.
    pub at_line: usize,
}

/// What a retry carries: the `state_token` it was offered and its choices.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    pub state_token: Option<String>,
    pub choices: Vec<Choice>,
}

impl Selection {
    /// Read `state_token` and `choices` from tool input. Absent fields give an
    /// empty selection; malformed ones are an error saying what to send.
    pub fn from_input(input: &serde_json::Value) -> Result<Self, String> {
        use serde_json::Value;
        let state_token = match input.get("state_token") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_str()
                    .ok_or(
                        "'state_token' must be the string from the NOT APPLIED result. \
                         Nothing was written.",
                    )?
                    .to_string(),
            ),
        };
        let mut choices = Vec::new();
        match input.get("choices") {
            None | Some(Value::Null) => {}
            Some(Value::Array(items)) => {
                for (i, item) in items.iter().enumerate() {
                    let file = item.get("file").and_then(Value::as_str);
                    let hunk = item.get("hunk").and_then(Value::as_u64);
                    let at_line = item.get("at_line").and_then(Value::as_u64);
                    match (file, hunk, at_line) {
                        (Some(f), Some(h), Some(l)) if h > 0 && l > 0 => choices.push(Choice {
                            file: f.to_string(),
                            hunk: h as usize,
                            at_line: l as usize,
                        }),
                        _ => {
                            return Err(format!(
                                "choices[{i}] must be {{\"file\": <path>, \"hunk\": <n>, \
                                 \"at_line\": <n>}} with at_line set to one of the numbers \
                                 offered (copy a candidate's `choice`); got {item}. \
                                 Nothing was written."
                            ))
                        }
                    }
                }
            }
            Some(other) => {
                return Err(format!(
                    "'choices' must be an array, got {other}. Nothing was written."
                ))
            }
        }
        if !choices.is_empty() && state_token.is_none() {
            return Err(
                "'choices' must be sent with the 'state_token' from the NOT APPLIED \
                        result they came from. Nothing was written."
                    .to_string(),
            );
        }
        Ok(Self {
            state_token,
            choices,
        })
    }

    fn choice_for(&self, file: &str, hunk: usize) -> Option<&Choice> {
        self.choices
            .iter()
            .find(|c| c.file == file && c.hunk == hunk)
    }
}

/// One quoted line: its 1-based number in the file on disk (`None` for a
/// line an earlier hunk of this patch adds) and its text.
pub type QuotedLine = (Option<usize>, String);

/// One place an ambiguous hunk could go.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// 1-based line where the hunk's anchor starts. This is the choice.
    pub at_line: usize,
    /// Nearest line above with less indentation than the anchor — usually
    /// the enclosing fn or block header.
    pub within: Option<QuotedLine>,
    /// Lines just before and just after the anchor. The anchor itself is
    /// identical at every candidate, so these are what tell them apart.
    pub before: Vec<QuotedLine>,
    pub after: Vec<QuotedLine>,
}

/// One hunk that needs the caller to choose where it goes.
#[derive(Debug, Clone)]
pub struct Decision {
    pub file: String,
    pub hunk: usize,
    /// Short identifier: the hunk's first changed line.
    pub label: String,
    /// Every match found, including any beyond the candidates listed.
    pub total: usize,
    pub candidates: Vec<Candidate>,
    pub note: Option<String>,
    /// The same situation as a one-paragraph error message.
    pub message: String,
}

/// ANAI-298 B: the patch was not applied because hunks need a choice.
#[derive(Debug, Clone)]
pub struct NeedsInput {
    pub state_token: String,
    pub decisions: Vec<Decision>,
    /// The call carried a state_token that no longer matched.
    pub stale_token: bool,
}

impl NeedsInput {
    /// The tool result text: one line of prose, then JSON. The prose line is
    /// deliberate — the result is sent with `is_error=false` so it is not
    /// read as "the tool is broken", and the first line keeps it from being
    /// read as success.
    pub fn render(&self, patch: &str) -> String {
        use serde_json::json;
        let n = self.decisions.len();
        let mut head = format!(
            "NOT APPLIED — {} needed. Nothing was changed. Some hunks match more than one \
             place, so apply_patch will not guess. Retry apply_patch with the same patch, \
             this state_token, and one choice per hunk listed below (copy the `choice` of \
             the candidate you mean).",
            plural(n, "decision")
        );
        if self.stale_token {
            head.push_str(
                " Your previous state_token no longer matched (a file or the patch changed), \
                 so its choices were ignored; these choices are fresh.",
            );
        }
        let quoted = |q: &QuotedLine| json!({ "line": q.0, "text": q.1 });
        let shown = &self.decisions[..n.min(MAX_DECISIONS)];
        let decisions: Vec<serde_json::Value> = shown
            .iter()
            .map(|d| {
                let candidates: Vec<serde_json::Value> = d
                    .candidates
                    .iter()
                    .map(|c| {
                        json!({
                            "at_line": c.at_line,
                            "within": c.within.as_ref().map(quoted),
                            "before": c.before.iter().map(quoted).collect::<Vec<_>>(),
                            "after": c.after.iter().map(quoted).collect::<Vec<_>>(),
                            "choice": { "file": d.file, "hunk": d.hunk, "at_line": c.at_line },
                        })
                    })
                    .collect();
                json!({
                    "file": d.file,
                    "hunk": d.hunk,
                    "first_change": d.label,
                    "matches": d.total,
                    "n_more": d.total.saturating_sub(d.candidates.len()),
                    "note": d.note,
                    "candidates": candidates,
                })
            })
            .collect();
        let retry_choices: Vec<serde_json::Value> = shown
            .iter()
            .map(|d| {
                let options: Vec<String> =
                    d.candidates.iter().map(|c| c.at_line.to_string()).collect();
                json!({
                    "file": d.file,
                    "hunk": d.hunk,
                    "at_line": format!("PICK ONE OF {}", options.join(", ")),
                })
            })
            .collect();
        let patch_arg = if patch.len() <= MAX_ECHO_PATCH {
            patch.to_string()
        } else {
            "(patch too large to echo here — resend your previous patch unchanged)".to_string()
        };
        let body = json!({
            "status": "needs_input",
            "applied": false,
            "state_token": self.state_token,
            "decisions": decisions,
            "n_more_decisions": n.saturating_sub(MAX_DECISIONS),
            "retry": {
                "tool": "apply_patch",
                "arguments": {
                    "patch": patch_arg,
                    "state_token": self.state_token,
                    "choices": retry_choices,
                },
            },
        });
        format!(
            "{head}\n{}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        )
    }
}

/// True for a line that starts a new file operation. Checked on the raw
/// line: hunk content always carries a ` `/`-`/`+` prefix, so a content line
/// such as ` *** banner ***` or `+@@ note` can never end a hunk early.
fn is_op_marker(raw: &str) -> bool {
    raw.starts_with("*** Add File:")
        || raw.starts_with("*** Update File:")
        || raw.starts_with("*** Delete File:")
}

/// `-12,5 +12,6` — the line-range part of a unified-diff hunk header.
fn looks_like_unified_range(s: &str) -> bool {
    let is_range = |t: &str, sign: char| {
        t.strip_prefix(sign)
            .is_some_and(|r| !r.is_empty() && r.chars().all(|c| c.is_ascii_digit() || c == ','))
    };
    let mut it = s.split_whitespace();
    matches!((it.next(), it.next()), (Some(a), Some(b)) if is_range(a, '-') && is_range(b, '+'))
}

/// Scope text from a `@@` header line. `@@`, `@@ @@` and a bare unified
/// range give `None`; `@@ fn foo`, `@@ fn foo @@` and
/// `@@ -3,4 +3,5 @@ fn foo` all give `fn foo`.
fn parse_hunk_header(raw: &str) -> Option<String> {
    let rest = raw.strip_prefix("@@")?.trim();
    let rest = if looks_like_unified_range(rest) {
        rest.find("@@").map_or("", |p| rest[p + 2..].trim())
    } else {
        rest.strip_suffix("@@").unwrap_or(rest).trim()
    };
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Parse one hunk body starting at `body[start]` (the line after its `@@`
/// header, if it had one). Returns the hunks it holds — more than one when
/// the body has several change regions — and the index of the first line
/// after it.
fn parse_hunk_body(
    body: &[&str],
    start: usize,
    scope: Option<String>,
    path: &str,
    line_base: usize,
) -> Result<(Vec<Hunk>, usize), String> {
    let mut out = Vec::new();
    let mut cur = Hunk {
        scope,
        ..Default::default()
    };
    let mut in_change = false;
    let mut past_change = false;
    // Unprefixed blank lines at the tail of the current context run. Those
    // at the very end of the hunk are formatting, not context: dropped below
    // so a stray blank line before `*** End Patch` cannot fail the match.
    let mut trailing_bare = 0usize;
    let mut i = start;

    while i < body.len() {
        let hl = body[i];
        if hl.starts_with("@@") || is_op_marker(hl) {
            break;
        }
        if hl.trim() == "*** End of File" {
            i += 1;
            break;
        }
        let (kind, text, bare) = if let Some(s) = hl.strip_prefix('-') {
            ('-', s, false)
        } else if let Some(s) = hl.strip_prefix('+') {
            ('+', s, false)
        } else if let Some(s) = hl.strip_prefix(' ') {
            (' ', s, false)
        } else if hl.trim().is_empty() {
            (' ', "", true)
        } else {
            // ANAI-298: this used to be folded in as context, so a malformed
            // patch was partly applied instead of refused.
            return Err(format!(
                "{path}: patch line {}: a hunk line must start with ' ' (context), \
                 '-' (remove) or '+' (add), got: {hl}",
                line_base + i
            ));
        };

        if kind == ' ' {
            if in_change || past_change {
                past_change = true;
                in_change = false;
                cur.context_after.push(text.to_string());
            } else {
                cur.context_before.push(text.to_string());
            }
            trailing_bare = if bare { trailing_bare + 1 } else { 0 };
        } else {
            // ANAI-254: a change line that resumes *after* trailing context
            // begins a new hunk; the shared context anchors both.
            if past_change {
                let carry = std::mem::take(&mut cur.context_after);
                let done = std::mem::replace(
                    &mut cur,
                    Hunk {
                        context_before: carry.clone(),
                        ..Default::default()
                    },
                );
                out.push(Hunk {
                    context_after: carry,
                    ..done
                });
                past_change = false;
            }
            in_change = true;
            trailing_bare = 0;
            if kind == '-' {
                cur.old_lines.push(text.to_string());
            } else {
                cur.new_lines.push(text.to_string());
            }
        }
        i += 1;
    }

    let tail = if in_change || past_change {
        &mut cur.context_after
    } else {
        &mut cur.context_before
    };
    for _ in 0..trailing_bare {
        tail.pop();
    }
    out.push(cur);
    Ok((out, i))
}

/// Parse a patch string into a list of `PatchOp`s.
///
/// Expects the format delimited by `*** Begin Patch` and `*** End Patch`.
/// Within that block, each file operation starts with `*** Add File:`,
/// `*** Update File:`, or `*** Delete File:`.
pub fn parse_patch(input: &str) -> Result<Vec<PatchOp>, String> {
    let lines: Vec<&str> = input.lines().collect();
    let mut ops = Vec::new();

    // Find begin/end markers
    let begin = lines
        .iter()
        .position(|l| l.trim() == "*** Begin Patch")
        .ok_or("Missing '*** Begin Patch' marker")?;
    let end = lines
        .iter()
        .rposition(|l| l.trim() == "*** End Patch")
        .ok_or("Missing '*** End Patch' marker")?;

    if end <= begin {
        return Err("'*** End Patch' must come after '*** Begin Patch'".to_string());
    }

    let body = &lines[begin + 1..end];
    // 1-based line number, within `input`, of `body[0]`.
    let line_base = begin + 2;
    let mut i = 0;

    while i < body.len() {
        let line = body[i].trim();

        if line.starts_with("*** Add File:") {
            let path = line
                .strip_prefix("*** Add File:")
                .unwrap()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("Empty path in '*** Add File:'".to_string());
            }
            i += 1;

            // Collect content lines (prefixed with +). An unprefixed blank
            // line is an empty line of content, except at the very end.
            let mut content_lines = Vec::new();
            let mut trailing_bare = 0usize;
            while i < body.len() && !is_op_marker(body[i]) {
                let l = body[i];
                if let Some(stripped) = l.strip_prefix('+') {
                    content_lines.push(stripped.to_string());
                    trailing_bare = 0;
                } else if l.trim().is_empty() {
                    content_lines.push(String::new());
                    trailing_bare += 1;
                } else {
                    return Err(format!(
                        "{path}: patch line {}: expected '+' prefix in Add File content, got: {l}",
                        line_base + i
                    ));
                }
                i += 1;
            }
            content_lines.truncate(content_lines.len() - trailing_bare);
            ops.push(PatchOp::AddFile {
                path,
                content: content_lines.join("\n"),
            });
        } else if line.starts_with("*** Update File:") {
            let rest = line.strip_prefix("*** Update File:").unwrap().trim();
            // Check for move syntax: "old_path -> new_path"
            let (path, move_to) = if let Some((old, new)) = rest.split_once("->") {
                (old.trim().to_string(), Some(new.trim().to_string()))
            } else {
                (rest.to_string(), None)
            };
            if path.is_empty() {
                return Err("Empty path in '*** Update File:'".to_string());
            }
            i += 1;

            // Parse hunks. A hunk starts at an `@@` header, or directly at a
            // prefixed line when the first hunk has no header.
            let mut hunks = Vec::new();
            while i < body.len() && !is_op_marker(body[i]) {
                let raw = body[i];
                if raw.trim().is_empty() || raw.trim() == "*** End of File" {
                    i += 1;
                    continue;
                }
                let scope = if raw.starts_with("@@") {
                    i += 1;
                    parse_hunk_header(raw)
                } else if raw.starts_with([' ', '-', '+']) {
                    None
                } else {
                    return Err(format!(
                        "{path}: patch line {}: expected an '@@' hunk header or a hunk \
                         line starting with ' ', '-' or '+', got: {raw}",
                        line_base + i
                    ));
                };
                let (mut parsed, next) = parse_hunk_body(body, i, scope, &path, line_base)?;
                hunks.append(&mut parsed);
                i = next;
            }

            if hunks.is_empty() {
                return Err(format!("Update File '{}' has no hunks", path));
            }

            ops.push(PatchOp::UpdateFile {
                path,
                move_to,
                hunks,
            });
        } else if line.starts_with("*** Delete File:") {
            let path = line
                .strip_prefix("*** Delete File:")
                .unwrap()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("Empty path in '*** Delete File:'".to_string());
            }
            i += 1;
            ops.push(PatchOp::DeleteFile { path });
        } else if line.is_empty() {
            i += 1;
        } else {
            return Err(format!(
                "patch line {}: unexpected line in patch: {}",
                line_base + i,
                line
            ));
        }
    }

    if ops.is_empty() {
        return Err("Patch contains no operations".to_string());
    }

    Ok(ops)
}

/// Resolve a patch path through workspace confinement and `file_policy`.
/// Patch targets are always writes; prompt-tier is fail-closed here because
/// apply_patch is multi-path and per-target interactive approval is out of
/// scope for v1.
fn resolve_patch_path(
    raw: &str,
    workspace_root: &Path,
    file_policy: Option<&openfang_types::config::FilePolicy>,
) -> Result<PathBuf, String> {
    crate::workspace_sandbox::resolve_with_policy(raw, workspace_root, file_policy, true, false)
}

/// Raw target path(s) an op will touch — for the pre-application policy pass.
fn op_targets(op: &PatchOp) -> Vec<&str> {
    match op {
        PatchOp::AddFile { path, .. } | PatchOp::DeleteFile { path } => vec![path.as_str()],
        PatchOp::UpdateFile { path, move_to, .. } => {
            let mut v = vec![path.as_str()];
            if let Some(m) = move_to {
                v.push(m.as_str());
            }
            v
        }
    }
}

/// One filesystem change the plan has committed to.
enum Action {
    Add {
        raw: String,
        path: PathBuf,
        content: String,
    },
    Update {
        raw: String,
        source: PathBuf,
        target: PathBuf,
        original: String,
        patched: String,
        moved: bool,
    },
    Delete {
        raw: String,
        path: PathBuf,
    },
    /// ANAI-298 B: a hunk needs the caller's choice; the patch will not be
    /// written, so there is nothing to do here.
    Pending,
}

/// In-memory view of files as earlier ops in this patch leave them.
/// `None` marks a file deleted (or moved away) earlier in the patch.
type Overlay = HashMap<PathBuf, Option<String>>;

async fn read_planned(path: &Path, raw: &str, overlay: &Overlay) -> Result<String, String> {
    match overlay.get(path) {
        Some(Some(c)) => Ok(c.clone()),
        Some(None) => Err(format!("{raw}: deleted or moved earlier in this patch")),
        None => tokio::fs::read_to_string(path)
            .await
            .map_err(|e| format!("read {raw}: {e}")),
    }
}

/// Resolve one op completely — paths, file contents, every hunk — without
/// writing anything.
async fn plan_op(
    op: &PatchOp,
    workspace_root: &Path,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    overlay: &mut Overlay,
    sel: &Selection,
    pending: &mut Vec<Decision>,
) -> Result<Action, String> {
    match op {
        PatchOp::AddFile { path, content } => {
            let resolved = resolve_patch_path(path, workspace_root, file_policy)
                .map_err(|e| format!("{path}: {e}"))?;
            overlay.insert(resolved.clone(), Some(content.clone()));
            Ok(Action::Add {
                raw: path.clone(),
                path: resolved,
                content: content.clone(),
            })
        }
        PatchOp::UpdateFile {
            path,
            move_to,
            hunks,
        } => {
            let source = resolve_patch_path(path, workspace_root, file_policy)
                .map_err(|e| format!("{path}: {e}"))?;
            let original = read_planned(&source, path, overlay).await?;
            let Some(patched) = apply_hunks_with(&original, hunks, path, sel, pending)
                .map_err(|e| format!("patch {path}: {e}"))?
            else {
                // ANAI-298 B: at least one hunk needs the caller to choose.
                return Ok(Action::Pending);
            };
            // ANAI-254: the counter must derive from a confirmed *change*, not
            // merely from a successful write.
            if move_to.is_none() && patched == original {
                return Err(format!(
                    "{}: hunks applied cleanly but produced no change — \
                     the file already matches the patched content. \
                     Nothing was written.",
                    path
                ));
            }
            let target = match move_to {
                Some(new_path) => resolve_patch_path(new_path, workspace_root, file_policy)
                    .map_err(|e| format!("{new_path}: {e}"))?,
                None => source.clone(),
            };
            if target != source {
                overlay.insert(source.clone(), None);
            }
            overlay.insert(target.clone(), Some(patched.clone()));
            Ok(Action::Update {
                raw: path.clone(),
                source,
                target,
                original,
                patched,
                moved: move_to.is_some(),
            })
        }
        PatchOp::DeleteFile { path } => {
            let resolved = resolve_patch_path(path, workspace_root, file_policy)
                .map_err(|e| format!("{path}: {e}"))?;
            let exists = match overlay.get(&resolved) {
                Some(state) => state.is_some(),
                None => tokio::fs::metadata(&resolved).await.is_ok(),
            };
            if !exists {
                return Err(format!("delete {path}: file not found"));
            }
            overlay.insert(resolved.clone(), None);
            Ok(Action::Delete {
                raw: path.clone(),
                path: resolved,
            })
        }
    }
}

/// Carry out one planned change and record it in the context audit.
async fn commit(action: Action, agent_id: Option<&str>, result: &mut PatchResult) {
    match action {
        Action::Add { raw, path, content } => {
            if let Some(parent) = path.parent() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    result.errors.push(format!("mkdir {}: {}", raw, e));
                    return;
                }
            }
            // ANAI-149 D2: an "add" can land on an existing context file, so
            // snapshot before overwriting.
            let before = if crate::context_audit::is_audited(&path) {
                crate::context_audit::capture_before(&path).await
            } else {
                None
            };
            match tokio::fs::write(&path, &content).await {
                Ok(()) => {
                    result.files_added += 1;
                    crate::context_audit::record_write(
                        agent_id,
                        "apply_patch",
                        &path,
                        before.as_deref(),
                        Some(content.as_str()),
                    )
                    .await;
                }
                Err(e) => result.errors.push(format!("write {}: {}", raw, e)),
            }
        }
        Action::Update {
            raw,
            source,
            target,
            original,
            patched,
            moved,
        } => {
            if let Some(parent) = target.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            match tokio::fs::write(&target, &patched).await {
                Ok(()) => {
                    result.files_updated += 1;
                    if moved {
                        result.files_moved += 1;
                    }
                    // ANAI-149 D2. On a move the destination has no prior
                    // content of its own, so the diff is against nothing
                    // rather than against the source file.
                    crate::context_audit::record_write(
                        agent_id,
                        "apply_patch",
                        &target,
                        if moved { None } else { Some(original.as_str()) },
                        Some(patched.as_str()),
                    )
                    .await;
                    if moved && target != source {
                        let _ = tokio::fs::remove_file(&source).await;
                        crate::context_audit::record_write(
                            agent_id,
                            "apply_patch",
                            &source,
                            Some(original.as_str()),
                            None,
                        )
                        .await;
                    }
                }
                Err(e) => result.errors.push(format!("write {}: {}", raw, e)),
            }
        }
        Action::Pending => {}
        Action::Delete { raw, path } => {
            // ANAI-149 D2: capture the content before it is gone, so a
            // deleted identity file is still recoverable from the audit record.
            let before = if crate::context_audit::is_audited(&path) {
                crate::context_audit::capture_before(&path).await
            } else {
                None
            };
            match tokio::fs::remove_file(&path).await {
                Ok(()) => {
                    result.files_deleted += 1;
                    crate::context_audit::record_write(
                        agent_id,
                        "apply_patch",
                        &path,
                        before.as_deref(),
                        None,
                    )
                    .await;
                }
                Err(e) => result.errors.push(format!("delete {}: {}", raw, e)),
            }
        }
    }
}

/// Apply parsed patch operations against the filesystem.
///
/// All file paths are confined to `workspace_root` via sandbox resolution and
/// governed by `file_policy` when active.
///
/// F3: a policy pre-pass validates *every* target before anything else.
/// ANAI-298: every op is then planned in memory — each file read, every hunk
/// located and applied — and the patch is written only if the whole plan
/// succeeded. A hunk that fails in the third file therefore leaves the first
/// two untouched. This is planning, not filesystem-transactional rollback: an
/// I/O failure part-way through the write phase can still leave earlier
/// writes in place.
///
/// ANAI-298 B: hunks that match several places are collected, across every
/// file, into `needs_input` instead of failing, and nothing is written.
/// `sel` carries a retry's `state_token` and choices; they are honoured only
/// if the token still matches the patch and the files.
pub async fn apply_patch(
    ops: &[PatchOp],
    workspace_root: &Path,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    agent_id: Option<&str>,
    sel: &Selection,
) -> PatchResult {
    let mut result = PatchResult::default();

    // F3: policy pre-pass — reject the whole patch if any target is denied.
    let mut precheck_errors = Vec::new();
    for op in ops {
        for raw in op_targets(op) {
            if let Err(e) = resolve_patch_path(raw, workspace_root, file_policy) {
                precheck_errors.push(format!("{}: {}", raw, e));
            }
        }
    }
    if !precheck_errors.is_empty() {
        result.errors = precheck_errors;
        pilot_log(agent_id, "rejected", "-", sel, false, 0);
        return result;
    }

    // ANAI-298 B: fingerprint the patch and every target before planning, so
    // choices are honoured only against the exact state they were offered for.
    let token = state_token(ops, workspace_root, file_policy).await;
    let token_matched = sel.state_token.as_deref() == Some(token.as_str());
    let no_choices = Selection::default();
    let active = if token_matched { sel } else { &no_choices };
    result.stale_selection = sel.state_token.is_some() && !token_matched;

    // ANAI-298: plan everything; report every failing op at once.
    let mut overlay = Overlay::new();
    let mut actions = Vec::new();
    let mut pending = Vec::new();
    for op in ops {
        match plan_op(
            op,
            workspace_root,
            file_policy,
            &mut overlay,
            active,
            &mut pending,
        )
        .await
        {
            Ok(action) => actions.push(action),
            Err(e) => result.errors.push(e),
        }
    }
    if !result.errors.is_empty() {
        pilot_log(
            agent_id,
            "rejected",
            &token,
            sel,
            token_matched,
            pending.len(),
        );
        return result;
    }
    if !pending.is_empty() {
        pilot_log(
            agent_id,
            "needs_input",
            &token,
            sel,
            token_matched,
            pending.len(),
        );
        result.needs_input = Some(NeedsInput {
            state_token: token,
            decisions: pending,
            stale_token: result.stale_selection,
        });
        return result;
    }

    for action in actions {
        commit(action, agent_id, &mut result).await;
    }
    let outcome = if result.is_ok() {
        "applied"
    } else {
        "rejected"
    };
    pilot_log(agent_id, outcome, &token, sel, token_matched, 0);
    result
}

/// Fingerprint of the patch and the current bytes of every file it targets.
async fn state_token(
    ops: &[PatchOp],
    workspace_root: &Path,
    file_policy: Option<&openfang_types::config::FilePolicy>,
) -> String {
    let mut h = Sha256::new();
    for op in ops {
        let desc = format!("{op:?}");
        h.update((desc.len() as u64).to_le_bytes());
        h.update(desc.as_bytes());
        for raw in op_targets(op) {
            match resolve_patch_path(raw, workspace_root, file_policy) {
                Ok(p) => match tokio::fs::read(&p).await {
                    Ok(bytes) => {
                        h.update(b"F");
                        h.update((bytes.len() as u64).to_le_bytes());
                        h.update(&bytes);
                    }
                    Err(_) => h.update(b"-"),
                },
                Err(_) => h.update(b"!"),
            }
        }
    }
    hex::encode(&h.finalize()[..12])
}

/// ANAI-298 B pilot. One line per call at info (the daemon's default level)
/// under `apply_patch.pilot`: "do agents retry after needs_input?" is the
/// count of `token_matched=true` lines against `outcome=needs_input` lines.
fn pilot_log(
    agent: Option<&str>,
    outcome: &str,
    token: &str,
    sel: &Selection,
    token_matched: bool,
    decisions: usize,
) {
    info!(
        target: "apply_patch.pilot",
        agent = agent.unwrap_or("-"),
        outcome,
        state_token = token,
        retry = sel.state_token.is_some(),
        token_matched,
        choices = sel.choices.len(),
        decisions,
        "apply_patch outcome"
    );
}

/// Split content into lines, keeping each line's terminator (`"\n"`,
/// `"\r\n"`, or `""` for a final line with none).
fn split_lines(content: &str) -> (Vec<String>, Vec<&'static str>) {
    let mut text = Vec::new();
    let mut ends = Vec::new();
    let mut rest = content;
    while !rest.is_empty() {
        match rest.find('\n') {
            Some(p) => {
                let line = &rest[..p];
                match line.strip_suffix('\r') {
                    Some(s) => {
                        text.push(s.to_string());
                        ends.push("\r\n");
                    }
                    None => {
                        text.push(line.to_string());
                        ends.push("\n");
                    }
                }
                rest = &rest[p + 1..];
            }
            None => {
                text.push(rest.to_string());
                ends.push("");
                rest = "";
            }
        }
    }
    (text, ends)
}

/// Apply a sequence of hunks to file content.
///
/// Each hunk's full anchor (`context_before` + `old_lines` + `context_after`)
/// must match exactly one place at or after where the previous hunk ended;
/// there, `old_lines` are replaced with `new_lines`. See the module docs for
/// the placement rules.
#[cfg(test)]
fn apply_hunks(content: &str, hunks: &[Hunk]) -> Result<String, String> {
    let mut pending = Vec::new();
    match apply_hunks_with(content, hunks, "", &Selection::default(), &mut pending)? {
        Some(out) => Ok(out),
        None => Err(pending
            .into_iter()
            .map(|d| d.message)
            .collect::<Vec<_>>()
            .join("; ")),
    }
}

/// [`apply_hunks`] with the caller's choices. `Ok(None)` means at least one
/// hunk matched several places with no valid choice: each such hunk is pushed
/// onto `pending`, and nothing may be written. Later hunks are still checked,
/// searching forward from the earliest place the unresolved hunk could end,
/// so one response covers every ambiguity in the file. If the caller then
/// picks a later place, a later hunk may need a second round.
fn apply_hunks_with(
    content: &str,
    hunks: &[Hunk],
    file: &str,
    sel: &Selection,
    pending: &mut Vec<Decision>,
) -> Result<Option<String>, String> {
    let (mut lines, mut ends) = split_lines(content);
    let crlf = ends.iter().filter(|e| **e == "\r\n").count();
    let lf = ends.iter().filter(|e| **e == "\n").count();
    let nl: &'static str = if crlf > lf { "\r\n" } else { "\n" };
    let had_trailing_newline = ends.last().is_some_and(|e| !e.is_empty());
    // 1-based line number in the original file of each current line; `None`
    // for lines inserted by an earlier hunk. Used only for messages.
    let mut origin: Vec<Option<usize>> = (1..=lines.len()).map(Some).collect();
    // ANAI-298: hunks search forward from where the previous one ended.
    let mut cursor = 0usize;
    let mut unresolved = false;

    for (hunk_idx, hunk) in hunks.iter().enumerate() {
        let n = hunk_idx + 1;

        // ANAI-254: a hunk carrying neither `-` nor `+` lines changes nothing.
        if hunk.old_lines.is_empty() && hunk.new_lines.is_empty() {
            return Err(format!(
                "Hunk {} is context-only: it has no '-' or '+' lines, so there is \
                 nothing to apply. A hunk must state at least one removal or addition.",
                n
            ));
        }

        let pattern: Vec<&str> = hunk
            .context_before
            .iter()
            .chain(hunk.old_lines.iter())
            .chain(hunk.context_after.iter())
            .map(|s| s.as_str())
            .collect();

        // A pure insertion with no anchor at all has nowhere to go but the
        // end of the file.
        let pos = if pattern.is_empty() {
            lines.len()
        } else {
            let chosen = sel.choice_for(file, n);
            match locate(&lines, &origin, &pattern, cursor, hunk, n)? {
                Located::At(p) => {
                    if let Some(c) = chosen {
                        if origin[p] != Some(c.at_line) {
                            return Err(format!(
                                "Hunk {n} ({}): the choice at_line {} is not where it \
                                 matches — it now matches only at {}. Drop that choice \
                                 or re-read the file.",
                                hunk_label(hunk),
                                c.at_line,
                                describe_line(&origin, p)
                            ));
                        }
                    }
                    p
                }
                Located::Ambiguous { hits, message } => {
                    let picked = chosen
                        .and_then(|c| hits.iter().copied().find(|&h| origin[h] == Some(c.at_line)));
                    if let Some(p) = picked {
                        p
                    } else {
                        let note = chosen.map(|c| {
                            format!(
                                "your choice at_line {} is not one of this hunk's matches; \
                                 pick one of the at_line values listed",
                                c.at_line
                            )
                        });
                        let d = decision(
                            &lines, &origin, &pattern, hunk, file, n, &hits, message, note,
                        );
                        if d.candidates.is_empty() {
                            // Every match sits in lines this patch adds; there is
                            // nothing on disk to choose.
                            return Err(d.message);
                        }
                        pending.push(d);
                        unresolved = true;
                        cursor = hits[0] + hunk.context_before.len() + hunk.old_lines.len();
                        continue;
                    }
                }
            }
        };

        let start = pos + hunk.context_before.len();
        let end = start + hunk.old_lines.len();
        let added = hunk.new_lines.len();
        lines.splice(start..end, hunk.new_lines.iter().cloned());
        ends.splice(start..end, std::iter::repeat_n(nl, added));
        origin.splice(start..end, std::iter::repeat_n(None, added));
        cursor = start + added;
    }
    if unresolved {
        return Ok(None);
    }

    let mut out = String::with_capacity(content.len() + 64);
    let last = lines.len();
    for (k, (line, end)) in lines.iter().zip(ends.iter()).enumerate() {
        out.push_str(line);
        if k + 1 < last || had_trailing_newline {
            out.push_str(if end.is_empty() { nl } else { end });
        }
    }
    Ok(Some(out))
}

/// Every start index `>= from` where `pattern` matches `lines`, compared
/// exactly or ignoring trailing whitespace.
fn find_matches(lines: &[String], pattern: &[&str], from: usize, fuzzy: bool) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > lines.len() {
        return Vec::new();
    }
    (from..=lines.len() - pattern.len())
        .filter(|&s| {
            pattern.iter().enumerate().all(|(j, p)| {
                if fuzzy {
                    lines[s + j].trim_end() == p.trim_end()
                } else {
                    lines[s + j] == *p
                }
            })
        })
        .collect()
}

/// Human description of the current line at `pos`, in original-file terms.
fn describe_line(origin: &[Option<usize>], pos: usize) -> String {
    match origin.get(pos) {
        Some(Some(l)) => format!("line {l}"),
        Some(None) => "a line added by an earlier hunk".to_string(),
        None => "the end of the file".to_string(),
    }
}

/// "lines 4, 19, 52 (and 3 more)" in original-file terms.
fn list_lines(origin: &[Option<usize>], hits: &[usize]) -> String {
    let shown: Vec<String> = hits
        .iter()
        .take(MAX_LISTED)
        .map(|&p| match origin.get(p) {
            Some(Some(l)) => l.to_string(),
            _ => "(added line)".to_string(),
        })
        .collect();
    let noun = if hits.len() == 1 { "line" } else { "lines" };
    let more = if hits.len() > MAX_LISTED {
        format!(" (and {} more)", hits.len() - MAX_LISTED)
    } else {
        String::new()
    };
    format!("{noun} {}{more}", shown.join(", "))
}

/// Short identifier for a hunk in messages: its first changed line.
fn hunk_label(h: &Hunk) -> String {
    let first = h
        .old_lines
        .first()
        .map(|l| ('-', l))
        .or_else(|| h.new_lines.first().map(|l| ('+', l)));
    match first {
        Some((sign, text)) => {
            let t = text.trim();
            let short: String = t.chars().take(60).collect();
            let ell = if t.chars().count() > 60 { "…" } else { "" };
            format!("first change `{sign}{short}{ell}`")
        }
        None => "empty".to_string(),
    }
}

/// Where a hunk's anchor matched.
enum Located {
    At(usize),
    /// Several places. `message` is the stand-alone error text.
    Ambiguous {
        hits: Vec<usize>,
        message: String,
    },
}

fn clip(s: &str) -> String {
    if s.chars().count() <= MAX_LINE_CHARS {
        s.to_string()
    } else {
        let head: String = s.chars().take(MAX_LINE_CHARS).collect();
        format!("{head}…")
    }
}

fn indent_of(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

/// Describe an ambiguous hunk's candidates for the caller to choose from.
#[allow(clippy::too_many_arguments)]
fn decision(
    lines: &[String],
    origin: &[Option<usize>],
    pattern: &[&str],
    hunk: &Hunk,
    file: &str,
    n: usize,
    hits: &[usize],
    message: String,
    note: Option<String>,
) -> Decision {
    let anchor_indent = pattern
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| indent_of(l))
        .unwrap_or(0);
    let quote = |k: usize| (origin[k], clip(&lines[k]));
    let mut candidates = Vec::new();
    let mut unselectable = 0usize;
    for &h in hits {
        let Some(at_line) = origin[h] else {
            unselectable += 1;
            continue;
        };
        if candidates.len() >= MAX_LISTED {
            continue;
        }
        let end = h + pattern.len();
        let within = (h.saturating_sub(WITHIN_SCAN)..h)
            .rev()
            .find(|&k| {
                let t = lines[k].trim();
                !t.is_empty()
                    && indent_of(&lines[k]) < anchor_indent
                    && !t.chars().all(|c| matches!(c, '}' | ']' | ')' | ';' | ','))
            })
            .map(quote);
        candidates.push(Candidate {
            at_line,
            within,
            before: (h.saturating_sub(SHOWN_CONTEXT)..h).map(quote).collect(),
            after: (end..(end + SHOWN_CONTEXT).min(lines.len()))
                .map(quote)
                .collect(),
        });
    }
    let mut notes: Vec<String> = note.into_iter().collect();
    if unselectable > 0 {
        notes.push(format!(
            "{} more {} inside lines this patch adds and cannot be chosen",
            unselectable,
            if unselectable == 1 {
                "match is"
            } else {
                "matches are"
            }
        ));
    }
    Decision {
        file: file.to_string(),
        hunk: n,
        label: hunk_label(hunk),
        total: hits.len(),
        candidates,
        note: (!notes.is_empty()).then(|| notes.join("; ")),
        message,
    }
}

/// Find the single place `pattern` belongs, searching from `cursor` (and
/// from the `@@` scope line, when the hunk names one). Zero or several
/// matches is an error that says which, and where.
fn locate(
    lines: &[String],
    origin: &[Option<usize>],
    pattern: &[&str],
    cursor: usize,
    hunk: &Hunk,
    n: usize,
) -> Result<Located, String> {
    let label = hunk_label(hunk);
    let mut from = cursor;
    let mut where_ = if cursor == 0 {
        " in the file".to_string()
    } else {
        format!(
            " at or after {} (where the previous hunk ended)",
            describe_line(origin, cursor)
        )
    };

    if let Some(scope) = hunk.scope.as_deref() {
        match (cursor..lines.len()).find(|&k| lines[k].contains(scope)) {
            Some(k) => {
                from = k;
                where_ = format!(
                    " at or after the '@@ {scope}' line ({})",
                    describe_line(origin, k)
                );
            }
            None => {
                return Err(format!(
                    "Hunk {n} ({label}) failed: its '@@ {scope}' header must quote text \
                     from a line at or above the change, but no line{where_} contains it. \
                     Copy the header text exactly from the file (e.g. `@@ fn handle_request`), \
                     or use a bare `@@`."
                ));
            }
        }
    }

    for fuzzy in [false, true] {
        let hits = find_matches(lines, pattern, from, fuzzy);
        match hits.as_slice() {
            [] => continue,
            [one] => {
                if fuzzy {
                    warn!(
                        "Patch hunk {} matched ignoring trailing whitespace at {}",
                        n,
                        describe_line(origin, *one)
                    );
                }
                return Ok(Located::At(*one));
            }
            _ => {
                let message = format!(
                    "Hunk {n} ({label}) is ambiguous: its context and '-' lines match \
                     {} places{where_}, starting at {}. Add context lines that occur only \
                     at the intended place, or start the hunk with `@@ <text of a unique \
                     line above it>` (e.g. the enclosing fn signature).",
                    hits.len(),
                    list_lines(origin, &hits)
                );
                return Ok(Located::Ambiguous { hits, message });
            }
        }
    }

    // Not found. Say why, as precisely as we can.
    let anywhere = find_matches(lines, pattern, 0, true);
    if !anywhere.is_empty() {
        return Err(format!(
            "Hunk {n} ({label}) failed: no match{where_}. It does match at {}, above \
             where the search started. Hunks must be listed in file order, top to bottom, \
             and an `@@` header must name a line above the change.",
            list_lines(origin, &anywhere)
        ));
    }
    if !hunk.context_after.is_empty() {
        let head: Vec<&str> = hunk
            .context_before
            .iter()
            .chain(hunk.old_lines.iter())
            .map(|s| s.as_str())
            .collect();
        let partial = find_matches(lines, &head, from, true);
        if !head.is_empty() && !partial.is_empty() {
            return Err(format!(
                "Hunk {n} ({label}) failed: its leading context and '-' lines match at {}, \
                 but its trailing context lines do not follow them there. Re-read the \
                 file and copy the lines after the change exactly.",
                list_lines(origin, &partial)
            ));
        }
    }
    Err(format!(
        "Hunk {n} ({label}) failed: could not find its context and '-' lines{where_}. \
         Re-read the file and copy them exactly; only trailing whitespace is ignored."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_add_file() {
        let patch = "\
*** Begin Patch
*** Add File: src/new.rs
+fn main() {
+    println!(\"hello\");
+}
*** End Patch";
        let ops = parse_patch(patch).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOp::AddFile { path, content } => {
                assert_eq!(path, "src/new.rs");
                assert!(content.contains("fn main()"));
            }
            _ => panic!("Expected AddFile"),
        }
    }

    #[test]
    fn test_parse_update_file() {
        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@ hunk 1 @@
 fn existing() {
-    old_code();
+    new_code();
 }
*** End Patch";
        let ops = parse_patch(patch).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOp::UpdateFile {
                path,
                hunks,
                move_to,
            } => {
                assert_eq!(path, "src/lib.rs");
                assert!(move_to.is_none());
                assert_eq!(hunks.len(), 1);
                assert_eq!(hunks[0].context_before, vec!["fn existing() {"]);
                assert_eq!(hunks[0].old_lines, vec!["    old_code();"]);
                assert_eq!(hunks[0].new_lines, vec!["    new_code();"]);
                assert_eq!(hunks[0].context_after, vec!["}"]);
            }
            _ => panic!("Expected UpdateFile"),
        }
    }

    #[test]
    fn test_parse_delete_file() {
        let patch = "\
*** Begin Patch
*** Delete File: src/old.rs
*** End Patch";
        let ops = parse_patch(patch).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOp::DeleteFile { path } => assert_eq!(path, "src/old.rs"),
            _ => panic!("Expected DeleteFile"),
        }
    }

    #[test]
    fn test_parse_move_file() {
        let patch = "\
*** Begin Patch
*** Update File: old/path.rs -> new/path.rs
@@ hunk @@
 keep_this
-remove_this
+add_this
*** End Patch";
        let ops = parse_patch(patch).unwrap();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOp::UpdateFile { path, move_to, .. } => {
                assert_eq!(path, "old/path.rs");
                assert_eq!(move_to.as_deref(), Some("new/path.rs"));
            }
            _ => panic!("Expected UpdateFile"),
        }
    }

    #[test]
    fn test_parse_multi_op() {
        let patch = "\
*** Begin Patch
*** Add File: a.txt
+hello
*** Delete File: b.txt
*** Update File: c.txt
@@ hunk @@
-old
+new
*** End Patch";
        let ops = parse_patch(patch).unwrap();
        assert_eq!(ops.len(), 3);
        assert!(matches!(&ops[0], PatchOp::AddFile { .. }));
        assert!(matches!(&ops[1], PatchOp::DeleteFile { .. }));
        assert!(matches!(&ops[2], PatchOp::UpdateFile { .. }));
    }

    #[test]
    fn test_parse_missing_begin() {
        let patch = "*** Add File: a.txt\n+hello\n*** End Patch";
        assert!(parse_patch(patch).is_err());
    }

    #[test]
    fn test_parse_missing_end() {
        let patch = "*** Begin Patch\n*** Add File: a.txt\n+hello";
        assert!(parse_patch(patch).is_err());
    }

    #[test]
    fn test_parse_empty_patch() {
        let patch = "*** Begin Patch\n*** End Patch";
        assert!(parse_patch(patch).is_err());
    }

    #[test]
    fn test_apply_hunks_simple() {
        let content = "line1\nline2\nline3\n";
        let hunks = vec![Hunk {
            context_before: vec!["line1".to_string()],
            old_lines: vec!["line2".to_string()],
            new_lines: vec!["replaced".to_string()],
            context_after: vec![],
            scope: None,
        }];
        let result = apply_hunks(content, &hunks).unwrap();
        assert!(result.contains("replaced"));
        assert!(!result.contains("line2"));
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
    }

    #[test]
    fn test_apply_hunks_multi_hunk() {
        let content = "a\nb\nc\nd\ne\n";
        let hunks = vec![
            Hunk {
                context_before: vec!["a".to_string()],
                old_lines: vec!["b".to_string()],
                new_lines: vec!["B".to_string()],
                context_after: vec![],
                scope: None,
            },
            Hunk {
                context_before: vec!["c".to_string()],
                old_lines: vec!["d".to_string()],
                new_lines: vec!["D".to_string(), "D2".to_string()],
                context_after: vec![],
                scope: None,
            },
        ];
        let result = apply_hunks(content, &hunks).unwrap();
        assert!(result.contains("B"));
        assert!(result.contains("D\nD2"));
        assert!(!result.contains("\nb\n"));
        assert!(!result.contains("\nd\n"));
    }

    #[test]
    fn test_apply_hunks_context_mismatch() {
        let content = "alpha\nbeta\ngamma\n";
        let hunks = vec![Hunk {
            context_before: vec!["nonexistent".to_string()],
            old_lines: vec!["also_nonexistent".to_string()],
            new_lines: vec!["new".to_string()],
            context_after: vec![],
            scope: None,
        }];
        assert!(apply_hunks(content, &hunks).is_err());
    }

    #[test]
    fn test_apply_hunks_fuzzy_whitespace() {
        let content = "line1  \nline2\t\nline3\n";
        let hunks = vec![Hunk {
            context_before: vec!["line1".to_string()],
            old_lines: vec!["line2".to_string()],
            new_lines: vec!["replaced".to_string()],
            context_after: vec![],
            scope: None,
        }];
        let result = apply_hunks(content, &hunks).unwrap();
        assert!(result.contains("replaced"));
    }

    #[test]
    fn test_apply_hunks_preserves_unchanged() {
        let content = "header\nkeep1\nkeep2\nold_line\nkeep3\nfooter\n";
        let hunks = vec![Hunk {
            context_before: vec!["keep2".to_string()],
            old_lines: vec!["old_line".to_string()],
            new_lines: vec!["new_line".to_string()],
            context_after: vec![],
            scope: None,
        }];
        let result = apply_hunks(content, &hunks).unwrap();
        assert!(result.contains("header"));
        assert!(result.contains("keep1"));
        assert!(result.contains("keep2"));
        assert!(result.contains("new_line"));
        assert!(result.contains("keep3"));
        assert!(result.contains("footer"));
        assert!(!result.contains("old_line"));
    }

    #[test]
    fn test_find_matches_exact() {
        let lines: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(find_matches(&lines, &["b", "c"], 0, false), vec![1]);
    }

    #[test]
    fn test_find_matches_not_found() {
        let lines: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        assert!(find_matches(&lines, &["x", "y"], 0, false).is_empty());
    }

    #[test]
    fn test_find_matches_fuzzy() {
        let lines: Vec<String> = vec!["a  ", "b\t", "c"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(find_matches(&lines, &["a", "b"], 0, true), vec![0]);
    }

    #[tokio::test]
    async fn test_apply_patch_integration() {
        let dir = std::env::temp_dir().join("openfang_patch_test");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        // Write a file to update
        tokio::fs::write(dir.join("existing.txt"), "line1\nline2\nline3\n")
            .await
            .unwrap();

        let ops = vec![
            PatchOp::AddFile {
                path: "new.txt".to_string(),
                content: "hello world".to_string(),
            },
            PatchOp::UpdateFile {
                path: "existing.txt".to_string(),
                move_to: None,
                hunks: vec![Hunk {
                    context_before: vec!["line1".to_string()],
                    old_lines: vec!["line2".to_string()],
                    new_lines: vec!["replaced".to_string()],
                    context_after: vec![],
                    scope: None,
                }],
            },
        ];

        let result = apply_patch(&ops, &dir, None, None, &Selection::default()).await;
        assert!(result.is_ok());
        assert_eq!(result.files_added, 1);
        assert_eq!(result.files_updated, 1);

        // Verify files
        let new_content = tokio::fs::read_to_string(dir.join("new.txt"))
            .await
            .unwrap();
        assert_eq!(new_content, "hello world");

        let updated = tokio::fs::read_to_string(dir.join("existing.txt"))
            .await
            .unwrap();
        assert!(updated.contains("replaced"));
        assert!(!updated.contains("line2"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_apply_patch_delete() {
        let dir = std::env::temp_dir().join("openfang_patch_del_test");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        tokio::fs::write(dir.join("doomed.txt"), "goodbye")
            .await
            .unwrap();

        let ops = vec![PatchOp::DeleteFile {
            path: "doomed.txt".to_string(),
        }];

        let result = apply_patch(&ops, &dir, None, None, &Selection::default()).await;
        assert!(result.is_ok());
        assert_eq!(result.files_deleted, 1);
        assert!(!dir.join("doomed.txt").exists());

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_apply_patch_f3_denied_target_blocks_all_writes() {
        use openfang_types::config::{FileAccessTier, FilePolicy, FileRule};
        let dir = std::env::temp_dir().join("openfang_patch_f3_test");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(dir.join("secret")).await.unwrap();

        // Workspace writable, but a `secret/` subtree is denied.
        let policy = FilePolicy::new(
            true,
            FileAccessTier::Write,
            vec![FileRule {
                path: "secret".to_string(),
                tier: FileAccessTier::Deny,
            }],
        );

        // First op is allowed; second targets the denied subtree.
        let ops = vec![
            PatchOp::AddFile {
                path: "allowed.txt".to_string(),
                content: "hi".to_string(),
            },
            PatchOp::AddFile {
                path: "secret/leak.txt".to_string(),
                content: "x".to_string(),
            },
        ];

        let result = apply_patch(&ops, &dir, Some(&policy), None, &Selection::default()).await;
        assert!(
            !result.is_ok(),
            "patch must be rejected when any target is policy-denied"
        );
        assert_eq!(
            result.files_added, 0,
            "no writes past a policy-denied target (pre-pass)"
        );
        assert!(
            !dir.join("allowed.txt").exists(),
            "the allowed target must not be written when the patch is rejected"
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    // ---- ANAI-254: silent no-op writes ----

    fn hunk(before: &[&str], old: &[&str], new: &[&str], after: &[&str]) -> Hunk {
        Hunk {
            context_before: before.iter().map(|s| s.to_string()).collect(),
            old_lines: old.iter().map(|s| s.to_string()).collect(),
            new_lines: new.iter().map(|s| s.to_string()).collect(),
            context_after: after.iter().map(|s| s.to_string()).collect(),
            scope: None,
        }
    }

    #[test]
    fn a_context_only_hunk_is_refused_rather_than_silently_applied() {
        // Previously this rewrote the file with identical bytes and reported
        // success. The whole point of the fix: refusing loudly is fine, lying
        // is not.
        let err = apply_hunks(
            "alpha\nbravo\n",
            &[hunk(&["alpha", "bravo"], &[], &[], &[])],
        )
        .expect_err("a hunk with no '-' or '+' lines must be an error");
        assert!(
            err.contains("context-only"),
            "the error must name the cause, got: {err}"
        );
    }

    #[tokio::test]
    async fn a_patch_that_changes_nothing_is_not_counted_as_updated() {
        let dir = std::env::temp_dir().join(format!("of_patch_noop_{}", std::process::id()));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("f.txt"), "alpha\nbravo\n")
            .await
            .unwrap();

        // Hunks that resolve to exactly the existing content.
        let ops = vec![PatchOp::UpdateFile {
            path: "f.txt".to_string(),
            move_to: None,
            hunks: vec![hunk(&["alpha"], &["bravo"], &["bravo"], &[])],
        }];
        let result = apply_patch(&ops, &dir, None, None, &Selection::default()).await;

        assert_eq!(
            result.files_updated, 0,
            "the counter must derive from a confirmed change, not from a successful write"
        );
        assert!(
            !result.is_ok(),
            "a no-op patch must surface an error, not report success"
        );
        assert!(
            result.summary().contains("error"),
            "the summary the agent reads must not say 'updated', got: {}",
            result.summary()
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[test]
    fn a_pure_insertion_lands_before_its_trailing_context_not_at_eof() {
        let out = apply_hunks(
            "alpha\nbravo\ncharlie\n",
            &[hunk(&[], &[], &["INSERTED"], &["charlie"])],
        )
        .unwrap();
        assert_eq!(out, "alpha\nbravo\nINSERTED\ncharlie\n");
    }

    #[test]
    fn a_pure_insertion_with_no_anchor_at_all_still_appends() {
        // Back-compat: with neither leading nor trailing context there is
        // nothing to anchor on, so end-of-file remains the only answer.
        let out = apply_hunks("alpha\nbravo\n", &[hunk(&[], &[], &["TAIL"], &[])]).unwrap();
        assert_eq!(out, "alpha\nbravo\nTAIL\n");
    }

    #[test]
    fn a_hunk_with_two_change_regions_splits_and_both_apply() {
        let ops = parse_patch(
            "*** Begin Patch\n\
             *** Update File: f.txt\n\
             @@\n\
             \x20alpha\n\
             -bravo\n\
             +BRAVO\n\
             \x20charlie\n\
             -delta\n\
             +DELTA\n\
             *** End Patch\n",
        )
        .expect("a two-region hunk must parse");

        let hunks = match &ops[0] {
            PatchOp::UpdateFile { hunks, .. } => hunks,
            other => panic!("expected an update op, got {other:?}"),
        };
        assert_eq!(
            hunks.len(),
            2,
            "a change resuming after trailing context starts a new hunk"
        );

        let out = apply_hunks("alpha\nbravo\ncharlie\ndelta\n", hunks)
            .expect("both regions must anchor against the file");
        assert_eq!(out, "alpha\nBRAVO\ncharlie\nDELTA\n");
    }

    // ---- ANAI-298: line endings ----

    #[test]
    fn crlf_line_endings_survive_a_patch() {
        let out = apply_hunks(
            "alpha\r\nbravo\r\ncharlie\r\n",
            &[hunk(
                &["alpha"],
                &["bravo"],
                &["BRAVO", "BRAVO2"],
                &["charlie"],
            )],
        )
        .unwrap();
        assert_eq!(out, "alpha\r\nBRAVO\r\nBRAVO2\r\ncharlie\r\n");
    }

    #[test]
    fn mixed_line_endings_are_preserved_line_by_line() {
        // Untouched lines keep their own terminator; inserted lines take the
        // file's dominant one (CRLF here, 2 of 3).
        let out = apply_hunks(
            "alpha\r\nbravo\ncharlie\r\n",
            &[hunk(&["bravo"], &[], &["NEW"], &["charlie"])],
        )
        .unwrap();
        assert_eq!(out, "alpha\r\nbravo\nNEW\r\ncharlie\r\n");
    }

    #[test]
    fn a_file_without_a_trailing_newline_stays_that_way() {
        let out = apply_hunks(
            "alpha\r\nbravo",
            &[hunk(&["alpha"], &["bravo"], &["B"], &[])],
        )
        .unwrap();
        assert_eq!(out, "alpha\r\nB");
    }

    // ---- ANAI-298 / ANAI-120: placement ----

    fn scoped(scope: &str, mut h: Hunk) -> Hunk {
        h.scope = Some(scope.to_string());
        h
    }

    fn update(path: &str, hunks: Vec<Hunk>) -> PatchOp {
        PatchOp::UpdateFile {
            path: path.to_string(),
            move_to: None,
            hunks,
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("of_patch_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_later_hunk_searches_forward_from_the_previous_one() {
        // ANAI-298: hunk 2's anchor also occurs above hunk 1. Searching from
        // line 1 edited the first copy; it must edit the one below hunk 1.
        let out = apply_hunks(
            "x\ntarget\ny\nA\nB\nx\ntarget\ny\n",
            &[
                hunk(&["A"], &["B"], &["B2"], &[]),
                hunk(&["x"], &["target"], &["T"], &["y"]),
            ],
        )
        .unwrap();
        assert_eq!(out, "x\ntarget\ny\nA\nB2\nx\nT\ny\n");
    }

    #[test]
    fn a_hunk_matching_two_places_is_refused_and_names_both() {
        // ANAI-120: near-duplicate blocks. Taking the first match silently
        // edited the wrong copy.
        let err = apply_hunks(
            "fn a() {\n    call();\n}\nfn b() {\n    call();\n}\n",
            &[hunk(&[], &["    call();"], &["    other();"], &["}"])],
        )
        .expect_err("two equal matches must not be guessed between");
        assert!(err.contains("ambiguous"), "got: {err}");
        assert!(
            err.contains("lines 2, 5"),
            "must list the candidates, got: {err}"
        );
    }

    #[test]
    fn trailing_context_is_part_of_the_anchor() {
        // media.rs: an insertion after a bare `}` landed after the FIRST `}`
        // because the trailing context was never checked.
        let out = apply_hunks(
            "fn a() {\n}\n\nfn b() {\n}\n\nfn tail() {}\n",
            &[hunk(&["}", ""], &[], &["// NEW", ""], &["fn tail() {}"])],
        )
        .unwrap();
        assert_eq!(
            out,
            "fn a() {\n}\n\nfn b() {\n}\n\n// NEW\n\nfn tail() {}\n"
        );
    }

    #[test]
    fn an_at_at_scope_narrows_the_search() {
        let out = apply_hunks(
            "fn a() {\n    call();\n}\nfn b() {\n    call();\n}\n",
            &[scoped(
                "fn b",
                hunk(&[], &["    call();"], &["    other();"], &["}"]),
            )],
        )
        .unwrap();
        assert_eq!(out, "fn a() {\n    call();\n}\nfn b() {\n    other();\n}\n");
    }

    #[test]
    fn an_at_at_scope_that_is_not_in_the_file_is_an_error() {
        let err = apply_hunks(
            "fn a() {\n    call();\n}\n",
            &[scoped(
                "fn zzz",
                hunk(&[], &["    call();"], &["    x();"], &[]),
            )],
        )
        .expect_err("a scope naming nothing must not be ignored");
        assert!(err.contains("fn zzz"), "got: {err}");
    }

    #[test]
    fn out_of_order_hunks_are_refused_with_a_file_order_hint() {
        let err = apply_hunks(
            "a\nb\nc\nd\n",
            &[
                hunk(&["c"], &["d"], &["D"], &[]),
                hunk(&["a"], &["b"], &["B"], &[]),
            ],
        )
        .expect_err("hunk 2 lies above hunk 1");
        assert!(err.contains("file order"), "got: {err}");
        assert!(
            err.contains("line 1"),
            "must say where it does match, got: {err}"
        );
    }

    #[test]
    fn a_wrong_trailing_context_is_diagnosed_as_such() {
        let err = apply_hunks("a\nb\nc\n", &[hunk(&["a"], &["b"], &["B"], &["WRONG"])])
            .expect_err("trailing context does not match");
        assert!(err.contains("trailing context"), "got: {err}");
    }

    #[test]
    fn context_lines_keep_the_files_own_whitespace() {
        // The fuzzy tier matches despite trailing whitespace; the context
        // lines themselves must not be rewritten to the patch's version.
        let out =
            apply_hunks("keep  \nold\n", &[hunk(&["keep"], &["old"], &["new"], &[])]).unwrap();
        assert_eq!(out, "keep  \nnew\n");
    }

    // ---- ANAI-298: parser strictness ----

    fn only_update_hunks(patch: &str) -> Vec<Hunk> {
        match parse_patch(patch).unwrap().remove(0) {
            PatchOp::UpdateFile { hunks, .. } => hunks,
            other => panic!("expected an update op, got {other:?}"),
        }
    }

    #[test]
    fn content_lines_that_look_like_markers_stay_in_the_hunk() {
        let hunks = only_update_hunks(
            "*** Begin Patch\n\
             *** Update File: README.md\n\
             @@\n\
             \x20*** bold banner ***\n\
             -@@ old note\n\
             +@@ new note\n\
             \x20tail\n\
             *** End Patch\n",
        );
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].context_before, vec!["*** bold banner ***"]);
        assert_eq!(hunks[0].old_lines, vec!["@@ old note"]);
        assert_eq!(hunks[0].new_lines, vec!["@@ new note"]);
        assert_eq!(hunks[0].context_after, vec!["tail"]);
    }

    #[test]
    fn an_unprefixed_hunk_line_is_rejected_with_its_line_number() {
        let err = parse_patch(
            "*** Begin Patch\n\
             *** Update File: f.txt\n\
             @@\n\
             \x20keep\n\
             oops no prefix\n\
             +new\n\
             *** End Patch\n",
        )
        .expect_err("a line with no ' '/'-'/'+' prefix is malformed");
        assert!(err.contains("patch line 5"), "got: {err}");
    }

    #[test]
    fn trailing_bare_blank_lines_are_not_context() {
        let hunks = only_update_hunks(
            "*** Begin Patch\n\
             *** Update File: f.txt\n\
             @@\n\
             \x20a\n\
             -b\n\
             +B\n\
             \n\
             \n\
             *** End Patch\n",
        );
        assert!(
            hunks[0].context_after.is_empty(),
            "got {:?}",
            hunks[0].context_after
        );
        assert_eq!(apply_hunks("a\nb\n", &hunks).unwrap(), "a\nB\n");
    }

    #[test]
    fn hunk_headers_yield_a_scope_only_when_they_name_something() {
        assert_eq!(parse_hunk_header("@@"), None);
        assert_eq!(parse_hunk_header("@@ @@"), None);
        assert_eq!(parse_hunk_header("@@ -12,5 +12,6 @@"), None);
        assert_eq!(
            parse_hunk_header("@@ -3 +3,2 @@ fn foo()"),
            Some("fn foo()".into())
        );
        assert_eq!(parse_hunk_header("@@ fn foo"), Some("fn foo".into()));
        assert_eq!(parse_hunk_header("@@ fn foo @@"), Some("fn foo".into()));
    }

    #[test]
    fn a_first_hunk_without_a_header_is_accepted() {
        let hunks = only_update_hunks(
            "*** Begin Patch\n*** Update File: f.txt\n a\n-b\n+B\n*** End Patch\n",
        );
        assert_eq!(apply_hunks("a\nb\n", &hunks).unwrap(), "a\nB\n");
    }

    #[test]
    fn a_split_hunk_keeps_its_scope_on_the_first_region_only() {
        let hunks = only_update_hunks(
            "*** Begin Patch\n*** Update File: f.txt\n@@ fn b\n a\n-b\n+B\n c\n-d\n+D\n*** End Patch\n",
        );
        assert_eq!(hunks.len(), 2);
        assert_eq!(hunks[0].scope.as_deref(), Some("fn b"));
        assert_eq!(hunks[1].scope, None);
    }

    #[test]
    fn blank_lines_inside_add_file_content_are_kept() {
        let ops =
            parse_patch("*** Begin Patch\n*** Add File: n.txt\n+one\n\n+three\n\n*** End Patch\n")
                .unwrap();
        match &ops[0] {
            PatchOp::AddFile { content, .. } => assert_eq!(content, "one\n\nthree"),
            other => panic!("expected an add op, got {other:?}"),
        }
    }

    // ---- ANAI-298: all or nothing across files ----

    #[tokio::test]
    async fn a_failing_hunk_in_a_later_file_leaves_earlier_files_untouched() {
        let dir = temp_dir("atomic");
        std::fs::write(dir.join("one.txt"), "a\nb\n").unwrap();
        std::fs::write(dir.join("two.txt"), "x\ny\n").unwrap();
        let ops = vec![
            update("one.txt", vec![hunk(&["a"], &["b"], &["B"], &[])]),
            update("two.txt", vec![hunk(&["x"], &["NOPE"], &["Y"], &[])]),
        ];
        let result = apply_patch(&ops, &dir, None, None, &Selection::default()).await;
        assert!(!result.is_ok());
        assert_eq!(result.files_updated, 0);
        assert_eq!(
            std::fs::read_to_string(dir.join("one.txt")).unwrap(),
            "a\nb\n"
        );
        assert!(
            result.summary().contains("nothing was written"),
            "{}",
            result.summary()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_update_can_follow_an_add_of_the_same_file_in_one_patch() {
        let dir = temp_dir("overlay");
        let ops = vec![
            PatchOp::AddFile {
                path: "n.txt".to_string(),
                content: "a\nb\n".to_string(),
            },
            update("n.txt", vec![hunk(&["a"], &["b"], &["B"], &[])]),
        ];
        let result = apply_patch(&ops, &dir, None, None, &Selection::default()).await;
        assert!(result.is_ok(), "{:?}", result.errors);
        assert_eq!(
            std::fs::read_to_string(dir.join("n.txt")).unwrap(),
            "a\nB\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn deleting_a_missing_file_rejects_the_patch_before_any_write() {
        let dir = temp_dir("delmissing");
        let ops = vec![
            PatchOp::AddFile {
                path: "n.txt".to_string(),
                content: "x".to_string(),
            },
            PatchOp::DeleteFile {
                path: "ghost.txt".to_string(),
            },
        ];
        let result = apply_patch(&ops, &dir, None, None, &Selection::default()).await;
        assert!(!result.is_ok());
        assert!(!dir.join("n.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- ANAI-298 B: needs_input round trip ----

    const TWIN: &str = "fn a() {\n    call();\n}\nfn b() {\n    call();\n}\n";

    fn twin_hunk() -> Hunk {
        hunk(&[], &["    call();"], &["    other();"], &["}"])
    }

    async fn ask(dir: &Path, ops: &[PatchOp]) -> NeedsInput {
        let r = apply_patch(ops, dir, None, None, &Selection::default()).await;
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert!(!r.is_ok());
        r.needs_input
            .expect("an ambiguous hunk must come back as needs_input")
    }

    fn pick(ni: &NeedsInput, file: &str, hunk: usize, at_line: usize) -> Selection {
        Selection {
            state_token: Some(ni.state_token.clone()),
            choices: vec![Choice {
                file: file.to_string(),
                hunk,
                at_line,
            }],
        }
    }

    fn at_lines(d: &Decision) -> Vec<usize> {
        d.candidates.iter().map(|c| c.at_line).collect()
    }

    #[tokio::test]
    async fn an_ambiguous_hunk_asks_instead_of_failing_and_writes_nothing() {
        let dir = temp_dir("ask");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        let ni = ask(&dir, &[update("t.rs", vec![twin_hunk()])]).await;
        assert_eq!(std::fs::read_to_string(dir.join("t.rs")).unwrap(), TWIN);
        assert_eq!(ni.decisions.len(), 1);
        let d = &ni.decisions[0];
        assert_eq!((d.file.as_str(), d.hunk, d.total), ("t.rs", 1, 2));
        assert_eq!(at_lines(d), vec![2, 5]);
        // The anchor is identical at both; `within` is what tells them apart.
        let within: Vec<&str> = d
            .candidates
            .iter()
            .map(|c| c.within.as_ref().unwrap().1.as_str())
            .collect();
        assert_eq!(within, vec!["fn a() {", "fn b() {"]);
        assert!(!ni.stale_token);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_retry_with_the_token_and_a_choice_applies_there() {
        let dir = temp_dir("choose");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        let ops = vec![update("t.rs", vec![twin_hunk()])];
        let ni = ask(&dir, &ops).await;
        let r = apply_patch(&ops, &dir, None, None, &pick(&ni, "t.rs", 1, 5)).await;
        assert!(r.is_ok(), "{:?} {:?}", r.errors, r.needs_input);
        assert_eq!(r.files_updated, 1);
        assert_eq!(
            std::fs::read_to_string(dir.join("t.rs")).unwrap(),
            "fn a() {\n    call();\n}\nfn b() {\n    other();\n}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_stale_token_voids_its_choices_and_asks_again() {
        let dir = temp_dir("stale");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        let ops = vec![update("t.rs", vec![twin_hunk()])];
        let ni = ask(&dir, &ops).await;
        // Another writer shifts the file between the question and the answer.
        let moved = format!("// moved\n{TWIN}");
        std::fs::write(dir.join("t.rs"), &moved).unwrap();
        let r = apply_patch(&ops, &dir, None, None, &pick(&ni, "t.rs", 1, 5)).await;
        let again = r.needs_input.expect("stale choices must not be applied");
        assert!(again.stale_token);
        assert_ne!(again.state_token, ni.state_token);
        assert_eq!(at_lines(&again.decisions[0]), vec![3, 6]);
        assert_eq!(std::fs::read_to_string(dir.join("t.rs")).unwrap(), moved);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_changed_patch_also_voids_the_token() {
        let dir = temp_dir("stalepatch");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        let ni = ask(&dir, &[update("t.rs", vec![twin_hunk()])]).await;
        let other = vec![update(
            "t.rs",
            vec![hunk(&[], &["    call();"], &["    third();"], &["}"])],
        )];
        let r = apply_patch(&other, &dir, None, None, &pick(&ni, "t.rs", 1, 5)).await;
        assert!(r.needs_input.expect("must ask again").stale_token);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_choice_that_is_not_a_match_is_asked_again_with_a_note() {
        let dir = temp_dir("badpick");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        let ops = vec![update("t.rs", vec![twin_hunk()])];
        let ni = ask(&dir, &ops).await;
        let r = apply_patch(&ops, &dir, None, None, &pick(&ni, "t.rs", 1, 4)).await;
        let again = r
            .needs_input
            .expect("an off-target pick must not be applied");
        let note = again.decisions[0].note.as_deref().unwrap_or("");
        assert!(note.contains("at_line 4"), "{note}");
        assert_eq!(std::fs::read_to_string(dir.join("t.rs")).unwrap(), TWIN);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn every_ambiguous_hunk_in_every_file_is_reported_at_once() {
        let dir = temp_dir("askall");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        std::fs::write(dir.join("u.rs"), TWIN).unwrap();
        std::fs::write(dir.join("c.txt"), "a\nb\n").unwrap();
        let ops = vec![
            update("c.txt", vec![hunk(&["a"], &["b"], &["B"], &[])]),
            update("t.rs", vec![twin_hunk()]),
            update("u.rs", vec![twin_hunk()]),
        ];
        let ni = ask(&dir, &ops).await;
        let files: Vec<&str> = ni.decisions.iter().map(|d| d.file.as_str()).collect();
        assert_eq!(files, vec!["t.rs", "u.rs"]);
        assert_eq!(
            std::fs::read_to_string(dir.join("c.txt")).unwrap(),
            "a\nb\n",
            "the unambiguous file must not be written either"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn later_hunks_are_offered_only_places_below_the_earliest_candidate() {
        let mut pending = Vec::new();
        let out = apply_hunks_with(
            "x\nk\nx\nk\nx\n",
            &[
                hunk(&[], &["k"], &["K"], &[]),
                hunk(&[], &["x"], &["X"], &[]),
            ],
            "f",
            &Selection::default(),
            &mut pending,
        )
        .unwrap();
        assert!(out.is_none());
        assert_eq!(pending.len(), 2);
        assert_eq!(at_lines(&pending[0]), vec![2, 4]);
        // Line 1 is above every place hunk 1 could go, so it is not offered.
        assert_eq!(at_lines(&pending[1]), vec![3, 5]);
    }

    #[test]
    fn candidates_are_capped_but_the_total_is_kept() {
        let content = "    dup();\n".repeat(500);
        let mut pending = Vec::new();
        apply_hunks_with(
            &content,
            &[hunk(&[], &["    dup();"], &["    one();"], &[])],
            "f",
            &Selection::default(),
            &mut pending,
        )
        .unwrap();
        let d = &pending[0];
        assert_eq!(d.total, 500);
        assert_eq!(d.candidates.len(), MAX_LISTED);
        let ni = NeedsInput {
            state_token: "t".to_string(),
            decisions: pending.clone(),
            stale_token: false,
        };
        let text = ni.render("p");
        assert!(text.len() < 32 * 1024, "response is {} bytes", text.len());
        assert!(text.contains("\"n_more\": 490"), "{text}");
    }

    #[test]
    fn selection_requires_a_token_and_numeric_picks() {
        use serde_json::json;
        let e =
            Selection::from_input(&json!({"choices": [{"file": "a", "hunk": 1, "at_line": 2}]}))
                .unwrap_err();
        assert!(e.contains("state_token"), "{e}");
        // Copying the retry template without making the pick is refused.
        let e = Selection::from_input(&json!({
            "state_token": "t",
            "choices": [{"file": "a", "hunk": 1, "at_line": "PICK ONE OF 2, 5"}]
        }))
        .unwrap_err();
        assert!(e.contains("at_line"), "{e}");
        let s = Selection::from_input(&json!({"patch": "x"})).unwrap();
        assert!(s.state_token.is_none() && s.choices.is_empty());
    }

    #[tokio::test]
    async fn the_rendered_question_leads_with_not_applied_and_its_choices_round_trip() {
        use serde_json::json;
        let dir = temp_dir("render");
        std::fs::write(dir.join("t.rs"), TWIN).unwrap();
        let ops = vec![update("t.rs", vec![twin_hunk()])];
        let ni = ask(&dir, &ops).await;
        let text = ni.render("PATCH TEXT");
        let (first, rest) = text.split_once('\n').unwrap();
        assert!(
            first.starts_with("NOT APPLIED — 1 decision needed. Nothing was changed."),
            "{first}"
        );
        let v: serde_json::Value = serde_json::from_str(rest).unwrap();
        assert_eq!(v["status"], "needs_input");
        assert_eq!(v["applied"], false);
        let args = &v["retry"]["arguments"];
        assert_eq!(args["patch"], "PATCH TEXT");
        assert_eq!(args["state_token"], ni.state_token.as_str());
        // A candidate's `choice`, copied verbatim with the token, is a valid retry.
        let choice = v["decisions"][0]["candidates"][1]["choice"].clone();
        let sel = Selection::from_input(
            &json!({"state_token": args["state_token"], "choices": [choice]}),
        )
        .unwrap();
        let r = apply_patch(&ops, &dir, None, None, &sel).await;
        assert!(r.is_ok(), "{:?} {:?}", r.errors, r.needs_input);
        assert!(std::fs::read_to_string(dir.join("t.rs"))
            .unwrap()
            .ends_with("fn b() {\n    other();\n}\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
