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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::warn;

/// Most match locations quoted in one error message.
const MAX_LISTED: usize = 10;

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
}

impl PatchResult {
    /// Returns true if no errors occurred.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Summary string for tool output.
    pub fn summary(&self) -> String {
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
            let patched =
                apply_hunks(&original, hunks).map_err(|e| format!("patch {path}: {e}"))?;
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
pub async fn apply_patch(
    ops: &[PatchOp],
    workspace_root: &Path,
    file_policy: Option<&openfang_types::config::FilePolicy>,
    agent_id: Option<&str>,
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
        return result;
    }

    // ANAI-298: plan everything; report every failing op at once.
    let mut overlay = Overlay::new();
    let mut actions = Vec::new();
    for op in ops {
        match plan_op(op, workspace_root, file_policy, &mut overlay).await {
            Ok(action) => actions.push(action),
            Err(e) => result.errors.push(e),
        }
    }
    if !result.errors.is_empty() {
        return result;
    }

    for action in actions {
        commit(action, agent_id, &mut result).await;
    }
    result
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
fn apply_hunks(content: &str, hunks: &[Hunk]) -> Result<String, String> {
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
            locate(&lines, &origin, &pattern, cursor, hunk, n)?
        };

        let start = pos + hunk.context_before.len();
        let end = start + hunk.old_lines.len();
        let added = hunk.new_lines.len();
        lines.splice(start..end, hunk.new_lines.iter().cloned());
        ends.splice(start..end, std::iter::repeat_n(nl, added));
        origin.splice(start..end, std::iter::repeat_n(None, added));
        cursor = start + added;
    }

    let mut out = String::with_capacity(content.len() + 64);
    let last = lines.len();
    for (k, (line, end)) in lines.iter().zip(ends.iter()).enumerate() {
        out.push_str(line);
        if k + 1 < last || had_trailing_newline {
            out.push_str(if end.is_empty() { nl } else { end });
        }
    }
    Ok(out)
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
) -> Result<usize, String> {
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
                return Ok(*one);
            }
            _ => {
                return Err(format!(
                    "Hunk {n} ({label}) is ambiguous: its context and '-' lines match \
                     {} places{where_}, starting at {}. Add context lines that occur only \
                     at the intended place, or start the hunk with `@@ <text of a unique \
                     line above it>` (e.g. the enclosing fn signature).",
                    hits.len(),
                    list_lines(origin, &hits)
                ));
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

        let result = apply_patch(&ops, &dir, None, None).await;
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

        let result = apply_patch(&ops, &dir, None, None).await;
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

        let result = apply_patch(&ops, &dir, Some(&policy), None).await;
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
        let result = apply_patch(&ops, &dir, None, None).await;

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
        let result = apply_patch(&ops, &dir, None, None).await;
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
        let result = apply_patch(&ops, &dir, None, None).await;
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
        let result = apply_patch(&ops, &dir, None, None).await;
        assert!(!result.is_ok());
        assert!(!dir.join("n.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
