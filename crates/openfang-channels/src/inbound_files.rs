//! Content-addressed tmpfile cache for **inbound** channel file attachments
//! (ANAI-137).
//!
//! # Why this exists
//!
//! Before ANAI-137 the bridge treated `ChannelContent::Image` and
//! `ChannelContent::File` asymmetrically:
//!
//! - `Image` bytes were downloaded by the bridge, base64-encoded into a
//!   `ContentBlock::Image`, and materialized to a real local path by the
//!   claude-code driver (see `openfang_runtime::image_cache`). Agents could
//!   read them.
//! - `File` was rendered as a bare text descriptor —
//!   `[User sent a file (name): <cdn url>]` — with **no download and no
//!   path**. Agents had no way to reach the bytes: the signed CDN URL
//!   contains `&`, which the shell metacharacter floor blocks; wrapping a
//!   fetch in a script trips the approval gate; and `web_fetch` re-serializes
//!   through the model's context, so it is text-only and lossy.
//!
//! Net effect: a user could hand an agent a PDF, a zip, or an oversize image
//! and the agent simply could not open it. This module closes that gap by
//! giving `File` what `Image` already had — bytes on disk at a stable path.
//!
//! # Division of labour
//!
//! This module is deliberately **pure and synchronous**: hashing, filename
//! derivation, atomic publish, TTL sweep. It performs no network I/O. The
//! *fetching* stays in the adapter (e.g. `discord::Fetcher`, which already
//! runs the SSRF preflight, redirect cap, and byte ceiling), so we inherit
//! that hardening instead of re-implementing it — and so tests can inject a
//! permissive fetcher without touching this code.
//!
//! # Properties (mirrors `openfang_runtime::image_cache` deliberately)
//!
//! - **Idempotent.** Filename is the first 64 bits of SHA-256(bytes), so the
//!   same attachment re-sent lands on the same path instead of a second copy.
//! - **Atomic publish.** Bytes go to a unique sibling tmpfile, then
//!   `rename(2)` into place; readers never see a torn file.
//! - **Time-bounded.** A best-effort sweep on first use (per process) removes
//!   files older than [`FILE_TMP_TTL_SECS`].
//! - **Extension-preserving.** Unlike the image cache (which derives the
//!   extension from the MIME type), we keep the *source* extension, because
//!   downstream tools dispatch on it — `file_convert` needs to see `.md`, not
//!   `.bin`.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Once;
use tracing::{debug, info, warn};

/// TTL for materialized inbound attachment tmpfiles (24 hours), matching
/// [`openfang_runtime::image_cache::IMAGE_TMP_TTL_SECS`]. Files older than
/// this are swept on first use per process.
pub const FILE_TMP_TTL_SECS: u64 = 24 * 60 * 60;

/// Default per-attachment size cap: 25 MiB.
///
/// Chosen to match Discord's own non-Nitro upload limit, so under normal
/// operation it cannot be exceeded from that channel at all — the cap is a
/// backstop against a hostile or misreporting source, not a routine limit.
/// Operators override it via `channels.discord.max_upload_bytes`.
pub const DEFAULT_MAX_UPLOAD_BYTES: u64 = 25 * 1024 * 1024;

/// Fallback extension when the source filename carries none.
const FALLBACK_EXT: &str = "bin";

/// Longest source extension we will echo into the tmpfile name. Guards
/// against a hostile "filename" whose extension is 4 KB of junk.
const MAX_EXT_LEN: usize = 16;

/// One-shot guard so the TTL sweep only fires once per process.
static FILE_TMP_SWEEP_ONCE: Once = Once::new();

/// Resolve the directory used for materializing inbound file attachments:
/// `$HOME/.openfang/tmp/files` (or `%USERPROFILE%\.openfang\tmp\files`).
///
/// Falls back to the OS temp dir when neither `$HOME` nor `%USERPROFILE%` is
/// set. `HOME` → `USERPROFILE` order matches the convention used by
/// `image_cache::image_tmp_dir` and `drivers/mod.rs`.
///
/// Note: this path already existed on deployed boxes as a **vestigial** dir —
/// something stubbed it long ago and never wired it up (grep for `tmp/files`
/// across the tree pre-ANAI-137 returned zero hits). This module is what
/// finally makes it live.
pub fn file_tmp_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let mut p = PathBuf::from(home);
        p.push(".openfang");
        p.push("tmp");
        p.push("files");
        p
    } else {
        let mut p = std::env::temp_dir();
        p.push("openfang-files");
        p
    }
}

/// Write `bytes` to a content-addressed file under `dir` and return its path.
///
/// The filename is `<hash16>__<sanitized-stem>.<ext>`, where `hash16` is the
/// first 64 bits of SHA-256(bytes) in hex and `<ext>` is preserved from
/// `original_name` (lowercased, `bin` if absent) so extension-dispatching
/// consumers like `file_convert` keep working. When the stem sanitizes to
/// nothing, the name degrades to `<hash16>.<ext>`.
///
/// Idempotent: if a file with the same content hash and extension already
/// exists, its mtime is refreshed and the existing path returned without
/// rewriting. Returns `None` on I/O failure — every caller must treat that as
/// "fall back to the URL-only descriptor", never as a hard error, so a full
/// disk degrades to the pre-ANAI-137 behaviour instead of dropping the
/// user's message.
pub fn materialize_bytes(bytes: &[u8], original_name: &str, dir: &Path) -> Option<PathBuf> {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hash = hasher.finalize();
    let hex: String = hash.iter().take(8).map(|b| format!("{:02x}", b)).collect();
    let ext = ext_for_filename(original_name);

    // Cache hit: same bytes, same extension → reuse whatever name landed
    // first. Refresh the mtime so the TTL sweep does not GC a file that a
    // live conversation is still referencing.
    if let Some(existing) = find_existing_for_hash(dir, &hex, &ext) {
        if let Err(e) = touch_mtime(&existing) {
            debug!(path = ?existing, error = %e, "failed to refresh inbound file tmpfile mtime");
        }
        return Some(existing);
    }

    if let Err(e) = std::fs::create_dir_all(dir) {
        warn!(dir = ?dir, error = %e, "failed to create openfang inbound file tmp dir");
        return None;
    }

    let filename = match sanitize_stem(original_name) {
        Some(stem) => format!("{hex}__{stem}.{ext}"),
        None => format!("{hex}.{ext}"),
    };
    let path = dir.join(filename);

    // Atomic publish: unique tmp sibling, then rename(2) into place. Losing a
    // race is harmless — the contents are identical by construction, and
    // POSIX rename replaces.
    let tmp_path = dir.join(format!(
        "{hex}.{pid}.{nanos}.tmp",
        pid = std::process::id(),
        nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    if let Err(e) = std::fs::write(&tmp_path, bytes) {
        warn!(path = ?tmp_path, error = %e, "failed to write openfang inbound file tmpfile");
        return None;
    }
    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        warn!(from = ?tmp_path, to = ?path, error = %e, "failed to rename openfang inbound file tmpfile into place");
        let _ = std::fs::remove_file(&tmp_path);
        return None;
    }
    Some(path)
}

/// Extract a lowercase extension from a source filename.
///
/// Returns [`FALLBACK_EXT`] when there is no extension, when it is empty or
/// implausibly long ([`MAX_EXT_LEN`]), when it is not purely ASCII
/// alphanumeric, or when the name is dotfile-shaped (`.env` → no stem, so the
/// leading dot is *not* an extension).
fn ext_for_filename(name: &str) -> String {
    let leaf = leaf_of(name);
    match leaf.rsplit_once('.') {
        Some((stem, ext))
            if !stem.is_empty()
                && !ext.is_empty()
                && ext.len() <= MAX_EXT_LEN
                && ext.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            ext.to_ascii_lowercase()
        }
        _ => FALLBACK_EXT.to_string(),
    }
}

/// Sanitize the filename stem for embedding in a tmpfile name: lowercase
/// ASCII, anything outside `[a-z0-9.-]` becomes `_`, runs of `_` collapse,
/// leading/trailing `_`/`.` are trimmed (no hidden files, no `..`), capped at
/// 60 chars. Returns `None` if nothing usable survives.
///
/// Kept byte-compatible in spirit with
/// `openfang_runtime::image_cache::sanitize_for_filename` so operators see one
/// naming convention across `tmp/images/` and `tmp/files/`.
fn sanitize_stem(name: &str) -> Option<String> {
    let leaf = leaf_of(name);
    let stem = match leaf.rsplit_once('.') {
        Some((s, _)) if !s.is_empty() => s,
        _ => leaf,
    };
    let mut out = String::with_capacity(stem.len());
    let mut last_underscore = false;
    for c in stem.chars() {
        let lc = c.to_ascii_lowercase();
        if lc.is_ascii_alphanumeric() || matches!(lc, '.' | '-') {
            out.push(lc);
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    let trimmed = out.trim_matches(|c: char| c == '_' || c == '.').to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(60).collect())
}

/// Strip any path components. Source filenames should never contain
/// separators, but a hostile or malformed platform payload could try
/// `../../etc/passwd`; taking the last segment defuses traversal before the
/// name is ever joined onto a directory.
fn leaf_of(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// Find a previously-materialized tmpfile carrying this content hash and
/// extension, regardless of the human-readable suffix. Best-effort: read
/// errors yield `None` and the caller falls through to a fresh write.
fn find_existing_for_hash(dir: &Path, hex: &str, ext: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let dot_ext = format!(".{ext}");
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(&dot_ext) {
            continue;
        }
        let stem = name.trim_end_matches(&dot_ext);
        if stem == hex || stem.starts_with(&format!("{hex}__")) {
            return Some(path);
        }
    }
    None
}

/// Refresh `path`'s mtime to now so it survives the next TTL sweep.
fn touch_mtime(path: &Path) -> std::io::Result<()> {
    let f = std::fs::OpenOptions::new().write(true).open(path)?;
    f.set_modified(std::time::SystemTime::now())
}

/// Delete inbound-file tmpfiles older than [`FILE_TMP_TTL_SECS`].
/// Best-effort: errors are logged at debug and the sweep moves on.
pub fn sweep_old_file_tmpfiles(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            debug!(dir = ?dir, error = %e, "inbound file tmp sweep: read_dir failed (likely missing dir, fine)");
            return;
        }
    };
    let now = std::time::SystemTime::now();
    let ttl = std::time::Duration::from_secs(FILE_TMP_TTL_SECS);
    let mut removed = 0u32;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let Ok(modified) = meta.modified() else {
            continue;
        };
        if let Ok(age) = now.duration_since(modified) {
            if age > ttl {
                if let Err(e) = std::fs::remove_file(&path) {
                    debug!(path = ?path, error = %e, "inbound file tmp sweep: remove failed");
                } else {
                    removed += 1;
                }
            }
        }
    }
    if removed > 0 {
        info!(removed, "swept stale openfang inbound file tmpfiles");
    }
}

/// Spawn the once-per-process TTL sweep on a background thread. Safe to call
/// from every adapter start — the [`Once`] guard means only the first call
/// does work.
pub fn spawn_sweep_once() {
    FILE_TMP_SWEEP_ONCE.call_once(|| {
        let dir = file_tmp_dir();
        std::thread::spawn(move || sweep_old_file_tmpfiles(&dir));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn materialize_is_content_addressed_and_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        let a = materialize_bytes(b"hello pdf bytes", "report.pdf", dir).unwrap();
        let b = materialize_bytes(b"hello pdf bytes", "report.pdf", dir).unwrap();
        assert_eq!(a, b, "same bytes must reuse the same path");
        assert_eq!(std::fs::read(&a).unwrap(), b"hello pdf bytes");

        let c = materialize_bytes(b"different bytes", "report.pdf", dir).unwrap();
        assert_ne!(a, c, "different bytes must land on different paths");

        // Exactly two published files (plus no leftover .tmp siblings).
        let published: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(published.len(), 2, "unexpected dir contents: {published:?}");
        assert!(
            published.iter().all(|n| !n.ends_with(".tmp")),
            "atomic publish leaked a tmpfile: {published:?}"
        );
    }

    /// The extension is what `file_convert` dispatches on, so preserving it
    /// from the source name is load-bearing, not cosmetic.
    #[test]
    fn preserves_source_extension_and_names_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = materialize_bytes(b"x", "Alchemists & Warlocks.MD", tmp.path()).unwrap();
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.ends_with(".md"), "expected .md, got {name}");
        assert!(
            name.contains("__alchemists_warlocks.md"),
            "expected sanitized stem, got {name}"
        );
        let hex: String = name.chars().take(16).collect();
        assert!(
            hex.chars().all(|c| c.is_ascii_hexdigit()),
            "expected 16 hex chars of content hash, got {name}"
        );
    }

    #[test]
    fn extension_edge_cases() {
        assert_eq!(ext_for_filename("notes.md"), "md");
        assert_eq!(ext_for_filename("ARCHIVE.TAR.GZ"), "gz");
        assert_eq!(ext_for_filename("noext"), FALLBACK_EXT);
        assert_eq!(ext_for_filename("trailing."), FALLBACK_EXT);
        // Dotfile: no stem before the dot, so `env` is not an extension.
        assert_eq!(ext_for_filename(".env"), FALLBACK_EXT);
        // Implausibly long / non-alphanumeric extensions are refused.
        assert_eq!(ext_for_filename("x.superlongextension"), FALLBACK_EXT);
        assert_eq!(ext_for_filename("x.p df"), FALLBACK_EXT);
    }

    /// A hostile filename must not be able to escape the tmp dir.
    #[test]
    fn rejects_path_traversal_in_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let path = materialize_bytes(b"pwn", "../../../../etc/passwd", dir).unwrap();
        assert_eq!(
            path.parent().unwrap(),
            dir,
            "materialized file escaped the tmp dir: {path:?}"
        );
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(!name.contains(".."), "traversal survived sanitize: {name}");
        assert!(!name.contains('/'), "separator survived sanitize: {name}");
    }

    #[test]
    fn unnamed_attachment_degrades_to_hash_only() {
        let tmp = tempfile::tempdir().unwrap();
        let path = materialize_bytes(b"y", "___", tmp.path()).unwrap();
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(!name.contains("__"), "expected hash-only name, got {name}");
        assert!(name.ends_with(".bin"), "expected .bin, got {name}");
    }

    #[test]
    fn sweep_removes_stale_but_cache_hit_refreshes_mtime() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let path = materialize_bytes(b"stale candidate", "a.txt", dir).unwrap();

        let backdate = |p: &Path| {
            let stale = SystemTime::now() - Duration::from_secs(FILE_TMP_TTL_SECS + 3600);
            let f = std::fs::OpenOptions::new().write(true).open(p).unwrap();
            f.set_modified(stale).unwrap();
        };

        // Cache hit must refresh the mtime, saving the file from the sweep —
        // otherwise a conversation longer than the TTL loses its bytes
        // mid-thread.
        backdate(&path);
        let again = materialize_bytes(b"stale candidate", "a.txt", dir).unwrap();
        assert_eq!(path, again);
        sweep_old_file_tmpfiles(dir);
        assert!(path.exists(), "refreshed tmpfile should survive the sweep");

        // Without a refresh, it must actually be collected.
        backdate(&path);
        sweep_old_file_tmpfiles(dir);
        assert!(!path.exists(), "stale tmpfile should have been swept");
    }
}
