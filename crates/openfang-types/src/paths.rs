//! Shared filesystem-path helpers for OpenFang's per-user tmp trees.
//!
//! These live in `openfang-types` because the dirs are **written by one
//! crate and read by another**: `openfang-channels` materializes inbound
//! attachment bytes into the file tmpdir, while `openfang-runtime` has to
//! grant its driver subprocesses read access to that exact path. Neither
//! crate depends on the other, so a single source of truth for the path has
//! to sit in the crate they share. Duplicating the resolution logic is how
//! you get a materializer and a read-guard that silently disagree.

use std::path::PathBuf;

/// Resolve the directory used for materializing inbound file attachments:
/// `$HOME/.openfang/tmp/files` (or `%USERPROFILE%\.openfang\tmp\files`).
///
/// Falls back to the OS temp dir when neither `$HOME` nor `%USERPROFILE%` is
/// set. `HOME` → `USERPROFILE` order matches the convention used by
/// `openfang_runtime::image_cache::image_tmp_dir` and `drivers/mod.rs`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_tmp_dir_ends_with_openfang_tmp_files() {
        let d = file_tmp_dir();
        // Either the HOME-derived path or the temp-dir fallback; both must be
        // absolute and identifiable as ours.
        assert!(d.is_absolute(), "expected absolute path, got {d:?}");
        let s = d.to_string_lossy();
        assert!(
            s.ends_with("openfang-files") || s.contains(".openfang"),
            "unexpected file tmp dir: {s}"
        );
    }
}
