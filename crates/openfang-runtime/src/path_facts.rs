//! ANAI-190: gathering the path fact sheet.
//!
//! The pure half — what a fact *means*, and what combination of facts is
//! suppress-eligible — lives in [`openfang_types::path_facts`]. This is the
//! half that touches the box: `symlink_metadata`, a bounded `git` query, and a
//! `file_policy` tier lookup.
//!
//! Three invariants hold everywhere in this module:
//!
//! 1. **No file contents.** Metadata only. There is nothing here to leak.
//! 2. **No symlink traversal.** Every stat is `symlink_metadata`; a link is
//!    reported as a link with its target named, never resolved through.
//! 3. **Every failure is `Unknown`, and `Unknown` is never recoverable.** A
//!    timed-out git query, an unstattable path, an unresolvable argument — all
//!    of them subtract confidence rather than adding it.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use openfang_types::config::{FileAccessTier, FilePolicy};
use openfang_types::path_facts::{
    extract_path_tokens, GitFact, PathAuthority, PathExistence, PathFact, PathFactSheet,
    MAX_PATH_FACTS,
};

/// Total wall-clock budget for *all* git queries behind one gate decision.
///
/// Deliberately far below the judge's own 10s budget. The corpus that motivated
/// ANAI-190 also showed plain `git status --porcelain` taking 4–10s on this box
/// under fleet load, so this cap will sometimes fire. That is the intended
/// behaviour: a git axis that degrades to `Unknown` costs us a suppression,
/// while a git axis that blocks costs us the latency budget the whole feature
/// was supposed to protect. The rate at which it fires is one of the numbers
/// commit 1 exists to measure.
const GIT_BUDGET: Duration = Duration::from_millis(300);

/// Build the sheet for one gated command.
///
/// `command` is the comment-stripped form and `inner` the commands lifted out
/// of a shell wrapper — a path named inside `bash -c '...'` is named just as
/// much as one at the top level.
pub async fn gather(
    command: &str,
    inner: &[String],
    workspace_root: Option<&Path>,
    file_policy: Option<&FilePolicy>,
) -> PathFactSheet {
    let (tokens, unresolved) = extract_path_tokens(command, inner);
    let truncated = tokens.len() > MAX_PATH_FACTS;
    let tokens: Vec<String> = tokens.into_iter().take(MAX_PATH_FACTS).collect();

    let canon_ws = workspace_root.map(|r| r.canonicalize().unwrap_or_else(|_| r.to_path_buf()));

    let mut facts: Vec<PathFact> = tokens
        .iter()
        .map(|raw| stat_fact(raw, canon_ws.as_deref(), file_policy))
        .collect();

    // Git is the only part that can block, so it runs once, batched by repo,
    // under a single budget. Anything it does not answer stays `Unknown`.
    annotate_git(&mut facts, canon_ws.as_deref()).await;

    PathFactSheet {
        facts,
        truncated,
        unresolved,
    }
}

/// Everything computable without spawning anything: resolution, stat,
/// containment, authority.
fn stat_fact(raw: &str, canon_ws: Option<&Path>, file_policy: Option<&FilePolicy>) -> PathFact {
    let resolved = resolve(raw, canon_ws);

    let mut fact = PathFact {
        raw: raw.to_string(),
        resolved: resolved.as_ref().map(|p| p.display().to_string()),
        existence: PathExistence::Unknown,
        size_bytes: None,
        symlink_target: None,
        mtime_secs_ago: None,
        git: GitFact::Unknown,
        inside_workspace: false,
        authority: PathAuthority::NoPolicy,
    };

    let Some(path) = resolved else {
        return fact;
    };

    fact.inside_workspace = canon_ws.is_some_and(|ws| is_within(&path, ws));
    fact.authority = authority_for(&path, canon_ws, file_policy);

    // `symlink_metadata`, never `metadata`: `./harmless -> /etc/passwd` must
    // report as a symlink, not as whatever it points at.
    match std::fs::symlink_metadata(&path) {
        Ok(meta) => {
            let ft = meta.file_type();
            fact.existence = if ft.is_symlink() {
                fact.symlink_target = std::fs::read_link(&path)
                    .ok()
                    .map(|t| t.display().to_string());
                PathExistence::Symlink
            } else if ft.is_dir() {
                PathExistence::Dir
            } else {
                fact.size_bytes = Some(meta.len());
                PathExistence::File
            };
            fact.mtime_secs_ago = meta
                .modified()
                .ok()
                .and_then(|m| m.elapsed().ok())
                .map(|d| d.as_secs());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fact.existence = PathExistence::Missing;
        }
        // Permissions, a dead mount, anything else. Not "fine".
        Err(_) => fact.existence = PathExistence::Unknown,
    }

    fact
}

/// Turn an argument into an absolute path without following the leaf link.
///
/// A leading `~` expands against `$HOME`; a relative path joins the workspace
/// root, because that is `shell_exec`'s working directory. `..` is folded
/// lexically rather than by `canonicalize`, so a path whose leaf is a symlink —
/// or does not exist yet — still resolves.
fn resolve(raw: &str, canon_ws: Option<&Path>) -> Option<PathBuf> {
    let expanded: PathBuf = if raw == "~" {
        home_dir()?
    } else if let Some(rest) = raw.strip_prefix("~/") {
        home_dir()?.join(rest)
    } else {
        PathBuf::from(raw)
    };

    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        canon_ws?.join(expanded)
    };

    Some(lexical_normalize(&absolute))
}

/// Cross-platform home directory.
///
/// The runtime does not depend on the `dirs` crate — `openfang-types` does, and
/// `openfang-runtime` resolves `$HOME` from the environment everywhere else
/// (see `drivers::qwen_code`). Matching the local idiom rather than adding a
/// dependency for two call sites.
fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Fold `.` and `..` without touching the filesystem.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Containment with a component boundary.
///
/// `/ws-evil` is not inside `/ws`, and a plain `starts_with` on strings would
/// say otherwise. `Path::starts_with` is component-wise, which is exactly the
/// semantics we want.
fn is_within(path: &Path, root: &Path) -> bool {
    path.starts_with(root)
}

/// What tier `file_policy` would have granted this agent for this path.
///
/// The guard is [`FilePolicy::is_active`], not the raw `enabled` flag, and the
/// distinction is the single most dangerous detail in ANAI-190. `tier_for`
/// returns `FileAccessTier::Write` **for every path on the filesystem** when
/// the policy is inert:
///
/// ```text
/// let own = if self.enabled { self.own_tier_for(..) } else { FileAccessTier::Write };
/// ```
///
/// Calling it unguarded would report an agent with no policy as maximally
/// authorized everywhere — the inverse of fail-closed, and precisely the shape
/// of bug that hands a deterministic fast-path permission to delete anything on
/// the box. `is_active()` also accounts for the global `[file_policy]` floor,
/// so a manifest with `enabled = false` beneath an enabled global still
/// resolves correctly. There is no global block today; gating on the function
/// rather than the flag means adding one later does not silently change this.
fn authority_for(
    path: &Path,
    canon_ws: Option<&Path>,
    file_policy: Option<&FilePolicy>,
) -> PathAuthority {
    let Some(policy) = file_policy else {
        return PathAuthority::NoPolicy;
    };
    if !policy.is_active() {
        return PathAuthority::NoPolicy;
    }
    // Rule paths may be workspace-relative, so a policy cannot be evaluated
    // without a root to resolve them against.
    let Some(ws) = canon_ws else {
        return PathAuthority::NoPolicy;
    };
    match policy.tier_for(path, ws) {
        FileAccessTier::Write => PathAuthority::Write,
        FileAccessTier::Read => PathAuthority::Read,
        FileAccessTier::Prompt => PathAuthority::Prompt,
        FileAccessTier::Deny => PathAuthority::Deny,
    }
}

/// Fill in [`GitFact`] for every fact that has a repository above it.
///
/// Repository roots are derived **per path**, not per box: each path walks up
/// from its own directory looking for `.git`, so one command can touch two
/// repositories, or none. Paths sharing a root are batched into one pair of
/// queries. Paths with no root above them get `NoRepo`, which is a real answer
/// and not a synonym for `Untracked`.
async fn annotate_git(facts: &mut [PathFact], canon_ws: Option<&Path>) {
    let mut by_root: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    let mut no_repo: Vec<usize> = Vec::new();
    for (i, fact) in facts.iter().enumerate() {
        let Some(resolved) = fact.resolved.as_ref().map(PathBuf::from) else {
            continue;
        };
        match discover_repo_root(&resolved, canon_ws) {
            Some(root) => by_root.entry(root).or_default().push(i),
            None => no_repo.push(i),
        }
    }
    for i in no_repo {
        facts[i].git = GitFact::NoRepo;
    }
    if by_root.is_empty() {
        return;
    }

    // One budget for the whole phase, not per repo: two slow repositories must
    // not cost twice the latency ceiling.
    let _ = tokio::time::timeout(GIT_BUDGET, async {
        for (root, indices) in &by_root {
            let paths: Vec<String> = indices
                .iter()
                .filter_map(|i| facts[*i].resolved.clone())
                .collect();
            let Some(status) = query_repo(root, &paths).await else {
                continue;
            };
            for i in indices {
                let Some(resolved) = facts[*i].resolved.clone() else {
                    continue;
                };
                let Ok(rel) = Path::new(&resolved).strip_prefix(root) else {
                    continue;
                };
                facts[*i].git = status.classify(&rel.display().to_string());
            }
        }
    })
    .await;
}

/// Walk up from a path looking for `.git`, stopping at a boundary.
///
/// The boundary matters: without it the walk continues past the workspace and
/// into `$HOME`, and if `$HOME` ever becomes a repository every agent-local
/// scratch file would inherit *that* repository's cleanliness. We stop at the
/// workspace root (inclusive) and at `$HOME` (exclusive) — and, failing both,
/// at the filesystem root.
fn discover_repo_root(path: &Path, canon_ws: Option<&Path>) -> Option<PathBuf> {
    let home = home_dir();
    let mut cursor = if path.is_dir() {
        Some(path)
    } else {
        path.parent()
    };
    while let Some(dir) = cursor {
        if home.as_deref() == Some(dir) {
            return None;
        }
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        // The workspace root is a hard stop, inclusive: we check it for `.git`
        // and then refuse to look above it.
        if canon_ws == Some(dir) {
            return None;
        }
        cursor = dir.parent();
    }
    None
}

/// The two sets one `git` pair of calls yields, keyed by repo-relative path.
struct RepoStatus {
    tracked: Vec<String>,
    dirty: Vec<String>,
    ignored: Vec<String>,
}

impl RepoStatus {
    fn classify(&self, rel: &str) -> GitFact {
        let rel = rel.to_string();
        if self.ignored.contains(&rel) {
            return GitFact::Ignored;
        }
        if self.tracked.contains(&rel) {
            if self.dirty.contains(&rel) {
                return GitFact::TrackedDirty;
            }
            return GitFact::TrackedClean;
        }
        GitFact::Untracked
    }
}

/// One repository, two bounded reads: the index, then the working tree.
///
/// `git ls-files` alone cannot distinguish tracked-and-clean from
/// absent-and-untracked, and `git status` alone cannot distinguish
/// tracked-and-clean from not-in-this-repo — both produce empty output for
/// cases we must tell apart. Hence the pair.
///
/// `--ignored=matching` is not decoration. Ignored files are the class that
/// holds essentially every `.env`, key and token on this box, and they must
/// never read as ordinary untracked scratch.
async fn query_repo(root: &Path, paths: &[String]) -> Option<RepoStatus> {
    let tracked = run_git(root, &["ls-files", "--"], paths)
        .await?
        .lines()
        .map(str::to_string)
        .collect();

    let status_raw = run_git(
        root,
        &["status", "--porcelain", "--ignored=matching", "--"],
        paths,
    )
    .await?;

    let mut dirty = Vec::new();
    let mut ignored = Vec::new();
    for line in status_raw.lines() {
        if line.len() < 4 {
            continue;
        }
        let (code, rest) = line.split_at(2);
        let name = rest.trim_start().trim_matches('"').to_string();
        if code == "!!" {
            ignored.push(name);
        } else {
            dirty.push(name);
        }
    }

    Some(RepoStatus {
        tracked,
        dirty,
        ignored,
    })
}

/// Run one `git` read, with the repository fixed by `-C` and the paths passed
/// after `--`.
///
/// `-c core.quotepath=false` keeps non-ASCII names from coming back
/// backslash-escaped, which would silently fail to match the path we asked
/// about and downgrade a real answer to `Untracked`. Every one of these is a
/// read; nothing here can mutate a repository.
async fn run_git(root: &Path, args: &[&str], paths: &[String]) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .arg("-c")
        .arg("core.quotepath=false")
        .arg("-C")
        .arg(root)
        .args(args)
        .args(paths)
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::config::{FilePolicy, FileRule};

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openfang-anai190-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    /// Ben's motivating case, end to end against a real directory: a scratch
    /// file in an agent workspace, no repository anywhere above it.
    #[tokio::test]
    async fn workspace_scratch_file_gathers_as_suppress_eligible() {
        let ws = tempdir("scratch");
        std::fs::write(ws.join("decision-scratch.txt"), "notes").unwrap();
        let policy = FilePolicy::new(
            true,
            FileAccessTier::Deny,
            vec![FileRule {
                path: ws.display().to_string(),
                tier: FileAccessTier::Write,
            }],
        );

        let sheet = gather("rm ./decision-scratch.txt", &[], Some(&ws), Some(&policy)).await;

        assert_eq!(sheet.facts.len(), 1, "{:?}", sheet.facts);
        let f = &sheet.facts[0];
        assert_eq!(f.existence, PathExistence::File);
        assert_eq!(f.git, GitFact::NoRepo);
        assert!(f.inside_workspace);
        assert_eq!(f.authority, PathAuthority::Write);
        assert!(sheet.suppress_eligible());
    }

    /// The `config.rs:1109` trap. An inert policy must report `no-policy`, not
    /// the `Write` that `tier_for` would hand back for every path on the disk.
    #[tokio::test]
    async fn inert_file_policy_never_reports_write() {
        let ws = tempdir("inert");
        std::fs::write(ws.join("a.txt"), "x").unwrap();
        let inert = FilePolicy::default();
        assert!(!inert.is_active());

        let sheet = gather("rm ./a.txt", &[], Some(&ws), Some(&inert)).await;
        assert_eq!(sheet.facts[0].authority, PathAuthority::NoPolicy);
        assert!(!sheet.suppress_eligible());

        // And the same when no policy is supplied at all.
        let sheet = gather("rm ./a.txt", &[], Some(&ws), None).await;
        assert_eq!(sheet.facts[0].authority, PathAuthority::NoPolicy);
        assert!(!sheet.suppress_eligible());
    }

    /// `./harmless -> /etc/hosts` must report as a link, with the target named,
    /// and must not inherit the target's properties.
    #[cfg(unix)]
    #[tokio::test]
    async fn symlinks_are_reported_not_followed() {
        let ws = tempdir("symlink");
        std::os::unix::fs::symlink("/etc/hosts", ws.join("harmless")).unwrap();

        let sheet = gather("rm ./harmless", &[], Some(&ws), None).await;
        let f = &sheet.facts[0];
        assert_eq!(f.existence, PathExistence::Symlink);
        assert_eq!(f.symlink_target.as_deref(), Some("/etc/hosts"));
        assert!(!f.recoverable());
    }

    #[tokio::test]
    async fn a_missing_path_is_missing_not_unknown() {
        let ws = tempdir("missing");
        let sheet = gather("rm ./nope.txt", &[], Some(&ws), None).await;
        assert_eq!(sheet.facts[0].existence, PathExistence::Missing);
    }

    /// The walk-up must not escape the workspace and adopt a parent
    /// repository's cleanliness.
    #[tokio::test]
    async fn repo_discovery_stops_at_the_workspace_root() {
        let outer = tempdir("boundary");
        std::fs::create_dir_all(outer.join(".git")).unwrap();
        let ws = outer.join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("a.txt"), "x").unwrap();

        assert_eq!(discover_repo_root(&ws.join("a.txt"), Some(&ws)), None);
        // ...and without the boundary it would have found the outer repo.
        assert_eq!(
            discover_repo_root(&ws.join("a.txt"), None),
            Some(outer.clone())
        );
    }

    /// A glob is not a path. The sheet must say it was blind rather than
    /// quietly reporting on the paths it did manage to resolve.
    #[tokio::test]
    async fn globs_make_the_sheet_ineligible() {
        let ws = tempdir("glob");
        std::fs::write(ws.join("a.txt"), "x").unwrap();
        let policy = FilePolicy::new(
            true,
            FileAccessTier::Write,
            vec![FileRule {
                path: ws.display().to_string(),
                tier: FileAccessTier::Write,
            }],
        );
        let sheet = gather("rm ./a.txt ./build/*.o", &[], Some(&ws), Some(&policy)).await;
        assert!(sheet.unresolved);
        assert!(!sheet.suppress_eligible());
    }

    /// The git axis is the only part of this module that spawns anything, and
    /// its three answers are distinguished by parsing two command outputs. That
    /// is the highest-risk code in ANAI-190, so it gets a real repository
    /// rather than a hand-built fixture: a mis-parse that silently classified
    /// everything as `Untracked` would look exactly like a working feature
    /// while handing the fast-path a wrong answer on tracked files.
    #[tokio::test]
    async fn git_classifies_clean_dirty_and_ignored_against_a_real_repo() {
        let repo = tempdir("gitaxis");
        if !git(&repo, &["init", "--quiet"]).await {
            eprintln!("git unavailable; skipping");
            return;
        }
        git(&repo, &["config", "user.email", "t@example.com"]).await;
        git(&repo, &["config", "user.name", "t"]).await;

        std::fs::write(repo.join(".gitignore"), ".env\n").unwrap();
        std::fs::write(repo.join("clean.txt"), "v1").unwrap();
        std::fs::write(repo.join("dirty.txt"), "v1").unwrap();
        std::fs::write(repo.join(".env"), "SECRET=1").unwrap();
        assert!(git(&repo, &["add", ".gitignore", "clean.txt", "dirty.txt"]).await);
        assert!(git(&repo, &["commit", "--quiet", "-m", "seed"]).await);

        // ...and now diverge one of them from the index.
        std::fs::write(repo.join("dirty.txt"), "v2").unwrap();
        std::fs::write(repo.join("new.txt"), "fresh").unwrap();

        let sheet = gather(
            "rm ./clean.txt ./dirty.txt ./new.txt ./.env",
            &[],
            Some(&repo),
            None,
        )
        .await;

        let of = |name: &str| {
            sheet
                .facts
                .iter()
                .find(|f| f.raw.ends_with(name))
                .unwrap_or_else(|| panic!("{name} missing from {:?}", sheet.facts))
                .git
        };
        assert_eq!(of("clean.txt"), GitFact::TrackedClean);
        assert_eq!(of("dirty.txt"), GitFact::TrackedDirty);
        assert_eq!(of("new.txt"), GitFact::Untracked);
        // The class that holds every `.env`, key and token on the box.
        assert_eq!(of(".env"), GitFact::Ignored);
    }

    async fn git(repo: &Path, args: &[&str]) -> bool {
        tokio::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .await
            .is_ok_and(|o| o.status.success())
    }

    #[test]
    fn parent_traversal_is_folded_lexically() {
        let got = resolve("../etc/passwd", Some(Path::new("/ws/sub")));
        assert_eq!(got, Some(PathBuf::from("/ws/etc/passwd")));
    }

    #[test]
    fn a_sibling_directory_is_not_inside_the_workspace() {
        assert!(!is_within(Path::new("/ws-evil/a.txt"), Path::new("/ws")));
        assert!(is_within(Path::new("/ws/a.txt"), Path::new("/ws")));
    }
}
