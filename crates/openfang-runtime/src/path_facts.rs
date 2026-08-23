//! ANAI-190: gathering the path fact sheet.
//!
//! The pure half — what a fact *means*, and what combination of facts is
//! suppress-eligible — lives in [`openfang_types::path_facts`]. This is the
//! half that touches the box: `symlink_metadata`, a bounded `git` query, and a
//! `file_policy` tier lookup.
//!
//! Three invariants hold everywhere in this module:
//!
//! 1. **Metadata by default.** The path facts themselves are stat, git and a
//!    tier lookup — no bytes. The single exception is ANAI-206's script body
//!    (see [`read_script_body`]), which reads exactly one file: the one the
//!    command itself is about to execute, and only when it is inside the
//!    requesting agent's own `file_policy` reach. The judge never names a path.
//! 2. **No symlink traversal.** Every stat is `symlink_metadata`; a link is
//!    reported as a link with its target named, never resolved through. A
//!    symlinked script is `Symlink`, never `File`, so it is never opened.
//!    ANAI-206 F8: that held at `stat` time and not at `open` time, so the
//!    single read is now `O_NOFOLLOW` and validates the handle it holds rather
//!    than re-resolving the name. The older TOCTOU note in this module — that
//!    stale facts are free because shadow mode acts on nothing — stopped being
//!    true the moment item 1 started reading bytes: the bytes reach the model
//!    provider and the audit chain whether or not the verdict is enforced.
//! 3. **Every failure is `Unknown`, and `Unknown` is never recoverable.** A
//!    timed-out git query, an unstattable path, an unresolvable argument — all
//!    of them subtract confidence rather than adding it.

use std::collections::BTreeMap;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use openfang_types::config::{FileAccessTier, FilePolicy};
use openfang_types::path_facts::{
    extract_body_path_tokens, extract_path_tokens, redact_secrets, script_body_target, GitFact,
    PathAuthority, PathExistence, PathFact, PathFactSheet, ScriptBody, ScriptBodyStatus,
    MAX_BODY_PATH_FACTS, MAX_PATH_FACTS, MAX_SCRIPT_BODY_BYTES,
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

    // ANAI-206 item 1. The one place in the gate that reads file bytes, and it
    // runs last on purpose: every guardrail it enforces is a fact the two
    // phases above computed.
    let mut script_body = script_body_target(command, inner)
        .map(|raw| read_script_body(&raw, &facts, canon_ws.as_deref(), file_policy));

    // ANAI-206 item 3, second phase: stat the paths the body named.
    //
    // A second git pass rather than one, because the body cannot be read until
    // the script's own facts exist and the two phases therefore cannot be
    // merged. It carries its own budget for the same reason the first one does:
    // a slow repository costs a `GitFact::Unknown`, never the latency ceiling.
    if let Some((body, tokens)) = script_body.as_mut() {
        body.body_facts = tokens
            .iter()
            .map(|raw| stat_fact(raw, canon_ws.as_deref(), file_policy))
            .collect();
        annotate_git(&mut body.body_facts, canon_ws.as_deref()).await;
    }
    let script_body = script_body.map(|(body, _)| body);

    PathFactSheet {
        facts,
        truncated,
        unresolved,
        script_body,
    }
}

/// Read the single script a command executes, or record why we would not.
///
/// Ordering is the security property. Reach is checked before anything is
/// touched, so a path outside the requesting agent's `file_policy` never
/// reaches an `open` — the gate must not become a privilege-escalation oracle
/// for an agent that can name a path in a command. Every later check is a
/// refusal too, and every refusal is recorded rather than dropped.
///
/// `raw` is [`script_body_target`]'s answer, so it is always a token the
/// command itself wrote and always one the sheet already stat'd — the judge
/// never chooses a path.
fn read_script_body(
    raw: &str,
    facts: &[PathFact],
    canon_ws: Option<&Path>,
    file_policy: Option<&FilePolicy>,
) -> (ScriptBody, Vec<String>) {
    let mut body = ScriptBody {
        raw: raw.to_string(),
        resolved: None,
        status: ScriptBodyStatus::Unreadable,
        content: None,
        redactions: 0,
        git: GitFact::Unknown,
        body_facts: Vec::new(),
        body_truncated: false,
        body_unresolved: false,
        writes_control_plane: false,
        destroys_substrate: false,
        writes_gatekeeper_policy: false,
        writes_agent_config: false,
        writes_runtime_config: false,
    };

    // No fact means the token was dropped by `MAX_PATH_FACTS` truncation or
    // failed to resolve. Either way we have nothing to check reach against.
    let Some(fact) = facts.iter().find(|f| f.raw == raw) else {
        return (body, Vec::new());
    };
    body.resolved = fact.resolved.clone();
    // Recorded, not decided on: guardrail 3 refuses `Ignored` and `Unknown`
    // below, so anything that survives is one of the other four. This exists so
    // the audit row can say how much of the population is `NoRepo`, where that
    // guardrail is structurally inert — `~/.openfang/scripts/` is not a
    // repository, and it is the directory item 1 exists to serve.
    body.git = fact.git;

    // Guardrail 2, first and unconditional. `Prompt` is excluded because the
    // operator already said they want to be asked about this path, and
    // `NoPolicy` because outside `file_policy` there is no reach to be inside
    // of — the ANAI-190 rule, unchanged.
    if !matches!(fact.authority, PathAuthority::Read | PathAuthority::Write) {
        body.status = ScriptBodyStatus::OutsideReach;
        return (body, Vec::new());
    }

    // Guardrail 2b — ANAI-206 commit 9, G3.
    //
    // `fact.authority` above was computed against a *lexically* normalized path
    // (`resolve` → `lexical_normalize`). That is correct for the fact sheet: an
    // unresolvable or symlinked leaf should still yield a fact rather than a
    // hole. It is the wrong basis for a *read*. `O_NOFOLLOW` below refuses a
    // symlinked leaf, but the kernel always traverses symlinked **parents** and
    // no open flag changes that:
    //
    // ```text
    //   ln -s /somewhere/outside ./s
    //   bash ./s/x.sh
    // ```
    //
    // resolves lexically to `<workspace>/s/x.sh`, so authority is `Write`,
    // guardrail 2 passes, the leaf is a real regular file so `symlink_metadata`
    // says `File`, and `O_NOFOLLOW` opens it without complaint. The body of a
    // file outside this agent's `file_policy` reach then lands in the judge
    // prompt and in the audit chain — with no race to win, which makes it worse
    // than the TOCTOU F8 filed and fixed with `O_NOFOLLOW`.
    //
    // Canonicalize the **parent**, which resolves every intermediate link, and
    // re-ask `file_policy` about the real location. The leaf is deliberately
    // left unresolved: a symlinked leaf must keep being refused by identity
    // (`PathExistence::File` from `symlink_metadata`, then `O_NOFOLLOW`) rather
    // than being laundered into its target's authority here.
    let Some(real) = canonical_parent_path(fact.resolved.as_deref()) else {
        // No parent, or a parent that does not exist. Nothing to read.
        return (body, Vec::new());
    };
    if !matches!(
        authority_for(&real, canon_ws, file_policy),
        PathAuthority::Read | PathAuthority::Write
    ) {
        body.status = ScriptBodyStatus::OutsideReach;
        return (body, Vec::new());
    }

    // Guardrail 3. `Unknown` is a refusal, not a pass: the git budget expiring
    // under fleet load must not launder an ignored file into a readable one.
    match fact.git {
        GitFact::Ignored => {
            body.status = ScriptBodyStatus::Ignored;
            return (body, Vec::new());
        }
        GitFact::Unknown => {
            body.status = ScriptBodyStatus::GitUnknown;
            return (body, Vec::new());
        }
        _ => {}
    }

    // `PathExistence::File` comes from `symlink_metadata`, so this also carries
    // the no-traversal invariant: a symlinked leaf is `Symlink`, never `File`,
    // and lands in `Unreadable` without being opened.
    if fact.existence != PathExistence::File {
        return (body, Vec::new());
    }

    // Guardrail 4, on the stat first so an oversized file is never read into
    // memory at all.
    if fact.size_bytes.unwrap_or(u64::MAX) > MAX_SCRIPT_BODY_BYTES {
        body.status = ScriptBodyStatus::TooLarge;
        return (body, Vec::new());
    }

    // Guardrail 4b — ANAI-206 F8, the TOCTOU the security review found.
    //
    // Everything above was decided against a *path*, stat'd earlier in
    // `gather()` with a 300ms git budget in between. `std::fs::read` then
    // re-resolved that path and followed symlinks, and the size re-check
    // caught growth but never identity. An agent that can write the script —
    // and it must be able to, for the script to be worth reading — can leave a
    // background flipper swapping the file for a link and have the gate render
    // a file outside its own `file_policy` reach into the judge prompt and the
    // audit chain. That is exactly what guardrail 2 exists to prevent.
    //
    // `O_NOFOLLOW` refuses at `open`, and every check from here down is against
    // the handle rather than the name: we validate the bytes we are holding,
    // not a path we re-resolve.
    //
    // Opened at `real` — the parent-canonicalized path guardrail 2b actually
    // authorized — rather than at the lexical form, so the bytes read come from
    // the location `file_policy` was asked about.
    let Ok(mut file) = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&real)
    else {
        return (body, Vec::new());
    };
    let Ok(meta) = file.metadata() else {
        return (body, Vec::new());
    };
    // `fstat` on the open handle. A directory or a device that replaced the
    // file between the two is not a script.
    if !meta.is_file() {
        return (body, Vec::new());
    }
    if meta.len() > MAX_SCRIPT_BODY_BYTES {
        body.status = ScriptBodyStatus::TooLarge;
        return (body, Vec::new());
    }
    let mut bytes: Vec<u8> = Vec::new();
    if file
        .by_ref()
        .take(MAX_SCRIPT_BODY_BYTES + 1)
        .read_to_end(&mut bytes)
        .is_err()
    {
        return (body, Vec::new());
    }
    // ...and again on what we actually got, because a file can grow between
    // the `fstat` and the last byte read.
    if bytes.len() as u64 > MAX_SCRIPT_BODY_BYTES {
        body.status = ScriptBodyStatus::TooLarge;
        return (body, Vec::new());
    }

    // Guardrail 5.
    let Ok(text) = String::from_utf8(bytes) else {
        body.status = ScriptBodyStatus::Binary;
        return (body, Vec::new());
    };
    if looks_binary(&text) {
        body.status = ScriptBodyStatus::Binary;
        return (body, Vec::new());
    }

    // ANAI-206 item 3, first phase. Both of these read the *unredacted* text,
    // for the reason in this function's doc comment: redaction can blank a line
    // that names a control path, and a scan of the redacted form would then
    // miss exactly what it is looking for.
    body.writes_control_plane = openfang_types::gatekeeper::body_writes_control_plane(&text);
    // ANAI-206 commit 6. `writes_control_plane` is now a fact the judge weighs,
    // so the hard case has to be named separately: a recursive removal of the
    // substrate on line 40 of the file this command runs bypasses the judge for
    // exactly the same reason it does on the command line.
    body.destroys_substrate = openfang_types::gatekeeper::body_destroys_substrate(&text);
    // ANAI-206 commit 9, C6-3. The other three hard flags, which commit 6 and
    // commit 8 both left scoped to the command line. A hard flag that stops at
    // the command line has a one-line bypass — put the line in a file and run
    // the file — and commit 6 had already rejected that argument for
    // `substrate_destruction` by adding the predicate above it.
    body.writes_gatekeeper_policy =
        openfang_types::gatekeeper::body_writes_gatekeeper_policy(&text);
    body.writes_agent_config = openfang_types::gatekeeper::body_writes_agent_config(&text);
    body.writes_runtime_config = openfang_types::gatekeeper::body_writes_runtime_config(&text);

    // Guardrail 6, and it now runs *before* extraction — ANAI-206 F9.
    //
    // `extract_body_path_tokens` used to read the unredacted text, and
    // `PathFact::render` prints `resolved.unwrap_or(&self.raw)`. So
    // `API_TOKEN=/Users/x/.ssh/deploy_key_abc123` was redacted *inside* the
    // `<script-body>` fence and printed in full as a path fact *outside* it,
    // where the facts block deliberately is not redacted. `looks_secret` cannot
    // catch it on the way out either: it excludes any token containing `/`,
    // which is every path.
    //
    // Extract twice rather than redacting the rendered block, which would eat
    // the ordinary paths the map exists for. The two predicates above keep
    // reading plaintext on purpose — redaction can blank a line that names a
    // control path — and only the rendered map moves behind the redactor.
    let (redacted, redactions) = redact_secrets(&text);
    let (tokens, unresolved) = extract_body_path_tokens(&redacted);
    body.body_truncated = tokens.len() > MAX_BODY_PATH_FACTS;
    body.body_unresolved = unresolved;
    let tokens: Vec<String> = tokens.into_iter().take(MAX_BODY_PATH_FACTS).collect();
    body.content = Some(redacted);
    body.redactions = redactions;
    body.status = ScriptBodyStatus::Included;
    (body, tokens)
}

/// True when text carries control bytes in a shape no script has.
///
/// A NUL settles it. Beyond that a handful of stray control characters is
/// tolerated — terminal escapes turn up in real deploy scripts — but a run of
/// them means we are looking at a compiled artifact that happened to decode as
/// UTF-8, and handing that to a model is noise at best.
fn looks_binary(text: &str) -> bool {
    if text.contains('\0') {
        return true;
    }
    text.chars()
        .filter(|c| c.is_control() && *c != '\n' && *c != '\r' && *c != '\t')
        .count()
        > 8
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

/// The path with every symlink in its **parent** chain resolved, and its final
/// component left exactly as written.
///
/// ANAI-206 commit 9, G3. `Path::canonicalize` on the whole path would follow a
/// symlinked leaf too, which is precisely the identity check `O_NOFOLLOW` and
/// `symlink_metadata` exist to keep. Canonicalizing only the parent gives the
/// real directory the file lives in — the thing `file_policy` must be asked
/// about — without dissolving the leaf.
///
/// `None` when the path has no parent (the filesystem root, which is not a
/// script) or when the parent does not resolve, in which case there is nothing
/// to read either.
fn canonical_parent_path(resolved: Option<&str>) -> Option<PathBuf> {
    let lexical = Path::new(resolved?);
    let parent = lexical.parent()?;
    let name = lexical.file_name()?;
    Some(parent.canonicalize().ok()?.join(name))
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
    use openfang_types::path_facts::MAX_SCRIPT_BODY_BYTES;

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

    // -----------------------------------------------------------------------
    // ANAI-206 item 3, end to end against real files.
    // -----------------------------------------------------------------------

    /// A workspace whose whole tree is writable, which is the shape the item 1
    /// guardrails require before a body is read at all.
    fn writable_ws(name: &str) -> (PathBuf, FilePolicy) {
        let ws = tempdir(name);
        let policy = FilePolicy::new(
            true,
            FileAccessTier::Deny,
            vec![FileRule {
                path: ws.display().to_string(),
                tier: FileAccessTier::Write,
            }],
        );
        (ws, policy)
    }

    /// The regression items 1 and 2 opened, closed end to end: a script that
    /// writes the substrate fires the floor even though the command line that
    /// runs it is unremarkable.
    #[tokio::test]
    async fn a_script_that_writes_the_control_plane_is_flagged() {
        let (ws, policy) = writable_ws("body-write");
        std::fs::write(
            ws.join("deploy.sh"),
            "#!/usr/bin/env bash\nset -e\ncargo build\nrm -rf ~/.openfang/agents/\n",
        )
        .unwrap();

        let sheet = gather("bash ./deploy.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.as_ref().expect("body read");
        assert_eq!(body.status, ScriptBodyStatus::Included);
        assert!(body.writes_control_plane);
        assert!(!sheet.suppress_eligible());
        assert!(sheet.as_log_token().contains("body_control_plane"));
    }

    /// ...and item 2's win survives: a script that merely *reads* the control
    /// plane still reaches the judge rather than short-circuiting.
    #[tokio::test]
    async fn a_script_that_only_reads_the_control_plane_is_not_flagged() {
        let (ws, policy) = writable_ws("body-read");
        std::fs::write(
            ws.join("status.sh"),
            "#!/usr/bin/env bash\ncat ~/.openfang/config.toml\nls ~/.openfang/agents\n",
        )
        .unwrap();

        let sheet = gather("bash ./status.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.as_ref().expect("body read");
        assert!(!body.writes_control_plane, "{body:?}");
    }

    /// The body's paths get the same treatment the command's do — stat, git,
    /// containment, authority — rather than being left as prose for the judge
    /// to parse.
    #[tokio::test]
    async fn the_bodys_paths_are_stat_like_any_other() {
        let (ws, policy) = writable_ws("body-facts");
        std::fs::write(ws.join("data.txt"), "x").unwrap();
        std::fs::write(
            ws.join("run.sh"),
            "#!/usr/bin/env bash\nrm ./data.txt\nrm ./gone.txt\n",
        )
        .unwrap();

        let sheet = gather("bash ./run.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.as_ref().expect("body read");
        let by_raw = |raw: &str| {
            body.body_facts
                .iter()
                .find(|f| f.raw == raw)
                .unwrap_or_else(|| panic!("{raw} missing from {:?}", body.body_facts))
                .clone()
        };
        let data = by_raw("./data.txt");
        assert_eq!(data.existence, PathExistence::File);
        assert!(data.inside_workspace);
        assert_eq!(data.authority, PathAuthority::Write);
        assert_eq!(by_raw("./gone.txt").existence, PathExistence::Missing);
    }

    /// Redaction runs *after* the scan, and that ordering is load-bearing: a
    /// key that reads like a secret blanks its value, so scanning the redacted
    /// form would lose the very path the map exists to record.
    #[tokio::test]
    async fn redaction_cannot_hide_a_path_from_the_map() {
        // Distinct from the item 1 redaction test's directory: `tempdir` wipes
        // the tree it is handed, and two tests sharing a name race.
        let (ws, policy) = writable_ws("body-redact-order");
        std::fs::write(
            ws.join("run.sh"),
            "#!/usr/bin/env bash\nAUTH_FILE=./creds.txt\nrm ./creds.txt\n",
        )
        .unwrap();

        let sheet = gather("bash ./run.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.as_ref().expect("body read");
        // The judge sees the redacted line...
        assert!(body.content.as_ref().unwrap().contains("[redacted]"));
        assert!(body.redactions > 0);
        // ...and the map still knows the path.
        assert!(
            body.body_facts.iter().any(|f| f.raw == "./creds.txt"),
            "{:?}",
            body.body_facts
        );
    }

    /// Guardrail 3 is structurally inert outside a repository, and that has to
    /// be countable rather than inferred.
    #[tokio::test]
    async fn the_audit_row_records_the_scripts_git_answer() {
        let (ws, policy) = writable_ws("body-git-token");
        std::fs::write(ws.join("run.sh"), "echo hello\n").unwrap();

        let sheet = gather("bash ./run.sh", &[], Some(&ws), Some(&policy)).await;
        assert_eq!(sheet.script_body.as_ref().unwrap().git, GitFact::NoRepo);
        assert!(sheet.as_log_token().contains("script_git=no-repo"));
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

    // -----------------------------------------------------------------------
    // ANAI-206 item 1: the script body. Each guardrail gets a test, and the
    // `file_policy` one gets two — it is the assertion that keeps the gate from
    // being a privilege-escalation oracle, and it is the one a future
    // refactor is most likely to quietly weaken.
    // -----------------------------------------------------------------------

    /// A policy granting `tier` over the whole of `ws`.
    fn policy_over(ws: &Path, tier: FileAccessTier) -> FilePolicy {
        FilePolicy::new(
            true,
            FileAccessTier::Deny,
            vec![FileRule {
                path: ws.display().to_string(),
                tier,
            }],
        )
    }

    /// The motivating case: 24 corpus rows are `bash ~/.openfang/scripts/*.sh`,
    /// and until item 2 the floor short-circuited every one of them. Now they
    /// reach the judge, and this is what the judge gets to read.
    #[tokio::test]
    async fn the_script_a_command_executes_is_read_and_handed_over() {
        let ws = tempdir("body-happy");
        std::fs::write(
            ws.join("deploy.sh"),
            "#!/bin/bash\nset -euo pipefail\ncargo build\n",
        )
        .unwrap();
        let policy = policy_over(&ws, FileAccessTier::Write);

        let sheet = gather("bash ./deploy.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.clone().expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Included);
        assert!(body.content.as_deref().unwrap().contains("cargo build"));
        assert_eq!(body.redactions, 0);
        assert!(sheet.as_log_token().contains("script=included"));
    }

    /// **The anti-oracle assertion.** The judge runs with daemon privileges. A
    /// path the requesting agent's `file_policy` does not reach must never be
    /// read, or any agent that can name a path in a command has an oracle for
    /// its contents. `Read` is enough — the agent could have `cat`-ed it —
    /// `Prompt` and `Deny` are not, and neither is an absent policy.
    #[tokio::test]
    async fn a_script_outside_the_agents_file_policy_reach_is_never_read() {
        let ws = tempdir("body-reach");
        std::fs::write(ws.join("secret.sh"), "echo the-content\n").unwrap();

        for tier in [FileAccessTier::Deny, FileAccessTier::Prompt] {
            let policy = policy_over(&ws, tier);
            let sheet = gather("bash ./secret.sh", &[], Some(&ws), Some(&policy)).await;
            let body = sheet.script_body.clone().expect("script body");
            assert_eq!(body.status, ScriptBodyStatus::OutsideReach, "{tier:?}");
            assert!(body.content.is_none(), "{tier:?}");
            assert!(!sheet.suppress_eligible(), "{tier:?}");
        }

        // No policy at all is not a permissive answer. `tier_for` would report
        // `Write` for every path on the box on an inert policy; `NoPolicy`
        // never reaches the read.
        for policy in [None, Some(FilePolicy::default())] {
            let sheet = gather("bash ./secret.sh", &[], Some(&ws), policy.as_ref()).await;
            let body = sheet.script_body.expect("script body");
            assert_eq!(body.status, ScriptBodyStatus::OutsideReach);
            assert!(body.content.is_none());
        }

        // ...and a readable-but-not-writable script is fine: the agent could
        // have read it itself, so the judge learns nothing new.
        let policy = policy_over(&ws, FileAccessTier::Read);
        let sheet = gather("bash ./secret.sh", &[], Some(&ws), Some(&policy)).await;
        assert_eq!(
            sheet.script_body.unwrap().status,
            ScriptBodyStatus::Included
        );
    }

    /// Over the cap the body is refused, not truncated: the interesting line
    /// goes at byte 17000 and a truncated read would show the judge a
    /// perfectly ordinary first 16KB.
    #[tokio::test]
    async fn an_oversized_script_is_refused_rather_than_truncated() {
        let ws = tempdir("body-large");
        let mut script = "echo padding\n".repeat(2000);
        script.push_str("rm -rf /\n");
        assert!(script.len() as u64 > MAX_SCRIPT_BODY_BYTES);
        std::fs::write(ws.join("big.sh"), &script).unwrap();

        let sheet = gather(
            "bash ./big.sh",
            &[],
            Some(&ws),
            Some(&policy_over(&ws, FileAccessTier::Write)),
        )
        .await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::TooLarge);
        assert!(body.content.is_none());
    }

    #[tokio::test]
    async fn a_binary_file_is_refused() {
        let ws = tempdir("body-binary");
        std::fs::write(
            ws.join("blob.sh"),
            [0x7f, 0x45, 0x4c, 0x46, 0x00, 0x01, 0x02],
        )
        .unwrap();

        let sheet = gather(
            "bash ./blob.sh",
            &[],
            Some(&ws),
            Some(&policy_over(&ws, FileAccessTier::Write)),
        )
        .await;
        assert_eq!(sheet.script_body.unwrap().status, ScriptBodyStatus::Binary);
    }

    /// `./harmless.sh -> /etc/hosts` must not be read through. The existence
    /// axis is `symlink_metadata`, so a link is `Symlink` and never `File`.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_script_is_not_followed() {
        let ws = tempdir("body-symlink");
        std::os::unix::fs::symlink("/etc/hosts", ws.join("harmless.sh")).unwrap();

        let sheet = gather(
            "bash ./harmless.sh",
            &[],
            Some(&ws),
            Some(&policy_over(&ws, FileAccessTier::Write)),
        )
        .await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Unreadable);
        assert!(body.content.is_none());
    }

    /// The class that holds every `.env`, key and token on this box. Also the
    /// git axis failing closed: an unproven answer is not a clean one.
    #[tokio::test]
    async fn a_git_ignored_script_is_never_read() {
        let repo = tempdir("body-ignored");
        if !git(&repo, &["init", "--quiet"]).await {
            eprintln!("git unavailable; skipping");
            return;
        }
        git(&repo, &["config", "user.email", "t@example.com"]).await;
        git(&repo, &["config", "user.name", "t"]).await;
        std::fs::write(repo.join(".gitignore"), "local-*.sh\n").unwrap();
        std::fs::write(repo.join("local-secrets.sh"), "export TOKEN=abc\n").unwrap();
        assert!(git(&repo, &["add", ".gitignore"]).await);
        assert!(git(&repo, &["commit", "--quiet", "-m", "seed"]).await);

        let sheet = gather(
            "bash ./local-secrets.sh",
            &[],
            Some(&repo),
            Some(&policy_over(&repo, FileAccessTier::Write)),
        )
        .await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Ignored);
        assert!(body.content.is_none());
    }

    /// Secrets are redacted before the bytes reach a model or the audit chain.
    #[tokio::test]
    async fn secrets_are_redacted_before_the_judge_sees_them() {
        let ws = tempdir("body-redact");
        std::fs::write(
            ws.join("env.sh"),
            "export ANTHROPIC_API_KEY=sk-ant-0123456789\ncargo build\n",
        )
        .unwrap();

        let sheet = gather(
            "bash ./env.sh",
            &[],
            Some(&ws),
            Some(&policy_over(&ws, FileAccessTier::Write)),
        )
        .await;
        let body = sheet.script_body.clone().expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Included);
        assert_eq!(body.redactions, 1);
        let content = body.content.unwrap();
        assert!(!content.contains("sk-ant-0123456789"), "{content}");
        assert!(content.contains("cargo build"), "{content}");
        assert!(sheet.as_log_token().contains("redacted=1"));
    }

    /// Guardrail 1 end to end: a command that does not execute exactly one
    /// readable file reads nothing, and the sheet says so by omission.
    #[tokio::test]
    async fn commands_that_do_not_execute_a_single_file_read_nothing() {
        let ws = tempdir("body-none");
        std::fs::write(ws.join("a.sh"), "echo hi\n").unwrap();
        let policy = policy_over(&ws, FileAccessTier::Write);

        for cmd in ["rm ./a.sh", "cat ./a.sh", "bash -c \"echo hi\""] {
            let sheet = gather(cmd, &[], Some(&ws), Some(&policy)).await;
            assert!(sheet.script_body.is_none(), "{cmd}");
        }
    }

    // -----------------------------------------------------------------------
    // ANAI-206 commit 7: F8 (TOCTOU at the read) and F9 (the redaction leak).
    // -----------------------------------------------------------------------

    /// **F8.** Every guardrail above the read is decided against a *path*,
    /// stat'd earlier in `gather()` with a git budget in between. `std::fs::read`
    /// re-resolved that path and followed links, so an agent that can write the
    /// script — and it must be able to, for the script to be worth reading — can
    /// swap it for a symlink inside that window and have the gate render a file
    /// outside its own `file_policy` reach into the judge prompt and the audit
    /// chain.
    ///
    /// The window is not reproducible in a test without a racing thread, so this
    /// asserts the property that closes it instead: a fact that *claims* the path
    /// is a regular file must not be enough to read through a link. The fact here
    /// is deliberately a lie, which is exactly what the attacker manufactures.
    #[tokio::test]
    async fn a_stale_fact_cannot_read_through_a_symlink() {
        let ws = tempdir("body-toctou");
        let link = ws.join("swapped.sh");
        std::os::unix::fs::symlink("/etc/hosts", &link).unwrap();

        let lying_fact = PathFact {
            raw: "./swapped.sh".to_string(),
            resolved: Some(link.display().to_string()),
            // What the stat said 300ms ago, before the flip.
            existence: PathExistence::File,
            size_bytes: Some(12),
            symlink_target: None,
            mtime_secs_ago: Some(0),
            git: GitFact::NoRepo,
            inside_workspace: true,
            authority: PathAuthority::Write,
        };

        let (body, tokens) = read_script_body("./swapped.sh", &[lying_fact], None, None);
        assert!(
            body.content.is_none(),
            "O_NOFOLLOW must refuse the link even when the fact says File"
        );
        assert_ne!(body.status, ScriptBodyStatus::Included);
        assert!(tokens.is_empty());
    }

    // -----------------------------------------------------------------------
    // ANAI-206 commit 9: G3 — a symlinked *parent* beats `O_NOFOLLOW`
    // -----------------------------------------------------------------------

    /// The finding F8's fix did not cover, and the worse half of it: no race at
    /// all.
    ///
    /// `O_NOFOLLOW` refuses a symlinked leaf. The kernel always traverses
    /// symlinked *parents* and no open flag changes that, and
    /// `path_facts::resolve` normalizes **lexically** on purpose — so the
    /// authority guardrail 2 checks is the authority of a directory that does
    /// not exist. Point a link inside the workspace at a directory outside the
    /// agent's `file_policy` reach, put an ordinary regular file behind it, and
    /// every guardrail passes: lexical path is in-workspace, tier is `Write`,
    /// `symlink_metadata` on the leaf says `File`, `O_NOFOLLOW` opens it.
    ///
    /// Guardrail 2b canonicalizes the parent and re-asks. Note the file is a
    /// real regular file with no link in sight at the leaf, which is what makes
    /// this different from the TOCTOU above.
    #[tokio::test]
    async fn a_symlinked_parent_cannot_launder_a_body_into_reach() {
        let (ws, policy) = writable_ws("body-parent-link");
        let outside = tempdir("body-parent-link-outside");
        std::fs::write(outside.join("x.sh"), "echo pwned\n").unwrap();
        std::os::unix::fs::symlink(&outside, ws.join("s")).unwrap();

        let sheet = gather("bash ./s/x.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(
            body.status,
            ScriptBodyStatus::OutsideReach,
            "a body behind a symlinked parent must be refused, not rendered"
        );
        assert!(body.content.is_none());
        assert!(
            !body.content.iter().any(|c| c.contains("pwned")),
            "the out-of-reach bytes must never reach the prompt or the audit row"
        );
    }

    /// The over-refusal direction. A symlinked parent that resolves to a
    /// directory still inside the agent's reach is ordinary — plenty of real
    /// workspaces have one — and must still read.
    #[tokio::test]
    async fn a_symlinked_parent_inside_reach_still_reads() {
        let (ws, policy) = writable_ws("body-parent-link-ok");
        std::fs::create_dir_all(ws.join("real")).unwrap();
        std::fs::write(ws.join("real/x.sh"), "cargo build\n").unwrap();
        std::os::unix::fs::symlink(ws.join("real"), ws.join("s")).unwrap();

        let sheet = gather("bash ./s/x.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Included);
        assert!(body.content.unwrap().contains("cargo build"));
    }

    /// The ordinary path still reads, so the `O_NOFOLLOW` open is not a blanket
    /// refusal.
    ///
    /// (Commit 9 adds guardrail 2b above the open; this is also the assertion
    /// that parent-canonicalization did not turn every ordinary read into a
    /// refusal.)
    #[tokio::test]
    async fn a_real_file_still_reads_through_the_nofollow_open() {
        let (ws, policy) = writable_ws("body-nofollow-ok");
        std::fs::write(
            ws.join("ok.sh"),
            "echo hi
",
        )
        .unwrap();
        let sheet = gather("bash ./ok.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Included);
        assert_eq!(body.content.as_deref(), Some("echo hi\n"));
    }

    /// **F9.** `PathFact::render` prints the token it was built from, and the
    /// body facts block renders *outside* the `<script-body>` fence where nothing
    /// redacts it. Extraction therefore has to run on the redacted text, or a
    /// secret-shaped assignment whose value is a path is blanked inside the fence
    /// and printed in full immediately below it. `looks_secret` cannot catch it
    /// on the way out: it excludes every token containing `/`, which is every
    /// path.
    #[tokio::test]
    async fn a_redacted_secret_path_is_not_re_emitted_as_a_fact() {
        let (ws, policy) = writable_ws("body-f9");
        std::fs::write(
            ws.join("deploy.sh"),
            "#!/usr/bin/env bash\nAPI_TOKEN=/Users/x/.ssh/deploy_key_abc123\ncargo build\n",
        )
        .unwrap();

        let sheet = gather("bash ./deploy.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.expect("script body");
        assert_eq!(body.status, ScriptBodyStatus::Included);
        assert!(body.redactions > 0);

        let content = body.content.as_deref().unwrap_or_default();
        assert!(!content.contains("deploy_key_abc123"), "{content}");

        let rendered = body.render_body_facts();
        assert!(
            !rendered.contains("deploy_key_abc123"),
            "the facts block renders outside the fence: {rendered}"
        );
        assert!(
            body.body_facts
                .iter()
                .all(|f| !f.raw.contains("deploy_key_abc123")),
            "{:?}",
            body.body_facts
        );
    }

    /// ...and the paths the map exists for are still there. Redacting the
    /// rendered block instead of extracting twice would have eaten these.
    #[tokio::test]
    async fn ordinary_paths_survive_the_redaction_reorder() {
        let (ws, policy) = writable_ws("body-f9-ok");
        std::fs::write(ws.join("data.txt"), "x").unwrap();
        std::fs::write(
            ws.join("run.sh"),
            "#!/usr/bin/env bash\nAPI_TOKEN=/Users/x/.ssh/k_abc123\nrm ./data.txt\n",
        )
        .unwrap();

        let sheet = gather("bash ./run.sh", &[], Some(&ws), Some(&policy)).await;
        let body = sheet.script_body.expect("script body");
        assert!(
            body.body_facts.iter().any(|f| f.raw == "./data.txt"),
            "{:?}",
            body.body_facts
        );
    }
}
