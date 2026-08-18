//! ANAI-190: the path fact sheet.
//!
//! The judge's weakness is that it reasons about a command as *text*. `rm
//! ./decision-scratch.txt` and `rm src/kernel.rs` are the same string shape and
//! wildly different acts, and no syntactic predicate can tell them apart. What
//! separates them is not danger, it is **recoverability** — one is
//! reconstructible from git in two seconds and the other is not — and
//! **authority**: whether the agent could have done this through its own file
//! tools anyway.
//!
//! Both are facts about the filesystem, not opinions about the command. So we
//! compute them and hand them to the judge rather than asking it to guess.
//!
//! # Why facts and not a tool call
//!
//! Giving the judge a `read_file` tool would add a round trip to a latency
//! budget whose p90 is already 3.9s, and would open a content-exfiltration
//! surface: the judge runs with daemon privileges, so a tool would let it read
//! files the *requesting agent itself* cannot — turning the gatekeeper into a
//! privilege-escalation oracle for any agent that can name a path in a command.
//!
//! Nothing in this module reads file *contents*. Only `symlink_metadata`, the
//! git index, and a pure tier lookup. There is no byte of user data in a
//! [`PathFact`], so there is nothing to leak.
//!
//! # Why symlinks are never followed
//!
//! `./harmless -> /etc/passwd` must not read as harmless. Every stat here is a
//! `symlink_metadata` — the link is reported *as a link*, with its target
//! named, and a symlink is never recoverable regardless of what it points at.
//!
//! # TOCTOU
//!
//! Facts are gathered at t0. Under shadow mode that is free: nothing acts on
//! them. Once the deterministic fast-path is permitted to suppress, the sheet
//! must be re-gathered immediately before execution and a mismatch must fail to
//! `Escalate` — a file that was tracked-and-clean when the judge saw it may not
//! be eight seconds later when the operator clicks.

use serde::{Deserialize, Serialize};

/// Hard cap on how many paths a single command contributes to the sheet.
///
/// A prompt is a budget. A command naming forty paths is not one the fast-path
/// should ever be confident about anyway, so truncation is recorded (see
/// [`PathFactSheet::truncated`]) and treated as a reason to withhold
/// suppression rather than as a cosmetic elision.
pub const MAX_PATH_FACTS: usize = 8;

/// What `symlink_metadata` found. Deliberately not "safe"/"unsafe".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathExistence {
    /// No such path. A destructive verb against it is a no-op.
    Missing,
    File,
    Dir,
    /// A symlink. Never followed; see [`PathFact::symlink_target`].
    Symlink,
    /// Stat failed for a reason other than absence (permissions, a broken
    /// mount, a path we could not resolve). Fails closed everywhere.
    Unknown,
}

/// What the git index says about a path.
///
/// `NoRepo` is a first-class answer, not a synonym for `Untracked`. It is also
/// the *common* answer on this fleet: `shell_exec` runs in the agent's
/// workspace root, and workspaces are not git repositories, so the walk-up from
/// a relative path terminates without ever finding a `.git`. The git axis is
/// therefore silent for most agent-local scratch files, and their
/// recoverability has to come from containment and authority instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitFact {
    /// No `.git` anywhere between the path and the search boundary.
    NoRepo,
    /// Tracked and identical to the index.
    TrackedClean,
    /// Tracked with uncommitted modifications. This is the case that most
    /// deserves a human: the content exists nowhere else.
    TrackedDirty,
    /// Inside a repo, not in the index.
    Untracked,
    /// Inside a repo and matched by `.gitignore`. Never recoverable, and the
    /// class that holds essentially every `.env`, key and credential file on
    /// the box.
    Ignored,
    /// The query timed out, errored, or was never run. Fails closed.
    Unknown,
}

/// The tier `file_policy` would have granted this agent for this exact path.
///
/// Shell bypasses `file_policy` entirely — `shell_exec` never routes through
/// `tier_for`, only `file_write` and `apply_patch` do. The gate is therefore
/// the first and only place we can ask whether a command is doing something the
/// agent could not have done through its own file tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathAuthority {
    /// The agent could have written this path with `file_write`. Shell adds no
    /// authority, so suppression adds no exposure.
    Write,
    /// Explicitly readable, explicitly not writable. A shell write here is a
    /// policy bypass and must be seen.
    Read,
    /// The operator already said they want to be asked about this path.
    Prompt,
    /// The agent cannot touch this through its file tools at all.
    Deny,
    /// No active `file_policy` for this agent — neither a manifest block nor a
    /// global floor.
    ///
    /// This is **not** a permissive answer and must never be treated as one.
    /// `FilePolicy::tier_for` returns `Write` for every path on the filesystem
    /// when the policy is inert (`config.rs`: `let own = if self.enabled { .. }
    /// else { FileAccessTier::Write }`), so calling it on an inert policy would
    /// report maximal authority everywhere — the exact inverse of fail-closed.
    /// We gate on `FilePolicy::is_active()` and never call `tier_for` at all in
    /// that case. Operator decision, 2026-08-17: outside `file_policy`, no
    /// suppression.
    NoPolicy,
}

impl PathAuthority {
    /// Short token for prompts and audit rows.
    #[must_use]
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Write => "write",
            Self::Read => "read",
            Self::Prompt => "prompt",
            Self::Deny => "deny",
            Self::NoPolicy => "no-policy",
        }
    }

    /// True only for `Write`. Every other tier — including `NoPolicy` — means
    /// the shell invocation reaches past what the agent's file tools grant.
    #[must_use]
    pub fn authorized(self) -> bool {
        matches!(self, Self::Write)
    }
}

/// Everything we know about one path named by one command. No file contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathFact {
    /// The token as the agent wrote it. Untrusted; rendered, never executed.
    pub raw: String,
    /// Absolute resolved form, with a leading `~` expanded and relative paths
    /// joined to the workspace root. `None` when resolution failed.
    pub resolved: Option<String>,
    pub existence: PathExistence,
    /// Size in bytes for a regular file. Metadata, not content.
    pub size_bytes: Option<u64>,
    /// Where a symlink points. Reported, never traversed.
    pub symlink_target: Option<String>,
    /// Age of the last modification, in seconds. `None` when unknown.
    pub mtime_secs_ago: Option<u64>,
    pub git: GitFact,
    /// Inside the requesting agent's own workspace root.
    pub inside_workspace: bool,
    pub authority: PathAuthority,
}

impl PathFact {
    /// A path whose loss can be undone without a human.
    ///
    /// Deliberately conservative, and deliberately **not** a danger judgement:
    /// the question is only "can this be reconstructed". Anything unknown,
    /// ignored, dirty, symlinked or directory-shaped answers no.
    #[must_use]
    pub fn recoverable(&self) -> bool {
        match self.existence {
            // Nothing to lose.
            PathExistence::Missing => true,
            // Never reason through a link, and never reason about a whole tree
            // from one stat.
            PathExistence::Symlink | PathExistence::Dir | PathExistence::Unknown => false,
            PathExistence::File => match self.git {
                GitFact::TrackedClean => true,
                // The motivating case: `rm ./decision-scratch.txt` in an agent
                // workspace. Git is silent here — workspaces are not repos — so
                // recoverability rests entirely on it being the agent's own
                // scratch space. Without this arm the fast-path buys nothing on
                // exactly the traffic it was built for.
                GitFact::NoRepo | GitFact::Untracked => self.inside_workspace,
                GitFact::TrackedDirty | GitFact::Ignored | GitFact::Unknown => false,
            },
        }
    }

    /// The conjunction. Both axes must be green.
    ///
    /// Recoverable-but-unauthorized still escalates: `rm` of a tracked, clean
    /// file in someone else's repository is trivially recoverable and
    /// emphatically not this agent's call. Authorized-but-unrecoverable still
    /// escalates: uncommitted work in a repo the agent owns is still the only
    /// copy.
    #[must_use]
    pub fn suppress_eligible(&self) -> bool {
        self.recoverable() && self.authority.authorized()
    }

    /// One line for the judge. Facts only, no recommendation — the model is
    /// being informed, not instructed.
    #[must_use]
    pub fn render(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        parts.push(
            match self.existence {
                PathExistence::Missing => "does not exist",
                PathExistence::File => "file",
                PathExistence::Dir => "directory",
                PathExistence::Symlink => "symlink (not followed)",
                PathExistence::Unknown => "unstattable",
            }
            .to_string(),
        );
        if let Some(target) = &self.symlink_target {
            parts.push(format!("-> {target}"));
        }
        if let Some(size) = self.size_bytes {
            parts.push(format!("{size}B"));
        }
        parts.push(
            match self.git {
                GitFact::NoRepo => "no git repo",
                GitFact::TrackedClean => "git-tracked, clean",
                GitFact::TrackedDirty => "git-tracked, UNCOMMITTED CHANGES",
                GitFact::Untracked => "untracked",
                GitFact::Ignored => "git-ignored",
                GitFact::Unknown => "git status unknown",
            }
            .to_string(),
        );
        parts.push(
            if self.inside_workspace {
                "inside agent workspace"
            } else {
                "outside agent workspace"
            }
            .to_string(),
        );
        parts.push(format!("file_policy tier: {}", self.authority.as_token()));
        if let Some(age) = self.mtime_secs_ago {
            parts.push(format!("modified {}", human_age(age)));
        }
        format!(
            "{} -> {}",
            self.resolved.as_deref().unwrap_or(&self.raw),
            parts.join(", ")
        )
    }
}

/// Every path fact for one command, plus the reasons the sheet may be blind.
///
/// The blindness flags are load-bearing. A fast-path that suppresses on a sheet
/// it knows to be incomplete is worse than no fast-path, because it looks
/// principled while acting on partial information.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathFactSheet {
    pub facts: Vec<PathFact>,
    /// More than [`MAX_PATH_FACTS`] paths were named; the tail was dropped.
    pub truncated: bool,
    /// At least one argument was a glob, a variable expansion, or otherwise not
    /// resolvable to a single path. We do not know what it names.
    pub unresolved: bool,
}

impl PathFactSheet {
    /// True when every named path is both recoverable and authorized, and the
    /// sheet saw all of them.
    ///
    /// Unused under shadow mode by design — commit 1 of ANAI-190 measures; the
    /// fast-path is only permitted to act once the corpus says what it would
    /// have done.
    #[must_use]
    pub fn suppress_eligible(&self) -> bool {
        !self.truncated
            && !self.unresolved
            && !self.facts.is_empty()
            && self.facts.iter().all(PathFact::suppress_eligible)
    }

    /// Compact token for the audit metadata string, so the corpus is queryable
    /// without re-parsing prompts.
    ///
    /// Absent from every row written before ANAI-190. Queries spanning both
    /// corpora must read absence as *unknown*, never as "no paths".
    #[must_use]
    pub fn as_log_token(&self) -> String {
        if self.facts.is_empty() && !self.unresolved {
            return "none".to_string();
        }
        let recoverable = self.facts.iter().filter(|f| f.recoverable()).count();
        let authorized = self
            .facts
            .iter()
            .filter(|f| f.authority.authorized())
            .count();
        let mut token = format!(
            "n={} rec={} auth={}",
            self.facts.len(),
            recoverable,
            authorized
        );
        if self.truncated {
            token.push_str(" truncated");
        }
        if self.unresolved {
            token.push_str(" unresolved");
        }
        token
    }

    /// The block that goes into the judge's user turn.
    #[must_use]
    pub fn render(&self) -> String {
        if self.facts.is_empty() {
            return if self.unresolved {
                "(no resolvable paths; at least one argument was a glob or expansion)".to_string()
            } else {
                "(none)".to_string()
            };
        }
        let mut out = self
            .facts
            .iter()
            .map(|f| format!("  {}", f.render()))
            .collect::<Vec<_>>()
            .join("\n");
        if self.truncated {
            out.push_str("\n  [...more paths named than shown; sheet is incomplete]");
        }
        if self.unresolved {
            out.push_str(
                "\n  [at least one argument was a glob or expansion and was not resolved]",
            );
        }
        out
    }
}

/// Round a second count into something a reader parses at a glance.
fn human_age(secs: u64) -> String {
    match secs {
        0..=90 => format!("{secs}s ago"),
        91..=5400 => format!("{}m ago", secs / 60),
        5401..=172_800 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86_400),
    }
}

/// True when a token is shaped like a path rather than a flag, a subcommand, or
/// a bare word.
///
/// Intentionally narrow. A false negative costs us a fact (the sheet is
/// smaller, the fast-path is less confident, nothing is unsafe). A false
/// positive costs us a `stat` on a nonsense string, which is also harmless but
/// pollutes the prompt. Bare words like `status` or `main` are excluded because
/// almost every one of them is a git subcommand or a branch name.
fn looks_like_path(token: &str) -> bool {
    if token.is_empty() || token.starts_with('-') {
        return false;
    }
    token.contains('/') || token.starts_with('.') || token.starts_with('~')
}

/// True when a token cannot be resolved to exactly one path.
fn is_unresolvable(token: &str) -> bool {
    token.contains('*')
        || token.contains('?')
        || token.contains('[')
        || token.contains('$')
        || token.contains('{')
}

/// Pull the path-shaped arguments out of a command.
///
/// Operates on the comment-stripped command and on any inner commands lifted
/// from a shell wrapper, because `bash -c 'rm ./x'` names `./x` just as much as
/// `rm ./x` does. Returns the tokens plus whether anything was seen that could
/// not be resolved.
#[must_use]
pub fn extract_path_tokens(command: &str, inner: &[String]) -> (Vec<String>, bool) {
    let mut seen: Vec<String> = Vec::new();
    let mut unresolved = false;
    for source in std::iter::once(command).chain(inner.iter().map(String::as_str)) {
        for token in source.split_whitespace() {
            let token = token.trim_matches(|c| c == '"' || c == '\'');
            if !looks_like_path(token) {
                continue;
            }
            if is_unresolvable(token) {
                unresolved = true;
                continue;
            }
            let owned = token.to_string();
            if !seen.contains(&owned) {
                seen.push(owned);
            }
        }
    }
    (seen, unresolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(existence: PathExistence, git: GitFact, inside: bool, auth: PathAuthority) -> PathFact {
        PathFact {
            raw: "./x".into(),
            resolved: Some("/ws/x".into()),
            existence,
            size_bytes: Some(412),
            symlink_target: None,
            mtime_secs_ago: Some(240),
            git,
            inside_workspace: inside,
            authority: auth,
        }
    }

    /// Ben's motivating case. Git is silent — an agent workspace is not a repo
    /// — so this has to pass on containment plus authority or the whole feature
    /// buys nothing on the traffic that prompted it.
    #[test]
    fn workspace_scratch_file_is_suppress_eligible() {
        let f = fact(
            PathExistence::File,
            GitFact::NoRepo,
            true,
            PathAuthority::Write,
        );
        assert!(f.recoverable());
        assert!(f.suppress_eligible());
    }

    #[test]
    fn tracked_clean_file_is_recoverable() {
        let f = fact(
            PathExistence::File,
            GitFact::TrackedClean,
            false,
            PathAuthority::Write,
        );
        assert!(f.suppress_eligible());
    }

    /// Uncommitted work is the only copy. Authority does not rescue it.
    #[test]
    fn dirty_file_is_not_recoverable_even_when_authorized() {
        let f = fact(
            PathExistence::File,
            GitFact::TrackedDirty,
            true,
            PathAuthority::Write,
        );
        assert!(!f.recoverable());
        assert!(!f.suppress_eligible());
    }

    /// Trivially recoverable, emphatically not this agent's call.
    #[test]
    fn recoverable_but_unauthorized_still_escalates() {
        let f = fact(
            PathExistence::File,
            GitFact::TrackedClean,
            false,
            PathAuthority::Deny,
        );
        assert!(f.recoverable());
        assert!(!f.suppress_eligible());
    }

    /// The `config.rs:1109` trap, encoded as a test. An inert policy reports
    /// `no-policy`, never `write`, and `no-policy` never satisfies authority.
    #[test]
    fn inert_file_policy_is_never_authorized() {
        assert!(!PathAuthority::NoPolicy.authorized());
        let f = fact(
            PathExistence::File,
            GitFact::TrackedClean,
            true,
            PathAuthority::NoPolicy,
        );
        assert!(f.recoverable());
        assert!(!f.suppress_eligible());
    }

    #[test]
    fn ignored_files_are_never_recoverable() {
        let f = fact(
            PathExistence::File,
            GitFact::Ignored,
            true,
            PathAuthority::Write,
        );
        assert!(!f.recoverable());
    }

    /// `./harmless -> /etc/passwd` must not read as harmless.
    #[test]
    fn symlinks_are_never_recoverable() {
        let mut f = fact(
            PathExistence::Symlink,
            GitFact::NoRepo,
            true,
            PathAuthority::Write,
        );
        f.symlink_target = Some("/etc/passwd".into());
        assert!(!f.recoverable());
        assert!(f.render().contains("/etc/passwd"));
    }

    /// A timed-out git query must not read as a clean one.
    #[test]
    fn unknown_git_status_fails_closed() {
        let f = fact(
            PathExistence::File,
            GitFact::Unknown,
            true,
            PathAuthority::Write,
        );
        assert!(!f.recoverable());
    }

    #[test]
    fn missing_path_is_recoverable() {
        let f = fact(
            PathExistence::Missing,
            GitFact::NoRepo,
            true,
            PathAuthority::Write,
        );
        assert!(f.recoverable());
    }

    #[test]
    fn directories_are_never_recoverable_from_one_stat() {
        let f = fact(
            PathExistence::Dir,
            GitFact::TrackedClean,
            true,
            PathAuthority::Write,
        );
        assert!(!f.recoverable());
    }

    #[test]
    fn a_blind_sheet_is_never_eligible() {
        let good = fact(
            PathExistence::File,
            GitFact::TrackedClean,
            true,
            PathAuthority::Write,
        );
        let sheet = PathFactSheet {
            facts: vec![good.clone()],
            truncated: false,
            unresolved: false,
        };
        assert!(sheet.suppress_eligible());
        let truncated = PathFactSheet {
            truncated: true,
            ..sheet.clone()
        };
        assert!(!truncated.suppress_eligible());
        let unresolved = PathFactSheet {
            unresolved: true,
            ..sheet.clone()
        };
        assert!(!unresolved.suppress_eligible());
    }

    /// An empty sheet means we learned nothing, not that everything is fine.
    #[test]
    fn an_empty_sheet_is_never_eligible() {
        assert!(!PathFactSheet::default().suppress_eligible());
    }

    #[test]
    fn one_bad_path_poisons_the_sheet() {
        let sheet = PathFactSheet {
            facts: vec![
                fact(
                    PathExistence::File,
                    GitFact::TrackedClean,
                    true,
                    PathAuthority::Write,
                ),
                fact(
                    PathExistence::File,
                    GitFact::TrackedDirty,
                    true,
                    PathAuthority::Write,
                ),
            ],
            truncated: false,
            unresolved: false,
        };
        assert!(!sheet.suppress_eligible());
    }

    #[test]
    fn path_shaped_arguments_are_extracted_and_flags_are_not() {
        let (paths, unresolved) = extract_path_tokens("rm -rf ./scratch/a.txt /tmp/b", &[]);
        assert_eq!(paths, vec!["./scratch/a.txt", "/tmp/b"]);
        assert!(!unresolved);
    }

    /// Bare words are git subcommands and branch names far more often than
    /// they are paths.
    #[test]
    fn bare_words_are_not_paths() {
        let (paths, _) = extract_path_tokens("git status --short main", &[]);
        assert!(paths.is_empty(), "{paths:?}");
    }

    /// `bash -c 'rm ./x'` names `./x` exactly as much as `rm ./x` does.
    #[test]
    fn inner_commands_contribute_paths() {
        let (paths, _) = extract_path_tokens("bash -c \"rm ./x\"", &["rm ./x".to_string()]);
        assert_eq!(paths, vec!["./x"]);
    }

    #[test]
    fn globs_are_recorded_as_unresolved_not_as_paths() {
        let (paths, unresolved) = extract_path_tokens("rm ./build/*.o", &[]);
        assert!(paths.is_empty(), "{paths:?}");
        assert!(unresolved);
    }

    #[test]
    fn variable_expansion_is_unresolvable() {
        let (_, unresolved) = extract_path_tokens("rm $HOME/notes.md", &[]);
        assert!(unresolved);
    }

    #[test]
    fn log_token_is_absent_shaped_when_there_is_nothing_to_say() {
        assert_eq!(PathFactSheet::default().as_log_token(), "none");
    }

    #[test]
    fn log_token_carries_both_axes() {
        let sheet = PathFactSheet {
            facts: vec![
                fact(
                    PathExistence::File,
                    GitFact::TrackedClean,
                    true,
                    PathAuthority::Write,
                ),
                fact(
                    PathExistence::File,
                    GitFact::TrackedDirty,
                    true,
                    PathAuthority::Deny,
                ),
            ],
            truncated: false,
            unresolved: false,
        };
        assert_eq!(sheet.as_log_token(), "n=2 rec=1 auth=1");
    }
}
