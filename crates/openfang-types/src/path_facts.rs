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
//! A [`PathFact`] therefore carries no byte of user data: only
//! `symlink_metadata`, the git index, and a pure tier lookup.
//!
//! # The one exception (ANAI-206)
//!
//! [`ScriptBody`] does read bytes, and it is the difference between a judge
//! that can decide `bash ~/.openfang/scripts/deploy-local.sh` and one that
//! guesses. It is not a tool and it is not a round trip: the daemon reads
//! exactly one path, chosen by [`script_body_target`] from the command's own
//! text, and only when that path is inside the requesting agent's `file_policy`
//! reach. The oracle above is closed by the *reach* test — the judge cannot see
//! anything the requesting agent could not have `cat`-ed itself — not by the
//! absence of reads.
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

/// Hard cap on how many paths the *body* of an executed script contributes.
///
/// Larger than [`MAX_PATH_FACTS`] because a script legitimately names more
/// paths than a command line does, and small enough that the block stays
/// readable. Overflow is recorded (see [`ScriptBody::body_truncated`]) and
/// withholds suppression, exactly as command-line truncation does: a script we
/// have only partly mapped is a script whose effects we do not know.
pub const MAX_BODY_PATH_FACTS: usize = 12;

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

impl GitFact {
    /// Short token for audit rows.
    ///
    /// Exists so the corpus can see how often the git axis is *structurally*
    /// silent. `~/.openfang/scripts/` is not a repository, so every script
    /// under it answers `NoRepo` and guardrail 3 never fires there — that is a
    /// coverage fact we want to be able to count, not infer.
    #[must_use]
    pub fn as_token(self) -> &'static str {
        match self {
            Self::NoRepo => "no-repo",
            Self::TrackedClean => "tracked-clean",
            Self::TrackedDirty => "tracked-dirty",
            Self::Untracked => "untracked",
            Self::Ignored => "ignored",
            Self::Unknown => "unknown",
        }
    }
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

/// Hard cap on the script body handed to the judge, in bytes.
///
/// Over the cap the body is refused outright rather than truncated. A
/// truncated script is a lie by omission: the interesting line goes at byte
/// 17000 and the judge reads the first 16KB of a perfectly ordinary deploy
/// script. Refusing is honest and the judge treats blindness as a reason to
/// escalate.
pub const MAX_SCRIPT_BODY_BYTES: u64 = 16 * 1024;

/// Interpreters whose first non-flag argument is a script file to execute.
///
/// Deliberately the two-token form only — see [`script_body_target`].
pub const SCRIPT_INTERPRETERS: &[&str] = &[
    "bash", "sh", "zsh", "dash", "ksh", "python", "python3", "node", "ruby", "perl",
];

/// Substrings in an assignment's left-hand side that make its value a secret.
pub const SECRET_KEY_HINTS: &[&str] = &[
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "api_key",
    "credential",
    "private_key",
    "access_key",
    "auth",
];

/// Value shapes that are a secret regardless of what they are assigned to.
pub const SECRET_VALUE_PREFIXES: &[&str] = &[
    "sk-",
    "ghp_",
    "gho_",
    "ghs_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "AKIA",
    "ASIA",
    "-----BEGIN",
];

/// What happened when we tried to read the script a command executes.
///
/// Every variant but [`Included`](Self::Included) is a refusal, and a refusal
/// is recorded rather than silently dropped: the judge is told it is blind, and
/// the audit row carries the reason so the corpus can say which guardrail is
/// costing coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScriptBodyStatus {
    /// Content was read, redacted, and handed to the judge.
    Included,
    /// The path is not within the requesting agent's `file_policy` reach.
    ///
    /// **The anti-oracle guardrail.** The judge runs with daemon privileges, so
    /// without this the gate would read files the requesting agent cannot and
    /// summarise them into a verdict — a privilege-escalation oracle for any
    /// agent that can name a path in a command. `NoPolicy` lands here too, for
    /// the ANAI-190 reason: outside `file_policy` there is no reach to be
    /// inside of.
    OutsideReach,
    /// Git-ignored. The class that holds every `.env`, key and token on the box.
    Ignored,
    /// The git axis did not answer in budget, so "not ignored" is unproven.
    GitUnknown,
    /// Larger than [`MAX_SCRIPT_BODY_BYTES`].
    TooLarge,
    /// Not valid UTF-8, or carrying control bytes in a shape no script has.
    Binary,
    /// Missing, a directory, a symlink, or the read itself failed.
    Unreadable,
}

impl ScriptBodyStatus {
    /// Short token for prompts and audit rows.
    #[must_use]
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Included => "included",
            Self::OutsideReach => "outside-reach",
            Self::Ignored => "git-ignored",
            Self::GitUnknown => "git-unknown",
            Self::TooLarge => "too-large",
            Self::Binary => "binary",
            Self::Unreadable => "unreadable",
        }
    }

    #[must_use]
    pub fn included(self) -> bool {
        matches!(self, Self::Included)
    }
}

/// ANAI-206 item 1: the body of the one script a command executes.
///
/// This is the only place in the gate that reads file *bytes*, and it exists
/// because the alternative is worse. `bash ~/.openfang/scripts/deploy-local.sh`
/// is structurally a read of that script; item 2 stopped the control-plane
/// floor from short-circuiting on it, which is only sound if the judge can then
/// see what the script does. Without this the judge would be guessing at 82% of
/// the population it was built for.
///
/// Guardrails, all enforced before a byte is read (see the runtime half):
///
/// 1. **One path**, and it is the command's own script argument — never a path
///    the judge chooses.
/// 2. Never outside the requesting agent's `file_policy` reach.
/// 3. Never git-ignored, and never on an unproven git answer.
/// 4. [`MAX_SCRIPT_BODY_BYTES`], refused rather than truncated.
/// 5. Binary content refused.
/// 6. Secret shapes redacted on read.
/// 7. The audit row says whether content was consulted, and why not when it
///    was not.
///
/// # ANAI-206 item 3: the body's own paths
///
/// Items 1 and 2 together traded a deterministic floor for a model's reading
/// comprehension: item 2 stopped `bash ~/.openfang/scripts/foo.sh` from
/// short-circuiting on the theory that item 1 shows the judge what `foo.sh`
/// does — but what item 1 hands over is *prose*. The command's own paths get
/// the full [`PathFact`] treatment; the body's paths got nothing, so a script
/// containing `rm -rf ~/.openfang/agents/` reached a judge that had to read
/// that line and notice.
///
/// So the body is now mapped as well as shown: [`body_facts`](Self::body_facts)
/// carries the same facts for the paths named *inside* the script, and
/// [`writes_control_plane`](Self::writes_control_plane) restores the floor item
/// 2 removed — a control-plane **write** anywhere in the body escalates by
/// rule, while a mere read still reaches the judge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScriptBody {
    /// The token as the agent wrote it. Untrusted; rendered, never executed.
    pub raw: String,
    /// Absolute resolved form, when resolution succeeded.
    pub resolved: Option<String>,
    pub status: ScriptBodyStatus,
    /// Redacted, capped content. `Some` only when `status` is `Included`.
    pub content: Option<String>,
    /// How many secret-shaped spans were replaced.
    pub redactions: usize,
    /// What the git index said about the script file itself.
    ///
    /// Recorded for the audit row rather than for a decision: guardrail 3
    /// already refused on `Ignored` and `Unknown`, so anything that gets this
    /// far is one of the other four. Present so the corpus can count how much
    /// of the population sits outside any repository, where that guardrail is
    /// structurally inert.
    #[serde(default = "git_unknown")]
    pub git: GitFact,
    /// ANAI-206 item 3: facts for every path named *inside* the script.
    ///
    /// Empty when the body was not read. Computed by the daemon, not parsed by
    /// the judge — the judge is handed facts, not text to interpret.
    #[serde(default)]
    pub body_facts: Vec<PathFact>,
    /// The body named more than [`MAX_BODY_PATH_FACTS`] paths; the tail was
    /// dropped and the map is incomplete.
    #[serde(default)]
    pub body_truncated: bool,
    /// The body named at least one path that is a glob or an expansion.
    ///
    /// Common and load-bearing: real scripts are full of `"$HOME/..."` and
    /// `$REPO/...`, and we do not know what those name at gather time.
    #[serde(default)]
    pub body_unresolved: bool,
    /// ANAI-206 item 3: some line of the body **writes** the control plane.
    ///
    /// Wired to a floor predicate, so this short-circuits to `Escalate` exactly
    /// as the same write would on the command line. Scanned line by line rather
    /// than over the whole body, because
    /// [`cmd_norm::deny_variants`](crate::cmd_norm::deny_variants) truncates its
    /// input at 8KB and a 16KB script would otherwise hide a write past that
    /// boundary.
    #[serde(default)]
    pub writes_control_plane: bool,
}

/// `serde` default for [`ScriptBody::git`]: unknown, never a benign answer.
fn git_unknown() -> GitFact {
    GitFact::Unknown
}

impl ScriptBody {
    /// The block that goes inside the judge's `<script-body>` fence.
    ///
    /// A refusal renders as a stated refusal, not as an empty span: "we could
    /// not read it" and "it was empty" must not look the same to the judge.
    #[must_use]
    pub fn render(&self) -> String {
        let name = self.resolved.as_deref().unwrap_or(&self.raw);
        match (&self.content, self.status) {
            (Some(text), ScriptBodyStatus::Included) => {
                let mut out = format!("# {name}\n");
                if self.redactions > 0 {
                    out.push_str(&format!(
                        "# [{} secret-shaped span(s) redacted before this was shown]\n",
                        self.redactions
                    ));
                }
                out.push_str(text);
                out
            }
            _ => format!(
                "(not read: {} — {}. You are blind to what this command executes.)",
                name,
                self.status.as_token()
            ),
        }
    }

    /// ANAI-206 item 3: the daemon's map of what the script touches.
    ///
    /// Rendered **outside** the `<script-body>` fence on purpose. These are
    /// daemon-computed facts about attacker-influenced tokens; putting them
    /// inside the untrusted fence would let the script's own text forge lines
    /// that read as facts. The tokens themselves are still neutralised by the
    /// caller before they reach the prompt.
    ///
    /// Empty string when the body was not read: the refusal is already stated
    /// inside the fence, and a "paths: (none)" block next to it would read as a
    /// script that touches nothing.
    #[must_use]
    pub fn render_body_facts(&self) -> String {
        if !self.status.included() {
            return String::new();
        }
        let mut lines: Vec<String> = Vec::new();
        if self.writes_control_plane {
            lines.push("  [this script WRITES the OpenFang control plane]".to_string());
        }
        if self.body_facts.is_empty() {
            lines.push("  (no resolvable paths named inside the script)".to_string());
        } else {
            lines.extend(self.body_facts.iter().map(|f| format!("  {}", f.render())));
        }
        if self.body_truncated {
            lines.push(
                "  [...more paths named inside the script than shown; map is incomplete]"
                    .to_string(),
            );
        }
        if self.body_unresolved {
            lines.push(
                "  [at least one path inside the script was a glob or expansion and was not \
                 resolved]"
                    .to_string(),
            );
        }
        lines.join("\n")
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
    /// ANAI-206 item 1: the body of the script this command executes, when the
    /// command is a bare interpreter invocation against a single readable file.
    ///
    /// `#[serde(default)]` for the same reason the sheet itself carries one:
    /// rows written before ANAI-206 have no such field, and a `GateRequest`
    /// that fails to deserialize is a gate that fails *open* into the caller's
    /// error path.
    #[serde(default)]
    pub script_body: Option<ScriptBody>,
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
        // A script we could not read is a command whose effects we do not know,
        // whatever the metadata says. Strictly more conservative than
        // pre-ANAI-206 behaviour, where the same command was eligible on
        // metadata alone.
        if matches!(&self.script_body, Some(b) if !b.status.included()) {
            return false;
        }
        // ANAI-206 item 3. The same argument one level down: a script we read
        // but could not fully map is a script whose effects we do not know. The
        // body's paths are held to exactly the bar the command's own paths are.
        if let Some(body) = &self.script_body {
            if body.writes_control_plane
                || body.body_truncated
                || body.body_unresolved
                || !body.body_facts.iter().all(PathFact::suppress_eligible)
            {
                return false;
            }
        }
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
        if self.facts.is_empty() && !self.unresolved && self.script_body.is_none() {
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
        // The audit flag for "content was consulted". Present on every row the
        // feature touched, refusal included, so the corpus can price each
        // guardrail instead of guessing at it.
        if let Some(body) = &self.script_body {
            token.push_str(&format!(" script={}", body.status.as_token()));
            if body.redactions > 0 {
                token.push_str(&format!(" redacted={}", body.redactions));
            }
            // Guardrail 3's coverage, made countable: every script under
            // `~/.openfang/scripts/` answers `no-repo`, and there the guardrail
            // is inert.
            token.push_str(&format!(" script_git={}", body.git.as_token()));
            if body.status.included() {
                token.push_str(&format!(" body_n={}", body.body_facts.len()));
                if body.body_truncated {
                    token.push_str(" body_truncated");
                }
                if body.body_unresolved {
                    token.push_str(" body_unresolved");
                }
                if body.writes_control_plane {
                    token.push_str(" body_control_plane");
                }
            }
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

/// ANAI-206 item 3: pull the path-shaped tokens out of a *script body*.
///
/// Separate from [`extract_path_tokens`] rather than a reuse of it, for three
/// reasons that all come from the input being a file instead of a command line:
///
/// - Comments are stripped per line. A path named in a comment is not a path
///   the shell acts on, and stating it as fact would be stating something
///   false — the same rule the command itself is held to.
/// - Leading redirect and grouping punctuation is trimmed, so `>~/x` yields
///   `~/x` rather than a nonsense token.
/// - The cap is [`MAX_BODY_PATH_FACTS`], applied by the caller, and overflow is
///   reported rather than swallowed.
///
/// A token carrying an expansion (`"$HOME/x"`) is *not* a fact — we do not know
/// what it names — so it sets the unresolved flag instead. That is the common
/// case in real scripts and it is why item 3 buys the judge a map rather than a
/// guarantee.
#[must_use]
pub fn extract_body_path_tokens(body: &str) -> (Vec<String>, bool) {
    let mut seen: Vec<String> = Vec::new();
    let mut unresolved = false;
    let stripped = crate::gatekeeper::strip_shell_comments(body);
    for line in stripped.lines() {
        for token in line.split_whitespace() {
            let token =
                token.trim_matches(|c| matches!(c, '"' | '\'' | '(' | ')' | ';' | ',' | '&' | '|'));
            let token = token.trim_start_matches(['>', '<']);
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

/// The single path whose contents the judge is allowed to see, if any.
///
/// Guardrail 1, and the one that makes the rest defensible: the judge never
/// picks a path. This returns *the command's own script argument* or nothing.
///
/// Deliberately narrow, and it will say `None` on forms that are obviously
/// fine:
///
/// - Any shell separator, redirect, substitution or expansion in the command →
///   `None`. A read is only sound when we can attribute the whole command to
///   one interpreter and one file.
/// - A non-empty `inner` → `None`. `bash -c '...'` has no file to read, and the
///   text it runs is already in front of the judge inside the command fence.
/// - An interpreter *flag* in the second position → `None`. `bash -x foo.sh` is
///   not worth the parser it would take to be sure.
/// - A second token that is not path-shaped → `None`. `bash deploy.sh` (bare
///   name, resolved against the cwd) is a known coverage gap: `looks_like_path`
///   excludes it, so there is no [`PathFact`] to check reach against. Left as a
///   gap on purpose — the corpus can say whether it matters.
///
/// Tokens after the script path are the *script's* arguments and do not affect
/// which file is read, so they are ignored rather than disqualifying.
#[must_use]
pub fn script_body_target(command: &str, inner: &[String]) -> Option<String> {
    if !inner.is_empty() {
        return None;
    }
    if command.contains([';', '&', '|', '\n', '`', '(', ')', '<', '>', '$']) {
        return None;
    }
    let mut tokens = command.split_whitespace();
    let bin = tokens.next()?;
    let bin = bin.rsplit('/').next().unwrap_or(bin);
    if !SCRIPT_INTERPRETERS.contains(&bin) {
        return None;
    }
    let candidate = tokens.next()?.trim_matches(|c| c == '"' || c == '\'');
    if candidate.starts_with('-') {
        return None;
    }
    if !looks_like_path(candidate) || is_unresolvable(candidate) {
        return None;
    }
    Some(candidate.to_string())
}

/// Guardrail 6. Replace secret-shaped spans before the body reaches a model.
///
/// Two rules, both crude on purpose. This is not a secret scanner and does not
/// need to be: the file is already inside the agent's own `file_policy` reach
/// and is about to be executed with the agent's own authority, so the exposure
/// this closes is narrow — a credential that would otherwise be copied verbatim
/// into a model prompt and an audit chain. Over-redaction costs the judge a
/// little context; under-redaction costs a token.
///
/// 1. An assignment whose left-hand side reads like a secret loses its value.
/// 2. A token with a known credential prefix, or a long opaque run with no path
///    separator in it, is replaced wholesale.
///
/// Returns the redacted text and how many spans were replaced.
#[must_use]
pub fn redact_secrets(text: &str) -> (String, usize) {
    const REDACTED: &str = "[redacted]";
    let mut count = 0usize;
    let mut out: Vec<String> = Vec::new();

    for line in text.lines() {
        // Rule 1: assignment with a secret-shaped key.
        if let Some(eq) = line.find('=') {
            let lhs = &line[..eq];
            let key = lhs
                .rsplit(|c: char| c.is_whitespace())
                .next()
                .unwrap_or(lhs)
                .to_ascii_lowercase();
            if SECRET_KEY_HINTS.iter().any(|h| key.contains(h)) && !line[eq + 1..].trim().is_empty()
            {
                count += 1;
                out.push(format!("{}={REDACTED}", &line[..eq]));
                continue;
            }
        }
        // Rule 2: value shapes, token by token, whitespace preserved well
        // enough for a reader.
        let mut parts: Vec<String> = Vec::new();
        for token in line.split(' ') {
            if looks_secret(token) {
                count += 1;
                parts.push(REDACTED.to_string());
            } else {
                parts.push(token.to_string());
            }
        }
        out.push(parts.join(" "));
    }

    let mut joined = out.join("\n");
    if text.ends_with('\n') {
        joined.push('\n');
    }
    (joined, count)
}

/// True when a bare token is shaped like a credential.
fn looks_secret(token: &str) -> bool {
    let trimmed = token.trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ';');
    if trimmed.is_empty() {
        return false;
    }
    if SECRET_VALUE_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
        return true;
    }
    // A long opaque run. Paths are excluded by the `/` test, and a run with no
    // digit is almost always prose or an identifier.
    trimmed.len() >= 40
        && !trimmed.contains('/')
        && trimmed.chars().any(|c| c.is_ascii_digit())
        && trimmed.chars().any(|c| c.is_ascii_alphabetic())
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '=' || c == '_' || c == '-')
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
            script_body: None,
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
            script_body: None,
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
            script_body: None,
        };
        assert_eq!(sheet.as_log_token(), "n=2 rec=1 auth=1");
    }

    // -----------------------------------------------------------------------
    // ANAI-206 item 1: script body selection, redaction, and the audit flag.
    // -----------------------------------------------------------------------

    fn body(status: ScriptBodyStatus) -> ScriptBody {
        ScriptBody {
            raw: "~/.openfang/scripts/deploy-local.sh".into(),
            resolved: Some("/home/x/.openfang/scripts/deploy-local.sh".into()),
            status,
            content: matches!(status, ScriptBodyStatus::Included).then(|| "set -e\n".to_string()),
            redactions: 0,
            git: GitFact::NoRepo,
            body_facts: Vec::new(),
            body_truncated: false,
            body_unresolved: false,
            writes_control_plane: false,
        }
    }

    /// The motivating case: one interpreter, one script path, nothing else.
    #[test]
    fn a_bare_interpreter_invocation_names_its_script() {
        assert_eq!(
            script_body_target("bash ~/.openfang/scripts/deploy-local.sh", &[]),
            Some("~/.openfang/scripts/deploy-local.sh".to_string())
        );
        // ...and the script's own arguments do not change which file is read.
        assert_eq!(
            script_body_target("bash ./scripts/build.sh --dry-run", &[]),
            Some("./scripts/build.sh".to_string())
        );
        assert_eq!(
            script_body_target("/bin/sh /usr/local/bin/x.sh", &[]),
            Some("/usr/local/bin/x.sh".to_string())
        );
    }

    /// Guardrail 1. Anything the tokenizer cannot attribute to exactly one
    /// interpreter and one file reads nothing at all.
    #[test]
    fn only_the_unambiguous_two_token_form_is_read() {
        for cmd in [
            // inline code, not a file
            "bash -c \"rm ./x\"",
            // interpreter flags: not worth the parser
            "bash -x ./scripts/build.sh",
            // separators, redirects, substitution, expansion
            "bash ./a.sh && bash ./b.sh",
            "bash ./a.sh > /tmp/out",
            "bash $(echo ./a.sh)",
            "bash ./a-$VERSION.sh",
            // not an interpreter at all
            "cat ./scripts/build.sh",
            "rm -rf ~/.openfang/agents",
            // no argument
            "bash",
            // stdin, a glob, a bare name with no path shape
            "bash -",
            "bash ./scripts/*.sh",
            "bash deploy.sh",
        ] {
            assert_eq!(script_body_target(cmd, &[]), None, "{cmd}");
        }
    }

    /// A lifted inner command means the wrapper carried code, not a file.
    #[test]
    fn a_wrapper_with_inner_commands_is_never_read() {
        assert_eq!(
            script_body_target("bash ./a.sh", &["rm ./x".to_string()]),
            None
        );
    }

    // -----------------------------------------------------------------------
    // ANAI-206 item 3: the body's own paths.
    // -----------------------------------------------------------------------

    /// Comments are stripped, redirect punctuation is trimmed, and an
    /// expansion is recorded as blindness rather than invented as a fact.
    #[test]
    fn body_tokens_are_extracted_per_line() {
        let (tokens, unresolved) = extract_body_path_tokens(
            "#!/usr/bin/env bash\n\
             set -e\n\
             # touches ~/.openfang/agents/decoy.toml\n\
             cargo build >./out/log.txt\n\
             cp ./a.txt \"./b.txt\"\n\
             rm -rf \"$HOME/scratch\"\n",
        );
        assert!(tokens.contains(&"./out/log.txt".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"./a.txt".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"./b.txt".to_string()), "{tokens:?}");
        // A path named only in a comment is not a path the shell acts on.
        assert!(
            !tokens.iter().any(|t| t.contains("decoy")),
            "comment token survived: {tokens:?}"
        );
        // ...and `$HOME/scratch` is blindness, not a fact.
        assert!(unresolved);
    }

    /// The regression items 1 and 2 opened: a script whose body writes the
    /// substrate must never be suppress-eligible on the strength of tidy
    /// command-line metadata.
    #[test]
    fn a_body_that_writes_the_control_plane_poisons_the_sheet() {
        let mut hostile = body(ScriptBodyStatus::Included);
        hostile.writes_control_plane = true;
        let sheet = PathFactSheet {
            facts: vec![fact(
                PathExistence::File,
                GitFact::NoRepo,
                true,
                PathAuthority::Write,
            )],
            truncated: false,
            unresolved: false,
            script_body: Some(hostile),
        };
        assert!(!sheet.suppress_eligible());
        assert!(sheet.as_log_token().contains("body_control_plane"));
    }

    /// A body we mapped only partly is a body whose effects we do not know —
    /// the same rule the command's own paths are held to, one level down.
    #[test]
    fn a_partly_mapped_body_poisons_the_sheet() {
        let base = PathFactSheet {
            facts: vec![fact(
                PathExistence::File,
                GitFact::NoRepo,
                true,
                PathAuthority::Write,
            )],
            truncated: false,
            unresolved: false,
            script_body: Some(body(ScriptBodyStatus::Included)),
        };
        assert!(base.suppress_eligible());

        for mutate in [
            (|b: &mut ScriptBody| b.body_truncated = true) as fn(&mut ScriptBody),
            |b: &mut ScriptBody| b.body_unresolved = true,
            |b: &mut ScriptBody| {
                b.body_facts = vec![fact(
                    PathExistence::File,
                    GitFact::Ignored,
                    true,
                    PathAuthority::Write,
                )]
            },
        ] {
            let mut b = body(ScriptBodyStatus::Included);
            mutate(&mut b);
            let sheet = PathFactSheet {
                script_body: Some(b),
                ..base.clone()
            };
            assert!(!sheet.suppress_eligible());
        }
    }

    /// A refusal already states itself inside the fence; a "paths: (none)"
    /// block beside it would read as a script that touches nothing.
    #[test]
    fn a_refused_body_renders_no_fact_block() {
        assert_eq!(body(ScriptBodyStatus::OutsideReach).render_body_facts(), "");
        assert!(!body(ScriptBodyStatus::Included)
            .render_body_facts()
            .is_empty());
    }

    /// Guardrail 7. Every row the feature touched says so, refusal included —
    /// otherwise the corpus cannot price a guardrail that is costing coverage.
    #[test]
    fn the_audit_token_records_whether_content_was_consulted() {
        let sheet = PathFactSheet {
            facts: vec![fact(
                PathExistence::File,
                GitFact::NoRepo,
                true,
                PathAuthority::Write,
            )],
            truncated: false,
            unresolved: false,
            script_body: Some(body(ScriptBodyStatus::Included)),
        };
        assert_eq!(
            sheet.as_log_token(),
            "n=1 rec=1 auth=1 script=included script_git=no-repo body_n=0"
        );

        let refused = PathFactSheet {
            script_body: Some(body(ScriptBodyStatus::OutsideReach)),
            ..sheet.clone()
        };
        assert_eq!(
            refused.as_log_token(),
            "n=1 rec=1 auth=1 script=outside-reach script_git=no-repo"
        );

        let mut redacted_body = body(ScriptBodyStatus::Included);
        redacted_body.redactions = 3;
        let redacted = PathFactSheet {
            script_body: Some(redacted_body),
            ..sheet.clone()
        };
        assert!(redacted.as_log_token().contains("redacted=3"));
    }

    /// A script we could not read is a command whose effects we do not know,
    /// whatever the metadata says.
    #[test]
    fn an_unread_script_poisons_the_sheet() {
        let sheet = PathFactSheet {
            facts: vec![fact(
                PathExistence::File,
                GitFact::NoRepo,
                true,
                PathAuthority::Write,
            )],
            truncated: false,
            unresolved: false,
            script_body: Some(body(ScriptBodyStatus::Included)),
        };
        assert!(sheet.suppress_eligible());
        for status in [
            ScriptBodyStatus::OutsideReach,
            ScriptBodyStatus::Ignored,
            ScriptBodyStatus::GitUnknown,
            ScriptBodyStatus::TooLarge,
            ScriptBodyStatus::Binary,
            ScriptBodyStatus::Unreadable,
        ] {
            let blind = PathFactSheet {
                script_body: Some(body(status)),
                ..sheet.clone()
            };
            assert!(!blind.suppress_eligible(), "{}", status.as_token());
        }
    }

    /// "We could not read it" and "it was empty" must not look the same.
    #[test]
    fn a_refusal_renders_as_a_stated_refusal() {
        let rendered = body(ScriptBodyStatus::OutsideReach).render();
        assert!(rendered.contains("outside-reach"), "{rendered}");
        assert!(rendered.contains("blind"), "{rendered}");
        assert!(!rendered.trim().is_empty());
    }

    #[test]
    fn secret_assignments_lose_their_values() {
        let (out, n) = redact_secrets(
            "export ANTHROPIC_API_KEY=sk-ant-0123456789\n\
             DB_PASSWORD='hunter2'\n\
             echo hello\n",
        );
        assert_eq!(n, 2, "{out}");
        assert!(!out.contains("sk-ant-0123456789"), "{out}");
        assert!(!out.contains("hunter2"), "{out}");
        // ...and the rest of the script survives, or the judge learns nothing.
        assert!(out.contains("echo hello"), "{out}");
        assert!(out.contains("export ANTHROPIC_API_KEY=[redacted]"), "{out}");
    }

    #[test]
    fn bare_credential_shapes_are_redacted_without_an_assignment() {
        let (out, n) = redact_secrets("curl -H \"Bearer ghp_abcdefghijklmnop0123\" https://x/\n");
        assert_eq!(n, 1, "{out}");
        assert!(!out.contains("ghp_"), "{out}");
    }

    /// Over-redaction is cheap; nuking every long path is not. A path has a
    /// separator in it and must survive.
    #[test]
    fn long_paths_and_ordinary_script_text_survive_redaction() {
        let script = "cargo build --release --manifest-path /Users/x/GitHub/Repos/openfang/crates/openfang-runtime/Cargo.toml\n\
                      rsync -a ./target/release/openfang ~/.openfang/bin/openfang\n";
        let (out, n) = redact_secrets(script);
        assert_eq!(n, 0, "{out}");
        assert_eq!(out, script);
    }
}
