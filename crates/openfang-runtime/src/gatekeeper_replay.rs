//! ANAI-252: offline replay of the gatekeeper verdict corpus under two
//! proposed narrowings of `PathFactSheet::suppress_eligible`.
//!
//! **Test-only.** `lib.rs` declares this module behind `#[cfg(test)]`, so
//! nothing here reaches the shipped binary and no production predicate is
//! modified. The point is to answer one question before any predicate moves:
//! *what would the suppress rate have been?*
//!
//! Two narrowings are under consideration (see the ANAI-252 write-up):
//!
//! - **A — directory recoverability.** [`PathFact::recoverable`] answers `false`
//!   for `PathExistence::Dir` unconditionally ("never reason about a whole tree
//!   from one stat"). That is right for `rm -rf <dir>` and wrong for
//!   `cd <dir>`. Variant A treats a directory at `PathAuthority::Write` as
//!   recoverable. This is the **blunt** form — operand-insensitive, an upper
//!   bound on what an operand-sensitive rule could buy.
//! - **B — assignment expansions.** `extract_*_path_tokens` sets `unresolved`
//!   for any path-shaped token containing `$`. That fires on
//!   `export PATH="$HOME/.cargo/bin:$PATH"`, which is in essentially every
//!   script an agent writes, and an expansion in an *assignment* is not an
//!   expansion in an *operand*. Variant B skips assignment-shaped tokens.
//!
//! ## Why the replay is trustworthy, and where it is not
//!
//! Every fact is gathered by calling the **production** [`crate::path_facts::gather`]
//! on the command string recorded in the audit chain. The only re-implemented
//! logic is the two narrowings themselves and the tokenizer they modify, and
//! the tokenizer mirror is checked against production output on every row.
//!
//! Two self-checks run per row, and a row that fails either is reported as
//! `DRIFT` and excluded from the rates rather than silently counted:
//!
//! 1. `sheet.as_log_token()` must equal the token stored in the audit row. The
//!    filesystem has moved since these verdicts were written — scripts under
//!    `scratch/` were edited between runs, some were deleted — so this is the
//!    load-bearing check. It says "the box still looks the way it looked when
//!    this verdict was made."
//! 2. The mirrored tokenizer, run with the narrowing *disabled*, must reproduce
//!    production's `unresolved` / `body_unresolved` booleans exactly.
//!
//! Run it:
//!
//! ```text
//! OF_GK_REPLAY_CORPUS=<corpus.json> cargo test -p openfang-runtime \
//!     --lib gatekeeper_replay -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use openfang_types::agent::AgentManifest;
use openfang_types::config::FilePolicy;
use openfang_types::path_facts::{PathAuthority, PathExistence, PathFact, PathFactSheet};

/// One audit row, as dumped by the `sqlite3 -json` query in the write-up.
#[derive(Debug, serde::Deserialize)]
struct Row {
    seq: i64,
    ts: String,
    agent: String,
    outcome: String,
    /// `paths=[...] det=... det_disagree=...`, verbatim from `detail`.
    sheet: String,
    command: String,
}

impl Row {
    /// The contents of `paths=[...]`, which is exactly `as_log_token()`.
    fn stored_token(&self) -> Option<&str> {
        let start = self.sheet.find("paths=[")? + "paths=[".len();
        let rest = &self.sheet[start..];
        let end = rest.find(']')?;
        Some(&rest[..end])
    }

    fn model_suppressed(&self) -> bool {
        self.outcome.contains("suppress")
    }
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

fn workspace_root(agent: &str) -> PathBuf {
    home().join(".openfang").join("workspaces").join(agent)
}

/// The agent's own `file_policy`, resolved exactly as `agent_loop` resolves it:
/// `manifest.file_policy.as_ref()`, no global merge, because there is no global
/// `[file_policy]` block today.
fn file_policy(agent: &str) -> Option<FilePolicy> {
    let path = home()
        .join(".openfang")
        .join("agents")
        .join(agent)
        .join("agent.toml");
    let text = std::fs::read_to_string(path).ok()?;
    let manifest: AgentManifest = toml::from_str(&text).ok()?;
    manifest.file_policy
}

// ---------------------------------------------------------------------------
// Mirrored tokenizer. Kept byte-faithful to `openfang_types::path_facts` so the
// self-check below is meaningful; the *only* intentional difference is the
// `skip_assignments` knob.
// ---------------------------------------------------------------------------

fn looks_like_path(token: &str) -> bool {
    if token.is_empty() || token.starts_with('-') {
        return false;
    }
    token.contains('/') || token.starts_with('.') || token.starts_with('~')
}

fn is_unresolvable(token: &str) -> bool {
    token.contains('*')
        || token.contains('?')
        || token.contains('[')
        || token.contains('$')
        || token.contains('{')
}

/// True for `FOO=...`, `export FOO=...`'s payload, and the `declare`/`local`
/// forms — a token whose leading run is a shell identifier followed by `=`.
///
/// Deliberately does not try to be a shell parser. `PATH="$HOME/bin:$PATH"` is
/// an assignment; `--manifest-path=$X/Cargo.toml` is not (leading `-` already
/// fails `looks_like_path`, but the identifier rule rejects it independently).
fn is_assignment(token: &str) -> bool {
    let Some(eq) = token.find('=') else {
        return false;
    };
    if eq == 0 {
        return false;
    }
    let name = &token[..eq];
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty by the eq == 0 guard");
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Mirror of `extract_body_path_tokens`, with the narrowing knob.
fn extract_body(body: &str, skip_assignments: bool) -> (Vec<String>, bool) {
    let mut seen: Vec<String> = Vec::new();
    let mut unresolved = false;
    let stripped = openfang_types::gatekeeper::strip_shell_comments(body);
    for line in stripped.lines() {
        for token in line.split_whitespace() {
            let token =
                token.trim_matches(|c| matches!(c, '"' | '\'' | '(' | ')' | ';' | ',' | '&' | '|'));
            let token = token.trim_start_matches(['>', '<']);
            if !looks_like_path(token) {
                continue;
            }
            if skip_assignments && is_assignment(token) {
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

/// Mirror of `extract_path_tokens`, with the narrowing knob.
fn extract_command(command: &str, inner: &[String], skip_assignments: bool) -> (Vec<String>, bool) {
    let mut seen: Vec<String> = Vec::new();
    let mut unresolved = false;
    for source in std::iter::once(command).chain(inner.iter().map(String::as_str)) {
        for token in source.split_whitespace() {
            let token = token.trim_matches(|c| c == '"' || c == '\'');
            if !looks_like_path(token) {
                continue;
            }
            if skip_assignments && is_assignment(token) {
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

// ---------------------------------------------------------------------------
// Parameterised eligibility.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Variant {
    /// Narrowing A: a directory at `Write` counts as recoverable.
    dir_ok: bool,
    /// Narrowing B: assignment-shaped tokens do not set `unresolved`.
    assign_ok: bool,
}

const BASE: Variant = Variant {
    dir_ok: false,
    assign_ok: false,
};
const VAR_A: Variant = Variant {
    dir_ok: true,
    assign_ok: false,
};
const VAR_B: Variant = Variant {
    dir_ok: false,
    assign_ok: true,
};
const VAR_AB: Variant = Variant {
    dir_ok: true,
    assign_ok: true,
};

/// Whether narrowing A's directory exception is available for this command at
/// all. `false` when anything in the command or the script body is a
/// destructive verb — the case `PathExistence::Dir => false` was written for.
///
/// This is the difference between the blunt variant A and the operand-sensitive
/// A'. Still an approximation of true operand-sensitivity: it withholds the
/// exception for the whole command rather than for the specific path the verb
/// targets. Conservative in the right direction.
fn dir_exception_available(sheet: &PathFactSheet, bases: &[String], inner: &[String]) -> bool {
    if openfang_types::gatekeeper::has_destructive_verb(bases, inner) {
        return false;
    }
    if let Some(body) = &sheet.script_body {
        if body.destroys_substrate {
            return false;
        }
        if let Some(text) = &body.content {
            let stripped = openfang_types::gatekeeper::strip_shell_comments(text);
            for line in stripped.lines() {
                for token in line.split_whitespace() {
                    let bare = token.rsplit('/').next().unwrap_or(token);
                    if matches!(bare, "rm" | "rmdir" | "shred" | "dd" | "truncate" | "mv") {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn fact_eligible(fact: &PathFact, v: Variant) -> bool {
    let recoverable = if v.dir_ok
        && matches!(fact.existence, PathExistence::Dir)
        && matches!(fact.authority, PathAuthority::Write)
    {
        true
    } else {
        fact.recoverable()
    };
    recoverable && fact.authority.authorized()
}

/// Mirror of `PathFactSheet::suppress_eligible`, returning *why* rather than
/// just whether. The blocking reasons are collected in the production order so
/// the first entry is the one the real predicate would have short-circuited on.
fn blockers(sheet: &PathFactSheet, command: &str, inner: &[String], v: Variant) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    if let Some(body) = &sheet.script_body {
        if !body.status.included() {
            out.push(format!("script_{}", body.status.as_token()));
            return out;
        }
    }

    if let Some(body) = &sheet.script_body {
        if body.writes_control_plane {
            out.push("body_control_plane".into());
        }
        if body.destroys_substrate {
            out.push("body_substrate_destruction".into());
        }
        if body.body_truncated {
            out.push("body_truncated".into());
        }
        let body_unresolved = match (&body.content, v.assign_ok) {
            (Some(text), true) => extract_body(text, true).1,
            _ => body.body_unresolved,
        };
        if body_unresolved {
            out.push("body_unresolved".into());
        }
        if body.body_facts.is_empty() {
            out.push("body_facts_empty".into());
        }
        for fact in &body.body_facts {
            if !fact_eligible(fact, v) {
                out.push(format!(
                    "body_fact({} {:?} {:?})",
                    fact.raw, fact.existence, fact.authority
                ));
            }
        }
    }

    if sheet.truncated {
        out.push("truncated".into());
    }
    let unresolved = if v.assign_ok {
        extract_command(command, inner, true).1
    } else {
        sheet.unresolved
    };
    if unresolved {
        out.push("unresolved".into());
    }
    if sheet.facts.is_empty() {
        out.push("facts_empty".into());
    }
    for fact in &sheet.facts {
        if !fact_eligible(fact, v) {
            out.push(format!(
                "fact({} {:?} {:?})",
                fact.raw, fact.existence, fact.authority
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "reads the live audit corpus and the live filesystem; run explicitly"]
async fn replay_corpus_under_narrowings() {
    // Env var wins; the fallback is where the write-up's `sqlite3 -json` dump
    // lands, so the test is runnable without a shell that can set env vars.
    let corpus_path = std::env::var("OF_GK_REPLAY_CORPUS").unwrap_or_else(|_| {
        home()
            .join(".openfang/workspaces/openfang-alpha/output/anai-252-replay/corpus.json")
            .display()
            .to_string()
    });
    let text = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let rows: Vec<Row> = serde_json::from_str(&text).expect("parse corpus");

    let mut drift: Vec<(i64, String, String)> = Vec::new();
    let mut mirror_drift: Vec<i64> = Vec::new();
    let mut counts = [0usize; 4]; // base, A, B, AB
    let mut counts_op = [0usize; 2]; // A' (operand-sensitive), AB'
    let mut unsafe_blunt: Vec<i64> = Vec::new();
    let mut clean = 0usize;
    let mut first_blocker: std::collections::BTreeMap<String, usize> = Default::default();
    let mut lines: Vec<String> = Vec::new();

    for row in &rows {
        let ws = workspace_root(&row.agent);
        let policy = file_policy(&row.agent);

        let command = openfang_types::gatekeeper::strip_shell_comments(&row.command);
        let (bases, inner) = match crate::subprocess_sandbox::collect_command_bases(&command) {
            Ok(extracted) => (extracted.bases, extracted.inner),
            Err(_) => (Vec::new(), Vec::new()),
        };

        let sheet =
            crate::path_facts::gather(&command, &inner, Some(ws.as_path()), policy.as_ref()).await;

        // Self-check 1: the box still looks the way it looked at verdict time.
        let replayed = sheet.as_log_token();
        let stored = row.stored_token().unwrap_or("<none>");
        if replayed != stored {
            drift.push((row.seq, stored.to_string(), replayed));
            continue;
        }

        // Self-check 2: the mirrored tokenizer reproduces production with the
        // narrowing switched off.
        let mirror_ok = extract_command(&command, &inner, false).1 == sheet.unresolved
            && sheet.script_body.as_ref().is_none_or(|b| {
                b.content
                    .as_ref()
                    .is_none_or(|t| extract_body(t, false).1 == b.body_unresolved)
            });
        if !mirror_ok {
            mirror_drift.push(row.seq);
            continue;
        }

        clean += 1;

        let results = [
            blockers(&sheet, &command, &inner, BASE),
            blockers(&sheet, &command, &inner, VAR_A),
            blockers(&sheet, &command, &inner, VAR_B),
            blockers(&sheet, &command, &inner, VAR_AB),
        ];
        for (i, blocked) in results.iter().enumerate() {
            if blocked.is_empty() {
                counts[i] += 1;
            }
        }

        // A' / AB': the same narrowings, with the directory exception withheld
        // whenever a destructive verb is in scope.
        let dir_avail = dir_exception_available(&sheet, &bases, &inner);
        let op_a = if dir_avail { VAR_A } else { BASE };
        let op_ab = if dir_avail { VAR_AB } else { VAR_B };
        let a_prime = blockers(&sheet, &command, &inner, op_a).is_empty();
        let ab_prime = blockers(&sheet, &command, &inner, op_ab).is_empty();
        if a_prime {
            counts_op[0] += 1;
        }
        if ab_prime {
            counts_op[1] += 1;
        }
        // The rows blunt A would suppress and operand-sensitive A' would not:
        // every one of these is a destructive command the blunt rule waves
        // through.
        if results[1].is_empty() && !a_prime {
            unsafe_blunt.push(row.seq);
        }

        // Production agreement check on the baseline arm.
        assert_eq!(
            results[0].is_empty(),
            sheet.suppress_eligible(),
            "seq {}: mirrored baseline disagrees with production suppress_eligible",
            row.seq
        );

        *first_blocker
            .entry(
                results[0]
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "<eligible>".into()),
            )
            .or_default() += 1;

        lines.push(format!(
            "seq={} {} agent={} model={} base={} A={} B={} AB={} blocked={:?}\n    cmd={}",
            row.seq,
            &row.ts[..19],
            row.agent,
            if row.model_suppressed() {
                "suppress"
            } else {
                "escalate"
            },
            results[0].is_empty(),
            results[1].is_empty(),
            results[2].is_empty(),
            results[3].is_empty(),
            results[0],
            row.command,
        ));
    }

    println!("\n=== ANAI-252 replay ===");
    println!("corpus rows              : {}", rows.len());
    println!("replayed clean           : {clean}");
    println!("dropped (fs drift)       : {}", drift.len());
    println!("dropped (mirror drift)   : {}", mirror_drift.len());
    println!("\n-- counterfactual det_eligible over the clean rows --");
    let pct = |n: usize| {
        if clean == 0 {
            0.0
        } else {
            100.0 * n as f64 / clean as f64
        }
    };
    println!(
        "baseline (shipped)       : {:>3}  ({:.1}%)",
        counts[0],
        pct(counts[0])
    );
    println!(
        "A  dir@write recoverable : {:>3}  ({:.1}%)",
        counts[1],
        pct(counts[1])
    );
    println!(
        "B  assignment expansions : {:>3}  ({:.1}%)",
        counts[2],
        pct(counts[2])
    );
    println!(
        "AB both                  : {:>3}  ({:.1}%)",
        counts[3],
        pct(counts[3])
    );
    println!(
        "A' operand-sensitive     : {:>3}  ({:.1}%)",
        counts_op[0],
        pct(counts_op[0])
    );
    println!(
        "A'B both, operand-safe   : {:>3}  ({:.1}%)",
        counts_op[1],
        pct(counts_op[1])
    );
    println!("\nrows blunt A suppresses that A' refuses (destructive): {unsafe_blunt:?}");

    println!("\n-- first blocking predicate, baseline --");
    let mut ranked: Vec<_> = first_blocker.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, n) in ranked {
        println!("{n:>3}  {reason}");
    }

    if !drift.is_empty() {
        println!("\n-- DRIFT: replayed token != stored token (excluded) --");
        for (seq, stored, replayed) in &drift {
            println!("seq={seq}\n  stored   = {stored}\n  replayed = {replayed}");
        }
    }
    if !mirror_drift.is_empty() {
        println!("\n-- MIRROR DRIFT (excluded): {mirror_drift:?}");
    }

    println!("\n-- per-row --");
    for line in &lines {
        println!("{line}");
    }
}
