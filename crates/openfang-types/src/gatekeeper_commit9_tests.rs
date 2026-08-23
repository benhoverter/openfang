//! ANAI-206 commit 9: the commit-6 review's two floor regressions (C6-1, C6-3)
//! and the two prompt defects (C6-4, C6-5).
//!
//! The through-line, and it is about how the predicates were written rather
//! than about any one of them: three commits running, the hard floor was
//! written to the *examples* in the conversation that produced it rather than
//! to the class those examples belong to. `substrate_destruction` was `rm` plus
//! a recursive flag because the two commands on the table were `rm -rf
//! ~/.openfang` and `rm -rf ~/.openfang/agents`. `policy_self_modification` was
//! command-line-only because the example was a command line. Both left the
//! floor lower than the pre-206 `flags.any()` baseline for a whole class of
//! substitute — which is the one direction commit 6 promised it would not move.

use super::*;

// ---------------------------------------------------------------------------
// C6-1 — the destruction class, not the `rm` example
// ---------------------------------------------------------------------------

/// Security's five substitutes, each of which cleared the hard floor at
/// `d531fae` and each of which escalated by rule before commit 6.
///
/// `mv` is the one that matters most: it is a complete wipe of the tree, it
/// needs no flag to take a directory, and it is fewer keystrokes than the form
/// the predicate was written for.
#[test]
fn the_substitutes_for_rm_rf_are_substrate_destruction() {
    assert!(destroys_substrate("mv ~/.openfang/agents /tmp/x"));
    assert!(destroys_substrate("find ~/.openfang/agents -delete"));
    assert!(destroys_substrate("chmod -R 000 ~/.openfang/daemon"));
    assert!(destroys_substrate(
        "truncate -s 0 ~/.openfang/data/audit.db"
    ));
    assert!(destroys_substrate(
        "rsync -a --delete /tmp/empty/ ~/.openfang/agents/"
    ));
}

/// The original form still fires, unchanged. A broadening that quietly drops
/// the case it was broadening from is not a broadening.
#[test]
fn the_original_recursive_removal_still_fires() {
    assert!(destroys_substrate("rm -rf ~/.openfang"));
    assert!(destroys_substrate("rm -rf ~/.openfang/agents"));
    assert!(destroys_substrate(
        "rm --recursive --force ~/.openfang/data"
    ));
    assert!(destroys_substrate("rm -rf ~/.openfang/*"));
}

/// `find` alone is a read. The removal has to be in the same segment, and
/// `-exec` only counts when what it execs removes.
#[test]
fn find_without_a_removal_is_not_destruction() {
    assert!(!destroys_substrate(
        "find ~/.openfang/agents -name '*.toml'"
    ));
    assert!(!destroys_substrate(
        "find ~/.openfang/agents -exec cat {} ;"
    ));
    assert!(destroys_substrate(
        "find ~/.openfang/agents -exec rm -f {} ;"
    ));
}

/// `rsync` without `--delete` mirrors; it does not remove.
#[test]
fn rsync_without_delete_is_not_destruction() {
    assert!(!destroys_substrate(
        "rsync -a ~/.openfang/agents/ /tmp/bak/"
    ));
}

/// A recursive flag on `chmod` is the whole difference: `chmod 644` on one file
/// under the substrate is an ordinary write that the judge should read, and
/// `chmod -R 000` on the tree is the thing no reading could excuse.
#[test]
fn chmod_needs_the_recursive_flag() {
    assert!(!destroys_substrate("chmod 644 ~/.openfang/daemon/plist"));
    assert!(destroys_substrate("chmod -R 000 ~/.openfang/daemon"));
}

/// `dd` reading the substrate into a backup elsewhere is a *read*, and the
/// whole point of commit 6 is that reads reach the judge. Only the `of=`
/// direction destroys.
#[test]
fn dd_only_destroys_in_the_of_direction() {
    assert!(!destroys_substrate(
        "dd if=~/.openfang/data/audit.db of=/tmp/backup.db"
    ));
    assert!(destroys_substrate(
        "dd if=/dev/zero of=~/.openfang/data/audit.db"
    ));
}

/// The class boundary, stated as a test so the next broadening has to argue
/// with it. These all change the substrate and all reach the judge with
/// `mutation` / `control_plane` on the sheet. Hard is not "dangerous", it is
/// "no reading of the evidence could make this fine".
#[test]
fn writing_the_substrate_is_not_destroying_it() {
    assert!(!destroys_substrate(
        "cp ./new.toml ~/.openfang/daemon/x.toml"
    ));
    assert!(!destroys_substrate("tee ~/.openfang/daemon/x.toml"));
    assert!(!destroys_substrate("sed -i 's/a/b/' ~/.openfang/daemon/x"));
    // The declared omission: a redirect target cannot be attributed from a
    // whitespace split, and keying on a bare `>` would send this *read* past
    // the judge.
    assert!(!destroys_substrate(
        "cat ~/.openfang/agents/a/agent.toml > /tmp/x"
    ));
}

/// Off-substrate stays off-substrate, at every verb in the widened class. This
/// is the over-fire direction, and it is the expensive one: every false hit
/// here is a command the judge never gets to suppress.
#[test]
fn the_wider_verb_set_does_not_widen_the_target_set() {
    assert!(!destroys_substrate("mv ./build ./build.old"));
    assert!(!destroys_substrate("truncate -s 0 ./logs/run.log"));
    assert!(!destroys_substrate("chmod -R 755 ./scripts"));
    assert!(!destroys_substrate("find ./target -delete"));
    // `scripts/` and `config.toml` are deliberately not substrate — commit 6's
    // call, unchanged here.
    assert!(!destroys_substrate("mv ~/.openfang/scripts /tmp/x"));
    assert!(!destroys_substrate("truncate -s 0 ~/.openfang/config.toml"));
}

/// The body variant inherits the widened class for free, because it is the same
/// segment predicate one level down. Pinned because "for free" is exactly the
/// kind of claim that stops being true.
#[test]
fn the_body_scanner_inherits_the_wider_class() {
    assert!(body_destroys_substrate(
        "set -e\nmv ~/.openfang/agents /tmp/x\n"
    ));
    assert!(body_destroys_substrate(
        "echo hi\ntruncate -s 0 ~/.openfang/data/audit.db\necho bye\n"
    ));
    // And the `cd`-relative form, which commit 7's tracker feeds.
    assert!(body_destroys_substrate(
        "cd ~/.openfang/agents\nmv * /tmp/x\n"
    ));
}

// ---------------------------------------------------------------------------
// C6-3 — the other three hard flags stop at the command line
// ---------------------------------------------------------------------------

/// The asymmetry commit 6 created and commit 8 documented as intentional:
/// `substrate_destruction` grew a body variant on the ground that a hard flag
/// scoped to the command line has a one-line bypass, and the other three did
/// not. Same bypass, same fix.
#[test]
fn the_hard_write_flags_now_read_the_body() {
    assert!(body_writes_gatekeeper_policy(
        "set -e\ncp ./mine.md ~/.openfang/gatekeeper.md\n"
    ));
    assert!(body_writes_agent_config(
        "echo start\ntee ~/.openfang/agents/openfang-alpha/agent.toml\n"
    ));
    assert!(body_writes_runtime_config(
        "sed -i 's/enabled = true/enabled = false/' ~/.openfang/config.toml\n"
    ));
}

/// Reads stay demoted one level down, exactly as they do on the command line.
/// `grep -rn model ~/.openfang/agents/` inside a deploy script is ordinary
/// fleet traffic and must keep reaching the judge.
#[test]
fn reading_those_paths_in_a_body_is_not_a_write() {
    assert!(!body_writes_agent_config(
        "grep -rn model ~/.openfang/agents/\n"
    ));
    assert!(!body_writes_runtime_config("cat ~/.openfang/config.toml\n"));
    assert!(!body_writes_gatekeeper_policy(
        "diff ./mine.md ~/.openfang/gatekeeper.md\n"
    ));
}

/// The three splitting shapes the body scanner already handles for
/// `writes_control_plane`, applied to the new predicates — a continuation, a
/// heredoc payload, and an opaque interpreter on a line that names the target.
#[test]
fn the_body_write_predicates_fold_like_the_others() {
    assert!(body_writes_agent_config(
        "tee \\\n  ~/.openfang/agents/a/agent.toml\n"
    ));
    assert!(body_writes_runtime_config(
        "cp $(cat <<'EOF'\n~/.openfang/config.toml\nEOF\n) /tmp/x\n"
    ));
    assert!(body_writes_gatekeeper_policy(
        "python3 -c 'open(sys.argv[1],\"w\")' ~/.openfang/gatekeeper.md\n"
    ));
}

/// Boundary behaviour is the command-line predicate's, unchanged: these run
/// through the same `writes_where`, so `config.toml.bak` is still a backup file
/// and a same-named file elsewhere is still not the control plane.
#[test]
fn the_body_write_predicates_keep_the_command_line_boundaries() {
    assert!(!body_writes_runtime_config(
        "tee ~/.openfang/config.toml.bak\n"
    ));
    assert!(!body_writes_gatekeeper_policy("tee ./docs/gatekeeper.md\n"));
}

// ---------------------------------------------------------------------------
// C6-4 / C6-5 — the prompt
// ---------------------------------------------------------------------------

/// Minimal request; the system prompt is a function of `policy` alone, so
/// nothing else here is load-bearing.
fn prompt_request() -> GateRequest {
    GateRequest {
        agent_name: "openfang-alpha".into(),
        workspace_root: None,
        command: "rm ./scratch/tmp-patch.sh".into(),
        bases: vec!["rm".into()],
        inner: vec![],
        safe_bins: vec![],
        trusted_commands: vec![],
        allowed_commands: vec!["rm".into()],
        flags: GateFlags::default(),
        policy: DEFAULT_POLICY.to_string(),
        path_facts: crate::path_facts::PathFactSheet::default(),
    }
}

/// C6-4. Rule 4's SUPPRESS was universally quantified over the path set, which
/// an **empty** set satisfies vacuously — while the code's own
/// `suppress_eligible` requires `!facts.is_empty()`. The prompt and the code
/// disagreed about the single most consequential case, and the prompt is the
/// one that will carry it when the fast path turns on.
#[test]
fn the_prompt_says_an_empty_sheet_is_not_a_suppression() {
    let sys = prompt_request().system_prompt();
    assert!(sys.contains("SUPPRESS requires evidence, not the absence of it"));
    assert!(sys.contains("names at least one path"));
    assert!(sys.contains("An empty path map is not a clean bill of health"));
}

/// C6-5. Rule 3 named six flag tokens; the judge can be shown ten. A demoted
/// flag arriving as undefined vocabulary is a demotion that bought nothing —
/// the judge either ignores it or invents a meaning for it.
#[test]
fn every_demoted_flag_the_judge_can_see_is_defined_in_the_prompt() {
    let sys = prompt_request().system_prompt();
    let demoted = GateFlags {
        touches_control_plane: true,
        destructive_verb: true,
        redirect_outside_workspace: true,
        network_binary: true,
        mutation_verb: true,
        egress_verb: true,
        opaque_execution: true,
        writes_outside_workspace: true,
        script_body_control_plane: true,
        // `script_body_blind` is deliberately absent: it reads like a demoted
        // flag but `hard()` lists it, and the runtime only sets it when the
        // command names the control plane. The judge is never shown it, so rule
        // 3 must not define it either — vocabulary for a flag that cannot
        // arrive is worse than no vocabulary.
        ..Default::default()
    };
    assert!(
        !demoted.hard(),
        "this fixture must contain only demoted flags"
    );
    for token in demoted.as_log_string().split('+') {
        assert!(
            sys.contains(&format!("`{token}`")),
            "rule 3 never defines `{token}`, but the judge is shown it"
        );
    }
}
