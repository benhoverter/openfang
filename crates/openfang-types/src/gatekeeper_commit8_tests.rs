//! ANAI-206 commit 8: G1 (the hard floor's path matching), plus two promotions
//! to hard that Ben called and security asked for — a write under
//! `~/.openfang/agents/` and a write to `~/.openfang/config.toml`.
//!
//! What is deliberately *not* here: G2 and G4, the two body-scanner findings.
//! Ben's call, and it is the same call as commit 6's — inside-the-script is the
//! judge's problem. The body scanner is not a shell parser and will lose every
//! race against one; the judge reads the body and decides. The body predicates
//! stay as demoted facts that put the command in front of a reasoner, which is
//! what they are good at.

use super::*;

// ---------------------------------------------------------------------------
// G1 — one character used to defeat the hard floor
// ---------------------------------------------------------------------------

/// The critical one. `~/.openfang/*` deletes `agents/`, `daemon/` and `data/`
/// in a single command, and the pre-commit-8 predicate did not see it: the tail
/// had to `strip_prefix` a subtree name or be empty, and `*` is neither.
#[test]
fn the_root_glob_is_substrate() {
    assert!(names_substrate("~/.openfang/*"));
    assert!(destroys_substrate("rm -rf ~/.openfang/*"));
}

/// A partial glob resolves without the filesystem no better than a total one,
/// so it is treated as naming whatever it might name. Deliberately generous in
/// the hard direction — this is the one predicate where over-reach costs an
/// escalation and under-reach costs the substrate.
#[test]
fn a_partial_glob_under_the_root_is_substrate() {
    assert!(names_substrate("~/.openfang/age*"));
    assert!(names_substrate("~/.openfang/?gents"));
}

/// `.` and `//` are no-ops to every shell and to `PathBuf`, and were not to us.
#[test]
fn redundant_separators_and_dots_resolve() {
    assert!(names_substrate("~/.openfang/./agents"));
    assert!(names_substrate("~/.openfang//agents"));
    assert!(names_substrate("~/.openfang/././/daemon/"));
    assert!(destroys_substrate("rm -rf ~/.openfang/./agents"));
}

/// `..` pops, so a path that walks out through a non-substrate subtree and back
/// in is still the substrate.
#[test]
fn dotdot_pops_back_into_the_substrate() {
    assert!(names_substrate("~/.openfang/scripts/../agents"));
    assert!(destroys_substrate("rm -rf ~/.openfang/scripts/../data"));
}

/// And a path that pops *past* the root has left our jurisdiction: that is the
/// home directory, not the control plane.
#[test]
fn dotdot_past_the_root_is_not_the_substrate() {
    assert!(!names_substrate("~/.openfang/.."));
    assert!(!names_substrate("~/.openfang/agents/../.."));
}

/// The boundary behaviour commit 6 shipped has to survive the rewrite.
#[test]
fn g1_does_not_widen_the_floor() {
    assert!(!names_substrate("~/.openfangx/agents"));
    assert!(!names_substrate("~/.openfang.tar.gz"));
    assert!(!names_substrate("~/.openfang/agents-archive"));
    assert!(!names_substrate("~/.openfang/scripts/deploy-local.sh"));
    assert!(!names_substrate("~/.openfang/config.toml"));
    assert!(names_substrate("~/.openfang"));
    assert!(names_substrate("~/.openfang/"));
    assert!(names_substrate("~/.openfang/."));
}

// ---------------------------------------------------------------------------
// agents/** — writing another agent's `allowed_commands`
// ---------------------------------------------------------------------------

#[test]
fn writing_an_agent_manifest_is_hard() {
    assert!(writes_agent_config(
        "cp ./agent.toml ~/.openfang/agents/openfang-tools/agent.toml"
    ));
    assert!(writes_agent_config(
        "sed -i 's/allowlist/full/' ~/.openfang/agents/openfang-alpha/agent.toml"
    ));
    assert!(writes_agent_config("tee ~/.openfang/agents/x.toml"));
    assert!(writes_agent_config("mv a.toml ~/.openfang/agents"));
}

/// Reads stay demoted. This is the traffic Ben flagged as ordinary, and the
/// judge has the whole command plus the fact sheet to decide it.
#[test]
fn reading_an_agent_manifest_is_not_hard() {
    assert!(!writes_agent_config(
        "grep -rn model ~/.openfang/agents/openfang-alpha/agent.toml"
    ));
    assert!(!writes_agent_config("cat ~/.openfang/agents/x/agent.toml"));
    assert!(!writes_agent_config("ls ~/.openfang/agents"));
    // And it does not fire the demoted write flag either — since item 2, reads
    // of the control plane fall through *because* the judge can see the whole
    // command. The path is still named, so it lands in the fact sheet.
    assert!(!touches_control_plane("ls ~/.openfang/agents"));
    assert!(names_control_plane("ls ~/.openfang/agents"));
}

/// Boundary: `agents-archive` is not `agents`.
#[test]
fn agent_config_is_component_bounded() {
    assert!(!writes_agent_config(
        "tee ~/.openfang/agents-archive/x.toml"
    ));
}

// ---------------------------------------------------------------------------
// config.toml — the switch that decides whether any of this runs
// ---------------------------------------------------------------------------

#[test]
fn writing_the_runtime_config_is_hard() {
    assert!(writes_runtime_config("tee ~/.openfang/config.toml"));
    assert!(writes_runtime_config(
        "sed -i 's/enabled = true/enabled = false/' ~/.openfang/config.toml"
    ));
    assert!(writes_runtime_config("cp ./c.toml ~/.openfang/config.toml"));
}

#[test]
fn reading_the_runtime_config_is_not_hard() {
    assert!(!writes_runtime_config("cat ~/.openfang/config.toml"));
    assert!(!writes_runtime_config(
        "sed -n '1,20p' ~/.openfang/config.toml"
    ));
}

/// A backup file is not the live config.
#[test]
fn runtime_config_is_component_bounded() {
    assert!(!writes_runtime_config("tee ~/.openfang/config.toml.bak"));
    assert!(!writes_runtime_config("tee ~/.openfang/config-old.toml"));
}

/// Both new predicates run on `deny_variants`, like every other command-line
/// predicate. One `""` in the middle of a path is the shape that defeats a raw
/// `contains`, and it is the shape security found four of in commit 7.
#[test]
fn the_new_predicates_read_deny_variants() {
    assert!(writes_agent_config(
        "cp ./a.toml ~/.open\"\"fang/agents/b.toml"
    ));
    assert!(writes_runtime_config("tee ~/.open\"\"fang/config.toml"));
}

// ---------------------------------------------------------------------------
// The floor itself
// ---------------------------------------------------------------------------

/// Both promotions have to reach `GateFlags::hard`, not just `GateFlags::any`.
/// A flag that fires and does not gate is a fact, and these two are not facts.
#[test]
fn both_promotions_are_hard() {
    let agents = GateFlags {
        agent_config_write: true,
        ..Default::default()
    };
    let config = GateFlags {
        runtime_config_write: true,
        ..Default::default()
    };
    for (name, flags) in [
        ("agent_config_write", agents),
        ("runtime_config_write", config),
    ] {
        assert!(flags.hard(), "{name} must bypass the judge");
        assert!(flags.any(), "{name} must count as noticed");
        assert!(
            flags.as_log_string().contains(name),
            "{name} must be distinguishable in the audit row"
        );
    }
}

/// The three control-plane writes stay separate flags on purpose: same
/// severity, different event. Collapsing them would make the audit corpus
/// unable to answer "how often does an agent try to edit another agent".
#[test]
fn the_three_control_writes_stay_distinguishable() {
    let row = GateFlags {
        policy_self_modification: true,
        agent_config_write: true,
        runtime_config_write: true,
        ..Default::default()
    }
    .as_log_string();
    assert!(row.contains("policy_self_modification"));
    assert!(row.contains("agent_config_write"));
    assert!(row.contains("runtime_config_write"));
}
