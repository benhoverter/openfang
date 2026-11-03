//! Security must-fix G1 for `feat/inject-rules-md`: a write to
//! `~/.openfang/RULES.md` is hard.
//!
//! That branch makes the daemon re-read RULES.md into every agent's prompt each
//! turn, so one write steers the fleet. `touches_control_plane` already fired on
//! it, but ANAI-206 c6 demoted that flag to the judge; this is the commit-8
//! shape (`runtime_config_write`) with a new subject.

use super::*;

// ---------------------------------------------------------------------------
// Writes: hard
// ---------------------------------------------------------------------------

#[test]
fn tee_writes_are_hard() {
    assert!(writes_fleet_rules("tee ~/.openfang/RULES.md"));
    assert!(writes_fleet_rules("tee -a /Users/rlyeh/.openfang/RULES.md"));
}

#[test]
fn redirect_writes_are_hard() {
    assert!(writes_fleet_rules("echo x > ~/.openfang/RULES.md"));
    assert!(writes_fleet_rules("echo x >> ~/.openfang/RULES.md"));
    assert!(writes_fleet_rules("cat x >| ~/.openfang/RULES.md"));
}

#[test]
fn cp_and_mv_writes_are_hard() {
    assert!(writes_fleet_rules("cp ./r.md ~/.openfang/RULES.md"));
    assert!(writes_fleet_rules("mv ./r.md ~/.openfang/RULES.md"));
    // Moving the live file away deletes the fleet's rules: also a write.
    assert!(writes_fleet_rules("mv ~/.openfang/RULES.md /tmp/x"));
}

#[test]
fn sed_in_place_is_hard() {
    assert!(writes_fleet_rules(
        "sed -i '' 's/Never/Always/' ~/.openfang/RULES.md"
    ));
}

/// The destination is the bare root and the file name rides on the source:
/// no token names `~/.openfang/RULES.md`, so `writes_where` alone missed it.
#[test]
fn depositing_into_the_root_is_hard() {
    assert!(writes_fleet_rules("cp ./RULES.md ~/.openfang/"));
    assert!(writes_fleet_rules("cp -t ~/.openfang ./RULES.md"));
    assert!(writes_fleet_rules("rsync -a ./stage/ ~/.openfang/"));
    assert!(writes_fleet_rules("tar -xzf b.tgz -C ~/.openfang"));
    // A non-deposit verb on the root is not this flag's business.
    assert!(!writes_fleet_rules("ls ~/.openfang"));
    // A deposit into a named subtree is not the root.
    assert!(!writes_fleet_rules("cp ./x.md ~/.openfang/shared/"));
}

/// A root glob expands, in a body, to whatever is there, RULES.md included.
#[test]
fn a_root_glob_write_is_hard() {
    assert!(writes_fleet_rules("tee ~/.openfang/R*"));
    assert!(writes_fleet_rules("sed -i '' s/a/b/ ~/.openfang/*.md"));
}

#[test]
fn deny_variants_are_read() {
    assert!(writes_fleet_rules("tee ~/.open\"\"fang/RULES.md"));
}

// ---------------------------------------------------------------------------
// Reads: not hard
// ---------------------------------------------------------------------------

#[test]
fn reads_are_not_hard() {
    assert!(!writes_fleet_rules("cat ~/.openfang/RULES.md"));
    assert!(!writes_fleet_rules("grep -n Never ~/.openfang/RULES.md"));
    assert!(!writes_fleet_rules("sed -n '1,20p' ~/.openfang/RULES.md"));
    assert!(!writes_fleet_rules("wc -l ~/.openfang/RULES.md"));
    // Still named, so it lands in front of the judge as a control-plane fact.
    assert!(names_control_plane("cat ~/.openfang/RULES.md"));
}

// ---------------------------------------------------------------------------
// Boundaries
// ---------------------------------------------------------------------------

#[test]
fn backups_are_not_hard() {
    assert!(!names_fleet_rules(
        "~/.openfang/rules.md.bak-20261102-tables"
    ));
    assert!(!writes_fleet_rules(
        "cp ./r.md ~/.openfang/RULES.md.bak-20261102-tables"
    ));
    assert!(!writes_fleet_rules("tee ~/.openfang/RULES.mdx"));
    assert!(!writes_fleet_rules("tee ~/.openfang/RULES-old.md"));
}

/// A workspace's own RULES.md is not the fleet's.
#[test]
fn a_nested_rules_md_is_not_hard() {
    assert!(!writes_fleet_rules("tee ~/.openfang/workspaces/a/RULES.md"));
    assert!(!writes_fleet_rules("tee ./RULES.md"));
}

/// APFS is case-insensitive: `rules.MD` opens the live file.
#[test]
fn case_variants_are_hard() {
    assert!(writes_fleet_rules("tee ~/.openfang/rules.MD"));
    assert!(writes_fleet_rules("cp x ~/.OpenFang/Rules.md"));
}

// ---------------------------------------------------------------------------
// Script bodies
// ---------------------------------------------------------------------------

#[test]
fn body_writes_are_flagged() {
    assert!(body_writes_fleet_rules(
        "#!/bin/bash\nset -e\necho 'new rule' >> ~/.openfang/RULES.md\n"
    ));
    assert!(body_writes_fleet_rules(
        "#!/bin/bash\ncp ./RULES.md ~/.openfang/\n"
    ));
    assert!(body_writes_fleet_rules(
        "#!/bin/bash\ncat <<'EOF' > ~/.openfang/RULES.md\nbe evil\nEOF\n"
    ));
}

#[test]
fn body_reads_are_not_flagged() {
    assert!(!body_writes_fleet_rules(
        "#!/bin/bash\ngrep -c Never ~/.openfang/RULES.md\n"
    ));
    assert!(!body_writes_fleet_rules(
        "#!/bin/bash\ntee ~/.openfang/RULES.md.bak-1\n"
    ));
}

// ---------------------------------------------------------------------------
// The floor
// ---------------------------------------------------------------------------

#[test]
fn the_flag_is_hard_and_auditable() {
    let flags = GateFlags {
        fleet_rules_write: true,
        ..Default::default()
    };
    assert!(flags.hard(), "a RULES.md write must bypass the judge");
    assert!(flags.any());
    assert!(flags.as_log_string().contains("fleet_rules_write"));
}

/// Old audit rows have no such field and must still load.
#[test]
fn old_corpus_rows_deserialize() {
    let mut row = serde_json::to_value(GateFlags::default()).unwrap();
    row.as_object_mut().unwrap().remove("fleet_rules_write");
    let flags: GateFlags = serde_json::from_value(row).unwrap();
    assert!(!flags.fleet_rules_write);
}
