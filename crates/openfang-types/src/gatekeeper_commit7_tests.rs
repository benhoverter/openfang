//! ANAI-206 commit 7: the remaining findings from the `0dba06f` security
//! review — F3 (`cd` laundering), F4 (`OPAQUE_EXEC_VERBS` unchecked in a body),
//! F6 (heredoc terminator column, phantom delimiters) and F7 (the above-cap
//! blind window).
//!
//! Kept in its own file for the reason commit 6's is: every one of these is a
//! predicate that failed *open* while the suite stayed green. Each test names
//! the bypass it closes, so a future edit that reopens one has to delete a
//! sentence saying so.

use super::*;

// ---------------------------------------------------------------------------
// F3 — `cd` moves the frame every later relative operand resolves against
// ---------------------------------------------------------------------------

/// The cheapest bypass in the review: two lines of ordinary shell.
#[test]
fn cd_into_the_substrate_launders_a_relative_removal() {
    let body = "cd ~/.openfang/agents\nrm -rf openfang-alpha\n";
    // Neither line fires on its own: line 1 names but does not write, line 2
    // writes but names nothing a path predicate can see.
    assert!(!destroys_substrate("cd ~/.openfang/agents"));
    assert!(!destroys_substrate("rm -rf openfang-alpha"));
    assert!(
        body_destroys_substrate(body),
        "a relative removal inside the substrate must still be hard"
    );
}

/// The control-plane half lands on the soft flag, not the hard one — `scripts/`
/// is deliberately not substrate.
#[test]
fn cd_into_the_control_plane_is_a_write_but_not_a_substrate_wipe() {
    let body = "cd ~/.openfang/scripts\ntee deploy-local.sh\n";
    assert!(body_writes_control_plane(body));
    assert!(!body_destroys_substrate(body));
}

/// An absolute operand is not being resolved against the frame, so the frame
/// cannot be what makes it dangerous. Firing here would spend a bypassed judge
/// for nothing.
#[test]
fn an_absolute_target_is_not_attributed_to_the_frame() {
    let body = "cd ~/.openfang/agents\nrm -rf /tmp/scratch\n";
    assert!(!body_destroys_substrate(body));
}

/// Walking back out ends the frame. Without this the predicate would treat the
/// whole tail of any script that ever visited `~/.openfang` as substrate work.
#[test]
fn leaving_the_frame_clears_it() {
    assert!(!body_destroys_substrate(
        "cd ~/.openfang/agents\ncd /tmp\nrm -rf scratch\n"
    ));
    assert!(!body_destroys_substrate(
        "cd ~/.openfang/agents\ncd ..\ncd ..\nrm -rf scratch\n"
    ));
    // ...but descending deeper stays inside it.
    assert!(body_destroys_substrate(
        "cd ~/.openfang/agents\ncd openfang-alpha\nrm -rf memory\n"
    ));
}

// ---------------------------------------------------------------------------
// F4 — a body sets one predicate where a command line sets ten
// ---------------------------------------------------------------------------

/// `has_opaque_execution` rescues this on the command line. Inside a body there
/// is no such flag, so `segment_writes` has to carry it.
#[test]
fn inline_interpreter_source_is_a_write_inside_a_body() {
    let body = "python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1])' ~/.openfang/agents\n";
    assert!(body_writes_control_plane(body));
}

/// The same on the command line, where it was already true — pinned so a future
/// edit cannot "simplify" `segment_writes` back to bins only.
#[test]
fn opaque_exec_verbs_are_writes_on_the_command_line_too() {
    assert!(touches_control_plane("node -e 'x' ~/.openfang/config.toml"));
}

// ---------------------------------------------------------------------------
// F6 — heredoc terminators, and delimiters that are not delimiters
// ---------------------------------------------------------------------------

/// Bash ends a heredoc on a line that is exactly the delimiter at column 0. The
/// old `raw.trim() == delim` ended our fold at the indented `EOF` while bash
/// kept consuming, dropping the control path out of the folded line.
#[test]
fn an_indented_terminator_does_not_end_the_fold() {
    let body = "rm -rf $(cat <<EOF\n  EOF\n~/.openfang/agents/\nEOF\n)\n";
    assert!(
        body_writes_control_plane(body),
        "the payload bash would consume must stay inside the logical line"
    );
}

/// `<<-` strips leading **tabs**, so a tab-indented terminator really does end
/// that heredoc. Spaces never do.
#[test]
fn the_dash_form_strips_tabs_and_only_tabs() {
    let tabbed = logical_lines("cat <<-EOF\npayload\n\tEOF\necho after\n");
    assert_eq!(tabbed.len(), 2, "the tab-indented EOF terminates `<<-`");

    let spaced = logical_lines("cat <<-EOF\npayload\n  EOF\necho after\n");
    assert_eq!(
        spaced.len(),
        1,
        "a space-indented terminator does not close `<<-`, so the fold runs on"
    );
}

/// An arithmetic shift is not a heredoc. Taking the first `<<` unconditionally
/// swallowed the rest of the file into one logical line — over-firing on its
/// own, and a free way to push that line past the normalizer cap.
#[test]
fn an_arithmetic_shift_does_not_open_a_phantom_heredoc() {
    let lines = logical_lines("x=$((1<<3))\necho hello\necho world\n");
    assert_eq!(lines.len(), 3, "no line should have been swallowed");
    assert!(lines[0].heredoc_payload.is_none());

    // A real delimiter next to a shift still opens.
    let real = logical_lines("cat <<EOF\npayload\nEOF\necho after\n");
    assert_eq!(real.len(), 2);
}

// ---------------------------------------------------------------------------
// F7 — the above-cap window
// ---------------------------------------------------------------------------

/// Over the cap the scan used to degrade to raw containment, which the attacker
/// controls the spelling of from inside the body.
#[test]
fn an_obfuscated_path_survives_the_fold_overflow() {
    let padding = "x".repeat(crate::cmd_norm::MAX_NORMALIZE_INPUT + 1000);
    let body = format!("rm -rf ~/.open\"\"fang/agents/ \\\n  --flag {padding}\n");
    assert!(
        body_writes_control_plane(&body),
        "chunked deny_variants must still deobfuscate an over-cap line"
    );
}

/// The worse half: a **hard** flag that silently did not evaluate. Plain
/// `rm -rf ~/.openfang` plus padding cleared it with no obfuscation at all.
#[test]
fn an_over_cap_line_is_scanned_in_chunks_not_skipped() {
    let padding = "x".repeat(crate::cmd_norm::MAX_NORMALIZE_INPUT + 1000);
    let body = format!("rm -rf ~/.openfang \\\n  --flag {padding}\n");
    assert!(
        body_destroys_substrate(&body),
        "an over-cap logical line must be evaluated, not skipped"
    );
}

/// The relaxation above the cap is bounded: an over-cap line that names nothing
/// substrate-shaped is still false, so padding alone cannot manufacture a hard
/// escalation.
#[test]
fn an_over_cap_line_with_no_substrate_is_still_clean() {
    let padding = "x".repeat(crate::cmd_norm::MAX_NORMALIZE_INPUT + 1000);
    let body = format!("rm -rf ./target \\\n  --flag {padding}\n");
    assert!(!body_destroys_substrate(&body));
}

/// Chunk boundaries overlap, so a control path landing across one is whole in
/// at least one chunk.
#[test]
fn chunks_overlap_enough_to_hold_a_path_whole() {
    let cap = crate::cmd_norm::MAX_NORMALIZE_INPUT;
    let text = "y".repeat(cap * 2);
    let chunks = overcap_chunks(&text);
    assert!(chunks.len() >= 2);
    assert!(chunks.iter().all(|c| c.chars().count() <= cap));
    const { assert!(OVERCAP_CHUNK_OVERLAP > 128) };
}

/// The guard and the normalizer must count the same units, or a multi-byte line
/// clears the guard and is byte-truncated inside the normalizer with no
/// fallback at all. Both count `chars()`.
#[test]
fn the_cap_guard_and_the_normalizer_agree_on_units() {
    let wide = "é".repeat(crate::cmd_norm::MAX_NORMALIZE_INPUT + 10);
    let variants = crate::cmd_norm::deny_variants(&wide);
    assert_eq!(
        variants[0].chars().count(),
        crate::cmd_norm::MAX_NORMALIZE_INPUT
    );
}
