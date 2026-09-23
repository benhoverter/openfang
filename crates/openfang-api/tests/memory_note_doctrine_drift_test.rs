//! ANAI-270: the runtime and the bridge must teach the SAME note doctrine.
//!
//! Same shape as `episode_close_doctrine_drift_test.rs`: the bridge does not
//! depend on `openfang-runtime`, so the `memory_note` description and its
//! `supersedes` parameter text are duplicated literals. `openfang-api` is the
//! only crate that sees both, so the pin lives here.

#[test]
fn the_bridge_and_the_runtime_ship_the_same_note_doctrine() {
    assert_eq!(
        openfang_mcp_bridge::MEMORY_NOTE_DESCRIPTION,
        openfang_runtime::tool_runner::MEMORY_NOTE_DESCRIPTION,
        "the bridge's memory_note doctrine has drifted from the runtime's. \
         Subprocess agents read the bridge; update both literals together."
    );
    assert_eq!(
        openfang_mcp_bridge::MEMORY_NOTE_SUPERSEDES_DESCRIPTION,
        openfang_runtime::tool_runner::MEMORY_NOTE_SUPERSEDES_DESCRIPTION,
        "the bridge's `supersedes` description has drifted from the runtime's"
    );
}

/// A shared const neither definition serves would pass the test above and
/// change nothing for any agent.
#[test]
fn both_tool_surfaces_actually_serve_that_doctrine() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "memory_note")
        .expect("the runtime defines memory_note");
    assert_eq!(
        runtime.description,
        openfang_runtime::tool_runner::MEMORY_NOTE_DESCRIPTION
    );
    assert_eq!(
        runtime.input_schema["properties"]["supersedes"]["description"],
        serde_json::json!(openfang_runtime::tool_runner::MEMORY_NOTE_SUPERSEDES_DESCRIPTION)
    );

    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "memory_note")
        .expect("the bridge advertises memory_note");
    assert_eq!(
        bridge.description.as_deref(),
        Some(openfang_mcp_bridge::MEMORY_NOTE_DESCRIPTION)
    );
    assert_eq!(
        bridge.input_schema["properties"]["supersedes"]["description"],
        serde_json::json!(openfang_mcp_bridge::MEMORY_NOTE_SUPERSEDES_DESCRIPTION)
    );
}

/// The three rules step 0 earned, by yield: a moving value is a fact, a
/// visible note being replaced is named, and replacement is whole-note.
#[test]
fn the_note_doctrine_carries_the_three_rules() {
    let d = openfang_runtime::tool_runner::MEMORY_NOTE_DESCRIPTION;
    assert!(d.contains("use memory_fact instead"), "status-as-fact rule");
    assert!(d.contains("name it in 'supersedes'"), "supersession rule");
    assert!(d.contains("Replacement is whole-note"), "whole-note rule");
    assert!(
        d.contains("everything from the old note that is still true"),
        "the whole-note rule must say what to carry"
    );
}
