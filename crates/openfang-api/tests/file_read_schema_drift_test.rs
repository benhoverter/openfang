//! ANAI-291: `file_read`'s ranged schema must be advertised identically on
//! both tool surfaces.
//!
//! Two surfaces describe `file_read`: the in-process definition in
//! `openfang_runtime::tool_runner`, and the bridge's copy — which is what a
//! subprocess agent (every Claude Code-backed agent in the fleet) actually
//! reads. The bridge deliberately does not depend on `openfang-runtime`; the
//! seam is one-way so the bridge stays out of the kernel/compactor blast
//! radius. The price is literals the compiler cannot reconcile, so the pin
//! lives here, in the one crate that depends on both.
//!
//! This matters more than a description usually would. `offset`/`limit` are
//! the *only* way an agent can read a bounded window, and an agent cannot
//! discover an argument that is not advertised to it. That is exactly how
//! `timeout_secs` sat implemented-but-unreachable on ANAI-201: the code was
//! right and no caller could use it. A schema present on one surface and
//! absent on the other is the same bug, half the time.

/// Byte equality, not "both mention offset". The two halves an agent needs —
/// that the units are LINES, and that a large file returns a head plus a
/// manifest rather than the file — are exactly the parts that get paraphrased
/// away when one copy is edited alone.
#[test]
fn the_bridge_and_the_runtime_describe_file_read_identically() {
    assert_eq!(
        openfang_mcp_bridge::FILE_READ_DESCRIPTION,
        openfang_runtime::tool_runner::FILE_READ_DESCRIPTION,
        "the bridge's file_read description has drifted from the runtime's. \
         Subprocess agents read the bridge; in-process agents read the \
         runtime. Update both literals together."
    );
}

/// A shared const that neither tool definition actually serves would pass the
/// test above and change nothing for any agent.
#[test]
fn both_tool_surfaces_actually_serve_that_description() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "file_read")
        .expect("the runtime defines the tool");
    assert_eq!(
        runtime.description,
        openfang_runtime::tool_runner::FILE_READ_DESCRIPTION,
        "the runtime's tool definition must serve the shared description"
    );

    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "file_read")
        .expect("the bridge advertises the tool");
    let served = bridge
        .description
        .as_deref()
        .expect("the bridge tool is described");
    assert_eq!(
        served,
        openfang_mcp_bridge::FILE_READ_DESCRIPTION,
        "the bridge's tool definition must serve the shared description"
    );
}

/// The load-bearing one: the *arguments* must match. An agent cannot call an
/// argument it was never told about, so a range arg advertised in-process and
/// missing from the bridge is a capability that exists for nobody who matters.
#[test]
fn both_tool_surfaces_advertise_the_range_arguments() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "file_read")
        .expect("the runtime defines the tool");
    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "file_read")
        .expect("the bridge advertises the tool");

    let runtime_props = &runtime.input_schema["properties"];
    let bridge_props = &bridge.input_schema["properties"];

    assert_eq!(
        runtime_props, bridge_props,
        "an agent must not be offered a file_read argument on one surface \
         that the other does not accept"
    );

    for arg in ["path", "offset", "limit"] {
        assert!(
            !runtime_props[arg].is_null(),
            "file_read must advertise '{arg}'"
        );
    }
    assert_eq!(
        runtime_props["offset"]["type"], "integer",
        "offset is a line number, not a byte offset or a string"
    );
    assert_eq!(runtime_props["limit"]["type"], "integer");

    // Ranges are additive: a caller that passes neither must still be valid,
    // or every existing call site breaks.
    assert_eq!(
        runtime.input_schema["required"],
        serde_json::json!(["path"]),
        "offset/limit must stay optional — they are additive, not a new contract"
    );
}
