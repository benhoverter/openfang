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

// ---- ANAI-292: file_grep, the consumer's other half -------------------------

#[test]
fn the_bridge_and_the_runtime_describe_file_grep_identically() {
    assert_eq!(
        openfang_mcp_bridge::FILE_GREP_DESCRIPTION,
        openfang_runtime::tool_runner::FILE_GREP_DESCRIPTION,
        "the bridge's file_grep description has drifted from the runtime's"
    );
}

/// The schema is duplicated by hand in the bridge, because the crate seam is
/// one-way. Seven arguments is more than enough to drift silently, and an
/// argument missing from the bridge is an argument no subprocess agent — which
/// is most of the fleet — can ever use.
#[test]
fn both_tool_surfaces_advertise_the_same_file_grep_arguments() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "file_grep")
        .expect("the runtime defines file_grep");
    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "file_grep")
        .expect("the bridge advertises file_grep");

    assert_eq!(
        runtime.input_schema,
        openfang_runtime::tool_runner::file_grep_input_schema(),
        "the runtime's definition must serve the shared schema helper"
    );
    assert_eq!(
        serde_json::to_value(&*bridge.input_schema).unwrap(),
        runtime.input_schema,
        "the bridge's hand-copied file_grep schema has drifted from the runtime's"
    );
    assert_eq!(
        runtime.input_schema["required"],
        serde_json::json!(["path", "pattern"]),
        "a grep with no pattern would match everything"
    );
}

/// `file_grep` must be granted wherever `file_read` is. The argument for
/// building it at all is that most of the fleet has no `shell_exec`, and that
/// only pays off if it is granted by default: opt-in would serve the agents
/// that already have a shell and help nobody. A grep also exposes a strict
/// subset of what a read returns, so granting one and denying the other
/// protects nothing.
#[test]
fn file_grep_is_granted_wherever_file_read_is() {
    assert!(
        openfang_mcp_bridge::DEFAULT_ALLOWED.contains(&"file_grep"),
        "file_grep must be in the bridge's default-allowed set, like file_read"
    );
    assert!(
        openfang_api::bridge_ipc::ALLOWED_TOOLS.contains(&"file_grep"),
        "file_grep must be dispatchable by the daemon-side IPC ceiling"
    );
    assert!(
        openfang_runtime::tool_runner::FS_SANDBOXED_TOOLS.contains(&"file_grep"),
        "file_grep takes a path argument, so it must be workspace-scoped for \
         the bridge surfaces — omitting it would be a sandbox bypass, which is \
         exactly how create_directory and shell_exec were missed before"
    );
    assert!(
        openfang_types::turn::READ_ONLY_TOOLS.contains(&"file_grep"),
        "file_grep writes nothing; omitting it makes a retrieval-only call \
         read as side-effecting and locks Assist-mode agents out of it"
    );
}

/// The alias that used to lie: `Grep` mapped to `file_list`, so an agent
/// asking to search a file got a directory listing — a plausible-looking
/// answer to a different question. There is a real target now.
#[test]
fn the_grep_alias_points_at_the_grep_tool() {
    use openfang_types::tool_compat::{map_tool_name, normalize_tool_name};
    assert_eq!(map_tool_name("Grep"), Some("file_grep"));
    assert_eq!(map_tool_name("grep"), Some("file_grep"));
    assert_eq!(map_tool_name("rg"), Some("file_grep"));
    assert_eq!(normalize_tool_name("Grep"), "file_grep");
    // Glob is a name search, not a content search, and stays where it was.
    assert_eq!(map_tool_name("Glob"), Some("file_list"));
}

// ---- ANAI-297: image_read -----------------------------------------------------

#[test]
fn the_bridge_and_the_runtime_describe_image_read_identically() {
    assert_eq!(
        openfang_mcp_bridge::IMAGE_READ_DESCRIPTION,
        openfang_runtime::tool_runner::IMAGE_READ_DESCRIPTION,
        "the bridge's image_read description has drifted from the runtime's"
    );
}

#[test]
fn both_tool_surfaces_serve_the_same_image_read_tool() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "image_read")
        .expect("the runtime defines image_read");
    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "image_read")
        .expect("the bridge advertises image_read");

    assert_eq!(
        runtime.description,
        openfang_runtime::tool_runner::IMAGE_READ_DESCRIPTION
    );
    assert_eq!(
        bridge.description.as_deref(),
        Some(openfang_mcp_bridge::IMAGE_READ_DESCRIPTION)
    );
    assert_eq!(
        runtime.input_schema,
        openfang_runtime::tool_runner::image_read_input_schema()
    );
    assert_eq!(
        serde_json::to_value(&*bridge.input_schema).unwrap(),
        runtime.input_schema,
        "the bridge's hand-copied image_read schema has drifted from the runtime's"
    );
}

/// Ben, 2026-09-23: "as ubiquitous as file_read". Every list that makes
/// file_read reachable and correctly classified must carry image_read too.
#[test]
fn image_read_is_granted_and_classified_wherever_file_read_is() {
    assert!(openfang_mcp_bridge::DEFAULT_ALLOWED.contains(&"image_read"));
    assert!(openfang_api::bridge_ipc::ALLOWED_TOOLS.contains(&"image_read"));
    assert!(
        openfang_runtime::tool_runner::FS_SANDBOXED_TOOLS.contains(&"image_read"),
        "image_read takes a path, so it must be workspace-scoped on the bridge"
    );
    assert!(
        openfang_types::turn::READ_ONLY_TOOLS.contains(&"image_read"),
        "image_read writes nothing"
    );
    assert!(openfang_runtime::mcp::RESERVED_BUILTIN_NAMES.contains(&"image_read"));
    assert!(openfang_types::tool_compat::is_known_openfang_tool(
        "image_read"
    ));
}
