//! ANAI-283: the runtime and the bridge must teach the SAME close doctrine.
//!
//! Three surfaces tell an agent when to close an episode: the `## Memory`
//! bullet in `prompt_builder`, the in-process tool description in
//! `tool_runner`, and the bridge's copy — which is what a subprocess agent
//! (every Claude Code-backed agent in the fleet) actually reads. The bridge
//! deliberately does not depend on `openfang-runtime`; the seam is one-way so
//! the bridge stays out of the kernel/compactor blast radius. The price is two
//! literals that the compiler cannot reconcile.
//!
//! They had already drifted, and the drift was silent: `tool_runner` and the
//! bridge agreed, but both disagreed with the prompt bullet, and a model
//! resolving a contradiction favours the text attached to the button it is
//! about to press. `memory_episode_close` produced 4 voluntary closes in three
//! weeks against 320 timer closes.
//!
//! `openfang-api` is the only crate that depends on both, so the pin lives
//! here rather than in either of them.

/// Byte equality, not "both mention topic-switch". A doctrine that agrees in
/// keywords and disagrees in its tie-break is exactly the failure this pins.
#[test]
fn the_bridge_and_the_runtime_ship_the_same_close_doctrine() {
    assert_eq!(
        openfang_mcp_bridge::EPISODE_CLOSE_DESCRIPTION,
        openfang_runtime::tool_runner::EPISODE_CLOSE_DESCRIPTION,
        "the bridge's copy of the episode-close doctrine has drifted from the \
         runtime's. Subprocess agents read the bridge; in-process agents read \
         the runtime. Update both literals together."
    );
}

/// And the constants must actually be what each side *serves* — a shared const
/// that neither tool definition uses would pass the test above and change
/// nothing for any agent.
#[test]
fn both_tool_surfaces_actually_serve_that_doctrine() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "memory_episode_close")
        .expect("the runtime defines the tool");
    assert_eq!(
        runtime.description,
        openfang_runtime::tool_runner::EPISODE_CLOSE_DESCRIPTION,
        "the runtime's tool definition must serve the shared doctrine"
    );

    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "memory_episode_close")
        .expect("the bridge advertises the tool");
    let served = bridge
        .description
        .as_deref()
        .expect("the bridge tool is described");
    assert_eq!(
        served,
        openfang_mcp_bridge::EPISODE_CLOSE_DESCRIPTION,
        "the bridge's tool definition must serve the shared doctrine"
    );
}

/// The reason vocabulary is the other half an agent reads off the button, and
/// it drifted the same way: the bridge advertised `["explicit"]` while the
/// handler's allowlist lived in the runtime.
#[test]
fn both_tool_surfaces_offer_the_same_close_reasons() {
    let runtime = openfang_runtime::tool_runner::builtin_tool_definitions()
        .into_iter()
        .find(|d| d.name == "memory_episode_close")
        .expect("the runtime defines the tool");
    let bridge = openfang_mcp_bridge::built_in_tools()
        .into_iter()
        .find(|t| t.name == "memory_episode_close")
        .expect("the bridge advertises the tool");

    let runtime_reasons = &runtime.input_schema["properties"]["reason"]["enum"];
    let bridge_reasons = &bridge.input_schema["properties"]["reason"]["enum"];

    assert_eq!(
        runtime_reasons, bridge_reasons,
        "an agent must not be offered a close reason on one surface that the \
         other refuses"
    );
    assert_eq!(
        runtime_reasons,
        &serde_json::json!(["topic-switch", "explicit"]),
        "ANAI-283 opened topic-switch; timer and abandoned stay the system's"
    );
}
