# Starter agents

Templates for `openfang init --starter` (see `docs/fork/quickstart.md`). They're compiled into the `openfang` binary, so edits here need a rebuild to take effect for new installs. Existing installs are never touched.

- `RULES.md`: shared operating rules, installed to `~/.openfang/RULES.md`.
- `common/`: persona files every agent gets (`SOUL.md`, `AGENTS.md`, `TOOLS.md`, `USER.md`, `MEMORY.md`, `users/<you>.md`). `CODING.md` is spliced into `AGENTS.md` for `coder` and `alpha`.
- `agents/<role>/agent.toml`: one manifest per role. `agents/IDENTITY.md` is shared, filled with each role's title and purpose (defined in `crates/openfang-cli/src/starter.rs`).

Placeholders are `{{NAME}}`. The installer refuses to write a file with any placeholder left unfilled, and a unit test renders and parses every role.
