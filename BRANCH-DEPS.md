# BRANCH-DEPS.md

Authoritative dependency map for OpenFang fork branches. Source of truth for the reroll script. Update this file whenever a topic is added, retired, or changes base.

The reroll script lives **outside** the repo at `~/.openfang/scripts/reroll-local-main.sh` (sibling to `deploy-local.sh`). Fork-management infrastructure isn't source — keeping it out of the tree avoids polluting `local-main`'s history with tooling churn.

See Linear: **OpenFang Fork: Branch Topology & Workflow (handoff)** for the full topology rules. Short version:

- `upstream/main` = read-only mirror of `RightNow-AI/openfang:main`.
- `origin/*` = `benhoverter/openfang` (the fork on GitHub).
- `local-main` = throwaway integration branch, force-pushed each reroll. **Never branch off it.**
- `topic/*` = single-purpose, rooted on `upstream/main` (or another topic, with the dep recorded here).

## Topics

| Topic / branch                          | Base                          | Origin (GitHub)                          | PR     | Status                       |
|-----------------------------------------|-------------------------------|------------------------------------------|--------|------------------------------|
| `topic/runtime-image-cache` *(adopted)* | `upstream/main`               | `origin/feat/runtime-image-cache`        | #1151  | open, awaiting review        |
| `topic/diagnostics`                     | `topic/runtime-image-cache`   | *(local-only)*                           | —      | local, holds context-drop hooks |
| `topic/discord-outbound-attachments`    | `topic/runtime-image-cache`   | *(local, not yet pushed)*                | —      | rebased onto image-cache after fold (see below) |
| `topic/lenient-binding-parse` *(adopted)* | `upstream/main`             | `origin/lenient-binding-parse`           | #1146  | open, awaiting review        |
| `topic/harden-channel-id-binding` *(adopted)* | `upstream/main`         | `origin/harden-channel-id-binding`       | #1147  | open, awaiting review        |
| `topic/discord-file-sharing` *(adopted)*| `upstream/main`               | `origin/discord-file-sharing`            | #1143  | open, awaiting review        |
| `feat/anai-32` *(adopted)*              | `upstream/main`               | `origin/feat/anai-32`                    | #1182  | open, awaiting review — shell capability gate + bridge tool registry |
| `feat/anai-40-file-policy` *(adopted)*  | `feat/anai-32`                | `origin/feat/anai-40-file-policy`        | #1183  | open, awaiting review — file policy schema/loader/evaluator + cross-component gates |
| `topic/branch-docs`                     | `upstream/main`               | *(local)*                                | —      | this file                    |

### "Adopted" branches

A few branches predate the topology rule. They're treated as topics in-place — no rename, no force-push, no PR churn — and documented here. The topology rule is about **base-branch discipline**, not naming purity.

### Cross-PR dependency: outbound-attachments ↔ image-cache

PRs #1143 (`discord-file-sharing`) and #1151 (`feat/runtime-image-cache`) independently rewrote `claude_code.rs::build_prompt` and both invented a `render_content` helper with the same role but different signatures. #1151 is strictly a superset (materializes images instead of just emitting a marker) and adds a `source_url: Option<String>` field to `ContentBlock::Image` that #1143 doesn't know about.

Resolution: locally, `topic/discord-outbound-attachments` is rebased onto `topic/runtime-image-cache` and the obsolete portions of commit `118eace` (textual `render_content`, `text_content()` replacement) are dropped during the cherry-pick. The bridge / outbound logic — the actual point of the topic — survives unchanged. This keeps `local-main` buildable while both PRs are open.

The live PR branches `discord-file-sharing` and `feat/runtime-image-cache` are **not** modified by this. Whichever merges first wins; the second will need a small fold against the new `upstream/main` at merge time. See Linear §9 (entry "Reroll surfaces real conflict") for the full reasoning.

## Daily-binary recipe

`local-main` is rebuilt by `~/.openfang/scripts/reroll-local-main.sh` as:

```
upstream/main
  ├── topic/runtime-image-cache              (= origin/feat/runtime-image-cache)
  │     ├── topic/diagnostics                 (local)
  │     └── topic/discord-outbound-attachments (local; supersedes claude_code portion of #1143's 118eace)
  ├── topic/lenient-binding-parse             (= origin/lenient-binding-parse)
  ├── topic/harden-channel-id-binding         (= origin/harden-channel-id-binding)
  ├── topic/discord-file-sharing              (= origin/discord-file-sharing)
  ├── feat/anai-32                            (= origin/feat/anai-32; PR #1182)
  │     └── feat/anai-40-file-policy          (= origin/feat/anai-40-file-policy; PR #1183 — merges anai-32 at cbb1105)
  └── topic/branch-docs                       (this file)
```

The reroll script octopus-merges the **leaf** of each chain into `local-main` (so ancestors come along automatically). Effective leaves:

1. `topic/diagnostics` (pulls in `topic/runtime-image-cache`)
2. `topic/discord-outbound-attachments` (pulls in `topic/runtime-image-cache`; supersedes claude_code portion of `topic/discord-file-sharing` so that branch is **not** a separate leaf — see "Cross-PR dependency")
3. `topic/lenient-binding-parse`
4. `topic/harden-channel-id-binding`
5. `origin/feat/anai-40-file-policy` (pulls in `feat/anai-32` via merge `cbb1105`; PR #1183 stacks on PR #1182)
6. `topic/branch-docs`

## Adding a topic

1. Branch from the right base (`upstream/main` or another topic). Record the dep here.
2. Add it to the recipe above. Add it to `LEAVES` in `~/.openfang/scripts/reroll-local-main.sh` if it's a leaf.
3. Re-roll: `~/.openfang/scripts/reroll-local-main.sh`.

## Retiring a topic

When a topic's PR merges to `upstream/main`:

1. Remove it from this file.
2. Remove it from the reroll script.
3. Re-roll. The merged commits arrive via `upstream/main`; the topic branch can be deleted locally and on origin.
