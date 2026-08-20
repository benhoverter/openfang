# ADR 0002 — The memory tool surface

- **Status:** Proposed
- **Date:** 2026-08-19
- **Author:** Annabelle (openfang-memory), with Ben Hoverter
- **Supersedes:** nothing. Depends on and refines ADR 0001 (§2.2 episodes, §2.3 keyed slots,
  §2.4 consolidation, §2.6 MEMORY.md rendering).
- **Code read at:** `5e33cf9` (fork `main`) plus the uncommitted episodes change on
  `feat/anai-84-turn-trigger-episodes`.
- **Linear:** ANAI-59 (memory epic), ANAI-84 (trigger/episodes), ANAI-165/167/168.

---

## 1. Context

ADR 0001 decided the *shape of storage*. It did not decide what an agent can say to that
storage. Stage 1b needs `memory_episode_close`, and cutting one bespoke tool per feature is
how a surface becomes nine overlapping verbs nobody can describe in one line. This ADR fixes
the whole surface first so each subsequent tool is a scheduled item, not an invention.

### 1.1 What exists today (measured, not inferred)

Read this turn from the worktree:

- **Two live agent-facing tools.** `memory_store(key, value)` and `memory_recall(key)`.
  Both are pure key–value, agent-scoped, with a `shared:` prefix as the only escape.
- **Two phantom tools.** `memory_delete` and `memory_list` appear in
  `prompt_builder.rs:664` (category map) and `:720` (description map) with no
  implementation behind them on the agent path. Agents are told about tools they do not
  have.
- **A human delete path does exist**, in the CLI: `MemoryCommands::Delete` →
  `cmd_memory_delete` (`openfang-cli/src/main.rs:1182`, `:6835`). That is the right home
  for it (see §4.3).
- **A naming lie.** The prompt describes `memory_recall` as *"search memory for relevant
  context"*. The implementation is exact key lookup. Agents are told they have retrieval
  and handed a dictionary; when the key does not match, memory reads as unreliable from the
  inside. This is a correctness bug in the prompt, not a nit.

### 1.2 The diagnosis

The surface grew per-feature, so it models *storage mechanics* (a KV pair) rather than
*concepts* (a fact, a note, an episode). Every future storage refactor therefore threatens
to move the tool surface, and every new memory behaviour tempts a new verb.

---

## 2. Decision

### 2.1 Five principles the surface must satisfy

1. **Tools model concepts, not tables.** No agent learns that facts live in `memories`
   behind a `kind` discriminator. Storage may be refactored without the surface moving.
2. **One read tool.** Retrieval is where ADR 0001's invariants live — never surface a
   superseded fact (§2.3.1), respect scope. Every additional read path is another place to
   forget the filter. One door, enforced once.
3. **Split writes by the cost of being wrong, not by data type.** A cheap unstructured note
   and a keyed fact that overwrites a live slot deserve different tools because they deserve
   different care.
4. **Do not tool what the system can derive.** If capture or activity already establishes
   the state, there is no verb for it.
5. **One-line describable, or it is the wrong shape.** If the prompt description needs a
   paragraph of when-to-use, two tools have been merged.

### 2.2 The surface

Six tools. Three of them are called in a normal week.

| Tool | Signature | Purpose | Stage |
|---|---|---|---|
| `memory_recall` | `(query, scope?, kind?, limit?)` | The single read. Semantic + keyword across facts and episode summaries. Exact-key lookup survives as `key:foo`. | 2 |
| `memory_note` | `(text, tags?)` | Cheap, unstructured capture. No vocabulary burden. Consolidation's raw input. | 2 |
| `memory_fact` | `(claim_key, text, scope?)` | Upsert into a keyed slot per ADR 0001 §2.3.2. Rejects keys outside the controlled vocabulary. | 3 |
| `memory_episode_close` | `(reason, title, summary?)` | The explicit episode boundary. | **1b — next** |
| `memory_status` | `()` | Open episode, turns since open, notes pending consolidation, idle countdown. | 1b or 2 |
| `memory_history` | `(claim_key)` | Read-only audit escape hatch into `fact_history`. | 3 |

### 2.3 `memory_recall` is a rename as much as an implementation

Making `memory_recall` genuinely retrieval-shaped closes §1.1's naming lie. It is the only
tool permitted to read tier-3 facts, and the superseded filter lives inside it, not in the
caller. Exact-key lookup is preserved as a `key:` prefix so existing agent habits and stored
keys keep working — this is an extension, not a breaking change.

### 2.3.1 Semantic recall is caller-scoped: no `shared:` escape

The key-value path keeps its `shared:` namespace. Semantic search does **not** get one, and
this asymmetry is deliberate rather than an unfinished feature.

Reaching into another agent's memory by exact key requires knowing that key — the caller has
to already possess the thing it is asking for, which bounds the disclosure to what was
effectively shared by prior arrangement. Semantic search needs no such knowledge: a topic
guess is enough. `memory_recall(query="the credentials Ben gave you")` against a fleet-wide
corpus is a materially wider door than `memory_recall(key="shared:build_host")`, and it is a
door no caller has to have been let through before.

So `memory_recall`'s search path filters to the calling agent's own scope, unconditionally,
below the tool boundary (`kernel.rs`, caller-scoped `MemoryFilter`). Cross-agent knowledge
transfer stays an explicit act — `agent_send`, or a key deliberately published under
`shared:` — not a side effect of a well-phrased query.

If a cross-agent read is ever wanted, it arrives as an explicit `scope` value with its own
capability check, not by relaxing the default.

### 2.4 `memory_status` is load-bearing, not convenience

It is the cheapest tool here and the one that makes the deferred topic-switch detector
possible. Consolidation-on-topic-switch (ADR 0001 §2.4) requires the agent to notice it is
drifting; today it has no introspection at all — it cannot see how long the episode has been
open or how much uncaptured material sits behind it. A judgment call with no instrument
panel is a guess. Ship the panel before asking for the judgment.

### 2.5 MEMORY.md is **rendered**, not authored

Confirmed as a decision here because it collapses two of the four competing memory
paradigms into one. MEMORY.md is a materialized view of the agent's active tier-3/4 state
(ADR 0001 §2.6 governs *what* renders), regenerated on write. Consequences:

- `memory_fact` is the only way its fact content changes. There is no reconciliation
  tooling, because file-vs-DB drift is structurally impossible.
- Agents stop hand-editing MEMORY.md with `file_write`. This is a real loss of narrative
  expressiveness, accepted deliberately: prose that matters becomes a note or a fact, and
  anything that survives neither was decoration.
- This is cheap *now* precisely because MEMORY.md is effectively unused. It will not be
  cheap later.

### 2.6 What we refuse, by name

Recorded so these are not re-proposed as reasonable-sounding features.

- **A tool per memory subdir** (`memory_store_project`, `memory_store_user`). That is
  `scope` as a parameter, not four tools.
- **A tool per episode verb.** No `episode_open` — capture opens it. No `episode_extend` —
  activity extends it. No `episode_relabel` — that is `close`'s `title`.
- **Agent-facing `memory_delete`.** Retraction is `memory_fact` writing the correction, or
  a later `forget` that moves a row to history. Hard delete stays a human at the CLI or a
  sqlite prompt (§1.1). An agent should not hold an irreversible verb over its own past.
- **Direct writes to `episodes`.** The database owns the lifecycle via the partial unique
  index; handing agents a row editor undoes the DDL guarantee.
- **A consolidation tool.** Consolidation is a *turn shape* — a prompt-level behaviour with
  approval — not a function call. Making it callable invites calling it mid-task, which is
  the exact failure the design avoids.
- **`memory_list`.** Enumeration is `memory_recall` with a broad query and a limit. Keep the
  phantom out of the description map either way (§3).

---

## 3. Consequences

- **Immediate cleanup, independent of any new tool:** delete `memory_delete` and
  `memory_list` from `prompt_builder.rs:664`/`:720`, or implement them. Advertising
  non-existent tools costs a wasted call and a confused agent every time one is believed.
- **Stage 1b is now scoped**, not invented: `memory_episode_close` plus `memory_status`,
  against the already-tested `EpisodeStore::close_episode`. Both are in this table, so
  neither is an ad hoc feature.
- **`memory_store` keeps working** through stage 2; `memory_note`/`memory_fact` land beside
  it and it is retired only once callers have moved.
- **Tool-surface files are shared with openfang-tools/alpha** (`tool_runner.rs`, the MCP
  bridge allowlist, `prompt_builder.rs`, capability lists). Ownership of *these* tools is
  ours by subject; the touch to shared files still gets flagged to alpha before landing.

---

## 4. Alternatives considered

**4.1 Cut `memory_episode_close` alone and design the rest later.** Fastest to stage 1b,
and how the current surface got its phantom entries. Rejected: the marginal cost of writing
the table now is one document.

**4.2 Separate read tools per tier** (`memory_recall_facts`, `memory_recall_episodes`).
Rejected by principle 2 — the superseded filter would have to be reimplemented per path,
and ADR 0001 §2.3.1 is only worth anything if it is enforced in exactly one place.

**4.3 Give agents `memory_delete` because the CLI has it.** Rejected: the CLI is a human
holding the whole context of a decision; an agent mid-task is not. Different actor,
different privilege.

**4.4 Keep MEMORY.md authored.** Preserves narrative freedom and needs no renderer work.
Rejected: it requires drift reconciliation and a policy for file-contradicts-fact, which is
the same read-time-arbitration failure ADR 0001 §2.3 already refused to accept.

---

## 5. Open questions

**5.1 Is `memory_note` vs `memory_fact` a real split or an artificial one?** It is designed
against the ADR, not against observed agent behaviour. If in practice every note is
immediately promoted to a fact, the split is ceremony and should collapse.

**5.2 What does `memory_status` return when no episode is open?** Probably a shaped null,
but "no open episode" may itself be a state worth naming for the topic-switch detector.

**5.3 Does `memory_recall`'s `key:` prefix stay forever, or is it a migration ramp?**
Depends on how many stored keys are load-bearing after consolidation runs once.

**5.4 Claim-key vocabulary** remains open per ADR 0001 §5.4 and gates `memory_fact`
(stage 3), not the earlier stages.

---

## 6. Amendment log

- 2026-08-19 — created. Written at Ben's instruction to document the surface *before*
  cutting `memory_episode_close`, so stage 1b is a scheduled item rather than an ad hoc
  feature. MEMORY.md confirmed **rendered** (§2.5) on his call that the file is currently
  unused and the change is therefore free.
- 2026-08-19 — added §2.3.1. Semantic recall was implemented caller-scoped with no
  `shared:` escape (stage 2, `93a9b40`); that is an access-control decision and was living
  only in a code comment. Recorded here so it is not later read as an oversight and
  "fixed".
