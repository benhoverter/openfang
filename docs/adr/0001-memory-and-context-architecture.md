# ADR 0001 — Memory & context architecture

- **Status:** Proposed
- **Date:** 2026-08-19 (amended same day — §2.3 keyed slots, see §6)
- **Author:** Annabelle (openfang-memory), with Ben Hoverter
- **Supersedes:** nothing
- **Code read at:** `a9e7a33` (fork `main` + the ANAI-165/167/168 stack)
- **Linear:** ANAI-59 (memory epic), ANAI-165/166/167/168/169 (M1–M5), ANAI-160

---

## 1. Context

OpenFang has five places that hold something like memory. None of them declares what it owns,
and no code path moves a fact from one to another. This ADR names the tiers, assigns each an
owner and a rule, and defines the reader contract that consumes them.

The trigger was a measurement, not a theory. The MEMORY.md sweep (ANAI-168) shipped, ran in
dry-run against the live daemon, and reported: **14 agents would be written, each with exactly
one fact, and in every case the same fact** — `delivery.last_channel`, a Discord snowflake
written by the delivery layer (`channel_bridge.rs:1114`, `kernel.rs:8749`) as routing
bookkeeping. That is the entire agent-authored content of `kv_store` across 52 registered
agents.

### 1.1 What exists today (measured, not inferred)

| Store | Written by | Read by |
|---|---|---|
| `sessions` / `events` / `canonical_sessions` | runtime, every turn | conversation continuation only |
| `memories` | **only** auto-capture in `agent_loop.rs` — verbatim `User asked… / I responded…`, every turn | **only** top-5 similarity recall at prompt build (`agent_loop.rs:608`, `build_memory_section`) |
| `kv_store` | `memory_store` tool (`kernel.rs`, now agent-scoped per ANAI-165) + internal routing keys | `memory_recall` tool on demand; the MEMORY.md sweep |
| `entities` / `relations` | `knowledge_add` — granted to almost no agent | `knowledge_query` — same |
| `MEMORY.md` + workspace files | humans, by hand; now also the sweep's fenced block | prompt persona section, every turn (`BUDGET_MEMORY_MD = 8000`, `prompt_builder.rs:45`) |

Supporting counts (read-only DB queries, 2026-08-04/08-18): `memories` 35,336 rows, 100 %
`source=conversation`, 100 % episodic, 6 % with null embeddings (ANAI-169). `kv_store` ~1,098
rows, of which the substantive majority predate ANAI-165's scoping and sit in the legacy
shared namespace. `entities`/`relations`: effectively unused.

### 1.2 The diagnosis, in one line

**The only automatic reader reads the store nobody curates, and the curated stores have no
automatic writer.**

That inversion explains every symptom we have hit:

- `memories` is transcript sludge, so recall surfaces five near-duplicate paraphrases of
  recent turns instead of knowledge.
- `kv_store` is empty of knowledge, because writing to it requires an agent to *choose* to
  call a tool with no incentive to do so.
- The knowledge graph is a schema with no population.
- `MEMORY.md` is the only human-legible tier and, until ANAI-168, nothing wrote it.

The load-bearing absence: **`consolidate()` is a stub.** It decays confidence and returns
`memories_merged: 0, // Phase 1: no merging` (`openfang-memory/src/consolidation.rs:49`).
There is no promotion path from *we said this* to *this is true*. That missing operation
**is** the missing paradigm.

### 1.3 The second requirement: context management

Ben's stated objective is not better recall — it is **managed context that does not need to be
refreshed before new work.** That adds a reader-side requirement and a behavioural one:

> On a topic change the agent notices, says "it looks like we're changing topics — give me a
> moment to wrap up," asks for approval, and runs a consolidation turn that outputs
> qualitative results and "ready to switch topics."

Decided: **topic-switch detection is the agent's judgment, not a deterministic trigger.** No
idle timeout or session boundary tracks a topic change with usable fidelity. The kernel's job
is to make the consolidation *operation* callable and idempotent; deciding *when* is a
behavioural pattern in prompt guidance, gated on human approval.

---

## 2. Decision

### 2.1 Five tiers, one owner and one rule each

| # | Tier | Store | Owner (sole writer) | Rule |
|---|---|---|---|---|
| 1 | **Transcript** | `sessions` / `events` | runtime | Raw, append-only. Never in the prompt beyond a bounded recency window. |
| 2 | **Episodic** | `memories` | auto-capture | **Evidence, not knowledge.** Decays. Raw material for promotion; not a permanent prompt citizen. |
| 3 | **Durable facts** | *(tier-3 store, §2.3)* | consolidation | Atomic — one claim per row, keyed, with provenance, confidence, scope, status. The tier that belongs in the prompt. |
| 4 | **Task state** | *(same store, distinct kind)* | consolidation + agent | Where we were: goal, last action, next step, open decisions, artifacts. Expires when the work lands. |
| 5 | **Narrative** | `MEMORY.md` + workspace files | human, plus the sweep's fenced block | Human-legible, hand-editable, audited by Ben. Rendered *from* tiers 3–4 plus hand prose. |

Tier 4 is separated from tier 3 deliberately. Task state is not a fact about the world; it is
a fact about the present that becomes false on success. Smuggling it into tier 3 as "facts
with weird lifetimes" is how fact stores rot.

### 2.2 Episodes are a first-class object

`memories` rows are currently grouped by nothing but `agent_id` and timestamp. Consolidation
needs a defensible input set, and "last N turns" cuts mid-thought and re-consolidates settled
work.

**Decision: assign an `episode_id` at capture time; close it on the approved topic switch.**

This is the cheapest thing on this list today and the most expensive to retrofit — adding a
column and one write now versus reconstructing episode boundaries from a year of ungrouped
rows later. It lands before any promotion work.

An episode has: id, agent, opened-at, closed-at, a short title written at close, and a
close reason (`topic-switch` / `explicit` / `timer` / `abandoned`).

### 2.3 Tier 3 shape: keyed slots, not labelled versions

#### 2.3.1 The invariant

> **A superseded fact never enters the prompt.** Not with a label, not de-emphasized, not
> ranked last.

This is the load-bearing decision of the whole tier. The rejected alternative — store both
versions with a `status` field and let the answering model tell current from stale — asks a
general-purpose model, mid-way through unrelated work, to arbitrate between two fluent
contradictory statements on the strength of a metadata field. That is a probabilistic
mitigation of a failure we can make structurally impossible. If both rows can reach the
context window, we have already lost.

So the judgment moves from **read time, general-purpose, adversarial** to **write time,
narrow, single-purpose**: the consolidation pass is the only thing that ever sees both
versions, and it faces exactly one question — *does this new claim replace any of these N
retrieved neighbours?* Small candidate set, binary answer, no competing task. Models are good
at that. They are not reliable at the read-time version.

#### 2.3.2 Enforcement is DDL, not wording

Tier 3 rows carry a **`claim_key`** — a slot name drawn from a controlled vocabulary
(`git.trunk_model`, `memory.sweep_status`, `project.<slug>.owner`), not free text.

- The live fact table holds **at most one active row per `(agent_scope, claim_key)`**, enforced
  by a uniqueness constraint. Not a convention; a constraint the database refuses to violate.
- Writing a fact for an existing key is an **update in place**. The previous row is copied to
  an append-only **`fact_history`** table with its provenance, its supersession timestamp, and
  the episode that replaced it.
- Supersession is therefore not a flag a reader has to interpret. It is a row that is no longer
  in the table the reader queries.

`fact_history` preserves the audit trail — *when did this change, and why* — reachable by an
explicit tool call, which is the only time it is wanted. It is never part of automatic recall.

Consequently the earlier `status` / `supersedes` design collapses:

- `supersedes` (nullable self-edge) is **dropped**. History lives in `fact_history`; the edge
  was only ever needed to let a reader skip stale rows, and readers no longer see them.
- `status` is **narrowed to `open` | `settled`** — the open-loop distinction from §2.5 survives
  because it is genuinely about *unfinished versus stable*, which is orthogonal to staleness.
  There is no `superseded` value, because there are no superseded rows in the live table.

Fields, final: `claim_key`, `claim`, `scope` (agent / project / user / global), `status`
(`open` | `settled`), `provenance` (episode + turn), `confidence`, `last_affirmed_at`.

#### 2.3.3 Where the risk actually goes

This does not eliminate the failure; it relocates it. **If consolidation fails to recognise two
claims as the same claim**, we get two active rows under *different* keys, both current-looking
and contradictory. Same confusion in the prompt, different cause — a dedup miss rather than a
supersession miss.

That trade is worth taking, because the two failures are not equally detectable. Duplicate keys
are visible in a table scan and reviewable. Stale-sitting-next-to-current is only discovered
when the agent confidently asserts something from two months ago. **Prefer the failure mode you
can see.**

Mitigations, in descending order of how much they are trusted:

1. **Controlled vocabulary.** Consolidation *selects* a key from the existing key space and may
   only mint a new one when nothing fits. Far narrower than "invent a name."
2. **Contradiction detection at write time is logged, not auto-resolved.** A new claim whose
   embedding is a near-neighbour of an active row under a different key raises a flag for Ben
   rather than silently landing.
3. **Periodic key-space review.** The same curation job MEMORY.md already is, now structured
   and small enough to eyeball.

#### 2.3.4 Storage choice

**Extend `memories` with a `kind` discriminator rather than adding a table** — still the call,
but it is now a closer thing than in the pre-amendment draft. The uniqueness constraint has to
be partial (`WHERE kind = 'fact'`), and `fact_history` is a new table regardless, so the
"avoid a third migration path" argument is weaker than it was.

It holds because: the embedding, decay, and per-agent scoping machinery already exists in
`memories` and is exercised; `kv_store` is untyped and shares a namespace with system keys like
`delivery.last_channel`; and one recall path with a `kind` filter is less surface than two.
`kv_store` remains what it actually is — an internal key-value scratchpad — and `memory_store` /
`memory_recall` are re-pointed at tier 3 as part of ANAI-166 option B.

*(This remains the part of the ADR most likely to be wrong, and the amendment moved it closer to
the alternative in §4.4. Recorded so that changing it is a visible decision rather than a drift.)*

### 2.4 Consolidation is an operation, not a timer

One implementation, three callers:

1. the periodic tick (`memory.consolidation_interval_hours`, currently 24, spawned in
   `kernel.rs`),
2. the agent-judged topic switch, after human approval,
3. an explicit request ("wrap up").

Requirements:

- **Idempotent.** Re-running over the same episode produces no new facts.
- **Interruptible.** A half-finished pass leaves episodic intact and durable facts unchanged.
  Promotion commits per-episode, not per-fact.
- **Keyed writes are upserts.** Every promotion targets a `claim_key`; the deterministic path
  handles exact-key replacement (free, always on) and only novel or ambiguous claims reach the
  LLM.
- **Two outputs.** Durable facts *and* task state.
- **LLM-gated and opt-in per the ANAI-160 split.** Deterministic sweeps (decay, dedup,
  same-key replacement) are free and always on; the LLM promotion pass is opt-in and budgeted.

### 2.5 The context builder: named slots with budgets

Today the only automatic reader is a top-5 similarity query over transcript sludge. Replace it
with an explicit assembler whose slots are named, budgeted, and have a stated eviction rule:

| Slot | Source | Budget | Eviction |
|---|---|---|---|
| identity / persona | workspace files | fixed | never — truncation is a bug, log it |
| open loops | tier 3, `status = open`, this agent's scope | small, unconditional | oldest-affirmed first, and say so in the prompt |
| task state | tier 4, current episode | small, unconditional | none — it is one record |
| settled facts | tier 3, retrieved by relevance | medium | lowest relevance first |
| recent turns | tier 1 window | large | oldest first |
| narrative | `MEMORY.md` | `BUDGET_MEMORY_MD` | log on truncation (ANAI-167) |

Every tier-3 slot reads the live fact table only. `fact_history` is not a source for any slot;
reaching it requires an explicit tool call. That is the read-side half of the §2.3.1 invariant,
and it should be asserted by a test, not by discipline.

The reader contract is what determines the writer's shape. Designing the promotion pass before
the assembler would get the schema wrong.

### 2.6 What renders into MEMORY.md

The sweep's fenced block renders **open loops and task state**, not the whole fact set. That
is the content which is worthless when stale and priceless at a cold start. Settled facts are
retrieved, not pasted.

Corollary, and the reason the current dry-run must not be applied: system keys never render.
Tier 5 renders from tiers 3–4 only, so `delivery.last_channel` is excluded structurally rather
than by a `delivery.*` denylist.

---

## 3. Consequences

**Kept.** The MEMORY.md sweep (ANAI-168) is the tier-5 renderer and is correct. It stays,
unapplied, until it has tier-3/4 input. Agent-scoped `memory_store` (ANAI-165) and the prompt
cap fix (ANAI-167) stand as-is.

**Parked.** "Layer 2" prompt guidance telling agents to write to memory is on hold. Telling
agents to store things before deciding which store and what shape just fills `kv_store` with
unstructured prose we would then migrate.

**Deferred as a symptom fix.** The `delivery.*` key filter is subsumed by §2.6.

**New work, in dependency order.** Episodes → tier-3 schema + `fact_history` + migration →
context builder → deterministic consolidation → LLM promotion → topic-switch behavioural
pattern → re-point `memory_store`/`memory_recall` (ANAI-166 B) → re-hang the sweep on tiers 3–4.

**Risks.**
- *Promotion quality.* An LLM promoter that writes bad facts poisons every future prompt, and
  unlike bad episodic rows these are asserted as true. Mitigation: promotion is proposal +
  ratification for anything beyond the deterministic cases, and every fact keeps provenance so
  a bad batch is revocable by episode.
- *Dedup miss.* Per §2.3.3, the residual failure is two active rows under different keys.
  Mitigation is a controlled key vocabulary plus a logged near-neighbour flag; it is a visible
  failure by construction.
- *Key-space sprawl.* If the vocabulary grows without bound, the uniqueness constraint stops
  buying much — every claim gets its own key and dedup degrades to the old problem. Untested;
  see §5.4.
- *Migration risk to a live fleet.* 35 k rows, 52 registered agents, sibling agents' memory in
  the same database. Every migration is additive-then-backfill, dry-run first, with a backup
  taken at the same commit.
- *Cost.* An LLM pass per topic switch across a fleet this size is a real bill. Hence opt-in,
  budgeted, and off by default (the same conclusion Hermes reached in v0.19).

---

## 4. Alternatives considered

1. **Keep stacking on the current shape.** Rejected: the sweep measurement showed the next
   process would render an empty tier. Every further layer inherits the inversion in §1.2.
2. **Rewrite the memory subsystem.** Rejected: the stores are individually fine. What is
   missing is a contract and one operation, not new storage engines.
3. **Make `kv_store` the fact tier.** Rejected: untyped, shares a namespace with system keys,
   no embedding, no status, no provenance.
4. **A dedicated `facts` table.** Viable, and the closest runner-up to §2.3.4 — *closer after
   the keyed-slot amendment*, since tier 3 now needs a partial uniqueness constraint and a
   companion history table anyway. Rejected for now on recall-path cost; revisit if the `kind`
   discriminator makes `memories` queries unwieldy.
5. **Deterministic topic-switch detection** (idle timeout, session boundary, command).
   Rejected on fidelity — see §1.3. Retained only as a backstop for closing abandoned
   episodes.
6. **Promote automatically without approval.** Rejected: the promoter writes into every future
   prompt. Human ratification is the cheap insurance.
7. **Version rows with a `status = superseded` and let the reader sort it out.** Rejected in
   the 2026-08-19 amendment — see §2.3.1. It makes prompt correctness depend on a model
   correctly reading a metadata field while doing something else; the keyed-slot model makes
   the bad state unrepresentable in the live table instead.

---

## 5. Open questions

1. Does task state (tier 4) survive as its own kind, or collapse into tier 3 with an expiry?
   Decided at implementation of the context builder, not before.
2. Cross-agent facts: does a `scope = global` fact render into every agent's prompt, and who
   is allowed to write one? Currently out of scope; it is the same threat surface ANAI-165
   just closed and should not be reopened casually.
3. The 879 legacy shared-namespace `kv_store` rows: leave in place under the constant UUID,
   or attempt attribution? Current answer is leave; authorship was never recorded.
4. **What does the claim-key space actually look like on real content?** The keyed-slot model
   in §2.3 rests on the vocabulary staying small and controlled. Nobody has enumerated it
   against a real corpus. First implementation step should be a read-only pass over one
   agent's episodic history proposing keys, reviewed by hand, before the constraint ships.
5. Does `claim_key` need to be scoped per-agent, per-project, or globally namespaced? The
   uniqueness constraint's column list depends on the answer and it is not obvious for
   `scope = project` facts shared by several agents.

---

## 6. Amendment log

- **2026-08-19 — §2.3 rewritten (keyed slots).** Ben raised that surfacing superseded facts
  alongside their replacements and relying on the model to tell them apart via a `status` field
  would not reliably prevent confusion. Agreed. The fix is to stop asking the model to
  arbitrate: superseded rows leave the live table entirely (`fact_history`), a partial
  uniqueness constraint on `(scope, claim_key)` makes the contradictory state unrepresentable,
  and the judgment moves to write time. `supersedes` dropped; `status` narrowed to
  `open` | `settled`. Knock-on edits: §2.1 (tier-3 rule), §2.4 (keyed upserts), §2.5 (readers
  never touch `fact_history`), §3 (new risks), §4.4 and §4.7, §5.4 and §5.5.
