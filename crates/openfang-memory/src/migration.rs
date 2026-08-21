//! SQLite schema creation and migration.
//!
//! Creates all tables needed by the memory substrate on first boot.

use rusqlite::Connection;

/// Current schema version.
const SCHEMA_VERSION: u32 = 14;

/// Run all migrations to bring the database up to date.
pub fn run_migrations(conn: &Connection) -> Result<(), rusqlite::Error> {
    let current_version = get_schema_version(conn);

    if current_version < 1 {
        migrate_v1(conn)?;
    }

    if current_version < 2 {
        migrate_v2(conn)?;
    }

    if current_version < 3 {
        migrate_v3(conn)?;
    }

    if current_version < 4 {
        migrate_v4(conn)?;
    }

    if current_version < 5 {
        migrate_v5(conn)?;
    }

    if current_version < 6 {
        migrate_v6(conn)?;
    }

    if current_version < 7 {
        migrate_v7(conn)?;
    }

    if current_version < 8 {
        migrate_v8(conn)?;
    }

    if current_version < 9 {
        migrate_v9(conn)?;
    }

    if current_version < 10 {
        migrate_v10(conn)?;
    }

    if current_version < 11 {
        migrate_v11(conn)?;
    }

    if current_version < 12 {
        migrate_v12(conn)?;
    }

    if current_version < 13 {
        migrate_v13(conn)?;
    }

    if current_version < 14 {
        migrate_v14(conn)?;
    }

    set_schema_version(conn, SCHEMA_VERSION)?;
    Ok(())
}

/// Get the current schema version from the database.
fn get_schema_version(conn: &Connection) -> u32 {
    conn.pragma_query_value(None, "user_version", |row| row.get(0))
        .unwrap_or(0)
}

/// Check if a column exists in a table (SQLite has no ADD COLUMN IF NOT EXISTS).
fn column_exists(conn: &Connection, table: &str, column: &str) -> bool {
    let sql = format!("PRAGMA table_info({})", table);
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return false;
    };
    let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(1)) else {
        return false;
    };
    let names: Vec<String> = rows.filter_map(|r| r.ok()).collect();
    names.iter().any(|n| n == column)
}

/// Set the schema version in the database.
fn set_schema_version(conn: &Connection, version: u32) -> Result<(), rusqlite::Error> {
    conn.pragma_update(None, "user_version", version)
}

/// Version 1: Create all core tables.
fn migrate_v1(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        -- Agent registry
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            manifest BLOB NOT NULL,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Session history
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            messages BLOB NOT NULL,
            context_window_tokens INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Event log
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            source_agent TEXT NOT NULL,
            target TEXT NOT NULL,
            payload BLOB NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_agent);

        -- Key-value store (per-agent)
        CREATE TABLE IF NOT EXISTS kv_store (
            agent_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value BLOB NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, key)
        );

        -- Task queue
        CREATE TABLE IF NOT EXISTS task_queue (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            payload BLOB NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 0,
            scheduled_at TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_task_status_priority ON task_queue(status, priority DESC);

        -- Semantic memories
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'episodic',
            confidence REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            deleted INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);

        -- Knowledge graph entities
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            properties TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Knowledge graph relations
        CREATE TABLE IF NOT EXISTS relations (
            id TEXT PRIMARY KEY,
            source_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            properties TEXT NOT NULL DEFAULT '{}',
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity);
        CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

        -- Migration tracking
        CREATE TABLE IF NOT EXISTS migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        );

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (1, datetime('now'), 'Initial schema');
        ",
    )?;
    Ok(())
}

/// Version 2: Add collaboration columns to task_queue for agent task delegation.
fn migrate_v2(conn: &Connection) -> Result<(), rusqlite::Error> {
    // SQLite requires one ALTER TABLE per statement; check before adding
    let cols = [
        ("title", "TEXT DEFAULT ''"),
        ("description", "TEXT DEFAULT ''"),
        ("assigned_to", "TEXT DEFAULT ''"),
        ("created_by", "TEXT DEFAULT ''"),
        ("result", "TEXT DEFAULT ''"),
    ];
    for (name, typedef) in &cols {
        if !column_exists(conn, "task_queue", name) {
            conn.execute(
                &format!("ALTER TABLE task_queue ADD COLUMN {} {}", name, typedef),
                [],
            )?;
        }
    }

    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (2, datetime('now'), 'Add collaboration columns to task_queue')",
        [],
    )?;

    Ok(())
}

/// Version 3: Add embedding column to memories table for vector search.
fn migrate_v3(conn: &Connection) -> Result<(), rusqlite::Error> {
    if !column_exists(conn, "memories", "embedding") {
        conn.execute(
            "ALTER TABLE memories ADD COLUMN embedding BLOB DEFAULT NULL",
            [],
        )?;
    }
    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (3, datetime('now'), 'Add embedding column to memories')",
        [],
    )?;
    Ok(())
}

/// Version 4: Add usage_events table for cost tracking and metering.
fn migrate_v4(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS usage_events (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            tool_calls INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_usage_agent_time ON usage_events(agent_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_events(timestamp);

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (4, datetime('now'), 'Add usage_events table for cost tracking');
        ",
    )?;
    Ok(())
}

/// Version 5: Add canonical_sessions table for cross-channel persistent memory.
fn migrate_v5(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS canonical_sessions (
            agent_id TEXT PRIMARY KEY,
            messages BLOB NOT NULL,
            compaction_cursor INTEGER NOT NULL DEFAULT 0,
            compacted_summary TEXT,
            updated_at TEXT NOT NULL
        );

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (5, datetime('now'), 'Add canonical_sessions for cross-channel memory');
        ",
    )?;
    Ok(())
}

/// Version 6: Add label column to sessions table.
fn migrate_v6(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Check if column already exists before ALTER (SQLite has no ADD COLUMN IF NOT EXISTS)
    if !column_exists(conn, "sessions", "label") {
        conn.execute("ALTER TABLE sessions ADD COLUMN label TEXT", [])?;
    }
    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (6, datetime('now'), 'Add label column to sessions for human-readable labels')",
        [],
    )?;
    Ok(())
}

/// Version 7: Add paired_devices table for device pairing persistence.
fn migrate_v7(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS paired_devices (
            device_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            platform TEXT NOT NULL,
            paired_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            push_token TEXT
        );

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (7, datetime('now'), 'Add paired_devices table for device pairing');
        ",
    )?;
    Ok(())
}

/// Version 8: Add audit_entries table for persistent Merkle audit trail.
fn migrate_v8(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS audit_entries (
            seq INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT NOT NULL,
            outcome TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_entries(agent_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_entries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_entries(action);

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (8, datetime('now'), 'Add audit_entries table for persistent Merkle audit trail');
        ",
    )?;
    Ok(())
}

/// Version 9: Add session_participants table for per-actor presence and
/// the snowflake -> identity binding (ANAI-127/128 turn-context envelope).
///
/// Keyed on (session_id, speaker_id) where speaker_id is the durable actor
/// snowflake. `last_msg_at` is the presence clock; `display_name` folds in
/// the identity label so identity and presence live in one artifact.
fn migrate_v9(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS session_participants (
            session_id TEXT NOT NULL,
            speaker_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_msg_at TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (session_id, speaker_id)
        );
        CREATE INDEX IF NOT EXISTS idx_participants_session_seen
            ON session_participants(session_id, last_msg_at DESC);

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (9, datetime('now'), 'Add session_participants for per-actor presence and identity');
        ",
    )?;
    Ok(())
}

/// Version 10: Add identity_bindings — the authoritative, curated snowflake ->
/// display-name map (ANAI-127 rung 1). This is the ONE place an operator can
/// assert "this snowflake is Teo" regardless of what the platform reports.
///
/// Resolution hierarchy for a turn's speaker name:
///   1. identity_bindings.openfang_name  (authoritative — this table)
///   2. Discord global_name              (user-chosen, nullable)
///   3. username / handle                (last resort)
///
/// Deliberately fleet-wide (keyed on speaker_id alone, NOT per-session): an
/// operator's mapping should hold across every channel and agent. Left EMPTY by
/// the migration on purpose — bindings are runtime data, not schema, so no
/// operator's snowflake is ever baked into a fresh database.
fn migrate_v10(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS identity_bindings (
            speaker_id TEXT PRIMARY KEY,
            openfang_name TEXT NOT NULL,
            note TEXT,
            updated_at TEXT NOT NULL
        );

        INSERT OR IGNORE INTO migrations (version, applied_at, description)
        VALUES (10, datetime('now'), 'Add identity_bindings for authoritative snowflake -> name mapping');
        ",
    )?;
    Ok(())
}

/// Version 11: Add `claimed_at` to `task_queue` (ANAI-147).
///
/// The wake queue's per-caller in-flight cap (ANAI-104) counts rows sitting in
/// `in_progress`, and only `task_complete` clears them. Nothing recorded WHEN a
/// row was claimed, so a claim whose dispatcher died could not be distinguished
/// from one still running — leaving no safe basis for a stale-claim sweep.
/// `claimed_at` is stamped at the claim/flip and is the reaper's clock.
///
/// Nullable with no default on purpose: rows claimed by a pre-migration binary
/// legitimately have no claim time, and the reaper falls back to `created_at`
/// for those rather than pretending they were claimed just now (which would
/// make a leaked pre-migration row immortal all over again).
fn migrate_v11(conn: &Connection) -> Result<(), rusqlite::Error> {
    if !column_exists(conn, "task_queue", "claimed_at") {
        conn.execute("ALTER TABLE task_queue ADD COLUMN claimed_at TEXT", [])?;
    }
    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (11, datetime('now'), 'Add claimed_at to task_queue for stale-wake reaping')",
        [],
    )?;
    Ok(())
}

/// Version 12: Episodes — the boundary object episodic capture groups into
/// (ADR 0001 §2.2, ANAI-74 family).
///
/// Two changes, one migration:
///
/// 1. **`episodes`** — id, agent, opened/closed timestamps, a title written at
///    close, an optional wrap-up summary, and a close reason (`topic-switch` /
///    `explicit` / `timer` / `abandoned`). `last_activity_at` is the idle
///    timer's clock and
///    `turn_count` is bookkeeping for sizing consolidation input.
///
///    The partial unique index is the load-bearing part: **at most one open
///    episode per agent**, enforced by the database rather than by whichever
///    code path happens to run first. The open episode IS the row with
///    `closed_at IS NULL`, which is what makes the lifecycle survive a daemon
///    restart without any in-memory state to lose.
///
/// 2. **`memories.episode_id`** — nullable, no backfill, indexed. Written by
///    `semantic.rs`, lifted out of the capture metadata map so the capture
///    signatures did not have to change fleet-wide.
///
///    Nullable and unbackfilled ON PURPOSE. Pre-episode rows (~35k) have no
///    defensible episode, and inventing one would be worse than admitting the
///    gap: NULL *is* the legacy marker. It also keeps this migration a pure
///    schema add — it rewrites no existing row, so rolling the binary back
///    loses the column's readers, never any data.
fn migrate_v12(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL,
            closed_at TEXT,
            title TEXT,
            summary TEXT,
            close_reason TEXT,
            turn_count INTEGER NOT NULL DEFAULT 0
        );

        -- At most one OPEN episode per agent. Partial, so any number of closed
        -- episodes coexist. This is the constraint the lifecycle relies on.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_one_open_per_agent
            ON episodes(agent_id) WHERE closed_at IS NULL;
        CREATE INDEX IF NOT EXISTS idx_episodes_agent_opened
            ON episodes(agent_id, opened_at DESC);
        -- Drives the idle sweep (`EpisodeStore::sweep_idle`).
        CREATE INDEX IF NOT EXISTS idx_episodes_open_activity
            ON episodes(last_activity_at) WHERE closed_at IS NULL;
        ",
    )?;

    if !column_exists(conn, "memories", "episode_id") {
        conn.execute("ALTER TABLE memories ADD COLUMN episode_id TEXT", [])?;
    }
    // `summary` (ANAI-194) was added to the CREATE TABLE above after v12 was
    // first written but before it shipped anywhere, so a fresh database gets it
    // inline. This guard covers the one case the inline column cannot: a
    // developer tree that already applied the earlier v12 and would otherwise
    // skip the whole migration on version number alone. Cheap, idempotent, and
    // it means nobody has to hand-repair a local database.
    if !column_exists(conn, "episodes", "summary") {
        conn.execute("ALTER TABLE episodes ADD COLUMN summary TEXT", [])?;
    }
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id)",
        [],
    )?;

    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (12, datetime('now'), 'Add episodes table and memories.episode_id (ADR 0001 2.2)')",
        [],
    )?;
    Ok(())
}

/// v13: promote `kind` from a metadata key to a real column on `memories`.
///
/// `kind` is the row-type discriminator the tool surface already writes into
/// the metadata JSON (`turn` / `note` / `store` / `summary`, with `fact`
/// reserved for stage 3). Filtering on it meant `json_extract` over every
/// undeleted row; stage 3 will filter on it constantly, and supersession keys
/// off `kind` + claim-key, so it earns storage.
///
/// Same shape as the v12 `episode_id` add and for the same reasons: a trailing
/// `ALTER` plus an index, nullable, no rewrite of existing rows. Rows written
/// before this migration keep `kind` in their JSON and have a NULL column;
/// `semantic.rs` hydrates the column back into the metadata map on read, so
/// old and new rows are indistinguishable to consumers. Backfill is a separate,
/// deliberate operation — a migration that guesses at 46k historical rows is a
/// migration that cannot be rolled back.
fn migrate_v13(conn: &Connection) -> Result<(), rusqlite::Error> {
    if !column_exists(conn, "memories", "kind") {
        conn.execute("ALTER TABLE memories ADD COLUMN kind TEXT", [])?;
    }
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)",
        [],
    )?;

    // Lift the key that already exists into the column it now belongs in.
    //
    // This is NOT the backfill: it invents nothing. `memory_note` and
    // `memory_store` have been writing `kind` into the metadata JSON since the
    // tool surface landed, and once `semantic.rs` filters on the column alone
    // those rows would silently stop matching. Copying the value across is what
    // makes "has the key" and "has the column" the same population, which is
    // the precondition the column-only filter relies on.
    //
    // Idempotent (`WHERE kind IS NULL`), lossless (the JSON copy is left in
    // place, and the read path prefers the column), and reversible: dropping
    // back to v12 loses nothing, because nothing was removed.
    //
    // Guessing a `kind` for the rows that never had one is a separate,
    // deliberate operation run by hand. A migration that infers row types from
    // `source` is a migration you cannot undo.
    conn.execute(
        "UPDATE memories SET kind = json_extract(metadata, '$.kind')
         WHERE kind IS NULL AND json_valid(metadata)
           AND json_type(metadata, '$.kind') = 'text'",
        [],
    )?;

    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (13, datetime('now'), 'Add memories.kind column and index (ADR 0002 2.2)')",
        [],
    )?;
    Ok(())
}

/// v14: the tier-3 claim store — keyed fact slots on `memories`, plus the
/// append-only `fact_history` audit table (ADR 0001 §2.3.2, ADR 0002 §2.5).
///
/// Five changes, one migration, all additive:
///
/// 1. **`memories.claim_key`** — the slot name a tier-3 row occupies, drawn
///    from a controlled vocabulary (`repo.trunk_model`, `memory.sweep_status`,
///    …). NULL for every non-fact row, which is every row that exists today.
///
/// 2. **`memories.status`** — `open` | `settled` per §2.3.2. Deliberately no
///    `superseded` value: a superseded fact is not a flagged row, it is a row
///    that has left this table (see 4). `last_affirmed_at` records the last
///    time a write re-asserted the same claim without changing it.
///
/// 3. **`memories.scope_ref`** — *what the fact is about*: the agent id for an
///    `agent`-scoped claim, the project slug for a `project` one, the user
///    slug for a `user` one. **`memories.authored_by`** records who wrote it,
///    which is a different question and is now metadata rather than address.
///
/// 4. **The uniqueness constraint is the whole design.** The partial unique
///    index holds **at most one live fact per `(scope, scope_ref, claim_key)`**.
///    Partial on `kind = 'fact' AND deleted = 0`, so it constrains nothing
///    outside tier 3 and a soft-deleted fact frees its slot. This is §2.3.1's
///    invariant enforced by DDL rather than by whichever reader remembers to
///    filter — a superseded fact cannot reach the prompt because it is not in
///    the table the recall path queries.
///
///    **Amended in place on 2026-08-21, before any fact row existed.** The
///    first cut keyed the slot on `(agent_id, scope, claim_key)`, which put
///    *the author* in the address. Three agents found the same defect
///    independently from three different use cases: two agents writing the
///    same project claim produce two live rows that disagree and neither
///    supersedes the other, so supersession — the entire feature — was off for
///    every scope except `agent`. `kimiya-alpha`'s framing is the one to keep:
///    a disposable worker that writes a fact and is then deleted leaves an
///    orphan row holding the live value for a claim nobody can overwrite, so
///    *make the row belong to the project, not the corpse*. `alpha`'s hot-file
///    ledger is the same shape — a ledger row is owned by the file, not by
///    whoever claimed it. Rewriting the migration rather than stacking a v15
///    is honest here precisely because no binary carrying v14 has ever booted:
///    there is no deployed schema to be inconsistent with.
///
///    `scope_ref` is never NULL on a fact row, and that is load-bearing rather
///    than tidy: SQLite treats NULLs as distinct inside a unique index, so a
///    NULL ref would silently switch off the constraint it is part of. The
///    writer derives or demands it before opening a transaction.
///
/// 5. **`fact_history`** — append-only, written by the upsert inside the same
///    transaction that replaces the live row. It carries the superseded
///    claim's text, its `authored_by`, and the episode that replaced it, so
///    "who believed this, and when did we stop" survives the overwrite.
///
///    **It has no `embedding` column, on purpose.** The only reader is
///    `memory_history`, an exact-`claim_key` lookup; it never needs vector
///    search. An embedded history row would be one forgotten `WHERE` away from
///    a superseded claim surfacing in semantic recall, which is precisely the
///    failure §2.3.1 exists to make structurally impossible. Decision ratified
///    by Ben 2026-08-21.
///
/// House style, same as v12/v13: guarded `ALTER`s and `CREATE ... IF NOT
/// EXISTS`, no existing row rewritten, no backfill. Rolling the binary back to
/// v13 loses this migration's readers and never any data.
fn migrate_v14(conn: &Connection) -> Result<(), rusqlite::Error> {
    let cols = [
        ("claim_key", "TEXT"),
        ("status", "TEXT"),
        ("last_affirmed_at", "TEXT"),
        ("scope_ref", "TEXT"),
        ("authored_by", "TEXT"),
    ];
    for (name, typedef) in &cols {
        if !column_exists(conn, "memories", name) {
            conn.execute(
                &format!("ALTER TABLE memories ADD COLUMN {} {}", name, typedef),
                [],
            )?;
        }
    }

    // A development database that ran the pre-amendment v14 carries the old
    // author-keyed table. `CREATE TABLE IF NOT EXISTS` would silently leave it
    // in place with the wrong columns, which is the worst of the three
    // outcomes: no error, wrong schema. Dropping is safe here and only here —
    // no shipped binary has ever written a `fact_history` row.
    if !column_exists(conn, "fact_history", "scope_ref") {
        conn.execute("DROP TABLE IF EXISTS fact_history", [])?;
    }

    conn.execute_batch(
        "
        -- Superseded by the 2026-08-21 amendment: the author was in the slot
        -- address, so two agents could hold two live claims for one key.
        DROP INDEX IF EXISTS idx_memories_one_live_fact_per_key;

        -- At most one LIVE fact per (scope, subject, claim key). Partial on
        -- both `kind` and `deleted`, so: non-fact rows are unconstrained, and a
        -- soft-deleted fact releases its slot for a future write.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_one_live_fact_per_slot
            ON memories(scope, scope_ref, claim_key)
            WHERE kind = 'fact' AND deleted = 0;

        -- Drives fact lookup by slot (the read side of the upsert).
        CREATE INDEX IF NOT EXISTS idx_memories_claim_key
            ON memories(claim_key) WHERE kind = 'fact';

        -- Append-only supersession audit. NO embedding column: see doc comment.
        -- `authored_by` rather than `agent_id`: the history row records who
        -- believed the superseded claim, which is metadata, not address.
        CREATE TABLE IF NOT EXISTS fact_history (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            authored_by TEXT,
            scope TEXT NOT NULL,
            scope_ref TEXT NOT NULL,
            claim_key TEXT NOT NULL,
            claim TEXT NOT NULL,
            status TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            episode_id TEXT,
            created_at TEXT NOT NULL,
            superseded_at TEXT NOT NULL,
            superseded_by_episode TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fact_history_key
            ON fact_history(scope, scope_ref, claim_key, superseded_at DESC);
        ",
    )?;

    conn.execute(
        "INSERT OR IGNORE INTO migrations (version, applied_at, description) VALUES (14, datetime('now'), 'Add tier-3 fact slots keyed on (scope, scope_ref, claim_key) and fact_history (ADR 0001 2.3.2)')",
        [],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v13 must move `kind` from the metadata JSON into the column for rows
    /// that already had it. Without this lift, `semantic.rs`'s column-only
    /// `kind` filter would silently stop matching every note and store written
    /// before the promotion.
    #[test]
    fn v13_lifts_an_existing_metadata_kind_into_the_column() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Two legacy shapes plus one that must be left alone.
        conn.execute_batch(
            "UPDATE memories SET kind = NULL;
             INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, kind)
             VALUES ('a', 'ag', 'note row', '\"conversation\"', 'episodic', 1.0, '{\"kind\":\"note\"}', 'now', 'now', 0, 0, NULL),
                    ('b', 'ag', 'malformed row', '\"conversation\"', 'episodic', 1.0, '{\"kind\":42}', 'now', 'now', 0, 0, NULL),
                    ('c', 'ag', 'no kind row', '\"conversation\"', 'episodic', 1.0, '{}', 'now', 'now', 0, 0, NULL);",
        )
        .unwrap();

        conn.pragma_update(None, "user_version", 12).unwrap();
        run_migrations(&conn).unwrap();

        let kind_of = |id: &str| -> Option<String> {
            conn.query_row("SELECT kind FROM memories WHERE id = ?1", [id], |r| {
                r.get(0)
            })
            .unwrap()
        };
        assert_eq!(kind_of("a").as_deref(), Some("note"), "string kind lifts");
        assert_eq!(kind_of("b"), None, "a non-string kind is not a kind");
        assert_eq!(kind_of("c"), None, "absent stays absent");

        // The JSON copy is left in place: the lift is additive and reversible.
        let still_json: Option<String> = conn
            .query_row(
                "SELECT json_extract(metadata, '$.kind') FROM memories WHERE id = 'a'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still_json.as_deref(), Some("note"));
    }

    #[test]
    fn test_migration_creates_tables() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Verify tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert!(tables.contains(&"agents".to_string()));
        assert!(tables.contains(&"sessions".to_string()));
        assert!(tables.contains(&"kv_store".to_string()));
        assert!(tables.contains(&"memories".to_string()));
        assert!(tables.contains(&"entities".to_string()));
        assert!(tables.contains(&"relations".to_string()));
        assert!(tables.contains(&"session_participants".to_string()));
        assert!(tables.contains(&"identity_bindings".to_string()));
        assert!(tables.contains(&"episodes".to_string()));
        assert!(column_exists(&conn, "memories", "episode_id"));
    }

    #[test]
    fn test_migration_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        run_migrations(&conn).unwrap(); // Should not error
    }

    /// Helper: insert a tier-3 fact row directly, bypassing the (not yet
    /// written) store, so these tests pin the DDL and nothing else.
    fn insert_fact(
        conn: &Connection,
        id: &str,
        author: &str,
        scope: &str,
        scope_ref: &str,
        key: &str,
        deleted: i64,
    ) -> Result<usize, rusqlite::Error> {
        conn.execute(
            "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata,
                                   created_at, accessed_at, access_count, deleted, kind,
                                   claim_key, status, scope_ref, authored_by)
             VALUES (?1, ?2, 'claim text', '\"tool\"', ?3, 1.0, '{}', 'now', 'now', 0, ?6,
                     'fact', ?4, 'settled', ?5, ?2)",
            rusqlite::params![id, author, scope, key, scope_ref, deleted],
        )
    }

    /// §2.3.1 enforced by DDL: two LIVE facts cannot occupy the same slot.
    /// If this test ever goes green-by-deletion, supersession has stopped being
    /// a constraint and gone back to being a convention.
    #[test]
    fn v14_rejects_a_second_live_fact_in_the_same_slot() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        insert_fact(&conn, "f1", "ag", "agent", "ag", "repo.trunk_model", 0).unwrap();
        let dup = insert_fact(&conn, "f2", "ag", "agent", "ag", "repo.trunk_model", 0);
        assert!(
            dup.is_err(),
            "a second live fact in one slot must be rejected"
        );

        // Different scope, different subject, and different key are all
        // distinct slots.
        insert_fact(&conn, "f3", "ag", "project", "tttb", "repo.trunk_model", 0).unwrap();
        insert_fact(&conn, "f4", "ag", "agent", "other", "repo.trunk_model", 0).unwrap();
        insert_fact(&conn, "f5", "ag", "agent", "ag", "memory.sweep_status", 0).unwrap();
    }

    /// The 2026-08-21 amendment, pinned: the slot belongs to its subject, so a
    /// second agent writing the same project claim collides instead of opening
    /// a private duplicate. Under the original `(agent_id, scope, claim_key)`
    /// index this insert succeeded and left two live rows disagreeing with each
    /// other — supersession switched off for every scope but `agent`.
    #[test]
    fn v14_slot_is_owned_by_its_subject_not_its_author() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        insert_fact(
            &conn,
            "f1",
            "alpha",
            "project",
            "openfang",
            "project.hot_file_owner",
            0,
        )
        .unwrap();
        let other_author = insert_fact(
            &conn,
            "f2",
            "tools",
            "project",
            "openfang",
            "project.hot_file_owner",
            0,
        );
        assert!(
            other_author.is_err(),
            "two authors, one project slot: the second write must collide, not fork"
        );
    }

    /// A soft-deleted fact frees its slot — otherwise `forget` would poison a
    /// claim key permanently.
    #[test]
    fn v14_soft_deleted_fact_frees_its_slot() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        insert_fact(&conn, "f1", "ag", "agent", "ag", "repo.trunk_model", 1).unwrap();
        insert_fact(&conn, "f2", "ag", "agent", "ag", "repo.trunk_model", 1).unwrap();
        insert_fact(&conn, "f3", "ag", "agent", "ag", "repo.trunk_model", 0).unwrap();
    }

    /// The partial index must not constrain the 13k non-fact rows, which all
    /// have a NULL claim_key.
    #[test]
    fn v14_leaves_non_fact_rows_unconstrained() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        conn.execute_batch(
            "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, kind)
             VALUES ('t1', 'ag', 'turn one', '\"conversation\"', 'episodic', 1.0, '{}', 'now', 'now', 0, 0, 'turn'),
                    ('t2', 'ag', 'turn two', '\"conversation\"', 'episodic', 1.0, '{}', 'now', 'now', 0, 0, 'turn');",
        )
        .unwrap();
    }

    /// `fact_history` is the audit path only. An embedding column here is one
    /// forgotten WHERE away from a superseded claim reaching the prompt.
    #[test]
    fn v14_fact_history_carries_no_embedding() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        assert!(
            !column_exists(&conn, "fact_history", "embedding"),
            "fact_history must never carry an embedding (ADR 0001 2.3.1)"
        );
        assert!(column_exists(&conn, "memories", "claim_key"));
        assert!(column_exists(&conn, "memories", "status"));
        assert!(column_exists(&conn, "memories", "last_affirmed_at"));
        assert!(column_exists(&conn, "memories", "scope_ref"));
        assert!(column_exists(&conn, "memories", "authored_by"));
        assert!(column_exists(&conn, "fact_history", "scope_ref"));
        assert!(column_exists(&conn, "fact_history", "authored_by"));
    }

    /// A database that ran the pre-amendment v14 must end up with the amended
    /// schema, not with the old table quietly surviving a
    /// `CREATE TABLE IF NOT EXISTS`. This is the only failure mode of amending
    /// a migration in place, so it gets a test rather than a comment.
    #[test]
    fn v14_amendment_replaces_a_pre_amendment_schema() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Rebuild the shape the first cut produced.
        conn.execute_batch(
            "DROP INDEX IF EXISTS idx_memories_one_live_fact_per_slot;
             DROP TABLE fact_history;
             CREATE UNIQUE INDEX idx_memories_one_live_fact_per_key
                 ON memories(agent_id, scope, claim_key)
                 WHERE kind = 'fact' AND deleted = 0;
             CREATE TABLE fact_history (
                 id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                 scope TEXT NOT NULL, claim_key TEXT NOT NULL, claim TEXT NOT NULL,
                 status TEXT, confidence REAL NOT NULL DEFAULT 1.0,
                 metadata TEXT NOT NULL DEFAULT '{}', episode_id TEXT,
                 created_at TEXT NOT NULL, superseded_at TEXT NOT NULL,
                 superseded_by_episode TEXT);",
        )
        .unwrap();

        migrate_v14(&conn).unwrap();

        assert!(column_exists(&conn, "fact_history", "scope_ref"));
        assert!(!column_exists(&conn, "fact_history", "agent_id"));
        let stale: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_memories_one_live_fact_per_key'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stale, 0, "the author-keyed index must not survive");

        // And the amended constraint is actually the live one.
        insert_fact(&conn, "f1", "a", "project", "p", "project.owner", 0).unwrap();
        assert!(insert_fact(&conn, "f2", "b", "project", "p", "project.owner", 0).is_err());
    }
}
