//! Tier-3 claim store: keyed fact slots with in-place supersession
//! (ADR 0001 §2.3, schema v14).
//!
//! A **fact** is a `memories` row with `kind = 'fact'` that occupies a *slot*
//! named by `(agent_id, scope, claim_key)`. The slot holds at most one live
//! row, enforced by the v14 partial unique index — not by convention.
//!
//! # What this module is for
//!
//! [`FactStore::upsert`] is the only supported way to write a slot, because it
//! is the only thing that keeps the two halves of §2.3.1 together:
//!
//! > **A superseded fact never enters the prompt.** Not with a label, not
//! > de-emphasized, not ranked last.
//!
//! Superseding a claim is *one transaction*: copy the outgoing row into the
//! append-only `fact_history` table, then overwrite the live row in place. If
//! those two steps could interleave or half-fail, the invariant would be a
//! hope rather than a property — either two live claims for one key (which the
//! index refuses) or a lost audit trail. So they share a transaction, and a
//! caller who hand-rolls an `INSERT` into `memories` with `kind = 'fact'` is
//! bypassing the audit path, not merely this API.
//!
//! # The slot id is stable across supersession
//!
//! Writing a new claim to an existing key **updates the row in place**; the
//! `memories.id` does not change. The id therefore names *the slot*, not *the
//! claim version*, which is what makes `fact_history.memory_id` a usable
//! join key for "show me every claim that has ever occupied this slot".
//!
//! `created_at` is re-stamped on supersession and the outgoing value travels
//! with the history row, so each history entry spans exactly the interval
//! `created_at .. superseded_at` during which that claim was the live one.
//! Without the re-stamp every version would report the slot's birthday and the
//! audit trail could not answer "how long did we believe that?".
//!
//! # Affirmation is not supersession
//!
//! Re-asserting a claim that has not changed is common (a consolidation pass
//! re-derives the same conclusion) and writing a history row for it would bury
//! the real changes in noise. So a write whose `claim` **and** `status` match
//! the live row is an [`FactOutcome::Affirmed`]: no history row, and
//! `last_affirmed_at` moves forward.
//!
//! Change is judged on `claim` and `status` only. Confidence, metadata and
//! provenance are current-state attributes of an unchanged claim, so an
//! affirmation refreshes them in place — otherwise a caller could never
//! correct a confidence without also faking a claim change.
//!
//! # What this module deliberately does not do
//!
//! - **No embedding generation.** The caller supplies the vector, same
//!   contract as [`crate::semantic::SemanticStore::remember_with_embedding`].
//! - **No recall.** Reading facts back into a prompt is the recall path's job
//!   (step 3); [`FactStore::get`] and [`FactStore::history`] here are the
//!   exact-key lookups the audit tool needs.
//!
//! # Vocabulary is enforced here, and that is not a silent refusal
//!
//! `upsert` validates `scope` and `claim_key` against
//! [`crate::vocabulary`] before it opens a transaction, and returns
//! [`OpenFangError::InvalidInput`] if either is outside the space (§2.3.3
//! mitigation 1). The earlier worry — that a storage layer refusing a key
//! makes a slot look empty when it is not — is about *silent* refusal. A
//! typed error that quotes the grammar back is the opposite: nothing is
//! written, nothing is hidden, and the caller learns why.
//!
//! The validator itself is pure and public, so the tool surface (step 5) can
//! reject a key and explain it to the model *before* the write is attempted.
//! This layer is the backstop that makes the vocabulary a property of the
//! store rather than a habit of its callers.

use crate::vocabulary::{ClaimKey, FactScope};
use chrono::Utc;
use openfang_types::agent::AgentId;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::memory::{MemoryId, MemorySource};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// `kind` value for a tier-3 claim row.
///
/// Reserved by the v13 doc comment and made real here. The v14 partial unique
/// index keys off this exact string, so it lives next to the writer rather
/// than being spelled out at each call site.
pub const KIND_FACT: &str = "fact";

/// How many existing keys a rejection lists back to the caller.
///
/// Enough to be a usable menu, few enough that it does not swamp the reason
/// for the rejection in a model's context.
const MAX_SUGGESTED_KEYS: i64 = 25;

/// Lifecycle of a claim: unfinished versus stable (ADR 0001 §2.3.2).
///
/// **There is no `Superseded` variant, and adding one would be a bug.** A
/// superseded claim is not a live row wearing a label; it is a row that has
/// left `memories` for `fact_history`. The whole point of §2.3.1 is that no
/// reader has to interpret a staleness flag, because no reader can see a stale
/// row. A `Superseded` status would hand that judgment back to the answering
/// model, which is the design this ADR rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactStatus {
    /// An open loop: something unfinished, and a candidate for the open-loop
    /// prompt slot (§2.5).
    Open,
    /// A settled claim: believed stable until something supersedes it.
    Settled,
}

impl FactStatus {
    /// The stored form. This is what lands in `memories.status`.
    pub fn as_str(self) -> &'static str {
        match self {
            FactStatus::Open => "open",
            FactStatus::Settled => "settled",
        }
    }

    /// Parse a stored value back.
    ///
    /// Unknown values are an error rather than a silent fallback to `Settled`:
    /// a row whose status we cannot read is exactly the row we must not guess
    /// about, and defaulting would quietly promote a stray value into the
    /// settled set.
    pub fn parse(s: &str) -> OpenFangResult<Self> {
        match s {
            "open" => Ok(FactStatus::Open),
            "settled" => Ok(FactStatus::Settled),
            other => Err(OpenFangError::Memory(format!(
                "unknown fact status {other:?} (expected \"open\" or \"settled\")"
            ))),
        }
    }
}

impl std::fmt::Display for FactStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One write against one slot.
///
/// Built by [`FactWrite::new`] and adjusted with the builder setters, so
/// adding a field later is not a breaking change at every call site.
#[derive(Debug, Clone)]
pub struct FactWrite {
    /// Owning agent — first component of the slot key.
    pub agent_id: AgentId,
    /// `agent` / `project` / `user` / `global` (§2.3.2). Second component of
    /// the slot key: the same claim key under two scopes is two slots.
    pub scope: String,
    /// Slot name from the controlled vocabulary, e.g. `git.trunk_model`.
    pub claim_key: String,
    /// The claim itself, in prose. Compared verbatim to decide affirm versus
    /// supersede.
    pub claim: String,
    /// Open loop or settled claim.
    pub status: FactStatus,
    /// How sure we are, 0.0..=1.0.
    pub confidence: f64,
    /// Where the claim came from.
    pub source: MemorySource,
    /// Provenance: the episode this write happened in. Recorded on the live
    /// row, and on supersession also as the history row's
    /// `superseded_by_episode` — "which episode replaced this".
    pub episode_id: Option<String>,
    /// Free-form sidecar, serialized to the `metadata` JSON column.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Embedding of `claim`, if the caller has one.
    pub embedding: Option<Vec<f32>>,
}

impl FactWrite {
    /// A settled claim with full confidence, no provenance, no embedding.
    pub fn new(
        agent_id: AgentId,
        scope: impl Into<String>,
        claim_key: impl Into<String>,
        claim: impl Into<String>,
    ) -> Self {
        Self {
            agent_id,
            scope: scope.into(),
            claim_key: claim_key.into(),
            claim: claim.into(),
            status: FactStatus::Settled,
            confidence: 1.0,
            source: MemorySource::Inference,
            episode_id: None,
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    /// Mark this claim open or settled.
    pub fn with_status(mut self, status: FactStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the originating source.
    pub fn with_source(mut self, source: MemorySource) -> Self {
        self.source = source;
        self
    }

    /// Attach the episode this write happened in.
    pub fn with_episode(mut self, episode_id: impl Into<String>) -> Self {
        self.episode_id = Some(episode_id.into());
        self
    }

    /// Attach the sidecar metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attach an embedding of the claim text.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// What [`FactStore::upsert`] actually did.
///
/// Returned rather than logged because the caller usually cares: a
/// consolidation pass wants to report supersessions to Ben and stay quiet
/// about affirmations, and it cannot tell them apart from the outside without
/// re-reading the row it just wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactOutcome {
    /// The slot was empty (or held only soft-deleted rows). A new row was
    /// inserted.
    Created { id: MemoryId },
    /// The claim and status were unchanged. `last_affirmed_at` moved forward;
    /// **no history row was written**, because nothing was superseded.
    Affirmed { id: MemoryId },
    /// The claim (or its status) changed. The outgoing version is in
    /// `fact_history` under `history_id`; the live row keeps its id.
    Superseded { id: MemoryId, history_id: Uuid },
}

impl FactOutcome {
    /// The slot's row id, whichever branch was taken.
    pub fn id(&self) -> MemoryId {
        match self {
            FactOutcome::Created { id }
            | FactOutcome::Affirmed { id }
            | FactOutcome::Superseded { id, .. } => *id,
        }
    }

    /// True when this write replaced an existing claim.
    pub fn superseded(&self) -> bool {
        matches!(self, FactOutcome::Superseded { .. })
    }
}

/// A live claim occupying a slot.
#[derive(Debug, Clone)]
pub struct Fact {
    /// Row id — stable for the life of the slot.
    pub id: MemoryId,
    /// Owning agent.
    pub agent_id: String,
    /// Slot scope.
    pub scope: String,
    /// Slot name.
    pub claim_key: String,
    /// The current claim.
    pub claim: String,
    /// Open or settled.
    pub status: FactStatus,
    /// Confidence, 0.0..=1.0.
    pub confidence: f64,
    /// Episode that wrote the current claim.
    pub episode_id: Option<String>,
    /// When *this claim* became live (re-stamped on supersession).
    pub created_at: String,
    /// When this claim was last re-asserted.
    pub last_affirmed_at: Option<String>,
    /// Sidecar metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A claim that used to occupy a slot.
///
/// Deliberately has no embedding field, mirroring the table: `fact_history`
/// carries none, so a superseded claim has no vector that could surface in
/// semantic recall (v14 doc comment, §2.3.1).
#[derive(Debug, Clone)]
pub struct FactHistoryEntry {
    /// This history row's id.
    pub id: Uuid,
    /// The slot row this claim used to be.
    pub memory_id: MemoryId,
    /// Slot scope.
    pub scope: String,
    /// Slot name.
    pub claim_key: String,
    /// The superseded claim.
    pub claim: String,
    /// Its status at the moment it was replaced.
    pub status: Option<FactStatus>,
    /// Its confidence.
    pub confidence: f64,
    /// The episode that wrote it.
    pub episode_id: Option<String>,
    /// When it became live.
    pub created_at: String,
    /// When it stopped being live.
    pub superseded_at: String,
    /// The episode that replaced it.
    pub superseded_by_episode: Option<String>,
}

/// Reader/writer for tier-3 claim slots.
#[derive(Clone)]
pub struct FactStore {
    conn: Arc<Mutex<Connection>>,
}

/// The subset of a live row `upsert` needs to decide what to do.
struct LiveRow {
    id: String,
    claim: String,
    status: Option<String>,
    confidence: f64,
    metadata: String,
    episode_id: Option<String>,
    created_at: String,
}

impl FactStore {
    /// Wrap a connection.
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Append the keys already in use under this `(agent, scope)` to a
    /// claim-key rejection.
    ///
    /// ADR 0001 §2.3.3 names "consolidation *selects* a key from the existing
    /// key space" as the most-trusted defence against a dedup miss. A caller
    /// that is only told *no* will invent a second legal name for a slot that
    /// already exists — which is the exact failure the vocabulary is meant to
    /// prevent, arrived at by a different road. So the rejection carries the
    /// space.
    ///
    /// Best-effort by construction: if the lookup itself fails we return the
    /// original validation error untouched. The caller's key is wrong either
    /// way, and replacing a precise message with a storage error would lose
    /// the only part they can act on.
    fn with_existing_keys(
        conn: &Connection,
        write: &FactWrite,
        err: OpenFangError,
    ) -> OpenFangError {
        let lookup = (|| -> rusqlite::Result<Vec<String>> {
            let mut stmt = conn.prepare(
                "SELECT claim_key FROM memories
                 WHERE agent_id = ?1 AND scope = ?2 AND kind = ?3
                   AND deleted = 0 AND claim_key IS NOT NULL
                 ORDER BY claim_key
                 LIMIT ?4",
            )?;
            let rows = stmt.query_map(
                rusqlite::params![
                    write.agent_id.0.to_string(),
                    write.scope,
                    KIND_FACT,
                    MAX_SUGGESTED_KEYS,
                ],
                |r| r.get::<_, String>(0),
            )?;
            rows.collect()
        })();

        match lookup {
            Ok(keys) if !keys.is_empty() => {
                let more = if keys.len() as i64 == MAX_SUGGESTED_KEYS {
                    " (first 25)"
                } else {
                    ""
                };
                OpenFangError::InvalidInput(format!(
                    "{err} Keys already in use for scope {:?}{more}: {}. \
                     Prefer one of these if it fits the claim.",
                    write.scope,
                    keys.join(", ")
                ))
            }
            Ok(_) => OpenFangError::InvalidInput(format!(
                "{err} No facts exist yet under scope {:?}, so this would mint the \
                 first key in that space.",
                write.scope
            )),
            Err(_) => err,
        }
    }

    /// Write a claim into its slot, superseding whatever was there.
    ///
    /// Exactly one of three things happens, atomically:
    ///
    /// - slot empty            -> insert, [`FactOutcome::Created`]
    /// - same claim and status -> touch `last_affirmed_at`, [`FactOutcome::Affirmed`]
    /// - anything else         -> copy old row to `fact_history`, overwrite live
    ///   row in place, [`FactOutcome::Superseded`]
    ///
    /// The history copy and the overwrite share one transaction. That is the
    /// mechanical guarantee behind §2.3.1: there is no instant at which both
    /// claims are live (the unique index forbids it) and no crash window in
    /// which the old claim is gone but unrecorded.
    ///
    /// # Rejections
    ///
    /// Returns [`OpenFangError::InvalidInput`] — before opening the
    /// transaction, so nothing is written — when `scope` or `claim_key` is
    /// outside the controlled vocabulary, or when `scope` is `global` (ADR
    /// 0001 §5.2, unanswered). A claim-key rejection carries the keys already
    /// in use under that scope, because §2.3.3's first mitigation is that a
    /// caller *selects* from the existing key space rather than inventing a
    /// name, and it cannot select from a space it has not been shown.
    pub fn upsert(&self, write: FactWrite) -> OpenFangResult<FactOutcome> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Vocabulary first: a bad key must cost nothing and change nothing.
        let scope = FactScope::parse(&write.scope)?;
        if !scope.is_shipped() {
            return Err(OpenFangError::InvalidInput(format!(
                "fact scope {scope} is not available: ADR 0001 §5.2 leaves cross-agent \
                 global facts an open question (who may write one, and whether it renders \
                 into every agent's prompt). Use agent, project or user scope."
            )));
        }
        let claim_key = ClaimKey::parse(&write.claim_key)
            .map_err(|e| Self::with_existing_keys(&conn, &write, e))?;
        let claim_key = claim_key.as_str().to_string();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let now = Utc::now().to_rfc3339();
        let agent = write.agent_id.0.to_string();
        let meta_str = serde_json::to_string(&write.metadata)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let source_str = serde_json::to_string(&write.source)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let embedding_bytes: Option<Vec<u8>> = write
            .embedding
            .as_deref()
            .map(crate::semantic::embedding_to_bytes);

        let live = tx
            .query_row(
                "SELECT id, content, status, confidence, metadata, episode_id, created_at
                 FROM memories
                 WHERE agent_id = ?1 AND scope = ?2 AND claim_key = ?3
                   AND kind = ?4 AND deleted = 0",
                rusqlite::params![agent, scope.as_str(), claim_key, KIND_FACT],
                |r| {
                    Ok(LiveRow {
                        id: r.get(0)?,
                        claim: r.get(1)?,
                        status: r.get(2)?,
                        confidence: r.get(3)?,
                        metadata: r.get(4)?,
                        episode_id: r.get(5)?,
                        created_at: r.get(6)?,
                    })
                },
            )
            .optional()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let outcome = match live {
            // Slot empty: a fresh claim. Also the path taken when the slot
            // holds only soft-deleted rows — the v14 index scopes uniqueness
            // to `deleted = 0` precisely so `forget` cannot poison a key.
            None => {
                let id = MemoryId::new();
                tx.execute(
                    "INSERT INTO memories (id, agent_id, content, source, scope, confidence,
                                           metadata, created_at, accessed_at, access_count,
                                           deleted, embedding, episode_id, kind, claim_key,
                                           status, last_affirmed_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, 0, 0, ?9, ?10, ?11, ?12, ?13, ?8)",
                    rusqlite::params![
                        id.0.to_string(),
                        agent,
                        write.claim,
                        source_str,
                        scope.as_str(),
                        write.confidence,
                        meta_str,
                        now,
                        embedding_bytes,
                        write.episode_id,
                        KIND_FACT,
                        claim_key,
                        write.status.as_str(),
                    ],
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                FactOutcome::Created { id }
            }

            // Unchanged claim: an affirmation, not a new version. No history
            // row — writing one per re-derivation would bury the real changes.
            // Confidence, metadata and provenance still refresh: they describe
            // the *current* belief in an unchanged claim, and freezing them
            // would leave a caller no way to correct a confidence short of
            // faking a claim edit.
            Some(row)
                if row.claim == write.claim
                    && row.status.as_deref() == Some(write.status.as_str()) =>
            {
                let id = parse_memory_id(&row.id)?;
                tx.execute(
                    "UPDATE memories
                     SET last_affirmed_at = ?2, accessed_at = ?2, confidence = ?3,
                         metadata = ?4, episode_id = COALESCE(?5, episode_id)
                     WHERE id = ?1",
                    rusqlite::params![row.id, now, write.confidence, meta_str, write.episode_id,],
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                FactOutcome::Affirmed { id }
            }

            // Changed claim (or status): supersede.
            Some(row) => {
                let id = parse_memory_id(&row.id)?;
                let history_id = Uuid::new_v4();
                tx.execute(
                    "INSERT INTO fact_history (id, memory_id, agent_id, scope, claim_key, claim,
                                               status, confidence, metadata, episode_id,
                                               created_at, superseded_at, superseded_by_episode)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    rusqlite::params![
                        history_id.to_string(),
                        row.id,
                        agent,
                        scope.as_str(),
                        claim_key,
                        row.claim,
                        row.status,
                        row.confidence,
                        row.metadata,
                        row.episode_id,
                        row.created_at,
                        now,
                        write.episode_id,
                    ],
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;

                // Overwrite in place: same id, new claim.
                //
                // `embedding` is assigned unconditionally, including to NULL.
                // Leaving the previous vector attached to replaced text would
                // be the §2.3.1 failure by the back door — the row a semantic
                // search matches would be the old claim, and the text it
                // returns would be the new one.
                //
                // `created_at` is re-stamped so it means "when this claim
                // became live"; the outgoing value went with the history row
                // above, which is what makes that row a closed interval.
                tx.execute(
                    "UPDATE memories
                     SET content = ?2, source = ?3, confidence = ?4, metadata = ?5,
                         embedding = ?6, episode_id = ?7, status = ?8,
                         created_at = ?9, accessed_at = ?9, last_affirmed_at = ?9
                     WHERE id = ?1",
                    rusqlite::params![
                        row.id,
                        write.claim,
                        source_str,
                        write.confidence,
                        meta_str,
                        embedding_bytes,
                        write.episode_id,
                        write.status.as_str(),
                        now,
                    ],
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                FactOutcome::Superseded { id, history_id }
            }
        };

        tx.commit()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(outcome)
    }

    /// The live claim in a slot, if any. Exact-key lookup, no ranking.
    pub fn get(
        &self,
        agent_id: AgentId,
        scope: &str,
        claim_key: &str,
    ) -> OpenFangResult<Option<Fact>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.query_row(
            "SELECT id, agent_id, scope, claim_key, content, status, confidence, episode_id,
                    created_at, last_affirmed_at, metadata
             FROM memories
             WHERE agent_id = ?1 AND scope = ?2 AND claim_key = ?3
               AND kind = ?4 AND deleted = 0",
            rusqlite::params![agent_id.0.to_string(), scope, claim_key, KIND_FACT],
            row_to_fact,
        )
        .optional()
        .map_err(|e| OpenFangError::Memory(e.to_string()))?
        .transpose()
    }

    /// Every claim that has occupied a slot, newest supersession first.
    ///
    /// This is the audit path and the only reader of `fact_history`. It is
    /// never part of automatic recall (§2.3.2) — reaching it takes an explicit
    /// call, which is the only time the history is wanted.
    pub fn history(
        &self,
        agent_id: AgentId,
        scope: &str,
        claim_key: &str,
        limit: usize,
    ) -> OpenFangResult<Vec<FactHistoryEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, memory_id, scope, claim_key, claim, status, confidence, episode_id,
                        created_at, superseded_at, superseded_by_episode
                 FROM fact_history
                 WHERE agent_id = ?1 AND scope = ?2 AND claim_key = ?3
                 ORDER BY superseded_at DESC
                 LIMIT ?4",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map(
                rusqlite::params![agent_id.0.to_string(), scope, claim_key, limit as i64],
                row_to_history,
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))??);
        }
        Ok(out)
    }
}

/// Parse a stored uuid text column into a `MemoryId`.
fn parse_memory_id(s: &str) -> OpenFangResult<MemoryId> {
    Uuid::parse_str(s)
        .map(MemoryId)
        .map_err(|e| OpenFangError::Memory(format!("malformed memory id {s:?}: {e}")))
}

/// Row mapper for a live fact.
///
/// Returns a nested `Result` because the outer one belongs to rusqlite and the
/// inner to our own parsing (status vocabulary, uuid, JSON). Collapsing them
/// would mean reporting a bad status as a SQL error, which sends whoever reads
/// the log looking in the wrong place.
#[allow(clippy::type_complexity)]
fn row_to_fact(row: &rusqlite::Row<'_>) -> rusqlite::Result<OpenFangResult<Fact>> {
    let id: String = row.get(0)?;
    let status: Option<String> = row.get(5)?;
    let metadata: String = row.get(10)?;
    Ok((|| {
        Ok(Fact {
            id: parse_memory_id(&id)?,
            agent_id: row.get::<_, String>(1).unwrap_or_default(),
            scope: row.get::<_, String>(2).unwrap_or_default(),
            claim_key: row.get::<_, String>(3).unwrap_or_default(),
            claim: row.get::<_, String>(4).unwrap_or_default(),
            status: match status.as_deref() {
                Some(s) => FactStatus::parse(s)?,
                None => FactStatus::Settled,
            },
            confidence: row.get::<_, f64>(6).unwrap_or(1.0),
            episode_id: row.get::<_, Option<String>>(7).unwrap_or_default(),
            created_at: row.get::<_, String>(8).unwrap_or_default(),
            last_affirmed_at: row.get::<_, Option<String>>(9).unwrap_or_default(),
            metadata: serde_json::from_str(&metadata).unwrap_or_default(),
        })
    })())
}

/// Row mapper for a superseded claim.
#[allow(clippy::type_complexity)]
fn row_to_history(row: &rusqlite::Row<'_>) -> rusqlite::Result<OpenFangResult<FactHistoryEntry>> {
    let id: String = row.get(0)?;
    let memory_id: String = row.get(1)?;
    let status: Option<String> = row.get(5)?;
    Ok((|| {
        Ok(FactHistoryEntry {
            id: Uuid::parse_str(&id)
                .map_err(|e| OpenFangError::Memory(format!("malformed history id {id:?}: {e}")))?,
            memory_id: parse_memory_id(&memory_id)?,
            scope: row.get::<_, String>(2).unwrap_or_default(),
            claim_key: row.get::<_, String>(3).unwrap_or_default(),
            claim: row.get::<_, String>(4).unwrap_or_default(),
            status: match status.as_deref() {
                Some(s) => Some(FactStatus::parse(s)?),
                None => None,
            },
            confidence: row.get::<_, f64>(6).unwrap_or(1.0),
            episode_id: row.get::<_, Option<String>>(7).unwrap_or_default(),
            created_at: row.get::<_, String>(8).unwrap_or_default(),
            superseded_at: row.get::<_, String>(9).unwrap_or_default(),
            superseded_by_episode: row.get::<_, Option<String>>(10).unwrap_or_default(),
        })
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;

    fn store() -> (FactStore, Arc<Mutex<Connection>>) {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        let conn = Arc::new(Mutex::new(conn));
        (FactStore::new(conn.clone()), conn)
    }

    fn agent() -> AgentId {
        AgentId::new()
    }

    fn write(a: AgentId, key: &str, claim: &str) -> FactWrite {
        FactWrite::new(a, "agent", key, claim)
    }

    #[test]
    fn upsert_creates_an_empty_slot() {
        let (facts, _c) = store();
        let a = agent();

        let out = facts
            .upsert(write(a, "git.trunk_model", "main is trunk"))
            .unwrap();
        assert!(matches!(out, FactOutcome::Created { .. }));

        let fact = facts.get(a, "agent", "git.trunk_model").unwrap().unwrap();
        assert_eq!(fact.claim, "main is trunk");
        assert_eq!(fact.status, FactStatus::Settled);
        assert_eq!(fact.id, out.id());
        assert!(
            fact.last_affirmed_at.is_some(),
            "a fresh claim counts as affirmed at write time"
        );
    }

    /// The slot id survives supersession, and the outgoing claim lands in
    /// history with the episode that replaced it.
    #[test]
    fn upsert_supersedes_in_place_and_keeps_the_slot_id() {
        let (facts, _c) = store();
        let a = agent();

        let first = facts
            .upsert(write(a, "git.trunk_model", "local-main is trunk").with_episode("ep-1"))
            .unwrap();
        let second = facts
            .upsert(write(a, "git.trunk_model", "main is trunk").with_episode("ep-2"))
            .unwrap();

        assert_eq!(
            first.id(),
            second.id(),
            "supersession updates in place; the id names the slot, not the version"
        );
        assert!(second.superseded());

        let live = facts.get(a, "agent", "git.trunk_model").unwrap().unwrap();
        assert_eq!(live.claim, "main is trunk");
        assert_eq!(live.episode_id.as_deref(), Some("ep-2"));

        let hist = facts.history(a, "agent", "git.trunk_model", 10).unwrap();
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].claim, "local-main is trunk");
        assert_eq!(hist[0].episode_id.as_deref(), Some("ep-1"));
        assert_eq!(hist[0].superseded_by_episode.as_deref(), Some("ep-2"));
        assert_eq!(hist[0].memory_id, first.id());
    }

    /// §2.3.1, the property the whole tier exists for: after any number of
    /// writes the reader can only ever see one claim per slot.
    #[test]
    fn a_slot_never_holds_more_than_one_live_claim() {
        let (facts, conn) = store();
        let a = agent();

        for claim in ["v1", "v2", "v3"] {
            facts.upsert(write(a, "git.trunk_model", claim)).unwrap();
        }

        let live: i64 = conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE kind = 'fact' AND deleted = 0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(live, 1, "superseded claims must not remain in `memories`");

        let hist = facts.history(a, "agent", "git.trunk_model", 10).unwrap();
        assert_eq!(hist.len(), 2, "two supersessions, two history rows");
        assert_eq!(
            hist.iter().map(|h| h.claim.as_str()).collect::<Vec<_>>(),
            vec!["v2", "v1"],
            "history is newest-superseded first"
        );
    }

    /// Re-deriving the same claim must not spam the audit trail.
    #[test]
    fn identical_reassertion_affirms_without_writing_history() {
        let (facts, _c) = store();
        let a = agent();

        facts
            .upsert(write(a, "git.trunk_model", "main is trunk").with_confidence(0.5))
            .unwrap();
        let out = facts
            .upsert(write(a, "git.trunk_model", "main is trunk").with_confidence(0.9))
            .unwrap();

        assert!(matches!(out, FactOutcome::Affirmed { .. }));
        assert!(
            facts
                .history(a, "agent", "git.trunk_model", 10)
                .unwrap()
                .is_empty(),
            "nothing was superseded, so nothing belongs in history"
        );

        let live = facts.get(a, "agent", "git.trunk_model").unwrap().unwrap();
        assert_eq!(
            live.confidence, 0.9,
            "an affirmation refreshes current-state attributes"
        );
    }

    /// Closing an open loop is a real change even when the wording is
    /// identical, and the audit trail should be able to say when it happened.
    #[test]
    fn a_status_change_alone_is_a_supersession() {
        let (facts, _c) = store();
        let a = agent();

        facts
            .upsert(
                write(a, "memory.sweep_status", "sweep in flight").with_status(FactStatus::Open),
            )
            .unwrap();
        let out = facts
            .upsert(
                write(a, "memory.sweep_status", "sweep in flight").with_status(FactStatus::Settled),
            )
            .unwrap();

        assert!(out.superseded());
        let hist = facts
            .history(a, "agent", "memory.sweep_status", 10)
            .unwrap();
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].status, Some(FactStatus::Open));
        assert_eq!(
            facts
                .get(a, "agent", "memory.sweep_status")
                .unwrap()
                .unwrap()
                .status,
            FactStatus::Settled
        );
    }

    /// §2.3.1 by the back door: a stale vector on replaced text would make
    /// semantic search match the old claim and return the new one.
    #[test]
    fn supersession_replaces_the_embedding_even_with_none() {
        let (facts, conn) = store();
        let a = agent();

        facts
            .upsert(write(a, "git.trunk_model", "old").with_embedding(vec![1.0, 2.0, 3.0]))
            .unwrap();
        let embedded: Option<Vec<u8>> = conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT embedding FROM memories WHERE kind = 'fact' AND deleted = 0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(embedded.is_some(), "precondition: the first claim had one");

        facts.upsert(write(a, "git.trunk_model", "new")).unwrap();
        let after: Option<Vec<u8>> = conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT embedding FROM memories WHERE kind = 'fact' AND deleted = 0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            after.is_none(),
            "the old vector must not survive onto replaced text"
        );
    }

    /// Scope, agent and key are all part of the slot identity.
    #[test]
    fn distinct_slots_do_not_collide() {
        let (facts, _c) = store();
        let a = agent();
        let b = agent();

        facts
            .upsert(write(a, "git.trunk_model", "agent scope"))
            .unwrap();
        facts
            .upsert(FactWrite::new(
                a,
                "project",
                "git.trunk_model",
                "project scope",
            ))
            .unwrap();
        facts
            .upsert(write(b, "git.trunk_model", "other agent"))
            .unwrap();
        facts
            .upsert(write(a, "memory.sweep_status", "other key"))
            .unwrap();

        assert_eq!(
            facts
                .get(a, "agent", "git.trunk_model")
                .unwrap()
                .unwrap()
                .claim,
            "agent scope"
        );
        assert_eq!(
            facts
                .get(a, "project", "git.trunk_model")
                .unwrap()
                .unwrap()
                .claim,
            "project scope"
        );
        assert_eq!(
            facts
                .get(b, "agent", "git.trunk_model")
                .unwrap()
                .unwrap()
                .claim,
            "other agent"
        );
    }

    /// `forget` must not poison a claim key forever — the v14 index is partial
    /// on `deleted = 0` for exactly this, and the store has to agree with it.
    #[test]
    fn a_soft_deleted_slot_can_be_written_again() {
        let (facts, conn) = store();
        let a = agent();

        let first = facts.upsert(write(a, "git.trunk_model", "old")).unwrap();
        conn.lock()
            .unwrap()
            .execute(
                "UPDATE memories SET deleted = 1 WHERE id = ?1",
                rusqlite::params![first.id().0.to_string()],
            )
            .unwrap();

        let second = facts.upsert(write(a, "git.trunk_model", "new")).unwrap();
        assert!(
            matches!(second, FactOutcome::Created { .. }),
            "a released slot is empty, not occupied by a tombstone"
        );
        assert_ne!(first.id(), second.id());
        assert_eq!(
            facts
                .get(a, "agent", "git.trunk_model")
                .unwrap()
                .unwrap()
                .claim,
            "new"
        );
    }

    /// The recall path filters on `kind`; a fact that does not carry it would
    /// be invisible to tier 3 and visible to everything else.
    #[test]
    fn facts_are_written_with_the_fact_kind() {
        let (facts, conn) = store();
        let a = agent();
        facts
            .upsert(write(a, "git.trunk_model", "main is trunk"))
            .unwrap();

        let kind: String = conn
            .lock()
            .unwrap()
            .query_row("SELECT kind FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(kind, "fact");
    }

    /// A rejected key must cost nothing. Not a partial row, not a bumped
    /// timestamp, not a history entry — the transaction is never opened.
    #[test]
    fn a_rejected_claim_key_writes_nothing() {
        let (facts, conn) = store();
        let a = agent();

        let err = facts
            .upsert(write(a, "anai-204-progress", "step 4 in flight"))
            .unwrap_err();
        assert!(
            matches!(err, OpenFangError::InvalidInput(_)),
            "vocabulary rejection must be InvalidInput, got {err:?}"
        );

        let rows: i64 = conn
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        let hist: i64 = conn
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM fact_history", [], |r| r.get(0))
            .unwrap();
        assert_eq!((rows, hist), (0, 0));
    }

    /// The rejection has to show the caller the space it should have selected
    /// from (ADR 0001 2.3.3 mitigation 1) — otherwise the next attempt is
    /// another invented name.
    #[test]
    fn a_rejection_lists_the_keys_already_in_use() {
        let (facts, _c) = store();
        let a = agent();
        facts
            .upsert(write(a, "git.trunk_model", "main is trunk"))
            .unwrap();
        facts
            .upsert(write(a, "memory.sweep_status", "clean"))
            .unwrap();

        let err = facts
            .upsert(write(a, "2026-08-21-trunk-note", "..."))
            .unwrap_err()
            .to_string();
        assert!(err.contains("git.trunk_model"), "{err}");
        assert!(err.contains("memory.sweep_status"), "{err}");
    }

    /// Empty key space is its own message: "there is nothing to select from"
    /// is different advice than "pick one of these".
    #[test]
    fn a_rejection_says_so_when_the_key_space_is_empty() {
        let (facts, _c) = store();
        let err = facts
            .upsert(write(agent(), "nope", "..."))
            .unwrap_err()
            .to_string();
        assert!(err.contains("No facts exist yet"), "{err}");
    }

    /// ADR 0001 5.2 is unanswered, so the writer refuses rather than letting
    /// a cross-agent fact ship because nobody said no.
    #[test]
    fn global_scope_is_refused_at_the_writer() {
        let (facts, conn) = store();
        let err = facts
            .upsert(FactWrite::new(
                agent(),
                "global",
                "git.trunk_model",
                "true for everyone",
            ))
            .unwrap_err()
            .to_string();
        assert!(err.contains("5.2"), "{err}");

        let rows: i64 = conn
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }

    /// Scope is half of the slot key, so a free-text scope would silently open
    /// a second slot for the same claim — the 2.3.3 dedup miss by another
    /// route. `episodic` is the scope every pre-v14 row carries, which makes
    /// it the most likely typo to arrive here.
    #[test]
    fn an_unknown_scope_is_refused() {
        let (facts, _c) = store();
        let err = facts
            .upsert(FactWrite::new(
                agent(),
                "episodic",
                "git.trunk_model",
                "main is trunk",
            ))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid fact scope"), "{err}");
    }

    #[test]
    fn status_round_trips_and_rejects_junk() {
        assert_eq!(FactStatus::parse("open").unwrap(), FactStatus::Open);
        assert_eq!(FactStatus::parse("settled").unwrap(), FactStatus::Settled);
        assert_eq!(FactStatus::Open.as_str(), "open");
        assert!(
            FactStatus::parse("superseded").is_err(),
            "there is no superseded status by design (ADR 0001 2.3.2)"
        );
    }
}
