//! MemorySubstrate: unified implementation of the `Memory` trait.
//!
//! Composes the structured store, semantic store, knowledge store,
//! session store, and consolidation engine behind a single async API.

use crate::consolidation::ConsolidationEngine;
use crate::episode::{CloseReason, Episode, EpisodeStatus, EpisodeStore};
use crate::fact::{FactOutcome, FactStore, FactWrite};
use crate::knowledge::KnowledgeStore;
use crate::migration::run_migrations;
use crate::semantic::SemanticStore;
use crate::session::{Session, SessionStore};
use crate::structured::StructuredStore;
use crate::usage::UsageStore;

use async_trait::async_trait;
use base64::Engine;
use openfang_types::agent::{AgentEntry, AgentId, SessionId};
use openfang_types::config::MemoryConfig;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::memory::{
    ConsolidationReport, Entity, ExportFormat, GraphMatch, GraphPattern, ImportReport, Memory,
    MemoryFilter, MemoryFragment, MemoryId, MemorySource, Relation,
};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// One wake row failed closed by [`MemorySubstrate::reap_in_flight_wakes`].
///
/// ## Why the payload rides along (ANAI-217)
///
/// The reaper's original contract was purely *unwedging*: free the per-caller
/// in-flight slot and preserve an audit trail. But a reaped wake is a
/// correlation the initiator is still owed a reply on, and the reply-right that
/// records that debt lives in kernel memory — so a daemon restart destroys the
/// debt record while the *sender's* expectation survives in its own transcript.
/// The result was the one hole left in the ANAI-196 guarantee: process-scoped,
/// not durable. Paying it requires the sender and the surfacing route, which
/// exist only inside the envelope, which exists only in the row's payload.
/// Hence: the reaper returns the payload and the kernel discharges the debt.
///
/// `payload` is the raw envelope BLOB, EMPTY when the column was null or
/// unreadable. Empty means "this row's debt cannot be paid" — the row
/// is still reaped (it holds a slot regardless), and the caller reports the
/// unpayable debt rather than dropping it silently.
#[derive(Debug, Clone)]
pub struct ReapedWake {
    /// Task-queue row id — the correlation id the sender is waiting on.
    pub task_id: String,
    /// The caller whose per-caller slot this row was holding.
    pub created_by: String,
    /// Decoded `WakeEnvelope` payload; empty if absent or undecodable.
    pub payload: Vec<u8>,
    /// True when the row was reaped for blowing its OWN stated deadline plus
    /// grace, rather than for sitting past the operator's flat stale cutoff.
    /// Changes only the wording of the diagnosis, but the two are genuinely
    /// different findings and the reply body should not conflate them.
    pub past_deadline: bool,
}

/// The unified memory substrate. Implements the `Memory` trait by delegating
/// to specialized stores backed by a shared SQLite connection.
pub struct MemorySubstrate {
    conn: Arc<Mutex<Connection>>,
    structured: StructuredStore,
    semantic: SemanticStore,
    knowledge: KnowledgeStore,
    sessions: SessionStore,
    consolidation: ConsolidationEngine,
    usage: UsageStore,
    episodes: EpisodeStore,
    facts: FactStore,
}

impl MemorySubstrate {
    /// Open or create a memory substrate at the given database path.
    ///
    /// When `memory_config.backend == "http"` and `http_url`/`http_token_env` are set,
    /// the semantic store routes `remember`/`recall` to the memory-api gateway.
    /// All other stores (KV, knowledge graph, sessions) remain local SQLite.
    pub fn open(
        db_path: &Path,
        decay_rate: f32,
        memory_config: &MemoryConfig,
    ) -> OpenFangResult<Self> {
        let conn = Connection::open(db_path).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        run_migrations(&conn).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let shared = Arc::new(Mutex::new(conn));

        let semantic = Self::create_semantic_store(Arc::clone(&shared), memory_config);

        Ok(Self {
            conn: Arc::clone(&shared),
            structured: StructuredStore::new(Arc::clone(&shared)),
            semantic,
            knowledge: KnowledgeStore::new(Arc::clone(&shared)),
            sessions: SessionStore::new(Arc::clone(&shared)),
            usage: UsageStore::new(Arc::clone(&shared)),
            episodes: EpisodeStore::with_idle_timeout(
                Arc::clone(&shared),
                memory_config.episode_idle_timeout_minutes,
            ),
            facts: FactStore::new(Arc::clone(&shared)),
            consolidation: ConsolidationEngine::new(shared, decay_rate),
        })
    }

    /// Create the semantic store, optionally with HTTP backend.
    fn create_semantic_store(
        conn: Arc<Mutex<Connection>>,
        memory_config: &MemoryConfig,
    ) -> SemanticStore {
        #[cfg(feature = "http-memory")]
        if memory_config.backend == "http" {
            if let (Some(url), Some(token_env)) =
                (&memory_config.http_url, &memory_config.http_token_env)
            {
                match crate::http_client::MemoryApiClient::new(url, token_env) {
                    Ok(client) => {
                        // Best-effort health check on startup
                        match client.health_check() {
                            Ok(()) => info!(url = %url, "HTTP memory backend connected"),
                            Err(e) => {
                                warn!(url = %url, error = %e, "HTTP memory backend health check failed, will retry on use")
                            }
                        }
                        return SemanticStore::new_with_http(conn, client);
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to create HTTP memory client, falling back to SQLite");
                    }
                }
            } else {
                warn!("backend=http but http_url/http_token_env not set, falling back to SQLite");
            }
        }

        #[cfg(not(feature = "http-memory"))]
        let _ = memory_config;

        SemanticStore::new(conn)
    }

    /// Create an in-memory substrate (for testing). Always uses SQLite backend.
    pub fn open_in_memory(decay_rate: f32) -> OpenFangResult<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| OpenFangError::Memory(e.to_string()))?;
        run_migrations(&conn).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let shared = Arc::new(Mutex::new(conn));

        Ok(Self {
            conn: Arc::clone(&shared),
            structured: StructuredStore::new(Arc::clone(&shared)),
            semantic: SemanticStore::new(Arc::clone(&shared)),
            knowledge: KnowledgeStore::new(Arc::clone(&shared)),
            sessions: SessionStore::new(Arc::clone(&shared)),
            usage: UsageStore::new(Arc::clone(&shared)),
            episodes: EpisodeStore::new(Arc::clone(&shared)),
            facts: FactStore::new(Arc::clone(&shared)),
            consolidation: ConsolidationEngine::new(shared, decay_rate),
        })
    }

    /// Get a reference to the usage store.
    pub fn usage(&self) -> &UsageStore {
        &self.usage
    }

    /// In-memory substrate with the episode idle timer armed (tests only).
    ///
    /// [`Self::open_in_memory`] takes the default timeout of 0, which is the
    /// right default but makes every timer path untestable through the
    /// substrate's own API.
    #[cfg(test)]
    pub(crate) fn open_in_memory_with_idle_timeout(
        decay_rate: f32,
        idle_timeout_minutes: i64,
    ) -> OpenFangResult<Self> {
        let mut substrate = Self::open_in_memory(decay_rate)?;
        substrate.episodes =
            EpisodeStore::with_idle_timeout(Arc::clone(&substrate.conn), idle_timeout_minutes);
        Ok(substrate)
    }

    // -----------------------------------------------------------------
    // Episodes (ADR 0001 §2.2)
    // -----------------------------------------------------------------

    /// Get a reference to the episode store.
    pub fn episodes(&self) -> &EpisodeStore {
        &self.episodes
    }

    /// Resolve the episode this turn belongs to, opening one and closing a
    /// timed-out predecessor as needed. See [`EpisodeStore::ensure_open`].
    pub fn ensure_open_episode(&self, agent_id: AgentId) -> OpenFangResult<uuid::Uuid> {
        self.episodes.ensure_open(agent_id)
    }

    /// Async wrapper for [`Self::ensure_open_episode`] — the capture path runs
    /// inside the agent loop's async context and must not block the reactor on
    /// a SQLite write.
    pub async fn ensure_open_episode_async(&self, agent_id: AgentId) -> OpenFangResult<uuid::Uuid> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.ensure_open(agent_id))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Close the agent's open episode. `None` when nothing was open.
    pub fn close_episode(
        &self,
        agent_id: AgentId,
        reason: CloseReason,
        title: Option<&str>,
        summary: Option<&str>,
    ) -> OpenFangResult<Option<uuid::Uuid>> {
        self.episodes
            .close_current(agent_id, reason, title, summary)
    }

    /// Async wrapper for [`Self::close_episode`]. The tool path runs inside the
    /// agent loop's async context, same reason as `ensure_open_episode_async`.
    pub async fn close_episode_async(
        &self,
        agent_id: AgentId,
        reason: CloseReason,
        title: Option<String>,
        summary: Option<String>,
    ) -> OpenFangResult<Option<uuid::Uuid>> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || {
            store.close_current(agent_id, reason, title.as_deref(), summary.as_deref())
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// The agent's memory status. See [`EpisodeStore::status`].
    pub fn episode_status(
        &self,
        agent_id: AgentId,
        recent_limit: usize,
    ) -> OpenFangResult<EpisodeStatus> {
        self.episodes.status(agent_id, recent_limit)
    }

    /// Async wrapper for [`Self::episode_status`].
    pub async fn episode_status_async(
        &self,
        agent_id: AgentId,
        recent_limit: usize,
    ) -> OpenFangResult<EpisodeStatus> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.status(agent_id, recent_limit))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// The agent's currently open episode, if any.
    pub fn current_episode(&self, agent_id: AgentId) -> OpenFangResult<Option<Episode>> {
        self.episodes.current(agent_id)
    }

    /// Async wrapper for [`EpisodeStore::sweep_idle`] — the fleet-wide reaper
    /// for open episodes past the idle gap. Runs from the kernel's sweep task
    /// (ANAI-219), which is an async context, so the SQLite `UPDATE` goes on
    /// the blocking pool like every other write path here.
    pub async fn sweep_idle_episodes_async(&self) -> OpenFangResult<usize> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.sweep_idle())
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Closed, unsummarised episodes inside the lookback window, plus the
    /// unclipped pending count. See [`EpisodeStore::awaiting_summary`] — the
    /// window is what keeps the live summariser from being a backfill.
    pub async fn episodes_awaiting_summary_async(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> OpenFangResult<(Vec<Episode>, usize)> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.awaiting_summary(cutoff, limit))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// The episode's linked rows, oldest first, capped.
    pub async fn episode_material_async(
        &self,
        id: uuid::Uuid,
        max_rows: usize,
    ) -> OpenFangResult<Vec<String>> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.material(id, max_rows))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Has a derived summary row already been written for this episode?
    pub async fn episode_has_summary_row_async(&self, id: uuid::Uuid) -> OpenFangResult<bool> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.has_summary_row(id))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Write a derived title/summary onto a closed episode. `false` means
    /// something else got there first — see [`EpisodeStore::set_summary`].
    pub async fn set_episode_summary_async(
        &self,
        id: uuid::Uuid,
        title: Option<String>,
        summary: String,
    ) -> OpenFangResult<bool> {
        let store = self.episodes.clone();
        tokio::task::spawn_blocking(move || store.set_summary(id, title.as_deref(), &summary))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    // -----------------------------------------------------------------
    // Tier-3 facts (ADR 0001 §2.3)
    // -----------------------------------------------------------------

    /// The tier-3 fact store: keyed claim slots and their supersession
    /// history.
    ///
    /// Handed out directly for reads, which are exact-key lookups cheap enough
    /// to run inline. Writes go through [`Self::fact_upsert_async`] instead —
    /// an upsert opens a transaction, and blocking the reactor on one from
    /// inside the agent loop is exactly what the other `*_async` wrappers here
    /// exist to avoid.
    pub fn facts(&self) -> &FactStore {
        &self.facts
    }

    /// Write a claim into its slot, off-reactor. See [`FactStore::upsert`].
    pub async fn fact_upsert_async(&self, write: FactWrite) -> OpenFangResult<FactOutcome> {
        let store = self.facts.clone();
        tokio::task::spawn_blocking(move || store.upsert(write))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Get the shared database connection (for constructing stores from outside).
    pub fn usage_conn(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }

    /// Save an agent entry to persistent storage.
    pub fn save_agent(&self, entry: &AgentEntry) -> OpenFangResult<()> {
        self.structured.save_agent(entry)
    }

    /// Load an agent entry from persistent storage.
    pub fn load_agent(&self, agent_id: AgentId) -> OpenFangResult<Option<AgentEntry>> {
        self.structured.load_agent(agent_id)
    }

    /// Remove an agent from persistent storage and cascade-delete sessions.
    pub fn remove_agent(&self, agent_id: AgentId) -> OpenFangResult<()> {
        // Delete associated sessions first
        let _ = self.sessions.delete_agent_sessions(agent_id);
        self.structured.remove_agent(agent_id)
    }

    /// Load all agent entries from persistent storage.
    pub fn load_all_agents(&self) -> OpenFangResult<Vec<AgentEntry>> {
        self.structured.load_all_agents()
    }

    /// List all saved agents.
    pub fn list_agents(&self) -> OpenFangResult<Vec<(String, String, String)>> {
        self.structured.list_agents()
    }

    /// Synchronous get from the structured store (for kernel handle use).
    pub fn structured_get(
        &self,
        agent_id: AgentId,
        key: &str,
    ) -> OpenFangResult<Option<serde_json::Value>> {
        self.structured.get(agent_id, key)
    }

    /// List all KV pairs for an agent.
    pub fn list_kv(&self, agent_id: AgentId) -> OpenFangResult<Vec<(String, serde_json::Value)>> {
        self.structured.list_kv(agent_id)
    }

    /// List an agent's KV pairs ranked by write recency, for the MEMORY.md
    /// managed-block sweep (ANAI-168).
    pub fn list_kv_ranked(
        &self,
        agent_id: AgentId,
        limit: usize,
    ) -> OpenFangResult<Vec<crate::memory_md::KvFact>> {
        self.structured.list_kv_ranked(agent_id, limit)
    }

    /// Delete a KV entry for an agent.
    pub fn structured_delete(&self, agent_id: AgentId, key: &str) -> OpenFangResult<()> {
        self.structured.delete(agent_id, key)
    }

    /// Synchronous set in the structured store (for kernel handle use).
    pub fn structured_set(
        &self,
        agent_id: AgentId,
        key: &str,
        value: serde_json::Value,
    ) -> OpenFangResult<()> {
        self.structured.set(agent_id, key, value)
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: SessionId) -> OpenFangResult<Option<Session>> {
        self.sessions.get_session(session_id)
    }

    /// Last-agent-activity stamp for a session (RFC3339), or `None` if the
    /// session is missing. Turn-context envelope (ANAI-128):
    /// `now - updated_at = since_agent_msg`.
    pub fn session_updated_at(&self, session_id: SessionId) -> OpenFangResult<Option<String>> {
        self.sessions.session_updated_at(session_id)
    }

    /// Record a genuine human inbound (trigger == User) from `speaker_id`,
    /// stamping presence at `now` (RFC3339). Returns the actor's PRIOR stamp —
    /// the anchor for `since_this_speaker`. See
    /// [`crate::session::SessionStore::record_participant`].
    pub fn record_participant(
        &self,
        session_id: SessionId,
        speaker_id: &str,
        display_name: &str,
        now: &str,
    ) -> OpenFangResult<Option<String>> {
        self.sessions
            .record_participant(session_id, speaker_id, display_name, now)
    }

    /// Session participants, most-recent-first, capped at `limit`. Substrate
    /// for the turn-context roster line (ANAI-128).
    pub fn session_roster(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> OpenFangResult<Vec<crate::session::Participant>> {
        self.sessions.session_roster(session_id, limit)
    }

    /// Resolve a speaker snowflake to its AUTHORITATIVE name (rung 1 of the
    /// identity hierarchy, ANAI-127). `None` => no operator binding; caller
    /// falls back to `global_name` then the raw handle. See
    /// [`crate::session::SessionStore::resolve_identity`].
    pub fn resolve_identity(&self, speaker_id: &str) -> OpenFangResult<Option<String>> {
        self.sessions.resolve_identity(speaker_id)
    }

    /// Create or update the authoritative snowflake -> name binding. See
    /// [`crate::session::SessionStore::upsert_identity_binding`].
    pub fn upsert_identity_binding(
        &self,
        speaker_id: &str,
        openfang_name: &str,
        note: Option<&str>,
    ) -> OpenFangResult<()> {
        self.sessions
            .upsert_identity_binding(speaker_id, openfang_name, note)
    }

    /// Save a session.
    pub fn save_session(&self, session: &Session) -> OpenFangResult<()> {
        self.sessions.save_session(session)
    }

    /// Save a session asynchronously — runs the SQLite write in a blocking
    /// thread so the tokio runtime stays responsive.
    pub async fn save_session_async(&self, session: &Session) -> OpenFangResult<()> {
        let sessions = self.sessions.clone();
        let session = session.clone();
        tokio::task::spawn_blocking(move || sessions.save_session(&session))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Create a new empty session for an agent.
    pub fn create_session(&self, agent_id: AgentId) -> OpenFangResult<Session> {
        self.sessions.create_session(agent_id)
    }

    /// List all sessions with metadata.
    pub fn list_sessions(&self) -> OpenFangResult<Vec<serde_json::Value>> {
        self.sessions.list_sessions()
    }

    /// Delete a session by ID.
    pub fn delete_session(&self, session_id: SessionId) -> OpenFangResult<()> {
        self.sessions.delete_session(session_id)
    }

    /// Delete all sessions belonging to an agent.
    pub fn delete_agent_sessions(&self, agent_id: AgentId) -> OpenFangResult<()> {
        self.sessions.delete_agent_sessions(agent_id)
    }

    /// Delete the canonical (cross-channel) session for an agent.
    pub fn delete_canonical_session(&self, agent_id: AgentId) -> OpenFangResult<()> {
        self.sessions.delete_canonical_session(agent_id)
    }

    /// ANAI-246: re-anchor the canonical session at an episode boundary —
    /// drop the pre-boundary verbatim messages, keep the compacted summary.
    /// Returns the number of messages dropped.
    pub fn reanchor_canonical(&self, agent_id: AgentId) -> OpenFangResult<usize> {
        self.sessions.reanchor_canonical(agent_id)
    }

    /// Set or clear a session label.
    pub fn set_session_label(
        &self,
        session_id: SessionId,
        label: Option<&str>,
    ) -> OpenFangResult<()> {
        self.sessions.set_session_label(session_id, label)
    }

    /// Find a session by label for a given agent.
    pub fn find_session_by_label(
        &self,
        agent_id: AgentId,
        label: &str,
    ) -> OpenFangResult<Option<Session>> {
        self.sessions.find_session_by_label(agent_id, label)
    }

    /// List all sessions for a specific agent.
    pub fn list_agent_sessions(&self, agent_id: AgentId) -> OpenFangResult<Vec<serde_json::Value>> {
        self.sessions.list_agent_sessions(agent_id)
    }

    /// Create a new session with an optional label.
    pub fn create_session_with_label(
        &self,
        agent_id: AgentId,
        label: Option<&str>,
    ) -> OpenFangResult<Session> {
        self.sessions.create_session_with_label(agent_id, label)
    }

    /// Load canonical session context for cross-channel memory.
    ///
    /// Returns the compacted summary (if any) and recent messages from the
    /// agent's persistent canonical session.
    pub fn canonical_context(
        &self,
        agent_id: AgentId,
        window_size: Option<usize>,
    ) -> OpenFangResult<(Option<String>, Vec<openfang_types::message::Message>)> {
        self.sessions.canonical_context(agent_id, window_size)
    }

    /// Store an LLM-generated summary, replacing older messages with the kept subset.
    ///
    /// Used by the compactor to replace text-truncation compaction with an
    /// LLM-generated summary of older conversation history.
    pub fn store_llm_summary(
        &self,
        agent_id: AgentId,
        summary: &str,
        kept_messages: Vec<openfang_types::message::Message>,
    ) -> OpenFangResult<()> {
        self.sessions
            .store_llm_summary(agent_id, summary, kept_messages)
    }

    /// Write a human-readable JSONL mirror of a session to disk.
    ///
    /// Best-effort — errors are returned but should be logged,
    /// never affecting the primary SQLite store.
    pub fn write_jsonl_mirror(
        &self,
        session: &Session,
        sessions_dir: &Path,
    ) -> Result<(), std::io::Error> {
        self.sessions.write_jsonl_mirror(session, sessions_dir)
    }

    /// Append messages to the agent's canonical session for cross-channel persistence.
    pub fn append_canonical(
        &self,
        agent_id: AgentId,
        messages: &[openfang_types::message::Message],
        compaction_threshold: Option<usize>,
    ) -> OpenFangResult<()> {
        self.sessions
            .append_canonical(agent_id, messages, compaction_threshold)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Paired devices persistence
    // -----------------------------------------------------------------

    /// Load all paired devices from the database.
    pub fn load_paired_devices(&self) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT device_id, display_name, platform, paired_at, last_seen, push_token FROM paired_devices"
        ).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(serde_json::json!({
                    "device_id": row.get::<_, String>(0)?,
                    "display_name": row.get::<_, String>(1)?,
                    "platform": row.get::<_, String>(2)?,
                    "paired_at": row.get::<_, String>(3)?,
                    "last_seen": row.get::<_, String>(4)?,
                    "push_token": row.get::<_, Option<String>>(5)?,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut devices = Vec::new();
        for row in rows {
            devices.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(devices)
    }

    /// Save a paired device to the database (insert or replace).
    pub fn save_paired_device(
        &self,
        device_id: &str,
        display_name: &str,
        platform: &str,
        paired_at: &str,
        last_seen: &str,
        push_token: Option<&str>,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO paired_devices (device_id, display_name, platform, paired_at, last_seen, push_token) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![device_id, display_name, platform, paired_at, last_seen, push_token],
        ).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Remove a paired device from the database.
    pub fn remove_paired_device(&self, device_id: &str) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute(
            "DELETE FROM paired_devices WHERE device_id = ?1",
            rusqlite::params![device_id],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Embedding-aware memory operations
    // -----------------------------------------------------------------

    /// Store a memory with an embedding vector.
    pub fn remember_with_embedding(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        self.semantic
            .remember_with_embedding(agent_id, content, source, scope, metadata, embedding)
    }

    /// Recall memories using vector similarity when a query embedding is provided.
    pub fn recall_with_embedding(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        self.semantic
            .recall_with_embedding(query, limit, filter, query_embedding)
    }

    /// Update the embedding for an existing memory.
    pub fn update_embedding(&self, id: MemoryId, embedding: &[f32]) -> OpenFangResult<()> {
        self.semantic.update_embedding(id, embedding)
    }

    /// Async wrapper for `recall_with_embedding` — runs in a blocking thread.
    pub async fn recall_with_embedding_async(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        let store = self.semantic.clone();
        let query = query.to_string();
        let embedding_owned = query_embedding.map(|e| e.to_vec());
        tokio::task::spawn_blocking(move || {
            store.recall_with_embedding(&query, limit, filter, embedding_owned.as_deref())
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Async wrapper for `remember_with_embedding` — runs in a blocking thread.
    pub async fn remember_with_embedding_async(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        let store = self.semantic.clone();
        let content = content.to_string();
        let scope = scope.to_string();
        let embedding_owned = embedding.map(|e| e.to_vec());
        tokio::task::spawn_blocking(move || {
            store.remember_with_embedding(
                agent_id,
                &content,
                source,
                &scope,
                metadata,
                embedding_owned.as_deref(),
            )
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    // -----------------------------------------------------------------
    // Task queue operations
    // -----------------------------------------------------------------

    // CONTRACT — shared task-queue payload (memory <-> agent_send_async):
    //   `payload` is OPAQUE BYTES. The queue imposes NO schema: each owner
    //   frames its own record/envelope. Encoding: raw `&[u8]` at this Rust
    //   boundary, base64-STANDARD on the JSON surface.
    //   Two invariants neither owner may move without a cross-agent heads-up:
    //     1. the base64-STANDARD convention, and
    //     2. the payload SELECT column index: idx 6 on claim, idx 9 on list.
    //   Owners: memory subsystem (this crate) + coder-openfang-tools
    //   (agent_send_async). Touch a column index -> ping the other owner first.

    /// Post a new task to the shared queue. Returns the task ID.
    pub async fn task_post(
        &self,
        title: &str,
        description: &str,
        assigned_to: Option<&str>,
        created_by: Option<&str>,
        payload: &[u8],
    ) -> OpenFangResult<String> {
        // SECURITY: the WAKE_TASK_PREFIX title namespace is reserved for the
        // capability-gated agent_send_async producer (via `task_post_wake`).
        // Reject it on the ordinary path so an agent holding only `task_post`
        // cannot forge a wake row (forged sender/target/trigger) that the
        // kernel wake-consumer would dispatch — which would bypass both the
        // agent_send_async allowlist and the cycle/depth guards.
        if title.starts_with(openfang_types::wake::WAKE_TASK_PREFIX) {
            return Err(OpenFangError::InvalidInput(format!(
                "task title prefix '{}' is reserved for agent_send_async",
                openfang_types::wake::WAKE_TASK_PREFIX
            )));
        }
        self.task_post_raw(title, description, assigned_to, created_by, payload)
            .await
    }

    /// Privileged wake enqueue — the ONLY writer permitted to use the
    /// `WAKE_TASK_PREFIX` title namespace. Reached only from the kernel's
    /// `wake_post`, which the capability-gated `agent_send_async` producer
    /// calls; it is not exposed as an agent tool. Ordinary `task_post` rejects
    /// the prefix, so this is the sole trusted path into the wake queue.
    pub async fn task_post_wake(
        &self,
        title: &str,
        description: &str,
        assigned_to: Option<&str>,
        created_by: Option<&str>,
        payload: &[u8],
    ) -> OpenFangResult<String> {
        self.task_post_raw(title, description, assigned_to, created_by, payload)
            .await
    }

    /// Shared INSERT for both task_post paths. Imposes no title policy — the
    /// public wrappers own that.
    async fn task_post_raw(
        &self,
        title: &str,
        description: &str,
        assigned_to: Option<&str>,
        created_by: Option<&str>,
        payload: &[u8],
    ) -> OpenFangResult<String> {
        let conn = Arc::clone(&self.conn);
        let title = title.to_string();
        let description = description.to_string();
        let assigned_to = assigned_to.unwrap_or("").to_string();
        let created_by = created_by.unwrap_or("").to_string();
        let payload = payload.to_vec();

        tokio::task::spawn_blocking(move || {
            let id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            db.execute(
                "INSERT INTO task_queue (id, agent_id, task_type, payload, status, priority, created_at, title, description, assigned_to, created_by)
                 VALUES (?1, ?2, ?3, ?4, 'pending', 0, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![id, &created_by, &title, payload, now, title, description, assigned_to, created_by],
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            Ok(id)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Claim the next pending task (optionally for a specific assignee). Returns task JSON or None.
    pub async fn task_claim(&self, agent_id: &str) -> OpenFangResult<Option<serde_json::Value>> {
        let conn = Arc::clone(&self.conn);
        let agent_id = agent_id.to_string();
        // Wake tasks (title prefixed WAKE_TASK_PREFIX) belong to the kernel
        // wake-consumer, claimed via `task_claim_wake`. Exclude them here so an
        // agent's ordinary task_claim can never pull a wake out from under the
        // consumer (and a regular collaboration task is never run as a wake).
        let wake_like = format!("{}%", openfang_types::wake::WAKE_TASK_PREFIX);

        tokio::task::spawn_blocking(move || {
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            // Find first pending task assigned to this agent, or any unassigned pending task
            let mut stmt = db.prepare(
                "SELECT id, title, description, assigned_to, created_by, created_at, payload
                 FROM task_queue
                 WHERE status = 'pending' AND (assigned_to = ?1 OR assigned_to = '')
                   AND title NOT LIKE ?2
                 ORDER BY priority DESC, created_at ASC
                 LIMIT 1"
            ).map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let result = stmt.query_row(rusqlite::params![agent_id, wake_like], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Vec<u8>>(6)?,
                ))
            });

            match result {
                Ok((id, title, description, assigned, created_by, created_at, payload)) => {
                    // Update status to in_progress
                    db.execute(
                        "UPDATE task_queue SET status = 'in_progress', assigned_to = ?2 WHERE id = ?1",
                        rusqlite::params![id, agent_id],
                    ).map_err(|e| OpenFangError::Memory(e.to_string()))?;

                    Ok(Some(serde_json::json!({
                        "id": id,
                        "title": title,
                        "description": description,
                        "status": "in_progress",
                        "assigned_to": if assigned.is_empty() { &agent_id } else { &assigned },
                        "created_by": created_by,
                        "created_at": created_at,
                        "payload": base64::engine::general_purpose::STANDARD.encode(&payload),
                    })))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(OpenFangError::Memory(e.to_string())),
            }
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Claim the next pending wake and hand the kernel consumer a ready-to-use
    /// `(task_id, WakeEnvelope)` — base64 decode and envelope parsing stay
    /// inside this crate (which already owns the payload contract), so the
    /// kernel needs no base64 dependency. A claimed task whose payload fails to
    /// decode is marked completed with an error result (so it is not re-claimed
    /// forever) and `None` is returned, letting the consumer poll on.
    pub async fn claim_wake_for_dispatch(
        &self,
        per_caller_cap: usize,
    ) -> OpenFangResult<Option<(String, openfang_types::wake::WakeEnvelope)>> {
        // ANAI-104: `per_caller_cap` is threaded to `task_claim_wake` so the
        // per-caller in-flight limit is enforced atomically at the claim/flip.
        // Bound on poison rows skipped per call, so a flood of malformed wakes
        // can't spin this method indefinitely before yielding.
        const MAX_POISON_SKIPS: usize = 64;
        for _ in 0..MAX_POISON_SKIPS {
            let Some(task) = self.task_claim_wake(per_caller_cap).await? else {
                return Ok(None); // queue genuinely empty
            };
            let task_id = task
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let payload_b64 = task.get("payload").and_then(|v| v.as_str()).unwrap_or("");
            let bytes = match base64::engine::general_purpose::STANDARD.decode(payload_b64) {
                Ok(b) => b,
                Err(e) => {
                    let _ = self
                        .task_complete(&task_id, &format!("wake payload base64 decode failed: {e}"))
                        .await;
                    continue; // poison — skip and claim the next
                }
            };
            match openfang_types::wake::WakeEnvelope::from_payload(&bytes) {
                Ok(env) => return Ok(Some((task_id, env))),
                Err(e) => {
                    let _ = self
                        .task_complete(&task_id, &format!("wake envelope parse failed: {e}"))
                        .await;
                    continue; // poison — skip and claim the next
                }
            }
        }
        Ok(None) // hit the skip cap this call; next tick resumes the drain
    }

    /// Claim the next pending **wake** task — the consumer half of
    /// `agent_send_async`.
    ///
    /// The inverse filter of [`Self::task_claim`]: returns ONLY tasks whose
    /// title bears `WAKE_TASK_PREFIX`. It is **assignee-agnostic** by design —
    /// a single central kernel wake-consumer drains the wake queue and routes
    /// each task by the `target` carried in its `WakeEnvelope` payload (resolved
    /// name-or-UUID at dispatch), so the row's `assigned_to` string never has to
    /// match a particular query form. The flip to `in_progress` leaves
    /// `assigned_to` untouched (the producer already recorded the target there).
    /// Ordinary agents never call this; their `task_claim` excludes wake titles.
    /// Same JSON shape as `task_claim` — payload stays at column idx 6 under the
    /// base64-STANDARD convention.
    ///
    /// ## Per-caller in-flight cap (ANAI-104)
    ///
    /// `per_caller_cap` bounds how many wakes a single caller (`created_by`) may
    /// have `in_progress` at once — restoring the backpressure that async
    /// dispatch removed (the old blocking `agent_send` self-throttled the
    /// caller's own turn). Enforced as a correlated subquery in the claim
    /// SELECT: a pending wake is claimable only if its caller currently has
    /// FEWER than `per_caller_cap` wakes in flight. Queue-over-limit semantics —
    /// an over-cap caller's wakes stay `pending` (nothing rejected/lost) until
    /// one of its runs completes, while `ORDER BY` still advances to the next
    /// *eligible* caller, so a saturated caller never head-of-line-blocks the
    /// others. The count and the flip-to-`in_progress` share one locked
    /// connection, so the cap is authoritative and race-free (the single
    /// wake-consumer is the only claimer).
    pub async fn task_claim_wake(
        &self,
        per_caller_cap: usize,
    ) -> OpenFangResult<Option<serde_json::Value>> {
        // Bind as i64 for rusqlite; already floored to >=1 by the resolver.
        let cap = per_caller_cap as i64;
        let conn = Arc::clone(&self.conn);
        let wake_like = format!("{}%", openfang_types::wake::WAKE_TASK_PREFIX);

        tokio::task::spawn_blocking(move || {
            let db = conn
                .lock()
                .map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let mut stmt = db
                .prepare(
                    "SELECT id, title, description, assigned_to, created_by, created_at, payload
                     FROM task_queue
                     WHERE status = 'pending' AND title LIKE ?1
                       AND (
                           SELECT COUNT(*) FROM task_queue AS inflight
                           WHERE inflight.status = 'in_progress'
                             AND inflight.title LIKE ?1
                             AND inflight.created_by = task_queue.created_by
                       ) < ?2
                     ORDER BY priority DESC, created_at ASC
                     LIMIT 1",
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let result = stmt.query_row(rusqlite::params![wake_like, cap], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Vec<u8>>(6)?,
                ))
            });

            match result {
                Ok((id, title, description, assigned, created_by, created_at, payload)) => {
                    // Flip to in_progress; do NOT touch assigned_to — the producer
                    // already recorded the wake target there. Stamp `claimed_at`
                    // (ANAI-147): it is the ONLY record of when this row began
                    // occupying one of its caller's per-caller in-flight slots,
                    // and so the stale-claim reaper's sole clock.
                    let claimed_at = chrono::Utc::now().to_rfc3339();
                    db.execute(
                        "UPDATE task_queue SET status = 'in_progress', claimed_at = ?2 WHERE id = ?1",
                        rusqlite::params![id, claimed_at],
                    )
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;

                    Ok(Some(serde_json::json!({
                        "id": id,
                        "title": title,
                        "description": description,
                        "status": "in_progress",
                        "assigned_to": assigned,
                        "created_by": created_by,
                        "created_at": created_at,
                        "payload": base64::engine::general_purpose::STANDARD.encode(&payload),
                    })))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(OpenFangError::Memory(e.to_string())),
            }
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Fail-closed sweep of wakes stuck `in_progress` (ANAI-147).
    ///
    /// ## Why this must exist
    ///
    /// The per-caller in-flight cap (ANAI-104) is a *concurrency* limit whose
    /// slots are released by exactly one thing: `task_complete`, called at the
    /// end of `run_woken_agent_loop`. That loop runs on a **detached tokio
    /// task**, so a daemon restart kills it mid-flight and the row stays
    /// `in_progress` with `completed_at = NULL` forever. Nothing else ever
    /// touches it. Accumulate `per_caller_max` such orphans for one caller and
    /// the claim predicate's `COUNT(...) < cap` is false permanently: every
    /// later wake from that caller stays `pending` and is never claimed —
    /// silent, total starvation that looks exactly like "async sends are
    /// dropped".
    ///
    /// ## Semantics: fail closed, never re-run
    ///
    /// Reaped wakes are marked **completed with an error result**, not requeued.
    /// A wake is an instruction to an agent about a moment; re-dispatching a
    /// day-old one into a changed world is a worse failure than a logged loss,
    /// and a requeue-on-boot policy turns a wake that reliably crashes its
    /// dispatcher into an infinite restart loop. The row is preserved with a
    /// diagnostic `result`, so the loss is auditable rather than invisible.
    ///
    /// ## Scope
    ///
    /// `stale_after` selects the two callers:
    /// * `None` — **boot sweep**. Every in-flight wake is an orphan by
    ///   construction: the wake-consumer is the sole claimer and its dispatch
    ///   tasks are process-bound, so at daemon start no in-flight row can have a
    ///   live dispatcher. Must run BEFORE the consumer starts, or it will reap
    ///   wakes the fresh consumer just claimed.
    /// * `Some(d)` — **periodic sweep**. Reaps only rows claimed longer than `d`
    ///   ago, catching the non-restart leaks (panicked dispatch task, wedged
    ///   agent loop) that the boot sweep can't see.
    ///
    /// Only wake-titled rows are touched; an ordinary agent's `in_progress`
    /// collaboration task is none of this function's business.
    ///
    /// ## Deadline-aware staleness (ANAI-217)
    ///
    /// `deadline_grace` narrows the periodic sweep for rows that carry an
    /// ANAI-201 deadline: such a row is ALSO reaped once
    /// `claimed_at + envelope.timeout() + grace` has passed, even if the flat
    /// `stale_after` cutoff has not. The flat cutoff has to be set far above
    /// any legitimate turn because it applies to rows whose intended duration
    /// is unknown; a row that states its own deadline is not in that position,
    /// and holding its caller's slot for the flat hour after a bound it already
    /// blew is pure latency. `None` disables the rule (behaviour identical to
    /// pre-ANAI-217). Ignored on the boot sweep, which reaps unconditionally.
    ///
    /// Returns the reaped rows, payload included, so the caller can decode the
    /// envelope and discharge the sender's reply debt (ANAI-217) rather than
    /// merely log which callers were unwedged.
    pub async fn reap_in_flight_wakes(
        &self,
        stale_after: Option<std::time::Duration>,
        deadline_grace: Option<std::time::Duration>,
        reason: &str,
    ) -> OpenFangResult<Vec<ReapedWake>> {
        let conn = Arc::clone(&self.conn);
        let wake_like = format!("{}%", openfang_types::wake::WAKE_TASK_PREFIX);
        let reason = reason.to_string();

        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now();
            let cutoff =
                stale_after.and_then(|d| chrono::Duration::from_std(d).ok().map(|d| now - d));
            let db = conn
                .lock()
                .map_err(|e| OpenFangError::Internal(e.to_string()))?;

            // Select candidates first so the caller can be told exactly what was
            // lost. `claimed_at` is NULL for rows claimed by a pre-v11 binary;
            // fall back to `created_at` rather than skipping them, or the very
            // orphans that motivated this sweep would survive it.
            let mut stmt = db
                .prepare(
                    "SELECT id, created_by, COALESCE(NULLIF(claimed_at, ''), created_at), payload
                     FROM task_queue
                     WHERE status = 'in_progress' AND title LIKE ?1",
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params![wake_like], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1).unwrap_or_default(),
                        row.get::<_, Option<String>>(2).unwrap_or(None),
                        row.get::<_, Vec<u8>>(3).unwrap_or_default(),
                    ))
                })
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let mut doomed: Vec<ReapedWake> = Vec::new();
            for row in rows {
                let (id, created_by, at, payload) =
                    row.map_err(|e| OpenFangError::Memory(e.to_string()))?;
                // Read as raw bytes: `payload` is a BLOB column and the queue's
                // contract is opaque bytes (the base64 in `task_claim`/
                // `task_list` is a JSON-transport artifact, not storage). A
                // missing or unreadable value yields an empty Vec rather than a
                // skip — the row still holds a per-caller slot and must still be
                // reaped; the caller treats empty-or-unparseable as "debt
                // unpayable" and says so in the log rather than dropping it.
                // Parse rather than compare timestamp strings: a row written in
                // a different RFC3339 offset would order wrong
                // lexicographically and get reaped while live.
                let claimed = at
                    .as_deref()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|t| t.with_timezone(&chrono::Utc));
                match cutoff {
                    // Boot sweep: no cutoff, everything in flight is an orphan.
                    None => doomed.push(ReapedWake {
                        task_id: id,
                        created_by,
                        payload,
                        past_deadline: false,
                    }),
                    Some(cut) => {
                        // An unparseable/missing stamp is NOT reaped — a live
                        // loop killed by a bad timestamp is worse than an
                        // orphan that waits for the next boot sweep. Both rules
                        // below key off it, so bail once here.
                        let Some(claimed) = claimed else { continue };
                        let stale = claimed < cut;
                        // ANAI-217: the row's own stated deadline, when it has
                        // one, is a tighter and better-founded bound than the
                        // flat cutoff. Any failure to derive it (no grace
                        // configured, unparseable payload, arithmetic
                        // overflow) falls back to `stale` alone — the rule can
                        // only ever reap MORE than before, never less, and
                        // never on a guess.
                        let past_deadline = deadline_grace
                            .and_then(|g| {
                                let env =
                                    openfang_types::wake::WakeEnvelope::from_payload(&payload)
                                        .ok()?;
                                let bound =
                                    chrono::Duration::from_std(env.timeout().checked_add(g)?)
                                        .ok()?;
                                Some(claimed.checked_add_signed(bound)? < now)
                            })
                            .unwrap_or(false);
                        if stale || past_deadline {
                            doomed.push(ReapedWake {
                                task_id: id,
                                created_by,
                                payload,
                                past_deadline,
                            });
                        }
                    }
                }
            }

            let now_s = now.to_rfc3339();
            for w in &doomed {
                // Say which rule fired: a row reaped for blowing its own
                // deadline is a different diagnosis from one that sat past the
                // operator's flat cutoff, and the row's `result` is the only
                // durable record of either.
                let result = if w.past_deadline {
                    format!("{reason} (stated deadline + grace elapsed)")
                } else {
                    reason.clone()
                };
                db.execute(
                    "UPDATE task_queue SET status = 'completed', result = ?2, completed_at = ?3
                     WHERE id = ?1 AND status = 'in_progress'",
                    rusqlite::params![&w.task_id, &result, &now_s],
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            }
            Ok(doomed)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Wake-queue depth for one caller: `(pending, in_flight)` (ANAI-147).
    ///
    /// Backs the honesty fix on `agent_send_async`'s tool result. The enqueue
    /// returns a task id and says "queued", which reads as "will run" — but if
    /// the caller is at its per-caller cap the wake sits `pending` behind the
    /// cap indefinitely, and nothing in the result distinguishes the two. This
    /// lets the producer report the queue it just joined.
    pub async fn wake_queue_depth(&self, created_by: &str) -> OpenFangResult<(usize, usize)> {
        let conn = Arc::clone(&self.conn);
        let wake_like = format!("{}%", openfang_types::wake::WAKE_TASK_PREFIX);
        let created_by = created_by.to_string();

        tokio::task::spawn_blocking(move || {
            let db = conn
                .lock()
                .map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let count = |status: &str| -> OpenFangResult<usize> {
                db.query_row(
                    "SELECT COUNT(*) FROM task_queue
                     WHERE status = ?1 AND title LIKE ?2 AND created_by = ?3",
                    rusqlite::params![status, &wake_like, &created_by],
                    |row| row.get::<_, i64>(0),
                )
                .map(|n| n.max(0) as usize)
                .map_err(|e| OpenFangError::Memory(e.to_string()))
            };
            Ok((count("pending")?, count("in_progress")?))
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Mark a task as completed with a result string.
    pub async fn task_complete(&self, task_id: &str, result: &str) -> OpenFangResult<()> {
        let conn = Arc::clone(&self.conn);
        let task_id = task_id.to_string();
        let result = result.to_string();

        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let rows = db.execute(
                "UPDATE task_queue SET status = 'completed', result = ?2, completed_at = ?3 WHERE id = ?1",
                rusqlite::params![task_id, result, now],
            ).map_err(|e| OpenFangError::Memory(e.to_string()))?;
            if rows == 0 {
                return Err(OpenFangError::Internal(format!("Task not found: {task_id}")));
            }
            Ok(())
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// List tasks, optionally filtered by status.
    pub async fn task_list(&self, status: Option<&str>) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = Arc::clone(&self.conn);
        let status = status.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match &status {
                Some(s) => (
                    "SELECT id, title, description, status, assigned_to, created_by, created_at, completed_at, result, payload FROM task_queue WHERE status = ?1 ORDER BY created_at DESC",
                    vec![Box::new(s.clone())],
                ),
                None => (
                    "SELECT id, title, description, status, assigned_to, created_by, created_at, completed_at, result, payload FROM task_queue ORDER BY created_at DESC",
                    vec![],
                ),
            };

            let mut stmt = db.prepare(sql).map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let params_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
            let rows = stmt.query_map(params_refs.as_slice(), |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "title": row.get::<_, String>(1).unwrap_or_default(),
                    "description": row.get::<_, String>(2).unwrap_or_default(),
                    "status": row.get::<_, String>(3)?,
                    "assigned_to": row.get::<_, String>(4).unwrap_or_default(),
                    "created_by": row.get::<_, String>(5).unwrap_or_default(),
                    "created_at": row.get::<_, String>(6).unwrap_or_default(),
                    "completed_at": row.get::<_, Option<String>>(7).unwrap_or(None),
                    "result": row.get::<_, Option<String>>(8).unwrap_or(None),
                    "payload": base64::engine::general_purpose::STANDARD.encode(row.get::<_, Vec<u8>>(9).unwrap_or_default()),
                }))
            }).map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let mut tasks = Vec::new();
            for row in rows {
                tasks.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
            }
            Ok(tasks)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }
}

#[async_trait]
impl Memory for MemorySubstrate {
    async fn get(&self, agent_id: AgentId, key: &str) -> OpenFangResult<Option<serde_json::Value>> {
        let store = self.structured.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || store.get(agent_id, &key))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn set(
        &self,
        agent_id: AgentId,
        key: &str,
        value: serde_json::Value,
    ) -> OpenFangResult<()> {
        let store = self.structured.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || store.set(agent_id, &key, value))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn delete(&self, agent_id: AgentId, key: &str) -> OpenFangResult<()> {
        let store = self.structured.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || store.delete(agent_id, &key))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn remember(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> OpenFangResult<MemoryId> {
        let store = self.semantic.clone();
        let content = content.to_string();
        let scope = scope.to_string();
        tokio::task::spawn_blocking(move || {
            store.remember(agent_id, &content, source, &scope, metadata)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        let store = self.semantic.clone();
        let query = query.to_string();
        tokio::task::spawn_blocking(move || store.recall(&query, limit, filter))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn forget(&self, id: MemoryId) -> OpenFangResult<()> {
        let store = self.semantic.clone();
        tokio::task::spawn_blocking(move || store.forget(id))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn add_entity(&self, entity: Entity) -> OpenFangResult<String> {
        let store = self.knowledge.clone();
        tokio::task::spawn_blocking(move || store.add_entity(entity))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn add_relation(&self, relation: Relation) -> OpenFangResult<String> {
        let store = self.knowledge.clone();
        tokio::task::spawn_blocking(move || store.add_relation(relation))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn query_graph(&self, pattern: GraphPattern) -> OpenFangResult<Vec<GraphMatch>> {
        let store = self.knowledge.clone();
        tokio::task::spawn_blocking(move || store.query_graph(pattern))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn consolidate(&self) -> OpenFangResult<ConsolidationReport> {
        let engine = self.consolidation.clone();
        tokio::task::spawn_blocking(move || engine.consolidate())
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn export(&self, format: ExportFormat) -> OpenFangResult<Vec<u8>> {
        let _ = format;
        Ok(Vec::new())
    }

    async fn import(&self, _data: &[u8], _format: ExportFormat) -> OpenFangResult<ImportReport> {
        Ok(ImportReport {
            entities_imported: 0,
            relations_imported: 0,
            memories_imported: 0,
            errors: vec!["Import not yet implemented in Phase 1".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_substrate_kv() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent_id = AgentId::new();
        substrate
            .set(agent_id, "key", serde_json::json!("value"))
            .await
            .unwrap();
        let val = substrate.get(agent_id, "key").await.unwrap();
        assert_eq!(val, Some(serde_json::json!("value")));
    }

    #[tokio::test]
    async fn test_substrate_remember_recall() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent_id = AgentId::new();
        substrate
            .remember(
                agent_id,
                "Rust is a great language",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .await
            .unwrap();
        let results = substrate.recall("Rust", 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_task_post_and_list() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let id = substrate
            .task_post(
                "Review code",
                "Check the auth module for issues",
                Some("auditor"),
                Some("orchestrator"),
                b"",
            )
            .await
            .unwrap();
        assert!(!id.is_empty());

        let tasks = substrate.task_list(Some("pending")).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["title"], "Review code");
        assert_eq!(tasks[0]["assigned_to"], "auditor");
        assert_eq!(tasks[0]["status"], "pending");
    }

    #[tokio::test]
    async fn test_task_claim_and_complete() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let task_id = substrate
            .task_post(
                "Audit endpoint",
                "Security audit the /api/login endpoint",
                Some("auditor"),
                None,
                b"",
            )
            .await
            .unwrap();

        // Claim the task
        let claimed = substrate.task_claim("auditor").await.unwrap();
        assert!(claimed.is_some());
        let claimed = claimed.unwrap();
        assert_eq!(claimed["id"], task_id);
        assert_eq!(claimed["status"], "in_progress");

        // Complete the task
        substrate
            .task_complete(&task_id, "No vulnerabilities found")
            .await
            .unwrap();

        // Verify it shows as completed
        let tasks = substrate.task_list(Some("completed")).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["result"], "No vulnerabilities found");
    }

    #[tokio::test]
    async fn test_task_claim_empty() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let claimed = substrate.task_claim("nobody").await.unwrap();
        assert!(claimed.is_none());
    }

    // --- agent_send_async wake-queue invariants ---------------------------

    use openfang_types::turn::TurnTrigger;
    use openfang_types::wake::{WakeEnvelope, WakeLineage, WAKE_TASK_PREFIX};

    fn sample_wake_envelope(target: &str, sender: &str) -> WakeEnvelope {
        WakeEnvelope {
            target: target.to_string(),
            sender: sender.to_string(),
            message: "do the thing — with an em dash \u{2014} and bytes".to_string(),
            lineage: WakeLineage::root_at(sender).extended(target),
            trigger: TurnTrigger::AgentCall,
            origin: None,
            is_reply: false,
            surface_to: None,
            reply_kind: Default::default(),
            timeout_secs: Some(600),
            requested_timeout_secs: None,
        }
    }

    /// EDIT 5: pin the payload column index on BOTH claim (idx 6) and list
    /// (idx 9). A future column reorder that silently shifts either index would
    /// corrupt every wake; this round-trip catches it at the source.
    #[tokio::test]
    async fn test_wake_payload_roundtrip_column_index() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let env = sample_wake_envelope("worker-b", "orchestrator");
        let payload = env.to_payload().unwrap();

        let title = format!("{WAKE_TASK_PREFIX}worker-b");
        substrate
            .task_post_wake(
                &title,
                &env.message,
                Some("worker-b"),
                Some("orchestrator"),
                &payload,
            )
            .await
            .unwrap();

        // idx 9 on list: payload survives base64 round-trip and deserializes.
        let listed = substrate.task_list(Some("pending")).await.unwrap();
        assert_eq!(listed.len(), 1);
        let listed_bytes = base64::engine::general_purpose::STANDARD
            .decode(listed[0]["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(WakeEnvelope::from_payload(&listed_bytes).unwrap(), env);

        // idx 6 on claim: same payload, recovered via the wake-scoped claim.
        let claimed = substrate.task_claim_wake(1_000).await.unwrap().unwrap();
        let claimed_bytes = base64::engine::general_purpose::STANDARD
            .decode(claimed["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(WakeEnvelope::from_payload(&claimed_bytes).unwrap(), env);
    }

    /// The core steal-prevention invariant: an ordinary `task_claim` must NEVER
    /// pull a wake task, and `task_claim_wake` must NEVER pull a regular task —
    /// even when both are pending and assigned to the same agent.
    #[tokio::test]
    async fn test_wake_and_regular_queues_do_not_cross() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let env = sample_wake_envelope("worker-b", "orchestrator");

        // A regular collaboration task AND a wake, both assigned to worker-b.
        substrate
            .task_post(
                "Regular job",
                "do normal work",
                Some("worker-b"),
                Some("orchestrator"),
                b"",
            )
            .await
            .unwrap();
        substrate
            .task_post_wake(
                &format!("{WAKE_TASK_PREFIX}worker-b"),
                &env.message,
                Some("worker-b"),
                Some("orchestrator"),
                &env.to_payload().unwrap(),
            )
            .await
            .unwrap();

        // Ordinary claim gets the regular task, never the wake.
        let regular = substrate.task_claim("worker-b").await.unwrap().unwrap();
        assert_eq!(regular["title"], "Regular job");

        // Wake claim gets the wake, never the regular task.
        let wake = substrate.task_claim_wake(1_000).await.unwrap().unwrap();
        // assigned_to is preserved (the producer's target), not overwritten.
        assert_eq!(wake["assigned_to"], "worker-b");
        assert_eq!(wake["title"], format!("{WAKE_TASK_PREFIX}worker-b"));
    }

    /// The central consumer drains the wake queue regardless of target, and
    /// each claim flips exactly one wake to in_progress (no double-claim).
    #[tokio::test]
    async fn test_wake_claim_drains_central_queue() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        for target in ["worker-b", "worker-c"] {
            let env = sample_wake_envelope(target, "orchestrator");
            substrate
                .task_post_wake(
                    &format!("{WAKE_TASK_PREFIX}{target}"),
                    &env.message,
                    Some(target),
                    Some("orchestrator"),
                    &env.to_payload().unwrap(),
                )
                .await
                .unwrap();
        }

        // Two distinct wakes drained in two claims; the third claim is empty.
        let first = substrate.task_claim_wake(1_000).await.unwrap().unwrap();
        let second = substrate.task_claim_wake(1_000).await.unwrap().unwrap();
        assert_ne!(first["id"], second["id"]);
        assert!(substrate.task_claim_wake(1_000).await.unwrap().is_none());
    }

    /// SECURITY: ordinary task_post must REJECT the reserved wake title prefix,
    /// so an agent holding only the generic task_post tool cannot forge a wake
    /// the kernel consumer would dispatch. Only the privileged task_post_wake
    /// may write the namespace.
    #[tokio::test]
    async fn test_ordinary_task_post_rejects_forged_wake() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let env = sample_wake_envelope("victim", "attacker");
        let forged = substrate
            .task_post(
                &format!("{WAKE_TASK_PREFIX}victim"),
                "forged",
                Some("victim"),
                Some("attacker"),
                &env.to_payload().unwrap(),
            )
            .await;
        assert!(
            forged.is_err(),
            "ordinary task_post must reject a wake-prefixed title"
        );
        // And nothing reached the wake queue.
        assert!(substrate.task_claim_wake(1_000).await.unwrap().is_none());
        // The privileged path still works.
        assert!(substrate
            .task_post_wake(
                &format!("{WAKE_TASK_PREFIX}victim"),
                "legit",
                Some("victim"),
                Some("orchestrator"),
                &env.to_payload().unwrap(),
            )
            .await
            .is_ok());
        assert!(substrate.task_claim_wake(1_000).await.unwrap().is_some());
    }

    /// ANAI-104: the per-caller in-flight cap queues an over-cap caller's extra
    /// wakes (leaves them `pending`) while still admitting a DIFFERENT caller's
    /// wake — no cross-caller head-of-line blocking — and releases the queued
    /// wake once one of that caller's in-flight runs completes.
    #[tokio::test]
    async fn test_wake_per_caller_in_flight_cap() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();

        // Three wakes from "orch", one from "other".
        for (target, sender) in [
            ("w1", "orch"),
            ("w2", "orch"),
            ("w3", "orch"),
            ("x1", "other"),
        ] {
            let env = sample_wake_envelope(target, sender);
            substrate
                .task_post_wake(
                    &format!("{WAKE_TASK_PREFIX}{target}"),
                    &env.message,
                    Some(target),
                    Some(sender),
                    &env.to_payload().unwrap(),
                )
                .await
                .unwrap();
        }

        // Cap = 2 per caller. The first two claims drain "orch"'s oldest two
        // wakes, so created_by = "orch" now has 2 in flight.
        let c1 = substrate.task_claim_wake(2).await.unwrap().unwrap();
        let c2 = substrate.task_claim_wake(2).await.unwrap().unwrap();
        assert_eq!(c1["created_by"], "orch");
        assert_eq!(c2["created_by"], "orch");
        assert_ne!(c1["id"], c2["id"]);

        // "orch" is now AT cap: its third wake is not claimable. The claim skips
        // to the next eligible caller and returns "other"'s wake instead —
        // proving a saturated caller does not starve the others.
        let c3 = substrate.task_claim_wake(2).await.unwrap().unwrap();
        assert_eq!(c3["created_by"], "other");

        // Nothing else is eligible now: "orch" still at cap (2 in flight), and
        // "other" is at 1 with no pending wake left.
        assert!(substrate.task_claim_wake(2).await.unwrap().is_none());

        // Complete one of "orch"'s in-flight runs -> it drops to 1 in flight,
        // so its queued third wake becomes claimable.
        substrate
            .task_complete(c1["id"].as_str().unwrap(), "done")
            .await
            .unwrap();
        let c4 = substrate.task_claim_wake(2).await.unwrap().unwrap();
        assert_eq!(c4["created_by"], "orch");
        assert_ne!(c4["id"], c2["id"]);
    }

    /// Helper: enqueue one wake from `sender` to `target`, return its task id.
    async fn post_wake(substrate: &MemorySubstrate, target: &str, sender: &str) -> String {
        let env = sample_wake_envelope(target, sender);
        substrate
            .task_post_wake(
                &format!("{WAKE_TASK_PREFIX}{target}"),
                &env.message,
                Some(target),
                Some(sender),
                &env.to_payload().unwrap(),
            )
            .await
            .unwrap()
    }

    /// ANAI-147, the keystone: reproduce the production wedge on a REAL
    /// file-backed substrate, then prove the boot reaper clears it.
    ///
    /// Sequence is the exact one that bit the fleet: a caller saturates its
    /// per-caller cap, the daemon dies (dropping the detached dispatch tasks
    /// that would have called `task_complete`), and on restart every later wake
    /// from that caller is unclaimable forever. In-memory tests cannot show
    /// this — the wedge only exists because the rows OUTLIVE the process.
    #[tokio::test]
    async fn test_boot_reaper_unwedges_starved_caller() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("wake_reap.db");
        let orphans: Vec<String>;

        // --- Pre-restart daemon: two wakes claimed (cap 2), never completed. ---
        {
            let substrate = MemorySubstrate::open(&db_path, 0.1, &MemoryConfig::default()).unwrap();
            for t in ["w1", "w2", "w3"] {
                post_wake(&substrate, t, "orch").await;
            }
            let a = substrate.task_claim_wake(2).await.unwrap().unwrap();
            let b = substrate.task_claim_wake(2).await.unwrap().unwrap();
            orphans = vec![
                a["id"].as_str().unwrap().to_string(),
                b["id"].as_str().unwrap().to_string(),
            ];
            // At cap: the third wake is already unclaimable.
            assert!(substrate.task_claim_wake(2).await.unwrap().is_none());
            // Substrate drops WITHOUT task_complete — the detached dispatch
            // tasks died with the process, exactly as on a daemon restart.
        }

        let substrate = MemorySubstrate::open(&db_path, 0.1, &MemoryConfig::default()).unwrap();

        // The wedge is real and survives the restart: w3 is still starved.
        assert!(
            substrate.task_claim_wake(2).await.unwrap().is_none(),
            "orphaned in-flight rows must starve the caller before the reap"
        );

        // Boot sweep: no cutoff, everything in flight is an orphan.
        let reaped = substrate
            .reap_in_flight_wakes(None, None, "wake orphaned by daemon restart")
            .await
            .unwrap();
        assert_eq!(reaped.len(), 2);
        for w in &reaped {
            assert!(orphans.contains(&w.task_id));
            assert_eq!(w.created_by, "orch");
            // ANAI-217: the payload must survive the reap, or the kernel has
            // nothing to address the outstanding reply to.
            assert!(
                !w.payload.is_empty(),
                "a reaped wake must carry its envelope so the debt can be paid"
            );
            assert!(!w.past_deadline, "the boot sweep reaps unconditionally");
        }

        // Fail-CLOSED, not requeued: the orphans are completed with a
        // diagnostic, and must never be dispatched late.
        let completed = substrate.task_list(Some("completed")).await.unwrap();
        assert_eq!(completed.len(), 2);
        for t in &completed {
            assert_eq!(
                t["result"].as_str().unwrap(),
                "wake orphaned by daemon restart"
            );
            assert!(t["completed_at"].as_str().is_some());
        }

        // And the caller is unwedged: w3 finally becomes claimable.
        let freed = substrate.task_claim_wake(2).await.unwrap().unwrap();
        assert_eq!(freed["created_by"], "orch");
        assert!(!orphans.contains(&freed["id"].as_str().unwrap().to_string()));
        // Exactly the three we posted; the reap invented no new rows.
        assert_eq!(substrate.task_list(None).await.unwrap().len(), 3);
    }

    /// The stale sweep must be a scalpel, not the boot hammer: a wake claimed
    /// moments ago is a LIVE agent loop, and reaping it would free its caller's
    /// slot while the loop still runs — double-dispatch pressure and a bogus
    /// "failed" result on work that is about to succeed.
    #[tokio::test]
    async fn test_stale_reaper_spares_fresh_claims_and_regular_tasks() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        post_wake(&substrate, "w1", "orch").await;
        substrate
            .task_post("Regular job", "normal work", Some("w1"), Some("orch"), b"")
            .await
            .unwrap();

        let wake = substrate.task_claim_wake(4).await.unwrap().unwrap();
        let regular = substrate.task_claim("w1").await.unwrap().unwrap();

        // Fresh claim, one-hour staleness bound: nothing is reaped.
        let spared = substrate
            .reap_in_flight_wakes(Some(std::time::Duration::from_secs(3600)), None, "stale")
            .await
            .unwrap();
        assert!(spared.is_empty(), "a just-claimed wake must not be reaped");

        // Zero-second bound reaps the wake — and ONLY the wake. An ordinary
        // agent's in-flight collaboration task is not the reaper's business.
        let reaped = substrate
            .reap_in_flight_wakes(Some(std::time::Duration::ZERO), None, "stale claim timeout")
            .await
            .unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].task_id, wake["id"].as_str().unwrap());

        let still_running = substrate.task_list(Some("in_progress")).await.unwrap();
        assert_eq!(still_running.len(), 1);
        assert_eq!(still_running[0]["id"], regular["id"]);
    }

    /// ANAI-217: a wake that states its OWN deadline is judged against that,
    /// not against the operator's flat stale cutoff.
    ///
    /// The pair is the point. Both rows are claimed in the same instant and
    /// swept with the same one-hour cutoff, so the flat rule spares both; the
    /// only thing separating them is the deadline each carries in its own
    /// payload. A regression that reverts to the flat rule fails on the first
    /// row; one that ignores the deadline entirely and reaps everything fails
    /// on the second.
    #[tokio::test]
    async fn test_stale_reaper_honors_the_wakes_own_deadline() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();

        let mut expired = sample_wake_envelope("worker-fast", "orch");
        expired.timeout_secs = Some(0); // already past its bound at claim time
        let mut patient = sample_wake_envelope("worker-slow", "orch");
        patient.timeout_secs = Some(3600); // nowhere near its bound

        for env in [&expired, &patient] {
            substrate
                .task_post_wake(
                    &format!("{WAKE_TASK_PREFIX}{}", env.target),
                    &env.message,
                    Some(&env.target),
                    Some("orch"),
                    &env.to_payload().unwrap(),
                )
                .await
                .unwrap();
        }
        let first = substrate.task_claim_wake(4).await.unwrap().unwrap();
        let second = substrate.task_claim_wake(4).await.unwrap().unwrap();
        let claimed: Vec<&str> = vec![
            first["title"].as_str().unwrap(),
            second["title"].as_str().unwrap(),
        ];
        assert_eq!(claimed.len(), 2, "both wakes must be in flight");

        // Deadline rule OFF: the flat hour spares both, as it always did.
        let spared = substrate
            .reap_in_flight_wakes(Some(std::time::Duration::from_secs(3600)), None, "stale")
            .await
            .unwrap();
        assert!(
            spared.is_empty(),
            "with the deadline rule disabled, the flat cutoff must spare both"
        );

        // Deadline rule ON: only the row that blew its own bound is reaped.
        let reaped = substrate
            .reap_in_flight_wakes(
                Some(std::time::Duration::from_secs(3600)),
                Some(std::time::Duration::ZERO),
                "stale claim timeout",
            )
            .await
            .unwrap();
        assert_eq!(
            reaped.len(),
            1,
            "exactly the wake past its own deadline is reaped"
        );
        assert!(
            reaped[0].past_deadline,
            "the row must report WHICH rule reaped it — the diagnosis differs"
        );
        let env = WakeEnvelope::from_payload(&reaped[0].payload).unwrap();
        assert_eq!(env.target, "worker-fast");

        // The spared one is still in flight, still dispatching.
        let still_running = substrate.task_list(Some("in_progress")).await.unwrap();
        assert_eq!(still_running.len(), 1);
        assert!(still_running[0]["title"]
            .as_str()
            .unwrap()
            .ends_with("worker-slow"));

        // And the reaped row's `result` names the rule, not just "stale".
        let completed = substrate.task_list(Some("completed")).await.unwrap();
        assert_eq!(completed.len(), 1);
        assert!(
            completed[0]["result"]
                .as_str()
                .unwrap()
                .contains("stated deadline + grace elapsed"),
            "the durable record must say why: {:?}",
            completed[0]["result"]
        );
    }

    /// The honesty fix's data source: per-caller wake depth, so the enqueue can
    /// say whether the wake it just queued is dispatchable or stacked behind a
    /// saturated cap.
    #[tokio::test]
    async fn test_wake_queue_depth_is_per_caller() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        for t in ["w1", "w2", "w3"] {
            post_wake(&substrate, t, "orch").await;
        }
        post_wake(&substrate, "x1", "other").await;

        assert_eq!(substrate.wake_queue_depth("orch").await.unwrap(), (3, 0));
        assert_eq!(substrate.wake_queue_depth("other").await.unwrap(), (1, 0));
        assert_eq!(substrate.wake_queue_depth("nobody").await.unwrap(), (0, 0));

        substrate.task_claim_wake(2).await.unwrap().unwrap();
        assert_eq!(substrate.wake_queue_depth("orch").await.unwrap(), (2, 1));
        // Another caller's traffic never shows up in this caller's depth.
        assert_eq!(substrate.wake_queue_depth("other").await.unwrap(), (1, 0));
    }

    /// ANAI-107 (Stage-A mechanism smoke): the keystone durability proof that
    /// no in-memory test can give. A queued wake must survive a real
    /// file-backed WAL substrate being dropped (daemon shutdown) and reopened
    /// (daemon restart), then come back claimable through the consumer's decode
    /// path with its cross-agent lineage byte-identical. Exercises
    /// `claim_wake_for_dispatch` (the base64 + envelope decode wrapper the
    /// kernel wake-consumer actually calls) end-to-end against on-disk state.
    ///
    /// Boundary: the woken turn itself (`TurnPolicy::woken()` + phantom guard)
    /// needs the kernel and is covered by ANAI-118's runtime tests; the budget
    /// trips by `wake_{emit,tree}_admit` unit tests. This closes the ONE gap
    /// neither covers — on-disk survival + claim + lineage integrity.
    #[tokio::test]
    async fn test_wake_survives_daemon_reload_claim_and_lineage_intact() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("wake_reload.db");

        // A three-hop chain: orchestrator -> worker-a -> worker-b. The root
        // (orchestrator) is the per-tree budget owner; current is worker-b.
        let env = WakeEnvelope {
            target: "worker-b".to_string(),
            sender: "worker-a".to_string(),
            message: "survive the reload \u{2014} em dash and unicode \u{2713}".to_string(),
            lineage: WakeLineage::root_at("orchestrator")
                .extended("worker-a")
                .extended("worker-b"),
            trigger: TurnTrigger::AgentCall,
            origin: Some("channel:1086446153098342510".to_string()),
            is_reply: false,
            surface_to: None,
            reply_kind: Default::default(),
            // ANAI-201: the sender's deadline is durable state, so it must
            // survive the substrate reload this test exists to prove. A
            // deadline that did not survive a restart would silently revert to
            // the configured default, changing the contract the orchestrator
            // set — the exact failure clamp-at-send exists to prevent.
            timeout_secs: Some(1234),
            requested_timeout_secs: Some(9),
        };

        // --- Producer half: post the wake into a file-backed WAL substrate. ---
        {
            let substrate = MemorySubstrate::open(&db_path, 0.1, &MemoryConfig::default()).unwrap();
            substrate
                .task_post_wake(
                    &format!("{WAKE_TASK_PREFIX}worker-b"),
                    &env.message,
                    Some("worker-b"),
                    Some("worker-a"),
                    &env.to_payload().unwrap(),
                )
                .await
                .unwrap();
            // substrate drops here -> connection closes -> WAL persists to disk.
        }

        // --- Simulated daemon restart: reopen the SAME on-disk database. ---
        let substrate = MemorySubstrate::open(&db_path, 0.1, &MemoryConfig::default()).unwrap();

        // --- Consumer half: the real dispatch-decode path claims it. ---
        let (task_id, decoded) = substrate
            .claim_wake_for_dispatch(4)
            .await
            .unwrap()
            .expect("wake must survive the reload and remain claimable");
        assert!(!task_id.is_empty());

        // The envelope round-tripped through SQLite + reload byte-for-byte...
        assert_eq!(decoded, env, "envelope must survive reload intact");
        // ...and the cross-agent lineage threaded through unchanged: root still
        // owns the budget, current is the last hop, depth is preserved.
        assert_eq!(decoded.lineage.root(), Some("orchestrator"));
        assert_eq!(decoded.lineage.current(), Some("worker-b"));
        assert_eq!(decoded.lineage.depth(), 3);
        assert_eq!(
            decoded.lineage.as_slice(),
            &["orchestrator", "worker-a", "worker-b"]
        );
        assert_eq!(decoded.trigger, TurnTrigger::AgentCall);

        // The claim flipped exactly one wake to in_progress; the queue is now
        // drained, so a second dispatch claim finds nothing.
        assert!(substrate
            .claim_wake_for_dispatch(4)
            .await
            .unwrap()
            .is_none());
    }

    /// ANAI-219: the sweep the kernel task calls is inert at the default
    /// timeout of 0. This is the whole safety argument for landing A unarmed —
    /// if it ever stops holding, arming becomes accidental.
    #[tokio::test]
    async fn idle_sweep_is_inert_when_the_timer_is_off() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent = AgentId::new();
        let episode = substrate.ensure_open_episode_async(agent).await.unwrap();
        backdate_episode(&substrate, episode, 10_000);

        assert_eq!(substrate.sweep_idle_episodes_async().await.unwrap(), 0);
        assert!(
            substrate.current_episode(agent).unwrap().unwrap().is_open(),
            "timeout 0 must leave even an ancient episode open"
        );
    }

    /// ...and with the timer armed, the same call closes the quiet agent's
    /// episode without that agent taking another turn. That is the gap the
    /// lazy `ensure_open` path cannot cover.
    #[tokio::test]
    async fn idle_sweep_closes_a_quiet_agents_episode() {
        let substrate = MemorySubstrate::open_in_memory_with_idle_timeout(0.1, 120).unwrap();
        let quiet = AgentId::new();
        let active = AgentId::new();
        let stale = substrate.ensure_open_episode_async(quiet).await.unwrap();
        substrate.ensure_open_episode_async(active).await.unwrap();
        backdate_episode(&substrate, stale, 121);

        assert_eq!(substrate.sweep_idle_episodes_async().await.unwrap(), 1);

        let closed = substrate.episodes().get(stale).unwrap().unwrap();
        assert!(!closed.is_open());
        assert_eq!(closed.close_reason, Some(CloseReason::Timer));
        assert!(
            closed.summary.is_none(),
            "A closes; it does not summarise — that is B (ANAI-220)"
        );
        assert!(
            substrate
                .current_episode(active)
                .unwrap()
                .unwrap()
                .is_open(),
            "the sweep must not touch an episode inside its idle gap"
        );

        // Idempotent: nothing left past the cutoff on the next tick.
        assert_eq!(substrate.sweep_idle_episodes_async().await.unwrap(), 0);
    }

    /// Force an episode's activity clock into the past so the timer path can be
    /// exercised without sleeping.
    fn backdate_episode(substrate: &MemorySubstrate, id: uuid::Uuid, minutes: i64) {
        let ts = (chrono::Utc::now() - chrono::Duration::minutes(minutes)).to_rfc3339();
        substrate
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE episodes SET opened_at = ?2, last_activity_at = ?2 WHERE id = ?1",
                rusqlite::params![id.to_string(), ts],
            )
            .unwrap();
    }
}
