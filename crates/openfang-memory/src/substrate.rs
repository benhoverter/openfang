//! MemorySubstrate: unified implementation of the `Memory` trait.
//!
//! Composes the structured store, semantic store, knowledge store,
//! session store, and consolidation engine behind a single async API.

use crate::consolidation::ConsolidationEngine;
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
            consolidation: ConsolidationEngine::new(shared, decay_rate),
        })
    }

    /// Get a reference to the usage store.
    pub fn usage(&self) -> &UsageStore {
        &self.usage
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
                    // already recorded the wake target there.
                    db.execute(
                        "UPDATE task_queue SET status = 'in_progress' WHERE id = ?1",
                        rusqlite::params![id],
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
}
