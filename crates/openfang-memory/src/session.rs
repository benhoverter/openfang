//! Session management — load/save conversation history.

use chrono::Utc;
use openfang_types::agent::{AgentId, SessionId};
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::message::{ContentBlock, Message, MessageContent, Role};
use rusqlite::Connection;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A conversation session with message history.
#[derive(Debug, Clone)]
pub struct Session {
    /// Session ID.
    pub id: SessionId,
    /// Owning agent ID.
    pub agent_id: AgentId,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Estimated token count for the context window.
    pub context_window_tokens: u64,
    /// Optional human-readable session label.
    pub label: Option<String>,
}

/// Session store backed by SQLite.
#[derive(Clone)]
pub struct SessionStore {
    conn: Arc<Mutex<Connection>>,
}

impl SessionStore {
    /// Create a new session store wrapping the given connection.
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Load a session from the database.
    pub fn get_session(&self, session_id: SessionId) -> OpenFangResult<Option<Session>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT agent_id, messages, context_window_tokens, label FROM sessions WHERE id = ?1")
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let result = stmt.query_row(rusqlite::params![session_id.0.to_string()], |row| {
            let agent_str: String = row.get(0)?;
            let messages_blob: Vec<u8> = row.get(1)?;
            let tokens: i64 = row.get(2)?;
            let label: Option<String> = row.get(3).unwrap_or(None);
            Ok((agent_str, messages_blob, tokens, label))
        });

        match result {
            Ok((agent_str, messages_blob, tokens, label)) => {
                let agent_id = uuid::Uuid::parse_str(&agent_str)
                    .map(AgentId)
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                let messages: Vec<Message> = rmp_serde::from_slice(&messages_blob)
                    .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
                Ok(Some(Session {
                    id: session_id,
                    agent_id,
                    messages,
                    context_window_tokens: tokens as u64,
                    label,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(OpenFangError::Memory(e.to_string())),
        }
    }

    /// Fetch just the `updated_at` timestamp (RFC3339) for a session.
    ///
    /// Returns `None` if the session does not exist. This is a targeted
    /// accessor for the turn-context envelope (ANAI-128): it needs the
    /// last-agent-activity stamp without paying to deserialize the message
    /// blob or forcing a field onto the widely-constructed `Session` struct.
    pub fn session_updated_at(&self, session_id: SessionId) -> OpenFangResult<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let result = conn.query_row(
            "SELECT updated_at FROM sessions WHERE id = ?1",
            rusqlite::params![session_id.0.to_string()],
            |row| row.get::<_, String>(0),
        );
        match result {
            Ok(updated_at) => Ok(Some(updated_at)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(OpenFangError::Memory(e.to_string())),
        }
    }

    /// Save a session to the database.
    pub fn save_session(&self, session: &Session) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let messages_blob = rmp_serde::to_vec_named(&session.messages)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO sessions (id, agent_id, messages, context_window_tokens, label, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)
             ON CONFLICT(id) DO UPDATE SET messages = ?3, context_window_tokens = ?4, label = ?5, updated_at = ?6",
            rusqlite::params![
                session.id.0.to_string(),
                session.agent_id.0.to_string(),
                messages_blob,
                session.context_window_tokens as i64,
                session.label.as_deref(),
                now,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Delete a session from the database.
    pub fn delete_session(&self, session_id: SessionId) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM sessions WHERE id = ?1",
            rusqlite::params![session_id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Delete all sessions belonging to an agent.
    pub fn delete_agent_sessions(&self, agent_id: AgentId) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM sessions WHERE agent_id = ?1",
            rusqlite::params![agent_id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Delete the canonical (cross-channel) session for an agent.
    pub fn delete_canonical_session(&self, agent_id: AgentId) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM canonical_sessions WHERE agent_id = ?1",
            rusqlite::params![agent_id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// List all sessions with metadata (session_id, agent_id, message_count, created_at).
    pub fn list_sessions(&self) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, agent_id, messages, created_at, label FROM sessions ORDER BY created_at DESC",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                let session_id: String = row.get(0)?;
                let agent_id: String = row.get(1)?;
                let messages_blob: Vec<u8> = row.get(2)?;
                let created_at: String = row.get(3)?;
                let label: Option<String> = row.get(4)?;
                // Deserialize just to count messages
                let msg_count = rmp_serde::from_slice::<Vec<Message>>(&messages_blob)
                    .map(|m| m.len())
                    .unwrap_or(0);
                Ok(serde_json::json!({
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "message_count": msg_count,
                    "created_at": created_at,
                    "label": label,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(sessions)
    }

    /// Create a new empty session for an agent.
    pub fn create_session(&self, agent_id: AgentId) -> OpenFangResult<Session> {
        let session = Session {
            id: SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: None,
        };
        self.save_session(&session)?;
        Ok(session)
    }

    /// Set the label on an existing session.
    pub fn set_session_label(
        &self,
        session_id: SessionId,
        label: Option<&str>,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE sessions SET label = ?1, updated_at = ?2 WHERE id = ?3",
            rusqlite::params![label, Utc::now().to_rfc3339(), session_id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Find a session by label for a given agent.
    pub fn find_session_by_label(
        &self,
        agent_id: AgentId,
        label: &str,
    ) -> OpenFangResult<Option<Session>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, messages, context_window_tokens, label FROM sessions \
                 WHERE agent_id = ?1 AND label = ?2 LIMIT 1",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let result = stmt.query_row(rusqlite::params![agent_id.0.to_string(), label], |row| {
            let id_str: String = row.get(0)?;
            let messages_blob: Vec<u8> = row.get(1)?;
            let tokens: i64 = row.get(2)?;
            let lbl: Option<String> = row.get(3).unwrap_or(None);
            Ok((id_str, messages_blob, tokens, lbl))
        });

        match result {
            Ok((id_str, messages_blob, tokens, lbl)) => {
                let session_id = uuid::Uuid::parse_str(&id_str)
                    .map(SessionId)
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                let messages: Vec<Message> = rmp_serde::from_slice(&messages_blob)
                    .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
                Ok(Some(Session {
                    id: session_id,
                    agent_id,
                    messages,
                    context_window_tokens: tokens as u64,
                    label: lbl,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(OpenFangError::Memory(e.to_string())),
        }
    }
}

impl SessionStore {
    /// List all sessions for a specific agent.
    pub fn list_agent_sessions(&self, agent_id: AgentId) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, messages, created_at, label FROM sessions WHERE agent_id = ?1 ORDER BY created_at DESC",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let rows = stmt
            .query_map(rusqlite::params![agent_id.0.to_string()], |row| {
                let session_id: String = row.get(0)?;
                let messages_blob: Vec<u8> = row.get(1)?;
                let created_at: String = row.get(2)?;
                let label: Option<String> = row.get(3)?;
                let msg_count = rmp_serde::from_slice::<Vec<Message>>(&messages_blob)
                    .map(|m| m.len())
                    .unwrap_or(0);
                Ok(serde_json::json!({
                    "session_id": session_id,
                    "message_count": msg_count,
                    "created_at": created_at,
                    "label": label,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(sessions)
    }

    /// Create a new session with an optional label.
    pub fn create_session_with_label(
        &self,
        agent_id: AgentId,
        label: Option<&str>,
    ) -> OpenFangResult<Session> {
        let session = Session {
            id: SessionId::new(),
            agent_id,
            messages: Vec::new(),
            context_window_tokens: 0,
            label: label.map(|s| s.to_string()),
        };
        self.save_session(&session)?;
        Ok(session)
    }

    /// Store an LLM-generated summary, replacing older messages with the summary
    /// and keeping only the specified recent messages.
    ///
    /// This is used by the LLM-based compactor to replace text-truncation compaction
    /// with an intelligent, LLM-generated summary of older conversation history.
    pub fn store_llm_summary(
        &self,
        agent_id: AgentId,
        summary: &str,
        kept_messages: Vec<Message>,
    ) -> OpenFangResult<()> {
        let mut canonical = self.load_canonical(agent_id)?;
        canonical.compacted_summary = Some(summary.to_string());
        canonical.messages = kept_messages;
        canonical.compaction_cursor = 0;
        canonical.updated_at = Utc::now().to_rfc3339();
        self.save_canonical(&canonical)
    }
}

impl SessionStore {
    /// ANAI-246: re-anchor the canonical session at an episode boundary.
    ///
    /// Drops the verbatim cross-channel messages accumulated before the
    /// boundary and **keeps `compacted_summary`**. This is deliberately not
    /// [`Self::delete_canonical_session`]: an episode close means "that topic
    /// ended", not "forget you have a past". The summary is the agent's only
    /// durable, cheap account of everything before the boundary, and it is the
    /// source of the index-0 `canonical_context_msg` that ANAI-242/244 spend
    /// real effort protecting from the trim ladder. Deleting it on every close
    /// would hand the ladder the amnesia it was hardened against.
    ///
    /// Returns the number of verbatim messages dropped, for the caller's log.
    ///
    /// ANAI-247: `prime_for` is set in the same write. `Some(slug)` primes the
    /// next episode for that project; `None` clears any prior priming. It is
    /// deliberately not "leave it alone on `None`" — a boundary drawn without
    /// a slug means the agent is no longer working on the old thing, and a
    /// stale pack is worse than no pack.
    pub fn reanchor_canonical(
        &self,
        agent_id: AgentId,
        prime_for: Option<&str>,
    ) -> OpenFangResult<usize> {
        let mut canonical = self.load_canonical(agent_id)?;
        let dropped = canonical.messages.len();
        let prime_for = prime_for.map(str::to_string);
        if dropped == 0 && canonical.prime_for == prime_for {
            // Nothing to re-anchor and nothing to prime. Return without a
            // write so a double close does not churn `updated_at` for no
            // reason. The `prime_for` half of the guard matters: a close that
            // drops no messages may still be the one that primes the next
            // episode, and skipping that write would lose the pack silently.
            return Ok(0);
        }
        canonical.messages.clear();
        canonical.compaction_cursor = 0;
        canonical.prime_for = prime_for;
        canonical.updated_at = Utc::now().to_rfc3339();
        self.save_canonical(&canonical)?;
        Ok(dropped)
    }
}

/// Default number of recent messages to include from canonical session.
const DEFAULT_CANONICAL_WINDOW: usize = 50;

/// Default compaction threshold: when message count exceeds this, compact older messages.
const DEFAULT_COMPACTION_THRESHOLD: usize = 100;

/// A canonical session stores persistent cross-channel context for an agent.
///
/// Unlike regular sessions (one per channel interaction), there is one canonical
/// session per agent. All channels contribute to it, so what a user tells an agent
/// on Telegram is remembered on Discord.
#[derive(Debug, Clone)]
pub struct CanonicalSession {
    /// The agent this session belongs to.
    pub agent_id: AgentId,
    /// Full message history (post-compaction window).
    pub messages: Vec<Message>,
    /// Index marking how far compaction has processed.
    pub compaction_cursor: usize,
    /// Summary of compacted (older) messages.
    pub compacted_summary: Option<String>,
    /// ANAI-247: the project slug this agent was primed for at the last
    /// episode boundary, or `None` when it is not primed.
    ///
    /// Control state, not a claim — see `migration::migrate_v15`. Read at
    /// prompt-build time to assemble the rehydration pack; written only by
    /// [`SessionStore::reanchor_canonical`].
    pub prime_for: Option<String>,
    /// Last update time.
    pub updated_at: String,
}

impl SessionStore {
    /// Load the canonical session for an agent, creating one if it doesn't exist.
    pub fn load_canonical(&self, agent_id: AgentId) -> OpenFangResult<CanonicalSession> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT messages, compaction_cursor, compacted_summary, updated_at, prime_for \
                 FROM canonical_sessions WHERE agent_id = ?1",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let result = stmt.query_row(rusqlite::params![agent_id.0.to_string()], |row| {
            let messages_blob: Vec<u8> = row.get(0)?;
            let cursor: i64 = row.get(1)?;
            let summary: Option<String> = row.get(2)?;
            let updated_at: String = row.get(3)?;
            let prime_for: Option<String> = row.get(4)?;
            Ok((messages_blob, cursor, summary, updated_at, prime_for))
        });

        match result {
            Ok((messages_blob, cursor, summary, updated_at, prime_for)) => {
                let messages: Vec<Message> = rmp_serde::from_slice(&messages_blob)
                    .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
                Ok(CanonicalSession {
                    agent_id,
                    messages,
                    compaction_cursor: cursor as usize,
                    compacted_summary: summary,
                    prime_for,
                    updated_at,
                })
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                let now = Utc::now().to_rfc3339();
                Ok(CanonicalSession {
                    agent_id,
                    messages: Vec::new(),
                    compaction_cursor: 0,
                    compacted_summary: None,
                    prime_for: None,
                    updated_at: now,
                })
            }
            Err(e) => Err(OpenFangError::Memory(e.to_string())),
        }
    }

    /// Append new messages to the canonical session and compact if over threshold.
    ///
    /// Compaction summarizes old messages into a text summary and trims the
    /// message list. The `compaction_threshold` controls when this happens
    /// (default: 100 messages).
    pub fn append_canonical(
        &self,
        agent_id: AgentId,
        new_messages: &[Message],
        compaction_threshold: Option<usize>,
    ) -> OpenFangResult<CanonicalSession> {
        let mut canonical = self.load_canonical(agent_id)?;
        canonical.messages.extend(new_messages.iter().cloned());

        let threshold = compaction_threshold.unwrap_or(DEFAULT_COMPACTION_THRESHOLD);

        // Compact if over threshold
        if canonical.messages.len() > threshold {
            let keep_count = DEFAULT_CANONICAL_WINDOW;
            let to_compact = canonical.messages.len().saturating_sub(keep_count);
            if to_compact > canonical.compaction_cursor {
                // Build a summary from the messages being compacted
                let compacting = &canonical.messages[canonical.compaction_cursor..to_compact];
                let mut summary_parts: Vec<String> = Vec::new();
                if let Some(ref existing) = canonical.compacted_summary {
                    summary_parts.push(existing.clone());
                }
                for msg in compacting {
                    let role = match msg.role {
                        openfang_types::message::Role::User => "User",
                        openfang_types::message::Role::Assistant => "Assistant",
                        openfang_types::message::Role::System => "System",
                    };
                    let text = msg.content.text_content();
                    if !text.is_empty() {
                        // Truncate individual messages in summary to keep it compact (UTF-8 safe)
                        let truncated = if text.len() > 200 {
                            format!("{}...", openfang_types::truncate_str(&text, 200))
                        } else {
                            text
                        };
                        summary_parts.push(format!("{role}: {truncated}"));
                    }
                }
                // Keep summary under ~4000 chars (UTF-8 safe)
                let mut full_summary = summary_parts.join("\n");
                if full_summary.len() > 4000 {
                    let start = full_summary.len() - 4000;
                    // Find the next char boundary at or after `start`
                    let safe_start = (start..full_summary.len())
                        .find(|&i| full_summary.is_char_boundary(i))
                        .unwrap_or(full_summary.len());
                    full_summary = full_summary[safe_start..].to_string();
                }
                canonical.compacted_summary = Some(full_summary);
                canonical.compaction_cursor = to_compact;
                // Trim messages: keep only the recent window
                canonical.messages = canonical.messages.split_off(to_compact);
                canonical.compaction_cursor = 0; // reset cursor since we trimmed
            }
        }

        canonical.updated_at = Utc::now().to_rfc3339();
        self.save_canonical(&canonical)?;
        Ok(canonical)
    }

    /// Get recent messages from canonical session for context injection.
    ///
    /// Returns up to `window_size` recent messages (default 50), plus
    /// the compacted summary if available.
    pub fn canonical_context(
        &self,
        agent_id: AgentId,
        window_size: Option<usize>,
    ) -> OpenFangResult<(Option<String>, Vec<Message>)> {
        let canonical = self.load_canonical(agent_id)?;
        let window = window_size.unwrap_or(DEFAULT_CANONICAL_WINDOW);
        let start = canonical.messages.len().saturating_sub(window);
        let recent = canonical.messages[start..].to_vec();
        Ok((canonical.compacted_summary.clone(), recent))
    }

    /// Persist a canonical session to SQLite.
    fn save_canonical(&self, canonical: &CanonicalSession) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let messages_blob = rmp_serde::to_vec_named(&canonical.messages)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        conn.execute(
            "INSERT INTO canonical_sessions (agent_id, messages, compaction_cursor, compacted_summary, updated_at, prime_for)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(agent_id) DO UPDATE SET messages = ?2, compaction_cursor = ?3, compacted_summary = ?4, updated_at = ?5, prime_for = ?6",
            rusqlite::params![
                canonical.agent_id.0.to_string(),
                messages_blob,
                canonical.compaction_cursor as i64,
                canonical.compacted_summary,
                canonical.updated_at,
                canonical.prime_for,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }
}

/// A single JSONL line in the session mirror file.
#[derive(serde::Serialize)]
struct JsonlLine {
    timestamp: String,
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_use: Option<serde_json::Value>,
}

impl SessionStore {
    /// Write a human-readable JSONL mirror of a session to disk.
    ///
    /// Best-effort: errors are returned but should be logged and never
    /// affect the primary SQLite store.
    pub fn write_jsonl_mirror(
        &self,
        session: &Session,
        sessions_dir: &Path,
    ) -> Result<(), std::io::Error> {
        std::fs::create_dir_all(sessions_dir)?;
        let path = sessions_dir.join(format!("{}.jsonl", session.id.0));
        let mut file = std::fs::File::create(&path)?;
        let now = Utc::now().to_rfc3339();

        for msg in &session.messages {
            let role_str = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };

            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_parts: Vec<serde_json::Value> = Vec::new();

            match &msg.content {
                MessageContent::Text(t) => {
                    text_parts.push(t.clone());
                }
                MessageContent::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text, .. } => {
                                text_parts.push(text.clone());
                            }
                            ContentBlock::ToolUse {
                                id, name, input, ..
                            } => {
                                tool_parts.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": input,
                                }));
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                tool_name: _,
                                content,
                                is_error,
                            } => {
                                tool_parts.push(serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": content,
                                    "is_error": is_error,
                                }));
                            }
                            ContentBlock::Image { media_type, .. } => {
                                text_parts.push(format!("[image: {media_type}]"));
                            }
                            ContentBlock::Thinking { thinking, .. } => {
                                text_parts.push(format!(
                                    "[thinking: {}]",
                                    openfang_types::truncate_str(thinking, 200)
                                ));
                            }
                            ContentBlock::RedactedThinking { .. } => {
                                text_parts.push("[redacted_thinking]".to_string());
                            }
                            ContentBlock::Unknown => {}
                        }
                    }
                }
            }

            let line = JsonlLine {
                timestamp: now.clone(),
                role: role_str.to_string(),
                content: serde_json::Value::String(text_parts.join("\n")),
                tool_use: if tool_parts.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Array(tool_parts))
                },
            };

            serde_json::to_writer(&mut file, &line).map_err(std::io::Error::other)?;
            file.write_all(b"\n")?;
        }

        Ok(())
    }
}

/// A participant in a session — one human (or agent) actor keyed by a durable
/// speaker id (snowflake). Carries the presence clock and the identity label.
#[derive(Debug, Clone)]
pub struct Participant {
    /// Durable actor key (e.g. Discord snowflake).
    pub speaker_id: String,
    /// Human-readable display name (may drift; snapshot of last inbound).
    pub display_name: String,
    /// RFC3339 timestamp of this actor's most recent inbound message.
    pub last_msg_at: String,
    /// RFC3339 timestamp of this actor's first observed message in the session.
    pub first_seen_at: String,
    /// Count of inbound messages recorded for this actor in the session.
    pub message_count: i64,
}

impl SessionStore {
    /// Record a genuine human inbound from `speaker_id` in `session`, stamping
    /// presence at `now` (RFC3339). Returns the actor's PRIOR `last_msg_at`
    /// (before this stamp), or `None` on first contact.
    ///
    /// The returned prior is the anchor for `since_this_speaker = now - prior`
    /// in the turn-context envelope (ANAI-128). MUST be called only on real
    /// user turns (trigger == User) so autonomous/cron turns never reset an
    /// actor's clock. `now` is passed in so the caller controls the clock.
    pub fn record_participant(
        &self,
        session_id: SessionId,
        speaker_id: &str,
        display_name: &str,
        now: &str,
    ) -> OpenFangResult<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Read the prior stamp before we overwrite it (serialized under the lock).
        let prior: Option<String> = conn
            .query_row(
                "SELECT last_msg_at FROM session_participants \
                 WHERE session_id = ?1 AND speaker_id = ?2",
                rusqlite::params![session_id.0.to_string(), speaker_id],
                |row| row.get::<_, String>(0),
            )
            .ok();

        conn.execute(
            "INSERT INTO session_participants \
                 (session_id, speaker_id, display_name, first_seen_at, last_msg_at, message_count) \
             VALUES (?1, ?2, ?3, ?4, ?4, 1) \
             ON CONFLICT(session_id, speaker_id) DO UPDATE SET \
                 display_name = excluded.display_name, \
                 last_msg_at = excluded.last_msg_at, \
                 message_count = session_participants.message_count + 1",
            rusqlite::params![session_id.0.to_string(), speaker_id, display_name, now],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        Ok(prior)
    }

    /// Return the session's participants ordered by most-recent activity first,
    /// capped at `limit`. Substrate for the envelope's presence roster.
    pub fn session_roster(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> OpenFangResult<Vec<Participant>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT speaker_id, display_name, last_msg_at, first_seen_at, message_count \
                 FROM session_participants WHERE session_id = ?1 \
                 ORDER BY last_msg_at DESC LIMIT ?2",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map(
                rusqlite::params![session_id.0.to_string(), limit as i64],
                |row| {
                    Ok(Participant {
                        speaker_id: row.get(0)?,
                        display_name: row.get(1)?,
                        last_msg_at: row.get(2)?,
                        first_seen_at: row.get(3)?,
                        message_count: row.get(4)?,
                    })
                },
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(out)
    }
}

impl SessionStore {
    /// Resolve a durable speaker id (snowflake) to its AUTHORITATIVE display
    /// name from the curated `identity_bindings` table — rung 1 of the identity
    /// hierarchy (ANAI-127). Returns `None` when the operator has asserted no
    /// binding for this speaker, in which case the caller falls back to the
    /// platform's `global_name`, then the raw handle.
    ///
    /// This is display identity only, never an authz carrier. The binding is
    /// fleet-wide (not per-session): one operator mapping holds everywhere.
    pub fn resolve_identity(&self, speaker_id: &str) -> OpenFangResult<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let result = conn.query_row(
            "SELECT openfang_name FROM identity_bindings WHERE speaker_id = ?1",
            rusqlite::params![speaker_id],
            |row| row.get::<_, String>(0),
        );
        match result {
            Ok(name) => Ok(Some(name)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(OpenFangError::Memory(e.to_string())),
        }
    }

    /// Create or update the authoritative binding for `speaker_id`. `note` is an
    /// optional operator memo (e.g. "Ben's son"). Upserts on the snowflake so a
    /// re-bind cleanly overwrites the prior name.
    pub fn upsert_identity_binding(
        &self,
        speaker_id: &str,
        openfang_name: &str,
        note: Option<&str>,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO identity_bindings (speaker_id, openfang_name, note, updated_at) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(speaker_id) DO UPDATE SET \
                 openfang_name = excluded.openfang_name, \
                 note = excluded.note, \
                 updated_at = excluded.updated_at",
            rusqlite::params![speaker_id, openfang_name, note, Utc::now().to_rfc3339()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;

    fn setup() -> SessionStore {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        SessionStore::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn test_create_and_load_session() {
        let store = setup();
        let agent_id = AgentId::new();
        let session = store.create_session(agent_id).unwrap();

        let loaded = store.get_session(session.id).unwrap().unwrap();
        assert_eq!(loaded.agent_id, agent_id);
        assert!(loaded.messages.is_empty());
    }

    #[test]
    fn test_save_and_load_with_messages() {
        let store = setup();
        let agent_id = AgentId::new();
        let mut session = store.create_session(agent_id).unwrap();
        session.messages.push(Message::user("Hello"));
        session.messages.push(Message::assistant("Hi there!"));
        store.save_session(&session).unwrap();

        let loaded = store.get_session(session.id).unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 2);
    }

    #[test]
    fn test_get_missing_session() {
        let store = setup();
        let result = store.get_session(SessionId::new()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_session() {
        let store = setup();
        let agent_id = AgentId::new();
        let session = store.create_session(agent_id).unwrap();
        let sid = session.id;
        assert!(store.get_session(sid).unwrap().is_some());
        store.delete_session(sid).unwrap();
        assert!(store.get_session(sid).unwrap().is_none());
    }

    #[test]
    fn test_delete_agent_sessions() {
        let store = setup();
        let agent_id = AgentId::new();
        let s1 = store.create_session(agent_id).unwrap();
        let s2 = store.create_session(agent_id).unwrap();
        assert!(store.get_session(s1.id).unwrap().is_some());
        assert!(store.get_session(s2.id).unwrap().is_some());
        store.delete_agent_sessions(agent_id).unwrap();
        assert!(store.get_session(s1.id).unwrap().is_none());
        assert!(store.get_session(s2.id).unwrap().is_none());
    }

    #[test]
    fn test_canonical_load_creates_empty() {
        let store = setup();
        let agent_id = AgentId::new();
        let canonical = store.load_canonical(agent_id).unwrap();
        assert_eq!(canonical.agent_id, agent_id);
        assert!(canonical.messages.is_empty());
        assert!(canonical.compacted_summary.is_none());
        assert_eq!(canonical.compaction_cursor, 0);
    }

    #[test]
    fn test_canonical_append_and_load() {
        let store = setup();
        let agent_id = AgentId::new();

        // Append from "Telegram"
        let msgs1 = vec![
            Message::user("Hello from Telegram"),
            Message::assistant("Hi! I'm your agent."),
        ];
        store.append_canonical(agent_id, &msgs1, None).unwrap();

        // Append from "Discord"
        let msgs2 = vec![
            Message::user("Now I'm on Discord"),
            Message::assistant("I remember you from Telegram!"),
        ];
        let canonical = store.append_canonical(agent_id, &msgs2, None).unwrap();

        // Should have all 4 messages
        assert_eq!(canonical.messages.len(), 4);
    }

    #[test]
    fn test_canonical_context_window() {
        let store = setup();
        let agent_id = AgentId::new();

        // Add 10 messages
        let msgs: Vec<Message> = (0..10)
            .map(|i| Message::user(format!("Message {i}")))
            .collect();
        store.append_canonical(agent_id, &msgs, None).unwrap();

        // Request window of 3
        let (summary, recent) = store.canonical_context(agent_id, Some(3)).unwrap();
        assert_eq!(recent.len(), 3);
        assert!(summary.is_none()); // No compaction yet
    }

    #[test]
    fn test_canonical_compaction() {
        let store = setup();
        let agent_id = AgentId::new();

        // Add 120 messages (over the default 100 threshold)
        let msgs: Vec<Message> = (0..120)
            .map(|i| Message::user(format!("Message number {i} with some content")))
            .collect();
        let canonical = store.append_canonical(agent_id, &msgs, Some(100)).unwrap();

        // After compaction: should keep DEFAULT_CANONICAL_WINDOW (50) messages
        assert!(canonical.messages.len() <= 60); // some tolerance
        assert!(canonical.compacted_summary.is_some());
    }

    #[test]
    fn test_canonical_cross_channel_roundtrip() {
        let store = setup();
        let agent_id = AgentId::new();

        // Channel 1: user tells agent their name
        store
            .append_canonical(
                agent_id,
                &[
                    Message::user("My name is Jaber"),
                    Message::assistant("Nice to meet you, Jaber!"),
                ],
                None,
            )
            .unwrap();

        // Channel 2: different channel queries same agent
        let (summary, recent) = store.canonical_context(agent_id, None).unwrap();
        // The agent should have context about "Jaber" from the previous channel
        let all_text: String = recent.iter().map(|m| m.content.text_content()).collect();
        assert!(all_text.contains("Jaber"));
        assert!(summary.is_none()); // Only 2 messages, no compaction
    }

    /// ANAI-246: the re-anchor drops the verbatim messages and KEEPS the
    /// compacted summary. If this ever inverts, every episode close becomes
    /// the amnesia `delete_canonical_session` would have caused.
    #[test]
    fn test_reanchor_canonical_keeps_the_summary_and_drops_the_messages() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .store_llm_summary(
                agent_id,
                "earlier: we retired the octopus",
                vec![Message::user("still here"), Message::assistant("indeed")],
            )
            .unwrap();

        let dropped = store.reanchor_canonical(agent_id, None).unwrap();
        assert_eq!(dropped, 2);

        let canonical = store.load_canonical(agent_id).unwrap();
        assert!(canonical.messages.is_empty(), "verbatim history is dropped");
        assert_eq!(
            canonical.compacted_summary.as_deref(),
            Some("earlier: we retired the octopus"),
            "the summary is the agent's memory of everything before the boundary"
        );
        assert_eq!(canonical.compaction_cursor, 0);

        // And it still reaches the prompt path.
        let (summary, recent) = store.canonical_context(agent_id, None).unwrap();
        assert_eq!(summary.as_deref(), Some("earlier: we retired the octopus"));
        assert!(recent.is_empty());
    }

    /// A second close with nothing to drop is a no-op, not a write.
    #[test]
    fn test_reanchor_canonical_is_idempotent() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .append_canonical(agent_id, &[Message::user("one")], None)
            .unwrap();
        assert_eq!(store.reanchor_canonical(agent_id, None).unwrap(), 1);
        assert_eq!(store.reanchor_canonical(agent_id, None).unwrap(), 0);
    }

    /// ANAI-247: the slug is persisted by the re-anchor and survives a
    /// reload. It has to be durable, not in-process: the amnesia event this
    /// pack exists to soften is most often a daemon restart.
    #[test]
    fn test_reanchor_canonical_records_and_clears_prime_for() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .append_canonical(agent_id, &[Message::user("one")], None)
            .unwrap();

        store
            .reanchor_canonical(agent_id, Some("openfang-fork"))
            .unwrap();
        assert_eq!(
            store.load_canonical(agent_id).unwrap().prime_for.as_deref(),
            Some("openfang-fork")
        );

        // A later boundary drawn without a slug must CLEAR the old one. A
        // stale pack is worse than no pack: it briefs the agent on work it
        // has already moved off.
        store.reanchor_canonical(agent_id, None).unwrap();
        assert_eq!(store.load_canonical(agent_id).unwrap().prime_for, None);
    }

    /// The idempotence shortcut must not swallow a priming. A close that
    /// drops nothing — an agent whose window is already empty — is exactly
    /// the case where the pack is the only context the next turn will get.
    #[test]
    fn test_a_reanchor_that_drops_nothing_still_primes() {
        let store = setup();
        let agent_id = AgentId::new();

        assert_eq!(
            store
                .reanchor_canonical(agent_id, Some("openfang-fork"))
                .unwrap(),
            0,
            "nothing verbatim to drop"
        );
        assert_eq!(
            store.load_canonical(agent_id).unwrap().prime_for.as_deref(),
            Some("openfang-fork"),
            "the priming is the whole point of this call; it must still be written"
        );
    }

    /// `prime_for` must round-trip through the paths that rewrite canonical
    /// for other reasons, or the pack would evaporate on the first compaction
    /// of the primed episode.
    #[test]
    fn test_prime_for_survives_append_and_compaction() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .reanchor_canonical(agent_id, Some("openfang-fork"))
            .unwrap();

        store
            .append_canonical(agent_id, &[Message::user("back to work")], None)
            .unwrap();
        assert_eq!(
            store.load_canonical(agent_id).unwrap().prime_for.as_deref(),
            Some("openfang-fork")
        );

        store
            .store_llm_summary(agent_id, "we did a thing", vec![Message::user("tail")])
            .unwrap();
        assert_eq!(
            store.load_canonical(agent_id).unwrap().prime_for.as_deref(),
            Some("openfang-fork"),
            "compaction rewrites canonical; it must not drop the priming"
        );
    }

    #[test]
    fn test_jsonl_mirror_write() {
        let store = setup();
        let agent_id = AgentId::new();
        let mut session = store.create_session(agent_id).unwrap();
        session
            .messages
            .push(openfang_types::message::Message::user("Hello"));
        session
            .messages
            .push(openfang_types::message::Message::assistant("Hi there!"));
        store.save_session(&session).unwrap();

        let dir = tempfile::TempDir::new().unwrap();
        let sessions_dir = dir.path().join("sessions");
        store.write_jsonl_mirror(&session, &sessions_dir).unwrap();

        let jsonl_path = sessions_dir.join(format!("{}.jsonl", session.id.0));
        assert!(jsonl_path.exists());

        let content = std::fs::read_to_string(&jsonl_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);

        // Verify first line is user message
        let line1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(line1["role"], "user");
        assert_eq!(line1["content"], "Hello");

        // Verify second line is assistant message
        let line2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(line2["role"], "assistant");
        assert_eq!(line2["content"], "Hi there!");
        assert!(line2.get("tool_use").is_none());
    }

    #[test]
    fn test_record_participant_returns_prior() {
        let store = setup();
        let agent_id = AgentId::new();
        let session = store.create_session(agent_id).unwrap();

        // First contact: no prior.
        let prior = store
            .record_participant(session.id, "snow_alice", "Alice", "2026-07-20T10:00:00Z")
            .unwrap();
        assert!(prior.is_none());

        // Second message from Alice: prior is her previous stamp.
        let prior = store
            .record_participant(session.id, "snow_alice", "Alice", "2026-07-20T10:05:00Z")
            .unwrap();
        assert_eq!(prior.as_deref(), Some("2026-07-20T10:00:00Z"));
    }

    #[test]
    fn test_record_participant_per_actor_clock() {
        let store = setup();
        let agent_id = AgentId::new();
        let session = store.create_session(agent_id).unwrap();

        // Alice yesterday, Bob today. Bob's gap must key off Bob, not Alice.
        store
            .record_participant(session.id, "snow_alice", "Alice", "2026-07-19T10:00:00Z")
            .unwrap();
        let bob_first = store
            .record_participant(session.id, "snow_bob", "Bob", "2026-07-20T10:05:00Z")
            .unwrap();
        assert!(bob_first.is_none(), "Bob's first message has no prior");

        // Bob again — prior is Bob's own last stamp, not Alice's.
        let bob_prior = store
            .record_participant(session.id, "snow_bob", "Bob", "2026-07-20T10:06:00Z")
            .unwrap();
        assert_eq!(bob_prior.as_deref(), Some("2026-07-20T10:05:00Z"));
    }

    #[test]
    fn test_session_roster_recency_order() {
        let store = setup();
        let agent_id = AgentId::new();
        let session = store.create_session(agent_id).unwrap();

        store
            .record_participant(session.id, "snow_carol", "Carol", "2026-07-16T10:00:00Z")
            .unwrap();
        store
            .record_participant(session.id, "snow_alice", "Alice", "2026-07-20T09:58:00Z")
            .unwrap();
        store
            .record_participant(session.id, "snow_bob", "Bob", "2026-07-19T10:05:00Z")
            .unwrap();

        let roster = store.session_roster(session.id, 10).unwrap();
        assert_eq!(roster.len(), 3);
        // Most recent first: Alice, Bob, Carol.
        assert_eq!(roster[0].speaker_id, "snow_alice");
        assert_eq!(roster[1].speaker_id, "snow_bob");
        assert_eq!(roster[2].speaker_id, "snow_carol");

        // Limit is honored.
        let capped = store.session_roster(session.id, 2).unwrap();
        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0].speaker_id, "snow_alice");
    }

    #[test]
    fn test_session_updated_at() {
        let store = setup();
        let agent_id = AgentId::new();
        let session = store.create_session(agent_id).unwrap();
        // A freshly created session has an updated_at stamp.
        let ts = store.session_updated_at(session.id).unwrap();
        assert!(ts.is_some(), "created session should have updated_at");
        // Missing session yields None, not an error.
        let missing = store.session_updated_at(SessionId::new()).unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_identity_binding_resolves_and_overrides() {
        let store = setup();
        // Unbound snowflake resolves to None (caller falls back to global_name).
        assert!(store.resolve_identity("snow_teo").unwrap().is_none());

        // Operator asserts the binding.
        store
            .upsert_identity_binding("snow_teo", "Teo", Some("Ben's son"))
            .unwrap();
        assert_eq!(
            store.resolve_identity("snow_teo").unwrap().as_deref(),
            Some("Teo")
        );

        // Re-bind overwrites cleanly (a kid's global_name can never beat this).
        store
            .upsert_identity_binding("snow_teo", "Teodoro", None)
            .unwrap();
        assert_eq!(
            store.resolve_identity("snow_teo").unwrap().as_deref(),
            Some("Teodoro")
        );
    }
}
