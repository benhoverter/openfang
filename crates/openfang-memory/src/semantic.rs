//! Semantic memory store with vector embedding support.
//!
//! Phase 1: SQLite LIKE matching (fallback when no embeddings).
//! Phase 2: Vector cosine similarity search using stored embeddings.
//!
//! Embeddings are stored as BLOBs in the `embedding` column of the memories table.
//! When a query embedding is provided, recall uses cosine similarity ranking.
//! When no embeddings are available, falls back to LIKE matching.

use chrono::Utc;
use openfang_types::agent::AgentId;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::memory::{MemoryFilter, MemoryFragment, MemoryId, MemorySource};
use rusqlite::Connection;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, warn};

#[cfg(feature = "http-memory")]
use crate::http_client::MemoryApiClient;

/// Metadata key carrying the row-type discriminator, promoted to a real column
/// in schema v13.
///
/// Canonical home for the string so the kernel's tool surface, the capture
/// path, and this store cannot drift on spelling. The vocabulary is a CLOSED
/// set — `turn`, `note`, `store`, `summary`, and `fact` (reserved for stage 3)
/// — and variation belongs in sibling keys, not in new `kind` values:
/// supersession keys off `kind` + claim-key, and a fuzzy vocabulary poisons
/// both the filter and the chain.
///
/// Not enforced here. The store lifts whatever well-formed string it is given,
/// because a storage layer that silently refused a value would make a
/// `kind = ?` filter report "no rows" for data that exists — a wrong answer is
/// worse than an unfashionable one. Vocabulary is enforced at the writers.
pub const KIND_KEY: &str = "kind";

/// Semantic store backed by SQLite with optional vector search.
///
/// Supports two backends:
/// - **SQLite** (default): Local LIKE matching / cosine similarity.
/// - **HTTP**: Routes `remember`/`recall` to the memory-api gateway
///   (PostgreSQL + pgvector + Jina AI embeddings).
#[derive(Clone)]
pub struct SemanticStore {
    conn: Arc<Mutex<Connection>>,
    #[cfg(feature = "http-memory")]
    http_client: Option<MemoryApiClient>,
}

impl SemanticStore {
    /// Create a new semantic store wrapping the given connection (SQLite backend).
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self {
            conn,
            #[cfg(feature = "http-memory")]
            http_client: None,
        }
    }

    /// Create a semantic store with an HTTP backend for the memory-api gateway.
    ///
    /// The SQLite connection is still required for local fallback and other stores.
    #[cfg(feature = "http-memory")]
    pub fn new_with_http(conn: Arc<Mutex<Connection>>, client: MemoryApiClient) -> Self {
        Self {
            conn,
            http_client: Some(client),
        }
    }

    /// Store a new memory fragment (without embedding).
    pub fn remember(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> OpenFangResult<MemoryId> {
        self.remember_with_embedding(agent_id, content, source, scope, metadata, None)
    }

    /// Store a new memory fragment with an optional embedding vector.
    ///
    /// When HTTP backend is configured, stores via memory-api (which handles
    /// embedding generation and deduplication). Falls back to local SQLite.
    pub fn remember_with_embedding(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        // HTTP backend: route to memory-api
        #[cfg(feature = "http-memory")]
        if let Some(ref client) = self.http_client {
            return self.remember_via_http(client, agent_id, content, source, scope, &metadata);
        }

        // SQLite backend (default)
        self.remember_sqlite(agent_id, content, source, scope, metadata, embedding)
    }

    /// SQLite implementation of remember_with_embedding.
    fn remember_sqlite(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let id = MemoryId::new();
        let now = Utc::now().to_rfc3339();
        let source_str = serde_json::to_string(&source)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        // The column is the store of record for `episode_id`; the metadata JSON
        // is a projection hydrated back on read (see `recall_with_embedding`).
        // Lift the key out before serializing so the fact is written once and
        // cannot drift, and so an episode filter can hit `idx_memories_episode`
        // instead of scanning `json_extract` over the whole corpus.
        //
        // Only a well-formed (string, non-empty) value is lifted. A malformed
        // one stays in the JSON untouched rather than being silently dropped:
        // it is not an episode id, so the column must not claim it, but
        // destroying a caller's data on the way past would be worse.
        let mut metadata = metadata;
        let episode_id = episode_id_from_metadata(&metadata);
        if episode_id.is_some() {
            metadata.remove(crate::episode::EPISODE_ID_KEY);
        }
        // Same inversion for `kind` (schema v13): the column is the store of
        // record, the metadata JSON is a projection hydrated back on read.
        let kind = kind_from_metadata(&metadata);
        if kind.is_some() {
            metadata.remove(KIND_KEY);
        }
        let meta_str = serde_json::to_string(&metadata)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let embedding_bytes: Option<Vec<u8>> = embedding.map(embedding_to_bytes);

        conn.execute(
            "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, embedding, episode_id, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, 1.0, ?6, ?7, ?7, 0, 0, ?8, ?9, ?10)",
            rusqlite::params![
                id.0.to_string(),
                agent_id.0.to_string(),
                content,
                source_str,
                scope,
                meta_str,
                now,
                embedding_bytes,
                episode_id,
                kind,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(id)
    }

    /// HTTP implementation of remember — routes to memory-api POST /memory/store.
    #[cfg(feature = "http-memory")]
    fn remember_via_http(
        &self,
        client: &MemoryApiClient,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> OpenFangResult<MemoryId> {
        let source_str = format!("{:?}", source).to_lowercase();
        let importance = metadata
            .get("importance")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(10) as u8)
            .unwrap_or(5);
        let tags: Option<Vec<String>> = metadata
            .get("tags")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        match client.store(
            content,
            Some(scope),
            Some(&agent_id.0.to_string()),
            Some(&source_str),
            Some(importance),
            tags,
        ) {
            Ok(resp) => {
                debug!(id = %resp.id, "Stored memory via HTTP backend");
                Ok(MemoryId::new())
            }
            Err(e) => {
                warn!(error = %e, "HTTP memory store failed, falling back to SQLite");
                self.remember_sqlite(agent_id, content, source, scope, metadata.clone(), None)
            }
        }
    }

    /// Search for memories using text matching (fallback, no embeddings).
    pub fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        self.recall_with_embedding(query, limit, filter, None)
    }

    /// Search for memories using vector similarity when a query embedding is provided,
    /// falling back to LIKE matching otherwise.
    ///
    /// When HTTP backend is configured, searches via memory-api (hybrid vector+BM25).
    /// Falls back to local SQLite on HTTP errors.
    pub fn recall_with_embedding(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        // HTTP backend: route to memory-api
        #[cfg(feature = "http-memory")]
        if let Some(ref client) = self.http_client {
            match self.recall_via_http(client, query, limit, &filter) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    warn!(error = %e, "HTTP memory search failed, falling back to SQLite");
                }
            }
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Build SQL: fetch candidates (broader than limit for vector re-ranking)
        let fetch_limit = if query_embedding.is_some() {
            // Candidate window for in-process vector re-ranking. Floor raised from 100 to 5000 [ANAI-60] so large agent corpora (e.g. orchestrator ~1k) are fully re-ranked, not just the 100 most-recently-accessed rows.
            (limit * 10).max(5000)
        } else {
            limit
        };

        let mut sql = String::from(
            "SELECT id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, embedding, episode_id, kind
             FROM memories WHERE deleted = 0",
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        // Text search filter (only when no embeddings — vector search handles relevance)
        if query_embedding.is_none() && !query.is_empty() {
            sql.push_str(&format!(" AND content LIKE ?{param_idx}"));
            params.push(Box::new(format!("%{query}%")));
            param_idx += 1;
        }

        // Apply filters
        if let Some(ref f) = filter {
            if let Some(agent_id) = f.agent_id {
                sql.push_str(&format!(" AND agent_id = ?{param_idx}"));
                params.push(Box::new(agent_id.0.to_string()));
                param_idx += 1;
            }
            if let Some(ref scope) = f.scope {
                sql.push_str(&format!(" AND scope = ?{param_idx}"));
                params.push(Box::new(scope.clone()));
                param_idx += 1;
            }
            if let Some(min_conf) = f.min_confidence {
                sql.push_str(&format!(" AND confidence >= ?{param_idx}"));
                params.push(Box::new(min_conf as f64));
                param_idx += 1;
            }
            if let Some(ref source) = f.source {
                let source_str = serde_json::to_string(source)
                    .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
                sql.push_str(&format!(" AND source = ?{param_idx}"));
                params.push(Box::new(source_str));
                param_idx += 1;
            }

            // ANAI-166: honour `MemoryFilter::metadata`.
            //
            // This field has existed on the filter since the type was written
            // and was silently ignored here — a filter that quietly matches
            // everything is worse than one that does not exist, because the
            // caller reads the result as "nothing else matched". No caller
            // populated it before now, so implementing it changes no existing
            // behaviour.
            //
            // Applied in SQL rather than post-filtered in the caller on
            // purpose: the vector path truncates to `limit` AFTER re-ranking,
            // so a caller-side filter would first discard candidates and then
            // return fewer than `limit` rows that did match.
            //
            // Keys are sorted so the generated SQL is deterministic (a
            // HashMap's iteration order is not), which keeps the statement
            // cacheable and the tests stable.
            let mut meta_keys: Vec<&String> = f.metadata.keys().collect();
            meta_keys.sort();
            for key in meta_keys {
                // The key lands inside a JSON path literal, which cannot be
                // parameterised. Restrict it to an identifier charset rather
                // than trusting callers: this is the one place in this query
                // builder where a string is interpolated instead of bound.
                if key.is_empty()
                    || !key
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    return Err(OpenFangError::Memory(format!(
                        "Invalid metadata filter key '{key}': only ASCII alphanumerics, \
                         '_' and '-' are allowed"
                    )));
                }
                // json_extract returns SQL NULL for an absent key, so a row
                // missing the key never matches — which is what "filter by
                // kind" has to mean for the 46k pre-`kind` rows.
                if key == crate::episode::EPISODE_ID_KEY {
                    // `episode_id` is promoted to a real column, so filter on
                    // the column and use `idx_memories_episode` rather than
                    // running `json_extract` over every undeleted row. This is
                    // the reason the column exists (ADR 0001 §2.2); until now
                    // it was written and never read.
                    //
                    // Pre-v12 rows have neither the column nor the key, so
                    // they are excluded either way — the semantics are
                    // unchanged, only the plan is.
                    sql.push_str(&format!(" AND episode_id = ?{param_idx}"));
                } else if key == KIND_KEY {
                    // Same promotion (v13): filter the column, hit
                    // `idx_memories_kind`, do not `json_extract` over the
                    // corpus. `kind` is the discriminator stage 3 filters on
                    // constantly, so this is the hot path, not a micro-opt.
                    //
                    // Unlike `episode_id`, rows written BEFORE the promotion
                    // do carry `kind` in their JSON, so a column-only filter
                    // would silently lose them. That is why `migrate_v13`
                    // lifts the existing key into the column for every such
                    // row — a copy of a fact already present, not a guess —
                    // leaving no population that has the key but not the
                    // column. Without that lift this branch would be wrong.
                    sql.push_str(&format!(" AND kind = ?{param_idx}"));
                } else {
                    sql.push_str(&format!(
                        " AND json_extract(metadata, '$.{key}') = ?{param_idx}"
                    ));
                }
                // Compare as text for strings; JSON-encode anything else so a
                // number or bool round-trips instead of arriving quoted.
                let bound = match &f.metadata[key] {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                params.push(Box::new(bound));
                param_idx += 1;
            }
            let _ = param_idx;
        }

        sql.push_str(" ORDER BY accessed_at DESC, access_count DESC");
        sql.push_str(&format!(" LIMIT {fetch_limit}"));

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let id_str: String = row.get(0)?;
                let agent_str: String = row.get(1)?;
                let content: String = row.get(2)?;
                let source_str: String = row.get(3)?;
                let scope: String = row.get(4)?;
                let confidence: f64 = row.get(5)?;
                let meta_str: String = row.get(6)?;
                let created_str: String = row.get(7)?;
                let accessed_str: String = row.get(8)?;
                let access_count: i64 = row.get(9)?;
                let embedding_bytes: Option<Vec<u8>> = row.get(10)?;
                let episode_id: Option<String> = row.get(11)?;
                let kind: Option<String> = row.get(12)?;
                Ok((
                    id_str,
                    agent_str,
                    content,
                    source_str,
                    scope,
                    confidence,
                    meta_str,
                    created_str,
                    accessed_str,
                    access_count,
                    embedding_bytes,
                    episode_id,
                    kind,
                ))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut fragments = Vec::new();
        for row_result in rows {
            let (
                id_str,
                agent_str,
                content,
                source_str,
                scope,
                confidence,
                meta_str,
                created_str,
                accessed_str,
                access_count,
                embedding_bytes,
                episode_id,
                kind,
            ) = row_result.map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let id = uuid::Uuid::parse_str(&id_str)
                .map(MemoryId)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let agent_id = uuid::Uuid::parse_str(&agent_str)
                .map(openfang_types::agent::AgentId)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let source: MemorySource =
                serde_json::from_str(&source_str).unwrap_or(MemorySource::System);
            let mut metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(&meta_str).unwrap_or_default();
            // Hydrate the column back into the metadata map so every consumer
            // (kernel's recall payload, consolidation, tests) keeps reading
            // `episode_id` where it always has. `or_insert` rather than an
            // overwrite: rows written before the inversion still carry the key
            // in their JSON, and the JSON is the value the column was derived
            // from, so they agree by construction — this just avoids a
            // pointless clone-and-replace on that population.
            if let Some(ep) = episode_id {
                metadata
                    .entry(crate::episode::EPISODE_ID_KEY.to_string())
                    .or_insert(serde_json::Value::String(ep));
            }
            // Same hydration for `kind` (v13). `or_insert` for the same
            // reason: pre-v13 rows still carry the key in their JSON and the
            // column is NULL for them, so the JSON is the only copy; rows
            // written after the inversion have only the column.
            if let Some(k) = kind {
                metadata
                    .entry(KIND_KEY.to_string())
                    .or_insert(serde_json::Value::String(k));
            }
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let accessed_at = chrono::DateTime::parse_from_rfc3339(&accessed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let embedding = embedding_bytes.as_deref().map(embedding_from_bytes);

            fragments.push(MemoryFragment {
                id,
                agent_id,
                content,
                embedding,
                metadata,
                source,
                confidence: confidence as f32,
                created_at,
                accessed_at,
                access_count: access_count as u64,
                scope,
            });
        }

        // If we have a query embedding, re-rank by cosine similarity
        if let Some(qe) = query_embedding {
            fragments.sort_by(|a, b| {
                let sim_a = a
                    .embedding
                    .as_deref()
                    .map(|e| cosine_similarity(qe, e))
                    .unwrap_or(-1.0);
                let sim_b = b
                    .embedding
                    .as_deref()
                    .map(|e| cosine_similarity(qe, e))
                    .unwrap_or(-1.0);
                sim_b
                    .partial_cmp(&sim_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            fragments.truncate(limit);
            debug!(
                "Vector recall: {} results from {} candidates",
                fragments.len(),
                fetch_limit
            );
        }

        // Update access counts for returned memories
        for frag in &fragments {
            let _ = conn.execute(
                "UPDATE memories SET access_count = access_count + 1, accessed_at = ?1 WHERE id = ?2",
                rusqlite::params![Utc::now().to_rfc3339(), frag.id.0.to_string()],
            );
        }

        Ok(fragments)
    }

    /// Soft-delete a memory fragment.
    ///
    /// In HTTP mode, logs a warning (memory-api doesn't support delete yet)
    /// and performs the soft-delete locally only.
    pub fn forget(&self, id: MemoryId) -> OpenFangResult<()> {
        #[cfg(feature = "http-memory")]
        if self.http_client.is_some() {
            warn!(id = %id.0, "forget() not supported via HTTP backend, local-only soft-delete");
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE memories SET deleted = 1 WHERE id = ?1",
            rusqlite::params![id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Update the embedding for an existing memory.
    pub fn update_embedding(&self, id: MemoryId, embedding: &[f32]) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let bytes = embedding_to_bytes(embedding);
        conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            rusqlite::params![bytes, id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// HTTP implementation of recall — routes to memory-api POST /memory/search.
    ///
    /// Maps memory-api search results to `MemoryFragment` structs. Fields not
    /// available from the HTTP API (agent_id, embedding, access_count) use defaults.
    #[cfg(feature = "http-memory")]
    fn recall_via_http(
        &self,
        client: &MemoryApiClient,
        query: &str,
        limit: usize,
        filter: &Option<MemoryFilter>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        let category = filter.as_ref().and_then(|f| f.scope.as_deref());

        let results = client
            .search(query, limit, category)
            .map_err(|e| OpenFangError::Memory(format!("HTTP search failed: {e}")))?;

        let fragments: Vec<MemoryFragment> = results
            .into_iter()
            .map(|r| {
                let created_at = r
                    .created_at
                    .map(|ms| {
                        chrono::DateTime::from_timestamp_millis(ms as i64).unwrap_or_else(Utc::now)
                    })
                    .unwrap_or_else(Utc::now);

                MemoryFragment {
                    id: MemoryId::new(),
                    agent_id: filter.as_ref().and_then(|f| f.agent_id).unwrap_or_default(),
                    content: r.content,
                    embedding: None,
                    metadata: HashMap::new(),
                    source: MemorySource::System,
                    confidence: r.score as f32,
                    created_at,
                    accessed_at: Utc::now(),
                    access_count: 0,
                    scope: r.category.unwrap_or_else(|| "general".to_string()),
                }
            })
            .collect();

        debug!(
            count = fragments.len(),
            "Recalled memories via HTTP backend"
        );
        Ok(fragments)
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Lift the episode id out of the capture metadata into its own column
/// (ADR 0001 §2.2).
///
/// The agent loop already threads `episode_id` through the metadata map, so
/// lifting it here keeps every `remember*` signature — and its four call sites
/// across two loops, the HTTP path, and the `Memory` trait — untouched, while
/// still giving consolidation a real indexed column to group by instead of a
/// `json_extract` over 35k rows.
///
/// Anything that is not a string is treated as absent rather than coerced: a
/// malformed episode id must degrade to "legacy row", never to a fabricated
/// grouping key.
fn episode_id_from_metadata(metadata: &HashMap<String, serde_json::Value>) -> Option<String> {
    metadata
        .get(crate::episode::EPISODE_ID_KEY)?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Lift the row-type discriminator out of the capture metadata into its own
/// column (schema v13). Sibling of [`episode_id_from_metadata`], same contract.
///
/// Anything that is not a non-empty string is treated as absent and left in the
/// JSON untouched: the column must not claim a value that is not a kind, and
/// silently destroying a caller's key on the way past would be worse.
fn kind_from_metadata(metadata: &HashMap<String, serde_json::Value>) -> Option<String> {
    metadata
        .get(KIND_KEY)?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Serialize embedding to bytes for SQLite BLOB storage.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize embedding from bytes.
fn embedding_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;

    fn setup() -> SemanticStore {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        SemanticStore::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn test_remember_and_recall() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .remember(
                agent_id,
                "The user likes Rust programming",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        let results = store.recall("Rust", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn test_recall_with_filter() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .remember(
                agent_id,
                "Memory A",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        store
            .remember(
                AgentId::new(),
                "Memory B",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        let filter = MemoryFilter::agent(agent_id);
        let results = store.recall("Memory", 10, Some(filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Memory A");
    }

    /// ANAI-166: `MemoryFilter::metadata` used to be accepted and silently
    /// ignored, so a `kind`-filtered search returned everything and the caller
    /// read it as "these are all the notes."
    #[test]
    fn metadata_filter_selects_only_matching_rows() {
        let store = setup();
        let agent_id = AgentId::new();

        let mut note_meta = HashMap::new();
        note_meta.insert("kind".to_string(), serde_json::json!("note"));
        store
            .remember(
                agent_id,
                "a deliberate note",
                MemorySource::Observation,
                "episodic",
                note_meta,
            )
            .unwrap();

        let mut turn_meta = HashMap::new();
        turn_meta.insert("kind".to_string(), serde_json::json!("turn"));
        store
            .remember(
                agent_id,
                "a captured turn",
                MemorySource::Conversation,
                "episodic",
                turn_meta,
            )
            .unwrap();

        // The 46k pre-`kind` rows: no discriminator at all.
        store
            .remember(
                agent_id,
                "a legacy row",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        let filter = MemoryFilter {
            agent_id: Some(agent_id),
            metadata: HashMap::from([("kind".to_string(), serde_json::json!("note"))]),
            ..Default::default()
        };
        let results = store.recall("", 10, Some(filter)).unwrap();
        assert_eq!(results.len(), 1, "got: {results:?}");
        assert_eq!(results[0].content, "a deliberate note");
    }

    /// A row missing the key must not match — `json_extract` returns SQL NULL
    /// and NULL never equals a bound value. Asserted explicitly because the
    /// alternative (legacy rows matching every filter) would make `kind`
    /// filtering useless on the corpus that actually exists.
    #[test]
    fn metadata_filter_excludes_rows_without_the_key() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .remember(
                agent_id,
                "a legacy row",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        let filter = MemoryFilter {
            agent_id: Some(agent_id),
            metadata: HashMap::from([("kind".to_string(), serde_json::json!("note"))]),
            ..Default::default()
        };
        assert!(store.recall("", 10, Some(filter)).unwrap().is_empty());
    }

    fn episode_meta(episode: &str) -> HashMap<String, serde_json::Value> {
        HashMap::from([(
            crate::episode::EPISODE_ID_KEY.to_string(),
            serde_json::json!(episode),
        )])
    }

    /// The column is the store of record: `episode_id` is written there and
    /// nowhere else, so the two can never disagree. Asserted at the storage
    /// layer rather than through `recall`, because `recall` hydrates the key
    /// back and would hide a double-write.
    #[test]
    fn episode_id_is_written_to_the_column_and_not_mirrored_in_metadata() {
        let store = setup();
        store
            .remember(
                AgentId::new(),
                "a captured turn",
                MemorySource::Conversation,
                "episodic",
                episode_meta("ep-1"),
            )
            .unwrap();

        let conn = store.conn.lock().unwrap();
        let (column, in_json): (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT episode_id, json_extract(metadata, '$.episode_id') FROM memories",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(column.as_deref(), Some("ep-1"));
        assert_eq!(
            in_json, None,
            "episode_id must not be mirrored into the metadata JSON"
        );
    }

    /// Consumers read `episode_id` out of `MemoryFragment::metadata` (the
    /// kernel's recall payload does exactly this). Moving the fact to a column
    /// is only safe because the read path puts it back.
    #[test]
    fn recall_hydrates_episode_id_back_into_metadata() {
        let store = setup();
        store
            .remember(
                AgentId::new(),
                "a captured turn",
                MemorySource::Conversation,
                "episodic",
                episode_meta("ep-1"),
            )
            .unwrap();

        let results = store.recall("captured", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].metadata.get(crate::episode::EPISODE_ID_KEY),
            Some(&serde_json::json!("ep-1"))
        );
    }

    /// Filtering by episode now compiles to `episode_id = ?` against the
    /// index. The observable contract is unchanged, which is the point: this
    /// test fails if the special case selects the wrong rows.
    #[test]
    fn episode_filter_selects_only_that_episode() {
        let store = setup();
        let agent_id = AgentId::new();
        for (content, episode) in [("turn one", "ep-1"), ("turn two", "ep-2")] {
            store
                .remember(
                    agent_id,
                    content,
                    MemorySource::Conversation,
                    "episodic",
                    episode_meta(episode),
                )
                .unwrap();
        }
        // A pre-v12 row: no column, no key. Must not match any episode.
        store
            .remember(
                agent_id,
                "turn zero",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        let filter = MemoryFilter {
            agent_id: Some(agent_id),
            metadata: HashMap::from([(
                crate::episode::EPISODE_ID_KEY.to_string(),
                serde_json::json!("ep-1"),
            )]),
            ..Default::default()
        };
        let results = store.recall("", 10, Some(filter)).unwrap();
        assert_eq!(results.len(), 1, "got: {results:?}");
        assert_eq!(results[0].content, "turn one");
    }

    /// A non-string episode id is not an episode id. It must not be promoted
    /// to the column (that would fabricate a grouping key) and it must not be
    /// silently discarded either — it stays in the JSON as the caller wrote it.
    #[test]
    fn a_malformed_episode_id_is_left_in_metadata_and_leaves_the_column_null() {
        let store = setup();
        store
            .remember(
                AgentId::new(),
                "a malformed turn",
                MemorySource::Conversation,
                "episodic",
                HashMap::from([(
                    crate::episode::EPISODE_ID_KEY.to_string(),
                    serde_json::json!(42),
                )]),
            )
            .unwrap();

        let conn = store.conn.lock().unwrap();
        let (column, in_json): (Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT episode_id, json_extract(metadata, '$.episode_id') FROM memories",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(column, None);
        assert_eq!(in_json, Some(42));
    }

    fn kind_meta(kind: serde_json::Value) -> HashMap<String, serde_json::Value> {
        HashMap::from([(KIND_KEY.to_string(), kind)])
    }

    /// v13: the column is the store of record for `kind` too. Asserted at the
    /// storage layer, not through `recall`, because `recall` hydrates the key
    /// back and would hide a double-write.
    #[test]
    fn kind_is_written_to_the_column_and_not_mirrored_in_metadata() {
        let store = setup();
        store
            .remember(
                AgentId::new(),
                "an agent-authored note",
                MemorySource::Conversation,
                "episodic",
                kind_meta(serde_json::json!("note")),
            )
            .unwrap();

        let conn = store.conn.lock().unwrap();
        let (column, in_json): (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT kind, json_extract(metadata, '$.kind') FROM memories",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(column.as_deref(), Some("note"));
        assert_eq!(
            in_json, None,
            "kind must not be mirrored into the metadata JSON"
        );
    }

    /// The kernel's recall payload reads `kind` out of
    /// `MemoryFragment::metadata`. Moving the fact to a column is only safe
    /// because the read path puts it back.
    #[test]
    fn recall_hydrates_kind_back_into_metadata() {
        let store = setup();
        store
            .remember(
                AgentId::new(),
                "an agent-authored note",
                MemorySource::Conversation,
                "episodic",
                kind_meta(serde_json::json!("note")),
            )
            .unwrap();

        let results = store.recall("note", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].metadata.get(KIND_KEY),
            Some(&serde_json::json!("note"))
        );
    }

    /// `memory_recall(kind = ...)` compiles to `kind = ?` against the index.
    /// The observable contract is unchanged, which is the point: this fails if
    /// the special case selects the wrong rows.
    #[test]
    fn kind_filter_selects_only_that_kind() {
        let store = setup();
        let agent_id = AgentId::new();
        for (content, kind) in [("a note", "note"), ("a turn", "turn")] {
            store
                .remember(
                    agent_id,
                    content,
                    MemorySource::Conversation,
                    "episodic",
                    kind_meta(serde_json::json!(kind)),
                )
                .unwrap();
        }
        // A row with no kind at all. Must not match any kind filter -- that is
        // what "filter by kind" has to mean for the unbackfilled corpus.
        store
            .remember(
                agent_id,
                "an unclassified row",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        let filter = MemoryFilter {
            agent_id: Some(agent_id),
            metadata: kind_meta(serde_json::json!("note")),
            ..Default::default()
        };
        let results = store.recall("", 10, Some(filter)).unwrap();
        assert_eq!(results.len(), 1, "got: {results:?}");
        assert_eq!(results[0].content, "a note");
    }

    /// A row written BEFORE the promotion carries `kind` in its JSON and has a
    /// NULL column. The column-only filter would silently lose it, so
    /// `migrate_v13` lifts the key across. This test stands in for that
    /// population: hand-write the legacy shape, then prove the filter still
    /// finds it after the lift.
    #[test]
    fn a_pre_v13_row_is_still_found_by_a_kind_filter_after_the_lift() {
        let store = setup();
        let agent_id = AgentId::new();
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, kind)
                 VALUES (?1, ?2, 'a legacy note', '\"conversation\"', 'episodic', 1.0, '{\"kind\":\"note\"}', ?3, ?3, 0, 0, NULL)",
                rusqlite::params![
                    MemoryId::new().0.to_string(),
                    agent_id.0.to_string(),
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
            // Re-run the lift the way a v12 -> v13 upgrade would.
            conn.pragma_update(None, "user_version", 12).unwrap();
            crate::migration::run_migrations(&conn).unwrap();
        }

        let filter = MemoryFilter {
            agent_id: Some(agent_id),
            metadata: kind_meta(serde_json::json!("note")),
            ..Default::default()
        };
        let results = store.recall("", 10, Some(filter)).unwrap();
        assert_eq!(results.len(), 1, "got: {results:?}");
        assert_eq!(results[0].content, "a legacy note");
    }

    /// A non-string kind is not a kind. It must not be promoted to the column
    /// (the column feeds an equality filter callers trust) and it must not be
    /// silently discarded either -- it stays in the JSON as the caller wrote it.
    #[test]
    fn a_malformed_kind_is_left_in_metadata_and_leaves_the_column_null() {
        let store = setup();
        store
            .remember(
                AgentId::new(),
                "a malformed row",
                MemorySource::Conversation,
                "episodic",
                kind_meta(serde_json::json!(42)),
            )
            .unwrap();

        let conn = store.conn.lock().unwrap();
        let (column, in_json): (Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT kind, json_extract(metadata, '$.kind') FROM memories",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(column, None);
        assert_eq!(in_json, Some(42));
    }

    /// The metadata key is interpolated into a JSON path literal, which cannot
    /// be parameterised. It is the one string in this query builder that is not
    /// bound, so it is charset-restricted rather than trusted.
    #[test]
    fn metadata_filter_rejects_a_key_that_could_escape_the_json_path() {
        let store = setup();
        let filter = MemoryFilter {
            metadata: HashMap::from([("kind') OR 1=1 --".to_string(), serde_json::json!("x"))]),
            ..Default::default()
        };
        assert!(store.recall("", 10, Some(filter)).is_err());
    }

    #[test]
    fn test_forget() {
        let store = setup();
        let agent_id = AgentId::new();
        let id = store
            .remember(
                agent_id,
                "To forget",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        store.forget(id).unwrap();
        let results = store.recall("To forget", 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remember_with_embedding() {
        let store = setup();
        let agent_id = AgentId::new();
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let id = store
            .remember_with_embedding(
                agent_id,
                "Rust is great",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&embedding),
            )
            .unwrap();
        assert_ne!(id.0.to_string(), "");
    }

    #[test]
    fn test_vector_recall_ranking() {
        let store = setup();
        let agent_id = AgentId::new();

        // Store 3 memories with embeddings pointing in different directions
        let emb_rust = vec![0.9, 0.1, 0.0, 0.0]; // "Rust" direction
        let emb_python = vec![0.0, 0.0, 0.9, 0.1]; // "Python" direction
        let emb_mixed = vec![0.5, 0.5, 0.0, 0.0]; // mixed

        store
            .remember_with_embedding(
                agent_id,
                "Rust is a systems language",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&emb_rust),
            )
            .unwrap();
        store
            .remember_with_embedding(
                agent_id,
                "Python is interpreted",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&emb_python),
            )
            .unwrap();
        store
            .remember_with_embedding(
                agent_id,
                "Both are popular",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&emb_mixed),
            )
            .unwrap();

        // Query with a "Rust"-like embedding
        let query_emb = vec![0.85, 0.15, 0.0, 0.0];
        let results = store
            .recall_with_embedding("", 3, None, Some(&query_emb))
            .unwrap();

        assert_eq!(results.len(), 3);
        // Rust memory should be first (highest cosine similarity)
        assert!(results[0].content.contains("Rust"));
        // Python memory should be last (lowest similarity)
        assert!(results[2].content.contains("Python"));
    }

    #[test]
    fn test_update_embedding() {
        let store = setup();
        let agent_id = AgentId::new();
        let id = store
            .remember(
                agent_id,
                "No embedding yet",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        // Update with embedding
        let emb = vec![1.0, 0.0, 0.0];
        store.update_embedding(id, &emb).unwrap();

        // Verify the embedding is stored by doing vector recall
        let query_emb = vec![1.0, 0.0, 0.0];
        let results = store
            .recall_with_embedding("", 10, None, Some(&query_emb))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].embedding.is_some());
        assert_eq!(results[0].embedding.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_mixed_embedded_and_non_embedded() {
        let store = setup();
        let agent_id = AgentId::new();

        // One memory with embedding, one without
        store
            .remember_with_embedding(
                agent_id,
                "Has embedding",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&[1.0, 0.0]),
            )
            .unwrap();
        store
            .remember(
                agent_id,
                "No embedding",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        // Vector recall should rank embedded memory higher
        let results = store
            .recall_with_embedding("", 10, None, Some(&[1.0, 0.0]))
            .unwrap();
        assert_eq!(results.len(), 2);
        // Embedded memory should rank first
        assert_eq!(results[0].content, "Has embedding");
    }
}
