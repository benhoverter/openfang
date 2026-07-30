//! Execution approval manager — gates dangerous operations behind human approval.

use chrono::Utc;
use dashmap::DashMap;
use openfang_types::approval::{
    ApprovalDecision, ApprovalPolicy, ApprovalRequest, ApprovalResponse, CacheScope, RiskLevel,
};
use std::collections::VecDeque;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Max pending requests per agent.
const MAX_PENDING_PER_AGENT: usize = 5;
/// Max recent approval records to retain for history and UI visibility.
const MAX_RECENT_APPROVALS: usize = 100;
/// Capacity of the additive approval-lifecycle broadcast channel. Sized to
/// absorb bursts without lagging slow surfacers; on overflow the oldest events
/// are dropped (lossy by design — decision semantics never depend on delivery).
const APPROVAL_EVENT_BUFFER: usize = 256;

/// Key for a cached approval decision. Always scoped to `agent_id` so one
/// agent's cached trust never clears another's. `tool` plus `scope` pin the
/// blast radius: `Tool` blankets a whole tool, `SimilarBinary` blankets one
/// shell binary (exact spelling).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    agent_id: String,
    tool: String,
    scope: CacheScope,
}

/// A cached approval. Bounded by both an absolute expiry and a remaining-use
/// count; whichever trips first evicts the entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    expires_at: chrono::DateTime<Utc>,
    uses_remaining: u32,
}

/// Manages approval requests with oneshot channels for blocking resolution.
pub struct ApprovalManager {
    pending: DashMap<Uuid, PendingRequest>,
    recent: std::sync::Mutex<VecDeque<ApprovalRecord>>,
    policy: std::sync::RwLock<ApprovalPolicy>,
    /// Additive lifecycle event source. Emitting requires no subscribers; if
    /// none are listening, sends fail silently and approval flow is unaffected.
    events: tokio::sync::broadcast::Sender<ApprovalEvent>,
    /// In-memory approval cache. Populated only when an operator picks a
    /// caching resolution (Approve Similar / Approve Tool); consulted on the
    /// way in to skip surfacing. Never persisted: a daemon bounce wipes it,
    /// and any policy change clears it (see `update_policy`).
    cache: DashMap<CacheKey, CacheEntry>,
}

struct PendingRequest {
    request: ApprovalRequest,
    sender: tokio::sync::oneshot::Sender<ApprovalDecision>,
}

#[derive(Debug, Clone)]
pub struct ApprovalRecord {
    pub request: ApprovalRequest,
    pub decision: ApprovalDecision,
    pub decided_at: chrono::DateTime<Utc>,
    pub decided_by: Option<String>,
}

/// Lifecycle events emitted by the [`ApprovalManager`] so out-of-band
/// surfacers (e.g. the channel bridge) can push prompts and post follow-ups.
///
/// Purely additive: delivered on a `tokio::sync::broadcast` channel that
/// requires no subscribers. With no listener, sends are dropped silently and
/// the approve / deny / timeout decision path is completely unaffected. The
/// carried [`ApprovalRequest`] includes its `origin`, which a surfacer may use
/// to resolve a push destination — `origin` is audit/targeting only and is
/// never an authorization carrier.
#[derive(Debug, Clone)]
pub enum ApprovalEvent {
    /// A new request was parked and is awaiting human resolution.
    Submitted(ApprovalRequest),
    /// A request was resolved by an operator (approved or denied).
    Resolved {
        request: ApprovalRequest,
        decision: ApprovalDecision,
        decided_by: Option<String>,
    },
    /// A request expired before any operator resolved it.
    TimedOut(ApprovalRequest),
}

impl ApprovalManager {
    pub fn new(policy: ApprovalPolicy) -> Self {
        let (events, _) = tokio::sync::broadcast::channel(APPROVAL_EVENT_BUFFER);
        Self {
            pending: DashMap::new(),
            recent: std::sync::Mutex::new(VecDeque::new()),
            policy: std::sync::RwLock::new(policy),
            events,
            cache: DashMap::new(),
        }
    }

    /// Subscribe to the additive approval-lifecycle event stream.
    ///
    /// Returns a fresh `broadcast::Receiver`. Subscribing is the only way to
    /// observe [`ApprovalEvent`]s; not subscribing leaves all existing behavior
    /// untouched. The channel is lossy under load (see [`APPROVAL_EVENT_BUFFER`]).
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ApprovalEvent> {
        self.events.subscribe()
    }

    /// Check if a tool requires approval based on current policy.
    pub fn requires_approval(&self, tool_name: &str) -> bool {
        let policy = self.policy.read().unwrap_or_else(|e| e.into_inner());
        policy.require_approval.iter().any(|t| t == tool_name)
    }

    /// Submit an approval request. Returns a future that resolves when approved/denied/timed out.
    pub async fn request_approval(&self, req: ApprovalRequest) -> ApprovalDecision {
        // Cache consult: a prior operator-cached decision for this
        // (agent, tool, scope) skips surfacing entirely and auto-approves.
        // Only ever approves — denials are never cached.
        if self.check_cache(&req) {
            debug!(
                request_id = %req.id,
                agent_id = %req.agent_id,
                tool = %req.tool_name,
                "Approval auto-granted from cache"
            );
            return ApprovalDecision::Approved;
        }

        // Check per-agent pending limit
        let agent_pending = self
            .pending
            .iter()
            .filter(|r| r.value().request.agent_id == req.agent_id)
            .count();
        if agent_pending >= MAX_PENDING_PER_AGENT {
            warn!(agent_id = %req.agent_id, "Approval request rejected: too many pending");
            return ApprovalDecision::Denied;
        }

        let timeout = std::time::Duration::from_secs(req.timeout_secs);
        let id = req.id;
        let req_for_timeout = req.clone();
        let req_for_event = req.clone();

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.insert(
            id,
            PendingRequest {
                request: req,
                sender: tx,
            },
        );

        info!(request_id = %id, "Approval request submitted, waiting for resolution");
        // Additive surfacing hook: lossy, non-blocking, no-op without subscribers.
        let _ = self.events.send(ApprovalEvent::Submitted(req_for_event));

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(decision)) => {
                debug!(request_id = %id, ?decision, "Approval resolved");
                decision
            }
            _ => {
                let request = self
                    .pending
                    .remove(&id)
                    .map(|(_, pending)| pending.request)
                    .unwrap_or(req_for_timeout);
                let _ = self.events.send(ApprovalEvent::TimedOut(request.clone()));
                self.push_recent(request, ApprovalDecision::TimedOut, None, Utc::now());
                warn!(request_id = %id, "Approval request timed out");
                ApprovalDecision::TimedOut
            }
        }
    }

    /// Resolve a pending request (called by API/UI).
    pub fn resolve(
        &self,
        request_id: Uuid,
        decision: ApprovalDecision,
        decided_by: Option<String>,
    ) -> Result<ApprovalResponse, String> {
        match self.pending.remove(&request_id) {
            Some((_, pending)) => {
                let response = ApprovalResponse {
                    request_id,
                    decision,
                    decided_at: Utc::now(),
                    decided_by,
                };
                self.push_recent(
                    pending.request.clone(),
                    decision,
                    response.decided_by.clone(),
                    response.decided_at,
                );
                // Send decision to waiting agent (ignore error if receiver dropped)
                let _ = pending.sender.send(decision);
                // Additive surfacing hook: lossy, non-blocking, no-op without subscribers.
                let _ = self.events.send(ApprovalEvent::Resolved {
                    request: pending.request,
                    decision,
                    decided_by: response.decided_by.clone(),
                });
                info!(request_id = %request_id, ?decision, "Approval request resolved");
                Ok(response)
            }
            None => Err(format!("No pending approval request with id {request_id}")),
        }
    }

    /// List all pending requests (for API/dashboard display).
    pub fn list_pending(&self) -> Vec<ApprovalRequest> {
        self.pending
            .iter()
            .map(|r| r.value().request.clone())
            .collect()
    }

    /// List recent non-pending approvals, newest first.
    pub fn list_recent(&self, limit: usize) -> Vec<ApprovalRecord> {
        let recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        recent.iter().take(limit).cloned().collect()
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Update the approval policy (for hot-reload).
    ///
    /// Any policy change clears the approval cache: cached trust was granted
    /// under the old policy and must not survive a posture change.
    pub fn update_policy(&self, policy: ApprovalPolicy) {
        *self.policy.write().unwrap_or_else(|e| e.into_inner()) = policy;
        self.cache.clear();
    }

    /// Get a copy of the current policy.
    pub fn policy(&self) -> ApprovalPolicy {
        self.policy
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Classify the risk level of a tool invocation.
    pub fn classify_risk(tool_name: &str) -> RiskLevel {
        match tool_name {
            "shell_exec" => RiskLevel::Critical,
            "file_write" | "file_delete" => RiskLevel::High,
            "web_fetch" | "browser_navigate" => RiskLevel::Medium,
            _ => RiskLevel::Low,
        }
    }

    /// Consult the cache for a prior operator-granted decision covering this
    /// request. Tries the broader `Tool` scope first, then `SimilarBinary`
    /// (only when the request carries a `cache_binary`). On a live hit the
    /// entry's use count is decremented (evicted at zero) and `true` is
    /// returned; an expired entry is evicted and treated as a miss.
    fn check_cache(&self, req: &ApprovalRequest) -> bool {
        let mut keys = vec![CacheKey {
            agent_id: req.agent_id.clone(),
            tool: req.tool_name.clone(),
            scope: CacheScope::Tool,
        }];
        if let Some(bin) = &req.cache_binary {
            keys.push(CacheKey {
                agent_id: req.agent_id.clone(),
                tool: req.tool_name.clone(),
                scope: CacheScope::SimilarBinary(bin.clone()),
            });
        }
        keys.into_iter().any(|k| self.try_consume(&k))
    }

    /// Try to consume one use of a cached entry. Returns `true` only if the
    /// entry exists, is unexpired, and had a remaining use (which is then
    /// spent). Expired or exhausted entries are evicted. The `get_mut` guard
    /// is dropped before any `remove` to avoid a same-shard self-deadlock.
    fn try_consume(&self, key: &CacheKey) -> bool {
        let now = Utc::now();
        let mut hit = false;
        let mut evict = false;
        if let Some(mut entry) = self.cache.get_mut(key) {
            if entry.expires_at <= now || entry.uses_remaining == 0 {
                evict = true;
            } else {
                entry.uses_remaining -= 1;
                hit = true;
                if entry.uses_remaining == 0 {
                    evict = true;
                }
            }
        }
        if evict {
            self.cache.remove(key);
        }
        hit
    }

    /// Populate the cache from an operator's caching resolution. Called from
    /// the resolve site only for `Approve Similar` / `Approve Tool` — never
    /// for `Approve Once` or `Deny`. A `cache_ttl_secs` or `cache_max_uses`
    /// of `0` disables caching, so this becomes a no-op.
    pub fn cache_decision(&self, req: &ApprovalRequest, scope: CacheScope) {
        let (ttl, max_uses) = {
            let p = self.policy.read().unwrap_or_else(|e| e.into_inner());
            (p.cache_ttl_secs, p.cache_max_uses)
        };
        if ttl == 0 || max_uses == 0 {
            debug!("Approval caching disabled (ttl or max_uses is 0); not caching");
            return;
        }
        let key = CacheKey {
            agent_id: req.agent_id.clone(),
            tool: req.tool_name.clone(),
            scope,
        };
        let entry = CacheEntry {
            expires_at: Utc::now() + chrono::Duration::seconds(ttl as i64),
            uses_remaining: max_uses,
        };
        info!(
            agent_id = %req.agent_id,
            tool = %req.tool_name,
            scope = ?key.scope,
            ttl_secs = ttl,
            max_uses,
            "Approval decision cached"
        );
        self.cache.insert(key, entry);
    }

    /// Number of live cache entries (for tests/diagnostics).
    #[cfg(test)]
    fn cache_len(&self) -> usize {
        self.cache.len()
    }

    fn push_recent(
        &self,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        decided_by: Option<String>,
        decided_at: chrono::DateTime<Utc>,
    ) {
        let mut recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        recent.push_front(ApprovalRecord {
            request,
            decision,
            decided_at,
            decided_by,
        });
        while recent.len() > MAX_RECENT_APPROVALS {
            recent.pop_back();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::approval::ApprovalPolicy;
    use std::sync::Arc;

    fn default_manager() -> ApprovalManager {
        ApprovalManager::new(ApprovalPolicy::default())
    }

    fn make_request(agent_id: &str, tool_name: &str, timeout_secs: u64) -> ApprovalRequest {
        ApprovalRequest {
            id: Uuid::new_v4(),
            agent_id: agent_id.to_string(),
            tool_name: tool_name.to_string(),
            description: "test operation".to_string(),
            action_summary: "test action".to_string(),
            risk_level: RiskLevel::High,
            requested_at: Utc::now(),
            timeout_secs,
            origin: None,
            cache_binary: None,
        }
    }

    // -----------------------------------------------------------------------
    // requires_approval
    // -----------------------------------------------------------------------

    #[test]
    fn test_requires_approval_default() {
        let mgr = default_manager();
        assert!(mgr.requires_approval("shell_exec"));
        assert!(!mgr.requires_approval("file_read"));
    }

    #[test]
    fn test_requires_approval_custom_policy() {
        let policy = ApprovalPolicy {
            require_approval: vec!["file_write".to_string(), "file_delete".to_string()],
            timeout_secs: 30,
            auto_approve_autonomous: false,
            auto_approve: false,
            cache_ttl_secs: 3600,
            cache_max_uses: 50,
        };
        let mgr = ApprovalManager::new(policy);
        assert!(mgr.requires_approval("file_write"));
        assert!(mgr.requires_approval("file_delete"));
        assert!(!mgr.requires_approval("shell_exec"));
        assert!(!mgr.requires_approval("file_read"));
    }

    // -----------------------------------------------------------------------
    // classify_risk
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_risk() {
        assert_eq!(
            ApprovalManager::classify_risk("shell_exec"),
            RiskLevel::Critical
        );
        assert_eq!(
            ApprovalManager::classify_risk("file_write"),
            RiskLevel::High
        );
        assert_eq!(
            ApprovalManager::classify_risk("file_delete"),
            RiskLevel::High
        );
        assert_eq!(
            ApprovalManager::classify_risk("web_fetch"),
            RiskLevel::Medium
        );
        assert_eq!(
            ApprovalManager::classify_risk("browser_navigate"),
            RiskLevel::Medium
        );
        assert_eq!(ApprovalManager::classify_risk("file_read"), RiskLevel::Low);
        assert_eq!(
            ApprovalManager::classify_risk("unknown_tool"),
            RiskLevel::Low
        );
    }

    // -----------------------------------------------------------------------
    // resolve nonexistent
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_nonexistent() {
        let mgr = default_manager();
        let result = mgr.resolve(Uuid::new_v4(), ApprovalDecision::Approved, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No pending approval request"));
    }

    // -----------------------------------------------------------------------
    // list_pending empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_pending_empty() {
        let mgr = default_manager();
        assert!(mgr.list_pending().is_empty());
        assert!(mgr.list_recent(10).is_empty());
    }

    // -----------------------------------------------------------------------
    // update_policy
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_policy() {
        let mgr = default_manager();
        assert!(mgr.requires_approval("shell_exec"));
        assert!(!mgr.requires_approval("file_write"));

        let new_policy = ApprovalPolicy {
            require_approval: vec!["file_write".to_string()],
            timeout_secs: 120,
            auto_approve_autonomous: true,
            auto_approve: false,
            cache_ttl_secs: 3600,
            cache_max_uses: 50,
        };
        mgr.update_policy(new_policy);

        assert!(!mgr.requires_approval("shell_exec"));
        assert!(mgr.requires_approval("file_write"));

        let policy = mgr.policy();
        assert_eq!(policy.timeout_secs, 120);
        assert!(policy.auto_approve_autonomous);
    }

    // -----------------------------------------------------------------------
    // pending_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_pending_count() {
        let mgr = default_manager();
        assert_eq!(mgr.pending_count(), 0);
    }

    // -----------------------------------------------------------------------
    // subscribe — additive lifecycle events (Y step 5 substrate)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_subscribe_receives_submitted_and_resolved() {
        let mgr = Arc::new(default_manager());
        let mut rx = mgr.subscribe();
        let req = make_request("agent-evt", "shell_exec", 60);
        let id = req.id;

        let mgr2 = Arc::clone(&mgr);
        let join = tokio::spawn(async move { mgr2.request_approval(req).await });

        match rx.recv().await.expect("submitted event") {
            ApprovalEvent::Submitted(r) => assert_eq!(r.id, id),
            other => panic!("expected Submitted, got {other:?}"),
        }

        mgr.resolve(id, ApprovalDecision::Approved, Some("ben".to_string()))
            .expect("resolve");

        match rx.recv().await.expect("resolved event") {
            ApprovalEvent::Resolved {
                request,
                decision,
                decided_by,
            } => {
                assert_eq!(request.id, id);
                assert_eq!(decision, ApprovalDecision::Approved);
                assert_eq!(decided_by.as_deref(), Some("ben"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }

        assert_eq!(join.await.unwrap(), ApprovalDecision::Approved);
    }

    #[tokio::test(start_paused = true)]
    async fn test_subscribe_receives_timed_out() {
        let mgr = Arc::new(default_manager());
        let mut rx = mgr.subscribe();
        let req = make_request("agent-evt", "shell_exec", 10);
        let id = req.id;

        let decision = mgr.request_approval(req).await;
        assert_eq!(decision, ApprovalDecision::TimedOut);

        match rx.recv().await.expect("submitted event") {
            ApprovalEvent::Submitted(r) => assert_eq!(r.id, id),
            other => panic!("expected Submitted, got {other:?}"),
        }
        match rx.recv().await.expect("timed-out event") {
            ApprovalEvent::TimedOut(r) => assert_eq!(r.id, id),
            other => panic!("expected TimedOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_subscribe_origin_carried_on_submitted() {
        // origin rides the event verbatim — audit/targeting only, never authz.
        let mgr = Arc::new(default_manager());
        let mut rx = mgr.subscribe();
        let mut req = make_request("agent-evt", "shell_exec", 60);
        req.origin = Some(openfang_types::approval::ApprovalOrigin {
            channel_type: "discord".to_string(),
            channel_id: Some("chan-123".to_string()),
            thread_id: Some("thread-9".to_string()),
            recipient: Some("peer-7".to_string()),
            sender_display_name: None,
        });
        let id = req.id;

        let mgr2 = Arc::clone(&mgr);
        let join = tokio::spawn(async move { mgr2.request_approval(req).await });

        match rx.recv().await.expect("submitted event") {
            ApprovalEvent::Submitted(r) => {
                assert_eq!(r.id, id);
                let o = r.origin.expect("origin present");
                assert_eq!(o.channel_type, "discord");
                assert_eq!(o.channel_id.as_deref(), Some("chan-123"));
                assert_eq!(o.thread_id.as_deref(), Some("thread-9"));
                assert_eq!(o.recipient.as_deref(), Some("peer-7"));
            }
            other => panic!("expected Submitted, got {other:?}"),
        }

        mgr.resolve(id, ApprovalDecision::Denied, None)
            .expect("resolve");
        assert_eq!(join.await.unwrap(), ApprovalDecision::Denied);
    }

    // -----------------------------------------------------------------------
    // request_approval — timeout
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_request_approval_timeout() {
        let mgr = Arc::new(default_manager());
        let req = make_request("agent-1", "shell_exec", 10);
        let decision = mgr.request_approval(req).await;
        assert_eq!(decision, ApprovalDecision::TimedOut);
        // After timeout, pending map should be cleaned up
        assert_eq!(mgr.pending_count(), 0);
        let recent = mgr.list_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].decision, ApprovalDecision::TimedOut);
        assert_eq!(recent[0].request.tool_name, "shell_exec");
    }

    // -----------------------------------------------------------------------
    // request_approval — approve
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_request_approval_approve() {
        let mgr = Arc::new(default_manager());
        let req = make_request("agent-1", "shell_exec", 60);
        let request_id = req.id;

        let mgr2 = Arc::clone(&mgr);
        tokio::spawn(async move {
            // Small delay to let the request register
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let result = mgr2.resolve(
                request_id,
                ApprovalDecision::Approved,
                Some("admin".to_string()),
            );
            assert!(result.is_ok());
            let resp = result.unwrap();
            assert_eq!(resp.decision, ApprovalDecision::Approved);
            assert_eq!(resp.decided_by, Some("admin".to_string()));
        });

        let decision = mgr.request_approval(req).await;
        assert_eq!(decision, ApprovalDecision::Approved);
        let recent = mgr.list_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].decision, ApprovalDecision::Approved);
        assert_eq!(recent[0].decided_by.as_deref(), Some("admin"));
    }

    // -----------------------------------------------------------------------
    // request_approval — deny
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_request_approval_deny() {
        let mgr = Arc::new(default_manager());
        let req = make_request("agent-1", "shell_exec", 60);
        let request_id = req.id;

        let mgr2 = Arc::clone(&mgr);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let result = mgr2.resolve(request_id, ApprovalDecision::Denied, None);
            assert!(result.is_ok());
        });

        let decision = mgr.request_approval(req).await;
        assert_eq!(decision, ApprovalDecision::Denied);
        let recent = mgr.list_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].decision, ApprovalDecision::Denied);
    }

    // -----------------------------------------------------------------------
    // max pending per agent
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_max_pending_per_agent() {
        let mgr = Arc::new(default_manager());

        // Fill up 5 pending requests for agent-1 (they will all be waiting)
        let mut ids = Vec::new();
        for _ in 0..MAX_PENDING_PER_AGENT {
            let req = make_request("agent-1", "shell_exec", 300);
            ids.push(req.id);
            let mgr_clone = Arc::clone(&mgr);
            tokio::spawn(async move {
                mgr_clone.request_approval(req).await;
            });
        }

        // Give spawned tasks time to register
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(mgr.pending_count(), MAX_PENDING_PER_AGENT);

        // 6th request for the same agent should be immediately denied
        let req6 = make_request("agent-1", "shell_exec", 300);
        let decision = mgr.request_approval(req6).await;
        assert_eq!(decision, ApprovalDecision::Denied);

        // A different agent should still be able to submit
        let req_other = make_request("agent-2", "shell_exec", 300);
        let other_id = req_other.id;
        let mgr2 = Arc::clone(&mgr);
        tokio::spawn(async move {
            mgr2.request_approval(req_other).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(mgr.pending_count(), MAX_PENDING_PER_AGENT + 1);

        // Cleanup: resolve all pending to avoid hanging tasks
        for id in &ids {
            let _ = mgr.resolve(*id, ApprovalDecision::Denied, None);
        }
        let _ = mgr.resolve(other_id, ApprovalDecision::Denied, None);
    }

    // -----------------------------------------------------------------------
    // policy defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_defaults() {
        let mgr = default_manager();
        let policy = mgr.policy();
        assert_eq!(policy.require_approval, vec!["shell_exec".to_string()]);
        assert_eq!(policy.timeout_secs, 60);
        assert!(!policy.auto_approve_autonomous);
    }

    // -----------------------------------------------------------------------
    // approval cache
    // -----------------------------------------------------------------------

    use openfang_types::approval::CacheScope;

    fn entry(uses: u32, offset_secs: i64) -> CacheEntry {
        CacheEntry {
            expires_at: Utc::now() + chrono::Duration::seconds(offset_secs),
            uses_remaining: uses,
        }
    }

    #[test]
    fn cache_miss_when_empty() {
        let mgr = default_manager();
        let mut req = make_request("a", "shell_exec", 60);
        req.cache_binary = Some("grep".to_string());
        assert!(!mgr.check_cache(&req));
    }

    #[test]
    fn cache_tool_scope_hit_and_use_count_eviction() {
        let mgr = default_manager();
        let req = make_request("a", "file_write", 60);
        mgr.cache.insert(
            CacheKey {
                agent_id: "a".into(),
                tool: "file_write".into(),
                scope: CacheScope::Tool,
            },
            entry(2, 60),
        );
        assert!(mgr.check_cache(&req)); // 2 -> 1
        assert!(mgr.check_cache(&req)); // 1 -> 0, evicted
        assert!(!mgr.check_cache(&req)); // gone
        assert_eq!(mgr.cache_len(), 0);
    }

    #[test]
    fn cache_similar_binary_scope_is_per_spelling() {
        let mgr = default_manager();
        let mut req = make_request("a", "shell_exec", 60);
        req.cache_binary = Some("grep".into());
        mgr.cache.insert(
            CacheKey {
                agent_id: "a".into(),
                tool: "shell_exec".into(),
                scope: CacheScope::SimilarBinary("grep".into()),
            },
            entry(5, 60),
        );
        assert!(mgr.check_cache(&req));
        // a different binary is a different key -> miss
        req.cache_binary = Some("ls".into());
        assert!(!mgr.check_cache(&req));
    }

    #[test]
    fn cache_expiry_evicts_on_consult() {
        let mgr = default_manager();
        let req = make_request("a", "file_write", 60);
        mgr.cache.insert(
            CacheKey {
                agent_id: "a".into(),
                tool: "file_write".into(),
                scope: CacheScope::Tool,
            },
            entry(5, -1), // already expired
        );
        assert!(!mgr.check_cache(&req));
        assert_eq!(mgr.cache_len(), 0);
    }

    #[test]
    fn cache_agent_isolation() {
        let mgr = default_manager();
        mgr.cache.insert(
            CacheKey {
                agent_id: "a".into(),
                tool: "file_write".into(),
                scope: CacheScope::Tool,
            },
            entry(5, 60),
        );
        // agent B is never cleared by agent A's cached approval
        assert!(!mgr.check_cache(&make_request("b", "file_write", 60)));
        assert!(mgr.check_cache(&make_request("a", "file_write", 60)));
    }

    #[test]
    fn cache_decision_populates_with_policy_bounds() {
        let mgr = default_manager();
        let seed = make_request("a", "file_write", 60);
        mgr.cache_decision(&seed, CacheScope::Tool);
        assert_eq!(mgr.cache_len(), 1);
        // default policy: 50 uses available
        assert!(mgr.check_cache(&make_request("a", "file_write", 60)));
    }

    #[test]
    fn cache_decision_noop_when_disabled() {
        let policy = ApprovalPolicy {
            cache_max_uses: 0,
            ..ApprovalPolicy::default()
        };
        let mgr = ApprovalManager::new(policy);
        mgr.cache_decision(&make_request("a", "file_write", 60), CacheScope::Tool);
        assert_eq!(mgr.cache_len(), 0);
    }

    #[tokio::test]
    async fn update_policy_clears_cache() {
        let mgr = default_manager();
        mgr.cache_decision(&make_request("a", "file_write", 60), CacheScope::Tool);
        assert_eq!(mgr.cache_len(), 1);
        mgr.update_policy(ApprovalPolicy::default());
        assert_eq!(mgr.cache_len(), 0);
    }

    #[tokio::test]
    async fn cached_request_auto_approves_without_surfacing() {
        let mgr = default_manager();
        mgr.cache_decision(&make_request("a", "file_write", 60), CacheScope::Tool);
        // A fresh request that would otherwise park (and here, with a short
        // timeout, time out) instead returns Approved immediately from cache.
        let req = make_request("a", "file_write", 1);
        assert_eq!(mgr.request_approval(req).await, ApprovalDecision::Approved);
        // one use spent
        let policy_uses = mgr.policy().cache_max_uses;
        assert!(policy_uses > 1);
    }
}
