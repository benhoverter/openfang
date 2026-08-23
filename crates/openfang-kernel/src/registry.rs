//! Agent registry — tracks all agents, their state, and indexes.

use dashmap::DashMap;
use openfang_types::agent::{AgentEntry, AgentId, AgentMode, AgentState};
use openfang_types::error::{OpenFangError, OpenFangResult};

/// Registry of all agents in the kernel.
pub struct AgentRegistry {
    /// Primary index: agent ID → entry.
    agents: DashMap<AgentId, AgentEntry>,
    /// Name index: human-readable name → agent ID.
    name_index: DashMap<String, AgentId>,
    /// Tag index: tag → list of agent IDs.
    tag_index: DashMap<String, Vec<AgentId>>,
}

impl AgentRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            agents: DashMap::new(),
            name_index: DashMap::new(),
            tag_index: DashMap::new(),
        }
    }

    /// Register a new agent.
    pub fn register(&self, entry: AgentEntry) -> OpenFangResult<()> {
        if self.name_index.contains_key(&entry.name) {
            return Err(OpenFangError::AgentAlreadyExists(entry.name.clone()));
        }
        let id = entry.id;
        self.name_index.insert(entry.name.clone(), id);
        for tag in &entry.tags {
            self.tag_index.entry(tag.clone()).or_default().push(id);
        }
        self.agents.insert(id, entry);
        Ok(())
    }

    /// Get an agent entry by ID.
    pub fn get(&self, id: AgentId) -> Option<AgentEntry> {
        self.agents.get(&id).map(|e| e.value().clone())
    }

    /// Find an agent by name.
    pub fn find_by_name(&self, name: &str) -> Option<AgentEntry> {
        self.name_index
            .get(name)
            .and_then(|id| self.agents.get(id.value()).map(|e| e.value().clone()))
    }

    /// Look up the id bound to a name without cloning the whole entry.
    ///
    /// Cheap pre-flight for callers that only need to know "is this name
    /// taken, and by whom" — notably `spawn_agent_with_parent`, which must
    /// answer that question *before* it starts mutating state.
    pub fn id_for_name(&self, name: &str) -> Option<AgentId> {
        self.name_index.get(name).map(|id| *id.value())
    }

    /// Update agent state.
    pub fn set_state(&self, id: AgentId, state: AgentState) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.state = state;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update agent operational mode.
    pub fn set_mode(&self, id: AgentId, mode: AgentMode) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.mode = mode;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Remove an agent from the registry.
    pub fn remove(&self, id: AgentId) -> OpenFangResult<AgentEntry> {
        let (_, entry) = self
            .agents
            .remove(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        self.name_index.remove(&entry.name);
        for tag in &entry.tags {
            if let Some(mut ids) = self.tag_index.get_mut(tag) {
                ids.retain(|&agent_id| agent_id != id);
            }
        }
        Ok(entry)
    }

    /// List all agents.
    pub fn list(&self) -> Vec<AgentEntry> {
        self.agents.iter().map(|e| e.value().clone()).collect()
    }

    /// ANAI-208. Every agent that declares membership in `project`.
    ///
    /// The fleet query "which agents are on tttb", answered from declarations
    /// instead of from Ben's memory. Order is registry order, which is
    /// unspecified — callers that render this should sort.
    pub fn agents_in_project(&self, project: &str) -> Vec<AgentEntry> {
        self.agents
            .iter()
            .filter(|e| e.value().manifest.is_member_of(project))
            .map(|e| e.value().clone())
            .collect()
    }

    /// ANAI-208. The projects one agent declares, or an empty vector if the
    /// agent is unknown.
    ///
    /// Unknown-agent and no-membership collapse to the same answer on purpose:
    /// both mean "no project claim can be made on this agent's behalf", and
    /// every consumer treats them identically.
    pub fn projects_of(&self, id: AgentId) -> Vec<String> {
        self.agents
            .get(&id)
            .map(|e| e.value().manifest.projects.clone())
            .unwrap_or_default()
    }

    /// ANAI-208. Set an agent's declared project membership in the registry.
    ///
    /// Exists because roughly a third of the running fleet has no `agent.toml`
    /// at all — spawned agents whose manifests live only in SQLite and are
    /// restored at boot. Backfilling membership by editing files would reach
    /// the ~45 agents that have files and silently miss the ~26 that do not,
    /// and the missing cohort is the largest single project group. A query
    /// that under-reports looks exactly like a query that works.
    ///
    /// Rejects a malformed slug rather than warning: this is a deliberate,
    /// attended call, not a daemon-restart manifest load, so there is someone
    /// to hand the error to. Persistence is the caller's job — see
    /// `Kernel::set_agent_projects`, which pairs this with `save_agent`.
    pub fn update_projects(&self, id: AgentId, projects: Vec<String>) -> OpenFangResult<()> {
        for slug in &projects {
            openfang_types::agent::validate_project_slug(slug)
                .map_err(OpenFangError::InvalidInput)?;
        }
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.projects = projects;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Add a child agent ID to a parent's children list.
    pub fn add_child(&self, parent_id: AgentId, child_id: AgentId) {
        if let Some(mut entry) = self.agents.get_mut(&parent_id) {
            entry.children.push(child_id);
        }
    }

    /// Count of registered agents.
    pub fn count(&self) -> usize {
        self.agents.len()
    }

    /// Update an agent's session ID (for session reset).
    pub fn update_session_id(
        &self,
        id: AgentId,
        new_session_id: openfang_types::agent::SessionId,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.session_id = new_session_id;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's workspace path.
    pub fn update_workspace(
        &self,
        id: AgentId,
        workspace: Option<std::path::PathBuf>,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.workspace = workspace;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's private state directory path. The state directory
    /// holds identity files, sessions, and per-agent memory and is always
    /// kept separate from the user-facing workspace. See issue #1097.
    pub fn update_state_dir(
        &self,
        id: AgentId,
        state_dir: Option<std::path::PathBuf>,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.state_dir = state_dir;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's visual identity (emoji, avatar, color).
    pub fn update_identity(
        &self,
        id: AgentId,
        identity: openfang_types::agent::AgentIdentity,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.identity = identity;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's model configuration.
    pub fn update_model(&self, id: AgentId, new_model: String) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.model.model = new_model;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's model AND provider together.
    pub fn update_model_and_provider(
        &self,
        id: AgentId,
        new_model: String,
        new_provider: String,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.model.model = new_model;
        entry.manifest.model.provider = new_provider;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's model, provider, and connection hints together.
    pub fn update_model_provider_config(
        &self,
        id: AgentId,
        new_model: String,
        new_provider: String,
        api_key_env: Option<String>,
        base_url: Option<String>,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.model.model = new_model;
        entry.manifest.model.provider = new_provider;
        entry.manifest.model.api_key_env = api_key_env;
        entry.manifest.model.base_url = base_url;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's fallback model chain.
    pub fn update_fallback_models(
        &self,
        id: AgentId,
        fallback_models: Vec<openfang_types::agent::FallbackModel>,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.fallback_models = fallback_models;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's skill allowlist.
    pub fn update_skills(&self, id: AgentId, skills: Vec<String>) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.skills = skills;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's MCP server allowlist.
    pub fn update_mcp_servers(&self, id: AgentId, servers: Vec<String>) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.mcp_servers = servers;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's tool allowlist and blocklist.
    pub fn update_tool_filters(
        &self,
        id: AgentId,
        allowlist: Option<Vec<String>>,
        blocklist: Option<Vec<String>>,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        if let Some(al) = allowlist {
            entry.manifest.tool_allowlist = al;
        }
        if let Some(bl) = blocklist {
            entry.manifest.tool_blocklist = bl;
        }
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Touch an agent — refresh last_active without changing any other state.
    /// Used by the agent loop to prevent heartbeat false-positives during long LLM calls.
    pub fn touch(&self, id: AgentId) {
        if let Some(mut entry) = self.agents.get_mut(&id) {
            entry.last_active = chrono::Utc::now();
        }
    }

    /// Update an agent's system prompt (hot-swap, takes effect on next message).
    pub fn update_system_prompt(&self, id: AgentId, new_prompt: String) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.model.system_prompt = new_prompt;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's name (also updates the name index).
    pub fn update_name(&self, id: AgentId, new_name: String) -> OpenFangResult<()> {
        if let Some(existing_id) = self.name_index.get(&new_name).as_deref().copied() {
            if existing_id != id {
                return Err(OpenFangError::AgentAlreadyExists(new_name));
            }
            // Same agent owns this name — no-op
            return Ok(());
        }
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        let old_name = entry.name.clone();
        entry.name = new_name.clone();
        entry.manifest.name = new_name.clone();
        entry.last_active = chrono::Utc::now();
        // Update name index
        drop(entry);
        self.name_index.remove(&old_name);
        self.name_index.insert(new_name, id);
        Ok(())
    }

    /// Update an agent's description.
    pub fn update_description(&self, id: AgentId, new_desc: String) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.manifest.description = new_desc;
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Update an agent's resource quota (budget limits).
    pub fn update_resources(
        &self,
        id: AgentId,
        hourly: Option<f64>,
        daily: Option<f64>,
        monthly: Option<f64>,
        tokens_per_hour: Option<u64>,
    ) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        if let Some(v) = hourly {
            entry.manifest.resources.max_cost_per_hour_usd = v;
        }
        if let Some(v) = daily {
            entry.manifest.resources.max_cost_per_day_usd = v;
        }
        if let Some(v) = monthly {
            entry.manifest.resources.max_cost_per_month_usd = v;
        }
        if let Some(v) = tokens_per_hour {
            entry.manifest.resources.max_llm_tokens_per_hour = v;
        }
        entry.last_active = chrono::Utc::now();
        Ok(())
    }

    /// Mark an agent's onboarding as complete.
    pub fn mark_onboarding_complete(&self, id: AgentId) -> OpenFangResult<()> {
        let mut entry = self
            .agents
            .get_mut(&id)
            .ok_or_else(|| OpenFangError::AgentNotFound(id.to_string()))?;
        entry.onboarding_completed = true;
        entry.onboarding_completed_at = Some(chrono::Utc::now());
        entry.last_active = chrono::Utc::now();
        Ok(())
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use openfang_types::agent::*;
    use std::collections::HashMap;

    fn test_entry(name: &str) -> AgentEntry {
        AgentEntry {
            id: AgentId::new(),
            name: name.to_string(),
            manifest: AgentManifest {
                file_policy: None,
                name: name.to_string(),
                version: "0.1.0".to_string(),
                description: "test".to_string(),
                author: "test".to_string(),
                module: "test".to_string(),
                schedule: ScheduleMode::default(),
                model: ModelConfig::default(),
                fallback_models: vec![],
                resources: ResourceQuota::default(),
                priority: Priority::default(),
                capabilities: ManifestCapabilities::default(),
                profile: None,
                tools: HashMap::new(),
                skills: vec![],
                mcp_servers: vec![],
                projects: vec![],
                metadata: HashMap::new(),
                tags: vec![],
                routing: None,
                autonomous: None,
                pinned_model: None,
                workspace: None,
                state_dir: None,
                generate_identity_files: true,
                exec_policy: None,
                tool_allowlist: vec![],
                tool_blocklist: vec![],
                cache_context: false,
                max_history_messages: None,
            },
            state: AgentState::Created,
            mode: AgentMode::default(),
            created_at: Utc::now(),
            last_active: Utc::now(),
            parent: None,
            children: vec![],
            session_id: SessionId::new(),
            tags: vec![],
            identity: Default::default(),
            onboarding_completed: false,
            onboarding_completed_at: None,
        }
    }

    #[test]
    fn agents_in_project_answers_from_declarations() {
        let registry = AgentRegistry::new();

        let mut on_fork = test_entry("openfang-alpha");
        on_fork.manifest.projects = vec!["openfang-fork".into(), "fleet".into()];
        let mut also_fork = test_entry("openfang-memory");
        also_fork.manifest.projects = vec!["openfang-fork".into()];
        let undeclared = test_entry("assistant");
        let undeclared_id = undeclared.id;

        registry.register(on_fork).unwrap();
        registry.register(also_fork).unwrap();
        registry.register(undeclared).unwrap();

        let mut members: Vec<String> = registry
            .agents_in_project("openfang-fork")
            .into_iter()
            .map(|e| e.name)
            .collect();
        members.sort();
        assert_eq!(members, vec!["openfang-alpha", "openfang-memory"]);

        assert_eq!(registry.agents_in_project("fleet").len(), 1);

        // The load-bearing negative: an agent that declares nothing is a
        // member of nothing, not a member of everything by omission. Get this
        // backwards and all 71 undeclared agents join every project at once.
        assert!(registry.projects_of(undeclared_id).is_empty());
        assert!(!registry
            .agents_in_project("openfang-fork")
            .iter()
            .any(|e| e.name == "assistant"));
    }

    #[test]
    fn projects_of_unknown_agent_is_empty_not_a_panic() {
        let registry = AgentRegistry::new();
        assert!(registry.projects_of(AgentId::new()).is_empty());
    }

    #[test]
    fn update_projects_sets_membership_and_refuses_a_malformed_slug() {
        let registry = AgentRegistry::new();
        let entry = test_entry("kimiya-spike05-c1");
        let id = entry.id;
        registry.register(entry).unwrap();

        registry.update_projects(id, vec!["kimiya".into()]).unwrap();
        assert_eq!(registry.projects_of(id), vec!["kimiya".to_string()]);
        assert_eq!(registry.agents_in_project("kimiya").len(), 1);

        // A bad slug is refused whole — no partial application, or the agent
        // would end up in some of the projects it asked for and not others.
        let err = registry
            .update_projects(id, vec!["openfang".into(), "Bad Slug".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("Bad Slug"), "{err}");
        assert_eq!(registry.projects_of(id), vec!["kimiya".to_string()]);

        // And membership can be cleared back to none.
        registry.update_projects(id, vec![]).unwrap();
        assert!(registry.projects_of(id).is_empty());
    }

    #[test]
    fn test_register_and_get() {
        let registry = AgentRegistry::new();
        let entry = test_entry("test-agent");
        let id = entry.id;
        registry.register(entry).unwrap();
        assert!(registry.get(id).is_some());
    }

    #[test]
    fn test_find_by_name() {
        let registry = AgentRegistry::new();
        let entry = test_entry("my-agent");
        registry.register(entry).unwrap();
        assert!(registry.find_by_name("my-agent").is_some());
    }

    #[test]
    fn test_duplicate_name() {
        let registry = AgentRegistry::new();
        registry.register(test_entry("dup")).unwrap();
        assert!(registry.register(test_entry("dup")).is_err());
    }

    #[test]
    fn test_remove() {
        let registry = AgentRegistry::new();
        let entry = test_entry("removable");
        let id = entry.id;
        registry.register(entry).unwrap();
        registry.remove(id).unwrap();
        assert!(registry.get(id).is_none());
    }
}
