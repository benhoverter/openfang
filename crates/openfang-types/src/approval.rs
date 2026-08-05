//! Execution approval types for the OpenFang agent OS.
//!
//! When an agent attempts a dangerous operation (e.g. `shell_exec`), the kernel
//! creates an [`ApprovalRequest`] and pauses the agent until a human operator
//! responds with an [`ApprovalResponse`]. The [`ApprovalPolicy`] configures
//! which tools require approval and how long to wait before auto-denying.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum length of tool names (chars).
const MAX_TOOL_NAME_LEN: usize = 64;

/// Maximum length of a request description (chars).
const MAX_DESCRIPTION_LEN: usize = 1024;

/// Maximum length of an action summary (chars).
const MAX_ACTION_SUMMARY_LEN: usize = 512;

/// Maximum length of the verbatim command carried on an approval request (chars).
///
/// Deliberately far above [`MAX_ACTION_SUMMARY_LEN`]: `action_summary` is a
/// *label*, `command` is the artifact the operator is actually authorizing.
/// Render surfaces elide for their own width budget; the request itself keeps
/// the command as close to verbatim as a sane bound allows.
pub const MAX_COMMAND_LEN: usize = 4096;

/// Minimum approval timeout in seconds.
const MIN_TIMEOUT_SECS: u64 = 10;

/// Maximum approval timeout in seconds.
const MAX_TIMEOUT_SECS: u64 = 300;

/// Maximum length of an origin routing field (channel_id / thread_id / recipient), chars.
const MAX_ORIGIN_FIELD_LEN: usize = 256;

/// Default approval-cache entry lifetime (seconds).
pub const DEFAULT_CACHE_TTL_SECS: u64 = 3600;

/// Default approval-cache max uses per cached entry.
pub const DEFAULT_CACHE_MAX_USES: u32 = 50;

/// Upper bound on a configured cache TTL (seconds). 24h — a daemon bounce
/// clears the in-memory cache anyway, so this just rejects absurd values.
const MAX_CACHE_TTL_SECS: u64 = 86_400;

/// Upper bound on a configured per-entry use count.
const MAX_CACHE_MAX_USES: u32 = 100_000;

/// Binaries for which the "Approve Similar" button is suppressed.
///
/// Two distinct reasons a binary lands here:
///
/// 1. **Destructive** (`rm`, `dd`, `mkfs`, …) — the binary alone carries no
///    risk signal, the args do, so caching the binary hands back the very gate
///    it exists to enforce.
/// 2. **Arbitrary** (`bash`, `python`, `env`, `sudo`, …) — an interpreter or
///    wrapper is not *dangerous*, it is *unconstrained*, which is strictly
///    worse. `argv[0] == "bash"` tells you nothing about what will run, so an
///    argv[0]-keyed cache is meaningless for it: one ✅ on
///    `bash -c "rm -rf /tmp/x"` would grant 50 unreviewed `bash` invocations
///    for an hour, routing around every entry in category 1 (ANAI-152).
///
/// Both categories always collapse to a per-command human decision
/// (Once / \[Tool\] / Deny).
pub const SIMILAR_DENYLIST: &[&str] = &[
    // -- destructive --------------------------------------------------------
    "rm",
    "dd",
    "mkfs",
    "kill",
    "killall",
    "chmod",
    "chown",
    "mv",
    "shutdown",
    "reboot",
    // -- shells / interpreters (arbitrary code via an argument) --------------
    "bash",
    "sh",
    "zsh",
    "fish",
    "ksh",
    "dash",
    "csh",
    "tcsh",
    "ash",
    "powershell",
    "pwsh",
    "cmd",
    "python",
    "python2",
    "python3",
    "node",
    "nodejs",
    "deno",
    "bun",
    "perl",
    "ruby",
    "php",
    "lua",
    "awk",
    "gawk",
    "eval",
    "exec",
    "source",
    // -- process wrappers (execute an inner command we cannot see) -----------
    "env",
    "sudo",
    "doas",
    "su",
    "nohup",
    "nice",
    "timeout",
    "xargs",
    "setsid",
    "stdbuf",
    "flock",
    "script",
    "ssh",
    "find",
    "strace",
    "gdb",
    "chroot",
    "unshare",
];

/// Returns true if `binary` (argv\[0\]) resolves to anything on the
/// Approve-Similar denylist.
///
/// Matching is deliberately **broader than the cache key**. The cache key stays
/// exact-spelling (`/bin/bash` and `bash` remain distinct keys, which only ever
/// narrows the cached radius), but the *denylist decision* folds path prefix,
/// case, `.exe` suffix, and the obfuscations in
/// [`crate::cmd_norm::deny_variants`] before comparing. Otherwise `/bin/bash`,
/// `BASH.exe` and `ba""sh` would each sail past a check meant to stop `bash`
/// (ANAI-152).
///
/// Union semantics: a hit on the raw spelling **or** any normalized variant
/// denies. A normalizer bug therefore suppresses a button that could have been
/// offered — never offers one that should have been suppressed.
pub fn is_similar_denylisted(binary: &str) -> bool {
    crate::cmd_norm::deny_variants(binary)
        .iter()
        .any(|v| denylist_hit(v))
}

/// Exact-form membership test applied to one (possibly normalized) spelling:
/// strip any path prefix, lowercase, drop a `.exe` suffix, then compare.
fn denylist_hit(token: &str) -> bool {
    let trimmed = token.trim();
    let base = trimmed
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(trimmed)
        .to_lowercase();
    let base = base.strip_suffix(".exe").unwrap_or(&base);
    SIMILAR_DENYLIST.contains(&base)
}

// ---------------------------------------------------------------------------
// RiskLevel
// ---------------------------------------------------------------------------

/// Risk level of an operation requiring approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    /// Returns a warning emoji suitable for display in dashboards and chat.
    pub fn emoji(&self) -> &'static str {
        match self {
            RiskLevel::Low => "\u{2139}\u{fe0f}",      // information source
            RiskLevel::Medium => "\u{26a0}\u{fe0f}",   // warning sign
            RiskLevel::High => "\u{1f6a8}",            // rotating light
            RiskLevel::Critical => "\u{2620}\u{fe0f}", // skull and crossbones
        }
    }
}

// ---------------------------------------------------------------------------
// ApprovalDecision
// ---------------------------------------------------------------------------

/// Decision on an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approved,
    Denied,
    TimedOut,
}

// ---------------------------------------------------------------------------
// ApprovalOrigin
// ---------------------------------------------------------------------------

/// Where an agent run originated — carried so an approval prompt can be pushed
/// back to the exact channel/conversation that triggered the run.
///
/// `None` on [`ApprovalRequest::origin`] ⇒ a non-channel trigger (cron, API
/// direct, agent_send): the emit site must fall back to the text
/// `/approve <id>` path (no proactive push).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovalOrigin {
    /// Adapter key, e.g. `"discord"`, `"slack"`. Maps to `channel_type_str()`.
    pub channel_type: String,
    /// Per-channel/conversation routing ID (Discord channel/thread, Slack
    /// conversation, Telegram chat). Sourced from `ChannelMessage::channel_id()`.
    pub channel_id: Option<String>,
    /// Thread/sub-conversation, if the trigger arrived inside one.
    pub thread_id: Option<String>,
    /// Platform user identity of the triggering sender (peer_id). Used ONLY for
    /// audit/recipient targeting — NEVER as an authz carrier (the clicker is
    /// re-authorized from the platform-attested interaction identity).
    pub recipient: Option<String>,
    /// Human-readable display name of the triggering sender (e.g. the Discord
    /// username). Rendered into the turn-context envelope / §9.1 "## Sender"
    /// alongside the snowflake carried in `recipient`. Display identity only —
    /// never an authz carrier. `None` for non-channel triggers (cron / API /
    /// agent_send). `#[serde(default)]` keeps pre-field serialized origins
    /// deserializing to `None`.
    #[serde(default)]
    pub sender_display_name: Option<String>,
}

// ---------------------------------------------------------------------------
// ApprovalRequest
// ---------------------------------------------------------------------------

/// An approval request for a dangerous agent operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Uuid,
    pub agent_id: String,
    pub tool_name: String,
    pub description: String,
    /// The specific action being requested (sanitized for display).
    pub action_summary: String,
    pub risk_level: RiskLevel,
    pub requested_at: DateTime<Utc>,
    /// Auto-deny timeout in seconds.
    pub timeout_secs: u64,
    /// Origin of the triggering run. `None` ⇒ fall back to the text approve path.
    #[serde(default)]
    pub origin: Option<ApprovalOrigin>,
    /// argv[0] of a `shell_exec` command (exact spelling), extracted once at
    /// the gate. Drives the "Approve Similar" cache key and button. `None` for
    /// non-shell tools and for commands with no parseable first token.
    #[serde(default)]
    pub cache_binary: Option<String>,
    /// The verbatim `shell_exec` command string, captured once at the gate
    /// where structured tool input still exists — never re-parsed from the
    /// mangled `action_summary` (same precedent as [`Self::cache_binary`]).
    ///
    /// This is the operator decision surface. `action_summary` is a serialized,
    /// JSON-escaped, tail-truncated *label* fit for a queue listing; deciding
    /// from it means deciding from a string whose dangerous tail was already
    /// cut (ANAI-151). Render surfaces must prefer this field when present.
    /// `None` for non-shell tools.
    ///
    /// `#[serde(default)]` keeps pre-field serialized requests deserializing.
    #[serde(default)]
    pub command: Option<String>,
}

/// Make `s` safe to place inside a triple-backtick fenced code block without
/// letting it break out of the fence.
///
/// The command is agent-controlled, so it can contain ``` and close the fence
/// the render site opened — turning the rest of the prompt into agent-authored
/// markdown. We insert a zero-width space inside any run of three or more
/// backticks: every visible character survives, the run stops being a fence
/// terminator, and nothing else about the command is rewritten.
///
/// This is the minimum mangle that closes the breakout. It is applied *only*
/// to backtick runs; deliberately not a general escaper, because the whole
/// point of ANAI-151 is that the operator sees the command as the shell will.
pub fn fence_escape(s: &str) -> String {
    if !s.contains("```") {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 8);
    let mut run = 0usize;
    for c in s.chars() {
        if c == '`' {
            run += 1;
            if run >= 3 {
                // Break the run: the third and every subsequent backtick gets a
                // zero-width space in front of it.
                out.push('\u{200b}');
                run = 1;
            }
            out.push(c);
        } else {
            run = 0;
            out.push(c);
        }
    }
    out
}

impl ApprovalRequest {
    /// Validate this request's fields.
    ///
    /// Returns `Ok(())` or an error message describing the first validation failure.
    pub fn validate(&self) -> Result<(), String> {
        // -- tool_name --
        if self.tool_name.is_empty() {
            return Err("tool_name must not be empty".into());
        }
        if self.tool_name.len() > MAX_TOOL_NAME_LEN {
            return Err(format!(
                "tool_name too long ({} chars, max {MAX_TOOL_NAME_LEN})",
                self.tool_name.len()
            ));
        }
        if !self
            .tool_name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return Err(
                "tool_name may only contain alphanumeric characters and underscores".into(),
            );
        }

        // -- description --
        if self.description.len() > MAX_DESCRIPTION_LEN {
            return Err(format!(
                "description too long ({} chars, max {MAX_DESCRIPTION_LEN})",
                self.description.len()
            ));
        }

        // -- action_summary --
        if self.action_summary.len() > MAX_ACTION_SUMMARY_LEN {
            return Err(format!(
                "action_summary too long ({} chars, max {MAX_ACTION_SUMMARY_LEN})",
                self.action_summary.len()
            ));
        }

        // -- command (optional) --
        if let Some(cmd) = &self.command {
            let n = cmd.chars().count();
            if n > MAX_COMMAND_LEN {
                return Err(format!(
                    "command too long ({n} chars, max {MAX_COMMAND_LEN})"
                ));
            }
        }

        // -- timeout_secs --
        if self.timeout_secs < MIN_TIMEOUT_SECS {
            return Err(format!(
                "timeout_secs too small ({}, min {MIN_TIMEOUT_SECS})",
                self.timeout_secs
            ));
        }
        if self.timeout_secs > MAX_TIMEOUT_SECS {
            return Err(format!(
                "timeout_secs too large ({}, max {MAX_TIMEOUT_SECS})",
                self.timeout_secs
            ));
        }

        // -- origin (optional) --
        if let Some(origin) = &self.origin {
            if origin.channel_type.len() > MAX_ORIGIN_FIELD_LEN {
                return Err(format!(
                    "origin.channel_type too long ({} chars, max {MAX_ORIGIN_FIELD_LEN})",
                    origin.channel_type.len()
                ));
            }
            for (label, value) in [
                ("origin.channel_id", &origin.channel_id),
                ("origin.thread_id", &origin.thread_id),
                ("origin.recipient", &origin.recipient),
                ("origin.sender_display_name", &origin.sender_display_name),
            ] {
                if let Some(v) = value {
                    if v.len() > MAX_ORIGIN_FIELD_LEN {
                        return Err(format!(
                            "{label} too long ({} chars, max {MAX_ORIGIN_FIELD_LEN})",
                            v.len()
                        ));
                    }
                }
            }
        }

        // -- cache_binary (optional) --
        if let Some(b) = &self.cache_binary {
            if b.len() > MAX_ORIGIN_FIELD_LEN {
                return Err(format!(
                    "cache_binary too long ({} chars, max {MAX_ORIGIN_FIELD_LEN})",
                    b.len()
                ));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApprovalResponse
// ---------------------------------------------------------------------------

/// Response to an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResponse {
    pub request_id: Uuid,
    pub decision: ApprovalDecision,
    pub decided_at: DateTime<Utc>,
    pub decided_by: Option<String>,
}

// ---------------------------------------------------------------------------
// CacheScope
// ---------------------------------------------------------------------------

/// The caching intent an operator selected when resolving an approval. This is
/// a *resolution* concept — it never reaches the requesting agent, which only
/// ever sees `Approved` / `Denied`.
///
/// - `SimilarBinary(argv0)` — "Approve Similar": cache all `shell_exec` calls
///   whose first token equals `argv0` (exact spelling). Only offered for
///   `shell_exec` and only when `argv0` is not [`is_similar_denylisted`].
/// - `Tool` — "Approve Tool": cache all calls to the request's tool. Never
///   offered for `shell_exec` (that blanket trust is `exec_policy.mode = full`
///   territory, set in `agent.toml`, not a per-prompt button).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    SimilarBinary(String),
    Tool,
}

// ---------------------------------------------------------------------------
// ApprovalPolicy
// ---------------------------------------------------------------------------

/// Configurable approval policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApprovalPolicy {
    /// Tools that always require approval. Default: `["shell_exec"]`.
    ///
    /// Accepts either a list of tool names or a boolean shorthand:
    /// - `require_approval = false` → empty list (no tools require approval)
    /// - `require_approval = true`  → `["shell_exec"]` (the default set)
    #[serde(deserialize_with = "deserialize_require_approval")]
    pub require_approval: Vec<String>,
    /// Timeout in seconds. Default: 180, range: 10..=300.
    ///
    /// 60s was the default from the first approval landing and it is too short
    /// for the thing we ask of the operator: read a full command, understand
    /// what it does, decide. A minute means "click or lose it", which trains
    /// reflex approval — the exact failure the gate exists to prevent
    /// (ANAI-151). 180s stays inside the existing `MAX_TIMEOUT_SECS` bound.
    pub timeout_secs: u64,
    /// Auto-approve in autonomous mode. Default: `false`.
    pub auto_approve_autonomous: bool,
    /// Alias: if `auto_approve = true`, clears the require list at boot.
    #[serde(default, alias = "auto_approve")]
    pub auto_approve: bool,
    /// Approval-cache entry lifetime in seconds. Default: 3600 (1h). A daemon
    /// bounce clears the in-memory cache regardless. `0` disables caching.
    pub cache_ttl_secs: u64,
    /// Max times a single cached approval may be reused before it is evicted.
    /// Default: 50. `0` disables caching.
    pub cache_max_uses: u32,
    /// Master switch for the "Approve Similar" relief valve. Default: `false`.
    ///
    /// Approve-Similar is the widest grant in the system: one click blankets a
    /// whole binary for `cache_max_uses` invocations over `cache_ttl_secs`,
    /// with no further human in the loop. It exists to relieve approval
    /// fatigue, which makes it exactly the control most likely to be clicked
    /// without reading.
    ///
    /// It ships **off** (ANAI-152). The narrowing fixes in the same change
    /// (interpreter denylist, normalized deny matching) apply underneath it
    /// regardless, so enabling it later lands on an already-closed hole rather
    /// than opening one. Turn it on deliberately:
    ///
    /// ```toml
    /// [approval]
    /// allow_similar = true
    /// ```
    ///
    /// When `false`, the button is not offered, a crafted `custom_id` asking
    /// for it is refused server-side, and [`CacheScope::SimilarBinary`] entries
    /// are refused at the cache itself — three independent points, because a
    /// gate that can be reached from more than one surface must be closed at
    /// the surface *and* at the store.
    pub allow_similar: bool,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        Self {
            require_approval: vec!["shell_exec".to_string()],
            timeout_secs: 180,
            auto_approve_autonomous: false,
            auto_approve: false,
            cache_ttl_secs: DEFAULT_CACHE_TTL_SECS,
            cache_max_uses: DEFAULT_CACHE_MAX_USES,
            allow_similar: false,
        }
    }
}

/// Custom deserializer that accepts:
/// - A list of strings: `["shell_exec", "file_write"]`
/// - A boolean: `false` → `[]`, `true` → `["shell_exec"]`
fn deserialize_require_approval<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct RequireApprovalVisitor;

    impl<'de> de::Visitor<'de> for RequireApprovalVisitor {
        type Value = Vec<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a list of tool names or a boolean")
        }

        fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
            Ok(if v {
                vec!["shell_exec".to_string()]
            } else {
                vec![]
            })
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                v.push(s);
            }
            Ok(v)
        }
    }

    deserializer.deserialize_any(RequireApprovalVisitor)
}

impl ApprovalPolicy {
    /// Apply the `auto_approve` shorthand: if true, clears the require list.
    pub fn apply_shorthands(&mut self) {
        if self.auto_approve {
            self.require_approval.clear();
        }
    }

    /// Validate this policy's fields.
    ///
    /// Returns `Ok(())` or an error message describing the first validation failure.
    pub fn validate(&self) -> Result<(), String> {
        // -- timeout_secs --
        if self.timeout_secs < MIN_TIMEOUT_SECS {
            return Err(format!(
                "timeout_secs too small ({}, min {MIN_TIMEOUT_SECS})",
                self.timeout_secs
            ));
        }
        if self.timeout_secs > MAX_TIMEOUT_SECS {
            return Err(format!(
                "timeout_secs too large ({}, max {MAX_TIMEOUT_SECS})",
                self.timeout_secs
            ));
        }

        // -- require_approval tool names --
        for (i, name) in self.require_approval.iter().enumerate() {
            if name.is_empty() {
                return Err(format!("require_approval[{i}] must not be empty"));
            }
            if name.len() > MAX_TOOL_NAME_LEN {
                return Err(format!(
                    "require_approval[{i}] too long ({} chars, max {MAX_TOOL_NAME_LEN})",
                    name.len()
                ));
            }
            if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return Err(format!(
                    "require_approval[{i}] may only contain alphanumeric characters and underscores: \"{name}\""
                ));
            }
        }

        // -- cache bounds --
        if self.cache_ttl_secs > MAX_CACHE_TTL_SECS {
            return Err(format!(
                "cache_ttl_secs too large ({}, max {MAX_CACHE_TTL_SECS})",
                self.cache_ttl_secs
            ));
        }
        if self.cache_max_uses > MAX_CACHE_MAX_USES {
            return Err(format!(
                "cache_max_uses too large ({}, max {MAX_CACHE_MAX_USES})",
                self.cache_max_uses
            ));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn valid_request() -> ApprovalRequest {
        ApprovalRequest {
            id: Uuid::new_v4(),
            agent_id: "agent-001".into(),
            tool_name: "shell_exec".into(),
            description: "Execute rm -rf /tmp/stale_cache".into(),
            action_summary: "rm -rf /tmp/stale_cache".into(),
            risk_level: RiskLevel::High,
            requested_at: Utc::now(),
            timeout_secs: 60,
            origin: None,
            cache_binary: None,
            command: None,
        }
    }

    fn valid_policy() -> ApprovalPolicy {
        ApprovalPolicy::default()
    }

    // -----------------------------------------------------------------------
    // RiskLevel
    // -----------------------------------------------------------------------

    #[test]
    fn risk_level_emoji() {
        assert_eq!(RiskLevel::Low.emoji(), "\u{2139}\u{fe0f}");
        assert_eq!(RiskLevel::Medium.emoji(), "\u{26a0}\u{fe0f}");
        assert_eq!(RiskLevel::High.emoji(), "\u{1f6a8}");
        assert_eq!(RiskLevel::Critical.emoji(), "\u{2620}\u{fe0f}");
    }

    #[test]
    fn risk_level_serde_roundtrip() {
        for level in [
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ] {
            let json = serde_json::to_string(&level).unwrap();
            let back: RiskLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, back);
        }
    }

    #[test]
    fn risk_level_rename_all() {
        let json = serde_json::to_string(&RiskLevel::Critical).unwrap();
        assert_eq!(json, "\"critical\"");
        let json = serde_json::to_string(&RiskLevel::Low).unwrap();
        assert_eq!(json, "\"low\"");
    }

    // -----------------------------------------------------------------------
    // ApprovalDecision
    // -----------------------------------------------------------------------

    #[test]
    fn decision_serde_roundtrip() {
        for decision in [
            ApprovalDecision::Approved,
            ApprovalDecision::Denied,
            ApprovalDecision::TimedOut,
        ] {
            let json = serde_json::to_string(&decision).unwrap();
            let back: ApprovalDecision = serde_json::from_str(&json).unwrap();
            assert_eq!(decision, back);
        }
    }

    #[test]
    fn decision_rename_all() {
        let json = serde_json::to_string(&ApprovalDecision::TimedOut).unwrap();
        assert_eq!(json, "\"timed_out\"");
    }

    // -----------------------------------------------------------------------
    // ApprovalRequest — valid
    // -----------------------------------------------------------------------

    #[test]
    fn valid_request_passes() {
        assert!(valid_request().validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // ApprovalRequest — tool_name
    // -----------------------------------------------------------------------

    #[test]
    fn request_empty_tool_name() {
        let mut req = valid_request();
        req.tool_name = String::new();
        let err = req.validate().unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn request_tool_name_too_long() {
        let mut req = valid_request();
        req.tool_name = "a".repeat(65);
        let err = req.validate().unwrap_err();
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn request_origin_channel_type_too_long_rejected() {
        let mut req = valid_request();
        req.origin = Some(ApprovalOrigin {
            channel_type: "a".repeat(257),
            channel_id: None,
            thread_id: None,
            recipient: None,
            sender_display_name: None,
        });
        let err = req.validate().unwrap_err();
        assert!(err.contains("origin.channel_type"), "{err}");
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn request_tool_name_64_chars_ok() {
        let mut req = valid_request();
        req.tool_name = "a".repeat(64);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_tool_name_invalid_chars() {
        let mut req = valid_request();
        req.tool_name = "shell-exec".into();
        let err = req.validate().unwrap_err();
        assert!(err.contains("alphanumeric"), "{err}");
    }

    #[test]
    fn request_tool_name_with_underscore_ok() {
        let mut req = valid_request();
        req.tool_name = "file_write".into();
        assert!(req.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // ApprovalRequest — description
    // -----------------------------------------------------------------------

    #[test]
    fn request_description_too_long() {
        let mut req = valid_request();
        req.description = "x".repeat(1025);
        let err = req.validate().unwrap_err();
        assert!(err.contains("description"), "{err}");
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn request_description_1024_ok() {
        let mut req = valid_request();
        req.description = "x".repeat(1024);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_description_empty_ok() {
        let mut req = valid_request();
        req.description = String::new();
        assert!(req.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // ApprovalRequest — action_summary
    // -----------------------------------------------------------------------

    #[test]
    fn request_action_summary_too_long() {
        let mut req = valid_request();
        req.action_summary = "x".repeat(513);
        let err = req.validate().unwrap_err();
        assert!(err.contains("action_summary"), "{err}");
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn request_action_summary_512_ok() {
        let mut req = valid_request();
        req.action_summary = "x".repeat(512);
        assert!(req.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // ApprovalRequest — timeout_secs
    // -----------------------------------------------------------------------

    #[test]
    fn request_timeout_too_small() {
        let mut req = valid_request();
        req.timeout_secs = 9;
        let err = req.validate().unwrap_err();
        assert!(err.contains("too small"), "{err}");
    }

    #[test]
    fn request_timeout_too_large() {
        let mut req = valid_request();
        req.timeout_secs = 301;
        let err = req.validate().unwrap_err();
        assert!(err.contains("too large"), "{err}");
    }

    #[test]
    fn request_timeout_min_boundary_ok() {
        let mut req = valid_request();
        req.timeout_secs = 10;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_timeout_max_boundary_ok() {
        let mut req = valid_request();
        req.timeout_secs = 300;
        assert!(req.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // ApprovalResponse — serde
    // -----------------------------------------------------------------------

    #[test]
    fn response_serde_roundtrip() {
        let resp = ApprovalResponse {
            request_id: Uuid::new_v4(),
            decision: ApprovalDecision::Approved,
            decided_at: Utc::now(),
            decided_by: Some("admin@example.com".into()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: ApprovalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.request_id, resp.request_id);
        assert_eq!(back.decision, ApprovalDecision::Approved);
        assert_eq!(back.decided_by, Some("admin@example.com".into()));
    }

    #[test]
    fn response_decided_by_none() {
        let resp = ApprovalResponse {
            request_id: Uuid::new_v4(),
            decision: ApprovalDecision::TimedOut,
            decided_at: Utc::now(),
            decided_by: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: ApprovalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.decided_by, None);
        assert_eq!(back.decision, ApprovalDecision::TimedOut);
    }

    // -----------------------------------------------------------------------
    // ApprovalPolicy — defaults
    // -----------------------------------------------------------------------

    #[test]
    fn policy_default_valid() {
        let policy = ApprovalPolicy::default();
        assert!(policy.validate().is_ok());
        assert_eq!(policy.require_approval, vec!["shell_exec".to_string()]);
        // ANAI-151: 180, not the historical 60. A minute is not enough time to
        // read a command and decide; it trains reflex approval.
        assert_eq!(policy.timeout_secs, 180);
        assert!(!policy.auto_approve_autonomous);
        assert!(!policy.auto_approve);
    }

    #[test]
    fn policy_serde_default() {
        // An empty JSON object should deserialize to defaults via #[serde(default)].
        let policy: ApprovalPolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(policy.timeout_secs, 180);
        assert_eq!(policy.require_approval, vec!["shell_exec".to_string()]);
        assert!(!policy.auto_approve_autonomous);
    }

    #[test]
    fn policy_require_approval_bool_false() {
        // require_approval = false → empty list
        let policy: ApprovalPolicy =
            serde_json::from_str(r#"{"require_approval": false}"#).unwrap();
        assert!(policy.require_approval.is_empty());
    }

    #[test]
    fn policy_require_approval_bool_true() {
        // require_approval = true → ["shell_exec"]
        let policy: ApprovalPolicy = serde_json::from_str(r#"{"require_approval": true}"#).unwrap();
        assert_eq!(policy.require_approval, vec!["shell_exec"]);
    }

    #[test]
    fn policy_auto_approve_clears_list() {
        let mut policy = ApprovalPolicy::default();
        assert!(!policy.require_approval.is_empty());
        policy.auto_approve = true;
        policy.apply_shorthands();
        assert!(policy.require_approval.is_empty());
    }

    // -----------------------------------------------------------------------
    // ApprovalPolicy — timeout_secs
    // -----------------------------------------------------------------------

    #[test]
    fn policy_timeout_too_small() {
        let mut policy = valid_policy();
        policy.timeout_secs = 9;
        let err = policy.validate().unwrap_err();
        assert!(err.contains("too small"), "{err}");
    }

    #[test]
    fn policy_timeout_too_large() {
        let mut policy = valid_policy();
        policy.timeout_secs = 301;
        let err = policy.validate().unwrap_err();
        assert!(err.contains("too large"), "{err}");
    }

    #[test]
    fn policy_timeout_boundaries_ok() {
        let mut policy = valid_policy();
        policy.timeout_secs = 10;
        assert!(policy.validate().is_ok());
        policy.timeout_secs = 300;
        assert!(policy.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // ApprovalPolicy — require_approval tool names
    // -----------------------------------------------------------------------

    #[test]
    fn policy_empty_tool_name() {
        let mut policy = valid_policy();
        policy.require_approval = vec!["shell_exec".into(), "".into()];
        let err = policy.validate().unwrap_err();
        assert!(err.contains("require_approval[1]"), "{err}");
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn policy_tool_name_too_long() {
        let mut policy = valid_policy();
        policy.require_approval = vec!["a".repeat(65)];
        let err = policy.validate().unwrap_err();
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn policy_tool_name_invalid_chars() {
        let mut policy = valid_policy();
        policy.require_approval = vec!["shell-exec".into()];
        let err = policy.validate().unwrap_err();
        assert!(err.contains("alphanumeric"), "{err}");
    }

    #[test]
    fn policy_tool_name_with_spaces_rejected() {
        let mut policy = valid_policy();
        policy.require_approval = vec!["shell exec".into()];
        let err = policy.validate().unwrap_err();
        assert!(err.contains("alphanumeric"), "{err}");
    }

    #[test]
    fn policy_multiple_valid_tools() {
        let mut policy = valid_policy();
        policy.require_approval = vec![
            "shell_exec".into(),
            "file_write".into(),
            "file_delete".into(),
        ];
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn policy_empty_require_approval_ok() {
        let mut policy = valid_policy();
        policy.require_approval = vec![];
        assert!(policy.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Full serde roundtrip — ApprovalRequest
    // -----------------------------------------------------------------------

    #[test]
    fn request_serde_roundtrip() {
        let req = valid_request();
        let json = serde_json::to_string_pretty(&req).unwrap();
        let back: ApprovalRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, req.id);
        assert_eq!(back.agent_id, req.agent_id);
        assert_eq!(back.tool_name, req.tool_name);
        assert_eq!(back.description, req.description);
        assert_eq!(back.action_summary, req.action_summary);
        assert_eq!(back.risk_level, req.risk_level);
        assert_eq!(back.timeout_secs, req.timeout_secs);
    }

    // -----------------------------------------------------------------------
    // ApprovalOrigin
    // -----------------------------------------------------------------------

    #[test]
    fn origin_serde_roundtrip() {
        let origin = ApprovalOrigin {
            channel_type: "discord".into(),
            channel_id: Some("123456789".into()),
            thread_id: Some("987654321".into()),
            recipient: Some("user-42".into()),
            sender_display_name: Some("Ben Hoverter".into()),
        };
        let json = serde_json::to_string(&origin).unwrap();
        let back: ApprovalOrigin = serde_json::from_str(&json).unwrap();
        assert_eq!(back, origin);
    }

    #[test]
    fn request_with_origin_roundtrip() {
        let mut req = valid_request();
        req.origin = Some(ApprovalOrigin {
            channel_type: "discord".into(),
            channel_id: Some("chan-1".into()),
            thread_id: None,
            recipient: Some("peer-1".into()),
            sender_display_name: Some("peer-name".into()),
        });
        let json = serde_json::to_string(&req).unwrap();
        let back: ApprovalRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.origin, req.origin);
    }

    #[test]
    fn request_legacy_json_without_origin_defaults_none() {
        // A persisted/in-flight request from before the `origin` field existed
        // must still deserialize (→ None) via #[serde(default)].
        let legacy = r#"{
            "id": "00000000-0000-0000-0000-000000000000",
            "agent_id": "agent-001",
            "tool_name": "shell_exec",
            "description": "legacy",
            "action_summary": "rm -rf /tmp/x",
            "risk_level": "high",
            "requested_at": "2026-06-14T00:00:00Z",
            "timeout_secs": 60
        }"#;
        let req: ApprovalRequest = serde_json::from_str(legacy).unwrap();
        assert_eq!(req.origin, None);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_origin_channel_id_too_long_rejected() {
        let mut req = valid_request();
        req.origin = Some(ApprovalOrigin {
            channel_type: "discord".into(),
            channel_id: Some("a".repeat(257)),
            thread_id: None,
            recipient: None,
            sender_display_name: None,
        });
        let err = req.validate().unwrap_err();
        assert!(err.contains("origin.channel_id"), "{err}");
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn request_origin_field_max_len_ok() {
        let mut req = valid_request();
        req.origin = Some(ApprovalOrigin {
            channel_type: "discord".into(),
            channel_id: Some("a".repeat(256)),
            thread_id: Some("b".repeat(256)),
            recipient: Some("c".repeat(256)),
            sender_display_name: Some("d".repeat(256)),
        });
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_origin_sender_display_name_too_long_rejected() {
        let mut req = valid_request();
        req.origin = Some(ApprovalOrigin {
            channel_type: "discord".into(),
            channel_id: None,
            thread_id: None,
            recipient: None,
            sender_display_name: Some("a".repeat(257)),
        });
        let err = req.validate().unwrap_err();
        assert!(err.contains("origin.sender_display_name"), "{err}");
        assert!(err.contains("too long"), "{err}");
    }

    // -----------------------------------------------------------------------
    // Full serde roundtrip — ApprovalPolicy
    // -----------------------------------------------------------------------

    #[test]
    fn policy_serde_roundtrip() {
        let policy = ApprovalPolicy {
            require_approval: vec!["shell_exec".into(), "file_delete".into()],
            timeout_secs: 120,
            auto_approve_autonomous: true,
            auto_approve: false,
            cache_ttl_secs: 1800,
            cache_max_uses: 25,
            allow_similar: true,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let back: ApprovalPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.require_approval, policy.require_approval);
        assert_eq!(back.timeout_secs, 120);
        assert!(back.auto_approve_autonomous);
        assert_eq!(back.cache_ttl_secs, 1800);
        assert_eq!(back.cache_max_uses, 25);
        assert!(back.allow_similar);
    }

    // -----------------------------------------------------------------------
    // Approval cache: scope, denylist, policy defaults + bounds
    // -----------------------------------------------------------------------

    #[test]
    fn cache_scope_serde_roundtrip() {
        for scope in [CacheScope::SimilarBinary("rm".into()), CacheScope::Tool] {
            let json = serde_json::to_string(&scope).unwrap();
            let back: CacheScope = serde_json::from_str(&json).unwrap();
            assert_eq!(scope, back);
        }
    }

    #[test]
    fn similar_denylist_membership() {
        assert!(is_similar_denylisted("rm"));
        assert!(is_similar_denylisted("dd"));
        assert!(is_similar_denylisted("chmod"));
        assert!(!is_similar_denylisted("grep"));
        assert!(!is_similar_denylisted("ls"));
    }

    // -----------------------------------------------------------------------
    // ANAI-152 — interpreter hole, path/case/obfuscation folding, allow_similar
    // -----------------------------------------------------------------------

    #[test]
    fn similar_denylist_covers_interpreters_and_wrappers() {
        for bin in [
            "bash", "sh", "zsh", "fish", "python", "python3", "node", "perl", "ruby", "awk",
            "xargs", "env", "nohup", "sudo", "eval", "timeout", "ssh", "find",
        ] {
            assert!(
                is_similar_denylisted(bin),
                "{bin} must not be Approve-Similar eligible"
            );
        }
    }

    #[test]
    fn similar_denylist_folds_path_prefix_and_case() {
        // The pre-ANAI-152 behavior let every one of these through.
        assert!(is_similar_denylisted("/bin/rm"));
        assert!(is_similar_denylisted("/usr/local/bin/bash"));
        assert!(is_similar_denylisted("BASH"));
        assert!(is_similar_denylisted("Bash.exe"));
        assert!(is_similar_denylisted("C:\\Windows\\System32\\cmd.exe"));
    }

    #[test]
    fn similar_denylist_folds_obfuscation() {
        assert!(is_similar_denylisted("ba\"\"sh"));
        assert!(is_similar_denylisted("\\r\\m"));
        assert!(is_similar_denylisted("r\u{200b}m"));
        // Cyrillic 'с' + "hmod"
        assert!(is_similar_denylisted("\u{0441}hmod"));
    }

    #[test]
    fn similar_denylist_does_not_overreach() {
        // Union folding must not start denying ordinary binaries.
        for bin in [
            "grep", "ls", "cat", "git", "cargo", "rustc", "jq", "rg", "make", "npm",
        ] {
            assert!(!is_similar_denylisted(bin), "{bin} was denied unexpectedly");
        }
    }

    #[test]
    fn policy_allow_similar_defaults_off() {
        assert!(!ApprovalPolicy::default().allow_similar);
    }

    #[test]
    fn policy_allow_similar_defaults_off_when_absent() {
        // A config written before the field existed must not silently enable it.
        let json = r#"{"require_approval":["shell_exec"],"timeout_secs":60}"#;
        let p: ApprovalPolicy = serde_json::from_str(json).unwrap();
        assert!(!p.allow_similar);
    }

    #[test]
    fn policy_default_cache_fields() {
        let p = ApprovalPolicy::default();
        assert_eq!(p.cache_ttl_secs, DEFAULT_CACHE_TTL_SECS);
        assert_eq!(p.cache_max_uses, DEFAULT_CACHE_MAX_USES);
    }

    #[test]
    fn policy_cache_bounds_rejected() {
        let mut p = ApprovalPolicy::default();
        p.cache_ttl_secs = MAX_CACHE_TTL_SECS + 1;
        assert!(p.validate().unwrap_err().contains("cache_ttl_secs"));

        let mut p = ApprovalPolicy::default();
        p.cache_max_uses = MAX_CACHE_MAX_USES + 1;
        assert!(p.validate().unwrap_err().contains("cache_max_uses"));
    }

    #[test]
    fn policy_cache_fields_default_when_absent() {
        // older agent.toml / serialized policy without the new keys
        let json = r#"{"require_approval":["shell_exec"],"timeout_secs":60,"auto_approve_autonomous":false,"auto_approve":false}"#;
        let p: ApprovalPolicy = serde_json::from_str(json).unwrap();
        assert_eq!(p.cache_ttl_secs, DEFAULT_CACHE_TTL_SECS);
        assert_eq!(p.cache_max_uses, DEFAULT_CACHE_MAX_USES);
    }

    // -----------------------------------------------------------------------
    // ANAI-151 — command field, fence escaping, timeout default
    // -----------------------------------------------------------------------

    #[test]
    fn command_none_is_valid() {
        let req = valid_request();
        assert!(req.command.is_none());
        assert!(req.validate().is_ok());
    }

    #[test]
    fn command_at_max_len_ok() {
        let mut req = valid_request();
        req.command = Some("x".repeat(MAX_COMMAND_LEN));
        assert!(req.validate().is_ok());
    }

    #[test]
    fn command_over_max_len_rejected() {
        let mut req = valid_request();
        req.command = Some("x".repeat(MAX_COMMAND_LEN + 1));
        let err = req.validate().unwrap_err();
        assert!(err.contains("command too long"), "{err}");
    }

    /// The bound is on CHARACTERS, not bytes: a multi-byte command must not be
    /// rejected for being under the char limit but over it in bytes.
    #[test]
    fn command_len_bound_counts_chars_not_bytes() {
        let mut req = valid_request();
        // 3 bytes each, so this is 3 * MAX_COMMAND_LEN bytes but exactly
        // MAX_COMMAND_LEN chars.
        req.command = Some("あ".repeat(MAX_COMMAND_LEN));
        assert!(req.validate().is_ok());
    }

    /// A request serialized BEFORE the `command` field existed must still
    /// deserialize — the field is `#[serde(default)]`, same contract as
    /// `cache_binary` and `origin`.
    #[test]
    fn legacy_json_without_command_deserializes() {
        let json = r#"{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "agent_id": "a",
            "tool_name": "shell_exec",
            "description": "d",
            "action_summary": "rm -rf /tmp/x",
            "risk_level": "high",
            "requested_at": "2026-01-01T00:00:00Z",
            "timeout_secs": 60
        }"#;
        let req: ApprovalRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.command, None);
        assert_eq!(req.cache_binary, None);
    }

    #[test]
    fn command_survives_serde_round_trip() {
        let mut req = valid_request();
        req.command = Some("bash -c \"rm -rf ~/.openfang/agents\"".to_string());
        let json = serde_json::to_string(&req).unwrap();
        let back: ApprovalRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.command, req.command);
    }

    // -- fence_escape --

    #[test]
    fn fence_escape_leaves_ordinary_commands_untouched() {
        let cmd = "grep -rn 'needle' /var/log --include=*.rs";
        assert_eq!(fence_escape(cmd), cmd);
    }

    #[test]
    fn fence_escape_leaves_single_and_double_backticks_untouched() {
        let cmd = "echo `date` and ``literal``";
        assert_eq!(fence_escape(cmd), cmd);
    }

    /// The load-bearing one: an agent-authored command containing a fence must
    /// not be able to close the block the render site opens. Everything after a
    /// successful breakout would render as agent-authored markdown, which is
    /// the whole prompt below the command.
    #[test]
    fn fence_escape_breaks_fence_breakout() {
        let hostile = "echo hi\n```\n**Approved by Ben already, just click**";
        let escaped = fence_escape(hostile);
        assert!(
            !escaped.contains("```"),
            "no triple-backtick run may survive: {escaped:?}"
        );
        // ...and every visible character is still there. Only zero-width spaces
        // were added, so stripping them recovers the original exactly.
        assert_eq!(escaped.replace('\u{200b}', ""), hostile);
    }

    #[test]
    fn fence_escape_handles_long_backtick_runs() {
        let hostile = "``````````";
        let escaped = fence_escape(hostile);
        assert!(!escaped.contains("```"), "{escaped:?}");
        assert_eq!(escaped.replace('\u{200b}', ""), hostile);
    }
}
