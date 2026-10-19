//! `openfang init --starter` — install the fork's starter agents.
//!
//! Copies the templates under `templates/starter/` (embedded at compile time,
//! so a built binary needs no repo checkout), fills in the placeholders, and
//! writes:
//! - `~/.openfang/RULES.md` (shared operating rules), if missing
//! - `~/.openfang/agents/<id>/agent.toml` for each chosen role
//! - `~/.openfang/workspaces/<id>/` persona files (SOUL, IDENTITY, AGENTS,
//!   TOOLS, USER, MEMORY, users/<you>.md)
//! - `~/.openfang/config.toml` set up for Claude Code + Discord, if missing;
//!   otherwise the lines to add are printed instead.
//!
//! Every write is create-only: an existing file is never overwritten, so a
//! re-run is safe and user edits survive. The one exception is an
//! `agent.toml` that is byte-identical to the stock template `openfang init`
//! ships under the same name (e.g. `assistant`): that file was never edited,
//! so it is replaced by the starter version.

use crate::{bundled_agents, ui};
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

const RULES_MD: &str = include_str!("../../../templates/starter/RULES.md");
const SOUL_MD: &str = include_str!("../../../templates/starter/common/SOUL.md");
const AGENTS_MD: &str = include_str!("../../../templates/starter/common/AGENTS.md");
const CODING_MD: &str = include_str!("../../../templates/starter/common/CODING.md");
const TOOLS_MD: &str = include_str!("../../../templates/starter/common/TOOLS.md");
const USER_MD: &str = include_str!("../../../templates/starter/common/USER.md");
const MEMORY_MD: &str = include_str!("../../../templates/starter/common/MEMORY.md");
const USER_PROFILE_MD: &str =
    include_str!("../../../templates/starter/common/users/USER_PROFILE.md");
const IDENTITY_MD: &str = include_str!("../../../templates/starter/agents/IDENTITY.md");

const DEFAULT_ROLES: &str = "assistant,guide";
const DEFAULT_PERSONA: &str = "Annabelle";

/// One starter role. `id` is also the default agent name.
struct Role {
    id: &'static str,
    title: &'static str,
    vibe: &'static str,
    purpose: &'static str,
    manifest: &'static str,
    /// Gets the Coding Discipline block in AGENTS.md.
    codes: bool,
    /// Needs `--repo` (a working repository it may write to).
    needs_repo: bool,
}

const ROLES: &[Role] = &[
    Role {
        id: "assistant",
        title: "Assistant",
        vibe: "General helper. Warm, quick, practical.",
        purpose: "General assistant: questions, writing, planning, small file tasks. The first agent to talk to.",
        manifest: include_str!("../../../templates/starter/agents/assistant/agent.toml"),
        codes: false,
        needs_repo: false,
    },
    Role {
        id: "coder",
        title: "Code",
        vibe: "Senior engineer. Reads before writing; small, reviewable diffs.",
        purpose: "Coding agent for one working repository: reads, changes, tests and commits code there.",
        manifest: include_str!("../../../templates/starter/agents/coder/agent.toml"),
        codes: true,
        needs_repo: true,
    },
    Role {
        id: "researcher",
        title: "Research",
        vibe: "Careful reader. Sources for every claim that matters.",
        purpose: "Research agent: web search, reading and sourced summaries. No shell.",
        manifest: include_str!("../../../templates/starter/agents/researcher/agent.toml"),
        codes: false,
        needs_repo: false,
    },
    Role {
        id: "guide",
        title: "Guide",
        vibe: "Patient explainer of this OpenFang install. Read-only.",
        purpose: "Answers questions about this OpenFang install (binding channels, why an agent has no tools, what a setting does). Read-only.",
        manifest: include_str!("../../../templates/starter/agents/guide/agent.toml"),
        codes: false,
        needs_repo: false,
    },
    Role {
        id: "alpha",
        title: "Alpha",
        vibe: "Lead engineer. Holds the map of the fleet; settles shared-file disputes.",
        purpose: "Lead agent: knows what every other agent is doing, settles disputes over shared files, escalates irreversible calls to the human, and ships code itself.",
        manifest: include_str!("../../../templates/starter/agents/alpha/agent.toml"),
        codes: true,
        needs_repo: true,
    },
];

/// Flags for `openfang init --starter`. Anything left unset is asked for in a
/// terminal, or defaulted (with a note) when not interactive.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct StarterArgs {
    /// Install the starter agents (Claude Code + Discord) instead of the stock templates.
    #[arg(long)]
    pub starter: bool,
    /// Comma-separated roles: assistant, coder, researcher, guide, alpha.
    /// Default: assistant,guide. alpha needs at least two other roles.
    #[arg(long, value_name = "ROLES", requires = "starter")]
    pub roles: Option<String>,
    /// Your name, as the agents should address you.
    #[arg(long, value_name = "NAME", requires = "starter")]
    pub user_name: Option<String>,
    /// The name your agents go by (they share one voice). Default: Annabelle.
    #[arg(long, value_name = "NAME", requires = "starter")]
    pub persona: Option<String>,
    /// Your timezone, e.g. America/Los_Angeles. Default: this Mac's setting.
    #[arg(long, value_name = "TZ", requires = "starter")]
    pub timezone: Option<String>,
    /// Working repository for coder/alpha (they get write access to it).
    #[arg(long, value_name = "PATH", requires = "starter")]
    pub repo: Option<PathBuf>,
    /// Bind a Discord channel to a role: ROLE=CHANNEL_ID (repeatable).
    #[arg(long = "channel", value_name = "ROLE=CHANNEL_ID", requires = "starter")]
    pub channels: Vec<String>,
    /// Your Discord user ID; only you can talk to the bot. Recommended.
    #[arg(long, value_name = "ID", requires = "starter")]
    pub discord_user: Option<String>,
    /// Show what would be written; change nothing.
    #[arg(long, requires = "starter")]
    pub dry_run: bool,
}

/// Fully resolved inputs. Built by [`resolve`], consumed by [`plan`].
#[derive(Debug, Clone)]
struct Inputs {
    roles: Vec<&'static str>,
    user_name: String,
    persona: String,
    timezone: String,
    repo: Option<PathBuf>,
    channels: Vec<(String, String)>,
    discord_user: Option<String>,
    openfang_repo: PathBuf,
}

#[derive(Debug, PartialEq)]
enum Action {
    Create,
    /// Replaces an untouched stock template of the same name.
    ReplaceStock,
    KeepExisting,
}

#[derive(Debug)]
struct PlannedFile {
    path: PathBuf,
    content: String,
    action: Action,
}

pub fn run(openfang_dir: &Path, args: &StarterArgs) {
    let inputs = match resolve(args) {
        Ok(i) => i,
        Err(e) => {
            ui::error(&e);
            std::process::exit(64);
        }
    };
    let files = match plan(openfang_dir, &inputs) {
        Ok(f) => f,
        Err(e) => {
            ui::error(&e);
            std::process::exit(64);
        }
    };

    ui::blank();
    for f in &files {
        let tag = match f.action {
            Action::Create => "create ",
            Action::ReplaceStock => "replace",
            Action::KeepExisting => "keep   ",
        };
        println!("  {tag}  {}", f.path.display());
    }
    let config_path = openfang_dir.join("config.toml");
    let config_exists = config_path.exists();
    ui::blank();

    if args.dry_run {
        ui::hint("Dry run: nothing written.");
        if config_exists {
            print_config_snippet(&inputs);
        } else {
            println!("  create   {}", config_path.display());
        }
        return;
    }

    let mut failed = 0;
    for f in files.iter().filter(|f| f.action != Action::KeepExisting) {
        if let Err(e) = write_file(f) {
            ui::error(&format!("{}: {e}", f.path.display()));
            failed += 1;
        }
    }

    if config_exists {
        ui::check_ok(&format!("Config already exists: {}", config_path.display()));
        print_config_snippet(&inputs);
    } else {
        match std::fs::write(&config_path, render_config(&inputs)) {
            Ok(()) => {
                crate::restrict_file_permissions(&config_path);
                ui::success(&format!("Created: {}", config_path.display()));
            }
            Err(e) => {
                ui::error(&format!("{}: {e}", config_path.display()));
                failed += 1;
            }
        }
    }

    let kept = files
        .iter()
        .filter(|f| f.action == Action::KeepExisting)
        .count();
    if kept > 0 {
        ui::hint(&format!(
            "{kept} existing file(s) kept as they are. Delete one and re-run to get the starter version."
        ));
    }
    if failed > 0 {
        std::process::exit(1);
    }
    ui::success(&format!(
        "Starter agents installed: {}",
        inputs.roles.join(", ")
    ));
    ui::next_steps(&[
        "Put your Discord bot token in ~/.openfang/secrets.env as DISCORD_BOT_TOKEN=...",
        "Install and start the daemon:  openfang service install",
        "Say hello in the Discord channel you bound to `assistant`",
        "Full walkthrough: docs/fork/quickstart.md",
    ]);
}

fn write_file(f: &PlannedFile) -> std::io::Result<()> {
    if let Some(parent) = f.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match f.action {
        Action::Create => {
            let mut out = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&f.path)?;
            out.write_all(f.content.as_bytes())
        }
        Action::ReplaceStock => std::fs::write(&f.path, &f.content),
        Action::KeepExisting => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Input resolution
// ---------------------------------------------------------------------------

fn resolve(args: &StarterArgs) -> Result<Inputs, String> {
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let ask = |label: &str, default: Option<&str>| -> Option<String> {
        if !interactive {
            return default.map(str::to_string);
        }
        prompt(label, default)
    };

    let roles_raw = match &args.roles {
        Some(r) => r.clone(),
        None => ask(
            "Roles (assistant, coder, researcher, guide, alpha)",
            Some(DEFAULT_ROLES),
        )
        .unwrap_or_else(|| DEFAULT_ROLES.into()),
    };
    let roles = parse_roles(&roles_raw)?;

    let user_name = match &args.user_name {
        Some(n) => n.clone(),
        None => ask("Your name", None)
            .ok_or("--user-name is required when not running in a terminal")?,
    };
    let persona = match &args.persona {
        Some(p) => p.clone(),
        None => ask("Your agents' name", Some(DEFAULT_PERSONA))
            .unwrap_or_else(|| DEFAULT_PERSONA.into()),
    };
    let tz_default = local_timezone();
    let timezone = match &args.timezone {
        Some(t) => t.clone(),
        None => ask("Your timezone", Some(&tz_default)).unwrap_or(tz_default),
    };

    let needs_repo = roles
        .iter()
        .any(|id| role(id).is_some_and(|r| r.needs_repo));
    let repo = match (&args.repo, needs_repo) {
        (Some(p), _) => Some(expand_home(p)),
        (None, true) => Some(expand_home(Path::new(
            &ask("Working repository for coder/alpha (path)", None)
                .ok_or("--repo is required for coder and alpha")?,
        ))),
        (None, false) => None,
    };

    let mut channels = Vec::new();
    for c in &args.channels {
        channels.push(parse_channel(c, &roles)?);
    }
    if channels.is_empty() && interactive && roles.contains(&"assistant") {
        if let Some(id) = prompt("Discord channel ID for `assistant` (blank to skip)", None) {
            channels.push(parse_channel(&format!("assistant={id}"), &roles)?);
        }
    }
    let discord_user = match &args.discord_user {
        Some(u) => Some(u.clone()),
        None if interactive => prompt("Your Discord user ID (blank to skip)", None),
        None => None,
    };
    if let Some(u) = &discord_user {
        if !is_snowflake(u) {
            return Err(format!("Discord user ID should be digits only, got {u:?}"));
        }
    }

    for field in [&user_name, &persona, &timezone] {
        validate_text(field)?;
    }

    Ok(Inputs {
        roles,
        user_name,
        persona,
        timezone,
        repo,
        channels,
        discord_user,
        openfang_repo: openfang_repo_guess(),
    })
}

fn prompt(label: &str, default: Option<&str>) -> Option<String> {
    match default {
        Some(d) => print!("  {label} [{d}]: "),
        None => print!("  {label}: "),
    }
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return default.map(str::to_string);
    }
    let line = line.trim();
    if line.is_empty() {
        default.map(str::to_string)
    } else {
        Some(line.to_string())
    }
}

fn role(id: &str) -> Option<&'static Role> {
    ROLES.iter().find(|r| r.id == id)
}

fn parse_roles(raw: &str) -> Result<Vec<&'static str>, String> {
    let mut out: Vec<&'static str> = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let r = role(part).ok_or_else(|| {
            format!("unknown role {part:?}; choose from assistant, coder, researcher, guide, alpha")
        })?;
        if !out.contains(&r.id) {
            out.push(r.id);
        }
    }
    if out.is_empty() {
        return Err("no roles chosen".into());
    }
    if out.contains(&"alpha") && out.len() < 3 {
        return Err(
            "alpha coordinates other agents; pick at least two other roles with it (e.g. assistant,coder,alpha)"
                .into(),
        );
    }
    Ok(out)
}

fn parse_channel(raw: &str, roles: &[&str]) -> Result<(String, String), String> {
    let (r, id) = raw
        .split_once('=')
        .ok_or_else(|| format!("--channel wants ROLE=CHANNEL_ID, got {raw:?}"))?;
    let (r, id) = (r.trim(), id.trim());
    if !roles.contains(&r) {
        return Err(format!(
            "--channel {raw:?}: role {r:?} isn't being installed"
        ));
    }
    if !is_snowflake(id) {
        return Err(format!(
            "--channel {raw:?}: a Discord channel ID is digits only (right-click the channel > Copy Channel ID)"
        ));
    }
    Ok((r.to_string(), id.to_string()))
}

fn is_snowflake(s: &str) -> bool {
    (15..=21).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit())
}

/// Names end up inside TOML strings and markdown. Refuse the characters that
/// would break out of a basic TOML string or inject template syntax.
fn validate_text(s: &str) -> Result<(), String> {
    if s.trim().is_empty() {
        return Err("empty value".into());
    }
    if s.chars()
        .any(|c| c == '"' || c == '\\' || c.is_control() || c == '{' || c == '}')
    {
        return Err(format!(
            "{s:?} contains a character that isn't allowed here (quotes, backslashes, braces, control characters)"
        ));
    }
    Ok(())
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "user".into()
    } else {
        out
    }
}

fn expand_home(p: &Path) -> PathBuf {
    match (p.strip_prefix("~"), dirs::home_dir()) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => p.to_path_buf(),
    }
}

/// macOS: /etc/localtime -> /var/db/timezone/zoneinfo/America/Los_Angeles.
fn local_timezone() -> String {
    std::fs::read_link("/etc/localtime")
        .ok()
        .and_then(|p| {
            let s = p.to_string_lossy().into_owned();
            s.split_once("zoneinfo/").map(|(_, tz)| tz.to_string())
        })
        .unwrap_or_else(|| "UTC".into())
}

/// Where the guide agent should read the fork's docs. The binary records the
/// repo it was built from; fall back to the conventional clone location.
fn openfang_repo_guess() -> PathBuf {
    let built_from = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .ok();
    match built_from {
        Some(p) if p.join("docs").is_dir() => p,
        _ => expand_home(Path::new("~/openfang")),
    }
}

// ---------------------------------------------------------------------------
// Planning + rendering
// ---------------------------------------------------------------------------

fn render(template: &str, vars: &[(&str, String)]) -> String {
    let mut out = template.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{{{k}}}}}"), v);
    }
    out
}

fn plan(openfang_dir: &Path, inputs: &Inputs) -> Result<Vec<PlannedFile>, String> {
    let home = openfang_dir.to_string_lossy().into_owned();
    let user_slug = slug(&inputs.user_name);
    let stock: Vec<(&str, &str)> = bundled_agents::bundled_agents();
    let mut files = Vec::new();

    files.push(planned(
        openfang_dir.join("RULES.md"),
        RULES_MD.to_string(),
        None,
    ));

    for &id in &inputs.roles {
        let r = role(id).expect("roles are validated");
        let workspace = openfang_dir.join("workspaces").join(id);
        let channel = inputs.channels.iter().find(|(role, _)| role == id);
        let fleet = inputs
            .roles
            .iter()
            .filter(|&&o| o != id)
            .map(|o| format!("- `{o}`: {}", role(o).map(|r| r.purpose).unwrap_or("")))
            .collect::<Vec<_>>()
            .join("\n");
        let vars: Vec<(&str, String)> = vec![
            ("AGENT_ID", id.to_string()),
            ("ROLE", id.to_string()),
            ("ROLE_TITLE", r.title.to_string()),
            ("ROLE_VIBE", r.vibe.to_string()),
            ("ROLE_PURPOSE", r.purpose.to_string()),
            ("PERSONA_NAME", inputs.persona.clone()),
            ("USER_NAME", inputs.user_name.clone()),
            ("USER_SLUG", user_slug.clone()),
            ("TIMEZONE", inputs.timezone.clone()),
            ("OPENFANG_HOME", home.clone()),
            (
                "OPENFANG_REPO",
                inputs.openfang_repo.to_string_lossy().into_owned(),
            ),
            ("WORKSPACE", workspace.to_string_lossy().into_owned()),
            (
                "CODE_REPO",
                inputs
                    .repo
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
            (
                "CHANNEL_LABEL",
                if channel.is_some() {
                    "the Discord channel bound to this agent in config.toml".into()
                } else {
                    "none yet (see docs/fork/quickstart.md, \"Bind a channel\")".into()
                },
            ),
            ("FLEET", fleet),
            (
                "CODING_DISCIPLINE",
                if r.codes {
                    CODING_MD.to_string()
                } else {
                    String::new()
                },
            ),
        ];

        let manifest = render(r.manifest, &vars);
        check_rendered(&manifest, &format!("{id}/agent.toml"))?;
        toml::from_str::<openfang_types::agent::AgentManifest>(&manifest)
            .map_err(|e| format!("starter template {id}/agent.toml doesn't parse: {e}"))?;
        let stock_toml = stock.iter().find(|(n, _)| *n == id).map(|(_, c)| *c);
        files.push(planned(
            openfang_dir.join("agents").join(id).join("agent.toml"),
            manifest,
            stock_toml,
        ));

        for (name, tpl) in [
            ("SOUL.md", SOUL_MD),
            ("IDENTITY.md", IDENTITY_MD),
            ("AGENTS.md", AGENTS_MD),
            ("TOOLS.md", TOOLS_MD),
            ("USER.md", USER_MD),
            ("MEMORY.md", MEMORY_MD),
        ] {
            let content = render(tpl, &vars);
            check_rendered(&content, &format!("{id}/{name}"))?;
            files.push(planned(workspace.join(name), content, None));
        }
        let profile = render(USER_PROFILE_MD, &vars);
        check_rendered(&profile, &format!("{id}/users/{user_slug}.md"))?;
        files.push(planned(
            workspace.join("users").join(format!("{user_slug}.md")),
            profile,
            None,
        ));
    }
    Ok(files)
}

fn planned(path: PathBuf, content: String, stock: Option<&str>) -> PlannedFile {
    let action = match std::fs::read_to_string(&path) {
        Err(_) if !path.exists() => Action::Create,
        Ok(existing) if stock.is_some_and(|s| s == existing) => Action::ReplaceStock,
        _ => Action::KeepExisting,
    };
    PlannedFile {
        path,
        content,
        action,
    }
}

fn check_rendered(content: &str, what: &str) -> Result<(), String> {
    match content.find("{{") {
        Some(i) => {
            let end = content[i..].find("}}").map_or(content.len(), |e| i + e + 2);
            Err(format!(
                "starter template {what} has an unfilled placeholder: {}",
                &content[i..end]
            ))
        }
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// config.toml
// ---------------------------------------------------------------------------

/// Global shell commands every agent may run with no prompt. Inspection only.
/// `ps` is deliberately absent: on macOS `ps -E`/`pgrep -lf` can print other
/// processes' environment variables, secrets included.
const SAFE_BINS: &[&str] = &[
    "ls", "cat", "head", "tail", "wc", "sort", "uniq", "cut", "tr", "grep", "rg", "jq", "stat",
    "diff", "file", "date", "echo", "printf", "basename", "dirname", "pwd", "which", "true",
    "false", "sleep",
];

fn discord_and_bindings(inputs: &Inputs) -> String {
    let quoted = |v: &[&str]| {
        v.iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let channel_ids: Vec<&str> = inputs.channels.iter().map(|(_, c)| c.as_str()).collect();
    let mut out = String::from("[channels.discord]\nbot_token_env = \"DISCORD_BOT_TOKEN\"\n");
    match &inputs.discord_user {
        Some(u) => out.push_str(&format!("allowed_users = [\"{u}\"]\n")),
        None => out.push_str(
            "# Recommended: only you can talk to the bot.\n# allowed_users = [\"<your Discord user ID>\"]\n",
        ),
    }
    out.push_str("# Channels where agents answer without being @mentioned.\n");
    out.push_str(&format!(
        "free_response_channels = [{}]\n",
        quoted(&channel_ids)
    ));
    if inputs.roles.contains(&"assistant") {
        out.push_str("default_agent = \"assistant\"\n");
    }
    for (r, c) in &inputs.channels {
        out.push_str(&format!(
            "\n[[bindings]]\nagent = \"{r}\"\nmatch_rule = {{ channel = \"discord\", channel_id = \"{c}\" }}\n"
        ));
    }
    out
}

fn render_config(inputs: &Inputs) -> String {
    let bins = SAFE_BINS
        .iter()
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"# OpenFang configuration, written by `openfang init --starter`.
# Fork docs: docs/fork/quickstart.md. Changes need `openfang service restart`.

api_listen = "127.0.0.1:4200"

# Agents run on your Claude Code subscription through the `claude` CLI.
# No API key needed; `claude` must be installed and logged in.
[default_model]
provider = "claude-code"
model = "sonnet"
api_key_env = ""

[memory]
decay_rate = 0.05

# Global shell policy. These inspection commands run for every agent with
# shell access, with no prompt. Per-agent extras go in each agent.toml
# (trusted_commands / allowed_commands).
[exec_policy]
mode = "allowlist"
safe_bins = [{bins}]

{discord}"#,
        discord = discord_and_bindings(inputs)
    )
}

fn print_config_snippet(inputs: &Inputs) {
    ui::hint("Your config.toml already exists, so it wasn't changed. For Claude Code + Discord it needs:");
    println!();
    println!("  [default_model]");
    println!("  provider = \"claude-code\"");
    println!("  model = \"sonnet\"");
    println!();
    for line in discord_and_bindings(inputs).lines() {
        println!("  {line}");
    }
    println!();
    ui::hint("If any of these sections already exist, edit them rather than adding a second copy.");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(roles: &[&'static str]) -> Inputs {
        Inputs {
            roles: roles.to_vec(),
            user_name: "Sam Rivera".into(),
            persona: "Annabelle".into(),
            timezone: "America/Chicago".into(),
            repo: Some(PathBuf::from("/tmp/code")),
            channels: vec![("assistant".into(), "123456789012345678".into())],
            discord_user: Some("987654321098765432".into()),
            openfang_repo: PathBuf::from("/tmp/openfang"),
        }
    }

    const ALL: &[&str] = &["assistant", "coder", "researcher", "guide", "alpha"];

    #[test]
    fn every_role_renders_and_parses() {
        let dir = tempfile::tempdir().unwrap();
        let files = plan(dir.path(), &inputs(ALL)).expect("plan");
        // RULES.md + 5 roles x (agent.toml + 6 persona files + profile)
        assert_eq!(files.len(), 1 + 5 * 8);
        for f in &files {
            assert!(!f.content.contains("{{"), "{} unfilled", f.path.display());
            assert_eq!(f.action, Action::Create);
        }
    }

    #[test]
    fn coding_block_only_for_coding_roles() {
        let dir = tempfile::tempdir().unwrap();
        let files = plan(dir.path(), &inputs(ALL)).unwrap();
        for id in ALL {
            let agents_md = files
                .iter()
                .find(|f| f.path.ends_with(format!("workspaces/{id}/AGENTS.md")))
                .unwrap();
            let has = agents_md.content.contains("## Coding Discipline");
            assert_eq!(has, role(id).unwrap().codes, "{id}");
        }
    }

    #[test]
    fn alpha_fleet_lists_the_others_not_itself() {
        let dir = tempfile::tempdir().unwrap();
        let files = plan(dir.path(), &inputs(&["assistant", "coder", "alpha"])).unwrap();
        let alpha = files
            .iter()
            .find(|f| f.path.ends_with("agents/alpha/agent.toml"))
            .unwrap();
        assert!(alpha.content.contains("- `assistant`:"));
        assert!(alpha.content.contains("- `coder`:"));
        assert!(!alpha.content.contains("- `alpha`:"));
    }

    #[test]
    fn alpha_needs_two_others() {
        assert!(parse_roles("alpha").is_err());
        assert!(parse_roles("assistant,alpha").is_err());
        assert!(parse_roles("assistant,coder,alpha").is_ok());
    }

    #[test]
    fn roles_dedupe_and_reject_unknown() {
        assert_eq!(
            parse_roles("guide, guide,assistant").unwrap(),
            vec!["guide", "assistant"]
        );
        assert!(parse_roles("assistant,wizard").is_err());
        assert!(parse_roles(" , ").is_err());
    }

    #[test]
    fn existing_files_are_kept_and_untouched_stock_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let stock = bundled_agents::bundled_agents()
            .into_iter()
            .find(|(n, _)| *n == "assistant")
            .unwrap()
            .1;
        let agent = dir.path().join("agents/assistant/agent.toml");
        std::fs::create_dir_all(agent.parent().unwrap()).unwrap();
        std::fs::write(&agent, stock).unwrap();
        let soul = dir.path().join("workspaces/assistant/SOUL.md");
        std::fs::create_dir_all(soul.parent().unwrap()).unwrap();
        std::fs::write(&soul, "my edits").unwrap();

        let files = plan(dir.path(), &inputs(&["assistant"])).unwrap();
        let get = |p: &Path| files.iter().find(|f| f.path == p).unwrap();
        assert_eq!(get(&agent).action, Action::ReplaceStock);
        assert_eq!(get(&soul).action, Action::KeepExisting);

        // An edited stock manifest is kept.
        std::fs::write(&agent, format!("{stock}\n# edited")).unwrap();
        let files = plan(dir.path(), &inputs(&["assistant"])).unwrap();
        assert_eq!(
            files.iter().find(|f| f.path == agent).unwrap().action,
            Action::KeepExisting
        );
    }

    #[test]
    fn create_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.md");
        std::fs::write(&p, "mine").unwrap();
        let f = PlannedFile {
            path: p.clone(),
            content: "theirs".into(),
            action: Action::Create,
        };
        assert!(write_file(&f).is_err());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "mine");
    }

    #[test]
    fn rendered_config_parses_with_bindings() {
        let cfg = render_config(&inputs(&["assistant", "guide"]));
        let parsed: openfang_types::config::KernelConfig =
            toml::from_str(&cfg).expect("config parses");
        assert_eq!(parsed.default_model.provider, "claude-code");
        assert_eq!(parsed.bindings.len(), 1);
        assert_eq!(parsed.bindings[0].agent, "assistant");
        assert!(parsed.exec_policy.safe_bins.iter().any(|b| b == "rg"));
        assert!(!parsed.exec_policy.safe_bins.iter().any(|b| b == "ps"));
        let discord = parsed.channels.discord.expect("discord section");
        assert_eq!(discord.allowed_users, vec!["987654321098765432"]);
        assert_eq!(discord.free_response_channels, vec!["123456789012345678"]);
    }

    #[test]
    fn channel_and_name_validation() {
        let roles = ["assistant", "guide"];
        assert!(parse_channel("assistant=123456789012345678", &roles).is_ok());
        assert!(parse_channel("coder=123456789012345678", &roles).is_err());
        assert!(parse_channel("assistant=#general", &roles).is_err());
        assert!(parse_channel("assistant", &roles).is_err());
        assert!(validate_text("Sam").is_ok());
        assert!(validate_text("Sam\" injected = \"x").is_err());
        assert!(validate_text("{{FLEET}}").is_err());
        assert!(validate_text("  ").is_err());
    }

    #[test]
    fn slug_is_filename_safe() {
        assert_eq!(slug("Sam Rivera"), "sam-rivera");
        assert_eq!(slug("  José!! "), "jos");
        assert_eq!(slug("!!!"), "user");
    }
}
