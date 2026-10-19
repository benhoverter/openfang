//! `openfang service` — run the daemon as a macOS launchd user agent.
//!
//! launchd (macOS's service manager) keeps the daemon alive across crashes
//! and logins. The job is described by a property-list file in
//! `~/Library/LaunchAgents/<label>.plist`. These commands generate that file,
//! load/unload it, and report its state, so nobody has to hand-write XML.
//!
//! Lifecycle notes that drive the design:
//! - The generated job sets `KeepAlive`, so killing the daemon (or
//!   `openfang stop`) only bounces it. `service stop` therefore unloads the
//!   job (`launchctl bootout`) — that is the only way it stays down.
//! - launchd reads the plist once, at load time. A plain `kickstart -k`
//!   relaunches the process with the OLD plist. So `service restart` keeps a
//!   snapshot of the plist it last loaded and does a full unload + load when
//!   the on-disk file differs (or when no snapshot exists).
//! - Bridge contract with `claude_code.rs`: install writes
//!   `OPENFANG_BRIDGE_ENABLED=1` plus an explicit `OPENFANG_BRIDGE_BIN`, which
//!   puts service installs in the fail-closed row of the bridge gate (missing
//!   bridge binary => Claude Code native tools stay denied). Install refuses to
//!   write that row when the bridge binary is absent; `--no-bridge` writes
//!   `OPENFANG_BRIDGE_ENABLED=0` instead.
//! - Secrets do not belong in the plist (it is a world-readable-by-default
//!   plaintext file). They go in `~/.openfang/secrets.env`, which the daemon
//!   loads itself at startup. `--env` rejects secret-shaped keys.

use crate::ui;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub const DEFAULT_LABEL: &str = "ai.openfang.daemon";
const BRIDGE_BIN_NAME: &str = "openfang-mcp-bridge";
const ENV_BRIDGE_ENABLED: &str = "OPENFANG_BRIDGE_ENABLED";
const ENV_BRIDGE_BIN: &str = "OPENFANG_BRIDGE_BIN";
const DEFAULT_RUST_LOG: &str = "info";

#[derive(clap::Subcommand)]
pub enum ServiceCommands {
    /// Generate the launchd plist and load it (starts the daemon).
    Install {
        #[command(flatten)]
        opts: InstallOpts,
    },
    /// Unload the job and remove its plist (a backup copy is kept).
    Uninstall {
        /// launchd job label.
        #[arg(long, default_value = DEFAULT_LABEL)]
        label: String,
    },
    /// Load the job if needed and start the daemon.
    Start {
        #[arg(long, default_value = DEFAULT_LABEL)]
        label: String,
    },
    /// Unload the job so the daemon stays down (until `service start` or next login).
    Stop {
        #[arg(long, default_value = DEFAULT_LABEL)]
        label: String,
    },
    /// Restart the daemon; does a full unload + load if the plist changed.
    Restart {
        #[arg(long, default_value = DEFAULT_LABEL)]
        label: String,
        /// Always do the full unload + load, even if the plist looks unchanged.
        #[arg(long)]
        full: bool,
    },
    /// Show whether the job is installed/loaded, its PID, last exit, and bridge settings.
    Status {
        #[arg(long, default_value = DEFAULT_LABEL)]
        label: String,
        /// Output as JSON for scripting.
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Args, Debug, Clone)]
pub struct InstallOpts {
    /// launchd job label (also the plist filename). Use a different label to
    /// test without touching an existing install.
    #[arg(long, default_value = DEFAULT_LABEL)]
    pub label: String,
    /// Daemon binary to run. Default: ~/.openfang/bin/openfang if present,
    /// otherwise the binary running this command.
    #[arg(long)]
    pub bin: Option<PathBuf>,
    /// RUST_LOG filter for the daemon.
    #[arg(long, default_value = DEFAULT_RUST_LOG)]
    pub rust_log: String,
    /// Extra environment variable for the daemon, KEY=VALUE (repeatable).
    /// Not for secrets: put those in ~/.openfang/secrets.env.
    #[arg(long = "env", value_name = "KEY=VALUE")]
    pub env: Vec<String>,
    /// Don't wire the MCP bridge (writes OPENFANG_BRIDGE_ENABLED=0).
    #[arg(long)]
    pub no_bridge: bool,
    /// Replace an existing plist (the old one is backed up first).
    #[arg(long)]
    pub force: bool,
    /// Print the plist that would be written; change nothing.
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(cmd: ServiceCommands) {
    let result = match cmd {
        ServiceCommands::Install { opts } => install(&opts),
        ServiceCommands::Uninstall { label } => uninstall(&label),
        ServiceCommands::Start { label } => start(&label),
        ServiceCommands::Stop { label } => stop(&label),
        ServiceCommands::Restart { label, full } => restart(&label, full),
        ServiceCommands::Status { label, json } => status(&label, json),
    };
    if let Err(e) = result {
        ui::error_with_fix(&e.msg, &e.fix);
        std::process::exit(1);
    }
}

#[derive(Debug)]
struct SvcError {
    msg: String,
    fix: String,
}

fn err(msg: impl Into<String>, fix: impl Into<String>) -> SvcError {
    SvcError {
        msg: msg.into(),
        fix: fix.into(),
    }
}

type SvcResult<T> = Result<T, SvcError>;

// ---------------------------------------------------------------------------
// Plist generation (pure)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct PlistSpec {
    pub label: String,
    pub program: PathBuf,
    pub working_dir: PathBuf,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
    pub env: BTreeMap<String, String>,
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

fn string_el(s: &str) -> String {
    format!("<string>{}</string>", xml_escape(s))
}

pub fn render_plist(spec: &PlistSpec) -> String {
    let mut env_xml = String::new();
    for (k, v) in &spec.env {
        env_xml.push_str(&format!(
            "        <key>{}</key>\n        {}\n",
            xml_escape(k),
            string_el(v)
        ));
    }
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Generated by `openfang service install`. Regenerate rather than hand-edit;
     if you do hand-edit, `openfang service restart` will notice and reload. -->
<plist version="1.0">
<dict>
    <key>Label</key>
    {label}
    <key>ProgramArguments</key>
    <array>
        {program}
        <string>start</string>
    </array>
    <key>WorkingDirectory</key>
    {wd}
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>Umask</key>
    <integer>63</integer>
    <key>StandardOutPath</key>
    {out}
    <key>StandardErrorPath</key>
    {errlog}
    <key>EnvironmentVariables</key>
    <dict>
{env_xml}    </dict>
</dict>
</plist>
"#,
        label = string_el(&spec.label),
        program = string_el(&spec.program.to_string_lossy()),
        wd = string_el(&spec.working_dir.to_string_lossy()),
        out = string_el(&spec.stdout_log.to_string_lossy()),
        errlog = string_el(&spec.stderr_log.to_string_lossy()),
    )
}

fn valid_env_key(k: &str) -> bool {
    let mut chars = k.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// Keys that look like credentials. The plist is plaintext on disk; secrets
/// belong in secrets.env.
fn secret_shaped(k: &str) -> bool {
    let u = k.to_ascii_uppercase();
    ["_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY", "_KEY"]
        .iter()
        .any(|s| u.ends_with(s))
}

/// Parse `--env KEY=VALUE` entries, rejecting reserved and secret-shaped keys.
fn parse_extra_env(entries: &[String]) -> SvcResult<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for e in entries {
        let (k, v) = e.split_once('=').ok_or_else(|| {
            err(
                format!("--env '{e}' is not KEY=VALUE"),
                "use the form --env NAME=value",
            )
        })?;
        let k = k.trim();
        if !valid_env_key(k) {
            return Err(err(
                format!("--env key '{k}' is not a valid variable name"),
                "letters, digits and underscores only; must not start with a digit",
            ));
        }
        if k == ENV_BRIDGE_ENABLED || k == ENV_BRIDGE_BIN {
            return Err(err(
                format!("{k} is managed by `service install`"),
                "the bridge is wired automatically; pass --no-bridge to turn it off",
            ));
        }
        if matches!(k, "HOME" | "PATH" | "RUST_LOG") {
            return Err(err(
                format!("{k} is set by `service install`"),
                "use --rust-log for RUST_LOG; HOME and PATH come from your shell",
            ));
        }
        if secret_shaped(k) {
            return Err(err(
                format!("{k} looks like a secret; the plist is a plaintext file"),
                "put it in ~/.openfang/secrets.env instead (the daemon loads it at startup)",
            ));
        }
        out.insert(k.to_string(), v.to_string());
    }
    Ok(out)
}

/// Bridge env for the plist: the vars to set, or an error when the bridge is
/// wanted but its binary is missing (we refuse rather than write the
/// fail-closed row and hand the user toolless agents).
fn bridge_env(
    program: &Path,
    no_bridge: bool,
    exists: impl Fn(&Path) -> bool,
) -> SvcResult<Vec<(String, String)>> {
    if no_bridge {
        return Ok(vec![(ENV_BRIDGE_ENABLED.into(), "0".into())]);
    }
    let dir = program.parent().unwrap_or_else(|| Path::new("."));
    let bridge = dir.join(BRIDGE_BIN_NAME);
    if !exists(&bridge) {
        return Err(err(
            format!(
                "MCP bridge binary not found at {} — without it, Claude Code agents would get no tools",
                bridge.display()
            ),
            format!(
                "build and install it next to the daemon (`cargo build --release -p openfang-mcp-bridge`, then copy target/release/{BRIDGE_BIN_NAME} into {}), or pass --no-bridge",
                dir.display()
            ),
        ));
    }
    Ok(vec![
        (ENV_BRIDGE_ENABLED.into(), "1".into()),
        (ENV_BRIDGE_BIN.into(), bridge.to_string_lossy().into_owned()),
    ])
}

fn openfang_home() -> PathBuf {
    if let Ok(h) = std::env::var("OPENFANG_HOME") {
        return PathBuf::from(h);
    }
    user_home().join(".openfang")
}

fn user_home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(std::env::temp_dir)
}

fn resolve_program(explicit: Option<&Path>, of_home: &Path) -> SvcResult<PathBuf> {
    let p = match explicit {
        Some(p) => p.to_path_buf(),
        None => {
            let installed = of_home.join("bin").join("openfang");
            if installed.is_file() {
                installed
            } else {
                std::env::current_exe().map_err(|e| {
                    err(
                        format!("can't locate the openfang binary: {e}"),
                        "pass --bin /path/to/openfang",
                    )
                })?
            }
        }
    };
    let p = p.canonicalize().map_err(|e| {
        err(
            format!("daemon binary {}: {e}", p.display()),
            "pass --bin with the path to an existing openfang binary",
        )
    })?;
    if !p.is_file() {
        return Err(err(
            format!("{} is not a file", p.display()),
            "pass --bin with the path to the openfang binary",
        ));
    }
    Ok(p)
}

/// PATH for the daemon: the installing shell's PATH (so `claude`, node, etc.
/// resolve the same way they do for you), with the daemon's own dir first.
fn daemon_path(program: &Path, shell_path: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(d) = program.parent() {
        parts.push(d.to_string_lossy().into_owned());
    }
    for p in shell_path.split(':') {
        if !p.is_empty() && !parts.iter().any(|x| x == p) {
            parts.push(p.to_string());
        }
    }
    for p in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"] {
        if !parts.iter().any(|x| x == p) {
            parts.push(p.to_string());
        }
    }
    parts.join(":")
}

fn find_on_path(bin: &str, path: &str) -> Option<PathBuf> {
    path.split(':')
        .filter(|p| !p.is_empty())
        .map(|p| Path::new(p).join(bin))
        .find(|p| p.is_file())
}

fn build_spec(opts: &InstallOpts) -> SvcResult<PlistSpec> {
    let of_home = openfang_home();
    let program = resolve_program(opts.bin.as_deref(), &of_home)?;
    let shell_path = std::env::var("PATH").unwrap_or_default();

    let mut env = parse_extra_env(&opts.env)?;
    env.insert("HOME".into(), user_home().to_string_lossy().into_owned());
    env.insert("PATH".into(), daemon_path(&program, &shell_path));
    env.insert("RUST_LOG".into(), opts.rust_log.clone());
    if std::env::var_os("OPENFANG_HOME").is_some() {
        env.insert(
            "OPENFANG_HOME".into(),
            of_home.to_string_lossy().into_owned(),
        );
    }
    for (k, v) in bridge_env(&program, opts.no_bridge, |p| p.is_file())? {
        env.insert(k, v);
    }

    Ok(PlistSpec {
        label: opts.label.clone(),
        program,
        working_dir: of_home.clone(),
        stdout_log: of_home.join("daemon.stdout.log"),
        stderr_log: of_home.join("daemon.stderr.log"),
        env,
    })
}

// ---------------------------------------------------------------------------
// launchctl status parsing + restart planning (pure)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
pub struct JobState {
    pub state: Option<String>,
    pub pid: Option<u32>,
    pub last_exit: Option<String>,
}

/// Parse the top-level fields of `launchctl print gui/<uid>/<label>`.
/// Only lines indented by exactly one tab are top-level; nested blocks
/// (endpoints, environment, ...) also contain `state = ` lines.
pub fn parse_launchctl_print(out: &str) -> JobState {
    let mut js = JobState::default();
    for line in out.lines() {
        let Some(rest) = line.strip_prefix('\t') else {
            continue;
        };
        if rest.starts_with('\t') || rest.starts_with(' ') {
            continue;
        }
        let Some((k, v)) = rest.split_once(" = ") else {
            continue;
        };
        let v = v.trim();
        match k.trim() {
            "state" if js.state.is_none() => js.state = Some(v.to_string()),
            "pid" if js.pid.is_none() => js.pid = v.parse().ok(),
            "last exit code" if js.last_exit.is_none() => js.last_exit = Some(v.to_string()),
            _ => {}
        }
    }
    js
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RestartPlan {
    /// Not loaded: just load it.
    Bootstrap,
    /// Loaded, plist unchanged: `kickstart -k`.
    Kickstart,
    /// Loaded, plist changed or unknown: bootout + bootstrap.
    Reload,
}

pub fn plan_restart(
    loaded: bool,
    on_disk: Option<&[u8]>,
    snapshot: Option<&[u8]>,
    force_full: bool,
) -> RestartPlan {
    if !loaded {
        return RestartPlan::Bootstrap;
    }
    let Some(disk) = on_disk else {
        // Loaded but the file is gone: nothing to reload from.
        return RestartPlan::Kickstart;
    };
    if force_full {
        return RestartPlan::Reload;
    }
    match snapshot {
        Some(s) if s == disk => RestartPlan::Kickstart,
        // No snapshot (hand-installed plist) or changed: we can't prove
        // launchd has what's on disk, so reload.
        _ => RestartPlan::Reload,
    }
}

// ---------------------------------------------------------------------------
// Filesystem + launchctl side
// ---------------------------------------------------------------------------

fn plist_path(label: &str) -> PathBuf {
    user_home()
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{label}.plist"))
}

fn snapshot_path(label: &str) -> PathBuf {
    openfang_home()
        .join("service")
        .join(format!("{label}.loaded.plist"))
}

/// Where install/uninstall keep copies of a replaced or removed plist.
/// Deliberately NOT ~/Library/LaunchAgents: keep that folder to live jobs only.
fn plist_backup_path(label: &str, kind: &str) -> PathBuf {
    openfang_home()
        .join("service")
        .join(format!("{label}.plist.{kind}-{}", unix_secs()))
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn require_macos() -> SvcResult<()> {
    if cfg!(target_os = "macos") {
        Ok(())
    } else {
        Err(err(
            "`openfang service` currently supports macOS (launchd) only",
            "on Linux, run `openfang start` under systemd or a terminal multiplexer",
        ))
    }
}

fn uid() -> SvcResult<String> {
    let out = Command::new("id")
        .arg("-u")
        .output()
        .map_err(|e| err(format!("couldn't run `id -u`: {e}"), "check your PATH"))?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit()) {
        return Err(err("couldn't determine your user id", "check `id -u`"));
    }
    Ok(s)
}

struct Launchd {
    domain: String,
    target: String,
}

impl Launchd {
    fn new(label: &str) -> SvcResult<Self> {
        let uid = uid()?;
        Ok(Self {
            domain: format!("gui/{uid}"),
            target: format!("gui/{uid}/{label}"),
        })
    }

    fn launchctl(args: &[&str]) -> SvcResult<std::process::Output> {
        Command::new("launchctl")
            .args(args)
            .output()
            .map_err(|e| err(format!("couldn't run launchctl: {e}"), "is this macOS?"))
    }

    fn print(&self) -> Option<String> {
        let out = Self::launchctl(&["print", &self.target]).ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn is_loaded(&self) -> bool {
        self.print().is_some()
    }

    fn check(out: std::process::Output, what: &str) -> SvcResult<()> {
        if out.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(err(
            format!(
                "launchctl {what} failed ({}): {}",
                out.status,
                stderr.trim()
            ),
            "see `openfang service status` and ~/.openfang/daemon.stderr.log",
        ))
    }

    fn bootstrap(&self, plist: &Path) -> SvcResult<()> {
        let p = plist.to_string_lossy();
        Self::check(
            Self::launchctl(&["bootstrap", &self.domain, &p])?,
            "bootstrap",
        )
    }

    /// Unload and wait until launchd has actually let go of the label;
    /// bootstrapping too soon after a bootout fails with an I/O error.
    fn bootout(&self) -> SvcResult<()> {
        Self::check(Self::launchctl(&["bootout", &self.target])?, "bootout")?;
        for _ in 0..40 {
            if !self.is_loaded() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err(err(
            "job still loaded 10s after bootout",
            "wait a moment and rerun; check `launchctl print` for the label",
        ))
    }

    fn kickstart(&self, kill: bool) -> SvcResult<()> {
        let mut args = vec!["kickstart"];
        if kill {
            args.push("-k");
        }
        args.push(&self.target);
        Self::check(Self::launchctl(&args)?, "kickstart")
    }
}

fn write_snapshot(label: &str, bytes: &[u8]) {
    let p = snapshot_path(label);
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Err(e) = std::fs::write(&p, bytes) {
        ui::warn_with_fix(
            &format!("couldn't record loaded plist at {}: {e}", p.display()),
            "next `service restart` will do a full reload to be safe",
        );
    }
}

fn load_and_snapshot(ld: &Launchd, label: &str, plist: &Path) -> SvcResult<()> {
    let bytes = std::fs::read(plist)
        .map_err(|e| err(format!("can't read {}: {e}", plist.display()), "reinstall"))?;
    ld.bootstrap(plist)?;
    write_snapshot(label, &bytes);
    Ok(())
}

fn write_plist(path: &Path, content: &str) -> SvcResult<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| {
            err(
                format!("can't create {}: {e}", d.display()),
                "check permissions",
            )
        })?;
    }
    let tmp = path.with_extension("plist.tmp");
    std::fs::write(&tmp, content).map_err(|e| {
        err(
            format!("can't write {}: {e}", tmp.display()),
            "check permissions",
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644));
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        err(
            format!("can't move plist into place: {e}"),
            "check permissions",
        )
    })
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn install(opts: &InstallOpts) -> SvcResult<()> {
    let spec = build_spec(opts)?;
    let xml = render_plist(&spec);
    let path = plist_path(&opts.label);

    let shell_path = spec.env.get("PATH").cloned().unwrap_or_default();
    let claude = find_on_path("claude", &shell_path);

    if opts.dry_run {
        println!("{xml}");
        eprintln!("(dry run) would write {}", path.display());
        if claude.is_none() {
            eprintln!("(dry run) warning: `claude` CLI not found on the daemon's PATH");
        }
        return Ok(());
    }
    require_macos()?;
    let ld = Launchd::new(&opts.label)?;

    if path.exists() {
        if !opts.force {
            return Err(err(
                format!("{} already exists", path.display()),
                "`openfang service restart` to apply changes, or `openfang service install --force` to regenerate it (the old file is backed up)",
            ));
        }
        let bak = plist_backup_path(&opts.label, "bak");
        if let Some(d) = bak.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        std::fs::copy(&path, &bak).map_err(|e| {
            err(
                format!("couldn't back up existing plist: {e}"),
                "check permissions",
            )
        })?;
        ui::success(&format!("Backed up existing plist to {}", bak.display()));
    }

    let loaded = ld.is_loaded();
    if !loaded && crate::find_daemon().is_some() {
        // Loading a second job would start a second daemon on the same home
        // and port, and KeepAlive would crash-loop it.
        return Err(err(
            "a daemon is already running (outside this launchd job)",
            "stop it first (`openfang stop`, or `openfang service stop --label <other>` if another job owns it), then rerun install",
        ));
    }

    write_plist(&path, &xml)?;
    ui::success(&format!("Wrote {}", path.display()));
    if loaded {
        ld.bootout()?;
    }
    load_and_snapshot(&ld, &opts.label, &path)?;
    ui::success(&format!("Loaded launchd job {}", opts.label));

    ui::blank();
    ui::kv("Daemon", &spec.program.to_string_lossy());
    match spec.env.get(ENV_BRIDGE_BIN) {
        Some(b) => ui::kv_ok("MCP bridge", b),
        None => ui::kv_warn("MCP bridge", "off (--no-bridge)"),
    }
    match &claude {
        Some(c) => ui::kv_ok("claude CLI", &c.to_string_lossy()),
        None => ui::kv_warn(
            "claude CLI",
            "not found on the daemon's PATH — Claude Code agents will fail to start",
        ),
    }
    ui::kv("Logs", &spec.stderr_log.to_string_lossy());
    ui::blank();
    ui::hint("Check it with `openfang service status`. Secrets go in ~/.openfang/secrets.env.");
    Ok(())
}

fn uninstall(label: &str) -> SvcResult<()> {
    require_macos()?;
    let ld = Launchd::new(label)?;
    let path = plist_path(label);
    if ld.is_loaded() {
        ld.bootout()?;
        ui::success(&format!("Unloaded {label} (daemon stopped)"));
    }
    if path.exists() {
        let keep = plist_backup_path(label, "uninstalled");
        if let Some(d) = keep.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        std::fs::copy(&path, &keep)
            .map_err(|e| err(format!("couldn't back up plist: {e}"), "check permissions"))?;
        std::fs::remove_file(&path).map_err(|e| {
            err(
                format!("couldn't remove {}: {e}", path.display()),
                "check permissions",
            )
        })?;
        ui::success(&format!(
            "Removed {} (copy kept at {})",
            path.display(),
            keep.display()
        ));
    } else {
        ui::hint(&format!("No plist at {}", path.display()));
    }
    let _ = std::fs::remove_file(snapshot_path(label));
    Ok(())
}

fn start(label: &str) -> SvcResult<()> {
    require_macos()?;
    let ld = Launchd::new(label)?;
    let path = plist_path(label);
    if ld.is_loaded() {
        ld.kickstart(false)?;
        ui::success(&format!(
            "{label} is loaded; started (no-op if already running)"
        ));
        return Ok(());
    }
    if !path.exists() {
        return Err(err(
            format!("{label} is not installed"),
            "run `openfang service install`",
        ));
    }
    load_and_snapshot(&ld, label, &path)?;
    ui::success(&format!("Loaded and started {label}"));
    Ok(())
}

fn stop(label: &str) -> SvcResult<()> {
    require_macos()?;
    let ld = Launchd::new(label)?;
    if !ld.is_loaded() {
        ui::hint(&format!("{label} is not loaded; nothing to stop"));
        return Ok(());
    }
    ld.bootout()?;
    ui::success(&format!(
        "Stopped {label} (job unloaded, so KeepAlive won't relaunch it)"
    ));
    ui::hint("It loads again at next login. `openfang service start` brings it back now.");
    Ok(())
}

fn restart(label: &str, full: bool) -> SvcResult<()> {
    require_macos()?;
    let ld = Launchd::new(label)?;
    let path = plist_path(label);
    let disk = std::fs::read(&path).ok();
    let snap = std::fs::read(snapshot_path(label)).ok();
    let plan = plan_restart(ld.is_loaded(), disk.as_deref(), snap.as_deref(), full);
    match plan {
        RestartPlan::Bootstrap => {
            if disk.is_none() {
                return Err(err(
                    format!("{label} is not installed"),
                    "run `openfang service install`",
                ));
            }
            load_and_snapshot(&ld, label, &path)?;
            ui::success(&format!("{label} wasn't loaded; loaded and started it"));
        }
        RestartPlan::Kickstart => {
            ld.kickstart(true)?;
            ui::success(&format!("Restarted {label} (plist unchanged)"));
        }
        RestartPlan::Reload => {
            ld.bootout()?;
            load_and_snapshot(&ld, label, &path)?;
            let why = if full {
                "--full"
            } else if snap.is_none() {
                "no record of the loaded plist"
            } else {
                "plist changed"
            };
            ui::success(&format!("Reloaded {label} from {} ({why})", path.display()));
        }
    }
    Ok(())
}

/// Read one EnvironmentVariables entry from a plist via `plutil`.
fn plist_env(path: &Path, key: &str) -> Option<String> {
    let out = Command::new("plutil")
        .args([
            "-extract",
            &format!("EnvironmentVariables.{key}"),
            "raw",
            "-o",
            "-",
        ])
        .arg(path)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn status(label: &str, json: bool) -> SvcResult<()> {
    require_macos()?;
    let ld = Launchd::new(label)?;
    let path = plist_path(label);
    let installed = path.exists();
    let printed = ld.print();
    let job = printed
        .as_deref()
        .map(parse_launchctl_print)
        .unwrap_or_default();
    let enabled = installed
        .then(|| plist_env(&path, ENV_BRIDGE_ENABLED))
        .flatten();
    let bin = installed
        .then(|| plist_env(&path, ENV_BRIDGE_BIN))
        .flatten();
    let bin_exists = bin.as_deref().map(|b| Path::new(b).is_file());
    let disk = std::fs::read(&path).ok();
    let snap = std::fs::read(snapshot_path(label)).ok();
    let pending_reload = printed.is_some()
        && plan_restart(true, disk.as_deref(), snap.as_deref(), false) == RestartPlan::Reload;

    if json {
        let v = serde_json::json!({
            "label": label,
            "plist": path,
            "installed": installed,
            "loaded": printed.is_some(),
            "state": job.state,
            "pid": job.pid,
            "last_exit": job.last_exit,
            "bridge_enabled": enabled,
            "bridge_bin": bin,
            "bridge_bin_exists": bin_exists,
            "plist_changed_since_load": pending_reload,
        });
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return Ok(());
    }

    ui::section(&format!("Service {label}"));
    if installed {
        ui::kv_ok("Plist", &path.to_string_lossy());
    } else {
        ui::kv_warn("Plist", "not installed (`openfang service install`)");
    }
    if printed.is_some() {
        ui::kv_ok("Loaded", "yes");
        ui::kv("State", job.state.as_deref().unwrap_or("unknown"));
        ui::kv(
            "PID",
            &job.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into()),
        );
        ui::kv("Last exit", job.last_exit.as_deref().unwrap_or("-"));
    } else {
        ui::kv_warn("Loaded", "no");
    }
    match (enabled.as_deref(), bin.as_deref(), bin_exists) {
        (Some("1"), Some(b), Some(true)) => ui::kv_ok("MCP bridge", &format!("on ({b})")),
        (Some("1"), Some(b), _) => ui::kv_warn(
            "MCP bridge",
            &format!("on, but {b} is missing — Claude Code agents will have no tools"),
        ),
        (Some("1"), None, _) => ui::kv("MCP bridge", "on (binary auto-detected next to daemon)"),
        (Some(v), _, _) => ui::kv_warn("MCP bridge", &format!("{ENV_BRIDGE_ENABLED}={v}")),
        (None, _, _) if installed => {
            ui::kv_warn("MCP bridge", &format!("{ENV_BRIDGE_ENABLED} not set"))
        }
        _ => {}
    }
    if pending_reload {
        ui::hint("The plist on disk may differ from what launchd loaded; `openfang service restart` will reload it.");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PlistSpec {
        let mut env = BTreeMap::new();
        env.insert("HOME".into(), "/Users/a".into());
        env.insert("OPENFANG_BRIDGE_ENABLED".into(), "1".into());
        PlistSpec {
            label: "ai.openfang.daemon".into(),
            program: "/Users/a/.openfang/bin/openfang".into(),
            working_dir: "/Users/a/.openfang".into(),
            stdout_log: "/Users/a/.openfang/daemon.stdout.log".into(),
            stderr_log: "/Users/a/.openfang/daemon.stderr.log".into(),
            env,
        }
    }

    #[test]
    fn plist_has_keepalive_program_and_env() {
        let x = render_plist(&spec());
        assert!(x.contains("<key>KeepAlive</key>\n    <true/>"));
        assert!(x.contains(
            "<string>/Users/a/.openfang/bin/openfang</string>\n        <string>start</string>"
        ));
        assert!(x.contains("<key>OPENFANG_BRIDGE_ENABLED</key>\n        <string>1</string>"));
        assert!(x.starts_with("<?xml"));
        assert!(x.trim_end().ends_with("</plist>"));
    }

    #[test]
    fn plist_escapes_xml() {
        let mut s = spec();
        s.env.insert("X".into(), "a<b>&\"c'".into());
        let x = render_plist(&s);
        assert!(x.contains("<string>a&lt;b&gt;&amp;&quot;c&apos;</string>"));
        assert!(!x.contains("a<b>"));
    }

    #[test]
    fn plist_output_is_deterministic() {
        // restart's change detection compares bytes; env order must be stable.
        assert_eq!(render_plist(&spec()), render_plist(&spec()));
    }

    #[test]
    fn extra_env_parses_and_rejects() {
        let ok =
            parse_extra_env(&["OPENFANG_TOOL_TIMEOUT_SECS=300".into(), "A=b=c".into()]).unwrap();
        assert_eq!(ok["OPENFANG_TOOL_TIMEOUT_SECS"], "300");
        assert_eq!(ok["A"], "b=c");
        for bad in [
            "NOEQUALS",
            "1BAD=x",
            "BAD-KEY=x",
            "OPENFANG_BRIDGE_ENABLED=0",
            "OPENFANG_BRIDGE_BIN=/x",
            "PATH=/x",
            "NETLIFY_AUTH_TOKEN=x",
            "ANTHROPIC_API_KEY=x",
            "DB_PASSWORD=x",
            "client_secret=x",
        ] {
            assert!(
                parse_extra_env(&[bad.into()]).is_err(),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn bridge_on_writes_explicit_enabled_and_bin() {
        let env = bridge_env(Path::new("/opt/of/bin/openfang"), false, |_| true).unwrap();
        assert_eq!(
            env,
            vec![
                ("OPENFANG_BRIDGE_ENABLED".to_string(), "1".to_string()),
                (
                    "OPENFANG_BRIDGE_BIN".to_string(),
                    "/opt/of/bin/openfang-mcp-bridge".to_string()
                ),
            ]
        );
    }

    #[test]
    fn bridge_missing_refuses_instead_of_writing_fail_closed_row() {
        let e = bridge_env(Path::new("/opt/of/bin/openfang"), false, |_| false).unwrap_err();
        assert!(e.msg.contains("/opt/of/bin/openfang-mcp-bridge"));
        assert!(e.fix.contains("--no-bridge"));
    }

    #[test]
    fn no_bridge_writes_explicit_off() {
        let env = bridge_env(Path::new("/x/openfang"), true, |_| false).unwrap();
        assert_eq!(
            env,
            vec![("OPENFANG_BRIDGE_ENABLED".to_string(), "0".to_string())]
        );
    }

    #[test]
    fn daemon_path_puts_bin_dir_first_and_dedups() {
        let p = daemon_path(
            Path::new("/a/bin/openfang"),
            "/opt/homebrew/bin:/a/bin:/usr/bin:",
        );
        assert_eq!(p, "/a/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin");
    }

    #[test]
    fn parses_top_level_launchctl_fields_only() {
        let out = "gui/501/ai.openfang.daemon = {\n\
            \tactive count = 1\n\
            \tpath = /Users/a/Library/LaunchAgents/ai.openfang.daemon.plist\n\
            \tstate = running\n\
            \n\
            \tprogram = /Users/a/.openfang/bin/openfang\n\
            \tenvironment = {\n\
            \t\tstate = nested-should-be-ignored\n\
            \t}\n\
            \tpid = 4242\n\
            \tlast exit code = (never exited)\n\
            }\n";
        let js = parse_launchctl_print(out);
        assert_eq!(js.state.as_deref(), Some("running"));
        assert_eq!(js.pid, Some(4242));
        assert_eq!(js.last_exit.as_deref(), Some("(never exited)"));
    }

    #[test]
    fn parses_stopped_job_without_pid() {
        let js = parse_launchctl_print("x = {\n\tstate = not running\n\tlast exit code = 1\n}\n");
        assert_eq!(js.state.as_deref(), Some("not running"));
        assert_eq!(js.pid, None);
        assert_eq!(js.last_exit.as_deref(), Some("1"));
    }

    #[test]
    fn restart_plan_matrix() {
        let a: &[u8] = b"a";
        let b: &[u8] = b"b";
        use RestartPlan::*;
        assert_eq!(plan_restart(false, Some(a), Some(a), false), Bootstrap);
        assert_eq!(plan_restart(true, Some(a), Some(a), false), Kickstart);
        assert_eq!(plan_restart(true, Some(a), Some(b), false), Reload);
        // Hand-installed plist, never loaded by us: reload to be sure.
        assert_eq!(plan_restart(true, Some(a), None, false), Reload);
        assert_eq!(plan_restart(true, Some(a), Some(a), true), Reload);
        // Loaded but file deleted: can't reload from nothing.
        assert_eq!(plan_restart(true, None, Some(a), true), Kickstart);
    }
}
