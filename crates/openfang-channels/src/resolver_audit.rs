//! Recipient-resolution audit log (ANAI-55, security follow-up).
//!
//! Writes one line per successful recipient resolution to a dedicated file
//! (`$OPENFANG_HOME/daemon/resolver_audit.log`, mode `0o600`), bypassing the
//! `tracing` subscriber stack entirely.
//!
//! ## Why a dedicated sink, not a `tracing` target?
//!
//! Security review (ANAI-55 §6.1, F1 disposition) required that the audit
//! stream be **structurally** isolated from payload logs so that a log scrape
//! cannot correlate "agent X resolved recipient Y" with adjacent message
//! bodies by timestamp.
//!
//! Routing an audit `tracing` target to a different file requires composing a
//! split `tracing-subscriber` layer graph at every binary entry point
//! (`openfang-cli`, `openfang-desktop`, ...). That correctness is fragile —
//! any future EnvFilter change or wildcard layer can silently re-merge the
//! streams. Writing directly to a dedicated file makes isolation a property
//! of the sink, not a consequence of the subscriber graph.
//!
//! ## Concurrency
//!
//! Resolutions are rare (cache-miss path), so `std::sync::Mutex<File>` is
//! sufficient and avoids any drop-on-shutdown hazard an mpsc writer task
//! would introduce. The lock window is one `writeln!`. **Never** hold this
//! lock across an `.await` — the file write is synchronous by design.
//!
//! ## File mode
//!
//! Created with `0o600` on Unix (audit log MUST NOT be world-readable). On
//! non-Unix platforms the mode hint is dropped silently — those targets are
//! not supported for production daemon use today.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

/// Process-wide singleton. `None` if the audit log could not be opened on
/// first use (e.g., permission denied); resolution proceeds without auditing
/// rather than failing closed on a logging error. The failure is reported
/// via `tracing::error!` exactly once.
static AUDIT_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

fn openfang_home() -> PathBuf {
    if let Ok(h) = std::env::var("OPENFANG_HOME") {
        return PathBuf::from(h);
    }
    if let Ok(h) = std::env::var("HOME") {
        return PathBuf::from(h).join(".openfang");
    }
    PathBuf::from(".openfang")
}

fn audit_log_path() -> PathBuf {
    openfang_home().join("daemon").join("resolver_audit.log")
}

fn open_audit_file() -> Option<Mutex<File>> {
    let path = audit_log_path();
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            tracing::error!(
                "resolver_audit: failed to create log directory {}: {}",
                parent.display(),
                e
            );
            return None;
        }
    }

    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        opts.mode(0o600);
    }
    match opts.open(&path) {
        Ok(f) => Some(Mutex::new(f)),
        Err(e) => {
            tracing::error!(
                "resolver_audit: failed to open log {}: {}",
                path.display(),
                e
            );
            None
        }
    }
}

/// Record a successful recipient resolution. Best-effort: a failure to write
/// the audit line is logged at error level but never propagated to the
/// caller.
///
/// `input` is the raw recipient string as supplied by the agent.
/// `platform_id` is the resolved platform-native ID.
/// `via` is a short tag for the resolution path (e.g. `"snowflake"`,
/// `"channel_mention"`, `"username_cache"`).
pub fn record_resolution(adapter: &str, input: &str, platform_id: &str, via: &str) {
    let slot = AUDIT_FILE.get_or_init(open_audit_file);
    let Some(mu) = slot else {
        return;
    };
    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let line = format!(
        "{ts} adapter={adapter} input={input:?} resolved_platform_id={platform_id} via={via}\n"
    );
    if let Ok(mut f) = mu.lock() {
        if let Err(e) = f.write_all(line.as_bytes()) {
            tracing::error!("resolver_audit: write failed: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `open_audit_file` creates the file with mode `0o600` on
    /// Unix and that the parent directory is created lazily.
    ///
    /// NOTE: `AUDIT_FILE` is a process-wide `OnceLock`, so the public
    /// `record_resolution` path can only be exercised hermetically once per
    /// test binary. We test the open path directly to keep this test
    /// independent of other tests in the crate.
    #[test]
    fn open_audit_file_creates_0600_file() {
        let tmp = tempfile::tempdir().unwrap();
        let prev = std::env::var("OPENFANG_HOME").ok();
        std::env::set_var("OPENFANG_HOME", tmp.path());

        let path = audit_log_path();
        assert!(open_audit_file().is_some());
        assert!(
            path.exists(),
            "audit file not created at {}",
            path.display()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let md = std::fs::metadata(&path).unwrap();
            let mode = md.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "audit file mode is {:o}, expected 0600", mode);
        }

        match prev {
            Some(v) => std::env::set_var("OPENFANG_HOME", v),
            None => std::env::remove_var("OPENFANG_HOME"),
        }
    }
}
