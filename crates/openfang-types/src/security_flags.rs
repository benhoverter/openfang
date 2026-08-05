//! Startup-frozen security flags (ANAI-150).
//!
//! # The invariant
//!
//! **An agent that can set an environment variable must not be able to use it
//! to relax security after the process has started.**
//!
//! Every security-relevant environment variable is read exactly once, into an
//! immutable process-wide snapshot, and every later consumer reads the
//! snapshot rather than the environment. `std::env::set_var` after that point
//! is inert: the value is already frozen.
//!
//! This mirrors Hermes v0.19, which snapshots `HERMES_YOLO_MODE` into
//! `_YOLO_MODE_FROZEN` at import time and does the same for its redaction
//! toggle.
//!
//! # Why a snapshot rather than lazy caching
//!
//! A `OnceLock` that initializes on *first read* is not the same guarantee. If
//! the first read happens late — after a turn has run, after a tool has
//! executed — then whatever the environment says at that moment is what gets
//! frozen, and the poisoned value becomes permanent instead of transient. That
//! is strictly worse than reading live.
//!
//! So the snapshot must be taken during daemon startup, before any agent runs.
//! [`init`] is that call. [`initialized`] reports whether it happened, and
//! [`init`] reports whether something had already forced the snapshot before
//! it ran — which is a bug in startup ordering, not a benign race.
//!
//! # What belongs here
//!
//! Flags whose failure mode is *relaxation*: turning an authentication check,
//! an audit record, or a scanner off. Path roots (`OPENFANG_HOME`), credential
//! material (`OPENFANG_API_KEY`, `OPENFANG_VAULT_KEY`) and tuning knobs
//! (timeouts) are deliberately **not** here — they are inputs, not policy
//! toggles, and freezing them would break the test harnesses that legitimately
//! repoint `OPENFANG_HOME` per test.
//!
//! # Adding a flag
//!
//! 1. Add the field and its env-var const.
//! 2. Give it a *secure default* in [`SecurityFlags::secure_defaults`] — the
//!    value that must hold when the variable is absent or unparseable.
//! 3. Report it in [`SecurityFlags::relaxations`] when it deviates, so the
//!    daemon logs it at boot.
//! 4. Replace every live `env::var` read with the accessor.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Opt *in* to running the HTTP/WS API without authentication.
pub const ALLOW_NO_AUTH_ENV: &str = "OPENFANG_ALLOW_NO_AUTH";

/// Opt *out* of read-side context-file injection scanning (ANAI-149 D1).
pub const CONTEXT_SCAN_ENV: &str = "OPENFANG_CONTEXT_SCAN";

/// Opt *out* of write-side context-file audit records (ANAI-149 D2).
pub const CONTEXT_AUDIT_ENV: &str = "OPENFANG_CONTEXT_AUDIT";

/// The frozen snapshot of every security-relevant environment flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityFlags {
    /// `true` iff `OPENFANG_ALLOW_NO_AUTH` opted in. Secure default: `false`.
    pub allow_no_auth: bool,
    /// `false` iff `OPENFANG_CONTEXT_SCAN` opted out. Secure default: `true`.
    pub context_scan: bool,
    /// `false` iff `OPENFANG_CONTEXT_AUDIT` opted out. Secure default: `true`.
    pub context_audit: bool,
}

impl SecurityFlags {
    /// The values that hold when no environment variable is set. Every field
    /// here is the *safe* side of its toggle.
    pub const fn secure_defaults() -> Self {
        Self {
            allow_no_auth: false,
            context_scan: true,
            context_audit: true,
        }
    }

    /// Read the current environment. Pure: no globals touched, so tests can
    /// exercise parsing without disturbing the process snapshot.
    pub fn from_env() -> Self {
        Self {
            allow_no_auth: read_opt_in(ALLOW_NO_AUTH_ENV),
            context_scan: read_opt_out(CONTEXT_SCAN_ENV),
            context_audit: read_opt_out(CONTEXT_AUDIT_ENV),
        }
    }

    /// Every flag that deviates from its secure default, as
    /// `(env_var, human description)`. Empty when the process is running fully
    /// locked down. The daemon logs this at boot so a relaxed flag is never
    /// silent.
    pub fn relaxations(&self) -> Vec<(&'static str, &'static str)> {
        let mut out = Vec::new();
        if self.allow_no_auth {
            out.push((
                ALLOW_NO_AUTH_ENV,
                "API authentication is DISABLED for non-loopback requests",
            ));
        }
        if !self.context_scan {
            out.push((
                CONTEXT_SCAN_ENV,
                "context-file injection scanning is disabled",
            ));
        }
        if !self.context_audit {
            out.push((CONTEXT_AUDIT_ENV, "context-file write auditing is disabled"));
        }
        out
    }
}

impl Default for SecurityFlags {
    fn default() -> Self {
        Self::secure_defaults()
    }
}

/// An opt-in toggle: absent, empty, or unrecognized means `false`.
fn read_opt_in(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "enabled"
        ),
        Err(_) => false,
    }
}

/// An opt-out toggle: absent, empty, or unrecognized means `true`. Only an
/// explicit off-word disables it — a typo must not silently switch a control
/// off.
fn read_opt_out(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no" | "disabled"
        ),
        Err(_) => true,
    }
}

static FLAGS: OnceLock<SecurityFlags> = OnceLock::new();
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Outcome of [`init`], so the caller can log it (this crate has no `tracing`
/// dependency by design).
#[derive(Debug, Clone, Copy)]
pub struct InitOutcome {
    /// The frozen snapshot.
    pub flags: SecurityFlags,
    /// `true` if the snapshot had *already* been taken by an earlier read.
    ///
    /// A startup-ordering bug: some consumer sampled the environment before
    /// the daemon froze it. The frozen value is still whatever that first read
    /// saw, so this deserves a warning, not silence.
    pub already_frozen: bool,
}

/// Freeze the security flags. Call once, as early in daemon startup as
/// possible — before the kernel boots, before any agent turn, before the API
/// server accepts a connection.
///
/// Idempotent: later calls return the original snapshot with
/// `already_frozen = true`.
pub fn init() -> InitOutcome {
    let already_frozen = FLAGS.get().is_some();
    let flags = *FLAGS.get_or_init(SecurityFlags::from_env);
    INITIALIZED.store(true, Ordering::SeqCst);
    InitOutcome {
        flags,
        already_frozen,
    }
}

/// The frozen snapshot.
///
/// If [`init`] has not run, this freezes on first call — the lazy fallback
/// exists so unit tests and one-shot CLI paths behave sanely. Production
/// startup should always have called [`init`] first; [`initialized`] lets a
/// caller assert that.
pub fn get() -> SecurityFlags {
    *FLAGS.get_or_init(SecurityFlags::from_env)
}

/// Whether [`init`] has been called. `false` means any frozen value came from
/// a lazy first read, not from startup.
pub fn initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

/// Whether the API may serve unauthenticated non-loopback requests.
pub fn allow_no_auth() -> bool {
    get().allow_no_auth
}

/// Whether read-side context-file injection scanning is on.
pub fn context_scan_enabled() -> bool {
    get().context_scan
}

/// Whether write-side context-file auditing is on.
pub fn context_audit_enabled() -> bool {
    get().context_audit
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Save/restore so a test cannot leak state into its neighbours.
    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn capture(vars: &[&'static str]) -> Self {
            Self {
                saved: vars.iter().map(|v| (*v, std::env::var(v).ok())).collect(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    const ALL: &[&str] = &[ALLOW_NO_AUTH_ENV, CONTEXT_SCAN_ENV, CONTEXT_AUDIT_ENV];

    /// Every parsing case in one test: this crate's test binary is
    /// multi-threaded and the environment is process-global, so splitting
    /// these into separate `#[test]` fns would race.
    #[test]
    fn env_parsing() {
        let _g = EnvGuard::capture(ALL);

        // Absent → secure defaults on every field.
        for v in ALL {
            std::env::remove_var(v);
        }
        assert_eq!(SecurityFlags::from_env(), SecurityFlags::secure_defaults());
        assert!(SecurityFlags::from_env().relaxations().is_empty());

        // Opt-in: only the affirmative words flip it.
        for word in ["1", "true", "TRUE", " yes ", "on", "enabled"] {
            std::env::set_var(ALLOW_NO_AUTH_ENV, word);
            assert!(
                SecurityFlags::from_env().allow_no_auth,
                "expected {word:?} to enable"
            );
        }
        // Anything else — including empty, junk, and near-misses — stays off.
        for word in ["", "0", "false", "no", "off", "yolo", "maybe", "  "] {
            std::env::set_var(ALLOW_NO_AUTH_ENV, word);
            assert!(
                !SecurityFlags::from_env().allow_no_auth,
                "expected {word:?} to stay disabled"
            );
        }
        std::env::remove_var(ALLOW_NO_AUTH_ENV);

        // Opt-out: only an explicit off-word disables it.
        for word in ["off", "0", "false", "no", "disabled", " OFF "] {
            std::env::set_var(CONTEXT_SCAN_ENV, word);
            assert!(
                !SecurityFlags::from_env().context_scan,
                "expected {word:?} to disable"
            );
            std::env::set_var(CONTEXT_AUDIT_ENV, word);
            assert!(
                !SecurityFlags::from_env().context_audit,
                "expected {word:?} to disable"
            );
        }
        // A typo must NOT silently switch a control off.
        for word in ["", "1", "true", "on", "offf", "disable", "junk"] {
            std::env::set_var(CONTEXT_SCAN_ENV, word);
            assert!(
                SecurityFlags::from_env().context_scan,
                "expected {word:?} to leave scanning ON"
            );
            std::env::set_var(CONTEXT_AUDIT_ENV, word);
            assert!(
                SecurityFlags::from_env().context_audit,
                "expected {word:?} to leave auditing ON"
            );
        }
    }

    #[test]
    fn relaxations_reports_each_deviation() {
        let flags = SecurityFlags {
            allow_no_auth: true,
            context_scan: false,
            context_audit: false,
        };
        let r = flags.relaxations();
        assert_eq!(r.len(), 3);
        let vars: Vec<&str> = r.iter().map(|(v, _)| *v).collect();
        assert!(vars.contains(&ALLOW_NO_AUTH_ENV));
        assert!(vars.contains(&CONTEXT_SCAN_ENV));
        assert!(vars.contains(&CONTEXT_AUDIT_ENV));
    }

    /// **The load-bearing test for ANAI-150.** Mutating the environment after
    /// the snapshot is taken must not move any gate.
    ///
    /// Written to be order-independent: it samples whatever the snapshot
    /// already is (another test in this binary may have forced it), flips
    /// every variable to the opposite of that, and asserts nothing moved. So
    /// it holds no matter which test runs first.
    #[test]
    fn env_mutation_after_freeze_has_no_effect() {
        let _g = EnvGuard::capture(ALL);

        let frozen = get();

        // Flip every flag to its most-relaxed setting, the way a compromised
        // agent would.
        std::env::set_var(ALLOW_NO_AUTH_ENV, "1");
        std::env::set_var(CONTEXT_SCAN_ENV, "off");
        std::env::set_var(CONTEXT_AUDIT_ENV, "off");

        assert_eq!(get(), frozen, "snapshot moved after env mutation");
        assert_eq!(allow_no_auth(), frozen.allow_no_auth);
        assert_eq!(context_scan_enabled(), frozen.context_scan);
        assert_eq!(context_audit_enabled(), frozen.context_audit);

        // And a re-init cannot re-read either.
        let outcome = init();
        assert!(outcome.already_frozen);
        assert_eq!(outcome.flags, frozen);
        assert!(initialized());

        // Sanity: `from_env` *does* see the mutation — proving the assertions
        // above passed because the value is frozen, not because the writes
        // silently failed.
        let live = SecurityFlags::from_env();
        assert!(live.allow_no_auth);
        assert!(!live.context_scan);
        assert!(!live.context_audit);
    }
}
