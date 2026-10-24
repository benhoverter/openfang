//! Operator alerts: problems only the human running the host can fix.
//!
//! A driver deep in the runtime (e.g. the Claude Code subprocess driver) has
//! no channel adapter to talk to. It calls [`raise`], which pushes onto a
//! process-global queue. The API server installs the receiving end at boot
//! ([`install_sink`]) and delivers to the channel named in
//! `[operator_alerts]` in `config.toml`, throttled per kind by
//! [`AlertThrottle`].
//!
//! When no sink is installed (alerts not configured, tests, the CLI), [`raise`]
//! only logs. Raising never blocks and never fails the caller.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokio::sync::mpsc::UnboundedSender;
use tracing::error;

/// What went wrong. One throttle slot per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorAlertKind {
    /// The Claude Code CLI's login expired or is missing.
    ClaudeCodeAuthExpired,
}

#[derive(Debug, Clone)]
pub struct OperatorAlert {
    pub kind: OperatorAlertKind,
    /// Short machine detail for the log (never shown in the channel).
    pub detail: String,
}

impl OperatorAlert {
    /// The text posted to the operator channel.
    pub fn channel_text(&self, window: Duration) -> String {
        let mins = window.as_secs() / 60;
        match self.kind {
            OperatorAlertKind::ClaudeCodeAuthExpired => format!(
                "⚠️ **Claude Code login expired on the OpenFang host.** Agents running on \
                 Claude Code will fail until you run `claude auth login` in a terminal on that \
                 machine. (I'll repeat this at most every {mins} min while it persists.)"
            ),
        }
    }
}

static SINK: OnceLock<UnboundedSender<OperatorAlert>> = OnceLock::new();

/// Install the receiving queue. Returns false if one was already installed.
pub fn install_sink(tx: UnboundedSender<OperatorAlert>) -> bool {
    SINK.set(tx).is_ok()
}

/// Report an operator-fixable problem. Always logs at error level; also queues
/// it for channel delivery when a sink is installed.
pub fn raise(kind: OperatorAlertKind, detail: impl Into<String>) {
    let alert = OperatorAlert {
        kind,
        detail: detail.into(),
    };
    error!(event = %"operator_alert", kind = ?alert.kind, detail = %alert.detail, "Operator action required");
    if let Some(tx) = SINK.get() {
        // Receiver gone means the delivery task died; the error log above stands.
        let _ = tx.send(alert);
    }
}

/// Per-kind rate limit: at most one delivery per `window`.
pub struct AlertThrottle {
    window: Duration,
    last_sent: HashMap<OperatorAlertKind, Instant>,
}

impl AlertThrottle {
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            last_sent: HashMap::new(),
        }
    }

    pub fn window(&self) -> Duration {
        self.window
    }

    /// True if this kind may be delivered at `now`; records the send if so.
    pub fn should_send(&mut self, kind: OperatorAlertKind, now: Instant) -> bool {
        match self.last_sent.get(&kind) {
            Some(prev) if now.saturating_duration_since(*prev) < self.window => false,
            _ => {
                self.last_sent.insert(kind, now);
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: OperatorAlertKind = OperatorAlertKind::ClaudeCodeAuthExpired;

    #[test]
    fn first_alert_sends() {
        let mut t = AlertThrottle::new(Duration::from_secs(1800));
        assert!(t.should_send(K, Instant::now()));
    }

    #[test]
    fn repeat_inside_window_is_suppressed() {
        let mut t = AlertThrottle::new(Duration::from_secs(1800));
        let t0 = Instant::now();
        assert!(t.should_send(K, t0));
        assert!(!t.should_send(K, t0 + Duration::from_secs(60)));
        assert!(!t.should_send(K, t0 + Duration::from_secs(1799)));
    }

    #[test]
    fn repeat_after_window_sends_again() {
        let mut t = AlertThrottle::new(Duration::from_secs(1800));
        let t0 = Instant::now();
        assert!(t.should_send(K, t0));
        assert!(t.should_send(K, t0 + Duration::from_secs(1800)));
        // and the window restarts from the second send
        assert!(!t.should_send(K, t0 + Duration::from_secs(1900)));
    }

    #[test]
    fn channel_text_names_the_command() {
        let a = OperatorAlert {
            kind: K,
            detail: String::new(),
        };
        let s = a.channel_text(Duration::from_secs(1800));
        assert!(s.contains("claude auth login"));
        assert!(s.contains("30 min"));
    }
}
