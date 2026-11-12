//! Pacing for recipient-less `channel_send` progress posts.
//!
//! The kernel installs the `[progress_updates]` value once at boot; the
//! runtime reads it through [`min_interval_secs`]. Same shape as
//! [`crate::async_reply`].

/// Default minimum gap between two progress posts by one agent to one channel.
pub const DEFAULT_MIN_INTERVAL_SECS: u64 = 60;

static INSTALLED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Install the configured interval. First writer wins.
pub fn install_min_interval_secs(secs: u64) {
    let _ = INSTALLED.set(secs);
}

/// Resolved interval in seconds. `0` means no limit.
pub fn min_interval_secs() -> u64 {
    INSTALLED
        .get()
        .copied()
        .unwrap_or(DEFAULT_MIN_INTERVAL_SECS)
}
