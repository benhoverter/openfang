//! Core types and traits for the OpenFang Agent Operating System.
//!
//! This crate defines all shared data structures used across the OpenFang kernel,
//! runtime, memory substrate, and wire protocol. It contains no business logic.

pub mod agent;
pub mod agent_wake;
pub mod approval;
pub mod async_reply;
pub mod bridge_auth;
pub mod capability;
pub mod cmd_norm;
pub mod commands;
pub mod comms;
pub mod config;
pub mod error;
pub mod event;
pub mod gatekeeper;
pub mod manifest_signing;
pub mod media;
pub mod memory;
pub mod message;
pub mod model_catalog;
pub mod path_facts;
pub mod paths;
pub mod scheduler;
pub mod security_flags;
pub mod serde_compat;
pub mod taint;
pub mod tool;
pub mod tool_compat;
pub mod turn;
pub mod turn_context;
pub mod wake;
pub mod watchdog;
pub mod webhook;

/// Safely truncate a string to at most `max_bytes`, never splitting a UTF-8 char.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Shorten `s` to roughly `max_chars` characters by eliding the *middle*,
/// keeping both the head and the tail.
///
/// Tail-first truncation ([`truncate_str`]) is actively dangerous on an
/// operator decision surface: for a shell command the risky part is usually the
/// argument list, i.e. exactly the part a tail cut discards (ANAI-151). This
/// keeps ~60% head / ~40% tail and states how much was dropped, so a reader can
/// never mistake an elided command for a complete one.
///
/// Character-based (not byte-based) so multi-byte input can never be split.
pub fn elide_middle(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    let head_len = max_chars * 3 / 5;
    let tail_len = max_chars - head_len;
    let head: String = s.chars().take(head_len).collect();
    let tail: String = s.chars().skip(total - tail_len).collect();
    let elided = total - head_len - tail_len;
    format!("{head}\n… [{elided} chars elided] …\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_str_ascii() {
        assert_eq!(truncate_str("hello world", 5), "hello");
    }

    #[test]
    fn truncate_str_chinese() {
        // Each Chinese character is 3 bytes
        let s = "\u{4F60}\u{597D}\u{4E16}\u{754C}"; // 你好世界
        assert_eq!(truncate_str(s, 6), "\u{4F60}\u{597D}"); // 你好
        assert_eq!(truncate_str(s, 7), "\u{4F60}\u{597D}"); // still 你好 (7 is mid-char)
        assert_eq!(truncate_str(s, 9), "\u{4F60}\u{597D}\u{4E16}"); // 你好世
    }

    #[test]
    fn truncate_str_emoji() {
        let s = "hi\u{1F600}there"; // hi😀there — emoji is 4 bytes
        assert_eq!(truncate_str(s, 3), "hi"); // 3 is mid-emoji
        assert_eq!(truncate_str(s, 6), "hi\u{1F600}"); // after emoji
    }

    #[test]
    fn truncate_str_em_dash() {
        // Em dash (—) is 3 bytes (0xE2 0x80 0x94) — the exact char that caused
        // production panics in kernel.rs and session.rs (issue #104)
        let s = "Here is a summary — with details";
        assert_eq!(truncate_str(s, 19), "Here is a summary ");
        assert_eq!(truncate_str(s, 20), "Here is a summary ");
        assert_eq!(truncate_str(s, 21), "Here is a summary \u{2014}");
    }

    #[test]
    fn truncate_str_no_truncation() {
        assert_eq!(truncate_str("short", 100), "short");
    }

    #[test]
    fn truncate_str_empty() {
        assert_eq!(truncate_str("", 10), "");
    }

    #[test]
    fn elide_middle_short_input_is_unchanged() {
        assert_eq!(elide_middle("rm -rf /tmp/x", 100), "rm -rf /tmp/x");
    }

    /// The point of the function: a tail cut drops the arguments, which is
    /// where a shell command's risk lives. Middle elision keeps both ends.
    #[test]
    fn elide_middle_keeps_the_dangerous_tail() {
        let cmd = format!("bash -c \"{} rm -rf /Users/ben/GitHub\"", "a".repeat(500));
        let out = elide_middle(&cmd, 100);
        assert!(out.starts_with("bash -c"), "{out}");
        assert!(
            out.ends_with("rm -rf /Users/ben/GitHub\""),
            "tail must survive: {out}"
        );
        assert!(
            out.contains("chars elided"),
            "elision must be stated: {out}"
        );
    }

    #[test]
    fn elide_middle_never_splits_a_char() {
        let s = "あ".repeat(100);
        let out = elide_middle(&s, 10);
        // Round-trips as valid UTF-8 with only whole chars retained.
        assert!(out.starts_with("あ"));
        assert!(out.ends_with("あ"));
    }
}
