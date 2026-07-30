//! Per-turn context envelope renderer (ANAI-128).
//!
//! Builds the ambient `<turn_context>` block that `agent_loop` injects as a
//! user-role message immediately ahead of each turn's real inbound. It is a
//! user-role message (not system) on purpose: an in-array system message is
//! dropped by every driver when `request.system` is already set — which it
//! always is in the agent loop — so a system-role envelope would never reach
//! the model. The `<turn_context>` wrapper recovers the "this is ambient
//! metadata, not the human speaking" framing. Kept out of the `system` field so
//! the cached system prefix stays stable across turns (prompt-cache friendly).
//!
//! Config gating lives in `openfang_types::turn_context`; the two injection
//! sites live in `agent_loop`. Timestamps in/out are RFC3339 (UTC) as stored by
//! the session/participants tables.

use chrono::{DateTime, Utc};
use openfang_memory::session::Participant;

/// Inputs for one envelope render. `now` is the turn-build instant; the rest are
/// RFC3339 (UTC) stamps sourced from the session and participants tables.
pub struct TurnContextInput<'a> {
    /// Turn-build instant (UTC).
    pub now: DateTime<Utc>,
    /// Durable speaker key (snowflake) for this turn, if known.
    pub sender_id: Option<&'a str>,
    /// Display name for this turn's speaker, if known.
    pub sender_name: Option<&'a str>,
    /// This speaker's PRIOR last-seen stamp (RFC3339), for `since_this_speaker`.
    pub prior_seen: Option<&'a str>,
    /// Session `updated_at` (RFC3339) = last agent activity, for `since_agent_msg`.
    pub updated_at: Option<&'a str>,
    /// Presence roster (most-recent-first). Empty when the roster is disabled.
    pub roster: &'a [Participant],
}

/// Humanize a non-negative duration (seconds) as a compact `"2h 04m"` /
/// `"3d 4h"` / `"5s"` label.
fn humanize(secs: i64) -> String {
    let s = secs.max(0);
    if s < 60 {
        return format!("{s}s");
    }
    let mins = s / 60;
    if mins < 60 {
        return format!("{mins}m");
    }
    let hours = mins / 60;
    if hours < 24 {
        let rem_m = mins % 60;
        return format!("{hours}h {rem_m:02}m");
    }
    let days = hours / 24;
    let rem_h = hours % 24;
    if rem_h == 0 {
        format!("{days}d")
    } else {
        format!("{days}d {rem_h}h")
    }
}

/// Seconds between `now` and an RFC3339 stamp, clamped to `>= 0`. `None` if the
/// stamp is unparseable (defensive — the tables always write RFC3339).
fn delta_secs(now: DateTime<Utc>, stamp: &str) -> Option<i64> {
    let then = DateTime::parse_from_rfc3339(stamp)
        .ok()?
        .with_timezone(&Utc);
    Some((now - then).num_seconds().max(0))
}

/// Render the `<turn_context>` block, or `None` when there is nothing to say.
pub fn render(input: &TurnContextInput) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();

    // now — always present, rendered in the operator's local zone for legibility.
    let now_local = input.now.with_timezone(&chrono::Local);
    lines.push(format!(
        "now:                {}",
        now_local.format("%Y-%m-%d %H:%M %Z")
    ));

    // speaker + per-actor gap — only when we know who spoke.
    if let Some(id) = input.sender_id {
        let name = input.sender_name.unwrap_or(id);
        lines.push(format!("speaker:            {name} (id:{id})"));
        if let Some(prior) = input.prior_seen {
            if let Some(d) = delta_secs(input.now, prior) {
                lines.push(format!("since_this_speaker: {}", humanize(d)));
            }
        }
    }

    // since_agent_msg — since any agent activity (now - updated_at).
    if let Some(ua) = input.updated_at {
        if let Some(d) = delta_secs(input.now, ua) {
            lines.push(format!("since_agent_msg:    {}", humanize(d)));
        }
    }

    // recently_present roster — excludes the current speaker (redundant with the
    // speaker line) and any actor whose stamp won't parse.
    if !input.roster.is_empty() {
        let parts: Vec<String> = input
            .roster
            .iter()
            .filter(|p| Some(p.speaker_id.as_str()) != input.sender_id)
            .filter_map(|p| {
                delta_secs(input.now, &p.last_msg_at)
                    .map(|d| format!("{} {} ago", p.display_name, humanize(d)))
            })
            .collect();
        if !parts.is_empty() {
            lines.push(format!("recently_present:   {}", parts.join(" · ")));
        }
    }

    // `now` alone is still a real ambient signal (the clock), so we keep a
    // single-line envelope. Only bail on the impossible empty case.
    if lines.is_empty() {
        return None;
    }

    Some(format!(
        "<turn_context>\n{}\n</turn_context>",
        lines.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(id: &str, name: &str, last: &str) -> Participant {
        Participant {
            speaker_id: id.to_string(),
            display_name: name.to_string(),
            last_msg_at: last.to_string(),
            first_seen_at: last.to_string(),
            message_count: 1,
        }
    }

    #[test]
    fn test_humanize_boundaries() {
        assert_eq!(humanize(-5), "0s");
        assert_eq!(humanize(0), "0s");
        assert_eq!(humanize(59), "59s");
        assert_eq!(humanize(60), "1m");
        assert_eq!(humanize(59 * 60), "59m");
        assert_eq!(humanize(2 * 3600 + 4 * 60), "2h 04m");
        assert_eq!(humanize(24 * 3600), "1d");
        assert_eq!(humanize(27 * 3600), "1d 3h");
    }

    #[test]
    fn test_render_full_envelope() {
        let now = DateTime::parse_from_rfc3339("2026-07-20T12:01:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let roster = vec![
            part("snow_bob", "Bob", "2026-07-19T12:00:00Z"),
            part("snow_ben", "Ben", "2026-07-20T12:01:00Z"),
        ];
        let input = TurnContextInput {
            now,
            sender_id: Some("snow_ben"),
            sender_name: Some("Ben Hoverter"),
            prior_seen: Some("2026-07-20T10:00:00Z"),
            updated_at: Some("2026-07-20T11:59:00Z"),
            roster: &roster,
        };
        let out = render(&input).unwrap();
        assert!(out.starts_with("<turn_context>\n"));
        assert!(out.ends_with("\n</turn_context>"));
        assert!(out.contains("speaker:            Ben Hoverter (id:snow_ben)"));
        assert!(out.contains("since_this_speaker: 2h 01m"));
        assert!(out.contains("since_agent_msg:    2m"));
        // Roster excludes the current speaker (Ben), includes Bob's ~1d gap.
        assert!(out.contains("recently_present:   Bob 1d ago"));
        assert!(!out.contains("Ben 0s ago"));
    }

    #[test]
    fn test_render_autonomous_no_speaker() {
        // Cron/autonomous turn: no sender, no prior. now + since_agent_msg only.
        let now = DateTime::parse_from_rfc3339("2026-07-20T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let input = TurnContextInput {
            now,
            sender_id: None,
            sender_name: None,
            prior_seen: None,
            updated_at: Some("2026-07-20T06:00:00Z"),
            roster: &[],
        };
        let out = render(&input).unwrap();
        assert!(!out.contains("speaker:"));
        assert!(!out.contains("since_this_speaker:"));
        assert!(out.contains("since_agent_msg:    6h 00m"));
    }
}
