//! ANAI-247: the rehydration pack — what an agent opens its next episode with.
//!
//! An episode close with `reset_context` leaves a deliberately thin session:
//! fresh message vector, canonical re-anchored, only the compacted summary
//! kept (ANAI-246). That is the right thing to *drop*, but on its own it
//! makes the next turn poorer than the one before it, and in practice the
//! human's next message has always been "please prime for X" typed by hand.
//!
//! The pack is that message, assembled by the system instead. It is
//! **declared, not swept**: the agent names a project slug at close time and
//! the pack is built from the durable tiers already addressed by that slug.
//! No general repo sweep, no "recent everything" — an undifferentiated dump
//! costs tokens on every primed turn and answers no particular question.
//!
//! Note what is deliberately absent: a per-project *recipe file*. The ticket
//! proposed one and it turned out to be unnecessary — tier-3 facts are
//! already addressed by `(scope, scope_ref)`, so the slug alone selects the
//! project's live claims. A recipe only earns its keep once the pack wants
//! something not already scoped that way (repo state for a named worktree is
//! the obvious candidate, and is deferred for exactly that reason).

use crate::episode::Episode;
use crate::fact::{Fact, FactStatus};

/// Closed episodes to recall. Three is a session's worth of recent history
/// without turning the pack into a changelog.
pub const MAX_EPISODES: usize = 3;

/// Live claims to surface about the primed project.
pub const MAX_FACTS: usize = 12;

/// How many verbatim canonical messages the new episode may accumulate before
/// the pack stops being emitted.
///
/// The pack is a boot briefing, not a permanent header. Once the primed
/// episode has this much of its own conversation, that conversation is the
/// better context and the pack is just rent on the protected index-0 slot.
/// Roughly six exchanges.
pub const PACK_TTL_MESSAGES: usize = 12;

/// Per-episode summary budget, characters.
const EPISODE_SUMMARY_BUDGET: usize = 400;

/// Per-claim budget, characters.
const CLAIM_BUDGET: usize = 240;

/// Whole-pack ceiling, characters.
///
/// The pack rides in the `canonical_context_msg` slot, which ANAI-242/244
/// protect at index 0 — it is the one message the trim ladder will not drop,
/// so an unbounded pack would be an unbounded permanent tax on every turn of
/// the primed episode. Roughly 750 tokens at the `chars/4` estimate the
/// ladder itself uses.
pub const PACK_BUDGET: usize = 3_000;

/// Assemble the pack text, or `None` when there is nothing worth saying.
///
/// Pure: it renders what it is handed and reads no database. The fetching
/// lives in [`crate::substrate::MemorySubstrate::rehydration_pack`], so the
/// wording — the part that actually reaches a model — is testable without
/// standing up a store.
///
/// `episodes` are the agent's own most recently closed episodes, newest
/// first, and are **not** filtered by project: episodes carry no project
/// column, and pretending otherwise would silently drop the agent's real
/// recent history. `facts` are the live claims about the primed project,
/// author-blind, open loops first.
pub fn render_pack(prime_for: &str, episodes: &[Episode], facts: &[Fact]) -> Option<String> {
    let episodes: Vec<&Episode> = episodes
        .iter()
        .filter(|e| !e.is_open())
        .filter(|e| e.title.is_some() || e.summary.is_some())
        .take(MAX_EPISODES)
        .collect();

    // A pack with a heading and nothing under it is worse than no pack: it
    // spends the protected slot to tell the agent that its memory is empty.
    if episodes.is_empty() && facts.is_empty() {
        return None;
    }

    let mut out = String::new();
    out.push_str(&format!("[Rehydration pack — primed for {prime_for}]\n"));
    out.push_str(
        "A new episode starts here. The conversation window was cleared on purpose; what \
         follows is drawn from durable memory, not from the turns that were dropped.\n",
    );

    if !episodes.is_empty() {
        out.push_str("\nRecently closed episodes, newest first:\n");
        for ep in episodes {
            let title = ep.title.as_deref().unwrap_or("(untitled)");
            match ep.summary.as_deref().filter(|s| !s.trim().is_empty()) {
                Some(summary) => out.push_str(&format!(
                    "- {title} — {}\n",
                    cap(summary, EPISODE_SUMMARY_BUDGET)
                )),
                // No summary yet is the normal state for an episode closed in
                // the last few minutes: the consolidator runs afterwards, out
                // of band. Say so rather than printing a bare title, so the
                // agent knows the gap is timing and not amnesia.
                None => out.push_str(&format!("- {title} (summary pending)\n")),
            }
        }
    }

    if !facts.is_empty() {
        out.push_str(&format!(
            "\nWhat is currently true about {prime_for} (fleet-wide, newest first):\n"
        ));
        for fact in facts.iter().take(MAX_FACTS) {
            let marker = match fact.status {
                FactStatus::Open => "[open] ",
                FactStatus::Settled => "",
            };
            out.push_str(&format!(
                "- {marker}{}: {}\n",
                fact.claim_key,
                cap(&fact.claim, CLAIM_BUDGET)
            ));
        }
    }

    Some(cap(&out, PACK_BUDGET).to_string())
}

/// UTF-8-safe truncation with an ellipsis, matching the rest of the crate.
fn cap(s: &str, budget: usize) -> std::borrow::Cow<'_, str> {
    if s.len() <= budget {
        return std::borrow::Cow::Borrowed(s);
    }
    std::borrow::Cow::Owned(format!(
        "{}...",
        openfang_types::truncate_str(s, budget.saturating_sub(3))
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::CloseReason;
    use crate::fact::FactStatus;
    use chrono::Utc;
    use openfang_types::agent::AgentId;
    use openfang_types::memory::MemoryId;
    use std::collections::HashMap;

    fn closed_episode(title: &str, summary: Option<&str>) -> Episode {
        let now = Utc::now();
        Episode {
            id: uuid::Uuid::new_v4(),
            agent_id: AgentId::new(),
            opened_at: now,
            last_activity_at: now,
            closed_at: Some(now),
            title: Some(title.to_string()),
            summary: summary.map(str::to_string),
            close_reason: Some(CloseReason::Explicit),
            turn_count: 4,
        }
    }

    fn open_episode(title: &str) -> Episode {
        let mut ep = closed_episode(title, Some("half-done"));
        ep.closed_at = None;
        ep.close_reason = None;
        ep
    }

    fn fact(key: &str, claim: &str, status: FactStatus) -> Fact {
        Fact {
            id: MemoryId::new(),
            authored_by: Some("someone-else".to_string()),
            scope: "project".to_string(),
            scope_ref: "openfang-fork".to_string(),
            claim_key: key.to_string(),
            claim: claim.to_string(),
            status,
            confidence: 1.0,
            episode_id: None,
            created_at: Utc::now().to_rfc3339(),
            last_affirmed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// A pack with a heading and nothing under it spends the protected
    /// index-0 slot to announce that memory is empty. Emit nothing instead.
    #[test]
    fn an_empty_pack_is_no_pack() {
        assert!(render_pack("openfang-fork", &[], &[]).is_none());
    }

    /// An agent with no closed episodes but live project claims is the normal
    /// state for a fresh agent joining existing work — the pack is still
    /// worth emitting.
    #[test]
    fn facts_alone_are_enough_to_earn_a_pack() {
        let pack = render_pack(
            "openfang-fork",
            &[],
            &[fact(
                "repo.trunk_model",
                "main is the trunk",
                FactStatus::Settled,
            )],
        )
        .expect("claims alone justify a pack");
        assert!(pack.contains("repo.trunk_model"), "{pack}");
        assert!(
            !pack.contains("Recently closed episodes"),
            "no empty section headings: {pack}"
        );
    }

    /// Open loops carry a marker. Without it the agent cannot tell a settled
    /// belief from an unfinished question, which is the distinction the whole
    /// status column exists to make.
    #[test]
    fn open_loops_are_marked_and_settled_claims_are_not() {
        let pack = render_pack(
            "openfang-fork",
            &[],
            &[
                fact("build.rebuild", "waiting on Ben", FactStatus::Open),
                fact("repo.trunk_model", "main is the trunk", FactStatus::Settled),
            ],
        )
        .unwrap();
        assert!(
            pack.contains("- [open] build.rebuild: waiting on Ben"),
            "{pack}"
        );
        assert!(
            pack.contains("- repo.trunk_model: main is the trunk"),
            "{pack}"
        );
    }

    /// An episode closed minutes ago has no summary yet — the consolidator
    /// runs out of band. Saying so beats printing a bare title, which reads
    /// like the summary was lost.
    #[test]
    fn a_summaryless_episode_says_pending_rather_than_vanishing() {
        let pack = render_pack("openfang-fork", &[closed_episode("epic 240", None)], &[]).unwrap();
        assert!(pack.contains("- epic 240 (summary pending)"), "{pack}");
    }

    /// The open episode is the one the agent is in. Briefing it on its own
    /// current work is noise at best and a stale half-summary at worst.
    #[test]
    fn the_open_episode_is_never_in_the_pack() {
        assert!(render_pack("openfang-fork", &[open_episode("in flight")], &[]).is_none());
    }

    #[test]
    fn at_most_max_episodes_are_listed() {
        let episodes: Vec<Episode> = (0..MAX_EPISODES + 3)
            .map(|i| closed_episode(&format!("episode-{i}"), Some("done")))
            .collect();
        let pack = render_pack("openfang-fork", &episodes, &[]).unwrap();
        assert!(pack.contains("episode-0"), "{pack}");
        assert!(
            !pack.contains(&format!("episode-{}", MAX_EPISODES)),
            "the list is newest-first and bounded: {pack}"
        );
    }

    /// The pack rides in the slot the trim ladder refuses to drop, so an
    /// unbounded pack is an unbounded permanent tax. The ceiling is the only
    /// thing standing between one enormous claim and every turn paying for it.
    #[test]
    fn the_pack_is_bounded_even_when_its_material_is_not() {
        let huge = "x".repeat(PACK_BUDGET * 3);
        let facts: Vec<Fact> = (0..MAX_FACTS)
            .map(|i| fact(&format!("k.{i}"), &huge, FactStatus::Settled))
            .collect();
        let pack = render_pack("openfang-fork", &[], &facts).unwrap();
        assert!(
            pack.len() <= PACK_BUDGET,
            "pack was {} chars, budget is {PACK_BUDGET}",
            pack.len()
        );
    }
}
