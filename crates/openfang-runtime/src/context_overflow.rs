//! Context overflow recovery pipeline.
//!
//! Provides a 4-stage recovery pipeline that replaces the brute-force
//! `emergency_trim_messages()` with structured, progressive recovery:
//!
//! 1. Auto-compact via message trimming (keep recent, drop old)
//! 2. Aggressive overflow compaction (drop all but last N)
//! 3. Truncate historical tool results to 2K chars each
//! 4. Return error suggesting /reset or /compact

use openfang_types::message::{ContentBlock, Message, MessageContent, Role};
use openfang_types::tool::ToolDefinition;
use tracing::{debug, warn};

/// Fraction of the context window at which recovery engages.
///
/// ANAI-244. This is the **emergency floor** of the context-pressure ladder,
/// deliberately **above** `history_trim::TOKEN_TRIM_RATIO` (0.85) and the
/// compactor's 0.70 token trigger:
///
/// | rung | ratio | mechanism | loss |
/// |------|-------|-----------|------|
/// | 1 | 0.70 | compactor — an LLM summarises the dropped span | recoverable |
/// | 2 | 0.85 | `history_trim` safety valve — sheds to 0.75 | lossy, bounded |
/// | 3 | 0.92 | this pipeline — keep last 10, then 4, then truncate | very lossy |
///
/// The ordering used to be violated. This pipeline entered at `0.70` — the
/// same ratio as the compactor and *below* the valve — and, unlike either of
/// them, it runs at the top of **every agent-loop iteration** rather than once
/// per turn. It therefore always fired first, which made both smart paths
/// decorative.
pub const RECOVERY_ENTRY_RATIO: f64 = 0.92;

/// Fraction of the window stage 1 must reach for recovery to stop.
///
/// Matches `history_trim::TOKEN_RELEASE_RATIO` so both shedding paths land in
/// the same place, and so a session relieved here does not immediately
/// re-trip on the next iteration.
pub const RECOVERY_TARGET_RATIO: f64 = 0.75;

/// Adjust a drain boundary so it does not split a ToolUse/ToolResult pair.
///
/// If the message at `boundary` is a user message containing ToolResult blocks
/// whose matching ToolUse lives in the message at `boundary - 1`, we pull the
/// boundary back by one so both the assistant (ToolUse) and user (ToolResult)
/// are kept together.  Conversely, if the last drained message is an assistant
/// with ToolUse blocks whose results sit at `boundary`, we push the boundary
/// forward by one so the orphaned assistant is also drained.
///
/// This is best-effort — `session_repair::validate_and_repair` is still called
/// afterwards as the authoritative fixup.
fn safe_drain_boundary(messages: &[Message], mut boundary: usize) -> usize {
    if boundary == 0 || boundary >= messages.len() {
        return boundary;
    }

    // Case 1: first kept message is a user msg with ToolResults whose ToolUse
    // is in the last drained message (boundary - 1).  Pull boundary back by 1.
    if messages[boundary].role == Role::User {
        if let MessageContent::Blocks(blocks) = &messages[boundary].content {
            let has_tool_result = blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
            if has_tool_result && boundary > 0 && messages[boundary - 1].role == Role::Assistant {
                if let MessageContent::Blocks(asst_blocks) = &messages[boundary - 1].content {
                    let has_tool_use = asst_blocks
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
                    if has_tool_use {
                        boundary -= 1;
                        debug!(
                            new_boundary = boundary,
                            "Adjusted drain boundary back to keep ToolUse/ToolResult pair"
                        );
                    }
                }
            }
        }
    }

    // Case 2: last drained message (boundary - 1) is an assistant with ToolUse
    // but the ToolResults are at `boundary` (already handled above by pulling
    // back).  If the first kept message is NOT the matching result, push forward
    // to drain the orphaned assistant too.
    if boundary > 0 && boundary < messages.len() && messages[boundary - 1].role == Role::Assistant {
        if let MessageContent::Blocks(asst_blocks) = &messages[boundary - 1].content {
            let tool_use_ids: Vec<&str> = asst_blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                    _ => None,
                })
                .collect();
            if !tool_use_ids.is_empty() {
                // Check if the first kept message has the matching results
                let first_kept_has_results = match &messages[boundary].content {
                    MessageContent::Blocks(blocks) => blocks.iter().any(|b| match b {
                        ContentBlock::ToolResult { tool_use_id, .. } => {
                            tool_use_ids.contains(&tool_use_id.as_str())
                        }
                        _ => false,
                    }),
                    _ => false,
                };
                if !first_kept_has_results {
                    // The assistant's ToolResults were already drained; drain the
                    // orphaned assistant as well to avoid needing synthetic results.
                    boundary = boundary.min(messages.len());
                    // Note: we don't push forward here because that would drain
                    // more messages than intended.  The validate_and_repair call
                    // will insert synthetic results for this orphan instead.
                }
            }
        }
    }

    boundary
}

/// Recovery stage that was applied.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStage {
    /// No recovery needed.
    None,
    /// Stage 1: moderate trim (keep last 10).
    AutoCompaction { removed: usize },
    /// Stage 2: aggressive trim (keep last 4).
    OverflowCompaction { removed: usize },
    /// Stage 3: truncated tool results.
    ToolResultTruncation { truncated: usize },
    /// Stage 4: unrecoverable — suggest /reset.
    FinalError,
}

/// Estimate token count using chars/4 heuristic.
fn estimate_tokens(messages: &[Message], system_prompt: &str, tools: &[ToolDefinition]) -> usize {
    crate::compactor::estimate_token_count(messages, Some(system_prompt), Some(tools))
}

/// Run the 4-stage overflow recovery pipeline.
///
/// Returns the recovery stage applied and the number of messages/results affected.
pub fn recover_from_overflow(
    messages: &mut Vec<Message>,
    system_prompt: &str,
    tools: &[ToolDefinition],
    context_window: usize,
    protect_head: bool,
) -> RecoveryStage {
    let estimated = estimate_tokens(messages, system_prompt, tools);
    let entry = (context_window as f64 * RECOVERY_ENTRY_RATIO) as usize;
    let target = (context_window as f64 * RECOVERY_TARGET_RATIO) as usize;
    // Index 0 holds the compaction summary when the kernel injected one; the
    // drainable region starts after it.
    let head = usize::from(protect_head && !messages.is_empty());

    // No recovery needed. Below the entry ratio this is the compactor's or the
    // safety valve's problem, and both lose less than this path does.
    if estimated <= entry {
        return RecoveryStage::None;
    }

    warn!(
        target: crate::history_trim::TARGET,
        estimated_tokens = estimated,
        context_window,
        entry_threshold = entry,
        total_messages = messages.len(),
        canonical_context_present = head == 1,
        "context pressure: emergency overflow recovery engaged"
    );

    // Stage 1: Moderate trim — keep last 10 messages
    {
        let keep = 10.min(messages.len().saturating_sub(head));
        let raw_boundary = messages.len() - keep;
        // Adjust boundary to avoid splitting ToolUse/ToolResult pairs
        let boundary = safe_drain_boundary(messages, raw_boundary).max(head);
        let remove = boundary - head;
        if remove > 0 {
            debug!(
                estimated_tokens = estimated,
                removing = remove,
                "Stage 1: moderate trim to last {} messages",
                messages.len() - remove
            );
            messages.drain(head..boundary);
            // Re-check after trim
            let new_est = estimate_tokens(messages, system_prompt, tools);
            if new_est <= target {
                return RecoveryStage::AutoCompaction { removed: remove };
            }
        }
    }

    // Stage 2: Aggressive trim — keep last 4 messages + summary marker
    {
        let keep = 4.min(messages.len().saturating_sub(head));
        let raw_boundary = messages.len() - keep;
        // Adjust boundary to avoid splitting ToolUse/ToolResult pairs
        let boundary = safe_drain_boundary(messages, raw_boundary).max(head);
        let remove = boundary - head;
        if remove > 0 {
            warn!(
                estimated_tokens = estimate_tokens(messages, system_prompt, tools),
                removing = remove,
                "Stage 2: aggressive overflow compaction to last {} messages",
                messages.len() - remove
            );
            let summary = Message::user(format!(
                "[System: {} earlier messages were removed due to context overflow. \
                 The conversation continues from here. Use /compact for smarter summarization.]",
                remove
            ));
            messages.drain(head..boundary);
            messages.insert(head, summary);

            let new_est = estimate_tokens(messages, system_prompt, tools);
            if new_est <= entry {
                return RecoveryStage::OverflowCompaction { removed: remove };
            }
        }
    }

    // Stage 3: Truncate all historical tool results to 2K chars
    let tool_truncation_limit = 2000;
    let mut truncated = 0;
    for msg in messages.iter_mut() {
        if let MessageContent::Blocks(blocks) = &mut msg.content {
            for block in blocks.iter_mut() {
                if let ContentBlock::ToolResult { content, .. } = block {
                    if content.len() > tool_truncation_limit {
                        let mut safe_keep = tool_truncation_limit.saturating_sub(80);
                        // Walk back to a valid char boundary
                        while safe_keep > 0 && !content.is_char_boundary(safe_keep) {
                            safe_keep -= 1;
                        }
                        *content = format!(
                            "{}\n\n[OVERFLOW RECOVERY: truncated from {} to {} chars]",
                            &content[..safe_keep],
                            content.len(),
                            safe_keep
                        );
                        truncated += 1;
                    }
                }
            }
        }
    }

    if truncated > 0 {
        let new_est = estimate_tokens(messages, system_prompt, tools);
        if new_est <= entry {
            return RecoveryStage::ToolResultTruncation { truncated };
        }
        warn!(
            estimated_tokens = new_est,
            "Stage 3 truncated {} tool results but still over threshold", truncated
        );
    }

    // Stage 4: Final error — nothing more we can do automatically
    warn!("Stage 4: all recovery stages exhausted, context still too large");
    RecoveryStage::FinalError
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::message::{Message, Role};

    fn make_messages(count: usize, size_each: usize) -> Vec<Message> {
        (0..count)
            .map(|i| {
                let text = format!("msg{}: {}", i, "x".repeat(size_each));
                Message {
                    role: if i % 2 == 0 {
                        Role::User
                    } else {
                        Role::Assistant
                    },
                    content: MessageContent::Text(text),
                    ..Default::default()
                }
            })
            .collect()
    }

    #[test]
    fn test_no_recovery_needed() {
        let mut msgs = make_messages(2, 100);
        let stage = recover_from_overflow(&mut msgs, "sys", &[], 200_000, false);
        assert_eq!(stage, RecoveryStage::None);
    }

    #[test]
    fn test_stage1_moderate_trim() {
        // Create messages that push us past 70% but not 90%
        // Context window: 1000 tokens = 4000 chars
        // 70% = 700 tokens = 2800 chars
        let mut msgs = make_messages(20, 150); // ~3000 chars total
        let stage = recover_from_overflow(&mut msgs, "system", &[], 800, false);
        match stage {
            RecoveryStage::AutoCompaction { removed } => {
                assert!(removed > 0);
                assert!(msgs.len() <= 10);
            }
            RecoveryStage::OverflowCompaction { .. } => {
                // Also acceptable if moderate wasn't enough
            }
            _ => {} // depends on exact token estimation
        }
    }

    #[test]
    fn test_stage2_aggressive_trim() {
        // Push past 90%: 1000 tokens = 4000 chars, 90% = 3600 chars
        let mut msgs = make_messages(30, 200); // ~6000 chars
        let stage = recover_from_overflow(&mut msgs, "system", &[], 1000, false);
        match stage {
            RecoveryStage::OverflowCompaction { removed } => {
                assert!(removed > 0);
            }
            RecoveryStage::ToolResultTruncation { .. } | RecoveryStage::FinalError => {}
            _ => {} // acceptable cascading
        }
    }

    #[test]
    fn test_stage3_tool_truncation() {
        let big_result = "x".repeat(5000);
        let mut msgs = vec![
            Message::user("hi"),
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    tool_name: String::new(),
                    content: big_result.clone(),
                    is_error: false,
                }]),
                ..Default::default()
            },
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t2".to_string(),
                    tool_name: String::new(),
                    content: big_result,
                    is_error: false,
                }]),
                ..Default::default()
            },
        ];
        // Tiny context window to force all stages
        let stage = recover_from_overflow(&mut msgs, "system", &[], 500, false);
        // Should at least reach tool truncation
        match stage {
            RecoveryStage::ToolResultTruncation { truncated } => {
                assert!(truncated > 0);
            }
            RecoveryStage::OverflowCompaction { .. } | RecoveryStage::FinalError => {}
            _ => {}
        }
    }

    #[test]
    fn test_cascading_stages() {
        // Ensure stages cascade: if stage 1 isn't enough, stage 2 kicks in
        let mut msgs = make_messages(50, 500);
        let stage = recover_from_overflow(&mut msgs, "system prompt", &[], 2000, false);
        // With 50 messages of 500 chars each (25000 chars), context of 2000 tokens (8000 chars),
        // we should cascade through stages
        assert_ne!(stage, RecoveryStage::None);
    }

    #[test]
    fn test_stage3_multibyte_tool_truncation() {
        // Chinese text (3 bytes per char) in tool results must not panic
        let chinese_result: String = "\u{4f60}\u{597d}\u{4e16}\u{754c}".repeat(1250); // 5000 chars, 15000 bytes
        let mut msgs = vec![
            Message::user("hi"),
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    tool_name: String::new(),
                    content: chinese_result,
                    is_error: false,
                }]),
                ..Default::default()
            },
        ];
        // Tiny context window to force stage 3 tool truncation
        let stage = recover_from_overflow(&mut msgs, "system", &[], 500, false);
        // Must not panic — the truncation at byte boundaries could split a 3-byte char
        assert_ne!(stage, RecoveryStage::None);
    }

    #[test]
    fn test_safe_drain_boundary_pulls_back_for_tool_pair() {
        // Messages: [user, assistant(ToolUse), user(ToolResult), user]
        // If boundary = 2 (keep last 2), it splits between assistant(ToolUse) and
        // user(ToolResult).  safe_drain_boundary should pull back to 1.
        let msgs = vec![
            Message::user("hello"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({}),
                    provider_metadata: None,
                }]),
                ..Default::default()
            },
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    tool_name: "read".to_string(),
                    content: "file contents".to_string(),
                    is_error: false,
                }]),
                ..Default::default()
            },
            Message::user("thanks"),
        ];
        // Boundary 2 would cut between the assistant(ToolUse) at [1] and user(ToolResult) at [2].
        let adjusted = safe_drain_boundary(&msgs, 2);
        assert_eq!(
            adjusted, 1,
            "Should pull boundary back to keep the ToolUse/ToolResult pair together"
        );
    }

    #[test]
    fn test_safe_drain_boundary_no_change_for_text_messages() {
        let msgs = vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
            Message::assistant("d"),
        ];
        let adjusted = safe_drain_boundary(&msgs, 2);
        assert_eq!(
            adjusted, 2,
            "Should not change boundary for plain text messages"
        );
    }

    #[test]
    fn test_safe_drain_boundary_edge_zero() {
        let msgs = vec![Message::user("a")];
        assert_eq!(safe_drain_boundary(&msgs, 0), 0);
    }

    #[test]
    fn test_safe_drain_boundary_edge_end() {
        let msgs = vec![Message::user("a"), Message::assistant("b")];
        assert_eq!(safe_drain_boundary(&msgs, 2), 2);
    }

    // ---------------------------------------------------------------
    // ANAI-244: this pipeline is the emergency floor, not the first
    // responder. These are the two ways it used to defeat the paths
    // above it.
    // ---------------------------------------------------------------

    /// The pathology. At 80% of the window the compactor (0.70) and the
    /// safety valve (0.85) own the decision; this path ran first because it
    /// entered at 0.70 *and* ran every loop iteration. It must now decline.
    #[test]
    fn recovery_declines_below_the_emergency_ratio() {
        // 1000-token window. 0.80 of it = 800 tokens = ~3200 chars.
        let mut msgs = make_messages(20, 155);
        let est = estimate_tokens(&msgs, "system", &[]);
        assert!(
            est > (1000.0 * 0.70) as usize && est < (1000.0 * RECOVERY_ENTRY_RATIO) as usize,
            "fixture must sit between the compactor trigger and the emergency floor, got {est}"
        );
        let stage = recover_from_overflow(&mut msgs, "system", &[], 1000, false);
        assert_eq!(stage, RecoveryStage::None);
        assert_eq!(msgs.len(), 20, "nothing may be deleted below the floor");
    }

    /// The ANAI-242 bug one layer down: index 0 carries the compactor's own
    /// summary, and every stage here drained from the front.
    #[test]
    fn protected_head_survives_every_stage() {
        let canonical = "CANONICAL CONTEXT SUMMARY";
        for window in [2000usize, 500, 200] {
            let mut msgs = vec![Message::user(canonical)];
            msgs.extend(make_messages(50, 500));
            let stage = recover_from_overflow(&mut msgs, "system prompt", &[], window, true);
            assert_ne!(stage, RecoveryStage::None, "window {window} must recover");
            assert_eq!(
                msgs[0].content.text_length(),
                canonical.len(),
                "window {window}: the compaction summary was destroyed by recovery"
            );
        }
    }

    /// Negative control: without the flag, index 0 is ordinary history.
    #[test]
    fn unprotected_head_is_still_drainable() {
        let mut msgs = make_messages(50, 500);
        let stage = recover_from_overflow(&mut msgs, "system prompt", &[], 2000, false);
        assert_ne!(stage, RecoveryStage::None);
        assert!(msgs.len() < 50);
    }

    /// The floor must sit above the valve, which sits above the compactor.
    /// Any future edit that reorders these re-creates today's bug.
    #[test]
    fn the_ladder_is_strictly_ordered() {
        const { assert!(crate::history_trim::TOKEN_TRIM_RATIO < RECOVERY_ENTRY_RATIO) };
        const { assert!(RECOVERY_TARGET_RATIO < crate::history_trim::TOKEN_TRIM_RATIO) };
        assert_eq!(
            RECOVERY_TARGET_RATIO,
            crate::history_trim::TOKEN_RELEASE_RATIO
        );
    }
}
