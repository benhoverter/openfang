//! Command-string normalization for **deny** matching (ANAI-152).
//!
//! Every base-extraction site in the exec wall tokenizes with
//! `split_whitespace()` — no quote handling, no escape collapsing, no Unicode
//! folding. A shell does not read a command line that way, so a command can be
//! written to tokenize one way for the validator and execute another way in the
//! shell:
//!
//! ```text
//!   ba""sh -c '…'      → argv[0] token is `ba""sh`, not `bash`
//!   \r\m -rf …         → token is `\r\m`, not `rm`
//!   rm  (U+0440 'р')   → a Cyrillic homoglyph is a different token entirely
//! ```
//!
//! That is a **parser differential**. Today it sits under a wall that only ever
//! *denies* — an unrecognized token is not on the allowlist, so it fails closed.
//! The moment a gate *grants* on a non-match (an auto-approve cache, an
//! Approve-Similar denylist, an LLM gatekeeper's "this looks fine"), the same
//! differential fails **unsafe**.
//!
//! # Union semantics — the load-bearing rule
//!
//! [`deny_variants`] returns the raw string **plus** progressively normalized
//! forms. Callers must match a deny rule against *every* variant and deny if
//! **any** matches. Never replace the raw form with a normalized one:
//!
//! - Union: a normalizer bug produces a *false positive deny* — noisy, visible,
//!   fails safe.
//! - Replace: a normalizer bug produces a *silent miss* — the exact failure this
//!   module exists to close.
//!
//! In-tree precedent for the union shape: `validate_command_allowlist` unions
//! `safe_bins ∪ allowed_commands ∪ trusted_commands`.
//!
//! # Deny side only
//!
//! These variants must **never** be fed to an *approve* comparison. Normalizing
//! before an allowlist check would let `r""m` fold to `rm` and match an
//! allowlisted `rm` — turning an obfuscation into a grant, which is the
//! inversion of the whole point.

/// Upper bound on input length considered for normalization (chars).
///
/// Normalization is O(n) per transform over a fixed number of transforms, but
/// the deny checks that consume the variants are themselves O(args × rules).
/// Bounding the input keeps a pathological 100 KB command line from turning a
/// linear check into a visible stall. Anything past the bound is dropped, which
/// can only *shorten* a variant — the raw string is still matched in full by the
/// caller's existing (unnormalized) pass.
pub const MAX_NORMALIZE_INPUT: usize = 8192;

/// Maximum number of variants returned (including the raw input).
pub const MAX_VARIANTS: usize = 8;

/// Invisible / formatting code points that a shell either ignores or that a
/// human reader cannot see. Their only purpose inside a command token is to
/// break a literal comparison, so they are stripped before matching.
fn is_invisible(c: char) -> bool {
    matches!(c,
        '\u{00ad}'              // soft hyphen
        | '\u{200b}'..='\u{200f}' // zero-width space/joiners, LRM/RLM
        | '\u{2028}' | '\u{2029}' // line/paragraph separators
        | '\u{202a}'..='\u{202e}' // bidi embedding/override
        | '\u{2060}'..='\u{2064}' // word joiner, invisible operators
        | '\u{feff}'            // BOM / zero-width no-break space
    )
}

/// Fold a small table of look-alike code points onto their ASCII equivalents.
///
/// Deliberately hand-rolled and small rather than a full Unicode confusables
/// table (which would be a new dependency and a much larger behavior change).
/// It covers what actually shows up: Cyrillic and Greek letters that render
/// identically to Latin ones in a chat prompt, fullwidth forms, smart quotes,
/// and non-ASCII dashes/spaces.
///
/// Under union semantics an incomplete table costs coverage, never safety.
fn fold_homoglyph(c: char) -> char {
    match c {
        // Cyrillic → Latin look-alikes
        '\u{0430}' => 'a', // а
        '\u{0435}' => 'e', // е
        '\u{043e}' => 'o', // о
        '\u{0440}' => 'p', // р
        '\u{0441}' => 'c', // с
        '\u{0445}' => 'x', // х
        '\u{0443}' => 'y', // у
        '\u{0456}' => 'i', // і
        '\u{0458}' => 'j', // ј
        '\u{04bb}' => 'h', // һ
        '\u{0410}' => 'A',
        '\u{0412}' => 'B',
        '\u{0415}' => 'E',
        '\u{041a}' => 'K',
        '\u{041c}' => 'M',
        '\u{041d}' => 'H',
        '\u{041e}' => 'O',
        '\u{0420}' => 'P',
        '\u{0421}' => 'C',
        '\u{0422}' => 'T',
        '\u{0425}' => 'X',
        // Greek → Latin look-alikes
        '\u{03b1}' => 'a', // α
        '\u{03bf}' => 'o', // ο
        '\u{03c1}' => 'p', // ρ
        '\u{03bd}' => 'v', // ν
        '\u{0391}' => 'A',
        '\u{0392}' => 'B',
        '\u{0395}' => 'E',
        '\u{039f}' => 'O',
        '\u{03a1}' => 'P',
        '\u{03a4}' => 'T',
        '\u{03a7}' => 'X',
        // Quotes
        '\u{2018}' | '\u{2019}' | '\u{201b}' | '\u{2032}' => '\'',
        '\u{201c}' | '\u{201d}' | '\u{201f}' | '\u{2033}' => '"',
        // Dashes / hyphens (a flag written with an en dash is still a flag)
        '\u{2010}'..='\u{2015}' | '\u{2212}' | '\u{fe58}' | '\u{fe63}' | '\u{ff0d}' => '-',
        // Spaces
        '\u{00a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
        // Fullwidth ASCII block → ASCII
        '\u{ff01}'..='\u{ff5e}' => {
            // FF01 ('！') maps to 0x21 ('!'), and so on across the block.
            char::from_u32(c as u32 - 0xff00 + 0x20).unwrap_or(c)
        }
        _ => c,
    }
}

fn strip_invisibles(s: &str) -> String {
    s.chars().filter(|c| !is_invisible(*c)).collect()
}

fn fold_homoglyphs(s: &str) -> String {
    s.chars().map(fold_homoglyph).collect()
}

/// Remove empty quote pairs (`""` / `''`) — the classic `pu""sh` split.
///
/// Only *adjacent* pairs are removed, so a legitimate empty argument written as
/// a standalone token still survives in the raw variant that the caller also
/// matches.
fn strip_empty_quote_pairs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if (c == '"' || c == '\'') && chars.peek() == Some(&c) {
            chars.next(); // drop the pair
            continue;
        }
        out.push(c);
    }
    out
}

/// Remove all quote characters. `"rm" -rf` and `rm -rf` run the same binary;
/// only the tokenizer disagrees.
fn strip_quotes(s: &str) -> String {
    s.chars().filter(|c| *c != '"' && *c != '\'').collect()
}

/// Collapse backslash escapes: `\r\m` → `rm`. A trailing lone backslash is
/// dropped. Note this also collapses Windows-style paths — acceptable, because
/// the result is only ever used to *widen* a deny match, and the raw form is
/// matched alongside it.
fn strip_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Deobfuscated variants of `command`, for **deny** matching only.
///
/// Returns the raw input first, then progressively normalized forms, deduped
/// and order-preserving. Callers must test a deny rule against every entry and
/// deny on any match (see the module docs on union semantics).
pub fn deny_variants(command: &str) -> Vec<String> {
    let raw: String = if command.chars().count() > MAX_NORMALIZE_INPUT {
        command.chars().take(MAX_NORMALIZE_INPUT).collect()
    } else {
        command.to_string()
    };

    let mut out: Vec<String> = Vec::with_capacity(MAX_VARIANTS);
    let push = |v: String, out: &mut Vec<String>| {
        if out.len() < MAX_VARIANTS && !out.iter().any(|e| e == &v) {
            out.push(v);
        }
    };

    push(raw.clone(), &mut out);

    // Cumulative: each step layers onto the previous, so an obfuscation that
    // stacks tricks (`ba\"\"sh` with a zero-width space inside) still folds.
    let v = strip_invisibles(&raw);
    push(v.clone(), &mut out);
    let v = fold_homoglyphs(&v);
    push(v.clone(), &mut out);
    let v = strip_empty_quote_pairs(&v);
    push(v.clone(), &mut out);
    let v = strip_quotes(&v);
    push(v.clone(), &mut out);
    let v = strip_escapes(&v);
    push(v, &mut out);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folds_to(input: &str, expected: &str) -> bool {
        deny_variants(input).iter().any(|v| v == expected)
    }

    #[test]
    fn raw_is_always_first_variant() {
        let v = deny_variants("git push --force");
        assert_eq!(v[0], "git push --force");
    }

    #[test]
    fn clean_command_yields_only_itself() {
        // Nothing to normalize ⇒ every transform is identity ⇒ one variant.
        assert_eq!(deny_variants("ls -la"), vec!["ls -la".to_string()]);
    }

    #[test]
    fn empty_quote_pairs_fold() {
        assert!(folds_to("git pu\"\"sh --force", "git push --force"));
        assert!(folds_to("git pu''sh --force", "git push --force"));
    }

    #[test]
    fn quotes_fold() {
        assert!(folds_to("\"rm\" -rf /tmp/x", "rm -rf /tmp/x"));
    }

    #[test]
    fn escapes_fold() {
        assert!(folds_to("\\r\\m -rf /tmp/x", "rm -rf /tmp/x"));
    }

    #[test]
    fn invisibles_fold() {
        assert!(folds_to("r\u{200b}m -rf /tmp/x", "rm -rf /tmp/x"));
        assert!(folds_to("ba\u{feff}sh -c x", "bash -c x"));
    }

    #[test]
    fn cyrillic_homoglyph_folds() {
        // Cyrillic 'р' (U+0440) + ASCII 'm'
        assert!(folds_to("\u{0440}m -rf /tmp/x", "pm -rf /tmp/x"));
        // Cyrillic 'с' in "sudo"-adjacent position: 'с' → 'c'
        assert!(folds_to("\u{0441}hmod 777 /", "chmod 777 /"));
    }

    #[test]
    fn fullwidth_folds() {
        assert!(folds_to(
            "\u{ff42}\u{ff41}\u{ff53}\u{ff48} -c x",
            "bash -c x"
        ));
    }

    #[test]
    fn en_dash_flag_folds() {
        assert!(folds_to("bash \u{2013}c x", "bash -c x"));
    }

    #[test]
    fn stacked_obfuscation_folds() {
        // zero-width + empty quotes + escape, all at once
        assert!(folds_to("ba\u{200b}\"\"\\sh -c x", "bash -c x"));
    }

    #[test]
    fn variants_are_bounded() {
        let nasty = "\"\"\\r\u{200b}\u{0440}m ".repeat(500);
        assert!(deny_variants(&nasty).len() <= MAX_VARIANTS);
    }

    #[test]
    fn oversized_input_is_truncated_not_rejected() {
        let long = "a".repeat(MAX_NORMALIZE_INPUT + 100);
        let v = deny_variants(&long);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].chars().count(), MAX_NORMALIZE_INPUT);
    }

    #[test]
    fn no_panic_on_empty_or_lone_backslash() {
        assert_eq!(deny_variants(""), vec!["".to_string()]);
        let v = deny_variants("rm \\");
        assert!(v.iter().any(|s| s == "rm "));
    }
}
