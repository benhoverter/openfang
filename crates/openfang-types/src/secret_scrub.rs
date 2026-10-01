//! Replace credentials in text before it reaches a model.
//!
//! ## Why this exists
//!
//! A tool result is copied verbatim into the model's context, the session
//! history, and memory. One `pgrep -lf` printed MCP server command lines with
//! their bearer tokens in argv, and every token in that output had to be
//! rotated. This module is the filter that sits between a tool's output and
//! the model.
//!
//! ## What it does
//!
//! Two passes, strongest first:
//!
//! 1. **Known secrets.** The daemon knows the actual values it holds (MCP
//!    server env and headers, credential-named process env). Any exact
//!    occurrence is replaced. This cannot misfire on ordinary text and catches
//!    every key we own regardless of its shape.
//! 2. **Secret-shaped spans.** For keys we do not hold: known credential
//!    prefixes (`ghp_`, `sk-`, `sbp_`, ...), long opaque mixed-case runs,
//!    PEM private-key blocks, and the value side of a credential-named
//!    assignment (`API_TOKEN=...`).
//!
//! ## How a replacement looks
//!
//! Each secret becomes a **stand-in of the same shape**: same length, same
//! prefix, same character classes. `sk-ant-abc123...` stays recognisable as
//! an Anthropic key. The stand-in is derived from a keyed SHA-256 over the
//! secret, with a key generated per [`SecretScrubber`] and never exposed, so:
//!
//! - the same secret always maps to the same stand-in for the life of the
//!   scrubber, so "the key in A matches the key in B" still holds;
//! - the stand-in cannot be reversed to the secret without the key.
//!
//! ## What it deliberately does not do
//!
//! **It is one-way.** Nothing maps a stand-in back to the real value on the
//! way out. If it did, a model could put the stand-in into a URL and the
//! daemon would helpfully substitute the real credential for it. Agents that
//! need a credential get it from their environment.
//!
//! Pure-hex runs are never treated as secrets by shape: they are git SHAs,
//! content digests and state tokens, and scrambling them breaks ordinary work.
//! A hex secret we hold is still caught by pass 1.

use sha2::{Digest, Sha256};

/// Credential prefixes. A run starting with one of these, followed by at least
/// [`MIN_PREFIXED_BODY`] more token characters including a digit, is a secret.
pub const CREDENTIAL_PREFIXES: &[&str] = &[
    "sk-ant-",
    "sk-proj-",
    "sk-",
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "pk_live_",
    "ghp_",
    "gho_",
    "ghs_",
    "ghu_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xapp-",
    "sbp_",
    "lin_api_",
    "lin_oauth_",
    "ntn_",
    "secret_",
    "hf_",
    "npm_",
    "AIza",
    "AKIA",
    "ASIA",
    "eyJ",
];

/// Minimum characters after a prefix for the run to count as a credential.
/// Keeps `sk-learn` and `secret_key` out.
const MIN_PREFIXED_BODY: usize = 16;

/// Minimum length of an unprefixed opaque run.
const MIN_OPAQUE_LEN: usize = 32;

/// Known secrets shorter than this are ignored: matching `true` or `8080`
/// everywhere would wreck output for no protection.
pub const MIN_KNOWN_SECRET_LEN: usize = 12;

/// Minimum value length for the credential-named assignment rule.
/// 12, not 8: at 8, Rust type annotations like `TOKEN_BITS: AtomicU64` match.
const MIN_ASSIGNED_VALUE_LEN: usize = 12;

/// An unprefixed opaque run must contain an unbroken segment (between `-` or
/// `_`) at least this long. Real random tokens do; hyphenated identifiers like
/// `Llama-4-Maverick-17B-128E-Instruct-FP8` do not.
const MIN_OPAQUE_SEGMENT: usize = 20;

/// Base64 file signatures. An inline image or PDF is not a credential, and
/// scrambling it breaks any patch whose context includes it.
const BASE64_FILE_MAGIC: &[&str] = &["iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR", "JVBERi0"];

/// Substrings of an assignment key that mark its value as a credential.
/// Narrower than `path_facts::SECRET_KEY_HINTS`: bare `auth` would also hit
/// `GIT_AUTHOR_EMAIL=` and `author=` in ordinary output.
const CREDENTIAL_KEY_HINTS: &[&str] = &[
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "api_key",
    "api-key",
    "credential",
    "private_key",
    "access_key",
    "authorization",
    "bearer",
];

/// Substrings of an environment variable *name* that mark its value as a
/// credential worth exact-matching. Used by callers assembling the known set.
pub const CREDENTIAL_ENV_HINTS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "AUTH",
];

/// True when an environment variable name reads like it holds a credential.
#[must_use]
pub fn env_name_is_credential(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    CREDENTIAL_ENV_HINTS.iter().any(|h| upper.contains(h))
}

/// Pull candidate secrets out of one command-line style string, e.g. an MCP
/// server's `args` entry or an `Authorization: Bearer ...` header value.
///
/// Returns the parts worth exact-matching: anything after `Bearer `, the value
/// of a `--flag=value` whose flag reads like a credential, and any part whose
/// shape is already credential-like.
#[must_use]
pub fn secrets_in_arg(arg: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in arg.split_whitespace() {
        let part = part.trim_matches(|c| c == '"' || c == '\'');
        if let Some((k, v)) = part.split_once('=') {
            if key_is_credential(k) {
                out.push(v.to_string());
            }
        }
        if run_is_secret_shaped(part) {
            out.push(part.to_string());
        }
    }
    if let Some(idx) = arg.find("Bearer ") {
        let rest = arg[idx + "Bearer ".len()..].trim();
        let rest = rest.split_whitespace().next().unwrap_or("");
        out.push(rest.trim_matches(|c| c == '"' || c == '\'').to_string());
    }
    if let Some((k, v)) = arg.split_once(':') {
        if key_is_credential(k.trim()) {
            let v = v.trim();
            let v = v.strip_prefix("Bearer ").unwrap_or(v).trim();
            out.push(v.to_string());
        }
    }
    out.retain(|s| s.len() >= MIN_KNOWN_SECRET_LEN);
    out
}

fn key_is_credential(key: &str) -> bool {
    let k = key.trim_start_matches('-').to_ascii_lowercase();
    CREDENTIAL_KEY_HINTS.iter().any(|h| k.contains(h))
}

/// Replaces credentials in text with same-shape stand-ins. See the module docs.
pub struct SecretScrubber {
    /// Exact values, longest first so a secret that contains another is
    /// replaced whole.
    known: Vec<String>,
    key: [u8; 32],
}

impl std::fmt::Debug for SecretScrubber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the known values or the key.
        f.debug_struct("SecretScrubber")
            .field("known_secrets", &self.known.len())
            .finish_non_exhaustive()
    }
}

/// Result of scrubbing one piece of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scrubbed {
    pub text: String,
    /// Replacements made by the known-secret pass.
    pub known: usize,
    /// Replacements made by the shape pass.
    pub shaped: usize,
}

impl Scrubbed {
    #[must_use]
    pub fn total(&self) -> usize {
        self.known + self.shaped
    }
}

impl SecretScrubber {
    /// Build a scrubber over the given known values with a fresh random key.
    /// Values shorter than [`MIN_KNOWN_SECRET_LEN`] are dropped.
    #[must_use]
    pub fn new(known: impl IntoIterator<Item = String>) -> Self {
        Self::with_key(known, rand::random())
    }

    /// Build with an explicit key. Tests use this for reproducible stand-ins.
    #[must_use]
    pub fn with_key(known: impl IntoIterator<Item = String>, key: [u8; 32]) -> Self {
        let mut known: Vec<String> = known
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() >= MIN_KNOWN_SECRET_LEN)
            .collect();
        known.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        known.dedup();
        Self { known, key }
    }

    /// How many exact values this scrubber watches for.
    #[must_use]
    pub fn known_len(&self) -> usize {
        self.known.len()
    }

    /// Scrub `text`. Cheap when there is nothing to do: no allocation beyond
    /// the returned copy.
    #[must_use]
    pub fn scrub(&self, text: &str) -> Scrubbed {
        let mut out = text.to_string();
        let mut known = 0usize;
        for secret in &self.known {
            let n = out.matches(secret.as_str()).count();
            if n > 0 {
                known += n;
                let fake = self.stand_in(secret);
                out = out.replace(secret.as_str(), &fake);
            }
        }
        let (out, pem) = self.scrub_pem(&out);
        let (out, assigned) = self.scrub_assignments(&out);
        let (out, runs) = self.scrub_runs(&out);
        Scrubbed {
            text: out,
            known,
            shaped: pem + assigned + runs,
        }
    }

    /// Same-shape stand-in: keep the credential prefix, then replace each
    /// character with one of the same class drawn from a keyed digest stream.
    /// Punctuation inside the body is kept so the shape survives.
    #[must_use]
    pub fn stand_in(&self, secret: &str) -> String {
        let prefix_len = CREDENTIAL_PREFIXES
            .iter()
            .find(|p| secret.starts_with(**p))
            .map_or(0, |p| p.len());
        let (prefix, body) = secret.split_at(prefix_len);

        let mut stream: Vec<u8> = Vec::with_capacity(body.len());
        let mut counter: u32 = 0;
        while stream.len() < body.len() {
            let mut h = Sha256::new();
            h.update(self.key);
            h.update(secret.as_bytes());
            h.update(counter.to_le_bytes());
            stream.extend_from_slice(&h.finalize());
            counter += 1;
        }

        let mut out = String::with_capacity(secret.len());
        out.push_str(prefix);
        for (c, r) in body.chars().zip(stream.iter()) {
            let r = *r;
            let replaced = if c.is_ascii_digit() {
                char::from(b'0' + r % 10)
            } else if c.is_ascii_lowercase() {
                char::from(b'a' + r % 26)
            } else if c.is_ascii_uppercase() {
                char::from(b'A' + r % 26)
            } else {
                c
            };
            out.push(replaced);
        }
        // Non-ASCII tails (not expected in a credential) are carried over
        // rather than dropped, so the length never shrinks silently.
        if body.chars().count() > stream.len() {
            out.extend(body.chars().skip(stream.len()));
        }
        out
    }

    /// Replace whole PEM private-key blocks.
    fn scrub_pem(&self, text: &str) -> (String, usize) {
        const BEGIN: &str = "-----BEGIN ";
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        let mut count = 0usize;
        while let Some(start) = rest.find(BEGIN) {
            let header_end = rest[start..].find('\n').map(|i| start + i);
            let is_private = header_end
                .map(|e| rest[start..e].contains("PRIVATE KEY"))
                .unwrap_or(false);
            let end = rest[start..].find("-----END ").and_then(|e| {
                let e = start + e;
                rest[e + 9..].find("-----").map(|t| e + 9 + t + 5)
            });
            match (is_private, end) {
                (true, Some(end)) => {
                    out.push_str(&rest[..start]);
                    out.push_str("-----BEGIN PRIVATE KEY-----\n[openfang: private key removed]\n-----END PRIVATE KEY-----");
                    count += 1;
                    rest = &rest[end..];
                }
                _ => {
                    out.push_str(&rest[..start + BEGIN.len()]);
                    rest = &rest[start + BEGIN.len()..];
                }
            }
        }
        out.push_str(rest);
        (out, count)
    }

    /// `NAME=value`, `NAME: value`, `"name": "value"` where NAME reads like a
    /// credential: replace the value.
    fn scrub_assignments(&self, text: &str) -> (String, usize) {
        let mut count = 0usize;
        let mut out = String::with_capacity(text.len());
        for (i, line) in text.split('\n').enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&self.scrub_assignment_line(line, &mut count));
        }
        (out, count)
    }

    fn scrub_assignment_line(&self, line: &str, count: &mut usize) -> String {
        let bytes = line.as_bytes();
        let mut out = String::with_capacity(line.len());
        let mut cursor = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            if b != b'=' && b != b':' {
                i += 1;
                continue;
            }
            // Key: the identifier run immediately left of the separator,
            // allowing one closing quote.
            let mut k_end = i;
            if k_end > 0 && (bytes[k_end - 1] == b'"' || bytes[k_end - 1] == b'\'') {
                k_end -= 1;
            }
            let mut k_start = k_end;
            while k_start > 0 && is_key_byte(bytes[k_start - 1]) {
                k_start -= 1;
            }
            let key = &line[k_start..k_end];
            // Value: skip spaces and one opening quote, then a token run.
            let mut v_start = i + 1;
            while v_start < bytes.len() && bytes[v_start] == b' ' {
                v_start += 1;
            }
            if v_start < bytes.len() && (bytes[v_start] == b'"' || bytes[v_start] == b'\'') {
                v_start += 1;
            }
            // `Bearer xyz` as a value: the credential is the second word.
            if line[v_start..].starts_with("Bearer ") {
                v_start += "Bearer ".len();
            }
            let mut v_end = v_start;
            while v_end < bytes.len() && is_value_byte(bytes[v_end]) {
                v_end += 1;
            }
            let value = &line[v_start..v_end];
            if !key.is_empty() && key_is_credential(key) && assigned_value_is_credential(value) {
                out.push_str(&line[cursor..v_start]);
                out.push_str(&self.stand_in(value));
                *count += 1;
                cursor = v_end;
                i = v_end;
            } else {
                i += 1;
            }
        }
        out.push_str(&line[cursor..]);
        out
    }

    /// Replace token runs that are credential-shaped on their own.
    fn scrub_runs(&self, text: &str) -> (String, usize) {
        let mut out = String::with_capacity(text.len());
        let mut count = 0usize;
        let mut run_start: Option<usize> = None;
        for (idx, c) in text.char_indices() {
            if is_run_char(c) {
                if run_start.is_none() {
                    run_start = Some(idx);
                }
                continue;
            }
            if let Some(s) = run_start.take() {
                self.emit_run(&text[s..idx], &mut out, &mut count);
            }
            out.push(c);
        }
        if let Some(s) = run_start {
            self.emit_run(&text[s..], &mut out, &mut count);
        }
        (out, count)
    }

    fn emit_run(&self, run: &str, out: &mut String, count: &mut usize) {
        if run_is_secret_shaped(run) {
            out.push_str(&self.stand_in(run));
            *count += 1;
        } else {
            out.push_str(run);
        }
    }
}

fn is_key_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

/// Value side of a credential-named assignment.
///
/// Each exclusion here came from running the shape pass over the repo's own
/// source: a credential value carries at least two digits and two letters
/// (`AtomicU64`, `4912661846655238145_u64` out), is not a path
/// (`API_TOKEN=/Users/x/.ssh/key` out), and is not itself an environment
/// variable name (`api_key_env: "AI21_API_KEY"` out).
fn assigned_value_is_credential(value: &str) -> bool {
    if value.len() < MIN_ASSIGNED_VALUE_LEN {
        return false;
    }
    if value.starts_with(['/', '~', '.']) {
        return false;
    }
    if value
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    {
        return false;
    }
    // Hex with optional dashes: a UUID or digest, same exemption as the run
    // rule. A hex credential we hold is still caught by the known pass.
    if value.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        return false;
    }
    value.chars().filter(|c| c.is_ascii_digit()).count() >= 2
        && value.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 2
}

fn is_value_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'+' | b'/' | b'=' | b'.' | b'~')
}

fn is_run_char(c: char) -> bool {
    // `=` is deliberately not a run character: it would weld `NAME=value`
    // into one run and scramble the variable name along with the value.
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+')
}

/// True when a token run is credential-shaped without any context.
fn run_is_secret_shaped(run: &str) -> bool {
    let run = run.trim_end_matches('=');
    if let Some(p) = CREDENTIAL_PREFIXES.iter().find(|p| run.starts_with(**p)) {
        let body = &run[p.len()..];
        return body.len() >= MIN_PREFIXED_BODY && body.chars().any(|c| c.is_ascii_digit());
    }
    if run.len() < MIN_OPAQUE_LEN {
        return false;
    }
    // Hex (with or without dashes) is a digest, a SHA, a UUID or a state token
    // far more often than a key.
    if run.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        return false;
    }
    // Lockfile integrity hashes.
    if run.starts_with("sha256-") || run.starts_with("sha384-") || run.starts_with("sha512-") {
        return false;
    }
    if BASE64_FILE_MAGIC.iter().any(|m| run.starts_with(m)) {
        return false;
    }
    let longest_segment = run.split(['-', '_']).map(str::len).max().unwrap_or(0);
    if longest_segment < MIN_OPAQUE_SEGMENT {
        return false;
    }
    run.chars().any(|c| c.is_ascii_digit())
        && run.chars().any(|c| c.is_ascii_lowercase())
        && run.chars().any(|c| c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; 32] = [7u8; 32];

    fn scrubber(known: &[&str]) -> SecretScrubber {
        SecretScrubber::with_key(known.iter().map(|s| (*s).to_string()), KEY)
    }

    /// The incident: `pgrep -lf` printing an MCP server's argv.
    #[test]
    fn pgrep_bearer_line_is_scrubbed_by_known_value() {
        let token = "sbp_live9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1";
        let s = scrubber(&[token]);
        let line = format!(
            "41234 node /x/mcp-remote https://mcp.example --header Authorization:Bearer {token}"
        );
        let r = s.scrub(&line);
        assert!(!r.text.contains(token), "{}", r.text);
        assert!(r.known >= 1);
        assert!(r.text.contains("sbp_"), "prefix survives: {}", r.text);
    }

    #[test]
    fn unknown_prefixed_token_is_scrubbed_by_shape() {
        let s = scrubber(&[]);
        let tok = "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8";
        let r = s.scrub(&format!("token is {tok} ok"));
        assert!(!r.text.contains(tok));
        assert!(r.text.starts_with("token is ghp_"));
        assert!(r.text.ends_with(" ok"));
        assert_eq!(r.shaped, 1);
    }

    #[test]
    fn stand_in_is_deterministic_and_same_shape() {
        let s = scrubber(&[]);
        let tok = "sk-ant-api03-AbCdEf0123456789xyzXYZ";
        let a = s.stand_in(tok);
        let b = s.stand_in(tok);
        assert_eq!(a, b);
        assert_ne!(a, tok);
        assert_eq!(a.len(), tok.len());
        assert!(a.starts_with("sk-ant-"));
        for (x, y) in a.chars().zip(tok.chars()) {
            assert_eq!(x.is_ascii_digit(), y.is_ascii_digit());
            assert_eq!(x.is_ascii_lowercase(), y.is_ascii_lowercase());
            assert_eq!(x.is_ascii_uppercase(), y.is_ascii_uppercase());
        }
    }

    #[test]
    fn different_keys_give_different_stand_ins() {
        let a = SecretScrubber::with_key(Vec::new(), [1u8; 32]);
        let b = SecretScrubber::with_key(Vec::new(), [2u8; 32]);
        let tok = "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8";
        assert_ne!(a.stand_in(tok), b.stand_in(tok));
    }

    #[test]
    fn same_secret_twice_maps_to_same_stand_in() {
        let tok = "lin_api_Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1";
        let s = scrubber(&[tok]);
        let r = s.scrub(&format!("a={tok}\nb={tok}"));
        let lines: Vec<&str> = r.text.lines().collect();
        assert_eq!(lines[0][2..], lines[1][2..]);
        assert!(!r.text.contains(tok));
    }

    #[test]
    fn git_shas_and_digests_are_left_alone() {
        let s = scrubber(&[]);
        let text = "commit 87ec603a8fae6c0d395d39ac0effdc94ebf0e1234\n\
                    sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n\
                    state_token 0123456789abcdef01234567\n\
                    integrity sha512-AbCdEf0123456789AbCdEf0123456789AbCdEf0123456789";
        let r = s.scrub(text);
        assert_eq!(r.text, text);
        assert_eq!(r.total(), 0);
    }

    #[test]
    fn ordinary_prose_and_code_are_left_alone() {
        let s = scrubber(&[]);
        let text = "fn redact_call_result(result: CallResult) -> CallResult {\n\
                    let max_tokens = 4096;\n\
                    GIT_AUTHOR_EMAIL=someone@example.com\n\
                    uuid 65433688-5edc-4a24-aa7a-56379be18aec\n\
                    tokens_out=2951 cache_read=307816 sk-learn secret_key";
        let r = s.scrub(text);
        assert_eq!(r.text, text, "changed: {}", r.text);
    }

    #[test]
    fn credential_named_assignment_value_is_scrubbed() {
        let s = scrubber(&[]);
        for line in [
            "LINEAR_API_KEY=plainlower9case7value",
            "export NOTION_TOKEN='lowercase7only3value'",
            "\"api_key\": \"lowercase7only3value\"",
            "Authorization: Bearer lowercase7only3value",
            "https://api.example.com/v1?access_token=lowercase7only3value&x=1",
        ] {
            let r = s.scrub(line);
            assert!(
                !r.text.contains("lowercase7only3value") && !r.text.contains("plainlower9case7value"),
                "{line} -> {}",
                r.text
            );
        }
    }

    #[test]
    fn source_code_naming_credentials_is_left_alone() {
        let s = scrubber(&[]);
        let text = "let access_token = response.access_token;\n\
                    password: String::new(),\n\
                    pub api_key_env: Option<String>,\n\
                    \"token_count\": 12,\n\
                    static COUNT_TRIGGER_MIN_TOKEN_RATIO_BITS: AtomicU64 = AtomicU64::new(0);\n\
                    api_key_env: \"AI21_API_KEY\".into(),\n\
                    API_TOKEN=/Users/x/.ssh/deploy_key_abc123\n\
                    \"message_token\": 4912661846655238145_u64,\n\
                    id: \"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8\".into(),\n\
                    const PNG: &str = \"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=\";";
        assert_eq!(s.scrub(text).text, text);
    }

    #[test]
    fn variable_name_survives_when_value_is_scrubbed() {
        let s = scrubber(&[]);
        let r = s.scrub("GITHUB_TOKEN=ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8");
        assert!(r.text.starts_with("GITHUB_TOKEN=ghp_"), "{}", r.text);
    }

    #[test]
    fn private_key_block_is_removed() {
        let s = scrubber(&[]);
        let text = "before\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nAAAA/BBBB+cccc\n-----END OPENSSH PRIVATE KEY-----\nafter";
        let r = s.scrub(text);
        assert!(!r.text.contains("b3BlbnNzaC1rZXktdjEAAAAA"));
        assert!(r.text.starts_with("before\n"));
        assert!(r.text.ends_with("\nafter"));
        // A certificate is public and stays.
        let cert = "-----BEGIN CERTIFICATE-----\nMIIBfake\n-----END CERTIFICATE-----";
        assert_eq!(s.scrub(cert).text, cert);
    }

    #[test]
    fn short_known_values_are_ignored() {
        let s = scrubber(&["true", "8080", "short"]);
        assert_eq!(s.known_len(), 0);
    }

    #[test]
    fn longest_known_secret_wins() {
        let inner = "abcdefghijklmnop";
        let outer = "abcdefghijklmnopQRSTUVWX";
        let s = scrubber(&[inner, outer]);
        let r = s.scrub(outer);
        assert!(!r.text.contains(inner));
        assert_eq!(r.known, 1);
    }

    #[test]
    fn secrets_in_arg_extracts_bearer_and_flag_values() {
        let got = secrets_in_arg("Authorization: Bearer sbp_live9f8e7d6c5b4a3f2e1d0c");
        assert!(got.iter().any(|s| s == "sbp_live9f8e7d6c5b4a3f2e1d0c"));
        let got = secrets_in_arg("--access-token=lowercase7only3value");
        assert!(got.iter().any(|s| s == "lowercase7only3value"));
        assert!(secrets_in_arg("--verbose").is_empty());
        assert!(secrets_in_arg("https://mcp.linear.app/sse").is_empty());
    }

    #[test]
    fn env_name_hint() {
        assert!(env_name_is_credential("LINEAR_API_KEY"));
        assert!(env_name_is_credential("notion_token"));
        assert!(!env_name_is_credential("PATH"));
        assert!(!env_name_is_credential("RUST_LOG"));
    }

    #[test]
    fn debug_never_prints_secrets() {
        let tok = "lin_api_Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1";
        let s = scrubber(&[tok]);
        assert!(!format!("{s:?}").contains(tok));
    }

    /// False-positive survey, run by hand:
    /// `SCRUB_CORPUS=<dir> cargo test -p openfang-types secret_scrub::tests::corpus_survey -- --ignored --nocapture`
    ///
    /// Walks `.rs`/`.md`/`.toml` files under the directory and prints each
    /// line the shape pass would change. Point it at a tree with no real
    /// credentials in it: it prints the original line.
    #[test]
    #[ignore]
    fn corpus_survey() {
        let Ok(root) = std::env::var("SCRUB_CORPUS") else {
            return;
        };
        let s = scrubber(&[]);
        let mut stack = vec![std::path::PathBuf::from(root)];
        let (mut files, mut lines, mut hits) = (0usize, 0usize, 0usize);
        while let Some(p) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&p) else {
                continue;
            };
            for e in rd.flatten() {
                let path = e.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if path.is_dir() {
                    if name != "target" && !name.starts_with('.') {
                        stack.push(path);
                    }
                    continue;
                }
                if !(name.ends_with(".rs") || name.ends_with(".md") || name.ends_with(".toml")) {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                files += 1;
                for (i, line) in text.lines().enumerate() {
                    lines += 1;
                    if s.scrub(line).total() > 0 {
                        hits += 1;
                        println!("{}:{}: {}", path.display(), i + 1, line.trim());
                    }
                }
            }
        }
        println!("files={files} lines={lines} changed_lines={hits}");
    }
}
