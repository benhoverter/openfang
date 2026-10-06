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
//! ## Where it runs
//!
//! The kernel owns one [`SecretScrubber`] and installs it with
//! [`install_global`]. Three call sites use it, and all are required:
//!
//! - `bridge_ipc::handle_connection`, for every tool result of a Claude Code
//!   agent (native and upstream MCP);
//! - `agent_loop`, for every tool result on the native driver path: agents on
//!   a non-CLI provider, and any CLI agent whose turn falls back to an API
//!   provider;
//! - `routes::mcp_http`, the MCP-over-HTTP endpoint. Anything calling it is a
//!   model client by construction.
//!
//! Only tool results are scrubbed. Recalled memories injected into the
//! system prompt, inbound `agent_send` text and channel messages reach the
//! model without passing any of these sites, and anything written to memory
//! before this filter existed comes back raw.
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
//! ## Stand-ins are refused on the way back in
//!
//! A stand-in written back to disk silently replaces a live credential with a
//! fake one: read a config file, write the whole thing back, and the next
//! bounce breaks auth with nothing reporting it. So every stand-in this
//! scrubber emits is recorded, and [`SecretScrubber::check_args_value`]
//! refuses any tool call whose arguments contain one. Every call site runs
//! that check
//! before dispatch, for every tool, because content is written by
//! `shell_exec`, MCP writes and `agent_send` as well as the file tools.
//!
//! The refusal applies to read and search calls too: `grep <stand-in>` to
//! find where a value came from is refused like a write. The removed-key
//! marker is refused only once this scrubber has actually removed a private
//! key, so the bare marker text is not blocked on a boot that never did.
//!
//! The record is per boot. A session that spans a daemon bounce can still
//! write an old stand-in; that is accepted. An operator can turn the refusal
//! off with `OPENFANG_SCRUB_ALLOW_STANDIN_WRITES=1` in the daemon environment.
//!
//! ## This file must scrub clean
//!
//! The module's own source passes through the scrubber on every read. Test
//! fixtures and the marker literals are therefore built with `concat!` so no
//! single literal is credential-shaped; `module_source_is_left_alone` pins it.
//! A fixture written as one literal makes this file unreadable and uneditable
//! for the whole fleet once deployed.
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
//!
//! ## Threat model, and accepted leaks
//!
//! **This filters accidents, not a hostile agent.** `rev`, `base64`, `xxd`,
//! or printing a value in pieces defeats it trivially. Do not build on it as
//! a security control.
//!
//! Two leaks are deliberate and accepted:
//!
//! - A stand-in reveals the secret's length, prefix and the character class
//!   at each position. Irrelevant for a random 40-character token; for a human
//!   password caught by a `PASSWORD` hint it hands a cracker the mask.
//! - Within one boot, the mapping is deterministic, so an agent holding a
//!   candidate value can confirm a guess by getting it scrubbed. That only
//!   matters for low-entropy secrets.
//!
//! Known gaps, accepted under the accidents-only model above:
//!
//! - A private key deliberately re-wrapped into lines shorter than 40
//!   characters (`fold`, `base64 -w32`) with no END marker, read so that it
//!   runs past 8 KiB of output, leaks the lines past 8 KiB. No standard tool
//!   wraps that short (openssl 64, ssh-keygen 70, `base64` 76).
//! - If a tool appends a notice after a truncated key, the short final key
//!   line before it is not followed: at most 39 characters, far below what
//!   factoring needs.
//! - An unrelated END marker within 8 KiB of a truncated key closes it early.
//! - PuTTY `.ppk` keys have no PEM header and are never caught by this pass.
//! - Forced colour output (`--color=always`, `bat -f`, `delta`) leaves an ANSI
//!   reset such as `\x1b[0m` at line end, which stops the follow at 8 KiB.
//!   `shell_exec` is not a terminal, so tools do not colour by default.
//! - Key lines stored as quoted array items (`"MIIE…",` in YAML/JSON) are not
//!   followed past 8 KiB. Trimming `",` would open the follow to every array
//!   of long strings.
//!
//! Accepted over-redaction (fails closed, nothing leaks): a stray PEM header
//! with no END, followed 8 KiB later by consecutive lines that each end in
//! 40+ base64 characters (go.sum, comment-less `known_hosts`, `git rev-list`,
//! dot-free path listings), is followed up to the 64 KiB hard stop. Do not
//! "fix" this by tightening the follow: that reopens the SS16/SS19 leak.

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

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
    "sb_secret_",
    "sb_publishable_",
    "lin_api_",
    "lin_oauth_",
    "lin_wh_",
    "ntn_",
    "whsec_",
    "secret_",
    "hf_",
    "npm_",
    "re_",
    "AIza",
    "AKIA",
    "ASIA",
    "eyJ",
];

/// Prefixes short or common enough to appear in identifiers
/// (`re_export_handles_v2_runtime`). For these the body must be one unbroken
/// segment, as real keys are.
const SINGLE_SEGMENT_PREFIXES: &[&str] = &["re_"];

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

/// What a removed private-key block is replaced with. Also refused on the way
/// back in: writing it over a key file destroys the key.
pub const PEM_REMOVED_MARKER: &str = concat!("[openfang: private", " key removed]");

/// The block a removed private key is replaced with, split so this source
/// file never holds a contiguous private-key header.
const PEM_BLOCK_OPEN: &str = concat!("-----BEGIN ", "PRIVATE", " KEY-----\n");
const PEM_BLOCK_CLOSE: &str = concat!("\n-----END ", "PRIVATE", " KEY-----");

/// Minimum redaction past a private-key header with no END marker. The bound
/// stops one `grep -r` hit from wiping the rest of the output. It counts
/// output bytes, not key bytes, so on its own a per-line prefix (`rg -n`,
/// `cat -n`) or a large key can push key lines past it: see
/// [`KEY_LINE_MIN_TOKEN`] for how the redaction continues.
const MAX_UNTERMINATED_PEM: usize = 8 * 1024;

/// Hard stop for an unterminated private-key redaction, however key-like the
/// lines keep looking. A 16384-bit RSA PEM is about 12.7 KB raw.
const MAX_UNTERMINATED_PEM_HARD: usize = 64 * 1024;

/// Past [`MAX_UNTERMINATED_PEM`], redaction continues line by line while a
/// line ends in at least this many consecutive base64 characters. That follows prefixed key lines without following
/// ordinary source. Not "stop at the first non-base64 line": the prefix would
/// make every key line fail that test and leak the tail, which holds the
/// factors.
const KEY_LINE_MIN_TOKEN: usize = 40;

/// Daemon env var that turns off [`SecretScrubber::check_args`].
pub const ALLOW_STANDIN_WRITES_ENV: &str = "OPENFANG_SCRUB_ALLOW_STANDIN_WRITES";

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
/// No bare `AUTH`: it matches `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL` and
/// `SSH_AUTH_SOCK`, and would scramble an author's name in every result.
/// `*_AUTH_TOKEN` is already covered by `TOKEN`.
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
    "AUTHORIZATION",
];

static GLOBAL: OnceLock<Arc<SecretScrubber>> = OnceLock::new();

/// Install the process-wide scrubber. The kernel calls this once at boot so
/// the bridge and the native agent loop share one key and one stand-in
/// record. Returns false if one was already installed (the first one stays);
/// callers should log that, because a second kernel then scrubs with the
/// first one's key and record.
pub fn install_global(scrubber: Arc<SecretScrubber>) -> bool {
    GLOBAL.set(scrubber).is_ok()
}

/// The process-wide scrubber, if the kernel has installed one. `None` in unit
/// tests and tools that never boot a kernel.
#[must_use]
pub fn global() -> Option<&'static Arc<SecretScrubber>> {
    GLOBAL.get()
}

/// True when an environment variable name reads like it holds a credential.
#[must_use]
pub fn env_name_is_credential(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    CREDENTIAL_ENV_HINTS.iter().any(|h| upper.contains(h))
}

/// A config-derived value that is not itself a credential: an unexpanded
/// placeholder (`${MERMAID_CHART_TOKEN}`) or a filesystem path. Exact-matching
/// either scrambles ordinary text, and a placeholder scrambled in a config
/// file the agent rewrites breaks auth at the next bounce.
fn is_placeholder_or_path(value: &str) -> bool {
    value.contains("${") || value.starts_with(['$', '/', '~'])
}

/// Pull candidate secrets out of one command-line style string, e.g. an MCP
/// server's `args` entry or an `Authorization: Bearer ...` header value.
///
/// Returns the parts worth exact-matching: anything after `Bearer `, the value
/// of a `--flag=value` whose flag reads like a credential, and any part whose
/// shape is already credential-like. Placeholders and paths are dropped.
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
    out.retain(|s| s.len() >= MIN_KNOWN_SECRET_LEN && !is_placeholder_or_path(s));
    out
}

/// Every credential value the daemon can see for a set of MCP servers, plus
/// credential-named daemon env vars, for exact-match scrubbing.
///
/// Pass the kernel's *effective* server list (configured plus
/// extension-installed), not `config.mcp_servers` alone. Sources per server:
/// passed-through env vars, headers, and stdio `args` (the motivating
/// incident was a bearer token in argv, printed by `pgrep -lf`). Placeholders
/// and paths are dropped; short values are dropped by [`SecretScrubber`].
#[must_use]
pub fn known_secrets_from(servers: &[crate::config::McpServerConfigEntry]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for server in servers {
        for name in &server.env {
            // Entries are documented as names; tolerate `NAME=value` too.
            match name.split_once('=') {
                Some((_, v)) => out.push(v.to_string()),
                None => {
                    if let Ok(v) = std::env::var(name) {
                        out.push(v);
                    }
                }
            }
        }
        for header in &server.headers {
            out.extend(secrets_in_arg(header));
        }
        if let crate::config::McpTransportEntry::Stdio { args, .. } = &server.transport {
            for arg in args {
                out.extend(secrets_in_arg(arg));
            }
        }
    }
    for (name, value) in std::env::vars() {
        if env_name_is_credential(&name) {
            out.push(value);
        }
    }
    out.retain(|v| !is_placeholder_or_path(v.trim()));
    out
}

fn key_is_credential(key: &str) -> bool {
    let k = key.trim_start_matches('-').to_ascii_lowercase();
    CREDENTIAL_KEY_HINTS.iter().any(|h| k.contains(h))
}

fn normalize_known(known: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut known: Vec<String> = known
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() >= MIN_KNOWN_SECRET_LEN)
        .collect();
    known.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    known.dedup();
    known
}

/// Replaces credentials in text with same-shape stand-ins. See the module docs.
pub struct SecretScrubber {
    /// Exact values, longest first so a secret that contains another is
    /// replaced whole. Behind a lock so the kernel can refresh it after MCP
    /// connect, which is when vault-only credentials reach the env.
    known: RwLock<Vec<String>>,
    /// Every stand-in emitted so far, exactly as it appeared in the output.
    /// An `RwLock` so refusal checks and the shape passes' lookups run in
    /// parallel; only emitting a new stand-in takes the write lock. Not
    /// capped by eviction: dropping an entry would let its write-back through.
    issued: RwLock<Issued>,
    /// Set once a private-key block has been removed. Until then the
    /// removed-key marker is ordinary text and is not refused.
    pem_redacted: AtomicBool,
    key: [u8; 32],
    allow_standin_writes: bool,
}

impl std::fmt::Debug for SecretScrubber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the known values or the key.
        f.debug_struct("SecretScrubber")
            .field("known_secrets", &self.known_len())
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
    /// Bytes removed after private-key headers that had no END marker.
    pub pem_cut_bytes: usize,
}

impl Scrubbed {
    #[must_use]
    pub fn total(&self) -> usize {
        self.known + self.shaped
    }
}

impl SecretScrubber {
    /// Build a scrubber over the given known values with a fresh random key.
    /// Values shorter than [`MIN_KNOWN_SECRET_LEN`] are dropped. Reads
    /// [`ALLOW_STANDIN_WRITES_ENV`] once.
    #[must_use]
    pub fn new(known: impl IntoIterator<Item = String>) -> Self {
        let allow = std::env::var(ALLOW_STANDIN_WRITES_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
            .unwrap_or(false);
        let mut s = Self::with_key(known, rand::random());
        s.allow_standin_writes = allow;
        s
    }

    /// Build with an explicit key. Tests use this for reproducible stand-ins.
    /// The stand-in refusal is always on.
    #[must_use]
    pub fn with_key(known: impl IntoIterator<Item = String>, key: [u8; 32]) -> Self {
        Self {
            known: RwLock::new(normalize_known(known)),
            issued: RwLock::new(Issued::default()),
            pem_redacted: AtomicBool::new(false),
            key,
            allow_standin_writes: false,
        }
    }

    /// Replace the known set. Stand-ins already issued stay recorded.
    /// Returns the new count.
    pub fn replace_known(&self, known: impl IntoIterator<Item = String>) -> usize {
        let v = normalize_known(known);
        let n = v.len();
        *self.known.write().unwrap_or_else(|e| e.into_inner()) = v;
        n
    }

    /// How many exact values this scrubber watches for.
    #[must_use]
    pub fn known_len(&self) -> usize {
        self.known.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// How many of the known values occur, verbatim, anywhere in `bytes`.
    /// Exact-match pass only: no shape pass, and no stand-ins are issued, so
    /// this is safe to run over binary data (an outbound file, say). It
    /// catches accidents, not a determined writer: base64, compression or any
    /// other re-encoding of a secret is not seen. Never reveals which value
    /// matched.
    #[must_use]
    pub fn known_hits_in_bytes(&self, bytes: &[u8]) -> usize {
        // Snapshot, then drop the guard before the scan: the scan is
        // known.len() x file size, and holding the read lock that long blocks
        // `replace_known`. The clone holds live secret values; it stays in
        // this frame and is never logged or stored.
        let set: Vec<String> = {
            let guard = self.known.read().unwrap_or_else(|e| e.into_inner());
            guard.clone()
        };
        set.iter()
            .filter(|s| {
                let needle = s.as_bytes();
                !needle.is_empty()
                    && needle.len() <= bytes.len()
                    && bytes.windows(needle.len()).any(|w| w == needle)
            })
            .count()
    }

    /// How many distinct stand-ins have been issued since this scrubber was
    /// built. Grows for the life of the boot (nothing is evicted); callers
    /// may log it to watch that growth.
    #[must_use]
    pub fn issued_len(&self) -> usize {
        self.issued.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether [`Self::check_args`] is switched off by the operator.
    #[must_use]
    pub fn allows_standin_writes(&self) -> bool {
        self.allow_standin_writes
    }

    /// Scrub `text`.
    #[must_use]
    pub fn scrub(&self, text: &str) -> Scrubbed {
        let mut out = text.to_string();
        let mut known = 0usize;
        {
            let set = self.known.read().unwrap_or_else(|e| e.into_inner());
            for secret in set.iter() {
                let n = out.matches(secret.as_str()).count();
                if n > 0 {
                    known += n;
                    let fake = self.issue(secret);
                    out = out.replace(secret.as_str(), &fake);
                }
            }
        }
        let (out, pem, pem_cut_bytes) = self.scrub_pem(&out);
        if pem > 0 {
            self.pem_redacted.store(true, Ordering::Relaxed);
        }
        let (out, assigned) = self.scrub_assignments(&out);
        let (out, runs) = self.scrub_runs(&out);
        Scrubbed {
            text: out,
            known,
            shaped: pem + assigned + runs,
            pem_cut_bytes,
        }
    }

    /// Scrub a tool result for the model. Returns the text unchanged when
    /// nothing was found; otherwise the scrubbed text with a one-line notice
    /// **first**, so a later truncation from the end can never drop it.
    #[must_use]
    pub fn scrub_for_model(&self, text: String) -> (String, usize) {
        let s = self.scrub(&text);
        let n = s.total();
        if n == 0 {
            return (text, 0);
        }
        let cut = if s.pem_cut_bytes > 0 {
            format!(
                " A private-key header with no END marker was found; {} byte(s) \
                 after it were removed.",
                s.pem_cut_bytes
            )
        } else {
            String::new()
        };
        (
            format!(
                "[openfang: {n} credential-shaped value(s) in this result were replaced \
                 with stand-ins of the same shape. They are not usable credentials, and \
                 a tool call that contains one is refused.{cut}]\n\n{}",
                s.text
            ),
            n,
        )
    }

    /// Scrub a tool result's content for the model. `tool_use_id` and
    /// `is_error` are never changed. Returns the result and the count.
    #[must_use]
    pub fn scrub_tool_result(
        &self,
        result: crate::tool::ToolResult,
    ) -> (crate::tool::ToolResult, usize) {
        let crate::tool::ToolResult {
            tool_use_id,
            content,
            is_error,
        } = result;
        let (content, n) = self.scrub_for_model(content);
        (
            crate::tool::ToolResult {
                tool_use_id,
                content,
                is_error,
            },
            n,
        )
    }

    /// The first issued stand-in found in `text`, or the removed-key marker
    /// if a private key has been removed this boot.
    #[must_use]
    pub fn find_issued(&self, text: &str) -> Option<String> {
        let issued = self.issued.read().unwrap_or_else(|e| e.into_inner());
        self.find_issued_in(&issued, text)
    }

    fn find_issued_in(&self, issued: &Issued, text: &str) -> Option<String> {
        if self.pem_redacted.load(Ordering::Relaxed) && text.contains(PEM_REMOVED_MARKER) {
            return Some(PEM_REMOVED_MARKER.to_string());
        }
        issued.find_in(text).map(str::to_string)
    }

    /// Refuse raw text containing a stand-in this scrubber issued. Prefer
    /// [`Self::check_args_value`] for tool arguments: in serialized JSON a
    /// stand-in containing `"` or `\` appears escaped and is missed here.
    pub fn check_args(&self, args_text: &str) -> Result<(), String> {
        if self.allow_standin_writes {
            return Ok(());
        }
        self.find_issued(args_text)
            .map_or(Ok(()), |s| Err(refusal(&s)))
    }

    /// Refuse a tool call whose arguments contain a stand-in this scrubber
    /// issued. Tests each unescaped string value and object key, so a
    /// stand-in containing `"` or `\` is still found. `Err` carries the
    /// message to return to the model.
    pub fn check_args_value(&self, args: &serde_json::Value) -> Result<(), String> {
        if self.allow_standin_writes {
            return Ok(());
        }
        let mut leaves: Vec<&str> = Vec::new();
        collect_strings(args, &mut leaves);
        let issued = self.issued.read().unwrap_or_else(|e| e.into_inner());
        leaves
            .into_iter()
            .find_map(|leaf| self.find_issued_in(&issued, leaf))
            .map_or(Ok(()), |s| Err(refusal(&s)))
    }

    /// Same-shape stand-in: keep the credential prefix, then replace each
    /// character with one of the same class drawn from a keyed digest stream.
    /// Punctuation inside the body is kept so the shape survives.
    ///
    /// Pure: does not record the stand-in. Scrubbing records what it emits.
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
            // Length-prefixed so (secret, counter) encodings cannot collide.
            h.update((secret.len() as u64).to_le_bytes());
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

    /// Emit a stand-in and record it for [`Self::check_args`].
    fn issue(&self, secret: &str) -> String {
        let fake = self.stand_in(secret);
        let mut issued = self.issued.write().unwrap_or_else(|e| e.into_inner());
        // No cap: an unrecorded stand-in is one whose write-back is not
        // refused, which reopens the hazard the refusal exists for.
        issued.insert(&fake);
        fake
    }

    /// True for text this scrubber already emitted as a stand-in. The shape
    /// passes skip those, so a known-pass stand-in is not replaced a second
    /// time (which would double-count, and leave the record pointing at text
    /// that never reached the model).
    fn is_issued(&self, s: &str) -> bool {
        self.issued
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(s)
    }

    /// Replace PEM private-key blocks.
    ///
    /// The header ends at the first of: its closing `-----`, a real newline,
    /// or a literal `\n` (single-line JSON). An END marker only closes the
    /// block if it falls inside the window [`pem_window`] defines; a private
    /// key with no END there (a truncated read) is redacted to the window's
    /// end. Returns the text, the block count, and the bytes cut that way.
    fn scrub_pem(&self, text: &str) -> (String, usize, usize) {
        const BEGIN: &str = "-----BEGIN ";
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        let mut count = 0usize;
        let mut cut_bytes = 0usize;
        while let Some(start) = rest.find(BEGIN) {
            let after = start + BEGIN.len();
            let header_end = pem_header_end(&rest[after..]).map(|i| after + i);
            let is_private = |e: usize| rest[start..e].contains(concat!("PRIVATE", " KEY"));
            let Some(header_end) = header_end.filter(|&e| is_private(e)) else {
                out.push_str(&rest[..after]);
                rest = &rest[after..];
                continue;
            };
            out.push_str(&rest[..start]);
            out.push_str(PEM_BLOCK_OPEN);
            out.push_str(PEM_REMOVED_MARKER);
            out.push_str(PEM_BLOCK_CLOSE);
            count += 1;
            rest = match pem_window(rest, header_end) {
                PemStop::End(resume) => &rest[resume..],
                PemStop::Cut(stop) => {
                    cut_bytes += stop - header_end;
                    out.push_str(&format!(
                        "\n[openfang: no END marker; {} byte(s) after this header were removed]\n",
                        stop - header_end
                    ));
                    &rest[stop..]
                }
            };
        }
        out.push_str(rest);
        (out, count, cut_bytes)
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
            if !key.is_empty()
                && key_is_credential(key)
                && assigned_value_is_credential(value)
                && !self.is_issued(value)
            {
                out.push_str(&line[cursor..v_start]);
                out.push_str(&self.issue(value));
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
        if run_is_secret_shaped(run) && !self.is_issued(run) {
            out.push_str(&self.issue(run));
            *count += 1;
        } else {
            out.push_str(run);
        }
    }
}

/// Bytes of a stand-in used as its lookup key. Every issued stand-in is at
/// least [`MIN_KNOWN_SECRET_LEN`] / [`MIN_ASSIGNED_VALUE_LEN`] bytes, so in
/// practice all of them are indexed; shorter ones fall back to a scan.
const ISSUED_KEY_LEN: usize = 8;

/// The stand-ins issued this boot, indexed so a refusal check costs time in
/// the length of the text checked rather than in the number issued (SS15).
///
/// `find_in` has exactly the semantics of "does `text` contain any issued
/// stand-in as a substring" — it is not a token lookup, so a stand-in pasted
/// inside a longer word is still found.
#[derive(Default)]
struct Issued {
    /// Exact membership, for [`SecretScrubber::is_issued`].
    set: HashSet<String>,
    /// First [`ISSUED_KEY_LEN`] bytes → the stand-ins starting with them.
    by_prefix: HashMap<[u8; ISSUED_KEY_LEN], Vec<Box<str>>>,
    /// Stand-ins shorter than the key. Not expected; scanned linearly.
    short: Vec<Box<str>>,
}

impl Issued {
    fn len(&self) -> usize {
        self.set.len()
    }

    fn contains(&self, s: &str) -> bool {
        self.set.contains(s)
    }

    fn insert(&mut self, s: &str) {
        if !self.set.insert(s.to_string()) {
            return;
        }
        match prefix_key(s.as_bytes()) {
            Some(k) => self.by_prefix.entry(k).or_default().push(s.into()),
            None => self.short.push(s.into()),
        }
    }

    /// The issued stand-in occurring earliest in `text`, if any.
    fn find_in(&self, text: &str) -> Option<&str> {
        let bytes = text.as_bytes();
        if !self.by_prefix.is_empty() && bytes.len() >= ISSUED_KEY_LEN {
            for i in 0..=bytes.len() - ISSUED_KEY_LEN {
                let Some(cands) = prefix_key(&bytes[i..]).and_then(|k| self.by_prefix.get(&k))
                else {
                    continue;
                };
                if let Some(hit) = cands.iter().find(|c| bytes[i..].starts_with(c.as_bytes())) {
                    return Some(hit);
                }
            }
        }
        self.short
            .iter()
            .find(|s| text.contains(&***s))
            .map(|s| &**s)
    }
}

fn prefix_key(b: &[u8]) -> Option<[u8; ISSUED_KEY_LEN]> {
    b.get(..ISSUED_KEY_LEN)?.try_into().ok()
}

/// Offset just past the end of a PEM header, measured from the byte after
/// `-----BEGIN `. Capped so an unterminated `-----BEGIN ` in prose does not
/// reach across the document.
fn pem_header_end(s: &str) -> Option<usize> {
    const MAX_HEADER: usize = 96;
    [s.find("-----").map(|i| i + 5), s.find('\n'), s.find("\\n")]
        .into_iter()
        .flatten()
        .min()
        .filter(|&i| i <= MAX_HEADER)
}

/// Where the redaction of a private key that starts at `header_end` stops.
#[derive(Debug, PartialEq, Eq)]
enum PemStop {
    /// An END marker inside the window; resume at this offset, just past its
    /// closing `-----` (or the end of its line).
    End(usize),
    /// No END marker inside the window; redact up to this offset.
    Cut(usize),
}

/// Walk the lines after a private-key header (split on real newlines and on
/// literal `\n`) and decide where the redaction stops:
/// - every line inside the first [`MAX_UNTERMINATED_PEM`] bytes is redacted;
/// - past that, lines are redacted while [`is_key_line`] holds;
/// - nothing past [`MAX_UNTERMINATED_PEM_HARD`] is redacted;
/// - an END marker closes the block only inside that window (SS17), so a
///   `-----END CERTIFICATE-----` from another file further down does not
///   make the redaction unbounded again.
fn pem_window(rest: &str, header_end: usize) -> PemStop {
    const END: &str = "-----END ";
    let soft = floor_boundary(rest, header_end + MAX_UNTERMINATED_PEM);
    let hard = floor_boundary(rest, header_end + MAX_UNTERMINATED_PEM_HARD);
    // Next literal `\n` at or after `ls`, rescanned only once passed, so the
    // walk stays linear in the window.
    let find_lit = |from: usize| rest[from..hard].find("\\n").map(|i| from + i);
    let mut lit = find_lit(header_end);
    let mut ls = header_end;
    while ls < hard {
        if lit.is_some_and(|p| p < ls) {
            lit = find_lit(ls);
        }
        let nl = rest[ls..hard].find('\n').map(|i| (ls + i, ls + i + 1));
        let (le, next) = match (nl, lit.map(|p| (p, p + 2))) {
            (Some(a), Some(b)) => a.min(b),
            (a, b) => a.or(b).unwrap_or((hard, hard)),
        };
        let line = &rest[ls..le];
        if let Some(i) = line.find(END) {
            let at = ls + i;
            if at > soft && ls < soft {
                // An END far along a line that started inside the minimum
                // window and is not a key line: not this key's END.
                return PemStop::Cut(soft);
            }
            let e = at + END.len();
            return PemStop::End(rest[e..le].find("-----").map_or(le, |t| e + t + 5));
        }
        if le <= soft {
            ls = next;
            continue;
        }
        if le == hard && hard < rest.len() {
            return PemStop::Cut(hard);
        }
        if is_key_line(line, next >= rest.len()) {
            ls = next;
            continue;
        }
        return PemStop::Cut(if ls < soft { soft } else { ls });
    }
    PemStop::Cut(hard)
}

/// True for a line that still looks like private-key body: it ENDS in a run
/// of base64 at least [`KEY_LINE_MIN_TOKEN`] long. Whatever precedes the run
/// is ignored, so no list of prefix separators is needed: `path:N:` (`rg -n`),
/// `path-N-` (grep/rg `-A` context lines, SS19), `cat -n` tabs all pass.
/// Base64 never contains `-`, `.` or `:`, so the run starts after the prefix.
/// Line-end decoration is trimmed first: a real CR, a literal `\r` (single-
/// line JSON of a CRLF key), `cat -A`/`cat -e`'s `$` and `^M`, and trailing
/// spaces/tabs (editor pastes, `diff -y`/`pr`/`column` padding). The final
/// line of the whole text may be shorter: that is where a truncated read
/// ends.
fn is_key_line(line: &str, last_in_text: bool) -> bool {
    let mut line = line;
    loop {
        let t = line
            .trim_end_matches([' ', '\t'])
            .trim_end_matches('\r')
            .trim_end_matches('$')
            .trim_end_matches("^M")
            .trim_end_matches("\\r");
        if t.len() == line.len() {
            break;
        }
        line = t;
    }
    let run = line
        .bytes()
        .rev()
        .take_while(|&b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        .count();
    run > 0 && (run >= KEY_LINE_MIN_TOKEN || last_in_text)
}

/// `min(i, s.len())`, moved back to a char boundary.
fn floor_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn is_key_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

/// The message a refused tool call returns to the model.
fn refusal(standin: &str) -> String {
    format!(
        "Refused before running: the arguments contain `{standin}`, which is a stand-in \
         OpenFang showed you in place of a real credential. Writing or sending it \
         would replace the real value with a fake one. Nothing was run. Leave that \
         value out: edit only the lines that do not contain it (apply_patch on \
         those lines), or refer to the credential by its environment variable \
         name. Operator override: {ALLOW_STANDIN_WRITES_ENV}=1 in the daemon \
         environment."
    )
}

/// Every string value and object key in `v`, unescaped.
fn collect_strings<'a>(v: &'a serde_json::Value, out: &mut Vec<&'a str>) {
    match v {
        serde_json::Value::String(s) => out.push(s),
        serde_json::Value::Array(items) => items.iter().for_each(|i| collect_strings(i, out)),
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                out.push(k);
                collect_strings(val, out);
            }
        }
        _ => {}
    }
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
        if SINGLE_SEGMENT_PREFIXES.contains(p) && body.contains(['_', '-']) {
            return false;
        }
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

    // Fixtures are split with `concat!` so this file scrubs clean (see the
    // module docs and `module_source_is_left_alone`). Keep every piece under
    // 32 characters and never put a prefix and 16+ body characters together.
    const GHP: &str = concat!("ghp_", "A1b2C3d4E5f6G7h8I9", "j0K1l2M3n4O5p6Q7r8");
    const SBP: &str = concat!("sbp_", "live9f8e7d6c5b4a3f", "2e1d0c9b8a7f6e5d4c3b2a1");
    const SBP_SHORT: &str = concat!("sbp_", "live9f8e7d6c", "5b4a3f2e1d0c");
    const SK_ANT: &str = concat!("sk-ant-", "api03-AbCdEf", "0123456789xyzXYZ");
    const LIN: &str = concat!("lin_api_", "Zz9Yy8Xx7Ww6", "Vv5Uu4Tt3Ss2Rr1");
    const LOWER: &str = concat!("lowercase7", "only3value");
    const PLAIN: &str = concat!("plainlower9", "case7value");

    fn scrubber(known: &[&str]) -> SecretScrubber {
        SecretScrubber::with_key(known.iter().map(|s| (*s).to_string()), KEY)
    }

    #[test]
    fn known_hits_in_bytes_finds_exact_values_in_binary() {
        let s = scrubber(&[SBP]);
        let mut pdf = b"%PDF-1.7\n\x00\xff".to_vec();
        pdf.extend_from_slice(format!("token = \"{SBP}\"\n").as_bytes());
        assert_eq!(s.known_hits_in_bytes(&pdf), 1);
        assert_eq!(s.known_hits_in_bytes(b"\x89PNG\r\n\x1a\n\x00\x00"), 0);
        assert_eq!(s.known_hits_in_bytes(b""), 0);
        assert_eq!(s.issued_len(), 0, "a byte scan must not issue stand-ins");
    }

    /// The incident: `pgrep -lf` printing an MCP server's argv.
    #[test]
    fn pgrep_bearer_line_is_scrubbed_by_known_value() {
        let token = SBP;
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
        let tok = GHP;
        let r = s.scrub(&format!("token is {tok} ok"));
        assert!(!r.text.contains(tok));
        assert!(r.text.starts_with("token is ghp_"));
        assert!(r.text.ends_with(" ok"));
        assert_eq!(r.shaped, 1);
    }

    #[test]
    fn stand_in_is_deterministic_and_same_shape() {
        let s = scrubber(&[]);
        let tok = SK_ANT;
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
        let tok = GHP;
        assert_ne!(a.stand_in(tok), b.stand_in(tok));
    }

    #[test]
    fn same_secret_twice_maps_to_same_stand_in() {
        let tok = LIN;
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
            format!("LINEAR_API_KEY={PLAIN}"),
            format!("export NOTION_TOKEN='{LOWER}'"),
            format!("\"api_key\": \"{LOWER}\""),
            format!("Authorization: Bearer {LOWER}"),
            format!("https://api.example.com/v1?access_token={LOWER}&x=1"),
        ] {
            let r = s.scrub(&line);
            assert!(
                !r.text.contains(LOWER) && !r.text.contains(PLAIN),
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
        let r = s.scrub(&format!("GITHUB_TOKEN={GHP}"));
        assert!(r.text.starts_with("GITHUB_TOKEN=ghp_"), "{}", r.text);
    }

    #[test]
    fn private_key_block_is_removed() {
        let s = scrubber(&[]);
        let text = concat!(
            "before\n-----BEGIN OPENSSH ",
            "PRIVATE",
            " KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nAAAA/BBBB+cccc\n-----END OPENSSH ",
            "PRIVATE",
            " KEY-----\nafter",
        );
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
        let got = secrets_in_arg(&format!("Authorization: Bearer {SBP_SHORT}"));
        assert!(got.iter().any(|s| s == SBP_SHORT));
        let got = secrets_in_arg(&format!("--access-token={LOWER}"));
        assert!(got.iter().any(|s| s == LOWER));
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

    /// SS14: author identity and the ssh-agent socket are not credentials.
    #[test]
    fn env_name_hint_skips_author_and_auth_sock() {
        for name in ["GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "SSH_AUTH_SOCK"] {
            assert!(!env_name_is_credential(name), "{name}");
        }
        assert!(env_name_is_credential("HTTP_AUTHORIZATION"));
        assert!(env_name_is_credential("GH_AUTH_TOKEN"));
    }

    #[test]
    fn debug_never_prints_secrets() {
        let tok = LIN;
        let s = scrubber(&[tok]);
        assert!(!format!("{s:?}").contains(tok));
    }

    /// SS2: a config placeholder is not a secret and must not be scrambled.
    #[test]
    fn placeholders_and_paths_are_not_known_secrets() {
        assert!(secrets_in_arg("Authorization:${MERMAID_CHART_TOKEN}").is_empty());
        assert!(secrets_in_arg("Authorization: Bearer ${LINEAR_TOKEN}").is_empty());
        assert!(secrets_in_arg("--api-token=$NOTION_TOKEN_VALUE").is_empty());
        assert!(secrets_in_arg("--auth-token=/Users/x/.config/token.json").is_empty());
        // A real value next to the placeholder shape still counts.
        let got = secrets_in_arg(&format!("Authorization:Bearer {SBP_SHORT}"));
        assert!(got.iter().any(|s| s == SBP_SHORT));
    }

    /// SS4: a truncated private key (no END marker) is redacted.
    #[test]
    fn truncated_private_key_is_removed_to_end() {
        let s = scrubber(&[]);
        let text = concat!(
            "head\n-----BEGIN RSA ",
            "PRIVATE",
            " KEY-----\nMIIEowIBAAKCAQEA/x+y/z\nAAAA/BBBB+cc",
        );
        let r = s.scrub(text);
        assert!(!r.text.contains("MIIEowIBAAKCAQEA"), "{}", r.text);
        assert!(!r.text.contains("AAAA/BBBB"), "{}", r.text);
        assert!(r.text.starts_with("head\n"));
        assert!(r.text.contains(PEM_REMOVED_MARKER));
        assert!(r.pem_cut_bytes > 0);
    }

    /// SS12: the unterminated redaction stops 8 KiB past the header, so one
    /// `grep -r` hit does not wipe every match after it.
    #[test]
    fn unterminated_private_key_redaction_is_bounded() {
        let s = scrubber(&[]);
        let header = concat!("-----BEGIN RSA ", "PRIVATE", " KEY-----");
        let body = "QUJD\n".repeat(3000);
        let text = format!("a.pem:1:{header}\n{body}other.rs:9: fn still_here() {{}}\n");
        let r = s.scrub(&text);
        assert_eq!(r.pem_cut_bytes, MAX_UNTERMINATED_PEM);
        assert!(r.text.contains("fn still_here()"), "tail lost");
        assert!(r.text.starts_with("a.pem:1:"));
        let (notice, _) = s.scrub_for_model(text);
        assert!(notice.contains("8192 byte(s)"), "{}", &notice[..300]);
    }

    /// Deterministic base64 body lines of PEM width, `/` and `+` included.
    fn fake_key_lines(n: usize) -> Vec<String> {
        let alphabet: Vec<u8> = concat!(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789+/",
        )
        .bytes()
        .collect();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        (0..n)
            .map(|_| {
                (0..64)
                    .map(|_| {
                        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        char::from(alphabet[(x >> 58) as usize])
                    })
                    .collect()
            })
            .collect()
    }

    /// SS16: an 8192-bit key read through `rg -n` with a long path prefix
    /// runs well past 8 KiB of output. None of it may survive, the tail
    /// least of all (that is where the factors are).
    #[test]
    fn prefixed_large_truncated_key_does_not_leak_past_minimum() {
        let s = scrubber(&[]);
        let header = concat!("-----BEGIN RSA ", "PRIVATE", " KEY-----");
        let path = "/Users/someone/.openfang/workspaces/some-agent/projects/some-project/keys/deploy-key-2026.pem";
        let lines = fake_key_lines(100); // ~6.4 KB of base64: 8192-bit size
        let mut text = format!("{path}:1:{header}\n");
        for (i, l) in lines.iter().enumerate() {
            text.push_str(&format!("{path}:{}:{l}\n", i + 2));
        }
        // Under the old fixed 8 KiB cap, roughly the last half leaked.
        assert!(
            text.len() > MAX_UNTERMINATED_PEM * 3 / 2,
            "fixture too small"
        );
        let r = s.scrub(&text);
        for l in &lines {
            assert!(!r.text.contains(&l[..20]), "key line survived: {l}");
        }
        assert!(r.pem_cut_bytes > MAX_UNTERMINATED_PEM);
        // Same body behind `cat -n` (tab after the number).
        let mut tabbed = format!("     1\t{header}\n");
        for (i, l) in lines.iter().enumerate() {
            tabbed.push_str(&format!("{:>6}\t{l}\n", i + 2));
        }
        let r = s.scrub(&tabbed);
        for l in &lines {
            assert!(!r.text.contains(&l[..20]), "cat -n line survived: {l}");
        }
    }

    /// SS19: `grep -A`/`rg -A` context lines use `-` as the separator
    /// (`path-12-MIIE…`, or `path-MIIE…` without `-n`). The tail must not
    /// survive.
    #[test]
    fn grep_context_prefixed_key_does_not_leak() {
        let s = scrubber(&[]);
        let header = concat!("-----BEGIN RSA ", "PRIVATE", " KEY-----");
        let path = "/Users/someone/.openfang/workspaces/some-agent/projects/some-project/keys/deploy-key-2026.pem";
        let lines = fake_key_lines(100);
        let mut numbered = format!("{path}:1:{header}\n");
        let mut bare = format!("{path}:{header}\n");
        for (i, l) in lines.iter().enumerate() {
            numbered.push_str(&format!("{path}-{}-{l}\n", i + 2));
            bare.push_str(&format!("{path}-{l}\n"));
        }
        assert!(numbered.len() > MAX_UNTERMINATED_PEM * 3 / 2);
        for text in [&numbered, &bare] {
            let r = s.scrub(text);
            for l in &lines {
                assert!(!r.text.contains(&l[..20]), "grep -A line survived: {l}");
            }
        }
    }

    /// SS19: line-end decoration does not stop the follow: literal `\r`
    /// from single-line JSON of a CRLF key, and `cat -A`'s `^M$`.
    #[test]
    fn key_line_trims_line_end_decoration() {
        let k = concat!("MIIEowIBAAKCAQEA/abc+", "defGHIJKLMNOPQRSTUVWXYZ012345");
        assert!(is_key_line(k, false));
        assert!(is_key_line(&format!("p-12-{k}"), false));
        assert!(is_key_line(&format!("{k}\\r"), false));
        assert!(is_key_line(&format!("{k}$"), false));
        assert!(is_key_line(&format!("{k}^M$"), false));
        assert!(is_key_line(&format!("{k}\r"), false));
        // L1: trailing whitespace (paste, `diff -y`/`column` padding).
        assert!(is_key_line(&format!("{k}   "), false));
        assert!(is_key_line(&format!("{k}\t\t"), false));
        assert!(is_key_line(&format!("{k} \r"), false));
        assert!(is_key_line(&format!("{k}  $"), false));
        assert!(!is_key_line(
            "    let value = compute_something(a, b);",
            false
        ));
        assert!(!is_key_line(
            "    let value = compute_something(a, b);   ",
            false
        ));
        assert!(!is_key_line("short/base64==", false));
        assert!(is_key_line("short/base64==", true));
    }

    /// SS16 negative: a `grep -r` header hit followed by ordinary source
    /// still stops at the 8 KiB minimum.
    #[test]
    fn header_hit_followed_by_source_stops_at_minimum() {
        let s = scrubber(&[]);
        let header = concat!("-----BEGIN RSA ", "PRIVATE", " KEY-----");
        let src = "src/lib.rs:12:    let value = compute_something(argument_one, argument_two);\n";
        let text = format!("docs/a.md:3:{header}\n{}", src.repeat(400));
        let r = s.scrub(&text);
        assert_eq!(r.pem_cut_bytes, MAX_UNTERMINATED_PEM);
        assert!(r.text.contains("compute_something"), "tail lost");
    }

    /// The redaction never runs past the hard stop, however key-like the
    /// lines keep looking.
    #[test]
    fn unterminated_redaction_has_a_hard_stop() {
        let s = scrubber(&[]);
        let header = concat!("-----BEGIN RSA ", "PRIVATE", " KEY-----");
        let body = fake_key_lines(2000).join("\n");
        let r = s.scrub(&format!("{header}\n{body}\n"));
        assert_eq!(r.pem_cut_bytes, MAX_UNTERMINATED_PEM_HARD);
    }

    /// SS17: an END marker far below, from another file, does not close an
    /// unterminated key and make the redaction unbounded.
    #[test]
    fn distant_end_marker_does_not_extend_redaction() {
        let s = scrubber(&[]);
        let header = concat!("-----BEGIN RSA ", "PRIVATE", " KEY-----");
        let src = "src/lib.rs:12:    let value = compute_something(argument_one, argument_two);\n";
        let text = format!(
            "a.md:3:{header}\n{}c.pem:9:-----END CERTIFICATE-----\nkept tail\n",
            src.repeat(400)
        );
        let r = s.scrub(&text);
        assert_eq!(r.pem_cut_bytes, MAX_UNTERMINATED_PEM);
        assert!(r.text.contains("END CERTIFICATE"), "distant END swallowed");
        assert!(r.text.contains("kept tail"));
        // A real END inside the window still closes the block.
        let lines = fake_key_lines(3).join("\n");
        let closed = format!(
            "{header}\n{lines}\n{}\nafter",
            concat!("-----END RSA ", "PRIVATE", " KEY-----")
        );
        let r = s.scrub(&closed);
        assert_eq!(r.pem_cut_bytes, 0);
        assert!(r.text.ends_with("\nafter"), "{}", r.text);
    }

    /// SS4: single-line JSON (`jq -c`) has a literal `\n`, not a newline.
    #[test]
    fn single_line_json_private_key_is_removed() {
        let s = scrubber(&[]);
        let text = concat!(
            r#"{"key":"-----BEGIN "#,
            "PRIVATE",
            r#" KEY-----\nMIIEvQIBADAN/Bgkq+hkiG9w\n-----END "#,
            "PRIVATE",
            r#" KEY-----\n","ok":true}"#,
        );
        let r = s.scrub(text);
        assert!(!r.text.contains("MIIEvQIBADAN"), "{}", r.text);
        assert!(r.text.starts_with(r#"{"key":""#));
        assert!(r.text.contains(r#""ok":true}"#), "{}", r.text);
        // Same shape, no END and no real newline at all.
        let cut = concat!(
            r#"{"key":"-----BEGIN EC "#,
            "PRIVATE",
            r#" KEY-----\nMHcCAQEEIBBB/xx+yy"#,
        );
        assert!(!s.scrub(cut).text.contains("MHcCAQEEIBBB"));
    }

    /// SS8: a known-pass stand-in is not replaced again by the shape pass.
    #[test]
    fn known_stand_in_is_not_rescrubbed() {
        let tok = GHP;
        let s = scrubber(&[tok]);
        let r = s.scrub(&format!("x {tok} y"));
        assert_eq!(r.known, 1);
        assert_eq!(r.shaped, 0, "double-counted: {}", r.text);
        assert!(r.text.contains(&s.stand_in(tok)));
    }

    /// (b): every emitted stand-in is refused on the way back in.
    #[test]
    fn issued_stand_ins_are_refused_in_args() {
        let known = LIN;
        let s = scrubber(&[known]);
        let shaped = GHP;
        let out = s.scrub(&format!("{known}\nGITHUB={shaped}")).text;
        let fake_known = out.lines().next().unwrap_or_default().to_string();
        let fake_shaped = out.lines().nth(1).unwrap_or_default()[7..].to_string();
        for fake in [&fake_known, &fake_shaped] {
            let args =
                serde_json::json!({"path": "config.toml", "content": format!("k = \"{fake}\"")});
            let err = s.check_args(&args.to_string()).unwrap_err();
            assert!(
                err.contains(fake.as_str()) && err.contains("Nothing was run"),
                "{err}"
            );
            assert!(s.check_args_value(&args).is_err());
        }
        // Clean args, and the original secret itself, pass.
        assert!(s.check_args(r#"{"command":"git status"}"#).is_ok());
        assert!(s.check_args(known).is_ok());
        assert!(s
            .check_args_value(&serde_json::json!({"command": "git status"}))
            .is_ok());
    }

    /// SS15: the indexed lookup keeps substring semantics. A stand-in glued
    /// to other word characters, or one of thousands, is still found, and
    /// the answer matches the old linear scan.
    #[test]
    fn issued_index_matches_linear_scan() {
        let s = scrubber(&[]);
        let mut fakes = Vec::new();
        for i in 0..3000u32 {
            let tok = format!("{}{:0>36}", concat!("gh", "p_"), i);
            let out = s.scrub(&format!("t={tok}")).text;
            fakes.push(out[2..].to_string());
        }
        assert_eq!(s.issued_len(), 3000);
        let issued = s.issued.read().unwrap_or_else(|e| e.into_inner());
        let linear = |t: &str| issued.set.iter().any(|f| t.contains(f.as_str()));
        let probes = [
            format!("prefix{}suffix", fakes[1234]),
            format!("{} at the end", fakes[2999]),
            fakes[0][..fakes[0].len() - 1].to_string(),
            "nothing here at all, just a plain sentence of text".to_string(),
            format!("ünï {} ünï", fakes[7]),
        ];
        for p in &probes {
            assert_eq!(issued.find_in(p).is_some(), linear(p), "{p}");
        }
        assert_eq!(issued.find_in(&probes[0]), Some(fakes[1234].as_str()));
    }

    /// SS15: stand-ins shorter than the index key still go through the
    /// fallback scan.
    #[test]
    fn issued_index_short_fallback() {
        let mut i = Issued::default();
        i.insert("abc");
        i.insert("abcdefghijkl");
        assert_eq!(i.len(), 2);
        assert_eq!(i.find_in("xxabcxx"), Some("abc"));
        assert_eq!(i.find_in("xxabcdefghijklxx"), Some("abcdefghijkl"));
        assert_eq!(i.find_in("ab"), None);
        i.insert("abc");
        assert_eq!(i.len(), 2, "re-insert must not duplicate");
    }

    /// Q3: the removed-key marker is refused only once a key was removed.
    #[test]
    fn marker_refused_only_after_a_key_was_removed() {
        let s = scrubber(&[]);
        let args = serde_json::json!({ "content": PEM_REMOVED_MARKER });
        assert!(s.check_args_value(&args).is_ok());
        let pem = concat!("-----BEGIN ", "PRIVATE", " KEY-----\nMIIEvQ\n-----END ");
        let _ = s.scrub(&format!("{pem}{}", concat!("PRIVATE", " KEY-----")));
        assert!(s.check_args_value(&args).is_err());
    }

    /// SS13: a known secret holding `"` or `\` is JSON-escaped in the
    /// serialized args; the value walk still finds its stand-in.
    #[test]
    fn stand_in_with_quote_is_found_in_parsed_args() {
        let secret = r#"Pa55"wo\rd-Xy9zQ"#;
        let s = scrubber(&[secret]);
        let fake = s.scrub(secret).text;
        assert_ne!(fake, secret);
        let args = serde_json::json!({ "content": format!("pw = {fake}") });
        assert!(
            s.check_args(&args.to_string()).is_ok(),
            "escaped form misses"
        );
        assert!(s.check_args_value(&args).is_err());
        // Object keys are checked too.
        let keyed = serde_json::json!({ fake.clone(): 1 });
        assert!(s.check_args_value(&keyed).is_err());
    }

    #[test]
    fn scrub_tool_result_keeps_id_and_error_flag() {
        let s = scrubber(&[]);
        let r = crate::tool::ToolResult {
            tool_use_id: "call_1".into(),
            content: format!("x {GHP}"),
            is_error: true,
        };
        let (out, n) = s.scrub_tool_result(r);
        assert_eq!(n, 1);
        assert_eq!(out.tool_use_id, "call_1");
        assert!(out.is_error);
        assert!(!out.content.contains(GHP));
    }

    /// SS11: this file is read and patched by agents through the scrubber.
    /// If any of it is credential-shaped, the fleet can neither read nor edit
    /// it once deployed.
    #[test]
    fn module_source_is_left_alone() {
        let src = include_str!("secret_scrub.rs");
        let r = scrubber(&[]).scrub(src);
        let first_diff = src
            .lines()
            .zip(r.text.lines())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, _))| format!("line {}: {a}", i + 1));
        assert_eq!(r.total(), 0, "first changed: {first_diff:?}");
    }

    #[test]
    fn scrub_for_model_puts_notice_first() {
        let s = scrubber(&[]);
        let tok = GHP;
        let (out, n) = s.scrub_for_model(format!("token {tok}"));
        assert_eq!(n, 1);
        assert!(
            out.starts_with("[openfang: 1 credential-shaped value(s)"),
            "{out}"
        );
        assert!(!out.contains(tok));
        let (clean, n) = s.scrub_for_model("nothing here".to_string());
        assert_eq!((clean.as_str(), n), ("nothing here", 0));
    }

    #[test]
    fn replace_known_refreshes_the_set() {
        let s = scrubber(&[]);
        let tok = "plainvaultonly0value9xyz";
        assert_eq!(s.scrub(tok).total(), 0);
        assert_eq!(s.replace_known(vec![tok.to_string()]), 1);
        assert!(!s.scrub(tok).text.contains(tok));
    }

    #[test]
    fn new_prefixes_and_single_segment_guard() {
        let s = scrubber(&[]);
        for tok in [
            concat!("whsec_", "A1b2C3d4E5f6", "G7h8I9j0K1l2"),
            concat!("re_", "A1b2C3d4E5f6", "G7h8I9j0K1l2"),
            concat!("sb_secret_", "A1b2C3d4E5f6", "G7h8I9j0"),
        ] {
            assert!(!s.scrub(tok).text.contains(tok), "{tok}");
        }
        let ident = "re_export_handles_for_v2_runtime";
        assert_eq!(s.scrub(ident).text, ident);
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
