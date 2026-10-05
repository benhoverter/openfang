//! `[[upload_targets]]`: the per-agent allowlist `file_upload` sends to.
//!
//! `file_upload` moves workspace bytes off the machine, so where it may send
//! them is operator policy, written in `agent.toml` and never supplied by the
//! agent. An agent with no `[[upload_targets]]` table has no targets, and the
//! tool refuses every call: absence is denial, not "anywhere".
//!
//! ```toml
//! [[upload_targets]]
//! name          = "zernio-media"
//! url_prefix    = "https://late-media.<account>.r2.cloudflarestorage.com/temp/"
//! method        = "PUT"
//! content_types = ["image/png", "image/jpeg", "image/webp", "video/mp4"]
//! max_bytes     = 50_000_000
//! ```
//!
//! Rows are `deny_unknown_fields`, like the binding types in `config.rs`: a
//! typo in an allowlist (`url_prefx`, `max_byte`) must fail the manifest load
//! loudly rather than leave a field defaulted. The consequence is a deploy
//! rule: no `agent.toml` gets an `[[upload_targets]]` block until the binary
//! that understands it is running, or that one manifest fails to load.
//!
//! This module is data plus the checks that need no URL parser. Matching a
//! request URL against `url_prefix` happens in the runtime, on the parsed URL
//! (scheme + host + port + path prefix), never on the raw string.

use serde::{Deserialize, Serialize};

/// One destination `file_upload` may send to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadTarget {
    /// Operator label, used in results and audit lines. Not matched against
    /// anything the agent sends.
    pub name: String,
    /// `https://host[:port]/path/prefix`. Must be https, carry no query,
    /// fragment or credentials, and is matched on whole path segments: a
    /// prefix ending in `/` covers everything beneath it, and one that does
    /// not covers itself and `<prefix>/...`, never `<prefix>x`.
    pub url_prefix: String,
    /// Request shape. Only `PUT` (raw body) exists today; a target that needs
    /// multipart POST gets a new variant, not a free-form string.
    #[serde(default)]
    pub method: UploadMethod,
    /// MIME types this target accepts, exact (`image/png`) or a whole
    /// top-level type (`image/*`). Required: there is no "anything" default.
    pub content_types: Vec<String>,
    /// Largest file, in bytes, sent to this target. Required.
    pub max_bytes: u64,
}

/// How the body is sent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UploadMethod {
    /// The file's bytes as the raw request body.
    #[default]
    #[serde(rename = "PUT")]
    Put,
}

impl UploadMethod {
    /// The HTTP verb.
    pub fn as_str(self) -> &'static str {
        match self {
            UploadMethod::Put => "PUT",
        }
    }
}

/// Hard ceiling on any target's `max_bytes`, whatever the manifest says. The
/// file is read whole into memory before sending, so this bounds that too.
pub const UPLOAD_MAX_BYTES_CEILING: u64 = 512 * 1024 * 1024;

impl UploadTarget {
    /// Does this target accept `content_type`? Case-insensitive. `type/*`
    /// covers every subtype of `type`; a bare `*` or `*/*` is never honoured,
    /// because "any type" is what the field exists to prevent.
    pub fn accepts_content_type(&self, content_type: &str) -> bool {
        let wanted = content_type.trim().to_ascii_lowercase();
        let Some((wanted_top, wanted_sub)) = wanted.split_once('/') else {
            return false;
        };
        if wanted_top.is_empty() || wanted_sub.is_empty() || wanted_sub.contains('*') {
            return false;
        }
        self.content_types.iter().any(|allowed| {
            let allowed = allowed.trim().to_ascii_lowercase();
            match allowed.split_once('/') {
                Some((top, "*")) => !top.is_empty() && top != "*" && top == wanted_top,
                Some(_) => allowed == wanted,
                None => false,
            }
        })
    }

    /// The size limit actually enforced: `max_bytes`, capped at
    /// [`UPLOAD_MAX_BYTES_CEILING`].
    pub fn effective_max_bytes(&self) -> u64 {
        self.max_bytes.min(UPLOAD_MAX_BYTES_CEILING)
    }

    /// Problems with this row that need no URL parser, in plain words. Empty
    /// means none found here; the runtime still parses `url_prefix` and
    /// refuses a target whose prefix does not parse as an https URL.
    pub fn static_errors(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.name.trim().is_empty() {
            errs.push("name is empty".to_string());
        }
        if !self
            .url_prefix
            .get(..8)
            .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
        {
            errs.push("url_prefix must start with https://".to_string());
        }
        if self.url_prefix.contains(['?', '#', '@', '\\']) {
            errs.push("url_prefix must not contain '?', '#', '@' or '\\'".to_string());
        }
        if self.content_types.is_empty() {
            errs.push("content_types is empty, so nothing could be sent".to_string());
        }
        for ct in &self.content_types {
            let ok = match ct.trim().split_once('/') {
                Some((top, sub)) => {
                    !top.is_empty()
                        && top != "*"
                        && !sub.is_empty()
                        && (sub == "*" || !sub.contains('*'))
                }
                None => false,
            };
            if !ok {
                errs.push(format!(
                    "content_types entry '{ct}' is not 'type/subtype' or 'type/*'"
                ));
            }
        }
        if self.max_bytes == 0 {
            errs.push("max_bytes is 0, so nothing could be sent".to_string());
        }
        errs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentManifest;

    const ZERNIO: &str = r#"
name = "kimiya-marketing"

[[upload_targets]]
name          = "zernio-media"
url_prefix    = "https://late-media.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com/temp/"
method        = "PUT"
content_types = ["image/png", "image/jpeg", "image/webp", "video/mp4"]
max_bytes     = 50_000_000
"#;

    #[test]
    fn a_manifest_without_the_table_has_no_targets() {
        let m: AgentManifest = toml::from_str("name = \"plain\"").unwrap();
        assert!(m.upload_targets.is_empty());
        assert!(AgentManifest::default().upload_targets.is_empty());
    }

    #[test]
    fn the_zernio_row_parses() {
        let m: AgentManifest = toml::from_str(ZERNIO).unwrap();
        assert_eq!(m.upload_targets.len(), 1);
        let t = &m.upload_targets[0];
        assert_eq!(t.name, "zernio-media");
        assert_eq!(t.method, UploadMethod::Put);
        assert_eq!(t.max_bytes, 50_000_000);
        assert!(t.static_errors().is_empty(), "{:?}", t.static_errors());
    }

    #[test]
    fn method_defaults_to_put() {
        let t: UploadTarget = toml::from_str(
            "name = \"a\"\nurl_prefix = \"https://h/\"\ncontent_types = [\"image/png\"]\nmax_bytes = 1",
        )
        .unwrap();
        assert_eq!(t.method, UploadMethod::Put);
    }

    /// A typo must fail the whole manifest load, not leave a field defaulted.
    #[test]
    fn an_unknown_field_fails_the_manifest_load() {
        let typo = ZERNIO.replace("url_prefix", "url_prefx");
        assert!(toml::from_str::<AgentManifest>(&typo).is_err());
        let extra = ZERNIO.replace("max_bytes", "headers = {}\nmax_bytes");
        assert!(toml::from_str::<AgentManifest>(&extra).is_err());
    }

    #[test]
    fn missing_limits_fail_the_load() {
        let no_max = ZERNIO.replace("max_bytes     = 50_000_000", "");
        assert!(toml::from_str::<AgentManifest>(&no_max).is_err());
        let no_types = ZERNIO.replace(
            "content_types = [\"image/png\", \"image/jpeg\", \"image/webp\", \"video/mp4\"]",
            "",
        );
        assert!(toml::from_str::<AgentManifest>(&no_types).is_err());
    }

    #[test]
    fn an_unknown_method_fails_the_load() {
        let post = ZERNIO.replace("\"PUT\"", "\"POST\"");
        assert!(toml::from_str::<AgentManifest>(&post).is_err());
        let lower = ZERNIO.replace("\"PUT\"", "\"put\"");
        assert!(toml::from_str::<AgentManifest>(&lower).is_err());
    }

    fn target(types: &[&str]) -> UploadTarget {
        UploadTarget {
            name: "t".into(),
            url_prefix: "https://h/".into(),
            method: UploadMethod::Put,
            content_types: types.iter().map(|s| s.to_string()).collect(),
            max_bytes: 10,
        }
    }

    #[test]
    fn content_type_matching() {
        let t = target(&["image/png", "video/*"]);
        assert!(t.accepts_content_type("image/png"));
        assert!(t.accepts_content_type("IMAGE/PNG"));
        assert!(t.accepts_content_type("video/mp4"));
        assert!(!t.accepts_content_type("image/jpeg"));
        assert!(!t.accepts_content_type("application/octet-stream"));
        assert!(
            !t.accepts_content_type("video/*"),
            "a wildcard request is not a type"
        );
        assert!(!t.accepts_content_type("png"));
        assert!(!t.accepts_content_type(""));
    }

    #[test]
    fn any_type_wildcards_are_never_honoured() {
        for t in [target(&["*/*"]), target(&["*"]), target(&["*/png"])] {
            assert!(
                !t.accepts_content_type("image/png"),
                "{:?}",
                t.content_types
            );
            assert!(!t.static_errors().is_empty());
        }
    }

    #[test]
    fn static_errors_catch_bad_rows() {
        let mut t = target(&["image/png"]);
        assert!(t.static_errors().is_empty());
        t.url_prefix = "http://h/".into();
        assert!(!t.static_errors().is_empty());
        t.url_prefix = "https://h/?x=1".into();
        assert!(!t.static_errors().is_empty());
        t.url_prefix = "https://user@h/".into();
        assert!(!t.static_errors().is_empty());
        let mut t = target(&[]);
        assert!(!t.static_errors().is_empty());
        t.content_types = vec!["image/png".into()];
        t.max_bytes = 0;
        assert!(!t.static_errors().is_empty());
    }

    #[test]
    fn the_ceiling_caps_max_bytes() {
        let mut t = target(&["image/png"]);
        t.max_bytes = u64::MAX;
        assert_eq!(t.effective_max_bytes(), UPLOAD_MAX_BYTES_CEILING);
        t.max_bytes = 5;
        assert_eq!(t.effective_max_bytes(), 5);
    }
}
