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
    ///
    /// Credentials (`user@` / `user:pass@` before the host) fail the
    /// manifest load itself, not just the call: a password in an allowlist
    /// row is a config error the operator should see at boot.
    #[serde(deserialize_with = "de_url_prefix")]
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

/// The authority (`user:pass@host:port`) of a URL-ish string: the text
/// between `scheme://` and the first `/`, `?`, `#` or `\` after it. Empty
/// when there is no `://`.
fn authority_of(raw: &str) -> &str {
    let Some((_, rest)) = raw.split_once("://") else {
        return "";
    };
    let end = rest.find(['/', '?', '#', '\\']).unwrap_or(rest.len());
    &rest[..end]
}

/// `url_prefix` deserializer: refuses userinfo at load time. Everything else
/// is left to [`UploadTarget::static_errors`] and the runtime's parser, which
/// refuse a bad row at call time.
fn de_url_prefix<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    if authority_of(&s).contains('@') {
        // Never echo the value: it may hold a password.
        return Err(serde::de::Error::custom(
            "url_prefix must not carry credentials (user@ or user:pass@ before \
             the host); put the bare https://host/path prefix here",
        ));
    }
    Ok(s)
}

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
        if let Some(host) = self.multi_tenant_root_host() {
            errs.push(format!(
                "url_prefix is the bare root of {host}, a host where anyone can own \
                 a bucket, so it would match URLs presigned for someone else's \
                 storage. Include the bucket in the prefix (https://{host}/<bucket>/) \
                 or use the bucket's own hostname"
            ));
        }
        errs
    }

    /// `Some(host)` when `url_prefix` names a shared, path-style object-store
    /// host with no bucket segment. On those hosts the first path segment is
    /// the owner, and anyone can create one.
    fn multi_tenant_root_host(&self) -> Option<String> {
        let rest = self.url_prefix.get(8..)?; // after "https://"
        let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
        let host = authority
            .rsplit_once(':')
            .map_or(authority, |(h, _)| h)
            .trim_end_matches('.')
            .to_ascii_lowercase();
        let shared = host == "storage.googleapis.com"
            || host == "storage.cloud.google.com"
            || host == "s3.amazonaws.com"
            || ((host.starts_with("s3.") || host.starts_with("s3-"))
                && host.ends_with(".amazonaws.com"))
            // Backblaze B2 S3 API: s3.<region>.backblazeb2.com
            || (host.starts_with("s3.") && host.ends_with(".backblazeb2.com"))
            // Wasabi: s3.wasabisys.com, s3.<region>.wasabisys.com
            || (host.starts_with("s3.") && host.ends_with(".wasabisys.com"))
            // DigitalOcean Spaces path-style: <region>.digitaloceanspaces.com
            // (bucket-style <bucket>.<region>.… has one more label and is
            // owned by that bucket).
            || host == "digitaloceanspaces.com"
            || (host.ends_with(".digitaloceanspaces.com") && host.split('.').count() == 3);
        let bucketless = path.trim_matches('/').is_empty();
        (shared && bucketless).then_some(host)
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
    fn bare_shared_object_store_roots_are_refused() {
        let mut t = target(&["image/png"]);
        for bad in [
            "https://s3.amazonaws.com/",
            "https://s3.us-west-2.amazonaws.com",
            "https://S3.AMAZONAWS.COM:443/",
            "https://storage.googleapis.com/",
            "https://s3.us-west-004.backblazeb2.com/",
            "https://s3.wasabisys.com/",
            "https://s3.eu-central-1.wasabisys.com",
            "https://nyc3.digitaloceanspaces.com/",
            "https://SFO3.DigitalOceanSpaces.com:443",
        ] {
            t.url_prefix = bad.into();
            assert!(!t.static_errors().is_empty(), "{bad} must be refused");
        }
        for ok in [
            "https://s3.amazonaws.com/my-bucket/",
            "https://my-bucket.s3.amazonaws.com/",
            "https://storage.googleapis.com/my-bucket/up/",
            "https://late-media.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com/temp/",
            "https://s3.us-west-004.backblazeb2.com/my-bucket/",
            "https://s3.wasabisys.com/my-bucket/",
            "https://my-space.nyc3.digitaloceanspaces.com/",
            "https://nyc3.digitaloceanspaces.com/my-space/",
        ] {
            t.url_prefix = ok.into();
            assert!(
                t.static_errors().is_empty(),
                "{ok}: {:?}",
                t.static_errors()
            );
        }
    }

    #[test]
    fn the_ceiling_caps_max_bytes() {
        let mut t = target(&["image/png"]);
        t.max_bytes = u64::MAX;
        assert_eq!(t.effective_max_bytes(), UPLOAD_MAX_BYTES_CEILING);
        t.max_bytes = 5;
        assert_eq!(t.effective_max_bytes(), 5);
    }

    /// Credentials in `url_prefix` fail the manifest load, not just the call.
    /// Our own message never repeats the value. The toml crate's error
    /// display does quote the offending source line, so the full error can
    /// contain it; that text is the operator's own file, already on disk.
    #[test]
    fn credentials_in_url_prefix_fail_the_manifest_load() {
        let host = "late-media.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com";
        let pw = concat!("hunter", "2pw");
        for creds in ["user@".to_string(), format!("user:{pw}@"), ":@".to_string()] {
            let bad = ZERNIO.replace(
                &format!("https://{host}/"),
                &format!("https://{creds}{host}/"),
            );
            assert_ne!(bad, ZERNIO, "fixture replace must hit");
            let err = toml::from_str::<AgentManifest>(&bad).unwrap_err();
            assert!(err.message().contains("credentials"), "{err}");
            assert!(!err.message().contains(pw), "password echoed: {err}");
        }
        // '@' after the host is not userinfo; static_errors still refuses it
        // at call time, but it is not a load failure.
        let in_path = ZERNIO.replace("/temp/", "/te@mp/");
        let m: AgentManifest = toml::from_str(&in_path).unwrap();
        assert!(!m.upload_targets[0].static_errors().is_empty());
    }
}
