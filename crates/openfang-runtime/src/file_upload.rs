//! `file_upload`: send one workspace file to a destination the operator
//! allowed in `agent.toml` (`[[upload_targets]]`).
//!
//! This is a new way for bytes to leave the machine, so it is narrow by
//! construction. It does one request shape (PUT the file as the raw body to a
//! URL under a configured prefix) and nothing else: no custom headers, no
//! request bodies the agent writes, no GETs, no response bodies returned. The
//! targets live only in the manifest; the agent supplies a URL, and the URL
//! must fall under one of them.
//!
//! Rules enforced here whatever the manifest says:
//!
//! * **The URL is matched parsed, never as a string.** Scheme, host, port and
//!   a whole-segment path prefix. `https://bucket.r2.dev.evil.com/` does not
//!   match `https://bucket.r2.dev/`, and `/temp/` does not match `/tempx/`.
//! * **The URL is sent exactly as received.** Presigned URLs carry a signature
//!   over their own text (`%2F` inside `X-Amz-Credential`, for one), so the
//!   tool refuses any URL that its parser would re-serialise differently
//!   rather than "fixing" it. The string on the wire is the string the agent
//!   passed.
//! * **https only, redirects never followed**, and the host's addresses are
//!   resolved, checked against loopback/private/link-local/metadata ranges,
//!   and then *pinned* for the request, so a DNS answer cannot change between
//!   the check and the connect.
//! * **The file's type comes from its bytes.** The declared `content_type`
//!   must match what the bytes are and be one the target accepts; it is then
//!   sent as the `Content-Type` header.
//! * **Presigned URLs are credentials.** The query string never appears in a
//!   result, an error, or a log line. Everything outward-facing uses
//!   [`redact`]: scheme, host, port, path.
//! * **Known secrets are refused in the bytes.** The type sniff only checks a
//!   file's first 16 bytes, so `%PDF-1.7\n<config.toml>` passes as a PDF.
//!   Every file is therefore scanned with the secret scrubber's exact-match
//!   pass and refused on any hit. The shape pass is not run: on compressed
//!   media it is noise. This stops accidents, not a determined agent: a
//!   base64'd or Flate-compressed secret is not seen. `web_fetch` with a body
//!   remains the wider outbound channel.
//! * **No proxy.** The client ignores `HTTPS_PROXY`/`ALL_PROXY` and system
//!   proxy settings: through a proxy the proxy resolves the host, and the
//!   pinned, checked addresses would describe a connection that never
//!   happens.
//!
//! Not covered: a presigner that signs with temporary credentials puts
//! `X-Amz-Security-Token=` in the URL, and the global scrubber's assignment
//! rule replaces that value with a stand-in before the call arrives. The
//! stand-in refusal then blocks the call. That fails closed, but such
//! presigners will not work until it is handled.

use std::net::{IpAddr, SocketAddr};
use std::path::Path;

use openfang_types::upload::{UploadMethod, UploadTarget};
use reqwest::Url;
use tracing::{info, warn};

/// Advertised description. Byte-identical to the bridge's copy and pinned
/// equal by the cross-crate test in `openfang-api`.
pub const FILE_UPLOAD_DESCRIPTION: &str = "Upload one workspace file to a URL the operator allowed for you in agent.toml ([[upload_targets]]), typically a presigned upload URL another tool just returned. Sends the file's bytes as the body of an HTTP PUT with Content-Type set to content_type, and returns only the HTTP status and ETag, never the response body. Pass the url exactly as you received it: it is sent unchanged, and a URL that is not under an allowed target is refused before anything is read or sent. The file's type is checked from its bytes and must match content_type. Paths resolve through the same file policy as file_read. Treat a presigned URL as a password: do not post it in channels.";

/// `file_upload`'s argument schema. Duplicated in the bridge and pinned equal.
pub fn file_upload_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "The workspace file to upload" },
            "url": { "type": "string", "description": "The upload URL, exactly as received (for a presigned URL, the whole string including its query)" },
            "content_type": { "type": "string", "description": "MIME type of the file, e.g. image/png. Must match the file's bytes and be accepted by the target." }
        },
        "required": ["path", "url", "content_type"]
    })
}

/// Total time allowed for one upload, connect included.
const UPLOAD_TIMEOUT_SECS: u64 = 300;
/// Connect phase only.
const CONNECT_TIMEOUT_SECS: u64 = 15;

// ---------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------

/// `scheme://host[:port]/path` — everything but the query and fragment.
/// The only form of a URL that is ever logged or returned.
pub(crate) fn redact(url: &Url) -> String {
    let mut out = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
    if let Some(port) = url.port() {
        out.push_str(&format!(":{port}"));
    }
    out.push_str(url.path());
    out
}

/// Redaction for a string that may not even parse: cut at the first `?` or
/// `#`, then drop any `user:pass@`.
pub(crate) fn redact_raw(raw: &str) -> String {
    let cut = raw.find(['?', '#']).map(|i| &raw[..i]).unwrap_or(raw);
    match cut.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
            let host = authority.rsplit('@').next().unwrap_or(authority);
            format!("{scheme}://{host}{path}")
        }
        None => cut.to_string(),
    }
}

// ---------------------------------------------------------------------------
// URL checks
// ---------------------------------------------------------------------------

/// Parse the agent's URL and refuse anything that is not a plain https URL
/// this tool would send byte-for-byte as given.
pub(crate) fn parse_request_url(raw: &str) -> Result<Url, String> {
    let shown = redact_raw(raw);
    let url = Url::parse(raw).map_err(|e| format!("'{shown}' is not a valid URL ({e})"))?;
    if url.scheme() != "https" {
        return Err(format!(
            "'{shown}' is not https. file_upload only sends over https."
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(format!(
            "'{shown}' carries credentials before the host (user@host). Refused."
        ));
    }
    if url.fragment().is_some() {
        return Err(format!("'{shown}' has a #fragment. Refused."));
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(format!("'{shown}' has no host."));
    }
    let lower_path = url.path().to_ascii_lowercase();
    if ["%2e", "%2f", "%5c"]
        .iter()
        .any(|enc| lower_path.contains(enc))
    {
        return Err(format!(
            "'{shown}' has an encoded '.', '/' or '\\' in its path, which could \
             step outside the allowed prefix after decoding. Refused."
        ));
    }
    if url.as_str() != raw {
        return Err(format!(
            "'{shown}' is not in canonical form (its host case, dot segments, \
             escaping or default port would be rewritten before sending), and \
             rewriting it could break its signature. Pass the URL exactly as the \
             issuing service returned it. Nothing was sent."
        ));
    }
    Ok(url)
}

/// Parse a target's `url_prefix`, refusing a row that cannot safely be
/// matched against.
pub(crate) fn parse_prefix(target: &UploadTarget) -> Result<Url, String> {
    let errs = target.static_errors();
    if !errs.is_empty() {
        return Err(errs.join("; "));
    }
    let url = Url::parse(&target.url_prefix).map_err(|e| format!("url_prefix: {e}"))?;
    if url.scheme() != "https" {
        return Err("url_prefix must be https".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("url_prefix must not carry credentials".to_string());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("url_prefix must not carry a query or fragment".to_string());
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err("url_prefix has no host".to_string());
    }
    Ok(url)
}

/// Does `prefix` cover `url`? Same scheme, same host, same effective port,
/// and a path prefix that ends on a segment boundary.
pub(crate) fn prefix_covers(prefix: &Url, url: &Url) -> bool {
    if prefix.scheme() != url.scheme()
        || prefix.host_str() != url.host_str()
        || prefix.port_or_known_default() != url.port_or_known_default()
    {
        return false;
    }
    let p = prefix.path();
    let u = url.path();
    if p.ends_with('/') {
        u.starts_with(p)
    } else {
        u == p || u.strip_prefix(p).is_some_and(|rest| rest.starts_with('/'))
    }
}

/// Pick the target that covers `url`: the most specific (longest prefix
/// path) if several do. Misconfigured rows are skipped and logged, never
/// matched.
pub(crate) fn select_target<'a>(
    targets: &'a [UploadTarget],
    url: &Url,
) -> Result<&'a UploadTarget, String> {
    if targets.is_empty() {
        return Err(
            "file_upload is not set up for this agent: its agent.toml has no \
             [[upload_targets]] table, so there is nowhere it may upload. Nothing \
             was sent. Ask the operator to add the destination if this upload is \
             expected."
                .to_string(),
        );
    }
    let mut best: Option<(&UploadTarget, usize)> = None;
    for t in targets {
        match parse_prefix(t) {
            Ok(prefix) => {
                if prefix_covers(&prefix, url) {
                    let len = prefix.path().len();
                    if best.is_none_or(|(_, l)| len > l) {
                        best = Some((t, len));
                    }
                }
            }
            Err(reason) => warn!(
                target_name = %t.name,
                %reason,
                "file_upload: upload target is misconfigured and will never match"
            ),
        }
    }
    best.map(|(t, _)| t).ok_or_else(|| {
        let allowed: Vec<String> = targets
            .iter()
            .map(|t| format!("'{}' ({})", t.name, t.url_prefix))
            .collect();
        format!(
            "'{}' is not under any upload target allowed for this agent. Allowed: {}. \
             Nothing was sent.",
            redact(url),
            allowed.join(", ")
        )
    })
}

// ---------------------------------------------------------------------------
// Network checks
// ---------------------------------------------------------------------------

/// Addresses an upload must never reach.
pub(crate) fn is_forbidden_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_unspecified()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_documentation()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10 CGNAT
                || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24 IETF protocol assignments
                || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18.0.0/15 benchmarking
                || o[0] >= 240
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_forbidden_ip(&IpAddr::V4(v4));
            }
            let s = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || s[..6].iter().all(|&x| x == 0) // ::/96 deprecated IPv4-compatible
                || (s[0] & 0xfe00) == 0xfc00 // unique local
                || (s[0] & 0xffc0) == 0xfe80 // link local
                || (s[0] & 0xffc0) == 0xfec0 // site local (deprecated)
                || (s[0] == 0x64 && s[1] == 0xff9b) // NAT64 64:ff9b::/32, incl. local-use 64:ff9b:1::/48
                || s[0] == 0x2002 // 6to4, embeds a v4 address
                || (s[0] == 0x2001 && s[1] == 0) // Teredo
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
        }
    }
}

/// Resolve `host` and refuse if ANY address is forbidden. The returned list
/// is what the request is pinned to.
async fn resolve_public(host: &str, port: u16) -> Result<Vec<SocketAddr>, String> {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let addrs: Vec<SocketAddr> = if let Ok(ip) = bare.parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((bare, port))
            .await
            .map_err(|e| format!("could not resolve {bare}: {e}"))?
            .collect()
    };
    if addrs.is_empty() {
        return Err(format!("{bare} resolved to no addresses"));
    }
    if let Some(bad) = addrs.iter().find(|a| is_forbidden_ip(&a.ip())) {
        return Err(format!(
            "{bare} resolves to {}, a loopback, private, link-local or metadata \
             address. Uploads only go to public hosts. Nothing was sent.",
            bad.ip()
        ));
    }
    Ok(addrs)
}

// ---------------------------------------------------------------------------
// File checks
// ---------------------------------------------------------------------------

/// The type of an uploadable file, from its leading bytes.
pub(crate) fn sniff_upload_mime(head: &[u8]) -> Option<&'static str> {
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if head.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg");
    }
    if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if head.starts_with(b"%PDF-") {
        return Some("application/pdf");
    }
    // ISO base media: `....ftyp<brand>`. Only the brands that are actually
    // MP4/QuickTime video; HEIC/AVIF share the box and are not video.
    if head.len() >= 12 && &head[4..8] == b"ftyp" {
        return match &head[8..12] {
            b"isom" | b"iso2" | b"iso4" | b"iso5" | b"iso6" | b"mp41" | b"mp42" | b"avc1"
            | b"M4V " | b"dash" => Some("video/mp4"),
            b"qt  " => Some("video/quicktime"),
            _ => None,
        };
    }
    None
}

/// Refuse bytes that contain a value the secret scrubber knows. Exact match
/// only; see the module docs for what this does and does not catch. With no
/// scrubber installed (tests, or a daemon that failed to build one) there is
/// nothing to match against and the upload proceeds, logged.
pub(crate) fn refuse_known_secrets(
    bytes: &[u8],
    scrubber: Option<&openfang_types::secret_scrub::SecretScrubber>,
    raw_path: &str,
) -> Result<(), String> {
    let Some(s) = scrubber else {
        warn!("file_upload: no secret scrubber installed; file bytes not scanned");
        return Ok(());
    };
    let hits = s.known_hits_in_bytes(bytes);
    if hits > 0 {
        return Err(format!(
            "'{raw_path}' contains {hits} known credential value(s) in its bytes. \
             Uploading it would send them off the machine. Nothing was sent."
        ));
    }
    Ok(())
}

/// The path the open file descriptor actually refers to, from the kernel.
/// `None` where the platform offers no way to ask.
#[cfg(target_os = "macos")]
fn fd_path(file: &std::fs::File) -> Option<std::path::PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::AsRawFd;
    let mut buf = vec![0u8; libc::PATH_MAX as usize + 1];
    // SAFETY: F_GETPATH writes a NUL-terminated path of at most MAXPATHLEN
    // (== PATH_MAX) bytes into the buffer, which is larger than that.
    let r = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETPATH, buf.as_mut_ptr()) };
    if r == -1 {
        return None;
    }
    let len = buf.iter().position(|&b| b == 0)?;
    Some(std::ffi::OsStr::from_bytes(&buf[..len]).into())
}

#[cfg(target_os = "linux")]
fn fd_path(file: &std::fs::File) -> Option<std::path::PathBuf> {
    use std::os::unix::io::AsRawFd;
    std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd())).ok()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn fd_path(_file: &std::fs::File) -> Option<std::path::PathBuf> {
    None
}

/// Read the file without following a symlink at its final component. The
/// path was canonicalised by the resolver; `O_NOFOLLOW` closes the window
/// where that last component is swapped for a link afterwards. After the
/// open, the kernel's own path for the descriptor must equal `path`, which
/// catches an intermediate directory swapped in between; and a file with
/// more than one hard link is refused, since canonicalising never sees a
/// hard link into the workspace from outside it.
async fn read_no_follow(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
        use std::io::Read;
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.custom_flags(libc::O_NOFOLLOW);
        }
        let file = opts
            .open(&path)
            .map_err(|e| format!("could not open the file: {e}"))?;
        let meta = file
            .metadata()
            .map_err(|e| format!("could not stat the file: {e}"))?;
        if !meta.is_file() {
            return Err("not a regular file".to_string());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if meta.nlink() > 1 {
                return Err(format!(
                    "the file has {} hard links; file_upload only sends files \
                     with exactly one",
                    meta.nlink()
                ));
            }
        }
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            match fd_path(&file) {
                Some(real) if real == path => {}
                _ => {
                    return Err("the file opened is not the one the path resolved to (a \
                         directory on the path changed during the call)"
                        .to_string())
                }
            }
        }
        if meta.len() > max {
            return Err(format!(
                "{} bytes, over this target's limit of {max}",
                meta.len()
            ));
        }
        let mut buf = Vec::with_capacity(meta.len() as usize);
        // +1 so growth past the limit after the stat is seen, not truncated.
        file.take(max + 1)
            .read_to_end(&mut buf)
            .map_err(|e| format!("could not read the file: {e}"))?;
        if buf.len() as u64 > max {
            return Err(format!(
                "grew past this target's limit of {max} bytes while being read"
            ));
        }
        Ok(buf)
    })
    .await
    .map_err(|e| format!("read task failed: {e}"))?
}

// ---------------------------------------------------------------------------
// The tool
// ---------------------------------------------------------------------------

/// Run one upload. `resolved` has already been through the file policy and
/// the prevalidation check in `tool_runner`; `targets` is the caller's
/// `[[upload_targets]]`.
pub(crate) async fn run(
    input: &serde_json::Value,
    raw_path: &str,
    resolved: &Path,
    targets: &[UploadTarget],
    caller: Option<&str>,
) -> Result<String, String> {
    let raw_url = input["url"].as_str().ok_or("Missing 'url' parameter")?;
    let content_type = input["content_type"]
        .as_str()
        .ok_or("Missing 'content_type' parameter")?
        .trim()
        .to_ascii_lowercase();

    // URL and target first: a refused destination never causes a file read.
    let url = parse_request_url(raw_url)?;
    let target = select_target(targets, &url)?;
    let shown = redact(&url);
    if !target.accepts_content_type(&content_type) {
        return Err(format!(
            "upload target '{}' does not accept {content_type}. It accepts: {}. \
             Nothing was sent.",
            target.name,
            target.content_types.join(", ")
        ));
    }
    match target.method {
        UploadMethod::Put => {}
    }

    let max = target.effective_max_bytes();
    let bytes = read_no_follow(resolved, max)
        .await
        .map_err(|e| format!("'{raw_path}': {e}. Nothing was sent."))?;
    if bytes.is_empty() {
        return Err(format!(
            "'{raw_path}' is empty (0 bytes). Nothing was sent."
        ));
    }
    let head = &bytes[..bytes.len().min(16)];
    let sniffed = sniff_upload_mime(head);
    if sniffed != Some(content_type.as_str()) {
        return Err(format!(
            "'{raw_path}' was declared {content_type}, but its bytes are {}. The type \
             is decided by the file's bytes, not its name or the declared type. \
             Nothing was sent.",
            sniffed.unwrap_or(
                "not a type file_upload recognises (PNG, JPEG, GIF, WebP, PDF, MP4, QuickTime)"
            )
        ));
    }
    refuse_known_secrets(
        &bytes,
        openfang_types::secret_scrub::global().map(|s| s.as_ref()),
        raw_path,
    )?;

    let sha256 = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(&bytes))
    };
    let host = url.host_str().unwrap_or_default().to_string();
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs = resolve_public(&host, port).await?;

    let client = reqwest::Client::builder()
        .user_agent(crate::USER_AGENT)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .https_only(true)
        .resolve_to_addrs(host.trim_start_matches('[').trim_end_matches(']'), &addrs)
        .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .timeout(std::time::Duration::from_secs(UPLOAD_TIMEOUT_SECS))
        .build()
        .map_err(|e| format!("could not build the HTTP client: {}", e.without_url()))?;

    let size = bytes.len();
    let sent = client
        .put(url.clone())
        .header(reqwest::header::CONTENT_TYPE, content_type.as_str())
        .body(bytes)
        .send()
        .await;

    let response = match sent {
        Ok(r) => r,
        Err(e) => {
            info!(
                target: "file_upload.audit",
                agent = caller.unwrap_or("unknown"),
                upload_target = %target.name,
                %host,
                path = url.path(),
                bytes = size,
                %sha256,
                outcome = "transport_error",
                "file_upload"
            );
            return Err(format!(
                "upload of '{raw_path}' to {shown} failed before a response: {}. \
                 It may or may not have arrived; nothing from the server was received.",
                e.without_url()
            ));
        }
    };
    let status = response.status();
    let etag = response
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.chars().take(128).collect::<String>());
    // Body deliberately dropped unread.
    drop(response);

    info!(
        target: "file_upload.audit",
        agent = caller.unwrap_or("unknown"),
        upload_target = %target.name,
        %host,
        path = url.path(),
        bytes = size,
        %sha256,
        status = status.as_u16(),
        outcome = if status.is_success() { "uploaded" } else { "http_error" },
        "file_upload"
    );

    let etag_note = etag.map(|e| format!(", ETag {e}")).unwrap_or_default();
    if status.is_success() {
        Ok(format!(
            "Uploaded '{raw_path}' ({content_type}, {size} bytes, sha256 {sha256}) to \
             upload target '{}' at {shown}: HTTP {}{etag_note}. The response body is \
             not returned.",
            target.name,
            status.as_u16()
        ))
    } else {
        Err(format!(
            "upload of '{raw_path}' to {shown} was refused: HTTP {}{etag_note}. The \
             response body is withheld. For a presigned URL, 403 usually means it \
             expired, was already used, or was issued for a different file or \
             content type: get a fresh one.",
            status.as_u16()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real Zernio shape (signature replaced): `%2F` inside the
    /// credential, `UNSIGNED-PAYLOAD`, host-only signing.
    const ZERNIO_URL: &str = "https://late-media.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com/temp/1791235047257_0x3wsp0c_kimiya-upload-test.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=0c2cf252753a4d0cae599921de485cdf%2F20261005%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261005T211727Z&X-Amz-Expires=3600&X-Amz-Signature=0000000000000000000000000000000000000000000000000000000000000000&X-Amz-SignedHeaders=host&x-id=PutObject";
    const ZERNIO_PREFIX: &str =
        "https://late-media.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com/temp/";

    fn target(prefix: &str) -> UploadTarget {
        UploadTarget {
            name: "zernio-media".into(),
            url_prefix: prefix.into(),
            method: UploadMethod::Put,
            content_types: vec!["image/png".into(), "video/mp4".into()],
            max_bytes: 1_000,
        }
    }

    #[test]
    fn the_zernio_url_is_accepted_and_kept_byte_for_byte() {
        let url = parse_request_url(ZERNIO_URL).expect("accepted");
        assert_eq!(
            url.as_str(),
            ZERNIO_URL,
            "the wire string must be the input"
        );
        assert!(
            url.as_str().contains("%2F20261005%2Fauto"),
            "%2F must survive"
        );
        let targets = [target(ZERNIO_PREFIX)];
        assert_eq!(select_target(&targets, &url).unwrap().name, "zernio-media");
    }

    #[test]
    fn redaction_never_shows_the_query() {
        let url = parse_request_url(ZERNIO_URL).unwrap();
        let shown = redact(&url);
        assert!(
            !shown.contains('?') && !shown.contains("Signature") && !shown.contains("Credential")
        );
        assert!(shown.ends_with("_kimiya-upload-test.png"));
        let raw = redact_raw("https://u:p@h.example/x?sig=secret#f");
        assert_eq!(raw, "https://h.example/x");
    }

    /// Every refusal message is built from the redacted form.
    #[test]
    fn errors_never_echo_the_signature() {
        let secret = "X-Amz-Signature=deadbeef";
        for raw in [
            format!("http://h.example/a?{secret}"),
            format!("https://H.EXAMPLE/a?{secret}"),
            format!("https://h.example/a/../b?{secret}"),
            format!("https://u@h.example/a?{secret}"),
            format!("not a url ?{secret}"),
        ] {
            let err = parse_request_url(&raw).unwrap_err();
            assert!(!err.contains("deadbeef"), "leaked in: {err}");
        }
        let url = parse_request_url(&format!("https://other.example/a?{secret}")).unwrap();
        let err = select_target(&[target(ZERNIO_PREFIX)], &url).unwrap_err();
        assert!(!err.contains("deadbeef"), "leaked in: {err}");
    }

    #[test]
    fn host_must_match_exactly_not_by_substring() {
        let prefix = Url::parse("https://bucket.r2.dev/").unwrap();
        for bad in [
            "https://bucket.r2.dev.evil.com/x",
            "https://evilbucket.r2.dev/x",
            "https://x.bucket.r2.dev/x",
            "https://bucket.r2.dev:8443/x",
            "http://bucket.r2.dev/x",
        ] {
            let u = Url::parse(bad).unwrap();
            assert!(!prefix_covers(&prefix, &u), "{bad} must not match");
        }
        let u = Url::parse("https://bucket.r2.dev:443/x").unwrap();
        assert!(
            prefix_covers(&prefix, &u),
            "explicit default port is the same port"
        );
    }

    #[test]
    fn path_prefix_stops_on_a_segment_boundary() {
        let slash = Url::parse("https://h.example/temp/").unwrap();
        let bare = Url::parse("https://h.example/temp").unwrap();
        let ok = Url::parse("https://h.example/temp/a.png").unwrap();
        let sibling = Url::parse("https://h.example/tempx/a.png").unwrap();
        let exact = Url::parse("https://h.example/temp").unwrap();
        assert!(prefix_covers(&slash, &ok));
        assert!(!prefix_covers(&slash, &sibling));
        assert!(prefix_covers(&bare, &ok));
        assert!(prefix_covers(&bare, &exact));
        assert!(!prefix_covers(&bare, &sibling));
    }

    #[test]
    fn traversal_and_encoded_separators_are_refused() {
        for bad in [
            "https://h.example/temp/../secret",
            "https://h.example/temp/./a",
            "https://h.example/temp/%2e%2e/secret",
            "https://h.example/temp/..%2Fsecret",
            "https://h.example/temp/a%5cb",
            "https://h.example/temp\\..\\x",
        ] {
            assert!(parse_request_url(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn non_canonical_urls_are_refused_not_rewritten() {
        for bad in [
            "https://H.example/temp/a",     // host case
            "https://h.example:443/temp/a", // default port spelled out
            "https://h.example/temp/a b",   // unescaped space
            "https://h.example",            // no path: parser adds '/'
        ] {
            assert!(parse_request_url(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn other_shapes_are_refused() {
        for bad in [
            "http://h.example/a",
            "ftp://h.example/a",
            "https://user:pw@h.example/a",
            "https://h.example/a#frag",
            "file:///etc/passwd",
        ] {
            assert!(parse_request_url(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn no_targets_means_refused_with_a_reason() {
        let url = parse_request_url("https://h.example/a").unwrap();
        let err = select_target(&[], &url).unwrap_err();
        assert!(err.contains("[[upload_targets]]"));
    }

    #[test]
    fn misconfigured_targets_never_match() {
        let url = parse_request_url("https://h.example/a/b").unwrap();
        for prefix in [
            "http://h.example/a/",
            "https://h.example/a/?x=1",
            "https://u@h.example/a/",
            "https://h.example/a/#f",
        ] {
            assert!(select_target(&[target(prefix)], &url).is_err(), "{prefix}");
        }
        let mut no_types = target("https://h.example/a/");
        no_types.content_types.clear();
        assert!(select_target(&[no_types], &url).is_err());
    }

    #[test]
    fn the_most_specific_target_wins() {
        let url = parse_request_url("https://h.example/a/b/c.png").unwrap();
        let mut wide = target("https://h.example/a/");
        wide.name = "wide".into();
        let mut narrow = target("https://h.example/a/b/");
        narrow.name = "narrow".into();
        assert_eq!(
            select_target(&[wide.clone(), narrow.clone()], &url)
                .unwrap()
                .name,
            "narrow"
        );
        assert_eq!(select_target(&[narrow, wide], &url).unwrap().name, "narrow");
    }

    #[test]
    fn forbidden_addresses() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "100.100.100.200",
            "0.0.0.0",
            "::1",
            "::",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "64:ff9b::a00:1",
            "64:ff9b:1::a00:1",
            "192.0.0.1",
            "192.0.0.170",
            "::7f00:1",
            "::a00:1",
            "2002:a00:1::",
            "2001:0:4136:e378::1",
            "fec0::1",
        ] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(is_forbidden_ip(&ip), "{ip} must be forbidden");
        }
        for ip in ["104.18.1.1", "1.1.1.1", "2606:4700::1111"] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(!is_forbidden_ip(&ip), "{ip} is public");
        }
    }

    #[tokio::test]
    async fn literal_private_hosts_are_refused_before_any_connect() {
        assert!(resolve_public("127.0.0.1", 443).await.is_err());
        assert!(resolve_public("[::1]", 443).await.is_err());
        assert!(resolve_public("localhost", 443).await.is_err());
    }

    #[test]
    fn sniffing() {
        assert_eq!(
            sniff_upload_mime(b"\x89PNG\r\n\x1a\n...."),
            Some("image/png")
        );
        assert_eq!(sniff_upload_mime(b"\xff\xd8\xff\xe0"), Some("image/jpeg"));
        assert_eq!(
            sniff_upload_mime(b"RIFF\0\0\0\0WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(sniff_upload_mime(b"\0\0\0\x20ftypisom"), Some("video/mp4"));
        assert_eq!(
            sniff_upload_mime(b"\0\0\0\x20ftypqt  "),
            Some("video/quicktime")
        );
        assert_eq!(
            sniff_upload_mime(b"\0\0\0\x20ftypheic"),
            None,
            "HEIC is not video"
        );
        assert_eq!(sniff_upload_mime(b"%PDF-1.7"), Some("application/pdf"));
        assert_eq!(sniff_upload_mime(b"hello world"), None);
    }

    fn png_bytes() -> Vec<u8> {
        let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
        v.extend_from_slice(&[0u8; 32]);
        v
    }

    async fn run_with(
        dir: &tempfile::TempDir,
        name: &str,
        url: &str,
        ct: &str,
        targets: &[UploadTarget],
    ) -> Result<String, String> {
        let input = serde_json::json!({ "path": name, "url": url, "content_type": ct });
        // The resolver hands `run` a canonical path; on macOS the tempdir
        // lives under /var -> /private/var, so canonicalise like it does.
        let root = dir.path().canonicalize().unwrap();
        run(&input, name, &root.join(name), targets, Some("tester")).await
    }

    #[test]
    fn known_secrets_in_the_bytes_are_refused() {
        let secret = concat!("sk-upload", "-test-0123456789abcdef");
        let s =
            openfang_types::secret_scrub::SecretScrubber::with_key([secret.to_string()], [7u8; 32]);
        let mut disguised = b"%PDF-1.7\n".to_vec();
        disguised.extend_from_slice(format!("api_key = \"{secret}\"\n").as_bytes());
        let e = refuse_known_secrets(&disguised, Some(&s), "x.pdf").unwrap_err();
        assert!(e.contains("Nothing was sent") && !e.contains(secret), "{e}");
        assert!(refuse_known_secrets(&png_bytes(), Some(&s), "a.png").is_ok());
        assert!(refuse_known_secrets(&disguised, None, "x.pdf").is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_hard_linked_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), png_bytes()).unwrap();
        std::fs::hard_link(dir.path().join("a.png"), dir.path().join("b.png")).unwrap();
        let e = run_with(
            &dir,
            "b.png",
            ZERNIO_URL,
            "image/png",
            &[target(ZERNIO_PREFIX)],
        )
        .await
        .unwrap_err();
        assert!(e.contains("hard links"), "{e}");
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn a_non_canonical_path_is_refused_after_open() {
        // Stands in for a directory swapped between resolve and open: the
        // descriptor's real path differs from the one we were handed.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.png"), png_bytes()).unwrap();
        let root = dir.path().canonicalize().unwrap();
        let e = read_no_follow(&root.join("sub/../a.png"), 1_000)
            .await
            .unwrap_err();
        assert!(e.contains("not the one the path resolved to"), "{e}");
        assert!(read_no_follow(&root.join("a.png"), 1_000).await.is_ok());
    }

    /// Refusals that must happen before any network I/O. The URL host is a
    /// public name, so a wrongly-passed check would try to connect; the
    /// expected errors all name a local reason instead.
    #[tokio::test]
    async fn local_refusals_happen_before_sending() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), png_bytes()).unwrap();
        std::fs::write(dir.path().join("a.txt"), b"just text, no secrets here").unwrap();
        std::fs::write(dir.path().join("big.png"), vec![0u8; 2_000]).unwrap();
        let targets = [target(ZERNIO_PREFIX)];

        // Not configured at all.
        let e = run_with(&dir, "a.png", ZERNIO_URL, "image/png", &[])
            .await
            .unwrap_err();
        assert!(e.contains("[[upload_targets]]"), "{e}");
        // Content type the target does not accept.
        let e = run_with(&dir, "a.png", ZERNIO_URL, "image/gif", &targets)
            .await
            .unwrap_err();
        assert!(e.contains("does not accept"), "{e}");
        // Declared PNG, bytes are text.
        let e = run_with(&dir, "a.txt", ZERNIO_URL, "image/png", &targets)
            .await
            .unwrap_err();
        assert!(e.contains("its bytes are"), "{e}");
        // Over max_bytes (1_000).
        let e = run_with(&dir, "big.png", ZERNIO_URL, "image/png", &targets)
            .await
            .unwrap_err();
        assert!(e.contains("limit"), "{e}");
        // Missing file.
        let e = run_with(&dir, "nope.png", ZERNIO_URL, "image/png", &targets)
            .await
            .unwrap_err();
        assert!(e.contains("Nothing was sent"), "{e}");
        assert!(!e.contains("X-Amz-Signature"), "{e}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_is_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("real.png"), png_bytes()).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real.png"), dir.path().join("link.png"))
            .unwrap();
        let e = run_with(
            &dir,
            "link.png",
            ZERNIO_URL,
            "image/png",
            &[target(ZERNIO_PREFIX)],
        )
        .await
        .unwrap_err();
        assert!(e.contains("could not open"), "{e}");
    }

    /// A target whose host resolves privately is refused at resolution, after
    /// the file checks pass, with nothing sent.
    #[tokio::test]
    async fn a_private_target_host_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), png_bytes()).unwrap();
        let t = target("https://127.0.0.1/up/");
        let e = run_with(
            &dir,
            "a.png",
            "https://127.0.0.1/up/a.png",
            "image/png",
            &[t],
        )
        .await
        .unwrap_err();
        assert!(e.contains("public hosts"), "{e}");
    }
}
