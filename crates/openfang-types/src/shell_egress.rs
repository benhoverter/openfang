//! `[shell_egress]`: hosts an agent's shell `curl` may send data to without
//! asking.
//!
//! ```toml
//! [shell_egress]
//! write_hosts = ["api.example.com", "*.uploads.example.com"]
//! ```
//!
//! Operator policy, written in `agent.toml` and never supplied by the agent.
//! A call that sends data (a body, a form, `-G` query data, a file, or
//! credentials) to a host not on this list goes to the operator for approval.
//! Absence of the table and an empty list mean the same thing: no host is
//! trusted for writes, so every such call escalates. Reads are not governed
//! here; there is deliberately no `read_hosts`, and `capabilities.network`
//! (often `"*"`) is not consulted.
//!
//! Entries are bare hostnames, matched case-insensitively on whole labels:
//! `example.com` matches only `example.com`, and `*.example.com` matches any
//! host with at least one label in front of `example.com`, never
//! `example.com` itself and never `evilexample.com`. IP literals, ports,
//! schemes, paths and any other `*` are refused.
//!
//! Some hosts are refused at manifest load because anyone can own data on
//! them, and with no path in an entry nothing pins the owner: the shared
//! object-store roots `file_upload` refuses, tunnel and paste services, and
//! wildcards over a multi-tenant parent (`*.amazonaws.com`, `*.workers.dev`).
//! The list is not a public-suffix list. What it does not catch, and what no
//! gate can (a domain expiring or changing hands), is caught by config
//! review, not at call time.
//!
//! The struct is `deny_unknown_fields` and a bad entry fails the whole
//! manifest load, as with `[[upload_targets]]`: a typo in an allowlist must be
//! loud. The same deploy rule follows: no `agent.toml` gets a
//! `[shell_egress]` table until the binary that understands it is running.

use serde::{Deserialize, Serialize};

/// The `[shell_egress]` table.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellEgress {
    /// Normalized entries: lowercase, no trailing dot.
    #[serde(default, deserialize_with = "de_write_hosts")]
    pub write_hosts: Vec<String>,
}

impl ShellEgress {
    /// Is `host` covered by a `write_hosts` entry? `host` is the parsed host
    /// of a request URL. Case and one trailing dot are ignored.
    pub fn allows_write_to(&self, host: &str) -> bool {
        let host = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
        if host.is_empty() || host.ends_with('.') {
            return false;
        }
        self.write_hosts
            .iter()
            .any(|entry| match entry.strip_prefix('*') {
                // `rest` is ".base": one or more whole labels must precede it.
                Some(rest) => {
                    host.len() > rest.len() && host.ends_with(rest) && {
                        let front = &host[..host.len() - rest.len()];
                        !front.starts_with('.') && !front.ends_with('.')
                    }
                }
                None => host == *entry,
            })
    }
}

fn de_write_hosts<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Vec::<String>::deserialize(d)?;
    raw.iter()
        .map(|e| normalize_write_host(e).map_err(serde::de::Error::custom))
        .collect()
}

/// Tenant services: any host at or under these is someone else's to hand out.
const TENANT_SUFFIXES: &[&str] = &[
    "ngrok.io",
    "ngrok.app",
    "ngrok.dev",
    "ngrok-free.app",
    "ngrok-free.dev",
    "trycloudflare.com",
    "r2.dev",
    "github.io",
    "pastebin.com",
    "paste.ee",
    "hastebin.com",
    "transfer.sh",
    "0x0.st",
    "webhook.site",
    "requestbin.com",
];

/// Parents whose direct subdomains belong to arbitrary customers. Exact hosts
/// under them are fine (`my-bucket.s3.amazonaws.com` is one owner); a
/// wildcard over them is everyone.
const MULTI_TENANT_PARENTS: &[&str] = &[
    "amazonaws.com",
    "googleapis.com",
    "backblazeb2.com",
    "wasabisys.com",
    "digitaloceanspaces.com",
    "cloudflarestorage.com",
    "workers.dev",
    "pages.dev",
    "vercel.app",
    "netlify.app",
    "herokuapp.com",
    "fly.dev",
    "azurewebsites.net",
    "blob.core.windows.net",
    "appspot.com",
];

/// Names that never leave the machine or local network.
const INTERNAL_SUFFIXES: &[&str] = &["localhost", "local", "internal", "arpa", "lan", "home"];

fn at_or_under(host: &str, suffix: &str) -> bool {
    host == suffix
        || host
            .strip_suffix(suffix)
            .is_some_and(|front| front.ends_with('.'))
}

/// Validate one `write_hosts` entry and return its normalized form, or a
/// plain-words reason. Entries are hostnames, not secrets, so echoing them is
/// fine.
fn normalize_write_host(raw: &str) -> Result<String, String> {
    let entry = raw.trim().to_ascii_lowercase();
    let entry = entry.strip_suffix('.').unwrap_or(&entry).to_string();
    let bad = |why: &str| Err(format!("[shell_egress] write_hosts entry '{raw}': {why}"));

    let (wildcard, base) = match entry.strip_prefix("*.") {
        Some(base) => (true, base),
        None => (false, entry.as_str()),
    };
    if base.contains('*') {
        return bad("'*' is allowed only as a whole first label ('*.example.com')");
    }
    if base.contains(['/', ':', '@', '?', '#', '[', ']', ' ']) {
        return bad("must be a bare hostname, with no scheme, port, path or credentials");
    }
    let labels: Vec<&str> = base.split('.').collect();
    if labels.len() < 2 {
        return bad("needs at least two labels (a registrable domain or below)");
    }
    if base.len() > 253 {
        return bad("longer than 253 characters");
    }
    for label in &labels {
        let ok = (1..=63).contains(&label.len())
            && label
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            && !label.starts_with('-')
            && !label.ends_with('-');
        if !ok {
            return bad("labels must be 1-63 letters, digits or inner hyphens");
        }
    }
    if labels
        .last()
        .is_some_and(|l| l.bytes().all(|b| b.is_ascii_digit()))
    {
        return bad("IP addresses are not accepted; name the host");
    }
    if INTERNAL_SUFFIXES.iter().any(|s| at_or_under(base, s)) {
        return bad("names a local or internal host");
    }
    if crate::upload::is_shared_object_store_root(base) {
        return bad(
            "is a shared object-store root where anyone can own a bucket; \
             use the bucket's own hostname",
        );
    }
    if TENANT_SUFFIXES.iter().any(|s| at_or_under(base, s)) {
        return bad("is a tunnel or paste service whose names anyone can claim");
    }
    if wildcard && MULTI_TENANT_PARENTS.iter().any(|s| at_or_under(base, s)) {
        // `*.<acct>.r2.cloudflarestorage.com` is one account; only the bare
        // parents are refused there.
        let r2_account = base.ends_with(".r2.cloudflarestorage.com");
        if !r2_account {
            return bad("a wildcard here covers other customers' hosts; list the exact host");
        }
    }
    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentManifest;

    fn load(toml_body: &str) -> Result<AgentManifest, toml::de::Error> {
        toml::from_str(&format!("name = \"a\"\n{toml_body}"))
    }

    fn egress(hosts: &[&str]) -> ShellEgress {
        let list = hosts
            .iter()
            .map(|h| format!("\"{h}\""))
            .collect::<Vec<_>>()
            .join(", ");
        load(&format!("[shell_egress]\nwrite_hosts = [{list}]"))
            .unwrap()
            .shell_egress
    }

    #[test]
    fn table_absent_means_no_write_hosts() {
        let m = load("").unwrap();
        assert!(m.shell_egress.write_hosts.is_empty());
        assert!(!m.shell_egress.allows_write_to("example.com"));
        assert!(AgentManifest::default().shell_egress.write_hosts.is_empty());
    }

    #[test]
    fn empty_list_and_empty_table_trust_nothing() {
        for body in ["[shell_egress]\nwrite_hosts = []", "[shell_egress]"] {
            let e = load(body).unwrap().shell_egress;
            assert!(e.write_hosts.is_empty(), "{body}");
            assert!(!e.allows_write_to("example.com"), "{body}");
        }
    }

    #[test]
    fn wildcard_matches_whole_labels_only() {
        let e = egress(&["*.example.com"]);
        assert!(e.allows_write_to("api.example.com"));
        assert!(e.allows_write_to("a.b.example.com"));
        assert!(e.allows_write_to("API.Example.COM."));
        assert!(!e.allows_write_to("example.com"));
        assert!(!e.allows_write_to(".example.com"));
        assert!(!e.allows_write_to("a..example.com"));
    }

    #[test]
    fn a_substring_never_matches() {
        let exact = egress(&["example.com"]);
        let wild = egress(&["*.example.com"]);
        for host in [
            "evilexample.com",
            "example.com.evil.net",
            "notexample.com",
            "example.co",
            "xample.com",
        ] {
            assert!(!exact.allows_write_to(host), "exact matched {host}");
            assert!(!wild.allows_write_to(host), "wildcard matched {host}");
        }
        assert!(exact.allows_write_to("EXAMPLE.com."));
        assert!(!exact.allows_write_to("api.example.com"));
    }

    #[test]
    fn an_unknown_key_fails_the_manifest_load() {
        assert!(load("[shell_egress]\nread_hosts = [\"example.com\"]").is_err());
        assert!(load("[shell_egress]\nwrite_host = [\"example.com\"]").is_err());
        assert!(load("[shell_egress]\nwrite_hosts = [\"example.com\"]\nmode = \"x\"").is_err());
    }

    #[test]
    fn malformed_entries_fail_the_load() {
        for bad in [
            "https://example.com",
            "example.com:443",
            "example.com/path",
            "user@example.com",
            "*",
            "*.com",
            "a.*.example.com",
            "*example.com",
            "ex*.com",
            "com",
            "-a.example.com",
            "a_b.example.com",
            "1.2.3.4",
            "127.1",
            "[::1]",
            "",
            "localhost",
            "api.localhost",
            "printer.local",
            "db.internal",
        ] {
            assert!(
                load(&format!("[shell_egress]\nwrite_hosts = [\"{bad}\"]")).is_err(),
                "{bad} must fail the load"
            );
        }
    }

    #[test]
    fn shared_tenant_hosts_fail_the_load() {
        for bad in [
            "s3.amazonaws.com",
            "s3.us-west-2.amazonaws.com",
            "storage.googleapis.com",
            "s3.us-west-004.backblazeb2.com",
            "s3.wasabisys.com",
            "nyc3.digitaloceanspaces.com",
            "pub-abc123.r2.dev",
            "*.r2.dev",
            "abcd.ngrok-free.app",
            "*.ngrok.io",
            "x-y-z.trycloudflare.com",
            "someone.github.io",
            "pastebin.com",
            "webhook.site",
            "*.amazonaws.com",
            "*.s3.amazonaws.com",
            "*.googleapis.com",
            "*.r2.cloudflarestorage.com",
            "*.workers.dev",
            "*.vercel.app",
        ] {
            assert!(
                load(&format!("[shell_egress]\nwrite_hosts = [\"{bad}\"]")).is_err(),
                "{bad} must fail the load"
            );
        }
    }

    #[test]
    fn single_owner_hosts_load() {
        let e = egress(&[
            "my-bucket.s3.amazonaws.com",
            "late-media.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com",
            "*.613d73a46130ed0083c059a837d6511b.r2.cloudflarestorage.com",
            "my-app.workers.dev",
            "API.Example.COM.",
        ]);
        assert!(e.write_hosts.contains(&"api.example.com".to_string()));
        assert!(e.allows_write_to("my-bucket.s3.amazonaws.com"));
        assert!(!e.allows_write_to("other-bucket.s3.amazonaws.com"));
    }

    /// Normalized entries survive a serialize / deserialize round trip.
    #[test]
    fn round_trips_through_json() {
        let e = egress(&["*.Example.com", "api.example.net."]);
        let back: ShellEgress = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
        assert_eq!(e, back);
    }
}
