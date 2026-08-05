//! Fleet sweep for the context-file injection scanner (ANAI-149).
//!
//! Read-only tuning harness: walks one or more directories, scans every
//! context file it finds, and prints one JSON object per hit plus a summary to
//! stderr. Used to build the false-positive corpus before any enforcement is
//! considered. It never writes to the files it reads.
//!
//! ```text
//! cargo run -p openfang-runtime --example context_scan_sweep -- ~/.openfang/workspaces
//! ```

use openfang_runtime::context_scan::{scan, Severity};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const NAMES: &[&str] = &[
    "SOUL.md",
    "USER.md",
    "MEMORY.md",
    "AGENTS.md",
    "BOOTSTRAP.md",
    "IDENTITY.md",
    "HEARTBEAT.md",
    "TOOLS.md",
    "context.md",
];

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            walk(&path, depth + 1, out);
        } else if ft.is_file() {
            let name = entry.file_name().to_string_lossy().to_string();
            if NAMES.contains(&name.as_str()) {
                out.push(path);
            }
        }
    }
}

fn main() {
    let roots: Vec<String> = std::env::args().skip(1).collect();
    if roots.is_empty() {
        eprintln!("usage: context_scan_sweep <dir> [dir...]");
        std::process::exit(2);
    }

    let mut files = Vec::new();
    for root in &roots {
        walk(Path::new(root), 0, &mut files);
    }
    files.sort();

    let mut by_rule: BTreeMap<&str, usize> = BTreeMap::new();
    let mut files_with_hits = 0usize;
    let mut total_hits = 0usize;
    let mut high_hits = 0usize;

    for path in &files {
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let hits = scan(&content);
        if hits.is_empty() {
            continue;
        }
        files_with_hits += 1;
        for h in &hits {
            total_hits += 1;
            if h.severity == Severity::High {
                high_hits += 1;
            }
            *by_rule.entry(h.rule).or_insert(0) += 1;
            println!(
                r#"{{"file":"{}","rule":"{}","category":"{}","severity":"{}","line":{},"excerpt":"{}"}}"#,
                esc(&path.display().to_string()),
                h.rule,
                h.category,
                h.severity.label(),
                h.line,
                esc(&h.excerpt),
            );
        }
    }

    eprintln!("--- sweep summary ---");
    eprintln!("files scanned:   {}", files.len());
    eprintln!("files with hits: {files_with_hits}");
    eprintln!("total hits:      {total_hits} ({high_hits} high)");
    for (rule, n) in &by_rule {
        eprintln!("  {n:>5}  {rule}");
    }
}
