//! Following the local files a script imports, sources or runs (2026-11-11).
//!
//! The gate reads the one script a command runs. Moving the interesting lines
//! into a sibling the script imports, sources or launches used to put them out
//! of sight: Kimiya's `install-fixtures.mjs` split did exactly that by accident.
//! This walks those edges. Every local file reached gets the same read
//! guardrails ([`read_guarded`]) and the same predicates ([`BodyScan`]) as the
//! entry, and its flags are OR'd into the entry's [`ScriptBody`].
//!
//! Edges, per security's review of the proposal:
//!
//! - **R1, any language:** a string literal (or, in shell, a word) with a
//!   script suffix that names an existing file, looked up against the
//!   script's own directory and then the working directory. Covers
//!   `subprocess.run(["python3", "w.py"])`, `spawn('./w.js')`, `runpy`,
//!   `exec(open('w.py').read())`, `./w.sh` and the like.
//! - **R2, Python:** `import a.b`, `from a import b`, relative imports, and a
//!   literal `importlib.import_module("a")` / `__import__("a")`. Absolute
//!   imports resolve against the *entry's* directory (`sys.path[0]`), relative
//!   ones against the module's own. A compiled artifact for a local name
//!   (`.pyc`, `.so`, `.dylib`, `.pyd`, or a `__pycache__`-only module) cannot be
//!   scanned and is blind.
//! - **R3, JS/TS:** relative `import`/`export … from`/`import()`/`require()`
//!   specifiers with Node's candidate list, `index.*` and `package.json` main.
//!   A `.node` addon, or a relative specifier that matches nothing, is blind.
//! - **R4, shell:** `source x` / `. x` against the tracked working directory
//!   first and the script's directory second, following literal `cd`s. A
//!   non-literal path, or one that resolves nowhere, is blind.
//!
//! Anything the walk cannot settle sets [`ScriptBody::links_blind`], which
//! routes the run to the judge. Depth, file-count and byte caps do the same;
//! the walk never truncates silently (R5c).
//!
//! Known gaps, filed: modules loaded by a name built at run time
//! (`import_module(name)`, `require(variable)`, template literals with
//! `${}`), `PYTHONPATH`/`NODE_PATH` pointing elsewhere, code read from data
//! files and `eval`'d, bash's `PATH` search for a slash-less `source` name, and
//! extensionless executables run by path.

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use openfang_types::config::FilePolicy;
use openfang_types::path_facts::{PathFact, ScriptBody};

use crate::path_facts::{annotate_git, read_guarded, resolve, stat_fact, BodyScan};

/// Levels of links below the entry. The entry is depth 0.
const MAX_DEPTH: usize = 3;
/// Distinct linked files, counted after de-duplication (R5b).
const MAX_FILES: usize = 20;
/// Bytes read across all linked files, on top of the entry.
const MAX_BYTES: u64 = 4 * 1024 * 1024;
/// Cap on the `scan_hits` block handed to the judge.
const MAX_HITS_BYTES: usize = 8 * 1024;

const SCRIPT_SUFFIXES: &[&str] = &[
    ".py", ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".sh", ".bash", ".zsh",
];
/// Node's extension candidates, plus what `tsx` adds.
const JS_EXTS: &[&str] = &[
    ".js", ".mjs", ".cjs", ".json", ".ts", ".mts", ".cts", ".jsx", ".tsx",
];
const PY_COMPILED: &[&str] = &["pyc", "pyo", "so", "dylib", "pyd"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Lang {
    Python,
    Js,
    Shell,
    Other,
}

fn lang_of(path: &Path, text: &str) -> Lang {
    match path.extension().and_then(|e| e.to_str()) {
        Some("py") => return Lang::Python,
        Some("js" | "mjs" | "cjs" | "ts" | "mts" | "cts" | "jsx" | "tsx") => return Lang::Js,
        Some("sh" | "bash" | "zsh") => return Lang::Shell,
        _ => {}
    }
    let first = text.lines().next().unwrap_or("");
    if !first.starts_with("#!") {
        return Lang::Other;
    }
    if first.contains("python") {
        Lang::Python
    } else if ["node", "deno", "bun", "tsx"]
        .iter()
        .any(|n| first.contains(n))
    {
        Lang::Js
    } else if first.ends_with("sh") || first.contains("sh ") || first.contains("bash") {
        Lang::Shell
    } else {
        Lang::Other
    }
}

/// The language an interpreter runs its file as, whatever the file is called
/// (`python3 ./notes.txt` is Python). `None` for launchers like `npx`, whose
/// argument is not the file's language.
///
/// Not recognised yet: `pypy`/`pypy3`, `pythonw` and `python3-dbg` map to
/// `None`, so their files fall back to the extension or shebang (security F3).
fn lang_of_interpreter(bin: &str) -> Option<Lang> {
    let base = bin.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.');
    match if base.is_empty() { bin } else { base } {
        "python" => Some(Lang::Python),
        "node" | "bun" | "deno" | "tsx" | "ts-node" => Some(Lang::Js),
        "bash" | "sh" | "zsh" | "dash" | "ksh" | "source" | "." => Some(Lang::Shell),
        _ => None,
    }
}

/// Words of a command string, split on whitespace and on the shell operators
/// that can be glued to a name: `python3 w.py;rm x`, `w.py>log`,
/// `cd sub&&python3 w.py` (security W2), and backticks (F2).
fn command_words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| {
        c.is_whitespace() || matches!(c, ';' | '&' | '|' | '<' | '>' | '(' | ')' | '`')
    })
    .filter(|w| !w.is_empty())
}

/// The language a word starts a file in, when the word is an interpreter
/// (bare or by path). `source` and `.` are excluded: shell handles them
/// itself, and a stray "." literal elsewhere is not a runner.
fn runner_lang(word: &str) -> Option<Lang> {
    let bin = word.rsplit('/').next().unwrap_or(word);
    if matches!(bin, "." | "source") {
        return None;
    }
    lang_of_interpreter(bin)
}

/// Interpreter flags whose value is the next word (security L3):
/// `python3 -X utf8 w`, `python3 -W ignore w`, `bash -o pipefail w`.
fn takes_value(lang: Lang, flag: &str) -> bool {
    match lang {
        Lang::Python => matches!(flag, "-X" | "-W" | "-Q"),
        Lang::Shell => matches!(flag, "-o" | "-O" | "+o" | "+O"),
        Lang::Js | Lang::Other => false,
    }
}

/// R1 edges out of a stream of command words. A word after an interpreter
/// (`python3 -u x`, flags skipped) is followed whatever its name and takes
/// that interpreter's language (security F1); any other word is followed
/// only when it has a script suffix.
fn run_links<'a>(
    words: impl Iterator<Item = &'a str>,
    dir: &Path,
    cwd: Option<&Path>,
    out: &mut Vec<Link>,
) {
    let mut runner: Option<Lang> = None;
    let mut flag_value = false;
    for w in words {
        let w = unquote(w);
        if flag_value {
            // A flag's value is not the file, and keeps the runner live.
            flag_value = false;
            continue;
        }
        for path in run_target(&w, runner.is_some(), dir, cwd) {
            out.push(Link {
                path,
                py_root: None,
                cwd: cwd.map(Path::to_path_buf),
                lang: runner,
            });
        }
        if let Some(l) = runner.filter(|_| w.starts_with('-') || w.starts_with('+')) {
            flag_value = takes_value(l, &w);
        } else {
            runner = runner_lang(&w);
        }
    }
}

/// One file the walk has read.
struct Node {
    path: PathBuf,
    text: String,
    lang: Lang,
    depth: usize,
    /// `sys.path[0]` for this file's absolute imports: the directory of the
    /// Python script that was *run*, inherited across imports.
    py_root: PathBuf,
    /// Working directory when this file runs, `None` when unknowable.
    cwd: Option<PathBuf>,
}

/// One edge out of a node.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Link {
    path: PathBuf,
    /// `Some` when the target is imported (inherits `sys.path[0]`), `None`
    /// when it is run (its own directory becomes `sys.path[0]`).
    py_root: Option<PathBuf>,
    cwd: Option<PathBuf>,
    /// The language of the interpreter that runs it, when one was named.
    lang: Option<Lang>,
}

/// Walk the links out of `entry`, folding every linked file's flags into
/// `body`. On anything it cannot settle it records why in
/// [`ScriptBody::links_blind`] and stops; flags already folded stay set.
pub(crate) async fn walk(
    body: &mut ScriptBody,
    entry: &Path,
    entry_text: &str,
    interpreter: Option<&str>,
    canon_ws: Option<&Path>,
    file_policy: Option<&FilePolicy>,
) {
    let mut hits: Vec<String> = Vec::new();
    let entry_lang = interpreter
        .and_then(lang_of_interpreter)
        .unwrap_or_else(|| lang_of(entry, entry_text));
    let result = walk_inner(
        body,
        &mut hits,
        entry,
        entry_text,
        entry_lang,
        canon_ws,
        file_policy,
    )
    .await;
    if let Err(reason) = result {
        body.links_blind = Some(reason);
    }
    if !hits.is_empty() {
        let mut all = body.scan_hits.take().unwrap_or_default();
        for h in hits {
            if all.len() + h.len() > MAX_HITS_BYTES {
                all.push_str("[...matches in more linked files not shown]\n");
                break;
            }
            all.push_str(&h);
        }
        body.scan_hits = Some(all);
    }
}

async fn walk_inner(
    body: &mut ScriptBody,
    hits: &mut Vec<String>,
    entry: &Path,
    entry_text: &str,
    entry_lang: Lang,
    canon_ws: Option<&Path>,
    file_policy: Option<&FilePolicy>,
) -> Result<(), String> {
    let dir_of = |p: &Path| p.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut finder = Finder::default();
    // A file is walked once per language it can run as (security G1): the
    // first link to reach it must not decide that for every later one.
    // `followed` de-duplicates links before a read, `walked` de-duplicates
    // the walks after it, `files` counts distinct files for the cap.
    let mut followed: BTreeSet<(PathBuf, Option<Lang>)> = BTreeSet::new();
    let mut walked: BTreeSet<(PathBuf, Lang)> = BTreeSet::new();
    let mut files: BTreeSet<PathBuf> = BTreeSet::new();
    let canon_entry = canonical(entry);
    followed.insert((canon_entry.clone(), Some(entry_lang)));
    if lang_of(entry, entry_text) == entry_lang {
        // A plain link back to the entry would walk it the same way again.
        followed.insert((canon_entry.clone(), None));
    }
    walked.insert((canon_entry.clone(), entry_lang));
    files.insert(canon_entry.clone());
    let mut bytes: u64 = 0;
    let mut level = vec![Node {
        path: entry.to_path_buf(),
        text: entry_text.to_string(),
        lang: entry_lang,
        depth: 0,
        py_root: dir_of(entry),
        cwd: canon_ws.map(Path::to_path_buf),
    }];

    while !level.is_empty() {
        let mut fresh: Vec<(Link, usize)> = Vec::new();
        for node in &level {
            for link in finder.links(node)? {
                // De-duplicate before any cap counts (R5b): a cycle or a
                // diamond is not more files.
                let key = canonical(&link.path);
                if !followed.insert((key.clone(), link.lang)) {
                    continue;
                }
                let depth = node.depth + 1;
                if depth > MAX_DEPTH {
                    return Err(format!(
                        "links nest deeper than {MAX_DEPTH} levels (at {})",
                        link.path.display()
                    ));
                }
                files.insert(key);
                if files.len() - 1 > MAX_FILES {
                    return Err(format!("links reach more than {MAX_FILES} files"));
                }
                fresh.push((link, depth));
            }
        }
        if fresh.is_empty() {
            break;
        }

        // Same fact pipeline as the entry: stat, then one bounded git pass
        // for the whole level, then the guarded read.
        let mut facts: Vec<PathFact> = fresh
            .iter()
            .map(|(l, _)| stat_fact(&l.path.display().to_string(), canon_ws, file_policy))
            .collect();
        annotate_git(&mut facts, canon_ws).await;

        let mut next = Vec::new();
        for ((link, depth), fact) in fresh.into_iter().zip(&facts) {
            let (real, text) = read_guarded(fact, canon_ws, file_policy)
                .map_err(|status| format!("{} is {}", link.path.display(), status.as_token()))?;
            bytes += text.len() as u64;
            if bytes > MAX_BYTES {
                return Err(format!("linked files exceed {MAX_BYTES} bytes"));
            }
            let scan = BodyScan::of(&text);
            scan.apply(body);
            let shown = real.display().to_string();
            let key = canonical(&real);
            if key != canon_entry && !body.linked.contains(&shown) {
                if let Some(h) = scan.excerpt(&real, &text) {
                    hits.push(h);
                }
                body.linked.push(shown);
            }
            // The runner's language and the file's own (suffix/shebang) can
            // differ: a planted "python3" literal before "./w.sh" must not
            // hide w.sh's shell lines (G1). Walk each language not yet walked.
            let own = lang_of(&real, &text);
            // Later links that would only repeat these walks skip the read.
            followed.insert((key.clone(), None));
            followed.insert((key.clone(), Some(own)));
            let mut langs = vec![link.lang.unwrap_or(own)];
            if link.lang.is_some_and(|l| l != own) {
                langs.push(own);
            }
            for lang in langs {
                if !walked.insert((key.clone(), lang)) {
                    continue;
                }
                next.push(Node {
                    lang,
                    py_root: link.py_root.clone().unwrap_or_else(|| dir_of(&real)),
                    cwd: link.cwd.clone(),
                    path: real.clone(),
                    text: text.clone(),
                    depth,
                });
            }
        }
        level = next;
    }
    Ok(())
}

/// Key for de-duplication. A path that will not canonicalize keys as itself;
/// the read that follows refuses it anyway.
fn canonical(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

fn exists(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok()
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let mut s: OsString = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Edge extraction, with a per-walk cache of directory listings for the
/// compiled-module check.
#[derive(Default)]
struct Finder {
    listings: HashMap<PathBuf, Vec<String>>,
}

impl Finder {
    fn links(&mut self, node: &Node) -> Result<Vec<Link>, String> {
        let dir = node.path.parent().unwrap_or(Path::new("/")).to_path_buf();
        let mut out: Vec<Link> = Vec::new();
        match node.lang {
            Lang::Python => self.python_links(node, &dir, &mut out)?,
            Lang::Js => js_links(node, &dir, &mut out)?,
            Lang::Shell => shell_links(node, &dir, &mut out)?,
            Lang::Other => {}
        }
        if node.lang != Lang::Shell {
            // R1. Shell does this per word, with its tracked cwd. Other
            // languages split each literal into words too, so the
            // single-string forms (os.system("python3 w.py"),
            // execSync('node w.js')) are followed. Words run on across
            // literals so the list form (["python3", "notes.txt"]) sees its
            // interpreter.
            let lits = string_literals(&node.text);
            run_links(
                lits.iter().flat_map(|(_, l)| command_words(l)),
                &dir,
                node.cwd.as_deref(),
                &mut out,
            );
        }
        Ok(out)
    }

    // -- R2 -----------------------------------------------------------------

    fn python_links(&mut self, node: &Node, dir: &Path, out: &mut Vec<Link>) -> Result<(), String> {
        for stmt in python_imports(&node.text) {
            let (base, module, names) = match &stmt {
                PyImport::Absolute { module, names } => (node.py_root.clone(), module, names),
                PyImport::Relative {
                    dots,
                    module,
                    names,
                } => {
                    let mut base = dir.to_path_buf();
                    for _ in 1..*dots {
                        base.pop();
                    }
                    (base, module, names)
                }
            };
            let parts: Vec<&str> = module.split('.').filter(|p| !p.is_empty()).collect();
            let mut found = Vec::new();
            self.resolve_py(&base, &parts, names, &mut found)?;
            out.extend(found.into_iter().map(|path| Link {
                path,
                py_root: Some(node.py_root.clone()),
                cwd: node.cwd.clone(),
                lang: None,
            }));
        }
        Ok(())
    }

    /// Walk `parts` down from `base` the way the path finder would, pushing
    /// every source file that executes. A first part with nothing local is
    /// not ours (stdlib or site-packages) and yields nothing.
    fn resolve_py(
        &mut self,
        base: &Path,
        parts: &[&str],
        names: &[String],
        out: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        let mut dir = base.to_path_buf();
        for part in parts {
            self.compiled_check(&dir, part)?;
            let pkg = dir.join(part);
            if pkg.is_dir() {
                // Regular package or namespace portion; both are local.
                self.compiled_check(&pkg, "__init__")?;
                let init = pkg.join("__init__.py");
                if exists(&init) {
                    out.push(init);
                }
                dir = pkg;
                continue;
            }
            let file = dir.join(format!("{part}.py"));
            if exists(&file) {
                out.push(file);
            }
            // A module, or nothing local. Either way the remaining parts and
            // the imported names are attributes, not files.
            return Ok(());
        }
        for name in names {
            self.compiled_check(&dir, name)?;
            let sub = dir.join(name);
            if sub.is_dir() {
                self.compiled_check(&sub, "__init__")?;
                let init = sub.join("__init__.py");
                if exists(&init) {
                    out.push(init);
                }
            } else {
                let file = dir.join(format!("{name}.py"));
                if exists(&file) {
                    out.push(file);
                }
            }
        }
        Ok(())
    }

    /// R2: a local name that resolves to something other than source. An
    /// extension module outranks `name.py` in the path finder, and a sourceless
    /// `.pyc` imports when the source is gone, so either is a one-line way to
    /// put code where no scan reads it.
    fn compiled_check(&mut self, dir: &Path, name: &str) -> Result<(), String> {
        let prefix = format!("{name}.");
        let compiled = |f: &str| {
            f.starts_with(&prefix)
                && f.rsplit('.')
                    .next()
                    .is_some_and(|ext| PY_COMPILED.contains(&ext))
        };
        if let Some(hit) = self.listing(dir).iter().find(|f| compiled(f)) {
            return Err(format!("compiled module {}", dir.join(hit).display()));
        }
        if !exists(&dir.join(format!("{name}.py"))) && !dir.join(name).is_dir() {
            let cache = dir.join("__pycache__");
            if let Some(hit) = self.listing(&cache).iter().find(|f| compiled(f)) {
                return Err(format!("compiled module {}", cache.join(hit).display()));
            }
        }
        Ok(())
    }

    fn listing(&mut self, dir: &Path) -> &Vec<String> {
        self.listings.entry(dir.to_path_buf()).or_insert_with(|| {
            std::fs::read_dir(dir)
                .map(|rd| {
                    rd.filter_map(Result::ok)
                        .filter_map(|e| e.file_name().into_string().ok())
                        .take(10_000)
                        .collect()
                })
                .unwrap_or_default()
        })
    }
}

/// R1: every existing script file a literal names, against the script's
/// directory AND the working directory. All candidates, not the first: a
/// subprocess resolves against cwd, so a clean decoy in the script dir must
/// not hide the file that actually runs. `after_runner` drops the suffix
/// requirement: whatever follows `python3` is run, whatever it is called.
fn run_target(lit: &str, after_runner: bool, dir: &Path, cwd: Option<&Path>) -> Vec<PathBuf> {
    if lit.len() > 4096
        || lit.is_empty()
        || lit
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '$' | '`' | '{' | '*'))
        || !(after_runner || SCRIPT_SUFFIXES.iter().any(|s| lit.ends_with(s)))
    {
        return Vec::new();
    }
    let mut found: Vec<PathBuf> = [Some(dir), cwd]
        .into_iter()
        .flatten()
        .filter_map(|base| resolve(lit, Some(base)))
        .filter(|p| exists(p))
        .collect();
    found.dedup_by(|a, b| canonical(a) == canonical(b));
    found
}

// -- Python import statements ---------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum PyImport {
    Absolute {
        module: String,
        names: Vec<String>,
    },
    Relative {
        dots: usize,
        module: String,
        names: Vec<String>,
    },
}

/// Import statements, one per logical line. Not a parser: it reads lines
/// that start with `import`/`from` (at any indent), joins parenthesised and
/// backslash continuations, and also takes a literal `import_module("x")` or
/// `__import__("x")`. A false edge only costs a read of a file that exists.
fn python_imports(text: &str) -> Vec<PyImport> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let mut stmt = strip_py_comment(line).trim().to_string();
        for call in ["import_module(", "__import__("] {
            for (prefix, lit) in string_literals(line) {
                if prefix.trim_end().ends_with(call) && !lit.starts_with('.') {
                    out.push(PyImport::Absolute {
                        module: lit,
                        names: Vec::new(),
                    });
                }
            }
        }
        if !(stmt.starts_with("import ") || stmt.starts_with("from ")) {
            continue;
        }
        while stmt.ends_with('\\') || (stmt.contains('(') && !stmt.contains(')')) {
            let Some(next) = lines.next() else { break };
            if stmt.ends_with('\\') {
                stmt.pop();
            }
            stmt.push(' ');
            stmt.push_str(strip_py_comment(next).trim());
        }
        for part in stmt.split(';') {
            out.extend(parse_py_import(part.trim()));
        }
    }
    out
}

fn strip_py_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or("")
}

fn import_names(list: &str) -> Vec<String> {
    list.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .map(|n| n.split(" as ").next().unwrap_or("").trim().to_string())
        .filter(|n| !n.is_empty() && n != "*" && is_ident_path(n))
        .collect()
}

fn is_ident_path(s: &str) -> bool {
    s.split('.')
        .all(|p| !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '_'))
}

fn parse_py_import(stmt: &str) -> Vec<PyImport> {
    if let Some(rest) = stmt.strip_prefix("import ") {
        // `import a.b as c, d` is several absolute imports with no names.
        return import_names(rest)
            .into_iter()
            .map(|module| PyImport::Absolute {
                module,
                names: Vec::new(),
            })
            .collect();
    }
    let Some((source, names)) = stmt
        .strip_prefix("from ")
        .and_then(|rest| rest.split_once(" import "))
    else {
        return Vec::new();
    };
    let source = source.trim();
    let dots = source.chars().take_while(|c| *c == '.').count();
    let module = source[dots..].to_string();
    if !module.is_empty() && !is_ident_path(&module) {
        return Vec::new();
    }
    let names = import_names(names);
    vec![if dots == 0 {
        PyImport::Absolute { module, names }
    } else {
        PyImport::Relative {
            dots,
            module,
            names,
        }
    }]
}

// -- R3 -------------------------------------------------------------------

fn js_links(node: &Node, dir: &Path, out: &mut Vec<Link>) -> Result<(), String> {
    for (prefix, spec) in string_literals(&node.text) {
        if !(spec.starts_with("./") || spec.starts_with("../") || spec.starts_with('/')) {
            continue;
        }
        if !is_module_position(&prefix) {
            continue;
        }
        let path = resolve_js(dir, &spec)?;
        out.push(Link {
            path,
            py_root: None,
            cwd: node.cwd.clone(),
            lang: None,
        });
    }
    Ok(())
}

/// The literal sits where a module specifier goes: after `from`, a bare
/// `import`, or as the argument of `import(` / `require(`.
fn is_module_position(prefix: &str) -> bool {
    let p = prefix.trim_end();
    if p.ends_with("from") || p.ends_with("import") {
        return true;
    }
    p.strip_suffix('(')
        .map(str::trim_end)
        .is_some_and(|q| q.ends_with("require") || q.ends_with("import"))
}

fn resolve_js(dir: &Path, spec: &str) -> Result<PathBuf, String> {
    let blind = |why: &str| Err(format!("{why} {spec}"));
    if spec.ends_with(".node") {
        return blind("native addon");
    }
    let Some(p) = resolve(spec, Some(dir)) else {
        return blind("unresolvable import");
    };
    if p.is_file() {
        return Ok(p);
    }
    if let Some(hit) = js_candidates(&p) {
        return Ok(hit);
    }
    // TypeScript's ESM convention: `./x.js` in source names `x.ts`.
    for (js, ts) in [
        (".js", &[".ts", ".tsx"][..]),
        (".mjs", &[".mts"]),
        (".cjs", &[".cts"]),
    ] {
        if let Some(stem) = spec.strip_suffix(js) {
            if let Some(hit) = ts
                .iter()
                .filter_map(|ext| resolve(&format!("{stem}{ext}"), Some(dir)))
                .find(|c| c.is_file())
            {
                return Ok(hit);
            }
        }
    }
    if exists(&with_suffix(&p, ".node")) {
        return blind("native addon");
    }
    if p.is_dir() {
        if let Some(main) = package_main(&p) {
            if let Some(m) = resolve(&main, Some(&p)) {
                if m.is_file() {
                    return Ok(m);
                }
                if let Some(hit) = js_candidates(&m) {
                    return Ok(hit);
                }
            }
        }
        if let Some(hit) = js_candidates(&p.join("index")) {
            return Ok(hit);
        }
        if exists(&p.join("index.node")) {
            return blind("native addon");
        }
    }
    blind("relative import that matches no file:")
}

fn js_candidates(stem: &Path) -> Option<PathBuf> {
    JS_EXTS
        .iter()
        .map(|ext| with_suffix(stem, ext))
        .find(|c| c.is_file())
}

fn package_main(dir: &Path) -> Option<String> {
    let manifest = dir.join("package.json");
    if std::fs::metadata(&manifest).ok()?.len() > 64 * 1024 {
        return None;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest).ok()?).ok()?;
    let main = v.get("main")?.as_str()?;
    Some(if main.starts_with("./") || main.starts_with("../") {
        main.to_string()
    } else {
        format!("./{main}")
    })
}

// -- R4 and R1 for shell ----------------------------------------------------

fn shell_links(node: &Node, dir: &Path, out: &mut Vec<Link>) -> Result<(), String> {
    let mut cwd = node.cwd.clone();
    for line in node.text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        for seg in line.split([';', '|', '&']) {
            let mut words: Vec<&str> = seg.split_whitespace().collect();
            while words.first().is_some_and(|w| {
                matches!(
                    *w,
                    "then"
                        | "do"
                        | "else"
                        | "{"
                        | "("
                        | "!"
                        | "if"
                        | "while"
                        | "until"
                        | "exec"
                        | "command"
                        | "builtin"
                )
            }) {
                words.remove(0);
            }
            let Some(first) = words.first().copied() else {
                continue;
            };
            match first {
                "cd" => {
                    let rest = words[1..].join(" ");
                    cwd = match words.get(1) {
                        Some(_) if is_dirname_of_self(&rest) => Some(dir.to_path_buf()),
                        Some(arg) => {
                            let a = unquote(arg);
                            if is_shell_literal(&a) {
                                cwd.as_deref().and_then(|c| resolve(&a, Some(c)))
                            } else {
                                None
                            }
                        }
                        None => None,
                    };
                }
                "source" | "." => {
                    let Some(arg) = words.get(1) else { continue };
                    let a = unquote(arg);
                    if !is_shell_literal(&a) {
                        return Err(format!("non-literal source path {arg}"));
                    }
                    let relative = !(a.starts_with('/') || a.starts_with('~'));
                    if relative && cwd.is_none() {
                        return Err(format!("source {a} after a cd the gate cannot follow"));
                    }
                    let hit = [cwd.as_deref(), Some(dir)]
                        .into_iter()
                        .flatten()
                        .filter_map(|base| resolve(&a, Some(base)))
                        .find(|p| exists(p));
                    let Some(path) = hit else {
                        return Err(format!("sourced file {a} not found"));
                    };
                    out.push(Link {
                        path,
                        py_root: None,
                        cwd: cwd.clone(),
                        lang: Some(Lang::Shell),
                    });
                }
                _ => {}
            }
            run_links(
                words.iter().flat_map(|w| command_words(w)),
                dir,
                cwd.as_deref(),
                out,
            );
        }
    }
    Ok(())
}

fn unquote(w: &str) -> String {
    w.trim_matches(|c| c == '"' || c == '\'').to_string()
}

fn is_shell_literal(a: &str) -> bool {
    !a.is_empty()
        && !a
            .chars()
            .any(|c| matches!(c, '$' | '`' | '*' | '?' | '(' | ')' | '[' | '{'))
}

/// `cd "$(dirname "$0")"` and the `BASH_SOURCE` form: the script's own dir.
fn is_dirname_of_self(arg: &str) -> bool {
    arg.contains("dirname") && (arg.contains("$0") || arg.contains("BASH_SOURCE"))
}

// -- shared -----------------------------------------------------------------

/// Quoted literals on each line, with the text before the opening quote.
///
/// Single-line only, backslash escapes honoured, and a backtick literal
/// containing `${` dropped as dynamic. Prose apostrophes pair oddly; the cost
/// is a garbage literal that names no file.
fn string_literals(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let q = chars[i];
            if !matches!(q, '"' | '\'' | '`') {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            let mut lit = String::new();
            let mut closed = false;
            while j < chars.len() {
                match chars[j] {
                    '\\' if j + 1 < chars.len() => {
                        lit.push(chars[j + 1]);
                        j += 2;
                    }
                    c if c == q => {
                        closed = true;
                        break;
                    }
                    c => {
                        lit.push(c);
                        j += 1;
                    }
                }
            }
            if closed && !(q == '`' && lit.contains("${")) {
                out.push((chars[..i].iter().collect(), lit));
            }
            i = j + 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_facts::gather;
    use openfang_types::config::{FileAccessTier, FileRule};
    use openfang_types::path_facts::{ScriptBodyStatus, MAX_SCRIPT_BODY_BYTES};

    const MENTION: &str = "p = os.path.expanduser('~/.openfang/RULES.md')\n";

    fn ws(name: &str) -> (PathBuf, FilePolicy) {
        let dir = std::env::temp_dir().join(format!("openfang-links-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dir = dir.canonicalize().unwrap();
        let policy = FilePolicy::new(
            true,
            FileAccessTier::Deny,
            vec![FileRule {
                path: dir.display().to_string(),
                tier: FileAccessTier::Write,
            }],
        );
        (dir, policy)
    }

    fn put(dir: &Path, rel: &str, text: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    async fn body(dir: &Path, policy: &FilePolicy, cmd: &str) -> ScriptBody {
        gather(cmd, &[], Some(dir), Some(policy))
            .await
            .script_body
            .expect("script body")
    }

    #[tokio::test]
    async fn a_clean_local_import_is_followed_and_skips_the_judge() {
        let (d, p) = ws("clean");
        put(&d, "main.py", "import helper\nhelper.go()\n");
        put(&d, "helper.py", "def go():\n    print('hi')\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(!b.needs_judge(), "{b:?}");
    }

    #[tokio::test]
    async fn an_imported_mention_routes_to_the_judge_with_an_excerpt() {
        let (d, p) = ws("import-hit");
        put(&d, "main.py", "from helper import go\ngo()\n");
        put(&d, "helper.py", &format!("import os\n{MENTION}"));
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.names_control_plane);
        assert!(b.needs_judge());
        let hits = b.scan_hits.expect("excerpt");
        assert!(
            hits.contains("helper.py") && hits.contains("RULES.md"),
            "{hits}"
        );
    }

    /// R1: running a sibling is as easy as importing it.
    #[tokio::test]
    async fn a_sibling_run_by_subprocess_is_followed() {
        let (d, p) = ws("r1-subprocess");
        put(
            &d,
            "main.py",
            "import subprocess\nsubprocess.run(['python3', 'w.py'])\n",
        );
        put(&d, "w.py", &format!("import os\n{MENTION}"));
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(b.needs_judge());
    }

    /// B1: the single-string form, split into words.
    #[tokio::test]
    async fn a_sibling_run_by_os_system_string_is_followed() {
        let (d, p) = ws("b1-os-system");
        put(&d, "main.py", "import os\nos.system(\"python3 w.py\")\n");
        put(&d, "w.py", &format!("import os\n{MENTION}"));
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(b.needs_judge());
    }

    /// B1, JS arm.
    #[tokio::test]
    async fn a_sibling_run_by_exec_sync_string_is_followed() {
        let (d, p) = ws("b1-exec-sync");
        put(
            &d,
            "main.js",
            "const { execSync } = require('child_process');\nexecSync('node ./w.mjs');\n",
        );
        put(&d, "w.mjs", "import fs from 'fs';\nfs.writeFileSync(process.env.HOME + '/.openfang/RULES.md', '');\n");
        let b = body(&d, &p, "node ./main.js").await;
        assert!(b.linked.iter().any(|l| l.ends_with("w.mjs")), "{b:?}");
        assert!(b.needs_judge());
    }

    /// B2: a clean decoy next to the script must not hide the cwd file.
    #[tokio::test]
    async fn a_clean_decoy_in_the_script_dir_does_not_hide_the_cwd_target() {
        let (d, p) = ws("b2-decoy");
        put(
            &d,
            "sub/main.py",
            "import subprocess\nsubprocess.run(['python3', 'w.py'])\n",
        );
        put(&d, "sub/w.py", "print('clean')\n");
        put(&d, "w.py", &format!("import os\n{MENTION}"));
        let b = body(&d, &p, "python3 sub/main.py").await;
        assert_eq!(b.linked.len(), 2, "{b:?}");
        assert!(b.needs_judge());
    }

    /// Security W2: a shell operator glued to the name does not hide it.
    #[tokio::test]
    async fn a_sibling_with_a_glued_operator_is_followed() {
        for (name, call) in [
            ("w2-semi", "os.system(\"python3 w.py;rm x\")"),
            ("w2-redir", "os.system(\"python3 w.py>log\")"),
            ("w2-and", "os.system(\"cd .&&python3 w.py\")"),
            ("w2-pipe", "os.system(\"python3 w.py|tee o\")"),
        ] {
            let (d, p) = ws(name);
            put(&d, "main.py", &format!("import os\n{call}\n"));
            put(&d, "w.py", &format!("import os\n{MENTION}"));
            let b = body(&d, &p, "python3 ./main.py").await;
            assert_eq!(b.linked.len(), 1, "{call}: {b:?}");
            assert!(b.needs_judge(), "{call}");
        }
    }

    /// W2, shell arm: `./w.sh>log` inside a shell script.
    #[tokio::test]
    async fn a_shell_word_with_a_glued_redirect_is_followed() {
        let (d, p) = ws("w2-shell");
        put(&d, "run.sh", "./w.sh>log\n");
        put(&d, "w.sh", "rm -rf ~/.openfang/agents/\n");
        let b = body(&d, &p, "bash ./run.sh").await;
        assert!(b.destroys_substrate, "{b:?}");
    }

    /// The interpreter decides the language, not the file name: an import
    /// in `notes.txt` run by python3 is still followed.
    #[tokio::test]
    async fn the_interpreter_sets_the_language_of_an_oddly_named_entry() {
        let (d, p) = ws("lang-interp");
        put(&d, "notes.txt", "import helper\n");
        put(&d, "helper.py", MENTION);
        let b = body(&d, &p, "python3 ./notes.txt").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(b.needs_judge());

        let (d, p) = ws("lang-interp-js");
        put(&d, "notes.txt", "const h = require('./helper')\n");
        put(
            &d,
            "helper.js",
            "module.exports = '~/.openfang/config.toml'\n",
        );
        let b = body(&d, &p, "node ./notes.txt").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(b.needs_judge());
    }

    #[test]
    fn interpreter_names_map_to_languages() {
        assert_eq!(lang_of_interpreter("python3.12"), Some(Lang::Python));
        assert_eq!(lang_of_interpreter("node22"), Some(Lang::Js));
        assert_eq!(lang_of_interpreter("bash"), Some(Lang::Shell));
        assert_eq!(lang_of_interpreter("."), Some(Lang::Shell));
        assert_eq!(lang_of_interpreter("npx"), None);
        assert_eq!(lang_of_interpreter("ruby"), None);
    }

    /// F1: a run file takes the language of the word that runs it.
    /// `bash w.py` is shell, so its `./inner.sh` run line is followed.
    #[tokio::test]
    async fn a_linked_file_takes_the_language_of_its_runner() {
        let (d, p) = ws("f1-runner-lang");
        put(&d, "main.py", "import os\nos.system(\"bash w.py\")\n");
        put(&d, "w.py", "./inner.sh\n");
        put(&d, "inner.sh", "rm -rf ~/.openfang/agents/\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.linked.iter().any(|l| l.ends_with("inner.sh")), "{b:?}");
        assert!(b.destroys_substrate, "{b:?}");
    }

    /// F1: after an interpreter word, a file is followed whatever its name.
    #[tokio::test]
    async fn a_file_with_no_script_suffix_run_by_an_interpreter_is_followed() {
        let (d, p) = ws("f1-any-name");
        put(
            &d,
            "main.py",
            "import subprocess\nsubprocess.run([\"python3\", \"-u\",\n    \"notes.txt\"])\n",
        );
        put(&d, "notes.txt", MENTION);
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(b.needs_judge());
    }

    /// G1: a file named first without a runner, then with one, is still
    /// walked in the runner's language.
    #[tokio::test]
    async fn a_runner_link_after_a_plain_link_still_sets_the_language() {
        let (d, p) = ws("g1-cat-then-bash");
        put(
            &d,
            "main.py",
            "import os\nos.system(\"cat w.py; bash w.py\")\n",
        );
        put(&d, "w.py", "./inner.sh\n");
        put(&d, "inner.sh", "rm -rf ~/.openfang/agents/\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.linked.iter().any(|l| l.ends_with("inner.sh")), "{b:?}");
        assert_eq!(b.linked.len(), 2, "w.py is listed once: {b:?}");
        assert!(b.destroys_substrate, "{b:?}");
    }

    /// G1: a planted "python3" literal must not hide a shell file's lines.
    #[tokio::test]
    async fn a_planted_runner_literal_does_not_hide_the_files_own_language() {
        let (d, p) = ws("g1-planted");
        put(
            &d,
            "main.py",
            "X = \"python3\"\nimport subprocess\nsubprocess.run([\"./w.sh\"])\n",
        );
        put(&d, "w.sh", "#!/bin/bash\nsource ./lib.sh\n");
        put(&d, "lib.sh", MENTION);
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.linked.iter().any(|l| l.ends_with("lib.sh")), "{b:?}");
        assert!(b.needs_judge());
    }

    /// L3: a flag that takes a value keeps the runner for the next word.
    #[tokio::test]
    async fn a_value_flag_does_not_drop_the_runner() {
        for (name, cmd) in [
            ("l3-py-x", "python3 -X utf8 notes.txt"),
            ("l3-py-w", "python3 -W ignore notes.txt"),
            ("l3-sh-o", "bash -o pipefail notes.txt"),
        ] {
            let (d, p) = ws(name);
            put(&d, "main.py", &format!("import os\nos.system(\"{cmd}\")\n"));
            put(&d, "notes.txt", MENTION);
            let b = body(&d, &p, "python3 ./main.py").await;
            assert_eq!(b.linked.len(), 1, "{cmd}: {b:?}");
            assert!(b.needs_judge(), "{cmd}");
        }
    }

    /// F1 negative: a data file named next to no interpreter stays unread.
    #[tokio::test]
    async fn a_data_file_not_after_an_interpreter_is_not_followed() {
        let (d, p) = ws("f1-data");
        put(&d, "main.py", "open(\"notes.txt\").read()\n");
        put(&d, "notes.txt", "plain data\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.linked.is_empty(), "{b:?}");
    }

    /// F2: backticks inside a string split like the other operators.
    #[tokio::test]
    async fn a_backticked_run_inside_a_string_is_followed() {
        let (d, p) = ws("f2-backtick");
        put(
            &d,
            "main.py",
            "import os\nos.system(\"echo `python3 w.py`\")\n",
        );
        put(&d, "w.py", &format!("import os\n{MENTION}"));
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(b.needs_judge());
    }

    #[test]
    fn runner_words_name_their_language() {
        assert_eq!(runner_lang("/usr/bin/python3"), Some(Lang::Python));
        assert_eq!(runner_lang("bash"), Some(Lang::Shell));
        assert_eq!(runner_lang("."), None);
        assert_eq!(runner_lang("source"), None);
        assert_eq!(runner_lang("`python3"), None);
    }

    #[tokio::test]
    async fn a_hard_write_in_a_linked_file_sets_the_hard_flag() {
        let (d, p) = ws("linked-hard");
        put(&d, "run.sh", "#!/bin/bash\n./inner.sh\n");
        put(&d, "inner.sh", "rm -rf ~/.openfang/agents/\n");
        let b = body(&d, &p, "bash ./run.sh").await;
        assert!(b.destroys_substrate, "{b:?}");
    }

    /// R2: `from pkg import sub` executes `pkg/sub.py`.
    #[tokio::test]
    async fn from_package_import_submodule_follows_the_submodule() {
        let (d, p) = ws("r2-pkg");
        put(&d, "main.py", "from pkg import sub\n");
        put(&d, "pkg/__init__.py", "");
        put(&d, "pkg/sub.py", MENTION);
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 2, "{b:?}");
        assert!(b.needs_judge());
    }

    /// R2: a namespace package (no `__init__.py`) is still local.
    #[tokio::test]
    async fn a_namespace_package_submodule_is_followed() {
        let (d, p) = ws("r2-ns");
        put(&d, "main.py", "import ns.mod\n");
        put(&d, "ns/mod.py", MENTION);
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.needs_judge(), "{b:?}");
    }

    /// R2: absolute imports inside a module resolve against the entry's dir.
    #[tokio::test]
    async fn absolute_imports_in_a_module_resolve_against_the_entry_dir() {
        let (d, p) = ws("r2-root");
        put(&d, "main.py", "from pkg import a\n");
        put(&d, "pkg/__init__.py", "");
        put(&d, "pkg/a.py", "import top\n");
        put(&d, "top.py", MENTION);
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.linked.iter().any(|l| l.ends_with("/top.py")), "{b:?}");
        assert!(b.needs_judge());
    }

    /// R2: compile helper.py to helper.pyc, delete the source.
    #[tokio::test]
    async fn a_pyc_only_module_is_blind() {
        let (d, p) = ws("r2-pyc");
        put(&d, "main.py", "import helper\n");
        put(&d, "helper.pyc", "\u{1}compiled");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(
            b.links_blind.as_deref().unwrap().contains("compiled"),
            "{b:?}"
        );
        assert!(b.needs_judge());
    }

    #[tokio::test]
    async fn an_extension_module_next_to_source_is_blind() {
        let (d, p) = ws("r2-so");
        put(&d, "main.py", "import helper\n");
        put(&d, "helper.py", "");
        put(&d, "helper.cpython-312-darwin.so", "x");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.links_blind.is_some(), "{b:?}");
    }

    #[tokio::test]
    async fn a_stdlib_import_is_not_local() {
        let (d, p) = ws("r2-stdlib");
        put(&d, "main.py", "import os, json\nfrom pathlib import Path\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.linked.is_empty() && b.links_blind.is_none(), "{b:?}");
        assert!(!b.needs_judge());
    }

    /// R3: extensionless specifier and a directory with `index.js`.
    #[tokio::test]
    async fn extensionless_and_index_js_imports_are_followed() {
        let (d, p) = ws("r3-ext");
        put(
            &d,
            "main.mjs",
            "import { a } from './lib'\nconst b = require('./dir')\n",
        );
        put(&d, "lib.js", "export const a = 1\n");
        put(
            &d,
            "dir/index.js",
            "module.exports = '~/.openfang/config.toml'\n",
        );
        let b = body(&d, &p, "node ./main.mjs").await;
        assert_eq!(b.linked.len(), 2, "{b:?}");
        assert!(b.needs_judge());
    }

    #[tokio::test]
    async fn an_unmatched_relative_import_is_blind() {
        let (d, p) = ws("r3-missing");
        put(&d, "main.mjs", "import x from './nope'\n");
        let b = body(&d, &p, "node ./main.mjs").await;
        assert!(b.links_blind.is_some(), "{b:?}");
    }

    #[tokio::test]
    async fn a_native_addon_is_blind() {
        let (d, p) = ws("r3-node");
        put(&d, "main.js", "const a = require('./addon')\n");
        put(&d, "addon.node", "x");
        let b = body(&d, &p, "node ./main.js").await;
        assert!(
            b.links_blind.as_deref().unwrap().contains("native"),
            "{b:?}"
        );
    }

    /// The Kimiya shape: a small entry re-exporting a large clean sibling.
    #[tokio::test]
    async fn a_large_clean_reexported_module_is_scanned_not_blind() {
        let (d, p) = ws("r3-kimiya");
        put(
            &d,
            "install.mjs",
            "export { check } from './fixture-checks.mjs'\n",
        );
        let big = "export function check() { return 1 }\n".repeat(1000);
        assert!(big.len() as u64 > MAX_SCRIPT_BODY_BYTES);
        put(&d, "fixture-checks.mjs", &big);
        let b = body(&d, &p, "node ./install.mjs").await;
        assert_eq!(b.status, ScriptBodyStatus::Included);
        assert_eq!(b.linked.len(), 1);
        assert!(!b.needs_judge(), "{b:?}");
    }

    /// R4: `source` resolves against the cwd a literal `cd` set.
    #[tokio::test]
    async fn source_after_cd_resolves_against_the_new_cwd() {
        let (d, p) = ws("r4-cd");
        put(&d, "run.sh", "cd sub\nsource lib.sh\n");
        put(&d, "sub/lib.sh", "cat ~/.openfang/config.toml\n");
        put(&d, "lib.sh", "echo decoy\n");
        let b = body(&d, &p, "bash ./run.sh").await;
        assert!(b.linked.iter().any(|l| l.ends_with("sub/lib.sh")), "{b:?}");
        assert!(b.needs_judge());
    }

    #[tokio::test]
    async fn a_non_literal_source_is_blind() {
        let (d, p) = ws("r4-dynamic");
        put(&d, "run.sh", "source \"$DIR/lib.sh\"\n");
        let b = body(&d, &p, "bash ./run.sh").await;
        assert!(
            b.links_blind.as_deref().unwrap().contains("non-literal"),
            "{b:?}"
        );
    }

    #[tokio::test]
    async fn the_dirname_idiom_resolves_to_the_script_dir() {
        let (d, p) = ws("r4-dirname");
        put(&d, "s/run.sh", "cd \"$(dirname \"$0\")\"\n. ./lib.sh\n");
        put(&d, "s/lib.sh", "echo ok\n");
        let b = body(&d, &p, "bash s/run.sh").await;
        assert_eq!(b.linked.len(), 1, "{b:?}");
        assert!(!b.needs_judge(), "{b:?}");
    }

    /// R5c: a chain deeper than the cap goes to the judge, and the deepest
    /// file's write is never silently dropped.
    #[tokio::test]
    async fn a_chain_past_the_depth_cap_is_blind() {
        let (d, p) = ws("caps-depth");
        put(&d, "main.py", "import a\n");
        put(&d, "a.py", "import b\n");
        put(&d, "b.py", "import c\n");
        put(&d, "c.py", "import e\n");
        put(&d, "e.py", "open('/Users/x/.openfang/RULES.md', 'w')\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(
            b.links_blind.as_deref().unwrap().contains("deeper"),
            "{b:?}"
        );
        assert!(b.needs_judge());
    }

    /// R5b: a cycle is de-duplicated before any cap counts.
    #[tokio::test]
    async fn an_import_cycle_terminates_without_tripping_a_cap() {
        let (d, p) = ws("caps-cycle");
        put(&d, "main.py", "import a\n");
        put(&d, "a.py", "import b\n");
        put(&d, "b.py", "import a\nimport main\n");
        let b = body(&d, &p, "python3 ./main.py").await;
        assert_eq!(b.linked.len(), 2, "{b:?}");
        assert!(b.links_blind.is_none(), "{b:?}");
    }

    #[tokio::test]
    async fn too_many_linked_files_is_blind() {
        let (d, p) = ws("caps-count");
        let imports: String = (0..25).map(|i| format!("import m{i}\n")).collect();
        put(&d, "main.py", &imports);
        for i in 0..25 {
            put(&d, &format!("m{i}.py"), "");
        }
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(b.links_blind.as_deref().unwrap().contains("files"), "{b:?}");
    }

    /// Guardrail 2 holds for linked files: no read outside reach.
    #[tokio::test]
    async fn a_linked_file_outside_reach_is_blind_and_unread() {
        let (d, p) = ws("reach");
        let (outside, _) = ws("reach-outside");
        put(&outside, "w.py", MENTION);
        put(
            &d,
            "main.py",
            &format!(
                "import subprocess\nsubprocess.run(['python3', '{}/w.py'])\n",
                outside.display()
            ),
        );
        let b = body(&d, &p, "python3 ./main.py").await;
        assert!(
            b.links_blind.as_deref().unwrap().contains("outside-reach"),
            "{b:?}"
        );
        assert!(
            b.scan_hits.is_none(),
            "nothing from outside reach is excerpted"
        );
    }

    /// Latency guard: a script just under the scan cap must not stall the
    /// gate. Measured 2026-11-25 on a ~1 MiB body: 0.46 s in release (what
    /// the daemon runs), 4-5.3 s in debug, almost all of it the per-line
    /// shell predicates in `BodyScan::of`. The cost is linear, so the debug
    /// bound only has to catch a blow-up, not a 2x drift under load.
    #[tokio::test]
    async fn a_near_cap_scan_stays_fast() {
        let (d, p) = ws("cost");
        let line = "    out.append('<td class=\"cell\">' + str(i) + '</td>')  # row\n";
        let script = line.repeat((1024 * 1024 - 1024) / line.len());
        put(&d, "big.py", &script);
        let t = std::time::Instant::now();
        let b = body(&d, &p, "python3 ./big.py").await;
        let took = t.elapsed();
        eprintln!("near-cap scan took {took:?}");
        assert_eq!(b.status, ScriptBodyStatus::Scanned);
        let bound = if cfg!(debug_assertions) { 20 } else { 2 };
        assert!(took < std::time::Duration::from_secs(bound), "{took:?}");
    }

    #[test]
    fn python_import_forms_parse() {
        let got = python_imports(
            "import a.b as c, d\nfrom .x import (y,\n  z as w)\nfrom .. import q\n\
             m = importlib.import_module('dyn')\n",
        );
        assert_eq!(
            got,
            vec![
                PyImport::Absolute {
                    module: "a.b".into(),
                    names: vec![]
                },
                PyImport::Absolute {
                    module: "d".into(),
                    names: vec![]
                },
                PyImport::Relative {
                    dots: 1,
                    module: "x".into(),
                    names: vec!["y".into(), "z".into()]
                },
                PyImport::Relative {
                    dots: 2,
                    module: "".into(),
                    names: vec!["q".into()]
                },
                PyImport::Absolute {
                    module: "dyn".into(),
                    names: vec![]
                },
            ]
        );
    }
}
