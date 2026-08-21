//! `file_convert` recipe manifest: the data-driven format-conversion table.
//!
//! A *recipe* maps a `(from, to)` format pair to an allowlisted argv template.
//! Recipes load from `<openfang_home>/convert/recipes.toml`. When that file is
//! absent, the compiled-in default table (one `md` -> `pdf` row, embedded from
//! `templates/recipes.default.toml`) is used, so a fresh install can convert
//! without any config present. A present-but-malformed manifest is a hard error
//! (fail-closed): it is never silently downgraded to the default.
//!
//! Scope: this module is the manifest schema + loader only (ANAI-68). Token
//! substitution, path/argv security, preflight, and dispatch wiring live in
//! later subissues (ANAI-69/70/71) and intentionally do NOT appear here.
//!
//! ANAI-131 adds recipe-declared *options*: caller-suppliable, typed render
//! knobs (`[recipe.options.*]`). Unlike presets — whose values are wholly
//! manifest-authored — an option's VALUE is supplied by the tool caller and
//! validated (type/enum) at dispatch before it is substituted into argv. The
//! loader here validates the option *declarations* and projects them into the
//! tool schema (see [`project_options_schema`]) so the advertised surface is
//! derived from, and cannot drift from, what the converter accepts.

use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Location of the recipe manifest, relative to the OpenFang home directory.
pub const RECIPES_REL_PATH: &str = "convert/recipes.toml";

/// The canonical default manifest, embedded at compile time. This is the single
/// source of truth for the built-in recipe table: it is parsed at load time
/// when no external manifest exists, and a unit test asserts it stays in sync
/// with [`default_recipes`].
pub const DEFAULT_RECIPES_TOML: &str = include_str!("../templates/recipes.default.toml");

/// A single conversion recipe: one `from` -> `to` format pair.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// Source format, matched (case-insensitively) against the input file's
    /// extension.
    pub from: String,
    /// Target format.
    pub to: String,
    /// Argv template. Tokens `{script}`, `{input}`, and `{output}` are
    /// substituted positionally at dispatch time. This is NEVER a shell string:
    /// `argv[0]` is the program and the remaining entries are literal arguments,
    /// with no shell parsing, globbing, or word-splitting.
    pub argv: Vec<String>,
    /// External binaries the recipe needs available, surfaced at preflight.
    #[serde(default)]
    pub needs: Vec<String>,
    /// Extension of the produced output file (no leading dot).
    pub out_ext: String,
    /// Named render presets: preset-name -> { token-name -> value }. Values
    /// are manifest-authored and substituted into argv via `{token}`s
    /// alongside `{script}`/`{input}`/`{output}`. Empty for recipes that
    /// take no presets.
    #[serde(default)]
    pub presets: BTreeMap<String, BTreeMap<String, String>>,
    /// Preset used when the caller omits `preset`. REQUIRED iff `presets` is
    /// non-empty (validated at load time); MUST be `None` when `presets` is
    /// empty. Names a key in `presets`.
    #[serde(default)]
    pub default_preset: Option<String>,
    /// Caller-suppliable render options: option-name -> typed declaration
    /// (ANAI-131). Empty for recipes that accept no options. Distinct from
    /// `presets`: option VALUES are supplied by the tool caller and are
    /// type/enum-validated at dispatch before reaching argv, whereas preset
    /// values are wholly manifest-authored. Option and preset names share a
    /// single substitution namespace, so a name may not appear in both (guarded
    /// at load time).
    #[serde(default)]
    pub options: BTreeMap<String, OptionDecl>,
}

/// The type of a recipe option's value (ANAI-131). `enum` constrains the value
/// to a fixed `values` list; `string` accepts any free-form string (still
/// substituted as a single literal argv element, never re-parsed by a shell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionKind {
    Enum,
    String,
}

/// A single caller-suppliable option declaration on a recipe (ANAI-131).
///
/// The `default` is applied when the caller omits the option, which guarantees
/// every option `{token}` in the recipe's argv resolves at dispatch time (the
/// substitution engine builds a fixed-length argv array — there is no
/// conditional inclusion of elements, so an "unset" option must still carry a
/// meaningful value; an empty string is the conventional "no override").
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionDecl {
    /// `enum` or `string`.
    #[serde(rename = "type")]
    pub kind: OptionKind,
    /// Allowed values. REQUIRED and non-empty iff `kind == Enum`; MUST be empty
    /// for `kind == String`. Validated at load time.
    #[serde(default)]
    pub values: Vec<String>,
    /// Value applied when the caller omits this option. For an enum option this
    /// MUST be one of `values`. Required so every option token resolves.
    pub default: String,
    /// One-line human/agent description, surfaced verbatim in the projected
    /// tool schema so callers discover what the option does.
    pub desc: String,
}

/// TOML envelope: the manifest is an array-of-tables under `[[recipe]]`.
#[derive(Debug, Clone, Deserialize)]
struct RecipesFile {
    #[serde(default)]
    recipe: Vec<Recipe>,
}

/// A validated set of recipes supporting `(from, to)` lookup.
#[derive(Debug, Clone)]
pub struct RecipeSet {
    recipes: Vec<Recipe>,
}

impl RecipeSet {
    /// Wrap a pre-validated list of recipes.
    pub fn new(recipes: Vec<Recipe>) -> Self {
        Self { recipes }
    }

    /// Look up a recipe by `(from, to)`, case-insensitive on both tokens.
    /// Returns `None` for any unknown pair; callers MUST treat that as a hard
    /// error (fail-closed), never as a reason to fall back to another recipe.
    pub fn lookup(&self, from: &str, to: &str) -> Option<&Recipe> {
        self.recipes
            .iter()
            .find(|r| r.from.eq_ignore_ascii_case(from) && r.to.eq_ignore_ascii_case(to))
    }

    /// All recipes in the set.
    pub fn recipes(&self) -> &[Recipe] {
        &self.recipes
    }
}

/// The set of `{token}` values substituted into a recipe's argv at dispatch
/// time. `script_dir` is the runtime's scripts directory and `input`/`output`
/// are sandbox-validated workspace paths.
///
/// `vars` carries the recipe's preset variables AND (ANAI-131) its resolved
/// option values. Preset values are manifest-authored; option values are
/// caller-supplied but type/enum-validated by the dispatcher before they reach
/// this map. Either way, substitution is the only path caller-influenced data
/// takes into argv, and the result is ALWAYS spawned via an argv array (never a
/// shell string) — so a value can never smuggle shell syntax such as `; rm -rf`
/// (it lands as one literal argv element).
pub struct ArgvTokens<'a> {
    /// Absolute path of `<openfang_home>/scripts`, substituted for `{script}`.
    pub script_dir: &'a Path,
    /// Sandbox-resolved absolute input path, substituted for `{input}`.
    pub input: &'a Path,
    /// Sandbox-resolved absolute output path, substituted for `{output}`.
    pub output: &'a Path,
    /// Substitution variables: preset vars merged with resolved option values
    /// (token-name -> value). Empty when the recipe defines neither. Preset
    /// values are manifest-authored; option values are caller-supplied and
    /// validated (type/enum) before reaching this map. Both substitute with the
    /// same argv-literal trust — no value is ever re-parsed by a shell.
    pub vars: &'a BTreeMap<String, String>,
}

impl Recipe {
    /// Substitute `{script}`, `{input}`, and `{output}` into this recipe's argv
    /// template, producing a concrete argv ready to spawn.
    ///
    /// The template is validated first: every `{...}` placeholder must be one of
    /// the three known tokens, so a typo'd or hostile manifest token (e.g.
    /// `{home}`) is a hard error rather than a silently-unsubstituted literal.
    /// Substitution itself is plain replacement of the known tokens only.
    pub fn resolve_argv(&self, tokens: &ArgvTokens) -> Result<Vec<String>, String> {
        const KNOWN: [&str; 3] = ["{script}", "{input}", "{output}"];
        if self.argv.is_empty() {
            return Err("recipe argv is empty".to_string());
        }
        let script = tokens.script_dir.to_string_lossy();
        let input = tokens.input.to_string_lossy();
        let output = tokens.output.to_string_lossy();

        let mut out = Vec::with_capacity(self.argv.len());
        for elem in &self.argv {
            // Validate the template element: reject any unknown `{...}` token.
            let mut rest = elem.as_str();
            while let Some(start) = rest.find('{') {
                let end = rest[start..].find('}').map(|i| start + i).ok_or_else(|| {
                    format!("recipe argv element {elem:?} has an unterminated '{{' token")
                })?;
                let tok = &rest[start..=end];
                let name = &rest[start + 1..end];
                if !KNOWN.contains(&tok) && !tokens.vars.contains_key(name) {
                    return Err(format!(
                        "recipe argv element {elem:?} contains unknown token {tok:?}; \
                         supported tokens are {{script}}, {{input}}, {{output}}, \
                         plus this recipe's preset variables and options"
                    ));
                }
                rest = &rest[end + 1..];
            }
            let mut resolved = elem
                .replace("{script}", &script)
                .replace("{input}", &input)
                .replace("{output}", &output);
            for (vname, vval) in tokens.vars {
                resolved = resolved.replace(&format!("{{{vname}}}"), vval);
            }
            out.push(resolved);
        }
        Ok(out)
    }
}

/// Render the `options` JSON-Schema property for the advertised `file_convert`
/// tool (ANAI-131), derived from the union of every recipe's declared options.
///
/// Called by BOTH tool-schema mirrors (the runtime `built_in_tools()` and,
/// through the dispatcher seam, the MCP bridge) so the advertised surface is a
/// projection of the live recipe set and cannot drift from what the dispatcher
/// accepts. Each declared option becomes a typed sub-property carrying its
/// `enum` (for enum options), `default` (when non-empty), and `desc`. The
/// property `description` also lists a prose catalog grouped by conversion pair,
/// for callers that read prose better than nested schema. `additionalProperties`
/// is `false`, so a client can reject unknown option keys before the call even
/// reaches the server-side fail-closed check.
pub fn project_options_schema(set: &RecipeSet) -> serde_json::Value {
    use serde_json::{Map, Value};

    let mut props: Map<String, Value> = Map::new();
    let mut catalog: Vec<String> = Vec::new();

    for recipe in set.recipes() {
        if recipe.options.is_empty() {
            continue;
        }
        let mut parts: Vec<String> = Vec::new();
        for (name, decl) in &recipe.options {
            let mut prop = Map::new();
            // Every option value substitutes as a string argv literal.
            prop.insert("type".to_string(), Value::String("string".to_string()));
            if matches!(decl.kind, OptionKind::Enum) {
                prop.insert(
                    "enum".to_string(),
                    Value::Array(decl.values.iter().cloned().map(Value::String).collect()),
                );
            }
            if !decl.default.is_empty() {
                prop.insert("default".to_string(), Value::String(decl.default.clone()));
            }
            prop.insert("description".to_string(), Value::String(decl.desc.clone()));
            // Union across recipes: first declaration of a name wins the
            // schema sub-property. Today the recipe set is single-row, so this
            // is unambiguous; identical names across future recipes are
            // expected to agree in type/enum.
            props.entry(name.clone()).or_insert(Value::Object(prop));

            // Prose catalog fragment for this option.
            let mut frag = name.clone();
            if matches!(decl.kind, OptionKind::Enum) {
                frag.push_str(&format!(" ({})", decl.values.join("|")));
            } else {
                frag.push_str(" (str)");
            }
            if !decl.default.is_empty() {
                frag.push_str(&format!(", def {}", decl.default));
            }
            parts.push(frag);
        }
        catalog.push(format!(
            "{}->{} accepts: {}",
            recipe.from,
            recipe.to,
            parts.join(", ")
        ));
    }

    let description = if catalog.is_empty() {
        "Per-format render options. The active recipe set declares no options.".to_string()
    } else {
        format!("Per-format render options. {}.", catalog.join(". "))
    };

    serde_json::json!({
        "type": "object",
        "description": description,
        "properties": Value::Object(props),
        "additionalProperties": false
    })
}

/// Canonical projection of the `file_convert` `options` sub-schema from the
/// live recipe manifest — the single source of truth both schema mirrors
/// render from: the runtime `built_in_tools()` directly, and the runtime-free
/// MCP bridge indirectly (the daemon computes this at handshake and ships it
/// across the IPC seam, so the bridge advertises the same option surface the
/// dispatcher accepts without importing this crate). On a recipe-load failure
/// this returns a permissive object rather than panicking: the dispatcher
/// still fail-closes on every option at call time, so keeping `file_convert`
/// advertised (options temporarily undiscoverable) preserves the core
/// md->pdf path even under a broken custom manifest.
pub fn file_convert_options_schema() -> serde_json::Value {
    match load_recipes(&openfang_home_dir()) {
        Ok(set) => project_options_schema(&set),
        Err(_) => serde_json::json!({
            "type": "object",
            "description": "Per-format render options. (Temporarily undiscoverable: the recipe manifest failed to load. Calls are still validated server-side.)",
            "additionalProperties": true
        }),
    }
}

/// Errors raised while loading the recipe manifest.
#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    /// The manifest exists but could not be read.
    #[error("failed to read recipe manifest at {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The manifest exists but is not valid TOML / does not match the schema.
    #[error("failed to parse recipe manifest at {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    /// The manifest parsed but failed semantic validation.
    #[error("invalid recipe manifest at {path}: {reason}")]
    Invalid { path: PathBuf, reason: String },
}

/// Resolve the manifest path under the given OpenFang home directory.
pub fn recipes_path(home_dir: &Path) -> PathBuf {
    home_dir.join(RECIPES_REL_PATH)
}

/// Resolve the OpenFang home directory used to locate the recipe manifest.
///
/// Priority: `OPENFANG_HOME` env var > `$HOME/.openfang` (or `%USERPROFILE%`).
/// Mirrors the canonical resolver in `openfang-types::config` (which is
/// private) using the same `HOME`/`USERPROFILE` convention the rest of the
/// runtime uses, so the convert tool finds the manifest the same way the rest
/// of OpenFang resolves its home — without pulling in a new dependency.
pub fn openfang_home_dir() -> PathBuf {
    if let Ok(home) = std::env::var("OPENFANG_HOME") {
        return PathBuf::from(home);
    }
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".openfang"))
        .unwrap_or_else(|_| PathBuf::from(".openfang"))
}

/// The compiled-in default recipe table, parsed from [`DEFAULT_RECIPES_TOML`].
///
/// Panics only if the embedded template is itself malformed, which a unit test
/// guards against — so this never panics in a shipped binary.
pub fn default_recipes() -> Vec<Recipe> {
    let parsed: RecipesFile = toml::from_str(DEFAULT_RECIPES_TOML)
        .expect("embedded default recipes.toml must be valid; covered by unit test");
    parsed.recipe
}

/// Load the recipe set for the given OpenFang home directory.
///
/// * Manifest absent  -> the compiled-in default table (fresh-install path).
/// * Manifest present  -> read, parsed, and validated. Any read, parse, or
///   validation failure is returned as a [`RecipeError`]; a present manifest is
///   never silently replaced by the default.
pub fn load_recipes(home_dir: &Path) -> Result<RecipeSet, RecipeError> {
    let path = recipes_path(home_dir);
    if !path.exists() {
        return Ok(RecipeSet::new(default_recipes()));
    }
    let raw = std::fs::read_to_string(&path).map_err(|source| RecipeError::Read {
        path: path.clone(),
        source,
    })?;
    let parsed: RecipesFile = toml::from_str(&raw).map_err(|source| RecipeError::Parse {
        path: path.clone(),
        source,
    })?;
    validate(&parsed.recipe, &path)?;
    Ok(RecipeSet::new(parsed.recipe))
}

/// Semantic validation applied to an external (present) manifest.
fn validate(recipes: &[Recipe], path: &Path) -> Result<(), RecipeError> {
    let invalid = |reason: String| RecipeError::Invalid {
        path: path.to_path_buf(),
        reason,
    };

    if recipes.is_empty() {
        return Err(invalid(
            "manifest contains no [[recipe]] entries".to_string(),
        ));
    }

    let mut seen: HashSet<(String, String)> = HashSet::new();
    for r in recipes {
        if r.from.trim().is_empty() || r.to.trim().is_empty() {
            return Err(invalid("a recipe has an empty `from` or `to`".to_string()));
        }
        if r.out_ext.trim().is_empty() {
            return Err(invalid(format!(
                "recipe {}->{} has an empty `out_ext`",
                r.from, r.to
            )));
        }
        if r.argv.is_empty() {
            return Err(invalid(format!(
                "recipe {}->{} has an empty `argv`",
                r.from, r.to
            )));
        }
        validate_presets(r).map_err(invalid)?;
        validate_options(r).map_err(invalid)?;
        validate_argv_tokens(r).map_err(invalid)?;
        let key = (r.from.to_ascii_lowercase(), r.to.to_ascii_lowercase());
        if !seen.insert(key) {
            return Err(invalid(format!(
                "duplicate recipe for {}->{}",
                r.from, r.to
            )));
        }
    }
    Ok(())
}

/// Reserved argv token names that never come from a preset or an option.
const RESERVED_TOKENS: [&str; 3] = ["script", "input", "output"];

/// Validate a recipe's preset table (the preset spec's load-time rules):
/// `default_preset` present-iff-`presets`, the default names a real preset, and
/// no preset reuses a reserved token name. (Argv `{token}` coverage is validated
/// separately in [`validate_argv_tokens`], which is option-aware.)
fn validate_presets(r: &Recipe) -> Result<(), String> {
    if r.presets.is_empty() {
        if r.default_preset.is_some() {
            return Err(format!(
                "recipe {}->{} sets `default_preset` but defines no presets",
                r.from, r.to
            ));
        }
        return Ok(());
    }

    match &r.default_preset {
        None => {
            return Err(format!(
                "recipe {}->{} defines presets but no `default_preset`",
                r.from, r.to
            ));
        }
        Some(d) if !r.presets.contains_key(d) => {
            return Err(format!(
                "recipe {}->{} `default_preset` = {:?} is not a defined preset",
                r.from, r.to, d
            ));
        }
        Some(_) => {}
    }

    for (pname, vars) in &r.presets {
        for key in vars.keys() {
            if RESERVED_TOKENS.contains(&key.as_str()) {
                return Err(format!(
                    "recipe {}->{} preset {:?} uses reserved token name {:?}",
                    r.from, r.to, pname, key
                ));
            }
        }
    }

    Ok(())
}

/// Validate a recipe's option declarations (ANAI-131):
/// * enum options declare a non-empty `values` list AND a `default` among them;
/// * string options declare NO `values`;
/// * an option name never collides with a reserved token (`script`/`input`/
///   `output`) nor with a preset var — both feed the same substitution map, so
///   a collision would be ambiguous.
///
/// Option VALUES are supplied by the caller and validated at dispatch; this
/// covers only the manifest-authored declarations.
fn validate_options(r: &Recipe) -> Result<(), String> {
    for (name, decl) in &r.options {
        if RESERVED_TOKENS.contains(&name.as_str()) {
            return Err(format!(
                "recipe {}->{} option {:?} uses reserved token name",
                r.from, r.to, name
            ));
        }
        for (pname, vars) in &r.presets {
            if vars.contains_key(name) {
                return Err(format!(
                    "recipe {}->{} option {:?} collides with a var in preset {:?}",
                    r.from, r.to, name, pname
                ));
            }
        }
        match decl.kind {
            OptionKind::Enum => {
                if decl.values.is_empty() {
                    return Err(format!(
                        "recipe {}->{} enum option {:?} declares no `values`",
                        r.from, r.to, name
                    ));
                }
                if !decl.values.contains(&decl.default) {
                    return Err(format!(
                        "recipe {}->{} enum option {:?} default {:?} is not among its values",
                        r.from, r.to, name, decl.default
                    ));
                }
            }
            OptionKind::String => {
                if !decl.values.is_empty() {
                    return Err(format!(
                        "recipe {}->{} string option {:?} must not declare `values`",
                        r.from, r.to, name
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Validate that every `{token}` in a recipe's argv resolves at dispatch time.
/// A token is acceptable when it is a reserved token (`script`/`input`/`output`),
/// a declared option, or — when the recipe defines presets — a var supplied by
/// EVERY preset (so no preset can leave a token unsubstituted). This is the
/// option-aware successor to the argv scan formerly inside `validate_presets`.
fn validate_argv_tokens(r: &Recipe) -> Result<(), String> {
    for elem in &r.argv {
        let mut rest = elem.as_str();
        while let Some(start) = rest.find('{') {
            let end = match rest[start..].find('}') {
                Some(i) => start + i,
                None => break,
            };
            let name = &rest[start + 1..end];
            rest = &rest[end + 1..];

            if RESERVED_TOKENS.contains(&name) || r.options.contains_key(name) {
                continue;
            }
            if r.presets.is_empty() {
                return Err(format!(
                    "recipe {}->{} argv token {{{}}} is not a reserved token, \
                     a declared option, or a preset var",
                    r.from, r.to, name
                ));
            }
            for (pname, vars) in &r.presets {
                if !vars.contains_key(name) {
                    return Err(format!(
                        "recipe {}->{} argv token {{{}}} is not defined by preset {:?}",
                        r.from, r.to, name, pname
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::TempDir;

    fn write_manifest(home: &Path, body: &str) -> PathBuf {
        let path = recipes_path(home);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, body).unwrap();
        path
    }

    fn recipe_with(
        argv: Vec<&str>,
        presets: BTreeMap<String, BTreeMap<String, String>>,
        default_preset: Option<&str>,
        options: BTreeMap<String, OptionDecl>,
    ) -> Recipe {
        Recipe {
            from: "md".into(),
            to: "pdf".into(),
            argv: argv.into_iter().map(String::from).collect(),
            needs: vec![],
            out_ext: "pdf".into(),
            presets,
            default_preset: default_preset.map(String::from),
            options,
        }
    }

    fn enum_opt(values: &[&str], default: &str) -> OptionDecl {
        OptionDecl {
            kind: OptionKind::Enum,
            values: values.iter().map(|s| s.to_string()).collect(),
            default: default.to_string(),
            desc: "test option".into(),
        }
    }

    fn string_opt(default: &str) -> OptionDecl {
        OptionDecl {
            kind: OptionKind::String,
            values: vec![],
            default: default.to_string(),
            desc: "test option".into(),
        }
    }

    #[test]
    fn resolve_argv_substitutes_known_tokens() {
        let recipe = &default_recipes()[0];
        // The default recipe now declares options (ANAI-131); the dispatcher
        // fills every option with its declared default before resolving. Mirror
        // that here so the fixed-length argv fully resolves.
        let mut vars = BTreeMap::new();
        for (name, decl) in &recipe.options {
            vars.insert(name.clone(), decl.default.clone());
        }
        let argv = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/home/.openfang/scripts"),
                input: Path::new("/ws/doc.md"),
                output: Path::new("/ws/doc.pdf"),
                vars: &vars,
            })
            .unwrap();
        assert_eq!(argv[0], "/home/.openfang/scripts/build-pdf.sh");
        assert!(argv.contains(&"/ws/doc.md".to_string()));
        assert!(argv.contains(&"/ws/doc.pdf".to_string()));
        // No template tokens survive substitution once option defaults fill in.
        assert!(!argv.iter().any(|a| a.contains('{')));
    }

    #[test]
    fn resolve_argv_rejects_unknown_token() {
        let recipe = recipe_with(
            vec!["{script}/x", "{evil}"],
            BTreeMap::new(),
            None,
            BTreeMap::new(),
        );
        let err = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/s"),
                input: Path::new("/i"),
                output: Path::new("/o"),
                vars: &BTreeMap::new(),
            })
            .unwrap_err();
        assert!(err.contains("unknown token"));
    }

    #[test]
    fn resolve_argv_rejects_unterminated_token() {
        let recipe = recipe_with(vec!["{input"], BTreeMap::new(), None, BTreeMap::new());
        let err = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/s"),
                input: Path::new("/i"),
                output: Path::new("/o"),
                vars: &BTreeMap::new(),
            })
            .unwrap_err();
        assert!(err.contains("unterminated"));
    }

    #[test]
    fn embedded_default_parses_and_matches_default_recipes() {
        // Guards the expect() in default_recipes() and keeps the embedded
        // template in sync with the typed expectation.
        let recipes = default_recipes();
        assert_eq!(recipes.len(), 1);
        let md = &recipes[0];
        assert_eq!(md.from, "md");
        assert_eq!(md.to, "pdf");
        assert_eq!(md.out_ext, "pdf");
        assert_eq!(md.needs, vec!["pandoc", "typst"]);
        assert_eq!(md.argv[0], "{script}/build-pdf.sh");
        assert!(md.argv.contains(&"{input}".to_string()));
        assert!(md.argv.contains(&"{output}".to_string()));
        // ANAI-131: the default md->pdf recipe now declares options with the
        // decisions Ben locked (orientation default portrait, embed_images true).
        assert!(md.options.contains_key("orientation"));
        assert_eq!(md.options["orientation"].default, "portrait");
        assert!(md.options.contains_key("embed_images"));
        assert_eq!(md.options["embed_images"].default, "true");
        // The embedded default must survive full semantic validation.
        let tmp = std::path::Path::new("/embedded/default/recipes.toml");
        validate(&recipes, tmp).expect("embedded default recipes must validate");
    }

    #[test]
    fn absent_manifest_yields_default_table() {
        let home = TempDir::new().unwrap();
        let set = load_recipes(home.path()).unwrap();
        assert_eq!(set.recipes(), default_recipes().as_slice());
        assert!(set.lookup("md", "pdf").is_some());
    }

    #[test]
    fn valid_manifest_loads_and_overrides_default() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "html"
                to = "pdf"
                argv = ["{script}/html2pdf.sh", "{input}", "{output}"]
                needs = ["wkhtmltopdf"]
                out_ext = "pdf"
            "#,
        );
        let set = load_recipes(home.path()).unwrap();
        assert_eq!(set.recipes().len(), 1);
        assert!(set.lookup("html", "pdf").is_some());
        // External manifest replaces the default, so md->pdf is gone.
        assert!(set.lookup("md", "pdf").is_none());
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let set = RecipeSet::new(default_recipes());
        assert!(set.lookup("MD", "PDF").is_some());
        assert!(set.lookup("Md", "Pdf").is_some());
    }

    #[test]
    fn unknown_pair_returns_none() {
        let set = RecipeSet::new(default_recipes());
        assert!(set.lookup("md", "docx").is_none());
        assert!(set.lookup("png", "pdf").is_none());
    }

    #[test]
    fn malformed_toml_is_parse_error_not_default() {
        let home = TempDir::new().unwrap();
        write_manifest(home.path(), "this is not = = valid toml [[[");
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Parse { .. }));
    }

    #[test]
    fn unknown_field_is_rejected() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["x"]
                out_ext = "pdf"
                surprise = "boom"
            "#,
        );
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Parse { .. }));
    }

    #[test]
    fn empty_manifest_is_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(home.path(), "# no recipes here\n");
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn duplicate_pair_is_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["a"]
                out_ext = "pdf"

                [[recipe]]
                from = "MD"
                to = "PDF"
                argv = ["b"]
                out_ext = "pdf"
            "#,
        );
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn empty_argv_is_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = []
                out_ext = "pdf"
            "#,
        );
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn resolve_argv_substitutes_preset_vars() {
        let mut vars = BTreeMap::new();
        vars.insert("viewport".to_string(), "390,844".to_string());
        vars.insert("scale".to_string(), "3".to_string());
        let recipe = Recipe {
            from: "html".into(),
            to: "png".into(),
            argv: vec![
                "{script}/html2png.sh".into(),
                "{input}".into(),
                "-o".into(),
                "{output}".into(),
                "--viewport".into(),
                "{viewport}".into(),
                "--scale".into(),
                "{scale}".into(),
            ],
            needs: vec![],
            out_ext: "png".into(),
            presets: BTreeMap::new(),
            default_preset: None,
            options: BTreeMap::new(),
        };
        let argv = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/s"),
                input: Path::new("/ws/a.html"),
                output: Path::new("/ws/a.png"),
                vars: &vars,
            })
            .unwrap();
        assert!(argv.contains(&"390,844".to_string()));
        assert!(argv.contains(&"3".to_string()));
        assert!(!argv.iter().any(|a| a.contains('{')));
    }

    #[test]
    fn resolve_argv_preset_value_is_literal_not_reparsed() {
        let mut vars = BTreeMap::new();
        vars.insert("vp".to_string(), "x; touch pwned".to_string());
        let recipe = Recipe {
            from: "html".into(),
            to: "png".into(),
            argv: vec!["{script}/r.sh".into(), "{vp}".into()],
            needs: vec![],
            out_ext: "png".into(),
            presets: BTreeMap::new(),
            default_preset: None,
            options: BTreeMap::new(),
        };
        let argv = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/s"),
                input: Path::new("/i"),
                output: Path::new("/o"),
                vars: &vars,
            })
            .unwrap();
        assert_eq!(argv[1], "x; touch pwned");
    }

    #[test]
    fn resolve_argv_option_value_is_literal_not_reparsed() {
        // ANAI-131: an option VALUE is the first caller-supplied string to
        // reach argv. It must land as a single literal element, never reparsed.
        let mut vars = BTreeMap::new();
        vars.insert("font_body".to_string(), "x; touch pwned".to_string());
        let recipe = recipe_with(
            vec!["{script}/build-pdf.sh", "--font-body", "{font_body}"],
            BTreeMap::new(),
            None,
            {
                let mut o = BTreeMap::new();
                o.insert("font_body".to_string(), string_opt(""));
                o
            },
        );
        let argv = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/s"),
                input: Path::new("/i"),
                output: Path::new("/o"),
                vars: &vars,
            })
            .unwrap();
        assert_eq!(argv[2], "x; touch pwned");
    }

    fn preset_manifest(default_line: &str, mobile_vp: &str) -> String {
        format!(
            r#"
                [[recipe]]
                from = "html"
                to = "png"
                argv = ["{{script}}/html2png.sh", "{{input}}", "{{output}}", "{{viewport}}"]
                out_ext = "png"
                {default_line}

                [recipe.presets.mobile]
                viewport = "{mobile_vp}"

                [recipe.presets.desktop]
                viewport = "1280,800"
            "#
        )
    }

    #[test]
    fn valid_preset_manifest_loads() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            &preset_manifest("default_preset = \"mobile\"", "390,844"),
        );
        let set = load_recipes(home.path()).unwrap();
        let r = set.lookup("html", "png").unwrap();
        assert_eq!(r.default_preset.as_deref(), Some("mobile"));
        assert_eq!(r.presets.len(), 2);
    }

    #[test]
    fn manifest_missing_default_preset_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(home.path(), &preset_manifest("", "390,844"));
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn manifest_default_preset_not_in_table_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            &preset_manifest("default_preset = \"phone\"", "390,844"),
        );
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn argv_var_missing_from_a_preset_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "html"
                to = "png"
                argv = ["{script}/h.sh", "{input}", "{output}", "{scale}"]
                out_ext = "png"
                default_preset = "mobile"

                [recipe.presets.mobile]
                viewport = "390,844"
            "#,
        );
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn default_preset_without_presets_invalid() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["{script}/build-pdf.sh", "{input}", "{output}"]
                out_ext = "pdf"
                default_preset = "mobile"
            "#,
        );
        let err = load_recipes(home.path()).unwrap_err();
        assert!(matches!(err, RecipeError::Invalid { .. }));
    }

    #[test]
    fn presetless_recipe_still_valid() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["{script}/build-pdf.sh", "{input}", "{output}"]
                needs = ["pandoc", "typst"]
                out_ext = "pdf"
            "#,
        );
        let set = load_recipes(home.path()).unwrap();
        let r = set.lookup("md", "pdf").unwrap();
        assert!(r.presets.is_empty());
        assert!(r.default_preset.is_none());
        assert!(r.options.is_empty());
    }

    // ------------------------------------------------------------------
    // ANAI-131: option declaration + projection tests
    // ------------------------------------------------------------------

    #[test]
    fn enum_option_requires_values_and_default_in_them() {
        // Missing values.
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["{script}/b.sh", "{input}", "{output}", "{orientation}"]
                out_ext = "pdf"

                [recipe.options.orientation]
                type = "enum"
                default = "portrait"
                desc = "Page orientation"
            "#,
        );
        assert!(matches!(
            load_recipes(home.path()).unwrap_err(),
            RecipeError::Invalid { .. }
        ));

        // Default not among values.
        let home2 = TempDir::new().unwrap();
        write_manifest(
            home2.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["{script}/b.sh", "{input}", "{output}", "{orientation}"]
                out_ext = "pdf"

                [recipe.options.orientation]
                type = "enum"
                values = ["portrait", "landscape"]
                default = "sideways"
                desc = "Page orientation"
            "#,
        );
        assert!(matches!(
            load_recipes(home2.path()).unwrap_err(),
            RecipeError::Invalid { .. }
        ));
    }

    #[test]
    fn string_option_must_not_declare_values() {
        let home = TempDir::new().unwrap();
        write_manifest(
            home.path(),
            r#"
                [[recipe]]
                from = "md"
                to = "pdf"
                argv = ["{script}/b.sh", "{input}", "{output}", "{font_body}"]
                out_ext = "pdf"

                [recipe.options.font_body]
                type = "string"
                values = ["nope"]
                default = ""
                desc = "Body font"
            "#,
        );
        assert!(matches!(
            load_recipes(home.path()).unwrap_err(),
            RecipeError::Invalid { .. }
        ));
    }

    #[test]
    fn option_name_colliding_with_reserved_token_invalid() {
        let r = recipe_with(
            vec!["{script}/b.sh", "{input}", "{output}"],
            BTreeMap::new(),
            None,
            {
                let mut o = BTreeMap::new();
                o.insert("input".to_string(), string_opt(""));
                o
            },
        );
        assert!(validate_options(&r).is_err());
    }

    #[test]
    fn option_name_colliding_with_preset_var_invalid() {
        let mut presets = BTreeMap::new();
        let mut mobile = BTreeMap::new();
        mobile.insert("viewport".to_string(), "390,844".to_string());
        presets.insert("mobile".to_string(), mobile);
        let mut options = BTreeMap::new();
        options.insert("viewport".to_string(), string_opt(""));
        let r = recipe_with(
            vec!["{script}/b.sh", "{input}", "{output}", "{viewport}"],
            presets,
            Some("mobile"),
            options,
        );
        assert!(validate_options(&r).is_err());
    }

    #[test]
    fn argv_token_referencing_undeclared_option_invalid() {
        let r = recipe_with(
            vec!["{script}/b.sh", "{input}", "{output}", "{ghost}"],
            BTreeMap::new(),
            None,
            BTreeMap::new(),
        );
        assert!(validate_argv_tokens(&r).is_err());
    }

    #[test]
    fn argv_token_referencing_declared_option_ok() {
        let mut options = BTreeMap::new();
        options.insert(
            "orientation".to_string(),
            enum_opt(&["portrait", "landscape"], "portrait"),
        );
        let r = recipe_with(
            vec!["{script}/b.sh", "{input}", "{output}", "{orientation}"],
            BTreeMap::new(),
            None,
            options,
        );
        assert!(validate_argv_tokens(&r).is_ok());
        assert!(validate_options(&r).is_ok());
    }

    #[test]
    fn project_options_schema_surfaces_enum_default_and_desc() {
        let mut options = BTreeMap::new();
        options.insert(
            "orientation".to_string(),
            OptionDecl {
                kind: OptionKind::Enum,
                values: vec!["portrait".into(), "landscape".into()],
                default: "portrait".into(),
                desc: "Page orientation".into(),
            },
        );
        options.insert("font_body".to_string(), string_opt(""));
        let r = recipe_with(
            vec![
                "{script}/b.sh",
                "{input}",
                "{output}",
                "{orientation}",
                "{font_body}",
            ],
            BTreeMap::new(),
            None,
            options,
        );
        let schema = project_options_schema(&RecipeSet::new(vec![r]));

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        // Catalog description is non-empty and names the pair.
        let desc = schema["description"].as_str().unwrap();
        assert!(desc.contains("md->pdf accepts"));
        assert!(desc.contains("orientation"));

        let props = &schema["properties"];
        assert_eq!(props["orientation"]["type"], "string");
        assert_eq!(props["orientation"]["default"], "portrait");
        assert_eq!(props["orientation"]["description"], "Page orientation");
        let en = props["orientation"]["enum"].as_array().unwrap();
        assert_eq!(en.len(), 2);
        assert!(en.iter().any(|v| v == "portrait"));
        // A string option carries no enum; empty default is omitted.
        assert_eq!(props["font_body"]["type"], "string");
        assert!(props["font_body"].get("enum").is_none());
        assert!(props["font_body"].get("default").is_none());
    }

    #[test]
    fn project_options_schema_empty_when_no_options() {
        let set = RecipeSet::new(vec![recipe_with(
            vec!["{script}/b.sh", "{input}", "{output}"],
            BTreeMap::new(),
            None,
            BTreeMap::new(),
        )]);
        let schema = project_options_schema(&set);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"].as_object().unwrap().len(), 0);
    }
}
