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
/// time. Every value is a resolved, trusted path: `script_dir` is the runtime's
/// scripts directory and `input`/`output` are sandbox-validated workspace
/// paths. Substitution is the only way caller-influenced data enters the argv,
/// and the result is always spawned via an argv array (never a shell string),
/// so a path can never smuggle shell syntax such as `; rm -rf`.
pub struct ArgvTokens<'a> {
    /// Absolute path of `<openfang_home>/scripts`, substituted for `{script}`.
    pub script_dir: &'a Path,
    /// Sandbox-resolved absolute input path, substituted for `{input}`.
    pub input: &'a Path,
    /// Sandbox-resolved absolute output path, substituted for `{output}`.
    pub output: &'a Path,
    /// Selected preset's variables: token-name -> value. Empty when the
    /// recipe defines no presets. Values are manifest-authored and
    /// substituted with the same trust as argv literals; no caller string
    /// ever reaches this map.
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
                         plus this recipe's preset variables"
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
        validate_presets(r).map_err(|reason| invalid(reason))?;
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

/// Validate a recipe's preset table (the preset spec's load-time rules):
/// `default_preset` present-iff-`presets`, the default names a real preset, no
/// preset reuses a reserved token name, and every argv `{var}` beyond
/// `{script}`/`{input}`/`{output}` is defined by *every* preset (so no preset
/// can leave a token unsubstituted at dispatch time). Returns the failure
/// reason as a `String`, which the caller wraps into a `RecipeError::Invalid`.
fn validate_presets(r: &Recipe) -> Result<(), String> {
    const RESERVED: [&str; 3] = ["script", "input", "output"];

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
            if RESERVED.contains(&key.as_str()) {
                return Err(format!(
                    "recipe {}->{} preset {:?} uses reserved token name {:?}",
                    r.from, r.to, pname, key
                ));
            }
        }
    }

    for elem in &r.argv {
        let mut rest = elem.as_str();
        while let Some(start) = rest.find('{') {
            let end = match rest[start..].find('}') {
                Some(i) => start + i,
                None => break,
            };
            let name = &rest[start + 1..end];
            if !RESERVED.contains(&name) {
                for (pname, vars) in &r.presets {
                    if !vars.contains_key(name) {
                        return Err(format!(
                            "recipe {}->{} argv token {{{}}} is not defined by preset {:?}",
                            r.from, r.to, name, pname
                        ));
                    }
                }
            }
            rest = &rest[end + 1..];
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

    #[test]
    fn resolve_argv_substitutes_known_tokens() {
        let recipe = &default_recipes()[0];
        let argv = recipe
            .resolve_argv(&ArgvTokens {
                script_dir: Path::new("/home/.openfang/scripts"),
                input: Path::new("/ws/doc.md"),
                output: Path::new("/ws/doc.pdf"),
                vars: &BTreeMap::new(),
            })
            .unwrap();
        assert_eq!(argv[0], "/home/.openfang/scripts/build-pdf.sh");
        assert!(argv.contains(&"/ws/doc.md".to_string()));
        assert!(argv.contains(&"/ws/doc.pdf".to_string()));
        // No template tokens survive substitution.
        assert!(!argv.iter().any(|a| a.contains('{')));
    }

    #[test]
    fn resolve_argv_rejects_unknown_token() {
        let recipe = Recipe {
            from: "md".into(),
            to: "pdf".into(),
            argv: vec!["{script}/x".into(), "{evil}".into()],
            needs: vec![],
            out_ext: "pdf".into(),
            presets: BTreeMap::new(),
            default_preset: None,
        };
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
        let recipe = Recipe {
            from: "md".into(),
            to: "pdf".into(),
            argv: vec!["{input".into()],
            needs: vec![],
            out_ext: "pdf".into(),
            presets: BTreeMap::new(),
            default_preset: None,
        };
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
    }
}

