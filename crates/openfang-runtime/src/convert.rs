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
use std::collections::HashSet;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_manifest(home: &Path, body: &str) -> PathBuf {
        let path = recipes_path(home);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, body).unwrap();
        path
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
}
