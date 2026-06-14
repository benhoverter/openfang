# file_convert

`file_convert` renders a workspace file from one format to another by dispatching
over a data-driven recipe table. The conversion runs at daemon privilege behind a
hardened security seam, so an agent can produce (for example) a PDF **without**
holding `shell_exec` and without routing through a privileged helper agent.

The v1 (MVP) recipe table ships exactly one row — **`md` -> `pdf`**, wrapping
`scripts/build-pdf.sh` (pandoc + typst). New formats are added as manifest rows,
with **zero Rust changes**.

## Tool surface

```
file_convert(format, input, output?)
```

| Param    | Required | Meaning |
|----------|----------|---------|
| `format` | yes      | Target format (e.g. `"pdf"`). Must match a recipe's `to`. |
| `input`  | yes      | Workspace-relative path to the source file. Its extension selects the recipe's `from`. |
| `output` | no       | Workspace-relative output path. Defaults to the input path with the recipe's `out_ext`. |

### Result envelope

Success and conversion failures are returned as a structured JSON envelope:

```json
{ "ok": true, "format": "pdf", "output_path": "out/report.pdf" }
```

```json
{
  "ok": false,
  "format": "pdf",
  "error": {
    "code": "MISSING_DEP",
    "message": "file_convert md->pdf needs 'typst', not found on PATH. Recipe present, binary missing."
  }
}
```

Error codes: `UNKNOWN_FORMAT`, `BAD_PATH`, `MISSING_DEP`, `CONVERT_FAILED`.
(Genuinely malformed calls — e.g. a missing `format`/`input` argument — surface
as a plain tool error, not an envelope.)

## Granting the tool

`file_convert` is a normal grantable capability. Add it to an agent's allowed
tools in its `agent.toml`:

```toml
[capabilities]
tools = [
  # ... existing tools ...
  "file_convert",
]
```

An agent that holds the grant sees `file_convert` advertised; one that does not,
will not. The tool is deliberately **not** a shell tool, so granting it does not
grant shell access, and it cannot bypass the approval gate via `exec_policy`.

## Adding a format (no Rust)

Conversions are data. To add one, drop a recipe row into the manifest at
`<openfang_home>/convert/recipes.toml` (`openfang_home` = `$OPENFANG_HOME`, else
`~/.openfang`):

```toml
[[recipe]]
from    = "html"
to      = "pdf"
argv    = ["{script}/html2pdf.sh", "{input}", "-o", "{output}"]
needs   = ["wkhtmltopdf"]
out_ext = "pdf"
```

Tokens, substituted at dispatch time (never by a shell):

| Token      | Becomes |
|------------|---------|
| `{script}` | `<openfang_home>/scripts` (absolute) |
| `{input}`  | sandbox-resolved absolute input path |
| `{output}` | sandbox-resolved absolute output path |

`argv[0]` is the program; every other entry is exactly one literal argument.
There is no shell parsing, globbing, or word-splitting.

Manifest rules:

- **Absent manifest** -> the compiled-in default table (the single `md -> pdf`
  row) is used, so a fresh install converts with no config.
- **Present manifest** -> it **replaces** the default entirely and is validated.
  A present-but-malformed manifest is a **hard error** (fail-closed) — never
  silently downgraded to the default.
- Each `(from, to)` pair must be unique; `argv` and `out_ext` are required;
  `needs` is optional.

## Security model (why this is a fix, not a hole)

1. **Allowlisted formats** — only `(from, to)` pairs present in the recipe table
   convert. Unknown pairs fail closed (`UNKNOWN_FORMAT`).
2. **Workspace-scoped paths** — both `input` and `output` are resolved through
   the workspace sandbox: `..`, absolute escapes, and symlink escapes are
   rejected (`BAD_PATH`).
3. **No shell string, ever** — the recipe `argv` is spawned as an argv array.
   Caller-supplied paths are substituted as literal arguments, so a filename
   like `evil; $(rm -rf ~)` reaches the program as one inert argument.
4. **Absolute launcher** — `argv[0]` must resolve to an absolute, existing file.
   The daemon's bare PATH is never trusted to find the launcher.
5. **Call-time preflight** — the launcher and every `needs` binary must resolve
   before the subprocess starts. A miss is `MISSING_DEP` with no partial run.
6. **Guaranteed PATH prefix** — the child gets a cleared environment plus a known
   tool-dir prefix on PATH, so a launcher's own `command -v` lookups resolve even
   under launchd's thin PATH.

## Retirement path (documented, NOT executed in this MVP)

Once `file_convert` is trusted in production, the existing PDF helpers collapse
onto it:

- **`pdf-attach` skill** — thin it to: call `file_convert(format="pdf", ...)`,
  then emit the `<openfang:attach>` tag for the produced path. The skill stops
  owning the pandoc/typst invocation.
- **`assistant-pdf-worker`** — retired *for this md->pdf use-case*; no-shell
  agents get PDFs directly via `file_convert` instead of routing a request to a
  privileged helper agent.
- **`scripts/build-pdf.sh` stays** — it is recipe row one. `file_convert` wraps
  it; it is not replaced.

This retirement is **doc only** for the MVP. Executing it (editing the skill,
decommissioning the worker) is a follow-up.

## Deferred: the other two dep-surfaces (STRETCH)

The dependency-surfacing model has three surfaces. v1 ships only the third:

1. **Provision-time loud line** in `deploy-local.sh`
   (`file_convert: recipe md->pdf DEGRADED — 'typst' not found`) — *deferred*.
2. **Registry introspection** (degraded / unavailable-with-reason; "what can you
   convert?") — *deferred*.
3. **Call-time backstop** (`MISSING_DEP` at invocation) — **shipped in v1.**

Surfaces 1 and 2 are ops ergonomics, not seam-critical. Tracked as a follow-up.
