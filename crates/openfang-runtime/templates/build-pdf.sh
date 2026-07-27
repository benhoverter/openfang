#!/usr/bin/env bash
#
# build-pdf.sh — Render a markdown doc to a PDF via pandoc + typst.
#
# Why this exists: the platform-eval docs are table-heavy, and two bugs bit us
# repeatedly when rendering them to PDF by hand:
#   1. Portrait wastes horizontal room and crushes wide inventory tables.
#   2. Pandoc wraps every table in a typst #figure, and typst figures are
#      NON-breakable by default. A table taller than the page overflows the
#      bottom margin and the NEXT page's text paints on top of it -- the
#      "layered text at the bottom of page 3" Ben hit on the component plan.
#
# Both fixes live in a header-include injected with `pandoc -H`:
#   #set page(flipped: <bool>, margin: <m>)     -> landscape + margins
#   #show figure: set block(breakable: true)    -> tall tables span pages
#
# `flipped` is not a parameter of pandoc's conf() wrapper, so it survives the
# template's own `set page` (which only touches paper + margin). That's the
# trick that makes header-injected landscape stick. Verified by render.
#
# ANAI-131: value-taking option flags. Every flag below is OPTIONAL and treats
# an empty value as "no override", so a caller passing no options gets
# byte-identical output to the pre-ANAI-131 script. The file_convert recipe
# passes these flags from recipe-declared, caller-supplied options.
#
# Usage:
#   build-pdf.sh INPUT.md [-o OUTPUT.pdf]
#                [--orientation portrait|landscape] [--portrait] [--landscape]
#                [--margin 1in] [--paper us-letter]
#                [--font-body FAMILY] [--font-header FAMILY] [--font-mono FAMILY]
#                [--font-size 11pt] [--highlight-style tango]
#                [--embed-resources true|false]
#                [--table-align left|center|right] [--table-justify true|false]
#
# Defaults: landscape, 1in margin, us-letter, OUTPUT = INPUT with .pdf suffix.
# (The file_convert md->pdf recipe overrides orientation to portrait via
#  --orientation, per ANAI-131 decision O1.)
#
# Examples:
#   build-pdf.sh out/2026-06-12-memory-system-component-plan.md
#   build-pdf.sh notes.md -o /tmp/notes.pdf --orientation portrait --margin 0.9in
#   build-pdf.sh notes.md --font-body "Libertinus Serif" --font-size 11pt \
#                --highlight-style tango --embed-resources true
#
set -euo pipefail

ORIENTATION="landscape"
MARGIN="1in"
PAPER="us-letter"
FONT_BODY=""
FONT_HEADER=""
FONT_MONO=""
FONT_SIZE=""
HIGHLIGHT_STYLE=""
EMBED_RESOURCES=""
TABLE_ALIGN=""
TABLE_JUSTIFY=""
INPUT=""
OUTPUT=""

usage() { sed -n '2,49p' "$0"; exit "${1:-0}"; }

# Normalise a boolean-ish flag value to "true"/"false"/"" (empty = no override).
norm_bool() {
  case "${1,,}" in
    true|1|yes|on)   echo "true";;
    false|0|no|off)  echo "false";;
    "")              echo "";;
    *)               echo "build-pdf: invalid boolean value: $1" >&2; exit 1;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)      OUTPUT="${2:?-o needs a path}"; shift 2;;
    --orientation)    ORIENTATION="${2:?--orientation needs a value}"; shift 2;;
    --portrait)       ORIENTATION="portrait"; shift;;
    --landscape)      ORIENTATION="landscape"; shift;;
    --margin)         MARGIN="${2:?--margin needs a value}"; shift 2;;
    --paper)          PAPER="${2:?--paper needs a value}"; shift 2;;
    --font-body)      FONT_BODY="${2-}"; shift 2;;
    --font-header)    FONT_HEADER="${2-}"; shift 2;;
    --font-mono)      FONT_MONO="${2-}"; shift 2;;
    --font-size)      FONT_SIZE="${2-}"; shift 2;;
    --highlight-style) HIGHLIGHT_STYLE="${2-}"; shift 2;;
    --embed-resources) EMBED_RESOURCES="$(norm_bool "${2-}")"; shift 2;;
    --table-align)    TABLE_ALIGN="${2-}"; shift 2;;
    --table-justify)  TABLE_JUSTIFY="$(norm_bool "${2-}")"; shift 2;;
    -h|--help)        usage 0;;
    -*)               echo "build-pdf: unknown option: $1" >&2; usage 1;;
    *)                INPUT="$1"; shift;;
  esac
done

# Empty margin/paper (a caller passing the empty-string default) falls back to
# the built-in defaults rather than emitting an empty typst value.
[[ -n "$MARGIN" ]] || MARGIN="1in"
[[ -n "$PAPER" ]]  || PAPER="us-letter"

case "$ORIENTATION" in
  portrait|landscape) ;;
  "") ORIENTATION="landscape";;
  *)  echo "build-pdf: invalid orientation: $ORIENTATION" >&2; exit 1;;
esac

case "$TABLE_ALIGN" in
  ""|left|center|right) ;;
  *) echo "build-pdf: invalid table-align: $TABLE_ALIGN" >&2; exit 1;;
esac

[[ -n "$INPUT" ]] || { echo "build-pdf: no input markdown given" >&2; usage 1; }
[[ -f "$INPUT" ]] || { echo "build-pdf: input not found: $INPUT" >&2; exit 1; }
[[ -n "$OUTPUT" ]] || OUTPUT="${INPUT%.md}.pdf"

command -v pandoc >/dev/null || { echo "build-pdf: pandoc not on PATH" >&2; exit 1; }
command -v typst  >/dev/null || { echo "build-pdf: typst not on PATH (--pdf-engine=typst needs it)" >&2; exit 1; }

FLIPPED="true"
[[ "$ORIENTATION" == "portrait" ]] && FLIPPED="false"

HEADER="$(mktemp -t build-pdf-header.XXXXXX)"
trap 'rm -f "$HEADER"' EXIT

# Base header include: orientation + margin + breakable figures (the original
# two fixes). New typst directives are appended below, each gated on a non-empty
# option value so an omitted option is a true no-op.
cat > "$HEADER" <<EOF
// Injected via \`pandoc -H\`. Runs before the template's conf(); \`flipped\`
// is not a conf parameter, so it survives conf's #set page.
#set page(flipped: ${FLIPPED}, margin: ${MARGIN})
#show figure: set block(breakable: true)
EOF

# Fonts (ANAI-131). Each is a single typst directive; empty value = skipped.
if [[ -n "$FONT_BODY" ]]; then
  printf '#set text(font: "%s")\n' "$FONT_BODY" >> "$HEADER"
fi
if [[ -n "$FONT_SIZE" ]]; then
  printf '#set text(size: %s)\n' "$FONT_SIZE" >> "$HEADER"
fi
if [[ -n "$FONT_HEADER" ]]; then
  printf '#show heading: set text(font: "%s")\n' "$FONT_HEADER" >> "$HEADER"
fi
if [[ -n "$FONT_MONO" ]]; then
  printf '#show raw: set text(font: "%s")\n' "$FONT_MONO" >> "$HEADER"
fi

# Table cell alignment / justification (ANAI-131, best-effort typst show-rules).
if [[ -n "$TABLE_ALIGN" ]]; then
  printf '#show table.cell: set align(%s)\n' "$TABLE_ALIGN" >> "$HEADER"
fi
if [[ "$TABLE_JUSTIFY" == "true" ]]; then
  printf '#show table.cell: set par(justify: true)\n' >> "$HEADER"
fi

# Assemble pandoc argv. Conditionally-added args live in a bash array so an
# omitted option contributes nothing (no empty argument reaches pandoc).
PANDOC_ARGS=( "$INPUT" --pdf-engine=typst -V papersize="$PAPER" -H "$HEADER" -o "$OUTPUT" )

if [[ -n "$HIGHLIGHT_STYLE" ]]; then
  PANDOC_ARGS+=( "--highlight-style=$HIGHLIGHT_STYLE" )
fi

if [[ "$EMBED_RESOURCES" == "true" ]]; then
  # Self-contained output: bundle referenced images. resource-path anchors
  # relative image refs at the input file's directory.
  INPUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
  PANDOC_ARGS+=( --embed-resources --standalone "--resource-path=$INPUT_DIR" )
fi

pandoc "${PANDOC_ARGS[@]}"

echo "build-pdf: wrote $OUTPUT (${ORIENTATION}, ${PAPER}, margin ${MARGIN})"
