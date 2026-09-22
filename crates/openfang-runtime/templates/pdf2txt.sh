#!/usr/bin/env bash
# pdf2txt.sh -- extract a PDF's text layer to a plain .txt file.
#
# Invoked by file_convert's pdf->txt recipe (never a shell string): argv is an
# array, so paths and option values arrive as literal args with no shell
# parsing, globbing, or word-splitting.
#
#   pdf2txt.sh <input.pdf> -o <output.txt>
#              [--layout true|false] [--first-page N] [--last-page N]
#              [--min-chars N]
#
# WHY A LAUNCHER AND NOT A BARE `pdftotext` argv ROW
#
# `pdftotext` exits 0 on a PDF with no text layer -- a scanned or image-only
# document -- and leaves behind an EMPTY .txt. The conversion would report
# success, the caller would read a 0-byte file, and nothing anywhere would say
# "there was no text to extract." That is the silent-success class, and it is
# the whole reason this wrapper exists: it counts what came out and REFUSES
# rather than handing back a plausible-looking empty result.
#
# Exit codes (3 is reserved by file_convert for MISSING_DEP):
#   0  success
#   2  usage / bad input
#   3  pdftotext not found                       -> MISSING_DEP
#   4  pdftotext failed                          -> CONVERT_FAILED
#   5  pdftotext exited 0 but produced no file   -> CONVERT_FAILED
#   6  extracted fewer than --min-chars chars    -> CONVERT_FAILED
set -euo pipefail

PDFTOTEXT="${OPENFANG_PDFTOTEXT:-pdftotext}"

INPUT=""
OUTPUT=""
LAYOUT="true"
FIRST_PAGE=""
LAST_PAGE=""
MIN_CHARS="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output)   OUTPUT="$2";     shift 2 ;;
    --layout)      LAYOUT="$2";     shift 2 ;;
    --first-page)  FIRST_PAGE="$2"; shift 2 ;;
    --last-page)   LAST_PAGE="$2";  shift 2 ;;
    --min-chars)   MIN_CHARS="$2";  shift 2 ;;
    -*)            echo "pdf2txt: unknown flag '$1'" >&2; exit 2 ;;
    *)             if [ -z "$INPUT" ]; then INPUT="$1"; else OUTPUT="$1"; fi; shift ;;
  esac
done

[ -n "$INPUT" ]  || { echo "pdf2txt: no input file given" >&2; exit 2; }
[ -n "$OUTPUT" ] || { echo "pdf2txt: no output (-o) given" >&2; exit 2; }
[ -f "$INPUT" ]  || { echo "pdf2txt: input not found: $INPUT" >&2; exit 2; }

# Validate the numeric option BEFORE spawning anything: a bad value should cost
# a refusal, not a full extraction followed by a refusal.
case "$MIN_CHARS" in
  ''|*[!0-9]*)
    echo "pdf2txt: --min-chars must be a non-negative integer, got '$MIN_CHARS'" >&2
    exit 2
    ;;
esac

command -v "$PDFTOTEXT" >/dev/null 2>&1 || {
  echo "pdf2txt: '$PDFTOTEXT' not found on PATH (poppler is not installed)" >&2
  exit 3
}

# An empty option value means "caller stated no preference" -- the recipe's
# argv is fixed-length, so every option is always passed and the empty string
# is how "no override" travels.
#
# Two shell footguns are deliberately avoided here:
#   * `[ -n "$X" ] && ARGS+=(...)` would make the whole AND-list return 1 when
#     X is empty, and under `set -e` that EXITS the script. if/then instead.
#   * `"${ARGS[@]}"` on an EMPTY array is an unbound-variable error under
#     `set -u` in bash 3.2, which is what /usr/bin/env bash still resolves to
#     on a stock macOS PATH. `${ARGS[@]+"${ARGS[@]}"}` expands to nothing at
#     all when the array is empty, which is the case reached by
#     `--layout false` with no page bounds.
ARGS=()
case "$LAYOUT" in
  true|"") ARGS+=("-layout") ;;
  false)   : ;;
  *)
    echo "pdf2txt: --layout must be true or false, got '$LAYOUT'" >&2
    exit 2
    ;;
esac
if [ -n "$FIRST_PAGE" ]; then ARGS+=("-f" "$FIRST_PAGE"); fi
if [ -n "$LAST_PAGE" ];  then ARGS+=("-l" "$LAST_PAGE");  fi

"$PDFTOTEXT" ${ARGS[@]+"${ARGS[@]}"} "$INPUT" "$OUTPUT" || {
  echo "pdf2txt: pdftotext failed on $INPUT" >&2
  exit 4
}

[ -f "$OUTPUT" ] || {
  echo "pdf2txt: pdftotext exited 0 but produced no output at $OUTPUT" >&2
  exit 5
}

# Count what actually came out. `wc -c` reads the file rather than slurping it
# into a variable, which matters for a large extraction.
CHARS="$(wc -c < "$OUTPUT" | tr -d '[:space:]')"
if [ "$CHARS" -lt "$MIN_CHARS" ]; then
  echo "pdf2txt: extracted only ${CHARS} characters (minimum ${MIN_CHARS})." >&2
  echo "pdf2txt: this PDF most likely has NO TEXT LAYER -- a scan or an" >&2
  echo "pdf2txt: image-only export. There is no OCR engine wired in, so there" >&2
  echo "pdf2txt: is nothing to extract. Refusing rather than returning an" >&2
  echo "pdf2txt: empty file that looks like a successful conversion." >&2
  rm -f "$OUTPUT"
  exit 6
fi

echo "pdf2txt: extracted ${CHARS} characters to ${OUTPUT}"
