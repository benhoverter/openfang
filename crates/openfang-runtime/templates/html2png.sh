#!/usr/bin/env bash
# html2png.sh -- render an HTML file to PNG via headless Google Chrome.
#
# Invoked by file_convert's html->png recipe (never a shell string): argv is an
# array, so paths/preset values arrive as literal args with no shell parsing.
#
#   html2png.sh <input.html> -o <output.png> [--viewport WxH|W,H] [--scale N]
#
# Chrome is pinned by absolute path (it is installed as a .app, NOT on PATH).
# Override with OPENFANG_CHROME to point at chromium on another host.
set -euo pipefail

CHROME="${OPENFANG_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

INPUT=""
OUTPUT=""
VIEWPORT="${VIEWPORT:-1280,800}"
SCALE="${SCALE:-2}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --viewport)  VIEWPORT="$2"; shift 2 ;;
    --scale)     SCALE="$2"; shift 2 ;;
    -*)          echo "html2png: unknown flag '$1'" >&2; exit 2 ;;
    *)           if [ -z "$INPUT" ]; then INPUT="$1"; else OUTPUT="$1"; fi; shift ;;
  esac
done

[ -n "$INPUT" ]  || { echo "html2png: no input file given" >&2; exit 2; }
[ -n "$OUTPUT" ] || { echo "html2png: no output (-o) given" >&2; exit 2; }
[ -f "$INPUT" ]  || { echo "html2png: input not found: $INPUT" >&2; exit 2; }
[ -x "$CHROME" ] || { echo "html2png: Chrome not found at $CHROME" >&2; exit 3; }

VIEWPORT="${VIEWPORT/x/,}"

case "$INPUT" in
  /*) ABS_INPUT="$INPUT" ;;
  *)  ABS_INPUT="$PWD/$INPUT" ;;
esac

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --hide-scrollbars \
  --no-sandbox \
  --force-device-scale-factor="$SCALE" \
  --window-size="$VIEWPORT" \
  --default-background-color=00000000 \
  --virtual-time-budget=2000 \
  --screenshot="$OUTPUT" \
  "file://$ABS_INPUT" >/dev/null 2>&1 || {
    echo "html2png: Chrome failed to render $INPUT" >&2
    exit 4
  }

[ -f "$OUTPUT" ] || { echo "html2png: no output produced at $OUTPUT" >&2; exit 5; }
