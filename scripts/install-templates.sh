#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# install-templates.sh — sync runtime script templates into ~/.openfang/scripts/
#
# The file_convert md->pdf recipe launches an absolute-pinned helper at
#   ~/.openfang/scripts/build-pdf.sh
# but nothing in the build/deploy pipeline keeps that copy in sync with the
# version-controlled source of truth at
#   crates/openfang-runtime/templates/*.sh
# so a rebuilt daemon on an old script rejects the new --orientation/--font-*/
# --highlight-style/--table-* flags and md->pdf breaks. [ANAI-131 follow-up]
#
# This installer closes that gap: idempotent, fail-closed, backup-on-change,
# exec-bit preserved. Safe to run repeatedly; a no-op when already current.
#
# Usage:
#   install-templates.sh [--dry-run]
#
# Options:
#   --dry-run   Report what WOULD change; touch nothing.
#
# Env overrides:
#   OPENFANG_REPO         repo root       (default: $HOME/GitHub/Repos/openfang)
#   OPENFANG_SCRIPTS_DIR  install target  (default: $HOME/.openfang/scripts)
#
# Exit codes:
#   0  success (all templates current, or synced)
#   1  source template dir missing / unreadable (fail-closed)
#   2  install target dir missing and not creatable
# ------------------------------------------------------------------------------
set -euo pipefail

REPO="${OPENFANG_REPO:-$HOME/GitHub/Repos/openfang}"
SRC_DIR="${REPO}/crates/openfang-runtime/templates"
DST_DIR="${OPENFANG_SCRIPTS_DIR:-$HOME/.openfang/scripts}"
DRY_RUN=0
STAMP="$(date +%Y%m%d-%H%M%S)"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_DIM=$'\033[2m'
else
  C_RESET=; C_BOLD=; C_GREEN=; C_YELLOW=; C_BLUE=; C_DIM=
fi

# --- Fail-closed: source dir must exist ---------------------------------------
if [[ ! -d "$SRC_DIR" ]]; then
  echo "${C_YELLOW}error:${C_RESET} template source dir not found: $SRC_DIR" >&2
  exit 1
fi

# --- Ensure target dir --------------------------------------------------------
if [[ ! -d "$DST_DIR" ]]; then
  if (( DRY_RUN )); then
    echo "${C_DIM}[dry-run] would create: $DST_DIR${C_RESET}"
  else
    mkdir -p "$DST_DIR" || { echo "error: cannot create $DST_DIR" >&2; exit 2; }
  fi
fi

echo "${C_BOLD}install-templates${C_RESET}  src=${C_DIM}${SRC_DIR}${C_RESET}  dst=${C_DIM}${DST_DIR}${C_RESET}"
(( DRY_RUN )) && echo "${C_YELLOW}(dry-run — no writes)${C_RESET}"

n_current=0; n_updated=0; n_new=0
shopt -s nullglob
for src in "$SRC_DIR"/*.sh; do
  base="$(basename "$src")"
  dst="$DST_DIR/$base"

  if [[ ! -e "$dst" ]]; then
    echo "  ${C_GREEN}NEW${C_RESET}      $base"
    n_new=$((n_new+1))
    if (( ! DRY_RUN )); then
      cp "$src" "$dst"
      chmod +x "$dst"
    fi
  elif cmp -s "$src" "$dst"; then
    echo "  ${C_DIM}current  $base${C_RESET}"
    n_current=$((n_current+1))
  else
    echo "  ${C_BLUE}UPDATE${C_RESET}   $base  ${C_DIM}(backup: ${base}.bak-${STAMP})${C_RESET}"
    n_updated=$((n_updated+1))
    if (( ! DRY_RUN )); then
      cp "$dst" "${dst}.bak-${STAMP}"
      cp "$src" "$dst"
      chmod +x "$dst"
    fi
  fi
done

echo "${C_BOLD}done${C_RESET}  current=${n_current} updated=${n_updated} new=${n_new}"
