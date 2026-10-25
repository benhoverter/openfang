#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# deploy-local.sh — OpenFang local build-test-deploy workflow (macOS, launchd)
#
# Phases:
#   0. Branch hygiene check (refuses non-main without --force-branch)
#   1. cargo check     (fast type gate)
#   2. cargo test      (THE critical gate — non-negotiable)
#   3. cargo build --release (daemon + MCP bridge)
#   4. Snapshot live binary
#   4b. Sync runtime script templates
#   5. Swap binaries + restart daemon (via `openfang service restart`)
#   6. Smoke test (daemon status + log tail)
#
# Usage:
#   scripts/deploy-local.sh <label> [--skip-tests] [--scope <crate>] [--dry-run] [--force-branch]
#
# First-time setup: install the daemon as a launchd service once with
#   openfang service install
# This script only updates and restarts an existing install.
#
# Arguments:
#   <label>          Required. Short label for the patch (e.g. "timeout-patch").
#                    Used in the backup filename: openfang.v<version>-pre-<label>.
#
# Options:
#   --skip-tests     Skip Phase 2. Emergency-rollback-redeploy only.
#   --scope <crate>  Run cargo test only for the named crate (e.g. openfang-runtime).
#                    Full workspace test still recommended for real deploys.
#   --dry-run        Run check, test, build and a template preview; skip the
#                    binary swap and daemon restart.
#   --force-branch   Allow deploy from a branch other than `main`. A feature
#                    branch ships a subset of trunk and can look like a
#                    regression. Override the expected branch with
#                    OPENFANG_DEPLOY_BRANCH=<name>.
#
# Environment overrides:
#   OPENFANG_REPO          repo root   (default: the repo this script lives in)
#   OPENFANG_BIN_DIR       install dir (default: ~/.openfang/bin)
#   OPENFANG_DAEMON_LABEL  launchd job label (default: ai.openfang.daemon)
#   OPENFANG_DAEMON_LOG    daemon log  (default: ~/.openfang/daemon.stderr.log)
#   OPENFANG_DEPLOY_BRANCH expected branch (default: main)
#
# Rollback (manual, on failure):
#   cp <backup-path> ~/.openfang/bin/openfang.tmp
#   mv -f ~/.openfang/bin/openfang.tmp ~/.openfang/bin/openfang   (never cp in place)
#   openfang service restart
#
# Exit codes:
#   0   Deploy successful (or dry-run completed without error)
#   1   Phase 0 (branch hygiene) — refused non-main without --force-branch
#   2   Phase 1 (cargo check) failed
#   3   Phase 2 (cargo test) failed
#   4   Phase 3 (cargo build) failed
#   5   Phase 4 (snapshot) failed
#   6   Phase 5 (swap/restart) failed
#   7   Phase 6 (smoke test) failed
#   8   Phase 4b (template sync) failed
#   64  Usage / setup error
# ------------------------------------------------------------------------------

set -euo pipefail

# --- Config -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OPENFANG_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN_DIR="${OPENFANG_BIN_DIR:-$HOME/.openfang/bin}"
BIN="${BIN_DIR}/openfang"
# Bridge subprocess binary spawned by Claude Code via --mcp-config. Installed
# next to the daemon so the daemon's boot probe finds it.
BRIDGE_BIN="${BIN_DIR}/openfang-mcp-bridge"
DAEMON_LABEL="${OPENFANG_DAEMON_LABEL:-ai.openfang.daemon}"
DAEMON_LOG="${OPENFANG_DAEMON_LOG:-$HOME/.openfang/daemon.stderr.log}"
LOG_TAIL_SECS=5

# --- Color output (only if stdout is a tty) ----------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_DIM=$'\033[2m'
else
  C_RESET= C_BOLD= C_RED= C_GREEN= C_YELLOW= C_BLUE= C_DIM=
fi

phase()  { printf '\n%s▶ Phase %s: %s%s\n' "${C_BOLD}${C_BLUE}" "$1" "$2" "${C_RESET}"; }
ok()     { printf '%s  ✓ %s%s\n' "${C_GREEN}" "$1" "${C_RESET}"; }
warn()   { printf '%s  ⚠ %s%s\n' "${C_YELLOW}" "$1" "${C_RESET}"; }
fail()   { printf '%s  ✗ %s%s\n' "${C_RED}${C_BOLD}" "$1" "${C_RESET}"; }
info()   { printf '%s    %s%s\n' "${C_DIM}" "$1" "${C_RESET}"; }

daemon_line() { launchctl list | awk -v label="$DAEMON_LABEL" '$3 == label {print $0}' || true; }
rollback_hint() { info "Rollback: cp $BACKUP $BIN.tmp && mv -f $BIN.tmp $BIN && $BIN service restart --label $DAEMON_LABEL"; }

# Install a binary by copy-to-temp + rename, never `cp` over the target.
# On macOS (Apple Silicon) overwriting a signed Mach-O in place reuses its
# inode, and the kernel's cached code signature no longer matches the new
# bytes: the next exec is SIGKILLed ("Killed: 9"). A rename gives a new inode.
install_bin() {
  local src="$1" dst="$2" tmp="$2.tmp.$$"
  cp "$src" "$tmp" && chmod 755 "$tmp" && mv -f "$tmp" "$dst" || { rm -f "$tmp"; return 1; }
}

# --- Argument parsing --------------------------------------------------------
LABEL=""
SKIP_TESTS=0
TEST_SCOPE=""
DRY_RUN=0
FORCE_BRANCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests)   SKIP_TESTS=1; shift ;;
    --scope)
      if [[ $# -lt 2 ]]; then fail "--scope needs a crate name"; exit 64; fi
      TEST_SCOPE="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --force-branch) FORCE_BRANCH=1; shift ;;
    -h|--help)
      sed -n '3,58p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*)
      fail "Unknown flag: $1"
      exit 64
      ;;
    *)
      if [[ -z "$LABEL" ]]; then
        LABEL="$1"
      else
        fail "Unexpected positional argument: $1 (label already set to '$LABEL')"
        exit 64
      fi
      shift
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  fail "Missing required <label> argument."
  echo "Usage: $(basename "$0") <label> [--skip-tests] [--scope <crate>] [--dry-run] [--force-branch]"
  exit 64
fi

# Sanitize label (allow alnum, dash, underscore, dot)
if [[ ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  fail "Label contains illegal characters. Use only [A-Za-z0-9._-]."
  exit 64
fi

# --- Sanity checks before any phase runs -------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  fail "deploy-local.sh supports macOS (launchd) only."
  exit 64
fi
if [[ ! -f "$REPO/Cargo.toml" ]]; then
  fail "No OpenFang repo at: $REPO"
  info "Override with OPENFANG_REPO=/path/to/repo"
  exit 64
fi
if [[ ! -f "$BIN" ]]; then
  fail "No installed binary at: $BIN"
  info "First install: cargo build --release, copy target/release/openfang and"
  info "openfang-mcp-bridge into $BIN_DIR, then run: $BIN_DIR/openfang service install"
  info "Or override with OPENFANG_BIN_DIR=/path/to/bin"
  exit 64
fi

cd "$REPO"

# Derive version string for backup filename; "unknown" if the binary won't say.
VERSION=$("$BIN" --version 2>/dev/null | awk '{print $NF}' | head -n1 || true)
VERSION="${VERSION:-unknown}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${BIN_DIR}/openfang.v${VERSION}-pre-${LABEL}-${TIMESTAMP}"

printf '%s%s OpenFang local deploy %s\n' "${C_BOLD}" "═══════════════════════════════════════" "${C_RESET}"
info "Repo:    $REPO"
info "Binary:  $BIN"
info "Job:     $DAEMON_LABEL"
info "Version: $VERSION"
info "Label:   $LABEL"
info "Backup:  $BACKUP"
[[ $DRY_RUN -eq 1 ]] && warn "DRY RUN — phases 4, 5, 6 will be skipped"
[[ $SKIP_TESTS -eq 1 ]] && warn "SKIP TESTS — phase 2 will be skipped (gate disabled)"
[[ -n "$TEST_SCOPE" ]] && info "Test scope: -p $TEST_SCOPE"

# ---------------------------------------------------------------------------
# Phase 0 — Branch hygiene
# ---------------------------------------------------------------------------
phase 0 "branch hygiene"
BRANCH="$(git branch --show-current 2>/dev/null || echo '<detached>')"
info "Current branch: $BRANCH"
if ! git diff --quiet || ! git diff --cached --quiet; then
  warn "Working tree has uncommitted changes."
  git status --short | sed 's/^/    /'
else
  ok "Working tree clean."
fi

# `main` is trunk. Any other branch ships a subset of it, and features merged
# elsewhere will look broken (ANAI-43: discord outbound attachments).
EXPECTED_BRANCH="${OPENFANG_DEPLOY_BRANCH:-main}"
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  if [[ $FORCE_BRANCH -eq 1 ]]; then
    warn "Deploying from '$BRANCH' (expected '$EXPECTED_BRANCH'). --force-branch acknowledged."
    warn "Reminder: a feature branch ships a SUBSET of trunk."
  else
    fail "Refusing to deploy from '$BRANCH'. Expected '$EXPECTED_BRANCH'."
    info "To deploy main:          git checkout main && $(basename "$0") $LABEL"
    info "To force this branch:    $(basename "$0") $LABEL --force-branch"
    info "To change the expected:  OPENFANG_DEPLOY_BRANCH=<branch> $(basename "$0") $LABEL"
    exit 1
  fi
else
  ok "On expected branch '$EXPECTED_BRANCH'."
fi

# ---------------------------------------------------------------------------
# Phase 1 — cargo check (fast type gate)
# ---------------------------------------------------------------------------
phase 1 "cargo check --workspace --all-targets"
if cargo check --workspace --all-targets; then
  ok "Type check passed."
else
  fail "cargo check failed. Aborting before any further work."
  exit 2
fi

# ---------------------------------------------------------------------------
# Phase 2 — cargo test (THE gate)
# ---------------------------------------------------------------------------
if [[ $SKIP_TESTS -eq 1 ]]; then
  phase 2 "cargo test (SKIPPED via --skip-tests)"
  warn "Tests skipped. Use sparingly."
else
  if [[ -n "$TEST_SCOPE" ]]; then
    phase 2 "cargo test -p $TEST_SCOPE"
    if cargo test -p "$TEST_SCOPE"; then
      ok "Scoped tests passed."
    else
      fail "cargo test failed. Aborting before build."
      exit 3
    fi
  else
    phase 2 "cargo test --workspace"
    if cargo test --workspace; then
      ok "All tests passed."
    else
      fail "cargo test failed. Aborting before build."
      exit 3
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Phase 3 — cargo build --release
# ---------------------------------------------------------------------------
# Tolerate checkouts that predate the openfang-mcp-bridge crate: build only the
# daemon and leave any installed bridge binary in place.
HAS_BRIDGE=0
if [[ -f "$REPO/crates/openfang-mcp-bridge/Cargo.toml" ]]; then
  HAS_BRIDGE=1
fi

if [[ $HAS_BRIDGE -eq 1 ]]; then
  phase 3 "cargo build --release --bin openfang --bin openfang-mcp-bridge"
  if cargo build --release --bin openfang --bin openfang-mcp-bridge; then
    ok "Release build complete."
  else
    fail "cargo build --release failed."
    exit 4
  fi
else
  phase 3 "cargo build --release --bin openfang (bridge crate absent — skipped)"
  if cargo build --release --bin openfang; then
    ok "Release build complete (daemon only)."
  else
    fail "cargo build --release failed."
    exit 4
  fi
fi

BUILT="$REPO/target/release/openfang"
BUILT_BRIDGE="$REPO/target/release/openfang-mcp-bridge"
if [[ ! -f "$BUILT" ]]; then
  fail "Expected binary not found: $BUILT"
  exit 4
fi
if [[ $HAS_BRIDGE -eq 1 && ! -f "$BUILT_BRIDGE" ]]; then
  fail "Expected bridge binary not found: $BUILT_BRIDGE"
  exit 4
fi

# ---------------------------------------------------------------------------
# Phase 4 — Snapshot the live binary
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
  phase 4 "snapshot (SKIPPED — dry run)"
else
  phase 4 "snapshot live binary → $BACKUP"
  if cp -p "$BIN" "$BACKUP"; then
    LIVE_SIZE=$(stat -f%z "$BIN")
    BACKUP_SIZE=$(stat -f%z "$BACKUP")
    if [[ "$LIVE_SIZE" -ne "$BACKUP_SIZE" ]]; then
      fail "Backup size mismatch (live=$LIVE_SIZE, backup=$BACKUP_SIZE). Aborting."
      exit 5
    fi
    ok "Snapshot created ($BACKUP_SIZE bytes)."
  else
    fail "Failed to create backup."
    exit 5
  fi
fi

# ---------------------------------------------------------------------------
# Phase 4b — Sync runtime script templates (ANAI-131)
# ---------------------------------------------------------------------------
# file_convert's md->pdf recipe runs ~/.openfang/scripts/build-pdf.sh, which
# must match crates/openfang-runtime/templates/*.sh. install-templates.sh is
# idempotent, fail-closed and backs up on change. Runs before the restart so
# the new binary serves file_convert against current templates.
INSTALL_TEMPLATES="$REPO/scripts/install-templates.sh"
if [[ -f "$INSTALL_TEMPLATES" ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    phase 4b "sync script templates (dry-run preview)"
    if OPENFANG_REPO="$REPO" bash "$INSTALL_TEMPLATES" --dry-run; then
      ok "Template sync preview complete."
    else
      fail "Template sync preview failed."
      exit 8
    fi
  else
    phase 4b "sync script templates into ~/.openfang/scripts"
    if OPENFANG_REPO="$REPO" bash "$INSTALL_TEMPLATES"; then
      ok "Templates synced."
    else
      fail "Template sync failed. Aborting before daemon restart."
      exit 8
    fi
  fi
else
  phase 4b "sync script templates (installer absent — skipped)"
  warn "No installer at $INSTALL_TEMPLATES; skipping template sync."
fi

# ---------------------------------------------------------------------------
# Phase 5 — Swap binary + restart daemon
# ---------------------------------------------------------------------------
# Restart goes through `openfang service restart` (the freshly installed
# binary). It kickstarts when the plist is unchanged and does a full
# bootout + bootstrap when the plist was edited, and it records which plist
# launchd loaded so `service status` stays accurate. If the installed binary
# predates `openfang service` (e.g. after a rollback to an old build), fall back
# to a raw bootout + bootstrap, which always re-reads the plist.
legacy_restart() {
  local domain plist
  domain="gui/$(id -u)"
  plist="$HOME/Library/LaunchAgents/${DAEMON_LABEL}.plist"
  if [[ ! -f "$plist" ]]; then
    fail "Daemon plist not found at: $plist"
    return 1
  fi
  launchctl bootout "${domain}/${DAEMON_LABEL}" 2>/dev/null || true
  # bootout returns before launchd finishes teardown; bootstrapping too early
  # fails with "5: Input/output error".
  local deadline=$((SECONDS + 10))
  while launchctl print "${domain}/${DAEMON_LABEL}" >/dev/null 2>&1; do
    (( SECONDS >= deadline )) && break
    sleep 0.2
  done
  launchctl bootstrap "$domain" "$plist" || { sleep 2; launchctl bootstrap "$domain" "$plist"; }
}

restart_daemon() {
  local rc=0
  "$BIN" service --help >/dev/null 2>&1 || rc=$?
  if (( rc > 128 )); then
    # Killed by a signal: the binary can't run at all. Not a missing command.
    fail "Installed binary was killed on exec (signal $((rc - 128))); not restarting with it."
    return 1
  elif (( rc == 0 )); then
    "$BIN" service restart --label "$DAEMON_LABEL"
  else
    warn "Installed binary has no 'service' command; using raw launchctl restart."
    legacy_restart
  fi
}

if [[ $DRY_RUN -eq 1 ]]; then
  phase 5 "deploy (SKIPPED — dry run)"
else
  phase 5 "swap binary and restart daemon"
  if ! launchctl print "gui/$(id -u)/${DAEMON_LABEL}" >/dev/null 2>&1 \
     && [[ ! -f "$HOME/Library/LaunchAgents/${DAEMON_LABEL}.plist" ]]; then
    fail "No launchd job '$DAEMON_LABEL' is installed. Live binary unchanged."
    info "Install the service once first: $BIN service install"
    exit 6
  fi

  OLD_PID=$(daemon_line | awk '{print $1}')
  info "Daemon PID before swap: ${OLD_PID:-<not running>}"

  if ! install_bin "$BUILT" "$BIN"; then
    fail "Failed to swap binary. Live binary unchanged."
    exit 6
  fi
  ok "Binary swapped."

  # The bridge isn't snapshotted: it's spawned per Claude Code turn, not a
  # long-running service. A failed copy only warns.
  if [[ $HAS_BRIDGE -eq 1 ]]; then
    if install_bin "$BUILT_BRIDGE" "$BRIDGE_BIN"; then
      ok "Bridge binary installed at $BRIDGE_BIN."
    else
      warn "Failed to install bridge binary at $BRIDGE_BIN; agents with the bridge explicitly enabled will have no tools until fixed."
    fi
  else
    warn "Bridge crate not in this checkout; leaving existing $BRIDGE_BIN in place (if any)."
  fi

  if ! restart_daemon; then
    fail "Daemon restart failed. Rolling back binary and retrying."
    install_bin "$BACKUP" "$BIN" || true
    restart_daemon || info "Manual recovery: $BIN service start --label $DAEMON_LABEL"
    exit 6
  fi

  # Give launchd a moment to bring the new process up
  sleep 2

  NEW_STATUS=$(daemon_line)
  if [[ -z "$NEW_STATUS" ]]; then
    fail "Daemon not running after restart."
    rollback_hint
    exit 6
  fi
  NEW_PID=$(echo "$NEW_STATUS" | awk '{print $1}')
  EXIT_STATUS=$(echo "$NEW_STATUS" | awk '{print $2}')

  if [[ "$EXIT_STATUS" != "0" && "$EXIT_STATUS" != "-" ]]; then
    fail "Daemon reports non-zero exit status: $EXIT_STATUS"
    rollback_hint
    exit 6
  fi

  if [[ -n "$OLD_PID" && "$OLD_PID" != "-" && "$NEW_PID" == "$OLD_PID" ]]; then
    warn "PID unchanged ($NEW_PID) — daemon may not have actually restarted."
  else
    ok "Daemon restarted (PID ${OLD_PID:-?} → $NEW_PID, status=$EXIT_STATUS)."
  fi
fi

# ---------------------------------------------------------------------------
# Phase 6 — Smoke test (daemon log sanity)
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
  phase 6 "smoke test (SKIPPED — dry run)"
else
  phase 6 "smoke test (tailing $DAEMON_LOG for ${LOG_TAIL_SECS}s)"
  if [[ ! -f "$DAEMON_LOG" ]]; then
    warn "Daemon log not found at $DAEMON_LOG — skipping log-based smoke test."
  else
    LOG_BEFORE=$(wc -l < "$DAEMON_LOG" || echo 0)
    sleep "$LOG_TAIL_SECS"
    LOG_AFTER=$(wc -l < "$DAEMON_LOG" || echo 0)
    NEW_LINES=$((LOG_AFTER - LOG_BEFORE))
    info "Log grew by $NEW_LINES lines in ${LOG_TAIL_SECS}s."

    RECENT=$(tail -n 50 "$DAEMON_LOG" 2>/dev/null || echo "")
    if echo "$RECENT" | grep -qE 'panicked at|FATAL|thread .* panicked'; then
      fail "Panic detected in recent daemon log:"
      echo "$RECENT" | grep -E 'panicked at|FATAL|thread .* panicked' | head -n 5 | sed 's/^/    /'
      rollback_hint
      exit 7
    fi
    ok "No panics in recent log."
  fi

  FINAL=$(daemon_line)
  if [[ -z "$FINAL" ]]; then
    fail "Daemon died during smoke window."
    rollback_hint
    exit 7
  fi
  ok "Daemon still running after smoke window."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
printf '\n%s%s ✅ Deploy complete %s\n' "${C_BOLD}${C_GREEN}" "═══════════════════════════════════════" "${C_RESET}"
if [[ $DRY_RUN -eq 0 ]]; then
  info "Backup:   $BACKUP"
  rollback_hint
  echo ""
  info "Next: '$BIN service status', then have an agent reply in chat."
else
  info "Dry run only — no binary swap, no daemon restart."
fi
