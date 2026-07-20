#!/usr/bin/env bash
# container-dep.sh — one-shot container bootstrap (called by run_docker.sh).
#   bash container-dep.sh              run all steps, then drop into a shell
#   bash container-dep.sh --no-shell   run setup only
#
# To add a step: define a step_* function, then add a run_step line below.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NO_SHELL=0
[ "${1:-}" = "--no-shell" ] && NO_SHELL=1

FAILED=()

# Run one step: time it, record failures without aborting, reload .bashrc so
# PATH/env written by earlier steps (gh, claude, codex) reach later ones.
run_step() {
  local label="$1"; shift
  local start=$SECONDS
  echo "──────────────── ${label} ────────────────"
  if "$@"; then
    echo "[ok]   ${label}  ($((SECONDS - start))s)"
  else
    local rc=$?
    echo "[FAIL] ${label}  (rc=${rc}, $((SECONDS - start))s)"
    FAILED+=("${label}")
  fi
  set +u; source /root/.bashrc >/dev/null 2>&1 || true; set -u
}

# Steps (echo '' feeds the one prompt in claude-code.sh)
step_claude_code() { echo '' | bash "$DIR/claude-code.sh"; }
step_claude_key()  { bash "$DIR/claude-code-key.sh"; }
step_gh()          { bash "$DIR/gh-setup.sh"; }
step_identity()    { bash "$DIR/setup-global-identity-guard.sh"; }
step_codex()       { bash "$DIR/codex-key.sh"; }
step_pip_extras()  { pip install openpyxl; }

# Order
run_step "claude-code CLI + plugins"  step_claude_code
run_step "claude-code API key/config" step_claude_key
run_step "codex + AMD gateway proxy"  step_codex
run_step "gh install + auth"          step_gh
run_step "git identity guard"         step_identity
run_step "pip extras"                 step_pip_extras

echo "════════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "✅ container-dep: all steps ok"
else
  echo "⚠️  container-dep: ${#FAILED[@]} step(s) FAILED — ${FAILED[*]}"
fi
echo "════════════════════════════════════════════════════════════"

[ "$NO_SHELL" = 1 ] || exec bash
