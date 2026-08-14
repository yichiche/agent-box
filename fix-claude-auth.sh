#!/usr/bin/env bash
# One-shot repair for container OAuth expiry / ghost claudeAiOauth on the shared mount.
# Safe to run on the host or inside a container (re-runs container auth wiring).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

HOST_CREDS="${HOST_HOME}/.claude/.credentials.json"

fix_ghost_oauth() {
  local creds="$1"
  [[ -f "$creds" ]] || return 0
  python3 - "$creds" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    creds = json.load(f)
oauth = creds.get("claudeAiOauth")
if not oauth:
    print("[claude] no claudeAiOauth entry — nothing to clean")
    sys.exit(0)
if oauth.get("accessToken") or oauth.get("refreshToken"):
    print("[claude] claudeAiOauth has tokens — leaving host login intact")
    sys.exit(0)
del creds["claudeAiOauth"]
with open(path, "w") as f:
    json.dump(creds, f, indent=2)
    f.write("\n")
print("[claude] removed ghost claudeAiOauth (empty tokens) from", path)
PY
}

echo "=== 1/2 clean shared credentials (host) ==="
fix_ghost_oauth "$HOST_CREDS"

echo "=== 2/2 wire container-local API-key auth ==="
bash "${SCRIPT_DIR}/skills/abox-live/claude-container-auth.sh"

echo
echo "Done. Restart any running 'claude' session so it picks up CLAUDE_CONFIG_DIR."
echo "Host subscription login: run '/login' on the HOST (not in container) when you need Remote Control."
