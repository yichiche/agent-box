#!/usr/bin/env bash
# Install Node.js 22 (nodesource) for Codex CLI. Idempotent; waits on apt locks.
set -euo pipefail

wait_for_apt() {
  local i=0
  while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
     || fuser /var/lib/apt/lists/lock >/dev/null 2>&1 \
     || fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    if [ "$i" -ge 90 ]; then
      echo "[nodejs] apt lock held too long (>3min)" >&2
      return 1
    fi
    echo "[nodejs] waiting for apt lock..."
    sleep 2
    i=$((i + 1))
  done
}

if command -v npm >/dev/null 2>&1; then
  echo "[nodejs] already installed ($(node --version), npm $(npm --version))"
  exit 0
fi

wait_for_apt
apt-get update
wait_for_apt
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
wait_for_apt
apt-get install -y nodejs
echo "[nodejs] installed ($(node --version), npm $(npm --version))"
