#!/usr/bin/env bash
# boot.sh — container lifecycle for the pr-conflict-fix skill.
#
#   bash boot.sh up <image> [--pr N] [--name NAME] [--no-gpu] [--fast]
#   bash boot.sh sh <name> <command...>        run a command inside the container
#   bash boot.sh down <name>                   remove the container (it is --rm)
#
# Mirrors the canonical invocation in ~/run_docker.sh, but detached + --rm so the
# agent can exec into it across many steps and have it disappear at teardown.
set -uo pipefail

HOST_HOME="${HOST_HOME:-/home/yichiche}"

sanitize() { echo "$1" | tr -c 'A-Za-z0-9._-' '-' | sed 's/-\+/-/g; s/^-//; s/-$//'; }

cmd_up() {
  local image="${1:?usage: boot.sh up <image> [--pr N] [--name NAME] [--no-gpu] [--fast]}"; shift
  local pr="" name="" gpu=1 fast=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --pr)     pr="$2"; shift 2 ;;
      --name)   name="$2"; shift 2 ;;
      --no-gpu) gpu=0; shift ;;
      --fast)   fast=1; shift ;;
      *) echo "boot.sh: unknown arg $1" >&2; return 2 ;;
    esac
  done

  if [ -z "$name" ]; then
    name="jacky-prfix-${pr:-x}-$(sanitize "${image##*:}")"
    name="${name:0:60}"
  fi

  if sudo docker ps -a --format '{{.Names}}' | grep -qxF "$name"; then
    echo "boot.sh: removing stale container $name"
    sudo docker rm -f "$name" >/dev/null 2>&1
  fi

  local args=(
    -d --rm --privileged --name "$name"
    --network=host --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined --ipc=host --shm-size 16G
    -v "${HOST_HOME}:/home/yichiche/"
    -v "${HOST_HOME}/.claude:/root/.claude"
    -v "${HOST_HOME}/.codex:/root/.codex"
  )
  if [ "$gpu" = 1 ] && [ -e /dev/kfd ]; then
    args+=( --device=/dev/kfd --device=/dev/dri --group-add video )
  fi
  # These exist on the MI355 hosts but not everywhere — mount only what is there.
  for d in /data /data2 /mnt /raid; do
    [ -d "$d" ] && args+=( -v "$d:$d" )
  done

  echo "boot.sh: starting $name from $image"
  sudo docker run "${args[@]}" "$image" sleep infinity >/dev/null || return 1

  # Bootstrap. Full run installs claude/codex/gh/identity-guard/pip extras;
  # --fast installs only what conflict resolution actually needs.
  echo "boot.sh: bootstrapping (this takes a few minutes; --fast to skip the extras)"
  if [ "$fast" = 1 ]; then
    sudo docker exec "$name" bash -c '
      bash /home/yichiche/agent-box/gh-setup.sh
      bash /home/yichiche/agent-box/setup-global-identity-guard.sh
    ' 2>&1 | tail -20
  else
    sudo docker exec "$name" bash /home/yichiche/agent-box/container-dep.sh --no-shell 2>&1 | tail -20
  fi

  echo "$name"
}

cmd_sh() {
  local name="${1:?usage: boot.sh sh <name> <command...>}"; shift
  sudo docker exec \
    -e GH_TOKEN="" \
    -e GH_CONFIG_DIR=/home/yichiche/.gh \
    -e PATH="/home/yichiche/bin:/usr/local/bin:/usr/bin:/bin" \
    "$name" bash -lc "$*"
}

cmd_down() {
  local name="${1:?usage: boot.sh down <name>}"
  sudo docker rm -f "$name" >/dev/null 2>&1
  echo "boot.sh: removed $name"
}

case "${1:-}" in
  up)   shift; cmd_up   "$@" ;;
  sh)   shift; cmd_sh   "$@" ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
