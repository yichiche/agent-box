# env.sh — source this from any agent-box script
AGENT_BOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_HOME="${AGENT_BOX_HOST_HOME:-$(dirname "$AGENT_BOX_DIR")}"
