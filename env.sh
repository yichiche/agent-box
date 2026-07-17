# env.sh — source this from any agent-box script
AGENT_BOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_HOME="${AGENT_BOX_HOST_HOME:-$(dirname "$AGENT_BOX_DIR")}"

# Default homes for agent-CREATED files — keep $HOST_HOME root clean.
#   AGENT_SCRATCH_DIR — ad-hoc scripts / logs / reports the agent writes (summarize_*.py,
#                       *_driver.sh, tmp_*.log, *_report.md, one-off analysis). NEVER $HOME root.
#   AGENT_RUNS_DIR    — structured run outputs (traces, sweeps) already organized per model.
# Canonical, human-owned files (e.g. ~/run_*.sh model scripts) stay at $HOST_HOME.
AGENT_SCRATCH_DIR="${AGENT_SCRATCH_DIR:-$HOST_HOME/agent-scratch}"
AGENT_RUNS_DIR="${AGENT_RUNS_DIR:-$HOST_HOME/agent-runs}"
export AGENT_SCRATCH_DIR AGENT_RUNS_DIR

# Jira REST creds (amd.atlassian.net) for track_jira.py --fetch / discover.js jira source.
# File holds `export JIRA_EMAIL=... JIRA_API_TOKEN=...`; keep it chmod 600.
if [ -z "${JIRA_API_TOKEN:-}" ] && [ -f "$HOST_HOME/.jira_credentials" ] \
   && ! grep -q "REPLACE_WITH" "$HOST_HOME/.jira_credentials"; then
  . "$HOST_HOME/.jira_credentials"
fi
