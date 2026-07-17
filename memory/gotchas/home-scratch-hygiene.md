# Don't scatter agent files in $HOME root — use AGENT_SCRATCH_DIR / AGENT_RUNS_DIR

Agents kept dropping ad-hoc scripts/logs/reports straight into `$HOST_HOME`
(`/home/yichiche`) root — `summarize_*.py`, `*_driver.sh`, `qwen35_orchestrate.sh`,
`tmp_*.log`, one-off `*_report.md` — cluttering the home dir (81 loose files at one
point). The per-model output subdirs (`qwen3.5-mxfp4/`, `dsv4/`, `benchmark_runs/`)
are fine; the problem is **loose files at the root**.

**Convention (from `env.sh`, documented in `CLAUDE.md`):**
- Ad-hoc scripts / logs / reports → **`$AGENT_SCRATCH_DIR`** (`$HOST_HOME/agent-scratch`),
  under a per-task subdir.
- Structured run outputs (traces, sweeps) → **`$AGENT_RUNS_DIR`** (`$HOST_HOME/agent-runs`)
  or existing per-model dirs.
- Ephemeral one-shot → `/tmp`.
- **Never** write to `$HOST_HOME` root.

**Exception:** canonical human-owned files already at `$HOST_HOME` — especially the
`~/run_*.sh` model launch scripts the model cards reference (see [[../models/INDEX]]) —
stay put. Do not relocate them.

Get the paths: `source "$AGENT_BOX_DIR/env.sh"` then `mkdir -p "$AGENT_SCRATCH_DIR/<task>"`.
