---
name: gh-setup
description: "Make the GitHub CLI (`gh`) usable in non-interactive shells. Installs `gh` to `~/bin` (no sudo) when missing and exports `GH_TOKEN` from `~/.git-credentials` so commands like `gh run list` work without `gh auth login`. Use whenever a script or skill (e.g. `inferencex-table`) fails with `command not found: gh`, `gh: To use GitHub CLI in automation, set the GH_TOKEN environment variable`, or `gh auth login` prompts."
---

# gh-setup

Most boxes here do **not** have `gh` pre-installed and have no passwordless `sudo` for general install paths, but they do have a long-lived PAT in `~/.git-credentials`. This skill is the canonical way to make `gh` work in those environments without touching root.

## When to use

Trigger this skill whenever you hit any of:

- `FileNotFoundError: ... 'gh'` from a Python `subprocess`
- Shell error: `gh: command not found`
- `gh: To use GitHub CLI in automation, set the GH_TOKEN environment variable.`
- `gh auth status` says "You are not logged into any GitHub hosts"
- Any other skill/script (most commonly `inferencex-table`) fails because `gh` is missing or unauthenticated.

Do **not** use this for repo-level git auth (cloning, pushing) — `~/.git-credentials` already handles HTTPS for `git`. This skill is specifically for the `gh` CLI.

---

## Two entry points

| Entry point | When it runs | What it does |
|---|---|---|
| **Auto-setup** — `~/agent-box/gh-setup.sh` (chained from `run_docker.sh`) | Once, at Docker container launch | Installs `gh` to `${HOST_HOME}/bin/gh` (host-mounted, persistent across containers) and appends a managed block to `~/.bashrc` so every interactive shell in the container has `gh` on `PATH` and `GH_TOKEN` exported. |
| **On-demand fix** — `scripts/setup_gh.sh` (this skill) | Manually, when an agent shell hits the failure modes below | Same install + token-export, but for the current shell only. Use when working on a host without the `agent-box` chain or when troubleshooting. |

### Auto-setup at Docker launch

`run_docker.sh` chains it into the container init line, e.g.:

```bash
bash -c "echo '' | bash $HOME/agent-box/claude-code.sh \
      && bash $HOME/agent-box/claude-code-key.sh \
      && bash $HOME/agent-box/gh-setup.sh \
      && pip install openpyxl \
      && exec bash"
```

After `exec bash`, the new interactive shell sources `~/.bashrc`, which puts `${HOST_HOME}/bin` on `PATH` and resolves `GH_TOKEN` from `$HOME/.git-credentials` (or `${HOST_HOME}/.git-credentials` as fallback for in-container shells where `$HOME=/root`).

### On-demand fix (one-liner)

If `gh` is missing in the current shell session — e.g. on a host that didn't run through `run_docker.sh`, or in an existing container that pre-dates this skill — source the bootstrap script:

```bash
source ~/agent-box/skills/gh-setup/scripts/setup_gh.sh
```

That script is idempotent and does the following:

1. Skips install if `gh` is already on `PATH` or already at `~/bin/gh`.
2. Otherwise downloads the pinned `gh` tarball, extracts the binary into `~/bin/gh`, makes it executable.
3. Prepends `~/bin` to `PATH` for the current shell.
4. Exports `GH_TOKEN` by parsing the PAT out of `~/.git-credentials`.
5. Verifies with `gh --version` and `gh auth status` (best effort).

After sourcing, the same shell can run `gh run list -R SemiAnalysisAI/InferenceX ...` directly, and any Python script that shells out to `gh` (e.g. `inferencex-table/scripts/search_bmk.py`) will work.

> **Important — invoking from another skill:** subprocesses inherit `PATH` and `GH_TOKEN` only if they're set in the **current** shell *before* the subprocess is spawned. Always `source` (not just run) the script in the same shell that will launch the downstream command.

---

## Manual fallback (when the bootstrap script is unavailable)

Run these by hand. They're the exact steps the script automates.

### Step 1: Install `gh` to `~/bin` (no sudo required)

```bash
GH_VERSION=2.62.0
mkdir -p ~/bin
cd /tmp
curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" -o gh.tar.gz
tar xzf gh.tar.gz
cp "gh_${GH_VERSION}_linux_amd64/bin/gh" ~/bin/gh
chmod +x ~/bin/gh
export PATH="$HOME/bin:$PATH"
gh --version
```

Pinned to `v2.62.0` — known to work with the `inferencex-table` scripts. Bump only if a newer feature is required.

### Step 2: Export `GH_TOKEN` from `~/.git-credentials`

`gh auth login` requires an interactive browser flow, which won't work in agent shells. Skip it and feed `gh` the PAT we already have:

```bash
export GH_TOKEN=$(grep -oP 'https://[^:]+:\K[^@]+' ~/.git-credentials | head -1)
gh auth status        # should now say "Logged in to github.com"
```

The token must have at minimum `repo` and `actions:read` scopes (the `inferencex-table` skill needs these to list workflow runs and download artifacts).

### Step 3 (optional): Make it permanent

If you don't want to re-source on every new shell, append to `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'

# gh-setup: PAT-based auth for the GitHub CLI
export PATH="$HOME/bin:$PATH"
if [ -f "$HOME/.git-credentials" ] && [ -z "$GH_TOKEN" ]; then
  export GH_TOKEN=$(grep -oP 'https://[^:]+:\K[^@]+' "$HOME/.git-credentials" | head -1)
fi
EOF
```

---

## Diagnosing common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `gh: command not found` | binary not installed or not on PATH | Source the bootstrap script (Step 1) |
| `To use GitHub CLI in automation, set the GH_TOKEN environment variable.` | `gh` installed but no auth context | Run Step 2 to export `GH_TOKEN` |
| `HTTP 401: Bad credentials` | PAT in `~/.git-credentials` is expired or has wrong scopes | Rotate the PAT; ensure scopes `repo` + `actions:read` |
| `gh: HTTP 403: Resource not accessible by personal access token` | PAT missing `actions:read` (private repos need this even for public-style endpoints) | Re-issue PAT with `actions:read` |
| `FileNotFoundError: ... 'gh'` from Python subprocess | Python child process didn't inherit the updated `PATH` | Make sure `~/bin` is on `PATH` **before** launching the Python script in the same shell |
| `gh auth login` opens a browser prompt and hangs | Don't use `gh auth login` here — it requires interactive flow | Use `GH_TOKEN` env var instead (Step 2) |
| `gh run list failed: gh: To use GitHub CLI in automation...` from `inferencex-table` | Script invoked `gh` without `GH_TOKEN` in env | `source` this skill's setup script before running |

---

## Verification

After setup, all of these should succeed without prompting:

```bash
gh --version
gh auth status
gh run list -R SemiAnalysisAI/InferenceX --limit 1
```

If `gh run list` returns a non-empty list, downstream skills like `inferencex-table` will work end-to-end.

---

## Notes

- The bootstrap script writes the binary to `~/bin/gh` only — it does not touch `/usr/local/bin` or anything that needs root.
- `GH_TOKEN` takes precedence over `gh auth login` state, so it's safe to set even if someone has run `gh auth login` previously.
- This skill assumes Linux x86_64 (the environments we run in). For other architectures, change the asset name (e.g. `gh_${GH_VERSION}_linux_arm64.tar.gz`).
- Do not commit `~/.git-credentials` or `GH_TOKEN` anywhere; these are user-local secrets.
