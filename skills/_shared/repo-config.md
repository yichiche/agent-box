# Repo Configuration

Shared configuration referenced by commit, commit-push, commit-push-pr, and pr skills.

## Author

Always use this author for commits:
```
jacky.cheng <yichiche@amd.com>
```

## Repo Table

| Repo root | Push remote URL | PR base repo | PR base branch | Commit on main? |
|---|---|---|---|---|
| `$SGLANG_ROOT` (detect via `python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])"`) | `https://github.com/yichiche/sglang` | `sgl-project/sglang` | `main` | No — must use feature branch |
| `$HOME/agent-box` | `https://github.com/yichiche/agent-box` | `yichiche/agent-box` | `main` | Yes — commit directly on main |
| InferenceX / InferenceMax (detect via `git rev-parse --show-toplevel` basename `InferenceX`, e.g. `/sgl-workspace/InferenceX`) | `origin` → `https://github.com/SemiAnalysisAI/InferenceX` (**direct push, no fork**) | `SemiAnalysisAI/InferenceX` | `main` | No — must use an `amd/<branch_name>` branch |

**IMPORTANT:** The SGLang repo root may be at different paths on different machines (e.g., `/sgl-workspace/sglang`, `$HOME/sglang`). Always detect it dynamically from the active Python environment instead of hardcoding a path.

For any repo not listed above, ask the user for the remote URL, PR base repo, and base branch.

## InferenceX / InferenceMax (`SemiAnalysisAI/InferenceX`)

Also referred to as **InferenceMax** — the repo is `InferenceX`, the benchmark/dashboard the user
and others call InferenceMax. Treat both names as the same target.

Benchmark-recipe repo (config YAML + launcher scripts), **not** a code repo. Read `AGENTS.md`,
`CONTRIBUTING.md`, and `.claude/commands/nuke.md` in the checkout before editing — they are the
source of truth and override anything here.

### Branch and title conventions (NON-NEGOTIABLE)

- **Branch**: `amd/<branch_name>`, pushed **to the InferenceX repo itself** (`origin` =
  `SemiAnalysisAI/InferenceX`), not to a personal fork. The PR is opened from that branch to `main`.
  Do NOT use `klaud-cold/*` — that namespace belongs to the automated `/nuke` cron bumps.
- **Title / commit subject**: `[AMD][<model_name>] <title>` — two bracket tags, e.g.
  `[AMD][Qwen3.5] …`, `[AMD][DSV4] …`, `[AMD][GLM5.2] …`. Confirm the existing spelling of the
  model tag with `git log origin/main --format='%s' -200 | grep -oE '^\[[^]]+\]\[[^]]+\]'`; the repo
  also uses SKU/scenario tags such as `[AMD][MI35X]` and `[AMD][AgentX]` where that fits better.
- **English only.** The repo's `AGENTS.md` asks for bilingual PR text, but the user has overridden
  this: write **no Chinese characters** in commit subjects, commit bodies, PR titles, PR bodies, or
  PR comments. English only, everywhere.

### Push (credential helper is NOT configured)

Plain `git push` fails with `fatal: could not read Username for 'https://github.com'` even though
`gh auth status` is logged in (the gh config lives at `/home/yichiche/.gh`, and `gh auth setup-git`
does not stick). Push through gh's credential helper inline — this does not mutate any git config:

```bash
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=credential.helper \
GIT_CONFIG_VALUE_0='!gh auth git-credential' \
git push -u origin <branch>
```

`GH_TOKEN=""` is **not** needed here — the OAuth token works against this repo.

### `gh pr edit` is broken on this repo

Any `gh pr edit` / `gh pr view` call that touches PR metadata fails with:
`GraphQL: Projects (classic) is being deprecated … (repository.pullRequest.projectCards)`.
Use the REST API for edits, and `gh api` for reads:

```bash
gh api -X PATCH repos/SemiAnalysisAI/InferenceX/pulls/<num> -f title="<title>"
gh api -X PATCH repos/SemiAnalysisAI/InferenceX/pulls/<num> -f body="$(cat <draft-file>)"
gh api repos/SemiAnalysisAI/InferenceX/pulls/<num> --jq '.title'
gh api -X POST repos/SemiAnalysisAI/InferenceX/issues/<num>/labels -f 'labels[]=full-sweep-fail-fast'
```

`gh pr create --label full-sweep-fail-fast` and `gh pr checks <num>` both work fine.

### Mandatory per-PR requirements

- **`perf-changelog.yaml` entry.** Every recipe change or perf-affecting change needs one. CI job
  `check-changelog` fails without it. The file is **append-only and byte-sensitive**: preserve all
  existing bytes and separator whitespace, append only at the tail, and it **must end with a
  trailing newline** (CI errors with `perf-changelog.yaml at <sha> does not end with a newline`).
- **`full-sweep-fail-fast` label** on the PR, or no benchmark sweep runs. Prefer it over
  `full-sweep-enabled` — a broken change burns one job per matrix instead of the full fan-out.
- **Commit subject `[AMD][<model_name>] <title>`** — see the conventions section above.
- **`Co-Authored-By` for humans is allowed** on this repo (unlike sglang) when the user names one.
  Resolve the noreply address from the numeric id: `gh api users/<login> --jq '.id'` →
  `<login> <id>+<login>@users.noreply.github.com`. Still **never** add a Claude co-author trailer.
- **No pre-commit config** exists in this repo — skip the pre-commit loop.

### Editing config YAML safely

`configs/amd-master.yaml` / `configs/nvidia-master.yaml` reuse the same image tag under many
top-level keys, so a blind `sed` is unsafe. Edit the `image:` line scoped to the specific config
key (see the `edit_image.py` helper in `.claude/commands/nuke.md`), or use `Edit` with the config
key line included in `old_string` for uniqueness.

Before bumping an image tag, verify it exists — never invent a tag. AMD/ROCm SGLang tags live in
`lmsysorg/sglang-rocm`, **not** `lmsysorg/sglang` (the ROCm-suffixed tag 404s there):

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags/<TAG>"   # want 200
```

### Python file-rewrite footgun (cost a broken CI run)

Never write `open(f,'w').write(open(f).read().replace(...))` — `open(f,'w')` truncates the file
before the inner read runs, silently emptying it. Always read first, then write:

```python
c = open(f).read()
c = c.replace(old, new, 1)
if not c.endswith('\n'): c += '\n'
open(f, 'w').write(c)
```

After any scripted edit to `perf-changelog.yaml`, verify before committing:

```bash
python3 -c "
import yaml; d=open('perf-changelog.yaml','rb').read()
assert d.endswith(b'\n'), 'missing trailing newline'
print('entries', len(yaml.safe_load(d.decode())))"
git diff origin/main --stat   # expect a small, additive diff
```

## Prerequisites

If `gh` (GitHub CLI) is not installed or is the wrong package (e.g., the pip `gh` v0.0.4), install the official CLI:
```bash
# Remove the pip gh if it shadows the real one
pip uninstall gh -y 2>/dev/null

# Install official GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update -qq && sudo apt install gh -y
```

Verify: `gh --version` should show v2.x+, not v0.0.4.

## PR Draft Location

PR drafts are written to:
```
$HOME/pr-drafts/
```

### Naming convention

The file is named by slugifying the PR title:
1. Remove the `[AMD]` tag prefix and any `amd/deepseek_v4 integration NN/N` series prefix
2. Take the remaining descriptive part
3. Lowercase, replace spaces with hyphens, remove special characters
4. Prepend `pr-draft-` and append `.md`

Example: title `[AMD] amd/deepseek_v4 integration 22/N fused softmax pool Triton kernel for compressor`
→ file `pr-draft-fused-softmax-pool-triton-kernel-for-compressor.md`

### Format

The draft file is pure HackMD markdown — **no YAML frontmatter**. It contains only the PR body content ready to copy-paste into the GitHub PR form.

### Rules

- **Always create the draft file** — no user confirmation needed for draft creation
- **Do NOT delete** draft files after PR creation — they accumulate as a record
- Multiple drafts can coexist in the directory

## Default Commit Tag

- If a `commit-msg` hook exists and enforces tags: use the hook's format.
- If no hook: use `[AMD]` as the default prefix.

## Commit Message Format

- `[Tag] <one sentence description>`
- No `Co-Authored-By` or any other trailers — forbidden by project convention.

## GitHub CLI Auth

The `GH_TOKEN` env var contains a fine-grained PAT that is blocked by the LMSYS Corp enterprise token lifetime policy (>366 days). **Always prefix `gh` commands with `GH_TOKEN=""`** to fall through to the OAuth token from `gh auth login` stored in `~/.config/gh/hosts.yml`.

```bash
# Correct
GH_TOKEN="" gh pr create --repo sgl-project/sglang ...
GH_TOKEN="" gh pr view ...
GH_TOKEN="" gh api ...

# Wrong — will fail with enterprise token lifetime error
gh pr create --repo sgl-project/sglang ...
```

This applies to all `gh` commands that target `sgl-project/sglang`. Commands targeting `yichiche/agent-box` may work with either token.

## Safety Rules

- NEVER force push unless the user explicitly asks
- NEVER amend a previous commit unless the user explicitly asks
- NEVER skip pre-commit hooks
- NEVER commit or push .env files, credentials, or secrets
- NEVER push to `main` or `master` directly (exception: agent-box allows it)
- NEVER create PRs to branches other than `main` unless the user explicitly asks
- ALWAYS confirm with the user before pushing (exception: `$HOME/agent-box` — standing
  authorization to commit on `main` and push to `origin` without asking)
- ALWAYS confirm the commit message with the user before committing (same agent-box exception)
