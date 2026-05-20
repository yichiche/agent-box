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
| `$HOME/sglang` | `https://github.com/yichiche/sglang` | `sgl-project/sglang` | `main` | No — must use feature branch |
| `$HOME/agent-box` | `https://github.com/yichiche/agent-box` | `yichiche/agent-box` | `main` | Yes — commit directly on main |

For any repo not listed above, ask the user for the remote URL, PR base repo, and base branch.

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

## Safety Rules

- NEVER force push unless the user explicitly asks
- NEVER amend a previous commit unless the user explicitly asks
- NEVER skip pre-commit hooks
- NEVER commit or push .env files, credentials, or secrets
- NEVER push to `main` or `master` directly (exception: agent-box allows it)
- NEVER create PRs to branches other than `main` unless the user explicitly asks
- ALWAYS confirm with the user before pushing
- ALWAYS confirm the commit message with the user before committing
