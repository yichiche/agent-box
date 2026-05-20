---
description: Create a GitHub Pull Request — collects accuracy/benchmark data, drafts in HackMD format for review before submitting
---

# Create a Pull Request

Follow these steps precisely to create a well-formatted PR.
Read `_shared/repo-config.md` for repo table, PR targets, PR draft naming convention, and safety rules.

## Step 0: Ensure `gh` CLI is available

Check that the GitHub CLI is the real one (v2.x+, not the pip `gh` v0.0.4):
```bash
gh --version 2>&1
```
If missing or wrong version, follow the install steps in `_shared/repo-config.md`.

## Step 1: Gather context

Run these in parallel:
- `git rev-parse --show-toplevel` to determine which repo this is
- `git status` to check for uncommitted changes
- `git branch --show-current` to get current branch
- `git log --oneline main..HEAD` to see all commits on this branch
- `git diff main...HEAD` to see all changes vs main
- `git remote -v` to see configured remotes
- Check if the branch has a remote tracking branch: `git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null`

If there are uncommitted changes, warn the user and ask if they want to commit first (suggest using `/commit`).

## Step 2: Confirm push target

Look up the repo in `_shared/repo-config.md` and confirm the target with the user using `AskUserQuestion`. Show them:

- **Push remote**: the remote name and URL
- **PR target repo**: the upstream repo the PR will be created against
- **PR base branch**: the branch the PR will target
- **Current branch**: the feature branch that will be pushed

Ask the user to confirm or change.

## Step 3: Push the branch

If the branch is not pushed or is behind the remote:

Find the matching remote from `git remote -v`, or add one.

**ALWAYS** confirm with the user before pushing.

```bash
git push -u <remote> <branch-name>
```

## Step 4: Analyze the changes

Read through all the diffs and commits to understand:
- What was the motivation / bug being fixed / feature being added?
- What specific files and code were modified?
- Does this change affect model accuracy (kernel changes, model forward code)?
- Does this change affect inference speed / performance?

## Step 5: Collect benchmark/profiling data

**BEFORE writing the PR draft**, interactively collect and confirm data with the user. This is a multi-round conversation — do NOT skip ahead to generating the draft.

### 5a: Propose what benchmark data to show

Based on the change analysis from Step 4, propose to the user what benchmark/profiling tables you think should appear in the PR.

Focus on **specific, concrete changes** — not general module-category overviews. Good table types:
- **Kernel replacement**: before/after kernel names and times for the specific kernels being replaced, with savings and % of total time (e.g., TTFT impact)
- **End-to-end performance**: latency, throughput before/after

Do NOT propose broad per-module breakdowns (e.g., "attention: X us, elementwise: Y us, gemm: Z us") — those are too high-level for a PR. The reader wants to see exactly which kernels changed and by how much.

Present your proposed table structure (column headers, what metrics) and ask the user to confirm or adjust using `AskUserQuestion`. The user may add, remove, or restructure tables.

### 5b: Collect benchmark data

After the user confirms what tables to show, ask them to paste or provide the raw benchmark data (logs, profiling output, xlsx file paths, numbers).

Clean and format the data into the agreed-upon markdown tables:
- Parse the raw data the user provides
- Organize into clear, readable tables with headers
- Include units (ms, tokens/s, %, etc.)
- Add before/after or baseline/optimized columns where applicable
- Round numbers appropriately

**Show the formatted tables to the user** and ask for confirmation using `AskUserQuestion`:
- **Approve**: tables are correct, move on to accuracy
- **Revise**: user provides corrections or additional data — update tables and re-confirm

Do NOT proceed to Step 5c until the user approves the benchmark tables.

If the user says benchmark data is not available yet, note "Pending — will be added after testing" and proceed to 5c.

### 5c: Collect accuracy data

After benchmark data is confirmed, repeat the same interactive process for accuracy:

1. Propose what accuracy data to show (e.g., accuracy score, invalid rate, loss comparison, token-level diffs).
2. Ask the user to confirm the table structure.
3. Collect the raw accuracy data from the user.
4. Format into markdown tables and **show to the user for confirmation**.
5. Wait for user approval before proceeding to Step 6.

If the user says accuracy data is not available yet, note "Pending — will be added after testing" and proceed to Step 6.

## Step 6: Write PR draft in HackMD format

**Always create a new draft file without asking for permission.**

### Determine the file name

Follow the naming convention in `_shared/repo-config.md`:
1. Remove `[AMD]` tag and series prefix (e.g., `amd/deepseek_v4 integration NN/N`) from the PR title
2. Take the remaining descriptive part
3. Lowercase, replace spaces with hyphens, remove special characters
4. File path: `$HOME/pr-drafts/pr-draft-<slug>.md`

Ensure the `pr-drafts` directory exists: `mkdir -p $HOME/pr-drafts`

**Each PR always gets its own new file** — never overwrite an existing draft.

### Write the draft

- **For sglang repo**: Read `_shared/pr-template-sglang.md` and use that template.
- **For agent-box or other repos**: Read `_shared/pr-template-simple.md` and use that template.

The draft file is **pure HackMD markdown** — no YAML frontmatter.

Fill in all sections with concrete content and the user-confirmed accuracy/benchmark tables from Step 5b and 5c.

## Step 7: User reviews PR draft

After writing the draft file, tell the user the file path and show them a summary. Ask the user to review using `AskUserQuestion`:

- **Approve**: proceed to submit the PR as-is
- **Edit**: the user will modify the draft, then re-confirm. Re-read the draft file after.
- **Cancel**: abort PR creation, keep the draft for later

Do NOT submit the PR until the user explicitly approves.

## Step 8: Submit the PR

After user approval, read the draft file for the PR body. Use the title and target from Step 2.

Create the PR using `gh pr create` or the GitHub API. If neither works, print the manual PR creation URL.

**Do NOT delete the draft file** — it stays as a record.

## Step 9: Report back

After the PR is created, show the user:
- The PR URL
- A summary of what was included
- The draft file location for reference
- For sglang: remind them about the review process (ping merge oncalls, get CODEOWNER approvals, trigger CI)
