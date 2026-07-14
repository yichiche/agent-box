---
description: Commit, push, and create a PR in one flow — checks state at each stage, confirms push target, collects accuracy/benchmark data, drafts PR in HackMD format for review before submitting
---

# Commit, Push, and Create PR

Follow these steps precisely. This skill chains to `/commit-push` (which itself chains to `/commit`).
Read `_shared/repo-config.md` for repo table, remote URLs, PR draft location/naming, and safety rules.

## PIPELINE_MODE (from `/kernel-fusion-pipeline`)

When invoked with a CONFIG object containing `"pipeline_mode": true`, run **fully autonomously** —
NEVER use `AskUserQuestion`, never wait for human review. Skip all interactive steps (2, 4-review,
6, 8, 11). Use CONFIG fields instead of asking. This is the SAME commit/PR contract as the
interactive flow; only the "ask the user" gates are replaced by CONFIG values.

**CONFIG fields:**
- `worktree` — run all git/gh commands from here (`cd "$worktree"` or `git -C "$worktree"`).
- `repo` — base repo for the PR (e.g. `sgl-project/sglang`).
- `base` — PR base branch (e.g. `main`).
- `commit_subject` — exact commit subject (already `[AMD] …` formatted).
- `pr_title` — exact PR title.
- `pr_body_file` — path to a file with the full PR body (already composed; use VERBATIM).
- `update_pr` — optional PR number; if set, UPDATE that PR's body instead of creating a new one.
- `gh_env` — optional env prefix for gh, e.g. `GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN=""`.
- `sglang_root` — repo root to return to for worktree cleanup.

**Flow:**

1. `cd "$worktree"`.

2. **Fix lint BEFORE committing** (CI enforces pre-commit; a dirty lint state fails CI):
   - `pip3 install -q -U pre-commit` (fine if already present)
   - `git add -A`
   - `pre-commit run --all-files` — let hooks auto-fix (isort, ruff F401/F821, black, codespell,
     clang-format, nbstripout).
   - If files changed: `git add -A && pre-commit run --all-files`; repeat up to 3×.
   - If a hook still fails (e.g. ruff F821 it cannot auto-fix), read the error, fix the code in the
     worktree, then retry from `git add -A`.
   - If `--all-files` reformats files UNRELATED to this change, reset them
     (`git checkout -- <file>`) so the commit stays focused on this change only.
   - Do not proceed until pre-commit exits clean.

3. Commit (only if there are staged changes): `git commit -m "<commit_subject>"`.
   **Never add `Co-Authored-By` trailers** (in subject or body).

4. Push the branch: `git push` (use `git push -u origin HEAD` if no upstream is set).

5. PR — pick ONE path:
   - **Create** (no `update_pr`):
     `<gh_env> gh pr create --repo <repo> --base <base> --title "<pr_title>" --body-file <pr_body_file>`
     Use `--body-file` (never `--body`) so markdown tables survive shell quoting.
   - **Update existing** (`update_pr` set): `gh pr edit` currently fails on a projects-classic
     GraphQL deprecation, so update the body via the REST API:
     `<gh_env> gh api -X PATCH repos/<repo>/pulls/<update_pr> -f body="$(cat <pr_body_file>)"`
   - If `gh pr create` fails on auth/permission, fall through Step 9's Attempts 2-4
     (`--web`, `curl` REST POST, prefill URL). In pipeline_mode do NOT block on the user —
     record the failure and return `status: fail` with the reason.

6. Cleanup: `cd "$sglang_root"` then `git worktree remove <worktree>` (keep the local branch).

7. Return JSON: `{ slug, commit_hash, pr_url, status }` (`status` = `pass`/`fail`, plus `error` if failed).

---

## Step 0: Ensure `gh` CLI is available and authenticated

Check that the GitHub CLI is the real one (v2.x+, not the pip `gh` v0.0.4):
```bash
gh --version 2>&1
```
If missing or wrong version, follow the install steps in `_shared/repo-config.md`.

After `gh` is installed, check authentication:
```bash
gh auth status 2>&1
```
If not logged in, ask the user to run `! gh auth login` in the prompt (the `!` prefix runs the command in this session so its output lands directly in the conversation). Wait for the user to confirm they've completed the login flow before proceeding.

## Step 1: Gather full state

Run these in parallel:
- `git rev-parse --show-toplevel` to determine which repo this is
- `git status` to see uncommitted changes
- `git branch --show-current` to get the current branch
- `git diff --cached` to see staged changes
- `git diff` to see unstaged changes
- `git log --oneline -5` to see recent commits
- `git remote -v` to see configured remotes
- Check for existing PR: `gh pr view --json url,state 2>/dev/null`

## Step 2: Confirm push target

Before any commit or push, look up the repo in `_shared/repo-config.md` and confirm the target with the user using `AskUserQuestion`. Show them:

- **Push remote**: the remote name and URL (e.g., `origin` → `https://github.com/yichiche/sglang`)
- **PR target repo**: the upstream repo the PR will be created against (e.g., `sgl-project/sglang`)
- **PR base branch**: the branch the PR will target (e.g., `main`, `amd/deepseek_v4`)
- **Current branch**: the feature branch that will be pushed

Ask the user to confirm or change the push remote, PR target repo, or PR base branch.

## Step 3: Commit and push if needed

Based on the state from Step 1, determine what is already done:

- **If there are uncommitted changes**: Invoke the `/commit-push` skill (it will handle both commit and push). Wait for it to complete before proceeding to Step 4.
- **If the working tree is clean but the branch has unpushed commits**: Invoke the `/commit-push` skill (it will skip commit and only push). Wait for it to complete before proceeding to Step 4.
- **If the working tree is clean and the branch is already pushed and up-to-date**: Skip directly to Step 4.

To check if the branch is already pushed and up-to-date:
```bash
git rev-parse @{u} 2>/dev/null && git log --oneline @{u}..HEAD
```

## Step 4: Check for existing PR

Check if a PR already exists for this branch (from Step 1's `gh pr view`):

- **If a PR already exists**: Show the user the existing PR URL and ask if they want to update it or skip.
- **If no PR exists**: Proceed to Step 5.

## Step 5: Analyze the changes

Run:
```bash
git log --oneline <base-branch>..HEAD
git diff <base-branch>...HEAD
```

(Use the PR base branch confirmed in Step 2.)

Read through all diffs and commits to understand:
- What was the motivation / bug being fixed / feature being added?
- What specific files and code were modified?
- Does this change affect model accuracy?
- Does this change affect inference speed / performance?

## Step 6: Collect benchmark/profiling data

**BEFORE writing the PR draft**, interactively collect and confirm data with the user. This is a multi-round conversation — do NOT skip ahead to generating the draft.

### 6a: Propose what benchmark data to show

Based on the change analysis from Step 5, propose to the user what benchmark/profiling tables you think should appear in the PR.

Focus on **specific, concrete changes** — not general module-category overviews. Good table types:
- **Kernel replacement**: before/after kernel names and times for the specific kernels being replaced, with savings and % of total time (e.g., TTFT impact)
- **End-to-end performance**: latency, throughput before/after

Do NOT propose broad per-module breakdowns (e.g., "attention: X us, elementwise: Y us, gemm: Z us") — those are too high-level for a PR. The reader wants to see exactly which kernels changed and by how much.

Present your proposed table structure (column headers, what metrics) and ask the user to confirm or adjust using `AskUserQuestion`. The user may add, remove, or restructure tables.

### 6b: Collect benchmark data

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

Do NOT proceed to Step 6c until the user approves the benchmark tables.

If the user says benchmark data is not available yet, note "Pending — will be added after testing" and proceed to 6c.

### 6c: Collect accuracy data

After benchmark data is confirmed, repeat the same interactive process for accuracy:

1. Propose what accuracy data to show (e.g., accuracy score, invalid rate, loss comparison, token-level diffs).
2. Ask the user to confirm the table structure.
3. Collect the raw accuracy data from the user.
4. Format into markdown tables and **show to the user for confirmation**.
5. Wait for user approval before proceeding to Step 7.

If the user says accuracy data is not available yet, note "Pending — will be added after testing" and proceed to Step 7.

## Step 7: Write PR draft in HackMD format

Based on the analysis and collected data, draft the full PR content.

**Always create the draft file without asking for permission.**

### Determine the file name

Follow the naming convention in `_shared/repo-config.md`:
1. Remove `[AMD]` tag and series prefix (e.g., `amd/deepseek_v4 integration NN/N`) from the PR title
2. Take the remaining descriptive part
3. Lowercase, replace spaces with hyphens, remove special characters
4. File path: `$HOME/pr-drafts/pr-draft-<slug>.md`

Ensure the `pr-drafts` directory exists: `mkdir -p $HOME/pr-drafts`

### Write the draft

- **For sglang repo**: Read `_shared/pr-template-sglang.md` and use that template.
- **For agent-box repo**: Read `_shared/pr-template-simple.md` and use that template.
- **For other repos**: Use the simple template.

The draft file is **pure HackMD markdown** — no YAML frontmatter. It contains only the PR body content.

**Line formatting — do NOT hard-wrap paragraphs.** GitHub renders a single newline inside a PR body as a hard `<br>` line break, so manually wrapping prose at ~80 chars shows up as broken mid-sentence lines (and indented list continuations render as nested breaks). Write each paragraph and each list item as **one unwrapped line**, however long; separate blocks with a blank line. Only use a newline where you actually want a break (between list items, table rows, headings). Do not indent continuation text under a list item. (Editors may soft-wrap the long lines visually — that's fine; the file must contain no mid-sentence `\n`.)

Fill in all template sections with concrete content. Insert the user-confirmed accuracy and benchmark tables from Step 6b and 6c into the appropriate sections.

If the user provided `$ARGUMENTS`, use that as the PR title. Otherwise, craft a concise title (under 70 chars) starting with `[AMD]`.

## Step 8: User reviews PR draft

After writing the draft file, tell the user the file path and show them a summary of its content. Ask the user to review and confirm using `AskUserQuestion`:

- **Approve**: proceed to submit the PR as-is
- **Edit**: the user will modify the draft file themselves, then re-confirm. After the user says they're done editing, re-read the draft file to pick up their changes.
- **Cancel**: abort PR creation (leave the draft file in place for future use)

Do NOT submit the PR until the user explicitly approves.

## Step 9: Submit the PR

After user approval, read the draft file to get the PR body.

Use the PR title and target information confirmed in Step 2 (not from the file — the file has no frontmatter).

Try the following four fallbacks **in order**. Stop at the first one that succeeds.

### Attempt 1 — `gh pr create` (API)

```bash
gh pr create \
  --repo <base_repo> \
  --base <base_branch> \
  --head <head> \
  --title "<title>" \
  --body-file <draft-file>
```

Use `--body-file` (not `--body`) so the draft markdown is passed verbatim without shell-escaping issues.

If this succeeds: capture the printed PR URL and skip to Step 10.

If Attempt 1 fails with an **auth error** (e.g., "please run `gh auth login`", "token lifetime", "OAuth token", expired/invalid token), do NOT fall through to Attempt 2 immediately. Instead:

1. Tell the user the auth failed and ask them to run `! gh auth login` in the prompt.
2. Use `AskUserQuestion` to wait for the user to confirm they've completed the login flow.
3. Verify with `gh auth status`.
4. **Retry Attempt 1** with `gh pr create`. Only proceed to Attempt 2 if this retry also fails.

### Attempt 2 — `gh pr create --web` (browser prefill)

Trigger this when Attempt 1 fails with **permission errors** (HTTP 403, "Resource not accessible by integration", "GraphQL: ...") even after successful auth. This is common when the API token can push to the user's fork but does not have write access to `<base_repo>`.

```bash
gh pr create \
  --repo <base_repo> \
  --base <base_branch> \
  --head <head> \
  --title "<title>" \
  --body-file <draft-file> \
  --web
```

`gh` opens a browser tab pointing at the GitHub compare page with `?expand=1&title=...&body=...` already filled in, using the user's logged-in session instead of the API token. Tell the user to review and click **Create pull request**.

### Attempt 3 — Direct GitHub API via `curl`

Trigger this when `gh` is not installed at all (`command -v gh` returns nothing). Build a JSON payload from the draft file and POST to the REST API:

```bash
jq -Rn --arg title "<title>" \
       --arg head  "<head>" \
       --arg base  "<base_branch>" \
       --rawfile body <draft-file> \
       '{title: $title, head: $head, base: $base, body: $body}' \
| curl -sS -X POST \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    --data-binary @- \
    https://api.github.com/repos/<base_repo>/pulls
```

If this returns the PR JSON: extract `.html_url` and skip to Step 10.

### Attempt 4 — Print a prefill URL for manual creation

Trigger this when none of the above work (no `gh`, and `curl` returns 403/422 because the token has no write access to `<base_repo>`).

Build a manual creation URL that prefills the title and body via query string parameters:

1. **URL-encode the title and body**. Use Python (always available) — never try to escape by hand:

   ```bash
   URLENC_TITLE=$(python3 -c "import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" "<title>")
   URLENC_BODY=$(python3 -c "import sys,urllib.parse; print(urllib.parse.quote(sys.stdin.read(), safe=''))" < <draft-file>)
   ```

   Important: pass `safe=''` so `/`, `?`, `&`, `#`, `=`, newlines, etc. all get percent-encoded. Otherwise the body will break the query string.

2. **Assemble the URL**:

   ```
   https://github.com/<base_repo>/compare/<base_branch>...<fork-owner>:<branch>?expand=1&title=<URLENC_TITLE>&body=<URLENC_BODY>
   ```

   `expand=1` forces GitHub to render the PR creation form (instead of just the diff view).

3. **URL length guard**. Real-world browsers and GitHub's frontend choke on extremely long URLs. Check the length **before** printing:

   ```bash
   FULL_URL="https://github.com/<base_repo>/compare/<base_branch>...<fork-owner>:<branch>?expand=1&title=${URLENC_TITLE}&body=${URLENC_BODY}"
   if [ ${#FULL_URL} -le 7000 ]; then
     echo "$FULL_URL"
   else
     echo "Body too long to prefill via URL. Open this page and paste the body manually:"
     echo "https://github.com/<base_repo>/compare/<base_branch>...<fork-owner>:<branch>?expand=1&title=${URLENC_TITLE}"
     echo "Body content is in: <absolute-path-to-draft-file>"
   fi
   ```

   When the URL would exceed 7000 characters, fall back to: print the plain compare URL (with title still prefilled if it fits, otherwise no query string) **and** the absolute path to the draft file so the user can copy-paste the body.

**Do NOT delete the draft file** — it stays as a record regardless of which attempt succeeded.

## Step 10: Report back

After the PR is created, show the user:
- The PR URL
- A summary of the full flow: what was committed, where it was pushed, and the PR link
- The draft file location for reference
- For sglang: remind them about the review process (ping merge oncalls, get CODEOWNER approvals, trigger CI)

## Step 11: Offer to return to the base branch

After reporting, ask the user with `AskUserQuestion` whether they want to switch back to the base integration branch (e.g., `amd/deepseek_v4`). This is useful when the PR was submitted from a feature branch and the user wants to continue working on the next task from the integration branch.

Options:
- **Yes — checkout `<base_branch>`**: Run `git checkout <base_branch>` to return to the integration branch. Show the resulting `git log --oneline -3` so the user can confirm they're at the right commit.
- **No — stay on `<current_branch>`**: Do nothing, keep the current feature branch checked out.

Only offer this step when the current branch is different from the PR base branch (i.e., the user is on a feature branch, not already on the base).
