---
description: Bump an InferenceX (SemiAnalysisAI/InferenceX) benchmark config to the latest lmsysorg/sglang-rocm image, append the perf-changelog trigger, then commit-push a feature branch, raise an English-only PR to main, and apply the full-sweep-fail-fast sweep label. Handles the repo's non-standard git/gh quirks (repo often not checked out, origin-branch workflow, no pre-commit, REST-API label/body edits).
category: deliver
---

# InferenceX image-bump PR

End-to-end flow for updating an InferenceX benchmark config to a newer SGLang
ROCm image and shipping it as a PR. This is the repo-specific companion to
`/commit-push` — it prepares the changes, then chains `/commit-push` to branch +
commit + push, and finally raises the PR itself.

**Default action:** bump the target config's `image:` to the **latest
`lmsysorg/sglang-rocm` mi35x image**, append a `perf-changelog.yaml` trigger,
and open the PR. Reference implementation: PR #2201 (fp4 variant) and #2349
(fp8 variant).

## Repo facts (SemiAnalysisAI/InferenceX)

This repo is **NOT** in `_shared/repo-config.md`'s table. Its rules:

- **Root:** `/sgl-workspace/InferenceX` (detect with `git rev-parse --show-toplevel`).
- **Remote:** `origin` → `https://github.com/SemiAnalysisAI/InferenceX.git`. This is a
  **direct-branch** workflow (contributor push access), NOT a fork. Push the feature
  branch to `origin`; PR base is `SemiAnalysisAI/InferenceX:main`.
- **No pre-commit config** (`.pre-commit-config.yaml` absent; global hooksPath doesn't
  exist here). Do NOT run `pre-commit`. Validate with the repo's own tests instead
  (Step 4).
- **Commit tag:** `[AMD][MI35X]` style, matching recent history (e.g. PR #2201/#2349).
- **No `Co-Authored-By: Claude ...` trailer** on commits in this repo (standing user
  preference). Omit it when writing the commit — do not add it and then strip it later.
- **Sweep label is mandatory, not optional.** Every PR raised by this skill gets
  `full-sweep-fail-fast` applied right after creation (Step 7) — `run-sweep.yml` is
  gated on a label, so an unlabelled PR runs nothing.
- **AGENTS.md normally requires bilingual (English + 中文) PR titles/bodies.** The
  default for THIS skill is **English-only** PR title and body (per the user's standing
  preference). Mention the bilingual policy once so the user can opt back in, but do not
  add Chinese unless asked.

### gh / git auth quirks (load-bearing)

- Prefix every `gh` command with `GH_TOKEN=""` so it uses the OAuth token from
  `gh auth login` (the `GH_TOKEN` PAT is blocked by an enterprise lifetime policy).
- `git push` over HTTPS has no cached creds — push with the gh credential helper:
  ```bash
  GH_TOKEN="" git -c credential.helper='!gh auth git-credential' push -u origin <branch>
  ```
- **`gh pr edit` fails** on a projects-classic GraphQL deprecation and silently leaves
  the body unchanged. To edit an existing PR's body/title, use the REST API:
  ```bash
  GH_TOKEN="" gh api -X PATCH repos/SemiAnalysisAI/InferenceX/pulls/<num> \
    -f body="$(cat <draft-file>)"
  ```
  `gh pr create` (Step 6) works fine — only edits need the REST path.

## Step 0: Make sure the repo is checked out

`/sgl-workspace/InferenceX` is **frequently absent** — most SGLang dev containers only
mount `sglang`/`aiter`. Check first, and clone if missing:

```bash
[ -d /sgl-workspace/InferenceX/.git ] || \
  (cd /sgl-workspace && GH_TOKEN="" git -c credential.helper='!gh auth git-credential' \
    clone https://github.com/SemiAnalysisAI/InferenceX.git)
```

Do **not** reuse a stray checkout found elsewhere on the box (e.g. under
`/data/*/…/source/InferenceX`) — those belong to other users' runs and git rejects them
with `dubious ownership`. Clone your own copy.

If the repo was already present, `git checkout main && git pull` before branching so the
new branch isn't based on a stale `main`.

## Step 1: Identify the target config and image

From `$ARGUMENTS` (or by asking), determine:

- **Config key(s)** to bump — e.g. `qwen3.5-fp8-mi355x-sglang` and its `-mtp` sibling.
  Bump the non-MTP + MTP pair together unless told otherwise (matches #2201/#2349).
  **Precision naming:** the user says "mxfp4"/"MXFP4"; the config keys say **`fp4`**
  (`qwen3.5-fp4-mi355x-sglang`, model `amd/Qwen3.5-397B-A17B-MXFP4`). Likewise "fp8" →
  `qwen3.5-fp8-…`. Don't grep for `mxfp4` in the config keys — you'll find nothing.
  Also ignore the `-agentic-mtp`, `-atom`, and `-disagg` siblings unless asked: the
  default pair is exactly `<model>-<prec>-<hw>-sglang` and `…-sglang-mtp`.
- **Target image tag.** Default = the latest `lmsysorg/sglang-rocm:vX.Y.Z-rocm720-mi35x-YYYYMMDD`.

  Grepping the repo only shows what's *already in use* — usually weeks stale — so use it
  for context, not as the source of truth:
  ```bash
  grep -rhoE 'lmsysorg/sglang-rocm:v[0-9.]+(post[0-9]+)?-rocm[0-9]+-mi35x-[0-9]{8}' \
    configs/ perf-changelog.yaml | sort -t- -k4 | uniq -c | tail -10
  ```
  For the genuinely newest tag, query Docker Hub. Note the API's `name=` filter returns
  nothing useful here — page through `ordering=last_updated` and filter client-side:
  ```bash
  for p in 1 2 3 4; do
    curl -s -m 60 "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags?page_size=100&page=$p&ordering=last_updated"
    echo
  done | python3 -c "
import sys, json
for line in sys.stdin:
    if not line.strip(): continue
    for t in json.loads(line).get('results', []):
        if 'mi35x' in t['name']: print(t['name'], t['last_updated'])
" | sort -u | tail -15
  ```
  Both `rocm700-` and `rocm720-` variants are published daily — the mi355x configs use
  **`rocm720`**; match whatever the config currently pins.

  Confirm the exact tag with the user (they usually supply it, e.g.
  `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260726`). Do not invent a tag. If they
  don't supply one, offer the choice explicitly: newest-on-Docker-Hub (freshest, untested
  in this repo) vs. the newest tag already referenced by another config (known-good).

Show the current `image:` line for each target key (`grep -n "<config-key>:" -A1 configs/amd-master.yaml`)
and the tag you'll set, then proceed.

## Step 2: Bump the image in `configs/amd-master.yaml`

For each target config key, replace its `image:` value with the new tag. Edit the
value only — keep `model`, `runner`, search-space, etc. untouched. The `image:` line
sits directly under the `<config-key>:` header, so anchor the edit on the header +
image line to stay unique.

## Step 3: (Optional) benchmark-script env changes

Only if the user asks (e.g. #2201 added `SGLANG_MAMBA_SSM_DTYPE=bfloat16`). The scripts
live at `benchmarks/single_node/fixed_seq_len/<model>_<prec>_<hw>.sh` (+ `_mtp.sh`).
Add/remove `export` lines after the existing `SGLANG_USE_AITER*` block. **Default: no
script change** — a plain image bump touches only YAML.

## Step 4: Append the perf-changelog trigger

`perf-changelog.yaml` is **append-only, read oldest→newest**. Append the new entry at
the very END — never insert or prepend, and don't touch existing whitespace (the
deletion check is whitespace-sensitive). Entry shape:

```yaml
- config-keys:
    - qwen3.5-fp8-mi355x-sglang
    - qwen3.5-fp8-mi355x-sglang-mtp
  description:
    - "Bump image from <old-tag> to <new-tag>"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/<PR_NUM>
```

`<PR_NUM>` is unknown until Step 6. Put a placeholder (the reference PR, e.g. #2201),
then fix it in Step 8 after the PR exists.

Append with a heredoc (`cat >> perf-changelog.yaml <<'EOF'`) rather than an Edit on the
last existing entry — the file ends with `\n\n` and editing near the tail risks eating
that trailing blank line, which the whitespace-sensitive deletion check flags.

## Step 5: Validate

No pre-commit here — run the repo's own checks:

```bash
python -m pytest utils/matrix_logic/ -q     # expect ~231 passed
```

The generator writes its JSON matrix to **stdout** and progress chatter to stderr, so
`2>&1 | grep -c` miscounts. Redirect stdout to a file and count per arm instead:

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/amd-master.yaml \
  --model-prefix <prefix> --precision <prec> --runner-type <hw> \
  > /tmp/sweep.json
python3 -c "
import json
from collections import Counter
d = json.load(open('/tmp/sweep.json'))
items = d if isinstance(d, list) else d.get('include', d)
c = Counter((i.get('image'), i.get('spec-decoding')) for i in items)
for k, v in sorted(c.items(), key=lambda x: str(x[0])): print(v, k)
"
```

Confirm the tests pass and the new tag appears on **both** the non-MTP
(`spec-decoding: none`) and MTP (`spec-decoding: mtp`) arms. Other frameworks for the
same model (e.g. `-atom`) legitimately still show their own image — only the two
`sglang` arms should have moved.

## Step 6: Commit, push, and open the PR

1. **Chain `/commit-push`** to create the feature branch (from `main`), commit, and
   push to `origin`. Suggested branch name: `amd/<model>-<prec>-<hw>-sglang-<version>`
   (e.g. `amd/qwen3.5-fp8-mi355x-sglang-v0.5.16`). Commit subject:
   `[AMD][MI35X] Bump <model> <hw> SGLang single-node image to <version>`.
   `/commit-push` will confirm the push target with the user before pushing.

   > `/commit-push` uses the standard repo-config table and **will not find this repo**.
   > Expect to do it inline: `git checkout -b <branch>`, stage the changed files by
   > name, commit, then push with the credential-helper command from the auth-quirks
   > section above.

   Write the commit **without** a `Co-Authored-By` trailer (see Repo facts). If one slips
   in, don't try `git rebase -i` (unsupported here) — reset to `main` and replay:
   ```bash
   git reset --hard origin/main
   git cherry-pick <sha1>            # NOTE: cherry-pick has no -q flag; use >/dev/null
   git commit --amend -q -m "<subject>" -m "<body>"
   git cherry-pick <sha2> >/dev/null
   git commit --amend -q -m "<subject2>"
   git diff --quiet <old-head> HEAD && echo "tree identical"   # verify before pushing
   GH_TOKEN="" git -c credential.helper='!gh auth git-credential' \
     push --force-with-lease origin <branch>
   ```
   After a force-push, `gh api …/pulls/<num>/commits` serves a **stale cached response**
   for a few seconds — re-query with `-H "Cache-Control: no-cache"` before believing the
   old SHAs are still there.

2. **Create the PR** (English-only) with a HackMD draft body. Write the draft to
   `$HOME/pr-drafts/pr-draft-<slug>.md` (Motivation / Modifications / Accuracy Tests /
   Benchmarking sections; one unwrapped line per paragraph — GitHub hard-breaks single
   newlines). Then:
   ```bash
   GH_TOKEN="" gh pr create \
     --repo SemiAnalysisAI/InferenceX --base main --head <branch> \
     --title "[AMD][MI35X] Bump <model> <hw> SGLang single-node image to <version>" \
     --body-file "$HOME/pr-drafts/pr-draft-<slug>.md"
   ```
   Capture the printed PR URL / number.

## Step 7: Apply the `full-sweep-fail-fast` label (required)

`run-sweep.yml` is gated on a sweep label — an unlabelled PR benchmarks nothing, so the
image bump can't be evaluated. Apply the label immediately after `gh pr create`; this is
part of the flow, not an offer to the user.

```bash
GH_TOKEN="" gh api -X POST \
  repos/SemiAnalysisAI/InferenceX/issues/<PR_NUM>/labels \
  -f 'labels[]=full-sweep-fail-fast'
```

Use the REST call above directly — `gh pr edit <num> --add-label` hits the same
projects-classic GraphQL deprecation as `gh pr edit --body` and fails. Verify:

```bash
GH_TOKEN="" gh api -H "Cache-Control: no-cache" \
  repos/SemiAnalysisAI/InferenceX/issues/<PR_NUM>/labels --jq '.[].name'
```

Use a different sweep label only if the user names one.

## Step 8: Point the changelog at the real PR

Update the Step-4 placeholder `pr-link` to the actual PR number, commit, and push:

```bash
# edit perf-changelog.yaml: pr-link .../pull/<PR_NUM>
git add perf-changelog.yaml
git commit -m "[AMD][MI35X] Point <model> changelog entry to PR #<PR_NUM>"
GH_TOKEN="" git -c credential.helper='!gh auth git-credential' push origin <branch>
```

Anchor the placeholder→real swap on the **full appended block** (the description line
plus its `pr-link`), not on the bare `pull/<placeholder>` string — the placeholder PR
number also appears in older entries, and a bare replace would rewrite history.

(If the PR body itself ever needs editing later, use the **REST API PATCH** — not
`gh pr edit`, per the auth-quirks section.)

## Step 9: Report

Show the user:
- Branch + the commit hashes pushed to `origin`.
- The PR URL.
- That `full-sweep-fail-fast` is applied and the sweep is gated on it (Step 7).
- Whether the repo had to be cloned in Step 0, so they know where it landed.
- Note that the PR body is English-only; AGENTS.md's bilingual policy may be flagged by
  a reviewer — offer to add a `## 中文说明` translation if the user wants.
