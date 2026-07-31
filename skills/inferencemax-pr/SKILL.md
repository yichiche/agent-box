---
description: Bump an InferenceX (SemiAnalysisAI/InferenceX) benchmark config to the latest lmsysorg/sglang-rocm image, append the perf-changelog trigger, then commit-push a feature branch and raise an English-only PR to main. Handles the repo's non-standard git/gh quirks (origin-branch workflow, no pre-commit, REST-API body edits).
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

## Step 1: Identify the target config and image

From `$ARGUMENTS` (or by asking), determine:

- **Config key(s)** to bump — e.g. `qwen3.5-fp8-mi355x-sglang` and its `-mtp` sibling.
  Bump the non-MTP + MTP pair together unless told otherwise (matches #2201/#2349).
- **Target image tag.** Default = the latest `lmsysorg/sglang-rocm:vX.Y.Z-rocm720-mi35x-YYYYMMDD`.
  To find the newest tag currently referenced in the repo:
  ```bash
  grep -rhoE 'lmsysorg/sglang-rocm:v[0-9.]+(post[0-9]+)?-rocm720-mi35x-[0-9]{8}' \
    configs/ perf-changelog.yaml | sort -t- -k4 | uniq | tail -5
  ```
  Confirm the exact tag with the user (they usually supply it, e.g.
  `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260726`). Do not invent a tag.

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
then fix it in Step 7 after the PR exists.

## Step 5: Validate

No pre-commit here — run the repo's own checks:

```bash
python -m pytest utils/matrix_logic/ -q
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/amd-master.yaml \
  --model-prefix <prefix> --precision <prec> --runner-type <hw> \
  2>&1 | grep -c "<new-tag>"
```

Confirm the tests pass and the generator emits the new tag for both the non-MTP
(`spec-decoding: none`) and MTP (`spec-decoding: mtp`) arms.

## Step 6: Commit, push, and open the PR

1. **Chain `/commit-push`** to create the feature branch (from `main`), commit, and
   push to `origin`. Suggested branch name: `amd/<model>-<prec>-<hw>-sglang-<version>`
   (e.g. `amd/qwen3.5-fp8-mi355x-sglang-v0.5.16`). Commit subject:
   `[AMD][MI35X] Bump <model> <hw> SGLang single-node image to <version>`.
   `/commit-push` will confirm the push target with the user before pushing.

   > If `/commit-push` uses the standard repo-config and can't find this repo, fall
   > back to doing it inline: `git checkout -b <branch>`, stage the changed files by
   > name, commit, then push with the credential-helper command from the auth-quirks
   > section above.

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

## Step 7: Point the changelog at the real PR

Update the Step-4 placeholder `pr-link` to the actual PR number, commit, and push:

```bash
# edit perf-changelog.yaml: pr-link .../pull/<PR_NUM>
git add perf-changelog.yaml
git commit -m "[AMD][MI35X] Point <model> changelog entry to PR #<PR_NUM>"
GH_TOKEN="" git -c credential.helper='!gh auth git-credential' push origin <branch>
```

(If the PR body itself ever needs editing later, use the **REST API PATCH** — not
`gh pr edit`, per the auth-quirks section.)

## Step 8: Report

Show the user:
- Branch + the commit hashes pushed to `origin`.
- The PR URL.
- Reminder: the PR has **no sweep label yet** — `run-sweep.yml` is gated on one.
  `full-sweep-fail-fast` is the recommended default for an image bump. Offer to apply it
  (`GH_TOKEN="" gh pr edit <num> --add-label full-sweep-fail-fast` may hit the
  projects-classic error; if so use
  `GH_TOKEN="" gh api -X POST repos/SemiAnalysisAI/InferenceX/issues/<num>/labels -f labels[]=full-sweep-fail-fast`).
- Note that the PR body is English-only; AGENTS.md's bilingual policy may be flagged by
  a reviewer — offer to add a `## 中文说明` translation if the user wants.
