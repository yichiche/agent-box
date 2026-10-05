---
name: ci-analysis
description: >-
  Analyze GitHub Actions CI failures for a pull request: trace root-cause jobs vs
  fast-fail cascades (check-pr-test-health), fetch job logs with gh, attribute each
  failure to the PR vs pre-existing/flaky/infra, look up known issues by error signature
  (time-bounded), check whether main already fixed the issue, and produce merge verdict
  plus resolution tables with Related (yes/no) and Action (re-run/merge main/code fix).
  Use when the user pastes a GitHub Actions job URL, asks to check CI status, asks
  whether red CI is caused by their PR, sees a check-pr-test-health failure, wants
  /ci-analysis, or requests CI failure triage.
---

# CI Analysis

Systematically triage PR CI: identify real failures, attribute them to the PR or not, check main for fixes, and deliver an actionable merge verdict.

## When to Use

- User pastes a GitHub Actions job/run URL (`.../actions/runs/.../job/...?pr=...`)
- User asks: "is this CI failure from my change?", "check CI for PR #N", `/ci-analysis`
- User wants the structured report format with merge verdict, failure tables, and fix steps

## Phase 1: Gather Context (parallel)

Run these concurrently:

```bash
# PR metadata + all checks
gh pr view <N> --repo <owner/repo> --json title,state,files,headRefName,baseRefName,statusCheckRollup,commits

# All jobs in the workflow run (extract run_id from job URL)
gh run view <RUN_ID> --repo <owner/repo> --json name,status,conclusion,headBranch,jobs,url

# Compact check table
gh pr checks <N> --repo <owner/repo>
```

From the job URL, extract:
- `owner/repo`, `run_id`, `job_id`, `pr=N`
- Job name (runner, shard, partition)

Record **changed files** from `gh pr view --json files` — this is the attribution baseline.

## Phase 2: Pull Failure Evidence

For every **failed** job (start with the linked job, then all failed siblings):

```bash
# Job metadata
gh api repos/<owner>/<repo>/actions/jobs/<JOB_ID> \
  --jq '{name: .name, conclusion: .conclusion, started: .started_at, completed: .completed_at, steps: [.steps[] | {name: .name, conclusion: .conclusion}]}'

# Logs — grep for signal, not noise
gh api repos/<owner>/<repo>/actions/jobs/<JOB_ID>/logs 2>/dev/null | rg \
  "Test passed|Test failed|Failed test files|❌|✅|FAILED |AssertionError|UnicodeDecodeError|Traceback|short test summary|Fast-fail|SIGTERM|CUDA error|Memory access fault|Process completed with exit code" \
  | head -80
```

For each failed job, extract:
1. **Exact failed test file(s)** — look for `Failed test files`, `❌ Test failed:`, `FAILED`
2. **Error type + message** — first root-cause line, not cascade SIGTERM
3. **Test sequence** — `Running test:` / `Test passed:` to see if failure is isolated
4. **Shard assignment** — `AITER_TEST=`, `--auto-partition-id`, partition model entries
5. **Fast-fail** — if job skipped/failed in <30s with "Fast-fail: root cause job(s): ..."

Batch failed jobs in one shell loop when there are many shards.

## Phase 2.5: Root Cause Analysis (mandatory)

**Do this before attributing failures.** Many red jobs never ran tests — they were killed by upstream fast-fail. Fixing or analyzing a cascade job wastes time.

### Step 1: Classify every failed job

Pull the failed **step** name from job metadata:

```bash
gh api repos/<owner>/<repo>/actions/jobs/<JOB_ID> \
  --jq '{name: .name, conclusion: .conclusion, duration_sec: ((.completed_at | fromdateiso8601) - (.started_at | fromdateiso8601)), failed_step: ([.steps[] | select(.conclusion == "failure") | .name] | first)}'
```

Assign each failed job to exactly one bucket:

| Bucket | How to recognize | Action |
|--------|------------------|--------|
| **Root cause** | Failed step is `Run test` / actual work step; ran ≥1 min; has real test error in logs | Analyze test failure; may be PR-related |
| **Cascade (fast-fail)** | Failed step is `Run ./.github/actions/check-pr-test-health` or log contains `Fast-fail: skipping — root cause job(s):` | **Do not analyze this job's tests** — find root job(s) from log |
| **Rollup gate** | Names like `finish`, `pr-test-finish`, `Standard Test Results`, `call-gate / pr-gate` | Report as aggregation only |
| **Skipped downstream** | `conclusion: skipped` because upstream red | Not a failure to fix |

### Step 2: `check-pr-test-health` = cascade, not root cause

When the failed step is **`Run ./.github/actions/check-pr-test-health`**:

> This job **never actually ran tests**. CI killed it early because an upstream job in the same workflow (or a sibling workflow the health check watches) already failed. **Find and fix the root-cause job first** — rerunning or debugging this job alone will not help.

Extract the cited root job(s) from logs:

```bash
gh api repos/<owner>/<repo>/actions/jobs/<JOB_ID>/logs 2>/dev/null | rg \
  "Fast-fail|root cause job|check-pr-test-health|skipping — root"
```

Typical log lines:
- `Fast-fail: skipping — root cause job(s): base-c-test-4-gpu-b200 / base-c-test-4-gpu-b200 (2)`
- `Fast-fail: root cause job(s): wait-for-base-b, base-b-test-1-gpu-small (4)`

Then open **those** job IDs and analyze their `Run test` step failures.

### Step 3: Build the root-cause tree (internal only)

Use this tree **during analysis** to trace cascades back to root jobs. **Do not include cascade or rollup jobs in the user-facing report** — only root-cause jobs go in **Root cause summary**.

```
Root cause (1–N independent):
  └─ base-c-test-4-gpu-b200 (2)  →  test_collectives.py  →  CUDA devices busy
```

If the user linked a cascade job, mention in one sentence that it never ran tests and redirect to the root job — do not list cascade/rollup job names as separate fix items.

**Count rule:** "N executed failures" = root-cause jobs only. Cascade + rollup jobs are **never** counted as additional PR regressions and **never** appear in Root cause summary.

### Step 4: Cross-workflow root causes

Some health checks watch jobs in **other workflow runs** on the same PR (e.g. sglang `call-gate`, `wait-for-stage-b-amd`). If `check-pr-test-health` fails but no root job is named in the same run:

```bash
# All failed checks across the PR
gh pr checks <N> --repo <owner/repo> 2>&1 | rg "fail"

# Find jobs that ran tests (long duration, failed at "Run test")
gh pr view <N> --repo <owner/repo> --json statusCheckRollup \
  --jq '.statusCheckRollup[] | select(.conclusion == "FAILURE") | .name'
```

Prioritize failed jobs where:
- Duration > 5 minutes, OR
- Failed step ≠ `check-pr-test-health`, OR
- Log contains `Test failed` / `FAILED` / `AssertionError`

### Step 5: Bypass label caveat

If log says `Skipping jobs-failed check (bypass-fastfail label present)`, the health check was intentionally skipped — look for real test failures elsewhere; do not treat health-check pass as full green.

## Phase 3: Check PR-Owned Tests

Search logs for tests that **match changed files** or new tests added in the PR:

```bash
gh api repos/<owner>/<repo>/actions/jobs/<JOB_ID>/logs 2>/dev/null \
  | rg "<new_test_file>|Test passed.*<new_test_file>|Test failed.*<new_test_file>"
```

If the PR's new test **passed** on relevant hardware (e.g. MI35X for AMD kernel PR), that is strong positive signal even when unrelated shards are red.

## Phase 4: Attribute Failures

For each executed failure, classify **Related** as `yes` or `no` only:

| Related | Criteria |
|---------|----------|
| **yes** | Failed test/file directly exercises changed code paths, symbols, or backends touched by the PR |
| **no** | Different subsystem, different backend (e.g. NVIDIA job for AMD-only PR), infra/profiler/runner, known upstream bug, or pre-existing flaky path |

When borderline (overlapping area but environmental), default to **no** and explain the caveat in **Why**; recommend **re-run** in **Action**.

**Rules of thumb:**
- Same test file as a PR-added test → yes
- Failure in unrelated directory/backend → no
- `UnicodeDecodeError` in torch profiler teardown after kernel passed → no (infra)
- GPU busy / SIGTERM after server teardown → no (resource cleanup)
- Fast-fail siblings → not independent failures; cite root job
- Skipped downstream jobs → consequence of upstream red, not new failures

## Phase 4.5: Known-Issue Lookup (by error signature — time-bounded)

**Goal:** Decide **Action** (`re-run`, `merge main`, `code fix`, `wait upstream`) by checking whether the same error already has an issue/PR fix — without searching one job at a time.

### Step 1: Dedupe error signatures across root-cause jobs

From Phase 2 logs, extract one **signature** per distinct failure (not per job):

| Signature part | Example |
|----------------|---------|
| Error class / tag | `DSLUserCodeError`, `TYPE_UNSTABLE_JOIN`, `AssertionError`, `UnicodeDecodeError` |
| Distinctive symbol or message fragment | `n_block_first`, `0.92 not greater than 0.92`, `devices busy` |
| Failing source file (if in traceback) | `flash_fwd_sm100.py`, `test_quark_mxfp4.py` |

**Do not search** on generic tokens alone (`failed`, `error`, `Traceback`, `exit code`).

If 3 jobs share the same signature → **one lookup**, not three.

**Budget:** max **3 issue searches** and **1 compare-to-main** per analysis. Skip lookup when Related is clearly **yes** (PR regression — Action is `code fix`).

### Step 2: One batched search per signature

Pick the 1–2 most distinctive terms and run **one** query:

```bash
# Issues (preferred — often documents root cause + fix version)
gh search issues --repo <owner/repo> "<distinctive_term_1> <distinctive_term_2>" --limit 5 \
  --json number,title,state,createdAt,closedAt,url

# If no hits, try the failing test filename or traceback path
gh search issues --repo <owner/repo> "<test_file_basename>" --limit 5 \
  --json number,title,state,createdAt,closedAt,url
```

Only open the **top 1–2** hits; read title + first comment block — do not paginate or scan every result.

```bash
gh issue view <NUM> --repo <owner/repo> --json number,title,state,body,closedAt \
  --jq '{number, title, state, closedAt, body: (.body | split("\n")[0:8] | join("\n"))}'
```

Also check whether main already contains the fix (reuse Phase 5 compare):

```bash
gh api repos/<owner>/<repo>/compare/<PR_HEAD>...main \
  --jq '.commits[] | select(.commit.message | test("<distinctive_term>"; "i")) | {sha: .sha[0:8], message: (.commit.message | split("\n")[0])}'
```

### Step 3: Map lookup result → Action

| Lookup result | Related | Action |
|---------------|---------|--------|
| Matching **closed** issue/PR documents same error + fix merged to main; PR branch lacks it | no | **merge main** |
| Matching issue documents upstream/dependency bug; fix not yet on PR branch | no | **re-run** (note: unblock after upstream pin/fix lands) |
| No issue; failure looks like threshold flake / GPU busy / profiler teardown | no | **re-run** |
| Failed test/file directly on PR changed paths; no known issue | yes | **code fix** |
| PR changed code but issue shows pre-existing infra bug on other PRs too | no | **re-run** |

**Action values (use exactly one per row):** `re-run` | `merge main` | `code fix` | `wait upstream`

- **merge main** — only when compare or issue shows fix commit exists on main and PR is behind
- **wait upstream** — external dependency / infra not fixable in PR (e.g. cutlass-dsl pin not yet bumped)
- **re-run** — flaky, environmental, or upstream fix expected soon; no PR code change needed
- **code fix** — PR regression; author must change this PR

### Worked lookup (sglang PR #34421 / B200 job)

Signature: `TYPE_UNSTABLE_JOIN` + `n_block_first` + `test_flash_attention_4.py`

```bash
gh search issues --repo sgl-project/sglang "TYPE_UNSTABLE_JOIN n_block_first" --limit 3 \
  --json number,title,state,url
# → #34433 (closed): cutlass-dsl 4.6.0 pin bug, fixed in 4.6.2
```

→ Related: **no** (FA4 CUTLASS DSL; PR touches GDN Triton on HIP). Action: **re-run** after cutlass-dsl pin bump on main (or **merge main** if pin fix already merged).

Second signature: `AssertionError` + `0.92 not greater than 0.92` + `test_quark_mxfp4.py`

```bash
gh search issues --repo sgl-project/sglang "gsm8k 0.92 quark mxfp4" --limit 3 --json number,title,state,url
# If no exact hit: check main for tolerance fixes (#34272 pattern) via compare
```

→ Related: **no** (Qwen3 MoE MXFP4 eval; PR touches Qwen3.5 GDN only; exact threshold boundary). Action: **re-run**.

## Phase 5: Check If main Already Fixed It

When failures look like known infra or test-harness bugs:

```bash
# Recent commits on failing test files
gh api "repos/<owner>/<repo>/commits?path=<test_file>&per_page=5" \
  --jq '.[] | "\(.sha[0:8]) \(.commit.message | split("\n")[0]) \(.commit.committer.date)"'

# PR branch vs main divergence
gh api repos/<owner>/<repo>/compare/<PR_HEAD>...main \
  --jq '{ahead_by: .ahead_by, behind_by: .behind_by, status: .status}'

# Does main contain a fix commit the PR lacks?
gh api repos/<owner>/<repo>/compare/<PR_HEAD>...main \
  --jq '.commits[] | select(.commit.message | test("<keyword>"; "i")) | .commit.message'

# Verify fix on main CI: find run at fix commit, check same shard
gh run list --repo <owner/repo> --branch main --workflow "<Workflow Name>" --limit 5
```

If main merged a fix **after** the PR branch last merged main:
- State clearly: "main has fix #XXXX; PR is N commits behind — merge/rebase main"

Read the fix diff to describe **how** main solved it (not just that it exists).

## Phase 6: Write the Report

Use this structure. Adapt section names to the repo (AMD/Other, MI35X/MI300X, etc.).

**Output rule:** Root cause summary lists **only** jobs that actually ran tests and failed (`Run test` step). Never list cascade (`check-pr-test-health`, `wait-for-*`), rollup (`pr-gate`, `*-finish`), or skipped downstream jobs there.

**Platform** — backend the job runs on (`nvidia`, `amd`, `npu`, `mlx`, `xpu`, `musa`, `cpu`). Derive from job/workflow name, not pytest function:
`nvidia` (b200/h100/h200/gb300/cuda gpu shards), `amd` (mi300/mi35x/rocm/*-amd*), `npu` (a2/a3/ascend), `mlx`, `xpu`, `musa`, `cpu`. Lowercase only.

**Root cause summary sort order:** group bullets under backend headings; include only backends that have failures. Order: **nvidia → amd → npu → mlx → xpu → musa → cpu → other**. Same order for rows in the Root Cause Failures table.

```markdown
CI Status for PR #<N>
Merge verdict: <one sentence — merge/no-merge + is red from this PR?>

Root cause summary
Root cause jobs (fix these):

**nvidia**
- <job name> — <test file>: <1-line error>

**amd**
- <job name> — <test file>: <1-line error>

**npu**  <!-- omit empty backend sections -->
- <job name> — <test file>: <1-line error>

Executed CI failure attribution: <N root-cause failures (M related)>

Root Cause Failures
| Job | Test File | Platform | Error | Related | Action | Why |
|-----|-----------|----------|-------|---------|--------|-----|
<!-- rows sorted: nvidia first, then amd, then other backends -->

Details / what to do before merge
- Fix root-cause job(s) first; cascade/rollup reds clear on rerun after roots are green
- <other actionable bullets; cite issue # if Phase 4.5 found one>

Positive signal: <PR tests that passed on correct hardware>
```

**Do not include** a Warning section, Changed files block, cascade-job inventory, or queued-job notes in the report — keep those internal unless the user explicitly asks.

If the user linked a cascade job (e.g. `check-pr-test-health`), add one sentence under Details: "The job you linked never ran tests — fix `<root job>` first." Do **not** add cascade or rollup jobs to Root cause summary.

### Resolution table (when user asks "how to solve")

| # | Failing job / test | Root cause | How to solve | Fix location | Effort | Blocks merge? |
|---|-------------------|------------|--------------|--------------|--------|---------------|

Include: rerun, merge main, code fix, separate issue, infra ticket.

## Attribution Checklist

Before marking Related = **no**, confirm:
- [ ] Identified root-cause jobs vs cascade (`check-pr-test-health`) vs rollup — internally, not in report
- [ ] If user linked a cascade job, redirected analysis to the cited root job
- [ ] Root cause summary contains **only** root-cause jobs (no cascade/rollup bullets), **grouped by backend** (nvidia → amd → others)
- [ ] Read actual failed test file name from **root-cause** job logs only
- [ ] Compared failed paths to PR `files` list
- [ ] Ran Phase 4.5 lookup for each **distinct error signature** (deduped; ≤3 searches)
- [ ] Assigned **Action** from lookup + main compare (`re-run` / `merge main` / `code fix` / `wait upstream`)
- [ ] Checked whether PR's own new tests ran and passed
- [ ] Distinguished root failure from SIGTERM/cascade within a test run
- [ ] Checked if main already has a fix the PR lacks

## Common Failure Patterns

| Pattern | Likely cause | Bucket | Related | Typical Action |
|---------|--------------|--------|---------|----------------|
| Failed step = `check-pr-test-health` | Upstream job failed; this job never ran tests | **Cascade** | — | (omit from table) |
| Log: `Fast-fail: root cause job(s): X` | Health check aborted; fix job X | **Cascade → root X** | — | Analyze X only |
| Job failed in <30s, no test output | Fast-fail or infra abort | **Cascade** | — | (omit from table) |
| `TYPE_UNSTABLE_JOIN` + `n_block_first` in FA4/CUTLASS | cutlass-dsl pin bug | **Root cause** | no | re-run / merge main |
| `UnicodeDecodeError` in `prof.key_averages()` | ROCm kineto profiler UTF-8 decode | **Root cause** | no | merge main |
| `CUDA-capable device(s) is/are busy` after server test | GPU not released after teardown | **Root cause** | no | re-run |
| `0.92 not greater than 0.92` (GSM8K threshold) | strict `assertGreater` at exact boundary | **Root cause** | no | re-run |
| `Memory access fault` in unrelated kernel | CK/CUDA kernel on runner | **Root cause** | no | re-run |
| `Standard Test Results` / `pr-test-finish` failed/skipped | Rollup gate | **Rollup** | — | (omit from table) |
| New unit test in PR fails | Direct regression | **Root cause** | yes | code fix |

## Important

- **Always run commands** — do not guess from the GitHub UI summary alone
- **Link the job URL** in the report
- **Root cause summary = root-cause jobs only** — grouped by backend (**nvidia → amd → npu → mlx → xpu → musa → cpu**); omit empty sections; never list cascade, rollup, or skipped jobs
- Classify cascade/rollup internally (Phase 2.5) but omit from the report unless the user explicitly asks for a full job inventory
- **One row per root-cause failure** in the Root Cause Failures table; **same backend sort order** as root cause summary
- **Platform** column = backend (`nvidia`, `amd`, `npu`, `mlx`, `xpu`, `musa`, `cpu`); derive from job name, not pytest function
- If user links a cascade job URL, **say so explicitly** in Details and redirect to the root job
- **Related** column uses `yes` / `no` only; **Action** uses `re-run` | `merge main` | `code fix` | `wait upstream`
- Phase 4.5: dedupe signatures, max 3 issue searches — never search per cascade job
- Cite issue `#NNNN` in **Why** or **Details** when lookup finds a match
- If user wants tables only, still include merge verdict at top

## Worked Examples

Abbreviated examples matching the expected output style.

### Example 1: Unrelated infra failures (ROCm/aiter PR #4491)

**Input:** Job URL for `Standard Tests (1 GPU) (MI35X, 8, 3)` on PR adding `gdr_decode_packed_bf16`.

**Merge verdict:**
> Do not merge on red — but none of the 5 failing shards are caused by this PR's GDR decode kernel. The PR's own test `op_tests/test_gdr_decode_packed_bf16.py` passed on MI35X shard 5 and MI300X shard 5.

**Failure table (sample row):**

| Job | Test File | Platform | Error | Related | Action | Why |
|-----|-----------|----------|-------|---------|--------|-----|
| MI35X shard 3 | `op_tests/test_aiter_add.py` | amd | `UnicodeDecodeError` in profiler `key_averages()` | no | merge main | Add kernel passed; ROCm kineto UTF-8 bug ([#4623](https://github.com/ROCm/aiter/pull/4623) on main); PR touches GDR decode only |

**Resolution:**

| # | Issue | How to solve |
|---|-------|--------------|
| 1 | profiler UTF-8 | **merge main** to pick up #4623 |
| 2 | `test_batch_prefill.py` GPU fault | **re-run** shard 6 |

---

### Example 2: GPU cleanup failure (sglang PR #31856)

**Input:** `base-c-test-4-gpu-b200 (2)` on AMD AITER FP8 Q unified-attention PR.

**Merge verdict:**
> Red is not this PR's fault. B200 partition failed on Kimi K3 collectives with GPUs busy after DeepSeek server teardown. PR test `test_aiter_fp8_q_unified_attention.py` passed on MI35X (11s).

**Failure table (sample row):**

| Job | Test File | Platform | Error | Related | Action | Why |
|-----|-----------|----------|-------|---------|--------|-----|
| B200 shard 2 | `test_collectives.py` | nvidia | `CUDA error: devices busy or unavailable` | no | re-run | Kimi collectives on NVIDIA; PR only changes AMD `aiter_backend.py` |

**Root cause summary:**
**nvidia**
- `base-c-test-4-gpu-b200 (2)` — `test_collectives.py`: CUDA devices busy after server teardown

**Positive signal:**
- `stage-b-test-1-gpu-small-amd-mi35x`: `test_aiter_fp8_q_unified_attention.py` ✅

---

### Example 3: `check-pr-test-health` cascade (user linked wrong job)

**Input:** User links `base-c-test-4-gpu-h100 (2)` which failed in 11s at step `Run ./.github/actions/check-pr-test-health`.

**Agent must NOT** analyze H100 (2) as a test failure. Instead:

1. Read job steps → failed step = `check-pr-test-health`, duration ~11s → **Cascade**
2. Grep logs → `Fast-fail: skipping — root cause job(s): base-c-test-4-gpu-b200 / base-c-test-4-gpu-b200 (2)`
3. Open B200 (2) → failed step = `Run test` → real failure in `test_collectives.py`
4. Tell user in Details: "The job you linked never ran tests. Fix B200 (2) first."

**Root cause summary:**
**nvidia**
- `base-c-test-4-gpu-b200 (2)` — `test_collectives.py`: CUDA devices busy

**Merge verdict:**
> Do not merge on red, but H100 (2) is **not** a separate failure — it was killed by fast-fail before tests ran. Only 1 root-cause job (B200 shard 2); unrelated to PR.

---

### Example 4: Minimal user prompt → agent actions

**User:** `check https://github.com/org/repo/actions/runs/123/job/456?pr=789`

**Agent steps:**
1. Parse: `org/repo`, run `123`, job `456`, PR `789`
2. `gh pr view 789 --json files,statusCheckRollup`
3. **Root cause pass:** for each failed job, check `failed_step` — if `check-pr-test-health`, extract root job from logs and skip test analysis on that job
4. Analyze **root-cause** jobs only → `Failed test files`, test errors
5. Loop all failed jobs from `gh pr checks 789`; bucket into root / cascade / rollup
6. Grep logs for PR-added test files on relevant (usually AMD) green jobs
7. Phase 4.5: dedupe error signatures → ≤3 `gh search issues` queries → assign Action
8. `gh api compare/<head>...main` when lookup or pattern suggests main fix
9. Output report: merge verdict + Root cause summary + Root Cause Failures table (with Related + Action)

---

### Example 5: Known upstream bug + flaky threshold (sglang PR #34421)

**Root cause summary:**
Root cause jobs (fix these):

**nvidia**
- `base-b-test-4-gpu-b200 (0)` — `test_flash_attention_4.py`: `DSLUserCodeError: n_block_first` type join → server exit -9

**amd**
- `stage-b-test-1-gpu-small-amd-mi35x` — `test_quark_mxfp4.py`: `AssertionError: 0.92 not greater than 0.92`

**Phase 4.5 lookups (2 signatures, not 2×N jobs):**
1. `gh search issues --repo sgl-project/sglang "TYPE_UNSTABLE_JOIN n_block_first"` → [#34433](https://github.com/sgl-project/sglang/issues/34433) (closed): cutlass-dsl 4.6.0 pin bug, fixed in 4.6.2
2. `gh search issues --repo sgl-project/sglang "gsm8k quark mxfp4"` → no exact hit; accuracy exactly at threshold → flaky boundary

**Failure table:**

| Job | Test File | Platform | Error | Related | Action | Why |
|-----|-----------|----------|-------|---------|--------|-----|
| `base-b-test-4-gpu-b200 (0)` | `test_flash_attention_4.py` | nvidia | `DSLUserCodeError: n_block_first` None vs Int32 | no | re-run | FA4 CUTLASS DSL compile ([#34433](https://github.com/sgl-project/sglang/issues/34433)); PR touches HIP GDN only |
| `stage-b-test-1-gpu-small-amd-mi35x` | `test_quark_mxfp4.py` | amd | `0.92 not greater than 0.92` | no | re-run | Qwen3 MoE MXFP4 eval; PR changes Qwen3.5 GDN; exact threshold flake |

Do **not** add `wait-for-base-b`, `wait-for-stage-b-amd`, `pr-gate`, or `*-finish` to root cause summary — cascade/rollup only.
