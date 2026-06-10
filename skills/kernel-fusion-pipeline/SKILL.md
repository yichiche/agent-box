---
description: "End-to-end kernel fusion pipeline: compare two trace analysis Excel files, identify Tier-1 fusion opportunities (existing aiter fused ops OR custom Triton kernels — no aiter modifications), implement each fusion on the SGLang side in git worktrees, validate (accuracy + profiling + benchmark) with upfront GPU slot planning, and commit. Composes /compare-kernels, /implement-kernel, /validate, and /commit. Use when the user says '/kernel-fusion-pipeline' followed by two xlsx paths."
---

# Kernel Fusion Pipeline

Automated pipeline: trace comparison → Tier-1 fusion identification → **parallel implement in git worktrees** → **validate each worktree independently** → **commit-push-pr per worktree** (1 worktree = 1 PR).

Scripts live in `$SKILL_DIR/scripts/` where:

```bash
SKILL_DIR="$HOME/agent-box/skills/kernel-fusion-pipeline"
STATE_FILE="$HOME/.kernel-fusion-pipeline/state.json"
```

## Usage

```
/kernel-fusion-pipeline <file_A.xlsx> <file_B.xlsx> [options]
```

Options (optional, parse from args):
- `--server=~/run_qwen3.5_mxfp4_perf.sh`
- `--client=~/run_qwen3.5_inferencemax_client.sh`
- `--threshold=0.85`
- `--label=Qwen3.5-MXFP4`
- `--only=slug1,slug2` — implement only these fusions
- `--interactive` — allow AskUserQuestion (default: autonomous)
- `--resume` — continue from `$STATE_FILE`

If two xlsx paths are missing, ask once for them. Otherwise **do not ask** for config, GPUs, or fusion selection.

---

## Operating Principles

1. **Autonomous by default** — no `AskUserQuestion` unless `--interactive`, missing xlsx, or zero free GPU slots for TP.
2. **Plan GPUs once at start** — run `rocm-smi` in Step 0; assign all slots before any implement/validate. Never re-scan at validate time except after a wave completes (optional refresh).
3. **Notify, don't confirm** — print config + GPU plan + fusion→slot map; proceed.
4. **Default scope = all Tier-1** — ranked by savings; use `--only` to narrow.
5. **Worktree isolation** — edit code only inside worktrees; `$SGLANG_ROOT` is orchestrator-only.
6. **Parallel implement, wave validate** — implement all worktrees in parallel (Task subagents); validate by **wave** using pre-assigned slots (wave 0 runs up to `max_parallel` jobs at once).
7. **Fail-forward** — on implement/validate failure: debug up to 3 retries → revert worktree + skip → next fusion. Only stop entire pipeline if `--interactive` and user says abort, or zero GPU slots at start.
8. **PIPELINE_MODE** — when calling sub-skills, pass pre-built CONFIG (see bottom). Sub-skills must not re-ask.

---

## Sub-Skill Contract (PIPELINE_MODE)

When this pipeline invokes sub-skills, ALL of them receive `pipeline_mode: true`. This means:

- **NO `AskUserQuestion`** — ever, for any reason, in any sub-skill
- **NO `EnterPlanMode`** — Tier 1 changes are implemented directly
- **NO interactive confirmation** — profiling commands, accuracy checks, and results are auto-evaluated
- **NO deep-dive prompts** — `/compare-kernels` stops after extracting the Tier-1 list (skips Step 7)
- **Fail = skip + continue** — sub-skill failures mark the fusion as SKIP and return control to the pipeline

If a sub-skill's default behavior would ask the user, PIPELINE_MODE overrides it to auto-proceed or auto-skip.

**Terminal stop points** (pipeline exits cleanly, no AskUserQuestion):
- No Tier-1 fusions found → print Tier 2/3 list and exit
- `max_parallel == 0` → print occupied GPUs and exit
- All fusions failed after retries → print failure summary and exit

---

## Step 0: Context + GPU Plan (upfront)

### 0a: Detect SGLang root

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
echo "Active SGLang root: $SGLANG_ROOT"
```

### 0b: Parse xlsx paths

From args, extract two paths and verify they exist. If missing, ask once.

### 0c: Auto-discover validation config

Infer from xlsx path / dirname (first match wins):

| Path contains | Label | Threshold | Server | Client |
|---|---|---|---|---|
| `qwen3.5`, `mxfp4` | Qwen3.5-MXFP4 | 0.85 | `~/run_qwen3.5_mxfp4_perf.sh` | `~/run_qwen3.5_inferencemax_client.sh` |
| `dsv4`, `deepseek` | DSv4 | 0.88 | `~/run_dsv4.sh` | `~/run_dsv4_client.sh` |

CLI flags `--server`, `--client`, `--threshold`, `--label` override inference.

Parse from server script:

```bash
TP=$(grep -oE '(^|[[:space:]])--tp[[:space:]]+[0-9]+' "$SERVER_SCRIPT" | grep -oE '[0-9]+$' | head -1)
TP=${TP:-2}
PORT_BASE=$(grep -oE '(^|[[:space:]])--port[[:space:]]+[0-9]+' "$SERVER_SCRIPT" | grep -oE '[0-9]+$' | head -1)
PORT_BASE=${PORT_BASE:-8000}
MODEL_PATH=$(grep -oE '--model-path[[:space:]]+"[^"]+"' "$SERVER_SCRIPT" | head -1)
```

### 0d: GPU inventory + slot plan (mandatory, run once)

**Before any analysis or coding**, run the planner (must execute in the same environment as validation — docker shell if applicable):

```bash
python3 "$SKILL_DIR/scripts/plan_gpu_slots.py" \
  --tp "$TP" \
  --port-base "$PORT_BASE"
```

This reads `rocm-smi`, marks GPUs free when VRAM < 10GB and no `sglang.launch_server` holds them, builds contiguous TP groups (0,1 / 2,3 / …), and writes `$STATE_FILE`.

Example output when 8 GPUs free and TP=2:

```
Max parallel: 4 validate job(s) at once
Slot 0: HIP_VISIBLE_DEVICES=0,1  port=8000
Slot 1: HIP_VISIBLE_DEVICES=2,3  port=8001
...
```

**If `max_parallel == 0`**: stop and report which GPUs are occupied (from state). Do not start implement. Suggest freeing GPUs or lowering scope.

Print full config (notify only — do not confirm):

```
Kernel Fusion Pipeline
  SGLang:     $SGLANG_ROOT
  File A/B:   ...
  Server:     $SERVER_SCRIPT  (TP=$TP)
  Client:     $CLIENT_SCRIPT
  Threshold:  $THRESHOLD  Label: $LABEL
  GPU slots:  $MAX_PARALLEL parallel (TP=$TP)
  State:      $STATE_FILE
```

---

## Step 1: Compare Kernels

### 1a: Recategorize

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py --recategorize <file_A.xlsx> <file_B.xlsx>
```

### 1b–1d: Compare + Tier-1 extraction

Follow `/compare-kernels` Steps 1–6, then extract Tier-1 opportunities. Tier-1 now includes:
- **Existing aiter fused op** → Python-only dispatch change
- **Custom Triton kernel** → write a new Triton kernel to fuse multiple elementwise/small ops (no aiter modifications needed)

Verify aiter ops under `$HOME/aiter/aiter/ops/` or `/sgl-workspace/aiter/aiter/ops/`. For custom Triton fusions, verify the pattern exists in SGLang source. Drop invalid candidates.

Rescan `fused*.py` and `gated*.py` at runtime.

For each opportunity record: **block name**, **slug** (snake_case), **target op**, **kernels**, **savings**, **sglang file**.

If none found, report Tier 2/3 list and stop.

---

## Step 2: Fusion list + upfront slot assignment

### 2a: Print ranked Tier-1 list

Order by estimated savings. Default: **implement all** (unless `--only`).

### 2b: Assign fusions to GPU slots (upfront)

After slugs are known:

```bash
python3 "$SKILL_DIR/scripts/plan_gpu_slots.py" \
  --tp "$TP" --port-base "$PORT_BASE" \
  --assign slug1 slug2 slug3 ...
```

This writes `fusion_plan` to `$STATE_FILE`:

```
wave 0  sigmoid_gate_mul           → GPUs 0,1  port 8000
wave 0  fused_qk_gemma_rmsnorm...  → GPUs 2,3  port 8001
wave 1  next_fusion                → GPUs 0,1  port 8000  (reuse slot after wave 0 done)
```

**Execution rule:** finish all fusions in wave N before starting wave N+1 (same physical GPUs reused across waves).

---

## Step 1.5: Worktree farm

For each fusion slug:

```bash
python3 "$SKILL_DIR/scripts/worktree_farm.py" \
  --sglang-root "$SGLANG_ROOT" \
  --slugs slug1 slug2 ...
```

Worktrees: `$HOME/.kernel-fusion-pipeline/worktrees/<repo-name>/<slug>/` on branch `fusion/<slug>`.

Implement only inside the worktree path. Set `PYTHONPATH=$WORKTREE/python` for server launches.

---

## Step 3: Implement → Validate → Commit

### 3a: Parallel implement (all worktrees)

Launch Task subagents in parallel — one per fusion worktree. Each agent:

```
PIPELINE_MODE=1
Worktree: <path>
Tier 1: Switch <block> to <aiter_op> — file <path>
Skip EnterPlanMode. Follow sglang-backend-gated-changes. Max 3 lint/syntax retries.
```

Or inline Tier-1 edits without `/implement-kernel` plan approval.

On failure after 3 retries: mark worktree `status: skip`, continue others.

### 3b: Wave validate (use pre-assigned slots only)

For each **wave** in `fusion_plan`:

1. Group fusions in that wave (same wave index).
2. For each fusion in the wave, build ephemeral scripts from the **pre-assigned** `gpus` and `port` (do not re-run GPU planner):

```bash
# Ephemeral server script for this fusion's slot
EPHEMERAL_SERVER=$(mktemp /tmp/kfp-server-XXXX.sh)
sed -e "s/HIP_VISIBLE_DEVICES=[0-9,]*/HIP_VISIBLE_DEVICES=${GPUS}/" \
    -e "s/--port [0-9]*/--port ${PORT}/" \
    "$SERVER_SCRIPT" > "$EPHEMERAL_SERVER"
chmod +x "$EPHEMERAL_SERVER"

EPHEMERAL_CLIENT=$(mktemp /tmp/kfp-client-XXXX.sh)
sed -e "s/^PORT=[0-9]*/PORT=${PORT}/" \
    "$CLIENT_SCRIPT" > "$EPHEMERAL_CLIENT"
chmod +x "$EPHEMERAL_CLIENT"
```

3. Invoke `/validate` with PIPELINE_MODE CONFIG:

```json
{
  "pipeline_mode": true,
  "worktree": "<worktree_path>",
  "pythonpath": "<worktree_path>/python",
  "server_script": "<EPHEMERAL_SERVER>",
  "client_script": "<EPHEMERAL_CLIENT>",
  "port": <PORT>,
  "gpus": "<GPUS>",
  "label": "<LABEL>",
  "threshold": <THRESHOLD>
}
```

4. Run validations in the wave **in parallel** (one subprocess per fusion, each with its own GPUS+PORT from the plan).

5. After wave completes: kill servers on those ports, then optional `plan_gpu_slots.py` refresh if GPUs may have leaked.

**Server launch in worktree:**

```bash
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/python:${PYTHONPATH:-}"
bash "$EPHEMERAL_SERVER"
```

Poll health in foreground on `$PORT` (never background TaskOutput for health checks).

### 3c: Validation pass/fail (per worktree)

Each worktree is validated and benchmarked **independently**. The benchmark results for each worktree are used in that worktree's individual PR.

Pass = accuracy ≥ threshold, fused kernel visible in profile, benchmark not regressed >2% ITL.

Fail: debug in worktree (≤3 retries) → `git checkout -- .` in worktree → mark SKIP → continue wave.

### 3d: Commit-push-PR per worktree

**Each worktree that passes validation gets its own commit, push, and PR — never batch multiple fusions into one PR.**

After validation passes for a worktree:

1. **Commit** in the worktree:
   ```bash
   cd "$WORKTREE"
   Skill(commit): "[AMD] <slug>: <description> — fuse N kernels, ~X us/layer"
   ```

2. **Push + PR** from the worktree:
   ```
   Skill(commit-push-pr)
   ```
   Each PR body includes the **individual benchmark results** for that worktree's fusion only.

3. Record commit hash and PR URL in `$STATE_FILE` under `results[<slug>]`.

**Key rule**: 1 worktree = 1 branch = 1 commit = 1 benchmark = 1 PR. No combining.

### 3e: Cleanup worktree after PR

After `/commit-push-pr` succeeds for a worktree:

```bash
cd "$SGLANG_ROOT"
git worktree remove "$WORKTREE"
# Keep the local branch — it tracks the remote and may need fixes
```

**Worktrees are only needed during parallel implementation.** Once the code is committed and pushed, the worktree is a redundant checkout. All further work on that fusion uses the branch directly:

```bash
# To fix a PR after review feedback:
cd "$SGLANG_ROOT"
git checkout fusion/<slug>
# make edits, commit, push
git checkout main   # return when done
```

**Do NOT delete the local branch** until the PR is merged. GitHub deletes the remote branch on merge; clean up the local branch afterward if desired.

### 3f: Next wave

Repeat 3b–3e for wave 1, 2, … until all fusions processed.

---

## Step 4: Final summary

Print table: slug, branch, slot (GPUs/port), wave, status, commit, PR URL.

All worktrees should be cleaned up by this point (removed in Step 3e). Branches remain for PR tracking.

Include individual benchmark results per fusion. List Tier 2/3 not attempted.

Suggest: `/generate-profile`, `/compare-kernels`.

To fix a PR later: `git checkout fusion/<slug>`, edit, commit, push, `git checkout main`.

---

## PIPELINE_MODE contract (sub-skills)

When `pipeline_mode: true` in CONFIG:

### `/implement-kernel`
- Skip `EnterPlanMode` / plan approval for Tier 1.
- `SGLANG_ROOT` = worktree path if provided.
- No AskUserQuestion.

### `/validate`
- Skip Step 0b AskUserQuestion; use CONFIG scripts/port/threshold/label.
- `git -C worktree` for stash/revert baseline.
- `PYTHONPATH=worktree/python` for all server launches.
- Baseline: `git stash` in worktree if dirty, else `git checkout HEAD~1` if one commit ahead of base.

---

## Important Notes

- **Tier 1 = existing aiter ops OR custom Triton kernels** — no C++/HIP kernel work, no aiter modifications. Custom Triton fusions (e.g., fusing `sigmoid + mul` into one kernel) are Tier 1.
- **1 worktree = 1 PR** — each fusion gets its own worktree, its own benchmark, its own commit, and its own PR via `/commit-push-pr`. NEVER combine multiple fusions into a single PR.
- **GPU plan is upfront** — `plan_gpu_slots.py` once at Step 0; `--assign` after fusion list. Slots fixed per wave; no last-minute allocation.
- **TP2 example** — 8 free GPUs → 4 slots → up to 4 parallel validates in wave 0; 5th fusion waits for wave 1 on reused slots 0,1 @ port 8000.
- **Backend-gated** — `_use_aiter`, no top-level aiter imports, common path byte-identical.
- **Dynamic paths** — detect `$SGLANG_ROOT` from Python env; aiter under `$HOME/aiter` or `/sgl-workspace/aiter`.
- **Commits** — `[AMD]` prefix, no `Co-Authored-By`.
- **Resume** — read `$STATE_FILE`; skip fusions with `status: pass|skip`; continue from next pending wave.
