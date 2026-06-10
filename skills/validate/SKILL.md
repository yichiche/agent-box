---
description: Validate SGLang code changes — baseline benchmark, accuracy test, profiling, and after benchmark with before/after comparison. Use when the user says "validate the result" or "/validate" after making SGLang changes.
---

# Validate SGLang Changes

End-to-end validation of code changes in SGLang. Collects baseline benchmark (with change reverted), then runs accuracy, profiling, and after benchmark (with change applied). Produces a before/after comparison table ready for PR submission.

## Step 0: Gather context

### 0a: Detect the active SGLang installation

Before doing anything else, determine which SGLang repo is in use. There may be multiple SGLang directories on the system (e.g., `/sgl-workspace/sglang`, `$HOME/sglang`). **Always use the one that's currently installed in the active Python environment.**

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
echo "Active SGLang root: $SGLANG_ROOT"
```

Use `$SGLANG_ROOT` for all subsequent paths (benchmarks, scripts, git operations, etc.) instead of hardcoded paths like `$HOME/sglang` or `/sgl-workspace/sglang`.

### 0b: Ask the user

Ask the user using `AskUserQuestion`:

1. **Server launch script** — the script that sets env vars and launches `sglang.launch_server`
2. **Client benchmark script** — the script that runs `sglang.bench_serving` against the server
3. **Model name** — for labeling (default: DSv4)
4. **Accuracy threshold** — default 0.88 for DSv4

Parse the server script to extract the **port** (look for `--port <N>` in the launch command).

### PIPELINE_MODE (from `/kernel-fusion-pipeline`)

When CONFIG contains `"pipeline_mode": true`:

- **Skip Step 0b** — use CONFIG fields directly:
  - `server_script`, `client_script`, `port`, `gpus`, `label`, `threshold`
  - `worktree` — run all git/server commands with `git -C "$WORKTREE"` and `cd "$WORKTREE"`
  - `pythonpath` — `export PYTHONPATH="$PYTHONPATH"` before launching server
- **NO `AskUserQuestion`** — ever, for any reason. Specifically:
  - Step 0b: skipped (use CONFIG fields).
  - Step 1a fallback: auto-skip baseline if revert method is ambiguous. The pipeline handles before/after via worktree git stash.
  - Step 4a: run profiling command immediately without confirmation.
  - Step 4e: auto-proceed if accuracy passed and fused kernel is visible in trace. Only return failure if the expected kernel change is NOT visible.
- Baseline revert (Step 1a): use worktree only; prefer `git stash` if dirty, else `git checkout HEAD~1` when exactly one fusion commit ahead of base.
- **On accuracy failure**: do NOT stop and ask. Return `status: fail` with the accuracy number. The pipeline decides whether to retry or skip.
- **On profiling mismatch**: log the mismatch, return `status: warn` with details. Do NOT ask for confirmation.

---

## Step 1: Baseline Benchmark

The purpose of this step is to collect performance numbers **without** the code change, so we can compare before/after in the PR.

### 1a: Determine how to revert

Check `git status` and `git log` to determine the state:

```bash
git status --porcelain
git log --oneline -3
```

- **If there are uncommitted changes** (modified/added files): use `git stash` to revert, `git stash pop` to restore.
- **If the working tree is clean but on a feature branch with commits ahead of the base**: use `git checkout HEAD~1` to go back one commit (the pre-change state), then `git checkout -` to return.
- **If neither applies** (e.g., changes are mixed with other work, or can't determine baseline): ask the user with `AskUserQuestion` whether to skip baseline or provide a baseline branch/commit. (**PIPELINE_MODE**: auto-skip baseline, do not ask.)

### 1b: Revert the changes

Based on 1a:

```bash
# Option A: uncommitted changes
git stash

# Option B: committed on feature branch
git checkout HEAD~1
```

Verify the revert worked — the changed files should no longer contain the optimization.

### 1c: Start server (baseline code)

Check if a server is already running:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health
```

If `200`, kill it first — we need the baseline code running:

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
sleep 3
```

Launch the server in the background:

```bash
bash <server_script>
```

Poll until ready:

```bash
for i in $(seq 1 120); do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health 2>/dev/null)
  if [ "$status" = "200" ]; then
    echo "Server ready after ${i} attempts"
    exit 0
  fi
  sleep 10
done
echo "Server failed to start after 1200s"
exit 1
```

If the server doesn't start within ~20 minutes, **restore changes immediately** (stash pop / checkout -), warn the user, and skip baseline.

### 1d: Run baseline e2e benchmark

Run the client benchmark script as-is:

```bash
bash <client_script>
```

**Parse and record these metrics from the output:**
- `Total token throughput (tok/s)`
- `Output token throughput (tok/s)`
- `Median TTFT (ms)`
- `Median ITL (ms)`
- `Median TPOT (ms)`
- `Median E2E Latency (ms)`

Store these as "baseline" numbers for the summary report.

### 1e: Kill the baseline server

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
sleep 3
```

Verify it's stopped:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health 2>/dev/null
```

### 1f: Restore the changes

```bash
# Option A: if we used git stash
git stash pop

# Option B: if we used git checkout HEAD~1
git checkout -
```

Verify the changes are back — check `git status` or `git diff` to confirm.

**IMPORTANT:** If anything in steps 1c-1e fails, ALWAYS restore the changes before proceeding. Never leave the user's working tree in the reverted state.

## Step 2: Start the server (with changes)

### 2a: Check if server is already running

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health
```

If `200`, skip to Step 3.

### 2b: Launch the server in background

```bash
bash <server_script>
```

### 2c: Wait for server readiness

Poll until ready (same loop as Step 1c). If the server doesn't start within ~20 minutes, **STOP** and report the failure.

## Step 3: Accuracy Test

Run the GSM8K accuracy benchmark:

```bash
cd "$SGLANG_ROOT" && python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 2000 --port <PORT>
```

**Interpret the results:**
- Parse the output for the accuracy score (look for "Accuracy:" or similar)
- Compare against the threshold (e.g., > 0.88 for DSv4)
- If accuracy is **below** the threshold or results look meaningless/garbage:
  - **STOP** — report the failure to the user immediately
  - Show the accuracy number and sample outputs
  - Ask the user how to proceed (fix code, re-run, or skip)
  - Do NOT continue to profiling until accuracy passes
  - (**PIPELINE_MODE**: do NOT stop or ask. Return `status: fail` with the accuracy number. The pipeline decides retry/skip.)
- If accuracy **passes**: record the exact accuracy number and proceed

**Save the result:**
- Record: `accuracy = <value>`, `num_questions = 200`, `parallel = 2000`

## Step 4: Profiling Run

The purpose of this step is to confirm the code change actually has the intended effect at the kernel/module level. This is NOT a performance benchmark — it is a correctness check for the optimization.

### 4a: Prepare the profiling client command

Take the client benchmark script and modify it for profiling:

1. **Add `--profile`** flag to the `sglang.bench_serving` command
2. **Change `num_prompts`** to `num_prompts=$((max_concurrency * 2))` — profiling only needs a short run

Show the modified profiling command to the user and confirm before running. (**PIPELINE_MODE**: skip confirmation, run immediately.)

### 4b: Run the profiling client

Execute the modified client command. The trace files will be generated under `/tmp/` by default.

### 4c: Locate and copy the trace files

After the profiling run completes:

1. Find the latest trace directory under `/tmp/` (timestamped folders containing `.trace.json.gz` files)
2. Copy the trace folder to a stable location:
   ```bash
   cp -r /tmp/<trace-folder> $HOME/dsv4/
   ```
3. Identify the GPU 0 trace file: `*TP-0.trace.json.gz`

### 4d: Analyze the trace

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py \
  $HOME/dsv4/<trace-folder>/<timestamp>-TP-0.trace.json.gz \
  -o analysis_<change_description>
```

### 4e: Interpret profiling results

Review the trace analysis output:
- Look for the specific kernels that the code change was supposed to affect
- Confirm the change is reflected (new kernel appears, old kernel is gone, time changed)
- If NOT visible, warn the user — the optimization may not be working

Ask the user to confirm the profiling results look correct before proceeding. (**PIPELINE_MODE**: auto-proceed if accuracy passed and expected kernel change is visible. Only return failure if the expected kernel change is NOT visible.)

## Step 5: E2E Benchmark (after)

Run the client benchmark script as-is — the **same command** used for baseline in Step 1d:

```bash
bash <client_script>
```

**Parse and record the same metrics** as baseline:
- `Total token throughput (tok/s)`
- `Output token throughput (tok/s)`
- `Median TTFT (ms)`
- `Median ITL (ms)`
- `Median TPOT (ms)`
- `Median E2E Latency (ms)`

Store these as "after" numbers.

## Step 6: Summary Report

### 6a: Kernel-to-E2E impact analysis

After profiling (Step 4) and the after benchmark (Step 5) are both complete, calculate the expected e2e impact from the kernel-level changes observed in the trace. This connects the micro-level optimization to macro-level metrics and helps the reader judge whether the benchmark deltas are real signal or noise.

**How to calculate:**

1. From the profiling trace analysis (Step 4d), read the detail sheets for each layer type. For each layer type, identify:
   - Which kernels changed (appeared, disappeared, or changed duration)
   - The per-layer kernel time savings (sum of eliminated/reduced kernel times)

2. Multiply per-layer savings by the number of layers of that type per iteration:
   ```
   total_savings_us = sum(per_layer_savings[type] * layer_count[type] for type in layer_types)
   ```

3. Compute expected ITL impact:
   ```
   expected_itl_delta_pct = total_savings_us / (baseline_itl_ms * 1000) * 100
   ```

4. For TTFT: note whether the kernel change applies to prefill. Prefill is typically dominated by large GEMMs, so small kernel fusions have negligible TTFT impact. State this explicitly.

5. Compare expected vs observed:
   - If observed delta is within ~2x of expected: consistent (kernel savings may be partially masked by run-to-run variance)
   - If observed delta is much larger than expected: other factors at play (warmup, scheduling, memory)
   - If observed delta is opposite direction from expected: the optimization may not be working as intended — investigate

### 6b: Compose the report

Compute deltas between baseline and after for each metric. For throughput metrics (higher is better), delta = `(after - baseline) / baseline * 100`. For latency metrics (lower is better), delta = `(after - baseline) / baseline * 100` (negative means improvement).

```
=== Validation Summary ===

Model: <model name>

1. E2E Benchmark Comparison:
   | Metric                    | Baseline | After   | Delta   |
   |---------------------------|----------|---------|---------|
   | Total throughput (tok/s)   | 1550.2   | 1583.5  | +2.1%   |
   | Output throughput (tok/s)  | 172.3    | 175.9   | +2.1%   |
   | Median TTFT (ms)           | 1560     | 1542    | -1.2%   |
   | Median ITL (ms)            | 20.58    | 20.56   | -0.1%   |
   | Median TPOT (ms)           | 21.15    | 21.10   | -0.2%   |
   | Median E2E Latency (ms)    | 23200    | 23079   | -0.5%   |

2. Kernel-to-E2E Impact Analysis:
   | Layer Type | Layers | Kernel Savings/Layer | Total Savings |
   |------------|--------|---------------------|---------------|
   | C-type     | 30     | 6.3 us              | 189 us        |
   | A-type     | 61     | 0 us                | 0 us          |
   | B-type     | 30     | 0 us                | 0 us          |
   | **Total**  |        |                     | **189 us**    |

   Expected ITL improvement: 189 / 20570 = 0.92%
   Observed ITL improvement: 0.05%
   Assessment: Kernel savings (~189 us) is <1% of iteration time (~20.6 ms),
               within run-to-run benchmark noise. Observed delta is consistent.

   TTFT impact: Negligible — prefill dominated by large GEMMs.

3. Accuracy Test: PASS / FAIL
   - Score: <accuracy> (threshold: <threshold>)
   - Questions: 200, Parallel: 2000

4. Profiling: CONFIRMED / NOT CONFIRMED
   - <Key observations about the code change's effect>
   - Trace analysis output: <path to analysis xlsx>
```

If baseline was skipped (Step 1 failed or was skipped), show only the "after" numbers and note "Baseline: not available". The kernel-to-e2e analysis can still be computed using the "after" profiling trace and baseline ITL.

Tell the user:
- Results are ready for PR submission
- Profiling analysis file path for reference
- If any step failed or was skipped, highlight it clearly

## Step 7: Kill the server

After the summary report is printed, shut down the server that was launched in Step 2:

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 3
pkill -9 -f "sglang" 2>/dev/null
```

Verify it's stopped:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health 2>/dev/null
```

If the health check returns `000` or a non-200 code, report "Server stopped." to the user.

**Note:** If the server was already running before validation started (i.e., Step 2a detected a running server and we skipped launching), do NOT kill it — the user may need it for other work. Only kill the server if we launched it ourselves in Step 2b.
