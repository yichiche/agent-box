---
name: benchmark
description: "Run before/after e2e benchmark of an SGLang code change. Collects baseline (change reverted), restores the change, re-benchmarks, and produces a comparison table. Optionally profiles with /generate-profile. Use when the user says '/benchmark' or asks for a before/after benchmark."
---

# Before / After Benchmark

Run an e2e benchmark with the code change reverted (baseline), then again with the change applied (after). Produces a side-by-side comparison table. Optionally generates profiling traces via the `/generate-profile` flow.

## Step 0: Gather context

### 0a: Confirm the SGLang path

Detect the active SGLang installation:

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
echo "Active SGLang root: $SGLANG_ROOT"
```

**Always confirm with the user** using `AskUserQuestion`:

```
"The detected SGLang installation is at: <SGLANG_ROOT>. Is this the correct repo for benchmarking?"
```

Options:
- **Yes, use this path** (Recommended)
- **No, use a different path** — ask the user for the correct path

If the user provides a different path, use that instead of the detected one for all subsequent operations.

### 0b: Collect benchmark parameters

**Resolve from the model card first (card-driven).** From `$ARGUMENTS` (or a
provided script) infer the model, then read [[../../memory/models/INDEX.md]] and
the matching card under `memory/models/` to get **server script, client script,
port, accuracy threshold, and output directory**. Resolve the **workload** via
[[../../memory/workflows/workloads.md]] — default `canonical-8k` (IL8192/OL1024),
`diag-1k` (1024/1024) for correctness/scaling only.

Then use `AskUserQuestion` **only for what the card did not resolve**. If the card
fully resolves server+client+port+threshold and the workload is the default
`canonical-8k`, skip straight to Step 1 (still show the resolved parameters once so
the user can catch a wrong model). Ask the questions below only for missing items.

#### Question 1: Server launch script

```
"Which server launch script should be used?"
```

This is the shell script that sets env vars and runs `python3 -m sglang.launch_server ...`. The user may provide a path like `~/run_dsv4_0525.sh`.

Parse the server script to extract:
- **Port**: look for `--port <N>` in the launch command
- **Model path**: look for `--model-path <path>`

#### Question 2: Client benchmark script

```
"Which client benchmark script should be used?"
```

This is the shell script that runs `python3 -m sglang.bench_serving ...`. The user may provide a path like `~/run_dsv4_client.sh`.

Parse the client script to extract and display:
- **Concurrencies**: the values in the `concurrencies=(...)` array
- **Input length (ISL)**: `--random-input` or `input_tokens` variable
- **Output length (OSL)**: `--random-output` or `output_tokens` variable
- **num_prompts formula**: how many prompts per concurrency (e.g., `max_concurrency * 8`)

#### Question 3: Confirm workload parameters

After parsing both scripts, show the extracted parameters and confirm:

```
"Here are the benchmark parameters extracted from the scripts:

  Server:  <server_script>
  Port:    <PORT>
  Model:   <model_path>

  Client:  <client_script>
  ISL:     <input_tokens>
  OSL:     <output_tokens>
  Concurrencies: <list>
  Prompts/concurrency: <formula>

Are these correct?"
```

Name the workload against a preset from [[../../memory/workflows/workloads.md]]:
if ISL/OSL is 8192/1024 it's `canonical-8k` (valid for perf claims); if 1024/1024
it's `diag-1k` (diagnostic only — a delta here is NOT an 8K speedup). Baseline and
after MUST use the same preset.

Options:
- **Yes, proceed** (Recommended)
- **Change concurrencies** — ask for new values
- **Change ISL/OSL** — ask for new values (name the resulting preset)
- **Change num_prompts** — ask for new formula

If the user changes parameters, note that you will need to modify the client script before running. Prepare the modified command but do NOT overwrite the user's script file — construct the command inline.

#### Question 4: Profiling

```
"Should profiling traces be captured after the 'after' benchmark?"
```

Options:
- **Yes, profile** — after the 'after' benchmark, run profiling using the `/generate-profile` flow (adds `--profile --profile-by-stage` to the client command)
- **No, benchmark only** (Recommended) — skip profiling, just compare benchmark numbers

#### Question 5: Label

```
"Short label for this benchmark run (used for output directory naming)?"
```

Options:
- **Use git branch name** (Recommended) — auto-detect from `git branch --show-current`
- **Custom label** — user provides a name like `fused_norm_rope`, `baseline_v2`

## Step 1: Baseline Benchmark

The purpose of this step is to collect performance numbers **without** the code change.

### 1a: Determine how to revert

```bash
cd "$SGLANG_ROOT" && git status --porcelain && git log --oneline -3
```

- **Uncommitted changes** (modified/added files): use `git stash` to revert, `git stash pop` to restore.
- **Clean tree, on feature branch ahead of base**: use `git checkout HEAD~1` to go back one commit, then `git checkout -` to return.
- **Cannot determine baseline**: ask the user with `AskUserQuestion` whether to skip baseline or provide a baseline branch/commit.

### 1b: Revert the changes

```bash
# Option A: uncommitted changes
cd "$SGLANG_ROOT" && git stash

# Option B: committed on feature branch
cd "$SGLANG_ROOT" && git checkout HEAD~1
```

Verify the revert worked — the changed files should no longer contain the modification.

### 1c: Reinstall if needed

If the code change touched Python source files (under `python/sglang/`), the editable install should pick it up automatically. If it touched `sgl-kernel/` or C++ code, rebuild:

```bash
cd "$SGLANG_ROOT/sgl-kernel" && make build
```

### 1d: Start server (baseline code)

Kill any existing server on the target port:

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
sleep 3
```

Launch the server:

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

If the server doesn't start within ~20 minutes, **restore changes immediately** (Step 1f), warn the user, and skip baseline.

### 1e: Run baseline benchmark

Run the client benchmark script (or the modified command if parameters were changed in Step 0b):

```bash
bash <client_script>
```

**Parse and record these metrics from the output** for each concurrency level:

| Metric | Where to find |
|---|---|
| Total token throughput (tok/s) | `Total token throughput:` line |
| Output token throughput (tok/s) | `Output token throughput:` line |
| Median TTFT (ms) | `Median TTFT:` line |
| Median ITL (ms) | `Median ITL:` line |
| Median TPOT (ms) | `Median TPOT:` line |
| Median E2E Latency (ms) | `Median E2E Latency:` line |

Store all metrics keyed by concurrency as **baseline** numbers.

### 1f: Kill baseline server and restore changes

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
sleep 3
```

Restore the changes:

```bash
# Option A: if we used git stash
cd "$SGLANG_ROOT" && git stash pop

# Option B: if we used git checkout HEAD~1
cd "$SGLANG_ROOT" && git checkout -
```

Verify the changes are back — check `git status` or `git diff`.

**IMPORTANT:** If anything in steps 1c-1e fails, ALWAYS restore the changes before proceeding. Never leave the user's working tree in the reverted state.

### 1g: Rebuild if needed

If a rebuild was done in 1c, rebuild again with the changes restored:

```bash
cd "$SGLANG_ROOT/sgl-kernel" && make build
```

## Step 2: After Benchmark

### 2a: Start server (with changes)

Kill any leftover server and launch fresh:

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
sleep 3
bash <server_script>
```

Poll until ready (same loop as Step 1d). If the server doesn't start within ~20 minutes, **STOP** and report.

### 2b: Run after benchmark

Run the **same** client command used in Step 1e:

```bash
bash <client_script>
```

Parse and record the same metrics, keyed by concurrency. Store as **after** numbers.

## Step 3: Profiling (optional)

Only if the user chose profiling in Step 0b Q4.

### 3a: Prepare profiling command

Take the client benchmark script and modify it:

1. **Add `--profile`** flag to the `sglang.bench_serving` command
2. **Add `--profile-by-stage`** for separate prefill/decode traces
3. **Change `num_prompts`** to `num_prompts=$((max_concurrency * 2))` — profiling only needs a short run
4. **Add `--profile-output-dir <OUTPUT_DIR>/<label>`** to save traces to the labeled directory, where `<OUTPUT_DIR>` is the card-resolved per-model dir (Step 0b) — e.g. `$HOME/qwen3.5-mxfp4` for qwen35-mxfp4. Do NOT hardcode `$HOME/dsv4` for non-DSv4 models.

Show the modified command to the user and confirm before running.

### 3b: Run profiling

Execute the modified client command. Wait for completion (typically 2-5 minutes).

### 3c: Locate trace files

```bash
ls -la <OUTPUT_DIR>/<label>/*.trace.json.gz 2>/dev/null
```

If traces landed in `/tmp/` instead:

```bash
ls -lt /tmp/*.trace.json.gz 2>/dev/null | head -5
```

Copy them to the labeled directory if needed.

### 3d: Analyze traces

For each GPU 0 trace file:

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py \
  <trace_file> \
  -o analysis_<phase>.xlsx \
  --detail-module <MODULE> --detail-instance <INSTANCES>
```

Where `<phase>` is `decode`, `extend`, or `combined`, and `<MODULE>`/`<INSTANCES>`
come from the model card (Step 0b) — e.g. qwen35-mxfp4 → `Qwen3_5LinearDecoderLayer 0 1`.
Do not rely on the analyzer's DSv4 defaults for non-DSv4 models.

Report key findings from the analysis.

## Step 4: Kill server

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 3
pkill -9 -f "sglang" 2>/dev/null
```

Verify:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health 2>/dev/null
```

## Step 5: Summary Report

Compute deltas for each metric at each concurrency level. For throughput (higher is better): `delta = (after - baseline) / baseline * 100`. For latency (lower is better): same formula (negative = improvement).

```
=== Benchmark Comparison: <label> ===

SGLang Root: <SGLANG_ROOT>
Model:       <model_path>
Server:      <server_script>
Client:      <client_script>
ISL/OSL:     <input_tokens> / <output_tokens>

### Concurrency <C1>

| Metric                    | Baseline | After   | Delta   |
|---------------------------|----------|---------|---------|
| Total throughput (tok/s)   | 1550.2   | 1583.5  | +2.1%   |
| Output throughput (tok/s)  | 172.3    | 175.9   | +2.1%   |
| Median TTFT (ms)           | 1560     | 1542    | -1.2%   |
| Median ITL (ms)            | 20.58    | 20.56   | -0.1%   |
| Median TPOT (ms)           | 21.15    | 21.10   | -0.2%   |
| Median E2E Latency (ms)    | 23200    | 23079   | -0.5%   |

### Concurrency <C2>
(same table format)

### Profiling (if captured)

Trace files:
  - Decode: <path>
  - Extend: <path>

Analysis reports:
  - Decode: <path>/analysis_decode.xlsx
  - Extend: <path>/analysis_extend.xlsx

Key observations:
  - <kernel-level findings>

Next steps:
  - Compare kernels: /compare-kernels <baseline.xlsx> <after.xlsx>
  - Detailed trace analysis: /profile <path>
```

If baseline was skipped, show only the "after" numbers and note "Baseline: not available — showing absolute numbers only."

## Notes

- **SGLang path confirmation is mandatory** — always run Step 0a and confirm with the user. Multiple SGLang copies may exist on the machine.
- **Server polling must run in foreground** — never use background + TaskOutput for health-check polling.
- **Always restore changes** — if baseline collection fails at any point, restore the working tree before doing anything else.
- **Same client command for both runs** — the baseline and after benchmarks must use identical parameters for a valid comparison.
- **Concurrency matters** — different concurrency levels stress different parts of the system. If the user didn't specify, use whatever the client script already has.
- **Profiling is separate from benchmarking** — profiling adds overhead and uses fewer prompts, so it should not be mixed with the performance measurement runs.
- **Rebuild for kernel changes** — if the change touches `sgl-kernel/` C++/CUDA code, both the revert and restore steps need a rebuild. This adds significant time (~5-10 min per build).
