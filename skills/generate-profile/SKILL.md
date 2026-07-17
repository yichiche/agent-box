---
description: Generate an e2e profiling trace of an SGLang server run. Launches a server, validates accuracy, captures a Chrome-compatible trace, and returns the profile path.
---

# Generate Profiling Trace

End-to-end workflow: launch server → validate accuracy → run profiling client → collect traces → analyze. Produces trace files and an analysis Excel report ready for `/profile` or `/compare-kernels`.

## Step 0: Gather context

### 0a: Detect the active SGLang installation

Before doing anything else, determine which SGLang repo is in use. There may be multiple SGLang directories on the system (e.g., `/sgl-workspace/sglang`, `$HOME/sglang`). **Always use the one that's currently installed in the active Python environment.**

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
echo "Active SGLang root: $SGLANG_ROOT"
```

This resolves the path from the **installed** package, so it works regardless of which copy is active. Use `$SGLANG_ROOT` for all subsequent paths (benchmarks, scripts, etc.) instead of hardcoded paths like `$HOME/sglang` or `/sgl-workspace/sglang`.

Verify it looks correct:
```bash
ls "$SGLANG_ROOT/benchmark/gsm8k/bench_sglang.py"
```

### 0b: Resolve from the model card first (card-driven)

**Resolve everything you can from the registry before asking the user.** From
`$ARGUMENTS` (or the server script path) infer the model, then read
[[../../memory/models/INDEX.md]] and the matching card under `memory/models/`.
The card provides: **server script, client script, port, accuracy threshold,
output directory convention, and profiling detail-module/instance.**

- Resolve the **workload** via [[../../memory/workflows/workloads.md]] — default
  `canonical-8k` (IL8192/OL1024); use `diag-1k` (1024/1024) for a cheap capture.
- Resolve the **output directory** from the card, NOT the DSv4 default. e.g.
  qwen35-mxfp4 → `$HOME/qwen3.5-mxfp4/<label>/`; DSv4 → `$HOME/dsv4/<label>/`.
  Never save Qwen artifacts under `~/dsv4/`.

Only `AskUserQuestion` for what the card does **not** resolve — typically just the
**profile label** (short name like `baseline`, `after_fix`). If everything else is
resolved and a label is derivable (e.g. git branch), proceed without asking at all.
Ask for server/client/threshold/model only when the model is unknown or unresolvable.

Fields to end up with:
1. **Server launch script** — from card, or ask
2. **Client benchmark script** — from card, or ask
3. **Profile label** — ask if not derivable
4. **Model name** — from card resolution (do NOT default to DSv4)
5. **Accuracy threshold** — from card (0.88 DSv4, 0.92 qwen35-mxfp4); ask only if unknown
6. **Profile by stage?** — default yes (separate prefill/decode traces). **Exception:** if the goal is *module-level* trace analysis on a model whose decode is CUDA-graph-replayed (e.g. qwen35-mxfp4, whose `*-DECODE` trace collapses to `CudaGraphReplay` with no per-module detail — see its card), capture a **combined** trace instead (no `--profile-by-stage`).

Parse the server script to extract the **port** (look for `--port <N>` in the launch command).

## Step 1: Start the server

### 1a: Check if server is already running

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health
```

If `200`, ask the user whether to:
- **Reuse** the running server (skip to Step 2)
- **Restart** it (kill and relaunch)

### 1b: Launch the server in background

```bash
bash <server_script>
```

### 1c: Wait for server readiness

Poll until ready — run this in the **foreground**, not background:

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

If the server doesn't start within ~20 minutes, **STOP** and report the failure.

## Step 2: Accuracy Test

Run the GSM8K accuracy benchmark using the detected `$SGLANG_ROOT`:

```bash
cd "$SGLANG_ROOT" && python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 100 --num-shots 5 --port <PORT>
```

Note: we use fewer questions (200) than full validation (2000) since this is a sanity check, not a full accuracy run.

**Interpret the results:**
- Parse the output for the accuracy score
- Compare against the threshold (e.g., > 0.88 for DSv4)
- If accuracy is **below** the threshold:
  - **STOP** — report the failure immediately
  - Show the accuracy number and sample outputs
  - Do NOT continue to profiling
- If accuracy **passes**: record the score and proceed

## Step 3: Run the Profiling Client

### 3a: Prepare the profiling client command

Take the client benchmark script and modify it for profiling:

1. **Add `--profile`** flag to the `sglang.bench_serving` command
2. **Change `num_prompts`** to `num_prompts=$((max_concurrency * 2))` — profiling only needs a short run
3. **Add `--profile-by-stage`** if the user opted for stage-separated traces (default: yes)
4. **Add `--profile-output-dir <OUTPUT_DIR>/<profile_label>`** to save traces to the labeled directory, where `<OUTPUT_DIR>` is the card-resolved per-model dir (Step 0b) — e.g. `$HOME/qwen3.5-mxfp4` for qwen35-mxfp4, `$HOME/dsv4` for DSv4. Do NOT hardcode `$HOME/dsv4` for non-DSv4 models.

Show the modified profiling command to the user and confirm before running.

### 3b: Run the profiling client

Execute the modified client command. Wait for it to complete — this typically takes 2-5 minutes.

### 3c: Locate the trace files

After the profiling run completes:

1. Find the trace files in the output directory. They follow the naming pattern:
   - Combined: `<timestamp>-TP-<rank>.trace.json.gz`
   - Stage-separated: `<timestamp>-TP-<rank>-EXTEND.trace.json.gz` and `<timestamp>-TP-<rank>-DECODE.trace.json.gz`

2. List the generated trace files:
   ```bash
   ls -la <OUTPUT_DIR>/<profile_label>/*.trace.json.gz
   ```

3. Identify the GPU 0 trace file(s) for analysis:
   - Combined: `*-TP-0.trace.json.gz`
   - Stage-separated: `*-TP-0-DECODE.trace.json.gz` and `*-TP-0-EXTEND.trace.json.gz`

If no trace files are found under the profile output dir, check `/tmp/` for the latest timestamped trace directory:
```bash
ls -lt /tmp/*.trace.json.gz 2>/dev/null | head -5
ls -ltd /tmp/[0-9]* 2>/dev/null | head -5
```

If traces landed in `/tmp/`, copy them to the labeled directory:
```bash
cp /tmp/<trace-files> <OUTPUT_DIR>/<profile_label>/
```

## Step 4: Analyze the Traces

### 4a: Run the trace module analyzer

For each GPU 0 trace file (decode and/or extend), pass the **card-resolved**
`--detail-module` / `--detail-instance` (Step 0b) — do not rely on the analyzer's
DSv4 defaults:

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py \
  <OUTPUT_DIR>/<profile_label>/<trace-file> \
  -o analysis_<phase>.xlsx \
  --detail-module <MODULE> --detail-instance <INSTANCES>
```

Where `<phase>` is `decode`, `extend`, or `combined`, and `<MODULE>`/`<INSTANCES>`
come from the model card (e.g. qwen35-mxfp4 → `Qwen3_5LinearDecoderLayer 0 1`;
DSv4 decode → `Layer 59 60 61 62`, prefill → `DeepseekV4DecoderLayer 31 32`).

### 4b: Verify analysis output

Check that the Excel report was generated:

```bash
ls -la <OUTPUT_DIR>/<profile_label>/analysis_*.xlsx
```

### 4c: Report key findings

Read the analyzer console output and report:
- Number of prefill vs decode iterations detected
- Top-3 modules by total GPU time
- Median decode iteration time
- Median prefill iteration time (if available)
- Any warnings about ROCm trace fixes applied

## Step 5: Summary Report

```
=== Profiling Summary ===

Model: <model_name>
Label: <profile_label>
Platform: <MI355/B200, detected from trace>
SGLang Root: <SGLANG_ROOT>

Accuracy: PASS (<score>, threshold: <threshold>)

Trace Files:
  - Decode: <OUTPUT_DIR>/<profile_label>/<decode_trace>
  - Extend: <OUTPUT_DIR>/<profile_label>/<extend_trace>

Analysis Reports:
  - Decode: <OUTPUT_DIR>/<profile_label>/analysis_decode.xlsx
  - Extend: <OUTPUT_DIR>/<profile_label>/analysis_extend.xlsx

Key Metrics:
  - Median decode iteration: <X> ms
  - Median prefill iteration: <Y> ms
  - Top modules by GPU time: <list>

Next Steps:
  - View traces in Perfetto UI: https://ui.perfetto.dev/
  - Compare with another trace: /compare-kernels <file1.xlsx> <file2.xlsx>
  - Detailed analysis: /profile
```

## Step 6: Kill the server (conditional)

Only kill the server if **we launched it** in Step 1b. If the user had a pre-existing server (Step 1a — reuse), leave it running.

```bash
pkill -f "sglang.launch_server.*--port <PORT>" 2>/dev/null
sleep 3
pkill -9 -f "sglang" 2>/dev/null
```

Verify it's stopped:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health 2>/dev/null
```

## Notes

- **SGLang detection is mandatory** — always run Step 0a first. Never hardcode `$HOME/sglang` or `/sgl-workspace/sglang`.
- **Server polling must run in foreground** — never use background + TaskOutput for health-check polling.
- **Profile by stage is recommended** — separate decode/extend traces are easier to analyze and compare.
- **Accuracy check is mandatory** — profiling garbage output wastes time and produces misleading kernel breakdowns.
- **Traces can be large** (100MB+) — the `.gz` compression reduces size ~10x.
- **The analyzer auto-detects ROCm traces** and applies hipGraphLaunch flow event fixes automatically.
