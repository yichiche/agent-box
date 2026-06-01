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

### 0b: Ask the user

Ask the user using `AskUserQuestion`:

1. **Server launch script** — the script that sets env vars and launches `sglang.launch_server`
2. **Client benchmark script** — the script that runs `sglang.bench_serving` against the server
3. **Profile label** — short name for this profiling run (e.g., `baseline`, `fused_norm_rope`, `after_fix`). Used for output directory naming.
4. **Model name** — for labeling (default: DSv4)
5. **Accuracy threshold** — default 0.88 for DSv4
6. **Profile by stage?** — whether to generate separate prefill/decode traces (default: yes)

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
4. **Add `--profile-output-dir $HOME/dsv4/<profile_label>`** to save traces to the labeled directory

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
   ls -la $HOME/dsv4/<profile_label>/*.trace.json.gz
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
cp /tmp/<trace-files> $HOME/dsv4/<profile_label>/
```

## Step 4: Analyze the Traces

### 4a: Run the trace module analyzer

For each GPU 0 trace file (decode and/or extend):

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py \
  $HOME/dsv4/<profile_label>/<trace-file> \
  -o analysis_<phase>.xlsx
```

Where `<phase>` is `decode`, `extend`, or `combined` depending on what trace is being analyzed.

### 4b: Verify analysis output

Check that the Excel report was generated:

```bash
ls -la $HOME/dsv4/<profile_label>/analysis_*.xlsx
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
  - Decode: $HOME/dsv4/<profile_label>/<decode_trace>
  - Extend: $HOME/dsv4/<profile_label>/<extend_trace>

Analysis Reports:
  - Decode: $HOME/dsv4/<profile_label>/analysis_decode.xlsx
  - Extend: $HOME/dsv4/<profile_label>/analysis_extend.xlsx

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
