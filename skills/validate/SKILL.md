---
description: Validate SGLang code changes — accuracy test, profiling, and benchmark. Use when the user says "validate the result" or "/validate" after making SGLang changes.
---

# Validate SGLang Changes

End-to-end validation of code changes in SGLang. Starts the server, confirms accuracy, profiles to verify the change actually works, and runs a normal benchmark. Collects all results so they are ready for PR submission.

## Step 0: Gather context

Ask the user using `AskUserQuestion`:

1. **Server launch script** — the script that sets env vars and launches `sglang.launch_server`
2. **Client benchmark script** — the script that runs `sglang.bench_serving` against the server
3. **Model name** — for labeling (default: DSv4)
4. **Accuracy threshold** — default 0.88 for DSv4

Parse the server script to extract the **port** (look for `--port <N>` in the launch command).

## Step 1: Start the server

### 1a: Check if server is already running

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health
```

If `200`, skip to Step 2.

### 1b: Launch the server in background

Run the server script in the background using the Bash tool with `run_in_background: true`:

```bash
bash <server_script>
```

### 1c: Wait for server readiness

Poll until the server is ready (health endpoint returns 200). Use a loop with short sleeps:

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

Run the GSM8K accuracy benchmark:

```bash
cd $HOME/sglang && python3 benchmarks/gsm8k/bench_sglang.py --num-questions 2000 --parallel 1000 --num-shots 5 --port <PORT>
```

**Interpret the results:**
- Parse the output for the accuracy score (look for "Accuracy:" or similar)
- Compare against the threshold (e.g., > 0.88 for DSv4)
- If accuracy is **below** the threshold or results look meaningless/garbage:
  - **STOP** — report the failure to the user immediately
  - Show the accuracy number and sample outputs
  - Ask the user how to proceed (fix code, re-run, or skip)
  - Do NOT continue to profiling until accuracy passes
- If accuracy **passes**: record the exact accuracy number and proceed

**Save the result:**
- Record: `accuracy = <value>`, `num_questions = 2000`, `num_shots = 5`, `parallel = 1000`

## Step 3: Profiling Run

The purpose of this step is to confirm the code change actually has the intended effect at the kernel/module level. This is NOT a performance benchmark — it is a correctness check for the optimization.

### 3a: Prepare the profiling client command

Take the client benchmark script and modify it for profiling:

1. **Add `--profile`** flag to the `sglang.bench_serving` command
2. **Change `num_prompts`** to `num_prompts=$((max_concurrency * 2))` — profiling only needs a short run

Show the modified profiling command to the user and confirm before running.

### 3b: Run the profiling client

Execute the modified client command. The trace files will be generated under `/tmp/` by default.

### 3c: Locate and copy the trace files

After the profiling run completes:

1. Find the latest trace directory under `/tmp/` (timestamped folders containing `.trace.json.gz` files)
2. Copy the trace folder to a stable location:
   ```bash
   cp -r /tmp/<trace-folder> $HOME/dsv4/
   ```
3. Identify the GPU 0 trace file: `*TP-0.trace.json.gz`

### 3d: Analyze the trace

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py \
  $HOME/dsv4/<trace-folder>/<timestamp>-TP-0.trace.json.gz \
  -o analysis_<change_description>
```

### 3e: Interpret profiling results

Review the trace analysis output:
- Look for the specific kernels that the code change was supposed to affect
- Confirm the change is reflected (new kernel appears, old kernel is gone, time changed)
- If NOT visible, warn the user — the optimization may not be working

Ask the user to confirm the profiling results look correct before proceeding.

## Step 4: Normal Benchmark

Run the client benchmark script as-is — full performance benchmark without `--profile`:

```bash
bash <client_script>
```

**Record the results:**
- Parse output for key metrics: throughput (tokens/s), latency (TTFT, ITL, E2E)
- Save all numbers for PR submission

## Step 5: Summary Report

```
=== Validation Summary ===

Model: <model name>

1. Accuracy Test: PASS / FAIL
   - Score: <accuracy> (threshold: <threshold>)
   - Questions: 2000, Shots: 5, Parallel: 1000

2. Profiling: CONFIRMED / NOT CONFIRMED
   - <Key observations about the code change's effect>
   - Trace analysis output: <path to analysis xlsx>

3. Benchmark:
   - <Key performance metrics>
   - <Throughput / latency numbers>
```

Tell the user:
- Results are ready for PR submission
- Profiling analysis file path for reference
- If any step failed or was skipped, highlight it clearly

## Step 6: Kill the server

After the summary report is printed, shut down the server that was launched in Step 1:

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

**Note:** If the server was already running before validation started (i.e., Step 1a detected a running server and we skipped launching), do NOT kill it — the user may need it for other work. Only kill the server if we launched it ourselves in Step 1b.
