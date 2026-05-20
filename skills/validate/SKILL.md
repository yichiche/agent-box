---
description: Validate SGLang code changes — accuracy test, profiling, and benchmark. Use when the user says "validate the result" or "/validate" after making SGLang changes.
---

# Validate SGLang Changes

End-to-end validation of code changes in SGLang. Confirms accuracy, profiles to verify the change actually works, and runs a normal benchmark. Collects all results so they are ready for PR submission.

## Step 0: Gather context from the user

Ask the user for two things using `AskUserQuestion`:

1. **Server launch script**: The command or script used to start the SGLang server (e.g., `python -m sglang.launch_server --model-path ... --tp ...`). The server must already be running or the user must provide the command so you can reference it.
2. **Client benchmark script**: The command or script used to run the normal (non-profiling) benchmark against the server (e.g., a shell script or `python3 benchmark/...`). This is needed for both the profiling run and the final benchmark run.

Also ask:
- **Model name** (for labeling output, e.g., "DSv4", "Llama-3-70B")
- **Accuracy threshold** (default: 0.88 for DSv4 — confirm with user)
- **Port** the server is listening on (default: 8000)

Do NOT proceed until you have the server script, client script, model name, and accuracy threshold.

## Step 1: Accuracy Test

Run the GSM8K accuracy benchmark:

```bash
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 1000 --num-shots 5 --port <PORT>
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

## Step 2: Profiling Run

The purpose of this step is to confirm the code change actually has the intended effect at the kernel/module level. This is NOT a performance benchmark — it is a correctness check for the optimization.

### 2a: Prepare the profiling client command

Take the user's client benchmark script from Step 0 and modify it for profiling:

1. **Add `--profile`** flag to the client command
2. **Change `num_prompts`** to `num_prompts=$((max_concurrency * 2))` — profiling only needs a short run, not a full benchmark. If the client script uses a variable like `num_prompts=N`, replace it with `num_prompts=$((max_concurrency * 2))`.

Show the modified profiling command to the user and confirm before running.

### 2b: Run the profiling client

Execute the modified client command. The trace files will be generated under `/tmp/` by default (the server writes them there).

### 2c: Locate and copy the trace files

After the profiling run completes:

1. Find the latest trace directory under `/tmp/` (look for directories matching the SGLang trace pattern — timestamped folders containing `.trace.json.gz` files)
2. Copy the trace folder to a stable location for analysis:
   ```bash
   cp -r /tmp/<trace-folder> $HOME/dsv4/
   ```
3. Identify the GPU 0 trace file: the file matching `*TP-0.trace.json.gz`

### 2d: Analyze the trace

Run the trace module analyzer on the GPU 0 trace:

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py \
  $HOME/dsv4/<trace-folder>/<timestamp>-TP-0.trace.json.gz \
  -o analysis_<change_description>
```

Where `<change_description>` is a short slug describing the code change (e.g., `fused_mla_kernel`, `bf16_attention`).

### 2e: Interpret profiling results

Review the trace analysis output:
- Look for the specific kernels or modules that the code change was supposed to affect
- Confirm the change is reflected in the trace (e.g., new kernel appears, old kernel is gone, time distribution changed)
- If the change is NOT visible in the trace, warn the user — the optimization may not be working as intended

**Report to the user:**
- Summary of what the profiling shows
- Whether the code change is confirmed working at the kernel level
- Key numbers (kernel times, module breakdown) relevant to the change

Ask the user to confirm the profiling results look correct before proceeding.

## Step 3: Normal Benchmark

Run the user's original (unmodified) client benchmark script from Step 0 — this is the full performance benchmark without `--profile` and with the original `num_prompts`.

```bash
# Run the user's client benchmark script as-is
<user's client script>
```

**Record the results:**
- Parse output for key metrics: throughput (tokens/s), latency (TTFT, ITL, E2E), or whatever the benchmark reports
- Save all numbers for PR submission

## Step 4: Summary Report

Present a complete validation summary to the user:

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
- These results are ready for PR submission (the `/pr` skill will ask for accuracy and benchmark data — the numbers from this validation can be used directly)
- The profiling analysis file path for reference
- If any step failed or was skipped, highlight it clearly
