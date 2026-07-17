---
name: parse-trace
description: Parse a profiling trace file with trace_module_analyzer.py. Supports default mode (full analysis) and specialized prefill/decode modes with configurable detail modules and instances.
category: measure
---

# Parse Trace

Run `trace_module_analyzer.py` on a trace file to produce an Excel analysis report. Three modes:

- **Default**: full analysis with auto-detected detail modules
- **Specialized**: prefill or decode mode with specific `--detail-module` and `--detail-instance` settings
- **Both**: runs decode then prefill analysis sequentially, producing two Excel reports

## Usage

```
/parse-trace <trace_file.json.gz>
```

The user must provide the trace file path as input. If not provided, ask for it.

## Step 0: Validate input

### 0a: Get the trace file path

The trace file path should come from the user's input (`$ARGUMENTS`). If not provided, ask:

```
AskUserQuestion: "What is the path to the trace file (.json.gz or .json)?"
```

Verify the file exists:
```bash
ls -la <trace_file>
```

If the file doesn't exist, report the error and stop.

### 0b: Detect trace type from filename

Infer the trace type from the filename:
- Contains `-DECODE` → decode trace
- Contains `-EXTEND` → prefill/extend trace
- Neither → unknown (will ask user)

### 0c: Resolve model → detail-module/instance from the model card (card-driven)

**Do this before asking anything.** The `--detail-module` / `--detail-instance`
values are model-specific and live in the model card — do NOT default to DSv4.

1. Infer the model from the trace path (e.g. `qwen3.5-mxfp4/…` → qwen35-mxfp4) or
   from `$ARGUMENTS`. If ambiguous, this is the one thing worth asking.
2. Read the matching card under `memory/models/` (see [[../../memory/models/INDEX.md]]).
   Each card's **Profiling** section gives the correct `--detail-module` and
   `--detail-instance`. Known values:

   | Model | `--detail-module` (decode & prefill) | `--detail-instance` |
   |---|---|---|
   | DSv4 / R1 | decode `Layer` / prefill `DeepseekV4DecoderLayer` | 59 60 61 62 / 31 32 |
   | qwen35-mxfp4 | `Qwen3_5LinearDecoderLayer` | 0 1 |

3. If the card resolves the module + instances, **skip the questions in Step 1 and
   go straight to Step 2** with those values. Only fall through to Step 1 when the
   model is unknown or the card has no profiling entry.
4. On first run for a model, confirm the chosen instances against the **Module
   Tree** sheet after the analyzer completes (Step 4), and update the card if they differ.

## Step 1: Ask the user for configuration (only if Step 0c couldn't resolve)

Skip this step entirely when the model card already provided the module/instances.
Otherwise use `AskUserQuestion` with these questions:

### Question 1: Analysis mode

```
"What analysis mode should be used?"
```

Options:
- **Both (prefill + decode) (Recommended)** — runs decode then prefill analysis sequentially, producing `analysis_decode.xlsx` and `analysis_extend.xlsx`. Best for combined traces.
- **Default (full analysis)** — runs with `-o analysis.xlsx`, auto-detects detail modules. Good for first-pass analysis.
- **Prefill/Extend mode** — uses `--detail-module DeepseekV4DecoderLayer --detail-instance 31 32`. For analyzing prefill/extend traces.
- **Decode mode** — uses `--detail-module Layer --detail-instance 59 60 61 62`. For analyzing decode traces.

If the trace type was detected in Step 0b, pre-select the matching mode (but still let the user override).

### Question 2: Custom detail instances?

```
"Use default detail instances, or specify custom ones?"
```

Options:
- **Use defaults** — prefill: instances 31 32, decode: instances 59 60 61 62 (skip this for default mode)
- **Specify custom instances** — user provides instance IDs

Only ask this if the user chose prefill or decode mode. If they choose custom, ask for the instance IDs as a follow-up.

### Question 3: Output file name and path

```
"Where should the output Excel file be saved?"
```

Options:
- **Same directory as trace file** — saves the xlsx next to the trace (recommended)
- **Custom path** — user specifies full output path

For the filename, propose a sensible default based on mode:
- Default mode: `analysis.xlsx`
- Prefill mode: `analysis_extend.xlsx`
- Decode mode: `analysis_decode.xlsx`

Show the proposed full output path and let the user confirm or change it.

### Question 4: Detail module override (specialized modes only)

```
"Which module type should be used for detail sheets?"
```

Options:
- **DeepseekV4DecoderLayer** — standard for DSv4 prefill/extend (Recommended for prefill)
- **Layer** — standard for DSv4 decode (Recommended for decode)
- **Custom** — user specifies a different module name

Only ask this if user chose prefill or decode mode. The user may need to specify a different module name for non-DSv4 models.

## Step 2: Build and confirm the command

### Both mode

Runs two sequential analyses on the same trace file:

1. **Decode analysis:**
```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <trace_file> \
    -o <output_dir>/analysis_decode.xlsx \
    --detail-module Layer --detail-instance 59 60 61 62
```

2. **Prefill/Extend analysis:**
```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <trace_file> \
    -o <output_dir>/analysis_extend.xlsx \
    --detail-module DeepseekV4DecoderLayer --detail-instance 31 32
```

Show both commands to the user and confirm before running. Run them sequentially (not in parallel) since both read the same large trace file.

### Default mode

```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <trace_file> \
    -o <output_path>
```

### Prefill/Extend mode

```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <trace_file> \
    -o <output_path> \
    --detail-module <module_name> \
    --detail-instance <instance_ids>
```

Default for prefill:
- `--detail-module DeepseekV4DecoderLayer`
- `--detail-instance 31 32`
- `-o analysis_extend.xlsx`

### Decode mode

```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <trace_file> \
    -o <output_path> \
    --detail-module <module_name> \
    --detail-instance <instance_ids>
```

Default for decode:
- `--detail-module Layer`
- `--detail-instance 59 60 61 62`
- `-o analysis_decode.xlsx`

**Show the full command to the user and confirm before running.**

## Step 3: Run the analyzer

Execute the command. This typically takes 30-120 seconds depending on trace size.

```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <trace_file> \
    -o <output_path> \
    [--detail-module <module>] \
    [--detail-instance <ids>]
```

Monitor the output for:
- Progress messages (module tree building, kernel correlation, etc.)
- ROCm trace fix warnings (auto-applied unless `--no-rocm-fix`)
- Any errors

## Step 4: Report results

After the analyzer completes:

### 4a: Verify output

```bash
ls -la <output_path>
```

### 4b: Read and summarize key findings from the console output

Report:
- Number of iterations detected (prefill vs decode)
- Median iteration time
- Top modules by GPU time
- Number of detail sheets generated
- Any warnings (ROCm fixes, uncategorized kernels, etc.)

### 4c: Print summary

```
=== Trace Analysis Complete ===

Trace:    <trace_file>
Output:   <output_path>
Mode:     <default/prefill/decode>
Platform: <detected from trace — MI355/B200/etc>

Iterations: <N prefill, M decode>
Median decode iteration: <X> ms
Median prefill iteration: <Y> ms (if applicable)

Top modules by GPU time:
  1. <module> — <time> us (<percent>%)
  2. <module> — <time> us (<percent>%)
  3. <module> — <time> us (<percent>%)

Detail sheets generated for:
  - <module> (instances: <ids>)

Next steps:
  - Compare with another trace: /compare-kernels <this_output.xlsx> <other.xlsx>
  - Deeper analysis: /profile <output_path>
```

## Notes

- The analyzer auto-detects ROCm traces and applies hipGraphLaunch flow event fixes automatically.
- For large traces (>500MB), analysis may take several minutes.
- The `--detail-instance` flag selects which specific layer instances get kernel-by-kernel detail sheets in the Excel. Choose instances that represent a "typical" layer.
- Default instances (31/32 for prefill, 59-62 for decode) are chosen for DSv4 architecture. Other models may need different instance IDs.
- The `--detail-module` flag selects the module type for detail sheets. `DeepseekV4DecoderLayer` is the outer decoder layer for prefill (contains full module tree); `Layer` is the CUDA-graph-replayed wrapper used in decode.
