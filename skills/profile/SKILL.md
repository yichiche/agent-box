---
name: profile
description: Analyze SGLang profiling traces from MI355/B200, run trace_module_analyzer.py, and find representative prefill/decode iterations for cross-platform comparison.
---

# Profile Trace Analysis — MI355 vs B200

Analyze existing SGLang profiling traces from MI355 and/or B200, run the trace module analyzer, and identify representative prefill/decode iterations for cross-platform comparison.

## Prerequisites

- Profiling traces (`.trace.json.gz` or `.trace.json`) from MI355 and/or B200 runs
- Traces may be full (prefill+decode combined) or stage-separated (prefill-only / decode-only)
- Python with `openpyxl` installed (`pip install openpyxl`)

## Step-by-step Workflow

### Step 1: Locate the traces

Ask the user for trace file paths. Traces typically come as:
- **Combined**: a single `.trace.json.gz` containing both prefill and decode phases
- **Stage-separated**: separate files for prefill and decode (generated with `--profile-by-stage`)

Expect traces from one or both platforms:
- MI355 (ROCm/HIP) — the analyzer auto-detects and fixes ROCm trace quirks (hipGraphLaunch flow events)
- B200 (CUDA)

### Step 2: Run the trace module analyzer

```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py <trace_file> -o <output.xlsx>
```

Run this for each trace file. The analyzer:
1. Loads and categorizes all trace events (kernels, cpu_ops, modules)
2. Builds the nn.Module hierarchy tree
3. Correlates GPU kernels to modules via cuda_runtime correlation IDs
4. Detects prefill vs decode phases automatically (via `forward_extend`/`forward_decode` markers)
5. Aggregates per-module statistics
6. Exports an Excel report with summary + detail sheets

**Key flags:**
- `-o <path.xlsx>` — output Excel report path (if relative, saved next to the trace)
- `--max-detail-modules N` — number of module types to generate detail sheets for (default: 3, 0=all)
- `--detail-module <name>` — specific module types for kernel-by-kernel detail (e.g. `DeepseekV3DecoderLayer`)
- `--module-index <N>` — pick a specific instance instead of the median
- `--detail-instance <id> [<id>...]` — specific instance IDs for detail sheets
- `--model-info` — generate interactive module tree HTML visualization
- `--no-rocm-fix` — disable automatic ROCm trace fix
- `-v` — verbose/debug logging

**Example for MI355 and B200 traces:**
```bash
# MI355 trace (ROCm auto-fix applied automatically)
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
  /path/to/mi355_trace.json.gz -o mi355_analysis.xlsx

# B200 trace
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
  /path/to/b200_trace.json.gz -o b200_analysis.xlsx
```

### Step 3: Find representative prefill/decode iterations

The analyzer auto-detects prefill vs decode phases and tags each module instance accordingly. To find good representatives:

1. **Check the console output** — the analyzer prints per-phase module counts and timing statistics
2. **Open the Excel report** — the "Module Tree" sheet shows each module instance with its phase tag (prefill/decode) and duration
3. **Use median instances** — by default, the analyzer picks the instance closest to the median duration for detail sheets. This is typically the best representative.
4. **For specific instances**, use `--detail-instance` to inspect particular iterations:
   ```bash
   python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
     trace.json.gz -o report.xlsx --detail-instance 59 60 61 62
   ```

### Step 4: Compare MI355 vs B200

With Excel reports from both platforms, compare:

1. **Use `/compare-kernels`** — the compare-kernels skill takes two xlsx files and compares kernel categories and timings
2. **Manual comparison** — open both Excel files side-by-side and compare:
   - Total prefill time vs decode time per platform
   - Per-module breakdown (which modules are slower on which platform)
   - Individual kernel timings within the detail sheets
3. **Recategorize if needed** — apply consistent kernel categories across both reports:
   ```bash
   python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
     --recategorize mi355_analysis.xlsx b200_analysis.xlsx
   ```

### Step 5: Report findings

Summarize:
- Which platform is faster for prefill vs decode
- Top-N modules/kernels with the largest gap between platforms
- Specific optimization opportunities (kernels that are disproportionately slow on MI355 vs B200)

## Example Full Run

```bash
# Analyze MI355 decode trace
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
  /data/traces/mi355_decode.trace.json.gz \
  -o /data/traces/mi355_decode_analysis.xlsx \
  --max-detail-modules 5

# Analyze B200 decode trace
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
  /data/traces/b200_decode.trace.json.gz \
  -o /data/traces/b200_decode_analysis.xlsx \
  --max-detail-modules 5

# Compare the two
# → use /compare-kernels mi355_decode_analysis.xlsx b200_decode_analysis.xlsx
```

## Viewing Traces Directly

If you need to inspect the raw trace visually:
- **Perfetto UI**: https://ui.perfetto.dev/ (drag and drop `.trace.json.gz`)
- **Chrome tracing**: `chrome://tracing`
- **Interactive module tree**: add `--model-info` flag to the analyzer command
