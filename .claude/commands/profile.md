---
description: Analyze profiling outputs (trace Excel, evaluation CSV, raw traces) to answer questions about kernel performance, layer breakdown, and model behavior
---

# Profile Analysis Guide

You are an expert at analyzing SGLang GPU profiling data. Use this guide to understand the profiling toolchain, locate the right files, and answer questions about kernel performance, layer structure, and model behavior.

## Profiling Toolchain Overview

The profiling tools live in `/home/yichiche/agent-box/profile/`. They form a pipeline:

```
Raw Trace (.trace.json.gz)
  │
  ├─► trace_analyzer.py ──► profile.csv.xlsx  (per-layer kernel breakdown)
  │                     └──► trace_analyzer.log (analysis log)
  │
  ├─► evaluate_parsing.py ──► evaluation_summary.csv (quality scores)
  │                       └──► evaluate_parsing.log  (diagnostics)
  │
  └─► compare_traces.py ──► diff_report.xlsx (kernel diff between two versions)

model_inspector.py ──► (standalone) static model structure from source code
```

## Output File Formats

### profile.csv.xlsx — Per-Layer Kernel Breakdown (Primary Analysis Output)

An Excel workbook produced by `trace_analyzer.py`. Contains:

- **Summary sheet**: Overall stats — total kernel time, prefill/decode split, time breakdown by kernel type (attention, MoE, quantization, communication, linear, memory, other), and a per-layer table with columns: Layer Index, Layer Type, Stage, Total Time, Kernel Count, plus per-type time columns.
- **Layer_N sheets** (one per detected layer): Detailed kernel sequence for that layer with columns: `#` (position), `Duration (us)`, `%` (fraction of layer), `Type` (kernel type), `Short Name` (simplified), `Kernel Name` (full GPU kernel name).

**Key layer types**: `MLA+MoE`, `MLA+FC`, `MHA+MoE`, `MHA+FC`, `GDN+MoE`, `GDN+FC`
**Key stages**: `prefill`, `decode`
**Key kernel types**: `attention`, `moe`, `quantization`, `communication`, `linear`, `memory`, `other`

### evaluation_summary.csv — Parsing Quality Assessment

CSV with columns: `section, metric, score, grade, detail`. Sections:

- **structural**: 4 metrics (S1-S4) each scored 0-100:
  - S1 Prefill Ordering: No decode layers appear before first prefill
  - S2 Architecture Signature: First prefill round matches expected model pattern
  - S3 Round Consistency: Decode rounds have consistent layer counts
  - S4 Type Sequence: Kernel type pattern repeats within each round
- **group**: Per-(stage, layer_type) metrics — `layers=N, time_cv=X, kern_cnt_mode=N(%), kern_pat=X%`
- **overall**: Weighted composite score

### evaluate_parsing.log — Diagnostic Details

Human-readable quality report including the structural test results, per-group diagnostics table, and outlier analysis with specific layer indices.

### trace_analyzer.log — Analysis Log

Detailed log from the trace analyzer run including: events loaded, layer boundaries detected, merge operations, phase breakdown, timing summaries.

### *.trace.json.gz — Raw Torch Profiler Trace

Compressed JSON containing all GPU kernel events. Typically 30-100+ MB. Fields per event include: name, timestamp, duration, args (grid, block dims). This is the source input to `trace_analyzer.py`.

## Typical Benchmark Run Directory Layout

Benchmark runs are stored under `/home/yichiche/benchmark_runs/`. A typical profiling run directory:

```
<version>_<hardware>_<date>_TP<N>_profile/
├── bench_c1.log, bench_c2.log, ...   (server logs per concurrency level)
├── bench_results.jsonl                 (detailed benchmark results)
├── bench_summary.csv                   (throughput/latency per concurrency)
├── server.log, client.log              (server and client logs)
├── orchestrator_output.log             (orchestration log)
├── version_snapshot.json               (version metadata)
└── trace_analysis/
    ├── *.trace.json.gz                 (raw trace file)
    ├── profile.csv.xlsx                (trace analysis output)
    ├── trace_analyzer.log              (analysis log)
    ├── evaluation_summary.csv          (quality scores)
    └── evaluate_parsing.log            (quality diagnostics)
```

## How to Analyze Profiling Data

### Step 1: Locate the files

If the user provides a path, use it directly. Otherwise, look for benchmark runs:
```bash
ls /home/yichiche/benchmark_runs/
```

Within a run directory, profiling outputs are in the `trace_analysis/` subdirectory.

### Step 2: Read the evaluation summary first

Start with `evaluation_summary.csv` to check data quality. If the overall score is below 85, warn the user that the trace parsing may be unreliable and point out which structural metrics failed.

### Step 3: Read the profile Excel for kernel analysis

Use `openpyxl` or read the file to extract:
- **Overall time split**: From the Summary sheet, get prefill vs decode percentage
- **Kernel type breakdown**: Which kernel types dominate (attention? communication? MoE?)
- **Per-layer patterns**: Read specific Layer_N sheets to see the kernel sequence

### Step 4: Answer the user's question

Common questions and how to answer them:

| Question | Where to look |
|----------|---------------|
| "What's the prefill/decode split?" | Summary sheet overall stats |
| "Which kernel type takes the most time?" | Summary sheet type breakdown |
| "What kernels run in layer N?" | Layer_N sheet in profile.csv.xlsx |
| "Is the trace parsing reliable?" | evaluation_summary.csv overall score |
| "How many layers were detected?" | Summary sheet layer table row count |
| "Which layers are outliers?" | evaluate_parsing.log outlier section |
| "Compare two versions" | Use compare_traces.py or read two profile.csv.xlsx files |
| "What's the model architecture?" | Summary sheet layer types (MLA/MHA/GDN, MoE/FC pattern) |

## Running the Tools

If the user needs to generate new analysis from a raw trace:

```bash
# Analyze a trace file
python3 /home/yichiche/agent-box/profile/trace_analyzer.py /path/to/trace.json.gz -o output.xlsx

# With verbose output
python3 /home/yichiche/agent-box/profile/trace_analyzer.py /path/to/trace.json.gz -o output.xlsx -v

# Debug specific layers
python3 /home/yichiche/agent-box/profile/trace_analyzer.py /path/to/trace.json.gz --debug-layers 0,1,2,3

# Show layer breakdown in terminal
python3 /home/yichiche/agent-box/profile/trace_analyzer.py /path/to/trace.json.gz --show-layer-terminal

# Evaluate parsing quality
python3 /home/yichiche/agent-box/profile/evaluate_parsing.py output.xlsx --json

# Compare two trace analyses
python3 /home/yichiche/agent-box/profile/compare_traces.py file_a.xlsx file_b.xlsx --output diff.xlsx --verbose

# Inspect model structure (no GPU needed)
python3 /home/yichiche/agent-box/profile/model_inspector.py /path/to/model.py --config /path/to/config.json
```

## Key Concepts

- **Prefill**: The initial phase processing the full input prompt. Typically compute-bound with large batch GEMMs.
- **Decode**: Auto-regressive token generation phase. Typically memory-bandwidth-bound.
- **MLA (Multi-head Latent Attention)**: DeepSeek-V3 style compressed KV attention.
- **MHA (Multi-head Attention)**: Standard multi-head attention (Llama, Qwen, Grok-2).
- **GDN (Gated Delta Network)**: Qwen3-Coder-Next linear attention variant.
- **MoE (Mixture of Experts)**: Sparse expert layers with routing (DeepSeek, Grok-2, Qwen3).
- **FC (Fully Connected)**: Dense MLP layers (first/last few layers in MoE models).
- **Half-layers**: Models with dual norms (Grok-2, Qwen3) produce two layer boundaries per transformer block. These are auto-merged by default.
- **ALLREDUCE**: Collective communication kernel marking tensor-parallel synchronization points.

## Debugging Trace Analysis Issues

If the user reports issues with trace analysis or wants to add support for a new model, refer to the detailed guide:
- Read `/home/yichiche/agent-box/profile/trace-analyzer.md` for pattern table update instructions
- Read `/home/yichiche/agent-box/profile/tests/README.md` for testing procedures

When modifying `trace_analyzer.py`, always run the test suite:
```bash
cd /home/yichiche/agent-box/profile && python -m pytest tests/ -v
```

## Important Notes

- Profile Excel files can be large (1+ MB with 20,000+ layer sheets). Read selectively — start with the Summary sheet, then drill into specific layers.
- Raw trace files are very large (30-100+ MB compressed). Do NOT attempt to read them directly. Use `trace_analyzer.py` to process them.
- When comparing versions, use `compare_traces.py` rather than manual diff — it handles layer matching, length-mismatch filtering, and confidence scoring.
- The evaluation score uses 85 as the default pass threshold. Scores below this indicate structural parsing issues that need investigation.

If the user provided `$ARGUMENTS`, treat it as their specific question or the path to profiling data to analyze.
