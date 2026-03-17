---
description: Analyze profiling outputs (trace Excel, evaluation CSV, raw traces) to answer questions about kernel performance, layer breakdown, and model behavior
---

# Profile Analysis Guide

You are an expert at analyzing SGLang GPU profiling data. Use this guide to understand the profiling toolchain, locate the right files, and answer questions about kernel performance, layer structure, and model behavior.

## Profiling Toolchain Overview

The profiling tools live in `/home/yichiche/agent-box/profile/`. The primary tool is `trace_module_analyzer.py`, which uses nn.Module correlation to classify GPU kernels by their owning module.

```
Raw Trace (.trace.json.gz)
  │
  ├─► trace_module_analyzer.py ──► report.xlsx  (module-level kernel breakdown)
  │     (optionally applies fix_rocm_trace_flow.py for ROCm traces)
  │
  └─► model_inspector.py ──► model structure tree / architecture diagrams
        (standalone, or enriches trace_module_analyzer output)

Legacy (used by perf-regression pipeline):
  trace_analyzer.py  ──► profile.csv.xlsx
  evaluate_parsing.py ──► evaluation_summary.csv
```

## Output File Formats

### report.xlsx — Module-Level Kernel Breakdown (trace_module_analyzer output)

An Excel workbook with module-level kernel breakdown:
- **Summary sheet**: Overall stats — total time, module type breakdown, top kernels per module type
- **Module type sheets**: Per-module-type detail with kernel lists, timing, and percentages
- **Model Info sheet** (with `--config`): HuggingFace model configuration details

### profile.csv.xlsx — Per-Layer Kernel Breakdown (Legacy trace_analyzer output)

An Excel workbook produced by `trace_analyzer.py`. Contains:

- **Summary sheet**: Overall stats — total kernel time, prefill/decode split, time breakdown by kernel type (attention, MoE, quantization, communication, linear, memory, other), and a per-layer table.
- **Layer_N sheets** (one per detected layer): Detailed kernel sequence for that layer.

**Key layer types**: `MLA+MoE`, `MLA+FC`, `MHA+MoE`, `MHA+FC`, `GDN+MoE`, `GDN+FC`
**Key stages**: `prefill`, `decode`
**Key kernel types**: `attention`, `moe`, `quantization`, `communication`, `linear`, `memory`, `other`

### evaluation_summary.csv — Parsing Quality Assessment (Legacy)

CSV with columns: `section, metric, score, grade, detail`. Produced by `evaluate_parsing.py` from `trace_analyzer.py` output. Used by perf-regression pipeline.

### *.trace.json.gz — Raw Torch Profiler Trace

Compressed JSON containing all GPU kernel events. Typically 30-100+ MB. This is the source input to the analysis tools.

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
    ├── profile.csv.xlsx                (legacy trace_analyzer output)
    ├── trace_analyzer.log              (legacy analysis log)
    ├── evaluation_summary.csv          (legacy quality scores)
    └── evaluate_parsing.log            (legacy quality diagnostics)
```

## How to Analyze Profiling Data

### Step 1: Locate the files

If the user provides a path, use it directly. Otherwise, look for benchmark runs:
```bash
ls /home/yichiche/benchmark_runs/
```

### Step 2: Run trace_module_analyzer for new analysis

```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py /path/to/trace.json.gz -o report.xlsx -v
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py trace.json.gz -o report.xlsx --config config.json
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py trace.json.gz --show-tree
```

For existing legacy analysis, read the `profile.csv.xlsx` and `evaluation_summary.csv` files.

### Step 3: Answer the user's question

| Question | Where to look |
|----------|---------------|
| "What modules take the most time?" | trace_module_analyzer summary sheet |
| "What kernels run in module X?" | trace_module_analyzer --detail-module X |
| "What's the model architecture?" | model_inspector --profiler-tree or --arch-diagram |
| "What's the prefill/decode split?" | Legacy: profile.csv.xlsx Summary sheet |
| "Is the trace parsing reliable?" | Legacy: evaluation_summary.csv overall score |
| "Which layers are outliers?" | Legacy: evaluate_parsing.log outlier section |

## Running the Tools

```bash
# Primary: Module-based trace analysis
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py /path/to/trace.json.gz -o report.xlsx

# ROCm trace fix (standalone; also auto-applied by trace_module_analyzer)
python3 /home/yichiche/agent-box/profile/fix_rocm_trace_flow.py trace.json.gz -o trace_fixed.json.gz

# Model structure inspection
python3 /home/yichiche/agent-box/profile/model_inspector.py /path/to/model.py --config config.json
python3 /home/yichiche/agent-box/profile/model_inspector.py --trace trace.json.gz --arch-diagram

# Legacy: Rule-based trace analysis (used by perf-regression pipeline)
python3 /home/yichiche/agent-box/profile/trace_analyzer.py /path/to/trace.json.gz -o output.xlsx
python3 /home/yichiche/agent-box/profile/evaluate_parsing.py output.xlsx --json
```

## Key Concepts

- **Prefill**: The initial phase processing the full input prompt. Typically compute-bound with large batch GEMMs.
- **Decode**: Auto-regressive token generation phase. Typically memory-bandwidth-bound.
- **MLA (Multi-head Latent Attention)**: DeepSeek-V3 style compressed KV attention.
- **MHA (Multi-head Attention)**: Standard multi-head attention (Llama, Qwen, Grok-2).
- **GDN (Gated Delta Network)**: Qwen3-Coder-Next linear attention variant.
- **MoE (Mixture of Experts)**: Sparse expert layers with routing (DeepSeek, Grok-2, Qwen3).
- **FC (Fully Connected)**: Dense MLP layers (first/last few layers in MoE models).
- **ALLREDUCE**: Collective communication kernel marking tensor-parallel synchronization points.

## Important Notes

- Profile Excel files can be large (1+ MB with 20,000+ layer sheets). Read selectively — start with the Summary sheet, then drill into specific layers.
- Raw trace files are very large (30-100+ MB compressed). Do NOT attempt to read them directly. Use the analysis tools to process them.
- The legacy evaluation score uses 85 as the default pass threshold. Scores below this indicate structural parsing issues.

If the user provided `$ARGUMENTS`, treat it as their specific question or the path to profiling data to analyze.
