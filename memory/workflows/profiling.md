# Profiling Workflow

## Capture

1. Server already running (from model's `run_*_perf.sh`)
2. `/generate-profile` — launches trace at fixed IL/OL/conc
3. Or manual: client with `--profile` / `--profile-output-dir` (see [[../gotchas/container-bench-flags]])

**Rule:** profiling is a **separate pass** — it distorts throughput at that concurrency; don't mix into perf table.

## Analyze

| Output | Tool |
|---|---|
| Raw Chrome trace | `profile/trace_module_analyzer.py` |
| Prefill/decode Excel | `/parse-trace` with mode `prefill` or `decode` |
| Kernel category diff | `/compare-kernels` two xlsx files |
| Action items | `/perf-summary` |

Guide: `profile/profile.md`

## Which concurrency to profile

A perf **sweep** keeps the whole frontier (`4 8 16 32 64 128 256`). A **deep
profile** does not — profiling distorts throughput and produces huge traces, so
capture only the three anchor points **c4 / c32 / c128**, each answering a
different question:

| Conc | Regime | What it tells you |
|---|---|---|
| **c4** | latency-bound / near-idle | Per-layer decode critical path; kernel launch + small-GEMM overhead. Best for decode deep-dives (won't OOM host). |
| **c32** | throughput knee | The point where batching starts to saturate; most representative of steady-state serving. |
| **c128** | saturation | Contention, memory pressure, scheduler/queueing effects; where high-conc regressions surface. |

**Rule:** run the full frontier for the perf table, then deep-profile only
c4/c32/c128. Don't profile every concurrency — it wastes hours and disk for no
extra signal. Workload shape stays fixed per [[workloads]] (default `canonical-8k`;
`diag-1k` for a cheap capture).

## Typical questions → mode

- "decode bound?" → `parse-trace` decode mode, **c4**
- "steady-state serving cost?" → **c32**
- "high-conc regression / contention?" → **c128**
- "prefill FMHA?" → prefill mode, long IL
- "did fused kernel show up?" → grep trace for kernel name; compare before/after xlsx

## Parser location

`$AGENT_BOX_DIR/profile/trace_module_analyzer.py` (submodule)
