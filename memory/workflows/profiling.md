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

A perf **sweep** keeps the whole frontier (`4 8 16 32 64 128 256`). A **profiling
pass** does not — profiling distorts throughput and produces huge traces, so capture
only the two anchor points **c4 / c64**, each answering a different question:

| Conc | Regime | What it tells you |
|---|---|---|
| **c4** | latency-bound / near-idle | Per-layer decode critical path; kernel launch + small-GEMM overhead. Best for decode deep-dives (won't OOM host). |
| **c64** | throughput / near-saturation | Steady-state serving cost and where batching contention starts to bite; the representative high-conc point for kernel-improvement confirmation. |

**Rule:** run the full frontier for the perf table, then profile only **c4 / c64**.
These are also the kernel-dev confirmation anchors — see [[kernel-dev]]. Don't profile
every concurrency — it wastes hours and disk for no extra signal.

**Capture cost:** a profiling pass uses `num_prompts = conc × 2`
(`PROFILE_NUM_PROMPTS_MULT=2`) — just enough for a representative trace, not a
measurement. Workload shape stays fixed per [[workloads]] (default `canonical-8k`;
`diag-1k` for a cheap capture).

## Typical questions → mode

- "decode bound?" → `parse-trace` decode mode, **c4**
- "steady-state serving cost?" → **c32**
- "high-conc regression / contention?" → **c128**
- "prefill FMHA?" → prefill mode, long IL
- "did fused kernel show up?" → grep trace for kernel name; compare before/after xlsx

## Parser location

`$AGENT_BOX_DIR/profile/trace_module_analyzer.py` (submodule)
