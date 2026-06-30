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

## Typical questions → mode

- "decode bound?" → `parse-trace` decode mode, conc 4
- "prefill FMHA?" → prefill mode, long IL
- "did fused kernel show up?" → grep trace for kernel name; compare before/after xlsx

## Parser location

`$AGENT_BOX_DIR/profile/trace_module_analyzer.py` (submodule)
