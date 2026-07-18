# Workload Presets

Named IL/OL workloads for benchmarking and profiling. **Always name the preset**
— never launch a sweep with ad-hoc IL/OL and then compare against numbers taken at
a different shape. The variables are `INPUT_LEN` / `OUTPUT_LEN` (see
[[benchmark]]; `IL`/`OL` are NOT read by `perf_sweep.sh`).

## Presets

| Preset | INPUT_LEN | OUTPUT_LEN | Valid for | NOT valid for |
|---|---|---|---|---|
| **`canonical-8k`** | 8192 | 1024 | Official perf claims, PR "Benchmarking", InferenceX/ATOM comparison | — |
| **`diag-1k`** | 1024 | 1024 | correctness, crash repro, scaling shape, profiler capture | any perf-improvement claim |

`canonical-8k` is the default when `INPUT_LEN`/`OUTPUT_LEN` are omitted, and is the
shape all existing infrastructure aligns on (`perf_sweep.sh` defaults,
`validate_patches.js`, the InferenceX baseline).

## The benchmark set — run BOTH shapes by default

Unless the user names a single shape, a **benchmark runs both presets**:

| Set member | Preset | Role |
|---|---|---|
| `diag-1k` | 1024 / 1024 | secondary shape — reported alongside for the low-IL / decode-heavy picture |
| `canonical-8k` | 8192 / 1024 | **the claim shape** |

- "benchmark X" / "full sweep" / "e2e confirm" with no shape given ⇒ run **`diag-1k`
  AND `canonical-8k`**, report both tables.
- **The perf claim is still made only on `canonical-8k`** (rule 1 below). `diag-1k`
  is reported for context — never quote its delta as the speedup.
- This is a *reporting* default, not a relaxation of the claim rule: two tables, one
  claimable number.

## num_prompts per pass

`num_prompts = concurrency × MULT`, and the multiplier depends on the pass:

| Pass | MULT | Env |
|---|---|---|
| **Benchmark / sweep** (measurement) | **×10** | `NUM_PROMPTS_MULT=10` |
| **Profiling capture** (trace only) | **×2** | `PROFILE_NUM_PROMPTS_MULT=2` |

A profiling pass only needs enough requests to fill a representative trace, so it uses
the cheaper ×2; a measurement pass uses ×10 for a stable throughput/latency number.
These are the `perf_sweep.sh` defaults.

> These presets and the claim rules below are **promoted from the journal**, not new
> conventions — they formalize the "1K/1K is diagnostic, 8K/1K is the perf shape,
> ≥5% to ship" practice that was already validated in session history.

## Semantic rules

1. **A perf improvement may only be claimed on `canonical-8k`** with a delta
   **≥ 5%** over baseline at matching concurrency and config (see [[benchmark]]).
2. **`diag-1k` is a diagnostic shape only.** Use it to check that a change is
   correct, doesn't crash, scales, or to grab a cheap/fast profiler trace. Never
   report a `diag-1k` delta as an 8K-workload speedup — the kernel mix differs.
3. **One preset per comparison.** Baseline and after must use the *same* preset.
   Mixing shapes invalidates the delta.
4. If a genuinely different shape is needed (e.g. long-output decode study), add a
   new named preset here first, then use it — don't smuggle it in as an unnamed
   override.

## Copy-paste — accuracy-gated sweep (`/perf-sweep`)

```bash
# canonical-8k — official perf sweep
MODEL=/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 \
GPUS="0,1" INPUT_LEN=8192 OUTPUT_LEN=1024 \
  bash ~/.claude/skills/perf-sweep/perf_sweep.sh

# diag-1k — correctness / scaling / cheap profiler capture
MODEL=/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 \
GPUS="0,1" INPUT_LEN=1024 OUTPUT_LEN=1024 \
  bash ~/.claude/skills/perf-sweep/perf_sweep.sh
```

Resolve `MODEL` / `GPUS` / server flags per model via [[../models/INDEX]] and
`/gpu-status`. Concurrency frontier defaults to `4 8 16 32 64 128 256`; for
deep profiling restrict to `4 32 128` — see [[profiling]].

## Related

- [[benchmark]] — sweep flow, env vars, output layout
- [[profiling]] — which concurrency points to profile
- [[../models/INDEX]] — model → server/client/threshold
