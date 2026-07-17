# TP2 live-server profiling can crash via gloo broadcast (intermittent)

Profiling a **live TP2 server** through `/start_profile` (what `bench_serving
--profile` triggers) is **crash-prone with the overlap scheduler**: `start_profile`
plus concurrent requests can crash the server via a gloo `broadcast_pyobj`
connection-reset. This is **profiling/TP fragility, NOT a kernel fault** — the served
model is fine.

## It's intermittent, not deterministic

- Journal case (2026-07-10): a live conc=4 torch-profiler trace could not be captured
  — `start_profile` + concurrent reqs crashed the TP2 server via gloo.
- Counter-case (2026-07-16): a `bench_serving --profile --profile-by-stage` run at
  conc=4 on TP2 **succeeded** and produced clean DECODE/EXTEND traces.

So it works sometimes and dies sometimes — treat it as fragile, especially on a
**shared node** (GPUs re-fill intermittently). If it crashes, don't chase it as a
kernel bug.

## Fallback for a clean decode trace

- Use **offline** `bench_one_batch --profile` (no live serving / overlap scheduler), or
- Profile a **single rank** instead of the full TP2 group.

If you do use live `--profile`, set the server's `SGLANG_TORCH_PROFILER_DIR` at
launch and keep the profiling run short (few steps, low concurrency) to minimize the
window where the gloo reset can hit.

Source: [[../journal/2026-07/-sgl-workspace-aiter__a16w4-net-benefit-verdict]].
Related: [[../workflows/profiling]], [[aiter-jit-baton-vram]].
