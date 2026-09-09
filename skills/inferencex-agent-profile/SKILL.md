---
name: inferencex-agent-profile
description: "Capture a torch profiler trace of an InferenceX AgentX (agent-mode) trace replay. Agent mode has no client-side --profile hook, so this reuses /inferencemax-benchmark's arm resolution and recipe execution unchanged and attaches a sidecar that drives SGLang's /start_profile endpoint once the replay is past aiperf warmup and inside its measurement window. Emits per-rank traces ready for /parse-trace and /kernel-profile-triage. Use when asked to profile agent mode, get kernel breakdown for AgentX, or find where agentic-coding replay time goes."
category: measure
---

# /inferencex-agent-profile — profile an AgentX replay

```
/inferencex-agent-profile <model-prefix> [tp=N] [conc=4,64] [steps=N] [dry-run]
```

`/inferencex-agent-profile qwen3.5 tp=2` · `/inferencex-agent-profile dsv4 conc=64 steps=200`

## The one idea

**Agent mode cannot be profiled from the client, so profile from the engine.**

`benchmark_lib.sh`'s `PROFILE=1` path (line ~682) only appends `--profile` to
the *fixed-seq* `bench_serving` client. The agentic path runs **aiperf**, which
has no such flag, and the recipe owns the entire server command line. So there
is no upstream seam to hang profiling on.

What there *is*: SGLang's `POST /start_profile`, which takes `output_dir` and
`num_steps` directly in the request body — no `SGLANG_TORCH_PROFILER_DIR`, no
server relaunch, and the profiler **self-terminates** after `num_steps` forward
steps. That means a purely external sidecar can capture a trace of the real
replay without touching the recipe, the server flags, or aiperf.

So this skill is `/inferencemax-benchmark` plus one watcher process. It
re-expresses no server flag, no client flag, and no arm. Everything comes from
`run_infmax.sh` → `resolve_arm.py` → the upstream recipe, exactly as in
[`/inferencemax-benchmark`](../inferencemax-benchmark/SKILL.md).

## What "the measurement window" means (and why the sidecar waits)

aiperf's agentic replay is three phases, and only the third is worth profiling:

1. **Dataset configuration** — 4–14 min, no traffic.
2. **Warmup** — `--warmup-requests-per-lane` (10, or **1** under
   `AIPERF_EXPERIMENTAL_FAST=1`) *one-token* requests per trajectory lane, then
   a drain with up to `--warmup-grace-period 1800`.
3. **Profiling / measurement** — the real trace replay for `--benchmark-duration`.

A naive "requests are running, start profiling" trigger fires in phase 2 and
captures a batch of 1-token requests that looks nothing like agentic coding.
The sidecar therefore waits for the **drain**: `sglang:num_running_reqs` goes
busy → 0 → busy again, and stays busy for `settle-polls × poll` (30 s default)
before it starts. `--trigger-regex` can pin an aiperf phase banner from
`recipe.log` instead, whichever fires first.

## Honor the global conventions in `~/agent-box/CLAUDE.md`

- **Anchors are conc4 and conc64** — that is the default `PROFILE_CONCS`. Do not
  profile the whole `conc-list`; each point is a full 397B load plus a 20-minute
  replay.
- **The profiled run is not a benchmark number.** A profiler window stalls the
  scheduler for seconds; its aggregate JSON is invalid as a submission and the
  skill relaxes `AIPERF_LIVE_FAILED_REQUEST_THRESHOLD` to 0.5 so the stall does
  not abort the replay. Report throughput from `/inferencemax-benchmark`, never
  from here.
- **Output** — everything under `$AGENT_RUNS_DIR/inferencemax/`, nothing in
  `$HOST_HOME` root.

## Steps

**1. Resolve the arm first** — same call as `/inferencemax-benchmark`, so the
user sees which recipe is about to run before a 397B load starts:

```bash
INFERENCEX_DIR=/home/yichiche/InferenceX \
  python3 ~/agent-box/skills/inferencemax-benchmark/resolve_arm.py \
  --model-prefix qwen3.5 --mode agent --tp 2
```

Report arm name, recipe path, CI image, TP/EP. If the checkout is behind
`origin/main`, say so and offer to pull.

**2. Dry run** when the model, TP or arm is new:

```bash
MODEL_PREFIX=qwen3.5 TP=2 DRY_RUN=1 \
  MODEL_PATH=/shared_nfs/models/Qwen/Qwen3.5-397B-A17B-MXFP4 \
  bash ~/agent-box/skills/inferencex-agent-profile/run_agent_profile.sh
```

**3. Capture.** Long — one server load + full replay *per concurrency point*.
Run it with `run_in_background`, never a foreground Bash call that can time out
mid-teardown:

```bash
MODEL_PREFIX=qwen3.5 TP=2 PROFILE_CONCS="4 64" NUM_STEPS=100 \
  MODEL_PATH=/shared_nfs/models/Qwen/Qwen3.5-397B-A17B-MXFP4 \
  bash ~/agent-box/skills/inferencex-agent-profile/run_agent_profile.sh
```

**4. Analyse.** Hand the TP-0 trace to the existing tooling — this skill
deliberately adds no analysis of its own:

- [`/parse-trace`](../parse-trace/SKILL.md) — module/kernel breakdown.
- [`/kernel-profile-triage`](../kernel-profile-triage/SKILL.md) — kernel time
  table, overlap and fusion candidates.
- [`/compare-kernels`](../compare-kernels/SKILL.md) — agent-mode vs fixed-seq,
  or MI355X vs B200, from two xlsx outputs.

**5. Report.** Relay: arm name, InferenceX commit, CI image vs the container
actually used, the sidecar's chosen start point (from `sidecar.log`), how many
steps were captured, and the kernel table. State explicitly that the run's
throughput numbers are profiling-distorted and not a benchmark result.

## Knobs

| var | default | meaning |
|---|---|---|
| `MODEL_PREFIX` | — | `qwen3.5`, `dsv4`, `glm5.2`, `kimik3`, `minimaxm3` … |
| `MODEL_PATH` | — | local weights dir (agent mode still needs `MODEL` = HF repo id; `run_infmax.sh` handles that) |
| `TP` / `SPEC` / `PRECISION` / `FRAMEWORK` / `HW` | as `/inferencemax-benchmark` | forwarded untouched to `resolve_arm.py` |
| `PROFILE_CONCS` | `4 64` | the global profiling anchors |
| `NUM_STEPS` | `100` | forward steps to capture; the profiler auto-stops |
| `ACTIVITIES` | `CPU,GPU` | also `MEM`, `RPD` (ROCm) |
| `PROFILE_BY_STAGE` | `0` | separate prefill/decode traces — see the gotcha below |
| `RECORD_SHAPES` / `WITH_STACK` | `0` | large; only for op-level attribution |
| `MERGE_PROFILES` | `0` | rank-0 merges all ranks into `merged-<id>.trace.json.gz` |
| `TRIGGER_REGEX` | unset | pin the aiperf phase banner in `recipe.log` instead of the drain heuristic |
| `COLLECT_TIMEOUT` | `1800` | how long to wait for every rank to finish gzipping |
| `AIPERF_LIVE_FAILED_REQUEST_THRESHOLD` | `0.5` | relaxed so the profiler stall does not abort the replay |
| `PORT` | `8888` | |
| `DRY_RUN` | `0` | resolve + print, launch nothing |
| `RUN_ROOT` | `~/agent-runs/inferencemax/<prefix>_agentprof_tp<N>_<ts>` | |

Everything not listed here — `DURATION`, `FULL`, `CONCURRENCIES`,
`CUDA_VISIBLE_DEVICES`, `KV_OFFLOADING` — is `run_infmax.sh`'s and is passed
through from the environment unchanged.

## Load-bearing gotchas

> - **All of [`/inferencemax-benchmark`'s gotchas still apply](../inferencemax-benchmark/SKILL.md#load-bearing-gotchas)** —
>   `/workspace` symlink, `MODEL` = HF repo id in agent mode, the `DURATION >= 900`
>   floor, aiperf's own uv venv on Python 3.11+, never `pkill`, never share GPUs.
> - **Do not start profiling during warmup.** Warmup requests are 1 token each.
>   A trace captured there shows a decode-only, tiny-batch engine and will send
>   you optimising a kernel mix the real replay never runs. This is the single
>   easiest way to get a confidently wrong answer out of this skill — check
>   `sidecar.log` for `after drain` before trusting the trace.
> - **`num_steps` is steps, not seconds, and the events live in RAM.** At conc64
>   a few hundred steps of a 397B MoE with CPU+GPU activities is multi-GB per
>   rank; push it far enough and the scheduler OOMs mid-replay. Start at 100.
> - **`PROFILE_BY_STAGE=1` is usually the wrong choice here.** On CUDA-graph
>   replayed decode (qwen35-mxfp4) the `*-DECODE` trace collapses to
>   `CudaGraphReplay` with no per-module detail. Agent mode is a *mixed*
>   prefill+decode stream anyway, and that mix is exactly what you want to see.
>   Capture combined.
> - **The profiler stall can trip aiperf's live error gate.** The wrapper relaxes
>   `AIPERF_LIVE_FAILED_REQUEST_THRESHOLD` to 0.5 for that reason. The flip side
>   is the obvious one: the resulting aggregate is not a submission and its
>   throughput must never be quoted.
> - **`output_dir` in the request beats the env var.** There is no need to export
>   `SGLANG_TORCH_PROFILER_DIR` before the recipe launches the server, which is
>   what makes a zero-touch sidecar possible. Note that `run_infmax.sh` runs the
>   recipe inside a private mount namespace with the conc dir bound at
>   `/workspace` — so the profile dir is passed as its *real* path, outside that
>   bind, and stays visible to both sides.
> - **One capture per server.** `num_steps` self-terminates the profiler; a
>   second `/start_profile` on the same server would overwrite state mid-replay.
>   Each concurrency point gets its own load, exactly as CI does.
> - **Traces are written per rank, asynchronously.** `<profile_id>-TP-<n>.trace.json.gz`
>   appear at different times; the sidecar waits for the file set *and* total
>   size to be quiet for 30 s before declaring success. Parsing a still-growing
>   gzip is a guaranteed crash in `/parse-trace`.

## Output

```
~/agent-runs/inferencemax/<prefix>_agentprof_tp<N>_<ts>/
  profiles/conc<N>/<profile_id>-TP-<r>.trace.json.gz   per-rank traces
  profiles/conc<N>/merged-<profile_id>.trace.json.gz   MERGE_PROFILES=1 only
  profiles/conc<N>/profile_manifest_conc<N>.json       request + trace paths
  profiles/conc<N>/sidecar.log                         when profiling started, and why
  conc<N>_run/run_infmax.log                           the reused runner
  conc<N>_run/conc<N>/recipe.log                       full recipe stdout/stderr
  conc<N>_run/conc<N>/aiperf_artifacts/                aiperf's own artifacts
  conc<N>_run/conc<N>/server.log
```

## Related

- [`/inferencemax-benchmark`](../inferencemax-benchmark/SKILL.md) — the same run
  without profiling; **this** is where the throughput number comes from.
- [`/generate-profile`](../generate-profile/SKILL.md) — our house profiling flow
  for a fixed-length workload against a locally launched server.
- [`/parse-trace`](../parse-trace/SKILL.md) · [`/kernel-profile-triage`](../kernel-profile-triage/SKILL.md)
  · [`/compare-kernels`](../compare-kernels/SKILL.md) — downstream analysis.
