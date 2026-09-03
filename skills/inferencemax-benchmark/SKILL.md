---
name: inferencemax-benchmark
description: "Run the current InferenceMax (SemiAnalysisAI/InferenceX) benchmark standard locally, end to end, for a model you name — fixed-length (ISL 8192 / OSL 1024) or agent mode (AgentX trace replay). Resolves the arm from configs/amd-master.yaml, exports exactly the env CI exports, and executes the upstream recipe script itself, so every server and client flag is aligned by construction rather than by copying. Use when asked to benchmark a model to InferenceMax standard, reproduce a dashboard number, or run agentic/fixed-seq benchmarks that must be comparable to InferenceX."
category: measure
---

# /inferencemax-benchmark — run the InferenceMax standard locally

```
/inferencemax-benchmark <model-prefix> <fixed|agent> [tp=N] [conc=...] [full] [dry-run]
```

`/inferencemax-benchmark qwen3.5 agent tp=2` · `/inferencemax-benchmark dsv4 fixed`

## The one idea

**This skill executes the upstream recipe. It never restates a flag.**

The InferenceX checkout at `$INFERENCEX_DIR` already contains the real recipes
under `benchmarks/single_node/`. Everything CI does *around* them —
`runners/launch_mi355x-amds.sh` (mounts + env + `bash <recipe>`) and
`.github/workflows/benchmark-tmpl.yml` (`RESULT_FILENAME`, `RESULT_DIR`,
thresholds) — is what `run_infmax.sh` reimplements. The server command line, the
client command line, the readiness wait and the teardown all stay upstream's.

So alignment is maintained by `git -C $INFERENCEX_DIR pull`, not by editing
anything here. This is not theoretical: between 2026-08-24 and 2026-09-02 the
qwen3.5 MI355X recipes changed `chunked-prefill-size` 32768 → 16384,
`cuda-graph-max-bs` from `CONC` (cap 64) to `2*CONC` (cap 128), dropped
`--enable-aiter-allreduce-fusion`, added `--kv-cache-dtype fp8_e4m3` to the
fixed path and `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` to both — and the agentic
TP2 arm moved from **EP2 to EP1**. Any hand-copied script is already wrong.

## Honor the global conventions in `~/agent-box/CLAUDE.md`

- **num_prompts** — do not set it. `CONC * 10` lives inside the fixed recipe.
- **Shape** — fixed mode is InferenceX's `canonical-8k` (ISL 8192 / OSL 1024),
  which is also the only shape a perf claim may be made on.
- **Concurrency** — comes from the arm's `conc-list` / `conc-start..conc-end`,
  not from our house sweep list. Override with `conc=` only to shorten a smoke run.
- **Output** — everything lands under `$AGENT_RUNS_DIR/inferencemax/`; nothing is
  written to `$HOST_HOME` root.
- **Reference table** — fixed mode renders side by side with
  `memory/models/qwen35-mxfp4-mi355-reference.csv` when the shape matches.

## Steps

**1. Resolve and show the arm before running anything.**

```bash
INFERENCEX_DIR=/home/yichiche/InferenceX \
  python3 ~/agent-box/skills/inferencemax-benchmark/resolve_arm.py \
  --model-prefix qwen3.5 --mode agent --tp 2
```

Report the arm name, recipe path, CI image, TP/EP, and the concurrency list back
to the user. If the checkout is behind `origin/main`, say so and offer to pull —
running a stale recipe defeats the purpose of the skill.

**2. Dry run first when anything about the request is new** (new model, new TP,
first use after a pull):

```bash
MODEL_PREFIX=qwen3.5 MODE=agent TP=2 DRY_RUN=1 \
  MODEL_PATH=/shared_nfs/models/Qwen/Qwen3.5-397B-A17B-MXFP4 \
  bash ~/agent-box/skills/inferencemax-benchmark/run_infmax.sh
```

This prints the full resolved env block and the exact `bash <recipe>` line per
concurrency, and launches nothing. Read it against
`runners/launch_mi355x-amds.sh` + `benchmark-tmpl.yml` — every var CI forwards
must be present.

**3. Run.** Drop `DRY_RUN`. One server load per concurrency, exactly as upstream;
the recipe's own EXIT trap tears it down.

```bash
MODEL_PREFIX=qwen3.5 MODE=fixed TP=2 \
  MODEL_PATH=/shared_nfs/models/Qwen/Qwen3.5-397B-A17B-MXFP4 \
  bash ~/agent-box/skills/inferencemax-benchmark/run_infmax.sh
```

Long runs (agent mode is ~20 min replay + a 397B load *per concurrency point*)
belong in the background with `run_in_background`, not a foreground Bash call
that can time out mid-teardown.

**4. Report.** `run_infmax.sh` writes `summary.md` in the run dir via
`render_table.py`. Relay the table, plus: the arm name, the InferenceX commit,
the CI image vs the container actually used, and any concurrency whose
`completed != conc*10` (fixed) or with a non-empty `warnings` (agent).

## Knobs

| var | default | meaning |
|---|---|---|
| `MODEL_PREFIX` | — | `qwen3.5`, `dsv4`, `glm5.2`, `kimik3`, `minimaxm3` … |
| `MODE` | — | `fixed` or `agent` |
| `MODEL_PATH` | — | local weights dir; keeps the recipe off the network |
| `TP` | first search-space row | 2 or 4 |
| `SPEC` | `mtp` | `none` picks the non-MTP arm |
| `PRECISION` / `FRAMEWORK` / `HW` | `fp4` / `sglang` / `mi355x` | arm selectors |
| `CONCURRENCIES` | the arm's list | override to shorten a smoke run |
| `DURATION` | 1200 (`FULL=1` → 3600) | agent mode only |
| `FULL` | `0` | `1` = full-fidelity 3600 s replay, no `AIPERF_EXPERIMENTAL_FAST` |
| `PORT` | `8888` | benchmark_lib's default |
| `DRY_RUN` | `0` | print env + command, launch nothing |
| `RUN_DIR` | `~/agent-runs/inferencemax/<prefix>_<mode>_tp<N>_<ts>` | |
| `CUDA_VISIBLE_DEVICES` | auto-picked | set to pin cards yourself |

## Load-bearing gotchas

> - **`/workspace` must be a symlink to the checkout.** CI mounts it there
>   (`-v $GITHUB_WORKSPACE:/workspace/`) and the fixed recipe hardcodes
>   `/workspace/server.log` and `--result-dir /workspace/`. The runner creates the
>   symlink and refuses to touch `/workspace` if it already exists as a real dir.
> - **Fixed mode: `MODEL` is a local path; agent mode: `MODEL` is the HF repo id.**
>   Agent's `MODEL` feeds `--tokenizer-path` and aiperf's `--tokenizer`; a bare
>   `Qwen3.5-397B-A17B-MXFP4` is not a repo and 404s. Agent weights come from
>   `MODEL_PATH`, and the recipe only skips the download when that dir is
>   non-empty — point it wrong and it pulls the full repo.
> - **The fixed MTP recipe calls `hf download "$MODEL"` unguarded** (the non-MTP one
>   guards with `if [[ "$MODEL" != /* ]]`). The runner sets `HF_HUB_OFFLINE=1` so
>   that call fails fast instead of reaching for the network; the recipe has no
>   `set -e`, so it continues.
> - **`DURATION < 900` makes aiperf add `--unsafe-override`** and the result is no
>   longer a valid submission. The runner refuses. 1200 is the floor.
> - **Never add a `pkill` of your own.** Upstream tears down with
>   `stop_background_process_tree` (SIGTERM the tree, then SIGKILL survivors). A
>   broad `pkill -9 -f sglang` kills other tenants and orphans TP workers into an
>   unreclaimable ~200 GB/shard VRAM leak that needs a host-side kill.
> - **Don't share GPUs.** The runner takes only fully free cards and aborts
>   otherwise. A model that loads into a contended card either OOMs mid-load or
>   produces a number that means nothing.
> - **aiperf runs in its own uv venv on Python 3.11+**, not the container's
>   `python3` (3.10 in sglang-rocm images). The first agent run needs `astral.sh`,
>   PyPI and HF reachable, and downloads the ~393-trace corpus.
> - **Agent mode had never run to completion on this box** before this skill.
>   Treat the first run of a new model as bring-up, not as a measurement.
> - **The CI image is not this container.** The arm pins e.g.
>   `lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260829`; you are running the
>   recipe against whatever sglang/aiter is installed here. Always report both —
>   flags are aligned, the software under them is not.

## Output

```
~/agent-runs/inferencemax/<prefix>_<mode>_tp<N>_<ts>/
  summary.md                     rendered table
  conc<N>/recipe.log             full recipe stdout/stderr
  conc<N>/<RESULT_FILENAME>.json result (fixed: bench_serving; agent: aggregate)
  conc<N>/agg_<...>.json         fixed only, from utils/process_result.py
  conc<N>/server.log             fixed only (agent writes it into RESULT_DIR itself)
  conc<N>/aiperf_artifacts/      agent only
  conc<N>/gpu_metrics*.csv
```

Fixed table: `ISL / OSL / conc / completed / Median E2E / total tok/s / tok/s/gpu
/ Median TTFT / Median TPOT`, with per-cell delta against the reference CSV.
Agent table: `conc / requests ok / TPOT mean / interactivity / E2E p90 / TTFT p50
/ total tok/s / tok/s/gpu / cache hit`.

## Related

- [`/inferencex-table`](../inferencex-table/SKILL.md) — fetch the *published*
  numbers to compare yours against.
- [`/perf-sweep`](../perf-sweep/SKILL.md) — our own accuracy-gated sweep, when the
  question is "did my change help", not "does this match InferenceMax".
