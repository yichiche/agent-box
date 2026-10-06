---
name: inferencex-benchmark
description: "Run the current InferenceMax (SemiAnalysisAI/InferenceX) benchmark standard locally, end to end, for a model you name — fixed-length (ISL 8192 / OSL 1024) or agent mode (AgentX trace replay). Resolves the arm from configs/amd-master.yaml, binds it to one srt-slurm recipe variant with CI's own selection and golden-acceptance code, and launches the server and the recipe's own client the way CI does, so every server and client flag is aligned by construction rather than by copying. Use when asked to benchmark a model to InferenceMax standard, reproduce a dashboard number, or run agentic/fixed-seq benchmarks that must be comparable to InferenceX."
category: measure
---

# /inferencex-benchmark — run the InferenceMax standard locally

```
/inferencex-benchmark <model-prefix> <fixed|agent> [tp=N] [conc=...] [full] [dry-run]
```

`/inferencex-benchmark qwen3.5 agent tp=2 full` · `/inferencex-benchmark dsv4 fixed`

## The one idea

**This skill executes the upstream recipe. It never restates a flag.**

Every single-node arm in `inferencex-e2e/configs/amd-master.yaml` names an
srt-slurm recipe (`srt-recipe:` in its search-space row). In CI,
`runners/slurm_utils.sh` binds the matrix point to exactly one variant of that
YAML (`infx.srt_slurm.single_node.select_recipe`), `apply_srt_recipe` adds the
golden MTP acceptance for AgentX (`infx.srt_slurm.synthetic_acceptance`), and
srtctl renders `sglang.launch_server` and runs the recipe's `benchmark.command`
against it.

`srt_point.py` imports CI's selection, validation and override code from the
checkout and runs it as-is; it mirrors only srtctl's variant expansion and CLI
rendering, because srtctl's own dependencies (marshmallow, pyarrow, mcp, …) are
not in the serving containers. `run_infmax.sh` launches that server, then the
recipe's client (`srt_agentic.sh` / `srt_fixed_sequence.sh`) with the env CI
layers: job env → `benchmarks/runtime_settings.sh` → recipe `benchmark.env` →
runtime bindings.

So alignment is maintained by `git -C $INFERENCEX_DIR pull`, not by editing
anything here. Recipes move: between 2026-08-24 and 2026-09-02 the qwen3.5
MI355X recipes changed `chunked-prefill-size`, `cuda-graph-max-bs`, the
quick-reduce quantization and the agentic TP2 EP. Any hand-copied script is
already wrong.

## Honor the global conventions in `~/agent-box/CLAUDE.md`

- **num_prompts** — do not set it. `CONC * 10` lives inside the fixed client.
- **Shape** — fixed mode is InferenceX's `canonical-8k` (ISL 8192 / OSL 1024),
  which is also the only shape a perf claim may be made on.
- **Concurrency** — comes from the arm's `conc-list` / `conc-start..conc-end`,
  not from our house sweep list. Override with `conc=` only to shorten a smoke run.
- **Output** — everything lands under `$AGENT_RUNS_DIR/inferencemax/`; nothing is
  written to `$HOST_HOME` root.
- **Reference table** — fixed mode renders side by side with
  `memory/models/qwen35-mxfp4-mi355-reference.csv` when the shape matches.

## Steps

Run everything inside a serving container that mounts `/home/yichiche`.

**1. Resolve and show the arm before running anything.**

```bash
INFERENCEX_DIR=/home/yichiche/InferenceX \
  python3 ~/agent-box/skills/inferencex-benchmark/resolve_arm.py \
  --model-prefix qwen3.5 --mode agent --tp 2
```

Report the arm name, srt-recipe, CI image, TP/EP, and the concurrency list back
to the user. If the checkout is behind `origin/main` (the runner warns), say so
and offer to pull — running a stale recipe defeats the purpose of the skill.

**2. Dry run first when anything about the request is new** (new model, new TP,
first use after a pull):

```bash
MODEL_PREFIX=qwen3.5 MODE=agent TP=2 FULL=1 DRY_RUN=1 \
  MODEL_PATH=/data/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2 \
  bash ~/agent-box/skills/inferencex-benchmark/run_infmax.sh
```

Per concurrency it prints the selected recipe variant, the overrides CI would
add, the exact `launch_server.sh`, the recipe's benchmark env, the client
command and the job env, and launches nothing. For AgentX with MTP the
`overrides` line must show the three `SGLANG_SIMULATE_ACC_*` values.

**3. Run.** Drop `DRY_RUN`. One server load per concurrency, exactly as upstream.

```bash
MODEL_PREFIX=qwen3.5 MODE=fixed TP=2 \
  MODEL_PATH=/data/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2 \
  bash ~/agent-box/skills/inferencex-benchmark/run_infmax.sh
```

Long runs (agent mode is a 20–60 min replay plus a 397B load *per concurrency
point*) belong in the background with `run_in_background`, not a foreground
Bash call that can time out mid-teardown.

**4. Report.** `run_infmax.sh` writes `summary.md` in the run dir via
`render_table.py`. Relay the table, plus: the arm name, the InferenceX commit,
the variant and overrides per point (`conc<N>/srt/point.json`), the CI image vs
the container actually used, whether it was `FULL=1`, and any concurrency whose
`completed != conc*10` (fixed) or with a non-empty `warnings` (agent). Compare
against published numbers with [`/inferencex-table`](../inferencex-table/SKILL.md).

## Knobs

| var | default | meaning |
|---|---|---|
| `MODEL_PREFIX` | — | `qwen3.5`, `dsv4`, `glm5.2`, `kimik3`, `minimaxm3` … |
| `MODE` | — | `fixed` or `agent` |
| `MODEL_PATH` | — | local weights dir; stands in for the HF cache CI mounts |
| `INFERENCEX_DIR` | `/home/yichiche/InferenceX` | repo root; `inferencex-e2e/` is found under it |
| `TP` | first search-space row | 2 or 4 |
| `SPEC` | `mtp` | `none` picks the non-MTP arm |
| `KV_OFFLOADING` | `none` | agent mode: `dram` picks the HiCache row |
| `PRECISION` / `FRAMEWORK` / `HW` | `fp4` / `sglang` / `mi355x` | arm selectors |
| `CONCURRENCIES` | the arm's list | override to shorten a smoke run |
| `FULL` | `0` | `1` = CI's 3600 s replay with 10 warmups per lane. **Required to compare with the dashboard** |
| `DURATION` | 1200 (`FULL=1` → 3600) | agent mode only |
| `SYNTHETIC_ACCEPTANCE` | `1` | `0` = real draft acceptance; not comparable to the dashboard |
| `THINKING_MODE` | `thinking_on` | golden-curve column; CI hard-codes `thinking_on` |
| `VISIBLE_DEVICES_ENV` | `ROCR_VISIBLE_DEVICES` | how the server is pinned (`runners/srt-slurm/mi355x-amds.yaml`) |
| `CUDA_VISIBLE_DEVICES` | auto-picked free GPUs | the indices to use; handed to the server via `VISIBLE_DEVICES_ENV` |
| `REQUIRE_POWER` | `0` | `1` fails the point when power collection fails |
| `PORT` | `8888` | |
| `SERVER_READY_TIMEOUT` | `2400` | seconds to wait for `/health` |
| `DRY_RUN` | `0` | print the per-point plan, launch nothing |
| `RUN_DIR` | `~/agent-runs/inferencemax/<prefix>_<mode>_tp<N>_<ts>` | |

## Load-bearing gotchas

> - **AgentX MTP acceptance is synthetic in CI.** `apply_srt_recipe` runs
>   `synthetic_acceptance.build_overrides`, which sets
>   `SGLANG_SIMULATE_ACC_LEN` from `infx/golden_al_distribution/<model>_mtp.yaml`
>   (qwen3.5, thinking_on, 3 draft tokens → 3.39),
>   `SGLANG_SIMULATE_ACC_METHOD=match-expected` and
>   `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`. Launching the recipe
>   without it put qwen3.5 TP2 conc4 at 4.32 ms median TPOT against CI's 3.18 ms.
>   Two things change: real acceptance on the trace is about 3.05, and with
>   simulation off SGLang on HIP turns on `speculative_use_rejection_sampling`;
>   one MTP step at batch 1 measured 11.0 ms without the overrides and 9.3 ms with
>   them. Fixed-seq-len runs never get them.
> - **FAST mode is not the dashboard.** The default 1200 s replay with one warmup
>   per lane samples earlier, shorter trajectories (mean input 68K vs 87K tokens at
>   qwen3.5 conc4), so its throughput is not comparable. Use `FULL=1` before
>   comparing any agent number with InferenceX.
> - **Fixed mode: the client's `MODEL` is the local path; agent mode: the HF repo id.**
>   `srt_point.py` serves the fixed model under the local dir name so bench_serving
>   tokenizes offline (`HF_HUB_OFFLINE=1`). aiperf's `--model`/`--tokenizer` use the
>   HF id, so the first agent run needs HF reachable for the tokenizer; weights
>   always come from `MODEL_PATH`.
> - **KV offload env is a pair.** `KV_OFFLOAD_BACKEND` is the backend *name* and
>   `KV_OFFLOAD_BACKEND_METADATA` its JSON, both empty when KV stays on GPU. Any
>   other combination makes the agentic aggregate fail validation.
> - **`DURATION < 900` makes aiperf add `--unsafe-override`** and the result is no
>   longer a valid submission. The runner refuses. 1200 is the floor.
> - **Never add a `pkill` of your own.** The runner stops its own server's process
>   group (SIGTERM, 60 s grace, then SIGKILL), like upstream's
>   `stop_background_process_tree`. A broad `pkill -9 -f sglang` kills other
>   tenants and orphans TP workers into an unreclaimable ~200 GB/shard VRAM leak
>   that needs a host-side kill.
> - **Don't share GPUs.** The runner takes only fully free cards and aborts
>   otherwise. A model that loads into a contended card either OOMs mid-load or
>   produces a number that means nothing.
> - **aiperf runs in its own uv venv on Python 3.11+**, not the container's
>   `python3` (3.10 in sglang-rocm images). The first agent run needs `astral.sh`,
>   PyPI and HF reachable, and downloads the ~393-trace corpus.
> - **The CI image is not necessarily this container.** The arm pins e.g.
>   `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260927`; flags are aligned, the
>   software under them is only aligned if the image digest matches. Always report
>   both.

## Output

```
~/agent-runs/inferencemax/<prefix>_<mode>_tp<N>_<ts>/
  summary.md                     rendered table
  conc<N>/srt/point.json         selected variant, CI overrides applied, argv, server env
  conc<N>/srt/launch_server.sh   the exact server command
  conc<N>/server.log
  conc<N>/recipe.log             client stdout/stderr (the agent-profile sidecar tails it)
  conc<N>/<RESULT_FILENAME>.json result (fixed: bench_serving; agent: aggregate)
  conc<N>/agg_<...>.json         fixed only, from infx.results.fixed_sequence
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
- [`/inferencex-agent-profile`](../inferencex-agent-profile/SKILL.md) — the same
  run with a torch-profiler sidecar.
- [`/perf-sweep`](../perf-sweep/SKILL.md) — our own accuracy-gated sweep, when the
  question is "did my change help", not "does this match InferenceMax".
