# Benchmark Workflow

## When to use what

| Goal | Tool |
|---|---|
| Full accuracy-gated conc sweep | `/perf-sweep` (skill) |
| Single before/after for a PR | `/benchmark` or `/validate` Step 1+5 |
| InferenceX table from CI | `/inferencex-table` |
| ATOM dashboard trend | `/atom-progress` |

## Standard flow (MI355)

1. `/gpu-status` → pick free CUDA indices → `export HIP_VISIBLE_DEVICES=...`
2. Resolve model → [[../models/INDEX]]
3. **cd /tmp** (or neutral cwd) — [[../gotchas/bench-cwd-shadow]]
4. Launch server from reference `run_*_perf.sh` (background, tee log)
5. Wait for healthy `/health` or log line `The server is fired up`
6. Run client script OR `/perf-sweep` with inherited server flags

## perf-sweep env (model-agnostic)

The workload is set by named preset — see [[workloads]]. The variable names are
`INPUT_LEN` / `OUTPUT_LEN` (NOT `IL` / `OL` — those are ignored, and omitting
`INPUT_LEN`/`OUTPUT_LEN` silently defaults to **8192 / 1024**, i.e. `canonical-8k`).

```bash
# canonical-8k (8192/1024) — the only workload valid for perf claims
MODEL=/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 \
GPUS="0,1" INPUT_LEN=8192 OUTPUT_LEN=1024 \
  bash ~/.claude/skills/perf-sweep/perf_sweep.sh

# diag-1k (1024/1024) — correctness / crash / scaling / profiler capture only
MODEL=/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 \
GPUS="0,1" INPUT_LEN=1024 OUTPUT_LEN=1024 \
  bash ~/.claude/skills/perf-sweep/perf_sweep.sh
```

## Output layout

Results often land under `$HOME/qwen3.5-mxfp4/.../conc{N}/...` with:
- `summary.csv` — throughput / latency per concurrency
- `analysis_prefill.xlsx`, `analysis_decode.xlsx` — from trace parser
- `module_tree_*.html` — layer breakdown

## Skills

- `skills/perf-sweep/SKILL.md`
- `skills/benchmark/SKILL.md`
- `skills/inferencex-table/SKILL.md`
