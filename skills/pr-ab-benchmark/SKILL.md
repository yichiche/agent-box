---
name: pr-ab-benchmark
description: "Before/after A/B benchmark of an aiter (or sglang) PR on an SGLang server, with per-kernel profiling confirmation. Applies one or more baseline PRs, benchmarks e2e throughput across concurrencies for one or more IL/OL workloads, then applies the PR-under-test (rebuilding any csrc/JIT module it touches), re-benchmarks, and profiles a target kernel at chosen concurrencies via the client --profile path. Emits before/after throughput tables + a target-kernel µs/launch comparison. Use when asked to 'benchmark before/after applying PR X', 'confirm kernel Y improves', or A/B a perf PR end to end."
category: measure
---

# pr-ab-benchmark — A/B a perf PR (e2e throughput + kernel profile)

Controlled before/after for ONE PR-under-test, layered on a baseline PR stack. Two server
loads total: **baseline → sweep + profile**, then **apply PR + rebuild → sweep + profile**.
The *delta* is the deliverable, not the absolute numbers (absolute numbers move with the
pinned aiter version / shared-node variance).

Model used in the canonical run: Qwen3.5-397B-A17B-MoE-MXFP4, MI355X gfx950, TP2.
Server ref: `/home/yichiche/run_qwen3.5_mxfp4_perf.sh`. Helpers live next to this file.

## When to use
- "apply PR A + B as baseline, then benchmark before/after PR C, IL1024/OL1024 & IL8192/OL1024"
- "confirm the topk/moe_sorting kernel improves at conc 4 / 64 before vs after"
For a single load accuracy→sweep use `/perf-sweep`; just a profile use `/generate-profile`;
just trace parsing use `/parse-trace`; kernel-only A/B use `/compare-kernels`.

## Phase 0 — apply baseline PRs (leave unrelated dirty files untouched)
- aiter: `gh pr diff <N> --repo ROCm/aiter | git apply --3way`
- sglang: `gh pr diff <N> --repo sgl-project/sglang | git apply --3way` (editable install → no rebuild for .py)
- Verify each landed (grep the new symbol). A stacked PR's diff (e.g. base not yet merged) brings
  its parent's hunks along — that is usually what you want for "apply as baseline".

### GOTCHA: tuned-GEMM CSV vs pinned aiter (version skew)
A tuned config CSV (e.g. aiter #4017) may reference an opus kernel id (`libtype=opus,solidx=NNNN`)
that does NOT exist in an older pinned aiter build → **hard abort on prefill**:
`opus_gemm_arch_gfx950.cuh:NNN Kernel id NNNN not found in a16w16 bf16 tune lookup table` →
`Fatal Python error: Aborted` (scheduler exit -6). It only fires once prefill hits the tuned
shape (e.g. M≈2048 N=4096), so a conc4 smoke test passes and hides it.
Fix: drop the offending rows `grep -v "opus,NNNN," <csv> > t && mv t <csv>`; confirm the build's
header has 0 hits for that id. See memory `aiter-4017-opus6401-version-skew`.

## Phase 1 — baseline server
- Clear JIT locks: `find /sgl-workspace/aiter/aiter/jit/build \( -name lock -o -name "lock_*" \) -delete`
- Clear stale merged config cache: `rm -rf /tmp/aiter_configs`
- Launch from a **neutral cwd (/tmp)** (else `/home/yichiche/{aiter,sglang}` shadow the editable
  installs — see memory `bench-cwd-shadow-gotcha`). Set `HIP_VISIBLE_DEVICES`, `PORT`, and
  **`SGLANG_TORCH_PROFILER_DIR`** (required for client `--profile`).
  ```bash
  cd /tmp && SGLANG_TORCH_PROFILER_DIR=<dir> HIP_VISIBLE_DEVICES=0,1 PORT=9000 \
    nohup bash /home/yichiche/run_qwen3.5_mxfp4_perf.sh > server_before.log 2>&1 &
  ```
- Confirm: `[aiter] import ... under /sgl-workspace/aiter/...`, no `Found N duplicate`, `fired up and ready`.

## Phase 2 — e2e sweep (both ILs, all conc)
`PHASE=before bash run_bench.sh` — drives `python -m sglang.bench_serving` directly
(the ref client's `./bench_serving/benchmark_serving.py` is a broken symlink). Flags that matter:
`--output-file` (APPENDS JSONL — take the LAST line per file), ignore-eos is default-on
(use `--disable-ignore-eos` to turn off), NO `--save-result/--result-dir/--percentile-metrics`.
Edit `CONCS`/`WORKLOADS` at the top of `run_bench.sh`.

## Phase 3 — profile target kernel at chosen conc
`PHASE=before bash run_profile.sh` — small bench at each conc with
`--profile --profile-start-step 60 --profile-steps 40 --profile-prefix <phase>_conc<c>`.
Traces land in `/tmp/<ts>/<prefix>-<id>-TP-{0,1}.trace.json.gz` (the client overrides the dir;
ignore SGLANG_TORCH_PROFILER_DIR for output location, it just enables the profiler).
Copy the finalized trace (`gzip -t` to confirm not truncated — the bench keeps running after the
profiler stops, so the file may still be flushing) and parse:
`python3 sort_kernel_time.py <trace>` → per-kernel total / launches / **avg µs/launch** (the robust
cross-run metric, independent of how many steps were captured). Adjust the `classify()` names in
`sort_kernel_time.py` for whatever kernel you're confirming.
For the Excel module breakdown the user may want:
`python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py <trace> -o <out.xlsx>`.

## Phase 4 — apply PR-under-test + rebuild
- Apply the diff. If it edits a `csrc/**.h/.cu`, the JIT must recompile the owning module:
  find it via `grep -rl '#include.*<header>' csrc/` → module name; then
  `rm -f aiter/jit/module_<X>.so && rm -rf aiter/jit/build/module_<X>` and clear locks.
  (For #3986 the header is `moe_sorting_opus.h` → module `module_moe_sorting_opus`; rebuilds ~9s at
  next server import.)
- Kill the baseline server (`pkill -9 -f sglang.launch_server`) and **wait for VRAM reclaim**
  (KFD takes ~60-90s; poll `rocm-smi --showmeminfo vram` until <~15GB) before relaunch — else OOM.
  See memory `aiter-jit-deadlock-gpu-reclaim`.

## Phase 5 — after server + repeat
Relaunch with `SGLANG_TORCH_PROFILER_DIR=<after_dir>`; confirm the modified module shows
`start build [module_<X>] ... finish build`. `PHASE=after bash run_bench.sh` + `run_profile.sh`.

## Phase 6 — compare
- e2e: `python3 compare_e2e.py` → before/after total-tok/s + ttft/tpot/e2el per (IL,conc).
  Verify all runs `completed == num_prompts` before trusting a row.
- kernel: `sort_kernel_time.py` before vs after → avg µs/launch. A perf dispatch PR should move the
  target kernel at the regime it targets (e.g. low conc) and be flat where it doesn't (high conc);
  that flat-where-expected check is the control that rules out global drift.
- Report tok/s/gpu = total ÷ TP when asked. Flag single-run dips at conc the PR can't touch as variance.

## Files
- `run_bench.sh` — e2e sweep wrapper (edit CONCS/WORKLOADS/MODEL/PORT at top)
- `run_profile.sh` — per-conc `--profile` capture (edit conc list / IL/OL at top)
- `sort_kernel_time.py` — sum GPU-kernel device time by name → total/launches/avg µs (edit classify())
- `compare_e2e.py` — before/after throughput+latency tables from results/*.json (JSONL last-line)
