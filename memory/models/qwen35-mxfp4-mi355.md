---
model: Qwen3.5-397B-A17B-MoE-MXFP4
hardware: MI355 (gfx95)
aliases: [qwen35, qwen3.5 mxfp4, qwen moe mxfp4]
---

# Qwen3.5 397B MoE MXFP4 on MI355

## Canonical scripts

| Role | Path |
|---|---|
| Server | `$HOME/run_qwen3.5_mxfp4_perf.sh` |
| Client (InferenceX-style sweep) | `$HOME/run_qwen3.5_mxfp4_inferencemax_client.sh` |
| Agent-tuned client | `$HOME/run_qwen3.5_mxfp4_perf_agent.sh` |
| ATOM variant | `$HOME/run_qwen3.5_mxfp4_perf_atom.sh` |

## Server flags (from `run_qwen3.5_mxfp4_perf.sh`)

```bash
AITER_FLYDSL_FORCE=1 \
SGLANG_USE_AITER_UNIFIED_ATTN=1 SGLANG_USE_AITER=1 \
python3 -m sglang.launch_server \
  --model-path /data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 --tp 2 \
  --attention-backend aiter --trust-remote-code \
  --chunked-prefill-size 32768 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 --mem-fraction-static 0.9 \
  --host 0.0.0.0 --port 9000 --disable-radix-cache \
  --enable-aiter-allreduce-fusion --max-running-requests 512 \
  --page-size 16
```

## Benchmark defaults (client)

- `PORT=9000`, `INPUT_LEN=70000`, `OUTPUT_LEN=300` (override with env)
- Concurrency sweep: `4 8 16 32 64 128 256`
- **CWD:** launch server AND client from `/tmp` or other neutral dir — see [[../gotchas/bench-cwd-shadow]]

## Accuracy

- Dataset: GSM8K (via `/perf-sweep` or `/validate`)
- Threshold: **0.92** on this box
- Verified: 0.945 @ TP2 IL8k/OL1k conc sweep (2026-06)

## Profiling

- Use `/generate-profile` or `python profile/trace_module_analyzer.py`
- Decode-heavy @ IL8192/OL1024: prefill optimizations have low Amdahl leverage (~2–3.5%)
- High-concurrency profiling can OOM host — profile at conc 4 for decode deep-dives

## Related memory

- [[../gotchas/bench-cwd-shadow]]
- [[../gotchas/container-bench-flags]]
- Imported: `~/.claude/.../qwen35-moe-decode-roundzero.md`, `qwen35-moe-gemm-e2e-amdahl.md`
