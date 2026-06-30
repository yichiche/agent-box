# Launch & Benchmark Script Catalog

All paths relative to `$HOME` unless noted. **Run from neutral cwd** (`/tmp`) — see [[../gotchas/bench-cwd-shadow]].

## Qwen3.5

| Script | Purpose |
|---|---|
| `run_qwen3.5_mxfp4_perf.sh` | Server: Qwen3.5 397B MoE MXFP4, TP2, aiter, port 9000 |
| `run_qwen3.5_mxfp4_inferencemax_client.sh` | Client: IL70k/OL300 conc sweep |
| `run_qwen3.5_mxfp4_perf_agent.sh` | Agent-tuned perf (may need perf_sweep for CLI compat) |
| `run_qwen3.5_mxfp4_perf_atom.sh` | ATOM-integrated server variant |
| `run_qwen3.5_perf.sh` | Server: non-MXFP4 / BF16 variants |
| `run_qwen3.5_inferencemax_client.sh` | Client for BF16 path |
| `run_qwen3.5_perf_client.sh` | Alternate client |
| `run_qwen3-coder-next_spec.sh` | Qwen3 Coder Next + speculative |
| `run_qwen3-coder-next_spec_client.sh` | Matching client |
| `run_qwen3_vl.sh` / `run_qwen3_vl_client.sh` | Vision-language |

## DeepSeek

| Script | Purpose |
|---|---|
| `run_dsv4.sh` | DSv4 server (canonical) |
| `run_dsv4_inferencemax_client.sh` | DSv4 benchmark client |
| `run_dsv4_atom.sh` / `run_dsv4_atom_inferencemax_client.sh` | ATOM variants |
| `run_dsv4_0512.sh` … `run_dsv4_0528_main.sh` | Dated config snapshots — read header before use |
| `run_dsv4_client.sh` | Simpler client |
| `run_deepseekR1_mxfp4_spec.sh` / `_client.sh` | R1 MXFP4 speculative |

## Other

| Script | Purpose |
|---|---|
| `run_docker.sh` | Container launch helper |
| `run_fp8_accuracy_test.sh` | FP8 accuracy |
| `run_wan2.2_T2V.sh` / `run_wan_accuracy.sh` | Wan video gen |
| `run_det_test.sh` | Determinism test |

## Agent-box internals

| Script | Purpose |
|---|---|
| `agent-box/skills/perf-sweep/perf_sweep.sh` | Accuracy-gated conc sweep |
| `agent-box/debug/perf-regression/run_daily.sh` | Daily regression |
| `agent-box/memory/scripts/sync_from_claude_memory.sh` | Import Claude memory |

## Auto-refresh

Re-scan: `ls -1 $HOME/run_*.sh` and diff against this table during `/memory-consolidate`.
