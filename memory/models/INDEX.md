# Model Registry

Canonical mapping: **model × hardware → launch config + benchmark + accuracy**.

Agents: when the user names a model vaguely ("qwen mxfp4", "dsv4", "跑 perf"), resolve via this table before guessing flags.

| Alias | Model path | HW | TP | Server script | Client script | Accuracy | Threshold | Notes |
|---|---|---|---|---|---|---|---|---|
| qwen35-mxfp4 | `/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4` | MI355 | 2 | `~/run_qwen3.5_mxfp4_perf.sh` | `~/run_qwen3.5_mxfp4_inferencemax_client.sh` | GSM8K | 0.92¹ | aiter unified attn, page-size 16; **thinking model — see card** |
| qwen35-bf16 | `/data/amd/Qwen3.5-397B-A17B-*` (non-MXFP4) | MI355 | 2–8 | `~/run_qwen3.5_perf.sh` | `~/run_qwen3.5_inferencemax_client.sh` | GSM8K | 0.92 | see script for kv-cache dtype |
| dsv4 | (see script) | MI355 | varies | `~/run_dsv4.sh` | `~/run_dsv4_inferencemax_client.sh` | GSM8K | 0.88 | multiple dated variants: `run_dsv4_0512.sh` etc. |
| dsr1-mxfp4-spec | (see script) | MI355 | varies | `~/run_deepseekR1_mxfp4_spec.sh` | `~/run_deepseekR1_mxfp4_spec_client.sh` | GSM8K | 0.88 | speculative decode |
| qwen3-coder-next | config in `agent-box/configs/qwen3_coder_next_config.json` | MI355 | varies | `~/run_qwen3-coder-next_spec.sh` | `~/run_qwen3-coder-next_spec_client.sh` | task-specific | — | MoE 512 experts |

## Per-model detail cards

- [[qwen35-mxfp4-mi355]] — full env flags, profiling notes, known ceilings

¹ **0.92 is the bar for an accuracy-VALID config under the mandatory thinking-model
eval protocol** (`--enable-thinking --max-new-tokens 8192`), NOT a number to relax.
Qwen3.5 is a thinking model; a naive bench gives misleadingly low scores. Diagnose a
low score by invalid rate — high invalid = eval artifact, low invalid + low acc =
known-bad config (fp4 shared expert, NEVER SHIP). Full two-tier rule: [[qwen35-mxfp4-mi355]]
+ [[../workflows/accuracy]].

## Resolution rules

1. **Hardware default:** MI355 / gfx95 unless user says otherwise → use aiter backends, `HIP_VISIBLE_DEVICES` (not empty!).
2. **"perf" / "benchmark"** → server script + client script + `workflows/benchmark.md`; prefer `/perf-sweep` for accuracy-gated sweep.
3. **"validate" / "測一下改動"** → `/validate` with server+client from this table.
4. **"profile"** → keep server running; use `/generate-profile` or `workflows/profiling.md`.
5. **Ambiguous model** → read the matching `run_*.sh` header comments; never invent `--tp` or env flags.
