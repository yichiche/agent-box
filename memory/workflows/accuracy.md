# Accuracy Testing

## Default datasets

| Model family | Dataset | Default threshold |
|---|---|---|
| DeepSeek V4 / R1 | GSM8K | 0.88 |
| Qwen3.5 MXFP4 | GSM8K | 0.92 |
| Thinking models | GSM8K + max_tokens tuning | see sglang memory `qwen35-thinking-eval-max-tokens` |

## How to run

- **Inside `/perf-sweep`:** GSM8K gate runs before any benchmark conc
- **Inside `/validate`:** explicit accuracy step between baseline and after
- **Standalone:** `ATOM/atom/mesh/scripts/run_gsm8k.sh` or `python -m sglang.test.accuracy_*` (check active sglang install)

## Thinking / reasoning models

- May need higher `max_tokens` for GSM8K chain-of-thought
- Client must not truncate with aggressive `OUTPUT_LEN` when testing accuracy

## Fail actions

- Accuracy fail → **do not** publish benchmark numbers as valid
- `/validate` returns `status: fail`; `/kernel-fusion-pipeline` auto-decides retry/skip
