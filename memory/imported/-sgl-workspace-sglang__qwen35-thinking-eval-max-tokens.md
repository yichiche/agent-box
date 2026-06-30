---
name: qwen35-thinking-eval-max-tokens
description: "Qwen3.5-397B-MXFP4 gsm8k accuracy collapses to ~0.55 if max_tokens too small — it's a thinking model needing ~1000-1350 CoT tokens"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7ab85c4-8dae-4065-b42f-b2e2aaed7bf1
---

Qwen3.5-397B-A17B-MXFP4 (e.g. checkpoint `-d1db9a1` at `/raid/models/`) is a **thinking/reasoning model** (emits `<think>...</think>` and chat-style `Assistant:` turns). Evaluating it on gsm8k with `sglang.test.run_eval --eval-name gsm8k` (or the deprecated `few_shot_gsm8k`) at the default `--max-tokens 2048` truncates the chain-of-thought before the final answer, giving a misleading ~0.555 score with high "invalid" rate.

**Why:** Correct answers need ~1000-1350 thinking tokens (some hard problems run to 6000+); at 2048 the model never reaches the final number, and the last-number extractor grabs an intermediate value.

**How to apply:** Use `--max-tokens 8192` (or more). At 8192 the model scores ~0.98 on gsm8k. When an MXFP4/quant accuracy "regression" shows up as ~0.55 with lots of truncation, suspect the token budget before suspecting the weights. Confirmed 2026-06-11: max_tokens=2048 → 0.555, max_tokens=8192 → 0.985 (same server). Related: [[qwen35-shared-expert-fusion-online-requant]].
