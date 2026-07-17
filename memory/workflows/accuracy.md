# Accuracy Testing

## Default thresholds (accuracy-VALID config)

| Model family | Dataset | Ship threshold |
|---|---|---|
| DeepSeek V4 / R1 | GSM8K | 0.88 |
| Qwen3.5 MXFP4 | GSM8K | **0.92** |

The threshold is the bar for an **accuracy-valid configuration evaluated with the
correct protocol**. It is NOT a bar you lower when a number comes in low — a low
score means either the eval is wrong (fix it) or the config is known-bad (don't
ship it). See the two-tier rule below.

## Qwen3.5 is a thinking model — eval protocol is mandatory

Qwen3.5-397B MXFP4 emits `<think>…</think>` chain-of-thought. A naive
`bench_sglang.py` run (bare 5-shot completion, default max_tokens) gives a
**misleadingly low** score even on a healthy model. Always evaluate with:

```bash
cd "$SGLANG_ROOT"
python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 200 --parallel 100 --num-shots 5 \
  --enable-thinking --tokenizer-path /data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 \
  --max-new-tokens 8192 --port <PORT>
```

- Correct answers need ~1000–1350 CoT tokens (hard ones 6000+). At `--max-new-tokens
  2048` the CoT truncates → ~0.555 with high invalid. At 8192 a healthy model
  scores ~0.98. ([[qwen35-thinking-eval-max-tokens]])

## The two-tier rule — how to read a low score

**Look at the invalid rate first; it discriminates the two failure modes.**

| Symptom | Meaning | Action |
|---|---|---|
| **invalid HIGH (>~0.05)** | Eval artifact — token budget too small, thinking not enabled, or bare-prompt pattern-continuation | **Fix the eval** (protocol above / instruction prompt), re-run. NOT a weights problem. |
| **invalid LOW (<~0.02) but acc LOW** | Genuine degradation — wrong reasoning | **Config is broken. Do not ship.** Suspect the known-bad configs below. |

### Tier 1 — accuracy-valid configs (target ≥0.92)

Measured healthy numbers for reference:
- `--max-new-tokens 8192` proper eval → **0.985**
- shared expert **unfused (BF16)** (`--disable-shared-experts-fusion`) → **0.904** (invalid 0.005)
- a16w4 + proper instruction prompt → **0.970** (invalid 0.000)
- GOOD checkpoint (`Qwen3.5-397B-A17B-MXFP4`, bf16-excluded shared expert) → **0.968**

### Tier 2 — KNOWN-BAD configs (structurally cap ~0.46–0.67 → NEVER SHIP)

These fail on **wrong reasoning (low invalid)**, not eval. The cause is the
`Qwen3.5-397B-A17B-MoE-MXFP4` checkpoint's **fp4 shared expert** (applied every
token, error compounds ~0.997^48≈0.87 across layers). Recognize and reject:

| Config | GSM8K | Why |
|---|---|---|
| fused fp4 shared expert (online requant) | ~0.61–0.67 | fp4 shared error compounds; kernel is correct, it's a checkpoint limit |
| a16w4 on MoE-MXFP4 (bad ckpt), bare bench | ~0.46 | fp4 shared degenerates under bf16 acts (invalid ~0.52 is partly eval artifact) |
| a4w4 on MoE-MXFP4 | ~0.62 | act-noise masks it but lower quality-when-valid |

**Fix for accuracy = give the shared expert bf16 precision** (unfused, or use the
`Qwen3.5-397B-A17B-MXFP4` bf16-shared checkpoint). Sources:
[[qwen35-shared-expert-fusion-online-requant]], [[a16w4-gsm8k-accuracy-regression]].

## How to run

- **Inside `/perf-sweep`:** GSM8K gate runs before any benchmark conc
- **Inside `/validate`:** explicit accuracy step between baseline and after
- **Standalone:** the protocol command above (thinking models); DSv4/R1 use plain `bench_sglang.py`

## Fail actions

- Accuracy fail → **do not** publish benchmark numbers as valid.
- **High invalid** → fix the eval and re-run before concluding anything about weights.
- **Low invalid + low acc** → real regression; if it's a known-bad config, don't ship
  and don't relax the threshold to make it pass.
- `/validate` returns `status: fail`; `/kernel-fusion-pipeline` auto-decides retry/skip.

## Related

- [[workloads]] — accuracy uses `canonical-8k` shape for the gate
- [[gates]] — where the accuracy gate sits in the ship funnel
