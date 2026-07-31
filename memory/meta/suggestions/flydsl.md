---
status: accepted            # proposed | accepted | rejected
theme: flydsl
notes: 33
generated: 2026-07-17
resolved: 2026-07-17 — promoted the highest-value repeated foot-gun to
  [[../../gotchas/flydsl-tuned-csv-head-mismatch]] (phantom speedup from tuned-CSV
  head mismatch). Other FlyDSL foot-guns already owned: JIT baton/VRAM
  [[../../gotchas/aiter-jit-baton-vram]], version skew [[../../gotchas/aiter-version-skew]].
  No standalone /flydsl skill needed — knowledge is now split across those gotchas +
  the GDN candidate card.
---

# Suggestion: consolidate "flydsl" knowledge

**33 vault notes** touch `flydsl` but no skill/workflow/gotcha owns it yet.
This is a candidate for a curated workflow or a `/`-skill. **Review and either
promote the stable facts (via /memory-capture) or set `status: rejected`.**

## Contributing notes
- [`gotchas/aiter-jit-baton-vram.md`](../../gotchas/aiter-jit-baton-vram.md)
- [`journal/2026-06/-sgl-workspace-aiter__aiter-ci-fail-verify-on-base.md`](../../journal/2026-06/-sgl-workspace-aiter__aiter-ci-fail-verify-on-base.md)
- [`journal/2026-06/-sgl-workspace-aiter__aiter-env-skew-flydsl-from-c-void-p.md`](../../journal/2026-06/-sgl-workspace-aiter__aiter-env-skew-flydsl-from-c-void-p.md)
- [`journal/2026-06/-sgl-workspace-aiter__aiter-jit-deadlock-gpu-reclaim.md`](../../journal/2026-06/-sgl-workspace-aiter__aiter-jit-deadlock-gpu-reclaim.md)
- [`journal/2026-06/-sgl-workspace-aiter__flydsl-public-wrapper-api.md`](../../journal/2026-06/-sgl-workspace-aiter__flydsl-public-wrapper-api.md)
- [`journal/2026-06/-sgl-workspace-aiter__qwen35-moe-gemm-e2e-amdahl.md`](../../journal/2026-06/-sgl-workspace-aiter__qwen35-moe-gemm-e2e-amdahl.md)
- [`journal/2026-06/-sgl-workspace-sglang__flydsl-moe-fp8-intermediate-accurate.md`](../../journal/2026-06/-sgl-workspace-sglang__flydsl-moe-fp8-intermediate-accurate.md)
- [`journal/2026-06/-sgl-workspace-sglang__flydsl-moe-reduction-grid-collapse-e2e-win.md`](../../journal/2026-06/-sgl-workspace-sglang__flydsl-moe-reduction-grid-collapse-e2e-win.md)
- [`journal/2026-06/-sgl-workspace-sglang__fusion-pipeline-postfusion-baseline.md`](../../journal/2026-06/-sgl-workspace-sglang__fusion-pipeline-postfusion-baseline.md)
- [`journal/2026-06/-sgl-workspace-sglang__mxfp4-moe-not-csv-tunable.md`](../../journal/2026-06/-sgl-workspace-sglang__mxfp4-moe-not-csv-tunable.md)
- [`journal/2026-06/-sgl-workspace-sglang__qwen35-mxfp4-flydsl-fully-fused.md`](../../journal/2026-06/-sgl-workspace-sglang__qwen35-mxfp4-flydsl-fully-fused.md)
- [`journal/2026-06/-sgl-workspace-sglang__qwen35-mxfp4-fp8-prefill-attn-config.md`](../../journal/2026-06/-sgl-workspace-sglang__qwen35-mxfp4-fp8-prefill-attn-config.md)
- [`journal/2026-06/-sgl-workspace__qwen35-bf16-gemm-offline-batch-tuning.md`](../../journal/2026-06/-sgl-workspace__qwen35-bf16-gemm-offline-batch-tuning.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__3b9a6811.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__3b9a6811.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__57121791.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__57121791.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__d8f4b06f.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__d8f4b06f.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__de15b4b0.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__de15b4b0.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__e100d4c2.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-fp4bf16-flydsl-crash__e100d4c2.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-gsm8k-accuracy-regression.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-gsm8k-accuracy-regression.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-gsm8k-accuracy-regression__b4fe106e.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-gsm8k-accuracy-regression__b4fe106e.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-gsm8k-accuracy-regression__ce37ecf5.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-gsm8k-accuracy-regression__ce37ecf5.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-kernel-perf-repro.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-kernel-perf-repro.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__21a32a9d.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__21a32a9d.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__225c2933.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__225c2933.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__3f0caf60.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__3f0caf60.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__5c87007d.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__5c87007d.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__974d0165.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__974d0165.md)
- [`journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__9dfaea56.md`](../../journal/2026-07/-sgl-workspace-aiter__a16w4-moe-gemm1-optimization__9dfaea56.md)
- [`journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state.md`](../../journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state.md)
- [`journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state__647a75f5.md`](../../journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state__647a75f5.md)
- [`journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state__da6d73a9.md`](../../journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state__da6d73a9.md)

## Proposed action (pick one, then edit)
- [ ] New gotcha  `memory/gotchas/flydsl.md` — if it's a repeated foot-gun
- [ ] New workflow `memory/workflows/flydsl.md` — if it's a repeatable procedure
- [ ] New skill    `skills/flydsl/SKILL.md` — if it deserves a slash command
- [ ] Reject — noise / already implicit elsewhere

_Auto-drafted by `memory/bin/skill-suggest.sh`; detect → draft → you approve._
