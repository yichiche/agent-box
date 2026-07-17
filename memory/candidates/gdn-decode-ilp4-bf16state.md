---
name: gdn-decode-ilp4-bf16state
target: GDN decode (gated delta-rule linear-attention recurrence)
phase: decode
source: B200-vs-MI355 conc4 IL8k/OL1k trace; MI355X microbench 2026-07-14
s: 0.11
speedup: 1.7
confidence: 0.85
status: queued
gate: 0
priority: 3.8
---

# GDN decode — bf16 SSM state (ILP4/FlyDSL deprioritized)

**Still a top-queue candidate, but the lever changed.** Microbench correction
(2026-07-14, [[qwen35-gdn-decode-flydsl-vs-bf16state]]) narrowed this from a broad
"borrow the B200 kernel" to one cheap, high-confidence knob.

- **AMD now:** `fused_recurrent_gated_delta_rule_packed_decode` (triton) — 7.0µs @ b4,
  42µs @ b64 at the REAL config (TP2 per-device H=8/HV=32/K=V=128, bf16).
- **THE LEVER = bf16 SSM state.** At b64 the recurrence is HBM-BW-bound (fp32 state
  R+W ~268MB/42µs ≈ 6.5 TB/s, near MI355X roofline). Halving the state to bf16 halves
  traffic → **1.70× @ b64 (42→24.8µs), 1.29× @ b4** on the EXISTING Triton kernel with
  **no kernel swap** — just allocate the GDN state pool as bf16 and feed it as
  `initial_state`. bf16-state b64 ~25µs ≈ B200 `wide_vec` 23.1µs (closes most of the gap).
- **Confidence 0.85:** it's near a config/pool-dtype change, not a kernel-author pass.
  Cheapest path = check whether sglang can allocate the GDN state pool in bf16.
- **⚠ Accuracy caveat (MUST validate first):** single-step out maxdiff fp32→bf16 state
  = 0.07 (b4) / 0.2 (b64), and error COMPOUNDS over OL1k decode steps. Run GSM8K
  (thinking protocol) before adopting. B200 ships bf16state so it's likely OK — verify
  on this stack. This is the gating risk, not perf.

**DEPRIORITIZED — FlyDSL `gdr_decode` / ILP4 kernel swap:** gives essentially NO win
over Triton at the real config (b4 6.2 vs 7.0µs = 1.13×; b64 41 vs 42µs = 1.00×). The
earlier "5× @ conc64" claim was an artifact of comparing MISMATCHED head configs
(num_v=8 vs real num_v=32) in `gdr_decode_tuned.csv`. Do not chase the HIP/FlyDSL
kernel swap for its own sake. (If integrated anyway: keep state permanently in FlyDSL's
swizzled layout + `need_shuffle_state=False`, else it's ~6× SLOWER re-shuffling every call.)

Runs on the 45 linear-attn layers/token (hybrid: 15 full-attn / 45 linear).
Land per [[../workflows/sglang-integration]]; verify wiring (Gate 2.5) + e2e verdict.
Sources: [[qwen35-gdn-decode-flydsl-vs-bf16state]] (correction), [[qwen35-decode-b200-vs-mi355x]] (original gap).
