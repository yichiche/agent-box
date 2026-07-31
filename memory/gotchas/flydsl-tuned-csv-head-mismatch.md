# FlyDSL phantom speedup — tuned-CSV head-config mismatch

A FlyDSL kernel can look dramatically faster than the Triton baseline **only because
the number you read came from a tuned-CSV row at a DIFFERENT head config than the model
actually runs.** Always re-microbench at the REAL per-device head config before believing
a speedup — and before letting it set optimization priority.

## Symptom

The GDN decode "5× @ conc64" claim (which seeded [[../candidates/gdn-decode-ilp4-bf16state]]
at 14×/priority-7.1) was an artifact: `gdr_decode_tuned.csv` was read at `num_v=8`, but the
real Qwen3.5-397B TP2 config is `num_v=32` (linear_num_value_heads=64 / TP2). At the real
config FlyDSL `gdr_decode` gives **~no win** over Triton `packed_decode` (b4 6.2 vs 7.0µs =
1.13×; b64 41 vs 42µs = 1.00×). Confirmed 2026-07-14, [[../journal/2026-07/-sgl-workspace-aiter__qwen35-gdn-decode-flydsl-vs-bf16state]].

## Why it's expensive

A phantom per-op speedup propagates straight into the Amdahl priority formula
([[../candidates/README]]): a false 14× inflated GDN's headroom and top-ranked the whole
queue on a lever that doesn't exist. Wrong `speedup` → wrong ranking → wasted engineering.

## Rule

- Derive the microbench config from the deployed `config.json` **÷ TP**, not from whatever
  the CSV was tuned at. For Qwen3.5-397B TP2: H=8, HV=32, K=V=128, bf16.
- Validate the harness reproduces the real trace timing before trusting a delta (sglang
  `fused_recurrent_gated_delta_rule_packed_decode` at b4/b64 ≈ trace conc4/conc64).
- Separate the *real* lever from the kernel-swap hype: for GDN it was **bf16 SSM state**
  (1.7× @ b64 on the existing Triton kernel), not FlyDSL/ILP4.
- FlyDSL drop-in gotcha: `need_shuffle_state=True` re-shuffles the whole SSM state every
  call → ~6× SLOWER; keep state permanently in swizzled layout + `need_shuffle_state=False`.

Related: [[aiter-version-skew]] (CSV version-locked to its aiter), [[aiter-jit-baton-vram]].
