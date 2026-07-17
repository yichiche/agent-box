# Candidate Queue

Optimization candidates, ranked by expected e2e payoff. A candidate enters here at
**Gate 0** ([[../workflows/gates]]); working it means walking it down the funnel. This
is the *upstream* of the ship gates — it answers "what should we even try next?"

## Candidate Card schema

One file per candidate (`<slug>.md`), frontmatter + body:

```yaml
name: <slug>
target: <logical op / functional block>        # e.g. GDN decode, MoE stage-1
phase: decode | prefill
source: <where the gap was measured>           # e.g. B200-vs-MI355 conc4 trace
s: <fraction of e2e time in this op>           # 0..1 (Amdahl s)
speedup: <achievable ratio ceiling>            # r; e.g. 4.2 (B200 parity)
confidence: <0..1>                             # feasibility in-scope
status: queued | in-progress | gated-out:<gateN> | shipped
gate: <0..4>                                    # current gate
```

## Priority formula (default — provisional, see confirm-item #2)

**Units:** `s` and `confidence` are fractions (0..1); **`headroom` and `priority` are
percentages (%)** — note the ×100.

```
headroom_pct = s × (1 − 1/speedup) × 100   # Amdahl: max e2e % removable at full parity
priority     = headroom_pct × confidence   # estimated e2e % after discounting feasibility  ← ranking key
```

- **headroom_pct** is the theoretical ceiling (get the op to the reference speedup).
- **confidence** discounts for how achievable that is in-scope (GEMM at HW floor →
  low; a clear algorithmic borrow → higher).
- Rank by `priority` (an *estimated e2e %*). The `priority:` field in each card is in
  the same **percent** unit. Refine `s`/`speedup`/`confidence` as real measurements
  come in.

Worked example (GDN): `s=0.11, speedup=14, confidence=0.7` →
`headroom = 0.11×(1−1/14)×100 = 10.2%`, `priority = 10.2×0.7 = 7.1`.

## Ranked queue (seeded 2026-07-16 from [[qwen35-decode-b200-vs-mi355x]])

Qwen3.5-397B-A17B-MXFP4 decode, conc4 IL8k/OL1k (decode ≈96% of e2e). `s` derived
from the note's "30%-cut ≈ e2e leverage" figures; **all values are estimates to
refine**.

| Rank | Candidate | s | speedup | headroom | conf | **priority** | note |
|---|---|---|---|---|---|---|---|
| 1 | [[moe-sorting-blockscan]] | 0.13 | 2.6× | 7.8% | 0.6 | **4.7** | block-scan routing, drop 16K-row decode padding |
| 2 | [[gdn-decode-ilp4-bf16state]] | 0.11 | 1.7× | 4.5% | 0.85 | **3.8** | bf16 SSM state on **existing** Triton kernel; ILP4/FlyDSL deprioritized (†) |
| 3 | [[attn-decode-fused-reduce]] | 0.063 | 6.5× | 5.3% | 0.6 | **3.2** | fold reduce_segments into unified_attention epilogue |
| 3 | [[moe-stage1-finalize-fusion]] | 0.21 | 4.2× | 16.0% | 0.2 | **3.2** | high headroom but GEMM at HW floor → only finalize-fusion angle |
| 5 | [[moe-stage2-gemm]] | 0.118 | 3.7× | 8.6% | 0.15 | **1.3** | fp4 GEMM at HW floor; lowest confidence |

† **GDN re-derived 2026-07-14** ([[qwen35-gdn-decode-flydsl-vs-bf16state]]): the seed's
14×/priority-7.1 assumed borrowing the full B200 `bf16state_mtp_ilp4` kernel. Microbench
at the REAL head config showed FlyDSL/ILP4 give ~no win (the 5× was a mismatched-head
artifact); the only real lever is **bf16 SSM state = 1.7× @ b64 on the existing Triton
kernel** (near a config knob → conf 0.85). Gated on accuracy: bf16-state error compounds
over OL1k, needs GSM8K validation. Net: lower headroom, higher confidence, cheaper to try.

**Reading it:** MoE sorting/routing now edges out GDN after the GDN re-derivation. GDN is
still attractive because its lever is nearly a config change (bf16 state pool) rather than a
kernel-author pass — try it early despite the mid rank, but validate accuracy first. MoE
stage-1 has the highest raw headroom but low confidence (the GEMM math is at the HW floor;
split-K is dead for fp4), so it ranks below feasible mid-size wins.

## Related

- [[../workflows/gates]] — the funnel a candidate walks down (Gate 0 = this queue)
- [[../workflows/profiling]] — how to (re)measure `s` for a candidate
- [[../workflows/sglang-integration]] — how to land it once it passes the gates
