# Kernel Development Loop

The fixed order for developing an aiter kernel change and getting it into SGLang.
Each stage is a filter: you do not advance until the current stage passes, and a
failure sends you **back to the unit test**, not forward. This loop is the on-ramp;
the keep/ship arbiter is still [[gates]].

```
unit test (aiter)  →  profile c4/c64 (served)  →  e2e full sweep  →  keep/ship
   ▲  geomean            │ no improvement            (>10% only)       (gates.md)
   └────────────────────┘  back to unit test
```

## Stage 1 — Unit test in aiter (correctness + shape-swept microbench)

Run the op test under `aiter/op_tests/`. Two things must hold:

1. **Correctness** — matches the reference. Fail ⇒ fix the kernel, stay here.
2. **Not overfit to one shape.** The test must sweep the shapes that reflect the
   **served conc4 ~ conc128 range** (the M / token dimension at those concurrencies) —
   not a single conc. A change tuned to one shape that flat-lines or regresses the
   others is not a kernel win.

**Kernel-improvement metric = geomean speedup across the conc4~conc128 shapes.** A
single-shape number is a red flag, never the metric. If the geomean does not improve,
the kernel is not better — go back and fix it before spending any GPU-server time.

## Stage 2 — Profiling confirm on the served path (c4 / c64)

Capture a served-model trace at **conc4 and conc64** (the kernel-dev anchors, see
[[profiling]]) and confirm the op is actually faster **in the served trace** — not
just in the isolated microbench (microbench lies 5 ways, [[gates]] Gate 2).

- **No improvement in the served trace ⇒ back to Stage 1.** The unit win did not
  transfer. The usual causes: the op is not on the hot path (wiring — [[gates]] Gate
  2.5), or the unit shapes did not reflect the served shapes. Diagnose, fix, re-loop.

## Stage 3 — Escalate to e2e (full sweep) when geomean > 10%

**Trigger:** kernel improvement (geomean across conc4~conc128) **> 10%** *and*
confirmed on the served trace at c4/c64.

Below 10%, do **not** spend a full sweep — iterate in Stages 1–2 or drop the
candidate. At / above 10%, run the **e2e full sweep**: `/perf-sweep`, whole frontier
`4 8 16 32 64 128 256`, **both** benchmark shapes ([[workloads]] benchmark set),
`canonical-8k` is the claim shape.

## Stage 4 — Keep / Ship (unchanged, per [[gates]])

The >10% geomean is only a **screen to justify the e2e spend** — it is not the keep
bar. The banking decisions stay exactly as in [[gates]]:

- **Gate K (keep)** — the op is ≥ 30% faster in the served trace, ×2 consistent, AND
  e2e did not regress. (A big unit/geomean win that regresses e2e is a REJECT.)
- **Gate S (ship)** — judged on the **stacked** e2e of all kept wins, on the raw
  `summary.csv`, `canonical-8k`.

## Why this order

- Unit test first because it is the cheapest signal and catches overfit before any
  server load.
- Geomean (not one conc) because a kernel that only helps one shape ships a regression
  at the others.
- Served-trace profile before e2e because the unit win frequently does **not** transfer
  (wiring / shape mismatch) — catching that at c4/c64 saves a wasted full sweep.
- >10% gate before e2e because a full sweep is the expensive step; screen for it.

## Related

- [[gates]] — the keep/ship funnel this loop feeds
- [[profiling]] — c4/c64 anchors, ×2 capture cost
- [[workloads]] — benchmark set (both shapes), claim rule
- [[benchmark]] — sweep flow / env
- `skills/implement-kernel/SKILL.md`, `skills/kernel-fusion-pipeline/` — automation
