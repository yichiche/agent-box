# Ship Gates — the optimization funnel

A kernel/optimization passes through gates before it can be claimed as a win. Each
gate is cheap-to-expensive; a candidate that fails a gate stops there. The gates
exist because the journal is full of expensive waste that a gate would have caught.

> **Golden rule:** a gate is a *filter*, not a *verdict*. Passing a cheap gate only
> earns the right to spend on the next one. The **keep** decision (Gate K) is made on
> the *served-trace per-op time*, not on aggregate e2e — a real kernel win must not be
> discarded just because its Amdahl share makes e2e move little. The **ship** claim
> (Gate S) is made on the *stacked* e2e of everything kept.

## The funnel

| Gate | Question | Tool | Failure = |
|---|---|---|---|
| 0 | Is this worth trying? (Amdahl leverage) | candidate card ([[../candidates/README]]) | skip |
| 1 | Does it build & load? | build + server health | fix |
| 2 | Does the kernel itself get faster? (microbench) | op test / microbench | **pre-filter only — never a keep/ship signal** |
| **2.5** | **Is the kernel actually on the hot path?** (wiring) | short trace + grep | **stop — the most valuable gate** |
| 3 | Accuracy still valid? | `[[accuracy]]` gate | do not ship |
| **K** | **Did the op get ≥30% faster IN THE SERVED TRACE (×2 consistent) AND e2e not regress?** | `/compare-kernels --budget` ×2 + `perf-sweep` verdict | discard / reject / re-sample |
| **S** | Does the **stacked** e2e of all kept wins meet the cumulative goal? | `/perf-sweep` on all-kept build | not yet shippable |

## Gate 2 — microbench is a FILTER, never a verdict

A microbench speedup does **not** predict e2e. Journal evidence:

- Prefill attention microbench **+16%**, but **e2e −27%** — the op got faster in
  isolation while the e2e regressed. A microbench win that ships on its own is how
  you ship a regression.

Use Gate 2 only to *kill* candidates that don't even get faster in isolation. A pass
means "proceed to 2.5", nothing more.

**Measure the microbench right (unit test in aiter):** sweep the shapes that reflect
the served **conc4 ~ conc128** range, and score improvement as the **geomean across
those shapes** — never a single conc. A single-shape win is overfit and ships a
regression elsewhere. Full loop: [[kernel-dev]].

## Gate 2.5 — wiring confirmation (cheapest gate, blocks the most expensive waste)

**Before running any sweep, confirm the new kernel actually executes on the target
path.** Capture a short trace and grep for the kernel name in the decode (or prefill)
path. If it's not there, the sweep will measure nothing.

- Journal evidence: a kernel was written and built correctly but the **decode
  dispatch was never wired** → e2e showed zero benefit. Hours of sweep time spent
  measuring a code path that never ran.

```bash
# short capture (conc 4, few steps) then confirm the kernel is present
grep -i "<your_kernel_name>" <trace-DECODE...>  # must appear on the decode path
```

Only after the kernel is confirmed on the hot path do you spend a full sweep.
Related dispatch/wiring conventions: [[sglang-integration]].

### Escalation to e2e — the >10% screen

Confirm the improvement on the **served trace at c4 / c64** ([[profiling]]). Then:

- **geomean improvement > 10%** (across the conc4~conc128 shapes, confirmed on the
  served trace) ⇒ **escalate to a full e2e sweep**.
- **no improvement on the served trace** ⇒ back to the unit test (the unit win did not
  transfer — wiring or shape-mismatch). See [[kernel-dev]].

This >10% is only a **screen to justify the e2e spend** — it is *not* the keep bar.
The keep decision below (Gate K) is still the served-trace **≥30%** per-op rule.

## Gate 3 — accuracy

Full rule in [[accuracy]]. Key: a low score is triaged by **invalid rate** —
high invalid = eval artifact (fix eval), low invalid + low acc = real degradation
(don't ship). Never relax the threshold to pass a known-bad config.

## Gate K — KEEP decision (per-op, on the served trace)

**This is where a change is banked, and it is NOT gated on aggregate e2e ≥5%.**
Rationale: `≥5% e2e` as a keep bar lets Amdahl hide a *real* win — an op that is
genuinely 30% faster but only ~3% of e2e disappears into the aggregate and would be
wrongly discarded. Keep it if it truly got faster where it runs, and it doesn't make
the whole system worse.

**Measure the causal thing, in-situ:** the target op's time **in the served-model
profiling trace**, not a standalone microbench (microbench lies 5 ways — see Gate 2).

Procedure:
1. Capture the served trace for **baseline ×2** and **after ×2** (decode conc4 is fine).
2. Extract the target logical-op's time each run via `/compare-kernels --budget` (the
   category/op budget works on decode traces too). Fusion that changes kernel count →
   compare the whole affected op's budget, not one kernel name.
3. **Consistency:** the two baseline runs must agree within **<5%**, likewise the two
   after runs (per-op trace time is far more stable than e2e throughput, so this is
   achievable). If not → capture more / longer before deciding.
4. **Improvement:** `(base − after)/base ≥ 30%` on the consistent means.
5. **e2e guardrail:** the e2e verdict (Gate S tool / `perf-sweep`) must **NOT be
   REGRESSION**. We drop the "e2e must *improve*" requirement, but never the "e2e must
   not get *worse*" one — an op can be 30% faster in isolation yet perturb
   scheduling/memory into an e2e loss (the +16% micro / −27% e2e class).

Outcomes:

| served-trace ×2 | e2e | → |
|---|---|---|
| consistent, ≥30% faster | not regressed | ✅ **KEEP** → append to the ledger (banked) |
| consistent, ≥30% faster | **REGRESSED** | 🔴 REJECT (net loss despite kernel win) |
| <30% / absent / 0 | — | 🔴 REJECT or fix wiring (Gate 2.5) — **this is the "unit win didn't transfer" case, now an explicit output** |
| two runs disagree >5% | — | ⚠️ re-sample |

Tool: `skills/perf-sweep/keep_decision.py` encodes this (unit-tested). It appends
KEEP entries to a ledger for the cumulative goal.

## Gate S — SHIP claim (stacked e2e)

- **KEEP ≠ shippable claim.** Kept wins are *banked*; the perf claim is the **stack**.
- Build with **all kept wins together**, run one `/perf-sweep`, and judge the e2e
  delta against the **cumulative goal** (e.g. 10%). Amdahl-small wins interact — the
  stack can be less *or* more than the sum of parts, so it must be re-measured together.
- **Read the raw `summary.csv`** (journal rule; agents fabricate). Workload =
  `canonical-8k` ([[workloads]]); baseline and after use the same preset/config.
- **Accuracy-correct but NET LOSS = REGRESSION** even at the stack level (a16w4 case).

### Auto-verdict — WIRED into perf-sweep (2026-07-16)

`perf_sweep.sh` now emits **WIN / REGRESSION / INCONCLUSIVE** at the finish when
`BASELINE_CSV` points at a prior run's `summary.csv`. It computes the verdict from
the **raw csv** and writes `verdict.json`. It mirrors
`debug/perf-regression/regression.py` (direction-aware per-metric deltas, monitored
metrics + thresholds from `config.py`: `total_throughput` higher-better,
`median_e2e_latency_ms` lower-better, `REGRESSION_THRESHOLD_PCT=2`; accuracy drop
>2pp). Rules:

- **Config exact-match gate** via `run_meta.env` (MODEL/TP/IL/OL + `SERVER_SIG`): a
  shape/config mismatch is **refused → INCONCLUSIVE**, never a win.
- **REGRESSION** if any monitored metric worsens beyond threshold OR accuracy drops
  beyond 2pp — **even if throughput improved** (accuracy-correct-but-net-loss rule).
- **WIN** = `total_throughput` ≥ `SHIP_THRESHOLD_PCT` (default 5%) at ≥ half the
  shared concurrencies, no regression.
- **INCONCLUSIVE** otherwise (noise band, or no/again-mismatched baseline) — silence
  is never treated as pass.

Knobs: `BASELINE_CSV`, `SHIP_THRESHOLD_PCT`, `REGRESSION_THRESHOLD_PCT`,
`ACC_REGRESSION_THRESHOLD_PP`, `VERDICT_EXIT` (see [[../../skills/perf-sweep/SKILL]]).
NOTE: CV needs repeated runs (a single sweep measures each conc once) — not yet
computed; a future `REPEATS>1` mode would add it.

## Related

- [[kernel-dev]] — the unit→profile→e2e on-ramp that feeds this funnel
- [[time-budget]] — how many candidates to run this funnel on, per budget
- [[workloads]] — canonical vs diagnostic shapes; ≥5% claim rule; benchmark set
- [[accuracy]] — Gate 3 detail, two-tier thresholds
- [[benchmark]] / [[profiling]] — the tools each gate uses
