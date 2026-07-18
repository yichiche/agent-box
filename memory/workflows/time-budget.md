# Time Budget → Exploration Scope

Turns a wall-clock budget into an explicit **breadth of exploration**, so a long
budget is spent trying many genuinely different things — not idling on trivial tweaks.
When the user says "you have 8 hours" / "run it overnight" / "spend the afternoon on
this", read the budget row below and commit to that scope.

## The anti-pattern (never do this with hours to spend)

- Flipping **one SGLang global/env var at a time** (chunked-prefill size,
  `max-running-requests`, cuda-graph batch sizes, `--schedule-conservativeness`…) and
  re-sweeping after each.
- Re-running the same sweep for reassurance.
- Stopping after the first candidate lands, with hours left on the clock.

A global-var sweep is a **warm-up**, not the plan. It explores a tiny, low-ceiling
corner of the space and burns a big budget for near-zero learning.

## How to read a budget

1. **Estimate unit costs first** (they set how many shots fit): one server load
   (~min), one full sweep (~min), one kernel build (~min), one unit-test shape-sweep
   (~min).
2. **Fill the budget with as many INDEPENDENT candidate lines as fit**, favoring
   breadth of *approach* over depth of tuning on any one.
3. **Breadth before depth** — start N candidate lines through Stage 1–2 of
   [[kernel-dev]] before deep-tuning any single one.
4. **Parallelize** — independent candidates go in separate worktrees
   (`implement-kernel` PIPELINE_MODE / `kernel-fusion-pipeline`), not serially.

## Budget → scope

| Budget | Expected scope |
|---|---|
| **< 1h** | One targeted thing: a single candidate through unit → profile c4/c64, **or** one before/after sweep. No exploration. |
| **1–3h** | 2–4 candidates through the [[kernel-dev]] loop (unit → profile c4/c64); escalate only the geomean-**>10%** ones to e2e. |
| **3–8h** | Full funnel on a *slate*: build a candidate list ([[../candidates/README]]), run unit-test shape-sweeps in parallel worktrees, profile the survivors at c4/c64, e2e the >10% ones, then stack-ship ([[gates]] Gate S). |
| **8h+ / overnight** | **Saturate.** Enumerate the candidate space (see menu), run many in parallel worktrees, keep a running ledger, and only fall back to global-var tuning *after* the algorithmic space is exhausted. Deliver a **ranked slate + stacked-ship verdict**, not one tweak. |

## The exploration menu (try ACROSS these — this is the breadth)

Roughly cheapest-first, but a big budget should touch several rows, not just the top:

- **kernel rewrite / new algorithm** for the hot op
- **fusion** of adjacent ops (`kernel-fusion-pipeline`)
- **data layout / packing / vectorization** changes
- **quantization / dtype variants** where accuracy allows ([[accuracy]] gate still applies)
- **tuned-GEMM config search** (aiter tuning CSVs)
- **alternate library kernels** (aiter vs `triton-custom` vs hipBLASLt)
- *last, only after the above:* **SGLang scheduler / global-var tuning**

## Rules

1. **Budget is declared up front** — in the prompt, or an env like
   `TIME_BUDGET_HOURS`. If none is given, assume the **< 1h "one targeted thing"**
   scope; never self-authorize an overnight run.
2. **A big budget buys more shots, never a lower bar.** Every candidate still passes
   the same [[gates]]; more time = more candidates, not relaxed thresholds.
3. **Keep a ledger** of every candidate and its verdict; end with a ranked slate and
   the stacked-ship verdict, not a single number.

## Related

- [[kernel-dev]] — the per-candidate loop a budget replicates in parallel
- [[gates]] — the bar every candidate meets regardless of budget
- [[workloads]] — benchmark set / claim shape for the e2e step
- `skills/kernel-fusion-pipeline/`, `skills/implement-kernel/SKILL.md` — parallel worktree automation
