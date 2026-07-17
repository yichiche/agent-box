# SGLang / aiter Integration Conventions

Rules for landing HIP/aiter kernel work in SGLang with **minimal blast radius** and
**common paths untouched**. Promoted from journal + PR-review feedback; these are the
conventions reviewers (e.g. PR #27636) enforce.

## 1. Gate with the RIGHT flag: `_use_aiter` vs `_is_hip`

| Flag | Use for | Disabled by |
|---|---|---|
| `_use_aiter` | kernels imported from `aiter.ops.*` | `SGLANG_USE_AITER=0` |
| `_is_hip` | non-aiter HIP kernels (e.g. a custom **Triton** kernel that imports nothing from aiter) | — (platform gate) |

- Gating a pure-Triton HIP kernel with `_use_aiter` would **wrongly disable it** when
  someone sets `SGLANG_USE_AITER=0`. Use `_is_hip` there.
- Source: [[hip-vs-aiter-guard]] (the `sigmoid_gate_mul` Triton kernel case).

## 2. Isolate the new path; gate at the dispatch site

- Put the new HIP/aiter path in **its own method**, and branch to it **at the
  dispatch site**. Do NOT thread `if _is_hip:` conditionals through the body of the
  shared/native function.
- Keep the platform-specific code behind one clean guard → minimal diff, easy revert.
- Source: [[forward-prepare-hip-separate-path]], [[hip-vs-aiter-guard]].

## 3. Common path stays byte-identical

- A backend-gated change must leave the **non-target path bit-identical**. Verify
  (op test `md/maxdiff ≈ 0`, or accuracy unchanged on the other backend).
- The user's standing convention: **minimal diffs, common paths untouched.**

## 4. New/narrow/WIP optimizations are opt-in (env default OFF)

- An opt that only helps a narrow regime (e.g. low concurrency) or is WIP must be
  **default OFF behind an env flag**, not silently on. Examples from journal:
  `AITER_FP4BF16_RELAX_VMCNT=1` (conc-4 −6% but conc-8 +2.4% → left opt-in, default OFF).
- Pass optional kwargs (e.g. `gate_mode`) **only when the feature applies**, so the
  default dispatch is unchanged.

## 5. Dispatch wiring checklist (kernel built ≠ kernel used)

This is the companion to **Gate 2.5** ([[gates]]). The most expensive silent failure
is a correct, built kernel that **never runs on the target path**. Before claiming any
e2e effect:

1. **Does the dispatch condition actually fire?** Confirm the server log prints the
   branch (e.g. `[fused_moe] ... INTERLEAVE: dispatching to FlyDSL kernel`).
2. **Do Parameter tags survive tensor transforms?** Canonical bug: `mxfp4.py` set
   `is_guinterleave` only as a *shuffle arg* and never tagged the **Parameter**;
   `.view(float4_e2m1fn_x2)` in `apply()` then **dropped the python attr** → aiter's
   `fused_moe_` read `is_guinterleave=False` → a16w4 **never auto-activated**. If you
   rely on a `getattr(weight, "…")` tag, re-tag after every `.view()`/reshape.
3. **Grep a short trace** for the kernel name on the target (decode/prefill) path —
   if it's absent, the sweep measures nothing (Gate 2.5).
4. Only then run the full sweep + [[gates]] Gate 4 verdict.

Source for the wiring failures: [[a16w4-gsm8k-accuracy-regression]] (integration gap
+ FlyDSL dispatch), [[aiter-version-skew]] (config/kernel-id skew).

## Related

- [[gates]] — where these sit in the ship funnel (esp. Gate 2.5)
- [[../gotchas/aiter-jit-baton-vram]], [[../gotchas/aiter-version-skew]] — ops failure modes
