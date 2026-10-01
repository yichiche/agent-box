# Analysis patterns by change type

Phase 3 of [`/sglang-pr-review`](../SKILL.md). Each section is a trace procedure:
what to follow, and the failure it is designed to catch. Apply only the sections
the diff actually touches — a review that runs every pattern on every PR reads
as noise and buries the one finding that matters.

The common shape of every pattern below: **the diff shows one path; the bug is in
the path it did not show.** Inference engines carry many variants of the same
computation (quantized / unquantized, fused / unfused, TP=1 / TP=8, normal /
dual-stream / DeepEP), and a change that is correct in the variant the author ran
is routinely wrong in one they did not.

---

## Weight loading changes

Touches `models/*.py` `load_weights`, `stacked_params_mapping`,
`expert_params_mapping`, or any `weight_loader`.

**Trace**: for every weight the model owns, follow
`checkpoint key → mapping entry → param name → weight_loader(shard_id) → param slice`.

1. List the checkpoint key patterns the model accepts (HF naming, plus any
   quantized-checkpoint variants: `.weight_scale`, `.weight_scale_inv`,
   `.input_scale`, `.g_idx`, `.qzeros`).
2. For each, confirm exactly one mapping entry matches, and that a renamed or
   fused parameter did not silently stop matching — an unmatched key usually
   falls through to a `continue`, so the weight stays at its initialized value
   and the model answers fluent nonsense with no error.
3. Confirm the loaded-key set is asserted somewhere (`loaded_params`, a
   `KeyError`, or a count check). A PR that adds a fused parameter but not its
   mapping entry is only caught by such a check.
4. Fused layers: verify `shard_id` ordering (`q`/`k`/`v`, `gate`/`up`) and that
   the fused parameter's shard offsets match the checkpoint's layout.
5. Per-expert vs bulk: `expert_params_mapping` must cover `w1`/`w2`/`w3` **and**
   their scales, for shared experts and redundant (EPLB) experts too.

**Classic failures**: a weight silently never loaded; `w1`/`w3` swapped (output
is plausible but wrong); scales loaded into the wrong expert slot; shared-expert
weights skipped when `n_shared_experts > 0`.

---

## Forward path changes

Touches `forward`, `forward_normal`, `forward_deepep`, a communicator, or a
layer boundary.

**Trace**: enumerate every forward variant the module can take, then check the
change in each. In this codebase that typically means:

- normal single-stream
- dual-stream / two-batch-overlap (TBO)
- DeepEP / MoRI dispatch-combine
- CUDA-graph capture vs eager
- prefill vs decode vs target-verify (speculative decoding)
- `enable_dp_attention` on/off

A change applied to only one of these branches is the single most common
structural bug. Grep the sibling branches by name before concluding the change
is complete.

**Also check**: anything captured into a CUDA graph must not depend on a Python
value that varies per batch; a new tensor allocated inside a captured region
must be pre-allocated or it is captured as a fixed address.

---

## Quantization changes

Touches `layers/quantization/*`, or any producer/consumer of a quantized tensor.

**Trace**: follow the `(tensor, scale)` pair as one unit from production to
consumption.

1. Find where the quantized value is produced. If the call returns a tuple and
   the diff discards an element (`out, _ = quant(...)`), that discarded element
   is usually the scale — the highest-yield single grep in a quantization
   review.
2. Follow the scale to every consumer. A dequant that defaults a missing scale
   to `1.0` degrades accuracy silently rather than raising.
3. Check block/group granularity consistency: per-tensor, per-channel,
   per-token, and 128×128 block scales are not interchangeable, and a shape that
   broadcasts is not therefore correct.
4. Check the unquantized path still works — `quant_config is None` is a real
   configuration, not a theoretical one.

**Verification bar**: a quantization change with no accuracy number (GSM8K or
equivalent) is unreviewable. Ask for it rather than guessing.

---

## Distributed / communication changes

Touches `distributed/`, `layers/communication*`, or any `all_reduce`,
`all_gather`, `reduce_scatter`, dispatch/combine.

**Trace**: re-derive shapes and ranks for TP ∈ {1, 2, 4, 8}, and for EP and PP
if the module participates.

- TP=1 must not execute a collective that assumes `world_size > 1`.
- A sharded dimension must divide evenly, or the code must handle the remainder;
  check against real head counts and intermediate sizes, not round numbers.
- Every rank must reach every collective on every path. A collective inside an
  `if` that is rank-dependent or batch-dependent hangs instead of failing —
  look specifically for early `return`s added above a collective.
- Fused comm kernels (AllReduce+RMSNorm, +Quant) must have a fallback that is
  numerically equivalent, not merely shape-compatible.

---

## Fallback / guard changes

Touches a `try/except`, a capability check, an `is_hip()` / `is_cuda()` branch,
or a "use fast path if available" guard.

**Trace**: take the fallback branch and compare its output to the fast path on
**dtype, shape, layout, and scale**, not just "it returns something".

- A fallback returning bf16 where the fast path returned fp8 produces a dtype
  error far downstream, or worse, a silent upcast that passes.
- `except Exception: pass` around a kernel import hides a build failure as a
  permanent silent slowdown. At minimum `logger.debug` with the exception.
- A guard keyed on a device property must have been evaluated on the device that
  the guard excludes — otherwise it is an untested branch by construction.

---

## Env var changes

**Trace**: default behaviour with the variable unset must equal the old
behaviour, unless the PR states the default change and justifies it.

- Where is it read? A read at import time cannot be changed by a server arg
  parsed later; a read in the hot path costs an `os.environ` lookup per call.
- Parsing: `bool(os.environ.get("X", "0"))` is `True` for `"0"`. Check for the
  codebase's `get_bool_env_var` helper rather than ad-hoc parsing.
- Interaction: does it combine with an existing flag in a way that produces an
  invalid configuration? Those should fail at startup, not at first forward.

---

## Cache / memoization changes

Touches `@lru_cache`, a dict keyed cache, a compiled-kernel cache, or a
KV/radix/hierarchical cache.

**Trace**: enumerate every input that changes the cached value, and check the
key contains all of them.

- Kernel caches: dtype, shape class, device capability, block size, TP rank —
  a key that omits dtype returns a bf16 kernel for an fp8 call.
- `@lru_cache` on a method keys on `self` and keeps the instance alive forever;
  on a hot path it is also a lock.
- KV-cache changes: check allocate/free symmetry on every exit path, including
  abort, retraction, and preemption — a leak here shows up as gradually falling
  throughput, not as an error.
- Invalidation: anything cached across a weight update (`update_weights`) must
  be invalidated by it.

---

## Scheduling / batching changes

Touches `managers/scheduler*`, batching policy, chunked prefill, retraction.

**Trace**: a request's lifecycle — queued → prefilled (possibly chunked) →
decoded → finished / aborted / retracted / preempted.

- Does the change hold a reference that outlives the request (leak), or free
  something a retracted request will need again (corruption)?
- Does an accounting counter (`num_running_reqs`, token budgets) get decremented
  on *every* exit path, including the exceptional ones?
- Starvation: a policy change that prioritises one class of request must still
  make progress on the others.

---

## Kernel changes (CUDA / HIP / Triton)

Touches `python/sglang/kernels/**` or a `.cu` / `.hip` / Triton source.

**Trace**: the launch configuration against the extreme inputs, not the
benchmark input.

- `grid = ceil(n / BLOCK)` — check `n == 0` (launches nothing, or launches a
  block that reads out of bounds) and `n` not a multiple of `BLOCK`.
- Triton: every `tl.load`/`tl.store` needs a mask when the dimension is not
  known to be a multiple of the block; `other=` must be the identity for the
  reduction that follows (`0.0` for sum, `-inf` for max).
- Shared memory / LDS request must fit the target architecture, and the target
  is not always the one the author built for.
- `num_warps` / `num_stages` tuned on one shape are not valid for all shapes;
  an autotune key must include the shapes that change the best config.
- Accumulation dtype: fp16/bf16 accumulation over a long reduction loses
  accuracy that a short benchmark will not show.

**Verification bar**: a kernel PR needs both a correctness test against a
reference implementation and a benchmark. Missing reference comparison is a
`request-changes`-grade gap for a numerics-affecting kernel.
