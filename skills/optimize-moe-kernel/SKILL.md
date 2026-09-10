---
name: optimize-moe-kernel
description: Diagnose and optimize Mixture-of-Experts routing, grouped GEMM, activation, reduction, and finalize kernels across decode and prefill shapes and weight/activation dtypes. Use for MoE kernel performance work that must transfer from microbenchmarks to distributed serving.
metadata:
  category: kernel-opt
---

# Optimize MoE Kernels

Optimize the served MoE operation, not an isolated GEMM row. Treat routing,
alignment, both expert GEMMs, nonlinear activation, quantization, reductions,
finalize, and distributed expert imbalance as one measured pipeline.

Read the references selectively:

- Read [architecture-playbook.md](references/architecture-playbook.md) when
  choosing an algorithm, tile, split-K design, fusion boundary, or selector.
- Read [dtype-strategies.md](references/dtype-strategies.md) before changing a
  low-precision path, scale handling, accumulation, or rounding boundary.
- Read [validation-and-attribution.md](references/validation-and-attribution.md)
  before benchmarking, making a performance claim, or installing an overlay.

## Establish the performance contract

Before editing code, write down:

- Target workload phase: decode, prefill, or both.
- Served shapes: token batch `M`, top-k, local expert count, hidden/intermediate
  dimensions, tensor/expert parallel topology, and weight/activation dtype.
- Success metric: target op time, output or total throughput, TPOT, TTFT, E2E,
  memory, and accuracy requirements.
- Baseline identity: repository commits, installed kernel source hashes, runtime
  flags, model/checkpoint, hardware, compiler/runtime versions, and benchmark
  command.
- Numerical contract: accumulator dtype, scale semantics, activation formula,
  normalization, and every cast or rounding point observable by the reference.

Do not infer a served shape from concurrency alone. Verify whether graph padding,
chunked prefill, speculative tokens, capacity padding, or routing changes the
actual kernel `M` or routed-row count.

## Prove which inputs can change

Trace selectors and dispatch predicates before running a benchmark. Build a small
boundary table showing the chosen code path and configuration for each relevant
`M`, routed-row count, dtype, and stage. A change guarded by `M > 128` cannot
explain concurrency 4--128 when decode really uses `M == concurrency`.

Separate results by source revision. A cancelled run, zero-valued JSON, missing
request, changed selector, or unknown installed source hash is not evidence for
the current candidate.

## Measure the pipeline by stage

Map the served call chain and time at least:

1. Top-k/routing and local-expert mapping.
2. Sort/alignment/padding.
3. Gate/up grouped GEMM.
4. Quantization/dequantization and scale loads, when present.
5. Activation such as SiLU and elementwise multiply.
6. Down grouped GEMM.
7. Per-expert normalization, routing-weight reduction, and output combine.

Record launch count, median and tail time, bytes moved, achieved bandwidth,
occupancy constraints, registers, LDS/shared memory, spills, and useful versus
padding workgroups. Compare against a simple streaming bandwidth probe and an
independent implementation when possible.

## Reproduce distributed routing

Average routing hides the slow rank that gates tensor-parallel decode. Benchmark
fixed local-route distributions such as 0/1/2/4/8 local routes per token,
all-local, all-remote, and a captured production distribution. Report both the
distribution and the maximum rank time. Use `M * topk` as the logical route count,
but use actual local rows and padded expert blocks to explain work launched.

## Form and test one causal hypothesis at a time

Use profiles and resource arithmetic to choose the next candidate. Typical
hypotheses include insufficient small-M parallelism, excessive padded grid work,
serial K dependence, weight or activation rereads, unpack/scale overhead, poor
stage-specific tiles, register-limited occupancy, LDS barriers, or extra
epilogue launches.

For each candidate:

1. State what changes, which stage it affects, and the expected counter/timing
   movement.
2. Compile in a private cache and preserve the previous source.
3. Check correctness before timing.
4. Sweep adjacent shapes and routing distributions, not only the winning point.
5. Profile the winning and worst cases to confirm the predicted mechanism.
6. Reject and fully revert candidates whose causal prediction fails.

Keep an experiment ledger, including negative results. Tile, pipeline, cache,
prefetch, occupancy, and staging experiments often converge to the same bandwidth
roof; repeating them under new names wastes time.

## Optimize stages and dtypes independently

Gate/up and down GEMMs are structurally different. Gate/up commonly has twice the
output width and a nonlinear epilogue; down commonly rereads a larger intermediate
as routed rows grow. Allow stage-specific block sizes, K unroll, waves/warps,
pipeline depth, and cache policy.

Dtype changes the useful work available to hide memory latency. Do not copy a
BF16 tile table to packed FP4, or a packed-FP4 table to native low-precision MFMA.
Use [dtype-strategies.md](references/dtype-strategies.md) as hypotheses, then
measure on the actual instruction and scale path.

## Integrate with a narrow selector

Encode only the measured region. Keep dtype, stage, phase, and shape guards
explicit. Test exact boundaries and neighboring rows. A selector change is part
of the kernel optimization and must be versioned and validated with it.

When the kernel lives outside the application repository, keep an installable
patch or pinned dependency revision. Verify that applying the patch to a clean
base reproduces the exact installed source hashes used for measurement.

## Escalate evidence in cost order

Use this funnel:

1. Structural and CPU-safe selector checks.
2. Compile/resource inspection.
3. Operator correctness and shape/routing sweep under graph replay.
4. Served trace confirming the candidate is on the hot path.
5. Matched end-to-end A/B with identical requests and complete result files.
6. Accuracy evaluation whenever quantization, accumulation semantics, or an
   observable rounding boundary changes.

Microbenchmarks decide what deserves an e2e run; they do not establish a serving
win. Keep small e2e deltas provisional until repeated runs show stable direction.

## Report the result

Lead with scope and attribution. Include:

- Exact code/config region changed and unchanged regions.
- Stage-level before/after data and the diagnosed bottleneck.
- Results across dtype, shape, and routing distribution.
- Matched e2e throughput and latency with raw result paths.
- Correctness/accuracy gates, source hashes, rejected approaches, and remaining
  limitations.

Do not describe a large-M or one-dtype improvement as a general MoE improvement.
