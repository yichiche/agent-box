# Dtype-Specific MoE Strategies

Use these as starting hypotheses. Confirm the actual hardware instructions,
checkpoint layout, scale format, compiler lowering, and served bottleneck.

## Decision table

| Path | Likely extra work | Common first experiments | Numerical gate |
|---|---|---|---|
| FP32 | bytes and compute | reduce precision only if authorized; fusion/reuse | reference tolerance |
| BF16/FP16 | weight bandwidth, small-M latency | more independent work, stage tiles, split-K | accumulation and rounding points |
| FP8 weights/activations | quantization and scale traffic | native MFMA, fused quant, scale reuse | model accuracy after activation quantization |
| Packed FP4 weights + BF16 activation | unpack, transpose, block scales | vectorized packed loads, larger unroll, lane transforms | randomized scale/index tests |
| Native FP4 with low-precision activation | activation quantization and scale semantics | native MFMA after correct quant path | full accuracy evaluation |
| INT8 | zero-point/scale epilogue, INT32 accumulation | native dot/MFMA, fused dequant | asymmetric/symmetric reference cases |
| Mixed per-stage dtype | conversions between stages | fuse casts/scales and keep stage-local configs | cross-stage rounding contract |

## BF16 and FP16

There is little dequantization work to overlap with memory. At small M, the kernel
can be latency and occupancy limited even when bytes dominate at larger M.

Try:

- More waves only when resource use permits and the extra waves hide VMEM latency.
- Narrower N tiles or split-K to create workgroups at low route counts.
- Stage-specific K unroll: gate/up accumulator pressure differs from down.
- Register software pipelines when K has enough iterations to amortize them.
- A fused or split-aware finalize when reduction launches erase the GEMM gain.

Keep accumulation and reference rounding explicit. Splitting K usually changes
floating-point association; compare against the production reference tolerance and
round at the same observable boundary.

## FP8

First determine whether weights, activations, or both are FP8, and whether scales
are per tensor, row, channel, or block. A nominal FP8 kernel may spend substantial
time quantizing BF16 activations or loading scales.

Try:

- Native FP8 MFMA/dot paths with the checkpoint's scale semantics.
- Fusing activation quantization with routing/sort or the GEMM producer.
- Reusing row scales across gate/up and avoiding repeated scale conversion.
- Vectorizing scale loads and aligning block boundaries with K tiles.
- Separating compute-heavy native MFMA tuning from quantization overhead.

Any new activation quantization or changed scale calculation moves beyond a
bitwise-preserving kernel optimization. Run task-level accuracy, not only tensor
allclose.

## Packed FP4 or INT4 weights with BF16 activation

This path often remains a BF16 MFMA path after manual unpack. Its performance model
includes packed weight bandwidth, E8M0 or group-scale loads, nibble decode,
transpose/shuffle, BF16 conversion, and MFMA.

Try:

- Contiguous 128-bit packed loads.
- Decoding multiple MFMA steps from one load and amortizing scale fetches.
- Register/permlane transpose before LDS staging when LDS traffic is the limiter.
- Direct global-to-LDS copies when they truly bypass VGPR staging.
- Fewer waves or larger K unroll when unpack supplies latency-hiding work.
- Different gate/up and down BN to balance weight streaming and intermediate reads.

Always test randomized, non-uniform scale exponents. Constant scales allow indexing,
edge, and lane-layout bugs to pass. Include K/scale-block boundaries and adjacent N
tiles.

## Native FP4

Native FP4 instructions can remove dozens of unpack/conversion instructions, but
they commonly require compatible low-precision activations and scale operands.
Quantizing BF16 activations to FP8/FP4 changes model numerics.

Before treating native FP4 as a performance candidate, require:

1. A correct activation quantizer and scale format.
2. Verified operand lane layout and instruction semantics.
3. Tensor correctness across random scales and edge values.
4. Resource comparison; a shorter instruction stream can still raise VGPR use.
5. Task-level accuracy on the target model.

Reject a probe with large output error even when one GEMM stage is faster. A native
instruction is not a valid optimization until its complete numerical path exists.

## INT8

Distinguish symmetric from asymmetric quantization. Track zero points, per-channel
or group scales, accumulator overflow, and the dtype used for dequantization.

Try native integer dot/MFMA, wider packed loads, scale/zero-point fusion in the
epilogue, and reuse of activation scales. Validate extreme values, non-zero zero
points, odd group boundaries, saturation, and INT32 accumulator behavior.

## Mixed dtype pipelines

Do not optimize each GEMM in isolation if a conversion is inserted between them.
Account for activation output dtype from gate/up, scales produced for down, scratch
dtype for split-K, finalize input, reference rounding, and routing accumulation
dtype. A faster GEMM that adds a quantize/dequantize launch can regress the complete
MoE pipeline.

## Dtype-specific selector discipline

Keep independent selector branches when instruction mix differs. BF16 may prefer
more waves and smaller K unroll while packed FP4 prefers fewer waves and a large
unroll. Share configuration only after measurements show the same optimum and
resource behavior.
