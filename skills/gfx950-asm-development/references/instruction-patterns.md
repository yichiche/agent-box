# gfx950 instruction patterns for attention kernels

An FP8 attention data path commonly stages K/V tiles in LDS, prepares QK operands, converts probabilities to packed FP8, then feeds V-transpose and probability fragments to a PV matrix operation. The instruction choices below follow those data layouts.

Each snippet has its own register assignments. Integrate it with the kernel's register map, metadata, bounds handling, and launch configuration. Use a ROCm LLVM version that supports these gfx950 instructions.

## Global memory directly to LDS

`global_load_lds_dwordx4` moves 16 bytes per active lane directly into LDS. It avoids a separate VGPR staging buffer and `ds_write` sequence.

```asm
; s[4:5]: global source base
; v0: per-lane global byte offset
; s12: this wave's LDS destination base, in bytes
s_mov_b32 m0, s12
s_nop 0
global_load_lds_dwordx4 v0, s[4:5]
s_waitcnt vmcnt(0)
```

The LDS destination for lane `l` is `M0 + 16*l`. Reserve that span in the LDS allocation and choose the global addresses to place each lane's data in the desired slot. Use aligned slots compatible with the instruction's access width.

Implementation details that matter:

- Leave the required separation after writing `M0`; the sequence above uses `s_nop 0`.
- The instruction's `offset:` contributes to both the global and LDS addresses. Use `M0` to change only the LDS placement.
- Completion is tracked by `vmcnt`. Waiting only for `lgkmcnt` does not complete this DMA.
- EXEC-masked lanes do not write their LDS slots. Consumers must not treat those slots as initialized data.
- If another wave consumes the data, complete the loads and use a barrier reached by every wave in the workgroup before the handoff.

## Transposed byte reads from LDS

`ds_read_b64_tr_b8` can prepare a transposed FP8 operand directly from LDS. It is a cross-lane operation, not an ordinary eight-byte read from each lane's own address.

Within a 16-lane group, let `j` be the output lane and `k` the output byte index, with `0 <= j < 16` and `0 <= k < 8`. Output byte `k` comes from:

```text
address supplied by input lane (2*k + floor(j/8)), byte (j mod 8)
```

For example, the per-input-lane address pattern

```text
address(i) = group_base + floor(i/2)*row_stride + 8*(i mod 2)
```

turns an eight-row by sixteen-byte block into sixteen output lanes, each holding one column's eight bytes. Select each group's base so those rows are the intended reduction-dimension slice.

```asm
; v20: the address pattern described above
; v[44:45]: a matching packed probability fragment
ds_read_b64_tr_b8 v[40:41], v20
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[40:41], v[44:45], 0
```

This arrangement is useful for a PV operation: V is staged in row-major LDS storage, while the matrix instruction consumes a V-transpose fragment. Verify the address mapping, byte order, and matching probability layout together. Check LDS behavior for the actual read width; a layout suitable for one access width may behave differently for another.

## Select MFMA shape together with operand packing

For FP8 operands on gfx950:

| Instruction | A/B payload per lane | Accumulator per lane |
|---|---:|---:|
| `v_mfma_f32_16x16x32_fp8_fp8` | 8 bytes each, two VGPRs | Four FP32 values |
| `v_mfma_f32_16x16x128_f8f6f4` | 32 bytes each, eight VGPRs | Four FP32 values |

The wider reduction instruction can consume QK fragments prepared as contiguous per-lane FP8 byte ranges. A first step can initialize the accumulator with zero; subsequent steps accumulate into the existing result:

```asm
v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[0:7],  v[8:15],  0
v_mfma_f32_16x16x128_f8f6f4 v[32:35], v[16:23], v[24:31], v[32:35]
```

Changing the MFMA shape changes the required fragments, register ranges, and reduction partition. An eight-byte transposed read is not by itself a complete operand for the x128 instruction. Confirm the FP8 encoding and instruction format controls before using a different FP8, FP6, or FP4 representation.

A consistent permutation of the reduction dimension applied to both A and B preserves the mathematical dot product. This can help align operand loads, but numerical validation is still required because floating-point accumulation order can affect rounding.

MFMA-to-MFMA accumulation and MFMA-to-VALU consumption have different dependency requirements. Recheck hazards when the next consumer changes from another MFMA to softmax, conversion, or other VALU work.

## Pack FP32 values into FP8 bytes

`v_cvt_pk_fp8_f32` converts two FP32 values into a packed FP8 pair. Populate both halves of a VGPR when four FP8 values are needed:

```asm
; v0..v3: values after the operation's required scaling
v_cvt_pk_fp8_f32 v16, v0, v1
v_cvt_pk_fp8_f32 v16, v2, v3 op_sel:[0,0,1]
```

The second instruction selects the upper half of the destination. Check the resulting byte order against the MFMA operand layout, and validate scaling, saturation, and rounding under the intended FP8 format. Bit reinterpretation does not perform this conversion.

Partial register writes are instruction-specific. On gfx950, `ds_read_u16_d16_hi` writes `loaded_u16 << 16` into the VGPR. Do not use it to append a halfword while assuming the low half survives. Read and combine the values explicitly when both halves are needed.

## Separate memory waits, barriers, and arithmetic hazards

- Use `vmcnt` for the relevant global-memory operations, including global-to-LDS DMA.
- Use `lgkmcnt` for LDS/SMEM dependencies. When scalar loads and DS operations are mixed, do not infer that a particular value is ready from a nonzero threshold alone.
- A barrier coordinates waves; it does not replace the required completion waits for the data being handed off.
- Nonzero wait thresholds depend on the operations issued along every path. Recheck them after changing load order, masks, or tail handling.

Bring up a sequence with explicit completion waits, then relax waits only after the dependency accounting is clear and correctness remains intact.

`v_exp_f32` computes a base-two exponential. For a natural-exponential formulation, account for `log2(e)` in the scaling. Keep the convention consistent through normalization and reductions.

Transcendental results also need the appropriate separation before a dependent VALU read. For example:

```asm
; v4: numerator, v5: nonzero denominator
v_rcp_f32_e32 v6, v5
s_nop 1
v_mul_f32_e32 v7, v4, v6
```

Independent work may supply the required separation instead. Memory wait counters do not substitute for this arithmetic dependency handling. The reciprocal is approximate; validate its accuracy, any refinement, and zero-denominator behavior as part of the operator contract.
