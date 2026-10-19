---
name: gfx950-asm-development
description: Develop, validate, and tune hand-written AMD gfx950 assembly kernels, with rocprof profiling. Use for .s kernel development, HIP code-object integration, or assembly-kernel performance work on MI355X.
category: kernel-opt
---

# gfx950 ASM Development

Produce a correct assembly kernel, a working launch path, and a reproducible measurement. Reuse the project's build, loader, and test harness where available. Target AMD gfx950 / MI355X; verify the actual device and ROCm toolchain before applying architecture-specific instructions.

## 1. Establish the kernel contract

Read the caller and reference implementation. Identify:

- Input/output shapes, dtypes, strides, alignment, and buffer ownership.
- Numerical requirements, reduction order, and any permitted approximation.
- Grid/workgroup dimensions, supported tail sizes, and empty-input behavior.
- The current kernel, representative workload cases, and the requested performance metric.

Infer these from the project when possible. Resolve a missing interface requirement before implementing code that depends on it. Use workload shapes supplied by the user or found in the application.

## 2. Implement the kernel

Establish a correct implementation and a callable entry point. Make the following explicit in the source or wrapper:

1. Kernel-argument offsets, sizes, and alignment; keep them consistent with the host launcher and code-object metadata.
2. Work-item/workgroup indexing, the declared wave mode, and the mapping from logical elements to threads.
3. Register use and the kernel descriptor's resource declarations.
4. Bounds checks and masks for partial tiles or irregular inputs.
5. Load/store completion, dependencies, and synchronization around shared data.

Use the target ISA and ABI documentation to check instruction semantics. An unfamiliar instruction or lane mapping needs an isolated correctness check on the target device. Do not assume timing or layout properties from a different GPU.

For LDS-DMA, transposed LDS reads, FP8 MFMA, packing, and dependency handling used in attention kernels, read [references/instruction-patterns.md](references/instruction-patterns.md).

For standalone assembly, a typical build sequence is:

```bash
ASM_LLVM_BIN="${ROCM_PATH:-/opt/rocm}/llvm/bin"
"$ASM_LLVM_BIN/clang" --target=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -x assembler -c kernel.s -o kernel.o
"$ASM_LLVM_BIN/ld.lld" -shared kernel.o -o kernel.co
"$ASM_LLVM_BIN/llvm-readobj" --notes --symbols kernel.co
"$ASM_LLVM_BIN/llvm-objdump" -d --mcpu=gfx950 kernel.co > kernel.disasm
```

The source must contain suitable AMDHSA kernel metadata. Match the project's ROCm version and code-object ABI; use its build command if it already handles these requirements. Inspect the emitted symbol, metadata, and disassembly after a successful build.

If compilation or loading fails, check the target, compiler, metadata, and argument ABI first. Fix an environment or interface mismatch before adding alternate execution paths.

## 3. Integrate and validate

Use the existing HIP module loader when possible. For a new launcher, load the code object, resolve the exported kernel name, pass arguments with the declared layout, and launch the matching grid/workgroup dimensions. Keep buffers alive until completion and check HIP errors at launch and synchronization.

Validate through the same entry point that the application will use:

- Compare with a trusted reference using tolerances appropriate to the dtype and operation. Check NaN/Inf behavior where relevant.
- Cover normal cases, small inputs, partial tiles, nontrivial strides, and irregular or empty segments supported by the interface.
- Use structured inputs and targeted cases to isolate indexing, masking, reduction, and synchronization errors.
- After framework integration, run the relevant operator tests and an application-level check. Numerical kernel tests do not validate argument packing or dispatch selection by themselves.

Do not benchmark a candidate as a replacement until its relevant correctness checks pass.

## 4. Measure and make one change at a time

Record the source revision, device, ROCm/compiler versions, input case, launch configuration, and clock/load conditions. Warm up compilation and execution before timing. Define whether allocation, copies, and synchronization belong in the timed region, and use the same definition for baseline and candidate.

Use repeated device-side timings where appropriate; report a representative latency and its variation. Keep a baseline and candidate on the same workload and entry point.

Choose an optimization from observed evidence. Useful first checks are:

- Whether accesses are aligned and coalesced, and whether address calculation is unnecessarily expensive.
- Whether register pressure, spills, or workgroup resource use limit useful concurrency.
- Whether instruction sequences contain redundant work or avoidable dependency chains.
- Whether LDS layout or synchronization contributes to the observed cost.
- Whether work distribution creates a long tail on irregular inputs.

Change one factor, rerun correctness checks, then repeat the same measurement. Inspect generated ISA when the expected change depends on instruction selection or scheduling. Keep a short experiment log with the change, result, and supporting evidence; discard candidates that fail correctness or show no reproducible benefit.

## 5. Profile and diagnose performance

Use [references/profiling.md](references/profiling.md) to rank kernels with rocprofv3 traces and investigate bottlenecks with hardware performance counters. Choose counters for a specific hypothesis, relate the results to the generated ISA, and test the proposed change with the same correctness checks and benchmark.

## Deliverables

Return the changed source and launch path, exact build/test/benchmark commands, correctness results, and a compact baseline/candidate comparison. Explain the observed reason for any improvement and distinguish measured conclusions from remaining hypotheses. State which checks could not be run on the available hardware.
