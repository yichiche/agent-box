---
description: Implement a kernel optimization for SGLang on MI355/ROCm. Takes a free-text description of the change, designs it in plan mode, implements it, validates with /validate, and commits only after validation passes. Use when the user says "/implement-kernel" followed by an optimization description.
---

# Implement Kernel Optimization

End-to-end workflow: design → implement → validate → commit a kernel optimization for SGLang on MI355 (ROCm/HIP). **Never commit before validation passes.**

## Usage

```
/implement-kernel <description of what to optimize>
```

If no description is provided, ask with `AskUserQuestion`.

---

## Step 0: Gather optimization design

### If the user provided a description after `/implement-kernel`:

Parse the free text for:
- **What to optimize**: the specific kernel, module, or operation
- **Tier**: infer from the description:
  - Tier 1 (code refactor) — keywords: "switch dispatch", "remove", "fuse existing", "enable flag", "set env var"
  - Tier 2 (Triton kernel) — keywords: "write Triton", "fuse into Triton", "Triton kernel"
  - Tier 3 (aiter C++/HIP) — keywords: "tune CK", "aiter kernel", "HIP kernel", "assembly"

### If no description provided:

Use `AskUserQuestion` with these questions:

1. **What to optimize?** (free text — e.g., "fuse softmax+weighted_sum in compressor decode", "switch Q proj to BF16 for small-M decode")
2. **Which tier?**
   - Tier 1: Code refactor (Python-level, no new kernels) (Recommended)
   - Tier 2: Triton kernel (write/modify Triton kernel)
   - Tier 3: aiter C++/HIP kernel (CK tile, assembly, low-level HIP)

### Identify target files

Use the source file reference table (at the bottom of this document) to identify which files the optimization will touch. If ambiguous, ask the user.

---

## Step 1: Design the implementation (plan mode)

Enter plan mode with `EnterPlanMode`.

### 1a. Read the current MI355 (HIP) code path

Read the target files identified in Step 0. Trace the full call chain from the SGLang layer Python code down to the kernel invocation on HIP:

1. Read the SGLang layer file — find the `is_hip()` / `_use_aiter` branch
2. Trace into aiter library at `/sgl-workspace/aiter/` if applicable
3. Identify what operations run as separate kernels

### 1b. Read the B200 (CUDA) reference path

Read the same SGLang layer file's CUDA path to understand what the "goal" looks like:

1. Find the non-HIP branch
2. Identify which operations are fused
3. Note kernel counts and expected timings (from `/compare-kernels` output if available)

### 1c. Check for existing aiter fused ops

**ALWAYS do this before writing new kernels.** Search `/sgl-workspace/aiter/aiter/ops/` for fused variants:

```bash
ls /sgl-workspace/aiter/aiter/ops/*.py
grep -l '<relevant_keyword>' /sgl-workspace/aiter/aiter/ops/*.py
```

Key fused ops to check:
- `gated_rmsnorm_fp8_group_quant.py` — fused norm+quant
- `fused_qk_norm_rope_cache_quant.py` — fused norm+rope+cache+quant
- `fused_qk_rmsnorm_group_quant.py` — fused rmsnorm+group_quant
- `fused_split_gdr_update.py` — fused split+gather+dequant+reduce+update

### 1d. Design the change

Write a plan with this structure:

```
## Optimization: <title>
Tier: <1/2/3>
Expected savings: ~<X> us per layer

## Files to modify

### <file_path_1> (lines X-Y)
Before: <what currently happens — describe which kernels fire>
After:  <what should happen — describe which kernels fire>
Change: <specific code change — function call, dispatch switch, new code>

### <file_path_2> (lines X-Y)
...

## Expected kernel trace change
Before: <list of kernels with durations>
After:  <list of kernels with durations>
Net savings: <X> us

## Risk assessment
- Accuracy risk: <Low/Medium/High — why>
- Performance risk: <Low/Medium/High — why>
- Rollback: <how to revert if it breaks>
```

### 1e. Exit plan mode

Call `ExitPlanMode` to present the plan to the user for approval.

**Do NOT proceed to Step 2 until the user approves the plan.**

---

## Step 2: Implement the changes

After plan approval, make the code edits.

### For Tier 1 (code refactor):
- Modify dispatch logic, switch function calls, remove unnecessary ops
- Change env var defaults, enable fused paths
- Remove redundant casts or copies

### For Tier 2 (new Triton kernel):
1. Write the Triton kernel in the appropriate file
2. Add a Python wrapper function
3. Add dispatch logic in the calling layer (guard with `is_hip()` or `_use_aiter`)
4. Add import statements
5. Add a fallback path that uses the old code (behind an env var or flag)

### For Tier 3 (aiter kernel):
1. Modify the aiter op wrapper at `/sgl-workspace/aiter/aiter/ops/<op>.py`
2. If adding new CK config: modify the config CSV at `/sgl-workspace/aiter/aiter/configs/`
3. If adding new C++ kernel: modify `csrc/` files (rare — usually tune existing)

### Syntax and lint check

After making changes, run:

```bash
# Syntax check — import the modified module
cd /home/yichiche/sglang && python3 -c "import sglang.srt.layers.<module_name>"

# Lint check
cd /home/yichiche/sglang && python3 -m ruff check <changed_files> --fix
```

If either fails, fix the issues before proceeding.

---

## Step 3: Validate

**Do NOT commit before validation passes.** Invoke the `/validate` skill first:

```
Skill: validate
```

This runs the full validation pipeline:
1. **Accuracy test** (GSM8K, 2000 questions) — must pass threshold (default 0.88 for DSv4)
2. **Profiling run** — trace analysis to confirm the kernel change took effect
3. **Benchmark** — full performance benchmark

### Interpreting profiling results

After `/validate` completes the profiling step, check the trace analysis output specifically for:

- **The old kernels should be gone** (or reduced in count/time)
- **The new/fused kernels should appear** (matching the plan's "After" prediction)
- **Total layer time should decrease** by approximately the expected savings

If the profiling does NOT show the expected change:
- The optimization may not be working — warn the user
- Check if the change is only active for certain shapes/modes (decode vs extend)
- Check if CUDA graph replay is caching the old kernel sequence

### If validation fails

**STOP immediately.** Report:
- Which step failed (accuracy, profiling, or benchmark)
- The exact error or unexpected result
- Suggest: revert the change, debug, or adjust

Do NOT commit or create a PR if validation fails.

---

## Step 4: Commit (only after validation passes)

Only after `/validate` confirms accuracy, profiling, and benchmark are all good, invoke the `/commit` skill:

```
Skill: commit
Args: <short description of the optimization>
```

This will:
1. Create a feature branch (if on main)
2. Stage the changed files
3. Ask the user to approve the commit message
4. Commit with `[AMD]` or `[Perf]` tag

**Wait for `/commit` to complete before proceeding.**

---

## Step 5: Report results

Present a summary to the user:

```
=== Kernel Optimization Results ===

Optimization: <description>
Tier: <1/2/3>

Files Changed:
  - <file1>:<lines> — <what changed>
  - <file2>:<lines> — <what changed>

Validation:
  1. Accuracy: PASS (<score>, threshold: <threshold>)
  2. Profiling: CONFIRMED
     Before: <old kernel sequence and total time>
     After:  <new kernel sequence and total time>
     Savings: <X> us per layer (<Y>% improvement)
  3. Benchmark:
     Throughput: <tokens/s>
     Latency:    TTFT=<X>ms, ITL=<Y>ms

Commit: <branch> @ <hash>
Status: Ready for /commit-push and /pr
```

If all validation passes, tell the user the changes are ready for `/commit-push` and `/pr`.

---

## Source file reference table

When identifying target files, use this mapping of module/block to source files.

All SGLang paths relative to `/home/yichiche/sglang/python/sglang/srt/`.

| Module / Block | SGLang Source Files |
|---|---|
| RMSNorm | `layers/layernorm.py` (CUDA/HIP dispatch) |
| ReplicatedLinear, ColumnParallelLinear, RowParallelLinear | `layers/linear.py` (HIP quant disable), `layers/quantization/fp8_utils.py` (`dispatch_w8a8_block_fp8_linear()` — deep_gemm vs ck_gemm vs a8w8) |
| FP8 quantization dispatch | `layers/quantization/fp8.py`, `layers/quantization/fp8_utils.py` |
| MQALayer / Attention | `layers/attention/dsv4/compressor.py` (HIP path at line 396), `layers/attention/hip_flash_mla.py`, `layers/attention/deepseek_v4_backend_hip_radix.py` |
| Compressor / MHC | `layers/attention/dsv4/compressor.py`, `layers/attention/dsv4/compress_hip.py` (HIP decode/extend), `layers/attention/dsv4/tilelang_kernel.py` (CUDA fused kernels), `layers/mhc.py` |
| FusedMoE | `layers/moe/moe_runner/deep_gemm.py`, `layers/moe/cutlass_moe.py`, `layers/moe/rocm_moe_utils.py` |
| MoE TopK | `layers/moe/topk.py` |
| MoE Token Dispatch | `layers/moe/token_dispatcher/standard.py`, `layers/moe/token_dispatcher/moriep.py` |
| Communicator (fused norm+quant) | `layers/communicator.py` (fused_rms_fp8_group_quant dispatch) |
| DeepSeek-V4 Model | `models/deepseek_v4.py` (MQALayer, wq_a, wq_b, wkv_gate) |
| linear_bf16_fp32 | `../../jit_kernel/deepseek_v4.py` (tgemm.mm dispatch on HIP) |
| RoPE | `layers/deepseek_v4_rope.py` (apply_rotary_emb_triton, fused_norm_rope_inplace_triton) |

### aiter library (`/sgl-workspace/aiter/`)

| Component | Path |
|---|---|
| Python op wrappers | `aiter/ops/*.py` (gemm_op_a8w8, norm, mhc, quant, rope, attention, moe_op, etc.) |
| Fused ops | `aiter/ops/gated_rmsnorm_fp8_group_quant.py`, `fused_qk_norm_rope_cache_quant.py`, `fused_split_gdr_update.py` |
| Triton ops | `aiter/ops/triton/` |
| CK GEMM kernels | `csrc/ck_gemm_a8w8_blockscale/`, `csrc/ck_tile_gemm_moe_2stages/` |
| Pre-compiled .so | `aiter/jit/module_*.so` |
| Tuned GEMM | `aiter/tuned_gemm.py` (TunedGemm, tgemm.mm → gemm_a16w16) |
| GEMM configs | `aiter/configs/` |
| Unit tests | `op_tests/test_*.py` |

---

## Important notes

- **Priority order is strict**: Tier 1 → Tier 2 → Tier 3. Never propose Tier 3 if Tier 1/2 can achieve the same savings.
- **Always check aiter ops first** — many fused variants already exist but aren't wired up in SGLang.
- **Guard new HIP paths** with `is_hip()` or `_use_aiter` — never break the CUDA path.
- **Add env var escape hatch** for Tier 2/3 changes: `SGLANG_OPT_<FEATURE>=1` so the old path remains accessible.
- **Test decode AND extend** — some optimizations only apply to one mode.
- **Small-M vs large-M** — decode uses M=1-8, extend uses M=hundreds. Kernel performance characteristics differ dramatically between these regimes.
