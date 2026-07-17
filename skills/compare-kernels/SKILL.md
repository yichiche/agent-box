---
description: Compare kernel categories and timings between two trace analysis Excel files. Use when the user says "/compare-kernels" followed by two xlsx paths, or asks to compare kernel breakdowns across platforms (e.g., B200 vs MI355).
category: research
data_sources: [trace-xlsx, aiter-upstream, sglang-upstream, jira-amd]
---

# Compare Kernels

Side-by-side kernel comparison between two trace analysis Excel files, grouped by nn.Module for prefill/extend traces.

## Usage

```
/compare-kernels <file_A.xlsx> <file_B.xlsx>
/compare-kernels --budget <file_A.xlsx> <file_B.xlsx>   # phase-level logical-op budget diff (works on decode)
/compare-kernels --refs <keyword>                       # research only: find upstream+Jira prior art for a kernel
```

If the user doesn't provide two paths, ask for them using `AskUserQuestion`.

**Research mode (`--refs`):** skip the trace diff and jump straight to Step 8 — given a
kernel/op keyword, find prior art in aiter + sglang upstream + Jira. Useful before
starting an optimization ("has anyone already done this?"). Sources resolve through
[`_shared/data-sources.md`](../_shared/data-sources.md).

**Two views:** the default module-grouped comparison (Steps 1–4) needs a module tree
→ **prefill/extend only**. The `--budget` view (Step 4.5) is a category-based
phase-level budget that **also works on decode** traces (which collapse to
`CudaGraphReplay`). When comparing decode traces, use `--budget`.

## Step 0: Recategorize both files

Before comparing, run `--recategorize` on both files to ensure they use the latest `kernel_categories.csv`:

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py --recategorize <file_A.xlsx> <file_B.xlsx>
```

## Step 1: Read detail sheets from both files

Use Python with openpyxl to read both Excel files. For each file:

1. List all sheet names
2. Identify **detail sheets** — any sheet that is NOT "Summary", "Overview", "Module Tree", or "GPU Kernels"
3. For each detail sheet, read:
   - Row 1: title (module name, phase, kernel count)
   - The kernel detail table (columns: Module, Input Dims, Kernel Name, Duration (us), % of wall time, Category, ...)
   - Extract every kernel's **module**, **name**, **duration**, and **current category**

```python
import openpyxl

def read_detail_kernels(xlsx_path):
    """Returns list of (sheet_name, title, [(module, kernel_name, duration, category), ...])"""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    skip = {"Summary", "Overview", "Module Tree", "GPU Kernels"}
    results = []
    for sname in wb.sheetnames:
        if sname in skip:
            continue
        ws = wb[sname]
        title = ws.cell(1, 1).value or sname
        kernels = []
        header_row = None
        for r in range(1, min(ws.max_row + 1, 30)):
            for c in range(1, ws.max_column + 1):
                if ws.cell(r, c).value == "Kernel Name":
                    header_row = r
                    break
            if header_row:
                break
        if not header_row:
            continue
        col_map = {}
        for c in range(1, ws.max_column + 1):
            val = ws.cell(header_row, c).value
            if val:
                col_map[val] = c
        mod_col = col_map.get("Module")
        name_col = col_map.get("Kernel Name")
        dur_col = col_map.get("Duration (us)")
        cat_col = col_map.get("Category")
        if not name_col:
            continue
        for r in range(header_row + 1, ws.max_row + 1):
            kname = ws.cell(r, name_col).value
            if not kname or "truncated" in str(kname):
                break
            mod = ws.cell(r, mod_col).value if mod_col else ""
            dur = ws.cell(r, dur_col).value if dur_col else 0
            cat = ws.cell(r, cat_col).value if cat_col else ""
            kernels.append((str(mod or ""), str(kname), float(dur or 0), str(cat or "")))
        results.append((sname, title, kernels))
    return results
```

## Step 2: Re-classify every kernel

Load the current `kernel_categories.csv` from the profile directory and re-classify every kernel name. Compare with the category stored in the Excel file.

```python
import re, csv, os

csv_path = os.path.expanduser("~/agent-box/profile/kernel_categories.csv")
categories = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        categories.append((row["category"].strip(),
                           re.compile(row["pattern"].strip(), re.IGNORECASE)))

def classify(name):
    for cat, pat in categories:
        if pat.search(name):
            return cat
    return "other"
```

Flag any kernel where the Excel category differs from the re-classified category.

## Step 3: Group kernels by Module and compare side-by-side

This is the **core analysis**. For each pair of matching detail sheets:

### 3a. Extract module groups

Strip numeric suffixes from Module names to get the module **type** (e.g., `MQALayer_31` → `MQALayer`, `ReplicatedLinear_61` → `ReplicatedLinear`). Group consecutive kernels by module type — preserve execution order within each group.

### 3b. Match module groups between files

Match module groups by type name. The same module type appears in both files but with different kernel implementations. Group them into **functional blocks**:

| Functional Block | Module Types | Role |
|---|---|---|
| MHC Pre | DeepseekV4DecoderLayer (mhc_pre kernels) | Multi-head compressor pre-processing |
| Q Proj | ReplicatedLinear | Query projection (quant + gemm) |
| KV Norm | RMSNorm | KV normalization |
| KV Proj | ColumnParallelLinear | KV projection (quant + gemm) |
| Attn Prep | MQALayer (norm + rope + cache) | Attention input preparation |
| Compressor | Compressor, C4Indexer | Compress KV (gemm + c4/c128 + rope + hadamard) |
| Attention | MQALayer (attn + combine + rope + o_proj) | Core attention computation |
| O Proj | RowParallelLinear | Output projection (quant + gemm + allreduce) |
| MHC Post | DeepseekV4DecoderLayer (mhc_post kernel) | Multi-head compressor post-processing |
| Gate Norm | RMSNorm | MLP/MoE gate normalization |
| Gate Proj | MergedColumnParallelLinear | Gate/up projection |
| MLP/Activation | DeepseekV2MLP, SiluAndMul | Activation function |
| Down Proj | RowParallelLinear | Down projection (quant + gemm + allreduce) |
| MoE Gate | MoEGate | Router gate (gemm) |
| MoE TopK | TopK | Top-K selection |
| MoE Compute | FusedMoE | MoE expert compute (sort + gemm1 + gemm2) |
| MoE Reduce | DeepseekV2MoE | MoE output reduce + allreduce |
| MHC Post (final) | DeepseekV4DecoderLayer (mhc_post) | Final MHC post-processing |

### 3c. Print module-grouped comparison

For each functional block, show the kernels from both sides:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Q Proj (ReplicatedLinear)        B200: 21.0 us (2 kernels)     MI355: 79.5 us (2 kernels)  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  B200                                          │  MI355                                     │
│  quantization   4.7  per_token_group_quant...  │  elementwise  4.9  elementwise_kernel...   │
│  quantization  16.3  deep_gemm fp8_fp4_gemm    │  gemm        74.6  ck_gemm_xdl_cshuf...   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Gap: +58.5 us (3.8x) — MI355 slower                                                        │
│ FUSION: B200 fuses quant+gemm into deep_gemm (1 kernel). MI355 separates quant + ck_gemm.  │
│ IMPROVE: ck_gemm (74.6 us) vs deep_gemm (16.3 us) = 4.6x gap                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key annotations to include for each block:

1. **FUSION opportunity**: When File_A uses fewer kernels for the same work by fusing operations (e.g., B200's `deep_gemm fp8_fp4_gemm` = fused quant+gemm vs MI355's separate quant + gemm). Call out which kernels could be fused.

2. **IMPROVE**: When the same category kernel is significantly slower on one side. Name the specific kernel and the gap.

3. **MISSING**: When one side has a category/operation that the other lacks (e.g., B200 has explicit `flash_fwd_splitkv_mla` attention but MI355 uses `_unified_sparse_decode` in "other").

## Step 4: Module-grouped time totals

Instead of flat category totals, show time breakdown by functional block:

```
Functional Block         B200 (us)    %     MI355 (us)    %     Gap (us)  Ratio
-------------------     ----------  -----   ----------  -----   --------  -----
Attention (MQALayer)        160.2   12.3%       862.4   51.8%    +702.2   5.4x  ← OPTIMIZE
  attention                 105.1            —(in other)
  other(sparse_decode)        —               622.8
  quant(gather_deq)           —               153.0
MoE Compute (FusedMoE)     585.0   45.0%       593.4   35.6%      +8.4   1.0x
O Proj (RowParallel)       1277.9   98.3%       327.6   19.7%    -950.3   0.3x  ← B200 comm
...
```

## Step 4.5: Logical-op budget diff (phase-level attribution — works on decode too)

The module-grouped view (Step 3/4) needs a module tree, so it only works on
**prefill/extend** traces. **Decode** traces are CUDA-graph-replayed and collapse to a
single `CudaGraphReplay` node (no per-module data — confirmed on Qwen3.5). This step
gives a phase-level **time budget** that works for BOTH by using the **kernel category**
(from `kernel_categories.csv`) as the logical op. It is the robust, always-available
attribution: where does each side spend its per-iteration budget, and where's the gap.

Invoke via `/compare-kernels --budget <A.xlsx> <B.xlsx>` (or always emit it — it's cheap).

### 4.5a. Build the budget from the GPU Kernels sheet

Read the flat **GPU Kernels** sheet (present in every report, prefill or decode),
classify each kernel via `kernel_categories.csv` (the `classify()` from Step 2), and
sum durations per category → a budget that sums to the phase total.

```python
def logical_op_budget(xlsx_path):
    """category -> total_us, from the flat GPU Kernels sheet (works for decode)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["GPU Kernels"]
    # find header row + Kernel Name / Duration columns (same shape as detail sheets)
    hdr = None
    for r in range(1, 30):
        vals = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
        if "Kernel Name" in vals:
            hdr = r; col = {v: i + 1 for i, v in enumerate(vals) if v}; break
    budget = {}
    for r in range(hdr + 1, ws.max_row + 1):
        kn = ws.cell(r, col["Kernel Name"]).value
        if not kn or "truncated" in str(kn): continue
        dur = float(ws.cell(r, col.get("Duration (us)", col.get("Total (us)"))).value or 0)
        budget[classify(str(kn))] = budget.get(classify(str(kn)), 0.0) + dur
    return budget
```

### 4.5b. Diff and rank by absolute gap

```python
A, B = logical_op_budget(file_A), logical_op_budget(file_B)
tot_A, tot_B = sum(A.values()) or 1, sum(B.values()) or 1
ops = sorted(set(A) | set(B), key=lambda k: abs(A.get(k,0) - B.get(k,0)), reverse=True)
print(f"{'logical op':<16}{'A us':>10}{'A%':>7}{'B us':>10}{'B%':>7}{'gap us':>10}{'ratio':>7}")
for k in ops:
    a, b = A.get(k,0), B.get(k,0)
    ratio = (a/b) if b else float('inf')
    print(f"{k:<16}{a:>10.1f}{100*a/tot_A:>6.1f}%{b:>10.1f}{100*b/tot_B:>6.1f}%{a-b:>+10.1f}{ratio:>6.1f}x")
print(f"{'TOTAL':<16}{tot_A:>10.1f}{'100%':>7}{tot_B:>10.1f}{'100%':>7}{tot_A-tot_B:>+10.1f}")
```

### 4.5c. Read it

- The budget **sums to 100%** per side — it's a complete phase attribution, not a
  cherry-picked kernel list.
- Rank by **absolute gap (µs)**, not ratio — a 14× gap on a 4µs op matters less than a
  1.3× gap on a 4000µs op. (This is the same logic the Candidate queue uses to pick
  what to work on: [[../../memory/candidates/README]].)
- For decode, expect `moe` + `gemm` to dominate (Qwen3.5 conc4 decode ≈ 74% moe+gemm,
  measured 2026-07-16). A large `other`/`communication` slice on one side is often a
  profiling artifact (e.g. symmetric-memory allreduce), not a real compute gap — sanity
  check before filing a candidate.
- Feed the top absolute-gap ops into the Candidate queue with an `s`/`speedup`
  estimate ([[../../memory/candidates/README]]).

## Step 5: Actionable optimization recommendations

For each significant gap, provide **specific, actionable** recommendations:

### Format per recommendation:

```
1. [Block]: [Module] — [gap] us ([ratio]x)
   Problem:  [Specific kernel] is [X] us vs [Y] us on the other side
   Action:   [What to do — fuse kernels, improve implementation, use different algorithm]
   Fusion:   [If applicable — which ops B200 fuses that MI355 doesn't]
   Files to check:
     - sglang/python/sglang/srt/layers/[...].py  (dispatch logic)
     - sglang/python/sglang/srt/layers/[...].py  (kernel selection)
```

### Source file reference table

When listing files to check, use this mapping of module → source files:

| Module | Source Files |
|---|---|
| RMSNorm | `layers/layernorm.py` (lines 42-106: CUDA/HIP dispatch) |
| ReplicatedLinear, ColumnParallelLinear, RowParallelLinear | `layers/linear.py` (lines 44-55: HIP quant disable), `layers/quantization/fp8_utils.py` (lines 350-465: `dispatch_w8a8_block_fp8_linear()` — deep_gemm vs ck_gemm vs a8w8) |
| FP8 quantization dispatch | `layers/quantization/fp8.py` (lines 106-115), `layers/quantization/fp8_utils.py` |
| MQALayer / Attention | `layers/attention/dsv4/compressor.py` (line 396: HIP path), `layers/attention/hip_flash_mla.py`, `layers/attention/deepseek_v4_backend_hip_radix.py` |
| Compressor / MHC | `layers/attention/dsv4/compressor.py`, `layers/attention/dsv4/compress_hip.py` (HIP decode/extend fused paths), `layers/attention/dsv4/tilelang_kernel.py`, `layers/mhc.py` |
| FusedMoE | `layers/moe/moe_runner/deep_gemm.py` (lines 45-66), `layers/moe/cutlass_moe.py`, `layers/moe/rocm_moe_utils.py` |
| MoE TopK | `layers/moe/topk.py` |
| MoE Token Dispatch | `layers/moe/token_dispatcher/standard.py`, `layers/moe/token_dispatcher/moriep.py` |
| MoE Triton Config | `layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py` |

All paths relative to `$SGLANG_ROOT/python/sglang/srt/` (where `$SGLANG_ROOT` is the active SGLang install, detected via `python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])"`). Do NOT hardcode `$HOME/sglang` or `/sgl-workspace/sglang`.

## Step 6: Flag category mismatches

If any kernels were re-classified differently than what's in the Excel, print:

```
Category Mismatches (Excel vs Current CSV):
  File A: "flash_c4_decode" was "other", now "mhc" — re-run analysis to update
  File B: "_combine_splitk_kernel" was "other", now "attention" — re-run analysis to update
```

Suggest running `--recategorize` if mismatches are found.

## Step 7: Deep-dive implementation guidance

> **PIPELINE_MODE**: Skip this entire step. Return after Step 6 with the Tier-1 fusion list. Do NOT use `AskUserQuestion`.

After printing the comparison and recommendations, use `AskUserQuestion` to ask the user which recommendation they want to deep-dive into. Present the numbered recommendations as options.

When the user picks one, perform a **detailed implementation analysis** following these steps:

### 7a. Read the B200 (CUDA) code path

Trace the full call chain from the SGLang layer Python code down to the kernel invocation:

1. Read the SGLang layer file (from the source file reference table) to find the CUDA dispatch path
2. Identify the exact function/method that selects the kernel
3. Read the kernel wrapper or binding to understand its inputs, outputs, and semantics
4. Note any fused operations (e.g., quant+gemm in one kernel, norm+rope+cache in one kernel)

### 7b. Read the MI355 (HIP/ROCm) code path

Do the same for the MI355 path:

1. Read the `is_hip()` / `_use_aiter` branch in the same SGLang layer file
2. Trace into the aiter library at `/sgl-workspace/aiter/` to find the actual kernel implementation
3. Identify what operations are run as separate kernels vs fused

### 7c. Diff the two paths

Print a structured comparison:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ B200 (CUDA) Path                      │ MI355 (HIP/aiter) Path             │
├───────────────────────────────────────┼─────────────────────────────────────┤
│ 1. input_quant + gemm FUSED          │ 1. elementwise (cast/scale)         │
│    deep_gemm.fp8_fp4_gemm()          │    torch elementwise_kernel         │
│    → 1 kernel, 16.3 us               │ 2. dynamic_per_group_scaled_quant   │
│                                       │    aiter.ops.quant                  │
│                                       │ 3. gemm_a8w8_blockscale            │
│                                       │    aiter.ops.gemm_op_a8w8          │
│                                       │    → 3 kernels, 79.5 us            │
├───────────────────────────────────────┴─────────────────────────────────────┤
│ Gap: 63.2 us — MI355 uses 3 separate kernels where B200 fuses into 1      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7d. Propose implementation plan with priority tiers

Organize recommendations into three tiers, **always propose in this order**:

#### Tier 1: Simple kernel fusion / code refactor (Python-level, no new kernels)

These are changes in SGLang Python code that reduce kernel count or improve dispatch:

- Fuse consecutive PyTorch ops that are currently separate (e.g., eliminate redundant elementwise kernels between quant and gemm)
- Switch dispatch to a different existing aiter kernel variant that fuses more ops
- Remove unnecessary intermediate copies/casts
- Reorder operations to enable fusion

**Format:**
```
Tier 1: Code refactor — estimated savings: ~X us
  Change: [specific code change]
  File:   [exact file path and line range]

  Before (N kernels, X us):
  ┌────────────────┬──────────┬────────────────────────────────────┐
  │ Category       │ Time(us) │ Kernel                             │
  ├────────────────┼──────────┼────────────────────────────────────┤
  │ normalization  │     3.8  │ _rms_normalize_kernel              │
  │ embedding      │     4.4  │ apply_rotary_emb_triton_kernel     │
  └────────────────┴──────────┴────────────────────────────────────┘
                                                       Total: 8.2 us

  After (M kernels, Y us):
  ┌────────────────┬──────────┬────────────────────────────────────┐
  │ Category       │ Time(us) │ Kernel                             │
  ├────────────────┼──────────┼────────────────────────────────────┤
  │ embedding      │    ~4.5  │ fused_norm_rope_inplace_triton     │
  └────────────────┴──────────┴────────────────────────────────────┘
                                                       Total: ~4.5 us

  Savings: ~3.7 us (2 kernels → 1 kernel)
  Risk:   Low — no new kernel code, just dispatch/ordering change
```

**Where to look for existing fused variants:**
- aiter ops: `/sgl-workspace/aiter/aiter/ops/` — check if a fused op already exists
  - `gated_rmsnorm_fp8_group_quant.py` — fused norm+quant
  - `fused_qk_norm_rope_cache_quant.py` — fused norm+rope+cache+quant
  - `fused_qk_rmsnorm_group_quant.py` — fused rmsnorm+group_quant
  - `fused_split_gdr_update.py` — fused split+gather+dequant+reduce+update
- aiter op tests: `/sgl-workspace/aiter/op_tests/` — check test files for usage examples

#### Tier 2: Triton kernel optimization (write/modify Triton kernels)

When fusion isn't possible with existing kernels, write a Triton kernel:

- Fuse 2-3 operations into one Triton kernel (e.g., norm + rope + cache_store)
- Optimize existing Triton kernels (tile sizes, memory access patterns, occupancy)
- Port B200's fused logic to a Triton kernel that runs on ROCm

**Format:**
```
Tier 2: Triton kernel — estimated savings: ~X us
  What to fuse: [op1 + op2 + op3]
  Reference:    [B200 kernel name that does this fused]
  Template:     [existing Triton kernel to base on, if any]
  File to create/modify: [path]

  Before (N kernels, X us):
  ┌────────────────┬──────────┬────────────────────────────────────┐
  │ Category       │ Time(us) │ Kernel                             │
  ├────────────────┼──────────┼────────────────────────────────────┤
  │ quantization   │     4.3  │ dynamic_per_group_scaled_quant     │
  │ gemm           │    19.5  │ ck_gemm_xdl_cshuffle_v3            │
  └────────────────┴──────────┴────────────────────────────────────┘
                                                       Total: 23.8 us

  After (M kernels, Y us):
  ┌────────────────┬──────────┬────────────────────────────────────┐
  │ Category       │ Time(us) │ Kernel                             │
  ├────────────────┼──────────┼────────────────────────────────────┤
  │ quantization   │   ~12.0  │ fused_quant_gemm_triton            │
  └────────────────┴──────────┴────────────────────────────────────┘
                                                       Total: ~12.0 us

  Savings: ~11.8 us (2 kernels → 1 kernel)
  Risk:   Medium — new Triton code, needs testing on ROCm
```

**Where to look for existing Triton kernels:**
- SGLang Triton ops: `sglang/python/sglang/srt/layers/attention/triton_ops/`
- aiter Triton ops: `/sgl-workspace/aiter/aiter/ops/triton/`

#### Tier 3: aiter C++/HIP kernel implementation / optimization

When Triton isn't sufficient (need CK tile, assembly, or low-level HIP):

- Tune CK gemm tile configurations for specific shapes
- Implement new fused HIP kernels in aiter
- Optimize existing aiter kernels (occupancy, memory coalescing, wavefront utilization)

**Format:**
```
Tier 3: aiter kernel — estimated savings: ~X us
  Target kernel: [aiter kernel name]
  Root cause:    [why it's slow — tile size, occupancy, memory pattern]
  Files to check:
    - /sgl-workspace/aiter/aiter/ops/[...].py        (Python binding)
    - /sgl-workspace/aiter/aiter/jit/module_[...].so  (compiled kernel)
    - /sgl-workspace/aiter/csrc/[...]/               (C++/HIP source)
    - /sgl-workspace/aiter/op_tests/test_[...].py    (unit test)

  Before (N kernels, X us):
  ┌────────────────┬──────────┬────────────────────────────────────┐
  │ Category       │ Time(us) │ Kernel                             │
  ├────────────────┼──────────┼────────────────────────────────────┤
  │ gemm           │    21.7  │ moe_gemm1_0                        │
  │ gemm           │    10.8  │ moe_gemm2_0                        │
  └────────────────┴──────────┴────────────────────────────────────┘
                                                       Total: 32.5 us

  After (target):
  ┌────────────────┬──────────┬────────────────────────────────────┐
  │ Category       │ Time(us) │ Kernel                             │
  ├────────────────┼──────────┼────────────────────────────────────┤
  │ gemm           │   ~15.0  │ moe_gemm1_0 (tuned CK tiles)       │
  │ gemm           │    ~8.0  │ moe_gemm2_0 (tuned CK tiles)       │
  └────────────────┴──────────┴────────────────────────────────────┘
                                                       Total: ~23.0 us

  Savings: ~9.5 us
  Risk:   High — requires CK/HIP expertise, longer dev cycle
```

### 7e. aiter library reference

When tracing into aiter kernels, use this directory structure:

```
/sgl-workspace/aiter/
├── aiter/
│   ├── ops/                    # Python op wrappers (entry points)
│   │   ├── gemm_op_a8w8.py     # FP8 GEMM (a8w8 blockscale)
│   │   ├── gemm_op_a16w16.py   # BF16/FP16 GEMM
│   │   ├── norm.py             # LayerNorm/RMSNorm
│   │   ├── rmsnorm.py          # RMSNorm variants
│   │   ├── quant.py            # Quantization ops
│   │   ├── mhc.py              # Multi-head compressor
│   │   ├── mha.py              # Multi-head attention
│   │   ├── attention.py        # Attention variants
│   │   ├── rope.py             # Rotary position encoding
│   │   ├── pos_encoding.py     # Position encoding
│   │   ├── activation.py       # Activation functions (SiLU, GELU, etc.)
│   │   ├── moe_op.py           # MoE operations
│   │   ├── moe_sorting.py      # MoE token sorting
│   │   ├── moe_sorting_opus.py # MoE sorting (Opus variant)
│   │   ├── topk.py             # TopK operations
│   │   ├── cache.py            # KV cache operations
│   │   ├── communication.py    # AllReduce / collective ops
│   │   ├── custom_all_reduce.py# Custom all-reduce
│   │   ├── quick_all_reduce.py # Quick all-reduce
│   │   ├── deepgemm.py         # DeepGemm wrapper
│   │   ├── gated_rmsnorm_fp8_group_quant.py        # FUSED: rmsnorm + fp8 group quant
│   │   ├── fused_qk_norm_rope_cache_quant.py       # FUSED: qk_norm + rope + cache + quant
│   │   ├── fused_qk_rmsnorm_group_quant.py         # FUSED: qk rmsnorm + group quant
│   │   ├── fused_split_gdr_update.py               # FUSED: split + gather_dequant_reduce + update
│   │   ├── fused_qk_norm_mrope_cache_quant.py      # FUSED: qk_norm + mrope + cache + quant
│   │   └── triton/             # Triton kernel implementations
│   ├── jit/                    # Pre-compiled .so kernels
│   │   ├── module_gemm_a8w8_blockscale.so
│   │   ├── module_gemm_a8w8_blockscale_bpreshuffle.so
│   │   ├── module_mhc.so
│   │   ├── module_rmsnorm.so
│   │   ├── module_rmsnorm_quant.so
│   │   ├── module_moe_sorting_opus.so
│   │   ├── module_mla_asm.so
│   │   └── ...
│   ├── mla.py                  # MLA (Multi-Latent Attention) top-level
│   ├── fused_moe.py            # FusedMoE top-level
│   ├── paged_attn.py           # Paged attention
│   ├── rotary_embedding.py     # Rotary embedding
│   └── tuned_gemm.py           # Tuned GEMM (hipBLAS/rocBLAS)
├── csrc/                       # C++/HIP kernel source
│   ├── ck_gemm_a8w8_blockscale/     # CK tile GEMM FP8 blockscale
│   ├── ck_gemm_a8w8_blockscale_bpreshuffle/  # CK tile GEMM with B pre-shuffle
│   ├── ck_tile_gemm_moe_2stages/    # CK tile MoE 2-stage GEMM
│   ├── kernels/                     # Custom HIP kernels
│   └── cpp_itfs/                    # C++ interfaces
├── op_tests/                   # Unit tests for each op
│   ├── test_gemm_a4w4.py
│   ├── test_mla.py
│   ├── test_mla_sparse.py
│   ├── test_rmsnorm2d.py
│   ├── test_rope.py
│   ├── test_moe_sorting.py
│   ├── test_fused_qk_rmsnorm_per_token_quant.py
│   └── ...
└── gradlib/                    # Tuning library
```

### 7f. Show the full implementation roadmap

After the tier analysis, print a summary roadmap:

```
Implementation Roadmap for [Block Name]:
  Current: MI355 = X us (N kernels) | B200 = Y us (M kernels)
  Target:  MI355 ≈ Y us

  Phase 1 (code refactor, ~1 day):
    - Switch from ck_gemm_xdl to fp8gemm_bf16_blockscale in dispatch
    - Remove redundant elementwise cast before gemm
    - Expected: X us → Z us

  Phase 2 (Triton kernel, ~3 days):
    - Write fused quant+gemm Triton kernel
    - Based on existing _fused_rms_fp8_group_quant_kernel pattern
    - Expected: Z us → W us

  Phase 3 (aiter kernel, ~1-2 weeks):
    - Tune CK tile sizes for this specific shape (M=..., N=..., K=...)
    - Or implement assembly-level fused kernel
    - Expected: W us → ~Y us (B200 parity)
```

---

## Step 8: Upstream & Jira reference finder (research)

> **PIPELINE_MODE**: Skip this step (same as Step 7).

Before finalizing the Tier 1/2/3 plan, **check whether the fix already exists upstream or
is in-flight in Jira** — cherry-picking beats reinventing. This is the `research` half of
the skill: it turns the slow kernel from Step 4/5 into a search across the sources
registered in [`_shared/data-sources.md`](../_shared/data-sources.md).

**Trigger it when** you have a concrete slow kernel/op (after Step 5), or directly via
`/compare-kernels --refs <keyword>`.

### 8a. Run the finder

Pick the keyword from the top absolute-gap op — the **kernel family name**, not the full
symbol (e.g. `ck_gemm`, `moe_sorting`, `fused_moe`, `rmsnorm`, `mla`), optionally the
budget category:

```bash
python3 ~/agent-box/skills/compare-kernels/scripts/find_references.py <keyword> [--category <cat>] [--json]
```

It queries three sources, each with graceful degradation (no MCP required today):

| source | preferred | fallback | last resort |
|--------|-----------|----------|-------------|
| `aiter-upstream` | `gh search prs ROCm/aiter` | GitHub REST search (+`GITHUB_TOKEN`) → local `git log`/grep on `/sgl-workspace/aiter` | GitHub search URL |
| `sglang-upstream` | `gh search prs sgl-project/sglang` | GitHub REST search → local `git log`/grep on `$SGLANG_ROOT` | GitHub search URL |
| `jira-amd` | atlassian-mcp *(planned)* | REST via `JIRA_EMAIL`+`JIRA_API_TOKEN` | JQL browse URL |

The `[live:...]` / `[fallback:...]` tag on each section tells you how the hits were
obtained, so you know when a section is empty because there's nothing vs because a tool
was missing.

### 8b. Read the results into the plan

- **Upstream PR that already implements the fusion/tune** → the optimization becomes a
  **cherry-pick / rebase**, not new code. Note the PR number in the Tier plan and drop the
  tier by one (a merged upstream PR is Tier 1 even if it's C++).
- **Local-checkout commit/file hits** → the kernel already exists in the installed aiter;
  the gap may be a **dispatch/wiring** issue (Tier 1), not a missing kernel. Cross-check
  with [[../../memory/workflows/sglang-integration]] (`_is_hip` vs `_use_aiter`).
- **Jira ticket touching the kernel** → someone owns related work; **coordinate / follow
  up** instead of duplicating. Feed transferable tickets through
  [`/qwen35-jira-track`](../qwen35-jira-track/SKILL.md) for full triage.
- **Nothing anywhere** → genuinely new work; proceed with the Tier plan as written.

### 8c. Enabling the live paths / MCP

- `GITHUB_TOKEN` → higher GitHub search rate limit (REST search is throttled hard when
  anonymous). `gh auth login` (if `gh` is installed) is preferred over REST.
- `JIRA_EMAIL` + `JIRA_API_TOKEN` (same vars as `/qwen35-jira-track`) → live Jira search;
  without them you get the JQL URL to open in a browser.
- When a **GitHub MCP** or **Atlassian MCP** is registered, flip the matching row in
  [`_shared/data-sources.md`](../_shared/data-sources.md) to `status: live` and update the
  `MCP_HINTS` call in `scripts/find_references.py` to call the MCP tool first (the REST/gh
  path stays as the offline fallback).

---

## PIPELINE_MODE contract (from `/kernel-fusion-pipeline`)

When invoked with `pipeline_mode: true`:

- **Stop after Step 6** (category mismatches). Do NOT enter Step 7 (deep-dive / AskUserQuestion) or Step 8 (reference finder).
- Return the ranked Tier-1 fusion list directly to the pipeline.
- Do **not** use `AskUserQuestion` for any reason.
- The pipeline only needs: block name, slug, target op, kernels, savings, sglang file.

---

## Important Notes

- **Don't just say "quantization is slow"** — break it down by module. Is it Q-proj quant? KV-proj quant? MoE quant? Each has different dispatch paths and optimization strategies.
- **Identify fusion opportunities explicitly** — when B200 uses N kernels and MI355 uses M>N kernels for the same module, that's a fusion gap. Name which kernels could be combined.
- **Name specific kernels and their durations** — not just categories. `ck_gemm_xdl (74.6 us) vs deep_gemm (16.3 us)` is actionable; `gemm is slower` is not.
- **Include the source file paths** so the user knows exactly where to look for the dispatch logic.
- **For decode traces** (CUDA graph replayed layers without Module column data), fall back to the sequential kernel-by-kernel comparison from the original approach.
- **Decode vs prefill compressor paths differ significantly on HIP**. In decode, the compressor uses fused kernels (`_c4_decode_kernel`, `_compress_norm_rope_kernel`, `fp8_paged_mqa_logits_kernel`, `_triton_fused_store_indexer_kernel`, `_triton_fused_store_flashmla_kernel`) — there is NO separate `cunn_SpatialSoftMaxForward`, elementwise mul, or reduce/sum kernel. The unfused `softmax + mul + sum` path (from `compress_decode_old` / the generic Python compressor) only appears in prefill/extend traces or when fused paths are disabled. When analyzing decode traces and seeing no softmax/mul/sum in the compressor region, this is expected — don't flag it as missing.
- **Always check for existing fused aiter ops first** before proposing new kernels. The `/sgl-workspace/aiter/aiter/ops/` directory has many fused variants that may already solve the problem.
- **Priority order is strict**: code refactor → Triton → aiter C++/HIP. Never propose Tier 3 if Tier 1 or 2 can achieve the same savings.
