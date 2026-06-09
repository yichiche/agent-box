---
description: "End-to-end kernel fusion pipeline: compare two trace analysis Excel files, identify Tier-1 fusion opportunities (existing aiter fused ops), implement each fusion on the SGLang side, validate (accuracy + profiling + benchmark), and commit. Composes /compare-kernels, /implement-kernel, /validate, and /commit. Use when the user says '/kernel-fusion-pipeline' followed by two xlsx paths."
---

# Kernel Fusion Pipeline

Automated pipeline that chains trace comparison → fusion identification → implementation → validation → commit. Focuses on **Tier 1 fusions only** — switching SGLang dispatch to existing fused aiter ops. Each fusion is implemented, validated, and committed individually (sequential, git-bisectable).

## Usage

```
/kernel-fusion-pipeline <file_A.xlsx> <file_B.xlsx>
```

If the user doesn't provide two xlsx paths, ask for them.

---

## Step 0: Upfront Context Gathering

**Gather ALL operational details before any analysis or implementation.** This prevents mid-pipeline interruptions.

### 0a: Detect the active SGLang installation

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
echo "Active SGLang root: $SGLANG_ROOT"
```

Use `$SGLANG_ROOT` for all subsequent paths.

### 0b: Parse the two xlsx paths

From `$ARGUMENTS`, extract two file paths. Verify both exist:

```bash
ls -la <file_A.xlsx> <file_B.xlsx>
```

If the user didn't provide two paths, ask:

```
AskUserQuestion: "Please provide two trace analysis Excel files to compare (e.g., B200 baseline vs MI355)."
```

### 0c: Collect validation parameters

Use `AskUserQuestion` with **all four questions** to gather everything needed for later `/validate` and `/implement-kernel` steps:

#### Question 1: Server launch script

```
"Which server launch script should be used for validation?"
```

Options:
- **Provide path** — the user gives a path like `~/run_dsv4.sh`

Parse the server script to extract:
- **Port**: look for `--port <N>` in the launch command
- **Model path**: look for `--model-path <path>`

#### Question 2: Client benchmark script

```
"Which client benchmark script should be used for validation?"
```

Options:
- **Provide path** — the user gives a path like `~/run_dsv4_client.sh`

#### Question 3: Accuracy threshold

```
"What accuracy threshold should be used for GSM8K validation?"
```

Options:
- **0.88 (DSv4 default)** (Recommended)
- **Custom threshold** — user provides a number

#### Question 4: Model label

```
"Short label for this model (used in reports)?"
```

Options:
- **DSv4** (Recommended)
- **Custom label**

### 0d: Confirm all parameters

Print the complete configuration and confirm with `AskUserQuestion`:

```
Kernel Fusion Pipeline Configuration:
  SGLang Root:      <SGLANG_ROOT>
  File A:           <file_A.xlsx>
  File B:           <file_B.xlsx>
  Server Script:    <server_script>
  Port:             <PORT>
  Model:            <model_path>
  Client Script:    <client_script>
  Accuracy Threshold: <threshold>
  Label:            <label>

Proceed with this configuration?
```

Options:
- **Yes, proceed** (Recommended)
- **Change something** — go back and re-ask

**Do NOT proceed past Step 0 until all parameters are confirmed.**

---

## Step 1: Compare Kernels

Invoke the `/compare-kernels` analysis on the two xlsx files. This step follows the full `/compare-kernels` workflow:

### 1a: Recategorize both files

```bash
python3 $HOME/agent-box/profile/trace_module_analyzer.py --recategorize <file_A.xlsx> <file_B.xlsx>
```

### 1b: Run the full comparison

Follow the `/compare-kernels` skill Steps 1-6:

1. Read detail sheets from both files (using openpyxl)
2. Re-classify every kernel against current `kernel_categories.csv`
3. Group kernels by Module and produce side-by-side comparison
4. Print module-grouped comparison with FUSION/IMPROVE/MISSING annotations
5. Print module-grouped time totals
6. Generate actionable optimization recommendations
7. Flag category mismatches

**Instead of Step 7 (deep-dive)**, this skill proceeds to extract fusion opportunities.

### 1c: Extract Tier 1 fusion opportunities

From the comparison output, identify all blocks where:

1. **File A (typically B200) uses fewer kernels** than File B (typically MI355) for the same module
2. **An existing fused aiter op** can replace the separate MI355 kernels
3. **The change is Python-level only** — switching dispatch, removing redundant ops, enabling existing fused paths

For each opportunity, record:
- **Block name** (e.g., "Q Proj", "Attn Prep", "Gate Norm")
- **Current MI355 kernels**: list of kernel names and durations
- **Target fused op**: the aiter fused op that replaces them (from the aiter ops directory)
- **Estimated savings**: sum of replaced kernel times minus estimated fused kernel time
- **SGLang file to modify**: from the source file reference table

### 1d: Verify aiter fused ops exist

For each candidate fusion, verify the target aiter op actually exists:

```bash
ls /sgl-workspace/aiter/aiter/ops/<fused_op_name>.py 2>/dev/null
```

Check the op's function signature to confirm it can replace the separate kernels:

```bash
grep -n "^def \|^class " /sgl-workspace/aiter/aiter/ops/<fused_op_name>.py
```

Also check for usage examples in aiter tests:

```bash
ls /sgl-workspace/aiter/op_tests/test_*<keyword>*.py 2>/dev/null
```

**Drop any candidate whose target aiter op doesn't exist or whose signature doesn't match.**

### Aiter fused ops reference

These are the known fused ops to check against:

| Fused Op | File | Replaces |
|---|---|---|
| gated_rmsnorm_fp8_group_quant | `aiter/ops/gated_rmsnorm_fp8_group_quant.py` | RMSNorm + FP8 group quantization |
| fused_qk_norm_rope_cache_quant | `aiter/ops/fused_qk_norm_rope_cache_quant.py` | QK norm + RoPE + cache store + quantization |
| fused_qk_rmsnorm_group_quant | `aiter/ops/fused_qk_rmsnorm_group_quant.py` | QK RMSNorm + group quantization |
| fused_split_gdr_update | `aiter/ops/fused_split_gdr_update.py` | Split + gather_dequant_reduce + update |
| fused_qk_norm_mrope_cache_quant | `aiter/ops/fused_qk_norm_mrope_cache_quant.py` | QK norm + mRoPE + cache + quantization |

**Always re-scan** `/sgl-workspace/aiter/aiter/ops/fused*.py` and `/sgl-workspace/aiter/aiter/ops/gated*.py` in case new fused ops have been added since this skill was written.

---

## Step 2: Present Opportunities and Select

### 2a: Print the ranked fusion list

Show the user all Tier 1 fusion opportunities, ordered by estimated savings (largest first):

```
=== Tier 1 Kernel Fusion Opportunities ===

#  Block              Current (MI355)           Target (aiter fused)           Est. Savings
1. Gate Norm           rmsnorm (3.8 us)         gated_rmsnorm_fp8_group_quant   ~3.8 us/layer
                      + fp8_quant (4.2 us)      (fused norm+quant)
                      = 8.0 us, 2 kernels       → 1 kernel, ~4.2 us

2. Attn Prep           rmsnorm (3.5 us)         fused_qk_norm_rope_cache_quant  ~6.0 us/layer
                      + rope x2 (8.8 us)        (fused norm+rope+cache+quant)
                      + cache_store (2.1 us)
                      = 14.4 us, 4 kernels      → 1 kernel, ~8.4 us

3. ...

Total estimated savings: ~X us/layer × N layers = ~Y us/iteration
```

### 2b: Ask user to select

Use `AskUserQuestion` with multi-select:

```
"Which kernel fusions should be implemented? (Select all that apply)"
```

Present each fusion as an option with its block name and estimated savings. Include an "All of the above" option as the first choice.

**If no Tier 1 opportunities were found**, report this to the user:

```
No Tier 1 (Python-level) fusion opportunities found.
All identified gaps require Tier 2 (Triton) or Tier 3 (aiter C++/HIP) changes.

Tier 2/3 opportunities for future work:
  - <list from compare-kernels output>
```

And stop the pipeline.

---

## Step 3: Sequential Implement → Validate → Commit Loop

For each selected fusion, execute the following sub-steps **sequentially**. Do NOT batch — each fusion is independently implemented, validated, and committed before moving to the next.

### 3a: Implement the fusion

Invoke `/implement-kernel` by calling:

```
Skill(skill="implement-kernel", args="Tier 1: Switch <block_name> to use <aiter_fused_op> — replace <N> separate kernels (<kernel_list>) with single fused call. SGLang file: <file_path>. Estimated savings: ~<X> us/layer.")
```

The `/implement-kernel` skill will:
1. Enter plan mode and read the CUDA vs HIP code paths
2. Check for the aiter fused op (which we already verified exists)
3. Design the dispatch change
4. Get user approval on the plan
5. Implement the code change
6. Run syntax and lint checks

**Wait for `/implement-kernel` to complete before proceeding.**

If `/implement-kernel` fails (syntax error, lint error, user rejects plan), report the failure and ask:

```
AskUserQuestion: "Fusion #N (<block_name>) failed during implementation: <error>.
How should we proceed?"
```

Options:
- **Skip this fusion and continue** — move to the next one
- **Retry with modifications** — user provides guidance
- **Abort pipeline** — stop all remaining fusions

### 3b: Validate the change

Invoke `/validate` by calling:

```
Skill(skill="validate")
```

The `/validate` skill will ask for server script, client script, etc. Since the user already provided these in Step 0, provide the answers immediately:

- Server script: `<server_script>` (from Step 0c)
- Client script: `<client_script>` (from Step 0c)
- Model name: `<label>` (from Step 0c)
- Accuracy threshold: `<threshold>` (from Step 0c)

**Wait for `/validate` to complete.**

### 3c: Interpret validation results

Check the validation output for:

1. **Accuracy**: Did GSM8K pass the threshold?
2. **Profiling**: Are the old kernels gone and new fused kernel visible?
3. **Benchmark**: Is throughput maintained or improved? Is latency not regressed?

**If validation PASSES:**

Print:
```
✓ Fusion #N (<block_name>) PASSED validation
  Accuracy:  <score> (threshold: <threshold>)
  Profiling: Old kernels removed, fused kernel visible
  Benchmark: <throughput> tok/s, ITL=<X>ms
```

Proceed to commit.

**If validation FAILS:**

Print:
```
✗ Fusion #N (<block_name>) FAILED validation
  Failed step: <accuracy/profiling/benchmark>
  Error: <details>
```

Ask the user:

```
AskUserQuestion: "Fusion #N failed validation. How should we proceed?"
```

Options:
- **Debug and fix** — investigate the failure, modify the implementation, re-validate
- **Revert and skip** — undo the change, move to next fusion
- **Abort pipeline** — stop all remaining fusions

If "Debug and fix": read the error output, identify the issue, make fixes, and re-run `/validate`. Allow up to 2 retry attempts before forcing a decision.

If "Revert and skip": revert the changes using `git checkout -- <modified_files>` and move to the next fusion.

### 3d: Commit the change

If validation passed, invoke `/commit`:

```
Skill(skill="commit", args="<block_name>: switch to <aiter_fused_op> — fuse <N> kernels into 1, saves ~<X> us/layer")
```

**Wait for `/commit` to complete before moving to the next fusion.**

### 3e: Record results and continue

Record the result for this fusion:
- Block name
- Fused op used
- Kernels replaced (count and names)
- Savings achieved (from profiling)
- Commit hash
- Pass/fail/skipped status

Move to the next selected fusion and repeat from 3a.

---

## Step 4: Final Summary Report

After all selected fusions have been processed, print the comprehensive summary:

```
=== Kernel Fusion Pipeline Results ===

Model: <label>
SGLang Root: <SGLANG_ROOT>
Comparison: <file_A.xlsx> vs <file_B.xlsx>

### Fusions Implemented

| #  | Block          | Fused Op                         | Kernels Replaced | Savings/Layer | Status  | Commit  |
|----|----------------|----------------------------------|-----------------|---------------|---------|---------|
| 1  | Gate Norm      | gated_rmsnorm_fp8_group_quant     | 2 → 1           | ~3.8 us       | PASS    | abc1234 |
| 2  | Attn Prep      | fused_qk_norm_rope_cache_quant    | 4 → 1           | ~6.0 us       | PASS    | def5678 |
| 3  | ...            | ...                               | ...             | ...           | SKIP    | —       |

### Cumulative Impact

Kernel-level savings:
  - Per-layer: ~<X> us (sum of all passed fusions)
  - Per-iteration (N layers): ~<Y> us
  - Expected ITL improvement: <Y>/<baseline_ITL_us> = <Z>%

E2E benchmark (first baseline vs final after):
  | Metric                   | First Baseline | Final After | Delta   |
  |--------------------------|----------------|-------------|---------|
  | Total throughput (tok/s)  | ...            | ...         | +X.X%   |
  | Median ITL (ms)           | ...            | ...         | -X.X%   |
  | ...                       | ...            | ...         | ...     |

### Remaining Opportunities (Tier 2/3 — not attempted)

These require new Triton kernels or aiter C++/HIP changes:
  - <block>: <description> — estimated <X> us savings
  - ...

### Next Steps

  - Push changes: /commit-push
  - Create PR: /pr
  - Profile the cumulative result: /generate-profile
  - Compare final traces: /compare-kernels <original_MI355.xlsx> <new_after.xlsx>
```

---

## Important Notes

- **Tier 1 only**: This skill only implements fusions using existing aiter fused ops. It does NOT write new Triton or C++/HIP kernels. For Tier 2/3, use `/implement-kernel` directly.
- **Sequential execution**: Each fusion is implemented → validated → committed individually. This ensures every change is independently verified and git-bisectable.
- **Upfront context**: All server/client/accuracy parameters are gathered once in Step 0 and reused throughout the pipeline. No mid-pipeline interruptions.
- **Fail-stop**: If validation fails, the pipeline stops for that fusion and asks the user. It never silently skips or continues past a failure.
- **Backend-gated changes**: All implementations must follow the Backend-Gated Changes rules from `/implement-kernel` — gate behind `_use_aiter`, no top-level aiter imports, common path byte-identical.
- **Dynamic paths**: Always use `$SGLANG_ROOT` detected from the active Python environment. Never hardcode `/sgl-workspace/sglang` or `$HOME/sglang`.
- **Aiter ops may change**: Always re-scan `/sgl-workspace/aiter/aiter/ops/fused*.py` and `gated*.py` at runtime rather than relying on the static reference table. New fused ops may be added.
- **Server polling in foreground**: When `/validate` polls for server readiness, it must run in the foreground (per user feedback memory). Never use background + TaskOutput for health-check polling.
- **Commit conventions**: All commits use the `[AMD]` tag prefix per the repo config. No `Co-Authored-By` trailers.
