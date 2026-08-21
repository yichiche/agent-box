---
name: kernel-profile-triage
description: Triage a torch profiler trace for kernel optimization opportunities — kernel time table, overlap opportunities, fuse patterns. Bridges upstream llm-torch-profiler-analysis with local /parse-trace and /compare-kernels workflow. Use after /generate-profile or when analyzing an existing trace.json.gz.
category: measure
---

# Kernel Profile Triage

Adapted from upstream `llm-torch-profiler-analysis`. Produces actionable optimization hints from a trace, then hands off to `/parse-trace` and `/compare-kernels` for deeper analysis.

## When to use

- Just captured a trace with `/generate-profile`
- Have an existing `*.trace.json.gz` and need to find what to optimize
- Comparing before/after kernel changes — pair with `/compare-kernels`

## Workflow overview

```
trace.json.gz
    ├── kernel-profile-triage  → 3 tables (kernel / overlap / fuse)
    ├── /parse-trace           → Excel module breakdown (decode/prefill)
    └── /compare-kernels       → before/after diff + --budget for decode
```

## Step 1: Run upstream triage (if available)

Check if the unified analyzer exists in the active SGLang install:

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
ANALYZER="$SGLANG_ROOT/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py"
if [ -f "$ANALYZER" ]; then
  python3 "$ANALYZER" triage --trace <trace.json.gz>
else
  echo "Upstream analyzer not found — use Step 2 fallback"
fi
```

If not in local SGLang, fetch from upstream:

```bash
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/main/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py" \
  -o /tmp/analyze_llm_torch_profile.py
python3 /tmp/analyze_llm_torch_profile.py triage --trace <trace.json.gz>
```

### Three output tables

| Table | What it shows | Action |
|-------|--------------|--------|
| **Kernel** | GPU time % by kernel name | Target top contributors |
| **Overlap opportunity** | Kernels that could overlap with others | Scheduling / stream optimization |
| **Fuse pattern** | Source-backed adjacent op pairs worth fusing | `/implement-kernel` Tier 1–2 candidates |

Default cutoff: rows ≥ 1.0% cumulative GPU time. Ask user for lower cutoff if needed.

Fuse-pattern table is **deterministic and source-backed** — not fuzzy matching. Weak matches get a `high`/`medium`/`low` confidence note.

## Step 2: Local fallback — /parse-trace

If upstream analyzer unavailable, use local toolchain:

```bash
/parse-trace <trace.json.gz>
```

Produces `analysis_decode.xlsx` and/or `analysis_prefill.xlsx` with module-grouped kernel breakdown via `trace_module_analyzer.py`.

For decode traces (collapsed to `CudaGraphReplay`), use:

```bash
/compare-kernels --budget before.xlsx after.xlsx
```

## Step 3: Prioritize optimization targets

Rank candidates by:

1. **GPU time %** in triage kernel table (or Excel `Total Time` column)
2. **Regression vs baseline** from `/compare-kernels` if available
3. **Tier feasibility** (prefer Tier 1 dispatch change > Tier 2 Triton > Tier 3 aiter)

Hand off top target to `/implement-kernel <description>`.

## Step 4: Stage-separated capture (prefill vs decode)

For SGLang servers supporting `--profile-by-stage`:

```bash
python3 -m sglang.test.send_one --profile --profile-by-stage
```

Run triage on each stage separately. Decode dominates serving latency for long outputs; prefill dominates TTFT.

## Step 5: Two-trace formal triage

When comparing baseline vs after:

```bash
python3 analyze_llm_torch_profile.py triage --trace baseline.json.gz --trace-after after.json.gz
```

Or locally:

```bash
/compare-kernels baseline_analysis_decode.xlsx after_analysis_decode.xlsx
/compare-kernels --budget baseline_analysis_decode.xlsx after_analysis_decode.xlsx
```

## Integration with existing workflow

| Step | Skill |
|------|-------|
| Capture trace | `/generate-profile` |
| Quick triage | **This skill** |
| Module breakdown | `/parse-trace` |
| Before/after diff | `/compare-kernels` |
| Implement fix | `/implement-kernel` |
| Validate | `/validate-pr` |
| Summarize action items | `/perf-summary` |

Upstream reference: https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-torch-profiler-analysis
