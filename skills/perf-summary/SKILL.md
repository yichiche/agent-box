---
name: perf-summary
description: "Summarize kernel performance from difference graphs and Excel data. Produces a 2-3 sentence status and bullet-point action items for kernels to optimize. Use when the user says '/perf-summary' or provides kernel difference graphs/Excel files for performance analysis."
---

# Performance Summary

## When To Use

- User provides kernel difference graphs (images) and/or Excel files comparing kernel timings across platforms or runs
- User asks to summarize current optimization status
- User wants to know which kernels to optimize next

## Input

The user will provide one or more of:

- **Kernel difference graph** (screenshot/image): bar chart or waterfall showing time delta per kernel category between two platforms (e.g., B200 vs MI355, before vs after)
- **Excel file** (`.xlsx`): kernel breakdown with columns like kernel name/category, time, percentage, platform comparison

Read all provided inputs before producing the summary.

## Output Format

### Status

2-3 sentences summarizing the overall performance gap, the dominant contributors, and any notable structural differences (e.g., kernel overlap, stream concurrency). The first sentence states the headline gap. The following sentences call out the top 2-3 contributors with numbers and any cross-cutting issues.

**Performance ratio format:** Express the headline gap as "X% of [baseline] performance" (e.g., "56% of B200 performance"), where 100% = parity. Never use "Nx slower" or "Nx faster" — those are ambiguous about which direction is better.

Example:
> MI355 decode is at 56% of B200 performance (350.9 us vs 197 us E2E), with GEMM and elementwise kernels accounting for 67% of the kernel-time gap. An additional ~60 us penalty comes from MI355's near-zero kernel overlap compared to B200's 56 us of overlap, amplifying the raw kernel gap into a larger E2E difference. Quantization and MoE fused kernels are at parity or faster on MI355, partially offsetting the gap.

### Action Items

Bullet-point list of kernels to optimize, ordered by impact (largest gap first). Each bullet should include:

- **Kernel category or name**
- **Gap magnitude** (absolute time or percentage of total gap)
- **Brief root cause or optimization direction** if inferable from the data

Example:
> - **FlashAttention decode**: +1.8ms (34% of gap) — explore FA3 or custom decode attention kernel for MI355
> - **FP8 GEMM (MoE)**: +1.2ms (23% of gap) — blocked on Triton tuning configs for MI355 block shape
> - **AllReduce**: +0.6ms (11% of gap) — current custom allreduce not optimized for MI355 topology

### Rules

1. **Status must be 2-3 sentences.** No preamble, no context paragraph. First sentence is the headline gap; subsequent sentences highlight top contributors and structural issues.
2. **Action items are bullet points only.** No sub-bullets, no paragraphs.
3. **Order by impact.** Largest time gap or percentage first.
4. **Be specific.** Use kernel names from the data, not generic categories.
5. **Include numbers.** Every action item must have a quantified gap.
6. **Skip kernels with <1% of total gap** unless the user asks for a full breakdown.
7. **If a kernel is already at parity or faster**, mention it briefly at the end as "at parity" — do not create action items for it.

## Workflow

1. Read the provided image(s) — extract kernel names, time differences, and relative contributions from the graph.
2. Read the provided Excel file(s) — parse kernel categories, per-kernel times, platform columns, and compute deltas.
3. Cross-reference graph and Excel data to validate numbers.
4. Rank kernels by gap magnitude.
5. Produce the 2-3 sentence status.
6. Produce the ordered action-item bullets.
7. If context from prior profiling sessions or known optimization work is available in memory, annotate action items with what's already in progress or blocked.
