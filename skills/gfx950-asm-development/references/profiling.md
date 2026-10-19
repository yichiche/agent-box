# Kernel profiling with rocprofv3

Kernel traces identify expensive dispatches; performance monitoring counters (PMC) provide hardware statistics for investigating their cost. Use the same kernel entry point and representative input as the ordinary benchmark. Keep baseline and candidate profiles separate. Check `rocprofv3 --help` for the installed version before using optional flags.

## Locate the hotspot

Create a fresh output directory for each collection, then substitute the project's actual benchmark command for `./bench`:

```bash
mkdir -p profile
ASM_PROFILE_DIR="$(mktemp -d profile/asm-XXXXXX)"
rocprofv3 --kernel-trace --stats -f csv \
    -d "$ASM_PROFILE_DIR/trace" -- ./bench
```

Find the emitted `*_kernel_stats.csv`. Run the helper from this skill directory, passing one summary file from one run/device:

```bash
python3 scripts/kernel_summary.py PATH_TO_KERNEL_STATS_CSV --top 10
```

The helper requires `Name`, `Calls`, and `TotalDurationNs`. Inspect a different schema explicitly before adapting it; do not silently guess units or mix files from different experiments.

Use total duration to identify expensive kernels and mean duration to assess an individual invocation. A large total may reflect many calls. Summed kernel durations describe accumulated GPU work and need not equal application wall time when work overlaps.

## Test one hypothesis with counters

Enumerate counters first:

```bash
rocprofv3 --list-avail
```

Select counters that test the hypothesis and filter the exact target kernel. For example, if the following cache counters are available:

```bash
ASM_KERNEL_REGEX='YOUR_KERNEL_REGEX'
rocprofv3 --pmc TCC_HIT TCC_MISS \
    --kernel-include-regex "$ASM_KERNEL_REGEX" \
    -f csv -d "$ASM_PROFILE_DIR/pmc" -- ./bench
```

Inspect the counter CSV schema. Some versions use one row per dispatch/counter, so values must be grouped by dispatch before ratios are computed. Keep the input, iteration range, and launch path consistent across comparisons.

Combine the observation with the generated ISA and an experiment. For example, cache misses alone do not establish a bandwidth bottleneck; a claim about memory cost also needs evidence about traffic, stalls, or the effect of a controlled change. Treat occupancy and utilization as clues rather than optimization goals in isolation.

Use the ordinary benchmark to confirm the effect of each change. Profiling can alter timing, so report performance from runs with the same instrumentation settings and keep profiler results as supporting evidence.

## Record the result

Keep the collection command, input case, tool version, kernel identity, and source revision beside the outputs. Report the hotspot, the actual values supporting a hypothesis, the proposed change, and its subsequent correctness and timing result.
