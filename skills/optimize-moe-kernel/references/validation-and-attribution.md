# Validation and Attribution

Use this reference before timing, integration, or a performance claim.

## Evidence levels

| Level | Establishes | Does not establish |
|---|---|---|
| Selector/structural check | intended shapes reach candidate | kernel correctness or speed |
| Compile/resource inspection | code lowers and fits | runtime correctness |
| Operator correctness | tensor contract on tested cases | serving performance |
| Operator microbenchmark | stage/op behavior in isolation | e2e win |
| Served trace | candidate executes and changes hot-path time | stable user-visible win |
| Matched e2e A/B | system effect for that workload | other dtypes/shapes or accuracy |
| Task accuracy | quality for that eval distribution | performance |

Do not promote evidence beyond its level.

## Baseline identity

Record before each run:

- Application, kernel-library, and compiler commits.
- Installed source hashes when overlays or editable installs are involved.
- Selector output for every measured shape.
- Model/checkpoint and quantization format.
- Hardware, device count, topology, clocks/power state when available.
- Environment variables, backend choices, graph mode, attention backend, and cache
  directory.
- Exact benchmark command, seed, warmup, repetitions, request counts, and output
  path.

Do not mix a pre-pipeline baseline, later installed kernel, and current selector in
one table without labeling them as different implementations.

## Correctness matrix

Cover the meaningful cross product without mechanically exploding it:

- Dtypes and scale dtypes.
- `M` at selector boundaries and neighbors.
- Local routing: none, sparse, dense, repeated expert, and all local.
- Shapes with multiple M/N blocks and scale-block edges.
- Norm on/off, optional affine scale, routing-weight reduction.
- Graph capture/replay and dynamic routing contents.
- Empty input and all-remote output.
- Random scales, zeros, finite extremes, NaN/Inf behavior when specified.

Use an independent reference for new numerical paths. Comparing two wrappers that
share the same buggy dequantizer is insufficient.

For split-K, test the nonlinearity and rounding contract specifically. Gate/up must
reduce partials before activation. Down must round at the reference's expert-output
boundary before normalization when that is the production behavior.

## Microbenchmark design

Use graph replay when production uses graphs. Warm compilation and allocations
outside the timed region. Report median plus a tail statistic and retain raw samples
when the expected change is small.

Measure fixed routing distributions and captured production routes. For expert
parallel serving, rank maximum or critical-path time matters more than the average
rank. Include all-remote to measure fixed overhead and all-local to expose maximum
expert work.

Time stages independently as well as the full MoE op. A full-op regression can hide
a faster GEMM plus slower reduction; a stage-only win can hide an extra launch that
erases it.

## Matched end-to-end A/B

Baseline and candidate must use identical model, workload, requests, concurrency,
warmup, backend settings, attention path, graph mode, and server lifecycle. Start a
fresh server for each backend and save to a new directory.

Validate every JSON before comparison:

- File is parseable and non-zero.
- Completed requests equal requested requests.
- Input/output token totals match between A and B.
- Model and concurrency metadata match.
- No server restart, compilation, or error occurred in the measured window.

Report output and total throughput, duration, completed requests, and mean/median/p99
for TTFT, TPOT/ITL, and E2E when available. Put raw paths next to the claim.

Repeat small deltas. A single `+0.6%` run is a pass for continued evaluation, not a
universal stable win. Compare run-to-run variance and keep the same request set.

## Accuracy escalation

Tensor allclose is normally sufficient when the implementation preserves dtype,
scale semantics, accumulation, and observable rounding points within the existing
contract.

Run task-level accuracy when any of these change:

- Activation or weight quantization.
- Scale calculation or granularity.
- Saturation, clipping, or zero-point semantics.
- An observable reduced-precision rounding boundary.
- Approximate activation or normalization math outside accepted tolerance.

Do not relax an accuracy threshold to ship a known regression.

## GPU and process isolation

Check device ownership before launching. Zero VRAM in the current namespace is not
proof that the host is idle; consult the scheduler or host/container mapping when
available.

Launch the server in its own process group, record its PID, and terminate only that
group. Avoid broad `pkill -f sglang`, `pkill -f python`, shared compile-cache deletion,
or replacing installed kernels while another benchmark runs. Use unique ports,
result directories, and compile caches.

If another workload appears during final checks, identify it read-only and leave it
alone unless the user explicitly owns and authorizes its termination.

## Overlay and source parity

When shipping kernel files as an overlay patch:

1. Generate the patch from the exact source tested.
2. Apply it to a clean checkout of the pinned base.
3. Compare hashes of reconstructed and installed sources.
4. Run syntax/compile checks and meaningful GPU correctness tests.
5. Keep source hash and result paths in the benchmark manifest.

An overlay that applies cleanly but differs from the installed benchmark source does
not reproduce the result.

## Attribution language

State the selector predicate and affected rows before the numbers. Examples:

- "This changes one dtype only for M>128; lower decode concurrency is unchanged."
- "The split-K branch applies below a routed-row threshold; larger M uses the prior path."
- "The full-stack gain includes attention changes and cannot be assigned to MoE."

Keep operator, component-stack, and full-stack comparisons separate. When combining
tables from different run dates or source revisions, label the cross-run comparison
as contextual rather than causal.

## Final artifact

Leave a reviewable package containing code/patch, selector tests, operator results,
matched e2e JSON, source hashes, benchmark commands, rejected experiments, and a
short explanation of why the winning mechanism matches the profile.
