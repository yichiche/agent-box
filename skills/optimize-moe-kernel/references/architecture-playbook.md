# MoE Architecture Optimization Playbook

Use this reference after the hot stages and served shapes are known. The entries
are candidate hypotheses, not universal tuning rules.

## Shape model

Track these dimensions separately:

| Symbol | Meaning | Performance consequence |
|---|---|---|
| `M` | tokens entering the MoE layer | decode parallelism and graph selection |
| `T = M * topk` | logical routed slots | scratch size and alignment input |
| `L_r` | local routed slots on rank `r` | useful expert work; maximum rank often gates latency |
| `E_local` | experts resident on a rank | padding blocks and weight working set |
| `H` | hidden dimension | gate/up K and down N |
| `I` | expert intermediate dimension | gate/up N and down K |
| `BM/BN/BK` | kernel tiles | grid size, reuse, registers, LDS, and loop depth |

Also record aligned capacity, number of non-empty expert blocks, and total grid.
A grid based on capacity may launch many workgroups that only read metadata and
exit. Bound it by proven useful blocks when the alignment contract allows it.

## Phase regimes

Small-M decode often has too little parallelism and serial K latency. Useful
directions are narrower N tiles, split-K, persistent scheduling, bounded grids,
single-program routing, and fused epilogues.

Mid-M decode often becomes a balance between weight streaming, route imbalance,
and activation rereads. Stage-specific BN/BK and occupancy matter more than raw
launch count.

Large-M decode and prefill can reuse weights and amplify intermediate traffic.
Wider tiles, larger BM, fewer redundant activation reads, and conventional
throughput-oriented pipelines become more valuable. Chunked prefill can enter
this regime even when ordinary decode never does.

## Gate/up versus down

Treat the stages independently:

- Gate/up reads the original activation, produces gate and up values, and must
  preserve `activation(sum(partials_gate)) * sum(partials_up)` when K is split.
  Applying the nonlinearity to each K partial is wrong.
- Down reads the activated intermediate. Wider N tiles can reduce intermediate
  rereads, particularly as local routed rows grow.
- Gate/up carries more accumulator fragments; its best unroll and BN can be
  smaller than down's.
- A combined config dictionary should support stage-prefixed overrides instead
  of forcing both stages into one compromise.

## Split-K decision

Consider split-K when the unsplit kernel has too few resident workgroups and a
long serial K loop. Quantify the extra cost first:

- FP32 partial scratch bytes.
- Additional GEMM workgroups.
- Gate/up reduction plus activation launch.
- Down reduction or split-aware finalize cost.
- Cache and bandwidth pressure from scratch.

For split factor `S`, decompose the program id into GEMM tile and split id, give
each split an equal K range, and store independent partials. Ensure K is divisible
by the split and instruction tile. Retune BN and K unroll after splitting: the
best unsplit tile is frequently wrong once each workgroup sees `K/S`.

Preserve the reference's rounding boundary. If the ordinary path produces BF16
expert outputs, sum down partials in FP32 and round to BF16 at the same logical
point before normalization or routing-weight accumulation.

Split-K is less attractive when unpack/scale work is coupled to every partial,
scratch dominates, or the original kernel already fills the device. Test it by
dtype rather than enabling it globally.

## Parallelism and tile levers

| Symptom | Candidate experiments | Evidence to inspect |
|---|---|---|
| Too few workgroups | smaller BN/BM, split-K, persistent grid | CU activity, waves, grid/useful blocks |
| Serial K stalls | K split, software pipeline, more waves | MFMA dependency gaps, VMEM latency |
| Activation rereads | wider BN, stage-specific down tile | bytes and cache counters |
| Weight stream dominates | wider N tile, vectorized loads, cache policy | achieved bandwidth and transactions |
| Register pressure | smaller unroll/repeats, fewer waves, staging | VGPR allocation, spills, occupancy |
| LDS pressure | fewer stages, smaller BK/BN, register path | LDS/workgroup and resident groups |
| Barrier cost | double-buffer data ownership, per-slot cells | barrier count and epilogue time |
| Padding dominates | bounded grid, compact alignment | early-exit workgroups and metadata loads |
| Epilogue dominates | fuse reduction/norm/combine | launch count and stage budget |

Resource arithmetic is a filter, not a predictor. A theoretically higher
occupancy configuration can lose because it adds instructions or reduces memory
coalescing.

## Staging and transpose

When testing global-to-LDS staging, distinguish direct DMA/copy instructions from
global-to-VGPR followed by DS writes. The second form can increase both registers
and traffic while being described as staging.

For packed weights, prove lane layout, alignment, EXEC requirements, scale
indexing, and destination register constraints. Assembler acceptance of an opcode
does not prove a correct or faster kernel.

Compare register-resident transpose, LDS transpose, permlane/shuffle, and direct
consumer layouts using generated ISA and resource metadata. Preserve a known-good
source and private compilation cache for each variant.

## Routing and alignment

At low route counts, sorting and padding may rival GEMM time. Candidate directions:

- One-workgroup histogram/prefix/scatter for small `T`.
- Mapping non-local experts before sorting.
- Capacity bounded to the actual contract.
- Reusing routing metadata between stages.
- Fusing quantization with sort only when it does not increase the critical rank.

Measure all-remote routing to reveal fixed alignment, launch, and early-exit costs.
Measure all-local routing to reveal the upper bound on expert work.

## Recognizing a roof

If several structurally different implementations converge near the same achieved
bandwidth and a pure streaming probe is only modestly higher, more tile tuning is
unlikely to produce a large win. Remaining directions are usually fewer bytes,
more independent work, an instruction-path change, or fusion that removes scratch
and launches.

State what was exhausted. Do not promise the pure streaming number as an attainable
MoE roof: routing, scales, activation, reduction, and imperfect coalescing are real
work.

## Experiment ledger

For each experiment record source hash, stage, dtype, shape, routing distribution,
config, correctness, median/tail, resources, and verdict. Include rejected attempts
such as pipeline depth, transpose path, waves-per-EU, cache policy, and prefetch.
Negative results prevent repeated work and make the final causal story reviewable.
