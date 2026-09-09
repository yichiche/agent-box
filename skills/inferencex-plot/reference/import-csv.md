# CSV Import Format

This document is the integration contract for projects that generate CSV/TSV
files for InferenceX Curve's `Import File` workflow.

For the independent Plot Tool workspace and its generic `Line ID` / `X` / `Y`
format, use [`import-plot-tool-csv.md`](import-plot-tool-csv.md) instead. The two
CSV contracts are not interchangeable.

The app accepts `.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson`, and `.zip`
archives containing those files. For external CSV/TSV integrations, prefer the
**editor CSV** format below. It is the same shape produced by `Download CSV`,
round-trips through the app, and is more deterministic than raw benchmark CSV.

## Recommended Editor CSV

Use this header order for generated CSV:

```csv
Line ID,Line Name,Title,Line Note,Model,Scenario,Precision,MTP,HW Key,Color Mode,Resolved Color,Line Type,Line Marker,Layer,Included in Chart,Active Line,Point Index,Roofline Point,Point Marker,Interactivity (tok/s/user),Throughput/GPU (tok/s/gpu),TTFT (s),End-to-end (s),P50 Interactivity (tok/s/user),P75 Interactivity (tok/s/user),P90 Interactivity (tok/s/user),P95 Interactivity (tok/s/user),P50 TTFT (s),P75 TTFT (s),P90 TTFT (s),P95 TTFT (s),P50 End-to-end (s),P75 End-to-end (s),P90 End-to-end (s),P95 End-to-end (s),P75 E2E Normalized Interactivity (tok/s/user),P90 E2E Normalized Interactivity (tok/s/user),Prefill GPUs,Decode GPUs,Total GPUs,Prefill TP,Prefill EP,Prefill DCP,Prefill DPA,Prefill Workers,Decode TP,Decode EP,Decode DCP,Decode DPA,Decode Workers,DPA,Disagg,Multi-node,KV Offload,Chip Cache Hit Rate,External Cache Hit Rate,CPU Cache Hit Rate,Theoretical Cache Hit Rate,Concurrency,Strategy,Note
```

The importer uses the editor parser when at least one row has a non-empty
`Line ID` / `series_id` / `line_id`. Repeat the same line fields for every point
row in the same curve.

### Required Fields

These fields should always be present and non-empty in generated editor CSV:

- `Line ID`: stable id for grouping rows into one curve.
- `Line Name`: legend label.
- `Model`: filter value, for example `DeepSeek-R1-0528`.
- `Scenario`: sequence/scenario value, for example `ISL 8192 / OSL 1024`, `ISL 1024 / OSL 1024`, or `Agentic Traces`. The UI classifies numeric ISL/OSL values under `Fixed Sequence Length` and agentic labels under `Agentic Traces`.
- `Precision`: filter value, for example `fp4` or `fp8`.
- `Throughput/GPU (tok/s/gpu)`: numeric Y value.

Each point row with data must also include at least one numeric X-axis metric:

- `Interactivity (tok/s/user)`
- `TTFT (s)`
- `End-to-end (s)`
- `P75 E2E Normalized Interactivity (tok/s/user)` or
  `P90 E2E Normalized Interactivity (tok/s/user)`

For `Agentic Traces`, use the twelve canonical latency-percentile columns:

- `P50/P75/P90/P95 Interactivity (tok/s/user)`
- `P50/P75/P90/P95 TTFT (s)`
- `P50/P75/P90/P95 End-to-end (s)`

These four percentiles are stored independently. The app's `Latency
Percentile` selector applies the selected percentile to all three metrics.
`P99`, `mean`, and `std` are not percentile columns. The unprefixed
`Interactivity (tok/s/user)`, `TTFT (s)`, and `End-to-end (s)` columns remain
the P90 compatibility baseline for older agentic files; they do not make P50,
P75, or P95 available when the corresponding columns are absent. Fixed-length
rows keep their existing unprefixed metric semantics.

`E2E Normalized Interactivity` is a higher-is-better Agentic metric. InferenceX
currently publishes only P75 and P90 for it through the derived Agentic
metrics associated with persisted per-request traces. The app therefore stores
only those two percentiles and does not synthesize P50 or P95 values.

Avoid relying on the app's internal fallbacks for empty `Line ID`, `Line Name`,
`Model`, `Scenario`, or `Precision`; those defaults depend on current app state
and are not suitable for generated CSV. `MTP` is technically optional, but
generated integrations should prefer explicit `MTP` or `Non-MTP`.

### Fields That May Be Empty

These fields are optional in editor CSV:

- `Title`: optional tooltip/title metadata.
- `Line Note`: optional curve-level notes shown only in the data editor. It is
  not used by the chart, legend, search, or tooltips. Repeat it for each point
  row belonging to the same line when generating CSV.
- `MTP`: empty values are inferred from `Line ID`, `Line Name`, and `Title`; if
  no `mtp` token is found, the line becomes `Non-MTP`.
- `Color Mode`: use `Custom` only when `Resolved Color` should be imported.
  Empty or `Auto` means automatic color.
- `Resolved Color`: CSS color such as `#22c55e`; ignored unless `Color Mode` is
  `Custom`.
- `Line Type`: empty defaults to `solid`. Supported named values are `solid`,
  `dashed`, `dotted`, `dashdot`, and `long-dash`; custom dash arrays are also
  accepted after import.
- `Line Marker`: empty, `default`, `auto`, or `precision` means use the
  precision/default marker. Supported explicit markers are `circle`, `square`,
  `triangle`, `diamond`, `star`, `plus`, and `cross`.
- `Layer`: empty falls back to row order.
- `Point Marker`: same marker values as `Line Marker`; empty inherits the line
  marker.
- `Interactivity (tok/s/user)`, `TTFT (s)`, `End-to-end (s)`,
  all twelve Agentic `P50/P75/P90/P95` latency columns,
  and `P75/P90 E2E Normalized Interactivity (tok/s/user)`: optional numeric
  X-axis metrics, as long as at least one is present for the row.
- `Prefill GPUs`, `Decode GPUs`, `Prefill TP`, `Prefill EP`, `Prefill DCP`,
  `Prefill Workers`, `Decode TP`, `Decode EP`, `Decode DCP`, `Decode Workers`, and
  `Concurrency`: numeric when present.
- `Prefill DPA`, `Decode DPA`, `DPA`, `Disagg`, and `Multi-node`: boolean when
  present. Accepted values are `true`, `false`, `1`, `0`, `yes`, `no`, `y`,
  and `n`.
- `Strategy`: optional display/tooltip text. If empty, the app derives a
  strategy label from decode TP/EP when possible.
- `KV Offload`: optional point metadata such as `KV offload off` or
  `KV offload DRAM via LMCache`. It is not a line grouping dimension, but it is
  preserved for tooltips/export and is included in Copy and split by config.
- `Chip Cache Hit Rate`, `External Cache Hit Rate`, `CPU Cache Hit Rate`, and
  `Theoretical Cache Hit Rate`: optional raw ratios in the `0`–`1` range. The
  tooltip formats them as percentages; external hit rate is preserved for data
  compatibility but is not displayed separately.
- `Note`: optional point-level tooltip/source text; it is distinct from `Line Note`.

Rows where all point columns are empty are skipped. A row with any point data
must have numeric `Throughput/GPU` and at least one numeric X-axis metric,
otherwise import fails.

### Ignored Export Columns

The app's `Download CSV` output contains derived columns. They are safe to keep
when round-tripping, but they are ignored on import:

- `HW Key`
- `Included in Chart`
- `Active Line`
- `Point Index`
- `Roofline Point`
- `Total GPUs`

`Total GPUs` is recomputed from point metadata: disaggregated deployments add
prefill and decode GPU counts, while non-disaggregated deployments treat them
as the same shared pool and use the larger count.

### Header Aliases

Header matching is case-insensitive and mostly punctuation-insensitive.
Generated integrations should still use the canonical headers above.

Accepted editor CSV aliases include:

- Line id/name: `series_id`, `line_id`, `Line ID`; `series_name`, `Line Name`,
  `name`
- Line fields: `model`, `islOsl`, `Scenario`, `ISL/OSL`, `precision`, `mtp`, `title`,
  `Line Note`, `line_note`, `series_note`,
  `lineStyle`, `Line Type`, `line_marker`, `Line Marker`, `renderOrder`,
  `Layer`
- Interactivity: `Interactivity`, `Interactivity (tok/s/user)`,
  `P90 Interactivity`, `P90 Interactivity (tok/s/user)`, `median_intvty`,
  `p90_intvty`, `p90_interactivity`, `metrics.median_intvty`,
  `metrics.p90_intvty`, `metrics.p90_interactivity`, `tok/s/user`
- Throughput: `Throughput/GPU`, `Throughput/GPU (tok/s/gpu)`,
  `throughput`, `tok/s/gpu`
- TTFT: `TTFT`, `TTFT (s)`, `Time To First Token`,
  `Time To First Token (s)`, `P90 TTFT`, `P90 Time To First Token`,
  `median_ttft`, `p90_ttft`, `metrics.median_ttft`, `metrics.p90_ttft`
- End-to-end: `End-to-end`, `End-to-end (s)`, `End-to-end Latency`,
  `End-to-end Latency (s)`, `endToEnd`, `end_to_end`, `E2E`, `E2E Latency`,
  `e2el`, `median_e2el`, `p90_e2el`, `p90_end_to_end`,
  `P90 End-to-end Latency`, `metrics.median_e2el`, `metrics.p90_e2el`
- Agentic percentile columns also accept lowercase metric aliases such as
  `p75_intvty`, `p75_ttft`, and `p75_e2el`, their `metrics.*` forms, and the
  nested JSON/flattened-header aliases
  `request_metrics.latency.intvty.p50|p75|p90|p95`,
  `request_metrics.latency.ttft.p50|p75|p90|p95`, and
  `request_metrics.latency.e2el.p50|p75|p90|p95`.
- E2E Normalized Interactivity: `P75 E2E Normalized Interactivity`,
  `P90 E2E Normalized Interactivity`, their canonical `(tok/s/user)` headers,
  `p75_e2e_norm_intvty`, `p90_e2e_norm_intvty`, their `metrics.*` and
  `derived_agentic_metrics.*` forms, the upstream Agentic aggregate paths
  `request_metrics.latency.e2e_norm_intvty.p75|p90`, plus nested
  `e2e_normalized_interactivity.p75|p90` and
  `e2eNormalizedInteractivityPercentiles.p75|p90`.
- Point metadata: `shape`, `Marker`, `Point Marker`; snake_case point keys such
  as `num_prefill_gpu`, `decode_tp`, and `prefill_dp_attention`; display labels
  such as `Prefill GPUs`, `Prefill DCP`, `Decode TP`, `Decode DCP`,
  `Prefill Workers`, `DPA`, `Disagg`, `Multi-node`, and `KV Offload`
- Cache hit rates: `server_gpu_cache_hit_rate`, `Chip Cache Hit Rate`,
  `server_external_cache_hit_rate`, `External Cache Hit Rate`,
  `server_cpu_cache_hit_rate`, `CPU Cache Hit Rate`,
  `theoretical_cache_hit_rate`, and `Theoretical Cache Hit Rate`

## Example

```csv
Line ID,Line Name,Title,Line Note,Model,Scenario,Precision,MTP,HW Key,Color Mode,Resolved Color,Line Type,Line Marker,Layer,Included in Chart,Active Line,Point Index,Roofline Point,Point Marker,Interactivity (tok/s/user),Throughput/GPU (tok/s/gpu),TTFT (s),End-to-end (s),P50 Interactivity (tok/s/user),P75 Interactivity (tok/s/user),P90 Interactivity (tok/s/user),P95 Interactivity (tok/s/user),P50 TTFT (s),P75 TTFT (s),P90 TTFT (s),P95 TTFT (s),P50 End-to-end (s),P75 End-to-end (s),P90 End-to-end (s),P95 End-to-end (s),P75 E2E Normalized Interactivity (tok/s/user),P90 E2E Normalized Interactivity (tok/s/user),Prefill GPUs,Decode GPUs,Total GPUs,Prefill TP,Prefill EP,Prefill DCP,Prefill DPA,Prefill Workers,Decode TP,Decode EP,Decode DCP,Decode DPA,Decode Workers,DPA,Disagg,Multi-node,KV Offload,Chip Cache Hit Rate,External Cache Hit Rate,CPU Cache Hit Rate,Theoretical Cache Hit Rate,Concurrency,Strategy,Note
dsr1-8192-fp8-b200-trt,B200 TRT,DeepSeek R1 B200 TRT,Production candidate,DeepSeek-R1-0528,ISL 8192 / OSL 1024,fp8,Non-MTP,,Auto,,solid,precision,1,,,,,,8.42,5220.5,0.12,9.04,,,,,,,,,,,,,,,4,8,,4,4,,true,,8,8,,true,,true,true,false,,,,,,1024,,run 123
dsr1-agentic-fp8-b200-trt,B200 TRT Agentic,DeepSeek R1 B200 TRT agentic traces,Agentic validation run,DeepSeek-R1-0528,Agentic Traces,fp8,Non-MTP,,Auto,,solid,precision,2,,,,,,8.5,4830.2,0.18,37.4,12,10,8.5,7.8,0.10,0.14,0.18,0.22,25,31,37.4,42,19.78,11.24,4,8,,4,4,8,true,,8,8,8,true,,true,false,false,KV offload off,0.92,0,0,0.97,64,,agentic preview
```

## Raw Benchmark CSV Fallback

Raw benchmark CSV is supported only as a fallback. If no row has `Line ID`,
the importer groups rows by parsed model, scenario, precision,
MTP/spec, hardware, and framework. KV/offload mode is kept as `KV Offload`
point metadata; it does not split rows into separate lines.

Each raw benchmark row must have a numeric throughput column and at least one
numeric X-axis metric column. Accepted aliases include:

- Interactivity: `metrics.median_intvty`, `metrics.interactivity`,
  `metrics.p90_intvty`, `metrics.p90_interactivity`, `median_intvty`,
  `median_interactivity`, `p90_intvty`, `p90_interactivity`,
  `interactivity`, `P90 Interactivity`, `tok/s/user`, `x`
- Nested agentic interactivity:
  `request_metrics.latency.intvty.p50|p75|p90|p95`
- Throughput: `metrics.tput_per_gpu`, `tput_per_gpu`, `throughput_per_gpu`,
  `token throughput per gpu`, `token throughput per gpu (tok/s/gpu)`,
  `throughput`, `Throughput/GPU`, `Throughput/GPU (tok/s/gpu)`, `tok/s/gpu`,
  `y`, `request_metrics.throughput.per_gpu.total_tput_tps`
- TTFT: `metrics.median_ttft`, `metrics.p90_ttft`, `median_ttft`,
  `p90_ttft`, `ttft`, `TTFT`, `TTFT (s)`, `Time To First Token`,
  `Time To First Token (s)`, `P90 TTFT`, `P90 Time To First Token`,
  `request_metrics.latency.ttft.p50|p75|p90|p95`
- End-to-end: `metrics.median_e2el`, `metrics.p90_e2el`,
  `metrics.p90_end_to_end`, `median_e2el`, `p90_e2el`, `p90_end_to_end`,
  `endToEnd`, `end_to_end`, `end-to-end`, `End-to-end (s)`,
  `End-to-end Latency`, `E2E`, `E2E Latency`, `e2el`,
  `P90 End-to-end Latency`,
  `request_metrics.latency.e2el.p50|p75|p90|p95`
- E2E Normalized Interactivity: `p75_e2e_norm_intvty`,
  `p90_e2e_norm_intvty`, their `metrics.*` and
  `derived_agentic_metrics.*` forms,
  `request_metrics.latency.e2e_norm_intvty.p75|p90`, the canonical P75/P90
  headers, and nested
  `e2e_normalized_interactivity.p75|p90` or
  `e2eNormalizedInteractivityPercentiles.p75|p90`

Optional raw benchmark aliases include:

- Model: `infmax_model_prefix`, `model_prefix`, `model_key`, `db_model`,
  `model`, `model_name`
- Hardware: `hardware`, `hw_key`, `hwKey`, `hw`, `gpu`, `accelerator`
- Framework: `framework`, `backend`, `runtime`
- MTP/spec: `mtp`, `spec_method`, `spec_decoding`, `speculation`
- Precision: `precision`, `dtype`, `quantization`
- Lengths: `isl`, `input_len`, `input_length`, `input sequence length`,
  `input_tokens`; `osl`, `output_len`, `output_length`,
  `output sequence length`, `output_tokens`
- Scenario/workload: `scenario`, `benchmark_scenario`, `benchmark scenario`,
  `benchmark type`, `benchmark_type`, `workload`, `workload_type`,
  `workload type`, `trace`, `trace_type`, `trace type`, `dataset`, `task`
- Offload: `offload_mode`, `kv_offloading`, and `kv_offload_backend`
  (top-level or under `metrics`). For example, agentic traces may import
  `benchmark_type=agentic_traces`, `offload_mode=on`, `kv_offloading=dram`,
  and `kv_offload_backend=lmcache`; the importer keeps that label on the
  point's `KV Offload` metadata while merging it into the same line as
  non-offload rows. Object-shaped backends such as
  `kv_offload_backend.name=hicache` are also supported.
- Parallelism/source: `num_prefill_gpu`, `num_decode_gpu`, `prefill_tp`,
  `prefill_ep`, `dcp_size`, `prefill_dcp_size`, `prefill_num_workers`,
  `decode_tp`, `decode_ep`, `decode_dcp_size`,
  `decode_num_workers`, `dp_attention`, `prefill_dp_attention`,
  `decode_dp_attention`, `disagg`, `is_multinode`, `multi_node`, `multinode`,
  `conc`, `concurrency`, `batch_size`, `date`, `created_at`, `run_date`,
  `timestamp`
- Cache hit rates: `server_gpu_cache_hit_rate`,
  `server_external_cache_hit_rate`, `server_cpu_cache_hit_rate`, and
  `theoretical_cache_hit_rate`, either top-level or nested under `metrics`.
  Native Agentic Trace artifacts are also supported: server-observed rates are
  read from `server_metrics.cache`, while the theoretical rate is read from
  `request_metrics.cache.theoretical_cache_hit_rate`.

In raw benchmark CSV, optional metadata may be empty. Agentic scenario/workload
values such as `agentic_traces` become `Agentic Traces`. Fixed-run values such
as `single_turn` do not replace sequence lengths: `isl` and `osl` are used when
available, preserving `ISL 8192 / OSL 1024` and `ISL 1024 / OSL 1024`. Missing
hardware/framework becomes `unknown`, missing precision becomes `default`,
missing scenario becomes `Default Scenario`, and missing spec/MTP becomes
`Non-MTP`.
