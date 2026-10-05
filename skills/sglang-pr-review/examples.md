# Calibration — sglang PR 39575

https://github.com/sgl-project/sglang/pull/39575

**Test-seam conclusion:** Mixed. The write-plan tests call the changed
interface. The recompression tests step past the orchestration seam.

**Changed behavior:** For an unaligned cached prefix, snapshot prefix-side keys
before `project_qk` overwrites their pending-ring slots, then recompress the
crossing group.

**Correct seam:** `_forward_impl`, because it owns the ordering:

```
_forward_impl
├─ _snapshot_cross_prefix_members
├─ project_qk                         writes the pending ring
└─ update_key_state_and_compress
   └─ _overwrite_cross_prefix_groups
```

## Test entry map

```
test_qsa_write_plan_* (3)
└─ QwenSparseAttnBackend._qsa_write_plan       interface reached

test_qsa_cross_prefix_* (3)
└─ fixture manually snapshots and mutates ring
   └─ _overwrite_cross_prefix_groups           enters below _forward_impl
```

## Findings

| Test | Seam it calls | Expected value source | Verdict |
|---|---|---|---|
| `test_qsa_write_plan_tracks_group_crossing_extend_prefix` | `_qsa_write_plan` | literals `[2, 0, 0]`, `[-2, 4, 8]` | Interface test |
| `test_qsa_write_plan_cross_rows_skip_rows_without_entries` | `_qsa_write_plan` | literals | Interface test |
| `test_qsa_write_plan_cross_rows_cover_every_straddling_entry` | `_qsa_write_plan` | invariant over the function's own outputs | Interface test; no copied clamp formula |
| `test_qsa_cross_prefix_recompress_reads_the_pre_store_ring` | private `_overwrite_cross_prefix_groups` | literal mean `(8+9+10+11)/4`, RoPE `8` | Past the seam; the expected value is independent |
| `test_qsa_cross_prefix_fused_scratch_denotes_the_same_groups` | private helper; fused store is stubbed | literal member layout | Past the seam; the stub is an adapter at an internal seam |
| `test_qsa_cross_prefix_fused_kernel_matches_eager` | both private-helper branches | one production branch against another | Past the seam; not a reimplementation |
| fixture fields and lambdas | none | none | Fixture fallout, not a test of new behavior |

## Missing interface test

Call `_forward_impl` with an unaligned prefix and assert the stored compressed
key. Let production itself snapshot the old ring, call `project_qk`, and perform
the overwrite. That test fails if the snapshot moves after the store.
