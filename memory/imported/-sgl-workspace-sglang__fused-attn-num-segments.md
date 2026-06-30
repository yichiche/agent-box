# Fused attn segment-reduce: NUM_SEGMENTS is the dominant lever

Op: kernel_unified_attention_3d_fused (atomic-flag last-finisher in-kernel cross-segment merge), aiter triton, MI355X.
Shape: B=1,HQ=64,HK=8,D=128,ctx=8192,block=512. Two-launch baseline=13.94us (attn 8.95 + reduce 4.99).

## Key diagnosis (round 7)
Measured with a temporary DEBUG_SKIP_MERGE flag (finisher returns before merge):
- legacy attn-only (skip_reduce) = 8.8us
- fused with merge skipped = 29.5us  <-- merge CODE compiled into every one of the 128 segment
  programs lowers their occupancy 3.3x even though only the finisher runs it (register/LDS reservation).
- fused full = 70us  <-- so ~40us is the actual merge+barrier on top.

This is why the single-kernel atomic-flag approach is structurally capped above the 14us two-launch:
you cannot have the heavy attn loop and the heavy merge in the same @triton.jit without the merge's
footprint taxing the attn programs. Only a SECOND lightweight merge kernel (= two launches) avoids it.

## NUM_SEGMENTS is the untapped dominant lever
select_3d_config picks NUM_SEGMENTS=128 because target_num_prgms/num_2d_prgms is large (only 1 q_block x
8 kv_heads = 8 2D programs). That fills the GPU for the ATTN-ONLY two-launch path. But in the fused kernel
NUM_SEGMENTS sets BOTH attn-grid parallelism AND merge footprint + atomic-barrier width, and the merge
cost scales with it. Sweep (chunk=min(64,seg)):
  seg 8=36us, 16=27.1us, 32=26.9us (best), 64=38us, 128=69.5us.
seg must be power of 2 (tl.arange over segments in merge).

## Baked winning config (round id: low_segment_count_refused)
FUSED_NUM_SEGMENTS=32, FUSED_MERGE_CHUNK=32 (single chunk), FUSED_TILE_SIZE=128,
FUSED_NUM_WARPS=2, FUSED_WAVES_PER_EU=0  => 25.8us (was 70.68 at seg128/warps4). 63.5% intra-kernel win.
Optimal warps FLIPPED from 4 (seg128) back to 2 (seg32): less per-segment work + lighter merge = higher occ.
Correctness: max_abs_diff=0.000244, harness PASS, pytest -k 3d 64 passed.

Override knobs live in aiter/ops/triton/attention/unified_attention.py (FUSED_* module globals),
wired into the use_fused_reduce launch block (FUSED_NUM_SEGMENTS also reassigns local NUM_SEGMENTS so
grid/segm_*/lock stay consistent).

## Next-round ideas
- 25.8us is still ~1.8x the 14us two-launch. The cap is the register-poisoning + global barrier.
- To beat 14us you almost certainly need to STOP fusing the merge into the attn kernel: either accept a
  2nd tiny launch (defeats the goal) OR a true B200-style single-pass streaming kernel that splits on
  head dim for CTA count and never materializes segm_* (no merge code at all).
- Within the atomic-flag family, seg=32 looks near-optimal; further % gains are second-order.
