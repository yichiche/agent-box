# Calibration — sglang PR 39575

The chat reply for this PR is only the conclusion and the canvas link. The
traces later in this file belong in the canvas, not in the chat.

Final conclusion:
- Can merge. NVIDIA keeps its original outputs and alignment assertion, and the unaligned-prefix correction runs only under `is_gfx95_supported()`.

Analysis:
Common path:
- Yes. Every backend reads `extend_prefix_lens_cpu`, receives a wider write-plan tuple, calls a snapshot helper, and evaluates one extra condition.
NVIDIA execution flow identical:
- No.
NVIDIA internal interface identical:
- No.
NVIDIA numerical results and original behavior identical:
- Yes.
Affected hardware scope:
- gfx950 / MI355 only (`is_gfx95_supported()`).

[PR 39575 code path](/home/yichiche/.cursor/projects/home-yichiche/canvases/pr-39575-code-path.canvas.tsx)

## Canvas source — do not print

`[AMD] [Fix] Handle unaligned QSA extend prefixes from chunk-cache tails`

https://github.com/sgl-project/sglang/pull/39575

The title says AMD, while all production files are shared. The correction is
gfx950-only, but some setup work is on the common path.

**Common Path:** Yes — every backend reads `extend_prefix_lens_cpu`, receives a
wider write-plan tuple, calls a snapshot helper, and reaches one extra condition.

**NVIDIA**
- **Execution flow completely identical: No.** It performs a new attribute read,
  computes three extra plan tensors, calls a helper that returns `None`, and
  evaluates an extra false condition.
- **Internal interface completely identical: No.** `_qsa_write_plan` returns
  seven values instead of four; metadata and the compression method gain fields
  and an argument. Production callers are updated. Partial fixtures need the
  already-existing `ForwardBatch` field added.
- **Numerical results and original behavior identical: Yes.** The original four
  plan values are unchanged; the alignment assertion remains on NVIDIA; the new
  values select no correction; final compressed keys are unchanged.

**Affected Hardware Scope:** The corrected unaligned-prefix behavior is
**gfx950 / MI355 only**, under `is_gfx95_supported()`. It is not all HIP, not
AITER-gated, and does not include MI300.

**Merge conclusion:** Safe for NVIDIA from this code-path analysis. Do not block
the PR merely because the common path executes extra setup.

Execution flow and the internal interface are both not identical, so the
numerical **Yes** above is only a label. The proof is the two traces below.
Same NVIDIA batch in both: `prefix_lens = 64`, `extend_len = 8`, `ratio = 4`,
`is_gfx95 = False`.

## Value trace — read `extend_prefix_lens_cpu`

```python
# qwen_sparse_attn_backend.py · _qsa_build_write_plan · line 562
# qsa_indexer.py · _forward_impl · line 691
# qsa_indexer.py · update_key_state_and_compress · line 330–367
prefix_lens = 64
extend_len = 8
ratio = 4
is_gfx95 = False

assert (prefix_lens % ratio == 0)  # 64 % 4 == 0 → True

token_k = project_qk()              # 8 new tokens
compressed = compress(token_k)
```

```python
# qwen_sparse_attn_backend.py · _metadata_from_forward_batch · line 707
# qwen_sparse_attn_backend.py · _qsa_build_write_plan · line 588
# qwen_sparse_attn_backend.py · _qsa_write_plan · line 536
# qsa_indexer.py · _snapshot_cross_prefix_members · line 292
# qsa_indexer.py · _forward_impl · line 812
# qsa_indexer.py · update_key_state_and_compress · line 410
prefix_lens = 64
extend_len = 8
ratio = 4
is_gfx95 = False

extend_prefix_lens_cpu = [64]       # new read
has_cross_prefix_group = bool(
    is_gfx95 and extend_prefix_lens_cpu is not None and any(...)
)  # False and ... → False

assert (prefix_lens % ratio == 0)  # 64 % 4 == 0 → True
prefix_members = [0]                 # 64 is aligned

if not (is_gfx95 and has_cross_prefix_group):
    cross_prefix_state = None        # not (False and False)
token_k = project_qk()              # 8 new tokens
if cross_prefix_state is not None:   # None → skip
    overwrite()
compressed = compress(token_k)
```

```python
before.compressed == after.compressed              # True
before.token_k == after.token_k                    # True
before.assert_passed == after.assert_passed        # True == True

# prefix_lens = 842 still asserts on both sides
before.assert_fired == after.assert_fired          # True == True
after.reached_overwrite                            # False
```

```python
# qsa_indexer.py · update_key_state_and_compress · line 359–420
# qwen_sparse_attn_backend.py · _metadata_from_forward_batch · line 776
is_gfx95 = False
has_cross_prefix_group = False
cross_prefix_state = None
is_extend = True
group_member_rows = tensor([0, 4])   # not None
member_rows = tensor([0, 4])

group_locs = member_rows[:, None] + arange(4)
# [[0, 1, 2, 3], [4, 5, 6, 7]]
if cross_prefix_state is not None:       # None → False
    prefix_members = compress_prefix_members
    group_locs = maximum(group_locs, ...)
source_keys = token_k                 # old extend path

if is_extend and cross_prefix_state is not None:
    # True and False → False
    overwrite(compress_cross_prefix_members)

if group_member_rows is None or has_cross_prefix_group:
    # False or False → False
    compress_group_ring_locs = build_group_ring_slots(...)

compressed = compress(source_keys[group_locs])
```

```python
# qsa_indexer.py · update_key_state_and_compress · line 359–420
# qwen_sparse_attn_backend.py · _metadata_from_forward_batch · line 776
is_gfx95 = True
extend_prefix_lens_cpu = [10]
has_cross_prefix_group = True        # 10 % 4 != 0
cross_prefix_state = (ring_keys, ring_rope)
member_rows = tensor([-2])
compress_prefix_members = tensor([2])

group_locs = member_rows[:, None] + arange(4)
# [[-2, -1, 0, 1]]
if cross_prefix_state is not None:       # entered
    group_locs = maximum(group_locs, member_rows + 2)
    # [[0, 0, 0, 1]]

if is_extend and cross_prefix_state is not None:
    # entered
    overwrite(prefix_members=2, ring_keys)

if group_member_rows is None or has_cross_prefix_group:
    # entered
    compress_group_ring_locs = build_group_ring_slots(...)
```

## Value trace — `_qsa_write_plan` returns 7 values instead of 4

```python
# qwen_sparse_attn_backend.py · _qsa_write_plan · line 488–530
prefix_lens = tensor([64])
extend_len = tensor([8])
ratio = 4
start_blocks = prefix_lens // ratio               # [16]
end_blocks = (prefix_lens + extend_len) // ratio  # [18]
row_token_starts = tensor([0])

write_locs = tensor([16, 17])
group_end_positions = tensor([67, 71])
rows = tensor([0, 0])
member_rows = tensor([0, 4])          # 0 + [16, 17] * 4 - 64

return (
    write_locs,            # [16, 17]
    group_end_positions,   # [67, 71]
    rows,                  # [0, 0]
    member_rows,           # [0, 4]
)
```

```python
# qwen_sparse_attn_backend.py · _qsa_write_plan · line 488–559
prefix_lens = tensor([64])
extend_len = tensor([8])
ratio = 4
start_blocks = prefix_lens // ratio               # [16]
end_blocks = (prefix_lens + extend_len) // ratio  # [18]
row_token_starts = tensor([0])

write_locs = tensor([16, 17])
group_end_positions = tensor([67, 71])
rows = tensor([0, 0])
member_rows = tensor([0, 4])          # 0 + [16, 17] * 4 - 64
prefix_members = (64 - [64, 68]).clamp(0, 3)   # [0, 0]
cross_rows = tensor([0])
cross_prefix_members = prefix_members[cross_rows]  # [0]

return (
    write_locs,            # [16, 17]
    group_end_positions,   # [67, 71]
    rows,                  # [0, 0]
    member_rows,           # [0, 4]
    prefix_members,        # [0, 0]
    cross_rows,            # [0]
    cross_prefix_members,  # [0]
)
```

```python
before == after[:4]                    # True
# write_locs            [16, 17] == [16, 17]
# group_end_positions   [67, 71] == [67, 71]
# rows                  [0, 0] == [0, 0]
# member_rows           [0, 4] == [0, 4]
```

```python
# qsa_indexer.py · update_key_state_and_compress · line 361–420
is_gfx95 = False
cross_prefix_state = None
member_rows = tensor([0, 4])
prefix_members = tensor([0, 0])

group_locs = member_rows[:, None] + arange(4)
# [[0, 1, 2, 3], [4, 5, 6, 7]]
if cross_prefix_state is not None:     # None → False
    group_locs = maximum(group_locs, member_rows + prefix_members)

# identity, if the zeros were applied anyway:
maximum(group_locs, member_rows + 0) == group_locs  # True

if is_extend and cross_prefix_state is not None:
    # True and False → False
    overwrite(cross_rows, cross_prefix_members)
source_keys = token_k
compressed = compress(source_keys[group_locs])
```

```python
# qsa_indexer.py · update_key_state_and_compress · line 361–420
is_gfx95 = True
prefix_lens = tensor([10])
cross_prefix_state = (ring_keys, ring_rope)
member_rows = tensor([-2])
prefix_members = tensor([2])         # (10 - 8).clamp(0, 3)
cross_rows = tensor([0])
cross_prefix_members = tensor([2])

group_locs = member_rows[:, None] + arange(4)
# [[-2, -1, 0, 1]]
if cross_prefix_state is not None:     # entered
    group_locs = maximum(group_locs, -2 + 2)
    # [[0, 0, 0, 1]]
if is_extend and cross_prefix_state is not None:
    overwrite(cross_rows=[0], prefix_members=[2])
```

## Path

```
_metadata_from_forward_batch                         common
├─ read extend_prefix_lens_cpu                       common structural change
├─ has_cross_prefix_group = is_gfx95 ∧ unaligned    gfx950-only decision
├─ _qsa_build_write_plan
│  ├─ assert prefix % ratio == 0                     kept on NVIDIA / non-gfx95
│  └─ _qsa_write_plan                                common structural change
│                                                    (4 → 7 values; original 4 unchanged)
└─ build_group_ring_slots                            common control-flow edit
                                                     (same NVIDIA result)
_forward_impl                                        common
├─ _snapshot_cross_prefix_members()                  common call; gfx950-only body
├─ project_qk                                        unchanged; this is the store the snapshot must beat
└─ update_key_state_and_compress
   ├─ fused store, no early return                   common control-flow edit
   │                                                 (next condition is false on NVIDIA)
   └─ _overwrite_cross_prefix_groups                 gfx950-only correction
```

## Common-path and NVIDIA evidence

| Edit | Runs on NVIDIA? | Execution/interface effect | Numerical/behavior effect |
|---|---|---|---|
| read `extend_prefix_lens_cpu` | yes | extra read; partial fixtures must expose the field | real `ForwardBatch` already defines it; short-circuit ignores the value |
| `_qsa_write_plan` adds three outputs | yes | extra tensor work; return arity changes 4 → 7 | aligned NVIDIA prefixes produce zero prefix members; original four outputs are identical |
| alignment assertion moves under `if not is_gfx95_supported()` | yes | one platform condition | NVIDIA raises under the same unaligned-prefix condition |
| metadata gains fields defaulted to `None` / `False` | yes | internal struct shape changes | old path observes old defaults |
| `_forward_impl` calls snapshot helper | yes | extra call | returns `None`; correction is skipped |
| fused branch no longer returns immediately | yes | reaches one extra `if` | condition is false because snapshot is `None` |
| extend ring-slot `else` narrows | yes | control-flow shape changes | NVIDIA extend reads packed rows; result is unchanged |

The real `ForwardBatch` already defines
`extend_prefix_lens_cpu: Optional[List[int]] = None`. The `AttributeError` in
tests came from partial `SimpleNamespace` fixtures, not production NVIDIA.

The diff realigns the old eager-compress body. On the PR head it is still the
`else` of `_use_fused_compress`; it is moved, not new.

## Hardware scope

| New behavior | Enclosing guard | AMD hardware affected | NVIDIA |
|---|---|---|---|
| allow an unaligned prefix past the assertion | `if not is_gfx95_supported(): assert` | gfx950 / MI355 | assertion remains |
| snapshot pending-ring prefix members | `is_gfx95_supported() and metadata.has_cross_prefix_group` | gfx950 / MI355 with unaligned prefix | helper returns `None` |
| clamp and overwrite a crossing group | `if cross_prefix_state is not None` after guarded snapshot | gfx950 / MI355 with crossing group | does not run |

`is_gfx95_supported()` excludes MI300. No `_use_aiter` guard controls this
correction, so AITER does not define its scope.
