#!/usr/bin/env python3
"""Sum GPU-kernel device time for moe_sorting / mxfp4_moe_sort kernels in a torch/rocm trace."""
import gzip, json, sys, collections, re

def load(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path) as f:
        return json.load(f)

def classify(name):
    n = name.lower()
    if 'opus_moe_sorting_entry' in n:
        return 'opus_moe_sorting'
    if 'fused_mx_quant_moe_sort' in n or 'mxfp4_moe_sort' in n:
        return 'mxfp4_moe_sort(fused_quant)'
    if 'topkgatingsoftmax' in n or ('topk_softmax' in n):
        return 'topk_softmax(gating)'
    return None

def main(path):
    d = load(path)
    ev = d['traceEvents']
    # GPU kernel events: cat == 'kernel'
    buckets = collections.defaultdict(lambda: [0.0, 0])  # us_total, count
    for e in ev:
        if e.get('cat') != 'kernel':
            continue
        c = classify(e.get('name', ''))
        if c is None:
            continue
        buckets[c][0] += e.get('dur', 0)   # dur is in us
        buckets[c][1] += 1
    print(f"\n=== {path.split('/')[-1]} ===")
    total_sort = 0.0
    for k in ['opus_moe_sorting', 'mxfp4_moe_sort(fused_quant)', 'topk_softmax(gating)']:
        us, cnt = buckets.get(k, [0.0, 0])
        avg = us/cnt if cnt else 0
        print(f"  {k:32s}: total={us/1000:10.3f} ms  launches={cnt:6d}  avg={avg:8.3f} us")
        if 'sort' in k:
            total_sort += us
    print(f"  {'SORT TOTAL (opus+mxfp4)':32s}: total={total_sort/1000:10.3f} ms")
    return buckets

if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
