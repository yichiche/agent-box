#!/usr/bin/env python3
"""Build before/after e2e throughput comparison tables from bench_serving JSONL outputs."""
import json, glob, os, collections

import os
RES = os.environ.get("WORKDIR", "/tmp/pr3986_bench") + "/results"

def last_obj(path):
    obj = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: obj = json.loads(line)
                except: pass
    return obj

def metrics(d):
    dur = d.get('duration', 0) or 1e-9
    tot = d.get('total_input_tokens', 0) + d.get('total_output_tokens', 0)
    return {
        'completed': d.get('completed'),
        'tot_tput': tot / dur,
        'out_tput': d.get('output_throughput', 0),
        'req_tput': d.get('request_throughput', 0),
        'med_ttft': d.get('median_ttft_ms', 0),
        'med_tpot': d.get('median_tpot_ms', 0),
        'med_e2el': d.get('median_e2e_latency_ms', d.get('median_e2el_ms', 0)),
    }

# collect: data[(il,ol,conc)][phase] = metrics
data = collections.defaultdict(dict)
for path in glob.glob(f"{RES}/*_il*_ol*_conc*.json"):
    base = os.path.basename(path)
    if base.startswith('before_profile') or base.startswith('after_profile'): continue
    phase = 'before' if base.startswith('before') else 'after'
    # parse il/ol/conc
    import re
    m = re.search(r'il(\d+)_ol(\d+)_conc(\d+)', base)
    if not m: continue
    il, ol, conc = int(m.group(1)), int(m.group(2)), int(m.group(3))
    d = last_obj(path)
    if d: data[(il, ol, conc)][phase] = metrics(d)

for il in sorted(set(k[0] for k in data)):
    ol = 1024
    print(f"\n{'='*100}\n  IL={il} / OL={ol}  —  total token throughput (tok/s) & median latencies\n{'='*100}")
    print(f"{'conc':>5} | {'tot_tput before':>15} {'after':>10} {'Δ%':>7} | {'out_tput b/a':>16} | {'ttft b/a ms':>16} | {'tpot b/a ms':>16} | {'e2el b/a ms':>18}")
    print('-'*120)
    for conc in sorted(c for (i,o,c) in data if i==il):
        rec = data[(il, ol, conc)]
        b = rec.get('before'); a = rec.get('after')
        if not b or not a:
            print(f"{conc:>5} | MISSING before={bool(b)} after={bool(a)}")
            continue
        d_tput = (a['tot_tput']/b['tot_tput']-1)*100 if b['tot_tput'] else 0
        print(f"{conc:>5} | {b['tot_tput']:>15.0f} {a['tot_tput']:>10.0f} {d_tput:>+6.1f}% | "
              f"{b['out_tput']:>7.0f}/{a['out_tput']:<8.0f} | "
              f"{b['med_ttft']:>7.0f}/{a['med_ttft']:<8.0f} | "
              f"{b['med_tpot']:>7.2f}/{a['med_tpot']:<8.2f} | "
              f"{b['med_e2el']:>8.0f}/{a['med_e2el']:<9.0f}")

print()
