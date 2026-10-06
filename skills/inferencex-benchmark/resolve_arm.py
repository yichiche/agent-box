#!/usr/bin/env python3
"""Resolve an InferenceX benchmark arm into the env block its recipe expects.

This is the local stand-in for the matrix step CI runs before
runners/slurm_utils.sh launch_srt_single_node (arm -> matrix entry -> the row's
srt-recipe). It reads configs/amd-master.yaml and prints `export VAR=...` lines,
so the runner can `eval` it. Nothing here re-expresses a server or client flag;
the srt-slurm recipe owns those.

  resolve_arm.py --model-prefix qwen3.5 --mode fixed [--tp 2] [--framework sglang]
                 [--precision fp4] [--hw mi355x] [--spec mtp|none]
"""

import argparse
import json
import os
import sys

import yaml

# generate_sweep_configs.py:20
SEQ_LEN_STOI = {"1k1k": (1024, 1024), "8k1k": (8192, 1024)}
SEQ_LEN_ITOS = {v: k for k, v in SEQ_LEN_STOI.items()}

# generate_sweep_configs.py: `conc *= args.step_size`, default 2
STEP_SIZE = 2

SCENARIO_BY_MODE = {"fixed": "fixed-seq-len", "agent": "agentic-coding"}
SUBDIR_BY_MODE = {"fixed": "fixed_seq_len/", "agent": "agentic/"}


def die(msg, arms=None):
    print(f"resolve_arm: {msg}", file=sys.stderr)
    if arms:
        print("arms that do exist for this model prefix:", file=sys.stderr)
        for a in sorted(arms):
            print(f"  {a}", file=sys.stderr)
    sys.exit(1)


def expand_conc(entry):
    """conc-list verbatim, else conc-start..conc-end stepping by STEP_SIZE.

    Mirrors generate_sweep_configs.py:596-609 including the clamp that makes the
    last point land exactly on conc-end.
    """
    if "conc-list" in entry:
        return list(entry["conc-list"])
    start, end = entry["conc-start"], entry["conc-end"]
    out, conc = [], start
    while conc <= end:
        out.append(conc)
        if conc == end:
            break
        conc *= STEP_SIZE
        if conc > end:
            conc = end
    return out


def seq_len_str(isl, osl):
    return SEQ_LEN_ITOS.get((isl, osl), f"{isl}_{osl}")


def pick_arm(cfg, prefix, mode, framework, precision, hw, spec):
    """Find the one arm matching the request, or explain what is available."""
    scenario = SCENARIO_BY_MODE[mode]
    candidates, same_prefix = {}, set()
    for name, arm in cfg.items():
        if not isinstance(arm, dict) or arm.get("model-prefix") != prefix:
            continue
        same_prefix.add(name)
        if scenario not in (arm.get("scenarios") or {}):
            continue
        if arm.get("framework") != framework or arm.get("precision") != precision:
            continue
        if hw not in name:
            continue
        # `disagg` arms are a different topology entirely -- never pick one implicitly.
        if "disagg" in name:
            continue
        candidates[name] = arm

    if not candidates:
        die(f"no {scenario} arm for model-prefix={prefix} framework={framework} "
            f"precision={precision} hw={hw}", same_prefix)

    # spec=mtp -> the `-mtp` arm; spec=none -> the arm without it.
    wanted_mtp = spec == "mtp"
    exact = {n: a for n, a in candidates.items() if n.endswith("-mtp") == wanted_mtp}
    if not exact:
        die(f"no arm with spec-decoding={spec} among: {', '.join(sorted(candidates))}")
    if len(exact) > 1:
        die(f"ambiguous, {len(exact)} arms match: {', '.join(sorted(exact))}")
    return exact.popitem()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="defaults to $INFERENCEX_DIR/configs/amd-master.yaml")
    p.add_argument("--model-prefix", required=True)
    p.add_argument("--mode", required=True, choices=["fixed", "agent"])
    p.add_argument("--tp", type=int, default=None, help="omit to take the first search-space row")
    p.add_argument("--framework", default="sglang")
    p.add_argument("--precision", default="fp4")
    p.add_argument("--hw", default="mi355x")
    p.add_argument("--spec", default="mtp", choices=["mtp", "none"])
    p.add_argument("--kv-offloading", default="none", choices=["none", "dram"],
                   help="agent mode: 'dram' selects the HiCache row of the same arm")
    p.add_argument("--isl", type=int, default=8192, help="fixed mode: block to use when an arm has several")
    p.add_argument("--osl", type=int, default=1024)
    args = p.parse_args()

    e2e = os.environ.get("INFERENCEX_E2E_DIR")
    if not e2e:
        root = os.environ.get("INFERENCEX_DIR", "/home/yichiche/InferenceX")
        nested = os.path.join(root, "inferencex-e2e")
        e2e = nested if os.path.isfile(os.path.join(nested, "configs", "amd-master.yaml")) else root
    cfg_path = args.config or os.path.join(e2e, "configs", "amd-master.yaml")
    if not os.path.isfile(cfg_path):
        die(f"config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    arm_name, arm = pick_arm(cfg, args.model_prefix, args.mode, args.framework,
                             args.precision, args.hw, args.spec)
    scenario = SCENARIO_BY_MODE[args.mode]
    blocks = arm["scenarios"][scenario]
    if args.mode == "fixed" and len(blocks) > 1:
        blocks = [b for b in blocks if (b.get("isl"), b.get("osl")) == (args.isl, args.osl)]
    if len(blocks) != 1:
        die(f"expected exactly one {scenario} block in {arm_name}, got {len(blocks)}")
    block = blocks[0]

    rows = block["search-space"]
    # An agentic arm carries several rows per TP -- one per kv-offloading mode.
    # conc 20 appears in BOTH the kvnone and the HiCache row on purpose: that is
    # where InferenceX hands the curve over to HiCache.
    if args.mode == "agent":
        rows = [r for r in rows if r.get("kv-offloading", "none") == args.kv_offloading]
        if not rows:
            die(f"{arm_name} has no kv-offloading={args.kv_offloading} row")
    if args.tp is None:
        row = rows[0]
    else:
        matching = [r for r in rows if r.get("tp") == args.tp]
        if not matching:
            have = ", ".join(str(r.get("tp")) for r in rows)
            die(f"{arm_name} has no tp={args.tp} row with "
                f"kv-offloading={args.kv_offloading} (has tp: {have})")
        row = matching[0]

    tp = row["tp"]
    ep = row.get("ep", 1)
    spec = row.get("spec-decoding", "none")
    concs = expand_conc(row)

    kv_off, kv_backend, kv_metadata, kv_suffix = "none", "", "", "kvnone"
    if args.mode == "fixed":
        isl, osl = block["isl"], block["osl"]
        exp_name = f"{args.model_prefix}_{seq_len_str(isl, osl)}"
    else:
        isl = osl = ""
        kv_off = row.get("kv-offloading", "none")
        if kv_off != "none":
            backend = row.get("kv-offload-backend")
            if not backend or not backend.get("name"):
                die(f"{arm_name} tp{tp} has kv-offloading={kv_off} but no kv-offload-backend")
            # benchmark-tmpl.yml: the name alone, and the whole mapping as metadata.
            kv_backend = backend["name"]
            kv_metadata = json.dumps(backend, separators=(",", ":"))
            kv_suffix = f"kv{kv_off}-{backend['name']}"
        # EXP_NAME carries the conc, so the runner rebuilds it per concurrency;
        # here we emit the stem only.
        exp_name = f"{args.model_prefix}"

    subdir = SUBDIR_BY_MODE[args.mode]
    recipe = row.get("srt-recipe")
    if not recipe:
        die(f"{arm_name} tp{tp} has no srt-recipe; this runner only launches srt-slurm recipes")
    if not os.path.isfile(os.path.join(e2e, recipe)):
        die(f"srt-recipe does not exist: {os.path.join(e2e, recipe)}")

    out = {
        "ARM_NAME": arm_name,
        "MODEL": arm["model"],
        "MODEL_PREFIX": arm["model-prefix"],
        "IMAGE": arm["image"],
        "RUNNER_TYPE": arm["runner"],
        "FRAMEWORK": arm["framework"],
        "PRECISION": arm["precision"],
        "TP": tp,
        "EP_SIZE": ep,
        "SPEC_DECODING": spec,
        "DISAGG": "false",
        "CONC_LIST": " ".join(str(c) for c in concs),
        "ISL": isl,
        "OSL": osl,
        "EXP_NAME_STEM": exp_name,
        "SCENARIO_SUBDIR": subdir,
        "RECIPE": recipe,
        "DRAM_UTILIZATION": block.get("dram-utilization", ""),
        "KV_OFFLOADING": kv_off,
        "KV_OFFLOAD_BACKEND": kv_backend,
        "KV_OFFLOAD_BACKEND_METADATA": kv_metadata,
        "KV_SUFFIX": kv_suffix,
    }
    for k, v in out.items():
        print(f"export {k}='{v}'")


if __name__ == "__main__":
    main()
