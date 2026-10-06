#!/usr/bin/env python3
"""Bind one InferenceX srt-slurm recipe to one local matrix point.

This is the local stand-in for what CI does between the matrix entry and the
worker process:

  infx.srt_slurm.single_node.select_recipe     exactly one recipe variant matches the point
  infx.srt_slurm.synthetic_acceptance          AgentX MTP runs get the golden acceptance
                                               (SGLANG_SIMULATE_ACC_*); runners/slurm_utils.sh
                                               apply_srt_recipe adds it at submission
  srtctl.backends.sglang.build_worker_command  frontend.type sglang (direct) -> launch_server argv

CI's selection, validation and override code is imported from the checkout and run
as-is. srtctl itself is imported when it can be; it needs marshmallow, pyarrow, mcp
and more, which the serving containers do not have, so its variant expansion and CLI
rendering fall back to the mirrors below.

  srt_point.py --e2e-dir DIR --recipe YAML --mode agent|fixed --model-path DIR \
               --port N --devices 0,1 --out-dir DIR

The matrix point comes from the environment the runner exports (TP, CONC, IMAGE,
ARM_MODEL, KV_OFFLOADING, ...). Prints `export` lines for the runner to eval.
"""

import argparse
import copy
import fnmatch
import json
import os
import re
import shlex
import sys
import types
from pathlib import Path

CONTAINER_WORKSPACE = "/infmax-workspace"


# Mirror of srtctl.core.config: deep_merge, expand_zip_override, generate_override_configs.
def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _list_lengths(d):
    out = []
    for v in d.values():
        if isinstance(v, list):
            out.append(len(v))
        elif isinstance(v, dict):
            out.extend(_list_lengths(v))
    return out


def _zip_length(zip_dict):
    lengths = _list_lengths(zip_dict)
    if not lengths:
        raise ValueError("zip_override section contains no list values — nothing to zip")
    if any(n == 0 for n in lengths):
        raise ValueError("zip_override contains an empty list — cannot zip zero-length lists")
    non_broadcast = {n for n in lengths if n != 1}
    if not non_broadcast:
        return 1
    if len(non_broadcast) > 1:
        raise ValueError(f"Incompatible zip lengths {sorted(non_broadcast)}")
    return non_broadcast.pop()


def _zip_slice(d, index):
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = v[0 if len(v) == 1 else index]
        elif isinstance(v, dict):
            out[k] = _zip_slice(v, index)
        else:
            out[k] = v
    return out


def _expand_zip(group, zip_dict, base):
    base_name = base.get("name", "unnamed")
    has_names = isinstance(zip_dict.get("name"), list)
    variants = []
    for i in range(_zip_length(zip_dict)):
        merged = deep_merge(base, _zip_slice(zip_dict, i))
        if not has_names:
            merged["name"] = f"{base_name}_{group}_{i}"
        variants.append((f"{group}_{i}", merged))
    return variants


def _expand_override(key, raw, base):
    suffix = key[len("override_"):]
    merged = deep_merge(base, raw[key])
    if "name" not in raw[key]:
        merged["name"] = f"{base.get('name', 'unnamed')}_{suffix}"
    return [(suffix, merged)]


def generate_override_configs(raw, selector=None):
    base = raw["base"]
    overrides = sorted(k for k in raw if k.startswith("override_"))
    zips = sorted(k for k in raw if k.startswith("zip_override_"))
    match = re.fullmatch(r"(zip_override_[\w-]+)\[(\d+)\]", selector or "")
    if match:
        group = match[1][len("zip_override_"):]
        variants = [_expand_zip(group, raw[match[1]], base)[int(match[2])]]
    elif selector == "base":
        variants = [("base", copy.deepcopy(base))]
    else:
        if selector is None:
            keys = overrides + zips
        elif "*" in selector or "?" in selector:
            keys = [k for k in sorted(overrides + zips) if fnmatch.fnmatch(k, selector)]
            if not keys:
                raise ValueError(f"No variants match '{selector}'")
        elif selector in raw:
            keys = [selector]
        else:
            raise ValueError(f"Override '{selector}' not found in config")
        variants = []
        for key in keys:
            if key.startswith("zip_override_"):
                variants.extend(_expand_zip(key[len("zip_override_"):], raw[key], base))
            else:
                variants.extend(_expand_override(key, raw, base))
    if "schema" in raw:
        for _, config in variants:
            config.setdefault("schema", raw["schema"])
    return variants


# Mirror of srtctl.backends.sglang._config_to_cli_args.
def config_to_cli_args(config):
    args = []
    for key, value in sorted(config.items()):
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag}")
        elif isinstance(value, list):
            args.append(f"--{flag}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag}", str(value)])
    return args


def import_ci_code(e2e_dir):
    """Import CI's selection and override modules; report which expander srtctl got."""
    sys.path.insert(0, str(e2e_dir))
    sys.path.insert(0, str(e2e_dir / "utils" / "srt-slurm" / "src"))
    expander = "srtctl"
    try:
        from srtctl.core.config import generate_override_configs as _  # noqa: F401
    except Exception:
        for name in [n for n in sys.modules if n == "srtctl" or n.startswith("srtctl.")]:
            del sys.modules[name]
        pkg, core = types.ModuleType("srtctl"), types.ModuleType("srtctl.core")
        pkg.__path__, core.__path__ = [], []
        cfg = types.ModuleType("srtctl.core.config")
        cfg.generate_override_configs = generate_override_configs
        sys.modules.update({"srtctl": pkg, "srtctl.core": core, "srtctl.core.config": cfg})
        expander = "mirror"
    from infx.srt_slurm.single_node import select_recipe
    from infx.srt_slurm.synthetic_acceptance import build_overrides
    return select_recipe, build_overrides, expander


def apply_overrides(recipe, overrides):
    """Apply srtctl-style ['--set', 'a.b=JSON', '--unset', 'a.b'] pairs to a recipe dict."""
    applied = []
    it = iter(overrides)
    for flag in it:
        operand = next(it)
        if flag == "--set":
            path, _, raw_value = operand.partition("=")
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            *parents, leaf = path.split(".")
            node = recipe
            for key in parents:
                node = node.setdefault(key, {})
            node[leaf] = value
        elif flag == "--unset":
            *parents, leaf = operand.split(".")
            node = recipe
            for key in parents:
                node = node.get(key) or {}
            node.pop(leaf, None)
        else:
            raise ValueError(f"unexpected override flag {flag!r}")
        applied.append(f"{flag} {operand}")
    return applied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e2e-dir", required=True, type=Path)
    p.add_argument("--recipe", required=True, help="srt-slurm recipe YAML, optionally PATH:selector")
    p.add_argument("--mode", required=True, choices=["agent", "fixed"])
    p.add_argument("--model-path", required=True, help="local weights; stands in for the hf: cache CI mounts")
    p.add_argument("--port", required=True)
    p.add_argument("--devices", required=True, help="GPU indices for the server, e.g. 0,1")
    p.add_argument("--visible-env", default="ROCR_VISIBLE_DEVICES",
                   help="visible_devices_env of runners/srt-slurm/<cluster>.yaml")
    p.add_argument("--no-synthetic-acceptance", action="store_true",
                   help="measure real draft acceptance; not comparable to the dashboard")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    select_recipe, build_overrides, expander = import_ci_code(args.e2e_dir)

    env = dict(os.environ)
    # The matrix MODEL is the HF id; the fixed-mode client swaps in the local path later.
    env["MODEL"] = env["ARM_MODEL"]
    variant, recipe = select_recipe(args.recipe, env)

    overrides = [] if args.no_synthetic_acceptance else build_overrides(recipe, env["FRAMEWORK"], env)
    applied = apply_overrides(recipe, overrides)

    role = recipe["roles"]["agg"]
    role_args = role.get("args") or {}
    config = dict(role_args)
    hf_id = env["ARM_MODEL"]
    served = config.get("served-model-name") or config.get("served_model_name")
    for key in ("model-path", "model_path", "served-model-name", "served_model_name",
                "nccl-port", "nccl_port"):
        config.pop(key, None)
    # CI resolves hf:<id> through the mounted HF cache; locally the weights dir is that cache.
    localized = {}
    for key, value in list(config.items()):
        if value == hf_id:
            config[key] = args.model_path
            localized[key] = args.model_path
    if served is None:
        served = Path(args.model_path).name
    if args.mode == "fixed" and served == hf_id:
        # The fixed client tokenizes with MODEL; keep it offline on the local dir.
        served = args.model_path
        localized["served-model-name"] = served

    argv = ["python3", "-m", "sglang.launch_server",
            "--model-path", args.model_path,
            "--served-model-name", served,
            "--host", "0.0.0.0",
            "--port", str(args.port)]
    if not any(k in role_args for k in ("enable-metrics", "enable_metrics")):
        argv.append("--enable-metrics")
    argv.extend(config_to_cli_args(config))

    # SRT applies the recipe-wide environment after the role environment.
    server_env = {**(role.get("env") or {}), **(recipe.get("environment") or {})}
    server_env = {str(k): str(v) for k, v in server_env.items()}
    server_env[args.visible_env] = args.devices
    unset = [v for v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
             if v != args.visible_env]

    bench = recipe["benchmark"]
    bench_env = {str(k): str(v) for k, v in (bench.get("env") or {}).items()}
    client = shlex.split(bench["command"].replace(CONTAINER_WORKSPACE, str(args.e2e_dir)))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "argv.json").write_text(json.dumps(argv, indent=1) + "\n")
    with open(out / "launch_server.sh", "w") as f:
        f.write(f"#!/usr/bin/env bash\n# {variant}\nexec env")
        f.write("".join(f" -u {v}" for v in unset))
        for key, value in server_env.items():
            f.write(f" \\\n  {key}={shlex.quote(value)}")
        f.write(" \\\n  " + " ".join(shlex.quote(a) for a in argv) + "\n")
    with open(out / "bench_env.sh", "w") as f:
        for key, value in bench_env.items():
            f.write(f"export {key}={shlex.quote(value)}\n")
    with open(out / "client.sh", "w") as f:
        f.write(f"#!/usr/bin/env bash\ncd {shlex.quote(str(args.e2e_dir))}\nexec "
                + " ".join(shlex.quote(a) for a in client) + "\n")
    point = {
        "variant": variant,
        "expander": expander,
        "synthetic_acceptance": not args.no_synthetic_acceptance,
        "overrides_applied": applied,
        "localized": localized,
        "server_env": server_env,
        "argv": argv,
        "client": client,
        "bench_env": bench_env,
    }
    (out / "point.json").write_text(json.dumps(point, indent=1) + "\n")

    print(f"export SRT_VARIANT={shlex.quote(variant)}")
    print(f"export SRT_EXPANDER={expander}")
    print(f"export SRT_OVERRIDES={shlex.quote('; '.join(applied) or 'none')}")


if __name__ == "__main__":
    main()
