#!/usr/bin/env python3
"""Search InferenceX benchmark results across multiple GitHub Actions runs.

Finds benchmark data by model, hardware platform, and framework, then renders
comprehensive tables with all metrics, Docker image, and script info for
reproducibility.

Usage:
    search_bmk.py --model dsv4 --hw mi355x --framework sglang
    search_bmk.py --model dsv4 --hw b200,mi355x --framework sglang,vllm,atom --compare
    search_bmk.py --model dsr1 --hw mi355x --all-metrics
    search_bmk.py --model dsv4 --hw b200 --isl 8192 --osl 1024 --show-script
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


REPO = "SemiAnalysisAI/InferenceX"
API_ROOT = "https://api.github.com"

HW_ALIASES = {
    "mi355": "mi355x",
    "mi300": "mi300x",
    "mi325": "mi325x",
}

KNOWN_TITLE_TAGS = {"atom", "sglang", "vllm", "trt", "trtllm", "tensorrt",
                     "mi355", "mi35x", "mi300", "mi325", "b200", "b300", "h200", "gb300"}

HW_TITLE_ALIASES = {
    "mi355x": ["mi355x", "mi355", "mi35x"],
    "mi300x": ["mi300x", "mi300"],
    "mi325x": ["mi325x", "mi325"],
    "b200": ["b200"],
    "b300": ["b300"],
    "h200": ["h200"],
    "gb300": ["gb300", "gb200"],
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    for env in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    cred = Path.home() / ".git-credentials"
    if cred.exists():
        for line in cred.read_text().splitlines():
            m = re.search(r"https://[^:]+:([^@]+)@github\.com", line)
            if m:
                return m.group(1)
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    sys.exit("No GitHub token found (set GH_TOKEN, populate ~/.git-credentials, or run gh auth login)")


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def gh_get(url: str, token: str) -> bytes:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "inferencex-table-skill",
    }
    no_redirect = urllib.request.build_opener(_NoRedirect)
    try:
        with no_redirect.open(urllib.request.Request(url, headers=headers)) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers.get("Location")
            if loc:
                with urllib.request.urlopen(
                    urllib.request.Request(loc, headers={"User-Agent": "inferencex-table-skill"})
                ) as r:
                    return r.read()
        raise RuntimeError(f"GitHub API error {e.code}: {e.read().decode(errors='replace')[:300]}")


def gh_get_json(url: str, token: str) -> dict:
    return json.loads(gh_get(url, token))


# ---------------------------------------------------------------------------
# Find runs via gh CLI
# ---------------------------------------------------------------------------

def find_runs(branch: str | None, limit: int) -> list[dict]:
    fetch_limit = max(limit * 50, 1000)
    cmd = [
        "gh", "run", "list",
        "-R", REPO,
        "--workflow=Run Sweep",
        f"--limit={fetch_limit}",
        "--json", "databaseId,displayTitle,headBranch,event,conclusion,createdAt,url",
    ]
    if branch:
        cmd.extend(["-b", branch])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"gh run list failed: {result.stderr.strip()}")
    runs = json.loads(result.stdout)
    return [r for r in runs if r.get("conclusion") in ("success", "failure", "cancelled", "")]


MODEL_TITLE_ALIASES = {
    "dsv4": ["dsv4", "deepseek-v4", "deepseekv4", "dsv4-pro"],
    "dsr1": ["dsr1", "deepseek-r1"],
    "qwen3.5": ["qwen3.5", "qwen3"],
    "glm5": ["glm5", "glm-5"],
    "kimik2.5": ["kimik2.5", "kimi-k2.5", "kimi"],
    "gptoss": ["gptoss", "gpt-oss"],
    "minimax": ["minimax"],
}


def _model_title_tags(model_query: str | None) -> set[str]:
    if not model_query:
        return set()
    q = model_query.lower().strip()
    for key, aliases in MODEL_TITLE_ALIASES.items():
        if q == key or q in aliases:
            return set(aliases)
    return {q}


def prefilter_runs(
    runs: list[dict],
    hw_list: list[str] | None,
    fw_list: list[str] | None,
    model: str | None,
    scan_limit: int,
) -> list[dict]:
    """Pre-filter runs by title keywords to avoid downloading irrelevant artifacts.

    Priority order:
    1. Runs whose title matches framework + model (strongest signal)
    2. Runs whose title matches framework (good signal)
    3. Runs with no recognizable tags (could contain anything)
    4. Runs that match hw but not framework (fallback)
    """
    if not hw_list and not fw_list:
        return runs[:scan_limit]

    fw_tags = set(fw_list) if fw_list else set()
    hw_tags = set()
    if hw_list:
        for h in hw_list:
            if h in HW_TITLE_ALIASES:
                hw_tags.update(HW_TITLE_ALIASES[h])
            else:
                hw_tags.add(h.rstrip("x"))
                hw_tags.add(h)
    model_tags = _model_title_tags(model)

    priority = []
    for r in runs:
        title_lower = r.get("displayTitle", "").lower()
        has_fw = bool(fw_tags) and any(tag in title_lower for tag in fw_tags)
        has_hw = bool(hw_tags) and any(tag in title_lower for tag in hw_tags)
        has_model = bool(model_tags) and any(tag in title_lower for tag in model_tags)
        has_any_known = any(tag in title_lower for tag in KNOWN_TITLE_TAGS)

        if has_fw and has_hw and has_model:
            priority.append((0, r))
        elif has_fw and has_model:
            priority.append((1, r))
        elif has_hw and has_model:
            priority.append((1, r))
        elif has_fw and has_hw:
            priority.append((1, r))
        elif has_fw:
            priority.append((2, r))
        elif has_hw:
            priority.append((2, r))
        elif not has_any_known:
            priority.append((3, r))

    priority.sort(key=lambda x: x[0])
    return [r for _, r in priority[:scan_limit]]


# ---------------------------------------------------------------------------
# Download and parse agg_bmk.json
# ---------------------------------------------------------------------------

def find_artifact(repo: str, run_id: str, token: str, name: str) -> dict | None:
    page = 1
    while True:
        data = gh_get_json(
            f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}",
            token,
        )
        for art in data.get("artifacts", []):
            if art["name"] == name:
                return art
        if len(data.get("artifacts", [])) < 100:
            return None
        page += 1


def download_agg_bmk(run_id: str, token: str, dest: Path) -> Path | None:
    art = find_artifact(REPO, run_id, token, "results_bmk")
    if not art:
        return None
    zip_bytes = gh_get(art["archive_download_url"], token)
    zpath = dest / f"results_bmk_{run_id}.zip"
    zpath.write_bytes(zip_bytes)
    extract_dir = dest / f"run_{run_id}"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract_dir)
    agg = extract_dir / "agg_bmk.json"
    return agg if agg.exists() else None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def normalize_hw(hw: str) -> str:
    hw = hw.lower().strip()
    return HW_ALIASES.get(hw, hw)


def match_model(entry: dict, model_query: str) -> bool:
    q = model_query.lower()
    prefix = str(entry.get("infmax_model_prefix", "")).lower()
    full_model = str(entry.get("model", "")).lower()
    if prefix == q:
        return True
    if q in prefix or q in full_model:
        return True
    q_nodash = q.replace("-", "").replace("_", "")
    prefix_nodash = prefix.replace("-", "").replace("_", "")
    full_nodash = full_model.replace("-", "").replace("_", "")
    return q_nodash in prefix_nodash or q_nodash in full_nodash


def filter_entries(
    entries: list[dict],
    model: str | None,
    hw_list: list[str] | None,
    fw_list: list[str] | None,
    precision: str | None,
    isl: int | None,
    osl: int | None,
) -> list[dict]:
    result = []
    for e in entries:
        if model and not match_model(e, model):
            continue
        if hw_list and normalize_hw(str(e.get("hw", ""))) not in hw_list:
            continue
        if fw_list and str(e.get("framework", "")).lower() not in fw_list:
            continue
        if precision and str(e.get("precision", "")).lower() != precision.lower():
            continue
        if isl is not None and e.get("isl") != isl:
            continue
        if osl is not None and e.get("osl") != osl:
            continue
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# Scan runs
# ---------------------------------------------------------------------------

def group_key(entry: dict) -> str:
    return f"{entry.get('hw', '?')}|{entry.get('framework', '?')}|{entry.get('precision', '?')}"


def scan_runs(
    runs: list[dict],
    token: str,
    model: str | None,
    hw_list: list[str] | None,
    fw_list: list[str] | None,
    precision: str | None,
    isl: int | None,
    osl: int | None,
) -> dict:
    """Scan runs newest-first. Return {group_key: {entries, run_meta}} for each combo found."""
    groups: dict[str, dict] = {}
    wanted_combos = set()
    if hw_list and fw_list:
        for h in hw_list:
            for f in fw_list:
                wanted_combos.add((h, f))

    with tempfile.TemporaryDirectory() as tmpdir:
        for run in runs:
            run_id = str(run["databaseId"])
            print(f"  Scanning run {run_id} ({run.get('displayTitle', '')[:60]})...", file=sys.stderr)

            try:
                agg_path = download_agg_bmk(run_id, token, Path(tmpdir))
            except Exception as exc:
                print(f"    Failed to download artifact: {exc}", file=sys.stderr)
                continue
            if not agg_path:
                print(f"    No results_bmk artifact found, skipping.", file=sys.stderr)
                continue

            entries = json.loads(agg_path.read_text())
            matched = filter_entries(entries, model, hw_list, fw_list, precision, isl, osl)

            if not matched:
                print(f"    No matching entries.", file=sys.stderr)
                continue

            print(f"    Found {len(matched)} matching entries.", file=sys.stderr)

            run_meta = {
                "run_id": run_id,
                "title": run.get("displayTitle", ""),
                "branch": run.get("headBranch", ""),
                "created_at": run.get("createdAt", ""),
                "url": f"https://github.com/{REPO}/actions/runs/{run_id}",
            }

            for e in matched:
                gk = group_key(e)
                if gk not in groups:
                    groups[gk] = {"entries": [], "run_meta": run_meta}
                    for me in matched:
                        if group_key(me) == gk:
                            groups[gk]["entries"].append(me)

            if wanted_combos:
                found_combos = set()
                for gk in groups:
                    parts = gk.split("|")
                    found_combos.add((parts[0], parts[1]))
                if wanted_combos.issubset(found_combos):
                    print(f"  All requested combos found, stopping scan.", file=sys.stderr)
                    break

    return groups


# ---------------------------------------------------------------------------
# Script / image info
# ---------------------------------------------------------------------------

def derive_script_candidates(entry: dict) -> list[str]:
    """Return candidate script paths in priority order."""
    prefix = entry.get("infmax_model_prefix", "unknown")
    prec = entry.get("precision", "unknown")
    hw = entry.get("hw", "unknown")
    fw = entry.get("framework", "unknown")
    spec = entry.get("spec_decoding", "none")

    base = f"benchmarks/single_node/{prefix}_{prec}_{hw}"
    spec_suffix = f"_{spec}" if spec and spec != "none" else ""

    candidates = []
    if fw == "sglang":
        candidates.append(f"{base}_sglang{spec_suffix}.sh")
        candidates.append(f"{base}{spec_suffix}.sh")
    else:
        candidates.append(f"{base}_{fw}{spec_suffix}.sh")
    candidates.append(f"{base}_{fw}.sh")
    candidates.append(f"{base}.sh")
    return candidates


def derive_script_path(entry: dict) -> str:
    return derive_script_candidates(entry)[0]


def derive_script_url(entry: dict) -> str:
    path = derive_script_path(entry)
    return f"https://github.com/{REPO}/blob/main/{path}"


def resolve_script_path(entry: dict, token: str) -> tuple[str | None, str | None]:
    """Try candidate paths, return (path, url) for the first one that exists."""
    import base64
    for path in derive_script_candidates(entry):
        try:
            gh_get_json(f"{API_ROOT}/repos/{REPO}/contents/{path}", token)
            url = f"https://github.com/{REPO}/blob/main/{path}"
            return path, url
        except Exception:
            continue
    return derive_script_path(entry), derive_script_url(entry)


def fetch_script_content(entry: dict, token: str) -> str | None:
    import base64
    for path in derive_script_candidates(entry):
        try:
            data = gh_get_json(f"{API_ROOT}/repos/{REPO}/contents/{path}", token)
            return base64.b64decode(data["content"]).decode()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def ms(x):
    return None if x is None else x * 1000.0


def fmt(v, prec=2):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def render_detail_table(entries: list[dict], all_metrics: bool = False) -> str:
    entries = sorted(entries, key=lambda e: (
        e.get("conc", 0),
        0 if str(e.get("dp_attention", "")).lower() == "false" else 1,
    ))

    if all_metrics:
        header = (
            "| Conc | TP, DP | TTT (tok/s) | Out TPut/GPU | "
            "Med E2EL | Med TTFT | Med ITL | Med TPOT | "
            "p90 TTFT | p99 TTFT | p90 ITL | p99 ITL | p90 E2EL | p99 E2EL | "
            "Avg Power (W) | J/out_tok |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    else:
        header = (
            "| Conc | TP, DP | TTT (tok/s) | Out TPut/GPU | "
            "Med E2EL (ms) | Med TTFT (ms) | Med ITL (ms) | Med TPOT (ms) |"
        )
        sep = "|---|---|---|---|---|---|---|---|"

    lines = [header, sep]
    for e in entries:
        dp = e["tp"] if str(e.get("dp_attention", "")).lower() in ("true",) else 1
        ttt = e.get("tput_per_gpu", 0) * e.get("tp", 1)
        out_tput = e.get("output_tput_per_gpu", None)

        row = (
            f"| {e.get('conc', '?')} "
            f"| {e.get('tp', '?')}, {dp} "
            f"| {fmt(ttt)} "
            f"| {fmt(out_tput)} "
            f"| {fmt(ms(e.get('median_e2el')))} "
            f"| {fmt(ms(e.get('median_ttft')))} "
            f"| {fmt(ms(e.get('median_itl')))} "
            f"| {fmt(ms(e.get('median_tpot')))} "
        )

        if all_metrics:
            row += (
                f"| {fmt(ms(e.get('p90_ttft')))} "
                f"| {fmt(ms(e.get('p99_ttft')))} "
                f"| {fmt(ms(e.get('p90_itl')))} "
                f"| {fmt(ms(e.get('p99_itl')))} "
                f"| {fmt(ms(e.get('p90_e2el')))} "
                f"| {fmt(ms(e.get('p99_e2el')))} "
                f"| {fmt(e.get('avg_power_w'))} "
                f"| {fmt(e.get('joules_per_output_token'))} "
            )

        row += "|"
        lines.append(row)

    return "\n".join(lines)


def render_compare_table(groups: dict, isl: int | None, osl: int | None) -> str:
    all_entries = []
    for gk, gdata in groups.items():
        for e in gdata["entries"]:
            if isl is not None and e.get("isl") != isl:
                continue
            if osl is not None and e.get("osl") != osl:
                continue
            e["_run_meta"] = gdata["run_meta"]
            all_entries.append(e)

    all_entries.sort(key=lambda e: (
        str(e.get("hw", "")),
        str(e.get("framework", "")),
        e.get("conc", 0),
    ))

    header = (
        "| Platform | Framework | Prec | Conc | TP, DP | TTT (tok/s) | Out TPut/GPU | "
        "Med E2EL (ms) | Med TTFT (ms) | Med ITL (ms) | Med TPOT (ms) | Image |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]

    for e in all_entries:
        dp = e["tp"] if str(e.get("dp_attention", "")).lower() == "true" else 1
        ttt = e.get("tput_per_gpu", 0) * e.get("tp", 1)
        img = str(e.get("image", ""))
        img_short = img.split("/")[-1] if "/" in img else img
        if len(img_short) > 40:
            img_short = img_short[:37] + "..."

        lines.append(
            f"| {e.get('hw', '?')} "
            f"| {e.get('framework', '?')} "
            f"| {e.get('precision', '?')} "
            f"| {e.get('conc', '?')} "
            f"| {e.get('tp', '?')}, {dp} "
            f"| {fmt(ttt)} "
            f"| {fmt(e.get('output_tput_per_gpu'))} "
            f"| {fmt(ms(e.get('median_e2el')))} "
            f"| {fmt(ms(e.get('median_ttft')))} "
            f"| {fmt(ms(e.get('median_itl')))} "
            f"| {fmt(ms(e.get('median_tpot')))} "
            f"| `{img_short}` |"
        )

    return "\n".join(lines)


def render_group(gk: str, gdata: dict, all_metrics: bool, show_script: bool, token: str | None) -> str:
    parts = gk.split("|")
    hw, fw, prec = parts[0], parts[1], parts[2]
    entries = gdata["entries"]
    run_meta = gdata["run_meta"]

    model_name = entries[0].get("model", "unknown") if entries else "unknown"
    image = entries[0].get("image", "unknown") if entries else "unknown"

    combos = sorted({(e.get("isl"), e.get("osl")) for e in entries})
    lines = []
    lines.append(f"## {model_name} on {hw} / {fw} ({prec})")
    lines.append("")
    script_path, script_url = resolve_script_path(entries[0], token) if token else (derive_script_path(entries[0]), derive_script_url(entries[0]))
    lines.append(f"- **Image**: `{image}`")
    lines.append(f"- **Script**: [`{script_path}`]({script_url})")
    lines.append(f"- **Run**: [{run_meta['run_id']}]({run_meta['url']}) ({run_meta['created_at'][:10]})")

    if entries:
        e0 = entries[0]
        spec = e0.get("spec_decoding", "none")
        if spec and spec != "none":
            lines.append(f"- **Speculative decoding**: {spec}")

    lines.append("")

    if show_script and token:
        content = fetch_script_content(entries[0], token)
        if content:
            lines.append("<details><summary>Benchmark script</summary>")
            lines.append("")
            lines.append("```bash")
            lines.append(content.rstrip())
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    for isl_val, osl_val in combos:
        combo_entries = [e for e in entries if e.get("isl") == isl_val and e.get("osl") == osl_val]
        if not combo_entries:
            continue
        lines.append(f"### ISL={isl_val} / OSL={osl_val}")
        lines.append("")
        lines.append(render_detail_table(combo_entries, all_metrics))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reproduce info
# ---------------------------------------------------------------------------

def render_reproduce_info(entry: dict) -> str:
    lines = []
    lines.append("### Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# Docker image")
    lines.append(f"docker pull {entry.get('image', 'UNKNOWN')}")
    lines.append(f"")
    lines.append(f"# Key parameters")
    lines.append(f"#   Model: {entry.get('model', '?')}")
    lines.append(f"#   TP={entry.get('tp', '?')}, EP={entry.get('ep', '?')}, DP-Attention={entry.get('dp_attention', '?')}")
    lines.append(f"#   ISL={entry.get('isl', '?')}, OSL={entry.get('osl', '?')}")
    lines.append(f"#   Precision: {entry.get('precision', '?')}")
    lines.append(f"#   Concurrency: {entry.get('conc', '?')}")
    if entry.get("spec_decoding") and entry["spec_decoding"] != "none":
        lines.append(f"#   Spec decoding: {entry['spec_decoding']}")
    lines.append(f"")
    lines.append(f"# Benchmark script (in InferenceX repo)")
    lines.append(f"#   {derive_script_path(entry)}")
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Search InferenceX benchmark results by model/hardware/framework."
    )
    p.add_argument("--model", help="Model prefix or substring (e.g. dsv4, dsr1, deepseek-v4)")
    p.add_argument("--hw", help="Hardware platform(s), comma-separated (e.g. b200,mi355x)")
    p.add_argument("--framework", help="Framework(s), comma-separated (e.g. sglang,vllm,atom)")
    p.add_argument("--precision", help="Precision filter (e.g. fp8, fp4)")
    p.add_argument("--isl", type=int, help="Input sequence length filter")
    p.add_argument("--osl", type=int, help="Output sequence length filter")
    p.add_argument("--branch", help="Git branch to search (default: scan all recent runs)")
    p.add_argument("--limit", type=int, default=30, help="Max number of title-matched runs to scan (default: 30)")
    p.add_argument("--all-metrics", action="store_true", help="Show expanded latency columns (p90, p99, power)")
    p.add_argument("--show-script", action="store_true", help="Fetch and display the benchmark script content")
    p.add_argument("--compare", action="store_true", help="Side-by-side comparison table across platforms")
    p.add_argument("--reproduce", action="store_true", help="Show reproduce info for each combo")
    p.add_argument("--token", help="GitHub PAT (overrides env / git-credentials)")
    p.add_argument("--out", help="Write output to file")
    p.add_argument("--json", action="store_true", help="Also dump raw matched entries as JSON")
    args = p.parse_args()

    if not any([args.model, args.hw, args.framework]):
        p.error("At least one of --model, --hw, or --framework is required")

    token = resolve_token(args.token)

    hw_list = [normalize_hw(h) for h in args.hw.split(",")] if args.hw else None
    fw_list = [f.strip().lower() for f in args.framework.split(",")] if args.framework else None

    print(f"Searching InferenceX runs...", file=sys.stderr)
    all_runs = find_runs(args.branch, args.limit)
    if not all_runs:
        sys.exit("No completed sweep runs found.")

    runs = prefilter_runs(all_runs, hw_list, fw_list, args.model, args.limit)
    print(f"Found {len(all_runs)} completed runs, {len(runs)} match title filter, scanning up to {args.limit}.", file=sys.stderr)

    groups = scan_runs(runs, token, args.model, hw_list, fw_list, args.precision, args.isl, args.osl)

    if not groups:
        sys.exit("No matching benchmark entries found across scanned runs.")

    out_lines = []

    if args.compare and len(groups) > 1:
        model_name = ""
        for gdata in groups.values():
            if gdata["entries"]:
                model_name = gdata["entries"][0].get("model", "")
                break

        all_combos = set()
        for gdata in groups.values():
            for e in gdata["entries"]:
                all_combos.add((e.get("isl"), e.get("osl")))

        for isl_val, osl_val in sorted(all_combos):
            out_lines.append(f"# ISL={isl_val} / OSL={osl_val} — {model_name}")
            out_lines.append("")

            for gk, gdata in sorted(groups.items()):
                parts = gk.split("|")
                entries = gdata["entries"]
                run_meta = gdata["run_meta"]
                if entries:
                    img = entries[0].get("image", "?")
                    out_lines.append(f"- **{parts[0]} / {parts[1]} ({parts[2]})**: image=`{img}`, "
                                     f"script=[`{derive_script_path(entries[0])}`]({derive_script_url(entries[0])}), "
                                     f"run=[{run_meta['run_id']}]({run_meta['url']})")

            out_lines.append("")
            out_lines.append(render_compare_table(groups, isl_val, osl_val))
            out_lines.append("")
    else:
        for gk in sorted(groups):
            out_lines.append(render_group(gk, groups[gk], args.all_metrics, args.show_script, token))

            if args.reproduce and groups[gk]["entries"]:
                out_lines.append(render_reproduce_info(groups[gk]["entries"][0]))
                out_lines.append("")

    text = "\n".join(out_lines).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(text)

    if args.json:
        all_matched = []
        for gdata in groups.values():
            all_matched.extend(gdata["entries"])
        json_path = Path(tempfile.mktemp(suffix=".json"))
        json_path.write_text(json.dumps(all_matched, indent=2))
        print(f"\n# raw json: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
