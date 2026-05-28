#!/usr/bin/env python3
"""Fetch ATOM benchmark dashboard data and report performance trends.

Detects >5% improvements, identifies contributing commits, and flags
changes potentially useful to sglang.

Usage:
    atom_progress.py [--model dsv4] [--isl 8192] [--osl 1024]
                     [--backend ATOM] [--runs N] [--threshold 5]
                     [--json] [--commits]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone

DATA_URL = "https://rocm.github.io/ATOM/benchmark-dashboard/data.js"
ATOM_REPO = "https://github.com/ROCm/ATOM"

MODEL_ALIASES = {
    "dsv4": "DeepSeek-V4-Pro",
    "deepseekv4": "DeepSeek-V4-Pro",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "deepseek-v4": "DeepSeek-V4-Pro",
    "dsr1": "DeepSeek-R1-0528",
    "deepseek-r1": "DeepSeek-R1-0528",
    "qwen3.5": "Qwen3.5-397B",
    "qwen": "Qwen3.5-397B",
    "glm5": "GLM-5",
    "glm5.1": "GLM-5.1",
    "kimik2.5": "Kimi-K2.5",
    "kimi": "Kimi-K2.5",
    "gptoss": "gpt-oss-120b",
    "minimax": "MiniMax-M2.7",
}

BACKEND_ALIASES = {
    "atom": "ATOM",
    "sglang": "ATOM-SGLang",
    "atom-sglang": "ATOM-SGLang",
    "vllm": "ATOM-vLLM",
    "atom-vllm": "ATOM-vLLM",
}

MODEL_RECIPES = {
    "DeepSeek-V4-Pro": "DeepSeek-V4.md",
    "DeepSeek-R1-0528": "DeepSeek-R1.md",
    "Qwen3.5-397B": "Qwen3.5.md",
    "GLM-5": "GLM-5.md",
    "Kimi-K2.5": "Kimi-K2.5.md",
    "gpt-oss-120b": "GPT-OSS.md",
    "MiniMax-M2.7": None,
}

SGLANG_RELEVANT_KEYWORDS = [
    "attention", "kv_cache", "kv cache", "scheduler", "batch",
    "prefill", "decode", "gemm", "moe", "expert", "quantiz",
    "fused", "flash", "triton", "radix", "paged", "chunked",
    "speculative", "mtp", "draft", "memory", "allocat",
    "overlap", "pipeline", "dispatch", "routing", "topk",
    "norm", "layernorm", "rmsnorm", "rope", "embedding",
    "cuda graph", "graph capture", "cudagraph",
    "fp8", "fp4", "int4", "int8", "mxfp4", "block scale",
]


def fetch_data() -> dict:
    print("Fetching ATOM dashboard data...", file=sys.stderr)
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "atom-progress-skill"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
    raw = raw.strip()
    if raw.startswith("window.BENCHMARK_DATA"):
        raw = raw.split("=", 1)[1].strip()
    if raw.endswith(";"):
        raw = raw[:-1]
    return json.loads(raw)


BENCH_RE = re.compile(
    r"^(.+?)::(.+?)\s+(\d+)/(\d+)\s+c=(\d+)\s+(.+?)\s*\((.+)\)$"
)


def parse_bench_name(name: str) -> dict | None:
    m = BENCH_RE.match(name)
    if not m:
        return None
    return {
        "backend": m.group(1),
        "model": m.group(2),
        "isl": int(m.group(3)),
        "osl": int(m.group(4)),
        "conc": int(m.group(5)),
        "metric": m.group(6),
        "unit": m.group(7),
    }


def parse_extra(extra: str) -> dict:
    result = {}
    for part in extra.split("|"):
        part = part.strip()
        if ":" in part:
            key, val = part.split(":", 1)
            result[key.strip()] = val.strip()
    return result


def resolve_model(query: str) -> str:
    q = query.lower().strip()
    if q in MODEL_ALIASES:
        return MODEL_ALIASES[q]
    for alias, display in MODEL_ALIASES.items():
        if q in alias or q in display.lower():
            return display
    return query


def resolve_backend(query: str) -> str:
    q = query.lower().strip()
    return BACKEND_ALIASES.get(q, query)


def build_history(data: dict, model: str, isl: int, osl: int, backend: str) -> list[dict]:
    """Build time-ordered history for a specific config from dashboard data."""
    entries = data.get("entries", {}).get("Benchmark", [])
    history = []

    for run in entries:
        commit = run.get("commit", {})
        date = run.get("date", 0)
        benches = run.get("benches", [])

        metrics = {}
        extra_info = {}
        matched = False

        meta = {}
        for b in benches:
            name = b.get("name", "")
            parsed = parse_bench_name(name)
            if parsed:
                if parsed["backend"] != backend:
                    continue
                if model.lower() not in parsed["model"].lower():
                    continue
                if parsed["isl"] != isl or parsed["osl"] != osl:
                    continue
                matched = True
                key = (parsed["conc"], parsed["metric"])
                metrics[key] = b.get("value")
                if b.get("extra") and not extra_info:
                    extra_info = parse_extra(b["extra"])
            else:
                # metadata entries like _tp, _gpu_count (no unit in name)
                meta_re = re.match(
                    r"^(.+?)::(.+?)\s+(\d+)/(\d+)\s+c=(\d+)\s+(_\w+)$", name
                )
                if meta_re:
                    mb = meta_re.group(1)
                    mm = meta_re.group(2)
                    misl, mosl = int(meta_re.group(3)), int(meta_re.group(4))
                    if mb != backend or model.lower() not in mm.lower():
                        continue
                    if misl != isl or mosl != osl:
                        continue
                    meta[meta_re.group(6)] = b.get("value")

        if matched:
            history.append({
                "commit": commit,
                "date": date,
                "date_str": datetime.fromtimestamp(date / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "metrics": metrics,
                "meta": meta,
                "extra": extra_info,
                "docker": extra_info.get("Docker", ""),
                "run_url": extra_info.get("Run", ""),
            })

    history.sort(key=lambda h: h["date"], reverse=True)
    return history


def detect_changes(history: list[dict], threshold: float) -> list[dict]:
    """Compare consecutive runs and detect significant changes."""
    changes = []
    for i in range(len(history) - 1):
        curr = history[i]
        prev = history[i + 1]

        for key in curr["metrics"]:
            if key not in prev["metrics"]:
                continue
            conc, metric = key
            curr_val = curr["metrics"][key]
            prev_val = prev["metrics"][key]
            if prev_val is None or prev_val == 0 or curr_val is None:
                continue

            if metric in ("throughput", "Total Tput"):
                pct = (curr_val - prev_val) / prev_val * 100
                improved = pct > 0
            else:
                pct = (prev_val - curr_val) / prev_val * 100
                improved = pct > 0

            if abs(pct) >= threshold:
                changes.append({
                    "date": curr["date_str"],
                    "commit": curr["commit"],
                    "conc": conc,
                    "metric": metric,
                    "prev_val": prev_val,
                    "curr_val": curr_val,
                    "pct": pct,
                    "improved": improved,
                    "docker_curr": curr.get("docker", ""),
                    "docker_prev": prev.get("docker", ""),
                    "run_url": curr.get("run_url", ""),
                })

    return changes


def check_sglang_relevance(commit_msg: str) -> list[str]:
    msg_lower = commit_msg.lower()
    hits = []
    for kw in SGLANG_RELEVANT_KEYWORDS:
        if kw in msg_lower:
            hits.append(kw)
    return hits


def fmt(v, prec=2):
    if v is None:
        return "-"
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else str(v)


def render_status(history: list[dict], model: str, isl: int, osl: int, backend: str) -> str:
    lines = []
    if not history:
        lines.append(f"No data found for {backend}::{model} {isl}/{osl}")
        return "\n".join(lines)

    latest = history[0]
    lines.append(f"## {backend}::{model} — ISL={isl} / OSL={osl}")
    lines.append("")
    lines.append(f"- **Latest run**: {latest['date_str']}")
    if latest.get("run_url"):
        lines.append(f"- **CI run**: {latest['run_url']}")
    lines.append(f"- **Commit**: `{latest['commit'].get('id', '?')[:12]}` — {latest['commit'].get('message', '?').split(chr(10))[0][:80]}")
    lines.append(f"- **History depth**: {len(history)} runs")
    lines.append("")

    docker_img = latest.get("docker", "")
    recipe_file = MODEL_RECIPES.get(model)
    lines.append("### Reproduce")
    lines.append("")
    if docker_img:
        lines.append(f"```bash\ndocker pull {docker_img}\n```")
        lines.append("")
    if recipe_file:
        recipe_url = f"{ATOM_REPO}/blob/main/recipes/{recipe_file}"
        lines.append(f"- **Recipe**: [{recipe_file}]({recipe_url})")
    lines.append(f"- **Benchmark workflow**: [`atom-benchmark.yaml`]({ATOM_REPO}/blob/main/.github/workflows/atom-benchmark.yaml)")
    if docker_img:
        lines.append(f"- **Image**: `{docker_img}`")
    lines.append("")

    tp = int(latest.get("meta", {}).get("_tp", 8))
    concs = sorted({k[0] for k in latest["metrics"]})

    lines.append("### Latest Numbers")
    lines.append("")
    lines.append("| Concurrency | TP, DP | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for c in concs:
        tput_total = latest["metrics"].get((c, "Total Tput"))
        out_tput = latest["metrics"].get((c, "throughput"))
        if tput_total is None and out_tput is not None:
            tput_total = out_tput * tp
        tpot = latest["metrics"].get((c, "TPOT"))
        ttft = latest["metrics"].get((c, "TTFT"))
        # E2EL not directly available from ATOM; approximate as TTFT + OSL * TPOT
        e2el = None
        if ttft is not None and tpot is not None:
            e2el = ttft + osl * tpot
        lines.append(f"| {c} | {tp}, 1 | {fmt(tput_total)} | {fmt(e2el)} | {fmt(ttft)} | {fmt(tpot)} |")
    lines.append("")

    return "\n".join(lines)


def render_trend(history: list[dict], runs: int) -> str:
    lines = []
    if len(history) < 2:
        return ""

    window = history[:runs]
    concs = sorted({k[0] for k in window[0]["metrics"]})
    tput_metric = "Total Tput" if any(k[1] == "Total Tput" for k in window[0]["metrics"]) else "throughput"

    lines.append("### Trend (recent runs)")
    lines.append("")

    header = "| Date | Docker |"
    for c in concs:
        header += f" c={c} {tput_metric} |"
    lines.append(header)
    sep = "|---|---|" + "---|" * len(concs)
    lines.append(sep)

    for h in window:
        tag = h.get("docker", "").split(":")[-1] if h.get("docker") else "?"
        if len(tag) > 25:
            tag = tag[:22] + "..."
        row = f"| {h['date_str']} | `{tag}` |"
        for c in concs:
            val = h["metrics"].get((c, tput_metric))
            row += f" {fmt(val)} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def render_changes(changes: list[dict], threshold: float) -> str:
    lines = []
    improvements = [c for c in changes if c["improved"]]
    regressions = [c for c in changes if not c["improved"]]

    if improvements:
        lines.append(f"### Improvements (>{threshold}%)")
        lines.append("")
        for ch in sorted(improvements, key=lambda x: -abs(x["pct"])):
            direction = "+" if ch["pct"] > 0 else ""
            commit_sha = ch["commit"].get("id", "?")[:12]
            commit_msg = ch["commit"].get("message", "?").split("\n")[0][:80]
            commit_url = ch["commit"].get("url", "")

            lines.append(
                f"- **{ch['metric']} c={ch['conc']}**: {fmt(ch['prev_val'])} -> {fmt(ch['curr_val'])} "
                f"(**{direction}{ch['pct']:.1f}%**) on {ch['date']}"
            )
            lines.append(f"  - Commit: [`{commit_sha}`]({commit_url}) {commit_msg}")
            if ch.get("docker_curr") != ch.get("docker_prev"):
                lines.append(f"  - Image changed: `{ch.get('docker_prev', '?').split(':')[-1]}` -> `{ch.get('docker_curr', '?').split(':')[-1]}`")

            sglang_hits = check_sglang_relevance(
                ch["commit"].get("message", "")
            )
            if sglang_hits:
                lines.append(f"  - **Potentially relevant to SGLang** (keywords: {', '.join(sglang_hits)})")

            if ch.get("run_url"):
                lines.append(f"  - [CI Run]({ch['run_url']})")
            lines.append("")

    if regressions:
        lines.append(f"### Regressions (>{threshold}%)")
        lines.append("")
        for ch in sorted(regressions, key=lambda x: -abs(x["pct"])):
            direction = "" if ch["pct"] < 0 else "+"
            commit_sha = ch["commit"].get("id", "?")[:12]
            commit_msg = ch["commit"].get("message", "?").split("\n")[0][:80]
            commit_url = ch["commit"].get("url", "")

            lines.append(
                f"- **{ch['metric']} c={ch['conc']}**: {fmt(ch['prev_val'])} -> {fmt(ch['curr_val'])} "
                f"(**{direction}{ch['pct']:.1f}%**) on {ch['date']}"
            )
            lines.append(f"  - Commit: [`{commit_sha}`]({commit_url}) {commit_msg}")
            lines.append("")

    return "\n".join(lines)


def render_sglang_analysis(changes: list[dict]) -> str:
    lines = []
    improvements = [c for c in changes if c["improved"]]
    if not improvements:
        return ""

    relevant = []
    for ch in improvements:
        msg = ch["commit"].get("message", "")
        hits = check_sglang_relevance(msg)
        if hits:
            relevant.append((ch, hits))

    if not relevant:
        lines.append("### SGLang Relevance")
        lines.append("")
        lines.append("No commits with obvious SGLang-applicable changes detected in improvements.")
        lines.append("Changes may be ATOM-internal (custom runtime, custom kernels).")
        lines.append("")
        return "\n".join(lines)

    lines.append("### SGLang Relevance")
    lines.append("")
    lines.append("The following improvement commits touch areas that may benefit SGLang:")
    lines.append("")
    for ch, hits in relevant:
        sha = ch["commit"].get("id", "?")[:12]
        msg = ch["commit"].get("message", "").split("\n")[0][:80]
        url = ch["commit"].get("url", "")
        lines.append(f"- [`{sha}`]({url}) {msg}")
        lines.append(f"  - Keywords: {', '.join(hits)}")
        lines.append(f"  - Impact: {ch['metric']} c={ch['conc']} **{ch['pct']:+.1f}%**")
        lines.append("")

    return "\n".join(lines)


def list_available(data: dict) -> str:
    entries = data.get("entries", {}).get("Benchmark", [])
    if not entries:
        return "No benchmark data found."

    configs = set()
    for run in entries[:3]:
        for b in run.get("benches", []):
            parsed = parse_bench_name(b.get("name", ""))
            if parsed and parsed["metric"] in ("throughput", "Total Tput"):
                configs.add((parsed["backend"], parsed["model"], f"{parsed['isl']}/{parsed['osl']}"))

    lines = ["## Available configs (from latest 3 runs)", ""]
    lines.append("| Backend | Model | ISL/OSL |")
    lines.append("|---|---|---|")
    for backend, model, islOsl in sorted(configs):
        lines.append(f"| {backend} | {model} | {islOsl} |")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Track ATOM benchmark progress.")
    p.add_argument("--model", default="dsv4", help="Model name or alias (default: dsv4)")
    p.add_argument("--isl", type=int, default=8192, help="Input sequence length (default: 8192)")
    p.add_argument("--osl", type=int, default=1024, help="Output sequence length (default: 1024)")
    p.add_argument("--backend", default="ATOM", help="Backend: ATOM, ATOM-SGLang, ATOM-vLLM (default: ATOM)")
    p.add_argument("--runs", type=int, default=10, help="Number of recent runs to analyze (default: 10)")
    p.add_argument("--threshold", type=float, default=5.0, help="Percent change threshold (default: 5.0)")
    p.add_argument("--list", action="store_true", help="List available model/backend/ISL/OSL configs")
    p.add_argument("--json", action="store_true", help="Dump raw history as JSON")
    p.add_argument("--commits", action="store_true", help="Show commit details for all runs")
    args = p.parse_args()

    data = fetch_data()

    if args.list:
        print(list_available(data))
        return

    model = resolve_model(args.model)
    backend = resolve_backend(args.backend)

    print(f"Analyzing {backend}::{model} {args.isl}/{args.osl}...", file=sys.stderr)

    history = build_history(data, model, args.isl, args.osl, backend)

    if args.json:
        serializable = []
        for h in history:
            entry = dict(h)
            entry["metrics"] = {f"{c}:{m}": v for (c, m), v in h["metrics"].items()}
            serializable.append(entry)
        print(json.dumps(serializable, indent=2, default=str))
        return

    output = []
    output.append(render_status(history, model, args.isl, args.osl, backend))

    if len(history) >= 2:
        output.append(render_trend(history, args.runs))

        changes = detect_changes(history, args.threshold)
        if changes:
            output.append(render_changes(changes, args.threshold))
            output.append(render_sglang_analysis(changes))
        else:
            output.append(f"No changes exceeding {args.threshold}% threshold detected.\n")

    if args.commits:
        output.append("### Commit Log")
        output.append("")
        for h in history[:args.runs]:
            sha = h["commit"].get("id", "?")[:12]
            msg = h["commit"].get("message", "?").split("\n")[0][:80]
            url = h["commit"].get("url", "")
            output.append(f"- {h['date_str']} [`{sha}`]({url}) {msg}")
        output.append("")

    print("\n".join(output))


if __name__ == "__main__":
    main()
