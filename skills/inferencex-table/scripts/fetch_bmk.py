#!/usr/bin/env python3
"""Fetch InferenceX benchmark results from a GitHub Actions run URL and
emit a markdown table (Concurrency, TP/DP, TTT, E2EL, TTFT, ITL).

Usage:
    fetch_bmk.py <run-url-or-id> [--isl N] [--osl N] [--dpa true|false|auto]
                                [--token <github_pat>] [--out path]
                                [--json] [--keep <dir>]

If --isl/--osl are omitted, all (isl,osl) combinations present in the run
are emitted as separate tables. --dpa=auto (default) shows both DPA variants
when they exist for the same concurrency.

The GitHub token is read from --token, $GH_TOKEN, $GITHUB_TOKEN, or
~/.git-credentials (in that order).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


REPO_DEFAULT = "SemiAnalysisAI/InferenceX"
API_ROOT = "https://api.github.com"


def parse_run(arg: str) -> tuple[str, str]:
    """Return (owner/repo, run_id) from a URL or bare id."""
    if arg.isdigit():
        return REPO_DEFAULT, arg
    m = re.search(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)", arg)
    if not m:
        sys.exit(f"Could not parse run id from: {arg}")
    return m.group(1), m.group(2)


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
    sys.exit("No GitHub token found (set GH_TOKEN or populate ~/.git-credentials)")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def gh_get(url: str, token: str) -> bytes:
    """GET that handles GitHub's signed-redirect to Azure blob storage."""
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
        sys.exit(f"GitHub API error {e.code}: {e.read().decode(errors='replace')[:300]}")


def find_artifact(repo: str, run_id: str, token: str, name: str) -> dict:
    page = 1
    while True:
        data = json.loads(
            gh_get(
                f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}",
                token,
            )
        )
        for art in data.get("artifacts", []):
            if art["name"] == name:
                return art
        if len(data.get("artifacts", [])) < 100:
            sys.exit(f"Artifact '{name}' not found in run {run_id}")
        page += 1


def download_results_bmk(repo: str, run_id: str, token: str, dest: Path) -> Path:
    art = find_artifact(repo, run_id, token, "results_bmk")
    zip_bytes = gh_get(art["archive_download_url"], token)
    zpath = dest / "results_bmk.zip"
    zpath.write_bytes(zip_bytes)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)
    return dest / "agg_bmk.json"


def ms(x):
    return None if x is None else x * 1000.0


def fmt(v, prec=2):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "-"


def render_table(rows: list[dict]) -> str:
    header = "| Concurrency | TP, DP | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for e in rows:
        dp = e["tp"] if e.get("dp_attention") in ("true", True, "True") else 1
        ttt = e["tput_per_gpu"] * e["tp"]
        lines.append(
            f"| {e['conc']} | {e['tp']}, {dp} | {fmt(ttt)} | "
            f"{fmt(ms(e.get('median_e2el')))} | {fmt(ms(e.get('median_ttft')))} | "
            f"{fmt(ms(e.get('median_itl')))} |"
        )
    return "\n".join(lines)


def select_rows(entries, isl, osl, dpa):
    rows = [e for e in entries if e.get("isl") == isl and e.get("osl") == osl]
    if dpa != "auto":
        want = str(dpa).lower()
        rows = [e for e in rows if str(e.get("dp_attention", "")).lower() == want]
    rows.sort(
        key=lambda e: (
            e["conc"],
            0 if str(e.get("dp_attention", "")).lower() == "false" else 1,
        )
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="GitHub Actions run URL or numeric id")
    p.add_argument("--isl", type=int, help="Input seq len filter (e.g. 8192)")
    p.add_argument("--osl", type=int, help="Output seq len filter (e.g. 1024)")
    p.add_argument(
        "--dpa",
        default="auto",
        choices=["auto", "true", "false"],
        help="DP-attention filter (default: auto = show both)",
    )
    p.add_argument("--token", help="GitHub PAT (overrides env / git-credentials)")
    p.add_argument("--out", help="Write markdown table to this file")
    p.add_argument("--json", action="store_true", help="Also print full agg_bmk.json path")
    p.add_argument("--keep", help="Keep downloaded artifact in this directory")
    args = p.parse_args()

    repo, run_id = parse_run(args.run)
    token = resolve_token(args.token)

    if args.keep:
        workdir = Path(args.keep).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        agg = download_results_bmk(repo, run_id, token, workdir)
    else:
        tmp = tempfile.TemporaryDirectory()
        agg = download_results_bmk(repo, run_id, token, Path(tmp.name))

    entries = json.loads(agg.read_text())

    combos = sorted({(e.get("isl"), e.get("osl")) for e in entries})
    if args.isl is not None and args.osl is not None:
        targets = [(args.isl, args.osl)]
    elif args.isl is not None:
        targets = [(args.isl, o) for (i, o) in combos if i == args.isl]
    elif args.osl is not None:
        targets = [(i, args.osl) for (i, o) in combos if o == args.osl]
    else:
        targets = combos

    out_lines = [f"# Run {run_id} ({repo})", ""]
    for isl, osl in targets:
        rows = select_rows(entries, isl, osl, args.dpa)
        if not rows:
            continue
        out_lines.append(f"## ISL={isl} / OSL={osl}")
        out_lines.append("")
        out_lines.append(render_table(rows))
        out_lines.append("")

    text = "\n".join(out_lines).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)
    if args.json:
        print(f"\n# raw json: {agg}")


if __name__ == "__main__":
    main()
