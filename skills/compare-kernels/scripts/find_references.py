#!/usr/bin/env python3
"""Reference finder for /compare-kernels Step 8 (research extension).

Given a slow kernel / logical-op keyword surfaced by the trace diff, find prior art:
  - aiter-upstream   : ROCm/aiter PRs + local checkout git log/grep
  - sglang-upstream  : sgl-project/sglang PRs + local checkout git log/grep
  - jira-amd         : amd.atlassian.net tickets touching the kernel

Every source is resolved through agent-box/skills/_shared/data-sources.md. Each source
degrades gracefully: preferred access first, then fallback, then a copy-pasteable URL +
the future-MCP hook. Nothing here requires an MCP server today — it uses git + gh + REST
(the fallbacks named in the registry). When an MCP is wired in, swap the marked call.

Stdlib only (urllib), so it runs on host or in any container.

Usage:
  find_references.py <keyword> [--category CAT] [--days 180] [--max 8] [--json]
  find_references.py "ck_gemm" --category gemm
  find_references.py "moe_sorting" --json

Env (all optional — absence just downgrades that source to URL + MCP hint):
  GITHUB_TOKEN                      raise GitHub search rate limit / private repos
  JIRA_EMAIL + JIRA_API_TOKEN      enable Jira REST search (same vars as qwen35-jira-track)
  AITER_ROOT, SGLANG_ROOT          override local-checkout autodetection
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

JIRA_SITE = "https://amd.atlassian.net"

# Future-MCP hooks — the tool to call once each MCP server is registered in
# _shared/data-sources.md (status: live). Until then we use the fallback below each.
MCP_HINTS = {
    "aiter-upstream": "github-mcp: search_pull_requests(repo='ROCm/aiter', query=<keyword>)",
    "sglang-upstream": "github-mcp: search_pull_requests(repo='sgl-project/sglang', query=<keyword>)",
    "jira-amd": "atlassian-mcp: searchJiraIssuesUsingJql(jql=<jql>)",
}


# ---------------------------------------------------------------------------- utils
def _run(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def _http_json(url: str, headers: dict | None = None, timeout: int = 20):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode())


def detect_aiter_root() -> str | None:
    if os.environ.get("AITER_ROOT"):
        return os.environ["AITER_ROOT"]
    for cand in ("/sgl-workspace/aiter", str(Path.home() / "aiter")):
        if Path(cand, ".git").exists() or Path(cand, "aiter").exists():
            return cand
    return None


def detect_sglang_root() -> str | None:
    if os.environ.get("SGLANG_ROOT"):
        return os.environ["SGLANG_ROOT"]
    rc, out = _run(
        [
            sys.executable,
            "-c",
            "import sglang,pathlib;print(pathlib.Path(sglang.__file__).resolve().parents[2])",
        ]
    )
    root = out.strip().splitlines()[-1] if rc == 0 and out.strip() else ""
    return root if root and Path(root).exists() else None


# --------------------------------------------------------------------------- github
def github_prs(repo: str, keyword: str, mcp_key: str, maxn: int) -> dict:
    """PRs touching `keyword`. gh -> REST -> URL, in that order."""
    out = {"source": mcp_key, "repo": repo, "results": [], "via": None,
           "mcp": MCP_HINTS[mcp_key]}
    # 1) gh CLI (live)
    if shutil.which("gh"):
        rc, raw = _run([
            "gh", "search", "prs", "--repo", repo, keyword,
            "--limit", str(maxn), "--json", "number,title,url,state,updatedAt",
        ])
        if rc == 0 and raw.strip():
            try:
                for it in json.loads(raw)[:maxn]:
                    out["results"].append({
                        "id": f"#{it['number']}", "title": it["title"],
                        "url": it["url"], "state": it.get("state", ""),
                        "updated": (it.get("updatedAt") or "")[:10],
                    })
                out["via"] = "live:gh"
                return out
            except Exception:  # noqa: BLE001
                pass
    # 2) GitHub REST search (live, token optional)
    q = urllib.parse.quote(f"{keyword} repo:{repo} type:pr")
    api = f"https://api.github.com/search/issues?q={q}&sort=updated&per_page={maxn}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "compare-kernels"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        data = _http_json(api, headers)
        for it in data.get("items", [])[:maxn]:
            out["results"].append({
                "id": f"#{it['number']}", "title": it["title"],
                "url": it["html_url"], "state": it.get("state", ""),
                "updated": (it.get("updated_at") or "")[:10],
            })
        out["via"] = "live:rest" + ("+token" if "Authorization" in headers else "")
        return out
    except urllib.error.HTTPError as e:
        out["error"] = f"github rest {e.code} (rate limit? set GITHUB_TOKEN)"
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    # 3) URL fallback
    out["via"] = "fallback:url"
    out["search_url"] = f"https://github.com/{repo}/pulls?q={urllib.parse.quote(keyword)}"
    return out


def local_repo_refs(root: str | None, keyword: str, subdirs: list[str], maxn: int) -> dict:
    """git log --grep + source grep on a local checkout (works offline)."""
    if not root:
        return {"root": None, "commits": [], "files": []}
    res: dict = {"root": root, "commits": [], "files": []}
    rc, out = _run(
        ["git", "log", "--all", "-i", f"--grep={keyword}", "--oneline", f"-{maxn}"],
        cwd=root,
    )
    if rc == 0:
        res["commits"] = [ln for ln in out.splitlines() if ln.strip()][:maxn]
    grep_paths = [str(Path(root, s)) for s in subdirs if Path(root, s).exists()] or [root]
    rc, out = _run(
        ["grep", "-rIl", "--include=*.py", "--include=*.cpp", "--include=*.hip",
         "--include=*.cu", "-e", keyword, *grep_paths],
        timeout=25,
    )
    if rc == 0:
        files = [ln for ln in out.splitlines() if ln.strip()]
        res["files"] = [str(Path(f).relative_to(root)) for f in files][:maxn]
    return res


# ----------------------------------------------------------------------------- jira
def jira_search(keyword: str, days: int, maxn: int) -> dict:
    jql = (
        f'text ~ "{keyword}" AND updated >= -{days}d '
        f"ORDER BY updated DESC"
    )
    out = {"source": "jira-amd", "jql": jql, "results": [], "via": None,
           "mcp": MCP_HINTS["jira-amd"]}
    email = os.environ.get("JIRA_EMAIL") or os.environ.get("ATLASSIAN_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN") or os.environ.get("ATLASSIAN_API_TOKEN")
    if email and token:
        params = urllib.parse.urlencode({
            "jql": jql, "maxResults": maxn,
            "fields": "summary,status,assignee,updated",
        })
        url = f"{JIRA_SITE}/rest/api/3/search/jql?{params}"
        auth = base64.b64encode(f"{email}:{token}".encode()).decode()
        try:
            data = _http_json(url, {
                "Authorization": f"Basic {auth}",
                "Accept": "application/json",
                "User-Agent": "compare-kernels-find-references",
            })
            for it in data.get("issues", [])[:maxn]:
                f = it.get("fields", {})
                asg = (f.get("assignee") or {}).get("displayName", "")
                out["results"].append({
                    "id": it.get("key"), "title": f.get("summary", ""),
                    "url": f"{JIRA_SITE}/browse/{it.get('key')}",
                    "state": (f.get("status") or {}).get("name", ""),
                    "assignee": asg, "updated": (f.get("updated") or "")[:10],
                })
            out["via"] = "live:rest"
            return out
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)
    # fallback: hand the JQL + browse URL back
    out["via"] = "fallback:url"
    out["search_url"] = f"{JIRA_SITE}/issues/?jql=" + urllib.parse.quote(jql)
    return out


# --------------------------------------------------------------------------- render
def _print_section(title: str, block: dict) -> None:
    via = block.get("via", "?")
    print(f"\n### {title}  [{via}]")
    if block.get("error"):
        print(f"  ! {block['error']}")
    results = block.get("results", [])
    if results:
        for r in results:
            meta = " ".join(x for x in (
                r.get("state"), r.get("assignee"), r.get("updated")) if x)
            print(f"  {r['id']:<10} {r['title'][:80]}")
            print(f"             {r['url']}  ({meta})")
    else:
        print("  (no direct hits)")
    if block.get("search_url"):
        print(f"  → search manually: {block['search_url']}")
    if block.get("mcp"):
        print(f"  → when MCP live: {block['mcp']}")


def _print_local(title: str, loc: dict) -> None:
    if not loc.get("root"):
        print(f"\n### {title} (local checkout)  [not found]")
        return
    print(f"\n### {title} (local checkout: {loc['root']})  [fallback:local-git]")
    if loc["commits"]:
        print("  recent commits mentioning keyword:")
        for c in loc["commits"]:
            print(f"    {c}")
    if loc["files"]:
        print("  source files matching keyword:")
        for f in loc["files"]:
            print(f"    {f}")
    if not loc["commits"] and not loc["files"]:
        print("  (no local matches)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keyword", help="kernel / logical-op keyword (e.g. ck_gemm, moe_sorting)")
    ap.add_argument("--category", default="", help="logical-op category from the budget diff")
    ap.add_argument("--days", type=int, default=180, help="Jira recency window")
    ap.add_argument("--max", type=int, default=8, help="max results per source")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    kw = args.keyword
    aiter_root = detect_aiter_root()
    sglang_root = detect_sglang_root()

    report = {
        "keyword": kw,
        "category": args.category,
        "aiter_prs": github_prs("ROCm/aiter", kw, "aiter-upstream", args.max),
        "aiter_local": local_repo_refs(aiter_root, kw, ["aiter/ops", "csrc", "op_tests"], args.max),
        "sglang_prs": github_prs("sgl-project/sglang", kw, "sglang-upstream", args.max),
        "sglang_local": local_repo_refs(
            sglang_root, kw, ["python/sglang/srt/layers"], args.max),
        "jira": jira_search(kw, args.days, args.max),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"# Reference finder — keyword: {kw!r}" +
          (f"  (category: {args.category})" if args.category else ""))
    print("# sources resolved via skills/_shared/data-sources.md")
    _print_section("aiter-upstream PRs (ROCm/aiter)", report["aiter_prs"])
    _print_local("aiter-upstream", report["aiter_local"])
    _print_section("sglang-upstream PRs (sgl-project/sglang)", report["sglang_prs"])
    _print_local("sglang-upstream", report["sglang_local"])
    _print_section("jira-amd tickets", report["jira"])
    print("\n# Next: cross-ref hits against the Tier 1/2/3 proposal — prefer cherry-pick "
          "over reinvent.")


if __name__ == "__main__":
    main()
