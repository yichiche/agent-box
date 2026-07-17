#!/usr/bin/env python3
"""Fetch and analyze Jira tickets for Qwen3.5-397B-A17B-MXFP4 watchlist engineers.

Modes:
  --jql-only       Print JQL (use --full for all tickets by person, no model filter)
  --fetch          Jira REST API (JIRA_EMAIL + JIRA_API_TOKEN)
  --analyze        Score helpfulness + optimization areas (use with --fetch or stdin JSON)
  --json           Machine-readable output

Usage:
  track_jira.py --full --jql-only              # JQL: all watchlist tickets, 90d
  track_jira.py --full --fetch --analyze       # fetch + score
  track_jira.py --person "Wen, Jiaxing" --full --fetch --analyze --json
  cat issues.json | track_jira.py --analyze --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

SKILL_DIR = Path(__file__).resolve().parents[1]
WATCHLIST = SKILL_DIR / "watchlist.yaml"
JIRA_SITE = "https://amd.atlassian.net"
DEFAULT_CLOUD_ID = "3ade9f4f-3a5e-4909-bc67-8816482a10f4"

DONE_STATUSES = frozenset(
    s.lower()
    for s in (
        "完成", "已關閉", "closed", "done", "implemented", "完成",
        "resolved", "已解決",
    )
)


def load_watchlist() -> dict:
    if yaml is None:
        sys.exit("PyYAML required: pip install pyyaml")
    return yaml.safe_load(WATCHLIST.read_text())


CRED_FILES = (Path.home() / ".jira_credentials", Path("/home/yichiche/.jira_credentials"))
_CRED_LINE = re.compile(
    r"""\s*(?:export\s+)?(JIRA_EMAIL|ATLASSIAN_EMAIL|JIRA_API_TOKEN|ATLASSIAN_API_TOKEN)"""
    r"""\s*=\s*["']?([^"'\n]+?)["']?\s*$"""
)


def load_cred_file(email: str | None, token: str | None) -> tuple[str | None, str | None]:
    """Fill missing creds from ~/.jira_credentials (shell export lines). Placeholders don't count."""
    for path in CRED_FILES:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            m = _CRED_LINE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if not val or val.startswith("REPLACE_WITH"):
                continue
            if "EMAIL" in key and not email:
                email = val
            elif "TOKEN" in key and not token:
                token = val
        break
    return email, token


def plain_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def person_clause(display_name: str) -> str:
    escaped = display_name.replace('"', '\\"')
    return f'(assignee = "{escaped}" OR reporter = "{escaped}")'


def all_people_jql(wl: dict, domain: str | None = None, person: str | None = None) -> str:
    if person:
        return person_clause(person)
    if domain:
        people = wl["domains"][domain]["people"]
        return "(" + " OR ".join(person_clause(p["display_name"]) for p in people) + ")"
    clauses = []
    for d in wl["domains"].values():
        for p in d["people"]:
            clauses.append(person_clause(p["display_name"]))
    return "(" + " OR ".join(clauses) + ")"


def model_filter_jql(wl: dict) -> str:
    parts = [f'text ~ "{a}"' for a in wl["model"]["aliases"]]
    for kw in ("mxfp4", "397B", "A17B", "Qwen3.5"):
        parts.append(f'text ~ "{kw}"')
    return "(" + " OR ".join(parts) + ")"


def recency_clause(days: int) -> str:
    """Only tickets with Jira activity in the last N days."""
    return f"updated >= -{days}d"


def excluded_keys(wl: dict) -> frozenset[str]:
    ex = wl.get("excluded_tickets") or {}
    if isinstance(ex, dict):
        return frozenset(ex.keys())
    if isinstance(ex, list):
        return frozenset(
            item.get("key", item) if isinstance(item, dict) else item for item in ex
        )
    return frozenset()


def filter_excluded(wl: dict, rows: list[dict]) -> list[dict]:
    skip = excluded_keys(wl)
    if not skip:
        return rows
    return [r for r in rows if r.get("key") not in skip]


def default_days(wl: dict) -> int:
    return int(wl.get("recency_days", 90))


def build_jql(
    wl: dict,
    domain: str | None = None,
    person: str | None = None,
    days: int | None = None,
    model_only: bool = False,
) -> str:
    if days is None:
        days = default_days(wl)
    parts = [all_people_jql(wl, domain, person), recency_clause(days)]
    if model_only:
        parts.append(model_filter_jql(wl))
    return " AND ".join(parts) + " ORDER BY updated DESC"


def build_anchor_jql(wl: dict, days: int | None = None) -> str:
    if days is None:
        days = default_days(wl)
    keys = ", ".join(wl.get("anchor_tickets", []))
    return f"key in ({keys}) AND {recency_clause(days)} ORDER BY updated DESC"


def score_relevance(wl: dict, text: str) -> str:
    lower = text.lower()
    for tier in ("high", "medium", "low"):
        for kw in wl["relevance"][tier]:
            if kw.lower() in lower:
                return tier
    return "none"


def tag_optimization_areas(wl: dict, text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for area_id, area in wl.get("optimization_areas", {}).items():
        for kw in area["keywords"]:
            if kw.lower() in lower:
                hits.append(area_id)
                break
    return hits


def score_helpfulness(wl: dict, row: dict) -> dict:
    """Heuristic pre-score; agent must refine after reading descriptions."""
    text = plain_text(row.get("summary", "")) + " " + plain_text(row.get("description", ""))
    lower = text.lower()
    relevance = row.get("relevance") or score_relevance(wl, text)
    areas = tag_optimization_areas(wl, text)
    status = (row.get("status") or "").lower()
    is_open = status not in DONE_STATUSES and "closed" not in status

    score = 0
    reasons: list[str] = []

    if relevance == "high":
        score += 40
        reasons.append("direct model/anchor match")
    elif relevance == "medium":
        score += 25
        reasons.append("mxfp4/MoE/transferable model family")
    elif relevance == "low":
        score += 10
        reasons.append("generic kernel/aiter keyword")

    if areas:
        score += min(15 * len(areas), 30)
        reasons.append(f"areas: {','.join(areas)}")

    if any(s.lower() in lower for s in wl.get("serve_signals", [])):
        score += 15
        reasons.append("SGLang/aiter serve config in ticket")

    if "mi355" in lower or "gfx95" in lower:
        score += 10
        reasons.append("MI355/gfx95 target")

    if "throughput" in lower or "TPOT" in lower or "perf" in lower or "10-17%" in lower:
        score += 10
        reasons.append("quantified perf impact mentioned")

    if "accuracy" in lower or "GSM8K" in lower:
        score += 5
        reasons.append("accuracy gate relevant")

    if not is_open and relevance in ("high", "medium"):
        score += 5
        reasons.append("done — may cherry-pick fix")

    # Verdict
    if score >= 55 or relevance == "high":
        verdict = "direct" if relevance == "high" else "transferable"
    elif score >= 35 or (areas and relevance != "none"):
        verdict = "transferable" if areas else "enabler"
    elif score >= 20:
        verdict = "watch"
    else:
        verdict = "low"

    if "accuracy" in lower and ("quant" in lower or "quark" in lower or "checkpoint" in lower):
        verdict = "enabler"

    action = wl["helpfulness_verdicts"].get(verdict, {}).get("action", "")

    return {
        "helpfulness_score": min(score, 100),
        "verdict": verdict,
        "optimization_areas": areas,
        "reasons": reasons,
        "suggested_action": action,
        "is_open": is_open,
    }


def domain_for_person(wl: dict, display_name: str) -> str | None:
    for key, domain in wl["domains"].items():
        for p in domain["people"]:
            if p["display_name"] == display_name:
                return key
    return None


def jira_api_get(path: str, email: str, token: str) -> dict:
    url = f"{JIRA_SITE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Basic "
            + __import__("base64").b64encode(f"{email}:{token}".encode()).decode(),
            "User-Agent": "qwen35-jira-track-skill",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"Jira API error {e.code}: {body[:500]}")


def format_updated(iso: str | None) -> str:
    """Jira ISO timestamp → YYYY-MM-DD (or empty)."""
    if not iso:
        return ""
    # 2026-07-16T22:20:55.101-0500
    m = re.match(r"(\d{4}-\d{2}-\d{2})", iso)
    return m.group(1) if m else iso[:10]


def with_updated_display(row: dict) -> dict:
    row["updated_display"] = format_updated(row.get("updated"))
    return row


def fetch_issues(jql: str, email: str, token: str, max_results: int = 100) -> list[dict]:
    fields = "summary,status,assignee,reporter,updated,created,priority,issuetype,description"
    params = urllib.parse.urlencode(
        {"jql": jql, "maxResults": str(max_results), "fields": fields}
    )
    data = jira_api_get(f"/rest/api/3/search/jql?{params}", email, token)
    return data.get("issues", [])


def flatten_issue(issue: dict, wl: dict, domain_hint: str | None = None) -> dict:
    f = issue.get("fields", {})
    assignee = (f.get("assignee") or {}).get("displayName", "")
    reporter = (f.get("reporter") or {}).get("displayName", "")
    desc = plain_text(f.get("description"))
    summary = f.get("summary") or ""
    text = summary + " " + desc
    row = {
        "key": issue.get("key"),
        "summary": summary,
        "status": (f.get("status") or {}).get("name", ""),
        "assignee": assignee,
        "reporter": reporter,
        "updated": f.get("updated"),
        "priority": (f.get("priority") or {}).get("name", ""),
        "issuetype": (f.get("issuetype") or {}).get("name", ""),
        "domain": domain_hint
        or domain_for_person(wl, assignee)
        or domain_for_person(wl, reporter),
        "relevance": score_relevance(wl, text),
        "description_excerpt": desc[:400].replace("\n", " "),
        "url": f"{JIRA_SITE}/browse/{issue.get('key')}",
        "description": desc,
    }
    if len(desc) > 400:
        row["description_excerpt"] += "..."
    return with_updated_display(row)


def analyze_rows(wl: dict, rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        analysis = score_helpfulness(wl, row)
        merged = {**row, **analysis}
        out.append(merged)
    order = {"direct": 0, "enabler": 1, "transferable": 2, "watch": 3, "low": 4}
    out.sort(
        key=lambda r: (-r["helpfulness_score"], order.get(r["verdict"], 9), r.get("updated") or ""),
    )
    return out


def print_analysis_table(rows: list[dict]) -> None:
    if not rows:
        print("No issues.")
        return
    helpful = [r for r in rows if r.get("verdict") in ("direct", "transferable", "enabler")]
    print(f"\n=== Likely helpful ({len(helpful)} / {len(rows)}) ===\n")
    for r in helpful:
        areas = ",".join(r.get("optimization_areas") or []) or "-"
        updated = r.get("updated_display") or format_updated(r.get("updated"))
        print(
            f"[{r['verdict']}|{r['helpfulness_score']}] {r['key']} ({r['status']}) "
            f"{r['assignee']} | updated {updated} | {areas}"
        )
        print(f"  {r['summary'][:90]}")
        print(f"  {r['url']}")
        if r.get("suggested_action"):
            print(f"  → {r['suggested_action']}")
        if r.get("reasons"):
            print(f"  reasons: {'; '.join(r['reasons'][:4])}")
        print()

    print(f"\n=== All tickets ({len(rows)}) ===\n")
    print(
        f"{'Key':<12} {'Verdict':<14} {'Score':<6} {'Updated':<12} "
        f"{'Rel':<8} {'Domain':<8} Status"
    )
    print("-" * 102)
    for r in rows:
        updated = r.get("updated_display") or format_updated(r.get("updated"))
        print(
            f"{r['key']:<12} {r.get('verdict','?'):<14} {r.get('helpfulness_score',0):<6} "
            f"{updated:<12} {r.get('relevance','?'):<8} "
            f"{str(r.get('domain') or '-'):<8} {r['status'][:20]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["moe", "quark", "a16w4"])
    parser.add_argument("--person", help='Display name, e.g. "Wen, Jiaxing"')
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="All tickets by watchlist people (no model keyword filter); default days=90",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Restrict JQL to model/mxfp4 keywords only",
    )
    parser.add_argument("--anchors", action="store_true", help="JQL for anchor tickets only")
    parser.add_argument("--jql-only", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()

    wl = load_watchlist()
    days = args.days if args.days is not None else default_days(wl)
    model_only = args.model_only and not args.full

    if args.anchors:
        jql = build_anchor_jql(wl, days=days)
    else:
        jql = build_jql(
            wl,
            domain=args.domain,
            person=args.person,
            days=days,
            model_only=model_only,
        )

    if args.jql_only or (not args.fetch and not args.analyze):
        print(f"# cloudId: {wl.get('cloud_id', DEFAULT_CLOUD_ID)}")
        print(jql)
        if args.full:
            print(
                f"\n# FULL sweep: watchlist people, updated in last {days}d only "
                "(see excluded_tickets in watchlist.yaml)"
            )
            print("# Agent must read descriptions + analyze")
        if not args.fetch:
            print(
                "\n# MCP: searchJiraIssuesUsingJql with fields including description, comment"
            )
        if args.jql_only:
            return

    rows: list[dict] = []

    if args.fetch:
        email = os.environ.get("JIRA_EMAIL") or os.environ.get("ATLASSIAN_EMAIL")
        token = os.environ.get("JIRA_API_TOKEN") or os.environ.get("ATLASSIAN_API_TOKEN")
        # unfilled ~/.jira_credentials template values may arrive via env.sh
        if email and email.startswith("REPLACE_WITH"):
            email = None
        if token and token.startswith("REPLACE_WITH"):
            token = None
        if not email or not token:
            email, token = load_cred_file(email, token)
        if not email or not token:
            sys.exit(
                "Set JIRA_EMAIL and JIRA_API_TOKEN for --fetch "
                "(or fill in ~/.jira_credentials — see agent-box/env.sh)"
            )
        issues = fetch_issues(jql, email, token, max_results=args.max)
        rows = [flatten_issue(i, wl, domain_hint=args.domain) for i in issues]
        # merge anchors if full sweep
        if args.full and not args.anchors:
            anchor_issues = fetch_issues(
                build_anchor_jql(wl, days=days), email, token, max_results=50
            )
            seen = {r["key"] for r in rows}
            for i in anchor_issues:
                k = i.get("key")
                if k and k not in seen:
                    rows.append(flatten_issue(i, wl))
                    seen.add(k)
    elif args.analyze and not sys.stdin.isatty():
        rows = json.load(sys.stdin)
    else:
        sys.exit("Use --fetch or pipe JSON to stdin for --analyze")

    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"])
            deduped.append(with_updated_display(r))

    if args.analyze or args.fetch:
        deduped = analyze_rows(wl, deduped)

    deduped = filter_excluded(wl, deduped)

    if args.json:
        # trim description for json unless full analyze requested
        out = []
        for r in deduped:
            item = dict(r)
            if not args.analyze and "description" in item:
                item.pop("description", None)
            out.append(item)
        print(json.dumps(out, indent=2))
    else:
        if args.analyze:
            print_analysis_table(deduped)
        else:
            for r in deduped:
                updated = r.get("updated_display") or format_updated(r.get("updated"))
                print(f"{r['key']}\t{updated}\t{r['relevance']}\t{r['summary'][:70]}")
        print(f"\n{len(deduped)} issue(s)")


if __name__ == "__main__":
    main()
