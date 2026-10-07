#!/usr/bin/env python3
"""Look a failing test up in sglang's CI tracking issue before reading any logs.

sgl-project/sglang#17050, `[Tracking] CI Test Failures and Fixes`, is
auto-updated roughly hourly from scheduled CI on `main`. It answers, for one
test file, the question that otherwise costs a log read and a `git log` search:

  ONGOING     main is failing this too, with a measured flake rate. Whatever
              your PR did, it did not cause this.
  FIXED       main was failing this and stopped, on a known date. If your
              branch predates that date, merging main is the likely fix.
  NOT LISTED  scheduled CI on main has not seen it. Fall through to the normal
              /ci-analysis log read — this tool has nothing to say.

Two limits, stated here because they decide what this can replace:

  * The issue's own `Fix` column is empty on all 1155 Recently Fixed rows, so
    this never names the fixing commit. It gives you a *date* to compare a
    branch against, not a PR to cite. Confirming the fix is still
    `gh api compare`.
  * It tracks **scheduled CI on main**. A test absent from it may still be
    broken on main in a way main's own schedule does not run.

Usage:
    known_failure.py test_glm53_flash_b200.py [more_test.py ...]
    known_failure.py --json test_glm53_flash_b200.py
    known_failure.py --refresh ...        # ignore the cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ISSUE = "sgl-project/sglang#17050"
ISSUE_API = "repos/sgl-project/sglang/issues/17050"
# The issue auto-updates about hourly; half that keeps a triage pass from
# re-downloading 109KB for every test it looks at, without going stale enough
# to matter for a decision a human is about to sanity-check anyway.
CACHE_TTL = 1800


def _agent_box() -> Path:
    """Walk up to `agent-box` rather than counting parents — a skill that moves
    one directory deeper must not silently start caching somewhere else."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name == "agent-box":
            return p
    return here.parents[2]


HOST_HOME = Path(os.environ.get("AGENT_BOX_HOST_HOME", _agent_box().parent))
CACHE = Path(os.environ.get("AGENT_SCRATCH_DIR", HOST_HOME / "agent-scratch")) \
    / "ci-analysis" / "known-failures.json"

# `| 2026-10-07 | `test_x.py` | backend | error | notes | status | assignee | related |`
ROW = re.compile(r"^\|([^|]*)\|([^|]*)\|(.*)\|\s*$")
TEST = re.compile(r"`([^`]+)`")
# "performance (3% fail, 131/5125)" -> the only hard number in the whole issue
RATE = re.compile(r"(\w+)\s*\((\d+)%\s*fail,\s*(\d+)/(\d+)\)")


def fetch(refresh: bool = False) -> str:
    if not refresh and CACHE.is_file() and time.time() - CACHE.stat().st_mtime < CACHE_TTL:
        try:
            return json.loads(CACHE.read_text())["body"]
        except Exception:
            pass  # corrupt cache is not a reason to fail the lookup
    env = dict(os.environ, GH_TOKEN="")
    env.setdefault("GH_CONFIG_DIR", str(HOST_HOME / ".gh"))
    env["PATH"] = os.pathsep.join(dict.fromkeys(
        [str(HOST_HOME / "bin"), "/root/.local/bin", "/usr/local/bin"]
        + env.get("PATH", "").split(os.pathsep)))
    try:
        p = subprocess.run(["gh", "api", ISSUE_API, "--jq", ".body"],
                           capture_output=True, text=True, env=env, timeout=120)
    except FileNotFoundError:
        sys.exit("`gh` is not installed or not on PATH")
    if p.returncode != 0:
        sys.exit(f"could not read {ISSUE}: {p.stderr.strip()}")
    body = p.stdout
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"at": time.time(), "body": body}))
    return body


def parse(body: str) -> dict:
    """-> {test_file: {...}}. One entry per test; the two sections are
    disjoint in the source, so a test cannot be both."""
    out, section = {}, "ongoing"
    for line in body.splitlines():
        if "Recently Fixed" in line:
            section = "fixed"
            continue
        if not line.startswith("|") or line.startswith("|---") or "| Date |" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = TEST.search(cells[1])
        if not m:
            continue
        rec = {"test": m.group(1), "date": cells[0], "state": section}
        if section == "ongoing":
            rec["backend"] = cells[2]
            rec["error"] = cells[3] if len(cells) > 3 else ""
            notes = cells[4] if len(cells) > 4 else ""
            rec["notes"] = notes
            r = RATE.search(notes)
            if r:
                rec["category"] = r.group(1)
                rec["fail_pct"] = int(r.group(2))
                rec["fail_n"], rec["runs"] = int(r.group(3)), int(r.group(4))
            rec["link"] = (re.search(r"\((https://\S+)\)", cells[-1]) or [None, ""])[1] \
                if "http" in cells[-1] else ""
        else:
            rec["error"] = cells[1].split("—", 1)[-1].strip() if "—" in cells[1] else ""
        out[rec["test"]] = rec
    return out


def verdict(rec: dict | None, test: str) -> tuple[str, str]:
    """(action, one-line reason) in /ci-analysis's own vocabulary.

    Every answer here is about **main's** history with a test. None of it says
    your PR did not cause your instance of the failure, and the difference is
    not academic: #39575 hit the exact recorded signature of `test_qsa.py`
    (`SimpleNamespace has no attribute …`) while main had already fixed it, and
    the right verdict was still `code-fix` — the PR had added that same read
    and fixed only one of the two fixtures. A matching signature is necessary,
    never sufficient. See `blocked_by_pr_scope` in the caller's checklist.
    """
    if rec is None:
        return "unknown", (f"{test} is not in {ISSUE} — scheduled CI on main has "
                           f"not recorded it. Read the job log as usual.")
    if rec["state"] == "fixed":
        return "merge-main?", (
            f"main was failing {test} and stopped on {rec['date']}. Two things "
            f"to confirm before trusting that: your error matches the recorded "
            f"signature, and your PR does not touch this test or the code under "
            f"it. If it does, this is `code-fix` however well the signature "
            f"matches. The issue names no fixing PR, so date-compare with "
            f"`gh api compare/<head>...main`.")
    pct = rec.get("fail_pct")
    if pct is None:
        return "wait-upstream", (f"{test} is an open item in {ISSUE} since "
                                 f"{rec['date']}, no flake rate recorded.")
    if pct >= 100:
        return "wait-upstream", (
            f"{test} fails 100% on main ({rec['fail_n']}/{rec['runs']}) since "
            f"{rec['date']} — broken, not flaky. A re-run cannot clear it.")
    return "re-run", (
        f"{test} is a known {rec.get('category', '?')} flake on main: "
        f"{pct}% fail ({rec['fail_n']}/{rec['runs']}) since {rec['date']}. "
        f"Not caused by this PR.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="+", help="test file name(s), e.g. test_x.py")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="bypass the cache")
    a = ap.parse_args()

    table = parse(fetch(a.refresh))
    results = []
    for t in a.tests:
        name = Path(t).name
        rec = table.get(name)
        act, why = verdict(rec, name)
        results.append({"test": name, "action": act, "reason": why,
                        "entry": rec})
    if a.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        e = r["entry"] or {}
        state = e.get("state", "not listed").upper()
        print(f"{r['test']}\n  {state} -> {r['action']}")
        # Always printed, never summarised away: the signature is the whole
        # basis for deciding the entry is even about your failure.
        if e.get("error"):
            print(f"  main's signature: {e['error']}")
        if e.get("backend"):
            print(f"  backend: {e['backend']}")
        print(f"  {r['reason']}\n")


if __name__ == "__main__":
    main()
