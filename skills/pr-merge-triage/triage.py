#!/usr/bin/env python3
"""Classify an sgl-project/sglang PR against the merge bar in SKILL.md.

Answers the mechanical half of the review — blast radius, guards, new flags,
interface churn, kernel kind, size, evidence — and prints the checklist
pre-filled. The judgement half (is the guard the *right* one? is this one
concern or three?) stays with the reviewer; rows the script cannot decide are
printed as `?` rather than guessed.

    python3 triage.py 41870 [--repo sgl-project/sglang] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

# A path is AMD-only if its *name* says so. This is the whole basis of the
# blast-radius check: a file nobody else builds or runs cannot break anybody
# else, so it earns a one-vendor review. Keep the pattern conservative —
# a false "AMD-only" is the one error here that costs someone a broken main.
AMD_PATH = re.compile(
    r"(rocm|/hip[_/]|hip_|_hip\b|aiter|quark|/amd/|_amd\b|gfx\d|mi[23]\d\d|triton_amd)",
    re.I,
)
# Common code whose behaviour every vendor inherits. Touching these is not
# forbidden — it is the trigger for pulling in a community reviewer.
HOT_COMMON = re.compile(
    r"^python/sglang/srt/(layers/(linear|layernorm|activation|rotary_embedding)\.py"
    r"|models/|managers/|model_executor/|distributed/|mem_cache/|environ\.py"
    r"|arg_groups/|server_args\.py|entrypoints/)",
)
KERNEL_PATH = re.compile(r"(python/sglang/kernels/|\.cu$|\.cuh$|\.hip$|\.cpp$|csrc/)")
TEST_PATH = re.compile(r"^(test/|python/sglang/test/)")
DOC_PATH = re.compile(r"^(docs/|benchmark/|.*\.md$)")

GUARD = re.compile(r"\b(_?is_hip|_?use_aiter|is_cuda|_?is_cuda|is_npu|is_xpu)\b")
AITER_IMPORT = re.compile(r"\b(from|import)\s+aiter\b|from\s+aiter\.")
NEW_ENV = re.compile(r"^\+\s*(SGLANG_\w+)\s*=\s*Env(\w+)\(([^)]*)\)")
NEW_GLOBAL = re.compile(r"^\+([A-Z_][A-Z0-9_]{2,})\s*=")
DEF_LINE = re.compile(r"^[-+]\s*(?:async\s+)?def\s+(\w+)\s*\(")
IMPORT_LINE = re.compile(r"^[-+]\s*(from\s+[\w.]+\s+import|import\s+\w|#)")

OK, WARN, BAD, UNK = "PASS", "CHECK", "FAIL", "?"


def sh(args: list[str]) -> str:
    env = dict(os.environ, GH_TOKEN="", GH_PAGER="cat", GIT_PAGER="cat")
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    if p.returncode:
        sys.exit(f"{' '.join(args[:4])}… failed: {p.stderr.strip()}")
    return p.stdout


def split_hunks(diff: str) -> list[dict]:
    """Per-file hunks, each carrying its added lines and its context."""
    files, cur, hunk = [], None, None
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            cur = {"path": line.split(" b/")[-1], "new": False, "hunks": []}
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith("new file mode"):
            cur["new"] = True
        elif line.startswith("@@"):
            hunk = {"header": line, "added": [], "removed": [], "context": []}
            cur["hunks"].append(hunk)
        elif hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                hunk["added"].append(line)
            elif line.startswith("-") and not line.startswith("---"):
                hunk["removed"].append(line)
            else:
                hunk["context"].append(line)
    return files


def classify_path(path: str) -> str:
    if TEST_PATH.search(path):
        return "test"
    if DOC_PATH.search(path):
        return "doc"
    if AMD_PATH.search(path):
        return "amd"
    if HOT_COMMON.search(path):
        return "hot-common"
    return "common"


def guard_state(f: dict) -> str:
    """Is every added hunk in this file reachable only on AMD?

    Heuristic, and deliberately a loose one: a hunk counts as guarded if a
    guard token appears in the hunk itself or in its context window. It cannot
    see a guard 40 lines up, so `partial` means *go read it*, not *it is wrong*.
    """
    if not f["hunks"]:
        return "none"
    guarded = 0
    for h in f["hunks"]:
        blob = "\n".join(h["added"] + h["context"] + [h["header"]])
        if GUARD.search(blob):
            guarded += 1
    if guarded == len(f["hunks"]):
        return "all"
    return "partial" if guarded else "unguarded"


def analyse(pr: str, repo: str) -> dict:
    meta = json.loads(sh([
        "gh", "pr", "view", pr, "--repo", repo, "--json",
        "title,body,author,additions,deletions,changedFiles,files,state,isDraft,labels",
    ]))
    diff = sh(["gh", "pr", "diff", pr, "--repo", repo])
    files = split_hunks(diff)

    buckets = Counter(classify_path(f["path"]) for f in files)
    amd_files = [f for f in files if classify_path(f["path"]) == "amd"]
    code_files = [f for f in files if classify_path(f["path"]) in
                  ("amd", "common", "hot-common")]
    nonamd = [f for f in code_files if classify_path(f["path"]) != "amd"]
    # A file added by this PR is not code anyone inherits — it did not exist
    # before, so it cannot change behaviour for a vendor that never imports it.
    # It still has to be *reached* from an AMD path, which is a judgement call,
    # so it is reported separately rather than counted as blast radius.
    nonamd_new = [f for f in nonamd if f["new"]]
    nonamd_mod = [f for f in nonamd if not f["new"]]

    # An edit that only *adds* lines to shared code — a new enum member, a new
    # accessor, a new branch keyed on a name nothing selects yet — cannot change
    # behaviour for anyone who does not opt into it. It is a shared-code touch,
    # and it is the cheap kind. Separating it from a real edit is what keeps the
    # blast-radius signal worth reading.
    # A removed *import* line is not a behaviour rewrite — widening an import to
    # pull in one more helper is the most common removal in an otherwise purely
    # additive diff, and counting it flips a cheap PR into the expensive bucket
    # for nothing.
    def rewrites(f: dict) -> bool:
        return any(not IMPORT_LINE.match(l) and l[1:].strip()
                   for h in f["hunks"] for l in h["removed"])

    additive = [f for f in nonamd_mod if not rewrites(f)]
    edited = [f for f in nonamd_mod if rewrites(f)]

    # Guard state only matters where existing shared behaviour was rewritten.
    unguarded = [f["path"] for f in edited if guard_state(f) == "unguarded"]
    partial = [f["path"] for f in edited if guard_state(f) == "partial"]

    added = [l for f in files for h in f["hunks"] for l in h["added"]]
    new_envs = [(m.group(1), m.group(2), m.group(3).strip())
                for l in added if (m := NEW_ENV.match(l))]
    # Only in *existing* shared files: a constant at the top of a brand-new
    # module is module scope, not global state leaking into everyone's path.
    new_globals = sorted({m.group(1) for f in nonamd_mod
                          for h in f["hunks"] for l in h["added"]
                          if (m := NEW_GLOBAL.match(l))})

    # An interface change = a def whose signature line appears on both sides of
    # the diff in common code. Same name removed and added is a rewrite of the
    # signature; added only is a new API, which is cheaper.
    sig_removed, sig_added = set(), set()
    for f in nonamd:
        for h in f["hunks"]:
            for l in h["removed"]:
                if m := DEF_LINE.match(l):
                    sig_removed.add(f"{f['path']}:{m.group(1)}")
            for l in h["added"]:
                if m := DEF_LINE.match(l):
                    sig_added.add(f"{f['path']}:{m.group(1)}")
    sig_changed = sorted(sig_removed & sig_added)

    kernels = [f for f in files if KERNEL_PATH.search(f["path"])]
    new_kernels = [f["path"] for f in kernels if f["new"]]
    uses_aiter = any(AITER_IMPORT.search(l) for l in added)
    uses_is_hip = any(re.search(r"\bis_hip\b", l) for l in added)

    body = (meta.get("body") or "").lower()
    has_accuracy = bool(re.search(r"gsm8k|mmlu|mmmu|accuracy|lm.?eval", body))
    has_perf = bool(re.search(r"throughput|ttft|tpot|latency|tok/s|speedup|us/|µs", body))

    return {
        "pr": pr, "repo": repo, "meta": meta, "files": files, "buckets": buckets,
        "amd_files": [f["path"] for f in amd_files],
        "common_files": [f["path"] for f in edited],
        "additive_files": [f["path"] for f in additive],
        "new_files": [f["path"] for f in nonamd_new],
        "hot_common": [f["path"] for f in edited
                       if classify_path(f["path"]) == "hot-common"],
        "unguarded": unguarded, "partial": partial,
        "new_envs": new_envs, "new_globals": new_globals,
        "sig_changed": sig_changed,
        "kernels": [f["path"] for f in kernels], "new_kernels": new_kernels,
        "uses_aiter": uses_aiter, "uses_is_hip": uses_is_hip,
        "has_accuracy": has_accuracy, "has_perf": has_perf,
    }


# --- scoring -----------------------------------------------------------------
#
# Two mechanisms, deliberately not merged into one number:
#
#   GATES  — binary, and no amount of good elsewhere substitutes for them. Each
#            one marks something that is either a crash, a silent wrong answer,
#            or a claim nobody can check. A PR with an open gate is not "high
#            risk", it is unreviewable until the gate is answered.
#   POINTS — risk that trades off. Shared code, size, flags, missing tests: each
#            costs points, and the total says how much review the PR needs.
#
# Routing is a third, separate thing: who has to look at it. A PR can be 0
# points and still need a community reviewer, because it edits shared code.

GATES = {
    "Guard choice": "aiter import must be gated by use_aiter, not is_hip alone "
                    "— otherwise it crashes on an AMD box without AITER",
    "Evidence": "a numerics change with no accuracy number cannot be reviewed, "
                "only believed",
    "Interface churn": "a rewritten shared signature must have every caller "
                       "updated in the same PR",
    "Correctness": "a CRITICAL finding from /sglang-pr-review produces wrong "
                   "model output — it must be fixed, not weighed",
}
BANDS = ((3, "LOW"), (7, "MEDIUM"), (12, "HIGH"))

# Severity lines in /sglang-pr-review output, in both the shapes it emits: the
# findings bullets (`[bug] CRITICAL — path:line — …`) and the Risk & Scope table
# (`| Critical | … |`). The table usually restates the bullets, so counting both
# and taking the larger per severity avoids double-counting without losing a
# finding that only appears in one of them.
SEV_BULLET = re.compile(r"^\s*[-*]\s*`?\[\w+\]\s*(CRITICAL|HIGH|MEDIUM|LOW)\b", re.I)
SEV_TABLE = re.compile(r"^\s*\|\s*(Critical|High|Medium|Low)\s*\|", re.I)
DECISION = re.compile(r"\b(approve|comment|request-changes)\b", re.I)


def parse_review(text: str) -> dict:
    """Severity counts from a /sglang-pr-review report."""
    bullets, table = Counter(), Counter()
    decision = ""
    for line in text.splitlines():
        if m := SEV_BULLET.match(line):
            bullets[m.group(1).upper()] += 1
        elif m := SEV_TABLE.match(line):
            table[m.group(1).upper()] += 1
        if line.lower().lstrip().startswith(("### decision", "**decision",
                                             "decision:")):
            if d := DECISION.search(line):
                decision = d.group(1).lower()
        elif not decision and "decision" in line.lower():
            if d := DECISION.search(line):
                decision = d.group(1).lower()
    counts = {s: max(bullets[s], table[s])
              for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    counts["decision"] = decision
    counts["source"] = "parsed"
    return counts


def band(points: int) -> str:
    for limit, name in BANDS:
        if points <= limit:
            return name
    return "SPLIT"


def rows(a: dict) -> list[dict]:
    """One dict per check: verdict, evidence, remediation, risk points, gate."""
    m, out = a["meta"], []
    hot, common = a["hot_common"], a["common_files"]
    unguarded, partial, additive = a["unguarded"], a["partial"], a["additive_files"]

    def row(check, vd, ev, act="", pts=0, gate=False):
        out.append({"check": check, "verdict": vd, "evidence": ev,
                    "action": act, "points": pts, "gate": gate})

    # 1. Blast radius — how much of the world inherits this change
    if hot:
        row("Blast radius", BAD,
            f"edits hot common code: {', '.join(hot[:3])}"
            + (f" (+{len(hot)-3})" if len(hot) > 3 else ""),
            "needs a community reviewer; check whether it can move behind an "
            "AMD-only module instead", 4)
    elif common:
        row("Blast radius", WARN,
            f"edits {len(common)} shared file(s): {', '.join(common[:3])}",
            "confirm no NVIDIA/CPU behaviour changes", 2)
    elif additive:
        row("Blast radius", WARN,
            f"shared files, additive only: {', '.join(additive[:3])}",
            "cheap kind — nothing existing changes behaviour; say so in review", 1)
    elif a["new_files"]:
        row("Blast radius", WARN,
            f"{len(a['new_files'])} new file(s) outside an AMD path: "
            f"{', '.join(a['new_files'][:3])}",
            "confirm the module is only imported from an AMD-guarded call site, "
            "or move it under an AMD-named path", 1)
    else:
        row("Blast radius", OK, f"{len(a['amd_files'])} file(s), all AMD-only paths")

    # 2. Guards — only meaningful where existing shared behaviour was rewritten
    if not common:
        row("AMD guard", OK, "n/a — no existing shared behaviour rewritten")
    elif unguarded:
        row("AMD guard", BAD,
            f"rewrites shared code with no is_hip/use_aiter in hunk: "
            f"{', '.join(unguarded[:3])}",
            "wrap in `if _is_hip:` (all AMD GPUs) or `if _use_aiter:` (needs the "
            "AITER library) — or, if the bug is shared, leave it unguarded and "
            "get a community reviewer", 3)
    elif partial:
        row("AMD guard", WARN,
            f"some hunks show no guard in context: {', '.join(partial[:3])}",
            "read those hunks — the guard may be above the window", 1)
    else:
        row("AMD guard", OK, "every rewritten hunk sits under a guard")

    # 3. Guard choice — GATE: the wrong one is a crash, not a style problem
    if a["uses_aiter"] and not a["uses_is_hip"]:
        row("Guard choice", WARN, "imports aiter; no is_hip/use_aiter token added",
            "gate with `_use_aiter` — `is_hip()` alone runs this on an AMD box "
            "with no AITER installed", 0, gate=True)
    elif a["uses_aiter"]:
        row("Guard choice", OK, "imports aiter, guard tokens present")
    elif a["uses_is_hip"]:
        row("Guard choice", OK, "is_hip — works on all AMD GPUs")
    else:
        row("Guard choice", UNK, "no guard token added")

    # 4. Flags
    if a["new_envs"]:
        offs = [f"{n}={d}" for n, _, d in a["new_envs"] if "False" in d or d == ""]
        if offs:
            row("New flags", BAD,
                f"{len(a['new_envs'])} new env var(s), default-off: "
                f"{', '.join(offs[:3])}",
                "if the hardware implies it, default it on and detect with "
                "is_hip/use_aiter — do not make users export a flag", 3)
        else:
            row("New flags", WARN,
                f"new env var(s): {', '.join(n for n, _, _ in a['new_envs'])}",
                "default-on is right; still ask whether the knob is needed", 1)
    else:
        row("New flags", OK, "no new env var")

    if a["new_globals"]:
        row("New globals", WARN,
            f"module-level globals in shared code: "
            f"{', '.join(a['new_globals'][:4])}",
            "prefer a derived constant or a config field over global state", 2)
    else:
        row("New globals", OK, "no new global in shared code")

    # 5. Interface churn — GATE when a shared signature is rewritten
    if a["sig_changed"]:
        row("Interface churn", WARN,
            f"signature changed: {', '.join(a['sig_changed'][:3])}",
            "verify every caller is updated in this PR; prefer a keyword arg "
            "with a behaviour-preserving default", 3, gate=True)
    else:
        row("Interface churn", OK, "no shared signature rewritten")

    # 6. Kernel kind
    if a["new_kernels"]:
        row("Kernel", WARN, f"new kernel file(s): {', '.join(a['new_kernels'][:3])}",
            "needs a reference-correctness test and a benchmark; check the "
            "non-AMD fallback still exists", 2)
    elif a["kernels"]:
        row("Kernel", WARN, f"modifies {len(a['kernels'])} existing kernel file(s)",
            "needs before/after numbers on the same shapes", 1)
    else:
        row("Kernel", OK, "no kernel source touched")

    # 7. Size / splittability
    tot = m["additions"] + m["deletions"]
    areas = {p.split("/")[3] if p.startswith("python/sglang/srt/") and
             len(p.split("/")) > 4 else p.split("/")[0]
             for p in a["amd_files"] + a["common_files"] + a["new_files"]
             + a["additive_files"]}
    if tot > 800 or len(areas) > 3:
        row("Size", BAD, f"+{m['additions']}/-{m['deletions']} across {len(areas)} areas",
            "split: one concern per PR (kernel / wiring / default flip)", 3)
    elif tot > 300:
        row("Size", WARN, f"+{m['additions']}/-{m['deletions']}",
            "reviewable, but check it is one concern", 1)
    else:
        row("Size", OK, f"+{m['additions']}/-{m['deletions']}")

    # 8. Evidence — GATE when numerics moved and nothing was measured
    need_acc = bool(a["kernels"]) or any(
        re.search(r"quant|moe|attention", p, re.I)
        for p in a["amd_files"] + a["common_files"] + a["new_files"]
        + a["additive_files"])
    if need_acc and not a["has_accuracy"]:
        row("Evidence", BAD, "numerics touched, no accuracy number in body",
            "ask for GSM8K (or equivalent) before/after", 0, gate=True)
    elif a["kernels"] and not a["has_perf"]:
        row("Evidence", WARN, "kernel change, no perf number in body",
            "ask for before/after on the shapes it targets", 1)
    else:
        row("Evidence", OK,
            f"accuracy={a['has_accuracy']} perf={a['has_perf']}")

    # 9. Tests
    if a["buckets"].get("test"):
        row("Tests", OK, f"{a['buckets']['test']} test file(s) touched")
    else:
        row("Tests", WARN, "no test file in the diff",
            "AMD-only tests belong in test/registered/amd/", 2)

    # 10. Correctness — GATE, and the only row this script cannot derive. It is
    # fed from /sglang-pr-review: that skill finds the bugs, this one decides
    # what they mean for merging. Absent a review, the row stays `?` and the
    # verdict refuses to say "merge" — triage measures cost to land, never
    # correctness.
    rv = a.get("review")
    if not rv:
        row("Correctness", UNK, "no /sglang-pr-review findings supplied",
            "run `/sglang-pr-review <pr>` and pass it back with --review")
    elif rv["CRITICAL"]:
        row("Correctness", BAD,
            f"{rv['CRITICAL']} CRITICAL finding(s) from /sglang-pr-review",
            "must be fixed before merge — a CRITICAL is wrong model output, "
            "not a risk to weigh", 0, gate=True)
    else:
        pts = min(3 * rv["HIGH"], 6) + min(rv["MEDIUM"], 3)
        ev = (f"no CRITICAL; {rv['HIGH']} high, {rv['MEDIUM']} medium, "
              f"{rv['LOW']} low")
        if rv["HIGH"]:
            row("Correctness", WARN, ev,
                "High findings crash or degrade under specific configurations "
                "— resolve or get the author's rationale on record", pts)
        elif pts:
            row("Correctness", WARN, ev,
                "medium findings: fix now or file them as follow-ups", pts)
        else:
            row("Correctness", OK, ev)
    return out


def verdict(a: dict, rs: list[dict]) -> dict:
    open_gates = [r["check"] for r in rs if r["gate"] and r["verdict"] != OK]
    points = sum(r["points"] for r in rs)
    b = band(points)
    needs_community = bool(a["hot_common"]) or bool(a["unguarded"])

    if b == "SPLIT":
        name, why = "SPLIT FIRST", (f"{points} risk points — too much in one PR to "
                                    f"review as a unit")
    elif open_gates:
        name, why = "BLOCKED ON AUTHOR", (f"open gate(s): {', '.join(open_gates)} — "
                                          f"no score substitutes for these")
    elif needs_community:
        name, why = "NEEDS COMMUNITY REVIEWER", (
            f"{points} risk points ({b}), but it changes code every vendor "
            f"inherits — an AMD-side review cannot land it")
    elif not a.get("review"):
        # Cheap-to-land is not the same as correct. Saying "merge" off the diff
        # shape alone is exactly the mistake this pairing exists to prevent.
        name, why = "TRIAGE CLEAR — NEEDS CORRECTNESS REVIEW", (
            f"{points} risk points ({b}), no open gate — cheap to land, but "
            f"nothing has read it for bugs yet: run /sglang-pr-review")
    elif b == "LOW":
        name, why = "LOW RISK — MERGE", (f"{points} risk points, no open gate, "
                                         f"no CRITICAL finding — AMD-side "
                                         f"review is enough")
    else:
        name, why = f"{b} RISK — REVIEW", (f"{points} risk points; read the "
                                           f"flagged rows before approving")
    # Whatever else the verdict says, never let "nobody has read this for bugs"
    # fall off the end of the sentence.
    if not a.get("review") and name != "TRIAGE CLEAR — NEEDS CORRECTNESS REVIEW":
        why += " · correctness not reviewed yet — run /sglang-pr-review"
    return {"name": name, "why": why, "points": points, "band": b,
            "open_gates": open_gates, "needs_community": needs_community,
            "reviewed": bool(a.get("review"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pr")
    ap.add_argument("--repo", default="sgl-project/sglang")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--review", metavar="PATH",
                    help="/sglang-pr-review report to fold in ('-' for stdin)")
    ap.add_argument("--critical", type=int, default=None,
                    help="severity counts by hand, instead of --review")
    ap.add_argument("--high", type=int, default=0)
    ap.add_argument("--medium", type=int, default=0)
    ap.add_argument("--low", type=int, default=0)
    a = ap.parse_args()
    pr = re.sub(r"\D", "", a.pr.split("/")[-1]) or a.pr

    review = None
    if a.review:
        text = sys.stdin.read() if a.review == "-" else open(a.review).read()
        review = parse_review(text)
    elif a.critical is not None:
        review = {"CRITICAL": a.critical, "HIGH": a.high, "MEDIUM": a.medium,
                  "LOW": a.low, "decision": "", "source": "manual"}

    data = analyse(pr, a.repo)
    data["review"] = review
    rs = rows(data)
    v = verdict(data, rs)

    if a.json:
        print(json.dumps({"pr": pr, **v, "rows": rs}, indent=2))
        return

    m = data["meta"]
    mark = {OK: "- [x]", WARN: "- [ ]", BAD: "- [ ]", UNK: "- [ ]"}
    print(f"## PR #{pr} — merge triage\n")
    print(f"**{m['title']}** — @{m['author']['login']}, "
          f"+{m['additions']}/-{m['deletions']} over {m['changedFiles']} files"
          f"{' (DRAFT)' if m['isDraft'] else ''}\n")
    print("| | Check | Verdict | Risk | Evidence |")
    print("|---|---|---|---|---|")
    for r in rs:
        g = " **GATE**" if r["gate"] and r["verdict"] != OK else ""
        pts = f"+{r['points']}" if r["points"] else "0"
        print(f"| {mark[r['verdict']]} | {r['check']}{g} | {r['verdict']} "
              f"| {pts} | {r['evidence']} |")
    print(f"| | **Total** | | **{v['points']}** | band: {v['band']} "
          f"(LOW ≤3, MEDIUM ≤7, HIGH ≤12, SPLIT >12) |")
    print(f"\n**Verdict: {v['name']}** — {v['why']}\n")

    if v["open_gates"]:
        print("### Gates (must be answered — points cannot buy these off)")
        for g in v["open_gates"]:
            print(f"- **{g}** — {GATES[g]}")
        print()
    acts = [r for r in rs if r["action"] and r["verdict"] in (BAD, WARN)]
    if acts:
        print("### Ask the author")
        for i, r in enumerate(acts, 1):
            print(f"{i}. **{r['check']}** — {r['action']}")
    print("\n> Mechanical half only. Still yours to judge: is the guard the right "
          "one, is this one concern or three, is it a new feature or a live "
          "regression, and is the bug AMD-only or shared "
          "(see SKILL.md 'What the script cannot decide').")


if __name__ == "__main__":
    main()
