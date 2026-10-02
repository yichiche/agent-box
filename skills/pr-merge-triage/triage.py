#!/usr/bin/env python3
"""Classify an sgl-project/sglang PR against the merge bar in SKILL.md.

Answers the mechanical half of the review — affected scope, guards, new flags,
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
# affected-scope check: a file nobody else builds or runs cannot break anybody
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

GUARD = re.compile(
    r"\b(_?is_hip|_?use_aiter|is_cuda|_?is_cuda|is_npu|is_xpu"
    r"|is_gfx95_supported|is_gfx95|gfx950|gfx942|gfx90a)\b"
    r"|SGLANG_USE_AITER|(?i:mi355|mi300|mi325)"
)
# Kinds the AMD-guard row must cite, with one example line from the diff.
# A device check (MI355 / gfx950) is a guard, and it is not is_hip.
GUARD_KINDS = (
    ("use_aiter", re.compile(r"\b_?use_aiter\b|SGLANG_USE_AITER")),
    ("is_hip", re.compile(r"\b_?is_hip\b")),
    ("MI355", re.compile(r"is_gfx95(?:_supported)?|\bgfx950\b|(?i:mi355)")),
    ("MI300", re.compile(r"\bgfx94[0-9]\b|\bgfx90a\b|(?i:mi30[025])")),
)
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
    # so it is reported separately rather than counted as affected scope.
    nonamd_new = [f for f in nonamd if f["new"]]
    nonamd_mod = [f for f in nonamd if not f["new"]]

    # An edit that only *adds* lines to shared code — a new enum member, a new
    # accessor, a new branch keyed on a name nothing selects yet — cannot change
    # behaviour for anyone who does not opt into it. It is a shared-code touch,
    # and it is the cheap kind. Separating it from a real edit is what keeps the
    # affected-scope signal worth reading.
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
        "guard_examples": guard_examples(files),
        "has_accuracy": has_accuracy, "has_perf": has_perf,
    }


def guard_examples(files: list[dict]) -> list[dict]:
    """First added line per guard kind, from non-test code.

    Comments are skipped: a docstring that mentions aiter is not a guard.
    """
    found = {}
    for f in files:
        if classify_path(f["path"]) not in ("amd", "common", "hot-common"):
            continue
        if TEST_PATH.search(f["path"]):
            continue
        for h in f["hunks"]:
            for line in h["added"]:
                body = line[1:].strip()
                if not body or body.startswith("#"):
                    continue
                for kind, rx in GUARD_KINDS:
                    if kind in found or not rx.search(body):
                        continue
                    found[kind] = {
                        "kind": kind,
                        "path": f["path"],
                        "example": body[:100],
                    }
    return [found[k] for k, _ in GUARD_KINDS if k in found]


def describe_guards(a: dict) -> str:
    """Name the guard and quote one line. Say which of the three it is not."""
    hits = a["guard_examples"]
    if hits:
        return "; ".join(
            f"{h['kind']} — `{h['example']}` in {h['path'].rsplit('/', 1)[-1]}"
            for h in hits
        )
    amd_code = [p for p in a["amd_files"] if not TEST_PATH.search(p)]
    if amd_code:
        names = ", ".join(p.rsplit("/", 1)[-1] for p in amd_code[:3])
        return (
            "no is_hip, use_aiter, or MI355/gfx950 check in the added lines; "
            f"scope is the AMD path ({names})"
        )
    return "no is_hip, use_aiter, or MI355/gfx950 check in the added lines"


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
    "Critical risk": "a CRITICAL finding from /sglang-pr-review blocks the "
                     "merge — wrong output, hang, crash, raise, or startup "
                     "failure — and must be fixed, not weighed",
}
BANDS = ((3, "LOW"), (7, "MEDIUM"), (12, "HIGH"))

# Severity lines in /sglang-pr-review output, in both the shapes it emits: the
# findings bullets (`[bug] CRITICAL — path:line — …`) and the Risk & Scope table
# (`| Critical | … |`). The table usually restates the bullets, so counting both
# and taking the larger per severity avoids double-counting without losing a
# finding that only appears in one of them.
SEV_BULLET = re.compile(r"^\s*[-*]\s*`?\[\w+\]\s*(CRITICAL|HIGH|MEDIUM|LOW)\b", re.I)
SEV_TABLE = re.compile(
    r"^\s*\|\s*(Critical|High|Medium|Low)\s*\|\s*(.*?)\s*\|?\s*$", re.I)
DECISION = re.compile(r"\b(approve|comment|request-changes)\b", re.I)


def parse_review(text: str) -> dict:
    """Severity counts from a /sglang-pr-review report."""
    bullets, table = Counter(), Counter()
    # Kept apart, not merged: the Risk & Scope table and the findings bullets are
    # two renderings of the same findings, so appending both prints every finding
    # twice in different words. The table wins when present — it is already the
    # one-line, reviewer-facing phrasing.
    det_bullet = {s: [] for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    det_table = {s: [] for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    decision = ""
    for line in text.splitlines():
        if m := SEV_BULLET.match(line):
            sev = m.group(1).upper()
            bullets[sev] += 1
            rest = line.split("—", 1)[-1].strip() if "—" in line else line.strip()
            if rest and rest not in det_bullet[sev]:
                det_bullet[sev].append(rest.rstrip("`"))
        elif m := SEV_TABLE.match(line):
            sev = m.group(1).upper()
            table[sev] += 1
            cell = m.group(2).strip().strip("|").strip()
            if cell and cell not in det_table[sev] and not set(cell) <= set("-: "):
                det_table[sev].append(cell)
        if line.lower().lstrip().startswith(("### decision", "**decision",
                                             "decision:")):
            if d := DECISION.search(line):
                decision = d.group(1).lower()
        elif not decision and "decision" in line.lower():
            if d := DECISION.search(line):
                decision = d.group(1).lower()
    details = {s: (det_table[s] or det_bullet[s])
               for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    counts = {s: max(bullets[s], table[s])
              for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    counts["decision"] = decision
    counts["source"] = "parsed"
    counts["details"] = details
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

    def row(check, vd, ev, act="", pts=0, gate=False, hard=True):
        out.append({"check": check, "verdict": vd, "evidence": ev,
                    "action": act, "points": pts, "gate": gate, "hard": hard})

    # 1. Affected Scope — how much of the world inherits this change
    if hot:
        row("Affected Scope", BAD,
            f"edits hot common code: {', '.join(hot[:3])}"
            + (f" (+{len(hot)-3})" if len(hot) > 3 else ""),
            "needs a community reviewer; check whether it can move behind an "
            "AMD-only module instead", 4)
    elif common:
        row("Affected Scope", WARN,
            f"edits {len(common)} shared file(s): {', '.join(common[:3])}",
            "confirm no NVIDIA/CPU behaviour changes", 2)
    elif additive:
        row("Affected Scope", WARN,
            f"shared files, additive only: {', '.join(additive[:3])}",
            "cheap kind — nothing existing changes behaviour; say so in review", 1)
    elif a["new_files"]:
        row("Affected Scope", WARN,
            f"{len(a['new_files'])} new file(s) outside an AMD path: "
            f"{', '.join(a['new_files'][:3])}",
            "confirm the module is only imported from an AMD-guarded call site, "
            "or move it under an AMD-named path", 1)
    else:
        row("Affected Scope", OK, f"{len(a['amd_files'])} file(s), all AMD-only paths")

    # 2. AMD guard — cite the guard with a line from the diff.
    ev = describe_guards(a)
    if not common:
        row("AMD guard", OK, ev)
    elif unguarded:
        row("AMD guard", BAD,
            f"rewrites shared code with no guard in the hunk: "
            f"{', '.join(unguarded[:3])}. {ev}",
            "wrap in `if _is_hip:` (all AMD GPUs), `if _use_aiter:` (needs the "
            "AITER library), or a device check such as `is_gfx95_supported()` "
            "(MI355 only) — or, if the bug is shared, leave it unguarded and "
            "get a community reviewer", 3)
    elif partial:
        row("AMD guard", WARN,
            f"some hunks show no guard in context: {', '.join(partial[:3])}. {ev}",
            "read those hunks — the guard may be above the window", 1)
    else:
        row("AMD guard", OK, ev)

    # 3. Guard choice — GATE: the wrong one is a crash, not a style problem
    if a["uses_aiter"] and not a["uses_is_hip"]:
        row("Guard choice", WARN, "imports aiter; no is_hip/use_aiter token added",
            "gate with `_use_aiter` — `is_hip()` alone runs this on an AMD box "
            "with no AITER installed", 0, gate=True)
    # When it passes there was nothing to choose, so it is context, not a bar —
    # a row that reads PASS on nearly every PR trains you to skim the table. It
    # climbs back into the must-pass rows only in the branch above, where the
    # wrong guard is a crash.
    elif a["uses_aiter"]:
        row("Guard choice", OK, "imports aiter, guard tokens present", hard=False)
    elif a["uses_is_hip"]:
        row("Guard choice", OK, "is_hip — works on all AMD GPUs", hard=False)
    else:
        row("Guard choice", OK,
            "no new aiter import; nothing to choose between is_hip and use_aiter",
            hard=False)

    # 4. Flags
    if a["new_envs"]:
        offs = [f"{n}={d}" for n, _, d in a["new_envs"] if "False" in d or d == ""]
        if offs:
            row("New flags", BAD,
                f"{len(a['new_envs'])} new env var(s), default-off: "
                f"{', '.join(offs[:3])}",
                "if the hardware implies it, default it on and detect with "
                "is_hip/use_aiter — do not make users export a flag",
                hard=False)
        else:
            row("New flags", WARN,
                f"new env var(s): {', '.join(n for n, _, _ in a['new_envs'])}",
                "default-on is right; still ask whether the knob is needed",
                hard=False)
    else:
        row("New flags", OK, "no new env var", hard=False)

    if a["new_globals"]:
        row("New globals", WARN,
            f"module-level globals in shared code: "
            f"{', '.join(a['new_globals'][:4])}",
            "prefer a derived constant or a config field over global state",
            hard=False)
    else:
        row("New globals", OK, "no new global in shared code", hard=False)

    # 5. Interface churn — GATE when a shared signature is rewritten
    if a["sig_changed"]:
        row("Interface churn", WARN,
            f"signature changed: {', '.join(a['sig_changed'][:3])}",
            "verify every caller is updated in this PR; prefer a keyword arg "
            "with a behaviour-preserving default", hard=False)
    else:
        row("Interface churn", OK, "no shared signature rewritten", hard=False)

    # 6. Kernel kind
    if a["new_kernels"]:
        row("Kernel", WARN, f"new kernel file(s): {', '.join(a['new_kernels'][:3])}",
            "needs a reference-correctness test and a benchmark; check the "
            "non-AMD fallback still exists", hard=False)
    elif a["kernels"]:
        row("Kernel", WARN, f"modifies {len(a['kernels'])} existing kernel file(s)",
            "needs before/after numbers on the same shapes", hard=False)
    else:
        row("Kernel", OK, "no kernel source touched", hard=False)

    # 7. Size / splittability
    tot = m["additions"] + m["deletions"]
    areas = {p.split("/")[3] if p.startswith("python/sglang/srt/") and
             len(p.split("/")) > 4 else p.split("/")[0]
             for p in a["amd_files"] + a["common_files"] + a["new_files"]
             + a["additive_files"]}
    a["oversized"] = tot > 800 or len(areas) > 3
    if a["oversized"]:
        row("Size", BAD, f"+{m['additions']}/-{m['deletions']} across {len(areas)} areas",
            "split: one concern per PR (kernel / wiring / default flip)", hard=False)
    elif tot > 300:
        row("Size", WARN, f"+{m['additions']}/-{m['deletions']}",
            "reviewable, but check it is one concern", hard=False)
    else:
        row("Size", OK, f"+{m['additions']}/-{m['deletions']}", hard=False)

    # 8. Evidence — GATE when numerics moved and nothing was measured
    need_acc = bool(a["kernels"]) or any(
        re.search(r"quant|moe|attention", p, re.I)
        for p in a["amd_files"] + a["common_files"] + a["new_files"]
        + a["additive_files"])
    if need_acc and not a["has_accuracy"]:
        row("Evidence", BAD, "numerics touched, no accuracy number in body",
            "ask for GSM8K (or equivalent) before/after", hard=False)
    elif a["kernels"] and not a["has_perf"]:
        row("Evidence", WARN, "kernel change, no perf number in body",
            "ask for before/after on the shapes it targets", hard=False)
    else:
        row("Evidence", OK,
            f"accuracy={a['has_accuracy']} perf={a['has_perf']}", hard=False)

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
        # `?`, not PASS. An unrun review is an open question, and the verdict
        # treats it as one — otherwise "nobody looked" reads the same as "clean".
        row("Critical risk", UNK,
            "not assessed — /sglang-pr-review has not read this diff",
            "run /sglang-pr-review and feed it back with --review", gate=True)
    elif rv["CRITICAL"]:
        detail = "; ".join((rv.get("details") or {}).get("CRITICAL") or [])
        row("Critical risk", BAD,
            f"{rv['CRITICAL']} CRITICAL" + (f" — {detail}" if detail else ""),
            "must be fixed before merge — a CRITICAL blocks the merge "
            "(wrong output, hang, crash, raise, or startup failure), "
            "not a risk to weigh", 0, gate=True)
    else:
        row("Critical risk", OK,
            f"none — 0 critical ({rv['HIGH']} high, {rv['MEDIUM']} medium, "
            f"{rv['LOW']} low)")

    # High/Medium/Low are context, not a bar: they shape the review conversation
    # and never block on their own.
    if rv and (rv["HIGH"] or rv["MEDIUM"] or rv["LOW"]):
        det = rv.get("details") or {}
        parts = [f"{n} {lab}: {'; '.join(det.get(k) or []) or 'see report'}"
                 for lab, k, n in (("high", "HIGH", rv["HIGH"]),
                                   ("medium", "MEDIUM", rv["MEDIUM"]),
                                   ("low", "LOW", rv["LOW"])) if n]
        pts = min(3 * rv["HIGH"], 6) + min(rv["MEDIUM"], 3)
        row("Other findings", WARN if rv["HIGH"] else OK, ". ".join(parts),
            "High findings crash or degrade under specific configurations — "
            "resolve them or get the author's rationale on record"
            if rv["HIGH"] else "", pts, hard=False)
    return out


def apply_ack(a: dict, rs: list[dict]) -> None:
    """Let the reviewer close a CHECK row they have read.

    Only a CHECK — a FAIL is a defect, and `Critical risk` is the one row whose
    whole purpose is that it cannot be waved through. Acking is recorded in the
    evidence so the table never claims the script verified something a human
    asserted.
    """
    for r in rs:
        if r["check"].lower() not in a.get("ack", []):
            continue
        if r["check"] == "Critical risk" or r["verdict"] == BAD:
            r["evidence"] += " · ack refused — this row cannot be waved through"
            continue
        r["verdict"], r["points"] = OK, 0
        r["evidence"] += " · reviewer acked"


def verdict(a: dict, rs: list[dict]) -> dict:
    open_gates = [r["check"] for r in rs if r["gate"] and r["verdict"] != OK]
    points = sum(r["points"] for r in rs if r.get("hard"))
    b = band(points)
    needs_community = bool(a["hot_common"]) or bool(a["unguarded"])
    not_pass = [r["check"] for r in rs if r.get("hard") and r["verdict"] != OK]

    if a.get("oversized"):
        name, why = "SPLIT FIRST", (
            "too much in one PR to review as a unit — see the risk picture")
    elif open_gates:
        name, why = "BLOCKED ON AUTHOR", (f"open gate(s): {', '.join(open_gates)} — "
                                          f"no score substitutes for these")
    elif needs_community:
        name, why = "NEEDS COMMUNITY REVIEWER", (
            f"{points} risk points ({b}), but it changes code every vendor "
            f"inherits — an AMD-side review cannot land it")
    elif not_pass:
        name, why = "NOT ALL PASS", (
            "hard row(s) still open: " + ", ".join(not_pass)
            + " — every row in the table has to pass before merge")
    elif not a.get("review"):
        # Cheap-to-land is not the same as correct. Saying "merge" off the diff
        # shape alone is exactly the mistake this pairing exists to prevent.
        name, why = "TRIAGE CLEAR — NEEDS CORRECTNESS REVIEW", (
            "every hard row passes, but nothing has read it for bugs yet: "
            "run /sglang-pr-review")
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


PASS_VERDICT = "LOW RISK — MERGE"


def approval(a: dict, rs: list[dict], pr: str) -> str:
    """One paragraph for the approving comment. Printed only on a clean pass.

    Assembled from rows that already passed, so it cannot claim something the
    table did not check: what the PR does, how far it reaches, what was
    measured, and what the review found.
    """
    m, by = a["meta"], {r["check"]: r for r in rs}
    s = [f"#{pr} {m['title'].split('] ')[-1].rstrip('.')} — "
         f"+{m['additions']}/-{m['deletions']} across {m['changedFiles']} files."]

    if a["common_files"] or a["additive_files"]:
        shared = (a["common_files"] + a["additive_files"])[:2]
        s.append("It reaches shared code only additively ("
                 + ", ".join(p.rsplit("/", 1)[-1] for p in shared)
                 + "), so nothing that already ships changes behaviour;")
    else:
        s.append("Every file it touches is on the AMD path, so no other vendor "
                 "inherits the change;")
    s.append(by["AMD guard"]["evidence"].rstrip(".") + ".")

    quiet = []
    if not a["new_envs"]:
        quiet.append("no new env var")
    if not a["new_globals"]:
        quiet.append("no new global")
    if not a["sig_changed"]:
        quiet.append("no shared signature rewritten")
    if quiet:
        s.append("There is " + ", ".join(quiet) + ".")

    rv = a.get("review") or {}
    tail = (f" ({rv.get('HIGH', 0)} high, {rv.get('MEDIUM', 0)} medium, "
            f"{rv.get('LOW', 0)} low — none blocking)"
            if any(rv.get(k) for k in ("HIGH", "MEDIUM", "LOW")) else "")
    s.append(f"/sglang-pr-review found no CRITICAL{tail}, "
             + ("the PR body carries accuracy and perf numbers, "
                if a["has_accuracy"] and a["has_perf"] else "")
             + ("and a test lands in the diff."
                if a["buckets"].get("test") else "and the bar is otherwise clean."))
    return " ".join(s) + " LGTM."


def risk_picture(a: dict, rs: list[dict]) -> list[tuple[str, str]]:
    """Answers for the reviewer. These do not have to pass for the PR to land."""
    by = {r["check"]: r for r in rs}
    title = a["meta"].get("title") or ""
    if re.search(r"bugfix|bug[- ]fix|\[fix\]", title, re.I):
        kind = ("Existing bug — the title marks it as a fix. "
                "Confirm it is already live on main.")
    elif re.search(r"\[feat|feature", title, re.I):
        kind = ("New feature — the title marks it as a feature. "
                "It must not change behaviour that already ships.")
    else:
        kind = ("The title does not say. Decide: new feature, or a fix for a "
                "bug that is already live on main.")

    if a["unguarded"] or (a["hot_common"] and a["common_files"]):
        plat = ("Shared code is rewritten. If the same bug happens on NVIDIA, "
                "the fix stays in the common path and needs a community reviewer. "
                "An is_hip wrapper would leave it live for everyone else.")
    elif a["amd_files"] and not a["common_files"]:
        plat = ("AMD-only path. Confirm the bug does not also reproduce on "
                "NVIDIA; if it does, this fix is in the wrong place.")
    else:
        plat = ("The diff touches shared code. Confirm whether the bug is "
                "AMD-only or hits every vendor before deciding the guard.")

    kern = by["Kernel"]["evidence"]
    if a["new_kernels"]:
        kern += ". New kernel, not an upgrade of an existing one."
    elif a["kernels"]:
        kern += ". Upgrade of an existing kernel, not a new one."
    else:
        kern += ". No new kernel and no upgrade of an existing kernel."

    if a["new_envs"]:
        flags = (by["New flags"]["evidence"]
                 + ". A hardware default belongs on is_hip / use_aiter, "
                 "not a knob the user has to export.")
    else:
        flags = "No new env var."
    if a["new_globals"]:
        flags += " " + by["New globals"]["evidence"] + "."
    else:
        flags += " No new global in shared code."

    notes = [
        ("Feature or bug", kind),
        ("AMD-only or both", plat),
        ("Kernel", kern),
        ("Flags and globals", flags),
    ]
    if a["sig_changed"]:
        notes.append(("Interface", by["Interface churn"]["evidence"]
                      + ". Every caller has to be updated in this PR."))
    if by["Evidence"]["verdict"] != OK:
        notes.append(("Evidence", by["Evidence"]["evidence"]))
    size = by["Size"]["evidence"]
    if a.get("oversized"):
        notes.append(("One concern", size + ". Split before review."))
    else:
        notes.append(("One concern", size + ". Small enough to review as one PR."))
    return notes


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
    ap.add_argument("--ack", action="append", metavar="ROW",
                    help="mark a CHECK row as read and accepted, e.g. "
                         "--ack 'Affected Scope'. Refused on FAIL rows and on "
                         "Critical risk — those are not yours to wave through")
    a = ap.parse_args()
    pr = re.sub(r"\D", "", a.pr.split("/")[-1]) or a.pr

    review = None
    if a.review:
        text = sys.stdin.read() if a.review == "-" else open(a.review).read()
        review = parse_review(text)
    elif a.critical is not None:
        review = {"CRITICAL": a.critical, "HIGH": a.high, "MEDIUM": a.medium,
                  "LOW": a.low, "decision": "", "source": "manual",
                  "details": {s: [] for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}}

    data = analyse(pr, a.repo)
    data["review"] = review
    data["ack"] = [s.strip().lower() for s in (a.ack or [])]
    rs = rows(data)
    apply_ack(data, rs)
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
    # One table. The `Must pass` column keeps the distinction that matters —
    # a row that blocks the merge vs a row that only tells you how the PR sits
    # — without splitting the reader's attention across three places.
    print("`Must pass = yes` rows block the merge. `context` rows describe how "
          "the PR sits; they are not extra gates.\n")
    print("| | Check | Must pass | Verdict | Evidence |")
    print("|---|---|---|---|---|")
    for r in rs:
        if not r.get("hard"):
            continue
        g = " **GATE**" if r["gate"] and r["verdict"] != OK else ""
        print(f"| {mark[r['verdict']]} | {r['check']}{g} | yes | {r['verdict']} "
              f"| {r['evidence']} |")
    # risk_picture() already states flags, globals, kernel, size and evidence
    # in reviewer language; printing the raw rows too would say each twice.
    for r in rs:
        if r.get("hard") or r["check"] not in ("Guard choice", "Other findings"):
            continue
        print(f"| | {r['check']} | context | {r['verdict']} | {r['evidence']} |")
    for title, body in risk_picture(data, rs):
        print(f"| | {title} | context | — | {body} |")
    print(f"\n**Verdict: {v['name']}** — {v['why']}\n")
    if v["name"] == PASS_VERDICT:
        # The only output that is meant to be pasted verbatim onto the PR.
        print("### Approval comment\n")
        print(approval(data, rs, pr) + "\n")
    if v["open_gates"]:
        print("### Gates (must be answered — nothing else buys these off)")
        for g in v["open_gates"]:
            print(f"- **{g}** — {GATES[g]}")
        print()
    acts = [r for r in rs if r.get("hard") and r["action"]
            and r["verdict"] in (BAD, WARN)]
    if acts:
        print("### Ask the author")
        for i, r in enumerate(acts, 1):
            print(f"{i}. **{r['check']}** — {r['action']}")
        print()


if __name__ == "__main__":
    main()
