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

    # Guard state only matters for edits to code that already existed and is
    # not already AMD-only by path.
    unguarded = [f["path"] for f in nonamd_mod if guard_state(f) == "unguarded"]
    partial = [f["path"] for f in nonamd_mod if guard_state(f) == "partial"]

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
        "common_files": [f["path"] for f in nonamd_mod],
        "new_files": [f["path"] for f in nonamd_new],
        "hot_common": [f["path"] for f in nonamd_mod
                       if classify_path(f["path"]) == "hot-common"],
        "unguarded": unguarded, "partial": partial,
        "new_envs": new_envs, "new_globals": new_globals,
        "sig_changed": sig_changed,
        "kernels": [f["path"] for f in kernels], "new_kernels": new_kernels,
        "uses_aiter": uses_aiter, "uses_is_hip": uses_is_hip,
        "has_accuracy": has_accuracy, "has_perf": has_perf,
    }


def rows(a: dict) -> list[tuple[str, str, str, str]]:
    """(check, verdict, evidence, remediation-if-not-pass)"""
    m, out = a["meta"], []
    hot, common, unguarded, partial = (a["hot_common"], a["common_files"],
                                       a["unguarded"], a["partial"])

    # 1. Blast radius
    if not common and a["new_files"]:
        out.append(("Blast radius", WARN,
                    f"no existing shared file edited; {len(a['new_files'])} new "
                    f"file(s) outside an AMD path: {', '.join(a['new_files'][:3])}",
                    "confirm the new module is only imported from an AMD-guarded "
                    "call site, or move it under an AMD-named path"))
    elif not common:
        out.append(("Blast radius", OK,
                    f"{len(a['amd_files'])} file(s), all AMD-only paths", ""))
    elif hot:
        out.append(("Blast radius", BAD,
                    f"touches hot common code: {', '.join(hot[:3])}"
                    + (f" (+{len(hot)-3})" if len(hot) > 3 else ""),
                    "needs a community reviewer; see if the change can move "
                    "behind an AMD-only module instead"))
    else:
        out.append(("Blast radius", WARN,
                    f"{len(common)} common file(s): {', '.join(common[:3])}",
                    "confirm no NVIDIA/CPU behaviour changes"))

    # 2. Guards
    if not common:
        out.append(("AMD guard", OK, "n/a — no common code touched", ""))
    elif unguarded:
        out.append(("AMD guard", BAD,
                    f"added code with no is_hip/use_aiter in hunk: "
                    f"{', '.join(unguarded[:3])}",
                    "wrap in `if _is_hip:` (all AMD GPUs) or `if _use_aiter:` "
                    "(needs the AITER library), or justify why it is shared"))
    elif partial:
        out.append(("AMD guard", WARN,
                    f"some hunks show no guard in context: {', '.join(partial[:3])}",
                    "read those hunks — the guard may be above the window"))
    else:
        out.append(("AMD guard", OK, "every common-code hunk sits under a guard", ""))

    # 3. Guard choice
    if a["uses_aiter"] and not a["uses_is_hip"]:
        out.append(("Guard choice", WARN, "imports aiter; uses_aiter expected",
                    "gate with `_use_aiter` — `is_hip()` alone will run this on "
                    "an AMD box that has no AITER installed"))
    elif a["uses_aiter"]:
        out.append(("Guard choice", OK, "imports aiter, both guards present", ""))
    elif a["uses_is_hip"]:
        out.append(("Guard choice", OK, "is_hip — works on all AMD GPUs", ""))
    else:
        out.append(("Guard choice", UNK, "no guard token added", ""))

    # 4. Flags
    if a["new_envs"]:
        offs = [f"{n}={d}" for n, _, d in a["new_envs"] if "False" in d or d == ""]
        if offs:
            out.append(("New flags", BAD,
                        f"{len(a['new_envs'])} new env var(s), default-off: "
                        f"{', '.join(offs[:3])}",
                        "if the hardware implies it, default it on and detect with "
                        "is_hip/use_aiter — do not make users export a flag"))
        else:
            out.append(("New flags", WARN,
                        f"new env var(s): {', '.join(n for n, _, _ in a['new_envs'])}",
                        "default-on is right; still ask whether the knob is needed"))
    else:
        out.append(("New flags", OK, "no new env var", ""))

    if a["new_globals"]:
        out.append(("New globals", WARN,
                    f"module-level globals in common code: "
                    f"{', '.join(a['new_globals'][:4])}",
                    "prefer a derived constant or a config field over global state"))
    else:
        out.append(("New globals", OK, "no new global in common code", ""))

    # 5. Interface churn
    if a["sig_changed"]:
        out.append(("Interface churn", WARN,
                    f"signature changed: {', '.join(a['sig_changed'][:3])}",
                    "every caller must be updated; prefer a keyword arg with a "
                    "default that preserves today's behaviour"))
    else:
        out.append(("Interface churn", OK, "no common signature rewritten", ""))

    # 6. Kernel kind
    if a["new_kernels"]:
        out.append(("Kernel", WARN,
                    f"new kernel file(s): {', '.join(a['new_kernels'][:3])}",
                    "needs a reference-correctness test and a benchmark; check "
                    "the non-AMD fallback still exists"))
    elif a["kernels"]:
        out.append(("Kernel", WARN,
                    f"modifies {len(a['kernels'])} existing kernel file(s)",
                    "needs before/after numbers on the same shapes"))
    else:
        out.append(("Kernel", OK, "no kernel source touched", ""))

    # 7. Size / splittability
    tot = m["additions"] + m["deletions"]
    areas = {p.split("/")[3] if p.startswith("python/sglang/srt/") and
             len(p.split("/")) > 4 else p.split("/")[0]
             for p in a["amd_files"] + a["common_files"] + a["new_files"]}
    if tot > 800 or len(areas) > 3:
        out.append(("Size", BAD,
                    f"+{m['additions']}/-{m['deletions']} across {len(areas)} areas",
                    "split: one concern per PR (kernel / model wiring / flag)"))
    elif tot > 300:
        out.append(("Size", WARN, f"+{m['additions']}/-{m['deletions']}",
                    "reviewable, but check it is one concern"))
    else:
        out.append(("Size", OK, f"+{m['additions']}/-{m['deletions']}", ""))

    # 8. Evidence
    need_acc = bool(a["kernels"]) or any(
        re.search(r"quant|moe|attention", p, re.I)
        for p in a["amd_files"] + a["common_files"] + a["new_files"])
    if need_acc and not a["has_accuracy"]:
        out.append(("Evidence", BAD, "numerics touched, no accuracy number in body",
                    "ask for GSM8K (or equivalent) before/after"))
    elif a["kernels"] and not a["has_perf"]:
        out.append(("Evidence", WARN, "kernel change, no perf number in body",
                    "ask for before/after on the shapes it targets"))
    else:
        out.append(("Evidence", OK,
                    f"accuracy={a['has_accuracy']} perf={a['has_perf']}", ""))

    # 9. Tests
    if a["buckets"].get("test"):
        out.append(("Tests", OK, f"{a['buckets']['test']} test file(s) touched", ""))
    else:
        out.append(("Tests", WARN, "no test file in the diff",
                    "AMD-only tests belong in test/registered/amd/"))
    return out


def verdict(rs: list[tuple[str, str, str, str]]) -> tuple[str, str]:
    bad = [r[0] for r in rs if r[1] == BAD]
    warn = [r[0] for r in rs if r[1] == WARN]
    if "Blast radius" in bad:
        return ("NEEDS COMMUNITY REVIEWER",
                "touches code every vendor inherits — an AMD-only review is not "
                "enough to land it")
    if "Size" in bad:
        return ("SPLIT FIRST", "too large or too many concerns to review as one PR")
    if bad:
        return ("BLOCKED ON AUTHOR", f"unresolved: {', '.join(bad)}")
    if warn:
        return ("MERGEABLE AFTER CHECKS", f"read before approving: {', '.join(warn)}")
    return ("EASY MERGE", "AMD-only, guarded, one concern, evidence attached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pr")
    ap.add_argument("--repo", default="sgl-project/sglang")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pr = re.sub(r"\D", "", a.pr.split("/")[-1]) or a.pr

    data = analyse(pr, a.repo)
    rs = rows(data)
    v, why = verdict(rs)

    if a.json:
        print(json.dumps({"pr": pr, "verdict": v, "why": why,
                          "rows": [dict(zip(("check", "verdict", "evidence",
                                             "action"), r)) for r in rs]}, indent=2))
        return

    m = data["meta"]
    mark = {OK: "- [x]", WARN: "- [ ]", BAD: "- [ ]", UNK: "- [ ]"}
    print(f"## PR #{pr} — merge triage\n")
    print(f"**{m['title']}** — @{m['author']['login']}, "
          f"+{m['additions']}/-{m['deletions']} over {m['changedFiles']} files"
          f"{' (DRAFT)' if m['isDraft'] else ''}\n")
    print("| | Check | Verdict | Evidence |")
    print("|---|---|---|---|")
    for check, vd, ev, _ in rs:
        print(f"| {mark[vd]} | {check} | {vd} | {ev} |")
    print(f"\n**Verdict: {v}** — {why}\n")
    acts = [(c, act) for c, vd, _, act in rs if act and vd in (BAD, WARN)]
    if acts:
        print("### Ask the author")
        for i, (c, act) in enumerate(acts, 1):
            print(f"{i}. **{c}** — {act}")
    print("\n> Mechanical half only. Still yours to judge: is the guard the right "
          "one, is this one concern or three, and is the bug AMD-only or shared "
          "(see SKILL.md 'What the script cannot decide').")


if __name__ == "__main__":
    main()
