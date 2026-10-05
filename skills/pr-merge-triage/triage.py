#!/usr/bin/env python3
"""Automated pre-review gate for an sgl-project/sglang PR.

Prints a fixed checklist. A passing result means a human can start reading
the code. It is never a merge approval.

Shape (size, flags, evidence) comes from the diff. Affected scope and the AMD
guard come only from a /pr-code-path judgment JSON. Unit-test quality comes
only from a /pr-test-seam judgment JSON. Missing judgments stay BLOCKED.

    python3 triage.py 41870 --shape
    python3 triage.py 41870 --code-path /tmp/pr-41870-code-path.json \
        --test-seam /tmp/pr-41870-test-seam.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

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


# --- pre-review gate ---------------------------------------------------------
#
# This script does not approve a merge. PASS means the automated pre-review
# can hand the PR to a human. Scope and guard come only from a /pr-code-path
# judgment file. Test quality comes only from a /pr-test-seam judgment file.
# A missing judgment is BLOCKED, never waved through.

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"
SPLIT = "BLOCKED — SPLIT FIRST"
FAILED = "BLOCKED — REQUIREMENTS FAILED"
INCOMPLETE = "BLOCKED — AUTOMATION INCOMPLETE"
READY = "READY FOR HUMAN REVIEW"
PROCEED = "PROCEED"

HARDWARE_SCOPES = {"all_backends", "all_amd", "aiter", "gfx950", "gfx94", "other"}
CODE_CONCLUSIONS = {"can_merge", "cannot_merge"}
SEAM_CONCLUSIONS = {"complete", "mixed", "past_the_seam", "reimplements", "no_test"}


def row(check: str, verdict: str, evidence: str, action: str = "") -> dict:
    return {"check": check, "verdict": verdict, "evidence": evidence, "action": action}


def _areas(paths: list[str]) -> set[str]:
    areas = set()
    for path in paths:
        parts = path.split("/")
        if path.startswith("python/sglang/srt/") and len(parts) > 4:
            areas.add(parts[3])
        else:
            areas.add(parts[0])
    return areas


def shape_rows(a: dict) -> list[dict]:
    meta = a["meta"]
    total = meta["additions"] + meta["deletions"]
    paths = (a["amd_files"] + a["common_files"] + a["new_files"] + a["additive_files"])
    a["oversized"] = total > 800 or len(_areas(paths)) > 3
    rows = []
    if a["oversized"]:
        rows.append(row(
            "One concern", FAIL,
            f"+{meta['additions']}/-{meta['deletions']} across "
            f"{len(_areas(paths))} areas",
            "Split this PR before review. Land one concern at a time: "
            "kernel and its test, then the wiring, then any default change.",
        ))

    default_off = [f"{name}={default}" for name, _, default in a["new_envs"]
                   if "False" in default or default == ""]
    if default_off:
        rows.append(row(
            "Flags", FAIL,
            "New default-off env var(s): " + ", ".join(default_off[:3]),
            "Detect the hardware and enable the path with is_hip, use_aiter, "
            "or a gfx check. Add an env var only when the user must choose a "
            "policy the code cannot see, and say when the default will flip.",
        ))
    elif a["new_envs"]:
        names = ", ".join(name for name, _, _ in a["new_envs"][:3])
        rows.append(row("Flags", PASS, f"New env var(s) are not default-off: {names}."))
    else:
        rows.append(row("Flags", PASS, "No new env var."))

    touched = a["kernels"] + a["amd_files"] + a["common_files"] + a["new_files"] + a["additive_files"]
    needs_accuracy = bool(a["kernels"]) or any(
        re.search(r"quant|moe|attention", path, re.I) for path in touched)
    if needs_accuracy and not a["has_accuracy"]:
        rows.append(row(
            "Accuracy evidence", FAIL,
            "Numerics are touched and the PR body has no accuracy number.",
            "Add a GSM8K (or equivalent) before/after result before human review.",
        ))
    elif needs_accuracy:
        rows.append(row("Accuracy evidence", PASS, "The PR body reports an accuracy result."))
    else:
        rows.append(row("Accuracy evidence", PASS, "No quantization, MoE, attention, or kernel path is touched."))

    if a["kernels"] and not a["has_perf"]:
        rows.append(row(
            "Performance evidence", FAIL,
            "A kernel file changed and the PR body has no performance number.",
            "Add before/after throughput, latency, or TTFT on the shapes this kernel targets.",
        ))
    elif a["kernels"]:
        rows.append(row("Performance evidence", PASS, "The PR body reports a performance result."))
    else:
        rows.append(row("Performance evidence", PASS, "No kernel source is touched."))
    return rows


def _bool_field(data: dict, key: str, problems: list[str]) -> None:
    if key not in data:
        problems.append(key)
    elif not isinstance(data[key], bool):
        problems.append(f"{key} must be a JSON boolean")


def _behavior_ok(value) -> bool:
    return value is True


def validate_code_path(data: dict) -> list[str]:
    problems = []
    for key in ("conclusion", "hardware_scope", "hardware_scope_detail", "summary"):
        if not str(data.get(key) or "").strip():
            problems.append(key)
    for key in ("common_path_changed", "nvidia_execution_identical",
                "nvidia_interface_identical"):
        _bool_field(data, key, problems)
    if "nvidia_behavior_identical" not in data:
        problems.append("nvidia_behavior_identical")
    elif data["nvidia_behavior_identical"] not in (True, False, "unproven"):
        problems.append('nvidia_behavior_identical must be true, false, or "unproven"')
    if data.get("conclusion") not in CODE_CONCLUSIONS:
        problems.append("conclusion must be can_merge or cannot_merge")
    if data.get("hardware_scope") not in HARDWARE_SCOPES:
        problems.append(
            "hardware_scope must be all_backends, all_amd, aiter, gfx950, gfx94, or other")
    needs_action = (data.get("conclusion") == "cannot_merge"
                    or not _behavior_ok(data.get("nvidia_behavior_identical")))
    if needs_action and not str(data.get("owner_action") or "").strip():
        problems.append("owner_action")
    if not _behavior_ok(data.get("nvidia_behavior_identical")):
        if not str(data.get("nvidia_diff_line") or "").strip():
            problems.append("nvidia_diff_line")
        if not str(data.get("nvidia_diff_why") or "").strip():
            problems.append("nvidia_diff_why")
    return problems


def validate_test_seam(data: dict) -> list[str]:
    problems = []
    for key in ("conclusion", "changed_behavior", "correct_seam"):
        if not str(data.get(key) or "").strip():
            problems.append(key)
    if data.get("conclusion") not in SEAM_CONCLUSIONS:
        problems.append(
            "conclusion must be complete, mixed, past_the_seam, reimplements, or no_test")
    if data.get("conclusion") != "complete" and not str(data.get("missing_test") or "").strip():
        problems.append("missing_test")
    return problems


def load_judgment(path: str | None, kind: str, validate) -> tuple[dict | None, str]:
    if not path:
        return None, f"{kind} judgment JSON was not provided"
    try:
        data = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{kind} judgment JSON could not be read: {exc}"
    if not isinstance(data, dict):
        return None, f"{kind} judgment JSON must be an object"
    problems = validate(data)
    if problems:
        return None, f"{kind} judgment JSON is incomplete: {', '.join(problems)}"
    return data, ""


def _layer(same: bool) -> str:
    return "identical" if same else "not identical"


def code_path_rows(cp: dict | None, error: str, uses_aiter: bool) -> list[dict]:
    if cp is None:
        blocked = f"Blocked: {error}."
        action = "Follow /pr-code-path and write its judgment JSON."
        return [
            row("Affected Scope", BLOCKED, blocked, action),
            row("AMD Guard", BLOCKED, blocked, action),
        ]
    behavior = cp["nvidia_behavior_identical"]
    summary = cp["summary"].strip()
    layers = (
        f"Common path changed: {'yes' if cp['common_path_changed'] else 'no'}. "
        f"NVIDIA execution flow {_layer(cp['nvidia_execution_identical'])}; "
        f"internal interface {_layer(cp['nvidia_interface_identical'])}; "
        f"numerical results and original behavior "
        f"{'identical' if behavior is True else 'not identical' if behavior is False else 'unproven'}. "
        f"{summary}"
    )
    if _behavior_ok(behavior):
        scope = row("Affected Scope", PASS, layers)
    else:
        scope = row("Affected Scope", FAIL, layers, cp.get("owner_action", "").strip())
        scope["nvidia_diff_line"] = str(cp.get("nvidia_diff_line") or "").strip()
        scope["nvidia_diff_why"] = str(cp.get("nvidia_diff_why") or "").strip()

    detail = cp["hardware_scope_detail"].strip()
    scope_ok = cp["conclusion"] == "can_merge" and _behavior_ok(behavior)
    aiter_ok = not (uses_aiter and cp["hardware_scope"] != "aiter")
    if scope_ok and aiter_ok:
        guard = row("AMD Guard", PASS, f"Code path accepts this scope: {detail}.")
    else:
        parts = []
        actions = []
        if not scope_ok:
            parts.append(f"Code path did not accept this scope ({detail}). {summary}")
            action = cp.get("owner_action", "").strip()
            if action:
                actions.append(action)
        if not aiter_ok:
            parts.append(f"The diff imports AITER, but the code-path scope is {detail}.")
            actions.append(
                "Gate the AITER import with _use_aiter. is_hip alone crashes on an "
                "AMD machine that does not have AITER installed."
            )
        guard = row("AMD Guard", FAIL, " ".join(parts), " ".join(actions))
    return [scope, guard]


def test_rows(seam: dict | None, error: str) -> list[dict]:
    if seam is None:
        return [row(
            "Unit Test Quality", BLOCKED, f"Blocked: {error}.",
            "Follow /pr-test-seam and write its judgment JSON.",
        )]
    conclusion = seam["conclusion"]
    evidence = (
        f"Test-seam conclusion: {conclusion}. "
        f"Changed behavior: {seam['changed_behavior'].strip()} "
        f"Correct seam: {seam['correct_seam'].strip()}."
    )
    if conclusion == "complete":
        return [row("Unit Test Quality", PASS, evidence)]
    return [row("Unit Test Quality", FAIL, evidence, seam["missing_test"].strip())]


def notes(a: dict) -> list[str]:
    title = a["meta"].get("title") or ""
    if re.search(r"bugfix|bug[- ]fix|\[fix\]", title, re.I):
        kind = "The title marks an existing-bug fix. Human review should confirm it is live on main."
    elif re.search(r"\[feat|feature", title, re.I):
        kind = "The title marks a new feature. It must not change behavior that already ships."
    else:
        kind = "The title does not say whether this is a new feature or a live bug fix."
    out = [kind]
    if a["new_kernels"]:
        out.append("Kernel: new kernel file(s): " + ", ".join(a["new_kernels"][:3]) + ".")
    elif a["kernels"]:
        out.append(f"Kernel: modifies {len(a['kernels'])} existing kernel file(s).")
    else:
        out.append("Kernel: no kernel source is touched.")
    if a["sig_changed"]:
        out.append("Interface: shared signature rewritten: " + ", ".join(a["sig_changed"][:3]) + ".")
    if a["new_globals"]:
        out.append("Globals: new shared global(s): " + ", ".join(a["new_globals"][:3]) + ".")
    return out


DISPLAY_ORDER = (
    "Affected Scope",
    "AMD Guard",
    "Unit Test Quality",
    "Accuracy evidence",
    "Performance evidence",
    "Flags",
)


def order_rows(rows: list[dict]) -> list[dict]:
    by_name = {item["check"]: item for item in rows}
    ordered = []
    concern = by_name.get("One concern")
    if concern is not None and concern["verdict"] != PASS:
        ordered.append(concern)
    ordered.extend(by_name[name] for name in DISPLAY_ORDER if name in by_name)
    return ordered


def verdict_of(rows: list[dict]) -> str:
    by_name = {item["check"]: item for item in rows}
    if by_name.get("One concern", {}).get("verdict") == FAIL:
        return SPLIT
    if any(item["verdict"] == BLOCKED for item in rows):
        return INCOMPLETE
    if any(item["verdict"] == FAIL for item in rows):
        return FAILED
    return READY


def owner_comment(pr: str, title: str, verdict: str, rows: list[dict]) -> str:
    failed = [item for item in rows if item["verdict"] == FAIL]
    lines = [
        f"Pre-review blocked #{pr} — {title}.",
        "",
        "Failed checks:",
    ]
    for index, item in enumerate(failed, 1):
        lines.append(f"{index}. {item['check']}")
        if item["check"] == "Affected Scope" and item.get("nvidia_diff_line"):
            lines.append(f"   Line: {item['nvidia_diff_line']}")
            lines.append(f"   Why: {item['nvidia_diff_why']}")
            lines.append(f"   Fix: {item['action']}")
            continue
        lines.append(f"   {item['evidence']}")
        if item["action"]:
            lines.append(f"   Required: {item['action']}")
    if verdict == SPLIT:
        lines.extend(["", "Split the PR before the other findings are reviewed."])
    return "\n".join(lines)



CI_LINK = re.compile(
    r"Latest PR Test \(([^)]+)\):.*?\[([^\]]+)\]\((https://github\.com/[^)\s]+)\)",
    re.S,
)


def ci_links(repo: str, pr: str, body: str) -> list[str]:
    """PR checks page, plus the summary runs recorded in the PR body."""
    lines = [f"- [CI checks](https://github.com/{repo}/pull/{pr}/checks)"]
    seen = set()
    for match in CI_LINK.finditer(body or ""):
        label, title, url = match.group(1), match.group(2), match.group(3)
        if url in seen:
            continue
        seen.add(url)
        lines.append(f"- [CI — {label} ({title})]({url})")
    return lines


def code_path_canvas(pr: str) -> Path:
    name = f"pr-{pr}-code-path.canvas.tsx"
    found = sorted(Path.home().glob(f".cursor/projects/*/canvases/{name}"))
    if found:
        return found[0]
    slug = str(Path.cwd().resolve()).lstrip("/").replace("/", "-")
    return Path.home() / ".cursor/projects" / slug / "canvases" / name


def link_lines(pr: str, a: dict) -> list[str]:
    repo = a.get("repo") or "sgl-project/sglang"
    body = (a.get("meta") or {}).get("body") or ""
    lines = ["", "### Links", *ci_links(repo, pr, body)]
    canvas = code_path_canvas(pr)
    label = f"PR {pr} code path"
    status_path = canvas.with_name(canvas.name.replace(".tsx", ".status.json"))
    missing = False
    if status_path.is_file():
        try:
            missing = json.loads(status_path.read_text()).get("status") == "canvas-missing"
        except (OSError, json.JSONDecodeError):
            missing = False
    if canvas.is_file() and not missing:
        lines.append(f"- [{label}]({canvas})")
    elif missing:
        lines.append(
            f"- [{label}]({canvas}) — host status is canvas-missing; rewrite the canvas in this session"
        )
    else:
        lines.append(f"- [{label}]({canvas}) — canvas file is not written yet")
    return lines


def render(pr: str, a: dict, rows: list[dict], verdict: str, shape_only: bool) -> str:
    meta = a["meta"]
    lines = [
        f"## PR #{pr} — pre-review",
        "",
        f"**{meta['title']}** — @{meta['author']['login']}, "
        f"+{meta['additions']}/-{meta['deletions']} over {meta['changedFiles']} files"
        + (" (DRAFT)" if meta["isDraft"] else ""),
        "",
        f"**Verdict: {verdict if not shape_only else (SPLIT if a.get('oversized') else PROCEED)}**",
        "",
    ]
    shown = verdict if not shape_only else (SPLIT if a.get("oversized") else PROCEED)
    if shown == READY:
        lines.append(
            "Automated checks passed. Human review can start. "
            "This is not a merge approval and it is not an LGTM."
        )
    elif shown == PROCEED:
        lines.append(
            "Shape checks passed. Write the code-path and test-seam judgment "
            "files, then re-run without --shape."
        )
    elif shown == INCOMPLETE:
        lines.append(
            "The automated check could not finish. Do not treat any PASS row "
            "as permission to skip the blocked rows."
        )
    else:
        lines.append(
            "Automated pre-review blocked this PR. Send the owner comment. "
            "Do not start a full code review until the failed checks are fixed."
        )
    lines.extend(link_lines(pr, a))
    lines.extend(["", "| Check | Verdict | Evidence |", "|---|---|---|"])
    for item in rows:
        lines.append(f"| {item['check']} | {item['verdict']} | {item['evidence']} |")
    failed = [item for item in rows if item["verdict"] in (FAIL, BLOCKED) and item["action"]]
    if failed:
        lines.extend(["", "### What has to change"])
        for index, item in enumerate(failed, 1):
            lines.append(f"{index}. **{item['check']}** — {item['action']}")
    if shown in (SPLIT, FAILED):
        lines.extend(["", "### Owner comment", "", "```",
                      owner_comment(pr, meta["title"], shown, rows), "```"])
    if not shape_only:
        lines.extend(["", "### Notes"])
        lines.extend(f"- {note}" for note in notes(a))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr")
    parser.add_argument("--repo", default="sgl-project/sglang")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--shape", action="store_true",
                        help="shape checks only; stop before code-path and test-seam")
    parser.add_argument("--code-path", metavar="PATH",
                        help="/pr-code-path judgment JSON")
    parser.add_argument("--test-seam", metavar="PATH",
                        help="/pr-test-seam judgment JSON")
    args = parser.parse_args()
    pr = re.sub(r"\D", "", args.pr.split("/")[-1]) or args.pr
    data = analyse(pr, args.repo)
    rows = shape_rows(data)
    if args.shape or data.get("oversized"):
        shown = rows
        verdict = SPLIT if data.get("oversized") else PROCEED
        if args.shape or data.get("oversized"):
            payload = {"pr": pr, "verdict": verdict, "rows": shown}
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(render(pr, data, shown, verdict, shape_only=True))
            if data.get("oversized"):
                return
            if args.shape:
                return
    code_path, code_error = load_judgment(args.code_path, "code-path", validate_code_path)
    test_seam, test_error = load_judgment(args.test_seam, "test-seam", validate_test_seam)
    rows.extend(code_path_rows(code_path, code_error, data["uses_aiter"]))
    rows.extend(test_rows(test_seam, test_error))
    rows = order_rows(rows)
    verdict = verdict_of(rows)
    if args.json:
        print(json.dumps({"pr": pr, "verdict": verdict, "rows": rows}, indent=2))
        return
    print(render(pr, data, rows, verdict, shape_only=False))


if __name__ == "__main__":
    main()
