#!/usr/bin/env python3
"""Production diff facts for /pr-code-path.

Prints added and removed executable lines, guard tokens, moved statements, and
removed returns. The report in SKILL.md decides common-path impact, NVIDIA
behavior, and affected hardware scope.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GUARD = re.compile(
    r"\b(_?is_hip|_?use_aiter|is_cuda|_?is_cuda|is_npu|is_xpu"
    r"|is_gfx95_supported|is_gfx95|gfx950|gfx942|gfx90a)\b"
    r"|SGLANG_USE_AITER|(?i:mi355|mi300|mi325)"
)
TEST_PATH = re.compile(r"^(test/|python/sglang/test/)")
DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
CLASS_DEF = re.compile(r"^\s*class\s+(\w+)\b")
HUNK_DEF = re.compile(r"\bdef\s+(\w+)\s*\(")
# A bare `return` is an early exit. Replacing `return a, b` with a wider tuple
# is a signature edit, not a deleted exit, so it must not raise REALIGN.
RETURN = re.compile(r"^\s*return\s*(?:#.*)?$")


def sh(args: list[str]) -> str:
    out, err = sh_try(args)
    if err:
        sys.exit(f"{' '.join(args[:4])}… failed: {err}")
    return out


def sh_try(args: list[str]) -> tuple[str, str]:
    env = dict(os.environ, GH_TOKEN="", GH_PAGER="cat", GIT_PAGER="cat")
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    if proc.returncode:
        return "", proc.stderr.strip() or f"exit {proc.returncode}"
    return proc.stdout, ""


def pr_number(text: str) -> str:
    match = re.search(r"/pull/(\d+)", text)
    return match.group(1) if match else text


def split_files(diff: str) -> list[dict]:
    files: list[dict] = []
    cur = None
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            cur = {"path": line.split(" b/")[-1], "lines": []}
            files.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    return files


def code(line: str) -> str:
    """Diff payload with the leading + or - stripped, or '' if not a change."""
    if line.startswith(("+++", "---", "@@", "diff ", "index ", "new file", "deleted file")):
        return ""
    if line[:1] in "+-":
        return line[1:]
    return ""


def executable(text: str) -> bool:
    body = text.strip()
    if not body or body.startswith("#"):
        return False
    if body.startswith(('"""', "'''")) and body.endswith(('"""', "'''")) and len(body) > 3:
        return False
    return True


def functions_in(lines: list[str]) -> list[dict]:
    """Group changed lines by the function the hunk header names."""
    groups: list[dict] = []
    current = None
    quote: str | None = None

    def open_group(name: str) -> dict:
        group = {"name": name, "added": [], "removed": [], "removed_return": False, "guards": []}
        groups.append(group)
        return group

    for line in lines:
        if quote is not None:
            payload = code(line)
            if quote in payload:
                quote = None
            continue
        if line.startswith("@@"):
            match = HUNK_DEF.search(line)
            current = open_group(match.group(1) if match else "(module)")
            continue
        payload = code(line)
        if not payload:
            continue
        stripped = payload.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            mark = stripped[:3]
            rest = stripped[3:]
            if rest.count(mark) == 0:
                quote = mark
            continue
        if not executable(payload):
            continue
        if current is None:
            current = open_group("(module)")
        added_def = DEF.match(payload) if line.startswith("+") else None
        if added_def and added_def.group(1) != current["name"]:
            current = open_group(added_def.group(1))
        bucket = "added" if line.startswith("+") else "removed"
        current[bucket].append(payload.rstrip())
        if bucket == "removed" and RETURN.match(payload):
            current["removed_return"] = True
        if bucket == "added":
            current["guards"].extend(GUARD.findall(payload))
    merged: list[dict] = []
    for group in groups:
        if merged and merged[-1]["name"] == group["name"]:
            prev = merged[-1]
            prev["added"].extend(group["added"])
            prev["removed"].extend(group["removed"])
            prev["removed_return"] = prev["removed_return"] or group["removed_return"]
            prev["guards"].extend(group["guards"])
        else:
            merged.append(group)
    return [g for g in merged if g["added"] or g["removed"]]


def prod_names(files: list[dict]) -> set[str]:
    names: set[str] = set()
    for f in files:
        if TEST_PATH.search(f["path"]):
            continue
        for line in f["lines"]:
            payload = code(line) if line[:1] in "+-" else ""
            header = HUNK_DEF.search(line) if line.startswith("@@") else None
            if header:
                names.add(header.group(1))
            if line.startswith("+"):
                for pattern in (DEF, CLASS_DEF):
                    match = pattern.match(payload)
                    if match:
                        names.add(match.group(1))
    return names


def _substantial(line: str) -> bool:
    """Skip punctuation-only lines so a moved `)` is not called a moved statement."""
    stripped = line.strip()
    return len(stripped) >= 12 and bool(re.search(r"[A-Za-z_]{3,}", stripped))


# A long new function is not on the common path until an existing file calls
# it. Keep the lines that decide the guard; leave the kernel body in the
# materialized head file.
_SUMMARY_LINE = re.compile(r"\b(assert|raise|return)\b")
_FULL_BODY_LINES = 48


def file_is_new(lines: list[str]) -> bool:
    return any(line.startswith("new file mode") for line in lines)


def _emit_file(f: dict, referenced: set[str] | None, brief: bool = False) -> None:
    """Print an existing file in full. A new file prints a function body only when an existing file names it."""
    groups = functions_in(f["lines"])
    if not groups:
        return
    full = referenced is None
    print(f"### {f['path']}\n")
    removed_text = {
        line.strip()
        for group in groups
        for line in group["removed"]
        if _substantial(line)
    }
    moved = sum(
        1
        for group in groups
        for line in group["added"]
        if line.strip() in removed_text
    )
    if moved:
        print(
            f"({moved} added line(s) repeat removed text in this file. "
            "The diff moved them. The head window in focus.md is the one to read.)\n"
            if brief
            else
            f"({moved} added line(s) repeat removed text in this file. "
            "The diff moved them. Read the function at the PR head "
            "before treating them as new.)\n"
        )
    for group in groups:
        guards = ", ".join(dict.fromkeys(group["guards"])) or "none in added lines"
        flag = ""
        if group["removed_return"]:
            flag = (
                "  REALIGN: a bare return was removed — the head window is in focus.md"
                if brief
                else "  REALIGN: a bare return was removed — read the whole function at the PR head"
            )
        added = group["added"]
        if brief:
            print(f"#### {group['name']}  (guards: {guards}){flag}\n")
            continue
        called = full or group["name"] in (referenced or ())
        # A short guarded function is the hardware check even when the caller
        # is another new function. Long unguarded bodies stay in head/.
        if not called and group["guards"] and len(added) <= _FULL_BODY_LINES:
            called = True
        omitted = called and not full and len(added) > _FULL_BODY_LINES
        if not called:
            print(
                f"#### {group['name']}  ({len(added)} lines, guards: {guards})"
                " — not called from an existing file\n"
            )
            continue
        note = ""
        if omitted:
            note = (
                f"  BODY OMITTED ({len(added)} lines) — the call site is above; "
                "open this function under head/ if the guard is not on these lines"
            )
        print(f"#### {group['name']}  (guard tokens on added lines: {guards}){flag}{note}\n")
        for line in group["removed"]:
            print(f"- {line}")
        shown = added
        if omitted:
            shown = [line for line in added if _SUMMARY_LINE.search(line) or GUARD.search(line)]
            if not shown:
                print("+ (no guard, assert, or return in this body)")
                print()
                continue
        for line in shown:
            moved_mark = (
                "  # moved: same text was removed in this file"
                if _substantial(line) and line.strip() in removed_text
                else ""
            )
            print(f"+ {line}{moved_mark}")
        print()


def _referenced_names(existing: list[dict]) -> set[str]:
    names: set[str] = set()
    for f in existing:
        for group in functions_in(f["lines"]):
            for line in group["added"]:
                names.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", line))
    return names


def print_prod(files: list[dict], brief: bool = False) -> None:
    prod = [f for f in files if not TEST_PATH.search(f["path"]) and functions_in(f["lines"])]
    existing = [f for f in prod if not file_is_new(f["lines"])]
    new = [f for f in prod if file_is_new(f["lines"])]
    print("## Read order\n")
    print(
        "Read `/tmp/pr-<number>-src/focus.md` for function bodies and line numbers. "
        "The index below names the functions. Do not open the source files, "
        "and do not run gh, git, or grep.\n"
        if brief
        else
        "Read existing files in full. A new file is not on the common path "
        "until an existing file calls it. Long new functions omit their body.\n"
    )
    if existing:
        print("Changed existing files:" if brief else "Existing, full hunks:")
        for f in existing:
            print(f"- `{f['path']}`")
        print()
    if new:
        print("New files:" if brief else "New, open a function only when an existing file calls it:")
        for f in new:
            print(f"- `{f['path']}`")
        print()
    print("## Existing files\n")
    if not existing:
        print("(no edits to existing production files)\n")
    for f in existing:
        _emit_file(f, referenced=None, brief=brief)
    print("## New files\n")
    if not new:
        print("(no new production files)\n")
    referenced = _referenced_names(existing)
    for f in new:
        _emit_file(f, referenced=referenced, brief=brief)


def _fetch_raw(repo: str, path: str, sha: str) -> tuple[str, str]:
    return sh_try([
        "gh", "api",
        "-H", "Accept: application/vnd.github.raw",
        f"/repos/{repo}/contents/{path}?ref={sha}",
    ])


def materialize(repo: str, number: str, files: list[dict]) -> None:
    """Write each production file once, at base and head, for the value trace."""
    meta_text, err = sh_try([
        "gh", "api", f"repos/{repo}/pulls/{number}",
        "--jq", "{base: .base.sha, head: .head.sha}",
    ])
    print("## Local revisions\n")
    if err:
        print(f"Could not resolve PR revisions: {err.splitlines()[0]}\n")
        return None
    meta = json.loads(meta_text)
    meta["baseRefOid"] = meta["base"]
    meta["headRefOid"] = meta["head"]
    root = Path(f"/tmp/pr-{number}-src")
    jobs: list[tuple[str, str, Path]] = []
    for f in files:
        if TEST_PATH.search(f["path"]):
            continue
        if not functions_in(f["lines"]) and not file_is_new(f["lines"]):
            continue
        jobs.append((f["path"], meta["headRefOid"], root / "head" / f["path"]))
        if not file_is_new(f["lines"]):
            jobs.append((f["path"], meta["baseRefOid"], root / "base" / f["path"]))

    def write_one(job: tuple[str, str, Path]) -> tuple[str, str]:
        path, sha, dest = job
        text, fetch_err = _fetch_raw(repo, path, sha)
        if fetch_err:
            return str(dest), fetch_err
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return str(dest), ""

    written: list[str] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for dest, fetch_err in pool.map(write_one, jobs):
            if fetch_err:
                failures.append(f"{dest}: {fetch_err}")
            else:
                written.append(dest)
    print(
        "Local copies are for focus.md. Read focus.md and stop. "
        "Do not open these files and do not fetch them again.\n"
    )
    print(f"- base `{root}/base`")
    print(f"- head `{root}/head`")
    for path in written:
        print(f"  - `{path}`")
    print()
    for failure in failures:
        print(f"fetch failed: {failure}")
    if failures:
        print()
    return root


_HUNK_AT = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DEF_AT = re.compile(r"^([ \t]*)(?:async\s+)?def\s+(\w+)\s*\(")
_CLASS_AT = re.compile(r"^([ \t]*)class\s+\w+")
_BARE_CALL = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]\s*)?\(")
_SELF_CALL = re.compile(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]\s*)?\(")
_BLOCK_HEAD = re.compile(r"^\s*(?:elif|else|if|for|while|try|with|except|finally|def|async\s+def)\b")
_WINDOW_CAP = 220
_FALLTHROUGH = 40
_WHOLE_FUNCTION = 180
_CALLEE_CAP = 8
_CALLEE_LINES = 160
_SKIP_CALLEE = {
    "if", "elif", "for", "while", "with", "return", "assert", "not", "and",
    "or", "in", "is", "lambda", "bool", "int", "any", "all", "len", "set",
    "list", "dict", "tuple", "range", "enumerate", "zip", "print", "super",
    "isinstance", "getattr", "setattr", "hasattr", "type", "min", "max",
    "sum", "abs", "round", "map", "filter", "sorted", "reversed", "open",
    "str", "float", "format", "self", "cls", "append", "extend", "pop",
    "get", "keys", "values", "items", "update", "add", "copy", "clear",
    "join", "split", "strip", "replace", "startswith", "endswith", "lower",
    "view", "reshape", "contiguous", "to", "item", "numel", "size", "shape",
    "float", "int", "cpu", "cuda", "numpy", "tolist", "clone", "detach",
    "zero_", "copy_", "empty", "zeros", "ones", "arange", "tensor", "cat",
    "stack", "where", "clamp", "mean", "expand", "permute", "transpose",
    "unsqueeze", "squeeze", "repeat", "narrow", "select", "index_select",
    "masked_fill", "masked_fill_", "fill_", "new_empty", "new_zeros",
    "new_ones", "expand_as", "type_as", "to_empty", "pin_memory",
}


def _line_sets(diff_lines: list[str]) -> tuple[set[int], set[int]]:
    """1-based line numbers touched on the base side and the head side."""
    old_set: set[int] = set()
    new_set: set[int] = set()
    old = new = 0
    for line in diff_lines:
        match = _HUNK_AT.match(line)
        if match:
            old = int(match.group(1))
            new = int(match.group(2))
            continue
        if line.startswith("\\") or old == 0:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            new_set.add(new)
            new += 1
        elif line.startswith("-") and not line.startswith("---"):
            old_set.add(old)
            old += 1
        else:
            old += 1
            new += 1
    return old_set, new_set


def _spans(text: str) -> list[tuple[str, int, int]]:
    """(name, start, end) inclusive, 1-based. A nested def ends at the next line indented no deeper."""
    lines = text.splitlines()
    defs: list[tuple[int, str, int]] = []
    for number, line in enumerate(lines, 1):
        defined = _DEF_AT.match(line)
        if defined:
            defs.append((len(defined.group(1)), defined.group(2), number))
    spans: list[tuple[str, int, int]] = []
    for indent, name, start in defs:
        end = len(lines)
        for number in range(start + 1, len(lines) + 1):
            raw = lines[number - 1]
            if not raw.strip():
                continue
            # A signature split across lines closes on `):` at the def's indent.
            if raw.strip().startswith((")", "]", "}")):
                continue
            if len(raw) - len(raw.lstrip(" ")) <= indent:
                end = number - 1
                break
        spans.append((name, start, end))
    return spans


def _span_at(spans: list[tuple[str, int, int]], line: int) -> tuple[str, int, int] | None:
    hit = None
    for name, start, end in spans:
        if start <= line <= end and (hit is None or start >= hit[1]):
            hit = (name, start, end)
    return hit


def _windows(start: int, end: int, changed: set[int]) -> list[tuple[int, int]]:
    if end - start + 1 <= _WHOLE_FUNCTION:
        return [(start, end)]
    points = sorted(number for number in changed if start <= number <= end)
    if not points:
        return []
    clusters = [[points[0]]]
    for number in points[1:]:
        if number - clusters[-1][-1] <= 40:
            clusters[-1].append(number)
        else:
            clusters.append([number])
    windows = []
    for cluster in clusters:
        windows.append((max(start, cluster[0] - 30), min(end, cluster[-1] + 30)))
    return windows


def _indent_of(line: str) -> int | None:
    if not line.strip():
        return None
    return len(line) - len(line.lstrip(" "))


def _header(lines: list[str], line_no: int, fn_start: int) -> tuple[int, int]:
    """Block header that encloses a changed line, and that header's indent."""
    gate = None
    for number in range(line_no, min(line_no + 20, len(lines) + 1)):
        raw = lines[number - 1]
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        gate = _indent_of(raw)
        break
    if gate is None:
        return line_no, 0
    for number in range(line_no, fn_start - 1, -1):
        raw = lines[number - 1]
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = _indent_of(raw)
        if indent is None or indent >= gate:
            continue
        if _BLOCK_HEAD.match(raw):
            return number, indent
        break
    return line_no, gate


def _grow(text: str, start: int, end: int, fn_start: int, fn_end: int, changed: set[int]) -> tuple[int, int]:
    """Extend a diff window through the branch it edits, plus a short fallthrough."""
    lines = text.splitlines()
    points = sorted(number for number in changed if start <= number <= end)
    if not points:
        return start, end
    header, header_indent = _header(lines, points[0], fn_start)
    block_end = header
    for number in range(header + 1, fn_end + 1):
        raw = lines[number - 1]
        if not raw.strip() or raw.lstrip().startswith("#"):
            block_end = number
            continue
        indent = _indent_of(raw)
        if indent is None or indent <= header_indent:
            break
        block_end = number
    new_start = min(start, header)
    new_end = min(fn_end, max(end, block_end) + _FALLTHROUGH)
    if new_end - new_start + 1 > _WINDOW_CAP:
        new_start = max(fn_start, points[0] - 30)
        new_end = min(fn_end, max(points[-1] + _FALLTHROUGH, new_start + _WINDOW_CAP - 1))
        if new_end - new_start + 1 > _WINDOW_CAP:
            new_start = max(fn_start, new_end - _WINDOW_CAP + 1)
    return new_start, new_end


def _numbered(text: str, start: int, end: int, marks: set[int], glyph: str) -> str:
    lines = text.splitlines()
    out = []
    for number in range(start, end + 1):
        body = lines[number - 1] if number - 1 < len(lines) else ""
        flag = glyph if number in marks else " "
        out.append(f"{number:5d}{flag}| {body}")
    return "\n".join(out)


def _call_names(code: str) -> list[str]:
    found = []
    for name in _SELF_CALL.findall(code) + _BARE_CALL.findall(code):
        if name in _SKIP_CALLEE or name.startswith("__") or name in found or name[:1].isupper():
            continue
        found.append(name)
    return found


def _added_calls(diff_lines: list[str]) -> list[str]:
    found: list[str] = []
    for line in diff_lines:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        code = line[1:].split("#", 1)[0]
        if code.lstrip().startswith("#"):
            continue
        for name in _call_names(code):
            if name not in found:
                found.append(name)
    return found


def _calls_in(text: str, start: int, end: int, prefer_from: int) -> list[str]:
    """Calls in a window. Ones at or after the first changed line come first."""
    lines = text.splitlines()
    late: list[str] = []
    early: list[str] = []
    for number in range(start, end + 1):
        if number - 1 >= len(lines):
            break
        raw = lines[number - 1]
        if raw.lstrip().startswith("#"):
            continue
        code = raw.split("#", 1)[0]
        bucket = late if number >= prefer_from else early
        for name in _call_names(code):
            if name in late or name in early:
                continue
            bucket.append(name)
    return late + early


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _local_tree() -> Path | None:
    for cand in (Path.home() / "sglang", Path("/sgl-workspace/sglang")):
        if (cand / "python" / "sglang").is_dir():
            return cand
    return None


def _locate_local(tree: Path, names: list[str]) -> dict[str, tuple[Path, int]]:
    if not names:
        return {}
    pattern = (
        r"^\s*(?:async\s+)?def\s+("
        + "|".join(re.escape(name) for name in names)
        + r")\s*\("
    )
    proc = subprocess.run(
        ["rg", "-n", "--glob", "*.py", pattern, str(tree / "python")],
        capture_output=True, text=True,
    )
    matches: dict[str, list[tuple[Path, int]]] = {}
    for line in proc.stdout.splitlines():
        path_text, number, _rest = line.split(":", 2)
        defined = _DEF_AT.match(_rest)
        # rg prints the source line after the second colon; the def is in _rest
        # only when the line itself matched. Re-parse the name from the pattern group.
        name_match = re.search(r"def\s+(\w+)\s*\(", line)
        if not name_match:
            continue
        matches.setdefault(name_match.group(1), []).append((Path(path_text), int(number)))
    chosen: dict[str, tuple[Path, int]] = {}
    for name, hits in matches.items():
        hits.sort(key=lambda item: (0 if "/srt/" in str(item[0]) and "/test/" not in str(item[0]) else 1, str(item[0])))
        chosen[name] = hits[0]
    return chosen


def write_focus(root: Path, files: list[dict]) -> Path | None:
    """Changed functions at base and head, plus one level of callees, with line numbers."""
    sections: list[str] = [
        "# Focus",
        "",
        "Read this file and stop. Do not open the source files. Do not run gh, git, or grep.",
        "Lines marked `>` were added on the head. Lines marked `<` were removed from the base.",
        "A `callee` section is one function the edited branch calls. The branch includes the fallthrough after the edit.",
        "",
    ]
    head_texts: dict[str, str] = {}
    produced = False
    callee_names: list[str] = []
    call_rank: dict[str, tuple[int, int]] = {}
    changed_names: set[str] = set()
    shown: list[tuple[str, int, int]] = []

    def note_calls(calls: list[str], weight: int, end_line: int) -> None:
        for name in calls:
            if name[:1].isupper():
                continue
            prev = call_rank.get(name)
            if prev is None or (weight, end_line) > prev:
                call_rank[name] = (weight, end_line)

    for f in files:
        if TEST_PATH.search(f["path"]):
            continue
        head_path = root / "head" / f["path"]
        base_path = root / "base" / f["path"]
        head = _read(head_path)
        base = _read(base_path)
        if head:
            head_texts[f["path"]] = head
        if not head and not base:
            continue
        old_lines, new_lines = _line_sets(f["lines"])
        head_spans = _spans(head) if head else []
        base_spans = _spans(base) if base else []
        groups: dict[str, tuple[tuple[str, int, int] | None, tuple[str, int, int] | None]] = {}
        for number in sorted(new_lines):
            span = _span_at(head_spans, number)
            key = span[0] if span else "(module)"
            prev = groups.get(key, (None, None))
            groups[key] = (span, prev[1])
        for number in sorted(old_lines):
            span = _span_at(base_spans, number)
            key = span[0] if span else "(module)"
            prev = groups.get(key, (None, None))
            groups[key] = (prev[0], span)
        for name in _added_calls(f["lines"]):
            if name not in callee_names:
                callee_names.append(name)
        changed_names.update(groups)

        for name, (head_span, base_span) in groups.items():
            if head and head_span:
                for start, end in _windows(head_span[1], head_span[2], new_lines):
                    start, end = _grow(head, start, end, head_span[1], head_span[2], new_lines)
                    points = [number for number in new_lines if start <= number <= end]
                    note_calls(_calls_in(head, start, end, min(points) if points else start), len(points), end)
                    shown.append((f["path"], start, end))
                    sections.append(
                        f"### head · {f['path']} · {name} · lines {start}-{end}"
                    )
                    sections.append("")
                    sections.append(_numbered(head, start, end, new_lines, ">"))
                    sections.append("")
                    produced = True
            elif head and name == "(module)":
                outside = {number for number in new_lines if _span_at(head_spans, number) is None}
                nlines = len(head.splitlines())
                for start, end in _windows(1, nlines, outside):
                    points = [number for number in outside if start <= number <= end]
                    note_calls(_calls_in(head, start, end, min(points) if points else start), len(points), end)
                    shown.append((f["path"], start, end))
                    sections.append(f"### head · {f['path']} · (module) · lines {start}-{end}")
                    sections.append("")
                    sections.append(_numbered(head, start, end, outside, ">"))
                    sections.append("")
                    produced = True
            if base and base_span:
                for start, end in _windows(base_span[1], base_span[2], old_lines):
                    sections.append(
                        f"### base · {f['path']} · {name} · lines {start}-{end}"
                    )
                    sections.append("")
                    sections.append(_numbered(base, start, end, old_lines, "<"))
                    sections.append("")
                    produced = True

    for name in callee_names:
        note_calls([name], 0, 0)
    candidates = [
        name for name in sorted(call_rank, key=lambda item: (-call_rank[item][0], -call_rank[item][1]))
        if name not in changed_names
    ]
    resolved: set[str] = set()

    def emit_callee(path_label: str, text: str, name: str, note: str) -> bool:
        spans = [span for span in _spans(text) if span[0] == name]
        if not spans:
            return False
        _n, start, end = spans[0]
        if end - start + 1 > _CALLEE_LINES:
            end = start + _CALLEE_LINES - 1
            note += f" Truncated to {_CALLEE_LINES} lines."
        sections.append(f"### callee · {name} · {path_label} · lines {start}-{end}")
        sections.append("")
        sections.append(note)
        sections.append("")
        sections.append(_numbered(text, start, end, set(), " "))
        sections.append("")
        return True

    def already_shown(path_label: str, start: int, end: int) -> bool:
        return any(
            path_label == shown_path and shown_start <= start and end <= shown_end
            for shown_path, shown_start, shown_end in shown
        )

    skipped: set[str] = set()
    for name in candidates:
        if len(resolved) >= _CALLEE_CAP:
            break
        for path, text in head_texts.items():
            spans = [span for span in _spans(text) if span[0] == name]
            if spans and already_shown(path, spans[0][1], spans[0][2]):
                skipped.add(name)
                break
            if emit_callee(path, text, name, "Called from the edited branch. PR head, function itself unchanged."):
                resolved.add(name)
                produced = True
                break

    missing = [name for name in candidates if name not in resolved and name not in skipped]
    tree = _local_tree()
    located = _locate_local(tree, missing) if tree and missing else {}
    for name in missing:
        if len(resolved) >= _CALLEE_CAP or name not in located:
            continue
        path, _line = located[name]
        text = _read(path)
        rel = str(path)
        if emit_callee(rel, text, name, "Called from the edited branch. Local tree; it may differ from the PR head."):
            resolved.add(name)
            produced = True

    if not produced:
        return None
    dest = root / "focus.md"
    dest.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", nargs="?", help="PR number or GitHub URL")
    parser.add_argument("--repo", default="sgl-project/sglang")
    parser.add_argument("--diff", help="read this diff instead of calling gh")
    args = parser.parse_args()
    if not args.pr and not args.diff:
        parser.error("pass a PR number or --diff")

    title = ""
    label = args.pr or args.diff
    if args.diff:
        diff = open(args.diff, encoding="utf-8").read()
    else:
        number = pr_number(args.pr)
        label = f"{args.repo}#{number}"
        meta = json.loads(sh([
            "gh", "pr", "view", number, "--repo", args.repo,
            "--json", "title",
        ]))
        title = meta.get("title", "")
        diff = sh(["gh", "pr", "diff", number, "--repo", args.repo])

    files = split_files(diff)
    root = None
    if not args.diff:
        root = materialize(args.repo, number, files)
    focus = write_focus(root, files) if root else None
    print(f"# path-cover facts — {label}")
    if title:
        print(f"\n{title}")
    if focus:
        print(
            f"\nRead `{focus}` and stop.\n"
            "It has the changed functions at base and head, with line numbers, "
            "and one level of callees. Do not open the source files. "
            "Do not run gh, git diff, or grep.\n"
        )
    else:
        print(
            "\nFacts only. A guard token on an added line is not coverage of the "
            "lines around it. Decide common-path impact, NVIDIA behavior, and "
            "hardware scope in the report.\n"
            "Chat reply is Final conclusion first, then Analysis, then one canvas link. "
            "Value traces go in the canvas, not the chat. See SKILL.md Report.\n"
        )
    names = prod_names(files)
    print_prod(files, brief=focus is not None)
    print("## Production names touched\n")
    print(", ".join(f"`{name}`" for name in sorted(names)) or "(none)")
    print()
    if focus:
        print(f"Read `{focus}` and stop. Do not run gh, git diff, or grep.\n")


if __name__ == "__main__":
    main()
