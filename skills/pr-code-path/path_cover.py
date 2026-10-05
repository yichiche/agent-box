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
    env = dict(os.environ, GH_TOKEN="", GH_PAGER="cat", GIT_PAGER="cat")
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    if proc.returncode:
        sys.exit(f"{' '.join(args[:4])}… failed: {proc.stderr.strip()}")
    return proc.stdout


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


def print_prod(files: list[dict]) -> None:
    print("## Production edits\n")
    any_prod = False
    for f in files:
        if TEST_PATH.search(f["path"]):
            continue
        groups = functions_in(f["lines"])
        if not groups:
            continue
        any_prod = True
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
                "The diff moved them. Read the function at the PR head "
                "before treating them as new.)\n"
            )
        for group in groups:
            guards = ", ".join(dict.fromkeys(group["guards"])) or "none in added lines"
            flag = ""
            if group["removed_return"]:
                flag = "  REALIGN: a bare return was removed — read the whole function at the PR head"
            print(f"#### {group['name']}  (guard tokens on added lines: {guards}){flag}\n")
            for line in group["removed"]:
                print(f"- {line}")
            for line in group["added"]:
                moved = "  # moved: same text was removed in this file" if _substantial(line) and line.strip() in removed_text else ""
                print(f"+ {line}{moved}")
            print()
    if not any_prod:
        print("(no production edits)\n")


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
    print(f"# path-cover facts — {label}")
    if title:
        print(f"\n{title}")
    print(
        "\nFacts only. A guard token on an added line is not coverage of the "
        "lines around it. Decide common-path impact, NVIDIA behavior, and "
        "hardware scope in the report.\n"
        "Chat reply is only the conclusion bullets and one canvas link. "
        "Value traces go in the canvas, not the chat. See SKILL.md Report.\n"
    )
    names = prod_names(files)
    print_prod(files)
    print("## Production names touched\n")
    print(", ".join(f"`{name}`" for name in sorted(names)) or "(none)")
    print()


if __name__ == "__main__":
    main()
