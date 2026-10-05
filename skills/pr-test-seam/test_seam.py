#!/usr/bin/env python3
"""Print test-entry facts for /pr-test-seam; the skill writes the verdict."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

TEST_PATH = re.compile(r"^(test/|python/sglang/test/)")
DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
TEST_DEF = re.compile(r"^def\s+(test_\w+)\s*\(")
HELPER_DEF = re.compile(r"^def\s+(_\w+)\s*\(")
HUNK_DEF = re.compile(r"\bdef\s+(\w+)\s*\(")
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
SKIP_CALL = {
    "if", "elif", "for", "while", "with", "return", "assert", "not", "and",
    "or", "in", "is", "lambda", "bool", "int", "any", "all", "len", "set",
    "list", "dict", "tuple", "range", "enumerate", "zip", "print", "super",
    "isinstance", "getattr", "setattr", "type",
}


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
    current = None
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            current = {"path": line.split(" b/")[-1], "lines": []}
            files.append(current)
        elif current is not None:
            current["lines"].append(line)
    return files


def payload(line: str) -> str:
    if line.startswith(("+++", "---", "@@", "diff ", "index ")):
        return ""
    return line[1:] if line[:1] in "+-" else ""


def production_names(files: list[dict]) -> set[str]:
    names: set[str] = set()
    for file in files:
        if TEST_PATH.search(file["path"]):
            continue
        for line in file["lines"]:
            header = HUNK_DEF.search(line) if line.startswith("@@") else None
            if header:
                names.add(header.group(1))
            if line.startswith("+") and not line.startswith("+++"):
                match = DEF.match(payload(line))
                if match:
                    names.add(match.group(1))
    return names


def test_blocks(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    current = None
    for line in lines:
        if line.startswith("@@"):
            current = None
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:]
        test = TEST_DEF.match(text)
        helper = HELPER_DEF.match(text)
        if test or helper:
            current = {
                "name": (test or helper).group(1),
                "kind": "test" if test else "helper",
                "body": [],
            }
            blocks.append(current)
            continue
        if current is None:
            current = {
                "name": "(edits inside existing tests)",
                "kind": "fixture fallout",
                "body": [],
            }
            blocks.append(current)
        current["body"].append(text.rstrip())
    fallout = [block for block in blocks if block["kind"] == "fixture fallout"]
    if len(fallout) > 1:
        kept = fallout[0]
        for extra in fallout[1:]:
            kept["body"].extend(extra["body"])
            blocks.remove(extra)
    return blocks


def calls(body: list[str], names: set[str]) -> list[str]:
    found: list[str] = []
    for line in body:
        for name in CALL.findall(line):
            if name in names and name not in SKIP_CALL and name not in found:
                found.append(name)
    return found


def stubs(body: list[str], names: set[str]) -> list[str]:
    found: list[str] = []
    for line in body:
        for name in names:
            if re.search(rf"\b{name}\s*=\s*lambda\b", line) and name not in found:
                found.append(name)
    return found


def copied_expressions(prod_lines: list[str], body: list[str]) -> list[str]:
    test_blob = "\n".join(re.sub(r"\s+", "", line) for line in body)
    hits: list[str] = []
    for line in prod_lines:
        compact = re.sub(r"\s+", "", line)
        if len(compact) < 28:
            continue
        if not re.search(r"(==|//|%|clamp|where|maximum|\+|\*)", compact):
            continue
        if compact in test_blob and compact not in hits:
            hits.append(compact)
    return hits


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
        title = json.loads(sh([
            "gh", "pr", "view", number, "--repo", args.repo, "--json", "title",
        ])).get("title", "")
        diff = sh(["gh", "pr", "diff", number, "--repo", args.repo])

    files = split_files(diff)
    names = production_names(files)
    prod_added = [
        payload(line)
        for file in files
        if not TEST_PATH.search(file["path"])
        for line in file["lines"]
        if line.startswith("+") and not line.startswith("+++")
    ]

    print(f"# test-seam facts — {label}")
    if title:
        print(f"\n{title}")
    print("\nFacts only. Decide the correct seam and whether expected values are independent.\n")

    found_test = False
    for file in files:
        if not TEST_PATH.search(file["path"]):
            continue
        blocks = test_blocks(file["lines"])
        if not blocks:
            continue
        found_test = True
        print(f"## {file['path']}\n")
        for block in blocks:
            called = calls(block["body"], names)
            replaced = stubs(block["body"], names)
            copied = copied_expressions(prod_added, block["body"])
            print(f"- `{block['name']}` ({block['kind']})")
            print(f"  - calls production: {', '.join(f'`{x}`' for x in called) or '—'}")
            if replaced:
                print(f"  - replaces with lambda: {', '.join(f'`{x}`' for x in replaced)}")
            if copied:
                print("  - possible copied expression:")
                for expression in copied:
                    print(f"    - `{expression}`")
        print()

    if not found_test:
        print("No test file in the diff.\n")
    print("## Production names touched\n")
    print(", ".join(f"`{name}`" for name in sorted(names)) or "(none)")


if __name__ == "__main__":
    main()
