#!/usr/bin/env python3
"""Print test-entry facts for /pr-test-seam.

Facts only. The skill classifies the seam and writes the judgment JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

TEST_PATH = re.compile(r"^(test/|python/sglang/test/)")
DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
DEF_ANY = re.compile(r"^([ \t]*)(?:async\s+)?def\s+(\w+)\s*\(")
REENTER = re.compile(r"--worker|torch\.distributed\.run|\bsubprocess\.")
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


def _block_kind(name: str) -> str:
    if name == "main":
        return "worker"
    if name.startswith("test_"):
        return "test"
    if name == "reference":
        return "reference"
    return "helper"


def test_blocks(lines: list[str]) -> list[dict]:
    """Every def starts a block. An indented def is not appended to the previous one."""
    blocks: list[dict] = []
    current = None
    for line in lines:
        if line.startswith("@@"):
            current = None
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:]
        defined = DEF_ANY.match(text)
        if defined:
            current = {
                "name": defined.group(2),
                "kind": _block_kind(defined.group(2)),
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


def _calls_name(body: list[str], name: str) -> bool:
    return any(re.search(rf"\b{name}\s*\(", line) for line in body)


def reentered_blocks(block: dict, by_name: dict[str, dict]) -> list[dict]:
    """A test that spawns this file again runs main. Those calls belong to the test."""
    if block["kind"] != "test":
        return []
    if "main" not in by_name:
        return []
    text = "\n".join(block["body"])
    if not (REENTER.search(text) or _calls_name(block["body"], "main")):
        return []
    found = [by_name["main"]]
    # One level: main() often calls a helper that calls production.
    for helper in by_name.values():
        if helper["kind"] == "helper" and _calls_name(by_name["main"]["body"], helper["name"]):
            found.append(helper)
    return found


def oracles(bodies: list[list[str]]) -> list[str]:
    if any(_calls_name(body, "reference") for body in bodies):
        return ["reference"]
    return []


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
    parser.add_argument("--json", action="store_true", help="print facts as JSON")
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

    records = []
    for file in files:
        if not TEST_PATH.search(file["path"]):
            continue
        blocks = test_blocks(file["lines"])
        if not blocks:
            continue
        by_name = {block["name"]: block for block in blocks}
        for block in blocks:
            entered = reentered_blocks(block, by_name)
            bodies = [block["body"], *[item["body"] for item in entered]]
            called = calls(block["body"], names)
            for body in bodies[1:]:
                for name in calls(body, names):
                    if name not in called:
                        called.append(name)
            replaced = stubs(block["body"], names)
            copied = copied_expressions(prod_added, [line for body in bodies for line in body])
            record = {
                "file": file["path"],
                "name": block["name"],
                "kind": block["kind"],
                "calls": called,
                "replaces": replaced,
                "copied": copied,
            }
            if entered:
                record["reenters"] = [item["name"] for item in entered]
            oracle = oracles(bodies)
            if oracle:
                record["oracle"] = oracle
            record["body"] = block["body"]
            records.append(record)

    if args.json:
        slim = [{key: value for key, value in record.items() if key != "body"} for record in records]
        print(json.dumps({
            "label": label,
            "title": title,
            "production_names": sorted(names),
            "tests": slim,
        }, indent=2))
        return

    print(f"# test-seam facts — {label}")
    if title:
        print(f"\n{title}")
    print(
        "\nFacts only. Decide the correct seam and whether expected values are independent. "
        "Test, worker, and reference bodies are below. Do not fetch the test diff.\n"
    )
    if not records:
        print("No test file in the diff.\n")
    current = None
    for record in records:
        if record["file"] != current:
            current = record["file"]
            print(f"## {current}\n")
        extra = ""
        if record.get("reenters"):
            extra += " reenters " + ", ".join(record["reenters"])
        if record.get("oracle"):
            extra += " oracle " + ", ".join(record["oracle"])
        print(f"- `{record['name']}` ({record['kind']}{extra})")
        called = ", ".join(f"`{name}`" for name in record["calls"]) or "—"
        print(f"  - calls production: {called}")
        if record["replaces"]:
            replaced = ", ".join(f"`{name}`" for name in record["replaces"])
            print(f"  - replaces with lambda: {replaced}")
        if record["copied"]:
            print("  - possible copied expression:")
            for expression in record["copied"]:
                print(f"    - `{expression}`")
        if record["kind"] in ("test", "worker", "reference") and record.get("body"):
            print("  - body:")
            shown = record["body"][:100]
            for line in shown:
                print(f"    {line}")
            extra_lines = len(record["body"]) - len(shown)
            if extra_lines:
                print(f"    … {extra_lines} more lines")
        print()
    print("## Production names touched\n")
    print(", ".join(f"`{name}`" for name in sorted(names)) or "(none)")


if __name__ == "__main__":
    main()
