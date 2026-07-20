#!/usr/bin/env python3
"""Print every local skill and its usage as a table, grouped by category.

Scans the skills root (the parent of this file's dir) for `<name>/SKILL.md`,
parses the YAML-ish frontmatter (name / description / category), and prints a
compact aligned table. Written for the case where the shell has no slash-command
autocomplete — run it to see what you can invoke and when.

Usage:
  list_skills.py                 # grouped table, truncated usage
  list_skills.py -l/--long       # full (untruncated) descriptions
  list_skills.py -f/--flat       # single flat table, no category grouping
  list_skills.py -s/--search STR # only skills whose name/desc/category match STR
  list_skills.py --plain         # no ANSI color (for piping / dumb terminals)
"""
from __future__ import annotations
import argparse
import os
import re
import sys

SKILLS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_frontmatter(path: str) -> dict:
    """Extract the leading --- ... --- block into a dict. Tolerant of quotes and
    of files with no frontmatter at all."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return {}
    m = re.match(r"^﻿?---\s*\n(.*?)\n---\s*(\n|$)", text, re.DOTALL)
    if not m:
        return {}
    block = m.group(1)
    data: dict[str, str] = {}
    key = None
    for line in block.splitlines():
        # top-level "key: value" (not indented — nested keys are ignored)
        mm = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if mm and not line.startswith((" ", "\t")):
            key = mm.group(1).strip()
            val = mm.group(2).strip()
            # YAML block scalar indicator (>, |, >-, |-, >+, |+): real content is the
            # indented lines that follow; the indicator itself is not part of the value.
            if re.fullmatch(r"[>|][+-]?", val):
                val = ""
            data[key] = val
        elif key and line.startswith((" ", "\t")):
            # continuation of a folded/multiline (or block) scalar
            data[key] = (data[key] + " " + line.strip()).strip()
    for k, v in data.items():
        v = v.strip()
        if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
            v = v[1:-1]
        data[k] = v.strip()
    return data


def first_sentence(desc: str, limit: int) -> str:
    desc = re.sub(r"\s+", " ", desc).strip()
    if not desc:
        return "(no description)"
    # prefer the first sentence, but don't exceed the limit
    cut = desc
    m = re.search(r"(?<=[.!?])\s", desc)
    if m and m.start() + 1 <= limit:
        cut = desc[: m.start() + 1]
    if len(cut) > limit:
        cut = cut[: limit - 1].rstrip() + "…"
    return cut


def discover() -> list[dict]:
    skills = []
    for entry in sorted(os.listdir(SKILLS_ROOT)):
        d = os.path.join(SKILLS_ROOT, entry)
        sk = os.path.join(d, "SKILL.md")
        if not os.path.isdir(d) or entry.startswith((".", "_")):
            continue
        if not os.path.isfile(sk):
            continue
        fm = parse_frontmatter(sk)
        skills.append(
            {
                "name": fm.get("name") or entry,
                "dir": entry,
                "category": fm.get("category") or "uncategorized",
                "description": fm.get("description", ""),
            }
        )
    return skills


CATEGORY_ORDER = [
    "kernel-opt", "measure", "research", "deliver",
    "infra", "meta", "orchestration", "uncategorized",
]

HISTORY_CANDIDATES = [
    os.path.join(os.environ.get("CLAUDE_CONFIG_DIR", ""), "history.jsonl"),
    os.path.expanduser("~/.claude/history.jsonl"),
]


def usage_counts(known: set[str]) -> dict[str, int]:
    """Count how often each KNOWN skill name was invoked, from history.jsonl.
    Only leading `/name` entries whose name is an actual local skill count — this
    drops path-like `/home/...` and built-ins (`/model`, `/resume`, `/fast`, …)."""
    import json
    counts = {k: 0 for k in known}
    seen_path = None
    for path in HISTORY_CANDIDATES:
        if path and os.path.isfile(path):
            seen_path = path
            break
    if not seen_path:
        return counts
    with open(seen_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                disp = json.loads(line).get("display", "")
            except (ValueError, AttributeError):
                continue
            m = re.match(r"\s*/([A-Za-z0-9][A-Za-z0-9-]+)", disp or "")
            if m and m.group(1) in counts:
                counts[m.group(1)] += 1
    return counts


def color(s: str, code: str, on: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if on else s


def render(skills, args) -> str:
    use_color = (not args.plain) and sys.stdout.isatty()
    name_w = max((len(s["name"]) for s in skills), default=4)
    name_w = min(max(name_w, 4), 26)
    try:
        term_w = os.get_terminal_size().columns
    except OSError:
        term_w = 100
    desc_w = 10_000 if args.long else max(30, term_w - name_w - 6)

    out: list[str] = []

    def row(name, desc, bullet="  "):
        usage = desc if args.long else first_sentence(desc, desc_w)
        nm = color(f"/{name}".ljust(name_w + 1), "1;36", use_color)
        if args.long:
            out.append(f"{bullet}{nm}")
            for i, chunk in enumerate(wrap(usage, term_w - 6)):
                out.append(f"      {chunk}")
        else:
            out.append(f"{bullet}{nm}  {usage}")

    total = len(skills)
    header = color(f"Local skills ({total}) — invoke with /<name>", "1", use_color)
    out.append(header)
    out.append("")

    if args.by_usage:
        counts = usage_counts({s["name"] for s in skills})
        ranked = sorted(skills, key=lambda x: (-counts[x["name"]], x["name"]))
        cnt_w = max((len(str(counts[s["name"]])) for s in skills), default=1)
        out.append(color(
            "  ranked by invocations in history.jsonl (× = never invoked / not recorded)",
            "2", use_color))
        out.append("")
        for s in ranked:
            n = counts[s["name"]]
            badge = ("×" if n == 0 else str(n)).rjust(cnt_w)
            badge = color(badge, "2" if n == 0 else "1;32", use_color)
            nm = color(f"/{s['name']}".ljust(name_w + 1), "1;36", use_color)
            usage = s["description"] if args.long else first_sentence(
                s["description"], max(30, desc_w - cnt_w - 2))
            out.append(f"  {badge}  {nm}  {usage}")
        return "\n".join(out)

    if args.flat:
        for s in sorted(skills, key=lambda x: x["name"]):
            row(s["name"], s["description"])
        return "\n".join(out)

    by_cat: dict[str, list[dict]] = {}
    for s in skills:
        by_cat.setdefault(s["category"], []).append(s)
    cats = [c for c in CATEGORY_ORDER if c in by_cat] + sorted(
        c for c in by_cat if c not in CATEGORY_ORDER
    )
    for c in cats:
        label = color(f"[{c}]", "1;33", use_color)
        out.append(f"{label}")
        for s in sorted(by_cat[c], key=lambda x: x["name"]):
            row(s["name"], s["description"])
        out.append("")
    return "\n".join(out).rstrip()


def wrap(text: str, width: int):
    words, line, lines = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines or [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--long", action="store_true", help="full descriptions")
    ap.add_argument("-f", "--flat", action="store_true", help="no category grouping")
    ap.add_argument("-u", "--by-usage", action="store_true",
                    help="rank by invocation count (from history.jsonl)")
    ap.add_argument("-s", "--search", metavar="STR", help="filter by substring")
    ap.add_argument("--plain", action="store_true", help="disable color")
    args = ap.parse_args()

    skills = discover()
    if args.search:
        q = args.search.lower()
        skills = [
            s for s in skills
            if q in s["name"].lower()
            or q in s["description"].lower()
            or q in s["category"].lower()
        ]
        if not skills:
            print(f"No skill matches {args.search!r}.")
            return
    print(render(skills, args))


if __name__ == "__main__":
    main()
