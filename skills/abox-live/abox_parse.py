#!/usr/bin/env python3
"""Parse a Claude Code session JSONL (from stdin) for the abox monitor.

Modes:
  latest        one-line summary of the most recent meaningful turn
  lastn <n>     last n user/assistant turns (truncated)
  meta          cwd + git branch + turn count (first-seen fields)
"""
import sys, json

mode = sys.argv[1] if len(sys.argv) > 1 else "latest"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def turn_text(o):
    m = o.get("message", {})
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        return " ".join(c.split())
    parts = []
    if isinstance(c, list):
        for p in c:
            if not isinstance(p, dict):
                continue
            pt = p.get("type")
            if pt == "text":
                parts.append(p.get("text", ""))
            elif pt == "tool_use":
                parts.append("[tool:" + p.get("name", "") + "]")
            elif pt == "tool_result":
                parts.append("[tool_result]")
    return " ".join(" ".join(parts).split())


turns = []
cwd = ""
for ln in sys.stdin:
    try:
        o = json.loads(ln)
    except Exception:
        continue
    if not cwd and o.get("cwd"):
        cwd = o["cwd"]
    if o.get("type") in ("user", "assistant"):
        txt = turn_text(o)
        if txt:
            turns.append((o["type"], txt))

if mode == "meta":
    print("cwd=%s turns=%d" % (cwd or "?", len(turns)))
elif mode == "latest":
    # prefer the last turn that carries real text, not just a tool_result echo
    pick = ""
    for role, txt in reversed(turns):
        if txt and txt != "[tool_result]":
            pick = "%s %s" % ("»" if role == "assistant" else "?", txt)
            break
    if not pick and turns:
        pick = turns[-1][1]
    print(pick)
else:  # lastn
    for role, txt in turns[-n:]:
        print("[%-9s] %s" % (role, txt[:500]))
