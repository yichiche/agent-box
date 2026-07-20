#!/usr/bin/env python3
"""abox report engine — runs as root (single sudo) so it can read every
container-written session JSONL (root:root 0600) without per-file sudo.

Commands:
  ps [lookback_min]      table of recent sessions, newest activity first
  tail <id> [n]          last n user/assistant turns of one session
  path <id>              print the JSONL path
"""
import os, sys, glob, json, time, re

# Overridable by a trailing argv (see main); default to the known host path because
# this runs as root under sudo, where ~ would wrongly expand to /root.
PROJ = "/home/yichiche/.claude/projects"


def scan_roots():
    """[(projects_dir, container_label)] — bind-mounted ~/.claude/projects."""
    return [(PROJ, "shared")]


def iter_sessions():
    for base, container in scan_roots():
        for f in glob.glob(os.path.join(base, "*", "*.jsonl")):
            yield f, container


def short_container(c):
    if c == "shared":
        return "shared?"
    return re.sub(r"^jacky-v[0-9.]+-rocm[0-9]+-mi[0-9a-z]+-", "", c)[:20] or c[:20]


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


def load_turns(path):
    turns, cwd = [], ""
    try:
        with open(path, errors="ignore") as fh:
            for ln in fh:
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if not cwd and o.get("cwd"):
                    cwd = o["cwd"]
                if o.get("type") in ("user", "assistant"):
                    t = turn_text(o)
                    if t:
                        turns.append((o["type"], t))
    except Exception:
        pass
    return turns, cwd


def latest_line(turns):
    for role, txt in reversed(turns):
        if txt and txt != "[tool_result]":
            return ("» " if role == "assistant" else "? ") + txt
    return turns[-1][1] if turns else ""


def hage(s):
    if s < 90:
        return "%ds" % s
    if s < 5400:
        return "%dm" % round(s / 60)
    if s < 86400:
        return "%dh" % round(s / 3600)
    return "%dd" % round(s / 86400)


def hsize(b):
    n, u = float(b), ["B", "K", "M", "G"]
    i = 0
    while n >= 1024 and i < 3:
        n /= 1024.0
        i += 1
    return ("%d%s" if i == 0 else "%.1f%s") % (n, u[i])


def find_one(sel):
    for base, _ in scan_roots():
        for pat in ("%s*.jsonl" % sel, "*%s*.jsonl" % sel):
            hits = glob.glob(os.path.join(base, "*", pat))
            if hits:
                return hits[0]
    return None


def cmd_ps(lookback_min=360):
    now = time.time()
    cutoff = now - lookback_min * 60
    rows = []
    for f, container in iter_sessions():
        try:
            st = os.stat(f)
        except OSError:
            continue
        if st.st_mtime < cutoff:
            continue
        rows.append((st.st_mtime, f, st.st_size, container))
    rows.sort(reverse=True)
    fmt = "%-2s %-20s %-22s %-8s %5s %7s  %s"
    print(fmt % ("", "CONTAINER", "PROJECT (cwd)", "SESSION", "AGE", "SIZE", "LATEST"))
    print(fmt % ("", "-" * 20, "-" * 22, "-" * 8, "-" * 5, "-" * 7, "-" * 26))
    for mt, f, sz, container in rows:
        age = int(now - mt)
        state = "\U0001F7E2" if age < 120 else ("⚪" if age < 1800 else "✓")
        proj = os.path.basename(os.path.dirname(f))[:22]
        sid = os.path.basename(f)[:-6][:8]
        turns, _ = load_turns(f)
        lat = latest_line(turns)[:52]
        print(fmt % (state, short_container(container), proj, sid, hage(age), hsize(sz), lat))
    print()
    print("\U0001F7E2 active(<2m)  ⚪ idle  ✓ older   ·  'shared?' = legacy shared dir (pre-isolation, container unknown)")


def cmd_tail(sel, n=8):
    f = find_one(sel)
    if not f:
        print("no session matching '%s'" % sel)
        return 1
    print("# %s   %s" % (os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-6]))
    turns, _ = load_turns(f)
    for role, txt in turns[-n:]:
        print("[%-9s] %s" % (role, txt[:500]))


def cmd_path(sel):
    f = find_one(sel)
    if not f:
        print("no match")
        return 1
    print(f)


def main():
    global PROJ
    args = list(sys.argv[1:])
    # a trailing arg that is an existing .../projects dir overrides PROJ
    if args and args[-1].rstrip("/").endswith("projects") and os.path.isdir(args[-1]):
        PROJ = args.pop()
    args = [a for a in args if a != ""]  # tolerate empty placeholder args from the shell
    cmd = args[0] if args else "ps"
    rest = args[1:]
    if cmd == "ps":
        cmd_ps(int(rest[0]) if rest else 360)
    elif cmd in ("tail", "show"):
        sel = rest[0] if rest else ""
        n = int(rest[1]) if len(rest) > 1 else 8
        sys.exit(cmd_tail(sel, n) or 0)
    elif cmd == "path":
        sys.exit(cmd_path(rest[0] if rest else "") or 0)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
