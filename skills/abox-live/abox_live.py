#!/usr/bin/env python3
"""abox-live — list LIVE claude agent sessions and the container each runs in.

Unlike `abox-live ps` (which lists transcript FILES on disk and cannot attribute the
legacy shared projects dir to a container), this starts from live PROCESSES:

    claude PID --> cgroup --> docker id --> container name
    claude PID --> cwd     --> project slug --> session .jsonl

Session id is resolved by strongest available evidence:
    exact:  --resume=<uuid> on the claude cmdline
    exact:  a descendant tool-call process carrying <uuid>.jsonl / SESSION_ID
    infer:  newest-mtime transcript in the project dir matching the process cwd

Run as root (needs /proc/<pid>/cgroup + root-owned transcripts).
"""
import os, re, sys, glob, json, time, subprocess

SHARED_PROJ = "/home/yichiche/.claude/projects"
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout
    except Exception:
        return ""


def my_home():
    u = os.environ.get("SUDO_USER") or os.environ.get("USER") or ""
    h = "/home/%s" % u if u and u != "root" else ""
    return h if h and os.path.isdir(h) else os.path.dirname(SHARED_PROJ.rstrip("/")
                                                            .rsplit("/.claude", 1)[0])


def docker_names():
    """full/short container id -> name"""
    m = {}
    for ln in sh("docker ps --no-trunc --format '{{.ID}}\t{{.Names}}'").splitlines():
        if "\t" in ln:
            cid, name = ln.split("\t", 1)
            m[cid.strip()] = name.strip()
    return m


def mine_map(names):
    """container name -> True when it bind-mounts MY home (so it's my agent, not a
    co-worker's). Name prefixes are unreliable here; the mount is definitive."""
    home = my_home()
    out = {}
    if not names:
        return out
    q = " ".join("'%s'" % n for n in names.values())
    raw = sh("docker inspect -f '{{.Name}}|{{range .Mounts}}{{.Source}},{{end}}' " + q)
    for ln in raw.splitlines():
        if "|" not in ln:
            continue
        nm, srcs = ln.split("|", 1)
        nm = nm.strip().lstrip("/")
        out[nm] = any(s == home or s.startswith(home + "/")
                      for s in srcs.split(",") if s)
    return out


def procs():
    """[{pid, ppid, args}] for every process (root view)."""
    out = []
    raw = sh("ps -eo pid=,ppid=,args=")
    for ln in raw.splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            out.append({"pid": int(parts[0]), "ppid": int(parts[1]), "args": parts[2]})
        except ValueError:
            continue
    return out


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""


def container_of(pid, names):
    cg = read("/proc/%d/cgroup" % pid).decode("utf-8", "replace")
    m = re.search(r"docker[-/]([0-9a-f]{12,64})", cg)
    if not m:
        return "host", None
    cid = m.group(1)
    for full, name in names.items():
        if full.startswith(cid) or cid.startswith(full):
            return name, cid[:12]
    return "?" + cid[:12], cid[:12]


def cwd_of(pid):
    try:
        return os.readlink("/proc/%d/cwd" % pid)
    except Exception:
        return ""


def slug(path):
    return re.sub(r"[^a-zA-Z0-9]", "-", path)


def roots_for(container):
    """project dirs to search (bind-mounted ~/.claude/projects)."""
    return [SHARED_PROJ]


def sessions_in(root, sl):
    d = os.path.join(root, sl)
    if not os.path.isdir(d):
        return []
    out = []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        try:
            out.append((os.path.getmtime(f), f))
        except OSError:
            pass
    return sorted(out, reverse=True)


def find_by_id(sid, container):
    for root in roots_for(container):
        hits = glob.glob(os.path.join(root, "*", sid + ".jsonl"))
        if hits:
            return hits[0]
    return None


def descendants(pid, kids):
    seen, stack, out = set(), [pid], []
    while stack:
        p = stack.pop()
        for c in kids.get(p, []):
            if c not in seen:
                seen.add(c)
                out.append(c)
                stack.append(c)
    return out


def last_turn(path, maxbytes=400000):
    """(mtime, short text of last assistant/user turn)"""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > maxbytes:
                f.seek(size - maxbytes)
                f.readline()
            lines = f.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return 0, ""
    for ln in reversed(lines):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("type") not in ("assistant", "user"):
            continue
        msg = o.get("message", {})
        c = msg.get("content") if isinstance(msg, dict) else None
        parts = []
        if isinstance(c, str):
            parts = [c]
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "tool_use":
                    parts.append("[tool:%s]" % p.get("name", ""))
                elif p.get("type") == "tool_result":
                    parts.append("[tool_result]")
        t = " ".join(" ".join(parts).split())
        if t:
            return os.path.getmtime(path), t
    return os.path.getmtime(path), ""


def age(sec):
    if sec < 0:
        return "-"
    if sec < 90:
        return "%ds" % int(sec)
    if sec < 5400:
        return "%dm" % int(sec // 60)
    return "%dh" % int(sec // 3600)


def collect():
    names = docker_names()
    all_procs = procs()
    kids = {}
    for p in all_procs:
        kids.setdefault(p["ppid"], []).append(p["pid"])
    by_pid = {p["pid"]: p for p in all_procs}

    claudes = [p for p in all_procs
               if re.search(r"(^|/)claude(\s|$)", p["args"].split(" -")[0])
               or re.match(r"^claude(\s|$)", p["args"])]

    # Pass 1 — exact ids only (cmdline / descendant tool-call process).
    staged = []
    for p in claudes:
        pid, args = p["pid"], p["args"]
        container, cid = container_of(pid, names)
        sid, ev = None, ""

        m = re.search(r"--resume[= ](" + UUID + ")", args)
        if m:
            sid, ev = m.group(1), "resume-flag"

        if not sid:
            for c in descendants(pid, kids):
                a = by_pid.get(c, {}).get("args", "")
                m = re.search(r"(" + UUID + r")\.jsonl", a) or \
                    re.search(r"SESSION_ID=.?(" + UUID + ")", a)
                if m:
                    sid, ev = m.group(1), "child-proc"
                    break

        path = find_by_id(sid, container) if sid else None
        staged.append({"pid": pid, "container": container, "cwd": cwd_of(pid),
                       "session": sid, "evidence": ev, "path": path})

    # Pass 2 — infer the rest from cwd, never re-claiming a transcript that some
    # process already owns by exact evidence.
    claimed = {s["path"] for s in staged if s["path"]}
    need = [s for s in staged if not s["path"] and s["cwd"]]
    # When N live procs share one cwd, the newest-first assignment gets the SET of
    # sessions right but may permute which pid owns which — flag the whole group.
    share = {}
    for s in need:
        share[(s["container"], s["cwd"])] = share.get((s["container"], s["cwd"]), 0) + 1
    for s in need:
        cands = []
        for root in roots_for(s["container"]):
            cands += sessions_in(root, slug(s["cwd"]))
        cands.sort(reverse=True)
        for _, f in cands:
            if f in claimed:
                continue
            s["path"], s["session"] = f, os.path.basename(f)[:-6]
            s["evidence"] = "ambiguous" if share[(s["container"], s["cwd"])] > 1 else "cwd+mtime"
            claimed.add(f)
            break

    mine = mine_map(names)
    rows = []
    for s in staged:
        mt, latest = (last_turn(s["path"]) if s["path"] else (0, ""))
        rows.append({
            "pid": s["pid"], "container": s["container"], "cwd": s["cwd"],
            "session": s["session"] or "?", "evidence": s["evidence"] or "none",
            "path": s["path"] or "", "mtime": mt, "latest": latest,
            "idle": (time.time() - mt) if mt else -1,
            # host sessions run as me; containers must mount my home to count as mine
            "mine": True if s["container"] == "host" else mine.get(s["container"], False),
        })
    rows.sort(key=lambda r: (r["idle"] if r["idle"] >= 0 else 1e9))
    return rows


def shorten(name, n):
    """drop the noisy image-version prefix container names share"""
    s = re.sub(r"^jacky-v[0-9.]+-rocm[0-9]+-mi[0-9a-z]+-", "", name)
    return s[:n]


def ltrunc(s, n):
    return s if len(s) <= n else "…" + s[-(n - 1):]


EV_SHORT = {"resume-flag": "exact", "child-proc": "exact",
            "cwd+mtime": "infer", "ambiguous": "ambig", "none": "-"}


def table(rows, verbose, with_pid):
    """Default view: a lookup table, sorted by container so a given container's
    sessions sit together and the whole thing stays greppable."""
    rows = sorted(rows, key=lambda r: (r["container"] == "host", r["container"],
                                       r["idle"] if r["idle"] >= 0 else 1e9))
    cw = max([len(r["container"]) for r in rows] + [9])
    cw = min(cw, 46)
    head = "%-2s " % ""
    fmt = "%-2s "
    if with_pid:
        head += "%7s  " % "PID"
        fmt += "%7s  "
    head += "%-*s  %-36s %5s  %-5s  %s" % (cw, "CONTAINER", "SESSION", "AGE", "EV", "CWD")
    fmt += "%-*s  %-36s %5s  %-5s  %s"
    print(head)
    print("-" * (len(head) + 6))
    for r in rows:
        state = "🟢" if 0 <= r["idle"] < 120 else ("⚪" if r["idle"] >= 0 else "❔")
        # Every row repeats the container name: this table is meant to be grepped
        # (`abox-live | grep <container>`), and blanks would drop the later rows.
        args = [state] + ([r["pid"]] if with_pid else []) + [
            cw, r["container"][:cw], r["session"][:36], age(r["idle"]),
            EV_SHORT.get(r["evidence"], r["evidence"]), ltrunc(r["cwd"] or "-", 30)]
        print(fmt % tuple(args))
        if verbose and r["latest"]:
            print("%s└─ %s" % (" " * (12 if with_pid else 3), r["latest"][:88]))


def by_container(rows, verbose):
    """Default view: full container name, then the sessions running inside it."""
    groups = {}
    for r in rows:
        groups.setdefault(r["container"], []).append(r)
    order = sorted(groups, key=lambda c: (c == "host", c.startswith("?"), c))
    for c in order:
        rs = groups[c]
        n = len(rs)
        print("\n%s   (%d session%s)" % (c, n, "" if n == 1 else "s"))
        for r in rs:
            state = "🟢" if 0 <= r["idle"] < 120 else ("⚪" if r["idle"] >= 0 else "❔")
            sid = r["session"] if r["session"] != "?" else "?  (transcript not readable from host)"
            print("  %s %-36s %5s  %-11s %s" % (
                state, sid, age(r["idle"]), r["evidence"], ltrunc(r["cwd"] or "-", 30)))
            if verbose and r["latest"]:
                print("       └─ %s" % r["latest"][:92])


def main():
    argv = sys.argv[1:]
    as_json = "--json" in argv
    verbose = "-v" in argv or "--verbose" in argv
    group = "--group" in argv
    with_pid = "--pid" in argv or "--flat" in argv
    show_all = "--all" in argv
    # First non-flag positional = container filter (name or substring), e.g.
    # `abox-live yct-aiter-test-0720` -> only that container's live sessions.
    container_filter = next((a for a in argv if not a.startswith("-")), None)
    rows = collect()
    if container_filter:
        rows = [r for r in rows if container_filter in r["container"]]
        # An explicitly named container is shown regardless of ownership —
        # the user asked for it by name, so don't hide it as "not mine".
        show_all = True
    hidden = 0
    if not show_all:
        hidden = sum(1 for r in rows if not r["mine"])
        rows = [r for r in rows if r["mine"]]
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        if container_filter:
            print("no live claude sessions in container matching '%s'" % container_filter)
            print("(the container may be up but idle — a headless `claude -p` exits when "
                  "it finishes, leaving no live session; check `docker ps` / `abox-live ps`)")
        else:
            print("no live claude sessions found")
        return

    w = 116
    if group:
        by_container(rows, verbose)
    else:
        table(rows, verbose, with_pid)
    print("-" * w)
    live = sum(1 for r in rows if 0 <= r["idle"] < 120)
    print("%d live claude session%s%s, %d actively writing" % (
        len(rows), "" if len(rows) == 1 else "s",
        "" if show_all else " of mine", live))
    print("🟢 wrote <2m ago (working)  ⚪ idle (may be waiting on you)   "
          "EV: exact = id confirmed · infer = newest transcript for that cwd · "
          "ambig = procs share a cwd, row may be permuted")
    print("next: abox-live tail <session>   (8-char prefix is enough)")
    if hidden:
        print("(%d session%s in other people's containers hidden — `abox-live --all`)"
              % (hidden, "" if hidden == 1 else "s"))


if __name__ == "__main__":
    main()
