#!/usr/bin/env python3
"""abox-inbox — queue instructions for container agents (Remote Control flow).

Remote Control Claude on the host cannot push buttons on the published Artifact, but
it CAN run shell commands. Two paths:

  abox-live say <id> "msg"       — immediate (refuses if agent is live unless --force)
  abox-live instruct <id> "msg"  — queue; then abox-live drain --yes executes all pending

The published dashboard shows pending items and copy-paste commands.
"""
import os, sys, json, time, glob, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL

INBOX_DIR = "/home/yichiche/agent-scratch/abox-dash"
INBOX_PATH = os.path.join(INBOX_DIR, "inbox.jsonl")
OWNER = "yichiche"


def _ensure():
    os.makedirs(INBOX_DIR, exist_ok=True)
    if not os.path.isfile(INBOX_PATH):
        open(INBOX_PATH, "a").close()
        try:
            import pwd
            pw = pwd.getpwnam(OWNER)
            os.chown(INBOX_PATH, pw.pw_uid, pw.pw_gid)
        except Exception:
            pass


def _read_all():
    _ensure()
    rows = []
    try:
        with open(INBOX_PATH, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    except OSError:
        pass
    return rows


def _write_all(rows):
    _ensure()
    tmp = INBOX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, INBOX_PATH)
    try:
        import pwd
        pw = pwd.getpwnam(OWNER)
        os.chown(INBOX_PATH, pw.pw_uid, pw.pw_gid)
    except Exception:
        pass


def resolve_session(prefix):
    hits = glob.glob(os.path.join(AL.SHARED_PROJ, "*", prefix + "*.jsonl"))
    hits = sorted(set(hits))
    if not hits:
        return None, None
    if len(hits) > 1:
        return None, hits
    sid = os.path.basename(hits[0])[:-6]
    return sid, hits[0]


def instruct(session_prefix, message):
    sid, path_or_hits = resolve_session(session_prefix)
    if sid is None and path_or_hits:
        print("'%s' matches %d sessions — be more specific:" % (session_prefix, len(path_or_hits)))
        for h in path_or_hits:
            print("  %s" % os.path.basename(h)[:-6][:8])
        return 1
    if not sid:
        print("no session matching '%s'" % session_prefix)
        return 1
    live = [r for r in AL.collect() if r["session"] == sid]
    entry = {
        "id": str(uuid.uuid4())[:8],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session": sid,
        "session_short": sid[:8],
        "message": message,
        "container": live[0]["container"] if live else "",
        "status": "pending",
        "result": "",
    }
    rows = _read_all()
    rows.append(entry)
    _write_all(rows)
    print("queued  %s → %s" % (entry["id"], sid[:8]))
    print("message %s" % message[:120])
    print("\nexecute:  abox-live drain --yes")
    return 0


def list_inbox(pending_only=False, limit=20):
    rows = _read_all()
    if pending_only:
        rows = [r for r in rows if r.get("status") == "pending"]
    return rows[-limit:]


def drain(yes=False, force=False, dangerous=False):
    rows = _read_all()
    pending = [r for r in rows if r.get("status") == "pending"]
    if not pending:
        print("inbox empty — nothing to drain")
        return 0
    if not yes:
        print("%d pending instruction(s):" % len(pending))
        for r in pending:
            print("  [%s] %s → %s" % (r.get("id", "?"), r.get("session_short", "?"),
                                      (r.get("message") or "")[:80]))
        print("\nrun with --yes to execute")
        return 0

    import abox_say as SAY
    rc = 0
    for r in pending:
        sid = r.get("session", "")
        msg = r.get("message", "")
        print("\n--- drain %s (%s) ---" % (r.get("id", "?"), sid[:8]))
        argv = ["abox_say.py", sid[:8], msg]
        if force:
            argv.append("--force")
        if dangerous:
            argv.append("--dangerous")
        if r.get("container") and r["container"] != "host":
            argv.extend(["--container", r["container"]])
        old_argv = sys.argv
        sys.argv = argv
        try:
            code = SAY.main()
        finally:
            sys.argv = old_argv
        r["status"] = "done" if code == 0 else "failed"
        r["drained_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if code != 0:
            rc = code
    _write_all(rows)
    done = sum(1 for r in rows if r.get("status") == "done" and r.get("drained_at"))
    print("\ndrained %d item(s)" % len(pending))
    return rc


def render_inbox_html(limit=10):
    """Pending + recent inbox for the published dashboard."""
    import abox_transcript as AT
    rows = _read_all()
    pending = [r for r in rows if r.get("status") == "pending"]
    recent = [r for r in rows if r.get("status") != "pending"][-5:]
    if not pending and not recent:
        return ""
    parts = ['<div class="inbox">', '<div class="inbox-h">&#128236; Instruction inbox</div>']
    if pending:
        parts.append('<div class="inbox-sub">%d pending — tell RC: '
                     '<code>abox-live drain --yes</code></div>' % len(pending))
        for r in pending[-limit:]:
            parts.append('<div class="inbox-item pending"><span class="mono">%s</span>'
                         ' &rarr; <b>%s</b> · %s<div class="inbox-msg">%s</div></div>'
                         % (AT.esc(r.get("id", "?")), AT.esc(r.get("session_short", "?")),
                            AT.esc(r.get("container") or "host"),
                            AT.esc(r.get("message", ""))))
    if recent:
        parts.append('<div class="inbox-sub">recent</div>')
        for r in recent:
            st = r.get("status", "?")
            parts.append('<div class="inbox-item %s"><span class="mono">%s</span>'
                         ' %s · %s</div>'
                         % (st, AT.esc(r.get("session_short", "?")), st,
                            AT.esc((r.get("message") or "")[:60])))
    parts.append("</div>")
    return "".join(parts)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("list", "inbox"):
        rows = list_inbox(pending_only="--pending" in argv)
        if not rows:
            print("(empty)")
            return 0
        for r in rows:
            print("[%s] %s %s → %s" % (
                r.get("status", "?"), r.get("id", "?"), r.get("session_short", "?"),
                (r.get("message") or "")[:100]))
        return 0
    if argv[0] == "instruct":
        pos = [a for a in argv[1:] if not a.startswith("--")]
        if len(pos) < 2:
            print('usage: abox-live instruct <session-id-prefix> "message"')
            return 2
        return instruct(pos[0], pos[1])
    if argv[0] == "drain":
        return drain(yes="--yes" in argv, force="--force" in argv,
                     dangerous="--dangerous" in argv)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
