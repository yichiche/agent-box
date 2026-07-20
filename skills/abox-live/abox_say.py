#!/usr/bin/env python3
"""abox-say — send a follow-up instruction to a specific agent session.

Continues an existing conversation with `claude --resume <session> -p "<message>"`,
run in the right place: inside the owning container for container sessions, on the
host for host sessions. The reply is printed.

  sudo -n python3 abox_say.py <session-id-prefix> "message" [--container C]
                              [--force] [--timeout S] [--dangerous]
"""
import os, re, sys, glob, json, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL

HOST_USER = "yichiche"


def find_transcript(sel):
    """(full_session_id, path) for a prefix under ~/.claude/projects."""
    hits = glob.glob(os.path.join(AL.SHARED_PROJ, "*", sel + "*.jsonl"))
    hits = sorted(set(hits))
    if not hits:
        return None, None
    if len(hits) > 1:
        print("'%s' matches %d transcripts:" % (sel, len(hits)))
        for h in hits:
            print("  %s" % os.path.basename(h)[:-6])
        return None, None
    return os.path.basename(hits[0])[:-6], hits[0]


def cwd_of(path):
    """the cwd recorded in the transcript — authoritative, unlike the dir-name slug"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if o.get("cwd"):
                    return o["cwd"]
    except Exception:
        pass
    return ""


def owned(container):
    """same rule as bridge.sh: ours iff it bind-mounts our home"""
    out = AL.sh("docker inspect %s --format '{{range .Mounts}}{{.Source}}{{\"\\n\"}}{{end}}'"
                % container)
    return any(l.strip() == "/home/" + HOST_USER for l in out.splitlines())


def main():
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) < 2:
        print('usage: abox-live say <session-id-prefix> "message" '
              "[--container C] [--force] [--timeout S] [--dangerous]")
        return 2
    sel, message = pos[0], pos[1]
    force = "--force" in argv
    dangerous = "--dangerous" in argv
    timeout = 600
    if "--timeout" in argv:
        timeout = int(argv[argv.index("--timeout") + 1])
    container = None
    if "--container" in argv:
        container = argv[argv.index("--container") + 1]

    sid, path = find_transcript(sel)
    if not sid:
        print("no transcript matching '%s'" % sel)
        return 1

    live = [r for r in AL.collect() if r["session"] == sid]
    cwd = (live[0]["cwd"] if live else "") or cwd_of(path)
    if not container:
        container = live[0]["container"] if live else None

    print("session   %s" % sid)
    print("cwd       %s" % (cwd or "(unknown)"))
    print("target    %s" % (container or "host"))

    if live and not force:
        # Two processes appending to one transcript interleave and corrupt the
        # conversation; make the user choose explicitly.
        print("\nREFUSING: a live process (pid %s) is still running this session, so "
              "resuming\nit now would write into the same transcript from two places."
              "\n\nEither stop it first:   abox-live stop %s --yes"
              "\nor override:            abox-live say %s \"…\" --force"
              % (live[0]["pid"], sid[:8], sid[:8]))
        return 1

    if not cwd:
        print("\ncannot resume: no cwd recorded for this session")
        return 1

    flags = "--dangerously-skip-permissions " if dangerous else ""
    inner = ('export PATH="$HOME/.local/bin:$PATH"; cd %s 2>/dev/null || cd /; '
             'claude --resume %s -p %s"$PROMPT"' % (quote(cwd), sid, flags))

    if container and container != "host":
        if not owned(container):
            print("\nREFUSED: '%s' does not mount /home/%s — not yours. Shared host."
                  % (container, HOST_USER))
            return 1
        cmd = ["docker", "exec", "-e", "PROMPT=" + message, container, "bash", "-lc", inner]
    else:
        # Host-side transcripts are frequently root:root (written by a container that
        # bind-mounts ~/.claude), so claude must run as root to read them; HOME points
        # at the real projects dir so --resume finds the session and its credentials.
        cmd = ["env", "HOME=/home/" + HOST_USER, "PROMPT=" + message,
               "bash", "-lc", inner]

    print("\n→ resuming (timeout %ds)…\n" % timeout)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("timed out after %ds — the agent may still be working; "
              "check with `abox-live tail %s`" % (timeout, sid[:8]))
        return 1
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip() or "(no output)")
    return r.returncode


def quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    sys.exit(main())
