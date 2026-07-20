#!/usr/bin/env python3
"""abox-stop — stop a live claude agent session by session id.

Kills the claude PROCESS that owns a transcript, resolved the same way abox-live
resolves it. Refuses to act on anything it cannot pin down, because killing the wrong
agent loses in-flight work.

  sudo -n python3 abox_stop.py <session-id-prefix> [--force] [--yes]
"""
import os, sys, time, signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    yes = "--yes" in sys.argv or "-y" in sys.argv
    if not args:
        print("usage: abox-live stop <session-id-prefix> [--force] [--yes]")
        return 2
    sel = args[0]

    rows = [r for r in AL.collect() if r["mine"]]
    hits = [r for r in rows if r["session"].startswith(sel)]
    if not hits:
        print("no live session matching '%s'" % sel)
        print("run `abox-live` to see what is live")
        return 1
    if len(hits) > 1:
        print("'%s' matches %d sessions — be more specific:" % (sel, len(hits)))
        for r in hits:
            print("  %s  %s" % (r["session"], r["container"]))
        return 1

    r = hits[0]
    me = os.getppid()
    print("session   %s" % r["session"])
    print("container %s" % r["container"])
    print("cwd       %s" % r["cwd"])
    print("pid       %s   (evidence: %s)" % (r["pid"], r["evidence"]))
    print("last      %s" % (r["latest"][:150] or "(nothing)"))

    if r["evidence"] in ("ambiguous", "none"):
        # the pid<->session mapping is a guess here; killing on a guess is not ok
        print("\nREFUSING: this row's session id is %s, so the pid may belong to a "
              "different session.\nKill it explicitly by pid instead if you are sure: "
              "sudo kill %s" % (r["evidence"], r["pid"]))
        return 1

    if r["evidence"] == "cwd+mtime":
        print("\nNOTE: this pid<->session link is INFERRED from the working directory "
              "(only\nsession in that container, so it is very likely right — but it "
              "was not confirmed\nfrom the command line).")

    if not yes:
        print("\nabout to send SIGTERM to pid %s — re-run with --yes to confirm"
              % r["pid"])
        return 0

    try:
        os.kill(r["pid"], signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        print("\nalready gone")
        return 0
    except PermissionError as e:
        print("\ncannot signal pid %s: %s" % (r["pid"], e))
        return 1

    for _ in range(20):
        time.sleep(0.25)
        try:
            os.kill(r["pid"], 0)
        except ProcessLookupError:
            print("\nstopped %s (%s)" % (r["session"][:8], r["container"]))
            return 0
    print("\npid %s still alive after SIGTERM — re-run with --force for SIGKILL"
          % r["pid"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
