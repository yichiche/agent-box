"""GitHub sign-in for the dashboard — a button on `gh auth login`, nothing more.

Deliberately not an auth system. The whole flow is driven by the `gh` CLI, so:

- there is **no OAuth App to register** — `gh auth login --web` is already a
  device flow against GitHub's own `gh` client, which is what someone cloning
  this skill would have run by hand anyway;
- **no token is stored, read or logged by this code**. `gh` writes it to the
  system credential store (or `hosts.yml`), which is where every `gh` call in
  `watch.py` already looks, so nothing downstream changes;
- signing out is `gh auth logout`, not deleting a file we own.

The one piece of real work is that `gh auth login --web` is interactive: it
prints a one-time code and waits for Enter before opening a browser. A web
handler has no terminal, so it is run on a pty, the code is scraped from the
output and shown on the page, and Enter is fed on our side with `BROWSER=true`
so the headless host does not try to launch one.
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import subprocess
import threading
import time

HOST = "github.com"
# What this dashboard actually needs: `repo` to re-run workflows and call
# update-branch, `read:org` so `gh` can resolve org membership on private
# forks. `gist` is in gh's own default set and asking for less than the token
# you already have triggers a re-auth prompt on every call.
#
# `workflow` is needed by Update branch even though this dashboard never edits
# a workflow file. `PUT /pulls/{n}/update-branch` merges main into the branch,
# and the moment main's side of that merge touches `.github/workflows/*`
# GitHub refuses the whole call with 403 "refusing to allow an OAuth App to
# create or update workflow ... without `workflow` scope" — it gates on what
# the resulting commit contains, not on who wrote the change.
SCOPES = "repo,read:org,gist,workflow"

CODE_RE = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{4})\b")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DEVICE_URL = "https://github.com/login/device"

# One login at a time — it is one person at one keyboard on their own machine.
LOGIN: dict = {"state": "idle", "code": "", "started": 0.0, "error": "",
               "pid": None}
LOGIN_LOCK = threading.Lock()

# GitHub expires a device code after 15 minutes; give up a little before that
# so a stale "waiting" panel cannot outlive the code it is showing.
LOGIN_TIMEOUT = 13 * 60


def _gh(args: list[str], **kw) -> subprocess.CompletedProcess:
    # GH_TOKEN is cleared for the same reason every other call here clears it:
    # a PAT in the environment silently wins over the OAuth token and this repo
    # rejects it. See _shared/repo-config.md.
    env = dict(os.environ, GH_TOKEN="", GITHUB_TOKEN="")
    return subprocess.run(["gh", *args], capture_output=True, text=True,
                          env=env, timeout=kw.pop("timeout", 60), **kw)


def status() -> dict:
    """-> {logged_in, user, scopes, error}. Never raises."""
    try:
        p = _gh(["auth", "status", "--hostname", HOST, "--active"])
    except FileNotFoundError:
        return {"logged_in": False, "user": "", "scopes": "",
                "error": "the `gh` CLI is not installed or not on PATH"}
    except Exception as e:
        return {"logged_in": False, "user": "", "scopes": "", "error": str(e)}
    text = ANSI_RE.sub("", (p.stdout or "") + (p.stderr or ""))
    if p.returncode != 0:
        return {"logged_in": False, "user": "", "scopes": "",
                "error": text.strip()}
    user = ""
    m = re.search(r"Logged in to \S+ account (\S+)", text)
    if m:
        user = m.group(1)
    scopes = ""
    m = re.search(r"Token scopes:\s*(.+)", text)
    if m:
        scopes = m.group(1).strip().replace("'", "")
    return {"logged_in": bool(user), "user": user, "scopes": scopes, "error": ""}


def missing_scopes() -> list[str]:
    """Scopes the dashboard needs that the signed-in token does not have.

    Worth surfacing: a token without `repo` reads every PR perfectly well and
    then fails only when you click Re-run — which reads as a broken button
    rather than a missing permission.
    """
    have = {s.strip() for s in (status().get("scopes") or "").split(",") if s.strip()}
    if not have:
        return []
    return [s for s in ("repo", "read:org", "workflow") if s not in have]


def login_state() -> dict:
    with LOGIN_LOCK:
        st = dict(LOGIN)
    if st["state"] == "waiting" and time.time() - st["started"] > LOGIN_TIMEOUT:
        cancel("the one-time code expired")
        with LOGIN_LOCK:
            st = dict(LOGIN)
    return st


def cancel(why: str = "cancelled") -> None:
    with LOGIN_LOCK:
        pid = LOGIN.get("pid")
        LOGIN.update(state="error" if why != "cancelled" else "idle",
                     code="", error="" if why == "cancelled" else why, pid=None)
    if pid:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def logout() -> tuple[bool, str]:
    try:
        p = _gh(["auth", "logout", "--hostname", HOST])
    except Exception as e:
        return False, str(e)
    text = ANSI_RE.sub("", (p.stdout or "") + (p.stderr or "")).strip()
    return p.returncode == 0, text or "signed out"


def login_with_token(token: str) -> tuple[bool, str]:
    """PAT fallback, for when device flow cannot reach GitHub (proxy, firewall).

    The token is passed on stdin and never touched again — not stored by us,
    not echoed back, not written to the log.
    """
    token = (token or "").strip()
    if not token:
        return False, "no token given"
    try:
        p = _gh(["auth", "login", "--hostname", HOST, "--with-token"],
                input=token)
    except Exception as e:
        return False, str(e)
    if p.returncode != 0:
        text = ANSI_RE.sub("", (p.stderr or p.stdout or "")).strip()
        return False, text or "gh rejected the token"
    s = status()
    return True, f"signed in as {s['user']}" if s["user"] else "signed in"


def start_login() -> str:
    """Begin the device flow. Returns "" on success, else why not."""
    with LOGIN_LOCK:
        if LOGIN["state"] == "waiting":
            return "a sign-in is already in progress"
        LOGIN.update(state="starting", code="", started=time.time(),
                     error="", pid=None)
    threading.Thread(target=_run_login, daemon=True).start()
    # Wait briefly for the code so the first page redraw already shows it;
    # the thread keeps running either way.
    for _ in range(60):
        time.sleep(0.25)
        with LOGIN_LOCK:
            if LOGIN["state"] != "starting":
                break
    return ""


def _run_login() -> None:
    env = dict(os.environ, GH_TOKEN="", GITHUB_TOKEN="",
               # Headless host: gh must not try to spawn a browser, and the
               # user opens the URL on whatever machine their browser is on.
               BROWSER="true", GH_PROMPTER_DISABLED="")
    master, slave = pty.openpty()
    try:
        p = subprocess.Popen(
            ["gh", "auth", "login", "--hostname", HOST, "--git-protocol",
             "https", "--web", "--skip-ssh-key", "-s", SCOPES],
            stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    except Exception as e:
        os.close(master)
        os.close(slave)
        with LOGIN_LOCK:
            LOGIN.update(state="error", error=f"could not start gh: {e}", pid=None)
        return
    os.close(slave)
    with LOGIN_LOCK:
        LOGIN["pid"] = p.pid

    buf, sent_enter, deadline = "", False, time.time() + LOGIN_TIMEOUT
    try:
        while p.poll() is None and time.time() < deadline:
            r, _, _ = select.select([master], [], [], 1.0)
            if r:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += ANSI_RE.sub("", chunk.decode(errors="replace"))
            m = CODE_RE.search(buf)
            if m:
                with LOGIN_LOCK:
                    if LOGIN["state"] != "waiting":
                        LOGIN.update(state="waiting", code=m.group(1))
                if not sent_enter and "Press Enter" in buf:
                    # Our side of the prompt. BROWSER=true makes this a no-op
                    # rather than a browser launch, and gh then starts polling
                    # GitHub for the code to be entered.
                    try:
                        os.write(master, b"\n")
                    except OSError:
                        pass
                    sent_enter = True
        rc = p.poll()
        if rc is None:
            p.send_signal(signal.SIGKILL)
            p.wait()
            with LOGIN_LOCK:
                LOGIN.update(state="error", code="", pid=None,
                             error="timed out waiting for the code to be entered")
            return
        if rc == 0:
            with LOGIN_LOCK:
                LOGIN.update(state="done", code="", error="", pid=None)
        else:
            # gh's failure text is the useful part ("error validating token",
            # a proxy refusal). Keep the tail, drop the prompt noise.
            tail = [ln for ln in buf.splitlines() if ln.strip()][-3:]
            with LOGIN_LOCK:
                LOGIN.update(state="error", code="", pid=None,
                             error=" / ".join(tail) or f"gh exited {rc}")
    finally:
        try:
            os.close(master)
        except OSError:
            pass
