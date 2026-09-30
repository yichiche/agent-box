#!/usr/bin/env python3
"""pr-ci-watch dashboard — stdlib HTTP server, no dependencies.

Deliberately inert: it reads and writes watchlist.json / state.json and nothing
else. It makes no `gh` calls and cannot re-run a workflow or post a comment.
Every outward-facing action stays in `watch.py sweep`, which runs under a Claude
turn — so the web layer never needs credentials and has no blast radius.

The one control that *does* bite immediately is the ON/OFF switch: it writes
`_config.enabled`, which every sweep (including a cron-fired one) checks before
doing anything. Pausing therefore works from the browser with no Claude turn.
"""

from __future__ import annotations

import html
import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watch import (  # noqa: E402
    DATA_DIR,
    PRIORITIES,
    STATE,
    WATCHLIST,
    ci_action,
    ci_verdict,
    load,
    monitoring_enabled,
    now,
    parse_pr,
    priority_of,
    report_text,
    save,
    set_monitoring,
    tw,
)

REFRESH_SECONDS = 60

# "Refresh now" runs phase A only, and deliberately WITHOUT --apply: it re-reads
# every PR's merge state and red NVIDIA CI so the table is current, but posts no
# comment and re-runs nothing. A button that could comment on someone else's PR
# is not something a stray click should reach. The full pipeline (which includes
# /ci-analysis) needs a Claude turn — ask for `/pr-ci-watch sweep now`.
SWEEP = {"running": False, "started": "", "finished": "", "output": "", "rc": None}
SWEEP_LOCK = threading.Lock()


def run_refresh() -> None:
    here = Path(__file__).resolve().parent
    try:
        p = subprocess.run(
            [sys.executable, str(here / "watch.py"), "sweep", "--track", "all", "--force"],
            capture_output=True, text=True, timeout=900, cwd=str(here),
        )
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "refresh timed out after 15 min", 1
    except Exception as e:  # never let the worker kill the server
        out, rc = f"refresh failed: {e}", 1
    with SWEEP_LOCK:
        SWEEP.update(running=False, finished=now(), output=out, rc=rc)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>pr-ci-watch</title>
<style>
  :root {{
    --bg:#0f1115; --panel:#171a21; --line:#272b34; --fg:#e6e8ee; --dim:#9aa1b1;
    --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#f6f7f9; --panel:#fff; --line:#e1e4e8; --fg:#1c2128; --dim:#59636e;
             --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --accent:#0969da; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
    font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  h1 {{ font-size:18px; margin:0; }}
  h2 {{ font-size:13px; margin:0 0 10px; color:var(--dim);
    text-transform:uppercase; letter-spacing:.04em; }}
  .sub {{ color:var(--dim); font-size:12px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px 16px; margin-bottom:18px; }}
  .bar {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap;
    margin-bottom:18px; }}
  .grow {{ flex:1; }}
  form.add {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
  input[type=text] {{ flex:1; min-width:280px; padding:8px 10px; border-radius:6px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg); font:inherit; }}
  select, button {{ padding:8px 12px; border-radius:6px; border:1px solid var(--line);
    background:var(--bg); color:var(--fg); font:inherit; cursor:pointer; }}
  button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }}
  button.on {{ background:var(--ok); border-color:var(--ok); color:#fff; font-weight:700; }}
  button.off {{ background:var(--bad); border-color:var(--bad); color:#fff; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
    color:var(--dim); font-weight:600; padding:0 8px 8px; border-bottom:1px solid var(--line); }}
  td {{ padding:9px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  a {{ color:var(--accent); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .title {{ display:block; max-width:32ch; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .dim {{ color:var(--dim); font-size:12px; }}
  .pill {{ display:inline-block; padding:1px 8px; border-radius:99px; font-size:11px;
    font-weight:600; border:1px solid currentColor; }}
  .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .bad {{ color:var(--bad); }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .empty {{ color:var(--dim); padding:28px; text-align:center; }}
  .reason {{ max-width:40ch; white-space:normal; word-break:break-word; margin-top:2px; }}
  .inline {{ display:inline; }}
  .linkish {{ background:none; border:none; color:var(--dim); padding:0 4px;
    font:inherit; font-size:12px; cursor:pointer; }}
  .linkish:hover {{ color:var(--bad); text-decoration:underline; }}
  .note {{ width:100%; min-width:160px; padding:5px 7px; border-radius:5px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg);
    font:inherit; font-size:12px; }}
  textarea#report {{ width:100%; height:190px; padding:11px; border-radius:6px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg);
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
    line-height:1.6; resize:vertical; }}
  .banner {{ padding:9px 13px; border-radius:6px; font-size:13px;
    border:1px solid currentColor; margin-bottom:14px; }}
</style></head><body>

<div class="bar">
  <div>
    <h1>pr-ci-watch</h1>
    <div class="sub">{repo} &middot; NVIDIA CI only &middot; {generated}</div>
  </div>
  <div class="grow"></div>
  <form class="inline" method="post" action="/api/refresh">
    <button class="primary" type="submit" {refresh_disabled}>{refresh_label}</button>
  </form>
  <form class="inline" method="post" action="/api/monitoring">
    <input type="hidden" name="on" value="{toggle_to}">
    <button class="{toggle_class}" type="submit">{toggle_label}</button>
  </form>
  <span class="sub">{arm_line}</span>
</div>

{banner}
{refresh_status}
{triage_panel}

<div class="panel"><form class="add" method="post" action="/api/add">
  <input type="text" name="ref" placeholder="Paste a PR link or number — https://github.com/{repo}/pull/41870" autofocus>
  <select name="track">
    <option value="regular">regular &middot; daily</option>
    <option value="high">high &middot; every 2h</option>
  </select>
  <select name="priority">
    <option value="">priority: auto</option>
    {prio_options}
  </select>
  <button class="primary" type="submit">Watch</button>
</form></div>

<div class="panel">{table}</div>

<div class="panel">
  <h2>Status block &mdash; paste into Teams / standup</h2>
  <textarea id="report" readonly>{report}</textarea>
  <div style="margin-top:10px; display:flex; gap:10px; align-items:center;">
    <button class="primary" type="button" onclick="copyReport(this)">Copy</button>
    <span class="sub">Edit the second line of each entry in the <b>Note</b>
      column above. <span class="mono">&lt;CI …&gt;</span> comes from the last sweep.</span>
  </div>
</div>

<div class="sub">Re-runs and conflict comments only happen during a sweep
(<span class="mono">/pr-ci-watch sweep</span>). This page edits the watchlist and
the ON/OFF switch. Data: <span class="mono">{data}</span></div>

<script>
  function copyEl(id, btn) {{
    const t = document.getElementById(id);
    t.select();
    navigator.clipboard.writeText(t.value).then(
      () => {{ btn.textContent = 'Copied'; setTimeout(() => btn.textContent = 'Copy', 1400); }},
      () => {{ document.execCommand('copy'); btn.textContent = 'Copied'; }}
    );
  }}
  function copyReport(btn) {{ copyEl('report', btn); }}
  // Refresh on a timer, but never while a field is focused — otherwise a note
  // being typed gets wiped mid-edit.
  setInterval(() => {{
    if (!document.querySelector('input:focus, textarea:focus, select:focus')) location.reload();
  }}, {refresh}000);
  // Submit a note on blur or Enter so there is no per-row save button.
  document.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('input.note').forEach(el => {{
      const initial = el.value;
      el.addEventListener('blur', () => {{ if (el.value !== initial) el.form.submit(); }});
      el.addEventListener('keydown', e => {{ if (e.key === 'Enter') {{ e.preventDefault(); el.form.submit(); }} }});
    }});
  }});
</script>
</body></html>
"""

EMPTY = '<div class="empty">Nothing watched yet — paste a PR link above.</div>'

ACTION_CLASS = {
    "green": "ok",
    "re-run": "warn",
    "awaiting-triage": "warn",
    "conflict": "bad",
    "code-fix": "bad",
    "merge-main": "warn",
    "wait-upstream": "warn",
    "out-of-scope": "dim",
}

# The whole point of the Verdict column: say what to DO, not just what happened.
# This is the "just re-run it" vs "stuck, needs a merge" split.
ACTION_HINT = {
    "green": "nothing to do",
    "awaiting-triage": "QUEUED — nothing is running; waiting for a Claude turn",
    "re-run": "re-ran; waiting on CI",
    "code-fix": "real bug in this PR — author must fix",
    "merge-main": "STUCK: PR is behind main — merge/rebase main",
    "conflict": "git conflict with main — author must resolve; /pr-conflict-fix",
    "wait-upstream": "STUCK: an NVIDIA job is blocked on an upstream fix",
    "out-of-scope": "nothing to do — red is outside NVIDIA scope",
}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def merge_cell(s: dict) -> str:
    m = s.get("mergeable", "?")
    if m == "CONFLICTING":
        return '<span class="pill bad">conflict</span>'
    if m == "MERGEABLE":
        return '<span class="pill ok">clean</span>'
    return f'<span class="pill warn">{esc(str(m).lower())}</span>'


def job_links(g: dict, names: list, limit: int = 4) -> str:
    """Link straight to the failing jobs — the point of the column is to get you
    to the log in one click, not to name a workflow you then have to go find."""
    links = g.get("job_links") or {}
    out = []
    for n in names[:limit]:
        short = n.split(" / ")[-1]
        url = links.get(n)
        out.append(f'<a href="{esc(url)}" target="_blank">{esc(short)}</a>'
                   if url else esc(short))
    if len(names) > limit:
        out.append(f'<span class="dim">+{len(names) - limit} more</span>')
    return ", ".join(out)


def ci_cell(s: dict) -> str:
    groups = s.get("failed_groups") or {}
    if s.get("last_action") == "green":
        return '<span class="ok">clean</span>'
    if not groups:
        return '<span class="dim">&mdash;</span>'
    out = []
    # After a re-run the stored failure list describes the run we *replaced*.
    # Showing it as if it were current is what made this column unreadable.
    last_rerun = max(
        (r.get("at", "") for r in (s.get("reruns") or {}).values()
         if r.get("sha") == s.get("head_sha")),
        default="",
    )
    if last_rerun and last_rerun > (s.get("last_sweep") or ""):
        out.append(
            f'<div class="warn"><b>stale</b> &mdash; state from before the '
            f'{esc(tw(last_rerun))} re-run. Hit <b>Refresh now</b>.</div>'
        )
    for wf, g in groups.items():
        if g.get("gate_only"):
            out.append(f'<div class="dim">{esc(wf)}: gate only, not re-runnable</div>')
            continue
        if g.get("watcher_jobs"):
            out.append(
                f'<div class="warn">{esc(wf)}: watcher died &mdash; '
                f'{job_links(g, g["watcher_jobs"])}</div>'
            )
        if g.get("jobs"):
            out.append(
                f'<div class="bad"><b>{esc(wf)}</b> ({len(g["jobs"])} failed):<br>'
                f'{job_links(g, g["jobs"])}</div>'
            )
    return "".join(out)


VERDICT_CLASS = {"Pass": "ok", "Running": "warn", "Fail": "bad", "—": "dim"}
ACTION_COLOR = {
    "Solve conflict": "bad",
    "Merge main": "bad",
    "Code fix": "bad",
    "Re-run failed CI": "warn",
    "Triage": "warn",
    "Wait upstream": "warn",
    "Pending": "dim",
    "-": "dim",
}


def verdict_cell(s: dict) -> str:
    """Current CI state: Running / Pass / Fail."""
    v = ci_verdict(s)
    t = s.get("tally") or {}
    bits = [f"{n} {k}" for k, n in
            (("pass", t.get("pass", 0)), ("fail", t.get("fail", 0)),
             ("running", t.get("pending", 0))) if n]
    detail = f'<div class="dim">{esc(", ".join(bits))}</div>' if bits else ""
    return f'<span class="pill {VERDICT_CLASS.get(v, "dim")}">{esc(v)}</span>{detail}'


def action_cell(s: dict) -> str:
    """What to do about it — and proof of what was already done."""
    a = ci_action(s)
    out = [f'<span class="pill {ACTION_COLOR.get(a, "dim")}">{esc(a)}</span>']
    sha = s.get("head_sha", "")
    for wf, rec in (s.get("reruns") or {}).items():
        if rec.get("sha") != sha:
            continue  # budget resets on a new push; stale rows are noise
        out.append(f'<div class="dim">{esc(wf)} re-run &times;{rec.get("count", 0)}'
                   f' &middot; {esc(tw(rec.get("at")))}</div>')
    if a == "Re-run failed CI" and not (s.get("reruns") or {}):
        out.append('<div class="dim">queued — GitHub refused while the run was '
                   "still going; next sweep retries</div>")
    return "".join(out)


def notify_cell(s: dict) -> str:
    """Conflicts are the one thing this tool says to a third party — so show
    proof it happened, or say plainly that it has not."""
    if s.get("mergeable") != "CONFLICTING":
        return ""
    if s.get("conflict_comment_sha") != s.get("head_sha"):
        return ('<div class="bad"><b>NOT notified yet</b> — the author has not '
                "been told about this conflict.</div>")
    url = s.get("conflict_comment_url", "")
    link = f' &middot; <a href="{esc(url)}" target="_blank">see comment</a>' if url else ""
    return (f'<div class="ok"><b>Notified</b> @{esc(s.get("author"))} '
            f"{esc(tw(s.get('conflict_comment_at')))}{link}</div>")


def detail_cell(s: dict) -> str:
    """Why, in words. No input field — this column is read-only now."""
    out = []
    n = notify_cell(s)
    if n:
        out.append(n)
    hint = ACTION_HINT.get(s.get("last_action", ""), "")
    if hint:
        out.append(f"<div>{esc(hint)}</div>")
    # Wrap, never truncate: cutting the reason mid-word ("…OOM on a 32GB GPU;
    # un") is worse than showing no reason at all.
    if s.get("last_verdict"):
        out.append(f'<div class="dim reason">{esc(s["last_verdict"])}</div>')
    return "".join(out) or '<span class="dim">&mdash;</span>'


def prio_cell(pr: str, meta: dict) -> str:
    opts = "".join(
        f'<option value="{p}"{" selected" if priority_of(meta) == p else ""}>{p}</option>'
        for p in PRIORITIES
    )
    return (
        f'<form class="inline" method="post" action="/api/priority">'
        f'<input type="hidden" name="pr" value="{esc(pr)}">'
        f'<select name="priority" onchange="this.form.submit()">{opts}</select></form>'
    )


def render_table(wl: dict, st: dict) -> str:
    if not wl:
        return EMPTY
    head = (
        "<tr><th>Pri</th><th>PR</th><th>Track</th><th>Merge</th><th>Red NVIDIA CI</th>"
        "<th>Verdict</th><th>Action</th><th>Why</th>"
        "<th>Last swept (TW)</th><th></th></tr>"
    )
    rows = []
    for pr, meta in sorted(wl.items(), key=lambda kv: (priority_of(kv[1]), int(kv[0]))):
        s = st.get(pr, {})
        track = meta.get("track", "regular")
        other = "regular" if track == "high" else "high"
        url = s.get("url") or f"https://github.com/{meta.get('repo', '')}/pull/{pr}"
        rows.append(
            "<tr>"
            f"<td>{prio_cell(pr, meta)}</td>"
            f'<td><a href="{esc(url)}" target="_blank"><b>#{esc(pr)}</b></a>'
            f'<span class="title dim" title="{esc(s.get("title"))}">{esc(s.get("title"))}</span>'
            f'<span class="dim">{"@" + esc(s.get("author")) if s.get("author") else ""}</span></td>'
            f'<td><form class="inline" method="post" action="/api/track">'
            f'<input type="hidden" name="pr" value="{esc(pr)}">'
            f'<input type="hidden" name="track" value="{other}">'
            f'<button class="linkish" title="switch to {other}">'
            f'<span class="pill {"bad" if track == "high" else "dim"}">{esc(track)}</span>'
            f"</button></form></td>"
            f"<td>{merge_cell(s)}</td>"
            f"<td>{ci_cell(s)}</td>"
            f"<td>{verdict_cell(s)}</td>"
            f"<td>{action_cell(s)}</td>"
            f"<td>{detail_cell(s)}</td>"
            f'<td class="mono dim">{esc(tw(s.get("last_sweep")))}</td>'
            f'<td><form class="inline" method="post" action="/api/remove">'
            f'<input type="hidden" name="pr" value="{esc(pr)}">'
            f'<button class="linkish" title="stop watching">remove</button>'
            f"</form></td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def render_banner(st: dict, enabled: bool) -> str:
    """Say plainly what the switch does and does not cover."""
    if not enabled:
        since = st.get("_config", {}).get("toggled_at", "?")
        return (
            f'<div class="banner bad">Monitoring is <b>PAUSED</b> since {esc(since)}. '
            f"Scheduled sweeps still fire but exit immediately &mdash; no re-runs, "
            f"no conflict comments.</div>"
        )
    if not st.get("_arm", {}).get("armed_at"):
        return (
            '<div class="banner warn">Monitoring is ON, but the schedule is '
            "<b>not armed</b>, so nothing sweeps on its own yet. Run "
            '<span class="mono">/pr-ci-watch arm</span> in Claude once &mdash; '
            "registering cron needs a Claude turn, so this page cannot do it.</div>"
        )
    return ""


def render_refresh_status() -> str:
    with SWEEP_LOCK:
        s = dict(SWEEP)
    if s["running"]:
        return (
            f'<div class="banner warn">Refreshing since {esc(s["started"])}… '
            f"reading merge state and CI for every watched PR. This page reloads "
            f"when it finishes; nothing is being commented or re-run.</div>"
        )
    if not s["finished"]:
        return ""
    cls = "ok" if s["rc"] == 0 else "bad"
    return (
        f'<details class="panel"><summary class="{cls}">'
        f'<b>Last refresh {esc(s["finished"])}</b> '
        f'<span class="dim">(read-only — click to see what it found)</span></summary>'
        f'<pre class="mono" style="white-space:pre-wrap; margin:10px 0 0;">'
        f'{esc(s["output"][-6000:])}</pre></details>'
    )


def render_triage_panel(wl: dict, st: dict) -> str:
    """PRs whose red CI has no verdict yet. Only Claude can clear these."""
    pending = [
        pr for pr in sorted(wl, key=int)
        if st.get(pr, {}).get("last_action") == "awaiting-triage"
    ]
    if not pending:
        return ""
    cmd = "/pr-ci-watch triage " + " ".join(pending)
    return (
        f'<div class="panel"><h2>Waiting on triage &mdash; {len(pending)} PR(s)</h2>'
        f'<div class="sub" style="margin-bottom:10px;">'
        f"<b>Nothing is running right now</b> &mdash; these are parked, not in "
        f"progress. They have real failing NVIDIA jobs but no verdict yet, and "
        f"deciding <b>re-run</b> vs <b>merge main</b> vs <b>real bug</b> means "
        f"reading the job logs, which needs a Claude turn &mdash; neither this "
        f"page nor <i>Refresh now</i> can do it. The next scheduled sweep picks "
        f"them up automatically; to do it now, paste this to Claude:</div>"
        f'<div style="display:flex; gap:10px; align-items:center;">'
        f'<input class="note" id="triagecmd" type="text" readonly value="{esc(cmd)}">'
        f'<button class="primary" type="button" onclick="copyEl(\'triagecmd\', this)">Copy</button>'
        f"</div></div>"
    )


def arm_line(st: dict) -> str:
    from datetime import datetime, timezone

    from watch import parse_ts

    armed = parse_ts(st.get("_arm", {}).get("armed_at"))
    if not armed:
        return "schedule: not armed"
    days = (datetime.now(timezone.utc) - armed).days
    tail = " — expires in 7d, re-arm" if days >= 6 else ""
    # "recorded" not "armed": this process cannot see Claude's scheduler.
    return f"schedule: recorded armed {days}d ago{tail}"


class Handler(BaseHTTPRequestHandler):
    repo = "sgl-project/sglang"
    server_version = "pr-ci-watch"

    def log_message(self, fmt, *args):  # quieter than the default stderr spam
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        wl, raw_st = load(WATCHLIST, {}), load(STATE, {})
        st = {k: v for k, v in raw_st.items() if not k.startswith("_")}
        if self.path.startswith("/api/report"):
            self._send(200, report_text(wl, st).encode(), "text/plain; charset=utf-8")
            return
        if self.path.startswith("/api/state"):
            self._send(
                200,
                json.dumps(
                    {
                        "watchlist": wl,
                        "state": st,
                        "enabled": monitoring_enabled(raw_st),
                    },
                    indent=2,
                ).encode(),
                "application/json",
            )
            return
        enabled = monitoring_enabled(raw_st)
        with SWEEP_LOCK:
            running = SWEEP["running"]
        page = PAGE.format(
            # Poll faster while a refresh is in flight so the result appears on
            # its own instead of after a 60s wait.
            refresh=5 if running else REFRESH_SECONDS,
            refresh_label="Refreshing…" if running else "Refresh now",
            refresh_disabled="disabled" if running else "",
            refresh_status=render_refresh_status(),
            triage_panel=render_triage_panel(wl, st),
            repo=html.escape(self.repo),
            generated=now(),
            data=html.escape(str(DATA_DIR)),
            table=render_table(wl, st),
            report=html.escape(report_text(wl, st)),
            banner=render_banner(raw_st, enabled),
            arm_line=html.escape(arm_line(raw_st)),
            toggle_to="0" if enabled else "1",
            toggle_class="on" if enabled else "off",
            toggle_label="Monitoring ON" if enabled else "Monitoring PAUSED",
            prio_options="".join(f'<option value="{p}">{p}</option>' for p in PRIORITIES),
        )
        self._send(200, page.encode(), "text/html; charset=utf-8")

    def _form(self) -> dict:
        from urllib.parse import parse_qs

        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode()
        return {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}

    def do_POST(self) -> None:
        form = self._form()
        try:
            if self.path == "/api/refresh":
                with SWEEP_LOCK:
                    if not SWEEP["running"]:
                        SWEEP.update(running=True, started=now(), finished="",
                                     output="", rc=None)
                        threading.Thread(target=run_refresh, daemon=True).start()
            elif self.path == "/api/monitoring":
                st = load(STATE, {})
                set_monitoring(st, form.get("on") == "1")
                save(STATE, st)
            else:
                wl = load(WATCHLIST, {})
                if self.path == "/api/add":
                    pr = parse_pr(form.get("ref", ""))
                    prev = wl.get(pr, {})
                    wl[pr] = {
                        "track": form.get("track", "regular"),
                        "added": prev.get("added", now()),
                        "note": prev.get("note", ""),
                        "repo": self.repo,
                    }
                    if form.get("priority") in PRIORITIES:
                        wl[pr]["priority"] = form["priority"]
                    elif prev.get("priority"):
                        wl[pr]["priority"] = prev["priority"]
                elif self.path == "/api/track":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl:
                        wl[pr]["track"] = form.get("track", "regular")
                elif self.path == "/api/priority":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl and form.get("priority") in PRIORITIES:
                        wl[pr]["priority"] = form["priority"]
                elif self.path == "/api/note":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl:
                        wl[pr]["note"] = form.get("note", "").strip()
                elif self.path == "/api/remove":
                    wl.pop(parse_pr(form.get("pr", "")), None)
                else:
                    self._send(404, b"no such endpoint", "text/plain")
                    return
                save(WATCHLIST, wl)
        except SystemExit:
            # parse_pr calls die() on garbage input; a bad paste should not 500.
            self._send(400, b"could not read a PR number out of that input",
                       "text/plain; charset=utf-8")
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


class V4Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class V6Server(ThreadingHTTPServer):
    # `localhost` resolves to ::1 first on this host, and an editor port
    # forwarder that does not fall back to IPv4 gets "connection reset" if only
    # 127.0.0.1 is bound. Binding both loopbacks is what inferencex-plot needed
    # a separate node bridge for; two listeners is cheaper and stays off the LAN
    # (binding :: would expose the port to the whole network).
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8812
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    v4 = V4Server(("127.0.0.1", port), Handler)
    try:
        v6 = V6Server(("::1", port), Handler)
        threading.Thread(target=v6.serve_forever, daemon=True).start()
        where = f"http://127.0.0.1:{port}/ and http://[::1]:{port}/"
    except OSError as e:
        where = f"http://127.0.0.1:{port}/ (IPv6 loopback unavailable: {e})"
    print(f"pr-ci-watch dashboard on {where}", flush=True)
    v4.serve_forever()


if __name__ == "__main__":
    main()
