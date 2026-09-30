#!/usr/bin/env python3
"""pr-ci-watch dashboard — stdlib HTTP server, no dependencies.

Deliberately inert: it reads and writes watchlist.json / state.json and nothing
else. It makes no `gh` calls and cannot re-run a workflow or post a comment.
Every outward-facing action stays in `watch.py sweep`, which runs under a Claude
turn — so the web layer never needs credentials and has no blast radius.

Binds dual-stack loopback (AF_INET6 with IPV6_V6ONLY off) so 127.0.0.1 and ::1
both answer. Never 0.0.0.0.
"""

from __future__ import annotations

import html
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watch import DATA_DIR, MAX_RERUNS, STATE, WATCHLIST, load, now, parse_pr, save  # noqa: E402

REFRESH_SECONDS = 60

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>pr-ci-watch</title>
<meta http-equiv="refresh" content="{refresh}">
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
  h1 {{ font-size:18px; margin:0 0 2px; }}
  .sub {{ color:var(--dim); font-size:12px; margin-bottom:18px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px 16px; margin-bottom:18px; }}
  form.add {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
  input[type=text] {{ flex:1; min-width:320px; padding:8px 10px; border-radius:6px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg); font:inherit; }}
  select, button {{ padding:8px 12px; border-radius:6px; border:1px solid var(--line);
    background:var(--bg); color:var(--fg); font:inherit; cursor:pointer; }}
  button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
    color:var(--dim); font-weight:600; padding:0 10px 8px; border-bottom:1px solid var(--line); }}
  td {{ padding:10px; border-bottom:1px solid var(--line); vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  a {{ color:var(--accent); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .title {{ display:block; max-width:34ch; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .dim {{ color:var(--dim); font-size:12px; }}
  .pill {{ display:inline-block; padding:1px 8px; border-radius:99px; font-size:11px;
    font-weight:600; border:1px solid currentColor; }}
  .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .bad {{ color:var(--bad); }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .empty {{ color:var(--dim); padding:28px; text-align:center; }}
  .inline {{ display:inline; }}
  .linkish {{ background:none; border:none; color:var(--dim); padding:0 4px;
    font:inherit; font-size:12px; cursor:pointer; }}
  .linkish:hover {{ color:var(--bad); text-decoration:underline; }}
</style></head><body>
<h1>pr-ci-watch</h1>
<div class="sub">{repo} &middot; NVIDIA CI only &middot; auto-refresh {refresh}s &middot; {generated}</div>

<div class="panel"><form class="add" method="post" action="/api/add">
  <input type="text" name="ref" placeholder="Paste a PR link or number — https://github.com/{repo}/pull/41870" autofocus>
  <select name="track">
    <option value="regular">regular &middot; daily sweep</option>
    <option value="high">high &middot; every 2h</option>
  </select>
  <button class="primary" type="submit">Watch</button>
</form></div>

<div class="panel">{table}</div>
<div class="sub">Re-runs and conflict comments only happen during a sweep
(<span class="mono">/pr-ci-watch sweep</span>). This page just edits the watchlist.
Data: <span class="mono">{data}</span></div>
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
}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def merge_cell(s: dict) -> str:
    m = s.get("mergeable", "?")
    if m == "CONFLICTING":
        return '<span class="pill bad">conflict</span>'
    if m == "MERGEABLE":
        return '<span class="pill ok">clean</span>'
    return f'<span class="pill warn">{esc(m.lower())}</span>'


def ci_cell(s: dict) -> str:
    groups = s.get("failed_groups") or {}
    if s.get("last_action") == "green":
        return '<span class="ok">clean</span>'
    if not groups:
        return '<span class="dim">—</span>'
    parts = []
    for wf, g in groups.items():
        if g.get("gate_only"):
            parts.append(
                f'<div class="mono warn">{esc(wf)} <span class="dim">'
                f"(gate only &mdash; not re-runnable)</span></div>"
            )
        else:
            parts.append(
                f'<div class="mono bad">{esc(wf)} <span class="dim">'
                f'({len(g.get("jobs", []))} failed)</span></div>'
            )
    return "".join(parts)


def rerun_cell(s: dict) -> str:
    sha = s.get("head_sha", "")
    rows = []
    for wf, rec in (s.get("reruns") or {}).items():
        if rec.get("sha") != sha:
            continue  # budget resets on a new push; stale rows are noise
        n = rec.get("count", 0)
        cls = "bad" if n >= MAX_RERUNS else "warn"
        rows.append(f'<div class="mono {cls}">{esc(wf)} {n}/{MAX_RERUNS}</div>')
    return "".join(rows) or '<span class="dim">—</span>'


def render_table(wl: dict, st: dict) -> str:
    if not wl:
        return EMPTY
    head = (
        "<tr><th>PR</th><th>Track</th><th>Merge</th><th>Red NVIDIA CI</th>"
        "<th>Verdict</th><th>Re-runs</th><th>Last swept</th><th></th></tr>"
    )
    rows = []
    for pr, meta in sorted(wl.items(), key=lambda kv: int(kv[0])):
        s = st.get(pr, {})
        track = meta.get("track", "regular")
        other = "regular" if track == "high" else "high"
        url = s.get("url") or f"https://github.com/{meta.get('repo', '')}/pull/{pr}"
        action = s.get("last_action", "—")
        cls = ACTION_CLASS.get(action, "dim")
        verdict = s.get("last_verdict", "")
        rows.append(
            f"<tr>"
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
            f'<td><span class="{cls}">{esc(action)}</span>'
            f'<div class="dim">{esc(verdict[:90])}</div></td>'
            f"<td>{rerun_cell(s)}</td>"
            f'<td class="mono dim">{esc(s.get("last_sweep", "never"))}</td>'
            f'<td><form class="inline" method="post" action="/api/remove">'
            f'<input type="hidden" name="pr" value="{esc(pr)}">'
            f'<button class="linkish" title="stop watching">remove</button>'
            f"</form></td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


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
        wl, st = load(WATCHLIST, {}), load(STATE, {})
        st = {k: v for k, v in st.items() if not k.startswith("_")}
        if self.path.startswith("/api/state"):
            self._send(
                200,
                json.dumps({"watchlist": wl, "state": st}, indent=2).encode(),
                "application/json",
            )
            return
        page = PAGE.format(
            refresh=REFRESH_SECONDS,
            repo=html.escape(self.repo),
            generated=now(),
            data=html.escape(str(DATA_DIR)),
            table=render_table(wl, st),
        )
        self._send(200, page.encode(), "text/html; charset=utf-8")

    def _form(self) -> dict:
        from urllib.parse import parse_qs

        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode()
        return {k: v[0] for k, v in parse_qs(raw).items()}

    def do_POST(self) -> None:
        form = self._form()
        wl = load(WATCHLIST, {})
        try:
            if self.path == "/api/add":
                pr = parse_pr(form.get("ref", ""))
                prev = wl.get(pr, {})
                wl[pr] = {
                    "track": form.get("track", "regular"),
                    "added": prev.get("added", now()),
                    "note": prev.get("note", ""),
                    "repo": self.repo,
                }
            elif self.path == "/api/track":
                pr = parse_pr(form.get("pr", ""))
                if pr in wl:
                    wl[pr]["track"] = form.get("track", "regular")
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
