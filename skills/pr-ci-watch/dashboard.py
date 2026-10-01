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
import re
import socket
from urllib.parse import quote, urlparse, parse_qs
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history import history_html  # noqa: E402
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
    UNGROUPED,
    all_groups,
    group_of,
    in_group,
    report_entries,
    report_text,
    row_order,
    tally_bits,
    TRACKS,
    GATE_BLOCKING,
    log_line,
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

# Result of the last outward-facing click, shown back to the user. A button that
# silently does nothing is worse than no button: you cannot tell "already sent"
# from "broken".
NOTICE = {"at": "", "ok": False, "text": ""}


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
  /* Scoped to the add box on purpose. As a bare `input[type=text]` rule this
     out-specifies the `.grp`/`.note` classes (an attribute selector counts like
     a class, so 0-1-1 beats 0-1-0) and forced the in-table Group field to
     280px — which under table-layout:fixed overflows its 104px column and
     paints its border straight across Track and Merge. */
  form.add input[type=text] {{ flex:1; min-width:280px; padding:8px 10px; border-radius:6px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg); font:inherit; }}
  select, button {{ padding:8px 12px; border-radius:6px; border:1px solid var(--line);
    background:var(--bg); color:var(--fg); font:inherit; cursor:pointer; }}
  button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }}
  button.on {{ background:var(--ok); border-color:var(--ok); color:#fff; font-weight:700; }}
  button.off {{ background:var(--bad); border-color:var(--bad); color:#fff; font-weight:700; }}
  /* Fixed layout is what stops a wide <select> in Group/Track from bullying
     Status — the column that carries all the prose — into a two-words-per-line
     ribbon. Widths come from the <colgroup>. */
  table {{ width:100%; border-collapse:collapse; table-layout:fixed;
    min-width:1150px; }}
  /* A 12-column table has a floor; below it, scroll rather than crush the
     knobs until their borders paint over the next column. */
  .panel.tabbed {{ overflow-x:auto; }}
  th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
    color:var(--dim); font-weight:600; padding:0 8px 8px; border-bottom:1px solid var(--line); }}
  td {{ padding:9px 8px; border-bottom:1px solid var(--line); vertical-align:top;
    overflow-wrap:anywhere; }}
  /* The knob columns: as small as the control, not as wide as its default. */
  td.knob {{ padding-left:4px; padding-right:4px; }}
  /* width:100% needs a block containing block; .inline forms would collapse. */
  td.knob form {{ display:block; }}
  /* Text wraps when a column is tight; a <select> does not — it just clips its
     own label ("P0" losing its 0). So the knob selects run a size smaller with
     minimal side padding, and their columns are sized for label + native
     dropdown arrow rather than for the text alone. */
  select.mini {{ width:100%; padding:4px 2px; font-size:11px; }}
  /* Drag handle. Only the grip starts a drag — a draggable <tr> would eat text
     selection and turn every PR link into a drag. */
  .grip {{ display:block; cursor:grab; color:var(--dim); user-select:none;
    text-align:center; font-size:15px; line-height:1; padding-top:2px; }}
  .grip:hover {{ color:var(--fg); }}
  tr.dragging {{ opacity:.4; }}
  tr.dragging .grip {{ cursor:grabbing; }}
  /* The one row a drop is not allowed into: a different sort bucket. */
  tr.nodrop {{ outline:1px dashed var(--bad); outline-offset:-1px; }}
  /* History toggle. It is labelled, not a bare triangle: in Status it sits
     under a stack of other small blocks, where a lone glyph reads as
     punctuation rather than a control. Only the triangle rotates. */
  .disc {{ background:none; border:none; color:var(--dim); cursor:pointer;
    font:inherit; font-size:11px; padding:3px 0 0; line-height:1;
    display:inline-flex; align-items:center; gap:4px; }}
  .disc:hover {{ color:var(--fg); }}
  .disc .tri {{ display:inline-block; transition:transform .12s; }}
  .disc.open .tri {{ transform:rotate(90deg); }}
  /* Full width, below the row: the long verdict summaries need the room, and
     indenting to Status would waste two thirds of the table on margin. */
  tr.hist > td {{ padding:2px 8px 10px 30px; border-bottom:1px solid var(--line); }}
  .histbox {{ font-size:12px; }}
  .histline {{ display:flex; gap:10px; align-items:baseline; padding:2px 0; }}
  .histat {{ flex:0 0 auto; width:76px; }}
  tr:last-child td {{ border-bottom:none; }}
  a {{ color:var(--accent); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .title {{ display:block; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .dim {{ color:var(--dim); font-size:12px; }}
  .pill {{ display:inline-block; max-width:100%; padding:1px 6px; border-radius:99px;
    font-size:11px; font-weight:600; border:1px solid currentColor; }}
  .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .bad {{ color:var(--bad); }}
  /* Filled rather than outlined: a draft row is otherwise all dim dashes, and
     the one cell carrying information should be the one that reads first. */
  .pill.draft {{ color:#fff; background:var(--dim); border-color:var(--dim);
    font-weight:600; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .empty {{ color:var(--dim); padding:28px; text-align:center; }}
  /* No ch cap any more: the column is sized by the colgroup, so capping the
     text as well only re-introduces the narrow ribbon this was widened to fix. */
  .reason {{ white-space:normal; word-break:break-word; margin-top:2px; }}
  .inline {{ display:inline; }}
  .linkish {{ background:none; border:none; color:var(--dim); padding:0 4px;
    font:inherit; font-size:12px; cursor:pointer; }}
  .linkish:hover {{ color:var(--bad); text-decoration:underline; }}
  .note {{ width:100%; min-width:160px; padding:5px 7px; border-radius:5px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg);
    font:inherit; font-size:12px; }}
  /* Nothing behind the status block: it is text to be read and copied, and
     any fill only competes with what it sits in. */
  .panel.plain {{ background:transparent; }}
  .tabs {{ display:flex; gap:2px; flex-wrap:wrap; margin-bottom:-1px;
    align-items:flex-end; }}
  .tab {{ padding:7px 14px; border:1px solid var(--line); border-bottom:none;
    border-radius:7px 7px 0 0; background:var(--bg); color:var(--dim);
    font-size:13px; text-decoration:none; user-select:none; }}
  .tab:hover {{ color:var(--fg); text-decoration:none; }}
  .tab.on {{ background:var(--panel); color:var(--fg); font-weight:600;
    border-color:var(--line); }}
  /* Chrome-style reordering: grab anywhere on the tab. The ghost stays in the
     bar while dragging so the drop position is visible before you let go. */
  .tab[draggable="true"] {{ cursor:grab; }}
  .tab.dragging {{ opacity:.45; cursor:grabbing; }}
  .panel.tabbed {{ border-radius:0 7px 7px 7px; }}
  .grp {{ min-width:0; }}
  .reportblock {{ width:100%; max-height:300px; overflow:auto; padding:2px 0;
    background:transparent; border:none;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
    line-height:1.65; }}
  .reportblock ul {{ margin:0; padding-left:1.3em; }}
  .reportblock > ul {{ padding-left:1.1em; }}
  .reportblock li {{ margin:1px 0; }}
  .reportblock ul ul {{ margin:1px 0 5px; }}
  .reportblock ul ul li {{ color:var(--dim); }}
  #reportsrc {{ position:absolute; left:-9999px; width:1px; height:1px; }}
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

{tabs}
<div class="panel tabbed">{table}</div>

<div class="panel plain">
  <h2>Status block &mdash; paste into Teams / standup</h2>
  <div id="report" class="reportblock">{report_html}</div>
  <textarea id="reportsrc" readonly aria-hidden="true">{report}</textarea>
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
  function flash(btn, msg) {{
    const old = btn.dataset.label || btn.textContent;
    btn.dataset.label = old;
    btn.textContent = msg;
    setTimeout(() => {{ btn.textContent = btn.dataset.label; }}, 1600);
  }}
  // execCommand works on plain http; navigator.clipboard does not exist at all
  // outside a secure context, so it must never be touched without a guard.
  function execCopy(text, btn) {{
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta);
    ta.focus(); ta.select();
    let ok = false;
    try {{ ok = document.execCommand('copy'); }} catch (e) {{}}
    document.body.removeChild(ta);
    if (ok) {{ flash(btn, 'Copied'); return; }}
    // Last resort: reveal the plain-text source so it can be copied by hand
    // rather than leaving the button looking dead.
    const src = document.getElementById('reportsrc');
    if (src) {{
      src.style.cssText = 'position:static;width:100%;height:170px';
      src.focus(); src.select();
    }}
    flash(btn, 'Press Ctrl+C');
  }}
  function plainCopy(text, btn) {{
    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(
        () => flash(btn, 'Copied'), () => execCopy(text, btn));
    }} else {{ execCopy(text, btn); }}
  }}
  function copyEl(id, btn) {{
    const el = document.getElementById(id);
    plainCopy(el.value !== undefined ? el.value : el.textContent, btn);
  }}
  function copyReport(btn) {{
    const text = document.getElementById('reportsrc').value;
    const html = document.getElementById('report').innerHTML;
    // text/html keeps <PRnnnnn> a hyperlink in Teams; text/plain is the
    // fallback for anywhere that strips markup.
    if (window.isSecureContext && window.ClipboardItem
        && navigator.clipboard && navigator.clipboard.write) {{
      navigator.clipboard.write([new ClipboardItem({{
        'text/html': new Blob([html], {{type: 'text/html'}}),
        'text/plain': new Blob([text], {{type: 'text/plain'}}),
      }})]).then(() => flash(btn, 'Copied'), () => plainCopy(text, btn));
    }} else {{ plainCopy(text, btn); }}
  }}
  // Refresh on a timer, but never while a field is focused — otherwise a note
  // being typed gets wiped mid-edit.
  // …or while a drag is in flight, which would drop the row mid-gesture.
  setInterval(() => {{
    if (document.querySelector('.dragging')) return;
    if (!document.querySelector('input:focus, textarea:focus, select:focus')) location.reload();
  }}, {refresh}000);
  // Submit a note on blur or Enter so there is no per-row save button.
  document.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('input.note, input.grp').forEach(el => {{
      const initial = el.value;
      el.addEventListener('blur', () => {{ if (el.value !== initial) el.form.submit(); }});
      el.addEventListener('keydown', e => {{ if (e.key === 'Enter') {{ e.preventDefault(); el.form.submit(); }} }});
    }});
    initTabDrag();
    initRowDrag();
    initHistory();
  }});

  // Per-PR history, fetched on first open rather than rendered into every row:
  // building it means reading the sweep archive, and a page that redraws every
  // 60s should not pay for panels nobody opened.
  function initHistory() {{
    document.querySelectorAll('.disc').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const pr = btn.dataset.pr;
        const row = document.querySelector(`tr.hist[data-for="${{pr}}"]`);
        if (!row) return;
        const open = row.hidden;
        row.hidden = !open;
        btn.classList.toggle('open', open);
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open && !row.dataset.loaded) {{
          row.dataset.loaded = '1';
          fetch('/api/history?pr=' + encodeURIComponent(pr))
            .then(r => r.text())
            .then(h => {{ row.querySelector('.histbox').innerHTML = h; }})
            .catch(() => {{
              row.dataset.loaded = '';
              row.querySelector('.histbox').textContent = 'could not load history';
            }});
        }}
      }});
    }});
  }}

  // Rows reorder by dragging their grip, same gesture as the tabs. The row
  // moves in the DOM as you drag, so the landing spot is visible before you
  // let go, and the order is persisted once on drop.
  function initRowDrag() {{
    const body = document.querySelector('#prtable tbody');
    if (!body) return;
    const rows = () => [...body.querySelectorAll('tr[data-pr]')];
    // A row owns the history panel directly beneath it; the two must travel
    // together or the panel ends up describing whichever row it lands under.
    const histOf = tr => {{
      const n = tr.nextElementSibling;
      return (n && n.classList.contains('hist')) ? n : null;
    }};
    const tailOf = tr => histOf(tr) || tr;
    let dragged = null, draggedHist = null, moved = false;
    rows().forEach(tr => {{
      const grip = tr.querySelector('.grip');
      if (!grip) return;
      // draggable is armed only while the grip is held: on the <tr> full-time
      // it would swallow text selection and link clicks across the whole row.
      grip.addEventListener('mousedown', () => {{ tr.draggable = true; }});
      grip.addEventListener('mouseup', () => {{ tr.draggable = false; }});
      tr.addEventListener('dragstart', e => {{
        dragged = tr; draggedHist = histOf(tr); moved = false;
        tr.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        try {{ e.dataTransfer.setData('text/plain', tr.dataset.pr); }} catch (_) {{}}
      }});
      tr.addEventListener('dragend', () => {{
        tr.classList.remove('dragging');
        tr.draggable = false;
        rows().forEach(r => r.classList.remove('nodrop'));
        if (moved) saveRowOrder();
        dragged = null; draggedHist = null;
      }});
      tr.addEventListener('dragover', e => {{
        if (!dragged || dragged === tr) return;
        // Pass-first / priority / conflict-last decide the buckets; manual
        // order is only a tiebreaker inside one. Refusing the drop is honest —
        // accepting it would just spring the row back on reload.
        if (tr.dataset.bucket !== dragged.dataset.bucket) {{
          tr.classList.add('nodrop');
          return;
        }}
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = tr.getBoundingClientRect();
        const before = (e.clientY - r.top) < r.height / 2;
        // Landing "after tr" means after tr's own history panel, not between
        // the two.
        body.insertBefore(dragged, before ? tr : tailOf(tr).nextSibling);
        if (draggedHist) body.insertBefore(draggedHist, dragged.nextSibling);
        moved = true;
      }});
      tr.addEventListener('dragleave', () => tr.classList.remove('nodrop'));
      tr.addEventListener('drop', e => e.preventDefault());
    }});
    function saveRowOrder() {{
      document.getElementById('roworder').value =
        rows().map(r => r.dataset.pr).join(',');
      document.getElementById('roworderform').submit();
    }}
  }}

  // Chrome-style tab reordering. The tab is moved in the DOM as you drag, so
  // the drop position is visible before you let go; the order is persisted on
  // drop, not on every move.
  function initTabDrag() {{
    const bar = document.getElementById('tabbar');
    if (!bar) return;
    const tabs = () => [...bar.querySelectorAll('.tab[draggable="true"]')];
    let dragged = null, moved = false;
    tabs().forEach(t => {{
      t.addEventListener('dragstart', e => {{
        dragged = t; moved = false;
        t.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        // Firefox starts no drag at all unless some data is set.
        try {{ e.dataTransfer.setData('text/plain', t.dataset.group); }} catch (_) {{}}
      }});
      t.addEventListener('dragend', () => {{
        t.classList.remove('dragging');
        // A click fires after a drag that ended where it started; suppressing
        // it unconditionally would break plain tab switching, so only guard
        // when the tab actually moved.
        if (moved) {{ t.dataset.dragged = '1'; saveOrder(); }}
        dragged = null;
      }});
      t.addEventListener('dragover', e => {{
        if (!dragged || dragged === t) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = t.getBoundingClientRect();
        const before = (e.clientX - r.left) < r.width / 2;
        bar.insertBefore(dragged, before ? t : t.nextSibling);
        moved = true;
      }});
      t.addEventListener('drop', e => e.preventDefault());
      t.addEventListener('click', e => {{
        if (t.dataset.dragged) {{ e.preventDefault(); delete t.dataset.dragged; }}
      }});
    }});
    function saveOrder() {{
      document.getElementById('grouporder').value =
        tabs().map(t => t.dataset.group).join('\\n');
      document.getElementById('grouporderform').submit();
    }}
  }}
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
    # Not red. Nothing is broken — the PR is simply not being tested yet, and
    # colouring that like a failure sends you chasing a CI problem that the
    # author (mark Ready) or a label is the only fix for.
    "draft": "dim",
    "gated": "warn",
}

# The whole point of the Verdict column: say what to DO, not just what happened.
# This is the "just re-run it" vs "stuck, needs a merge" split.
# Actions that mean "no NVIDIA failure to act on". Their hint and stored
# verdict are both just long ways of saying so, and the Action column already
# shows `-`, so Status stays blank instead of repeating it on every clean row.
QUIET_ACTIONS = {"green", "out-of-scope"}
# A draft shows its one-line reason and nothing else: no tally, no failure
# block, no history of red gates that were never a verdict in the first place.
SILENT_ACTIONS = {"draft"}

ACTION_HINT = {
    "green": "nothing to do",
    "awaiting-triage": "needs /ci-analysis — run `/pr-ci-watch triage <pr>`",
    "re-run": "re-ran; waiting on CI",
    "code-fix": "real bug in this PR — author must fix",
    "merge-main": "STUCK: PR is behind main — merge/rebase main",
    "conflict": "git conflict with main — author must resolve; /pr-conflict-fix",
    "wait-upstream": "STUCK: an NVIDIA job is blocked on an upstream fix",
    "out-of-scope": "nothing to do — red is outside NVIDIA scope",
    "draft": "draft — CI and code not checked until marked Ready for review",
    "gated": "CI never started — the gate rejected it before any test ran",
}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def merge_cell(s: dict) -> str:
    # Draft wins over the merge state. "clean" on a draft invites you to read
    # the row as ready-to-land, and this column is the only place on the row
    # that still has something true to say about a PR we do not check.
    if s.get("is_draft"):
        return '<span class="pill draft">draft</span>'
    # `clean` is true (no git conflict) but on its own it reads as ready to
    # land, which is wrong when the required CI never started. Shown only for a
    # genuinely blocking gate, so ordinary rows keep the clean/conflict split
    # this column exists for rather than every row turning amber.
    if any(d.get("blocking", d.get("reason") in GATE_BLOCKING)
           for d in (s.get("gated") or {}).values()):
        return ('<span class="pill warn" title="no git conflict, but required '
                'CI never ran">gate blocked</span>')
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


def last_failure_block(s: dict, prefix: str = "Last failure") -> str:
    """What actually broke, from the most recent sweep that saw a real failure.

    Live checks go back to "only aggregation gates are red" as soon as a re-run
    starts, and `gate only, not re-runnable` says nothing about what broke. This
    keeps the useful part on screen.
    """
    lf = s.get("last_real_failure") or {}
    if not lf or lf.get("sha") != s.get("head_sha"):
        return ""  # a new push invalidates it
    rows = []
    for wf, g in (lf.get("groups") or {}).items():
        names = (g.get("jobs") or []) + (g.get("watcher_jobs") or [])
        if names:
            rows.append(f'<div>{esc(wf)}: {job_links(g, names)}</div>')
    if not rows:
        return ""
    return (f'<div class="dim" style="margin-top:4px;"><b>{prefix}</b> '
            f'{esc(tw(lf.get("at")))}:</div>' + "".join(rows))


def failure_block(s: dict) -> str:
    """The failing NVIDIA jobs, and nothing else.

    This used to be its own `Red NVIDIA CI` column, which spent most of its
    width telling you that nothing was wrong — "no NVIDIA job is failing; the
    red is aggregation gates…" on every clean row. An empty cell says that
    already. What is left is the part that only exists when something is
    actually broken, so it lives in Status with the rest of the evidence.
    """
    groups = s.get("failed_groups") or {}
    if not groups or s.get("last_action") == "green":
        return ""
    out = []
    # After a re-run the stored failure list describes the run we *replaced*.
    # Showing it as if it were current is what made this unreadable.
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
            continue
        if g.get("repeat_after_rerun"):
            out.append(
                f'<div class="bad"><b>{esc(wf)}: SAME failure after '
                f'{g["repeat_after_rerun"]} re-run(s)</b> &mdash; re-running '
                f"again will not help; evaluate <b>Merge main</b>.</div>"
            )
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


VERDICT_CLASS = {"Pass": "ok", "Pending": "warn", "Fail": "bad", "—": "dim"}
ACTION_COLOR = {
    "Solve conflict": "bad",
    "Merge main": "bad",
    "Code fix": "bad",
    "CI re-run": "warn",
    "Triage": "warn",
    "Wait upstream": "warn",
    "Need run-ci tag": "bad",
    "-": "dim",
}


def verdict_cell(s: dict) -> str:
    """Just the state word. The job counts live in Status."""
    v = ci_verdict(s)
    return f'<span class="pill {VERDICT_CLASS.get(v, "dim")}">{esc(v)}</span>'


def tally_line(s: dict) -> str:
    bits = tally_bits(s.get("tally"))
    if not bits:
        return ""
    # Say so when these counts predate the re-run, instead of quietly showing
    # the run we already replaced.
    sha = s.get("head_sha", "")
    last_rerun = max((r.get("at", "") for r in (s.get("reruns") or {}).values()
                      if r.get("sha") == sha), default="")
    if last_rerun and last_rerun > (s.get("last_sweep") or ""):
        return (f'<div class="warn"><b>{esc(bits)}</b> — counts from '
                f'<i>before</i> the {esc(tw(last_rerun))} re-run; hit '
                f"<b>Refresh now</b></div>")
    return (f'<div><b>{esc(bits)}</b>'
            f'<span class="dim"> @ {esc(tw(s.get("last_sweep")))}</span></div>')


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
    if a == "CI re-run" and not (s.get("reruns") or {}):
        # Two very different states used to render identically. Only say GitHub
        # refused when a POST was actually made and rejected — otherwise this
        # is a decision the sweep recorded without `--apply`, and claiming an
        # attempt that never happened is worse than saying nothing.
        deferred = [r for r in (s.get("rerun_deferred") or {}).values()
                    if r.get("sha") == sha]
        if deferred:
            out.append('<div class="dim">queued &mdash; '
                       f'{esc(deferred[0].get("why", "re-run refused"))}</div>')
        else:
            out.append('<div class="dim">decided, not yet applied &mdash; the '
                       "next <code>--apply</code> sweep performs it</div>")
    return "".join(out)


def notify_cell(s: dict) -> str:
    """Conflicts are the one thing this tool says to a third party — so show
    proof it happened, or say plainly that it has not."""
    if s.get("mergeable") != "CONFLICTING":
        return ""
    if s.get("conflict_comment_sha") != s.get("head_sha"):
        return ('<div class="bad"><b>NOT notified yet</b> — the author has not '
                "been told about this conflict.</div>")  # button added by caller
    url = s.get("conflict_comment_url", "")
    link = f' &middot; <a href="{esc(url)}" target="_blank">see comment</a>' if url else ""
    return (f'<div class="ok"><b>Notified</b> @{esc(s.get("author"))} '
            f"{esc(tw(s.get('conflict_comment_at')))}{link}</div>")


def status_cell(pr: str, s: dict) -> str:
    """Where CI actually stands: counts, then what broke, then why.

    Says nothing when nothing is wrong. `green` and `out-of-scope` both mean
    "no NVIDIA failure to act on", and their stored hint and verdict only
    restate that at length — on a Pass row the whole cell is the job counts.
    """
    action = s.get("last_action", "")
    quiet = action in QUIET_ACTIONS
    if action in SILENT_ACTIONS:
        return (
            f'<div class="dim">{esc(ACTION_HINT.get(action, action))}</div>'
            + history_toggle(pr)
        )
    out = []
    g = gate_block(s)
    if g:
        out.append(g)
    b = notify_button(pr, s)
    if b:
        out.append(b)
    t = tally_line(s)
    if t:
        out.append(t)
    n = notify_cell(s)
    if n:
        out.append(n)
    live = failure_block(s)
    out.append(live)
    # Only as a fallback. What broke last is worth keeping on screen while a
    # re-run is in flight — live checks drop back to "only gates are red" the
    # moment one starts — but printing it beside an identical live list just
    # says the same thing twice. Once CI is green it is history, not status.
    if not live and ci_verdict(s) != "Pass":
        out.append(last_failure_block(s))
    if not quiet:
        hint = ACTION_HINT.get(action, "")
        if hint:
            out.append(f"<div>{esc(hint)}</div>")
        # Wrap, never truncate: cutting the reason mid-word ("…OOM on a 32GB
        # GPU; un") is worse than showing no reason at all.
        if s.get("last_verdict"):
            out.append(f'<div class="dim reason">{esc(s["last_verdict"])}</div>')
    # The toggle sits in Status because history is this column's long form —
    # how the state got here. The panel it opens is a full-width row below,
    # where the long verdict summaries have room to read.
    out.append(history_toggle(pr))
    return "".join(out)


def history_toggle(pr: str) -> str:
    return (f'<button class="disc" type="button" data-pr="{esc(pr)}" '
            f'aria-expanded="false" title="show this PR\'s history">'
            f'<span class="tri">&#9656;</span> history</button>')


# How a blocked gate reads, per reason. The point of naming each one is that
# the fix differs: a label you can add from here, a Ready-for-review click only
# the author can make, a cooldown that a write-access re-run simply ignores.
GATE_LABEL = {
    "draft": "was a draft when CI ran",
    "stale-draft": "CI ran while this was a draft; it is Ready now",
    "missing-run-ci": "missing the <code>run-ci</code> label",
    "missing-label": "missing a workflow opt-in label",
    "rate-limit": "author rate-limited (low-permission cooldown)",
    # The common one, and the reason the dashboard now shows non-blocking gates
    # at all: most PRs never opt into PR Test Extra, so its red gate + rollup
    # are a permanent, meaningless pair of fails on nearly every row.
    "opt-in-extra": "<code>PR Test Extra</code> is opt-in and this PR did not opt in",
}


def pick_track(value: str | None) -> str:
    """Form values come off the wire; an unknown track would silently exclude
    the PR from every sweep (`--track` filters on exact match)."""
    return value if value in TRACKS else "regular"


def gate_block(s: dict) -> str:
    """Account for the fail count: which failures are a gate, and why.

    Without this a gated PR renders as a plain fail count — the shape that made
    "4 pass, 4 fail" look like a test problem on a PR whose tests never ran, and
    "59 pass, 2 fail" look like two broken tests when both reds were the
    not-opted-in Extra workflow. Non-blocking reasons are shown too, dimmed:
    the whole point is that no number on this row is unexplained.
    """
    gated = s.get("gated") or {}
    if not gated:
        return ""
    reasons = {}
    for wf, d in gated.items():
        reasons.setdefault(d.get("reason", "?"), []).append(wf)
    bits = []
    for reason, wfs in sorted(reasons.items(),
                              key=lambda kv: not gated[kv[1][0]].get(
                                  "blocking", kv[0] in GATE_BLOCKING)):
        d = gated[wfs[0]]
        label = GATE_LABEL.get(reason, esc(reason))
        # How many checks this accounts for, so the reader can subtract it from
        # the fail count rather than wondering which reds are covered.
        n = sum(len(gated[w].get("jobs") or []) for w in wfs)
        count = f' <span class="dim">({n} of the fails)</span>' if n else ""
        # State written before `blocking` existed has no such key; deriving it
        # from the reason stops a pre-upgrade row from rendering a real block
        # as "not a failure" for one sweep.
        blocking = d.get("blocking", reason in GATE_BLOCKING)
        if not blocking:
            bits.append(f'<div class="dim">not a failure &middot; {label}{count}</div>')
            continue
        fix = " &mdash; a re-run from here clears it" if d.get("rerunnable") else ""
        bits.append(
            f'<div class="bad"><b>CI never started</b> &middot; {label}{fix}{count}'
            f'<div class="dim">{esc(", ".join(sorted(wfs)))}</div></div>'
        )
    return "".join(bits)


def track_cell(pr: str, track: str) -> str:
    """A dropdown, not a toggle button: you asked to set a track explicitly
    after tagging, and a one-way flip makes the current value ambiguous."""
    opts = "".join(
        f'<option value="{t}"{" selected" if track == t else ""}>{t}</option>'
        for t in TRACKS
    )
    return (f'<form class="inline" method="post" action="/api/track">'
            f'<input type="hidden" name="pr" value="{esc(pr)}">'
            f'<select class="mini" name="track" onchange="this.form.submit()">{opts}</select>'
            f"</form>")


def grip_cell() -> str:
    """The drag handle. A handle rather than a draggable row: making the whole
    row draggable costs you text selection and turns every link into a drag."""
    return ('<span class="grip" title="drag to reorder" '
            'aria-label="drag to reorder">&#x283F;</span>')


def notify_button(pr: str, s: dict) -> str:
    """Only offered when there is actually an unsent notice to send."""
    if s.get("mergeable") != "CONFLICTING":
        return ""
    if s.get("conflict_comment_sha") == s.get("head_sha"):
        return ""
    return (
        f'<form class="inline" method="post" action="/api/notify">'
        f'<input type="hidden" name="pr" value="{esc(pr)}">'
        f'<button class="primary" type="submit" '
        f'onclick="return confirm(\'Post the conflict notice on PR {esc(pr)}? '
        f'This comments on the author\\\'s PR.\')">Notify author</button>'
        f"</form>"
    )


def prio_cell(pr: str, meta: dict) -> str:
    opts = "".join(
        f'<option value="{p}"{" selected" if priority_of(meta) == p else ""}>{p}</option>'
        for p in PRIORITIES
    )
    return (
        f'<form class="inline" method="post" action="/api/priority">'
        f'<input type="hidden" name="pr" value="{esc(pr)}">'
        f'<select class="mini" name="priority" onchange="this.form.submit()">{opts}</select></form>'
    )


def report_html(wl: dict, st: dict, group: str | None = None) -> str:
    """Same block, but <PRnnnnn> is a real link. Copying this as rich text keeps
    the hyperlink when it lands in Teams; the plain-text flavour is copied
    alongside for anywhere that strips HTML."""
    out = []
    for e in report_entries(wl, st, group):
        # A real nested <ul> so a rich paste lands in Teams as a proper list.
        note = (f'<ul><li>{esc(e["note"])}</li></ul>' if e["note"] else "")
        out.append(
            f'<li>&lt;{esc(e["pri"])}&gt;&lt;{esc(e["token"])}&gt;'
            f'&lt;<a href="{esc(e["url"])}" target="_blank">PR{esc(e["pr"])}</a>&gt;'
            f'{esc(e["title"])}{note}</li>'
        )
    return (f"<ul>{''.join(out)}</ul>" if out
            else '<div class="dim">(watchlist is empty)</div>')


def bucket_of(st: dict, pr: str, meta: dict) -> str:
    """The part of the sort a drag cannot cross.

    Manual order is only a tiebreaker: Pass-first, priority and conflict-last
    still decide the buckets, so a row dropped into another bucket would
    silently snap back. The row carries this as `data-bucket` and the drag
    refuses a drop when the two differ — a rejected drop is honest, a drop that
    springs back reads as a bug.
    """
    return "|".join(str(x) for x in row_order(st)((pr, meta))[:3])


def reorder_rows(wl: dict, st: dict, prs: list[str]) -> None:
    """Apply a dragged sequence, permuting only the rows that were dragged.

    The table may be filtered to one tab, so the submitted list is a subset.
    Re-ranking it densely from zero would shove every hidden row to the end;
    instead the dragged PRs are re-seated into the set of slots they already
    occupied, which leaves everything else exactly where it was.
    """
    prs = [p for p in prs if p in wl]
    if not prs:
        return
    # Normalise to dense ranks first; stored orders may all be 0 initially.
    for rank, (_, meta) in enumerate(sorted(wl.items(), key=row_order(st))):
        meta["order"] = rank
    slots = sorted(wl[p]["order"] for p in prs)
    for p, slot in zip(prs, slots):
        wl[p]["order"] = slot


def render_tabs(wl: dict, active: str | None, order: list[str] | None = None) -> str:
    groups = all_groups(wl, order)
    if not groups or groups == [UNGROUPED]:
        return ""  # one bucket is not a tab bar

    def tab(label, key, n, drag=False):
        on = " on" if (key or None) == (active or None) else ""
        href = f"/?tab={quote(key)}" if key else "/"
        # `All` and `Ungrouped` are fixed ends of the bar, so they are not
        # draggable — and because nothing listens for a drag over them, a tab
        # cannot be dropped outside the named range either.
        d = f' draggable="true" data-group="{esc(key)}"' if drag else ""
        return (f'<a class="tab{on}"{d} href="{esc(href)}">{esc(label)}'
                f'<span class="dim"> {n}</span></a>')

    count = lambda g: sum(1 for m in wl.values() if group_of(m) == g)
    out = [tab("All", "", len(wl))]
    named = [g for g in groups if g != UNGROUPED]
    out += [tab(g, g, count(g), drag=True) for g in named]
    if UNGROUPED in groups:
        out.append(tab(UNGROUPED, UNGROUPED, count(UNGROUPED)))
    return (
        f'<div class="tabs" id="tabbar">{"".join(out)}</div>'
        f'<form id="grouporderform" method="post" action="/api/group-order" '
        f'style="display:none">'
        f'<input type="hidden" id="grouporder" name="order" value="">'
        f'<input type="hidden" name="tab" value="{esc(active or "")}"></form>'
    )


def group_cell(pr: str, meta: dict) -> str:
    """Plain free text — no datalist. The dropdown marker it adds costs real
    width in a column this narrow, and typing the name is the only thing it
    ever did: an unknown name creates a tab either way."""
    cur = meta.get("group", "")
    return (
        f'<form method="post" action="/api/group">'
        f'<input type="hidden" name="pr" value="{esc(pr)}">'
        f'<input class="note grp" type="text" name="group" value="{esc(cur)}" '
        f'placeholder="{esc(UNGROUPED)}"></form>'
    )


def render_table(wl: dict, st: dict, tab: str = "") -> str:
    if not wl:
        return EMPTY
    # Widths, not guesses: with table-layout:fixed these are what the browser
    # uses, so the two prose columns keep their share no matter how long a
    # group name someone types.
    # All percentages, summing to 100. Mixing px and % here is a trap: when the
    # declared widths exceed the table width the browser rescales *everything*
    # proportionally, so the px columns get squeezed too and the knobs overflow
    # again. The floor is held by `table { min-width }` instead.
    widths = (
        3,    # grip   — leftmost, where a drag handle is looked for
        6,    # Pri    — a <select>; see note below
        18,   # PR + title + author
        7,    # Group
        7,    # Track  — a <select>
        7,    # Merge  — fits the `conflict` / `unknown` pill unwrapped
        6,    # Verdict
        9,    # Action
        29,   # Status — all the prose now lives here
        5,    # Swept  — wraps to date / time, both halves unbroken
        3,    # remove — a glyph, not the word; see below
    )
    cols = "".join(f'<col style="width:{w}%">' for w in widths)
    # The history panel spans the table, so its colspan is derived rather than
    # written down twice — a stale literal here silently misaligns every row.
    span = len(widths)
    head = (
        "<tr><th></th><th>Pri</th><th>PR</th><th>Group</th><th>Track</th><th>Merge</th>"
        "<th>Verdict</th><th>Action</th><th>Status</th>"
        "<th>Swept (TW)</th><th></th></tr>"
    )
    rows = []
    for pr, meta in sorted(wl.items(), key=row_order(st)):
        s = st.get(pr, {})
        track = meta.get("track", "regular")
        other = "regular" if track == "high" else "high"
        url = s.get("url") or f"https://github.com/{meta.get('repo', '')}/pull/{pr}"
        rows.append(
            f'<tr data-pr="{esc(pr)}" data-bucket="{esc(bucket_of(st, pr, meta))}">'
            f'<td class="knob">{grip_cell()}</td>'
            f'<td class="knob">{prio_cell(pr, meta)}</td>'
            f'<td><a href="{esc(url)}" target="_blank"><b>#{esc(pr)}</b></a>'
            f'<span class="title dim" title="{esc(s.get("title"))}">{esc(s.get("title"))}</span>'
            f'<span class="dim">{"@" + esc(s.get("author")) if s.get("author") else ""}</span></td>'
            f'<td class="knob">{group_cell(pr, meta)}</td>'
            f'<td class="knob">{track_cell(pr, track)}</td>'
            f"<td>{merge_cell(s)}</td>"
            f"<td>{verdict_cell(s)}</td>"
            f"<td>{action_cell(s)}</td>"
            f"<td>{status_cell(pr, s)}</td>"
            f'<td class="mono dim">{esc(tw(s.get("last_sweep")))}</td>'
            # A glyph, not the word "remove": a <button> cannot wrap or shrink,
            # so a 6-character label in a 3% column overhangs into nothing.
            f'<td class="knob"><form class="inline" method="post" action="/api/remove">'
            f'<input type="hidden" name="pr" value="{esc(pr)}">'
            f'<button class="linkish" title="stop watching #{esc(pr)}" '
            f'aria-label="stop watching #{esc(pr)}">&times;</button>'
            f"</form></td></tr>"
            # Kept immediately after its own row, and the drag moves the pair
            # together — a history panel stranded under someone else's row is
            # worse than no panel.
            f'<tr class="hist" data-for="{esc(pr)}" hidden>'
            f'<td colspan="{span}"><div class="histbox">loading…</div></td></tr>'
        )
    return (
        f'<table id="prtable"><colgroup>{cols}</colgroup><thead>{head}</thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        f'<form id="roworderform" method="post" action="/api/order" '
        f'style="display:none">'
        f'<input type="hidden" id="roworder" name="order" value="">'
        f'<input type="hidden" name="tab" value="{esc(tab or "")}"></form>'
    )


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


def render_notice() -> str:
    if not NOTICE["at"]:
        return ""
    cls = "ok" if NOTICE["ok"] else "bad"
    label = "Sent" if NOTICE["ok"] else "Not sent"
    body = esc(NOTICE["text"])
    # Linkify the comment URL the command prints on success.
    body = re.sub(r"(https://\S+)", r'<a href="\1" target="_blank">\1</a>', body)
    return (f'<div class="banner {cls}"><b>{label}</b> &middot; '
            f'{esc(tw(NOTICE["at"]))}<br>'
            f'<span class="mono">{body}</span></div>')


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
        f"page nor <i>Refresh now</i> can do it.<br>"
        # The old wording said "needs a Claude turn" and "the next sweep picks
        # them up automatically" in one breath, which reads as a contradiction
        # unless you already know the schedule IS a Claude turn. Say so.
        f"<b>Yes, it does this by itself.</b> The scheduled sweep is a Claude "
        f"cron job, not a headless script, so it runs <code>/ci-analysis</code> "
        f"on everything in this panel in the same turn. Two caveats: it only "
        f"fires while the session is <b>idle</b>, and the cron jobs "
        f"<b>expire after 7 days</b> (see the arm line at the top). "
        f"To do it now instead, paste this to Claude:</div>"
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
        if self.path.startswith("/api/history"):
            q = parse_qs(urlparse(self.path).query).get("pr") or [""]
            try:
                pr = parse_pr(q[0])
            except SystemExit:
                self._send(400, b"bad pr", "text/plain; charset=utf-8")
                return
            self._send(200, history_html(pr).encode(),
                       "text/html; charset=utf-8")
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
        gorder = (raw_st.get("_config") or {}).get("group_order") or []
        tab = (parse_qs(urlparse(self.path).query).get("tab") or [""])[0]
        # A tab that no longer exists (last PR moved out) falls back to All
        # rather than showing an empty table with no way back.
        if tab and tab not in all_groups(wl, gorder):
            tab = ""
        shown = {k: v for k, v in wl.items() if in_group(v, tab or None)}
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
            triage_panel=render_triage_panel(shown, st),
            repo=html.escape(self.repo),
            generated=now(),
            data=html.escape(str(DATA_DIR)),
            table=render_table(shown, st, tab),
            tabs=render_tabs(wl, tab, gorder),
            report=html.escape(report_text(wl, st, tab or None)),
            report_html=report_html(wl, st, tab or None),
            banner=render_notice() + render_banner(raw_st, enabled),
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
            if self.path == "/api/notify":
                pr = parse_pr(form.get("pr", ""))
                # The one outward-facing button on this page, so it runs the
                # same guarded code path as the sweep (one comment per head
                # SHA) rather than posting anything itself.
                here = Path(__file__).resolve().parent
                p = subprocess.run(
                    [sys.executable, str(here / "watch.py"), "notify",
                     "--pr", pr, "--apply"],
                    capture_output=True, text=True, timeout=180, cwd=str(here),
                )
                msg = (p.stdout or "").strip() or (p.stderr or "").strip()
                NOTICE.update(at=now(), ok=(p.returncode == 0),
                              text=msg or f"notify #{pr}: no output")
                # The refusal reasons are all "state moved on" (conflict already
                # resolved, already notified), so re-read this PR before the
                # page redraws or the button lingers on stale state.
                subprocess.run(
                    [sys.executable, str(here / "watch.py"), "sweep",
                     "--pr", pr],
                    capture_output=True, text=True, timeout=300, cwd=str(here),
                )
            elif self.path == "/api/refresh":
                with SWEEP_LOCK:
                    if not SWEEP["running"]:
                        SWEEP.update(running=True, started=now(), finished="",
                                     output="", rc=None)
                        threading.Thread(target=run_refresh, daemon=True).start()
            elif self.path == "/api/monitoring":
                st = load(STATE, {})
                set_monitoring(st, form.get("on") == "1")
                save(STATE, st)
            elif self.path == "/api/group-order":
                # Tab order is a view preference, so it lives beside the other
                # dashboard config rather than in the watchlist. Stored as the
                # names the user dragged; all_groups() reconciles it with the
                # groups that actually exist on every render.
                st = load(STATE, {})
                names = [g.strip() for g in form.get("order", "").split("\n")]
                seen, clean = set(), []
                for g in names:
                    if g and g != UNGROUPED and g not in seen:
                        seen.add(g)
                        clean.append(g)
                st.setdefault("_config", {})["group_order"] = clean
                save(STATE, st)
            else:
                wl = load(WATCHLIST, {})
                if self.path == "/api/add":
                    pr = parse_pr(form.get("ref", ""))
                    prev = wl.get(pr, {})
                    wl[pr] = {
                        "track": pick_track(form.get("track")),
                        "added": prev.get("added", now()),
                        "note": prev.get("note", ""),
                        "repo": self.repo,
                    }
                    if form.get("priority") in PRIORITIES:
                        wl[pr]["priority"] = form["priority"]
                    elif prev.get("priority"):
                        wl[pr]["priority"] = prev["priority"]
                    log_line(f"#{pr} added via dashboard "
                             f"(track={wl[pr]['track']})")
                elif self.path == "/api/track":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl:
                        wl[pr]["track"] = pick_track(form.get("track"))
                        # Moving off `draft` by hand must not leave the restore
                        # target behind: the next sweep would read a stale
                        # prev_track and bounce the PR back to it.
                        if wl[pr]["track"] != "draft":
                            wl[pr].pop("prev_track", None)
                elif self.path == "/api/priority":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl and form.get("priority") in PRIORITIES:
                        wl[pr]["priority"] = form["priority"]
                elif self.path == "/api/note":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl:
                        wl[pr]["note"] = form.get("note", "").strip()
                elif self.path == "/api/group":
                    pr = parse_pr(form.get("pr", ""))
                    if pr in wl:
                        wl[pr]["group"] = form.get("group", "").strip()
                elif self.path == "/api/order":
                    reorder_rows(
                        wl, load(STATE, {}),
                        [p.strip() for p in form.get("order", "").split(",")
                         if p.strip()],
                    )
                elif self.path == "/api/remove":
                    pr = parse_pr(form.get("pr", ""))
                    if wl.pop(pr, None) is not None:
                        # Audit it: an open PR vanishing from the list with no
                        # trace is indistinguishable from a bug.
                        log_line(f"#{pr} removed from watchlist via dashboard")
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
        # Only the tab-order form carries `tab`; everything else keeps landing
        # on All, which is where it landed before.
        back = form.get("tab", "")
        self.send_header("Location", f"/?tab={quote(back)}" if back else "/")
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
