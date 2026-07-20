#!/usr/bin/env python3
"""abox-snapshot — build a SELF-CONTAINED dashboard of live sessions + transcripts.

No server, no ports: every session and its turns are embedded into one .html file that
works by double-clicking it. Built for the case where the only channel to this box is
the Claude conversation itself (port forwarding unavailable), so the file is handed
over through that channel instead of being served.

  sudo -n python3 abox_snapshot.py [-o out.html] [--all] [--turns N] [--recent MIN]
"""
import os, re, sys, json, glob, time, pwd, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL

MAX_TURNS = 120
MAX_BLOCK = 4000
MAX_SESSION_BYTES = 400_000

# --- Remote dashboard: the ONE published artifact ---------------------------
# `--publish` regenerates the snapshot at this stable path and emits the exact
# Artifact-tool call for the agent to run, so every conversation refreshes the
# SAME artifact URL (bookmarked on the user's phone) instead of minting a new one.
# The agent cannot publish from a shell — only the Artifact tool can — so the
# script's job is to (re)build the file and hand the agent the parameters.
PUBLISH_OWNER = "yichiche"
STABLE_OUT = "/home/yichiche/agent-scratch/abox-dash/abox-dashboard.html"
ARTIFACT_URL = "https://claude.ai/code/artifact/605cb93f-db89-4777-bbfb-6a4a9f50276c"
ARTIFACT_TITLE = "abox dashboard"          # must match <title>; renames artifact if changed
ARTIFACT_FAVICON = "🛰️"
ARTIFACT_DESC = "Live status of Claude agents running in my containers"


def all_roots():
    """[(projects_dir, container_label)] — bind-mounted ~/.claude/projects."""
    return [(AL.SHARED_PROJ, "shared")]


def parse_turns(path, max_turns):
    out = []
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except Exception as e:
        return [{"role": "error", "ts": "", "blocks": [{"kind": "text",
                 "text": "cannot read transcript: %s" % e}]}]
    for ln in raw.splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("type") not in ("assistant", "user"):
            continue
        m = o.get("message", {})
        c = m.get("content") if isinstance(m, dict) else None
        blocks = []
        if isinstance(c, str):
            blocks.append({"kind": "text", "text": c[:MAX_BLOCK]})
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                k = p.get("type")
                if k == "text":
                    blocks.append({"kind": "text", "text": p.get("text", "")[:MAX_BLOCK]})
                elif k == "thinking":
                    blocks.append({"kind": "think", "text": p.get("thinking", "")[:MAX_BLOCK]})
                elif k == "tool_use":
                    blocks.append({"kind": "tool", "text": p.get("name", ""),
                                   "extra": json.dumps(p.get("input", {}))[:1200]})
                elif k == "tool_result":
                    t = p.get("content")
                    if isinstance(t, list):
                        t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                    blocks.append({"kind": "result", "text": str(t)[:MAX_BLOCK]})
        if blocks:
            out.append({"role": o.get("type"), "ts": (o.get("timestamp") or "")[:19],
                        "blocks": blocks})
    out = out[-max_turns:]
    # keep any single session from dominating the file
    while len(json.dumps(out)) > MAX_SESSION_BYTES and len(out) > 8:
        out = out[len(out) // 4:]
    return out


def build(show_all, max_turns, recent_min):
    rows = AL.collect()
    if not show_all:
        rows = [r for r in rows if r["mine"]]
    seen = {r["path"] for r in rows if r["path"]}
    data = []
    for r in rows:
        data.append({
            "session": r["session"], "container": r["container"], "cwd": r["cwd"],
            "evidence": r["evidence"], "age": AL.age(r["idle"]), "idle": r["idle"],
            "pid": r["pid"], "live": True,
            "state": "live" if 0 <= r["idle"] < 120 else ("idle" if r["idle"] >= 0 else "unknown"),
            "turns": parse_turns(r["path"], max_turns) if r["path"] else [],
        })

    # optionally fold in recently-written transcripts whose process already exited
    if recent_min:
        cut = time.time() - recent_min * 60
        for root, label in all_roots():
            for f in glob.glob(os.path.join(root, "*", "*.jsonl")):
                if f in seen:
                    continue
                try:
                    mt = os.path.getmtime(f)
                except OSError:
                    continue
                if mt < cut:
                    continue
                idle = time.time() - mt
                data.append({
                    "session": os.path.basename(f)[:-6],
                    "container": label if label != "shared" else "(ended — container unknown)",
                    "cwd": os.path.basename(os.path.dirname(f)).replace("-", "/"),
                    "evidence": "ended", "age": AL.age(idle), "idle": idle,
                    "pid": "", "live": False, "state": "ended",
                    "turns": parse_turns(f, max_turns),
                })

    data.sort(key=lambda s: (not s["live"], s["container"] == "host",
                             s["container"], s["idle"]))
    return data


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e0e0e0;--card:#f7f7f8;--acc:#0b6bcb}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c3038;--card:#1d2026;--acc:#5aa9ff}}
*{box-sizing:border-box}
body{margin:0;padding:18px;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;margin-bottom:26px;font-size:12px;display:block;overflow-x:auto}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td.wrap{white-space:normal}
a{color:var(--acc)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.live{background:#2ea043}.idle{background:#9aa0a6}.ended{background:#5b5b5b}.unknown{background:#d29922}
details.s{border:1px solid var(--line);border-radius:8px;margin-bottom:12px;background:var(--card)}
details.s>summary{cursor:pointer;padding:10px 12px;font-size:13px;list-style:none}
details.s>summary::-webkit-details-marker{display:none}
details.s>summary:before{content:"\\25b6";color:var(--mut);margin-right:8px;font-size:10px}
details.s[open]>summary:before{content:"\\25bc"}
.body{padding:4px 14px 14px;background:var(--bg);border-top:1px solid var(--line)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meta{color:var(--mut);font-size:11px;margin-left:6px}
.t{margin:12px 0;border-left:3px solid var(--line);padding-left:10px}
.t.assistant{border-color:var(--acc)}
.role{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
pre{white-space:pre-wrap;word-wrap:break-word;margin:4px 0;font:12px/1.5 ui-monospace,Menlo,monospace;background:var(--card);padding:8px;border-radius:6px;max-height:320px;overflow:auto}
.tool{color:var(--acc);font-weight:600;font-size:12px;margin-top:6px}
.think{opacity:.6;font-style:italic}
.note{color:var(--mut);font-size:12px;border-left:3px solid var(--line);padding-left:10px;margin:18px 0}
.work{color:var(--mut);font-size:11px;margin:8px 0 8px 4px}
.cmd{font-size:11px;color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:10px 0;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
.cmd code{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--fg);user-select:all}
.cpy{font:inherit;font-size:11px;padding:6px 11px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--acc);cursor:pointer;min-height:32px}
.cpy:active{opacity:.6}
.t.user{border-color:#d29922}
.t>div:last-child{white-space:pre-wrap;word-wrap:break-word}
.top{position:sticky;bottom:10px;float:right;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 9px;text-decoration:none}
/* phone: the summary table is the least useful part on a small screen, and the
   details rows must stay comfortably tappable */
@media(max-width:640px){
  body{padding:12px}
  h1{font-size:16px}
  table{font-size:11px}
  td,th{padding:5px 7px}
  details.s>summary{padding:13px 12px;font-size:14px}
  pre{max-height:220px;font-size:11px}
  .hide-sm{display:none}
}
"""


def esc(s):
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def condense(turns):
    """Strip the agent's internal chatter.

    A raw transcript is mostly tool_use/tool_result/thinking — the agent talking to
    itself. What tracks PROGRESS is the assistant's prose and the human's instructions.
    Everything else collapses into one muted 'N tool calls' line so the work is still
    visible without drowning the narrative.
    """
    out, pend = [], {}

    def flush():
        if pend:
            n = sum(pend.values())
            top = sorted(pend.items(), key=lambda kv: -kv[1])[:4]
            out.append({"kind": "work", "n": n,
                        "names": " ".join("%s×%d" % (k or "?", v) for k, v in top)})
            pend.clear()

    for t in turns:
        said = " ".join(b.get("text", "") for b in t["blocks"]
                        if b.get("kind") == "text").strip()
        if said:
            flush()
            out.append({"kind": "say" if t["role"] == "assistant" else "ask",
                        "ts": t["ts"], "text": said})
            continue
        for b in t["blocks"]:
            if b.get("kind") == "tool":
                pend[b.get("text")] = pend.get(b.get("text"), 0) + 1
    flush()
    return out


def render_item(it):
    if it["kind"] == "work":
        return ('<div class="work">&#8943; %d tool calls <span class="meta">%s</span>'
                "</div>" % (it["n"], esc(it["names"])))
    who = "agent" if it["kind"] == "say" else "instruction"
    return ('<div class="t %s"><div class="role">%s %s</div><div>%s</div></div>'
            % ("assistant" if it["kind"] == "say" else "user", who,
               esc(it["ts"][11:19]), esc(it["text"]).replace("\n", "<br>")))


def render_turn(t):
    out = ['<div class="t %s"><div class="role">%s %s</div>'
           % (esc(t["role"]), esc(t["role"]), esc(t["ts"][11:19]))]
    for b in t["blocks"]:
        k = b.get("kind")
        if k == "tool":
            out.append('<div class="tool">&#9656; %s</div>' % esc(b.get("text")))
            if b.get("extra"):
                out.append("<pre>%s</pre>" % esc(b["extra"]))
        elif k == "result":
            out.append("<pre>%s</pre>" % esc(b.get("text")))
        elif k == "think":
            out.append('<pre class="think">%s</pre>' % esc(b.get("text")))
        else:
            out.append("<div>%s</div>" % esc(b.get("text")).replace("\n", "<br>"))
    out.append("</div>")
    return "".join(out)


def render_html(data, when, raw=False):
    """No JavaScript at all: sandboxed viewers (and strict CSP) blank a JS page, and
    this file has to survive being opened anywhere. <details> gives the click-to-open
    behaviour natively."""
    p = ['<!doctype html><meta charset="utf-8">',
         # Stable title on purpose: this is republished to the same artifact URL on a
         # schedule, and a timestamped title would rename the artifact every run.
         "<title>abox dashboard</title>",
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<style>%s</style>" % CSS,
         "<h1>abox dashboard</h1>",
         '<div class="sub">snapshot %s &middot; %d sessions &middot; %d live &middot; '
         "%d containers &middot; static file, does not auto-refresh</div>"
         % (esc(when), len(data), sum(1 for s in data if s["live"]),
            len({s["container"] for s in data}))]

    p.append('<table id="top"><tr><th></th><th>container</th><th>session</th>'
             "<th>age</th><th>turns</th><th class=\"hide-sm\">ev</th>"
             "<th class=\"hide-sm\">cwd</th></tr>")
    for s in data:
        p.append('<tr><td><span class="dot %s"></span></td><td>%s</td>'
                 '<td class="mono"><a href="#s-%s">%s</a></td><td>%s</td><td>%d</td>'
                 '<td class="hide-sm">%s</td><td class="hide-sm">%s</td></tr>'
                 % (s["state"], esc(s["container"]), esc(s["session"][:8]),
                    esc(s["session"][:8]), esc(s["age"]), len(s["turns"]),
                    esc(s["evidence"]), esc(s["cwd"])))
    p.append("</table>")

    p.append('<div class="note">Click a session below to expand its transcript. '
             "Dots: green = wrote &lt;2 min ago, grey = idle (may be waiting on you), "
             "dark = process already exited.</div>")

    for s in data:
        p.append('<details class="s" id="s-%s"><summary>'
                 '<span class="dot %s"></span><span class="mono">%s</span>'
                 '<span class="meta">%s &middot; %s &middot; %d turns &middot; %s</span>'
                 '<div class="meta">%s</div></summary><div class="body">'
                 % (esc(s["session"][:8]), s["state"], esc(s["session"][:8]),
                    esc(s["container"]), esc(s["age"]), len(s["turns"]),
                    esc(s["evidence"]), esc(s["cwd"])))
        items = condense(s["turns"])
        if raw:
            p.extend(render_turn(t) for t in s["turns"])
        elif items:
            cmd = "abox-live stop %s --yes" % s["session"][:8]
            p.append('<div class="cmd"><code>%s</code>'
                     '<button class="cpy" type="button">tap→select</button>'
                     "<span>點一下選取指令，貼到 Command Center 送出即停止</span></div>"
                     % esc(cmd))
            p.extend(render_item(i) for i in items[-40:])
        else:
            p.append('<div class="note">no narrative yet — the agent has only run '
                     "tools so far, or the transcript is not readable</div>")
        p.append("</div></details>")
    p.append('<a class="top" href="#top">&uarr; top</a>')
    # Progressive enhancement ONLY: every command is already visible and selectable as
    # text, so the page stays fully usable where inline scripts are sandboxed away.
    # Enhancement only. Published artifacts run in a SANDBOXED iframe where
    # navigator.clipboard and execCommand('copy') are usually BLOCKED — so a pure
    # "copy" button looks dead on a phone ("點了沒反應"). Instead, tapping the command
    # SELECTS its text (always works, no permission needed) and then tries a real copy
    # as a bonus; either way the user gets immediate visual feedback and can copy from
    # the OS menu. The command is already visible text if scripts are stripped entirely.
    p.append("""<script>
function selText(el){try{var r=document.createRange();r.selectNodeContents(el);
  var s=getSelection();s.removeAllRanges();s.addRange(r);return s.toString()}catch(_){return el.textContent||''}}
document.addEventListener('click',function(e){
  var box=e.target.closest&&e.target.closest('.cmd'); if(!box)return;
  var code=box.querySelector('code'), btn=box.querySelector('.cpy');
  var txt=selText(code), ok=function(){if(btn){btn.textContent='copied \\u2713';
    setTimeout(function(){btn.textContent='copy'},1600)}};
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(txt).then(ok,function(){if(btn)btn.textContent='已選取，長按複製'});
  } else if(btn){btn.textContent='已選取，長按複製'}
});
</script>""")
    return "\n".join(p)


PAGE_HEAD = """<!doctype html><meta charset="utf-8">
<title>abox dashboard — %(when)s</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#f7f7f8;--acc:#0b6bcb}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c3038;--card:#1d2026;--acc:#5aa9ff}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
#side{width:400px;min-width:280px;border-right:1px solid var(--line);display:flex;flex-direction:column}
#head{padding:10px 12px;border-bottom:1px solid var(--line)}
#head h1{font-size:14px;margin:0 0 3px}
#stat{font-size:11px;color:var(--mut);margin-bottom:8px}
input{width:100%%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit;font-size:12px}
#list{overflow:auto;flex:1}
.it{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.it:hover{background:var(--card)}
.it.sel{background:var(--card);box-shadow:inset 3px 0 0 var(--acc)}
.sid{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.ct{font-size:11px;color:var(--mut);word-break:break-all;margin-top:2px}
.lat{font-size:11px;color:var(--mut);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%%;margin-right:6px}
.live{background:#2ea043}.idle{background:#9aa0a6}.ended{background:#5b5b5b}.unknown{background:#d29922}
.badge{font-size:10px;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:5px}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:10px 14px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut);word-break:break-all}
#turns{overflow:auto;flex:1;padding:14px}
.t{margin-bottom:14px;border-left:3px solid var(--line);padding-left:10px}
.t.assistant{border-color:var(--acc)}
.role{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
pre{white-space:pre-wrap;word-wrap:break-word;margin:4px 0;font:12px/1.5 ui-monospace,Menlo,monospace;background:var(--card);padding:8px;border-radius:6px;max-height:340px;overflow:auto}
.tool{color:var(--acc);font-weight:600;font-size:12px;margin-top:4px}
.think{opacity:.6;font-style:italic}
.empty{color:var(--mut);padding:30px;text-align:center}
@media(max-width:760px){body{flex-direction:column}#side{width:100%%;height:44%%}}
</style>
<div id=side>
 <div id=head><h1>abox dashboard</h1>
  <div id=stat></div>
  <input id=q placeholder="filter container / session / cwd…" oninput=render()>
 </div>
 <div id=list></div>
</div>
<div id=main><div id=bar>select a session</div><div id=turns><div class=empty>&larr; pick one</div></div></div>
<script id=data type="application/json">"""

PAGE_TAIL = """</script>
<script>
const S=JSON.parse(document.getElementById('data').textContent);
let cur=null;
const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const last=s=>{const t=s.turns[s.turns.length-1];if(!t)return'(no turns)';
  const b=t.blocks[0]||{};return b.kind==='tool'?'\\u25b8 '+b.text:(b.text||'').slice(0,120)};
document.getElementById('stat').textContent =
  S.filter(s=>s.live).length+' live \\u00b7 '+S.filter(s=>s.state==='live').length+
  ' working \\u00b7 '+new Set(S.map(s=>s.container)).size+' containers';
function render(){
  const q=document.getElementById('q').value.toLowerCase();
  const f=S.filter(s=>!q||(s.container+s.session+s.cwd).toLowerCase().includes(q));
  document.getElementById('list').innerHTML = f.length? f.map((s,i)=>
    `<div class="it ${s.session===cur?'sel':''}" onclick="open_('${s.session}')">
      <div><span class="dot ${s.state}"></span><span class="sid">${esc(s.session.slice(0,8))}</span>
        <span class="badge">${esc(s.age)}</span><span class="badge">${esc(s.evidence)}</span>
        <span class="badge">${s.turns.length} turns</span></div>
      <div class="ct">${esc(s.container)}</div><div class="ct">${esc(s.cwd)}</div>
      <div class="lat">${esc(last(s))}</div></div>`).join('') : '<div class=empty>no match</div>';
}
function open_(id){
  cur=id; const s=S.find(x=>x.session===id); render();
  document.getElementById('bar').innerHTML=
    `<b>${esc(s.container)}</b> \\u00b7 <span class="sid">${esc(s.session)}</span> \\u00b7 ${esc(s.cwd)} \\u00b7 ${esc(s.age)} \\u00b7 ${esc(s.evidence)}`;
  document.getElementById('turns').innerHTML = s.turns.length? s.turns.map(t=>
    `<div class="t ${t.role}"><div class="role">${t.role} ${esc(t.ts.slice(11,19))}</div>`+
    t.blocks.map(b=> b.kind==='tool'?`<div class="tool">\\u25b8 ${esc(b.text)}</div><pre>${esc(b.extra||'')}</pre>`
      : b.kind==='result'?`<pre>${esc(b.text)}</pre>`
      : b.kind==='think'?`<pre class="think">${esc(b.text)}</pre>`
      : `<div>${esc(b.text).replace(/\\n/g,'<br>')}</div>`).join('')+`</div>`).join('')
    : '<div class=empty>transcript not readable</div>';
  document.getElementById('turns').scrollTop=1e9;
}
render(); if(S.length) open_(S[0].session);
</script>"""


def _own_to_user(path):
    """Hand a root-created file back to the box user so later non-sudo runs and the
    Artifact tool can read it. No-op if we're not root or the user is unknown."""
    try:
        pw = pwd.getpwnam(PUBLISH_OWNER)
        os.chown(path, pw.pw_uid, pw.pw_gid)
    except (KeyError, PermissionError, OSError):
        pass


def main():
    publish = "--publish" in sys.argv
    # In publish mode the output goes to the stable path unless -o overrides it, so a
    # bare `abox-live publish` always refreshes the bookmarked artifact's source file.
    out = STABLE_OUT if publish else "abox-dashboard.html"
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
    turns_n = MAX_TURNS
    if "--turns" in sys.argv:
        turns_n = int(sys.argv[sys.argv.index("--turns") + 1])
    recent = 0
    if "--recent" in sys.argv:
        recent = int(sys.argv[sys.argv.index("--recent") + 1])
    elif publish:
        recent = 180   # dashboards want recently-finished sessions too, not just live
    data = build("--all" in sys.argv, turns_n, recent)

    when = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_html(data, when, raw="--raw" in sys.argv))
    size = os.path.getsize(out)
    os.chmod(out, 0o644)
    _own_to_user(out)
    if publish:
        _own_to_user(os.path.dirname(os.path.abspath(out)))
    print("%s  (%d sessions, %.1f MB, snapshot at %s)" %
          (out, len(data), size / 1e6, when))

    if publish:
        # The script can't publish; only the agent's Artifact tool can. Emit the exact
        # parameters (esp. url= — omitting it mints a NEW url and breaks the bookmark).
        print(
            "\n=== PUBLISH ARTIFACT — agent: call the Artifact tool now with these ===\n"
            "file_path:   %s\n"
            "url:         %s\n"
            "title:       %s\n"
            "favicon:     %s\n"
            "description: %s\n"
            "=====================================================================\n"
            "(url= is REQUIRED — it keeps the same artifact so the phone bookmark stays valid.)"
            % (os.path.abspath(out), ARTIFACT_URL, ARTIFACT_TITLE,
               ARTIFACT_FAVICON, ARTIFACT_DESC))


if __name__ == "__main__":
    main()
