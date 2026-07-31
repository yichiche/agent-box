#!/usr/bin/env python3
"""abox-snapshot — build a SELF-CONTAINED dashboard of live sessions + transcripts.

Human-centric layout: user instructions and agent replies up front; internal tool
work collapsed in <details>. Each session has copy-paste blocks for Remote Control.

Live sessions only, always — there is no arg that folds ended sessions back in.

  sudo -n python3 abox_snapshot.py [-o out.html] [--all] [--turns N]
"""
import os, re, sys, json, time, pwd, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL
import abox_transcript as AT
import abox_inbox as INBOX

MAX_TURNS = 120

PUBLISH_OWNER = "yichiche"
STABLE_OUT = "/home/yichiche/agent-scratch/abox-dash/abox-dashboard.html"
ARTIFACT_URL = "https://claude.ai/code/artifact/605cb93f-db89-4777-bbfb-6a4a9f50276c"
ARTIFACT_TITLE = "abox dashboard"
ARTIFACT_FAVICON = "🛰️"
ARTIFACT_DESC = "Live status of Claude agents — human dialogue + RC instruction panel"


def build(show_all, max_turns):
    rows = AL.collect()
    if not show_all:
        rows = [r for r in rows if r["mine"]]
    data = []
    for r in rows:
        turns, title = AT.parse_turns(r["path"], max_turns) if r["path"] else ([], "")
        data.append({
            "session": r["session"], "container": r["container"], "cwd": r["cwd"],
            "evidence": r["evidence"], "age": AL.age(r["idle"]), "idle": r["idle"],
            "pid": r["pid"], "live": True, "title": title,
            "state": "live" if 0 <= r["idle"] < 120 else ("idle" if r["idle"] >= 0 else "unknown"),
            "turns": turns,
            "dialogue": AT.condense(turns),
        })

    data.sort(key=lambda s: s["idle"] if s["idle"] >= 0 else float("inf"))
    return data


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e0e0e0;--card:#f7f7f8;--acc:#0b6bcb;
     --you:#d29922;--agent:#0b6bcb;--work:#888}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c3038;
     --card:#1d2026;--acc:#5aa9ff;--you:#e3b341;--agent:#5aa9ff;--work:#777}}
*{box-sizing:border-box}
body{margin:0;padding:18px;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
.hdr{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:4px;flex-wrap:wrap}
h1{font-size:18px;margin:0}
.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.inbox{border:1px solid var(--acc);border-radius:8px;padding:12px 14px;margin-bottom:20px;background:var(--card)}
.inbox-h{font-weight:600;font-size:13px;margin-bottom:6px}
.inbox-sub{font-size:11px;color:var(--mut);margin:8px 0 4px}
.inbox-item{font-size:12px;padding:6px 0;border-top:1px solid var(--line)}
.inbox-item.pending{border-left:3px solid var(--acc);padding-left:8px;margin-left:-4px}
.inbox-msg{color:var(--fg);margin-top:3px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;margin-bottom:26px;font-size:12px;display:block;overflow-x:auto}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td.wrap{white-space:normal}
a{color:var(--acc)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.live{background:#2ea043}.idle{background:#9aa0a6}.unknown{background:#d29922}
details.s{border:1px solid var(--line);border-radius:8px;margin-bottom:12px;background:var(--card)}
details.s>summary{cursor:pointer;padding:10px 12px;font-size:13px;list-style:none}
details.s>summary::-webkit-details-marker{display:none}
details.s>summary:before{content:"\\25b6";color:var(--mut);margin-right:8px;font-size:10px}
details.s[open]>summary:before{content:"\\25bc"}
.body{padding:4px 14px 14px;background:var(--bg);border-top:1px solid var(--line)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meta{color:var(--mut);font-size:11px;margin-left:6px}
td.desc{max-width:340px;color:var(--fg)}
.title{font-weight:600}
.t{margin:12px 0;border-left:3px solid var(--line);padding-left:10px}
.t.hmi.user{border-color:var(--you)}
.t.hmi.assistant{border-color:var(--agent)}
.t.hmi .hmi-text{white-space:pre-wrap;word-wrap:break-word}
.role{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.t.user .role{color:var(--you)}.t.assistant .role{color:var(--agent)}
pre{white-space:pre-wrap;word-wrap:break-word;margin:4px 0;font:12px/1.5 ui-monospace,Menlo,monospace;background:var(--card);padding:8px;border-radius:6px;max-height:280px;overflow:auto}
pre.result{opacity:.85;font-size:11px}
.tool{color:var(--acc);font-weight:600;font-size:12px;margin-top:4px}
.think{opacity:.6;font-style:italic}
.note{color:var(--mut);font-size:12px;border-left:3px solid var(--line);padding-left:10px;margin:18px 0}
details.work{border:1px dashed var(--line);border-radius:6px;margin:10px 0;background:var(--card)}
details.work>summary.work-sum{cursor:pointer;padding:8px 10px;font-size:12px;color:var(--work);list-style:none}
details.work>summary::-webkit-details-marker{display:none}
details.work>summary:before{content:"\\25b8 ";color:var(--mut)}
details.work[open]>summary:before{content:"\\25be "}
.work-body{padding:4px 10px 10px;border-top:1px dashed var(--line)}
.tool-step{margin-bottom:8px}
.instr{border:1px solid var(--acc);border-radius:8px;padding:10px 12px;margin:0 0 14px;background:var(--card)}
.instr-h{font-size:12px;font-weight:600;color:var(--acc);margin-bottom:8px}
.cmd{font-size:11px;color:var(--mut);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:6px 0;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
.cmd-label{font-size:10px;font-weight:600;color:var(--acc);min-width:42px}
.cmd code{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--fg);user-select:all;flex:1;word-break:break-all}
.rc-hint{font-size:11px;color:var(--mut);margin-top:8px;line-height:1.5}
.rc-text{color:var(--fg);font-style:italic}
.cpy{font:inherit;font-size:11px;padding:6px 11px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--acc);cursor:pointer;min-height:32px;white-space:nowrap}
.cpy:active{opacity:.6}
.top{position:sticky;bottom:10px;float:right;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 9px;text-decoration:none}
.view-toggle{font-size:11px;color:var(--mut);margin:10px 0}
@media(max-width:640px){
  body{padding:12px}
  h1{font-size:16px}
  table{font-size:11px}
  td,th{padding:5px 7px}
  details.s>summary{padding:13px 12px;font-size:14px}
  pre{max-height:200px;font-size:11px}
  .hide-sm{display:none}
}
"""

COPY_JS = """<script>
function selText(el){try{var r=document.createRange();r.selectNodeContents(el);
  var s=getSelection();s.removeAllRanges();s.addRange(r);return s.toString()}catch(_){return el.textContent||''}}
document.addEventListener('click',function(e){
  var btn=e.target.closest&&e.target.closest('.cpy'); if(!btn)return;
  e.preventDefault();
  var box=btn.closest('.cmd')||btn.closest('.instr')||btn.closest('.rc-hint');
  var el=box&&(box.querySelector('code')||box.querySelector('.rc-text'));
  if(!el)return;
  var txt=selText(el);
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(txt).then(function(){btn.textContent='copied \\u2713';
      setTimeout(function(){btn.textContent='select'},1600)});
  } else {btn.textContent='已選取';}
});
</script>"""


def esc(s):
    return AT.esc(s)


def shorten(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def render_html(data, when, raw=False):
    p = ['<!doctype html><meta charset="utf-8">',
         "<title>abox dashboard</title>",
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">',
         "<style>%s</style>" % CSS,
         '<div class="hdr"><h1>abox dashboard</h1></div>',
         '<div class="sub">snapshot %s &middot; %d sessions &middot; %d live &middot; '
         "%d containers &middot; reload after <code>abox-live publish</code>"
         % (esc(when), len(data), sum(1 for s in data if s["live"]),
            len({s["container"] for s in data}))]

    p.append(INBOX.render_inbox_html())

    p.append('<table id="top"><tr><th></th><th>container</th><th>session</th>'
             "<th>description</th><th>age</th><th>dialogue</th><th class=\"hide-sm\">ev</th>"
             "<th class=\"hide-sm\">cwd</th></tr>")
    for s in data:
        desc = s.get("title") or ""
        dlg = s.get("dialogue") or []
        n_ask = sum(1 for d in dlg if d["kind"] == "ask")
        n_say = sum(1 for d in dlg if d["kind"] == "say")
        p.append('<tr><td><span class="dot %s"></span></td><td>%s</td>'
                 '<td class="mono"><a href="#s-%s">%s</a></td>'
                 '<td class="wrap desc">%s</td><td>%s</td><td>%d you / %d agent</td>'
                 '<td class="hide-sm">%s</td><td class="hide-sm">%s</td></tr>'
                 % (s["state"], esc(s["container"]), esc(s["session"][:8]),
                    esc(s["session"][:8]), esc(shorten(desc, 80)), esc(s["age"]),
                    n_ask, n_say, esc(s["evidence"]), esc(s["cwd"])))
    p.append("</table>")

    p.append('<div class="note"><b>Human view</b> — your instructions and agent replies are '
             "shown directly; internal tool/thinking work is folded under "
             '<code>⋯ internal steps</code>. Use the <b>instruction panel</b> in each session '
             "to copy a command for Command Center (Remote Control).</div>")

    for s in data:
        title = s.get("title") or "(no title)"
        p.append('<details class="s" id="s-%s"><summary>'
                 '<span class="dot %s"></span><span class="title">%s</span>'
                 '<span class="mono">%s</span>'
                 '<span class="meta">%s &middot; %s &middot; %s</span>'
                 '<div class="meta">%s</div></summary><div class="body">'
                 % (esc(s["session"][:8]), s["state"], esc(shorten(title, 120)),
                    esc(s["session"][:8]), esc(s["container"]), esc(s["age"]),
                    esc(s["evidence"]), esc(s["cwd"])))

        if s.get("live"):
            p.append(AT.render_instruction_panel(s["session"], s["container"], title))

        if s.get("live"):
            stop_cmd = "abox-live stop %s --yes" % s["session"][:8]
            p.append('<div class="cmd stop-cmd"><span class="cmd-label">停止</span>'
                     '<code>%s</code><button class="cpy" type="button">select</button></div>'
                     % esc(stop_cmd))

        dlg = s.get("dialogue") or []
        if raw:
            p.append('<div class="view-toggle">raw transcript</div>')
            p.extend(AT.render_turn_raw(t) for t in s["turns"])
        elif dlg:
            p.append(AT.render_condensed(dlg, limit=50))
        else:
            p.append('<div class="note">no dialogue yet — agent has only run internal tools</div>')
        p.append("</div></details>")

    p.append('<a class="top" href="#top">&uarr; top</a>')
    p.append(COPY_JS)
    return "\n".join(p)


def _own_to_user(path):
    try:
        pw = pwd.getpwnam(PUBLISH_OWNER)
        os.chown(path, pw.pw_uid, pw.pw_gid)
    except (KeyError, PermissionError, OSError):
        pass


def main():
    publish = "--publish" in sys.argv
    out = STABLE_OUT if publish else "abox-dashboard.html"
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
    turns_n = MAX_TURNS
    if "--turns" in sys.argv:
        turns_n = int(sys.argv[sys.argv.index("--turns") + 1])
    data = build("--all" in sys.argv, turns_n)

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
