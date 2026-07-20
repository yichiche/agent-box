#!/usr/bin/env python3
"""abox-web — a LOCAL web view of live claude sessions and their transcripts.

Serves on 127.0.0.1 only. Transcripts are read fresh on every request, so the page
reflects current state; nothing is uploaded anywhere.

  sudo -n python3 abox_web.py [--port 8848] [--all]
"""
import os, re, sys, json, time, html, glob, secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL

MAX_TURNS = 300
MAX_CHARS = 12000
OPTS = {"all": False}


def sessions():
    rows = AL.collect()
    if not OPTS["all"]:
        rows = [r for r in rows if r["mine"]]
    out = []
    for r in rows:
        out.append({
            "session": r["session"], "container": r["container"], "cwd": r["cwd"],
            "evidence": r["evidence"], "idle": r["idle"], "age": AL.age(r["idle"]),
            "latest": r["latest"][:160], "pid": r["pid"], "path": r["path"],
            "state": "live" if 0 <= r["idle"] < 120 else ("idle" if r["idle"] >= 0 else "unknown"),
        })
    out.sort(key=lambda s: (s["container"] == "host", s["container"],
                            s["idle"] if s["idle"] >= 0 else 1e9))
    return out


def find_path(sid):
    for s in sessions():
        if s["session"].startswith(sid) and s["path"]:
            return s["path"]
    for root in AL.roots_for("host"):
        hits = glob.glob(os.path.join(root, "*", sid + "*.jsonl"))
        if hits:
            return hits[0]
    return None


def turns(path):
    out = []
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except Exception as e:
        return [{"role": "error", "text": "cannot read transcript: %s" % e, "ts": ""}]
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
            blocks.append({"kind": "text", "text": c})
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                k = p.get("type")
                if k == "text":
                    blocks.append({"kind": "text", "text": p.get("text", "")})
                elif k == "thinking":
                    blocks.append({"kind": "thinking", "text": p.get("thinking", "")})
                elif k == "tool_use":
                    blocks.append({"kind": "tool", "text": p.get("name", ""),
                                   "input": json.dumps(p.get("input", {}))[:2000]})
                elif k == "tool_result":
                    t = p.get("content")
                    if isinstance(t, list):
                        t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                    blocks.append({"kind": "result", "text": str(t)[:MAX_CHARS]})
        if not blocks:
            continue
        for b in blocks:
            b["text"] = b.get("text", "")[:MAX_CHARS]
        out.append({"role": o.get("type"), "ts": o.get("timestamp", ""), "blocks": blocks})
    return out[-MAX_TURNS:]


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>abox — live sessions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#fafafa;--acc:#0b6bcb}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c3038;--card:#1d2026;--acc:#5aa9ff}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
#side{width:390px;min-width:300px;border-right:1px solid var(--line);display:flex;flex-direction:column}
#head{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#head b{font-size:14px}#head .sp{flex:1}
input[type=search]{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
#list{overflow:auto;flex:1}
.it{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.it:hover{background:var(--card)}.it.sel{background:var(--card);border-left:3px solid var(--acc);padding-left:9px}
.ct{font-size:11px;color:var(--mut);word-break:break-all}
.sid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.lat{font-size:11px;color:var(--mut);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.live{background:#2ea043}.idle{background:#9aa0a6}.unknown{background:#d29922}
.badge{font-size:10px;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:6px}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:10px 14px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut);word-break:break-all}
#turns{overflow:auto;flex:1;padding:14px}
.t{margin-bottom:14px;border-left:3px solid var(--line);padding-left:10px}
.t.assistant{border-color:var(--acc)}
.role{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
pre{white-space:pre-wrap;word-wrap:break-word;margin:4px 0;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:8px;border-radius:6px;overflow-x:auto}
.tool{color:var(--acc);font-weight:600;font-size:12px}
.think{opacity:.65;font-style:italic}
.empty{color:var(--mut);padding:30px;text-align:center}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 9px;cursor:pointer;font-size:12px}
</style></head><body>
<div id="side">
 <div id="head"><b>abox live</b><span class="sp"></span>
   <label style="font-size:11px;color:var(--mut)"><input type="checkbox" id="auto" checked> auto</label>
   <button onclick="load()">↻</button>
   <input type="search" id="q" placeholder="filter container / session / cwd…" oninput="render()">
 </div>
 <div id="list"><div class="empty">loading…</div></div>
</div>
<div id="main"><div id="bar">select a session on the left</div><div id="turns"></div></div>
<script>
let S=[],cur=null;
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function load(){
  try{S=await (await fetch('/api/sessions')).json()}catch(e){return}
  render(); if(cur) openS(cur,true);
}
function render(){
  const q=document.getElementById('q').value.toLowerCase();
  const L=document.getElementById('list');
  const f=S.filter(s=>!q||(s.container+s.session+s.cwd).toLowerCase().includes(q));
  if(!f.length){L.innerHTML='<div class="empty">no sessions</div>';return}
  L.innerHTML=f.map(s=>`<div class="it ${s.session===cur?'sel':''}" onclick="openS('${s.session}')">
    <div><span class="dot ${s.state}"></span><span class="sid">${esc(s.session.slice(0,8))}</span>
      <span class="badge">${esc(s.age)}</span><span class="badge">${esc(s.evidence)}</span></div>
    <div class="ct">${esc(s.container)} · ${esc(s.cwd)}</div>
    <div class="lat">${esc(s.latest)}</div></div>`).join('');
}
async function openS(id,keep){
  cur=id; if(!keep)document.getElementById('turns').innerHTML='<div class="empty">loading…</div>';
  render();
  const s=S.find(x=>x.session===id)||{};
  document.getElementById('bar').innerHTML=`<b>${esc(s.container||'')}</b> · <span class="sid">${esc(id)}</span> · ${esc(s.cwd||'')} · ${esc(s.age||'')} · ${esc(s.evidence||'')}`;
  let d; try{d=await (await fetch('/api/session/'+id)).json()}catch(e){return}
  const box=document.getElementById('turns');
  const atBottom=box.scrollTop+box.clientHeight>=box.scrollHeight-80;
  box.innerHTML=d.turns.map(t=>`<div class="t ${t.role}"><div class="role">${t.role} ${esc((t.ts||'').slice(11,19))}</div>`+
    t.blocks.map(b=>b.kind==='tool'?`<div class="tool">▸ ${esc(b.text)}</div><pre>${esc(b.input)}</pre>`
      :b.kind==='result'?`<pre>${esc(b.text)}</pre>`
      :b.kind==='thinking'?`<pre class="think">${esc(b.text)}</pre>`
      :`<div>${esc(b.text).replace(/\n/g,'<br>')}</div>`).join('')+`</div>`).join('')
    ||'<div class="empty">no turns yet</div>';
  if(!keep||atBottom)box.scrollTop=box.scrollHeight;
}
load(); setInterval(()=>{if(document.getElementById('auto').checked)load()},8000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _authed(self):
        """This box is multi-user, so 127.0.0.1 is not private — anyone logged in here
        could otherwise read every transcript. One shared token, passed as ?k= once and
        then kept in a cookie."""
        tok = OPTS.get("token")
        if not tok:
            return True
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        if ("k=" + tok) in q:
            return True
        return ("abox_k=" + tok) in (self.headers.get("Cookie") or "")

    def do_GET(self):
        p = self.path.split("?")[0]
        if not self._authed():
            return self._send(403, "forbidden — open the URL printed by `abox-live web` "
                                   "(it carries the ?k= token)", "text/plain")
        if p == "/":
            tok = OPTS.get("token")
            b = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            if tok:
                self.send_header("Set-Cookie", "abox_k=%s; Path=/; SameSite=Strict" % tok)
            self.end_headers()
            return self.wfile.write(b)
        if p == "/api/sessions":
            return self._send(200, json.dumps(sessions()), "application/json")
        if p.startswith("/api/session/"):
            sid = re.sub(r"[^0-9a-fA-F-]", "", p.rsplit("/", 1)[1])[:36]
            path = find_path(sid) if sid else None
            if not path:
                return self._send(404, json.dumps({"turns": []}), "application/json")
            return self._send(200, json.dumps({"turns": turns(path)}), "application/json")
        self._send(404, "not found", "text/plain")

    def log_message(self, *a):
        pass


def main():
    port = 8848
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    host = "127.0.0.1"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]
    OPTS["all"] = "--all" in sys.argv
    OPTS["token"] = None if "--no-token" in sys.argv else secrets.token_urlsafe(9)

    srv = ThreadingHTTPServer((host, port), H)
    url = "http://%s:%d/" % ("127.0.0.1" if host == "127.0.0.1" else host, port)
    if OPTS["token"]:
        url += "?k=" + OPTS["token"]
    print("abox-web  ->  %s" % url)
    if host != "127.0.0.1":
        print("!! bound to %s — reachable from the network. Transcripts are internal;"
              " prefer 127.0.0.1 + an SSH tunnel." % host)
    print("this host is multi-user, so the ?k= token gates other local accounts "
          "(--no-token to disable)")
    print("browsing from your laptop? forward the port first:")
    print("   ssh -L %d:127.0.0.1:%d %s@%s" % (
        port, port, os.environ.get("SUDO_USER", "user"),
        os.uname().nodename))
    print("   or VS Code: PORTS panel -> Forward a Port -> %d" % port)
    sys.stdout.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
