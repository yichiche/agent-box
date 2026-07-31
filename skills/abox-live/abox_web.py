#!/usr/bin/env python3
"""abox-web — LOCAL web view of live claude sessions (human-centric + say API).

Serves on 127.0.0.1 only. Transcripts read fresh each request.

  sudo -n python3 abox_web.py [--port 8848] [--all]
"""
import os, re, sys, json, time, glob, secrets, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abox_live as AL
import abox_transcript as AT
import abox_inbox as INBOX

MAX_TURNS = 300
OPTS = {"all": False}


def sessions():
    rows = AL.collect()
    if not OPTS["all"]:
        rows = [r for r in rows if r["mine"]]
    out = []
    for r in rows:
        title = ""
        dlg = []
        if r["path"]:
            turns, title = AT.parse_turns(r["path"], MAX_TURNS)
            dlg = AT.condense(turns)
        out.append({
            "session": r["session"], "container": r["container"], "cwd": r["cwd"],
            "evidence": r["evidence"], "idle": r["idle"], "age": AL.age(r["idle"]),
            "latest": r["latest"][:160], "pid": r["pid"], "path": r["path"],
            "title": title,
            "state": "live" if 0 <= r["idle"] < 120 else ("idle" if r["idle"] >= 0 else "unknown"),
            "dialogue_len": len(dlg),
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


def session_detail(sid):
    path = find_path(sid)
    if not path:
        return None
    turns, title = AT.parse_turns(path, MAX_TURNS)
    return {"turns": turns, "dialogue": AT.condense(turns), "title": title}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>abox — live sessions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#fafafa;--acc:#0b6bcb;
     --you:#d29922;--agent:#0b6bcb;--work:#888}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c3038;
     --card:#1d2026;--acc:#5aa9ff;--you:#e3b341;--agent:#5aa9ff;--work:#777}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
#side{width:390px;min-width:300px;border-right:1px solid var(--line);display:flex;flex-direction:column}
#head{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#head b{font-size:14px}#head .sp{flex:1}
input[type=search],textarea{font-family:inherit;width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
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
#say-bar{padding:10px 14px;border-bottom:1px solid var(--line);display:none;gap:8px;flex-direction:column}
#say-bar.show{display:flex}
#turns{overflow:auto;flex:1;padding:14px}
.t{margin-bottom:14px;border-left:3px solid var(--line);padding-left:10px}
.t.hmi.user{border-color:var(--you)}.t.hmi.assistant{border-color:var(--agent)}
.role{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.t.user .role{color:var(--you)}.t.assistant .role{color:var(--agent)}
.hmi-text{white-space:pre-wrap;word-wrap:break-word}
pre{white-space:pre-wrap;word-wrap:break-word;margin:4px 0;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:8px;border-radius:6px;overflow-x:auto;max-height:280px}
.tool{color:var(--acc);font-weight:600;font-size:12px}
details.work{border:1px dashed var(--line);border-radius:6px;margin:10px 0;background:var(--card)}
details.work>summary{cursor:pointer;padding:8px 10px;font-size:12px;color:var(--work);list-style:none}
details.work[open]>summary:before{content:"▾ "}details.work>summary:before{content:"▸ "}
.work-body{padding:4px 10px 10px;border-top:1px dashed var(--line)}
.empty{color:var(--mut);padding:30px;text-align:center}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 9px;cursor:pointer;font-size:12px}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
#say-status{font-size:11px;color:var(--mut);min-height:16px}
label.chk{font-size:11px;color:var(--mut);display:flex;align-items:center;gap:6px}
</style></head><body>
<div id="side">
 <div id="head"><b>abox live</b><span class="sp"></span>
   <label class="chk"><input type="checkbox" id="hmi" checked> human view</label>
   <label class="chk"><input type="checkbox" id="auto" checked> auto</label>
   <button onclick="load()">↻</button>
   <input type="search" id="q" placeholder="filter container / session / cwd…" oninput="render()">
 </div>
 <div id="list"><div class="empty">loading…</div></div>
</div>
<div id="main">
 <div id="bar">select a session on the left</div>
 <div id="say-bar">
   <textarea id="say-msg" rows="2" placeholder="送指令給這個 agent…"></textarea>
   <div style="display:flex;gap:8px;align-items:center">
     <button class="primary" onclick="sendSay(false)">直接送 (say)</button>
     <button onclick="sendSay(true)">加入佇列 (instruct)</button>
     <label class="chk"><input type="checkbox" id="say-force"> --force</label>
     <span id="say-status"></span>
   </div>
 </div>
 <div id="turns"></div>
</div>
<script>
let S=[],cur=null,curDetail=null;
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function renderHmi(dlg){
  return dlg.map(it=>{
    if(it.kind==='work'){
      const top=Object.entries(it.names||{}).sort((a,b)=>b[1]-a[1]).slice(0,4);
      const lbl=top.map(([k,v])=>k+'×'+v).join(' ');
      let body=(it.tools||[]).map(t=>'<div class="tool">▸ '+esc(t.name)+'</div>'
        +(t.input?'<pre>'+esc(t.input)+'</pre>':'')
        +(t.result?'<pre>'+esc(t.result)+'</pre>':'')).join('');
      return '<details class="work"><summary>⋯ '+it.n+' internal steps'
        +(lbl?' <span style="opacity:.7">'+esc(lbl)+'</span>':'')+'</summary><div class="work-body">'+body+'</div></details>';
    }
    const who=it.kind==='ask'?'you':'agent';
    const cls=it.kind==='ask'?'user':'assistant';
    return '<div class="t hmi '+cls+'"><div class="role">'+who+' '+esc((it.ts||'').slice(11,19))+'</div>'
      +'<div class="hmi-text">'+esc(it.text).replace(/\n/g,'<br>')+'</div></div>';
  }).join('');
}
function renderRaw(turns){
  return turns.map(t=>'<div class="t '+t.role+'"><div class="role">'+t.role+' '+esc((t.ts||'').slice(11,19))+'</div>'
    +t.blocks.map(b=>b.kind==='tool'?'<div class="tool">▸ '+esc(b.text)+'</div><pre>'+esc(b.input||b.extra||'')+'</pre>'
      :b.kind==='result'?'<pre>'+esc(b.text)+'</pre>'
      :b.kind==='thinking'||b.kind==='think'?'<pre style="opacity:.65;font-style:italic">'+esc(b.text)+'</pre>'
      :'<div>'+esc(b.text).replace(/\n/g,'<br>')+'</div>').join('')+'</div>').join('');
}
async function load(){
  try{S=await (await fetch('/api/sessions')).json()}catch(e){return}
  render(); if(cur) openS(cur,true);
}
function render(){
  const q=document.getElementById('q').value.toLowerCase();
  const L=document.getElementById('list');
  const f=S.filter(s=>!q||(s.container+s.session+s.cwd+(s.title||'')).toLowerCase().includes(q));
  if(!f.length){L.innerHTML='<div class="empty">no sessions</div>';return}
  L.innerHTML=f.map(s=>`<div class="it ${s.session===cur?'sel':''}" onclick="openS('${s.session}')">
    <div><span class="dot ${s.state}"></span><span class="sid">${esc(s.session.slice(0,8))}</span>
      <span class="badge">${esc(s.age)}</span><span class="badge">${esc(s.evidence)}</span></div>
    <div class="ct">${esc(s.container)} · ${esc(s.cwd)}</div>
    <div class="lat">${esc(s.title||s.latest)}</div></div>`).join('');
}
async function openS(id,keep){
  cur=id;
  document.getElementById('say-bar').classList.add('show');
  if(!keep)document.getElementById('turns').innerHTML='<div class="empty">loading…</div>';
  render();
  const s=S.find(x=>x.session===id)||{};
  document.getElementById('bar').innerHTML=`<b>${esc(s.container||'')}</b> · <span class="sid">${esc(id.slice(0,8))}</span> · ${esc(s.cwd||'')} · ${esc(s.age||'')} · ${esc(s.evidence||'')}`;
  let d; try{d=await (await fetch('/api/session/'+id)).json()}catch(e){return}
  curDetail=d;
  const box=document.getElementById('turns');
  const atBottom=box.scrollTop+box.clientHeight>=box.scrollHeight-80;
  const hmi=document.getElementById('hmi').checked;
  box.innerHTML=(hmi&&d.dialogue&&d.dialogue.length?renderHmi(d.dialogue):renderRaw(d.turns||[]))
    ||'<div class="empty">no turns yet</div>';
  if(!keep||atBottom)box.scrollTop=box.scrollHeight;
}
document.getElementById('hmi').onchange=()=>{if(curDetail&&cur)openS(cur,true)};
async function sendSay(queue){
  if(!cur)return;
  const msg=document.getElementById('say-msg').value.trim();
  if(!msg){document.getElementById('say-status').textContent='enter a message';return}
  const st=document.getElementById('say-status');
  st.textContent='sending…';
  const body={session:cur,message:msg,force:document.getElementById('say-force').checked,queue:queue};
  try{
    const r=await fetch('/api/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    st.textContent=j.ok?(j.output||'ok').slice(0,200):(j.error||'failed');
    if(j.ok){document.getElementById('say-msg').value=''; setTimeout(()=>openS(cur,true),1500);}
  }catch(e){st.textContent='error: '+e;}
}
load(); setInterval(()=>{if(document.getElementById('auto').checked)load()},8000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _authed(self):
        tok = OPTS.get("token")
        if not tok:
            return True
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        if ("k=" + tok) in q:
            return True
        return ("abox_k=" + tok) in (self.headers.get("Cookie") or "")

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        p = self.path.split("?")[0]
        if not self._authed():
            return self._send(403, "forbidden — open the URL printed by `abox-live web`", "text/plain")
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
            det = session_detail(sid) if sid else None
            if not det:
                return self._send(404, json.dumps({"turns": [], "dialogue": []}), "application/json")
            return self._send(200, json.dumps(det), "application/json")
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        p = self.path.split("?")[0]
        if not self._authed():
            return self._send(403, json.dumps({"ok": False, "error": "forbidden"}), "application/json")
        if p == "/api/say":
            body = self._read_json()
            sid = (body.get("session") or "")[:36]
            msg = (body.get("message") or "").strip()
            if not sid or not msg:
                return self._send(400, json.dumps({"ok": False, "error": "session and message required"}),
                                  "application/json")
            if body.get("queue"):
                rc = INBOX.instruct(sid[:8], msg)
                return self._send(200, json.dumps({"ok": rc == 0,
                                                     "output": "queued — run abox-live drain --yes"}),
                                  "application/json")
            argv = [sys.executable, os.path.join(os.path.dirname(__file__), "abox_say.py"),
                    sid[:8], msg]
            if body.get("force"):
                argv.append("--force")
            if body.get("dangerous"):
                argv.append("--dangerous")
            try:
                r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
                out = (r.stdout or "") + (r.stderr or "")
                return self._send(200, json.dumps({"ok": r.returncode == 0, "output": out.strip()}),
                                  "application/json")
            except subprocess.TimeoutExpired:
                return self._send(200, json.dumps({"ok": False, "error": "timeout"}), "application/json")
        self._send(404, json.dumps({"ok": False}), "application/json")

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
        print("!! bound to %s — reachable from the network." % host)
    print("human view: instructions + agent replies; internal work folded in <details>")
    print("say API: POST /api/say {session, message, force?, queue?}")
    print("forward from laptop: ssh -L %d:127.0.0.1:%d …" % (port, port))
    sys.stdout.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
