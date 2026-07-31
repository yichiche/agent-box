#!/usr/bin/env python3
"""Shared transcript parsing and human-centric rendering for abox-live.

Turns raw Claude Code JSONL into:
  - full turns (for debugging / local web raw mode)
  - condensed dialogue (user instructions + agent prose, tool bursts collapsed)
  - HTML fragments (no-JS friendly: <details> for internal work)
"""
import json

MAX_BLOCK = 4000
MAX_SESSION_BYTES = 400_000


def _first_user_text(o):
    m = o.get("message")
    if not isinstance(m, dict):
        return ""
    c = m.get("content")
    if isinstance(c, str):
        txt = c
    elif isinstance(c, list):
        txt = " ".join(b.get("text", "") for b in c
                        if isinstance(b, dict) and b.get("type") == "text")
    else:
        return ""
    txt = txt.strip()
    if not txt or txt.startswith("<") or txt.startswith("/"):
        return ""
    return txt


def parse_turns(path, max_turns=120, max_block=MAX_BLOCK):
    """Returns (turns, title)."""
    out = []
    ai_title = ""
    summary = ""
    first_user = ""
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except Exception as e:
        return ([{"role": "error", "ts": "", "blocks": [{"kind": "text",
                 "text": "cannot read transcript: %s" % e}]}], "")
    for ln in raw.splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = o.get("type")
        if t == "ai-title":
            v = (o.get("aiTitle") or "").strip()
            if v:
                ai_title = v
            continue
        if t == "summary":
            v = (o.get("summary") or "").strip()
            if v:
                summary = v
            continue
        if t not in ("assistant", "user"):
            continue
        if t == "user" and not first_user:
            first_user = _first_user_text(o)
        m = o.get("message", {})
        c = m.get("content") if isinstance(m, dict) else None
        blocks = []
        if isinstance(c, str):
            blocks.append({"kind": "text", "text": c[:max_block]})
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                k = p.get("type")
                if k == "text":
                    blocks.append({"kind": "text", "text": p.get("text", "")[:max_block]})
                elif k == "thinking":
                    blocks.append({"kind": "think", "text": p.get("thinking", "")[:max_block]})
                elif k == "tool_use":
                    blocks.append({"kind": "tool", "text": p.get("name", ""),
                                   "extra": json.dumps(p.get("input", {}))[:1200]})
                elif k == "tool_result":
                    t = p.get("content")
                    if isinstance(t, list):
                        t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                    blocks.append({"kind": "result", "text": str(t)[:max_block]})
        if blocks:
            out.append({"role": o.get("type"), "ts": (o.get("timestamp") or "")[:19],
                        "blocks": blocks})
    out = out[-max_turns:]
    while len(json.dumps(out)) > MAX_SESSION_BYTES and len(out) > 8:
        out = out[len(out) // 4:]
    title = ai_title or summary or first_user
    return (out, title)


def _turn_text(turn):
    return " ".join(b.get("text", "") for b in turn.get("blocks", [])
                    if b.get("kind") == "text").strip()


def condense(turns):
    """Human-centric timeline: instructions, agent replies, collapsed tool bursts."""
    out = []
    pending = []

    def flush():
        if not pending:
            return
        names = {}
        for t in pending:
            n = t.get("name") or "?"
            names[n] = names.get(n, 0) + 1
        out.append({"kind": "work", "n": len(pending), "names": names, "tools": pending})
        pending.clear()

    for t in turns:
        said = _turn_text(t)
        if said:
            flush()
            out.append({"kind": "ask" if t["role"] == "user" else "say",
                        "ts": t.get("ts", ""), "text": said})
            continue
        for b in t.get("blocks", []):
            k = b.get("kind")
            if k == "tool":
                pending.append({"name": b.get("text", ""), "input": b.get("extra", ""),
                                "result": None})
            elif k == "result" and pending:
                pending[-1]["result"] = b.get("text", "")
            elif k == "think" and b.get("text"):
                pending.append({"name": "thinking", "input": b.get("text", "")[:800],
                                "result": None})
    flush()
    return out


def esc(s):
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_work_item(it):
    """Collapsible internal agent work (<details>, no JS required)."""
    top = sorted(it.get("names", {}).items(), key=lambda kv: -kv[1])[:5]
    label = " ".join("%s×%d" % (k, v) for k, v in top)
    parts = ['<details class="work"><summary class="work-sum">&#8943; %d internal steps'
             % it["n"]]
    if label:
        parts.append(' <span class="meta">%s</span>' % esc(label))
    parts.append("</summary><div class=\"work-body\">")
    for tool in it.get("tools", []):
        nm = tool.get("name") or "?"
        parts.append('<div class="tool-step"><div class="tool">&#9656; %s</div>' % esc(nm))
        if tool.get("input"):
            parts.append("<pre>%s</pre>" % esc(tool["input"]))
        if tool.get("result"):
            parts.append("<pre class=\"result\">%s</pre>" % esc(tool["result"]))
        parts.append("</div>")
    parts.append("</div></details>")
    return "".join(parts)


def render_dialogue_item(it):
    who = "you" if it["kind"] == "ask" else "agent"
    role_cls = "user" if it["kind"] == "ask" else "assistant"
    ts = esc(it.get("ts", "")[11:19])
    return ('<div class="t %s hmi"><div class="role">%s %s</div><div class="hmi-text">%s</div></div>'
            % (role_cls, who, ts, esc(it.get("text", "")).replace("\n", "<br>")))


def render_condensed(items, limit=40):
    """Render condensed HMI timeline (newest `limit` items)."""
    chunk = items[-limit:] if limit else items
    return "".join(render_dialogue_item(i) if i["kind"] in ("ask", "say")
                   else render_work_item(i) for i in chunk)


def render_instruction_panel(session, container, title=""):
    """Copy-paste blocks for Remote Control / Command Center."""
    sid = session[:8]
    placeholder = "在此輸入要給 agent 的指令"
    say_cmd = 'abox-live say %s "%s"' % (sid, placeholder)
    queue_cmd = 'abox-live instruct %s "%s"' % (sid, placeholder)
    rc_natural = ('請對 session %s（container: %s%s）下指令：「%s」'
                  % (sid, container or "host",
                     (' · ' + title[:40]) if title else "", placeholder))
    return ('<div class="instr">'
            '<div class="instr-h">&#9993; 從 Command Center 下指令</div>'
            '<div class="cmd say-cmd" title="直接送給 agent（agent 閒置時）">'
            '<span class="cmd-label">直接送</span><code>%s</code>'
            '<button class="cpy" type="button">select</button></div>'
            '<div class="cmd queue-cmd" title="加入佇列，再執行 abox-live drain --yes">'
            '<span class="cmd-label">佇列</span><code>%s</code>'
            '<button class="cpy" type="button">select</button></div>'
            '<div class="rc-hint">或在 RC 對話貼上：<span class="rc-text">%s</span>'
            '<button class="cpy rc-cpy" type="button">select</button></div>'
            '</div>' % (esc(say_cmd), esc(queue_cmd), esc(rc_natural)))


def render_turn_raw(t):
    out = ['<div class="t %s"><div class="role">%s %s</div>'
           % (esc(t["role"]), esc(t["role"]), esc(t.get("ts", "")[11:19]))]
    for b in t.get("blocks", []):
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
