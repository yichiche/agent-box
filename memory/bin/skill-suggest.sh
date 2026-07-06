#!/usr/bin/env bash
# skill-suggest.sh — detect recurring themes across the memory vault and
# draft workflow/skill improvement STUBS for human review.
#
# Philosophy: detect -> draft -> you approve. It NEVER creates a real skill.
# It writes review stubs to memory/meta/suggestions/ that link the notes that
# motivated them; you promote the good ones by hand (or with /memory-capture).
#
# Usage:
#   skill-suggest.sh            scan and write/update suggestion stubs
#   skill-suggest.sh --dry-run  print ranked candidates, write nothing
#   skill-suggest.sh --min N    minimum notes sharing a theme (default 3)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SUGG_DIR="$META_DIR/suggestions"
MIN=3
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --min) MIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

AGENT_BOX_DIR="$AGENT_BOX_DIR" MEM_DIR="$MEM_DIR" SUGG_DIR="$SUGG_DIR" \
MIN="$MIN" DRY="$DRY" python3 <<'PY'
import os, re, glob, collections, datetime, pathlib, textwrap

MEM = pathlib.Path(os.environ["MEM_DIR"])
BOX = pathlib.Path(os.environ["AGENT_BOX_DIR"])
SUGG = pathlib.Path(os.environ["SUGG_DIR"])
MIN = int(os.environ["MIN"]); DRY = os.environ["DRY"] == "1"

STOP = set("""the and for you not with your this that from into via already run runs
gate perf config test tests note notes are was were has have will can use used using
feedback skill skills workflow model models home yichiche sgl workspace sglang aiter main
when what how why fix add new old set get one two per off out end pre post never always
type name description metadata node nodeid originsessionid session project status tags
created updated title summary body about here there they them then than each also more
most some such very much need needs make made just like only over under into onto
all user apply path does done both same other while after before because""".split())

# short domain terms worth keeping despite length
KEEP_SHORT = {"moe", "gpu", "kv", "tp", "fp8", "fp4", "il", "ol", "rc"}

def strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end+4:]
    return text

def tokens(name, text):
    base = re.sub(r'^-.*?__', '', name)          # drop project prefix
    base = re.sub(r'\.md$', '', base)
    base = re.sub(r'__[0-9a-f]{8}$', '', base)    # drop sha suffix
    words = re.split(r'[^a-z0-9]+', base.lower())
    body = strip_frontmatter(text).lower()
    words += re.findall(r'[a-z][a-z0-9\-]{2,}', body[:600])
    out = set()
    for w in words:
        w = w.strip('-')
        if w in STOP or w.isdigit():
            continue
        if len(w) >= 3 or w in KEEP_SHORT:
            out.add(w)
    return out

# --- gather notes: journal + gotchas -------------------------------------
notes = []  # (relpath, tokenset)
for pat in ("journal/**/*.md", "gotchas/*.md"):
    for f in glob.glob(str(MEM / pat), recursive=True):
        p = pathlib.Path(f)
        if p.name == "MEMORY.md":
            continue
        try:
            t = p.read_text(errors="ignore")
        except Exception:
            continue
        notes.append((str(p.relative_to(MEM)), tokens(p.name, t)))

# --- existing coverage: skills / workflows / gotchas names ---------------
covered = set()
for d in glob.glob(str(BOX / "skills/*")):
    covered |= set(re.split(r'[^a-z0-9]+', pathlib.Path(d).name.lower()))
for f in (glob.glob(str(MEM / "workflows/*.md")) + glob.glob(str(MEM / "gotchas/*.md"))
          + glob.glob(str(MEM / "models/*.md"))):
    covered |= set(re.split(r'[^a-z0-9]+', pathlib.Path(f).stem.lower()))

# --- theme frequency -----------------------------------------------------
theme = collections.defaultdict(list)
for rel, toks in notes:
    for w in toks:
        theme[w].append(rel)

cands = []
for w, rels in theme.items():
    rels = sorted(set(rels))
    if len(rels) >= MIN and w not in covered:
        cands.append((len(rels), w, rels))
cands.sort(reverse=True)

if not cands:
    print("[skill-suggest] no recurring uncovered themes (min=%d)" % MIN)
    raise SystemExit(0)

print("[skill-suggest] ranked candidate themes (min=%d):" % MIN)
for n, w, rels in cands[:12]:
    print(f"  {n:2d}  {w}")

if DRY:
    raise SystemExit(0)

SUGG.mkdir(parents=True, exist_ok=True)
today = datetime.date.today().isoformat()
written = 0
for n, w, rels in cands[:6]:
    dest = SUGG / f"{w}.md"
    if dest.exists():
        continue
    body = textwrap.dedent(f"""\
    ---
    status: proposed            # proposed | accepted | rejected
    theme: {w}
    notes: {n}
    generated: {today}
    ---

    # Suggestion: consolidate "{w}" knowledge

    **{n} vault notes** touch `{w}` but no skill/workflow/gotcha owns it yet.
    This is a candidate for a curated workflow or a `/`-skill. **Review and either
    promote the stable facts (via /memory-capture) or set `status: rejected`.**

    ## Contributing notes
    """)
    for r in rels:
        body += f"- [`{r}`]({os.path.relpath(MEM / r, dest.parent)})\n"
    body += textwrap.dedent(f"""
    ## Proposed action (pick one, then edit)
    - [ ] New gotcha  `memory/gotchas/{w}.md` — if it's a repeated foot-gun
    - [ ] New workflow `memory/workflows/{w}.md` — if it's a repeatable procedure
    - [ ] New skill    `skills/{w}/SKILL.md` — if it deserves a slash command
    - [ ] Reject — noise / already implicit elsewhere

    _Auto-drafted by `memory/bin/skill-suggest.sh`; detect → draft → you approve._
    """)
    dest.write_text(body)
    written += 1
    print(f"  + wrote {dest.relative_to(MEM)}")

print(f"[skill-suggest] wrote {written} new stub(s) → {SUGG.relative_to(MEM)}/  (review before acting)")
PY
