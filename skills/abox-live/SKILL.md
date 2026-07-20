---
name: abox-live
category: orchestration
description: "Monitor Claude agents running inside your containers, from the host or phone — the single entry point (absorbed the old `abox`). Two views: LIVE (process-based, exact container attribution via cgroup) and `ps` (transcript-file view incl. finished sessions). Read any session (`tail`/`watch`/`path`), and act on live ones (`stop`/`say`/`web`). Use when asked 'what are my agents doing', 'list container agents', 'which sessions are live', 'what container is X running in', 'is anything stuck/blocked', 'show me what <session> is doing', or to check on / talk to background container work."
---

# abox-live — the single container-agent monitor (live sessions ↔ containers)

In-container `claude` (billed via the AMD gateway API key) writes its transcript to
`/root/.claude/projects/<proj>/<id>.jsonl`, bind-mounted to the host's
`~/.claude/projects/...`. So the host watches every container agent by reading those
JSONL files — no docker exec, no resume needed for monitoring.

**This is the one entry point.** It replaced the old standalone `abox` CLI, whose
reader engine (`ps`/`tail`/`watch`/`path`) now lives here (`abox_report.py`,
`abox_parse.py`). Two complementary views:

- **`ps`** — lists transcript **files**, so it includes **finished** sessions; but in
  the legacy shared dir it cannot attribute the container and prints `shared?`.
- **default (live)** — starts from running **processes**, so container attribution is
  **exact** even in the shared dir:

```
claude PID ──► /proc/PID/cgroup ──► docker id ──► container name   (always exact)
claude PID ──► /proc/PID/cwd    ──► project slug ──► session .jsonl (see evidence)
```

## Run

```bash
abox-live               # DEFAULT: LIVE table, my containers ↔ sessions (exact attribution)
abox-live <name>        # only live sessions in the container matching <name> (substring ok)
abox-live --all         # include other people's containers (akao_*, soga_*, …)
abox-live -v            # + each session's latest turn
abox-live --pid         # add the owning PID column
abox-live --group       # grouped-by-container view instead of the table
abox-live --json        # machine-readable (full session id + transcript path; has "mine")
# reader engine (absorbed from `abox`):
abox-live ps [min]      # ALL recent sessions incl. finished (transcript-file view; may show shared?)
abox-live tail <id> [n] # last n turns of a session (id prefix ok)
abox-live watch <id>    # live-follow a session (Ctrl-C to stop)
abox-live path <id>     # print the session's JSONL path
# act on a live session:
abox-live stop <id>     # stop a live session (prints the target; needs --yes to act)
abox-live say <id> "msg"  # send a follow-up to that session via `claude --resume`
abox-live web           # LOCAL web UI at http://127.0.0.1:8848 — click a session, read it
# remote dashboard (phone / laptop Claude):
abox-live publish       # rebuild the stable HTML + print the Artifact params → then call the Artifact tool
abox-live snapshot -o F # self-contained HTML file for hand-off (SendUserFile), NOT published
```

## How to answer common asks

- **"list agents" / "what's running"** → `abox-live` (live, exact container). Add
  `abox-live ps` if the user wants finished sessions too.
- **"what is <session> doing" / "show me the aiter one"** → `abox-live tail <id> 8`
  (resolve the id from the table first).
- **"is anything stuck / waiting for input"** → `abox-live -v`; a session whose latest
  is a question or `[Request interrupted…]` and is idle is likely waiting; confirm with
  `tail`.

## Talking to a session (`abox-live say`)

Send a follow-up instruction to a specific agent and get its reply, by continuing the
conversation with `claude --resume <session> -p "<msg>"` run in the right place:

```bash
abox-live say fda052ce "往下繼續找 sglang 今天的重點"
abox-live say 5a699b21 "跳過 conc128，先把結果貼出來" --container jacky-…-inferencemax-0714
abox-live say fda052ce "…" --force        # resume even though a live process holds it
abox-live say fda052ce "…" --dangerous    # pass --dangerously-skip-permissions
```

How it routes and guards:
- **cwd** comes from the transcript's recorded `cwd` field (authoritative), not the
  dir-name slug.
- **Container sessions** resume via `docker exec` into the owning container, gated by
  the same bind-mount ownership check as `bridge.sh` (refuses other users' containers).
- **Host sessions** run as **root** with `HOME=/home/yichiche` — host transcripts are
  usually `root:root` (a container wrote them through the bind-mounted `~/.claude`), so
  running as `yichiche` gives "No conversation found". This was the one non-obvious bug
  in building it.
- **Refuses a session with a live process** unless `--force`: two `claude` processes
  appending to one transcript interleave and corrupt it. Stop it first, or `--force`.
- Default timeout 600s; the reply is printed. Long tasks may outlive it — then poll
  with `abox-live tail`.

This overlaps `bridge.sh exec`, but that starts a **fresh** headless agent; `say`
**continues an existing session** so the agent keeps all its context.

## Stopping a session

```bash
abox-live stop 259baf41         # shows container / cwd / pid / last line, kills nothing
abox-live stop 259baf41 --yes   # SIGTERM
abox-live stop 259baf41 --yes --force   # SIGKILL if it ignores SIGTERM
```

It **refuses** when the row's evidence is `ambiguous` or `none` — the pid could belong
to a different session and killing the wrong agent loses in-flight work — and prints
the pid so the user can act deliberately instead. On `cwd+mtime` it warns but proceeds.

**The published dashboard cannot do this.** An artifact page has no channel back to the
box (only `downloads` and `mcp` capabilities exist here, and no connectors are
attached), so each session block just prints the copy-paste `abox-live stop` command.
Do not promise a working kill button on the web page.

## Remote dashboard (phone / laptop) — `abox-live publish`

The user tracks agent progress away from their desk. The GPU box has **no usable port
forwarding**, so the only thing reachable from a phone is a published Artifact. When the
user says **"update / refresh my dashboard"**, "publish the dashboard", `/abox-live
publish`, or asks to see their agents on **mobile / laptop Claude**, do this two-step flow:

**1. Regenerate + get the params:**

```bash
abox-live publish        # rebuilds the stable HTML, chowns it, prints the Artifact params
```

`publish` owns the stable path, the stable URL, and the title/favicon/description, and
folds in sessions that finished <180 min ago. It ends with a `=== PUBLISH ARTIFACT ===`
block listing `file_path` / `url` / `title` / `favicon` / `description`.

**2. Call the `Artifact` tool with exactly those printed params** — always including
`url=`. A shell script *cannot* publish; only the Artifact tool can, which is why the
script hands you the parameters instead of doing it itself.

- **URL (stable, never changes):** https://claude.ai/code/artifact/605cb93f-db89-4777-bbfb-6a4a9f50276c
- **Source file (stable path):** `/home/yichiche/agent-scratch/abox-dash/abox-dashboard.html`

Omitting `url` from a conversation that did not originally publish it mints a *new* URL
and the user's phone bookmark goes stale — the `publish` block prints the right `url` so
you never have to remember it. To change the URL / title / favicon, edit the constants at
the top of `abox_snapshot.py` (single source of truth), not the call site.

The user approved publishing **full transcripts** to claude.ai for this dashboard
(artifacts are private to their account). That approval covers this dashboard; it is
not a blanket approval to publish other internal content.

Keep `<title>` exactly `abox dashboard` — it is republished repeatedly to the same
artifact, and the tag (not the `title` parameter) names it, so a timestamped title would
rename it every run. `publish` already emits the constant title.

**Auto-refresh (optional):** to keep the artifact current while away, run `abox-live
publish` on a `/loop` (e.g. every 10–15 min) and re-call Artifact with the same `url`
each iteration — same URL, so the phone bookmark keeps showing fresh data.

## Dashboard file (`abox-live snapshot`) — offline / no-upload path

**This box has no usable port forwarding to the user's laptop; the Claude conversation
is the only channel.** So do not tell them to open a URL or set up `ssh -L`. Build a
self-contained file and hand it over the conversation instead:

```bash
abox-live snapshot -o /path/out.html            # live sessions + transcripts, embedded
abox-live snapshot --recent 180                 # also fold in transcripts that ended <180m ago
abox-live snapshot --turns 200                  # more turns per session (default 120)
abox-live snapshot --all                        # include other people's containers
```

Then deliver it with **SendUserFile** (`display: "render"`). One HTML file, no server,
no network: a summary table at the top, then one `<details>` block per session holding
its full transcript (tool calls, results, thinking).

**All CONTENT must render with zero JavaScript.** The first version drew the whole page
from JS and came out completely blank in this user's file viewer, which sandboxes inline
scripts. `<details>`/`<summary>` gives click-to-expand natively.

The only JS allowed is *progressive enhancement* that the page works without — today
that is the copy-to-clipboard button on each stop command, where the command itself is
already visible, selectable `<code>` text. Never move content, state, or navigation
into script.

Keep it small — it travels through the conversation. Defaults (120 turns/session,
4000 chars/block, 400 KB/session) land ~0.4 MB for ~9 sessions. Raise `--turns` only
when asked.

It is a **snapshot**, not live — regenerate and resend to refresh. Say that plainly
rather than letting them think the page updates itself.

Do **not** publish this as an Artifact: transcripts are internal work product and an
Artifact would upload them to claude.ai.

## Web UI (`abox-live web`) — only when ports actually work

A click-through view of the same data: sessions in a sidebar (state dot, container,
cwd, latest line), transcript in the main pane. Filter box narrows by container /
session / cwd; the list auto-refreshes every 8s and the open transcript refreshes with
it, holding your scroll position unless you're already at the bottom.

```bash
abox-live web                  # foreground, Ctrl-C to stop
abox-live web --port 9000      # different port
abox-live web --all            # include other people's containers
nohup abox-live web >/tmp/aboxweb.log 2>&1 &   # background
```

**Binds 127.0.0.1 only** and reads transcripts straight off disk on each request —
nothing leaves the box. This is deliberate: session transcripts are internal work
product, so they must not be published to an external host (no Artifact, no tunnel)
unless the user explicitly asks for that.

### Reaching it from a laptop — the usual "can't connect"

`127.0.0.1:8848` in a laptop browser hits **the laptop**, not the GPU box. The server
prints the fix on startup; forward the port first:

```bash
ssh -L 8848:127.0.0.1:8848 yichiche@<gpu-host>     # then open the printed URL
# or in VS Code Remote: PORTS panel -> Forward a Port -> 8848
```

Do **not** "fix" this by binding wider. `--host 0.0.0.0` exists but the box is shared
with other people's containers, and it would expose every transcript to the network.

### Token

The GPU box is multi-user, so `127.0.0.1` is *not* private — any local account could
read the transcripts. Startup mints a random token and prints it in the URL
(`http://127.0.0.1:8848/?k=…`); the page then stores it in a cookie, so open the printed
URL once and normal navigation works. `--no-token` disables it. The token changes on
every restart.

Caps per transcript: last 300 turns, 12000 chars per block — enough for monitoring, and
keeps huge sessions (some are >2 MB) responsive.

The default is a flat table — one row per live session, columns
`STATE · CONTAINER · SESSION · AGE · EV · CWD` — sorted by container so a container's
sessions sit together, containers before `host`. The container name is **repeated on
every row** (not blanked) so the table stays greppable:

```bash
abox-live | grep my-new-container      # every session in one container
abox-live | grep 🟢                    # only what's actively working
```

CONTAINER and SESSION are both printed in full so a row can be copied straight into
`docker exec` / `abox-live tail`. `tail` accepts an 8-char prefix.

**Ownership** is decided by bind mount, not by name: a container is mine when it mounts
my home (`docker inspect` → `.Mounts.Source` under `/home/<me>`). Name prefixes like
`jacky-*` / `yct_*` are not reliable and are never used for this. Host sessions always
count as mine. Everything else is hidden by default, with a one-line count of what was
suppressed.

## Typical loop (launch → watch)

This skill is the *watch* half of spinning up work in containers:

1. Start a container and run `claude` inside it.
2. `abox-live` — the new container appears once its session writes a first turn.
   Until then it may show `❔` / no session, which is expected, not an error.
3. `abox-live | grep <container>` to isolate it; 🟢 means it is working, ⚪ means it has
   gone quiet and is probably waiting on you.
4. `abox-live tail <session>` to see what it actually said.

Needs one `sudo -n` (reads `/proc/<pid>/cgroup` and root-owned transcripts). The
wrapper handles it; the script itself must run as root.

## Reading the output

State: 🟢 transcript written <2 min ago (actively working) · ⚪ transcript idle, so the
agent is likely **waiting on you** (confirm with `abox-live tail <id>`) · ❔ no transcript
matched.

**The `EV` column is the point — do not report a session id without it.** The table
prints short codes; `--group` and `--json` print the long form.

| EV | long form | meaning | trust |
|---|---|---|---|
| `exact` | `resume-flag` | id read from `--resume=<uuid>` on the cmdline | exact |
| `exact` | `child-proc` | id read from a descendant tool-call process | exact |
| `infer` | `cwd+mtime` | newest unclaimed transcript for that cwd | inferred |
| `ambig` | `ambiguous` | N live procs share one cwd — see below | set right, row may be permuted |
| `-` | `none` | no transcript found for that cwd | container still exact |

A transcript is only ever assigned to one process, and exact evidence always wins, so
an `ambiguous` group of N processes gets the N most recent sessions for that cwd. The
**set** of live sessions is right and the **container** is right; which pid owns which
row inside that group is a guess. This is normal on the host, where several sessions
(and any subagents) share `/home/yichiche`.

`❔ / none` rows are typically **another user's container** (e.g. `akao_*`) — its
`~/.claude` is not bind-mounted into yours, so the transcript is unreadable from here.
The container name is still exact, which is usually the question being asked.

## Reporting to the user

**Default: show the table.** Print the `abox-live` table verbatim as the answer — this
user wants the table, not a prose rewrite of it. A one-line lead above it is fine
("5 live sessions, 2 working") and a short note below for anything the table can't show
(e.g. "the `inferencemax` rows are idle — probably waiting on you"), but do not replace
the table with sentences.

Say "inferred" out loud whenever a row you quote is `cwd+mtime` or `ambiguous`. To
explain what an agent is *doing*, follow with `abox-live tail <id>` — the live table
answers *where*, `tail`/`ps` answer *what*.

When the user names a container, use `abox-live <name>` to scope the table to it.

## Limits

- Read-only. It cannot steer or resume an agent.
- Transcripts are appended per turn, so a mid-turn agent shows its last completed turn
  (≈ one turn of lag); AGE is transcript-write age, not process age.
- Sessions whose process has exited are not shown at all — use `abox-live ps` for history.

## Container auth (`claude-container-auth.sh`)

Run **inside a container** during bootstrap (replaces the credential-stripping part of
`claude-code-key.sh` safely). Wires API-key auth via a container-local `CLAUDE_CONFIG_DIR`
and symlinks shared skills/settings — host subscription login is never touched.

```bash
bash ~/agent-box/skills/abox-live/claude-container-auth.sh
```

From `container-dep.sh`, call this instead of `claude-code-key.sh` when migrating.

## Related

- **`/create-new-rc`** — persistent Remote Control session for phone monitoring
- `gpu-to-containers` — same cgroup-join trick, for GPU owners
- Implementation: `~/agent-box/skills/abox-live/`
