#!/usr/bin/env python3
"""pr-ci-watch engine — watchlist, sweep, guarded re-run, conflict notice.

Split of responsibilities, on purpose:

  * This script does the mechanical part — read PR state and checks via `gh`,
    decide what is *in scope*, enforce the re-run cap and the once-per-head-SHA
    conflict-comment rule, and perform the two mutations.
  * It never decides *whether* a red CI deserves a re-run. That judgement comes
    from `/ci-analysis`, which is an agent procedure, not a script. `sweep`
    stops at a TRIAGE request; `apply-verdict` resumes once the agent has a
    verdict.

Nothing mutates without --apply.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_DEFAULT = "sgl-project/sglang"

# --- scope -----------------------------------------------------------------
# NVIDIA/generic CI only. Decided per *workflow*, not per job name: the user's
# own example `PR Test Base / base-a-test-cpu` is a CPU job inside the NVIDIA
# workflow and must stay in scope. Exclude-list (rather than an allow-list) so
# a newly added NVIDIA workflow is picked up without a code change.
VENDOR_RE = re.compile(
    r"\b(AMD|ROCm|Arm64|aarch64|MLX|MUSA|NPU|Ascend|XPU|Xeon|Gaudi|HPU|TPU|Mori)\b",
    re.I,
)
ADMIN_RE = re.compile(
    r"^(Lint|Auto Label|PR States|Release|Close Inactive|Nightly|Deploy|Docs)",
    re.I,
)
# sgl-router is Rust-only and has no NVIDIA test content; it is not what the
# watch is for.
EXTRA_SKIP_RE = re.compile(r"sgl-router", re.I)

# Jobs that aggregate other jobs rather than running tests. Same taxonomy as
# skills/ci-analysis/SKILL.md Phase 2.5. A workflow whose only in-scope failures
# are these has no root cause *here* — the real failure is in a skipped job or
# an out-of-scope vendor workflow, and re-running the gate just re-fails it.
ROLLUP_RE = re.compile(r"(-finish$|\bpr-gate\b|Standard Test Results|^finish$)", re.I)
# Watchers are NOT rollups. A rollup only mirrors other jobs' results, so
# re-running it can never change the outcome. A watcher polls the GitHub API and
# dies on its own — HttpError 5xx, timeout — while the jobs it watches are still
# green and running. That failure IS re-runnable, and lumping the two together
# made an identical `wait-for-base-b` HttpError get re-run on one PR and written
# off as out-of-scope on another. Watchers go to triage so the log decides.
WATCHER_RE = re.compile(r"^(wait-for-|check-pr-test-health)", re.I)

MAX_RERUNS = 2  # per (PR, head SHA, workflow)
SWEEP_DEDUP_MINUTES = 30
ACTIONS = ("re-run", "code-fix", "merge-main", "wait-upstream", "out-of-scope")

PRIORITIES = ("P0", "P1", "P2")
# Reporting label, not the sweep cadence. Defaults off the track so you only
# override when a P-level and a cadence genuinely disagree.
TRACK_PRIORITY = {"high": "P0", "regular": "P1"}

# Short status token for the report line, e.g. <CI clear>.
CI_TOKEN = {
    "green": "CI clear",
    "conflict": "conflict",
    "awaiting-triage": "CI red",
    "re-run": "CI rerun",
    "code-fix": "CI fail",
    "merge-main": "merge main",
    "wait-upstream": "blocked",
    "out-of-scope": "CI n/a",
}

AGENT_BOX = Path(__file__).resolve().parents[2]
HOST_HOME = Path(os.environ.get("AGENT_BOX_HOST_HOME", AGENT_BOX.parent))
DATA_DIR = Path(
    os.environ.get("PR_CI_WATCH_DIR")
    or Path(os.environ.get("AGENT_SCRATCH_DIR", HOST_HOME / "agent-scratch"))
    / "pr-ci-watch"
)
WATCHLIST = DATA_DIR / "watchlist.json"
STATE = DATA_DIR / "state.json"
SWEEPS = DATA_DIR / "sweeps"
LOG = DATA_DIR / "sweep.log"


# --- small utils -----------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        die(f"{path} is unreadable ({e}). Fix or delete it; refusing to overwrite.")


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)  # atomic; the dashboard reads these files concurrently


def monitoring_enabled(st: dict) -> bool:
    return bool(st.get("_config", {}).get("enabled", True))


def set_monitoring(st: dict, on: bool) -> None:
    st.setdefault("_config", {})["enabled"] = on
    st["_config"]["toggled_at"] = now()


def priority_of(meta: dict) -> str:
    return meta.get("priority") or TRACK_PRIORITY.get(meta.get("track", "regular"), "P1")


def ci_token(s: dict) -> str:
    action = s.get("last_action")
    if not action:
        return "CI ?"
    return CI_TOKEN.get(action, action)


def report_text(wl: dict, st: dict) -> str:
    """The paste-into-Teams block.

    <P0><CI clear><PR39987>[AMD] Tune Qwen3.5 TP4 GDN recurrent launch on gfx950
    ~5% P90 E2E improvement at TP4 conc4 agent mode
    """
    lines = []
    for pr, meta in sorted(
        wl.items(), key=lambda kv: (priority_of(kv[1]), int(kv[0]))
    ):
        s = st.get(pr, {})
        title = s.get("title") or "(not swept yet — title unknown)"
        lines.append(f"<{priority_of(meta)}><{ci_token(s)}><PR{pr}>{title}")
        note = (meta.get("note") or "").strip()
        if note:
            lines.append(note)
    return "\n".join(lines)


def die(msg: str):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def log_line(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(f"{now()} {msg}\n")


def gh(args: list[str], check: bool = True) -> str:
    """Run gh with GH_TOKEN cleared.

    The GH_TOKEN in the environment is a fine-grained PAT that the LMSYS
    enterprise blocks on token lifetime; clearing it falls through to the OAuth
    token from `gh auth login`. See _shared/repo-config.md.
    """
    env = dict(os.environ, GH_TOKEN="")
    proc = subprocess.run(
        ["gh", *args], capture_output=True, text=True, env=env, timeout=180
    )
    if check and proc.returncode != 0:
        die(f"gh {' '.join(args[:3])}… failed: {proc.stderr.strip()}")
    return proc.stdout


def gh_try(args: list[str]) -> tuple[int, str, str]:
    """gh that reports failure instead of exiting — for calls with expected
    non-fatal errors, e.g. re-running a workflow that is still in progress."""
    env = dict(os.environ, GH_TOKEN="")
    p = subprocess.run(["gh", *args], capture_output=True, text=True,
                       env=env, timeout=180)
    return p.returncode, p.stdout, p.stderr.strip()


def parse_pr(ref: str) -> str:
    """Accept a full PR URL, `#41870`, or a bare number."""
    ref = ref.strip()
    m = re.search(r"/pull/(\d+)", ref) or re.fullmatch(r"#?(\d+)", ref)
    if not m:
        die(f"cannot read a PR number out of {ref!r}")
    return m.group(1)


def run_id_of(link: str) -> str | None:
    m = re.search(r"/actions/runs/(\d+)", link or "")
    return m.group(1) if m else None


def in_scope(workflow: str) -> bool:
    w = workflow or ""
    return not (VENDOR_RE.search(w) or ADMIN_RE.match(w) or EXTRA_SKIP_RE.search(w))


# --- watchlist commands ----------------------------------------------------


def cmd_add(a) -> None:
    wl = load(WATCHLIST, {})
    track = "high" if a.high else "regular"
    added = []
    for ref in a.refs:
        pr = parse_pr(ref)
        prev = wl.get(pr, {})
        wl[pr] = {
            "track": track,
            "added": prev.get("added", now()),
            "note": a.note or prev.get("note", ""),
            "repo": a.repo,
        }
        if a.priority or prev.get("priority"):
            wl[pr]["priority"] = a.priority or prev["priority"]
        added.append(f"#{pr} [{track}/{priority_of(wl[pr])}]")
    save(WATCHLIST, wl)
    print("watching: " + ", ".join(added))
    log_line(f"add {' '.join(added)}")


def cmd_remove(a) -> None:
    wl = load(WATCHLIST, {})
    st = load(STATE, {})
    for ref in a.refs:
        pr = parse_pr(ref)
        if wl.pop(pr, None) is None:
            print(f"#{pr} was not on the watchlist")
        else:
            print(f"removed #{pr}")
            log_line(f"remove #{pr}")
        st.pop(pr, None)
    save(WATCHLIST, wl)
    save(STATE, st)


def cmd_list(a) -> None:
    wl = load(WATCHLIST, {})
    st = load(STATE, {})
    if a.json:
        print(json.dumps({"watchlist": wl, "state": st}, indent=2, sort_keys=True))
        return
    if not wl:
        print("watchlist is empty — add one with: watch.py add <pr url|number> [--high]")
        return
    print(f"{'PR':>7}  {'track':<8} {'merge':<12} {'action':<13} {'last swept':<21} title")
    for pr, meta in sorted(wl.items(), key=lambda kv: int(kv[0])):
        s = st.get(pr, {})
        print(
            f"{'#' + pr:>7}  {meta['track']:<8} {s.get('mergeable', '?'):<12} "
            f"{s.get('last_action', '-'):<13} {s.get('last_sweep', '-'):<21} "
            f"{(s.get('title') or '')[:60]}"
        )


# --- sweep -----------------------------------------------------------------


def conflict_comment(author: str, sha: str, branch: str) -> str:
    return (
        f"@{author} heads-up — this PR now conflicts with `main` "
        f"(`mergeStateStatus: DIRTY`) as of `{sha[:8]}`, so CI cannot run to "
        f"completion.\n\n"
        f"Could you merge or rebase `main` into `{branch}`? Happy to help if the "
        f"conflict lands in AMD/ROCm code."
    )


def pr_snapshot(pr: str, repo: str) -> dict:
    fields = (
        "state,mergeable,mergeStateStatus,headRefOid,headRefName,author,title,url,isDraft"
    )
    data = json.loads(gh(["pr", "view", pr, "--repo", repo, "--json", fields]))
    if data.get("mergeable") == "UNKNOWN":
        # GitHub computes mergeability lazily; one re-query is enough in practice.
        time.sleep(15)
        data = json.loads(gh(["pr", "view", pr, "--repo", repo, "--json", fields]))
    return data


def failed_in_scope(pr: str, repo: str) -> tuple[dict, dict]:
    """-> ({workflow: {run_id, jobs[]}}, tally)."""
    raw = gh(
        [
            "pr",
            "checks",
            pr,
            "--repo",
            repo,
            "--json",
            "name,workflow,state,bucket,link",
        ],
        check=False,
    )
    checks = json.loads(raw) if raw.strip() else []
    groups: dict[str, dict] = {}
    tally = {"pass": 0, "fail": 0, "pending": 0, "skipping": 0}
    for c in checks:
        if not in_scope(c.get("workflow", "")):
            continue
        bucket = c.get("bucket", "")
        tally[bucket] = tally.get(bucket, 0) + 1
        if bucket != "fail":
            continue
        wf = c["workflow"]
        name = c.get("name", "?")
        g = groups.setdefault(
            wf,
            {"run_id": run_id_of(c.get("link", "")), "jobs": [],
             "gate_jobs": [], "watcher_jobs": []},
        )
        if WATCHER_RE.match(name):
            g["watcher_jobs"].append(name)
        elif ROLLUP_RE.search(name):
            g["gate_jobs"].append(name)
        else:
            g["jobs"].append(name)
        if not g["run_id"]:
            g["run_id"] = run_id_of(c.get("link", ""))
    for g in groups.values():
        # Only a pure rollup failure is unactionable. A failed watcher still
        # needs a human/agent to read why it died.
        g["gate_only"] = not g["jobs"] and not g["watcher_jobs"]
    return groups, tally


def cmd_sweep(a) -> None:
    wl = load(WATCHLIST, {})
    st = load(STATE, {})
    if not monitoring_enabled(st) and not a.pr:
        # The dashboard's OFF switch. Honoured here rather than by unregistering
        # cron, so pausing works instantly from the browser/phone with no Claude
        # turn — and a scheduled sweep that fires while paused is a clean no-op.
        toggled = st.get("_config", {}).get("toggled_at", "?")
        print(f"monitoring is PAUSED (since {toggled}) — no sweep. "
              f"Resume on the dashboard or with `watch.py resume`.")
        return
    if a.pr:
        targets = {parse_pr(p): wl.get(parse_pr(p), {"track": "high", "repo": a.repo})
                   for p in a.pr}
    else:
        targets = {
            pr: m for pr, m in wl.items()
            if a.track in ("all", m.get("track", "regular"))
        }
    if not targets:
        print(f"no PRs on the '{a.track}' track")
        return

    dry = "" if a.apply else "  [DRY RUN — no mutations; pass --apply]"
    print(f"sweep track={a.track} prs={len(targets)} at {now()}{dry}\n")

    report: list[dict] = []
    triage: list[str] = []
    for pr in sorted(targets, key=int):
        meta = targets[pr] or {}
        repo = meta.get("repo") or a.repo
        s = st.setdefault(pr, {})

        last = parse_ts(s.get("last_sweep"))
        if (
            not a.force
            and not a.pr
            and last
            and datetime.now(timezone.utc) - last < timedelta(minutes=SWEEP_DEDUP_MINUTES)
        ):
            print(f"#{pr}  skipped — swept {s['last_sweep']} (<{SWEEP_DEDUP_MINUTES}m ago)")
            continue

        snap = pr_snapshot(pr, repo)
        sha = snap.get("headRefOid", "")
        author = (snap.get("author") or {}).get("login", "")
        s.update(
            title=snap.get("title", ""),
            author=author,
            url=snap.get("url", ""),
            head_sha=sha,
            mergeable=snap.get("mergeable", "?"),
            merge_state=snap.get("mergeStateStatus", "?"),
            state=snap.get("state", "?"),
            is_draft=snap.get("isDraft", False),
            last_sweep=now(),
        )
        row = {"pr": pr, "title": s["title"], "author": author, "sha": sha[:8]}

        if snap.get("state") != "OPEN":
            print(f"#{pr}  {snap.get('state')} — dropping from watchlist")
            if a.apply:
                wl.pop(pr, None)
            row["outcome"] = f"closed:{snap.get('state')}"
            report.append(row)
            continue

        # 1. conflict, once per head SHA
        if snap.get("mergeable") == "UNKNOWN":
            # Still not computed after a re-query. Conflict status is unknowable
            # this sweep, but the CI checks below are still valid, so carry on
            # and let the next sweep catch a conflict.
            print(f"#{pr}  mergeable still UNKNOWN after re-query — "
                  f"conflict check deferred to next sweep")
        if snap.get("mergeable") == "CONFLICTING":
            if s.get("conflict_comment_sha") == sha:
                print(f"#{pr}  CONFLICTING — already notified for {sha[:8]}")
                row["outcome"] = "conflict-already-notified"
            elif not author:
                print(f"#{pr}  CONFLICTING — no author login resolved, skipping comment")
                row["outcome"] = "conflict-no-author"
            else:
                body = conflict_comment(author, sha, snap.get("headRefName", "?"))
                if a.apply:
                    # gh prints the comment URL; keep it so the dashboard can
                    # link to the actual proof rather than just claiming it.
                    out = gh(["pr", "comment", pr, "--repo", repo, "--body", body])
                    url = next((ln.strip() for ln in out.splitlines()
                                if ln.strip().startswith("http")), "")
                    s["conflict_comment_sha"] = sha
                    s["conflict_comment_url"] = url
                    s["conflict_comment_at"] = now()
                    log_line(f"#{pr} conflict comment posted for {sha[:8]} -> @{author} {url}")
                    print(f"#{pr}  CONFLICTING — commented to @{author}  {url}")
                else:
                    print(f"#{pr}  CONFLICTING — would comment to @{author}:")
                    print("      " + body.replace("\n", "\n      "))
                row["outcome"] = "conflict-notified" if a.apply else "conflict-would-notify"
            s["last_action"] = "conflict"
            report.append(row)
            continue  # CI is meaningless until the conflict is resolved

        # 2. in-scope CI
        groups, tally = failed_in_scope(pr, repo)
        row["tally"] = tally
        if not groups:
            print(f"#{pr}  in-scope CI clean ({tally['pass']} pass, {tally['pending']} pending)")
            s["last_action"] = "green"
            s["last_verdict"] = ""
            row["outcome"] = "green"
            report.append(row)
            continue

        real = {w: g for w, g in groups.items() if not g["gate_only"]}
        print(f"#{pr}  {len(groups)} in-scope workflow(s) red "
              f"({len(real)} with a real failing job)")
        for w, g in groups.items():
            used = rerun_count(s, w, sha)
            if g["gate_only"]:
                print(f"        {w}: GATE-ONLY ({', '.join(g['gate_jobs'][:4])}) — "
                      f"no test job failed here; root cause is in a skipped job or an "
                      f"out-of-scope vendor workflow. Re-running would just re-fail.")
                continue
            if g["watcher_jobs"]:
                print(f"        {w}: WATCHER FAILED ({', '.join(g['watcher_jobs'][:3])}) "
                      f"— check the log: an API/timeout death is re-runnable, a "
                      f"watched-job failure is not.")
            if g["jobs"]:
                print(f"        {w}: {', '.join(g['jobs'][:6])}"
                      f"{' …' if len(g['jobs']) > 6 else ''}")
            print(f"           [run {g['run_id']}, reruns {used}/{MAX_RERUNS}]")
        s["failed_groups"] = groups
        if not real:
            # Every in-scope failure is an aggregation gate, so the root cause is
            # in a vendor workflow we deliberately ignore (or a job that never
            # ran). There is nothing here to re-run and nothing for /ci-analysis
            # to decide — spending a triage pass on it just burns time.
            s["last_action"] = "out-of-scope"
            s["last_verdict"] = (
                "no in-scope NVIDIA job failed; only aggregation gates are red"
            )
            s["verdict_at"] = now()
            print("        -> out-of-scope: nothing in NVIDIA scope to act on; "
                  "no triage needed")
            row["outcome"] = "out-of-scope"
            report.append(row)
            continue
        s["last_action"] = "awaiting-triage"
        row["outcome"] = "needs-triage"
        row["groups"] = groups
        triage.append(pr)
        report.append(row)

    save(WATCHLIST, wl)
    save(STATE, st)
    SWEEPS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save(SWEEPS / f"{stamp}.json", {"track": a.track, "at": now(), "rows": report})
    log_line(f"sweep track={a.track} prs={len(targets)} triage={len(triage)} apply={a.apply}")

    if triage:
        print("\n=== TRIAGE REQUIRED ===")
        print("Run /ci-analysis on each PR below, reduce its Root Cause Failures table")
        print("to one action, then record it:\n")
        for pr in triage:
            print(f"  /ci-analysis {st[pr].get('url') or pr}")
            print(f"  python3 {Path(__file__).name} apply-verdict --pr {pr} "
                  f"--action <re-run|code-fix|merge-main|wait-upstream> "
                  f"--summary \"<one line>\" --apply\n")
    else:
        print("\nno triage needed.")


# --- verdict / re-run ------------------------------------------------------


def rerun_count(s: dict, workflow: str, sha: str) -> int:
    rec = (s.get("reruns") or {}).get(workflow)
    # The cap is per head SHA: a fresh push earns a fresh budget.
    return rec["count"] if rec and rec.get("sha") == sha else 0


def cmd_apply_verdict(a) -> None:
    pr = parse_pr(a.pr)
    st = load(STATE, {})
    s = st.get(pr)
    if not s:
        die(f"#{pr} has no sweep state — run `sweep --pr {pr}` first")
    if a.action not in ACTIONS:
        die(f"--action must be one of {', '.join(ACTIONS)}")

    sha = s.get("head_sha", "")
    s["last_verdict"] = a.summary
    s["last_action"] = a.action
    s["verdict_at"] = now()
    s["verdict_sha"] = sha

    if a.action != "re-run":
        note = {
            "code-fix": "real failure attributed to this PR — NOT re-running; author must fix",
            "merge-main": "main already has the fix — NOT re-running; PR should merge main",
            "wait-upstream": "an in-scope NVIDIA job is blocked on an upstream fix",
            "out-of-scope": "no in-scope NVIDIA failure — the red is from a vendor "
                            "workflow we ignore; nothing to do here",
        }[a.action]
        print(f"#{pr}  {a.action}: {note}")
        print(f"       {a.summary}")
        save(STATE, st)
        log_line(f"#{pr} verdict={a.action} :: {a.summary}")
        return

    groups = s.get("failed_groups") or {}
    if not groups:
        die(f"#{pr} has no recorded failed in-scope workflows — re-run `sweep --pr {pr}`")

    reruns = s.setdefault("reruns", {})
    did, skipped = [], []
    for wf, g in groups.items():
        used = rerun_count(s, wf, sha)
        if g.get("gate_only") and not a.force_gates:
            skipped.append(
                f"{wf} (gate-only: only {', '.join(g.get('gate_jobs', [])[:3])} failed; "
                f"re-running an aggregation gate cannot turn it green — override with "
                f"--force-gates)"
            )
            continue
        if used >= MAX_RERUNS:
            skipped.append(f"{wf} (cap reached {used}/{MAX_RERUNS} for {sha[:8]})")
            continue
        run_id = g.get("run_id")
        if not run_id:
            skipped.append(f"{wf} (no run id parsed from check link)")
            continue
        endpoint = f"repos/{a.repo}/actions/runs/{run_id}/rerun-failed-jobs"
        if a.apply:
            rc, _, err = gh_try(["api", "-X", "POST", endpoint])
            if rc != 0:
                # "already running" is the common one: some shards are still in
                # progress, so GitHub refuses. Not an error worth losing the
                # verdict over — the next sweep retries, and the attempt is not
                # counted against the cap because nothing was re-run.
                if "already running" in err.lower():
                    why = "deferred — workflow still running, next sweep retries"
                else:
                    why = err.splitlines()[-1] if err else "rerun failed"
                skipped.append(f"{wf} ({why})")
                continue
            reruns[wf] = {"sha": sha, "count": used + 1, "at": now()}
            did.append(f"{wf} (run {run_id}, attempt {used + 1}/{MAX_RERUNS})")
        else:
            did.append(f"WOULD rerun {wf} (run {run_id}, attempt {used + 1}/{MAX_RERUNS})")

    for line in did:
        print(f"#{pr}  re-run: {line}")
    for line in skipped:
        print(f"#{pr}  skipped: {line}")
    if not did:
        print(f"#{pr}  nothing re-run.")
    if a.apply:
        save(STATE, st)
        log_line(f"#{pr} rerun {len(did)} group(s) :: {a.summary}")


# --- scheduling bookkeeping ------------------------------------------------


def cmd_arm_status(a) -> None:
    st = load(STATE, {})
    meta = st.setdefault("_arm", {})
    if a.record:
        meta["armed_at"] = now()
        meta["jobs"] = a.record
        save(STATE, st)
        print(f"recorded cron arm at {meta['armed_at']}: {a.record}")
        return
    armed = parse_ts(meta.get("armed_at"))
    if not armed:
        print("cron: NOT ARMED — run `/pr-ci-watch arm`")
        return
    age = datetime.now(timezone.utc) - armed
    days = age.days
    warn = "  <-- Claude cron jobs expire after 7 days; RE-ARM NOW" if days >= 6 else ""
    # Bookkeeping only — this script cannot see Claude's scheduler, so the word
    # "recorded" is load-bearing. CronList in Claude is the source of truth.
    print(f"cron: recorded as armed {meta['armed_at']} "
          f"({days}d {age.seconds // 3600}h ago){warn}")
    print(f"  jobs: {meta.get('jobs', '?')}")


def cmd_toggle(a) -> None:
    st = load(STATE, {})
    on = a.cmd == "resume"
    set_monitoring(st, on)
    save(STATE, st)
    print(f"monitoring {'RESUMED' if on else 'PAUSED'} at {now()}")
    log_line(f"monitoring {'resumed' if on else 'paused'}")


def cmd_report(a) -> None:
    wl, st = load(WATCHLIST, {}), load(STATE, {})
    if not wl:
        print("(watchlist is empty)")
        return
    print(report_text(wl, st))


def cmd_set(a) -> None:
    """Edit the two reporting fields: priority and the one-line note."""
    wl = load(WATCHLIST, {})
    pr = parse_pr(a.pr)
    if pr not in wl:
        die(f"#{pr} is not on the watchlist")
    if a.priority:
        if a.priority not in PRIORITIES:
            die(f"--priority must be one of {', '.join(PRIORITIES)}")
        wl[pr]["priority"] = a.priority
    if a.note is not None:
        wl[pr]["note"] = a.note
    save(WATCHLIST, wl)
    print(f"#{pr}: priority={priority_of(wl[pr])} note={wl[pr].get('note', '')!r}")


def cmd_status(a) -> None:
    st = load(STATE, {})
    on = monitoring_enabled(st)
    print(f"monitoring: {'ON' if on else 'PAUSED'}")
    cmd_arm_status(argparse.Namespace(record=None))
    print()
    cmd_list(argparse.Namespace(json=False))
    print(f"\ndata: {DATA_DIR}")


# --- cli -------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="watch.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=REPO_DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="add PR(s) by URL or number")
    s.add_argument("refs", nargs="+")
    s.add_argument("--high", action="store_true", help="high-priority track (2h)")
    s.add_argument("--note", default="")
    s.add_argument("--priority", choices=list(PRIORITIES),
                   help="report label; defaults to P0 for --high, else P1")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("remove", help="stop watching PR(s)")
    s.add_argument("refs", nargs="+")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("list", help="show the watchlist")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("sweep", help="phase A: gather state, emit triage requests")
    s.add_argument("--track", default="all", choices=["high", "regular", "all"])
    s.add_argument("--pr", nargs="*", help="sweep these PRs only (ignores dedup)")
    s.add_argument("--apply", action="store_true", help="allow mutations")
    s.add_argument("--force", action="store_true", help="ignore the 30m dedup window")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("apply-verdict", help="phase C: record /ci-analysis verdict, maybe re-run")
    s.add_argument("--pr", required=True)
    s.add_argument("--action", required=True, choices=list(ACTIONS))
    s.add_argument("--summary", required=True)
    s.add_argument("--apply", action="store_true")
    s.add_argument("--force-gates", action="store_true",
                   help="also re-run workflows whose only failures are rollup gates")
    s.set_defaults(func=cmd_apply_verdict)

    s = sub.add_parser("report", help="print the <P0><CI clear><PR…> status block")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("set", help="set a PR's report priority / note")
    s.add_argument("pr")
    s.add_argument("--priority", choices=list(PRIORITIES))
    s.add_argument("--note")
    s.set_defaults(func=cmd_set)

    for name, help_ in (("pause", "stop all sweeps"), ("resume", "re-enable sweeps")):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(func=cmd_toggle)

    s = sub.add_parser("arm-status", help="when were the cron jobs recorded as armed?")
    s.add_argument("--record",
                   help="INTERNAL: only call this in the same turn as a successful "
                        "CronCreate, or the dashboard will claim a schedule that "
                        "does not exist")
    s.set_defaults(func=cmd_arm_status)

    s = sub.add_parser("status", help="arm status + watchlist")
    s.set_defaults(func=cmd_status)

    a = p.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    a.func(a)


if __name__ == "__main__":
    main()
