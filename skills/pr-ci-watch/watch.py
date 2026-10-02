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

# This file is invoked by absolute path from cron and from the dashboard, so its
# own directory is not guaranteed to be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import perf  # noqa: E402

REPO_DEFAULT = "sgl-project/sglang"

# --- scope -----------------------------------------------------------------
# NVIDIA/generic CI only. Decided per *workflow*, not per job name: the user's
# own example `PR Test Base / base-a-test-cpu` is a CPU job inside the NVIDIA
# workflow and must stay in scope. Exclude-list (rather than an allow-list) so
# a newly added NVIDIA workflow is picked up without a code change.
VENDOR_RE = re.compile(
    r"\b(AMD|ROCm|Arm64|aarch64|MLX|MUSA|NPU|Ascend|XPU|Xeon|Gaudi|HPU|TPU|Mori|PPU)\b",
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

# The `pr-gate` job itself. Distinct from the other rollups because its failure
# has a *knowable* cause: .github/workflows/pr-gate.yml fails on exactly one of
# a handful of named steps, and each one implies a different fix. Without this
# every gated PR looks identical on the dashboard ("4 pass, 4 fail") when in
# fact one needs a label, one needs the author to click Ready for review, and
# one needs nothing but a re-run from someone with write access.
GATE_JOB_RE = re.compile(r"\bpr-gate\b", re.I)
# failing step name -> (reason token, what unblocks it, can a maintainer re-run
# fix it on its own?). Ordered: first substring match wins.
GATE_STEP_REASONS = (
    ("Block draft PR", (
        "draft",
        "PR was a draft when CI ran — mark it Ready for review, then re-run",
        False,
    )),
    ("Require run-ci label", (
        "missing-run-ci",
        "missing the `run-ci` label — add it (e.g. `/tag-and-rerun-ci`), then re-run",
        False,
    )),
    # Not a failure to chase. pr-test-extra.yml is opt-in: every PR without the
    # label shows this red gate, and treating it as a problem would put a
    # permanent false alarm on most of the watchlist.
    ("Require run-ci-extra label", (
        "opt-in-extra",
        "`PR Test Extra` is opt-in and this PR has not opted in — expected, not a failure",
        False,
    )),
    ("Require additional label", (
        "missing-label",
        "missing a workflow-specific opt-in label — add it, then re-run",
        False,
    )),
    # The one case a re-run genuinely fixes. The rate limit is evaluated against
    # the run's *triggering actor*, so when someone with write access presses
    # re-run the check is skipped outright and the gated CI finally starts.
    ("Enforce rate limit", (
        "rate-limit",
        "author is rate-limited (low-permission cooldown) — a re-run triggered by "
        "someone with write access bypasses it",
        True,
    )),
)
# Gate reasons that mean "this PR is not actually being tested", as opposed to
# `opt-in-extra`, which means "this workflow was never asked to run".
GATE_BLOCKING = {"draft", "stale-draft", "missing-run-ci", "missing-label",
                 "rate-limit"}
# Without this label the whole NVIDIA suite refuses to start, so an open PR that
# lacks it is not "waiting on CI" — it is waiting on a one-word fix.
RUN_CI_LABEL = "run-ci"

# No re-run cap. The gate on re-running is /ci-analysis: if a failure is
# attributed to the PR it returns `code fix` and nothing is re-run at all. A
# count limit on top of that would only ever block the case /ci-analysis has
# already cleared as unrelated. Attempts are still counted, per head SHA, so the
# dashboard can show how many times a workflow has been retried.
SWEEP_DEDUP_MINUTES = 30
# Backoff for an UNKNOWN mergeability, in seconds. Bounded on purpose: a sweep
# walks the whole watchlist, so this is the per-PR worst case, paid only when
# GitHub has not answered yet. Never keep the previous known value instead —
# UNKNOWN shows up precisely when the branch just changed, which is also when a
# conflict is most likely to have appeared, so stale `clean` is the one wrong
# answer that actually misleads.
MERGEABLE_RETRY_DELAYS = (5, 10, 20, 30)
ACTIONS = ("re-run", "code-fix", "merge-main", "wait-upstream", "out-of-scope")

PRIORITIES = ("P0", "P1", "P2")
# A track is a sweep cadence, and nothing else. Draft-ness is a live property of
# the PR, read fresh every sweep — filing it as a track meant the watchlist held
# a second, staler copy of a fact GitHub already answers authoritatively.
TRACKS = ("regular", "high")
# Reporting label, not the sweep cadence. Defaults off the track so you only
# override when a P-level and a cadence genuinely disagree.
TRACK_PRIORITY = {"high": "P0", "regular": "P1"}

# Short status token for the report line, e.g. <CI clear>.
# Actions that say more than the raw CI state, so they win over it in the
# report token. Everything else falls through to the Verdict.
CI_TOKEN_OVERRIDE = {
    "code-fix": "CI fail",
    "merge-main": "merge main",
    "wait-upstream": "blocked",
    # Neither is a CI result. "CI red" on a PR whose tests never started is the
    # exact misreading these two tokens exist to stop.
    "draft": "draft",
    "gated": "gated",
}
VERDICT_TOKEN = {"Pass": "CI clear", "Pending": "CI running",
                 "Fail": "CI red", "\u2014": "CI ?"}

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


# Per-PR autopilot, one switch per phase. Both default ON because that is
# exactly what every sweep did before the switches existed — an existing
# watchlist entry keeps behaving the way its owner last saw it, and only an
# explicit click changes that. Turning one off means "look at this PR, do not
# act on it": the sweep still reads and displays its CI either way.
AUTO_FIELDS = ("triage", "rerun")


def auto_on(meta: dict, field: str) -> bool:
    return bool((meta.get("auto") or {}).get(field, True))


def set_auto(meta: dict, field: str, on: bool) -> None:
    meta.setdefault("auto", {})[field] = bool(on)


UNGROUPED = "Ungrouped"


def group_of(meta: dict) -> str:
    """Free-text bucket you assign, e.g. a model name or 'debug'."""
    return (meta.get("group") or "").strip() or UNGROUPED


def all_groups(wl: dict, order: list[str] | None = None) -> list[str]:
    """Every group in use, Ungrouped last so it never heads the tab bar.

    `order` is the tab order dragged on the dashboard. It is a *preference*,
    not the source of truth: names in it that no longer have a PR drop out, and
    a group created since the last drag is appended alphabetically rather than
    vanishing, so the bar can never end up missing a tab.
    """
    seen = {group_of(m) for m in wl.values()}
    named = {g for g in seen if g != UNGROUPED}
    ranked = [g for g in (order or []) if g in named]
    rest = sorted(named - set(ranked))
    return ranked + rest + ([UNGROUPED] if UNGROUPED in seen else [])


def in_group(meta: dict, group: str | None) -> bool:
    return not group or group_of(meta) == group


def priority_of(meta: dict) -> str:
    return meta.get("priority") or TRACK_PRIORITY.get(meta.get("track", "regular"), "P1")


def ci_token(s: dict) -> str:
    """Report token. Derived from the same Verdict the table shows, so a row
    cannot say `Pass` in one place and `CI n/a` in the other."""
    if s.get("mergeable") == "CONFLICTING":
        return "conflict"
    override = CI_TOKEN_OVERRIDE.get(s.get("last_action", ""))
    if override:
        return override
    return VERDICT_TOKEN.get(ci_verdict(s), "CI ?")


TAIPEI = timezone(timedelta(hours=8))  # Asia/Taipei — no DST, so a fixed offset
                                       # is exact and needs no tzdata package.


def tw(iso: str | None) -> str:
    """UTC ISO string -> Taiwan local, e.g. '09/30 23:50'."""
    t = parse_ts(iso)
    return t.astimezone(TAIPEI).strftime("%m/%d %H:%M") if t else "never"


def tally_bits(t: dict | None) -> str:
    """`48 pass, 16 fail, 19 cancelled, 10 running, 13 queued` — zeroes omitted.

    `running` is pending minus queued, so the two never double-count. Tallies
    recorded before `queued` existed have none, which reads as "all of it is
    running" — the old behaviour, not a wrong number.
    """
    t = t or {}
    queued = t.get("queued", 0)
    counts = (
        ("pass", t.get("pass", 0)),
        ("fail", t.get("fail", 0)),
        # Counted all along, never rendered: #39575 showed "9 pass, 3 fail"
        # with 19 cancelled jobs — including every required `base-a-test-cpu`
        # shard — invisible behind it.
        ("cancelled", t.get("cancel", 0)),
        ("running", max(t.get("pending", 0) - queued, 0)),
        ("queued", queued),
    )
    return ", ".join(f"{n} {k}" for k, n in counts if n)


def failure_fingerprint(groups: dict) -> str:
    """Identity of the actionable failures, so a verdict can be known to still
    apply. Gate-only workflows are excluded: they flap as other runs finish."""
    return "||".join(sorted(
        f"{wf}:{g.get('sig', '')}" for wf, g in groups.items()
        if not g.get("gate_only")
    ))


def fingerprint_jobs(fp: str) -> set[str]:
    """The individual `workflow:job` pairs inside a fingerprint."""
    out = set()
    for part in (fp or "").split("||"):
        if ":" not in part:
            continue
        wf, _, sig = part.partition(":")
        out |= {f"{wf}:{j}" for j in sig.split("|") if j}
    return out


def verdict_still_covers(judged_fp: str, current_fp: str) -> bool:
    """True when every failure still red was already covered by the verdict.

    Exact equality was wrong in one direction. The failing set legitimately
    *shrinks* between sweeps — a flaky shard goes green on its own, a cascade
    gets demoted — and an equality test reads that as "new situation" and throws
    away a verdict that still accounts for everything left. #41134 was judged
    `wait-upstream` for shards 2, 9 and 10; 9 and 10 later passed, and the PR
    went straight back into the triage queue to be told the same thing about
    shard 2. A genuinely *new* failing job is different: nobody has judged it,
    so that correctly re-opens triage.
    """
    judged, current = fingerprint_jobs(judged_fp), fingerprint_jobs(current_fp)
    return bool(current) and current <= judged


def ci_verdict(s: dict) -> str:
    """Current in-scope CI state, not our internal bookkeeping."""
    # A draft has no CI verdict to give. Its checks are the author's scratchpad
    # and we deliberately do not read them, so anything but "—" would be a
    # claim we have not earned. The Merge column carries `draft` instead.
    if s.get("is_draft"):
        return "\u2014"
    # A blocking gate means a whole in-scope workflow never ran a single test.
    # The remaining counts are admin checks, and `Pass` off those is a claim
    # nobody earned — #41982 read `Pass` beside "4 pass, 4 fail" while both
    # NVIDIA workflows sat un-started at the gate. Unknown, not green.
    # Fail, not "—". Nothing ran, so this is not a test result; but the PR
    # cannot merge and someone has to act, and a dash reads as "no data yet"
    # and sinks the row to the bottom of the table next to the quiet ones.
    if any(d.get("blocking", d.get("reason") in GATE_BLOCKING)
           for d in (s.get("gated") or {}).values()):
        return "Fail"
    # A conflicting branch is a failure in its own right: CI cannot complete, so
    # say Fail rather than showing an empty verdict because no tally was taken.
    if s.get("mergeable") == "CONFLICTING":
        return "Fail"
    t = s.get("tally") or {}
    if not s.get("last_sweep") or not t:
        return "—"
    # A red aggregation gate on its own is not a failure — it mirrors jobs that
    # are still running, or a vendor workflow we ignore. Only a real (or
    # watcher) job failure makes the verdict Fail while work is still in flight.
    real_fail = any(
        g.get("jobs") or g.get("watcher_jobs")
        for g in (s.get("failed_groups") or {}).values()
    )
    if real_fail:
        return "Fail"
    # In-scope jobs were cancelled and nothing is running to take their place,
    # so there is no result to report. Usually the concurrency cancel:
    # `pr-test.yml` kills the whole run when a higher-priority request lands on
    # the same group, and the jobs read "Cancelled after 5m". `Pass` off the
    # handful that happened to finish first is the same unearned claim the
    # blocking-gate branch above exists to stop — #39575 read `Pass` while all
    # 11 required `base-a-test-cpu` shards sat cancelled. `Fail` for that
    # branch's reason too: nothing ran, the PR cannot merge, and a dash reads
    # as "no data yet". Gated on `pending` because a cancel is routine when a
    # push supersedes a run — there the replacement is already in flight.
    if t.get("cancel") and not t.get("pending"):
        return "Fail"
    if t.get("pending"):
        # In flight — first run or a re-run, same thing from here.
        return "Pending"
    # Every NVIDIA job finished and none failed. Aggregation gates may still be
    # red, but a gate is not a job — it is mirroring a vendor workflow this tool
    # excludes by design. Calling that `Fail` reports someone else's failure as
    # this PR's NVIDIA result. The Status column carries the nuance.
    return "Pass" if t.get("pass") else "—"


def ci_action(s: dict) -> str:
    """What needs doing about it."""
    action = s.get("last_action")
    # Two different problems that must not share a label:
    #   Solve conflict -> the branch has git conflicts with main; the author has
    #                     to resolve them before CI can even finish.
    #   Merge main     -> no conflict; main simply already contains the fix for
    #                     a CI failure and the PR is behind.
    # Nothing is asked of us, or of the author, on account of CI: the PR is not
    # finished being written. Said before the conflict branch on purpose — a
    # draft's conflicts are the author's to discover in their own time.
    if s.get("is_draft"):
        return "-"
    # Before the conflict branch: a PR whose CI never started has nothing to
    # merge-check against yet, and this is the cheaper fix of the two.
    if s.get("needs_run_ci"):
        return "Need run-ci tag"
    if s.get("mergeable") == "CONFLICTING":
        return "Solve conflict"
    if action == "merge-main":
        return "Merge main"
    if action == "code-fix":
        return "Code fix"
    if action == "re-run":
        return "CI re-run"
    if action == "awaiting-triage":
        return "Triage"
    if action == "wait-upstream":
        return "Wait upstream"
    # A re-run we fired for this head SHA that is still in flight stays visible
    # as `CI re-run`, even if a later sweep overwrote last_action (e.g. the only
    # remaining red became gate-only). Otherwise the row reads "-" while our own
    # re-run is the thing everyone is waiting on.
    sha = s.get("head_sha", "")
    if (ci_verdict(s) == "Pending"
            and any(r.get("sha") == sha for r in (s.get("reruns") or {}).values())):
        return "CI re-run"
    # Pending lives in the Verdict column now; an in-flight run we did not
    # touch needs no action from us.
    return "-"


def row_order(st: dict):
    """Display order, shared by the dashboard table and the report block so the
    two can never drift: Pass first (those are the ones you can go merge), then
    P0 -> P2, conflicts last within a priority, then your manual order, then
    PR number."""
    def key(kv):
        pr, meta = kv
        s = st.get(pr, {})
        return (
            0 if ci_verdict(s) == "Pass" else 1,
            priority_of(meta),
            # Conflicts sink to the bottom of their priority: nothing can
            # progress on them until the author rebases, so they are the least
            # useful thing to read first.
            1 if s.get("mergeable") == "CONFLICTING" else 0,
            meta.get("order", 0),  # manual nudge, only within the same bucket
            int(pr),
        )
    return key


def report_entries(wl: dict, st: dict, group: str | None = None) -> list[dict]:
    """Structured rows behind the status block, so the plain-text and the
    hyperlinked HTML renderings cannot disagree."""
    out = []
    for pr, meta in sorted(wl.items(), key=row_order(st)):
        if not in_group(meta, group):
            continue
        s = st.get(pr, {})
        out.append({
            "pri": priority_of(meta),
            "group": group_of(meta),
            "token": ci_token(s),
            "pr": pr,
            "url": s.get("url") or
                   f"https://github.com/{meta.get('repo', REPO_DEFAULT)}/pull/{pr}",
            "title": s.get("title") or "(not swept yet — title unknown)",
            # Line 2 is the PR's own performance claim, pulled from its body.
            # `watch.py set --note` overrides it when the extraction is poor or
            # the headline is something the tables do not capture.
            "note": (meta.get("note") or s.get("perf") or "").strip(),
        })
    return out


def report_text(wl: dict, st: dict, group: str | None = None) -> str:
    """The paste-into-Teams block.

    <P0><CI clear><PR39987>[AMD] Tune Qwen3.5 TP4 GDN recurrent launch on gfx950
    ~5% P90 E2E improvement at TP4 conc4 agent mode
    """
    lines = []
    for e in report_entries(wl, st, group):
        # Markdown bullets: Teams turns "- " into a real bullet, and two spaces
        # of indent into a nested one, so the plain-text flavour still reads as
        # a list wherever the rich one does not survive the paste.
        lines.append(f"- <{e['pri']}><{e['token']}><PR{e['pr']}>{e['title']}")
        if e["note"]:
            lines.append(f"  - {e['note']}")
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


_CASCADE_CACHE: dict[str, bool] = {}


def is_cascade(link: str) -> bool:
    """True if this job failed at `check-pr-test-health` — i.e. fast-fail killed
    it before it ran any test.

    Job *names* cannot reveal this: `base-b-test-2-gpu-large (5)` looks like a
    real shard whether it ran tests or was skipped by fail-fast. Only the failed
    step name distinguishes them, which costs one API call per failed job.
    Getting it wrong sends a PR to triage that has no root cause of its own.
    """
    m = re.search(r"/job/(\d+)", link or "")
    if not m:
        return False
    job_id = m.group(1)
    if job_id in _CASCADE_CACHE:
        return _CASCADE_CACHE[job_id]
    rc, out, _ = gh_try([
        "api", f"repos/{REPO_DEFAULT}/actions/jobs/{job_id}",
        "--jq", '[.steps[] | select(.conclusion=="failure") | .name] | first',
    ])
    step = out.strip() if rc == 0 else ""
    result = "check-pr-test-health" in step
    _CASCADE_CACHE[job_id] = result
    return result


_GATE_CACHE: dict[str, tuple] = {}


def gate_reason(link: str) -> tuple[str, str, bool] | None:
    """Why a failed `pr-gate` job blocked this workflow.

    -> (reason token, human sentence, re-run-fixes-it) or None if the step is
    not one we recognise. Same one-API-call-per-job shape as `is_cascade`, and
    only ever called for jobs whose name already matched `GATE_JOB_RE`.
    """
    m = re.search(r"/job/(\d+)", link or "")
    if not m:
        return None
    job_id = m.group(1)
    if job_id in _GATE_CACHE:
        return _GATE_CACHE[job_id]
    rc, out, _ = gh_try([
        "api", f"repos/{REPO_DEFAULT}/actions/jobs/{job_id}",
        "--jq", '[.steps[] | select(.conclusion=="failure") | .name] | first',
    ])
    step = out.strip() if rc == 0 else ""
    found = next((r for needle, r in GATE_STEP_REASONS if needle in step), None)
    _GATE_CACHE[job_id] = found
    return found


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
        if a.group or prev.get("group"):
            wl[pr]["group"] = (a.group or prev.get("group", "")).strip()
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
    # `body` rides along on the call we already make — the perf claim is
    # extracted from it here and only the one-line result is stored, so
    # state.json does not grow a copy of every PR description.
    fields = (
        "state,mergeable,mergeStateStatus,headRefOid,headRefName,baseRefName,"
        "author,title,url,isDraft,body,labels"
    )
    data = json.loads(gh(["pr", "view", pr, "--repo", repo, "--json", fields]))
    # GitHub computes mergeability lazily, and asking is what schedules the
    # computation — so the first answer on a recently-touched PR is routinely
    # UNKNOWN. One 15s retry was not enough: #41982 and #39575 both stored
    # UNKNOWN and then sat that way on the dashboard, while the real answer
    # (MERGEABLE for both) had been available within the minute. A stale
    # UNKNOWN survives until the next sweep, which on the regular track is a
    # whole day. Poll instead, returning the moment it resolves.
    for delay in MERGEABLE_RETRY_DELAYS:
        if data.get("mergeable") != "UNKNOWN":
            break
        time.sleep(delay)
        data = json.loads(gh(["pr", "view", pr, "--repo", repo, "--json", fields]))
    return data


def behind_by(repo: str, base: str, sha: str) -> int | None:
    """How many commits main is ahead of this head. `None` if GitHub would not say.

    Not available from `mergeStateStatus`: `BEHIND` only appears when the repo
    *requires* branches to be up to date, and sglang does not — so every
    out-of-date PR here reads `BLOCKED` or `UNSTABLE` like any other. The
    compare API is the only thing that answers the question, and the answer is
    what decides whether `Update branch` has anything to do.
    """
    if not sha:
        return None
    rc, out, _ = gh_try(["api", f"repos/{repo}/compare/{base}...{sha}",
                         "--jq", ".behind_by"])
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def failed_in_scope(pr: str, repo: str) -> tuple[dict, dict, dict]:
    """-> ({workflow: {run_id, jobs[]}}, tally, {workflow: {run_id, jobs[]}}).

    The third value is the cancelled in-scope work. It is kept apart from
    `groups` because a cancelled job is not a failure and must not be counted
    as one — but it is not nothing either, and treating it as nothing is how
    #39575 reported `Pass` with every required shard killed mid-run.
    """
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
    cancelled: dict[str, dict] = {}
    tally = {"pass": 0, "fail": 0, "pending": 0, "skipping": 0, "queued": 0}
    for c in checks:
        if not in_scope(c.get("workflow", "")):
            continue
        bucket = c.get("bucket", "")
        tally[bucket] = tally.get(bucket, 0) + 1
        # gh lumps QUEUED and IN_PROGRESS into one `pending` bucket, but they
        # mean different things to someone deciding whether to wait: queued is
        # "the runners are busy", in progress is "it is actually testing".
        # Counted as a *subset* of pending so every existing `pending` check
        # (the Verdict column, the clean-CI print) keeps working untouched.
        if bucket == "pending" and c.get("state") == "QUEUED":
            tally["queued"] += 1
        if bucket == "cancel":
            cg = cancelled.setdefault(
                c["workflow"], {"run_id": run_id_of(c.get("link", "")), "jobs": []})
            cg["jobs"].append(c.get("name", "?"))
            if not cg["run_id"]:
                cg["run_id"] = run_id_of(c.get("link", ""))
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
        # Keep the per-job URL so the dashboard can link straight to the failing
        # job instead of making you hunt for it in the Checks tab.
        if c.get("link"):
            g.setdefault("job_links", {})[name] = c["link"]
        if not g["run_id"]:
            g["run_id"] = run_id_of(c.get("link", ""))
    # Second pass: demote jobs that fail-fast killed before they ran a test.
    # Only done for jobs that still look real, so the API cost stays small.
    for g in groups.values():
        real, cascaded = [], []
        for name in g["jobs"]:
            (cascaded if is_cascade((g.get("job_links") or {}).get(name, "")) else real
             ).append(name)
        if cascaded:
            g["jobs"] = real
            g["cascade_jobs"] = cascaded
            g["gate_jobs"].extend(cascaded)

    # Third pass: ask *why* the gate said no. Only for workflows whose failure
    # is gate-shaped — a workflow with a real failing job was clearly allowed to
    # run, so its gate (if any) is just mirroring that.
    for g in groups.values():
        if g["jobs"] or g["watcher_jobs"]:
            continue
        for name in g["gate_jobs"]:
            if not GATE_JOB_RE.search(name):
                continue
            found = gate_reason((g.get("job_links") or {}).get(name, ""))
            if found:
                g["gate_reason"], g["gate_hint"], g["gate_rerunnable"] = found
                break

    for g in groups.values():
        # Only a pure rollup failure is unactionable. A failed watcher still
        # needs a human/agent to read why it died.
        g["gate_only"] = not g["jobs"] and not g["watcher_jobs"]
        # A watcher that died with no real job failure beside it died on its
        # own — API 5xx, timeout — while the jobs it watched were fine. That is
        # unambiguous infra, so it does not need a triage pass.
        g["watcher_only"] = bool(g["watcher_jobs"]) and not g["jobs"]
        # Signature of *what* failed, so a repeat after a re-run is detectable.
        g["sig"] = "|".join(sorted(g["jobs"] + g["watcher_jobs"]))
    return groups, tally, cancelled


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
    rerun: list[str] = []
    for pr in sorted(targets, key=int):
        meta = targets[pr] or {}
        repo = meta.get("repo") or a.repo
        s = st.setdefault(pr, {})

        # Dedup per mode. A read-only refresh (the dashboard button, or a bare
        # `sweep`) must never suppress a scheduled --apply sweep: the dry run
        # posts no conflict notice and re-runs nothing, so skipping the real
        # sweep behind it silently drops the work.
        last_key = "last_apply_sweep" if a.apply else "last_sweep"
        last = parse_ts(s.get(last_key))
        if (
            not a.force
            and not a.pr
            and last
            and datetime.now(timezone.utc) - last < timedelta(minutes=SWEEP_DEDUP_MINUTES)
        ):
            mode = "applied" if a.apply else "swept"
            print(f"#{pr}  skipped — {mode} {s[last_key]} (<{SWEEP_DEDUP_MINUTES}m ago)")
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
            perf=perf.extract(snap.get("body") or "") or s.get("perf", ""),
            last_sweep=now(),
        )
        # One extra API call per PR per sweep, and it is what makes the
        # `Update branch` button honest: without it the dashboard cannot tell a
        # branch that is 114 commits behind from one cut this morning.
        s["behind_by"] = behind_by(repo, snap.get("baseRefName") or "main", sha)
        if a.apply:
            s["last_apply_sweep"] = now()
        row = {"pr": pr, "title": s["title"], "author": author, "sha": sha[:8]}

        if snap.get("state") != "OPEN":
            print(f"#{pr}  {snap.get('state')} — dropping from watchlist")
            if a.apply:
                wl.pop(pr, None)
            row["outcome"] = f"closed:{snap.get('state')}"
            report.append(row)
            continue

        # 0. draft. A draft PR is still being written: its CI is the author's
        # own scratchpad and its code is not up for review yet, so there is
        # nothing here to triage and nothing we should re-run on their behalf.
        # (`pr-gate.yml` agrees — it fails `Block draft PR` outright, which is
        # why a watched draft otherwise shows up as a mysterious all-red gate.)
        # The PR stays on the watchlist and keeps being snapshotted, so the
        # sweep that follows it being marked Ready picks it straight back up.
        if snap.get("isDraft"):
            s["last_action"] = "draft"
            s["last_verdict"] = "draft PR — CI and code not checked until it is marked Ready for review"
            s["verdict_at"] = now()
            s["tally"] = {}  # a draft's red gates are not a CI verdict
            print(f"#{pr}  DRAFT — not checking CI or code until it is "
                  f"marked Ready for review")
            row["outcome"] = "draft"
            report.append(row)
            continue

        # Missing `run-ci` on an OPEN, non-draft PR. Checked against the PR's
        # labels right now, not against whichever gate step happened to fail
        # first: #41982's recorded gate failure was `Block draft PR`, so the
        # run-ci step never even got evaluated, yet that label is what actually
        # blocks it today. A draft is exempt — we leave drafts alone entirely.
        labels = {l.get("name") for l in (snap.get("labels") or [])}
        needs_run_ci = RUN_CI_LABEL not in labels
        s["needs_run_ci"] = needs_run_ci
        if needs_run_ci:
            if a.apply:
                # REST, not `gh pr edit`: that path resolves projectCards over
                # GraphQL, which now hard-errors on the Projects-classic sunset
                # and fails the whole edit even though we only want a label.
                rc, _, err = gh_try([
                    "api", f"repos/{repo}/issues/{pr}/labels",
                    "-f", f"labels[]={RUN_CI_LABEL}",
                ])
                if rc == 0:
                    s["run_ci_added_at"] = now()
                    s["needs_run_ci"] = needs_run_ci = False
                    labels.add(RUN_CI_LABEL)
                    print(f"#{pr}  added the `{RUN_CI_LABEL}` label — NVIDIA CI can "
                          f"start now")
                    log_line(f"#{pr} added {RUN_CI_LABEL} label")
                else:
                    print(f"#{pr}  could not add `{RUN_CI_LABEL}`: "
                          f"{err.splitlines()[-1] if err else 'failed'}")
            else:
                print(f"#{pr}  missing `{RUN_CI_LABEL}` — would add it (pass --apply)")

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
        groups, tally, cancelled = failed_in_scope(pr, repo)
        row["tally"] = tally
        s["tally"] = tally  # drives the Running / Pass / Fail column
        if not groups:
            print(f"#{pr}  in-scope CI clean ({tally['pass']} pass, {tally['pending']} pending)")
            s["last_action"] = "green"
            s["last_verdict"] = ""
            row["outcome"] = "green"
            report.append(row)
            continue

        real = {w: g for w, g in groups.items() if not g["gate_only"]}
        if real:
            # Remember the last genuinely failing jobs. Once a re-run is in
            # flight the live checks often show nothing but red aggregation
            # gates, and "gate only" tells you nothing about what actually
            # broke — this is what the dashboard falls back to.
            s["last_real_failure"] = {
                "at": now(),
                "sha": sha,
                "groups": {
                    w: {"jobs": g["jobs"], "watcher_jobs": g["watcher_jobs"],
                        "job_links": g.get("job_links", {}), "run_id": g["run_id"]}
                    for w, g in real.items()
                },
            }
        # A `draft` gate records what the PR was when CI ran, not what it is
        # now — marking a PR Ready re-triggers nothing, so the red gate just
        # sits there forever. This sweep's snapshot says it is Ready, so the
        # gate is stale and a re-run is the entire fix. (Reached here only when
        # the PR is not a draft: a live draft returned above.)
        for g in groups.values():
            if g.get("gate_reason") == "draft":
                g["gate_reason"] = "stale-draft"
                g["gate_rerunnable"] = True
                g["gate_hint"] = ("CI ran while this was a draft; it is Ready for "
                                  "review now, so a re-run gets past the gate")
        # A recorded gate reason describes the run that already happened. If the
        # label is missing *now*, that is what a re-run would hit next, so say
        # so rather than repeating a draft gate the PR has already outgrown.
        if needs_run_ci:
            for g in groups.values():
                if g.get("gate_reason") in GATE_BLOCKING:
                    g["gate_reason"] = "missing-run-ci"
                    g["gate_rerunnable"] = False
                    g["gate_hint"] = (f"missing the `{RUN_CI_LABEL}` label — the "
                                      f"whole NVIDIA suite refuses to start without it")
        print(f"#{pr}  {len(groups)} in-scope workflow(s) red "
              f"({len(real)} with a real failing job)")
        for w, g in groups.items():
            used = rerun_count(s, w, sha)
            if g["gate_only"]:
                if g.get("gate_reason"):
                    print(f"        {w}: GATED [{g['gate_reason']}] — {g['gate_hint']}")
                else:
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
            tried = f", re-run x{used} so far" if used else ""
            print(f"           [run {g['run_id']}{tried}]")
        # Did a workflow come back with the *same* failure after we re-ran it?
        # Re-running again would just repeat it; that is the signal to evaluate
        # `merge main` instead.
        repeats = []
        for w, g in groups.items():
            rec = (s.get("reruns") or {}).get(w)
            if not rec or rec.get("sha") != sha:
                continue
            # Intersection, not equality. The failing set legitimately shifts
            # between sweeps — a shard finishes, a cascade gets demoted — and an
            # exact-match test lets a genuine repeat slip through whenever it
            # does. What matters is whether a job we already re-ran has failed
            # again.
            before = set((rec.get("sig") or "").split("|")) - {""}
            nowset = set((g.get("sig") or "").split("|")) - {""}
            again = before & nowset
            if again:
                g["repeat_after_rerun"] = rec.get("count", 1)
                g["repeat_jobs"] = sorted(again)
                repeats.append(f"{w} (x{rec.get('count', 1)}: "
                               f"{', '.join(sorted(again))})")
        if repeats:
            print(f"        !! SAME FAILURE AFTER RE-RUN: {', '.join(repeats)} — "
                  f"stop re-running; evaluate `merge main`")

        s["failed_groups"] = groups
        # Keep the gate picture at PR level so the dashboard can say "blocked at
        # the gate, here is which one" instead of printing a bare fail count.
        # Store *every* resolved gate reason, not just the blocking ones. The
        # common case by far is `opt-in-extra`, and dropping it meant a row read
        # "59 pass, 2 fail" with no explanation anywhere — the fail count is the
        # not-opted-in Extra workflow, which we had already diagnosed and then
        # discarded. A count you cannot account for is worse than no count.
        all_gates = {
            w: {"reason": g["gate_reason"], "hint": g["gate_hint"],
                "rerunnable": g.get("gate_rerunnable", False),
                "blocking": g["gate_reason"] in GATE_BLOCKING,
                "jobs": list(g.get("gate_jobs", []))}
            for w, g in groups.items() if g.get("gate_reason")
        }
        s["gated"] = all_gates
        gated = {w: d for w, d in all_gates.items() if d["blocking"]}
        if gated:
            print(f"        !! BLOCKED AT THE GATE: "
                  + "; ".join(f"{w} [{d['reason']}]" for w, d in gated.items())
                  + " — these workflows never ran a single test")
        # A gate that only this account's write access can clear is not a triage
        # question: /ci-analysis has no log to read, because no job ran.
        rerunnable_gate = {w: d for w, d in gated.items() if d["rerunnable"]}
        if rerunnable_gate and not real:
            s["last_action"] = "re-run"
            s["last_verdict"] = "; ".join(
                f"{w}: {d['hint']}" for w, d in rerunnable_gate.items()
            )
            s["verdict_at"] = now()
            print("        -> re-run: gate rejected the author, not the code; "
                  "a write-access re-run clears it")
            row["outcome"] = "auto-re-run-gate"
            rerun.append(pr)
            report.append(row)
            continue
        # The rest of the blocking gates need a human act first — mark ready for
        # review, add a label. Re-running changes nothing until that happens, so
        # do not spend a triage pass or a re-run attempt on them.
        if gated and not real:
            s["last_action"] = "gated"
            s["last_verdict"] = "; ".join(f"{w}: {d['hint']}" for w, d in gated.items())
            s["verdict_at"] = now()
            print("        -> gated: needs a label or Ready-for-review first; "
                  "re-running now would just re-fail the gate")
            row["outcome"] = "gated"
            report.append(row)
            continue
        # In-scope jobs were cancelled and nothing is in flight to replace
        # them. Not a test result and not nothing: the jobs have to actually
        # run before this PR has a verdict, and no log exists for /ci-analysis
        # to read, so this is a re-run and not a triage question. Placed ahead
        # of the out-of-scope branch because that branch's premise — "nothing
        # here to re-run" — is exactly what a cancelled run disproves.
        #
        # Bounded to one attempt per head SHA. A cancel is often the
        # concurrency group doing its job, and a sweep that re-runs every time
        # it sees one can cancel-and-retry in a loop.
        cancelled_now = {
            w: g for w, g in (cancelled or {}).items()
            if g.get("run_id") and not rerun_count(s, w, sha)
        }
        if cancelled_now and not tally.get("pending") and not real:
            s["cancelled_groups"] = cancelled_now
            s["last_action"] = "re-run"
            s["last_verdict"] = (
                "; ".join(f"{w}: {len(g['jobs'])} job(s) cancelled mid-run"
                          for w, g in cancelled_now.items())
                + " — cancelled, not failed; the jobs never produced a result"
            )
            s["verdict_at"] = now()
            print("        -> re-run: "
                  + ", ".join(f"{w} had {len(g['jobs'])} job(s) cancelled"
                              for w, g in cancelled_now.items())
                  + "; nothing is running to replace them")
            row["outcome"] = "auto-re-run-cancelled"
            rerun.append(pr)
            report.append(row)
            continue
        s.pop("cancelled_groups", None)
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
        # Watcher-only failure, first time round: the watcher died on its own
        # (API 5xx / timeout) while the jobs it watched were still fine — if a
        # watched job had actually failed it would be sitting in `jobs` too.
        # That is unambiguous infra, so re-run it without spending a triage pass.
        # A *repeat* of the same failure is different: stop and evaluate main.
        if (all(g["watcher_only"] or g["gate_only"] for g in groups.values())
                and not any(g.get("repeat_after_rerun") for g in groups.values())):
            watchers = [w for w, g in groups.items() if g["watcher_only"]]
            s["last_action"] = "re-run"
            s["last_verdict"] = (
                f"watcher {', '.join(sorted({j for w in watchers for j in groups[w]['watcher_jobs']}))} "
                f"died on its own (API/timeout) with no failing test beside it"
            )
            s["verdict_at"] = now()
            if not auto_on(meta, "rerun"):
                # The verdict is still recorded — the switch gates the action,
                # not the judgement, so flipping it back on later re-runs with
                # a reason already attached instead of starting from nothing.
                print("        -> re-run WITHHELD: auto re-run is off for this PR")
                row["outcome"] = "auto-off:rerun"
                report.append(row)
                continue
            print("        -> re-run: watcher-only failure, no triage needed")
            row["outcome"] = "auto-re-run"
            rerun.append(pr)
            report.append(row)
            continue

        fp = failure_fingerprint(groups)
        judged_fp = s.get("verdict_fingerprint", "")
        judged_action = s.get("verdict_action") or s.get("last_action")
        if (s.get("verdict_sha") == sha and judged_action in ACTIONS
                and (judged_fp == fp or verdict_still_covers(judged_fp, fp))):
            s["last_action"] = judged_action  # restore if a sweep clobbered it
            # Already judged, same head SHA, and nothing red that the verdict
            # did not already cover. Re-triaging would throw away the verdict —
            # and for `code-fix` that means quietly re-queueing a PR we already
            # decided the author has to fix.
            gone = len(fingerprint_jobs(judged_fp) - fingerprint_jobs(fp))
            how = "same failures" if judged_fp == fp else (
                f"{gone} of the judged failures have since gone green; the rest "
                f"are unchanged")
            print(f"        -> keeping verdict `{judged_action}` "
                  f"({how}, already judged {tw(s.get('verdict_at'))})")
            row["outcome"] = f"verdict-held:{judged_action}"
            report.append(row)
            continue

        if not auto_on(meta, "triage"):
            # Deliberately not "awaiting-triage": nothing is waiting, because
            # nothing will pick it up. Saying otherwise puts the PR in the
            # triage panel forever and makes the queue look stuck.
            s["last_action"] = "triage-off"
            s["last_verdict"] = (
                "real in-scope NVIDIA failure, but auto triage is off for this "
                "PR — turn it on or hit Run now on the dashboard"
            )
            s["verdict_at"] = now()
            print("        -> triage SKIPPED: auto triage is off for this PR")
            row["outcome"] = "auto-off:triage"
            row["groups"] = groups
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

    if rerun:
        print("\n=== AUTO RE-RUN (no triage needed) ===")
        for pr in rerun:
            print(f"  python3 {Path(__file__).name} apply-verdict --pr {pr} "
                  f"--action re-run --summary \"{st[pr]['last_verdict']}\" --apply")

    if triage:
        print("\n=== TRIAGE REQUIRED ===")
        print("Run /ci-analysis on each PR below, reduce its Root Cause Failures table")
        print("to one action, then record it:\n")
        for pr in triage:
            reps = [w for w, g in (st[pr].get("failed_groups") or {}).items()
                    if g.get("repeat_after_rerun")]
            if reps:
                print(f"  !! {', '.join(reps)} failed the SAME way after a re-run.")
                print("     Do NOT record re-run again. Check whether main already")
                print("     has the fix (gh api compare/<head>...main) and prefer")
                print("     `merge-main`; use `code-fix` if it is the PR's own bug.")
            print(f"  /ci-analysis {st[pr].get('url') or pr}")
            print(f"  python3 {Path(__file__).name} apply-verdict --pr {pr} "
                  f"--action <re-run|code-fix|merge-main|wait-upstream|out-of-scope> "
                  f"--summary \"<one line>\" --apply\n")

    if not triage and not rerun:
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
    s["verdict_fingerprint"] = failure_fingerprint(s.get("failed_groups") or {})
    # Kept apart from `last_action` on purpose. `last_action` is scratch — any
    # sweep may overwrite it with `awaiting-triage` — so holding the verdict off
    # it loses the decision the moment one sweep re-opens triage, and nothing
    # can then restore it. This is the durable record of what was judged.
    s["verdict_action"] = a.action

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

    # Cancelled work is folded in here rather than kept as a separate pass: a
    # workflow can be in both (#39575's `PR Test Base` had 19 cancelled jobs
    # and one failed gate), and re-running it twice would have the second
    # attempt cancel the first through the very concurrency group that caused
    # this.
    groups = {w: dict(g) for w, g in (s.get("failed_groups") or {}).items()}
    for w, cg in (s.get("cancelled_groups") or {}).items():
        g = groups.setdefault(
            w, {"run_id": cg.get("run_id"), "jobs": [], "gate_jobs": [],
                "watcher_jobs": []})
        g["cancelled_jobs"] = cg.get("jobs") or []
        if not g.get("run_id"):
            g["run_id"] = cg.get("run_id")
    if not groups:
        die(f"#{pr} has no recorded failed in-scope workflows — re-run `sweep --pr {pr}`")

    # Checked here rather than only in the sweep: this is the single place a
    # re-run is actually POSTed, so every caller — sweep, dashboard, a hand-run
    # command — is gated by the same switch.
    if not auto_on(load(WATCHLIST, {}).get(pr, {}), "rerun") and not a.force_auto:
        save(STATE, st)  # keep the verdict; only the action is withheld
        print(f"#{pr}  verdict `re-run` recorded, but auto re-run is OFF for "
              f"this PR — nothing re-run. Flip it on in the dashboard's Auto "
              f"column, or pass --force-auto.")
        log_line(f"#{pr} rerun withheld (auto off) :: {a.summary}")
        return

    # Never re-queue a run that is not on the PR's current head. `pr-test.yml`
    # uses `concurrency: pr-test-pull_request-<n>-all` with
    # `cancel-in-progress: true`, so POSTing rerun-failed-jobs at an older run
    # does not merely waste CI: the re-queued run outranks the one already
    # going and GitHub kills the real CI with "Canceling since a higher
    # priority waiting request for pr-test-pull_request-<n>-all exists".
    #
    # #39575 on 2026-10-02 is the worked example. The author pushed at 14:38
    # and run 37021312519 started on 41b89e5e; the button had been drawn from a
    # pre-push sweep, so the 14:43 click re-ran 36947209065 (321d4ed0) and
    # cancelled 37021312519 five minutes in. The jobs read "Cancelled after
    # 5m", which looks exactly like a timeout and is not one.
    #
    # `Update branch` has refused this class of mistake all along via
    # `expected_head_sha`; this is the same guard for the other button.
    live_head = ""
    rc, out, _ = gh_try(["api", f"repos/{a.repo}/pulls/{pr}", "-q", ".head.sha"])
    if rc == 0:
        live_head = out.strip()
    if live_head and sha and live_head != sha:
        die(f"#{pr} state is stale: swept at {sha[:8]}, head is now "
            f"{live_head[:8]}. Re-sweep first — re-running a run from the old "
            f"head would cancel the CI now going on {live_head[:8]}.")
    current = live_head or sha

    reruns = s.setdefault("reruns", {})
    did, skipped = [], []
    for wf, g in groups.items():
        used = rerun_count(s, wf, sha)
        # `gate_rerunnable` is the one gate-only case worth re-running: the gate
        # rejected the *author* (rate limit), not the code, and the check is
        # evaluated against whoever triggers the run — so a re-run from this
        # account, which has write access, turns it green. Skipping it here is
        # what left #34502's NVIDIA CI permanently unstarted.
        if (g.get("gate_only") and not g.get("gate_rerunnable")
                and not g.get("cancelled_jobs") and not a.force_gates):
            why = g.get("gate_hint") or (
                f"only {', '.join(g.get('gate_jobs', [])[:3])} failed; re-running an "
                f"aggregation gate cannot turn it green"
            )
            skipped.append(f"{wf} (gate-only: {why} — override with --force-gates)")
            continue
        run_id = g.get("run_id")
        if not run_id:
            skipped.append(f"{wf} (no run id parsed from check link)")
            continue
        # Second half of the staleness guard. The PR-level check above catches
        # "the author pushed since the sweep"; this catches a `failed_groups`
        # entry that points at an older run even though the snapshot itself is
        # current — a re-run recorded against a superseded workflow run.
        rc, out, _ = gh_try(["api", f"repos/{a.repo}/actions/runs/{run_id}",
                             "-q", ".head_sha + \" \" + (.conclusion // \"\")"])
        run_sha, _, run_conclusion = (out.strip() if rc == 0 else "").partition(" ")
        if run_sha and current and run_sha != current:
            skipped.append(
                f"{wf} (run {run_id} is on {run_sha[:8]}, not the current head "
                f"{current[:8]} — re-queuing it would cancel the live run)")
            continue

        # `rerun-failed-jobs` re-runs failed jobs; a cancelled job is not one,
        # so on a cancelled run it restarts the aggregation gate and leaves
        # every killed test exactly where it was. #39575's run 37021312519 had
        # 19 cancelled jobs against a single failure — the narrow endpoint
        # would have re-run that one gate and reported success.
        full = bool(g.get("cancelled_jobs")) or run_conclusion == "cancelled"
        how = ", all jobs — the run was cancelled" if full else ""
        endpoint = (f"repos/{a.repo}/actions/runs/{run_id}/"
                    + ("rerun" if full else "rerun-failed-jobs"))
        if a.apply:
            rc, _, err = gh_try(["api", "-X", "POST", endpoint])
            if rc != 0:
                # "already running" is the common one: some shards are still in
                # progress, so GitHub refuses. Not an error worth losing the
                # verdict over — the next sweep retries, and the attempt is not
                # counted because nothing was actually re-run.
                if "already running" in err.lower():
                    why = "deferred — workflow still running, next sweep retries"
                else:
                    why = err.splitlines()[-1] if err else "rerun failed"
                skipped.append(f"{wf} ({why})")
                # Proof that the POST was actually made and refused. Without it
                # the dashboard cannot tell a refused attempt from a decision
                # that was never applied, and it used to assume the former —
                # printing "GitHub refused" for a re-run nobody had tried yet.
                s.setdefault("rerun_deferred", {})[wf] = {
                    "sha": sha, "at": now(), "why": why}
                continue
            # Store what failed, so the next sweep can tell "same failure again"
            # from "a different failure this time".
            reruns[wf] = {"sha": sha, "count": used + 1, "at": now(),
                          "sig": g.get("sig", "")}
            did.append(f"{wf} (run {run_id}, attempt {used + 1}{how})")
        else:
            did.append(f"WOULD rerun {wf} (run {run_id}, attempt {used + 1}{how})")

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


def cmd_notify(a) -> None:
    """Post the conflict notice for one PR, on demand.

    Same idempotency as the sweep: one comment per head SHA, never a repeat.
    Exists so the dashboard can offer an explicit "notify now" button instead of
    making you wait for the next scheduled sweep.
    """
    pr = parse_pr(a.pr)
    st = load(STATE, {})
    wl = load(WATCHLIST, {})
    repo = (wl.get(pr) or {}).get("repo") or a.repo
    snap = pr_snapshot(pr, repo)
    s = st.setdefault(pr, {})
    sha = snap.get("headRefOid", "")
    author = (snap.get("author") or {}).get("login", "")

    if snap.get("state") != "OPEN":
        die(f"#{pr} is {snap.get('state')}, not OPEN")
    if snap.get("mergeable") != "CONFLICTING":
        die(f"#{pr} is {snap.get('mergeable')} — nothing to notify about")
    if not author:
        die(f"#{pr} has no author login to notify")
    if s.get("conflict_comment_sha") == sha:
        print(f"#{pr}  already notified @{author} for {sha[:8]} — not repeating")
        print(f"       {s.get('conflict_comment_url', '')}")
        return

    body = conflict_comment(author, sha, snap.get("headRefName", "?"))
    if not a.apply:
        print(f"#{pr}  would comment to @{author}:\n{body}")
        return
    out = gh(["pr", "comment", pr, "--repo", repo, "--body", body])
    url = next((ln.strip() for ln in out.splitlines()
                if ln.strip().startswith("http")), "")
    s.update(conflict_comment_sha=sha, conflict_comment_url=url,
             conflict_comment_at=now(), mergeable="CONFLICTING",
             head_sha=sha, author=author, title=snap.get("title", ""),
             url=snap.get("url", ""), last_action="conflict")
    save(STATE, st)
    log_line(f"#{pr} conflict comment posted on demand for {sha[:8]} -> @{author} {url}")
    print(f"#{pr}  commented to @{author}  {url}")


def cmd_toggle(a) -> None:
    st = load(STATE, {})
    on = a.cmd == "resume"
    set_monitoring(st, on)
    save(STATE, st)
    print(f"monitoring {'RESUMED' if on else 'PAUSED'} at {now()}")
    log_line(f"monitoring {'resumed' if on else 'paused'}")


def cmd_update_branch(a) -> None:
    """GitHub's own `Update branch` button, nothing more.

    `PUT /pulls/{n}/update-branch` asks GitHub to make the merge commit
    server-side. It does not touch a working tree, cannot resolve a conflict,
    and never rewrites the author's commits — so it is safe to offer on
    someone else's PR in a way that cloning and pushing would not be. When the
    branch genuinely conflicts GitHub refuses, and that is `/pr-conflict-fix`'s
    job, not this one.
    """
    pr = parse_pr(a.pr)
    st = load(STATE, {})
    s = st.get(pr)
    if not s:
        die(f"#{pr} has no sweep state — run `sweep --pr {pr}` first")

    sha = s.get("head_sha", "")
    if s.get("mergeable") == "CONFLICTING":
        die(f"#{pr} conflicts with main — GitHub's Update branch cannot merge it. "
            f"That needs a working tree: `/pr-conflict-fix {pr}`.")
    behind = s.get("behind_by")
    if behind == 0 and not a.force:
        die(f"#{pr} is already up to date with main — nothing to merge.")

    prev = (s.get("update_branch") or {})
    if prev.get("sha") == sha and not a.force:
        die(f"#{pr} was already updated at {prev.get('at')} and has not moved "
            f"since ({sha[:8]}). Re-sweep first, or pass --force.")

    if not a.apply:
        print(f"#{pr}  WOULD update branch ({behind} commit(s) behind main)")
        return

    # expected_head_sha makes this a no-op if the author pushed between the
    # sweep that drew the button and the click: better a 422 than merging into
    # a head we never looked at.
    rc, out, err = gh_try(["api", "-X", "PUT",
                           f"repos/{a.repo}/pulls/{pr}/update-branch",
                           "-f", f"expected_head_sha={sha}"])
    if rc != 0:
        detail = (err or out or "").strip().splitlines()
        why = detail[-1] if detail else "update-branch failed"
        if "expected_head_sha" in (err or "") or "422" in (err or ""):
            why += (" — the head moved since the last sweep; refresh and try again")
        print(f"#{pr}  NOT updated: {why}")
        log_line(f"#{pr} update-branch failed :: {why}")
        sys.exit(1)

    s["update_branch"] = {"sha": sha, "at": now(), "behind_was": behind}
    save(STATE, st)
    print(f"#{pr}  update branch requested — GitHub is merging main "
          f"({behind} commit(s)) into the branch; CI restarts on the new head.")
    log_line(f"#{pr} update-branch applied ({behind} behind) at {sha[:8]}")


def cmd_auto(a) -> None:
    pr = parse_pr(a.pr)
    wl = load(WATCHLIST, {})
    if pr not in wl:
        die(f"#{pr} is not on the watchlist")
    changed = []
    for f in AUTO_FIELDS:
        want = getattr(a, f.replace("-", "_"), None)
        if want is None:
            continue
        set_auto(wl[pr], f, want == "on")
        changed.append(f"{f}={want}")
    if changed:
        save(WATCHLIST, wl)
        log_line(f"#{pr} auto {' '.join(changed)}")
    print(f"#{pr} auto: "
          + "  ".join(f"{f}={'on' if auto_on(wl[pr], f) else 'OFF'}"
                      for f in AUTO_FIELDS))


def cmd_report(a) -> None:
    wl, st = load(WATCHLIST, {}), load(STATE, {})
    if not wl:
        print("(watchlist is empty)")
        return
    print(report_text(wl, st, getattr(a, "group", None)))


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
    if a.group is not None:
        wl[pr]["group"] = a.group.strip()
    save(WATCHLIST, wl)
    print(f"#{pr}: priority={priority_of(wl[pr])} group={group_of(wl[pr])!r} "
          f"note={wl[pr].get('note', '')!r}")


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
    s.add_argument("--group", default="", help="tab to file it under, e.g. a model name")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("remove", help="stop watching PR(s)")
    s.add_argument("refs", nargs="+")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("list", help="show the watchlist")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("sweep", help="phase A: gather state, emit triage requests")
    s.add_argument("--track", default="all", choices=[*TRACKS, "all"])
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
    s.add_argument("--force-auto", action="store_true",
                   help="re-run even if this PR's auto re-run switch is off")
    s.set_defaults(func=cmd_apply_verdict)

    s = sub.add_parser("update-branch",
                       help="GitHub's Update branch: merge main into the PR branch")
    s.add_argument("--pr", required=True)
    s.add_argument("--apply", action="store_true")
    s.add_argument("--force", action="store_true",
                   help="update even if already up to date / already updated on this head")
    s.set_defaults(func=cmd_update_branch)

    s = sub.add_parser("auto", help="per-PR autopilot switches (both default on)")
    s.add_argument("pr")
    for f in AUTO_FIELDS:
        s.add_argument(f"--{f}", choices=["on", "off"])
    s.set_defaults(func=cmd_auto)

    s = sub.add_parser("notify", help="post the conflict notice for one PR now")
    s.add_argument("--pr", required=True)
    s.add_argument("--apply", action="store_true")
    s.set_defaults(func=cmd_notify)

    s = sub.add_parser("report", help="print the <P0><CI clear><PR…> status block")
    s.add_argument("--group", help="only this tab")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("set", help="set a PR's report priority / note")
    s.add_argument("pr")
    s.add_argument("--priority", choices=list(PRIORITIES))
    s.add_argument("--note")
    s.add_argument("--group", help='tab to file it under ("" clears it)')
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
