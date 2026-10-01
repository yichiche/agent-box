"""Per-PR timeline, reconstructed from what the sweep already wrote down.

Nothing new is recorded for this. Two files already hold the history between
them, and neither is readable as a story on its own:

  * `sweep.log`  — every mutation, but interleaved across all PRs.
  * `sweeps/*.json` — every sweep's view of every PR, including the long runs
    where nothing changed at all.

So the log is filtered to one PR, the sweep records are collapsed to their
*transitions* (a new head SHA, a changed outcome), and the two are merged into
one list. Collapsing is the load-bearing part: 36 sweeps of an idle PR is 36
identical rows, which is not a history, it is noise.
"""

from __future__ import annotations

import html
import json
import re

from watch import LOG, SWEEPS, parse_ts, tally_bits, tw


def esc(x) -> str:
    return html.escape(str(x or ""))

# How far back to read. The log is small and append-only, but the sweep
# directory grows one file per sweep forever; the recent past is what anyone
# opening a row is actually asking about.
MAX_SWEEPS = 400

# A sweep outcome, read as a CI state: (css class, wording). The point of the
# panel is "from when was this green, and when was it red", so the state line
# is colour-coded and everything else stays quiet around it.
#
# `out-of-scope` IS the green state, not a third thing: it means every red
# check is an aggregation gate mirroring a vendor workflow, so no NVIDIA job is
# failing. Calling it anything but green would hide the moment a PR came good.
OUTCOME_STATE = {
    "needs-triage": ("bad", "CI red — a real NVIDIA job failed"),
    "verdict-held:code-fix": ("bad", "CI red — real bug in this PR"),
    "conflict-would-notify": ("bad", "conflicts with main — CI cannot complete"),
    "conflict-notified": ("bad", "conflicts with main — CI cannot complete"),
    "conflict-already-notified": ("bad", "conflicts with main — CI cannot complete"),
    "out-of-scope": ("ok", "CI green — no NVIDIA job failing"),
    "auto-re-run": ("warn", "re-run in flight — watcher died on its own"),
    "verdict-held:re-run": ("warn", "re-run in flight"),
    "verdict-held:merge-main": ("warn", "stuck — main has the fix, PR is behind"),
    "verdict-held:wait-upstream": ("warn", "stuck — blocked on an upstream fix"),
}


def _state(outcome: str, tally: dict | None) -> tuple[str, str] | None:
    """(css class, wording) for a sweep outcome, or None if it says nothing."""
    if outcome.startswith("closed:"):
        return "dim", f"no longer open ({outcome.split(':', 1)[1].lower()})"
    hit = OUTCOME_STATE.get(outcome)
    if not hit:
        return None
    # `out-of-scope` is recorded as soon as nothing *in scope* is failing, which
    # happens while most of the run is still queued. Calling that green would
    # put the "went green" timestamp an hour before CI actually finished, and
    # that timestamp is the whole question this panel answers.
    if hit[0] == "ok" and (tally or {}).get("pending"):
        return "warn", "CI running — nothing failing so far"
    return hit


def _counts(tally: dict | None) -> str:
    """The job counts, as evidence beside the state.

    "CI green" next to `59 pass, 2 fail` is honest about the two aggregation
    gates still red; the bare word would read like a contradiction.
    """
    bits = tally_bits(tally)
    return f' <span class="dim">({esc(bits)})</span>' if bits else ""


# A verdict is an instruction, and its colour should match how bad the news is.
VERDICT_CLASS = {
    "code-fix": "bad",
    "merge-main": "warn",
    "wait-upstream": "warn",
    "conflict": "bad",
    "re-run": "warn",
    "out-of-scope": "dim",
    "green": "ok",
}


def _kind(msg: str) -> tuple[str, str]:
    """Classify an audit line into (css class, human HTML).

    Every value lifted out of the log is escaped here; only the markup this
    function adds itself is literal. Verdict summaries are free text written by
    whoever ran `apply-verdict`, so they cannot be trusted into the page raw.
    """
    m = re.search(r"verdict=(\S+) :: (.*)", msg)
    if m:
        return VERDICT_CLASS.get(m.group(1), "warn"), (
            f"verdict <b>{esc(m.group(1))}</b> &mdash; {esc(m.group(2))}")
    m = re.search(r"rerun (\d+) group\(s\) :: (.*)", msg)
    if m:
        return "warn", (f"re-ran {esc(m.group(1))} workflow(s) "
                        f"&mdash; {esc(m.group(2))}")
    if "conflict comment posted" in msg:
        m = re.search(r"-> (@\S+) (\S+)", msg)
        who = f" to {esc(m.group(1))}" if m else ""
        link = (f' <a href="{esc(m.group(2))}" target="_blank">see comment</a>'
                if m else "")
        return "bad", f"conflict notice posted{who}{link}"
    if "added" in msg or msg.startswith("add "):
        return "dim", "added to the watchlist"
    if "removed" in msg or msg.startswith("remove "):
        return "dim", "removed from the watchlist"
    return "dim", esc(msg)


def _log_events(pr: str) -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    needle = f"#{pr}"
    for line in LOG.read_text(errors="replace").splitlines():
        at, _, msg = line.partition(" ")
        if needle not in msg:
            continue
        cls, text = _kind(msg)
        out.append({"at": at, "cls": cls, "text": text})
    return out


def _sweep_events(pr: str) -> list[dict]:
    """Transitions only — a sweep that saw no change contributes nothing."""
    if not SWEEPS.exists():
        return []
    files = sorted(SWEEPS.glob("*.json"))[-MAX_SWEEPS:]
    out: list[dict] = []
    last_sha = last_state = None
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        row = next((r for r in rec.get("rows", []) if str(r.get("pr")) == pr), None)
        if not row:
            continue
        at = rec.get("at", "")
        sha = row.get("sha")
        if sha and sha != last_sha:
            # Not on the first sighting: that is "we started watching", which
            # the log already says, not "the author pushed".
            if last_sha is not None:
                out.append({"at": at, "cls": "dim",
                            "text": f"author pushed &mdash; new head "
                                    f'<span class="mono">{esc(sha)}</span>'})
            last_sha = sha
        st = _state(row.get("outcome") or "", row.get("tally"))
        if not st:
            continue
        # Collapse on the *state*, not the raw outcome: `needs-triage` followed
        # by `verdict-held:code-fix` is one unbroken stretch of red, and
        # printing it twice would read as two separate failures.
        if st == last_state:
            continue
        cls, text = st
        out.append({"at": at, "cls": cls,
                    "text": esc(text) + _counts(row.get("tally"))})
        last_state = st
    return out


def history_html(pr: str) -> str:
    """Newest first — the question behind opening a row is "what happened
    lately", not "how did this PR begin".

    Because only transitions are emitted, a state line's timestamp *is* the
    moment the PR entered that state: the topmost green line answers "since
    when has this been clean" without any arithmetic.
    """
    events = _log_events(pr) + _sweep_events(pr)
    if not events:
        return '<div class="dim">no history recorded yet</div>'
    events.sort(key=lambda e: e["at"], reverse=True)
    rows = []
    for e in events:
        rows.append(
            f'<div class="histline"><span class="mono dim histat">'
            f'{tw(e["at"]) if parse_ts(e["at"]) else e["at"]}</span>'
            f'<span class="{e["cls"]}">{e["text"]}</span></div>'
        )
    return "".join(rows)
