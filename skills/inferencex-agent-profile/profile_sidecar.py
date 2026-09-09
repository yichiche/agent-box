#!/usr/bin/env python3
"""Drive SGLang's profiler through the HTTP API during an AgentX trace replay.

The agentic path in InferenceX has no client-side `--profile` hook: `PROFILE=1`
in benchmarks/benchmark_lib.sh only reaches the fixed-seq client
(bench_serving `--profile`, line ~682). aiperf has no equivalent flag, and the
recipe owns the whole server + client command line.

So we profile from the side: watch the engine's own Prometheus counters until
the replay is demonstrably in its measurement window, then POST /start_profile
with `num_steps` (the profiler auto-stops after that many forward steps -- no
race with a /stop_profile call that might land after the replay ended).

Nothing here launches or kills anything. If profiling fails, the replay is
untouched and still produces its normal aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RUNNING = "sglang:num_running_reqs"
QUEUE = "sglang:num_queue_reqs"


def log(msg: str) -> None:
    print(f"[agent-profile {time.strftime('%T')}] {msg}", flush=True)


def http_get(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def http_post(url: str, payload: dict, timeout: float = 120.0) -> tuple[int, str]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, repr(e)


def scrape(port: int) -> dict[str, float]:
    """Return {metric_name: value} for the gauges we care about."""
    text = http_get(f"http://127.0.0.1:{port}/metrics", timeout=5)
    if text is None:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in (RUNNING, QUEUE):
            if line.startswith(name):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    return out


def wait_healthy(port: int, timeout: int) -> bool:
    """A 397B load behind a cold page cache can take well over 20 minutes."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if http_get(f"http://127.0.0.1:{port}/health", timeout=3) is not None:
            log(f"server healthy on :{port} after {int(time.time() - t0)}s")
            return True
        time.sleep(10)
    log(f"ERROR: server never became healthy on :{port} within {timeout}s")
    return False


def log_says_profiling(log_path: Path | None, pattern: re.Pattern | None) -> bool:
    if log_path is None or pattern is None or not log_path.exists():
        return False
    try:
        tail = log_path.read_bytes()[-200_000:].decode("utf-8", "replace")
    except OSError:
        return False
    return bool(pattern.search(tail))


def wait_for_measurement_window(
    port: int,
    conc: int,
    timeout: int,
    poll: int,
    settle_polls: int,
    recipe_log: Path | None,
    trigger_re: re.Pattern | None,
    require_drain: bool,
) -> bool:
    """Block until the replay is past warmup and into steady-state profiling.

    aiperf's shape is: warmup (`--warmup-requests-per-lane` one-token requests
    per lane) -> drain -> measurement. Both phases keep the engine busy, so
    "requests are running" alone would fire during warmup and profile a batch
    of 1-token requests that looks nothing like the workload.

    Primary signal is the drain: busy -> idle -> busy. Secondary (and cheaper
    when it works) is the aiperf phase banner in the recipe log, which the
    caller can pin with --trigger-regex. Either one wins.
    """
    want = max(1, int(round(0.5 * conc)))
    t0 = time.time()
    seen_busy = False
    seen_drain = not require_drain
    idle_run = 0
    busy_run = 0

    while time.time() - t0 < timeout:
        m = scrape(port)
        running = m.get(RUNNING)

        if log_says_profiling(recipe_log, trigger_re):
            log("measurement window: matched trigger regex in recipe log")
            return True

        if running is None:
            time.sleep(poll)
            continue

        if running >= want:
            busy_run += 1
            idle_run = 0
            if not seen_busy:
                seen_busy = True
                log(f"warmup traffic seen (running={running:.0f} >= {want})")
            if seen_drain and busy_run >= settle_polls:
                log(
                    f"measurement window: running={running:.0f} for "
                    f"{busy_run * poll}s after drain"
                )
                return True
        else:
            busy_run = 0
            if seen_busy and running == 0:
                idle_run += 1
                if not seen_drain and idle_run >= 2:
                    seen_drain = True
                    log("warmup drain observed (running=0); waiting for steady state")
        time.sleep(poll)

    log(f"ERROR: no measurement window within {timeout}s (last running={running})")
    return False


def collect(out_dir: Path, profile_id: str, quiet_for: int, timeout: int) -> list[Path]:
    """Wait for the per-rank traces to finish being written.

    Each TP rank gzips its own `<profile_id>-TP-<n>[-EP-<n>].trace.json.gz`.
    They land at different times and a partially written file will crash any
    downstream parser, so we require the file set *and* its total size to stop
    changing for `quiet_for` seconds.
    """
    t0 = time.time()
    last_sig: tuple | None = None
    quiet_since = None
    while time.time() - t0 < timeout:
        files = sorted(out_dir.glob(f"*{profile_id}*.trace.json.gz"))
        sig = tuple((f.name, f.stat().st_size) for f in files)
        if files and sig == last_sig:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= quiet_for:
                return files
        else:
            quiet_since = None
        last_sig = sig
        time.sleep(5)
    log(f"WARNING: traces still settling after {timeout}s; returning what exists")
    return sorted(out_dir.glob(f"*{profile_id}*.trace.json.gz"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8888)))
    p.add_argument("--conc", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--profile-id", default=None)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--activities", default="CPU,GPU")
    p.add_argument("--profile-by-stage", action="store_true")
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--merge-profiles", action="store_true")
    p.add_argument("--server-wait", type=int, default=3600)
    p.add_argument("--window-wait", type=int, default=3600)
    p.add_argument("--poll", type=int, default=5)
    p.add_argument("--settle-polls", type=int, default=6)
    p.add_argument("--no-require-drain", action="store_true")
    p.add_argument("--collect-timeout", type=int, default=1800)
    p.add_argument("--recipe-log", default=None)
    p.add_argument("--trigger-regex", default=None)
    a = p.parse_args()

    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_id = a.profile_id or f"agentx-conc{a.conc}-{time.strftime('%Y%m%d_%H%M%S')}"
    recipe_log = Path(a.recipe_log) if a.recipe_log else None
    trigger_re = re.compile(a.trigger_regex) if a.trigger_regex else None

    if not wait_healthy(a.port, a.server_wait):
        return 2
    if not wait_for_measurement_window(
        a.port,
        a.conc,
        a.window_wait,
        a.poll,
        a.settle_polls,
        recipe_log,
        trigger_re,
        require_drain=not a.no_require_drain,
    ):
        return 3

    payload = {
        "output_dir": str(out_dir),
        "num_steps": a.num_steps,
        "activities": [x for x in a.activities.split(",") if x],
        "profile_by_stage": a.profile_by_stage,
        "record_shapes": a.record_shapes,
        "with_stack": a.with_stack,
        "merge_profiles": a.merge_profiles,
        "profile_id": profile_id,
    }
    log(f"POST /start_profile {json.dumps(payload)}")
    # num_steps makes the profiler self-terminate, so this call is the only one
    # we make. A /stop_profile from here could otherwise arrive after the replay
    # has ended and the scheduler is already tearing down.
    code, body = http_post(f"http://127.0.0.1:{a.port}/start_profile", payload)
    if code != 200:
        log(f"ERROR: start_profile returned {code}: {body[:500]}")
        return 4

    files = collect(out_dir, profile_id, quiet_for=30, timeout=a.collect_timeout)
    if not files:
        log("ERROR: no trace files were produced")
        return 5

    manifest = {
        "profile_id": profile_id,
        "conc": a.conc,
        "port": a.port,
        "request": payload,
        "traces": [str(f) for f in files],
    }
    (out_dir / f"profile_manifest_conc{a.conc}.json").write_text(
        json.dumps(manifest, indent=2)
    )
    for f in files:
        log(f"trace {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
