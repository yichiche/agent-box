"""Compare library versions between two benchmark runs.

Queries the version_snapshots table to identify what changed between a
baseline run and a regression run, suggests root causes, and generates
git bisect plans for git-tracked repositories.

Works both as an importable module (used by dashboard.py) and as a
standalone CLI tool.
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as _cfg
from config import DB_PATH

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────


def _versions_differ(a: sqlite3.Row, b: sqlite3.Row) -> bool:
    """Return True if two version snapshot rows represent different versions.

    Compares by git_commit first when both rows carry one; falls back to
    comparing the version string.
    """
    commit_a = a['git_commit'] or ''
    commit_b = b['git_commit'] or ''
    if commit_a and commit_b:
        return commit_a != commit_b
    return (a['version'] or '') != (b['version'] or '')


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return {k: row[k] for k in row.keys()}


# ── Core API ──────────────────────────────────────────────────────────────


def compute_version_diff(conn, run_id_a, run_id_b) -> dict:
    """Compare version snapshots between two benchmark runs.

    Returns a dict with keys: changed, unchanged, added, removed.

    * changed  – libraries present in both runs whose version differs
    * unchanged – libraries present in both runs with the same version
    * added    – libraries only in run B
    * removed  – libraries only in run A
    """
    query = """
        SELECT library_name, version, source_type, git_commit
        FROM version_snapshots
        WHERE run_id = ?
        ORDER BY library_name
    """
    rows_a = conn.execute(query, (run_id_a,)).fetchall()
    rows_b = conn.execute(query, (run_id_b,)).fetchall()

    dict_a = {r['library_name']: r for r in rows_a}
    dict_b = {r['library_name']: r for r in rows_b}

    names_a = set(dict_a)
    names_b = set(dict_b)

    changed = []
    unchanged = []

    for name in sorted(names_a & names_b):
        ra, rb = dict_a[name], dict_b[name]
        if _versions_differ(ra, rb):
            changed.append({
                'name': name,
                'old_version': ra['version'],
                'new_version': rb['version'],
                'old_commit': ra['git_commit'],
                'new_commit': rb['git_commit'],
                'source_type': rb['source_type'],
            })
        else:
            unchanged.append({
                'name': name,
                'version': ra['version'],
                'source_type': ra['source_type'],
            })

    added = [
        {
            'name': name,
            'version': dict_b[name]['version'],
            'source_type': dict_b[name]['source_type'],
            'git_commit': dict_b[name]['git_commit'],
        }
        for name in sorted(names_b - names_a)
    ]

    removed = [
        {
            'name': name,
            'version': dict_a[name]['version'],
            'source_type': dict_a[name]['source_type'],
            'git_commit': dict_a[name]['git_commit'],
        }
        for name in sorted(names_a - names_b)
    ]

    return {
        'changed': changed,
        'unchanged': unchanged,
        'added': added,
        'removed': removed,
    }


def suggest_root_cause(diff: dict) -> list[dict]:
    """Rank changed libraries by likelihood of causing a regression.

    Uses HIGH_PRIORITY_LIBRARIES from config to assign priority
    (lower index = more likely).  Returns a sorted list of suggestion dicts.
    """
    high_priority = getattr(_cfg, 'HIGH_PRIORITY_LIBRARIES', [])

    changed = diff.get('changed', [])
    if not changed:
        return []

    sole_change = len(changed) == 1

    suggestions = []
    for entry in changed:
        name = entry['name']
        try:
            priority = high_priority.index(name)
        except ValueError:
            priority = 999

        if sole_change:
            reason = 'sole change -- definitive'
        elif priority < 999:
            reason = f'high-priority library (rank {priority + 1})'
        else:
            reason = 'changed between runs'

        suggestions.append({
            'name': name,
            'priority': priority,
            'reason': reason,
        })

    suggestions.sort(key=lambda s: s['priority'])
    return suggestions


def generate_bisect_plan(diff, conn, run_id_a, run_id_b) -> list[dict]:
    """Generate git-bisect info for git-tracked repos that changed.

    Returns a list of dicts with fields: name, good_commit, bad_commit,
    repo_path, commit_range_cmd.
    """
    git_repos_list = getattr(_cfg, 'GIT_TRACKED_REPOS', [])
    git_repos = {r['name']: r['path'] for r in git_repos_list}

    plans = []
    for entry in diff.get('changed', []):
        if entry.get('source_type') != 'git':
            continue

        old_commit = entry.get('old_commit')
        new_commit = entry.get('new_commit')
        if not old_commit or not new_commit:
            continue

        name = entry['name']
        repo_path = git_repos.get(name, '')

        plan = {
            'name': name,
            'good_commit': old_commit,
            'bad_commit': new_commit,
            'repo_path': repo_path,
        }

        if repo_path:
            plan['commit_range_cmd'] = (
                f'git -C {repo_path} log --oneline {old_commit}..{new_commit}'
            )
        else:
            plan['commit_range_cmd'] = (
                f'git log --oneline {old_commit}..{new_commit}'
            )

        plans.append(plan)

    return plans


# ── Docker / DB helpers (CLI) ─────────────────────────────────────────────


def _docker_cmd() -> list[str]:
    """Return the docker command prefix (with or without sudo).

    Always passes --config pointing to the user's docker config so that
    sudo docker uses the user's credentials rather than root's.
    """
    docker_config = f'{_cfg.HOST_HOME_DIR}/.docker'
    if _cfg.USE_SUDO_DOCKER:
        return ['sudo', 'docker', '--config', docker_config]
    return ['docker', '--config', docker_config]


def _get_baseline_metric(conn: sqlite3.Connection, run_id: int,
                         metric_name: str, concurrency: int):
    """Query benchmark_metrics for a run's metric value at a given concurrency.

    The benchmark_metrics table stores each metric as its own column
    (e.g. output_throughput, median_e2e_latency_ms).
    """
    cursor = conn.execute('PRAGMA table_info(benchmark_metrics)')
    valid_columns = {row[1] for row in cursor}
    if metric_name not in valid_columns:
        logger.error('Unknown metric column: %s', metric_name)
        return None

    row = conn.execute(
        f'SELECT {metric_name} FROM benchmark_metrics '
        f'WHERE run_id = ? AND concurrency = ? LIMIT 1',
        (run_id, concurrency)
    ).fetchone()

    if row and row[metric_name] is not None:
        return float(row[metric_name])
    return None


def _get_image_for_run(conn: sqlite3.Connection, run_id: int):
    """Return the full docker image string for a benchmark run."""
    row = conn.execute(
        'SELECT image_tag FROM benchmark_runs WHERE id = ?',
        (run_id,)
    ).fetchone()
    if row is None:
        return None
    return f'{_cfg.DOCKER_HUB_REPO}:{row["image_tag"]}'


def _get_run_status(conn: sqlite3.Connection, run_id: int):
    """Return the status of a benchmark run."""
    row = conn.execute(
        'SELECT status FROM benchmark_runs WHERE id = ?',
        (run_id,)
    ).fetchone()
    if row:
        return row['status']
    return None


def _get_sglang_commit_for_run(conn: sqlite3.Connection, run_id: int):
    """Return the sglang git commit recorded for a benchmark run."""
    row = conn.execute(
        "SELECT git_commit FROM version_snapshots\n"
        "       WHERE run_id = ? AND library_name = 'sglang' LIMIT 1",
        (run_id,)
    ).fetchone()
    if row and row['git_commit']:
        return row['git_commit']
    return None


def _list_commits(image, repo_path, good_commit, bad_commit):
    """List commits between good and bad (exclusive good, inclusive bad).

    Starts a temporary container from the given image, runs git rev-list,
    then cleans up the container.
    """
    container_name = 'bisect-revlist-tmp'
    docker = _docker_cmd()

    # Clean up any existing container
    subprocess.run(docker + ['rm', '-f', container_name],
                   capture_output=True)

    try:
        # Start temporary container
        subprocess.run(
            docker + ['run', '-d', '--name', container_name,
                      image, 'bash', '-c', 'sleep infinity'],
            check=True, capture_output=True
        )

        # Run git rev-list inside container
        result = subprocess.run(
            docker + ['exec', container_name,
                      'git', '-C', repo_path,
                      'rev-list', '--reverse',
                      f'{good_commit}..{bad_commit}'],
            check=True, capture_output=True, text=True
        )

        commits = [c.strip() for c in result.stdout.strip().splitlines()
                   if c.strip()]
        return commits
    finally:
        subprocess.run(docker + ['rm', '-f', container_name],
                       capture_output=True)


def run_bisect(plan, image, model_path, *, mode='benchmark',
               baseline_metric=None, metric_name='output_throughput',
               concurrency=1, launch_script=None, port=30000) -> dict:
    """Binary-search through commits to find the first bad one.

    Args:
        plan: bisect plan dict with good_commit, bad_commit, repo_path, name
        image: docker image to use for containers
        model_path: model path for the benchmark
        mode: 'benchmark' (compare metrics) or 'launch' (just check startup)
        baseline_metric: metric value of the good run (for benchmark mode)
        metric_name: which metric column to compare
        concurrency: concurrency level for the benchmark
        launch_script: optional custom launch script
        port: port for the server
    """
    good_commit = plan['good_commit']
    bad_commit = plan['bad_commit']
    repo_path = plan.get('repo_path', '/sgl-workspace/sglang')

    logger.info('Listing commits between %s..%s in %s',
                good_commit[:7], bad_commit[:7], plan['name'])

    commits = _list_commits(image, repo_path, good_commit, bad_commit)
    if not commits:
        logger.warning('No commits found between %s and %s',
                       good_commit[:7], bad_commit[:7])
        return {
            'first_bad_commit': bad_commit,
            'first_bad_subject': '(no intermediate commits)',
            'total_steps': 0,
            'log': [],
        }

    logger.info('Found %d commits to bisect', len(commits))

    bisect_script = str(Path(__file__).resolve().parent / 'bisect_runner.sh')
    bisect_log = []
    lo, hi = 0, len(commits) - 1
    step = 0
    total_steps = math.ceil(math.log2(len(commits) + 1)) if commits else 0

    while lo <= hi:
        mid = (lo + hi) // 2
        commit = commits[mid]
        step += 1

        logger.info('Step %d/%d: testing commit %s (index %d/%d)',
                    step, total_steps, commit[:7], mid, len(commits) - 1)

        # Build bisect command
        cmd = ['bash', bisect_script,
               '--image', image,
               '--commit', commit,
               '--model-path', model_path,
               '--home-dir', _cfg.HOST_HOME_DIR,
               '--port', str(port)]

        if mode == 'launch':
            cmd.append('--launch-only')
        else:
            cmd += ['--baseline-metric', str(baseline_metric),
                    '--metric-name', metric_name,
                    '--concurrency', str(concurrency)]

        if launch_script:
            cmd += ['--launch-script', launch_script]

        env = dict(os.environ)
        if _cfg.USE_SUDO_DOCKER:
            env['USE_SUDO_DOCKER'] = '1'

        proc = subprocess.run(cmd, env=env,
                              capture_output=True, text=True)
        rc = proc.returncode

        if rc == 0:
            verdict = 'GOOD'
        elif rc == 1:
            verdict = 'BAD'
        else:
            verdict = 'SKIP'

        # Parse metric value from stdout
        metric_value = None
        for line in proc.stdout.splitlines():
            if line.startswith(f'Metric {metric_name}:'):
                try:
                    metric_value = float(
                        line.split('result=')[1].split(',')[0]
                    )
                except (IndexError, ValueError):
                    pass

        entry = {
            'step': step,
            'commit': commit,
            'verdict': verdict,
            'metric_value': metric_value,
            'exit_code': rc,
        }
        bisect_log.append(entry)

        # Print progress
        if mode == 'launch':
            print(f'  Step {step}/{total_steps}: {commit[:7]} -> {verdict}')
        else:
            print(f'  Step {step}/{total_steps}: {commit[:7]} -> {verdict}'
                  f' (metric={metric_value})')

        # Binary search logic
        if verdict == 'GOOD':
            lo = mid + 1
        elif verdict == 'BAD':
            hi = mid - 1
        else:  # SKIP
            logger.warning('Commit %s skipped (exit code %d), trying neighbor',
                           commit[:7], rc)
            if mid + 1 <= hi:
                lo = mid + 1
            elif mid - 1 >= lo:
                hi = mid - 1
            else:
                break

    # Determine first bad commit
    first_bad_idx = lo if lo < len(commits) else len(commits) - 1
    first_bad_commit = commits[first_bad_idx]

    # Get commit subject via docker
    docker = _docker_cmd()
    container_name = 'bisect-subject-tmp'
    subprocess.run(docker + ['rm', '-f', container_name],
                   capture_output=True)
    first_bad_subject = ''
    try:
        subprocess.run(
            docker + ['run', '-d', '--name', container_name,
                      image, 'bash', '-c', 'sleep infinity'],
            check=True, capture_output=True
        )
        result = subprocess.run(
            docker + ['exec', container_name,
                      'git', '-C', repo_path,
                      'log', '--format=%s', '-1', first_bad_commit],
            capture_output=True, text=True
        )
        first_bad_subject = result.stdout.strip()
    finally:
        subprocess.run(docker + ['rm', '-f', container_name],
                       capture_output=True)

    return {
        'first_bad_commit': first_bad_commit,
        'first_bad_subject': first_bad_subject,
        'total_steps': step,
        'log': bisect_log,
    }


# ── Text formatting ──────────────────────────────────────────────────────


def _format_text(diff, run_id_a, run_id_b, suggestions=None,
                 bisect_plans=None) -> str:
    """Render a human-readable text report."""
    lines = []
    lines.append(f'Version Diff: Run #{run_id_a} vs Run #{run_id_b}')
    lines.append('')

    if diff['changed']:
        lines.append('CHANGED:')
        for c in diff['changed']:
            old_v = c['old_version'] or '?'
            new_v = c['new_version'] or '?'
            old_c = f' ({c["old_commit"][:7]})' if c.get('old_commit') else ''
            new_c = f' ({c["new_commit"][:7]})' if c.get('new_commit') else ''
            lines.append(f'  {c["name"]:<20s} {old_v}{old_c} -> {new_v}{new_c}')
        lines.append('')

    if diff['unchanged']:
        lines.append('UNCHANGED:')
        for u in diff['unchanged']:
            lines.append(f'  {u["name"]:<20s} {u["version"] or "?"}')
        lines.append('')

    if diff['added']:
        lines.append('ADDED (in run B only):')
        for a in diff['added']:
            lines.append(f'  {a["name"]:<20s} {a["version"] or "?"}')
        lines.append('')

    if diff['removed']:
        lines.append('REMOVED (in run A only):')
        for r in diff['removed']:
            lines.append(f'  {r["name"]:<20s} {r["version"] or "?"}')
        lines.append('')

    if suggestions:
        lines.append('Root Cause Analysis:')
        for i, s in enumerate(suggestions, 1):
            lines.append(f'  {i}. {s["name"]} -- {s["reason"]}')
        lines.append('')

    if bisect_plans:
        for plan in bisect_plans:
            lines.append(f'Bisect Plan ({plan["name"]}):')
            lines.append(f'  Good: {plan["good_commit"][:7]}  Bad: {plan["bad_commit"][:7]}')
            lines.append(f'  {plan["commit_range_cmd"]}')
            lines.append('')

    return '\n'.join(lines)


def _lookup_run_id_by_tag(conn: sqlite3.Connection, tag: str):
    """Find the most recent run_id for a given image_tag."""
    row = conn.execute(
        'SELECT id FROM benchmark_runs WHERE image_tag = ? '
        'ORDER BY id DESC LIMIT 1',
        (tag,)
    ).fetchone()
    if row:
        return row['id']
    return None


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare library versions between two benchmark runs.')
    parser.add_argument('--run-a', type=int, default=None,
                        help='First run ID (baseline / good)')
    parser.add_argument('--run-b', type=int, default=None,
                        help='Second run ID (current / bad)')
    parser.add_argument('--tag-a', type=str, default=None,
                        help='First image tag (alternative to --run-a)')
    parser.add_argument('--tag-b', type=str, default=None,
                        help='Second image tag (alternative to --run-b)')
    parser.add_argument('--analyze', action='store_true',
                        help='Include root cause analysis and bisect plan')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='Output format')
    parser.add_argument('--db', type=str, default=None,
                        help='Path to database')
    parser.add_argument('--bisect', action='store_true',
                        help='Run automatic bisect after analysis')
    parser.add_argument('--bisect-mode', choices=['benchmark', 'launch'],
                        default='benchmark',
                        help='Bisect mode')
    parser.add_argument('--bisect-metric', type=str,
                        default='output_throughput',
                        help='Metric to use for bisect comparison')
    parser.add_argument('--bisect-concurrency', type=int, default=1,
                        help='Concurrency for bisect benchmarks')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Model path for bisect benchmarks')
    parser.add_argument('--launch-script', type=str, default=None,
                        help='Custom launch script for bisect')
    parser.add_argument('--port', type=int, default=30000,
                        help='Port for the server')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    db_path = args.db or str(DB_PATH)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        # Resolve tags to run IDs if needed
        run_a = args.run_a
        run_b = args.run_b

        if args.tag_a:
            run_a = _lookup_run_id_by_tag(conn, args.tag_a)
            if run_a is None:
                print(f'Error: no run found for tag {args.tag_a!r}',
                      file=sys.stderr)
                sys.exit(1)

        if args.tag_b:
            run_b = _lookup_run_id_by_tag(conn, args.tag_b)
            if run_b is None:
                print(f'Error: no run found for tag {args.tag_b!r}',
                      file=sys.stderr)
                sys.exit(1)

        if run_a is None or run_b is None:
            print('Error: must specify --run-a/--run-b or --tag-a/--tag-b',
                  file=sys.stderr)
            sys.exit(1)

        # Compute diff
        diff = compute_version_diff(conn, run_a, run_b)

        suggestions = None
        bisect_plans = None
        if args.analyze:
            suggestions = suggest_root_cause(diff)
            bisect_plans = generate_bisect_plan(diff, conn, run_a, run_b)

        # Output
        if args.format == 'json':
            output = {'diff': diff}
            if suggestions is not None:
                output['suggestions'] = suggestions
            if bisect_plans is not None:
                output['bisect_plans'] = bisect_plans
            print(json.dumps(output, indent=2))
        else:
            print(_format_text(diff, run_a, run_b, suggestions, bisect_plans))

        # Bisect
        if args.bisect:
            if not bisect_plans:
                bisect_plans = generate_bisect_plan(diff, conn, run_a, run_b)

            model_path = args.model_path or _cfg.MODEL_PATH

            for plan in bisect_plans:
                image = _get_image_for_run(conn, run_b)
                if not image:
                    print(f'Error: no image found for run {run_b}',
                          file=sys.stderr)
                    continue

                if args.bisect_mode == 'launch':
                    # Determine good/bad based on run status
                    status_a = _get_run_status(conn, run_a)
                    status_b = _get_run_status(conn, run_b)

                    if status_a == 'completed' and status_b != 'completed':
                        # run_a is good, run_b is bad -- default order
                        pass
                    elif status_b == 'completed' and status_a != 'completed':
                        # Swap: run_b is good, run_a is bad
                        plan['good_commit'], plan['bad_commit'] = (
                            plan['bad_commit'], plan['good_commit'])
                    # If both completed or both failed, keep default order

                    result = run_bisect(
                        plan=plan, image=image, model_path=model_path,
                        mode='launch',
                        launch_script=args.launch_script,
                        port=args.port,
                    )
                else:
                    # Benchmark mode: compare metrics
                    metric_a = _get_baseline_metric(
                        conn, run_a, args.bisect_metric,
                        args.bisect_concurrency)
                    metric_b = _get_baseline_metric(
                        conn, run_b, args.bisect_metric,
                        args.bisect_concurrency)

                    if metric_a is None or metric_b is None:
                        print(f'Error: could not get metric '
                              f'{args.bisect_metric} for both runs',
                              file=sys.stderr)
                        continue

                    # Determine which is "good" based on metric direction
                    # For throughput: higher is better
                    # For latency: lower is better
                    is_latency = 'latency' in args.bisect_metric.lower()

                    if is_latency:
                        # Lower is better: run with lower metric is "good"
                        if metric_a <= metric_b:
                            baseline_metric = metric_a
                        else:
                            baseline_metric = metric_b
                            plan['good_commit'], plan['bad_commit'] = (
                                plan['bad_commit'], plan['good_commit'])
                    else:
                        # Higher is better: run with higher metric is "good"
                        if metric_a >= metric_b:
                            baseline_metric = metric_a
                        else:
                            baseline_metric = metric_b
                            plan['good_commit'], plan['bad_commit'] = (
                                plan['bad_commit'], plan['good_commit'])

                    result = run_bisect(
                        plan=plan, image=image, model_path=model_path,
                        mode='benchmark',
                        baseline_metric=baseline_metric,
                        metric_name=args.bisect_metric,
                        concurrency=args.bisect_concurrency,
                        launch_script=args.launch_script,
                        port=args.port,
                    )

                print(f'\nBisect result for {plan["name"]}:')
                print(f'  First bad commit: {result["first_bad_commit"][:7]}')
                print(f'  Subject: {result["first_bad_subject"]}')
                print(f'  Steps: {result["total_steps"]}')

    finally:
        conn.close()


if __name__ == '__main__':
    main()
