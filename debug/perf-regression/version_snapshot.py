#!/usr/bin/env python3
"""Collect software version snapshots inside a Docker container.

This script is self-contained (no imports from config.py) so it can be
copied into and executed inside arbitrary Docker images.
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone

# ── Built-in defaults (mirror config.py values) ──────────────────────────
DEFAULT_GIT_REPOS = [
    {"name": "sglang", "path": "/sgl-workspace/sglang"},
    {"name": "aiter", "path": "/sgl-workspace/aiter", "pip_name": "amd-aiter"},
]

DEFAULT_PIP_PACKAGES = [
    "torch",
    "transformers",
    "flashinfer",
    "triton",
]


def _run(cmd: list[str], timeout: int = 30) -> str | None:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _pip_show_version(name: str) -> str | None:
    """Get the installed version of a package via pip show."""
    output = _run(["pip", "show", name])
    if output is None:
        return None
    for line in output.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def collect_git_version(name: str, path: str, pip_name: str | None = None) -> dict:
    """Collect version info for a git repository.

    Returns a dict with name, version, source_type, and git metadata.
    Falls back to pip-only info if the repo path does not exist.
    ``pip_name`` overrides ``name`` for the ``pip show`` lookup when the
    pip package name differs from the repo name (e.g. amd-aiter vs aiter).
    """
    pip_version = _pip_show_version(pip_name or name)

    git_log = _run(["git", "-C", path, "log", "-1", "--format=%H|%ai|%s"])
    if git_log is None:
        # Repo doesn't exist or git failed; return pip-only record
        return {
            "name": name,
            "version": pip_version,
            "source_type": "pip",
            "git_commit": None,
            "git_commit_date": None,
            "git_commit_subject": None,
        }

    parts = git_log.split("|", 2)
    if len(parts) < 3:
        return {
            "name": name,
            "version": pip_version,
            "source_type": "pip",
            "git_commit": None,
            "git_commit_date": None,
            "git_commit_subject": None,
        }

    return {
        "name": name,
        "version": pip_version,
        "source_type": "git",
        "git_commit": parts[0],
        "git_commit_date": parts[1],
        "git_commit_subject": parts[2],
    }


def collect_pip_versions(packages: list[str]) -> list[dict]:
    """Collect installed versions for a list of pip packages.

    Uses a single ``pip list --format=json`` call and filters to the
    requested package names.
    """
    output = _run(["pip", "list", "--format=json"])
    if output is None:
        return []

    try:
        installed = json.loads(output)
    except json.JSONDecodeError:
        return []

    lookup = {p.lower(): p for p in packages}
    results = []
    for pkg in installed:
        pkg_name_lower = pkg.get("name", "").lower()
        if pkg_name_lower in lookup:
            results.append({
                "name": lookup[pkg_name_lower],
                "version": pkg.get("version"),
                "source_type": "pip",
            })
    return results


def collect_all(
    git_repos: list[dict] | None = None,
    pip_packages: list[str] | None = None,
) -> dict:
    """Collect a full version snapshot.

    Returns a dict with ``snapshot_timestamp`` and ``libraries`` list.
    """
    if git_repos is None:
        git_repos = DEFAULT_GIT_REPOS
    if pip_packages is None:
        pip_packages = DEFAULT_PIP_PACKAGES

    libraries: list[dict] = []

    # Git-tracked repos (also include pip version)
    git_names = set()
    for repo in git_repos:
        info = collect_git_version(repo["name"], repo["path"], repo.get("pip_name"))
        libraries.append(info)
        git_names.add(repo["name"].lower())

    # Pip-only packages (skip if already covered by git repos)
    remaining = [p for p in pip_packages if p.lower() not in git_names]
    libraries.extend(collect_pip_versions(remaining))

    return {
        "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        "libraries": libraries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect software version snapshot inside a container."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format", "-f", dest="fmt", choices=["json", "csv"], default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    snapshot = collect_all()

    if args.fmt == "json":
        text = json.dumps(snapshot, indent=2)
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["name", "version", "source_type",
                         "git_commit", "git_commit_date", "git_commit_subject"],
        )
        writer.writeheader()
        for lib in snapshot["libraries"]:
            writer.writerow(lib)
        text = buf.getvalue()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
