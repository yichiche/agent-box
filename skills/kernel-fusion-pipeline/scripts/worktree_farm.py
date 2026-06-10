#!/usr/bin/env python3
"""Create git worktrees for parallel fusion development."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

STATE_DIR = Path.home() / ".kernel-fusion-pipeline"
DEFAULT_STATE = STATE_DIR / "state.json"


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "fusion"


def git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(cmd)} failed in {cwd}:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def worktree_root(sglang_root: Path) -> Path:
    return Path.home() / ".kernel-fusion-pipeline" / "worktrees" / sglang_root.name


def create_worktree(sglang_root: Path, slug: str, base_ref: str | None) -> dict:
    root = worktree_root(sglang_root)
    root.mkdir(parents=True, exist_ok=True)
    wt_path = root / slug
    branch = f"fusion/{slug}"
    ref = base_ref or git(["rev-parse", "HEAD"], sglang_root)

    existing = git(["worktree", "list", "--porcelain"], sglang_root)
    if str(wt_path) in existing:
        return {"slug": slug, "path": str(wt_path), "branch": branch, "existed": True}

    if wt_path.exists():
        raise RuntimeError(f"Path exists but is not a worktree: {wt_path}")

    branches = git(["branch", "--list", branch], sglang_root)
    if branches:
        git(["worktree", "add", str(wt_path), branch], sglang_root)
    else:
        git(["worktree", "add", "-b", branch, str(wt_path), ref], sglang_root)

    return {"slug": slug, "path": str(wt_path), "branch": branch, "base_ref": ref}


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--slugs", nargs="+", required=True)
    parser.add_argument("--base-ref", default=None)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()

    sglang_root = Path(args.sglang_root).resolve()
    state = load_state(args.state)
    worktrees: dict[str, dict] = state.get("worktrees", {})

    for raw in args.slugs:
        slug = slugify(raw)
        info = create_worktree(sglang_root, slug, args.base_ref)
        worktrees[slug] = info

    state["worktrees"] = worktrees
    state["sglang_root"] = str(sglang_root)
    save_state(args.state, state)
    print(json.dumps(list(worktrees.values()), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
