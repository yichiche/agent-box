#!/usr/bin/env python3
"""Compatibility wrapper. The script lives in /pr-test-seam."""
import runpy
from pathlib import Path

target = Path.home() / "agent-box/skills/pr-test-seam/test_seam.py"
runpy.run_path(str(target), run_name="__main__")
