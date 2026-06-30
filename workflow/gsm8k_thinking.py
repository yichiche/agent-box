#!/usr/bin/env python3
"""Thin wrapper around sglang's gsm8k bench_sglang.py for THINKING models.

perf_sweep.sh invokes `python3 $GSM8K_SCRIPT <args>` without a generation-length
flag, and bench_sglang.py defaults --max-new-tokens to 512, which truncates a
thinking model (e.g. Qwen3.5) and tanks gsm8k accuracy (observed 0.69, 9.5%
invalid). This wrapper forwards all caller args and appends a larger
--max-new-tokens (env GSM8K_MAX_NEW_TOKENS, default 8192) when the caller did
not already specify one. Set GSM8K_SCRIPT to this file.
"""
import os
import sys
import runpy

REAL = os.environ.get(
    "GSM8K_REAL_SCRIPT",
    "/sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py",
)
max_new = os.environ.get("GSM8K_MAX_NEW_TOKENS", "8192")

if "--max-new-tokens" not in sys.argv:
    sys.argv += ["--max-new-tokens", max_new]
sys.argv[0] = REAL
runpy.run_path(REAL, run_name="__main__")
