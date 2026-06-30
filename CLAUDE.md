# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models, providing low-latency, high-throughput inference. The workspace contains several interconnected projects under `sglang/`:

- **python/sglang/** — Main Python package (frontend language APIs + serving runtime)
- **sgl-kernel/** — Optimized CUDA/C++ compute kernels
- **sgl-model-gateway/** — Rust-based routing/API gateway (OpenAI-compatible)

## Setup

After cloning, run setup (initializes submodules, activates git hooks):
```bash
bash setup-hooks.sh
```

## Commit & PR Conventions

All commit messages and PR titles must follow: `[Tag] Description` .

Allowed tags: `[Feature]`, `[Fix]`, `[Refactor]`, `[Docs]`, `[Test]`, `[CI]`, `[Chore]`, `[Perf]`

Trailers like `Co-Authored-By:` is not allowed to show nerither in the commit body or the subject line.

## Build & Development Commands

### Python package (main runtime)
```bash
cd sglang/python && pip install -e .
```

### sgl-kernel (CUDA kernels)
```bash
cd sglang/sgl-kernel && make build
# Resource-limited: make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### Running the server
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-hf
# or: sglang serve --model-path meta-llama/Llama-2-7b-hf
```

### Linting
```bash
pre-commit run --all-files
```
Hooks: isort, ruff (F401/F821), black-jupyter, codespell, clang-format, nbstripout.


## Architecture
### Serving Runtime (`python/sglang/srt/`)
The core inference engine. Major subsystems:
- **`entrypoints/`** — Engine entry points, gRPC server
- **`managers/`** — Scheduler and model executor managers (orchestrate batching, scheduling)
- **`model_executor/`** — Model execution layer (forward passes, memory management)
- **`models/`** — 70+ model implementations (llama, qwen, deepseek, mistral, etc.)
- **`layers/`** — Reusable model layers (attention, MLP, quantization, etc.)
- **`sampling/`** — Sampling algorithms
- **`speculative/`** — Speculative decoding
- **`distributed/`** — Multi-GPU tensor/pipeline/expert parallelism
- **`disaggregation/`** — Prefill-decode disaggregation (separate prefill/decode clusters)
- **`constrained/`** — Structured output generation (JSON schema, regex, grammar)
- **`lora/`** — LoRA adapter support with multi-adapter batching
- **`multimodal/`** — Vision and audio processing
- **`hardware_backend/`** — Hardware-specific optimizations (CUDA, HIP, HPU, etc.)

### sgl-kernel (`sgl-kernel/`)
CMake-based C++/CUDA project with Python bindings. Contains optimized kernels for attention, quantization, and other compute-intensive operations. Built via scikit-build-core.

### Key Design Patterns
- **RadixAttention**: Prefix-aware KV cache reuse across requests
- **Continuous batching**: Dynamic request scheduling for throughput
- **Paged attention**: Virtual memory management for KV cache
- **Elastic expert parallelism**: Dynamic MoE expert distribution

## Version Management

Versions are managed via `setuptools-scm` from git tags. Generated gRPC files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded from linting.

## agent-box Layout

```
memory/      — Obsidian-style long-term vault (models, workflows, gotchas, script catalog)
skills/      — Slash-command procedures (/validate, /perf-sweep, /memory-capture, …)
benchmark/   — Performance benchmarking (run, compare, analyze CSVs)
profile/     — Profiler trace analysis & model structure inspection (git submodule → torch-profiler-parser)
debug/       — Regression detection (perf-regression subsystem)
configs/     — Shared model configuration files
env.sh       — Central environment config (HOST_HOME, AGENT_BOX_DIR)
```

## Long-Term Memory

- **Index:** `memory/MEMORY.md` — route to model cards, workflows, gotchas
- **Model registry:** `memory/models/INDEX.md` — which `~/run_*.sh` + accuracy threshold per model
- **Capture:** `/memory-capture` — promote session learnings into vault
- **Consolidate:** `/memory-consolidate` — sync `~/.claude/projects/*/memory/`, refresh this file + AGENTS.md
- **Source of truth:** `agent-box/memory/` (not per-project Claude memory shards)
- **Cross-container bridge:** `memory/remote/` — STATUS / INBOX / OUTBOX; skill `/remote-bridge`

## Profiling & Trace Analysis

- When analyzing profiling outputs (trace Excel, evaluation CSV, raw traces, kernel performance), use `/profile` or see `profile/profile.md`.
- Trace parser: `trace_module_analyzer.py` (nn.Module correlation-based).
- Quality evaluator: `evaluate_module_parsing.py` (structural scoring for trace_module_analyzer output).

## Skills (Slash Commands)

- **`/memory-capture`** — Save a gotcha, model config, or workflow into `memory/`
- **`/remote-bridge`** — Host ↔ container STATUS + INBOX/OUTBOX; optional native `/remote-control`
- **`/memory-consolidate`** — Import Claude memory shards, refresh AGENTS.md + this file
- **`/validate`** — Baseline + accuracy + profile + after benchmark for PRs
- **`/perf-sweep`** — Accuracy-gated concurrency sweep (model-agnostic via env)
- **`/benchmark`** — Before/after e2e benchmark comparison
- **`/generate-profile`** — Capture Chrome-compatible trace
- **`/parse-trace`** — Run trace_module_analyzer (prefill/decode modes)
- **`/gpu-status`** — Free MI355 GPUs + correct HIP_VISIBLE_DEVICES
- **`/commit`** — Stage changes, ensure you're on a feature branch (creates one if on main), and commit with an `[AMD]` prefixed message. Usage: `/commit` or `/commit <description>`.
- **`/pr`** — Push the branch and create a GitHub PR with the full SGLang template (Motivation, Modifications, Accuracy Tests, Benchmarking, Checklist, Review Process). Usage: `/pr` or `/pr <title>`.
- **`/profile`** — Analyze profiling outputs (trace Excel, evaluation CSV, raw traces) for kernel performance, layer breakdown, and model behavior. Usage: `/profile` or `/profile <question or path>`. Full guide at `profile/profile.md`.