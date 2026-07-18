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
memory/      — Long-term vault: journal/ (auto-imported shards), gotchas/models/workflows (curated), bin/ (memory-sync, skill-suggest), bridge/ (host↔container), meta/ (provenance, log)
skills/      — Slash-command procedures (/validate, /perf-sweep, /memory-capture, …)
benchmark/   — Performance benchmarking (run, compare, analyze CSVs)
profile/     — Profiler trace analysis & model structure inspection (git submodule → torch-profiler-parser)
debug/       — Regression detection (perf-regression subsystem)
configs/     — Shared model configuration files
env.sh       — Central environment config (HOST_HOME, AGENT_BOX_DIR, AGENT_SCRATCH_DIR, AGENT_RUNS_DIR)
```

## Scratch & Output File Hygiene

**Never write agent-created files into `$HOST_HOME` (`/home/yichiche`) root.** It clutters
the home dir. Use the defaults from `env.sh`:

- **Ad-hoc scripts / logs / reports** (e.g. `summarize_*.py`, `*_driver.sh`,
  `tmp_*.log`, one-off `*_report.md`, quick analysis) → **`$AGENT_SCRATCH_DIR`**
  (`$HOST_HOME/agent-scratch`), ideally under a per-task subdir.
- **Structured run outputs** (traces, sweep result dirs) → **`$AGENT_RUNS_DIR`**
  (`$HOST_HOME/agent-runs`) or the existing per-model dirs (`agent-runs/<model>/<label>`).
- **Truly ephemeral** one-shot temp → `/tmp`.

Source `env.sh` to get the paths: `source "$AGENT_BOX_DIR/env.sh"`. Create the target
dir before writing (`mkdir -p "$AGENT_SCRATCH_DIR/<task>"`). **Exception:** canonical,
human-owned files already at `$HOST_HOME` — especially `~/run_*.sh` model launch
scripts — stay where they are; do not move or relocate them.

## Container Task Execution

**Any task executed inside a container must run Claude Code with the API key, never the personal claude.ai subscription.**
- Source `agent-box/claude-code-key.sh` in the container — it wires `ANTHROPIC_BASE_URL`/`ANTHROPIC_CUSTOM_HEADERS` to the AMD gateway API key (from `~/.claude_api_key`), so the session bills against the API key instead of the host's subscription.
- Do **not** run `claude /login` inside a container for normal task execution — that authenticates against the personal claude.ai subscription and burns subscription tokens instead.
- Exception: `claude /login` + `claude --remote-control` inside a container is fine when the explicit goal is Native Remote Control (phone / claude.ai/code live session) — see `memory/bridge/README.md`. That's a deliberate opt-in, not the default for running tasks.

## Long-Term Memory

- **Index:** `memory/MEMORY.md` — route to model cards, workflows, gotchas; has the dataflow diagram
- **Model registry:** `memory/models/INDEX.md` — which `~/run_*.sh` + accuracy threshold per model
- **Auto-converge:** `memory/bin/memory-sync.sh` copies Claude/Codex session shards into `memory/journal/YYYY-MM/` (verbatim, sha-deduped, provenance in `meta/provenance.tsv`). Runs via a Claude Code **Stop hook**; disable with `--uninstall-hook`.
- **Capture:** `/memory-capture` — promote session learnings into the curated vault
- **Consolidate:** `/memory-consolidate` — promote journal facts, refresh this file + AGENTS.md
- **Suggest:** `/skill-suggest` — draft workflow/skill stubs from recurring journal themes (review in `meta/suggestions/`)
- **Source of truth:** `agent-box/memory/` (git-tracked); journal is raw history, curated dirs are canonical
- **Cross-container bridge:** `memory/bridge/` — `/remote-bridge`: file bus + `bridge.sh exec` (allowlisted `docker exec` into your own containers)

## Benchmark, Profiling & Kernel-Dev Conventions

Load-bearing defaults every session must honor. Detail lives in `memory/workflows/`;
these numbers are defaults, not suggestions.

- **num_prompts** — benchmark/sweep `num_prompts = conc × 10`; profiling capture `× 2`.
- **Benchmark shapes** — no shape given ⇒ run **both** `diag-1k` (IL1024/OL1024) and `canonical-8k` (IL8192/OL1024); report both, **claim perf only on `canonical-8k`**.
- **Reference table** — when the model card has one (e.g. `memory/models/qwen35-mxfp4-mi355-reference.csv`), the default benchmark output is measured **side by side with the reference** (per-cell delta), columns `Median E2E / total tok/s / tok/s/gpu / TTFT / TPOT`.
- **Profiling anchors** — profile at **conc4 / conc64** (also the kernel-confirm anchors).
- **Kernel-dev loop** (`memory/workflows/kernel-dev.md`) — unit test in aiter first (shapes reflecting conc4~conc128; improvement = **geomean across conc4~conc128**, not one shape) → profile c4/c64 to confirm on the served trace → no improvement ⇒ back to unit test → geomean **>10% ⇒ escalate to e2e full sweep** → keep/ship still per `gates.md` (≥30% served-trace, stacked ship).
- **Time budget → scope** (`memory/workflows/time-budget.md`) — a long budget (e.g. 8h) means *explore broadly* (kernel rewrites, fusions, algorithm/layout/quant variants, tuned configs, in parallel worktrees), **not** flipping one SGLang global var at a time.

## Profiling & Trace Analysis

- When analyzing profiling outputs (trace Excel, evaluation CSV, raw traces, kernel performance), use `/profile` or see `profile/profile.md`.
- Trace parser: `trace_module_analyzer.py` (nn.Module correlation-based).
- Quality evaluator: `evaluate_module_parsing.py` (structural scoring for trace_module_analyzer output).

## Skills (Slash Commands)

- **`/memory-capture`** — Save a gotcha, model config, or workflow into `memory/`
- **`/remote-bridge`** — Host ↔ container: file bus + `bridge.sh exec` (allowlisted `docker exec`, claude/codex headless); optional native `/remote-control`
- **`/skill-suggest`** — Draft workflow/skill stubs from recurring memory themes (detect → draft → approve)
- **`/memory-consolidate`** — Promote journal facts, refresh AGENTS.md + this file
- **`/validate`** — Baseline + accuracy + profile + after benchmark for PRs
- **`/perf-sweep`** — Accuracy-gated concurrency sweep (model-agnostic via env)
- **`/benchmark`** — Before/after e2e benchmark comparison
- **`/generate-profile`** — Capture Chrome-compatible trace
- **`/parse-trace`** — Run trace_module_analyzer (prefill/decode modes)
- **`/gpu-status`** — Free MI355 GPUs + correct HIP_VISIBLE_DEVICES
- **`/commit`** — Stage changes, ensure you're on a feature branch (creates one if on main), and commit with an `[AMD]` prefixed message. Usage: `/commit` or `/commit <description>`.
- **`/pr`** — Push the branch and create a GitHub PR with the full SGLang template (Motivation, Modifications, Accuracy Tests, Benchmarking, Checklist, Review Process). Usage: `/pr` or `/pr <title>`.
- **`/profile`** — Analyze profiling outputs (trace Excel, evaluation CSV, raw traces) for kernel performance, layer breakdown, and model behavior. Usage: `/profile` or `/profile <question or path>`. Full guide at `profile/profile.md`.