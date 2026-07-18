# Repository Guidelines

## Long-Term Memory (`memory/`)

Cross-model knowledge vault (Cursor, Claude Code, Codex). Start at [`memory/MEMORY.md`](memory/MEMORY.md).

**Top gotchas (read before benchmarking):**
- Launch server/client from `/tmp`, not `$HOME` — stale `aiter/`/`sglang/` shadows ([`memory/gotchas/bench-cwd-shadow.md`](memory/gotchas/bench-cwd-shadow.md))
- Never commit SGLang changes on `main` — feature branch only ([`memory/gotchas/sglang-branch-hygiene.md`](memory/gotchas/sglang-branch-hygiene.md))
- GPU index: use `/gpu-status`; never empty `HIP_VISIBLE_DEVICES` ([`memory/gotchas/gpu-pinning.md`](memory/gotchas/gpu-pinning.md))
- Container tasks run Claude Code via API key (`agent-box/claude-code-key.sh`), never `claude /login` subscription — see CLAUDE.md § Container Task Execution

**Model → scripts:** see [`memory/models/INDEX.md`](memory/models/INDEX.md) (e.g. Qwen3.5 MXFP4 → `~/run_qwen3.5_mxfp4_perf.sh` + `~/run_qwen3.5_mxfp4_inferencemax_client.sh`, GSM8K ≥ 0.92).

## Benchmark, Profiling & Kernel-Dev Conventions (load-bearing — every session honors)

Detail in [`memory/workflows/`](memory/workflows/); the numbers below are defaults, not suggestions.

- **num_prompts** — benchmark/sweep `num_prompts = conc × 10`; profiling capture `× 2`. ([`benchmark`](memory/workflows/benchmark.md), [`profiling`](memory/workflows/profiling.md))
- **Benchmark shapes** — with no shape given, run **both** `diag-1k` (IL1024/OL1024) and `canonical-8k` (IL8192/OL1024); report both, but **claim perf only on `canonical-8k`**. ([`workloads`](memory/workflows/workloads.md))
- **Reference table** — when the model card has a reference table (e.g. [`qwen35-mxfp4-mi355-reference.csv`](memory/models/qwen35-mxfp4-mi355-reference.csv)), the default benchmark output is measured **side by side with the reference** (per-cell delta), columns `Median E2E / total tok/s / tok/s/gpu / TTFT / TPOT`. ([`benchmark`](memory/workflows/benchmark.md))
- **Profiling anchors** — profile at **conc4 / conc64** (these are also the kernel-confirm anchors). ([`profiling`](memory/workflows/profiling.md))
- **Kernel-dev loop** — unit test in aiter first (shapes reflecting conc4~conc128; improvement = **geomean across conc4~conc128**, not one shape) → profile c4/c64 to confirm on the served trace → no improvement ⇒ back to unit test → geomean **>10% ⇒ escalate to e2e full sweep** → keep/ship still per gates (≥30% served-trace, stacked ship). ([`kernel-dev`](memory/workflows/kernel-dev.md), [`gates`](memory/workflows/gates.md))
- **Time budget → scope** — a long budget (e.g. 8h) means *explore broadly* (kernel rewrites, fusions, algorithm/layout/quant variants, tuned configs, in parallel worktrees), **not** flipping one SGLang global var at a time. ([`time-budget`](memory/workflows/time-budget.md))

**Workflows:** benchmark [`memory/workflows/benchmark.md`](memory/workflows/benchmark.md) · validate [`memory/workflows/validate.md`](memory/workflows/validate.md) · profile [`memory/workflows/profiling.md`](memory/workflows/profiling.md)

**Maintenance:** session shards auto-converge into `memory/journal/YYYY-MM/` (a Stop hook runs `memory/bin/memory-sync.sh`; provenance in `memory/meta/provenance.tsv`). `/memory-capture` after sessions · `/memory-consolidate` weekly promotes journal facts and refreshes this file · `/skill-suggest` drafts workflow improvements.

**Cross-container:** Host ↔ GPU container agents via [`memory/bridge/`](memory/bridge/README.md) — `/remote-bridge`: file bus (STATUS/INBOX/OUTBOX) + `bridge.sh exec` (allowlisted `docker exec` into your own containers). Memory sharing rides the `$HOME/.claude` + `$HOME/.codex` mounts, so it's host-local — the bridge is only for live coordination.

## Scope of This Workspace
This workspace is a multi-repo collection. Each top-level folder (`sglang/`, `Mooncake/`, `aiter/`, `triton-custom/`, `fast-hadamard-transform/`) is its own Git repo; run commands from the relevant subdirectory.

## Project Structure & Module Organization
- `sglang/`: LLM serving framework. Core Python package in `sglang/python/sglang/`; tests in `sglang/test/`; benchmarks in `sglang/benchmark/`; docs in `sglang/docs/`; kernels in `sglang/sgl-kernel/`; gateway in `sglang/sgl-model-gateway/`.
- `Mooncake/`: KV-cache transfer/store platform. Components in `mooncake-transfer-engine/`, `mooncake-store/`, `mooncake-p2p-store/`; docs in `doc/` and `docs/`.
- `aiter/`: AMD operator library. Python package in `aiter/aiter/`; C++/kernel sources in `aiter/csrc/`; operator tests in `aiter/op_tests/`.
- `triton-custom/`: Triton compiler fork. Python in `triton-custom/python/`; C++/LLVM in `triton-custom/lib/`; docs in `triton-custom/docs/`.
- `fast-hadamard-transform/`: CUDA/PyTorch extension with `setup.py` and a Python import API.

## Build, Test, and Development Commands
- `sglang/`:
  - Install from source: `pip install -e "python"` (run in `sglang/`).
  - Tests: `cd sglang/test/srt && python3 test_srt_endpoint.py` or `python3 run_suite.py --suite per-commit`.
  - Docs: `cd sglang/docs && make html` (after `pip install -r requirements.txt`).
- `Mooncake/`:
  - Build from source: `bash dependencies.sh && mkdir build && cd build && cmake .. && make -j` (then `sudo make install`).
  - PyPI wheel: `pip3 install mooncake-transfer-engine --upgrade`.
- `aiter/`:
  - Dev install: `python3 setup.py develop` (or `pip install -e .`).
  - Run an op test: `python3 op_tests/test_layernorm2d.py`.
- `triton-custom/`:
  - Dev install: `pip install -r python/requirements.txt && pip install -e .`.
  - Tests: `make test` (GPU) or `make test-nogpu`.
- `fast-hadamard-transform/`:
  - Usage entry point: `from fast_hadamard_transform import hadamard_transform`.

## Coding Style & Naming Conventions
- Respect per-repo format configs: `.editorconfig`, `.clang-format`, and `.pre-commit-config.yaml` (notably in `sglang/` and `triton-custom/`).
- Follow language-native formatters where present (e.g., `rustfmt` in `sglang/sgl-model-gateway/`).
- Keep test file names aligned with existing patterns (`test_*.py` in `sglang/test/` and `sglang/sgl-kernel/tests/`).

## Testing Guidelines
- `sglang/` uses `unittest` for most suites and runs tests via `python3 test_file.py`; some areas use `pytest` for specific modules.
- Add new SGLang tests under `sglang/test/srt` or `sglang/test/lang` and register suites in `run_suite.py`.

## Commit & Pull Request Guidelines
- Commit messages in subrepos are short, imperative, and often include scopes or tags (e.g., `fix(ci): …`, `[AMD] …`) plus a PR number `(#1234)`.
- There is no root PR template; follow each repo’s contributor docs (e.g., `Mooncake/CONTRIBUTING.md` and `sglang/docs/README.md` for pre-commit expectations).
