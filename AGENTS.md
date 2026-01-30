# Repository Guidelines

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
