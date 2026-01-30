# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models, providing low-latency, high-throughput inference. The workspace contains several interconnected projects under `sglang/`:

- **python/sglang/** — Main Python package (frontend language APIs + serving runtime)
- **sgl-kernel/** — Optimized CUDA/C++ compute kernels
- **sgl-model-gateway/** — Rust-based routing/API gateway (OpenAI-compatible)

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

### sgl-model-gateway (Rust)
```bash
cd sglang/sgl-model-gateway && make build
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

### Testing
```bash
# Single test file
python3 test/srt/test_srt_endpoint.py

# Single test case
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Test suite (per-commit)
python3 test/srt/run_suite.py --suite per-commit

# sgl-kernel tests
cd sglang/sgl-kernel && pytest tests/
```

## Architecture

### Frontend (`python/sglang/lang/`)
Language-level API for composing LLM programs. Key files:
- `api.py` — Public API exports (Engine, function, gen, etc.)
- `interpreter.py` — Program interpreter
- `ir.py` — Intermediate representation
- `backend/` — Backend connectors (OpenAI, Anthropic, LiteLLM, VertexAI, SGLang runtime)

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

### sgl-model-gateway (`sgl-model-gateway/`)
Rust async service (tokio/tonic) providing HTTP and gRPC routing, load balancing, and an OpenAI-compatible API layer. Has Python and Go bindings in `bindings/`.

### Key Design Patterns
- **RadixAttention**: Prefix-aware KV cache reuse across requests
- **Continuous batching**: Dynamic request scheduling for throughput
- **Paged attention**: Virtual memory management for KV cache
- **Elastic expert parallelism**: Dynamic MoE expert distribution

## Version Management

Versions are managed via `setuptools-scm` from git tags. Generated gRPC files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded from linting.

## Hardware Support

NVIDIA CUDA (primary), AMD ROCm/HIP, Intel Gaudi, Ascend NPU, Google TPU (experimental). Hardware-specific code lives in `srt/hardware_backend/`.

## Key Dependencies

PyTorch >= 2.8.0, Transformers 4.57.1, Flash Attention, Triton (kernels), FastAPI/Uvicorn (HTTP), tonic (gRPC/Rust), outlines/xgrammar/llguidance (structured generation).
