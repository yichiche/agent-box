---
name: atom-progress
description: "Track ATOM inference engine benchmark progress from https://rocm.github.io/ATOM/benchmark-dashboard/. Reports latest numbers, detects >5% perf changes, identifies contributing commits, and flags changes potentially useful to SGLang. Default config: DSV4 ISL=8192 / OSL=1024."
---

# atom-progress

## When to use

Trigger this skill when the user:

- Asks about ATOM benchmark status, progress, or trends
- Wants to know if ATOM performance improved recently
- Asks to compare ATOM versions or Docker image performance
- Wants to check if ATOM improvements are relevant to SGLang

## Config confirmation

**Always confirm the config with the user before running.** Use AskUserQuestion with these dimensions:

- **Model**: default `dsv4` (DeepSeek-V4-Pro). Options: `dsr1`, `qwen3.5`, `glm5`, `kimik2.5`, `gptoss`, `minimax`
- **ISL/OSL**: default `8192/1024`. Options: `1024/1024`, `8192/1024`, `1024/8192`
- **Backend**: default `ATOM`. Options: `ATOM`, `ATOM-SGLang`, `ATOM-vLLM`

If the user already specified the config (e.g., "atom progress on dsv4 8k/1k"), skip the question and run directly.

## Workflow

1. Confirm config with user (or use defaults/user-specified values).
2. Run the script:

   ```bash
   python3 ~/.claude/skills/atom-progress/scripts/atom_progress.py \
       --model dsv4 --isl 8192 --osl 1024 \
       [--backend ATOM] [--runs 10] [--threshold 5]
   ```

3. Review and relay the output. Key sections:
   - **Reproduce**: Docker pull command, recipe link (model-specific launch guide), benchmark workflow link, and Docker image tag — shown by default
   - **Latest Numbers**: current throughput/latency table
   - **Trend**: recent runs side-by-side with Docker image tags
   - **Improvements (>5%)**: commits that caused significant perf gains
   - **Regressions (>5%)**: commits that caused perf drops
   - **SGLang Relevance**: which improvement commits touch areas applicable to SGLang (attention, quantization, MoE, scheduling, etc.)

4. If there are SGLang-relevant improvements:
   - Note the commit hash and what changed
   - Explain whether the optimization is ATOM-internal or could be ported to SGLang
   - If it's a kernel-level change (GEMM, attention, quantization), check if a similar kernel exists in `aiter/` or `sgl-kernel/`

## Script flags

| Flag | Default | Behavior |
|---|---|---|
| `--model TEXT` | `dsv4` | Model alias or name |
| `--isl N` | `8192` | Input sequence length |
| `--osl N` | `1024` | Output sequence length |
| `--backend TEXT` | `ATOM` | Backend: ATOM, ATOM-SGLang, ATOM-vLLM |
| `--runs N` | `10` | Number of recent runs to show in trend |
| `--threshold N` | `5.0` | Percent change threshold for flagging |
| `--list` | off | List all available configs |
| `--json` | off | Dump raw history as JSON |
| `--commits` | off | Show commit log for all runs |

## Data source

Data is fetched from `https://rocm.github.io/ATOM/benchmark-dashboard/data.js` which contains up to 90 nightly runs. The data is updated daily (~17:00 UTC) by GitHub Actions in the ROCm/ATOM repo.

## Model aliases

| Alias | Dashboard name |
|---|---|
| `dsv4`, `deepseekv4`, `deepseek-v4-pro` | DeepSeek-V4-Pro |
| `dsr1`, `deepseek-r1` | DeepSeek-R1-0528 |
| `qwen3.5`, `qwen` | Qwen3.5-397B |
| `glm5` | GLM-5 |
| `kimik2.5`, `kimi` | Kimi-K2.5 |
| `gptoss` | gpt-oss-120b |
| `minimax` | MiniMax-M2.7 |

## SGLang relevance detection

The script flags commits whose messages mention keywords relevant to SGLang:
attention, kv_cache, scheduler, batch, prefill, decode, gemm, moe, expert, quantization, fused, flash, triton, radix, paged, speculative, mtp, fp8, fp4, int4, block scale, norm, rope, embedding, cuda graph, overlap, pipeline, dispatch, routing, topk, memory, allocat.

## Examples

**User asks about ATOM status:**

> What's the latest atom progress?

→ Confirm config (default dsv4 8k/1k), then run:
```bash
python3 ~/.claude/skills/atom-progress/scripts/atom_progress.py --model dsv4 --isl 8192 --osl 1024
```

**User specifies model:**

> atom progress on dsr1 1k/1k

→ Run directly (no need to ask):
```bash
python3 ~/.claude/skills/atom-progress/scripts/atom_progress.py --model dsr1 --isl 1024 --osl 1024
```

**User wants all available configs:**

> what models does atom benchmark?

→ Run with --list:
```bash
python3 ~/.claude/skills/atom-progress/scripts/atom_progress.py --list
```
