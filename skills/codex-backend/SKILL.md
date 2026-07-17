---
name: codex-backend
description: "Point the Codex CLI at a LOCAL SGLang server as its LLM backend. Launches a model server from one of the ~/run_*.sh scripts (Qwen3.5, DeepSeek-R1/V4, etc.), ensures the server emits structured tool calls + reasoning, wires ~/.codex/config.toml to it, and verifies Codex agentic tool-use end to end. Use when the user says '/codex-backend', asks to make codex use a local server, switch the codex backend model, or serve dsr1/dsv4/qwen for codex."
category: infra
---

# Codex local backend setup

Make the **Codex CLI** talk to a **local SGLang server** instead of a remote API.
The whole point: Codex is agentic, so the server MUST return *structured* `tool_calls`
and separated `reasoning_content` — otherwise Codex prints the tool call as text and
nothing executes. A plain chat that "works" is NOT enough; always run the verification.

The user names a model/script (e.g. "dsr1-mxfp4 on 8 cards", "dsv4", "qwen3.5"). Map it
to a launch script under `/home/yichiche/run_*.sh`, make sure that script has the right
parser flags, launch it, point Codex at it, and verify.

## Inputs to resolve first

1. **Which model / launch script.** List candidates: `ls /home/yichiche/run_*.sh`.
   Common ones:
   - `run_qwen3.5_mxfp4_perf.sh` — Qwen3.5 MXFP4, tp 2, port 8000
   - `run_deepseekR1_mxfp4_spec.sh` — DeepSeek-R1 MXFP4 + EAGLE, tp 8, port 9000
   - `run_dsv4.sh` (and `run_dsv4_*.sh`) — DeepSeek-V4, tp 8, port 8000
   If the user is vague, `AskUserQuestion` with the matching scripts as options.
2. **Card count / GPUs.** The script picks GPUs via `HIP_VISIBLE_DEVICES=` / `--tp`.
   If the user wants a different card count, edit `--tp`/`--tp-size` and the visible
   devices in the script before launching. Use the `/gpu-status` skill to find free GPUs
   (remember: SGLang on ROCm selects via `CUDA_VISIBLE_DEVICES`, not `HIP_*`).
3. **Port.** Read it from the script's `--port`. The helper auto-detects it.

## Required server flags (the #1 thing that breaks Codex)

Every launch script used as a Codex backend MUST pass a tool-call parser and a reasoning
parser matched to the model. If they're missing, add them to the `launch_server` command.

| Model family | `--tool-call-parser` | `--reasoning-parser` |
|---|---|---|
| Qwen3.5 / Qwen3-Coder | `qwen3_coder` | `qwen3` |
| Qwen2.5 / older Qwen   | `qwen25`      | `qwen3` |
| DeepSeek-V4            | `deepseekv4`  | `deepseek-v4` |
| DeepSeek-V3.2 / V3.1 / V3 | `deepseekv32` / `deepseekv31` / `deepseekv3` | `deepseek-v3` |
| DeepSeek-R1            | `deepseekv3`  | `deepseek-r1` |
| GLM-4.5 / 4.7          | `glm45` / `glm47` | `glm45` |
| Kimi-K2               | `kimi_k2`     | `kimi_k2` |
| gpt-oss               | `gpt-oss`     | `gpt-oss` |

This table is a starting point, not gospel. The verification probe is the source of
truth: if `TOOLCALL_OK=no`, the parser is wrong — try another from
`python3 -c "from sglang.srt.function_call.function_call_parser import FunctionCallParser as F; print(sorted(F.ToolCallParserEnum))"` and relaunch.

`run_dsv4.sh` already has the right flags; `run_qwen3.5_mxfp4_perf.sh` was fixed to use
`qwen3_coder`/`qwen3`; `run_deepseekR1_mxfp4_spec.sh` currently has NONE — add them.

## Procedure

### 1. Prepare the launch script
Open the chosen script. Confirm `--tool-call-parser` and `--reasoning-parser` are present
and correct per the table. Add/fix them in the `python3 -m sglang.launch_server` command
if needed. Adjust `--tp` / visible devices for the requested card count.

### 2. Launch + verify the server (helper, deterministic)
This kills any server already on that port, launches the script in the background, waits
until `/v1/models` responds, and probes tool-call/reasoning parsing:
```bash
bash ~/.claude/skills/codex-backend/launch_and_verify.sh /home/yichiche/<script>.sh
# add --port N to override; --no-restart to reuse a running server; --timeout SECS
```
Read the `SUMMARY` block it prints. You need `TOOLCALL_OK=yes`. If it's `no`, fix the
parser flag and re-run. Note the `BASE_URL` and `MODEL` values.

> Restarting kills the running server (a big-model reload takes minutes). If a server is
> already serving the right model, prefer `--no-restart`. If restarting an in-use server,
> confirm with the user first.

### 3. Point Codex at it
```bash
python3 ~/.claude/skills/codex-backend/patch_codex_config.py \
  --base-url "<BASE_URL>" --model "<MODEL>" --context-window "<CONTEXT_WINDOW>"
```
This backs up `~/.codex/config.toml`, sets the default `model` + `model_provider="sglang"`,
sets `[model_providers.sglang]` `base_url`/`wire_api="responses"`, and writes
`model_context_window` (= the server's `max_model_len`). Everything else (profiles,
`[projects.*]`, the gateway provider) is preserved.

> Always pass `--context-window`. Without it Codex warns *"Model metadata for `<path>`
> not found. Defaulting to fallback metadata"* and uses a conservative context size for
> your custom model id. The launcher prints `CONTEXT_WINDOW=` in its summary.

Codex 0.132 constraints the patcher already honors — do not fight them:
- `wire_api` must be `"responses"` (`"chat"` was removed).
- Provider id `openai` is reserved; the local provider is named `sglang`.

### 4. Verify Codex end-to-end (do NOT skip)
Run a real agentic task that forces a tool call:
```bash
cd /tmp && rm -rf cbtest && mkdir cbtest && cd cbtest && printf 'secret_number=42\n' > note.txt
codex exec --skip-git-repo-check "Read note.txt with your tools and tell me secret_number."
```
PASS = Codex actually runs a command (e.g. `cat note.txt`) and answers `42`. If it instead
prints `<tool_call>`/`<function=...>` as text, the server parser is wrong → back to step 1.
Clean up `/tmp/cbtest` afterward.

## Switching backends later
- Default is now the local server: just run `codex`.
- A `[profiles.gateway]` (AMD gateway) is preserved: `codex --profile gateway`.
- To re-point at a different local model, re-run this skill with that script.

## Notes
- The `bubblewrap not found` warning from `codex exec` is harmless (uses bundled bwrap).
- SGLang accepts requests without an API key by default, so no `env_key` is set on the
  provider. If a server is launched with `--api-key`, add `env_key`/export the key.
- Backups accumulate as `~/.codex/config.toml.bak.*` — fine to prune.
