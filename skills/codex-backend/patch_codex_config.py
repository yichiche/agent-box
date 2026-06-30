#!/usr/bin/env python3
"""Point Codex's default backend at a local SGLang server.

Edits ~/.codex/config.toml in place (with a timestamped backup), changing ONLY:
  - top-level   model = "<MODEL>"
  - top-level   model_provider = "sglang"
  - [model_providers.sglang]  base_url / wire_api / name

Everything else (profiles, [projects.*], [tui.*], the gateway provider) is preserved
verbatim. No TOML library required (none is installed on this box).

Codex 0.132 gotchas baked in here:
  - wire_api MUST be "responses" ("chat" was removed).
  - The provider id "openai" is reserved and cannot be overridden -> we use "sglang".

Usage:
  python3 patch_codex_config.py --base-url http://127.0.0.1:8000/v1 \
      --model /data/amd/Qwen3.5-... [--config ~/.codex/config.toml]
"""
import argparse
import os
import shutil
import sys
import time

PROVIDER = "sglang"


def patch(lines, base_url, model, context_window=None):
    out = []
    section = None  # current [section] header, None == top-level preamble
    seen_top_model = False
    seen_top_provider = False
    seen_top_ctx = False
    in_sglang = False
    sglang_keys_done = {"base_url": False, "wire_api": False, "name": False}

    def flush_sglang_missing(buf):
        if not sglang_keys_done["name"]:
            buf.append(f'name = "Local SGLang"\n')
        if not sglang_keys_done["base_url"]:
            buf.append(f'base_url = "{base_url}"\n')
        if not sglang_keys_done["wire_api"]:
            buf.append('wire_api = "responses"\n')

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            # Leaving the sglang block? Append any missing keys before the new header.
            if in_sglang:
                flush_sglang_missing(out)
            section = stripped[1:-1].strip()
            in_sglang = section == f"model_providers.{PROVIDER}"
            out.append(raw)
            continue

        if section is None:  # top-level preamble
            key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
            if key == "model":
                out.append(f'model = "{model}"\n')
                seen_top_model = True
                continue
            if key == "model_provider":
                out.append(f'model_provider = "{PROVIDER}"\n')
                seen_top_provider = True
                continue
            if key == "model_context_window" and context_window:
                out.append(f"model_context_window = {context_window}\n")
                seen_top_ctx = True
                continue
            out.append(raw)
            continue

        if in_sglang:
            key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
            if key == "base_url":
                out.append(f'base_url = "{base_url}"\n')
                sglang_keys_done["base_url"] = True
                continue
            if key == "wire_api":
                out.append('wire_api = "responses"\n')
                sglang_keys_done["wire_api"] = True
                continue
            if key == "name":
                sglang_keys_done["name"] = True
                out.append(raw)
                continue
            out.append(raw)
            continue

        out.append(raw)

    # File ended while still inside the sglang block.
    if in_sglang:
        flush_sglang_missing(out)

    # Ensure top-level model / model_provider exist (insert at very top).
    prefix = []
    if not seen_top_model:
        prefix.append(f'model = "{model}"\n')
    if not seen_top_provider:
        prefix.append(f'model_provider = "{PROVIDER}"\n')
    if context_window and not seen_top_ctx:
        prefix.append(f"model_context_window = {context_window}\n")
    out = prefix + out

    # Ensure the [model_providers.sglang] block exists at all.
    text = "".join(out)
    if f"[model_providers.{PROVIDER}]" not in text:
        block = (
            f"\n# Local SGLang OpenAI-compatible server (added by /codex-backend)\n"
            f"[model_providers.{PROVIDER}]\n"
            f'name = "Local SGLang"\n'
            f'base_url = "{base_url}"\n'
            f'wire_api = "responses"\n'
        )
        out.append(block)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-window", type=int, default=None,
                    help="model_context_window (server max_model_len); silences metadata warning")
    ap.add_argument("--config", default=os.path.expanduser("~/.codex/config.toml"))
    args = ap.parse_args()

    cfg = os.path.expanduser(args.config)
    if not os.path.isfile(cfg):
        # Fresh minimal config.
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        with open(cfg, "w") as f:
            f.write("")
        lines = []
    else:
        bak = f"{cfg}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(cfg, bak)
        print(f"backup: {bak}")
        with open(cfg) as f:
            lines = f.readlines()

    new = patch(lines, args.base_url, args.model, args.context_window)
    with open(cfg, "w") as f:
        f.writelines(new)
    print(f"patched: {cfg}")
    print(f"  model          = {args.model}")
    print(f"  model_provider = {PROVIDER}")
    print(f"  base_url       = {args.base_url}")
    print("  wire_api       = responses")
    if args.context_window:
        print(f"  model_context_window = {args.context_window}")


if __name__ == "__main__":
    sys.exit(main())
