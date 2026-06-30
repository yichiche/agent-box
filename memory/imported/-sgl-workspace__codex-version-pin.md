---
name: codex-version-pin
description: OpenAI Codex CLI must be pinned to 0.132.0 — newer versions break with AMD Gateway
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5fb47745-679b-408b-a79a-c7bb4bf8a4f0
---

Install @openai/codex at version 0.132.0 exactly: `npm install -g @openai/codex@0.132.0`

**Why:** Versions newer than 0.132.0 are incompatible with the AMD Gateway and fail to work.

**How to apply:** When installing or reinstalling Codex CLI on any machine, always pin to 0.132.0. Do not install latest.
