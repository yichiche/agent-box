---
name: feedback-gh-auth
description: Use OAuth token (not GH_TOKEN PAT) for gh CLI commands that hit sgl-project/sglang
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fd66b1ec-5cde-45f5-a32e-f536d70a42ee
---

Always prefix `gh` commands with `GH_TOKEN=""` to bypass the fine-grained PAT (which is blocked by LMSYS enterprise token lifetime policy) and use the OAuth token from `gh auth login` instead.

**Why:** The `GH_TOKEN` env var contains a fine-grained PAT with lifetime >366 days, which the LMSYS Corp enterprise policy rejects for API access to `sgl-project/sglang`. The OAuth token stored in `~/.config/gh/hosts.yml` (from `gh auth login`) has `repo` scope and works fine.

**How to apply:** In all skills that run `gh pr create`, `gh pr view`, `gh api`, or any `gh` command targeting `sgl-project/sglang`, use `GH_TOKEN="" gh ...` instead of plain `gh ...`.
