---
name: validate-before-commit
description: Always validate changes (accuracy + profiling + benchmark) before committing — never commit untested code
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5717c721-0ef3-493c-8ed9-e9b0f9e91dfb
---

Run validation before committing kernel changes. Do not commit until the change is confirmed working via the validation pipeline (accuracy test, profiling, benchmark).

**Why:** The user wants to minimize wasted commits and ensure every committed change is verified. Committing untested code creates noise and may need reverts.

**How to apply:** In the `/implement-kernel` workflow, reorder steps: implement → validate → commit (not implement → commit → validate). Also minimize human intervention during the dev/validation cycle — check server/client scripts upfront and run the full pipeline without asking unnecessary questions.
