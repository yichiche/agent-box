---
name: sglang-branch-hygiene
description: "Never commit on main in SGLang — always use a feature branch or worktree, return to main after PR"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 33bce402-0dfd-437e-bd91-f810faf40d9d
---

Never commit directly on `main` in the SGLang repo. Always work on a feature branch (or git worktree).

**Full workflow:**
1. **Before starting work:** verify HEAD is on `main` and clean. If not, stash or abort.
2. **Create a feature branch** (or worktree) from `main` before any modifications.
3. **Commit only on the feature branch**, never on `main`.
4. **After completing the PR:** switch back to `main` (`git checkout main`) so there's no pollution between modifications.

**Why:** The user got burned by commits landing on `main` — it pollutes the local main branch, makes future rebases messy, and violates the repo-config rule that SGLang requires feature branches. Each task should be isolated.

**How to apply:** At the start of every SGLang modification task, run `git branch --show-current` and refuse to proceed if on `main`. Create a branch first. After `/commit-push-pr` or `/pr` completes, switch back to `main`.
