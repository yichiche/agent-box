---
type: feedback
---

# SGLang branch hygiene

Never commit on `main` in SGLang. Always feature branch or worktree; return to `main` after PR.

At task start: `git branch --show-current` — refuse if on `main`.

See `skills/_shared/repo-config.md` and `/commit-push-pr`.
