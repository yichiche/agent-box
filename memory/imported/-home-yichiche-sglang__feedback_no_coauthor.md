---
name: no-coauthor
description: Never add Co-Authored-By Claude lines to git commits — strictly prohibited for PRs
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5f551250-27c5-42f2-906b-346f29c0e395
---

Never add `Co-Authored-By: Claude` (or any Claude/Anthropic co-author line) to git commit messages.

**Why:** The user's organization strictly prohibits co-author attribution to AI tools in PRs. Adding it causes PR rejection and requires commit rewriting.

**How to apply:** When creating git commits (via `/commit`, `/commit-push`, `/commit-push-pr`, or manual `git commit`), omit the `Co-Authored-By` line entirely. Do not include any AI attribution in commit messages.
