---
name: feedback-server-polling
description: "Always run server polling loops in foreground with timeout, never via background+TaskOutput"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 69b4ac4f-af35-488d-8cac-abc1245fad89
---

Always run server readiness polling loops (`curl` health check in a `for` loop) in the **foreground** with a proper `timeout` parameter on the Bash tool (e.g., `timeout: 600000`). Never launch the poll as a background task and then block on `TaskOutput` — if `TaskOutput` gets interrupted, the agent loses track and gets stuck in a reactive loop.

**Why:** During validation, the agent launched a polling loop in the background, then called `TaskOutput` to wait. When the user interrupted, the agent couldn't recover and kept answering status questions instead of relaunching the server.

**How to apply:** In `/validate` and any skill that starts a server, always use foreground polling. The server launch itself can be `run_in_background: true`, but the health-check loop must be foreground.
