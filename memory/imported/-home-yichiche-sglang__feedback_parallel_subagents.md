---
name: parallel-subagents
description: Always use parallel subagents for independent investigations — never sequential one-by-one
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7f1f7c9f-9a4f-4054-bf24-ebb976e7c25f
---

When investigating multiple independent questions (kernel sources, fusion candidates, code paths), launch parallel subagents instead of doing them sequentially one-by-one.

**Why:** User wants to see multiple threads working simultaneously in the main window. Sequential investigation is too slow and wastes time.

**How to apply:** Whenever you have 2+ independent lookups/investigations, batch them into parallel Agent calls in a single message.
