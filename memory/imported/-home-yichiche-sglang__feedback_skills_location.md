---
name: feedback-skills-location
description: All Claude skills must live in /home/yichiche/agent-box/skills and sync to ~/.claude — never in project-local .claude/skills/
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e7f4467-13a1-4563-97f7-2bc847d97ec4
---

All skills should be placed in `/home/yichiche/agent-box/skills`, not in project-local `.claude/skills/` directories.

**Why:** The user maintains a central skills repository at `/home/yichiche/agent-box/skills` that syncs with `~/.claude`. Putting skills in project-local paths breaks the centralized management workflow.

**How to apply:** When creating, modifying, or referencing skill files, always use `/home/yichiche/agent-box/skills/<skill-name>/` as the canonical location. Never create skills under `<project>/.claude/skills/`.
