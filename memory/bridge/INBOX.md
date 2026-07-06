# INBOX — Host → Container

Messages for Claude Code **inside the container**. Container agent: read this at session start; mark `pending` → `done` when handled.

---

<!-- Add new messages at the top -->

## Template

```markdown
## [YYYY-MM-DD HH:MM host] @container
<question or task>

- context: ...
- paths: ...
- status: pending | in_progress | done
```
