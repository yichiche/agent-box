---
type: gotcha
---

# Do not edit a running bash script

If a process is executing a `.sh` file, **do not use Edit** on that file. The shell has already loaded bytes at launch; mid-run edits cause byte-offset misalignment → subtle wrong behavior.

**Fix:** stop the process, edit, relaunch. Or write a new script variant.
