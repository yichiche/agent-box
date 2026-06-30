---
name: no-edit-running-bash-script
description: "Don't edit a bash script file while a process is still executing it"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8d336da0-5c21-4099-ae78-5bfcfc312a1d
---

Never edit a `.sh` file (via Edit/Write) while a process is still running it — e.g. a
long benchmark driver launched with `docker exec ... bash script.sh`. Bash reads scripts
by **byte offset** from the file on disk; mutating the file mid-run can misalign the
read position and make the running process execute garbage or skip its tail (summarize,
cleanup, etc.).

**Why:** happened during the `perf-sweep` skill build — I edited `perf_sweep.sh` (bench
arg fixes, summarizer keys, profiling pass) while the container was mid-sweep. It
survived by luck this time, but the tail could have broken.

**How to apply:** if a fix is needed mid-run, either (a) copy the script to a new path
and edit the copy for the *next* run, or (b) wait for the current run to finish. The
data the run writes to disk (per-conc JSON here) is safe regardless, so prefer
summarizing/post-processing from those artifacts rather than racing the live script.
Related: [[perf-sweep-skill]].
