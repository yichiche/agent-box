---
name: gpu-to-containers
description: >-
  Maps GPUs to Docker/Podman/containerd workloads by joining driver PIDs to
  cgroups. Use when the user asks which container uses which GPU, who is on
  GPU 0/1, GPU occupancy vs containers, or after rocm-smi/nvidia-smi shows busy
  cards and they need the owning workload name.
---

# GPU → container mapping

Answer **which container (or host process)** is using **which GPU** by: (1) list
PIDs per GPU from the driver, (2) resolve each PID to a container id/name from
`/proc/<pid>/cgroup` (and optional `docker inspect` / `crictl`).

**Privilege:** Reading other users' `/proc/<pid>/cgroup` may require root or the
same user as the workload. If resolution fails, say so and suggest `sudo` or
running from the host as root.

## AMD ROCm (MI300 / MI355 / generic KFD)

1. **PIDs per GPU** (KFD compute clients):

   ```bash
   rocm-smi --showpids
   # more detail when supported:
   rocm-smi --showpids details
   ```

   To query **which GPUs a PID uses**:

   ```bash
   rocm-smi --showpidgpus <PID>
   ```

   `rocm-smi` **device index** here is the same ordering as concise info
   (`rocm-smi` without args). It is **not** the same as PyTorch `CUDA_VISIBLE_DEVICES`
   index on permuted hosts — if the user needs the torch index, use the personal
   skill `gpu-status` (`~/.claude/skills/gpu-status/SKILL.md`) separately.

   **Important:** KFD’s `GPU(s)` column in `--showpids` is often a **process-local**
   logical index (e.g. after `CUDA_VISIBLE_DEVICES` remapping), **not** the same as
   physical `GPU[n]` / `rocm-smi` Device rows. Do **not** use it alone to answer
   “which **physical** GPU is which container”.

### Per **physical** GPU: VRAM / GPU% → main container (preferred on ROCm 7.x)

Use **`amd-smi process`** (JSON) — it returns **per GPU id 0…N-1** a `process_list`
with `pid` and `mem_usage` in bytes. For each GPU, take the PID with **maximum**
`mem_usage` on that GPU, then resolve the PID to a Docker name via cgroup (same
as above). Combine with **`rocm-smi`** concise output for **VRAM%** and **GPU%**
on the same host snapshot.

```bash
python3 ~/.cursor/skills/gpu-to-containers/scripts/gpu_vram_owner.py
```

Requires `amd-smi` on PATH (ROCm 7.x ships it) and `docker` for name resolution.

**PID → container (cgroup)** — for each PID from `amd-smi` or `--showpids`, run:

   ```bash
   PID=12345
   tr '\0' '\n' < "/proc/$PID/cgroup" 2>/dev/null || cat "/proc/$PID/cgroup"
   ```

   Interpretation:

   - **Docker Engine**: cgroup path often contains `docker-` + **64-char** container id
     (e.g. `.../docker-<id>.scope`). Match with:

     ```bash
     docker ps --no-trunc --format '{{.ID}}\t{{.Names}}'
     ```

   - **containerd / Kubernetes**: look for `cri-containerd-` + id, or `kubepods`
     slice; resolve pod/container with:

     ```bash
     sudo crictl ps -a
     # or kubectl get pods -A -o wide  # then correlate node-local PIDs with admin tooling
     ```

   - **Podman**: paths may contain `libpod-` and scope names with the container id.

3. **If `--showpids` is empty but VRAM is high** — the process may have exited
   while VRAM is not reclaimed yet, or the query needs root / a newer `rocm-smi`.
   Cross-check with `rocm-smi --showmeminfo vram` and retry after workloads settle.

## NVIDIA

1. **PIDs per GPU:**

   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_memory --format=csv
   nvidia-smi pmon -c 1
   ```

2. Same **PID → cgroup → `docker ps` / `crictl`** flow as above.

## One-shot host helper (bash)

Run on the **host** (not inside a container), adjust `sudo` if needed:

```bash
pid_container_hint() {
  local pid="$1"
  [[ -z "$pid" || ! -r "/proc/$pid/cgroup" ]] && { echo "$pid	(cgroup unreadable)"; return; }
  local line
  line=$(tr '\0' '\n' < "/proc/$pid/cgroup" 2>/dev/null | paste -sd' ')
  local id=""
  if [[ "$line" =~ docker-([a-f0-9]{64}) ]]; then id="${BASH_REMATCH[1]}"; fi
  if [[ -z "$id" && "$line" =~ cri-containerd-([a-f0-9]+) ]]; then id="${BASH_REMATCH[1]}"; fi
  if [[ -n "$id" ]]; then
    docker ps --no-trunc --filter "id=$id" --format '{{.Names}}\t{{.Image}}\t{{.ID}}' 2>/dev/null \
      || echo "id=$id (docker CLI unavailable or not Docker)"
  else
    echo "$pid	$line"
  fi
}

echo "=== rocm-smi PIDs ==="
rocm-smi --showpids 2>/dev/null || true

echo "=== nvidia-smi PIDs ==="
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv 2>/dev/null || true

echo "=== cgroup hints for ROCm PIDs (first column of KFD table rows) ==="
rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[\t ]/ { print $1 }' | sort -u | while read -r p; do
  echo "--- PID $p ---"
  pid_container_hint "$p"
done
```

The `awk` pattern skips the WARNING banner and header lines; it keeps lines whose
first field is a PID (digits then tab or space), matching typical `--showpids`
layouts.

## What to report to the user

- Table: **GPU id (driver)** → **PID(s)** → **container name or cgroup snippet**.
- Note if mapping is incomplete (permissions, rootless Docker, only partial id
  in cgroup, or Kubernetes without `crictl`).
- If they need **free GPUs + correct `CUDA_VISIBLE_DEVICES` for SGLang on ROCm**,
  that is the **gpu-status** skill, not this one.
