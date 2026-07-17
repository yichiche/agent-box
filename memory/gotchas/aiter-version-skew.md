# aiter version skew — PR 4017 opus/6401 kernel missing → prefill hard-abort

A tuned-GEMM CSV from a newer aiter can reference a kernel id that doesn't exist in
an older pinned checkout, causing a **hard server abort during prefill** (not a
graceful fallback).

## Symptom

aiter PR **4017** (`qwen3_5_397b_bf16_tuned_gemm.csv`) has two rows using
`libtype=opus, solidx=6401`. That kernel id is absent from an older checkout's
gfx950 lookup table, so the server aborts:

```
[AITER] ...opus_gemm_arch_gfx950.cuh:139 Kernel id 6401 not found in a16w16 bf16 tune lookup table
Fatal Python error: Aborted   # scheduler exit -6
```

## Why it hides from smoke tests

The crash only triggers once prefill hits **M≈2048** (e.g. conc≥8 at IL1024, or
**any IL8192**). A **conc-4-only smoke test passes and hides it** — then the sweep
crashes mid-run at higher concurrency. If a server dies partway through a sweep but a
quick conc-4 check was fine, suspect a GEMM-CSV/kernel skew at large M.

## Fix

```bash
grep -v "opus,6401," <csv> > t && mv t <csv>   # 128 → 126 lines
rm -rf /tmp/aiter_configs                        # stale merged runtime cache
```

The dropped shapes are large-M prefill dense GEMM → fall back to default solution;
they do NOT touch decode or moe_sorting. Rebuild after editing a csrc header, then
clear JIT locks ([[aiter-jit-baton-vram]]) and relaunch.

**General rule:** a tuned CSV / config is version-locked to the aiter it was tuned
against. When landing a PR's config onto a pinned checkout, verify referenced kernel
ids exist for this gfx arch before a long run.

Source: [[../journal/2026-06/-sgl-workspace-aiter__aiter-4017-opus6401-version-skew]].
Related: [[aiter-jit-baton-vram]], [[bench-cwd-shadow]].
