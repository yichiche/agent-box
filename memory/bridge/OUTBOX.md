# OUTBOX — Container → Host

Reports from Claude Code **inside the container**. Host Cursor / Claude Code: read for latest findings.

---

## [2026-07-20 08:26 yct-aiter-test-0720] @host (via bridge exec, agent=claude)
You are in the aiter git repo at /home/yichiche/aiter. List aiter updates from TODAY (2026-07-20). First run: git -C /home/yichiche/aiter rev-parse --abbrev-ref HEAD ; then git -C /home/yichiche/aiter log --since="2026-07-20 00:00" --until="2026-07-21 00:00" --pretty="%h | %an | %ad | %s" --date=format:"%Y-%m-%d %H:%M" ; also run git -C /home/yichiche/aiter fetch -q origin 2>/dev/null && git -C /home/yichiche/aiter log --since="2026-07-20 00:00" --pretty="%h | %an | %s" origin/main. Then report: current branch, and a concise bullet list of todays commits (hash, author, subject). If there are zero commits today on both, say clearly "No commits today" and instead show the 5 most recent commits with their dates.

```
⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth source is set and takes precedence over your claude.ai login · Unset it to load your organization's connectors
## aiter updates — 2026-07-20

**Current branch:** `main`

**Local `main`:** No commits today (your checkout is behind `origin/main`).

**`origin/main` — today's commits (8):**

- `3a17355f6` | Copilot — Fixing tag parsing syntax issue (#4296)
- `2f476b872` | carlushuang — perf(unified_attention): use 8 warps for gfx1151 3D decode (#3915)
- `b34e04b36` | carlushuang — feat(gfx1151): allow gfx1151 in cpp_itfs JIT arch validation (#3919)
- `11f7f79c0` | carlushuang — feat(gfx1151): INT8 W8A8 GEMM config + int8 fused-MoE forward + RDNA a8w8 routing (#3917)
- `a973f0db5` | JiaoliangYu — CI(atom-di): track ATOM main + pin cases to 1p1d via --case (#4257)
- `d6de77692` | JiaLuo-CAN — Repining CK to b6759456 and fixes for CI tests (#4224)
- `967a03ac1` | la — fix fused_qk_norm_rope_cache_quant build without ck (#4220)
- `874840aef` | shay-li77 — Add OPUS gfx950 bf16 fmha d192x128 kernel (#4205)

**Note:** These commits exist on `origin/main` but not on your local `main` (the `--since` query on local returned nothing). Your local branch hasn't pulled today's work yet — run `git -C /home/yichiche/aiter pull` to sync.

Themes today: gfx1151 (RDNA) enablement (INT8 W8A8 GEMM, fused-MoE, JIT arch validation, decode warp tuning), an OPUS gfx950 bf16 FMHA kernel, plus CI/CK repinning and build fixes.
```
- rc: 0
- status: done

## [2026-07-06 03:05 jacky-v0.5.13-rocm720-mi35x-qwen3.5-mxfp4-0626-PR26858] @host (via bridge exec, agent=claude)
Reply in one line only: run 'hostname' and 'git -C /sgl-workspace/sglang branch --show-current'. No preamble.

```
`smci355-ccs-aus-m15-21`, branch `main`
```
- rc: 0
- status: done

<!-- Add new messages at the top -->

## Template

```markdown
## [YYYY-MM-DD HH:MM container] @host
<summary>

- findings: ...
- next: ...
- status: done | blocked | needs_host
```
