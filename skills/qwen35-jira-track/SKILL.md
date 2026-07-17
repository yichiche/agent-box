---
name: qwen35-jira-track
description: >-
  Fetch ALL amd.atlassian.net Jira tickets for MoE (Jin/Lingpeng, Wen/Jiaxing,
  Li/Felix), Quark (Lin/Zhao, Ren/Jiangyong), and a16w4 (Picciau/Andrea), read
  each ticket, and analyze which are helpful for Qwen3.5-397B-A17B-MXFP4
  optimization (kernels, MoE, MHA, quark, flydsl). Use for /qwen35-jira,
  ticket triage, or finding transferable aiter/MoE work for GPUAI-6500.
category: research
data_sources: [jira-amd]
---

# qwen35-jira-track

**Goal:** Not just list tickets — **fetch every ticket** from the watchlist engineers, **read summaries + descriptions**, and **analyze which ones can help** [GPUAI-6500](https://amd.atlassian.net/browse/GPUAI-6500) (Qwen3.5-397B-A17B-mxfp4 on MI355x).

Config: [watchlist.yaml](watchlist.yaml) · Script: [scripts/track_jira.py](scripts/track_jira.py)

## When to use

- Track **and analyze** Jira work from Lingpeng, Jiaxing, Felix, Zhao, Jiangyong, Andrea
- Find **transferable** mxfp4/aiter/MoE kernel tickets (even if filed for DSV4/Kimi/GLM)
- Triage: what to follow up, cherry-pick, or benchmark for Qwen3.5 mxfp4
- `/qwen35-jira`, `qwen35 jira analyze`, `哪些 ticket 对 mxfp4 有帮助`

## Watchlist

| Domain | People | What to look for |
|--------|--------|------------------|
| **moe** | Jin, Lingpeng · Wen, Jiaxing · Li, Felix | MoE GEMM, dispatch/combine, FSE, aiter fused MoE |
| **quark** | Lin, Zhao · Ren, Jiangyong | MXFP4 checkpoint, rotation, file2file, AttnFP8 |
| **a16w4** | Picciau, Andrea | A16W4 / attention weight quant paths |

---

## Mandatory workflow (track + analyze)

Do **all** steps. Do not stop at a keyword-filtered subset.

### Step 1 — Full fetch (all tickets by person)

**Recency rule:** Only tickets **`updated >= -90d`**. No exception for old Open tickets — if it has not been touched in 3 months, skip it.

**Hard exclude:** Keys in `excluded_tickets` ([watchlist.yaml](watchlist.yaml)) are never reported (e.g. stale SWDEV-537308, SWDEV-516036).

Default: **`--full`** = no model filter, **90 days**, up to 100 issues per query.

```bash
python3 ~/.claude/skills/qwen35-jira-track/scripts/track_jira.py --full --jql-only
```

Run **three parallel MCP searches** (one per domain) plus anchors:

```
cloudId: 3ade9f4f-3a5e-4909-bc67-8816482a10f4
searchResultMode: issues
maxResults: 100
fields: summary,status,assignee,reporter,updated,priority,description,issuetype,key
responseContentFormat: markdown
```

**JQL templates (`--full`, 90d recency):**

MoE team — all tickets:
```jql
(
  assignee = "Jin, Lingpeng" OR reporter = "Jin, Lingpeng" OR
  assignee = "Wen, Jiaxing" OR reporter = "Wen, Jiaxing" OR
  assignee = "Li, Felix" OR reporter = "Li, Felix"
) AND updated >= -90d ORDER BY updated DESC
```

Quark team — all tickets:
```jql
(
  assignee = "Lin, Zhao" OR reporter = "Lin, Zhao" OR
  assignee = "Ren, Jiangyong" OR reporter = "Ren, Jiangyong"
) AND updated >= -90d ORDER BY updated DESC
```

A16W4 — all tickets:
```jql
(assignee = "Picciau, Andrea" OR reporter = "Picciau, Andrea")
AND updated >= -90d ORDER BY updated DESC
```

Anchors (merge if updated within window):
```jql
key in (GPUAI-6500, GPUAI-6382, QUARK-744, QUARK-774, QUARK-743, QUARK-716, QUARK-760, QUARK-677, QUARK-763)
AND updated >= -90d ORDER BY updated DESC
```

If a person query returns 0 rows, verify display name in Jira UI; update [watchlist.yaml](watchlist.yaml).

Paginate with `nextPageToken` if `isLast: false`.

### Step 2 — Read every ticket

For **each** unique issue key:

1. Read **summary + full description** (not summary alone)
2. Note: status, assignee, **last updated** (`updated` field, report as YYYY-MM-DD), perf numbers (e.g. 10–17% FSE), repro commands, linked PRs
3. Map to **optimization area** (see watchlist.yaml `optimization_areas`)

Optional: fetch comments for top-scored tickets only (`fields` includes `comment`).

### Step 3 — Analyze helpfulness

Use verdicts from [watchlist.yaml](watchlist.yaml):

| Verdict | Meaning | Typical action |
|---------|---------|----------------|
| **direct** | Qwen3.5-397B mxfp4 / GPUAI-6500 / QUARK-744 | Prioritize; sync with owner |
| **transferable** | Other mxfp4 MoE model, same aiter/SGLang path on MI355 | Port test; A/B on Qwen3.5 |
| **enabler** | Quark checkpoint / accuracy fix unblocks perf | Track HF revision / Quark export |
| **watch** | Related but outcome unclear | Re-check when Done |
| **low** | Unrelated stack or no MI355 angle | Skip |

**Optimization areas** to tag each ticket:

- `moe_gemm` · `moe_dispatch` · `shared_expert_fusion` · `attention` · `quantization_checkpoint` · `comm_fusion` · `flydsl_kernels`

Heuristic pre-score (script):

```bash
python3 ~/.claude/skills/qwen35-jira-track/scripts/track_jira.py --full --fetch --analyze
# needs JIRA_EMAIL + JIRA_API_TOKEN
```

**Agent must override script scores** after reading descriptions — e.g. DSV4 dispatch work is transferable even if script says `medium`.

### Step 4 — Cross-check with your workload

Relate findings to:

- [GPUAI-6500](https://amd.atlassian.net/browse/GPUAI-6500) profiling bottlenecks (kernel breakdown, prefill vs decode)
- Local scripts: `run_qwen3.5_mxfp4_perf.sh`, aiter/flydsl env flags
- Known gaps: shared-expert fusion, online MHA quant, file2file double-rounding (QUARK-763)

---

## Report template (required output)

```markdown
# Qwen3.5-397B-A17B-MXFP4 — Jira analysis (DATE)

## Executive summary
- Scanned N tickets from 6 engineers (**90d updated only**; `excluded_tickets` skipped)
- **M likely helpful** (direct + transferable + enabler)
- Top 3 actions: ...

## Recommended for GPUAI-6500 (actionable)

### Direct
| Ticket | Last updated | Owner | Area | Why it helps | Suggested next step |
| KEY | YYYY-MM-DD | Name | moe_gemm | ... | ... |

### Transferable (port / benchmark on Qwen3.5)
| Ticket | Last updated | Source model | Area | Why it helps | Risk |
| ... |

### Enablers (checkpoint / accuracy)
| Ticket | Last updated | Blocker removed | Status |
| ... |

## By engineer
### Wen, Jiaxing (moe)
- **Helpful:** ...
- **Noise / skip:** ...

(repeat per person)

## Watch list (open, re-check later)
| Ticket | Last updated | Why watching |
| ... |

## Full inventory (reference)
<details>
<summary>All N tickets scored</summary>
compact table: Key | Verdict | Score | Last updated | Status | Summary
</details>
```

Link: `[KEY](https://amd.atlassian.net/browse/KEY)`

---

## Analysis heuristics (agent judgment)

Mark **helpful** when ticket mentions any of:

- MXFP4 MoE GEMM, blockscale, expert fusion, dispatch/combine on MI355/gfx95
- Shared expert / FSE / `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` (+10–17% cited)
- Aiter unified attn, MHA assembly, flydsl, `--attention-backend aiter`
- Quark mxfp4 export, rotation (QUARK-677), AttnFP8 + MoE combined checkpoint
- A16W4 attention quant path applicable to Qwen3.5 linear/self attn
- Perf repro using SGLang + aiter flags (see `serve_signals` in watchlist)

Mark **low** when:

- Different hardware (MI300 only, navi, gfx1250) with no gfx95 port mentioned
- Closed vLLM-only path with no SGLang/aiter overlap
- Pure infra / storage / unrelated model family with no mxfp4 keyword
- **Stale / abandoned:** old Open ticket with no Qwen3.5/mxfp4 progress (e.g. SWDEV-537308 Llama FP4 OOM, SWDEV-516036 MI308 blockscale MoE)
- Listed in `excluded_tickets` in watchlist.yaml

**Transferable examples:** DSV4-Flash EP4 MoE GEMM tuning, Kimi/GLM mxfp4, generic aiter `_combined_routing` fixes — same kernel names often appear in Qwen3.5 traces.

---

## Script reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--full` | off | **Use for analysis** — all person tickets, 90d recency, no model filter |
| `--model-only` | off | Narrow JQL to mxfp4/397B keywords only |
| `--analyze` | off | Add helpfulness score + verdict + areas |
| `--fetch` | off | REST API (JIRA_EMAIL + JIRA_API_TOKEN) |
| `--anchors` | off | Anchor tickets JQL only |
| `--domain` / `--person` | all | Scope fetch |
| `--days N` | 90 (`recency_days` in watchlist.yaml) | Override time window |
| `--max N` | 100 | Issue cap per query |
| `--json` | off | Machine-readable |

```bash
# Full analysis pipeline (REST)
export JIRA_EMAIL=... JIRA_API_TOKEN=...
python3 ~/.claude/skills/qwen35-jira-track/scripts/track_jira.py --full --fetch --analyze

# Per domain
python3 ~/.claude/skills/qwen35-jira-track/scripts/track_jira.py --domain moe --full --fetch --analyze --json
```

In Cursor without REST creds: MCP fetch + agent applies Step 3–4 manually.

---

## Examples

**User:** 分析 MoE 團隊 ticket 哪些對 Qwen3.5 mxfp4 有幫助

→ `--domain moe --full` × MCP × read all descriptions → report **Recommended** + per-engineer sections.

**User:** `/qwen35-jira`

→ Full 3-domain sweep + anchors → analyze → executive summary + top 5 actions for GPUAI-6500.

**User:** Jiaxing 最近做了什麼可能能 reuse？

→ `--person "Wen, Jiaxing" --full` → filter verdict ∈ {direct, transferable, enabler} → explain port path to SGLang Qwen3.5.

---

## Maintenance

Edit [watchlist.yaml](watchlist.yaml): people, anchor tickets, `optimization_areas`, `serve_signals`, `excluded_tickets`.

Do not hardcode sprint dates.
