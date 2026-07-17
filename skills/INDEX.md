# Skills Index

> Category routing for all slash-command skills. Skill dirs stay **flat**
> (`skills/<name>/SKILL.md`) because Claude Code only registers slash commands from that
> one level — categorization is metadata, not folders. Each SKILL.md carries a
> `category:` (and, for `research`, a `data_sources:`) frontmatter field; this file is the
> grouped view. External data sources are registered in
> [`_shared/data-sources.md`](_shared/data-sources.md).

## Categories

| category      | purpose                                            |
|---------------|----------------------------------------------------|
| **research**  | **Fetch external reference data** (MCP growth surface) |
| **kernel-opt**| Implement / optimize a kernel                      |
| **measure**   | Benchmark, profile, quantify                       |
| **deliver**   | Git / PR delivery                                  |
| **infra**     | GPU / container / environment plumbing             |
| **meta**      | Memory, self-improvement, reporting                |

---

## research — get reference data for kernel optimization ⭐

> When optimizing a kernel and you need prior art, upstream impls, or in-flight work,
> **these skills fetch it.** Each declares `data_sources:` → resolve via
> [`_shared/data-sources.md`](_shared/data-sources.md). New MCP integrations land here.

| skill | fetches | data_sources |
|-------|---------|--------------|
| [`/compare-kernels`](compare-kernels/SKILL.md) | Kernel category/timing diff between two traces + `--refs` upstream/Jira prior-art finder (Step 8) | `trace-xlsx`, `aiter-upstream`, `sglang-upstream`, `jira-amd` |
| [`/qwen35-jira-track`](qwen35-jira-track/SKILL.md) | Jira tickets from MoE/Quark/a16w4 watchlist, analyzed for transferability | `jira-amd` |
| [`/atom-progress`](atom-progress/SKILL.md) | ATOM engine benchmark trend + relevant-to-SGLang flags | `atom-upstream` |
| [`/inferencex-table`](inferencex-table/SKILL.md) | InferenceX benchmark numbers by model/hw/framework or run URL | `inferencex` |

## kernel-opt — do the optimization

| skill | does |
|-------|------|
| [`/implement-kernel`](implement-kernel/SKILL.md) | Design → implement → validate → commit a kernel change (MI355/ROCm) |
| `kernel-fusion-pipeline` | Orchestration scripts only — **no SKILL.md yet**, not a registered slash command |

## measure — benchmark & profile

| skill | does |
|-------|------|
| [`/validate`](validate/SKILL.md) | Baseline + accuracy + profile + after benchmark for a change |
| [`/benchmark`](benchmark/SKILL.md) | Before/after e2e benchmark of a change |
| [`/perf-sweep`](perf-sweep/SKILL.md) | Accuracy-gated concurrency sweep |
| [`/pr-ab-benchmark`](pr-ab-benchmark/SKILL.md) | Before/after A/B benchmark of an aiter/sglang PR |
| [`/generate-profile`](generate-profile/SKILL.md) | Capture Chrome-compatible trace |
| [`/parse-trace`](parse-trace/SKILL.md) | Run trace_module_analyzer (prefill/decode) |
| [`/perf-summary`](perf-summary/SKILL.md) | Summarize kernel perf into status + action items |

## deliver — git & PR

| skill | does |
|-------|------|
| [`/commit`](commit/SKILL.md) | Stage, branch, commit per repo conventions |
| [`/commit-push`](commit-push/SKILL.md) | Commit + push to fork |
| [`/commit-push-pr`](commit-push-pr/SKILL.md) | Commit + push + PR in one flow |
| [`/pr`](pr/SKILL.md) | Create a GitHub PR (HackMD draft first) |

## infra — GPU / container / environment

| skill | does |
|-------|------|
| [`/gpu-status`](gpu-status/SKILL.md) | Free AMD GPUs + correct CUDA_VISIBLE_DEVICES |
| [`/gpu-to-containers`](gpu-to-containers/SKILL.md) | Map GPUs → owning containers |
| [`/codex-backend`](codex-backend/SKILL.md) | Point Codex CLI at a local SGLang server |
| [`/remote-bridge`](remote-bridge/SKILL.md) | Host ↔ container agent bridge |

## meta — memory, self-improvement, reporting

| skill | does |
|-------|------|
| [`/memory-capture`](memory-capture/SKILL.md) | Save gotcha/config/workflow into memory vault |
| [`/memory-consolidate`](memory-consolidate/SKILL.md) | Distill vault → AGENTS.md / CLAUDE.md |
| [`/skill-suggest`](skill-suggest/SKILL.md) | Draft skill/workflow stubs from journal themes |
| [`/standup`](standup/SKILL.md) | Generate / formalize daily standup |
| [`/prompt-to-goal-command`](prompt-to-goal-command/SKILL.md) | Rewrite prompts into goal-style commands |

---

## Conventions

- **Add a skill:** create `skills/<name>/SKILL.md` with `category:` frontmatter, then add a
  row to the matching section here.
- **`category`** is a single controlled value from the table above.
- **`research` skills** additionally declare `data_sources: [key, ...]` matching
  [`_shared/data-sources.md`](_shared/data-sources.md).
- Shared assets (templates, registry, repo config) live in [`_shared/`](_shared/).
