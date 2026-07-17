# Data-Source Registry

> Single source of truth for **where kernel-optimization reference data comes from**.
> The `research` category of skills (see [`../INDEX.md`](../INDEX.md)) reads from these
> sources. This is the **MCP growth surface**: when a new MCP server is wired in, add a
> row here, then reference its `key` in the consuming skill's `data_sources:` frontmatter.

## How this works

1. Each SKILL.md in the `research` category declares which sources it pulls from:
   ```yaml
   category: research
   data_sources: [jira-amd, aiter-upstream]
   ```
2. The `key` in that list must match a row in the table below.
3. An agent doing kernel optimization scans `research` skills → sees their `data_sources`
   → looks up here **how** to reach each source (MCP tool / web / CLI / script) + auth.

## Registry

| key              | what it is                                   | access method (target: MCP)              | auth            | consumed_by                          | status  |
|------------------|----------------------------------------------|------------------------------------------|-----------------|--------------------------------------|---------|
| `trace-xlsx`     | Local trace-analyzer Excel output            | filesystem (`profile/trace_module_analyzer.py`) | —        | compare-kernels, perf-summary        | live    |
| `aiter-upstream` | ROCm/aiter kernels & PRs                     | github-mcp *(planned)* → REST search / `gh` / local `git`+grep | GITHUB_TOKEN (opt) | compare-kernels (`--refs`)   | live    |
| `sglang-upstream`| sgl-project/sglang kernels & PRs             | github-mcp *(planned)* → REST search / `gh` / local `git`+grep | GITHUB_TOKEN (opt) | compare-kernels (`--refs`)   | live    |
| `atom-upstream`  | ATOM engine dashboard + repo                 | web (`rocm.github.io/ATOM`) / repo       | —               | atom-progress                        | live    |
| `jira-amd`       | amd.atlassian.net tickets                    | atlassian-mcp *(planned)* → REST + token | JIRA_EMAIL+JIRA_API_TOKEN | qwen35-jira-track, compare-kernels (`--refs`) | live    |
| `inferencex`     | InferenceX benchmark runs                    | GitHub Actions API (`gh`)                | gh token        | inferencex-table                     | live    |
| `hackmd`         | HackMD PR-draft docs                         | web / HackMD API *(planned)*             | hackmd token    | pr, commit-push-pr                   | partial |

## Adding a new MCP source (checklist)

1. Register the MCP server (`.mcp.json` / Claude Code MCP config).
2. Add a row above: `key`, what it is, the MCP tool name in the access column, auth, `status: live`.
3. In every consuming SKILL.md, add the `key` to `data_sources:` frontmatter.
4. In the skill body, document the exact MCP tool call + fallback (CLI/web) if the MCP is unavailable.
5. Cross-link from [`../INDEX.md`](../INDEX.md) research section.

## Roadmap (kernel-optimization reference-finding)

The flagship target is **`/compare-kernels`** evolving from "diff two local xlsx" into a
reference finder: after the trace diff, consult `aiter-upstream` + `sglang-upstream` for
the slow kernel's upstream impl, and `jira-amd` for in-flight tickets touching it. Each
hop resolves through this registry, so wiring an MCP once benefits every research skill.
