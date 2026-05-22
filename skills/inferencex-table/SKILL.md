---
name: inferencex-table
description: "Fetch SGLang / InferenceX benchmark results from a GitHub Actions run URL and render a Concurrency / TP,DP / TTT / E2EL / TTFT / ITL markdown table. Use when the user pastes a https://github.com/SemiAnalysisAI/InferenceX/actions/runs/<id> link (or the bare run id) and asks to fill in or build a benchmark table, especially for DSV4 / sweep runs."
---

# inferencex-table

## When to use

Trigger this skill whenever the user:

- Pastes a GitHub Actions URL from `SemiAnalysisAI/InferenceX` (typically a `Run Sweep` workflow) and asks for a benchmark table, or
- Provides a bare run id and an existing table to fill in (TTT, E2EL, TTFT, ITL columns).

Do **not** use this for kernel-trace summaries (see `perf-summary`) or for accuracy/benchmark validation after code changes (see `validate`).

## What it produces

A markdown table per (ISL, OSL) combo with these columns:

| Concurrency | TP, DP | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) |

Field mapping from `agg_bmk.json`:

- `TTT (tok/s)` = `tput_per_gpu * tp`  (total throughput across all GPUs)
- `TP, DP` = `tp, tp` if `dp_attention=true`, else `tp, 1`
- `Median E2EL/TTFT/ITL (ms)` = `median_e2el/median_ttft/median_itl * 1000`

## Workflow

1. Extract the run id from the URL (regex: `actions/runs/(\d+)`). If only a number is given, use it as the run id.
2. Run the helper script — it handles auth (reads `~/.git-credentials` or `$GH_TOKEN`), downloads the `results_bmk` artifact via the GitHub API, and renders the table:

   ```bash
   python3 ~/.claude/skills/inferencex-table/scripts/fetch_bmk.py <url-or-id> \
       [--isl 8192] [--osl 1024] [--dpa auto|true|false]
   ```

3. If the user already provided a partial table (e.g. they filled TTT and want the latency columns), match rows by `Concurrency`. **Verify by TTT first**: `tput_per_gpu * tp` should equal the user's TTT — if not, you picked the wrong dpa variant (see step 4).
4. Resolving duplicates:
   - Some concurrencies have both `dp_attention=true` and `dp_attention=false` rows. When the user's table shows `TP, DP = 8, 1` pick `dpa=false`; when it shows `TP, DP = 8, 8` pick `dpa=true`.
   - If the user did NOT pre-fill TTT, **ask which dpa variant to use** (or output both rows side-by-side).
5. Filter `(isl, osl)`:
   - Infer from the user's context (e.g. paths like `il8k_ol1k` → `isl=8192, osl=1024`; `il1k_ol1k` → `1024, 1024`).
   - If unclear and the run has both, emit one table per combo and let the user pick.
6. Return the markdown table directly in the reply. Do **not** wrap it in a code fence unless the user asks — keep it renderable inline.

## Script reference

`scripts/fetch_bmk.py` — single-file Python 3 (stdlib only). Key flags:

| Flag | Default | Behavior |
|---|---|---|
| `--isl N` / `--osl N` | none | Filter rows; if both omitted, emits one table per `(isl, osl)` combo found |
| `--dpa auto\|true\|false` | `auto` | `auto` keeps both DPA variants (sorted false-then-true) |
| `--token PAT` | env / `~/.git-credentials` | Override GitHub auth |
| `--out FILE` | stdout | Write the rendered markdown to a file |
| `--keep DIR` | tmp | Persist the downloaded `agg_bmk.json` for further inspection |
| `--json` | off | Print path to the raw `agg_bmk.json` after the table |

The script downloads only the `results_bmk` artifact (~10 KB), so it is fast and cheap to re-run.

## Examples

**User pastes URL only:**

> https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26118426149
> 幫我做成 benchmark table

→ Run with no filter, ask which `(isl, osl)` they want if multiple exist, or default to the combo matching their currently-open files (e.g. `il8k_ol1k` → `--isl 8192 --osl 1024`).

**User provides table to fill in:**

> [TTT column already populated, TP/DP shows 8,1 for low conc and 8,8 for high conc]

→ Run with the matching `(isl, osl)`. Use the user's `TP, DP` value to pick the dpa variant (`8,1`=false, `8,8`=true). Verify TTT matches `tput_per_gpu * 8` for each row before pasting latency numbers.

## Notes

- The PAT in `~/.git-credentials` must have `actions:read` and `repo` (for private repos like InferenceX). The script does not echo the token.
- GitHub redirects artifact zip downloads to Azure blob storage with a signed URL; the script strips the `Authorization` header on redirect to avoid a `401 InvalidAuthenticationInfo`.
- Latency fields in `agg_bmk.json` are in **seconds** — always multiply by 1000 when rendering.
