---
name: inferencex-table
description: "Fetch and search InferenceX benchmark results. Two modes: (1) paste a GitHub Actions run URL to get a benchmark table from that specific run, or (2) search by model/hardware/framework (e.g. 'deepseekv4-pro on B200 sglang, MI355 vllm') to find the latest numbers across runs, including Docker image and benchmark script for reproducibility."
---

# inferencex-table

## When to use

Trigger this skill whenever the user:

- Pastes a GitHub Actions URL from `SemiAnalysisAI/InferenceX` and asks for a benchmark table (**Mode 1: Single Run**), or
- Asks for benchmark numbers by model, hardware, and/or framework — e.g. "show me DSV4 numbers on B200 sglang and MI355 vllm" (**Mode 2: Search**), or
- Wants to compare performance across platforms or frameworks for a given model, or
- Asks for the Docker image or benchmark script used for a specific configuration.

Do **not** use this for kernel-trace summaries (see `perf-summary`) or for accuracy/benchmark validation after code changes (see `validate`).

---

## Mode 1: Single Run (by URL or run ID)

When the user provides a specific run URL or numeric ID.

### Workflow

1. Extract the run id from the URL (regex: `actions/runs/(\d+)`). If only a number is given, use it directly.
2. Run:

   ```bash
   python3 ~/.claude/skills/inferencex-table/scripts/fetch_bmk.py <url-or-id> \
       [--isl 8192] [--osl 1024] [--dpa auto|true|false]
   ```

3. If the user already provided a partial table, match rows by `Concurrency` and verify by TTT (`tput_per_gpu * tp`).
4. Resolving DPA duplicates:
   - `TP, DP = 8, 1` → `dpa=false`; `TP, DP = 8, 8` → `dpa=true`.
   - If no pre-filled TTT, ask or show both variants.
5. Return the markdown table directly (no code fence unless requested).

### fetch_bmk.py flags

| Flag | Default | Behavior |
|---|---|---|
| `--isl N` / `--osl N` | none | Filter by sequence length |
| `--dpa auto\|true\|false` | `auto` | DPA variant filter |
| `--token PAT` | env / `~/.git-credentials` | GitHub auth override |
| `--out FILE` | stdout | Write to file |
| `--keep DIR` | tmp | Persist downloaded artifact |
| `--json` | off | Print raw JSON path |

---

## Mode 2: Search (by model / hardware / framework)

When the user asks for benchmark numbers without a specific run URL. The script scans recent completed "Run Sweep" runs to find matching data.

### Workflow

1. Parse the user's request to extract model, hardware, and framework. **If any required dimension is missing or ambiguous, ask the user before running the script.** In particular:
   - **Model is REQUIRED.** If the user did not specify a model, use AskUserQuestion to ask which model they want. Common options: `dsv4` (DeepSeek-V4-Pro), `dsr1` (DeepSeek-R1), `qwen3.5` (Qwen 3.5), `glm5` (GLM-5), `kimik2.5` (Kimi K2.5), `gptoss` (GPT-OSS), `minimaxm2.5` (MiniMax M2.5).
   - **Hardware**: infer from context (e.g. "amd" → `mi355x,mi300x,mi325x`; "nvidia" → `b200,h200,b300`). Ask if ambiguous.
   - **Framework**: infer from context. Ask if ambiguous.

2. Map common names to InferenceX identifiers:

   | User says | `--model` value | Notes |
   |---|---|---|
   | deepseekv4-pro, dsv4, deepseek-v4 | `dsv4` | model-prefix match |
   | deepseek-r1, dsr1 | `dsr1` | |
   | qwen3.5, qwen | `qwen3.5` | |
   | glm5, glm | `glm5` | |

   | User says | `--hw` value |
   |---|---|
   | MI355, mi355x | `mi355x` |
   | B200 | `b200` |
   | H200 | `h200` |
   | MI300, mi300x | `mi300x` |

3. Run the search script:

   ```bash
   python3 ~/.claude/skills/inferencex-table/scripts/search_bmk.py \
       --model dsv4 \
       --hw b200,mi355x \
       --framework sglang,vllm,atom \
       [--precision fp8] \
       [--isl 8192] [--osl 1024] \
       [--branch main] \
       [--limit 30] \
       [--all-metrics] \
       [--show-script] \
       [--compare] \
       [--reproduce]
   ```

4. Review the output:
   - **Default mode**: one section per (hw, framework, precision) combo, with image + script link + table.
   - **`--compare` mode**: side-by-side comparison table with all combos in one table (use when multiple hw/framework combos requested).
   - **`--all-metrics`**: expanded columns (p90/p99 latencies, power, energy).
   - **`--show-script`**: fetches and displays the full benchmark script content from the repo.
   - **`--reproduce`**: shows Docker pull command and key parameters for reproducing the run.

5. Return the output directly as markdown. Do not wrap in code fences.

### search_bmk.py flags

| Flag | Default | Behavior |
|---|---|---|
| `--model TEXT` | none | Model prefix or substring match (e.g. `dsv4`, `deepseek-v4`) |
| `--hw LIST` | none | Comma-separated hardware filter (e.g. `b200,mi355x`) |
| `--framework LIST` | none | Comma-separated framework filter (e.g. `sglang,vllm,atom`) |
| `--precision TEXT` | none | Precision filter (e.g. `fp8`, `fp4`) |
| `--isl N` / `--osl N` | none | Sequence length filter |
| `--branch TEXT` | all branches | Git branch to search (e.g. `main`) |
| `--limit N` | 30 | Max title-matched runs to scan |
| `--all-metrics` | off | Show p90/p99 latencies, power, energy columns |
| `--show-script` | off | Fetch and display benchmark script content |
| `--compare` | off | Side-by-side comparison table |
| `--reproduce` | off | Show reproducibility info (Docker image, params) |
| `--token PAT` | env / `~/.git-credentials` | GitHub auth override |
| `--out FILE` | stdout | Write to file |
| `--json` | off | Dump raw matched entries as JSON |

### Output format

**Default (per combo):**

```
## deepseek-ai/DeepSeek-V4-Pro on mi355x / sglang (fp4)

- **Image**: `rocm/sgl-dev:rocm720-mi35x-8c3b5aa-20260521-DSv4`
- **Script**: [`benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh`](https://github.com/...)
- **Run**: [26484490371](https://github.com/...) (2026-05-25)

### ISL=1024 / OSL=1024

| Conc | TP, DP | TTT (tok/s) | Out TPut/GPU | Med E2EL (ms) | Med TTFT (ms) | Med ITL (ms) | Med TPOT (ms) |
|---|---|---|---|---|---|---|---|
| 4 | 8, 1 | 1234.56 | 154.32 | 450.12 | 35.67 | 12.34 | 13.45 |
...
```

**Compare mode:**

```
# ISL=1024 / OSL=1024 — deepseek-ai/DeepSeek-V4-Pro

- **b200 / sglang (fp8)**: image=`...`, script=[`...`](...), run=[123](...)
- **mi355x / vllm (fp8)**: image=`...`, script=[`...`](...), run=[456](...)

| Platform | Framework | Prec | Conc | TP, DP | TTT (tok/s) | Out TPut/GPU | Med E2EL (ms) | Med TTFT (ms) | Med ITL (ms) | Med TPOT (ms) | Image |
```

### Data fields from `agg_bmk.json`

All metrics available per entry:

| Category | Fields |
|---|---|
| Identity | `hw`, `model`, `infmax_model_prefix`, `framework`, `precision`, `image` |
| Config | `tp`, `ep`, `dp_attention`, `conc`, `isl`, `osl`, `spec_decoding`, `disagg` |
| Throughput | `tput_per_gpu`, `output_tput_per_gpu`, `input_tput_per_gpu` |
| TTFT (s) | `mean_ttft`, `median_ttft`, `p90_ttft`, `p99_ttft`, `p99.9_ttft`, `std_ttft` |
| TPOT (s) | `mean_tpot`, `median_tpot`, `p90_tpot`, `p99_tpot`, `std_tpot` |
| ITL (s) | `mean_itl`, `median_itl`, `p90_itl`, `p99_itl`, `std_itl` |
| E2EL (s) | `mean_e2el`, `median_e2el`, `p90_e2el`, `p99_e2el`, `std_e2el` |
| Power | `avg_power_w`, `joules_per_output_token`, `joules_per_total_token` |

Latency fields are in **seconds** — always multiply by 1000 when rendering as ms.
TTT = `tput_per_gpu * tp` (total throughput across all GPUs).

---

## Deciding which mode to use

| User input | Mode | Script |
|---|---|---|
| Pastes a `github.com/.../actions/runs/12345` URL | Mode 1 | `fetch_bmk.py` |
| Provides a bare numeric run ID | Mode 1 | `fetch_bmk.py` |
| "Show me DSV4 numbers on B200 sglang" | Mode 2 | `search_bmk.py` |
| "Compare MI355 vs B200 for deepseek-r1" | Mode 2 + `--compare` | `search_bmk.py` |
| "What image does sglang use for DSV4 on MI355?" | Mode 2 | `search_bmk.py` |
| "How to reproduce DSR1 benchmark on B200?" | Mode 2 + `--reproduce --show-script` | `search_bmk.py` |

---

## Examples

### Example 1: User pastes URL

> https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26118426149
> Make a benchmark table

→ Use Mode 1 with `fetch_bmk.py`.

### Example 2: Search by model and platforms

> I want deepseekv4-pro version, B200 sglang / MI355 vllm / MI355 atom / MI355 sglang

→ Run:
```bash
python3 ~/.claude/skills/inferencex-table/scripts/search_bmk.py \
    --model dsv4 \
    --hw b200,mi355x \
    --framework sglang,vllm,atom \
    --compare
```

### Example 3: Full reproducibility info

> What's the latest DSR1 setup on MI355? I want to reproduce it.

→ Run:
```bash
python3 ~/.claude/skills/inferencex-table/scripts/search_bmk.py \
    --model dsr1 --hw mi355x \
    --show-script --reproduce
```

### Example 4: All metrics for a specific config

> Show me all latency percentiles for DSV4 on B200 sglang, ISL=8192

→ Run:
```bash
python3 ~/.claude/skills/inferencex-table/scripts/search_bmk.py \
    --model dsv4 --hw b200 --framework sglang \
    --isl 8192 --all-metrics
```

---

## Notes

- The PAT in `~/.git-credentials` must have `actions:read` and `repo` scope.
- GitHub redirects artifact downloads to Azure blob storage; both scripts strip the `Authorization` header on redirect to avoid `401 InvalidAuthenticationInfo`.
- Latency fields in `agg_bmk.json` are in **seconds** — always multiply by 1000 when rendering.
- Search mode scans up to `--limit` recent completed "Run Sweep" runs; it stops early once all requested (hw, framework) combos are found.
- Benchmark script paths follow the convention: `benchmarks/single_node/{model_prefix}_{precision}_{hw}_{framework}.sh`.
