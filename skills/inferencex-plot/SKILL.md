---
name: inferencex-plot
description: "Turn local benchmark output into an import-ready CSV for the InferenceX Curve site (https://duyi-wang.github.io/InferenceXCurve/), so curves land on the chart in one Import File click instead of being typed point by point. Accepts a perf-sweep run dir, summary.csv, benchmark_serving result JSONs, or numbers pasted into the chat, converts units to the site's contract (tok/s/gpu, tok/s/user, ms→s), and validates before writing. Use when the user says '/inferencex-plot', 'plot these on InferenceX Curve', 'make a CSV for InferenceXCurve', or wants a before/after Pareto curve on that site."
category: meta
---

# inferencex-plot — benchmark output → InferenceX Curve CSV

The site: <https://duyi-wang.github.io/InferenceXCurve/> (repo `duyi-wang/InferenceXCurve`).

It has **no URL/query data loading** — data lives in the browser's `localStorage`.
So the only agent-side integration point is: **write a contract-valid CSV → the
user clicks `Import File` → they manipulate the chart in the UI.** That is the
whole point of this skill; do not try to drive the chart itself.

## When to use

- "plot my sweep on InferenceX Curve" / "make me a CSV for that curve site"
- before/after A/B: two curves on one chart from one import
- any time the alternative is hand-typing points into the site's data editor

Not this skill: **`/inferencex-table`** for fetching *upstream* InferenceX
GitHub Actions numbers; **`/perf-sweep`** to actually produce the benchmark data.
This skill only formats numbers you already have.

## The two CSV contracts are NOT interchangeable

| Workspace | Hash | Button | Columns | Contract |
| --- | --- | --- | --- | --- |
| **InferenceX Curve** (default) | `#/inferencex` | `Import File` | 57 canonical | `reference/import-csv.md` |
| Plot Tool | `#/plot-tool` | `Import CSV` | `Line ID,Line Name,X,Y,+style` | `reference/import-plot-tool-csv.md` |

Default to **InferenceX Curve** — it keeps model / scenario / precision / MTP /
parallelism, so the site's filters and roofline work. Use `--target plot-tool`
only when the user wants a bare Pareto plot with no benchmark metadata.

`reference/*.md` are vendored copies of the upstream contracts. **Read them
before hand-writing any CSV** — they are the source of truth for columns and
aliases, and they work offline.

## Workflow

1. **Locate the numbers.** Sweep dir, `summary.csv`, `result_conc*.json`, or
   text the user pasted.
2. **Resolve the metadata checklist** below — this is the part the benchmark
   output does *not* contain. Infer what you can; ask only for the rest.
3. **Convert** with `scripts/ix_plot.py` (it validates before writing; it will
   refuse rather than emit a file the site would reject).
4. **Report** the path *and* the echoed CSV, plus the import steps.

### Metadata checklist (resolve before converting)

| Field | Required | Where to infer it |
| --- | --- | --- |
| `--gpus` | **yes** | `run_meta.env`, `TP`/`GPUS` in the reference `run_*.sh`, `HIP_VISIBLE_DEVICES`. **Never guess** — the script refuses, because a wrong count rescales the entire Y axis. |
| `--model` | **yes** | model path in the launch script. Use the site's spelling, e.g. `DeepSeek-R1-0528`, `Qwen3.5-397B-A17B-MXFP4`. |
| `--scenario` | **yes** | `ISL 8192 / OSL 1024` from `INPUT_LEN`/`OUTPUT_LEN`. Numeric ISL/OSL files under *Fixed Sequence Length*; use the literal `Agentic Traces` for trace replay. |
| `--precision` | **yes** | `fp4` / `fp8` / `bf16` — from the model name or quant flags. |
| `--line-id` | **yes** | stable slug; the grouping key. Same id = same curve. |
| `--line-name` | **yes** | the legend label — make it read well: `Qwen3.5 MXFP4 MI355 TP8 (aiter PR)`. |
| `--mtp` | preferred | `MTP` or `Non-MTP`; defaults to `Non-MTP`. Be explicit. |
| `--decode-tp/-ep`, `--disagg`, `--hw`, `--line-note` | optional | tooltips and filters; fill when known. |

Per the CLAUDE.md defaults: no shape given ⇒ produce **both** `diag-1k`
(ISL 1024 / OSL 1024) and `canonical-8k` (ISL 8192 / OSL 1024) as separate
scenarios, and claim only on `canonical-8k`.

## One line per SKU — merge topologies, do NOT split them

The InferenceX convention is **one curve per hardware SKU** (per model /
precision / framework / MTP), whose points are the **best achievable across every
tested topology** — TP2 and TP4, offload on and off, all on the same line. The
site's own `Import Action Data` does exactly this: a whole sweep arrives as one
line named e.g. `B200 (SGLang MTP)`.

**Do not emit one line per TP.** A chart with `B200 TP2` and `B200 TP4` as
separate curves is wrong for this site even though it validates — the user
expects the envelope, and splitting it hides which topology wins where.

Because `Throughput/GPU` normalizes by GPU count, TP2 (2 GPU) and TP4 (4 GPU)
points are directly comparable on both axes, so one Pareto pass over the union is
meaningful. It also reproduces the real shape: on Qwen3.5 FP4 B200 AgentX, TP4
owns the high-interactivity end (conc 4-24, 390 → 199 tok/s/user) and TP2 with
HiCache owns the high-throughput end (conc 16-32, up to 71.3k tok/s/gpu) — one
curve, crossing over at ~166 tok/s/user.

Mechanically: give every topology the **same `line_id`**, and put `gpus`,
`decode_gpus`, `decode_tp`, `decode_ep` on each **point** rather than on the line
(`point_metrics` reads `pt["gpus"]` before `line["gpus"]`, and the deployment
columns already fall back point → line). Line-level fields (`Line Name`, `Model`,
`Scenario`, `Precision`, `MTP`, `Line Type`, …) must still be identical on every
row of that id. Put the topology in each point's `note`
(`TP4 (4 GPU), conc 16, offload off, P90`) so the tooltip still tells you which
arm a point came from.

Keep a separate line only for things that are genuinely different *experiments*
— a before/after A/B, a different image, your own run vs the published one — not
for different parallelism of the same run.

## Pareto frontier is the DEFAULT

`ix_plot.py` keeps only the **non-dominated** points per line and sorts them by
X descending. A point is dropped when another point on the same line is at least
as good on BOTH axes (X = interactivity, Y = throughput/GPU) and strictly better
on one.

This is the default because the user expects "一條最佳化那樣的結果" — a clean
optimization curve. A raw sweep runs past the knee, and those post-saturation
points sit far inside the frontier: on a real B200 TP4 AgentX sweep, conc 60-64
fell to ~13k tok/s/gpu with P90 E2E over 260 s while conc 40 held 44k at 19.6 s.
Plotted, that reads as a broken chart, not a result.

The script prints exactly what it dropped, per line, so nothing disappears
silently:

```
pareto: qwen35-fp4-b200-agentx-published-tp4: dropped 10 dominated point(s) (conc 1, 32, 48, 56, 60, 62, 64, 68, 70, 72)
pareto: kept 54/76 point(s) (pass --all-points to keep every point)
```

**Only pass `--all-points` when the user explicitly asks** for the full sweep —
"full", "every point", "完整", "show the knee", "where does it saturate". When
they do, write it to a **separate** `<stem>-full.csv` and keep the Pareto file as
the primary one; do not overwrite the default with the full set.

Relay the kept/total count and the dropped concurrencies when reporting — the
dropped points are usually the interesting ones (saturation, a bad run, a
harness artifact), even though they do not belong on the curve.

## Preferred delivery: serve it locally, already loaded

Do NOT stop at "here is a CSV, go click Import File". The user is on a laptop and
the data is on the GPU box, so a download → upload round trip is the whole cost
of this workflow. Serve the real app locally with the CSV already on the chart:

```bash
~/.claude/skills/inferencex-plot/scripts/serve_local.sh <csv>...
```

It copies the CSVs into the app's `public/`, starts Vite on 127.0.0.1:5173 if it
is not already up, and prints the SSH tunnel command plus a `?seed=` URL per CSV.

Setup, once per machine:

```bash
git clone https://github.com/duyi-wang/InferenceXCurve.git ~/InferenceXCurve
cd ~/InferenceXCurve && npm install
# then apply the local/dev-seed branch (the ?seed= bootstrap at the end of src/main.ts)
```

Two payoffs beyond skipping the round trip:

- **Sync works.** The published Pages app cannot reach `inferencex.semianalysis.com`
  because the `/api/v1/*` routes send no `Access-Control-Allow-Origin`. Vite's dev
  proxy is same-origin, so `InferenceX Sync` and `Import Action Data` both work
  locally with no CORS extension and no third-party proxy.
- **The user's browser stays clean** — no token pasted into the site, no
  CORS-unblocking extension left enabled.

### Two failure modes that make a seed silently no-op

Both were hit for real; the import reports success while the chart is unchanged.

1. **Seeding before the app finishes initializing.** The app boots asynchronously
   and loads its example data during init, overwriting anything seeded at module
   scope. Wait for `#model-filter` to have options before importing.
2. **Not moving the filters.** The lines import fine but the filter selects keep
   their default model and scenario, so the chart still draws the example series.
   Set `#model-filter`, `#scenario-filter`, `#precision-filter`, `#mtp-filter`
   from the CSV's first data row and dispatch `change` on each.

Verify with Playwright rather than trusting the status text — the in-page
"Appended N lines" message appears in both failure modes. Check that the model
option list actually contains the seeded model, and screenshot the chart.

## Output location — ALWAYS `/home/yichiche/inferencex-plots/`

Every generated CSV goes in this fixed folder, so the user always knows where to
look and the browser-side `Import File` step points at one stable directory:

```bash
IXDIR=/home/yichiche/inferencex-plots     # create with: mkdir -p "$IXDIR"
```

**Always pass `--out "$IXDIR/<name>.csv"` explicitly.** Two ways the default
would put the file somewhere the user cannot find it:

- The script falls back to `~/inferencex-plots`, and the agent usually runs as
  **root in a container** — that resolves to `/root/inferencex-plots`, not the
  user's home.
- When reading a sweep dir or `summary.csv`, the script writes **next to the
  source file** instead, scattering CSVs across run directories.

If the session runs as root, hand ownership back so the user can edit the file:

```bash
mkdir -p "$IXDIR" && chown -R 11065:11065 "$IXDIR"
```

Naming: `<model-slug>-<hw>-<scenario-or-purpose>.csv`, e.g.
`qwen35-mxfp4-mi355x-agentx-ab.csv`. Keep the file stable across re-runs of the
same comparison so `--append` keeps working. Also save the spec JSON beside it as
`<same-stem>.spec.json` — it is the only record of how the numbers were mapped,
and it makes a later regeneration (fixing a metadata guess, adding a curve) a
one-line edit instead of a re-derivation.

## Usage

```bash
IX=~/.claude/skills/inferencex-plot/scripts/ix_plot.py
IXDIR=/home/yichiche/inferencex-plots

# perf-sweep summary.csv → CSV
python3 $IX --from-sweep RUNDIR/summary.csv --out "$IXDIR/qwen35-mxfp4-mi355x-8k.csv" \
  --line-id qwen35-mxfp4-mi355-8k --line-name "Qwen3.5 MXFP4 MI355 TP8" \
  --model "Qwen3.5-397B-A17B-MXFP4" --scenario "ISL 8192 / OSL 1024" \
  --precision fp4 --mtp Non-MTP --gpus 8 --decode-tp 8 --decode-ep 8 --hw mi355x

# raw benchmark_serving result JSONs
python3 $IX --from-bench-json 'RUNDIR/result_conc*.json' --out "$IXDIR/<name>.csv" \
  --line-id ... (same flags)

# anything else (pasted stdout, odd CSV shapes): you build the spec
python3 $IX --spec spec.json --out "$IXDIR/<name>.csv"   # '-' reads stdin

python3 $IX --check "$IXDIR/<name>.csv"    # validate an existing file

# full sweep, every point — ONLY when the user explicitly asks; separate file
python3 $IX --spec spec.json --all-points --out "$IXDIR/<name>-full.csv"
```

Key flags added by this workflow: `--all-points` (opt out of the Pareto default),
`--pareto` (explicit no-op, harmless to pass for clarity).

Key flags: `--append` (merge into an existing CSV — same `Line ID` replaces that
curve, new ids are added), `--target plot-tool`, `--y-metric total|output`,
`--out`, `--stdout` (print only), `--no-echo` (suppress the copy/paste dump).

### Spec JSON (the general path)

Point keys are exactly sglang's `benchmark_serving.py` field names, so pasted
output maps over with no renaming:

```json
{"lines": [{
  "line_id": "qwen35-mxfp4-8k", "line_name": "Qwen3.5 MXFP4 MI355 TP8",
  "model": "Qwen3.5-397B-A17B-MXFP4", "scenario": "ISL 8192 / OSL 1024",
  "precision": "fp4", "mtp": "Non-MTP", "gpus": 8,
  "decode_tp": 8, "decode_ep": 8, "disagg": false,
  "line_type": "solid", "line_note": "baseline",
  "points": [
    {"concurrency": 4, "total_throughput": 8280, "median_tpot_ms": 46.1,
     "median_ttft_ms": 420, "median_e2e_latency_ms": 47600, "note": "conc4"}
  ]}]}
```

Escape hatches when you only have derived numbers: `interactivity`,
`throughput_per_gpu`, `ttft_s`, `e2e_s` are accepted directly and skip the
conversion.

## Unit conversion (done by the script — do not pre-convert)

| InferenceX column | From |
| --- | --- |
| `Throughput/GPU (tok/s/gpu)` — **Y** | `total_throughput / gpus` (`--y-metric output` for output-only) |
| `Interactivity (tok/s/user)` — **X** | `1000 / median_tpot_ms` (falls back to `median_itl_ms`) |
| `TTFT (s)` | `median_ttft_ms / 1000` |
| `End-to-end (s)` | `median_e2e_latency_ms / 1000` |
| `Concurrency` | `max_concurrency` |
| `Total GPUs` | left empty — the app recomputes it (a derived column, ignored on import) |

**Y is total (input+output) throughput per GPU by default** — the user's choice.
At ISL 8192 that is prefill-weighted and therefore *not* directly comparable to
InferenceMax dashboard numbers; say so when reporting, and offer
`--y-metric output` if they want dashboard-comparable curves.

## Parsing recipes for pasted output

**perf-sweep's printed table** (`=== SWEEP SUMMARY ===`) — columns are
`conc, out_tok/s, tot_tok/s, med_TTFT, med_TPOT, med_ITL, med_E2E`; TTFT/ITL/E2E
are **ms**. Map `tot_tok/s`→`total_throughput`, `med_TPOT`→`median_tpot_ms`,
`med_TTFT`→`median_ttft_ms`, `med_E2E`→`median_e2e_latency_ms`.

**Raw `============ Serving Benchmark Result ============` blocks** — one block
per concurrency; read `Total token throughput (tok/s)`, `Median TPOT (ms)`,
`Median TTFT (ms)`, `Median E2E Latency (ms)`, and the `Maximum request
concurrency`. One point per block.

**A/B comparison tables** (e.g. `agent-scratch/benchmark_comparisons/*/summary.csv`,
which has a two-row header and `baseline` / `<variant>` column pairs) — these do
**not** match `--from-sweep`. Parse them yourself into a spec with **two lines**,
one per column pair, sharing scenario/model. Watch the units and check whether
the throughput column is total or output before assigning `--y-metric`.

## Multi-curve / before-after

Put every curve in **one file** so the user imports once:

```bash
python3 $IX --from-sweep before/summary.csv --line-id run-before \
  --line-name "baseline" ... --out "$IXDIR/run-ab.csv"
python3 $IX --from-sweep after/summary.csv  --line-id run-after \
  --line-name "aiter PR #1234" --line-type dashed ... --out "$IXDIR/run-ab.csv" --append
```

Line-level fields must be **identical on every row of a given `Line ID`** — the
importer rejects the whole file otherwise. The script enforces this; don't
hand-edit rows afterwards.

## Reporting to the user

Give them the path, then the import steps, then the echoed CSV (the benchmark
usually runs in a container while the browser is on their laptop, so the
copy/paste fallback matters):

> Open <https://duyi-wang.github.io/InferenceXCurve/#/inferencex> → `Import File`
> → pick the CSV → review the staged lines → `Add`.

Zero/negative values block the site's Log Scale toggle; the script warns, so
pass the warning along.

## Related: importing upstream Actions runs (browser-side, no agent involvement)

The site can also pull an InferenceX **GitHub Actions run URL** directly:
`Import Action Data` → paste `https://github.com/owner/repo/actions/runs/<id>` →
enter a GitHub token → staged for review.

**The token is entered in the site's own field in the user's browser** (stored in
`localStorage` under `inferencex-curve:github-token:v1`, plaintext, that browser
only). Never ask the user to paste a token into the chat, never put one in a
file, and never handle it yourself — just describe these steps. If a token does
land in the transcript, tell them to revoke it.

Token guidance from the site's README: repo you own ⇒ fine-grained PAT with
`Actions: Read-only` on that repo; private repo owned by someone else ⇒ classic
PAT with `repo` scope.

**Token-free alternative:** `Import File` also accepts `.zip`, and unpacks it in
the browser. So `gh run download <id>` locally, then import the artifact zip
directly — no token in the browser at all. Prefer offering this first.
