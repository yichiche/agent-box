export const meta = {
  name: 'kernel-fusion-pipeline',
  description: 'Compare two trace xlsx files, identify Tier-1 fusions, implement in parallel worktrees, validate with agent script, commit+push+PR per worktree',
  whenToUse: 'Use when the user says /kernel-fusion-pipeline followed by two xlsx paths, or asks to run the full kernel fusion pipeline end-to-end.',
  phases: [
    { title: 'Setup', detail: 'GPU planning, trace comparison, slot assignment' },
    { title: 'Implement', detail: 'Parallel implementation in git worktrees' },
    { title: 'Validate', detail: 'Wave-based validation using agent script' },
    { title: 'Ship', detail: 'Commit, push, and create PRs' },
  ],
}

const FUSION_SCHEMA = {
  type: 'object',
  properties: {
    fusions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          slug: { type: 'string', description: 'snake_case identifier' },
          block_name: { type: 'string', description: 'Human-readable name of the fused block' },
          target_op: { type: 'string', description: 'aiter op or custom Triton kernel' },
          kernels: { type: 'array', items: { type: 'string' }, description: 'Kernels being fused' },
          savings_us: { type: 'number', description: 'Estimated savings in microseconds per layer' },
          sglang_file: { type: 'string', description: 'Path to the SGLang file to modify' },
          tier: { type: 'string', enum: ['1', '2', '3'] },
        },
        required: ['slug', 'block_name', 'target_op', 'sglang_file', 'tier'],
      },
    },
  },
  required: ['fusions'],
}

const IMPL_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    status: { type: 'string', enum: ['pass', 'fail'] },
    error: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
  },
  required: ['slug', 'status'],
}

const METRIC_SCHEMA = {
  type: 'object',
  description: 'baseline vs after for one metric (baseline null if not collected)',
  properties: {
    baseline: { type: ['number', 'null'] },
    after: { type: ['number', 'null'] },
    delta_pct: { type: ['number', 'null'], description: '(after-baseline)/baseline*100' },
  },
}

const VALIDATE_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    status: { type: 'string', enum: ['pass', 'fail', 'skip'] },
    error: { type: 'string' },
    model: { type: 'string', description: 'Model name used for validation' },

    // Accuracy (Step 3)
    accuracy: { type: 'number', description: 'GSM8K accuracy score' },
    accuracy_threshold: { type: 'number' },

    // After median ITL (ms) — used by the pipeline pass/fail + logging
    itl_ms: { type: 'number' },

    // Kernel-time perf gate (REQUIRED for the gate). Measured on the REPLACED kernel region
    // from the profiling trace: before = sum of the kernels removed, after = the fused kernel(s).
    kernel_time_before_us: { type: ['number', 'null'], description: 'Replaced-region kernel time before fusion (us)' },
    kernel_time_after_us: { type: ['number', 'null'], description: 'Replaced-region kernel time after fusion (us)' },
    kernel_time_improvement_pct: { type: ['number', 'null'], description: '(before-after)/before*100; positive = faster. Drives the >=1% perf gate.' },

    // E2E benchmark comparison (Step 6b table)
    baseline_available: { type: 'boolean' },
    benchmark: {
      type: 'object',
      description: 'Six metrics, each baseline/after/delta_pct',
      properties: {
        total_throughput: METRIC_SCHEMA,
        output_throughput: METRIC_SCHEMA,
        median_ttft_ms: METRIC_SCHEMA,
        median_itl_ms: METRIC_SCHEMA,
        median_tpot_ms: METRIC_SCHEMA,
        median_e2e_ms: METRIC_SCHEMA,
      },
    },

    // Kernel-to-E2E impact analysis (Step 6a)
    kernel_to_e2e: {
      type: 'object',
      properties: {
        layers: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              layer_type: { type: 'string' },
              layer_count: { type: 'number' },
              savings_per_layer_us: { type: 'number' },
              total_savings_us: { type: 'number' },
            },
          },
        },
        total_savings_us: { type: 'number' },
        expected_itl_pct: { type: 'number' },
        observed_itl_pct: { type: 'number' },
        assessment: { type: 'string' },
        ttft_impact: { type: 'string' },
      },
    },

    // Profiling confirmation (Step 4)
    profiling_confirmed: { type: 'boolean' },
    profiling_observations: { type: 'string' },
    trace_analysis_path: { type: 'string' },

    // Fully-composed markdown report (Step 6b), embedded verbatim in the PR body
    validation_report: {
      type: 'string',
      description: 'Complete "=== Validation Summary ===" markdown per /validate SKILL Step 6b',
    },
  },
  required: ['slug', 'status', 'validation_report'],
}

const PR_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    commit_hash: { type: 'string' },
    pr_url: { type: 'string' },
    pr_title: { type: 'string', description: 'The descriptive PR title actually used' },
    status: { type: 'string', enum: ['pass', 'fail'] },
    error: { type: 'string' },
  },
  required: ['slug', 'status'],
}

// ── Phase 1: Setup ─────────────────────────────────────────────────────────

phase('Setup')

const fileA = args?.fileA
const fileB = args?.fileB
if (!fileA || !fileB) {
  log('ERROR: args.fileA and args.fileB are required (two xlsx paths)')
  return { error: 'Missing xlsx paths. Pass as args: { fileA: "...", fileB: "..." }' }
}

log(`Analyzing: ${fileA} vs ${fileB}`)

const setup = await agent(`
You are setting up a kernel fusion pipeline. Do the following steps IN ORDER:

1. Detect SGLang root:
   SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")

2. Run GPU slot planning:
   python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp 2 --port-base 8001 --json
   If max_parallel == 0, return an error saying which GPUs are occupied.

3. Recategorize traces:
   python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py --recategorize ${fileA} ${fileB}

4. Compare kernels between the two xlsx files by READING AND FOLLOWING the skill file
   /home/yichiche/agent-box/skills/compare-kernels/SKILL.md (Steps 0-6, PIPELINE_MODE — do not just
   work from the skill name; the file holds the required module-grouped output format).
   Produce its full comparison (Step 3c functional-block boxes with named kernels + us, Step 4 time
   totals) and SAVE that markdown to ~/.kernel-fusion-pipeline/comparison.md. The distilled fusion
   list below is for routing; comparison.md preserves the kernel-level Before/After tables the PR needs.
   Extract ALL Tier-1 fusion opportunities. Tier-1 includes:
   - Existing aiter fused ops (Python-only dispatch change)
   - Custom Triton kernels (fuse multiple elementwise/small ops)
   Verify aiter ops exist under /home/yichiche/aiter/aiter/ops/ or /sgl-workspace/aiter/aiter/ops/.
   Drop invalid candidates.

5. For each Tier-1 fusion, record: slug, block_name, target_op, kernels, savings_us, sglang_file, tier.
   Rank by estimated savings (highest first).

6. Assign fusions to GPU slots:
   python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp 2 --port-base 8001 --assign <slug1> <slug2> ...

7. Create worktree farm:
   python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/worktree_farm.py --sglang-root "$SGLANG_ROOT" --slugs <slug1> <slug2> ...

8. Read and return the full state from ~/.kernel-fusion-pipeline/state.json

Return the Tier-1 fusion list in the structured format.
Do NOT ask any questions. This is PIPELINE_MODE — fully autonomous.

File A: ${fileA}
File B: ${fileB}
`, { label: 'setup', schema: FUSION_SCHEMA })

if (!setup || !setup.fusions || setup.fusions.length === 0) {
  log('No Tier-1 fusions found. Pipeline complete.')
  return { status: 'no_fusions', setup }
}

const tier1 = setup.fusions.filter(f => f.tier === '1')
if (tier1.length === 0) {
  log('No Tier-1 fusions found. Higher-tier fusions listed in setup results.')
  return { status: 'no_tier1', fusions: setup.fusions }
}

log(`Found ${tier1.length} Tier-1 fusion(s): ${tier1.map(f => f.slug).join(', ')}`)

// ── Phase 2: Implement ────────────────────────────────────────────────────

phase('Implement')
log(`Implementing ${tier1.length} fusion(s) in parallel worktrees...`)

const stateRaw = await agent('Read ~/.kernel-fusion-pipeline/state.json and return its full contents as a string.', { label: 'read-state' })

const implResults = await parallel(tier1.map(fusion => () =>
  agent(`
You are implementing a kernel fusion in a git worktree. PIPELINE_MODE — no EnterPlanMode, no AskUserQuestion.
If you hit a transient API/rate-limit error, back off and retry a few times rather than aborting.

Fusion: ${fusion.slug}
  Block: ${fusion.block_name}
  Target op: ${fusion.target_op}
  SGLang file: ${fusion.sglang_file}
  Kernels to fuse: ${JSON.stringify(fusion.kernels || [])}

Read ~/.kernel-fusion-pipeline/state.json to find your worktree path under worktrees.${fusion.slug}.path.
cd into that worktree and implement the Tier-1 change.

Rules:
- Follow sglang-backend-gated-changes: use _use_aiter guard, no top-level aiter imports, common path byte-identical
- For aiter ops: switch dispatch to the fused op, guarded by _use_aiter
- For custom Triton kernels: write a new Triton kernel in the appropriate location
- Max 3 lint/syntax retries
- Run: python3 -c "import sglang" to verify no import errors
- Return your slug, status (pass/fail), any error message, and list of files changed

Do NOT modify any files outside your worktree.
`, { label: `impl:${fusion.slug}`, schema: IMPL_RESULT_SCHEMA })
))

const implemented = []
const skipped = []
for (let i = 0; i < tier1.length; i++) {
  const r = implResults[i]
  if (r && r.status === 'pass') {
    implemented.push({ ...tier1[i], impl: r })
    log(`PASS: ${tier1[i].slug}`)
  } else {
    skipped.push({ ...tier1[i], impl: r })
    log(`SKIP: ${tier1[i].slug} — ${r?.error || 'agent returned null'}`)
  }
}

if (implemented.length === 0) {
  log('All implementations failed. Pipeline complete.')
  return { status: 'all_impl_failed', skipped }
}

// ── Phase 3: Validate ─────────────────────────────────────────────────────

phase('Validate')

const stateForValidate = await agent('Read ~/.kernel-fusion-pipeline/state.json and return its full JSON contents as a string.', { label: 'read-state-2' })

let fusionPlan
try {
  const parsed = JSON.parse(stateForValidate)
  fusionPlan = parsed.fusion_plan || []
} catch {
  fusionPlan = []
}

const maxWave = fusionPlan.reduce((m, f) => Math.max(m, f.wave || 0), 0)

const ACCURACY_THRESHOLD = args?.threshold || 0.85
// Perf gate: a fusion must improve the replaced-kernel-region time by at least this %.
// On-par or negative is NOT acceptable as a merge-ready perf change — such fusions still get a
// PR (so the work is visible) but it is opened as a DRAFT with an explicit judgement.
const PERF_GATE_PCT = args?.perf_gate_pct ?? 1.0
const MAX_DEBUG_RETRIES = 3
const validated = []
const failedValidation = []

function makeValidatePrompt(fusion, gpus, port, retryContext) {
  const retryBlock = retryContext
    ? `\nPREVIOUS ATTEMPT FAILED: ${retryContext}\nFIRST fix the code in the worktree (find the root cause, edit), THEN re-run validation.\n`
    : ''

  // SSOT: the full validate procedure, 6 metrics, kernel-to-E2E math, pass/fail rule, and the
  // "=== Validation Summary ===" report format all live in validate/SKILL.md. The orchestrator
  // only injects CONFIG + fusion vars and enforces the output schema.
  const CONFIG = {
    pipeline_mode: true,
    worktree: `read from ~/.kernel-fusion-pipeline/state.json -> worktrees.${fusion.slug}.path`,
    gpus, port,
    threshold: ACCURACY_THRESHOLD,
    // single command that BOTH launches the server and runs the benchmark — use it for the
    // baseline AND the after runs (replaces SKILL's separate server_script/client_script; model
    // is set inside the script):
    server_bench_cmd: `PORT=${port} HIP_VISIBLE_DEVICES=${gpus} PYTHONPATH_OVERRIDE=<worktree>/python KEEP_SERVER=1 bash /home/yichiche/run_qwen3.5_mxfp4_perf_agent.sh`,
    profiler: '/home/yichiche/agent-box/profile/trace_module_analyzer.py',
    fusion: {
      slug: fusion.slug,
      block_name: fusion.block_name,
      target_op: fusion.target_op,
      kernels: fusion.kernels || [],
      estimated_savings_us: fusion.savings_us,
      sglang_file: fusion.sglang_file,
    },
  }

  return `
You are validating a kernel fusion END-TO-END and producing a PR-ready report. PIPELINE_MODE — fully autonomous.

Read and FOLLOW /home/yichiche/agent-box/skills/validate/SKILL.md Steps 1-6 with its PIPELINE_MODE rules
(use the CONFIG below instead of asking; baseline via git per Step 1a; never AskUserQuestion/EnterPlanMode).
${retryBlock}
CONFIG = ${JSON.stringify(CONFIG, null, 2)}

Output: return every field of the provided result schema. In particular, validation_report MUST be the
complete "=== Validation Summary ===" markdown EXACTLY per SKILL Step 6b.

PERF GATE (REQUIRED): from the profiling trace, measure the REPLACED kernel region:
  kernel_time_before_us = sum of the kernels this fusion removes (baseline launches)
  kernel_time_after_us  = time of the fused kernel(s) that replace them
  kernel_time_improvement_pct = (before - after) / before * 100   (positive = faster)
Report all three. If the trace cannot establish before/after for the region, set
kernel_time_improvement_pct = null and explain in profiling_observations. Do NOT guess a positive number.
`
}

for (let wave = 0; wave <= maxWave; wave++) {
  const waveItems = implemented.filter(f => {
    const plan = fusionPlan.find(p => p.fusion_slug === f.slug)
    return plan && plan.wave === wave
  })

  if (waveItems.length === 0) continue
  log(`Wave ${wave}: validating ${waveItems.map(f => f.slug).join(', ')}`)

  // First attempt: all wave items in parallel
  const waveResults = await parallel(waveItems.map(fusion => () => {
    const plan = fusionPlan.find(p => p.fusion_slug === fusion.slug)
    const gpus = plan?.gpus || '4,5'
    const port = plan?.port || 8001
    return agent(makeValidatePrompt(fusion, gpus, port, null), {
      label: `validate:${fusion.slug}`,
      schema: VALIDATE_RESULT_SCHEMA,
    })
  }))

  // Check results, retry failures up to MAX_DEBUG_RETRIES
  for (let i = 0; i < waveItems.length; i++) {
    const fusion = waveItems[i]
    const plan = fusionPlan.find(p => p.fusion_slug === fusion.slug)
    const gpus = plan?.gpus || '4,5'
    const port = plan?.port || 8001
    let r = waveResults[i]
    let passed = false

    for (let attempt = 0; attempt <= MAX_DEBUG_RETRIES; attempt++) {
      const agentSaysPass = r && r.status === 'pass'
      const accuracyOk = r && typeof r.accuracy === 'number' && r.accuracy >= ACCURACY_THRESHOLD

      if (agentSaysPass && accuracyOk) {
        // Accuracy is the hard correctness gate (retried above). Perf is NOT a retry condition —
        // an on-par/negative result is a real outcome, so route it to a DRAFT PR with judgement.
        const impr = typeof r.kernel_time_improvement_pct === 'number' ? r.kernel_time_improvement_pct : null
        const meetsPerf = impr !== null && impr >= PERF_GATE_PCT
        const shipMode = meetsPerf ? 'open' : 'draft'
        const imprText = impr === null ? 'unmeasurable' : `${impr.toFixed(1)}%`
        const perfJudgement = meetsPerf
          ? `Kernel-region time improved ${imprText} (>= ${PERF_GATE_PCT}% gate). Ready for review.`
          : `Kernel-region time improvement is ${imprText} — below the ${PERF_GATE_PCT}% gate (on-par or worse). `
            + `This is NOT a merge-ready perf win; opened as DRAFT for maintainer judgement. `
            + `Accuracy passed (${r.accuracy}), so the change is correct — it just does not speed up the hot path here.`
        validated.push({ ...fusion, validation: r, shipMode, perfJudgement })
        log(`${meetsPerf ? 'PASS (open)' : 'PASS (DRAFT — perf below gate)'}: ${fusion.slug} `
          + `(accuracy=${r.accuracy}, kernelΔ=${imprText}, ITL=${r.itl_ms}ms)`)
        passed = true
        break
      }

      // Build failure reason
      const reason = agentSaysPass && !accuracyOk
        ? `accuracy ${r?.accuracy ?? 'N/A'} < ${ACCURACY_THRESHOLD} (agent wrongly reported pass)`
        : (r?.error || r?.status || 'validation failed')

      if (attempt === MAX_DEBUG_RETRIES) {
        log(`FAIL: ${fusion.slug} — ${reason} (gave up after ${MAX_DEBUG_RETRIES} retries)`)
        failedValidation.push({ ...fusion, validation: { ...r, error: reason } })
        break
      }

      // Retry: spawn a debug+revalidate agent
      log(`RETRY ${attempt + 1}/${MAX_DEBUG_RETRIES}: ${fusion.slug} — ${reason}`)
      r = await agent(
        makeValidatePrompt(fusion, gpus, port, reason),
        { label: `validate-retry${attempt + 1}:${fusion.slug}`, schema: VALIDATE_RESULT_SCHEMA }
      )
    }
  }
}

if (validated.length === 0) {
  log('All validations failed. Pipeline complete.')
  return { status: 'all_validate_failed', failedValidation, skipped }
}

// ── Phase 4: Ship ─────────────────────────────────────────────────────────

phase('Ship')
log(`Shipping ${validated.length} fusion(s): ${validated.map(f => f.slug).join(', ')}`)

// House-style exemplar (condensed real PR #27656). The ship agent matches THIS structure/tone —
// narrative Motivation naming before-kernels+us, per-file Modifications, Accuracy table,
// kernel-level Before/After table + savings math, E2E table. NOT a verbatim template.
const REFERENCE_PR = `[AMD][Perf] Fuse QK RMSNorm + gate extraction Triton kernel for Qwen3.5 on HIP

## Motivation
In Qwen3.5's attention layers on HIP, the forward_prepare path runs three separate kernels:
1. \`elementwise_kernel\` -- copy q from interleaved buffer (4.5 us)
2. \`elementwise_kernel\` -- copy gate from interleaved buffer (4.8 us)
3. \`_fused_qk_gemma_rmsnorm_kernel\` -- normalize q and k (4.5 us)
This PR fuses all three into a single Triton kernel, eliminating 2 launches per attention layer.

## Modifications
- **python/sglang/srt/models/utils.py**: Add \`_fused_qk_gemma_rmsnorm_gate_kernel\` Triton kernel and wrapper ...
- **python/sglang/srt/models/qwen3_5.py**: Add \`forward_prepare_hip()\`; dispatch when \`_is_hip and self.attn_output_gate\`.

## Accuracy Tests
Model: Qwen3.5-397B-A17B-MXFP4, TP=2, MI355x
| Benchmark | Score | Threshold |
|-----------|:-----:|:---------:|
| GSM8K (2000 questions, parallel=2000) | 0.911 | 0.880 |

## Speed Tests and Profiling
### Kernel-level (per attention layer, decode)
| Kernel | Before (us) | After (us) | Notes |
|--------|:-:|:-:|-------|
| \`elementwise_kernel\` (q deinterleave) | 4.5 | -- | eliminated |
| \`_fused_qk_gemma_rmsnorm_gate_kernel\` | -- | 4.6 | new fused kernel |
| **Attention layer total** | **91.0** | **81.6** | **-9.4 us (-10.3%)** |
15 attention layers x 9.4 us = ~141 us savings per decode iteration.
### E2E benchmark
| Metric | Before | After | Delta |
|--------|:-:|:-:|:-:|
| Median ITL (ms) | 10.59 | 10.47 | -1.1% |
(+ official Checklist and Review-and-Merge-Process sections, kept verbatim from the repo template)`

const prResults = await parallel(validated.map(fusion => () => {
  const v = fusion.validation || {}
  const model = v.model || 'Qwen3.5-397B-A17B-MXFP4'
  const isTriton = (fusion.target_op || '').toLowerCase().includes('triton')
  const isDraft = fusion.shipMode === 'draft'
  const perfJudgement = fusion.perfJudgement || ''

  // Repo-relative paths only — never leak worktree/absolute paths into the PR.
  const toRepoRel = (p) => {
    if (!p) return p
    const s = String(p)
    const m = s.match(/(python\/sglang\/.*)$/) || s.match(/(sgl-kernel\/.*)$/)
    return m ? m[1] : s
  }
  const sglangFileRel = toRepoRel(fusion.sglang_file)
  const filesChangedRel = (fusion.impl?.files_changed || []).map(toRepoRel)

  // SSOT: the lint/commit/push/PR-create+update process lives in commit-push-pr/SKILL.md
  // (PIPELINE_MODE). The ship agent COMPOSES the title+body (no verbatim template) then injects CONFIG.
  const shipConfig = {
    pipeline_mode: true,
    worktree: `read from ~/.kernel-fusion-pipeline/state.json -> worktrees.${fusion.slug}.path`,
    sglang_root: 'read sglang_root from ~/.kernel-fusion-pipeline/state.json',
    repo: 'sgl-project/sglang',
    base: 'main',
    pr_body_file: `/tmp/pr_body_${fusion.slug}.md`,
    gh_env: 'GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN=""',
    draft: isDraft,
  }

  return agent(`
You are COMPOSING and shipping a PR for a validated kernel fusion. PIPELINE_MODE — fully autonomous.
Produce a PR indistinguishable in style from the reference below. Do NOT dump raw validation logs;
COMPOSE clean prose + tables from the data.

== Fusion ==
slug: ${fusion.slug}
block: ${fusion.block_name}
target_op: ${fusion.target_op}
kernels_fused: ${JSON.stringify(fusion.kernels || [])}
model: ${model}
file (repo-relative): ${sglangFileRel}
files_changed (repo-relative): ${JSON.stringify(filesChangedRel)}
implementation_type: ${isTriton ? 'custom Triton kernel' : 'existing fused aiter op (dispatch switch)'}

== Validation data — DISTILL into tables, DO NOT paste verbatim ==
${JSON.stringify(v, null, 2)}

If ~/.kernel-fusion-pipeline/comparison.md exists, read it for the kernel-level Before/After table.

STEPS:
1. cd into the worktree (path from state.json). Run \`git diff\` against ${shipConfig.base} so Modifications
   reflects the ACTUAL change. Read .github/pull_request_template.md in the worktree — its section structure
   (Motivation / Modifications / Accuracy Tests / Speed Tests and Profiling / Checklist / Review and Merge
   Process) is REQUIRED. Keep its Checklist + Review-and-Merge-Process sections verbatim from the template
   (tick only boxes you actually satisfied).
2. Compose the BODY, matching the reference:
   - Motivation: narrative; name the BEFORE kernels with their us timings and how many launches are eliminated.
   - Modifications: one **bold repo-relative path** bullet per changed file, human description; note the
     \`_use_aiter\` gating and that the common (non-aiter) path is byte-identical.
   - Accuracy Tests: table | Benchmark | Score | Threshold | Status | + model config line.
   - Speed Tests and Profiling: kernel-level Before/After table (named kernels + us), savings/iter math
     (savings/layer x layer_count), and an E2E table | Metric | Before | After | Delta |.
   - HONESTY: if there is no measurable e2e speedup, say so plainly and frame as kernel-launch reduction /
     CUDA-path parity. Do not oversell.
3. HARD EXCLUSIONS — must not appear anywhere in the body:
   - absolute/worktree paths (/root/..., /tmp/..., /sgl-workspace/..., .kernel-fusion-pipeline/...). Repo-relative only.
   - "=== Validation Summary ===" headers, "NOTE ON CONFIG", "Code fixes required", "REMAINING BLOCKER",
     or any "previous attempt failed / accuracy collapsed / ROOT CAUSE" debugging narrative.
   A genuine caveat/follow-up (e.g. a graph-capture limitation) goes in ONE neutral sentence under a "Notes"
   bullet — no log dumps.
4. TITLE (descriptive, like the reference): "[AMD][Perf] Fuse <A> + <B> into single ${isTriton ? 'Triton kernel' : 'aiter op'} for ${model} on HIP".
   If correctness/parity only with no measurable speedup, drop "[Perf]" → "[AMD] ...".
${isDraft ? `
DRAFT MODE — this fusion did NOT clear the >=1% kernel-time perf gate.
   - Use the "[AMD]" tag (NOT "[Perf]"). Frame the PR honestly as a kernel-launch reduction / correctness
     change that is NOT a measured speedup on this hardware. Do not imply a perf win in the title or Motivation.
   - Add a "## Maintainer Judgement" section near the top of the body with this verbatim assessment:
     ${JSON.stringify(perfJudgement)}
   - Create the PR as a DRAFT: pass --draft to \`gh pr create\` (or after creating, run
     \`${shipConfig.gh_env} gh pr ready <number> --undo --repo ${shipConfig.repo}\` to convert it to draft).
` : `
This fusion CLEARED the perf gate — open the PR normally (not a draft).
`}
5. Write the composed body VERBATIM to ${shipConfig.pr_body_file}. Then read and FOLLOW
   /home/yichiche/agent-box/skills/commit-push-pr/SKILL.md PIPELINE_MODE with:
   CONFIG = ${JSON.stringify(shipConfig, null, 2)}
   Use your composed title as both commit_subject and pr_title.
   NOTE: CONFIG.draft=${isDraft}. If true, the PR MUST end up as a draft (verify with \`gh pr view <n> --json isDraft\`).

== REFERENCE PR (match this style) ==
${REFERENCE_PR}

Return: slug, commit_hash, pr_url, pr_title, status.
`, { label: `ship:${fusion.slug}`, schema: PR_RESULT_SCHEMA })
}))

// ── Summary ───────────────────────────────────────────────────────────────

const shipped = []
const shipFailed = []
for (let i = 0; i < validated.length; i++) {
  const r = prResults[i]
  if (r && r.status === 'pass') {
    const mode = validated[i].shipMode === 'draft' ? 'DRAFT' : 'open'
    shipped.push({ slug: validated[i].slug, pr_url: r.pr_url, commit: r.commit_hash, mode, judgement: validated[i].perfJudgement })
    log(`PR created (${mode}): ${validated[i].slug} → ${r.pr_url}`)
  } else {
    shipFailed.push({ slug: validated[i].slug, error: r?.error })
    log(`Ship failed: ${validated[i].slug} — ${r?.error || 'unknown'}`)
  }
}

log('=== Pipeline Complete ===')
log(`Shipped: ${shipped.length} | Validate-failed: ${failedValidation.length} | Impl-skipped: ${skipped.length} | Ship-failed: ${shipFailed.length}`)

return {
  shipped,
  failedValidation: failedValidation.map(f => ({ slug: f.slug, error: f.validation?.error })),
  skipped: skipped.map(f => ({ slug: f.slug, error: f.impl?.error })),
  shipFailed,
}
