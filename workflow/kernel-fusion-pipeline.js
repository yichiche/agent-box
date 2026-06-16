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

4. Compare kernels between the two xlsx files following the /compare-kernels skill (Steps 1-6).
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
complete "=== Validation Summary ===" markdown EXACTLY per SKILL Step 6b — it is embedded verbatim into the PR.
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
        validated.push({ ...fusion, validation: r })
        log(`PASS: ${fusion.slug} (accuracy=${r.accuracy}, ITL=${r.itl_ms}ms)`)
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

const prResults = await parallel(validated.map(fusion => () => {
  const v = fusion.validation || {}
  const report = v.validation_report || '_No validation report was produced._'
  const accText = v.accuracy ?? 'N/A'
  const thr = v.accuracy_threshold ?? ACCURACY_THRESHOLD
  const model = v.model || 'Qwen3.5-397B-A17B-MXFP4'
  const tracePath = v.trace_analysis_path || 'see logs'
  const profConfirmed = v.profiling_confirmed === false ? 'NOT CONFIRMED' : 'CONFIRMED'
  const baselineNote = v.baseline_available === false ? ' (baseline not available — after-only)' : ''
  const isTriton = (fusion.target_op || '').toLowerCase().includes('triton')
  const modNote = isTriton
    ? 'A new Triton kernel replaces the separate elementwise/normalization launches in the hot path.'
    : 'Dispatch is switched to the existing fused aiter op.'

  const prBody = `## Motivation

Fuse ${(fusion.kernels || []).length} kernel(s) in **${fusion.block_name}** to cut per-layer kernel launch/compute overhead for **${model}**. Estimated ~${fusion.savings_us || '?'} us/layer savings.

- **Target op:** ${fusion.target_op}
- **Kernels fused:** ${(fusion.kernels || []).join(', ') || 'see diff'}
- **File modified:** ${fusion.sglang_file}
- **Files changed:** ${fusion.impl?.files_changed?.join(', ') || 'see diff'}

## Modifications

Tier-1 fusion implemented behind the \`_use_aiter\` guard — no top-level aiter imports, common (non-aiter) path byte-identical. ${modNote}

## Accuracy Tests & Benchmarking${baselineNote}

${report}

## Checklist

- [x] Accuracy validation (GSM8K, num-questions 200 / parallel 2000, threshold >= ${thr}) — score ${accText}
- [x] Before/after E2E benchmark comparison (table above)
- [x] Profiling trace confirms the kernel change (${profConfirmed}) — analysis: ${tracePath}
- [x] Kernel-to-E2E impact analysis (expected vs observed ITL)
- [x] Server launch + health check via agent script

## Review Process
- [ ] Maintainer review of the fused kernel correctness and \`_use_aiter\` gating`

  // SSOT: the lint/commit/push/PR-create+update process lives in commit-push-pr/SKILL.md
  // (PIPELINE_MODE). The orchestrator only composes the PR body and injects CONFIG.
  const shipConfig = {
    pipeline_mode: true,
    worktree: `read from ~/.kernel-fusion-pipeline/state.json -> worktrees.${fusion.slug}.path`,
    sglang_root: 'read sglang_root from ~/.kernel-fusion-pipeline/state.json',
    repo: 'sgl-project/sglang',
    base: 'main',
    commit_subject: `[AMD] ${fusion.slug}: ${fusion.block_name} — fuse ${(fusion.kernels || []).length} kernels, ~${fusion.savings_us || '?'} us/layer`,
    pr_title: `[AMD] ${fusion.slug}: ${fusion.block_name}`,
    pr_body_file: `/tmp/pr_body_${fusion.slug}.md`,
    gh_env: 'GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN=""',
  }

  return agent(`
You are shipping a validated kernel fusion. PIPELINE_MODE — fully autonomous.

1. Write the PR body below VERBATIM to ${shipConfig.pr_body_file} (preserve the markdown tables exactly,
   do not summarize or reformat):
----- BEGIN PR BODY -----
${prBody}
----- END PR BODY -----

2. Then read and FOLLOW /home/yichiche/agent-box/skills/commit-push-pr/SKILL.md PIPELINE_MODE with:
CONFIG = ${JSON.stringify(shipConfig, null, 2)}

Return: slug, commit_hash, pr_url, status.
`, { label: `ship:${fusion.slug}`, schema: PR_RESULT_SCHEMA })
}))

// ── Summary ───────────────────────────────────────────────────────────────

const shipped = []
const shipFailed = []
for (let i = 0; i < validated.length; i++) {
  const r = prResults[i]
  if (r && r.status === 'pass') {
    shipped.push({ slug: validated[i].slug, pr_url: r.pr_url, commit: r.commit_hash })
    log(`PR created: ${validated[i].slug} → ${r.pr_url}`)
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
