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

const VALIDATE_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    status: { type: 'string', enum: ['pass', 'fail', 'skip'] },
    accuracy: { type: 'number' },
    itl_ms: { type: 'number' },
    error: { type: 'string' },
    benchmark_summary: { type: 'string' },
  },
  required: ['slug', 'status'],
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
   python3 ~/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp 2 --port-base 8001 --json
   If max_parallel == 0, return an error saying which GPUs are occupied.

3. Recategorize traces:
   python3 ~/agent-box/profile/trace_module_analyzer.py --recategorize ${fileA} ${fileB}

4. Compare kernels between the two xlsx files following the /compare-kernels skill (Steps 1-6).
   Extract ALL Tier-1 fusion opportunities. Tier-1 includes:
   - Existing aiter fused ops (Python-only dispatch change)
   - Custom Triton kernels (fuse multiple elementwise/small ops)
   Verify aiter ops exist under ~/aiter/aiter/ops/ or /sgl-workspace/aiter/aiter/ops/.
   Drop invalid candidates.

5. For each Tier-1 fusion, record: slug, block_name, target_op, kernels, savings_us, sglang_file, tier.
   Rank by estimated savings (highest first).

6. Assign fusions to GPU slots:
   python3 ~/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp 2 --port-base 8001 --assign <slug1> <slug2> ...

7. Create worktree farm:
   python3 ~/agent-box/skills/kernel-fusion-pipeline/scripts/worktree_farm.py --sglang-root "$SGLANG_ROOT" --slugs <slug1> <slug2> ...

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
    ? `\n\nPREVIOUS ATTEMPT FAILED:\n${retryContext}\nYou MUST fix the code in the worktree before re-validating. Read the error, diagnose the root cause, edit the code, then re-run validation.\n`
    : ''

  return `
You are validating a kernel fusion. PIPELINE_MODE — fully autonomous.
${retryBlock}
Fusion: ${fusion.slug}
  Worktree: read from ~/.kernel-fusion-pipeline/state.json -> worktrees.${fusion.slug}.path
  GPUs: ${gpus}
  Port: ${port}

Steps:
1. Read the worktree path from ~/.kernel-fusion-pipeline/state.json${retryContext ? '\n2. FIRST: Fix the code based on the failure reason above. Edit files in the worktree.' : ''}
${retryContext ? '3' : '2'}. Run the agent script to launch server + benchmark:
   PORT=${port} HIP_VISIBLE_DEVICES=${gpus} PYTHONPATH_OVERRIDE=<worktree>/python KEEP_SERVER=1 \\
     bash ~/run_qwen3.5_mxfp4_perf_agent.sh
${retryContext ? '4' : '3'}. After benchmark completes, run accuracy test against the still-running server on port ${port}.
   Use /validate skill's accuracy test approach.
${retryContext ? '5' : '4'}. Kill the server: pkill -f "sglang.launch_server.*--port ${port}"
${retryContext ? '6' : '5'}. Evaluate: accuracy >= ${ACCURACY_THRESHOLD} AND ITL not regressed >2% = PASS

Return: slug, status, accuracy, itl_ms, benchmark_summary, any error.
If server fails to start (agent script will detect quickly), return status=fail with the error.
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
  const benchText = fusion.validation?.benchmark_summary || 'No benchmark data'
  const accText = fusion.validation?.accuracy ?? 'N/A'
  const itlText = fusion.validation?.itl_ms ?? 'N/A'

  const prBody = `## Summary
- ${fusion.block_name}: fuse ${(fusion.kernels || []).length} kernels, ~${fusion.savings_us || '?'} us/layer savings
- Target op: ${fusion.target_op}
- Files changed: ${fusion.impl?.files_changed?.join(', ') || 'see diff'}

## Validation Results

**Benchmark:**
${benchText}

**Accuracy:** ${accText}
**Mean ITL:** ${itlText} ms

## Test plan
- [x] Accuracy validation (>=${ACCURACY_THRESHOLD} threshold) — ${accText}
- [x] Benchmark ITL regression check
- [x] Server launch + health check via agent script`

  return agent(`
You are committing and creating a PR for a validated kernel fusion. PIPELINE_MODE — fully autonomous.

Fusion: ${fusion.slug}
  Block: ${fusion.block_name}

Steps:
1. Read worktree path from ~/.kernel-fusion-pipeline/state.json -> worktrees.${fusion.slug}.path
2. cd into the worktree
3. Stage and commit:
   git add -A
   git commit -m "[AMD] ${fusion.slug}: ${fusion.block_name} — fuse ${(fusion.kernels || []).length} kernels, ~${fusion.savings_us || '?'} us/layer"
   Do NOT add Co-Authored-By lines.
4. Push the branch. Then create a PR to sgl-project/sglang with EXACTLY this body (do not modify it):

${prBody}

   Use: GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN="" gh pr create --repo sgl-project/sglang --base main --title "[AMD] ${fusion.slug}: ${fusion.block_name}" --body "<the body above>"
   If gh pr create fails due to PAT issues, try with GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN="" prefix.
5. After PR is created, clean up the worktree:
   cd to the SGLang root (read sglang_root from state.json)
   git worktree remove <worktree_path>
   Keep the local branch.
6. Record the commit hash and PR URL.

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
