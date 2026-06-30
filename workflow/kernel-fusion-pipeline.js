export const meta = {
  name: 'kernel-fusion-pipeline',
  description: 'Compare two trace xlsx files, identify Tier-1 opportunities, then run two tracks: (a) sglang code fusions in parallel worktrees -> validate -> sglang PR, and (b) aiter GEMM/MoE autotune (no .so rebuild) serially on the shared aiter install -> validate -> aiter PR. Each candidate touches exactly one repo.',
  whenToUse: 'Use when the user says /kernel-fusion-pipeline followed by two xlsx paths, or asks to run the full kernel fusion pipeline end-to-end. Candidates that need aiter config retuning are shipped as aiter PRs; sglang code fusions as sglang PRs.',
  phases: [
    { title: 'Setup', detail: 'GPU planning, trace comparison, slot assignment' },
    { title: 'Implement', detail: 'Parallel implementation in git worktrees' },
    { title: 'Validate', detail: 'Wave-based validation using agent script' },
    { title: 'Ship', detail: 'Commit, push, and create PRs' },
    { title: 'Tune (aiter)', detail: 'Serial aiter GEMM/MoE autotune + aiter PR' },
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
          block_name: { type: 'string', description: 'Human-readable name of the fused block / tuned op' },
          target_op: { type: 'string', description: 'aiter op, custom Triton kernel, or the GEMM/MoE op to autotune' },
          kernels: { type: 'array', items: { type: 'string' }, description: 'Kernels being fused / the kernel being tuned' },
          savings_us: { type: 'number', description: 'Estimated savings in microseconds per layer' },
          tier: { type: 'string', enum: ['1', '2', '3'] },

          // Routing: which repo the change lands in, and what kind of change it is.
          repo: { type: 'string', enum: ['sglang', 'aiter'], description: 'sglang = code fusion (worktree + sglang PR); aiter = GEMM/MoE autotune (aiter config CSV + aiter PR)' },
          kind: { type: 'string', enum: ['fuse', 'tune'], description: 'fuse = code change; tune = run the official aiter autotuner to regenerate a tuned config CSV (no .so rebuild)' },

          // FUSE (repo=sglang) only.
          sglang_file: { type: 'string', description: 'FUSE only: path to the SGLang file to modify' },

          // TUNE (repo=aiter) only.
          kernel_type: { type: 'string', enum: ['gemm_tune', 'moe_tune'], description: 'TUNE only: which official aiter tuner family' },
          tune_script: { type: 'string', description: 'TUNE only: official tuner, e.g. csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py' },
          config_csv: { type: 'string', description: 'TUNE only: the tuned CSV the op reads at runtime, repo-relative under aiter/configs/' },
          untuned_csv: { type: 'string', description: 'TUNE only: the matching *_untuned_*.csv listing shapes to tune' },
          shape: { type: 'string', description: 'TUNE only: the workload shape (M,N,K / tokens,experts) to tune for' },
        },
        required: ['slug', 'block_name', 'target_op', 'tier', 'repo', 'kind'],
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

// Pre-tune gate: BEFORE running the official tuner, prove the runtime actually selects this op's
// kernel FROM the tuned CSV for the DEPLOYED model signature. Many aiter MoE/GEMM ops dispatch the
// kernel structurally (by dtype/quant/activation/ksplit conditionals), never reading kernelName1/2
// from the tuned CSV — for those, regenerating the CSV is a no-op (the classic "using 2stage default"
// trap). This gate also corrects the tune shape to the bucket the runtime actually keys on.
const TUNE_GATE_SCHEMA = {
  type: 'object',
  description: 'Verdict on whether an aiter tune candidate is genuinely CSV-tunable for the deployed model.',
  properties: {
    slug: { type: 'string' },
    tunable: {
      type: 'boolean',
      description: 'TRUE only if (a) the runtime selects the kernel FROM the tuned CSV (kernelName1/2) for '
        + 'the model signature — NOT a structural/hardcoded dispatch — AND (b) the tuner emits kernels in the '
        + 'SAME family as the kernels actually seen in the baseline trace for this op.',
    },
    reason: { type: 'string', description: 'Concise justification, citing the dispatch code path and trace kernels.' },
    runtime_kernels: { type: 'array', items: { type: 'string' }, description: 'Kernels actually executed for this op in the baseline trace.' },
    tuner_kernel_family: { type: 'string', description: 'Kernel family the official tuner emits into the tuned CSV.' },
    dispatch_path: { type: 'string', description: 'The exact fused_moe/gemm dispatch branch (file:line) that picks the kernel for this signature.' },
    key_token_bucket: { type: ['number', 'null'], description: 'The token value the runtime keys the CSV on = get_padded_M(token_num) = nextPow2(token_num). The tune shape MUST use this, not the raw token count.' },
    corrected_shape: { type: 'string', description: 'The tune shape rewritten to use the padded token bucket (and any other corrected dims).' },
  },
  required: ['slug', 'tunable', 'reason'],
}

const TUNE_IMPL_SCHEMA = {
  type: 'object',
  description: 'Result of running the official aiter autotuner for one tune candidate (Route B).',
  properties: {
    slug: { type: 'string' },
    status: { type: 'string', enum: ['pass', 'fail'] },
    error: { type: 'string' },
    patch_path: { type: 'string', description: 'File the tuned-CSV diff was written to (the deliverable carrier)' },
    config_csv: { type: 'string', description: 'Repo-relative tuned CSV that changed' },
    rows_changed: { type: 'number', description: 'Number of CSV rows added/updated by the tuner' },
    tuner_summary: { type: 'string', description: 'What shapes/kernels improved + any TFLOPS/us numbers from tuner output' },
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
    repo: { type: 'string', enum: ['sglang', 'aiter'] },
    commit_hash: { type: 'string' },
    pr_url: { type: 'string' },
    pr_title: { type: 'string', description: 'The descriptive PR title actually used' },
    mode: { type: 'string', enum: ['open', 'draft', 'patch_only'], description: 'patch_only = no remote configured; commit + .patch only, no PR' },
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

const focus = args?.focus || ''
const maxGpus = args?.max_gpus ?? null
const maxGpusFlag = maxGpus != null ? ` --max-gpus ${maxGpus}` : ''
log(`Analyzing: ${fileA} vs ${fileB}${focus ? ` | FOCUS: ${focus}` : ''}${maxGpus != null ? ` | MAX GPUs: ${maxGpus}` : ''}`)

const setup = await agent(`
You are setting up a kernel fusion pipeline. Do the following steps IN ORDER:
${focus ? `\nFOCUS: Restrict this run to "${focus}". In steps 4-5, only extract and keep candidates whose
functional block is part of "${focus}"; drop unrelated blocks (MoE/MLP/router/etc. if the focus is attention).
Rank the in-focus candidates by savings.\n` : ''}

1. Detect SGLang root:
   SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")

2. Run GPU slot planning:
   python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp 2 --port-base 8001${maxGpusFlag} --json
   If max_parallel == 0, return an error saying which GPUs are occupied.

3. Recategorize traces:
   python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py --recategorize ${fileA} ${fileB}

4. Compare kernels between the two xlsx files by READING AND FOLLOWING the skill file
   /home/yichiche/agent-box/skills/compare-kernels/SKILL.md (Steps 0-6, PIPELINE_MODE — do not just
   work from the skill name; the file holds the required module-grouped output format).
   Produce its full comparison (Step 3c functional-block boxes with named kernels + us, Step 4 time
   totals) and SAVE that markdown to ~/.kernel-fusion-pipeline/comparison.md. The distilled candidate
   list below is for routing; comparison.md preserves the kernel-level Before/After tables the PR needs.
   Extract ALL Tier-1 opportunities${focus ? ` THAT ARE WITHIN THE FOCUS "${focus}"` : ''}. Tier-1 has TWO families:

   FAMILY 1 — sglang code fusion (repo="sglang", kind="fuse"):
   - Existing aiter fused ops (Python-only dispatch change)
   - Custom Triton kernels (fuse multiple elementwise/small ops)
   Verify aiter ops exist under /home/yichiche/aiter/aiter/ops/ or /sgl-workspace/aiter/aiter/ops/.

   FAMILY 2 — aiter GEMM / fused-MoE AUTOTUNE (repo="aiter", kind="tune"):
   - A slow GEMM or fused-MoE kernel whose op has an OFFICIAL aiter tuner (csrc/<op>/<op>_tune.py) AND a
     tuned CSV (aiter/configs/<op>_tuned_*.csv) the op reads at runtime, where the WORKLOAD shape is MISSING
     or sub-optimal in that CSV. Re-tuning needs NO .so rebuild — the tuner just rewrites the CSV.
   - Verify the tuner script AND the tuned/untuned CSVs exist under the installed aiter
     (AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")).
   - CRITICAL TUNABILITY PRECHECK (do NOT propose a tune candidate that fails this):
     (1) The runtime must SELECT the kernel FROM the tuned CSV (i.e. consume kernelName1/kernelName2 from the
         matched row) for the DEPLOYED model's exact signature. Many MoE/GEMM ops dispatch the kernel
         STRUCTURALLY (by dtype/quant/activation/ksplit conditionals) and never read the CSV kernel names —
         e.g. the Qwen3.5 MXFP4 per_1x32/fp4x2/Silu path picks cktile (MoeFlatmmKernel) / mxgemm_2lds
         structurally, so tuning tuned_fmoe.csv is a NO-OP. Read aiter/fused_moe.py get_2stage_cfgs + the
         dispatch branches and confirm the model's signature lands on a CSV-driven branch.
     (2) The tuner's emitted kernel family MUST match the kernels actually seen in the baseline trace for this
         op (e.g. tuner emits moe_ck2stages_gemm/asm/flydsl but the trace shows mxgemm/cktile -> NOT tunable).
     (3) The CSV lookup is an EXACT match keyed on token = get_padded_M(token_num) = nextPow2(token_num)
         (see fused_moe.py get_padded_M). Record the tune "shape" using this PADDED bucket, never the raw
         observed token count, or the tuned row will never be looked up.

   Drop invalid candidates${focus ? ` and drop anything outside the focus` : ''}, including any FAMILY 2
   candidate that fails the tunability precheck.

5. For each Tier-1 candidate record: slug, block_name, target_op, kernels, savings_us, tier, and:
   - repo: "sglang" for code fusions, "aiter" for autotune candidates.
   - kind: "fuse" for code fusions, "tune" for autotune candidates.
   - FUSE (repo=sglang): also record sglang_file (the SGLang file to modify).
   - TUNE (repo=aiter): also record kernel_type ("gemm_tune"|"moe_tune"), tune_script (csrc/.../*_tune.py),
     config_csv (aiter/configs/*_tuned_*.csv), untuned_csv (the matching *_untuned_*.csv), and shape (the
     M,N,K or tokens/experts workload to tune). Leave sglang_file empty for tune candidates.
   Rank by estimated savings (highest first).

6. Assign ALL Tier-1 candidates (both fuse AND tune slugs) to GPU slots:
   python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp 2 --port-base 8001${maxGpusFlag} --assign <all_slugs...>

7. Create the worktree farm — ONLY for repo=sglang (fuse) slugs. Tune candidates change the shared aiter
   install (not a worktree), so they get NO worktree:
   python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/worktree_farm.py --sglang-root "$SGLANG_ROOT" --slugs <sglang_fuse_slugs...>
   (If there are no sglang fuse slugs, skip this step.)

8. Read and return the full state from ~/.kernel-fusion-pipeline/state.json

Return the Tier-1 candidate list in the structured format.
Do NOT ask any questions. This is PIPELINE_MODE — fully autonomous.

File A: ${fileA}
File B: ${fileB}
`, { label: 'setup', schema: FUSION_SCHEMA })

if (!setup || !setup.fusions || setup.fusions.length === 0) {
  log('No Tier-1 candidates found. Pipeline complete.')
  return { status: 'no_fusions', setup }
}

const tier1 = setup.fusions.filter(f => f.tier === '1')
if (tier1.length === 0) {
  log('No Tier-1 candidates found. Higher-tier opportunities listed in setup results.')
  return { status: 'no_tier1', fusions: setup.fusions }
}

// Split Tier-1 by target repo: sglang code fusions (parallel worktrees, sglang PR) vs
// aiter autotune candidates (serial on the shared aiter install, aiter PR).
const fusions = tier1.filter(f => (f.repo || 'sglang') === 'sglang')
const tunes = tier1.filter(f => f.repo === 'aiter')
log(`Tier-1: ${fusions.length} sglang fusion(s) [${fusions.map(f => f.slug).join(', ') || '-'}] | `
  + `${tunes.length} aiter tune(s) [${tunes.map(f => f.slug).join(', ') || '-'}]`)

// ── Shared config ───────────────────────────────────────────────────────────

const ACCURACY_THRESHOLD = args?.threshold || 0.85
// Perf gate: a candidate must improve the replaced-kernel-region time by at least this %. On-par or
// negative is NOT a merge-ready perf change — it still gets a PR (so the work is visible) but the PR is
// opened as a DRAFT with an explicit judgement.
const PERF_GATE_PCT = args?.perf_gate_pct ?? 1.0
const MAX_DEBUG_RETRIES = 3

// aiter PR ship config (Route B). With NO push remote configured, the tune track produces a
// report + .patch (committed on a local branch) instead of opening a PR, so work is never lost.
const AITER_GH_ENV = args?.aiter_gh_env || 'GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN=""'
const AITER_REMOTE = args?.aiter_remote || ''
const AITER_BASE = args?.aiter_base || 'main'
const AITER_REPO = args?.aiter_repo || 'ROCm/aiter'

// GPU/wave plan written by setup's plan_gpu_slots --assign (covers fuse AND tune slugs).
const stateRaw = await agent('Read ~/.kernel-fusion-pipeline/state.json and return its full JSON contents as a string.', { label: 'read-state' })
let fusionPlan = []
try {
  fusionPlan = JSON.parse(stateRaw).fusion_plan || []
} catch {
  fusionPlan = []
}
function planFor(slug, fallbackGpus = '4,5', fallbackPort = 8001) {
  const p = fusionPlan.find(x => x.fusion_slug === slug)
  return { gpus: p?.gpus || fallbackGpus, port: p?.port || fallbackPort, wave: p?.wave || 0 }
}

// Result accumulators shared by both tracks.
const shipped = []
const shipFailed = []
const failedValidation = []
const skipped = []

// ── Shared validate-prompt builder (sglang fusions AND aiter tunes) ───────────

function makeValidatePrompt(item, gpus, port, retryContext) {
  const isTune = item.kind === 'tune'
  const retryBlock = retryContext
    ? `\nPREVIOUS ATTEMPT FAILED: ${retryContext}\nFIRST fix the root cause (${isTune ? 'aiter config / tuner' : 'code in the worktree'}), THEN re-run validation.\n`
    : ''

  // SSOT: the full validate procedure, 6 metrics, kernel-to-E2E math, pass/fail rule, and the
  // "=== Validation Summary ===" report format all live in validate/SKILL.md. The orchestrator only
  // injects CONFIG + candidate vars and enforces the output schema.
  const CONFIG = {
    pipeline_mode: true,
    target_repo: isTune ? 'aiter' : 'sglang',
    worktree: isTune
      ? 'N/A — aiter tune; the change is a tuned CSV in the shared aiter install, applied via a patch'
      : `read from ~/.kernel-fusion-pipeline/state.json -> worktrees.${item.slug}.path`,
    gpus, port,
    threshold: ACCURACY_THRESHOLD,
    // single command that BOTH launches the server and runs the benchmark — use it for baseline AND
    // after runs (model is set inside the script). For aiter tune, do NOT override PYTHONPATH (no worktree).
    server_bench_cmd: `PORT=${port} HIP_VISIBLE_DEVICES=${gpus} ${isTune ? '' : 'PYTHONPATH_OVERRIDE=<worktree>/python '}KEEP_SERVER=1 bash /home/yichiche/run_qwen3.5_mxfp4_perf_agent.sh`,
    profiler: '/home/yichiche/agent-box/profile/trace_module_analyzer.py',
    fusion: {
      slug: item.slug,
      block_name: item.block_name,
      target_op: item.target_op,
      kernels: item.kernels || [],
      estimated_savings_us: item.savings_us,
      sglang_file: isTune ? (item.config_csv || '') : item.sglang_file,
      ...(isTune ? { kind: 'tune', kernel_type: item.kernel_type, config_csv: item.config_csv, shape: item.shape } : {}),
    },
  }

  return `
You are validating a kernel ${isTune ? 'autotune' : 'fusion'} END-TO-END and producing a PR-ready report. PIPELINE_MODE — fully autonomous.

Read and FOLLOW /home/yichiche/agent-box/skills/validate/SKILL.md Steps 1-6 with its PIPELINE_MODE rules
(use the CONFIG below instead of asking; never AskUserQuestion/EnterPlanMode).
${retryBlock}${isTune ? `
AITER TUNE VALIDATION — the change is an aiter tuned-config CSV applied via a patch (NOT a worktree):
  AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")  (fallback /sgl-workspace/aiter)
  patch: ${item.patch_path}
  tuned CSV: ${item.config_csv}
  - BASELINE run: 'git -C $AITER_ROOT checkout -- ${item.config_csv}' (clean committed config), then launch server + bench + profile.
  - AFTER run: 'git -C $AITER_ROOT apply ${item.patch_path}' (apply the tuned rows), then launch server + bench + profile.
  - kernel_time_before/after = the SAME GEMM/MoE kernel time with committed vs tuned config (same kernel, faster).
  - WHEN DONE revert: 'git -C $AITER_ROOT checkout -- ${item.config_csv}' so the shared install is clean for the next candidate. LEAVE ${item.patch_path} in place.
  - The server reads the SHARED aiter install; set HIP_VISIBLE_DEVICES and the applied/clean CSV state correctly for each run.
` : `
Baseline via git in the worktree per SKILL Step 1a.
`}
CONFIG = ${JSON.stringify(CONFIG, null, 2)}

Output: return every field of the provided result schema. In particular, validation_report MUST be the
complete "=== Validation Summary ===" markdown EXACTLY per SKILL Step 6b.

PERF GATE (REQUIRED): from the profiling trace, measure the REPLACED kernel region:
  kernel_time_before_us = ${isTune ? 'the GEMM/MoE kernel time with the committed (baseline) config' : 'sum of the kernels this fusion removes (baseline launches)'}
  kernel_time_after_us  = ${isTune ? 'the same kernel time with the tuned config' : 'time of the fused kernel(s) that replace them'}
  kernel_time_improvement_pct = (before - after) / before * 100   (positive = faster)
Report all three. If the trace cannot establish before/after for the region, set
kernel_time_improvement_pct = null and explain in profiling_observations. Do NOT guess a positive number.
`
}

// Evaluate accuracy/perf gates + retries for one item. `initial` is an already-run first result
// (sglang waves run the first attempt in parallel); omit it to run the first attempt here (aiter, serial).
// Accuracy is the hard correctness gate (retried). Perf is NOT a retry condition — on-par/negative is a
// real outcome routed to a DRAFT PR with a judgement.
async function validateWithRetries(item, gpus, port, initial) {
  const phaseName = item.kind === 'tune' ? 'Tune (aiter)' : 'Validate'
  let r = initial !== undefined
    ? initial
    : await agent(makeValidatePrompt(item, gpus, port, null), { label: `validate:${item.slug}`, phase: phaseName, schema: VALIDATE_RESULT_SCHEMA })

  for (let attempt = 0; attempt <= MAX_DEBUG_RETRIES; attempt++) {
    const agentSaysPass = r && r.status === 'pass'
    const accuracyOk = r && typeof r.accuracy === 'number' && r.accuracy >= ACCURACY_THRESHOLD

    if (agentSaysPass && accuracyOk) {
      const impr = typeof r.kernel_time_improvement_pct === 'number' ? r.kernel_time_improvement_pct : null
      const meetsPerf = impr !== null && impr >= PERF_GATE_PCT
      const shipMode = meetsPerf ? 'open' : 'draft'
      const imprText = impr === null ? 'unmeasurable' : `${impr.toFixed(1)}%`
      const perfJudgement = meetsPerf
        ? `Kernel-region time improved ${imprText} (>= ${PERF_GATE_PCT}% gate). Ready for review.`
        : `Kernel-region time improvement is ${imprText} — below the ${PERF_GATE_PCT}% gate (on-par or worse). `
          + `This is NOT a merge-ready perf win; opened as DRAFT for maintainer judgement. `
          + `Accuracy passed (${r.accuracy}), so the change is correct — it just does not speed up the hot path here.`
      return { passed: true, r, shipMode, perfJudgement, imprText }
    }

    const reason = agentSaysPass && !accuracyOk
      ? `accuracy ${r?.accuracy ?? 'N/A'} < ${ACCURACY_THRESHOLD} (agent wrongly reported pass)`
      : (r?.error || r?.status || 'validation failed')

    if (attempt === MAX_DEBUG_RETRIES) {
      log(`FAIL: ${item.slug} — ${reason} (gave up after ${MAX_DEBUG_RETRIES} retries)`)
      return { passed: false, r: { ...r, error: reason }, reason }
    }

    log(`RETRY ${attempt + 1}/${MAX_DEBUG_RETRIES}: ${item.slug} — ${reason}`)
    r = await agent(makeValidatePrompt(item, gpus, port, reason), { label: `validate-retry${attempt + 1}:${item.slug}`, phase: phaseName, schema: VALIDATE_RESULT_SCHEMA })
  }
}

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

// ── Track 1: sglang code fusions (parallel worktrees -> sglang PR) ────────────

async function runSglangTrack() {
  // Implement (parallel worktrees)
  phase('Implement')
  log(`Implementing ${fusions.length} sglang fusion(s) in parallel worktrees...`)

  const implResults = await parallel(fusions.map(fusion => () =>
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
`, { label: `impl:${fusion.slug}`, phase: 'Implement', schema: IMPL_RESULT_SCHEMA })
  ))

  const implemented = []
  for (let i = 0; i < fusions.length; i++) {
    const r = implResults[i]
    if (r && r.status === 'pass') {
      implemented.push({ ...fusions[i], impl: r })
      log(`PASS (impl): ${fusions[i].slug}`)
    } else {
      skipped.push({ ...fusions[i], impl: r })
      log(`SKIP (impl): ${fusions[i].slug} — ${r?.error || 'agent returned null'}`)
    }
  }
  if (implemented.length === 0) {
    log('All sglang implementations failed — skipping sglang validate/ship.')
    return
  }

  // Validate (wave-based, parallel within a wave)
  phase('Validate')
  const maxWave = fusionPlan.reduce((m, f) => Math.max(m, f.wave || 0), 0)
  const validated = []

  for (let wave = 0; wave <= maxWave; wave++) {
    const waveItems = implemented.filter(f => planFor(f.slug).wave === wave)
    if (waveItems.length === 0) continue
    log(`Wave ${wave}: validating ${waveItems.map(f => f.slug).join(', ')}`)

    // First attempt: all wave items in parallel
    const waveResults = await parallel(waveItems.map(fusion => () => {
      const { gpus, port } = planFor(fusion.slug)
      return agent(makeValidatePrompt(fusion, gpus, port, null), { label: `validate:${fusion.slug}`, phase: 'Validate', schema: VALIDATE_RESULT_SCHEMA })
    }))

    // Check results, retry failures up to MAX_DEBUG_RETRIES
    for (let i = 0; i < waveItems.length; i++) {
      const fusion = waveItems[i]
      const { gpus, port } = planFor(fusion.slug)
      const v = await validateWithRetries(fusion, gpus, port, waveResults[i])
      if (v.passed) {
        validated.push({ ...fusion, validation: v.r, shipMode: v.shipMode, perfJudgement: v.perfJudgement })
        log(`${v.shipMode === 'open' ? 'PASS (open)' : 'PASS (DRAFT — perf below gate)'}: ${fusion.slug} `
          + `(accuracy=${v.r.accuracy}, kernelΔ=${v.imprText}, ITL=${v.r.itl_ms}ms)`)
      } else {
        failedValidation.push({ ...fusion, validation: v.r })
      }
    }
  }
  if (validated.length === 0) {
    log('All sglang validations failed — skipping sglang ship.')
    return
  }

  // Ship (sglang PRs)
  phase('Ship')
  log(`Shipping ${validated.length} sglang fusion(s): ${validated.map(f => f.slug).join(', ')}`)

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
`, { label: `ship:${fusion.slug}`, phase: 'Ship', schema: PR_RESULT_SCHEMA })
  }))

  for (let i = 0; i < validated.length; i++) {
    const r = prResults[i]
    if (r && r.status === 'pass') {
      const mode = validated[i].shipMode === 'draft' ? 'DRAFT' : 'open'
      shipped.push({ slug: validated[i].slug, repo: 'sglang', pr_url: r.pr_url, commit: r.commit_hash, mode, judgement: validated[i].perfJudgement })
      log(`PR created (${mode}, sglang): ${validated[i].slug} → ${r.pr_url}`)
    } else {
      shipFailed.push({ slug: validated[i].slug, repo: 'sglang', error: r?.error })
      log(`Ship failed (sglang): ${validated[i].slug} — ${r?.error || 'unknown'}`)
    }
  }
}

// ── Track 2: aiter GEMM/MoE autotune (serial -> aiter PR) ─────────────────────
// The aiter install is a SHARED editable tree (its configs are global, NOT worktree-isolated), so tune
// candidates MUST run one at a time: each one snapshots the baseline CSV, runs the tuner, captures the
// diff as a patch, validates E2E, then reverts the shared install to clean so the next one is isolated.

async function runAiterTuneTrack() {
  phase('Tune (aiter)')
  log(`Aiter tune track (serial, shared aiter install): ${tunes.map(t => t.slug).join(', ')}`)

  for (const tune of tunes) {
    const { gpus, port } = planFor(tune.slug)
    const patchFile = `/tmp/aiter_tune_${tune.slug}.patch`
    log(`Tune ${tune.slug}: op=${tune.target_op} type=${tune.kernel_type} shape="${tune.shape || 'n/a'}" gpus=${gpus} port=${port}`)

    // 0. TUNABILITY GATE — prove the runtime actually reads the tuned CSV for this op/signature BEFORE
    // spending a tuner run + 3 validation retries. Drops the "using 2stage default" / structural-dispatch
    // trap (e.g. MXFP4 cktile/mxgemm) up front, and corrects the shape to the padded token bucket.
    const gate = await agent(`
You are a STATIC-ANALYSIS gate. Decide whether an aiter tune candidate is genuinely CSV-tunable for the
DEPLOYED model BEFORE any tuner runs. Do NOT launch a server or run the tuner. PIPELINE_MODE — autonomous.

Candidate: ${tune.slug}
  Op / block: ${tune.block_name} (${tune.target_op})
  kernel_type: ${tune.kernel_type}
  Proposed shape: ${tune.shape || '(none given)'}
  Tuner script: ${tune.tune_script || '(unknown)'}
  Tuned CSV: ${tune.config_csv || '(unknown)'}

AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")  (fallback /sgl-workspace/aiter)

CHECK ALL THREE (candidate is tunable ONLY if all pass):
1. CSV-DRIVEN DISPATCH: Read $AITER_ROOT/aiter/fused_moe.py (get_2stage_cfgs + the dispatch branches it
   returns, ~lines 735-1140) or the relevant gemm dispatch. Determine whether, for the DEPLOYED model
   signature (quant_type, q_dtype_a/w, activation, g1u1, etc.), the runtime SELECTS the kernel from the
   matched CSV row's kernelName1/kernelName2, or picks it STRUCTURALLY (hardcoded by dtype/quant/activation/
   ksplit conditionals). Structural dispatch -> NOT tunable (regenerating the CSV cannot change the kernel).
   Record dispatch_path = the deciding branch as file:line.
2. KERNEL-FAMILY MATCH: Compare the kernel family the tuner emits into the tuned CSV against runtime_kernels =
   the kernels actually seen in the baseline trace for this op (read ~/.kernel-fusion-pipeline/comparison.md
   and/or the candidate's listed kernels). If they differ (tuner emits moe_ck2stages_gemm/asm/flydsl but the
   trace shows mxgemm_2lds / MoeFlatmm / cktile), the tuned rows are never executed -> NOT tunable.
3. KEY BUCKET: The CSV lookup is an EXACT match keyed on token = get_padded_M(token_num) = nextPow2(token_num)
   (fused_moe.py get_padded_M). Compute key_token_bucket from the workload's observed token count and rewrite
   corrected_shape to use that padded bucket (keep other dims). If the proposed shape used a raw token count,
   FIX it here.

Return the gate schema fields. Set tunable=false (with a precise reason) if ANY check fails. Be conservative:
if you cannot CONFIRM the runtime reads the CSV kernel name for this signature, tunable=false.
`, { label: `tune-gate:${tune.slug}`, phase: 'Tune (aiter)', schema: TUNE_GATE_SCHEMA })

    if (!gate || gate.tunable !== true) {
      skipped.push({ ...tune, gate })
      log(`SKIP (not CSV-tunable): ${tune.slug} — ${gate?.reason || 'gate failed'}`
        + (gate?.runtime_kernels?.length ? ` [trace kernels: ${gate.runtime_kernels.join(', ')}]` : ''))
      continue
    }
    const tuneShape = gate.corrected_shape || tune.shape
    log(`Gate PASS (CSV-tunable): ${tune.slug} — shape="${tuneShape}"`
      + (gate.key_token_bucket != null ? ` (token bucket ${gate.key_token_bucket})` : ''))

    // 1. Run the official aiter tuner -> tuned CSV diff captured as a patch, then revert to clean.
    const impl = await agent(`
You are running the OFFICIAL aiter autotuner for a GEMM / fused-MoE op to produce a faster tuned config.
PIPELINE_MODE — fully autonomous, no EnterPlanMode/AskUserQuestion. Report MEASURED results only.

Candidate: ${tune.slug}
  Op / block: ${tune.block_name} (${tune.target_op})
  kernel_type: ${tune.kernel_type}        (gemm_tune | moe_tune)
  Workload shape (gate-corrected, padded token bucket): ${tuneShape || '(infer a decode-relevant shape and STATE it)'}
  Tuner script: ${tune.tune_script || '(locate csrc/<op>/<op>_tune.py for this op)'}
  Tuned CSV (op reads at runtime): ${tune.config_csv || '(locate aiter/configs/*_tuned_*.csv)'}
  Untuned CSV (shapes to tune): ${tune.untuned_csv || '(the matching *_untuned_*.csv)'}
  GPU pin: HIP_VISIBLE_DEVICES=${gpus}

AITER tuner contract:
- AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")  (fallback /sgl-workspace/aiter)
- gemm_tune/moe_tune: DO NOT hand-edit kernel templates. Run the official tuner (csrc/<op>/<op>_tune.py via
  base_tuner/mp_tuner; see docs/autotuning_pipeline.md). It benchmarks candidates and writes the best config to
  aiter/configs/<op>_tuned_*.csv. The op reads that CSV at runtime — NO .so rebuild needed.
- This change touches ONLY aiter config CSV(s). Do NOT modify any sglang file.

STEPS:
1. Detect AITER_ROOT. Make sure 'git -C $AITER_ROOT status --short' is clean for the tuned CSV
   (this committed CSV is your baseline). If dirty from a prior run: git -C $AITER_ROOT checkout -- <tuned_csv>.
2. Ensure the WORKLOAD shape row exists in the untuned CSV (add it if missing).
3. Run the official tuner PINNED to HIP_VISIBLE_DEVICES=${gpus}. It must benchmark candidates and WRITE updated
   rows into the tuned CSV. Confirm it actually changed: git -C $AITER_ROOT diff --stat -- <tuned_csv>.
4. Capture the change as a patch BEFORE reverting:
     git -C $AITER_ROOT diff -- <tuned_csv> > ${patchFile}
   Count the rows changed.
5. REVERT the shared aiter install to clean baseline so the validation step controls baseline/after itself:
     git -C $AITER_ROOT checkout -- <tuned_csv>
   Confirm 'git -C $AITER_ROOT status' is clean for the CSV.
6. Return: slug, status ("pass" only if the tuner ran AND ${patchFile} is non-empty, else "fail"),
   patch_path="${patchFile}", config_csv (repo-relative under aiter/configs/), rows_changed,
   tuner_summary (shapes/kernels improved + any TFLOPS/us numbers from tuner output), error.
`, { label: `tune:${tune.slug}`, phase: 'Tune (aiter)', schema: TUNE_IMPL_SCHEMA, effort: 'high' })

    if (!impl || impl.status !== 'pass' || !impl.patch_path) {
      skipped.push({ ...tune, impl })
      log(`SKIP (tune): ${tune.slug} — ${impl?.error || 'tuner produced no change'}`)
      continue
    }
    log(`PASS (tune): ${tune.slug} — ${impl.rows_changed ?? '?'} CSV row(s) changed`)

    // 2. Validate E2E (serial; the patch carries the change).
    const item = { ...tune, patch_path: impl.patch_path, config_csv: impl.config_csv || tune.config_csv }
    const v = await validateWithRetries(item, gpus, port)
    if (!v.passed) {
      failedValidation.push({ ...tune, validation: v.r })
      continue
    }
    log(`${v.shipMode === 'open' ? 'PASS (open)' : 'PASS (DRAFT — perf below gate)'}: ${tune.slug} `
      + `(accuracy=${v.r.accuracy}, kernelΔ=${v.imprText})`)

    // 3. Ship as an aiter PR (or report+patch when no remote configured).
    const isDraft = v.shipMode === 'draft'
    const pr = await agent(`
You are shipping an aiter tuned-config improvement to ${AITER_REPO}. PIPELINE_MODE — fully autonomous.
Do NOT dump raw logs; COMPOSE a clean PR body from the data. NO Co-Authored-By trailer anywhere.

== Candidate ==
slug: ${tune.slug}
op: ${tune.target_op}  (${tune.kernel_type})
workload shape: ${tuneShape || 'n/a'}
tuned CSV (repo-relative): ${impl.config_csv}
patch (tuned rows): ${impl.patch_path}
rows changed: ${impl.rows_changed ?? '?'}
tuner summary: ${impl.tuner_summary || ''}
perf judgement: ${v.perfJudgement}

== Validation data — DISTILL into tables, DO NOT paste verbatim ==
${JSON.stringify(v.r, null, 2)}

== aiter ship config ==
AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")  (fallback /sgl-workspace/aiter)
base: ${AITER_BASE}
repo: ${AITER_REPO}
push remote: ${AITER_REMOTE || '(NONE — patch-only fallback)'}
gh env: ${AITER_GH_ENV}
draft: ${isDraft}

STEPS:
1. cd $AITER_ROOT. Ensure a clean tree on ${AITER_BASE} (git fetch; git checkout ${AITER_BASE}; git checkout -- .).
2. Create branch tune/${tune.slug}. Apply the patch: git apply ${impl.patch_path}. Confirm ONLY the tuned CSV
   changed (git status; git diff --stat).
3. Commit. Title MUST follow the aiter convention (see https://github.com/ROCm/aiter/commit/46e6c92b3eb33f64823aaa1ff39a14586b059ef5):
     "${isDraft ? '[Chore]' : '[Perf]'} Tune ${tune.target_op} for ${tuneShape || 'target shapes'}"
   Body: 1-3 lines — which shapes were tuned, the measured kernel speedup, and the arch (gfx/MI). No trailers.
4. PUSH / PR — if a push remote is configured (${AITER_REMOTE ? 'YES' : 'NO'}):
   - push branch tune/${tune.slug} to ${AITER_REMOTE}; then
     ${AITER_GH_ENV} gh pr create --repo ${AITER_REPO} --base ${AITER_BASE} ${isDraft ? '--draft ' : ''}--title "<title>" --body-file <file>
     Compose the body: Motivation (slow GEMM/MoE shape), What changed (tuned CSV rows / kernel substitution),
     Accuracy (GSM8K score/threshold table), Speed (kernel before/after us + TFLOPS, E2E delta). Repo-relative paths only.
   - return pr_url and mode="${isDraft ? 'draft' : 'open'}".
   ELSE (no remote configured): DO NOT push. Keep the commit on branch tune/${tune.slug} and the patch at
     ${impl.patch_path}. Set pr_url="" and mode="patch_only".
5. After shipping, return to ${AITER_BASE} and ensure a CLEAN tree (git checkout ${AITER_BASE}; git checkout -- .)
   so the next tune candidate is isolated from the shared aiter install.
6. Return: slug, repo="aiter", commit_hash, pr_url, pr_title, mode, status.
`, { label: `ship-aiter:${tune.slug}`, phase: 'Tune (aiter)', schema: PR_RESULT_SCHEMA })

    if (pr && pr.status === 'pass') {
      const mode = pr.mode || (isDraft ? 'draft' : 'open')
      shipped.push({ slug: tune.slug, repo: 'aiter', pr_url: pr.pr_url || '', commit: pr.commit_hash, mode, judgement: v.perfJudgement })
      log(mode === 'patch_only'
        ? `Patch saved (aiter, no remote): ${tune.slug} → ${impl.patch_path}`
        : `PR created (${mode}, aiter): ${tune.slug} → ${pr.pr_url}`)
    } else {
      shipFailed.push({ slug: tune.slug, repo: 'aiter', error: pr?.error })
      log(`Ship failed (aiter): ${tune.slug} — ${pr?.error || 'unknown'}`)
    }
  }
}

// ── Run tracks ───────────────────────────────────────────────────────────────
// sglang fusions run first (parallel) while the shared aiter install is at clean baseline; then the
// aiter tune track runs serially, isolating each candidate's CSV change.

if (fusions.length) {
  await runSglangTrack()
} else {
  log('No sglang fusion candidates — skipping sglang track.')
}

if (tunes.length) {
  await runAiterTuneTrack()
} else {
  log('No aiter tune candidates — skipping aiter tune track.')
}

// ── Summary ───────────────────────────────────────────────────────────────────

log('=== Pipeline Complete ===')
log(`Shipped: ${shipped.length} (sglang ${shipped.filter(s => s.repo === 'sglang').length}, aiter ${shipped.filter(s => s.repo === 'aiter').length}) | `
  + `Validate-failed: ${failedValidation.length} | Skipped: ${skipped.length} | Ship-failed: ${shipFailed.length}`)

return {
  shipped,
  failedValidation: failedValidation.map(f => ({ slug: f.slug, repo: f.repo || 'sglang', error: f.validation?.error })),
  skipped: skipped.map(f => ({ slug: f.slug, repo: f.repo || 'sglang', error: f.impl?.error })),
  shipFailed,
}
