
export const meta = {
  name: 'implement-deep',
  description: 'Deeper kernel sweep (tune or author) per target. Single: args.target_op. Multi: args.tasks = [{ target_op, workload?, gpus?, priority?, ... }]. Tasks run SEQUENTIALLY (shared editable aiter). Optional args.schedule=auto (default when len(tasks)>1): assigns HIP_VISIBLE_DEVICES from plan_gpu_slots.py free slots (--tp args.schedule_tp, default 1). Report-only; does NOT open a PR.',
  whenToUse: 'Pass args.target_op OR args.tasks[].target_op. schedule:"auto"|"off" (off when single target unless forced). continue_on_fail (default true) runs remaining tasks after recon/baseline failures.',
  phases: [
    { title: 'Schedule', detail: 'Optional GPU slot probe for per-task gpus assignment' },
    { title: 'Recon', detail: 'Per task: mode, sources, bench, baseline' },
    { title: 'Tune', detail: 'Per task: adaptive loop' },
    { title: 'Finalize', detail: 'Per task: verify, report, patch' },
  ],
}

// ── Schemas ──────────────────────────────────────────────────────────────────

const KNOB_SCHEMA = {
  type: 'object',
  properties: {
    name: { type: 'string', description: 'Tunable parameter (e.g. BLOCK_M, num_warps, tile_shape, pipeline, occupancy, csv shape row)' },
    description: { type: 'string' },
    candidate_values: { type: 'array', items: { type: 'string' }, description: 'Concrete values worth trying' },
  },
  required: ['name'],
}

const EXP_PLAN_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string', description: 'short_snake_case id for this experiment' },
    description: { type: 'string', description: 'One concrete, measurable change to try' },
  },
  required: ['id', 'description'],
}

const RECON_SCHEMA = {
  type: 'object',
  properties: {
    target_op: { type: 'string' },
    mode: { type: 'string', enum: ['tune', 'author'], description: 'tune=optimize existing kernel; author=write a new fused kernel that does not exist yet' },
    kernel_type: { type: 'string', enum: ['ck', 'triton', 'flydsl', 'gemm_tune', 'moe_tune'], description: 'For author mode, the kernel_type of the NEW kernel to write (usually triton or ck)' },
    strategy: { type: 'string', enum: ['tuning_script', 'param_search', 'author_kernel'], description: 'tuning_script = run official csrc/*_tune.py; param_search = hand-edit existing kernel knobs + rebuild; author_kernel = implement a brand-new fused kernel' },
    replaced_kernels: { type: 'array', items: { type: 'string' }, description: 'AUTHOR mode: the unfused kernels the new kernel replaces (baseline = sum of their times)' },
    reference_impl: { type: 'string', description: 'AUTHOR mode: the reference to match — the CUDA fused kernel name and/or an existing aiter kernel to model the new one on' },
    sources: { type: 'array', items: { type: 'string' }, description: 'tune: source file(s) a variant edits / the tune script. author: the NEW source file(s) to create + the equivalence-test harness file' },
    so_modules: { type: 'array', items: { type: 'string' }, description: 'JIT module base name(s) whose aiter/jit/<name>.so must be deleted to rebuild after a source edit (empty for triton/flydsl/csv-only)' },
    config_csv: { type: ['string', 'null'], description: 'aiter/configs tuned CSV the op reads at runtime, if tuning-based' },
    op_test: { type: 'string', description: 'tune: the existing correctness unit test command. author: the command running the NEW numerical-equivalence harness (fused vs unfused reference, allclose).' },
    bench: { type: 'string', description: 'Exact shell command that microbenchmarks the target and prints a latency; recon must confirm it runs (author: benches the new fused path)' },
    rebuild: { type: 'string', description: 'Exact rebuild recipe (tune: which .so to delete / tuner to run; author: how to compile the new kernel — full setup.py develop if a new CK module, none for new triton)' },
    baseline: {
      type: 'object',
      properties: {
        correct: { type: 'boolean' },
        latency_us: { type: ['number', 'null'], description: 'Baseline microbench latency (us) for the target shape; null if not parseable' },
        metric: { type: 'string', description: 'What latency_us measures (e.g. "decode us @ M=128,N=K=8192")' },
        notes: { type: 'string' },
      },
      required: ['correct'],
    },
    knobs: { type: 'array', items: KNOB_SCHEMA },
    experiments: { type: 'array', items: EXP_PLAN_SCHEMA, description: 'Ordered initial plan, most-promising first' },
    notes: { type: 'string' },
  },
  required: ['target_op', 'mode', 'kernel_type', 'strategy', 'sources', 'op_test', 'bench', 'rebuild', 'baseline', 'experiments'],
}

const EXP_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    applied: { type: 'boolean', description: 'Was the variant actually applied + built' },
    change_summary: { type: 'string', description: 'What concretely changed (param old→new / tune args / csv row)' },
    diff_files: { type: 'array', items: { type: 'string' } },
    rebuilt: { type: 'boolean', description: 'Was the relevant .so deleted + recompiled (or tune script run) successfully' },
    correct: { type: ['boolean', 'null'], description: 'op_test result; null if not reached' },
    latency_us: { type: ['number', 'null'], description: 'Measured microbench latency (us) from THIS run output, not estimated' },
    improvement_pct: { type: ['number', 'null'], description: '(baseline-after)/baseline*100; positive = faster' },
    error: { type: 'string' },
    observations: { type: 'string', description: 'What the result suggests for the next variant' },
  },
  required: ['id', 'applied', 'rebuilt'],
}

const FINAL_SCHEMA = {
  type: 'object',
  properties: {
    best_id: { type: ['string', 'null'] },
    baseline_latency_us: { type: ['number', 'null'] },
    best_latency_us: { type: ['number', 'null'] },
    improvement_pct: { type: ['number', 'null'] },
    correct: { type: 'boolean' },
    repo_state: { type: 'string', enum: ['best_applied', 'reverted_win', 'restored_baseline'], description: 'best_applied = winning diff left applied; reverted_win = win saved as patch then reverted for isolation; restored_baseline = no win, reverted clean' },
    diff: { type: ['string', 'null'], description: 'git diff of the winning change (repo-relative); captured even when reverted' },
    patch_path: { type: ['string', 'null'], description: 'file the winning diff was saved to (re-appliable later), or null if no win' },
    report_path: { type: 'string' },
    summary: { type: 'string' },
    ship_ready: { type: 'boolean', description: 'true iff correct AND improvement_pct >= perf gate AND repo_state == best_applied' },
  },
  required: ['correct', 'repo_state', 'report_path', 'ship_ready'],
}

const GPU_SCHEDULER_SCHEMA = {
  type: 'object',
  properties: {
    max_parallel: { type: ['number', 'null'], description: 'From JSON.max_parallel or slots.length' },
    slots: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          slot_id: { type: 'number' },
          gpus: { type: 'string', description: 'HIP_VISIBLE_DEVICES value e.g. 4 or 4,5' },
        },
        required: ['gpus'],
      },
    },
    error_note: { type: 'string' },
  },
  required: ['slots'],
}

const GPU_PLAN_CMD = (tp, portBase) =>
  `python3 /home/yichiche/agent-box/skills/kernel-fusion-pipeline/scripts/plan_gpu_slots.py --tp ${tp} --port-base ${portBase} --json`

function makeRebuildGuide(gpus, aiterRootArg) {
  return `
=== aiter rebuild contract (follow exactly; reference aiter/jit/core.py) ===
- aiter is an EDITABLE 'python3 setup.py develop' install: the imported package IS the source tree at
  AITER_ROOT/aiter. Detect AITER_ROOT once:
    AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")
  (fallback: /sgl-workspace/aiter, then /home/yichiche/aiter)${aiterRootArg ? `\n  Use AITER_ROOT=${aiterRootArg} (provided).` : ''}
- JIT model: each C++/HIP (CK) module compiles to AITER_ROOT/aiter/jit/<module>.so on first use.
- REBUILD AFTER EDITING CK/HIP SOURCE = delete ONLY that module's .so, then re-import/run so JIT recompiles:
    rm -f AITER_ROOT/aiter/jit/<module>.so       # this is exactly rm_module() in core.py
  Equivalent official env (deletes .so for you on next build): AITER_REBUILD=2 <cmd>
  (AITER_REBUILD=1 also clears the build dir bd_dir/<module> — use only if a stale build is suspected).
- DO NOT delete unrelated .so files and DO NOT run a full 'python3 setup.py develop' rebuild unless the set
  of compiled modules actually changed (new source file / new instances). Deleting only the relevant .so is
  the fast path and is what 'rebuild aiter' means here.
- TRITON kernels: no .so. Triton re-JITs automatically when the source changes; if a stale compile is
  suspected clear the triton cache (~/.triton/cache) for that kernel.
- FLYDSL kernels: no .so; cached under AITER_ROOT/aiter/jit/flydsl_cache/<hash>. If a source edit is not
  taking effect, remove the specific stale launch_<kernel>_<hash> dir for this kernel and re-run.
- gemm_tune / moe_tune: DO NOT hand-edit kernel templates. Run the official tuner
  (csrc/<op>/<op>_tune.py via base_tuner/mp_tuner) which benchmarks candidates and writes the best config to
  AITER_ROOT/aiter/configs/<op>_tuned_*.csv. The op reads that CSV at runtime — usually no .so deletion
  needed (delete the op .so only if the kernel instance set changed). See docs/autotuning_pipeline.md.
${gpus ? `- GPU PINNING: prefix EVERY op_test / bench / tuner command with HIP_VISIBLE_DEVICES=${gpus} so this
  sweep stays on its assigned GPU(s) and does not disturb other work on the box.` : `- GPU: no pin set (gpus empty); tuners may use multiple GPUs. Pass per-task gpus or global args.gpus to restrict.`}
`
}

async function fetchGpuSlots(scheduleTp, portBase) {
  const cmd = GPU_PLAN_CMD(scheduleTp, portBase)
  return agent(
    `You are scheduling GPU slots for sequential implement-deep tasks. PIPELINE_MODE — no questions.

Run this EXACT command in bash and parse stdout as JSON (ignore stderr unless the command failed):
${cmd}

Return:
- slots: array of { slot_id: number from JSON.slots[i].slot_id if present, else i, gpus: JSON.slots[i].gpus string }
- max_parallel: JSON.max_parallel if present, else slots.length
- error_note: empty string if OK; if command failed or JSON invalid, slots=[] and explain briefly

The script may write ~/.kernel-fusion-pipeline/state.json — that is expected.`,
    { label: 'gpu-schedule', phase: 'Schedule', schema: GPU_SCHEDULER_SCHEMA, effort: 'low' },
  )
}

function normalizeTaskList(args) {
  const out = []
  if (Array.isArray(args?.tasks) && args.tasks.length) {
    for (const t of args.tasks) {
      if (t && typeof t.target_op === 'string' && t.target_op.trim())
        out.push(t)
      else log('WARN: skip task without target_op')
    }
  }
  if (!out.length && args?.target_op && String(args.target_op).trim())
    out.push({ target_op: String(args.target_op).trim() })
  return out
}

function mergeTaskGlobals(raw, g, taskIdx, totalTasks, slotGpus) {
  const pin = (raw.gpus && String(raw.gpus).trim()) || (slotGpus && String(slotGpus).trim()) || (g.gpus && String(g.gpus).trim()) || ''
  const slug = raw.target_op.replace(/[^a-z0-9_]+/gi, '_').slice(0, 45).replace(/_+$/, '')
  const fileTag = totalTasks <= 1 ? slug : `${taskIdx}_${slug}`
  return {
    targetOp: raw.target_op.trim(),
    taskId: raw.id || `t${taskIdx}`,
    taskIndex: taskIdx,
    modeHint: raw.mode ?? g.mode,
    kernelTypeHint: raw.kernel_type ?? g.kernelType,
    workload: raw.workload ?? g.workload,
    benchCmdHint: raw.bench_cmd ?? g.benchCmd,
    aiterRootArg: raw.aiter_root ?? g.aiterRoot,
    perfGatePct: raw.perf_gate_pct ?? g.perfGatePct,
    maxRounds: raw.max_rounds ?? g.maxRounds,
    maxNoImprove: raw.max_no_improve ?? g.maxNoImprove,
    keepBest: raw.keep_best ?? g.keepBest,
    minBudgetReserve: raw.min_budget_reserve ?? g.minBudgetReserve,
    gpus: pin,
    reportPath: `${g.outDir}/deep_${fileTag}.md`,
    patchPath: `${g.outDir}/deep_${fileTag}.patch`,
  }
}

async function runSingleImplementDeep(spec) {
  const {
    targetOp, modeHint, kernelTypeHint, workload, benchCmdHint, aiterRootArg,
    perfGatePct, maxRounds, maxNoImprove, keepBest, minBudgetReserve, gpus,
    reportPath, patchPath,
  } = spec

  const REBUILD_GUIDE = makeRebuildGuide(gpus, aiterRootArg)

  log(`Deep-tune target=${targetOp} kernel_type=${kernelTypeHint} workload="${workload || 'n/a'}" gpus="${gpus || 'unset'}"`)
  log(`perf_gate=${perfGatePct}% max_rounds=${maxRounds} converge_after=${maxNoImprove} keep_best=${keepBest}`)

  phase('Recon')

  const recon = await agent(`
You are the RECON step of a single-kernel deep-tuning sweep for aiter on ROCm/MI355. PIPELINE_MODE — no
EnterPlanMode, no AskUserQuestion. Be exact and run commands to verify everything you report.

TARGET OP / OPTIMIZATION: ${targetOp}
MODE HINT: ${modeHint}   (if "auto", decide tune vs author yourself per STEP 0)
KERNEL TYPE HINT: ${kernelTypeHint}   (if "auto", classify it yourself)
WORKLOAD / SHAPES: ${workload || '(not given — pick a representative decode-relevant shape and state it)'}
${benchCmdHint ? `BENCH COMMAND HINT (verify it works): ${benchCmdHint}` : ''}

${REBUILD_GUIDE}

DO THE FOLLOWING:

0. DETECT MODE (set "mode"):
   - "tune"   → the target is an EXISTING single aiter kernel/op that is too slow; you will optimize it.
   - "author" → the target is a FUSION/new kernel that does NOT exist yet (the description names multiple
     separate kernels to fuse, or a CUDA fused kernel with no HIP/aiter equivalent). Verify there is no existing
     fused op under aiter/ops/ that already does it (grep). If none exists → author.
   Honor MODE HINT unless evidence clearly contradicts it (explain in notes if you override).

   IF mode == "author":
     - Set strategy="author_kernel". kernel_type = the type of the NEW kernel (usually "triton"; "ck" only if a
       Triton kernel genuinely cannot express it).
     - replaced_kernels[] = the unfused kernels it replaces; baseline.latency_us = the SUM of their measured
       times for the workload shape (measure them), baseline.metric = "sum of <k1>+<k2>+... us @ <shape>".
     - reference_impl = the CUDA fused kernel name to match and/or the closest existing aiter kernel to model on.
     - sources[] = the NEW source file path(s) to create (pick the right aiter/ops/... or sglang location) PLUS
       the equivalence-test harness file you will create.
     - op_test = the command running a NEW numerical-equivalence harness you describe: it builds inputs, runs the
       UNFUSED reference path AND the new fused path, asserts torch.allclose, and is the correctness gate.
     - bench = the command timing the new fused path on the workload shape.
     - experiments[] = an author plan: experiment 1 = "implement a first CORRECT version", then optimization
       passes (tiling, vectorization, fewer passes, etc.).
     - Do NOT write the kernel in this recon step — only design it. Then SKIP the tune-specific steps 1-2 below
       (still fill kernel_type via the new-kernel type) and continue to steps 3-7 framed for the new kernel.

   IF mode == "tune": continue with steps 1-7 as written.

1. (tune) Locate the target op in the installed aiter (AITER_ROOT as above). Classify kernel_type into exactly one of
   ck | triton | flydsl | gemm_tune | moe_tune:
     - gemm_tune  → a GEMM with an official csrc/<...>/*_tune.py + aiter/configs/*_tuned_gemm.csv (a8w8, a4w4,
       a16w16, bpreshuffle, blockscale, batched_gemm, ...).
     - moe_tune   → fused MoE with csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py + configs/tuned_fmoe.csv.
     - ck         → a CK C++/HIP kernel under csrc/ compiled to a aiter/jit/<module>.so (param_search).
     - triton     → a kernel under aiter/ops/triton/ (autotune configs / BLOCK sizes / num_warps/num_stages).
     - flydsl     → a kernel under aiter/ops/flydsl/ (+ flydsl_cache / a *_tuned.csv).
2. Set strategy: 'tuning_script' for gemm_tune/moe_tune (the official tuner does the search), else 'param_search'.
3. Record sources[] (the file(s) a variant edits, or the tune script to run, repo-relative), so_modules[] (the
   aiter/jit/<name> base name(s) to delete to rebuild after a CK source edit — empty for triton/flydsl/csv-only),
   config_csv (the tuned CSV the op reads, or null), and a precise rebuild recipe for this kernel_type.
4. Find the correctness test: the matching op_tests/test_*.py. Give the EXACT command (op_test). Run it once to
   confirm the baseline is correct=true.
5. Find/confirm the microbench: prefer op_tests/op_benchmarks/{hip,triton}/bench_*.py for this op, else the
   test's --bench mode, else the tune script's pre-tune perf run. Give the EXACT command (bench). RUN IT and
   parse a single representative latency_us for the WORKLOAD shape — this is baseline.latency_us. State exactly
   what it measures in baseline.metric. If you truly cannot parse a number, set latency_us=null and explain.
6. Enumerate the real tunable knobs for THIS kernel_type (param_search: tile/BLOCK sizes, num_warps, num_stages,
   pipeline, occupancy, vectorization, splitK, kpack, waves-per-eu; tuning_script: which untuned-CSV shape rows
   to add, tuner args, candidate kernel instances). Give concrete candidate_values.
7. Propose an ORDERED initial experiment plan (3–6 items, most-promising first). Each experiment = ONE concrete,
   measurable change. For tuning_script kernels the first experiment is usually "ensure the workload shape is in
   the untuned CSV and run the official tuner".

Do NOT edit any source or rebuild anything in this step. Return the full schema.
`, { label: `recon:${spec.taskId}`, phase: 'Recon', schema: RECON_SCHEMA, effort: 'high' })

  if (!recon || !recon.bench || !recon.op_test) {
    log('Recon failed — could not locate the op / test / bench.')
    return { status: 'recon_failed', target_op: targetOp, recon }
  }

  const baselineLatency = (recon.baseline && typeof recon.baseline.latency_us === 'number')
    ? recon.baseline.latency_us : null
  log(`Recon: kernel_type=${recon.kernel_type} strategy=${recon.strategy} baseline=${baselineLatency ?? 'unparsed'}us correct=${recon.baseline?.correct}`)
  log(`Sources: ${(recon.sources || []).join(', ') || 'n/a'} | .so: ${(recon.so_modules || []).join(', ') || 'none'} | csv: ${recon.config_csv || 'none'}`)
  log(`Initial plan: ${(recon.experiments || []).map(e => e.id).join(', ')}`)

  if (recon.baseline && recon.baseline.correct === false) {
    log('Baseline op_test is already failing — aborting before tuning a broken kernel.')
    return { status: 'baseline_incorrect', target_op: targetOp, recon }
  }

  phase('Tune')

  const plan = (recon.experiments || []).slice()
  const history = []
  let best = null
  let noImprove = 0

  function reconContext() {
    return JSON.stringify({
      target_op: recon.target_op, mode: recon.mode, kernel_type: recon.kernel_type, strategy: recon.strategy,
      replaced_kernels: recon.replaced_kernels, reference_impl: recon.reference_impl,
      sources: recon.sources, so_modules: recon.so_modules, config_csv: recon.config_csv,
      op_test: recon.op_test, bench: recon.bench, rebuild: recon.rebuild,
      knobs: recon.knobs, baseline: recon.baseline,
    }, null, 2)
  }

  const isAuthor = recon.mode === 'author'
  const roundIntro = isAuthor
    ? `You are ONE iteration of an AUTHORING loop for a NEW fused aiter kernel on ROCm/MI355. You build or refine
the work-in-progress new kernel toward CORRECT-then-FASTER. Report MEASURED numbers only (never estimate).`
    : `You are ONE round of a single-kernel TUNING loop for aiter on ROCm/MI355. You apply EXACTLY ONE variant to
an existing kernel, rebuild, gate on correctness, then microbench. Report MEASURED numbers only (never estimate).`
  const roundStep1 = isAuthor
    ? `1. DO NOT restore baseline — KEEP the current work-in-progress new kernel and build on it. (Round 1: create
   the first version of the new kernel + its equivalence harness per the plan / reference_impl.) If a previous
   round left the build or harness broken, FIX that first before optimizing.`
    : `1. RESTORE BASELINE first so this experiment is measured cleanly and independently:
     git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '<source files>'}
     git -C <AITER_ROOT> stash --include-untracked 2>/dev/null || true   # only if other edits linger
   and delete any .so you will rebuild (so_modules above) so the next build is baseline+your edit only.`
  const roundStep2 = isAuthor
    ? `2. IMPLEMENT/REFINE exactly one coherent improvement to the new kernel (round 1: a first correct version;
   later rounds: one optimization — better tiling, vectorization, fewer memory passes, occupancy). Describe it precisely.`
    : `2. APPLY exactly one concrete change (param_search: edit the source knob; tuning_script: edit the untuned CSV
   shape row / run the official tuner with the right args). Keep it minimal and described precisely.`
  const correctnessLine = isAuthor
    ? `4. CORRECTNESS GATE: run the equivalence harness (op_test) — it must assert the new fused path matches the
   UNFUSED reference (torch.allclose). If it fails or does not compile, set correct=false, DO NOT benchmark, and
   report what broke so the next round can fix it.`
    : `4. CORRECTNESS GATE: run the op_test command. If it fails, set correct=false, DO NOT benchmark, explain, and
   stop — a faster-but-wrong kernel is not a result.`

  for (let round = 0; round < maxRounds; round++) {
    if (noImprove >= maxNoImprove) { log(`Converged: ${noImprove} rounds with no new best.`); break }
    if (budget.total && budget.remaining() < minBudgetReserve) {
      log(`Stopping: token budget low (${Math.round(budget.remaining() / 1000)}k < ${Math.round(minBudgetReserve / 1000)}k reserve).`)
      break
    }

    const nextPlanned = plan[round]
    const bestLine = best
      ? `Current BEST: id=${best.id} latency=${best.latency_us}us (${best.improvement_pct?.toFixed(1)}% faster than baseline).`
      : 'No winning variant yet.'

    log(`Round ${round + 1}/${maxRounds}${nextPlanned ? ` — ${nextPlanned.id}` : ' — adaptive'}`)

    const result = await agent(`
${roundIntro} PIPELINE_MODE — fully autonomous, no EnterPlanMode/AskUserQuestion.

${REBUILD_GUIDE}

RECON (ground truth for this op):
${reconContext()}

BASELINE latency (us): ${baselineLatency ?? 'unparsed — report absolute latency_us and leave improvement_pct null'}
${bestLine}

HISTORY of prior experiments (id → outcome):
${history.length ? JSON.stringify(history.map(h => ({ id: h.id, correct: h.correct, latency_us: h.latency_us, improvement_pct: h.improvement_pct, change: h.change_summary, note: h.observations })), null, 2) : '(none yet)'}

THIS ROUND:
${nextPlanned
  ? `Run the planned experiment id="${nextPlanned.id}": ${nextPlanned.description}`
  : `The initial plan is exhausted. Using the history above, pick the SINGLE most promising NEW variant
     (extrapolate from what helped / hurt) and give it a fresh snake_case id.`}

PROCEDURE (do every step; stop early only on a hard failure and report it):
${roundStep1}
${roundStep2}
3. REBUILD per the contract above (tune: delete ONLY the relevant aiter/jit/<module>.so, or run the tuner;
   author: compile the new kernel — new triton needs no .so, a new CK module needs setup.py develop). Confirm
   the rebuild/recompile actually happened (set rebuilt).
${correctnessLine}
5. MICROBENCH: run the bench command for the SAME workload shape recon used. Parse latency_us from the actual
   output. Compute improvement_pct = (baseline - after)/baseline*100 (positive = faster) if baseline is known.
   ${isAuthor ? 'AUTHOR: baseline is the SUMMED time of the replaced unfused kernels; the new fused kernel must beat it.' : ''}
6. Return the full result schema, including diff_files and an observations note that informs the next variant.

Be honest: report regressions and failures as-is. Do not leave the repo in a half-built state — if you abort,
restore the baseline source + remove the half-built .so${isAuthor ? ' / delete the broken new source file' : ''}.
`, { label: `tune:${spec.taskId}:${nextPlanned?.id || `round${round + 1}`}`, phase: 'Tune', schema: EXP_RESULT_SCHEMA })

    if (!result) { log(`Round ${round + 1}: agent returned null — skipping.`); noImprove++; continue }
    history.push(result)

    const ok = result.correct === true
    const lat = typeof result.latency_us === 'number' ? result.latency_us : null
    const impr = typeof result.improvement_pct === 'number'
      ? result.improvement_pct
      : (ok && lat !== null && baselineLatency ? (baselineLatency - lat) / baselineLatency * 100 : null)

    if (ok && lat !== null && (best === null || lat < best.latency_us)) {
      best = { ...result, latency_us: lat, improvement_pct: impr }
      noImprove = 0
      const vsBase = impr !== null ? `${impr.toFixed(1)}% vs baseline` : 'n/a'
      log(`  fastest correct so far: ${result.id} → ${lat}us (${vsBase})`)
    } else {
      noImprove++
      const why = !ok ? `incorrect (${result.error || 'op_test failed'})` : (lat === null ? 'no latency parsed' : `${lat}us (not faster than current best ${best?.latency_us ?? '?'}us)`)
      log(`  no progress: ${result.id} — ${why}`)
    }
  }

  if (!best) {
    log('No CORRECT variant produced at all. Restoring baseline.')
    await agent(`
Restore the aiter install to a clean baseline after a deep-tune sweep that found no winning variant.
${REBUILD_GUIDE}
Run: git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '.'} ; remove any rebuilt .so for
modules ${JSON.stringify(recon.so_modules || [])} so the next use recompiles the committed source; revert any
edited untuned CSV (${recon.config_csv || 'n/a'}) if you changed it.${isAuthor ? ` AUTHOR mode: also DELETE the new
(untracked) kernel/harness files you created: ${JSON.stringify(recon.sources || [])} (rm -f / git clean -f).` : ''}
Confirm 'git -C <AITER_ROOT> status' is clean for these files. Return a one-line confirmation.`, { label: `restore-baseline:${spec.taskId}`, phase: 'Tune' })

    return {
      status: 'no_win', target_op: targetOp, kernel_type: recon.kernel_type,
      baseline_latency_us: baselineLatency, rounds: history.length,
      history: history.map(h => ({ id: h.id, correct: h.correct, latency_us: h.latency_us, improvement_pct: h.improvement_pct, note: h.observations })),
      recon,
    }
  }

  phase('Finalize')

  const final = await agent(`
You are the FINALIZE step of a single-kernel deep-tuning sweep. PIPELINE_MODE — autonomous. Independently
re-verify the best variant and either leave it applied for shipping or restore the baseline.

${REBUILD_GUIDE}

RECON:
${reconContext()}

BASELINE latency (us): ${baselineLatency ?? 'unparsed'}
PERF GATE: a variant is ship-ready only if correct AND improvement_pct >= ${perfGatePct}%.
keep_best=${keepBest}. If false (isolation/sweep mode): SAVE the winning diff as a patch, then REVERT to a
clean baseline so the next operation is measured without pollution — the patch is the deliverable, not a left-applied diff.

WINNING VARIANT to reproduce:
${JSON.stringify({ id: best.id, change_summary: best.change_summary, diff_files: best.diff_files, latency_us: best.latency_us, improvement_pct: best.improvement_pct }, null, 2)}

MODE: ${recon.mode}${isAuthor ? ' (author — the winning variant IS the new fused kernel + its harness; correctness = the allclose equivalence test; baseline = summed replaced-kernel time)' : ' (tune — winning variant is an edit/config of an existing kernel)'}

DO:
1. Restore baseline (tune: git checkout -- sources; remove relevant .so. author: clean out any WIP, then re-add
   ONLY the winning new file(s)), then re-apply ONLY the winning change.
2. Rebuild per the contract (tune: delete only the relevant .so / re-run the tuner; author: compile the new kernel). Confirm it built.
3. Run the op_test (correctness${isAuthor ? ' = allclose vs unfused reference' : ''}) AND the bench (latency) AGAIN, same workload shape.
   Recompute improvement_pct from these fresh numbers — do not trust the loop's number.
4. ALWAYS capture the winning change as a patch FIRST (before any revert):
   ${isAuthor ? 'run `git -C <AITER_ROOT> add -N <new files>` so untracked files show, then ' : ''}\`git -C <AITER_ROOT> diff\` > ${patchPath}
   and put that same diff text (repo-relative paths) into the returned "diff"; set patch_path=${patchPath}.
   THEN decide repo_state:
   - correct AND improvement_pct >= ${perfGatePct}% AND keep_best=${keepBest}==true → LEAVE applied, repo_state="best_applied", ship_ready=true.
   - correct AND improvement_pct >= ${perfGatePct}% AND keep_best==false → it IS a win, but isolation is requested:
     REVERT to a clean baseline now (so the next operation starts unpolluted), repo_state="reverted_win", ship_ready=true
     (the saved patch at ${patchPath} is the deliverable).
   - no win (incorrect or below gate) → REVERT to clean, repo_state="restored_baseline", patch_path=null, ship_ready=false.
   REVERT means: ${isAuthor ? 'delete the new untracked files + ' : ''}git checkout -- the edited sources, remove any rebuilt .so so the committed source recompiles, revert any edited CSV; confirm 'git status' is clean.
5. Write a markdown report to ${reportPath}: target op, mode, kernel_type, baseline vs best (named knob old→new),
   the op_test + bench commands, improvement %, perf-gate verdict, repo_state, patch path, and a short table of all
   tried experiments (id | change | correct | latency | Δ%). Repo-relative paths only; no /root or /tmp paths in prose.
6. Return the full schema (set baseline_latency_us=${baselineLatency ?? 'null'}).

ALL EXPERIMENTS (for the report table):
${JSON.stringify(history.map(h => ({ id: h.id, change: h.change_summary, correct: h.correct, latency_us: h.latency_us, improvement_pct: h.improvement_pct })), null, 2)}
`, { label: `finalize:${spec.taskId}`, phase: 'Finalize', schema: FINAL_SCHEMA, effort: 'high' })

  const fb = final || {}
  log('=== Deep-tune complete (single task) ===')
  log(`Best: ${best.id} | baseline=${baselineLatency ?? '?'}us → ${fb.best_latency_us ?? best.latency_us}us `
    + `(${(fb.improvement_pct ?? best.improvement_pct ?? 0).toFixed?.(1) ?? '?'}%) | gate ${perfGatePct}% | `
    + `repo=${fb.repo_state || '?'} | ship_ready=${fb.ship_ready}`)
  log(`Report: ${fb.report_path || reportPath}`)

  return {
    status: 'done',
    target_op: targetOp,
    kernel_type: recon.kernel_type,
    strategy: recon.strategy,
    baseline_latency_us: baselineLatency,
    best_id: fb.best_id || best.id,
    best_latency_us: fb.best_latency_us ?? best.latency_us,
    improvement_pct: fb.improvement_pct ?? best.improvement_pct,
    correct: fb.correct ?? true,
    repo_state: fb.repo_state,
    ship_ready: fb.ship_ready ?? false,
    diff: fb.diff || null,
    patch_path: fb.patch_path || null,
    report_path: fb.report_path || reportPath,
    summary: fb.summary,
    sources: recon.sources,
    so_modules: recon.so_modules,
    config_csv: recon.config_csv,
    rounds: history.length,
    experiments: history.map(h => ({ id: h.id, change: h.change_summary, correct: h.correct, latency_us: h.latency_us, improvement_pct: h.improvement_pct })),
  }
}

// ── Orchestrator: multi-task + optional GPU schedule ─────────────────────────

const g = {
  mode: args?.mode || 'auto',
  kernelType: args?.kernel_type || 'auto',
  workload: args?.workload || '',
  benchCmd: args?.bench_cmd || '',
  aiterRoot: args?.aiter_root || '',
  perfGatePct: args?.perf_gate_pct ?? 1.0,
  maxRounds: args?.max_rounds ?? 6,
  maxNoImprove: args?.max_no_improve ?? 2,
  keepBest: args?.keep_best ?? true,
  minBudgetReserve: args?.min_budget_reserve ?? 60000,
  gpus: args?.gpus || '',
  outDir: args?.outDir || '~/.kernel-fusion-pipeline',
  scheduleTp: args?.schedule_tp ?? 1,
  schedulePortBase: args?.schedule_port_base ?? 9100,
  continueOnFail: args?.continue_on_fail !== false,
}

let taskList = normalizeTaskList(args)
if (!taskList.length) {
  log('ERROR: pass args.target_op (string) or args.tasks: [{ target_op, workload?, gpus?, priority?, ... }]')
  return {
    error: 'Missing targets. Example: { tasks: [{ target_op: "fused_moe", workload: "decode M=4" }, ...], schedule: "auto" }',
  }
}

taskList.sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999))

const scheduleMode = args?.schedule ?? (taskList.length > 1 ? 'auto' : 'off')
let slotPlan = null
if (scheduleMode === 'auto') {
  phase('Schedule')
  slotPlan = await fetchGpuSlots(g.scheduleTp, g.schedulePortBase)
  const n = slotPlan?.slots?.length ?? 0
  log(`GPU schedule (${GPU_PLAN_CMD(g.scheduleTp, g.schedulePortBase)}): slots=${n} ${(slotPlan?.slots || []).map(s => s.gpus).join(' | ') || '(none)'} ${slotPlan?.error_note || ''}`)
  if (!n && taskList.some(t => !(t.gpus && String(t.gpus).trim())) && !g.gpus)
    log('WARN: no free GPU slots and no per-task/global gpus — recon may use any GPU')
}

const results = []
for (let taskIdx = 0; taskIdx < taskList.length; taskIdx++) {
  const raw = taskList[taskIdx]
  const slotGpus = slotPlan?.slots?.length ? slotPlan.slots[taskIdx % slotPlan.slots.length].gpus : ''
  const merged = mergeTaskGlobals(raw, g, taskIdx, taskList.length, slotGpus)
  log(`=== implement-deep task ${taskIdx + 1}/${taskList.length} (${merged.taskId}) target=${merged.targetOp} gpus=${merged.gpus || 'unset'} ===`)

  const one = await runSingleImplementDeep(merged)
  results.push({ task_id: merged.taskId, task_index: taskIdx, assigned_gpus: merged.gpus || null, ...one })

  const fatal = one?.status === 'recon_failed' || one?.status === 'baseline_incorrect'
  if (fatal && !g.continueOnFail) {
    log(`Stopping multi-task run: ${one.status} and continue_on_fail=false`)
    break
  }
}

if (taskList.length === 1)
  return results[0]

return {
  status: 'done_all',
  task_count: results.length,
  schedule: scheduleMode,
  schedule_tp: g.scheduleTp,
  gpu_slots: slotPlan?.slots ?? null,
  gpu_scheduler_error: slotPlan?.error_note || null,
  results,
}
