
export const meta = {
  name: 'implement-e2e',
  description: 'E2E-gated ALGORITHMIC kernel improvement per target. Each task is optimized by an ALGORITHM change (fuse / replace-asm / restructure — not config tuning), filtered by a MULTI-SHAPE microbench, then gated on its INDIVIDUAL end-to-end serving throughput (sglang.bench_serving total_throughput) measured in ISOLATION vs a clean baseline. Tasks run SEQUENTIALLY on a shared editable aiter; each win is saved as a patch then reverted so the next task measures clean. Report-only; does NOT open a PR.',
  whenToUse: 'Use when the goal is real e2e/throughput gain (>5% per kernel) from ALGORITHM changes, not microbench tuning. Pass args.tasks=[{ target_op, workload?, regime?, mode?, priority? }] and the e2e config (model/tp/gpus/concurrencies). Gate is per-kernel individual e2e total_throughput, NOT the cumulative sum.',
  phases: [
    { title: 'Baseline', detail: 'One clean-aiter e2e sweep (total_throughput per concurrency + accuracy)' },
    { title: 'Feasibility', detail: 'Optional (amdahl_ref): target kernel share of prefill/e2e + reachability (report-only)' },
    { title: 'Recon', detail: 'Per task: mode, sources, multi-shape microbench, baseline, + a LOCAL test that reproduces the conc=4 trace per-call latency' },
    { title: 'Develop', detail: 'Per task: algorithmic loop, gated on multi-shape microbench (geomean across bs) + allclose' },
    { title: 'E2E', detail: 'Per task: apply best, staged conc sweep (stage1 gate -> expand), measure total_throughput delta' },
    { title: 'Profile', detail: 'Optional (profile_concs): conc4/64 prefill traces base vs opt, compare target-kernel sum opt/base' },
    { title: 'Finalize', detail: 'Per task: patch + revert + report (micro geomean / staged e2e / profile / Amdahl)' },
  ],
}

// ── Schemas ──────────────────────────────────────────────────────────────────

const KNOB_SCHEMA = {
  type: 'object',
  properties: {
    name: { type: 'string', description: 'An ALGORITHMIC lever (e.g. fuse-quant-into-gemm, replace-with-asm, remove-extra-pass, change-parallel-decomposition, vectorize-load) — not just a config knob' },
    description: { type: 'string' },
    candidate_values: { type: 'array', items: { type: 'string' }, description: 'Concrete approaches worth trying' },
  },
  required: ['name'],
}

const EXP_PLAN_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string', description: 'short_snake_case id for this experiment' },
    description: { type: 'string', description: 'One concrete, measurable ALGORITHM change to try' },
  },
  required: ['id', 'description'],
}

const SHAPE_SCHEMA = {
  type: 'object',
  properties: {
    label: { type: 'string', description: 'short id for the shape e.g. "conc16_il8k" / "bs128_decode"' },
    shape_desc: { type: 'string', description: 'the concrete problem shape this microbench measures' },
    bench: { type: 'string', description: 'EXACT shell command that microbenches THIS shape and prints a latency' },
  },
  required: ['label', 'bench'],
}

const SHAPE_LAT_SCHEMA = {
  type: 'object',
  properties: {
    label: { type: 'string' },
    latency_us: { type: ['number', 'null'] },
    correct: { type: ['boolean', 'null'] },
  },
  required: ['label'],
}

const RECON_SCHEMA = {
  type: 'object',
  properties: {
    target_op: { type: 'string' },
    mode: { type: 'string', enum: ['author', 'restructure', 'replace'], description: 'author=write a new fused kernel that does not exist; replace=swap the current kernel for a faster existing/asm implementation; restructure=change the algorithm of the existing kernel (NOT param tuning)' },
    kernel_type: { type: 'string', enum: ['ck', 'triton', 'flydsl', 'asm', 'hip'], description: 'The kernel_type being authored/restructured/replaced' },
    replaced_kernels: { type: 'array', items: { type: 'string' }, description: 'The kernel(s) this change replaces or fuses (baseline microbench = their combined time)' },
    reference_impl: { type: 'string', description: 'The reference to match — a CUDA fused kernel, an asm kernel, or an existing aiter kernel to model the new algorithm on' },
    sources: { type: 'array', items: { type: 'string' }, description: 'Source file(s) edited/created (repo-relative) PLUS the equivalence-test harness file for author/replace modes' },
    so_modules: { type: 'array', items: { type: 'string' }, description: 'aiter/jit/<name> base name(s) to delete to rebuild after a CK/HIP/asm source edit (empty for triton/flydsl)' },
    config_csv: { type: ['string', 'null'], description: 'aiter/configs CSV the op reads at runtime, if any' },
    op_test: { type: 'string', description: 'EXACT correctness command: an allclose equivalence harness (new/replaced path vs the original reference). Must be the correctness gate.' },
    repro_unit_test: { type: 'string', description: 'MANDATORY: the exact LOCAL command (repo unit test e.g. op_tests/test_*.py, or a microbench) configured to the kernel PRODUCTION shape so its measured latency REPRODUCES the kernel per-call latency seen in the conc=4 prefill trace. This is the trustworthy local proxy for production perf.' },
    repro_match_note: { type: 'string', description: 'MANDATORY: trace per-call us (from the conc=4 prefill trace / amdahl_ref) vs the local repro_unit_test baseline us, and the match verdict (within ~25% = faithful). State the production shape derived from the trace.' },
    microbench_shapes: { type: 'array', items: SHAPE_SCHEMA, description: 'MULTIPLE representative shapes — MUST include the conc=4 production shape (the one repro_unit_test validates) PLUS bs in {4,8,16,32,64} for the geomean gate. NEVER a single point. Each with an exact bench command recon has confirmed runs.' },
    baseline_microbench: { type: 'array', items: SHAPE_LAT_SCHEMA, description: 'Measured baseline latency per shape (run each microbench_shapes.bench once).' },
    e2e_share_note: { type: 'string', description: 'The target op profiled share of e2e (TTFT/ITL %) and which serving regime (prefill/decode) it dominates — sets the realistic e2e ceiling.' },
    rebuild: { type: 'string', description: 'Exact rebuild recipe for this kernel_type' },
    knobs: { type: 'array', items: KNOB_SCHEMA },
    experiments: { type: 'array', items: EXP_PLAN_SCHEMA, description: 'Ordered ALGORITHM plan, most-promising first (experiment 1 for author = first correct version)' },
    notes: { type: 'string' },
  },
  required: ['target_op', 'mode', 'kernel_type', 'sources', 'op_test', 'repro_unit_test', 'repro_match_note', 'microbench_shapes', 'rebuild', 'experiments'],
}

const EXP_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    applied: { type: 'boolean', description: 'Was the variant actually applied + built' },
    change_summary: { type: 'string', description: 'The concrete ALGORITHM change made' },
    diff_files: { type: 'array', items: { type: 'string' } },
    rebuilt: { type: 'boolean', description: 'Was the relevant module rebuilt successfully' },
    correct: { type: ['boolean', 'null'], description: 'op_test (allclose) result; null if not reached' },
    per_shape: { type: 'array', items: SHAPE_LAT_SCHEMA, description: 'MEASURED latency per microbench shape THIS run (never estimated)' },
    agg_improvement_pct: { type: ['number', 'null'], description: 'Mean across shapes of (baseline-after)/baseline*100; positive=faster' },
    min_improvement_pct: { type: ['number', 'null'], description: 'Worst single-shape improvement_pct (catches per-shape regressions)' },
    error: { type: 'string' },
    observations: { type: 'string', description: 'What the result suggests for the next variant' },
  },
  required: ['id', 'applied', 'rebuilt'],
}

const E2E_CONC_SCHEMA = {
  type: 'object',
  properties: {
    concurrency: { type: 'number' },
    total_throughput_tok_s: { type: ['number', 'null'] },
    output_throughput_tok_s: { type: ['number', 'null'] },
    median_ttft_ms: { type: ['number', 'null'] },
    median_itl_ms: { type: ['number', 'null'] },
  },
  required: ['concurrency'],
}

const E2E_SCHEMA = {
  type: 'object',
  properties: {
    label: { type: 'string' },
    ran: { type: 'boolean', description: 'Did the sweep complete and produce summary.csv' },
    accuracy: { type: ['number', 'null'], description: 'GSM8K accuracy parsed from the run' },
    acc_pass: { type: ['boolean', 'null'], description: 'accuracy >= threshold' },
    per_conc: { type: 'array', items: E2E_CONC_SCHEMA, description: 'One row per swept concurrency from summary.csv' },
    result_dir: { type: 'string' },
    error: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['label', 'ran', 'per_conc'],
}

const FINAL_SCHEMA = {
  type: 'object',
  properties: {
    best_id: { type: ['string', 'null'] },
    microbench_after: { type: 'array', items: SHAPE_LAT_SCHEMA },
    e2e_after: { type: 'array', items: E2E_CONC_SCHEMA, description: 'Per-conc total_throughput with this task applied (echo from the E2E phase)' },
    e2e_improvement_pct_median: { type: ['number', 'null'], description: 'Median across concurrencies of (after-baseline)/baseline*100 on total_throughput; positive=faster' },
    e2e_min_improvement_pct: { type: ['number', 'null'], description: 'Worst per-conc total_throughput delta % (catches regressions)' },
    accuracy_after: { type: ['number', 'null'] },
    acc_pass: { type: ['boolean', 'null'] },
    correct: { type: 'boolean', description: 'op_test allclose passed for the applied best' },
    repo_state: { type: 'string', enum: ['reverted_win', 'restored_baseline'], description: 'reverted_win = win saved as patch then reverted for isolation; restored_baseline = no win, reverted clean' },
    diff: { type: ['string', 'null'], description: 'git diff of the winning change (repo-relative); captured even though reverted' },
    patch_path: { type: ['string', 'null'] },
    report_path: { type: 'string' },
    summary: { type: 'string' },
    micro_geomean_pct: { type: ['number', 'null'], description: 'Geomean opt/base speedup across bs shapes for the winning variant' },
    profile_gate_pass: { type: ['boolean', 'null'], description: 'Profile phase: every conc target-kernel opt/base <= profile_gate (null if not run)' },
    kernel_win: { type: ['boolean', 'null'], description: 'true iff correct AND micro geomean cleared the micro gate AND (profile not run OR profile_gate_pass) — a real kernel-level win independent of the e2e Amdahl ceiling' },
    ship_ready: { type: 'boolean', description: 'true iff acc_pass AND e2e_improvement_pct_median >= perf gate AND e2e_min_improvement_pct >= -1 (no real per-conc regression)' },
  },
  required: ['correct', 'repo_state', 'report_path', 'ship_ready'],
}

const FEAS_SCHEMA = {
  type: 'object',
  properties: {
    target_kernel_us: { type: ['number', 'null'], description: 'Summed duration (us) of the target kernel(s) in the reference prefill trace' },
    prefill_total_us: { type: ['number', 'null'], description: 'Total prefill kernel time (us) in the reference trace' },
    kernel_share_pct: { type: ['number', 'null'], description: 'target_kernel_us / prefill_total_us * 100' },
    prefill_share_of_e2e_pct: { type: ['number', 'null'], description: 'Prefill share of e2e (TTFT/e2el) if known, else best estimate' },
    e2e_ceiling_pct: { type: ['number', 'null'], description: 'Max plausible e2e improvement = kernel_share% * (100% kernel removal) * prefill_share_of_e2e% — the Amdahl ceiling' },
    reachable: { type: ['boolean', 'null'], description: 'Is the e2e gate plausibly reachable given the ceiling?' },
    note: { type: 'string' },
  },
  required: ['note'],
}

const PROFILE_SCHEMA = {
  type: 'object',
  properties: {
    ran: { type: 'boolean' },
    per_conc: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          concurrency: { type: 'number' },
          base_kernel_us: { type: ['number', 'null'], description: 'Summed target-kernel us in the BASE prefill trace at this conc' },
          opt_kernel_us: { type: ['number', 'null'], description: 'Summed target-kernel us in the OPT prefill trace at this conc' },
          ratio: { type: ['number', 'null'], description: 'opt_kernel_us / base_kernel_us (<=profile_gate = pass)' },
          base_xlsx: { type: ['string', 'null'] },
          opt_xlsx: { type: ['string', 'null'] },
        },
        required: ['concurrency'],
      },
    },
    gate_pass: { type: ['boolean', 'null'], description: 'true iff every conc ratio <= profile_gate' },
    error: { type: 'string' },
    note: { type: 'string' },
  },
  required: ['ran', 'per_conc'],
}

// ── E2E primitive (perf_sweep.sh) ────────────────────────────────────────────

const PERF_SWEEP = '/home/yichiche/agent-box/skills/perf-sweep/perf_sweep.sh'

function e2eCmd(cfg, resultDir, launchServer = true) {
  const env = [
    `MODEL=${cfg.model}`,
    `TP=${cfg.tp}`,
    `GPUS=${cfg.gpus}`,
    cfg.port ? `PORT=${cfg.port}` : '',
    `INPUT_LEN=${cfg.inputLen}`,
    `OUTPUT_LEN=${cfg.outputLen}`,
    `CONCURRENCIES="${cfg.concurrencies}"`,
    `NUM_PROMPTS_MULT=${cfg.numPromptsMult}`,
    `NUM_PROMPTS_CAP=${cfg.numPromptsCap}`,
    `ACCURACY=${cfg.accuracy ? 1 : 0}`,
    `ACC_THRESHOLD=${cfg.accThreshold}`,
    `ACC_NUM_Q=${cfg.accNumQ}`,
    `GSM8K_SCRIPT=${cfg.gsm8kScript}`,
    `GSM8K_MAX_NEW_TOKENS=${cfg.gsm8kMaxNewTokens}`,
    `RESULT_DIR=${resultDir}`,
    launchServer ? '' : 'LAUNCH_SERVER=0',
  ].filter(Boolean).join(' ')
  return `${env} bash ${PERF_SWEEP}`
}

function e2eAgentPrompt(cfg, resultDir, label, contextNote) {
  const cmd = e2eCmd(cfg, resultDir, true)
  return `
You run ONE end-to-end serving benchmark for an SGLang server and return its throughput table.
PIPELINE_MODE — fully autonomous, no EnterPlanMode/AskUserQuestion.

CONTEXT: ${contextNote}

GPU SAFETY (MANDATORY): use ONLY GPUs ${cfg.gpus}. NEVER kill, pkill, or signal a sglang server you did not
start in THIS task — other users' servers may be running on other GPUs. perf_sweep launches its own server on
port ${cfg.port || 8001} and tears down only that port; do NOT broaden the kill.
CRITICAL — ${cfg.gpus} are CUDA indices (perf_sweep sets CUDA_VISIBLE_DEVICES=${cfg.gpus}). On this host CUDA index
!= rocm-smi GPU label, so DO NOT use rocm-smi to decide if they are free — it checks the wrong cards. Verify the
ASSIGNED CUDA indices with HIP: for each idx in ${cfg.gpus} run
  HIP_VISIBLE_DEVICES=<idx> python3 -c "import torch;f,t=torch.cuda.mem_get_info(0);print(idx, f/1e9)"
Only abort (ran=false) if a CUDA-index probe shows an assigned device with <250GB free, OR a FOREIGN server is
already bound to port ${cfg.port || 8001} (curl /v1/models returns a different model). Otherwise PROCEED — do not
bail based on rocm-smi labels. Never kill the occupant and never benchmark a foreign server.

The benchmark is ${PERF_SWEEP}: it launches the server ONCE (host-native python3 -m sglang.launch_server,
so it picks up the current editable aiter — your kernel edits ARE in effect), runs a GSM8K accuracy gate, then
sweeps concurrency at IL=${cfg.inputLen}/OL=${cfg.outputLen}, and writes ${resultDir}/summary.csv.

CRITICAL — this run takes 20-45 min, which EXCEEDS the 10-min Bash limit. You MUST run it in the BACKGROUND and
poll, NOT as one blocking call:
1. Launch in background, teeing all output to a log:
   mkdir -p ${resultDir}; nohup bash -c '${cmd}' > ${resultDir}/run.log 2>&1 &
   (use the Bash tool's run_in_background, or nohup ... & — either way do not block on it).
2. POLL every ~60-120s (short Bash calls): tail ${resultDir}/run.log and check for progress
   ("Server ready", "ACCURACY PASS", "conc=" lines, "DONE"). Watch ${resultDir}/server.log for
   "Traceback|OutOfMemory|HIP error|CUDA error" — if the server crashed, stop polling and report ran=false
   with the server.log tail.
3. The run is COMPLETE when ${resultDir}/summary.csv exists AND run.log shows the "DONE." line. Then proceed.
4. Budget up to ~50 min of polling; if it has not finished by then, capture logs and report ran=false.

BEFORE launching: cd /tmp (run perf_sweep from cwd /tmp — running from /home/yichiche causes an sglang
import-shadowing bug that yields an EMPTY summary.csv). Also "rm -rf ${resultDir} && mkdir -p ${resultDir}" first
so NO stale data from a prior run can be mistaken for this run's result.

THEN:
1. Parse ONLY ${resultDir}/summary.csv, with this exact command, and copy the printed rows verbatim:
     python3 -c "import csv; r=list(csv.DictReader(open('${resultDir}/summary.csv'))); [print(d['max_concurrency'],d.get('total_throughput'),d.get('median_ttft_ms'),d.get('median_itl_ms')) for d in r]"
   Emit one per_conc row per DATA line (total_throughput_tok_s = total_throughput).
2. Read ${resultDir}/accuracy.txt for the GSM8K accuracy (RAW number). NOTE: this checkpoint has a KNOWN low
   absolute accuracy (~0.69) — EXPECTED, not a failure; absolute value is not a gate (relative no-drop is checked later).
   Set acc_pass=true if gsm8k ran and a number was parsed.
3. FAIL-CLOSED (mandatory): if ${resultDir}/summary.csv is missing, header-only, or has ZERO data rows, set
   ran=false, per_conc=[], and put the tail of ${resultDir}/server.log into error. NEVER fabricate or reuse numbers
   from another directory, a prior run, a report .md, or memory — if there is no data row in THIS summary.csv, there
   is no result. Set ran=true ONLY if ≥1 real data row was parsed.
4. Make sure the server is stopped at the end (the script does this on exit unless KEEP_SERVER=1; it is not set, so it cleans up).

Return the E2E schema (label="${label}", result_dir="${resultDir}").
`
}

function e2eDelta(baseRows, afterRows) {
  // returns { median_pct, min_pct, per_conc: [{concurrency, base, after, improvement_pct}] }
  const baseMap = {}
  for (const r of (baseRows || [])) if (r && typeof r.concurrency === 'number') baseMap[r.concurrency] = r.total_throughput_tok_s
  const per = []
  for (const r of (afterRows || [])) {
    if (!r || typeof r.concurrency !== 'number') continue
    const b = baseMap[r.concurrency]
    const a = r.total_throughput_tok_s
    if (typeof b === 'number' && b > 0 && typeof a === 'number') {
      per.push({ concurrency: r.concurrency, base: b, after: a, improvement_pct: (a - b) / b * 100 })
    }
  }
  const pcts = per.map(p => p.improvement_pct).sort((x, y) => x - y)
  const median = pcts.length ? (pcts.length % 2 ? pcts[(pcts.length - 1) / 2] : (pcts[pcts.length / 2 - 1] + pcts[pcts.length / 2]) / 2) : null
  const min = pcts.length ? pcts[0] : null
  return { median_pct: median, min_pct: min, per_conc: per }
}

// Geomean speedup across bs shapes from baseline_microbench[] + result.per_shape[].
// Returns { geomean_pct, min_pct } where geomean_pct = (1 - geomean(after/base))*100 (positive = faster).
function geomeanImprovement(baselineMicro, perShape) {
  const base = {}
  for (const s of (baselineMicro || [])) if (s && s.label && typeof s.latency_us === 'number' && s.latency_us > 0) base[s.label] = s.latency_us
  const ratios = []
  const imprs = []
  for (const s of (perShape || [])) {
    if (!s || !s.label || typeof s.latency_us !== 'number' || s.latency_us <= 0) continue
    const b = base[s.label]
    if (typeof b !== 'number' || b <= 0) continue
    ratios.push(s.latency_us / b)               // after/base
    imprs.push((b - s.latency_us) / b * 100)     // improvement %
  }
  if (!ratios.length) return { geomean_pct: null, min_pct: null }
  const logSum = ratios.reduce((a, r) => a + Math.log(r), 0)
  const geoRatio = Math.exp(logSum / ratios.length)
  return { geomean_pct: (1 - geoRatio) * 100, min_pct: Math.min(...imprs) }
}

// ── Rebuild contract (shared with implement-deep) ────────────────────────────

function makeRebuildGuide(gpus, aiterRootArg) {
  return `
=== aiter rebuild contract (follow exactly; reference aiter/jit/core.py) ===
- aiter is an EDITABLE 'python3 setup.py develop' install: the imported package IS the source tree at
  AITER_ROOT/aiter. Detect AITER_ROOT once:
    AITER_ROOT=$(python3 -c "import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parent.parent)")
  (fallback: /sgl-workspace/aiter, then /home/yichiche/aiter)${aiterRootArg ? `\n  Use AITER_ROOT=${aiterRootArg} (provided).` : ''}
- JIT model: each C++/HIP/asm (CK) module compiles to AITER_ROOT/aiter/jit/<module>.so on first use.
- REBUILD AFTER EDITING CK/HIP/asm SOURCE = delete ONLY that module's .so, then re-import/run so JIT recompiles:
    rm -f AITER_ROOT/aiter/jit/<module>.so       # this is exactly rm_module() in core.py
  Equivalent official env: AITER_REBUILD=2 <cmd>  (AITER_REBUILD=1 also clears the build dir).
- DO NOT delete unrelated .so files and DO NOT run a full 'python3 setup.py develop' unless the set of compiled
  modules actually changed (new source file / new instances). Deleting only the relevant .so is the fast path.
- TRITON kernels: no .so; Triton re-JITs on source change (clear ~/.triton/cache if stale).
- FLYDSL kernels: no .so; cached under AITER_ROOT/aiter/jit/flydsl_cache/<hash>; remove the stale launch_<kernel>_<hash> dir if an edit is not taking effect.
- A NEW CK/HIP/asm module (author mode) needs setup.py develop ONCE to register it, then JIT compiles it.
${gpus ? `- GPU PINNING for microbench/op_test: prefix commands with HIP_VISIBLE_DEVICES=${gpus} so the develop loop
  stays on its assigned GPU(s). (The E2E phase uses the full tp set and is handled separately.)` : `- GPU: no microbench pin set.`}
`
}

// ── Per-task driver ──────────────────────────────────────────────────────────

async function runTaskE2E(spec, e2eCfg, baselineE2E) {
  const {
    targetOp, modeHint, regime, workload, aiterRootArg,
    perfGatePct, maxRounds, maxNoImprove, minBudgetReserve,
    microGpus, reportPath, patchPath, taskId,
    microGatePct, e2eStage1Concs, e2eStage2Concs, expandGatePct,
    profileConcs, profileKernels, profileGate, profileMode,
    amdahlRef, amdahlMinSharePct,
  } = spec

  const REBUILD_GUIDE = makeRebuildGuide(microGpus, aiterRootArg)

  log(`[${taskId}] target=${targetOp} regime=${regime || 'n/a'} mode_hint=${modeHint || 'auto'} workload="${workload || 'n/a'}"`)
  log(`[${taskId}] perf_gate=${perfGatePct}% (individual e2e) max_rounds=${maxRounds} converge_after=${maxNoImprove}`)
  if (microGatePct != null) log(`[${taskId}] micro gate: geomean opt/base across bs MUST be >= ${microGatePct}% before e2e`)
  if (e2eStage1Concs) log(`[${taskId}] staged e2e: stage1="${e2eStage1Concs}" (gate ${perfGatePct}%) -> expand "${e2eStage2Concs || '(none)'}" (gate ${expandGatePct}%)`)
  if (profileConcs) log(`[${taskId}] profile: concs="${profileConcs}" kernels=${JSON.stringify(profileKernels)} gate opt/base<=${profileGate}`)

  // ── Feasibility (Amdahl) — report-only unless amdahl_min_share_pct set ──
  let feasibility = null
  if (amdahlRef) {
    phase('Feasibility')
    feasibility = await agent(`
You are the FEASIBILITY (Amdahl) step. PIPELINE_MODE — autonomous. Decide whether optimizing ONE kernel can
plausibly move the e2e gate, BEFORE any tuning, using a reference prefill profile.

TARGET KERNEL(S): ${JSON.stringify(profileKernels && profileKernels.length ? profileKernels : [targetOp])}
REFERENCE PREFILL PROFILE (xlsx): ${amdahlRef}
E2E GATE: ${perfGatePct}% total_throughput at IL=${e2eCfg.inputLen}/OL=${e2eCfg.outputLen}.

DO (read-only; no edits):
1. Open the xlsx (pandas). Find the total prefill kernel time and the summed duration of the target kernel(s)
   (match by name substring). Compute kernel_share_pct = target/total*100.
2. Estimate prefill_share_of_e2e_pct (TTFT vs e2el for this IL/OL; if unknown, state your assumption, e.g.
   OL>>1 => decode-dominated => prefill is a small % of e2e).
3. e2e_ceiling_pct = kernel_share_pct/100 * prefill_share_of_e2e_pct  (the max e2e gain if the kernel went to ZERO).
4. reachable = (e2e_ceiling_pct >= ${perfGatePct}). Be blunt in note if the ceiling is far below the gate.
Return FEAS_SCHEMA.
`, { label: `feas:${taskId}`, phase: 'Feasibility', schema: FEAS_SCHEMA, effort: 'medium' })
    log(`[${taskId}] Amdahl: kernel_share=${feasibility?.kernel_share_pct?.toFixed?.(2) ?? '?'}% prefill_of_e2e=${feasibility?.prefill_share_of_e2e_pct?.toFixed?.(2) ?? '?'}% e2e_ceiling=${feasibility?.e2e_ceiling_pct?.toFixed?.(3) ?? '?'}% reachable=${feasibility?.reachable}`)
    if (amdahlMinSharePct != null && typeof feasibility?.kernel_share_pct === 'number' && feasibility.kernel_share_pct < amdahlMinSharePct) {
      log(`[${taskId}] STOP (feasibility): kernel share ${feasibility.kernel_share_pct.toFixed(2)}% < amdahl_min_share_pct ${amdahlMinSharePct}% — not worth optimizing for the e2e gate.`)
      return { status: 'amdahl_infeasible', target_op: targetOp, feasibility }
    }
  }

  // ── Recon ──
  phase('Recon')
  const recon = await agent(`
You are the RECON step of an E2E-GATED ALGORITHMIC kernel improvement on ROCm/MI355 (aiter, editable install).
PIPELINE_MODE — no EnterPlanMode/AskUserQuestion. Be exact; run commands to verify everything you report.

TARGET / OPTIMIZATION: ${targetOp}
SERVING REGIME: ${regime || '(infer: prefill or decode)'}
WORKLOAD / SHAPES: ${workload || '(pick representative shapes for the regime and state them)'}
MODE HINT: ${modeHint || 'auto — choose author | restructure | replace'}

${REBUILD_GUIDE}

GOAL CONTEXT: this op will ultimately be judged by its INDIVIDUAL end-to-end serving throughput delta
(sglang.bench_serving total_throughput at IL=${e2eCfg.inputLen}/OL=${e2eCfg.outputLen}). The perf gate is
${perfGatePct}% e2e. So the change must be a real ALGORITHM improvement, NOT config tuning.

DO:
0. DECIDE MODE:
   - "author"      → write a NEW fused kernel that does not exist yet (e.g. fuse quant into the following GEMM).
   - "replace"     → swap the current kernel for a faster existing/asm kernel (e.g. opus CK sort -> asm sort).
   - "restructure" → change the ALGORITHM of the existing kernel (tiling/parallel decomposition/memory passes),
                     NOT just a tuning knob.
   Honor MODE HINT unless evidence clearly contradicts it (explain in notes). NEVER pick pure param tuning —
   if the only lever is a config knob, say so in notes and still propose the most algorithmic angle available.
1. Locate the target in the installed aiter; set kernel_type (ck|triton|flydsl|asm|hip). For author/replace,
   record replaced_kernels[] and reference_impl (the CUDA/asm/aiter reference to match), and the NEW sources[].
2. Define op_test: an allclose EQUIVALENCE harness (new/replaced path vs the ORIGINAL reference). Give the exact
   command and run it once on the baseline to confirm the reference path is correct.
2b. LOCAL PERF REPRODUCTION (MANDATORY — repro_unit_test + repro_match_note):
   Find a LOCAL test (prefer the repo unit test under op_tests/test_*.py if it prints per-shape us; else a
   microbench) and configure it to the kernel's PRODUCTION shape so its measured latency REPRODUCES the kernel
   per-call latency seen in the conc=4 PREFILL trace.
   - Derive the production shape (dims / dtype / varlen N / seqlen / BT) from the conc=4 prefill trace${amdahlRef ? ` (reference: ${amdahlRef} — read the per-kernel avg us and the module shape)` : ' (state how you obtained it: a reference trace, the model config, or a quick local profile)'}.
   - RUN the local test at that shape and compare its baseline latency to the trace per-call us. They must match
     within ~25% — otherwise the local number is NOT a faithful proxy; iterate the shape until it does (or explain).
   - repro_unit_test = the EXACT local command. repro_match_note = "trace per-call <X>us vs local <Y>us (<match>); production shape = <...>".
   This local test is the trustworthy proxy the develop loop and the micro gate rely on — without it the perf
   numbers are not credible.
3. Define MULTIPLE microbench_shapes — MUST include the conc=4 production shape from 2b AND bs in {4,8,16,32,64}
   (the geomean gate operates over these), spanning the regime. NEVER a single point. For each give an exact
   bench command, RUN it, and record baseline_microbench latency.
4. Note e2e_share_note: the op's profiled share of e2e (TTFT/ITL %) and the regime it dominates (the realistic
   e2e ceiling — be honest if the ceiling is below ${perfGatePct}%).
5. Propose an ORDERED algorithm experiment plan (3-6 items). For author/replace, experiment 1 = "first CORRECT version".
6. Record so_modules[], config_csv, and a precise rebuild recipe.

Do NOT edit source or rebuild in this step (other than the baseline microbench/op_test runs). Return the full schema.
`, { label: `recon:${taskId}`, phase: 'Recon', schema: RECON_SCHEMA, effort: 'high' })

  if (!recon || !recon.op_test || !(recon.microbench_shapes || []).length) {
    log(`[${taskId}] Recon failed — missing op_test / microbench_shapes.`)
    return { status: 'recon_failed', target_op: targetOp, recon }
  }

  const baseMicro = {}
  for (const s of (recon.baseline_microbench || [])) if (s && s.label) baseMicro[s.label] = s.latency_us
  log(`[${taskId}] mode=${recon.mode} kernel_type=${recon.kernel_type} shapes=${(recon.microbench_shapes || []).map(s => s.label).join(',')}`)
  log(`[${taskId}] baseline microbench: ${Object.entries(baseMicro).map(([k, v]) => `${k}=${v}us`).join(' ') || 'unparsed'}`)
  log(`[${taskId}] e2e ceiling note: ${recon.e2e_share_note || 'n/a'}`)

  // ── Develop loop (multi-shape microbench + allclose; NO e2e here) ──
  phase('Develop')
  const plan = (recon.experiments || []).slice()
  const history = []
  let best = null
  let noImprove = 0
  const isAuthor = recon.mode === 'author' || recon.mode === 'replace'

  function reconContext() {
    return JSON.stringify({
      target_op: recon.target_op, mode: recon.mode, kernel_type: recon.kernel_type,
      replaced_kernels: recon.replaced_kernels, reference_impl: recon.reference_impl,
      sources: recon.sources, so_modules: recon.so_modules, config_csv: recon.config_csv,
      op_test: recon.op_test, microbench_shapes: recon.microbench_shapes,
      baseline_microbench: recon.baseline_microbench, rebuild: recon.rebuild, knobs: recon.knobs,
    }, null, 2)
  }

  for (let round = 0; round < maxRounds; round++) {
    if (noImprove >= maxNoImprove) { log(`[${taskId}] Converged: ${noImprove} rounds with no new best.`); break }
    if (budget.total && budget.remaining() < minBudgetReserve) {
      log(`[${taskId}] Stopping develop: token budget low (${Math.round(budget.remaining() / 1000)}k).`); break
    }
    const nextPlanned = plan[round]
    const bestLine = best
      ? `Current BEST: id=${best.id} geomean=${(best._geomean_pct ?? best.agg_improvement_pct)?.toFixed?.(1)}%${microGatePct != null ? ` (gate ${microGatePct}%)` : ''}.`
      : 'No winning variant yet.'
    log(`[${taskId}] Develop round ${round + 1}/${maxRounds}${nextPlanned ? ` — ${nextPlanned.id}` : ' — adaptive'}`)

    const result = await agent(`
You are ONE round of an E2E-GATED ALGORITHMIC improvement loop for aiter on ROCm/MI355.
PIPELINE_MODE — fully autonomous. You make EXACTLY ONE coherent ALGORITHM change, rebuild, gate on correctness
(allclose), then MULTI-SHAPE microbench. Report MEASURED numbers only (never estimate).

${REBUILD_GUIDE}

RECON (ground truth):
${reconContext()}

BASELINE microbench (per shape, us): ${JSON.stringify(recon.baseline_microbench)}
${bestLine}

HISTORY:
${history.length ? JSON.stringify(history.map(h => ({ id: h.id, correct: h.correct, agg: h.agg_improvement_pct, min: h.min_improvement_pct, change: h.change_summary, note: h.observations })), null, 2) : '(none yet)'}

THIS ROUND:
${nextPlanned
    ? `Run planned experiment id="${nextPlanned.id}": ${nextPlanned.description}`
    : `Plan exhausted. From the history, pick the SINGLE most promising NEW algorithm variant and give it a fresh snake_case id.`}

PROCEDURE (do every step; stop early only on hard failure and report it):
${isAuthor
    ? `1. KEEP the current work-in-progress new/replaced kernel and build on it (round 1: create the first correct
   version + the allclose harness per reference_impl). If a previous round left it broken, FIX that first.`
    : `1. RESTORE BASELINE first (git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '<sources>'}; delete the so_modules .so) so this variant is measured independently.`}
2. Make exactly ONE coherent ALGORITHM change (not a tuning knob). Describe it precisely.
3. REBUILD per the contract (delete only the relevant .so / triton re-JIT / flydsl cache; author: compile the new module). Confirm rebuilt.
4. CORRECTNESS GATE: run op_test (allclose vs the original reference). If it fails or does not compile, set
   correct=false, DO NOT microbench, report what broke, and (non-author) restore baseline.
5. MULTI-SHAPE MICROBENCH: run the bench for EVERY shape in microbench_shapes (same shapes recon used). Fill
   per_shape[].latency_us from actual output. Compute agg_improvement_pct = mean over shapes of
   (baseline-after)/baseline*100, and min_improvement_pct = the worst single shape. A variant that speeds up one
   shape but REGRESSES another (min << 0) is NOT a win — say so.
6. Return the full schema with diff_files and an observations note for the next variant. Do not leave the repo
   half-built — if you abort, restore baseline + remove the half-built artifacts${isAuthor ? ' / new files' : ''}.
`, { label: `dev:${taskId}:${nextPlanned?.id || `r${round + 1}`}`, phase: 'Develop', schema: EXP_RESULT_SCHEMA })

    if (!result) { log(`[${taskId}] round ${round + 1}: null — skip`); noImprove++; continue }
    history.push(result)
    const ok = result.correct === true
    // Prefer GEOMEAN across bs (the requested metric); fall back to the agent's mean agg.
    const geo = geomeanImprovement(recon.baseline_microbench, result.per_shape)
    result._geomean_pct = geo.geomean_pct
    const agg = typeof geo.geomean_pct === 'number' ? geo.geomean_pct
      : (typeof result.agg_improvement_pct === 'number' ? result.agg_improvement_pct : null)
    const mn = typeof geo.min_pct === 'number' ? geo.min_pct
      : (typeof result.min_improvement_pct === 'number' ? result.min_improvement_pct : null)
    const bestAgg = best ? (best._geomean_pct ?? best.agg_improvement_pct) : null
    // a win must be correct, net-positive, not regress any shape by >1%, and beat the current best
    const isWin = ok && agg !== null && agg > 0 && (mn === null || mn >= -1) && (best === null || agg > bestAgg)
    if (isWin) {
      best = result
      noImprove = 0
      log(`[${taskId}]   new best: ${result.id} geomean=${agg.toFixed(1)}% (min-shape ${mn?.toFixed?.(1) ?? '?'}%)`)
    } else {
      noImprove++
      const why = !ok ? `incorrect (${result.error || 'op_test failed'})`
        : agg === null ? 'no latency parsed'
          : (mn !== null && mn < -1) ? `regresses a shape (min ${mn.toFixed(1)}%)`
            : `agg ${agg?.toFixed?.(1)}% not better than best`
      log(`[${taskId}]   no progress: ${result.id} — ${why}`)
    }
  }

  if (!best) {
    log(`[${taskId}] No correct net-positive variant. Restoring baseline.`)
    await agent(`Restore aiter to a clean baseline after an E2E-tune task that found no winning variant.
${REBUILD_GUIDE}
git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '.'} ; remove rebuilt .so for ${JSON.stringify(recon.so_modules || [])} ;
revert any edited CSV (${recon.config_csv || 'n/a'}).${isAuthor ? ` DELETE the new untracked files: ${JSON.stringify(recon.sources || [])}.` : ''}
Confirm 'git -C <AITER_ROOT> status' is clean for these files. Return one line.`, { label: `restore:${taskId}`, phase: 'Develop' })
    return {
      status: 'no_microbench_win', target_op: targetOp, kernel_type: recon.kernel_type, mode: recon.mode,
      rounds: history.length,
      history: history.map(h => ({ id: h.id, correct: h.correct, agg: h.agg_improvement_pct, min: h.min_improvement_pct, note: h.observations })),
      recon,
    }
  }

  // ── Micro perf gate (geomean across bs) — must clear before spending e2e ──
  const bestGeo = best._geomean_pct ?? best.agg_improvement_pct ?? null
  if (microGatePct != null && (bestGeo == null || bestGeo < microGatePct)) {
    log(`[${taskId}] MICRO GATE FAIL: best geomean ${bestGeo?.toFixed?.(1) ?? '?'}% < ${microGatePct}% — not proceeding to e2e. Restoring baseline.`)
    await agent(`Restore aiter to a clean baseline after a micro-gate failure. ${REBUILD_GUIDE}
git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '.'}; remove rebuilt .so ${JSON.stringify(recon.so_modules || [])}. Confirm 'git status' clean. One line.`, { label: `restore-microgate:${taskId}`, phase: 'Develop' })
    return {
      status: 'micro_gate_failed', target_op: targetOp, kernel_type: recon.kernel_type, mode: recon.mode,
      best_id: best.id, micro_geomean_pct: bestGeo, micro_gate_pct: microGatePct,
      rounds: history.length, feasibility, recon,
      history: history.map(h => ({ id: h.id, correct: h.correct, geomean: h._geomean_pct, note: h.observations })),
    }
  }
  if (microGatePct != null) log(`[${taskId}] MICRO GATE PASS: geomean ${bestGeo?.toFixed?.(1)}% >= ${microGatePct}%.`)

  // ── Apply best in isolation + rebuild + confirm correctness ──
  phase('E2E')
  log(`[${taskId}] Applying best (${best.id}) in isolation for e2e measurement.`)
  const applied = await agent(`
Apply ONLY the winning variant for an E2E measurement, in ISOLATION on a clean baseline. PIPELINE_MODE.
${REBUILD_GUIDE}
RECON:
${reconContext()}
WINNING VARIANT (reproduce exactly):
${JSON.stringify({ id: best.id, change_summary: best.change_summary, diff_files: best.diff_files }, null, 2)}

DO:
1. Restore a clean baseline first (${isAuthor ? 'remove any WIP/untracked files, then re-add ONLY the winning new file(s)' : `git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '<sources>'}`}), then APPLY ONLY the winning change.
2. Rebuild per the contract; delete the relevant .so so the next import/server-launch JIT-recompiles the change.
3. Run op_test (allclose) to confirm correct=true with ONLY this change applied.
4. Capture the change as a patch NOW (before any later revert): ${isAuthor ? 'git -C <AITER_ROOT> add -N <new files>; ' : ''}git -C <AITER_ROOT> diff > ${patchPath} ; put the same diff text (repo-relative) into "diff", set patch_path=${patchPath}.
5. Return EXP_RESULT schema (applied, rebuilt, correct, change_summary, diff_files). Leave the change APPLIED (do NOT revert yet — the e2e run needs it live).
`, { label: `apply:${taskId}`, phase: 'E2E', schema: EXP_RESULT_SCHEMA, effort: 'high' })

  if (!applied || applied.correct !== true) {
    log(`[${taskId}] Could not re-apply best correctly — skipping e2e, reverting.`)
    await agent(`Revert aiter to clean baseline (apply-best failed). ${REBUILD_GUIDE}
${isAuthor ? `Delete new files ${JSON.stringify(recon.sources || [])}; ` : `git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '.'}; `}remove rebuilt .so ${JSON.stringify(recon.so_modules || [])}; confirm clean. One line.`, { label: `revert:${taskId}`, phase: 'E2E' })
    return { status: 'apply_failed', target_op: targetOp, kernel_type: recon.kernel_type, mode: recon.mode, best_id: best.id, rounds: history.length, recon }
  }

  // ── Isolated e2e sweep (server picks up the applied edit) ──
  const resultDir = `${spec.outDir}/e2e_runs/${spec.runTag}/${spec.fileTag}`
  let e2e, delta
  if (e2eStage1Concs) {
    // Staged: stage1 must clear the gate before spending the expand concs.
    log(`[${taskId}] E2E stage1 concs="${e2eStage1Concs}" (gate ${perfGatePct}%) — server restart + sweep, 20-45 min.`)
    const s1 = await agent(
      e2eAgentPrompt({ ...e2eCfg, concurrencies: e2eStage1Concs }, `${resultDir}/stage1`, `task:${taskId}:${best.id}:s1`,
        `Only the "${targetOp}" change (${best.id}) is applied on the clean baseline. STAGE-1 gate concurrencies.`),
      { label: `e2e:${taskId}:s1`, phase: 'E2E', schema: E2E_SCHEMA, effort: 'medium' },
    )
    const d1 = e2eDelta(baselineE2E?.per_conc, s1?.per_conc)
    log(`[${taskId}] stage1 e2e median Δ=${d1.median_pct?.toFixed?.(2) ?? '?'}% min=${d1.min_pct?.toFixed?.(2) ?? '?'}%`)
    let mergedPer = (s1?.per_conc || []).slice()
    let acc = s1?.accuracy, accP = s1?.acc_pass
    const ran = s1?.ran
    if (e2eStage2Concs && d1.median_pct != null && d1.median_pct >= perfGatePct) {
      log(`[${taskId}] stage1 PASS -> expand concs="${e2eStage2Concs}" (gate ${expandGatePct}%).`)
      const s2 = await agent(
        e2eAgentPrompt({ ...e2eCfg, concurrencies: e2eStage2Concs }, `${resultDir}/stage2`, `task:${taskId}:${best.id}:s2`,
          `Only the "${targetOp}" change (${best.id}) is applied. STAGE-2 expand concurrencies.`),
        { label: `e2e:${taskId}:s2`, phase: 'E2E', schema: E2E_SCHEMA, effort: 'medium' },
      )
      mergedPer = mergedPer.concat(s2?.per_conc || [])
      if (acc == null) { acc = s2?.accuracy; accP = s2?.acc_pass }
    } else if (e2eStage2Concs) {
      log(`[${taskId}] stage1 FAIL (median ${d1.median_pct?.toFixed?.(2) ?? '?'}% < ${perfGatePct}%) -> NOT expanding to "${e2eStage2Concs}".`)
    }
    e2e = { label: `task:${taskId}:${best.id}`, ran, accuracy: acc, acc_pass: accP, per_conc: mergedPer, result_dir: resultDir }
    delta = e2eDelta(baselineE2E?.per_conc, mergedPer)
  } else {
    log(`[${taskId}] Running ISOLATED e2e sweep (this task only) — server restart + sweep, may take 20-45 min.`)
    e2e = await agent(
      e2eAgentPrompt(e2eCfg, resultDir, `task:${taskId}:${best.id}`,
        `Only the "${targetOp}" change (${best.id}) is applied on top of the clean baseline. Measuring its INDIVIDUAL e2e effect.`),
      { label: `e2e:${taskId}`, phase: 'E2E', schema: E2E_SCHEMA, effort: 'medium' },
    )
    delta = e2eDelta(baselineE2E?.per_conc, e2e?.per_conc)
  }
  const baseAcc = typeof baselineE2E?.accuracy === 'number' ? baselineE2E.accuracy : null
  const afterAcc = typeof e2e?.accuracy === 'number' ? e2e.accuracy : null
  const accDrop = (baseAcc !== null && afterAcc !== null) ? (baseAcc - afterAcc) : null
  // RELATIVE gate: accuracy must not drop more than tol vs baseline (absolute value is irrelevant — checkpoint is known-low)
  const accNoDrop = (accDrop === null) ? null : (accDrop <= spec.accDropTol)
  log(`[${taskId}] e2e total_throughput delta: median=${delta.median_pct?.toFixed?.(2) ?? '?'}% min=${delta.min_pct?.toFixed?.(2) ?? '?'}% | acc ${baseAcc ?? '?'}→${afterAcc ?? '?'} (drop=${accDrop?.toFixed?.(3) ?? '?'}, no_drop=${accNoDrop})`)

  // ── Profile (optional): conc-wise prefill traces OPT vs BASE, compare target-kernel sum ──
  let profile = null
  if (profileConcs) {
    phase('Profile')
    const kernels = (profileKernels && profileKernels.length) ? profileKernels : [targetOp]
    const profDir = `${resultDir}/profile`
    log(`[${taskId}] Profiling prefill at concs="${profileConcs}" (OPT applied, then BASE) — 2 server runs, may take 40-90 min.`)
    profile = await agent(`
You are the PROFILE step. PIPELINE_MODE — autonomous, background+poll for the long runs. Produce per-conc
prefill traces for BOTH the OPT (winning change, currently APPLIED) and the clean BASE, then compare the summed
duration of the TARGET kernel(s) opt/base.

${REBUILD_GUIDE}
TARGET KERNEL(S) (sum their durations, match by name substring): ${JSON.stringify(kernels)}
PROFILE CONCURRENCIES: ${profileConcs}
MODE: ${profileMode}   GATE: per-conc opt/base ratio <= ${profileGate}
PERF_SWEEP: ${PERF_SWEEP}   ANALYZER: /home/yichiche/agent-box/profile/trace_module_analyzer.py
SERVER: model=${e2eCfg.model} tp=${e2eCfg.tp} gpus=${e2eCfg.gpus} IL=${e2eCfg.inputLen} OL=${e2eCfg.outputLen}
SOURCES (for revert): ${JSON.stringify(recon.sources || [])}   so_modules: ${JSON.stringify(recon.so_modules || [])}

DO (run perf_sweep from cwd /tmp to avoid the sglang import-shadow bug):
1. OPT trace (the winning change is applied right now — confirm with git status before starting):
   For each conc in "${profileConcs}", capture a PREFILL torch profiler trace. Use perf_sweep with profiling:
     PROFILE_CONCS="${profileConcs}" PROFILE_DIR=${profDir}/opt MODEL=${e2eCfg.model} TP=${e2eCfg.tp} GPUS=${e2eCfg.gpus}${e2eCfg.port ? ` PORT=${e2eCfg.port}` : ''} \\
       INPUT_LEN=${e2eCfg.inputLen} OUTPUT_LEN=${e2eCfg.outputLen} CONCURRENCIES="${profileConcs}" RESULT_DIR=${profDir}/opt bash ${PERF_SWEEP}
   (use the SAME PORT for the BASE trace in step 4.)
   (background + poll; it launches its own server on its port and tears it down). Find the per-conc trace files
   under ${profDir}/opt.
2. For each conc trace, run: python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py <trace> -o ${profDir}/opt_conc<conc>_${profileMode}.xlsx --mode ${profileMode}
   then sum the duration (us) of the target kernel(s) from the analyzer output → opt_kernel_us per conc.
3. Revert to clean BASE: git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '<sources>'}; remove rebuilt .so ${JSON.stringify(recon.so_modules || [])}; confirm 'git status' clean. (Triton/flydsl: clear cache so base recompiles.)
4. BASE trace + analyze the same way → ${profDir}/base_conc<conc>_${profileMode}.xlsx → base_kernel_us per conc.
5. Compute ratio = opt_kernel_us/base_kernel_us per conc. gate_pass = every ratio <= ${profileGate}.
   Leave the tree at clean BASE (do NOT re-apply; the patch is the deliverable).
Return PROFILE_SCHEMA (per_conc with base/opt kernel us + ratio + xlsx paths).
`, { label: `profile:${taskId}`, phase: 'Profile', schema: PROFILE_SCHEMA, effort: 'medium' })
    log(`[${taskId}] profile: ${(profile?.per_conc || []).map(p => `c${p.concurrency} opt/base=${p.ratio?.toFixed?.(2) ?? '?'}`).join(' ') || 'n/a'} gate(<=${profileGate})=${profile?.gate_pass}`)
  }

  // ── Finalize: report + revert (isolation) ──
  phase('Finalize')
  const final = await agent(`
You are the FINALIZE step of an E2E-gated algorithmic task. PIPELINE_MODE. The winning change was applied for the
e2e measurement; if the Profile step ran it has since reverted the tree to clean BASE — that is fine, the saved
patch at ${patchPath} (written during the apply step) is the source of truth and the deliverable.

${REBUILD_GUIDE}
RECON:
${reconContext()}

WINNING VARIANT: ${JSON.stringify({ id: best.id, change_summary: best.change_summary, diff_files: best.diff_files }, null, 2)}
MICRO geomean across bs: ${bestGeo?.toFixed?.(1) ?? '?'}%${microGatePct != null ? ` (gate ${microGatePct}% — PASSED)` : ''}
LOCAL REPRO (faithful proxy of conc=4 trace): ${recon.repro_unit_test || 'n/a'} | ${recon.repro_match_note || ''}
FEASIBILITY (Amdahl): ${feasibility ? JSON.stringify({ kernel_share_pct: feasibility.kernel_share_pct, prefill_share_of_e2e_pct: feasibility.prefill_share_of_e2e_pct, e2e_ceiling_pct: feasibility.e2e_ceiling_pct, reachable: feasibility.reachable }) : 'not run'}
PROFILE (prefill target-kernel opt/base): ${profile ? JSON.stringify({ per_conc: profile.per_conc, gate_pass: profile.gate_pass, gate: profileGate }) : 'not run'}
STAGED E2E: ${e2eStage1Concs ? `stage1="${e2eStage1Concs}" expand="${e2eStage2Concs || '(none)'}" (expanded only if stage1 median Δ>=${perfGatePct}%)` : 'single sweep'}
BASELINE e2e (per conc total_throughput): ${JSON.stringify(baselineE2E?.per_conc || [])}
THIS-TASK e2e (per conc total_throughput): ${JSON.stringify(e2e?.per_conc || [])}
COMPUTED e2e delta: median=${delta.median_pct}% min=${delta.min_pct}% per_conc=${JSON.stringify(delta.per_conc)}
GSM8K accuracy: baseline=${baseAcc ?? 'n/a'} after=${afterAcc ?? 'n/a'} drop=${accDrop ?? 'n/a'} no_drop=${accNoDrop} (tol=${spec.accDropTol})
ACCURACY NOTE: this checkpoint has a KNOWN-low absolute gsm8k accuracy (~0.69) — that is EXPECTED. The gate is
RELATIVE: accuracy must NOT drop more than ${spec.accDropTol} vs baseline (greedy/deterministic ⇒ a real drop = a numeric bug in the change).
PERF GATE: ship_ready iff no_drop (accuracy not dropped beyond tol) AND e2e median delta >= ${perfGatePct}% AND e2e min delta >= -1% (no real throughput regression).

DO:
1. Confirm the patch ${patchPath} exists and holds the winning diff (it was saved in the apply step). If missing, recreate it from the applied tree BEFORE reverting.
2. Write a markdown report to ${reportPath}: target op, mode (${recon.mode}), kernel_type, the ALGORITHM change
   (what it replaced/fused/restructured), the op_test (allclose) command + verdict, the LOCAL REPRO command +
   its match-to-conc4-trace note (repro_unit_test / repro_match_note), the multi-shape microbench table
   (shape | baseline us | after us | Δ%) with the GEOMEAN across bs and the micro-gate verdict, the FEASIBILITY
   (Amdahl) numbers (kernel share of prefill, prefill share of e2e, e2e ceiling, reachable), the STAGED e2e
   tables (stage1 + expand if it ran) (conc | baseline tok/s | after tok/s | Δ%) with median/min e2e Δ%, the
   PROFILE table if it ran (conc | base kernel us | opt kernel us | opt/base ratio | gate<=${profileGate}),
   GSM8K accuracy (baseline→after with the RELATIVE no-drop verdict; note the absolute ~0.69 is a known checkpoint
   issue, NOT a regression), the perf-gate verdict, and the patch path. Repo-relative paths only in prose.
   Be honest: if the e2e gate failed because the kernel's Amdahl ceiling is below it, SAY SO and report the
   kernel-level + profile win as the real, bounded result.
3. REVERT to a clean baseline now (isolation so the next task measures clean) — if the Profile step already
   reverted, just CONFIRM clean:
   ${isAuthor ? `delete the new untracked files ${JSON.stringify(recon.sources || [])}; ` : `git -C <AITER_ROOT> checkout -- ${(recon.sources || []).join(' ') || '<sources>'}; `}remove rebuilt .so ${JSON.stringify(recon.so_modules || [])} so committed source recompiles; revert any edited CSV. Confirm 'git status' clean.
4. Set repo_state="reverted_win" if it passed the gate (patch is the deliverable) else "restored_baseline" (and patch_path may still be kept for reference). Set ship_ready per the gate. Echo e2e_after + the computed deltas + accuracy.

Return the full FINAL schema.
`, { label: `final:${taskId}`, phase: 'Finalize', schema: FINAL_SCHEMA, effort: 'high' })

  const fb = final || {}
  log(`[${taskId}] === done === best=${best.id} e2e median Δ=${(fb.e2e_improvement_pct_median ?? delta.median_pct ?? 0).toFixed?.(2) ?? '?'}% gate=${perfGatePct}% acc_pass=${fb.acc_pass ?? e2e?.acc_pass} ship_ready=${fb.ship_ready}`)
  log(`[${taskId}] report: ${fb.report_path || reportPath}`)

  return {
    status: 'done',
    target_op: targetOp,
    mode: recon.mode,
    kernel_type: recon.kernel_type,
    best_id: fb.best_id || best.id,
    microbench_agg_improvement_pct: best.agg_improvement_pct ?? null,
    micro_geomean_pct: fb.micro_geomean_pct ?? bestGeo ?? null,
    repro_unit_test: recon.repro_unit_test ?? null,
    repro_match_note: recon.repro_match_note ?? null,
    feasibility: feasibility || null,
    profile: profile ? { per_conc: profile.per_conc, gate_pass: profile.gate_pass, gate: profileGate } : null,
    profile_gate_pass: fb.profile_gate_pass ?? (profile ? profile.gate_pass : null),
    kernel_win: fb.kernel_win ?? (
      (fb.correct ?? true)
      && (microGatePct == null || (bestGeo != null && bestGeo >= microGatePct))
      && (profile == null || profile.gate_pass === true)
    ),
    e2e_improvement_pct_median: fb.e2e_improvement_pct_median ?? delta.median_pct,
    e2e_min_improvement_pct: fb.e2e_min_improvement_pct ?? delta.min_pct,
    e2e_after: e2e?.per_conc || [],
    accuracy_baseline: baseAcc,
    accuracy_after: fb.accuracy_after ?? afterAcc,
    accuracy_drop: accDrop,
    acc_no_drop: accNoDrop,
    correct: fb.correct ?? true,
    repo_state: fb.repo_state,
    ship_ready: fb.ship_ready ?? (
      accNoDrop !== false
      && (fb.e2e_improvement_pct_median ?? delta.median_pct ?? -1) >= perfGatePct
      && (fb.e2e_min_improvement_pct ?? delta.min_pct ?? -99) >= -1
    ),
    diff: fb.diff || null,
    patch_path: fb.patch_path || patchPath,
    report_path: fb.report_path || reportPath,
    summary: fb.summary,
    sources: recon.sources,
    so_modules: recon.so_modules,
    rounds: history.length,
    experiments: history.map(h => ({ id: h.id, change: h.change_summary, correct: h.correct, agg: h.agg_improvement_pct, min: h.min_improvement_pct })),
  }
}

// ── Orchestrator ─────────────────────────────────────────────────────────────

const g = {
  model: args?.model || '/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4',
  tp: args?.tp ?? 8,
  gpus: args?.gpus || '0,1,2,3,4,5,6,7',
  port: args?.port ?? null,   // perf_sweep server port; null -> perf_sweep default (8001)
  microGpus: args?.microbench_gpus || (args?.gpus ? String(args.gpus).split(',')[0] : '0'),
  inputLen: args?.input_len ?? 8192,
  outputLen: args?.output_len ?? 1024,
  concurrencies: args?.e2e_concurrencies || '16 64 128',
  numPromptsMult: args?.num_prompts_mult ?? 4,
  numPromptsCap: args?.num_prompts_cap ?? 512,
  accuracy: args?.accuracy !== false,
  accThreshold: args?.acc_threshold ?? 0.0,   // 0 = perf_sweep never aborts on absolute acc; gate is RELATIVE drop vs baseline
  accDropTol: args?.acc_drop_tol ?? 0.02,     // allowed gsm8k accuracy drop (greedy/deterministic ⇒ a real drop = numeric bug)
  accNumQ: args?.acc_num_q ?? 200,
  gsm8kScript: args?.gsm8k_script || '/home/yichiche/agent-box/workflow/gsm8k_thinking.py',
  gsm8kMaxNewTokens: args?.gsm8k_max_new_tokens ?? 8192,
  perfGatePct: args?.perf_gate_pct ?? 5.0,
  maxRounds: args?.max_rounds ?? 8,
  maxNoImprove: args?.max_no_improve ?? 3,
  minBudgetReserve: args?.min_budget_reserve ?? 80000,
  aiterRoot: args?.aiter_root || '',
  outDir: args?.outDir || '~/.kernel-fusion-pipeline',
  runTag: args?.run_tag || 'run',   // unique per invocation → e2e sweep dirs never collide across runs
  continueOnFail: args?.continue_on_fail !== false,
  // ── new (args-gated; null/empty = legacy behavior preserved) ──
  microGatePct: args?.micro_gate_pct ?? null,        // geomean opt/base across bs MUST clear this before e2e
  e2eStage1Concs: args?.e2e_stage1_concs || null,    // staged sweep: stage1 (e.g. "4 8 16 32") gates expand
  e2eStage2Concs: args?.e2e_stage2_concs || null,    // stage2 expand (e.g. "64 128"), only if stage1 passes
  expandGatePct: args?.expand_gate_pct ?? 2.0,       // looser gate for the stage2 expand concs
  profileConcs: args?.profile_concs || null,         // e.g. "4 64" → Profile phase (base vs opt prefill traces)
  profileKernels: Array.isArray(args?.profile_kernels) ? args.profile_kernels : (args?.profile_kernels ? [args.profile_kernels] : []),
  profileGate: args?.profile_gate ?? 0.70,           // target-kernel-sum opt/base must be <= this
  profileMode: args?.profile_mode || 'prefill',
  amdahlRef: args?.amdahl_ref || null,               // path to a reference profile xlsx → Feasibility phase
  amdahlMinSharePct: args?.amdahl_min_share_pct ?? null, // optional hard-stop if kernel share below this
}

// When a staged sweep is requested, the baseline must cover the UNION of stage1+stage2 concs
// so every opt stage can be compared against a matching baseline concurrency.
if (g.e2eStage1Concs) {
  const union = [...new Set(`${g.e2eStage1Concs} ${g.e2eStage2Concs || ''}`.trim().split(/\s+/).filter(Boolean))]
  g.concurrencies = union.join(' ')
}

const e2eCfg = {
  model: g.model, tp: g.tp, gpus: g.gpus, port: g.port, inputLen: g.inputLen, outputLen: g.outputLen,
  concurrencies: g.concurrencies, numPromptsMult: g.numPromptsMult, numPromptsCap: g.numPromptsCap,
  accuracy: g.accuracy, accThreshold: g.accThreshold, accNumQ: g.accNumQ,
  gsm8kScript: g.gsm8kScript, gsm8kMaxNewTokens: g.gsm8kMaxNewTokens,
}

function normalizeTasks(a) {
  const out = []
  if (Array.isArray(a?.tasks) && a.tasks.length) {
    for (const t of a.tasks) {
      if (t && typeof t.target_op === 'string' && t.target_op.trim()) out.push(t)
      else log('WARN: skip task without target_op')
    }
  }
  if (!out.length && a?.target_op && String(a.target_op).trim()) out.push({ target_op: String(a.target_op).trim() })
  return out
}

let taskList = normalizeTasks(args)
if (!taskList.length) {
  log('ERROR: pass args.tasks=[{ target_op, workload?, regime?, mode?, priority? }] or args.target_op')
  return { error: 'Missing targets. Example: { tasks: [{ target_op: "fuse quant into first moe gemm", regime: "decode" }], model: "/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4" }' }
}
taskList.sort((x, y) => (x.priority ?? 999) - (y.priority ?? 999))

log(`implement-e2e: ${taskList.length} task(s) | model=${g.model} tp=${g.tp} gpus=${g.gpus}`)
log(`e2e workload: IL=${g.inputLen}/OL=${g.outputLen} conc="${g.concurrencies}" gate=${g.perfGatePct}% (individual) acc>=${g.accThreshold}`)

// ── Phase 0: global clean-baseline e2e (once) ──
phase('Baseline')
const baseDir = `${g.outDir}/e2e_runs/${g.runTag}/baseline`
log(`Measuring CLEAN baseline e2e (one server launch + sweep + GSM8K) — may take 20-45 min.`)
const baselineE2E = await agent(
  e2eAgentPrompt(e2eCfg, baseDir, 'baseline',
    'Clean aiter baseline (NO task changes applied; any pre-existing unrelated working-tree edits are left as-is). This is the reference for every per-task e2e delta.'),
  { label: 'e2e:baseline', phase: 'Baseline', schema: E2E_SCHEMA, effort: 'medium' },
)
if (!baselineE2E || baselineE2E.ran === false || !(baselineE2E.per_conc || []).length) {
  log('ERROR: baseline e2e failed — cannot gate per-task deltas without it.')
  return { status: 'baseline_failed', baseline: baselineE2E }
}
log(`Baseline total_throughput: ${(baselineE2E.per_conc || []).map(r => `c${r.concurrency}=${r.total_throughput_tok_s}`).join(' ')} | acc=${baselineE2E.accuracy} (pass=${baselineE2E.acc_pass})`)

// ── Per-task (sequential, isolated) ──
const results = []
for (let i = 0; i < taskList.length; i++) {
  const raw = taskList[i]
  const slug = raw.target_op.replace(/[^a-z0-9_]+/gi, '_').slice(0, 45).replace(/_+$/, '')
  const fileTag = taskList.length <= 1 ? slug : `${i}_${slug}`
  const spec = {
    targetOp: raw.target_op.trim(),
    taskId: raw.id || `t${i}`,
    regime: raw.regime || '',
    modeHint: raw.mode || '',
    workload: raw.workload || '',
    aiterRootArg: raw.aiter_root || g.aiterRoot,
    perfGatePct: raw.perf_gate_pct ?? g.perfGatePct,
    maxRounds: raw.max_rounds ?? g.maxRounds,
    maxNoImprove: raw.max_no_improve ?? g.maxNoImprove,
    minBudgetReserve: g.minBudgetReserve,
    microGpus: g.microGpus,
    accDropTol: g.accDropTol,
    outDir: g.outDir,
    runTag: g.runTag,
    fileTag,
    reportPath: `${g.outDir}/e2e_${fileTag}.md`,
    patchPath: `${g.outDir}/e2e_${fileTag}.patch`,
    // ── new (args-gated) ──
    microGatePct: raw.micro_gate_pct ?? g.microGatePct,
    e2eStage1Concs: g.e2eStage1Concs,
    e2eStage2Concs: g.e2eStage2Concs,
    expandGatePct: g.expandGatePct,
    profileConcs: g.profileConcs,
    profileKernels: g.profileKernels,
    profileGate: g.profileGate,
    profileMode: g.profileMode,
    amdahlRef: raw.amdahl_ref || g.amdahlRef,
    amdahlMinSharePct: g.amdahlMinSharePct,
  }
  log(`=== implement-e2e task ${i + 1}/${taskList.length} (${spec.taskId}) target=${spec.targetOp} ===`)
  const one = await runTaskE2E(spec, e2eCfg, baselineE2E)
  results.push({ task_id: spec.taskId, task_index: i, ...one })

  const fatal = one?.status === 'recon_failed' || one?.status === 'baseline_incorrect'
  if (fatal && !g.continueOnFail) { log(`Stopping: ${one.status} and continue_on_fail=false`); break }
}

if (taskList.length === 1) return { baseline_e2e: baselineE2E.per_conc, ...results[0] }

return {
  status: 'done_all',
  task_count: results.length,
  model: g.model,
  e2e_workload: `IL${g.inputLen}/OL${g.outputLen} conc=${g.concurrencies}`,
  perf_gate_pct: g.perfGatePct,
  gate_note: 'per-kernel INDIVIDUAL e2e total_throughput delta (NOT cumulative sum)',
  baseline_e2e: baselineE2E.per_conc,
  baseline_accuracy: baselineE2E.accuracy,
  results,
}
