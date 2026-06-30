
export const meta = {
  name: 'validate-patches',
  description: 'Separate before/after validation of kernel patches across one or more model checkpoints. For each variant (baseline + each patch, in isolation) it applies the patch ONCE, then for each model runs a concurrency sweep at the given ISL/OSL workloads via perf_sweep.sh plus a conc=4 profiling capture, then trace_module_analyzer to show the TARGET kernel before/after. Emits per-model throughput/latency tables + the kernel-level diff. Report-only.',
  whenToUse: 'Validate a kernel patch helps e2e AND that the win comes from the intended kernel, across checkpoints. Pass args.patches, args.models, args.isls, tp/gpus.',
  phases: [
    { title: 'Apply', detail: 'Per variant: apply patch + rebuild (or clean baseline)' },
    { title: 'Sweep', detail: 'Per (variant,model): perf_sweep both ISL + conc4 profile' },
    { title: 'Revert', detail: 'Per variant: restore clean tree' },
    { title: 'Analyze', detail: 'trace_module_analyzer on conc4 traces; target kernel before/after' },
    { title: 'Report', detail: 'Assemble per-model tables + kernel diff' },
  ],
}

const PERF_SWEEP = '/home/yichiche/agent-box/skills/perf-sweep/perf_sweep.sh'
const ANALYZER = '/home/yichiche/agent-box/profile/trace_module_analyzer.py'

// ── Schemas ──────────────────────────────────────────────────────────────────

const ROW_SCHEMA = {
  type: 'object',
  properties: {
    isl: { type: 'number' }, osl: { type: 'number' }, concurrency: { type: 'number' },
    median_e2e_ms: { type: ['number', 'null'] },
    total_tok_s: { type: ['number', 'null'] },
    total_tok_s_per_gpu: { type: ['number', 'null'] },
    median_ttft_ms: { type: ['number', 'null'] },
    median_tpot_ms: { type: ['number', 'null'] },
  },
  required: ['isl', 'osl', 'concurrency'],
}
const PREP_SCHEMA = {
  type: 'object',
  properties: { variant: { type: 'string' }, applied_ok: { type: 'boolean' }, error: { type: 'string' }, notes: { type: 'string' } },
  required: ['variant', 'applied_ok'],
}
const SWEEP_SCHEMA = {
  type: 'object',
  properties: {
    variant: { type: 'string' }, model_tag: { type: 'string' }, ran: { type: 'boolean' },
    rows: { type: 'array', items: ROW_SCHEMA },
    trace_path: { type: ['string', 'null'] },
    result_dirs: { type: 'array', items: { type: 'string' } },
    error: { type: 'string' }, notes: { type: 'string' },
  },
  required: ['variant', 'ran', 'rows'],
}
const REVERT_SCHEMA = {
  type: 'object',
  properties: { variant: { type: 'string' }, reverted_ok: { type: 'boolean' }, error: { type: 'string' } },
  required: ['variant', 'reverted_ok'],
}
const KERNEL_ROW = {
  type: 'object',
  properties: { name: { type: 'string' }, total_us: { type: ['number', 'null'] }, count: { type: ['number', 'null'] }, avg_us: { type: ['number', 'null'] } },
  required: ['name'],
}
const ANALYZE_SCHEMA = {
  type: 'object',
  properties: {
    variant: { type: 'string' }, model_tag: { type: 'string' }, xlsx_path: { type: ['string', 'null'] },
    target_kernel: KERNEL_ROW, control_kernel: KERNEL_ROW, error: { type: 'string' }, notes: { type: 'string' },
  },
  required: ['variant'],
}

// ── Config ───────────────────────────────────────────────────────────────────

const g = {
  models: args?.models || ['/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4', '/data/amd/Qwen3.5-397B-A17B-MXFP4-7f34fa9'],
  tp: args?.tp ?? 2,
  gpus: args?.gpus || 'auto',   // 'auto' = pick free CUDA indices via gpu_status.py; else explicit CUDA indices
  isls: args?.isls || [
    { isl: 1024, osl: 1024, concs: '4 8 16 32 64 128 256' },
    { isl: 8192, osl: 1024, concs: '4 8 16 32 64 128 256' },
  ],
  profileIsl: args?.profile_isl ?? 8192,
  profileConc: args?.profile_conc ?? 4,
  numPromptsMult: args?.num_prompts_mult ?? 4,
  numPromptsCap: args?.num_prompts_cap ?? 512,
  outDir: args?.outDir || '~/.kernel-fusion-pipeline/validate',
  runTag: args?.run_tag || 'run',   // unique per invocation → no stale cross-run data
  aiterRoot: args?.aiter_root || '/sgl-workspace/aiter',
}
const numGpus = g.tp   // a TP server pins exactly tp GPUs; tok/s/gpu = total / tp
const modelTag = (m) => m.split('/').filter(Boolean).pop().replace(/[^A-Za-z0-9]+/g, '_').slice(-16)

const GPU_SCHED_SCHEMA = {
  type: 'object',
  properties: {
    gpus: { type: 'string', description: 'comma-joined CUDA indices to use (CUDA_VISIBLE_DEVICES), or "" if not enough free' },
    free_cuda: { type: 'array', items: { type: 'number' } },
    error: { type: 'string' },
  },
  required: ['gpus'],
}

const patches = args?.patches || [
  {
    id: 'patch0_attn', name: 'attention prefill (CK fmha bk0/bk1 32→64)',
    patch_path: '~/.kernel-fusion-pipeline/e2e_0_prefill_large_batch_attention_kernel_the_domi.patch',
    apply_dir: '3rdparty/composable_kernel', so_glob: 'aiter/jit/mha_batch_prefill_bf16*.so',
    revert_path_in_dir: 'example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py', revert_in_submodule: true,
    target_kernel: 'FmhaBatchPrefill (CK ck_tile fmha batch_prefill fwd — dense full-attention prefill kernel)',
    control_kernel: 'chunk_gated_delta_rule_fwd_kernel (linear/gated-delta attention — must be UNCHANGED)',
  },
  {
    id: 'patch1_moe', name: 'flydsl MoE stage2 reduction grid-collapse',
    patch_path: '~/.kernel-fusion-pipeline/e2e_1_flydsl_fused_MoE_GEMM_kernels_mfma_moe1_silu.patch',
    apply_dir: '', so_glob: '',
    revert_path_in_dir: 'aiter/ops/flydsl/kernels/moe_gemm_2stage.py', revert_in_submodule: false,
    target_kernel: 'the flydsl MoE stage2 topk-reduction kernel (moe_reduction in the MoE 2-stage path)',
    control_kernel: 'FmhaBatchPrefill (attention — must be UNCHANGED by the MoE patch)',
  },
]
const variants = [{ id: 'baseline', name: 'clean baseline (no patch)', patch_path: null }, ...patches]

// ── helpers ──────────────────────────────────────────────────────────────────

function sweepCmd(model, isl, osl, concs, resultDir, profileConc) {
  const prof = profileConc ? `PROFILE_CONCS="${profileConc}" PROFILE_DIR=${resultDir}/profiles ` : ''
  return `MODEL=${model} TP=${g.tp} GPUS=${g.gpus} INPUT_LEN=${isl} OUTPUT_LEN=${osl} `
    + `CONCURRENCIES="${concs}" NUM_PROMPTS_MULT=${g.numPromptsMult} NUM_PROMPTS_CAP=${g.numPromptsCap} `
    + `ACCURACY=0 ${prof}RESULT_DIR=${resultDir} bash ${PERF_SWEEP}`
}
function revertCmd(v) {
  if (!v.patch_path) return `# baseline — nothing to revert`
  const co = v.revert_in_submodule
    ? `git -C ${g.aiterRoot}/${v.apply_dir} checkout -- ${v.revert_path_in_dir}`
    : `git -C ${g.aiterRoot} checkout -- ${v.revert_path_in_dir}`
  const rmso = v.so_glob ? `\n  rm -f ${g.aiterRoot}/${v.so_glob}` : ''
  return `${co}${rmso}`
}

// ── Apply ────────────────────────────────────────────────────────────────────
async function applyVariant(v) {
  const steps = v.patch_path
    ? `Apply this patch in ISOLATION on a clean tree, then rebuild:
  cd ${g.aiterRoot}
  ${v.apply_dir ? `git apply --directory=${v.apply_dir} ${v.patch_path}` : `git apply ${v.patch_path}`}
  ${v.so_glob ? `rm -f ${g.aiterRoot}/${v.so_glob}   # force JIT recompile of the edited CK module on next server launch` : `# (triton/flydsl re-JIT on source change; no .so)`}
  Verify the changed line is present (grep). ${v.so_glob ? 'Optionally trigger the CK rebuild now (import aiter / run the op once) so a compile error is caught before the server launches.' : ''}`
    : `BASELINE — do NOT apply any patch. Just confirm the target files are clean:
  cd ${g.aiterRoot}; git status --porcelain   # only unrelated qwen CSV CRLF noise is acceptable`
  return agent(`Prepare the aiter tree for variant "${v.id}" (${v.name}). PIPELINE_MODE, autonomous.
${steps}
Return PREP schema: variant="${v.id}", applied_ok (true if patch applied+rebuilt cleanly, or true for a clean baseline), notes/error.`,
    { label: `apply:${v.id}`, phase: 'Apply', schema: PREP_SCHEMA, effort: 'low' })
}

// ── Sweep one (variant, model) ───────────────────────────────────────────────
async function sweepVariantModel(v, model) {
  const mt = modelTag(model)
  const sweeps = g.isls.map((s) => {
    const rd = `${g.outDir}/${g.runTag}/${mt}/${v.id}/isl${s.isl}_osl${s.osl}`
    const prof = (s.isl === g.profileIsl) ? g.profileConc : 0
    return { ...s, rd, prof, cmd: sweepCmd(model, s.isl, s.osl, s.concs, rd, prof) }
  })
  return agent(`
Benchmark ONE variant on ONE model. PIPELINE_MODE, autonomous. The patch for this variant is ALREADY APPLIED to
the editable aiter — do NOT re-apply or revert here; just run the sweeps. Report MEASURED numbers only.

VARIANT: ${v.id} (${v.name})
MODEL:   ${model}   (tag=${mt})   TP=${g.tp}  GPUS=${g.gpus}

GPU SAFETY (MANDATORY): the GPU values ${g.gpus} are CUDA indices (perf_sweep pins via CUDA_VISIBLE_DEVICES — sglang
ignores HIP_VISIBLE_DEVICES, and rocm-smi index != CUDA index on this box). Before launching, VERIFY with
"python3 /home/yichiche/agent-box/skills/gpu-status/gpu_status.py" that EVERY CUDA index in ${g.gpus} is 🟢FREE.
If any is 🔴OCCUPIED, STOP and report ran=false — do NOT run, do NOT kill the occupant. NEVER kill/pkill/signal a
server you did not start in THIS task; perf_sweep tears down only its own port.

CRITICAL — cwd: run EVERY perf_sweep command from cwd /tmp (cd /tmp first). Running from /home/yichiche triggers
an sglang import-shadowing bug that makes the bench client write NO result JSON → an empty summary.csv.

CRITICAL — no stale data: before each sweep, "rm -rf <rd>" so nothing from a prior run remains, then mkdir -p <rd>.

Run these perf_sweep.sh sweeps SEQUENTIALLY (they share GPUs). Each launches the server host-native (picks up the
applied aiter edit), sweeps concurrency, writes <rd>/summary.csv. Each is LONG (20-60 min) and EXCEEDS the 10-min
Bash limit — run each in the BACKGROUND (cd /tmp && nohup ... > <rd>/run.log 2>&1 &) and POLL every ~90s (done when
run.log shows "DONE."). Watch <rd>/server.log for "Traceback|OutOfMemory|HIP error".

${sweeps.map((s, i) => `SWEEP ${i + 1} (ISL=${s.isl}/OSL=${s.osl}${s.prof ? `, profile conc=${s.prof}` : ''}):\n  cd /tmp && rm -rf ${s.rd} && mkdir -p ${s.rd} && nohup bash -c '${s.cmd}' > ${s.rd}/run.log 2>&1 &`).join('\n\n')}

FAIL-CLOSED PARSING (mandatory — do NOT fabricate):
- Parse ONLY each sweep's OWN <rd>/summary.csv. Use this exact command per sweep and copy the printed rows verbatim:
    python3 -c "import csv,sys; r=list(csv.DictReader(open('<rd>/summary.csv'))); [print(d['max_concurrency'],d.get('total_throughput'),d.get('median_e2e_latency_ms'),d.get('median_ttft_ms'),d.get('median_tpot_ms')) for d in r]"
- A row counts ONLY if summary.csv has that data line with a numeric total_throughput. If summary.csv is missing,
  header-only, or has zero data rows, that sweep FAILED: emit NO rows for it and put the cause in error/notes.
- NEVER copy or infer numbers from another directory, a prior run, a report .md, or memory. If you have no data, say so.
- Set ran=true ONLY if at least one real data row was parsed from a summary.csv produced by THIS task. Otherwise ran=false.

For each parsed (isl,conc) row emit: isl, osl, concurrency, median_e2e_ms, total_tok_s=total_throughput,
total_tok_s_per_gpu = total_throughput/${numGpus}, median_ttft_ms, median_tpot_ms.

Find the conc=${g.profileConc} ISL=${g.profileIsl} trace (*.trace.json.gz under the ISL=${g.profileIsl} sweep's
<rd>/profiles, or grep its server.log for "Traces are saved to"); return its absolute path in trace_path (null if none).

Return SWEEP schema: variant="${v.id}", model_tag="${mt}", ran (per rule above), rows[], trace_path,
result_dirs=${JSON.stringify(sweeps.map(s => s.rd))}, notes.
`, { label: `sweep:${v.id}:${mt}`, phase: 'Sweep', schema: SWEEP_SCHEMA, effort: 'medium' })
}

// ── Revert ───────────────────────────────────────────────────────────────────
async function revertVariant(v) {
  if (!v.patch_path) return { variant: v.id, reverted_ok: true }
  return agent(`Revert the aiter tree to clean after benchmarking variant "${v.id}". PIPELINE_MODE.
  cd ${g.aiterRoot}
  ${revertCmd(v)}
Confirm 'git status' shows the target file clean (only unrelated qwen CSV CRLF noise may remain).
Return REVERT schema: variant="${v.id}", reverted_ok, error.`,
    { label: `revert:${v.id}`, phase: 'Revert', schema: REVERT_SCHEMA, effort: 'low' })
}

// ── Analyze a conc4 trace ────────────────────────────────────────────────────
async function analyzeTrace(v, model, tracePath) {
  const mt = modelTag(model)
  if (!tracePath) return { variant: v.id, model_tag: mt, error: 'no trace', target_kernel: { name: 'n/a' } }
  const xlsx = `${g.outDir}/${mt}/${v.id}/analysis_conc${g.profileConc}.xlsx`
  const what = v.id === 'baseline'
    ? 'Extract BOTH the FmhaBatchPrefill attention kernel AND the flydsl MoE reduction kernel (targets of patch0 & patch1). Put FMHA in target_kernel and the MoE reduction in control_kernel.'
    : `Extract TARGET kernel: ${v.target_kernel}. Also CONTROL (must be ~unchanged): ${v.control_kernel}.`
  return agent(`Analyze ONE profiling trace for a per-kernel before/after comparison. PIPELINE_MODE.
TRACE: ${tracePath}
Produce the Excel report (same format as the user's analysis_prefill.xlsx):
  python3 ${ANALYZER} ${tracePath} -o ${xlsx}
If it errors, capture it and fall back to reading kernel durations from the trace directly.
Read the "GPU Kernels" sheet (cols: Kernel Name, Total Duration (us), Count, Avg (us)). Match by case-insensitive
substring: FMHA → "FmhaBatchPrefill"/"fmha"; flydsl MoE reduction → "moe_reduction" (the reduction in the MoE
2-stage path, NOT the mfma_moe1/moe2 GEMMs unless that is the only reduction kernel — say what you matched);
gated-delta → "chunk_gated_delta_rule".
${what}
Return ANALYZE schema: variant="${v.id}", model_tag="${mt}", xlsx_path, target_kernel{name,total_us,count,avg_us}, control_kernel{...}, notes.`,
    { label: `analyze:${v.id}:${mt}`, phase: 'Analyze', schema: ANALYZE_SCHEMA, effort: 'medium' })
}

// ── Auto-select free GPUs (CUDA indices) ─────────────────────────────────────
if (!g.gpus || g.gpus === 'auto') {
  phase('Schedule')
  const sched = await agent(`Pick ${g.tp} FREE GPUs for a TP=${g.tp} sglang run. PIPELINE_MODE, autonomous.
Run EXACTLY: python3 /home/yichiche/agent-box/skills/gpu-status/gpu_status.py
Parse its "Free CUDA indices: [...]" line. These are CUDA_VISIBLE_DEVICES indices (NOT rocm-smi indices — the two
are permuted on this box; trust ONLY gpu_status.py). Choose the FIRST ${g.tp} free CUDA indices and return them as a
comma-joined string in "gpus" (e.g. "0,1"). Also return the full free list in free_cuda. If fewer than ${g.tp} are
free, set gpus="" and explain in error. Do NOT launch anything or touch any GPU here — this is selection only.`,
    { label: 'gpu-schedule', phase: 'Schedule', schema: GPU_SCHED_SCHEMA, effort: 'low' })
  if (!sched || !sched.gpus || !String(sched.gpus).trim()) {
    log(`No free ${g.tp}-GPU set available (${sched?.error || 'gpu_status reported none'}). Aborting.`)
    return { status: 'no_free_gpus', tp: g.tp, detail: sched }
  }
  g.gpus = String(sched.gpus).trim()
  log(`Auto-selected CUDA GPUs: ${g.gpus} (free: ${(sched.free_cuda || []).join(',')})`)
}

// ── Orchestrate ──────────────────────────────────────────────────────────────
log(`validate-patches: models=[${g.models.map(modelTag).join(', ')}] tp=${g.tp} gpus=${g.gpus} numGpus=${numGpus}`)
log(`variants=${variants.map(v => v.id).join(', ')} ISLs=${g.isls.map(s => `${s.isl}/${s.osl}`).join(',')} conc=${g.isls[0].concs}`)
log(`TOTAL server launches ≈ ${variants.length} × ${g.models.length} × ${g.isls.length} = ${variants.length * g.models.length * g.isls.length} — multi-hour run. (User server on GPU6,7 is untouched.)`)

const measured = []   // {variant, model, mt, sweep}
for (const v of variants) {
  phase('Apply')
  const prep = await applyVariant(v)
  log(`[${v.id}] applied_ok=${prep?.applied_ok} ${prep?.error ? '(' + prep.error + ')' : ''}`)
  if (prep && prep.applied_ok === false) {
    log(`[${v.id}] apply failed — skipping its sweeps.`)
    for (const m of g.models) measured.push({ variant: v, model: m, mt: modelTag(m), sweep: null, applyFailed: true })
    await revertVariant(v)
    continue
  }
  phase('Sweep')
  for (const m of g.models) {
    log(`--- sweep ${v.id} on ${modelTag(m)} ---`)
    const sw = await sweepVariantModel(v, m)
    measured.push({ variant: v, model: m, mt: modelTag(m), sweep: sw })
    log(`[${v.id}/${modelTag(m)}] ran=${sw?.ran} rows=${(sw?.rows || []).length} trace=${sw?.trace_path ? 'y' : 'n'}`)
  }
  phase('Revert')
  const rv = await revertVariant(v)
  log(`[${v.id}] reverted_ok=${rv?.reverted_ok}`)
}

phase('Analyze')
const analyses = []
for (const mm of measured) {
  if (!mm.sweep) continue
  const a = await analyzeTrace(mm.variant, mm.model, mm.sweep.trace_path)
  analyses.push({ ...a, _vid: mm.variant.id, _mt: mm.mt })
  log(`[${mm.variant.id}/${mm.mt}] target ${a?.target_kernel?.name}: avg=${a?.target_kernel?.avg_us ?? '?'}us`)
}

phase('Report')
// Per-model tables with throughput delta vs that model's baseline; kernel diff vs baseline.
const byModel = {}
for (const m of g.models) {
  const mt = modelTag(m)
  const baseRows = measured.find(x => x.variant.id === 'baseline' && x.mt === mt)?.sweep?.rows || []
  const baseMap = {}
  for (const r of baseRows) baseMap[`${r.isl}|${r.osl}|${r.concurrency}`] = r
  const tables = measured.filter(x => x.mt === mt).map(x => ({
    variant: x.variant.id, name: x.variant.name,
    rows: (x.sweep?.rows || []).map(r => {
      const b = baseMap[`${r.isl}|${r.osl}|${r.concurrency}`]
      const d = (b?.total_tok_s && r.total_tok_s) ? (r.total_tok_s - b.total_tok_s) / b.total_tok_s * 100 : null
      return { ...r, tput_delta_pct: d }
    }),
  }))
  // kernel diff vs baseline for this model
  const baseA = analyses.find(a => a._vid === 'baseline' && a._mt === mt)
  const kdiff = analyses.filter(a => a._vid !== 'baseline' && a._mt === mt).map(a => {
    const tk = a.target_kernel || {}
    let baseTk = null
    for (const c of [baseA?.target_kernel, baseA?.control_kernel]) {
      if (c?.name && tk.name && (
        (c.name.toLowerCase().includes('fmha') && tk.name.toLowerCase().includes('fmha'))
        || (c.name.toLowerCase().includes('moe') && tk.name.toLowerCase().includes('moe'))
        || (c.name.toLowerCase().includes('reduction') && tk.name.toLowerCase().includes('reduction'))
      )) { baseTk = c; break }
    }
    const sp = (baseTk?.avg_us && tk?.avg_us) ? (baseTk.avg_us - tk.avg_us) / baseTk.avg_us * 100 : null
    return { variant: a._vid, target_kernel: tk.name, baseline_avg_us: baseTk?.avg_us ?? null, patched_avg_us: tk?.avg_us ?? null, kernel_speedup_pct: sp, control_kernel: a.control_kernel, notes: a.notes }
  })
  byModel[mt] = { model: m, tables, kernel_diff: kdiff }
}

log('=== validate-patches complete ===')
for (const mt of Object.keys(byModel))
  for (const kd of byModel[mt].kernel_diff)
    log(`[${mt}/${kd.variant}] ${kd.target_kernel}: ${kd.baseline_avg_us ?? '?'}→${kd.patched_avg_us ?? '?'}us (${kd.kernel_speedup_pct?.toFixed?.(2) ?? '?'}%)`)

return {
  status: 'done', tp: g.tp, num_gpus: numGpus, isls: g.isls,
  models: g.models, by_model: byModel,
  raw: measured.map(x => ({ variant: x.variant.id, model_tag: x.mt, ran: x.sweep?.ran, trace: x.sweep?.trace_path, dirs: x.sweep?.result_dirs, notes: x.sweep?.notes, applyFailed: x.applyFailed || false })),
}
