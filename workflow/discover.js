export const meta = {
  name: 'discover',
  description: 'Wider multi-source kernel-optimization discovery: fan out across trace-diff (HIP vs CUDA), ATOM commits, InferenceX gaps, and CUDA-only ops; synthesize into one ranked, evidence-backed candidate list. Report-only — does NOT implement or ship.',
  whenToUse: 'Run before kernel-fusion-pipeline to widen the candidate pool beyond a single trace diff. Pass args.fileA/fileB (xlsx traces) and optionally args.model/sources/isl/osl.',
  phases: [
    { title: 'Discover', detail: 'Parallel candidate generation per source' },
    { title: 'Synthesize', detail: 'Dedup, rank, and write the candidate report' },
  ],
}

// One optimization candidate, regardless of which source surfaced it.
const CANDIDATE_SCHEMA_ITEM = {
  type: 'object',
  properties: {
    slug: { type: 'string', description: 'snake_case identifier' },
    block_name: { type: 'string', description: 'Human-readable functional block' },
    target_op: { type: 'string', description: 'aiter op / Triton kernel / new kernel to write' },
    kernels: { type: 'array', items: { type: 'string' }, description: 'Kernels involved/replaced' },
    savings_us: { type: ['number', 'null'], description: 'Estimated per-layer savings (us), null if unknown' },
    sglang_file: { type: 'string', description: 'SGLang (or aiter) file to modify' },
    tier: { type: 'string', enum: ['1', '2', '3'], description: '1=Python dispatch, 2=Triton, 3=aiter C++/HIP' },
    source: { type: 'string', enum: ['trace', 'atom', 'inferencex', 'cuda_gap'] },
    evidence: { type: 'string', description: 'Concrete proof: kernel us / commit hash / benchmark gap / CUDA op name' },
    confidence: { type: 'string', enum: ['high', 'med', 'low'] },
    rationale: { type: 'string', description: 'Why this is worth doing and how' },
  },
  required: ['slug', 'block_name', 'target_op', 'sglang_file', 'tier', 'source', 'confidence'],
}

const SOURCE_SCHEMA = {
  type: 'object',
  properties: {
    candidates: { type: 'array', items: CANDIDATE_SCHEMA_ITEM },
    notes: { type: 'string', description: 'What was inspected and any caveats' },
  },
  required: ['candidates'],
}

const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    candidates: { type: 'array', items: CANDIDATE_SCHEMA_ITEM },
    report_path: { type: 'string', description: 'Path the markdown report was written to' },
    summary: { type: 'string' },
  },
  required: ['candidates', 'report_path'],
}

// ── Args / config ──────────────────────────────────────────────────────────

const fileA = args?.fileA
const fileB = args?.fileB
if (!fileA || !fileB) {
  log('ERROR: args.fileA and args.fileB are required (HIP and CUDA/B200 trace xlsx)')
  return { error: 'Missing xlsx paths. Pass args: { fileA: "<HIP>.xlsx", fileB: "<B200>.xlsx" }' }
}
const model = args?.model || 'qwen3.5'
const isl = args?.isl || 8192
const osl = args?.osl || 1024
const tp = args?.tp || 2
const sources = Array.isArray(args?.sources) && args.sources.length
  ? args.sources
  : ['trace', 'aiter', 'atom', 'inferencex', 'cuda_gap']
const outDir = args?.outDir || '~/.kernel-fusion-pipeline'
const reportPath = `${outDir}/discovery.md`
// Provenance: local container repos are the ACTIONABLE ground truth (the trace was produced by, and
// implement/validate run against, this exact code). check_upstream adds a network freshness overlay
// (github diff) that flags upstream-only ops as "requires bump" — off by default (local-first, no auth).
const checkUpstream = args?.check_upstream ?? false
const ghEnv = 'GH_CONFIG_DIR=/home/yichiche/.gh GH_TOKEN=""'

log(`Discovery model=${model} isl=${isl} osl=${osl} sources=[${sources.join(', ')}]`)
log(`HIP trace: ${fileA}`)
log(`CUDA trace: ${fileB}`)

// ── Phase 1: Discover (one agent per enabled source, in parallel) ───────────

phase('Discover')

const aiterOps = '/home/yichiche/aiter/aiter/ops/ or /sgl-workspace/aiter/aiter/ops/'

const SOURCE_PROMPTS = {
  // Concrete fusion candidates straight from the HIP-vs-CUDA trace diff.
  trace: `
You are the TRACE source of a kernel-optimization discovery sweep. PIPELINE_MODE — no questions.
First recategorize:
  python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py --recategorize ${fileA} ${fileB}
Then READ AND FOLLOW the skill file /home/yichiche/agent-box/skills/compare-kernels/SKILL.md (Steps 0-6)
to compare the HIP trace (${fileA}) against the CUDA/B200 trace (${fileB}). Save the full module-grouped
comparison to ${outDir}/comparison.md.
From the comparison, extract optimization candidates of ALL tiers (1=Python dispatch to an existing aiter
fused op; 2=new/modified Triton kernel; 3=aiter C++/HIP). Verify any named aiter op actually exists under
${aiterOps}; drop invented ops. For each candidate set source="trace" and put the kernel names + us in evidence.`,

  // Portable optimizations from recent ATOM improvement commits.
  atom: `
You are the ATOM source of a kernel-optimization discovery sweep. PIPELINE_MODE — no questions.
Run:
  python3 ~/.claude/skills/atom-progress/scripts/atom_progress.py --model ${model} --isl ${isl} --osl ${osl} --commits --threshold 5
Read AND FOLLOW /home/yichiche/agent-box/skills/atom-progress/SKILL.md for interpretation. Focus on the
"Improvements (>5%)" and "SGLang Relevance" sections. For each SGLang-relevant improvement commit, decide
whether the optimization is portable to SGLang-on-HIP (kernel fusion, dispatch, quant, attention, MoE,
rope/norm, etc.). For portable ones, produce a candidate: pick the likely SGLang file to change, set
source="atom", tier per the change type, and put the commit hash + one-line description in evidence.
Skip ATOM-internal-only changes (note them in notes). If the model has no ATOM config, return empty candidates with a note.
Provenance = NETWORK (dashboard, not the ATOM repo source). If the script fails, fall back to WebFetch on
https://rocm.github.io/ATOM/benchmark-dashboard/#model=DeepSeek-V4-Pro&isl=8192/1024&tab=trends (adjust model).`,

  // Cross-platform/framework gaps: where SGLang-MI355 lags, that's a direction.
  inferencex: `
You are the INFERENCEX source of a kernel-optimization discovery sweep. PIPELINE_MODE — no questions.
Run:
  python3 ~/.claude/skills/inferencex-table/scripts/search_bmk.py --model ${model} --hw mi355x,b200 --framework sglang,vllm,atom --isl ${isl} --osl ${osl} --compare
Read AND FOLLOW /home/yichiche/agent-box/skills/inferencex-table/SKILL.md. Identify where SGLang on MI355x
lags the best comparable result (B200-sglang, or MI355x vllm/atom) — and by how much (ITL / TPOT / throughput).
A large gap points at a subsystem to attack (attention, MoE, gemm, gating, comms). For each meaningful gap,
produce a DIRECTIONAL candidate: source="inferencex", best-guess SGLang file/subsystem, tier per likely fix,
confidence usually "med"/"low" (it's a direction not a proven fusion), and put the concrete gap numbers in evidence.
If no data is found for ${model} on mi355x, return empty candidates with a note.
Provenance = NETWORK (benchmark numbers only). If the script fails, fall back to WebFetch on
https://inferencex.semianalysis.com/ for the same model/hardware.`,

  // Ops that CUDA/B200 has fused but HIP runs unfused AND no existing aiter op covers → write-new candidates.
  cuda_gap: `
You are the CUDA_GAP source of a kernel-optimization discovery sweep. PIPELINE_MODE — no questions.
Compare the CUDA/B200 trace (${fileB}) against the HIP trace (${fileA}) and read the relevant SGLang model/
layer code (CUDA dispatch path vs the is_hip()/_use_aiter path). Find operations that B200/CUDA performs in a
FUSED kernel where HIP runs multiple separate launches AND there is NO existing aiter fused op for it under
${aiterOps} (verify). These are "write-new" opportunities the trace source's Tier-1 list will miss. For each,
produce a candidate: source="cuda_gap", target_op = the new Triton/HIP kernel to write, tier 2 (Triton) or 3
(aiter C++/HIP), evidence = the CUDA fused kernel name + the HIP separate kernels it would replace.`,

  // aiter coverage: aiter already HAS a fused op that the SGLang HIP path is not dispatching to (cheap Tier-1).
  aiter: `
You are the AITER source of a kernel-optimization discovery sweep. PIPELINE_MODE — no questions.

GROUND TRUTH is the LOCAL container aiter — that is what implement/validate run against. Detect and record:
  SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
  AITER=/sgl-workspace/aiter   # (fallback /home/yichiche/aiter)
  git -C $AITER rev-parse --short HEAD    # record this commit in every candidate's evidence

PRIMARY scan (LOCAL, high confidence — actionable right now):
  1. List fused ops in the installed aiter:
       ls $AITER/aiter/ops/fused_*.py ; ls $AITER/aiter/ops/ | grep -iE 'fused|gated_rmsnorm|.*quant|rope|norm'
  2. For each fused op, grep whether the SGLang HIP / _use_aiter path actually calls it:
       grep -rn "<op_name>" $SGLANG_ROOT/python/sglang/srt
  3. An op that EXISTS locally but SGLang does NOT dispatch to on the HIP path = a Tier-1 candidate (just wire
     the dispatch, no rebuild). Cross-check $AITER/op_tests/ to confirm the op is usable and input semantics match.
  4. Candidate: source="aiter", tier="1", target_op=<aiter op>, sglang_file=<where dispatch should go>,
     evidence="aiter op aiter/ops/<file> exists (commit <hash>) but SGLang HIP path at <file:line> does not use it",
     confidence high if op_tests confirm usage, else med.

SECONDARY overlay (NETWORK — run ONLY if check_upstream=${checkUpstream}):
  Find fused ops added UPSTREAM (https://github.com/ROCm/aiter) but absent from the local aiter:
    ${ghEnv} gh api repos/ROCm/aiter/contents/aiter/ops --jq '.[].name' 2>/dev/null    # compare vs local ls
  Any upstream-only fused op that would help = candidate with rationale "requires aiter bump to commit/upstream",
  confidence "low", tier per change type. Do NOT claim these are dispatchable now.

If check_upstream is false, do the PRIMARY local scan only. Return source="aiter" candidates.`,
}

const enabled = sources.filter(s => SOURCE_PROMPTS[s])
const sourceResults = await parallel(enabled.map(s => () =>
  agent(SOURCE_PROMPTS[s], { label: `discover:${s}`, phase: 'Discover', schema: SOURCE_SCHEMA })
    .then(r => ({ source: s, ...(r || { candidates: [], notes: 'agent returned null' }) }))
))

const allCandidates = sourceResults.filter(Boolean).flatMap(r => (r.candidates || []))
for (const r of sourceResults.filter(Boolean)) {
  log(`${r.source}: ${(r.candidates || []).length} candidate(s)`)
}

if (allCandidates.length === 0) {
  log('No candidates from any source.')
  return { status: 'no_candidates', sourceResults }
}

// ── Phase 2: Synthesize (dedup, rank, write report) ─────────────────────────

phase('Synthesize')

const synth = await agent(`
You are the SYNTHESIS step of a kernel-optimization discovery sweep. PIPELINE_MODE — no questions.

You are given raw candidates from up to 4 sources (trace / atom / inferencex / cuda_gap) for ${model}
(ISL=${isl}, OSL=${osl}, TP=${tp}). Do the following:

1. DEDUP: merge candidates that target the same block/op even if surfaced by different sources. When merged,
   keep ALL contributing sources (e.g. source "trace+cuda_gap") and combine their evidence — cross-source
   agreement RAISES confidence.
2. RANK: order by expected impact = estimated savings x confidence, then by tier (prefer lower tier / cheaper).
   Directional inferencex-only candidates with no concrete op rank below proven trace/cuda_gap ones.
3. VALIDATE: drop any candidate whose named aiter op does not exist under ${aiterOps}. Keep write-new (Tier 2/3)
   candidates that legitimately have no existing op.
4. WRITE a markdown report to ${reportPath} with: a ranked table (Rank | Slug | Block | Tier | Source(s) |
   Est. savings | Confidence | Provenance | Evidence), then one short paragraph per top candidate (what to
   change, which file, why it should help, risk). PROVENANCE column = "actionable now" for candidates that
   match the LOCAL container code, or "requires bump" for upstream-only ones (aiter rationale says so). Group
   or clearly flag the "requires bump" ones separately so they are not mistaken for ready-to-implement.
   Use repo-relative paths. No internal /root or /tmp paths in prose.
5. Return the final deduped+ranked candidate list (every field of the item schema; for merged items set
   source to the primary one) plus report_path and a one-line summary.

RAW CANDIDATES:
${JSON.stringify(allCandidates, null, 2)}
`, { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA })

const ranked = (synth && synth.candidates) || []
log(`Synthesized ${ranked.length} ranked candidate(s). Report: ${synth?.report_path || reportPath}`)
const byTier = ranked.reduce((m, c) => { m[c.tier] = (m[c.tier] || 0) + 1; return m }, {})
log(`By tier — T1: ${byTier['1'] || 0}  T2: ${byTier['2'] || 0}  T3: ${byTier['3'] || 0}`)

return {
  model, isl, osl, tp,
  sources: enabled,
  report_path: synth?.report_path || reportPath,
  summary: synth?.summary,
  candidates: ranked,
  per_source: sourceResults.filter(Boolean).map(r => ({ source: r.source, count: (r.candidates || []).length, notes: r.notes })),
}
