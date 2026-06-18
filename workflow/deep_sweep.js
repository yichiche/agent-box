export const meta = {
  name: 'deep-sweep',
  description: 'Sequentially run implement_deep.js over a perf-ranked list of deep optimization candidates (author or tune mode), one at a time (each rebuilds the shared aiter install + contends for the GPU, so parallel is unsafe). Aggregates baseline→best, correctness, and perf-gate verdict per candidate. Report-only — leaves each winning diff applied; does NOT open PRs.',
  whenToUse: 'Run after discovery to auto-execute the Deeper candidates by expected savings without stopping between each. Override args.candidates to change the list/order; args.gpus to pin.',
  phases: [
    { title: 'Sweep', detail: 'For each candidate (highest expected savings first): run implement_deep, collect verdict' },
  ],
}

const IMPLEMENT_DEEP = '/home/yichiche/agent-box/workflow/implement_deep.js'

// Perf-ranked Deeper candidates from discovery (qwen3.5-mxfp4, TP2, MI355x).
// #1 (fused softmax+topk+sort routing) is run separately; this sweep is #2→#5.
const DEFAULT_CANDIDATES = [
  {
    rank: 2,
    target_op: 'fused_shared_expert_gate_combine — fuse shared-expert gate GEMV + sigmoid + broadcast-mul + add-into-routed into ONE kernel',
    mode: 'author',
    kernel_type: 'triton',
    est_savings_us: 10,
    workload: 'Qwen3.5-397B-A17B-MXFP4 MoE decode, M=4 (conc4), TP2 MI355x. Replaces HIP unfused: shared gate GEMV (Cijk...MT1x2x512 ~8.1us) + _sigmoid_gate_mul_broadcast_kernel (~4.1us) + at::native::CUDAFunctor_add (~4.2us) ≈ 16us, 3 launches + an intermediate shared_output round-trip. Reference: B200 finalizeKernel<bf16,bf16,2,true> 4.8us (true = shared/bias combine). Model the new kernel on SGLang Triton _fused_gate_sigmoid_mul_add semantics. SGLang dispatch site: python/sglang/srt/models/qwen2_moe.py. No existing aiter op (write-new). Correctness: allclose final_hidden_states vs the unfused reference path.',
  },
  {
    rank: 3,
    target_op: 'fused_attn_segment_reduce — fold the split-KV partial-softmax reduction INTO the decode attention kernel (eliminate the separate reduce_segments launch)',
    mode: 'author',
    kernel_type: 'triton',
    est_savings_us: 8,
    workload: 'Qwen3.5 full-attention decode, 8k context, 128 segments, TP2 MI355x. Replaces HIP: kernel_unified_attention_3d (~11.5us) + separate reduce_segments (~9.4us) = 20.9us across two launches. Reference: B200 fmhaSm100f PagedKvCausal MultiCtasKv does the reduce in-kernel in one ~13.8us launch. Edit aiter/ops/triton/attention/unified_attention.py to do the split-k partial-softmax reduction in-kernel (MultiCtasKv-style). SGLang dispatch: python/sglang/srt/layers/attention/aiter_backend.py. CAUTION: unified_attention has many callers — KEEP the separate-reduce path for num_segments==1 / other callers; only the multi-segment decode path changes. Correctness: allclose attention output vs the current two-launch path.',
  },
  {
    rank: 4,
    target_op: 'Qwen3.5 MXFP4 MoE expert GEMM (a4w4 blockscale / 2-stage gemm_moe) — tune the GEMM configs for the decode shapes',
    mode: 'tune',
    kernel_type: 'auto',
    est_savings_us: null,
    workload: 'Qwen3.5-397B-A17B-MXFP4 MoE expert GEMM at decode M=4 (and the prefill shapes), TP2 MI355x. Detect whether the dispatched path is a4w4_blockscale (csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py) or the 2-stage fused MoE (csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py). Ensure the model MoE GEMM shapes are in the matching untuned CSV, then run the OFFICIAL tuner and compare the resulting tuned config vs the current one. This is the safest high-leverage tune (official tuner, no hand-written kernel).',
  },
  {
    rank: 5,
    target_op: 'fused_recurrent_gated_delta_rule — tune the GDN decode recurrent kernel',
    mode: 'tune',
    kernel_type: 'auto',
    est_savings_us: 5,
    workload: 'Qwen3.5 GatedDeltaNet decode recurrent update, TP2 MI355x. HIP fused_recurrent_gated_delta_rule ~8.6us vs B200 cutlass GDN decode ~3.7us (2.3x). param_search the Triton kernel knobs (BLOCK sizes, num_warps, num_stages, vectorization, occupancy). Source under aiter/ops/triton/ (gated_delta / fused_recurrent). Use the matching op_tests test + bench harness.',
  },
]

const candidates = Array.isArray(args?.candidates) && args.candidates.length ? args.candidates : DEFAULT_CANDIDATES
const gpus = args?.gpus || '6,7'
const perfGatePct = args?.perf_gate_pct ?? 1.0
const maxRounds = args?.max_rounds ?? 8

// Snapshot the clean baseline code state ONCE. Every candidate is measured from THIS exact state and reverts
// back to it (implement_deep runs with keep_best=false → save-patch-then-revert), so each operation's delta is
// independently attributable against the same baseline — no commit pollution across operations.
phase('Snapshot')
const snapshot = await agent(`
Guarantee a CLEAN baseline of the editable aiter install before a sequential optimization sweep. PIPELINE_MODE — no questions.
  AITER=$(python3 -c "import aiter,pathlib;print(pathlib.Path(aiter.__file__).resolve().parent.parent)")   # fallback /sgl-workspace/aiter
If 'git -C $AITER status --porcelain' shows stray edits from a crashed prior run, restore tracked files
(git -C $AITER checkout -- .). Do NOT touch flydsl_cache/ or jit/*.so (those rebuild on demand).
Return one line: "<AITER_ROOT> @ <short-commit> clean".`, { label: 'snapshot-baseline', phase: 'Snapshot' })
log(`Clean baseline: ${snapshot || 'unknown'}`)

phase('Sweep')
log(`Deep sweep: ${candidates.length} candidate(s) sequentially on GPU(s) ${gpus}, perf_gate=${perfGatePct}% — isolated (keep_best=false, revert between)`)

const results = []
for (let i = 0; i < candidates.length; i++) {
  const c = candidates[i]
  const tag = `#${c.rank ?? i + 1}`

  // Guard: ensure a clean tree before this candidate (implement_deep reverts itself, but a crashed run might not).
  await agent(`Restore the editable aiter install to a clean tracked tree before the next optimization. PIPELINE_MODE.
AITER=$(python3 -c "import aiter,pathlib;print(pathlib.Path(aiter.__file__).resolve().parent.parent)").
Run: git -C $AITER checkout -- . . Leave flydsl_cache/ and jit/*.so alone. Confirm 'git -C $AITER status --porcelain'
has no tracked-file edits and return one line.`, { label: `clean:${tag}`, phase: 'Sweep' })

  log(`=== ${tag} START (${c.mode}) — ${String(c.target_op).slice(0, 80)} ===`)

  let r = null
  try {
    r = await workflow({ scriptPath: IMPLEMENT_DEEP }, {
      target_op: c.target_op,
      mode: c.mode || 'auto',
      kernel_type: c.kernel_type || 'auto',
      workload: c.workload || '',
      gpus,
      perf_gate_pct: perfGatePct,
      max_rounds: c.max_rounds ?? maxRounds,
      keep_best: false,   // isolation: save the winning diff as a patch, then revert to the shared clean baseline
    })
  } catch (e) {
    log(`${tag} ERROR: ${e && e.message ? e.message : String(e)}`)
  }

  const verdict = !r ? 'error/null'
    : r.status === 'no_win' ? 'no win (baseline restored)'
    : r.ship_ready ? `SHIP-READY ${r.improvement_pct?.toFixed?.(1) ?? '?'}%`
    : `win below gate / not applied (${r.improvement_pct?.toFixed?.(1) ?? '?'}%)`
  log(`=== ${tag} DONE — ${verdict} | baseline=${r?.baseline_latency_us ?? '?'}us best=${r?.best_latency_us ?? '?'}us | report=${r?.report_path ?? 'n/a'} ===`)

  results.push({
    rank: c.rank ?? i + 1,
    target_op: c.target_op,
    mode: c.mode,
    status: r?.status,
    ship_ready: r?.ship_ready ?? false,
    baseline_latency_us: r?.baseline_latency_us ?? null,
    best_latency_us: r?.best_latency_us ?? null,
    improvement_pct: r?.improvement_pct ?? null,
    repo_state: r?.repo_state,
    report_path: r?.report_path,
    patch_path: r?.patch_path || null,   // re-appliable winning diff (saved even though reverted)
    diff: r?.diff || null,
  })
}

const shipReady = results.filter(r => r.ship_ready)
log('=== Deep sweep complete ===')
log(`Ship-ready: ${shipReady.length}/${results.length} — ${shipReady.map(r => `#${r.rank}(${r.improvement_pct?.toFixed?.(1)}%)`).join(', ') || 'none'}`)

return { gpus, perf_gate_pct: perfGatePct, count: results.length, ship_ready: shipReady.length, results }
