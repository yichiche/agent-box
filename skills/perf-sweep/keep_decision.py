#!/usr/bin/env python3
"""Gate K — KEEP decision for a kernel/op optimization (see memory/workflows/gates.md).

The keep decision is made on the *served-trace per-op time*, run TWICE for
consistency — NOT on aggregate e2e ≥5% (which Amdahl-hides small real wins). A change
is banked into the ledger toward the cumulative ship goal (Gate S) when:

  * the target op is >= KEEP_THRESHOLD% faster in the served trace, AND
  * the two baseline runs agree within CONSISTENCY_TOL%, likewise the two after runs, AND
  * the e2e verdict is NOT a REGRESSION (we drop "e2e must improve", never "e2e must
    not get worse").

The per-op times come from the served profiling traces via `/compare-kernels --budget`
(which works on decode traces too). Feed them here as µs.

Outcomes: KEEP | REJECT_E2E_REGRESSION | REJECT_INSUFFICIENT | RESAMPLE_INCONSISTENT

Usage:
  keep_decision.py --target "moe stage-1" --baseline 76.6 77.1 --after 52.0 51.4 \
      [--e2e-verdict RESULT_DIR/verdict.json] [--ledger ~/qwen3.5-mxfp4/keep_ledger.json] \
      [--keep-threshold 30] [--consistency-tol 5] [--e2e-share 0.21]
  keep_decision.py --ledger ~/.../keep_ledger.json --summary
"""
import argparse, json, os, sys, time


def _consistent(a, b, tol_pct):
    m = (a + b) / 2.0
    if m == 0:
        return a == b, 0.0
    spread = abs(a - b) / m * 100.0
    return spread <= tol_pct, spread


def _e2e_is_regression(verdict_path):
    """Returns (is_regression, verdict_str_or_None)."""
    if not verdict_path or not os.path.isfile(verdict_path):
        return False, None
    try:
        v = json.load(open(verdict_path)).get("verdict")
    except Exception:
        return False, None
    return (v == "REGRESSION"), v


def decide(target, baseline, after, keep_threshold, consistency_tol,
           verdict_path=None, e2e_share=None):
    b1, b2 = baseline
    a1, a2 = after
    base_ok, base_spread = _consistent(b1, b2, consistency_tol)
    aft_ok, aft_spread = _consistent(a1, a2, consistency_tol)
    base = (b1 + b2) / 2.0
    aft = (a1 + a2) / 2.0
    improve_pct = ((base - aft) / base * 100.0) if base else 0.0
    e2e_regr, e2e_verdict = _e2e_is_regression(verdict_path)

    if not (base_ok and aft_ok):
        outcome = "RESAMPLE_INCONSISTENT"
    elif e2e_regr:
        outcome = "REJECT_E2E_REGRESSION"
    elif improve_pct >= keep_threshold:
        outcome = "KEEP"
    else:
        outcome = "REJECT_INSUFFICIENT"

    entry = {
        "target": target,
        "baseline_us": round(base, 3), "after_us": round(aft, 3),
        "improve_pct": round(improve_pct, 2),
        "baseline_spread_pct": round(base_spread, 2),
        "after_spread_pct": round(aft_spread, 2),
        "e2e_verdict": e2e_verdict,
        "outcome": outcome,
        "keep_threshold": keep_threshold, "consistency_tol": consistency_tol,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # optional: banked e2e-headroom estimate for the cumulative goal (Gate S is the truth)
    if e2e_share is not None and outcome == "KEEP":
        entry["est_e2e_pct"] = round(e2e_share * (improve_pct / 100.0) * 100.0, 3)
    return entry


def append_ledger(ledger_path, entry):
    data = []
    if os.path.isfile(ledger_path):
        try:
            data = json.load(open(ledger_path))
        except Exception:
            data = []
    data.append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    json.dump(data, open(ledger_path, "w"), indent=2)


def summarize(ledger_path):
    if not os.path.isfile(ledger_path):
        print(f"no ledger at {ledger_path}"); return
    data = json.load(open(ledger_path))
    kept = [e for e in data if e.get("outcome") == "KEEP"]
    print(f"=== KEEP ledger ({ledger_path}) — {len(kept)} banked / {len(data)} evaluated ===")
    total_est = 0.0
    for e in kept:
        est = e.get("est_e2e_pct")
        total_est += est or 0.0
        print(f"  ✅ {e['target']:<28} {e['baseline_us']:>8.1f}→{e['after_us']:>8.1f}us "
              f"({e['improve_pct']:+.1f}%)" + (f"  est e2e {est:+.2f}%" if est is not None else ""))
    if total_est:
        print(f"  Σ estimated e2e headroom (banked): ~{total_est:.2f}%  "
              f"— NOTE: Gate S must re-measure the STACK together (interactions).")


def main(argv=None):
    p = argparse.ArgumentParser(description="Gate K keep decision + cumulative ledger")
    p.add_argument("--target")
    p.add_argument("--baseline", nargs=2, type=float, metavar=("RUN1", "RUN2"))
    p.add_argument("--after", nargs=2, type=float, metavar=("RUN1", "RUN2"))
    p.add_argument("--e2e-verdict", default=None)
    p.add_argument("--ledger", default=None)
    p.add_argument("--keep-threshold", type=float, default=30.0)
    p.add_argument("--consistency-tol", type=float, default=5.0)
    p.add_argument("--e2e-share", type=float, default=None,
                   help="op's fraction of e2e (0..1) for a banked headroom estimate")
    p.add_argument("--summary", action="store_true", help="print the ledger and exit")
    a = p.parse_args(argv)

    if a.summary:
        if not a.ledger:
            p.error("--summary needs --ledger")
        summarize(a.ledger); return 0

    if not (a.target and a.baseline and a.after):
        p.error("need --target, --baseline R1 R2, --after R1 R2")

    entry = decide(a.target, a.baseline, a.after, a.keep_threshold,
                   a.consistency_tol, a.e2e_verdict, a.e2e_share)
    print(json.dumps(entry, indent=2))
    print(f"OUTCOME: {entry['outcome']}  ({entry['target']}: "
          f"{entry['baseline_us']}→{entry['after_us']}us, {entry['improve_pct']:+.1f}%)")
    if a.ledger:
        append_ledger(a.ledger, entry)
        print(f"appended to ledger: {a.ledger}")
    # exit code: 0=KEEP, 3=reject/resample (non-fatal, for scripting)
    return 0 if entry["outcome"] == "KEEP" else 3


if __name__ == "__main__":
    sys.exit(main())
