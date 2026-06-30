# Validate Workflow (PR gate)

End-to-end validation before `/pr`. Skill: `skills/validate/SKILL.md` (`/validate`).

## Inputs (agent must resolve or ask once)

1. **Server script** — from [[../models/INDEX]] or user-provided `run_*.sh`
2. **Client script** — matching client for that model
3. **Accuracy threshold** — 0.88 (DSv4), 0.92 (Qwen3.5 MXFP4), user override
4. **SGLANG_ROOT** — detect dynamically:
   ```bash
   python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])"
   ```

## Steps (summary)

1. **Baseline benchmark** — revert change (stash/checkout), run client
2. **Accuracy test** — GSM8K or model-specific; must pass threshold
3. **Profiling** — `/generate-profile` or trace capture at representative conc
4. **After benchmark** — restore change, rerun client
5. **Compare** — table for PR "Benchmarking" section

## Branch hygiene

Never commit on SGLang `main`. See [[../gotchas/sglang-branch-hygiene]] and `skills/_shared/repo-config.md`.

## Pipeline mode

`/kernel-fusion-pipeline` sets `pipeline_mode: true` — skip interactive questions, use CONFIG fields.
