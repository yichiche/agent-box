---
name: pr-test-seam
description: >-
  Judge whether tests for an sglang PR enter through the production interface
  that owns the changed behavior. Use when the user says '/pr-test-seam', or
  when /pr-merge-triage or /sglang-pr-review needs the test-seam conclusion.
  Facts come from test_seam.py. The conclusion is complete, mixed, past the
  seam, reimplements, or no test.
category: deliver
---

# /pr-test-seam — do the tests enter through the owning interface

The interface is the test surface. A test that must change whenever a private
step moves is testing past that interface.

## Facts

```bash
python3 ~/agent-box/skills/pr-test-seam/test_seam.py N
python3 ~/agent-box/skills/pr-test-seam/test_seam.py N --json
```

The script lists new tests, production functions they call, production
functions replaced by lambdas, and possible copied expressions. These are
facts, not the verdict. The calibration case is PR 39575, recorded in
`~/agent-box/skills/sglang-pr-review/examples.md`.

## Terms

- **module** — implementation hidden behind one interface
- **interface** — everything a caller must know to use the module correctly
- **seam** — where the interface lives

## 1. Name the behavior under test

State the regression in caller-visible terms. Then identify the highest
production interface whose single call should exercise it.

If the bug is ordering in a caller — snapshot before store, validate before
commit, acquire before publish — the caller is the seam. A test that manually
performs those steps and calls the final helper does not cover the ordering.

## 2. Classify every test

**Interface test**
- calls the production interface at the changed seam
- asserts an observable result, error, or invariant
- survives an internal refactor

**Past the seam**
- calls a private helper below the interface
- manually performs steps that production is responsible for ordering
- can stay green if the caller forgets or reorders one of those steps

**Reimplements**
- computes the expected answer with the same formula or control flow as production
- production and test can share the same mistake

**Adapter at an internal seam**
- replaces a collaborator such as a pool, filesystem, or kernel
- acceptable when the production interface still performs the orchestration
- not evidence that an outer seam was exercised

**Fixture fallout**
- adds a field, default, or lambda only because shared production setup changed
- keeps an old test running but does not test the new behavior

## 3. Judge expected values independently

Good expected values:
- literal values derived from the behavior specification
- invariants over the production function's own outputs
- one production implementation compared with another

Not independent:
- the same clamp, index, mask, or arithmetic expression copied from production
- a helper in the test that recreates the new algorithm

## 4. State the missing test

For each `Past the seam` result, name one concrete interface-level test:

> Call `[interface]` with `[trigger]` and assert `[observable result]`, allowing
> production itself to perform `[ordering or hidden steps]`.

## Conclusion

Use exactly one:

| Conclusion | When |
|---|---|
| `complete` | every changed behavior is covered by an interface test, and expected values are independent |
| `mixed` | at least one interface test exists, and at least one changed behavior is past the seam, reimplemented, or untested |
| `past_the_seam` | the tests call below the owning interface |
| `reimplements` | expected values copy the production formula or control flow |
| `no_test` | no test covers the changed behavior |

## Report

When the user invoked `/pr-test-seam` directly, the chat reply is the conclusion,
the changed behavior, the correct seam, the test entry map, and the missing
test. No table is required.

When `/pr-merge-triage` invoked this skill, do not print that report. Write
`/tmp/pr-N-test-seam.json`:

```json
{
  "conclusion": "mixed",
  "changed_behavior": "one sentence a caller can observe",
  "correct_seam": "production function callers use",
  "missing_test": "Call [interface] with [trigger] and assert [result], letting production perform [hidden steps]."
}
```

`missing_test` is required unless `conclusion` is `complete`. For `complete`,
set `missing_test` to `""`.

PR 39575's conclusion is `mixed`. The write-plan tests call `_qsa_write_plan`.
The recompress tests call `_overwrite_cross_prefix_groups` after the fixture
has already snapshotted and overwritten the ring. The owning seam is
`_forward_impl`.
