---
name: pr-test-seam
description: >-
  Judge whether tests for an sglang PR enter through the production interface
  that owns the changed behavior. Use when the user says '/pr-test-seam', or
  when /pr-merge-triage needs the test-seam conclusion.
  Facts come from test_seam.py. The conclusion is complete, mixed, past the
  seam, reimplements, or no test.
category: deliver
---

# /pr-test-seam — do the tests enter through the owning interface

The interface is the test surface. A test that must change whenever a private
step moves is testing past that interface.

## Facts

```bash
python3 test_seam.py N
```

Run that command once, from the directory that contains this file. The script lists new tests, production functions they
call, production functions replaced by lambdas, and possible copied
expressions. These are facts, not the verdict. The calibration case is PR
39575. Do not open `examples.md`.

`reenters: main` means the test spawns this file again (`--worker`,
`torch.distributed.run`, or `subprocess`) and the production calls on `main`
belong to that test. `oracle: reference` means the expected value is computed
by a helper in the test file. That is reimplementation unless the helper calls
a second production implementation. The facts include the body of each test,
worker, and reference. Do not fetch the test diff or open the test file.

## Terms

Matt Pocock's glossary, from his
[codebase-design](https://github.com/mattpocock/skills/blob/master/skills/engineering/codebase-design/SKILL.md)
skill:

- **module** — implementation hidden behind one interface
- **interface** — everything a caller must know to use the module correctly
- **seam** — where the interface lives

**Seam** is Michael Feathers' term (*Working Effectively with Legacy Code*): a
place where you can alter behaviour without editing in that place. Hiding a
lot of behaviour behind a small interface is John Ousterhout's deep module
(*A Philosophy of Software Design*). Pocock uses that idea, and measures depth
by how much behaviour a caller can exercise, not by lines of code.

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

**Does not count as a test of the behavior**
- `return True`, `assert True`, or any assert that still passes when the behavior changes
- the test checks the platform, the flag, or the kernel result itself and returns
- the test reimplements the kernel, or copies the production formula or control flow

Fold these into `conclusion`. They do not add a checklist row. Do not read or
edit `triage.py`. The table in `/pr-merge-triage` is the score. A coverage
conclusion other than `complete` does not fail Unit Test Quality. A
`return True` test leaves the behavior untested: `no_test`, or `mixed` when
another interface test exists. `official_pass` is also false, because the
assert does not check the result, and that fails the row. A copied kernel is
`reimplements`, or `mixed` when another interface test exists. Neither is
`complete`, and neither fails the row by itself.

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
  "official_pass": true,
  "official_evidence": "unittest covers the change; one scenario per function; the name states the purpose; asserts check the result; tests do not share a server; the file stays under 500s.",
  "official_action": "",
  "conclusion": "mixed",
  "changed_behavior": "one sentence a caller can observe",
  "correct_seam": "production function callers use",
  "missing_test": "Call [interface] with [trigger] and assert [result], letting production perform [hidden steps]."
}
```

`official_pass` is the stage-1 bar from the SGLang contribution guide and
`test/README.md`. Set it true only when all of these hold:

- the change has a corresponding unittest
- the file uses stdlib `unittest`
- each test function covers one scenario
- the name states the purpose
- asserts check the result. `return True` or an assert that cannot fail does not count
- tests clean up and do not affect one another
- small models, a reused server, and the file stays under 500 seconds

`no_test` means `official_pass` is false. `official_action` is then required
and names the concrete test change.

`/pr-merge-triage` passes Unit Test Quality when `official_pass` is true.
A `conclusion` other than `complete` does not fail the row. `missing_test` is
still required unless `conclusion` is `complete`. For `complete`, set
`missing_test` to `""`.

PR 39575's conclusion is `mixed`. The write-plan tests call `_qsa_write_plan`.
The recompress tests call `_overwrite_cross_prefix_groups` after the fixture
has already snapshotted and overwritten the ring. The owning seam is
`_forward_impl`.
