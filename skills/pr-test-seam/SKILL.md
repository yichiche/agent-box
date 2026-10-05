---
name: pr-test-seam
description: >-
  Review a PR's tests using Matt Pocock's deep-module rule: identify the
  interface and seam changed by production code, verify tests call that
  interface, and detect tests that step past the seam or reimplement the
  production formula. Use when the user says '/pr-test-seam', asks whether a
  unit test calls the real interface, or whether tests duplicate implementation.
category: deliver
---

# /pr-test-seam — where tests enter the module

This skill answers only test design. Use `/pr-code-path` for common-path,
NVIDIA, and hardware-scope analysis.

## Run

```bash
python3 ~/agent-box/skills/pr-test-seam/test_seam.py 39575
python3 ~/agent-box/skills/pr-test-seam/test_seam.py https://github.com/sgl-project/sglang/pull/39575
```

The script lists new tests, production functions they call, production
functions replaced by lambdas, and possible copied expressions. These are
facts, not the verdict.

Use [examples.md](examples.md) as the calibration case. When the user asks for
a visual result in Cursor, render the test-entry map as a separate canvas; do
not add it to `/pr-code-path`'s hardware-path visualization.

## Vocabulary

Use Matt Pocock's terms:

- **module** — implementation hidden behind one interface
- **interface** — everything a caller must know to use the module correctly
- **seam** — where the interface lives

The interface is the test surface. Callers and tests should cross the same seam.
A test that must change whenever implementation details move is testing past the
interface.

## Review

### 1. Name the behavior under test

State the regression in caller-visible terms. Then identify the highest
production interface whose single call should exercise it.

If the bug is ordering in a caller—snapshot before store, validate before
commit, acquire before publish—the caller is the seam. A test that manually
performs those steps and calls the final helper does not cover the ordering.

### 2. Classify every test

**Interface test**
- calls the production interface at the changed seam
- asserts an observable result, error, or invariant
- survives an internal refactor

**Past the seam**
- calls a private helper below the interface
- manually performs steps that production is responsible for ordering
- can stay green if the caller forgets or reorders one of those steps

**Reimplements**
- computes the expected answer with the same formula or control flow as
  production
- production and test can share the same mistake

**Adapter at an internal seam**
- replaces a collaborator such as a pool, filesystem, or kernel
- acceptable when the production interface still performs the orchestration
- not evidence that an outer seam was exercised

**Fixture fallout**
- adds a field, default, or lambda only because shared production setup changed
- keeps an old test running but does not test the new behavior

### 3. Judge expected values independently

Good expected values:
- literal values derived from the behavior specification
- invariants over the production function's own outputs
- one production implementation compared with another

Not independent:
- the same clamp, index, mask, or arithmetic expression copied from production
- a helper in the test that recreates the new algorithm

### 4. State the missing test

For each `Past the seam` result, name one concrete interface-level test:

> Call `[interface]` with `[trigger]` and assert `[observable result]`, allowing
> production itself to perform `[ordering or hidden steps]`.

## Report

```
**Test-seam conclusion:** Complete | Mixed | Past the seam | Reimplements | No test
**Changed behavior:** [caller-visible regression]
**Correct seam:** [production interface callers use]

## Test entry map
<ASCII path from each test to production>

## Findings
| Test | Seam it calls | Expected value source | Verdict |
|---|---|---|---|

## Missing interface test
[one concrete test description, or “None”]
```

## Related

| Skill | Question it answers |
|---|---|
| [`/pr-code-path`](../pr-code-path/SKILL.md) | Common path, NVIDIA impact, and affected hardware |
| [`/sglang-pr-review`](../sglang-pr-review/SKILL.md) | Is the implementation correct? |
