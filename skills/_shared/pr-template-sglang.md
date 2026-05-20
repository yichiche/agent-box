# SGLang PR Template (HackMD format)

Use this template when creating PRs for the sglang repo (`sgl-project/sglang`).
The draft file is pure HackMD markdown — no YAML frontmatter.

```markdown
## Motivation

<Describe the purpose and goals based on the actual changes.>

## Modifications

<Bullet list of concrete changes to files/functions/behavior.>

## Accuracy Tests

<If accuracy-related: include a table of accuracy test results provided by the user.>
<If not accuracy-related: "N/A - this change does not affect model outputs.">

## Benchmarking and Profiling

<If performance-related: include a table of benchmark results provided by the user.>
<If not performance-related: "N/A - this change does not affect inference speed.">

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [ ] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review Process

1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - `/tag-run-ci-label`, `/rerun-failed-ci`, `/tag-and-rerun-ci`
4. After green CI and required approvals, ask Merge Oncalls to merge.
```
