---
description: Create a GitHub Pull Request following SGLang project conventions with proper sections
---

# Create a Pull Request for SGLang

Follow these steps precisely to create a well-formatted PR.

## Step 1: Gather context

Run these in parallel:
- `git status` to check for uncommitted changes
- `git branch --show-current` to get current branch
- `git log --oneline main..HEAD` to see all commits on this branch
- `git diff main...HEAD` to see all changes vs main
- Check if the branch has a remote tracking branch: `git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null`

If there are uncommitted changes, warn the user and ask if they want to commit first (suggest using `/commit`).

## Step 2: Push the branch

If the branch is not pushed or is behind the remote:
- Push with: `git push -u origin <branch-name>`

## Step 3: Analyze the changes

Read through all the diffs and commits to understand:
- What was the motivation / bug being fixed / feature being added?
- What specific files and code were modified?
- Does this change affect model accuracy (kernel changes, model forward code)?
- Does this change affect inference speed / performance?

## Step 4: Determine the category of change

Based on the analysis, determine if this PR:
- **Affects accuracy**: Changes to kernels, model forward code, attention mechanisms, etc. -> Need accuracy test results
- **Affects performance**: Changes to scheduling, batching, memory management, kernels, etc. -> Need benchmark results
- **Neither**: Documentation, CI, refactoring with no behavioral change, etc.

## Step 5: Create the PR

Use `gh pr create` with the following template. Fill in each section based on your analysis.

If the user provided `$ARGUMENTS`, use that as the PR title. Otherwise, craft a concise title (under 70 chars) starting with `[AMD]`.

```bash
gh pr create --title "[AMD] <concise title>" --body "$(cat <<'EOF'
## Motivation

<Describe the purpose and goals based on the actual changes. Explain what problem this solves or what feature this adds.>

## Modifications

<Bullet list of concrete changes to files/functions/behavior.>

## Accuracy Tests

<If accuracy-related: describe what accuracy tests should be run, or paste results if available.>
<If not accuracy-related: write "N/A - this change does not affect model outputs.">

## Benchmarking and Profiling

<If performance-related: describe what benchmarks should be run, or paste results if available.>
<If not performance-related: write "N/A - this change does not affect inference speed.">

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
EOF
)"
```

## Step 6: Report back

After the PR is created, show the user:
- The PR URL
- A summary of what was included
- Remind them about the review process (ping merge oncalls, get CODEOWNER approvals, trigger CI)
- If accuracy-related: remind them to run accuracy tests and post results
- If performance-related: remind them to run benchmarks and post results

## Important Rules

- NEVER force push
- NEVER create PRs to branches other than `main` unless the user explicitly asks
- Always use the full template with all sections filled in
- The PR title should start with `[AMD]` and be under 70 characters
