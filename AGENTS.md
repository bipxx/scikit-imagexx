# Repository workflow

- Merge and push validated, reviewable increments back to `main` promptly; do not accumulate completed feature work in worktrees.
- After merging, remove obsolete worktrees with `git worktree remove`, first checking that their tracked and untracked source changes are clean and their commits are ancestors of `main`.
- Keep build outputs in ignored build directories. Remove obsolete build caches after verification; retain active environments, running-job files, reproducible baseline packages, raw benchmark results, and unfinished source work.
- Resolve and check cleanup paths against this workspace before recursive deletion. Never delete the entire `build/` directory: it also holds active worktrees.
- Keep performance measurements isolated from builds and tests. Do not remove or rebuild files used by a running benchmark.

# Task closeout and waste review

- After each bounded task reaches a terminal outcome, record result/evidence, avoidable work and reuse, and the next-plan correction with owner/trigger (or "reviewed, no change") in its existing handoff/checkpoint. Distinguish useful rejected experiments from preventable setup or duplicate work. Use observed costs when available; do not rerun work merely to measure waste.
- Before the next dependent task, update the affected action in `coverage/EXECUTION_PLAN.md` and existing operational task fields when the review changes execution. Include material process corrections in the validated increment; preserve historical evidence and use existing records rather than adding a reporting framework.

- At a task transition, reconcile actual agent/job status and evidence before dispatch. Use `followup_task` to resume a completed agent; a queued message alone does not start work. Retire terminal dispatch instructions when updating the next gate.

- Finalize receipt paths, labels and closeout before handing off a hash. Handed-off bytes are immutable; corrections require a versioned successor with the predecessor retained. Review only the affected metadata delta and do not replay unchanged scientific evidence.
- Inspect branch ancestry before merging a reference packet into main. If it descends from unfinished integration work, restore only its independently reviewed files after confirming their dependencies are already present; do not import the private implementation accidentally. A failed merge or check must stop dependent mutations. Commit only after conflict markers and index checks pass; verify the final shipping source projection against the validated candidate, especially after a reverted merge.

- Before hosted preparation, run changed source-only benchmark/metadata guards on both retained supported Python versions; source-plan commands do not execute those guards. For new cross-version code-immutability checks, compare pinned source bytes or parse the baseline and candidate on the same interpreter. Frozen AST dumps include interpreter schema and must not be treated as portable source hashes. Preserve historical pins and failures.

- Once required readiness and the affected next gate are closed, dispatch eligible hosted work promptly; batch unrelated publication, broad ledger refresh and next-family reviews while it runs. A single explicitly assigned owner may author shared integration in an isolated candidate; root retains independent admission and main delivery.

# Engineering judgment and simplicity

- Match defensive complexity to a concrete risk. Address ordinary correctness issues and realistic concurrency or security risks within the agreed scope. Before adding substantial machinery for a speculative or very unlikely edge case, explain the failure scenario, evidence for its likelihood and impact, simpler alternatives, and estimated code and maintenance cost; obtain the user's approval before expanding scope. Do not use this rule to omit necessary input validation, standard security controls, or requirements already authorized.
- Validate and normalize external data at the boundary, then pass explicit, stable types and established invariants through internal code. Avoid repeating the same checks at every layer. Revalidate at a new trust boundary or when mutation, callbacks, external state, or another documented condition can invalidate an earlier check; make that reason clear.
- Prefer straightforward local code for one-off logic. Extract a helper when it is reused, names a meaningful operation, isolates a genuinely complex section, or materially improves readability or testing. Avoid both unnecessary indirection and oversized functions that mix responsibilities.
- Catch exceptions when there is a specific recovery action, useful domain-level translation, required cleanup, or a need to preserve diagnostic evidence before propagating failure. Otherwise let errors propagate. Prefer scoped resource management and `finally` for cleanup; do not swallow failures or add catch-and-rethrow blocks without value.
- Keep a cohesive operation together when that makes its flow easy to follow. Split it when distinct responsibilities, nesting, or complexity make the code harder to understand or change. Optimize for clear behavior and maintenance, rather than a fixed function count or line limit.
