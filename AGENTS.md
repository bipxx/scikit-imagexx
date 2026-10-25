# Repository workflow

- Merge and push validated, reviewable increments back to `main` promptly; do not accumulate completed feature work in worktrees.
- After merging, remove obsolete worktrees with `git worktree remove`, first checking that their tracked and untracked source changes are clean and their commits are ancestors of `main`.
- Keep build outputs in ignored build directories. Remove obsolete build caches after verification; retain active environments, running-job files, reproducible baseline packages, raw benchmark results, and unfinished source work.
- Resolve and check cleanup paths against this workspace before recursive deletion. Never delete the entire `build/` directory: it also holds active worktrees.
- Keep performance measurements isolated from builds and tests. Do not remove or rebuild files used by a running benchmark.

# Engineering judgment and simplicity

- Match defensive complexity to a concrete risk. Address ordinary correctness issues and realistic concurrency or security risks within the agreed scope. Before adding substantial machinery for a speculative or very unlikely edge case, explain the failure scenario, evidence for its likelihood and impact, simpler alternatives, and estimated code and maintenance cost; obtain the user's approval before expanding scope. Do not use this rule to omit necessary input validation, standard security controls, or requirements already authorized.
- Validate and normalize external data at the boundary, then pass explicit, stable types and established invariants through internal code. Avoid repeating the same checks at every layer. Revalidate at a new trust boundary or when mutation, callbacks, external state, or another documented condition can invalidate an earlier check; make that reason clear.
- Prefer straightforward local code for one-off logic. Extract a helper when it is reused, names a meaningful operation, isolates a genuinely complex section, or materially improves readability or testing. Avoid both unnecessary indirection and oversized functions that mix responsibilities.
- Catch exceptions when there is a specific recovery action, useful domain-level translation, required cleanup, or a need to preserve diagnostic evidence before propagating failure. Otherwise let errors propagate. Prefer scoped resource management and `finally` for cleanup; do not swallow failures or add catch-and-rethrow blocks without value.
- Keep a cohesive operation together when that makes its flow easy to follow. Split it when distinct responsibilities, nesting, or complexity make the code harder to understand or change. Optimize for clear behavior and maintenance, rather than a fixed function count or line limit.
