# Repository workflow

- Merge and push validated, reviewable increments back to `main` promptly; do not accumulate completed feature work in worktrees.
- After merging, remove obsolete worktrees with `git worktree remove`, first checking that their tracked and untracked source changes are clean and their commits are ancestors of `main`.
- Keep build outputs in ignored build directories. Remove obsolete build caches after verification; retain active environments, running-job files, reproducible baseline packages, raw benchmark results, and unfinished source work.
- Resolve and check cleanup paths against this workspace before recursive deletion. Never delete the entire `build/` directory: it also holds active worktrees.
- Keep performance measurements isolated from builds and tests. Do not remove or rebuild files used by a running benchmark.
