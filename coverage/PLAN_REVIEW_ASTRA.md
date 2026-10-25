# Astra review of the implementation plan

Planning-only audit, 2026-10-24–25. Source baseline:
`12fad3488178f52d52d8b4fd161e59790fe63d17` on main, with the frozen candidate
and reference summaries listed in [the execution plan](EXECUTION_PLAN.md).
The implementation goal remains paused. This review inspected source, ledger,
workflow definitions, small proof summaries, and retained reports; it did not
build, call image algorithms, benchmark, dispatch, merge, push, or independently
requalify the scientific artifacts.

The current evidence discipline has caught real defects: unsafe original-call
boundaries, incorrect dtype preservation, missing SDK oracle arguments, runtime
math dispatch differences, and source/product identity errors. Those protections
should remain. The avoidable cost is treating each new case or small optimization
as another bespoke evidence project and making the integrator the serial owner
of most of that work. The easiest viable path is a small set of reusable native
primitives, a per-API contract, a common acceptance format, and short releases.

## Findings and corrections

| Finding | Local evidence | Consequence | Recommended correction |
|---|---|---|---|
| The schedule drifted away from the accepted source | The old plan still assigned observation of TV run `40087742726` and described 190/191-path work as next, while [TV admission evidence](evidence/tv-single-admission-whole.json) accepts `af7b2af` at 192 paths. The old ledger advertised active workers and `tv-finite-single-admission` validating. | A syntactically valid ready queue could launch obsolete work or repeat a completed stage. | The planning update now records pause/empty runnable assignments and frozen checkpoints. Reconcile historical packet states and dependencies before resuming. Keep historical evidence in an appendix rather than a live handoff. |
| The checker proves structure, not freshness or completeness | `tools/work_queue.py` validates ownership/cycles/proof shape and compares assigned-plus-unassigned counts, but does not inspect actual agent liveness or infer newest accepted sources. At audit it passed 177 packets and offered only `matrix-transform-general-binding` as ready. | Passing `--check` is useful but insufficient to schedule the next release. | Use the checkpoint map and exact source/evidence scope for transitions; maintain an explicit per-API roadmap and a small runnable frontier. Do not mistake 391 unassigned keys for 391 unimplemented algorithms. |
| Root is a potential serial bottleneck | Of 177 historical packets, 88 list the integrator as owner; 41 are `tooling`, four `benchmark_tooling`, and 35 `integration`. [Private review](evidence/current-private-increments-root-review.json) reverified 949 records and about 1.18 GB; [TV public acceptance](evidence/tv-single-admission-whole.json) records 1,669 records and about 854 MB. | This is evidence of a large coordination surface, not a measured time share. Repeated manual replay and growing bespoke tooling can delay useful algorithm work. | Give one common handoff manifest to an independent worker verifier. Root reviews changed code, scope, exceptions, provenance closure, and that receipt. Reuse immutable verified objects by hash; retain fresh execution for changed source. |
| Small changes repeatedly acquire new controllers | The frozen TV admission comparison commit `8dad040` adds four files and 996 lines; its proof records 312 metadata guards and no numeric calls, builds, or timings. Existing comparison and qualification infrastructure already exists. | Administrative progress can be mistaken for numerical progress; another controller expands the future maintenance surface. | Finish the frozen work where it is already near acceptance. For future families, supply case/comparison adapters to common qualification, measurement, retention, and reporting tools. Consolidate prospectively; do not alter historical raw bytes or controllers. |
| A combined candidate couples independent APIs | The 199-path composition joins six max-tree APIs and matrix transform. The predecessor run failed all four jobs, including matrix runtime admission and separate max-tree metadata/SDK faults. `3c87fbf` has 2,001 metadata guards and four empty blocker plans but zero whole-product builds. | An arithmetic-runtime issue in one family can delay an otherwise independent family. | Review this frozen candidate once. If matrix remains the blocker after bounded preflight, create a distinct max-tree-only source and qualify it independently; no acceptance can be borrowed from failed 199. Prefer one coherent family or two truly independent small increments per candidate. |
| Whole-product qualification is on the critical path too early | `.github/workflows/public-qualification.yml` builds four genuine products and `tools/public_qualification.py` executes mixed public groups, corpora, and SDK checks. This is valuable release evidence. | Discovering a local family error only at this stage increases the correction cycle and serializes unrelated work. | Run a real installed-family gate first; retain the complete four-profile qualification before public acceptance. Contract drafting, native work, and timing-corpus design proceed in parallel. Planning-only changes do not need product builds. |
| CPU-specific bitwise policies can expand a small algorithm into a runtime-certification project | [Matrix kernel proof](evidence/matrix-transform-general-kernel.json) scopes 96,000 checks to finite F64 dimensions 2–4 and authenticated Haswell arithmetic. [Profile tooling](evidence/matrix-transform-profile-tooling.json) records distinct default cores and remaining SkylakeX rows. Prepared donor replay is explicitly not real-host admission. | Exact compatibility is sometimes required, but universal instruction-sequence imitation can dominate work unrelated to useful hot-swap accuracy. | Keep existing strict proofs and routes. For future APIs, freeze exact structural/discrete behavior and a mathematically justified floating policy before new candidate outputs. Use portable scalar/SIMD dispatch; require CPU-specific arithmetic only where public semantics demand it. |
| Assertion/metadata counts conceal the actual native frontier | Inventory has 192 native-partial image keys, three metadata functions, four partial classes and one exception provider; `tests/test_api.py` uses declared dtypes and default cases. `_geometric.py` explicitly retains NumPy/SciPy inversion, composition, SVD/Umeyama and Euler providers. | Thousands of metadata checks or a class name cannot demonstrate independent fitting or full arguments. | Report unique APIs and contract partitions separately from numerical executions, metadata, expected errors, providers, and assertions. Track each class's state and solver path, including runtime fields beyond 284 static members. |
| Measurement exists, but slower cases and breadth still need closure | The retained metrics, footprint, component, rank and TV reports contain real measurements and regressions. The TV cursor report retains 129 union regressions; `af7b2af` has no new timing yet. | The problem is not an absence of benchmarks. It is delayed closure of known costs and incomplete per-function coverage. | Prepare cases with contracts; measure each qualified family promptly; allocate one worker to high-cost retained regressions. Preserve the complete corpus and actual source labels. |
| The full scope lacks an executable route per key | `coverage/api.json` contains 477 names across 20 namespace entries, 29 classes and 284 static member paths. The previous plan ended with a broad list of future namespaces. | Data, I/O, stateful classes, and difficult dependency families can remain indefinitely outside actual packets. | Use [477 per-API paths](API_IMPLEMENTATION_PATHS.md) and [machine-readable records](API_IMPLEMENTATION_PATHS.json), with shared primitives, explicit hazards, wave, native frontier, and next action. Start the independent provider/asset/state lane early. |

The repaired planning metadata does not retroactively update all 177 historical
packets. Old packet states, old counts inside immutable proof records, and old
branch names must remain distinguishable from the new runnable frontier. The
number of registered worktrees is also not a deletion list: many preserve
unmerged source, environments, exact products, or unfinished experiments.

## Easiest viable routes by family

The linked roadmap names the route for each individual public key. These
principles explain why those routes are smaller than separate implementations
and qualification systems for every name.

| Family | First viable implementation | Reuse / main compatibility hazard |
|---|---|---|
| Already native-partial functions and aliases | Close one missing dtype/layout/parameter partition at a time; reuse canonical dispatch for true aliases | Existing C++ kernels, wrappers and contracts; preserve dtype/metadata, view/mutation and exceptional behavior. An alias adds a path, not another algorithm. |
| Draw and simple util | Direct integer/scanline/coordinate loops, plus correct array views/copies | Existing array conversion and shape facilities; clipping, order, negative indices, overlap and RNG semantics matter more than elaborate scheduling. |
| Exposure, thresholds, reductions, simple measurements | Shared histograms, prefix sums, reductions, moments and explicit stable ties | Reuse existing threshold/rank/metrics code where arithmetic agrees. Li callback order and multi-Otsu's floating objective/tie policy make casual reassociation unsafe. |
| Local filters, footprints, morphology, rank | Existing boundary/offset logic and direct typed loops; rolling histogram or separable passes only where equivalent | Footprint orientation, origin, border modes, mask/output overlap and feedback traversal. SIMD arithmetic or independent rows; retain serial order where output feeds later reads. |
| Reconstruction, components, max-tree and topology | Stable source-faithful scalar queue/tree/union-find first | Reuse intake/ordering/connectivity facilities. Parent/traverser/label ties and signed zero can be observable; unsafe upstream memory behavior is never a parity target. |
| Geometry and resampling | Compose native coordinate generation, interpolation, dtype normalization and clipping | Existing typed resampling/projective/integral facilities. Expand classes with a common native fitting layer; low-dimensional linear algebra need not become per-CPU BLAS imitation. |
| FFT, restoration, registration | One C++ FFT/convolution facility plus direct algorithm ports | Pin transform normalization/padding and control all provider threads. Gauss-Seidel/iterative dependencies and stopping semantics limit safe parallelism. |
| Features and multiscale methods | Reuse gradients, Hessians/eigenvalues, pyramids, neighborhoods and descriptor reductions | Keypoint ordering, tie suppression, feature state, border rules and dtype precision; port one coherent method before generalizing its framework. |
| Region measurements and model classes | One label-statistics/moments accumulator; native QR/SVD/least-squares through a vetted facility | Class construction, live views, fitted-state mutation, degeneracy, weights, failed estimates and residual behavior require tests beyond class metadata. |
| Graph and segmentation | Reuse accepted labels/features, then source-faithful queue/graph/sparse-solver code | Connectivity, stable traversal, callback ordering, convergence and randomness. Parallelize independent preparation; do not change observable topology to obtain workers. |
| Data and I/O | Asset manifest plus cache/download/loader behavior; ordinary codec and collection adapters | Licensing/asset identity, dtype/channel metadata, plugin behavior, laziness, iteration and mutation. Measure defined cache states; metadata/asset access does not need artificial SIMD. |
| Future/custom-model and UI adapters | Own the orchestration and state contract; preserve defined user-provider protocols | Arbitrary estimator/callback behavior remains external by definition. It gives no credit for an unimplemented native algorithm or solver. |

These are routes, not evidence of completion. Python glue is appropriate for
binding, object state and ecosystem protocols; required numerical work must
actually reach repository-owned C++ or the declared low-level C++ dependency.
Keeping a NumPy/SciPy/skimage algorithm provider is an explicit remaining gap,
even when its output passes a numerical tolerance.

## Numerical policy and safety boundaries

Freeze the comparator and complete behavioral contract prospectively. Exactness
is required for structure, dtype, discrete outputs/ties, state, mutation, errors,
warnings and callback order. Floating error bounds should depend on dtype,
conditioning and algorithm semantics; a residual or backward-error check can
supplement an output comparison for fitted models. A blanket tolerance cannot
excuse the wrong branch, class state, failed fit, warning, or provider path.

Do not use observed candidate failures to choose new tolerances, alter a frozen
corpus, or learn CPU/fixture-specific results from the oracle. Preserve current
bitwise proofs unchanged. Any changed future compatibility policy needs its own
reviewed rationale and new independent cases; it does not recertify old artifacts.
This policy is especially constrained for iterative thresholds and TV, whose
branching and stopping can change after a tiny arithmetic perturbation.

Keep the existing safe source-intake exclusions. The reconstruction reference's
309 checks per runtime include only 259 actual original Cython cases; metadata
or blocked unsafe inputs are not algorithm executions. The threshold reference
and Li supplement likewise separate original cases, native solver calls,
provider behavior and expected errors. Original undefined memory access should
not be executed merely to increase coverage.

Use the single existing process-global worker policy, default one. Preserve a
scalar path and portable CPU dispatch. Parallel reductions need a declared
ordering/error policy; callbacks and dependent traversals may remain serial.
Provider threads must not multiply native worker counts. SIMD and worker support
are workload-specific design choices, not a demand to parallelize metadata,
lazy I/O, or every node in an inherently dependent algorithm.

## Execution and acceptance recommendations

Use W0 to close the frozen 199, TV, threshold and reconstruction decisions; W1
for cheap APIs and native-partial breadth; W2 for neighborhood/order/connectivity;
W3 for geometry/resampling/solvers/classes; W4 for FFT/multiscale/registration/
restoration/features; W5 for dependent graph/segmentation/model solvers. Lane P
handles providers/assets/I/O/state independently from W1 onward. These are
dependency groups, not barriers that hold every API until a whole wave finishes.

Give root the shared wiring and final acceptance role, with three active workers
on independent writable files. Rotate independent review to a non-author worker.
Use one active packet per worker and a small bounded integration queue. When a
remote job starts, its observer retains the same handle while the worker moves
to another independent contract or kernel. A timeout never means the job failed
or should be restarted.

Every family follows reference contract → real installed native family gate →
exact cumulative product qualification → isolated original comparison. Freeze
timing cases with the contract so benchmark authoring is not another long stage
after integration. Keep full release qualification, fallback-disabled execution,
real SDK consumers, scalar/SIMD coverage, optional-worker parity, source/product
identity and raw retention. Scope intermediate checks by actual dependencies;
do not replace those gates with metadata guards or repeated smoke tests.

After validation, merge/push the reviewable increment promptly on resume and
record any still-pending performance work. Remove a worktree only after the
repository's clean/ancestor/retention checks. Preserve baseline products, raw
results, live-job files and unfinished source; never delete `build/` wholesale.

Track per-API completion and per-class state/solver completion, tested contract
partitions, the native/provider frontier, isolated one-worker coverage, and
retained regressions with absolute cost. Separately track queue age and repeated
qualification failures. This makes the implementation bottleneck visible
without confusing the number of proof files or assertions with progress.

## Audit limits

This review can establish planning contradictions and suggest simpler routes
from inspected code and recorded evidence. It cannot quantify time lost to
administration, prove a new numerical policy safe, establish unmeasured speed,
or accept the frozen 199/TV-controller/threshold/reconstruction handoffs. Those
remain concrete gates in the revised plan. No accepted proof or raw scientific
artifact was changed by this audit.
