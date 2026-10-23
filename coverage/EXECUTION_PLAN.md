# Parallel implementation and validation plan

Implement the complete scikit-image API in optimized C++ with interchangeable
Python calls, per-function parity, optional process-global workers (default one),
and isolated one-worker comparisons against scikit-image 0.26.0. This includes
classes, nested namespaces, data and I/O. Private helpers and compatibility
forwarding do not count as completed public implementations.

The current inventory contains 455 public names: 123 native-partial image entries,
two metadata entries, and 330 upstream-only entries. It includes 29 classes with
284 statically discovered member paths; constructors, state, runtime fields and
additional namespaces still need assessment. These counts describe implementation
classification, not verified parity or performance.

The broad implementation goal is currently **paused**. This revision was reviewed
by three worker agents in parallel, plus the integrator. All three reviews are
complete, and all local implementation handles reached terminal checkpoints.
Uncommitted work, failed trials, caches and exact artifacts are preserved. This
plan is ready to execute when the goal is resumed; inspecting it does not resume
implementation. Existing hosted measurement jobs retain their authoritative run.

## What changed after review

The previous sequence waited for each foundation, wrapper and benchmark in turn.
Only shared integration needs that ordering. Kernel families, Python contracts,
benchmark preparation and class discovery can proceed independently.

Use four active agent slots: one integrator and three workers. Each worker owns
an isolated worktree, explicit files and its own build cache. Frozen candidate
wheels may be shared read-only. The integrator owns shared CMake/binding entry
points, public exports, aliases, registry/inventory changes, installation and
merges. Workers must not revert another worker's changes or edit shared files.

Test and benchmark paths in the inventory establish that files exist. They do
not prove a passing contract, a measured source revision, or a speedup. Track
those separately in [work_items.json](work_items.json) and packet evidence.

## Three-agent review and worker lanes

| Lane | Agent | Independent responsibility | Preserved checkpoint / next handoff |
|---|---|---|---|
| Integrator | `root` | Shared wiring/exports, queue, merges, exact artifacts and hosted measurements | Three optimized metrics wheels built; combined optimized parity and public registration pending |
| Worker 1 | `metrics_reductions` | Kernel/interface review; metrics FP stages | Frozen `070fc64` passes six profiles. Later callback work has four initial boundary failures; its fix is unvalidated. Then take independent footprint utilities or rank contracts |
| Worker 2 | `resize_api` | Contract/dependency review; labeling contracts and kernels | Three runtimes passed 1,600 contract checks with 1,596 native-pending cases each; two source files uncommitted. Freeze that increment, then implement labeling |
| Worker 3 | `resize_bench` | Scheduling/benchmark review; dense footprints and decomposition | Contracts `54d416d` merged. Dense source is uncommitted; fixes after eight initial failures are unvalidated. Repair retained build dependencies, validate dense work, then decomposition |

The agent names are reused from earlier packets; they do not describe their
current assignments. The recorded three-agent review and safe checkpoints are
visible with `tools/work_queue.py --agents`. Live process state is checked with
the collaboration agent tools, rather than inferred from this ledger.

All three reviewers identified the same avoidable dependency: independent
decomposition work waited for dense-kernel completion. The revised queue releases
it from a versioned, hashed atom interface and requires joint kernel parity at
integration. Component-consumer original contracts are also independent of native
labeling. Their implementation waits for the frozen labeling interface.

The read-only planning reviewer completed its audit. Native half identity
(`aed3349`) and public resize/rescale (`21ac60b`) are merged. The exact registered
wheels passed 9,968 tests in each of six runtime/SIMD profiles, and the 373-case
corpus passed untimed checks on all three runtimes at workers one and four.
Performance run [39957875987](https://github.com/bipxx/scikit-imagexx/actions/runs/39957875987)
uses source `66e5f78`; all preparation and both Linux measurement jobs passed,
while Windows measurement remains live at this snapshot. Observe this same run;
do not restart it because of a queue or timeout. All-profile performance acceptance
remains pending.

The private metrics baseline (`03581ae`, integrated as `99abcfc`) passed 4,082
tests in each of six profiles through the actual shared `_core`. It is merged as
`147bd8b`; original worker ancestry is retained by `1945c2e`. Follow-up `070fc64`
fixes buffered reductions and default-policy details, passing six 4,308-case
profiles plus native sanitizers. Stateful callback policies and exact combined
optimized wheels remain registration gates. Private parity is scoped evidence.
Metrics benchmark tooling (`6c74563`, merged as `d94ef6c`) has a fixed 1,231-case
corpus, including 19 expected errors; no metrics timing job has been dispatched.
Transform class contracts (`6fa7f0a`, merged as `ad66094`) passed 228 reference and
strict-rejection checks per runtime, with 224 native-pending cases. Class
implementation remains pending. Completed worktrees are removed only after
evidence archiving and clean/ancestor checks.

## Dependency and merge order

```mermaid
flowchart LR
    R[Range foundation: merged] --> A[Resize wrapper parity]
    F[Native endian-half identity finish] --> I
    A --> I[Exports, aliases, registry and inventory]
    H[Fixed benchmark harness] --> B[Isolated one-worker measurements]
    I --> B
    B --> T[Profile and improve retained slow cases]
    M[Independent error metrics contract and kernels] --> MI[Metrics integration]
    MC[Stateful FP stages and combined wheel parity] --> MI
    L[Label contracts] --> LK[Label kernels] --> LI[Label integration]
    LC[Independent component-consumer contracts] --> CC[Component consumers]
    LK -. Frozen label interface .-> CC
    P[Footprint contracts] --> PK[Dense footprint kernels]
    P --> D[Decomposition tables and formulas]
    PK -. Frozen atom interface .-> D
    P --> U[Mirror and pad utilities]
    PK --> PI[Combined footprint parity and integration]
    D --> PI
    U --> PI
    C[Completed transform class contracts] --> CK[Class implementation] --> CI[Class integration]
```

1. Workers freeze interfaces early: shape/dtype/ownership rules, worker policy,
   error/warning behavior and any runtime math providers. They report changes
   before another packet depends on them.
2. Workers submit a commit, advertised contract, exact commands/results, source
   and artifact hashes, unresolved cases and required shared wiring. Commit
   validated increments promptly; do not accumulate completed work in worktrees.
3. The integrator reviews and applies shared wiring. For resize/rescale, exercise
   the complete existing contract against the candidate without weakening the
   pending/native guards. Temporary in-memory candidate adapters are test tools,
   not public registration or coverage evidence.
4. Merge validated private kernels and benchmark tooling independently. Register
   public functions only with their tested wrapper, aliases, default API cases
   and regenerated inventory. Record `parity_passed / performance_pending` until
   exact-source measurements exist. A green path-existence test is insufficient.
5. Freeze the candidate wheel, installed SDK, source fingerprint, dependency
   versions and corpus hash. Benchmark the original and native public calls in
   an isolated Actions job, including allocations and normalization. Require one
   worker; additional worker counts are supplementary results.
6. Return measured bottlenecks to their owning worker. Keep every workload, raw
   sample and rejected trial. Merge validated improvements and remeasure the same
   corpus against the exact baseline; do not remove adverse cases.

Registration/build edits pass through one integrator. That short integration
step is the serialization point; independent implementation and contract work
continues while another packet builds, runs CI or waits for measurements.

## Next independent packets

| Priority | Packet and scope | Shared dependency / handoff |
|---|---|---|
| Active | `mean_squared_error`, `normalized_root_mse`, `peak_signal_noise_ratio` | Existing dtype/worker infrastructure; freeze promotions, ranges, shape checks, NaNs, infinities and warnings first |
| Next free worker | Connected-component `label` and its inventoried aliases | Freeze canonical numbering and connectivity before parallel block stitching; SIMD comparisons/run discovery, efficient scalar union/find where appropriate |
| Independent contracts | Small-object/hole removal, `clear_border`, `relabel_sequential` | Original contracts start now; kernels consume the frozen label interface, with joint parity required later |
| Independent | Footprint rectangle, diamond, disk, ball, mirror/pad and decompositions | Preserve tuple/decomposition forms, output dtype/shape and errors; raster generation is an image operation, not metadata |
| Independent contract first | Rank minimum/maximum/mean/sum/pop, then histogram-based rank APIs | Integrator freezes nested-namespace exposure; worker owns a reusable rolling-histogram engine |
| Independent class contracts | Projective, affine, Euclidean and similarity transforms | Inventory constructors, inherited methods, fields, mutation, composition and inversion; implement parent models before subclasses |

Keep all other inventoried names explicitly unassigned until a bounded packet
claims them. Repeat the process for remaining transforms/pyramids, measurements,
features, segmentation, restoration, registration, graph, draw, data, I/O and
utilities. Review additional namespaces and runtime class attributes rather than
treating the current static inventory as exhaustive.

## Execute the queue

The integrator uses the ledger checker before assigning work and after every
handoff. These commands only inspect the plan; they do not launch agents, mutate
source, compile, time code or cancel jobs.

```powershell
.venv/Scripts/python.exe tools/work_queue.py --check
.venv/Scripts/python.exe tools/work_queue.py --check --verify-local-proofs
.venv/Scripts/python.exe tools/work_queue.py --agents
.venv/Scripts/python.exe tools/work_queue.py --ready
.venv/Scripts/python.exe tools/work_queue.py --packet footprints-decomposition
```

Each scoped packet records exact owned paths, a base revision, read/link
dependencies, its build policy, a concrete `collaboration.followup_task` handoff
and acceptance gates. `--ready` lists scoped queued packets whose required proof
scope or frozen interface is available, excluding occupied writable files.
It does not resume the paused goal or assert spare
capacity: the integrator keeps at most three worker agents active, reuses a
completed agent with `followup_task`, and supplies the printed handoff with an
explicit isolated worktree and branch. Unscoped inventory items are not launched.
The checker rejects cycles, missing dependencies, excess worker owners, escaping
paths and concurrent ownership of the same file, including Windows case aliases.
It validates interface hashes and requires source/hash-bearing evidence manifests
with numeric native/pending counts. Registration explicitly requires private
native proofs; measurement requires actual public-native and harness proofs.
Reference contracts and expected failures cannot release either gate.
The optional local-proof check hashes retained manifest bytes in this workspace;
a fresh checkout needs those preserved artifacts before that check can pass.

The next handoffs are executable in this order:

1. Freeze and merge the preserved labeling contract increment after checking its
   three-runtime evidence. Keep native-pending cases explicit, then give that
   same worker the native packet. Footprint contracts are already merged.
   Component-consumer contracts can be assigned independently.
2. Keep the metrics agent on its private correctness/optimization follow-up.
   Freeze and merge that increment after all parity/sanitizer gates. Handle
   stateful NumPy callback policies in a separate stagewise packet; an explicit
   fallback guard is a documented gap, not completed native parity.
3. While those workers build or test, the integrator performs short shared
   wiring steps, checks actual combined wheels and merges validated increments.
   Never make independent contract/kernel work wait for an unrelated build.
   After a safe terminal handoff, immediately give a free worker a disjoint ready
   packet while integration or hosted measurements continue. Use the recorded
   worker queues; do not spawn a fourth worker alongside the integrator.
4. Register only the advertised, verified public contract, regenerate inventory,
   then freeze its package and dispatch its original-library measurement job.
   A private helper, original-only test, expected failure or development adapter
   cannot satisfy the public registration gate.
5. Return every measured regression to an owning optimization packet. Preserve
   the complete corpus, raw samples and failed trials, then remeasure retained
   improvements at one worker. Keep runtime-specific results separate.

All active packets use their own ignored cache. An agent's handoff includes its
live process handles; observation timeouts leave those jobs active. A finished
agent reports terminal handles, source SHA, exact artifacts and evidence paths
before the integrator schedules worktree cleanup.
Build environments must be retained: Python, NumPy and pybind11 CMake paths must
not point into an expired temporary build-isolation directory. Interface changes
create a new version/hash and an explicit dependent handoff; frozen interfaces are
not edited silently. Standalone adapters never create a second worker policy in
the installed main `_core`.

## Evidence and coordination rules

- `implementation` and packet state are separate. Suggested packet states are
  `queued`, `contracting`, `implementing`, `validating`, `parity_passed`,
  `performance_pending`, `measuring`, `measured`, and `accepted`.
- Per-API evidence must identify the tested contract, source SHA, wheel/binary
  hash, runtime profile, strict fallback-disabled result, scalar/SIMD result,
  default-one/global-worker behavior and original-library measurement. Class
  evidence also covers state and members. Null evidence means pending.
- Do not claim all-API completion from representative cases or native-partial
  registration. Every advertised contract and remaining inventory item must
  survive the completion audit. Measure non-image APIs according to their actual
  semantics; do not invent an image-kernel speedup for asset loading or metadata.
- Only the integrator changes this task ledger. Workers report evidence and
  request integration; they do not edit another packet's registry or inventory.
  Planning-ledger-only updates do not trigger the full build/test workflow;
  product, contract-test and integration changes still do.
- Never run local timing experiments. Untimed parity and `--check-only` are
  allowed. Never overlap measurements with builds/tests on their measurement
  host, or rebuild/delete a package a benchmark is using.
- Poll the same live process/Actions handle after observation timeouts. A timeout
  is not terminal and does not authorize cancellation or restart.
- After merging, remove only obsolete, clean worktrees whose commits are
  ancestors of `main`, after auditing retained files. Preserve active
  environments/caches, exact baselines/SDKs, running-job files, raw results,
  rejected trials and unfinished source. Never delete the whole `build/` tree.
  Previously rejected deletion paths remain protected.

The integrator fills a free worker slot from the ready queue, prioritizing work
that does not share writable files or require an unfinished interface. When a
measurement job is running remotely, workers can continue independent contracts
and kernels in their own worktrees. Reuse existing agents for follow-up packets.
