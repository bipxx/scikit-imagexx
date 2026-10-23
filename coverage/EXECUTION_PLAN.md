# Parallel implementation and validation plan

Implement the complete scikit-image API in optimized C++ with interchangeable
Python calls, per-function parity, optional process-global workers (default one),
and isolated one-worker comparisons against scikit-image 0.26.0. This includes
classes, nested namespaces, data and I/O. Private helpers and compatibility
forwarding do not count as completed public implementations.

The current inventory contains 474 public paths: 133 native-partial image entries,
three metadata entries, and 338 upstream-only entries. The expansion adds 19
source-backed public/legacy rank.generic aliases of existing algorithms, not
19 additional algorithms. Imported dependencies are excluded from that count. It includes 29 classes with
284 statically discovered member paths; constructors, state, runtime fields and
additional namespaces still need assessment. These counts describe implementation
classification, not verified parity or performance.

The user has resumed implementation. Three worker agents are executing the
rank, component-consumer and public transform-class lanes in parallel, with the integrator handling
shared wiring, measured bottlenecks and prompt merges. The earlier three-agent
review and preserved checkpoints remain recorded in the ledger. Existing hosted
measurement jobs retain their authoritative run.

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

| Lane | Agent | Independent responsibility | Current checkpoint |
|---|---|---|---|
| Integrator | `root` | Public exports, registry/inventory, measured resize bottlenecks, prompt merges and cleanup | Public product `05c3506` validated across six profiles; publishing proof and launching isolated measurements; resize copy optimization under validation |
| Worker 1 | `metrics_reductions` | Reusable rank histogram engine and five canonical/generic aliases | Rank contracts `eabc946` merged; private SIMD/serial-alias implementation running |
| Worker 2 | `resize_api` | Component counts/filtering, clear_border, relabeling and returned ArrayMap | Contracts `a3714af` merged; own native consumers/ArrayMap implementation running |
| Worker 3 | `resize_bench` | Public transform-class exports and separate class coverage/evidence | Whole-package wave validation complete; integrating four privately implemented classes without inflating image-function counts |

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
uses source `66e5f78`; all six preparation/measurement jobs completed successfully. Observe this same run;
do not restart it because of a queue or timeout. The [baseline report](../benchmarks/resize-public-initial.md) retains every sample
and 49/52/61 slower cases across Windows/modern/minimum. Row-span normalization
copies are the first measured-bottleneck trial; parity checks are running before
remeasurement of the unchanged corpus.

The private metrics baseline (`03581ae`, integrated as `99abcfc`) passed 4,082
tests in each of six profiles through the actual shared `_core`. It is merged as
`147bd8b`; original worker ancestry is retained by `1945c2e`. Follow-up `070fc64`
fixes buffered reductions and default-policy details, passing six 4,308-case
profiles plus native sanitizers. Stateful callback policies and exact combined
optimized wheels remain registration gates. Private parity is scoped evidence.
Metrics benchmark tooling (`6c74563`, merged as `d94ef6c`) has a fixed 1,231-case
corpus, including 19 expected errors; its one-worker timing workflow is now queued.
The combined public metrics, footprints and label package at `05c3506` now passes
14,963 public-suite checks and 6,506 metadata/shared API checks in each of six
runtime/SIMD profiles. All 18 dedicated untimed corpus runs pass at workers one
and four, plus native tests, sanitizers and external SDK consumers. Counts include
reference and metadata assertions; they are not distinct algorithm counts.
The [acceptance record](evidence/public-wave1.json) seals 702 retained artifacts.
These public integrations are parity-passed with performance pending. The one-worker
[metrics](https://github.com/bipxx/scikit-imagexx/actions/runs/39983272390),
[footprint](https://github.com/bipxx/scikit-imagexx/actions/runs/39983468427) and
[label](https://github.com/bipxx/scikit-imagexx/actions/runs/39983471905)
measurement workflows are dispatched at recorded exact revisions. Poll these
runs unchanged, including while queued.

Transform class contracts (`6fa7f0a`, merged as `ad66094`) passed 228 reference and
strict-rejection checks per runtime, with 224 public-native-pending cases. Four
private class adapters are implemented; public exports and class-specific
coverage are now a separate worker packet. Completed worktrees are removed only after
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
| Measurement ready | Three error metrics, public label aliases and six footprint functions | Exact public parity accepted; isolated one-worker workflows are the next gate |
| Active native implementation | Small-object/hole removal, `clear_border`, `relabel_sequential` and returned ArrayMap | Frozen contracts merged; native kernels and adapter parity proceeding independently |
| Active native implementation | Rank minimum/maximum/mean/sum/pop and matching generic aliases | Frozen nested namespace and histogram contracts merged; reusable rolling engine under validation |
| Active public integration | Projective, affine, Euclidean and similarity transforms | Private adapters validated; separate class registry, public member/state/ownership checks and explicit math-provider scope |
| Active optimization | Resize/rescale normalization copies | Retain all baseline slow cases; exact byte parity before isolated same-corpus remeasurement |

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
It does not change goal status or assert spare
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
