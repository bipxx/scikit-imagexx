# Phased fast run

This is the short execution route, using the existing tests and evidence. All earlier raw results, failures, products and scoped admissions remain unchanged.

Historical phase-1 Windows development run on the retained public203 wheel passed **1,286 checks in 4.68 seconds**: 477 surface checks, 206 function signatures, 602 supported-dtype numerical cases and one registry check. This is a bounded development result, not a new product qualification or performance benchmark. The selector's 17 regression tests also passed. Reports are retained locally in `build/feature-tests/`; that historical source had 266 upstream-only entries. The package no longer matches the current source registry.

## Phase 1: test each feature

Use `tools/run_feature_tests.py` to select existing tests and write one report. Before pytest starts, the runner checks the selected Python package against the expected native registry; a mismatch fails rather than testing an unintended install.

| Phase selector | Scope | Meaning |
|---|---|---|
| `surface` | All 477 inventoried public API keys | Check the public surface. Presence/signature or declared compatibility behavior does not establish native numerical implementation. |
| `smoke` (default) | Surface checks plus existing parity cases for the current source registry (215 native-partial image APIs and 3 metadata functions) | Compact worker-1 projection across supported `uint8`, `float32` and `float64` routes, using shape `3×19` or source-safe `7×19` where required. Unsupported dtype rejection tests remain in the full contracts; existing tolerances are unchanged. |
| `contract` | Existing inventoried parity targets for the selected implemented APIs | Run their existing full contract matrices. Do not replace them with the compact smoke projection or weaken their assertions. |

The current source has 253 upstream-only APIs pending native implementations: report them as numerical gaps, never numerical passes. An implemented API may also retain unsupported dtype, layout or parameter partitions; surface and smoke results do not close those gaps. Original-unsafe shapes remain blocked according to the existing tests; never use a generic small array to bypass source safety.

```powershell
python tools/run_feature_tests.py
python tools/run_feature_tests.py --phase surface
python tools/run_feature_tests.py --phase smoke --api 'morphology.*'
python tools/run_feature_tests.py --phase contract --api 'filters.sobel'
python tools/run_feature_tests.py --phase smoke --package-dir C:/path/to/unpacked-package-parent --report build/feature-tests/selected-report.json
```

`--api` is a repeatable fnmatch glob over public API keys. `--package-dir` optionally identifies the parent directory containing the explicit installed or unpacked package to test; omission uses the selected interpreter's installed package. `--report` defaults to `build/feature-tests/report.json`. The default phase is `smoke`. A selected upstream-only API runs its surface check and remains listed in `numerical_gaps`. Shared contract files may exercise additional APIs. The runner is a small selector/report wrapper around existing parity targets, not a new qualification framework.

The initial phase-1 command below is **historical only**, for its matching public203 source revision. It must not be used to imply that the retained203 package can test the current215-image registry; that mismatch correctly fails before pytest.

```powershell
# Historical matching-source phase-1 invocation, not a current-tree command:
.venv/Scripts/python.exe -I -X utf8 -B tools/run_feature_tests.py --package-dir build/feature-tests/public203-package
```

For current-tree tests, use an installed build of the current revision, or an explicit `--package-dir` whose package registry matches that exact source. The older `.venv` and retained203 baseline remain protected evidence, not substitutes for a current combined product.

## Phase 2: implement small groups

The initial two source batches added eight routes. Butterworth and the three LPI functions bring the current inventory to **215 native-partial image APIs + 3 metadata functions = 218 registered functions**, plus **5 partial classes**, with **253 upstream-only entries**. Accepted qualified binary coverage remains **203 image APIs**. Component tests do not qualify a combined wheel/SDK or establish performance.

**LPI source integration adds `filters.filter_forward`, `filter_inverse`, `wiener` and the partial `LPIFilter2D` class.** Modern and minimum profiles each pass 336 component checks in normal and scalar modes (161 LPI + 175 Butterworth), totaling **1,344 passes: 644 LPI + 700 Butterworth**. Six native modes pass **8,056,540 controls**. The corrected full-real FFT route resolves the 92 retained earlier failures without changing the frozen 80 LPI or 87 Butterworth specifications or numerical policy. NumPy arithmetic/array stages and Python callbacks remain providers; this is partial native FFT execution. The source-derived shared projection passes 54 analytic comparisons while all 645 previous API/dtype budgets remain unchanged. The actual source registry also passes 98 shared checks per runtime/dispatch mode (392 total): signatures, all three default dtypes and shapes, both worker settings, equivalent descriptors and dtype metadata. The exact smoke selector collects 1,334 nodes; collection does not establish their execution. Installed Release validation, supported-profile qualification and LPI performance are still pending; the workflow is wired for 1,135 contracts per mode, 1,334 derived smoke checks and 18 prospective one-thread cases. Accepted combined coverage remains 203 image APIs.

Component proof is preserved at `build/lpi-filter-private-v1/fft-full-real-proof-v1/manifest.json`, SHA-256 `05107d5e9ab0edd9b08c2c4963b47867692ab2de999be2008161f6c70b391e2a`. The registered-source proof `lpi-shared-source-proof-v1/manifest.json` under that directory has SHA-256 `36bfe6752d9f098a91b0f1a874f38714d7e9aa82556a6762d38d224f96bd057d`: 392 shared checks plus 994 inventory/class/runner metadata checks pass. The merged worktree is retired; its entire build/evidence directory is preserved with unchanged proof hashes. These are component results, not installed whole-product admission. Source is merged/pushed as `c895db2`; [installed Release and isolated one-thread measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40311937607) are running at that exact source.

**First source batch completed:** `morphology.ellipse`, `octagon`, `octahedron`, `star`, and `footprint_from_sequence` are integrated into the Python public source routes.

The public component contract passed **240 checks in both normal and scalar modes**: 226 frozen worker-1 cases, five registration checks, five optional worker-4 samples, two guards and two large ellipse comparisons. The large cases exceed the actual parallel-fill size threshold. Inventory and runner tests bring the normal scoped run to **735 passed**; direct fast smoke for the five additions passed **26 checks**. Kernel, binder and shell bytes remain the reviewed ad6 source; the retained Windows module supplies one actual worker owner. Main Python public routes were exercised through the development-only `build/phase2-footprints/component_bootstrap.py`, not a newly built combined wheel/SDK. Portable CMake configuration also verifies source inclusion, strict FP flags and the worker-refusal test definition. Existing tolerances, full contracts and all failed trials remain unchanged.

**Second source batch completed:** `measure.intersection_coeff`, `manders_coloc_coeff`, and `manders_overlap_coeff` now have public exports, core/CMake wiring, registry entries and shared smoke arguments. The public component contract passes **545 checks in both normal and scalar modes**, including 536 frozen cases, three registration checks, three optional-worker samples, one owner guard and two large-intersection cases above the parallel-count threshold. The normal run with inventory/runner checks passes **1,040 tests**; the three-feature fast smoke passes **16 checks** with zero tolerance. Actual AVX2 and portable CMake configurations verify conditional leaf inclusion, baseline ISA separation, strict FP and optional-worker test flags without a whole-product rebuild.

The checked-in compressed fixture carries all **1,608 semantic projections** from the existing 536 original cases on three profiles. The public test checks each row against the frozen matrix, so a fresh checkout needs no ignored reference journals. Provider-only outcomes retain zero native credit. No combined-product qualification or speed result follows from these component checks.

Choose small, independently mergeable API groups or concrete gaps in existing native APIs. Reuse source-faithful primitives, keep ownership explicit, and preserve fixed reference cases and unsupported/provider accounting. Validate the changed feature with its existing contract targets, obtain the appropriate independent review, and merge reviewable increments promptly. A green metadata or surface check is not a substitute for actual native execution.

## Phase 3: benchmark one thread

**Measured and kept:** both initial Linux profiles and successive colocalization
optimizations have completed isolated one-thread measurements. The
[minimum baseline](../benchmarks/phase3-minimum-initial.md) has 52/52 footprint
workloads faster than scikit-image; the
[modern baseline](../benchmarks/phase3-modern-initial.md) has 50/52, with two
large dense octahedron regressions. All 21 colocalization workloads are slower
than the original on both baselines. Each summary links the exact raw report;
every regression is retained.

[Affine v1](../benchmarks/phase3-affine-v1.md) and
[width v2](../benchmarks/phase3-width-v2.md) are merged. Width v2's measured
source `4a78e94` is integrated in main `78976b7` with unchanged product source.
On minimum Linux, the fixed 21-case geometric mean of original/current ratios
progresses **0.0342836736027× → 0.0574776398746× → 0.0919599844761×**.
Width v2 improves **1.599926× over v1**, or **2.682326× over baseline**. Every
case improves over v1, but **all 21 calls were slower than scikit-image in that v2 campaign**.
This compares normalized ratios from separate hosted runs, not direct paired
baseline/candidate latencies on one host. No modern/Windows optimization result
or combined-product qualification follows from it.

Earlier kept [float64 v3](../benchmarks/phase3-f64-same-host.md), source
`ac87e23`, showed +2.94% normalized and +2.03% direct gain on one EPYC 7763
ABBA job. Its earlier cross-CPU HOLD and one-ULP difference remain historical
evidence; no cumulative cross-CPU code-gain claim follows.

Earlier kept [byte masks v4](../benchmarks/phase3-byte-same-host.md), source
`0135305`, established five intersection wins over the original library on one
9V74 host. Its raw reports and all losses remain unchanged.

**Earlier kept colocalization: [float32 v5](../benchmarks/phase3-f32-same-host.md).** Main/checked
source `5b03eded` reuses the exact wheel compiled from `c53198a`, without a
rebuild. The controlled 9V45 ABBA job gives fixed21 scores
**0.2042894437937× baseline vs 0.2181707370842× candidate**, or
**1.067949× normalized / 1.059884× direct public-call gain**. All ten targeted
float32 cases improve **1.093572–1.247566×**; both candidate medians beat both
baseline medians in every target. All five normalized and seven direct losses
are published. The same five intersection cases beat the original; **16
colocalization regressions remain**. All 73 workloads and 4,088 samples are
retained in the exact compact ZIP. This is one minimum-Linux host, not a
universal-profile, cumulative cross-CPU or full-product claim.

The initial [preparation 40273141051](https://github.com/bipxx/scikit-imagexx/actions/runs/40273141051)
remains **741 passed / 44 failed**, with no timing from that failed attempt.
[Diagnostic 40275501905](https://github.com/bipxx/scikit-imagexx/actions/runs/40275501905)
and its [independent receipt](evidence/colocalization-live-reference-checkpoint.json)
show both old/new products matching fresh originals for all 536 cases in
normal/scalar modes. Fresh originals reproduce the same 44 MOC word-only
+1-ULP frozen differences, so v5 is unnecessary for that discrepancy on this
host. Its precise runtime cause remains unknown; the diagnostic itself has
`acceptance=false`.

The separately reviewed live-reference policy takes only a successful MOC
floating result word from the fresh original; result metadata and all other
frozen fields remain exact. Native observations must match the full fresh
observation. No tolerance, fixture, matrix or observer changes erase the old
failure. [Successful preparation 40277634220](https://github.com/bipxx/scikit-imagexx/actions/runs/40277634220)
passes 785 public checks per mode, 1,318 smoke checks, **21 metadata tests plus
30 subtests**, and 73 corpus checks per mode. Raw review confirms all 536 cases
plus three worker-4 samples per mode against their fresh originals. The
separate controlled measurement supplies the KEEP evidence.

**Rejected: [packed float32 v6](../benchmarks/phase3-f32-packed-same-host.md).**
Candidate `8641d0c` remains unmerged. Actual `movups`/`addps`, 2,787 native
controls per mode and [788 passing installed checks per mode](https://github.com/bipxx/scikit-imagexx/actions/runs/40285975359)
establish correctness, not a speedup. [Controlled ABBA 40286133008](https://github.com/bipxx/scikit-imagexx/actions/runs/40286133008)
gives **0.986506× normalized / 0.994889× direct gain**, with 13 normalized and
10 direct fixed21 losses. The exact raw ZIP, all ten float32 targets and all
losses are published. The packed path remains excluded; no identical rerun is planned.

**Kept and merged: [integer octahedron](../benchmarks/phase3-octahedron-integer-same-host.md).**
Main `f079c6d` preserves the exact measured product source `9da7ccf`.
The independent review retains 254 public component checks and 2,997 native
controls per mode; [preparation 40286601199](https://github.com/bipxx/scikit-imagexx/actions/runs/40286601199)
passes 799 installed checks per mode. [Controlled ABBA 40286815914](https://github.com/bipxx/scikit-imagexx/actions/runs/40286815914)
gives **12.138864× normalized / 12.070450× direct gain** over all 12 octahedron
cases on one 9V45 host. Both candidate medians beat both baselines in every
target. All 35 direct and 28 normalized losses outside the target are
published; the same 16 colocalization regressions remain.

**Held: [byte-affine candidate](../benchmarks/phase3-byte-affine-same-host.md).**
Source `887aff6` passes its 785-check installed gates per mode, but controlled
run 40288858109 gives **0.999459× direct / 0.999468× normalized** over fixed21.
Stepped intersection improves 2.313035×; all 14 Manders cases lose directly.
The target gain does not establish an aggregate improvement. Source remains
unmerged. Completed local and hosted-object comparisons found unchanged Manders
hot-loop instructions; no cause is inferred from placement alone, and no
identical rerun is planned.

**Current kept colocalization: [finite float32 scan](../benchmarks/phase3-finite-scan-same-host.md).**
Exact measured source `552d445` is merged in main and already includes the
kept octahedron change. On one Xeon 6973P-C host, controlled ABBA run
[40295836642](https://github.com/bipxx/scikit-imagexx/actions/runs/40295836642)
gives fixed21 **1.253970× direct / 1.184646× normalized gain**; scores move
**0.3476071029603× → 0.4117914024166×**. All ten float32 Manders targets
improve, **1.513748× direct / 1.426518× normalized** in aggregate, with both
candidate medians below both baselines. All 14 Manders cases give 1.349543×
direct gain. All 73 workloads, 4,088 samples, **35 direct / 39 normalized
losses** and absolute penalties are published. Five intersection cases beat
the original; **16 colocalization cases remain slower**.

Both product preparations passed **799 public checks per normal/scalar mode**,
1,318 smoke checks, 21 metadata tests plus 30 subtests and 73 corpus checks per
mode. Baseline and candidate each passed **20,975 native controls per mode**.
The new integer scan emits baseline SSE2, not a newly added AVX2 leaf; floating
reductions and provider boundaries stay unchanged. This is one minimum-Linux
host, not a cumulative cross-CPU gain or full supported-profile acceptance.

**Butterworth is kept in main `936481f`.** Its [first minimum-Linux Release result](../benchmarks/butterworth-minimum-initial.md) has **5/6 wins against the original**, geometric mean **1.112821796×**. The retained uint8 256² loss is **+0.2204691 ms**. Genuine installed gates pass **175 normal + 175 scalar tests and 1,322 smoke checks**; prior components pass 700 and shared defaults 72, with 8,046,020 native controls across six modes. This is owned C++ FFT with NumPy mask/padding/product/crop providers, not a prior-C++ gain or full API/profile claim. At that Butterworth checkpoint, source coverage was **212 partial image APIs + 3 metadata functions = 215**, with 257 upstream-only entries. The later LPI component integration is described above; the separate qualified combined baseline remains public203.

The [legacy regression comparison after Butterworth integration](../benchmarks/phase3-butterworth-legacy-same-host.md) is flat across the unchanged 73 cases: **1.000748117× direct / 0.999452166× normalized**, with **44 direct and 46 normalized losses** retained. Both candidate passes are slower than both baseline passes for intersection stepped, masked and cache-F cases (direct gains **0.918100× / 0.894351× / 0.877160×**). This one-host result establishes neither a cause nor a waiver. Preparation passes **974 public checks in each normal/scalar mode, 1,322 smoke checks, 21 metadata tests plus 30 subtests, and all 73 corpus checks per mode**. The corpus contains no Butterworth cases; its six-case first result and pending LPI work remain separate.

The executable campaign is `.github/workflows/phase3-public-benchmark.yml` with
`benchmarks/phase3_public.py`: 73 fixed workloads, all eight APIs, one worker,
seven paired rounds, exact untimed parity before any timing, and every slower
case retained with its absolute latency penalty. It reuses an authenticated
Release wheel selected by `producer_run` and `producer_revision`, requiring
the measurement checkout's product source to match. The initial profiles use
producer `1d37745`; affine v1 uses `5bc2109`, width v2 uses `4a78e94`, and
f64 v3 uses `ac87e23`; byte v4 uses `0135305`; float32 v5 reuses compiled
`c53198a` under checked source `5b03eded`; the current finite scan uses exact
`552d445` against the kept-octahedron baseline `9da7ccf`.
Earlier colocalization campaigns passed 785 installed public checks per mode;
octahedron/finite-scan preparations passed 799 per mode and 1,318 smoke checks.
Recorded Butterworth integration preparation passes 974 per mode and 1,322
smoke checks. New LPI preparation is wired for 1,135 per mode and 1,334 smoke
checks; its execution is pending. All retain 73 corpus checks per mode, with 21 metadata tests
plus 30 subtests (earlier campaigns had 13 metadata tests). A
separate host installs hash-locked wheels offline and performs measurement.
There are no builds or pytest runs on that measurement host. The workflow also
requires `source_revision` to match the selected workflow commit exactly.

For new controlled candidates, `prepare_only=true` retains all preparation
gates and artifacts while skipping redundant standalone timing. Then reuse the
frozen baseline/candidate artifacts in `.github/workflows/phase3-same-host.yml`
for ABBA measurement. The default `false` behavior remains available.

The benchmark/control-flow and existing artifact-isolation tests pass 31 checks;
both workflow files pass actionlint. Separately, five frozen original source
files and their license now supply the reconstruction tests' missing CI input:
all 80 affected original-reference tests pass without assertion changes. Exact
historical Git objects are fetched at depth one. Remaining H trace/provenance and
publication/import-isolation failures remain separate phase-4 work; scoped results
do not turn an unrelated failed broad suite into accepted product coverage.

## Phase 4: accept supported profiles

Run the full required supported-profile qualification before acceptance, including actual installed Python/C++ SDK interfaces, scalar/SIMD behavior, worker policy and source/runtime/product identity. Compact phase-1 checks accelerate diagnosis; they do not replace full contracts, profile gates or performance evidence. Report private parity, public-product acceptance and measured performance as separate states.

The [execution plan](EXECUTION_PLAN.md) and machine queue retain detailed ownership and acceptance boundaries. This guide supplies the short command sequence; historical evidence stays authoritative only for its exact source and scope.
