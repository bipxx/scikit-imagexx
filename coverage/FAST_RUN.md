# Phased fast run

This is the short execution route, using the existing tests and evidence. All earlier raw results, failures, products and scoped admissions remain unchanged.

Historical phase-1 Windows development run on the retained public203 wheel passed **1,286 checks in 4.68 seconds**: 477 surface checks, 206 function signatures, 602 supported-dtype numerical cases and one registry check. This is a bounded development result, not a new product qualification or performance benchmark. The selector's 17 regression tests also passed. Reports are retained locally in `build/feature-tests/`; that historical source had 266 upstream-only entries. The package no longer matches the current source registry.

## Phase 1: test each feature

Use `tools/run_feature_tests.py` to select existing tests and write one report. Before pytest starts, the runner checks the selected Python package against the expected native registry; a mismatch fails rather than testing an unintended install.

| Phase selector | Scope | Meaning |
|---|---|---|
| `surface` | All 477 inventoried public API keys | Check the public surface. Presence/signature or declared compatibility behavior does not establish native numerical implementation. |
| `smoke` (default) | Surface checks plus existing parity cases for the current source registry (211 native-partial image APIs and 3 metadata functions) | Compact worker-1 projection across supported `uint8`, `float32` and `float64` routes, using shape `3×19` or source-safe `7×19` where required. Unsupported dtype rejection tests remain in the full contracts; existing tolerances are unchanged. |
| `contract` | Existing inventoried parity targets for the selected implemented APIs | Run their existing full contract matrices. Do not replace them with the compact smoke projection or weaken their assertions. |

The current source has 258 upstream-only APIs pending native implementations: report them as numerical gaps, never numerical passes. An implemented API may also retain unsupported dtype, layout or parameter partitions; surface and smoke results do not close those gaps. Original-unsafe shapes remain blocked according to the existing tests; never use a generic small array to bypass source safety.

```powershell
python tools/run_feature_tests.py
python tools/run_feature_tests.py --phase surface
python tools/run_feature_tests.py --phase smoke --api 'morphology.*'
python tools/run_feature_tests.py --phase contract --api 'filters.sobel'
python tools/run_feature_tests.py --phase smoke --package-dir C:/path/to/unpacked-package-parent --report build/feature-tests/selected-report.json
```

`--api` is a repeatable fnmatch glob over public API keys. `--package-dir` optionally identifies the parent directory containing the explicit installed or unpacked package to test; omission uses the selected interpreter's installed package. `--report` defaults to `build/feature-tests/report.json`. The default phase is `smoke`. A selected upstream-only API runs its surface check and remains listed in `numerical_gaps`. Shared contract files may exercise additional APIs. The runner is a small selector/report wrapper around existing parity targets, not a new qualification framework.

The initial phase-1 command below is **historical only**, for its matching public203 source revision. It must not be used to imply that the retained203 package can test the current211-image registry; that mismatch correctly fails before pytest.

```powershell
# Historical matching-source phase-1 invocation, not a current-tree command:
.venv/Scripts/python.exe -I -X utf8 -B tools/run_feature_tests.py --package-dir build/feature-tests/public203-package
```

For current-tree tests, use an installed build of the current revision, or an explicit `--package-dir` whose package registry matches that exact source. The older `.venv` and retained203 baseline remain protected evidence, not substitutes for a current combined product.

## Phase 2: implement small groups

The two source batches bring the inventory to **211 native-partial image APIs + 3 metadata functions = 214 registered functions**, with **258 upstream-only entries**. Accepted qualified binary coverage remains **203 image APIs**. Component tests do not qualify a combined wheel/SDK or establish performance.

**First source batch completed:** `morphology.ellipse`, `octagon`, `octahedron`, `star`, and `footprint_from_sequence` are integrated into the Python public source routes.

The public component contract passed **240 checks in both normal and scalar modes**: 226 frozen worker-1 cases, five registration checks, five optional worker-4 samples, two guards and two large ellipse comparisons. The large cases exceed the actual parallel-fill size threshold. Inventory and runner tests bring the normal scoped run to **735 passed**; direct fast smoke for the five additions passed **26 checks**. Kernel, binder and shell bytes remain the reviewed ad6 source; the retained Windows module supplies one actual worker owner. Main Python public routes were exercised through the development-only `build/phase2-footprints/component_bootstrap.py`, not a newly built combined wheel/SDK. Portable CMake configuration also verifies source inclusion, strict FP flags and the worker-refusal test definition. Existing tolerances, full contracts and all failed trials remain unchanged.

**Second source batch completed:** `measure.intersection_coeff`, `manders_coloc_coeff`, and `manders_overlap_coeff` now have public exports, core/CMake wiring, registry entries and shared smoke arguments. The public component contract passes **545 checks in both normal and scalar modes**, including 536 frozen cases, three registration checks, three optional-worker samples, one owner guard and two large-intersection cases above the parallel-count threshold. The normal run with inventory/runner checks passes **1,040 tests**; the three-feature fast smoke passes **16 checks** with zero tolerance. Actual AVX2 and portable CMake configurations verify conditional leaf inclusion, baseline ISA separation, strict FP and optional-worker test flags without a whole-product rebuild.

The checked-in compressed fixture carries all **1,608 semantic projections** from the existing 536 original cases on three profiles. The public test checks each row against the frozen matrix, so a fresh checkout needs no ignored reference journals. Provider-only outcomes retain zero native credit. No combined-product qualification or speed result follows from these component checks.

Choose small, independently mergeable API groups or concrete gaps in existing native APIs. Reuse source-faithful primitives, keep ownership explicit, and preserve fixed reference cases and unsupported/provider accounting. Validate the changed feature with its existing contract targets, obtain the appropriate independent review, and merge reviewable increments promptly. A green metadata or surface check is not a substitute for actual native execution.

## Phase 3: benchmark one thread

**Measured and kept:** both initial Linux profiles and four colocalization
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

**Current kept increment: [dense byte masks v4](../benchmarks/phase3-byte-same-host.md).**
Source/main `0135305` is the exact measured product. One EPYC 9V74 ABBA job
compares fixed21 aggregates **0.0918373561074× baseline vs 0.2091629408619×
candidate**, giving **2.277537× normalized / 2.287450× direct**. All seven
intersection workloads improve; tiny, cache, memory, volume and cache-F beat
scikit-image in both candidate passes. Stepped and masked remain slower, as do
all 14 MCC/MOC workloads: **16 colocalization regressions remain**. All nine
normalized and nine direct losses are published, with every original penalty.
The compact ZIP retains all four 73-case reports and 4,088 samples; their exact
correctness observations and before/after product identities pass.

This is one minimum-Linux host, not a universal-profile or cumulative
cross-CPU speed claim. **Next:** guarded float32 loads for ten workloads are in
source/control work, with no performance result. Keep the fixed21 metric,
all 73 workloads and exact gates. Root owns launches/keep decisions; C++ and
focused-test ownership remain separate.

The executable campaign is `.github/workflows/phase3-public-benchmark.yml` with
`benchmarks/phase3_public.py`: 73 fixed workloads, all eight APIs, one worker,
seven paired rounds, exact untimed parity before any timing, and every slower
case retained with its absolute latency penalty. It reuses an authenticated
Release wheel selected by `producer_run` and `producer_revision`, requiring
the measurement checkout's product source to match. The initial profiles use
producer `1d37745`; affine v1 uses `5bc2109`, width v2 uses `4a78e94`, and
f64 v3 uses `ac87e23`; byte v4 uses `0135305`.
Preparation passed 785 installed public checks in each normal/scalar mode, 1,318 registry smoke
checks, 13 benchmark metadata checks and 73 corpus checks in each mode. A
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
