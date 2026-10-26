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

**Next:** prepare actual Release products for the eight additions and run complete-public-call, one-thread comparisons using the existing measurement tools. Record the actual tested product; retain raw samples, every regression, absolute runtime penalties and the next bottleneck owner. Keep builds and tests off measurement hosts. Development timings do not establish full product acceptance or a speedup.

The executable campaign is `.github/workflows/phase3-public-benchmark.yml` with
`benchmarks/phase3_public.py`: 73 fixed workloads, all eight APIs, one worker,
seven paired rounds, exact untimed parity before any timing, and every slower
case retained with its absolute latency penalty. It reuses a completed Release
wheel job from run `40256588339`, source `1d377456856685a670333f29b364bd73c2cd9cf0`,
and verifies that the current compiled/product source is unchanged. Select
`runtime=minimum` first because its artifact is available; `modern` uses the same
workload set when that producer artifact is published. Preparation checks the
installed public contracts, scalar dispatch and complete registry smoke; a
separate host installs hash-locked wheels offline and performs measurement.
There are no builds or pytest runs on that measurement host. The workflow also
requires `source_revision` to match the selected workflow commit exactly.

The benchmark/control-flow and existing artifact-isolation tests pass 31 checks;
both workflow files pass actionlint. Separately, five frozen original source
files and their license now supply the reconstruction tests' missing CI input:
all 80 affected original-reference tests pass without assertion changes. Exact
historical Git objects are fetched at depth one. Remaining H trace/provenance and
publication/import-isolation failures remain separate phase-4 work; this setup
does not turn an unrelated failed broad suite into accepted product coverage.

## Phase 4: accept supported profiles

Run the full required supported-profile qualification before acceptance, including actual installed Python/C++ SDK interfaces, scalar/SIMD behavior, worker policy and source/runtime/product identity. Compact phase-1 checks accelerate diagnosis; they do not replace full contracts, profile gates or performance evidence. Report private parity, public-product acceptance and measured performance as separate states.

The [execution plan](EXECUTION_PLAN.md) and machine queue retain detailed ownership and acceptance boundaries. This guide supplies the short command sequence; historical evidence stays authoritative only for its exact source and scope.
