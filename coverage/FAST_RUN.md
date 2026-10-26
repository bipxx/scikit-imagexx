# Phased fast run

This is the short execution route, using the existing tests and evidence. All earlier raw results, failures, products and scoped admissions remain unchanged.

Initial Windows development run on the retained public203 wheel passed **1,286 checks in 4.68 seconds**: 477 surface checks, 206 function signatures, 602 supported-dtype numerical cases and one registry check. This is a bounded development result, not a new product qualification or performance benchmark. The selector's 17 regression tests also passed. Reports are retained locally in `build/feature-tests/`; the 266 upstream-only entries remain gaps.

## Phase 1: test each feature

Use `tools/run_feature_tests.py` to select existing tests and write one report. Before pytest starts, the runner checks the selected Python package against the expected native registry; a mismatch fails rather than testing an unintended install.

| Phase selector | Scope | Meaning |
|---|---|---|
| `surface` | All 477 inventoried public API keys | Check the public surface. Presence/signature or declared compatibility behavior does not establish native numerical implementation. |
| `smoke` (default) | Surface checks plus existing parity cases for 203 native image APIs and 3 metadata functions | Compact worker-1 projection across supported `uint8`, `float32` and `float64` routes, using shape `3×19` or source-safe `7×19` where required. Unsupported dtype rejection tests remain in the full contracts; existing tolerances are unchanged. |
| `contract` | Existing inventoried parity targets for the selected implemented APIs | Run their existing full contract matrices. Do not replace them with the compact smoke projection or weaken their assertions. |

The remaining 266 APIs are pending native implementations: report them as numerical gaps, never numerical passes. An implemented API may also retain unsupported dtype, layout or parameter partitions; surface and smoke results do not close those gaps. Original-unsafe shapes remain blocked according to the existing tests; never use a generic small array to bypass source safety.

```powershell
python tools/run_feature_tests.py
python tools/run_feature_tests.py --phase surface
python tools/run_feature_tests.py --phase smoke --api 'morphology.*'
python tools/run_feature_tests.py --phase contract --api 'filters.sobel'
python tools/run_feature_tests.py --phase smoke --package-dir C:/path/to/unpacked-package-parent --report build/feature-tests/selected-report.json
```

`--api` is a repeatable fnmatch glob over public API keys. `--package-dir` optionally identifies the parent directory containing the explicit installed or unpacked package to test; omission uses the selected interpreter's installed package. `--report` defaults to `build/feature-tests/report.json`. The default phase is `smoke`. A selected upstream-only API runs its surface check and remains listed in `numerical_gaps`. Shared contract files may exercise additional APIs. The runner is a small selector/report wrapper around existing parity targets, not a new qualification framework.

This workspace's older `.venv` install does not match the registry. To reuse the retained qualified package without rebuilding:

```powershell
.venv/Scripts/python.exe -I -X utf8 -B tools/run_feature_tests.py --package-dir build/feature-tests/public203-package
```

## Phase 2: implement small groups

Choose small, independently mergeable API groups or concrete gaps in existing native APIs. Reuse source-faithful primitives, keep ownership explicit, and preserve fixed reference cases and unsupported/provider accounting. Validate the changed feature with its existing contract targets, obtain the appropriate independent review, and merge reviewable increments promptly. A green metadata or surface check is not a substitute for actual native execution.

## Phase 3: benchmark one thread

After feature correctness, compare complete public calls against the original library on isolated single-thread measurement hosts. Record the actual tested product and reuse per-function cases and existing measurement tools; retain raw samples, every regression, absolute runtime penalties and the next bottleneck owner. Keep builds and tests off measurement hosts. Development timings do not establish full product acceptance.

## Phase 4: accept supported profiles

Run the full required supported-profile qualification before acceptance, including actual installed Python/C++ SDK interfaces, scalar/SIMD behavior, worker policy and source/runtime/product identity. Compact phase-1 checks accelerate diagnosis; they do not replace full contracts, profile gates or performance evidence. Report private parity, public-product acceptance and measured performance as separate states.

The [execution plan](EXECUTION_PLAN.md) and machine queue retain detailed ownership and acceptance boundaries. This guide supplies the short command sequence; historical evidence stays authoritative only for its exact source and scope.
