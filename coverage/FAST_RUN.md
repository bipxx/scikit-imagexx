# Phased fast run

This is the short execution route, using the existing tests and evidence. All earlier raw results, failures, products and scoped admissions remain unchanged.

Historical phase-1 Windows development run on the retained public203 wheel passed **1,286 checks in 4.68 seconds**: 477 surface checks, 206 function signatures, 602 supported-dtype numerical cases and one registry check. This is a bounded development result, not a new product qualification or performance benchmark. The selector's 17 regression tests also passed. Reports are retained locally in `build/feature-tests/`; that historical source had 266 upstream-only entries. The package no longer matches the current source registry.

## Phase 1: test each feature

Use `tools/run_feature_tests.py` to select existing tests and write one report. Before pytest starts, the runner checks the selected Python package against the expected native registry; a mismatch fails rather than testing an unintended install.

| Phase selector | Scope | Meaning |
|---|---|---|
| `surface` | All 477 inventoried public API keys | Check the public surface. Presence/signature or declared compatibility behavior does not establish native numerical implementation. |
| `smoke` (default) | Surface checks plus existing parity cases for the current source registry (226 native-partial image APIs and 3 metadata functions) | Compact worker-1 projection across supported `uint8`, `float32` and `float64` routes, using shape `3×19` or source-safe `7×19` where required. Unsupported dtype rejection tests remain in the full contracts; existing tolerances are unchanged. |
| `contract` | Existing inventoried parity targets for the selected implemented APIs | Run their existing full contract matrices. Do not replace them with the compact smoke projection or weaken their assertions. |

The current source has 242 upstream-only APIs pending native implementations: report them as numerical gaps, never numerical passes. An implemented API may also retain unsupported dtype, layout or parameter partitions; surface and smoke results do not close those gaps. Original-unsafe shapes remain blocked according to the existing tests; never use a generic small array to bypass source safety.

```powershell
python tools/run_feature_tests.py
python tools/run_feature_tests.py --phase surface
python tools/run_feature_tests.py --phase smoke --api 'morphology.*'
python tools/run_feature_tests.py --phase contract --api 'filters.sobel'
python tools/run_feature_tests.py --phase smoke --package-dir C:/path/to/unpacked-package-parent --report build/feature-tests/selected-report.json
```

`--api` is a repeatable fnmatch glob over public API keys. `--package-dir` optionally identifies the parent directory containing the explicit installed or unpacked package to test; omission uses the selected interpreter's installed package. `--report` defaults to `build/feature-tests/report.json`. The default phase is `smoke`. A selected upstream-only API runs its surface check and remains listed in `numerical_gaps`. Shared contract files may exercise additional APIs. The runner is a small selector/report wrapper around existing parity targets, not a new qualification framework.

The initial phase-1 command below is **historical only**, for its matching public203 source revision. It must not be used to imply that the retained203 package can test the current219-image registry; that mismatch correctly fails before pytest.

```powershell
# Historical matching-source phase-1 invocation, not a current-tree command:
.venv/Scripts/python.exe -I -X utf8 -B tools/run_feature_tests.py --package-dir build/feature-tests/public203-package
```

For current-tree tests, use an installed build of the current revision, or an explicit `--package-dir` whose package registry matches that exact source. The older `.venv` and retained203 baseline remain protected evidence, not substitutes for a current combined product.

## Phase 2: implement small groups

**Kept packed uint8 color conversion:** [the fixed12 same-host comparison](../benchmarks/rgb2yuv-u8-same-host.md) gives **1.267172460x direct / 1.267374129x normalized gain** over the previous C++ product. The two uint8 targets improve **3.532116x at 256-square / 4.045675x at 1024-square**, both repeating against both baselines under both metrics. All twelve cases beat scikit-image in both candidate passes. All **672 samples**, **five direct/five normalized control losses**, and three direct/four normalized repeat losses are retained; the largest control penalty is +96.537 microseconds for 1024 swapped float32. Same-host original/current scores move 4.351957200x to 5.515557963x; the earlier separate-host 3.886 result is not multiplied into this gain. Source `12d5e89` is merged as `277b766`. Fifteen additions and two removals replace scalar uint8 conversion/stack staging with bounded packed conversion, preserving multiplication/addition order, offsets, tails and workers. Kept/candidate native byte/FP streams match within each corresponding mode; 7,540 public component checks pass. Genuine preparation passes 2,818 focused checks, 3,696 color regressions and 1,377 smoke checks. This is one EPYC 9V74 host and the measured 229-function product; full supported-profile qualification stays 203 image APIs.

**Block reduction is in Release validation:** the 72 original-first specifications now pass **580 public and 88 shared component checks**, with both **1,381-node collection censuses** verified. All 43 frozen native routes, references and error bounds remain unchanged. Source fixes reproduce NumPy half stores at inner-segment boundaries and its zero/unit-axis-aware output allocation; the NumPy1 padding-metadata route uses the equivalent ndarray method. All failed versions remain retained. Ordinary typed default sums are owned; custom reducers/kwargs, exceptional sums and unassessed half layouts/buffering remain explicit primitive providers. Half admission currently requires original padded native/aligned C storage and block volume within the runtime buffer limit. [Release40417700622](https://github.com/bipxx/scikit-imagexx/actions/runs/40417700622) is validating combined source `21e0ba9`, including the kept color optimization. Its new six-case original-library benchmark is pending. The opt-in newest-batch timing mode retains full build/parity/identity gates while skipping eleven older standalone timing batches. This API is not yet merged or counted in main coverage.

**RGB2YUV storage is integrated:** float16 and swapped float16/float32/float64 preparation now reaches the existing owned matrix kernel, with per-worker FP capture/restoration and source warning/error dispatch. The 18 original-first specifications pass **148 storage checks plus 7,392 shared color checks = 7,540 component checks** across both runtimes and normal/scalar modes. Source `e9c7237` is merged as `ab7abc0`. [Genuine Release40408342309](https://github.com/bipxx/scikit-imagexx/actions/runs/40408342309) passes **2,818 focused public checks**, **3,696 full color regressions**, **1,377 smoke checks** and **1,612 matching regressions**. Its [fixed12 one-thread comparison](../benchmarks/rgb2yuv-storage-minimum-initial.md) has **12/12 original-library wins**, geometric mean **3.886110368x** overall, **3.731809864x** for six new storage targets and **4.046790791x** for six native controls. All **168 samples**, actual dtype/byte-order fields and zero measured output errors are retained. This closes partitions of an existing API: source coverage remains 226 native-partial image APIs plus three metadata functions, while full supported-profile qualification stays 203. Other transform half routes, swapped integers and unassessed protocols remain explicit gaps; this is not a prior-C++ performance gain.

**Shannon entropy is integrated:** `measure.shannon_entropy` owns canonical sorted symbol counts and binary64 probability/log/pairwise reduction, reusing the existing counting and global worker primitives. Wide integer identities, NaN/signed-zero grouping, base-validation order, warning/error policy and scalar dtype are covered by 30 original-first specifications. NumPy unique for unsupported symbol types and scalar base conversion remain explicit providers. Source `62d1270` is merged as `9f4fe3c`. Components pass **244 public checks and 88 shared checks** across both runtimes and normal/scalar dispatch; both profiles collect **1,377 smoke nodes**. [Genuine minimum Release40403748672](https://github.com/bipxx/scikit-imagexx/actions/runs/40403748672) has passed its installed producer, including both Shannon modes, smoke and the complete histogram-matching suite. Its [first six-case one-thread comparison](../benchmarks/shannon-entropy-minimum-initial.md) is verified: **6/6 wins**, geometric mean **22.457450019x**, with all **84 samples** and zero measured output errors retained. Eleven installed normal/scalar pairs pass **2,744 public checks**, plus **1,377 smoke checks** and **1,612 histogram-matching regressions**. This is the exact measured source, not a combined-current-main or prior-C++ gain. Current source coverage is **226 native-partial image APIs + 3 metadata functions = 229 registered functions**, with **242 upstream-only entries**; full supported-profile qualification stays **203 image APIs**.

**Kept float32 tiled streaming:** [the fixed12 one-thread comparison](../benchmarks/inertia-f32-streaming-same-host.md) gives **1.119084834x direct / 1.122603981x normalized gain** over the kept double-streaming product. The two 1024-square float32 targets improve **1.922135447x / 1.842475618x**, both repeating against both baselines under both metrics. All twelve cases beat scikit-image in both candidate passes. All **672 samples**, **four direct/two normalized losses** and ten controls (1.004335502x direct) are retained. Tensor256 float32 and eigenvalues1024 uint8 repeat direct losses; the largest penalty is +6.511 microseconds. Source `e360eee` is merged as `b29377e`; it enables the existing typed K=3 tile for float32 without changing formulas, reduction order, workers or numerical policy. Genuine preparation passes nine normal/scalar pairs and 1,365 smoke checks; 2,256 component checks preserve 2,224 numerical records and arrays exactly. The measured source predates coordinate/Shannon additions; full supported-profile qualification stays 203 image APIs.


Li callbacks need a larger, explicit follow-up: the existing iteration hook alone does not preserve warning-before-callback order, and falling back after a callback can repeat its side effects. All 51 frozen callback cases use a policy outside the current native solver admission. Reuse those references when the operation-level warning route is implemented; a nominal adapter with no assessed cases is not a closure. Keep this work open rather than recapturing easier references or weakening the policy.

**Coordinate moments are integrated:** `measure.moments_coords` and `moments_coords_central` add owned float32/float64 monomial products and ordered point reductions, with SIMD across independent outputs and the existing global worker budget. NumPy tuple/center/cast/power preparation and K1/complex/exceptional reductions remain explicit providers. The 48 original-first specifications pass **1,940 family and 176 shared checks** across both profiles and dispatch modes; old records and all 716 archives remain exact. Source `a1dafca` is merged as `d1b6198`. [Genuine minimum Release40394363351](https://github.com/bipxx/scikit-imagexx/actions/runs/40394363351) passes **2,622 public checks and 1,373 smoke checks**. Its [first one-thread comparison](../benchmarks/moments-coords-minimum-initial.md) has **12/12 wins**, geometric mean **2.273374651x** (raw 2.712598165x; central 1.905270147x), retaining all 168 samples. Inputs are actual 256/1024-by-2 point clouds; size means points, not square images. Only a provider NaN-sign cross-worker assertion changed after saved originals demonstrated that sign instability; finite values, signed zeros, nonfinite masks, errors and state remain exact, with raw words/failure preserved. Native normal/scalar/sanitizer and separate portable/no-OpenMP controls pass. At the coordinate checkpoint, source coverage was **225 native-partial image APIs + 3 metadata functions = 228 registered functions**, with **243 upstream-only entries**. Full supported-profile qualification remains **203 image APIs**. The later Shannon integration is described above.

**Kept large-double tiled streaming:** [the fixed12 controlled comparison](../benchmarks/inertia-streaming-same-host.md) gives **1.110031361x direct / 1.137751467x normalized gain** over the kept K=3 version. The four targeted 1024-square uint8/float64 cases improve **1.49886244x / 1.52056178x**, each repeating against both baselines under both metrics. Ten of twelve cases beat scikit-image in both candidate passes. All **672 samples**, **six direct/four normalized losses**, and the eight controls' 0.95526086x direct result remain published. Large float32 tensor/eigenvalue controls add 59.339/57.945 microseconds versus baseline and have mixed original-library outcomes; no cause is assigned to those regressions. Exact measured source `64b7fca` is merged as `cbf3050`, with 29 lines preserving reduction order, bounds and worker rules. Native output/FP streams match in eight modes, and 2,256 public checks retain all 2,224 old numerical rows exactly. Genuine preparation passes nine contract pairs and 1,365 smoke checks. This measured wheel predates the independent coordinate API integration; full supported-profile qualification stays 203 image APIs.

**Combined tensor/eigenvalue measurement:** [the 12-case controlled campaign](../benchmarks/inertia-pipeline-k3-same-host.md) verifies the actual combined `070acc7` product against pre-K3 `812a870`: **3.150593082x direct / 3.137168950x normalized gain**, with every case beating both baseline passes under both metrics. Tensor and eigenvalue groups improve **3.299412022x / 3.008486573x directly**. All **672 samples** and all rows are published. Eight of twelve cases beat scikit-image; four 1024-square uint8/float64 cases remain slower, including their absolute penalties in the report. Balanced original/current scores move 0.366327875x to 1.149232434x on one EPYC 9V74 host. Combined validation passes 1,548 component and 176 shared checks with unchanged saved arrays; genuine preparation passes nine normal/scalar contract pairs and 1,365 smoke checks. Those four residual losses motivated the separately measured and kept tiled-streaming experiment above; this first combined report remains unchanged. Full supported-profile qualification remains 203 image APIs.

**Kept K=3 projection optimization:** [the fixed six-case controlled comparison](../benchmarks/inertia-k3-same-host.md) gives **2.415797494x direct / 2.387212955x normalized gain** over the previous C++ inertia path. The three 1024-square targets improve **4.442269025x / 4.379652768x**, each repeating against both baseline passes under both metrics. All six cases beat scikit-image in both candidate passes; there are no direct or normalized regressions in this cohort. All **336 raw samples** are published. Source `a60df8e` is merged as `046f406`; it adds K=3 to the existing fused SIMD template with no formula, reduction-order, worker or tolerance changes. Eight focused kept/candidate native runs agree byte-for-byte on outputs/FP flags; 1,980 public checks and all 1,952 retained numerical records agree with baseline. Genuine preparation passes eight normal/scalar pairs and 1,361 smoke nodes without repeating standalone timings. The [first mixed benchmark](../benchmarks/inertia-tensor-minimum-initial.md) remains intact. This one-host measured wheel predates the independent eigenvalue integration; it is not a combined-main or full supported-profile qualification.

**Inertia eigenvalues are integrated:** `measure.inertia_tensor_eigvals` reuses the existing C++ symmetric solver through a 31-line binding and a 17-line frontend. Supplied T takes precedence; lower-triangle interpretation, clipping, descending scalar-list output and dtype/error behavior are covered by 34 original-first cases on both runtimes. Component validation passes **1,548 checks plus 88 shared checks**, with all prior tensor/moment observations and arrays unchanged; both profiles collect **1,365 smoke nodes**. The generic metadata caller validates this new API as a whole list before comparing item metadata, retaining the full context required by the frozen solver-plus-Weyl bound. Source `812a870` is merged as `d54f664`; [Genuine Release40383375851](https://github.com/bipxx/scikit-imagexx/actions/runs/40383375851) passes all nine installed contract pairs (2,426 checks) and 1,365 smoke checks. Its [first six-case benchmark](../benchmarks/inertia-eigvals-minimum-initial.md), before K=3 fusion, has one win and five losses (0.393826x geometric mean); all 84 raw samples remain published. The completed combined-product preparation40386034670 and 12-case controlled campaign above now measure the eigenvalue gain independently; its four remaining original-library losses stay visible. At that eigenvalue checkpoint source coverage was **223 native-partial image APIs + 3 metadata functions = 226 registered functions**, with **245 upstream-only entries**. Full supported-profile qualification stays **203 image APIs**. NumPy preparation, clipping, list wrapping and unsupported solver domains remain explicit providers.

**Inertia tensor is integrated:** `measure.inertia_tensor` adds one native-partial API, reusing owned central moments and a small typed C++ tensor assembly. NumPy indexing, diagonal summation and exceptional/integer/complex tensor stages remain explicit providers. The 57 original-first cases pass 1,272 component checks across both profiles and dispatch modes, plus 88 shared checks; old moments records and arrays match exactly. Source `7456bfd` is merged as `37b296e`. The first installed run exposed an unnecessary FFT storage check on a valid 3D moment tensor; removing six NumPy-stage checks fixed it without changing arithmetic, fixtures or bounds. [Genuine minimum Release](https://github.com/bipxx/scikit-imagexx/actions/runs/40378736391) passes all eight installed contract pairs, older regressions and 1,361 smoke checks. Its [first one-thread comparison](../benchmarks/inertia-tensor-minimum-initial.md) retains all 84 samples: three of six wins, geometric mean 0.627507734x original/current. All 256-square cases win; all 1024-square cases lose (+5.379077 ms uint8, +0.955479 ms float32, +5.322718 ms float64). Those initial losses motivated the separately kept K=3 optimization above; the original first-run data remain unchanged. At that tensor checkpoint source coverage was **222 native-partial image APIs + 3 metadata functions = 225 registered functions**, with **246 upstream-only entries**. Supported-profile qualification stays **203 image APIs**. The subsequent eigenvalue integration is described above.

**Kept C/F layout optimization:** [the fixed 13-case one-thread comparison](../benchmarks/moments-normalized-layouts-same-host.md) gives **3.346642894x direct / 3.338426256x normalized gain** over the kept C-only version. The four real F-layout targets improve **50.841838826x / 51.004018874x**, with both candidate passes beating both baselines; all 13 cases beat scikit-image in both candidate passes. Nine C controls remain approximately flat (0.998731020x direct), with four direct and six normalized losses preserved. Exact measured source `7733003` is merged as `1344699`; genuine Release passes all seven normal/scalar groups plus 1,357 smoke checks, and component tests pass 4,180 checks. The unchanged kernel now receives C/F arrays with matching output strides. All 728 samples are published. At that checkpoint source coverage was 221 native-partial image APIs, with 203 qualified image APIs. The later inertia tensor integration is described above.

For future paired optimization runs, dispatch `ci.yml` with `minimum_only=true` and `prepare_only=true`: all Release/parity gates and the verified offline artifact remain, while redundant standalone timings are skipped. Default CI behavior is unchanged; actionlint and all 33 measurement-isolation metadata checks pass.

The initial two source batches added eight routes. Butterworth, the three LPI functions, window, three moments functions two derived moments functions, inertia_tensor, inertia_tensor_eigvals and two coordinate-moment functions bring the current inventory to **225 native-partial image APIs + 3 metadata functions = 228 registered functions**, plus **5 partial classes**, with **243 upstream-only entries**. Accepted qualified binary coverage remains **203 image APIs**. Component tests do not qualify a combined wheel/SDK or establish performance.

**Kept normalized-moments batch optimization:** [the controlled one-thread comparison](../benchmarks/moments-normalized-batch-same-host.md) measures **51.483161594x float32 / 38.987071152x float64 gain over the previous C++ path**. Across all five cases the direct gain is **4.478043258x**, reference-normalized gain **4.649099324x**; every candidate case beats scikit-image in both passes on one EPYC 7763 host. All three Hu direct losses remain recorded, including float32 +0.221164 microseconds and losses in both metrics. The bounded default 4x4 C-contiguous real path performs powers and SIMD/scalar division in one native call; other layouts/options retain the tested scalar path. Source `aecf521` is merged as `f2eee88` with identical production code. Validation retains 4,819 native controls, 4,276 component/shared passes and 64 additional accepted/fallback checks; None probes have no arithmetic credit. Full qualification stays 203. That C-only checkpoint exposed the F-layout gap addressed by the separate kept 13-case experiment above. No whole-image pipeline speedup is claimed.

**Derived moments source integration adds `measure.moments_normalized` and `moments_hu`.** The 76 prospective specifications produce 155 nodes per runtime/dispatch mode: **620 new checks plus 3,408 unchanged-family checks = 4,028 component passes**. Actual shared-source validation adds 248 parity/error checks and 998 metadata checks; the selector collects 1,357 smoke nodes without executing unrelated tiny-core families. Hu owns the typed C++ polynomial and passes 1,518 native controls in both normal and sanitizer builds. The initial normalization path owns its final typed scalar division, retaining NumPy scalar providers; the optimized C-contiguous default path above now owns the batched arithmetic. A GCC scheduling bug that hid 13 zero-mass warnings is fixed by materializing the real quotient before reading FP status. Both APIs preserve the frozen outcomes and remain native-partial. Their five distinct 4x4 benchmark cases are prepared outside timing; integer normalization has exact source-error coverage and no numerical or timing credit. Source is merged/pushed as `76cdd70`; [the genuine minimum Release](https://github.com/bipxx/scikit-imagexx/actions/runs/40350415600) passes both installed contract modes and all 1,357 smoke checks. Its [first original-library comparison](../benchmarks/moments-derived-minimum-initial.md) retains all 70 samples: Hu wins 3/3 cases (1.09250x/1.33762x/1.28224x); normalization loses both (0.400917x float32, 0.290620x float64), adding 68.814/59.928 microseconds. Overall geometric mean is 0.737598829x. Those first losses motivated the kept batch path above. The later C/F optimization above admits bounded default F-layout inputs too. Full supported-profile qualification remains 203 image APIs. The 2,208-per-mode phase3 public barrier is configured, not yet an executed qualification. Component manifest `f2251086a232cc36079ae1807581980a09b5cf5a102c28083b253325596fbeac` and shared-source manifest `71db6f2aab6997d9ddcecd66b37eede96c56975f0024144daa81c2f3270d0aac` retain the original warning failure and metadata-only corrections.

**Kept moments optimization:** [the one-thread same-host comparison](../benchmarks/moments-f64-wide-same-host.md) reports **1.323055310x direct / 1.338931833x reference-normalized gain** over the previous C++ version across all 18 fixed cases. The six large uint8/float64 targets improve **1.742253016x directly**, with both candidate runs faster than both baseline runs for every target. Three small float32 cases regress by 0.000530–0.000831 ms; all losses and 1,008 samples are retained. On this EPYC 9V45 host, only 9/18 cases beat scikit-image in both candidate runs. Source `19464f2` passes 2,872 public component checks and 6,569,804 native controls, with saved arrays/journals exact against the earlier implementation. Its wider double SIMD loop is merged as `fb3ea27`; main also contains the independently validated buffer guard, so the measured wheel is not an exact combined-main qualification. The subsequent normalized/Hu implementation is described above; test checkpoint `670cf96` and its original-only observations remain preserved.

**Read-only buffer compatibility is validated.** The shared storage boundary now admits exact immutable `bytes` roots, including stepped, reversed, Fortran, swapped-endian and unaligned views. The frozen 62 public specifications and 10 storage-only controls pass with workers1/4: 134 new nodes plus 718 unchanged family nodes in each minimum/modern and normal/scalar mode, totaling **3,408 passes**. Mutable/foreign roots and out-of-bounds views remain rejected. The four initial recorder-only failures are preserved separately; the corrected recorder leaves product code and parity formulas unchanged. Final evidence manifest: `77987b564027fa76811adf50e57ad5ce0ceda6953c8cfa2dd1b5d608305c23eb`. Both CI preparation paths retain the tests; that integration configured **1,651 public checks per mode** before the derived-moments addition. API counts and full supported-profile qualification are unchanged.

**Moments source integration adds `measure.moments`, `moments_central` and `centroid`.** The 100 prospective cases give 203 contract nodes per runtime/dispatch mode. Both pinned runtime profiles pass normal/scalar: **812 moments checks plus 2,060 unchanged-family checks = 2,872 passing component checks**. Standard real images and finite complex spacing/centers use owned f32/f64/c64/c128 per-axis projections; NumPy coordinate powers, exact owned analytical formulas, layout and centroid division remain providers. NP1 dot stays silent for contraction FP flags; the source bmm/matmul path dispatches its captured per-axis policy. Real contiguous SIMD paths preserve ascending reductions/no FMA; all saved arrays/journals match the initial generic implementation exactly. Six final native modes pass 6,568,856 controls, with changed units sanitized and the retained SDK donor uninstrumented. Actual registration passes 392 shared checks, 998 inventory/class/runner checks and 360 exact frozen-policy comparisons, with all 657 prior API/dtype budgets unchanged. The smoke selector collects 1,350 nodes without executing unrelated tiny-core families. Genuine minimum-Linux Release run `40333236658` on exact source `07a3074` passes 203 moments, 179 window, 161 LPI and 175 Butterworth installed tests in each normal/scalar mode, all 1,350 smoke checks and the older minimum regressions. Its [first moments result](../benchmarks/moments-minimum-initial.md) retains all 252 samples: 12/18 original-library wins and a 1.319698648× geometric mean (raw 1.350012476×, central 1.334406078×, centroid 1.275846780×). All six losses are 1024² uint8/float64 cases; centroid adds 1.052761700 ms / 1.033449486 ms respectively. Every case passes the frozen per-cell checker; this is a first original/current comparison, not an earlier-C++ gain. That first run identified float64 projection locality as the next bounded performance target; the measured follow-up above keeps the numerical policy unchanged. The moments integration configured 1,517 public checks per mode and registry222; full supported-profile acceptance remains 203 image APIs. High signed-mass conditioning, arbitrary protocols, other external buffer-root storage variants, extended arithmetic and print/log FP policies remain open; this is native-partial, not full C++ moments coverage.

Moments proofs are preserved under `build/moments-private-v1/` after retiring the merged worktree: `moments-fused-proof-v1/manifest.json` SHA `2614c273ec399c78f1817e57c8a062482c969e577bddf6824ce847bf668084be`; `moments-component-proof-v3/manifest.json` SHA `bbaa0d84d354c375d29891bc8a50b0657e58040d92fe711ae1c9999388e9cf50`; `moments-shared-source-proof-v2/manifest.json` SHA `87e449c4dc2a051f36000a96cbf8b9e38b09f2cf1bb7394f19315fa44486e0a5`. Earlier metadata-only preparation failures and both original-only cohorts are retained. The pre-candidate policy scalar-type correction preserved all formula factors/fixtures; cell-specific checks are reused instead of substituting a looser scalar allowance. Source is merged/pushed as `07a3074`; [installed Release and isolated one-thread measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40333236658) completed successfully at that exact source. All three final proof hashes are unchanged after preservation.

**Window source integration adds `filters.window`.** Its 89 frozen specifications give 179 nodes per runtime/dispatch mode: **716 window contract checks**, plus unchanged LPI/Butterworth, total **2,060 passing component checks** on minimum/modern and normal/scalar. Orders0/1 use owned C++ grid-constant sampling, including partial endpoints, zero-weight nonfinite propagation and source clipping. Four higher-order specifications remain explicit initial refusals (32 checks, no numerical parity credit), and mixed-empty shapes remain pending. SciPy 1D window generation and NumPy coordinates/clipping are providers, not fully C++ window math. Scalar/runtime AVX2 and optional global workers are validated; six native modes pass 2,623,728 controls. Sanitizers cover the changed native units, not the retained SDK donor. Actual source registration passes 136 shared checks and 994 metadata checks; the smoke selector collects 1,338 nodes without executing unrelated tiny-core families. All 654 prior API/dtype budgets remain unchanged; 21 formula checks match the prospective window policy. Genuine minimum-Linux Release run `40317428784` on exact source `33030dc` passes 179 window, 161 LPI and 175 Butterworth installed tests in each normal/scalar mode, all 1,338 smoke checks and the older minimum regressions. Its [first window result](../benchmarks/window-minimum-initial.md) has 2/2 original-library wins and a 7.288846162× geometric mean: Hann256² takes 0.391585759 ms versus 2.690202600 ms original; Hann1024² takes 5.049911500 ms versus 39.051996000 ms. All 28 samples and zero reported output errors are retained. These are shape-only float64-output calls, not three input-dtype cases; SciPy window and NumPy coordinate/clipping stages remain providers. Full supported-profile acceptance remains pending. Combined phase-3 preparation now requires 1,314 public checks per mode and registry219; qualified combined coverage remains 203 image APIs.

Window proofs are preserved under `build/window-filter-private-v1/` after retiring the merged worktree: `window-kernel-proof-v1/manifest.json` SHA `60882fc111d35678aa4756520fdd19a2fbdf7ff1612ec0171a5b81427484183f`; `window-component-proof-v1/manifest.json` SHA `f132fd6947b8f5f93ccbaae43454c02240e877b39742438d34497bf7ef1eea8e`; `window-shared-source-proof-v1/manifest.json` SHA `4b272e57820cb5d85aef75db06de859752830223b6320ec00185b8010eacee4f`. Earlier selector/hook-order harness failures are preserved; the clean exact-scope run passes without scientific source or tolerance changes. Source is merged/pushed as `33030dc`; [installed Release validation and isolated measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40317428784) are running at that exact source. All proof hashes are unchanged. CMake probe caches were preserved because automatic approval review rejected their recursive deletion.

**LPI source integration adds `filters.filter_forward`, `filter_inverse`, `wiener` and the partial `LPIFilter2D` class.** Modern and minimum profiles each pass 336 component checks in normal and scalar modes (161 LPI + 175 Butterworth), totaling **1,344 passes: 644 LPI + 700 Butterworth**. Six native modes pass **8,056,540 controls**. The corrected full-real FFT route resolves the 92 retained earlier failures without changing the frozen 80 LPI or 87 Butterworth specifications or numerical policy. NumPy arithmetic/array stages and Python callbacks remain providers; this is partial native FFT execution. The source-derived shared projection passes 54 analytic comparisons while all 645 previous API/dtype budgets remain unchanged. The actual source registry also passes 98 shared checks per runtime/dispatch mode (392 total): signatures, all three default dtypes and shapes, both worker settings, equivalent descriptors and dtype metadata. Genuine minimum-Linux Release run `40311937607` on source `c895db2` now passes 161 LPI and 175 Butterworth installed tests in each normal/scalar mode, plus all 1,334 smoke checks. Its [first LPI result](../benchmarks/lpi-minimum-initial.md) has 18/18 original-library wins and a 1.330318164× geometric mean, retaining 252 raw samples. Calls include a fresh filter/cache and Python callback; NumPy stages remain providers. At the LPI checkpoint, combined phase-3 preparation was wired for 1,135 contracts per mode but remains unrun for that product. Supported-profile qualification remains pending; accepted combined coverage stays 203 image APIs.

Component proof is preserved at `build/lpi-filter-private-v1/fft-full-real-proof-v1/manifest.json`, SHA-256 `05107d5e9ab0edd9b08c2c4963b47867692ab2de999be2008161f6c70b391e2a`. The registered-source proof `lpi-shared-source-proof-v1/manifest.json` under that directory has SHA-256 `36bfe6752d9f098a91b0f1a874f38714d7e9aa82556a6762d38d224f96bd057d`: 392 shared checks plus 994 inventory/class/runner metadata checks pass. The merged worktree is retired; its entire build/evidence directory is preserved with unchanged proof hashes. These are component results, not installed whole-product admission. Source is merged/pushed as `c895db2`; [installed Release and isolated one-thread measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40311937607) completed successfully at that exact source.

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

The [legacy regression comparison after Butterworth integration](../benchmarks/phase3-butterworth-legacy-same-host.md) is flat across the unchanged 73 cases: **1.000748117× direct / 0.999452166× normalized**, with **44 direct and 46 normalized losses** retained. Both candidate passes are slower than both baseline passes for intersection stepped, masked and cache-F cases (direct gains **0.918100× / 0.894351× / 0.877160×**). This one-host result establishes neither a cause nor a waiver. Preparation passes **974 public checks in each normal/scalar mode, 1,322 smoke checks, 21 metadata tests plus 30 subtests, and all 73 corpus checks per mode**. The corpus contains no Butterworth cases; the six-case Butterworth and 18-case LPI first results remain separate.

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
smoke checks. New LPI phase-3 preparation is wired for 1,135 per mode and remains pending;
the genuine minimum Release separately passed 1,334 smoke checks and the
18-case LPI benchmark. The phase-3 corpus retains 73 cases per mode and 21 metadata tests
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
