# Complete scikit-image implementation plan

## Current priority: phased fast run

The user's latest direction is tests first, then small implementation groups,
isolated one-thread benchmarks, and full supported-profile acceptance. Use the
[fast-run guide](FAST_RUN.md) for executable commands. The initial Windows
development smoke gate passed 1,286 checks: all 477 API surfaces, 206 registered
function signatures, 602 supported-dtype parity cases and one registry check.
That phase-1 run used the retained public203 package without rebuilding.
Phase 2 integrates five footprint, three colocalization, one Butterworth, three LPI and one window and three moments and two derived moments public functions, plus the partial `LPIFilter2D` class.
Current source coverage is 221 partial image APIs + 3 metadata functions (224 total), five partial classes and 247 upstream-only entries. The footprint component
contract passes 240 checks in normal and scalar modes, including two large
ellipse cases above the optional parallel-fill threshold. Inventory/runner
checks bring the normal scoped run to 735 passing tests; the five-feature
shared smoke also passes 26 checks. The CMake portable configuration verifies
strict FP and worker-refusal test wiring without rebuilding the whole product.
Colocalization's 545 public component checks pass in normal and scalar modes;
its normal run with inventory/runner checks passes 1,040 tests, and its shared
smoke passes 16. Its 1,608 original semantic projections are now a self-contained
224 KB compressed test fixture, checked against the existing frozen matrix.
AVX2 and portable CMake configurations confirm baseline/leaf ISA separation,
strict FP and optional-worker test wiring. Accepted combined binaries remain
public203; combined wheel/SDK qualification for the additions is pending.
One-thread performance results are now available below. Existing full contracts retain their scope; older
campaign priorities below do not supersede this phased route.

**Kept normalized-moments batch optimization:** [the controlled one-thread comparison](../benchmarks/moments-normalized-batch-same-host.md) measures **51.483161594x float32 / 38.987071152x float64 gain over the previous C++ path**. Across all five cases the direct gain is **4.478043258x**, reference-normalized gain **4.649099324x**; every candidate case beats scikit-image in both passes on one EPYC 7763 host. All three Hu direct losses remain recorded, including float32 +0.221164 microseconds and losses in both metrics. The bounded default 4x4 C-contiguous real path performs powers and SIMD/scalar division in one native call; other layouts/options retain the tested scalar path. Source `aecf521` is merged as `f2eee88` with identical production code. Validation retains 4,819 native controls, 4,276 component/shared passes and 64 additional accepted/fallback checks; None probes have no arithmetic credit. Full qualification stays 203. The next concrete gap is Fortran layout: all 16 saved original 2D central-moment outputs are F-contiguous, and 11 finite real tensors meet the numeric bounds but fail only the C-layout gate. No end-to-end image-pipeline speedup is claimed; those saved tensors and C copies are the next prospective cohort.

**Derived moments source integration adds `measure.moments_normalized` and `moments_hu`.** The 76 prospective specifications produce 155 nodes per runtime/dispatch mode: **620 new checks plus 3,408 unchanged-family checks = 4,028 component passes**. Actual shared-source validation adds 248 parity/error checks and 998 metadata checks; the selector collects 1,357 smoke nodes without executing unrelated tiny-core families. Hu owns the typed C++ polynomial and passes 1,518 native controls in both normal and sanitizer builds. The initial normalization path owns its final typed scalar division, retaining NumPy scalar providers; the optimized C-contiguous default path above now owns the batched arithmetic. A GCC scheduling bug that hid 13 zero-mass warnings is fixed by materializing the real quotient before reading FP status. Both APIs preserve the frozen outcomes and remain native-partial. Their five distinct 4x4 benchmark cases are prepared outside timing; integer normalization has exact source-error coverage and no numerical or timing credit. Source is merged/pushed as `76cdd70`; [the genuine minimum Release](https://github.com/bipxx/scikit-imagexx/actions/runs/40350415600) passes both installed contract modes and all 1,357 smoke checks. Its [first original-library comparison](../benchmarks/moments-derived-minimum-initial.md) retains all 70 samples: Hu wins 3/3 cases (1.09250x/1.33762x/1.28224x); normalization loses both (0.400917x float32, 0.290620x float64), adding 68.814/59.928 microseconds. Overall geometric mean is 0.737598829x. Those first losses motivated the kept batch path above. Fortran-layout inputs still use the scalar path. Full supported-profile qualification remains 203 image APIs. The 1,806-per-mode phase3 public barrier is configured, not yet an executed qualification. Component manifest `f2251086a232cc36079ae1807581980a09b5cf5a102c28083b253325596fbeac` and shared-source manifest `71db6f2aab6997d9ddcecd66b37eede96c56975f0024144daa81c2f3270d0aac` retain the original warning failure and metadata-only corrections.

**Kept moments optimization:** [the one-thread same-host comparison](../benchmarks/moments-f64-wide-same-host.md) reports **1.323055310x direct / 1.338931833x reference-normalized gain** over the previous C++ version across all 18 fixed cases. The six large uint8/float64 targets improve **1.742253016x directly**, with both candidate runs faster than both baseline runs for every target. Three small float32 cases regress by 0.000530–0.000831 ms; all losses and 1,008 samples are retained. On this EPYC 9V45 host, only 9/18 cases beat scikit-image in both candidate runs. Source `19464f2` passes 2,872 public component checks and 6,569,804 native controls, with saved arrays/journals exact against the earlier implementation. Its wider double SIMD loop is merged as `fb3ea27`; main also contains the independently validated buffer guard, so the measured wheel is not an exact combined-main qualification. The subsequent normalized/Hu implementation is described above; test checkpoint `670cf96` and its original-only observations remain preserved.

**Read-only buffer compatibility is validated.** The shared storage boundary now admits exact immutable `bytes` roots, including stepped, reversed, Fortran, swapped-endian and unaligned views. The frozen 62 public specifications and 10 storage-only controls pass with workers1/4: 134 new nodes plus 718 unchanged family nodes in each minimum/modern and normal/scalar mode, totaling **3,408 passes**. Mutable/foreign roots and out-of-bounds views remain rejected. The four initial recorder-only failures are preserved separately; the corrected recorder leaves product code and parity formulas unchanged. Final evidence manifest: `77987b564027fa76811adf50e57ad5ce0ceda6953c8cfa2dd1b5d608305c23eb`. Both CI preparation paths retain the tests; that integration configured **1,651 public checks per mode** before the derived-moments addition. API counts and full supported-profile qualification are unchanged.

**Moments source integration adds `measure.moments`, `moments_central` and `centroid`.** The 100 prospective cases give 203 contract nodes per runtime/dispatch mode. Both pinned runtime profiles pass normal/scalar: **812 moments checks plus 2,060 unchanged-family checks = 2,872 passing component checks**. Standard real images and finite complex spacing/centers use owned f32/f64/c64/c128 per-axis projections; NumPy coordinate powers, exact owned analytical formulas, layout and centroid division remain providers. NP1 dot stays silent for contraction FP flags; the source bmm/matmul path dispatches its captured per-axis policy. Real contiguous SIMD paths preserve ascending reductions/no FMA; all saved arrays/journals match the initial generic implementation exactly. Six final native modes pass 6,568,856 controls, with changed units sanitized and the retained SDK donor uninstrumented. Actual registration passes 392 shared checks, 998 inventory/class/runner checks and 360 exact frozen-policy comparisons, with all 657 prior API/dtype budgets unchanged. The smoke selector collects 1,350 nodes without executing unrelated tiny-core families. Genuine minimum-Linux Release run `40333236658` on exact source `07a3074` passes 203 moments, 179 window, 161 LPI and 175 Butterworth installed tests in each normal/scalar mode, all 1,350 smoke checks and the older minimum regressions. Its [first moments result](../benchmarks/moments-minimum-initial.md) retains all 252 samples: 12/18 original-library wins and a 1.319698648× geometric mean (raw 1.350012476×, central 1.334406078×, centroid 1.275846780×). All six losses are 1024² uint8/float64 cases; centroid adds 1.052761700 ms / 1.033449486 ms respectively. Every case passes the frozen per-cell checker; this is a first original/current comparison, not an earlier-C++ gain. That first run identified float64 projection locality as the next bounded performance target; the measured follow-up above keeps the numerical policy unchanged. The moments integration configured 1,517 public checks per mode and registry222; full supported-profile acceptance remains 203 image APIs. High signed-mass conditioning, arbitrary protocols, other external buffer-root storage variants, extended arithmetic and print/log FP policies remain open; this is native-partial, not full C++ moments coverage.

Moments proofs are preserved under `build/moments-private-v1/` after retiring the merged worktree: `moments-fused-proof-v1/manifest.json` SHA `2614c273ec399c78f1817e57c8a062482c969e577bddf6824ce847bf668084be`; `moments-component-proof-v3/manifest.json` SHA `bbaa0d84d354c375d29891bc8a50b0657e58040d92fe711ae1c9999388e9cf50`; `moments-shared-source-proof-v2/manifest.json` SHA `87e449c4dc2a051f36000a96cbf8b9e38b09f2cf1bb7394f19315fa44486e0a5`. Earlier metadata-only preparation failures and both original-only cohorts are retained. The pre-candidate policy scalar-type correction preserved all formula factors/fixtures; cell-specific checks are reused instead of substituting a looser scalar allowance. Source is merged/pushed as `07a3074`; [installed Release and isolated one-thread measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40333236658) completed successfully at that exact source. All three final proof hashes are unchanged after preservation.

**Window source integration adds `filters.window`.** Its 89 frozen specifications give 179 nodes per runtime/dispatch mode: **716 window contract checks**, plus unchanged LPI/Butterworth, total **2,060 passing component checks** on minimum/modern and normal/scalar. Orders0/1 use owned C++ grid-constant sampling, including partial endpoints, zero-weight nonfinite propagation and source clipping. Four higher-order specifications remain explicit initial refusals (32 checks, no numerical parity credit), and mixed-empty shapes remain pending. SciPy 1D window generation and NumPy coordinates/clipping are providers, not fully C++ window math. Scalar/runtime AVX2 and optional global workers are validated; six native modes pass 2,623,728 controls. Sanitizers cover the changed native units, not the retained SDK donor. Actual source registration passes 136 shared checks and 994 metadata checks; the smoke selector collects 1,338 nodes without executing unrelated tiny-core families. All 654 prior API/dtype budgets remain unchanged; 21 formula checks match the prospective window policy. Genuine minimum-Linux Release run `40317428784` on exact source `33030dc` passes 179 window, 161 LPI and 175 Butterworth installed tests in each normal/scalar mode, all 1,338 smoke checks and the older minimum regressions. Its [first window result](../benchmarks/window-minimum-initial.md) has 2/2 original-library wins and a 7.288846162× geometric mean: Hann256² takes 0.391585759 ms versus 2.690202600 ms original; Hann1024² takes 5.049911500 ms versus 39.051996000 ms. All 28 samples and zero reported output errors are retained. These are shape-only float64-output calls, not three input-dtype cases; SciPy window and NumPy coordinate/clipping stages remain providers. Full supported-profile acceptance remains pending. Combined phase-3 preparation now requires 1,314 public checks per mode and registry219; qualified combined coverage remains 203 image APIs.

Window proofs are preserved under `build/window-filter-private-v1/` after retiring the merged worktree: `window-kernel-proof-v1/manifest.json` SHA `60882fc111d35678aa4756520fdd19a2fbdf7ff1612ec0171a5b81427484183f`; `window-component-proof-v1/manifest.json` SHA `f132fd6947b8f5f93ccbaae43454c02240e877b39742438d34497bf7ef1eea8e`; `window-shared-source-proof-v1/manifest.json` SHA `4b272e57820cb5d85aef75db06de859752830223b6320ec00185b8010eacee4f`. Earlier selector/hook-order harness failures are preserved; the clean exact-scope run passes without scientific source or tolerance changes. Source is merged/pushed as `33030dc`; [installed Release validation and isolated measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40317428784) are running at that exact source. All proof hashes are unchanged. CMake probe caches were preserved because automatic approval review rejected their recursive deletion.

**LPI source integration adds `filters.filter_forward`, `filter_inverse`, `wiener` and the partial `LPIFilter2D` class.** Modern and minimum profiles each pass 336 component checks in normal and scalar modes (161 LPI + 175 Butterworth), totaling **1,344 passes: 644 LPI + 700 Butterworth**. Six native modes pass **8,056,540 controls**. The corrected full-real FFT route resolves the 92 retained earlier failures without changing the frozen 80 LPI or 87 Butterworth specifications or numerical policy. NumPy arithmetic/array stages and Python callbacks remain providers; this is partial native FFT execution. The source-derived shared projection passes 54 analytic comparisons while all 645 previous API/dtype budgets remain unchanged. The actual source registry also passes 98 shared checks per runtime/dispatch mode (392 total): signatures, all three default dtypes and shapes, both worker settings, equivalent descriptors and dtype metadata. Genuine minimum-Linux Release run `40311937607` on source `c895db2` now passes 161 LPI and 175 Butterworth installed tests in each normal/scalar mode, plus all 1,334 smoke checks. Its [first LPI result](../benchmarks/lpi-minimum-initial.md) has 18/18 original-library wins and a 1.330318164× geometric mean, retaining 252 raw samples. Calls include a fresh filter/cache and Python callback; NumPy stages remain providers. At the LPI checkpoint, combined phase-3 preparation was wired for 1,135 contracts per mode but remains unrun for that product. Supported-profile qualification remains pending; accepted combined coverage stays 203 image APIs.

Component proof is preserved at `build/lpi-filter-private-v1/fft-full-real-proof-v1/manifest.json`, SHA-256 `05107d5e9ab0edd9b08c2c4963b47867692ab2de999be2008161f6c70b391e2a`. The registered-source proof `lpi-shared-source-proof-v1/manifest.json` under that directory has SHA-256 `36bfe6752d9f098a91b0f1a874f38714d7e9aa82556a6762d38d224f96bd057d`: 392 shared checks plus 994 inventory/class/runner metadata checks pass. The merged worktree is retired; its entire build/evidence directory is preserved with unchanged proof hashes. These are component results, not installed whole-product admission. Source is merged/pushed as `c895db2`; [installed Release and isolated one-thread measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/40311937607) completed successfully at that exact source.

Phase 3 has measured all 73 fixed workloads on fresh offline Linux hosts after
785 public checks in each dispatch mode, 1,318 smoke checks and exact corpus
parity. The [minimum baseline](../benchmarks/phase3-minimum-initial.md) has 52/52
footprint workloads faster than the original; the
[modern baseline](../benchmarks/phase3-modern-initial.md) has 50/52. All 21
colocalization workloads are slower on both baselines.

Merged [affine v1](../benchmarks/phase3-affine-v1.md) and
[width v2](../benchmarks/phase3-width-v2.md) improve the minimum-profile fixed
21-case geometric-mean ratio along **0.0342836736027× → 0.0574776398746× →
0.0919599844761×**. Width v2's measured source `4a78e94` is integrated in
main `78976b7` with unchanged product source. Its **1.599926× improvement over
v1** is **2.682326× over baseline**; every case improves over v1, and all 21
remain slower than scikit-image. These are original/current ratios from
separate hosted runs, not same-host paired candidate/baseline timings. All raw
samples and regressions are published; no all-profile or whole-product claim
follows.

Earlier kept [float64 v3](../benchmarks/phase3-f64-same-host.md) gave
+2.94% normalized / +2.03% direct on one 7763 host. Its cross-CPU HOLD is
historical and resolved; that earlier apparent 12% is not a causal code gain.

Earlier kept [byte-mask v4](../benchmarks/phase3-byte-same-host.md), source
`0135305`, established five original-library intersection wins. Its retained
9V74 measurements are historical, not factors for cumulative cross-CPU gains.

**Earlier kept colocalization: [float32 v5](../benchmarks/phase3-f32-same-host.md), main/checked
source `5b03eded`, exact compiled product `c53198a`.** The unchanged wheel was
reused. One 9V45 ABBA job gives fixed21 scores **0.2042894437937× →
0.2181707370842×**, or **1.067949× normalized / 1.059884× direct gain**.
All ten targeted float32 cases improve 1.093572–1.247566×, with both candidate
medians below both baseline medians. Five normalized and seven direct losses
are retained. The same five cases beat scikit-image; **16 colocalization
regressions remain**. All 73 workloads, 4,088 samples, reference drift and
product identities are published; no universal-profile or full-product
qualification follows.

The initial [preparation 40273141051](https://github.com/bipxx/scikit-imagexx/actions/runs/40273141051)
remains 741 passed / 44 failed. [Diagnostic 40275501905](https://github.com/bipxx/scikit-imagexx/actions/runs/40275501905)
and its [raw receipt](evidence/colocalization-live-reference-checkpoint.json)
prove old/new native observations exactly match fresh originals for all 536
normal/scalar cases, while all reproduce the same 44 frozen MOC word
differences. V5 is unnecessary for that discrepancy on this host; its precise
runtime cause is still unknown. The diagnostic keeps `acceptance=false`.

The separately accepted test policy uses a fresh original only for successful
MOC floating result words, preserving frozen metadata and every other field;
full native/fresh matching remains exact, with no tolerance relaxation or
fixture/matrix/helper rewrite. [New preparation 40277634220](https://github.com/bipxx/scikit-imagexx/actions/runs/40277634220)
passes 785 public checks per mode, 1,318 smoke, 21 metadata tests plus 30
subtests, and 73 corpus checks per mode. Its raw fresh-reference pairs are
verified; the old failure is not relabelled. The separate controlled
measurement, not the diagnostic alone, supports KEEP.

**Packed float32 v6 is rejected:** candidate `8641d0c` remains unmerged;
[the published comparison](../benchmarks/phase3-f32-packed-same-host.md) retains
all raw reports and losses. Actual `movups`/`addps`, 2,787 native controls per
mode and [788 installed tests per mode](https://github.com/bipxx/scikit-imagexx/actions/runs/40285975359)
passed. Same-host ABBA gives **0.986506× normalized / 0.994889× direct gain**,
with 13 normalized and 10 direct fixed21 losses. The packed path remains excluded.

**Integer octahedron is kept and merged in main `f079c6d`**, with product
source exact to measured `9da7ccf`. Its [published same-host comparison](../benchmarks/phase3-octahedron-integer-same-host.md)
gives **12.138864× normalized / 12.070450× direct gain** across all 12 targets
on one 9V45 host; both candidate medians beat both baselines in each target.
Private component checks are 254 public / 2,997 native per mode, followed by
799 installed checks per mode. All 35 direct and 28 normalized losses outside
the target remain published, as do 16 colocalization regressions. This does
not expand whole-product acceptance or claim all-profile speedups.

The [byte-affine candidate](../benchmarks/phase3-byte-affine-same-host.md)
`887aff6` passes 785 installed checks per mode but remains **HOLD/unmerged**.
Controlled run 40288858109 gives 0.999459× direct / 0.999468× normalized over
fixed21: stepped intersection improves 2.313035×, offset by direct losses in
all 14 Manders cases. Local and actual hosted SDK objects have unchanged
Manders hot-loop instructions; placement alone does not establish causality.
Preserve all results; do not change the metric or repeat identical code.
Packed v6 stays rejected.

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

## Historical execution card — public203 integration checkpoint

The following card preserves an earlier checkpoint; it does not replace the
current phased priority above. The [work queue](work_items.json) records file
ownership. Accepted qualified coverage remains **203 native-partial image APIs**,
against the unchanged full inventory.

Execution checkpoint: **public203 main integration, 2026-10-25**. A prepared
the artifact-only TV workflow/publication/certificate handoff; B prepared the
validated public203 source composition; C completed saved-raw profile reviews
and prepared the sanitizer runtime handoff. Root owns exact launches, integration
and acceptance. The queue records ownership; live handles and terminal receipts
establish execution state. Completed handoffs are not active processes.

Modern [colocalization private parity](evidence/colocalization-modern-private-parity-checkpoint.json)
now has **1,072 passing normal assertion replays** and a fresh **1,073-pass scalar
gate**. The original normal pytest outcome remains 1,055 passed / 18 failed;
the source-derived test-only repair preserves every semantic assertion and zero
native credit for provider-only rows. Windows normal/scalar each passed 1,073
tests; its independent raw review verifies 2,144 exact observations. Minimum
normal/scalar each passed 1,073 tests and independently verified 2,144 rows,
retaining all 141 original profile differences. Corrected Windows and minimum
native controls each passed 387. Portable normal passed 538 nodes/536 serial
observations; scalar is running. Sanitizer family parity and public/SDK/performance
gates remain pending. Windows raw receipt:
`build/resume-20261025/colocalization-windows-private-parity-independent-review-v1/final-review.json`.

Footprint source `ad6f4cf` repairs the actual null-root and finite-complex
boundary defects. Modern normal and scalar gates each have **454 passed,
zero failures/errors/skips**; independent review verifies **904 exact rows**
against the frozen 226-case originals. [Minimum private parity](evidence/footprint-minimum-private-parity-checkpoint.json)
also passed 454 tests per mode and independently verified all 904 observations,
including the three original profile differences. Native controls passed
minimum 2,925 and portable 1,464. Windows v1 CRT-link and v2 resource-tool failures
are preserved; v3 relink passed independent review without recompilation.
Windows normal/scalar each passed 454 tests; portable normal/scalar each passed
229. Independent saved-raw review verifies all 1,356 observations across those
profiles. Portable's reviewed test successor has 226 serial observations plus two existing
metadata guards and one global worker-budget refusal control: 229 nodes, zero
parallel numerical credit. Sanitizer native normal/scalar each passed 2,925
controls, with four new TUs instrumented and SDK primitives/worker explicitly
uninstrumented. Sanitizer Python family gates and public, SDK and performance
acceptance remain pending. NumPy providers remain explicit.
The [original causal diagnoses](evidence/private-normal-gate-failure-diagnosis.json)
and failed ce3d run remain unchanged. Raw review:
`build/resume-20261025/footprint-ad6-modern-parity-independent-review/final-review.json`.

The [four-profile H diagnostic](evidence/h-four-profile-installed-diagnostic.json)
passed on exact failed741 binaries: **28,352 rows**, including 448 blocked
zero-call cases. Windows-only successor `40231513839` passed. Fresh ea35
whole-product qualification [40237475475](https://github.com/bipxx/scikit-imagexx/actions/runs/40237475475)
has [root whole-product admission](evidence/public203-whole-product-admission.json):
four profiles, **828,304 XML checks**, zero failures/errors/skips, three Release
products, 68 SDK members and 23 families. All seven raw archives are retained and
independently verified. The [main composition](evidence/public203-main-integration.json)
matches all 298 qualified computational Git paths and preserves newer tooling.
The exact admitted binaries remain ea35; this source merge is not a new binary
qualification. Original whole qualification `40200919182` stays failed.

Max-tree comparison `40200431797` ended cancelled: minimum preparation hit the
explicit six-hour limit; Windows and modern preparation passed, measurement was
skipped. Three receipt ZIPs are retained and independently verified. The reviewed
six-job profile/mode split is merged as `834b1ae`; successor
[40237980827](https://github.com/bipxx/scikit-imagexx/actions/runs/40237980827)
is in progress, preserving all cases, workers 1/4 and the isolated one-worker
measurement method. Require all six fresh receipts; do not redispatch or relabel
old partial receipts. No new timing is accepted. See the
[terminal record](evidence/max-tree-default-comparison-timeout.json).

The [TV comparator successor](evidence/tv-positive-zero-ea35-source-admission.json)
`6662516` explicitly requires ea35/run40237475475, with 723 metadata checks.
All cases, before-product pins and timing bodies remain unchanged. Workflow and
certificate source `9845cad8` is ready for root review; publication/dispatch and
the actual isolated comparison remain. No zero-helper speedup is established. Retained
af7 results identify short-iteration F-order rasters as an absolute-cost target,
with a maximum minimum-profile median penalty of 1.477 ms. This is an existing
measurement, not attribution to a particular scan or a new optimization result.

Manifest-read deduplication and queue ownership repairs are complete. Preserve
policy-blocked scratch paths and active inputs. The [compilexx project notes](../../compilexx/scikit-imagexx-plan-inefficiencies.md)
record lessons; this card and the queue own live execution state.

## Historical execution notes

Execution resumed by explicit user request on 2026-10-25: implement this plan
using three agents. At that checkpoint, A owned the H floating production successor;
B independently diagnoses failed public200 and reviews the new TV campaign;
C owns H original references and the historical assertion migration plan.
Root owns shared registration, CMake, public integration and final acceptance.
The paused planning snapshot below records the audit context; its prohibition
on implementation has been superseded by the explicit resume. Existing
acceptance gates and frozen evidence remain unchanged.

Resume progress: threshold references `1329cbb` and `337ba1b` are reviewed and
merged as `767e2ca`. Their [expanded audit](evidence/iterative-threshold-expanded-audit.json)
verifies 1,137 original-reference checks and distinguishes 456 default-policy
Li solver matches from 24 unsupported warning-policy projections. The broader
312-case audit retains its unsupported/provider/unsafe partitions. This adds no
native API paths or performance result.

The independently reviewed TV comparison controller is merged as `9f834bb`;
its root-reviewed experiment keeps actual `5e068a5` and `af7b2af` product labels.
Run `40163025231` is terminal successful, with all ten raw artifacts independently
authenticated. Its [policy-gap audit](evidence/tv-admission-comparison-policy-gap.json)
keeps the measurements diagnostic: the legacy all-warn observer bypassed the
changed default-policy finite-loop route. A separate passive default-policy70
observer and fail-closed v2 receipts are being implemented before a fresh run.
No new speedup is accepted.

The original 199-path run `40161615893` remains terminal failed and fully retained.
The corrected exact `d92d9ff` product/run `40166462583` now has
[root whole-product admission](evidence/public199-whole-product-admission.json):
four passing profiles, 813,192 XML checks with zero failures/errors/skips,
three actual Release products and 20 SDK families. Its seven new partial paths
are merged as `32f6488`. Accepted native-partial inventory is now 199; this does
not establish full argument coverage or new performance.

The [max-tree comparison consumer](evidence/max-tree-comparison-reviewed.json)
is reviewed and merged as `9c18e27`, with 345 three-runtime metadata checks and
115 main-layout checks. Genuine199 products are now admitted; certificate and
exact timed-path review remain before native preparation/measurement. Two merged
consumer worktrees were [retired with archived proofs](evidence/max-tree-consumer-worktree-retirement.json);
the remaining worktree is retained for separate opaque-link review.

The [shared experiment publisher](evidence/shared-experiment-publication.json)
is merged as `c24f7b3`; its run `40162423182` published the authenticated TV
certificate. Its merged worktree is retired with all ignored proof bytes archived.
NMI's [original reference freeze](evidence/nmi-original-reference-audit.json)
is merged as `b5fadfa`, while [private native admission](evidence/nmi-private-native-review.json)
for `fadef4c` adds no public/SDK or performance acceptance.

Reconstruction's [offset repair](evidence/reconstruction-offset-private-review.json)
has scoped private admission. Its [public200 composition `b4a5c3a`](evidence/reconstruction-public200-source-review.json)
has 1,947 metadata checks and four clean plans; a subsequent source-only
direct-C++ descriptor-overlap finding requires narrow hardening before freezing; its
595 public checks were collected only. Whole-product/SDK acceptance is pending.

The [metrics SIMD increment `0322181`](evidence/metrics-endian-range-private.json)
is merged as `edb897e` after independent private source/parity/closure review:
172,832 raw-range/FEnv controls, six full original case reports and 288 focused
PSNR checks. It is an unmeasured optimization of a retained endian-float32
bottleneck. Compose this exact delta into the next cumulative200 candidate
before whole qualification, avoiding a separate product build solely for it.
The admitted199 products remain pinned to `d92d9ff`; the separately merged
metrics delta is not part of those binaries.

Li's bounded uint16 extension `2abd0ac` is [privately admitted](evidence/li-u16-private-admission.json)
and merged as `4d8f54f`: values0..255 reuse the unchanged byte histogram/solver,
with checked owner extents and alignment-safe copies. The independently reviewed
[installed gate](evidence/li-u16-installed-gate-source-review.json) requires 86 exact
observations and 42 actual solver completions per mode, plus an assertion-enabled
native CTest. Its runtime execution, cumulative product and timing remain pending.

The [reconstruction benchmark reference](evidence/reconstruction-benchmark-reference-admission.json)
is merged as `dd32011`: 192 original observations and 270 checks, with no native
or timing credit. Multiple markers, barrier/plateau/corridor and high-cardinality
floating terrain supplements are mandatory before its first timing campaign.

Latest integration checkpoint: [cumulative200 source `bedab75`](evidence/public200-cumulative-source-preflight.json)
combines main's metrics/Li increments and reconstruction descriptor hardening.
It has 716 metadata checks and four explicit200-path plans with zero blockers;
whole-product qualification waits final independent private-successor admission.
The initial strict staging mismatch and corrected successor pins are retained.

TV's repaired controller is merged as `28cfc58`. Its independently reviewed new
certificate is [publishing once as run40173372973](evidence/tv-default-certificate-publication-dispatch.json).
Publication artifact acceptance and a new actual-default-path comparison remain
pending; old run40163025231 stays diagnostic. Agent C's prospective16 terrain
supplement fills the original48 cohort's marker/barrier/high-cardinality gaps.
Agents A/B are independently reviewing each other's sealed private successors.

Historical checkpoint: [parallel implementation evidence](evidence/parallel-implementation-checkpoint-20261025.json).
Original public200 source2835001/run40174473080 is terminal failed across four
profiles, fully retained and unaccepted. Independent actual-failure diagnosis is
in progress. The inventory prebuild guard is merged as de5e669; candidate source
declarations must agree with registration before builds. Accepted products remain
exact d92d9ff/199; privately admitted Li, metrics and reconstruction successors
still require successful cumulative qualification.

New TV comparison40174240551 completed all eight jobs successfully. All ten raw
artifacts are authenticated; root raw/source/auth review validates4410 timing
intervals and fresh default70 dispatch. Independent final evidence/math review
is pending, so no new performance result is accepted. Old40163025231 remains
diagnostic. NMI staging4b82e81 closes its scoped source blocker but is unqualified.
H public202 remains blocked on preserved floating/default availability; the
owned floating successor must cover all378 proven follow-ups, not a temporary
narrow bridge. The additional70 source-frozen controls have original-only first
captures and exact repeats; implementation and independent final review continue.
Reconstruction's16-case terrain supplement is reviewed and merged as ccc6197.
Earlier progress paragraphs above remain historical checkpoints.

Latest bounded measurement: [TV default-policy campaign](TV_ADMISSION_PERFORMANCE.md)
is root accepted for exact192 products,70fixtures/threeprofiles/globalworker1.
All4410intervals/3780default observations are independently checked on Windows
and Linux; allregressions remain in the210-rowCSV. After/original geometric
mean speedups are1.557x/1.276x/1.358x;27/31/28cases remain slower. This is no
full-API performance claim. B owns its measured-bottleneck follow-up plan.

The exact archived failed283 Li products are being diagnosed by reviewed
controller92f2a79/run40183231914, dispatched once with no build/timing. Original
observer8118 remains the authority; source-only checkout portability follow-up
does not mutate that source/run. C owns the exactzero-testobserver collection
policy and metadata guards. H9b metadata migration is reviewed/admitted; A
continues floating production, with source/donor review before any native calls.
Earlier progress paragraphs retain their historical pending checkpoints.

Historical paused planning snapshot (superseded by active execution above):
Planning revision: 2026-10-25, audited on 2026-10-24–25 from main
`12fad3488178f52d52d8b4fd161e59790fe63d17`. The implementation goal is **paused**.
This revision authorizes no implementation, build, image call, measurement,
hosted dispatch, implementation-candidate merge/push, or cleanup. Validation and
publication of these planning documents are permitted. The assignments below are
the execution plan for an explicit resume, not a claim that workers are running.

Implement the complete pinned scikit-image 0.26.0 public API in optimized C++,
with interchangeable Python calls, CPU SIMD where useful, one optional
process-global worker setting that defaults to one, per-function parity, and
isolated one-worker comparisons with the original library. Classes, their state
and members, nested namespaces, data, and I/O remain in scope. A Python wrapper,
upstream forwarding, a metadata count, or a private helper is not a completed
native algorithm.

Use these documents together:

- [Per-API implementation paths](API_IMPLEMENTATION_PATHS.md) and their
  [machine-readable records](API_IMPLEMENTATION_PATHS.json) give every one of the
  477 inventoried keys an easiest viable route, dependencies, hazards, native
  frontier, SIMD/worker policy, next action, and measurement recipe.
- [Astra plan review](PLAN_REVIEW_ASTRA.md) explains the evidence and tradeoffs
  behind this revision.
- [work_items.json](work_items.json) records packet ownership and proof pointers.
  Its explicitly historical paused checkpoints are retained; current assignments
  and admissions are recorded in execution_policy and packet checkpoints.
- [api.json](api.json) is the observed API inventory. It describes implementation
  classification, not accepted full argument coverage or speed.
- The appendix below retains the important historical checkpoints. Historical
  source labels, raw results, failures, and proof scopes must not be rewritten.

## Accepted scope and frozen handoffs

| Inventory classification | Public keys | Meaning |
|---|---:|---|
| Native-partial image functions | 199 | An advertised subset executes natively; full argument and performance closure is still required. |
| Metadata functions | 3 | Separate from computational algorithms. |
| Native-partial classes | 4 | Coordinate/state support exists; independent fitting and matrix solvers remain incomplete. |
| Exception metadata provider | 1 | Compatibility identity, not a numerical implementation. |
| Upstream-only | 270 | Available through compatibility machinery, not native completion. |
| Total | 477 | Includes aliases, classes, metadata, assets, and I/O; it is not an algorithm count. |

The inventory includes 29 classes and 284 statically discovered member paths.
Runtime fields, generated attributes, constructor/state behavior, and additional
public namespaces need their own completion audit. The 19 `rank.generic` and
three bilateral aliases reuse canonical algorithms but still need public path
and signature checks. The old ledger's 391 unassigned API keys are assignment
accounting, not 391 unimplemented algorithms.

| Checkpoint | Evidence already available | First action after resume | Acceptance boundary |
|---|---|---|---|
| Accepted TV admission, actual source `af7b2af`, merged as `bbc1fc1` | [Four-product qualification](evidence/tv-single-admission-whole.json): 707,800 mixed checks, 144 SDK checks, 110 wheel members, 63 SDK members, 18 SDK families | Consume the actual immutable `af7b2af` products for the new comparison | Accepted 192-partial-path parity; this change adds no API paths and has no new speed result. |
| Combined 199-path candidate `3c87fbfdc425c953a280eaabf5934cada746f56d` | `build/worktrees/public199-skylakex-readiness/build/composition/manifest.json`: 2,001 metadata guards, four plans with zero listed blockers, zero whole-product builds | Review the composed source and runtime admission, then qualify that exact source once across all four profiles | Not whole-qualified. Never promote seven APIs from plans, metadata guards, or a failed predecessor. |
| TV comparison controller `8dad040aef9c1cb0c7c4135ad8eff0fc8a136ddd` | `build/worktrees/tv-single-admission-compare/build/admission-controller-review/manifest.json`: four new files, 312 metadata guards, immutable product-pair checks | Review the exact header projection, common inputs, original/`5e068a5`/`af7b2af` roles and genuine product frames; reuse the fixed corpus | Draft controller only; no numerical experiment or timing has run. |
| Iterative thresholds `1329cbb` and Li default supplement `337ba1b` | Original reference and actual qualified-kernel audit are frozen; the separate Li supplement has 38 solver cases and two intentional error-policy exclusions | Review source-backed findings; separate executed solver cases from metadata, unsupported inputs, and expected errors; choose one concrete coverage gap | Root review pending. The expanded 312-case reference does not mean 312 native solver matches. |
| Reconstruction reference `600a403c1368b99e5d85200bc4f6eb6ed262ea0c` | 309 checks per runtime: 259 actual original Cython cases, 13 preentry rejects, three unsafe intake cases, plus other checks | Review the reference; decide signed-zero/provider behavior; preserve and then validate the unfinished scalar C++ draft | Reference only. The draft is unbuilt and has no native acceptance or timing. |

These handoffs have exact paths and next actions in the paused ledger. No frozen
source is silently rebased, relabelled as main, or replaced with a reconstructed
product. If matrix-specific admission still blocks a bounded 199-candidate
preflight, prepare a separate fresh max-tree-only candidate from reviewed
sources. That candidate must earn its own complete product qualification;
failed 199 evidence cannot accept its six APIs. Do not rerun or cancel an already
live campaign merely because an observer timed out.

## Decide the contract before choosing the implementation

For each API row, freeze a small contract matrix before inspecting new candidate
outputs. Cover the pinned signature/defaults, dtype and promotion, dimensions,
channel/axis semantics, memory layout and overlap, allocation/view behavior,
mutation, warnings/errors, callbacks, exceptional values, and provider calls.
Select cases by behavioral partitions and boundary conditions, not by maximizing
assertion counts. Add the relevant upstream tests and generated adversarial
cases; record unassessed partitions explicitly. A common registry smoke test is
necessary but does not replace this matrix.

Use the least complex route that satisfies the complete contract:

1. Reuse an already accepted canonical implementation for true aliases, while
   checking each public path independently.
2. Compose existing native primitives when their dtype, boundary, ordering, and
   ownership rules match. Do not introduce another kernel or provider adapter
   just to obtain a new API name.
3. Port a simple scalar C++ implementation of the reference algorithm where a
   primitive is missing. Establish finite, empty, exceptional, strided, and
   mutable-input behavior before adding fast paths.
4. Add SIMD and optional parallel work to independently computable hot loops.
   Preserve scalar dispatch and the existing global worker setting. Measure the
   public call, including conversions and allocation, before tuning further.
5. For FFT, sparse/dense linear algebra, and codecs, evaluate a suitable C++
   library behind a repository-owned contract instead of recreating a numerical
   ecosystem. Pin and document its build and threading behavior. A call to the
   original scikit-image algorithm, or a NumPy/SciPy algorithm provider, remains
   an explicit native gap. Genuine external protocols such as user callbacks,
   custom estimators, and codec plugins must retain their defined integration
   behavior without being counted as native solvers.

### Numerical parity policy

Structural behavior is exact: public routing, signature, shape, dtype/metadata,
container structure, view and mutation semantics, labels, parent/traverser
identities, stable ties, state transitions, warnings/errors, and callback order.
Byte-preserving copies/rearrangements and integer/discrete algorithms retain
their exact values. Preserve sign bits, NaN behavior, and arithmetic ordering
where they are observable in the advertised contract or already required by an
accepted strict proof.

For floating arithmetic, each contract chooses exactness or a justified
absolute/relative/ULP or residual-based comparison in advance. Use dtype,
conditioning, reference guarantees, and the algorithm's error propagation to
justify that choice. A generic `allclose` threshold is not a contract. Existing
strict proofs remain valid and unchanged; any different future numerical policy
is a separately reviewed prospective contract, with independent cases that were
not used to select it. Do not widen tolerances after looking at failures, adapt
to an oracle result, dispatch on fixture identities, or claim a provider result
as a C++ result.

Iterative algorithms need more than a final value check. Li callbacks, TV
stopping, reconstruction ordering, local optima, threshold ties, and random/model
solvers can amplify small differences. Check observable iteration/callback
behavior, convergence/failure semantics, downstream masks or labels, and
conditioning/near-boundary cases. Retain source-faithful ordering where those
semantics require it. CPU-specific bitwise imitation is justified only by a
required observable contract; it is not the default prerequisite for adding an
unrelated API.

## Three workers and one integrator

On resume, reconcile the frozen checkpoints first, then use the same three
workers for bounded follow-up packets. Worker names are handles, not permanent
specialties. The ledger must record actual source, worktree, writable paths,
read-only dependencies, runtime handles, exit gate, and next handoff.

| Role | First bounded packet after resume | Continuing lane |
|---|---|---|
| Root / integrator | Review the 199 candidate delta and checkpoint receipts; reconcile stale packets; authorize one exact candidate's qualification | Shared CMake/core/exports/SDK/registry wiring, final independent acceptance, prompt merge/push and safe cleanup. Delegate mechanical evidence verification; review code, scope, exceptions, and the verifier's receipt. |
| Worker A | Finish the frozen TV comparison review/preflight using existing products; hand observation to the existing controller once running | Close measured regressions in metrics, TV, rank, resize, or labeling, one measured hypothesis at a time. A running remote job alone does not occupy this worker. |
| Worker B | Review the threshold/Li native frontier and implement one justified missing contract partition | Native-partial breadth, simple draw/util/reduction APIs, and region statistics. Reuse shared primitives and obtain an independent reviewer before handoff. |
| Worker C | Resolve reconstruction's reference/provider boundary and finish the smallest scalar native packet from the preserved draft | Neighborhood, morphology, connectivity, then a dependent family. During external waits, take an independent class/state or data/I/O contract packet from lane P. |

Root assigns independent review to a worker who did not author the changed
algorithm. Rotate reviews as packets finish; do not add a fourth worker or make
root manually reproduce every mechanical validation. A worker is not alone in
the repository and must preserve others' changes. Each worker owns an isolated
worktree and cache; shared product artifacts are read-only.

Bound work in progress:

- One active implementation packet per worker, with at most one frozen handoff
  awaiting integration. A packet normally changes one shared primitive or a
  small coherent API family, not an entire namespace.
- Root keeps at most two new integration candidates waiting for review. Drain
  the existing frozen backlog before creating more competing controllers or
  cumulative products. A blocker is recorded with the exact missing dependency;
  the worker then takes an independent ready packet.
- Commit reviewable source and its scoped proof promptly. Merge and push a
  validated increment promptly on resume; performance-pending status may remain
  explicit after parity acceptance. Do not hold completed private kernels or
  tooling until unrelated algorithms finish.
- Prepare the next contract and timing cases during another packet's build or
  remote measurement. Keep builds/tests off the measurement host and never touch
  packages in use by the measurement.

## Waves and shared primitives

Waves describe dependency order, not whole-library barriers. An API may start
when its own dependencies are ready. The per-API roadmap assigns each key one
primary wave and records the actual prerequisites. Lane P starts early and runs
alongside the numerical work so data, I/O, and class state do not become a late
unplanned project.

| Wave | Deliverable and easiest route | Release criterion |
|---|---|---|
| W0 — frozen checkpoint closure | Review/qualify the 199 candidate, measure qualified TV, classify threshold gaps, resolve reconstruction's entry boundary | Each checkpoint gets its true accepted/pending result; no invented new coverage. |
| W1 — cheap independent APIs and partial-contract closure | Existing native dtype/layout/keyword gaps; direct draw loops; util views/copies and conversions; basic statistics/moments; canonical alias reuse | Small public contracts with minimal new machinery, plus per-function timing. |
| W2 — neighborhood, ordering, connectivity | Reuse histogram/reduction/boundary primitives; rolling windows; reconstruction queue; stable union-find/tree traversal; rank and morphology extensions | Exact topology/tie/output-buffer behavior and safe boundary handling; dependencies shared by later families. |
| W3 — geometry, resampling, solvers, classes | Existing typed resampling/integral primitives; coordinate generation; C++ least-squares/SVD/QR where justified; common fitting/state layer | Full constructor/state/member behavior and actual solver paths, not class metadata alone. |
| W4 — FFT, multiscale, registration, restoration, features | One FFT/convolution layer, pyramids and gradients; direct C++ ports of feature/denoising loops; reuse stable eigensolvers | Fixed numerical policies, boundary/normalization correctness, actual one-worker provider control. |
| W5 — dependent graph, segmentation, model solvers | Graph construction/traversal over accepted labels and features; source-faithful watershed and optimization; reuse sparse/dense solver facilities | Exact discrete decisions plus full state/callback/randomness and convergence coverage. |
| P — providers, assets, I/O, class/state contracts | Asset manifest/cache and loader rules; codec adapters; collection laziness; plotting/UI adapters; external model protocols and runtime attributes | Exact API behavior and measured relevant work, explicit dependency/native frontier, no fake SIMD or image-kernel speed claim. |

Start from the current implementations rather than building a second foundation:

| Shared facility | Existing starting points | Ownership and extension rule |
|---|---|---|
| Typed array intake, strides, conversion and output ownership | `src/image.cpp`, `src/bind_numeric.hpp`, typed resampling bindings | One owned descriptor/boundary contract; extend it for a real family need, avoiding implicit copies that alter mutation or dtype semantics. |
| Global workers and CPU dispatch | `src/execution.hpp`, `src/image.cpp`, `src/simd.hpp`, existing `simd_*_avx2.cpp` | Preserve portable scalar operation and one worker default. Choose a workload-size threshold; prohibit nested provider oversubscription. |
| Histograms, reductions, ordering | `src/threshold.cpp`, `src/li.cpp`, `src/multiotsu.cpp`, `src/rank.cpp`, `src/metrics.cpp` | Reuse code where arithmetic/tie contracts match. Do not force numerically different reductions through one implementation merely to share code. |
| Neighborhoods and boundaries | `src/gaussian_nd.cpp`, `src/morphology.cpp`, `src/morphology_boolean.cpp`, footprints and rank | Share offsets/modes/halos after freezing their rules; keep feedback/output-alias traversals serial when required. |
| Connectivity and ordered queues | `src/components.cpp`, `src/component_consumers.cpp`, reviewed max-tree work, reconstruction draft | Stable labels and parents are public results. Parallelize independent preparation before changing dependent traversal. |
| Geometry and scale | `src/resample.cpp`, `src/resize_range.cpp`, `src/resize_clip.cpp`, `src/projective.cpp`, integral/downscale code | Compose native coordinates, sampling, and dtype policy; separate state/wrappers from the numerical solver. |
| FFT, sparse solvers, codecs and assets | New narrowly scoped facilities only when a roadmap family needs them | Compare viable libraries against the pinned contract; do not create a general framework before the first consumer. |

SIMD applies to independent pixels, coordinates, scans, arithmetic, and selected
reduction stages. Parallelism applies to independent rows, tiles, channels, or
preparation work after a measured size threshold. Tree/queue dependencies,
Gauss-Seidel updates, callback order, stable tie reduction, lazy I/O state, and
metadata often require serial execution. They still honor the global worker
policy, but forcing a parallel algorithm is not an acceptance criterion.

## Qualification, evidence reuse, and early measurement

```mermaid
flowchart LR
    A[Per-API contract and numerical policy] --> K[Small native kernel or composition]
    A --> H[Original cases and timing corpus]
    S[Frozen shared interface] --> K
    K --> P[Actual installed family parity and SDK]
    P --> Q[Exact cumulative product qualification]
    H --> M[Isolated original vs native measurement]
    Q --> M
    M --> R[Retained regression and bottleneck packet]
    R --> K
```

Keep four distinct gates:

1. **Reference gate:** original source and safe original calls establish the
   contract. Metadata checks, expected failures, and unsafe blocked inputs are
   reported separately from successful numerical calls.
2. **Native family gate:** execute the real C++ binding and installed interface
   with fallback disabled. Check the changed family, dependent shared primitives,
   scalar/SIMD modes, default-one/optional workers, error/state behavior, and
   appropriate native/sanitizer/portable cases. A temporary adapter is private
   evidence only.
   Before the broad campaign, reuse the relevant schema/checkout regressions and
   run one ordinary owned-array call through each changed binding. Check the
   required cohort identities, not only a nonempty green XML or total count.
3. **Public product gate:** freeze source, wheel, SDK, dependency versions and
   build/runtime identity; run the existing genuine four-profile whole-product
   qualification for the actual cumulative candidate. The family gate catches
   defects early; it does not replace this release gate. Unaffected private
   evidence may be referenced only with explicit identical dependency closure.
4. **Performance gate:** consume the actual qualified artifacts and original
   0.26.0 in isolated one-worker roles. Verify the frozen corpus first, then time
   complete public calls. Retain every case, sample, regression and failed trial.

Use one reusable proof/measurement format and the existing qualification tools
for future families. A family supplies cases, comparisons, expected scope, and
commands; it should not need a new controller, retainer, publisher, and bespoke
certificate for each API. Any tool consolidation is a separate future packet:
preserve historical controllers and raw bytes, and validate the shared tool on
known success, failure, and tampered-input fixtures before adopting it.

A handoff needs source and owned-path deltas, full dependency fingerprints,
commands/results, actual artifact hashes, native/reference/provider/metadata
case counts, unresolved cases, process handles, and one independent verifier
receipt. Root reviews that bounded packet and its exceptions. Reuse an existing
verified immutable object by content hash and exact source/dependency scope;
never infer current behavior from a historical test count or relabel a product.
Fresh changed-source execution remains necessary. Do not repeatedly copy and
manually replay unchanged archives solely to create a new administrative seal.
State the changed inputs and reused receipt/closure explicitly in the existing
handoff. Do not add a separate manifest or controller just to repeat that split.
Name the next missing gate, owner and public/SDK integration destination in each
private family's existing packet. For future footprint observations, reuse the
colocalization node-to-raw-file mapping so worker identity is independently
recoverable. Preserve current evidence without inventing per-row labels.

Prepare performance corpora with the contract, and measure each newly qualified
family promptly. Existing comparisons already identify useful work: metrics,
large multivalued/high-connectivity labeling, footprint decomposition, rank
windowed histograms/noise-filter cases, and TV raster/volume cases. Prioritize
absolute time and representative workload impact, not only large speedup ratios
on tiny calls. Keep allocation/conversion/setup costs visible. Asset/I/O APIs
need reproducible fixture and cache-state measurements; metadata and state-only
APIs need relevant latency/behavior checks, not invented image throughput.
At the next qualified function, update its existing performance record with the
corpus, qualified product, measurement or precise blocker, worst retained
absolute regression, and next owner. After the current review, reserve one
execution lane for a measured bottleneck; choose its case from retained absolute
costs rather than a geometric mean. Missing absolute-cost data is a reporting
gap, not a reason to guess the most important case or claim it optimized.

No local timing experiments. Hosted timing uses one native worker and one
original provider thread, matched inputs, repeated paired rounds, and recorded
runtime/CPU identities. Additional worker counts are supplementary. Do not time
alongside builds/tests, rebuild a package being timed, discard adverse cases, or
claim a later source inherits an earlier speedup. A measured baseline with
regressions is progress, not completion of the optimization objective.

## Completion tracking and operations

Report progress by evidence rather than assertion totals:

- 477 public keys: contract assessed / native partial / full advertised native
  contract passed / measured / performance target met; split aliases, metadata,
  assets, and providers from canonical computational algorithms.
- 29 classes and all 284 static member paths: constructor/state/member/solver
  status, plus runtime-discovered fields and namespaces still to assess.
- Per family: tested dtype/layout/parameter partitions, native cases versus
  reference/provider/metadata cases, unsupported partitions, and retained
  regressions with absolute cost and next owner.
- Queue health: active owners and writable paths, frozen handoff age, blockers,
  exact live handles, next merge, and clean obsolete worktrees eligible for
  removal. These operational measures are not implementation acceptance.

The goal is complete only when every inventoried and newly discovered in-scope
public contract is accounted for, required computation is genuinely native,
per-function and class/state parity passes, installed C++/Python products are
qualified, and original-library one-worker evidence establishes the declared
performance result for each relevant API. Do not mark the goal complete with
unsupported argument partitions, hidden algorithm providers, unmeasured APIs,
or unresolved optimization targets disguised by aggregate averages.

These commands inspect the queue; they do not resume the goal or launch work:

```powershell
.venv/Scripts/python.exe tools/work_queue.py --check
.venv/Scripts/python.exe tools/work_queue.py --check --verify-local-proofs
.venv/Scripts/python.exe tools/work_queue.py --agents
.venv/Scripts/python.exe tools/work_queue.py --ready
.venv/Scripts/python.exe tools/work_queue.py --packet <reviewed-packet-id>
.venv/Scripts/python.exe tools/generate_api_implementation_paths.py --check
```

The checker catches cycles, malformed proofs, ownership conflicts and inventory
accounting. It does not prove that a packet's state is current, that an agent is
live, that a proof covers all arguments, or that a performance target was met.
On resume, root reconciles the old packet states with the paused checkpoint map
and the per-API roadmap before choosing any ready packet.

Follow [AGENTS.md](../AGENTS.md) after each validated merge. Remove an obsolete
worktree only with `git worktree remove`, after confirming its commits are
ancestors of main and its tracked/untracked source work is clean. Retain active
environments, running-job files, exact baseline/qualified products, raw results,
rejected trials, and unfinished source. Resolve every cleanup path inside the
workspace before deletion; never delete the whole `build/` directory. Existing
cleanup rejection records stay explicit. No cleanup is authorized by this
planning revision.

## Appendix: retained historical evidence

The prior narrative is preserved in Git at
[`12fad348` / `coverage/EXECUTION_PLAN.md`](https://github.com/bipxx/scikit-imagexx/blob/12fad3488178f52d52d8b4fd161e59790fe63d17/coverage/EXECUTION_PLAN.md).
Its old live assignments, handoff instructions, and counts are historical.
The following records remain authoritative only for their own exact sources:

| Checkpoint | Retained evidence and limitation |
|---|---|
| Resize/rescale and half identity | Sources `aed3349`/`21ac60b`; six actual 9,968-check profiles and the 373-case corpus. [Resize initial report](../benchmarks/resize-public-initial.md) measures source `66e5f78` and retains 49/52/61 slower cases. Later optimization sources require their own results. |
| First public metrics, footprints, and labels | [public-wave1](evidence/public-wave1.json) seals the public/SDK/corpus evidence. Reports for [metrics](../benchmarks/metrics-public-initial.md), [footprints](../benchmarks/footprints-public-initial.md), and [components](../benchmarks/components-public-initial.md) preserve measured regressions; a measured baseline is not optimization completion. |
| Transform classes and component consumers | [integrated-wave2](evidence/integrated-wave2.json), [integrated-wave3](evidence/integrated-wave3.json), and [component consumers](evidence/component-consumers-public.json) retain their exact scopes. Four classes remain partial with explicit numerical providers. |
| Rank expansion and corrected 186-path source | [public-wave7](evidence/public-wave7.json), [failed wave9](evidence/public-wave9-failed-trial.json), and [public-wave10](evidence/public-wave10.json) retain accepted and rejected trials. The corrected `05e79eb` does not erase endian/metadata failures of its predecessors. |
| Rank Otsu/majority and counts optimization | [public-wave11](evidence/public-wave11.json) covers the exact before/after products and the 190-path scope. [Counts comparison](../benchmarks/rank-statistics-counts-comparison.md) retains all 376 IDs, seven rounds, raw samples, and every slower case from run `40084877921`. |
| TV-Chambolle and Bregman additions | [public-wave12](evidence/public-wave12.json) accepts `f87120c` at 191 paths; [public-wave13](evidence/public-wave13.json) accepts `3cdd363` at 192. The initial TV measurement remains tied to `f87120c`; Bregman and later changes have independent performance gates. |
| TV strided cursor | [Cursor public evidence](evidence/tv-chambolle-cursor-public.json) qualifies `5e068a5`. [Controlled comparison](../benchmarks/tv-chambolle-cursor-comparison.md), run `40119412201`, retains 210 rows, 4,410 samples, and 129 union regressions. Actual before/after products stay `3cdd363`/`5e068a5`; reported gains cannot be relabelled as `af7b2af` results. |
| Matrix and max-tree private work | [General matrix kernel](evidence/matrix-transform-general-kernel.json), [profile tooling](evidence/matrix-transform-profile-tooling.json), [max-tree bridge](evidence/max-tree-private-bridge.json), [closing dtype correction](evidence/max-tree-closing-dtype-private.json), and [SDK oracle correction](evidence/max-tree-sdk-oracle-private.json) retain their scoped private proofs. The failed combined run `40117331038` remains failed. |
| Max-tree measurement preparation | [Benchmark reference](evidence/max-tree-benchmark-reference.json) retains 351 source observations and 252 finite candidates per runtime; it is not a native measurement result. |
| Archive and worktree retention | [Released metadata scratch cleanup](evidence/released-metadata-scratch-cleanup-20261024.json) and [TV admission worktree retirement](evidence/tv-admission-public-worktree-retirement.json) preserve archived files, redirects, and protected/rejected paths. They are not instructions to delete other worktrees. |

The merged publisher worktree has been [retired with all proof bytes archived](evidence/shared-publication-worktree-retirement.json); active environments, donor products and benchmark inputs remain protected.


## Execution checkpoint: 2026-10-25, combined H203 and TV

The historical snapshot for this checkpoint is
[checkpoint v2](evidence/parallel-implementation-checkpoint-20261025-v2.json).
Earlier handoff tables and checkpoints remain historical. Accepted image coverage
is still 199 native-partial paths; no full-library or new speedup claim follows.

Three workers are assigned independently: A reviews the frozen original-only
colocalization reference; B implements its three owned native APIs in new files;
C prepares source-first contracts for the remaining footprint family. The root
owns shared integration, admissions, hosted qualification and performance review.

TV's helper-only optimization has passed private parity and is merged to main.
Combined source `7417676` includes it, H203 and the independently reviewed
reconstruction internal-tie assertion repair. Hosted qualification run
[40200919182](https://github.com/bipxx/scikit-imagexx/actions/runs/40200919182)
was dispatched once for four fresh profiles; it has no product acceptance yet.
The failed201 predecessor remains failed with all raw observations retained.

Independent max-tree comparison
[40200431797](https://github.com/bipxx/scikit-imagexx/actions/runs/40200431797)
reuses exact accepted199 products, authenticates ordinary defaults on preparation
hosts and gates new artifact-only one-worker measurement hosts. It does not
depend on H203, rebuild a product, or measure current unqualified source.
Keep every timing/regression and separate optional-worker scaling from the
required one-worker original-library comparison. Source review, private parity,
qualified product and measured performance remain separate acceptance boundaries.
