# Compare images: controlled fused optimization kept

**KEEP.** The fixed 18-case same-host experiment improves prior C++ public-call performance by **2.244206996x direct** and **2.208701939x reference-normalized**. The same-host original/current aggregate rises from **0.749675982x baseline to 1.655810794x candidate**. Both candidate passes beat scikit-image in **18/18 cases**. All 12 diff/blend targets repeatedly beat both baseline passes under both metrics; all six checker controls beat the original in both candidate passes. Correctness and deadline gates passed.

The [combined raw JSON](compare-images-fused-same-host.json) embeds all four complete original reports and the comparison without rewriting number text: **1,008 samples**, seven paired rounds, 18 cases and one global worker. Reference-library pools also use one thread. Full public decorator, conversion, validation, preparation, owned math, allocation and output destruction remain timed.

| Side | Dtype | Method | Baseline C++ ms | Candidate C++ ms | Original/baseline | Original/candidate | Direct gain | Normalized gain | Added ms | Repeat direct / normalized |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | uint8 | diff | 0.186791053 | 0.107647585 | 0.668758507x | 1.158833116x | 1.735208946x | 1.732812524x | -0.079143468 | yes / yes |
| 256 | uint8 | blend | 0.174332923 | 0.104936378 | 0.669228425x | 1.099112389x | 1.661320190x | 1.642357597x | -0.069396545 | yes / yes |
| 256 | uint8 | checkerboard | 0.193558099 | 0.141387323 | 1.453771004x | 1.966563699x | 1.368991891x | 1.352732785x | -0.052170776 | yes / yes |
| 256 | float32 | diff | 0.055866848 | 0.021561710 | 0.452987114x | 1.183588665x | 2.591021202x | 2.612852832x | -0.034305138 | yes / yes |
| 256 | float32 | blend | 0.115004276 | 0.020877637 | 0.268659297x | 1.380470799x | 5.508490978x | 5.138369721x | -0.094126639 | yes / yes |
| 256 | float32 | checkerboard | 0.081821595 | 0.052706873 | 2.350848148x | 3.625244781x | 1.552389472x | 1.542100788x | -0.029114722 | yes / yes |
| 256 | float64 | diff | 0.093876676 | 0.033599507 | 0.476916272x | 1.337707249x | 2.793989709x | 2.804910060x | -0.060277169 | yes / yes |
| 256 | float64 | blend | 0.095723712 | 0.032330695 | 0.477832145x | 1.296778436x | 2.960768768x | 2.713878609x | -0.063393017 | yes / yes |
| 256 | float64 | checkerboard | 0.108867649 | 0.068031171 | 1.859369917x | 2.972804374x | 1.600261290x | 1.598823529x | -0.040836478 | yes / yes |
| 1,024 | uint8 | diff | 7.554444442 | 4.239574209 | 0.664967033x | 1.406951348x | 1.781887536x | 2.115821206x | -3.314870232 | yes / yes |
| 1,024 | uint8 | blend | 3.980497295 | 1.642480408 | 0.516098709x | 1.183391423x | 2.423467139x | 2.292955597x | -2.338016887 | yes / yes |
| 1,024 | uint8 | checkerboard | 4.469247957 | 4.648417698 | 1.270498058x | 1.529186941x | 0.961455757x | 1.203612183x | +0.179169740 | no / yes |
| 1,024 | float32 | diff | 0.780370208 | 0.321485437 | 0.459925726x | 1.034616982x | 2.427388979x | 2.249530573x | -0.458884770 | yes / yes |
| 1,024 | float32 | blend | 0.863490604 | 0.200004039 | 0.347992621x | 1.457340632x | 4.317365833x | 4.187849242x | -0.663486565 | yes / yes |
| 1,024 | float32 | checkerboard | 0.921865608 | 0.509323764 | 3.189790126x | 5.225368018x | 1.809979572x | 1.638154176x | -0.412541844 | yes / yes |
| 1,024 | float64 | diff | 2.200282384 | 0.669524462 | 0.473582942x | 1.504601373x | 3.286336062x | 3.177059894x | -1.530757922 | yes / yes |
| 1,024 | float64 | blend | 2.195090366 | 0.605681951 | 0.378031803x | 1.067688527x | 3.624163410x | 2.824335200x | -1.589408415 | yes / yes |
| 1,024 | float64 | checkerboard | 2.245916458 | 1.169472582 | 1.779665221x | 3.550352973x | 1.920452426x | 1.994955530x | -1.076443875 | yes / yes |

Each C++ latency is the geometric mean of its two seven-round medians. Direct gain is balanced baseline/candidate C++ latency; normalized gain divides candidate original/current speedup by baseline original/current speedup. Original/C++ columns use the corresponding two passes. Added milliseconds are candidate minus baseline balanced latency. Repeat wins require both candidate passes to beat both baseline passes.

**One direct C++ control regression remains:** 1024 x 1024 uint8 checkerboard adds **0.179169740 ms**, with **0.961455757x direct**, **1.203612183x normalized**, and **1.529186941x original/candidate**. Its candidate original ratios are 1.405304298x and 1.663990285x, so both passes still beat scikit-image. This is not a repeat C++ regression: only the first candidate pass is slower than both baseline passes. The frozen checker gate requires both original wins, which pass. There are no normalized losses, balanced original losses, or individual candidate-pass original losses.

| Group | Cases | Direct gain | Normalized gain | Original/candidate |
|---|---:|---:|---:|---:|
| targets_diff_blend | 12 | 2.745540060x | 2.648788888x | 1.250050833x |
| checker_math_controls | 6 | 1.499454065x | 1.535736285x | 2.905204982x |
| diff | 6 | 2.372993403x | 2.401948118x | 1.260889859x |
| blend | 6 | 3.176574453x | 2.920996718x | 1.239304983x |
| checkerboard | 6 | 1.499454065x | 1.535736285x | 2.905204982x |
| uint8 | 6 | 1.593708129x | 1.679959262x | 1.361778635x |
| float32 | 6 | 2.736881745x | 2.614659431x | 1.897450636x |
| float64 | 6 | 2.591337113x | 2.452999492x | 1.756932699x |

The measured bundle contains two changes: bypass two preparation copies when converted inputs already own aligned, native-endian C-contiguous buffers; and fuse finite masks into diff/blend scalar/SIMD math. Other inputs retain the original preparation expression. Typed arithmetic order, floating-point refusal and literal NumPy fallback are preserved. The scalar NumPy 1.24 zero-dimensional float32 blend partition still uses its literal provider. Binding validation, public signature, CMake, the old 54-case factory, and benchmark harness are unchanged. Checker native math and its prescan are unchanged, but its shared frontend also benefits from the copy bypass; these are not untouched public controls.

For the assessed diff/blend C buffers, source-level counted element traffic falls from **9N to 3N**. This is a structural count, not measured memory traffic or bandwidth. Timing establishes the gain of the two-change bundle; it does not attribute gains to individual stages.

Before each timed pass, the fixed harness checks exact converted inputs, complete public array metadata and output words, actual C returns, native acceptance and zero error. Exports retain hashes rather than duplicate full image byte payloads; all output hashes equal their reference hashes across all four passes. The 18 cases cover sides 256/1024, uint8/float32/float64 and diff/blend/checkerboard, using independent seeds 2026/2027 and default 8 x 8 tiles.

Both genuine CI products retain their original byte-exact artifacts. Original installations passed source/package/core/SDK identity checks. Versioned measurement copies use the retained baseline pytest **9.2.0** wheel (`07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588`) with authenticated manifests and locks; only the test-runner wheel is force-reinstalled. Both originals already contained that wheel. All **13 dependency wheels**, including the other 12 unchanged dependencies, match; scientific packages and product artifacts remain unchanged. Both normalized installations and all **eight before/after identities** passed the official verifier.

Correctness is separate from timing. Genuine candidate preparation passed **4,710 focused checks** in 21 normal/scalar XML pairs, **3,696 color regressions**, **1,612 matching checks**, and **1,417 smoke checks for 239 registered functions**, plus separate four Blur/four Polygon/four Euler/nine integer-width guards. All 20 preparation timings were skipped. Its 1,662 source blobs and 14-wheel runtime were authenticated. Private validation retained **572 public/shared checks**, **1,248 literal floating-point fallback controls**, and 432 exact saved original/candidate/admission rows. All baseline/candidate native six-mode checks passed, including old 5,297 normal/scalar/sanitizer and 4,603 portable controls per applicable mode, plus new 2,159,171 normal/scalar/sanitizer and 1,079,587 portable controls per applicable mode. Code review retained typed order and no FMA.

[Controlled run 40660965476](https://github.com/bipxx/scikit-imagexx/actions/runs/40660965476) used **AMD EPYC 7763 64-Core Processor**, Linux x86_64, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Comparison completed **2026-10-28 09:02:54.726480 UTC**, before the frozen **09:53:11.706388 UTC** deadline. Baseline [`31765844`](https://github.com/bipxx/scikit-imagexx/commit/317658444d7de04c751fe92d912c60853054c0e6) came from [Release 40649386523](https://github.com/bipxx/scikit-imagexx/actions/runs/40649386523); candidate [`6fa16120`](https://github.com/bipxx/scikit-imagexx/commit/6fa1612083ac65d7e0309862ad573112f7810a7b) came from [prepare-only Release 40658484627](https://github.com/bipxx/scikit-imagexx/actions/runs/40658484627) and was integrated as [`022780a3`](https://github.com/bipxx/scikit-imagexx/commit/022780a3ff71435ce988c9bb2bb2ce18b1bc7e43). These are same-host comparisons; earlier cross-host scores are not multiplied into this result.

Full supported-profile qualification remains **203 image APIs**; qualification of the expanded current product is pending. A separate four-profile run on source `fa264f3` measures an older Compare baseline and cannot qualify this merged product. This experiment does not claim all 236 current image APIs are fully qualified.

Provenance: [comparison artifact 12804270806](https://github.com/bipxx/scikit-imagexx/actions/runs/40660965476/artifacts/12804270806), 450,915 bytes, 49 members, SHA256 `0289488cb53a79aa8b5ea0ca90ba4b564ea37df8575c17b1e72836da1ce4c6db`; every member size/CRC/SHA was verified. Baseline original artifact `12798648924`: 84,730,999 bytes, 68 members, SHA256 `0179be5c3b7b3232b5c005bbfc4ded321a11f34e64207913b0a6382ea684265e`. Candidate original artifact `12802239761`: 84,723,807 bytes, 66 members, SHA256 `459f44aa066a810d25ef6036541b82de533a6d3c678c492248f1605e41207722`. All original and normalized identity/lock proofs are retained in the comparison artifact.

Contract SHA256 `4fae66de81eefb42a13a899c6a0a8e843f5aac8d5ab5f2e24818a1ba5679ecf8`. Independent review SHA256 `42c2c480bb50f8971791f19ec32be4127656e401dbda6286583183ad903700aa`. Private handoff SHA256 `e62326d9a8036d76e629bde18e3391cb3ab86ed1aa0b16153edcf5d271693c7f`. Combined raw JSON SHA256 `fddc3ac751a066098bd28df8cea14138165f5c8c99ba242987a4fb5ed8269358`.
