# Profile coordinates: controlled optimization kept

**KEEP.** The fixed 18-case same-host experiment improves prior C++ public-call performance by **24.639736453× direct** and **24.415711211× reference-normalized**. The same-host original/current aggregate rises from **0.979624253× baseline to 23.918222863× candidate**. Both candidate passes beat scikit-image in **18/18 cases**. All six width-33 targets repeatedly beat both baseline passes under both metrics; there are **no direct or normalized losses**. Correctness and deadline gates passed.

The [combined raw JSON](profile-coordinates-same-host.json) embeds all four complete original reports and the comparison without rewriting number text: **1,008 samples**, seven paired rounds, 18 cases and one global worker. All reference-library pools also use one thread.

| Side | Dtype | Width | Baseline C++ ms | Candidate C++ ms | Original/baseline | Original/candidate | Direct gain | Normalized gain | Added ms | Repeat direct / normalized |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | uint8 | 1 | 8.222666907 | 0.104782808 | 0.997065651× | 77.432091952× | 78.473435089× | 77.659973416× | -8.117884099 | yes / yes |
| 256 | uint8 | 9 | 8.221571955 | 0.294442802 | 0.990660255× | 27.810356258× | 27.922475615× | 28.072546688× | -7.927129153 | yes / yes |
| 256 | uint8 | 33 | 8.780304002 | 0.819761363 | 0.966282361× | 10.344166607× | 10.710804871× | 10.705117908× | -7.960542639 | yes / yes |
| 256 | float32 | 1 | 8.232685389 | 0.155319680 | 1.001242089× | 52.326048338× | 53.004779647× | 52.261135366× | -8.077365710 | yes / yes |
| 256 | float32 | 9 | 8.177869498 | 0.335096618 | 0.997032926× | 24.248775137× | 24.404512198× | 24.320937155× | -7.842772880 | yes / yes |
| 256 | float32 | 33 | 8.740015115 | 0.877597964 | 0.967438594× | 9.594524184× | 9.959019367× | 9.917450307× | -7.862417152 | yes / yes |
| 256 | float64 | 1 | 8.289703722 | 0.144714032 | 0.995647495× | 56.102884269× | 57.283344214× | 56.348139835× | -8.144989690 | yes / yes |
| 256 | float64 | 9 | 8.272901286 | 0.316434493 | 0.994081749× | 25.731832189× | 26.144119806× | 25.885026271× | -7.956466794 | yes / yes |
| 256 | float64 | 33 | 8.746102826 | 0.794642218 | 0.974257165× | 10.685370839× | 11.006340503× | 10.967710811× | -7.951460608 | yes / yes |
| 1,024 | uint8 | 1 | 32.220705739 | 0.222093209 | 0.997923374× | 143.039214250× | 145.077402116× | 143.336871268× | -31.998612530 | yes / yes |
| 1,024 | uint8 | 9 | 32.318377622 | 0.978548232 | 0.990755816× | 32.123902423× | 33.026862219× | 32.423632443× | -31.339829391 | yes / yes |
| 1,024 | uint8 | 33 | 34.313599630 | 3.039104311 | 0.964863052× | 10.823317684× | 11.290694927× | 11.217465176× | -31.274495318 | yes / yes |
| 1,024 | float32 | 1 | 32.954398230 | 1.037754816 | 0.982045652× | 30.704712069× | 31.755476082× | 31.266074040× | -31.916643414 | yes / yes |
| 1,024 | float32 | 9 | 32.638791632 | 1.760362442 | 0.975771642× | 17.923970250× | 18.540949785× | 18.369021483× | -30.878429191 | yes / yes |
| 1,024 | float32 | 33 | 34.761634033 | 3.779608839 | 0.955218382× | 8.685492910× | 9.197151217× | 9.092677728× | -30.982025194 | yes / yes |
| 1,024 | float64 | 1 | 33.347723338 | 0.906171341 | 0.974028259× | 35.309199515× | 36.800681968× | 36.250693131× | -32.441551997 | yes / yes |
| 1,024 | float64 | 9 | 33.107725161 | 1.585226463 | 0.963410281× | 20.068354478× | 20.885170630× | 20.830538011× | -31.522498699 | yes / yes |
| 1,024 | float64 | 33 | 35.272089992 | 3.531941459 | 0.947874355× | 9.314154600× | 9.986600967× | 9.826359948× | -31.740148533 | yes / yes |

Each C++ latency is the geometric mean of its two seven-round medians. Direct gain is balanced baseline/candidate C++ latency; normalized gain divides candidate original/current speedup by baseline original/current speedup. Original/C++ columns use the corresponding two passes. Added milliseconds are candidate minus baseline balanced latency; negative values mean lower latency. Repeat wins require both candidate passes to beat both baseline passes. Every target, control and dtype remains included.

| Group | Cases | Direct gain | Normalized gain | Original/candidate |
|---|---:|---:|---:|---:|
| Width 33 targets | 6 | 10.333455043× | 10.260797404× | 9.877214327× |
| Width 1/9 controls | 12 | 38.047954210× | 37.662878814× | 37.219978975× |
| uint8 | 6 | 32.906453215× | 32.673636307× | 32.167043417× |
| float32 | 6 | 20.289325634× | 20.094092261× | 19.685395742× |
| float64 | 6 | 22.405750089× | 22.168804621× | 21.608838784× |

The candidate adds **117 production lines** across the Python helper, binding, native declarations and scalar/SIMD coordinate routines. After the unchanged NumPy endpoint/scalar/trigonometric prelude and two center-line `linspace` calls, C++ constructs the assessed perpendicular grids. It preserves NumPy's ordered start/stop/delta/step arithmetic, row-specific zero-step branch, single-value behavior and endpoint overwrite without FMA. Exceptional numerical refusals restore floating-point state and run the unchanged perpendicular NumPy fallback once.

**Provider scope:** the frozen benchmark text still describes baseline NumPy coordinate construction. For this candidate, perpendicular grids on the measured routes are owned C++; the scalar prelude and two center-line calls remain NumPy providers. Existing sampling, whole-image finite admission, flip/channel assembly and width mean/callback behavior are unchanged. Full public preparation, owned coordinates and sampling, allocations and output destruction remain timed. The earlier rejected `99fdb2a` sampler-address/packed-scan candidate is **not included**; its [DISCARD report](profile-sampling-same-host.md) remains separate.

Both genuine CI products retain their original byte-exact artifacts. Original installations passed their source/package/core/SDK verifier. Versioned measurement copies then used the retained baseline pytest **9.2.0** wheel (`07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588`), authenticating a copied manifest and full hash lock while force-reinstalling only that test-runner wheel. Both originals already contained that exact pytest wheel; all **13 dependency wheels** match after normalization. Scientific packages, product wheels, source maps and original artifacts remain unchanged. Both normalized installations and all **eight before/after identities** passed the official verifier.

Correctness is separate from timing. Candidate preparation passed **4,594 focused checks**, **3,696 color regressions**, **1,612 matching checks**, and **1,413 smoke checks for its 238 registered functions**, plus separate four Blur/four Polygon/four Euler/nine integer-width guards. All 19 preparation timings were skipped. Retained private validation passed **916 component checks** (372 old public, 136 shared and 408 new coordinate checks), six native modes, old Profile controls and the old **712 zoom controls per normal/scalar mode**. All 51 coordinate cases passed on each genuine dispatch; 30 ordinary/benchmark coordinate cases require the actual owned binding and exactly two center-line calls. Every timed case passes exact sampled-pixel/public metadata and byte checks with zero error.

[Controlled run 40653900195](https://github.com/bipxx/scikit-imagexx/actions/runs/40653900195) used **AMD EPYC 7763**, Linux x86_64, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Comparison completed **2026-10-28 08:03:49.496621 UTC**, before the frozen **08:53:40.730139 UTC** deadline. Baseline [`e190a765`](https://github.com/bipxx/scikit-imagexx/commit/e190a76500ebd33f3a2d3cb2aee26668a486b72f) came from [Release 40645607032](https://github.com/bipxx/scikit-imagexx/actions/runs/40645607032); candidate [`0e8b9de6`](https://github.com/bipxx/scikit-imagexx/commit/0e8b9de6ed849f12b23330c50b374cd4caa8de63) came from [prepare-only Release 40651258735](https://github.com/bipxx/scikit-imagexx/actions/runs/40651258735) and was integrated as [`7d75c9f`](https://github.com/bipxx/scikit-imagexx/commit/7d75c9f). These are same-host comparisons, with no multiplication of earlier cross-host scores. Full supported-profile qualification remains **203 image APIs**.

Provenance: [exact artifact 12800529554](https://github.com/bipxx/scikit-imagexx/actions/runs/40653900195/artifacts/12800529554), 799,086 bytes, 49 members, SHA256 `b114eaa29904d823e900b722ae617b05447eb9e40f3273032f23a69deed554a0`; every member size/CRC/SHA was verified. Contract SHA256 `85d2536b85ecfd6f4d75c9eee3694b081fb909073cb8106061877da9b50c4a70`. Independent review SHA256 `5128a5fe39385d8cdc5998f4344f97854b8fe66af98f46be6d2baf072f6ddb51`. Combined raw JSON SHA256 `cdc31f32e6be134040d1013c6623e73e0dd3328c3622dfaa44989d36899e1fdd`.
