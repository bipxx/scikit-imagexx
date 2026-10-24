# Controlled rank statistics counts comparison

The measured `windowed_histogram` cohort improved by **1.239317× on Windows, 1.159187× on modern Linux, and 1.181435× on minimum Linux** versus the prior C++ product. Overall ratios across the four APIs were **1.066436×, 1.043381×, and 1.044028×**. Results vary by workload; every slower case is retained.

[Actions run 40084877921](https://github.com/bipxx/scikit-imagexx/actions/runs/40084877921) completed all seven jobs successfully. This is the frozen [before 66860ae](https://github.com/bipxx/scikit-imagexx/commit/66860aef0d2483ca1d728589d9b3b0952433db91) → [after 94d2d3f](https://github.com/bipxx/scikit-imagexx/commit/94d2d3f48a4cf58ae9a34be5133eed6816f322c7) comparison, with **190 image API paths, three metadata paths, 14 SDK families and 59 SDK members**. The upstream comparator is **scikit-image 0.26.0**. The separate TV191 product is excluded.

## Results

Ratios are geometric means of per-case median-time ratios; **higher is better**. `Before/upstream` and `After/upstream` divide the upstream time by the C++ time. `After/before` divides the prior C++ time by the new C++ time. Slow counts use the original strict median comparisons, with no noise threshold. Displayed aggregate ratios are rounded; the CSV preserves full precision.

| Profile | Cases | Before/upstream | After/upstream | After/before | After slower than before | After slower than upstream |
|---|---:|---:|---:|---:|---:|---:|
| windows | 376 | 2.532433× | 2.700679× | 1.066436× | 102 | 43 |
| modern | 376 | 2.732102× | 2.850625× | 1.043381× | 106 | 24 |
| minimum | 376 | 2.783532× | 2.906085× | 1.044028× | 145 | 30 |

| API | Profile | Cases | Before/upstream | After/upstream | After/before | Slower than before | Slower than upstream |
|---|---|---:|---:|---:|---:|---:|---:|
| `geometric_mean` | windows | 94 | 3.987126× | 4.013645× | 1.006651× | 44 | 3 |
| `geometric_mean` | modern | 94 | 3.689046× | 3.662368× | 0.992768× | 43 | 3 |
| `geometric_mean` | minimum | 94 | 3.866573× | 3.891438× | 1.006431× | 46 | 4 |
| `noise_filter` | windows | 94 | 1.504518× | 1.531925× | 1.018217× | 27 | 24 |
| `noise_filter` | modern | 94 | 1.804762× | 1.827124× | 1.012390× | 35 | 11 |
| `noise_filter` | minimum | 94 | 1.728248× | 1.744976× | 1.009679× | 45 | 13 |
| `subtract_mean` | windows | 94 | 4.639908× | 4.724409× | 1.018212× | 30 | 4 |
| `subtract_mean` | modern | 94 | 4.127129× | 4.198277× | 1.017239× | 27 | 5 |
| `subtract_mean` | minimum | 94 | 4.916243× | 4.865246× | 0.989627× | 52 | 5 |
| `windowed_histogram` | windows | 94 | 1.477696× | 1.831333× | 1.239317× | 1 | 12 |
| `windowed_histogram` | modern | 94 | 2.027709× | 2.350495× | 1.159187× | 1 | 5 |
| `windowed_histogram` | minimum | 94 | 1.827339× | 2.158882× | 1.181435× | 2 | 8 |

The change skips sums and auxiliary histogram tables that `windowed_histogram` does not use. Its 94-case cohort is the consistent improvement here; it still has **1/1/2 cases slower than prior C++** and **12/5/8 slower than upstream** on Windows/modern/minimum. The other three APIs retain their source arithmetic. Their observed timing changes must not be attributed specifically to this histogram work: modern `geometric_mean` is 0.992768× and minimum `subtract_mean` is 0.989627× versus prior C++.

The histogram bin cohorts show where the measured effect occurs; the two four-case cohorts are small samples.

| Histogram bins | Cases/profile | Windows after/before | Modern after/before | Minimum after/before |
|---:|---:|---:|---:|---:|
| 4 | 4 | 1.181951× | 1.044931× | 1.134620× |
| 256 | 43 | 1.226202× | 1.145974× | 1.131566× |
| 1024 | 43 | 1.254484× | 1.183817× | 1.226107× |
| 65536 | 4 | 1.278326× | 1.160341× | 1.312325× |

## Method and scope

Each of the three preparation profiles checked all **1,379 canonical cases plus four generic identity aliases**, in normal and SIMD-disabled modes at **one and four workers**. All three preparations and the receipt barrier finished before the fresh measurement jobs started. Each fresh measurement host reauthenticated the frozen products and revalidated the full corpus before any warm-up, calibration or timer.

Measurement used **one global worker, initially one by default**, normal SIMD mode, and three separate ordinary-name processes for upstream, before and after. Backend operations ran serially on the same host/CPU within each profile. Each of the **376 fixed timed IDs** has **seven seeded, shuffled, paired rounds**; each backend has separately calibrated batches of 1–1,000 calls. Medians are milliseconds per call. IPC, fixture creation, integrity checks and the shared 20 ms cooldown are outside the timer. There were no builds or correctness tests concurrent with timed operations on these fresh hosts, and no local operation timings.

The remaining **1,003 canonical cases**, including errors/metadata, and all four aliases remain untimed. The measured scalar/vector API counts are **282 scalar rows and 94 vector rows per profile**. Workloads include uint8/uint16 images, 2D/3D scalar cases, C/F/reversed/stepped/broadcast/tagged/unaligned layouts, masks, output types and histogram depths. The [CSV](rank-statistics-counts-comparison.csv) contains **all 1,128 timed rows**, exact workload specifications, full-precision median ratios/timings and both regression flags. Its `cohort` is a transparent grouping of API kind, image shape, bin count and output depth; it introduces no filtering. Full raw samples remain unchanged in the [persistent raw archive](rank-statistics-counts-comparison-raw.zip), the Actions artifacts, and the retained audit closure. The ZIP contains the three original report byte streams, the exact experiment and frozen668 contract, the original campaign verification, and one set of all three preparation receipt/report pairs. Its `manifest.json` lists each payload member’s SHA256, size and provenance; every member was checked against its original source bytes and CRC. Product wheels and environment packages are excluded.

The production source delta is exactly the approved `src/rank.cpp` counts patch; two accompanying test files are the only other source differences. Compiler/toolchain, flags, used headers, other compiled sources and external link-input identities match before/after within each profile. Generated object/link output hashes are retained per product, rather than assumed byte-identical under LTO.

| Profile | Python | NumPy / SciPy | CPU reported by all three workers |
|---|---|---|---|
| windows | 3.13.16 | 2.5.4 / 1.18.1 | Intel64 Family 6 Model 207 Stepping 2, GenuineIntel |
| modern | 3.13.16 | 2.5.4 / 1.18.1 | AMD EPYC 7763 64-Core Processor |
| minimum | 3.11.17 | 1.24.4 / 1.15.3 | AMD EPYC 9V74 80-Core Processor |

These ratios describe this fixed corpus and these hosts. They establish neither a library-wide speedup nor performance on other CPUs. Profiles use different CPUs and are not pooled. Small changes in the unchanged API cohorts can reflect sampling and generated-code effects; no statistical-significance claim is made.

## Evidence and review status

The [machine-readable evidence](../coverage/evidence/rank-statistics-counts-measured.json) binds exact source/product/wheel/core, full SDK/runtime and common-build identities, the external experiment, all profile reports, all sample/order checks and the raw Actions artifact digests. The benchmark result is **verified for this frozen source pair**. The integrator reran all three source, CPU, receipt-barrier and paired-sample gates without new timing calls; verification outputs and adverse-case records match byte for byte.

| Profile | Raw result artifact ID | Raw artifact SHA256 |
|---|---:|---|
| windows | [12557341824](https://github.com/bipxx/scikit-imagexx/actions/runs/40084877921/artifacts/12557341824) | `a9ac4a6cdef3fa30f4f646f079b68ae02071f396fe7721a471f646dbb5ef60a2` |
| modern | [12557863647](https://github.com/bipxx/scikit-imagexx/actions/runs/40084877921/artifacts/12557863647) | `19ef572f9d125b1fc6a2fa8f2d1a036b17e63cf3e301b4ad2a2a5e48d11124bd` |
| minimum | [12557966739](https://github.com/bipxx/scikit-imagexx/actions/runs/40084877921/artifacts/12557966739) | `3a5e553acf77938cca7cc12172137bf4a265f40ed1e4584d5cbb9ec1bea7c03b` |

Local provenance: original handoff SHA256 `5155bda7fc48b6aec5f1a061bad2dcd44c3866f67ecce4475242a4841b0e122a`; campaign verification `00b27bf8c0498e9ff6e12c622001633f7c23bc19359aa7dc901818a8d003dc63`; accounting clarification `b01e1b6a4f44dd730c2c6eb4fa81e377a1680a02b288105e25a8bd219f4b2b58`. The handoff authenticates **210 records totaling 185,634,576 bytes**. Its separate `owned_bytes` field also counted two excluded CPython bytecode caches (23,263 bytes); the clarification records that exact difference without rewriting the handoff or raw science.

Raw ZIP SHA256 `f2feab3b8aa4324eef372f83ae9eaa82ec917f8001598aab70cdf109dedc09e3` (4540259 bytes; 13 members). ZIP member-manifest SHA256 `9b1252e85fa369d89360b6e876c618d23f252237798c6f2cfa54076d79a6bb56`. This preserves the numeric samples and preparation evidence beyond Actions artifact expiry.
