# Packed uint8 block sums: controlled same-host result

**KEEP.** Balanced fixed6 direct gain is **3.100525978450x** and reference-normalized gain **3.101247587476x**. Both uint8 targets beat both baseline passes in both candidate passes under both metrics. Their geometric gains are 31.732233952936x direct / 31.585396145497x normalized. The four float controls regress in aggregate: 0.969176194119x / 0.971765541485x. Every control penalty is retained below.

The same-ABBA original/current geometric scores are **4.074701122670x baseline** and **12.636657026366x candidate**. All six cases beat original 0.26.0 in both candidate passes. These are same-host comparisons; the earlier cross-host 3.8445 score is not multiplied into this result.

Baseline `21e0ba92c527aede8fb7995c9c10ee7ede77384b` ([Release40417700622](https://github.com/bipxx/scikit-imagexx/actions/runs/40417700622)) is compared with `0bb6753ae992f6fc483dcba7c1322f0d46a22833` ([preparation40422600556](https://github.com/bipxx/scikit-imagexx/actions/runs/40422600556)), kept and merged as `59193647cff1a1565906408f219110b605bd1d78`. [Controlled run40424219957](https://github.com/bipxx/scikit-imagexx/actions/runs/40424219957) uses controller `29f4084e444f357fde633c67293c0a12936aae94`.

The [byte-exact compact archive](block-reduce-u8-same-host.zip) contains all four reports and **336 raw samples**, eight before/after identity logs, CPU/platform evidence, offline-install logs and exact product-selection receipts. No producer ZIPs or wheels are duplicated in Git.

| Size | Input dtype | Role | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added us | Repeat win D/N | Repeat loss D/N |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 256 | float32 | control | 0.116830403 | 0.121237837 | 0.963646385x | 0.963839770x | +4.407434 | 0/0 | 0/0 |
| 256 | float64 | control | 0.151282037 | 0.153775249 | 0.983786646x | 0.987123600x | +2.493213 | 0/0 | 0/0 |
| 256 | uint8 | target | 0.442425042 | 0.029138723 | 15.183405165x | 15.131179671x | -413.286319 | 1/1 | 0/0 |
| 1024 | float32 | control | 1.426198468 | 1.525315332 | 0.935018772x | 0.937328795x | +99.116864 | 0/0 | 1/1 |
| 1024 | float64 | control | 1.990677278 | 1.999994757 | 0.995341248x | 0.999948524x | +9.317479 | 0/0 | 0/0 |
| 1024 | uint8 | target | 6.630427718 | 0.099979148 | 66.318105901x | 65.932549304x | -6530.448570 | 1/1 | 0/0 |

Each image is size x size, ordinary C layout with default 2x2 blocks. Uint8 returns uint64; float32/float64 retain their dtype. Each balanced latency is the geometric mean of the two per-pass seven-sample medians. Direct gain is baseline/current latency; normalized gain is candidate original/current speedup divided by baseline original/current speedup. Added microseconds are candidate minus baseline: positive numbers are penalties. Repeat wins/losses require both candidate passes to beat/lose to both baseline passes.

**Four direct losses and four normalized losses**, all in the float controls. Only 1024 float32 is a repeat loss in both metrics: +99.116864 us. The other penalties are 256 float32 +4.407434 us, 256 float64 +2.493213 us, 1024 float64 +9.317479 us. These unchanged-path variations are retained without attributing their cause to the uint8 change.

| Size | Input dtype | Baseline original ms | Baseline original/current | Candidate original ms | Candidate original/current | Max abs error, all passes |
|---:|---|---:|---:|---:|---:|---:|
| 256 | float32 | 0.740309256 | 6.336614734x | 0.740457822 | 6.107481290x | 2.38418579e-07 |
| 256 | float64 | 0.745664764 | 4.928970945x | 0.748194021 | 4.865503545x | 4.4408921e-16 |
| 256 | uint8 | 0.796650193 | 1.800644440x | 0.793910000 | 27.245874548x | 0 |
| 1024 | float32 | 11.172918228 | 7.834055693x | 11.200521635 | 7.343085985x | 2.38418579e-07 |
| 1024 | float64 | 11.254586617 | 5.653646997x | 11.306682299 | 5.653355969x | 4.4408921e-16 |
| 1024 | uint8 | 12.183139868 | 1.837459118x | 12.112310192 | 121.148363918x | 0 |

A fresh Ubuntu host with **AMD EPYC 9V74 80-Core Processor**, CPython 3.11.17, NumPy 1.24.4 and SciPy 1.15.3 executes baseline → candidate → candidate → baseline in fresh processes. Both environments install offline before measurement. Seven warmed paired rounds use one global worker; inputs are created outside timing, and public preparation/allocation/summation remain inside. The frozen per-block checker runs outside timing. Exact source, runtime, thirteen dependency wheels, package/core/SDK and all eight pre/post identities match the authenticated prepared products.

**Deadline honored:** the contract was created 2026-10-26 23:43:19 UTC with a 30-minute budget ending 2026-10-27 00:13:19 UTC. The sole ABBA job completed 2026-10-27 **00:07:48 UTC**, before that deadline. No source, metric, cases, tolerances or timing policy were retuned.

Only three production files change: `src/block_reduce.cpp`, `src/block_reduce_internal.hpp`, `src/simd_block_reduce_avx2.cpp` (+21 lines). The new path requires uint8, rank 2, 2x2 factors and a contiguous output row. Two bounded 16-byte row loads form eight exact four-pixel sums with packed unsigned-byte pair sums, addition in 16-bit lanes (maximum 1020), then widening to uint64 output. Existing row/worker clipping, optional global worker owner, scalar tails, bool, F output, general factors, floating arithmetic and FP policy remain intact. Actual codegen contains packed pair-sum/widen instructions and no FMA.

All eight kept/candidate native runs pass 7,173,011 controls each across normal/scalar and sanitizer modes. Full output and FP streams are exact **within each corresponding mode**; this is not a claim of floating cross-mode equality. The unchanged native controls plus independent byte-extreme, packet-boundary, row-tail, worker-split, F-layout and general-factor checks remain retained. Two genuine component relinks pass **668 checks** (145 frozen public +22 shared in each of four profile/modes); all 576 original/candidate journal rows match retained baseline bytes.

Genuine candidate Release passes **3,108 focused public checks**, **3,696 full-color regressions**, **1,381 installed smoke checks** at **230 registered functions**, and the existing numerical regressions. Preparation-only skips all 12 standalone timings; this controlled run is the performance evidence. Full supported-profile qualification remains **203 image APIs**, separate from registration.

NumPy padding, block-view metadata, storage/layout preparation and allocation remain providers. Half requires original padded native/aligned C storage and block volume within np.getbufsize; other half layouts/buffering remain open. Custom reducers/kwargs, complex padding and exceptional/type-frontier reductions retain explicit uncredited primitive providers. This packet does not close those partitions. One minimum-runtime host and six default workloads do not establish universal or full-profile performance.

Artifact `12698968474`: 26,264 bytes/36 members; SHA-256 `662197b6113d08fe173eb39999a381c7478a053781c7324c10c30852787ffac3`. Every member CRC, size and SHA is verified. Contract SHA-256: `a87357864060f451365ca7ed454a742ae46d5c22691b154be43a74a2aa91670e`. Final review SHA-256: `9a9e18e9c22208619b56118bdf57f3c3d70ae347b04be582fe5c2f2370997b3c`. Sealed review/handoff bytes remain unchanged; a separate root decision record authorizes KEEP/merge.

Baseline wheel SHA-256: `a805a83e853709f474a51e80e8806668e67d42c2431f29f44e7e8a572caa4762`; sole core `98c7c4854de736cdce3a225775a6bcd1736f906f98029f14e169b57297c7e493`. Candidate wheel: `a5d8e86848059af03d8f7a45aaab0745fdbe5c428023a44f24bde3080438bf06`; sole core `ff88c75623c51495ae5894f8887861553f6491efd523abcd55820d1e6112d14a`.

Product fingerprints baseline `17ad4b663682cee5e2dd914004262d2fd41ef9667ee71444c3b0c1e7177a1077` and candidate `a57fac3d03d0c1d1652bbce4c744104bd524aae292d13d9d180190e2b8368f52`. Unchanged coverage harness canonical-LF SHA-256: `75c8b3aa5ef38b67800598d19cb166505e3265ca7757e9ebdce36507ac0d01ef`.
