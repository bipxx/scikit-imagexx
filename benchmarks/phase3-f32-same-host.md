# Float32 v5: controlled comparison and keep

**Kept and merged:** checked source/main `5b03eded95b86c22de7cea6c76243b1748969a64` reuses the exact wheel compiled from `c53198aff0a502fdfe236c6b46834bbc58547abd`. Against kept byte-v4, the fixed 21-case metric improves **6.79% normalized / 5.99% direct public-call speedup**. All ten targeted float32 cases improve by **9.36–24.76%**; both candidate medians beat both baseline medians in every target. Five cases beat the original library; **16 colocalization cases remain slower**.

[Controlled run 40277901243](https://github.com/bipxx/scikit-imagexx/actions/runs/40277901243) used one AMD EPYC 9V45 host, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. All C++ and upstream pools used one thread. Order: baseline, candidate, candidate, baseline (ABBA), with all 73 workloads and seven paired original/current rounds per pass. The C++ change specializes float32 loads while preserving the pairwise reduction tree and rounding points; this increment makes no new SIMD-reduction claim.

## Four reports and fixed metric

Each score is the equal-weight geometric mean of original/current median ratios over the fixed 21 colocalization cases. Compare the geometric mean of both candidate scores with that of both baseline scores; no case or repeat is removed.

| Pass | Product | Fixed 21-case score |
|---|---|---:|
| 1-baseline | byte-v4 | 0.20632518921484× |
| 2-candidate | float32-v5 | 0.21775189078463× |
| 3-candidate | float32-v5 | 0.21859038903571× |
| 4-baseline | byte-v4 | 0.20227378442908× |

Baseline pair **0.20428944379368×**, candidate pair **0.21817073708423×**: **1.067949146× normalized / 1.059884336× direct**. Original-reference latency in candidate passes is **1.007609142×** baseline latency (+0.761%). Reference drift is **0.967763646×** for baseline pass 4/1 and **1.016589076×** for candidate pass 3/2.

## All 21 comparisons, including every loss

Normalized gain compares original/current ratios; direct gain compares baseline/candidate current-call latencies. Values below 1 are losses: **five normalized and seven direct losses**. Candidate original/current below 1 remains a regression against scikit-image. IC = `intersection_coeff`, MCC = `manders_coloc_coeff`, MOC = `manders_overlap_coeff`; suffixes are the fixed `future/<workload>` IDs.

| Case | Normalized gain | Direct gain | Candidate original/current |
|---|---:|---:|---:|
| IC/tiny | 0.990430× **loss** | 0.963198× **loss** | 1.587671× |
| IC/cache | 0.983546× **loss** | 0.973156× **loss** | 1.559467× |
| IC/memory | 1.001884× | 1.005548× | 1.651349× |
| IC/volume | 1.016796× | 1.012737× | 1.792882× |
| IC/stepped | 0.948691× **loss** | 0.937296× **loss** | 0.280298× |
| IC/masked | 0.824829× **loss** | 0.823302× **loss** | 0.619121× |
| MCC/tiny | 1.104905× | 1.093572× | 0.395904× |
| MCC/cache | 1.206040× | 1.190808× | 0.057993× |
| MCC/memory | 1.000002× | 0.946290× **loss** | 0.075244× |
| MCC/volume | 1.005130× | 0.997068× **loss** | 0.075619× |
| MCC/stepped | 1.176951× | 1.188679× | 0.134085× |
| MCC/masked | 1.138814× | 1.128173× | 0.310041× |
| MOC/tiny | 1.124396× | 1.106735× | 0.281544× |
| MOC/cache | 1.222686× | 1.211603× | 0.039029× |
| MOC/memory | 1.000679× | 0.948142× **loss** | 0.064276× |
| MOC/volume | 1.017638× | 1.009144× | 0.051709× |
| MOC/stepped | 1.220631× | 1.247566× | 0.079411× |
| MOC/masked | 1.180719× | 1.180714× | 0.182720× |
| IC/cache-F | 0.987473× **loss** | 1.002231× | 1.573300× |
| MCC/cache-F | 1.194216× | 1.201778× | 0.058489× |
| MOC/cache-F | 1.199004× | 1.227679× | 0.038658× |

IC tiny, cache, memory, volume and cache-F beat the original in both candidate passes. IC stepped/masked and every MCC/MOC case remain slower. All 52 footprint cases beat the original in each pass. Every original-library penalty and all 73 baseline/candidate comparisons remain in the raw reports.

## Ten float32 targets: both candidate medians beat both baseline medians

Current-call median milliseconds; no timing sample is excluded.

| Case | Baseline 1 | Baseline 4 | Candidate 2 | Candidate 3 | Direct gain |
|---|---:|---:|---:|---:|---:|
| MCC/tiny | 0.023533 | 0.023017 | 0.021123 | 0.021443 | 1.093572× |
| MCC/cache | 0.761618 | 0.767899 | 0.643372 | 0.641056 | 1.190808× |
| MCC/stepped | 0.787004 | 0.758366 | 0.655199 | 0.644694 | 1.188679× |
| MCC/masked | 0.772467 | 0.751348 | 0.675897 | 0.674666 | 1.128173× |
| MOC/tiny | 0.048049 | 0.047636 | 0.042474 | 0.043995 | 1.106735× |
| MOC/cache | 1.765869 | 1.739908 | 1.431979 | 1.461597 | 1.211603× |
| MOC/stepped | 1.815995 | 1.787521 | 1.448091 | 1.440266 | 1.247566× |
| MOC/masked | 1.437324 | 1.423791 | 1.219214 | 1.204014 | 1.180714× |
| MCC/cache-F | 0.774597 | 0.759964 | 0.633048 | 0.643849 | 1.201778× |
| MOC/cache-F | 1.788371 | 1.745152 | 1.440506 | 1.437493 | 1.227679× |

Target-ten direct geometric-mean gain is **1.176698×**. The other 11 colocalization cases have **0.963784×**; the unchanged 52 footprint controls have **0.999914×**. The non-target losses are retained, not attributed away.

## Failed preparation, diagnostic and accepted reference policy

[Preparation 40273141051](https://github.com/bipxx/scikit-imagexx/actions/runs/40273141051) remains **741 passed / 44 failed**. Its 44 MOC final-result differences are +1 ULP, including integer inputs unaffected by this float32 change. No timing ran from that failed preparation. The raw failure is preserved.

[Diagnostic 40275501905](https://github.com/bipxx/scikit-imagexx/actions/runs/40275501905) and its [independent raw receipt](../coverage/evidence/colocalization-live-reference-checkpoint.json) establish 536/536 exact native-versus-fresh-original cases in all four old/new normal/scalar processes. Fresh originals and both products reproduce the same 44 frozen-result differences. The change is unnecessary for that discrepancy on this host; its precise runtime cause remains unknown. The diagnostic itself retains `acceptance=false`.

The separately reviewed test policy now takes only a successful MOC floating-scalar result word from a fresh original. Return type/dtype and every other frozen field remain exact; the native result must match the full fresh observation exactly. Frozen fixtures, matrix and observers are unchanged. This is an explicit reference-policy change, not a tolerance or failure erasure.

[Successful preparation 40277634220](https://github.com/bipxx/scikit-imagexx/actions/runs/40277634220) reused the same compiled wheel and passed **785 public checks per normal/scalar mode, 1,318 smoke checks, 21 metadata tests plus 30 subtests, and 73 corpus checks per mode**. Independent raw review verifies all 536 frozen cases plus three worker-4 samples per mode against their fresh originals; only the same 44 MOC words differ from frozen.

The controlled measurement host ran no builds or pytest. Its four fresh 73-case comparisons are exactly identical, with before/after offline product identities and unchanged runtime dependencies. Baseline preparation versus this host differs by one ULP in masked MOC for both original/current; all four within-job outputs match exactly.

## Exact raw evidence and scope

The [byte-exact compact ZIP](phase3-f32-same-host.zip) retains all four 73-case reports, **4,088 raw samples**, logs, exits, before/after identities and product archive selections. Artifact `12630599449`: **135,162 bytes**, 30 CRC-checked members, SHA-256:

`2946c15fde5f13b38816900e3255f0ddd7d92e4d555b087b6c3c482fc93a4a8d`

Baseline preparation: run `40270607172`, source/compiled product `0135305`, artifact `12627936362`. Candidate preparation: run `40277634220`, checked source `5b03eded`, compiled producer `c53198a` from run `40272309555`, artifact `12630374206`. Exact candidate wheel SHA is `06b043b1559d9923ee02ad26dd3b4bd090d50139177df826eec9649396041aec`; the wheel was reused, not rebuilt.

Local independent receipts: `build/phase3-f32-live-preparation-v5/preparation-review.json` (SHA `1ec458f4e53c747206ce6f5a0658f109e4144d2b1b16c9543dcec68df1be04b2`) and `build/phase3-f32-same-host-results-v5/review.json` (SHA `ce1cb20e555ba8a0d91ac4d7452899d2dc38b50d254c0004e8c0c69b2057a646`). The large prepared archives remain local and in Actions; they are not duplicated here.

This is one minimum-Linux 9V45 host with two repeats per product, not a universal-profile or statistical-significance claim. Do not multiply gains from different CPUs into a cumulative code-speedup claim. Timings include allocation, owned orchestration, explicit NumPy providers and output destruction; they do not isolate C++ kernels or establish full product/SDK qualification.

Next: explicit SSE float32 summation is under development in the root branch, while an independent octahedron experiment uses its own worktree. Neither has a measured result yet.
