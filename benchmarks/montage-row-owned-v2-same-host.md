# Montage row-owned byte-fill v2: unresolved result

[Complete raw reports, journals, identities, independent/root reviews and duration audit](montage-row-owned-v2-same-host.json).

Result: **UNRESOLVED_MEASUREMENT**, not KEEP. Every fixed performance gate passes, but the Fortran normalized null/worst-pair separation rule fails. The unchanged rule prevents acceptance.

Run 40792259128, controller `5c272d8dba04421c0ba71e96d40a3c571cb67d52`. Candidate `b5bd1df7e27099e528fb4c7db67e2cb0bf35abae`; baseline `4f69eba77a262056459eaab1e0a39e53610e9305`. Artifact 12861459608, SHA256 `041d281c4e9dce0b8ca60056a5c71500b83d6d20750607c489af1921ac197706`.

Fresh host: AMD EPYC 9V74 80-Core Processor. Eight independent processes, A B B A B A A B, CPU 0, workers/reference pools 1, unchanged seven rounds / 19 ms. All 1,008 samples, 72 returned timing records, source/runtime/core/SDK identities, output hashes/layouts and deadline passed the frozen reviewer on its first attempt.

Balanced direct GM is 1.719903535981; normalized GM is 1.691622894225, only for these nine frozen whole-public workloads. Every case improves on both balanced metrics, but aggregate or balanced gains do not override the failed worst-pair condition. V1 remains separately closed as UNRESOLVED_MEASUREMENT with its fixed incomplete-u8 floor failures.

## Every workload

| Case | Balanced A CPP / B CPP | Normalized B / A | Candidate/original GM | Worst A/B direct | Worst normalized |
|---|---:|---:|---:|---:|---:|
| gray-small-u8 | 1.073795863 | 1.061572441 | 1.665471220 | 1.044078259 | 1.037520974 |
| gray-incomplete-u8 | 1.355156165 | 1.348046172 | 1.517794774 | 1.313869999 | 1.298017829 |
| gray-dense-f32 | 1.554037820 | 1.486145859 | 2.231763497 | 1.433853656 | 1.417655298 |
| gray-large-u16 | 1.423687574 | 1.427259917 | 2.026135633 | 1.307901708 | 1.317997120 |
| gray-mean-f64 | 1.327650853 | 1.313237720 | 1.302585739 | 1.274707444 | 1.270145787 |
| rgb-padded-u8 | 1.285749997 | 1.268423757 | 6.769598570 | 1.210609055 | 1.201677990 |
| rgb-first-f32 | 1.534556096 | 1.527224619 | 2.854084285 | 1.451839470 | 1.443544276 |
| gray-strided-reverse-f32 | 13.960532006 | 13.928357308 | 2.277137070 | 12.740443606 | 12.640492857 |
| gray-fortran-f64 | 1.118469913 | 1.054635894 | 1.091406134 | 1.018541251 | 0.885140365 |

## Exact blocker

Fortran balanced direct and normalized gains are 1.118469913 and 1.054635894. Its B2/B3 direct null ratio is 1.110570897 and normalized null ratio is 1.087690272, both outside [0.97, 1/0.97]. The direct worst-pair gain 1.018541251 passes the required 0.97 floor; the normalized worst-pair gain 0.885140365 fails it. The only unresolved metric is therefore Fortran normalized timing.

Fortran B2 original/current is 0.983922393; that individual loss is retained. Its other candidate original/current ratios are 1.070202816, 1.236823650 and 1.089458888. All six prospectively preserved original-win cases and all target original comparisons pass in every candidate pass.

The previously failing incomplete-u8 case now has balanced direct/normalized gains 1.355156165 / 1.348046172 and worst gains 1.313869999 / 1.298017829. Both same-product null metrics stay inside the band. This is a full-public comparison on this fresh host, not an isolated stage attribution or a cross-host v1/v2 ratio.

## Fixed gates

- `overall_direct`: PASS
- `overall_normalized`: PASS
- `target_each_candidate_direct`: PASS
- `target_each_candidate_normalized`: PASS
- `target_each_candidate_original`: PASS
- `controls_balanced_direct`: PASS
- `controls_balanced_normalized`: PASS
- `preserved_original_each_candidate`: PASS
- `rgb_first_each_candidate_direct`: PASS
- `rgb_first_each_candidate_normalized`: PASS

## All null controls

| Case | Pair | Direct earlier/later | Reference earlier/later | Normalized later/earlier |
|---|---|---:|---:|---:|
| gray-small-u8 | B2 / B3 | 0.989372458 | 0.995330968 | 0.994013539 |
| gray-small-u8 | A6 / A7 | 1.030491308 | 1.014765259 | 1.015497228 |
| gray-incomplete-u8 | B2 / B3 | 1.023029494 | 1.005224763 | 1.017712189 |
| gray-incomplete-u8 | A6 / A7 | 0.998330001 | 1.008687684 | 0.989731527 |
| gray-dense-f32 | B2 / B3 | 1.092904487 | 1.064442518 | 1.026738850 |
| gray-dense-f32 | A6 / A7 | 0.993214428 | 1.017299188 | 0.976324801 |
| gray-large-u16 | B2 / B3 | 1.008429396 | 0.985467652 | 1.023300353 |
| gray-large-u16 | A6 / A7 | 0.996986497 | 0.996396868 | 1.000591761 |
| gray-mean-f64 | B2 / B3 | 1.040952880 | 1.009040651 | 1.031626307 |
| gray-mean-f64 | A6 / A7 | 0.999199805 | 1.002590473 | 0.996618094 |
| rgb-padded-u8 | B2 / B3 | 1.098394427 | 1.011780267 | 1.085605702 |
| rgb-padded-u8 | A6 / A7 | 1.023972945 | 1.007200064 | 1.016652978 |
| rgb-first-f32 | B2 / B3 | 1.008719151 | 1.012547425 | 0.996219166 |
| rgb-first-f32 | A6 / A7 | 0.997985093 | 0.999428231 | 0.998556036 |
| gray-strided-reverse-f32 | B2 / B3 | 0.870072135 | 0.988056485 | 0.880589468 |
| gray-strided-reverse-f32 | A6 / A7 | 1.002997090 | 1.007029000 | 0.995996232 |
| gray-fortran-f64 | B2 / B3 | 1.110570897 | 1.021035975 | 1.087690272 |
| gray-fortran-f64 | A6 / A7 | 1.015283343 | 1.017880632 | 0.997448336 |

## Whole-public pass medians (milliseconds)

| Case | A1 original / CPP | B2 original / CPP | B3 original / CPP | A4 original / CPP | B5 original / CPP | A6 original / CPP | A7 original / CPP | B8 original / CPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.008117858 / 0.005262808 | 0.007982496 / 0.004773782 | 0.008019942 / 0.004825061 | 0.008092706 / 0.005169909 | 0.008081074 / 0.004834582 | 0.008192114 / 0.005211681 | 0.008072916 / 0.005057472 | 0.008022214 / 0.004843959 |
| gray-incomplete-u8 | 0.017040411 / 0.015465757 | 0.016978117 / 0.011407202 | 0.016889872 / 0.011150414 | 0.017215200 / 0.015013514 | 0.017136865 / 0.011426940 | 0.017362378 / 0.015315128 | 0.017212838 / 0.015340747 | 0.017469009 / 0.011129182 |
| gray-dense-f32 | 0.074997432 / 0.050207839 | 0.075456944 / 0.034620894 | 0.070888697 / 0.031677877 | 0.075175688 / 0.050954228 | 0.071384061 / 0.031734282 | 0.076319127 / 0.049641295 | 0.075021319 / 0.049980441 | 0.070707232 / 0.031269835 |
| gray-large-u16 | 0.116476753 / 0.082306175 | 0.115735482 / 0.056515632 | 0.117442193 / 0.056043221 | 0.116763865 / 0.082232341 | 0.117378596 / 0.062625531 | 0.116410629 / 0.081908038 | 0.116831588 / 0.082155615 | 0.117104759 / 0.055890369 |
| gray-mean-f64 | 0.178754213 / 0.178118169 | 0.178114525 / 0.139732588 | 0.176518681 / 0.134235267 | 0.178247955 / 0.183606768 | 0.175093689 / 0.134725718 | 0.178813961 / 0.179094329 | 0.178351945 / 0.179237754 | 0.176700384 / 0.133708151 |
| rgb-padded-u8 | 0.158014449 / 0.029626124 | 0.158042884 / 0.024255636 | 0.156202774 / 0.022082811 | 0.158879516 / 0.029948931 | 0.156233909 / 0.024080084 | 0.159684044 / 0.030068036 | 0.158542528 / 0.029364093 | 0.156086150 / 0.022224228 |
| rgb-first-f32 | 0.125222430 / 0.068233769 | 0.125898505 / 0.043748229 | 0.124338379 / 0.043370079 | 0.125199895 / 0.070527311 | 0.124864587 / 0.044671686 | 0.125582110 / 0.064856117 | 0.125653955 / 0.064987060 | 0.124166820 / 0.043153935 |
| gray-strided-reverse-f32 | 0.081583394 / 0.500033394 | 0.080799505 / 0.031487193 | 0.081776200 / 0.036189175 | 0.080911180 / 0.501098613 | 0.081708634 / 0.039174033 | 0.082601544 / 0.500590394 | 0.082024990 / 0.499094563 | 0.082079142 / 0.036920601 |
| gray-fortran-f64 | 0.839652650 / 0.755354750 | 0.716319913 / 0.728024810 | 0.701561875 / 0.655541048 | 0.716432304 / 0.741523300 | 0.802154571 / 0.648560182 | 0.785096227 / 0.758752158 | 0.771304810 / 0.747330450 | 0.714341125 / 0.655684333 |

## Scope and retention

The unchanged harness checks full original/current/oracle output words, ownership/layout/mutation and caller FP state internally; it exports hashes/layouts, not full buffers or separate FP values. Independent review verifies all exported raw samples, timing journals, hashes/layouts and exact gate booleans.

One artifact was downloaded once. Frozen reviewer SHA `48753d6b63a23fbb802332fe1216e7c18119ff606e5431a9553ea5bae81a0928` was unchanged, including the independently calculated gate fragment `da1f9b8531a8694a64864fc55cd257ddf466b231bffe7424c2ee81367401d68b`. No source edits, changed gates, automatic rerun, additional timing or old-host causal claim followed this result. Qualified whole-product baseline remains 241.

## Retained calibration audit

The root reconstruction from raw per-call samples and loop counts finds all 144 function/case/pass median batch durations below the requested 19 ms: **1.695730–17.272117 ms**. This demonstrates a weakness in the old calibration protocol; it is not a new timing or proof of host causality. The Fortran blocker is **normalized-timing uncertainty**, not a reference-only-noise claim. A separately developed sustained protocol cannot alter either closed result. The full duration receipt is embedded verbatim in the JSON.
