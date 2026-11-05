# Region/stain warm-lookup candidate: DISCARD

Run41969983206 compares baseline`eea321b2` with repaired candidate`58b54456` using the unchanged controller`119eb28b` and fixed20-case protocol. One isolated AMD EPYC9V45 host, one global worker, four ABBA passes and1,120 accepted samples; all six preflights and before/after installed-input checks passed.

The tiny-constructor control now passes both pairs. One other fixed check fails: `old-f32-combine-hed-f64matrix`, pair1, original-normalized ratio1.082979863 above1.05. Its direct C++ ratio1.000552484 passes. Candidate acceptance remains **DISCARD**; the first candidate decision and all losses remain unchanged.

| Case | Pair1 candidate/prior | Pair1 normalized | Pair2 candidate/prior | Pair2 normalized | Limit | Result |
|---|---:|---:|---:|---:|---:|---|
| regionprops-state/tiny2d-constructor | 1.015638380 | 1.011625630 | 1.007971245 | 0.999553323 | 1.05 | PASS |
| regionprops-state/tiny2d-properties | 1.008758976 | 0.980407544 | 1.013997784 | 0.982278664 | 1.05 | PASS |
| regionprops-state/medium2d-constructor-sparse | 0.582551432 | 0.564755821 | 0.557742433 | 0.559811832 | 1.05 | PASS |
| regionprops-state/medium2d-properties-cache-on | 0.396344627 | 0.384789679 | 0.382625957 | 0.387855550 | 1.05 | PASS |
| regionprops-state/medium2d-transpose-cache-off | 0.105658408 | 0.100114516 | 0.110514465 | 0.107008189 | 1.05 | PASS |
| regionprops-state/large2d-c-cache-off | 0.414350674 | 0.427905853 | 0.392176322 | 0.420254477 | 1.05 | PASS |
| regionprops-state/large2d-f-coords | 0.150732177 | 0.182881365 | 0.155649474 | 0.163036644 | 0.50 | PASS |
| regionprops-state/medium2d-reverse-sparse | 0.347495872 | 0.395603187 | 0.328095871 | 0.341147144 | 1.05 | PASS |
| regionprops-state/small3d-constructor | 0.921403835 | 0.925556915 | 0.896312939 | 0.918884167 | 1.05 | PASS |
| regionprops-state/medium3d-f-cache-off | 0.080473037 | 0.087227080 | 0.081639662 | 0.082905630 | 0.50 | PASS |
| regionprops-state/large3d-cache-off | 0.340738834 | 0.391402815 | 0.351319116 | 0.364877026 | 1.05 | PASS |
| regionprops-state/large3d-sparse-spacing-offset | 0.205652946 | 0.228695858 | 0.213303911 | 0.249745710 | 1.05 | PASS |
| combine-half-bench/standard-f32 | 0.460750490 | 0.524450545 | 0.490070563 | 0.478059290 | 0.80 | PASS |
| combine-half-bench/standard-f64 | 0.981548697 | 0.990628338 | 0.986054536 | 0.990167911 | 1.05 | PASS |
| combine-half-bench/custom-k7 | 0.407766410 | 0.443658757 | 0.402238475 | 0.400188715 | 0.80 | PASS |
| combine-half-bench/reverse-k1 | 0.571276180 | 0.590115818 | 0.566798465 | 0.594559690 | 1.05 | PASS |
| old-f32-combine-custom1 | 0.227873767 | 0.226755780 | 0.218420447 | 0.221130947 | 1.05 | PASS |
| old-f32-combine-custom7 | 0.302450481 | 0.299030549 | 0.296951252 | 0.301730213 | 1.05 | PASS |
| old-f32-separate-custom7 | 1.008014236 | 1.038429450 | 0.989512392 | 0.998284630 | 1.05 | PASS |
| old-f32-combine-hed-f64matrix | 1.000552484 | 1.082979863 | 0.969245788 | 0.931396153 | 1.05 | FAIL |

Both direct and original-normalized limits apply separately to baselinepass1/candidatepass2 and baselinepass4/candidatepass3. No average, noise waiver, retry or partial promotion follows the rejection. All40 gates remain in `decision.json`; every accepted sample and all12 journals are adjacent.

For the sole failed workload, these are the exact paired medians:

| Pass | Role | Original ms | C++ ms | Original sample CV | C++ sample CV |
|---|---|---:|---:|---:|---:|
| 1 | baseline | 1.642591257 | 2.155517130 | 15.152% | 2.933% |
| 2 | candidate | 1.517570935 | 2.156708019 | 2.667% | 1.261% |
| 3 | candidate | 1.532274849 | 2.072739889 | 2.021% | 1.666% |
| 4 | baseline | 1.472438590 | 2.138508019 | 0.850% | 1.024% |

Pair1 C++ latency rises0.055%, while the original median falls7.611%; division by that smaller reference produces the normalized failure. Pair2 passes both limits. These numbers identify the denominator component; they do not establish a CPU, cache or other timing cause. The underlying reference-window mechanism remains unknown.

The float32 image plus float64 HED matrix produces float64 output. That route uses the existing double-output SIMD path and cannot enter the new float-input/float-output SIMD subset. The relevant existing bodies and wrapper are unchanged. This rules out a missing-SIMD explanation for this particular holdout; it does not prove complete machine-code/layout equality or justify another C++ patch.

Full original report bodies and validation payloads remain in artifact13389417631 (`5383d4cb67967afbba7264a2180c6bdbb1ab83013c45898822d8167e23237d34`); their exact member hashes are recorded alongside the compact timing projection. Source/runtime/core identities and the independent review receipt are in the linked [evidence capsule](../../coverage/evidence/region-stain-warm-same-host-discard-v1.json).

No diagnostic, measurement, download or product build was performed for publication. This is a rejected optimization experiment, not whole-library or full-profile qualification.
