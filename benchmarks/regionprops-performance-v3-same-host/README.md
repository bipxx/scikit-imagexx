# Region comparison v3: rejected candidate

DISCARD: five paired checks / eight inequalities fail the fixed gates. All1568 samples and5032 batch-journal events are retained. No rerun or threshold waiver.

Both warm targets improved1.14–1.23x versus prior C++, but sparse2D construction regressed10.6% and12.3%; large3D sparse regressed5.7% in one pair. Two stain controls had nearly unchanged C++ latency but failed original-normalized gates. Their source cause remains unknown.

Each candidate pass beats original scikit-image in8/28 cases and loses20/28; all eight scalar cases still lose. No code promotion or library-wide speed claim follows. The input package was genuinely prepared; four-profile release remains separate.

Prior/C++ >1 means faster than prior C++; normalized ratios must satisfy their fixed cap. Original/C++ >1 beats scikit-image.

| Case | Prior/C++ pair1 | Prior/C++ pair2 | Normalized pair1 | Normalized pair2 | Original/C++ pass2 | Original/C++ pass3 |
|---|---:|---:|---:|---:|---:|---:|
| regionprops-state/tiny2d-constructor | 1.014326 | 1.021712 | 0.992134 | 0.989540 | 0.847070 | 0.842767 |
| regionprops-state/tiny2d-properties | 1.004154 | 1.011696 | 0.993683 | 0.985920 | 1.259609 | 1.249476 |
| regionprops-state/medium2d-constructor-sparse | 0.904010 | 0.890716 | 1.120880 | 1.131323 | 0.965923 | 0.965105 |
| regionprops-state/medium2d-properties-cache-on | 0.988648 | 0.964893 | 1.006035 | 1.034076 | 1.466708 | 1.473768 |
| regionprops-state/medium2d-transpose-cache-off | 0.982789 | 0.967345 | 1.033319 | 1.031844 | 1.121843 | 1.130878 |
| regionprops-state/large2d-c-cache-off | 0.978530 | 0.972735 | 1.026558 | 1.044747 | 1.155299 | 1.156410 |
| regionprops-state/large2d-f-coords | 0.967814 | 0.965973 | 1.044443 | 1.039668 | 1.411243 | 1.429636 |
| regionprops-state/medium2d-reverse-sparse | 0.961755 | 0.959239 | 1.040243 | 1.042961 | 1.610865 | 1.610474 |
| regionprops-state/small3d-constructor | 1.021469 | 1.025584 | 0.989684 | 0.984566 | 0.861447 | 0.854416 |
| regionprops-state/medium3d-f-cache-off | 0.996137 | 0.989026 | 1.010141 | 1.012608 | 0.952445 | 0.957161 |
| regionprops-state/large3d-cache-off | 1.000186 | 1.030839 | 1.000986 | 0.970210 | 1.259561 | 1.241351 |
| regionprops-state/large3d-sparse-spacing-offset | 0.945671 | 0.961458 | 1.057670 | 1.049129 | 1.837838 | 1.827443 |
| regionprops-scalars/tiny-ctor-c-off | 1.031950 | 1.011848 | 0.960127 | 0.977557 | 0.900553 | 0.887473 |
| regionprops-scalars/medium-ctor-f-on | 1.043932 | 1.038838 | 0.951419 | 0.928019 | 0.960603 | 0.976990 |
| regionprops-scalars/tiny-warm-c-on | 1.228895 | 1.174740 | 0.797370 | 0.861153 | 0.580734 | 0.553756 |
| regionprops-scalars/medium-getters-c-off | 1.058313 | 1.043198 | 0.991577 | 1.000285 | 0.608194 | 0.601146 |
| regionprops-scalars/medium-getters-f-off | 1.046422 | 1.050784 | 0.963596 | 0.941090 | 0.398236 | 0.402421 |
| regionprops-scalars/3d-warm-f-on | 1.209038 | 1.135813 | 0.817890 | 0.899756 | 0.582496 | 0.540928 |
| regionprops-scalars/pixel-cycle-c-off | 1.017280 | 1.006357 | 0.991028 | 0.980557 | 0.960444 | 0.963000 |
| regionprops-scalars/slice-cycle-3d-on | 1.002412 | 0.992380 | 0.997207 | 0.987714 | 0.903626 | 0.908879 |
| combine-half-bench/standard-f32 | 0.998546 | 0.998891 | 0.990345 | 0.988206 | 0.498082 | 0.496947 |
| combine-half-bench/standard-f64 | 1.000462 | 1.001545 | 0.996511 | 1.032846 | 0.701943 | 0.699977 |
| combine-half-bench/custom-k7 | 0.998288 | 0.999998 | 1.007654 | 1.034044 | 0.581092 | 0.575294 |
| combine-half-bench/reverse-k1 | 0.999366 | 1.000043 | 0.992305 | 1.009990 | 0.579464 | 0.574528 |
| old-f32-combine-custom1 | 1.001939 | 1.002593 | 0.981110 | 1.073243 | 0.257255 | 0.235820 |
| old-f32-combine-custom7 | 0.998153 | 1.006132 | 0.984324 | 0.857543 | 0.535301 | 0.629092 |
| old-f32-separate-custom7 | 1.005065 | 0.999979 | 0.999121 | 1.010175 | 0.270640 | 0.271178 |
| old-f32-combine-hed-f64matrix | 1.002008 | 1.012625 | 1.058848 | 1.030893 | 0.547357 | 0.555968 |

Source: baseline `cf0abd3f`, candidate `bd16b09e`; [run42032588452](https://github.com/bipxx/scikit-imagexx/actions/runs/42032588452). Fixed one worker and numerical pools1, four ABBA passes, seven accepted100ms samples per arm, at least250ms warmup. Raw calibration attempts remain in journals and are not counted as accepted samples. Full reports and two nested prepared packages remain in retained artifact13417530830; the nested archives were not extracted/rehashed for publication.

Closeout: useful rejected experiment; warm improvement cannot override regressions. The unchanged SDK prepare machine body rules out a changed geometry algorithm as the cause; linked caller/Python metadata attribution remains open. A local intrusive cProfile diagnostic is separate evidence and cannot relabel this failed result.
