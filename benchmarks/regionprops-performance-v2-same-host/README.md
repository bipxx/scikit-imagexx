# Region optimization v2: fixed one-worker comparison

KEEP versus prior C++: all56 paired gates (112 direct/normalized inequalities) pass across1568 samples. Every gate was fixed before this run. Ten raw C++ increases across paired comparisons remain within the declared1.05 limit; this does not mean every latency improved.

Each candidate pass wins9/12 Region state cases,0/8 scalar cases and0/8 unchanged stain controls against original scikit-image. All19 original losses remain open. All four release qualification jobs for exactcf0 completed successfully in run42009401281. Independent archive admission and production promotion remain pending.

Prior C++ / candidate >1 is faster. Original / candidate >1 beats scikit-image. Values below1 remain visible.

| Case | Prior/C++ pass2 | Prior/C++ pass3 | Original/C++ pass2 | Original/C++ pass3 |
|---|---:|---:|---:|---:|
| regionprops-state/tiny2d-constructor | 0.982033 | 0.995994 | 0.851105 | 0.855863 |
| regionprops-state/tiny2d-properties | 1.032945 | 1.054322 | 1.238323 | 1.263468 |
| regionprops-state/medium2d-constructor-sparse | 2.104864 | 2.161567 | 1.184703 | 1.207998 |
| regionprops-state/medium2d-properties-cache-on | 3.134507 | 3.172319 | 1.463386 | 1.471861 |
| regionprops-state/medium2d-transpose-cache-off | 11.146782 | 11.091108 | 1.205971 | 1.197778 |
| regionprops-state/large2d-c-cache-off | 2.937647 | 2.917910 | 1.407820 | 1.407082 |
| regionprops-state/large2d-f-coords | 7.165213 | 7.162796 | 1.549654 | 1.578618 |
| regionprops-state/medium2d-reverse-sparse | 4.451432 | 4.413335 | 1.702514 | 1.699280 |
| regionprops-state/small3d-constructor | 1.091199 | 1.117920 | 0.875207 | 0.876212 |
| regionprops-state/medium3d-f-cache-off | 12.801473 | 12.869421 | 0.875988 | 0.882960 |
| regionprops-state/large3d-cache-off | 3.155267 | 3.144401 | 1.238527 | 1.245943 |
| regionprops-state/large3d-sparse-spacing-offset | 5.526886 | 5.519539 | 1.592653 | 1.599844 |
| regionprops-scalars/tiny-ctor-c-off | 1.148391 | 1.122683 | 0.868684 | 0.864898 |
| regionprops-scalars/medium-ctor-f-on | 5.079807 | 5.044877 | 0.835002 | 0.846619 |
| regionprops-scalars/tiny-warm-c-on | 1.722571 | 1.712597 | 0.452331 | 0.444566 |
| regionprops-scalars/medium-getters-c-off | 1.053430 | 1.057622 | 0.630619 | 0.648702 |
| regionprops-scalars/medium-getters-f-off | 15.431756 | 15.534846 | 0.442096 | 0.442266 |
| regionprops-scalars/3d-warm-f-on | 1.731509 | 1.778432 | 0.444081 | 0.449083 |
| regionprops-scalars/pixel-cycle-c-off | 1.206142 | 1.176753 | 0.852818 | 0.860582 |
| regionprops-scalars/slice-cycle-3d-on | 2.604914 | 2.584810 | 0.860903 | 0.865466 |
| combine-half-bench/standard-f32 | 0.995641 | 0.994850 | 0.505890 | 0.506016 |
| combine-half-bench/standard-f64 | 0.997890 | 1.001799 | 0.973436 | 0.971419 |
| combine-half-bench/custom-k7 | 0.999037 | 0.998085 | 0.823538 | 0.834177 |
| combine-half-bench/reverse-k1 | 1.002443 | 0.999072 | 0.652107 | 0.652210 |
| old-f32-combine-custom1 | 1.004772 | 1.005622 | 0.452054 | 0.455765 |
| old-f32-combine-custom7 | 0.997105 | 0.997849 | 0.801955 | 0.805991 |
| old-f32-separate-custom7 | 1.000265 | 1.000233 | 0.258296 | 0.259257 |
| old-f32-combine-hed-f64matrix | 1.000082 | 1.002286 | 0.922387 | 0.921721 |

Host: Intel(R) Xeon(R) 6975P-C; one native worker and all reference pools1. Fixed ABBA roles are baseline/candidate/candidate/baseline. Source6a/core9de versus cf0/core52be; all14 offline wheels and inputs verified before/after. All8 role/family preflights completed before any timed batch.

This is a bounded optimization decision, not full API ownership, whole-library superiority or four-profile release acceptance. Earlier rejected experiments retain their rejected decisions. Full reports/validation and process evidence remain in artifact13404827454.
