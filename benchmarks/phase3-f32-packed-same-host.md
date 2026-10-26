# Packed float32 v6: rejected — no end-to-end gain

**DISCARD.** Candidate `8641d0c0ba763c06169080741641f4fb53fb2aeb` remains on its experiment branch; its production change is not merged into main. Kept [float32 v5](phase3-f32-same-host.md) remains unchanged. Actual `movups`/`addps` instructions and passing correctness controls did not produce a useful public-call improvement.

[Controlled run 40286133008](https://github.com/bipxx/scikit-imagexx/actions/runs/40286133008) ran baseline–candidate–candidate–baseline on one AMD EPYC 7763 host: CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Each pass retained all 73 cases, seven paired rounds, one worker and exact untimed parity: 1,022 samples per report, 4,088 total.

The fixed metric is the equal-weight geometric mean of original/current ratios for all 21 colocalization cases. Compare the geometric mean of the two candidate scores with that of the two baseline scores. Direct gain instead compares baseline/candidate public-call times; greater than one is better in both columns.

| ABBA pass | Fixed21 score |
|---|---:|
| 1-baseline | 0.2411219943859× |
| 2-candidate | 0.2368088538961× |
| 3-candidate | 0.2453397722039× |
| 4-baseline | 0.2475879102715× |

Combined baseline **0.2443335644370×** versus candidate **0.2410365745499×** gives **0.986506× normalized gain (−1.35%)** and **0.994889× direct gain (−0.51%)**. There are **13 normalized losses and 10 direct losses** among the fixed 21. The ten targeted float32 cases give only **1.004120× direct gain**, with four direct losses; only four targets have both candidate medians below both baseline medians.

Reference candidate/baseline latency is **0.991574×**; baseline pass4/pass1 is **1.003773×**, and candidate pass3/pass2 is **1.008536×**. The unchanged 52 footprint cases have direct gain **1.010838×**. These small shifts do not establish a universal CPU result or a cumulative gain across earlier hosts. No identical repeat is used to seek a preferred outcome.

All 21 cases are listed below. **Bold ratios are losses.** IC is intersection coefficient, MCC is Manders colocalization coefficient, and MOC is Manders overlap coefficient. Original/current columns retain both candidate passes; the same five IC cases beat the original, while **16 colocalization cases remain slower**.

| Case | Normalized gain | Direct gain | Candidate2 original/current | Candidate3 original/current |
|---|---:|---:|---:|---:|
| IC / tiny | 1.043264× | 1.018352× | 1.879708× | 1.813262× |
| IC / cache | **0.989545×** | **0.976840×** | 2.213920× | 2.203635× |
| IC / memory | 1.005213× | 1.020834× | 2.258713× | 2.213997× |
| IC / volume | **0.992777×** | **0.998361×** | 2.263175× | 2.365211× |
| IC / stepped | **0.995243×** | 1.000132× | 0.328612× | 0.325353× |
| IC / masked | **0.905840×** | **0.895562×** | 0.832426× | 0.812561× |
| MCC / tiny | 1.029276× | 1.000181× | 0.418626× | 0.402415× |
| MCC / cache | 1.000506× | 1.030021× | 0.061394× | 0.065801× |
| MCC / memory | **0.962357×** | 1.001284× | 0.069966× | 0.071499× |
| MCC / volume | 1.007724× | **0.996401×** | 0.069258× | 0.070857× |
| MCC / stepped | 1.031859× | 1.014199× | 0.125539× | 0.125609× |
| MCC / masked | **0.990036×** | **0.978489×** | 0.338821× | 0.335013× |
| MOC / tiny | **0.992929×** | **0.994703×** | 0.296059× | 0.287608× |
| MOC / cache | **0.953794×** | 1.003797× | 0.044603× | 0.045226× |
| MOC / memory | **0.948140×** | 1.002020× | 0.061219× | 0.063810× |
| MOC / volume | 1.010989× | **0.999176×** | 0.054048× | 0.054412× |
| MOC / stepped | 1.025095× | 1.010341× | 0.080158× | 0.081469× |
| MOC / masked | **0.973909×** | **0.983456×** | 0.198159× | 0.200283× |
| IC / cache-F | **0.975112×** | **0.950133×** | 1.188453× | 2.361620× |
| MCC / cache-F | **0.976320×** | 1.028947× | 0.063429× | 0.062628× |
| MOC / cache-F | **0.919506×** | **0.998383×** | 0.045165× | 0.044999× |

The ten targeted float32 cases retain all four direct-call medians (milliseconds), including unfavorable repetitions.

| Target | Baseline1 ms | Candidate2 ms | Candidate3 ms | Baseline4 ms |
|---|---:|---:|---:|---:|
| MCC / tiny | 0.042099949 | 0.040535481 | 0.044176506 | 0.042550255 |
| MCC / cache | 1.202802111 | 1.166676111 | 1.163622900 | 1.197458222 |
| MCC / stepped | 1.200880500 | 1.184463889 | 1.181092182 | 1.198263800 |
| MCC / masked | 1.264370375 | 1.279311125 | 1.277385875 | 1.237474111 |
| MOC / tiny | 0.083222435 | 0.083695397 | 0.082337386 | 0.081930370 |
| MOC / cache | 2.638283400 | 2.642543750 | 2.614979800 | 2.639132750 |
| MOC / stepped | 2.650943200 | 2.621381000 | 2.617245800 | 2.641863000 |
| MOC / masked | 2.231586800 | 2.243825400 | 2.252231600 | 2.190272667 |
| MCC / cache-F | 1.193971727 | 1.164076500 | 1.158762333 | 1.196100800 |
| MOC / cache-F | 2.645286400 | 2.657068000 | 2.635092250 | 2.638275800 |

The remaining losses among the 52 unchanged footprint cases are also retained. Across all 73 cases there are **35 normalized and 22 direct losses**; a case appears below if either ratio is below one.

| Footprint case | Normalized gain | Direct gain |
|---|---:|---:|
| ellipse/tiny/uint8/None | **0.891514×** | **0.948623×** |
| ellipse/cache/uint8/None | **0.991831×** | **0.991721×** |
| ellipse/cache/uint8/crosses | **0.980886×** | **0.994371×** |
| ellipse/large/uint8/None | **0.967577×** | **0.972362×** |
| ellipse/large/uint8/crosses | **0.966384×** | **0.975243×** |
| ellipse/large/float64/None | **0.994100×** | 1.011712× |
| ellipse/large/float64/crosses | **0.987390×** | **0.995873×** |
| octagon/tiny/float64/None | **0.999394×** | 1.030722× |
| octagon/tiny/float64/sequence | **0.980974×** | 1.020062× |
| octagon/cache/uint8/None | **0.989569×** | 1.001583× |
| octagon/cache/float64/None | **0.994231×** | **0.997306×** |
| octagon/cache/float64/sequence | **0.978827×** | 1.010872× |
| octahedron/tiny/uint8/None | **0.973209×** | 1.003495× |
| octahedron/cache/uint8/sequence | **0.979510×** | 1.011919× |
| octahedron/cache/float64/None | **0.998626×** | 1.015542× |
| octahedron/large/uint8/None | 1.023080× | **0.998831×** |
| octahedron/large/float64/sequence | **0.981951×** | **0.995521×** |
| star/tiny/uint8/None | **0.969492×** | **0.964520×** |
| star/large/float64/None | **0.998308×** | 1.003946× |
| footprint_from_sequence/tiny/C | **0.996917×** | 1.014266× |
| footprint_from_sequence/tiny/F | **0.976952×** | **0.989956×** |
| footprint_from_sequence/cache/C | **0.999245×** | 1.010964× |
| octagon/tiny/even-sequence-warning | **0.999031×** | **0.981535×** |

Baseline source `5b03eded95b86c22de7cea6c76243b1748969a64` reused the wheel compiled from `c53198aff0a502fdfe236c6b46834bbc58547abd`, prepared in [40277634220](https://github.com/bipxx/scikit-imagexx/actions/runs/40277634220). Candidate source/producer `8641d0c0ba763c06169080741641f4fb53fb2aeb` passed [build 40285296611](https://github.com/bipxx/scikit-imagexx/actions/runs/40285296611) and [preparation 40285975359](https://github.com/bipxx/scikit-imagexx/actions/runs/40285975359). The ABBA controller was `5d37e10aebe0f79c54a6a81903a797988e7c2301`.

Candidate preparation passed **788 installed public tests per mode** (240 footprint +545 colocalization +3 new reduction controls), **1,318 smoke checks**, **21 metadata tests plus 30 subtests**, and **73 corpus checks per mode**. The 539 fresh-original/native pairs per mode match all nine semantic fields. Their 44 frozen MOC word differences use the reviewed live-reference policy; all other frozen fields remain exact. Baseline preparation retains its historical 785 count. Both source packages, wheels and cores are authenticated; 14 dependency wheels are identical, and all eight before/after identity checks passed. No whole-product/SDK qualification follows.

The [original failed preparation](https://github.com/bipxx/scikit-imagexx/actions/runs/40273141051), [live-reference diagnosis](https://github.com/bipxx/scikit-imagexx/actions/runs/40275501905) and [policy checkpoint](../coverage/evidence/colocalization-live-reference-checkpoint.json) remain unchanged. No tolerance, fixed case or failure was removed for this experiment.

[Exact raw artifact ZIP](phase3-f32-packed-same-host.zip): all four complete reports, logs, identity checks and selected prepared-archive records; **133,676 bytes**, SHA256 `737bcd76cb333aae6dc1a704ccd1cd01baecc1e65ecb05246acdf34bad8f9c94`. Every original-library regression and all samples remain in those reports. The independent local review is `build/phase3-f32-packed-same-host-results-v6/review.json` (SHA256 `5865467de14787d12f19a20d977e80c1b80f5bc82755b05b5f4b7194ae8f53c1`).
