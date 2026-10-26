# Integer octahedron: kept after controlled same-host comparison

**KEEP.** Reviewed source `9da7ccfd0f38902bc231de9021732861b757df2b` is merged into main `f079c6d35e41d836828fbe0741dbf7af1bf336cd`, with unchanged product source. Exact builtin nonnegative integer radii use the owned C++ plane/row fill; other radius types retain the original grid path and final dtype conversion. The independent source review and the measured target gains support this bounded change. No universal-profile or whole-product qualification is claimed.

[Controlled run 40286815914](https://github.com/bipxx/scikit-imagexx/actions/runs/40286815914) used one AMD EPYC 9V45 host, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Order was baseline–candidate–candidate–baseline. Each report contains the same 73 cases, seven paired rounds and one worker: **4,088 samples total**, with exact untimed parity before timing.

For each report, take the median of its seven samples first. The target score is the equal-weight geometric mean of original/current ratios for the 12 fixed octahedron cases; compare the geometric means of the two candidate and two baseline scores. Direct gain compares baseline/candidate public-call medians. **No 14-sample pooling across repetitions is used.** The earlier pooled preliminary arithmetic remains retained but is not the publication method.

| ABBA pass | Target12 score |
|---|---:|
| 1-baseline | 1.127923600417× |
| 2-candidate | 13.835559305697× |
| 3-candidate | 13.676111860489× |
| 4-baseline | 1.138475228894× |

Balanced baseline **1.133187133336×** versus candidate **13.755604556585×** gives **12.138864× normalized gain** and **12.070450× direct gain**. In **all 12 cases**, both candidate medians are below both baseline medians, and both candidates beat the original library. Large dense radius 47 improves **41.276711× directly for uint8** and **12.559258× for float64**. Sequence cases retain their source-prescribed radius 1 construction, rather than claiming a dense radius 47 result.

Reference candidate/baseline latency is **1.005668×**; baseline4/baseline1 is **1.004836×**, and candidate3/candidate2 is **1.005949×**. The other 61 cases give direct gain **0.993250×**; their losses are retained below. These are allocation-inclusive public calls on this host, not isolated kernel timings or cumulative gains across different CPUs.

All 12 target gains and both original/current ratios:

| Octahedron case | Normalized gain | Direct gain | Candidate2 original/current | Candidate3 original/current |
|---|---:|---:|---:|---:|
| tiny/uint8/None | 10.841440× | 10.715814× | 11.744582× | 11.339362× |
| tiny/uint8/sequence | 10.067806× | 9.837364× | 10.748656× | 10.840632× |
| tiny/float64/None | 10.620664× | 10.494271× | 11.397530× | 11.058690× |
| tiny/float64/sequence | 9.985931× | 10.017194× | 10.864197× | 10.566773× |
| cache/uint8/None | 18.284732× | 18.968311× | 22.656824× | 22.276726× |
| cache/uint8/sequence | 10.200879× | 9.827464× | 10.797419× | 10.741691× |
| cache/float64/None | 9.399242× | 9.534826× | 11.471677× | 11.165684× |
| cache/float64/sequence | 9.906624× | 9.737954× | 10.659790× | 10.593658× |
| large/uint8/None | 40.675389× | 41.276711× | 52.662384× | 57.921579× |
| large/uint8/sequence | 10.042428× | 9.836974× | 10.822361× | 10.656413× |
| large/float64/None | 12.460280× | 12.559258× | 16.961568× | 16.378895× |
| large/float64/sequence | 10.003031× | 9.779080× | 10.883723× | 10.415284× |

All four direct-call medians are retained in milliseconds:

| Octahedron case | Baseline1 ms | Candidate2 ms | Candidate3 ms | Baseline4 ms |
|---|---:|---:|---:|---:|
| tiny/uint8/None | 0.013607923 | 0.001271693 | 0.001313053 | 0.014090388 |
| tiny/uint8/sequence | 0.013351823 | 0.001387713 | 0.001381214 | 0.013892409 |
| tiny/float64/None | 0.013533961 | 0.001309857 | 0.001336217 | 0.014242312 |
| tiny/float64/sequence | 0.013816070 | 0.001357628 | 0.001392408 | 0.013729506 |
| cache/uint8/None | 0.068140756 | 0.003718771 | 0.003748416 | 0.073603355 |
| cache/uint8/sequence | 0.013675471 | 0.001364959 | 0.001422090 | 0.013708414 |
| cache/float64/None | 0.071589994 | 0.007773934 | 0.007951402 | 0.078497844 |
| cache/float64/sequence | 0.013336063 | 0.001373182 | 0.001384799 | 0.013521429 |
| large/uint8/None | 2.236689000 | 0.050763359 | 0.051437621 | 1.989004667 |
| large/uint8/sequence | 0.013490753 | 0.001357168 | 0.001369426 | 0.013330888 |
| large/float64/None | 2.423510500 | 0.172028694 | 0.172140033 | 1.927373333 |
| large/float64/sequence | 0.013395760 | 0.001345961 | 0.001393852 | 0.013392998 |

Across all 73 cases there are **35 direct losses and 28 normalized losses**, all outside the octahedron target. The 42-case union follows; **bold ratios are losses**. Latencies are the geometric means of the two report medians per role. Added milliseconds are candidate minus baseline; a negative value means direct time improved despite a normalized loss.

| Case | Normalized gain | Direct gain | Baseline ms | Candidate ms | Added ms |
|---|---:|---:|---:|---:|---:|
| ellipse/tiny/uint8/None | **0.973178×** | 1.012915× | 0.002776211 | 0.002740813 | -0.000035398 |
| ellipse/tiny/uint8/crosses | **0.962498×** | **0.983017×** | 0.013576383 | 0.013810937 | +0.000234554 |
| ellipse/tiny/float64/None | **0.981435×** | 1.022474× | 0.002856216 | 0.002793436 | -0.000062780 |
| ellipse/tiny/float64/crosses | **0.991040×** | 1.000325× | 0.013425655 | 0.013421289 | -0.000004367 |
| ellipse/cache/uint8/crosses | 1.025949× | **0.997575×** | 0.077095397 | 0.077282844 | +0.000187447 |
| ellipse/cache/float64/None | **0.979031×** | **0.962836×** | 0.051780417 | 0.053779041 | +0.001998624 |
| ellipse/large/uint8/None | 1.014812× | **0.988317×** | 3.007239068 | 3.042786859 | +0.035547791 |
| ellipse/large/float64/None | 1.009236× | **0.993697×** | 3.063367747 | 3.082798632 | +0.019430885 |
| octagon/tiny/uint8/None | 1.029957× | **0.981434×** | 0.003706129 | 0.003776241 | +0.000070111 |
| octagon/tiny/uint8/sequence | **0.967372×** | **0.940146×** | 0.004237427 | 0.004507202 | +0.000269775 |
| octagon/tiny/float64/None | 1.022451× | **0.991521×** | 0.003763485 | 0.003795670 | +0.000032185 |
| octagon/tiny/float64/sequence | **0.975021×** | **0.974560×** | 0.004329437 | 0.004442454 | +0.000113017 |
| octagon/cache/uint8/None | **0.990940×** | **0.998579×** | 0.175939087 | 0.176189422 | +0.000250336 |
| octagon/cache/uint8/sequence | **0.989116×** | **0.992444×** | 0.004282028 | 0.004314630 | +0.000032601 |
| octagon/cache/float64/None | 1.010355× | **0.998411×** | 0.176480880 | 0.176761672 | +0.000280792 |
| octagon/cache/float64/sequence | 1.046803× | **0.964092×** | 0.004243418 | 0.004401465 | +0.000158047 |
| octagon/large/uint8/None | **0.991345×** | 1.002288× | 10.439280317 | 10.415450217 | -0.023830099 |
| octagon/large/uint8/sequence | **0.996343×** | **0.981055×** | 0.004282658 | 0.004365359 | +0.000082700 |
| octagon/large/float64/None | **0.997031×** | **0.993688×** | 10.599792191 | 10.667122191 | +0.067330001 |
| octagon/large/float64/sequence | **0.996241×** | **0.983675×** | 0.004349440 | 0.004421624 | +0.000072184 |
| star/tiny/uint8/None | 1.008066× | **0.979401×** | 0.005898967 | 0.006023034 | +0.000124067 |
| star/tiny/float64/None | **0.996066×** | **0.971623×** | 0.005808311 | 0.005977950 | +0.000169639 |
| star/cache/float64/None | 1.010847× | **0.986947×** | 0.132653947 | 0.134408361 | +0.001754413 |
| star/large/uint8/None | **0.998316×** | **0.979929×** | 8.141412138 | 8.308164493 | +0.166752355 |
| footprint_from_sequence/tiny/C | **0.981790×** | **0.987654×** | 0.018469989 | 0.018700865 | +0.000230876 |
| footprint_from_sequence/tiny/F | **0.991082×** | **0.986544×** | 0.018574931 | 0.018828276 | +0.000253345 |
| footprint_from_sequence/tiny/reverse | **0.991615×** | **0.973779×** | 0.018444003 | 0.018940637 | +0.000496634 |
| octagon/tiny/even-sequence-warning | **0.997339×** | **0.986684×** | 0.007851315 | 0.007957271 | +0.000105957 |
| IC / memory | 1.000529× | **0.999362×** | 0.203132332 | 0.203261969 | +0.000129637 |
| IC / volume | 1.027739× | **0.999595×** | 0.053111360 | 0.053132853 | +0.000021493 |
| IC / stepped | **0.896463×** | **0.888006×** | 0.501388916 | 0.564623208 | +0.063234292 |
| IC / masked | **0.961135×** | **0.955967×** | 0.248974981 | 0.260443046 | +0.011468065 |
| MCC / tiny | 1.011415× | **0.980711×** | 0.020638652 | 0.021044577 | +0.000405925 |
| MCC / memory | **0.933065×** | **0.986732×** | 10.502691395 | 10.643916240 | +0.141224845 |
| MCC / masked | **0.937473×** | **0.952360×** | 0.651389009 | 0.683973341 | +0.032584333 |
| MOC / tiny | **0.986668×** | 1.005253× | 0.042044075 | 0.041824388 | -0.000219688 |
| MOC / cache | **0.988116×** | **0.998956×** | 1.415290570 | 1.416769243 | +0.001478673 |
| MOC / volume | 1.040582× | **0.988772×** | 5.827160080 | 5.893327800 | +0.066167720 |
| MOC / stepped | **0.972736×** | **0.953760×** | 1.417797866 | 1.486535649 | +0.068737783 |
| MOC / masked | 1.013861× | **0.984459×** | 1.195766771 | 1.214643265 | +0.018876494 |
| IC / cache-F | **0.995824×** | 1.010629× | 0.018693504 | 0.018496906 | -0.000196598 |
| MCC / cache-F | **0.999970×** | 1.016727× | 0.634786614 | 0.624343340 | -0.010443274 |

All 52 footprint workloads beat the original in both candidate reports. The same five intersection workloads do too; **16 colocalization workloads remain slower** in each candidate report. Their absolute penalties against the original are retained here, using balanced candidate-role medians. IC is intersection coefficient, MCC Manders colocalization coefficient, and MOC Manders overlap coefficient.

| Original-library regression | Original ms | Candidate ms | Added ms | Candidate2 original/current | Candidate3 original/current |
|---|---:|---:|---:|---:|---:|
| IC / stepped | 0.147914297 | 0.564623208 | +0.416708911 | 0.262227× | 0.261713× |
| IC / masked | 0.199858292 | 0.260443046 | +0.060584755 | 0.776352× | 0.758508× |
| MCC / tiny | 0.008191979 | 0.021044577 | +0.012852598 | 0.386713× | 0.391840× |
| MCC / cache | 0.036322006 | 0.621707740 | +0.585385733 | 0.058289× | 0.058557× |
| MCC / memory | 0.781642740 | 10.643916240 | +9.862273500 | 0.071742× | 0.075170× |
| MCC / volume | 0.199802707 | 2.715586333 | +2.515783626 | 0.071276× | 0.075951× |
| MCC / stepped | 0.088151611 | 0.632841014 | +0.544689403 | 0.142857× | 0.135822× |
| MCC / masked | 0.206402022 | 0.683973341 | +0.477571320 | 0.299241× | 0.304319× |
| MOC / tiny | 0.011765933 | 0.041824388 | +0.030058455 | 0.280596× | 0.282041× |
| MOC / cache | 0.055051612 | 1.416769243 | +1.361717631 | 0.038727× | 0.038988× |
| MOC / memory | 1.563534059 | 24.145716378 | +22.582182319 | 0.060959× | 0.068786× |
| MOC / volume | 0.307588509 | 5.893327800 | +5.585739291 | 0.051761× | 0.052628× |
| MOC / stepped | 0.114650704 | 1.486535649 | +1.371884945 | 0.072985× | 0.081502× |
| MOC / masked | 0.225141509 | 1.214643265 | +0.989501756 | 0.185671× | 0.185042× |
| MCC / cache-F | 0.036586960 | 0.624343340 | +0.587756380 | 0.057700× | 0.059516× |
| MOC / cache-F | 0.054491157 | 1.394976373 | +1.340485216 | 0.038677× | 0.039452× |

Baseline preparation [40277634220](https://github.com/bipxx/scikit-imagexx/actions/runs/40277634220) used checked source `5b03eded95b86c22de7cea6c76243b1748969a64` and compiled producer `c53198aff0a502fdfe236c6b46834bbc58547abd`. Candidate producer [40285568194](https://github.com/bipxx/scikit-imagexx/actions/runs/40285568194) and preparation [40286601199](https://github.com/bipxx/scikit-imagexx/actions/runs/40286601199) used exact `9da7ccf`. Controller revision was `5d37e10aebe0f79c54a6a81903a797988e7c2301`.

Private component checks passed 254 public and 2,997 native controls in both normal/scalar modes. Candidate installed preparation then passed **799 public checks per mode**, **1,318 smoke checks**, **21 metadata tests plus 30 subtests**, and **73 corpus checks per mode**. The baseline retains its historical 785 count. Separate retained reviews verify source/wheel/core/runtime identity and all before/after checks. The full frozen cases, dtype/complex/provider frontiers and reviewed live-reference policy remain unchanged; no tolerance or failed historical result is relabelled.

Candidate wheel SHA256 `ff2ed94672d40a0f44c67d21b428a688dd332a4f333a6c9aa70981db5d48dd5b`; core SHA256 `2cde9e731d28030faa06c0d173f86ab169dff3c4db4320a3cfba7adfa0ed64ff`. The independent metric review is `build/phase3-octahedron-integer-independent-review-v1/review.json` (SHA256 `c5307c562b4181ba46561112451b2b6b3a2c5e69edde2e7ac0df30639522e0b8`). Prepared identity review SHA256 is `f22c1f2c80f4e1436aa73c0853c2ae080098873b7210e88395c92723997c9084`; final source/runtime review SHA256 is `8a5e73442b1f763808d6f96fd8b6ffa9078f0f72831820d3a3efb0834f220503`.

[Exact raw artifact ZIP](phase3-octahedron-integer-same-host.zip), **135,709 bytes**, SHA256 `96dab3a124ac69b9e532298c0c4ba4c0ac3226a4231a24a3379783a9edec53a7`, contains all four full reports, raw samples, logs, identities and archive selections. It was retained once; no large baseline archive was downloaded or rehashed for this review.

Kept colocalization v5 is unchanged. [Packed float32 v6](phase3-f32-packed-same-host.md) remains rejected and unmerged. The separate byte-affine candidate is still awaiting installed gates and measurement.
