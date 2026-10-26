# Finite float32 scan: kept after the fixed metric improved

**KEEP; merged in main `552d445d1f574f359915205559a9e5d6a68e362d`.** The eight-line dense native-float32 finite scan improves the fixed 21-case colocalization metric. Both baseline and candidate already contain the kept integer-octahedron change; this is a comparison of the combined products, with no later integration extrapolation. Packed float32 v6 remains rejected and byte-affine `887aff6` remains held and unmerged.

[Controlled run 40295836642](https://github.com/bipxx/scikit-imagexx/actions/runs/40295836642) ran baseline–candidate–candidate–baseline on one **Intel Xeon 6973P-C** host, using CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. All 73 workloads, one worker and seven paired rounds are unchanged. The four reports contain **292 exact untimed parity checks and 4,088 raw timing samples**. All dependencies were installed offline before measurement; the measurement host ran no builds or pytest.

Balanced fixed21 original/current scores are **0.3476071029603× → 0.4117914024166×**. The candidate gains **1.253969940× directly / 1.184646110× normalized**. The ten targeted float32 Manders cases gain **1.513747596× directly / 1.426518299× normalized**; both candidate medians beat both baseline medians for every target. All 14 Manders cases gain **1.349543295× directly / 1.287415999× normalized**.

The same five intersection workloads beat the original in both candidate passes; **16 of the 21 colocalization workloads remain slower than the original**. Fixed21 has three direct and six normalized losses, including the non-target float64 MOC volume case (0.976377× direct / 0.957668× normalized). Across all 73 workloads there are **35 direct and 39 normalized losses**, retained below. The unchanged footprint group has 1.002062× direct aggregate gain.

## Source, products and correctness gates

| Identity | Baseline | Candidate |
|---|---|---|
| Source and compiled producer | `9da7ccfd0f38902bc231de9021732861b757df2b` | `552d445d1f574f359915205559a9e5d6a68e362d` |
| Preparation run | [40286601199](https://github.com/bipxx/scikit-imagexx/actions/runs/40286601199) | [40295473176](https://github.com/bipxx/scikit-imagexx/actions/runs/40295473176) |
| Prepared artifact | `12634621268` | `12638490196` |
| Prepared ZIP SHA-256 | `f22ad0f4be4aeb8aff8e588821845e0978ea30fa2d5336ff6f8d9f915eaf31ad` | `d90673c4edffc8fab05f03f8ba3ac697a998427c42bd776f604aca236eaed284` |
| Wheel SHA-256 | `ff2ed94672d40a0f44c67d21b428a688dd332a4f333a6c9aa70981db5d48dd5b` | `87581a5592252f32a7f6fd9d765ffcb8944a1f245ec82bc653ba3cdde80a0c69` |
| Core SHA-256 | `2cde9e731d28030faa06c0d173f86ab169dff3c4db4320a3cfba7adfa0ed64ff` | `390697922eb1513c88d3e3fe6a267716949276795a9c68bfceff8b5862529400` |
| Product-source fingerprint | `657daea3a6ca1cba05241b77f511862c7095ad682355150ae721a512234849e3` | `2e65e51cd4578c21b2544677654310c8b46318835ad3ea30e9e828b5b69e30af` |

Both preparations passed **799 installed public checks in normal and scalar modes**, 1,318 registry smoke checks, 21 metadata tests plus 30 subtests, and 73 exact corpus checks per mode. The baseline and candidate bounded native executables each passed **20,975 controls in normal and scalar modes**, preserving the old 2,727-control body exactly. New controls cover finite/nonfinite IEEE words, zero/subnormal/NaN/infinity behavior, vector tails, unaligned/swapped/reversed/stepped layouts, prewrite canaries and floating-point-environment preservation. The full fresh-original public policy is unchanged; historical frozen-word failures remain retained.

The new path uses integer exponent masks for a validated, native-endian, affine float32 view with four-byte stride. It leaves the generic loop, provider boundaries, floating-point reductions and worker policy unchanged. Retained compiler output shows baseline **SSE2 integer loads/masks/comparisons/OR**, with bounded scalar tails; this is not a claim of a new AVX2 leaf. Optional AVX2 dispatch is tested separately by the existing normal/scalar controls.

Four before/after identity checks authenticate the selected source, core, Python files and runtime. The 14 dependency wheels and seven benchmark/factory helpers are identical across products. NumPy keeps divide/over/invalid=`warn`, under=`ignore`, buffer size 8,192; every reported BLAS/OpenMP pool has one thread. The host is Linux 6.17.0-1022-azure x86_64/glibc 2.39; the exact runtime and CPU feature maps remain in each raw report. Candidate CI producer is [40294220904](https://github.com/bipxx/scikit-imagexx/actions/runs/40294220904).

## All 21 colocalization cases and the ten targets

Each latency is the median of seven raw per-call samples, including public allocations. B1 and B4 are baseline passes; C2 and C3 are candidate passes. `f32 target` identifies the ten prospective targets. Direct gain is `GM(B1, B4) / GM(C2, C3)`. Normalized gain compares the corresponding geometric means of original/current ratios. Greater than one is better. No cross-CPU cumulative factor is used.

| Case | Scope | B1 ms | C2 ms | C3 ms | B4 ms | Direct gain | Normalized gain |
|---|---|---:|---:|---:|---:|---:|---:|
| measure.intersection_coeff/future/tiny | fixed non-target | 0.008889137 | 0.007919872 | 0.007821681 | 0.007875046 | 1.063034× | 0.972721× |
| measure.intersection_coeff/future/cache | fixed non-target | 0.017587868 | 0.017654779 | 0.017502734 | 0.017553060 | 0.999537× | 0.967462× |
| measure.intersection_coeff/future/memory | fixed non-target | 0.301289923 | 0.269117265 | 0.269020829 | 0.269011025 | 1.058068× | 0.990989× |
| measure.intersection_coeff/future/volume | fixed non-target | 0.050482927 | 0.046467133 | 0.050579058 | 0.046728697 | 1.001857× | 0.890688× |
| measure.intersection_coeff/future/stepped | fixed non-target | 0.742261182 | 0.647083100 | 0.644922550 | 0.737693136 | 1.145467× | 1.002171× |
| measure.intersection_coeff/future/masked | fixed non-target | 0.353180771 | 0.270722725 | 0.271819209 | 0.391984946 | 1.371610× | 1.238891× |
| measure.manders_coloc_coeff/future/tiny | f32 target | 0.023266401 | 0.017265038 | 0.017145655 | 0.023154943 | 1.349043× | 1.197886× |
| measure.manders_coloc_coeff/future/cache | f32 target | 0.564053632 | 0.290131644 | 0.288465766 | 0.566288833 | 1.953595× | 1.754825× |
| measure.manders_coloc_coeff/future/memory | fixed non-target | 13.207109000 | 12.903805000 | 11.761440000 | 12.378354000 | 1.037878× | 1.012258× |
| measure.manders_coloc_coeff/future/volume | fixed non-target | 2.372868200 | 2.328683400 | 2.505559500 | 2.635113600 | 1.035211× | 1.000767× |
| measure.manders_coloc_coeff/future/stepped | f32 target | 0.566684407 | 0.448538774 | 0.473214769 | 0.569717000 | 1.233307× | 1.090881× |
| measure.manders_coloc_coeff/future/masked | f32 target | 0.651051632 | 0.466943172 | 0.464014536 | 0.730944895 | 1.482014× | 1.394467× |
| measure.manders_overlap_coeff/future/tiny | f32 target | 0.042994394 | 0.034538716 | 0.034373221 | 0.047779000 | 1.315410× | 1.231477× |
| measure.manders_overlap_coeff/future/cache | f32 target | 1.090250692 | 0.616755810 | 0.617216045 | 1.081182818 | 1.759696× | 1.760251× |
| measure.manders_overlap_coeff/future/memory | fixed non-target | 33.290636000 | 32.519508000 | 33.072012000 | 32.498161000 | 1.002970× | 1.014967× |
| measure.manders_overlap_coeff/future/volume | fixed non-target | 5.846947500 | 5.935245000 | 5.914581000 | 5.723592000 | 0.976377× | 0.957668× |
| measure.manders_overlap_coeff/future/stepped | f32 target | 1.122735083 | 0.947817538 | 0.946839308 | 1.114418727 | 1.180762× | 1.131797× |
| measure.manders_overlap_coeff/future/masked | f32 target | 1.063551100 | 0.715686333 | 0.708441176 | 1.136130923 | 1.543761× | 1.499415× |
| measure.intersection_coeff/future/cache-F | fixed non-target | 0.017647749 | 0.018893198 | 0.017877615 | 0.018577892 | 0.985225× | 0.990615× |
| measure.manders_coloc_coeff/future/cache-F | f32 target | 0.494843077 | 0.294446375 | 0.289292143 | 0.565424609 | 1.812382× | 1.716440× |
| measure.manders_overlap_coeff/future/cache-F | f32 target | 1.080749923 | 0.701840158 | 0.619134045 | 1.182546727 | 1.714985× | 1.728299× |

Original reference medians also drifted: candidate-pass references were 0.944717× the baseline-pass references in aggregate. Both direct and normalized outcomes are therefore reported. This comparison establishes the measured public-call change on this host; it does not isolate a universal causal kernel speedup.

## Every workload, loss and absolute penalty

`D` marks a direct regression, `N` a normalized regression. `Added vs baseline` is `GM(C2, C3) − GM(B1, B4)` in milliseconds. `Added vs original` is the candidate geometric-mean latency minus its paired original geometric-mean latency. Positive values are penalties; the final column gives original/current ratios for C2/C3. All 73 rows are retained, including the 35 D and 39 N rows.

| Case | Direct gain | Normalized gain | Added vs baseline ms | Added vs original ms | Loss | Original/current C2 / C3 |
|---|---:|---:|---:|---:|---|---:|
| morphology.ellipse/tiny/uint8/None | 1.100381× | 1.056472× | -0.000298341 | -0.030921504 | — | 11.361137× / 11.447061× |
| morphology.ellipse/tiny/uint8/crosses | 0.928952× | 0.917074× | +0.001194298 | -0.026958073 | D,N | 2.590000× / 2.617502× |
| morphology.ellipse/tiny/float64/None | 0.954942× | 0.965900× | +0.000135985 | -0.030447946 | D,N | 11.287497× / 10.893571× |
| morphology.ellipse/tiny/float64/crosses | 0.994786× | 0.976829× | +0.000081437 | -0.025606828 | D,N | 2.663947× / 2.615383× |
| morphology.ellipse/cache/uint8/None | 1.006167× | 0.991034× | -0.000296645 | -0.091237233 | N | 2.897307× / 2.896463× |
| morphology.ellipse/cache/uint8/crosses | 0.997236× | 1.000704× | +0.000222652 | -0.091515452 | D | 2.141976× / 2.129890× |
| morphology.ellipse/cache/float64/None | 0.993290× | 1.004485× | +0.000325582 | -0.092124369 | D | 2.899072× / 2.898336× |
| morphology.ellipse/cache/float64/crosses | 0.997599× | 0.986977× | +0.000194641 | -0.093104809 | D,N | 2.145801× / 2.151553× |
| morphology.ellipse/large/uint8/None | 1.068411× | 1.007559× | -0.193495322 | -6.481141733 | — | 3.255010× / 3.328245× |
| morphology.ellipse/large/uint8/crosses | 0.999926× | 0.963824× | +0.000231848 | -6.510496164 | D,N | 3.058205× / 3.124096× |
| morphology.ellipse/large/float64/None | 1.008283× | 0.988557× | -0.025375730 | -6.707318851 | N | 3.181709× / 3.197214× |
| morphology.ellipse/large/float64/crosses | 0.929258× | 0.930924× | +0.250202583 | -6.876427000 | D,N | 3.036048× / 2.855175× |
| morphology.octagon/tiny/uint8/None | 0.963369× | 0.970284× | +0.000169187 | -0.061020934 | D,N | 13.937031× / 14.491967× |
| morphology.octagon/tiny/uint8/sequence | 1.074358× | 1.032004× | -0.000357176 | -0.010996771 | — | 3.288786× / 3.289885× |
| morphology.octagon/tiny/float64/None | 1.092721× | 1.031847× | -0.000398704 | -0.056699736 | — | 14.272407× / 14.099714× |
| morphology.octagon/tiny/float64/sequence | 1.090436× | 0.951108× | -0.000434096 | -0.011028801 | N | 3.307956× / 3.287369× |
| morphology.octagon/cache/uint8/None | 1.002494× | 0.997550× | -0.000845172 | -0.100953070 | N | 1.290403× / 1.305466× |
| morphology.octagon/cache/uint8/sequence | 0.974828× | 1.003218× | +0.000128098 | -0.011717202 | D | 3.319683× / 3.285451× |
| morphology.octagon/cache/float64/None | 0.988465× | 1.002424× | +0.003940399 | -0.104694057 | D | 1.313909× / 1.299066× |
| morphology.octagon/cache/float64/sequence | 1.012410× | 0.995956× | -0.000060354 | -0.010922100 | N | 3.239028× / 3.252427× |
| morphology.octagon/large/uint8/None | 0.994652× | 0.975504× | +0.117509124 | -3.589071656 | D,N | 1.164530× / 1.162164× |
| morphology.octagon/large/uint8/sequence | 0.996225× | 1.017077× | +0.000018144 | -0.011315352 | D | 3.307770× / 3.401177× |
| morphology.octagon/large/float64/None | 0.992843× | 0.987268× | +0.162969810 | -3.441508108 | D,N | 1.152042× / 1.150240× |
| morphology.octagon/large/float64/sequence | 1.001109× | 0.991976× | -0.000005429 | -0.010991647 | N | 3.242072× / 3.249774× |
| morphology.octahedron/tiny/uint8/None | 1.305250× | 1.188277× | -0.000366642 | -0.016410770 | — | 14.179688× / 15.162492× |
| morphology.octahedron/tiny/uint8/sequence | 0.982537× | 1.057061× | +0.000023750 | -0.017423992 | D | 13.241404× / 14.406914× |
| morphology.octahedron/tiny/float64/None | 0.913884× | 0.958428× | +0.000111687 | -0.017028504 | D,N | 13.974195× / 14.287097× |
| morphology.octahedron/tiny/float64/sequence | 0.956628× | 1.012960× | +0.000058017 | -0.017070088 | D | 13.506661× / 14.020076× |
| morphology.octahedron/cache/uint8/None | 1.024028× | 1.003832× | -0.000100247 | -0.116164957 | — | 28.926262× / 28.761444× |
| morphology.octahedron/cache/uint8/sequence | 1.075374× | 0.999388× | -0.000094777 | -0.016014813 | N | 13.567954× / 13.906513× |
| morphology.octahedron/cache/float64/None | 1.026437× | 0.983695× | -0.000238454 | -0.115914743 | N | 13.816964× / 13.885734× |
| morphology.octahedron/cache/float64/sequence | 1.026255× | 1.018593× | -0.000032492 | -0.015999308 | — | 13.817052× / 14.040051× |
| morphology.octahedron/large/uint8/None | 1.015498× | 0.948827× | -0.001349390 | -9.555152292 | N | 111.055723× / 110.428859× |
| morphology.octahedron/large/uint8/sequence | 0.990948× | 0.990948× | +0.000011679 | -0.016260269 | D,N | 14.051137× / 13.168747× |
| morphology.octahedron/large/float64/None | 0.976275× | 0.980528× | +0.008842294 | -9.956115096 | D,N | 27.385821× / 28.045356× |
| morphology.octahedron/large/float64/sequence | 1.066241× | 1.012793× | -0.000084012 | -0.016152892 | — | 13.994080× / 13.482835× |
| morphology.star/tiny/uint8/None | 1.057558× | 0.983731× | -0.000426975 | -0.054228571 | N | 8.255934× / 8.364810× |
| morphology.star/tiny/float64/None | 1.083689× | 1.102894× | -0.000644199 | -0.056246385 | — | 8.345485× / 8.268883× |
| morphology.star/cache/uint8/None | 0.954369× | 1.009110× | +0.010861194 | -0.112622365 | D | 1.462461× / 1.483927× |
| morphology.star/cache/float64/None | 0.942357× | 0.975080× | +0.013777174 | -0.109792664 | D,N | 1.455783× / 1.462953× |
| morphology.star/large/uint8/None | 0.952925× | 0.983025× | +0.806972282 | -2.942714599 | D,N | 1.167031× / 1.176317× |
| morphology.star/large/float64/None | 0.995016× | 0.984270× | +0.089543170 | -2.938440242 | D,N | 1.173577× / 1.153629× |
| morphology.footprint_from_sequence/tiny/C | 0.929887× | 0.972663× | +0.001816165 | -0.040214993 | D,N | 2.576709× / 2.528499× |
| morphology.footprint_from_sequence/tiny/F | 0.921552× | 0.986180× | +0.002076822 | -0.040779442 | D,N | 2.591999× / 2.489778× |
| morphology.footprint_from_sequence/tiny/reverse | 0.955625× | 1.014230× | +0.001168025 | -0.040034019 | D | 2.555439× / 2.486914× |
| morphology.footprint_from_sequence/cache/C | 0.963558× | 0.922984× | +0.049632365 | -5.343636960 | D,N | 5.035824× / 4.813727× |
| morphology.footprint_from_sequence/cache/F | 0.998705× | 1.004314× | +0.001713053 | -5.434100284 | D | 5.094815× / 5.118119× |
| morphology.footprint_from_sequence/cache/reverse | 1.040764× | 0.972933× | -0.055333899 | -5.343339100 | N | 4.865432× / 5.008388× |
| morphology.footprint_from_sequence/large/C | 0.945408× | 0.954278× | +0.308959858 | -44.700146528 | D,N | 8.948321× / 8.848552× |
| morphology.footprint_from_sequence/large/F | 0.942506× | 0.983843× | +0.326476079 | -45.939565421 | D,N | 8.992787× / 9.188588× |
| morphology.footprint_from_sequence/large/reverse | 0.999606× | 0.961496× | +0.002099792 | -42.604966932 | D,N | 8.867460× / 9.109414× |
| morphology.octagon/tiny/even-sequence-warning | 0.996817× | 0.993102× | +0.000028798 | -0.009664167 | D,N | 2.045276× / 2.091540× |
| measure.intersection_coeff/future/tiny | 1.063034× | 0.972721× | -0.000496118 | -0.005086590 | N | 1.654025× / 1.638562× |
| measure.intersection_coeff/future/cache | 0.999537× | 0.967462× | +0.000008137 | -0.017423395 | D,N | 1.987012× / 1.995339× |
| measure.intersection_coeff/future/memory | 1.058068× | 0.990989× | -0.015624321 | -0.455832603 | N | 2.625901× / 2.764091× |
| measure.intersection_coeff/future/volume | 1.001857× | 0.890688× | -0.000090033 | -0.055452641 | N | 2.232922× / 2.058305× |
| measure.intersection_coeff/future/stepped | 1.145467× | 1.002171× | -0.093971712 | +0.463933741 | — | 0.280864× / 0.282817× |
| measure.intersection_coeff/future/masked | 1.371610× | 1.238891× | -0.100806925 | +0.004126922 | — | 0.981413× / 0.988172× |
| measure.manders_coloc_coeff/future/tiny | 1.349043× | 1.197886× | -0.006005362 | +0.006863960 | — | 0.599792× / 0.602319× |
| measure.manders_coloc_coeff/future/cache | 1.953595× | 1.754825× | -0.275872621 | +0.239182804 | — | 0.172957× / 0.173501× |
| measure.manders_coloc_coeff/future/memory | 1.037878× | 1.012258× | -0.466630248 | +10.246228505 | — | 0.166272× / 0.170321× |
| measure.manders_coloc_coeff/future/volume | 1.035211× | 1.000767× | -0.085052376 | +1.991705777 | — | 0.181687× / 0.169425× |
| measure.manders_coloc_coeff/future/stepped | 1.233307× | 1.090881× | -0.107487087 | +0.344486386 | — | 0.258006× / 0.246668× |
| measure.manders_coloc_coeff/future/masked | 1.482014× | 1.394467× | -0.224366088 | +0.179294354 | — | 0.615917× / 0.613716× |
| measure.manders_overlap_coeff/future/tiny | 1.315410× | 1.231477× | -0.010867736 | +0.019739973 | — | 0.430508× / 0.423707× |
| measure.manders_overlap_coeff/future/cache | 1.759696× | 1.760251× | -0.468721404 | +0.544479407 | — | 0.117279× / 0.117756× |
| measure.manders_overlap_coeff/future/memory | 1.002970× | 1.014967× | -0.097415438 | +28.348851665 | — | 0.132687× / 0.138502× |
| measure.manders_overlap_coeff/future/volume | 0.976377× | 0.957668× | +0.139963029 | +5.020239658 | D,N | 0.152511× / 0.152866× |
| measure.manders_overlap_coeff/future/stepped | 1.180762× | 1.131797× | -0.171240880 | +0.774206932 | — | 0.182158× / 0.183337× |
| measure.manders_overlap_coeff/future/masked | 1.543761× | 1.499415× | -0.387187605 | +0.399577204 | — | 0.435064× / 0.442647× |
| measure.intersection_coeff/future/cache-F | 0.985225× | 0.990615× | +0.000271544 | -0.016967808 | D,N | 1.875262× / 1.972462× |
| measure.manders_coloc_coeff/future/cache-F | 1.812382× | 1.716440× | -0.237100012 | +0.241326357 | — | 0.171952× / 0.174331× |
| measure.manders_overlap_coeff/future/cache-F | 1.714985× | 1.728299× | -0.471311837 | +0.582424308 | — | 0.115921× / 0.116994× |

## Raw evidence and limits

[Exact compact raw archive](phase3-finite-scan-same-host.zip): artifact `12638740866`, 135,299 bytes, 30 members, SHA-256 `242886b314eea30ed01692867c7dbdabe629223702107db24ff9267ec2e9a7c5`. It is copied byte-for-byte, not repackaged. The workflow revision is `3f1f1c38a7636d0f4f1a0edbecd9b16a938025de`; the four report paths are `1-baseline`, `2-candidate`, `3-candidate` and `4-baseline`.

Local retained review: `build/phase3-finite-scan-same-host-v1/review.json`, SHA-256 `5ff2c16e621eca3065ed345f8f37e04db2cd8378f6268da9c78b322d42bbe339`. Both prepared archives and their full raw checks remain protected. The completed, clean ancestral finite-scan worktree was removed only after its entire private proof was moved unchanged to `build/phase3-finite-scan-private-v1`; handoff SHA-256 remains `d2b590bab800354585e28253fe649b81daaad34bcecc8a44d30be93b1a3925ea`. Other unfinished worktrees are retained.

This is one minimum-Linux, normal-dispatch, one-host ABBA result. Scalar correctness passed separately; no scalar timing, modern/Windows speedup, cumulative cross-CPU gain or new whole-product acceptance is claimed. Public calls still include declared NumPy provider stages. Main source coverage remains **211 native-partial image APIs + 3 metadata functions**, with accepted combined binary coverage still **203 image APIs**. The separate Butterworth branch has 700 passing component checks, a privately tested FFT and a NumPy mask provider; its whole-Release gate and six timings remain pending, and its 212+3 branch registry is not main coverage.
