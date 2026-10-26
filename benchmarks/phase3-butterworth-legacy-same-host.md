# Legacy workloads after Butterworth integration: flat aggregate, retained regressions

[Controlled ABBA run 40307547940](https://github.com/bipxx/scikit-imagexx/actions/runs/40307547940) compares the kept finite-scan product `552d445` with the Butterworth-integrated product compiled from `936481f`, checked under workflow/source `fff18b7`. It reuses both existing prepared wheels. This is the **unchanged 73-workload legacy corpus**: 52 footprint and 21 colocalization cases. **It contains no Butterworth workload.** The separate [six-case Butterworth first result](butterworth-minimum-initial.md) remains independent.

Observed aggregate performance is flat: **1.000748117× direct / 0.999452166× normalized** over all 73 cases. This is not a blanket regression pass: **44 direct and 46 normalized losses** remain visible. Several intersection losses recur in both candidate passes, and all absolute deltas are published below. The source of these differences is undetermined; this result does not establish an FFT, cache, code-layout or CPU mechanism.

One **AMD EPYC 7763** host ran baseline–candidate–candidate–baseline, one worker and seven paired rounds per case. Each role used a separate process and offline-installed environment. The review verifies four product identities before and after measurement, all **292 exact untimed parity checks** and **4,088 raw samples**, with the same seven harness/factory helpers and 14 dependency wheels. There were no builds or pytest runs on the measurement host.

## Aggregate diagnostics

| Group | Cases | Direct gain | Normalized gain | Direct losses | Normalized losses |
|---|---:|---:|---:|---:|---:|
| All legacy workloads | 73 | 1.000748117× | 0.999452166× | 44 | 46 |
| Colocalization | 21 | 1.007846502× | 1.023131133× | 10 | 8 |
| Footprints | 52 | 0.997895653× | 0.990045574× | 34 | 38 |

The fixed21 original/current score moves from 0.2889927095493× to 0.2956774383115×. All 52 footprint cases and the same five intersection cases beat the original in both candidate passes; **16 colocalization cases remain slower**. The fixed21 reference latency itself changes by a factor of 1.015165634 between the baseline and candidate passes, so direct and reference-normalized results are reported together. No cumulative gain is calculated from earlier runs on different CPUs.

## Intersection regressions repeated within this run

For each of these three cases, both candidate medians are slower than both baseline medians. The table establishes that observation in this one ABBA run, not a universal regression rate or a causal explanation.

| Case | B1 ms | C2 ms | C3 ms | B4 ms | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| intersection_coeff/stepped | 0.963330077 | 1.048378455 | 1.049876500 | 0.963078250 | 0.918100× | 0.960646× | +0.085923055 |
| intersection_coeff/masked | 0.457162875 | 0.510115714 | 0.511045476 | 0.456113609 | 0.894351× | 0.910370× | +0.053942443 |
| intersection_coeff/cache-F | 0.049900424 | 0.053004423 | 0.059026571 | 0.048240616 | 0.877160× | 0.939176× | +0.006871007 |

The largest absolute increase is `star/large/float64/None`: **+0.537625758 ms**, 0.965341× direct gain, with both candidate medians above both baseline medians. These losses are retained for follow-up; aggregate flatness does not waive them.

## Exact source and preparation identities

| Identity | Baseline | Butterworth-integrated product |
|---|---|---|
| Checked source | `552d445d1f574f359915205559a9e5d6a68e362d` | `fff18b713478be867b36cac47b42d6c1124f4b95` |
| Compiled producer source | `552d445d1f574f359915205559a9e5d6a68e362d` | `936481f1ddcbd2db0f98946ee5a4d788fc861902` |
| Preparation | [40295473176](https://github.com/bipxx/scikit-imagexx/actions/runs/40295473176) | [40306991592](https://github.com/bipxx/scikit-imagexx/actions/runs/40306991592) |
| Prepared artifact | `12638490196` | `12642253052` |
| Prepared ZIP SHA-256 | `d90673c4edffc8fab05f03f8ba3ac697a998427c42bd776f604aca236eaed284` | `aadfb0fe74f990db2b3399cff311be3b1ba88d1971d2da5366f20f45ed9c80d1` |
| Wheel SHA-256 | `87581a5592252f32a7f6fd9d765ffcb8944a1f245ec82bc653ba3cdde80a0c69` | `e1a5a120e25a037a9b90865622cc47ce210ee9eee1ccc59b0871bebdc2524661` |
| Core SHA-256 | `390697922eb1513c88d3e3fe6a267716949276795a9c68bfceff8b5862529400` | `695fd45385f0a24cc0fc0fcc29a12b755d7fb220790324b38d4b7fd2406f32fd` |
| Product-source fingerprint | `2e65e51cd4578c21b2544677654310c8b46318835ad3ea30e9e828b5b69e30af` | `8eadefd45f13759b2b382cff7c7b7126638332eb33f030a9bc574736a40e7cbe` |
| Public checks in each normal/scalar mode | 799 | 974 = 240 footprint + 545 colocalization + 14 octahedron + 175 Butterworth |
| Registry smoke checks | 1,318 | 1,322 |
| Metadata checks | 21 + 30 subtests | 21 + 30 subtests |
| Untimed legacy corpus in each mode | 73 | 73 |

Candidate preparation reuses the genuine [40300864201 producer](https://github.com/bipxx/scikit-imagexx/actions/runs/40300864201); its product-source diff against checked source `fff18b7` is empty. Workflow `fff18b7` adds the 175 existing Butterworth public checks to both preparation modes and updates the derived registry barrier. It changes neither the 73-case corpus nor measurement arithmetic.

Runtime: CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0; Linux x86_64. Every reported BLAS/OpenMP pool is at one thread; NumPy divide/over/invalid policy is `warn`, underflow is `ignore`, and buffer size is 8,192. Normal AVX2 dispatch is recorded; scalar correctness was checked separately without scalar timing. Exact CPU features, runtime libraries, source, core and Python maps remain in each report. Raw wheel/installed-package equality is checked separately from the existing helper’s canonical-LF Python hashes.

Preparation retains 539 genuine fresh-original/native pairs per mode. All nine semantic fields match fresh originals exactly. The candidate preparation reproduces the known 44 frozen MOC word-only differences in both modes under the unchanged reviewed policy; every other frozen field remains exact. Earlier failed attempts and those raw observations are not replaced. The four timed reports also match all 73 public observations across roles.

## All 73 workloads and losses

Each B/C latency below is the geometric mean of the two per-report medians. Direct gain is B/C. Normalized gain is `GM(original/current in C2,C3) / GM(original/current in B1,B4)`. `Added vs baseline` is C−B; `Added vs original` is C minus the geometric mean of its paired original medians. Positive deltas are penalties. `D` and `N` mark direct and normalized losses. All 44 D and 46 N rows are retained.

| Case | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added vs baseline ms | Added vs original ms | Loss |
|---|---:|---:|---:|---:|---:|---:|---|
| morphology.ellipse/tiny/uint8/None | 0.005244768 | 0.005271307 | 0.994965× | 0.975609× | +0.000026539 | -0.055478781 | D,N |
| morphology.ellipse/tiny/uint8/crosses | 0.028146811 | 0.028036032 | 1.003951× | 0.991955× | -0.000110778 | -0.058633293 | N |
| morphology.ellipse/tiny/float64/None | 0.005185515 | 0.005296285 | 0.979085× | 0.952486× | +0.000110771 | -0.055309868 | D,N |
| morphology.ellipse/tiny/float64/crosses | 0.028217228 | 0.028147475 | 1.002478× | 1.023706× | -0.000069753 | -0.060715722 | — |
| morphology.ellipse/cache/uint8/None | 0.084237530 | 0.085356872 | 0.986886× | 0.976898× | +0.001119342 | -0.160143371 | D,N |
| morphology.ellipse/cache/uint8/crosses | 0.141158655 | 0.141572058 | 0.997080× | 0.995969× | +0.000413403 | -0.167823363 | D,N |
| morphology.ellipse/cache/float64/None | 0.078154633 | 0.079427971 | 0.983969× | 0.984805× | +0.001273338 | -0.169199635 | D,N |
| morphology.ellipse/cache/float64/crosses | 0.137904609 | 0.139289301 | 0.990059× | 0.984354× | +0.001384692 | -0.173915890 | D,N |
| morphology.ellipse/large/uint8/None | 5.079554704 | 5.170817243 | 0.982350× | 0.985670× | +0.091262539 | -5.957534225 | D,N |
| morphology.ellipse/large/uint8/crosses | 5.331152945 | 5.368370291 | 0.993067× | 0.999283× | +0.037217346 | -6.107704994 | D,N |
| morphology.ellipse/large/float64/None | 4.734916654 | 4.759290962 | 0.994879× | 0.988124× | +0.024374308 | -6.497004035 | D,N |
| morphology.ellipse/large/float64/crosses | 5.039307148 | 5.098215465 | 0.988445× | 0.981127× | +0.058908317 | -6.568450657 | D,N |
| morphology.octagon/tiny/uint8/None | 0.007017885 | 0.007147256 | 0.981899× | 0.998069× | +0.000129371 | -0.158149843 | D,N |
| morphology.octagon/tiny/uint8/sequence | 0.008736895 | 0.008715235 | 1.002485× | 0.966093× | -0.000021660 | -0.024239097 | N |
| morphology.octagon/tiny/float64/None | 0.006924572 | 0.007075000 | 0.978738× | 0.974556× | +0.000150429 | -0.156855163 | D,N |
| morphology.octagon/tiny/float64/sequence | 0.008689910 | 0.008701040 | 0.998721× | 0.956319× | +0.000011130 | -0.023436894 | D,N |
| morphology.octagon/cache/uint8/None | 0.328020752 | 0.329574223 | 0.995286× | 0.998651× | +0.001553472 | -0.494172455 | D,N |
| morphology.octagon/cache/uint8/sequence | 0.008694908 | 0.008694191 | 1.000083× | 0.969058× | -0.000000718 | -0.023700483 | N |
| morphology.octagon/cache/float64/None | 0.330829273 | 0.332282272 | 0.995627× | 0.994305× | +0.001452998 | -0.491213782 | D,N |
| morphology.octagon/cache/float64/sequence | 0.008624388 | 0.008643235 | 0.997819× | 0.952230× | +0.000018847 | -0.023524177 | D,N |
| morphology.octagon/large/uint8/None | 20.015407443 | 20.208647333 | 0.990438× | 0.987458× | +0.193239889 | -18.898091272 | D,N |
| morphology.octagon/large/uint8/sequence | 0.008739595 | 0.008938046 | 0.977797× | 0.954687× | +0.000198451 | -0.023431427 | D,N |
| morphology.octagon/large/float64/None | 20.245651567 | 20.496771380 | 0.987748× | 0.990030× | +0.251119814 | -19.060841737 | D,N |
| morphology.octagon/large/float64/sequence | 0.008884416 | 0.008583047 | 1.035112× | 1.002035× | -0.000301370 | -0.023407172 | — |
| morphology.octahedron/tiny/uint8/None | 0.002273779 | 0.002293470 | 0.991414× | 0.988705× | +0.000019691 | -0.028127008 | D,N |
| morphology.octahedron/tiny/uint8/sequence | 0.002458465 | 0.002497135 | 0.984514× | 0.977601× | +0.000038670 | -0.027707727 | D,N |
| morphology.octahedron/tiny/float64/None | 0.002453836 | 0.002367766 | 1.036351× | 1.012376× | -0.000086070 | -0.027848274 | — |
| morphology.octahedron/tiny/float64/sequence | 0.002477513 | 0.002489395 | 0.995227× | 0.986650× | +0.000011882 | -0.027718653 | D,N |
| morphology.octahedron/cache/uint8/None | 0.006975519 | 0.007333860 | 0.951139× | 0.934977× | +0.000358341 | -0.183650270 | D,N |
| morphology.octahedron/cache/uint8/sequence | 0.002450204 | 0.002395683 | 1.022758× | 1.012007× | -0.000054521 | -0.027626338 | — |
| morphology.octahedron/cache/float64/None | 0.012537159 | 0.013008113 | 0.963795× | 0.952873× | +0.000470954 | -0.183761086 | D,N |
| morphology.octahedron/cache/float64/sequence | 0.002472882 | 0.002482032 | 0.996313× | 0.990521× | +0.000009150 | -0.027757248 | D,N |
| morphology.octahedron/large/uint8/None | 0.107758997 | 0.107591332 | 1.001558× | 1.002115× | -0.000167666 | -4.378741825 | — |
| morphology.octahedron/large/uint8/sequence | 0.002421726 | 0.002402352 | 1.008065× | 0.998135× | -0.000019375 | -0.027579861 | N |
| morphology.octahedron/large/float64/None | 0.259543830 | 0.262661575 | 0.988130× | 0.995858× | +0.003117746 | -4.309788259 | D,N |
| morphology.octahedron/large/float64/sequence | 0.002462128 | 0.002505432 | 0.982716× | 0.976267× | +0.000043303 | -0.027850343 | D,N |
| morphology.star/tiny/uint8/None | 0.012008201 | 0.012348228 | 0.972464× | 0.975803× | +0.000340027 | -0.151303348 | D,N |
| morphology.star/tiny/float64/None | 0.012161104 | 0.012390630 | 0.981476× | 0.976364× | +0.000229526 | -0.149740126 | D,N |
| morphology.star/cache/uint8/None | 0.243781826 | 0.245559840 | 0.992759× | 0.988408× | +0.001778014 | -0.506015092 | D,N |
| morphology.star/cache/float64/None | 0.244821256 | 0.246383470 | 0.993659× | 0.992871× | +0.001562214 | -0.506888880 | D,N |
| morphology.star/large/uint8/None | 14.663788944 | 14.977508729 | 0.979054× | 0.983897× | +0.313719785 | -20.888541275 | D,N |
| morphology.star/large/float64/None | 14.974179189 | 15.511804946 | 0.965341× | 0.968275× | +0.537625758 | -20.733877039 | D,N |
| morphology.footprint_from_sequence/tiny/C | 0.056936327 | 0.055376930 | 1.028160× | 1.015863× | -0.001559397 | -0.070426837 | — |
| morphology.footprint_from_sequence/tiny/F | 0.056290916 | 0.054851899 | 1.026235× | 1.016775× | -0.001439017 | -0.071576271 | — |
| morphology.footprint_from_sequence/tiny/reverse | 0.055993681 | 0.054557141 | 1.026331× | 1.006828× | -0.001436540 | -0.068216728 | — |
| morphology.footprint_from_sequence/cache/C | 1.970332454 | 1.925409277 | 1.023332× | 1.026911× | -0.044923177 | -7.672134493 | — |
| morphology.footprint_from_sequence/cache/F | 1.972694654 | 1.929851462 | 1.022200× | 1.030055× | -0.042843192 | -7.763458390 | — |
| morphology.footprint_from_sequence/cache/reverse | 1.972782614 | 1.916619334 | 1.029303× | 1.037959× | -0.056163280 | -7.713389073 | — |
| morphology.footprint_from_sequence/large/C | 8.017723984 | 7.723921492 | 1.038038× | 1.034993× | -0.293802492 | -57.560726906 | — |
| morphology.footprint_from_sequence/large/F | 8.004901810 | 7.728166011 | 1.035809× | 1.032442× | -0.276735799 | -57.662981109 | — |
| morphology.footprint_from_sequence/large/reverse | 8.005806629 | 7.790188715 | 1.027678× | 1.028897× | -0.215617914 | -57.708823003 | — |
| morphology.octagon/tiny/even-sequence-warning | 0.015744177 | 0.015772006 | 0.998236× | 0.968690× | +0.000027829 | -0.023121404 | D,N |
| measure.intersection_coeff/future/tiny | 0.013611926 | 0.014124220 | 0.963729× | 0.963329× | +0.000512294 | -0.011227783 | D,N |
| measure.intersection_coeff/future/cache | 0.027546697 | 0.027820000 | 0.990176× | 1.057992× | +0.000273303 | -0.037455373 | D |
| measure.intersection_coeff/future/memory | 0.249276594 | 0.237799135 | 1.048265× | 1.014401× | -0.011477459 | -0.300576091 | — |
| measure.intersection_coeff/future/volume | 0.070501001 | 0.071178238 | 0.990485× | 1.001692× | +0.000677237 | -0.100009105 | D |
| measure.intersection_coeff/future/stepped | 0.963204155 | 1.049127210 | 0.918100× | 0.960646× | +0.085923055 | +0.733157636 | D,N |
| measure.intersection_coeff/future/masked | 0.456637940 | 0.510580384 | 0.894351× | 0.910370× | +0.053942443 | +0.082764416 | D,N |
| measure.manders_coloc_coeff/future/tiny | 0.030026153 | 0.030891666 | 0.971982× | 0.981801× | +0.000865513 | +0.014132293 | D,N |
| measure.manders_coloc_coeff/future/cache | 0.539392470 | 0.538736538 | 1.001218× | 1.017926× | -0.000655932 | +0.467355810 | — |
| measure.manders_coloc_coeff/future/memory | 20.349315970 | 18.425443803 | 1.104414× | 1.114612× | -1.923872167 | +17.203591493 | — |
| measure.manders_coloc_coeff/future/volume | 5.155941500 | 4.646061163 | 1.109745× | 1.109541× | -0.509880337 | +4.331102003 | — |
| measure.manders_coloc_coeff/future/stepped | 1.091373638 | 1.025879798 | 1.063842× | 1.073073× | -0.065493840 | +0.881950395 | — |
| measure.manders_coloc_coeff/future/masked | 0.794257034 | 0.817321599 | 0.971780× | 0.976385× | +0.023064565 | +0.407832436 | D,N |
| measure.manders_overlap_coeff/future/tiny | 0.056561740 | 0.057272630 | 0.987588× | 0.977748× | +0.000710890 | +0.034065548 | D,N |
| measure.manders_overlap_coeff/future/cache | 1.202709777 | 1.189352627 | 1.011231× | 1.033733× | -0.013357150 | +1.074065485 | — |
| measure.manders_overlap_coeff/future/memory | 44.996929430 | 40.170018911 | 1.120162× | 1.185961× | -4.826910520 | +37.756339666 | — |
| measure.manders_overlap_coeff/future/volume | 11.423949328 | 10.210860740 | 1.118804× | 1.117302× | -1.213088588 | +9.672917339 | — |
| measure.manders_overlap_coeff/future/stepped | 2.303760515 | 2.159136899 | 1.066982× | 1.074703× | -0.144623616 | +1.961993082 | — |
| measure.manders_overlap_coeff/future/masked | 1.245065530 | 1.260322929 | 0.987894× | 0.988738× | +0.015257398 | +0.831330747 | D,N |
| measure.intersection_coeff/future/cache-F | 0.049063502 | 0.055934509 | 0.877160× | 0.939176× | +0.006871007 | -0.010970140 | D,N |
| measure.manders_coloc_coeff/future/cache-F | 0.540800901 | 0.536631569 | 1.007769× | 1.010426× | -0.004169332 | +0.465541239 | — |
| measure.manders_overlap_coeff/future/cache-F | 1.202932778 | 1.194259692 | 1.007262× | 1.020518× | -0.008673086 | +1.079824063 | — |

## Raw evidence and scope

[Exact compact raw archive](phase3-butterworth-legacy-same-host.zip): artifact `12643860023`, 134,502 bytes, 30 members, SHA-256 `715356006f7d2c6ecd65ff31d91d34c7d94e29a53ebe3674a68e1b89cbbcc2b0`. This publication is a byte-for-byte copy, not a repackaged archive. All four reports, identity logs, raw samples and selection receipts remain present.

Local review: `build/phase3-butterworth-legacy-same-host-v1/review.json`, SHA-256 `de5cc3a04d4725316813b82e198827842b8f0b90efe4e17307d4e729a0b7dfe4`; absolute-delta handoff SHA-256 `33acc9d5eaad87205b7c938aa9ea743bba73f9b2ed9962d0d494b4c1fa9be636`. Candidate preparation review SHA-256 `b8edd19a57bc2339094a9a81ce411de3d6b7bcf651d04ef2f259037c14b0967a`. Existing prepared archives remain retained without duplicate downloads.

This is a verified one-host minimum-Linux regression measurement, not a declaration that every case passed a performance threshold. It adds no Butterworth speed claim, no cross-profile or cross-CPU guarantee, no full-product qualification and no LPI implementation/parity credit. Main remains 212 native-partial image APIs plus three metadata functions; accepted combined supported-profile coverage remains 203 image APIs.
