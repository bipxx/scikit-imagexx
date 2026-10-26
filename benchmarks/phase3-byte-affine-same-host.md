# Affine byte conversion: held because the fixed metric is flat

**HOLD; unmerged.** Candidate `887aff695ce10ea705091f50bfb12c35bb6ee286` improves stepped intersection coefficient but does not improve the fixed 21-case metric. Main retains the prior colocalization implementation and the separately merged octahedron optimization. This isolated experiment compares two products before that octahedron change; it is not a benchmark of the combined current main.

[Controlled run 40288858109](https://github.com/bipxx/scikit-imagexx/actions/runs/40288858109) used baseline–candidate–candidate–baseline on one AMD EPYC 7763 host: CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. All 73 workloads, seven paired rounds and one worker remain unchanged: 4,088 samples with exact untimed parity.

Balanced per-report medians give **0.999458890× direct gain** and **0.999467617× normalized gain** across the fixed 21 cases. All seven intersection cases give **1.131724× direct gain**. The stepped case improves **2.313035×**, from baseline medians 0.961402/0.963000 ms to candidate 0.414715/0.417270 ms, but remains slower than the original (0.741714×/0.745703×). All 14 Manders cases have direct losses. The same five intersection workloads beat the original in both candidate passes.

Preparation passed 785 public checks in each normal/scalar mode, 1,318 smoke checks, 21 metadata tests plus 30 subtests, and 73 corpus checks per mode. The source preserves the existing 2,727 native controls and adds 32 stride/canary checks; baseline and candidate each passed 2,759 controls per mode. These are scoped checks, not full-profile product qualification.

No identical rerun or metric change is used to promote the target gain. Source and compiled-code diagnosis are separate next work. The artifact review authenticates all four source/wheel/runtime identities, before/after verification, 14 dependency wheels and every raw sample. There are 24 direct and 45 normalized losses across all 73 cases; all cases are listed below. Greater than one is better. Added latency compares geometric means of the two candidate and two baseline medians.

| Case | Direct gain | Normalized gain | Added C++ ms | Candidate2 original/current | Candidate3 original/current |
|---|---:|---:|---:|---:|---:|
| morphology.ellipse/tiny/uint8/None | 1.084783× | 1.072040× | -0.000469264 | 11.408559× | 11.511177× |
| morphology.ellipse/tiny/uint8/crosses | 1.033789× | 1.016417× | -0.000966922 | 3.111904× | 3.154374× |
| morphology.ellipse/tiny/float64/None | 1.051910× | 1.034490× | -0.000276711 | 11.716079× | 11.635740× |
| morphology.ellipse/tiny/float64/crosses | 1.054753× | 1.028052× | -0.001546839 | 3.171602× | 3.157675× |
| morphology.ellipse/cache/uint8/None | 1.001216× | 0.997231× | -0.000104337 | 2.871579× | 2.922451× |
| morphology.ellipse/cache/uint8/crosses | 1.004065× | 0.985047× | -0.000582068 | 2.163793× | 2.158148× |
| morphology.ellipse/cache/float64/None | 1.004411× | 0.998180× | -0.000345352 | 3.200118× | 3.172636× |
| morphology.ellipse/cache/float64/crosses | 0.995954× | 0.991476× | +0.000564703 | 2.253385× | 2.285990× |
| morphology.ellipse/large/uint8/None | 0.999694× | 0.989156× | +0.001558990 | 2.169635× | 2.193717× |
| morphology.ellipse/large/uint8/crosses | 1.003739× | 0.993373× | -0.019896584 | 2.135142× | 2.166530× |
| morphology.ellipse/large/float64/None | 1.017946× | 0.988150× | -0.084304904 | 2.394223× | 2.386291× |
| morphology.ellipse/large/float64/crosses | 1.004241× | 0.992308× | -0.021454515 | 2.349343× | 2.292045× |
| morphology.octagon/tiny/uint8/None | 0.994625× | 0.982477× | +0.000038661 | 22.933363× | 23.006089× |
| morphology.octagon/tiny/uint8/sequence | 1.004707× | 1.037074× | -0.000042573 | 3.813317× | 3.785165× |
| morphology.octagon/tiny/float64/None | 1.031399× | 1.013303× | -0.000224963 | 22.995191× | 23.424189× |
| morphology.octagon/tiny/float64/sequence | 0.999944× | 0.991209× | +0.000000502 | 3.726166× | 3.709356× |
| morphology.octagon/cache/uint8/None | 1.001651× | 0.991161× | -0.000539454 | 2.530559× | 2.505422× |
| morphology.octagon/cache/uint8/sequence | 0.997262× | 0.992296× | +0.000024454 | 3.672968× | 3.775455× |
| morphology.octagon/cache/float64/None | 0.994009× | 0.985260× | +0.002002026 | 2.481811× | 2.474321× |
| morphology.octagon/cache/float64/sequence | 1.042748× | 1.022732× | -0.000374062 | 3.759379× | 3.740816× |
| morphology.octagon/large/uint8/None | 1.012327× | 1.004478× | -0.246345796 | 1.952027× | 1.951103× |
| morphology.octagon/large/uint8/sequence | 1.027654× | 0.966492× | -0.000252265 | 3.680861× | 3.473989× |
| morphology.octagon/large/float64/None | 1.018079× | 1.009651× | -0.365386921 | 1.938113× | 1.951792× |
| morphology.octagon/large/float64/sequence | 1.034118× | 0.996428× | -0.000304862 | 3.607408× | 3.712919× |
| morphology.octahedron/tiny/uint8/None | 1.014126× | 0.995170× | -0.000419401 | 1.039420× | 1.055427× |
| morphology.octahedron/tiny/uint8/sequence | 1.008095× | 0.990158× | -0.000236448 | 1.059977× | 1.051364× |
| morphology.octahedron/tiny/float64/None | 1.019620× | 0.985398× | -0.000582038 | 1.036327× | 1.030187× |
| morphology.octahedron/tiny/float64/sequence | 1.045810× | 1.002307× | -0.001320476 | 1.075609× | 1.054875× |
| morphology.octahedron/cache/uint8/None | 1.005623× | 1.016946× | -0.000892572 | 1.249322× | 1.241353× |
| morphology.octahedron/cache/uint8/sequence | 1.016278× | 0.966597× | -0.000477457 | 1.032636× | 1.030915× |
| morphology.octahedron/cache/float64/None | 0.973961× | 0.961773× | +0.004334986 | 1.147996× | 1.210285× |
| morphology.octahedron/cache/float64/sequence | 1.011862× | 0.971807× | -0.000345998 | 1.051482× | 1.033368× |
| morphology.octahedron/large/uint8/None | 1.021175× | 0.971668× | -0.071241517 | 1.306109× | 1.353422× |
| morphology.octahedron/large/uint8/sequence | 1.008815× | 0.959172× | -0.000256459 | 1.061354× | 1.031653× |
| morphology.octahedron/large/float64/None | 1.007043× | 0.956566× | -0.025444986 | 1.276955× | 1.276764× |
| morphology.octahedron/large/float64/sequence | 1.037467× | 1.016708× | -0.001093465 | 1.043439× | 1.073809× |
| morphology.star/tiny/uint8/None | 1.019539× | 1.003203× | -0.000251501 | 13.142945× | 12.573583× |
| morphology.star/tiny/float64/None | 0.996169× | 0.983512× | +0.000048823 | 12.671025× | 12.859160× |
| morphology.star/cache/uint8/None | 1.006870× | 0.992679× | -0.001679362 | 3.101962× | 3.077990× |
| morphology.star/cache/float64/None | 1.002700× | 0.994017× | -0.000664241 | 3.103110× | 3.052781× |
| morphology.star/large/uint8/None | 1.041125× | 1.027293× | -0.603201762 | 2.429703× | 2.424033× |
| morphology.star/large/float64/None | 1.015729× | 1.002939× | -0.237671926 | 2.384750× | 2.384748× |
| morphology.footprint_from_sequence/tiny/C | 1.031261× | 1.001837× | -0.001726183 | 2.283078× | 2.238366× |
| morphology.footprint_from_sequence/tiny/F | 1.070545× | 1.028884× | -0.003895831 | 2.294206× | 2.289528× |
| morphology.footprint_from_sequence/tiny/reverse | 1.038458× | 0.994334× | -0.002118498 | 2.235259× | 2.273989× |
| morphology.footprint_from_sequence/cache/C | 1.015663× | 1.006327× | -0.031266837 | 4.793584× | 4.845818× |
| morphology.footprint_from_sequence/cache/F | 1.328961× | 1.323344× | -0.659580856 | 4.819574× | 4.874814× |
| morphology.footprint_from_sequence/cache/reverse | 1.032974× | 1.017442× | -0.066287435 | 4.806748× | 4.757869× |
| morphology.footprint_from_sequence/large/C | 0.999243× | 0.994278× | +0.006169099 | 8.033233× | 8.008100× |
| morphology.footprint_from_sequence/large/F | 1.006409× | 1.004260× | -0.051797282 | 8.127515× | 8.075123× |
| morphology.footprint_from_sequence/large/reverse | 1.006515× | 1.002508× | -0.053031814 | 7.995804× | 8.071074× |
| morphology.octagon/tiny/even-sequence-warning | 1.000418× | 1.017664× | -0.000006745 | 2.578136× | 2.555149× |
| measure.intersection_coeff/future/tiny | 1.003599× | 1.020558× | -0.000050864 | 1.882688× | 1.767811× |
| measure.intersection_coeff/future/cache | 1.024125× | 1.014234× | -0.000663403 | 2.235855× | 2.220642× |
| measure.intersection_coeff/future/memory | 1.009141× | 1.005374× | -0.002208443 | 2.256380× | 2.240325× |
| measure.intersection_coeff/future/volume | 0.972480× | 0.997346× | +0.002029162 | 2.400502× | 2.231098× |
| measure.intersection_coeff/future/stepped | 2.313035× | 2.343405× | -0.546210053 | 0.741714× | 0.745703× |
| measure.intersection_coeff/future/masked | 1.010186× | 0.978333× | -0.004753712 | 0.900622× | 0.906633× |
| measure.manders_coloc_coeff/future/tiny | 0.966587× | 0.977649× | +0.001427912 | 0.395539× | 0.387750× |
| measure.manders_coloc_coeff/future/cache | 0.937821× | 0.935064× | +0.078869898 | 0.055475× | 0.054948× |
| measure.manders_coloc_coeff/future/memory | 0.939938× | 0.941313× | +1.216097957 | 0.059101× | 0.060050× |
| measure.manders_coloc_coeff/future/volume | 0.938785× | 0.926054× | +0.315587578 | 0.061010× | 0.060401× |
| measure.manders_coloc_coeff/future/stepped | 0.938104× | 0.929148× | +0.079314792 | 0.109664× | 0.108792× |
| measure.manders_coloc_coeff/future/masked | 0.970160× | 0.959755× | +0.038179892 | 0.316313× | 0.321788× |
| measure.manders_overlap_coeff/future/tiny | 0.964936× | 0.975359× | +0.002960903 | 0.287211× | 0.284079× |
| measure.manders_overlap_coeff/future/cache | 0.911346× | 0.930751× | +0.252865556 | 0.042213× | 0.040647× |
| measure.manders_overlap_coeff/future/memory | 0.953918× | 0.945771× | +2.040289534 | 0.053243× | 0.050935× |
| measure.manders_overlap_coeff/future/volume | 0.950245× | 0.909089× | +0.559763065 | 0.045193× | 0.044880× |
| measure.manders_overlap_coeff/future/stepped | 0.906234× | 0.906748× | +0.269721140 | 0.068028× | 0.067663× |
| measure.manders_overlap_coeff/future/masked | 0.931842× | 0.937757× | +0.160521880 | 0.185027× | 0.182524× |
| measure.intersection_coeff/future/cache-F | 1.008917× | 1.011944× | -0.000477452 | 1.284012× | 1.107109× |
| measure.manders_coloc_coeff/future/cache-F | 0.933699× | 0.944180× | +0.084676793 | 0.055631× | 0.054771× |
| measure.manders_overlap_coeff/future/cache-F | 0.908742× | 0.922236× | +0.262248386 | 0.040481× | 0.039735× |

[Exact compact raw archive](phase3-byte-affine-same-host.zip): artifact `12635460517`, 133,843 bytes, 30 members, SHA-256 `3fe24a73b9024773c44f2ffbaa1bfa9a6fcfe3b28051722f4722ae18a247ca3b`. The preparation archive and local review receipts are retained separately.
