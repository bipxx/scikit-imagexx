# LabelPoints: rejected zero-provider candidate

**DISCARD under the frozen gates.** The complete candidate is slower than the kept fused C++ baseline on this host: **0.959864258x direct / 0.954331433x normalized** geometric-mean gain. Original/current changes from 1.534370869x baseline to 1.464298350x candidate. All nine balanced direct gains are below 1; eight normalized gains are below 1.

**No sparse target passed.** None of the three targets repeats above 1.05x direct and normalized or beats original in both candidate passes. All six other original wins remain wins, but only two of six meet both inclusive 0.97x C++ control floors. Balanced original wins change from eight of nine baseline cases to six of nine candidate cases. The candidate remains unmerged; the kept fused implementation is unchanged.

[All four raw reports and comparison](label-points-zero-provider-same-host.json) retain **504 samples**, original numeric text, medians, loops, input/output hashes and metadata. The [initial mixed result](label-points-minimum-initial.md) and [kept fused comparison](label-points-fused-same-host.md) remain separate evidence.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse-64x64-i64 | 0.014171690 | 0.015036365 | 0.948127253x | 0.869857588x | 0.942494408x | 0.917448143x | 0.000864675 |
| dense-64x64-f64 | 0.033632321 | 0.034045882 | 1.893455041x | 1.849037030x | 0.987852816x | 0.976541291x | 0.000413562 |
| sparse-1024x1024-i64 | 0.181038747 | 0.181517985 | 1.018144728x | 0.999223248x | 0.997359830x | 0.981415727x | 0.000479238 |
| dense-1024x1024-f64 | 9.493323741 | 9.888486600 | 2.392929728x | 1.982876409x | 0.960038085x | 0.828639632x | 0.395162859 |
| sparse-64x64x64-f32 | 0.058628959 | 0.062036811 | 1.002878759x | 0.950385537x | 0.945067252x | 0.947657460x | 0.003407853 |
| dense-64x64x64-i64 | 0.995704027 | 1.031326691 | 3.023205810x | 2.941111759x | 0.965459380x | 0.972845365x | 0.035622664 |
| duplicates-1024x1024-f64 | 0.456680638 | 0.465936762 | 2.118481548x | 2.089721753x | 0.980134376x | 0.986424335x | 0.009256124 |
| strided-1024x1024-f32 | 2.177849720 | 2.321805409 | 1.232807037x | 1.201999397x | 0.937998383x | 0.975010169x | 0.143955689 |
| strided-64x64x64-i64 | 0.903500391 | 0.976867705 | 1.361056252x | 1.383280190x | 0.924895343x | 1.016328449x | 0.073367314 |

Ratios below 1 indicate a loss against the stated denominator. Added milliseconds are candidate minus baseline. Direct gain compares baseline C++ to candidate; normalized gain adjusts it using paired original timings. Balanced values use the geometric mean of the two respective passes.

| Workload | Original ms during baseline (GM) | Original ms during candidate (GM) | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|
| sparse-64x64-i64 | 0.013436566 | 0.013079496 | 0.875789878x | 0.863965480x |
| dense-64x64-f64 | 0.063681287 | 0.062952097 | 1.862983607x | 1.835194859x |
| sparse-1024x1024-i64 | 0.184323646 | 0.181376991 | 1.017602107x | 0.981176329x |
| dense-1024x1024-f64 | 22.716856598 | 19.607646796 | 1.980403418x | 1.985352488x |
| sparse-64x64x64-f32 | 0.058797737 | 0.058958888 | 0.946030285x | 0.954760840x |
| dense-64x64x64-i64 | 3.010218199 | 3.033247058 | 2.932933566x | 2.949312757x |
| duplicates-1024x1024-f64 | 0.967469505 | 0.973678188 | 2.126121376x | 2.053945299x |
| strided-1024x1024-f32 | 2.684868460 | 2.790808701 | 1.193272503x | 1.210790114x |
| strided-64x64x64-i64 | 1.229714856 | 1.351281745 | 1.396832278x | 1.369859585x |

All three sparse cases are balanced original losses. Pass 2 loses on sparse 64虏 and sparse 64鲁; pass 3 loses on all three sparse cases. The other six cases beat original in both candidate passes.

All eight initially slower workloads are retained below. Initial values came from an earlier host and product; they are historical context, not denominators for this same-host comparison.

| Initial loss case | First Release original/current | This baseline original/current | This candidate original/current |
|---|---:|---:|---:|
| sparse-64x64-i64 | 0.948811878x | 0.948127253x | 0.869857588x |
| dense-64x64-f64 | 0.722953725x | 1.893455041x | 1.849037030x |
| sparse-1024x1024-i64 | 0.939637313x | 1.018144728x | 0.999223248x |
| sparse-64x64x64-f32 | 0.988007864x | 1.002878759x | 0.950385537x |
| dense-64x64x64-i64 | 0.517598189x | 3.023205810x | 2.941111759x |
| duplicates-1024x1024-f64 | 0.674750524x | 2.118481548x | 2.089721753x |
| strided-1024x1024-f32 | 0.841805898x | 1.232807037x | 1.201999397x |
| strided-64x64x64-i64 | 0.732058768x | 1.361056252x | 1.383280190x |

Every candidate pass has a direct or normalized C++ regression for every workload (18 observations). All are retained here, including cases where the other metric improves.

| Workload / pass | Direct gain | Normalized gain | Original / candidate |
|---|---:|---:|---:|
| sparse-64x64-i64, pass 2 | 0.958944724x | 0.923704994x | 0.875789878x |
| sparse-64x64-i64, pass 3 | 0.926326290x | 0.911233674x | 0.863965480x |
| dense-64x64-f64, pass 2 | 1.001378141x | 0.983906967x | 1.862983607x |
| dense-64x64-f64, pass 3 | 0.974510173x | 0.969230755x | 1.835194859x |
| sparse-1024x1024-i64, pass 2 | 1.008086952x | 0.999467049x | 1.017602107x |
| sparse-1024x1024-i64, pass 3 | 0.986746855x | 0.963690429x | 0.981176329x |
| dense-1024x1024-f64, pass 2 | 0.942110625x | 0.827606174x | 1.980403418x |
| dense-1024x1024-f64, pass 3 | 0.978306688x | 0.829674380x | 1.985352488x |
| sparse-64x64x64-f32, pass 2 | 0.932497894x | 0.943314709x | 0.946030285x |
| sparse-64x64x64-f32, pass 3 | 0.957806035x | 0.952020204x | 0.954760840x |
| dense-64x64x64-i64, pass 2 | 0.945212508x | 0.970140225x | 2.932933566x |
| dense-64x64x64-i64, pass 3 | 0.986139949x | 0.975558047x | 2.949312757x |
| duplicates-1024x1024-f64, pass 2 | 0.992835018x | 1.003606275x | 2.126121376x |
| duplicates-1024x1024-f64, pass 3 | 0.967596204x | 0.969536553x | 2.053945299x |
| strided-1024x1024-f32, pass 2 | 0.916689950x | 0.967931288x | 1.193272503x |
| strided-1024x1024-f32, pass 3 | 0.959802131x | 0.982140820x | 1.210790114x |
| strided-64x64x64-i64, pass 2 | 0.965224378x | 1.026285486x | 1.396832278x |
| strided-64x64x64-i64, pass 3 | 0.886251336x | 1.006468016x | 1.369859585x |

| Group | Cases | Direct gain | Normalized gain | Original / candidate |
|---|---:|---:|---:|---:|
| targets | 3 | 0.961312205x | 0.948481013x | 0.938289454x |
| preserved_original_wins | 6 | 0.959141102x | 0.957270161x | 1.829262124x |
| dense | 3 | 0.971042497x | 0.923350755x | 2.209280846x |
| duplicates | 1 | 0.980134376x | 0.986424335x | 2.089721753x |
| sparse | 3 | 0.961312205x | 0.948481013x | 0.938289454x |
| strided | 2 | 0.931423822x | 0.995454958x | 1.289458008x |
| float32 | 2 | 0.941526184x | 0.961236527x | 1.068813755x |
| float64 | 3 | 0.975937730x | 0.927626962x | 1.971408264x |
| int64 | 4 | 0.957170518x | 0.971355288x | 1.371301947x |

## Fixed decision and scope

One candidate was frozen at **2026-10-28T16:42:47.988490Z**, with the unchanged two-hour deadline **18:42:47.988490Z**. Metrics completed **17:24:09.336947Z**, 4,718.651543 seconds before deadline. Dispatch-to-job-start delay was **3 seconds** (17:21:57Z to 17:22:00Z). No extension, alternate variant, retry or retiming occurred.

KEEP requires overall direct and normalized geometric means strictly above 1.02x. Each of the three sparse targets must exceed 1.05x direct and normalized in each candidate pass against the two-baseline geometric mean, and beat original in both passes. The other six cases must retain original wins in both candidate passes and each have balanced direct and normalized gains at least 0.97x. Source, identity, correctness and deadline gates are required. Correctness/identity/deadline passed; performance gates failed. Four controls miss a floor: dense 1024虏, dense 64鲁 and both strided workloads.

The rejected bundle moves literal `np.zeros(output_shape, dtype=np.uint64)` to its original prelude position, adds a typed-output overload under the same `label_points_apply` C entry name, and exposes ordered assignment without clearing. The existing standalone `label_points_into` retains its ABI and clear-then-assign semantics. Ordered stores preserve duplicate last-wins behavior. **This experiment tests the whole bundle, including allocation and boundary costs; it does not measure per-stage causality or establish a general claim about `np.zeros`.**

The candidate metadata explicitly identifies NumPy zero allocation as a provider and ordered scatter as owned. The baseline metadata retains owned zero-fill/scatter. Only one prospectively frozen descriptive scope assignment differs in the harness; exact inverse bytes and AST match the baseline. Numerical timing, nine workloads, original/independent-oracle prechecks and actual C-return guards are unchanged. The frozen case contract and old reports were not rewritten.

The fixed order is baseline/candidate/candidate/baseline on one fresh Linux host. Each report contains nine cases, seven paired rounds targeting 19 ms, seed 2026, global workers one and all four reference pools one. Whole public calls include NumPy negative check, round/int conversion/transpose, allocation, label generation/scatter and output destruction. Generation and full-word/dtype/layout/oracle/native-admission checks are outside timers; hooks are restored before timing. Inputs and backing gaps remain unchanged. Raw exports use hashes after exact-byte prechecks.

All 504 samples and medians, input/output metadata and original/native/oracle hashes, genuine C returns, both original product identities and eight before/after identities passed independent review. All 13 dependency wheels and scientific package/library identities match exactly, without normalization. No builds or tests ran on the timing host.

Local bounded correctness evidence: all six native modes passed, with previous standalone-into streams byte-exact; 424 private plus 396 public checks passed. All 384 frozen original observations and 232 native acceptances remain exact. The only eight historical intake differences are negative-shape errors occurring at literal zeros before C/worker intake; the initial post-suite audit failure was retained and corrected as metadata without replaying the passing suite. The SDK79 component client passed normal/scalar workers 1/4 with only its label header/archive member replaced. Genuine preparations passed 32 normal/scalar XML pairs: baseline 7,134, candidate 7,146; color 3,696, matching 1,612, smoke 1,435 across 245 functions and 4/4/4/9 guards. Minimum producers do not execute the entire declared 26-client/52-check SDK matrix.

## Provenance

- Exact measured baseline `910c31a609c157c28cbbe56933b8448b710af95e`, containing the kept fused LabelPoints implementation. Its five LabelPoints production blobs are byte-exact in main at publication `e6bd19b46cbd801f93af2301fe91357b3d5d1410`. This verifies the unchanged implementation; it does not relabel the later whole main product as measured.
- Rejected candidate `168e6b736221a488aad74b821dcfe74dbc87470c` remains unmerged on its experimental branch. Only the approved eight files and prospective candidate contract differ; CMake, registry, counts, SIMD source and frozen references remain fixed.
- [Single ABBA run 40735009233](https://github.com/bipxx/scikit-imagexx/actions/runs/40735009233), controller `15ecf51d2ef2d931933574b3bffba46577644e8b`.
- [Baseline preparation 40724150045](https://github.com/bipxx/scikit-imagexx/actions/runs/40724150045) / [artifact 12833680393](https://github.com/bipxx/scikit-imagexx/actions/runs/40724150045/artifacts/12833680393).
- [Candidate preparation 40730844227](https://github.com/bipxx/scikit-imagexx/actions/runs/40730844227) / [artifact 12835986984](https://github.com/bipxx/scikit-imagexx/actions/runs/40730844227/artifacts/12835986984). Both preparations skipped all 25 timing steps.
- [Comparison artifact 12836308583](https://github.com/bipxx/scikit-imagexx/actions/runs/40735009233/artifacts/12836308583): 38,236 bytes, 39 members, downloaded once; every member size/CRC/SHA verified.
- CPU: **AMD EPYC 7763 64-Core Processor**. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, AVX2 enabled. Both measured products contain 168 wheel members and SDK79, with 242 partially native image APIs plus three metadata APIs. The separate 241-image qualification campaign is a different source scope; full accepted qualification remains 203 image APIs. This rejected comparison grants no promotion.

| Proof | SHA-256 |
|---|---:|
| Combined raw publication | `ec92e2a8fc1a62e96930523f933d4dea0cd115c7fae0ed36f5aeccfc14169534` |
| Comparison ZIP | `659f5d829ca7c2f10569f395b05ef677ef4641c67f1557cc7454a483aba1d12a` |
| Baseline ZIP | `38797cf494532e16da41f751062c57815de2d566d8ebf11f3b59be26ba4be536` |
| Candidate ZIP | `6f7d6fcbd1506c2e61031d925575526ea8abf4c30c7770bf61a06aa04965cc08` |
| Prospective candidate contract, raw and LF | `b70415e455b317b44b8ce803d2f60544b022ae6bedaad4b02ef03c1eed923fc6` |
| Fixed nine-case contract | `3fceb3b66c254f1330f30f779a34a042162c08cf59921bf1343b06376f450221` |
| Baseline harness | `903c46306ed0a3e7a57d21a32e794fb9b48fe6af7ea7b685d7cf3ea66b1d0850` |
| Candidate harness, descriptive-only correction | `1436e8baf8b9012d0ee22ed26d3625ea7c1d606ad7794b2b136c9c093cbf04d2` |
| Independent comparison review | `9eb3a65b01c6460d50edf2fb5f61b9725a31445ef12bfa527b6f7a8208ebafe6` |
| Independent pass-loss audit | `84b09dbd795306028b61bbc15e9e0b0c870ab245ac67a45eded623ed8d4723da` |
| Baseline genuine native core | `ae226acb15af37f4b9f2968ef99d41b3bee2d85fe9f1f3e2ab9ad9384360168b` |
| Baseline preparation review | `b80b3db26e2da75708960ba0a12749741c98adb7f70a99865edae11a31c8134b` |
| Candidate genuine native core | `9f22e7a5e0665e9bb8d8788e903b8fb5fd48cfff765fc942e2d4f24c7d55fd96` |
| Candidate preparation review | `545335cbafae19501b3a81194003c66756c81ccdb81aa32b7836707a62f2bff1` |
