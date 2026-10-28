# LabelPoints: rejected binder metadata candidate

**DISCARD under the frozen gates.** Overall gains pass their thresholds: **1.018087346x direct / 1.013675040x normalized** versus the kept fused C++ baseline. Original/current geometric mean changes from 1.504055242x baseline to 1.524623258x candidate. The tiny target fails repeated improvement and repeated original-win requirements; one fixed preserved-original case and one normalized C++ control floor also fail. Small aggregate gains do not override those gates.

**The candidate remains unmerged.** The sparse 1024² case is below original in both baseline passes on this same host as well as both candidate passes. This is an unmet fixed preservation gate, not a newly lost same-host original win. The previously [rejected zero-provider bundle](label-points-zero-provider-same-host.md) is a separate closed experiment and is absent from all four reports here.

[All four complete raw reports and comparison](label-points-metadata-same-host.json) retain **504 samples**, original numeric text and terminal LF bytes exactly once, all medians, loops, input/output hashes and metadata. The [first mixed result](label-points-minimum-initial.md) and [kept fused comparison](label-points-fused-same-host.md) remain separate evidence.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse-64x64-i64 | 0.014470563 | 0.013724327 | 0.950460386x | 0.998345970x | 1.054373242x | 1.050381462x | -0.000746236 |
| dense-64x64-f64 | 0.032749936 | 0.032027734 | 1.924478331x | 1.968573246x | 1.022549256x | 1.022912659x | -0.000722202 |
| sparse-1024x1024-i64 | 0.159499323 | 0.152542476 | 0.958984501x | 0.959238109x | 1.045605963x | 1.000264455x | -0.006956846 |
| dense-1024x1024-f64 | 9.058683048 | 9.033009315 | 2.182257207x | 2.173337510x | 1.002842213x | 0.995912628x | -0.025673733 |
| sparse-64x64x64-f32 | 0.056668614 | 0.053502920 | 0.935555077x | 1.002121193x | 1.059168611x | 1.071151467x | -0.003165693 |
| dense-64x64x64-i64 | 0.977959028 | 0.977693794 | 2.919270056x | 2.964805250x | 1.000271285x | 1.015598144x | -0.000265234 |
| duplicates-1024x1024-f64 | 0.410715571 | 0.407043584 | 2.235302736x | 2.250517029x | 1.009021115x | 1.006806368x | -0.003671987 |
| strided-1024x1024-f32 | 2.095700283 | 2.158397441 | 1.236887209x | 1.191820210x | 0.970951986x | 0.963564181x | 0.062697158 |
| strided-64x64x64-i64 | 0.830089784 | 0.828933887 | 1.362697432x | 1.363223772x | 1.001394438x | 1.000386249x | -0.001155897 |

Ratios below 1 indicate losses against their stated denominator. Added time is candidate minus baseline. Balanced values use geometric means of the two respective passes. Eight balanced direct gains exceed 1; the strided 1024² float32 case is a direct loss. Seven normalized gains exceed 1; dense 1024² and strided 1024² float32 are normalized losses. Every loss remains visible above.

| Workload | Original ms during baseline (GM) | Original ms during candidate (GM) | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|
| sparse-64x64-i64 | 0.013753697 | 0.013701626 | 0.980874925x | 1.016128204x |
| dense-64x64-f64 | 0.063026541 | 0.063048940 | 1.967956639x | 1.969190047x |
| sparse-1024x1024-i64 | 0.152957378 | 0.146324556 | 0.931456082x | 0.987848775x |
| dense-1024x1024-f64 | 19.768376366 | 19.631777973 | 2.183910717x | 2.162815493x |
| sparse-64x64x64-f32 | 0.053016609 | 0.053616410 | 0.997635424x | 1.006627133x |
| dense-64x64x64-i64 | 2.854926507 | 2.898671694 | 2.988059068x | 2.941732398x |
| duplicates-1024x1024-f64 | 0.918073639 | 0.916058517 | 2.295044858x | 2.206853116x |
| strided-1024x1024-f32 | 2.592144874 | 2.572421692 | 1.187633325x | 1.196021857x |
| strided-64x64x64-i64 | 1.131161216 | 1.130022380 | 1.395811517x | 1.331396846x |

Tiny sparse 64² and sparse 1024² remain balanced original losses. Candidate pass 2 loses on all three sparse cases; pass 3 loses only on sparse 1024². Sparse 64³ has a balanced original ratio of 1.002121193x but remains mixed by pass (0.997635424x / 1.006627133x).

The fixed tiny-target requirements apply to each pass, not its favorable balanced gain:

| Tiny sparse 64² candidate pass | Direct gain (>1.05 required) | Normalized gain (>1.05 required) | Original / candidate (>1 required) |
|---|---:|---:|---:|
| 2 | 1.069877234x | 1.031999797x | 0.980874925x |
| 3 | 1.039093924x | 1.069090537x | 1.016128204x |

| Sparse 1024² original ratio | Pass 1 baseline | Pass 2 candidate | Pass 3 candidate | Pass 4 baseline |
|---|---:|---:|---:|---:|
| Original / current | 0.969285775x | 0.931456082x | 0.987848775x | 0.948792705x |

All eight initially slower cases are retained below. Initial ratios came from an earlier host/product; they are context, not denominators for this experiment.

| Initial loss case | First Release original/current | This baseline original/current | This candidate original/current |
|---|---:|---:|---:|
| sparse-64x64-i64 | 0.948811878x | 0.950460386x | 0.998345970x |
| dense-64x64-f64 | 0.722953725x | 1.924478331x | 1.968573246x |
| sparse-1024x1024-i64 | 0.939637313x | 0.958984501x | 0.959238109x |
| sparse-64x64x64-f32 | 0.988007864x | 0.935555077x | 1.002121193x |
| dense-64x64x64-i64 | 0.517598189x | 2.919270056x | 2.964805250x |
| duplicates-1024x1024-f64 | 0.674750524x | 2.235302736x | 2.250517029x |
| strided-1024x1024-f32 | 0.841805898x | 1.236887209x | 1.191820210x |
| strided-64x64x64-i64 | 0.732058768x | 1.362697432x | 1.363223772x |

All seven per-pass C++ regression observations against the two-baseline geometric mean:

| Workload / pass | Direct gain | Normalized gain | Original / candidate |
|---|---:|---:|---:|
| sparse-1024x1024-i64, pass 2 | 1.055376555x | 0.971294198x | 0.931456082x |
| dense-1024x1024-f64, pass 3 | 1.003407712x | 0.991091007x | 2.162815493x |
| dense-64x64x64-i64, pass 3 | 0.997962522x | 1.007694506x | 2.941732398x |
| duplicates-1024x1024-f64, pass 3 | 0.989314463x | 0.987272588x | 2.206853116x |
| strided-1024x1024-f32, pass 2 | 0.972701193x | 0.960179163x | 1.187633325x |
| strided-1024x1024-f32, pass 3 | 0.969205926x | 0.966961133x | 1.196021857x |
| strided-64x64x64-i64, pass 3 | 0.987991959x | 0.977030421x | 1.331396846x |

| Group | Cases | Direct gain | Normalized gain | Original / candidate |
|---|---:|---:|---:|---:|
| targets | 1 | 1.054373242x | 1.050381462x | 0.998345970x |
| preserved_original_wins | 7 | 1.007298026x | 1.000622168x | 1.719752445x |
| dense | 3 | 1.008505370x | 1.011409970x | 2.332159818x |
| duplicates | 1 | 1.009021115x | 1.006806368x | 2.250517029x |
| sparse | 3 | 1.053034281x | 1.040169840x | 0.986376190x |
| strided | 2 | 0.986055738x | 0.981802605x | 1.274644124x |
| float32 | 2 | 1.014101507x | 1.015934637x | 1.092862430x |
| float64 | 3 | 1.011437456x | 1.008483001x | 2.127422195x |
| int64 | 4 | 1.025112079x | 1.016454339x | 1.402628644x |

## Frozen decision

| Gate | Observed | Result |
|---|---:|---:|
| Overall direct and normalized GM >1.01 | 1.018087346x / 1.013675040x | pass |
| Tiny target repeated direct and normalized >1.05 | 0 / 1 target | fail |
| Tiny target original win in both passes | 0 / 1 target | fail |
| Fixed seven original-win cases preserved in both passes | 6 / 7 cases | fail: sparse 1024² |
| Other eight balanced direct and normalized gains >=0.97 | 7 / 8 cases | fail: strided 1024² float32 normalized 0.963564181x |
| Correctness, source, identities and deadline | all verified | pass |

The seven preserved case IDs were fixed prospectively from earlier evidence. Sparse 1024² is already an original loss in this matched baseline; its two original losses here are not evidence of a new same-host lost win. The control floor concerns balanced gains: strided 1024² direct 0.970951986x passes, while normalized 0.963564181x fails.

One candidate was frozen **2026-10-28T17:39:29.511203Z**, with the unchanged two-hour deadline **19:39:29.511203Z**. Metrics completed **18:18:01.463172Z**, 4,888.048031 seconds before deadline. Dispatch-to-job-start waiting was **331 seconds** (18:10:44Z to 18:16:15Z). No extension, alternate variant, retiming or retry occurred.

## Measured scope

The candidate removes the redundant array-owners vector, reserves the known rank for four remaining metadata vectors, and caches one dtype borrow per axis. The immutable tuple argument retains arrays and their base chains through the C invocation and GIL release. The Python prelude, output allocation, numerical clear/scatter leaf, native headers, SIMD source, public/private signatures, harness and nine-case contract remain unchanged. All 79 SDK payloads, including the numerical archive, are byte-exact to baseline; `_core.so` is the only changed packaged Python component. This tests the complete binder metadata bundle, without measured per-stage attribution.

The fixed order is baseline/candidate/candidate/baseline on one fresh Linux host. Each report contains nine cases, seven paired rounds targeting 19 ms, seed 2026, one global worker and all four reference pools limited to one. Whole public calls include the NumPy negative check/round/int conversion/transpose, metadata preparation, output allocation, owned zero-fill/ordered scatter and output destruction. Untimed guards compare full words and dtype/layout/ownership with pinned original and independent oracle; genuine C returns and native acceptance are observed with original forwarding blocked, then hooks restored before timers. Inputs/backing gaps stay unchanged. Exports retain hashes after exact-word prechecks.

All 504 samples/medians, source fingerprints, original/native/oracle hashes, both original product identities and eight before/after identities passed independent review. All 13 dependency wheels and scientific libraries match exactly, without normalization. No builds, tests or SDK clients execute on the timing host.

Before genuine preparation, two changed binder TUs and two tiny component links passed first attempt; 400 private plus 396 public checks preserved all 384 frozen observations/intake rows and 232 native acceptances exactly. The unchanged numerical six-mode proof was reused by authenticated hashes. Both genuine preparations passed 32 normal/scalar XML pairs (7,134), color 3,696, matching 1,612, smoke 1,435 across 245 functions and 4/4/4/9 guards; all 25 preparation timing steps were skipped. These minimum producers do not execute the full declared 26-client/52-check SDK matrix.

## Provenance

- Exact measured baseline `910c31a609c157c28cbbe56933b8448b710af95e`, containing the kept fused LabelPoints implementation. Rejected candidate `69e390dd5e48d5199c9a249fccafd83daa027d3c` remains unmerged.
- Candidate base `5df3a0fb679d2649b50f0974e6d645a08a19958c` contains a closed, hash-authenticated set of 21 pre-existing metadata/test/evidence changes relative to `910`; no product source, LP harness or case-contract change is admitted there. Only the binder and prospective metadata contract differ between `5df` and `69e390d`.
- The closed zero-provider candidate `168e6b7` is not included. No source from either rejected candidate is merged as part of this publication.
- [Single ABBA run 40742666831](https://github.com/bipxx/scikit-imagexx/actions/runs/40742666831), controller `d56b6e674e714d35d5b357c9ea152213401cd5a7`.
- [Baseline preparation 40724150045](https://github.com/bipxx/scikit-imagexx/actions/runs/40724150045) / [artifact 12833680393](https://github.com/bipxx/scikit-imagexx/actions/runs/40724150045/artifacts/12833680393).
- [Candidate preparation 40738793434](https://github.com/bipxx/scikit-imagexx/actions/runs/40738793434) / [artifact 12839223780](https://github.com/bipxx/scikit-imagexx/actions/runs/40738793434/artifacts/12839223780).
- [Comparison artifact 12841360278](https://github.com/bipxx/scikit-imagexx/actions/runs/40742666831/artifacts/12841360278): 38,127 bytes, 39 members, downloaded once; every member size/CRC/SHA verified.
- CPU: **AMD EPYC 7763 64-Core Processor**. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, AVX2 enabled. The measured products contain 168 wheel members/SDK79 and 242 partially native image APIs plus three metadata APIs. Later Montage/current-main scopes are separate. Full accepted four-profile qualification remains 203 image APIs; this rejected experiment grants no promotion.

| Proof | SHA-256 |
|---|---:|
| Combined raw publication | `10812ed3c4458d8dc2099e8a2824c513b1953383e4cd0d868e87704bc08ea28d` |
| Comparison ZIP | `87086e9471a0eb40ba9abca66f39b6138164cbb926cbe574d8795055cc2a5078` |
| Baseline ZIP | `38797cf494532e16da41f751062c57815de2d566d8ebf11f3b59be26ba4be536` |
| Candidate ZIP | `700cfabeae84a880753995117cd3fe2df6ec5e552c72c8c20309128d103f4503` |
| Prospective metadata contract, raw and LF | `2276bdf9fc048b923613f9964c6a096169f7483d1babd9b95c802347cc75b3f9` |
| Unchanged harness | `903c46306ed0a3e7a57d21a32e794fb9b48fe6af7ea7b685d7cf3ea66b1d0850` |
| Fixed nine-case contract | `3fceb3b66c254f1330f30f779a34a042162c08cf59921bf1343b06376f450221` |
| Independent raw review | `1b0c7e38f30965e538961de2b3272b0d1745f9419f61b8caa2c4e5ec3d038048` |
| Independent pass-loss audit | `b677bdc5f0697e4ad80a227a294d3b27fc7b61d40fed99b92b6372a7e6815a5b` |
| Baseline genuine native core | `ae226acb15af37f4b9f2968ef99d41b3bee2d85fe9f1f3e2ab9ad9384360168b` |
| Baseline preparation review | `b80b3db26e2da75708960ba0a12749741c98adb7f70a99865edae11a31c8134b` |
| Candidate genuine native core | `de209fce9e8b3569d84f6741af7542105ecda7360eea1efc7fa78333fbb19f5e` |
| Candidate preparation review | `ef84e337e65926847811c598e9d18c90b422a7485d0edfd962c23c9c8caf7795` |
