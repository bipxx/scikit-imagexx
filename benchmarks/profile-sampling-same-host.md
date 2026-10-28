# Profile sampling: controlled candidate discarded

**DISCARD: do not merge candidate `99fdb2a`.** The fixed 18 same-host experiment improved the prior C++ implementation by **1.036574192567× direct** and **1.009142039953× reference-normalized**, but failed two required gates: original/candidate speed ratio was **0.983057566936×** (below 1), and only **4/6** linewidth 33 targets repeatedly beat both baseline passes under both metrics (required 6/6). Correctness and deadline gates passed. No further tuning or timing was performed.

The [combined raw JSON](profile-sampling-same-host.json) embeds all four original reports and the comparison without rewriting their number text: **1,008 samples**, seven paired rounds, 18 cases and one global worker. Full public preparation, NumPy coordinates, finite admission, owned sampling, allocation, flip and width reduction remain timed. Original scientific-library pools also use one thread.

| Side | Dtype | Width | Direct gain | Normalized gain | Original/candidate | Added ms | Repeat direct / normalized |
|---:|---|---:|---:|---:|---:|---:|---|
| 256 | uint8 | 1 | 1.023343303305× | 1.009877042203× | 1.006400974247× | -0.098911375 | no / no |
| 256 | uint8 | 9 | 1.008012454325× | 1.006016154042× | 0.995020961321× | -0.034995540 | no / yes |
| 256 | uint8 | 33 | 1.030579532007× | 1.015256044372× | 0.971658084529× | -0.144307597 | no / yes |
| 256 | float32 | 1 | 1.037239097711× | 0.992649769577× | 0.997087328634× | -0.156809683 | yes / no |
| 256 | float32 | 9 | 1.049597649310× | 1.001564904050× | 0.996931458391× | -0.211989381 | yes / no |
| 256 | float32 | 33 | 1.036575007989× | 1.013432631038× | 0.968738456919× | -0.166269333 | yes / yes |
| 256 | float64 | 1 | 1.025266363130× | 0.987990046232× | 0.993207831510× | -0.107858683 | no / no |
| 256 | float64 | 9 | 1.042100850476× | 1.016568175274× | 0.999477868924× | -0.181558750 | yes / yes |
| 256 | float64 | 33 | 1.052458718324× | 1.008218963277× | 0.980931846541× | -0.238724974 | yes / yes |
| 1024 | uint8 | 1 | 1.047430585394× | 1.002089407104× | 0.995813494086× | -0.796084943 | yes / no |
| 1024 | uint8 | 9 | 1.037455441081× | 1.006638415278× | 0.984912814114× | -0.642841009 | yes / no |
| 1024 | uint8 | 33 | 1.043309860677× | 0.997976083836× | 0.970425394075× | -0.791215492 | yes / no |
| 1024 | float32 | 1 | 1.041286444171× | 1.008944627055× | 0.986487726815× | -0.705236750 | yes / yes |
| 1024 | float32 | 9 | 1.036939584148× | 1.024956203429× | 0.991086299637× | -0.646845955 | no / yes |
| 1024 | float32 | 33 | 1.045182651561× | 1.027801916848× | 0.965585891004× | -0.841706197 | yes / yes |
| 1024 | float64 | 1 | 1.039305899322× | 1.015354456844× | 0.970198509036× | -0.699238880 | yes / no |
| 1024 | float64 | 9 | 1.030006268611× | 1.011372928757× | 0.965779672195× | -0.546545241 | yes / no |
| 1024 | float64 | 33 | 1.033182796164× | 1.018749013209× | 0.957142257164× | -0.638216541 | yes / yes |

Direct gain is the geometric mean of the two baseline C++ medians divided by the two candidate medians. Normalized gain divides candidate original/current speedup by baseline original/current speedup. Original/candidate uses the two candidate passes. Added milliseconds are candidate minus baseline balanced medians; negative values mean lower C++ latency. Repeat wins require both candidate passes to beat both baseline passes.

**Zero direct C++ losses; three reference-normalized losses:** 256 float32 width 1 (0.992649769577×), 256 float64 width 1 (0.987990046232×), and 1024 uint8 width 33 (0.997976083836×). These normalized losses are not absolute C++ latency regressions. The 256 uint8 width 33 target failed the direct-repeat gate;1024 uint8 width 33 failed the normalized-repeat gate.

| Group | Cases | Direct gain | Normalized gain | Original/candidate |
|---|---:|---:|---:|---:|
| Width 33 targets | 6 | 1.040187581435× | 1.013530948504× | 0.969054195857× |
| Width 1/9 controls | 12 | 1.034772207915× | 1.006954717838× | 0.990134954381× |
| uint8 | 6 | 1.031603475172× | 1.006293939410× | 0.987284796077× |
| float32 | 6 | 1.041125428284× | 1.011483218759× | 0.984237372105× |
| float64 | 6 | 1.037015686892× | 1.009655830652× | 0.977675091947× |

The same-host original/current aggregate was **0.974151831968× baseline →0.983057566936× candidate**. Individual candidate passes won only **1/18 and 5/18** original comparisons. All 12 controls and all dtype groups remain included. These scores are not multiplied by the earlier result from another host.

The unchanged candidate touches only `src/resample.cpp`, `src/resample_detail.hpp` and `src/simd_resample_avx2.cpp`: paired-point addressing removes indexing divisions from Profile packets, and a packed integer exponent-mask scan retains whole-image finite admission, including signaling NaNs. Ordered sample arithmetic, old zoom addressing, Python orchestration, providers, worker policy, harness and frozen expectations remain unchanged. This candidate remains unmerged because its measured gains did not satisfy the frozen acceptance contract.

Dependency preparation exposed one test-tool difference: baseline bundled pytest 9.1.1, candidate pytest 9.2.0. Each original offline installation passed its original full source/package/core/SDK verifier first. Versioned measurement copies then replaced only the pytest wheel record, runtime pytest version and derived hash lock with the retained baseline pytest 9.1.1 wheel; both measurement environments used the same 13 dependency wheels. The scientific packages, product wheels, core, SDK, source maps and original artifacts stayed unchanged. Both normalized installations and all eight pre/post measurement identities passed the same official verifier. Original and normalized manifest hashes and the exact old/new pytest records are retained in the artifact.

Correctness remained separate from timing: candidate preparation passed **4,250 focused checks**, **3,696 color regressions**, **1,612 matching checks**, **1,409 smoke checks** for 237 registered functions, plus separate 4 Polygon/4 Euler/9 integer-width guards. All 18 preparation timing steps were skipped. Retained component validation passed 372 public and 136 shared checks, all 368 saved rows exact; six native and finite-mask modes and the old 712 zoom controls per normal/scalar mode also passed. Every measured case passed exact pre-reducer snapshots and public output bytes with zero error.

[Controlled run 40646958982](https://github.com/bipxx/scikit-imagexx/actions/runs/40646958982) used **AMD EPYC 9V45**, Linux x86_64, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Comparison completed **2026-10-28 06:58:41.388360 UTC**, before the unchanged **08:08:50.185599 UTC** deadline. Baseline source `9ead9b3` came from [Release 40448470579](https://github.com/bipxx/scikit-imagexx/actions/runs/40448470579); unchanged candidate `99fdb2a` came from [prepare-only Release 40645126837](https://github.com/bipxx/scikit-imagexx/actions/runs/40645126837). The expired pre-pause contract remains recorded without a performance or KEEP claim. This single-host experiment does not advance full supported-profile qualification beyond **203 image APIs**.

Provenance: [exact artifact 12797573572](https://github.com/bipxx/scikit-imagexx/actions/runs/40646958982/artifacts/12797573572), 796,908 bytes, 49 members, SHA256 `5ea410d258d301184a7f606f51b9fd51a55e4eb418d13c00a579b24cbc9cbcee`; every member size/CRC/SHA was verified. Contract SHA256 `05c198d0f01adfc37ef1d31beec9f46f30b9fe8e39735b851f6b64912fc2017f`. Independent review SHA256 `588e8cc29c3b6722c22c670f024ceea5d9b07b5c67ca0d255b77562155a2410a`. Combined raw JSON SHA256 `72ddb29ede3a433525068b51921b2b4720678d91d4aade03f210224bbc68ef67`.
