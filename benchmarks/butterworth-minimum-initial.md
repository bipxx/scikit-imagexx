# Butterworth: first minimum-Linux Release result

The new native-partial `filters.butterworth` is kept. On one fresh artifact-only Linux host, **five of six default workloads beat original scikit-image**, with a **1.112821796× geometric mean of original/current ratios**. This is the first original-library comparison for this new function, not an improvement over an earlier C++ implementation.

The loss is retained: **uint8 256×256 takes 2.0944475 ms versus 1.8739784 ms**, an absolute penalty of **+0.2204691 ms** (0.894736×). No case was removed.

[Exact raw report: all 84 samples](butterworth-minimum-initial.json). Seven warmed paired rounds per case include allocation and output destruction; calibration loop counts and all medians remain in that untouched JSON.

| Shape | Input dtype | Original ms | Current ms | Original/current | Current − original ms |
|---|---|---:|---:|---:|---:|
| 256×256 | uint8 | 1.873978400 | 2.094447500 | 0.894736× | +0.220469100 |
| 256×256 | float32 | 0.912974727 | 0.724430000 | 1.260266× | -0.188544727 |
| 256×256 | float64 | 1.347253571 | 1.120450875 | 1.202421× | -0.226802696 |
| 1024×1024 | uint8 | 27.507816000 | 23.880932000 | 1.151874× | -3.626884000 |
| 1024×1024 | float32 | 13.624933000 | 12.481585000 | 1.091603× | -1.143348000 |
| 1024×1024 | float64 | 22.213886000 | 19.941335000 | 1.113962× | -2.272551000 |

All six report zero maximum absolute error in the untimed parity check. Native workers were asserted to remain 1; the original SciPy FFT getter was recorded as **1 before and after**. Fallback was disabled. Both numerical implementations ran on the same measurement host, using the same inputs and offline dependency wheels.

## Product and evidence

- Source/main: `936481f1ddcbd2db0f98946ee5a4d788fc861902`.
- [Release and measurement run 40300864201](https://github.com/bipxx/scikit-imagexx/actions/runs/40300864201): producer `121498125528`, fresh measurement job `121501057895`, both successful.
- Genuine installed wheel gates: **175 normal + 175 scalar tests**, and **1,322 current-registry smoke checks**. Smoke preserves numerical gaps; it does not turn all 477 surfaces into native implementations.
- Declared source coverage: **212 native-partial image APIs + 3 metadata functions = 215**, with **257 upstream-only entries**. The separate full supported-profile combined baseline remains public203.
- Runtime: CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0; Linux x86_64, AVX2 enabled. A CPU model was not retained beyond architecture/platform metadata.
- Product wheel SHA256: `e1a5a120e25a037a9b90865622cc47ce210ee9eee1ccc59b0871bebdc2524661`.
- Sole installed core SHA256: `695fd45385f0a24cc0fc0fcc29a12b755d7fb220790324b38d4b7fd2406f32fd`.
- Core-source fingerprint (`cpp-python-build-posix-lf-v2`): `8eadefd45f13759b2b382cff7c7b7126638332eb33f030a9bc574736a40e7cbe`.
- Before/after source, wheel, complete installed package, SDK and runtime verification matched the prepared manifest SHA256 `0b172f41b46016777e9b61cf6efdc2948164343662da9661e157ae4d9fe535f5`.
- Sole downloaded performance artifact: `12640636973` (`ci-performance-minimum`), ZIP SHA256 `8368af59d04d2f1466e0f6b22d26bd61c33076661b66001f7ed2bbae9fc2bf59`. The producer ZIP was not duplicated.
- Protected raw archive, API/logs, 23-member CRC/SHA inventory and retained analysis: `build/phase3-butterworth-first-release-v1/`. Final review SHA256 `52102bf38630bed5ff837d55a472d0fc60e77b7495ee52d9d0c8395854956b48`; retention manifest SHA256 `99ac58d5f09c00b6246cad8ec8d6163aa2a45a0eb2cec7d9b38665246c725430`. Large ZIP/wheel bytes remain outside Git.

## Scope and retained history

Owned C++ executes the forward/inverse FFT. NumPy remains the explicit provider for padding, frequency-grid/mask arithmetic, spectrum multiplication and cropping. The existing upstream PocketFFT plan-cache option is 16, matching the minimum SciPy build configuration; this result does not establish a causal cache speedup. Extended precision and storage/protocol partitions remain limited or unassessed. Six ordinary 2D default cases on one minimum-Linux Release host do not establish full API, full supported-profile or universal performance.

Before Release, the genuine private component passed **700 public checks** across modern/minimum normal/scalar modes, plus **72 shared default projections**. Six native modes passed **8,046,020 controls** in total, including portable refusal and bounded sanitizer scopes. Those component results remain separate from the installed wheel gates above.

The earlier modern large-uint8 generic-tolerance failure remains failed historical evidence. The shared Butterworth projection was subsequently wired to its exact pre-output dtype/size/input-peak policy, with all old API budgets unchanged; neither the 87-case scientific tests nor their tolerances were fitted or weakened. Cache-0 evidence and all failed compile/bootstrap/metadata attempts remain protected.

Next: the LPI family has a separate test-first worktree with 80 prospective specs. It receives no native credit from this FFT consumer.

The FFT-foundation/component author performed this retention and result analysis; separate source reviews remain recorded. No additional numerical execution or timing was performed for publication.
