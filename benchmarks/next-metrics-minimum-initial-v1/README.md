# Initial metrics, ellipse and boundary comparison

Qualified C++ source: `683f0043a6cd6bb61626241a003ffd615ba48205`.
Controller: `20b9d4e70e091ab84f334bf5823d1d4242933b20`.

[Hosted run42191638073](https://github.com/bipxx/scikit-imagexx/actions/runs/42191638073) measured the same installed runtime with one worker and reference thread pools limited to one. Python3.11.17, NumPy1.24.4, SciPy1.15.3, original scikit-image0.26.0; AMD EPYC7763. Seven samples per arm/case, accepted batches at least100ms, warmup at least250ms per arm. Four actual CLI checks passed before timing.

**18 wins,10 losses across28cases;392raw samples and56warmup groups.** These are scoped initial comparisons, not whole-library speed superiority, previous-C++ improvement, or full API/domain parity. Every loss is retained.

| Case | Original ms | C++ ms | Original / C++ |
|---|---:|---:|---:|
| metrics.adapted_rand_error/tiny | 0.256979 | 0.013443 | 19.1161x |
| metrics.adapted_rand_error/small | 0.331325 | 0.077818 | 4.2577x |
| metrics.adapted_rand_error/large-C | 8.627440 | 9.812910 | 0.8792x |
| metrics.adapted_rand_error/large-F | 11.910751 | 12.108820 | 0.9836x |
| metrics.contingency_table/tiny | 0.145221 | 0.098523 | 1.4740x |
| metrics.contingency_table/small | 0.230824 | 0.187749 | 1.2294x |
| metrics.contingency_table/large-C | 8.562626 | 9.898035 | 0.8651x |
| metrics.contingency_table/large-F | 12.742825 | 13.381596 | 0.9523x |
| ellipse-perimeter/point | 0.000625 | 0.001568 | 0.3987x |
| ellipse-perimeter/unit | 0.001772 | 0.001895 | 0.9350x |
| ellipse-perimeter/unequal | 0.002552 | 0.001934 | 1.3197x |
| ellipse-perimeter/flat | 0.003616 | 0.001978 | 1.8283x |
| ellipse-perimeter/zero-axis | 0.003580 | 0.002002 | 1.7882x |
| ellipse-perimeter/translated-clip | 0.009327 | 0.002410 | 3.8703x |
| ellipse-perimeter/unit-angle-positive | 0.025172 | 0.002184 | 11.5284x |
| ellipse-perimeter/unit-angle-negative | 0.025236 | 0.002176 | 11.5991x |
| ellipse-perimeter/medium-axis | 0.033718 | 0.003194 | 10.5558x |
| ellipse-perimeter/large-axis | 2.460121 | 0.080091 | 30.7166x |
| ellipse-perimeter/large-partclip | 2.505244 | 0.034846 | 71.8957x |
| ellipse-perimeter/fully-clipped-empty | 0.046660 | 0.002796 | 16.6861x |
| cross-thick | 0.162516 | 0.004429 | 36.6975x |
| u8-thick-512 | 4.426930 | 0.146958 | 30.1238x |
| u16-inner-F | 1.864229 | 6.310963 | 0.2954x |
| i32-outer-reverse | 2.994724 | 5.417648 | 0.5528x |
| u8-subpixel | 274.075863 | 1.272783 | 215.3359x |
| u16-outer-3d | 0.780556 | 1.780913 | 0.4383x |
| bool-inner-reverse | 2.437912 | 4.585460 | 0.5317x |
| u8-subpixel-1d | 2.860488 | 0.015940 | 179.4533x |

Ratios below1mean C++ is slower. Large label-metric cases and four boundary cases need optimization; point/unit ellipse overhead remains visible. Source inspection identifies the boundary C-contiguous2D fast-path restriction as a diagnosis target; no elapsed-time bottleneck attribution is claimed without profiling.

Provider computation, unsupported domains, scalar/SIMD and global-worker behavior remain governed by the source contracts and four-profile admission; this benchmark covers one-worker public calls only. Raw reports and batch journals accompany this table. The full62-member authenticated artifact is retained as artifact13489478349, SHA256 `b5aafd70b243be64528f80f1c77320d71d19f10886686de564b2be76f4fc07a7` (2115950bytes). No measurement retry was performed.

Independent reconstruction passed all28cases/625batches. [Root admission](../../coverage/evidence/next-metrics-initial-performance-admission-v1.json) records qualification, the matching shipping production projection, raw artifact identity and remaining scope. The historical ellipse metadata field `numpy_binary_sha256` identifies `skimage.draw._draw`; its label is preserved and its actual runtime package slot was verified.
