# Ellipse perimeter: initial one-worker comparison

Genuine minimum run41972961044, exact `dca7c03dab19a4c6fe3a1e6879973590c51d273a`, original scikit-image0.26.0, NumPy1.24.4, Python3.11.17, AMD EPYC 9V74 80-Core Processor. Allocation-inclusive public calls; one worker, seven paired samples/backend/case, at least100ms accepted batches and250ms warmup groups.

10/12 cases faster; 2/12 slower. Geometric mean original/current 4.839609x. The point and unit losses remain open; no library-wide superiority or optimization acceptance is claimed. Full258 qualification is separate and its minimum job has failed the pre-existing matrix runtime admission guard before the ellipse checks.

| Case | Original ms | C++ ms | Original/current |
|---|---:|---:|---:|
| ellipse-perimeter/point | 0.000444492 | 0.001280028 | 0.347252x |
| ellipse-perimeter/unit | 0.001338720 | 0.001546021 | 0.865913x |
| ellipse-perimeter/unequal | 0.001914674 | 0.001559506 | 1.227744x |
| ellipse-perimeter/flat | 0.002767716 | 0.001584046 | 1.747244x |
| ellipse-perimeter/zero-axis | 0.002750431 | 0.001605229 | 1.713420x |
| ellipse-perimeter/translated-clip | 0.006912073 | 0.001904643 | 3.629064x |
| ellipse-perimeter/unit-angle-positive | 0.018901165 | 0.001765397 | 10.706469x |
| ellipse-perimeter/unit-angle-negative | 0.018797955 | 0.001768556 | 10.628984x |
| ellipse-perimeter/medium-axis | 0.025931597 | 0.002628526 | 9.865451x |
| ellipse-perimeter/large-axis | 1.922931459 | 0.063359248 | 30.349657x |
| ellipse-perimeter/large-partclip | 1.983369172 | 0.025848045 | 76.731885x |
| ellipse-perimeter/fully-clipped-empty | 0.035373232 | 0.002246853 | 15.743456x |

All168 samples and566 journal events are retained alongside this report. Strict preflight validates exact ordered intp words, dtype/shape/strides/ownership and real C-return/provider refusal outside the timed region. The point returns owning zero-dimensional arrays; fully clipped outputs are empty with zero strides. Bounded axis-aligned cases and two unit-radius rotation cases only; general rotated ellipses and unassessed protocols remain open.

Historical metadata field `numpy_binary_sha256` actually identifies the original `skimage.draw._draw` binary. Admission verifies that identity against the reference provenance and genuine scikit-image wheel; the NumPy wheel is authenticated separately. Frozen raw metadata is preserved.
