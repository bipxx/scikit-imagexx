# One-thread public comparison: qualified 271-path product

Exact product `a6bae9ac86dbba362537758784edf06c973e9c53`, scikit-image 0.26.0, NumPy 1.24.4; one original thread and one native worker. Allocation-inclusive public calls on the same hosted runtime.

**34 wins and 34 losses across 68 cases.** This is scoped evidence, not full API parity or library-wide speed superiority.

Hosted run 42278375082 completed all measurements but failed its final report validator: that validator assumed pass index 1 for utility drivers whose frozen caller uses 0. The failed run and raw ordering remain unchanged. Independently reviewed report-only successor `c7cd06fc` validated these retained outputs without rerunning timings. Its summary is published separately; the failed artifact still has no summary.

| Cohort | Cases | Wins | Losses |
|---|---:|---:|---:|
| metrics-are | 8 | 8 | 0 |
| boundary | 8 | 8 | 0 |
| join | 8 | 0 | 8 |
| crop | 8 | 0 | 8 |
| regular-grid | 8 | 8 | 0 |
| slice | 8 | 0 | 8 |
| blocks | 4 | 0 | 4 |
| windows | 4 | 0 | 4 |
| ellipse | 12 | 10 | 2 |

All 952 accepted samples use seven rounds per arm and at least 100 ms per accepted batch. All 136 arm-specific warmup groups reach at least 250 ms. All 1,508 journal batches were independently reconstructed, including calibration/loop updates; no underlength samples were accepted. Utility drivers use pass 0; dedicated drivers use pass 1. Both alternate arm order across rounds and cases.

| API / case | Original ms | Current ms | Original/current | Outcome |
|---|---:|---:|---:|---|
| metrics.adapted_rand_error / metrics.adapted_rand_error/tiny | 0.254126026 | 0.013584473 | 18.707094x | win |
| metrics.adapted_rand_error / metrics.adapted_rand_error/small | 0.327929125 | 0.079696561 | 4.114721x | win |
| metrics.adapted_rand_error / metrics.adapted_rand_error/large-C | 8.146078600 | 0.729438519 | 11.167601x | win |
| metrics.adapted_rand_error / metrics.adapted_rand_error/large-F | 11.086950818 | 2.558820674 | 4.332836x | win |
| metrics.contingency_table / metrics.contingency_table/tiny | 0.143690046 | 0.097162770 | 1.478859x | win |
| metrics.contingency_table / metrics.contingency_table/small | 0.222751041 | 0.185043551 | 1.203776x | win |
| metrics.contingency_table / metrics.contingency_table/large-C | 7.959333143 | 2.152088926 | 3.698422x | win |
| metrics.contingency_table / metrics.contingency_table/large-F | 11.685638200 | 4.923397083 | 2.373491x | win |
| segmentation.find_boundaries / cross-thick | 0.157193658 | 0.004436782 | 35.429654x | win |
| segmentation.find_boundaries / u8-thick-512 | 4.424487192 | 0.142075531 | 31.141796x | win |
| segmentation.find_boundaries / u16-inner-F | 1.864413903 | 0.239094720 | 7.797805x | win |
| segmentation.find_boundaries / i32-outer-reverse | 2.966641769 | 1.594787260 | 1.860212x | win |
| segmentation.find_boundaries / u8-subpixel | 276.385922000 | 1.273404967 | 217.044797x | win |
| segmentation.find_boundaries / u16-outer-3d | 0.766163807 | 0.654316023 | 1.170938x | win |
| segmentation.find_boundaries / bool-inner-reverse | 2.412007042 | 1.345740791 | 1.792327x | win |
| segmentation.find_boundaries / u8-subpixel-1d | 2.876903128 | 0.015814306 | 181.917764x | win |
| segmentation.join_segmentations / u8-labels-C | 1.131797539 | 1.979493746 | 0.571761x | loss |
| segmentation.join_segmentations / u8-maps-reverse | 1.642547886 | 2.533374391 | 0.648364x | loss |
| segmentation.join_segmentations / i16-labels-F | 4.644804120 | 7.497286812 | 0.619531x | loss |
| segmentation.join_segmentations / i16-maps-C | 6.553924667 | 9.075114308 | 0.722186x | loss |
| segmentation.join_segmentations / u16-labels-C | 4.557251769 | 6.949585765 | 0.655759x | loss |
| segmentation.join_segmentations / u16-maps-F | 6.645884111 | 9.392800692 | 0.707551x | loss |
| segmentation.join_segmentations / i32-labels-reverse | 17.890711714 | 34.721689750 | 0.515260x | loss |
| segmentation.join_segmentations / i32-maps-C | 24.682360600 | 40.801952333 | 0.604931x | loss |
| util.crop / view-C | 0.002206601 | 0.002720089 | 0.811224x | loss |
| util.crop / view-F | 0.002207257 | 0.002693196 | 0.819568x | loss |
| util.crop / view-reverse | 0.002214778 | 0.002713344 | 0.816254x | loss |
| util.crop / view-3D | 0.002371426 | 0.002863463 | 0.828167x | loss |
| util.crop / copy-C | 0.035889548 | 0.037851716 | 0.948162x | loss |
| util.crop / copy-F | 0.019762964 | 0.022041181 | 0.896638x | loss |
| util.crop / copy-reverse-K | 0.026892242 | 0.720042006 | 0.037348x | loss |
| util.crop / copy-object-provider | 0.003611843 | 0.004310713 | 0.837876x | loss |
| util.regular_grid / doc2d | 0.023092056 | 0.005004779 | 4.614001x | win |
| util.regular_grid / dense2d | 0.023308919 | 0.004982759 | 4.677914x | win |
| util.regular_grid / thin3d | 0.032943143 | 0.005878122 | 5.604365x | win |
| util.regular_grid / permuted-readonly | 0.032141090 | 0.005415688 | 5.934812x | win |
| util.regular_grid / rank8 | 0.054634509 | 0.009557770 | 5.716240x | win |
| util.regular_grid / half-even-down | 0.021203840 | 0.004203706 | 5.044082x | win |
| util.regular_grid / half-even-up | 0.021070871 | 0.004186850 | 5.032631x | win |
| util.regular_grid / large-shape-metadata | 0.032944726 | 0.005906051 | 5.578131x | win |
| util.slice_along_axes / view-C | 0.002043984 | 0.003097459 | 0.659890x | loss |
| util.slice_along_axes / copy-C | 0.019561745 | 0.023972231 | 0.816017x | loss |
| util.slice_along_axes / view-F | 0.002024252 | 0.003039607 | 0.665959x | loss |
| util.slice_along_axes / copy-F-C-order | 0.917653016 | 2.314624200 | 0.396459x | loss |
| util.slice_along_axes / reverse-view | 0.002048074 | 0.003073235 | 0.666423x | loss |
| util.slice_along_axes / reverse-copy | 0.087090872 | 2.632915909 | 0.033078x | loss |
| util.slice_along_axes / permuted-axes | 0.002019606 | 0.003161133 | 0.638887x | loss |
| util.slice_along_axes / negative-bound-copy | 0.021739483 | 0.025227159 | 0.861749x | loss |
| util.view_as_blocks / view_as_blocks/C | 0.015413368 | 0.021626029 | 0.712723x | loss |
| util.view_as_blocks / view_as_blocks/F | 0.015394450 | 0.021676281 | 0.710198x | loss |
| util.view_as_blocks / view_as_blocks/reverse | 0.015418008 | 0.021738985 | 0.709233x | loss |
| util.view_as_blocks / view_as_blocks/readonly | 0.015200263 | 0.021377172 | 0.711051x | loss |
| util.view_as_windows / view_as_windows/C | 0.022719313 | 0.026912814 | 0.844182x | loss |
| util.view_as_windows / view_as_windows/F | 0.022710943 | 0.026889276 | 0.844610x | loss |
| util.view_as_windows / view_as_windows/reverse | 0.022964754 | 0.027087813 | 0.847789x | loss |
| util.view_as_windows / view_as_windows/readonly | 0.022524119 | 0.025911499 | 0.869271x | loss |
| draw.ellipse_perimeter / ellipse-perimeter/point | 0.000621633 | 0.001586304 | 0.391875x | loss |
| draw.ellipse_perimeter / ellipse-perimeter/unit | 0.001737085 | 0.001878311 | 0.924812x | loss |
| draw.ellipse_perimeter / ellipse-perimeter/unequal | 0.002472054 | 0.001919992 | 1.287534x | win |
| draw.ellipse_perimeter / ellipse-perimeter/flat | 0.003574903 | 0.001962086 | 1.821991x | win |
| draw.ellipse_perimeter / ellipse-perimeter/zero-axis | 0.003534546 | 0.001954135 | 1.808752x | win |
| draw.ellipse_perimeter / ellipse-perimeter/translated-clip | 0.009291760 | 0.002331928 | 3.984583x | win |
| draw.ellipse_perimeter / ellipse-perimeter/unit-angle-positive | 0.025116396 | 0.002161944 | 11.617508x | win |
| draw.ellipse_perimeter / ellipse-perimeter/unit-angle-negative | 0.025626656 | 0.002165136 | 11.836048x | win |
| draw.ellipse_perimeter / ellipse-perimeter/medium-axis | 0.033713457 | 0.003204443 | 10.520848x | win |
| draw.ellipse_perimeter / ellipse-perimeter/large-axis | 2.438169958 | 0.078217073 | 31.171838x | win |
| draw.ellipse_perimeter / ellipse-perimeter/large-partclip | 2.456348043 | 0.034752321 | 70.681555x | win |
| draw.ellipse_perimeter / ellipse-perimeter/fully-clipped-empty | 0.046164984 | 0.002687144 | 17.179945x | win |

Every loss is retained. The 32 JOIN/crop/slice/block/window losses and two ellipse losses require further optimization; no average hides them. Utility copy-loop and boundary overhead diagnostics are separate ongoing work, not gains established by this run. Small provider/representation work, object-copy providers and the returned ArrayMap class frontier remain explicit in the raw reports.

The ten actual public CLI cohorts passed namespace/native-return ownership, input/FP/worker checks, enforced stops before sampler bodies, post-stop source/authority checks and one post-batch offline identity verification. Their helper was built and controlled on the measurement host; no producer-host helper was transplanted. Benchmark results cover only this minimum profile and fixed case set.

Four-profile whole-product qualification is a separate admission; it does not imply speed on unmeasured domains. Historical 265-product results and failed experiments remain unchanged.
