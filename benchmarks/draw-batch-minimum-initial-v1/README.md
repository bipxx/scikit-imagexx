# Initial minimum-profile drawing comparison

This records one genuine-product run against scikit-image 0.26.0: **28 of 32 cases faster, with four polygon losses retained**. It measures public calls including preparation, allocation and output destruction. Ratios are original latency divided by current latency; below 1 means slower. These are fixed-case observations, not a library-wide claim, a gain over previous C++, or a release qualification.

Measured source: `db4f59b8aa69840591266e607e253a8611676ced`. [Run 42132995271](https://github.com/bipxx/scikit-imagexx/actions/runs/42132995271) used the genuine minimum product from [producer 42127381279](https://github.com/bipxx/scikit-imagexx/actions/runs/42127381279), controller `9c37eb7825e167f96537fb0571f822de7ed31910`, on AMD EPYC 7763 (4 logical CPUs available, AVX2 enabled). Python 3.11.17, NumPy 1.24.4; one library worker and all four reference pool limits set to 1.

The unchanged sustained sampler collected 7 accepted pairs per case, each batch at least 100 ms, after at least 250 ms warm-up per arm: **448 samples, 64 warm groups and 1,454 journal events**. Seven preceding real CLI probes each called original/native once and stopped at the real sampler CALL before its body. Their [terminals and call journals](first-cohort/) remain separate from measurements; the generic CLI uses its actual `run.measure` entry, while dedicated drivers use `montage_sustained.measure`.

| API | Cases | Faster | Slower | Geometric mean ratio |
|---|---:|---:|---:|---:|
| draw.polygon | 6 | 4 | 2 | 4.160551x |
| draw.polygon2mask | 6 | 4 | 2 | 3.551035x |
| draw.bezier_curve | 12 | 12 | 0 | 12.017409x |
| draw.set_color | 8 | 8 | 0 | 2.072418x |

The combined 12-case polygon family geometric mean is 3.843730x. Means weight cases equally; they do not represent total application time.

| API / case | Original ms | Current ms | Original/current | Outcome |
|---|---:|---:|---:|---|
| draw.polygon/tiny-triangle | 0.013907 | 0.018341 | 0.758248x | **slower** |
| draw.polygon/medium-rectangle | 8.878073 | 0.681558 | 13.026150x | faster |
| draw.polygon/large-concave | 35.922025 | 3.128596 | 11.481837x | faster |
| draw.polygon/strided-int32 | 2.228876 | 0.206317 | 10.803169x | faster |
| draw.polygon/negative-unclipped | 0.281445 | 0.041859 | 6.723596x | faster |
| draw.polygon/empty-clip | 0.009281 | 0.014739 | 0.629670x | **slower** |
| draw.polygon2mask/tiny-triangle | 0.017687 | 0.020395 | 0.867208x | **slower** |
| draw.polygon2mask/medium-rectangle-F | 9.562719 | 0.847475 | 11.283774x | faster |
| draw.polygon2mask/large-concave | 41.889650 | 3.596512 | 11.647297x | faster |
| draw.polygon2mask/strided-f32 | 2.344841 | 0.245488 | 9.551750x | faster |
| draw.polygon2mask/many-vertices | 2.467150 | 1.068377 | 2.309249x | faster |
| draw.polygon2mask/outside | 0.017127 | 0.021474 | 0.797578x | **slower** |
| draw.bezier_curve/bezier/point | 0.006873 | 0.002412 | 2.849153x | faster |
| draw.bezier_curve/bezier/tiny | 0.004627 | 0.002575 | 1.797016x | faster |
| draw.bezier_curve/bezier/straight-512 | 0.100725 | 0.004175 | 24.128160x | faster |
| draw.bezier_curve/bezier/half-medium | 0.076294 | 0.005263 | 14.495001x | faster |
| draw.bezier_curve/bezier/heavy-medium | 0.086534 | 0.005818 | 14.874054x | faster |
| draw.bezier_curve/bezier/flat-quarter | 0.057977 | 0.004762 | 12.174343x | faster |
| draw.bezier_curve/bezier/column-extremum-unit | 0.117734 | 0.006471 | 18.195071x | faster |
| draw.bezier_curve/bezier/column-extremum-weighted | 0.118497 | 0.006529 | 18.149516x | faster |
| draw.bezier_curve/bezier/row-extremum-weighted | 0.117907 | 0.006708 | 17.576824x | faster |
| draw.bezier_curve/bezier/both-extrema-weighted | 0.152576 | 0.007703 | 19.806898x | faster |
| draw.bezier_curve/bezier/large-clipped | 0.587625 | 0.018786 | 31.279722x | faster |
| draw.bezier_curve/bezier/fully-clipped | 0.022596 | 0.002904 | 7.780481x | faster |
| draw.set_color/tiny-gray | 0.015756 | 0.007503 | 2.099865x | faster |
| draw.set_color/rgb-dense | 4.442392 | 2.068598 | 2.147538x | faster |
| draw.set_color/duplicates | 4.411203 | 2.002178 | 2.203202x | faster |
| draw.set_color/clipped | 3.368772 | 1.366776 | 2.464758x | faster |
| draw.set_color/fortran-large | 18.106183 | 14.786786 | 1.224484x | faster |
| draw.set_color/color-alias | 4.423868 | 2.057322 | 2.150304x | faster |
| draw.set_color/f32-rgb-dense | 4.980625 | 2.203927 | 2.259887x | faster |
| draw.set_color/f32-duplicates | 5.061796 | 2.167639 | 2.335166x | faster |

The four slower cases are tiny-triangle polygon, empty clipped polygon, tiny-triangle mask and outside mask. Their extra measured latency is approximately 2.7–5.5 microseconds per call. All eight larger/nonempty polygon-family wins are retained too; this run does not establish a causal explanation for the losses.

Ownership and measurement scope:

- **Polygon/polygon2mask:** the ordered raster classification and emission/zero-scatter are owned. NumPy structural conversion, min/max/ceil bounds, storage adaptation and allocation remain explicit providers. These 12 bounded cases cover list/C/F/strided carriers and clipping; they do not establish every protocol, label magnitude or FP mode.
- **Bézier:** the bounded rational recurrence, extrema and clipping are owned. Ordered recurrence remains serial; independent fill SIMD is distinct. Scalar conversion, allocation, unsupported shape protocols and exceptional domains remain explicit frontiers. All 12 frozen cases remain in the table.
- **Set color:** the in-place API returns `None`; parity observes the entire independently mutated image. Both arms include the same fresh K-order image copy per call. The eight cases cover six float64 and two float32 storage workloads; NumPy index/conversion/broadcast/promotion and unsupported dtype/FP paths remain providers as declared by the source contract. This is no blanket claim of fully owned preparation or all dtype parity.
- The driver preflights enforce actual native acceptance/C returns, input/ownership checks and restoration of instrumentation, worker budget and recorded NumPy error policy/handler/buffer. The assessed kernel frontier requires nearest rounding and gradual underflow; other FP modes were not benchmarked here.

Raw evidence is unedited. In particular, the set-color report retains its inherited prospective “no timings yet” scope text and lacks `method_sha256`. Its actual pinned harness identity, 112 samples and reconstructed sustained batch journals prove this execution; the [independent review](proof/independent-review.json) explains this distinction without rewriting the report.

The [three reports and journals](raw/), [workflow summary](proof/workflow-summary.json), [independent review](proof/independent-review.json), and [single-retention receipt](proof/retention.json) are copied byte-for-byte. The [published controller contract](../../coverage/contracts/draw-batch-measurement-v1.json) and [genuine input admission](../../coverage/evidence/draw-batch-genuine-minimum-admission-v1.json) carry the complete source/runtime/product closure; the prepared package and its 2,370-file source map are not duplicated here.

Four-profile run 42135532859 failed before preparation on three stale source guards. The metadata-only successor `334e68e2e6717421501742365b6fe58328fc1019` preserves all 493 production blobs; its qualification is a separate pending gate. This benchmark publication grants no four-profile acceptance.

Closeout: preserved all ratios, losses, raw journals and seven untimed probe records without any algorithm, benchmark, build or test replay. Root owns byte-projection validation and publication; the next release action is independent admission of the corrected complete source barriers and a separately authorized exact-source qualification. No additional performance run is requested by this publication.
