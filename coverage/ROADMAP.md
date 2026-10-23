# Native implementation roadmap

Objective: iterate through the scikit-image API, implement native CPU kernels,
preserve Python call contracts, and verify both parity and performance against
the original library. The project remains incomplete until that work is done;
upstream forwarding is a migration aid, not implementation progress.

Follow the [parallel execution plan](EXECUTION_PLAN.md) and
[task/evidence ledger](work_items.json) for current ownership, dependency gates
and integration order.

`api.json` is generated from exported names in scikit-image 0.26.0. It records
455 callable names across 18 modules, including the experimental `future`
namespace and four callable deprecated binary
morphology functions that upstream omits from its export list. Other nested namespaces, plugin APIs,
and later upstream additions still need inventory review. Static discovery also
records 284 member paths across 29 exported classes, including inherited methods
and properties. This establishes no numerical parity: constructors, methods,
properties, instance fields and runtime-generated attributes still need complete
implementation classification, state/parity cases and appropriate benchmarks.

For every native entry:

1. Capture upstream signature, dtype conversion, output/alias behavior, warnings,
   errors, masks, axes, boundary rules, and special values in executable cases.
2. Implement C++ with a portable path, CPU dispatch, and optional parallel work
   controlled by the process-wide worker limit (serial by default).
3. Run reference parity tests with fallback disabled, both SIMD enabled/disabled,
   and one/multiple workers. Test the same pipeline with either backend.
4. Benchmark original scikit-image and native execution with matching options,
   representative shapes/dtypes, allocation costs, and recorded raw samples.
   Retain regressions; investigate them rather than removing adverse workloads.
5. Update the registry, inventory, documented native input contract, and CI.

Current native coverage includes N-D low-contrast detection with typed percentile interpolation, broadcast compression and SIMD selection; raw N-D local-block means with virtual padding, dtype/layout preservation and SIMD block reductions; N-D integral images with ordered prefix sums, modular integer arithmetic and physical-layout preservation, plus direct strided integral-window queries with exact corner order; N-D Laplace, Farid derivatives and Roberts cross filters with eroded masks.
Other native coverage: RGB/gray and RGBA conversion, grayscale alpha expansion; N-D Gaussian, difference of Gaussians and unsharp masking; Sobel, Prewitt, and
Scharr with directional variants; uint8 Otsu; mean, Yen, Triangle, ISODATA, minimum, Niblack and Sauvola thresholding; inversion;
gamma/log/sigmoid correction; intensity rescaling; uint8 histogram, CDF, and
histogram equalization; seven dtype conversion APIs; YUV, YIQ, YPbPr, YCbCr, YDbDr,
CIE RGB and XYZ conversion pairs; HSV, Lab/XYZ/sRGB, Lab/LCh and Luv/XYZ/sRGB conversions;
six grayscale and four binary morphology functions.
Native stain separation/reconstruction and HED presets cover unbatched 3×K real matrices.
CIE76, CIE94, CIEDE2000 and CMC support raw Lab dtype promotion, broadcasting and extra channels;
color-space routing composes the native kernels for all nine upstream spaces.
Label-to-RGB overlays and region averages use native ranks, palette compositing,
typed group reductions and optional global workers.
Current implementation and pending counts are generated in the
[API implementation inventory](README.md), which is the authoritative per-entry
coverage record for all 455 names. The `future` namespace is available through explicit
compatibility dispatch, without adding native algorithm coverage.
Each has partial native input coverage with a dedicated parity and benchmark
case. Their unsupported options still require upstream compatibility fallback.

The NumPy 1.x color-distance promotion gaps are fixed in native code. CIE76/CIE94
scalar paths preserve float32 differences/chroma before promoted double math;
CIE94 also preserves NumPy-float32 weight-product rounding. CIEDE2000 selects a
mixed-precision SIMD/scalar path when NumPy promotes `c7 + 25**7`: chroma/hue
become double while lightness and in-place output accumulation remain float32.
Runtime dtype probes select both behaviors. The eight existing NumPy 1.24
CIEDE2000 failures now pass without relaxing tolerances or nonfinite masks.
Minimum-version CI includes the complete dtype-metadata and distance test files,
plus packet/tail, signed-layout, branch and zero-weight regression cases. Other
color algorithms and the rest of the API still need complete minimum-version
coverage; these scoped passes do not establish whole-library compatibility.

Next work:

- Broaden color conversion input/option parity and address CIE76 scalar dispatch
  overhead. The current inventory has no entirely upstream-only color entries;
  their native contracts remain partial.
- Extend exposure coverage to adaptive equalization and broaden histogram matching, with
  per-function parity and one-worker performance checks against scikit-image.
- Optimize histogram-matching layout preparation and continuous float64 ranks,
  and investigate the signed-16-bit regressions retained by the storage experiment.
  Removing unused ranks and specializing floating storage improved the previous
  build by 1.124× on Windows and 1.168× on minimum NumPy in independent one-worker
  runs. Eight Windows cases and three minimum-NumPy cases remain slower than
  original scikit-image; adaptive dense-vector reservation was rejected after an
  inconclusive result and integer-layout regressions. Keep the fixed workloads
  and preserved packages; see `benchmarks/match-memory.md`.
- Broaden distance parameter-array, strong-scalar and float16 contracts;
  extend direct nonuniform-row processing to the weighted distance kernels.
- Broaden local-threshold exceptional arithmetic, float16, array parameters
  and overflowing typed-window products; reuse ordered prefix infrastructure
  for integral-image APIs and additional local statistics.
- Broaden histogram thresholds to remaining integer widths, float16 and extreme
  arithmetic-warning modes; share the floating histogram kernel with exposure.
- Complete dtype, dimension, boundary, mask, output-buffer, and channel handling
  for existing kernels; improve reduction parity for extreme dynamic ranges.
- Broaden morphology NaN/output-dtype parity and optimize remaining memory-bound
  conversions, strided color transforms and other measured regressions.
- Broaden stain batched/vector matrix, float16 and warning contracts, and improve
  remaining uint16 separation layout regressions and custom-output-width costs.
- Complete float16/endian conversion coverage and floating warning contracts;
  port remaining exposure operations, broaden morphology's exceptional-value and
  output-dtype contracts, and add connected-component labeling.
- Measure and tune the registered resize/rescale pipeline against scikit-image.
  Spline coefficients, typed grid zoom, raw Gaussian anti-aliasing, typed ranges and
  output clipping now have independent native fixtures and Python parity
  checks. Integer anti-aliasing
  stores back into the source dtype after every axis, including full-width
  casting behavior. Clipping preserves all integer bound bits and uses explicit
  policies for the three observed NumPy signed-zero/NaN behaviors. Range scans
  preserve integer precision and report NaNs/zero signs. The Python bridge uses
  NumPy's compiled reduction only for observable mixed-zero ties or all-NaN
  payload ordering; ordinary ranges use the C++ scan. Raw Half identity has a
  separate SIMD finish preserving zero ties and NaN payloads. Public
  argument/warning parity passes on Windows and both Linux dependency profiles,
  with scalar/SIMD execution and global worker caps one/four. Isolated one-worker
  end-to-end benchmarks and bottleneck tuning remain pending; registration is
  not evidence of measured speedup.
- Port geometric transforms, measurements, features, restoration, registration,
  segmentation, and graph operations with representative per-function cases.
- Extend to nested APIs/classes and audit the inventory against upstream docs.
- Expand cross-platform validation and hardware-specific performance gates
  after enough repeat measurements establish stable thresholds.
