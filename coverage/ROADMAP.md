# Native implementation roadmap

Objective: iterate through the scikit-image API, implement native CPU kernels,
preserve Python call contracts, and verify both parity and performance against
the original library. The project remains incomplete until that work is done;
upstream forwarding is a migration aid, not implementation progress.

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
There are 96 partial native image entries, two metadata helpers,
and 357 pending entries in the current
455-name inventory. The `future` namespace is available through explicit
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

- Port remaining color conversions with the same
  native parity and one-worker performance requirements. Broaden RGB/gray parity.
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
- Port geometric transforms, measurements, features, restoration, registration,
  segmentation, and graph operations with representative per-function cases.
- Extend to nested APIs/classes and audit the inventory against upstream docs.
- Expand cross-platform validation and hardware-specific performance gates
  after enough repeat measurements establish stable thresholds.
