# Native implementation roadmap

Objective: iterate through the scikit-image API, implement native CPU kernels,
preserve Python call contracts, and verify both parity and performance against
the original library. The project remains incomplete until that work is done;
upstream forwarding is a migration aid, not implementation progress.

`api.json` is generated from exported names in scikit-image 0.26.0. It records
446 callable names across 17 modules. Other nested namespaces, plugin APIs,
and future upstream additions still need inventory review. Class APIs also
need constructor, method, and state parity cases beyond their top-level name.

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

Current native coverage: RGB/gray conversion; Gaussian; Sobel, Prewitt, and
Scharr with directional variants; uint8 Otsu; mean thresholding; inversion;
gamma/log/sigmoid correction; intensity rescaling; uint8 histogram, CDF, and
histogram equalization; seven dtype conversion APIs. There are 29 partial native
image entries, one dtype metadata helper, and 416 pending entries in the current
446-name inventory.
Each has partial native input coverage with a dedicated parity and benchmark
case. Their unsupported options still require upstream compatibility fallback.

Next work:

- Complete dtype, dimension, boundary, mask, output-buffer, and channel handling
  for existing kernels; improve reduction parity for extreme dynamic ranges.
- Complete float16/endian conversion coverage and floating warning contracts;
  port remaining exposure operations, followed by native
  binary/grayscale morphology and connected-component labeling.
- Port geometric transforms, measurements, features, restoration, registration,
  segmentation, and graph operations with representative per-function cases.
- Extend to nested APIs/classes and audit the inventory against upstream docs.
- Expand cross-platform validation and hardware-specific performance gates
  after enough repeat measurements establish stable thresholds.
