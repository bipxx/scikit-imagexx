# Recorded build

The checked-in `results.json` was generated locally on Windows 11 with an
Intel Core i7-12700K (12 physical cores, 20 logical processors), using:

- CPython 3.13.14, 64 bit.
- MSVC 19.44.35225.0, Visual Studio 2022 Community, x64 target.
- Release configuration, C++17, OpenMP enabled (`/openmp`).
- AVX2 enabled only in `src/simd_avx2.cpp` (`/arch:AVX2`, `/fp:strict`).
- Baseline code compiled with `/fp:precise`; no fast-math flag.
- Package and numerical dependency versions in `requirements.txt`.

The current build also uses those AVX2/strict floating-point flags for
`src/simd_convert_avx2.cpp`. `src/simd_color_avx2.cpp` and
`src/simd_morphology_avx2.cpp` use the same flags, as do the stain kernels in
`src/simd_stain_avx2.cpp`. There is no FMA requirement or fast-math option. Color arithmetic uses a uint8
decode LUT and double-precision vector log/exp range reduction for sRGB powers.
Portable image and conversion code retains the
baseline compiler target; AVX2 is selected only after runtime CPU/OS checks.

The Python wheel was built and installed using:

```powershell
.\.venv\Scripts\python -m pip install . --no-build-isolation
.\.venv\Scripts\python benchmarks/run.py --output benchmarks/results.json
```

Initial validation passed 444 Python parity cases with AVX2 enabled
and the same 444 cases with runtime dispatch disabled. The optimized version
passes **520 cases** in each mode, including output tails and unaligned inputs.
Both the default
standalone C++ build and a build without AVX2/OpenMP passed CTest. A
Windows/Linux CI workflow is included; remote CI execution was not part of
this local run.

`optimized-single-thread.json` compares the optimized code to scikit-image and
a preserved pre-optimization extension in the same process. Its companion
Markdown report includes both comparisons. The previous binary's SHA-256 is
recorded; it is retained locally under ignored `build/baseline`, and is not
shipped in the repository. `histogram-results.json` also covers constant,
binary, skewed, and late-outlier input distributions. The older reports are
historical source snapshots and should not be mixed as simultaneous samples.

The current function benchmark covers 88 native image functions, two metadata
helpers, and 524 one-worker per-function performance cases. The suite now covers 25329 checks in both SIMD
modes; 450 inventory availability/signature checks still do not establish
numerical parity for unported functions. `exposure-parameters.json` records
24 additional one-thread measurements with non-default options. All checked-in
performance results retain slower cases as well as faster ones.
`conversion-widths.json` records 56 one-worker cases across boolean and
8/16/32/64-bit integer inputs, with exact parity required before timing. Identity,
copy, and metadata rows are identified separately from conversion kernels.
`color-layouts.json` adds 120 one-worker comparisons on uint16/float32 strided and
channel-first images; input movement, normalization and any required copying are
included. Uniform floating layouts are read directly without a full copy.
`perceptual-parameters.json` adds 30 one-worker cases on alternate white points, wrapped
hues, four-channel polar conversions, negative-Z warning counts and Luv lightness branches. Warnings
are checked before timing and ignored during timed calls in that report.
`morphology-parameters.json` adds 120 one-worker comparisons across all ten new APIs,
four input dtypes, large rectangles, disks and asymmetric even footprints.
The latter reuse output buffers; all require exact values and dtype, and binary
deprecation warnings are verified before timed calls. Rectangle filtering uses
linear-work extrema scans above the small-stencil threshold. Binary SIMD stencils
stop when every lane reaches its terminal value, and top-hat subtraction is vectorized.
Canonical boolean rectangles now pack pixels into words, apply bitwise extrema,
and unpack with SIMD; other storage/modes retain the general native kernel.
`morphology-tuning.json` records the historical 18-case experiment from kernel
`3b4355a` to `2795b77`: geometric-mean native latency fell from 3.6556 to
0.1891 ms. It is excluded from the current-source report aggregate.
`stain-parameters.json` adds 78 one-worker comparisons over 11 standard families and
custom 3×1/3×7 matrices, with input-precision and coefficient-scaled parity
budgets. `linear-color-tuning.json` is a separate historical 16-case experiment
from `aa6e251` to `0316dba`, where geometric-mean native latency fell from
4.1235 to 0.7731 ms by skipping unused nonlinear vector stages.

These are local measurements under ordinary system load, without CPU affinity
or frequency locking. Medians and individual samples are available in the
JSON; small differences should not be treated as stable performance guarantees.

`color-distance.json` adds 123 one-worker comparisons for all four Lab distance metrics and layouts, mixed dtypes and
representative composed color routes, including identity and out-of-gamut HSV.
The latter verify each stage and a propagated ratio bound; identity rows are
excluded from native processing speedup counts. CIE76 uses a fused one-pass
SIMD kernel, raw integer casts and the process-wide worker limit.
`distance-tuning.json` retains the historical five-case uint8 experiment from
`c98d286` to `f2910d0`: geometric-mean native latency fell from 2.6574 to
0.7389 ms. Its snapshots are excluded from the current-source aggregate.

The weighted distances preserve float32 operation order and promoted trig stages.
NumPy AVX-512 atan2/power differences require the documented float32 CIEDE2000
error budget; float64 comparisons retain the base budget. CMC scalar output
retains its float32 intermediate budget while promoting the final accumulation.
CIE76 nonuniform layouts now pass row offsets to C++ and avoid full image copies.
`distance-layout-tuning.json` records the historical fixed one-worker experiment
from `83402fc` to `f3b3ff9`: native median latency fell from 1.6552 to 0.7387 ms.
Its two 123-case snapshots are excluded from the current-source aggregate.

`threshold-parameters.json` adds 76 one-worker comparisons for Yen, Triangle and ISODATA,
including four layouts, two non-default floating bin counts, constant inputs,
ISODATA all-threshold output and supplied histograms. Histogram counts and selected
thresholds require exact reference parity; warning text/order and output dtype
are checked before timing. The C++ float32 log approximation preserves NumPy's
near-tie behavior without requiring a hardware FMA instruction. Pixel scans and
floating bin assignment use the SIMD/runtime-worker paths; selectors operate on
small histogram buffers.

Yen's compatibility logarithm has an optional eight-lane AVX2/FMA path in
`src/simd_threshold_fma.cpp`. It is dispatched only after the runtime AVX2/OS
checks and a separate FMA CPU-feature check. CPUs without FMA and builds with
SIMD disabled use scalar `std::fma`; the library has no hardware FMA requirement.
Only this translation unit uses the additional `-mfma` flag on GCC/Clang.

`local-threshold-parameters.json` adds 128 one-worker Niblack/Sauvola comparisons. Two
float64 integral buffers share reflected input preparation and ordered prefix
passes; output math preserves float32 casts before division and the original
variance clipping. AVX2 operates across independent prefix columns and output
pixels without reassociating each cumulative sequence. All 11 supported real
input types are included; additional cases cover anisotropic windows, size 127,
Fortran/reversed/stepped storage, singleton-depth volumes and strong scalars.

Alpha-family validation adds 1020 dedicated parity cases and direct C++ byte-copy,
compositing, reversed-stride and empty-buffer checks. `alpha-parameters.json`
records 184 one-worker comparisons against original scikit-image,
covering all eleven compositing numeric dtypes, float16 grayscale expansion,
planar/Fortran/negative/stepped strides, endian storage, backgrounds and alpha
broadcasting. `src/simd_alpha_avx2.cpp` uses the same runtime-dispatched AVX2
and strict floating-point flags as the existing color kernels.

The Laplace/Roberts family adds 2894 dedicated cases, including all 12 real image
dtypes, nonnative storage, memory-mapped arrays, N-D Laplace, masks and singleton
axes. The full 23593-case suite passes with SIMD enabled and disabled. Linux
passes the family and fallback-policy checks in both modes; address/undefined
sanitizer C++ checks also pass. Native/portable CTest covers typed buffers,
normalization, half decoding, reflected edges, byte swapping and signed zeros.
`edge-parameters.json` records 260 one-worker comparisons across every numeric
input dtype, ordinary/strided/Fortran/unaligned/thin layouts, N-D volumes and
broadcast masks. The fixed column experiment retains initial and confirmation
measurements separately from the current-source aggregate. The new SIMD source
is `src/simd_small_edge_avx2.cpp`, compiled with the existing strict AVX2 flags.

The descriptor compatibility suite adds 538 checks across every covered default
entry point, equivalent native descriptors, copy/identity behavior and rejection
of opposite-endian typed buffers. All 23593 tests pass in both SIMD modes.
Linux passes the 3432 descriptor/edge cases in both modes. The dtype fix changes
binding comparisons; image arithmetic kernels and their native/portable C++
and sanitizer validation remain unchanged. The separate metadata experiment
records allocation-inclusive one-worker validation-overhead measurements.


Farid adds 1658 dedicated checks across 12 real input dtypes, reflected/constant/
nearest/mirror/wrap boundaries and grid aliases, ordered/repeated/negative axes,
N-D kernels through rank eight, high-amplitude constant cancellation, eroded masks,
endian descriptors, memory maps, subnormal policies and parallel skinny images.
Both full Windows SIMD and scalar runs pass 25329 tests. Linux passes 4054
Farid/shared-API/descriptor checks in both modes. Native and portable CTest pass on Windows; Linux C++ checks
pass with address and undefined-behavior sanitizers.

`farid-parameters.json` records 243 one-worker comparisons, with the same
allocation/dispatch/conversion costs and strict no-fallback provenance checks.
The full ordered coefficient kernel retains upstream convolution cancellation;
AVX2 works across independent output pixels without changing each sum's order.
Float32 gradients round before squaring, addition, square root and the typed
sqrt(ndim) divisor. `src/simd_farid_avx2.cpp` uses the existing strict AVX2 flags;
the global optional worker limit partitions the same row and output stages.

The fixed nine-case empty-axis allocation experiment retains baseline, two
optimizations and confirmation in `farid-empty-tuning.json`. Owned zero allocation
and omission of unused input copies reduce geometric-mean native latency by
2.79x initially and 2.45x in confirmation. Reference drift and slower observations
are reported; the small dispatch-only increment is not claimed as stable.
The initial 243-case profile is retained separately from the current aggregate.

A further 240 cases check dtype metadata on every real input type and native/
endian storage. Generic Farid preserves tags on native float32/64 output;
directional SciPy convolution drops them. Empty-axis allocation follows the
same generic rule. The metadata-compatible confirmation is retained as a fifth
fixed-workload observation, with its own source fingerprint and raw samples.

The final Farid source snapshot (`8f0a084`) also passes the complete
Windows/Linux CI workflow: full parity in both SIMD modes, all benchmark smoke
suites and native/portable CTest. The workflow run is
https://github.com/bipxx/scikit-imagexx/actions/runs/39620952451.
