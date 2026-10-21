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

The current function benchmark covers 77 native image functions, two metadata
helpers, and 916 per-function performance cases. The suite now has 18025 checks in both SIMD
modes; 450 inventory availability/signature checks still do not establish
numerical parity for unported functions. `exposure-parameters.json` records
24 additional one-thread measurements with non-default options. All checked-in
performance results retain slower cases as well as faster ones.
`conversion-widths.json` records 112 additional cases across boolean and
8/16/32/64-bit integer inputs, with exact parity required before timing. Identity,
copy, and metadata rows are identified separately from conversion kernels.
`color-layouts.json` adds 120 one-worker comparisons on uint16/float32 strided and
channel-first images; input movement, normalization and any required copying are
included. Uniform floating layouts are read directly without a full copy.
`perceptual-parameters.json` adds 60 cases on alternate white points, wrapped
hues, four-channel polar conversions, negative-Z warning counts and Luv lightness branches. Warnings
are checked before timing and ignored during timed calls in that report.
`morphology-parameters.json` adds 240 comparisons across all ten new APIs,
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
`stain-parameters.json` adds 156 comparisons over 11 standard families and
custom 3×1/3×7 matrices, with input-precision and coefficient-scaled parity
budgets. `linear-color-tuning.json` is a separate historical 16-case experiment
from `aa6e251` to `0316dba`, where geometric-mean native latency fell from
4.1235 to 0.7731 ms by skipping unused nonlinear vector stages.

These are local measurements under ordinary system load, without CPU affinity
or frequency locking. Medians and individual samples are available in the
JSON; small differences should not be treated as stable performance guarantees.

`color-distance.json` adds 246 comparisons for all four Lab distance metrics and layouts, mixed dtypes and
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

`threshold-parameters.json` adds 152 comparisons for Yen, Triangle and ISODATA,
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
