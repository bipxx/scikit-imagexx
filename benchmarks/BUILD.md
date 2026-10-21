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
`src/simd_morphology_avx2.cpp` use the same flags;
there is no FMA requirement or fast-math option. Color arithmetic uses a uint8
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

The current function benchmark covers 65 native image functions, two metadata
helpers, and 772 per-function performance cases. The suite now has 13397 checks in both SIMD
modes; 450 inventory availability/signature checks still do not establish
numerical parity for unported functions. `exposure-parameters.json` records
24 additional one-thread measurements with non-default options. All checked-in
performance results retain slower cases as well as faster ones.
`conversion-widths.json` records 112 additional cases across boolean and
8/16/32/64-bit integer inputs, with exact parity required before timing. Identity,
copy, and metadata rows are identified separately from conversion kernels.
`color-layouts.json` adds 104 one-worker comparisons on uint16/float32 strided and
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

These are local measurements under ordinary system load, without CPU affinity
or frequency locking. Medians and individual samples are available in the
JSON; small differences should not be treated as stable performance guarantees.
