# Recorded build

The checked-in `results.json` was generated locally on Windows 11 with an
Intel Core i7-12700K (12 physical cores, 20 logical processors), using:

- CPython 3.13.14, 64 bit.
- MSVC 19.44.35225.0, Visual Studio 2022 Community, x64 target.
- Release configuration, C++17, OpenMP enabled (`/openmp`).
- AVX2 enabled only in `src/simd_avx2.cpp` (`/arch:AVX2`, `/fp:strict`).
- Baseline code compiled with `/fp:precise`; no fast-math flag.
- Package and numerical dependency versions in `requirements.txt`.

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

These are local measurements under ordinary system load, without CPU affinity
or frequency locking. Medians and individual samples are available in the
JSON; small differences should not be treated as stable performance guarantees.
