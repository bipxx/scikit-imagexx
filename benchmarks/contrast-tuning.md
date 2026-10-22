# Low-contrast optimization experiment

The port reads raw real storage and preserves upstream percentile decisions, input precision during interpolation, warning behavior and optional global worker limits. It reuses the existing native RGB/RGBA conversions. Every benchmark times the public function, including dispatch, conversions, allocation and buffer destruction.

| Observation | Native geometric mean ms | Original/current | Preserved/current | Faster than original |
|---|---:|---:|---:|---:|
| initial | 0.973612 | 1.3736x | — | 34/126 |
| row_histogram_endpoints | 0.469860 | 2.7018x | 1.9955x | 61/126 |
| avx2_selection | 0.133659 | 9.2250x | 6.8441x | 125/126 |
| single_allocation | 0.190380 | 7.8753x | 5.7263x | 125/126 |
| separate_uninitialized | 0.125216 | 9.9142x | 7.3815x | 126/126 |
| independent_confirmation | 0.133494 | 9.4186x | 7.0812x | 126/126 |

Independent confirmation measures every one of the 126 cases faster than original with either worker limit. Four workers achieve 10.3708x original across their separate 126 cases; the fixed metric above remains one worker. The final source fingerprint is `acb2ae0434191c9257218d9191ea4e23d9fba4e2a701d233147951ec4aeb87a0`.

All headline rows use one worker. The [complete JSON](contrast-tuning.json) retains every raw sample, source fingerprint, binary hash and slower case. The initial aggregate was dominated by broadcast inputs despite 92 slower cases; individual regressions drove the next experiments. Row-wise normalization removed per-element multidimensional divisions. Eight-bit histograms and endpoint extrema scans avoided sorting. AVX2 three-way partitioning then addressed general selection, including duplicate-heavy data and unsigned 64-bit order.

The single-allocation experiment regressed the paired ratio and was rejected. Separate buffers retain the validated selection kernel; unnecessary clearing was removed. A boolean prefix can prove mixed truth values before allocation or OpenMP setup; a constant prefix still requires checking the rest of the image. The extended-precision parameter guard is a correctness fix, retained independently of timing outcomes.

Seven warmed medians use randomized backend order. Absolute timings vary between sessions, so claims over the initial C++ port use paired ratios. The preserved baseline is the exact installed package identified by its initial measurement, not a rebuilt wheel. `tools/preserve_baseline.py --installed-package ... --measurement ...` checks its source fingerprint, Python sources and measured binary hashes before preserving it.

Parity covers all twelve real input dtypes, signed/zero/unaligned/endian strides, scalar and N-D inputs, dtype tags, ignored method/boolean arguments, RGB/RGBA routes, exact decision boundaries, duplicate-heavy selection, packet tails, large worker splits, warning/callback behavior and explicit fallback. Native sort oracles check guarded storage across 32-/64-bit lanes; Linux address/undefined-behavior sanitizers check AVX2 and scalar dispatch. Portable C++ and minimum-NumPy CI checks are recorded with the final source.

The registered API, dtype, inventory, baseline-tool and contrast selection passed 7,428 selected tests in each SIMD mode on Linux, and 7,426 plus two long-double platform skips in each mode on Windows. [Core candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39700827035) passed its complete cross-platform SIMD, portable/native C++, dynamic-team and minimum-NumPy matrix. [Registered candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39704346577) includes the final color safeguards and registry; its minimum-NumPy job has passed and the complete matrix is still running at this publication checkpoint.

A final regression suite found fourteen color decisions changed by a grayscale rounding bit. The final wrapper validates bounded floating color inputs and delegates decisions inside a conservative color-rounding margin, including narrow-threshold promotion. This mandatory correctness guard keeps native execution for the fixed ordinary color workloads.

Nonfinite or risky floating arithmetic, unsupported subclasses/complex input, nondefault underflow/callback policies, extended-precision percentile parameters, unsupported color widths/byte order, floating color inputs outside [-1,1], and ambiguous color decisions use explicit fallback. This is partial native coverage. Speedups describe the fixed seeded workloads on the recorded machine.

The independent confirmation includes four-worker limits in its raw report; these are separate from the fixed one-worker metric. [Parameter results](contrast-parameters.md) record their explicit worker limits and all measured cases.

Final cases slower than original scikit-image:

| Input | Workload | Worker limit | Original/current |
|---|---|---:|---:|

None in this confirmation.

Final one-worker cases slower than the initial C++ build:

| Input | Workload | Preserved/current | Original/current |
|---|---|---:|---:|
| bool | endian | 0.9939x | 2.7484x |
| int8 | broadcast | 0.9149x | 103.9645x |
| float64 | broadcast | 0.9329x | 83.3870x |
