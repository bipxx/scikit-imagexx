# Current performance verification

Reference: original scikit-image 0.26.0, NumPy 2.5.4. Local Windows/MSVC Release
build on an Intel Core i7-12700K. All three reports below use source SHA-256
`83c35626701322bd2a90fe0d370078ffca3f7ccfc26683234253248fda4de825`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 328 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |

Each case verifies values and dtype against original scikit-image before timing
with upstream fallback disabled. Conversion-width cases also require exact values
and matching identity behavior. Timing includes allocation and Python dispatch;
seven warmed samples, shuffled execution order, and one-thread reference pools.
Input generation and I/O are excluded. Raw samples are in the companion JSON files.

For example, 1024² uint8 images with one C++ worker measured:

| Operation | Original ms | C++ ms | Reference / C++ |
|---|---:|---:|---:|
| Gaussian, default sigma=1 | 24.2552 | 5.2935 | 4.58x |
| Sobel magnitude | 25.6046 | 3.2302 | 7.93x |
| Otsu | 3.3625 | 0.2541 | 13.23x |

Across the reports, 218 of 225 one-worker native processing workloads measured
faster than the reference. These are workload cases, including repeated functions
and parameter variants. Identity returns and dtype metadata are excluded from
that count and labeled separately in the full reports.

Seven one-worker processing workloads remained slower:

| Operation | Input | Size | Reference / C++ |
|---|---|---:|---:|
| invert | float64 | 256² | 0.56x |
| img_as_bool | float64 | 256² | 0.70x |
| img_as_float64 | float32 | 1024² | 0.93x |
| img_as_bool | float64 | 1024² | 0.83x |
| img_as_float64 | bool | 1024² | 0.95x |
| img_as_bool | int8 | 1024² | 0.93x |
| img_as_int | uint16 | 1024² | 0.67x |

Remaining optimization work includes reducing dispatch/packing costs for simple
conversions and eliminating unnecessary work in 16-bit narrowing. Output
expansion and memory traffic are candidate limits for float casts; this run does
not include hardware-counter profiling to establish their exact contribution.
32/64-bit integer-to-float conversion still uses portable loops. Other conversion
paths now use explicit AVX2 comparisons, scaling, packing, and validation maxima.

Measurements use ordinary system load without affinity or frequency locking.
Small differences near 1x can change between runs. These results establish
measured comparisons on this machine, not universal speed guarantees. CI runs
parity and benchmark smoke checks on Windows/Linux; it does not assert speed
ratios on shared runners. Full native API parity remains in progress, as tracked
in the [inventory](../coverage/README.md) and [roadmap](../coverage/ROADMAP.md).
