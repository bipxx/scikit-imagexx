# CIE76 integer-load optimization

Historical fixed experiment: five uint8 CIE76 workloads, 512 squared pixels, one worker, allocation-inclusive calls. The same seed, layouts, dtype promotion and seven warmed samples are used in both snapshots. Parity is checked against original scikit-image before timing, with fallback disabled.

Baseline `c98d286`; candidate `f2910d0`. The mutable kernel surface is `src/simd_color_avx2.cpp`. [Raw samples](distance-tuning.json) retain both complete 84-case snapshots. These snapshots are excluded from the [current-source aggregate](STATUS.md).

Geometric-mean native latency: **2.6574 ms to 0.7389 ms**, a **3.60x** improvement over the prior C++ code.

The first implementation converted integer lanes individually. The candidate loads packed bytes, converts four raw integer lanes to doubles with SIMD, broadcasts constant channels, and uses vector float-to-double conversion for mixed inputs. Twelve-byte interleaved loads avoid reading past the final pixel. The remaining arithmetic is unchanged.

| Workload | Previous C++ ms | New C++ ms | C++ improvement | Original / new |
|---|---:|---:|---:|---:|
| contiguous | 2.4113 | 0.6321 | 3.81x | 4.68x |
| channel-first | 2.3747 | 0.5917 | 4.01x | 3.41x |
| strided | 3.1917 | 1.4142 | 2.26x | 3.09x |
| broadcast | 2.9957 | 0.6836 | 4.38x | 2.86x |
| extra-channels | 2.4207 | 0.6092 | 3.97x | 10.59x |

The full candidate snapshot also verifies mixed dtypes and composed color routes. All 39 one-worker processing cases measured faster than the original in that run; three RGB identity cases are labeled separately. Performance depends on the machine and system load.
