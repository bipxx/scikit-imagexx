# Conversion layout optimization

The fixed 52-case suite measures boolean conversion across 11 real dtypes and four layouts, plus full-range and narrow uint16-to-int16 conversion in the same layouts. Four boolean identity controls are labeled separately; 48 cases process pixels. The primary metric is the geometric mean public-call latency across all 52 one-worker calls. The [complete history](conversion-packet-tuning.json) retains raw samples, source/binary hashes, warnings and all adverse observations.

| Observation | Latency geometric mean ms | Original/current | Initial/current,paired |
|---|---:|---:|---:|
| initial | 0.261196 | 0.4751x | — |
| fortran_order | 0.147748 | 0.9239x | 1.9760x |

The baseline forced every input into C order. Fortran boolean conversion now reads physical storage directly and allocates a Fortran output without reading pixels during allocation. Unscaled uint16 narrowing also preserves upstream Fortran order; its scaled branch explicitly returns C order. Unaligned inputs receive an aligned copy in the required order. Existing C++ kernels and global worker control remain unchanged.

The initial package and its exact 52-case measurement were preserved before editing. The paired harness validates measurement, source and binary hashes, then randomizes all three backends. [Parameter results](conversion-packets.md) include every case; speedups below 1 remain visible. The aggregate improves despite regressions in some individual non-Fortran cases.

Validation passed 5,284 selected Windows tests in each SIMD mode and 4,132 conversion/non-distance metadata cases on minimum NumPy.84 new physical-layout assertions failed before the fix and pass afterward. Additional raw C++ packet/tail/parallel oracles passed on the original kernels before further SIMD experiments. A separate expanded minimum-NumPy audit exposed 16 preexisting CIE76/CIE94 scalar dtype failures, reproduced with the archived pre-change wheel and tracked in the roadmap. No dtype checks were relaxed.

This is a focused conversion checkpoint. The 21-report whole-library checkpoint at `984141f` predates these binding changes and remains historical until refreshed.
