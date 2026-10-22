# CLAHE performance experiments

The initial native `equalize_adapthist` source is retained on main after full final-source CI. These results do not establish parity or performance for every possible input.

[Initial run 39847406392](https://github.com/bipxx/scikit-imagexx/actions/runs/39847406392) measures source `d7a79ef`, with 40 fixed cases per OS at worker limits one/four (80 rows per OS), size512, seed6603 and seven samples. Both libraries are warmed; allocation, dtype conversion, Python dispatch and output handling are timed. Reference pools are fixed to one and native fallback is disabled. All timed outputs, including RGB, are exactly equal in this run; the general RGB parity budget remains the established HSV budget. Downloaded measured binary hashes agree with both report and source manifests.

| OS | one-worker upstream/native geometric mean | minimum one-worker speedup | cases slower than upstream |
|---|---:|---:|---:|
| Linux | 9.746494x | 2.702971x | 0 / 40 |
| Windows | 13.461295x | 3.986180x | 0 / 40 |

Raw samples: [Linux](clahe-initial-linux.json), [Windows](clahe-initial-windows.json). Worker limits are caps: small cases remain serial below the activation threshold. The global default remains one.

The C++ implementation avoids padded image copies, builds histograms from contiguous row interiors, terminates redistribution when every bin is saturated, and preserves the original quantization and ordered float32 interpolation. AVX2 handles quantization and multilinear lookup/interpolation; global workers optionally partition independent stages. Native scalar, portable and ASan/UBSan tests cover both output precisions and large one-versus-four-worker results. Dedicated Python parity passes858 cases per SIMD mode on Windows and minimum NumPy.

The lowest speedups are floating Fortran layouts. Their current validation/conversion path can create redundant contiguous copies; this is a measured optimization target, not a reason to omit those cases.

| OS | dtype | case | upstream ms | native ms | upstream/native |
|---|---|---|---:|---:|---:|
| Linux | float64 | fortran | 13.407481 | 4.960276 | 2.702971x |
| Linux | float32 | fortran | 12.932230 | 4.220669 | 3.064024x |
| Linux | float64 | reversed | 8.133134 | 1.839448 | 4.421507x |
| Linux | float32 | reversed | 7.543748 | 1.689722 | 4.464491x |
| Linux | uint8 | unclipped | 12.701305 | 2.823556 | 4.498337x |
| Windows | float64 | fortran | 30.544700 | 7.662650 | 3.986180x |
| Windows | float32 | fortran | 28.345600 | 5.015800 | 5.651262x |
| Windows | float64 | reversed | 15.670700 | 2.502180 | 6.262819x |
| Windows | float32 | reversed | 14.113300 | 1.904733 | 7.409594x |
| Windows | float64 | volume | 83.794800 | 10.909700 | 7.680761x |

Source fingerprint and exact package hashes belong to this initial source only. Later optimization candidates retain separate raw samples and use the same fixed corpus with paired native controls.


## Retained source

[Full source CI39847405130](https://github.com/bipxx/scikit-imagexx/actions/runs/39847405130) passes all three jobs, including minimum NumPy, both OSes, SIMD enabled/disabled and native/portable checks. Source `d7a79ef` is merged as [1ddd03f](https://github.com/bipxx/scikit-imagexx/commit/1ddd03f). The merged product fingerprint equals the exact measured source. The validated Windows wheel is installed in the workspace environment, with its Python sources and binary hash checked and numerical smoke checks at worker caps one/four. Earlier merge-gate language records the chronological trial state.
