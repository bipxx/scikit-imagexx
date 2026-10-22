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


## Float-buffer reuse trial (`2b6e454`)

[Paired run39850919881](https://github.com/bipxx/scikit-imagexx/actions/runs/39850919881) compares the preserved initial package against the current adapter, retaining all40 case definitions and80 rows per OS. Both packages pass timed-case parity. No C++ source, public headers or build configuration changed. The candidate makes one contiguous grayscale float preparation serve bounds and conversion, releasing that temporary before the core.

| OS | one-worker upstream/current geometric mean | paired initial/current geometric mean | float32 Fortran initial/current | float64 Fortran initial/current |
|---|---:|---:|---:|---:|
| Linux | 9.785597x | 1.011057x | 1.244758x | 1.388248x |
| Windows | 14.073307x | 1.013340x | 1.192723x | 1.359699x |

Raw reports: [Linux](clahe-layout-linux.json), [Windows](clahe-layout-windows.json). The minimum paired corpus score is1.011057x, passing the fixed performance gate. [Full candidate CI39850919677](https://github.com/bipxx/scikit-imagexx/actions/runs/39850919677) passed all three jobs; the optimization was retained and merged into main at `31c7fde`. Dedicated parity passes882 cases per SIMD mode on both runtimes, and minimum API/metadata4729 cases pass.

Fortran float targets improve19–39% against same-run controls. Several other controls, including unchanged integer/RGB paths, are slower in this trial; no claim is made that the change improves every case. The largest recorded native-control regression is about11% in Linux uint8 contiguous data. All samples remain in the raw reports. The table shows the five lowest native-control ratios per OS:

| OS | dtype | case | initial/current |
|---|---|---|---:|
| Linux | uint8 | contiguous | 0.886587x |
| Linux | uint8 | rgb | 0.893977x |
| Linux | uint16 | tiny | 0.980951x |
| Linux | uint16 | contiguous | 0.989375x |
| Linux | uint8 | large-kernel | 0.995097x |
| Windows | float32 | rgb | 0.932583x |
| Windows | uint16 | rgb | 0.966118x |
| Windows | uint16 | fortran | 0.971907x |
| Windows | uint16 | low-clip | 0.980550x |
| Windows | float64 | bins64 | 0.981356x |
