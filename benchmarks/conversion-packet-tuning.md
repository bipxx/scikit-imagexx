# Conversion layout and word-lane optimization

The fixed 52-case workload measures boolean conversion across 11 real dtypes and four layouts, plus full-range and narrow uint16-to-int16 conversion in those layouts. Four identity controls are labeled separately from the 48 pixel conversions. The primary metric is geometric mean public-call latency over the same 52 one-worker calls. The [complete history](conversion-packet-tuning.json) retains raw samples, source/binary hashes, comparator identities, rejected code and every adverse observation.

| Observation | Comparator | Public-call geometric mean ms | Original/current | Preserved/current,paired |
|---|---|---:|---:|---:|
| initial | — | 0.261196 | 0.4751x | — |
| fortran_order | initial | 0.147748 | 0.9239x | 1.9760x |
| word_lanes | fortran_order | 0.252542 | 0.9595x | 1.0847x |
| independent_confirmation | initial | 0.224000 | 0.9506x | 1.9715x |

The binding formerly forced every input into C order. Boolean conversion now reads Fortran-contiguous inputs physically and allocates a Fortran output without reading pixels during allocation. Unscaled uint16 narrowing also preserves upstream Fortran order; scaled narrowing returns C order. Unaligned inputs receive an aligned copy in the required order.

The retained SIMD change shifts 16 unsigned words directly into signed-word output, with an eight-word packet and portable scalar tail. It avoids widening to 32 bits and saturating back to 16. A combined replacement for 64-bit boolean packing was rejected: it improved the 52-case aggregate by only 0.7% and substantially regressed several boolean cases. Disassembly confirmed that helpers were inlined. All rejected samples and the source patch remain in the history.

Initial and layout-only packages were preserved exactly as measured. Paired runs verify measurement/source/binary hashes and ISA before randomizing reference/current/preserved execution. Absolute latencies vary between sessions; keep/discard decisions use same-run paired ratios. Ratios from separate sessions should not be chained; each paired ratio identifies its own comparator.

Independent confirmation versus the initial package:

| Workers | Public calls | Original/current | Initial/current,paired | Faster than original |
|---:|---:|---:|---:|---:|
| 1 | 52 | 0.9506x | 1.9715x | 26/52 |
| 4 | 52 | 1.1058x | 2.0868x | 33/52 |

[Every parameter result](conversion-packets.md) remains visible. The serial aggregate remains slower than original scikit-image; reversed and stepped inputs can require an input copy. These are unresolved performance targets, not excluded workloads.

Validation passed 5,284 selected Windows tests in each SIMD mode and 4,132 conversion/non-distance metadata cases on NumPy 1.24. Raw C++ packet, tail and one/four-worker oracles passed Windows checks and Linux address/undefined-behavior sanitizers. The portable path passed the same fixed fixture and remains unchanged by the word-lane edit. The binding correction fixed 84 previously failing physical-layout assertions.

A GCC 13 sanitizer fixture issue was isolated in a minimal program with no library calls: selecting between two arrays directly inside a subscript constructed the wrong input values. Selecting an explicit pointer before indexing fixes the fixture; literal expected outputs and input-immutability checks remain intact. Failed probes and validation logs are archived locally.

An expanded NumPy 1.24 audit separately exposed 16 preexisting CIE76/CIE94 scalar dtype failures, reproduced with the archived pre-change wheel. They were unresolved at this conversion checkpoint and subsequently fixed by the native [scalar-distance correction](distance-scalar.md); no dtype checks were relaxed. Minimum-version CI includes the complete conversion packet/alias suite.

Word-lane [source CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39728802450) passed: Linux recorded 41,229 Python tests in each SIMD mode; Windows recorded 41,227 passed and two skipped in each mode. Native C++ tests passed in SIMD, forced-scalar and portable builds on both platforms. Minimum-NumPy CI passed 9,071 selected image-processing tests and 3,272 conversion tests. This selected minimum-version suite does not cover the unresolved scalar distance promotion failures above. The 21-report whole-library checkpoint at `984141f` predates these changes and remains separate.

Final one-worker cases slower than original or initial C++:

| API | Input | Workload | Original/current | Initial/current |
|---|---|---|---:|---:|
| util.img_as_bool | bool | contiguous | 2.0356x | 0.9768x |
| util.img_as_bool | bool | fortran | 2.0107x | 0.9337x |
| util.img_as_bool | bool | reversed | 1.8060x | 0.9313x |
| util.img_as_bool | uint8 | contiguous | 1.2541x | 0.8824x |
| util.img_as_bool | uint8 | fortran | 0.9793x | 11.1407x |
| util.img_as_bool | uint8 | reversed | 1.2922x | 0.8645x |
| util.img_as_bool | int8 | contiguous | 0.9512x | 0.9164x |
| util.img_as_bool | int8 | fortran | 0.9939x | 12.9210x |
| util.img_as_bool | int8 | stepped | 1.0908x | 0.7344x |
| util.img_as_bool | uint16 | contiguous | 0.8310x | 0.7353x |
| util.img_as_bool | uint16 | reversed | 0.7194x | 1.0856x |
| util.img_as_bool | int16 | contiguous | 0.8877x | 0.9109x |
| util.img_as_bool | int16 | fortran | 0.8643x | 27.8860x |
| util.img_as_bool | int16 | reversed | 0.8890x | 0.9443x |
| util.img_as_bool | uint32 | contiguous | 1.4424x | 0.9969x |
| util.img_as_bool | uint32 | reversed | 0.8653x | 1.0080x |
| util.img_as_bool | uint32 | stepped | 0.7448x | 1.0506x |
| util.img_as_bool | int32 | fortran | 0.9168x | 24.0636x |
| util.img_as_bool | int32 | reversed | 0.4679x | 1.0070x |
| util.img_as_bool | int32 | stepped | 0.4933x | 0.7667x |
| util.img_as_bool | uint64 | reversed | 0.2485x | 0.9266x |
| util.img_as_bool | uint64 | stepped | 0.5337x | 1.4261x |
| util.img_as_bool | int64 | contiguous | 1.0825x | 0.8680x |
| util.img_as_bool | int64 | reversed | 0.3015x | 0.9136x |
| util.img_as_bool | int64 | stepped | 0.4892x | 1.0075x |
| util.img_as_bool | float32 | contiguous | 0.9474x | 1.1132x |
| util.img_as_bool | float32 | fortran | 0.8967x | 34.8199x |
| util.img_as_bool | float32 | reversed | 0.9775x | 1.3003x |
| util.img_as_bool | float32 | stepped | 0.7811x | 0.7562x |
| util.img_as_bool | float64 | contiguous | 1.1086x | 0.8242x |
| util.img_as_bool | float64 | fortran | 0.5633x | 25.6372x |
| util.img_as_bool | float64 | reversed | 0.2801x | 1.1103x |
| util.img_as_bool | float64 | stepped | 0.6462x | 1.0585x |
| util.img_as_int | uint16 | full-fortran | 0.6963x | 0.9055x |
| util.img_as_int | uint16 | narrow-reversed | 0.9924x | 1.0501x |
