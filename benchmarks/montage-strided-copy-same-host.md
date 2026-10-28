# Montage: rejected constant-size copy experiment

**DISCARD under the frozen gates. The candidate remains unmerged; no retuning or repeated timing.** Overall gains pass: **1.144594135x direct / 1.121037218x normalized** over the existing C++ baseline. Balanced original/current is **1.548945661x**. These aggregate gains do not satisfy the required target and control gates.

The strided-reverse float32 target improves **2.650797501x direct / 2.686255332x normalized**, but fails the required >3x improvement in each candidate pass and remains below original in both passes (**0.492205915x / 0.451914991x**). The incomplete uint8 control is below its 0.95 floor (**0.948965866x direct / 0.948698532x normalized**). All six original wins selected for preservation remain wins in both candidate passes.

[All four raw reports, comparison, independent review and frozen contract](montage-strided-copy-same-host.json) preserve **504 samples**, original numeric text and terminal LF bytes. The [initial nine-case result](montage-minimum-initial.md) remains separate: its three historical original losses were strided float32, mean float64 and Fortran float64.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.003513316 | 0.003383899 | 1.728873723x | 1.787015458x | 1.038245121x | 1.033629833x | -0.000129418 |
| gray-incomplete-u8 | 0.008497100 | 0.008954063 | 1.418588251x | 1.345812591x | 0.948965866x | 0.948698532x | 0.000456963 |
| gray-dense-f32 | 0.036984638 | 0.035347712 | 1.518847373x | 1.526054274x | 1.046309253x | 1.004744980x | -0.001636926 |
| gray-large-u16 | 0.064948893 | 0.066185985 | 1.305246377x | 1.308643175x | 0.981308858x | 1.002602419x | 0.001237092 |
| gray-mean-f64 | 0.134909027 | 0.132908544 | 1.047340315x | 1.077181296x | 1.015051580x | 1.028492154x | -0.002000484 |
| rgb-padded-u8 | 0.019371806 | 0.018999304 | 6.617218579x | 6.845291460x | 1.019606054x | 1.034466578x | -0.000372501 |
| rgb-first-f32 | 0.040099396 | 0.040671194 | 2.461159509x | 2.429165487x | 0.985940970x | 0.987000427x | 0.000571798 |
| gray-strided-reverse-f32 | 0.338404165 | 0.127661266 | 0.175571693x | 0.471630397x | 2.650797501x | 2.686255332x | -0.210742899 |
| gray-fortran-f64 | 0.644781143 | 0.523254464 | 1.260554245x | 1.264974829x | 1.232251586x | 1.003506857x | -0.121526679 |

Added time is candidate minus baseline; balanced values use the geometric mean of the two respective passes. Direct losses are incomplete uint8, large uint16 and channels-first float32. Normalized losses are incomplete uint8 and channels-first float32. Strided float32 is the only original/current loss in each candidate pass. All losses are retained.

| Workload | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|
| gray-small-u8 | 1.803067166x | 1.771106650x |
| gray-incomplete-u8 | 1.266676213x | 1.429893063x |
| gray-dense-f32 | 1.555406651x | 1.497255811x |
| gray-large-u16 | 1.314389686x | 1.302921787x |
| gray-mean-f64 | 1.084505744x | 1.069906315x |
| rgb-padded-u8 | 7.001639207x | 6.692434985x |
| rgb-first-f32 | 2.483381922x | 2.376132689x |
| gray-strided-reverse-f32 | 0.492205915x | 0.451914991x |
| gray-fortran-f64 | 1.516057195x | 1.055475561x |

| Strided float32 target | Candidate pass 2 | Candidate pass 3 | Required in both |
|---|---:|---:|---:|
| Direct gain vs balanced baseline | 2.800245141x | 2.509325804x | >3x |
| Normalized gain vs balanced baseline | 2.803446876x | 2.573962706x | >3x |

The frozen keep rule required global direct and normalized GM >1.05; target >3 direct and normalized gain against the balanced baseline in both passes and >1 original/current in both; all six preserved original wins in both passes; each other case balanced direct and normalized >=0.95; identity/parity gates and completion before **20:27:56 UTC**. Measurement completed **18:53:37 UTC**, so the deadline passed. Target and control gates failed.

Source and execution scope: one variant specializing POD copy sizes 1/2/4/8/16, preserving generic fallback, fill, row-copy fast path, ABI, frontend and worker behavior. No stage timing establishes how much runtime any individual mechanism caused. Grid/shape/channel/dtype conversion, optional rescale and mean remain explicit providers. This is a minimum-profile, one-host whole-call measurement, not whole-library superiority or qualification.

Both prepared products contain **243 image APIs + 3 metadata APIs, 170 wheel members and SDK80**. All 13 dependency wheels, runtime versions and shared libraries match; only `montage.cpp.o` differs in the SDK archive. The producer ran 7,778 focused checks, 3,696 color checks and 1,439 smoke checks, with all 26 timings skipped. **No genuine installed SDK client was executed by these producers**; declared 26 families/52 checks are a separate boundary.

- Baseline: `ec3dba8e5eb73a0b02dffd70c501abe2adba923c`, producer `40740008474`, artifact `12840295821`.
- Rejected candidate: `894edfd7519404f3c4b341aa0124c96976b83486`, producer `40746039506`, artifact `12842995489`.
- [Measurement run 40749112415](https://github.com/bipxx/scikit-imagexx/actions/runs/40749112415), controller `5fb8ba788fb2c2dbbbea879adf35a13115b3bdd8`.
- [Complete retained result archive](https://github.com/bipxx/scikit-imagexx/actions/runs/40749112415/artifacts/12842532973): SHA256 `103ca8300360bddc66948625084ee3cb89a7aef7b77dcc7853f4c31c970beda2`.
- Independent review SHA256 `1ddc50dfa2c337c26912cd90382071f5505ba27aa63fadc4e8836e58aed225cc`; frozen candidate contract SHA256 `a677ff44b69c8c25285b5c702c698fc99f2998c688848b96e84f12724d75debc`.
