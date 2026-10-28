# Montage: rejected isolated-wrapper SIMD experiment

**DISCARD under the prospectively frozen gates. Candidate `61b4f79821b0c7fd0d224e2a12d4263d14308cb7` remains private and unmerged. No retuning or rerun.** The strided float32 target improves **11.271047508x direct / 11.302008217x normalized**, beating scikit-image by **2.115484521x / 2.139695330x** in the two candidate passes. RGB-first float32 passes its new per-pass floors. **All nine workloads beat scikit-image in both candidate passes.**

The other C++ control floors still fail: incomplete uint8 is **0.895325380x direct / 0.938950902x normalized**; padded RGB uint8 is **0.938569004x / 0.942767842x**; Fortran float64 is **0.909324941x normalized**, despite a **1.203408869x direct** gain. Global gains of **1.324073531x direct / 1.300198745x normalized** do not override those failures.

[Four complete raw reports, comparison, independent review, frozen contract, strict codegen proofs and placement receipt](montage-wrapper-simd-same-host.json) preserve **504 samples**, original number text and terminal line endings. The [constant-size copy variant](montage-strided-copy-same-host.md), [in-body stride-2 variant](montage-stride2-simd-same-host.md), and this isolated-wrapper variant are three distinct closed experiments. The [initial nine-case result](montage-minimum-initial.md) remains separate.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.003147160 | 0.003138298 | 1.718853024x | 1.736850282x | 1.002823997x | 1.010470505x | -0.000008863 |
| gray-incomplete-u8 | 0.007439744 | 0.008309542 | 1.406494868x | 1.320629626x | 0.895325380x | 0.938950902x | 0.000869798 |
| gray-dense-f32 | 0.032696396 | 0.032968886 | 1.482062390x | 1.515719509x | 0.991734940x | 1.022709651x | 0.000272490 |
| gray-large-u16 | 0.061634108 | 0.061393796 | 1.243687284x | 1.276848940x | 1.003914270x | 1.026663983x | -0.000240312 |
| gray-mean-f64 | 0.116703634 | 0.112824554 | 1.081403304x | 1.114741504x | 1.034381524x | 1.030828647x | -0.003879080 |
| rgb-padded-u8 | 0.017966050 | 0.019141960 | 6.639138641x | 6.259166409x | 0.938569004x | 0.942767842x | 0.001175910 |
| rgb-first-f32 | 0.036566833 | 0.034411272 | 2.457740521x | 2.623074020x | 1.062641145x | 1.067270526x | -0.002155561 |
| gray-strided-reverse-f32 | 0.277511585 | 0.024621632 | 0.188245792x | 2.127555487x | 11.271047508x | 11.302008217x | -0.252889953 |
| gray-fortran-f64 | 0.601344201 | 0.499700656 | 1.441104797x | 1.310432534x | 1.203408869x | 0.909324941x | -0.101643545 |

Added time is candidate minus baseline. Balanced direct C++ losses are incomplete uint8, dense float32 and padded RGB uint8; normalized losses are incomplete uint8, padded RGB uint8 and Fortran float64. The Fortran direct and normalized results differ and are both retained. There are no candidate original losses in either pass. The initial three historical original losses (strided float32, mean float64 and Fortran float64) remain preserved in the evidence; on this host, only strided float32 loses to original in either baseline pass.

| Workload | Pass 2 direct | Pass 3 direct | Pass 2 normalized | Pass 3 normalized | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 1.007681428x | 0.997989981x | 1.017156700x | 1.003828261x | 1.748342870x | 1.725433241x |
| gray-incomplete-u8 | 0.981204634x | 0.816962649x | 0.991183914x | 0.889470445x | 1.394095088x | 1.251035616x |
| gray-dense-f32 | 0.997349934x | 0.986151558x | 1.003668114x | 1.042112442x | 1.487498763x | 1.544475657x |
| gray-large-u16 | 1.005987641x | 1.001845173x | 1.010633510x | 1.042948729x | 1.256912044x | 1.297102071x |
| gray-mean-f64 | 1.024018703x | 1.044849215x | 1.011483609x | 1.050543666x | 1.093821717x | 1.136061391x |
| rgb-padded-u8 | 0.965428891x | 0.912456406x | 0.960474797x | 0.925387325x | 6.376725341x | 6.143774750x |
| rgb-first-f32 | 1.065535524x | 1.059754629x | 1.062782520x | 1.071777485x | 2.612043664x | 2.634150955x |
| gray-strided-reverse-f32 | 11.157179935x | 11.386077188x | 11.237884786x | 11.366497537x | 2.115484521x | 2.139695330x |
| gray-fortran-f64 | 1.179071115x | 1.228248990x | 0.806704558x | 1.024999599x | 1.162545808x | 1.477131838x |

Per-pass C++ ratios compare each candidate with the geometric mean of both baseline passes. Pass 2 direct regressions: incomplete uint8, dense float32 and padded RGB uint8. Pass 3 direct regressions: small uint8, incomplete uint8, dense float32 and padded RGB uint8. Pass 2 normalized regressions: incomplete uint8, padded RGB uint8 and Fortran float64. Pass 3 normalized regressions: incomplete uint8 and padded RGB uint8. All ratios remain visible in the table and raw data.

The fixed keep rule required global direct and normalized GM >1.05; target direct and normalized gain >1.5 in **each** candidate pass against the balanced baseline, plus original/current >1 in each; all six selected original wins in both passes; each other case balanced direct and normalized >=0.97; **RGB-first direct and normalized >=0.97 in each candidate pass**; exact identity/parity/codegen and completion before **22:35:00.116019 UTC**. The failed gates are only the other-case direct and normalized floors. RGB-first repeat gains are **1.065535524x / 1.059754629x direct** and **1.062782520x / 1.071777485x normalized**. All six preserved original wins pass. These frozen results and thresholds are immutable.

Queued **2026-10-28 21:17:56 UTC**; measurement/comparison completed **21:19:12.399630 UTC**, before the unchanged deadline. The fresh host was **AMD EPYC 9V45 96-Core Processor**. No builds or tests ran on the timing host. Different closed experiments used different hosts; this result does not establish a causal comparison with those runs.

The public C++ `montage_into` ABI and Python/binder remain unchanged. Only the old definition is renamed to `montage_generic_into`; a separate wrapper dispatches the same packed stride-2 copy and calls the original generic operation with a zero-image descriptor for its exact fill once. No new allocation or duplicated clear is introduced. Noneligible calls forward to the old generic body. Both SDK and Python calls reach the public wrapper.

The strict precondition passed on like-toolchain local objects and the **actual genuine SDK pair**: all three generic functions (entry, fill worker and copy worker) retain identical object-relative instruction offsets/bytes, registers, stack offsets, branches and text relocations after **only symbol-name normalization** and exclusion of the object filename header. The original source body is recovered by the exact inverse of one definition-token rename. No flags, padding, aliases or objcopy machinery were used.

Actual linked placement is recorded: baseline entry **0x464410**; candidate public wrapper **0x464260**; candidate generic entry **0x5eafd0**. Generic size remains **1,677 bytes**, stack **0x118**; generic entry and inner-loop modulo-64 alignment remain 16 and 48 respectively. The wrapper has a **0xf8 stack frame** and an extra **call/return**, not a tail jump, on the generic route. Placement differs despite identical generic code. These are structural facts; no stage timing assigns the gains or regressions to any one mechanism. The whole public call includes NumPy argument/grid/fill conversion, allocation, optional mean/rescale providers and destruction.

Both products contain **245 image APIs + 3 metadata APIs, 172 wheel members and SDK81**. All 13 dependency wheels, runtime versions/libraries, installed Python and SDK headers match. Only existing `montage.cpp.o` changes its symbol names; `montage_wrapper.cpp.o` and `simd_montage_avx2.cpp.o` are added (159 to 161 archive objects); all other existing objects match. Each producer passed **8,238 focused checks**, **3,696 color checks**, **1,443 smoke checks**, rolling geometry 230 nodes per mode and separate 4/4/4/9 guards; all **27 timings were skipped**. Exact oracle/output/layout and actual builtin C-return checks precede timing, and ten runtime identity verifications cover setup plus before/after every pass. These prepare-only producers did not execute installed SDK clients. This experiment grants no broader qualification or whole-library speed claim.

- [Timing run 40770769989](https://github.com/bipxx/scikit-imagexx/actions/runs/40770769989), controller `53e20232f40721cc3f38466fc724ca307a7706a4`.
- [Baseline producer 40758210302](https://github.com/bipxx/scikit-imagexx/actions/runs/40758210302): source `390dff892ad13c1f9b8cee0e76dce9a016f14c01`, artifact `12846458475`, SHA256 `d064c77793dbadd81c8de20e108f49467d19ffa59910fbccdd8609061cd42b89`.
- [Candidate producer 40766081804](https://github.com/bipxx/scikit-imagexx/actions/runs/40766081804): source `61b4f79821b0c7fd0d224e2a12d4263d14308cb7`, artifact `12849749460`, SHA256 `c4420f7d4e95d4cb85ffcbd66e24ac7ee8c0f5684b8619b061663a2b98697989`.
- Result artifact `12852297329`: **87,955 bytes, 52 members**, every member size/CRC/SHA verified; ZIP SHA256 `f92582c37319e30ec73fa515bb80d68883bb9241f86a81024b4d707d5b777552`.
- Independent review SHA256 `7d787ea1cf69e66d5244d8c44a04b395204bd22151c40afe21793f24409692cd`; candidate contract raw/LF SHA256 `f39a341c447b872e1d2ec9b77d50fc5fb6a79d8d374ce5cd6574f4bfc975d3df`.
- Harness SHA256 `93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd`; nine-workload contract SHA256 `434d7540206cd2adcf3253bc097e36db7131a15b0c6054fa06b2c97fd5da3d68`.
