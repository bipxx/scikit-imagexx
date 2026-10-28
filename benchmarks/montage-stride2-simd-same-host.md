# Montage: rejected stride-2 SIMD experiment

**DISCARD under the frozen gates. The candidate remains private and unmerged; no retuning or rerun.** The strided float32 target improves **11.696343035x direct / 11.708077127x normalized** over the existing C++ baseline and beats scikit-image in both passes (**2.049476800x / 2.041840863x**). Overall direct and normalized gains are **1.310861581x / 1.315489212x**.

The channels-first float32 control fails both required 0.97 floors: **0.857388248x direct / 0.858565558x normalized**, adding **0.010168656 ms**. Every other fixed gate passes, including preservation of all six selected original wins. These results do not authorize merging the candidate.

[All four complete raw reports, comparison, independent review and prospective contract](montage-stride2-simd-same-host.json) preserve **504 samples**, original number text and terminal line endings. The [closed constant-size copy experiment](montage-strided-copy-same-host.md) remains a separate rejected variant; the [initial nine-case result](montage-minimum-initial.md) is retained.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.005067158 | 0.005018914 | 1.575909539x | 1.625887631x | 1.009612450x | 1.031713807x | -0.000048244 |
| gray-incomplete-u8 | 0.014961937 | 0.014702309 | 1.139297415x | 1.148139125x | 1.017658969x | 1.007760669x | -0.000259628 |
| gray-dense-f32 | 0.049363694 | 0.049079768 | 1.513992036x | 1.553139053x | 1.005784985x | 1.025856819x | -0.000283926 |
| gray-large-u16 | 0.081235525 | 0.081122242 | 1.433915688x | 1.433893586x | 1.001396447x | 0.999984586x | -0.000113283 |
| gray-mean-f64 | 0.192551616 | 0.177398729 | 0.921915827x | 1.006725098x | 1.085417112x | 1.091992423x | -0.015152887 |
| rgb-padded-u8 | 0.030102848 | 0.029290911 | 5.252358352x | 5.399628066x | 1.027719764x | 1.028038779x | -0.000811937 |
| rgb-first-f32 | 0.061134417 | 0.071303074 | 2.042540836x | 1.753655212x | 0.857388248x | 0.858565558x | 0.010168656 |
| gray-strided-reverse-f32 | 0.460681016 | 0.039386757 | 0.174721711x | 2.045655269x | 11.696343035x | 11.708077127x | -0.421294259 |
| gray-fortran-f64 | 0.696172583 | 0.705135449 | 1.042906096x | 1.022223742x | 0.987289156x | 0.980168537x | 0.008962867 |

Added time is candidate minus baseline. Balanced direct C++ losses are channels-first float32 and Fortran float64; normalized losses also include large uint16. Every candidate case beats original on a balanced basis, but mean float64 loses to original in candidate pass 3 (0.991925790x). Mean float64 was already below original in **both baseline passes** (0.863948168x / 0.983772897x); this is not a newly lost same-host baseline win.

| Workload | Pass 2 direct | Pass 3 direct | Pass 2 normalized | Pass 3 normalized | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.992173316x | 1.027358106x | 1.010371672x | 1.053506754x | 1.592254356x | 1.660231343x |
| gray-incomplete-u8 | 1.015269379x | 1.020054183x | 1.017723379x | 0.997895486x | 1.159489614x | 1.136899748x |
| gray-dense-f32 | 1.013427655x | 0.998199951x | 1.048460942x | 1.003740025x | 1.587361516x | 1.519654405x |
| gray-large-u16 | 0.997927731x | 1.004877219x | 0.999814534x | 1.000154668x | 1.433649744x | 1.434137468x |
| gray-mean-f64 | 1.092803023x | 1.078081119x | 1.108284702x | 1.075939648x | 1.021745208x | 0.991925790x |
| rgb-padded-u8 | 1.026066586x | 1.029375606x | 1.030356604x | 1.025726168x | 5.411802115x | 5.387481404x |
| rgb-first-f32 | 0.851459251x | 0.863358529x | 0.856390830x | 0.860745808x | 1.749213242x | 1.758108462x |
| gray-strided-reverse-f32 | 11.715589327x | 11.677128361x | 11.729949232x | 11.686245805x | 2.049476800x | 2.041840863x |
| gray-fortran-f64 | 0.982745136x | 0.991854187x | 0.983534057x | 0.976814532x | 1.025733664x | 1.018725831x |

Per-pass direct ratios compare each candidate with the geometric mean of both C++ baseline passes; normalized ratios divide its original/current ratio by the corresponding baseline geometric mean. Pass 2 C++ regressions: small uint8, large uint16, channels-first float32 and Fortran float64. Pass 3 C++ regressions: dense float32, channels-first float32 and Fortran float64. All original and C++ losses remain visible above and in the raw JSON.

The fixed keep rule required global direct and normalized GM >1.05; target direct and normalized gains >1.5 against the balanced baseline in **each** candidate pass, plus original/current >1 in each; all six selected original wins in both passes; each other case balanced direct and normalized >=0.97; exact identity/parity and completion before **21:54:15.637024 UTC**. Only the channels-first control floors fail. The target repeat gains are 11.715589327x / 11.677128361x direct and 11.729949232x / 11.686245805x normalized.

The run was queued **2026-10-28 20:13:49 UTC**; measurement/comparison completed **20:15:26.900343 UTC**, before the unchanged deadline. The fresh timing host was **AMD EPYC 9V74 80-Core Processor**. No builds or tests ran on that timing host.

Source scope: one raw-word SIMD copy route for single-channel 4-byte pixels at positive 8-byte column stride. Two loads compact eight words; at least nine remaining pixels are required, avoiding any read past the checked row span. Negative rows/images, fill, generic channel copying, public ABI, Python/binder and worker policy remain unchanged. This is a whole-call bundle measurement; no stage timing establishes the cause of either the gain or the control regression. Axis/grid/fill conversion, allocation, optional mean/rescale remain explicit providers.

Both genuine prepared products contain **245 image APIs + 3 metadata APIs, 172 wheel members and SDK81**. The 13 dependency wheels, runtime versions/libraries, installed Python and SDK headers match exactly; only existing `montage.cpp.o` changes, with new `simd_montage_avx2.cpp.o` added (159 to 160 archive objects). All other archive objects match. Each producer passed **8,238 focused checks**, **3,696 color checks**, **1,443 smoke checks**, rolling geometry 230 checks per mode, and separate 4/4/4/9 guards; all **27 timings were skipped**. The unchanged harness performs exact oracle/output/layout and actual builtin C-return checks before timing. Ten runtime identity verifications cover offline setup plus before/after all four passes. These minimum producers did not execute installed SDK clients; separate SDK-client evidence and whole-product qualification are not inferred from this experiment.

- [Timing run 40761382968](https://github.com/bipxx/scikit-imagexx/actions/runs/40761382968), controller `827d5026f60441046036be91be2e211d079b1771`.
- [Baseline producer 40758210302](https://github.com/bipxx/scikit-imagexx/actions/runs/40758210302): source `390dff892ad13c1f9b8cee0e76dce9a016f14c01`, artifact `12846458475`, SHA256 `d064c77793dbadd81c8de20e108f49467d19ffa59910fbccdd8609061cd42b89`.
- [Candidate producer 40758988448](https://github.com/bipxx/scikit-imagexx/actions/runs/40758988448): source `e66770d55ea7c342ef763b9c842bb4fa1939f6ad`, artifact `12848346695`, SHA256 `0ecf50ec8eb8bfdcc60b914e9ce65b0b8888d6e6c251be71566e5a299b0a6126`.
- Result artifact `12848172002`: **59,512 bytes, 44 members**, every member size/CRC/SHA verified; ZIP SHA256 `7272082646e642b1e182fd73477c5962a6e1238dda7260e4270a03cd3ce7cfb6`.
- Independent review SHA256 `0c2e62c1ce7a6306c8089ca0a181edac8676d6f79ebd827a930901d132f79a2d`; prospective contract raw SHA256 `9e5a4b0727ba0629742431234f1ff385c9d3c1a648238bf398db0035439854d9` (LF source hash `4ca2b78afa6fb886c49b1d15bd25baafc855c73ccbf1d6ddcd8ef13d318b8efe`).
- Harness SHA256 `93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd`; nine-workload contract SHA256 `434d7540206cd2adcf3253bc097e36db7131a15b0c6054fa06b2c97fd5da3d68`.
