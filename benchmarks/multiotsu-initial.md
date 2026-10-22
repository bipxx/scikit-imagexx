# Multi-Otsu performance experiments

The initial native feature is measured here; it is not yet merged to main. Values below apply only to the exact pinned source and the fixed timed cases, not to full API parity.

[Initial run 39838846191](https://github.com/bipxx/scikit-imagexx/actions/runs/39838846191) measures source `4352556`, with 33 cases per OS at global worker limits one/four (66 rows per OS). Both versions are warmed, reference pools are fixed to one, allocation and Python dispatch are included, and all seven raw samples are retained. Timed outputs match scikit-image 0.26.0 with native fallback disabled.

| OS | one-worker upstream/native geometric mean | one-worker cases slower than upstream |
|---|---:|---:|
| Linux | 2.306988x | 6 / 33 |
| Windows | 3.370553x | 6 / 33 |

Raw reports: [Linux](multiotsu-initial-linux.json), [Windows](multiotsu-initial-windows.json). Exact wheel/package source and binary hashes are retained in the run artifacts and checked against the downloaded measured binaries.

All adverse one-worker cases are supplied histograms with two classes:

| OS | case | upstream ms | native ms | upstream/native |
|---|---|---:|---:|---:|
| Linux | histogram-random-256 | 0.050376 | 0.073978 | 0.680968x |
| Linux | histogram-flat-256 | 0.050708 | 0.069480 | 0.729825x |
| Linux | histogram-gaps-256 | 0.057931 | 0.077504 | 0.747464x |
| Linux | histogram-random-1024 | 0.603940 | 0.673619 | 0.896561x |
| Linux | histogram-flat-1024 | 0.607116 | 0.651458 | 0.931935x |
| Linux | histogram-gaps-1024 | 0.622824 | 0.687506 | 0.905918x |
| Windows | histogram-random-256 | 0.054894 | 0.070110 | 0.782965x |
| Windows | histogram-flat-256 | 0.054743 | 0.069954 | 0.782553x |
| Windows | histogram-gaps-256 | 0.064080 | 0.078751 | 0.813710x |
| Windows | histogram-random-1024 | 0.945292 | 1.145811 | 0.824998x |
| Windows | histogram-flat-1024 | 0.950617 | 1.113844 | 0.853455x |
| Windows | histogram-gaps-1024 | 0.965950 | 1.137600 | 0.849112x |

This implementation constructs a quadratic lookup table even for two classes. A subsequent candidate replaces that work with linear prefix/suffix scoring while retaining float32 rounding and first-maximum ties. Future decisions use same-run initial/current native controls across the complete fixed corpus, including these adverse cases.

The initial source passes the timed case corpus, but later full CI exposes NumPy 1.24 float32 image-derived dtype metadata differences. Source `74be4e4` corrects metadata handling and expands supplied-center indexing tests; that correction must not be attributed to the earlier measured source. Further guards reject zero-rounded objectives from native execution because upstream leaves their index buffer unwritten. No full API parity or universal performance claim follows from this benchmark.


## Linear two-class candidate (`bb06801`)

[Paired run 39840449828](https://github.com/bipxx/scikit-imagexx/actions/runs/39840449828) uses the exact initial packages from run 39838846191, with all 33 case definitions unchanged. Both baseline and current results pass exact timed-case parity at worker limits one/four. Downloaded current binary hashes match the reports. Source and raw samples remain distinct from the initial trial.

| OS | one-worker upstream/current geometric mean | paired initial/current geometric mean | minimum upstream/current across 33 cases |
|---|---:|---:|---:|
| Linux | 3.770356x | 1.596471x | 1.929294x |
| Windows | 5.443809x | 1.681086x | 2.069799x |

Raw reports: [Linux](multiotsu-two-linux.json), [Windows](multiotsu-two-windows.json). All fixed one-worker cases beat scikit-image in this run; this is not a claim about unmeasured inputs. The six formerly adverse two-class histogram cases now reach 2.01–19.82x upstream on Linux and 2.07–29.36x on Windows. The 1024-bin cases improve 17.4–20.9x and 27.8–34.6x against the same-run initial native control.

The candidate also fixes dtype metadata and supplied-center indexing. It explicitly delegates all-zero rounded objectives, where the upstream Cython index buffer is unwritten. The initial timing control is valid for the fixed timed cases but does not satisfy these expanded delivery guards.

The minimum OS paired corpus score is 1.596471x. Its performance gate passes; delivery still requires the full final-source CI. The local minimum-NumPy suite passes 964 cases plus 24 undefined-reference skips in each SIMD mode, and five native configurations pass 2/2 including portable and ASan/UBSan normal/dynamic. Exact-source Windows benchmark parity also passes.

Adverse one-worker native controls are retained below; the largest measured regression is about 8% in a Linux float32 three-class image case. Most are three-class cases whose search remains quadratic/exhaustive.

| OS | dtype | case | classes | initial/current |
|---|---|---|---:|---:|
| Linux | uint8 | contiguous | 3 | 0.983538x |
| Linux | uint8 | reversed | 3 | 0.985183x |
| Linux | uint8 | tiny | 3 | 0.995422x |
| Linux | float32 | contiguous | 2 | 0.976263x |
| Linux | float32 | reversed | 2 | 0.997349x |
| Linux | float32 | contiguous | 3 | 0.921582x |
| Linux | float32 | reversed | 3 | 0.948764x |
| Linux | float32 | tiny | 3 | 0.988866x |
| Linux | float64 | tiny | 3 | 0.992003x |
| Linux | float32 | histogram-random-256 | 3 | 0.978132x |
| Linux | float32 | histogram-flat-256 | 3 | 0.993516x |
| Linux | float32 | histogram-random-1024 | 3 | 0.969585x |
| Linux | float32 | histogram-flat-1024 | 3 | 0.967746x |
| Linux | float32 | histogram-gaps-1024 | 3 | 0.958183x |
| Windows | uint8 | contiguous | 3 | 0.983682x |
| Windows | uint8 | reversed | 3 | 0.991621x |
| Windows | uint8 | tiny | 3 | 0.969857x |
| Windows | float32 | contiguous | 3 | 0.970232x |
| Windows | float32 | reversed | 3 | 0.966146x |
| Windows | float32 | tiny | 3 | 0.975907x |
| Windows | float64 | reversed | 3 | 0.989002x |
| Windows | float32 | histogram-random-1024 | 3 | 0.986933x |
