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
