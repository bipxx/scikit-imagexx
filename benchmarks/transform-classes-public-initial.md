# transform-classes one-worker initial

Source `5d1b3da3418657a8b6e04ac9bb76d324310e9717` measured against scikit-image 0.26.0 in the [isolated hosted run](https://github.com/bipxx/scikit-imagexx/actions/runs/40000631852). All 296 parity checks passed per runtime; 276 calls have seven paired samples. Raw reports and every slower case are retained.

| Runtime | Work kind | Cases | Reference/current geometric mean | Slower cases |
|---|---|---:|---:|---:|
| windows-modern | public-coordinate-call | 204 | 2.304x | 0 |
| windows-modern | runtime-fitting-or-matrix-provider | 56 | 1.000x | 24 |
| windows-modern | state-metadata-or-serialization | 16 | 1.008x | 4 |
| linux-modern | public-coordinate-call | 204 | 3.030x | 0 |
| linux-modern | runtime-fitting-or-matrix-provider | 56 | 1.009x | 17 |
| linux-modern | state-metadata-or-serialization | 16 | 1.011x | 6 |
| linux-minimum | public-coordinate-call | 204 | 3.370x | 0 |
| linux-minimum | runtime-fitting-or-matrix-provider | 56 | 0.996x | 43 |
| linux-minimum | state-metadata-or-serialization | 16 | 1.002x | 7 |

Coordinate calls, NumPy/SciPy fitting or matrix providers, and state/serialization are separate groups. Provider and metadata ratios are not image-kernel speedups. Fitting and solvers remain explicit NumPy/SciPy providers.

[Summary, raw reports and all regressions](transform-classes-public-initial.json). No local operation timing was used.
