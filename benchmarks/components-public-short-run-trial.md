# Connected-component labeling: short-run trial

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40035393397) measured exact source `845608d8a035eebb31edcb491ee853bf11ad44fe` against scikit-image 0.26.0. All fixed parity cases passed before seven paired samples per successful workload, using one native worker and one reference thread pool. Ratios above 1 favor the C++ public call.

The cohorts have different work and remain separate. The table reports geometric mean public-call ratios, with equally weighted cases within each cohort.

| Cohort (cases/runtime) | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| cast_warning (6) | 1.0270x | 1.1509x | 1.1838x |
| empty (18) | 1.8981x | 2.2052x | 2.2617x |
| raster (200) | 0.9924x | 1.6678x | 1.8291x |
| tiny (578) | 1.2678x | 1.6176x | 1.5462x |

| Runtime | All slower public calls | Worst public-call speedup |
|---|---:|---:|
| windows-modern | 357/802 | 0.2657x |
| linux-modern | 265/802 | 0.2625x |
| linux-minimum | 269/802 | 0.2888x |

Each runtime passed all 954 cases. The 802 timings comprise 200 raster calls, 578 tiny calls, 18 empty calls and six successful cast-warning calls; 152 expected errors are untimed. Warning display is suppressed symmetrically after warning parity checks. Windows raster performance is approximately parity and slightly below original in this campaign; tiny/empty ratios do not establish large-image speedups.

Measured slower examples (complete samples and workload specifications are retained in the JSON):

- windows-modern: `measure.label`, `large/dtype=float32/multivalued/connectivity=2`, 0.2657x; workload index 429.
- linux-modern: `measure.label`, `large/dtype=float64/multivalued/connectivity=2`, 0.2625x; workload index 441.
- linux-minimum: `morphology.label`, `large/dtype=float64/multivalued/connectivity=2`, 0.2888x; workload index 918.

These are observed ratios on the recorded CPUs for the exact source above. Hosted CPU/run differences prevent causal before/after attribution to the named patch. Individual slower calls remain, and this campaign does not establish universal performance superiority.

Offline verification covered complete product wheels/RECORDs, preserved package bytes, all 59 SDK members, source headers/Python and numerical runtime pins/freeze text. These older workflows did not archive a full dependency-binary closure. The installed product/SDK seals were checked against the wheel.

All three preparation jobs finished before fresh measurement jobs began; measurement jobs performed no builds or tests. The [summary and complete slower-case list](components-public-short-run-trial.json) links three byte-exact raw reports, authoritative artifact IDs/hashes, all samples and source workload specifications. No image operations, timers, foreign-binary imports or source extraction were performed locally for this publication.
