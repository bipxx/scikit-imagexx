# Footprint factories/utilities: setup trial

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40035389599) measured exact source `845608d8a035eebb31edcb491ee853bf11ad44fe` against scikit-image 0.26.0. All fixed parity cases passed before seven paired samples per successful workload, using one native worker and one reference thread pool. Ratios above 1 favor the C++ public call.

The cohorts have different work and remain separate. The table reports geometric mean public-call ratios, with equally weighted cases within each cohort.

| Cohort (cases/runtime) | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| decomposition_plan (144) | 1.0610x | 1.0172x | 1.1904x |
| raster (254) | 2.5348x | 2.2325x | 2.4804x |
| view_metadata (129) | 1.0212x | 1.0176x | 1.0281x |

| Runtime | All slower public calls | Worst public-call speedup |
|---|---:|---:|
| windows-modern | 134/527 | 0.4171x |
| linux-modern | 116/527 | 0.4328x |
| linux-minimum | 77/527 | 0.4157x |

Each runtime passed all 556 cases. The 527 timings comprise 254 raster-generation calls, 144 decomposition-plan calls and 129 view/metadata calls; 29 expected errors are untimed. Plan/metadata ratios are not raster or downstream morphology-kernel speed claims.

Measured slower examples (complete samples and workload specifications are retained in the JSON):

- windows-modern: `morphology.footprint_rectangle`, `dtype=S3/decomposition=sequence`, 0.4171x; workload index 331.
- linux-modern: `morphology.footprint_rectangle`, `dtype=structured/decomposition=separable`, 0.4328x; workload index 376.
- linux-minimum: `morphology.footprint_rectangle`, `dtype=U3/decomposition=separable`, 0.4157x; workload index 310.

These are observed ratios on the recorded CPUs for the exact source above. Hosted CPU/run differences prevent causal before/after attribution to the named patch. Individual slower calls remain, and this campaign does not establish universal performance superiority.

Offline verification covered complete product wheels/RECORDs, preserved package bytes, all 59 SDK members, source headers/Python and numerical runtime pins/freeze text. These older workflows did not archive a full dependency-binary closure. The installed product/SDK seals were checked against the wheel.

All three preparation jobs finished before fresh measurement jobs began; measurement jobs performed no builds or tests. The [summary and complete slower-case list](footprints-public-setup-trial.json) links three byte-exact raw reports, authoritative artifact IDs/hashes, all samples and source workload specifications. No image operations, timers, foreign-binary imports or source extraction were performed locally for this publication.
