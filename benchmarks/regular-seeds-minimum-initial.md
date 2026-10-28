# Regular seeds: initial one-thread comparison

All nine frozen cases were faster than scikit-image 0.26.0: **9 wins, 0 losses**, geometric mean **1.590992375371×**. These are seven-round medians from one initial host/cohort, with all 126 raw samples retained in [the JSON report](regular-seeds-minimum-initial.json).

CPU: AMD EPYC 9V74 80-Core Processor. Runtime: Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3; one worker, 19 ms target per batch. [Timing run 40728558127](https://github.com/bipxx/scikit-imagexx/actions/runs/40728558127) used the authenticated, offline, hash-locked producer artifact; original/before/after runtime, core and SDK identities matched.

| Case | Shape / requested points / dtype | Original ms | Current ms | Original/current | Added ms |
|---|---|---:|---:|---:|---:|
| tiny2d-i64 | 16×16 / 16 / int64 | 0.020205051 | 0.017441476 | 1.158448463× | -0.002763575 |
| dense2d-i32 | 512×512 / 262144 / int32 | 0.187756582 | 0.037776451 | 4.970201725× | -0.149980131 |
| sparse2d-i64 | 1000×1000 / 100 / int64 | 0.114262200 | 0.111004309 | 1.029349229× | -0.003257891 |
| dense3d-f32 | 64×64×64 / 262144 / float32 | 0.224581098 | 0.147801072 | 1.519482197× | -0.076780026 |
| sparse3d-f64 | 100×100×100 / 125 / float64 | 0.117935894 | 0.115125959 | 1.024407479× | -0.002809934 |
| dense2d-u8-wrap | 512×512 / 262144 / uint8 | 0.175050484 | 0.045470404 | 3.849767510× | -0.129580080 |
| dense2d-f64 | 768×768 / 589824 / float64 | 0.532445091 | 0.323123816 | 1.647805160× | -0.209321275 |
| moderate3d-i32 | 96×96×96 / 4096 / int32 | 0.070944025 | 0.067705000 | 1.047840268× | -0.003239025 |
| sparse2d-f32 | 768×1024 / 1024 / float32 | 0.063299345 | 0.059429710 | 1.065112806× | -0.003869635 |

Added ms is current minus original. Sparse 3D float64 and sparse 2D int64 gains were about 2.4% and 2.9%; these small margins come from this single initial cohort. The sparse float32 grid selected 999 labels for a request of 1,024, matching the original grid provider. Every case passed exact dtype, shape, storage and complete-word parity, actual C-return/admission guards, and an independent output-coordinate label oracle. This cohort establishes no prior-C++ improvement or library-wide superiority.

The public call retains explicit `regular_grid` preparation and `np.zeros` allocation/zero initialization; C++ owns label generation and selected-grid scatter. Timings include grid preparation, allocation, labeling and output destruction. Sparse output keeps the original zero allocation and receives no duplicate full clear.

Measured source: `910c31a609c157c28cbbe56933b8448b710af95e`, with 242 image APIs, 245 registered functions, 168 product files and 79 SDK files. The separately qualified baseline remains source `4e657ee64fb71e4d707d32e9539fdb4d7cc528af`, with 241 image APIs and 71 SDK files. This benchmark does not expand whole-product qualification.

A separate check linked the existing `recent_primitives` client solely against this untouched genuine SDK and passed normal/scalar execution: **one family, two checks** (receipt SHA `47207306d45886a579fca18682e9bdce11fe37dd46054e653eb069b3f7dc5c32`). The source-declared 26-family/52-check campaign remains unexecuted. The preparation run itself executed zero SDK clients.

Producer run/artifact: `40724150045` / `12833680393`, ZIP SHA `38797cf494532e16da41f751062c57815de2d566d8ebf11f3b59be26ba4be536`. Result artifact: `12833992888`, ZIP SHA `7dec3949c34b0a4fedb3fe47f60dcfcb9a1adba610b4e28f56acc213839e2d94`. Independent raw review SHA: `f422297a920740f29906be9910bf4b98ff6d30c92448e099004fa0732bb67143`. The result archive retains the complete provenance, CPU details and identity logs; the JSON publication embeds all raw benchmark, summary and review numeric text unchanged.
