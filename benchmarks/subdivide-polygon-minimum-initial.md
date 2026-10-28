# Subdivide polygon: initial one-thread comparison

**14/14 workloads beat scikit-image 0.26.0**, with a **4.000437801× geometric mean speedup** and a **2.868446992×–6.803114516× range**. There are no losses in this frozen cohort. All **196 raw samples**, seven rounds per original/current pair, use one global worker and one reference thread pool.

[Complete report, summary, returned-timing journal, independent/root reviews and historical failures](subdivide-polygon-minimum-initial.json) preserve original number text and line endings. The fourteen workloads cover degrees 1–7, open/closed curves, endpoint preservation and C/Fortran/reversed/strided coordinates.

| Case | Layout | Original ms | C++ ms | Original/C++ | Added ms |
|---|---|---:|---:|---:|---:|
| open-degree1-256 | F | 0.035904847 | 0.012517173 | 2.868446992× | -0.023387674 |
| closed-degree1-256 | reverse-rows | 0.041513208 | 0.013771526 | 3.014423361× | -0.027741683 |
| open-degree2-4096 | reverse-rows | 0.244076644 | 0.064726363 | 3.770900040× | -0.179350282 |
| closed-degree2-4096 | step-rows | 0.250534913 | 0.068021426 | 3.683176435× | -0.182513487 |
| open-degree3-65536 | step-rows | 3.655709500 | 0.979740857 | 3.731302490× | -2.675968643 |
| closed-degree3-65536 | step-columns | 3.962881000 | 0.582509818 | 6.803114516× | -3.380371182 |
| open-degree4-256 | step-columns | 0.044751955 | 0.012491758 | 3.582518421× | -0.032260196 |
| closed-degree4-256 | C | 0.043911600 | 0.011894310 | 3.691815559× | -0.032017290 |
| open-degree5-4096 | C | 0.281359825 | 0.047110407 | 5.972349679× | -0.234249418 |
| closed-degree5-4096 | F | 0.362297118 | 0.074745816 | 4.847055518× | -0.287551302 |
| open-degree6-65536 | F | 4.228888250 | 0.973787267 | 4.342722887× | -3.255100983 |
| closed-degree6-65536 | reverse-rows | 4.380074500 | 1.175501786 | 3.726131728× | -3.204572714 |
| open-degree7-65536 | reverse-rows | 4.527559667 | 1.195407077 | 3.787462659× | -3.332152590 |
| closed-degree7-65536 | step-rows | 4.822834667 | 1.283229250 | 3.758357804× | -3.539605417 |

Added time is C++ minus original. Open-curve GM is **3.919218640×**; closed-curve GM **4.083340092×**. Every individual sample, median and loss classification is retained; there is no prior-C++ comparison or whole-library speed claim.

Timing covers the whole registered public call, including validation, normalization, allocation and output destruction. Native ownership covers both ordered B-spline convolutions, interleave, circular closure and ordinary endpoint copies. NumPy closure detection, conversion and allocation remain providers. Before timing, exact independent Pascal-mask/scalar-oracle and original/native words, input/backing state, dtype, layout and owned output were checked; the actual builtin C-return and accepted array path were required. FP policy/handler/buffer and worker count stayed unchanged. All 14 returned timing dictionaries match the final report.

**The product was not rebuilt for this run.** It is the immutable genuine producer source `4f69eba77a262056459eaab1e0a39e53610e9305`, core `144054dc…`. The original prepared harness hash remains `101265e4…`; the executed external observer-v2 hash is separately recorded as `0bce058e…`. The correction changes the observer from singleton `True` to the frontend’s array-or-`None` contract and adds source-root/provenance metadata. An exact six-edit inverse recovers the entire prepared v1 file. Frozen dataset `2cdfe419…`, pure helpers, fourteen cases, numerical operations and timed calls are unchanged.

The earlier v1 run **40781858205** failed in that untimed observer before its first timing call: **zero timing samples**, and no output-parity pass claim. Its failure review and traceback remain included. In the two authorized local first-cohort preflights, normal reached the timing sentinel after all untimed checks but an extra receipt assertion raised `KeyError`; it was not replayed. Scalar completed its receipt. Both consumed one original/native precheck each and zero timings, using local CPython 3.11.14; neither is host timing evidence. These failures remain separate from this successful result.

Host: **AMD EPYC 7763 64-Core Processor**, **CPython 3.11.17**, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0. Three runtime identity logs match the authenticated product and all 13 exact dependencies before/after. The admitted producer has **176 wheel members / SDK83 / 247 image APIs + 3 metadata APIs**, passed **9,346 focused checks** and **1,447 smoke checks**, with all **29 timings skipped**. Actual genuine SDK83 client v2 evidence covers **one family/two checks separately**; no SDK clients, builds or tests ran on the timing host. The qualified baseline remains 241; this is not full four-profile qualification.

- [Successful timing run 40783931593](https://github.com/bipxx/scikit-imagexx/actions/runs/40783931593), controller `14df18060ee42410d7d3a377f49a15ee5899743f`.
- [Genuine producer 40778508940](https://github.com/bipxx/scikit-imagexx/actions/runs/40778508940): artifact `12855199919`, SHA256 `4918246748c2d7d263f26f8eb05bc65f299d2ba2caf9672a3fcd1683338feb96`.
- Result artifact `12859105093`: **234,717 bytes**, retained once; SHA256 `4d99123840b192065e0fe589a26cd70c2588800f8b99f04f492a0f045f4822b9`.
- Core SHA256 `144054dc3f7ed0251426f0a22292048c7b4510557d64f0aecfd0f50e53926cb6`.
- Executed harness `0bce058ebf921b7f80f3a1dac74632de3e2ab5e7a7ee08369339077cfbb53010`; prepared harness `101265e40bdd2c7026cdbb396f9670e2c07d5e32502be55de2411eea623faa38`.
- Frozen dataset `2cdfe4191810ee7ae2de72a11fc921ced6edec72cb5ac282bdc4ae32780dd8da`.
- Independent review `56c535800fb288f6f20b22d402cd92edf5cb8abe21941a30e7e74c7c9616e8bf`; root recomputation `265bb4970373a540c061eed701c39d64ff93f3adeb99f41ffc9b3e54bf84d67d`.

All raw JSON documents are embedded verbatim, including their terminal newlines. Journals, logs and driver source are base64 with decoded length/SHA; decoding reproduces every original byte. No benchmark or original/native call ran during publication verification.
