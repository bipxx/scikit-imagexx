# Approximate polygon: initial one-thread comparison

All nine frozen float64 workloads were faster than scikit-image 0.26.0: **9 wins, 0 losses**, geometric mean **6.728133285482×**. All 126 raw samples, report numeric text, independent review, root recomputation and diagnostic evidence are preserved in [the JSON report](approximate-polygon-minimum-initial.json).

CPU: AMD EPYC 7763 64-Core Processor. Runtime: Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3. Seven rounds, one worker, 19 ms target per batch. [Run 40780548412](https://github.com/bipxx/scikit-imagexx/actions/runs/40780548412) measured whole public calls, including preparation, allocation and output destruction, from the authenticated offline artifact.

| Workload | Original ms | Current ms | Original/current | Added ms |
|---|---:|---:|---:|---:|
| collinear-256-C | 0.052103667 | 0.011456137 | 4.548100969× | -0.040647530 |
| collinear-4096-F | 0.119727753 | 0.057981168 | 2.064942081× | -0.061746585 |
| collinear-65536-step | 1.209809462 | 0.918384800 | 1.317323045× | -0.291424662 |
| zigzag-64-C | 5.137272667 | 0.374532447 | 13.716495601× | -4.762740219 |
| zigzag-512-reverse | 42.688063000 | 3.194240400 | 13.364073349× | -39.493822600 |
| zigzag-2048-step | 184.449531000 | 15.724577000 | 11.730015440× | -168.724954000 |
| smooth-256-C | 6.811446500 | 0.493313517 | 13.807540767× | -6.318132983 |
| smooth-4096-F | 7.745032500 | 0.703138348 | 11.014948230× | -7.041894152 |
| smooth-32768-C | 14.049125000 | 2.011905875 | 6.982993178× | -12.037219125 |

Added ms is current minus original. Gains ranged from 1.317× for the 65,536-point strided collinear chain to 13.808× for the 256-point smooth chain. This is one initial host and cohort; it establishes neither a prior-C++ improvement nor library-wide speed superiority.

C++ owns the full iterative Ramer–Douglas–Peucker distance, projection, split, deterministic first-maximum selection, stack and keep-mask path. The Python public path retains NumPy scalar trigonometry and final original-coordinate gathering as explicit providers, plus input normalization and exceptional paths. The assessed benchmark inputs are ordinary finite float64 point chains with positive tolerance. Exact input/output words, independent oracle hashes, output layout, FP state, worker state and actual native C-entry ownership passed.

The first independent Windows review stopped at exact input-hash reconstruction for the three sine-based cohorts; all six collinear/zigzag hashes matched. The identical frozen reviewer passed under retained Linux stdlib/libm, with no validator, result, oracle, case or harness changes and no repeated timing or scientific call. Both the Windows failure and Linux success are embedded. This provider portability boundary is not a performance regression or a claim that the generated sine words are portable across math libraries.

Measured source: `4f69eba77a262056459eaab1e0a39e53610e9305`, with 247 image APIs, 250 registered functions, 176 product files and 83 SDK files. The separately qualified baseline remains source `4e657ee64fb71e4d707d32e9539fdb4d7cc528af`, with 241 image APIs and 71 SDK files. This benchmark does not expand full product qualification.

A separate installed recent-primitives client linked only the untouched genuine SDK83 and passed normal/scalar execution with worker budgets 1 and 4: **one family, two checks**. The source-declared 26-family/52-check campaign remains unexecuted. The preparation run itself executed zero SDK clients. The complete SDK client receipt is embedded separately in the JSON.

Producer run/artifact: `40778508940` / `12855199919`, ZIP SHA `4918246748c2d7d263f26f8eb05bc65f299d2ba2caf9672a3fcd1683338feb96`. Result artifact: `12857011469`, ZIP SHA `fde9322b7d51685faa2f0e3232b14cff142015b8773b58f81b274d2e2d392892`. Independent review SHA: `455b1601ae555d982bb03715833cf41b2bb2c9d81462fa996caeddd65b54e96a`. Original/before/after source, runtime, core and SDK identities matched exactly.
