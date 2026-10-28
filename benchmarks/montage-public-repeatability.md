# Montage: full public repeatability diagnostic

Eight independent processes measured the same nine full public workloads in **A B B A B A A B** order on one fresh host. The [JSON publication](montage-public-repeatability.json) retains all **1,008 samples**, eight journals containing **72 returned timing records**, eight complete raw reports, initial/final process and identity receipts, CPU/source evidence, independent review and root recomputation.

This describes repeatability and drift of unchanged products. It does not establish the cause of observations on an earlier host, create a new acceptance test, or reopen any closed experiment. **All prior DISCARD decisions remain unchanged.**

CPU: AMD EPYC 7763 64-Core Processor; every process pinned to CPU 0 before scientific imports. Workers and reference pools: one; seven rounds; unchanged 19 ms target. [Run 40781580047](https://github.com/bipxx/scikit-imagexx/actions/runs/40781580047) used product A `390dff892ad13c1f9b8cee0e76dce9a016f14c01` and product B `61b4f79821b0c7fd0d224e2a12d4263d14308cb7`, with their original source, core, runtime and SDK identities.

Direct = GM(A C++ medians) / GM(B C++ medians). Normalized = GM(B original/C++ ratios) / GM(A original/C++ ratios). Original/C++ columns are four-pass geometric means for each product; values below one retain original-library losses.

| Case | Direct A/B | Normalized B/A | Original/C++ A | Original/C++ B |
|---|---:|---:|---:|---:|
| gray-small-u8 | 1.013463343× | 1.015266210× | 1.709396825× | 1.735492836× |
| gray-incomplete-u8 | 1.017790179× | 1.000053343× | 1.268202256× | 1.268269905× |
| gray-dense-f32 | 0.962857396× | 0.987229471× | 1.400730933× | 1.382842858× |
| gray-large-u16 | 0.978556538× | 0.979294750× | 1.287129319× | 1.260478985× |
| gray-mean-f64 | 0.988735830× | 0.978648036× | 0.987942262× | 0.966847754× |
| rgb-padded-u8 | 1.017128220× | 1.016931700× | 4.441583034× | 4.516786587× |
| rgb-first-f32 | 0.937266003× | 0.982492728× | 1.946146917× | 1.912075194× |
| gray-strided-reverse-f32 | 8.197473157× | 8.597426760× | 0.189118020× | 1.625928327× |
| gray-fortran-f64 | 1.017711735× | 1.016906623× | 0.920796531× | 0.936364091× |

Across these nine cases, descriptive direct GM is 1.253538091795× and normalized GM is 1.266624791050×, strongly influenced by the strided target. Balanced residual regressions remain in gray-dense-f32, gray-large-u16, gray-mean-f64 and rgb-first-f32. Mean and Fortran float64 still lose to the original library in the four-pass aggregates; product A also retains its strided-reverse loss. The earlier incomplete-u8, padded-RGB-u8 and Fortran-f64 control failures are not balanced A/B losses on this fresh host. None of these observations changes the earlier decisions.

Same-product null pairs compare **B2/B3** and **A6/A7**, with no product change. C++ spread is `(max/min - 1) × 100%`; original ratios are earlier/later, and normalized ratios are later/earlier original-over-C++.

| Case | Pair | C++ spread | Original earlier/later | Normalized later/earlier |
|---|---|---:|---:|---:|
| gray-small-u8 | B2/B3 | 0.83% | 1.013634394× | 0.994714579× |
| gray-small-u8 | A6/A7 | 2.49% | 1.022977027× | 0.953827577× |
| gray-incomplete-u8 | B2/B3 | 0.84% | 1.010345593× | 0.981549846× |
| gray-incomplete-u8 | A6/A7 | 1.89% | 0.957441996× | 1.025053823× |
| gray-dense-f32 | B2/B3 | 2.02% | 1.031924999× | 0.949902480× |
| gray-dense-f32 | A6/A7 | 2.74% | 1.027813986× | 0.999636903× |
| gray-large-u16 | B2/B3 | 2.81% | 0.985586308× | 0.986912439× |
| gray-large-u16 | A6/A7 | 0.92% | 0.992237632× | 1.017047316× |
| gray-mean-f64 | B2/B3 | 16.50% | 0.891798585× | 0.962501651× |
| gray-mean-f64 | A6/A7 | 0.96% | 0.971680090× | 1.019316651× |
| rgb-padded-u8 | B2/B3 | 8.72% | 0.982651411× | 0.936002268× |
| rgb-padded-u8 | A6/A7 | 4.11% | 1.008089914× | 0.952816251× |
| rgb-first-f32 | B2/B3 | 6.22% | 0.935303873× | 1.006599778× |
| rgb-first-f32 | A6/A7 | 3.34% | 0.967403456× | 1.000281838× |
| gray-strided-reverse-f32 | B2/B3 | 3.02% | 0.921956329× | 1.052822953× |
| gray-strided-reverse-f32 | A6/A7 | 0.34% | 0.993907407× | 1.009591844× |
| gray-fortran-f64 | B2/B3 | 3.41% | 0.953759724× | 1.013893759× |
| gray-fortran-f64 | A6/A7 | 0.64% | 1.005907046× | 1.000453840× |

With unchanged B, mean-f64 C++ medians differ by **16.50%** between B2/B3, and padded RGB differs by **8.72%**. The unchanged A padded-RGB A6/A7 pair differs by **4.11%**. Original-library timing also drifts, so direct and normalized null ratios differ. These are measured variations, not proof of a CPU, allocator, frequency, cache or scheduling cause. Full per-round min/max/median/MAD and four-pass product spreads are retained.

The unchanged harness checks full original/current/oracle output words, layouts, input/backing preservation, actual native admission and caller FP/worker state internally. Exported evidence contains checked hashes and descriptors, not complete output buffers or numeric FP-register snapshots. The public NumPy prelude, allocation, mean/rescale/axis/type providers remain included in the timed calls. This diagnostic establishes no new parity coverage, product qualification or accepted optimization.

Artifact `12856434416`, ZIP SHA `5a2667220ccfdde3a9cf6ab915841de3d6dd2ed6acb0f48aeae7fe8903fcc88b`. Independent review SHA `a5f58af615be6d82d4018b4a05ba57a067507af755bfd7d7a81f8a1bfc7eab35`; root recomputation SHA `4de0c0471fbeb04da7ad163363048e3a94f4251bbde2d54cac73582c8c6a10c0`. Every archive member is embedded byte-exactly as base64, and all eight report objects plus both reviews retain their original numeric text and terminal line endings.
