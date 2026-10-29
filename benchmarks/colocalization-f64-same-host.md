# Float64 finite classifier: discarded experiment

**DISCARD — candidate source remains unmerged.** All four target cohorts pass the repeated 1.05 direct and normalized floors, and all five existing original wins remain wins in both candidate passes. Three controls fail the strict 0.97 normalized floor. Aggregate gains do not override these failures.

[Run 40798470833](https://github.com/bipxx/scikit-imagexx/actions/runs/40798470833) compared genuine baseline `4f69eba` with private candidate `d496dd1` in four independent ABBA processes, one worker and one fixed CPU. All 1,176 accepted batches were at least 100 ms; all 168 function warmup groups accumulated at least 250 ms. The complete reports, calibration journals, runtime identities and all losses are preserved in [the JSON archive](colocalization-f64-same-host.json).

| Case | Direct baseline/current | Normalized gain | CPP added ms | B2 original/current | B3 original/current |
|---|---:|---:|---:|---:|---:|
| intersection_coeff/future/tiny | 1.004075816× | 0.999210030× | -0.000043 | 1.635913147× | 1.642411044× |
| intersection_coeff/future/cache | 1.011855975× | 0.999173115× | -0.000281 | 2.005732281× | 2.020167079× |
| intersection_coeff/future/memory | 0.993742499× | 0.806665663× | +0.001930 | 2.601325828× | 2.622759890× |
| intersection_coeff/future/volume | 1.000449193× | 0.997087449× | -0.000028 | 2.195705878× | 2.191921334× |
| intersection_coeff/future/stepped | 0.977498755× | 0.975980912× | +0.019931 | 0.316748216× | 0.314999953× |
| intersection_coeff/future/masked | 0.987922946× | 0.982085586× | +0.005382 | 0.816118713× | 0.813230401× |
| manders_coloc_coeff/future/tiny | 1.034393877× | 1.041282901× | -0.000830 | 0.519175836× | 0.513258723× |
| manders_coloc_coeff/future/cache | 1.104929734× | 1.096801118× | -0.049298 | 0.125771275× | 0.130154932× |
| manders_coloc_coeff/future/memory | 2.037128403× | 2.044286212× | -8.412578 | 0.230480285× | 0.232954725× |
| manders_coloc_coeff/future/volume | 2.123664330× | 2.138013895× | -2.288638 | 0.232393713× | 0.231280617× |
| manders_coloc_coeff/future/stepped | 1.147006008× | 1.146856473× | -0.138826 | 0.178869737× | 0.180348398× |
| manders_coloc_coeff/future/masked | 1.040377925× | 1.049045203× | -0.029434 | 0.529651520× | 0.508708030× |
| manders_overlap_coeff/future/tiny | 1.011431812× | 0.999172624× | -0.000555 | 0.370549653× | 0.366426930× |
| manders_overlap_coeff/future/cache | 1.037908715× | 0.944767299× | -0.041782 | 0.077299502× | 0.079746867× |
| manders_overlap_coeff/future/memory | 2.021716121× | 2.157708492× | -19.255804 | 0.224709919× | 0.224698489× |
| manders_overlap_coeff/future/volume | 2.192006150× | 2.173671052× | -5.215364 | 0.214067296× | 0.218918715× |
| manders_overlap_coeff/future/stepped | 1.124395335× | 1.112523108× | -0.244131 | 0.117974827× | 0.117475200× |
| manders_overlap_coeff/future/masked | 1.041811241× | 1.025733028× | -0.047633 | 0.344223463× | 0.347434140× |
| intersection_coeff/future/cache-F | 1.002969793× | 1.002586927× | -0.000070 | 2.051495724× | 2.010683954× |
| manders_coloc_coeff/future/cache-F | 1.103020227× | 1.094194844× | -0.048363 | 0.126389948× | 0.127619422× |
| manders_overlap_coeff/future/cache-F | 1.046126310× | 0.969737974× | -0.050625 | 0.079300730× | 0.081426748× |

Balanced direct geometric mean is **1.186535612974×**; normalized geometric mean is **1.165305226268×**. Failed normalized controls: intersection/memory **0.806665663×**, overlap/cache **0.944767299×**, overlap/cache-F **0.969737974×**. Nine cases retain a balanced direct or normalized CPP loss. Sixteen cases still lose to scikit-image in both candidate passes.

The change classifies finite float64 words using integer bits; it does not remove scans, own all scalar arithmetic, or establish stage-level causality. Complete public calls include provider work, allocation and destruction. No further variant or timing retry is part of this experiment.

Product identity is 247 partial image APIs plus three metadata APIs, 176 product files and 83 SDK files. The genuine minimum preparation passed 9,346 focused checks and 1,447 smoke checks; the new finite-f64 installed C++ client separately passed one family/two modes with workers 1/4. Supported-profile qualification remains the separately accepted 241-API baseline.

The original protocol and zero-image-call missing-threadpoolctl preflight failure remain intact. V2 supplies one authenticated threadpoolctl 3.6.0 wheel equally to both benchmark runtimes, separate from the unchanged 13 product dependencies. The Windows reviewer encountered one geometric-mean ULP difference; the exact unchanged frozen reviewer passed under Linux stdlib. This did not rerun measurements or original algorithms.
