# Large-double tiled streaming: same-host comparison

[ABBA40394034621](https://github.com/bipxx/scikit-imagexx/actions/runs/40394034621) compares baseline `070acc7590b1c60444fd49364894428b2b04533b` with candidate `64b7fca8537b6dee03e3374d6b3c70c5ad979001` on one **AMD EPYC 9V74 80-Core Processor**, using one worker. Fixed12 direct gain is **1.110031361x** and reference-normalized gain **1.137751467x**. The four targeted large uint8/float64 calls improve **1.49886244x directly / 1.52056178x normalized**; each beats both baseline passes under both metrics. The candidate meets the frozen aggregate acceptance criteria and is kept, merged as `cbf3050`.

**Six direct and four normalized regressions remain visible.** The eight controls regress to **0.95526086x directly / 0.98416762x normalized**. The largest direct penalties are 1024-square float32 tensor/eigenvalue calls (+59.339/+57.945 microseconds). Tensor256 uint8 is slower in both candidate passes than both baselines directly; tensor256 float64 repeats a normalized loss. These tradeoffs are retained rather than described as uniform improvement.

**Ten of twelve cases beat scikit-image in both candidate passes.** The two 1024-square float32 cases are mixed across passes, with balanced original-library penalties +37.122/+22.566 microseconds. Balanced original/current scores move **1.17292471x to 1.33449681x**. The [exact compact artifact](inertia-streaming-same-host.zip) retains every row and all **672 raw latency samples**; the [previous combined campaign](inertia-pipeline-k3-same-host.md) remains unchanged.

## Every case

| API | Image size | Dtype | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added vs baseline us | Original/current |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `inertia_tensor` | 256x256 | uint8 | 0.106016 | 0.114488 | 0.926000123x | 0.957142256x | +8.472 | 1.338484555x |
| `inertia_tensor` | 256x256 | float32 | 0.090903 | 0.092642 | 0.981225983x | 1.024720114x | +1.739 | 1.426769888x |
| `inertia_tensor` | 256x256 | float64 | 0.101779 | 0.107731 | 0.944750321x | 0.972254103x | +5.952 | 1.310919450x |
| `inertia_tensor` | 1024x1024 | uint8 | 0.768646 | 0.564521 | 1.361588983x | 1.372608294x | -204.125 | 1.398407646x |
| `inertia_tensor` | 1024x1024 | float32 | 0.708841 | 0.768181 | 0.922753243x | 0.881468427x | +59.339 | 0.951675456x |
| `inertia_tensor` | 1024x1024 | float64 | 0.659540 | 0.410192 | 1.607880915x | 1.668866908x | -249.348 | 1.522176290x |
| `inertia_tensor_eigvals` | 256x256 | uint8 | 0.133642 | 0.143995 | 0.928103497x | 1.042713377x | +10.353 | 1.464423219x |
| `inertia_tensor_eigvals` | 256x256 | float32 | 0.114931 | 0.113472 | 1.012857233x | 1.051357042x | -1.459 | 1.505401432x |
| `inertia_tensor_eigvals` | 256x256 | float64 | 0.127430 | 0.126602 | 1.006538045x | 1.068418104x | -0.828 | 1.430336571x |
| `inertia_tensor_eigvals` | 1024x1024 | uint8 | 0.818847 | 0.587336 | 1.394170781x | 1.329323431x | -231.511 | 1.398585354x |
| `inertia_tensor_eigvals` | 1024x1024 | float32 | 0.716338 | 0.774283 | 0.925163164x | 0.893980108x | +57.945 | 0.970855736x |
| `inertia_tensor_eigvals` | 1024x1024 | float64 | 0.796947 | 0.481946 | 1.653602863x | 1.755566573x | -315.001 | 1.469192547x |

## Group results

| Group | Direct gain | Normalized gain |
|---|---:|---:|
| Fixed12 | 1.110031361x | 1.137751467x |
| Four large-double targets | 1.498862435x | 1.520561784x |
| Eight controls | 0.955260864x | 0.984167616x |
| `measure.inertia_tensor` | 1.096092241x | 1.115379787x |
| `measure.inertia_tensor_eigvals` | 1.124147746x | 1.160571866x |

## Method and validation

The unchanged fixed12 set covers automatic tensor/eigenvalue calls, image sizes 256/1024, actual uint8/float32/float64 dtypes and one global worker. Each public timing includes preparation, moments, allocation and result construction. Frozen numerical checks prepare original reference data outside timing. B1-C2-C3-B4 use fresh processes and seven warmed paired samples per case. Balanced times are geometric means of role medians; normalized gain compares original/current ratios. Positive added time is a regression to the prior C++ version.

Both authenticated Release products install offline with 13 identical dependency wheels. All eight pre/post source/package/core/runtime/SDK identities pass. There are no builds or tests on the measurement host. Candidate preparation40392230688 passes nine normal/scalar contract pairs and 1,365 smoke checks while skipping all eight standalone timings. The baseline40386034670 is reused without rebuilding.

The only production change is a 29-line AVX2 leaf addition for double K=3 projections with n>=256 and inner>=512 inside complete worker-owned blocks. It streams ascending rows through bounded 128-column tiles. Each output retains the original ascending reduction order; formulas, dtype/worker rules, numerical bounds and fixtures stay unchanged. Small/float32/K2/K4 source paths are retained; the measured control regressions are not assigned a cause here. Native validation has 258 cases in each of eight kept/candidate normal/scalar/sanitizer runs, all output/FP streams exact and no FMA in inspected code. Public components pass 2,256 checks, with all 2,224 numerical rows and arrays exact to the kept baseline.

This is one minimum-Linux host and the exact measured candidate; main also contains the independently validated coordinate APIs. It is not a combined-current-main, cross-CPU, portable-performance or full supported-profile qualification. Accepted supported-profile coverage stays 203 image APIs.

Exact ZIP: 31,406 bytes, artifact12685526866, SHA-256 `6778c87f74c33b8295f2157ac367dc0d608e9942f47d6cbe30b49567e1fec061`. Review SHA-256: `0ef86efaa2f0a5960346a73fae0f726b44d3312e38235bd380513312aa481c91`.
