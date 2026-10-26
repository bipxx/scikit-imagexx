# Coordinate moments: first minimum-Linux comparison

[Release40394363351](https://github.com/bipxx/scikit-imagexx/actions/runs/40394363351) measures exact source `a1dafca0815993397febe10de303dfbcf0ba7b50`, merged as `d1b6198`. **All 12 cases beat scikit-image**, with geometric mean **2.273374651x**. Raw coordinate moments give **2.712598165x** and central coordinate moments **1.905270147x**. There are no losses in this cohort. The [exact raw JSON](moments-coords-minimum-initial.json) preserves all **168 latency samples**.

## Every case

| API | Points | Actual input shape | Actual dtype | Original ms | Current ms | Original/current | Added ms |
|---|---:|---|---|---:|---:|---:|---:|
| `moments_coords` | 256 | 256x2 | uint8 | 0.076793 | 0.029101 | 2.638868930x | -0.047692 |
| `moments_coords_central` | 256 | 256x2 | uint8 | 0.105447 | 0.057458 | 1.835203489x | -0.047989 |
| `moments_coords` | 256 | 256x2 | float32 | 0.069132 | 0.026533 | 2.605546158x | -0.042599 |
| `moments_coords_central` | 256 | 256x2 | float32 | 0.107509 | 0.058597 | 1.834733289x | -0.048913 |
| `moments_coords` | 256 | 256x2 | float64 | 0.075466 | 0.029319 | 2.573930951x | -0.046147 |
| `moments_coords_central` | 256 | 256x2 | float64 | 0.100816 | 0.052153 | 1.933072678x | -0.048663 |
| `moments_coords` | 1024 | 1024x2 | uint8 | 0.184172 | 0.065171 | 2.825968546x | -0.119001 |
| `moments_coords_central` | 1024 | 1024x2 | uint8 | 0.241658 | 0.124734 | 1.937387308x | -0.116924 |
| `moments_coords` | 1024 | 1024x2 | float32 | 0.157651 | 0.056369 | 2.796765867x | -0.101282 |
| `moments_coords_central` | 1024 | 1024x2 | float32 | 0.253208 | 0.137969 | 1.835255445x | -0.115239 |
| `moments_coords` | 1024 | 1024x2 | float64 | 0.182585 | 0.064105 | 2.848235297x | -0.118480 |
| `moments_coords_central` | 1024 | 1024x2 | float64 | 0.230664 | 0.111599 | 2.066907565x | -0.119065 |

## Method and implementation

Each input is an owning N-by-2 coordinate cloud, prepared outside timing from the first 2*N seeded gray values while retaining the actual coordinate dtype. N is 256 or 1024 points; these are not N-square images, image labels, or duplicate workloads. Each public call includes centering where applicable, casts, coordinate powers, allocation, ordered monomial products and reduction. One global worker and seven warmed paired rounds compare both implementations in a fresh offline process. Reference monomials for the frozen 2*gamma_(N+D)*S allowance are captured outside timing.

C++ owns ordinary real float32/float64 products and point reductions for assessed dimensions1..3 and K>1, without materializing the large N*K^D product tensor. SIMD spans independent output cells and preserves each cell's ascending point order, with no FMA/reassociation. NumPy tuple/center/cast/power preparation remains explicit. K1, complex, unsupported and exceptional reductions retain the literal primitive route without native reduction credit. Raw float32 and automatic-center dtype promotions follow the pinned original, including runtime-specific scalar-center promotion.

The installed genuine product passes ten public normal/scalar contract pairs (2,622 checks) and 1,373 smoke checks. Two-profile components separately pass 1,940 family and 176 shared checks; prior records and all 716 old lossless array archives remain exact. Native controls pass normal/scalar/sanitizer modes; actual portable/no-OpenMP controls and worker refusals are recorded separately. Counts distinguish controls/calls from public API checks.

One recorder-only correction preserves complex input JSON. Another provider-only worker comparison follows observed original-vs-original NaN-sign instability: finite values, signed zeros, Inf/NaN masks, warnings and state stay exact; all raw words and the initial failure remain retained. Native/error word comparisons and all numerical bounds are unchanged.

Source, wheel, runtime, installed package/core and SDK identities match before and after timing. The host is x86_64 minimum Linux (CPython3.11.17/NumPy1.24.4/SciPy1.15.3); a detailed CPU model was not captured. This first original-library comparison is not a prior-C++ gain, cross-CPU claim or full supported-profile qualification. Source coverage is 225 native-partial image APIs plus three metadata functions; accepted full-profile coverage remains 203 image APIs.

Raw JSON SHA-256: `8ba1299fc45c110eb104c1f2d4876fc381817482138078f99d3de2372f508262`. Parent artifact12686047919 SHA-256: `6d930078a1146ab6552f8881a4052dbe1e985fc6d94f8dff702984003894cdf6`. Final review: `4bad9d17d47ecf17652d5aeaaeeebbd975adcaa34e92bc6356254b89d091a1f9`.
