# Inertia tensor K=3 fusion: same-host comparison

[ABBA run 40384025458](https://github.com/bipxx/scikit-imagexx/actions/runs/40384025458) compares baseline `7456bfd5c03c32dcb617ff05f8ef1b8d9b21e3aa` with candidate `a60df8eed63bce619de978072c6253d252c17f70` on one **AMD EPYC 7763 64-Core Processor**, using one worker. The fixed six-case geometric mean improves **2.415797494x directly / 2.387212955x after reference normalization**. The candidate is kept and merged as `046f406`.

All six cases beat scikit-image in both candidate passes. There are **no direct or normalized regressions** in this cohort. The three 1024-square targets improve **4.442269025x directly / 4.379652768x normalized**; both candidate passes beat both baselines for each target under both metrics. Balanced original/current scores move **0.651304052x to 1.554801471x** on this host.

The [exact compact artifact](inertia-k3-same-host.zip) retains all **336 raw latency samples**, four complete reports, eight pre/post identity checks and CPU metadata. The [first six-case report](inertia-tensor-minimum-initial.md), including all its large-image losses, remains unchanged.

## All six cases

| Image size | Dtype | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added ms | Original / candidate time |
|---|---|---:|---:|---:|---:|---:|---:|
| 256x256 | uint8 | 0.298750 | 0.202149 | 1.477867306x | 1.441707520x | -0.096601 | 1.544203475x |
| 256x256 | float32 | 0.191446 | 0.173145 | 1.105698427x | 1.113202359x | -0.018301 | 1.680163310x |
| 256x256 | float64 | 0.262032 | 0.188833 | 1.387638873x | 1.372703229x | -0.073199 | 1.586090888x |
| 1024x1024 | uint8 | 5.133375 | 1.029882 | 4.984428416x | 4.763195983x | -4.103493 | 1.560752732x |
| 1024x1024 | float32 | 2.567002 | 0.854813 | 3.002999486x | 3.001139358x | -1.712189 | 1.461957404x |
| 1024x1024 | float64 | 5.748242 | 0.981502 | 5.856578203x | 5.876712241x | -4.766741 | 1.504515759x |

## Method and validation

The fixed workload is six automatic `measure.inertia_tensor(image)` calls: sizes 256/1024 and actual image dtypes uint8/float32/float64. Each timed call includes image moments, tensor preparation and allocation. The frozen per-cell checker prepares its genuine original central moments outside timing. Pass order is baseline B1, candidate C2, candidate C3, baseline B4, using fresh processes and seven warmed paired samples per case. Balanced times are geometric means of the two role medians. Direct gain compares baseline/candidate times; normalized gain compares their original/current ratios. Positive added time would indicate a regression.

Both archived Release wheels are installed offline on a fresh measurement host, with 13 identical dependency wheels. Source, runtime, installed package, core and SDK identities are authenticated before and after each pass. No builds or pytest run on that host. Baseline is reused from Release40378736391; candidate preparation40382437574 skips all seven standalone timing steps while retaining all eight normal/scalar contract pairs and 1,361 smoke checks.

The sole production change is three added/two removed lines in the AVX2 moment projection dispatcher: instantiate the existing fused ascending reduction for K=3, alongside K=2/K=4. Arithmetic formulas, reduction order, dtype rules, worker policy, fixtures and numeric bounds remain unchanged. Focused native validation has 232 cases in each of eight kept/candidate normal/scalar/sanitizer runs; output bytes and FP flags match exactly, with no FMA in the inspected code. Public components pass 1,980 checks across both runtimes/dispatch modes; all 1,952 retained numerical records and arrays match the kept baseline.

This is a one-host, six-case result for the exact measured candidate. Main also includes the independently validated eigenvalue API; this wheel is not a combined-main or full supported-profile qualification. NumPy preparation, diagonal summation/indexing and exceptional tensor arithmetic remain explicit providers. Accepted supported-profile coverage stays 203 image APIs.

Archive SHA-256: `d879a68ff918f998b5199fe7d0de22e0bdb08244688a31947db87f5ecff77751` (25,685 bytes, artifact12679233383). Final review SHA-256: `2fd879fc918b52a9900f11abffe674e3fcf2a58f9a729d1b74abbc4f2462e964`.
