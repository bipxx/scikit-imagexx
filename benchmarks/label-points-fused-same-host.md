# LabelPoints: fixed same-host comparison

**KEEP under the frozen gates.** Balanced C++ gain is 1.677277611×; reference-normalized gain is 1.671533208×. Candidate original/current ratio is 1.340704225× versus baseline 0.802080520× on this host.

[Combined raw evidence](label-points-fused-same-host.json) preserves all four reports, all 504 timing samples, the comparison, independent review and frozen contract without rewriting original numeric text. Direct gain is baseline C++ / candidate C++; normalized gain adjusts that comparison using the paired original timings. Values below one identify losses. Added time is candidate minus baseline.

| Workload | Baseline C++ ms | Candidate C++ ms | Direct gain | Normalized gain | Original/candidate | Added ms |
|---|---:|---:|---:|---:|---:|---:|
| sparse-64x64-i64 | 0.006146400 | 0.006321703 | 0.972269717× | 0.992781137× | 0.949976059× | +0.000175303 |
| dense-64x64-f64 | 0.031654212 | 0.015343996 | 2.062970489× | 2.046136360× | 1.497427436× | -0.016310215 |
| sparse-1024x1024-i64 | 0.085476565 | 0.084469438 | 1.011922977× | 1.016073716× | 0.901124009× | -0.001007127 |
| dense-1024x1024-f64 | 10.762506406 | 6.421737082 | 1.675949399× | 1.672097467× | 1.730233451× | -4.340769323 |
| sparse-64x64x64-f32 | 0.027926587 | 0.026894471 | 1.038376536× | 1.015748329× | 0.992706431× | -0.001032117 |
| dense-64x64x64-i64 | 2.097169012 | 0.426858312 | 4.913033087× | 4.835604804× | 2.602911106× | -1.670310700 |
| duplicates-1024x1024-f64 | 0.527459177 | 0.229173493 | 2.301571485× | 2.283550887× | 1.518187722× | -0.298285684 |
| strided-1024x1024-f32 | 0.962193670 | 0.676990694 | 1.421280498× | 1.391776821× | 1.169026708× | -0.285202977 |
| strided-64x64x64-i64 | 0.446189742 | 0.241085997 | 1.850749307× | 1.890788606× | 1.376001824× | -0.205103745 |

## Frozen decision

Overall direct and normalized geometric means must both exceed 1.05. Dense 64³ int64 must exceed 1.10 in both gains on each candidate pass relative to baseline geometric means. Dense 1024² float64 must remain faster than the original in both candidate passes. Each of the other eight cases must retain balanced direct and normalized gains of at least 0.97. All correctness/identity checks and the fixed deadline are required.

Dense-3D repeated target gates passed: 1 / 1. Preserved original-win gates passed: 1 / 1. Other-case floor failures: none. Completed 2026-10-28T13:36:51.067414+00:00 before deadline 2026-10-28T14:58:45+00:00: True.

## Retained losses

Direct C++ losses: sparse-64x64-i64. Reference-normalized losses: sparse-64x64-i64.
Candidate balanced losses to the original: sparse-64x64-i64, sparse-1024x1024-i64, sparse-64x64x64-f32.

- `2-candidate` loses to the original on: sparse-64x64-i64, sparse-1024x1024-i64, sparse-64x64x64-f32.
- `3-candidate` loses to the original on: sparse-64x64-i64, sparse-1024x1024-i64.

The [first original/current report](label-points-minimum-initial.md) retains all eight initially slower workloads and its 126 samples separately. Every one of those workloads is retained here; a KEEP decision does not mean every workload beats the original or that the full library is faster.

## Scope and provenance

One genuine minimum Linux host, one worker, seven paired rounds per report, 19 ms target, seed 2026; fixed baseline/candidate/candidate/baseline order. Both products retain exactly the same 13 dependency wheels and scientific packages. The whole public call includes output allocation/destruction and the unchanged NumPy prelude. Coordinate generation and independent exact-word/native-admission checks are outside timers.

The candidate fuses bounds validation, offset calculation and ordered scatter, with rank-2/rank-3 specialization. This tests the bundle; it does not measure individual-stage causality. Duplicate scatter remains serial, optional workers cover disjoint zero-fill, and these measurements use one worker only. The Python frontend, public C signature, zero-fill SIMD, harness, nine workloads and frozen 48-case references remain unchanged.

Both genuine preparations and all eight before/after measurement identities passed. All inputs, backing gaps, output storage metadata and original/native/oracle output hashes match; actual genuine C admission and original-forwarding guards run before timers. This is a scoped performance experiment, without full four-profile qualification credit.

- [Run 40696160717](https://github.com/bipxx/scikit-imagexx/actions/runs/40696160717); CPU: AMD EPYC 9V45 96-Core Processor.
- Baseline source `7463a7b4c24b3212aa57757a7bac3305f01fde9f`; candidate `fd8b792b82ac1e9635d5d1431776275f5dc882bc`.
- [Comparison artifact 12820176938](https://github.com/bipxx/scikit-imagexx/actions/runs/40696160717/artifacts/12820176938), 38,267 bytes, 39 members; retained once with every member CRC/SHA verified.
- Artifact SHA-256: `e5c8a62311ee150a9babfdd90b03ac33cc16e4980c1f97b97f172686c2f77790`.
- Frozen contract SHA-256: `7c0f0269bf7ffc60cf7c34a1c919444e9554df2b3d327870c021161356d6354d`.
- Independent review SHA-256: `74120d6950706a04c49451f87744d9ef574660ff29d5e2b13d0d3b48aaabb9ed`.
