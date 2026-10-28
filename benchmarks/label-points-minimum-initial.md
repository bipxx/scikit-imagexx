# LabelPoints: first minimum-profile comparison

**Regression across this cohort: 8 of 9 workloads are slower; geometric mean speed ratio 0.802491952×.** The single ratio above one is 1.007073146× on dense 1024² coordinates; it is a small point-estimate difference. This is the first original/current comparison for `util.label_points` on one minimum Linux host at source `7463a7b4c24b3212aa57757a7bac3305f01fde9f`.

[Raw report](label-points-minimum-initial.json) preserves all 126 timing samples and their original numeric text. Ratio is scikit-image median / C++ public-call median; below one means slower. Added time is C++ minus original. No performance superiority or prior-C++ gain is established.

| Workload | Coordinate dtype | Output shape | Points | Original ms | C++ ms | Ratio | Added ms |
|---|---|---|---:|---:|---:|---:|---:|
| sparse-64x64-i64 | int64 | 64 × 64 | 16 | 0.005268738 | 0.005552985 | 0.948811878× | +0.000284247 |
| dense-64x64-f64 | float64 | 64 × 64 | 4,096 | 0.020722437 | 0.028663573 | 0.722953725× | +0.007941136 |
| sparse-1024x1024-i64 | int64 | 1024 × 1024 | 16 | 0.067334503 | 0.071660099 | 0.939637313× | +0.004325596 |
| dense-1024x1024-f64 | float64 | 1024 × 1024 | 1,048,576 | 10.557462000 | 10.483312000 | 1.007073146× | -0.074150000 |
| sparse-64x64x64-f32 | float32 | 64 × 64 × 64 | 32 | 0.023881502 | 0.024171368 | 0.988007864× | +0.000289866 |
| dense-64x64x64-i64 | int64 | 64 × 64 × 64 | 262,144 | 0.994362769 | 1.921109444 | 0.517598189× | +0.926746675 |
| duplicates-1024x1024-f64 | float64 | 1024 × 1024 | 65,536 | 0.307256814 | 0.455363579 | 0.674750524× | +0.148106765 |
| strided-1024x1024-f32 | float32 | 1024 × 1024 | 65,536 | 0.718922750 | 0.854024368 | 0.841805898× | +0.135101618 |
| strided-64x64x64-i64 | int64 | 64 × 64 × 64 | 32,768 | 0.295603516 | 0.403797522 | 0.732058768× | +0.108194006 |

| Cohort | Ratios above one / cases | Geometric mean |
|---|---:|---:|
| dense | 1 / 3 | 0.722306331× |
| duplicates | 0 / 1 | 0.674750524× |
| sparse | 0 / 3 | 0.958591535× |
| strided | 0 / 2 | 0.785016807× |

The largest penalty is `dense-64x64x64-i64`: 0.994362769 ms becomes 1.921109444 ms (+0.926746675 ms), a 0.517598189× ratio. All three sparse workloads, both strided workloads and the duplicate workload are slower. These whole-call results identify unresolved workloads; they do not attribute the losses to a particular stage.

## Measurement and correctness scope

Seed 2026; seven paired rounds targeting 19 ms, one global worker, all four reference pool limits set to one. Every timing includes the whole public call: negative-coordinate checking, NumPy round/int conversion/transpose, output allocation, owned zero-fill and ordered scatter, and output destruction. Coordinate generation, independent oracle, exact parity and native admission checks are outside timing.

The nine frozen workloads include sparse and dense 64², 1024² and 64³ output masks, a duplicate-heavy 1024² case, and two row-strided/column-reversed coordinate views. Real coordinate dtypes are int64, float32 and float64. These are direct coordinate workloads, separate from the shared image-to-coordinate default adapter. There is no preallocated-output timing case.

Before timers, fresh scikit-image 0.26 and the native public API match full output words, dtype, shape, strides, ownership and flags against an independent reverse-point-order bytearray oracle using Python nearest-even rounding. The original whole API is blocked during native precheck; actual C returns and the frontend native-result local prove admission. Inputs and backing gaps, NumPy error policy and global worker policy remain unchanged.

The provider boundary remains explicit: NumPy handles the literal validation/rounding/cast/transpose prelude; C++ owns zero-fill and serial point-order scatter. SIMD and optional parallel zero-fill are implemented, but this report measures one worker only. Exceptional/protocol/scalar behavior is covered by separate frozen parity tests and is outside this timing cohort.

The genuine producer passed 28 focused normal/scalar XML pairs (6,270 checks), 3,696 color regressions, 1,612 matching checks and 1,427 smoke checks across 243 registered functions. Additional Blur/Polygon/Euler boundary guards passed (4 each), plus 9 integer-width guards. Source coverage is 240 image APIs with partial native support and 3 metadata APIs. This single-host result does not promote full four-profile qualification.

## Provenance

- [Genuine run 40689772340](https://github.com/bipxx/scikit-imagexx/actions/runs/40689772340), isolated `measure-minimum` after `numpy-minimum`; 22 older timing steps skipped.
- [Artifact 12817656081](https://github.com/bipxx/scikit-imagexx/actions/runs/40689772340/artifacts/12817656081): `ci-performance-minimum`, 84,788,517 bytes, 82 members, downloaded once; every member CRC/SHA verified.
- Host: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`; CPU model was not recorded. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0; AVX2 enabled. Report timestamp: 2026-10-28T12:49:31.043231+00:00.
- Exact source, installed core, SDK, product wheel, 13 runtime dependencies, original Python source and before/after runtime identities authenticated. The committed manifest contains 1,738 blobs; product wheel 158 members, SDK 71.
- [Pinned harness](https://github.com/bipxx/scikit-imagexx/blob/7463a7b4c24b3212aa57757a7bac3305f01fde9f/benchmarks/label_points.py) and its prospective workload contract are unchanged from dispatch.

| Proof | SHA-256 |
|---|---|
| Raw report | `c91865cf6d85a84bc072651e87b38743c95d534a588cf39e0eb04b7d430ba365` |
| Artifact ZIP | `142fd759cc792b0c34485a496756f48614f273574250af87e6b439467ae570d6` |
| Harness | `903c46306ed0a3e7a57d21a32e794fb9b48fe6af7ea7b685d7cf3ea66b1d0850` |
| Prospective workload contract | `3fceb3b66c254f1330f30f779a34a042162c08cf59921bf1343b06376f450221` |
| Original Python source | `8c409baaf3527b3466c8d8e973e4482182bdf172e8486c33d91eb469403742b2` |
| Installed native core | `4212f5f69be2d060acb160fc5a93a75295cdd95c19cb831aa9010d82c4da4821` |
| Canonical product source fingerprint | `6ee770b75e824e5a3afcf9dfd586f677f98b80f340f8b9b910009c6840c9d94c` |
| Independent review | `610a1765ded87028a94170dba604228428187f22e66f659c7e360a17a8bc33c8` |
