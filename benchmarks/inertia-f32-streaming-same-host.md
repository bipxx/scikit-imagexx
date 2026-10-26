# Float32 tiled streaming: same-host comparison

[ABBA40403106534](https://github.com/bipxx/scikit-imagexx/actions/runs/40403106534) compares baseline `64b7fca8537b6dee03e3374d6b3c70c5ad979001` with candidate `e360eeec30ac268bb593c004bb6c6b5dc37f9621` on one AMD EPYC 9V74 host, using one global worker.

The fixed12 geometric-mean gain is **1.119084834x directly / 1.122603981x reference-normalized**. The two targeted 1024-square float32 calls improve **1.922135447x / 1.842475618x**. The ten controls give **1.004335502x / 1.016697004x**. The candidate meets the frozen aggregate keep criteria and is merged as `b29377e`; this does not assert every control improves.

**4 direct and 2 normalized losses are retained below.** **12/12 cases beat scikit-image in both candidate passes.** Balanced original/current scores move 1.392543525x to 1.563274906x. The [exact compact artifact](inertia-f32-streaming-same-host.zip) retains all **672 raw samples**, source identities and every case. The [previous double-streaming result](inertia-streaming-same-host.md) remains unchanged.

## Every case

| API | Image size | Dtype | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added vs baseline us | Original/current |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `inertia_tensor` | 256x256 | uint8 | 0.105783 | 0.103804 | 1.019062959x | 1.045683476x | -1.979 | 1.394461707x |
| `inertia_tensor` | 256x256 | float32 | 0.084578 | 0.085099 | 0.993887853x | 1.023547305x | +0.520 | 1.419172003x |
| `inertia_tensor` | 256x256 | float64 | 0.097741 | 0.096566 | 1.012166443x | 1.034555446x | -1.175 | 1.343220980x |
| `inertia_tensor` | 1024x1024 | uint8 | 0.543671 | 0.546085 | 0.995578928x | 0.989852145x | +2.414 | 1.378367182x |
| `inertia_tensor` | 1024x1024 | float32 | 0.502006 | 0.255836 | 1.962216073x | 1.860766779x | -246.170 | 2.487332775x |
| `inertia_tensor` | 1024x1024 | float64 | 0.391602 | 0.391758 | 0.999601507x | 1.027687709x | +0.156 | 1.539055435x |
| `inertia_tensor_eigvals` | 256x256 | uint8 | 0.125146 | 0.125123 | 1.000180477x | 1.017351999x | -0.023 | 1.424015554x |
| `inertia_tensor_eigvals` | 256x256 | float32 | 0.104300 | 0.103741 | 1.005384043x | 1.009153429x | -0.559 | 1.486142338x |
| `inertia_tensor_eigvals` | 256x256 | float64 | 0.118475 | 0.116485 | 1.017087085x | 1.011740725x | -1.990 | 1.403223767x |
| `inertia_tensor_eigvals` | 1024x1024 | uint8 | 0.567241 | 0.573752 | 0.988652615x | 0.993687836x | +6.511 | 1.379719064x |
| `inertia_tensor_eigvals` | 1024x1024 | float32 | 0.529875 | 0.281418 | 1.882873516x | 1.824364258x | -248.457 | 2.422048101x |
| `inertia_tensor_eigvals` | 1024x1024 | float64 | 0.413688 | 0.408686 | 1.012238832x | 1.015019450x | -5.002 | 1.530405804x |

## Repeat comparisons

| Case | Both candidate passes beat both baselines, direct | Same, normalized | Both candidate passes lose to both baselines, direct | Same, normalized |
|---|---|---|---|---|
| measure.inertia_tensor / 256 / uint8 | no | yes | no | no |
| measure.inertia_tensor / 256 / float32 | no | no | yes | no |
| measure.inertia_tensor / 256 / float64 | no | no | no | no |
| measure.inertia_tensor / 1024 / uint8 | no | no | no | no |
| measure.inertia_tensor / 1024 / float32 | yes | yes | no | no |
| measure.inertia_tensor / 1024 / float64 | no | no | no | no |
| measure.inertia_tensor_eigvals / 256 / uint8 | no | no | no | no |
| measure.inertia_tensor_eigvals / 256 / float32 | no | no | no | no |
| measure.inertia_tensor_eigvals / 256 / float64 | yes | no | no | no |
| measure.inertia_tensor_eigvals / 1024 / uint8 | no | no | yes | no |
| measure.inertia_tensor_eigvals / 1024 / float32 | yes | yes | no | no |
| measure.inertia_tensor_eigvals / 1024 / float64 | no | no | no | no |

## Method and validation

The unchanged fixed12 cohort covers inertia_tensor and inertia_tensor_eigvals at 256/1024 square, uint8/float32/float64, with one global worker. Public-call timings include preparation, allocation and result construction. Reference preparation and frozen numerical verification are outside timing. Fresh-process B1-C2-C3-B4 passes retain seven paired samples per case. Balanced times are geometric means of role medians; normalized gain compares original/current ratios. Positive added time is a regression.

Both genuine Release products install offline with 13 identical dependency wheels. All eight pre/post source/package/core/runtime/SDK identities pass. The measurement host runs no builds or tests. Candidate Release40401421542 passes nine normal/scalar contract pairs and 1,365 smoke checks, skipping standalone timing steps. Baseline Release40392230688 is reused without rebuilding.

The production change enables the existing bounded K=3 tiled AVX2 traversal for float32 as well as double and types its local accumulator buffer accordingly. The n>=256/inner>=512 gate, complete worker-owned blocks, ascending reduction order, formulas and numerical policies remain unchanged. Actual packed float32 multiply/add instructions are inspected without FMA. Native kept/candidate normal/scalar/sanitizer controls agree exactly on outputs and FP streams; 2,256 public component checks preserve all 2,224 numerical rows and array contents.

This is one minimum-Linux host and the exact measured source, which predates the independent coordinate and Shannon additions. It is not a combined-main or full supported-profile qualification; accepted supported-profile coverage stays 203 image APIs.

Exact ZIP: 31,390 bytes, artifact12689064076, SHA-256 `774254a1b69d28f0e9f843c70100ec54cea136d81478ae017e457fb2223008fc`.
