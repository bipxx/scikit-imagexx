# Combined inertia pipeline: controlled K=3 comparison

[ABBA40387745457](https://github.com/bipxx/scikit-imagexx/actions/runs/40387745457) compares baseline `812a87003898cbe41706933ac9399648d0604316` with combined candidate `070acc7590b1c60444fd49364894428b2b04533b` on one **AMD EPYC 9V74 80-Core Processor**, using one worker. Across 12 fixed cases, direct gain is **3.150593082x** and reference-normalized gain is **3.137168950x**. All 12 cases beat both baseline passes under both metrics; there are no regressions against the previous C++ implementation.

**Eight of twelve cases beat scikit-image in both candidate passes. Four repeat losses remain**, all at 1024-square uint8/float64 inputs: tensor +21.501/+159.106 microseconds and eigenvalues +19.084/+267.106 microseconds. The table keeps those losses visible. Balanced original/current scores move **0.366327875x to 1.149232434x**.

The [exact compact artifact](inertia-pipeline-k3-same-host.zip) preserves four complete two-API reports, all **672 raw latency samples**, eight identity checks and CPU metadata. The earlier [tensor-only campaign](inertia-k3-same-host.md) and both initial reports ([tensor](inertia-tensor-minimum-initial.md), [eigenvalues](inertia-eigvals-minimum-initial.md)) remain separate; gains are not multiplied across hosts or cohorts.

## Every case

| API | Image size | Dtype | Baseline ms | Candidate ms | Direct gain | Normalized gain | Original/current | Added vs original us |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `inertia_tensor` | 256x256 | uint8 | 0.206924 | 0.104308 | 1.983775352x | 2.002079047x | 1.380413678x | -39.680 |
| `inertia_tensor` | 256x256 | float32 | 0.102698 | 0.087772 | 1.170064569x | 1.175338383x | 1.347260947x | -30.480 |
| `inertia_tensor` | 256x256 | float64 | 0.186561 | 0.097952 | 1.904614843x | 1.891370115x | 1.323532865x | -31.691 |
| `inertia_tensor` | 1024x1024 | uint8 | 6.950436 | 0.786787 | 8.833950677x | 8.790333832x | 0.972672198x | +21.501 |
| `inertia_tensor` | 1024x1024 | float32 | 2.482319 | 0.551262 | 4.502978257x | 4.315815430x | 1.183253317x | -101.021 |
| `inertia_tensor` | 1024x1024 | float64 | 5.527886 | 0.753536 | 7.335927182x | 7.187345377x | 0.788853830x | +159.106 |
| `inertia_tensor_eigvals` | 256x256 | uint8 | 0.214008 | 0.128077 | 1.670934581x | 1.689356193x | 1.395498123x | -50.654 |
| `inertia_tensor_eigvals` | 256x256 | float32 | 0.124740 | 0.107634 | 1.158930516x | 1.161651667x | 1.447865947x | -48.206 |
| `inertia_tensor_eigvals` | 256x256 | float64 | 0.209596 | 0.119708 | 1.750891256x | 1.725336395x | 1.367352691x | -43.975 |
| `inertia_tensor_eigvals` | 1024x1024 | uint8 | 6.928277 | 0.828575 | 8.361679146x | 8.220477138x | 0.976967861x | +19.084 |
| `inertia_tensor_eigvals` | 1024x1024 | float32 | 2.524157 | 0.583133 | 4.328616753x | 4.314205389x | 1.240696794x | -140.358 |
| `inertia_tensor_eigvals` | 1024x1024 | float64 | 5.549725 | 0.918548 | 6.041848220x | 6.236324603x | 0.709208477x | +267.106 |

## Group results

| Group | Direct gain | Normalized gain | Original wins in both candidate passes |
|---|---:|---:|---:|
| All 12 | 3.150593082x | 3.137168950x | 8/12 |
| `measure.inertia_tensor` | 3.299412022x | 3.265950275x | 4/6 |
| `measure.inertia_tensor_eigvals` | 3.008486573x | 3.013465667x | 4/6 |
| Large 1024-square | 6.321503454x | 6.262204082x | 2/6 |
| Small 256-square | 1.570233543x | 1.571623808x | 6/6 |

## Method, qualification and remaining work

The prospective fixed set is two automatic image APIs by sizes 256/1024 and actual uint8/float32/float64 image dtypes. Public timing includes moments, tensor/eigenvalue arithmetic, allocations, conversion, clipping and output construction. Frozen cell-specific checks prepare genuine original data outside timing. Order is baseline B1, candidate C2, candidate C3, baseline B4; each uses a fresh process and seven warmed paired samples. Balanced times are geometric means of the two role medians. Direct gain compares baseline/current with candidate/current; normalized gain compares their original/current ratios. Positive added time in the table is a loss to the original.

Both authenticated 14-wheel products are installed offline on a fresh host with 13 identical dependency wheels. Source, package, core, runtime and SDK identities match before and after all four passes. There are no builds or pytest on the measurement host. Baseline Release40383375851 is reused; combined preparation40386034670 passes all nine normal/scalar contract pairs and 1,365 smoke checks while skipping all eight standalone timings. Only the declared K=3 AVX2 leaf differs in production; API wrappers, benchmarks, fixtures and policies are fixed.

The actual combined package separately passes 1,548 component and 176 shared checks across both runtime profiles and dispatch modes. All old eigenvalue/tensor/moment records and arrays remain exact. This is a minimum-Linux one-host result, not full supported-profile qualification; accepted coverage stays 203 image APIs. NumPy preparation, exceptional arithmetic, clipping and list wrapping remain explicit providers.

The four remaining original-library losses motivate a new, separately frozen large-double row-streaming experiment. That experiment retains this 12-case workload and all numeric bounds; no gain is claimed before its own measurement.

Exact ZIP: 31,065 bytes, artifact12682146610, SHA-256 `e904017e9b228a26a85077f101b1633e2322e41f94e93408cbc5372fb0738663`. Final review SHA-256: `0add928a5992d9e9945c06a3938d89967744c7ce7c3a29de72f305ec85013f56`.
