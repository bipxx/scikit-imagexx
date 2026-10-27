# Profile line: first minimum-profile Release comparison

Four of 18 measured cases are faster than scikit-image 0.26.0; 14 are slower. The equal-case geometric mean of original/current median ratios is **0.977475055×**. Every case records accepted owned sampling, exact pre-reducer sample checks, exact public output bytes and zero error. All 252 raw samples, including every loss, are retained in the [byte-exact JSON](profile-line-minimum-initial.json).

Inputs are seeded square images of side 256 or 1,024, with uint8, float32 or float64 storage. Float images are normalized by 255 before timing. Calls use default order 1, reflect mode and width mean, from `(0.25, 1.5)` to `(side-1.25, side-1.75)`. Profile lengths are 360 or 1,446; sampled values equal length times linewidth.

| Side | Input dtype | Linewidth | Length | Original ms | Current ms | Original/current | Current − original ms | Max error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | uint8 | 1 | 360 | 4.185346500 | 4.156881250 | 1.006847742× | -0.028465250 | 0.0 |
| 256 | uint8 | 9 | 360 | 4.235294250 | 4.274353250 | 0.990862009× | +0.039059000 | 0.0 |
| 256 | uint8 | 33 | 360 | 4.366997000 | 4.600377500 | 0.949269272× | +0.233380500 | 0.0 |
| 256 | float32 | 1 | 360 | 4.182241750 | 4.125309000 | 1.013800845× | -0.056932750 | 0.0 |
| 256 | float32 | 9 | 360 | 4.205126250 | 4.257197000 | 0.987768771× | +0.052070750 | 0.0 |
| 256 | float32 | 33 | 360 | 4.365785250 | 4.582348000 | 0.952739785× | +0.216562750 | 0.0 |
| 256 | float64 | 1 | 360 | 4.206142750 | 4.203185750 | 1.000703514× | -0.002957000 | 0.0 |
| 256 | float64 | 9 | 360 | 4.255484500 | 4.299490250 | 0.989764891× | +0.044005750 | 0.0 |
| 256 | float64 | 33 | 360 | 4.429520500 | 4.624879000 | 0.957759219× | +0.195358500 | 0.0 |
| 1024 | uint8 | 1 | 1,446 | 16.631491000 | 16.526564000 | 1.006348991× | -0.104927000 | 0.0 |
| 1024 | uint8 | 9 | 1,446 | 17.162248000 | 17.252063000 | 0.994793956× | +0.089815000 | 0.0 |
| 1024 | uint8 | 33 | 1,446 | 18.022920000 | 18.859308000 | 0.955651183× | +0.836388000 | 0.0 |
| 1024 | float32 | 1 | 1,446 | 17.098522000 | 17.421187000 | 0.981478587× | +0.322665000 | 0.0 |
| 1024 | float32 | 9 | 1,446 | 17.187326000 | 17.664432000 | 0.972990584× | +0.477106000 | 0.0 |
| 1024 | float32 | 33 | 1,446 | 17.861108000 | 18.766838000 | 0.951737741× | +0.905730000 | 0.0 |
| 1024 | float64 | 1 | 1,446 | 17.056869000 | 17.572594000 | 0.970651743× | +0.515725000 | 0.0 |
| 1024 | float64 | 9 | 1,446 | 17.618684000 | 18.143331000 | 0.971083204× | +0.524647000 | 0.0 |
| 1024 | float64 | 33 | 1,446 | 18.066446000 | 19.125847000 | 0.944608937× | +1.059401000 | 0.0 |

Timed public calls include NumPy coordinate construction, storage preparation and the whole floating-image finite-admission scan, owned typed sampling, flipping and the literal NumPy width mean. These results do not identify which stage causes the losses. Nonfinite images, higher orders and other primitive-provider routes are outside this cohort. Passive sampling callbacks are restored before timing.

The authenticated harness checks complete original/current sample snapshots and public ndarray metadata/bytes before timing. The JSON retains sampled-pixel metadata and SHA, plus complete public output bytes; sampled-pixel payloads themselves are not exported. Float32 inputs return float32 arrays; uint8 and float64 inputs return float64 arrays on this minimum profile.

The fresh artifact-only run uses one global worker and seven rounds per implementation on Linux x86_64, Python 3.11.17, NumPy 1.24.4 and SciPy 1.15.3. The CPU field records only `x86_64`, so the processor model is unknown. This is a first original/current comparison, with no prior-C++ or cumulative cross-host gain claim. The last complete profile qualification remains **203 image APIs**.

Validation: 19 focused normal/scalar XML pairs total 4,250 passing checks, plus 3,696 color checks, all 1,612 existing matching checks, and 1,409 smoke checks for 237 registered APIs. Separate controls passed: four Polygon, four Euler and nine integer-width checks. The private component passed 372 Profile public checks and 136 shared checks across both pinned profiles and normal/scalar modes, with frozen expectations unchanged.

[Release run 40448470579](https://github.com/bipxx/scikit-imagexx/actions/runs/40448470579) measured source [`9ead9b326c8880958b60341921d5ef4091e7b57e`](https://github.com/bipxx/scikit-imagexx/commit/9ead9b326c8880958b60341921d5ef4091e7b57e). Review authenticated all 1,633 committed source blobs, all 63 artifact members, 14 wheels including 13 runtime dependencies, the complete installed package/core/SDK, and identical before/after measurement identities. Only this new 18-case timing ran; 17 earlier timing steps were skipped.

Raw JSON SHA-256: `43991caa3f4bf31d02af51b7f94b632ea98741b3bc99a5a4409cb3b8fc351f6c`. The sole retained artifact is `12709343477.zip` (84,754,347 bytes), SHA-256 `505339a1423bd4f440d4a2d137d628b909d8885f439d183a9852844fc017aaf0`; its archive remains in ignored local evidence. Review SHA-256: `1c2bc1d1d4f5d8051ae405b88404d09bf7c50e87fa5e59459140681a908cccdb`.
