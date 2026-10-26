# Normalized moments C/F layout: same-host comparison

[ABBA run 40374541146](https://github.com/bipxx/scikit-imagexx/actions/runs/40374541146) compares baseline `aecf5218563b09981bf329b6975e4e43cbf4334a` with candidate `7733003ed797c7796f355b23f062d01c8ca9c603` on one **Intel Xeon Platinum 8573C** host, using one worker. The fixed 13-case geometric mean improves **3.346642894x directly / 3.338426256x after reference normalization**. The four F-layout targets improve **50.841838826x / 51.004018874x**; both candidate passes beat both baseline passes for all four targets on both metrics.

All 13 cases beat scikit-image in both candidate passes. The nine C-layout controls are approximately flat: **0.998731020x direct / 0.993783401x normalized**. All **four direct and six normalized losses** remain below. Three C controls are slower in both candidate passes than both baselines: synthetic uint8 Hu, synthetic float32 normalization and real u8-auto normalization (approximately +0.046, +0.034 and +0.030 us respectively). This is a bounded public-call result on one minimum-Linux host; it establishes neither a whole-image pipeline speedup nor full supported-profile qualification.

The [exact compact artifact](moments-normalized-layouts-same-host.zip) retains four complete reports, identity checks, CPU metadata and all **728 raw latency samples** (364 per implementation). The prior [five-case C-layout campaign](moments-normalized-batch-same-host.md) remains unchanged.

## Method and all 13 cases

The common external harness uses five existing synthetic controls and four saved original central-moment tensors, each copied in C and F order. Every input is an owning 4x4 array; dtype names describe the actual tensor. Preparation stays outside timing; public eligibility checks, conversion, allocation and wrapping remain inside. Output dtype, strides and storage flags match the original before timing. Fixture SHA-256 is `dd460b4bd3d5c6cdc6e2f121af00bf592f26f65d7ad8fb19841215c5ff2f4746`.

Exact archived Release wheels and 13 identical dependency wheels are installed offline. Product source/package/SDK identities are verified before and after each fresh process. No builds or pytest run on the measurement host. Pass order is B1, C2, C3, B4; each latency is the median of seven warmed paired samples. Balanced times are geometric means of the two role medians. Direct gain is baseline/current divided by candidate/current; normalized gain is the candidate original/current ratio divided by the baseline ratio. Positive added time is a regression.

| Case | Dtype | Layout | Baseline us | Candidate us | Direct gain | Normalized gain | Added us | Original / candidate time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `synthetic/measure.moments_hu/uint8/C` | uint8 | C | 1.676311 | 1.722151 | 0.973382165x | 0.968396964x | +0.045840 | 1.174666845x |
| `synthetic/measure.moments_normalized/float32/C` | float32 | C | 1.350072 | 1.384467 | 0.975156515x | 0.958337874x | +0.034395 | 24.058114259x |
| `synthetic/measure.moments_hu/float32/C` | float32 | C | 1.267597 | 1.258773 | 1.007010615x | 0.977481637x | -0.008825 | 1.334447166x |
| `synthetic/measure.moments_normalized/float64/C` | float64 | C | 1.409216 | 1.397064 | 1.008698571x | 1.026019536x | -0.012152 | 12.581911794x |
| `synthetic/measure.moments_hu/float64/C` | float64 | C | 1.359062 | 1.295928 | 1.048717206x | 1.039089356x | -0.063134 | 1.359039194x |
| `real/u8-auto/C` | float64 | C | 1.367952 | 1.397943 | 0.978546473x | 0.992967815x | +0.029991 | 12.669542026x |
| `real/u8-auto/F` | float64 | F | 62.775661 | 1.438264 | 43.646821750x | 43.540164147x | -61.337397 | 12.748374950x |
| `real/f32-auto/C` | float32 | C | 1.367307 | 1.358677 | 1.006351954x | 0.979545984x | -0.008630 | 24.063739042x |
| `real/f32-auto/F` | float32 | F | 82.746425 | 1.380478 | 59.940433821x | 60.508581562x | -81.365947 | 24.520877574x |
| `real/central-layout-step/C` | float64 | C | 1.384764 | 1.372465 | 1.008961657x | 1.011249999x | -0.012300 | 13.049063774x |
| `real/central-layout-step/F` | float64 | F | 63.484737 | 1.484446 | 42.766625938x | 42.289259771x | -62.000291 | 12.180932882x |
| `real/supplied-fractional-anisotropic/C` | float32 | C | 1.334183 | 1.355802 | 0.984054794x | 0.993843279x | +0.021619 | 24.200245110x |
| `real/supplied-fractional-anisotropic/F` | float32 | F | 83.187117 | 1.392992 | 59.718309227x | 60.740795221x | -81.794125 | 24.718153722x |

## Group results

| Group | Cases | Direct gain | Normalized gain |
|---|---:|---:|---:|
| all13 | 13 | 3.346642894x | 3.338426256x |
| F_targets4 | 4 | 50.841838826x | 51.004018874x |
| C_controls9 | 9 | 0.998731020x | 0.993783401x |
| legacy5 | 5 | 1.002218261x | 0.993339810x |
| Hu3 | 3 | 1.009234161x | 0.994500428x |
| normalization10 | 10 | 4.795066145x | 4.800901894x |

## Validation and limits

Candidate Release [40372577172](https://github.com/bipxx/scikit-imagexx/actions/runs/40372577172) passes all seven installed normal/scalar contract groups, including the new 22-node layout cohort, plus 1,357 installed smoke checks. Component validation separately passes 4,180 checks across both runtime profiles and dispatch modes. The C/F change modifies only admission and output strides; arithmetic, SIMD kernel, numeric bounds and frozen tolerances remain unchanged. Other layouts/options retain the scalar path. NumPy preparation/scalar providers and Hu wrapping remain explicit.

The exact published ZIP is 32,250 bytes, SHA-256 `0dfcfee1eec220a648a560242efefc34ccbe978db1728b1be24fa4799e4609ac` (artifact `12674730768`). Source coverage stays 221 native-partial image APIs plus three metadata functions; accepted combined supported-profile coverage stays 203 image APIs.
