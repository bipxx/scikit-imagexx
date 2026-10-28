# Rolling kernel constructors: initial one-thread comparison

The twelve frozen constructor workloads produced **12 wins and no losses** against scikit-image 0.26.0, with geometric mean **11.956836613518×**. All 168 raw samples, parameters, complete report, summary and independent review are retained unchanged in [the JSON publication](rolling-ball-kernels-minimum-initial.json).

CPU: AMD EPYC 7763 64-Core Processor (4 logical CPUs reported). Runtime: Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3; one worker, seven rounds, 19 ms batch target. [Timing run 40760482122](https://github.com/bipxx/scikit-imagexx/actions/runs/40760482122) used the authenticated offline product on a fresh host without builds or tests. Original/before/after runtime, core and SDK identities matched.

| Case | Constructor parameters | Output shape | Original ms | Current ms | Original/current | Added ms |
|---|---|---|---:|---:|---:|---:|
| ball-1d-fractional | radius=127.5, ndim=1 | 257 | 0.053486883 | 0.009431057 | 5.671356259× | -0.044055826 |
| ball-2d | radius=192, ndim=2 | 385×385 | 3.983118250 | 0.228600193 | 17.423949639× | -3.754518057 |
| ball-3d | radius=31, ndim=3 | 63×63×63 | 12.017076000 | 0.467236393 | 25.719477728× | -11.549839607 |
| ball-4d | radius=8, ndim=4 | 17×17×17×17 | 2.913498400 | 0.247776064 | 11.758595059× | -2.665722336 |
| ball-5d-small | radius=3, ndim=5 | 7×7×7×7×7 | 0.750284842 | 0.109967444 | 6.822790562× | -0.640317398 |
| ball-8d-small | radius=1, ndim=8 | 3×3×3×3×3×3×3×3 | 0.458203828 | 0.090794444 | 5.046606435× | -0.367409383 |
| ellipsoid-2d-odd | shape=(255, 127), intensity=64.0 | 255×127 | 1.580794100 | 0.066418225 | 23.800607263× | -1.514375875 |
| ellipsoid-2d-even | shape=(256, 128), intensity=32.0 | 257×129 | 1.598025200 | 0.066357582 | 24.082028921× | -1.531667618 |
| ellipsoid-3d-odd | shape=(63, 31, 15), intensity=12.5 | 63×31×15 | 1.565796200 | 0.109921811 | 14.244636112× | -1.455874389 |
| ellipsoid-3d-even | shape=(64, 32, 16), intensity=64.0 | 65×33×17 | 1.907224625 | 0.112884918 | 16.895300672× | -1.794339707 |
| ellipsoid-4d-odd | shape=(15, 11, 9, 5), intensity=10.0 | 15×11×9×5 | 0.533357321 | 0.082674200 | 6.451315180× | -0.450683121 |
| ellipsoid-4d-even | shape=(16, 12, 10, 6), intensity=2.5 | 17×13×11×7 | 1.119036857 | 0.120007421 | 9.324730482× | -0.999029436 |

Added ms is current minus original. Every case contributes to the geometric mean. These initial results describe this one-thread, twelve-case original/current cohort; they establish no prior-C++ improvement or whole-library superiority.

Literal ceil/arange, shape asarray/floor/clip, scalar radius**2 promotion, dtype/axis preparation and fresh output allocation remain providers. C++ owns prepared coordinate arithmetic, exact reduction, square roots and masks. Native credit requires actual ball_kernel_apply/ellipsoid_kernel_apply C-return and native_result=True; refused/nonfinite/protocol routes remain explicit providers. Entire allocation-inclusive public constructor call, including preparation, coordinates/reductions, temporaries, fresh output and output destruction. No supplied pixels or fabricated dtype cohorts; oracle and parity/admission guards remain untimed.

The independent coordinate oracle preserves source arithmetic and reduction order. Every output passed exact float64 words, positive-infinity masks, shape, strides, ownership and writeability checks. Each case reached the actual native builtin and returned native acceptance. Timings include the whole public constructor call, fresh output, preparation, temporary allocation and output destruction; parity checks and the oracle remain untimed.

Measured source: `390dff892ad13c1f9b8cee0e76dce9a016f14c01`, with 245 image APIs, 248 registered functions, 172 wheel members and 81 SDK members. The qualified whole-product baseline remains source `4e657ee64fb71e4d707d32e9539fdb4d7cc528af`, with 241 image APIs and 71 SDK members. This cohort does not extend whole-product qualification.

A separate client linked solely against the actual genuine SDK81 and passed normal/scalar execution, each exercising workers 1 and 4: **one family, two checks**. Its receipt SHA is `fc43987e4e4990e0a71d5ef3d8de168ca0a44ff68a6ddbecb6e82bb1205e8fa5`. The producer executed zero SDK clients; the source-declared 26-family/52-check campaign remains unexecuted.

NumPy error policy (`divide=warn`, `over=warn`, `under=ignore`, `invalid=warn`), handler (`NoneType`) and buffer size (8192) were exported and remained unchanged across prechecks and timings.

Producer run/artifact: `40758210302` / `12846458475`, ZIP SHA `d064c77793dbadd81c8de20e108f49467d19ffa59910fbccdd8609061cd42b89`. Controller: `4231ef483dcfe259a911b41ad50367a7b815a11e`. Result artifact: `12847708051`, ZIP SHA `94d5b98aefc69115aa6de9bd52fb6ab1ad51af5fe395fae4d2218f31f0ebfa38`. Independent raw review SHA: `14c72b5b6bf69bcfcb24cda28e2e9fda8380074d9d37876e151774142e98c83c`. The publication preserves original numeric text and terminal newlines; the result archive retains CPU details, offline installation, producer authentication and identity logs.
