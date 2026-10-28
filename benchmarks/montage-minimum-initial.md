# Montage: initial one-thread comparison

The nine frozen workloads produced **6 wins and 3 losses** against scikit-image 0.26.0, with geometric mean **1.188252972075×**. All 126 raw samples, complete report, summary and independent review are retained unchanged in [the JSON publication](montage-minimum-initial.json).

CPU: AMD EPYC 7763 64-Core Processor (4 logical CPUs reported). Runtime: Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3; one worker, seven rounds, 19 ms batch target. [Timing run 40742758297](https://github.com/bipxx/scikit-imagexx/actions/runs/40742758297) used the authenticated offline product on a fresh host without builds or tests. Original/before/after runtime, core and SDK identities matched.

| Case | Batch shape / dtype / layout | Original ms | Current ms | Original/current | Added ms |
|---|---|---:|---:|---:|---:|
| gray-small-u8 | 4×32×32 / uint8 / C | 0.012296173 | 0.007009821 | 1.754135144× | -0.005286352 |
| gray-incomplete-u8 | 7×128×128 / uint8 / C | 0.026288287 | 0.021850580 | 1.203093309× | -0.004437707 |
| gray-dense-f32 | 16×128×128 / float32 / C | 0.100231569 | 0.074053684 | 1.353498749× | -0.026177885 |
| gray-large-u16 | 16×240×240 / uint16 / C | 0.155749807 | 0.125520271 | 1.240833901× | -0.030229536 |
| gray-mean-f64 | 7×192×192 / float64 / C | 0.248553063 | 0.266094070 | 0.934079676× | +0.017541007 |
| rgb-padded-u8 | 9×128×128×3 / uint8 / C | 0.192629161 | 0.049081604 | 3.924671237× | -0.143547556 |
| rgb-first-f32 | 9×3×96×96 / float32 / channels-first | 0.168343295 | 0.091472585 | 1.840368851× | -0.076870711 |
| gray-strided-reverse-f32 | 8×128×128 / float32 / strided-reverse | 0.117468152 | 0.593540800 | 0.197910830× | +0.476072648 |
| gray-fortran-f64 | 12×128×128 / float64 / F | 0.960117000 | 0.962142692 | 0.997894603× | +0.002025692 |

Added ms is current minus original. The strided-reverse float32 case is about 5.05 times slower; mean-fill float64 and Fortran float64 also lose. The Fortran difference is small in this single cohort. Every loss remains in the geometric mean and raw report. These initial results establish no prior-C++ improvement or whole-library superiority.

C++ owns prepared POD fill and tile-row copying. Axis/asarray/grid/fill conversion and mean remain explicit providers; rescale is outside these timed workloads. Timings include the complete public call, fresh allocation, preparation, copying, temporary allocation and output destruction. Exact output dtype, layout, ownership and words passed the independent fill/placement oracle; every case reached the actual native builtin and returned native acceptance.

Measured source: `ec3dba8e5eb73a0b02dffd70c501abe2adba923c`, with 243 image APIs, 246 registered functions, 170 wheel members and 80 SDK members. The qualified baseline remains source `4e657ee64fb71e4d707d32e9539fdb4d7cc528af`, with 241 image APIs and 71 SDK members. This cohort does not extend whole-product qualification.

A separate client compiled solely against the actual genuine SDK80 and passed normal/scalar execution, each exercising workers 1 and 4: **one family, two checks**. Its receipt SHA is `040c77b4f39d1664bd07293666f2705ebd54de8d227c74edf91065ed3ad4122d`. This is distinct from the earlier assembled-SDK proof. The producer executed zero SDK clients; the source-declared 26-family/52-check campaign remains unexecuted.

The authenticated harness asserts NumPy error-policy, handler and buffer invariance before and after every timing. Initial values are not separately exported; the JSON retains this evidence boundary.

Producer run/artifact: `40740008474` / `12840295821`, ZIP SHA `6025439512b27a0a594ee6df1fc27aa7a609ada5c50ab439398ea0fa4e09f951`. Result artifact: `12840806523`, ZIP SHA `7020d06c48a3712cf1bdaf7e75d75ec4b624fc853dee70d1c9adc2b409d42b04`. Independent raw review SHA: `452d0267aac695f8c966dbf387e531858171808020ecfac8be749b780062876f`. The publication preserves original numeric text and terminal newlines; the result archive retains CPU details, installation and identity logs.
