# Circle perimeter: initial one-thread comparison

All eight frozen workloads beat scikit-image0.26.0: **8 wins, 0 losses**, geometric mean **21.527045058×**. All112 raw samples, calibration batches and native-output witnesses are retained in [the JSON report](circle-perimeter-minimum-initial.json).

The public calls include conversion, output allocation, native count/fill and output destruction. Each accepted batch lasts at least100ms, with at least250ms warmup per implementation/case; workers and all four reference pools are1. Exact output words, order, dtype, layout, ownership and actual native admission are checked before timing.

| Case | Original ms | C++ ms | Original/current |
|---|---:|---:|---:|
| bresenham/tiny | 0.002189602 | 0.001141444 | 1.918273663× |
| bresenham/cache | 0.055689353 | 0.002310565 | 24.102054959× |
| bresenham/long | 12.425245200 | 0.140852564 | 88.214547698× |
| bresenham/clipped | 0.269429252 | 0.005650805 | 47.679797760× |
| andres/tiny | 0.002564563 | 0.001157378 | 2.215838464× |
| andres/cache | 0.062869001 | 0.002499379 | 25.153852038× |
| andres/long | 14.031663000 | 0.156556130 | 89.627043184× |
| andres/clipped | 0.298631522 | 0.006290443 | 47.473848301× |

Measured source: `892c5aa70c5a9ca65f92b4060c407caf61b3af5c`, Python3.11.17, NumPy1.24.4, SciPy1.15.3. [Run41876409255](https://github.com/bipxx/scikit-imagexx/actions/runs/41876409255) succeeded. The CPU model was not recorded; this is one host/cohort, not a hardware-general or whole-library claim.

Genuine preparation contains180 product/85 SDK members and passes147 new public checks per normal/scalar mode. A client compiled solely against the genuine relocated SDK85 passes1,228 controls per normal/scalar process with workers1/4 and default restoration. These scoped results do not replace complete supported-profile qualification, which remains247 APIs.

C++ owns the ordered circle recurrence and clipping for admitted integer geometry. The recurrence remains serial. Floating-shape comparisons and unassessed method/shape protocols remain explicit providers; no custom SIMD or parallel-circle claim is made.

The JSON embeds one compact archive with original report/journal/manifest, both circle XML files and all three reviews; it records their exact hashes. The complete retained artifact is13348585270, SHA256 `8d9cb26c4a79efc749e5a2f9ba1adb3f2fde62f7dd41be79a278e04475f27fcd`.
