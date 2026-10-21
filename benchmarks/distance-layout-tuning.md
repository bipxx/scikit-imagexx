# CIE76 nonuniform-layout optimization

Historical fixed experiment: a 512x512 float32 Lab pair with stepped/reversed pixel layouts, one worker, and allocation-inclusive Python calls. Both snapshots use the same seed and seven warmed samples, with parity checked against original scikit-image and fallback disabled.

Baseline `83402fc`; candidate `f3b3ff9`. [Raw samples](distance-layout-tuning.json) preserve the complete 123-case one-worker reports, including the other weighted distances and routes. These snapshots are separate from the [current-source aggregate](STATUS.md).

Native median: **1.6552 ms to 0.7387 ms**, a **2.24x** improvement. Reference medians were 1.8007 and 1.8025 ms. The candidate measured 2.44x faster than the original on this workload.

The candidate sends row offsets and signed element strides to C++ instead of copying nonuniform inputs into contiguous images. Threads partition the complete output once, and SIMD processes each valid row segment with scalar tails. Unaligned storage still requires copying.

| Implementation | Original ms | C++ ms | Original / C++ |
|---|---:|---:|---:|
| Before | 1.8007 | 1.6552 | 1.09x |
| After | 1.8025 | 0.7387 | 2.44x |

All 120 candidate one-worker processing cases measured faster than the reference in this run. Three RGB identity cases are labeled separately. Measurements used ordinary system load without affinity or frequency locking; raw samples retain the timing variation.
