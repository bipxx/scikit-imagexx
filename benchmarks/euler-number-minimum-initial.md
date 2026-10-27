# Euler number: first minimum-profile comparison

All 12 fixed `measure.euler_number` cases were faster than scikit-image 0.26.0, with a geometric mean of **3.175814243864×** and **zero observed integer error**. The six 2D cases had a geometric mean of **2.804614968292×**; the six 3D cases, **3.596142866509×**. All 168 raw latencies (seven paired rounds, two implementations, 12 cases) remain in the [unchanged JSON report](euler-number-minimum-initial.json). There were no measured losses.

| Actual shape | Actual input dtype | Original ms | Current ms | Original/current | Max error |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 × 256 | uint8 | 1.050811167 | 0.644216471 | 1.631146× | 0 |
| 256 × 256 | float32 | 1.201744286 | 0.234228571 | 5.130648× | 0 |
| 256 × 256 | float64 | 1.206300714 | 0.239999950 | 5.026254× | 0 |
| 1024 × 1024 | uint8 | 10.377619000 | 4.856178000 | 2.136993× | 0 |
| 1024 × 1024 | float32 | 11.648560000 | 4.904684500 | 2.374987× | 0 |
| 1024 × 1024 | float64 | 11.881995000 | 5.212231000 | 2.279637× | 0 |
| 32 × 32 × 32 | uint8 | 0.534988316 | 0.167495662 | 3.194043× | 0 |
| 32 × 32 × 32 | float32 | 0.532727227 | 0.166650971 | 3.196664× | 0 |
| 32 × 32 × 32 | float64 | 0.538336429 | 0.164625530 | 3.270066× | 0 |
| 64 × 64 × 64 | uint8 | 3.155831250 | 0.774198462 | 4.076256× | 0 |
| 64 × 64 × 64 | float32 | 3.182176000 | 0.783953692 | 4.059138× | 0 |
| 64 × 64 × 64 | float64 | 3.238235000 | 0.827134923 | 3.915002× | 0 |

Source `86bd994b33e4146695cd9de6a81caf146fe9d2f2` passed genuine [Release run 40434056458](https://github.com/bipxx/scikit-imagexx/actions/runs/40434056458). The producer passed 3,714 focused tests across 16 normal/scalar pairs, 3,696 color regression tests, 1,612 matching tests and the 1,393-node installed smoke cohort (233 registered functions). The separate four Euler intake guards and nine integer-width guards also passed. A fresh measurement host installed the exact archived runtime offline: all 14 wheels, 13 dependencies, package/SDK files and sole core matched before and after measurement. Review authenticated 405 committed source blobs and all 56 outer artifact members by CRC/SHA. The 14 older timing commands were skipped.

The runtime was Linux x86_64, Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Native and reference pools used one worker; AVX2 was available. The raw CPU field identifies only `x86_64`, not a specific model. Timings are warmed, allocation-inclusive public calls using the unchanged seeded-shuffle seven-round measure helper.

Inputs are binary masks prepared outside timing from seed-2026 uint8 gray values `>127`, then cast to the actual dtype. Each geometry uses identical binary pixels across its three dtypes. Public binary conversion, padding, the private checked copy, allocation, configuration counting and weighted arithmetic remain inside timing. Binary conversion/padding, source lookup-table selection and scalar boxing are explicit providers; configuration histograms and bounded exact integer lookup-table arithmetic are owned C++. The `Snum <= 2**53` checker ran before timing. Default connectivity is 2 for 2D and 3 for 3D; result types matched the original (`numpy.int64` and Python `int`, respectively).

This first comparison with the original library on one minimum-profile host does not establish a gain over a prior C++ product or extend the last full supported-profile qualification of **203 image APIs**.

Raw JSON SHA-256: `794756bb3bba19786bc3896ff4603d21d49c9a5b5d0c64812d6d67320bc6f015`. Retained artifact `12703374745` (84,594,783 bytes): `cde747de0a68956b68aaa5a4a6c762e749df2f327fe40ea76f2220f7cf10c78e`. The complete archive remains outside Git.

Review: `build/phase3-euler-number-first-release-v1/final-review.json`, SHA-256 `b92a3f60229898494835e3cd05ae38cc2dc5497147d062c188bc7c7f46bad5b9`. Product wheel: `2e7c205618e5567e2ecec9dce15058af853122bfe5d2f42fef5ef797038037fe`; sole core: `7faad46e31c1dbfcfd5ce0d3c5a79be7f2b3aa93ec66d0a76b56ef4df487d0a7`. The report's canonical-LF harness SHA-256 is `cdf6690b52cb8e4d6c2a12ac4dc85a68f8dd18fe19c80ea290d22a0e50860a13`.
