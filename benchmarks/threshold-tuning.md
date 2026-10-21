# Yen logarithm tuning experiment

The fixed metric is geometric-mean C++ latency across nine histogram-only Yen
workloads with one worker, lower being better. The unchanged benchmark command
uses seven samples, checks exact reference thresholds and warnings, and retains
all 76 serial workloads as guards. Results are historical snapshots excluded
from the current-source aggregate.

| Snapshot | Native geometric mean | Original geometric mean |
|---|---:|---:|
| Baseline `7d5b9d0` | 19.2105 µs | 13.6511 µs |
| First candidate | 18.3661 µs | 18.0396 µs |
| Unchanged candidate confirmation `86393d2` | 13.3577 µs | 13.8510 µs |

The confirmation measured **1.438×** lower native latency versus the
baseline. The first candidate run showed substantial reference timing drift;
the unchanged command/source was repeated to check that observation. The JSON
retains all samples from all three runs. No CPU affinity or frequency locking
was used, and small timing differences remain subject to system noise.

The kept change batches the compatibility logarithm into eight AVX2/FMA lanes.
A separate CPU FMA check guards that translation unit; scalar std::fma remains
available on other CPUs and when SIMD is disabled. A profile of 20,000 supplied
histogram calls then identified metadata validation/reductions as the larger
remaining cost, so that is the next optimization target.

A subsequent native metadata-validation scan was **discarded**. Its two raw
native geometric means were 18.2192 and 19.8738 µs, above the kept 13.3577 µs.
Reference means also drifted to 35.1886 and 30.2315 µs, so the runs do not isolate
the cause. The fixed raw-latency criterion was not improved; the source patch
and both complete observations are preserved in the JSON for a future paired
experiment. The discarded scan is absent from the shipped implementation.
