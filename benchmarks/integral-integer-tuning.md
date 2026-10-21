# Integer integral-image prefix experiment

The direct-read path removes the preparation copy for one-threaded 1D int64/uint64 input and integer output. Unsigned modular arithmetic preserves signed wraparound. The four-lane SIMD scan remains; floating addition order is unchanged.

| Observation | Current C++ geometric mean, ms | Preserved C++ geometric mean, ms | Baseline/current | Faster than scikit-image |
|---|---:|---:|---:|---:|
| Paired run | 0.163626 | 0.164358 | 1.0045x | 144/148 |
| Independent confirmation | 0.174833 | 0.179625 | 1.0274x | 140/148 |

Kept implementation: `084e760`. The aggregate covers different shapes/dtypes and is nearly tied in the first run; confirmation lowers latency by 2.67%. [All148 workloads and raw samples](integral-integer-tuning.json) remain available.

| Long 1D input | First run: old/current C++ | Confirmation: old/current C++ | First run: scikit-image/current | Confirmation: scikit-image/current |
|---|---:|---:|---:|---:|
| uint64 | 1.727x | 3.390x | 1.170x | 1.754x |
| int64 | 1.722x | 2.296x | 1.217x | 1.122x |

Confirmation retains slower uint8/int16/uint64/int64 single columns, float16 lines/rows/columns and float64 single columns. Near-parity and individual timings vary under ordinary system load. These are machine-specific observations, not universal speed guarantees.

An earlier sixteen-element unroll did not improve historical absolute latency; the reference slowed too. That variant was discarded. Its [complete raw report](integral-stream16-discard.json) and [reconstructable patch](integral-stream16-discard.patch) are retained. The patch applied to `e2b5c5a` exactly reproduces the measured source fingerprint.

To reduce machine-state drift, this experiment loads the preserved `e2b5c5a` C++ package beside the current package and original scikit-image in one process. The baseline manifest records the source revision/fingerprint and extension binary hash, and the preservation helper compares wheel Python sources with the clean checkout. Randomized measurements include all three backends for each unchanged workload.

Reproduction: create a detached checkout at `e2b5c5a`, build its wheel with `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir <wheel-directory>`, then run `python tools/preserve_baseline.py --wheel <wheel-file> --source-root <clean-checkout> --output build/paired-baseline`. Run `python benchmarks/integral.py --threads 1 --repeat 7 --baseline-dir build/paired-baseline`. Finish all local builds/tests before timing.

Validation: 5,861 contract cases passed in each SIMD mode on Windows and Linux, including 16-element boundary cases, signed/unsigned overflow and true large 1D arrays with one/four workers. Native C++ and Linux AddressSanitizer/UndefinedBehaviorSanitizer checks passed. AVX2/OpenMP-disabled C++ validation also passed for the corresponding portable code. The broader verification aggregate is refreshed separately after the final implementation.
