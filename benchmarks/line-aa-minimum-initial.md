# Anti-aliased line: initial one-thread comparison

All nine frozen source-safe workloads beat scikit-image 0.26.0: **9 wins, 0 losses**, geometric mean **28.185764431×**. The [JSON report](line-aa-minimum-initial.json) retains all 126 samples and 407 journal events in one compact exact-byte archive.

The measurement includes the public call, endpoint conversion, output allocation and native count/fill. It uses one worker and one thread in each reference pool, seven samples per implementation/case, accepted batches of at least 100 ms, and at least 250 ms warmup across each of 18 implementation/case groups. Before timing, the pinned harness requires exact coordinates/order/weights and three independent owning outputs (`intp`, `intp`, `float64`), with an actual C-return/native-admission witness.

| Case | Original ms | C++ ms | Original/current |
|---|---:|---:|---:|
| line-aa/tiny | 0.004277067 | 0.001737487 | 2.461639624× |
| line-aa/horizontal | 0.247681501 | 0.009072637 | 27.299834423× |
| line-aa/vertical | 0.230498261 | 0.009708932 | 23.740846500× |
| line-aa/diagonal | 0.370717936 | 0.009756532 | 37.996897695× |
| line-aa/shallow | 0.557646551 | 0.013144838 | 42.423235414× |
| line-aa/steep | 0.561183810 | 0.013523967 | 41.495501438× |
| line-aa/reverse | 0.556436846 | 0.013070979 | 42.570402673× |
| line-aa/long | 5.496031714 | 0.106076979 | 51.811729315× |
| line-aa/translated | 0.623171578 | 0.013064980 | 47.697858880× |

Measured source: `28a6790b2a393ab4a4ffcc062e4962693fb4e12b`. [Run 41889026308](https://github.com/bipxx/scikit-imagexx/actions/runs/41889026308) succeeded on an AMD EPYC 7763 host with four logical CPUs available and actual AVX2 availability. Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0 were frozen. This is one Linux minimum-profile host/cohort.

The genuine wheel contains 182 files, including the 86-file SDK. Actual line-AA XML records 188 normal and 186 scalar checks, with zero failures, errors or skips. A client compiled against the genuine relocated SDK passes 121 controls per normal/scalar mode with global worker settings 1/4 and restoration to 1; this validates one drawing-family client. Full four-profile qualification remains **252 image APIs** at source `6fae25533efd28bf48c4d4ad33b5e3e8ce396c84`, recorded in the [root admission](../coverage/evidence/public252-whole-product-admission.json).

Native admission uses `operator.index` once per endpoint followed by Py_ssize bounds and source-audited int32 geometry, neighbor headroom and squared-delta safety. Unsafe converted geometry raises `NotImplementedError`; it earns no parity credit. The ordered recurrence is serial. AVX2 availability does not establish a custom AVX2 implementation or parallel line kernel. Strict floating-point flags are verified from source/build recipes; the artifact contains no raw compiler command log.

This initial port comparison makes no prior-C++ optimization KEEP decision or library-wide speed claim. Historical raw review notes are preserved unchanged, including older baseline references; the current qualified baseline is 252.

The JSON archive stores the original report, journal, Markdown, manifest, both line-AA XML files and three independent reviews exactly once. Retained artifact: `13353601382`, SHA256 `5f98aa4feaab686000215dd748facd23fca851c13b956cbe12280b8bcfa8cf3f`.
