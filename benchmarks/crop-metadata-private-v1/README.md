# Private C++ crop comparison

Candidate `f02056536f879df477cf4c2aded5979379fcb339` versus the prior `d7e7193e` binder, sharing the unchanged `12d130a2` pixel leaf, SDK and Python frontend. Both callables are private C++ components on local CPython 3.11.14 / NumPy 1.24.4, one worker. This gives no original scikit-image speed or shipping credit.

One frozen pass: eight unchanged public-call fixtures, seven samples per callable/case, minimum 100ms samples and 250ms warmup per callable/case; 112 samples and 16 warmup groups. Root reconstructed every median from the [exact journal](batches.jsonl), checked against the [exact report](report.json). Builds/tests were held quiet; no unchanged repeat. Returned array allocation is included.

The unchanged generic sampler emits literal `scikit-image/current` sample labels and `reference_ms/cpp_ms` field names. **For these files, slot 0 is prior C++ and slot 1 is candidate C++.** Their actual identities are recorded in the report's `slot_meaning`; zero original calls occurred. Raw bytes are preserved.

| Case | Prior C++ (us) | Candidate C++ (us) | Prior / candidate |
|---|---:|---:|---:|
| `view-C` | 1.457 | 1.346 | 1.083x |
| `view-F` | 1.435 | 1.365 | 1.051x |
| `view-reverse` | 1.428 | 1.323 | 1.079x |
| `view-3D` | 1.603 | 1.461 | 1.097x |
| `copy-C` | 39.994 | 39.623 | 1.009x |
| `copy-F` | 15.512 | 16.100 | 0.964x |
| `copy-reverse-K` | 21.189 | 21.117 | 1.003x |
| `copy-object-provider` | 2.429 | 2.349 | 1.034x |

Seven faster medians and one slower median: F-order copy latency increased 3.8% (0.964x speed ratio). Candidate remains unaccepted. View speed ratios are 1.051–1.097x; the copy slowdown's cause is unproved. A successor will restrict the optimization to view calls and require fresh parity and a new source-bound comparison. Keep this outcome and every raw sample.

This is local triage, with private links and local microversion/provider/CPU scope; CPU metadata was not recorded during this experiment. It establishes neither an original-public FP oracle nor a genuine installed product/SDK or four-profile release. Hosted original-library comparisons remain mandatory.
