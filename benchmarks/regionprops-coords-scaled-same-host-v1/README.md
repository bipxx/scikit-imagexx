# coords_scaled: fixed one-worker comparison

KEEP: [independent raw reconstruction](../../coverage/evidence/regionprops-coords-scaled-performance-independent-review-v1.json) confirms all32 inequalities across12 paired checks. Candidate wins all12 original comparisons (six cases in each candidate pass) and has no prior-C++ regression in these cohorts. This scoped performance admission does not qualify a release. All336 samples and1096 batch events are retained here; full semantic payload reports remain in authenticated artifact13437762797. No timing was repeated for publication.

| Case | Original/C++ pass2 | Original/C++ pass3 |
|---|---:|---:|
| tiny-c-off | 1.589027 | 1.609869 |
| tiny-f-warm | 1.959461 | 1.968795 |
| medium2d-f-warm | 1.272854 | 1.243244 |
| medium3d-f-warm | 1.203196 | 1.189948 |
| sparse2d-ctor-read | 1.099593 | 1.089247 |
| integer3d-partial | 1.571060 | 1.575776 |

Baseline: `cf0abd3f8dba9acbc988a59a53ae32cbcccfab4b`; candidate: `4ac77e4bc6f0f7bb26ba509c7f13869a4349a59e`. [Run42074843238](https://github.com/bipxx/scikit-imagexx/actions/runs/42074843238) uses controller `cf1222b2701968fff175e6098fb58bef08a33a86`. Four ABBA passes,7 accepted100ms samples per arm/case and at least250ms warmup per arm group; native and numerical provider thread budgets1. Prior-C++ direct and original-normalized ratios must be<=1.05 in both pairs; two medium targets additionally require direct prior ratios<=.90 and candidate/original<=.90. All32 inequalities pass in the root reconstruction.

Five float64 cohorts exercise the owned four-phase path. Integer-spacing multiply/add remain explicit NumPy providers. Sparse constructor-plus-read timing does not establish a pure-constructor regression fix. No whole-class, whole-library, all-SIMD, scaling or four-profile release claim follows. Earlier failedv3 evidence and the accepted baseline's original-library losses remain unchanged.

Retained outer archive SHA256: `5c4c8aabd5be5faabcda130da08c0d2595e2c94faf8667eda57e99f0ae9b3115` (171052651 bytes). Its two nested prepared packages were not read/extracted/rehashed for this publication. Bounded reports retain metadata, source report hashes, exits, medians and samples; batch journals retain calibration and attempts. Full original semantic reports remain in the archive.

Baseline reports retain nine slower-than-original case/pass medians; candidate reports have none in this fixed six-case experiment. No unchanged timing rerun or threshold waiver occurred. Closeout: existing sampler, genuine artifacts and retained raw data reused; independent reader recomputed gates once, then checked the bounded publication. Next gate is four-profile run42099127401, followed by exact-source admission and delivery.
