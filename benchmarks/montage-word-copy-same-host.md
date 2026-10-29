# Montage word-copy bundle: accepted comparison

**KEEP.** All ten fixed gates, variance-separation rules and the deadline pass. All nine workloads improve over the earlier C++ baseline and beat scikit-image in every one of the four candidate passes. The complete 1,008 raw samples, 72 journals, eight processes, reports and reviews are retained in [the JSON publication](montage-word-copy-same-host.json).

[Run 40812393430](https://github.com/bipxx/scikit-imagexx/actions/runs/40812393430) used AMD EPYC 9V74 80-Core Processor, CPU 0, one worker/reference pool and eight independent A B B A B A A B processes. Every accepted batch lasted at least 100 ms (observed 106.334–124.319 ms); all 144 function warmup groups accumulated at least 250 ms. No builds or tests ran on the timing host.

| Workload | Baseline C++ ms | Candidate C++ ms | Direct gain | Normalized gain | Original/candidate | Added ms |
|---|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.005017802 | 0.004715043 | 1.064211429× | 1.068745094× | 1.715629323× | -0.000302760 |
| gray-incomplete-u8 | 0.014857759 | 0.011072356 | 1.341878690× | 1.322564723× | 1.519628269× | -0.003785403 |
| gray-dense-f32 | 0.047185413 | 0.036047812 | 1.308967466× | 1.339091278× | 2.075304485× | -0.011137601 |
| gray-large-u16 | 0.080294800 | 0.055563838 | 1.445090962× | 1.441384101× | 2.049513630× | -0.024730962 |
| gray-mean-f64 | 0.172070900 | 0.136588589 | 1.259775070× | 1.263163300× | 1.284265836× | -0.035482310 |
| rgb-padded-u8 | 0.029570043 | 0.024371728 | 1.213292809× | 1.215064054× | 6.463922031× | -0.005198314 |
| rgb-first-f32 | 0.061718820 | 0.043581546 | 1.416168670× | 1.420580471× | 2.839611879× | -0.018137274 |
| gray-strided-reverse-f32 | 0.493905038 | 0.031504309 | 15.677380398× | 15.651625930× | 2.523672567× | -0.462400729 |
| gray-fortran-f64 | 0.755569479 | 0.610506524 | 1.237610819× | 1.208127932× | 1.175200224× | -0.145062955 |

Direct geometric mean is **1.691650780420×** and normalized geometric mean is **1.690042171479×**. The strided-float32 target improves **15.677380398× direct** and **15.651625930× normalized**, and runs **2.523672567× faster than scikit-image** in this cohort. Geometric mean speedup over scikit-image across all nine cases is 2.094352682236×. There are no balanced direct/normalized losses or candidate original losses in this run.

The measured change is the **whole b5 row-owned/byte-fill/stride2-SIMD bundle plus the eight-byte strided-copy specialization**. Fortran codegen removes 196,608 dynamic-size per-pixel copy calls, but this experiment does not isolate that four-line change or explain every historical regression. All earlier DISCARD and UNRESOLVED decisions, including their original/control losses, remain unchanged.

| Control | Worst direct gain | Worst normalized gain |
|---|---:|---:|
| gray-small-u8 | 1.013641885× | 1.020306370× |
| gray-incomplete-u8 | 1.271591502× | 1.254838354× |
| gray-dense-f32 | 1.149000745× | 1.236393087× |
| gray-large-u16 | 1.425072448× | 1.425818270× |
| gray-mean-f64 | 1.244108787× | 1.236491988× |
| rgb-padded-u8 | 1.151211351× | 1.151637427× |
| rgb-first-f32 | 1.330447649× | 1.336177811× |
| gray-fortran-f64 | 1.187313632× | 1.135228385× |

Same-product null ratios outside the fixed 3% interval invoke the corresponding worst-pair floor; all required floors pass. This establishes separation under the sealed rules, not an absence of variation. Every direct/reference/normalized null ratio and all per-pass gains are preserved below and in the raw evidence.

| Control | Same-product pair | CPP earlier/later | Original earlier/later | Normalized later/earlier |
|---|---|---:|---:|---:|
| gray-small-u8 | 2/3 | 0.981010533× | 0.986254697× | 0.994682748× |
| gray-small-u8 | 6/7 | 1.017706129× | 1.005389005× | 1.012251102× |
| gray-incomplete-u8 | 2/3 | 0.948954636× | 0.990155745× | 0.958389265× |
| gray-incomplete-u8 | 6/7 | 0.967468210× | 1.012293999× | 0.955718607× |
| gray-dense-f32 | 2/3 | 0.944394552× | 0.962192983× | 0.981502224× |
| gray-dense-f32 | 6/7 | 1.026787817× | 1.025734608× | 1.001026785× |
| gray-large-u16 | 2/3 | 1.006068555× | 1.005572413× | 1.000493392× |
| gray-large-u16 | 6/7 | 1.008410616× | 1.012524431× | 0.995937071× |
| gray-mean-f64 | 2/3 | 0.977997865× | 1.002253858× | 0.975798553× |
| gray-mean-f64 | 6/7 | 0.999501421× | 1.011193535× | 0.988437314× |
| rgb-padded-u8 | 2/3 | 0.961953382× | 1.001159443× | 0.960839343× |
| rgb-padded-u8 | 6/7 | 0.983122380× | 1.011157249× | 0.972274472× |
| rgb-first-f32 | 2/3 | 0.990240287× | 0.993264908× | 0.996954869× |
| rgb-first-f32 | 6/7 | 1.008909591× | 1.009877094× | 0.999041959× |
| gray-fortran-f64 | 2/3 | 1.005747518× | 1.033277182× | 0.973356942× |
| gray-fortran-f64 | 6/7 | 0.928411160× | 1.010828724× | 0.918465352× |

Both measured products use the same eager Python frontend, 13 exact dependencies, 247 partial image APIs plus 3 metadata APIs, 176 product files and 83 SDK files. Measured source `9e1edffc` and native source `c29dc44` were integrated at `6a107a7730baa2c4c6d64210ac52dfcea3a0ec44`. The integration retains the current lazy frontend; 518 separately scoped Python-overlay checks support its correctness. The timing validates 9e, not the current wrapper cost or whole-library performance.

The genuine installed SDK separately passed one recent-primitives client family in two modes, each with workers 1/4. This is distinct from the declared 26-family/52-check campaign. The untimed local interface preflight used retained CPython 3.11.14/pytest 9.1.1 and authenticated scientific packages/core/SDK; the hosted timed job used the locked genuine runtime. Full supported-profile qualification remains the separate 241-API baseline.

NumPy axis/layout conversion, fill/mean and exceptional/rescale routes remain explicit providers. The benchmark includes the complete public call, allocation and destruction. All exact output/oracle/input words, storage, FP/worker state and before/after identities passed. No isolated-stage causal or full-domain completion claim is made.
