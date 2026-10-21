# Linear color branch optimization

Historical fixed 16-case experiment: eight APIs, float32/float64, uniform dark RGB/XYZ or low-lightness Lab/Luv, allocation-inclusive calls, and one worker for both libraries. Values and dtype are checked against original scikit-image before seven warmed samples. Fallback is disabled.

Baseline kernel `aa6e25125217fa5a478fcacf172e612c6cb0839e`; candidate `0316dbabad686de4b54c680545c025918114a8a3`. These snapshots predate the stain port. [Raw samples](linear-color-tuning.json) are separate from [current-source verification](STATUS.md).

Native geometric-mean latency: **4.1235 ms → 0.7731 ms**, a **5.33×** improvement over the prior C++ code.

| API | Input | Size | Previous C++ ms | New C++ ms | C++ improvement | Original / new |
|---|---|---:|---:|---:|---:|---:|
| rgb2xyz | float32 | 512² | 4.4747 | 0.2380 | 18.80× | 16.92× |
| rgb2lab | float32 | 512² | 10.2878 | 0.9789 | 10.51× | 11.39× |
| rgb2luv | float32 | 512² | 7.4767 | 0.8970 | 8.33× | 8.07× |
| xyz2lab | float32 | 512² | 4.8097 | 0.3150 | 15.27× | 20.73× |
| xyz2luv | float32 | 512² | 1.9338 | 0.2148 | 9.00× | 17.09× |
| xyz2rgb | float32 | 512² | 5.1278 | 3.8617 | 1.33× | 2.96× |
| lab2rgb | float32 | 256² | 1.4372 | 0.1715 | 8.38× | 10.07× |
| luv2rgb | float32 | 256² | 1.3071 | 0.1285 | 10.17× | 11.21× |
| rgb2xyz | float64 | 512² | 5.7210 | 1.6514 | 3.46× | 4.17× |
| rgb2lab | float64 | 512² | 12.4789 | 2.0868 | 5.98× | 8.73× |
| rgb2luv | float64 | 512² | 10.1043 | 2.1385 | 4.73× | 7.82× |
| xyz2lab | float64 | 512² | 6.1028 | 1.8918 | 3.23× | 6.27× |
| xyz2luv | float64 | 512² | 3.2518 | 1.5034 | 2.16× | 4.54× |
| xyz2rgb | float64 | 512² | 6.1688 | 3.9655 | 1.56× | 3.93× |
| lab2rgb | float64 | 256² | 1.6446 | 0.5671 | 2.90× | 7.90× |
| luv2rgb | float64 | 256² | 1.5598 | 0.2870 | 5.43× | 11.58× |

The implementation avoids power/cube-root evaluation only when every SIMD lane selects the linear branch. Mixed blocks keep the existing arithmetic. Use `python benchmarks/tune_linear_color.py --output build/linear.json` for the current fixed workload. Keep the harness and its `run.py` helper available when installing historical kernels. Ordinary system load, no affinity or frequency locking; these ratios are not universal guarantees.
