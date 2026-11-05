# RegionProperties warmed self-import diagnostic

This is a local same-core diagnostic, not an original-scikit-image comparison or release acceptance. Function0 is the unchanged warmed wrapper from55465253; function1 removes only its repeated from-self import AST node after provider globals have resolved. Both use the same prepared C++ core and inputs. Production code was unchanged.

| Workload | Source wrapper ms | Without import ms | Source/diagnostic |
|---|---:|---:|---:|
| regionprops-state/tiny2d-constructor | 0.021216313 | 0.019777059 | 1.072773887x |
| regionprops-state/tiny2d-properties | 0.063467166 | 0.059972741 | 1.058266886x |
| regionprops-state/large2d-f-coords | 5.903418500 | 5.851182053 | 1.008927503x |

42 paired samples, seven per variant/workload, accepted batches at least100ms and warmup at least250ms per variant. Global worker and dependency pools were1. Root and both local implementation workers reserved a quiet window; no builds, tests or captures overlapped timing.

Hardware: Intel Core i7-12700K under WSL; Python3.11.14, NumPy1.24.4, SciPy1.15.3. The hosted rejected experiment used Python3.11.17 and a different CPU. These local ratios isolate a warmed import contribution; they do not establish that the hosted regression is fully resolved. Cold lazy resolution, patched aliases and the frozen20-cohort acceptance gates remain mandatory. Legacy sampler labels in raw data are mapped explicitly in function_roles; neither variant is original scikit-image.
