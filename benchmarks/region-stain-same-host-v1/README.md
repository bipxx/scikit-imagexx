# Region/stain candidate: DISCARD

Exact controller119eb28b, baselineeea321b2 and candidate55465253; one isolated AMD EPYC9V45 host, one global worker, four ABBA passes and1,120 accepted samples. All six real preflights and before/after installed-input checks passed.

Two fixed checks fail, both for tiny2d-constructor. Targets passing does not waive these controls; candidate source remains private. These are workload-specific results, not full-library superiority or complete API parity.

| Case | Pair1 candidate/prior | Pair1 normalized | Pair2 candidate/prior | Pair2 normalized | Limit | Result |
|---|---:|---:|---:|---:|---:|---|
| regionprops-state/tiny2d-constructor | 1.065461560 | 1.040142382 | 1.044481259 | 1.059337091 | 1.05 | FAIL |
| regionprops-state/tiny2d-properties | 1.012120160 | 0.977770836 | 1.002725363 | 1.005312121 | 1.05 | PASS |
| regionprops-state/medium2d-constructor-sparse | 0.564173128 | 0.554959774 | 0.559043354 | 0.561044777 | 1.05 | PASS |
| regionprops-state/medium2d-properties-cache-on | 0.380224114 | 0.383448394 | 0.380505735 | 0.378998243 | 1.05 | PASS |
| regionprops-state/medium2d-transpose-cache-off | 0.106891360 | 0.106377389 | 0.108006116 | 0.109411922 | 1.05 | PASS |
| regionprops-state/large2d-c-cache-off | 0.405078434 | 0.386900889 | 0.405784615 | 0.441520962 | 1.05 | PASS |
| regionprops-state/large2d-f-coords | 0.152727160 | 0.159628936 | 0.154536065 | 0.154586361 | 0.50 | PASS |
| regionprops-state/medium2d-reverse-sparse | 0.344726003 | 0.340117851 | 0.348653468 | 0.390909037 | 1.05 | PASS |
| regionprops-state/small3d-constructor | 0.967211407 | 0.937062844 | 0.939077108 | 0.940408580 | 1.05 | PASS |
| regionprops-state/medium3d-f-cache-off | 0.084216399 | 0.083282354 | 0.084917304 | 0.088299621 | 0.50 | PASS |
| regionprops-state/large3d-cache-off | 0.354053070 | 0.359744606 | 0.354376353 | 0.376745387 | 1.05 | PASS |
| regionprops-state/large3d-sparse-spacing-offset | 0.217818309 | 0.214534355 | 0.212714475 | 0.212536692 | 1.05 | PASS |
| combine-half-bench/standard-f32 | 0.475543609 | 0.479414546 | 0.472565118 | 0.477557465 | 0.80 | PASS |
| combine-half-bench/standard-f64 | 1.001082754 | 1.000055138 | 0.991208670 | 0.986076742 | 1.05 | PASS |
| combine-half-bench/custom-k7 | 0.424214064 | 0.425507226 | 0.424592135 | 0.437518860 | 0.80 | PASS |
| combine-half-bench/reverse-k1 | 0.571890232 | 0.568923326 | 0.583794644 | 0.590959853 | 1.05 | PASS |
| old-f32-combine-custom1 | 0.229199666 | 0.227332613 | 0.220842734 | 0.220073672 | 1.05 | PASS |
| old-f32-combine-custom7 | 0.304515078 | 0.333372440 | 0.306414019 | 0.280552904 | 1.05 | PASS |
| old-f32-separate-custom7 | 0.996999497 | 1.004500934 | 0.993982517 | 0.955759302 | 1.05 | PASS |
| old-f32-combine-hed-f64matrix | 1.007347821 | 1.013392916 | 0.985546183 | 0.910469948 | 1.05 | PASS |

Pairs compare baseline pass1/candidate pass2 and baseline pass4/candidate pass3. Ratios below1 are faster. Both direct and original-normalized ratios must meet the frozen limit separately in both pairs. All original-reference and current-C++ raw times, journals and slower cases are retained in the adjacent files.

The recurring provider self-import added by the wildcard-import correctness fix is one demonstrated source difference to investigate alongside the changed native traversal. Its causal share of the tiny-call regression remains unmeasured. No unchanged experiment retry, relaxed threshold or automatic partial promotion follows this rejection.
