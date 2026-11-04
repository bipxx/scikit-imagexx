# Drawing line-boundary candidate: rejected comparison

**DISCARD.** The tiny-line target improves **1.981077× direct** and **1.924067× normalized** over the C++ baseline, and beats original scikit-image in all four candidate passes (1.056623–1.086061×). Five control workloads fail the unchanged 0.97 worst-pair floors. The candidate remains unmerged; this is no accepted performance advance.

[Run 41869540371](https://github.com/bipxx/scikit-imagexx/actions/runs/41869540371) completed all eight scientific passes. Its failure is the fixed scientific decision, not a bootstrap failure. The [JSON publication](draw-line-boundary-same-host.json) embeds the complete once-retained 666,967-byte ZIP, all 117 member hashes, eight reports, 4,408 journal events, 1,344 samples, runtime/source receipts, and the independent review program.

AMD EPYC 9V45 96-Core Processor, CPU affinity `[1]`, one worker and one reference pool, eight distinct processes in A B B A B A A B order. Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0. All 192 function warmup groups exceeded 250 ms; accepted batches lasted 100.872865–154.759371 ms. Builds and tests were isolated from measurement.

| Workload | Baseline C++ ms | Candidate C++ ms | Direct gain | Normalized gain | Original/candidate range | Added ms |
|---|---:|---:|---:|---:|---:|---:|
| line/tiny | 0.002465226 | 0.001244387 | 1.981077× | 1.924067× | 1.056623–1.086061× | -0.001220839 |
| line/long | 0.069483079 | 0.068291373 | 1.017450× | 1.016679× | 1.672162–1.681927× | -0.001191705 |
| rectangle/tiny | 0.006657257 | 0.006583619 | 1.011185× | 1.011876× | 2.221812–2.294583× | -0.000073639 |
| rectangle/2d | 0.038667772 | 0.038093930 | 1.015064× | 1.022932× | 0.946603–0.968757× | -0.000573842 |
| rectangle/3d | 0.029928562 | 0.030279419 | 0.988413× | 1.002735× | 1.121991–1.198994× | +0.000350857 |
| disk/tiny | 0.017494181 | 0.017246698 | 1.014350× | 1.006446× | 1.475080–1.480197× | -0.000247483 |
| disk/dense | 0.586062473 | 0.590486924 | 0.992507× | 0.966780× | 3.143057–3.271593× | +0.004424451 |
| disk/clipped | 0.112436302 | 0.112474983 | 0.999656× | 0.992271× | 2.948094–3.093417× | +0.000038682 |
| ellipse/tiny | 0.017263524 | 0.017175243 | 1.005140× | 1.003687× | 1.472599–1.484017× | -0.000088281 |
| ellipse/axes | 0.199958291 | 0.201249004 | 0.993586× | 0.967783× | 3.034319–3.059946× | +0.001290713 |
| ellipse/rotated | 0.235002261 | 0.234431646 | 1.002434× | 0.995328× | 2.859598–2.982575× | -0.000570615 |
| ellipse/clipped | 0.076243086 | 0.075636093 | 1.008025× | 1.013152× | 2.648932–2.675909× | -0.000606993 |

Times are geometric means of the four pass medians; each median uses seven accepted samples. Direct gain is baseline/candidate C++ time; normalized gain compares original/C++ speedups. Ratios below one and positive added time are losses. The complete raw samples and all 16 direct and normalized pair ratios per workload remain in JSON.

| Fixed gate | Result |
|---|---|
| `target_balanced_direct` | PASS |
| `target_balanced_normalized` | PASS |
| `target_all_pairs_direct` | PASS |
| `target_all_pairs_normalized` | PASS |
| `controls_all_pairs_direct` | FAIL |
| `controls_all_pairs_normalized` | FAIL |
| `preserved_original_wins` | PASS |

Target balanced floors are 1.20 and all-pair floors 1.10. Every control requires all 16 direct and normalized baseline/candidate pairs to reach 0.97. All ten preserved original-win workloads beat scikit-image in every candidate pass. Rectangle/2d retains its original-performance loss in the table and raw evidence; it was not one of the ten preserved original wins.

| Control | Worst direct | Worst normalized | Direct ≥0.97 | Normalized ≥0.97 |
|---|---:|---:|---|---|
| line/long | 0.998716× | 1.007176× | PASS | PASS |
| rectangle/tiny | 0.971049× | 0.996216× | PASS | PASS |
| rectangle/2d | 0.980480× | 0.986289× | PASS | PASS |
| rectangle/3d | 0.965433× | 0.953717× | FAIL | FAIL |
| disk/tiny | 0.991815× | 0.999808× | PASS | PASS |
| disk/dense | 0.967899× | 0.913001× | FAIL | FAIL |
| disk/clipped | 0.969410× | 0.946422× | FAIL | FAIL |
| ellipse/tiny | 0.979058× | 0.994095× | PASS | PASS |
| ellipse/axes | 0.968601× | 0.937820× | FAIL | FAIL |
| ellipse/rotated | 0.979418× | 0.947683× | PASS | FAIL |
| ellipse/clipped | 0.982217× | 0.989641× | PASS | PASS |

The explicit variation policy is these all-16 worst-pair floors. There is no separate variance estimator or claim that variation is absent. The measured two-file boundary candidate addresses the tiny-line target bottleneck, but this experiment does not establish the cause of the control losses. It does not isolate a causal contribution from each changed file.

Controller `b10b9c65d4ad8fc0dc81b53716e17ba1a559d425` used frozen contract `75529f3a4377f95d10be0374c8880eed349ea079931fe854d25dafac3e82a0df`. Baseline source `566c1fcdc50dc6cdc0bb2c7a9de0a0e56449b6f8` and candidate source `0755818b0b8416947a47a8ba074b8e2388af27a5` retain identical 13 dependency wheels, 14 runtime versions and 84 SDK files. Each source closure has 453 authenticated production files; only the two pinned production files change. The candidate native change `b2d431661c3f834949f5057b9dfdcb419c1d9d44` remains unmerged. The comparison measures these prepared products, not the later current-252 product or whole-library parity/speed.

The independent review reproduced the archived decision and every median/gate exactly using the frozen controller validator, checked raw calibration journals and exact output/native witnesses, authenticated source bytes from Git, checked runtime equality and eight unchanged process receipts, and verified the two-hour execution clock. The measurement artifact was uploaded within the sealed deadline. The prior failed preparation, rejected controller contracts, and artifact-kind correction remain historical evidence; none is relabeled as scientific success.

Closeout: reused the once-retained ZIP and its existing SHA receipt; no duplicate archive hashing, download, original/native execution, timing, build, environment, or POST. One metadata-only assertion initially assumed the terminal receipt had no job fields; its diagnostic is retained, and required-field validation corrected it. Next owner is root: publish this DISCARD, retain the target gain and every control failure, and require a new evidenced mechanism and prospective contract before another experiment. No unchanged rerun or gate waiver.
