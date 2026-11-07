# Crop view-only private C++ comparison

Source `9e689ac283f15fec40eedb8cfefa8edf1144bc48`, minimum Python3.11.14 / NumPy1.24.4, workers1. One fixed pass, repeat7, samples>=100ms, warmup>=250ms. Actual112samples and16warmup groups; six faster medians, two slower. Source remains unaccepted.

**Both callables are C++.** Unmodified sampler raw labels `scikit-image`/`current` mean slot0=prior C++d7 binder/12d leaf and slot1=new C++9e binder/frontend/same12d leaf. Original scikit-image calls:0. These results do not establish original-library speed or release acceptance. Local triage; CPU identity was not retained with this experiment.

| Case | Prior C++ (us) | New C++ (us) | Prior/new |
|---|---:|---:|---:|
| view-C | 1.591 | 1.517 | 1.048972x |
| view-F | 1.400 | 1.387 | 1.009655x |
| view-reverse | 1.328 | 1.262 | 1.052050x |
| view-3D | 1.426 | 1.470 | 0.969704x |
| copy-C | 41.850 | 41.395 | 1.010989x |
| copy-F | 15.354 | 15.204 | 1.009807x |
| copy-reverse-K | 21.377 | 21.210 | 1.007856x |
| copy-object-provider | 2.424 | 2.508 | 0.966692x |

[Exact report](report.json), [exact journal](batches.jsonl). All raw labels and slower cases retained. View3D0.969704x and object-copy0.966692x remain losses. F-copy now1.009807x in this changed-source experiment; this does not erase the [f020 predecessor loss](../crop-metadata-private-v1/README.md). No unchanged retry. Independent result review, original-public FP/performance and genuine integration remain pending.
