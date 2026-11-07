# Current281 minimum-host initial comparisons

Scoped candidate performance for qualified production `dda9cc83d9434a10616f84cf1e1125d93f5f36fd`, comparison controller `373af1af261e5c6bc522f533f19036fc65603c81`. Main still ships271 partial paths.

Original scikit-image0.26.0 vs installed native public calls, allocation-inclusive, one worker; seven samples per arm, at least100ms per sample and250ms warmup. Both arms share the actual minimum runtime/host. All losses are retained. These24 cases cover only integer finite Radon N<=257 and uint8/float32 C/F flood operations; no full API, platform, parallel scaling or prior-C++ claim.

[Root performance admission](../../coverage/evidence/current281-radon-flood-performance-root-admission-v1.json). Actual runs: [Radon42360529492](https://github.com/bipxx/scikit-imagexx/actions/runs/42360529492), [flood42360529644](https://github.com/bipxx/scikit-imagexx/actions/runs/42360529644).

| Family / case | Original ms | C++ ms | Speedup | Outcome |
|---|---:|---:|---:|---|
| finite-radon / `transform.frt2/n7/uint8/C` | 0.163957 | 0.00165999 | 98.77x | win |
| finite-radon / `transform.frt2/n31/uint16/C` | 3.87194 | 0.0143579 | 269.7x | win |
| finite-radon / `transform.frt2/n127/uint32/C` | 67.1933 | 0.468782 | 143.3x | win |
| finite-radon / `transform.frt2/n127/uint32/F` | 67.1729 | 0.472901 | 142x | win |
| finite-radon / `transform.frt2/n127/uint32/reverse` | 66.9224 | 0.479046 | 139.7x | win |
| finite-radon / `transform.frt2/n257/uint16/C` | 278.312 | 2.73961 | 101.6x | win |
| finite-radon / `transform.frt2/n257/uint16/F` | 277.332 | 2.78372 | 99.63x | win |
| finite-radon / `transform.frt2/n257/uint16/step` | 278.666 | 2.76105 | 100.9x | win |
| finite-radon / `transform.ifrt2/n7/uint8/C` | 0.168535 | 0.0022651 | 74.4x | win |
| finite-radon / `transform.ifrt2/n31/uint16/C` | 3.94209 | 0.0159164 | 247.7x | win |
| finite-radon / `transform.ifrt2/n127/uint32/C` | 67.3292 | 0.490549 | 137.3x | win |
| finite-radon / `transform.ifrt2/n127/uint32/F` | 67.5514 | 0.493362 | 136.9x | win |
| finite-radon / `transform.ifrt2/n127/uint32/reverse` | 67.2955 | 0.500772 | 134.4x | win |
| finite-radon / `transform.ifrt2/n257/uint16/C` | 278.144 | 2.87363 | 96.79x | win |
| finite-radon / `transform.ifrt2/n257/uint16/F` | 279.963 | 2.88987 | 96.88x | win |
| finite-radon / `transform.ifrt2/n257/uint16/step` | 279.653 | 2.88859 | 96.81x | win |
| flood-fill / `flood-uint8-C` | 0.78655 | 0.686513 | 1.146x | win |
| flood-fill / `flood-uint8-F` | 0.776897 | 0.74138 | 1.048x | win |
| flood-fill / `flood-float32-C` | 0.778725 | 0.725172 | 1.074x | win |
| flood-fill / `flood-float32-F` | 0.799692 | 0.772267 | 1.036x | win |
| flood-fill / `flood_fill-uint8-C` | 0.840405 | 1.49715 | 0.5613x | loss |
| flood-fill / `flood_fill-uint8-F` | 0.834163 | 1.57642 | 0.5291x | loss |
| flood-fill / `flood_fill-float32-C` | 0.845146 | 1.55658 | 0.543x | loss |
| flood-fill / `flood_fill-float32-F` | 0.860736 | 1.62778 | 0.5288x | loss |

20wins, four losses;336retained samples and48warmup groups.

- [Raw frt2.json](finite-radon/frt2.json)
- [Raw frt2.batches.jsonl](finite-radon/frt2.batches.jsonl)
- [Raw ifrt2.json](finite-radon/ifrt2.json)
- [Raw ifrt2.batches.jsonl](finite-radon/ifrt2.batches.jsonl)
- [Raw flood-fill.json](flood-fill/flood-fill.json)
- [Raw flood-fill.batches.jsonl](flood-fill/flood-fill.batches.jsonl)

Cohort caller restoration and native-entry acceptance are retained source-checked terminal evidence; host helpers additionally retain32positive/sixnegative full32 controls. No additional target process-entry FP dump or independent inner worker-getter observation is claimed. Ordered flood BFS remains serial within its existing worker contract. Broader dtype/layout/nonfinite/platform/provider/FP domains remain open.
