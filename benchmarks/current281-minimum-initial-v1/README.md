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

## Corrected JOIN / utilities and rolling-ball cohort

Comparison controller `8ed0e2d223719adb5512c3f1e037e90c9265695f`; qualified native production and sampling method above are unchanged. Earlier373runs failed before timing and remain failed. New runs: [JOIN42362296756](https://github.com/bipxx/scikit-imagexx/actions/runs/42362296756), [rolling42362296718](https://github.com/bipxx/scikit-imagexx/actions/runs/42362296718).

[Root scoped admission](../../coverage/evidence/current281-join-rolling-performance-root-admission-v1.json):40cases,560samples,80warmupgroups,18wins and22losses. Combined with the24cases above:64cases/896samples/128warmupgroups,38wins and26losses. These are candidate-domain results, not a library-wide or prior-C++ superiority claim.

| Stream / case | Original ms | C++ ms | Speedup | Outcome |
|---|---:|---:|---:|---|
| join / `u8-labels-C` | 0.863478 | 1.53202 | 0.5636x | loss |
| join / `u8-maps-reverse` | 1.27712 | 1.63149 | 0.7828x | loss |
| join / `i16-labels-F` | 3.81406 | 5.89431 | 0.6471x | loss |
| join / `i16-maps-C` | 5.48819 | 5.546 | 0.9896x | loss |
| join / `u16-labels-C` | 3.76999 | 5.4889 | 0.6868x | loss |
| join / `u16-maps-F` | 5.53414 | 5.85608 | 0.945x | loss |
| join / `i32-labels-reverse` | 53.9315 | 27.3148 | 1.974x | win |
| join / `i32-maps-C` | 59.8795 | 25.9102 | 2.311x | win |
| crop / `view-C` | 0.00169664 | 0.00212165 | 0.7997x | loss |
| crop / `view-F` | 0.00171262 | 0.00213286 | 0.803x | loss |
| crop / `view-reverse` | 0.00169638 | 0.00211287 | 0.8029x | loss |
| crop / `view-3D` | 0.00186192 | 0.00233601 | 0.797x | loss |
| crop / `copy-C` | 0.0313102 | 0.0321452 | 0.974x | loss |
| crop / `copy-F` | 0.0164125 | 0.0178288 | 0.9206x | loss |
| crop / `copy-reverse-K` | 0.0222946 | 0.0422796 | 0.5273x | loss |
| crop / `copy-object-provider` | 0.00283675 | 0.00335468 | 0.8456x | loss |
| slice / `view-C` | 0.0015828 | 0.00240539 | 0.658x | loss |
| slice / `copy-C` | 0.0116649 | 0.016462 | 0.7086x | loss |
| slice / `view-F` | 0.00156078 | 0.00239727 | 0.6511x | loss |
| slice / `copy-F-C-order` | 0.628689 | 0.638182 | 0.9851x | loss |
| slice / `reverse-view` | 0.00159946 | 0.00241813 | 0.6614x | loss |
| slice / `reverse-copy` | 0.072359 | 0.0789025 | 0.9171x | loss |
| slice / `permuted-axes` | 0.00155271 | 0.00246779 | 0.6292x | loss |
| slice / `negative-bound-copy` | 0.0171364 | 0.0193097 | 0.8874x | loss |
| blocks / `view_as_blocks/C` | 0.0102807 | 0.00938404 | 1.096x | win |
| blocks / `view_as_blocks/F` | 0.0102346 | 0.00932673 | 1.097x | win |
| blocks / `view_as_blocks/reverse` | 0.010334 | 0.00945387 | 1.093x | win |
| blocks / `view_as_blocks/readonly` | 0.0100693 | 0.00916903 | 1.098x | win |
| windows / `view_as_windows/C` | 0.0142415 | 0.0121522 | 1.172x | win |
| windows / `view_as_windows/F` | 0.0142987 | 0.0122354 | 1.169x | win |
| windows / `view_as_windows/reverse` | 0.0142814 | 0.0122162 | 1.169x | win |
| windows / `view_as_windows/readonly` | 0.0140528 | 0.0118729 | 1.184x | win |
| rolling-ball / `small-u8` | 0.0603771 | 0.0187724 | 3.216x | win |
| rolling-ball / `dense-f32` | 1.09778 | 0.78314 | 1.402x | win |
| rolling-ball / `dense-f64` | 0.336545 | 0.22133 | 1.521x | win |
| rolling-ball / `wide-f32` | 1.43064 | 1.14832 | 1.246x | win |
| rolling-ball / `stride-f32` | 0.325596 | 0.216169 | 1.506x | win |
| rolling-ball / `3d-u16` | 0.374346 | 0.110032 | 3.402x | win |
| rolling-ball / `nan-f32` | 0.420779 | 0.217872 | 1.931x | win |
| rolling-ball / `border-tiny` | 0.0562517 | 0.0148668 | 3.784x | win |

Raw reports preserve original source/runtime metadata; journals retain every warmup/calibration/sample batch. Duplicate case IDs are distinguished by their stream:

- [Raw join-util-current275/join.json](join-util-current275/join.json)
- [Raw join-util-current275/join.batches.jsonl](join-util-current275/join.batches.jsonl)
- [Raw join-util-current275/crop.json](join-util-current275/crop.json)
- [Raw join-util-current275/crop.batches.jsonl](join-util-current275/crop.batches.jsonl)
- [Raw join-util-current275/slice.json](join-util-current275/slice.json)
- [Raw join-util-current275/slice.batches.jsonl](join-util-current275/slice.batches.jsonl)
- [Raw join-util-current275/blocks.json](join-util-current275/blocks.json)
- [Raw join-util-current275/blocks.batches.jsonl](join-util-current275/blocks.batches.jsonl)
- [Raw join-util-current275/windows.json](join-util-current275/windows.json)
- [Raw join-util-current275/windows.batches.jsonl](join-util-current275/windows.batches.jsonl)
- [Raw rolling-ball/rolling-ball.json](rolling-ball/rolling-ball.json)
- [Raw rolling-ball/rolling-ball.batches.jsonl](rolling-ball/rolling-ball.batches.jsonl)

ArrayMap class ownership, inner worker-getter observation, broad dtype/layout/provider/FP/platform coverage and main delivery remain open as specified in the receipt. Native views do not imply copy ownership or fully native class protocols.
