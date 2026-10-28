# MapArray: small-map and strided-output candidate

**KEEP under the frozen contract.** Same-host geometric-mean gain over the C++ baseline is **2.607984454x direct / 2.606253914x normalized** across nine public workloads. The balanced speedup over scikit-image changes from 1.267406754x to 3.303183812x. All three targets repeat above the 1.03x gates, and all five previous original wins remain wins in both candidate passes.

**Remaining losses:** dense uint8 256x256 regresses against the C++ baseline by +0.000086275 ms (0.998309707x direct, 0.998930119x normalized). Signed sparse int64 1024x1024 loses to scikit-image in the second candidate pass (0.991050822x); its first pass is 1.025673780x and balanced ratio 1.008213689x. There are no balanced original losses. These results do not establish complete performance superiority.

[Raw reports and comparison](map-array-small-same-host.json) preserve all four original JSON reports, the comparison, their numeric text, input/output metadata and hashes, and **504 individual samples**. The [initial mixed result](map-array-minimum-initial.md) remains published.

| Workload | Baseline C++ ms | Candidate C++ ms | Original / baseline C++ | Original / candidate C++ | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense-u8-u16-64 | 0.007051050 | 0.006999604 | 2.919101956x | 2.961694582x | 1.007349879x | 1.014591003x | -0.000051446 |
| dense-u8-u16-256 | 0.050954904 | 0.051041178 | 2.780521754x | 2.777546928x | 0.998309707x | 0.998930119x | +0.000086275 |
| dense-u8-u16-1024 | 0.754073409 | 0.752986597 | 2.741945732x | 2.748234321x | 1.001443335x | 1.002293477x | -0.001086812 |
| signed64-sparse-f64-256 | 0.947789508 | 0.064352944 | 0.272185632x | 4.014447726x | 14.727989941x | 14.748933276x | -0.883436565 |
| signed64-sparse-f64-1024 | 19.492064055 | 5.428598345 | 0.280610128x | 1.008213689x | 3.590625575x | 3.592934072x | -14.063465709 |
| uint64-wide-c128-256 | 0.584902982 | 0.108459647 | 0.923330324x | 4.943119980x | 5.392816588x | 5.353576995x | -0.476443335 |
| uint64-wide-c128-1024 | 9.573527355 | 1.726723725 | 1.012705895x | 5.606827723x | 5.544330696x | 5.536481766x | -7.846803629 |
| strided-u16-u32-512 | 0.315702796 | 0.313972968 | 7.787847666x | 7.795055507x | 1.005509480x | 1.000925524x | -0.001729828 |
| strided-out-u8-f32-65536 | 0.205431831 | 0.058933103 | 0.681684336x | 2.368226659x | 3.485847848x | 3.474081087x | -0.146498728 |

C++ times and original ratios above are geometric means of the two respective passes. Original ratios mean original time / C++ time. All per-pass medians remain in the raw reports. For clarity, the original reference medians and both candidate original ratios are also shown below.

| Workload | Original ms during baseline (GM) | Original ms during candidate (GM) | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|
| dense-u8-u16-64 | 0.020582735 | 0.020730690 | 2.958140266x | 2.965253169x |
| dense-u8-u16-256 | 0.141681219 | 0.141769269 | 2.781399569x | 2.773699624x |
| dense-u8-u16-1024 | 2.067628366 | 2.069383609 | 2.749763756x | 2.746705736x |
| signed64-sparse-f64-256 | 0.257974686 | 0.258341529 | 3.930406493x | 4.100285957x |
| signed64-sparse-f64-1024 | 5.469670586 | 5.473187165 | 1.025673780x | 0.991050822x |
| uint64-wide-c128-256 | 0.540058660 | 0.536129047 | 4.869583215x | 5.017767242x |
| uint64-wide-c128-1024 | 9.695167589 | 9.681442453 | 5.613921730x | 5.599742679x |
| strided-u16-u32-512 | 2.458645280 | 2.447436712 | 7.765703687x | 7.824518267x |
| strided-out-u8-f32-65536 | 0.140039661 | 0.139566946 | 2.336450813x | 2.400434658x |

| Group | Cases | Direct gain | Normalized gain | Original / candidate C++ |
|---|---:|---:|---:|---:|
| targets | 3 | 5.691245788x | 5.688746789x | 2.124225188x |
| preserved_original_wins | 5 | 1.412087022x | 1.412834630x | 3.971534795x |
| dense | 3 | 1.002360639x | 1.005249054x | 2.827603290x |
| signed-sparse | 2 | 7.272049047x | 7.279556641x | 2.011820358x |
| strided-input | 1 | 1.005509480x | 1.000925524x | 7.795055507x |
| strided-output | 1 | 3.485847848x | 3.474081087x | 2.368226659x |
| wide | 2 | 5.468048879x | 5.444261329x | 5.264524873x |
| complex128 | 2 | 5.468048879x | 5.444261329x | 5.264524873x |
| float32 | 1 | 3.485847848x | 3.474081087x | 2.368226659x |
| float64 | 2 | 7.272049047x | 7.279556641x | 2.011820358x |
| uint16 | 3 | 1.002360639x | 1.005249054x | 2.827603290x |
| uint32 | 1 | 1.005509480x | 1.000925524x | 7.795055507x |

## Fixed decision and scope

One candidate, frozen at 2026-10-28T12:27:11.951171Z with deadline 14:27:11.951171Z. Completed 12:46:55.330894Z. Baseline source is `b08aef5fb21cb7bfe87ffa04c35767eab1727d8e`; candidate is `ef5d9ae4647c844ab948a2a36ddfdb86de665ea3`. The candidate changes only the four owned native/kernel-test files plus its prospective contract. Public prepared-buffer ABI, binder, frontend, factory, original source, harness and tolerances remain fixed.

The bundle selects up to eight sparse mapping entries using packed integer comparisons and last-match selection, and permits dense SIMD gathers with strided destination stores. Larger sparse maps retain the existing hash route. Value snapshots, duplicate last-key behavior, aliases, integer wrapping, raw floating/complex words, signed strides and optional global workers retain their parity requirements. Observed gains belong to this whole public-call bundle; no per-stage causal contribution or measured memory-traffic reduction is claimed.

The fixed order is baseline/candidate/candidate/baseline, all on one fresh host, one global worker and four reference pool limits of one. Each of nine workloads has seven paired rounds targeting 19 ms per report. Eight `out=None` workloads include output allocation and destruction; the one preallocated strided-output workload excludes output allocation and retains its backing storage. No input/value-alias benchmark is introduced.

Direct gain compares geometric means of baseline and candidate C++ times. Normalized gain compares their original/C++ ratios. KEEP requires overall direct and normalized geometric means strictly above 1.03; each candidate pass for each of the three signed-sparse/strided-output targets must exceed 1.03 against the respective baseline geometric means. The five previous original wins must beat original in both candidate passes, all correctness/identity gates must pass, and completion must precede the deadline. Remaining original losses are reported, without adding a new all-original-wins gate.

Before each report timer, the unchanged harness verifies full output bytes, dtype/layout, unchanged inputs/mappings/backing gaps, output identity and an independent last-key oracle against pinned scikit-image 0.26. Genuine `xx._core.map_array_apply` returns are observed with original Python and Cython forwarding blocked, and hooks restored before timing. Exports retain hashes rather than full image payloads. Offline review checks those hashes/metadata against all four reports and the initial report, authenticates the source prechecks, all samples/medians and eight before/after identities, and recomputes the frozen decision independently.

Six native modes passed on the first source/compile attempt: 15,777 controls each for normal/scalar/sanitizer modes, 7,889 each for portable modes, including all 74 frozen profile returns and new 8/9-entry boundary, tail, alias, stride, width, worker and FP controls. Old control streams remain exact after removing only the prospectively added records. Both genuine product preparations passed 26 normal/scalar XML pairs (5,872), color 3,696, matching 1,612, smoke 1,423/242 registered functions, and the 4/4/4/9 intake guards. Current source census is 239 partially native image APIs plus three metadata APIs; full four-profile qualification remains **203 image APIs**.

## Provenance

- [Single ABBA run 40690409031](https://github.com/bipxx/scikit-imagexx/actions/runs/40690409031), workflow source `312406cf1fe28ec32060ce3c425a5152ecb7ee59`.
- [Baseline genuine Release 40685511797](https://github.com/bipxx/scikit-imagexx/actions/runs/40685511797) / [artifact 12814229633](https://github.com/bipxx/scikit-imagexx/actions/runs/40685511797/artifacts/12814229633).
- [Candidate prepare 40688421947](https://github.com/bipxx/scikit-imagexx/actions/runs/40688421947) / [artifact 12817505014](https://github.com/bipxx/scikit-imagexx/actions/runs/40688421947/artifacts/12817505014); all 22 timing steps skipped.
- [Comparison artifact 12817720322](https://github.com/bipxx/scikit-imagexx/actions/runs/40690409031/artifacts/12817720322): 41,787 bytes, 39 members, downloaded once and all member size/SHA/CRC checked.
- CPU: AMD EPYC 9V74 80-Core Processor. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, AVX2 enabled. All 13 runtime dependency wheels and scientific package/library hashes match exactly; no test-runner normalization was used. Each product wheel has 157 members and 71 SDK members. Baseline has 1,717 committed source blobs; candidate 1,718.

| Proof | SHA-256 |
|---|---|
| Combined raw publication | `a8944655eeb931e969601402dbc820b6a67f239acf2acef0f934c840b959ad45` |
| Comparison artifact ZIP | `1403ac17feeeccf86c1e1fda7aa3b495eb1ef7643498bc1d3231ff98e0040ce8` |
| Baseline ZIP | `b006fa18d41fdb350d0e754829b13bad9a1fe09687df84f807e4a1ff72e5ee8d` |
| Candidate ZIP | `6899333059779e988b1e1cd8bb11a12aca98766c95b36e764277f13207787de6` |
| Unchanged harness | `a6f25c1e7fe60ab9a86cb38406a2a49054a052d1b0e29dc555545a1cb021667b` |
| Nine-workload contract | `360e96314eb6e4ddbbab566acd23790ab8e8dfe38e7a1525242132f980dc17b8` |
| Prospective candidate contract, exact bytes | `eb9305a7ec836e79899c9ad8558a4e588beb2a1d1eeff885b75f7c2faad32dad` |
| Candidate contract, normalized source fingerprint | `771727f45957ea0f021f9afc0811b9245dcfe723ea2fd2bc97cf59760486de01` |
| Baseline native core | `cbbbba06720563d586f1acee6aa8beda93dc41053d05b99c1490701b7923dea0` |
| Candidate native core | `893ce8d0a2a814920da33e69fc6352d68be6699f24557310f93e37fab4cae462` |
| Independent preparation review | `d1abd6c44f6684cf54eaa2dace0d871efb2249d68fe80004882db180afde095b` |
| Independent comparison review | `30bdb80b974de26f80b3dbf215144a64e5164be2cca7e7330093fd6daf827118` |
