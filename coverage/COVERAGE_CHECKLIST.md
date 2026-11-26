# Coverage checklist

This is the detailed completion view for the existing [execution plan](EXECUTION_PLAN.md) and [operational ledger](work_items.json). Those records still govern assignments, execution and evidence. This file adds no dispatch queue or permission to run work.

Updated: 2026-11-26. Shipping classifications come from [api.json](api.json); per-key contract focus comes from [the implementation paths](API_IMPLEMENTATION_PATHS.json). The paths are proposals, not proof of completed coverage.

Historical genuine speed result (`75b`, unshipped predecessor): [exact75b fixed24](evidence/speed-parity-genuine24-root-result-v1.json) passed output/FP/protocol and all4genuineprofiles/3stagers, but **fails the per-case speed gate**:20wins/4losses. Float32flood_fill C0.948235/F0.982839 and int64/uint64outer-find0.891151/0.845565 were the optimization gaps addressed by the delivered successor below; this batch is unshipped. No unchanged retiming. Native coverage counts/full-contract checkboxes remain unchanged.

Current delivered speed result: [all24 genuine release-build cases meet parity](evidence/speed-parity-genuine24-v2-root-result-v1.json), **1.013-9.734x**, in main `a5f3aa97`. [Four profiles and three packaging checks passed](evidence/speed-parity-whole-product-v2-root-admission-v1.json):965678XML0bad/304SDK/48corpora; [both complete source barriers passed](evidence/speed-parity-source-both-v2-root-admission-v1.json). Fixed one-thread, allocation-inclusive comparison on12700K/retainedLinux3.11.14/NumPy1.24.4/original0.26; other hosts/profiles and wider domains are unmeasured. The old75b4losses and [private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) remain separately scoped. [Seven merged checkouts retired](evidence/speed-parity-merged-checkout-retirement-v1.json),5.266GBlogical/5.324GBobserved free increase; [five temporary-folder policy blocks](evidence/speed-parity-v2-cleanup-policy-block-v1.json),21.73GBlogical, remain manual-only. Native classification and full-contract checkboxes stay unchanged.

The [previous main CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44147273549) failed with 88 region profile/lookup failures on NumPy 2.5.5 and 22 weighted-distance mismatches. Debug-gil passed. The historical record lacks the effective compiler commands. The two corrections below are in the current qualification candidate. [Failure triage](evidence/speed-parity-delivery-CI-triage-v1.json)

Main `14c95b05` pins NumPy 2.5.4 and SciPy 1.18.1. Retained installed Linux and Windows releases passed 88 region cases per dispatch mode on that tuple. Both complete source commands passed 4,710 tests and eight subtests with zero failures. [Region evidence](../build/speed-parity-ci-correction-v1/region-runtime/root-region-runtime-admission-v1.json) · [Source evidence](../build/speed-parity-ci-correction-v1/source-entry-v1/root-complete183-both-admission-v1.json)

The weighted contraction witness reproduced a scalar hue mismatch on the matching minimum Linux profile. Disabling contraction matched the original exactly in that primitive comparison. [Weighted contraction witness](evidence/speed-parity-weighted-contraction-witness-v1.json)

The first corrected installed build passed 3,360 affected public tests in each dispatch mode, the installed SDK consumer and all eight scoped FP restorations. It failed the speed gate for float32 CMC broadcast and float32 CIE76 strided and extra-channel layouts. The optimized successor resolved all three cases. [Earlier results](evidence/speed-parity-distance-correction-attempt-v1.json)

The optimized successor passed **100 of 100** affected one-thread cases, **1.011× to 7.492×**, on the matched hosted profile. It passed 3,414 public tests per mode, both installed SDK cases and eight FP restorations. Main includes its exact native changes and regression tests. Four-profile qualification passed with 965,678 outcomes, 304 SDK checks and 48 corpora, and all three packaging checks passed. The previous main CI passed the region and weighted-distance failure groups, then exposed setup failures in Linux JOIN and the polygon benchmark. Main `e27633bf` delivers the setup correction with 1,059 passing source outcomes per Python. Its automatic CI and library-wide speed coverage remain open. [Optimized results](evidence/speed-parity-distance-optimized-v1.json) · [Qualification run](https://github.com/bipxx/scikit-imagexx/actions/runs/44189709782) · [Previous CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44201387625) · [Corrected CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44207792316)

Ordinary CI on e27633bf passed JOIN in both modes on Linux and Windows, both minimum jobs and debug-gil. The broad normal run stopped on both platforms. Windows repeated 1,033 earlier failures. Ubuntu 26.04 reproduced 82 live native parity failures in fresh processes in both dispatch modes across Farid, local thresholds, LUV and Roberts, and 70 differences in upstream reconstruction traces. The retained Ubuntu 24.04 product passes the 82 live comparisons in both modes in fresh WSL processes with Python 3.13.8. The Ubuntu 26.04 x86-v4 diagnostic restored all 70 upstream traces. The candidate scopes that setting to original-only tests and disables compiler contraction for the four affected scalar files. The corrected wheel passes all 7,690 affected tests in each dispatch mode, including all 82 earlier failures. In the measured one-thread default cases, 56 of 63 meet speed parity with normal dispatch and 40 of 63 with SIMD disabled. The Farid and packed-byte color successor passes 6,966 installed tests per mode. On the measured Ubuntu host without x86-v4, all 45 changed-family default cases meet one-thread speed parity, with a minimum ratio of 1.251×. With SIMD disabled, 39 of 45 meet parity. On x86-v4, 17 of 21 focused default cases meet parity. Packed uint8 XYZ-to-Lab reaches 7.70× and Lab-to-LCh reaches 2.44×. Four float Lab conversions and 17 forced-scalar cases in that profile remain slower. [x86-v4 results](https://github.com/bipxx/scikit-imagexx/actions/runs/44226093016) [Changed-family timing](https://github.com/bipxx/scikit-imagexx/actions/runs/44225408426) Complete source, broader CI, release qualification and speed closure remain pending. [Installed parity](https://github.com/bipxx/scikit-imagexx/actions/runs/44217081215) · [Affected timing](https://github.com/bipxx/scikit-imagexx/actions/runs/44218556655) [Ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44207792316) · [Exact Ubuntu replay](https://github.com/bipxx/scikit-imagexx/actions/runs/44214933486)

Main `7a9bdee6` delivered the exact qualified portable parity corrections, including the H-extrema entry and historical rank-statistics packaging pin. Four profiles and three packaging checks passed, with 965,678 test outcomes, 304 SDK checks and 48 frozen corpora. The broad ordinary Linux and Windows jobs failed the same three crop and Li source assertions. CI-only successor `ab757ebc` preserves the installed production and SDK inputs and passed 1,839 affected tests and eight subtests per Python. Both full normal-dispatch suites passed. Ubuntu SIMD-disabled CI exposed 12 exact uint8-to-float rescale mismatches and 12 inherited-environment failures in controller fixtures. Windows SIMD-disabled CI reported only the same 12 controller failures. V28 passed both complete source checks, both full ordinary installed dispatch modes on both hosts, and independent four-profile qualification. Its Ubuntu constrained native suite found dynamic OpenMP floating-point and worker-count failures. V29 `2de9cbb0` passed both complete local GCC native modes, the complete portable CTest, and 50 installed SDK tests in each mode. Windows passed 121 normal native tests and 50 relocated SDK consumers. Constrained Windows passed 120 native tests and failed the RegionProps fixture’s cross-team register check. The successor fixture passed both MSVC modes and the affected GNU modes. The full constrained Windows command passed all 121 with the corrected test leaf and retained production library. [Local native and SDK results](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/windows-native-includes-v29-v1/root-native-corrected-scope-v30-v1.json) Both complete Python source runs failed the same 26 guards, with 4,720 passing tests and eight passing subtests per version. V31 `4f80eb70` passed 163 focused source checks on each Python after keeping the exact DDA executor and observer. Both complete V31 source checks passed 4,746 tests and eight subtests. V31 ordinary CI failed the same historical JOIN source assertion on Ubuntu and Windows. All four V31 qualification profiles failed at the SDK build. The authenticated compiler log showed SDK staging omitted a required native test header. V32 `e2c37695` corrects both setup defects and passed all 81 focused checks per Python. Both complete V32 commands passed 4,746 tests and eight subtests. Both V32 installed normal suites passed. Windows qualification found a build-identity path spelling issue after the SDK staging repair. V33 `ed980bcc` preserves exact output membership and verifies these same-file aliases. Focused checks and both complete source commands passed, including 4,752 tests and eight subtests per Python. V33 qualification passed all four profiles and three stagers. Ordinary V33 passed five jobs and failed both offline measurements on the old `regular_grid` comparison. [V33 ordinary terminal](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v33-complete-source-v1/ordinary-run44334711332-v1/terminal-monitor-handoff-v2.json) [Complete V33 source](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v33-complete-source-v1/root-release-entry-admission-v33-v1.json) · [V33 ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44334711332) · [V33 qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/44334711472) Rescale met the one-thread threshold in 18 of 18 normal cases and 11 of 18 SIMD-disabled cases. The minimum SIMD-disabled result was 0.703× against original scikit-image. A V34 private combined comparison passed all 36 cases, with a minimum 1.097× in SIMD-disabled mode. The exact hosted check passed 590 exposure tests per mode and met the speed target in 35 of 36 cases. The remaining miss is the default 1024×1024 uint8 case with SIMD disabled at 0.883×. V34 release was held. The V35 byte-range fix passed scoped functional checks in each mode and the changed-input local scan comparison. The exact V35 hosted result passed all 36 measured one-thread cases on Intel Xeon 6975P-C, with minimum 2.736× normal and 2.058× SIMD-disabled, and 590 exposure tests passing per mode. Both complete V35 source runs passed 4,753 tests and eight subtests. Hosted qualification passed all four profiles and three stagers. Independent verification passed 965,686 test outcomes, 304 SDK checks and 48 frozen corpora. Ordinary CI passed all seven jobs, including all 29 offline measurements per platform. Main now contains the exact V35 source. The shipped one-thread DDA baseline measured 32 cases on Intel Core i7-12700K, with nine wins and 23 losses. Blocks and windows passed all eight speed comparisons. JOIN was faster in one of eight cases, and all 16 crop and slice cases were slower. The largest measured gap is reversed int32 JOIN labels at 0.612× original speed and 8.210 ms added per call. Private signed-label optimization `09269135` passed its native parity checks. The three-case interposed comparison reached 1.065× for contiguous int32 mappings and 1.308× for int16 mappings. Reversed int32 labels remained at 0.702×, so this change is held on its private branch. [Private JOIN results](../build/speed-parity-ci-correction-v1/next-speed-entry/current-8a55ccfc-v1/join-private-public-preparation-v1/root-private-join-admission-v35-v1.json) Exact JOIN ArrayMap class parity and library-wide speed coverage remain open. [JOIN phase evidence](../build/speed-parity-ci-correction-v1/next-speed-entry/current-8a55ccfc-v1/join-loss-diagnostic-v1/root-join-phase-admission-v35-v1.json) [Exact shipped DDA32 baseline and raw receipts](../build/speed-parity-ci-correction-v1/next-speed-entry/current-8a55ccfc-v1/timing-preparation-v1/execution-v1/root-exact-dda32-baseline-admission-v35-v1.json) [Exact ordinary CI and promotion admission](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v35-complete-source-v1/root-exact-ordinary-and-promotion-admission-v35-v1.json) · [Verified main delivery](../build/speed-parity-ci-correction-v1/next-speed-entry/current-8a55ccfc-v1/root-main-delivery-v35-v1.json) [Independent exact product verification](../build/speed-parity-ci-correction-v1/release-entry-v1/run44352115032-v1/root-whole-product-admission-v1.json) [V35 hosted qualification terminal](../build/speed-parity-ci-correction-v1/release-entry-v1/run44352115032-v1/monitor/terminal-monitor-handoff-v1.json) [Complete V35 source](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v35-complete-source-v1/root-release-entry-admission-v35-v1.json) · [Full ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44352115100) · [Four-profile qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/44352115032) [V35 authenticated result](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/rescale-diagnostic-successor-v1/early-exact-rescale-v35-v1/run44349278898-v1/terminal-admission-v1.json) [V35 source admission](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v35-complete-source-v1/root-independent-source-admission-v35-v1.json) · [V35 hosted speed](https://github.com/bipxx/scikit-imagexx/actions/runs/44349278898) [Exact V34 result](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/rescale-diagnostic-successor-v1/early-exact-rescale-v34-v1/run44345502133-v1/terminal-admission-v1.json) [V34 independent source](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v34-complete-source-v1/root-independent-source-admission-v34-v1.json) · [V34 hosted speed](https://github.com/bipxx/scikit-imagexx/actions/runs/44345502133) [Authenticated timing](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/rescale-diagnostic-successor-v1/run44339577094-v1/root-rescale-medians-v1.json) [V33 source admission](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v33-complete-source-v1/root-independent-source-admission-v33-v1.json) [Complete source evidence](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v32-complete-source-v1/root-release-entry-admission-v32-v1.json) · [V32 ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44322890584) · [V32 qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/44322891169) [V32 source admission](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v32-complete-source-v1/root-independent-source-admission-v32-v1.json) [Ordinary failures](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v31-complete-source-v1/ordinary-run44310680729-v1/terminal-monitor-handoff-v2.json) · [SDK compiler failure](../build/speed-parity-ci-correction-v1/release-entry-v1/run44310680314-v1/monitor/modern-failed-sdk-transport-receipt-v1.json) [Complete V31 source admission](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v31-complete-source-v1/root-release-entry-admission-v31-v1.json) · [V31 ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44310680729) · [V31 qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/44310680314) [V31 independent admission](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v31-complete-source-v1/root-independent-source-admission-v31-v1.json) [Complete V29 source failures](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/scalar-tail-v28/v29-complete-source-v1/root-source-barrier-failed-v29-v1.json) [V28 ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44284344130) · [V28 qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/44284344089) [Qualified release](https://github.com/bipxx/scikit-imagexx/actions/runs/44245395473) · [Package admission](../build/speed-parity-ci-correction-v1/release-entry-v1/run44245395473-v1/root-whole-product-admission-v1.json) · [Earlier ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44256638241) · [Current ordinary CI](https://github.com/bipxx/scikit-imagexx/actions/runs/44264846992)

The wider float64 polynomial passed bitwise checks and 5,308 installed tests per mode. On the measured x86-v4 host, float64 XYZ-to-Lab and RGB-to-Lab reached 0.632× and 0.770× upstream speed. In a same-host comparison, it increased the native medians from 94.9 to 171.4 µs and 200.9 to 337.1 µs, losing all eight pairs for each API. The wide float64 path was rejected. The separate float32 wide path passed local fallback comparisons. Two hosted runners lacked effective x86-v4, so its enabled path and speed are still unverified. [Float64 bitwise checks](https://github.com/bipxx/scikit-imagexx/actions/runs/44241560332) · [Upstream comparison](https://github.com/bipxx/scikit-imagexx/actions/runs/44242610136) · [Same-host comparison](https://github.com/bipxx/scikit-imagexx/actions/runs/44244965094) · [Float32 preflight](https://github.com/bipxx/scikit-imagexx/actions/runs/44243143313)

The separate packed float32 Lab candidate b71bd4a2 passed 5,308 affected installed tests per dispatch mode. On the measured x86-v4 host, float32 XYZ-to-Lab reached 1.009× original and float32 RGB-to-Lab reached 0.990×. Float64 remained slower in both measured routes. A private float64 follow-up passed 1,539 bitwise cases per mode and shortened the two local kernel measurements by 24.0% and 28.6%. That follow-up passed 5,308 installed cases per dispatch mode. On its measured x86-v4 host, the two float32 cases reached 0.902× and 0.848× upstream speed, and the two float64 cases reached 0.880× and 0.892×. The packed float follow-up remains private. [Installed follow-up](https://github.com/bipxx/scikit-imagexx/actions/runs/44235086774) · [Follow-up one-thread comparison](https://github.com/bipxx/scikit-imagexx/actions/runs/44236037020) [Installed tests](https://github.com/bipxx/scikit-imagexx/actions/runs/44229432874) · [One-thread comparison](https://github.com/bipxx/scikit-imagexx/actions/runs/44230822072) · [Private kernel receipt](../build/ci26-cie-lab-v2-timing-v1/handoff-v1.json)

## How to read and update this checklist

- An API checkbox closes only when every applicable gate below is proved for its full declared contract. An unchecked box means full closure has not been established here; it does not discard existing scoped work.
- Gate codes: `C` contract/domain inventory; `N` native/provider ownership; `H` hot swap; `P` public parity; `S` installed C++ SDK; `V` scalar/SIMD; `W` global workers; `B` one-thread original benchmark; `R` release/delivery.
- `?` means full-gate evidence has not been reconciled. `~` means a bounded/partial slice is credited. A failed domain must be named beside the row with its receipt; partial credit never overrides that failure. Replace a code with `done` only alongside a receipt, exact source, supported profiles and domain. Use `N/A` only with a recorded applicability reason.
- Test and benchmark links below are **locators**, not passing receipts. A registered default dtype does not prove all layouts, values, errors, provider branches or platforms.
- After each bounded task, update its existing ledger/handoff first, then update the affected API/member rows, active-increment table and evidence links here. Preserve failures and slower cases. Do not rerun science just to refresh this document or regenerate away reviewed marks.
- Root verifies the changed rows before merge/push/closeout. Workers identify affected keys and exact passed/pending/failed scopes in their existing handoff; root owns the cross-family checklist update.

## Shipping snapshot

### Latest verified checkpoint — 2026-11-26 (`opt-wave4-477`, 477/477 shipped)

This checkpoint and the linked terminal receipts take precedence over older dated progress notes below. Historical candidate descriptions remain evidence of their original scope, not current dispatch instructions. A passing diagnostic that does not reproduce an earlier failure leaves that failure unresolved.

Main `02555c92697e1c16477068529dbfd51de2ff363c` (tree `a378c72fcfa1e844bafbc66c8bf94300cfd352ce`, `opt-wave4-477`, fast-forwarded onto `c8b21e4d`) ships all `477/477` inventoried public APIs (`408` `native-partial`, `37` `packaged-asset-partial`, `27` `native-class-partial`, `3` `metadata`, `2` `exception-metadata-provider`, `0` `upstream-only`), integrates 11 retained parallel Codex CLI `gpt-6.2-sol` `medium` optimization scopes across 44 modified source/Python files (including `src/simd_avx2.cpp` MSVC `/arch:AVX2` kernels for `quickshift`, colocalization/metrics/components AVX2/direct fast paths, `thin`/`medial_axis`/`find_contours`, DAISY/RANSAC/moments, restoration/graph-rag/tomography/canny/peak_local_max; `opt4-w06` `felzenszwalb`/`slic`/`watershed` reverted cleanly to Wave 3 after benchmarking showed regressions), and syncs `v63`/`v54` source pins. On isolated 1-thread Windows Release comparison ([benchmark comparison](../build/evidence/u182-post-haar/opt-wave4-v1/benchmark-comparison.json)), slower APIs drop from `32` to `26` (`72 -> 43 -> 32 -> 26` across Baseline/Wave 2/Wave 3/Wave 4), lifting median speedup `3.183x -> 3.359x` (`2.387x` baseline) and geomean `3.48x -> 3.692x` (`2.557x` baseline). Dual toolchains passed after the `opt4-w06` revert (`121/121` WSL GCC 13 `-O3` CTest; `5,480 passed` across 29 affected public suites + `69/69` `w06` recovery checks in both normal & `SKIMAGEXX_DISABLE_SIMD=1` scalar modes on MSVC 2022 Release), and both complete 186-selector Tooling guard barriers (`4,918 passed, 8 subtests passed, 0 failed` on both Python 3.13.14 and Python 3.11.11) plus 3 standalone TV process-isolation checks per Python passed. Exact four-profile release qualification ([whole-product admission](../build/evidence/u182-post-haar/opt-wave4-v1/whole-product-root-admission-v1.json), `sha256: 1a3d2d0c434cc4208f0e8516f7bd40cf7546c698c77c9b650e11432f823abeb8`, [run 45788671362](https://github.com/bipxx/scikit-imagexx/actions/runs/45788671362)) passed all four hosted profiles (`modern` job `139801868012` / artifact `15105558696`, `minimum` `139801867217` / `15106402725`, `debug` `139801867821` / `15105519842`, `windows` `139801867841` / `15109445078`) with `0` failed commands (`113`/`113`/`113`/`80`), `896` installed SDK checks (`224` per profile across `112` families), `46,875` / `48,950` / `46,738` / `45,588` rehashed manifest records, and `archive_retained: false`. `D:\Scratch\opt-wave4` (13 worktrees + temporary build dirs + `work/opt*` branches) is retired (`D:` free `320,850,595,840 -> 360,122,306,560` bytes, `+39,271,710,720` bytes reclaimed).

### Historical checkpoint — 2026-11-25 (`opt-wave3-477`, 477/477 shipped)

Main `aceb39b526913ff124f908fe316b185034558276` (`opt-wave3-477`, fast-forwarded onto `15a291da`, preserving superseded failed debug-GIL run `45714742225` / `0898c137` resolved by hoisting `mutable_data()` and `itemsize()` above `py::gil_scoped_release` in `src/bind_components.cpp`) ships all `477/477` inventoried public APIs (`408` `native-partial`, `37` `packaged-asset-partial`, `27` `native-class-partial`, `3` `metadata`, `2` `exception-metadata-provider`, `0` `upstream-only`), integrates 12 parallel Codex CLI `gpt-6.2-sol` `medium` optimization scopes with `CMAKE_CXX_VISIBILITY_PRESET=hidden`, and reduces 1-thread slower APIs from `43` to `32` (`72 -> 32` from baseline), lifting median speedup `2.522x -> 3.183x` (`2.387x -> 3.183x`) and geomean `2.996x -> 3.48x` (`2.557x -> 3.48x`) ([benchmark comparison](../build/evidence/u182-post-haar/opt-wave3-v1/benchmark-comparison.json), measured on `c9c2203e` prior to restoring 3 non-Wave3-gain hard-pinned C++ files and hoisting `bind_components.cpp` GIL metadata). Exact four-profile release qualification ([whole-product admission](../build/evidence/u182-post-haar/opt-wave3-v1/whole-product-root-admission-v1.json), [run 45732125393](https://github.com/bipxx/scikit-imagexx/actions/runs/45732125393)) passed all four hosted profiles (`modern` `15079789435`, `minimum` `15082565442`, `debug` `15081369727`, `windows` `15082322107`) with `0` failed commands, `896` installed SDK checks (`224` per profile across `112` families), `46,873` / `48,950` / `46,738` / `45,589` rehashed manifest records, and `archive_retained: false`. Both complete 186-selector Tooling guard barriers (`4,918 passed, 8 subtests passed, 0 failed/skipped` on Python 3.13.14 and 3.11.11) and standalone TV process-isolation checks passed.

### Historical checkpoint — 2026-11-25 (`opt-wave2-477`, 477/477 shipped)

Main `15a291dad7de274e2db6462a8bf862951a32ddd8` (`opt-wave2-477`, fast-forwarded onto `c4135c8d`) ships all `477/477` inventoried public APIs (`408` `native-partial`, `37` `packaged-asset-partial`, `27` `native-class-partial`, `3` `metadata`, `2` `exception-metadata-provider`, `0` `upstream-only`), removes 21 unreferenced `benchmarks/*.zip` archives (`-16.08 MiB`), and lands 11 optimization scopes reducing 1-thread slower APIs from `72` to `43`, lifting median speedup `2.387x -> 2.522x` and geomean `2.557x -> 2.996x` ([benchmark comparison](../build/evidence/u182-post-haar/opt-wave2-v1/benchmark-comparison.json)). Exact four-profile release qualification ([whole-product admission](../build/evidence/u182-post-haar/opt-wave2-v1/whole-product-root-admission-v1.json), [run 45644063673](https://github.com/bipxx/scikit-imagexx/actions/runs/45644063673)) passed all four hosted profiles (`modern` `15040224100`, `minimum` `15041374346`, `debug` `15042509785`, `windows` `15045015863`) with `0` failed commands, `896` installed SDK checks (`224` per profile across `112` families), `46,871` / `48,946` / `46,738` / `45,589` rehashed manifest records, and `archive_retained: false`.

### Historical checkpoint — 2026-11-25 (`current408`, 477/477 baseline closure)

Main `c4135c8d38603c1fc51f3d100431a754f9d6196f` (`current408`) ships **all `477/477` inventoried public APIs**: **`408` `native-partial`**, **`37` `packaged-asset-partial`**, **`27` `native-class-partial`**, **`3` `metadata`**, **`2` `exception-metadata-provider`**, and **`0` `upstream-only`** (across `448` callables, `29` classes, and `284` static member paths). Exact four-profile release qualification ([whole-product admission](../build/evidence/u182-post-haar/current408-qualification-v1/whole-product-root-admission-v1.json), [run 45609669785](https://github.com/bipxx/scikit-imagexx/actions/runs/45609669785)) passed all four hosted profiles (`modern`, `minimum`, `debug`, `windows`) with `0` failed commands, **`896` installed SDK checks** (`224` per profile across `112` families), `46,869` / `48,944` / `46,738` / `45,588` rehashed manifest records, and `archive_retained: false`. Prior historical failed runs (`45557858466`, `45582555940`, `45596316544`) remain preserved in `build/evidence/u182-post-haar/current408-qualification-v1/`.

### Historical checkpoint — 2026-11-24 (`current312`)

Main `1d9a0bbf` (`current312`) ships bounded native `transform.hough_circle_peaks`, `transform.hough_ellipse`, `transform.probabilistic_hough_line`, and `morphology.medial_axis`, bringing shipping inventory to **312 `native-partial`**, **10 `packaged-asset-partial`**, **5 `native-class-partial`**, **3 `metadata`**, **1 `exception-metadata-provider`**, and **146 `upstream-only`** APIs (closing 36 of the original 182 upstream-only gaps). Exact four-profile release qualification ([whole-product admission](../build/evidence/u182-post-haar/current312-qualification-v1/whole-product-root-admission-v1.json), [run 45344078252](https://github.com/bipxx/scikit-imagexx/actions/runs/45344078252)) passed **1,008,468 public outcomes** and **488 installed SDK checks** (`122` per profile across `61` targets) plus 3 packaging stagers (`archive_retained: false`). Dual-Python exact-source barrier ([source readback](../build/evidence/u182-post-haar/current312-source-barrier-v4/complete-312-intake-readback-v4.json), [run 45340542569](https://github.com/bipxx/scikit-imagexx/actions/runs/45340542569)) passed `4824/4824` on `windows311` (`3.11.11`) and `windows313` (`3.13.14`). Hosted Windows focused parity/SDK and 16-case 1-worker timing ([focus & timing admission](../build/evidence/u182-post-haar/current312-windows-focus-v1/root-focus-timing-admission-v1.json), [run 45340574164](https://github.com/bipxx/scikit-imagexx/actions/runs/45340574164)) passed `150/150` public normal & scalar, `4/4` SDK normal & scalar, and **16/16** timing cases at **1.709×–157.942×** (`losses: []`, `pre/post CPU: 7%/2%`), resolving the earlier Candidate 3 (`e87df809`) `0.933×` loss on `dense-default-bands` (`45336457806`) to `8.171×` (`0.8261ms`); `medial_axis` modern quiet speed (`45110288114`) is separately admitted at `6.704×–14.550×`.

### Historical checkpoint — 2026-11-22

Prior main `94522169` shipped bounded `pyramid_reduce` and `pyramid_expand` plus packaged `camera` and `coffee`. That release left 162 upstream-only APIs. Exact release passed four profiles and three relocated products with 1,001,988 public outcomes and 376 SDK checks. [Verified main delivery](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/root-main-delivery-v1.json) · [Whole-product evidence](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/whole-product-root-admission-v1.json) · [Five-API qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45079540984)

Current301 Linux modern passed independent readback with 252,106 public checks, 100 SDK checks and a verified relocated artifact. Windows also passed independent diagnostic readback with 248,446 public checks, 100 SDK checks and a verified relocated artifact. Debug passed 252,106 public and 100 SDK checks in its diagnostic profile. [Debug profile](../build/evidence/u182-post-haar/hosted-current301-45079540984/debug/root-readback.json) [Windows profile](../build/evidence/u182-post-haar/hosted-current301-45079540984/windows/root-readback.json) · [Windows relocated product](../build/evidence/u182-post-haar/hosted-current301-45079540984/stager-windows/root-readback.json) [Modern profile](../build/evidence/u182-post-haar/hosted-current301-45079540984/modern/root-readback.json) · [Relocated product](../build/evidence/u182-post-haar/hosted-current301-45079540984/stager-modern/root-readback.json)

Current301 Linux minimum failed the same three `pyramid_laplacian` first-yield cases in normal and scalar modes. Root retained the six failed XML nodes. Promotion is held while the live comparison against genuine Linux minimum stock is prepared. Later source runs continue, and the later installed build waits for the affected dependency scope. [Minimum failure](../build/evidence/u182-post-haar/hosted-current301-45079540984/minimum-failed-v1/root-minimum-failure-admission-v1.json)

The hosted comparison reproduced the three first-yield failure hashes in both modes. Genuine minimum stock matched the exact installed candidate across 12 yields and 30 intermediate stages, including the RGB later yields. The next correction touches only the generator test and keeps all assertions outside the host-sensitive bytes. Root requires focused installed readback, both full source runs, and fresh four-profile qualification before release. [Live comparison](../build/evidence/u182-post-haar/current301-minimum-laplacian-live-v1/actual-result-v1/comparison-v1.json)

Corrected current301 `a80be034` shipped bounded `pyramid_gaussian`, `pyramid_laplacian`, `find_contours`, `random_noise` and `random_shapes` to `main`. This closes 25 of the original 182 gaps and leaves **157 upstream-only APIs**. All four exact profiles and three relocated artifacts passed independent readback with 1,004,764 public checks and 400 SDK checks. The three earlier Laplacian first-yield failures passed in both restored minimum modes. [Main delivery](../build/evidence/u182-post-haar/hosted-current301-45090228543/root-main-delivery-v1.json) [Whole-product admission](../build/evidence/u182-post-haar/hosted-current301-45090228543/whole-product-root-admission-v1.json) [Source proof](../build/evidence/u182-post-haar/current301-minimum-laplacian-fix-source-v1/actual-result-v1/complete-source-readback-v1.json) · [Fresh release](https://github.com/bipxx/scikit-imagexx/actions/runs/45090228543) [Independent installed result](../build/evidence/u182-post-haar/current301-minimum-laplacian-fix-focus-v1/actual-result-v1/comparison-v1.json) · [Complete source run](https://github.com/bipxx/scikit-imagexx/actions/runs/45087596920)

Before the corrected release, main had 166 upstream-only APIs. The original `7f8d4c5a` was the preceding candidate. It passed 468 focused installed checks and all 12 fixed one-worker public comparisons. Both full source runs exposed five matching guard assertions. Source-only correction `136b0d16` passed all 17 affected checks on each retained Windows Python. Both complete commands passed 4,781 ordinary tests and eight subtests per Python. Four-profile qualification is live for `136b0d16`. [Source admission](../build/evidence/u182-post-haar/complete-296-root-admission-hosted-v2.json) · [Qualification run](https://github.com/bipxx/scikit-imagexx/actions/runs/45058138452) [Corrective checks](../build/evidence/u182-post-haar/current296-source-correction-root-admission-v1.json) [Source failure](../build/evidence/u182-post-haar/complete-296-source-failure-root-v1.json) [Installed parity](../build/evidence/u182-post-haar/installed-focused-root-admission-v1.json) · [Installed speed](../build/evidence/u182-post-haar/installed-six-timing-root-admission-v1.json)

The next five-API draft covers both pyramid generators, contour tracing, random noise and random shapes. Its family source matches the privately timed implementations for 16 fixed Linux modern calls at 1.090×–4.528× original speed. The draft projects 157 upstream-only APIs after delivery. Shared installed parity, SDK, speed and release checks remain pending. [Private timing source match](../build/evidence/u182-post-haar/following-private-timing-source-match-v1.json)

The Linux minimum release profile failed one float64 Gaussian stage of pyramid_expand in normal and forced-scalar mode. Linux modern passed. Main had 166 upstream-only APIs at that checkpoint. The pairwise original-versus-candidate diagnostic is pending alongside the frozen five-API draft's complete source checks. [Minimum failure readback](../build/evidence/u182-post-haar/hosted-45058138452/failed-range-minimum-v1/root-failure-readback.json)

A four-node Intel diagnostic reproduced the float64 bytes and proved exact agreement with live original scikit-image. The same pinned runtime on AMD returned the original frozen bytes for both implementations. The corrected source keeps the frozen fixture and requires exact same-host parity for the affected minimum case. The four-API batch remains held while the corrected source and installed gates run, and the next five-API batch has recovered its 65 affected source checks on both Windows Python versions. [Intel proof](../build/evidence/u182-post-haar/pyramid-hosted-minimum-prefix-v1/prefix-result-v1/root-prefix-causal-admission-v1.json) · [Source run](https://github.com/bipxx/scikit-imagexx/actions/runs/45068075597)

The corrected four-API increment passed 80 installed Intel checks and both complete source runs. Its corrected qualification passed and shipped. The next five-API revision passed complete source on both retained Pythons, 678 focused Windows installed tests and six Windows relocated SDK checks. All 16 installed Windows fixed one-worker comparisons passed at 1.240×–3.276×. Its corrected successor shipped after complete qualification. [Installed timing](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json) [Current296 qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45070502340) · [Current301 exact source](https://github.com/bipxx/scikit-imagexx/actions/runs/45068719061) · [Windows installed](../build/evidence/u182-post-haar/following-windows-installed-focus-v1/root-installed-focus-admission-v1.json) · [Windows SDK](../build/evidence/u182-post-haar/following-windows-sdk-focus-v1/root-sdk-focus-admission-v1.json)

The next private five-API projection covers `feature.match_descriptors`, `morphology.thin`, bounded 2D Zhang `morphology.skeletonize`, `transform.hough_circle` and `transform.hough_line_peaks`. Source attachment is verified. Shared integration, exact installed parity and SDK, public speed, and release qualification remain pending. Explicit Lee and default 3D `skeletonize` remain on the compatibility route. [Source projection](../build/evidence/u182-post-haar/next-five-leaf-projection-v1.json)

Source-only integration for the five private APIs now has verified bindings, exports and relocated SDK test targets. The compilation and actual installed SDK runs are pending. The `resize_local_mean` axis-order replay refutes reversed order on every fixed normalized case. Five genuine vertical intermediate hashes were reconstructed per profile, before the remaining capture below. [Wiring readback](../build/evidence/u182-post-haar/next-five-shared-wiring-v1/root-admission-v1.json) · [Resize witness](../build/evidence/u182-post-haar/resize-vertical-witness-v1/handoff.json)

The remaining normalized `resize_local_mean` stock captures passed on both retained profiles. The vertical model and genuine horizontal provider matched the 24 new cases per profile exactly. The proposed normalized hybrid remains held until four/eight-term rounding under the supported providers is proved. Component parity, fixed speed and installed qualification remain open. [Capture receipt](../build/evidence/u182-post-haar/resize-vertical-witness-v2/handoff.json) [Arithmetic hold](../build/evidence/u182-post-haar/resize-hybrid-source-v1/handoff.json)

The new four stock discriminator observations selected left-fold results over adjacent-pair grouping in the tested resize placements. The supported-domain proof remains open. Both exact Moravec runtime probes confirmed no supported constructor for the stock exporter. [Resize rounding witness](../build/evidence/u182-post-haar/resize-provider-association-v2/handoff.json) · [Moravec exporter witness](../build/evidence/u182-post-haar/moravec-exporter-capability-v1/handoff-v1.json)

The four-file private registry and generic parity patch now passes independent source checks. Installed testing, benchmark registration and qualification still remain open. Exact Moravec Linux-minimum private source `ff044f66` passed all four named single-worker comparisons at 1.382×–2.672× with 56 accepted samples. The first runner stopped before timing because its isolated environment lacked the provider inventory helper. The successor reused the retained helper. Moravec still has an observable memoryview exporter difference and needs Windows and modern speed and installed qualification. [Registry readback](../build/evidence/u182-post-haar/next-five-registry-generic-v1/root-admission-v1.json) · [Moravec timing](../build/evidence/u182-post-haar/corner-moravec-minimum-speed-v2/root-timing-admission-v1.json)

Private source projection now includes all five bounded routes in inventory and generic benchmark dispatch, with out-of-domain cases unassessed. The projected upstream-only count is 152 after that later batch ships. Main is at 162. [Inventory and benchmark source](../build/evidence/u182-post-haar/next-five-inventory-benchmark-v1/root-admission-v1.json) · [Generated inventory README](../build/evidence/u182-post-haar/next-five-inventory-benchmark-v1/root-generated-readme-admission-v1.json)

The source qualifier independently projected all four profiles with 282 installed files, 125 SDK files and 55 clients checked in both modes. Complete source commands, actual installed parity and SDK, exclusive public timings and release remain pending. [Qualifier readback](../build/evidence/u182-post-haar/next-five-qualification-wiring-v1/root-admission-v1.json)

Nine source-guard modules now match the projected qualifier. The existing workflow remains byte-identical and includes all nine among its 186 exact selectors. Actual source runs are still pending. [Source guard readback](../build/evidence/u182-post-haar/next-five-source-guards-v1/root-admission-v1.json)

The active history and raw dependency pins for the later five-API batch passed independent source verification. The projected tree is `44937060`. The workflow and its 186 exact selectors are unchanged. Both complete source runs, actual installed parity and speed, and current301 main delivery still gate qualification. [History readback](../build/evidence/u182-post-haar/next-five-history-pins-v1/root-admission-v1.json)

Exact current306 `0b33a90c` ran the 186-selector source command on Windows Python 3.11.11 and 3.13.14 after passing capacity checks. Each run ended with 4,803 tests and eight subtests passing and the same metadata test failing. The test omitted the thinning function set from its isolated benchmark namespace. Root holds installed work until a successor passes both complete source commands. The prepared timing runner covers 20 fixed cases and still requires the genuine Windows product. [Failed source receipt](../build/evidence/u182-post-haar/next-five-source-carrier-v1/complete-306-root-failure-v1.json) · [Repaired-source timing runner](../build/evidence/u182-post-haar/next-five-installed-speed-v2/handoff-v1.json)

Corrected source `1fdc51e3` passed the exact failed test on both retained Windows versions. It loads the same registry as the qualifier. The native code and benchmark recipes are unchanged. Both complete runs then exposed a history test that reused its v4 variable for v5. [Repair evidence](../build/evidence/u182-post-haar/next-five-source-fix-v1/root-repair-admission-v1.json) · [Complete source rerun](https://github.com/bipxx/scikit-imagexx/actions/runs/45085004799)

Rebased source `5b94b20d` inherits the corrected Laplacian test and repairs the history guard. Both complete Windows source runs passed 4,804 tests and eight subtests. The genuine Windows product passed 436 installed public checks and ten relocated SDK checks. The first 20-case timing job stopped because its shallow checkout omitted the exact parent commit. The next job reached the exact source check and stopped on a raw versus canonical line-ending comparison. Both attempts stopped before taking samples. The fixed source binding passed on all 53 paths on both retained Windows versions. The exact hosted Windows product then passed all 20 one-worker comparisons at 1.375×–99.657× against stock over 280 accepted batches. Exact source `5b94b20d` shipped to `main` after independent readback of four qualifying profiles and three relocated artifacts. The release passed 1,006,572 public checks and 440 SDK checks. It adds bounded native `match_descriptors`, `thin`, 2D Zhang `skeletonize`, `hough_circle` and `hough_line_peaks`. **152 upstream-only APIs remain**, closing 30 of the original 182 gaps. [Exact delivery](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/root-main-delivery-v1.json) [Whole-product admission](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/whole-product-root-admission-v1.json) [Release run](https://github.com/bipxx/scikit-imagexx/actions/runs/45098117810) [Timing admission](../build/evidence/u182-post-haar/next-five-hosted-speed-v4/actual-result-v1/root-timing-admission-v1.json) [Second setup failure](../build/evidence/u182-post-haar/next-five-hosted-speed-v3/actual-result-v1/comparison-v1.json) [Setup failure](../build/evidence/u182-post-haar/next-five-hosted-speed-v2/actual-result-v1/comparison-v1.json) · [Passing timing job](https://github.com/bipxx/scikit-imagexx/actions/runs/45094010741) [Installed and SDK evidence](../build/evidence/u182-post-haar/next-five-hosted-product-v4/actual-result-v1/root-focus-admission-v1.json) [Second failure](../build/evidence/u182-post-haar/next-five-source-carrier-v2/complete-306-root-failure-v2.json) · [Rebase](../build/evidence/u182-post-haar/next-five-source-fix-v2/root-repair-rebase-admission-v1.json) · [Complete source proof](../build/evidence/u182-post-haar/next-five-source-carrier-v3/complete-306-root-admission-v3.json) · [Windows installed job](https://github.com/bipxx/scikit-imagexx/actions/runs/45090847657)

Automatic approval blocked retirement of `D:/Scratch/u182-294-src-v2/h` before execution. The 8.88 GB logical fixture remains present. Both local volumes are below the Windows build floor. Root will use the isolated hosted Windows product route when both full source checks pass. [Policy outcome](../build/evidence/u182-post-haar/next-five-capacity-options-v1/root-haar-h-retirement-blocked-v1.json) · [Hosted product plan](../build/evidence/u182-post-haar/next-five-hosted-product-v1/hosted-windows-product-plan-v1.json) · [Prepared Windows job](../build/evidence/u182-post-haar/next-five-hosted-product-v3/author-handoff-v1.json)

Modern Linux stock `convex_hull_object` passed all 18 frozen cases plus one controlled Qhull failure. Public outputs, layout, diagnostics, and provider observations matched Windows and Linux minimum. Modern native parity and speed remain pending. [Stock evidence](../build/evidence/u182-post-haar/lookahead-object-hull-modern-v1/handoff.json)

Exact private bilateral successor `2dcf3154` passed all six Windows one-worker cases at 1.418×–1.901× with 84 accepted samples. All six Linux minimum cases passed separately. Modern compiled parity and speed, shared installed evidence and release remain open. [Windows raw timing admission](../build/evidence/u182-post-haar/lookahead-bilateral-windows-speed-v1/root-timing-admission-v1.json)

Three future Hough leaves add 30 disjoint files against current306. Windows and Linux minimum passed 24 fixed private pairs and six standalone SDK checks. The narrowest speed margin was 1.011× for the dense Windows probabilistic case. All 55 modern stock observations matched the prior profiles. The exact modern direct-build entry is prepared and held at the local capacity gate. Modern native parity and speed, shared installed checks, and release remain open. [Modern build entry](../build/evidence/u182-post-haar/lookahead-hough-modern-components-v1/handoff-v1.json) · [Hough leaf and evidence](../build/evidence/u182-post-haar/lookahead-hough-leaf-admission-v1/handoff-v1.json)

Private retained Linux-modern `medial_axis` matched 27 stock outcomes against the Windows reference and passed all 30 normal-mode checks and one direct SDK client. Two Linux-minimum compatibility differences are recorded. Forced-scalar and one-worker timing remain pending. Windows `regionprops_table` and Linux-modern `hough_circle_peaks` passed input authentication and await free space to run. [Medial-axis readback](../build/evidence/u182-post-haar/lookahead-morphology-v1/medial-modern-v1/readback-v1.json) [Hosted scalar and fixed timing entry](../build/evidence/u182-post-haar/lookahead-morphology-v1/medial-modern-hosted-prep-v1/handoff-v1.md) · [Regionprops Windows entry](../build/evidence/u182-post-haar/lookahead-measure-hull-v1/region-windows-v1/handoff.json) · [Circle-peaks modern entry](../build/evidence/u182-post-haar/lookahead-hough-v1/circle-peaks-modern-stock-v1/readiness-v1.json)

The hosted Linux modern circle-peaks stock capture retained all 29 cases. It differs from Windows on the default real case and from Linux minimum on five cases, including all four fixed timings. The exact private modern candidate passed 29 checks per mode, including 19 native calls and ten controlled compatibility or validation routes. The standalone GNU SDK passed. After correcting a duplicate witness argument in the timing harness, all four modern fixed pairs passed at 8.688×–114.437× on the same tested ELF. Installed parity and release remain pending. [Fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-v1/circle-peaks-modern-timing-v1/successor-v2/actual-result-v1/root-circle-peaks-speed-admission-v1.json) [Private component admission](../build/evidence/u182-post-haar/lookahead-hough-v1/circle-peaks-modern-component-v1/carrier-v2/actual-result-v1/root-private-component-admission-v1.json) [Stock admission](../build/evidence/u182-post-haar/lookahead-hough-v1/circle-peaks-modern-hosted-stock-v1/carrier-v1/actual-result-v1/root-modern-stock-admission-v1.json)

Private Linux modern Hough line, ellipse and probabilistic line passed 110 normal/scalar checks across matching source and retained direct SDK evidence. The modern fixed comparison matched outputs in all twelve pairs and passed speed in eleven. Dense Hough line measured 0.99859×. Ellipse and probabilistic passed all of their four cases. Root holds the line and retains separate shared installed and release gates. [Fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-modern-speed-timing-carrier-v1/actual-result-v1/root-fixed-speed-admission-v1.json) [Modern private Hough admission](../build/evidence/u182-post-haar/lookahead-hough-hosted-carrier-v4/actual-result-v1/root-modern-hough-admission-v1.json)

Private Linux modern `convex_hull_image` and `convex_hull_object` each passed 36 normal and scalar checks and reused both direct GNU SDK results. All eight fixed private one-worker cases passed, image at 1.411×–23.026× and object at 3.130×–180.171× over 112 accepted batches. The exact shared hull draft failed the same 28 source guards on both supported Pythons, involving a missing AST helper dependency, a stale qualification filename and a metadata envelope collision. The corrected 55 affected checks passed on both Pythons, including all 28 former failures. Both complete 186-selector commands passed 4,808 ordinary tests and eight subtests, with no errors or skips. The scoped Windows installed check passed in normal and forced scalar modes, with 61 ordinary tests, 18 object-hull subtests and two relocated SDK checks per mode. All eight fixed Windows one-worker comparisons passed at 1.722x-366.394x against scikit-image 0.26.0. The exact four-profile qualification failed in the shared fallback test in both modes. The existing test expects the now-native contiguous Boolean hull route to refuse. The correction uses an unsupported strided input. The affected source guards and four installed Windows regression checks passed. Both complete source jobs passed 4,808 ordinary tests and eight subtests on the corrected source. Corrected source `b1d67cc0` shipped on `main` after independent verification of all four profiles and three relocated products. The release passed 1,007,220 public checks and 456 SDK checks. This closes 32 of the original 182 upstream-only gaps and leaves 150. [Exact delivery](../build/evidence/u182-post-haar/next-hull-release-reader-v1/hosted-current308-45219397159/root-main-delivery-v1.json) [Independent qualification](../build/evidence/u182-post-haar/next-hull-release-reader-v1/hosted-current308-45219397159/whole-product-root-admission-v1.json) [Dual source run](https://github.com/bipxx/scikit-imagexx/actions/runs/45214940795) [Corrected release](https://github.com/bipxx/scikit-imagexx/actions/runs/45219397159) [Failure admission](../build/evidence/u182-post-haar/next-hull-release-reader-v1/hosted-45110532095/root-failed-release-admission-v1.json) [Raw timing admission](../build/evidence/u182-post-haar/next-hull-windows-focus-v1/timing-carrier-v1/actual-result-v1/root-timing-admission-v1.json) [Failed whole-product qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45110532095) [Installed admission](../build/evidence/u182-post-haar/next-hull-windows-focus-v1/actual-result-v1/root-focus-admission-v1.json) [Windows installed job](https://github.com/bipxx/scikit-imagexx/actions/runs/45108014478) [Complete source readback](../build/evidence/u182-post-haar/next-hull-source-fix-v1/full-carrier-v2/actual-result-v2/complete-308-intake-readback-v2.json) [Affected passing checks](../build/evidence/u182-post-haar/next-hull-source-fix-v1/affected-actual-result-v2/passed-308-affected-readback-v2.json) [Complete rerun](https://github.com/bipxx/scikit-imagexx/actions/runs/45104615926) [Complete failure evidence](../build/evidence/u182-post-haar/next-hull-source-carrier-v1/actual-result-v1/failed-308-readback-v1.json) [Image timing](../build/evidence/u182-next-numerical-lookahead/convex-hull/modern-timing-v1/root-image-timing-admission-v1.json) [Object timing](../build/evidence/u182-next-numerical-lookahead/convex-hull/modern-timing-v1/root-object-timing-admission-v1.json) [Component proof](../build/evidence/u182-next-numerical-lookahead/convex-hull/modern-components-v2/root-private-component-admission-v1.json)

Main `817b6f82` includes **294 partial native image API paths**, **eight packaged-asset routes**, and **166 upstream-only public keys**. The checklist covers **477 public keys, 29 classes and 284 static-member paths**. Sixteen keys have moved out of upstream-only status since the 182-key checkpoint. The latest release adds bounded Haar drawing, coordinates, and values. All four hosted profiles and three stagers verified, totaling 1,001,068 installed outcomes and 376 SDK checks. All ten measured installed minimum cases met one-worker parity at 1.026×–108.450×. [Main delivery](../build/evidence/u182-haar-features/hosted-45041856008/root-main-delivery-v1.json) · [Whole-product admission](../build/evidence/u182-haar-features/hosted-45041856008/whole-product-root-admission-v1.json) · [Installed speed](../build/evidence/u182-haar-features/haar-294-installed-speed-root-admission-v2.json)

The delivered revision passed 972,900 public outcomes across four profiles, 360 SDK checks and all three staging checks. All 29 fixed one-worker calls on the installed minimum profile met speed parity at 2.212×–216.266×. The earlier failed campaign and its 260 failures remain archived as diagnostic receipts. Automatic approval blocked retirement of the obsolete fixtures, with no reclaimed bytes. [Whole-product admission](../build/evidence/u182-first-integration/hosted-45004954877/whole-product-root-admission-v1.json) · [One-worker admission](../build/evidence/u182-first-integration/installed-speed-root-admission-v1.json) · [Failed campaign](../build/evidence/u182-first-integration/hosted-44997239007/four-profile-failed-root-intake-v1.json)

The new exact main revision passed **997,948 public outcomes, 360 SDK checks and three staged-package checks** in four hosted profiles. Both complete source commands passed 4,771 tests and eight subtests each. The eight fixed installed minimum `binary_blobs` comparisons passed at 1.886×–3.102× with exact output and 112 accepted one-worker samples. Sixteen fixed installed packaged-accessor comparisons passed at 7.458×–86.199× across Windows and WSL minimum with 224 accepted samples. [Qualification](../build/evidence/u182-next-data-integration/hosted-45020725998/whole-product-root-admission-v1.json) · [Complete source](../build/evidence/u182-next-data-integration/complete-291-root-admission-v3.json) · [Binary blobs performance](../build/evidence/u182-next-data-integration/binary-installed-timing/root-installed-speed-admission-v1.json) · [Packaged accessors performance](../build/evidence/u182-next-data-integration/installed-asset-speed-root-admission-v2.json)

The separate Haar draft `e4c45f8d` attaches `feature.draw_haar_like_feature`, `feature.haar_like_feature`, and `feature.haar_like_feature_coord`. Its 294 partial native paths and eight packaged routes would leave 166 upstream-only keys. The draft passed independent source admission. The complete commands, installed parity and SDK, fixed installed speed and hosted qualification remain open. [Haar source admission](../build/evidence/u182-haar-features/haar-294-root-source-admission-v1.json)

Haar release gates remain open. Run 45011599601 failed all four profiles on shared registry assertions. Minimum public binary blobs outputs passed before six Gaussian-stage digest failures per mode. All six exact old-wheel stages matched the retained local stock. The exact corrective child passed 6,508 installed checks in each of the four profile and mode combinations. Both complete 184-selector source checks passed with 4,771 tests and eight subtests each. The corrected data candidate passed all four hosted profiles in run 45020725998 and shipped. [Source admission](../build/evidence/u182-next-data-integration/complete-291-root-admission-v3.json) · [Hosted qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45020725998) All sixteen fixed warm packaged-accessor comparisons passed across Windows and WSL minimum, with 224 accepted one-worker samples. Windows gains were 11.647×–86.199× and WSL minimum gains were 7.458×–11.125×. The Haar draft passed 5,868 staged public checks per profile and mode and eight focused SDK checks in aggregate. Its complete source checks failed matching metadata guards on both interpreters. The first corrective source passed 1,039 affected tests per interpreter. Its exact data attachment `384cd104` passed another 64 affected checks per interpreter. The final attachment also passed 6,612 installed checks in each of four local combinations. The two complete 186-selector commands ended with 4,777 tests and eight subtests passing per interpreter. Both failed the same three fixture and collector guards for the new active history. Exact source `817b6f82` passed all 166 affected checks on each Python version and preserves the public workflow, production code, SDK and Haar timing inputs. Its full exact rerun requires 45.1 GB free on D:. The agent cleanup was blocked and manual capacity remains pending. [Scoped correction admission](../build/evidence/u182-haar-features/haar-294-fixture-collector-root-admission-v1.json) [Complete failure evidence](../build/evidence/u182-haar-features/complete-294-failed-root-admission-v2.json) [Minimum diagnosis](../build/evidence/u182-next-data-integration/hosted-45011599601/minimum-failure-root-admission-v1.json) · [Data correction](../build/evidence/u182-next-data-integration/data-291-corrective-root-source-admission-v1.json) · [Asset speed](../build/evidence/u182-next-data-integration/installed-asset-speed-root-admission-v2.json) · [Haar staged results](../build/evidence/u182-haar-features/installed-294-root-admission-v1.json) · [Haar source correction](../build/evidence/u182-haar-features/haar-294-root-corrective-source-admission-v1.json)

Private `transform.hough_line_peaks` passed 104 checks and four SDK runs. Private `transform.hough_circle` passed 88 checks and four SDK runs. Private `morphology.thin` passed 84 checks and both SDK clients. The private one-worker comparisons for match descriptors, Hough line peaks and Hough circle passed all 24 cases with exact output and 336 accepted samples. Their speedup ranges were 4.96×–13.08×, 13.41×–53.57× and 2.29×–3.78×. The initial thinning entry stopped before any calls because it reused a compile-space assertion. The corrected read-only entry passed all eight fixed private comparisons across Windows and WSL at 8.07×–106.98×, with exact outputs and 112 accepted one-worker samples. Installed parity and release remain open. [Thinning timing admission](../build/evidence/u182-next-numerical-lookahead/thin/timing-v4/timing-root-admission-v4.json) [Timing evidence](../build/evidence/u182-next-numerical-lookahead/private-timing-window-root-admission-v1.json) Circle peaks still needs a native route that preserves profile-specific ties and its stock output contracts. FAST and corner orientations expose the original Cython memoryview exporter, which leaves their exact ownership gate open. [Hough line peaks](../build/evidence/u182-hough-line-peaks/private-component-root-admission-v1.json) · [Hough circle](../build/evidence/u182-hough-circle/private-component-root-admission-v1.json) · [Thin](../build/evidence/u182-next-numerical-lookahead/thin/private-component-root-admission-v1.json) · [FAST](../build/evidence/u182-next-numerical-lookahead/corner-fast/handoff-v1.json) · [Orientation](../build/evidence/u182-next-numerical-lookahead/corner-orientations/handoff-v1.json)

Three sparse private implementations have scoped verification. Planar `convex_hull_image` passed 72 checks and two standalone SDK clients. `find_contours` passed 87 checks in each of four runs across two Linux runtimes and one standalone SDK client. `hough_ellipse` matched 48 reference rows, passed eight strict-fallback refusals, and passed both standalone SDK clients. All fixed private one-worker cases passed for planar hull at 1.40×–156.46× and contour tracing at 3.27×–6.41×, with exact outputs and 112 accepted samples per family. Hough ellipse also passed all eight fixed Windows and WSL comparisons at 15.21×–23.04× with exact outputs and 112 accepted samples. [Ellipse timing](../build/evidence/u182-hough-ellipse/timing-v1/timing-root-admission-v2.json) Installed release gates remain open. [Planar hull timing](../build/evidence/u182-next-numerical-lookahead/convex-hull/image-timing-v1/timing-root-admission-v1.json) · [Contour timing](../build/evidence/u182-next-numerical-lookahead/find-contours/timing-v1/timing-root-admission-v1.json) [Planar hull](../build/evidence/u182-next-numerical-lookahead/convex-hull/image-root-admission-v1.json) · [Contour tracing](../build/evidence/u182-next-numerical-lookahead/find-contours/native-root-admission-v1.json) · [Ellipse transform](../build/evidence/u182-hough-ellipse/private-component-root-admission-v1.json)

Rectangle and polygon perimeter reference coverage now includes 22 calls in two existing Windows Python 3.12 environments with Matplotlib. Both produced the same five successful coordinate rows and six error controls. The installed ordered line kernel reproduced all five captured coordinate rows from the exact AGG vertices in both Windows and WSL, in normal and forced-scalar modes. [Installed edge evidence](../build/evidence/u182-next-numerical-lookahead/draw-perimeters/verified-installed-perimeter-edge-reuse-v1.json) Paired native perimeter components passed 44 checks across two provider-equipped Windows environments and two provider-absent WSL modes. Both standalone SDK clients passed. The paired perimeter private one-worker cases all passed at 3.98×–5.76× across the two Matplotlib-equipped environments with exact outputs and 112 accepted samples. Shared wiring, installed provider-capable parity, installed speed and release remain open. [Perimeter timing](../build/evidence/u182-next-numerical-lookahead/draw-perimeters/timing-v1/timing-root-admission-v1.json) [Private component admission](../build/evidence/u182-next-numerical-lookahead/draw-perimeters/native-root-admission-v1.json) [Provider-present evidence](../build/evidence/u182-next-numerical-lookahead/draw-perimeters/verified-provider-positive-v1.json) · [Provider-absent evidence](../build/evidence/u182-next-numerical-lookahead/draw-perimeters/handoff-v1.json)

Three more private components passed scoped Windows and WSL checks in normal and forced-scalar modes. Probabilistic Hough line matched 68 ordered native outputs, including all 20 checked PCG64 states. Bounded 2D Zhang skeletonize passed 92 checks. Convex hull objects passed 72 checks, including 40 compiled returns, and depend on the admitted planar hull leaf. All six standalone SDK clients passed. Lee skeletonization and exceptional Qhull behavior continue through compatibility. Fixed one-worker comparisons, shared installed validation and exact release remain pending. [Probabilistic Hough](../build/evidence/u182-probabilistic-hough-line/private-component-root-admission-v1.json) · [Zhang skeletonize](../build/evidence/u182-next-numerical-lookahead/skeletonize/native-root-admission-zhang-v1.json) · [Convex hull objects](../build/evidence/u182-next-numerical-lookahead/convex-hull/object-native-root-admission-v1.json)

The fixed one-worker comparisons produced 20 parity-speed passes out of 24, with exact outputs throughout. All eight probabilistic Hough line cases passed at 1.01×–2.01× and all eight convex hull object cases passed at 3.23×–354.34×. Zhang skeletonize passed all four WSL cases at 1.07×–2.13× and fell below parity in all four Windows cases at 0.54×–0.93×. Hold Zhang from integration and measure where time goes in the Windows frontend, conversion and traversal before changing the kernel. Exact shared installed validation and qualification remain open for the private cohort. [Probabilistic Hough timing](../build/evidence/u182-probabilistic-hough-line/timing-v1/timing-root-admission-v2.json) · [Convex hull object timing](../build/evidence/u182-next-numerical-lookahead/convex-hull/object-timing-v1/timing-root-admission-v1.json) · [Zhang timing](../build/evidence/u182-next-numerical-lookahead/skeletonize/timing-zhang-v1/timing-root-admission-zhang-v1.json)

The two-case Windows Zhang diagnostic matched exact outputs and recorded 28 one-worker samples. The compiled comparison still lost at 0.560× and 0.870× with both public wrappers bypassed. The deficit remains in compiled validation, allocation, copy or traversal. The first diagnostic stopped before timing because `functools.partial` bypassed a profiler event; the corrected witness verified the native return and preserved both timed calls. The earlier four fixed Windows losses remain the release gate. [Windows direct diagnostic](../build/evidence/u182-next-numerical-lookahead/skeletonize/windows-phase-zhang-v2/phase-root-admission-zhang-v2.json)

The copy-removal Zhang successor passed another 92 focused checks and both standalone SDK clients. It matched every fixed output and retained all four WSL speed passes at 1.04×–1.60×. All four Windows cases still fell below parity at 0.62×–0.92×, so both Zhang drafts remain held. Before another optimization, isolate repeated ordered removal from compiled validation and output copy. [Successor correctness](../build/evidence/u182-next-numerical-lookahead/skeletonize/native-root-admission-zhang-v2.json) · [Successor timing](../build/evidence/u182-next-numerical-lookahead/skeletonize/timing-zhang-v2/timing-root-admission-zhang-v2.json)

Stock reference work passed on both profiles for `hough_circle_peaks` (29 cases each) and `LineModelND` (52 cases each, all eight static members). Hough circle peak ties require each runtime’s ordering, and the exact corner pruning case depends on SciPy’s radius calculation. LineModelND retains a two-ULP SVD direction difference and a warning punctuation difference across profiles. Private native implementations, SDK evidence and speed are open. [Circle peaks reference](../build/evidence/u182-hough-circle-peaks/root-stock-reference-admission-v1.json) · [LineModelND reference](../build/evidence/u182-line-model-nd/root-reference-admission-v1.json)

Private `hough_circle_peaks` and `LineModelND` passed 116 and 208 scoped outcomes respectively, each on two runtimes and both dispatch modes. Both standalone SDK clients passed for each family. The four peak and eight model one-worker comparisons are authenticated and wait for an exclusive host window. Shared/installed gates and class fallback identity with stock absent stay open. [Circle peaks admission](../build/evidence/u182-hough-circle-peaks/private-component-root-admission-v1.json) · [LineModelND admission](../build/evidence/u182-line-model-nd/native-root-admission-v1.json)

Windows Zhang probe attributed a median 96.4% and 75.0% of the two instrumented spans to repeated removal. The passive probe ran 1.27×–1.29× faster than the retained uninstrumented build, so the result prioritizes traversal and gives approximate shares. Root reconstructed all 28 passive batches, 896 active observations and 896 empty controls. The current worker owns a traversal-only successor; the four public Windows losses stay open until changed-source timing passes. [Compiled diagnostic](../build/evidence/u182-next-numerical-lookahead/skeletonize/windows-compiled-phase-zhang-v1/compiled-phase-root-admission-v1.json)

This private round passed 30 of 32 fixed one-worker comparisons with exact outputs. `hough_circle_peaks` passed all eight at 9.48×–140.26× including genuine NumPy ranking and SciPy pruning. The traversal-only Zhang successor `d6d20ac7` passed 96 focused outcomes and both SDKs, then all eight original speed cases at 1.27×–3.02×, resolving all previous fixed Windows losses. Lee and default 3D retain compatibility. `LineModelND` passed 14 of 16 and remains held. Its bounded native and legacy two-point fit cases ran 2.71×–5.98×, but Windows `predict_y` and WSL `predict_x` measured 0.9958× and 0.9995× respectively. Preserve both losses and review genuine provider-route cost before changing source. Shared install/SDK/speed and exact release remain open. [Circle peaks speed](../build/evidence/u182-hough-circle-peaks/timing-v1/timing-root-admission-v1.json) · [Zhang speed](../build/evidence/u182-next-numerical-lookahead/skeletonize/timing-zhang-v3/timing-root-admission-zhang-v3.json) · [LineModelND speed](../build/evidence/u182-line-model-nd/timing-v1/timing-root-admission-v1.json)

Root preserved the first hosted Python 3.11 setup failure. The managed interpreter rejected package installation before any source tests. A verified virtual-environment correction restarted only 3.11 in [run 45039381229](https://github.com/bipxx/scikit-imagexx/actions/runs/45039381229). The original 3.13 lane remains active in [run 45038914055](https://github.com/bipxx/scikit-imagexx/actions/runs/45038914055). Both active lanes retain the exact candidate and 186 selectors. No source, installed or release pass is inferred from a live job.

Bounded `resize_local_mean` passed 104 native checks and both direct SDK clients. The largest normalized workload still uses compatibility. The two-profile phase diagnostic found that the four historical output differences first appear in the horizontal reduction. Retaining NumPy for that axis matched all five frozen diagnostic rows. A broader vertical reduction proof and the original four-case speed gate remain open. [Component admission](../build/evidence/u182-next-numerical-lookahead/resize-local-mean/bounded-component-root-admission-v1.json) · [Reduction diagnostic](../build/evidence/u182-next-numerical-lookahead/resize-local-mean/axis-order-diagnostic-handoff-v1.json)

Bounded `medial_axis` passed 120 focused tests with both dispatch modes on Windows and WSL. GNU and MSVC standalone SDK clients passed. C++ owns exact ordered topology, with SciPy distance and NumPy seeded ranks retained. The four fixed public speed cases, installed and release gates remain open. [Private admission](../build/evidence/u182-next-numerical-lookahead/medial-axis/native-root-admission-v1.json)

The corrected `LineModelND` matched all 16 public workloads and met speed parity on 14. `predict_x` passed both profiles. Windows and WSL `predict_y` remained below parity at 0.9852× and 0.9959×. Keep the class held. The diagnosed boundary will compare live alias/owner-view overhead against allocating NumPy prediction when the host is quiet. [Changed-source timing](../build/evidence/u182-line-model-nd/timing-v2/timing-root-admission-v2.json) · [Diagnostic readiness](../build/evidence/u182-line-model-nd/predict-y-diagnostic-v1/diagnostic-readiness-v1.json)

Root verified 47 disjoint family paths and four exact expansion dependencies against current Haar `817b6f82`. The next compatible installed bundle may reuse this audit after delivery. No additional cumulative candidate was created. [Projection check](../build/evidence/u182-next-numerical-lookahead/after-haar-attachment-feasibility-v2.json)

Haar `817b6f82` passed both complete Windows source runs with 4,780 tests and eight subtests per Python. Root independently verified all 186 selectors, 286 source files, both process exits, matching collections, normal plugins and zero scientific imports. The first 3.11 managed-Python setup failed before tests. A virtual environment fixed that lane, and 3.13 ran once. Root admitted all ten installed one-worker cases at 1.026×–108.45×, with 140 accepted samples and exact outputs. Four-profile qualification is live on the same source, with all four capacity steps passed. Exact public, SDK and packaging intake and main delivery remain pending. [Complete source admission](../build/evidence/u182-haar-features/complete-294-root-admission-hosted-v1.json) · [Installed speed](../build/evidence/u182-haar-features/haar-294-installed-speed-root-admission-v2.json) · [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45041856008)

Bounded `medial_axis` passed all eight fixed one-worker comparisons with exact outputs at 6.42×–15.45×. The full public call keeps SciPy distance and NumPy seeded ordering. Shared installation and SDK packaging, installed speed and four-profile release remain open. [Private timing admission](../build/evidence/u182-next-numerical-lookahead/medial-axis/timing-v1/timing-root-admission-v1.json)

Private `draw.random_shapes` at `ce878576` passed 96 reference cases across Windows and WSL minimum, both dispatch modes and both standalone SDK clients. Private `transform.pyramid_reduce` and `transform.pyramid_expand` at `738df1a9` passed 160 focused checks and 16 exact native-stage witnesses with the existing Gaussian and resize binaries. Pyramids passed all eight fixed one-worker comparisons at 1.090×–1.820×. Random shapes passed all eight fixed comparisons at 1.053×–1.551×. Shared installation and release for both families remain open. [Random-shapes speed](../build/evidence/u182-random-shapes/timing-v1/timing-root-admission-v2.json) [Random-shapes admission](../build/evidence/u182-random-shapes/private-component-root-admission-v1.json) · [Pyramid admission](../build/evidence/u182-pyramid-spatial/native-root-admission-v1.json) · [Pyramid speed](../build/evidence/u182-pyramid-spatial/timing-v1/timing-root-admission-v1.json)

Private `LineModelND` at `b7f8be0d` passed native lifecycle and standalone SDK checks. Fixed x/y speed reached 5.896×–11.234×, with 14 of 16 public comparisons meeting parity. The 3D prediction on Windows and SVD fit on WSL minimum remain below parity. Camera and coffee passed all four v3 measured cases at 1.153×–14.461× after correcting the failed Windows v2 pool inventory setup. Shared installation and release remain open. Eagle still needs its compressed stock input for timing. [LineModelND timing](../build/evidence/u182-line-model-nd/timing-v3/timing-root-admission-v4.json) · [Camera and coffee failed-run intake](../build/evidence/u182-next-asset-provenance/camera-coffee-eagle-gate-v1/windows-final-guard-failure-root-intake-v1.json) · [Camera and coffee speed](../build/evidence/u182-next-asset-provenance/camera-coffee-eagle-gate-v1/timing-root-admission-v3.json)

The corrected private Haar scopes passed 96 drawing checks and 126 coordinate/value checks on each profile. All ten fixed cases passed at 1.104×–95.368×. The private `regionprops_table` cases passed at 8.959×–366.730×. Camera, coffee and Eagle have separate frozen source companions. These scopes still need installed integration and release. [Haar timing](../build/evidence/u182-haar-features/haar-descriptor-timing-root-admission-v1.json) · [Region-table timing](../build/evidence/u182-next-numerical-lookahead/regionprops-table/timing-v1/root-timing-admission-v1.json) · [Eagle source](../build/evidence/u182-next-asset-provenance/next-four-more/root-eagle-source-admission-v1.json)

Current private speed work (2026-11-17, authorized after the completed delivery): acceptance requires original median / candidate median **>=1.0 for every assessed public-call case**, with fixed cases, allocation included and one worker. [All24 current private cases now pass](evidence/speed-parity-private24-root-admission-v1.json) on12700K/Linux3.11.14/NumPy1.24.4: LBP4 at1.068–3.373x, flood/fill8 at1.018–1.363x, current pooled-find MARK8 at1.089–9.063x, and supplemental direct64-bit find4 at1.073–5.737x. [Current MARK/find runtime admission](../build/speed-parity-root-v1/find64-pooled-v1/root-runtime-admission-v1.json) passes474Python checks, five installed-static clients and eight strict wideFP records; the separate LBP/flood admissions retain their exact unchanged source scopes. Source/runtime/performance receipts appear in the affected API rows. The [initial20-case baseline](../build/speed-parity-root-v1/private_public_compare_v3.py) retains16 losses, and the later uint64 loss that drove the pooled conversion fix remains retained. [All four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json), including32 emitted wideFP records, with Python/provider parity separate. [Complete public Python runtime-disabled checks also pass](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json):962XML0bad/399raw caller32 records, with two LBP outside-API observer diagnostics disclosed. [Complete183 source barriers pass on both Pythons](../build/speed-parity-root-v1/source-both-root-admission-v1.json):4693XML each/0bad/44normalplugins. Candidate `75b21947` preserves the tested tree and is privately pushed. [Exact75b genuine four-profile/three-stager qualification](evidence/speed-parity-whole-product-root-admission-v1.json) passes; the current genuine controlled24 speed gate fails4cases, so main delivery is pending. These private gains change no shipped classification or full-contract checkbox; owners/handles remain in the existing plan/ledger.


Current delivered main281 increment: [root genuine admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) closes exact-source four-profile qualification and three authenticated staged products for `aaea1b060e8acb4df08e855a6574e1449416ee42`:963868 public XML checks,304SDK checks,48corpora; all seven independent readers exited0 and source/authority postidentity passed. The earlier import/reporting failures remain retained. Current public-matmul [VI4run43979924366](https://github.com/bipxx/scikit-imagexx/actions/runs/43979924366), controller166ccd, completed all capacity/install/current-host-F1/cohort/measurement steps successfully. Root extracted133resultfiles/15.14MB from one authenticated memory transfer, retaining no archive. [Root VI4 performance admission](evidence/vi-public-matmul-minimum-performance-root-admission-v1.json) now passes independent1242+17 metadata checks/all9proofs and four raw-derived wins (2.84–4.69x),56samples/8warmupgroups. Successful output/fullFP/caller32 buffers were not emitted; source-bound runtime assertions and F1/rawjournal evidence retain their stated grade. [Final source projection](evidence/current281-main-source-projection-v1.json) matches qualifiedAA Git inputs and actualrawsource bytes;22 main-only additions/userWIP are preserved. All10 affected API rows now carry partialN/B/R and scopedqualification credit. [Mainf62c97dc delivery and checkout closeout](evidence/current281-main-delivery-v1.json) are verified; the completedcheckout isretired,811branchheads/userWIP preserved, andno activework remains inthisboundedgoal. Priorpolicyblockedcleanup ismanual-only; full-contract boxes remainopen.

Historical failed source/release transition (2026-11-16): [failed438 root intake](evidence/vi-public-log2-four-profile-failure-root-intake-v1.json) verifies four failed profiles and 48 current VI pairs: **20 pass / 28 strict-word failures**. Caller32, semantic FP, ownership and provider-count gates passed within those records. [Actual case0 stage intake](evidence/vi-public-log2-case0-stage-root-intake-v1.json), run43840022512/controller84cf832e, retains one original/native pair, four log2 batches, 11 controlled saves / 5 restores, exact final32 state and no secondary error. CSR and both log2 input/output buffers match. The [fixed source model](../build/vi-public-log2-provider-v1/qualification-stage-reduction-analysis-v1/root-model-source-admission-v1.json) reproduces the native word and matches observed original products; row reduction versus final dot remains unresolved. Entropy and dot vectors are modeled. The [one retained-row study](evidence/vi-public-log2-retained-reduceat-ineligible-root-intake-v1.json), run43850878993/controllerc9c641bc, stopped at CPU eligibility: EPYC7763/model1 versus captured9V74/model17. It made zero numerical/helper calls and skipped checkout, large transfer and runtime setup. Its GO is consumed; no host search or retry. The [row-source admission](../build/vi-public-log2-provider-v1/qualification-row-reduction-static-v1/root-source-admission-v1.json) and [dot-source admission](../build/vi-public-log2-provider-v1/qualification-dot-static-v1/root-source-admission-v1.json) now establish different reduction graphs in the captured source/binary candidates. The actual selected dot pointer and original entropy remain unobserved; neither source analysis proves the sole numerical cause. [Windows raw-CRLF failure](evidence/vi-public-log2-windows-byte-policy-root-failure-v1.json) has an attrs-only2270 correction and [scoped full173 Python3.11 pass](../build/current281-final-source-v5/root-windows311-admission-v5.json); Python3.13 is held until numerical correction. Final source still requires both complete barriers, four-profile qualification and VI4. Product/dependency retention is complete. Four transport ZIPs (4,696,883,335 bytes), the small stage ZIP (260,160 bytes) and the v5/311 fixture remain policy-blocked manual cleanup. No API/member gate or shipping count is promoted.

Current VI correction is **compiled and privately validated at `77088f5387e0734584a4d1da91934312215b33c1`**: use the current public NumPy log2 for two batches and matmul for two dense inner products, in L/D/L/D order, while retaining C++ counts, scaling, products and reductions. Old scalar/log2-only SDK ABI, defaults and frozen golds remain separate. [Corrected source-reader equations](../build/vi-public-matmul-shared-history-v1/root-history-equations-v6.json) pass32 inverses,32 tamper refusals,2051 baseline source identities,8 actual direct-reader equations,14 frozen prepared dependencies and the actual frozen-controller boundary/support. The [complete affected311 predecessor](../build/vi-public-matmul-provider-v1/windows311-full-v1/terminal.json) failed91 of1832 source checks, with no scientific calls; its full records and source snapshots are retained. The [second complete311 campaign](../build/vi-public-matmul-provider-v1/windows311-full-v2/terminal.json) failed5 of1834 checks, confined to a frozen controller loading current MARK inputs. The [third complete311 campaign](../build/vi-public-matmul-provider-v1/windows311-full-v3/terminal.json) failed3 of1834 checks because its frozen driver and sampler frame used different roots. The [five actual controller regressions](../build/vi-public-matmul-provider-v1/windows311-controller-v3/terminal.json) now pass with coherent17-file authenticated inputs,44 normal plugins, unchanged source and zero scientific calls. [Root completeaffectedboth admission](../build/vi-public-matmul-provider-v1/root-both-admission-v4.json) now passes1834 checks on eachPython,identicalorderedIDs/44normalplugins/conftest/25unchangedsourcepins/zero scientificcalls. This complete33-module+integration check remains separate from the final complete workflow and release barriers. Ordinary incremental compilation and installation passed; [private32 root admission](evidence/vi-public-matmul-private32-root-admission-v1.json) verifies 32 checks, including 24 current original/native pairs: 22 exact two-word returns and two matching errors, 44 log2 plus 44 matmul callbacks, matched target FP and exact caller32 restoration on the retained Linux AVX2 runtime. Eight legacy checks retain their saved-gold scope. The [current35 controls perPython](../build/vi-public-matmul-provider-v1/independent-current-results-v3.json), [prospective runtime envelope](../build/vi-public-matmul-runtime-envelope-v1/handoff-v2.json) and [SDK source proposal](../build/vi-public-matmul-SDK-source-v1/handoff-v1.json) remain preparation evidence; the new ordinary compiled private32 result does not close installed SDK, scalar/SIMD, complete workflow, four-profile release, VI4 or delivery gates. Five completed fullsource fixtures (28,185,637,848 logical bytes total, including the three failures) are manual-only after automatic deletion rejection;0removed. Their commands, XML, logs, source and failures remain outside the disposable roots. No shipped or full-contract promotion.

Current shared source integration: [complete five-module source checks](../build/vi-public-matmul-provider-v1/root-integration-current-both-admission-v2.json) passed170 each on Python3.11/3.13 with identicalIDs/77rawsourcepins/44normalplugins/conftest and no science. The earlier170-check311 run failed24 old capture-fixture cases because default root arguments were bound before the historical-root fixture; a test-helper correction preserves every assertion and explicit wrong-root refusal. A staged check found two excess EOF blank lines in the new SDK guard; after removing only those lines and refreshing matching hash entries, the [final170 both source admission](evidence/vi-public-matmul-shared-integration-root-source-admission-v1.json) passed again. Exact21 integration files are privately frozen at `ba17f859261fb60af7f1b2ca5814fcfbddc79039`; the [complete181 source311 campaign](evidence/vi-public-matmul-full-source-failure-root-intake-v1.json) failed50 source checks (4583 ordinary passes plus8passed subtests), with313 held. The source corrections subsequently passed complete affected checks on both Pythons; final full-workflow progress is recorded below. Actual workflow now selects181 prepipe tests, including the complete prior affected33 closure and current SDK/qualification/VI4 guards. Windows34 runtime and the versioned VI4 controller are source-integrated and UNBOUND; their actual runtime/release/performance gates remain pending. Five bounded fixture trees totaling490449010logicalB are manual-only after automatic deletion rejection; no bytes reclaimed. No API/full-contract/shipping promotion.

Latest complete-source failure: [root intake](evidence/vi-public-matmul-full-source-failure-root-intake-v1.json) preserves exactba17/4641XML/50fail/44normalplugins/no-failfast and independently unchanged HEAD/index/77rawpins. The launcher supplied a relative GO path and missed terminal.json after changing directories; no closing scientific counters are available. Actual source assertions exposed V7/V8 count drift, missing newgroup CLI choices and historical cached-path/borrowed-factory scope gaps. Targeted50 passed44 and failed6, prompting the two remaining corrections; the52case triage passed with unchanged source/no scientific calls. These checks do not replace complete41affected modules plus pureintegration node BOTH, followed by complete181workflowBOTH. InstalledSDK/Windows/current modes/genuine4profiles/VI4/main gates staypending; no shipping promotion. Failedfullfixture7,071,043,014logicalB remainsmanual after one exact policy-rejected removal,0removed.

Historical input closure follow-up: [complete affected311 failure](../build/vi-public-matmul-provider-v1/windows311-sourcebarrier-repair-v1/closeout-v1.json) retained2052XML/2047pass/5FileNotFound/44normalplugins/52unchangedsourcepins/science-import0. The [exact frozen-input audit](../build/vi-public-matmul-provider-v1/historical-input-closure-v1/closure-report-v1.json) identifies13 required2270files/235,873B:two receipts,three fixtures,eightlicenses. Root verified all13 against NativeWindowsGit blobs/currentrawbytes; [V6 equations](../build/vi-public-matmul-shared-history-v1/root-history-equations-v12.json) now authenticate2064sourceinputs/73,980,143B plus54inverses/tamperrefusals. [Five missing-input cases](../build/vi-public-matmul-provider-v1/windows311-missing-epoch-repair-v1/terminal.json) pass. The [complete41-module+source-node311 successor](../build/vi-public-matmul-provider-v1/root-complete-affected-windows311-admission-v2.json) passes2052checks/0bad/44normalplugins/conftest/53rawpostpins/science0. The [root BOTH admission](../build/vi-public-matmul-provider-v1/root-complete-affected-both-admission-v2.json) passes2052 each with identicalIDs/53sourcepins/44plugins and science0; exact15 correctedfiles are privately committed/pushed at6c027543. Complete181BOTH remains pending. A predispatch NativeGit filter check found .gitignore/LICENSE newline conversion under Windows autocrlf=true; [both attributes and the strict .gitattributes V7 inverse](evidence/vi-public-matmul-windows-raw-transport-root-admission-v1.json) are now privately frozen ataaea1b06, with exact Windows filter bytes/source equations/pins verified. The [complete full181/311 root admission](../build/current281-final-source-v7/root-windows311-admission-v1.json) passes4635tests+8subtests/zero bad/44normalplugins+conftest/55rawpins/science0;the [complete181 BOTH root admission](../build/current281-final-source-v7/root-both-admission-v1.json) now passes4635tests+8subtests each, identicalorderedIDs/selectors/sourceindex/55rawpins, under [bound root GO](../build/current281-final-source-v7/root-GO-v7.json). CompleteaffectedBOTH is passed within6c027 scope; corrected final full181BOTH is now independentlypassed onaaea1b06; SDK/runtime/genuine/VI4/main gates remainpending. Passed313 fixture5,825,496,627logicalB is also manual-only after one exact policy-rejected native removal;0removed. Failed5.825GBfixture and74MBtriagefixture each hadoneexactnative removal policy-rejected;0removed/manual-only,rawoutside. No API promotion.

Latest private VI SDK closure: [ordinary/runtime-disabled owning controls](evidence/vi-public-log2-private-owning-sdk-root-admission-v1.json) and [true compiler-scalar/ASan+UBSan controls](evidence/vi-public-log2-private-sdk-current-modes-root-admission-v1.json) pass on the unchanged114ed production/frozen123 SDK source. Each of the four private SDK runs passed its full owning test with20 caller32 and20 refusal32 comparisons,142 new compatible calls/220 callbacks/10 default refusals, defaultworker1; scalar/sanitizer emitted no sanitizer diagnostics. Disclosed scalarstdlog2 callback plumbing is separate from current public NumPylog2 parity. Fresh large current-runtime original/native tests and final173 source checks, genuine four-profile qualification, VI4 timing and main delivery remain pending. The historical scalar gold records retain their original scope. No full-contract or shipped API/member gate is promoted by these private passes.

Historical complete source gate PASSED at8ac: [exact8ac actualboth173 readiness](evidence/vi-public-log2-current-complete-source-readiness-v1.json) verifies4338ordinary+8subtests per3.11/3.13,0bad/44normalplugins/conftest andcleanHEAD/index/rawpostidentity, with no science/FP/helper/SDK/build/timing calls. Its later [four-profile43817382620](https://github.com/bipxx/scikit-imagexx/actions/runs/43817382620) failed, as recorded above; product/SDK/stager acceptance and VI4/main281 remain pending. No shipped API/member promotion. Localfixturecleanup waspolicy-blocked, rawproofs preservedoutside; source failures27/41 remainfailed historical receipts.

Complete release-source check: [actual3.11 full173 admission](../build/current281-final-source-v4/root-windows311-admission-v4.json) passes4338ordinary+8subtests/4346suiteXML,0bad/44normalplugins/conftest/cleanHEAD8ac/index/postidentity/science0. Exact3.13 later passed under the same76b binding, as recorded in the complete8ac admission above; the live PID/session instruction is retired. Frozenprivate8ac isremoteverified; main remains271, with genuine4profiles/actualcapacity4/SDK304/3stagers/VI4/shipping281 stillpending. Completed311fixture6,947,955,808logicalB waspolicy-blocked before execution; manual-only, no deletion orparent/alternate retry. All raw/source/index evidence remains outsideit.

Current source successor: [root admission30e078](../build/vi-public-log2-provider-v1/shared-history-correction-v2/root-source-admission-v2.json) verifies20source/40raw-canonical inverse equations,71author proofs and the literalcurrent benchmark25/capture38 source pins. Computation/provider/sampler code is preserved; prospective VI4authority remainsNULL. [Actualaffected root admission](../build/vi-public-log2-provider-v1/shared-history-correction-v2/integrated-controls-v2/root-both-admission-v2.json) passes780XML each on3.11/3.13,0bad/44normalplugins/90sourcepostpins/science0 andidentical780IDs. Rootmayfreeze26sourcefiles; fullactual173both remainsmandatory. Completeactual173both/genuinefourprofiles/VI4timing/main281 remainpending; no API gate promotion or scientific replay. Prior41/27 failures stayretained.

Integrated source correction (2026-11-16, root): the complete21-module shared-history check at context2051 is **FAILED**, actual3.11 exit1/651XML/610passed/41failed/0errors/skips, source unchanged and science0;3.13 was not started. [Raw failed handoff](../build/vi-public-log2-provider-v1/shared-history-controls-v1/handoff-v1.json) and [root failure admission plus17exact predecessor snapshots](../build/vi-public-log2-provider-v1/shared-history-correction-v2/entry-v2.json) remain immutable. VI37 literal shared pins, MARK2 double projection and rolling2 old-input fixtures require narrow corrections (A VI/rolling, C controller/currentcontract pins, root shared wiring; B remaining direct-consumer source review). Prior224/19/46 source passes retain their earlier scopes and do not close integration; complete173both remains mandatory after successor freeze. Private16 and4SDKmode results remain within unchanged mathematical/product scope. No hosted dispatch or numerical replay. Waste:103.891s exposed missed shared dependencies; stopped3.13/full173/science.

Fresh private VI large-reference closure: [16-node admission](evidence/vi-public-log2-private-large-pairs-root-admission-v1.json) passes8historical scalar SDK checks plus8fresh original/native pairs at workers1/4. Current pairs match allfloat64[2] words, targetFP/callerraw32, input/outputownership,88native returns/16current publicNumPylog2 batches, defaultworker1 andsource/runtimepostidentity. This is private3.11.14/NumPy1.24.4/AVX2 evidence; no genuine/SKX, fullcontract orperformance promotion. The completef10 source barrier failed27metadata guards (4322XML/0errors/skips); allfailures are retained and source-only corrections now require integratedchecks and COMPLETE173both. Main remains271; genuinefourprofiles/VI4timing/main delivery remainpending.

Latest private LBP closure: [Compiler scalar/ASanUBSan saved-domain admission](evidence/lbp-entry-fp-private-modes-root-saved-admission-v1.json) passes752 strict matches on94recipes/profile with workers1/4, including fullcaller/process32/default1. It complements retained normal752; the scoped installed SDK and genuine four-profile gates now pass; the bounded MARK8/LBP4 original-public comparison now passes as described below; main remains open. Exact test-only`16a5b501` source is independently admitted; [both complete169 barriers passed](evidence/mark-lbp16a5-complete-source-readiness-v1.json) with4134ordinary+8subtests each. [Exact16a5 genuine admission](evidence/mark-lbp16a5-whole-product-root-admission-v2.json) now passes all4profiles plus3stagers:963560publicXML/304SDKchecks/48corpora, no failures/errors/skips. This is candidate qualification; named private SDK modes and MARK8/LBP4 performance now pass their scoped gates; VI4 and shipping remain pending.

Historical resume transition (superseded): tracking repair `1cd85168` is delivered on main with no production coverage change. Exact9e15 [complete169 source checks](evidence/mark-lbp9e15-complete-source-readiness-v1.json) passed both retained Pythons, but [hosted qualification43702559008](https://github.com/bipxx/scikit-imagexx/actions/runs/43702559008) failed tooling guards before preparation on all4 profiles. No genuine product or SDK gate passed in that campaign. Private normal752 LBP records/normal SDK clients are retained and scoped; [actual entry admission](evidence/lbp-entry-fp-private-modes-root-entry-v1.json) authorizes mode compilation only. True scalar/sanitizer numerical, whole owning SDK FP and original-public performance gates remain pending; shipped paths stay271 and candidate281 stays unshipped. Root owns admission/main; A diagnoses hosted guards, C owns bounded LBP modes, B owns SDK closure. The review paragraph below is historical.

Historical progress review (superseded): main/remote are `4f28df7d`, with unchanged production paths since delivered `ef465027`. Inventory still has 271 partial native image paths, five partial native classes and 197 upstream-only entries; three metadata entries and one exception provider complete the 477-key inventory. The latest MARK/LBP source [failed both complete169 barriers](evidence/mark-lbp93f-complete-source-failure-v1.json), three metadata failures each; the [clean frozen 9e15 correction](../build/mark-lbp-linux-group-fix-v1/handoff-v1.json) still needs independent admission and complete reruns before downstream release gates. No API/member gate is promoted by its reported 105 affected checks per Python. Shipped latest68 performance remains 34 wins/34 losses; candidate admitted64 remains 38 wins/26 losses, while the newest crop comparison is prior-C++ private scope. Implementation is paused and current local/hosted work checks are idle. The [current coverage checkpoint](EXECUTION_PLAN.md#current-coverage-checkpoint--supersedes-older-live-status-text) supersedes old live assignments; exact receipts, remaining source/release/performance and cleanup gates are recorded in the existing ledger. Review validation is document/state only.

Tracking checks in that review: structural ledger validation passes368 and ready count is0, but worker-status output fails on missing branch fields and the generated roadmap check fails with stale268/200 classifications. Blind regeneration would also erase four reviewed IGG route/method/safety fields; those remain preserved. The existing checkpoint records both failures and root-owned narrow repairs. They do not change any API gate or authorize execution.

Historical storage-only closeout: [fixture removal](../build/cleanup/storage-20261115/fixture-removal-receipt.json) and [nine merged-worktree retirements](../build/cleanup/storage-20261115/worktree-removal-receipt.json) removed 67,189,758,281 logical bytes; all 102 retained evidence hashes and retired branch heads remain exact. Kept unfinished source, environments and 154 failed/unmatched fixture roots. [Required archive outputs](../build/cleanup/storage-20261115/archive-products-receipt.json) add 5,316,903,559 retained logical bytes, including raw diagnostics/reproducers and unpacked baseline products. Automatic approval review blocked the fifteen D ZIP removals before execution; all remain present, with [exact paths and retention review](../build/cleanup/storage-20261115/archive-discard-review.json). Earlier missing C archives are unavailable local evidence. The [existing storage checkpoint](EXECUTION_PLAN.md#storage-maintenance-checkpoint--latest-2026-11-15) and ledger record the remaining manual gates and observed space changes. No private/candidate/shipping API gate or domain classification changed; implementation remains paused and validation is document/state-only.

Remaining-space assessment: the same storage checkpoint now links the scoped 219-C/28-D worktree source projection and selected remaining-root sizes. Most known C content is checkout duplication; qualification data and retired products/proofs are separate substantial groups. Source-tree sizes, readable-file logical sizes, historical blocked-scratch measurements and physical allocation are explicitly distinguished. No further deletion or API gate change; unmerged source, unique dependencies and previously blocked targets remain protected pending the existing cleanup gates.

Current cleanup closeout: user approved the clean-copy ancestry exception. [Root terminal admission](evidence/clean-worktree153-retirement-root-closeout-v1.json) confirms **153** copies removed, **105.08 GB cached logical source**; all branches/commits retained. Observed free space rose **101.46 GB** across C:/D:; physical allocation was not independently measured. The active9e15 SDK source, environments, raw evidence, products and baseline dependencies remain protected. Removed source paths require restoration from their pinned commits before reuse. Previously policy-blocked targets remain manual-only.

Current measurement gate: [17a source successor](evidence/mark-lbp-output-root-successor-root-source-admission-v1.json) passed99 complete controls per retained Python; [actual run43727894422](https://github.com/bipxx/scikit-imagexx/actions/runs/43727894422) and [independent root performance admission](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json) passed for MARK8/LBP4:12cases,168samples,24warmup groups, **3wins/9losses**. [Every median/raw loss is published](../benchmarks/mark-lbp16a5-minimum-initial-v1/README.md). Three MARK subpixel wins are6.57–7.81x; other MARK0.313–0.461x as fast; all four LBP cases0.0206–0.0546x as fast. Source-enforced FP/caller/C1 checks passed; reports have target words but no raw per-case32-byte payloads. Qualified source16a5, oldDDA64 and shipped271 remain distinct. VI4 and shipping281 are held. The [earlier containment failure](evidence/mark-lbp437217-output-containment-root-failure-closeout-v1.json) is retained, with no timing by source-order evidence. All dispatches are terminal. The [new2.96MB ZIP](evidence/mark-lbp43727894422-result-transport-manual-closeout-v1.json) is manual-only after automatic approval rejected deletion.

| Current scope | Verified evidence / result | Remaining gate and owner |
|---|---|---|
| Shipped 271 image paths | [Shipping product/SDK admission](evidence/join-util-views-whole-product-admission-v1.json), delivered in main `ef465027`; [68-case published comparison](../benchmarks/join-util-views-minimum-initial-v1/README.md): 34 wins, 34 losses. | Root: reconcile further API/domain closure and delivered increments. These classifications remain partial. |
| Current qualified 281-path candidate `16a5b501` | [Four-profile/three-stager admission](evidence/mark-lbp16a5-whole-product-root-admission-v2.json), [private named SDK modes](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json), and [MARK8/LBP4 one-worker comparison](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json). | Root: VI4 causal/performance gate remains held. [Source correction c60](evidence/current281-source-lineage-root-admission-v1.json) passes24affected modules/1,565checks perPython; private cumulative1df4 preserves qualified16 production. Complete169both remains mandatory/held until VI’s production decision is stable; no affected-test union or shipping promotion. Full contracts remain open. |
| Historical qualified 281-path candidate `dda9cc83` | [Four-profile admission](evidence/current281-whole-product-root-admission-v3.json); [Radon/flood performance admission](evidence/current281-radon-flood-performance-root-admission-v1.json) and [JOIN/rolling admission](evidence/current281-join-rolling-performance-root-admission-v1.json). | Root: complete remaining performance/disposition and shipping projection/delivery gates. Candidate coverage is separate from shipped coverage. |
| Candidate one-thread comparisons | [Historical DDA64](../benchmarks/current281-minimum-initial-v1/README.md):896samples,128warmup groups, **38wins/26losses**. Separately, [16a5 MARK8/LBP4](../benchmarks/mark-lbp16a5-minimum-initial-v1/README.md):168samples,24warmup groups, **3wins/9losses**. | VI4 remains held. [Root historical64 reuse](evidence/current281-historical64-root-reuse-admission-v1.json) now credits unchanged measured family scope. These retain distinct source/runtime/host labels, including twoAMDhosts within64; no combinedsamehost80-case or library-wide superiority claim. |
| MARK wide-label outer domain | [Saved whole-operation comparison](evidence/mark-wide-comparison-fp-native-root-closeout-v1.json), [exact16a5 genuine product/SDK admission](evidence/mark-lbp16a5-whole-product-root-admission-v2.json), and [private compiler-scalar/ASanUBSan named SDK scope](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) passed. | [MARK8 public-call performance](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json) now passes its bounded gate,3wins/5losses; shipping remains open. Legacy saved values and wide compiled FP assertions have distinct scopes; raw per-case SDK hardware words were not emitted. |
| VI failing-value diagnostic | [Current16 one-pair witness](evidence/vi-current16-genuine-value-root-admission-v1.json) passes on AVX2. [Controlled log2 result](evidence/vi-log2-avx512-provider-root-runtime-admission-v1.json): default NumPy differs at6/112words by1ULP; disabling compiled/enabled AVX512 targets matches the core-version scalar112/112, with raw FP/caller/identity checks. | Root: [actual default public failure](evidence/vi-avx512-default-public-word-root-failure-admission-v1.json), run43765280748/bbe, verifies94proofs:1original+1native, firstfloat64 differs1ULP, second equal, matched input/targetFP/native phases and caller/source/identity checks. Later assertions137-139 are unverified. [Correctedsource281both](evidence/vi-avx512-public-v2-root-source-admission-v1.json) was dispatched once, but [AMD7763 was ineligible](evidence/vi-avx512-public-v2-ineligible-host-root-closeout-v1.json): zero targets/control/timing, no retry. Private77088f53 uses explicit public NumPy log2 and matmul providers; both complete affected source checks and ordinary compiled private32 passed within the retained AVX2 scope. Shared SDK/qualification/Windows/VI4 integration is active; installed SDK, broader profiles, final workflow, release and VI4 gates remain open. The old9V45/samehost cause remains unproved. |
| LBP independent comparison | The [earlier FP failure](evidence/current281-diagnostics-lbp-terminal-root-closeout-v1.json) is preserved. Its correction has [genuine16a5](evidence/mark-lbp16a5-whole-product-root-admission-v2.json), [named private SDK](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) and [new four-case public-call evidence](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json):all4losses,18–48x slower. | Root: full-domain parity and main delivery remain open; retain every slower case. |
| IGG 282-path candidate `852cc188` | [All-four-profile failed qualification](evidence/igg282-failed-campaign-root-closeout-v1.json): 38 failures. [Half exception-parity source correction](evidence/igg-half-error-parity-root-source-admission-v1.json). [Local actual-core three-case diagnostic](evidence/igg282-minimum-native-diagnostic-root-closeout-v1.json) passed saved observations. | Metrics integrator/root: numerical host failure remains unlocalized; local pass is not a fix. [Hosted diagnostic source](evidence/igg-host-native-diagnostic-root-source-admission-v1.json) and [workflow](evidence/igg-native-three-workflow-root-source-admission-v1.json) are admitted for their source-only scope; the [hosted terminal closeout](evidence/current281-diagnostics-lbp-terminal-root-closeout-v1.json) records three passing native diagnostic cases. The prior host failure did not reproduce; causal closure and successor qualification remain pending. |
| Golden-angle 283-path source | [Corrected complete-source readiness](evidence/golden283-complete-source-readiness-v1.json): 168 selectors, 4,041 ordinary nodes plus eight subtests on each retained Python. | Root: inherited genuine-release failures must close before release qualification. Source readiness alone is not release acceptance. |

Failed predecessor campaigns and their zero-timing outcomes remain in the [execution plan](EXECUTION_PLAN.md) and retained receipts. [Storage relocation](evidence/cold-worktree-storage-relocation-v1.json) changed source locators, not coverage. Source-only admissions above grant no execution authority.

Latest terminal diagnostics: [root closeout](evidence/current281-diagnostics-lbp-terminal-root-closeout-v1.json) retains VI42370505430 and IGG42370507136 successes and LBP42370508973 failure before timing. VI/IGG prior failures did not reproduce and remain unresolved. LBP failed the hardware-FP equality gate after one original/native pair; the [successor capture](evidence/lbp-actual-fp-diagnostic-root-closeout-v1.json) now retains actual dictionaries and identical output payloads. A conditional-clear correction still requires independent admission and changed-source parity. No new parity/release gate or speed credit; candidate results remain64cases38wins26losses.

Pending LBP diagnostic: [f59 review](evidence/lbp-cohort-fp-diagnostic-root-review-v1.json) required a retained-reference fix after old C1 failures; [4e94 source correction](evidence/lbp-cohort-fp-diagnostic-root-source-admission-v2.json) now passes complete37source/inert checks per Python. The [actual82ec one-pair successor](evidence/lbp-actual-fp-diagnostic-root-closeout-v1.json) is terminal failed at strict FP, zero timing; earlier preparation failure remains retained. [Flood recorder review](evidence/flood-fill-stage-diagnostic-root-review-v1.json) requires primary/current-case retention fixes; intake and eight native profiles remain blocked.

Hosted-source correction: [4e failed preparation and 82ec test-only admission](evidence/lbp-diagnostic-gh-sha-fixture-root-admission-v1.json). The real GitHub SHA exposed a synthetic-fixture mismatch: 36 source checks passed, one failed, helper/target steps skipped. Both retained Windows versions now pass all38 under representative hosted variables. Wrapper/assertions unchanged; the [actual successor diagnostic](evidence/lbp-actual-fp-diagnostic-root-closeout-v1.json) failed strict FP after one pair: output words match, x87 status1058 versus32 differs. No speed or full parity gate closes.

Current optimization disposition: [crop private parity](evidence/crop-metadata-private-parity-root-admission-v1.json) admits 924 scoped checks, and [controlled FP/storage comparison](evidence/crop-metadata-controlled-prior-cpp-root-admission-v1.json) covers 32 calls. The [eight-case prior-C++ comparison](../benchmarks/crop-metadata-private-v1/README.md) has seven faster medians and one slower F-copy case. This is **not** performance against original scikit-image; the optimization remains unaccepted and original-public gates remain open.

**Update at every bounded closeout:** identify affected API/member keys, changed domains, exact source/profile/receipt, passed/pending/failed gates and next owner/trigger in the existing task handoff; update affected rows and this snapshot in the same validated increment. Root verifies evidence before marking a gate. Reuse valid retained evidence and preserve failures; do not rerun science to refresh this document.

Navigation: [gate definitions](#completion-gates-for-every-api-and-applicable-member), [domain checklist](#domain-checklist-within-each-api-gate), [module index](#module-index), [per-API rows](#per-api-checklist), [whole-library gaps](#runtime-and-whole-library-closure), [closeout rule](#closeout-update-rule).

| Inventory category | Public keys | Meaning |
|---|---:|---|
| `native-partial` | 308 | Native image paths with incomplete or unassessed full-contract partitions. |
| `native-class-partial` | 5 | Classes with only part of computation/state/member ownership closed. |
| `metadata` | 3 | Metadata operations; applicability and full public contract still need explicit disposition. |
| `exception-metadata-provider` | 1 | Exception/class metadata provider; not a native image algorithm. |
| `upstream-only` | 150 | Shipping implementation still uses the upstream path. |
| `packaged-asset-partial` | 10 | Packaged asset routes with bounded shipped coverage. |

Total: **477 public keys**, including **29 classes and 284 statically inventoried member paths**. Static members are additional contract work, not 284 extra top-level APIs. Current member classifications: 63 native-class-member-partial, 11 exception-metadata members and 210 upstream-only; none of those labels establishes full member closure. Runtime-returned objects and discovered members remain open below.

The earlier 271-path release has [four-profile admission](evidence/join-util-views-whole-product-admission-v1.json), delivered in main `ef465027`. This is bounded release credit, not full parity. The latest published 68-case cohort has [34 wins and 34 losses](../benchmarks/join-util-views-minimum-initial-v1/README.md); it does not establish library-wide speed superiority. The earlier 265-product [18 wins and 10 losses](../benchmarks/next-metrics-minimum-initial-v1/README.md) remains historical evidence.

Flood/fill bottleneck: [eight-call stage attribution](evidence/flood-stage-profile-root-closeout-v1.json) found a consistent 0.385-0.449ms assignment stage; all eight inputs and caller/process FP restoration checks passed. This is profiling evidence, not parity or speed acceptance. A changed indexing path still needs its own parity and one-thread benchmark.

MARK corrected native scope: [exact16a5 genuine four profiles](evidence/mark-lbp16a5-whole-product-root-admission-v2.json) and [private installed family modes](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) now pass their saved-domain scope. [MARK8 public-call comparison](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json) now passes its bounded evidence gate,3wins/5losses; delivery remains open; shipping stays 271.

FP regression scope: [MARK/LBP native controls](evidence/mark-lbp-native-fp-regression-root-closeout-v1.json), [LBP saved compiler modes](evidence/lbp-entry-fp-private-modes-root-saved-admission-v1.json), and [scoped installed SDK witnesses](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) support the corrections. [MARK8/LBP4 performance](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json) now passes its bounded gate; full public-domain parity and delivery remain open; source-only controls add no native credit.

Current validated increment: [LBP large first-case correction](evidence/lbp-entry-fp-firstcase-root-closeout-v1.json) matches original minimum-host bytes and FP; modern is a candidate control. [Flood assignment private parity](evidence/flood-assignment-lines-private-parity-root-closeout-v1.json) passes636checks with strict FP and protocol coverage, but performance/genuine worker gates remain open. GLCM count correction has [eight current setup prefixes](evidence/glcm-large-count-corrected-prefix-intake-v3.json); no new original observations or native coverage credit yet.

## Completion gates for every API and applicable member

| Gate | Required checklist coverage and evidence |
|---|---|
| C — contract | Pinned upstream signature/defaults; positional/keyword behavior; dtype/promotion/byte order/metadata; shape/rank/axes/channel axis; C/F/transposed/reversed/stepped/readonly/empty inputs; nonfinite/edge values; outputs, ownership/base/aliasing/mutation; warnings/errors and ordering; callbacks, FP/RNG/state where applicable. Quarantines and unsupported domains stay explicit. |
| N — computation | Computation is owned by the C++ implementation in the advertised domain. Disclose NumPy/SciPy/other provider work and whole-original fallback. Cover unusual domains or retain a named open frontier. Metadata/provider-only APIs need an explicit applicability decision. |
| H — hot swap | Public import locations, signatures/defaults, return types and class/state behavior remain compatible. Real installed namespace routes to the owned builtin/core; unsupported cases follow the declared contract. Exercise representative swapped pipelines and refuse unapproved original/helper fallback. |
| P — parity | A per-function/member contract suite against retained original observations, with actual args/options and source/runtime identity. Include errors/warnings/protocols/layout/alias/state/FP policies, not only default values. Independent admission must cover the claimed domains/platforms. Verify exact caller restoration and incoming/outgoing FP semantics; use full32/MXCSR checks only within the admitted platform envelope and retain unsupported-platform gaps. |
| S — SDK | A complete named C++ operation through the genuinely installed exported CMake target; header/archive/runtime use, ownership/lifetimes/output capacity/errors, supported dtypes/layouts and global-worker behavior. A primitive, syntax-only compile or caller-composed fragment is insufficient. |
| V — scalar/SIMD | Relevant baseline/scalar, runtime-disabled SIMD and applicable vector paths agree. Check boundaries, tails, alignment, signed strides and exceptional values; sanitizer/MSVC/platform checks as required. Ordered or metadata-only cores may stay serial with a specific rationale. |
| W — workers | Default 1; optional process-global worker budget; setters/getters and restoration; workers 1/4 where supported. One captured budget per operation/composition when applicable; stable ordering and no hidden nested pools. Record justified no-worker-lookup serial paths. |
| B — speed | Allocation-inclusive actual public calls vs original scikit-image with one thread. Freeze cases, sampling, warmup and acceptance before calls; preserve raw journals and every slower case. Separate local C++ diagnostics, controlled prior-C++ comparisons and optional-worker scaling from original-public speed claims. |
| R — delivery | Exact source, complete workflow source checks on both retained Pythons with normal conftest/plugins, required genuine four-profile qualification and independent SDK/product admission, reviewed performance disposition, final shipping projection, main merge/push and eligible cleanup. Partial runs and collection audits do not close this gate. |

## Domain checklist within each API gate

Apply these partitions to each API/member's existing contract record; record passed, pending, failed or justified N/A there. These boxes define completion requirements, not a new task queue or a claim that every function supports every dtype.

- [ ] Signature, positional/keyword arguments, defaults and sentinel identity; aliases, return types and lazy iteration behavior.
- [ ] Supported bool/integer/half/float/complex dtypes, promotion, byte order and dtype metadata; explicit unsupported/provider domains.
- [ ] Rank, shape, axes/channels, empty/singleton inputs; C/F, transposed, reversed, stepped and readonly layouts; stride and output-capacity boundaries.
- [ ] Values at algorithm boundaries, ties, overflow, NaN/Inf/subnormals; warnings, exceptions and validation order; applicable FP/RNG/callback state.
- [ ] Output values/tolerances, shape/dtype/layout, ownership/aliasing, mutation, caches and generator or class state.
- [ ] Per-function original-reference parity and actual installed hot-swap routing, including declared fallback behavior; source/runtime/domain-scoped admission.
- [ ] Complete installed C++ SDK operation, lifetimes and errors; applicable scalar/vector paths, tails and sanitizers, with serial-core rationale where needed.
- [ ] Process-global worker default 1, optional parallelism, workers 1/4 and restoration; do not infer parallel execution from a setter alone.
- [ ] One-thread allocation-inclusive public comparison against original scikit-image, retained raw samples and every slower case; separate comparisons against prior C++.
- [ ] Exact-source release qualification, independent product/SDK admission, reviewed performance disposition and main delivery.

## Delivered increment: six new public APIs

Main delivers **271 partial image paths** at source `a6bae9ac86dbba362537758784edf06c973e9c53` (216 product files, 102 SDK files); production code is unchanged from b557. Exact shipping projection passed and main merge `ef465027` was pushed. Three clean, merged obsolete metadata worktrees were removed; active source, packages, raw evidence and environments were preserved.

- [x] Compatible private native components and retained original-reference evidence reviewed for the six paths below.
- [x] Shared registry, single-core bindings, SDK hooks, installed-source selectors and explicit test-helper preparation/collection wired.
- [x] [Independent source review](evidence/join-util-views-final-source-admission-v1.json) and [five-test metadata repair](evidence/join-util-views-sourcebarrier-fix-admission-v1.json).
- [x] [Byte-preserving checkout policy](evidence/join-util-views-windows-bytepolicy-admission-v1.json) and [complete faithful source readiness](evidence/join-util-views-source-readiness-v3.json): 3,035 cases plus 8 subtests on each retained Python with core.autocrlf=true; no failures/errors/skips/scientific imports. Production blobs remain unchanged from the predecessor.
- [x] [Current a6 complete source readiness](evidence/join-util-views-source-readiness-v4.json): 3,056 tests plus 8 subtests on each retained Python, normal conftest/plugins, zero science; the earlier v3 scope above remains historical.
- [x] [Exact-source four-profile genuine product/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) for a6bae9/run42262940227: 952,116 public XML checks, 12,202 hosted source XML checks, 256 SDK checks and 48 corpora, all zero failures/errors/skips. Product216/SDK102/32families and each actual core/runtime/source/selector/helper scope admitted. Seven archives and their profile/stage seals authenticated once. Prior failed campaigns remain failed; main now ships271 after scoped performance admission and exact-source delivery.
- [x] [Bound controller v3 source admission](evidence/join-util-views-bound-controller-source-admission-v3.json) passes for its fixed56-case scope. Actual host helper and real zero-body cohorts are pending. The frozen v4 successor `57072ba25f38785e79d930951a56ab4adfc0583f` includes the 12 frozen ellipse workloads because prior dependency/runtime/core reuse does not match: fixed 10 API cohorts/68 cases/952 samples/136 warmup groups. [Independent v4 source admission](evidence/join-util-views-bound-controller-source-admission-v4.json) passes; all ten real cohorts passed and all 68 measurements completed. The report-only failure and correction are recorded below.
- [x] Actual run42278375082 passed ten real cohorts and completed all 68 measurements, then failed the final validator: utility drivers use frozen pass0 while the validator assumed pass1. [Independent raw review](evidence/join-util-views-failed-campaign-raw-review-v2.json) reconstructs all 952 samples/136 warmup groups/1,508 batches: 34 wins and 34 losses. [Root scoped performance admission](evidence/join-util-views-initial-performance-admission-v1.json) now passes after independently reviewed c7 report-only correction and complete retained-output validation. [All cases and raw journals](../benchmarks/join-util-views-minimum-initial-v1/README.md) are published; optimization remains open. Preserve the failed campaign, raw ordering and all losses; no timing rerun. Prior ellipse10wins/2losses remain historical.
- [x] Final source projection, main merge/push `ef465027`, and safe removal of three clean merged obsolete metadata worktrees. Active baseline/optimization worktrees and raw/package/environment evidence retained.

| API | Existing private/reference scope | Remaining release work |
|---|---|---|
| `segmentation.join_segmentations` | Integer-label composition and complete private operation; returned ArrayMap class/protocol frontiers remain explicit. | Scoped whole-core/SDK and public comparison admitted (8 cases, 8 losses); class/domain closure, optimization remains; delivered in ef465027. |
| `util.crop` | View geometry and real POD copying; object/zero-item/subclass provider scopes disclosed. | Scoped genuine SDK/public and comparison admitted (8 cases, 8 losses); platform/domain frontiers, optimization remains; delivered in ef465027. |
| `util.regular_grid` | Source-faithful grid preparation/finish and complete private SDK; full 49-node saved Linux selector wired. | Scoped genuine runs and comparison admitted (8 cases, 8 wins); Windows/default-integer/libm domains remain; delivered in ef465027. |
| `util.slice_along_axes` | Saved source/error/view/copy observations and native geometry/POD transport; FP envelope admitted. | Scoped installed core/SDK, saved-domain and comparison admitted (8 cases, 8 losses); providers/platforms, optimization remains; delivered in ef465027. |
| `util.view_as_blocks` | Integral borrowed geometry/strides; NumPy representation/provider boundaries explicit. | Scoped installed core/SDK, own cohort and comparison admitted (4 cases, 4 losses); full domains/platforms, optimization remains; delivered in ef465027. |
| `util.view_as_windows` | Integral borrowed sliding-window geometry/strides; provider and metadata overflow frontiers explicit. | Scoped installed core/SDK, separate cohort and comparison admitted (4 cases, 4 losses); full domains/platforms, optimization remains; delivered in ef465027. |

Failed predecessor 42238215124 observation (2026-11-06; retained below). Its failures remain historical. Successor 42262940227 completed successfully on all four profiles, received scoped performance admission, and was delivered in main ef465027. This table preserves the predecessor failures:

| Profile | Hosted job result | Acceptance status |
|---|---|---|
| Windows / Python 3.13.16 | Source barrier passed; genuine qualification failed (100 JOIN setup failures). | Historical failed gate; corrected a6 successor independently qualified. |
| Modern Linux / Python 3.13.16 | Job completed successfully. | Retained scoped evidence; does not close four-profile release. |
| Minimum Linux / Python 3.11.17 | Job completed successfully. | Retained scoped evidence; does not close four-profile release. |
| Debug Linux / Python 3.13.16 | Job completed successfully. | Retained scoped evidence; does not close four-profile release. |

The same increment contains reviewed density and F-layout boundary optimizations. Their private C++ ratios do not close original-public speed gates. The rejected [complete source attempt](evidence/join-util-views-complete-sourcebarrier-failure-v1.json) remains failed.

## Next source/reference work

- [ ] `segmentation.mark_boundaries`: [Saved original admission](evidence/mark-boundaries-saved-raw-admission-v2.json) covers 125 cases per profile (250 complete observations, 251 consumed calls including the retained failed attempt). One separately authorized fixed-codec witness closed the structured-error gap; no successful original rows were replayed. [Independent native successor admission](evidence/mark-boundaries-fp-successor-admission-v1.json) passes on source 6d4373b5720b5ee959c1d313e14e3d8f754085e4 for 1,500 saved worker/mode comparisons (125 cases per profile, workers 1/4, normal/disabled/scalar), including strict tiny-color target FP. Small RHS palette assignment remains explicit NumPy provider work. Genuine installation, platform/domain closure, one-thread original-public performance and delivery remain open. [Executable benchmark source review](evidence/mark-boundaries-benchmark-source-admission-v1.json) and [handoff reconciliation](evidence/mark-boundaries-benchmark-handoff-reconciliation-v1.json) pass on 446bde47e7462278389c0a87affb881ed775f1ec: 52 source/zero-body controls per Python, fixed 8 cases, no actual targets or timings. Genuine binding, outer first-cohort and separate timing GO remain pending.
- [ ] `feature.draw_multiblock_lbp`: [Saved original admission](evidence/draw-multiblock-lbp-saved-raw-admission-v1.json) covers 94 cases per profile. [Independent private-component review](evidence/draw-multiblock-lbp-native-independent-review-v2.json) covers source 0e51dbc1e2c4741e243a91652d8c0b9771c96363: 1,128 target comparisons across two profiles, three scalar/SIMD modes and workers 1/4 pass. Of these, 1,048 complete records match and 80 differ only in post-return JSON collector MXCSR diagnostics, retained explicitly. Target FP and exact caller restoration remain strict. [Root scoped component admission](evidence/draw-multiblock-lbp-native-root-admission-v2.json) passes. [Private integration source/ABI review](evidence/draw-multiblock-lbp-private-integration-admission-v1.json) and [immutable handoff reconciliation](evidence/draw-multiblock-lbp-private-handoff-reconciliation-v1.json) pass in their scoped checks; the qualifier rejects the temporary conditional install rule, so genuine qualification is not passed. Normal enabled MARK/LBP integration is frozen at `2991881411972e6266f7e132c9dca8b7226a8567`; [root shared-source admission](evidence/mark-lbp-enabled-root-source-admission-v1.json) and [complete source barriers](evidence/mark-lbp-enabled-source-readiness-v1.json) pass (3,168 tests plus 8 subtests per retained Python). [Failed four-profile campaign admission](evidence/mark-lbp-enabled-failure-admission-v1.json) for299/run42280271105 preserves 956,246 XML checks with 48 identical LBP grayscale dtype-metadata failures (12 per profile), zero errors/skips, and 272 passing SDK checks. [Root-admitted binder correction](evidence/draw-multiblock-lbp-metadata-fix-admission-v1.json) at99d6c428 passes 1,128 saved target comparisons and 144 private metadata checks; it is entering shared integration. New genuine qualification and performance remain pending; no unchanged retry. Shared genuine core/installed SDK, full domains/platforms, one-thread original-public performance and delivery remain open. [Executable LBP benchmark source](evidence/lbp-benchmark-source-admission-v1.json) passes scoped independent review: 97 source/zero-body controls per Python, fixed four LBP cases and eight MARK cases through reused binding/F1/sampler logic. Actual genuine binding, outer cohorts and timing remain pending.
- [ ] `transform.frt2` and `transform.ifrt2`: [Source/storage review](evidence/finite-radon-source-storage-admission-v1.json) passes on d2109de2 for 108 recipes, 54 per API. [Actual original capture](evidence/finite-radon-saved-raw-admission-v1.json) completed 108 observations per profile once after [independent controller admission](evidence/finite-radon-controller-admission-v1.json). [Observed native admission](evidence/finite-radon-observed-native-admission-v2.json) at ee639c0c now passes 1,296 strict saved worker/mode comparisons across both profiles plus scoped assembled SDK evidence. Wider dtype/protocol/platform/genuine/performance gates remain open.
- [ ] `draw.polygon_perimeter` and `draw.rectangle_perimeter`: reuse retained clipping assessment; pinned Matplotlib/AGG reference/native dependency strategy is still required. Do not repeat the already-recorded dependency discovery.
- [ ] Performance controller C1 successor: [post-stop source/authority checks and one post-batch verifier](evidence/join-util-views-controller-admission-v2.json) pass scoped independent review; bind only after actual genuine product admission. No measurement authority is inferred from pure tests.
- [x] MARK/LBP unbound controller source only: [root admission](evidence/mark-lbp-controller-root-admission-v1.json) at `3548cf16d474375f53068db295c99c1275173377`, 82 retained controls per Windows Python; actual driver pass0 verified. Failed producer299/run42280271105 cannot supply binding. Genuine successor, two actual cohorts and 12-case original-public timing remain unchecked.

- [x] Radon benchmark driver source only: [root admission](evidence/finite-radon-benchmark-source-admission-v1.json) at `7fcee23ddcd6c965c1d56e58d87e1dadb7684479` covers 16 prospective unsigned cases, eight per API, with 37 retained controls per Windows Python. Current185 source contexts, larger-case numerical parity, genuine cohorts/bindings and actual one-thread results remain pending.

## Candidate coverage awaiting admission or delivery

These rows describe bounded evidence, not additional shipping coverage. Main remains 271 partial image paths. A worker handoff cannot close an API gate before independent admission.

| Affected APIs | Source and retained result | Still open |
|---|---|---|
| `segmentation.join_segmentations` | Candidate `e03a00633d7146b78391ad65833d3728eb7fd86e`; frozen handoff-v2 SHA256 `d6c8b4aa83971ce05cafa58592b1ad95cfdcd192f0e998f55fef75cac806c077` reports 252 private XML checks / 200 saved rows across minimum/modern, normal/disabled. [Captured-input source admission](evidence/join-captured-input-source-admission-v1.json) is separate from matrix admission. | [Root matrix admission](evidence/join-segmentations-matrix-admission-v2.json) passes within captured-input/private scope. Shared genuine core and Linux helper transport, class/FP/domain closure and new performance acceptance remain pending. Earlier 43 FP failures and 59 setup errors stay retained; eight shipped timing losses remain losses. |
| `util.crop`, `util.slice_along_axes` (copy), `util.view_as_blocks`, `util.view_as_windows` | Candidate `f48fe19edeb336163249383d61a51b186ca610fa`; frozen component handoff and independent peer review retained. Private evidence reports 584 saved nodes in each of four profile/mode combinations; scalar/sanitizer evidence covers changed leaf/assembled SDK only. | [Root scoped component admission](evidence/util-view-boundary-root-admission-v1.json) passes. Dependent LBP copy closure, shared integration, genuine qualification and measured performance remain pending. Borrowed crop/slice and regular-grid paths are unchanged; copied slice shares the optimized POD copier. |
| `transform.frt2`, `transform.ifrt2` | [Exceptional source admission](evidence/finite-radon-exceptional-source-admission-v1.json) and [raw admission v2](evidence/finite-radon-exceptional-raw-admission-v2.json): four fixed NaN/Inf recipes per profile, eight original calls consumed once; no native parity inferred from capture. | [Root exceptional-cast component admission](evidence/finite-radon-exceptional-native-admission-v1.json) passes on `1855377b4baea97293bc7eeb5cb8e2236ac427fc`: 1,344 strict saved worker/mode comparisons plus scoped SDK checks. Wider domains/platforms, genuine hot swap/whole installed SDK, benchmark binding and actual one-thread results remain pending. Old native evidence remains scoped to its captured domain. |

## Current candidate and next-family gates

Reconciled 2026-11-07 against retained receipts. These are scoped checkpoints, separate from the shipping snapshot and full-contract boxes below.

| API / increment | Passed or retained evidence | Pending coverage / next owner |
|---|---|---|
| MARK, multiblock LBP, finite Radon; JOIN and utility optimizations | Cumulative candidate `8c4fb4307746a0eb5fac5d125b6384fa80e88621`: 275 partial image paths, 222 product files, 105 SDK files. [Root reader-fix admission](evidence/mark-lbp-radon-util-join-reader-admission-v1.json) preserves the historical JOIN fixture pins. Both complete source-barrier terminals report 3,255 XML checks, zero failures/errors/skips, zero scientific imports/calls and clean source: [Python 3.11](evidence/mark-lbp-radon-util-sourcebarrier-windows311-v2.json), [Python 3.13](evidence/mark-lbp-radon-util-sourcebarrier-windows313-v2.json). Actual complete source scope is **140 selectors**; 150 denotes numerical/public contract entries. | [Root complete-command admission](evidence/mark-lbp-radon-util-complete-readiness-admission-v1.json) passes. [Run42306808073](https://github.com/bipxx/scikit-imagexx/actions/runs/42306808073): all four source guards pass; minimum profile failed initial import because the Radon binding built with NumPy 2.x headers requires C API 0x12 while NumPy 1.24 provides 0x10. [The failure audit](evidence/mark-lbp-radon-util-failed-campaign-root-admission-v1.json) is complete: three scoped successes (718,560 public XML checks, 210 SDK checks, 36 corpora), minimum zero public/SDK/corpus execution. [Runtime-aware ABI correction](evidence/finite-radon-numpy-abi-root-admission-v1.json) passes private cross-version checks. The earlier combined candidate was frozen at `ab7872a5483a5d5e3660ba20fdd873d78463a1b5`: [author handoff](evidence/abi-vi-cumulative-handoff-v1.json) reports 280 partial image paths, 283 registered functions, 226 product files, 107 SDK files, 37 families/74 checks and 1,037 passing affected source checks per retained Python. [Root scoped source admission](evidence/abi-vi-cumulative-root-source-admission-v1.json) and [complete source-command admission](evidence/abi-vi-cumulative-complete-readiness-admission-v1.json) pass: all 148 actual selectors, 3,336 tests plus 8 subtests per retained Python, zero failures/errors/skips/science, normal conftest/plugins. This is staging, not genuine product admission or shipping. Its larger VI preflight subsequently failed; the corrected successor below includes the admitted correction. Genuine qualification, hot swap, public measurements and delivery remain pending. Main stays 271. |
| Earlier cumulative candidate | [Failed complete source barriers](evidence/mark-lbp-radon-util-sourcebarrier-failure-v1.json) each retained one historical JOIN literal-reader failure. Earlier `0e17d061` also omitted Radon star exports; the export and subsequent reader corrections are versioned successors. | Preserve failures; reuse unchanged numerical evidence only within its valid dependency scope. No relabeling or unchanged retry. |
| MARK / LBP / Radon benchmarks | [Author handoff](evidence/current275-family-measurement-handoff-v1.json), source `95e64f135c7d3d8cbef7c8acb543ed31ea86c079`, aligns controllers with candidate `8c4fb430`. Retained 183 prototype checks and 8 affected checks per Python; timing and genuine authority fields are unbound. | [Independent review](evidence/current275-family-measurement-independent-review-v1.json) and [root source admission](evidence/current275-family-measurement-root-admission-v1.json) pass. Actual qualified producer/runtime/core/SDK/helper binding, real entry cohorts and separate measurement authority remain pending. Workflow registration is not a performance result. Owner: root. |
| `metrics.variation_of_information` | [Independent scoped review](evidence/variation-of-information-independent-review-v1.json), source `b331a559af038ef2231f5327b864627513bdc2d8`: 8 original saved cases per profile, 64 exact public observations, 112 private XML checks; scoped native scalar/sanitizer and assembled SDK evidence. Prior compile and NaN-sign failures retained; ordered multiply corrects demonstrated operand precedence. | [Root old-component admission](evidence/variation-of-information-root-admission-v1.json) passes. Successor `1a060891` [closes scoped current-count dependency](evidence/variation-of-information-current-counts-root-admission-v1.json): 160 private XML checks, 64 unchanged saved observations and candidate-only dense/fallback/logical-NNZ controls. Assembled SDK96 plus current counts and VI is not the genuine 107-file candidate SDK. [Executable consumer source](evidence/variation-of-information-benchmark-root-admission-v1.json) and [Windows native smoke against retained GNU words](evidence/variation-of-information-windows-native-root-admission-v1.json) pass. [Independent larger-case readiness](evidence/vi-large-reference-independent-readiness-v1.json) and [root readiness](evidence/vi-large-reference-root-readiness-v1.json) pass. Exactly four additional frozen benchmark references per runtime completed once after separate minimum/modern authority; [independent raw review](evidence/vi-large-reference-independent-raw-v1.json) and [root raw admission](evidence/vi-large-reference-root-raw-v1.json) pass. All four return-word pairs match across profiles; incoming/outgoing FP differences remain semantic. Actual current-native comparison now reports four passing and four result-word mismatches across eight profile/case runs: both 64-label workloads fail on each runtime (2–5 ULP in affected components). Target FP, caller/process restoration and metadata scopes pass; numerical qualification, optional-mode expansion and timing stop for root-cause correction. The frozen references and tolerances are unchanged. [Root-cause admission](evidence/vi-large-native-failure-root-admission-v1.json) isolates seeded pairwise CSR reductions and final DDOT association; identical-argument logs match. A [reviewed narrow correction](evidence/vi-reduction-order-independent-review-v1.json) has [root scoped admission](evidence/vi-reduction-order-root-admission-v1.json): all eight large cases now match exactly, with 192 affected public and 12 pure checks plus scoped scalar/sanitizer/SDK/MSVC evidence. Genuine and broader backend/platform gates remain open. Source inspection found that the executable consumer replaces the SciPy CSR class and would break the frontend type check; [the versioned class-identity correction](evidence/current280-controller-root-admission-v1.json) is source-admitted, with 422 pure controls. Actual genuine benchmark bindings/cohorts remain pending. Existing numerical evidence is unchanged. Selective shared integration, Windows-original/full-FP evidence, genuine binding and one-thread public performance remain pending. No shipped API promotion.  Private owning SDK modes: [normal/disabled](evidence/vi-public-log2-private-owning-sdk-root-admission-v1.json), [true scalar/SAN](evidence/vi-public-log2-private-sdk-current-modes-root-admission-v1.json). Latest [fresh private public-log2 pairs](evidence/vi-public-log2-private-large-pairs-root-admission-v1.json), exactf10 source/private3.11.14/NumPy1.24.4/AVX2, pass8current original/native workers1/4 pairs plus8legacy SDK checks:16XML/88native returns/16publicNumPy batches, exactwords/FP/caller32/default1/postidentity. This preserves private scope; genuine/SKX, broader contracts and performance remain pending. Shared source2051 failed41guards; [root source successor30e078](../build/vi-public-log2-provider-v1/shared-history-correction-v2/root-source-admission-v2.json) fixes current/history entry and literalcurrent benchmark binding withinsource-only scope. Actualaffected checks both, complete173both and exactfour-profile release remain mandatory before main271-to281 delivery. Latest [actualfailed438 rootintake](evidence/vi-public-log2-four-profile-failure-root-intake-v1.json): current48pairs20pass/28strictwordfail acrossLinux3/modes/workers despitecaller32/ownership/FPmatches. Private16remainsAVX2-scoped; broadernumerical/provider/releasegatesopen, exactcausecorrectionpending. [Actual case0 stage capture](evidence/vi-public-log2-case0-stage-root-intake-v1.json) verifies matching CSR/log2 buffers and restored caller/software state despite the one-ULP result difference. The admitted source model matches products/native output; row reduction versus final dot remains pending. [One retained-row study](evidence/vi-public-log2-retained-reduceat-ineligible-root-intake-v1.json) was CPU-ineligible and executed no primitive; it does not close the remaining numerical cause. No coverage promotion. Current public log2+matmul successor: [source history equations](../build/vi-public-matmul-shared-history-v1/root-history-equations-v2.json) pass the corrected reader boundary; the earlier full311 campaigns remain failed91/1832,5/1834 and3/1834. Actual5controller regressions and the [complete1834both source successor](../build/vi-public-matmul-provider-v1/root-both-admission-v4.json) now pass. The [new ordinary compiled private32](evidence/vi-public-matmul-private32-root-admission-v1.json) passes 24 current original/native pairs (22 exact returns, two expected errors), eight legacy checks, provider/ownership/default1/FP/caller32/postidentity on Linux AVX2 only. Source is frozen privately at77088f53; legacy SDK evidence retains its old source scope. Latest [root local readiness/qualification dispatch](evidence/vi-public-matmul-source-private-SDK-root-progress-v2.json), exactprivateaaea1b06: completeactual181 BOTH passes4635tests+8subtests each; private60 passes16saved/56caller32/normalplugins/default1/postidentity; ordinaryinstalledSDKnormal+disabled passes190raw32pairs. Truecompiler-scalar/SAN owningSDK runtimes now independently pass190additional raw32pairs/184callbackwords/default1/SIMD0and1/no sanitizerdiagnostics/fullpostidentity, with40verifiedproofs. Priorprivate32 fresh24pair/naturaltargetFP scope staysseparate; SDK scalarcallbacks do not establish publicNumPy parity. [Genuine43949036485](https://github.com/bipxx/scikit-imagexx/actions/runs/43949036485) completed successfully on all four profiles; root admitted seven artifact transfers/seals and retained products. The reporting-only reader failure is retained; its correction passed33 source controls per Python and all four complete successor readers pass root admission:963868 XML checks/304SDK/48corpora. All three staged-product readers and root genuine aggregate admission now pass. Current VI4run43979924366 completedSUCCESS and root performance admission passes4wins/56samples/8warmupgroups with all scope limits retained. Final source projection passes; main281 source and scoped performance are included in this increment, with actualmainf62c97dc delivery/remote/checkout closeout verified. Shipping271 andfullcontractboxesunchanged. |
| `morphology.flood`, `morphology.flood_fill`; `segmentation` aliases | [Independent zero-body readiness review](evidence/flood-fill-readiness-independent-review-v1.json), source `7c14ee863934a2a6c5370d0ca699d3d299fba166`: bounded source/geometry, 96 recipes per profile, exact caller restoration and entry refusal controls. | [Root readiness admission](evidence/flood-fill-root-readiness-admission-v1.json) passes. Separate minimum96 then modern96 captures completed once with exact caller/process restoration; 79 returns/17 API errors and 78 returns/18 API errors respectively. [Independent192raw review](evidence/flood-fill-raw-independent-review-v1.json) and [root raw admission](evidence/flood-fill-raw-root-admission-v1.json) pass; captured FP/error differences remain semantic and unnormalized. [Private native component](evidence/flood-fill-native-root-admission-v1.json) and [independent review](evidence/flood-fill-native-independent-review-v1.json) pass: 800 nodes across two profiles/four modes with all 192 saved observations preserved, plus scoped complete SDK checks. [C++ traversal successor](evidence/flood-fill-cost-root-admission-v1.json) removes measured per-pixel allocations and passes the same private scopes; its local prior-C++ comparison is 2.39–3.41× faster across eight diagnostic cases, with extra padded-buffer memory. This is not original-public speed evidence. NumPy full-image minimum, mask count and converted-RHS broadcast remain provider computation. Genuine global-worker/alias/hot-swap/SDK/four-profile/Windows/public-performance gates remain pending. Four public names share two canonical operations; no duplicate algorithm credit. The [source-only outer benchmark controller](evidence/flood-fill-controller-root-admission-v1.json), source30ce23ca, is independently root-admitted: 91 pure controls per Python, two API-specific real-entry cohorts and one unchanged eight-case comparison. The [213-source retarget](evidence/flood-fill-controller-root-admission-v2.json), b0385768, now has root admission: 96 pure/inert controls per Python, exact earlier executable bodies after the versioned manifest-path inverse, and current prospective108 SDK metadata. All actual bindings remain null; cohorts and timings have not run. Its cases are the actual eight executable `flood_fill_native.py` cases. The separate 12-case `flood_fill.py` proposal has no executable fixture and remains pending; no performance evidence is inferred from it. |

Corrective bundle `352dc55681b3774ae3f304ac4783757a9d911b52` passed independent shared-source review but [failed its complete 153-selector commands](evidence/vi-reduction-controller-complete-failure-v1.json) on both retained Windows Pythons: 3,624 XML checks each, eight failures each (seven stale count assertions and one Windows scratch-path limit). No scientific work ran. The [independently admitted five-file metadata successor](evidence/vi-reduction-controller-successor-source-admission-v1.json), `a5e019c6a4b0fff3a18635ab0e2f552986abba3e`, preserves production, qualifier and workflow bytes. Its [complete 153-selector commands pass on both retained Windows Pythons](evidence/vi-reduction-controller-complete-readiness-v2.json): 3,616 ordinary tests plus eight subtests each, no failures/errors/skips/science, normal conftest/plugins and identical collected IDs. The corrected short scratch path resolves the Windows setup failure. [Exact-source four-profile qualification42329688164](https://github.com/bipxx/scikit-imagexx/actions/runs/42329688164) completed **failure on all four profiles**. Watcher15437 is terminal. Retained diagnostics identify three harness defects: generic VI fixtures used floating labels, a VI fixture was unpacked as two rather than three values, and the flood expected mask strides omitted its padded base. Each profile has 28 failures; these are failed release gates, even where dedicated native tests passed. All four archives are retained; product admission, benchmark binding and delivery remain pending. No unchanged retry. Counts remain **280 staged partial paths / 226 product files / 107 SDK files / 37 families**, not shipping. Earlier ab78 source checks and its four large numerical failures remain historical; neither predecessor qualifies the successor. Root owns the complete-check, genuine four-profile, performance and delivery gates. The checklist records those actual jobs; it grants no additional execution authority.

Affected JOIN/util benchmark preparation: [source admission](evidence/current275-join-util-root-admission-v1.json) passes for unbound `7b01014e`: five affected APIs, 32 fixed cases, 448 samples and 64 warmup groups. Unchanged grid/metrics/boundary/ellipse workloads are excluded; prior results retain their historical source/runtime scope. Corrected genuine product, actual bindings/cohorts and separate timing authority remain pending. Manual workflow registration is setup only.

`restoration.rolling_ball` remains upstream-only. Source `2e99dd5c` has [independent source/storage review](evidence/rolling-ball-independent-source-v1.json) and [root source admission](evidence/rolling-ball-root-source-v1.json): 75 prospective recipes, 85 pure checks per Python and source-derived bounded geometry. [Actual runtime/controller readiness](evidence/rolling-ball-root-readiness-v1.json) passes root independent admission. Separate minimum75 then modern75 captures completed once; [root 150-row raw admission](evidence/rolling-ball-root-raw-admission-v1.json) covers 126 returns, 24 API errors, 16 warnings, exact caller/process restoration and unchanged inputs/state. The [root-admitted private native component](evidence/rolling-ball-native-root-admission-v1.json), `c821f6532b3c429bf83357baf161adf711e2d0a1`, passes 750 saved observations plus 60 boundary checks; only the 750 observations establish the stated FP/caller scope. Its complete float32/float64 private SDK now has an [independently admitted dtype/byte-order successor](evidence/rolling-ball-sdk-dtypes-root-admission-v1.json), `42f08d83009621bcd958d2653920d228935f064e`: 11 saved bool/integer/half/swapped cases, 396 new SDK executions, 312 prior float-case executions and 132 strict FP comparisons across the admitted modes. The new operation owns conversion, kernel/traversal and output bytes. Unsafe integer casts, wider half/nonfinite/rounding domains, Windows runtime and broader protocols/FP remain open. The [separate frozen outer controller](evidence/rolling-ball-controller-root-admission-v1.json) is independently root-admitted for its four-file source/64 pure checks per Python only; actual bindings remain null. The cumulative rolling281 source is frozen at `188a91ef31609c796d4d83e8957c772544f31428`; its corrective successor `cbb3d2cd993f71fff08bb4a548d9c456c7d2de60` is not ready for qualification because of the source-projection blocker below. Shared integration, genuine installation and original-public one-thread speed remain pending. Existing ball/ellipsoid-kernel evidence does not prove the full background-removal algorithm. All original grants are exhausted; new original/timing authority is zero.

`segmentation.inverse_gaussian_gradient` remains upstream-only. [Frozen source/test/safety handoff](evidence/inverse-gaussian-gradient-source-handoff-v1.json) reports 79 prospective recipes and 88 pure checks per Python; [root source admission](evidence/inverse-gaussian-gradient-source-root-admission-v1.json) passes. [Independent actual readiness review](evidence/inverse-gaussian-gradient-readiness-independent-review-v1.json) passes for successor `769811615ffb168f5c42bc15267026eba631fc4b`, including corrected alpha/sigma factories and four zero-body prefixes. [Root actual readiness](evidence/inverse-gaussian-gradient-root-readiness-v1.json) and [independent root admission of all 158 raw observations](evidence/inverse-gaussian-gradient-root-raw-admission-v1.json) pass after separate minimum/modern captures. Totals: 138 returns, 20 API errors, six warnings, all 158 caller restorations and both process restorations exact. Three result differences and one error-module difference between runtimes remain semantic. All original grants are exhausted. Native component `9b8e870ecdeadf614b9e0a632598d39f518f8dcb`, warning-policy successor `065cc3a79391e4c5343511b53bbbaac992a4c185`, and supplemental private Windows SDK results now have [scoped combined root admission](evidence/inverse-gaussian-gradient-native-root-admission-v1.json). They do not establish shipped parity or timing authority. Its route now explicitly preserves input-dtype derivative/square/sum/sqrt stages and alpha broadcasting; the API has no mask, channel-axis or output-buffer keyword.

[Scratch cleanup](evidence/obsolete-sourcebarrier-cleanup-v1.json) removed 25.63 GB of obsolete source-guard fixtures from two exact scratch roots; observed free space rose by 25.62 GB. Source, environments, raw evidence and active worktrees were retained.

The utility-to-LBP copy dependency is privately checked in the cumulative candidate (1,128 saved comparisons and 144 metadata controls), superseding the earlier pending private-dependency wording above. It does not establish genuine installed-product closure. Checklist updates reuse implementation receipts; no scientific execution is performed solely to refresh this document.

Checklist closeout (2026-11-06): reconciled all 477 public keys, 29 classes and 284 static member paths against the existing inventory. Updated VI candidate failures/successor, private rolling-ball and inverse-gradient readiness scopes; no full-contract box closed. Waste review: reused retained receipts and source metadata; document/state checks only, no scientific replay or new tracking system. Next owner/trigger: root updates affected rows at each bounded task closeout or main delivery. Subsequent reconciliation: a5 complete source checks passed but its four-profile release failed; rolling281 double-inverse finding is corrected, but the213complete barrier failed one stale family assertion; cd703complete reruns are pending. IGG private native/policy/SDK scope is now root-admitted, with sharedintegration stillunready. Useful captured profile differences retained. Avoidable root receipt-reader mistake counted subtests as ordinary XML nodes; corrected schema reading without rerunning checks.

Current admission update (2026-11-07): [Exact dda9 four-profile product/SDK admission](evidence/current281-whole-product-root-admission-v2.json) passes **963,512 public XML checks, 304 SDK checks and 48 corpora**, plus three authenticated stagers. This credits only the candidate domain; main remains271. Root next prepares the existing six comparison controllers for80cases,1120samples/160warmupgroups; actual one-thread results and delivery remain pending.

[Golden-angle private root admission](evidence/order-angles-golden-ratio-native-root-admission-v1.json) credits548XML/540saved comparisons and scoped SDK on fb2a; C now integrates21exact family files into an isolated852 successor. [GLCM private root admission](evidence/graycomatrix-graycoprops-native-root-admission-v1.json) credits1520XML/1504strict saved comparisons and five189-return/6-control SDK runs on de6; Windows is compile-only. [IGG complete source readiness](evidence/inverse-gaussian-gradient-complete-source-readiness-v1.json) credits both literal166-selector commands,3998ordinary tests+8subtests each, but larger8 accuracy GO is blocked on the retained F1 terminal-diagnostic finding. No private component becomes shipped or full-contract closed through these admissions.

Latest execution (2026-11-07): [IGG F1 readiness correction](evidence/inverse-gaussian-gradient-large-accuracy-root-readiness-v2.json), [16 once-only larger original references](evidence/inverse-gaussian-gradient-large-accuracy-root-raw-v1.json), and [16 one-worker native accuracy comparisons](evidence/inverse-gaussian-gradient-large-accuracy-native-root-admission-v1.json) pass. Both original budgets are exhausted. [Exact852 candidate282 qualification42356138491](https://github.com/bipxx/scikit-imagexx/actions/runs/42356138491) is live; product/SDK admission, performance and delivery remain pending. Separately, qualified281 uses [consumer-compatible admission v3](evidence/current281-whole-product-root-admission-v3.json) and [actual minimum stager](evidence/current281-minimum-stager-admission-v1.json) for its six controller bindings. Shipping stays271; these accuracy checks are not timings.

Current controller/source checkpoints: [Bound six-family281 benchmark source](evidence/current281-bound-controller-root-admission-v1.json) passes514source/inert checks perPython; source-only registration of three absent manual workflows is the next execution prerequisite. Actual13cohorts/80case one-thread comparisons remain pending. [Golden e0E complete168 failure](evidence/golden283-complete-source-failure-v1.json) is retained; [two-file1fb reader correction](evidence/finite-radon-export-reader-root-admission-v2.json) passes20affected source checks perPython and full168 reruns are live. [GLCM outerC1/postbatch source](evidence/gray-texture-outer-controller-root-admission-v1.json) and [282 result-reader metadata](evidence/igg282-result-readers-root-admission-v1.json) are admitted only within their non-scientific scopes; actual products/cohorts/performance are not inferred. Shipping remains271.

## Latest gate reconciliation

| Scope | Retained evidence | What remains open |
|---|---|---|
| Failed 280-path release | [Failed campaign closeout](evidence/vi-reduction-controller-failed-campaign-closeout-v1.json); [terminal run](../build/vi-reduction-controller-qualification-v2/terminal-run-API-v1.json); [remaining three-profile diagnostics](../build/vi-reduction-controller-qualification-v2/42329688164/retained-v1/remaining-three-profile-diagnostics-v1.json); [debug diagnosis](../build/vi-reduction-controller-qualification-v2/debug-retained-v1/handoff-v1.json). | All four release profiles failed; successful individual tests do not close R. Retain the campaign and correct only demonstrated causes. |
| Corrective 281-path candidate | [Independent review: CHANGES_REQUIRED](evidence/current281-corrective-independent-review-v1.json), source `cbb3d2cd993f71fff08bb4a548d9c456c7d2de60`. The raw metadata adapter applies the same predecessor inverse twice. | Versioned successor `213b986504b6f62f79a98b21be0581f1bc9b3559` is frozen with 397 affected source checks per Python and raw/canonical overlap regressions. [Independent root source admission](evidence/current281-projection-root-admission-v2.json) passes for213 after B narrow reviewb9ccdc16. [Complete actual161-selector source barriers failed on213](evidence/current281-complete-source-failure-v1.json), one stale SDK-family assertion per Python: 3,872 ordinary passes, one failure and eight passing subtests each; no science. Sessions30295/89016 are terminal. [Assertion-only successor cd703fbd](evidence/current281-sdk-family-assertion-root-admission-v1.json) passes independent/root source admission and 29 affected checks per Python. [Complete161 reruns failed on cd](evidence/current281-complete-source-failure-v2.json): another reader of the changed test file bypassed the local inverse. Each Python reported 3,874 ordinary passes, one VI-history-source failure and eight passing subtests, with zero science. Sessions74915/45650 are terminal. [Common-entry successor3e2a0fd1](evidence/current281-shared-assertion-projection-root-admission-v2.json) passes independent/root source admission and 1,465 affected checks per Python, covering all 18 identified adapter-consumer modules and seven context-map closures. [Complete161 passed on both retained Pythons](evidence/current281-complete-source-readiness-v3.json): 3,938 ordinary tests plus eight subtests each, no failures/errors/skips/science. [Exact-source qualification42342525409](https://github.com/bipxx/scikit-imagexx/actions/runs/42342525409) is terminal FAILED on all four profiles; watcher93876 exited1. Modern and minimum diagnoses identify the same four strict rolling-ball signature failures. [Complete failed-campaign closeout](evidence/current281-sentinel-failed-campaign-closeout-v1.json) authenticates963,512 XML checks/16failures,304SDK checks and48corpora across four once-retained archives; no release acceptance. [Root diagnosis](evidence/current281-modern-signature-failure-root-v1.json) identifies a private default marker where upstream uses its canonical DEPRECATED class. [Canonical-marker successor dda9 source admission](evidence/current281-rolling-sentinel-root-v1.json) and [complete162 readiness](evidence/current281-complete-source-readiness-v4.json) pass:3,958 ordinary tests plus8subtests per Python. [Successor42347579925](https://github.com/bipxx/scikit-imagexx/actions/runs/42347579925) is terminal SUCCESS at exact source `dda9cc83d9434a10616f84cf1e1125d93f5f36fd` (GitHub status checked 2026-11-07). [Independent whole-product/SDK/stager admission](evidence/current281-whole-product-root-admission-v2.json) now passes for the bounded281candidate. Performance and main delivery remain pending; shipping stays271. |
| Inverse Gaussian gradient | [Native handoff](../build/inverse-gaussian-gradient-native-v1/handoff-v1.json); [policy successor](../build/inverse-gaussian-gradient-policy-v1/handoff-v1.json); [private Windows SDK supplement](../build/inverse-gaussian-gradient-Windows-sdk-readiness-v1/validation-v1.json). | Combined private root admission passes; [fresh frontend-mode root admission](evidence/inverse-gaussian-gradient-frontend-modes-root-v1.json) additionally covers 664 checks across workers4/disabled/scalar/sanitizer on both retained Linux profiles. Platform/domain gaps, shared integration, genuine installed release and one-thread public comparison remain open. Shipping remains upstream-only. |
| Golden-angle generator | [Root source/storage admission](evidence/order-angles-golden-ratio-root-source-v1.json): 45 prospective recipes. [Actual readiness handoff](../build/order-angles-golden-ratio-readiness-v1/handoff-v1.json), `fb17c0cd`, and [author-frozen capture v3](../build/order-angles-golden-ratio-capture-v3/handoff-v3.json), `ff9b0949`, remain separately scoped. | V2 failed review retained; [v3 independent/combined readiness admission](evidence/order-angles-golden-ratio-capture-root-v3.json) passes. [Minimum raw admission](evidence/order-angles-golden-ratio-minimum-root-raw-v1.json) and [modern/paired raw admission](evidence/order-angles-golden-ratio-modern-root-raw-v1.json) cover90 once-captured cases. All147 index values/profile agree;474semantic x87 before/after differences are retained. [Independent private native review](../build/order-angles-golden-ratio-native-independent-review-v1/review-v1.json) passes on `fb2a7c0ccf99f1ea27e6c31e212a608463831baf`: 548 public XML nodes, 540 saved comparisons and scoped SDK evidence. [Root component admission](evidence/order-angles-golden-ratio-native-root-admission-v1.json) now passes privately; shared genuine hot swap/installed release, broader domains and one-thread speed remain pending. Shipping stays upstream-only. |
| GLCM texture pair | [Root source/storage admission](evidence/graycomatrix-graycoprops-root-source-v1.json), `22873ac5`: 50 matrix and 44 property recipes. [Controller/readiness handoff](../build/graycomatrix-graycoprops-controller-v1/handoff-v1.json), `d2631867`, is author-frozen. | [Versioned terminal correction and combined readiness](evidence/graycomatrix-graycoprops-capture-root-v2.json) pass with strict reuse of8historical zero-body prefixes. [Minimum](evidence/graycomatrix-graycoprops-minimum-root-raw-v1.json) and [modern/paired raw admission](evidence/graycomatrix-graycoprops-modern-root-raw-v1.json) cover188once-captured cases,64returns/30errors/5warnings per profile. Successful payloads agree; typed error/warning metadata and376semanticFPdifferences remain constraints. [Independent private native review](../build/graycomatrix-graycoprops-native-independent-review-v1/review-v1.json) passes on `de6c329a43b4b5c90f54960a6ef10556d6c75940`: 1,520 public XML checks and scoped SDK/mode evidence. [Root component admission](evidence/graycomatrix-graycoprops-native-root-admission-v1.json) now passes privately; genuine hot swap/installed release and actual public performance remain pending. The new executable controller is peer-reviewed, with outer C1 admission still pending; Windows evidence is compile-only and shipping stays upstream-only. |
| Public performance | Published 68-case cohort: 34 wins, 34 losses. [Flood controller retarget admission](evidence/flood-fill-controller-root-admission-v2.json) covers eight executable cases; [VI retarget admission](evidence/vi-measurement-controller-root-admission-v2.json) covers four. Both are source-only and unbound. | Bind future controllers to an accepted actual product; pass real API cohorts before authorized timing. Retain all losses; controller/source tests are not measurements. |

This reconciliation changes no coverage classification. C/N/H/P/S/V/W/B/R markers describe evidence scope, not a percentage inferred from test counts. Root updates this view after each bounded closeout; implementation owners report exact source, domains, profiles, evidence, failures and next owner/trigger in the existing handoff.

## Module index

Counts are shipping inventory classifications, not full-contract completion percentages.

| Module | Public keys | Partial native images | Upstream-only | Other categories |
|---|---:|---:|---:|---:|
| [color](#module-color) | 41 | 40 | 0 | 1 |
| [data](#module-data) | 41 | 1 | 30 | 10 |
| [draw](#module-draw) | 18 | 16 | 2 | 0 |
| [exposure](#module-exposure) | 10 | 10 | 0 | 0 |
| [feature](#module-feature) | 41 | 15 | 26 | 0 |
| [filters](#module-filters) | 100 | 98 | 1 | 1 |
| [filters.rank](#module-filters.rank) | 0 | 0 | 0 | 0 |
| [filters.rank.bilateral](#module-filters.rank.bilateral) | 0 | 0 | 0 | 0 |
| [filters.rank.generic](#module-filters.rank.generic) | 0 | 0 | 0 | 0 |
| [future](#module-future) | 5 | 0 | 5 | 0 |
| [graph](#module-graph) | 15 | 0 | 15 | 0 |
| [io](#module-io) | 11 | 0 | 11 | 0 |
| [measure](#module-measure) | 35 | 28 | 7 | 0 |
| [metrics](#module-metrics) | 10 | 10 | 0 | 0 |
| [morphology](#module-morphology) | 47 | 40 | 6 | 1 |
| [registration](#module-registration) | 3 | 0 | 3 | 0 |
| [restoration](#module-restoration) | 17 | 5 | 12 | 0 |
| [segmentation](#module-segmentation) | 20 | 10 | 10 | 0 |
| [transform](#module-transform) | 38 | 15 | 19 | 4 |
| [util](#module-util) | 25 | 20 | 3 | 2 |

## Per-API checklist

Each row below is a full-closure checkbox. Gate markers initially credit only what the current inventory and shipping admission establish: `N~` is native-partial/class-partial; `R~` is the shipped bounded image-path release. Other gates stay `?` until their exact per-key scope is reconciled. Existing tests/benchmarks are not being erased or reclassified as failures.

<a id="module-color"></a>
### color — 41 public keys

<details><summary>Expand color contract checklist</summary>

<a id="api-color-combine-stains"></a>
- [ ] **`color.combine_stains`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float16, float32, float64`.
  - Signature: `(stains, conv_matrix, *, channel_axis=-1)`.
  - Required focus: remaining half nonfinite/extreme values, warning and FP-callback behavior; retain version-dependent promotion and finite-case evidence. Cases to close: half zero/subnormal/NaN/Inf and supported error-policy cases with exact dtype, metadata, state and warning order.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse the existing native 3xK core and reviewed half-to-float32 boundary; finish affected-domain timing and exceptional-value contracts.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-convert-colorspace"></a>
- [ ] **`color.convert_colorspace`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(arr, fromspace, tospace, *, channel_axis=-1)`.
  - Required focus: case-insensitive names, unsupported-space errors, intermediate dtype/range. Cases to close: all supported source/destination pairs including identity.
  - Easiest remaining route: Audit/extend existing native implementation: Extend existing named-space routing through the native RGB hub.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-deltae-cie76"></a>
- [ ] **`color.deltaE_cie76`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab1, lab2, channel_axis=-1)`.
  - Required focus: broadcast shapes, float32 intermediates, NaNs and extra channels. Cases to close: broadcast Lab tensors and near-zero differences.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Euclidean Lab difference with original promotion points.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-deltae-ciede2000"></a>
- [ ] **`color.deltaE_ciede2000`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1)`.
  - Required focus: hue wrap, zero chroma, branch ties, weighting and cancellation. Cases to close: hue-wrap/achromatic pairs plus dense Lab images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native hue-adjusted CIEDE2000 formula.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-deltae-ciede94"></a>
- [ ] **`color.deltaE_ciede94`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, k2=0.015, *, channel_axis=-1)`.
  - Required focus: reference/sample asymmetry, negative roundoff under square root, k parameters. Cases to close: swapped color pairs and textile parameter sets.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native asymmetric CIE94 chroma/lightness weighting.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-deltae-cmc"></a>
- [ ] **`color.deltaE_cmc`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab1, lab2, kL=1, kC=1, *, channel_axis=-1)`.
  - Required focus: low-lightness branch, hue sectors, zero chroma, kL/kC. Cases to close: sector-boundary colors and low-lightness pairs.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native CMC lightness/chroma/hue ellipse formula.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-gray2rgb"></a>
- [ ] **`color.gray2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, *, channel_axis=-1)`.
  - Required focus: arbitrary channel_axis, dtype preservation and copy behavior. Cases to close: scalar-channel images and non-last channel insertion.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native channel replication.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-gray2rgba"></a>
- [ ] **`color.gray2rgba`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, alpha=None, *, channel_axis=-1)`.
  - Required focus: scalar/array alpha, dtype scaling, broadcast errors. Cases to close: varied alpha shapes and integer/float images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native RGB replication plus alpha broadcast/cast.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-hed2rgb"></a>
- [ ] **`color.hed2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(hed, *, channel_axis=-1)`.
  - Required focus: fixed matrix constants, channel_axis and clipping. Cases to close: HED basis vectors and mixed stain images.
  - Easiest remaining route: Audit/extend existing native implementation: Compose native combine_stains with fixed HED inverse matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-hsv2rgb"></a>
- [ ] **`color.hsv2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(hsv, *, channel_axis=-1)`.
  - Required focus: hue boundary/wrap behavior, saturation zero and range violations. Cases to close: six sector boundaries and grayscale HSV.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse sector-based native HSV conversion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-lab2lch"></a>
- [ ] **`color.lab2lch`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab, *, channel_axis=-1)`.
  - Required focus: angle units/wrap, signed zero and extra channels. Cases to close: achromatic and quadrant-boundary Lab.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native hypot/atan2 conversion preserving remaining channels.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-lab2rgb"></a>
- [ ] **`color.lab2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab, illuminant='D65', observer='2', *, channel_axis=-1)`.
  - Required focus: illuminant/observer, negative XYZ warning, final clipping. Cases to close: all whitepoints and out-of-gamut Lab.
  - Easiest remaining route: Audit/extend existing native implementation: Compose existing Lab-to-XYZ and XYZ-to-sRGB kernels.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-lab2xyz"></a>
- [ ] **`color.lab2xyz`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lab, illuminant='D65', observer='2', *, channel_axis=-1)`.
  - Required focus: illuminant/observer validation and negative-Z warning contract. Cases to close: threshold-adjacent Lab and whitepoint grid.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse inverse piecewise Lab transform and whitepoint lookup.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-label2rgb"></a>
- [ ] **`color.label2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(label, image=None, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay', *, saturation=0, channel_axis=-1)`.
  - Required focus: bg_label/colors, negative labels, saturation, alpha, kind and channel_axis. Cases to close: sparse/large label IDs and overlay/average modes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse palette overlay and grouped-average native paths.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-lch2lab"></a>
- [ ] **`color.lch2lab`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(lch, *, channel_axis=-1)`.
  - Required focus: radians, extra channels and dtype preservation. Cases to close: multi-turn hue values and zero chroma.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native chroma times sine/cosine.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-luv2rgb"></a>
- [ ] **`color.luv2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(luv, *, channel_axis=-1)`.
  - Required focus: zero lightness/chromaticity, final clip and channel_axis. Cases to close: black/near-black and out-of-gamut Luv.
  - Easiest remaining route: Audit/extend existing native implementation: Compose native Luv-to-XYZ then XYZ-to-sRGB.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-luv2xyz"></a>
- [ ] **`color.luv2xyz`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(luv, illuminant='D65', observer='2', *, channel_axis=-1)`.
  - Required focus: illuminant/observer, zero denominators and lightness threshold. Cases to close: zero lightness and whitepoint branches.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse Luv formula with whitepoint.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2gray"></a>
- [ ] **`color.rgb2gray`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer scaling, 3-channel validation, dtype and channel_axis. Cases to close: integer/float RGB with non-last channels.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native fixed luma dot product.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2hed"></a>
- [ ] **`color.rgb2hed`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: log floor, fixed constants, nonnegative clipping. Cases to close: black/white RGB and stain mixtures.
  - Easiest remaining route: Audit/extend existing native implementation: Compose native stain separation with fixed HED matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2hsv"></a>
- [ ] **`color.rgb2hsv`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: equal-max ties, zero delta/value and hue wrapping. Cases to close: tied channels, black and six hue sectors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native max/min sector conversion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2lab"></a>
- [ ] **`color.rgb2lab`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, illuminant='D65', observer='2', *, channel_axis=-1)`.
  - Required focus: conversion range, illuminant/observer and promotion. Cases to close: low/high transfer-function branches and whitepoints.
  - Delivered qualification covers the current packed-byte and float dispatch. On x86-v4, the shipped correction’s measured float32 and float64 cases were below one-thread parity. The later wide float64 experiment failed the same-host speed check and remains private. [Shipped source timing scope](https://github.com/bipxx/scikit-imagexx/actions/runs/44226093016) · [Same-host result](https://github.com/bipxx/scikit-imagexx/actions/runs/44244965094)
  - Easiest remaining route: Audit/extend existing native implementation: Compose native sRGB-to-XYZ and XYZ-to-Lab.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2luv"></a>
- [ ] **`color.rgb2luv`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: dark pixels and chromaticity denominator. Cases to close: black, primaries and near-black RGB.
  - Easiest remaining route: Audit/extend existing native implementation: Compose native sRGB-to-XYZ then XYZ-to-Luv.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2rgbcie"></a>
- [ ] **`color.rgb2rgbcie`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization, coefficient order, channel_axis and dtype. Cases to close: basis/negative RGB and strided channel layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native forward CIE RGB linear color matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2xyz"></a>
- [ ] **`color.rgb2xyz`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization and transfer threshold. Cases to close: uint8/float RGB at transfer boundary.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native sRGB inverse transfer then XYZ matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2ycbcr"></a>
- [ ] **`color.rgb2ycbcr`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization, coefficient order, channel_axis and dtype. Cases to close: basis/negative RGB and strided channel layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native forward YCbCr matrix with encoded offsets/ranges.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2ydbdr"></a>
- [ ] **`color.rgb2ydbdr`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization, coefficient order, channel_axis and dtype. Cases to close: basis/negative RGB and strided channel layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native forward YDbDr luma/chroma matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2yiq"></a>
- [ ] **`color.rgb2yiq`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization, coefficient order, channel_axis and dtype. Cases to close: basis/negative RGB and strided channel layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native forward YIQ luma/in-phase/quadrature matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2ypbpr"></a>
- [ ] **`color.rgb2ypbpr`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization, coefficient order, channel_axis and dtype. Cases to close: basis/negative RGB and strided channel layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native forward YPbPr luma/color-difference matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgb2yuv"></a>
- [ ] **`color.rgb2yuv`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, *, channel_axis=-1)`.
  - Required focus: integer normalization, coefficient order, channel_axis and dtype. Cases to close: basis/negative RGB and strided channel layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native forward YUV luma/chroma matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgba2rgb"></a>
- [ ] **`color.rgba2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgba, background=(1, 1, 1), *, channel_axis=-1)`.
  - Required focus: alpha integer scaling, invalid background, channel axis. Cases to close: opaque/transparent/mixed alpha and broadcast background.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native alpha blend against background.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-rgbcie2rgb"></a>
- [ ] **`color.rgbcie2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgbcie, *, channel_axis=-1)`.
  - Required focus: inverse offsets/scaling, out-of-range behavior, channel_axis. Cases to close: encoded basis/offset boundary colors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse CIE RGB linear color matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-separate-stains"></a>
- [ ] **`color.separate_stains`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(rgb, conv_matrix, *, channel_axis=-1)`.
  - Required focus: matrix K, log floor, float dtype and nonnegative output. Cases to close: 3xK matrices and black/negative input.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native clamped log transform and matrix product.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-xyz2lab"></a>
- [ ] **`color.xyz2lab`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(xyz, illuminant='D65', observer='2', *, channel_axis=-1)`.
  - Required focus: exact piecewise threshold, illuminant/observer and negative XYZ. Cases to close: threshold-adjacent values for every whitepoint.
  - The packed uint8 case passed the measured x86-v4 one-thread speed gate. The measured float32 and float64 cases remain below parity for the shipped source. [Measured cases](https://github.com/bipxx/scikit-imagexx/actions/runs/44226093016) · [Exact release](https://github.com/bipxx/scikit-imagexx/actions/runs/44245395473)
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native whitepoint normalization and piecewise Lab transform.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-xyz2luv"></a>
- [ ] **`color.xyz2luv`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(xyz, illuminant='D65', observer='2', *, channel_axis=-1)`.
  - Required focus: zero XYZ sum, whitepoint and threshold branch. Cases to close: black, cancelling XYZ and observer combinations.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native lightness and chromaticity transform.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-xyz2rgb"></a>
- [ ] **`color.xyz2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(xyz, *, channel_axis=-1)`.
  - Required focus: negative values, transfer threshold and final clipping. Cases to close: out-of-gamut and threshold-adjacent XYZ.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse XYZ matrix and piecewise sRGB transfer.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-xyz-tristimulus-values"></a>
- [ ] **`color.xyz_tristimulus_values`** — full contract closed.
  - Main: `metadata`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(*, illuminant, observer, dtype=<class 'float'>)`.
  - Required focus: lookup keys, invalid illuminant/observer and returned dtype/copy. Cases to close: all table keys plus invalid keys.
  - Easiest remaining route: Reuse owned illuminant/observer table and dtype cast.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-ycbcr2rgb"></a>
- [ ] **`color.ycbcr2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ycbcr, *, channel_axis=-1)`.
  - Required focus: inverse offsets/scaling, out-of-range behavior, channel_axis. Cases to close: encoded basis/offset boundary colors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse YCbCr matrix with encoded offsets/ranges.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-ydbdr2rgb"></a>
- [ ] **`color.ydbdr2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ydbdr, *, channel_axis=-1)`.
  - Required focus: inverse offsets/scaling, out-of-range behavior, channel_axis. Cases to close: encoded basis/offset boundary colors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse YDbDr luma/chroma matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-yiq2rgb"></a>
- [ ] **`color.yiq2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(yiq, *, channel_axis=-1)`.
  - Required focus: inverse offsets/scaling, out-of-range behavior, channel_axis. Cases to close: encoded basis/offset boundary colors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse YIQ luma/in-phase/quadrature matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-ypbpr2rgb"></a>
- [ ] **`color.ypbpr2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ypbpr, *, channel_axis=-1)`.
  - Required focus: inverse offsets/scaling, out-of-range behavior, channel_axis. Cases to close: encoded basis/offset boundary colors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse YPbPr luma/color-difference matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-color-yuv2rgb"></a>
- [ ] **`color.yuv2rgb`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(yuv, *, channel_axis=-1)`.
  - Required focus: inverse offsets/scaling, out-of-range behavior, channel_axis. Cases to close: encoded basis/offset boundary colors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native inverse YUV luma/chroma matrix.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

</details>

<a id="module-data"></a>
### data — 41 public keys

<details><summary>Expand data contract checklist</summary>

<a id="api-data-astronaut"></a>
- [ ] **`data.astronaut`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB astronaut image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/astronaut.png; preserve RGB astronaut image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-binary-blobs"></a>
- [ ] **`data.binary_blobs`** — full contract closed.
  - Current scoped progress: Private `68c87325` passed 36 checks each with scalar and AVX2 dispatch, including seeded RNG state. Two fixed one-worker cases passed and six failed. The two-case diagnostic attributed the largest excess to Gaussian smoothing. Successor `13a2b5e0` passed all eight private timings at 1.335–2.579× and 262 checks per dispatch mode. Its shared Gaussian and installed release gates remain pending. [Successor admission](../build/evidence/u182-binary-blobs-speed/root-private-admission-v1.json) [Private parity and timings](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json) · [Attribution plan](../build/evidence/u182-binary-blobs/phase-attribution-plan-v1.md)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(length=512, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.5, rng=None, *, boundary_mode='nearest')`.
  - Required focus: RNG consumption, volume_fraction ties, n_dim, blob_size_fraction and boundary_mode. Cases to close: seeded 2D/3D fields at extreme volume fractions.
  - Easiest remaining route: Preserve upstream random point sampling then Gaussian smoothing and percentile threshold.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-brain"></a>
- [ ] **`data.brain`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; multipage brain volume with preserved axes. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/brain.tiff; preserve multipage brain volume with preserved axes.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-brick"></a>
- [ ] **`data.brick`** — full contract closed.
  - Current scoped progress: Private `e48a1aa9` passed exact warm-local pixels, layout and mutability on both retained profiles. Both fixed warm-local timing cases passed. Missing or corrupt resources use the compatibility route. Installed wheel and release are pending. [Private root admission](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/brick.png; preserve as_gray conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-camera"></a>
- [ ] **`data.camera`** — full contract closed.
  - Main: `packaged-asset-partial`; gate status `C~ N~ H? P~ S? V? W? B~ R~`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale camera image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/camera.png; preserve grayscale camera image.
  - Evidence locators: [tests/test_data_camera_coffee_public.py](../tests/test_data_camera_coffee_public.py) · [benchmarks/data_camera_coffee.py](../benchmarks/data_camera_coffee.py).
  - Private camera/coffee source `fee98881` has exact warm original PNG and packaged bytes. Windows and WSL minimum passed the warm-local measurement. Shared package, missing/corrupt path and release gates remain open. [Private speed](../build/evidence/u182-next-asset-provenance/camera-coffee-eagle-gate-v1/timing-root-admission-v3.json)
  - Candidate `7f8d4c5a` passed its selected installed parity and one-worker comparisons. Bundled camera bytes, fresh writable output, and the scoped warm public decode comparison. The corrected product has identical installed bytes and is in four-profile qualification. [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45058138452) · [Parity](../build/evidence/u182-post-haar/installed-focused-root-admission-v1.json) · [Speed](../build/evidence/u182-post-haar/installed-six-timing-root-admission-v1.json)
  - Packaged local decode and scoped warm public speed shipped. Full missing, corruption and provider behavior remains under review. [Verified main delivery](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/root-main-delivery-v1.json) · [Whole-product evidence](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/whole-product-root-admission-v1.json)

<a id="api-data-cat"></a>
- [ ] **`data.cat`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Canonical alias: `data.chelsea`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: asset hash/license/cache; RGB cat image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Exact alias of data.chelsea; reuse its route: Manifest-backed fetch/decode of data/chelsea.png; preserve RGB cat image. No extra algorithm credit..
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - The private candidate preserves callable identity with `data.chelsea`. The alias passed both fixed profile comparisons, and installed identity and package delivery remain pending. [Private admission](../build/evidence/u182-chelsea-cat/root-private-admission-v1.json)

<a id="api-data-cell"></a>
- [ ] **`data.cell`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale cell image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/cell.png; preserve grayscale cell image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-cells3d"></a>
- [ ] **`data.cells3d`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; multichannel z-stack axes/dtype. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/cells3d.tif; preserve multichannel z-stack axes/dtype.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-checkerboard"></a>
- [ ] **`data.checkerboard`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale checkerboard pixels. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/chessboard_GRAY.png; preserve grayscale checkerboard pixels.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-chelsea"></a>
- [ ] **`data.chelsea`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB cat image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/chelsea.png; preserve RGB cat image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private parity verified fresh writable RGB arrays and a 405,900-byte CC0 resource. The fixed local comparisons passed on Windows and Linux. Actual wheel resource closure and installed qualification remain pending. [Private admission](../build/evidence/u182-chelsea-cat/root-private-admission-v1.json)

<a id="api-data-clock"></a>
- [ ] **`data.clock`** — full contract closed.
  - Current scoped progress: Both retained scikit-image 0.26 packages matched the pinned source PNG and asset-level redistribution evidence. Private `3446ec38` passed exact output and fault checks on both profiles. All four fixed private speed cases across the two functions passed. Installed wheel and release remain pending. [Timed private handoff](../build/evidence/u182-clock-horse/handoff-v2.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; motion-blurred clock image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/clock_motion.png; preserve motion-blurred clock image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-coffee"></a>
- [ ] **`data.coffee`** — full contract closed.
  - Main: `packaged-asset-partial`; gate status `C~ N~ H? P~ S? V? W? B~ R~`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB coffee image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/coffee.png; preserve RGB coffee image.
  - Evidence locators: [tests/test_data_camera_coffee_public.py](../tests/test_data_camera_coffee_public.py) · [benchmarks/data_camera_coffee.py](../benchmarks/data_camera_coffee.py).
  - Private camera/coffee source `fee98881` preserves the exact RGB result. Windows and WSL minimum passed the warm-local measurement. Shared package and release remain open. [Private speed](../build/evidence/u182-next-asset-provenance/camera-coffee-eagle-gate-v1/timing-root-admission-v3.json)
  - Candidate `7f8d4c5a` passed its selected installed parity and one-worker comparisons. Bundled coffee bytes, fresh writable output, and the scoped warm public decode comparison. The corrected product has identical installed bytes and is in four-profile qualification. [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45058138452) · [Parity](../build/evidence/u182-post-haar/installed-focused-root-admission-v1.json) · [Speed](../build/evidence/u182-post-haar/installed-six-timing-root-admission-v1.json)
  - Packaged local decode and scoped warm public speed shipped. Full missing, corruption and provider behavior remains under review. [Verified main delivery](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/root-main-delivery-v1.json) · [Whole-product evidence](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/whole-product-root-admission-v1.json)

<a id="api-data-coins"></a>
- [ ] **`data.coins`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale coins image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/coins.png; preserve grayscale coins image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-colorwheel"></a>
- [ ] **`data.colorwheel`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; color wheel channels/alpha contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/color.png; preserve color wheel channels/alpha contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-download-all"></a>
- [ ] **`data.download_all`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(directory=None)`.
  - Required focus: directory override, partial cache, offline failure, license and archive expansion. Cases to close: prepopulated/partial/offline cache fixtures.
  - Easiest remaining route: Enumerate pinned registry and fetch/cache every declared resource with hash validation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-eagle"></a>
- [ ] **`data.eagle`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; eagle image decoded dtype/channels. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/eagle.png; preserve eagle image decoded dtype/channels.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private source `829846e4` retains exact decoded bytes. Warm-original timing requires the authentic compressed PNG. [Asset provenance](../build/evidence/u182-next-asset-provenance/camera-coffee-eagle-gate-v1/handoff-v1.json)

<a id="api-data-file-hash"></a>
- [ ] **`data.file_hash`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(fname, alg='sha256')`.
  - Required focus: algorithm names, missing/unreadable file and exact hexadecimal result. Cases to close: empty/small/large local files for supported algorithms.
  - Easiest remaining route: Stream file bytes through standard cryptographic hash provider in bounded chunks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Pinned Windows and Linux profiles each recorded seven digest returns and ten errors. Both private provider routes passed the reference cases. Linux preserves direct Pooch callable identity. Two Windows one-worker medians fell below parity, so release remains pending. [Private handoff](../build/evidence/u182-data-file-hash/handoff-v2.json) · [Linux timing](../build/evidence/u182-data-file-hash/linux-timing-v2.json)

<a id="api-data-grass"></a>
- [ ] **`data.grass`** — full contract closed.
  - Current scoped progress: Private `e48a1aa9` passed exact warm-local pixels, layout and mutability on both retained profiles. Both fixed warm-local timing cases passed. Missing or corrupt resources use the compatibility route. Installed wheel and release are pending. [Private root admission](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/grass.png; preserve as_gray conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-gravel"></a>
- [ ] **`data.gravel`** — full contract closed.
  - Current scoped progress: Private `e48a1aa9` passed exact warm-local pixels, layout and mutability on both retained profiles. Both fixed warm-local timing cases passed. Missing or corrupt resources use the compatibility route. Installed wheel and release are pending. [Private root admission](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/gravel.png; preserve as_gray conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-horse"></a>
- [ ] **`data.horse`** — full contract closed.
  - Current scoped progress: Both retained scikit-image 0.26 packages matched the pinned source PNG and asset-level redistribution evidence. Private `3446ec38` passed exact output and fault checks on both profiles. All four fixed private speed cases across the two functions passed. Installed wheel and release remain pending. [Timed private handoff](../build/evidence/u182-clock-horse/handoff-v2.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray then img_as_bool conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/horse.png; preserve as_gray then img_as_bool conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-hubble-deep-field"></a>
- [ ] **`data.hubble_deep_field`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB JPEG decode contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/hubble_deep_field.jpg; preserve RGB JPEG decode contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-human-mitosis"></a>
- [ ] **`data.human_mitosis`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; microscopy TIFF axes/dtype. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/mitosis.tif; preserve microscopy TIFF axes/dtype.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-immunohistochemistry"></a>
- [ ] **`data.immunohistochemistry`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB tissue image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/ihc.png; preserve RGB tissue image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-kidney"></a>
- [ ] **`data.kidney`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; multichannel volume axes/dtype. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/kidney.tif; preserve multichannel volume axes/dtype.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-lbp-frontal-face-cascade-filename"></a>
- [ ] **`data.lbp_frontal_face_cascade_filename`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: path existence/type, cache/offline behavior and model license. Cases to close: cached XML path and missing-resource fixture.
  - Easiest remaining route: Resolve manifest entry data/lbpcascade_frontalface_opencv.xml and return verified local path.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-lfw-subset"></a>
- [ ] **`data.lfw_subset`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; NPY array dtype/shape and byte order. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/lfw_subset.npy; preserve NPY array dtype/shape and byte order.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-lily"></a>
- [ ] **`data.lily`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; multichannel microscopy axes/dtype. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/lily.tif; preserve multichannel microscopy axes/dtype.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-logo"></a>
- [ ] **`data.logo`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; logo alpha/channel contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/logo.png; preserve logo alpha/channel contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-microaneurysms"></a>
- [ ] **`data.microaneurysms`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; retinal grayscale image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/microaneurysms.png; preserve retinal grayscale image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-moon"></a>
- [ ] **`data.moon`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale moon image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/moon.png; preserve grayscale moon image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-nickel-solidification"></a>
- [ ] **`data.nickel_solidification`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; time/volume frame ordering. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/solidification.tif; preserve time/volume frame ordering.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-page"></a>
- [ ] **`data.page`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale document image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/page.png; preserve grayscale document image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-palisades-of-vogt"></a>
- [ ] **`data.palisades_of_vogt`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; TIFF shape/channel contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/palisades_of_vogt.tif; preserve TIFF shape/channel contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-protein-transport"></a>
- [ ] **`data.protein_transport`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; time/volume frame ordering. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/protein_transport.tif; preserve time/volume frame ordering.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-retina"></a>
- [ ] **`data.retina`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB JPEG decode contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/retina.jpg; preserve RGB JPEG decode contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-rocket"></a>
- [ ] **`data.rocket`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB JPEG decode contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/rocket.jpg; preserve RGB JPEG decode contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-shepp-logan-phantom"></a>
- [ ] **`data.shepp_logan_phantom`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray phantom intensity conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/phantom.png; preserve as_gray phantom intensity conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-skin"></a>
- [ ] **`data.skin`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB JPEG decode contract. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/skin.jpg; preserve RGB JPEG decode contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-stereo-motorcycle"></a>
- [ ] **`data.stereo_motorcycle`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; ordered left/right/disparity tuple including invalid disparity values. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/motorcycle_left.png + motorcycle_right.png + motorcycle_disp.npz; preserve ordered left/right/disparity tuple including invalid disparity values.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-text"></a>
- [ ] **`data.text`** — full contract closed.
  - Current scoped progress: Private `e48a1aa9` passed exact warm-local pixels, layout and mutability on both retained profiles. Both fixed warm-local timing cases passed. Missing or corrupt resources use the compatibility route. Installed wheel and release are pending. [Private root admission](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale texture image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/text.png; preserve grayscale texture image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-vortex"></a>
- [ ] **`data.vortex`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; ordered pair of frames. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/pivchallenge-B-B001_1.tif + pivchallenge-B-B001_2.tif; preserve ordered pair of frames.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-draw"></a>
### draw — 18 public keys

<details><summary>Expand draw contract checklist</summary>

<a id="api-draw-bezier-curve"></a>
- [ ] **`draw.bezier_curve`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r0, c0, r1, c1, r2, c2, weight, shape=None)`.
  - Required focus: control-point/weight degeneracy, output order, clipping. Cases to close: straight/curved/degenerate weighted arcs.
  - Easiest remaining route: Audit/extend existing native implementation: Port rational quadratic Bezier integer raster recurrence with weight.
  - Evidence locators: parity [tests/test_bezier_curve_public.py::test_bezier_saved_public](../tests/test_bezier_curve_public.py); benchmark [benchmarks/bezier_curve.py](../benchmarks/bezier_curve.py).

<a id="api-draw-circle-perimeter"></a>
- [ ] **`draw.circle_perimeter`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r, c, radius, method='bresenham', shape=None)`.
  - Required focus: method-specific duplicates/order, radius zero and clipping. Cases to close: radius sweep and off-image circles.
  - Easiest remaining route: Audit/extend existing native implementation: Port selected Bresenham or Andres integer circle recurrence.
  - Evidence locators: parity [tests/test_circle_perimeter_public.py::test_circle_perimeter_saved_public_contract](../tests/test_circle_perimeter_public.py); benchmark [benchmarks/circle_perimeter.py](../benchmarks/circle_perimeter.py).

<a id="api-draw-circle-perimeter-aa"></a>
- [ ] **`draw.circle_perimeter_aa`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r, c, radius, shape=None)`.
  - Required focus: coordinate duplicates/order, intensity weights and radius zero. Cases to close: tiny/large subpixel coverage circles.
  - Easiest remaining route: Audit/extend existing native implementation: Port weighted anti-aliased circle recurrence.
  - Evidence locators: parity [tests/test_circle_perimeter_aa_public.py::test_circle_perimeter_aa_saved_public_pipeline](../tests/test_circle_perimeter_aa_public.py); benchmark [benchmarks/circle_perimeter_aa.py](../benchmarks/circle_perimeter_aa.py).

<a id="api-draw-disk"></a>
- [ ] **`draw.disk`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(center, radius, *, shape=None)`.
  - Required focus: center floats, strict radius edge and shape clipping. Cases to close: fractional centers and boundary radii.
  - Easiest remaining route: Audit/extend existing native implementation: Generate bounding-box pixels satisfying the radial interior test.
  - Evidence locators: parity [tests/test_draw_coordinates_integration.py::test_draw_captured_public_pipeline](../tests/test_draw_coordinates_integration.py); benchmark [benchmarks/draw_coordinates.py](../benchmarks/draw_coordinates.py).

<a id="api-draw-ellipse"></a>
- [ ] **`draw.ellipse`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r, c, r_radius, c_radius, shape=None, rotation=0.0)`.
  - Required focus: rotation, unequal radii, degenerate radii and clipping. Cases to close: rotated thin/large ellipses.
  - Easiest remaining route: Audit/extend existing native implementation: Rotate bounding-box coordinates and test normalized quadratic interior.
  - Evidence locators: parity [tests/test_draw_coordinates_integration.py::test_draw_captured_public_pipeline](../tests/test_draw_coordinates_integration.py); benchmark [benchmarks/draw_coordinates.py](../benchmarks/draw_coordinates.py).

<a id="api-draw-ellipse-perimeter"></a>
- [ ] **`draw.ellipse_perimeter`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r, c, r_radius, c_radius, orientation=0, shape=None)`.
  - Required focus: orientation rounding, degeneracy, duplicates and clipping. Cases to close: axis-aligned/rotated outlines.
  - Easiest remaining route: Audit/extend existing native implementation: Port ordered integer ellipse outline recurrence with orientation.
  - Evidence locators: parity [tests/test_draw_ellipse_perimeter_public.py::test_ellipse_perimeter_saved_public_pipeline](../tests/test_draw_ellipse_perimeter_public.py); benchmark [benchmarks/ellipse_perimeter.py](../benchmarks/ellipse_perimeter.py).

<a id="api-draw-ellipsoid"></a>
- [ ] **`draw.ellipsoid`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False)`.
  - Required focus: grid extents/spacing, boolean versus levelset output. Cases to close: anisotropic ellipsoids and degenerate dimensions.
  - Easiest remaining route: Audit/extend existing native implementation: Evaluate scaled 3D quadratic field on exact spacing-derived grid.
  - Evidence locators: parity [tests/test_draw_ellipsoid_public.py::test_ellipsoid_saved_public_pipeline](../tests/test_draw_ellipsoid_public.py); benchmark [benchmarks/draw_ellipsoid.py](../benchmarks/draw_ellipsoid.py).

<a id="api-draw-ellipsoid-stats"></a>
- [ ] **`draw.ellipsoid_stats`** — full contract closed.
  - Current scoped progress: Candidate `8deb0d48` reuses the admitted minimum installed public and SDK product. Fixed one-worker speed passed for six fixed cases at 4.456–5.369× across bounded positive axes. The general surface formula uses the declared SciPy special-function provider. Exact hosted qualification is underway. [Installed admission](../build/evidence/u182-first-integration/minimum-local-root-admission-v1.json) · [Timing](../build/evidence/u182-first-integration/installed-speed-root-admission-v1.json)
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(a, b, c)`.
  - Required focus: sphere/spheroid limits, axis ordering and special-function accuracy. Cases to close: spheres, oblate/prolate and triaxial cases.
  - Easiest remaining route: Direct volume formula plus reviewed elliptic-integral surface-area implementation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Original-only evidence covers 73 cases on each retained Linux profile, including radius permutations, scalar promotion and exceptional warnings. Native parity, installed SDK, timing and release are pending. [Minimum observations](../build/evidence/u182-ellipsoid-stats/original-minimum-v1.json) · [Modern observations](../build/evidence/u182-ellipsoid-stats/original-modern-v1.json)

<a id="api-draw-line"></a>
- [ ] **`draw.line`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r0, c0, r1, c1)`.
  - Required focus: octants, reversed endpoints, degenerate point. Cases to close: all octants and long diagonals.
  - Easiest remaining route: Audit/extend existing native implementation: Port Bresenham integer line with exact endpoint order.
  - Evidence locators: parity [tests/test_draw_coordinates_integration.py::test_draw_captured_public_pipeline](../tests/test_draw_coordinates_integration.py); benchmark [benchmarks/draw_coordinates.py](../benchmarks/draw_coordinates.py).

<a id="api-draw-line-aa"></a>
- [ ] **`draw.line_aa`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r0, c0, r1, c1)`.
  - Required focus: octant symmetry, duplicate/order and endpoint intensities. Cases to close: all octants and one-pixel segments.
  - Easiest remaining route: Audit/extend existing native implementation: Port anti-aliased line coverage recurrence.
  - Evidence locators: parity [tests/test_line_aa_public.py::test_line_aa_saved_public_pipeline](../tests/test_line_aa_public.py); benchmark [benchmarks/line_aa.py](../benchmarks/line_aa.py).

<a id="api-draw-line-nd"></a>
- [ ] **`draw.line_nd`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float16, float32, float64, int8, uint8, int64, uint64`.
  - Signature: `(start, stop, *, endpoint=False, integer=True)`.
  - Required focus: endpoint flag, profile-specific f16/f32 half ties, original-dtype count wrap, scalar/provider frontiers and K/F integer backing. Cases to close: existing frozen54 rows plus promotion supplement; unassessed mixed/0D typed scalar domains.
  - Easiest remaining route: Audit/extend existing native implementation: Extend reviewed typed count/interpolation/rounding with runtime scalar-modulo policy, preserve shared row-view base.
  - Evidence locators: parity [tests/test_line_nd_public.py::test_line_nd_saved_public_pipeline](../tests/test_line_nd_public.py); benchmark [benchmarks/line_nd.py](../benchmarks/line_nd.py).

<a id="api-draw-polygon"></a>
- [ ] **`draw.polygon`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(r, c, shape=None)`.
  - Required focus: self-intersection, concavity, edge inclusion and shape clipping. Cases to close: concave/self-touching/out-of-bounds polygons.
  - Easiest remaining route: Audit/extend existing native implementation: Scan bounding box with exact polygon inclusion/edge rule.
  - Evidence locators: parity [tests/test_draw_polygon_public.py::test_draw_polygon_saved_public_pipeline](../tests/test_draw_polygon_public.py); benchmark [benchmarks/draw_polygon.py](../benchmarks/draw_polygon.py).

<a id="api-draw-polygon2mask"></a>
- [ ] **`draw.polygon2mask`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image_shape, polygon)`.
  - Required focus: image_shape and polygon coordinate convention. Cases to close: fractional vertices and clipped polygons.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse polygon fill coordinates and scatter into boolean mask.
  - Evidence locators: parity [tests/test_draw_polygon_public.py::test_draw_polygon_saved_public_pipeline](../tests/test_draw_polygon_public.py); benchmark [benchmarks/draw_polygon.py](../benchmarks/draw_polygon.py).

<a id="api-draw-polygon-perimeter"></a>
- [ ] **`draw.polygon_perimeter`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(r, c, shape=None, clip=False)`.
  - Required focus: closure, duplicate vertices, clipping versus shape filtering. Cases to close: open inputs, repeated vertices and clipped outlines.
  - Easiest remaining route: Clip polygon when requested then rasterize ordered edges.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-draw-random-shapes"></a>
- [ ] **`draw.random_shapes`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `uint8`.
  - Shipped on `a80be034` for the documented seeded routes. Shared installed public tests and direct C++ SDK passed. Wider modes or choices retain compatibility behavior. [Qualification](../build/evidence/u182-post-haar/hosted-current301-45090228543/whole-product-root-admission-v1.json) [Fixed benchmark](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)
  - Signature: `(image_shape, max_shapes, min_shapes=1, min_size=2, max_size=None, num_channels=3, shape=None, intensity_range=None, allow_overlap=False, num_trials=100, rng=None, *, channel_axis=-1)`.
  - Required focus: num_trials, overlap, intensity_range, channel_axis and RNG draw order. Cases to close: seeded mixed-shape crowded and sparse canvases.
  - Shipped scope: The uint8 descriptor carries only image shape. NumPy supplies RNG and ordered metadata. Native raster and paint stages produce a uint8 image. Unsupported carriers and options use compatibility.
  - Evidence locators: parity [tests/test_random_shapes_public.py](../tests/test_random_shapes_public.py); benchmark [benchmarks/random_shapes.py](../benchmarks/random_shapes.py).
  - Private `ce878576` matches fixed seed, warning, image and box order, and caller PCG64 state on Windows and WSL minimum in both modes. Standalone SDK and all eight fixed speed checks passed. Shared install and release remain open. [Private contract admission](../build/evidence/u182-random-shapes/private-component-root-admission-v1.json) · [Random-shapes speed](../build/evidence/u182-random-shapes/timing-v1/timing-root-admission-v2.json)
  - Retained Linux modern stock matched every Windows row. The qualified AxisError name differs from minimum. Installed hosted 3.13.16 scope remains open. [Modern reference admission](../build/evidence/u182-random-shapes/modern-stock-root-admission-v1.json)
  - Corrected private `66a2d76f` matched 48 retained Linux modern outcomes and 38 RNG states across normal and forced-scalar modes. The four fixed private modern timings met parity at 1.13×–1.54×. Hosted CPython 3.13.16, installed SDK and installed speed remain pending. [Modern speed](../build/evidence/u182-post-haar/random-modern-timing-v1/root-modern-fixed-timing-admission-v1.json) · [Linux modern component](../build/evidence/u182-random-shapes/modern-component-v1/private-handoff-v1.json)
  - The private installed-scope tests verified real native dispatch and enabled stock fallback in both dispatch modes. Shared installed aliases, genuine hosted SDK and benchmark checks remain pending. [Fallback and RNG proof](../build/evidence/u182-post-haar/random-fallback-tests-v1/installed-family-test-handoff-v1.json)
  - Candidate eddec69f passed installed Windows parity in both modes. The fixed one-worker comparison was 1.322×–1.801× over four seeded calls. Four-profile release is pending. [Installed timing](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)

<a id="api-draw-rectangle"></a>
- [ ] **`draw.rectangle`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(start, end=None, extent=None, shape=None)`.
  - Required focus: end inclusivity, negative extent, shape clipping. Cases to close: reversed corners and partially clipped boxes.
  - Easiest remaining route: Audit/extend existing native implementation: Build coordinate mesh between normalized start/end or extent bounds.
  - Evidence locators: parity [tests/test_draw_coordinates_integration.py::test_draw_captured_public_pipeline](../tests/test_draw_coordinates_integration.py); benchmark [benchmarks/draw_coordinates.py](../benchmarks/draw_coordinates.py).

<a id="api-draw-rectangle-perimeter"></a>
- [ ] **`draw.rectangle_perimeter`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(start, end=None, extent=None, shape=None, clip=False)`.
  - Required focus: corner duplication, clip flag and off-image coordinates. Cases to close: tiny boxes and clipped outlines.
  - Easiest remaining route: Construct expanded border polygon and reuse perimeter rasterizer.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-draw-set-color"></a>
- [ ] **`draw.set_color`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float32, float64`.
  - Signature: `(image, coords, color, alpha=1)`.
  - Required focus: in-place mutation, duplicate coordinates/order, alpha broadcasting and dtype. Cases to close: repeated coordinates, scalar/vector alpha and RGB arrays.
  - Easiest remaining route: Audit/extend existing native implementation: Scatter blend color at supplied coordinates using native alpha arithmetic.
  - Evidence locators: parity [tests/test_draw_set_color_public.py::test_saved_native](../tests/test_draw_set_color_public.py); benchmark [benchmarks/draw_set_color.py](../benchmarks/draw_set_color.py).

</details>

<a id="module-exposure"></a>
### exposure — 10 public keys

<details><summary>Expand exposure contract checklist</summary>

<a id="api-exposure-adjust-gamma"></a>
- [ ] **`exposure.adjust_gamma`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, gamma=1, gain=1)`.
  - Required focus: nonnegative admission, gamma/gain, cast/round/saturation. Cases to close: uint8 LUT and floating extreme exponents.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native power-law correction and uint8 LUT.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-adjust-log"></a>
- [ ] **`exposure.adjust_log`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, gain=1, inv=False)`.
  - Required focus: inv branch, negative input and integer output conversion. Cases to close: forward/inverse integer and floating ramps.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native log1p/expm1 correction with input range scaling.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-adjust-sigmoid"></a>
- [ ] **`exposure.adjust_sigmoid`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, cutoff=0.5, gain=10, inv=False)`.
  - Required focus: gain/cutoff, inv branch, overflow and cast. Cases to close: near-cutoff ramps and extreme gain.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native logistic correction or inverse mapping.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-cumulative-distribution"></a>
- [ ] **`exposure.cumulative_distribution`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(image, nbins=256)`.
  - Required focus: integer versus float bins and empty/range behavior. Cases to close: sparse and dense histograms.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse histogram then exact-count cumulative normalization.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-equalize-adapthist"></a>
- [ ] **`exposure.equalize_adapthist`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, kernel_size=None, clip_limit=0.01, nbins=256)`.
  - Required focus: quantization, clip_limit, kernel_size, edge tiles and color conversion. Cases to close: tile-boundary/constant/noisy uint and float images.
  - Easiest remaining route: Audit/extend existing native implementation: Audit existing CLAHE tile histograms/clipping/interpolation and extend uncovered cases.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-equalize-hist"></a>
- [ ] **`exposure.equalize_hist`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(image, nbins=256, mask=None)`.
  - Required focus: masked CDF but full-image mapping, constant/empty mask and dtype. Cases to close: masked bimodal/constant images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse histogram/CDF interpolation with optional mask.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-histogram"></a>
- [ ] **`exposure.histogram`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(image, nbins=256, source_range='image', normalize=False, *, channel_axis=None)`.
  - Required focus: source_range, normalize, channel_axis, bin edges and large integers. Cases to close: all range/channel modes and wide sparse integers.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse typed dense integer and floating-bin histograms.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-is-low-contrast"></a>
- [ ] **`exposure.is_low_contrast`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, fraction_threshold=0.05, lower_percentile=1, upper_percentile=99, method='linear')`.
  - Required focus: percentile interpolation, channel conversion and constant/nonfinite values. Cases to close: near-decision-threshold and strided images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native percentile endpoints and dtype-range ratio.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-match-histograms"></a>
- [ ] **`exposure.match_histograms`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, reference, *, channel_axis=None)`.
  - Required focus: unequal shapes, channel axis, repeated values and dtype. Cases to close: unequal-resolution and low/high-cardinality images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse sorted-value/count CDF matching and interpolation.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-exposure-rescale-intensity"></a>
- [ ] **`exposure.rescale_intensity`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, in_range='image', out_range='dtype')`.
  - Required focus: dtype/string/range options, constant images, NaNs and clipping. Cases to close: constant/nonfinite and explicit input/output ranges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native min/max selection and affine range remap.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).
  - Candidate `4970cadc` matches the original bits in both dispatch modes for the 12 previously failing explicit-range cases and a larger worker-four tail case. V26 failed both complete source runs on historical MARK and active benchmark source guards. V28 passed both complete source checks and retains the exact rescale leaf. V28 passed exact installed Ubuntu parity in both modes and four-profile qualification. The one-thread speed check and delivery remain pending on the corrected producer. [Local arithmetic proof](../build/speed-parity-ci-correction-v1/release-entry-v1/main-ci-run44264846992-v1/exposure-rescale-scalar-v1/handoff-v1.json)

</details>

<a id="module-feature"></a>
### feature — 41 public keys

<details><summary>Expand feature contract checklist</summary>

<a id="api-feature-brief"></a>
- [ ] **`feature.BRIEF`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(descriptor_size=256, patch_size=49, mode='normal', sigma=1, rng=1)`.
  - Required focus: mode/rng pair generation, patch border mask, boolean descriptors and extract state. Cases to close: fixed keypoints near borders and seeded normal/uniform pairs.
  - Easiest remaining route: Own descriptor class using Gaussian preprocessing and seeded sampled-pair intensity comparisons.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-censure"></a>
- [ ] **`feature.CENSURE`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10)`.
  - Required focus: scale endpoints, filter normalization, subpixel/order and detect state. Cases to close: multiscale synthetic blobs and blank images.
  - Easiest remaining route: Scale-space box/octagon/star center-surround responses plus extrema/line suppression.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-cascade"></a>
- [ ] **`feature.Cascade`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: model format, stage threshold/eps, window ordering, detector state/pickle. Cases to close: pinned XML with empty/dense candidate windows.
  - Easiest remaining route: Parse licensed LBP cascade XML; integral-image weak classifiers and multiscale window scanning.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-orb"></a>
- [ ] **`feature.ORB`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(downscale=1.2, n_scales=8, n_keypoints=500, fast_n=9, fast_threshold=0.08, harris_k=0.04)`.
  - Required focus: keypoint ordering, scale mapping, border rejection and detect/extract consistency. Cases to close: detect/extract combined and separate on textured/blank images.
  - Easiest remaining route: Compose image pyramid, FAST, Harris ranking, orientation and rotated BRIEF with own class state.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-sift"></a>
- [ ] **`feature.SIFT`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5, c_dog=0.013333333333333334, c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8)`.
  - Required focus: octave/sigma schedule, contrast/edge rejection, descriptor quantization and mutable fields. Cases to close: scale/rotation scenes and low-contrast/border keypoints.
  - Easiest remaining route: Gaussian/DoG octaves, subpixel extrema, orientation histograms and normalized descriptors.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-blob-dog"></a>
- [ ] **`feature.blob_dog`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.5, overlap=0.5, *, threshold_rel=None, exclude_border=False)`.
  - Required focus: sigma_ratio, threshold_rel, border exclusion and pruning tie order. Cases to close: overlapping 2D/3D Gaussian blobs across scales.
  - Easiest remaining route: Compose Gaussian pyramid differences, N-D local extrema and overlap pruning.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-blob-doh"></a>
- [ ] **`feature.blob_doh`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False, *, threshold_rel=None)`.
  - Required focus: log_scale, approximate response normalization and pruning ties. Cases to close: small/large/overlapping blobs and scale endpoints.
  - Easiest remaining route: Use integral-image box Hessian determinant scale responses and overlap pruning.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-blob-log"></a>
- [ ] **`feature.blob_log`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5, log_scale=False, *, threshold_rel=None, exclude_border=False)`.
  - Required focus: scale normalization, log_scale, per-axis sigma and border rules. Cases to close: anisotropic 2D/3D blobs and threshold boundaries.
  - Easiest remaining route: Gaussian-Laplacian scale response plus N-D extrema and overlap pruning.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-canny"></a>
- [ ] **`feature.canny`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False, *, mode='constant', cval=0.0)`.
  - Required focus: quantile thresholds, mask erosion, interpolation ties and boundary modes. Cases to close: edge angles, broken/connected edges and masked boundaries.
  - Easiest remaining route: Gaussian smoothing, Sobel gradient, nonmaximum suppression and hysteresis connectivity.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-corner-fast"></a>
- [ ] **`feature.corner_fast`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, n=12, threshold=0.15)`.
  - Required focus: n segment length, wraparound runs, strict threshold and border zeros. Cases to close: bright/dark corners at exact threshold.
  - Easiest remaining route: Port FAST circle segment test with early rejection.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-corner-foerstner"></a>
- [ ] **`feature.corner_foerstner`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1)`.
  - Required focus: zero trace, sigma, dtype and two-result contract. Cases to close: flat/isotropic/edge patches.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse structure tensor then determinant/trace and roundness ratios.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-corner-harris"></a>
- [ ] **`feature.corner_harris`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, method='k', k=0.05, eps=1e-06, sigma=1)`.
  - Required focus: k versus eps method, negative determinant and promotion. Cases to close: corners/edges and both methods.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse structure tensor and selected Harris response formula.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-corner-kitchen-rosenfeld"></a>
- [ ] **`feature.corner_kitchen_rosenfeld`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mode='constant', cval=0)`.
  - Required focus: zero gradient, mode/cval and 2D admission. Cases to close: flat/ramp/corner images and borders.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native first/second derivatives and curvature response.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-corner-moravec"></a>
- [ ] **`feature.corner_moravec`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, window_size=1)`.
  - Required focus: window border, shift set and accumulation dtype. Cases to close: synthetic junctions and constant/strided images.
  - Easiest remaining route: Sliding shifted-patch squared-difference minima.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-corner-orientations"></a>
- [ ] **`feature.corner_orientations`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, corners, mask)`.
  - Required focus: mask oddness, border points, zero moments and angular convention. Cases to close: symmetric/asymmetric patches and border keypoints.
  - Easiest remaining route: Intensity centroid moments in each masked keypoint patch.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-corner-peaks"></a>
- [ ] **`feature.corner_peaks`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True, indices=True, num_peaks=inf, footprint=None, labels=None, *, num_peaks_per_label=inf, p_norm=inf)`.
  - Required focus: indices=False, labels, num_peaks_per_label, distance metric and stable ties. Cases to close: flat corner plateaus and per-label quotas.
  - Easiest remaining route: Compose peak candidates with corner-specific connected-plateau suppression.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-corner-shi-tomasi"></a>
- [ ] **`feature.corner_shi_tomasi`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1)`.
  - Required focus: symmetric eigenvalue order, tiny negative roundoff and sigma. Cases to close: rank-one/zero tensors and corners.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native structure tensor smallest eigenvalue.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-corner-subpix"></a>
- [ ] **`feature.corner_subpix`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, corners, window_size=11, alpha=0.99)`.
  - Required focus: alpha hypothesis tests, border NaNs, singular patches and coordinate order. Cases to close: ideal/noisy/edge-only patches and degenerate points.
  - Easiest remaining route: Fit local corner/edge intersection statistics with small linear solves.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-daisy"></a>
- [ ] **`feature.daisy`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, step=4, radius=15, rings=3, histograms=8, orientations=8, normalization='l1', sigmas=None, ring_radii=None, visualize=False)`.
  - Required focus: custom sigmas/ring radii, normalization, visualize output and channel geometry. Cases to close: dense descriptor grids with every normalization.
  - Easiest remaining route: Dense gradient orientation histograms sampled on concentric rings and normalized.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-draw-haar-like-feature"></a>
- [ ] **`feature.draw_haar_like_feature`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, uint8, float32, float64`.
  - Signature: `(image, r, c, width, height, feature_coord, color_positive_block=(1.0, 0.0, 0.0), color_negative_block=(0.0, 1.0, 0.0), alpha=0.5, max_n_features=None, rng=None)`.
  - Required focus: random feature selection state, signed colors, ROI clipping and copy. Cases to close: overlapping positive/negative blocks and seeded subsets.
  - Easiest remaining route: Reuse rectangle coordinate fill and alpha overlay for selected Haar blocks.
  - Evidence locators: [Focused parity](../tests/test_draw_haar_like_feature_public.py) · [Benchmark cases](../benchmarks/draw_haar_like_feature.py).
  - Private GNU and MSVC SDK checks and 79 public checks per Linux profile passed. The five one-worker cases passed at 14.86×–62.34×. NumPy RNG and unsupported-domain compatibility remain explicit. [Private admission](../build/evidence/u182-draw-haar/root-private-admission-v1.json)
  - Shipped scope `817b6f82` across Windows, Linux minimum, modern and debug. Five installed minimum cases passed at 16.358×–73.285×. NumPy supplies RNG for seeded selection. Other image and coordinate domains remain open. [Delivery](../build/evidence/u182-haar-features/hosted-45041856008/root-main-delivery-v1.json) · [Installed speed](../build/evidence/u182-haar-features/haar-294-installed-speed-root-admission-v2.json)

<a id="api-feature-draw-multiblock-lbp"></a>
- [ ] **`feature.draw_multiblock_lbp`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 4/4 cases >=1.0, 1.013-3.148x. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): 4/4 cases pass1.103–3.193x. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private speed candidate `36065c57` (2026-11-17): [all4 public cases meet parity,1.068–3.373x](../build/speed-parity-lbp-v1/root-performance-admission-iteration-2.json); [275 runtime checks/188 strict saved records](../build/speed-parity-lbp-v1/root-runtime-admission-iteration-2.json). Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1. Current genuine profiles/current-host release performance and delivery remain pending. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Latest scoped candidate (2026-11-15): [Exact16a5 qualification](evidence/mark-lbp16a5-whole-product-root-admission-v2.json), [named private SDK modes](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) and [four one-worker original-public cases](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json) pass their stated gates. All4 cases are18–48x slower; target FP words/source assertions passed but raw per-case32 payloads were not emitted. Full domain and shipping remain open; earlier failed campaigns below are historical.
  - Signature: `(image, r, c, width, height, lbp_code=0, color_greater_block=(1, 1, 1), color_less_block=(0, 0.69, 0.96), alpha=0.5)`.
  - Required focus: bit-to-neighbor orientation, center treatment and dtype/range. Cases to close: each individual bit and mixed block codes.
  - Easiest remaining route: Draw fixed nine-block LBP overlay through rectangle/alpha primitives.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Saved original admission](evidence/draw-multiblock-lbp-saved-raw-admission-v1.json) covers 94 cases per profile. [Independent private-component review](evidence/draw-multiblock-lbp-native-independent-review-v2.json) covers source 0e51dbc1e2c4741e243a91652d8c0b9771c96363: 1,128 target comparisons across two profiles, three scalar/SIMD modes and workers 1/4 pass. Of these, 1,048 complete records match and 80 differ only in post-return JSON collector MXCSR diagnostics, retained explicitly. Target FP and exact caller restoration remain strict. [Root scoped component admission](evidence/draw-multiblock-lbp-native-root-admission-v2.json) passes. [Private integration source/ABI review](evidence/draw-multiblock-lbp-private-integration-admission-v1.json) and [immutable handoff reconciliation](evidence/draw-multiblock-lbp-private-handoff-reconciliation-v1.json) pass in their scoped checks; the qualifier rejects the temporary conditional install rule, so genuine qualification is not passed. Normal enabled MARK/LBP integration is frozen at `2991881411972e6266f7e132c9dca8b7226a8567`; [root shared-source admission](evidence/mark-lbp-enabled-root-source-admission-v1.json) and [complete source barriers](evidence/mark-lbp-enabled-source-readiness-v1.json) pass (3,168 tests plus 8 subtests per retained Python). [Failed four-profile campaign admission](evidence/mark-lbp-enabled-failure-admission-v1.json) for299/run42280271105 preserves 956,246 XML checks with 48 identical LBP grayscale dtype-metadata failures (12 per profile), zero errors/skips, and 272 passing SDK checks. [Root-admitted binder correction](evidence/draw-multiblock-lbp-metadata-fix-admission-v1.json) at99d6c428 passes 1,128 saved target comparisons and 144 private metadata checks; it is entering shared integration. New genuine qualification and performance remain pending; no unchanged retry. Shared genuine core/installed SDK, full domains/platforms, one-thread original-public performance and delivery remain open. [Executable LBP benchmark source](evidence/lbp-benchmark-source-admission-v1.json) passes scoped independent review: 97 source/zero-body controls per Python, fixed four LBP cases and eight MARK cases through reused binding/F1/sampler logic. Actual genuine binding, outer cohorts and timing remain pending.
  - Current scoped progress (2026-11-15): [Root compiler-mode admission](evidence/lbp-entry-fp-private-modes-root-saved-admission-v1.json), source `9e15e991`, passes all94 saved recipes/profile with workers1/4 in true compiler-scalar and ASanUBSan LBP/crop/converter modes:752 strict full portable records,0formatter/sanitizer diagnostics, fullcaller/process32/default1. [Retained normal/runtime-disabled evidence](evidence/mark-lbp9e15-private-runtime-root-admission-v1.json) supplies752 separate entries in the same bounded domain. Ordinary ABI/thread/CPU/AVX2support is excluded from whole-mode credit. Test-only successor `16a5b501` preserves native bytes; its complete169 barriers and [genuine4profiles/3stagers](evidence/mark-lbp16a5-whole-product-root-admission-v2.json) passed. [Scoped installed SDK mode admission](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) and5new whole-owning FP witnesses passed, with startupdefault1/source-reset vs absent finalgetter limits explicit. Broader domains/platforms, four original-public performance cases and main delivery remainpending. Main stays upstream-only for this API.


<a id="api-feature-fisher-vector"></a>
- [ ] **`feature.fisher_vector`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(descriptors, gmm, *, improved=False, alpha=0.5)`.
  - Required focus: d_pi/d_mu/d_sigma scaling/order, covariance convention, alpha and zero normalization. Cases to close: hand-computed small GMM with K+2KD output and large descriptor batches.
  - Easiest remaining route: Read GMM probabilities/parameters through provider; compute mixture-weight, mean and variance gradients, concatenate K+2KD values, then optional power/L2 normalization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-graycomatrix"></a>
- [ ] **`feature.graycomatrix`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, distances, angles, levels=None, symmetric=False, normed=False)`.
  - Required focus: levels bounds, angle rounding, symmetric counts, normalize and output dtype. Cases to close: small exact matrices and high-level sparse textures.
  - Easiest remaining route: Count quantized intensity pairs at rounded distance/angle offsets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

  - Current scoped progress: [Root source/storage admission](evidence/graycomatrix-graycoprops-root-source-v1.json) covers the paired 94 prospective cases (50 matrix/44 properties), source custom rounding/count/reduction/validation order and bounded arrays/work. 202 source/stub checks passed per Python. [Corrected controller/readiness admission](evidence/graycomatrix-graycoprops-capture-root-v2.json), source56303, reuses8exact historical d263 prefixes without relabeling. [Minimum](evidence/graycomatrix-graycoprops-minimum-root-raw-v1.json) and [modern/paired](evidence/graycomatrix-graycoprops-modern-root-raw-v1.json) raw admissions cover94cases/profile (64returns/30errors/5warnings each), preserving typed metadata and376FPdifferences. Both original budgets are exhausted. [Private native/SDK/parity root admission](evidence/graycomatrix-graycoprops-native-root-admission-v1.json) now passes on de6 for the saved domain. Shared installation, broader domains and actual speed remain pending. Nonfinite/huge offsets and unbounded levels remain quarantined.

<a id="api-feature-graycoprops"></a>
- [ ] **`feature.graycoprops`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(P, prop='contrast')`.
  - Required focus: zero-variance correlation, normalization and property validation. Cases to close: constant/diagonal/asymmetric cooccurrence matrices.
  - Easiest remaining route: Reduce GLCM probabilities to named contrast/correlation/energy statistics.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

  - Current scoped progress: [Root source/storage admission](evidence/graycomatrix-graycoprops-root-source-v1.json) covers the paired 94 prospective cases (50 matrix/44 properties), source custom rounding/count/reduction/validation order and bounded arrays/work. 202 source/stub checks passed per Python. [Corrected controller/readiness admission](evidence/graycomatrix-graycoprops-capture-root-v2.json), source56303, reuses8exact historical d263 prefixes without relabeling. [Minimum](evidence/graycomatrix-graycoprops-minimum-root-raw-v1.json) and [modern/paired](evidence/graycomatrix-graycoprops-modern-root-raw-v1.json) raw admissions cover94cases/profile (64returns/30errors/5warnings each), preserving typed metadata and376FPdifferences. Both original budgets are exhausted. [Private native/SDK/parity root admission](evidence/graycomatrix-graycoprops-native-root-admission-v1.json) now passes on de6 for the saved domain. Shared installation, broader domains and actual speed remain pending. Nonfinite/huge offsets and unbounded levels remain quarantined.

<a id="api-feature-haar-like-feature"></a>
- [ ] **`feature.haar_like_feature`** — full contract closed.
  - Earlier private scope: Private `c5fc72c7` passed the safe supported ROI/coordinate cases on both retained profiles, with GNU and MSVC SDK smoke checks. All three private value timings passed. The shipped scope is recorded below. [Private root admission](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json)
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64`.
  - Signature: `(int_image, r, c, width, height, feature_type=None, feature_coord=None)`.
  - Required focus: feature ordering/types, ROI offsets and signed output overflow. Cases to close: all feature families and custom coordinates.
  - Easiest remaining route: Use integral-image rectangle inclusion-exclusion for supplied/enumerated Haar blocks.
  - Evidence locators: [Focused parity](../tests/test_haar_like_features_contract.py) · [Benchmark cases](../benchmarks/haar_like_features.py).
  - Shipped scope `817b6f82` across Windows, Linux minimum, modern and debug. Three installed minimum cases passed at 2.310×–108.450×. Exact supported integral-image ROI and coordinate paths shipped. Wider layouts and domains remain open. [Delivery](../build/evidence/u182-haar-features/hosted-45041856008/root-main-delivery-v1.json) · [Installed speed](../build/evidence/u182-haar-features/haar-294-installed-speed-root-admission-v2.json)

<a id="api-feature-haar-like-feature-coord"></a>
- [ ] **`feature.haar_like_feature_coord`** — full contract closed.
  - Earlier private scope: Private `c5fc72c7` passed the bounded coordinate ordering cases on both retained profiles. Both fixed private timings passed, with 1.010× on the larger case. The shipped scope is recorded below. [Private root admission](../build/evidence/u182-first-integration/parallel-private-root-review-v1.json)
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `uint8`.
  - Signature: `(width, height, feature_type=None)`.
  - Required focus: combinatorial ordering, feature_type selection and empty geometry. Cases to close: tiny exact enumeration and larger windows.
  - Easiest remaining route: Enumerate ordered rectangle tuples for each Haar feature family.
  - Evidence locators: [Focused parity](../tests/test_haar_like_features_contract.py) · [Benchmark cases](../benchmarks/haar_like_features.py).
  - Shipped scope `817b6f82` across Windows, Linux minimum, modern and debug. Two installed minimum cases passed at 1.026×–3.124×. Supported ordered coordinate enumeration shipped. Wider feature descriptors and geometry remain open. [Delivery](../build/evidence/u182-haar-features/hosted-45041856008/root-main-delivery-v1.json) · [Installed speed](../build/evidence/u182-haar-features/haar-294-installed-speed-root-admission-v2.json)

<a id="api-feature-hessian-matrix"></a>
- [ ] **`feature.hessian_matrix`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=None)`.
  - Required focus: order rc/xy, warning default, sigma axes and boundary modes. Cases to close: derivative-mode/order combinations in 2D/3D.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Gaussian-derivative or finite-difference Hessian paths.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-hessian-matrix-det"></a>
- [ ] **`feature.hessian_matrix_det`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1, approximate=True)`.
  - Required focus: approximate branch dimensions, sigma cutoff and dtype. Cases to close: exact/approximate response across sigma ranges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse exact determinant or integral-box approximation.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-hessian-matrix-eigvals"></a>
- [ ] **`feature.hessian_matrix_eigvals`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float32, float64`.
  - Signature: `(H_elems)`.
  - Required focus: 2D/3D tuple validation, order and repeated eigenvalues. Cases to close: repeated/ill-scaled symmetric tensors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native symmetric eigenvalue kernel over Hessian components.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-hog"></a>
- [ ] **`feature.hog`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, *, channel_axis=None)`.
  - Required focus: orientation bin ties, L2-Hys clipping, channel max ties, visualize and flatten. Cases to close: gradients at bin edges and multichannel cell grids.
  - Easiest remaining route: Gradient channels, orientation bins, cell histograms and block normalization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-learn-gmm"></a>
- [ ] **`feature.learn_gmm`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(descriptors, *, n_modes=32, gm_args=None)`.
  - Required focus: gm_args, model identity/state, initialization RNG, convergence warnings and covariance types. Cases to close: stub estimator call contract and fixed-seed fitted-model comparisons.
  - Easiest remaining route: Preserve optional GaussianMixture constructor/fit adapter; standalone alternative is C++ EM with reviewed covariance solvers.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-local-binary-pattern"></a>
- [ ] **`feature.local_binary_pattern`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, P, R, method='default')`.
  - Required focus: interpolation, float equality, rotation tie mapping and output dtype. Cases to close: constant/rotated patterns and every method.
  - Easiest remaining route: Bilinearly sample circular neighbors then encode default/ror/uniform/nri_uniform/var.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-match-descriptors"></a>
- [ ] **`feature.match_descriptors`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool`.
  - Shipped on `5b94b20d` within its recorded native domain. Windows installed fixed speed passed for the batch. [Whole-product admission](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/whole-product-root-admission-v1.json) [Windows one-worker evidence](../build/evidence/u182-post-haar/next-five-hosted-speed-v4/actual-result-v1/root-timing-admission-v1.json)
  - Signature: `(descriptors1, descriptors2, metric=None, p=2, max_distance=inf, cross_check=True, max_ratio=1.0)`.
  - Required focus: metric selection, equal-distance tie, zero denominator and empty descriptors. Cases to close: binary/float descriptors with duplicates and outliers.
  - Shipped scope: Two independent bounded C-order Boolean descriptor arrays, each with 1–1024 rows and a shared width of 1–256. Native ordered Hamming matches; equivalent and tagged descriptors preserve dispatch. Other metrics, layouts and parameter domains use compatibility.
  - Evidence locators: parity [tests/test_match_descriptors_public.py](../tests/test_match_descriptors_public.py); benchmark [benchmarks/match_descriptors.py](../benchmarks/match_descriptors.py).
  - All 42 retained Linux modern stock cases matched Windows and Linux minimum, including ordered results, descriptor markers, and three errors. The retained modern private product passed 86 checks and four fixed timings at 4.97×–10.81×. Genuine installed and release gates remain open. [Modern speed](../build/evidence/u182-post-haar/retained-modern-match-peaks-zhang-timing-root-admission-v1.json) · [Modern stock](../build/evidence/u182-next-numerical-lookahead/match-descriptors/modern-stock-handoff-v2.json)

<a id="api-feature-match-template"></a>
- [ ] **`feature.match_template`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, template, pad_input=False, mode='constant', constant_values=0)`.
  - Required focus: zero variance, pad_input origin, boundary mode and dimensionality. Cases to close: constant templates, odd/even shapes and boundary peaks.
  - Easiest remaining route: Start with direct correlation plus integral sum/squared-sum normalization; add an FFT backend only for measured large-template crossovers.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-multiblock-lbp"></a>
- [ ] **`feature.multiblock_lbp`** — full contract closed.
  - Current scoped progress: Source review identified the exact ordered float32 rectangle query and safe reference bounds. Private `34a9a5c4` passed 43 public checks on each retained profile and the GNU and MSVC staged SDK checks. Two float32 cases beat the original and both casting cases were 1.1% to 1.3% slower. The broad performance gate remains open. [Private timing closeout](../build/evidence/u182-multiblock-lbp/private-timing-closeout-v1.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(int_image, r, c, width, height)`.
  - Required focus: bit order, equal sums, ROI boundaries and integer promotion. Cases to close: exact 3x3 block patterns and large integral values.
  - Easiest remaining route: Reuse integral rectangle sums and compare eight blocks with center.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-multiscale-basic-features"></a>
- [ ] **`feature.multiscale_basic_features`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16, num_sigma=None, workers=None, *, channel_axis=None)`.
  - Required focus: scale/channel ordering, feature flags, sigma endpoints and workers semantics. Cases to close: channel/scale/feature combinations and non-last axes.
  - Easiest remaining route: Compose intensity, gradient magnitude and Hessian eigenvalue features at each Gaussian scale.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-peak-local-max"></a>
- [ ] **`feature.peak_local_max`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True, num_peaks=inf, footprint=None, labels=None, num_peaks_per_label=inf, p_norm=inf)`.
  - Required focus: plateau tie, exclude_border tuples, quotas, p_norm and inf distances. Cases to close: plateaus, per-label masks and close/equidistant peaks.
  - Easiest remaining route: Neighborhood maximum mask, threshold/label filtering and ordered spatial suppression.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-plot-matched-features"></a>
- [ ] **`feature.plot_matched_features`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0, image1, *, keypoints0, keypoints1, matches, ax, keypoints_color='k', matches_color=None, only_matches=False, alignment='horizontal')`.
  - Required focus: alignment, only_matches, colors, axes identity and artist ordering. Cases to close: horizontal/vertical layouts and unmatched points.
  - Easiest remaining route: Preserve artist adapter; reuse image composition and keypoint coordinate offsets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-shape-index"></a>
- [ ] **`feature.shape_index`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1, mode='constant', cval=0)`.
  - Required focus: zero curvatures, NaN convention and eigenvalue ordering. Cases to close: planes, saddles and spheres.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse Hessian eigenvalues and native curvature ratio angle.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-structure-tensor"></a>
- [ ] **`feature.structure_tensor`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1, mode='constant', cval=0, order='rc')`.
  - Required focus: component ordering, sigma, rc/xy and boundary mode. Cases to close: 2D/3D oriented ramps and borders.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native directional gradients then Gaussian product smoothing.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-feature-structure-tensor-eigenvalues"></a>
- [ ] **`feature.structure_tensor_eigenvalues`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float32, float64`.
  - Signature: `(A_elems)`.
  - Required focus: component count, tensor shape and repeated roots. Cases to close: isotropic/rank-deficient tensors.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native descending symmetric eigenvalue kernel.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

</details>

<a id="module-filters"></a>
### filters — 100 public keys

<details><summary>Expand filters contract checklist</summary>

<a id="api-filters-lpifilter2d"></a>
- [ ] **`filters.LPIFilter2D`** — full contract closed.
  - Main: `native-class-partial`; gate status `C? N~ H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(impulse_response, **filter_params)`.
  - Required focus: callable evaluation grid, parameter state, cache shape and real result. Cases to close: repeated same/different shapes and custom impulses.
  - Easiest remaining route: Own callable state/cache around sampled impulse response and padded FFT convolution.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark not registered in the shipping inventory.

<a id="api-filters-apply-hysteresis-threshold"></a>
- [ ] **`filters.apply_hysteresis_threshold`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, low, high)`.
  - Required focus: strict comparison, broadcast low/high, connectivity and nonfinite input. Cases to close: weak bridges, isolated strong pixels and array thresholds.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native low-mask labels and retain components touching high mask.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-butterworth"></a>
- [ ] **`filters.butterworth`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, cutoff_frequency_ratio=0.005, high_pass=True, order=2.0, channel_axis=None, *, squared_butterworth=True, npad=0)`.
  - Required focus: squared_butterworth, high_pass, npad, odd/even grids and channels. Cases to close: low/high pass, order/cutoff sweep and channel axes.
  - Easiest remaining route: Audit/extend existing native implementation: Construct N-D frequency-distance mask then FFT filter.
  - Evidence locators: parity [tests/test_butterworth_public.py](../tests/test_butterworth_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-correlate-sparse"></a>
- [ ] **`filters.correlate_sparse`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, kernel, mode='reflect')`.
  - Required focus: sparse zero taps, kernel center, N-D boundary modes and dtype. Cases to close: sparse/dense taps on 2D/3D strided arrays.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native nonzero-kernel tap traversal and direct strided correlation.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-difference-of-gaussians"></a>
- [ ] **`filters.difference_of_gaussians`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, low_sigma, high_sigma=None, *, mode='nearest', cval=0, channel_axis=None, truncate=4.0)`.
  - Required focus: per-axis low/high sigma validation, channel_axis, mode and preserve dtype. Cases to close: narrow/wide and anisotropic sigma pairs.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse two native Gaussian passes and subtraction.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-farid"></a>
- [ ] **`filters.farid`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None, *, axis=None, mode='reflect', cval=0.0)`.
  - Required focus: axis tuple, normalization, mask erosion and mode/cval. Cases to close: directional ramps and axis combinations.
  - The delivered scalar loop passed the scoped installed parity checks and all measured default Farid speed cases in the x86-v4 run. Broader mask, axis and layout coverage remains open. [Measured cases](https://github.com/bipxx/scikit-imagexx/actions/runs/44226093016) · [Exact release](https://github.com/bipxx/scikit-imagexx/actions/runs/44245395473)
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Farid five-tap derivative/smoothing filters; combine selected-axis squared responses.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-farid-h"></a>
- [ ] **`filters.farid_h`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, *, mask=None)`.
  - Required focus: exact row derivative orientation and 2D mask border. Cases to close: horizontal ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Farid five-tap derivative/smoothing filters along the horizontal-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-farid-v"></a>
- [ ] **`filters.farid_v`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, *, mask=None)`.
  - Required focus: exact column derivative orientation and 2D mask border. Cases to close: vertical ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Farid five-tap derivative/smoothing filters along the vertical-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-filter-forward"></a>
- [ ] **`filters.filter_forward`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(data, impulse_response=None, filter_params=None, predefined_filter=None)`.
  - Required focus: predefined_filter versus impulse_response, shape crop and normalization. Cases to close: impulse/delta and repeated filter use.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse LPIFilter2D transfer function and FFT forward convolution.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-filter-inverse"></a>
- [ ] **`filters.filter_inverse`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(data, impulse_response=None, filter_params=None, max_gain=2, predefined_filter=None)`.
  - Required focus: zero response, max_gain clipping and parameter precedence. Cases to close: near-zero spectral responses and gain sweep.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse transfer function with magnitude-limited inverse.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-frangi"></a>
- [ ] **`filters.frangi`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0)`.
  - Required focus: eigenvalue sorting/sign, gamma auto choice, scale deprecations and black_ridges. Cases to close: 2D/3D ridge polarity and repeated eigenvalues.
  - Easiest remaining route: Audit/extend existing native implementation: Audit existing native multiscale Hessian vesselness; retain measured precision/regression scopes.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-gabor"></a>
- [ ] **`filters.gabor`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0)`.
  - Required focus: frequency/theta/bandwidth, explicit sigma, mode/cval and output dtype. Cases to close: oriented frequencies and complex response borders.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse complex Gabor kernel then real/imag convolution.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-gabor-kernel"></a>
- [ ] **`filters.gabor_kernel`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3, offset=0, dtype=<class 'numpy.complex128'>)`.
  - Required focus: support rounding, theta/sigma, offset and complex dtype. Cases to close: frequency/support extremes and complex64/complex128.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native sampled complex Gaussian-envelope sinusoid.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-gaussian"></a>
- [ ] **`filters.gaussian`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigma=1.0, *, mode='nearest', cval=0, preserve_range=False, truncate=4.0, channel_axis=None, out=None)`.
  - Required focus: sigma axes, truncate radius, out aliasing, channel_axis and preserve_range. Cases to close: tiny/large sigma, all modes and out layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Extend existing native separable Gaussian derivative-free passes.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-hessian"></a>
- [ ] **`filters.hessian`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect', cval=0)`.
  - Required focus: nonpositive response replacement, gamma and ridge polarity. Cases to close: flat/background and multiscale vessels.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native hybrid Hessian vesselness and final background handling.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-laplace"></a>
- [ ] **`filters.laplace`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, ksize=3, mask=None)`.
  - Required focus: ksize, dimensionality, mask border and dtype. Cases to close: impulses, constant images and masked boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native N-D Laplacian taps and mask erosion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-median"></a>
- [ ] **`filters.median`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, mode='nearest', cval=0.0, behavior='ndimage')`.
  - Required focus: behavior, even footprints, out overlap, mode/cval and dtype. Cases to close: odd/even irregular footprints and all supported dtypes.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native neighborhood selection and rank/ndimage behavior routing.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-meijering"></a>
- [ ] **`filters.meijering`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0)`.
  - Required focus: alpha default by dimension, sign and per-scale normalization. Cases to close: 2D/3D neurites and constant inputs.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Hessian eigenvalue combination and scale maximum.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-prewitt"></a>
- [ ] **`filters.prewitt`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None, *, axis=None, mode='reflect', cval=0.0)`.
  - Required focus: axis tuple, normalization, mask erosion and mode/cval. Cases to close: directional ramps and axis combinations.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Prewitt derivative/box taps; combine selected-axis squared responses.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-prewitt-h"></a>
- [ ] **`filters.prewitt_h`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: exact row derivative orientation and 2D mask border. Cases to close: horizontal ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Prewitt derivative/box taps along the horizontal-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-prewitt-v"></a>
- [ ] **`filters.prewitt_v`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: exact column derivative orientation and 2D mask border. Cases to close: vertical ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Prewitt derivative/box taps along the vertical-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-rank-order"></a>
- [ ] **`filters.rank_order`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image)`.
  - Required focus: NaNs, signed zero, dtype preservation and return tuple. Cases to close: low/high-cardinality arrays and float special values.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse stable value sort/unique mapping to dense ranks.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-roberts"></a>
- [ ] **`filters.roberts`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: 2D mask erosion, normalization and dtype. Cases to close: diagonal edges and masked boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse both native diagonal Roberts differences and normalized magnitude.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-roberts-neg-diag"></a>
- [ ] **`filters.roberts_neg_diag`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: tap orientation and mask border. Cases to close: negative diagonal ramps.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native negative-diagonal Roberts 2x2 stencil.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-roberts-pos-diag"></a>
- [ ] **`filters.roberts_pos_diag`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: tap orientation and mask border. Cases to close: positive diagonal ramps.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native positive-diagonal Roberts 2x2 stencil.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-sato"></a>
- [ ] **`filters.sato`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, sigmas=range(1, 10, 2), black_ridges=True, mode='reflect', cval=0)`.
  - Required focus: eigenvalue sign/order, 2D/3D formula and polarity. Cases to close: bright/dark tubes across sigma.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native selected Hessian eigenvalue tubeness and scale maximum.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-scharr"></a>
- [ ] **`filters.scharr`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None, *, axis=None, mode='reflect', cval=0.0)`.
  - Required focus: axis tuple, normalization, mask erosion and mode/cval. Cases to close: directional ramps and axis combinations.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Scharr derivative/smoothing taps; combine selected-axis squared responses.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-scharr-h"></a>
- [ ] **`filters.scharr_h`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: exact row derivative orientation and 2D mask border. Cases to close: horizontal ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Scharr derivative/smoothing taps along the horizontal-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-scharr-v"></a>
- [ ] **`filters.scharr_v`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: exact column derivative orientation and 2D mask border. Cases to close: vertical ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Scharr derivative/smoothing taps along the vertical-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-sobel"></a>
- [ ] **`filters.sobel`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None, *, axis=None, mode='reflect', cval=0.0)`.
  - Required focus: axis tuple, normalization, mask erosion and mode/cval. Cases to close: directional ramps and axis combinations.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Sobel derivative/smoothing taps; combine selected-axis squared responses.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-sobel-h"></a>
- [ ] **`filters.sobel_h`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: exact row derivative orientation and 2D mask border. Cases to close: horizontal ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Sobel derivative/smoothing taps along the horizontal-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-sobel-v"></a>
- [ ] **`filters.sobel_v`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mask=None)`.
  - Required focus: exact column derivative orientation and 2D mask border. Cases to close: vertical ramps/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Sobel derivative/smoothing taps along the vertical-edge convention.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-isodata"></a>
- [ ] **`filters.threshold_isodata`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image=None, nbins=256, return_all=False, *, hist=None)`.
  - Required focus: return_all, integer midpoint tolerance and empty split. Cases to close: multiple/zero fixed-point histogram solutions.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse cumulative class means and accepted midpoint scan.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-li"></a>
- [ ] **`filters.threshold_li`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, *, tolerance=None, initial_guess=None, iter_callback=None)`.
  - Required focus: NaN/Inf admission, initial_guess callback, iteration callback/order and tolerance. Cases to close: captured difficult initialization/iteration cases and histogram shapes.
  - Easiest remaining route: Audit/extend existing native implementation: Audit existing typed Li iteration; reuse frozen reference and private audit only after review.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-local"></a>
- [ ] **`filters.threshold_local`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, block_size=3, method='gaussian', offset=0, mode='reflect', param=None, cval=0)`.
  - Required focus: method/param, odd block_size, mode and callback order/errors. Cases to close: all methods, anisotropic blocks and callback trace.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse Gaussian/mean/median local filters; preserve generic callback adapter.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-mean"></a>
- [ ] **`filters.threshold_mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image)`.
  - Required focus: accumulator promotion, empty input and nonfinite behavior. Cases to close: integer extrema and cancelling float arrays.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native ordered mean reduction.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-minimum"></a>
- [ ] **`filters.threshold_minimum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image=None, nbins=256, max_num_iter=10000, *, hist=None)`.
  - Required focus: plateau peak detection, max_num_iter error and valley ties. Cases to close: unimodal/noisy/bimodal distributions.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse iterative histogram smoothing and two-peak valley selection.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-multiotsu"></a>
- [ ] **`filters.threshold_multiotsu`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image=None, classes=3, nbins=256, *, hist=None)`.
  - Required focus: class count, insufficient values, tie order and bin quantization. Cases to close: class/bin sweep and sparse histograms.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native dynamic/multiclass histogram threshold search.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-niblack"></a>
- [ ] **`filters.threshold_niblack`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, window_size=15, k=0.2)`.
  - Required focus: window_size, boundary convention, cancellation and dtype. Cases to close: constant/high-offset and varying-window images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local first/second moments then mean-minus-k-sigma.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-otsu"></a>
- [ ] **`filters.threshold_otsu`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(image=None, nbins=256, *, hist=None)`.
  - Required focus: hist override, bin centers, degenerate classes and first tie. Cases to close: multimodal/tied/constant histograms.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse histogram cumulative class weights/means and first maximizing split.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-sauvola"></a>
- [ ] **`filters.threshold_sauvola`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, window_size=15, k=0.2, r=None)`.
  - Required focus: r default, window_size, constant images and overflow avoidance. Cases to close: explicit/default r and high-offset images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local moments and dtype-derived dynamic range.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-triangle"></a>
- [ ] **`filters.threshold_triangle`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, nbins=256)`.
  - Required focus: tail reversal, zero extent, bin centers and tie. Cases to close: left/right skewed and constant histograms.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse peak-to-tail histogram distance geometry.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-threshold-yen"></a>
- [ ] **`filters.threshold_yen`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image=None, nbins=256, *, hist=None)`.
  - Required focus: zeros/log domain, histogram validation and tie selection. Cases to close: sparse and symmetric histograms.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse histogram cumulative squared-probability criterion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-try-all-threshold"></a>
- [ ] **`filters.try_all_threshold`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, figsize=(8, 5), verbose=True)`.
  - Required focus: failed-method display, axes order, figsize and printed output. Cases to close: constant and multimodal images with headless artist checks.
  - Easiest remaining route: Run owned threshold functions then preserve plotting/verbosity adapter.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-filters-unsharp-mask"></a>
- [ ] **`filters.unsharp_mask`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, radius=1.0, amount=1.0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: radius, channel_axis, range/clipping and dtype. Cases to close: all preserve_range and strength branches.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native Gaussian blur then image-plus-amount residual.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-wiener"></a>
- [ ] **`filters.wiener`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(data, impulse_response=None, filter_params=None, K=0.25, predefined_filter=None)`.
  - Required focus: K broadcasting/range, predefined filter and real/complex output. Cases to close: weak/zero transfer spectra and K sweep.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse LPI transfer function and conjugate-over-power-plus-K multiplier.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-window"></a>
- [ ] **`filters.window`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(window_type, shape, warp_kwargs=None)`.
  - Required focus: window_type parameter tuples, center/radius convention and warp_kwargs. Cases to close: named windows and odd/even anisotropic shapes.
  - Easiest remaining route: Audit/extend existing native implementation: Generate named 1D window through reviewed formulas/provider then radially resample to N-D.
  - Evidence locators: parity [tests/test_window_public.py](../tests/test_window_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-filters-rank-autolevel"></a>
- [ ] **`filters.rank.autolevel`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: zero span, masked population and output integer cast. Cases to close: constant/two-level windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse rolling local min/max then full-range affine stretch.
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-autolevel-percentile"></a>
- [ ] **`filters.rank.autolevel_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: p0/p1 boundaries, empty selected population and clipping. Cases to close: quantile-boundary windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse percentile-restricted endpoints then local affine stretch.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-enhance-contrast"></a>
- [ ] **`filters.rank.enhance_contrast`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: equal-distance endpoint tie and masked centers. Cases to close: equidistant center values.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse closest local min/max selection.
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-enhance-contrast-percentile"></a>
- [ ] **`filters.rank.enhance_contrast_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: percentile cut bins and equal-distance ties. Cases to close: outlier-rich percentile windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse closest selected-percentile endpoint.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-entropy"></a>
- [ ] **`filters.rank.entropy`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: empty population, log base, float output and feedback stores. Cases to close: uniform/singleton/high-entropy windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse occupied-bin count entropy reduction.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-equalize"></a>
- [ ] **`filters.rank.equalize`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: inclusive cumulative bound, normalization and integer cast. Cases to close: tied intensities and empty masks.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse cumulative local histogram rank of center.
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-geometric-mean"></a>
- [ ] **`filters.rank.geometric_mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: zero bin handling, rounding and empty population. Cases to close: windows containing zero and wide intensity range.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse log-domain occupied-bin weighted mean and inverse transform.
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-gradient"></a>
- [ ] **`filters.rank.gradient`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: empty window and unsigned output. Cases to close: high dynamic range windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local occupied-bin max minus min.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-gradient-percentile"></a>
- [ ] **`filters.rank.gradient_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: p0/p1 selected-bin inclusivity. Cases to close: tied/boundary quantiles.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse percentile endpoint difference.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-majority"></a>
- [ ] **`filters.rank.majority`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, *, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: equal-frequency tie, empty population and captured output dtype. Cases to close: multiway ties and masked windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native lowest-bin maximal-count selection.
  - Evidence locators: parity [tests/test_rank_decisions_public.py](../tests/test_rank_decisions_public.py); benchmark [benchmarks/rank_decisions.py](../benchmarks/rank_decisions.py).

<a id="api-filters-rank-maximum"></a>
- [ ] **`filters.rank.maximum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: empty population, shifts and mask membership. Cases to close: shifted anisotropic footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse highest occupied local histogram bin.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-mean"></a>
- [ ] **`filters.rank.mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: signed integer division and output casting. Cases to close: nondivisible sums and masks.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse integer weighted count sum divided by population.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-mean-bilateral"></a>
- [ ] **`filters.rank.mean_bilateral`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10)`.
  - Required focus: strict g-s1<i<g+s0, empty set, shifts and feedback. Cases to close: exact bilateral interval endpoints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse strict center-relative intensity-window weighted mean.
  - Evidence locators: parity [tests/test_rank_bilateral_public.py](../tests/test_rank_bilateral_public.py); benchmark [benchmarks/rank_bilateral.py](../benchmarks/rank_bilateral.py).

<a id="api-filters-rank-mean-percentile"></a>
- [ ] **`filters.rank.mean_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: fractional count cutoff and empty interval. Cases to close: repeated bins crossing p0/p1.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse mean over count-selected percentile population.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-median"></a>
- [ ] **`filters.rank.median`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: even population tie convention and default footprint. Cases to close: even/odd masked populations.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse rolling histogram median selection.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-minimum"></a>
- [ ] **`filters.rank.minimum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: empty population, shifts and mask membership. Cases to close: shifted anisotropic footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse lowest occupied local histogram bin.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-modal"></a>
- [ ] **`filters.rank.modal`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: equal maximum frequencies and empty population. Cases to close: tied modes and dtype limits.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local mode with source-qualified tie convention.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-noise-filter"></a>
- [ ] **`filters.rank.noise_filter`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: automatic center exclusion, absent neighbors and output cast. Cases to close: isolated impulses and existing slow sparse windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse nearest occupied-neighbor intensity distance with center-excluded footprint.
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-otsu"></a>
- [ ] **`filters.rank.otsu`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: strict floating winner comparisons, degenerate classes and empty population. Cases to close: flat/tied bimodal windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse ascending occupied-bin local Otsu criterion.
  - Evidence locators: parity [tests/test_rank_decisions_public.py](../tests/test_rank_decisions_public.py); benchmark [benchmarks/rank_decisions.py](../benchmarks/rank_decisions.py).

<a id="api-filters-rank-percentile"></a>
- [ ] **`filters.rank.percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0)`.
  - Required focus: endpoint quantiles and discrete count tie. Cases to close: p0=0/1 and exact cumulative boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse cumulative count target selection for p0.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-pop"></a>
- [ ] **`filters.rank.pop`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: output width overflow/cast and empty footprint. Cases to close: huge populations and masks.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local masked population count.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-pop-bilateral"></a>
- [ ] **`filters.rank.pop_bilateral`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10)`.
  - Required focus: s0/s1 conversion/order and interval endpoint exclusion. Cases to close: empty and endpoint-only intensity windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse strict center-relative interval population.
  - Evidence locators: parity [tests/test_rank_bilateral_public.py](../tests/test_rank_bilateral_public.py); benchmark [benchmarks/rank_bilateral.py](../benchmarks/rank_bilateral.py).

<a id="api-filters-rank-pop-percentile"></a>
- [ ] **`filters.rank.pop_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: fractional cumulative cutoffs and p0/p1 order. Cases to close: repeated-bin count boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse count in percentile-selected population.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-subtract-mean"></a>
- [ ] **`filters.rank.subtract_mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: signed intermediate, integer division and wrap/store behavior. Cases to close: center above/below mean and integer extrema.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local mean subtraction with upstream midpoint scaling/offset.
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-subtract-mean-percentile"></a>
- [ ] **`filters.rank.subtract_mean_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: selected-population rounding and output cast. Cases to close: narrow selected populations and extrema.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse percentile mean subtraction and upstream scaling/offset.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-sum"></a>
- [ ] **`filters.rank.sum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: overflow/store conversion and masks. Cases to close: high-intensity large windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse weighted local histogram sum.
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-sum-bilateral"></a>
- [ ] **`filters.rank.sum_bilateral`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10)`.
  - Required focus: interval bounds, integer accumulation and out dtype. Cases to close: high-intensity selected bins.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse strict center-relative interval weighted sum.
  - Evidence locators: parity [tests/test_rank_bilateral_public.py](../tests/test_rank_bilateral_public.py); benchmark [benchmarks/rank_bilateral.py](../benchmarks/rank_bilateral.py).

<a id="api-filters-rank-sum-percentile"></a>
- [ ] **`filters.rank.sum_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1)`.
  - Required focus: partial-bin/count cutoff and output conversion. Cases to close: selected bins at exact cumulative fractions.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse percentile-selected weighted sum.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-threshold"></a>
- [ ] **`filters.rank.threshold`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Required focus: equality direction, scaling and empty population. Cases to close: center exactly equal to mean.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse center-versus-local-mean binary decision.
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-threshold-percentile"></a>
- [ ] **`filters.rank.threshold_percentile`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0)`.
  - Required focus: equality direction, p0 endpoint and output scaling. Cases to close: center exactly equal to percentile.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse center-versus-selected-percentile binary decision.
  - Evidence locators: parity [tests/test_rank_percentile_public.py](../tests/test_rank_percentile_public.py); benchmark [benchmarks/rank_percentile.py](../benchmarks/rank_percentile.py).

<a id="api-filters-rank-windowed-histogram"></a>
- [ ] **`filters.rank.windowed_histogram`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, n_bins=None)`.
  - Required focus: n_bins truncation/inference, trailing axis, zero population and large output cost. Cases to close: small/large bin counts and retained slow large windows.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse local normalized histogram vector per center.
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-bilateral-mean-bilateral"></a>
- [ ] **`filters.rank.bilateral.mean_bilateral`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10)`.
  - Canonical alias: `filters.rank.mean_bilateral`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: strict g-s1<i<g+s0, empty set, shifts and feedback. Cases to close: exact bilateral interval endpoints.
  - Easiest remaining route: Exact alias of filters.rank.mean_bilateral; reuse its route: Reuse strict center-relative intensity-window weighted mean. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_bilateral_public.py](../tests/test_rank_bilateral_public.py); benchmark [benchmarks/rank_bilateral.py](../benchmarks/rank_bilateral.py).

<a id="api-filters-rank-bilateral-pop-bilateral"></a>
- [ ] **`filters.rank.bilateral.pop_bilateral`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10)`.
  - Canonical alias: `filters.rank.pop_bilateral`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: s0/s1 conversion/order and interval endpoint exclusion. Cases to close: empty and endpoint-only intensity windows.
  - Easiest remaining route: Exact alias of filters.rank.pop_bilateral; reuse its route: Reuse strict center-relative interval population. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_bilateral_public.py](../tests/test_rank_bilateral_public.py); benchmark [benchmarks/rank_bilateral.py](../benchmarks/rank_bilateral.py).

<a id="api-filters-rank-bilateral-sum-bilateral"></a>
- [ ] **`filters.rank.bilateral.sum_bilateral`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10)`.
  - Canonical alias: `filters.rank.sum_bilateral`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: interval bounds, integer accumulation and out dtype. Cases to close: high-intensity selected bins.
  - Easiest remaining route: Exact alias of filters.rank.sum_bilateral; reuse its route: Reuse strict center-relative interval weighted sum. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_bilateral_public.py](../tests/test_rank_bilateral_public.py); benchmark [benchmarks/rank_bilateral.py](../benchmarks/rank_bilateral.py).

<a id="api-filters-rank-generic-autolevel"></a>
- [ ] **`filters.rank.generic.autolevel`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.autolevel`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: zero span, masked population and output integer cast. Cases to close: constant/two-level windows.
  - Easiest remaining route: Exact alias of filters.rank.autolevel; reuse its route: Reuse rolling local min/max then full-range affine stretch. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-generic-enhance-contrast"></a>
- [ ] **`filters.rank.generic.enhance_contrast`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.enhance_contrast`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: equal-distance endpoint tie and masked centers. Cases to close: equidistant center values.
  - Easiest remaining route: Exact alias of filters.rank.enhance_contrast; reuse its route: Reuse closest local min/max selection. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-generic-entropy"></a>
- [ ] **`filters.rank.generic.entropy`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.entropy`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: empty population, log base, float output and feedback stores. Cases to close: uniform/singleton/high-entropy windows.
  - Easiest remaining route: Exact alias of filters.rank.entropy; reuse its route: Reuse occupied-bin count entropy reduction. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-generic-equalize"></a>
- [ ] **`filters.rank.generic.equalize`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.equalize`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: inclusive cumulative bound, normalization and integer cast. Cases to close: tied intensities and empty masks.
  - Easiest remaining route: Exact alias of filters.rank.equalize; reuse its route: Reuse cumulative local histogram rank of center. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-generic-geometric-mean"></a>
- [ ] **`filters.rank.generic.geometric_mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.geometric_mean`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: zero bin handling, rounding and empty population. Cases to close: windows containing zero and wide intensity range.
  - Easiest remaining route: Exact alias of filters.rank.geometric_mean; reuse its route: Reuse log-domain occupied-bin weighted mean and inverse transform. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-generic-gradient"></a>
- [ ] **`filters.rank.generic.gradient`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.gradient`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: empty window and unsigned output. Cases to close: high dynamic range windows.
  - Easiest remaining route: Exact alias of filters.rank.gradient; reuse its route: Reuse local occupied-bin max minus min. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-generic-majority"></a>
- [ ] **`filters.rank.generic.majority`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, *, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.majority`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: equal-frequency tie, empty population and captured output dtype. Cases to close: multiway ties and masked windows.
  - Easiest remaining route: Exact alias of filters.rank.majority; reuse its route: Reuse native lowest-bin maximal-count selection. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_decisions_public.py](../tests/test_rank_decisions_public.py); benchmark [benchmarks/rank_decisions.py](../benchmarks/rank_decisions.py).

<a id="api-filters-rank-generic-maximum"></a>
- [ ] **`filters.rank.generic.maximum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.maximum`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: empty population, shifts and mask membership. Cases to close: shifted anisotropic footprints.
  - Easiest remaining route: Exact alias of filters.rank.maximum; reuse its route: Reuse highest occupied local histogram bin. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-generic-mean"></a>
- [ ] **`filters.rank.generic.mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.mean`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: signed integer division and output casting. Cases to close: nondivisible sums and masks.
  - Easiest remaining route: Exact alias of filters.rank.mean; reuse its route: Reuse integer weighted count sum divided by population. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-generic-median"></a>
- [ ] **`filters.rank.generic.median`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.median`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: even population tie convention and default footprint. Cases to close: even/odd masked populations.
  - Easiest remaining route: Exact alias of filters.rank.median; reuse its route: Reuse rolling histogram median selection. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-generic-minimum"></a>
- [ ] **`filters.rank.generic.minimum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.minimum`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: empty population, shifts and mask membership. Cases to close: shifted anisotropic footprints.
  - Easiest remaining route: Exact alias of filters.rank.minimum; reuse its route: Reuse lowest occupied local histogram bin. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-generic-modal"></a>
- [ ] **`filters.rank.generic.modal`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.modal`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: equal maximum frequencies and empty population. Cases to close: tied modes and dtype limits.
  - Easiest remaining route: Exact alias of filters.rank.modal; reuse its route: Reuse local mode with source-qualified tie convention. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank_distribution.py](../benchmarks/rank_distribution.py).

<a id="api-filters-rank-generic-noise-filter"></a>
- [ ] **`filters.rank.generic.noise_filter`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.noise_filter`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: automatic center exclusion, absent neighbors and output cast. Cases to close: isolated impulses and existing slow sparse windows.
  - Easiest remaining route: Exact alias of filters.rank.noise_filter; reuse its route: Reuse nearest occupied-neighbor intensity distance with center-excluded footprint. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-generic-otsu"></a>
- [ ] **`filters.rank.generic.otsu`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.otsu`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: strict floating winner comparisons, degenerate classes and empty population. Cases to close: flat/tied bimodal windows.
  - Easiest remaining route: Exact alias of filters.rank.otsu; reuse its route: Reuse ascending occupied-bin local Otsu criterion. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_decisions_public.py](../tests/test_rank_decisions_public.py); benchmark [benchmarks/rank_decisions.py](../benchmarks/rank_decisions.py).

<a id="api-filters-rank-generic-pop"></a>
- [ ] **`filters.rank.generic.pop`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.pop`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: output width overflow/cast and empty footprint. Cases to close: huge populations and masks.
  - Easiest remaining route: Exact alias of filters.rank.pop; reuse its route: Reuse local masked population count. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-generic-subtract-mean"></a>
- [ ] **`filters.rank.generic.subtract_mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.subtract_mean`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: signed intermediate, integer division and wrap/store behavior. Cases to close: center above/below mean and integer extrema.
  - Easiest remaining route: Exact alias of filters.rank.subtract_mean; reuse its route: Reuse local mean subtraction with upstream midpoint scaling/offset. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

<a id="api-filters-rank-generic-sum"></a>
- [ ] **`filters.rank.generic.sum`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.sum`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: overflow/store conversion and masks. Cases to close: high-intensity large windows.
  - Easiest remaining route: Exact alias of filters.rank.sum; reuse its route: Reuse weighted local histogram sum. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_public.py](../tests/test_rank_public.py); benchmark [benchmarks/rank.py](../benchmarks/rank.py).

<a id="api-filters-rank-generic-threshold"></a>
- [ ] **`filters.rank.generic.threshold`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0)`.
  - Canonical alias: `filters.rank.threshold`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: equality direction, scaling and empty population. Cases to close: center exactly equal to mean.
  - Easiest remaining route: Exact alias of filters.rank.threshold; reuse its route: Reuse center-versus-local-mean binary decision. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_intensity_public.py](../tests/test_rank_intensity_public.py); benchmark [benchmarks/rank_intensity.py](../benchmarks/rank_intensity.py).

<a id="api-filters-rank-generic-windowed-histogram"></a>
- [ ] **`filters.rank.generic.windowed_histogram`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, n_bins=None)`.
  - Canonical alias: `filters.rank.windowed_histogram`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: n_bins truncation/inference, trailing axis, zero population and large output cost. Cases to close: small/large bin counts and retained slow large windows.
  - Easiest remaining route: Exact alias of filters.rank.windowed_histogram; reuse its route: Reuse local normalized histogram vector per center. No extra algorithm credit..
  - Evidence locators: parity [tests/test_rank_statistics_public.py](../tests/test_rank_statistics_public.py); benchmark [benchmarks/rank_statistics.py](../benchmarks/rank_statistics.py).

</details>

<a id="module-filters.rank"></a>
### filters.rank — 0 public keys

<details><summary>Expand filters.rank contract checklist</summary>

</details>

<a id="module-filters.rank.bilateral"></a>
### filters.rank.bilateral — 0 public keys

<details><summary>Expand filters.rank.bilateral contract checklist</summary>

</details>

<a id="module-filters.rank.generic"></a>
### filters.rank.generic — 0 public keys

<details><summary>Expand filters.rank.generic contract checklist</summary>

</details>

<a id="module-future"></a>
### future — 5 public keys

<details><summary>Expand future contract checklist</summary>

<a id="api-future-trainablesegmenter"></a>
- [ ] **`future.TrainableSegmenter`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(clf=None, features_func=None)`.
  - Required focus: constructor defaults, fitted state, custom features_func and estimator identity. Cases to close: stub estimator lifecycle and image/feature pipelines.
  - Easiest remaining route: Own estimator/feature-function state wrapper composing fit_segmenter and predict_segmenter.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-future-fit-segmenter"></a>
- [ ] **`future.fit_segmenter`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(labels, features, clf)`.
  - Required focus: background label convention, sample order, label shape and estimator return. Cases to close: sparse labels and custom estimator call capture.
  - Easiest remaining route: Select positive labeled pixels, flatten corresponding feature rows and call supplied estimator.fit.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-future-manual-lasso-segmentation"></a>
- [ ] **`future.manual_lasso_segmentation`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, alpha=0.4, return_all=False)`.
  - Required focus: event ordering, multiple selections, alpha and return_all/cancel. Cases to close: scripted lasso selections and cancellation.
  - Easiest remaining route: Preserve interactive lasso adapter and polygon-mask generation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-future-manual-polygon-segmentation"></a>
- [ ] **`future.manual_polygon_segmentation`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, alpha=0.4, return_all=False)`.
  - Required focus: vertex closure, deletion/completion events and return_all. Cases to close: scripted concave polygons and cancellation.
  - Easiest remaining route: Preserve polygon-vertex event adapter and raster masks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-future-predict-segmenter"></a>
- [ ] **`future.predict_segmenter`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(features, clf)`.
  - Required focus: feature axis, prediction dtype/shape and estimator error propagation. Cases to close: noncontiguous features and custom prediction outputs.
  - Easiest remaining route: Flatten feature rows, call supplied estimator.predict and reshape labels.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-graph"></a>
### graph — 15 public keys

<details><summary>Expand graph contract checklist</summary>

<a id="api-graph-mcp"></a>
- [ ] **`graph.MCP`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: offset ordering, starts/ends, max_cost, tie queue order and traceback state. Cases to close: weighted grids with multiple sources and unreachable targets.
  - Easiest remaining route: Own Dijkstra-style grid cost front with predecessor storage and public state wrapper.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-mcp-connect"></a>
- [ ] **`graph.MCP_Connect`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: create_connection arguments/event order and retained path state. Cases to close: colliding multiple-source fronts.
  - Easiest remaining route: Extend shared MCP front with ordered meeting-front connection hooks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-mcp-flexible"></a>
- [ ] **`graph.MCP_Flexible`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: callback order, exceptions, goal stopping and mutable subclass state. Cases to close: instrumented custom costs and early termination.
  - Easiest remaining route: Extend shared MCP front with Python travel/update/examine callbacks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-mcp-geometric"></a>
- [ ] **`graph.MCP_Geometric`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: sampling spacing, diagonal lengths and cumulative tie order. Cases to close: anisotropic 2D/3D grids.
  - Easiest remaining route: Extend MCP using edge length times endpoint cost interpolation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-rag"></a>
- [ ] **`graph.RAG`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(label_image=None, connectivity=1, data=None, **attr)`.
  - Required focus: 46 static inherited members, node/edge attrs, views, merges and copy semantics. Cases to close: adjacency/attribute/view mutation fixtures and label graph construction.
  - Easiest remaining route: Preserve NetworkX-compatible mutable graph adapter; native label adjacency and region statistics.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-central-pixel"></a>
- [ ] **`graph.central_pixel`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(graph, nodes=None, shape=None, partition_size=100)`.
  - Required focus: disconnected graph, node mapping, partition_size and tie node. Cases to close: disconnected and symmetric grid graphs.
  - Easiest remaining route: Run partitioned sparse shortest paths and minimize total distance.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-cut-normalized"></a>
- [ ] **`graph.cut_normalized`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(labels, rag, thresh=0.001, num_cuts=10, in_place=True, max_edge=1.0, *, rng=None)`.
  - Required focus: rng, eigensolver degeneracy, in_place mutation, num_cuts and label mapping. Cases to close: weakly connected and repeated-eigenvalue RAGs.
  - Easiest remaining route: Build normalized graph Laplacian and recursive spectral threshold cuts.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-cut-threshold"></a>
- [ ] **`graph.cut_threshold`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(labels, rag, thresh, in_place=True)`.
  - Required focus: threshold equality, in_place and node-label mapping. Cases to close: threshold-equal edges and disconnected RAGs.
  - Easiest remaining route: Remove threshold-qualified edges then connected-component relabeling.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-merge-hierarchical"></a>
- [ ] **`graph.merge_hierarchical`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(labels, rag, thresh, rag_copy, in_place_merge, merge_func, weight_func)`.
  - Required focus: stale entries, rag_copy/in_place_merge and exact callback order. Cases to close: equal-weight merges and instrumented callbacks.
  - Easiest remaining route: Stable priority queue of region edges with merge/weight callbacks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-pixel-graph"></a>
- [ ] **`graph.pixel_graph`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, *, mask=None, edge_function=None, connectivity=1, spacing=None, sparse_type='matrix')`.
  - Required focus: mask/node mapping, edge_function callbacks, sparse matrix/array and connectivity. Cases to close: sparse masks and custom edge functions.
  - Easiest remaining route: Enumerate neighbor pairs, calculate spacing/intensity edge weights and emit sparse graph.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-rag-boundary"></a>
- [ ] **`graph.rag_boundary`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(labels, edge_map, connectivity=2)`.
  - Required focus: background/zero labels, connectivity and pair ordering. Cases to close: thin boundaries and triple junctions.
  - Easiest remaining route: Aggregate region-pair boundary edge values and counts.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-rag-mean-color"></a>
- [ ] **`graph.rag_mean_color`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, labels, connectivity=2, mode='distance', sigma=255.0)`.
  - Required focus: mode/sigma, channel count and node attributes. Cases to close: sparse labels, constant colors and mode variants.
  - Easiest remaining route: Reuse native grouped color sums and label adjacency; compute mean-color edge distance/similarity.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-route-through-array"></a>
- [ ] **`graph.route_through_array`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(array, start, end, fully_connected=True, geometric=True)`.
  - Required focus: fully_connected, geometric flag, start/end validation and ties. Cases to close: axis/diagonal routes and unreachable endpoints.
  - Easiest remaining route: Construct MCP or MCP_Geometric and extract one path/cost.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-shortest-path"></a>
- [ ] **`graph.shortest_path`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(arr, reach=1, axis=-1, output_indexlist=False)`.
  - Required focus: reach, axis, negative costs and output_indexlist ordering. Cases to close: narrow/wide reach and degenerate axes.
  - Easiest remaining route: Dynamic programming along selected axis with bounded reach transitions.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-graph-show-rag"></a>
- [ ] **`graph.show_rag`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(labels, rag, image, border_color='black', edge_width=1.5, edge_cmap='magma', img_cmap='bone', in_place=True, ax=None)`.
  - Required focus: in_place graph position attrs, colormaps, axes and artist returns. Cases to close: fixed RAG artist positions and mutation checks.
  - Easiest remaining route: Preserve plotting adapter using native region centroids/edge geometry.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-io"></a>
### io — 11 public keys

<details><summary>Expand io contract checklist</summary>

<a id="api-io-imagecollection"></a>
- [ ] **`io.ImageCollection`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(load_pattern, conserve_memory=True, load_func=None, **load_func_kwargs)`.
  - Required focus: glob/list ordering, conserve_memory, slices, reload and custom load_func. Cases to close: multi-pattern collections with instrumented lazy loader.
  - Easiest remaining route: Own ordered filename/loader/cache collection wrapper around optional decode provider.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-multiimage"></a>
- [ ] **`io.MultiImage`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(filename, conserve_memory=True, dtype=None, **imread_kwargs)`.
  - Required focus: frame indexing, dtype option, conserve_memory and inherited methods. Cases to close: multipage TIFF/GIF-style local fixtures.
  - Easiest remaining route: Reuse collection wrapper with multipage image decoding and filename state.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-concatenate-images"></a>
- [ ] **`io.concatenate_images`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(ic)`.
  - Required focus: loading order, mismatched shapes and dtype promotion. Cases to close: homogeneous/mixed-shape collections.
  - Easiest remaining route: Stack collection arrays after compatible-shape validation using native copy path.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-imread"></a>
- [ ] **`io.imread`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(fname, as_gray=False, plugin=<DEPRECATED>, **plugin_args)`.
  - Required focus: plugin deprecation, URI/file objects, alpha/gray conversion and codec errors. Cases to close: PNG/JPEG/TIFF/NPY-supported provider matrix and local file objects.
  - Easiest remaining route: Preserve codec/plugin adapter, then own dtype/channel/as_gray postprocessing.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-imread-collection"></a>
- [ ] **`io.imread_collection`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(load_pattern, conserve_memory=True, plugin=<DEPRECATED>, **plugin_args)`.
  - Required focus: load_pattern ordering, conserve_memory and deprecated plugin arguments. Cases to close: patterns, explicit lists and lazy load traces.
  - Easiest remaining route: Construct ImageCollection through plugin-aware reader.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-imread-collection-wrapper"></a>
- [ ] **`io.imread_collection_wrapper`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(imread)`.
  - Required focus: callable identity, kwargs forwarding and loader exceptions. Cases to close: stub reader call order and collection reload.
  - Easiest remaining route: Wrap supplied imread callable into compatible collection reader.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-imsave"></a>
- [ ] **`io.imsave`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(fname, arr, plugin=<DEPRECATED>, *, check_contrast=True, **plugin_args)`.
  - Required focus: check_contrast warning, dtype conversion, plugin kwargs and file side effects. Cases to close: lossless round trips and lossy-codec documented tolerances.
  - Easiest remaining route: Preserve encoder/plugin adapter with native range/contrast checks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-load-sift"></a>
- [ ] **`io.load_sift`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(f)`.
  - Required focus: header counts, field order, malformed/truncated text and file objects. Cases to close: tiny hand-authored and large SIFT fixtures.
  - Easiest remaining route: Parse text SIFT header/keypoints/descriptors into exact structured output.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-load-surf"></a>
- [ ] **`io.load_surf`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(f)`.
  - Required focus: descriptor length/header, malformed records and dtype. Cases to close: hand-authored and large SURF fixtures.
  - Easiest remaining route: Parse SURF text records with exact coordinate/descriptor field layout.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-pop"></a>
- [ ] **`io.pop`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: LIFO identity, empty error and shared module state. Cases to close: push/pop interleavings.
  - Easiest remaining route: Own process-local image stack removal with identical empty-stack behavior.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-io-push"></a>
- [ ] **`io.push`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(img)`.
  - Required focus: accepted object types, aliasing and return contract. Cases to close: repeated references and mixed image objects.
  - Easiest remaining route: Own image stack insertion retaining exact object identity.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-measure"></a>
### measure — 35 public keys

<details><summary>Expand measure contract checklist</summary>

<a id="api-measure-circlemodel"></a>
- [ ] **`measure.CircleModel`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(center, radius)`.
  - Required focus: 0.26 constructor/from_estimate failure behavior, collinear points, negative radius and params. Cases to close: minimal/collinear/noisy circles and predict angles.
  - Easiest remaining route: Own class state with centered algebraic circle least squares and geometric radial residuals.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-measure-ellipsemodel"></a>
- [ ] **`measure.EllipseModel`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(center, axis_lengths, theta)`.
  - Required focus: conic rank/axis ordering, angle convention, degeneracy and solver convergence. Cases to close: near-circles, eccentric/degenerate ellipses and residual points.
  - Easiest remaining route: Normalized constrained ellipse least-squares solve; geometric closest-ellipse residual solve.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-measure-linemodelnd"></a>
- [ ] **`measure.LineModelND`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - The latest private source remains held at 15 of 16 fixed comparisons. Linux minimum residuals measured 0.99649× and the retained data does not establish a cause. [Residual diagnosis](../build/evidence/u182-line-model-nd/remaining-residual-feasibility-v2.json)
  - Signature: `(origin, direction)`.
  - Required focus: direction sign, rank failure, predict axis zero and dimensionality. Cases to close: collinear/noisy/high-dimensional points.
  - Easiest remaining route: Center points and use leading singular vector for orthogonal line fit/residual projection.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private `b7f8be0d` passes the scoped lifecycle and both SDK clients. Compiled x/y pass speed; the Windows 3D prediction and WSL minimum SVD fit remain below parity. [Private prediction and fit timing](../build/evidence/u182-line-model-nd/timing-v3/timing-root-admission-v4.json)
  - The strict 3D successor `f227d486` now passes the scoped lifecycle and SDK cases. Both fixed 3D prediction and SVD cases meet one-worker parity. The WSL minimum 3D residual case remains at 0.996486× and holds the class. [Private v5 timing](../build/evidence/u182-line-model-nd/timing-v3/timing-root-admission-v5.json) [3D admission](../build/evidence/u182-line-model-nd/native-prediction-3d-root-admission-v1.json)

<a id="api-measure-ransacmodelprotocol"></a>
- [ ] **`measure.RansacModelProtocol`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(*args, **kwargs)`.
  - Required focus: runtime protocol behavior, subclass hook and inherited typing members. Cases to close: conforming/nonconforming custom model classes.
  - Easiest remaining route: Preserve typing/protocol and factory/residual method contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-measure-approximate-polygon"></a>
- [ ] **`measure.approximate_polygon`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float64`.
  - Signature: `(coords, tolerance)`.
  - Required focus: closed endpoints, tolerance equality, zero-length segments and order. Cases to close: open/closed/collinear/self-touching polylines.
  - Easiest remaining route: Audit/extend existing native implementation: Port ordered Douglas-Peucker segment simplification.
  - Evidence locators: parity [tests/test_approximate_polygon_integration.py::test_polygon_registered_public_frozen_contract](../tests/test_approximate_polygon_integration.py); benchmark [benchmarks/approximate_polygon.py](../benchmarks/approximate_polygon.py).

<a id="api-measure-block-reduce"></a>
- [ ] **`measure.block_reduce`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, block_size=2, func=<function sum>, cval=0, func_kwargs=None)`.
  - Required focus: arbitrary func callback/provider, cval, dtype and func_kwargs. Cases to close: nondivisible shapes and built-in/custom reductions.
  - Easiest remaining route: Audit/extend existing native implementation: Pad to block multiples then use native standard reducers over block axes.
  - Evidence locators: parity [tests/test_block_reduce_public.py](../tests/test_block_reduce_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-blur-effect"></a>
- [ ] **`measure.blur_effect`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, h_size=11, channel_axis=None, reduce_func=<function max>)`.
  - Required focus: channel-axis move/color conversion and printed errors; dtype/half behavior; h_size=1 sliding recurrence; positive tiny axes produce empty crop and NaN warnings; NumPy float64 epsilon promotion; final reducer receives the actual scalar list once. Cases to close: positive 1D/2D/3D ramps, constants and textures; tiny extents <=3, odd/even/unit filter sizes, RGB channels, layouts and callback/error state; exclude zero spatial axes until source C safety is certified.
  - Easiest remaining route: Audit/extend existing native implementation: Add source-ordered reflect uniform_filter1d; reuse float64 sparse correlation with source nonzero Sobel taps and one output cast; retain literal cropped NumPy reductions initially.
  - Evidence locators: parity [tests/test_blur_effect_public.py](../tests/test_blur_effect_public.py); benchmark [benchmarks/blur_effect.py](../benchmarks/blur_effect.py).

<a id="api-measure-centroid"></a>
- [ ] **`measure.centroid`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, *, spacing=None)`.
  - Required focus: spacing, zero mass, signed/nonfinite weights and dimensionality. Cases to close: weighted 2D/3D impulses and zero mass.
  - Easiest remaining route: Audit/extend existing native implementation: Compute zeroth/first raw moments then coordinate ratios.
  - Evidence locators: parity [tests/test_moments_public.py](../tests/test_moments_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-euler-number"></a>
- [ ] **`measure.euler_number`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, connectivity=None)`.
  - Required focus: foreground/background dual connectivity, padding and 3D unsupported connectivity. Cases to close: holes/tunnels/cavities and border-touching objects.
  - Easiest remaining route: Audit/extend existing native implementation: Use connectivity-specific 2D/3D neighborhood configuration LUT sums.
  - Evidence locators: parity [tests/test_euler_number_public.py](../tests/test_euler_number_public.py); benchmark [benchmarks/euler_number.py](../benchmarks/euler_number.py).

<a id="api-measure-find-contours"></a>
- [ ] **`measure.find_contours`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `uint8, float32, float64`.
  - Shipped on `a80be034` within the bounded finite 2D crossing-level domain. Shared installed tests and SDK passed with 16 fixed comparisons across this five-API batch. [Qualification](../build/evidence/u182-post-haar/hosted-current301-45090228543/whole-product-root-admission-v1.json) [Fixed benchmark](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)
  - Signature: `(image, level=None, fully_connected='low', positive_orientation='low', *, mask=None)`.
  - Required focus: saddle fully_connected choice, positive_orientation, NaNs/mask and open contours. Cases to close: ambiguous saddles and border/masked level sets.
  - Shipped scope: Bounded finite 2D image and explicit crossing level. C++ follows ordered marching squares and NumPy owns contiguous contour arrays; unsupported requests use compatibility.
  - Private `523e1553` matched 85 original Windows cases and four raw output streams across Linux profiles. Windows passed 87 checks per mode and the standalone SDK. All four fixed Windows timings met one-worker parity at 2.19×–2.55×. Successor `3125c60e` passed 384 private Windows and Linux minimum checks, including equivalent, tagged and swapped descriptors. Fresh fixed timing passed all eight cases at 2.18×–4.98× on Windows and Linux minimum. Linux modern passed an additional 192 private checks and 3.33×–4.53× over the same four fixed calls. The ordered-list comparator passed 30 focused private checks across three profiles. The later bounded installed route qualified and shipped. [Comparator checks](../build/evidence/u182-next-numerical-lookahead/find-contours/contour-comparator-handoff-v1.json) [Modern speed](../build/evidence/u182-post-haar/random-modern-timing-v1/root-modern-fixed-timing-admission-v1.json) [Successor speed](../build/evidence/u182-next-numerical-lookahead/find-contours/descriptor-timing-v2/root-timing-admission-v2.json) [Descriptor runtime](../build/evidence/u182-next-numerical-lookahead/find-contours/descriptor-component-handoff-v2.json) [Fixed speed](../build/evidence/u182-next-numerical-lookahead/find-contours/windows-timing-v1/root-admission-v1.json) · [Windows private component](../build/evidence/u182-next-numerical-lookahead/find-contours/windows-component-root-admission-v1.json)
  - Evidence locators: parity [tests/test_find_contours_public.py](../tests/test_find_contours_public.py); benchmark [benchmarks/find_contours.py](../benchmarks/find_contours.py).
  - Candidate eddec69f passed installed Windows parity in both modes. The fixed one-worker comparison was 2.414×–3.276× over four calls. Four-profile release is pending. [Installed timing](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)

<a id="api-measure-grid-points-in-poly"></a>
- [ ] **`measure.grid_points_in_poly`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(shape, verts, binarize=True)`.
  - Required focus: binarize false classification, coordinate orientation and edge ties. Cases to close: polygons passing through grid vertices/edges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse exact point-in-polygon evaluator on grid, retaining vertex/edge/interior codes.
  - Evidence locators: parity [tests/test_polygon_public.py](../tests/test_polygon_public.py); benchmark [benchmarks/polygon.py](../benchmarks/polygon.py).

<a id="api-measure-inertia-tensor"></a>
- [ ] **`measure.inertia_tensor`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mu=None, *, spacing=None)`.
  - Required focus: supplied mu, spacing, zero mass and component signs. Cases to close: anisotropic shapes and supplied moments.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse central second moments to form normalized symmetric inertia tensor.
  - Evidence locators: parity [tests/test_inertia_tensor_public.py](../tests/test_inertia_tensor_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-inertia-tensor-eigvals"></a>
- [ ] **`measure.inertia_tensor_eigvals`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, mu=None, T=None, *, spacing=None)`.
  - Required focus: descending order, tiny-negative clipping and supplied T/mu. Cases to close: repeated/near-zero eigenvalues.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse supplied/computed tensor and symmetric eigenvalue solve.
  - Evidence locators: parity [tests/test_inertia_eigvals_public.py](../tests/test_inertia_eigvals_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-intersection-coeff"></a>
- [ ] **`measure.intersection_coeff`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image0_mask, image1_mask, mask=None)`.
  - Required focus: optional mask and zero denominator. Cases to close: empty/disjoint/full-overlap masks.
  - Easiest remaining route: Audit/extend existing native implementation: Count boolean mask intersection divided by first-image support.
  - Evidence locators: parity [tests/test_colocalization_public.py](../tests/test_colocalization_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-label"></a>
- [ ] **`measure.label`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(label_image, background=None, return_num=False, connectivity=None)`.
  - Required focus: signed/unsigned dtype, background, connectivity, return_num and numbering. Cases to close: adversarial equal-value components and strided 2D/3D.
  - Easiest remaining route: Audit/extend existing native implementation: Extend existing deterministic equality-based union-find labels.
  - Evidence locators: parity [tests/test_connected_components_contracts.py](../tests/test_connected_components_contracts.py); benchmark [benchmarks/components.py](../benchmarks/components.py).

<a id="api-measure-manders-coloc-coeff"></a>
- [ ] **`measure.manders_coloc_coeff`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image0, image1_mask, mask=None)`.
  - Required focus: negativity, optional mask and zero total. Cases to close: sparse masks and zero/negative signals.
  - Easiest remaining route: Audit/extend existing native implementation: Sum nonnegative image0 signal inside image1_mask over total signal.
  - Evidence locators: parity [tests/test_colocalization_public.py](../tests/test_colocalization_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-manders-overlap-coeff"></a>
- [ ] **`measure.manders_overlap_coeff`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image0, image1, mask=None)`.
  - Required focus: range/shape validation, mask and zero energy. Cases to close: proportional/orthogonal/zero channels.
  - Easiest remaining route: Audit/extend existing native implementation: Normalize cross intensity product by energy product.
  - Evidence locators: parity [tests/test_colocalization_public.py](../tests/test_colocalization_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-marching-cubes"></a>
- [ ] **`measure.marching_cubes`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(volume, level=None, *, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, method='lewiner', mask=None)`.
  - Required focus: ambiguous cells, gradient_direction, step_size, degenerates, mask and output ordering. Cases to close: topology fixtures, spheres and anisotropic volumes.
  - Easiest remaining route: Port licensed Lewiner tables/topology rules and cell interpolation; offer classic method branch.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-measure-mesh-surface-area"></a>
- [ ] **`measure.mesh_surface_area`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(verts, faces)`.
  - Required focus: degenerate/repeated faces, index validity and floating accumulation. Cases to close: tiny/large meshes and zero-area triangles.
  - Easiest remaining route: Audit/extend existing native implementation: Gather triangle edges and reduce half cross-product norms.
  - Evidence locators: parity [tests/test_mesh_surface_area_public.py](../tests/test_mesh_surface_area_public.py); benchmark [benchmarks/mesh_surface_area.py](../benchmarks/mesh_surface_area.py).

<a id="api-measure-moments"></a>
- [ ] **`measure.moments`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, order=3, *, spacing=None)`.
  - Required focus: order, spacing, accumulator precision and dimensionality. Cases to close: high order, spacing and large-coordinate images.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse separable coordinate-power reductions for raw N-D moments.
  - Evidence locators: parity [tests/test_moments_public.py](../tests/test_moments_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-moments-central"></a>
- [ ] **`measure.moments_central`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, center=None, order=3, *, spacing=None, **kwargs)`.
  - Required focus: center=None, order, deprecated kwargs and cancellation. Cases to close: large-offset objects and explicit centers.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse raw/coordinate reductions about supplied or inferred centroid.
  - Evidence locators: parity [tests/test_moments_public.py](../tests/test_moments_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-moments-coords"></a>
- [ ] **`measure.moments_coords`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(coords, order=3)`.
  - Required focus: coordinate orientation, empty points and order. Cases to close: integer/float point clouds.
  - Easiest remaining route: Audit/extend existing native implementation: Accumulate coordinate monomials for point sets.
  - Evidence locators: parity [tests/test_moments_coords_public.py](../tests/test_moments_coords_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-moments-coords-central"></a>
- [ ] **`measure.moments_coords_central`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(coords, center=None, order=3)`.
  - Required focus: inferred center, empty points and floating cancellation. Cases to close: translated point clouds and explicit centers.
  - Easiest remaining route: Audit/extend existing native implementation: Center point coordinates then accumulate monomials.
  - Evidence locators: parity [tests/test_moments_coords_public.py](../tests/test_moments_coords_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-moments-hu"></a>
- [ ] **`measure.moments_hu`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(nu)`.
  - Required focus: coefficient/order/sign and 2D input contract. Cases to close: reflected/rotated shapes and near-zero invariants.
  - Easiest remaining route: Audit/extend existing native implementation: Evaluate seven fixed Hu polynomial invariants from normalized moments.
  - Evidence locators: parity [tests/test_moments_derived_public.py](../tests/test_moments_derived_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-moments-normalized"></a>
- [ ] **`measure.moments_normalized`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float32, float64`.
  - Signature: `(mu, order=3, spacing=None)`.
  - Required focus: low-order NaNs, zero mass, order and spacing. Cases to close: scaled shapes and zero mass.
  - Easiest remaining route: Audit/extend existing native implementation: Divide central moments by mass/spacing scale powers.
  - Evidence locators: parity [tests/test_moments_derived_public.py](../tests/test_moments_derived_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-pearson-corr-coeff"></a>
- [ ] **`measure.pearson_corr_coeff`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `float64`.
  - Signature: `(image0, image1, mask=None)`.
  - Required focus: constant input warnings, sample count, mask and p-value tails. Cases to close: constant/near-perfect/uncorrelated channels.
  - Shipped scope: aligned C-contiguous 1D or 2D float64 pairs with 2–1,048,576 input elements and an optional matching Boolean mask. The correlation core is ordered and serial. Two distinct selected pairs use an exact p-value of one. Larger supported samples use one scalar SciPy beta-tail call. Unsupported layouts, dtypes or samples use compatibility routing.
  - Evidence locators: [Public parity](../tests/test_pearson_public.py) · [One-worker benchmark](../benchmarks/pearson.py).
  - The original-only reference contract on `528b047b` pins 70 Windows observations and eight performance cases. The combined local draft passed installed public and relocated SDK checks in both modes. Its timed calls included the scalar SciPy p-value provider. [Reference contract](contracts/pearson-reference-v1.json) · [Installed speed admission](../build/evidence/golden-pearson-integration/local-installed-speed-root-admission-v2.json)
  - Main `9a30600e` passed both complete source checks, 624 hosted Pearson public outcomes across four profiles and two dispatch modes, the linked SDK checks and all three packaging stagers. The eight measured one-worker cases passed at 1.375×–55.503× on the retained WSL minimum install. [Release admission](../build/evidence/golden-pearson-integration/hosted-44887385792/whole-product-root-admission-v1.json)

<a id="api-measure-perimeter"></a>
- [ ] **`measure.perimeter`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, neighborhood=4)`.
  - Required focus: neighborhood=4/8, boundary padding and diagonal weights. Cases to close: exact small shapes and diagonal boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse binary erosion/boundary classification and weighted configuration LUT.
  - Evidence locators: parity [tests/test_perimeter_public.py](../tests/test_perimeter_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-perimeter-crofton"></a>
- [ ] **`measure.perimeter_crofton`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, directions=4)`.
  - Required focus: direction option, border padding and binary conversion. Cases to close: rotated shapes and thin objects.
  - Easiest remaining route: Audit/extend existing native implementation: Count intercept configurations and apply 2/4-direction Crofton weights.
  - Evidence locators: parity [tests/test_perimeter_crofton_public.py](../tests/test_perimeter_crofton_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-points-in-poly"></a>
- [ ] **`measure.points_in_poly`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(points, verts)`.
  - Required focus: vertex hits, horizontal edges, nonfinite points and self-crossings. Cases to close: points on/inside/outside polygon boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Port ray-crossing/winding-compatible inclusion tests with edge handling.
  - Evidence locators: parity [tests/test_polygon_public.py](../tests/test_polygon_public.py); benchmark [benchmarks/polygon.py](../benchmarks/polygon.py).

<a id="api-measure-profile-line"></a>
- [ ] **`measure.profile_line`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, src, dst, linewidth=1, order=None, mode='reflect', cval=0.0, *, reduce_func=<function mean>)`.
  - Required focus: inclusive endpoint and ceil length; default bool order; mode remapping; source C-order four-tap accumulation; integer half-away rounding/clipping before reduction; negative-stride None output; TypeError callback retry and mutation. Cases to close: fractional/reversed/zero-length profiles, linewidth 0/1/many, RGB channels, raw sampled pixels, strided/endian storage, reducer callbacks and safe Python intake errors; exclude zero spatial axes and unsafe C coordinates from original capture.
  - Easiest remaining route: Audit/extend existing native implementation: Keep literal float64 line/perpendicular coordinates; add a 2D order-0/1 typed sampler beside resample, then preserve flip/channel assembly and literal width reduction.
  - Evidence locators: parity [tests/test_profile_line_public.py](../tests/test_profile_line_public.py); benchmark [benchmarks/profile_line.py](../benchmarks/profile_line.py).

<a id="api-measure-ransac"></a>
- [ ] **`measure.ransac`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(data, model_class, min_samples, residual_threshold, is_data_valid=None, is_model_valid=None, max_trials=100, stop_sample_num=inf, stop_residuals_sum=0, stop_probability=1, rng=None, initial_inliers=None)`.
  - Required focus: rng draw order, validators, dynamic trial bound, initial_inliers and tie selection. Cases to close: fixed samples, failed fits, custom model callbacks and early stops.
  - Easiest remaining route: Preserve seeded sample/control loop with model protocol; accelerate native model residual batches.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-measure-regionprops"></a>
- [ ] **`measure.regionprops`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Signature: `(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None)`.
  - Required focus: cache, spacing/offset, intensity channels, property aliases and extra_properties. Cases to close: every property on tiny canonical and irregular regions.
  - Easiest remaining route: Audit/extend existing native implementation: Build own lazy region objects around component bounds/moments/geometry primitives.
  - Evidence locators: parity [tests/test_regionprops_state_public.py::test_regionprops_state_saved_rows_exact_except_declared_distinct_class](../tests/test_regionprops_state_public.py); benchmark [benchmarks/regionprops_state.py](../benchmarks/regionprops_state.py).

<a id="api-measure-regionprops-table"></a>
- [ ] **`measure.regionprops_table`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(label_image, intensity_image=None, properties=('label', 'bbox'), *, cache=True, separator='-', extra_properties=None, spacing=None)`.
  - Required focus: separator, scalar/vector/object columns, empty labels and extra_properties. Cases to close: all property shapes and empty/noncontiguous labels.
  - Easiest remaining route: Reuse region objects/property kernels and flatten outputs into columns.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-measure-shannon-entropy"></a>
- [ ] **`measure.shannon_entropy`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, base=2)`.
  - Required focus: base validation, dtype/NaN equality and empty image. Cases to close: binary/multilevel and continuous-valued images.
  - Easiest remaining route: Audit/extend existing native implementation: Count unique pixel symbols and reduce entropy in requested base.
  - Evidence locators: parity [tests/test_shannon_entropy_public.py](../tests/test_shannon_entropy_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-measure-subdivide-polygon"></a>
- [ ] **`measure.subdivide_polygon`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float64`.
  - Signature: `(coords, degree=2, preserve_ends=False)`.
  - Required focus: degree support, closure and preserve_ends behavior. Cases to close: open/closed polygons and short vertex sequences.
  - Easiest remaining route: Audit/extend existing native implementation: Apply degree-specific B-spline mask convolution to ordered vertices.
  - Evidence locators: parity [tests/test_subdivide_polygon_integration.py::test_subdivide_registered_public_frozen_contract](../tests/test_subdivide_polygon_integration.py); benchmark [benchmarks/subdivide_polygon.py](../benchmarks/subdivide_polygon.py).

</details>

<a id="module-metrics"></a>
### metrics — 10 public keys

<details><summary>Expand metrics contract checklist</summary>

<a id="api-metrics-adapted-rand-error"></a>
- [ ] **`metrics.adapted_rand_error`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Signature: `(image_true=None, image_test=None, *, table=None, ignore_labels=(0,), alpha=0.5)`.
  - Required focus: ignore_labels, supplied table, alpha, singleton/all-singleton zero divisions and returned order. Cases to close: exact tiny partitions, singleton/all-singleton cases and sparse large labels.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse contingency counts and self-pair-excluding n*(n-1) sums for joint and marginal totals; compute precision, recall and alpha-weighted error.
  - Evidence locators: parity [tests/test_label_pair_rand_public.py](../tests/test_label_pair_rand_public.py); benchmark [benchmarks/label_pair_metrics.py](../benchmarks/label_pair_metrics.py).

<a id="api-metrics-contingency-table"></a>
- [ ] **`metrics.contingency_table`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Signature: `(im_true, im_test, *, ignore_labels=None, normalize=False, sparse_type='matrix')`.
  - Required focus: ignore_labels, normalization, sparse_type and label index range. Cases to close: sparse high-ID labels and all-ignored pixels.
  - Easiest remaining route: Audit/extend existing native implementation: Stable pair-label counts into CSR sparse matrix/array adapter.
  - Evidence locators: parity [tests/test_contingency_table_public.py](../tests/test_contingency_table_public.py); benchmark [benchmarks/label_pair_metrics.py](../benchmarks/label_pair_metrics.py).

<a id="api-metrics-hausdorff-distance"></a>
- [ ] **`metrics.hausdorff_distance`** — full contract closed.
  - Current scoped progress: Candidate `8deb0d48` reuses the admitted minimum installed public and SDK product. Fixed one-worker speed passed for 10 fixed standard and modified cases at 6.496–34.794× across the measured Boolean 1D–3D domain. Exact hosted qualification is underway. [Installed admission](../build/evidence/u182-first-integration/minimum-local-root-admission-v1.json) · [Timing](../build/evidence/u182-first-integration/installed-speed-root-admission-v1.json)
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0, image1, method='standard')`.
  - Required focus: standard versus modified mean, empty sets and max ties. Cases to close: sparse/dense/equidistant 2D/3D masks.
  - Easiest remaining route: Extract foreground points and compute bidirectional nearest-neighbor distances.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - The family-only Windows capture recorded 28 original outcomes across distance and pair. Equal-rank inputs can have different shapes. Native parity, installed SDK, timing and release are pending. [Original capture log](../build/evidence/u182-hausdorff/original.log)

<a id="api-metrics-hausdorff-pair"></a>
- [ ] **`metrics.hausdorff_pair`** — full contract closed.
  - Current scoped progress: Candidate `8deb0d48` reuses the admitted minimum installed public and SDK product. Fixed one-worker speed passed for five fixed cases at 6.272–30.137× across the measured Boolean 1D–3D domain, including the order of the attaining pair. Exact hosted qualification is underway. [Installed admission](../build/evidence/u182-first-integration/minimum-local-root-admission-v1.json) · [Timing](../build/evidence/u182-first-integration/installed-speed-root-admission-v1.json)
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0, image1)`.
  - Required focus: empty-set warning, coordinate order and tie selection. Cases to close: multiple maximizing pairs and empty masks.
  - Easiest remaining route: Reuse Hausdorff nearest-neighbor search and return attaining point pair.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - The shared Windows original capture includes empty sets and attaining-pair tie cases. The native route still needs to establish exact tie selection, coordinate representation and installed behavior. [Original capture log](../build/evidence/u182-hausdorff/original.log)

<a id="api-metrics-mean-squared-error"></a>
- [ ] **`metrics.mean_squared_error`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image0, image1)`.
  - Required focus: integer promotion, strided/reversed layout and floating reduction order. Cases to close: retained reverse-stride cases and high-dynamic-range differences.
  - Easiest remaining route: Audit/extend existing native implementation: Audit existing native dtype promotion and squared-difference reduction.
  - Evidence locators: parity [tests/test_metrics_reductions.py](../tests/test_metrics_reductions.py); benchmark [benchmarks/metrics_reductions.py](../benchmarks/metrics_reductions.py).

<a id="api-metrics-normalized-mutual-information"></a>
- [ ] **`metrics.normalized_mutual_information`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image0, image1, *, bins=100)`.
  - Required focus: bins, zero entropy, shape padding and histogram edges. Cases to close: constant/independent/identical images and mismatched shape.
  - Easiest remaining route: Audit/extend existing native implementation: Pad unequal shapes, joint histogram and entropy ratio.
  - Evidence locators: parity [tests/test_normalized_mutual_information_public.py](../tests/test_normalized_mutual_information_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-metrics-normalized-root-mse"></a>
- [ ] **`metrics.normalized_root_mse`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image_true, image_test, *, normalization='euclidean')`.
  - Required focus: euclidean/min-max/mean choices, zero denominator and dtype. Cases to close: constant/zero/shifted images for every normalization.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native MSE and selected norm denominator.
  - Evidence locators: parity [tests/test_metrics_reductions.py](../tests/test_metrics_reductions.py); benchmark [benchmarks/metrics_reductions.py](../benchmarks/metrics_reductions.py).

<a id="api-metrics-peak-signal-noise-ratio"></a>
- [ ] **`metrics.peak_signal_noise_ratio`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image_true, image_test, *, data_range=None)`.
  - Required focus: data_range inference/warnings, zero MSE and invalid range. Cases to close: exact matches and integer/float ranges.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native MSE then dtype/range-aware logarithmic score.
  - Evidence locators: parity [tests/test_metrics_reductions.py](../tests/test_metrics_reductions.py); benchmark [benchmarks/metrics_reductions.py](../benchmarks/metrics_reductions.py).

<a id="api-metrics-structural-similarity"></a>
- [ ] **`metrics.structural_similarity`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `float32, float64`.
  - Signature: `(im1, im2, *, win_size=None, gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, full=False, **kwargs)`.
  - Required focus: data_range, sample covariance, channel axis, crop, gaussian weights and return tuple. Cases to close: constant/tiny images, channel layouts and gradient checks.
  - Easiest remaining route: Reuse local mean/variance/covariance filters and SSIM formula; derive gradient/full map paths.
  - Evidence locators: parity [tests/test_structural_similarity.py](../tests/test_structural_similarity.py) and benchmark [benchmarks/structural_similarity.py](../benchmarks/structural_similarity.py).
  - Shipped in `15c79172` for matching C-contiguous 2D grayscale float32/float64 inputs, scalar scores, uniform odd windows, and bounded explicit `data_range`. The owned filter uses applicable SIMD and optional process-global workers. NumPy supplies array and scalar representation. Remaining layouts, channels, gradients, full maps and Gaussian windows use compatibility. Four exact hosted profiles passed 30 SSIM public checks per dispatch mode, with a relocated C++ consumer in both modes. The verified minimum Linux binary beat scikit-image 0.26.0 in all eight one-thread public benchmarks at 1.230×–7.108× in the retained WSL runtime. [Four-profile admission](../build/ssim-grayscale-v1/integration-genuine-admission-v39.json) · [Hosted qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/44692035719) · [Benchmark samples](../build/ssim-grayscale-v1/hosted-minimum-sustained.batches.jsonl)

<a id="api-metrics-variation-of-information"></a>
- [ ] **`metrics.variation_of_information`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open. Current public NumPy log2/matmul route passed qualification; [root current VI4 performance admission](evidence/vi-public-matmul-minimum-performance-root-admission-v1.json) passes four fixed one-thread public cases with4wins/56samples/8warmupgroups on the actual minimum host. [Raw results and scope](../benchmarks/vi-public-matmul-minimum-initial-v1/README.md) retain successful-runtime assertion grading and un-emitted raw-buffer limits.
  - Historical failed gate: [current16 default public witness43765280748](evidence/vi-avx512-default-public-word-root-failure-admission-v1.json), controllerbbe, fails exact execute136 float64 words on genuine minimum1.24.4/Python3.11.17/Intel6975P-C/SKX. First output differs1ULP, second matches; exactly1original+1native. Inputs/targetFP/native phases/caller restoration/source/postidentity pass. Later137-139 remain unverified; no control/timing/B credit. Earlier373 missing words stay unavailable and its cause is not relabeled.
  - Signature: `(image0=None, image1=None, *, table=None, ignore_labels=())`.
  - Required focus: image versus table input, ignore_labels, zeros and result order. Cases to close: merge/split partitions with known conditional entropies. The NumPy CPU-provider axis now has a demonstrated AVX512 exact-word gap; pre-cleanup software/worker/alias capture and controlled public intervention remain pending.
  - Easiest remaining route: Reuse normalized contingency counts and conditional entropy reductions.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current scoped progress: [qualified16a5 candidate](evidence/mark-lbp16a5-whole-product-root-admission-v2.json) passes its four-profile product/SDK scope; shipping remains271/upstream-only for this API. Earlier component/reduction evidence and failures retain their scopes. [bbe diagnostic source](evidence/vi-avx512-public-default-root-source-admission-v1.json) passed255complete source checks perPython; [Correctedcb863019 late-metadata source](evidence/vi-avx512-public-v2-root-source-admission-v1.json) now passes281complete controls perPython/97rootverifiedproofs. Its [actual conditional allocation was ineligible](evidence/vi-avx512-public-v2-ineligible-host-root-closeout-v1.json), with zero targets and no control. Root admits the explicit publicNumPylog2 batch-provider successor `114ed9b3` and its ordinary private build: [source/compile receipt](../build/vi-public-log2-provider-v1/root-source-compile-admission-v1.json), 1706 affected source controls on each retained Python, five compile/install stages exit0, 216 private installed files/108 SDK members. The [fresh four-pair private witness](evidence/vi-public-log2-private-four-case-root-admission-v1.json) now passes exact float64[2] words/ownership/targetFP/caller32/global restoration/postidentity: four original+four native calls/eight publiclog2 delegations,35FPsaves/14restores, no timing/retry. Its Python3.11.14/AVX2 scope is separate from genuine975/AVX512/release; existing16 qualification cannot qualify changed VI. Owning SDK successor nowpasses [ordinary/runtime-disabled](evidence/vi-public-log2-private-owning-sdk-root-admission-v1.json) and [true compiler-scalar/ASan+UBSan](evidence/vi-public-log2-private-sdk-current-modes-root-admission-v1.json) scopes,20caller32+20refusal32 rows perrun/default1; disclosed scalarlog2callback is separate frompublicUF parity. The [new16node private run](evidence/vi-public-log2-private-large-pairs-root-admission-v1.json) passeslegacy8 pluscurrent8 original/native comparisons at workers1/4,16realUF batches/88Creturns/exactwords+FP+caller+identity. Primitive120FPsave/48restore/16ABI/glibc budgets are source-derived, notinstrumented totals. Completef10source173 failed27metadata assertions/no313; [failurecloseout](../build/current281-final-source-v3/full173-failure-closeout-v2.json) preserved and current source-only correction2051 is pending integratedchecks then full173both. Freshgenuinefourprofiles/actualhostcapacity/SKXaxis, VI4 one-threadpublicperformance/alllosses, full domains/backend axes andmain delivery remainpending; no full-contractcheckbox closes.

</details>

<a id="module-morphology"></a>
### morphology — 47 public keys

<details><summary>Expand morphology contract checklist</summary>

<a id="api-morphology-area-closing"></a>
- [ ] **`morphology.area_closing`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None)`.
  - Required focus: parent/tree_traverser reuse, inversion dtype, connectivity and threshold equality. Cases to close: nested dark plateaus with/without supplied tree.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse reviewed private inverted max-tree candidate; accumulate component area and reconstruct surviving minima.
  - Evidence locators: parity [tests/test_max_tree_public.py](../tests/test_max_tree_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-area-opening"></a>
- [ ] **`morphology.area_opening`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None)`.
  - Required focus: parent/traverser consistency, equal-area threshold and stable levels. Cases to close: bright plateaus and supplied/recomputed trees.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse reviewed private max-tree area attribute filter.
  - Evidence locators: parity [tests/test_max_tree_public.py](../tests/test_max_tree_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-ball"></a>
- [ ] **`morphology.ball`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(radius, dtype=<class 'numpy.uint8'>, *, strict_radius=True, decomposition=None)`.
  - Required focus: strict_radius, dtype, approximation contract and sequence order. Cases to close: small exact and large decomposed balls.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native 3D radius predicate and supported sequence decomposition.
  - Evidence locators: parity [tests/test_footprint_contracts.py](../tests/test_footprint_contracts.py); benchmark [benchmarks/footprints.py](../benchmarks/footprints.py).

<a id="api-morphology-binary-closing"></a>
- [ ] **`morphology.binary_closing`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='ignore')`.
  - Required focus: deprecated API warnings, mode ignore/min/max, even footprint origin and out alias. Cases to close: border-touching holes and asymmetric footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native binary dilation then erosion with binary border policy.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-binary-dilation"></a>
- [ ] **`morphology.binary_dilation`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='ignore')`.
  - Required focus: reflected versus gray footprint, mode and output dtype. Cases to close: asymmetric even footprints and border objects.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native boolean max stencil with binary footprint orientation.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-binary-erosion"></a>
- [ ] **`morphology.binary_erosion`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='ignore')`.
  - Required focus: mode, even footprint anchor and out aliasing. Cases to close: all-true images and border erosion.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native boolean min stencil and neutral outside value.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-binary-opening"></a>
- [ ] **`morphology.binary_opening`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='ignore')`.
  - Required focus: sequence order, even footprints, warning and out buffers. Cases to close: thin components and border cases.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native erosion then dilation using binary policy.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-black-tophat"></a>
- [ ] **`morphology.black_tophat`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='reflect', cval=0.0)`.
  - Required focus: subtraction dtype/overflow, out overlap and mode/cval. Cases to close: dark small structures and integer extremes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native grayscale closing minus source.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-closing"></a>
- [ ] **`morphology.closing`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='reflect', cval=0.0)`.
  - Required focus: footprint reflection/padding, mode/cval, dtype and out alias. Cases to close: even/asymmetric footprints and retained dtype cases.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native dilation-erosion composition.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-convex-hull-image"></a>
- [ ] **`morphology.convex_hull_image`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, uint8` (bounded canonical binary arrays).
  - The modern private bounded case set passed 36 checks and all four fixed one-worker comparisons at 1.411×–23.026×. Shipped on `b1d67cc0` in the bounded 2D native domain. Installed and relocated SDK checks passed in normal and forced-scalar modes. [Delivery](../build/evidence/u182-post-haar/next-hull-release-reader-v1/hosted-current308-45219397159/root-main-delivery-v1.json) [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45219397159)
  - Signature: `(image, offset_coordinates=True, tolerance=1e-10, include_borders=True)`.
  - Required focus: offset_coordinates, include_borders, tolerance, empty input and N-D hull dependency. Cases to close: concave/border/degenerate shapes and 2D/3D hulls.
  - Easiest remaining route: Collect occupied boundary/offset points, compute convex hull and classify grid support.
  - Evidence locators: parity [installed public cases](../tests/test_convex_hull_image_public.py); benchmark [fixed public calls](../benchmarks/convex_hull_image.py).
  - All 17 modern stock outputs, storage and provider traces matched Windows and minimum. Two degenerate Qhull cases have profile-specific diagnostic text. Modern native and fixed speed passed on the bounded private case set. [Modern stock receipt](../build/evidence/u182-post-haar/lookahead-planar-hull-modern-v1/handoff.json)

<a id="api-morphology-convex-hull-object"></a>
- [ ] **`morphology.convex_hull_object`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, uint8` (bounded canonical binary arrays).
  - The modern private object route passed 36 checks and all four fixed one-worker comparisons at 3.130×–180.171×. Shipped on `b1d67cc0` in the bounded 2D native domain. Installed and relocated SDK checks passed in normal and forced-scalar modes. [Delivery](../build/evidence/u182-post-haar/next-hull-release-reader-v1/hosted-current308-45219397159/root-main-delivery-v1.json) [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45219397159)
  - Signature: `(image, *, connectivity=2)`.
  - Required focus: connectivity, component overlap and accepted dimensionality. Cases to close: touching/disconnected concave components.
  - Easiest remaining route: Label objects then reuse per-component convex_hull_image and combine masks.
  - Evidence locators: parity [installed public cases](../tests/test_morphology_convex_hull_object.py); benchmark [fixed public calls](../benchmarks/convex_hull_object.py).

<a id="api-morphology-diameter-closing"></a>
- [ ] **`morphology.diameter_closing`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None)`.
  - Required focus: closing output dtype, extent definition, connectivity and threshold equality. Cases to close: long thin dark structures and dtype extremes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse private inverted max-tree bounding-box extent filter.
  - Evidence locators: parity [tests/test_max_tree_public.py](../tests/test_max_tree_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-diameter-opening"></a>
- [ ] **`morphology.diameter_opening`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None)`.
  - Required focus: axis extents, supplied tree and threshold equality. Cases to close: thin bright structures and anisotropic boxes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse private max-tree component bounding-box extent filter.
  - Evidence locators: parity [tests/test_max_tree_public.py](../tests/test_max_tree_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-diamond"></a>
- [ ] **`morphology.diamond`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(radius, dtype=<class 'numpy.uint8'>, *, decomposition=None)`.
  - Required focus: radius type, dtype and iteration counts. Cases to close: radius zero and decomposition equivalence.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native 2D L1-radius predicate/decomposition.
  - Evidence locators: parity [tests/test_footprint_contracts.py](../tests/test_footprint_contracts.py); benchmark [benchmarks/footprints.py](../benchmarks/footprints.py).

<a id="api-morphology-dilation"></a>
- [ ] **`morphology.dilation`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='reflect', cval=0.0)`.
  - Required focus: NaN ordering, footprint sequences, reflected origin and modes. Cases to close: NaNs and asymmetric/even footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native typed neighborhood maximum.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-disk"></a>
- [ ] **`morphology.disk`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(radius, dtype=<class 'numpy.uint8'>, *, strict_radius=True, decomposition=None)`.
  - Required focus: strict_radius, sequence/crosses option and approximate cells. Cases to close: radius sweep at strict boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native 2D radial predicate and source-qualified decomposition.
  - Evidence locators: parity [tests/test_footprint_contracts.py](../tests/test_footprint_contracts.py); benchmark [benchmarks/footprints.py](../benchmarks/footprints.py).

<a id="api-morphology-ellipse"></a>
- [ ] **`morphology.ellipse`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(width, height, dtype=<class 'numpy.uint8'>, *, decomposition=None)`.
  - Required focus: width/height semantics, dtype and decomposition equivalence. Cases to close: eccentric and zero-width/height cases.
  - Easiest remaining route: Audit/extend existing native implementation: Construct exact discrete ellipse footprint and optional crosses sequence.
  - Evidence locators: parity [tests/test_footprint_shapes_public.py](../tests/test_footprint_shapes_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-erosion"></a>
- [ ] **`morphology.erosion`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='reflect', cval=0.0)`.
  - Required focus: NaN ordering, cval/mode, footprint origin and out overlap. Cases to close: signed extremes and boundary modes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native typed neighborhood minimum.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-flood"></a>
- [ ] **`morphology.flood`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 4/4 cases >=1.0, 1.027-1.403x. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): 4/4 cases pass1.081–1.356x. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private speed candidate `6718fa46` (2026-11-17): [all four flood cases,1.128–1.363x meet parity](../build/speed-parity-flood-v1/root-performance-admission-iteration-2.json); [161 current runtime checks/159 exact caller32 records](../build/speed-parity-flood-v1/root-runtime-admission-iteration-2.json), with unchanged52copy/79SDK evidence reused. Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1. Genuine profiles/release performance/delivery pending; reductions and cast providers remain disclosed. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Candidate B~ admitted: four flood cases/four wins with real-cohort/helper/authority closure on the qualified minimum-host domain. [Root performance receipt](evidence/current281-radon-flood-performance-root-admission-v1.json) and [published raw cases](../benchmarks/current281-minimum-initial-v1/README.md). Main/full-contract B/R gates remain open.
  - Signature: `(image, seed_point, *, footprint=None, connectivity=None, tolerance=None)`.
  - Required focus: seed bounds, tolerance promotion, connectivity versus footprint and NaN. Cases to close: exact/tolerant 2D/3D connected regions.
  - Easiest remaining route: Queue-based connected tolerance predicate from seed using shared offsets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current scoped progress: Source/readiness `7c14ee86` root-admitted; 192 original observations captured once across two profiles and independently root-admitted; native implementation assigned. Private native/cost successors are root-admitted within the scopes above. NumPy reductions/conversion providers remain; current shared projection, genuine worker/alias/hot-swap/SDK/performance/release stay open. Aliases share canonical operations.

<a id="api-morphology-flood-fill"></a>
- [ ] **`morphology.flood_fill`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 4/4 cases >=1.0, 1.063-1.184x. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): 2/4 pass; float32 C0.948235x/F0.982839x fail parity. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private speed candidate `6718fa46` (2026-11-17): [all four fill cases,1.018–1.170x meet parity](../build/speed-parity-flood-v1/root-performance-admission-iteration-2.json); [161 current runtime checks/159 exact caller32 records](../build/speed-parity-flood-v1/root-runtime-admission-iteration-2.json), with unchanged52copy/79SDK evidence reused. Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1. Genuine profiles/release performance/delivery pending; reductions and cast providers remain disclosed. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Candidate B~ admitted: four flood_fill cases/four losses, shared executor and authority/cohort/helper closure verified. [Root performance receipt](evidence/current281-radon-flood-performance-root-admission-v1.json) and [all published losses/raw](../benchmarks/current281-minimum-initial-v1/README.md). Optimization, broader domains and main delivery remain open.
  - Signature: `(image, seed_point, new_value, *, footprint=None, connectivity=None, tolerance=None, in_place=False)`.
  - Required focus: in_place, new_value casting, overlap and return identity. Cases to close: typed replacements and in-place aliases.
  - Easiest remaining route: Reuse flood mask then dtype-correct replacement into copy or original.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current scoped progress: Source/readiness `7c14ee86` root-admitted; 192 original observations captured once across two profiles and independently root-admitted; native implementation assigned. Private native/cost successors are root-admitted within the scopes above. NumPy reductions/conversion providers remain; current shared projection, genuine worker/alias/hot-swap/SDK/performance/release stay open. Aliases share canonical operations.

<a id="api-morphology-footprint-from-sequence"></a>
- [ ] **`morphology.footprint_from_sequence`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(footprints)`.
  - Required focus: tuple validation, dimensionality, repetitions and origin/shape. Cases to close: asymmetric/even footprints and repeated sequence steps.
  - Easiest remaining route: Audit/extend existing native implementation: Apply repeated dilations to singleton seed to materialize sequence support.
  - Evidence locators: parity [tests/test_footprint_shapes_public.py](../tests/test_footprint_shapes_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-footprint-rectangle"></a>
- [ ] **`morphology.footprint_rectangle`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(shape, *, dtype=<class 'numpy.uint8'>, decomposition=None)`.
  - Required focus: zero/negative shape, dtype, decomposition warnings and order. Cases to close: N-D odd/even shapes and decomposition variants.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native all-ones N-D shape and sequence/separable decomposition.
  - Evidence locators: parity [tests/test_footprint_contracts.py](../tests/test_footprint_contracts.py); benchmark [benchmarks/footprints.py](../benchmarks/footprints.py).

<a id="api-morphology-h-maxima"></a>
- [ ] **`morphology.h_maxima`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, h, footprint=None)`.
  - Required focus: integer saturation, floating resolution correction, h=0 and footprint. Cases to close: maxima at exactly h and dtype limits.
  - The delivered H-extrema audit passed the shared 886-case suite in normal and SIMD-disabled modes on each qualified profile. The source-unsafe cases remain outside the numerical claim. [Package admission](../build/speed-parity-ci-correction-v1/release-entry-v1/run44245395473-v1/root-whole-product-admission-v1.json)
  - Easiest remaining route: Audit/extend existing native implementation: Reuse grayscale reconstruction of intensity-shifted seed then h-difference test.
  - Evidence locators: parity [tests/test_h_extrema_public.py](../tests/test_h_extrema_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-h-minima"></a>
- [ ] **`morphology.h_minima`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, h, footprint=None)`.
  - Required focus: integer saturation, floating correction, h=0 and minima threshold. Cases to close: minima at exactly h and dtype limits.
  - Included in the shared qualified H-extrema audit and the installed SDK checks. Full dtype and footprint coverage remains open. [Exact release](https://github.com/bipxx/scikit-imagexx/actions/runs/44245395473)
  - Easiest remaining route: Audit/extend existing native implementation: Reuse erosion reconstruction of upward-shifted seed then h-difference test.
  - Evidence locators: parity [tests/test_h_extrema_public.py](../tests/test_h_extrema_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-isotropic-closing"></a>
- [ ] **`morphology.isotropic_closing`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: radius inequality, spacing, border convention and out. Cases to close: large-radius masks and anisotropic spacing.
  - Easiest remaining route: Compose exact-distance binary dilation then erosion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private exactness and relocated SDK passed. Both one-worker Windows closing cases passed, with the exact label-expansion dependency. Installed integration and release remain pending. [Private admission](../build/evidence/u182-isotropic/root-private-admission-v1.json)

<a id="api-morphology-isotropic-dilation"></a>
- [ ] **`morphology.isotropic_dilation`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: radius equality, all-true/all-false input and spacing. Cases to close: equidistant boundary pixels and large radii.
  - Easiest remaining route: Threshold distance transform of complement.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private exactness and relocated SDK passed. Both one-worker Windows dilation cases passed, with the exact label-expansion dependency. Installed integration and release remain pending. [Private admission](../build/evidence/u182-isotropic/root-private-admission-v1.json)

<a id="api-morphology-isotropic-erosion"></a>
- [ ] **`morphology.isotropic_erosion`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: strict versus nonstrict radius, outside-image convention and out. Cases to close: thin features and all-true images.
  - Easiest remaining route: Threshold foreground distance transform.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private exactness and relocated SDK passed. Both one-worker Windows erosion cases passed, with the exact label-expansion dependency. Installed integration and release remain pending. [Private admission](../build/evidence/u182-isotropic/root-private-admission-v1.json)

<a id="api-morphology-isotropic-opening"></a>
- [ ] **`morphology.isotropic_opening`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: intermediate border effects, radius and out. Cases to close: sparse/small objects and large radii.
  - Easiest remaining route: Compose exact-distance erosion then dilation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private exactness and relocated SDK passed. Both one-worker Windows opening cases passed, with the exact label-expansion dependency. Installed integration and release remain pending. [Private admission](../build/evidence/u182-isotropic/root-private-admission-v1.json)

<a id="api-morphology-label"></a>
- [ ] **`morphology.label`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(label_image, background=None, return_num=False, connectivity=None)`.
  - Canonical alias: `measure.label`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: signed/unsigned dtype, background, connectivity, return_num and numbering. Cases to close: adversarial equal-value components and strided 2D/3D.
  - Easiest remaining route: Exact alias of measure.label; reuse its route: Extend existing deterministic equality-based union-find labels. No extra algorithm credit..
  - Evidence locators: parity [tests/test_connected_components_contracts.py](../tests/test_connected_components_contracts.py); benchmark [benchmarks/components.py](../benchmarks/components.py).

<a id="api-morphology-local-maxima"></a>
- [ ] **`morphology.local_maxima`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: eight fixed-width signed and unsigned integer dtypes.
  - Signature: `(image, footprint=None, connectivity=None, indices=False, allow_borders=True)`.
  - Required focus: allow_borders, connectivity/footprint, indices and plateau tie. Cases to close: border and interior flat maxima.
  - Easiest remaining route: Flood equal-valued plateaus and reject those touching larger neighbors.
  - Evidence locators: parity [tests/test_local_extrema_public.py](../tests/test_local_extrema_public.py) and benchmark [benchmarks/local_extrema.py](../benchmarks/local_extrema.py).
  - Private `4c25a9f6`: centered 2D integer footprints passed scoped contract and native leaf checks. One-thread original/candidate median ratios were 17.322× for tiny, 0.375× for dense, 0.423× for plateaus and 0.348× for indices. The three losses block integration for this domain. [Correctness receipt](../build/evidence/local-extrema-v1/receipt.txt) · [Timing receipt](../build/evidence/local-extrema-v1/one-worker-receipt.txt)
  - Shipped in `f493cec8` for native-endian 2D integers with both dimensions at least three, bounded NumPy storage, standard connectivity or its centered footprint, and mask or index output. Traversal is serial. NumPy normalizes the input and supplies mask storage and `nonzero`. Other domains use compatibility. The four fixed cases passed at 1.426×–27.851× on the exact hosted minimum binary in the retained WSL runtime. [Benchmark and release admission](../build/evidence/local-extrema-v1/hosted-44881239674/whole-product-root-admission-v1.json)

<a id="api-morphology-local-minima"></a>
- [ ] **`morphology.local_minima`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: eight fixed-width signed and unsigned integer dtypes.
  - Signature: `(image, footprint=None, connectivity=None, indices=False, allow_borders=True)`.
  - Required focus: allow_borders, connectivity/footprint, indices and dtype extremes. Cases to close: border and interior flat minima.
  - Easiest remaining route: Reuse plateau flood with reversed comparison.
  - Evidence locators: parity [tests/test_local_extrema_public.py](../tests/test_local_extrema_public.py) and benchmark [benchmarks/local_extrema.py](../benchmarks/local_extrema.py).
  - Private `4c25a9f6`: centered 2D integer footprints passed scoped contract and native leaf checks. One-thread original/candidate median ratios were 18.748× for tiny, 0.384× for dense, 0.453× for plateaus and 0.356× for indices. The three losses block integration for this domain. [Correctness receipt](../build/evidence/local-extrema-v1/receipt.txt) · [Timing receipt](../build/evidence/local-extrema-v1/one-worker-receipt.txt)
  - Shipped in `f493cec8` with the same integer, storage, footprint and output limits as local maxima. The serial native traversal reverses the level comparison. The four fixed cases passed at 1.481×–32.760× on the exact hosted minimum binary in the retained WSL runtime. [Benchmark and release admission](../build/evidence/local-extrema-v1/hosted-44881239674/whole-product-root-admission-v1.json)

<a id="api-morphology-max-tree"></a>
- [ ] **`morphology.max_tree`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, connectivity=1)`.
  - Required focus: plateau canonicalization, parent dtype, connectivity and total order. Cases to close: equal plateaus, NaNs if admitted, 2D/3D trees.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse private deterministic level sorting, union-find parent construction and traversal candidate.
  - Evidence locators: parity [tests/test_max_tree_public.py](../tests/test_max_tree_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-max-tree-local-maxima"></a>
- [ ] **`morphology.max_tree_local_maxima`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, connectivity=1, parent=None, tree_traverser=None)`.
  - Required focus: supplied tree, label numbering and equal-level plateaus. Cases to close: multiple equal maxima and tree reuse.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse private max-tree leaf/plateau maxima labeling.
  - Evidence locators: parity [tests/test_max_tree_public.py](../tests/test_max_tree_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-medial-axis"></a>
- [ ] **`morphology.medial_axis`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool`.
  - Shipped on `1d9a0bbf` within its bounded 2D topology domain (SciPy EDT and NumPy PCG64 providers preserved). Four-profile qualification [whole-product admission](../build/evidence/u182-post-haar/current312-qualification-v1/whole-product-root-admission-v1.json) ([run 45344078252](https://github.com/bipxx/scikit-imagexx/actions/runs/45344078252)), Windows 1-worker timing `7.911×–19.001×` ([run 45340574164](../build/evidence/u182-post-haar/current312-windows-focus-v1/root-focus-timing-admission-v1.json)), and modern quiet speed `6.704×–14.550×` ([run 45110288114](../build/evidence/u182-post-haar/lookahead-morphology-v1/medial-modern-speed-v1/root-medial-speed-admission-v1.json)); `archive_retained: false`.
  - Signature: `(image, mask=None, return_distance=False, *, rng=None)`.
  - Required focus: RNG state, masked pixels, return_distance and tie order. Cases to close: symmetric shapes, holes and seeded ties.
  - Easiest remaining route: Distance transform then distance/cornerness/seeded-tie ordered topology thinning.
  - Evidence locators: [tests/test_medial_axis_public.py](../tests/test_medial_axis_public.py) · [benchmarks/medial_axis.py](../benchmarks/medial_axis.py).

<a id="api-morphology-mirror-footprint"></a>
- [ ] **`morphology.mirror_footprint`** — full contract closed.
  - Main: `metadata`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(footprint)`.
  - Required focus: view sharing, sequence iteration count and arbitrary rank. Cases to close: asymmetric footprints and sequence views.
  - Easiest remaining route: Reuse owned axis reversal for array or each footprint in sequence.
  - Evidence locators: parity [tests/test_footprint_contracts.py](../tests/test_footprint_contracts.py); benchmark [benchmarks/footprints.py](../benchmarks/footprints.py).

<a id="api-morphology-octagon"></a>
- [ ] **`morphology.octagon`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(m, n, dtype=<class 'numpy.uint8'>, *, decomposition=None)`.
  - Required focus: m/n edge lengths, degenerate shapes and decomposition order. Cases to close: flat/degenerate/large octagons.
  - Easiest remaining route: Audit/extend existing native implementation: Build octagonal binary support and optional rectangle/diamond decomposition.
  - Evidence locators: parity [tests/test_footprint_shapes_public.py](../tests/test_footprint_shapes_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-octahedron"></a>
- [ ] **`morphology.octahedron`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(radius, dtype=<class 'numpy.uint8'>, *, decomposition=None)`.
  - Required focus: radius/dtype and exact sequence support. Cases to close: radius zero and volume footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Generate 3D L1 ball and optional repeated small-footprint decomposition.
  - Evidence locators: parity [tests/test_footprint_shapes_public.py](../tests/test_footprint_shapes_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-opening"></a>
- [ ] **`morphology.opening`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='reflect', cval=0.0)`.
  - Required focus: even footprint shift, sequence order, dtype and out. Cases to close: bright small structures and asymmetric footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native erosion-dilation composition.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-pad-footprint"></a>
- [ ] **`morphology.pad_footprint`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(footprint, *, pad_end=True)`.
  - Required focus: pad_end, odd axes unchanged, dtype and sequence handling. Cases to close: mixed even/odd N-D footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native padding to odd shape with upstream end/start convention.
  - Evidence locators: parity [tests/test_footprint_contracts.py](../tests/test_footprint_contracts.py); benchmark [benchmarks/footprints.py](../benchmarks/footprints.py).

<a id="api-morphology-reconstruction"></a>
- [ ] **`morphology.reconstruction`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(seed, mask, method='dilation', footprint=None, offset=None)`.
  - Required focus: seed-mask ordering, dilation/erosion, footprint/offset, integer extremes, signed-zero representative/provider decision and N-D output ownership. Cases to close: plateau/sparse-marker/3D frozen-reference cases.
  - Easiest remaining route: Audit/extend existing native implementation: Review retained unbuilt draft, then stable queue/rank propagation constrained by mask.
  - Evidence locators: parity [tests/test_reconstruction_public.py](../tests/test_reconstruction_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-remove-objects-by-distance"></a>
- [ ] **`morphology.remove_objects_by_distance`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(label_image, min_distance, *, priority=None, p_norm=2, spacing=None, out=None)`.
  - Required focus: priority ties, p_norm, spacing, min_distance and out dtype. Cases to close: equidistant neighboring objects and custom priority.
  - Easiest remaining route: Build object boundary points/spatial index and greedily remove by priority.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-remove-small-holes"></a>
- [ ] **`morphology.remove_small_holes`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(ar, area_threshold=<DEPRECATED>, connectivity=1, *, max_size=64, out=None)`.
  - Required focus: boolean versus label input, max_size/deprecated area_threshold, connectivity and out. Cases to close: nested holes, threshold equality and warning paths.
  - Easiest remaining route: Audit/extend existing native implementation: Extend native complement-label/count/fill consumer.
  - Evidence locators: parity [tests/test_component_consumers_public.py](../tests/test_component_consumers_public.py); benchmark [benchmarks/component_consumers.py](../benchmarks/component_consumers.py).

<a id="api-morphology-remove-small-objects"></a>
- [ ] **`morphology.remove_small_objects`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(ar, min_size=<DEPRECATED>, connectivity=1, *, max_size=64, out=None)`.
  - Required focus: bool versus integer labels, max_size/deprecated min_size, single-label warning and out. Cases to close: sparse huge IDs and exact-size components.
  - Easiest remaining route: Audit/extend existing native implementation: Extend native label/count/drop consumer.
  - Evidence locators: parity [tests/test_component_consumers_public.py](../tests/test_component_consumers_public.py); benchmark [benchmarks/component_consumers.py](../benchmarks/component_consumers.py).

<a id="api-morphology-skeletonize"></a>
- [ ] **`morphology.skeletonize`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool, uint8`.
  - Shipped on `5b94b20d` within its recorded native domain. Windows installed fixed speed passed for the batch. [Whole-product admission](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/whole-product-root-admission-v1.json) [Windows one-worker evidence](../build/evidence/u182-post-haar/next-five-hosted-speed-v4/actual-result-v1/root-timing-admission-v1.json)
  - Signature: `(image, *, method=None)`.
  - Required focus: method/dimension defaults, topology, iteration order and boolean output. Cases to close: loops/branches/thin bridges and 3D topology.
  - Shipped scope: Bounded C-order 2D canonical Boolean or uint8 input with extents through 512. C++ owns default and explicit Zhang. Explicit Lee and default 3D are compatibility routes.
  - Evidence locators: parity [tests/test_skeletonize_public.py](../tests/test_skeletonize_public.py); benchmark [benchmarks/skeletonize.py](../benchmarks/skeletonize.py).
  - The 18 retained Linux modern Zhang stock cases matched both prior profiles. The two additional Lee witnesses remain compatibility-only. The retained modern private product passed 48 checks and four fixed timings at 1.65×–2.83×. Genuine installed and release gates remain open. [Modern speed](../build/evidence/u182-post-haar/retained-modern-match-peaks-zhang-timing-root-admission-v1.json) · [Modern stock](../build/evidence/u182-next-numerical-lookahead/skeletonize/modern-stock-zhang-v2/readback.json)

<a id="api-morphology-star"></a>
- [ ] **`morphology.star`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(a, dtype=<class 'numpy.uint8'>)`.
  - Required focus: a=0/small sizes, parity and boundary cells. Cases to close: small exhaustive star footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Union axis-aligned and rotated-square supports using exact source shape convention.
  - Evidence locators: parity [tests/test_footprint_shapes_public.py](../tests/test_footprint_shapes_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-thin"></a>
- [ ] **`morphology.thin`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool, uint8`.
  - Shipped on `5b94b20d` within its recorded native domain. Windows installed fixed speed passed for the batch. [Whole-product admission](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/whole-product-root-admission-v1.json) [Windows one-worker evidence](../build/evidence/u182-post-haar/next-five-hosted-speed-v4/actual-result-v1/root-timing-admission-v1.json)
  - Signature: `(image, max_num_iter=None)`.
  - Required focus: max_num_iter, convergence, deletion order and border padding. Cases to close: loops, isolated pixels and truncated iterations.
  - Private `9a0f1a48` on Linux modern matched 20 original calls and passed 21 checks per mode. Four fixed private one-worker comparisons were 6.614x to 70.909x the original. Installed ownership, SDK and release remain open. [Private evidence](../build/evidence/u182-next-numerical-lookahead/thin/retained-modern-stock-component-root-admission-v1.json) [Private speed](../build/evidence/u182-post-haar/retained-modern-thin-circle-timing-root-admission-v1.json)
  - Shipped scope: Bounded C-order 2D canonical Boolean or uint8 input with extents through 512. C++ owns ordered Guo–Hall subiterations. None or 0 converges, and a positive exact integer through 512 caps full iterations. Other layouts and protocols use compatibility.
  - Evidence locators: parity [tests/test_thin_public.py](../tests/test_thin_public.py); benchmark [benchmarks/thin.py](../benchmarks/thin.py).

<a id="api-morphology-white-tophat"></a>
- [ ] **`morphology.white_tophat`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, footprint=None, out=None, *, mode='reflect', cval=0.0)`.
  - Required focus: unsigned/signed subtraction, out alias and boundary behavior. Cases to close: bright structures and dtype extremes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse source minus native opening.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

</details>

<a id="module-registration"></a>
### registration — 3 public keys

<details><summary>Expand registration contract checklist</summary>

<a id="api-registration-optical-flow-ilk"></a>
- [ ] **`registration.optical_flow_ilk`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(reference_image, moving_image, *, radius=7, num_warp=10, gaussian=False, prefilter=False, dtype=<class 'numpy.float32'>)`.
  - Required focus: radius/gaussian weighting, prefilter, dtype, singular tensor and warp count. Cases to close: translated textures, flat fields and 2D/3D motions.
  - Easiest remaining route: Gaussian pyramid, repeated warp, local gradient tensor sums and small per-pixel Lucas-Kanade solves.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-registration-optical-flow-tvl1"></a>
- [ ] **`registration.optical_flow_tvl1`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(reference_image, moving_image, *, attachment=15, tightness=0.3, num_warp=5, num_iter=10, tol=0.0001, prefilter=False, dtype=<class 'numpy.float32'>)`.
  - Required focus: attachment/tightness, median prefilter, stopping tolerance and dtype. Cases to close: discontinuous flows and convergence edge cases.
  - Easiest remaining route: Coarse-to-fine warping with TV-L1 primal/dual updates.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-registration-phase-cross-correlation"></a>
- [ ] **`registration.phase_cross_correlation`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(reference_image, moving_image, *, upsample_factor=1, space='real', disambiguate=False, reference_mask=None, moving_mask=None, overlap_ratio=0.3, normalization='phase')`.
  - Required focus: normalization, space=fourier, overlap_ratio, disambiguation, peak ties and error outputs. Cases to close: integer/subpixel shifts, periodic ambiguity and masks.
  - Easiest remaining route: FFT cross-power peak then localized upsampled DFT; separate masked normalized-correlation branch.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-restoration"></a>
### restoration — 17 public keys

<details><summary>Expand restoration contract checklist</summary>

<a id="api-restoration-ball-kernel"></a>
- [ ] **`restoration.ball_kernel`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float64`.
  - Signature: `(radius, ndim)`.
  - Required focus: radius/ndim, grid extents and center intensity. Cases to close: small/large N-D kernels.
  - Easiest remaining route: Audit/extend existing native implementation: Construct spherical rolling-ball intensity kernel with outside-support infinities.
  - Evidence locators: parity [tests/test_rolling_ball_kernels_integration.py::test_geometry_registered_public_frozen_contract](../tests/test_rolling_ball_kernels_integration.py); benchmark [benchmarks/rolling_ball_kernels.py](../benchmarks/rolling_ball_kernels.py).

<a id="api-restoration-calibrate-denoiser"></a>
- [ ] **`restoration.calibrate_denoiser`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, denoise_function, denoise_parameters, *, stride=4, approximate_loss=True, extra_output=False)`.
  - Required focus: denoise_function identity/callbacks, approximate_loss, stride, extra_output and parameter order. Cases to close: instrumented denoiser with deterministic parameter grid.
  - Easiest remaining route: Enumerate parameter grid and score J-invariant masked loss, returning configured callable.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-cycle-spin"></a>
- [ ] **`restoration.cycle_spin`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(x, func, max_shifts, shift_steps=1, num_workers=<DEPRECATED>, func_kw=None, *, workers=None, channel_axis=None)`.
  - Required focus: shift_steps/channel_axis, deprecated num_workers, callbacks and sum order. Cases to close: identity/custom denoisers and exhaustive shift grids.
  - Easiest remaining route: Enumerate circular shifts, invoke supplied func, unshift and average.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-denoise-bilateral"></a>
- [ ] **`restoration.denoise_bilateral`** — full contract closed.
  - Current scoped progress: Private `badac388` passed 35 checks on each retained profile and six Windows one-worker cases at 1.315–1.769×. Exact scope is bounded float32/float64 grayscale and RGB with constant borders. Installed release and folded borders remain open. [Private root admission](../build/evidence/u182-denoise-bilateral/root-private-admission-v1.json)
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, *, channel_axis=None)`.
  - Required focus: bins quantization, sigma_color default, channels, mode and zero weights. Cases to close: constant/edge/color images and wide windows.
  - Easiest remaining route: Spatial Gaussian times intensity-distance LUT, normalize local weighted sums.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Private `2dcf315` passed all six genuine Linux minimum whole-call comparisons at 1.410×–1.903× over 84 accepted samples. The Windows successor result is recorded above. Modern component timing and installed qualification remain open. [Scoped result](../build/evidence/u182-post-haar/lookahead-bilateral-minimum-speed-v1/root-timing-admission-v1.json)
  - Modern stock captured 28 public controls and six fixed cases. The six fixed prepared and output arrays matched Windows; four differed from Linux minimum. Modern compiled and descriptor parity and fixed speed remain pending. [Modern stock receipt](../build/evidence/u182-post-haar/lookahead-bilateral-modern-stock-v1/handoff.json)

<a id="api-restoration-denoise-invariant"></a>
- [ ] **`restoration.denoise_invariant`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, denoise_function, *, stride=4, masks=None, denoiser_kwargs=None)`.
  - Required focus: masks generator, stride, denoiser_kwargs and callback/error order. Cases to close: fixed masks and instrumented identity/linear denoisers.
  - Easiest remaining route: Build phase masks, interpolate held-out pixels, call supplied denoiser and combine predictions.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-denoise-nl-means"></a>
- [ ] **`restoration.denoise_nl_means`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, patch_size=7, patch_distance=11, h=0.1, fast_mode=True, sigma=0.0, *, preserve_range=False, channel_axis=None)`.
  - Required focus: fast_mode difference, sigma correction, patch/search borders and channels. Cases to close: repeated textures and patch/search size sweeps.
  - Easiest remaining route: Direct patch-distance baseline then fast summed-distance shifts with Gaussian correction.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-denoise-tv-bregman"></a>
- [ ] **`restoration.denoise_tv_bregman`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, weight=5.0, max_num_iter=100, eps=0.001, isotropic=True, *, channel_axis=None)`.
  - Required focus: isotropic/anisotropic shrinkage, Gauss-Seidel order, eps, channels and dtype. Cases to close: frozen 70-case suite plus uncovered options.
  - Easiest remaining route: Audit/extend existing native implementation: Audit accepted split-Bregman native updates and existing cursor optimization.
  - Evidence locators: parity [tests/test_tv_bregman_public.py](../tests/test_tv_bregman_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-restoration-denoise-tv-chambolle"></a>
- [ ] **`restoration.denoise_tv_chambolle`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, weight=0.1, eps=0.0002, max_num_iter=200, *, channel_axis=None)`.
  - Required focus: energy stopping order, strided channels, weight/eps and provider calls. Cases to close: frozen one-worker cases with all slower observations retained.
  - Easiest remaining route: Audit/extend existing native implementation: Audit accepted primal/dual divergence updates and merged single-admission optimization.
  - Evidence locators: parity [tests/test_tv_chambolle_contracts.py](../tests/test_tv_chambolle_contracts.py); benchmark [benchmarks/tv_chambolle.py](../benchmarks/tv_chambolle.py).

<a id="api-restoration-denoise-wavelet"></a>
- [ ] **`restoration.denoise_wavelet`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, sigma=None, wavelet='db1', mode='soft', wavelet_levels=None, convert2ycbcr=False, method='BayesShrink', rescale_sigma=True, *, channel_axis=None)`.
  - Required focus: wavelet/mode/levels, BayesShrink/VisuShrink, sigma scaling and YCbCr channel handling. Cases to close: multiple wavelets, odd sizes and color paths.
  - Easiest remaining route: Reviewed wavelet filter-bank backend, threshold details, then inverse transform.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-ellipsoid-kernel"></a>
- [ ] **`restoration.ellipsoid_kernel`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `float64`.
  - Signature: `(shape, intensity)`.
  - Required focus: shape parity, intensity scaling and outside-support infinities. Cases to close: anisotropic odd/even ellipsoids.
  - Easiest remaining route: Audit/extend existing native implementation: Evaluate scaled ellipsoid cap on requested grid and intensity.
  - Evidence locators: parity [tests/test_rolling_ball_kernels_integration.py::test_geometry_registered_public_frozen_contract](../tests/test_rolling_ball_kernels_integration.py); benchmark [benchmarks/rolling_ball_kernels.py](../benchmarks/rolling_ball_kernels.py).

<a id="api-restoration-estimate-sigma"></a>
- [ ] **`restoration.estimate_sigma`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, average_sigmas=False, *, channel_axis=None)`.
  - Required focus: supported transform/backend, zero detail, channel averaging and dtype. Cases to close: known synthetic noise and constant/odd-shape inputs.
  - Easiest remaining route: Wavelet high-frequency detail coefficients and median absolute-deviation estimate.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-inpaint-biharmonic"></a>
- [ ] **`restoration.inpaint_biharmonic`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, mask, *, split_into_regions=False, channel_axis=None)`.
  - Required focus: boundary stencil, split_into_regions, empty/full mask, channels and solver failure. Cases to close: small exact holes and large/disconnected missing regions.
  - Easiest remaining route: Assemble biharmonic finite-difference sparse equations on mask and solve by region.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-richardson-lucy"></a>
- [ ] **`restoration.richardson_lucy`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, psf, num_iter=50, clip=True, filter_epsilon=None)`.
  - Required focus: filter_epsilon, zero denominators, clip and PSF orientation. Cases to close: delta/asymmetric PSFs and short/long iterations.
  - Easiest remaining route: Iterated PSF convolution, ratio and reversed-PSF correction with reusable direct/FFT paths.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-rolling-ball"></a>
- [ ] **`restoration.rolling_ball`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64`.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Candidate B~ admitted: rolling-ball stream one-thread public comparison, actual cohort/C1/helper and qualifiedDDA runtime/SDK chain accepted. [Exact cases/all losses](../benchmarks/current281-minimum-initial-v1/README.md), [root receipt](evidence/current281-join-rolling-performance-root-admission-v1.json). Full domains, provider/class ownership, optimization and main delivery remain open.
  - Signature: `(image, *, radius=100, kernel=None, nansafe=False, num_threads=<DEPRECATED>, workers=None)`.
  - Required focus: nansafe, kernel origin, radius, workers/num_threads warning and dtype. Cases to close: NaNs, bright background and wide kernels.
  - Easiest remaining route: Min-plus neighborhood erosion using nonflat ball/ellipsoid height kernel.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current scoped progress: Shipping remains upstream-only. Source/readiness and all 150 originals are root-admitted; private c821 native admission covers 750 saved observations plus 60 boundary controls and complete float32/float64 SDK scopes. Provider coercion/casting/custom-kernel arithmetic remain disclosed. The 42f SDK successor now covers 11 saved bool/integer/half/swapped cases with strict captured FP and complete owning output. Unsafe casts and wider half/nonfinite/rounding domains, Windows runtime, broader protocols/FP, integration, genuine release and one-thread original benchmark remain open. The optional global-worker snapshot and ordered scalar neighborhood core have private evidence only; no blanket SIMD or speed claim. Zero additional original/timing authority.

<a id="api-restoration-unsupervised-wiener"></a>
- [ ] **`restoration.unsupervised_wiener`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, psf, reg=None, user_params=None, is_real=True, clip=True, *, rng=None)`.
  - Required focus: RNG draw sequence, user_params callbacks/burnin/threshold and returned chains. Cases to close: saved-state short chains and posterior convergence fixtures.
  - Easiest remaining route: Reuse spectral Wiener conditional sampling with noise/prior precision chain.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-unwrap-phase"></a>
- [ ] **`restoration.unwrap_phase`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, wrap_around=False, rng=None)`.
  - Required focus: 1D/2D/3D branches, masked arrays, wrap_around and RNG tie sequence. Cases to close: phase jumps, residues, masks and periodic boundaries.
  - Easiest remaining route: Reliability-guided sorted-edge region merging with wrap offsets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-restoration-wiener"></a>
- [ ] **`restoration.wiener`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, psf, balance, reg=None, is_real=True, clip=True)`.
  - Required focus: is_real, supplied Fourier PSF/reg, normalization and clip. Cases to close: odd/even PSFs, complex mode and balance extremes.
  - Easiest remaining route: FFT PSF/regularizer and apply conjugate transfer over power-plus-balance regularization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-segmentation"></a>
### segmentation — 20 public keys

<details><summary>Expand segmentation contract checklist</summary>

<a id="api-segmentation-active-contour"></a>
- [ ] **`segmentation.active_contour`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, snake, alpha=0.01, beta=0.1, w_line=0.0, w_edge=1, gamma=0.01, max_px_move=1.0, max_num_iter=2500, convergence=0.1, *, boundary_condition='periodic')`.
  - Required focus: boundary_condition, convergence history, endpoint constraints and interpolation. Cases to close: periodic/fixed/free snakes and near-convergence cases.
  - Easiest remaining route: Sample native image/edge forces along snake; solve banded/cyclic internal-energy system then capped updates.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-chan-vese"></a>
- [ ] **`segmentation.chan_vese`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, mu=0.25, lambda1=1.0, lambda2=1.0, tol=0.001, max_num_iter=500, dt=0.5, init_level_set='checkerboard', extended_output=False)`.
  - Required focus: init_level_set, energy history, stopping rule, dt and extended_output. Cases to close: constant/two-phase images and max-iteration boundaries.
  - Easiest remaining route: Evolve level-set curvature and region-mean data force with source update schedule.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-checkerboard-level-set"></a>
- [ ] **`segmentation.checkerboard_level_set`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(image_shape, square_size=5)`.
  - Required focus: square_size and N-D shape. Cases to close: small exact parity grids and large volumes.
  - Easiest remaining route: Audit/extend existing native implementation: Fill alternating coordinate-block parity as binary level set.
  - Evidence locators: parity [tests/test_level_set_public.py::test_level_set_native_public_contract](../tests/test_level_set_public.py); benchmark [benchmarks/level_set.py](../benchmarks/level_set.py).

<a id="api-segmentation-clear-border"></a>
- [ ] **`segmentation.clear_border`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(labels, buffer_size=0, bgval=0, mask=None, *, out=None)`.
  - Required focus: buffer_size, mask precedence, bgval, source label dtype and out. Cases to close: border/mask-touching labels with aliases.
  - Easiest remaining route: Audit/extend existing native implementation: Extend native component IDs touching border/mask and relabel background.
  - Evidence locators: parity [tests/test_component_consumers_public.py](../tests/test_component_consumers_public.py); benchmark [benchmarks/component_consumers.py](../benchmarks/component_consumers.py).

<a id="api-segmentation-disk-level-set"></a>
- [ ] **`segmentation.disk_level_set`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(image_shape, *, center=None, radius=None)`.
  - Required focus: default center/radius, dimension and strict boundary. Cases to close: even/odd shapes and fractional centers.
  - Easiest remaining route: Audit/extend existing native implementation: Evaluate center/radius radial inclusion on image grid.
  - Evidence locators: parity [tests/test_level_set_public.py::test_level_set_native_public_contract](../tests/test_level_set_public.py); benchmark [benchmarks/level_set.py](../benchmarks/level_set.py).

<a id="api-segmentation-expand-labels"></a>
- [ ] **`segmentation.expand_labels`** — full contract closed.
  - Current scoped progress: Candidate `8deb0d48` reuses the admitted minimum installed public and SDK product. Fixed one-worker speed passed for eight fixed cases at 2.212–216.266× in the measured integer-label subset. The largest speedup was the negative-radius early return. Exact hosted qualification is underway. [Installed admission](../build/evidence/u182-first-integration/minimum-local-root-admission-v1.json) · [Timing](../build/evidence/u182-first-integration/installed-speed-root-admission-v1.json)
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(label_image, distance=1, spacing=1)`.
  - Required focus: equidistant-label tie, spacing, background and output label dtype. Cases to close: competing labels and anisotropic 2D/3D volumes.
  - Easiest remaining route: Reuse exact Euclidean nearest-label indices and threshold distance.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - The family-only Windows capture froze 45 original contract observations and eight benchmark endpoints. Native parity, exact distance and tie behavior, installed SDK, timing and release are pending. [Original capture log](../build/evidence/u182-expand-labels/reference-v1.log)

<a id="api-segmentation-felzenszwalb"></a>
- [ ] **`segmentation.felzenszwalb`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, scale=1, sigma=0.8, min_size=20, *, channel_axis=-1)`.
  - Required focus: edge-order ties, scale/min_size postpass and channel_axis. Cases to close: flat/equal-weight images and fine/coarse partitions.
  - Easiest remaining route: Gaussian preprocessing, sorted neighbor edges and size/internal-difference union-find merges.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-find-boundaries"></a>
- [ ] **`segmentation.find_boundaries`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 4/4 cases >=1.0, 1.625-4.228x. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): 2/4 supplemental64-bit cases pass; int64-C-outer0.891151x/uint64-C-outer0.845565x fail parity. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private shared-kernel successor `9fc98a4e`, source `eadb2ee2` (2026-11-17): [all four supplemental direct64-bit public cases pass,1.073–5.737x](../build/speed-parity-root-v1/find64-pooled-v1/root-find64-performance-admission-v1.json); [474 affected runtime checks/five installed clients/eight wide FP records](../build/speed-parity-root-v1/find64-pooled-v1/root-runtime-admission-v1.json). Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1; int64 C-thick/F-inner/C-outer and uint64 C-outer. The old12 fixed cases only cover<=32-bit/bool; the earlier uint64 loss remains retained. Current genuine/release performance/delivery pending. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Signature: `(label_img, connectivity=1, mode='thick', background=0)`.
  - Required focus: connectivity, background and doubled subpixel shape. Cases to close: triple junctions and thin labels.
  - Easiest remaining route: Audit/extend existing native implementation: Compare neighbor labels and apply thick/inner/outer/subpixel construction.
  - Evidence locators: parity [tests/test_find_boundaries_public.py](../tests/test_find_boundaries_public.py); benchmark [benchmarks/find_boundaries.py](../benchmarks/find_boundaries.py).
  - New outer reference scope: [eight admitted original observations](evidence/mark-wide-outer-eight-original-root-admission-v3.json), four literal3x4 cases across minimum/modern, includes int64/uint64 find and gray/RGB mark. Result records match across profiles; native whole-operation/FP comparison remains pending. No full gate or shipping classification changes.

<a id="api-segmentation-flood"></a>
- [ ] **`segmentation.flood`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 4/4 cases >=1.0, 1.027-1.403x. Canonical morphology entry only; the exact alias adds no independent measurement credit. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): Canonical morphology.flood4/4 pass1.081–1.356x; alias adds no measurement credit. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private speed candidate `6718fa46` (2026-11-17): [all four flood cases,1.128–1.363x meet parity](../build/speed-parity-flood-v1/root-performance-admission-iteration-2.json); [161 current runtime checks/159 exact caller32 records](../build/speed-parity-flood-v1/root-runtime-admission-iteration-2.json), with unchanged52copy/79SDK evidence reused. Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1. Canonical measurement reused for this exact alias; no additional algorithm or measurement credit. Genuine profiles/release performance/delivery pending; reductions and cast providers remain disclosed. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Signature: `(image, seed_point, *, footprint=None, connectivity=None, tolerance=None)`.
  - Canonical alias: `morphology.flood`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: seed bounds, tolerance promotion, connectivity versus footprint and NaN. Cases to close: exact/tolerant 2D/3D connected regions.
  - Easiest remaining route: Exact alias of morphology.flood; reuse its route: Queue-based connected tolerance predicate from seed using shared offsets. No extra algorithm credit..
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current scoped progress: Source/readiness `7c14ee86` root-admitted; 192 original observations captured once across two profiles and independently root-admitted; native implementation assigned. Private native/cost successors are root-admitted within the scopes above. NumPy reductions/conversion providers remain; current shared projection, genuine worker/alias/hot-swap/SDK/performance/release stay open. Aliases share canonical operations.

<a id="api-segmentation-flood-fill"></a>
- [ ] **`segmentation.flood_fill`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 4/4 cases >=1.0, 1.063-1.184x. Canonical morphology entry only; the exact alias adds no independent measurement credit. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): Canonical morphology.flood_fill2/4 pass; float32 C0.948235x/F0.982839x fail parity; alias adds no measurement credit. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private speed candidate `6718fa46` (2026-11-17): [all four fill cases,1.018–1.170x meet parity](../build/speed-parity-flood-v1/root-performance-admission-iteration-2.json); [161 current runtime checks/159 exact caller32 records](../build/speed-parity-flood-v1/root-runtime-admission-iteration-2.json), with unchanged52copy/79SDK evidence reused. Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1. Canonical measurement reused for this exact alias; no additional algorithm or measurement credit. Genuine profiles/release performance/delivery pending; reductions and cast providers remain disclosed. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Signature: `(image, seed_point, new_value, *, footprint=None, connectivity=None, tolerance=None, in_place=False)`.
  - Canonical alias: `morphology.flood_fill`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: in_place, new_value casting, overlap and return identity. Cases to close: typed replacements and in-place aliases.
  - Easiest remaining route: Exact alias of morphology.flood_fill; reuse its route: Reuse flood mask then dtype-correct replacement into copy or original. No extra algorithm credit..
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current scoped progress: Source/readiness `7c14ee86` root-admitted; 192 original observations captured once across two profiles and independently root-admitted; native implementation assigned. Private native/cost successors are root-admitted within the scopes above. NumPy reductions/conversion providers remain; current shared projection, genuine worker/alias/hot-swap/SDK/performance/release stay open. Aliases share canonical operations.

<a id="api-segmentation-inverse-gaussian-gradient"></a>
- [ ] **`segmentation.inverse_gaussian_gradient`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N~ H~ P~ S~ V~ W~ B? R?` (private scoped component admission; no shipping promotion); default dtype slice: `none registered`.
  - Current failed release gate: exact852 has generic half error-domain test failures across all profiles, plus three minimum-profile saved-result byte mismatches in normal/scalar modes. Private accuracy does not waive these failures. [Four-profile closeout](evidence/igg282-failed-campaign-root-closeout-v1.json).
  - Signature: `(image, alpha=100.0, sigma=5.0)`.
  - Required focus: alpha, sigma, dtype and boundaries. Cases to close: smooth/step edges and constant images.
  - Easiest remaining route: Reuse the reviewed Hessian first-derivative correlation pattern; preserve input-dtype derivative/square/sum/sqrt, nearest borders, sigma thresholds and alpha broadcasting. The symmetric Gaussian kernel is insufficient.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - The source-only port passed 131 guards per retained Windows Python. The six old minimum mismatches remain open. A paired same-host original/native stage diagnostic gates further admission. [Source-only admission](../build/evidence/u182-igg/root-source-admission-v1.json)
  - Current scoped progress: Source34d has 79 prospective recipes and 88 pure checks per Python; root source/storage admission passes. Independent/root readiness of 76981161 passes corrected image/alpha/sigma factories and zero-body controls. Root raw admission covers all 158 actual observations, 138 returns/20 errors/six warnings, 316 ordered journal events and 63 decoded typed payloads. All caller/process restoration and within-profile input/state/target-semantic FP checks pass; three result and one error-module profile differences remain intact. [Combined root admission](evidence/inverse-gaussian-gradient-native-root-admission-v1.json) now passes for frozen9b8e and policy successor065cc3a: 16native sources/432smallproofs, independent8file policy delta/35proofs, Linux private owning SDK450 operations and Windows private normal150 operations/24proofs. Fresh successor frontend evidence is normal-mode only on two retained Linux profiles; unchanged native-mode and private Windows SDK evidence has narrower reusable scope. No further original or timing authority; hot swap, acceleration, speed and release remain open.

<a id="api-segmentation-join-segmentations"></a>
- [ ] **`segmentation.join_segmentations`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Candidate B~ admitted: JOIN stream one-thread public comparison, actual cohort/C1/helper and qualifiedDDA runtime/SDK chain accepted. [Exact cases/all losses](../benchmarks/current281-minimum-initial-v1/README.md), [root receipt](evidence/current281-join-rolling-performance-root-admission-v1.json). Full domains, provider/class ownership, optimization and main delivery remain open.
  - Signature: `(s1, s2, return_mapping: bool = False)`.
  - Required focus: dtype overflow, mapping objects, return_mapping and zero labels. Cases to close: sparse huge label pairs and mapping round trips.
  - Easiest remaining route: Pair-label encode then stable sequential relabel with inverse mappings.
  - Evidence locators: parity [tests/test_join_segmentations_integration.py](../tests/test_join_segmentations_integration.py); benchmark [benchmarks/join_segmentations.py](../benchmarks/join_segmentations.py).
  - Current scoped progress: Candidate `e03a0063`, composed in `8c4fb430`; captured-input private matrix admitted. Arbitrary incoming flags, returned-class domains, genuine integration and new performance pending; eight shipped losses retained.

  - Delivered scope: exact a6/main ef465027, [four-profile/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) and [one-thread public cases with every loss](evidence/join-util-views-initial-performance-admission-v1.json). Full domain closure remains unchecked.
  - Optimization candidate: [e03 captured-input private matrix admission](evidence/join-segmentations-matrix-admission-v2.json) passes 252 XML checks and 200 saved rows; authentic numerical incoming states and explicit metadata-only setup are the scope. Returned-class/full-FP/shared genuine/performance gates remain open; eight earlier losses are preserved.

<a id="api-segmentation-mark-boundaries"></a>
- [ ] **`segmentation.mark_boundaries`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64`.
  - Delivered genuine v2 (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-v2-root-result-v1.json), qualified source `e8f459ed`, main `a5f3aa97`: 8/8 cases >=1.0, 1.335-9.734x. [Four profiles/304SDK/48corpora/three stagers](evidence/speed-parity-whole-product-v2-root-admission-v1.json) and [source BOTH](evidence/speed-parity-source-both-v2-root-admission-v1.json) pass. Scope:12700K/Linux3.11.14/NumPy1.24.4/original0.26, allocation included, worker1; private true compiler-scalar/SAN evidence is separate. Full contracts/domains and other-profile performance remain open. [Private precursor](evidence/speed-parity-private24-v2-root-admission-v1.json) and failed predecessor receipts remain retained.
  - Historical genuine candidate `75b21947` (2026-11-17): [fixed-case result](evidence/speed-parity-genuine24-root-result-v1.json): 8/8 cases pass1.086–9.364x. [All4profiles/304SDK/48corpora/3stagers pass](evidence/speed-parity-whole-product-root-admission-v1.json); this batch remains unshipped because four predecessor cases missed the required>=1.0 gate. Measurement:12700K/retained3.11.14/NumPy1.24.4, original0.26, allocation included, worker1. Full domains/contracts remain open.
  - Private speed candidate `39f1c6ce` plus pooled-find successor `9fc98a4e` (2026-11-17): [all8 current public cases pass,1.089–9.063x](../build/speed-parity-root-v1/find64-pooled-v1/root-mark-performance-admission-v1.json); [474 affected MARK/find runtime checks and installed ownership/FP controls](../build/speed-parity-root-v1/find64-pooled-v1/root-runtime-admission-v1.json). Scope:12700K/Linux3.11.14/NumPy1.24.4, allocation included, worker1; provider costs included. Genuine/current release performance/delivery remain pending; historical losses retained. [Four private C++ SDK modes pass](../build/speed-parity-root-v1/sdk-modes-root-admission-v1.json); SDK evidence does not close Python/provider or genuine release gates. [Complete public Python-disabled scope passes](../build/speed-parity-root-v1/public-disabled-root-admission-v1.json); genuine/release/shipping gates stay pending.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Latest scoped candidate (2026-11-15): [Exact16a5 genuine4profile/SDK admission](evidence/mark-lbp16a5-whole-product-root-admission-v2.json) and [private installed compiler-scalar/ASanUBSan named-family admission](evidence/mark-lbp-private-sdk-modes-root-terminal-admission-v8.json) pass. All201savedrows incl22subpixel atworkers1/4 plus8wide24calls/4refusals each; wideFP/caller32/getter checks are compiledassertions, no rawpercase hardwarewords or legacyFP instrumentation claimed. [MARK8 original-public performance](evidence/mark-lbp16a5-minimum-performance-root-admission-v1.json) now passes its bounded gate:3wins/5losses; main remains pending; fullcontract checkbox staysopen. Earlier failedfrontier/campaign notes are historical.
  - Historical domain failure, superseded in the bounded16a5 candidate above: the earlier frozen int64 outer-label benchmark hit the exact-promotion frontier for labels0..6 and produced no timings. [Retained intake](evidence/current281-initial-comparisons-root-intake-v1.json).
  - Signature: `(image, label_img, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)`.
  - Required focus: mode, background_label, subpixel upsampling, image dtype and color. Cases to close: inner/outer/subpixel boundaries and grayscale/RGB images.
  - Easiest remaining route: Reuse find_boundaries and color overlay with optional outline.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Saved original admission](evidence/mark-boundaries-saved-raw-admission-v2.json) covers 125 cases per profile (250 complete observations, 251 consumed calls including the retained failed attempt). One separately authorized fixed-codec witness closed the structured-error gap; no successful original rows were replayed. [Independent native successor admission](evidence/mark-boundaries-fp-successor-admission-v1.json) passes on source 6d4373b5720b5ee959c1d313e14e3d8f754085e4 for 1,500 saved worker/mode comparisons (125 cases per profile, workers 1/4, normal/disabled/scalar), including strict tiny-color target FP. Small RHS palette assignment remains explicit NumPy provider work. Genuine installation, platform/domain closure, one-thread original-public performance and delivery remain open. [Executable benchmark source review](evidence/mark-boundaries-benchmark-source-admission-v1.json) and [handoff reconciliation](evidence/mark-boundaries-benchmark-handoff-reconciliation-v1.json) pass on 446bde47e7462278389c0a87affb881ed775f1ec: 52 source/zero-body controls per Python, fixed 8 cases, no actual targets or timings. Genuine binding, outer first-cohort and separate timing GO remain pending.
  - Current scoped progress: Candidate `8c4fb430`; scoped native/reference evidence above. Complete source readiness admitted; qualification42306808073 minimum import failed, three other profiles continue. Runtime-aware ABI correction is privately admitted; the combined 280-path source plan is staging only. Corrected genuine qualification, public performance and delivery remain pending.
  - New outer reference scope: [eight admitted original observations](evidence/mark-wide-outer-eight-original-root-admission-v3.json), four literal3x4 cases across minimum/modern, includes int64/uint64 find and gray/RGB mark. Result records match across profiles; native whole-operation/FP comparison remains pending. No full gate or shipping classification changes.


<a id="api-segmentation-morphological-chan-vese"></a>
- [ ] **`segmentation.morphological_chan_vese`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, num_iter, init_level_set='checkerboard', smoothing=1, lambda1=1, lambda2=1, iter_callback=<function <lambda>>)`.
  - Required focus: initializer, smoothing cycle state, lambda weights and iter_callback. Cases to close: exact iteration traces on binary shape fixtures.
  - Easiest remaining route: Alternating binary curvature morphology and region-energy threshold updates.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-morphological-geodesic-active-contour"></a>
- [ ] **`segmentation.morphological_geodesic_active_contour`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(gimage, num_iter, init_level_set='disk', smoothing=1, threshold='auto', balloon=0, iter_callback=<function <lambda>>)`.
  - Required focus: threshold=auto, balloon sign, initializer and callback order. Cases to close: weak-edge gaps and balloon pressure signs.
  - Easiest remaining route: Binary balloon/attachment updates plus alternating morphological smoothing.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-quickshift"></a>
- [ ] **`segmentation.quickshift`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, rng=42, *, channel_axis=-1)`.
  - Required focus: rng jitter/ties, Lab conversion, max_dist, return_tree and channels. Cases to close: constant colors, density ties and tree outputs.
  - Easiest remaining route: Kernel density in color/spatial space then nearest higher-density parent tree.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-random-walker"></a>
- [ ] **`segmentation.random_walker`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(data, labels, beta=130, mode='cg_j', tol=0.001, copy=True, return_full_prob=False, spacing=None, *, prob_tol=0.001, channel_axis=None)`.
  - Required focus: inactive labels, beta/spacing, cg modes/preconditioners, prob_tol and full_prob. Cases to close: disconnected seeds, high contrast and convergence warnings.
  - Easiest remaining route: Build weighted grid Laplacian, partition seeds/unlabeled nodes and solve probabilities.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-relabel-sequential"></a>
- [ ] **`segmentation.relabel_sequential`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(label_field, offset=1)`.
  - Required focus: offset, dtype growth, negative rejection and mapping provider state. Cases to close: huge sparse labels, zero and offset boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native sorted positive-ID compression and forward/inverse ArrayMap providers.
  - Evidence locators: parity [tests/test_component_consumers_public.py](../tests/test_component_consumers_public.py); benchmark [benchmarks/component_consumers.py](../benchmarks/component_consumers.py).

<a id="api-segmentation-slic"></a>
- [ ] **`segmentation.slic`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, n_segments=100, compactness=10.0, max_num_iter=10, sigma=0, spacing=None, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=1, mask=None, *, channel_axis=-1)`.
  - Required focus: mask seeding, spacing, slic_zero, start_label and cluster tie order. Cases to close: masked/anisotropic 2D/3D and low-contrast images.
  - Easiest remaining route: Lab/spatial grid seeds, local k-means assignment/update and connectivity cleanup.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-watershed"></a>
- [ ] **`segmentation.watershed`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False)`.
  - Required focus: entry-time ties, connectivity/offset, mask and marker generation. Cases to close: plateaus, touching fronts and compactness variants.
  - Easiest remaining route: Stable priority flood from markers with optional compactness/line rules.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-transform"></a>
### transform — 38 public keys

<details><summary>Expand transform contract checklist</summary>

<a id="api-transform-affinetransform"></a>
- [ ] **`transform.AffineTransform`** — full contract closed.
  - Main: `native-class-partial`; gate status `C? N~ H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(matrix=None, *, scale=None, shear=None, rotation=None, translation=None, dimensionality=None)`.
  - Required focus: scale/shear/rotation conventions, dimension, inverse state, weights and failure descriptors. Cases to close: 2D/3D affine fits, singular matrices and class lifecycle.
  - Easiest remaining route: Audit owned partial class/native coordinate path; extend affine parameter decomposition and normalized least-squares fit.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered in the shipping inventory.

<a id="api-transform-essentialmatrixtransform"></a>
- [ ] **`transform.EssentialMatrixTransform`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(*, rotation=None, translation=None, matrix=None, dimensionality=None)`.
  - Required focus: rotation/translation validation, rank failure, epipolar residual and inverse. Cases to close: known camera poses and degenerate correspondences.
  - Easiest remaining route: Reuse normalized eight-point solver then enforce essential singular values; own epipolar class state.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-euclideantransform"></a>
- [ ] **`transform.EuclideanTransform`** — full contract closed.
  - Main: `native-class-partial`; gate status `C? N~ H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(matrix=None, *, rotation=None, translation=None, dimensionality=None)`.
  - Required focus: rotation representations, reflection handling, dimension and failure state. Cases to close: 2D/3D rigid fits and reflected/degenerate point sets.
  - Easiest remaining route: Audit owned partial class; rigid Procrustes/SVD fit with determinant correction and native coordinate calls.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered in the shipping inventory.

<a id="api-transform-fundamentalmatrixtransform"></a>
- [ ] **`transform.FundamentalMatrixTransform`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(matrix=None, *, dimensionality=None)`.
  - Required focus: minimal points, normalization degeneracy, matrix scale and failure state. Cases to close: exact/noisy epipolar pairs and rank-deficient data.
  - Easiest remaining route: Normalized eight-point SVD fit with rank-two projection and Sampson residuals.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-piecewiseaffinetransform"></a>
- [ ] **`transform.PiecewiseAffineTransform`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: Delaunay backend, hull-outside sentinel, simplex ties and inverse triangulation. Cases to close: boundary/outside points and degenerate triangulations.
  - Easiest remaining route: Triangulate source/destination points then fit per-simplex affine maps and point location.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-polynomialtransform"></a>
- [ ] **`transform.PolynomialTransform`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(params=None, *, dimensionality=None)`.
  - Required focus: monomial order, weights, rank deficiency and inverse support/error. Cases to close: order sweep and ill-conditioned coordinates.
  - Easiest remaining route: Build polynomial design matrix and weighted normalized least-squares coefficients.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-projectivetransform"></a>
- [ ] **`transform.ProjectiveTransform`** — full contract closed.
  - Main: `native-class-partial`; gate status `C? N~ H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(matrix=None, *, dimensionality=None)`.
  - Required focus: homogeneous zero denominator, dimension, composition order and failure object. Cases to close: singular/weighted fits and large batched point calls.
  - Easiest remaining route: Audit owned partial class/native homography evaluation; extend normalized DLT/SVD fit and state.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered in the shipping inventory.

<a id="api-transform-similaritytransform"></a>
- [ ] **`transform.SimilarityTransform`** — full contract closed.
  - Main: `native-class-partial`; gate status `C? N~ H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(matrix=None, *, scale=None, rotation=None, translation=None, dimensionality=None)`.
  - Required focus: reflection/rank handling, dimensionality and parameter decomposition. Cases to close: scaled/reflected/degenerate point sets.
  - Easiest remaining route: Audit owned partial class; centered Umeyama SVD scale/rotation/translation fit.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered in the shipping inventory.

<a id="api-transform-thinplatesplinetransform"></a>
- [ ] **`transform.ThinPlateSplineTransform`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: duplicate/collinear controls, singular solve, zero radius and inverse availability. Cases to close: minimal/degenerate controls and dense warp queries.
  - Easiest remaining route: Build radial-basis plus affine saddle-point system, solve weights and evaluate batched kernel.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-downscale-local-mean"></a>
- [ ] **`transform.downscale_local_mean`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, factors, cval=0, clip=True)`.
  - Required focus: factors, cval, clip, dtype/range and output shape. Cases to close: nondivisible shapes and uint/float/strided blocks.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native padded block mean reduction.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-transform-estimate-transform"></a>
- [ ] **`transform.estimate_transform`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(ttype, src, dst, *args, **kwargs)`.
  - Required focus: recognized aliases/type errors, kwargs, weights and returned class. Cases to close: every registered transform type and failed fits.
  - Easiest remaining route: Route type string to owned transform class.from_estimate preserving failure contract.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-frt2"></a>
- [ ] **`transform.frt2`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, complex64, complex128`.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Candidate B~ admitted: eight finite-Radon cases/eight wins with actual authority/cohort/helper closure in the declared integer minimum-host domain. [Root performance receipt](evidence/current281-radon-flood-performance-root-admission-v1.json) and [published raw cases](../benchmarks/current281-minimum-initial-v1/README.md). Broader domains and main/full-contract B/R gates remain open.
  - Signature: `(a)`.
  - Required focus: supported size, modular line indexing and unsigned overflow. Cases to close: tiny exact prime grids and integer limits.
  - Easiest remaining route: Implement finite Radon integer line sums on prime-sized square grid.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Source/storage admission](evidence/finite-radon-source-storage-admission-v1.json) at d2109de2a0b7f07e27f0a892585cd7414e36bfbf: 54 recipes for this API, bounded factory/protocol scope only. [Controller admission](evidence/finite-radon-controller-admission-v1.json) and [saved-raw admission](evidence/finite-radon-saved-raw-admission-v1.json) now cover 108 original observations per profile (216 total consumed once, 61 returns and 47 errors per profile), preserving four runtime error-class differences. [Observed native admission](evidence/finite-radon-observed-native-admission-v2.json) at ee639c0c passes 1,296 strict saved comparisons (108 cases/profile, workers 1/4, normal/disabled/scalar), with target control/status/MXCSR and exact caller32 preserved. Scoped assembled SDK has 244 successful gold comparisons per mode; MSVC is compile-only. Wider exceptional casts/cancellation/half/object/temporal/protocol/platform domains, genuine hot swap/whole installed SDK and public one-thread performance remain open.
  - Candidate extension: [1855377b observed exceptional-cast component](evidence/finite-radon-exceptional-native-admission-v1.json) adds four NaN/Inf recipes per profile across both Radon APIs; all 1,344 old/new worker/mode comparisons pass. Other exceptional widths, payloads and positions remain open; no genuine/public performance credit.
  - Current scoped progress: Candidate `8c4fb430` includes exceptional-cast component `1855377b` and corrected star exports; saved-domain evidence above. Full domains, genuine SDK/hot swap and public performance pending.

<a id="api-transform-hough-circle"></a>
- [ ] **`transform.hough_circle`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool, uint8`.
  - Shipped on `5b94b20d` within its recorded native domain. Windows installed fixed speed passed for the batch. [Whole-product admission](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/whole-product-root-admission-v1.json) [Windows one-worker evidence](../build/evidence/u182-post-haar/next-five-hosted-speed-v4/actual-result-v1/root-timing-admission-v1.json)
  - Signature: `(image, radius, normalize=True, full_output=False)`.
  - Required focus: normalize, full_output padding, radius cast and duplicate perimeter votes. Cases to close: exact single circles and dense edge masks.
  - Private `43d925b5` on Linux modern matched 21 original calls and passed 22 checks per mode. Four fixed private one-worker comparisons were 2.427x to 3.021x the original. Installed ownership, SDK and release remain open. [Private evidence](../build/evidence/u182-hough-circle/retained-modern-private-root-admission-v1.json) [Private speed](../build/evidence/u182-post-haar/retained-modern-thin-circle-timing-root-admission-v1.json)
  - Shipped scope: Bounded untagged C-order 2D Boolean or uint8 image with extents through 512 and ordered integer radii 0–64. C++ owns Bresenham votes and normalization; unsupported radii, tags and layouts use compatibility.
  - Evidence locators: parity [tests/test_hough_circle_public.py](../tests/test_hough_circle_public.py); benchmark [benchmarks/hough_circle.py](../benchmarks/hough_circle.py).

<a id="api-transform-hough-circle-peaks"></a>
- [ ] **`transform.hough_circle_peaks`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `float64`.
  - Shipped on `1d9a0bbf` within its bounded 3D peak-selection domain. Four-profile qualification [whole-product admission](../build/evidence/u182-post-haar/current312-qualification-v1/whole-product-root-admission-v1.json) ([run 45344078252](https://github.com/bipxx/scikit-imagexx/actions/runs/45344078252)) and Windows 1-worker timing `12.488×–157.942×` ([run 45340574164](../build/evidence/u182-post-haar/current312-windows-focus-v1/root-focus-timing-admission-v1.json)); modern fixed speed at `8.688×–114.437×` ([Fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-v1/circle-peaks-modern-timing-v1/successor-v2/actual-result-v1/root-circle-peaks-speed-admission-v1.json)); `archive_retained: false`.
  - Signature: `(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=inf, total_num_peaks=inf, normalize=False)`.
  - Required focus: normalize scoring, per-radius/total limits, x/y suppression ties. Cases to close: competing radii and equal peaks.
  - Easiest remaining route: Peak-suppress each circle accumulator then stable cross-radius selection.
  - Evidence locators: [tests/test_hough_circle_peaks_public.py](../tests/test_hough_circle_peaks_public.py) · [benchmarks/hough_circle_peaks.py](../benchmarks/hough_circle_peaks.py).

<a id="api-transform-hough-ellipse"></a>
- [ ] **`transform.hough_ellipse`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool, uint8`.
  - Shipped on `1d9a0bbf` within its bounded 2D edge-mask domain. Four-profile qualification [whole-product admission](../build/evidence/u182-post-haar/current312-qualification-v1/whole-product-root-admission-v1.json) ([run 45344078252](https://github.com/bipxx/scikit-imagexx/actions/runs/45344078252)), Windows 1-worker timing `23.279×–25.946×` ([run 45340574164](../build/evidence/u182-post-haar/current312-windows-focus-v1/root-focus-timing-admission-v1.json)), and modern fixed speed `19.691×–22.657×` ([Fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-modern-speed-timing-carrier-v1/actual-result-v1/root-fixed-speed-admission-v1.json)); `archive_retained: false`.
  - Signature: `(image, threshold=4, accuracy=1, min_size=4, max_size=None)`.
  - Required focus: accuracy bins, threshold, min/max size, output field order and ties. Cases to close: synthetic ellipses and dense edge clutter.
  - Easiest remaining route: Port triplet edge-point ellipse hypothesis/voting algorithm.
  - Evidence locators: [tests/test_hough_ellipse_public.py](../tests/test_hough_ellipse_public.py) · [benchmarks/hough_ellipse.py](../benchmarks/hough_ellipse.py).
  - Fourteen modern stock outcomes and three histogram provider probes matched both retained older profiles. [Modern stock receipt](../build/evidence/u182-post-haar/lookahead-hough-ellipse-modern-v1/handoff-v1.json)

<a id="api-transform-hough-line"></a>
- [ ] **`transform.hough_line`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Private Linux modern passed 23 named checks in each dispatch mode, with scoped direct GNU SDK evidence. The earlier AMD run matched outputs but measured 0.99859× on dense-custom-91. The wide-index successor passed all four changed modern comparisons at 1.217×–1.717× on Intel. Revised Windows and minimum coverage remains pending. [Changed fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-line-wide-speed-v1/actual-result-v1/root-changed-speed-admission-v1.json) [Fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-modern-speed-timing-carrier-v1/actual-result-v1/root-fixed-speed-admission-v1.json) Installed coverage and release remain pending. [Private admission](../build/evidence/u182-post-haar/lookahead-hough-hosted-carrier-v4/actual-result-v1/root-modern-hough-admission-v1.json) [Timing run](https://github.com/bipxx/scikit-imagexx/actions/runs/45101684310)
  - Signature: `(image, theta=None)`.
  - Required focus: theta order/range, rho rounding, accumulator shape and overflow. Cases to close: horizontal/diagonal lines at bin boundaries.
  - Easiest remaining route: Vectorized rho computation per theta with exact integer accumulator votes.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - All 23 frozen original Hough line outcomes matched on retained modern CPython 3.13.8, Windows and Linux minimum. Modern native parity and its unchanged four fixed comparisons remain open. [Scoped result](../build/evidence/u182-post-haar/lookahead-hough-line-modern-v1/handoff-v1.json)

<a id="api-transform-hough-line-peaks"></a>
- [ ] **`transform.hough_line_peaks`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `uint64`.
  - Shipped on `5b94b20d` within its recorded native domain. Windows installed fixed speed passed for the batch. [Whole-product admission](../build/evidence/u182-post-haar/next-five-release-reader-v1/hosted-current306-45098117810/whole-product-root-admission-v1.json) [Windows one-worker evidence](../build/evidence/u182-post-haar/next-five-hosted-speed-v4/actual-result-v1/root-timing-admission-v1.json)
  - Signature: `(hspace, angles, dists, min_distance=9, min_angle=10, threshold=None, num_peaks=inf)`.
  - Required focus: min_distance/min_angle, threshold default, tie order and peak count. Cases to close: equal/symmetric peaks near angle boundaries.
  - Shipped scope: Bounded untagged C-order uint64 accumulator with fewer than 262144 cells and matching finite float64 angles and distances. C++ owns peak selection; NumPy provides capped ordering and coordinate gathers. Tagged carriers and unsupported requests use compatibility.
  - Evidence locators: parity [tests/test_hough_line_peaks_public.py](../tests/test_hough_line_peaks_public.py); benchmark [benchmarks/hough_line_peaks.py](../benchmarks/hough_line_peaks.py).
  - All 25 retained Linux modern stock cases matched Windows byte for byte. Linux minimum differs on two capped-tie rankings. The retained modern private product passed 52 checks and four fixed timings at 11.41×–41.02×. Genuine installed and release gates remain open. [Modern speed](../build/evidence/u182-post-haar/retained-modern-match-peaks-zhang-timing-root-admission-v1.json) · [Modern stock](../build/evidence/u182-hough-line-peaks/modern-stock-v1/handoff-v1.json)

<a id="api-transform-ifrt2"></a>
- [ ] **`transform.ifrt2`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, uint8, uint16, uint32, uint64`.
  - Scoped main281 release `aaea1b06`: [exact-source genuine four-profile/304SDK/three-stager admission](evidence/vi-public-matmul-whole-product-root-admission-v1.json) passes its retained domain. [Shipping source projection](evidence/current281-main-source-projection-v1.json) passes the retained source/domain slice; full contracts remain open.
  - Candidate B~ admitted: eight finite-Radon cases/eight wins with actual authority/cohort/helper closure in the declared integer minimum-host domain. [Root performance receipt](evidence/current281-radon-flood-performance-root-admission-v1.json) and [published raw cases](../benchmarks/current281-minimum-initial-v1/README.md). Broader domains and main/full-contract B/R gates remain open.
  - Signature: `(a)`.
  - Required focus: prime-size assumptions, division and signed/unsigned dtype. Cases to close: exact forward/inverse integer fixtures.
  - Easiest remaining route: Invert finite Radon modular line sums with exact normalization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Source/storage admission](evidence/finite-radon-source-storage-admission-v1.json) at d2109de2a0b7f07e27f0a892585cd7414e36bfbf: 54 recipes for this API, bounded factory/protocol scope only. [Controller admission](evidence/finite-radon-controller-admission-v1.json) and [saved-raw admission](evidence/finite-radon-saved-raw-admission-v1.json) now cover 108 original observations per profile (216 total consumed once, 61 returns and 47 errors per profile), preserving four runtime error-class differences. [Observed native admission](evidence/finite-radon-observed-native-admission-v2.json) at ee639c0c passes 1,296 strict saved comparisons (108 cases/profile, workers 1/4, normal/disabled/scalar), with target control/status/MXCSR and exact caller32 preserved. Scoped assembled SDK has 244 successful gold comparisons per mode; MSVC is compile-only. Wider exceptional casts/cancellation/half/object/temporal/protocol/platform domains, genuine hot swap/whole installed SDK and public one-thread performance remain open.
  - Candidate extension: [1855377b observed exceptional-cast component](evidence/finite-radon-exceptional-native-admission-v1.json) retains owned inverse-prefix typed errors and strict FP/policy evidence for the four-recipe/profile extension shared with frt2. Wider exceptional domains, genuine installation and performance remain open.
  - Current scoped progress: Candidate `8c4fb430` includes exceptional-cast component `1855377b` and corrected star exports; saved-domain evidence above. Full domains, genuine SDK/hot swap and public performance pending.

<a id="api-transform-integral-image"></a>
- [ ] **`transform.integral_image`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, *, dtype=None)`.
  - Required focus: dtype choice, axis pass order, overflow and physical layout. Cases to close: cancellation and integer wrap on 2D/3D layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native ordered N-D prefix scans with modular integer arithmetic.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-transform-integrate"></a>
- [ ] **`transform.integrate`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ii, start, end)`.
  - Required focus: start/end normalization, negative indices, batched windows and corner order. Cases to close: tiny/large query batches and reverse layouts.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native integral-corner inclusion-exclusion queries.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-transform-iradon"></a>
- [ ] **`transform.iradon`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(radon_image, theta=None, output_size=None, filter_name='ramp', interpolation='linear', circle=True, preserve_range=True)`.
  - Required focus: filter_name, circle masking, interpolation, output_size and angular normalization. Cases to close: analytic phantom and no-filter reconstructions.
  - Easiest remaining route: FFT-filter projections then interpolate/backproject and scale.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-iradon-sart"></a>
- [ ] **`transform.iradon_sart`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(radon_image, theta=None, image=None, projection_shifts=None, clip=None, relaxation=0.15, dtype=None)`.
  - Required focus: golden-angle order, projection_shifts, image alias, clip and dtype. Cases to close: one/multiple sweeps and shifted rays.
  - Easiest remaining route: Sequential projection updates with ray interpolation and relaxation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-matrix-transform"></a>
- [ ] **`transform.matrix_transform`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(coords, matrix)`.
  - Required focus: dimensions, denominator zeros, dtype promotion and authenticated numerical profile. Cases to close: frozen matrix profiles including ill-conditioned coordinates.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse existing private native homogeneous coordinate candidate and public wrapper after whole-product acceptance.
  - Evidence locators: parity [tests/test_matrix_transform_public.py](../tests/test_matrix_transform_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-transform-order-angles-golden-ratio"></a>
- [ ] **`transform.order_angles_golden_ratio`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64`.
  - Signature: `(theta)`.
  - Required focus: ties, duplicates, angle periodicity and generator semantics. Cases to close: repeated/unsorted angle lists.
  - Shipped scope: lazy serial order for at most 16 supported finite angles in `[-360, 360]` under round-to-nearest, with live ndarray reads on resume and preserved deferred carrier errors. Larger or unsupported initial inputs take the configured compatibility route on first resume. Live native values must remain in scope after iteration begins.
  - Evidence locators: [Public parity](../tests/test_order_angles_golden_ratio_integration.py) · [One-worker benchmark](../benchmarks/order_angles_golden_ratio_native.py).
  - Private `0cd7e277` passed 29 scoped source checks on each retained Python version. Successor `c33c8341` added first-resume routing and passed 420 public checks across two retained Linux profiles and both dispatch modes, along with the scoped SDK client. The combined local draft passed installed ownership and relocated SDK checks in both modes. [Earlier private native admission](evidence/order-angles-golden-ratio-native-root-admission-v1.json) · [Successor handoff](../build/golden-angle-current-main-v1/compatibility-v1/handoff-v1.json) · [Installed speed admission](../build/evidence/golden-pearson-integration/local-installed-speed-root-admission-v2.json)
  - Main `9a30600e` passed both complete source checks, 72 portable hosted Golden-angle outcomes and 630 Linux saved outcomes across two dispatch modes, the linked SDK checks and all three packaging stagers. The four measured one-worker cases passed at 1.359×–3.705× on the retained WSL minimum install. [Release admission](../build/evidence/golden-pearson-integration/hosted-44887385792/whole-product-root-admission-v1.json)
  - Current scoped progress: [Root source/storage admission](evidence/order-angles-golden-ratio-root-source-v1.json) covers 45 prospective recipes/N<=16, 128-byte actual backing (256-byte cap), 120 gather elements/17 resumes, and 98 pure checks per Python. Original lazy bodies, per-resume FP/index types and live mutations are not observed yet. [Actual factory/zero-body readiness handoff](../build/order-angles-golden-ratio-readiness-v1/handoff-v1.json), source `fb17c0cd`, is frozen. Capture v2 received [CHANGES_REQUIRED](../build/order-angles-golden-ratio-capture-independent-review-v2/review-v2.json) for post-consumption evidence loss and journal completion ordering. [Author-frozen v3 successor](../build/order-angles-golden-ratio-capture-v3/handoff-v3.json), source `ff9b0949`, reports 35 pure/inert controls per Python; [Independent and root combined readiness admission](evidence/order-angles-golden-ratio-capture-root-v3.json) passes. [Minimum](evidence/order-angles-golden-ratio-minimum-root-raw-v1.json) and [modern/paired](evidence/order-angles-golden-ratio-modern-root-raw-v1.json) raw admissions cover45cases/profile,187resumes and147typed indices each, with474semantic FPdifferences preserved. Both reference budgets are exhausted. The isolated source port above reuses these observations. Native execution and timing are still pending.

<a id="api-transform-probabilistic-hough-line"></a>
- [ ] **`transform.probabilistic_hough_line`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `bool, uint8`.
  - Shipped on `1d9a0bbf` within its bounded 2D edge-mask domain (compact `[angle, rho]` `int16` accumulator, batched vote removal, SSE2 2-wide projection). Four-profile qualification [whole-product admission](../build/evidence/u182-post-haar/current312-qualification-v1/whole-product-root-admission-v1.json) ([run 45344078252](https://github.com/bipxx/scikit-imagexx/actions/runs/45344078252)) and Windows 1-worker timing `1.709×–8.171×` ([run 45340574164](../build/evidence/u182-post-haar/current312-windows-focus-v1/root-focus-timing-admission-v1.json)); prior `e87df809` `0.933×` loss on `dense-default-bands` (`45336457806`) retained as superseded history; modern fixed speed measured `1.046×–1.778×` ([Fixed speed admission](../build/evidence/u182-post-haar/lookahead-hough-modern-speed-timing-carrier-v1/actual-result-v1/root-fixed-speed-admission-v1.json)); `archive_retained: false`.
  - Signature: `(image, threshold=10, line_length=50, line_gap=10, theta=None, rng=None)`.
  - Required focus: RNG draw order, line_gap/length, threshold and endpoint order. Cases to close: crossing lines, gaps and fixed saved RNG state.
  - Easiest remaining route: Preserve seeded edge visitation, line votes, extension and pixel removal.
  - Evidence locators: [tests/test_probabilistic_hough_line_public.py](../tests/test_probabilistic_hough_line_public.py) · [benchmarks/probabilistic_hough_line.py](../benchmarks/probabilistic_hough_line.py).
  - All 18 retained modern stock cases, including fixed endpoints and intersection order, match Windows and minimum. Five caller PCG64 states also matched. [Modern stock receipt](../build/evidence/u182-post-haar/lookahead-probabilistic-hough-modern-v1/handoff-v1.json)

<a id="api-transform-pyramid-expand"></a>
- [ ] **`transform.pyramid_expand`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S? V~ W? B~ R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, upscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: channel_axis, rounding, mode and preserve_range. Cases to close: odd/even sizes and per-axis channels.
  - Easiest remaining route: Reuse resize-up and Gaussian smoothing with exact scale-derived sigma.
  - Evidence locators: [tests/test_pyramid_spatial_public.py](../tests/test_pyramid_spatial_public.py) · [benchmarks/pyramid_spatial.py](../benchmarks/pyramid_spatial.py).
  - Private `738df1a9` composes resize then Gaussian for bounded finite rank-2 and channel-axis rank-3 inputs. Both fixed workloads meet one-worker parity on Windows and WSL minimum. Unproven dtype/shape/mode domains use compatibility. [Private speed](../build/evidence/u182-pyramid-spatial/timing-v1/timing-root-admission-v1.json)
  - Candidate `7f8d4c5a` passed its selected installed parity and one-worker comparisons. Finite spatial expansion on the selected Windows and Linux minimum inputs, including native typed resize and Gaussian. The corrected source entered four-profile qualification. Minimum failed the Gaussian stage for expand-rgb-u8-reflect in both modes. [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45058138452) · [Parity](../build/evidence/u182-post-haar/installed-focused-root-admission-v1.json) · [Speed](../build/evidence/u182-post-haar/installed-six-timing-root-admission-v1.json)
  - On a fresh genuine Linux minimum runner, the exact failed candidate and the original matched the frozen output and all Gaussian weights. The original suite failure remains open. [Diagnostic](../build/evidence/u182-post-haar/pyramid-hosted-minimum-causal-v1/causal-result-v1/root-diagnostic-admission-v1.json)
  - Bounded spatial composition, hot-swap checks, both dispatch modes and scoped one-worker speed shipped. Unsupported dtype, shape or option domains and additional worker/SDK scope remain open. [Verified main delivery](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/root-main-delivery-v1.json) · [Whole-product evidence](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/whole-product-root-admission-v1.json)

<a id="api-transform-pyramid-gaussian"></a>
- [ ] **`transform.pyramid_gaussian`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `uint8, float32, float64`.
  - Shipped on `a80be034`. The generator reuses the installed C++ primitives and exhausts every timed iterator. The documented wider cases remain open. [Qualification](../build/evidence/u182-post-haar/hosted-current301-45090228543/whole-product-root-admission-v1.json) [Fixed benchmark](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)
  - Signature: `(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: yield count/order, original first level, channel_axis and lazy exceptions. Cases to close: tiny/odd shapes and finite/infinite max_layer.
  - Shipped scope: Bounded owning C-order image. Existing C++ Gaussian and resize stages with lazy stock-order yields. Benchmarks exhaust the iterator. Unsupported requests use compatibility.
  - Evidence locators: parity [tests/test_pyramid_generators_public.py](../tests/test_pyramid_generators_public.py); benchmark [benchmarks/pyramid_generators.py](../benchmarks/pyramid_generators.py).
  - Windows and WSL minimum now have exact generator stock outputs, lazy transitions, first-yield alias and mutation controls. Preserve each NaN warning behavior. Native and installed checks are pending. [Stock reference](../build/evidence/u182-pyramid-generators/stock-root-admission-v1.json)
  - Private `3a7dff2d` passed exact lazy-state and compiled-stage tests. Fixed full-consumption timing met one-worker parity on all three retained profiles. Linux modern passed 82 tests and measured 1.41×–1.68× across the four cases. The prepared attachment adds installed alias checks and passed four focused private lazy fallback checks. The complete installed selector is still pending after the spatial pyramid parent ships. [Attachment checks](../build/evidence/u182-pyramid-generators/installed-family-attachment-v1/readback.json) [Modern parity](../build/evidence/u182-pyramid-generators/modern-component-v2/root-admission-v1.json) · [Modern timing](../build/evidence/u182-pyramid-generators/modern-timing-v1/root-timing-admission-v1.json) [Timing](../build/evidence/u182-pyramid-generators/timing-v1/timing-root-admission-v1.json) · [Private admission](../build/evidence/u182-pyramid-generators/private-component-root-admission-v1.json)
  - Candidate eddec69f passed installed Windows parity in both modes. The fixed one-worker comparison was 1.351×–1.529× over two calls with full iterator consumption. Four-profile release is pending. [Installed timing](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)

<a id="api-transform-pyramid-laplacian"></a>
- [ ] **`transform.pyramid_laplacian`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `uint8, float32, float64`.
  - Shipped on `a80be034`. The generator reuses the installed C++ primitives and exhausts every timed iterator. The documented wider cases remain open. [Qualification](../build/evidence/u182-post-haar/hosted-current301-45090228543/whole-product-root-admission-v1.json) [Fixed benchmark](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)
  - Signature: `(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: first/last level, residual dtype, sigma and lazy state. Cases to close: reconstruction-oriented pyramids and odd shapes.
  - Shipped scope: Bounded owning C-order image. Existing C++ Gaussian and resize stages with lazy stock-order yields. Benchmarks exhaust the iterator. Unsupported requests use compatibility.
  - Evidence locators: parity [tests/test_pyramid_generators_public.py](../tests/test_pyramid_generators_public.py); benchmark [benchmarks/pyramid_generators.py](../benchmarks/pyramid_generators.py).
  - Windows and WSL minimum now have exact ordered residual, smoothed continuation and repeated-size termination observations. Preserve each NaN warning behavior. Native and installed checks are pending. [Stock reference](../build/evidence/u182-pyramid-generators/stock-root-admission-v1.json)
  - Private `3a7dff2d` passed exact smoothed-state, residual, and compiled-stage tests. Fixed full-consumption timing met one-worker parity on all three retained profiles. Linux modern passed 82 tests and measured 1.41×–1.68× across the four cases. The prepared attachment adds installed alias checks and passed four focused private lazy fallback checks. The complete installed selector is still pending after the spatial pyramid parent ships. [Attachment checks](../build/evidence/u182-pyramid-generators/installed-family-attachment-v1/readback.json) [Modern parity](../build/evidence/u182-pyramid-generators/modern-component-v2/root-admission-v1.json) · [Modern timing](../build/evidence/u182-pyramid-generators/modern-timing-v1/root-timing-admission-v1.json) [Timing](../build/evidence/u182-pyramid-generators/timing-v1/timing-root-admission-v1.json) · [Private admission](../build/evidence/u182-pyramid-generators/private-component-root-admission-v1.json)
  - Candidate eddec69f passed installed Windows parity in both modes. The fixed one-worker comparison was 1.363×–1.442× over two calls with full iterator consumption. Four-profile release is pending. [Installed timing](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)

<a id="api-transform-pyramid-reduce"></a>
- [ ] **`transform.pyramid_reduce`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S? V~ W? B~ R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: downscale validation, sigma default, shape rounding and channels. Cases to close: noninteger downscale and anisotropic inputs.
  - Easiest remaining route: Gaussian smooth then native resize-down at source-defined output shape.
  - Evidence locators: [tests/test_pyramid_spatial_public.py](../tests/test_pyramid_spatial_public.py) · [benchmarks/pyramid_spatial.py](../benchmarks/pyramid_spatial.py).
  - Private `738df1a9` composes Gaussian then resize for bounded finite rank-2 and channel-axis rank-3 inputs. Both fixed workloads meet one-worker parity on Windows and WSL minimum. Unproven dtype/shape/mode domains use compatibility. [Private speed](../build/evidence/u182-pyramid-spatial/timing-v1/timing-root-admission-v1.json)
  - Candidate `7f8d4c5a` passed its selected installed parity and one-worker comparisons. Finite spatial reduction on the selected Windows and Linux minimum inputs, including native Gaussian and typed resize. The corrected product has identical installed bytes and is in four-profile qualification. [Qualification](https://github.com/bipxx/scikit-imagexx/actions/runs/45058138452) · [Parity](../build/evidence/u182-post-haar/installed-focused-root-admission-v1.json) · [Speed](../build/evidence/u182-post-haar/installed-six-timing-root-admission-v1.json)
  - Bounded spatial composition, hot-swap checks, both dispatch modes and scoped one-worker speed shipped. Unsupported dtype, shape or option domains and additional worker/SDK scope remain open. [Verified main delivery](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/root-main-delivery-v1.json) · [Whole-product evidence](../build/evidence/u182-post-haar/hosted-current296-corrected-45070502340/whole-product-root-admission-v1.json)

<a id="api-transform-radon"></a>
- [ ] **`transform.radon`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, theta=None, circle=True, *, preserve_range=False)`.
  - Required focus: circle crop/pad/warning, center convention, preserve_range and angle order. Cases to close: delta/phantom projections at cardinal/arbitrary angles.
  - Easiest remaining route: Rotate/sample image for each angle then sum projection rays.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-rescale"></a>
- [ ] **`transform.rescale`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, scale, order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None, *, channel_axis=None)`.
  - Required focus: channel_axis, scalar/vector scale, anti_aliasing and rounding. Cases to close: small/large/noninteger scale and channel axes.
  - Easiest remaining route: Audit/extend existing native implementation: Audit native scale-to-output-shape wrapper then resize core.
  - Evidence locators: parity [tests/test_resize_contract.py](../tests/test_resize_contract.py); benchmark [benchmarks/resize.py](../benchmarks/resize.py).

<a id="api-transform-resize"></a>
- [ ] **`transform.resize`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, output_shape, order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)`.
  - Required focus: order/bool restrictions, mode, range, output_shape, overlap and numerical profiles. Cases to close: frozen cases plus uncovered modes/orders/shapes.
  - Easiest remaining route: Audit/extend existing native implementation: Extend existing native spline/prefilter/antialias/clip implementation.
  - Evidence locators: parity [tests/test_resize_contract.py](../tests/test_resize_contract.py); benchmark [benchmarks/resize.py](../benchmarks/resize.py).

<a id="api-transform-resize-local-mean"></a>
- [ ] **`transform.resize_local_mean`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, output_shape, grid_mode=True, preserve_range=False, *, channel_axis=None)`.
  - Required focus: grid_mode pixel centers/extents, preserve_range and channel_axis. Cases to close: arbitrary rational scale ratios and integer-factor equivalence.
  - Easiest remaining route: Exact area-overlap weights per axis followed by separable averaging.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-rotate"></a>
- [ ] **`transform.rotate`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, angle, resize=False, center=None, order=None, mode='constant', cval=0, clip=True, preserve_range=False)`.
  - Required focus: default center, resize corner rounding, angle special cases and clipping. Cases to close: right angles, arbitrary angles and odd/even centers.
  - Easiest remaining route: Build centered inverse affine rotation and optional expanded bounds then reuse warp.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-swirl"></a>
- [ ] **`transform.swirl`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, center=None, strength=1, radius=100, rotation=0, output_shape=None, order=None, mode='reflect', cval=0, clip=True, preserve_range=False)`.
  - Required focus: center, radius/strength, output_shape and coordinate order. Cases to close: identity/strong swirl and boundary center.
  - Easiest remaining route: Generate radial exponential swirl inverse coordinates then reuse warp.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-warp"></a>
- [ ] **`transform.warp`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, inverse_map, map_args=None, output_shape=None, order=None, mode='constant', cval=0.0, clip=True, preserve_range=False)`.
  - Required focus: inverse_map forms, map_args callbacks, channel handling, order/prefilter and range clip. Cases to close: matrix/callable/coordinate-array maps with modes and layouts.
  - Easiest remaining route: Dispatch matrix/owned transform or callable map; reuse N-D coordinate interpolation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-warp-coords"></a>
- [ ] **`transform.warp_coords`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(coord_map, shape, dtype=<class 'numpy.float64'>)`.
  - Required focus: shape 2D/3D meaning, dtype, callback order and returned shape. Cases to close: identity/nonlinear callback traces.
  - Easiest remaining route: Generate ordered output grid and call coord_map with upstream reshape convention.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-warp-polar"></a>
- [ ] **`transform.warp_polar`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, center=None, *, radius=None, output_shape=None, scaling='linear', channel_axis=None, **kwargs)`.
  - Required focus: center/radius defaults, angular endpoint, log origin and channel_axis. Cases to close: radial/angular patterns and all scaling modes.
  - Easiest remaining route: Generate linear/log radial polar inverse map then reuse warp.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

<a id="module-util"></a>
### util — 25 public keys

<details><summary>Expand util contract checklist</summary>

<a id="api-util-failedestimationaccesserror"></a>
- [ ] **`util.FailedEstimationAccessError`** — full contract closed.
  - Main: `exception-metadata-provider`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: args/name/obj, notes, pickle, traceback and string/state behavior. Cases to close: construct/raise/catch/pickle exception lifecycle.
  - Easiest remaining route: Preserve owned AttributeError-compatible exception provider and inherited metadata.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered in the shipping inventory.

<a id="api-util-pendingskimage2change"></a>
- [ ] **`util.PendingSkimage2Change`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `None`.
  - Required focus: filtering category, args/notes, pickle and traceback. Cases to close: warning filter and round-trip lifecycle.
  - Easiest remaining route: Own warning subclass preserving PendingDeprecationWarning inheritance and metadata.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-apply-parallel"></a>
- [ ] **`util.apply_parallel`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(function, array, chunks=None, depth=0, mode=None, extra_arguments=(), extra_keywords=None, *, dtype=None, compute=None, channel_axis=None)`.
  - Required focus: compute lazy/eager, chunks/depth/mode, channel_axis, callbacks and dtype inference. Cases to close: instrumented boundary-aware function on irregular chunks.
  - Easiest remaining route: Preserve Dask/lazy-array overlap adapter or own bounded chunk executor for eager arrays.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-compare-images"></a>
- [ ] **`util.compare_images`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image0, image1, *, method='diff', n_tiles=(8, 8))`.
  - Required focus: shape/conversion/method and tile protocol order; mixed dtype; K-order layout and dtype metadata; NumPy1.24 float32 zero-dimensional blend stays a literal provider. Cases to close: exact small checkerboards, negative/zero-step and remainder tiles, scalar promotion, mixed dtype and C/F/strided/metadata storage.
  - Easiest remaining route: Audit/extend existing native implementation: Own typed subtraction/absolute value, add-then-half blend and direct checker coordinate selection; preserve conversion and metadata allocation.
  - Evidence locators: parity [tests/test_compare_images_public.py](../tests/test_compare_images_public.py); benchmark [benchmarks/compare_images.py](../benchmarks/compare_images.py).

<a id="api-util-crop"></a>
- [ ] **`util.crop`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, complex64, complex128`.
  - Candidate B~ admitted: crop stream one-thread public comparison, actual cohort/C1/helper and qualifiedDDA runtime/SDK chain accepted. [Exact cases/all losses](../benchmarks/current281-minimum-initial-v1/README.md), [root receipt](evidence/current281-join-rolling-performance-root-admission-v1.json). Full domains, provider/class ownership, optimization and main delivery remain open.
  - Signature: `(ar, crop_width, copy=False, order='K')`.
  - Required focus: scalar/tuple widths, negative/large crops, order and memory sharing. Cases to close: asymmetric N-D crops and C/F/negative strides.
  - Easiest remaining route: Return slice view or requested-order copy after crop-width normalization.
  - Evidence locators: parity [tests/test_util_metadata_integration.py](../tests/test_util_metadata_integration.py); benchmark [benchmarks/crop.py](../benchmarks/crop.py).
  - Current scoped progress: Optimization `f48fe19e`, composed in `8c4fb430`; private component and dependent LBP checks retained. Genuine candidate qualification and fresh public performance acceptance pending; shipped losses retained.
  - New private optimization f020: [source admission](evidence/crop-metadata-optimization-root-source-admission-v1.json), [924 saved/protocol regression checks](evidence/crop-metadata-private-parity-root-admission-v1.json), [32 controlled prior-C++/new-C++ FP/storage calls](evidence/crop-metadata-controlled-prior-cpp-root-admission-v1.json). Only immutable slice-header reuse changed; pixel leaf/SDK untouched. No original-library speed or shipping credit. Original-public hardware FP, remaining portability/modes, genuine/current-context and performance gates stay open.
  - View-only successor `9e689ac2`: [1,024 actual saved/protocol checks](evidence/crop-view-only-private-parity-root-closeout-v1.json) pass across two retained profiles and normal/disabled SIMD. This is private parity only; controlled FP/storage, original-public speed, genuine installation and release remain open.
  - View-only measurement: [eight changed-source comparisons](../benchmarks/crop-view-only-private-v1/README.md) yield six faster medians and two slower (3D view/object copy). Independent raw review passes; source remains unaccepted because of the two slower medians. No original-library speed credit or unchanged retiming. [32 controlled FP/storage calls](evidence/crop-view-only-controlled-fp-root-closeout-v1.json) pass privately.
  - Private performance disposition: [eight prior-C++ comparisons/raw files](../benchmarks/crop-metadata-private-v1/README.md), seven faster medians and one slower (F-copy0.964x). F020 remains unaccepted; a view-only source variant is in progress. No original baseline or candidate64-case result is replaced.

  - Delivered scope: exact a6/main ef465027, [four-profile/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) and [one-thread public cases with every loss](evidence/join-util-views-initial-performance-admission-v1.json). Full domain closure remains unchecked.
  - Optimization candidate: [f48 scoped root admission](evidence/util-view-boundary-root-admission-v1.json) covers strided POD copy correctness; borrowed views unchanged. Dependent LBP/shared genuine/performance gates remain open; earlier eight benchmark losses are preserved.

<a id="api-util-dtype-limits"></a>
- [ ] **`util.dtype_limits`** — full contract closed.
  - Main: `metadata`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, clip_negative=False)`.
  - Required focus: dtype recognition, signed range and unsupported dtype errors. Cases to close: every supported dtype and clip flag.
  - Easiest remaining route: Reuse owned dtype metadata lookup with clip_negative.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-bool"></a>
- [ ] **`util.img_as_bool`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: signed input, exact half ties, warning, copy and nonfinite values. Cases to close: dtype extrema and threshold-adjacent values.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native dtype-specific half-range threshold conversion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-float"></a>
- [ ] **`util.img_as_float`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: float16/float32 preservation, signed scaling and force_copy. Cases to close: all scalar dtypes and alias/copy cases.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native integer-to-float scaling while retaining floating precision behavior.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-float32"></a>
- [ ] **`util.img_as_float32`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: integer endpoints, precision loss and force_copy. Cases to close: wide integers and float64 rounding boundaries.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native normalized cast to float32.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-float64"></a>
- [ ] **`util.img_as_float64`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: signed minimum scaling and force_copy. Cases to close: integer extrema and strided arrays.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native normalized cast to float64.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-int"></a>
- [ ] **`util.img_as_int`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: input range error, clipping, rounding and warning. Cases to close: range endpoints and float half steps.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native normalized signed-int16 conversion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-ubyte"></a>
- [ ] **`util.img_as_ubyte`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: signed negative clipping, precision warning, rounding and copy. Cases to close: negative signed input and 255-scale ties.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native normalized uint8 conversion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-img-as-uint"></a>
- [ ] **`util.img_as_uint`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, force_copy=False)`.
  - Required focus: input bit-depth scaling, signed negatives and rounding. Cases to close: uint8/uint64 and float threshold cases.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native normalized uint16 conversion.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-invert"></a>
- [ ] **`util.invert`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, signed_float=False)`.
  - Required focus: bool, signed integer, unsigned, signed_float and NaNs. Cases to close: dtype extrema and both floating conventions.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native dtype-dependent range complement.
  - Evidence locators: parity [tests/test_api.py::test_every_native_function](../tests/test_api.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-util-label-points"></a>
- [ ] **`util.label_points`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(coords, output_shape)`.
  - Required focus: duplicate pixel overwrite, bounds, N-D output_shape and rounding. Cases to close: half-integer and coincident points.
  - Easiest remaining route: Audit/extend existing native implementation: Round coordinates with exact tie rule and scatter sequential point IDs.
  - Evidence locators: parity [tests/test_label_points_public.py::test_label_points_native_public_contract](../tests/test_label_points_public.py); benchmark [benchmarks/label_points.py](../benchmarks/label_points.py).

<a id="api-util-lookfor"></a>
- [ ] **`util.lookfor`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(what)`.
  - Required focus: search normalization, printed output and optional documentation source. Cases to close: fixed query hits and no-match output.
  - Easiest remaining route: Preserve documentation-search adapter over pinned public namespace.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-map-array"></a>
- [ ] **`util.map_array`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8`.
  - Signature: `(input_arr, input_vals, output_vals, out=None)`.
  - Required focus: unsorted/duplicate input keys, missing keys, out shape/alias and large integers. Cases to close: sparse/wide keys and overlap cases.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse native dense/sparse label lookup and output casting.
  - Evidence locators: parity [tests/test_map_array_public.py::test_map_array_native_public_contract](../tests/test_map_array_public.py); benchmark [benchmarks/map_array.py](../benchmarks/map_array.py).

<a id="api-util-montage"></a>
- [ ] **`util.montage`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(arr_in, fill='mean', rescale_intensity=False, grid_shape=None, padding_width=0, *, channel_axis=None)`.
  - Required focus: fill=mean, channel_axis, grid_shape, padding and dtype. Cases to close: incomplete grids and multichannel tiles.
  - Easiest remaining route: Audit/extend existing native implementation: Compute tile grid/padding, optional native rescale and fill before block copies.
  - Evidence locators: parity [tests/test_montage_integration.py::test_montage_registered_public_frozen_contract](../tests/test_montage_integration.py); benchmark [benchmarks/montage.py](../benchmarks/montage.py).

<a id="api-util-random-noise"></a>
- [ ] **`util.random_noise`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W? B~ R~`; default dtype slice: `uint8, uint16, float64`.
  - Shipped on `a80be034` for the documented seeded routes. Shared installed public tests and direct C++ SDK passed. Wider modes or choices retain compatibility behavior. [Qualification](../build/evidence/u182-post-haar/hosted-current301-45090228543/whole-product-root-admission-v1.json) [Fixed benchmark](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)
  - Signature: `(image, mode='gaussian', rng=None, clip=True, **kwargs)`.
  - Required focus: gaussian/localvar/poisson/salt/pepper/s&p/speckle, signed range and RNG state. Cases to close: saved-state mode matrix and clipping/range edges.
  - Shipped scope: Bounded seeded Gaussian, speckle, salt-and-pepper and pepper routes. NumPy supplies conversion, RNG, range and clipping; C++ combines or replaces pixels. Other modes use compatibility.
  - Evidence locators: parity [tests/test_random_noise_public.py](../tests/test_random_noise_public.py); benchmark [benchmarks/random_noise.py](../benchmarks/random_noise.py).
  - Genuine Windows and WSL minimum references cover seven modes and 38 calls per profile. Preserve exact profile differences for signed-zero clipping and speckle. The bounded Gaussian, speckle, salt-and-pepper and pepper candidate `e6f5bd69` passed scoped native and SDK checks. Remaining domains use compatibility. Fixed speed, installed and release gates remain open. [Private admission](../build/evidence/u182-random-noise/private-component-root-admission-v1.json) [Stock admission](../build/evidence/u182-random-noise/stock-root-admission-v1.json)
  - Retained Linux modern CPython 3.13.8 passed 76 exact native or pre-draw refusal checks in normal and forced-scalar modes. The four fixed private modern timings met parity at 1.09×–1.56×. Hosted CPython 3.13.16 installed and relocated SDK gates remain open. [Modern speed](../build/evidence/u182-post-haar/random-modern-timing-v1/root-modern-fixed-timing-admission-v1.json) · [Linux modern component](../build/evidence/u182-random-noise/modern-component-v1/root-modern-component-admission-v1.json)
  - All eight fixed private one-worker cases met parity at 1.107×–1.907×. Shared installed and release gates remain open. [Exact speed](../build/evidence/u182-random-noise/timing-v1/timing-root-admission-v1.json)
  - The private installed-scope tests verified real native dispatch and enabled stock fallback in both dispatch modes. Shared installed aliases, genuine hosted SDK and benchmark checks remain pending. [Fallback and RNG proof](../build/evidence/u182-post-haar/random-fallback-tests-v1/installed-family-test-handoff-v1.json)
  - Candidate eddec69f passed installed Windows parity in both modes. The fixed one-worker comparison was 1.240×–2.397× over four seeded calls. Four-profile release is pending. [Installed timing](../build/evidence/u182-post-haar/following-windows-installed-timing-v2/root-timing-admission-v2.json)

<a id="api-util-regular-grid"></a>
- [ ] **`util.regular_grid`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64`.
  - Signature: `(ar_shape, n_points)`.
  - Required focus: n_points extremes, dimensionality and slice start/stop convention. Cases to close: anisotropic/tiny shapes and target counts.
  - Easiest remaining route: Choose rounded per-axis strides and slices for approximately uniform sample count.
  - Evidence locators: parity [tests/test_util_metadata_integration.py](../tests/test_util_metadata_integration.py); benchmark [benchmarks/regular_grid.py](../benchmarks/regular_grid.py).

  - Delivered scope: exact a6/main ef465027, [four-profile/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) and [one-thread public cases with every loss](evidence/join-util-views-initial-performance-admission-v1.json). Full domain closure remains unchecked.

<a id="api-util-regular-seeds"></a>
- [ ] **`util.regular_seeds`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ar_shape, n_points, dtype=<class 'int'>)`.
  - Required focus: dtype, label numbering and requested-count approximation. Cases to close: anisotropic grids and limited integer dtypes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse regular_grid slices then assign ordered seed labels.
  - Evidence locators: parity [tests/test_regular_seeds_public.py::test_regular_seeds_native_public_contract](../tests/test_regular_seeds_public.py); benchmark [benchmarks/regular_seeds.py](../benchmarks/regular_seeds.py).

<a id="api-util-slice-along-axes"></a>
- [ ] **`util.slice_along_axes`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, complex64, complex128`.
  - Candidate B~ admitted: slice stream one-thread public comparison, actual cohort/C1/helper and qualifiedDDA runtime/SDK chain accepted. [Exact cases/all losses](../benchmarks/current281-minimum-initial-v1/README.md), [root receipt](evidence/current281-join-rolling-performance-root-admission-v1.json). Full domains, provider/class ownership, optimization and main delivery remain open.
  - Signature: `(image, slices, axes=None, copy=False)`.
  - Required focus: duplicate/negative axes, bounds and sharing. Cases to close: reordered/negative axes and copy flags.
  - Easiest remaining route: Normalize axes/slice pairs then return view or copy.
  - Evidence locators: parity [tests/test_util_views_integration.py](../tests/test_util_views_integration.py); benchmark [benchmarks/slice_along_axes.py](../benchmarks/slice_along_axes.py).
  - Current scoped progress: Optimization `f48fe19e`, composed in `8c4fb430`; private component and dependent LBP checks retained. Genuine candidate qualification and fresh public performance acceptance pending; shipped losses retained.

  - Delivered scope: exact a6/main ef465027, [four-profile/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) and [one-thread public cases with every loss](evidence/join-util-views-initial-performance-admission-v1.json). Full domain closure remains unchecked.
  - Optimization candidate: [f48 scoped root admission](evidence/util-view-boundary-root-admission-v1.json) covers the shared copier used by copied slices; borrowed slices unchanged. Genuine integration and new performance remain open; earlier eight losses are preserved.

<a id="api-util-unique-rows"></a>
- [ ] **`util.unique_rows`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ar)`.
  - Required focus: dtype, NaNs, row order and noncontiguous input. Cases to close: duplicate/noncontiguous rows with special values.
  - Easiest remaining route: Audit/extend existing native implementation: Stable/contract-ordered row deduplication using contiguous byte keys or typed comparisons.
  - Evidence locators: parity [tests/test_unique_rows_public.py::test_unique_rows_native_public_contract](../tests/test_unique_rows_public.py); benchmark [benchmarks/unique_rows.py](../benchmarks/unique_rows.py).

<a id="api-util-view-as-blocks"></a>
- [ ] **`util.view_as_blocks`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, complex64, complex128`.
  - Candidate B~ admitted: blocks stream one-thread public comparison, actual cohort/C1/helper and qualifiedDDA runtime/SDK chain accepted. [Exact cases/all losses](../benchmarks/current281-minimum-initial-v1/README.md), [root receipt](evidence/current281-join-rolling-performance-root-admission-v1.json). Full domains, provider/class ownership, optimization and main delivery remain open.
  - Signature: `(arr_in, block_shape)`.
  - Required focus: divisibility, dimensionality, contiguous assumptions and write sharing. Cases to close: C/F/noncontiguous arrays and block mutation.
  - Easiest remaining route: Construct validated reshape/strided block view.
  - Evidence locators: parity [tests/test_util_views_integration.py](../tests/test_util_views_integration.py); benchmark [benchmarks/array_views.py](../benchmarks/array_views.py).
  - Current scoped progress: Optimization `f48fe19e`, composed in `8c4fb430`; private component and dependent LBP checks retained. Genuine candidate qualification and fresh public performance acceptance pending; shipped losses retained.

  - Delivered scope: exact a6/main ef465027, [four-profile/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) and [one-thread public cases with every loss](evidence/join-util-views-initial-performance-admission-v1.json). Full domain closure remains unchecked.
  - Optimization candidate: [f48 scoped root admission](evidence/util-view-boundary-root-admission-v1.json) covers cached actual-platform integer metadata with existing validation/protocol behavior. Genuine integration and new performance remain open; earlier four losses are preserved.

<a id="api-util-view-as-windows"></a>
- [ ] **`util.view_as_windows`** — full contract closed.
  - Main: `native-partial`; gate status `C~ N~ H~ P~ S~ V~ W~ B~ R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, complex64, complex128`.
  - Candidate B~ admitted: windows stream one-thread public comparison, actual cohort/C1/helper and qualifiedDDA runtime/SDK chain accepted. [Exact cases/all losses](../benchmarks/current281-minimum-initial-v1/README.md), [root receipt](evidence/current281-join-rolling-performance-root-admission-v1.json). Full domains, provider/class ownership, optimization and main delivery remain open.
  - Signature: `(arr_in, window_shape, step=1)`.
  - Required focus: scalar/vector step, shape, strides and dangerous overlap semantics. Cases to close: overlapping windows and negative/zero invalid steps.
  - Easiest remaining route: Construct rolling strided view with validated window/step.
  - Evidence locators: parity [tests/test_util_views_integration.py](../tests/test_util_views_integration.py); benchmark [benchmarks/array_views.py](../benchmarks/array_views.py).
  - Current scoped progress: Optimization `f48fe19e`, composed in `8c4fb430`; private component and dependent LBP checks retained. Genuine candidate qualification and fresh public performance acceptance pending; shipped losses retained.

  - Delivered scope: exact a6/main ef465027, [four-profile/installed SDK admission](evidence/join-util-views-whole-product-admission-v1.json) and [one-thread public cases with every loss](evidence/join-util-views-initial-performance-admission-v1.json). Full domain closure remains unchecked.

</details>
  - Optimization candidate: [f48 scoped root admission](evidence/util-view-boundary-root-admission-v1.json) covers cached actual-platform integer metadata with existing validation/protocol behavior. Genuine integration and new performance remain open; earlier four losses are preserved.

## Class/member checklist — 29 classes, 284 static paths

Top-level class completion also requires every applicable member and runtime-discovered state/protocol below. Native member labels remain partial. Each member must satisfy the same C/N/H/P/S/V/W/B/R gates; constructor/getter metadata cannot substitute for solver or descriptor computation.

### `feature.BRIEF` — 2 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.BRIEF.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, descriptor_size=256, patch_size=49, mode='normal', sigma=1, rng=1)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.BRIEF.extract`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image, keypoints)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.CENSURE` — 2 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.CENSURE.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.CENSURE.detect`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.Cascade` — 11 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.Cascade.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.__new__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.__reduce__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.__setstate__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.detect_multi_scale`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, img, scale_factor, step_ratio, min_size, max_size, min_neighbor_number=4, intersection_score_threshold=0.5)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.eps`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.features_number`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.stages_number`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.stumps_number`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.window_height`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.window_width`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.ORB` — 4 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.ORB.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, downscale=1.2, n_scales=8, n_keypoints=500, fast_n=9, fast_threshold=0.08, harris_k=0.04)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.ORB.detect`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.ORB.detect_and_extract`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.ORB.extract`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image, keypoints, scales, orientations)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.SIFT` — 5 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.SIFT.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5, c_dog=0.013333333333333334, c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.deltas`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.detect`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.detect_and_extract`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.extract`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `filters.LPIFilter2D` — 2 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`filters.LPIFilter2D.__call__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data)`.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark not registered.

- [ ] **`filters.LPIFilter2D.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, impulse_response, **filter_params)`.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark not registered.

### `future.TrainableSegmenter` — 4 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`future.TrainableSegmenter.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, clf=None, features_func=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`future.TrainableSegmenter.compute_features`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`future.TrainableSegmenter.fit`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image, labels)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`future.TrainableSegmenter.predict`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__new__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__reduce__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__setstate__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.find_costs`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.goal_reached`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.offsets`** — attribute, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.traceback`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP_Connect` — 11 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP_Connect.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /, *args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__new__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__reduce__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__setstate__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.create_connection`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, id1, id2, tb1, tb2, cost1, cost2)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.find_costs`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.goal_reached`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.offsets`** — attribute, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.traceback`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP_Flexible` — 13 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP_Flexible.__init__`** — constructor, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__new__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__reduce__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__setstate__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.examine_neighbor`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, index, new_index, offset_length)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.find_costs`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.goal_reached`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.offsets`** — attribute, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.traceback`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.travel_cost`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, old_cost, new_cost, offset_length)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.update_node`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, index, new_index, offset_length)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP_Geometric` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP_Geometric.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__new__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__reduce__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__setstate__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.find_costs`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.goal_reached`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.offsets`** — attribute, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.traceback`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.RAG` — 46 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.RAG.__contains__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__getitem__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, label_image=None, connectivity=1, data=None, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__iter__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__len__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__new__`** — staticmethod, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(cls, *args, backend=None, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__str__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_edge`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, u, v, attr_dict=None, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_edges_from`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, ebunch_to_add, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_node`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n, attr_dict=None, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_nodes_from`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, nodes_for_adding, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_weighted_edges_from`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, ebunch_to_add, weight='weight', **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.adj`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.adjacency`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.clear`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.clear_edges`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.copy`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.degree`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.edge_subgraph`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, edges)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.edges`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.fresh_copy`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.get_edge_data`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, u, v, default=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.has_edge`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, u, v)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.has_node`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.is_directed`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.is_multigraph`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.merge_nodes`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst, weight_func=<function min_weight>, in_place=True, extra_arguments=None, extra_keywords=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.name`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.nbunch_iter`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, nbunch=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.neighbors`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.next_id`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.nodes`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.number_of_edges`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, u=None, v=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.number_of_nodes`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.order`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_edge`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, u, v)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_edges_from`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, ebunch)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_node`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_nodes_from`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, nodes)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.size`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, weight=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.subgraph`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, nodes)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_directed`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, as_view=False)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_directed_class`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_undirected`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, as_view=False)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_undirected_class`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.update`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, edges=None, nodes=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `io.ImageCollection` — 9 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`io.ImageCollection.__getitem__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, load_pattern, conserve_memory=True, load_func=None, **load_func_kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__iter__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__len__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__str__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.concatenate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.conserve_memory`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.files`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.reload`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `io.MultiImage` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`io.MultiImage.__getitem__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, filename, conserve_memory=True, dtype=None, **imread_kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__iter__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__len__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__str__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.concatenate`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.conserve_memory`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.filename`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.files`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.reload`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, n=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.CircleModel` — 6 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.CircleModel.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, center, radius)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.params`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.predict_xy`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, t, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.residuals`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.EllipseModel` — 6 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.EllipseModel.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, center, axis_lengths, theta)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.params`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.predict_xy`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, t, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.residuals`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.LineModelND` — 8 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.LineModelND.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, origin, direction)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.params`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.predict`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, x, axis=0, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.predict_x`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, y, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.predict_y`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, x, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.residuals`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, data, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.RansacModelProtocol` — 6 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.RansacModelProtocol.__class_getitem__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, *args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.__init_subclass__`** — classmethod, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.__subclasshook__`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(other)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.residuals`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, *data)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.AffineTransform` — 17 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.AffineTransform.__add__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__array__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__call__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, matrix=None, *, scale=None, shear=None, rotation=None, translation=None, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__nice__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__repr__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__str__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.dimensionality`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.identity`** — classmethod, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.inverse`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.rotation`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.scale`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.shear`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.translation`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.EssentialMatrixTransform` — 8 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.EssentialMatrixTransform.__call__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, *, rotation=None, translation=None, matrix=None, dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.dimensionality`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.identity`** — classmethod, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.inverse`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.residuals`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.EuclideanTransform` — 15 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.EuclideanTransform.__add__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__array__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__call__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, matrix=None, *, rotation=None, translation=None, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__nice__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__repr__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__str__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.dimensionality`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst) -> Union[Self, skimage._shared.utils.FailedEstimation]`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.identity`** — classmethod, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.inverse`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.rotation`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.translation`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.FundamentalMatrixTransform` — 8 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.FundamentalMatrixTransform.__call__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.__init__`** — constructor, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, matrix=None, *, dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.dimensionality`** — property, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.identity`** — classmethod, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.inverse`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.residuals`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.PiecewiseAffineTransform` — 7 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.PiecewiseAffineTransform.__call__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.identity`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.inverse`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.residuals`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.PolynomialTransform` — 7 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.PolynomialTransform.__call__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, params=None, *, dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst, order=2, weights=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst, order=2, weights=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.identity`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.inverse`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.residuals`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.ProjectiveTransform` — 13 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.ProjectiveTransform.__add__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__array__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__call__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__init__`** — constructor, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, matrix=None, *, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__nice__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__repr__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__str__`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.dimensionality`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.identity`** — classmethod, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.inverse`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.SimilarityTransform` — 16 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.SimilarityTransform.__add__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__array__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__call__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, matrix=None, *, scale=None, rotation=None, translation=None, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__nice__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__repr__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__str__`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.dimensionality`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dst) -> Union[Self, skimage._shared.utils.FailedEstimation]`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.identity`** — classmethod, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.inverse`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.rotation`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.scale`** — property, defined; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.translation`** — property, inherited; current `native-class-member-partial`.
  - Gate status: `C? N~ H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.ThinPlateSplineTransform` — 5 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.ThinPlateSplineTransform.__call__`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.estimate`** — method, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(src, dst) -> Union[Self, skimage._shared.utils.FailedEstimation]`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.inverse`** — property, defined; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

### `util.FailedEstimationAccessError` — 13 static members

Main: `exception-metadata-provider`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`util.FailedEstimationAccessError.__getattribute__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, name, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__getstate__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__init__`** — constructor, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /, *args, **kwargs)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__new__`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__reduce__`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__repr__`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__setstate__`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__str__`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.add_note`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.args`** — attribute, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.name`** — attribute, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.obj`** — attribute, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.with_traceback`** — method, inherited; current `exception-metadata-member`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

### `util.PendingSkimage2Change` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`util.PendingSkimage2Change.__getattribute__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, name, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__init__`** — constructor, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /, *args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__new__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__reduce__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__repr__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__setstate__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__str__`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.add_note`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.args`** — attribute, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.with_traceback`** — method, inherited; current `upstream-only`.
  - Gate status: `C? N? H? P? S? V? W? B? R?`; full member evidence remains unreconciled.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

## Runtime and whole-library closure

- [ ] Inventory runtime RegionProperties objects and every exposed attribute, including the still-unassessed property partitions; verify computation, cache invalidation and scalar/layout/state behavior.
  - `solidity` remains provider-backed. A private native prototype bypassed an error filter scoped to the upstream warning module. A subsequent shortcut would have changed observable cache lookups and relied on a mutable key. Both were rejected before timing. [Warning evidence](../build/regionprops-solidity-packet-v1/handoff-v1.json) · [Cache evidence](../build/regionprops-solidity-packet-v1/handoff-v2.json)
  - Raw `moments` remains provider-backed. The available native projector requires extra layout copies, and warm cache hits already skip computation. Exact bool-mask arithmetic and floating-point behavior on cache misses remain unverified. [Source and layout evidence](../build/regionprops-moments-preflight-v1/handoff-v1.json)
- [ ] Close returned ArrayMap identity/protocols, shared keys/values, lazy caches, indexing/mutation and native ownership; keep exact upstream-class differences visible.
- [ ] Inventory generators/iterators, callback-created objects, fitted instance fields and dynamically exposed members absent from the static 284 paths.
- [ ] Review nested/plugin namespaces, callable and noncallable public exports, aliases and identity relationships omitted by the pinned 477-key snapshot.
- [ ] Close required native backend/provider choices for FFT, sparse and small solvers, triangulation/hulls, wavelets, special functions, classifiers and resampling, with licenses/build/ABI/global-thread behavior.
- [ ] Account for codecs/containers, sample assets, licensing and cache/catalogue behavior using an explicit applicability/provider contract.
- [ ] Run meaningful hot-swapped end-to-end pipelines and genuinely installed C++ consumers across the complete supported product.
- [ ] Resolve or explicitly retain all original-public and controlled prior-C++ performance losses; no averages or wins count may conceal a required failed target.
- [ ] Verify every applicable scalar/SIMD/global-worker mode and supported platform/profile at the final exact source.
- [ ] Independently audit every applicable checklist gate against current source/evidence, deliver main, and clean only eligible obsolete outputs/worktrees. No completion claim while a required API/domain/member/backend/performance/profile gate remains open.

## Closeout update rule

1. Update the existing task’s source/receipt, terminal result, reusable evidence and next owner/trigger in `work_items.json` and `EXECUTION_PLAN.md`.
2. Update only affected API/member rows here: main classification, specific domain gap, each newly proved gate and its exact evidence/source/profile. Keep private, candidate and delivered scopes distinct.
3. Check affected links, inventory/member counts and Markdown/JSON/state consistency. Do not run builds, numerical tests, benchmarks, downloads or cleanup solely to update the checklist.
4. Report progress as delivered API/domain closure and named pending or failed gates. Do not turn partial marks, assertion totals, candidate registrations or benchmark wins into a full-completion percentage.
5. Include the checklist delta in the same validated increment. If coverage did not change, record “reviewed, no coverage change” in the existing closeout; do not add a separate audit task or bookkeeping-only scientific campaign.

Documentation validation (2026-11-07): all 477 API keys and 284 static-member keys match the inventory without duplicates; all 761 rows have nine gates and local checklist links resolve. Ledger JSON parses. Reviewed, no coverage change; existing receipts reused without scientific reruns. Root owns updates at the next bounded closeout.
