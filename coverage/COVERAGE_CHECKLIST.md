# Coverage checklist

This is the detailed completion view for the existing [execution plan](EXECUTION_PLAN.md) and [operational ledger](work_items.json). Those records still govern assignments, execution and evidence. This file adds no dispatch queue or permission to run work.

Updated: 2026-11-06. Shipping classifications come from [api.json](api.json); per-key contract focus comes from [the implementation paths](API_IMPLEMENTATION_PATHS.json). The paths are proposals, not proof of completed coverage.

## How to read and update this checklist

- An API checkbox closes only when every applicable gate below is proved for its full declared contract. An unchecked box means full closure has not been established here; it does not discard existing scoped work.
- Gate codes: `C` contract/domain inventory; `N` native/provider ownership; `H` hot swap; `P` public parity; `S` installed C++ SDK; `V` scalar/SIMD; `W` global workers; `B` one-thread original benchmark; `R` release/delivery.
- `?` means full-gate evidence has not been reconciled. `~` means a bounded/partial slice is credited. Replace a code with `done` only alongside a receipt, exact source, supported profiles and domain. Use `N/A` only with a recorded applicability reason.
- Test and benchmark links below are **locators**, not passing receipts. A registered default dtype does not prove all layouts, values, errors, provider branches or platforms.
- After each bounded task, update its existing ledger/handoff first, then update the affected API/member rows, active-increment table and evidence links here. Preserve failures and slower cases. Do not rerun science just to refresh this document or regenerate away reviewed marks.
- Root verifies the changed rows before merge/push/closeout. Workers identify affected keys and exact passed/pending/failed scopes in their existing handoff; root owns the cross-family checklist update.

## Shipping snapshot

| Inventory category | Public keys | Meaning |
|---|---:|---|
| `native-partial` | 265 | Native image paths with incomplete or unassessed full-contract partitions. |
| `native-class-partial` | 5 | Classes with only part of computation/state/member ownership closed. |
| `metadata` | 3 | Metadata operations; applicability and full public contract still need explicit disposition. |
| `exception-metadata-provider` | 1 | Exception/class metadata provider; not a native image algorithm. |
| `upstream-only` | 203 | Shipping implementation still uses the upstream path. |

Total: **477 public keys**, including **29 classes and 284 statically inventoried member paths**. Static members are additional contract work, not 284 extra top-level APIs. Current member classifications: 63 native-class-member-partial, 11 exception-metadata members and 210 upstream-only; none of those labels establishes full member closure. Runtime-returned objects and discovered members remain open below.

The shipped 265 partial image paths have [four-profile admission](evidence/next-metrics-whole-product-admission-v1.json). This is bounded release credit, not full parity. The latest published 28-case cohort has [18 wins and 10 losses](../benchmarks/next-metrics-minimum-initial-v1/README.md); it does not establish library-wide speed superiority.

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

## Current cumulative increment: six new public APIs

The candidate is **271 partial image paths** at source `a6bae9ac86dbba362537758784edf06c973e9c53` (216 product files, 102 SDK files); production code is unchanged from b557. It has not been delivered to main.

- [x] Compatible private native components and retained original-reference evidence reviewed for the six paths below.
- [x] Shared registry, single-core bindings, SDK hooks, installed-source selectors and explicit test-helper preparation/collection wired.
- [x] [Independent source review](evidence/join-util-views-final-source-admission-v1.json) and [five-test metadata repair](evidence/join-util-views-sourcebarrier-fix-admission-v1.json).
- [x] [Byte-preserving checkout policy](evidence/join-util-views-windows-bytepolicy-admission-v1.json) and [complete faithful source readiness](evidence/join-util-views-source-readiness-v3.json): 3,035 cases plus 8 subtests on each retained Python with core.autocrlf=true; no failures/errors/skips/scientific imports. Production blobs remain unchanged from the predecessor.
- [ ] Four-profile genuine product/installed SDK admission. [Run 42238215124](https://github.com/bipxx/scikit-imagexx/actions/runs/42238215124) failed on exact b557: all four source barriers passed, three Linux jobs succeeded, and Windows had 100 saved-JOIN setup failures because UCRT does not export feraiseexcept. The [retained audit](evidence/join-util-views-failed-campaign-audit-v2.json) preserves this failed campaign. [Independent test-fix admission](evidence/join-windows-fenv-fixture-admission-v1.json) passes for a6bae9ac86dbba362537758784edf06c973e9c53; [Both complete source barriers](evidence/join-util-views-source-readiness-v4.json) pass: 3,056 tests plus 8 subtests per Python, no failures/errors/skips/scientific imports. [Successor run 42262940227](https://github.com/bipxx/scikit-imagexx/actions/runs/42262940227) is live on exact a6bae9, watched through retained session 82134. Modern Linux job completed successfully; the other three jobs and independent artifact/release admission remain pending. No mixed-source or whole-release acceptance.
- [ ] Actual measurement controller/helper/runtime/core/SDK/root binding and each API’s real zero-body sampler-entry cohort accepted.
- [ ] Separate original-public measurement GO, one-thread comparison, raw-result admission and every loss disposition.
- [ ] Final source projection, main merge/push and safe eligible worktree/build cleanup.

| API | Existing private/reference scope | Remaining release work |
|---|---|---|
| `segmentation.join_segmentations` | Integer-label composition and complete private operation; returned ArrayMap class/protocol frontiers remain explicit. | Installed whole-core/genuine SDK, class/domain closure, one-thread public evidence, delivery. |
| `util.crop` | View geometry and real POD copying; object/zero-item/subclass provider scopes disclosed. | Genuine complete SDK/public operation, platform/domain frontiers, one-thread public evidence, delivery. |
| `util.regular_grid` | Source-faithful grid preparation/finish and complete private SDK; full 49-node saved Linux selector wired. | Actual genuine runs; Windows/default-integer/libm domains; one-thread public evidence and delivery. |
| `util.slice_along_axes` | Saved source/error/view/copy observations and native geometry/POD transport; FP envelope admitted. | Genuine installed core/SDK and saved-domain selectors; providers/platforms, public benchmark, delivery. |
| `util.view_as_blocks` | Integral borrowed geometry/strides; NumPy representation/provider boundaries explicit. | Genuine installed core/SDK, full domains/platforms, own API cohort and public benchmark, delivery. |
| `util.view_as_windows` | Integral borrowed sliding-window geometry/strides; provider and metadata overflow frontiers explicit. | Genuine installed core/SDK, full domains/platforms, separate API cohort and public benchmark, delivery. |

Failed predecessor 42238215124 observation (2026-11-06; retained below). Successor 42262940227 has four live jobs; their qualification results are pending:

| Profile | Hosted job result | Acceptance status |
|---|---|---|
| Windows / Python 3.13.16 | Source barrier passed; genuine qualification failed (100 JOIN setup failures). | Release gate failed; frozen test-only correction awaits independent review and complete barriers. |
| Modern Linux / Python 3.13.16 | Job completed successfully. | Retained scoped evidence; does not close four-profile release. |
| Minimum Linux / Python 3.11.17 | Job completed successfully. | Retained scoped evidence; does not close four-profile release. |
| Debug Linux / Python 3.13.16 | Job completed successfully. | Retained scoped evidence; does not close four-profile release. |

The same increment contains reviewed density and F-layout boundary optimizations. Their private C++ ratios do not close original-public speed gates. The rejected [complete source attempt](evidence/join-util-views-complete-sourcebarrier-failure-v1.json) remains failed.

## Next source/reference work

- [ ] `segmentation.mark_boundaries`: [Saved original admission](evidence/mark-boundaries-saved-raw-admission-v2.json) covers 125 cases per profile (250 complete observations, 251 consumed calls including the retained failed attempt). One separately authorized fixed-codec witness closed the structured-error gap; no successful original rows were replayed. [Independent native successor admission](evidence/mark-boundaries-fp-successor-admission-v1.json) passes on source 6d4373b5720b5ee959c1d313e14e3d8f754085e4 for 1,500 saved worker/mode comparisons (125 cases per profile, workers 1/4, normal/disabled/scalar), including strict tiny-color target FP. Small RHS palette assignment remains explicit NumPy provider work. Genuine installation, platform/domain closure, one-thread original-public performance and delivery remain open. [Executable benchmark source review](evidence/mark-boundaries-benchmark-source-admission-v1.json) and [handoff reconciliation](evidence/mark-boundaries-benchmark-handoff-reconciliation-v1.json) pass on 446bde47e7462278389c0a87affb881ed775f1ec: 52 source/zero-body controls per Python, fixed 8 cases, no actual targets or timings. Genuine binding, outer first-cohort and separate timing GO remain pending.
- [ ] `feature.draw_multiblock_lbp`: [Saved original admission](evidence/draw-multiblock-lbp-saved-raw-admission-v1.json) covers 94 cases per profile. [Independent private-component review](evidence/draw-multiblock-lbp-native-independent-review-v2.json) covers source 0e51dbc1e2c4741e243a91652d8c0b9771c96363: 1,128 target comparisons across two profiles, three scalar/SIMD modes and workers 1/4 pass. Of these, 1,048 complete records match and 80 differ only in post-return JSON collector MXCSR diagnostics, retained explicitly. Target FP and exact caller restoration remain strict. [Root scoped component admission](evidence/draw-multiblock-lbp-native-root-admission-v2.json) passes. [Private integration source/ABI review](evidence/draw-multiblock-lbp-private-integration-admission-v1.json) and [immutable handoff reconciliation](evidence/draw-multiblock-lbp-private-handoff-reconciliation-v1.json) pass in their scoped checks; the qualifier rejects the temporary conditional install rule, so genuine qualification is not passed. Normal enabled MARK/LBP integration is now being authored in a separate successor. Shared genuine core/installed SDK, full domains/platforms, one-thread original-public performance and delivery remain open.
- [ ] `transform.frt2` and `transform.ifrt2`: [Source/storage review](evidence/finite-radon-source-storage-admission-v1.json) passes on d2109de2 for 108 recipes, 54 per API. Integer-only native/SDK invariant checks are not original parity; no original calls yet. Controller/readiness and full dtype/protocol/platform/performance gates remain open.
- [ ] `draw.polygon_perimeter` and `draw.rectangle_perimeter`: reuse retained clipping assessment; pinned Matplotlib/AGG reference/native dependency strategy is still required. Do not repeat the already-recorded dependency discovery.
- [ ] Performance controller C1 successor: [post-stop source/authority checks and one post-batch verifier](evidence/join-util-views-controller-admission-v2.json) pass scoped independent review; bind only after actual genuine product admission. No measurement authority is inferred from pure tests.

## Module index

Counts are shipping inventory classifications, not full-contract completion percentages.

| Module | Public keys | Partial native images | Upstream-only | Other categories |
|---|---:|---:|---:|---:|
| [color](#module-color) | 41 | 40 | 0 | 1 |
| [data](#module-data) | 41 | 0 | 41 | 0 |
| [draw](#module-draw) | 18 | 14 | 4 | 0 |
| [exposure](#module-exposure) | 10 | 10 | 0 | 0 |
| [feature](#module-feature) | 41 | 10 | 31 | 0 |
| [filters](#module-filters) | 100 | 98 | 1 | 1 |
| [filters.rank](#module-filters.rank) | 0 | 0 | 0 | 0 |
| [filters.rank.bilateral](#module-filters.rank.bilateral) | 0 | 0 | 0 | 0 |
| [filters.rank.generic](#module-filters.rank.generic) | 0 | 0 | 0 | 0 |
| [future](#module-future) | 5 | 0 | 5 | 0 |
| [graph](#module-graph) | 15 | 0 | 15 | 0 |
| [io](#module-io) | 11 | 0 | 11 | 0 |
| [measure](#module-measure) | 35 | 26 | 9 | 0 |
| [metrics](#module-metrics) | 10 | 6 | 4 | 0 |
| [morphology](#module-morphology) | 47 | 32 | 14 | 1 |
| [registration](#module-registration) | 3 | 0 | 3 | 0 |
| [restoration](#module-restoration) | 17 | 4 | 13 | 0 |
| [segmentation](#module-segmentation) | 20 | 5 | 15 | 0 |
| [transform](#module-transform) | 38 | 6 | 28 | 4 |
| [util](#module-util) | 25 | 14 | 9 | 2 |

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/brick.png; preserve as_gray conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-camera"></a>
- [ ] **`data.camera`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; grayscale camera image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/camera.png; preserve grayscale camera image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-cat"></a>
- [ ] **`data.cat`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Canonical alias: `data.chelsea`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: asset hash/license/cache; RGB cat image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Exact alias of data.chelsea; reuse its route: Manifest-backed fetch/decode of data/chelsea.png; preserve RGB cat image. No extra algorithm credit..
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB cat image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/chelsea.png; preserve RGB cat image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-clock"></a>
- [ ] **`data.clock`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; motion-blurred clock image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/clock_motion.png; preserve motion-blurred clock image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-coffee"></a>
- [ ] **`data.coffee`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; RGB coffee image. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/coffee.png; preserve RGB coffee image.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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

<a id="api-data-file-hash"></a>
- [ ] **`data.file_hash`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(fname, alg='sha256')`.
  - Required focus: algorithm names, missing/unreadable file and exact hexadecimal result. Cases to close: empty/small/large local files for supported algorithms.
  - Easiest remaining route: Stream file bytes through standard cryptographic hash provider in bounded chunks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-grass"></a>
- [ ] **`data.grass`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/grass.png; preserve as_gray conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-gravel"></a>
- [ ] **`data.gravel`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `()`.
  - Required focus: asset hash/license/cache; as_gray conversion. Cases to close: warm-cache exact fixture and offline/missing/corrupt resource.
  - Easiest remaining route: Manifest-backed fetch/decode of data/gravel.png; preserve as_gray conversion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-data-horse"></a>
- [ ] **`data.horse`** — full contract closed.
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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(a, b, c)`.
  - Required focus: sphere/spheroid limits, axis ordering and special-function accuracy. Cases to close: spheres, oblate/prolate and triaxial cases.
  - Easiest remaining route: Direct volume formula plus reviewed elliptic-integral surface-area implementation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image_shape, max_shapes, min_shapes=1, min_size=2, max_size=None, num_channels=3, shape=None, intensity_range=None, allow_overlap=False, num_trials=100, rng=None, *, channel_axis=-1)`.
  - Required focus: num_trials, overlap, intensity_range, channel_axis and RNG draw order. Cases to close: seeded mixed-shape crowded and sparse canvases.
  - Easiest remaining route: Preserve seeded shape-choice/placement rejection loop and reuse raster kernels.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, r, c, width, height, feature_coord, color_positive_block=(1.0, 0.0, 0.0), color_negative_block=(0.0, 1.0, 0.0), alpha=0.5, max_n_features=None, rng=None)`.
  - Required focus: random feature selection state, signed colors, ROI clipping and copy. Cases to close: overlapping positive/negative blocks and seeded subsets.
  - Easiest remaining route: Reuse rectangle coordinate fill and alpha overlay for selected Haar blocks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-draw-multiblock-lbp"></a>
- [ ] **`feature.draw_multiblock_lbp`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, r, c, width, height, lbp_code=0, color_greater_block=(1, 1, 1), color_less_block=(0, 0.69, 0.96), alpha=0.5)`.
  - Required focus: bit-to-neighbor orientation, center treatment and dtype/range. Cases to close: each individual bit and mixed block codes.
  - Easiest remaining route: Draw fixed nine-block LBP overlay through rectangle/alpha primitives.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Saved original admission](evidence/draw-multiblock-lbp-saved-raw-admission-v1.json) covers 94 cases per profile. [Independent private-component review](evidence/draw-multiblock-lbp-native-independent-review-v2.json) covers source 0e51dbc1e2c4741e243a91652d8c0b9771c96363: 1,128 target comparisons across two profiles, three scalar/SIMD modes and workers 1/4 pass. Of these, 1,048 complete records match and 80 differ only in post-return JSON collector MXCSR diagnostics, retained explicitly. Target FP and exact caller restoration remain strict. [Root scoped component admission](evidence/draw-multiblock-lbp-native-root-admission-v2.json) passes. [Private integration source/ABI review](evidence/draw-multiblock-lbp-private-integration-admission-v1.json) and [immutable handoff reconciliation](evidence/draw-multiblock-lbp-private-handoff-reconciliation-v1.json) pass in their scoped checks; the qualifier rejects the temporary conditional install rule, so genuine qualification is not passed. Normal enabled MARK/LBP integration is now being authored in a separate successor. Shared genuine core/installed SDK, full domains/platforms, one-thread original-public performance and delivery remain open.


<a id="api-feature-fisher-vector"></a>
- [ ] **`feature.fisher_vector`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(descriptors, gmm, *, improved=False, alpha=0.5)`.
  - Required focus: d_pi/d_mu/d_sigma scaling/order, covariance convention, alpha and zero normalization. Cases to close: hand-computed small GMM with K+2KD output and large descriptor batches.
  - Easiest remaining route: Read GMM probabilities/parameters through provider; compute mixture-weight, mean and variance gradients, concatenate K+2KD values, then optional power/L2 normalization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-graycomatrix"></a>
- [ ] **`feature.graycomatrix`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, distances, angles, levels=None, symmetric=False, normed=False)`.
  - Required focus: levels bounds, angle rounding, symmetric counts, normalize and output dtype. Cases to close: small exact matrices and high-level sparse textures.
  - Easiest remaining route: Count quantized intensity pairs at rounded distance/angle offsets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-graycoprops"></a>
- [ ] **`feature.graycoprops`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(P, prop='contrast')`.
  - Required focus: zero-variance correlation, normalization and property validation. Cases to close: constant/diagonal/asymmetric cooccurrence matrices.
  - Easiest remaining route: Reduce GLCM probabilities to named contrast/correlation/energy statistics.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-haar-like-feature"></a>
- [ ] **`feature.haar_like_feature`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(int_image, r, c, width, height, feature_type=None, feature_coord=None)`.
  - Required focus: feature ordering/types, ROI offsets and signed output overflow. Cases to close: all feature families and custom coordinates.
  - Easiest remaining route: Use integral-image rectangle inclusion-exclusion for supplied/enumerated Haar blocks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-haar-like-feature-coord"></a>
- [ ] **`feature.haar_like_feature_coord`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(width, height, feature_type=None)`.
  - Required focus: combinatorial ordering, feature_type selection and empty geometry. Cases to close: tiny exact enumeration and larger windows.
  - Easiest remaining route: Enumerate ordered rectangle tuples for each Haar feature family.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(descriptors1, descriptors2, metric=None, p=2, max_distance=inf, cross_check=True, max_ratio=1.0)`.
  - Required focus: metric selection, equal-distance tie, zero denominator and empty descriptors. Cases to close: binary/float descriptors with duplicates and outliers.
  - Easiest remaining route: Blocked pairwise Hamming/Lp distances plus reciprocal and ratio filtering.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-match-template"></a>
- [ ] **`feature.match_template`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, template, pad_input=False, mode='constant', constant_values=0)`.
  - Required focus: zero variance, pad_input origin, boundary mode and dimensionality. Cases to close: constant templates, odd/even shapes and boundary peaks.
  - Easiest remaining route: Start with direct correlation plus integral sum/squared-sum normalization; add an FFT backend only for measured large-template crossovers.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-feature-multiblock-lbp"></a>
- [ ] **`feature.multiblock_lbp`** — full contract closed.
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
  - Signature: `(origin, direction)`.
  - Required focus: direction sign, rank failure, predict axis zero and dimensionality. Cases to close: collinear/noisy/high-dimensional points.
  - Easiest remaining route: Center points and use leading singular vector for orthogonal line fit/residual projection.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, level=None, fully_connected='low', positive_orientation='low', *, mask=None)`.
  - Required focus: saddle fully_connected choice, positive_orientation, NaNs/mask and open contours. Cases to close: ambiguous saddles and border/masked level sets.
  - Easiest remaining route: Marching squares with interpolated crossings and stable segment stitching.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0, image1, mask=None)`.
  - Required focus: constant input warnings, sample count, mask and p-value tails. Cases to close: constant/near-perfect/uncorrelated channels.
  - Easiest remaining route: Reuse stable masked centered sums plus reviewed p-value special function/provider.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0, image1, method='standard')`.
  - Required focus: standard versus modified mean, empty sets and max ties. Cases to close: sparse/dense/equidistant 2D/3D masks.
  - Easiest remaining route: Extract foreground points and compute bidirectional nearest-neighbor distances.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-metrics-hausdorff-pair"></a>
- [ ] **`metrics.hausdorff_pair`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0, image1)`.
  - Required focus: empty-set warning, coordinate order and tie selection. Cases to close: multiple maximizing pairs and empty masks.
  - Easiest remaining route: Reuse Hausdorff nearest-neighbor search and return attaining point pair.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(im1, im2, *, win_size=None, gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, full=False, **kwargs)`.
  - Required focus: data_range, sample covariance, channel axis, crop, gaussian weights and return tuple. Cases to close: constant/tiny images, channel layouts and gradient checks.
  - Easiest remaining route: Reuse local mean/variance/covariance filters and SSIM formula; derive gradient/full map paths.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-metrics-variation-of-information"></a>
- [ ] **`metrics.variation_of_information`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image0=None, image1=None, *, table=None, ignore_labels=())`.
  - Required focus: image versus table input, ignore_labels, zeros and result order. Cases to close: merge/split partitions with known conditional entropies.
  - Easiest remaining route: Reuse normalized contingency counts and conditional entropy reductions.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, offset_coordinates=True, tolerance=1e-10, include_borders=True)`.
  - Required focus: offset_coordinates, include_borders, tolerance, empty input and N-D hull dependency. Cases to close: concave/border/degenerate shapes and 2D/3D hulls.
  - Easiest remaining route: Collect occupied boundary/offset points, compute convex hull and classify grid support.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-convex-hull-object"></a>
- [ ] **`morphology.convex_hull_object`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, *, connectivity=2)`.
  - Required focus: connectivity, component overlap and accepted dimensionality. Cases to close: touching/disconnected concave components.
  - Easiest remaining route: Label objects then reuse per-component convex_hull_image and combine masks.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, seed_point, *, footprint=None, connectivity=None, tolerance=None)`.
  - Required focus: seed bounds, tolerance promotion, connectivity versus footprint and NaN. Cases to close: exact/tolerant 2D/3D connected regions.
  - Easiest remaining route: Queue-based connected tolerance predicate from seed using shared offsets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-flood-fill"></a>
- [ ] **`morphology.flood_fill`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, seed_point, new_value, *, footprint=None, connectivity=None, tolerance=None, in_place=False)`.
  - Required focus: in_place, new_value casting, overlap and return identity. Cases to close: typed replacements and in-place aliases.
  - Easiest remaining route: Reuse flood mask then dtype-correct replacement into copy or original.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Easiest remaining route: Audit/extend existing native implementation: Reuse grayscale reconstruction of intensity-shifted seed then h-difference test.
  - Evidence locators: parity [tests/test_h_extrema_public.py](../tests/test_h_extrema_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-h-minima"></a>
- [ ] **`morphology.h_minima`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(image, h, footprint=None)`.
  - Required focus: integer saturation, floating correction, h=0 and minima threshold. Cases to close: minima at exactly h and dtype limits.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse erosion reconstruction of upward-shifted seed then h-difference test.
  - Evidence locators: parity [tests/test_h_extrema_public.py](../tests/test_h_extrema_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-isotropic-closing"></a>
- [ ] **`morphology.isotropic_closing`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: radius inequality, spacing, border convention and out. Cases to close: large-radius masks and anisotropic spacing.
  - Easiest remaining route: Compose exact-distance binary dilation then erosion.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-isotropic-dilation"></a>
- [ ] **`morphology.isotropic_dilation`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: radius equality, all-true/all-false input and spacing. Cases to close: equidistant boundary pixels and large radii.
  - Easiest remaining route: Threshold distance transform of complement.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-isotropic-erosion"></a>
- [ ] **`morphology.isotropic_erosion`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: strict versus nonstrict radius, outside-image convention and out. Cases to close: thin features and all-true images.
  - Easiest remaining route: Threshold foreground distance transform.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-isotropic-opening"></a>
- [ ] **`morphology.isotropic_opening`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, out=None, spacing=None)`.
  - Required focus: intermediate border effects, radius and out. Cases to close: sparse/small objects and large radii.
  - Easiest remaining route: Compose exact-distance erosion then dilation.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, footprint=None, connectivity=None, indices=False, allow_borders=True)`.
  - Required focus: allow_borders, connectivity/footprint, indices and plateau tie. Cases to close: border and interior flat maxima.
  - Easiest remaining route: Flood equal-valued plateaus and reject those touching larger neighbors.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-local-minima"></a>
- [ ] **`morphology.local_minima`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, footprint=None, connectivity=None, indices=False, allow_borders=True)`.
  - Required focus: allow_borders, connectivity/footprint, indices and dtype extremes. Cases to close: border and interior flat minima.
  - Easiest remaining route: Reuse plateau flood with reversed comparison.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, mask=None, return_distance=False, *, rng=None)`.
  - Required focus: RNG state, masked pixels, return_distance and tie order. Cases to close: symmetric shapes, holes and seeded ties.
  - Easiest remaining route: Distance transform then distance/cornerness/seeded-tie ordered topology thinning.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, *, method=None)`.
  - Required focus: method/dimension defaults, topology, iteration order and boolean output. Cases to close: loops/branches/thin bridges and 3D topology.
  - Easiest remaining route: Port source LUT-based Zhang 2D and Lee 2D/3D thinning with exact deletion schedule.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-morphology-star"></a>
- [ ] **`morphology.star`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(a, dtype=<class 'numpy.uint8'>)`.
  - Required focus: a=0/small sizes, parity and boundary cells. Cases to close: small exhaustive star footprints.
  - Easiest remaining route: Audit/extend existing native implementation: Union axis-aligned and rotated-square supports using exact source shape convention.
  - Evidence locators: parity [tests/test_footprint_shapes_public.py](../tests/test_footprint_shapes_public.py); benchmark [benchmarks/coverage.py](../benchmarks/coverage.py).

<a id="api-morphology-thin"></a>
- [ ] **`morphology.thin`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, max_num_iter=None)`.
  - Required focus: max_num_iter, convergence, deletion order and border padding. Cases to close: loops, isolated pixels and truncated iterations.
  - Easiest remaining route: Port two-subiteration Guo-Hall LUT thinning.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, *, channel_axis=None)`.
  - Required focus: bins quantization, sigma_color default, channels, mode and zero weights. Cases to close: constant/edge/color images and wide windows.
  - Easiest remaining route: Spatial Gaussian times intensity-distance LUT, normalize local weighted sums.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, *, radius=100, kernel=None, nansafe=False, num_threads=<DEPRECATED>, workers=None)`.
  - Required focus: nansafe, kernel origin, radius, workers/num_threads warning and dtype. Cases to close: NaNs, bright background and wide kernels.
  - Easiest remaining route: Min-plus neighborhood erosion using nonflat ball/ellipsoid height kernel.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(label_image, distance=1, spacing=1)`.
  - Required focus: equidistant-label tie, spacing, background and output label dtype. Cases to close: competing labels and anisotropic 2D/3D volumes.
  - Easiest remaining route: Reuse exact Euclidean nearest-label indices and threshold distance.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-felzenszwalb"></a>
- [ ] **`segmentation.felzenszwalb`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, scale=1, sigma=0.8, min_size=20, *, channel_axis=-1)`.
  - Required focus: edge-order ties, scale/min_size postpass and channel_axis. Cases to close: flat/equal-weight images and fine/coarse partitions.
  - Easiest remaining route: Gaussian preprocessing, sorted neighbor edges and size/internal-difference union-find merges.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-find-boundaries"></a>
- [ ] **`segmentation.find_boundaries`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64`.
  - Signature: `(label_img, connectivity=1, mode='thick', background=0)`.
  - Required focus: connectivity, background and doubled subpixel shape. Cases to close: triple junctions and thin labels.
  - Easiest remaining route: Audit/extend existing native implementation: Compare neighbor labels and apply thick/inner/outer/subpixel construction.
  - Evidence locators: parity [tests/test_find_boundaries_public.py](../tests/test_find_boundaries_public.py); benchmark [benchmarks/find_boundaries.py](../benchmarks/find_boundaries.py).

<a id="api-segmentation-flood"></a>
- [ ] **`segmentation.flood`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, seed_point, *, footprint=None, connectivity=None, tolerance=None)`.
  - Canonical alias: `morphology.flood`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: seed bounds, tolerance promotion, connectivity versus footprint and NaN. Cases to close: exact/tolerant 2D/3D connected regions.
  - Easiest remaining route: Exact alias of morphology.flood; reuse its route: Queue-based connected tolerance predicate from seed using shared offsets. No extra algorithm credit..
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-flood-fill"></a>
- [ ] **`segmentation.flood_fill`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, seed_point, new_value, *, footprint=None, connectivity=None, tolerance=None, in_place=False)`.
  - Canonical alias: `morphology.flood_fill`. Reuse the kernel; verify alias identity and public contract separately.
  - Required focus: in_place, new_value casting, overlap and return identity. Cases to close: typed replacements and in-place aliases.
  - Easiest remaining route: Exact alias of morphology.flood_fill; reuse its route: Reuse flood mask then dtype-correct replacement into copy or original. No extra algorithm credit..
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-inverse-gaussian-gradient"></a>
- [ ] **`segmentation.inverse_gaussian_gradient`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, alpha=100.0, sigma=5.0)`.
  - Required focus: alpha, sigma, dtype and boundaries. Cases to close: smooth/step edges and constant images.
  - Easiest remaining route: Reuse Gaussian gradient magnitude and inverse square-root edge-stop formula.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-join-segmentations"></a>
- [ ] **`segmentation.join_segmentations`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(s1, s2, return_mapping: bool = False)`.
  - Required focus: dtype overflow, mapping objects, return_mapping and zero labels. Cases to close: sparse huge label pairs and mapping round trips.
  - Easiest remaining route: Pair-label encode then stable sequential relabel with inverse mappings.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-segmentation-mark-boundaries"></a>
- [ ] **`segmentation.mark_boundaries`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, label_img, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)`.
  - Required focus: mode, background_label, subpixel upsampling, image dtype and color. Cases to close: inner/outer/subpixel boundaries and grayscale/RGB images.
  - Easiest remaining route: Reuse find_boundaries and color overlay with optional outline.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Saved original admission](evidence/mark-boundaries-saved-raw-admission-v2.json) covers 125 cases per profile (250 complete observations, 251 consumed calls including the retained failed attempt). One separately authorized fixed-codec witness closed the structured-error gap; no successful original rows were replayed. [Independent native successor admission](evidence/mark-boundaries-fp-successor-admission-v1.json) passes on source 6d4373b5720b5ee959c1d313e14e3d8f754085e4 for 1,500 saved worker/mode comparisons (125 cases per profile, workers 1/4, normal/disabled/scalar), including strict tiny-color target FP. Small RHS palette assignment remains explicit NumPy provider work. Genuine installation, platform/domain closure, one-thread original-public performance and delivery remain open. [Executable benchmark source review](evidence/mark-boundaries-benchmark-source-admission-v1.json) and [handoff reconciliation](evidence/mark-boundaries-benchmark-handoff-reconciliation-v1.json) pass on 446bde47e7462278389c0a87affb881ed775f1ec: 52 source/zero-body controls per Python, fixed 8 cases, no actual targets or timings. Genuine binding, outer first-cohort and separate timing GO remain pending.


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
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(a)`.
  - Required focus: supported size, modular line indexing and unsigned overflow. Cases to close: tiny exact prime grids and integer limits.
  - Easiest remaining route: Implement finite Radon integer line sums on prime-sized square grid.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Source/storage admission](evidence/finite-radon-source-storage-admission-v1.json) at d2109de2a0b7f07e27f0a892585cd7414e36bfbf: 54 recipes for this API, bounded factory/protocol scope only. Integer native/SDK invariants are authored and not independently admitted original parity. Actual original observations: 0. Guarded capture controller/entry readiness, floating/complex/protocol/platform domains, genuine hot swap/complete installed SDK and one-thread performance remain open.

<a id="api-transform-hough-circle"></a>
- [ ] **`transform.hough_circle`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, radius, normalize=True, full_output=False)`.
  - Required focus: normalize, full_output padding, radius cast and duplicate perimeter votes. Cases to close: exact single circles and dense edge masks.
  - Easiest remaining route: Vote precomputed discrete perimeter offsets for each radius.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-hough-circle-peaks"></a>
- [ ] **`transform.hough_circle_peaks`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=inf, total_num_peaks=inf, normalize=False)`.
  - Required focus: normalize scoring, per-radius/total limits, x/y suppression ties. Cases to close: competing radii and equal peaks.
  - Easiest remaining route: Peak-suppress each circle accumulator then stable cross-radius selection.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-hough-ellipse"></a>
- [ ] **`transform.hough_ellipse`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, threshold=4, accuracy=1, min_size=4, max_size=None)`.
  - Required focus: accuracy bins, threshold, min/max size, output field order and ties. Cases to close: synthetic ellipses and dense edge clutter.
  - Easiest remaining route: Port triplet edge-point ellipse hypothesis/voting algorithm.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-hough-line"></a>
- [ ] **`transform.hough_line`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, theta=None)`.
  - Required focus: theta order/range, rho rounding, accumulator shape and overflow. Cases to close: horizontal/diagonal lines at bin boundaries.
  - Easiest remaining route: Vectorized rho computation per theta with exact integer accumulator votes.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-hough-line-peaks"></a>
- [ ] **`transform.hough_line_peaks`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(hspace, angles, dists, min_distance=9, min_angle=10, threshold=None, num_peaks=inf)`.
  - Required focus: min_distance/min_angle, threshold default, tie order and peak count. Cases to close: equal/symmetric peaks near angle boundaries.
  - Easiest remaining route: Separable accumulator maximum suppression with angle wrap logic.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-ifrt2"></a>
- [ ] **`transform.ifrt2`** — full contract closed.
  - Main: `upstream-only`; gate status `C~ N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(a)`.
  - Required focus: prime-size assumptions, division and signed/unsigned dtype. Cases to close: exact forward/inverse integer fixtures.
  - Easiest remaining route: Invert finite Radon modular line sums with exact normalization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.
  - Current private gate: [Source/storage admission](evidence/finite-radon-source-storage-admission-v1.json) at d2109de2a0b7f07e27f0a892585cd7414e36bfbf: 54 recipes for this API, bounded factory/protocol scope only. Integer native/SDK invariants are authored and not independently admitted original parity. Actual original observations: 0. Guarded capture controller/entry readiness, floating/complex/protocol/platform domains, genuine hot swap/complete installed SDK and one-thread performance remain open.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(theta)`.
  - Required focus: ties, duplicates, angle periodicity and generator semantics. Cases to close: repeated/unsorted angle lists.
  - Easiest remaining route: Ordered nearest-angle selection around golden-ratio angular targets.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-probabilistic-hough-line"></a>
- [ ] **`transform.probabilistic_hough_line`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, threshold=10, line_length=50, line_gap=10, theta=None, rng=None)`.
  - Required focus: RNG draw order, line_gap/length, threshold and endpoint order. Cases to close: crossing lines, gaps and fixed saved RNG state.
  - Easiest remaining route: Preserve seeded edge visitation, line votes, extension and pixel removal.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-pyramid-expand"></a>
- [ ] **`transform.pyramid_expand`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, upscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: channel_axis, rounding, mode and preserve_range. Cases to close: odd/even sizes and per-axis channels.
  - Easiest remaining route: Reuse resize-up and Gaussian smoothing with exact scale-derived sigma.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-pyramid-gaussian"></a>
- [ ] **`transform.pyramid_gaussian`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: yield count/order, original first level, channel_axis and lazy exceptions. Cases to close: tiny/odd shapes and finite/infinite max_layer.
  - Easiest remaining route: Lazy generator chaining native pyramid_reduce until shape/max_layer termination.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-pyramid-laplacian"></a>
- [ ] **`transform.pyramid_laplacian`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: first/last level, residual dtype, sigma and lazy state. Cases to close: reconstruction-oriented pyramids and odd shapes.
  - Easiest remaining route: Lazy levels of image minus smoothed image then reduction.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-transform-pyramid-reduce"></a>
- [ ] **`transform.pyramid_reduce`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)`.
  - Required focus: downscale validation, sigma default, shape rounding and channels. Cases to close: noninteger downscale and anisotropic inputs.
  - Easiest remaining route: Gaussian smooth then native resize-down at source-defined output shape.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(ar, crop_width, copy=False, order='K')`.
  - Required focus: scalar/tuple widths, negative/large crops, order and memory sharing. Cases to close: asymmetric N-D crops and C/F/negative strides.
  - Easiest remaining route: Return slice view or requested-order copy after crop-width normalization.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

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
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, mode='gaussian', rng=None, clip=True, **kwargs)`.
  - Required focus: gaussian/localvar/poisson/salt/pepper/s&p/speckle, signed range and RNG state. Cases to close: saved-state mode matrix and clipping/range edges.
  - Easiest remaining route: Preserve Generator draws/distributions and native mode-specific transforms/clipping.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-regular-grid"></a>
- [ ] **`util.regular_grid`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(ar_shape, n_points)`.
  - Required focus: n_points extremes, dimensionality and slice start/stop convention. Cases to close: anisotropic/tiny shapes and target counts.
  - Easiest remaining route: Choose rounded per-axis strides and slices for approximately uniform sample count.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-regular-seeds"></a>
- [ ] **`util.regular_seeds`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ar_shape, n_points, dtype=<class 'int'>)`.
  - Required focus: dtype, label numbering and requested-count approximation. Cases to close: anisotropic grids and limited integer dtypes.
  - Easiest remaining route: Audit/extend existing native implementation: Reuse regular_grid slices then assign ordered seed labels.
  - Evidence locators: parity [tests/test_regular_seeds_public.py::test_regular_seeds_native_public_contract](../tests/test_regular_seeds_public.py); benchmark [benchmarks/regular_seeds.py](../benchmarks/regular_seeds.py).

<a id="api-util-slice-along-axes"></a>
- [ ] **`util.slice_along_axes`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(image, slices, axes=None, copy=False)`.
  - Required focus: duplicate/negative axes, bounds and sharing. Cases to close: reordered/negative axes and copy flags.
  - Easiest remaining route: Normalize axes/slice pairs then return view or copy.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-unique-rows"></a>
- [ ] **`util.unique_rows`** — full contract closed.
  - Main: `native-partial`; gate status `C? N~ H? P? S? V? W? B? R~`; default dtype slice: `uint8, float32, float64`.
  - Signature: `(ar)`.
  - Required focus: dtype, NaNs, row order and noncontiguous input. Cases to close: duplicate/noncontiguous rows with special values.
  - Easiest remaining route: Audit/extend existing native implementation: Stable/contract-ordered row deduplication using contiguous byte keys or typed comparisons.
  - Evidence locators: parity [tests/test_unique_rows_public.py::test_unique_rows_native_public_contract](../tests/test_unique_rows_public.py); benchmark [benchmarks/unique_rows.py](../benchmarks/unique_rows.py).

<a id="api-util-view-as-blocks"></a>
- [ ] **`util.view_as_blocks`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(arr_in, block_shape)`.
  - Required focus: divisibility, dimensionality, contiguous assumptions and write sharing. Cases to close: C/F/noncontiguous arrays and block mutation.
  - Easiest remaining route: Construct validated reshape/strided block view.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

<a id="api-util-view-as-windows"></a>
- [ ] **`util.view_as_windows`** — full contract closed.
  - Main: `upstream-only`; gate status `C? N? H? P? S? V? W? B? R?`; default dtype slice: `none registered`.
  - Signature: `(arr_in, window_shape, step=1)`.
  - Required focus: scalar/vector step, shape, strides and dangerous overlap semantics. Cases to close: overlapping windows and negative/zero invalid steps.
  - Easiest remaining route: Construct rolling strided view with validated window/step.
  - Evidence locators: parity not registered in the shipping inventory; benchmark not registered in the shipping inventory.

</details>

## Class/member checklist — 29 classes, 284 static paths

Top-level class completion also requires every applicable member and runtime-discovered state/protocol below. Native member labels remain partial. Each member must satisfy the same C/N/H/P/S/V/W/B/R gates; constructor/getter metadata cannot substitute for solver or descriptor computation.

### `feature.BRIEF` — 2 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.BRIEF.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, descriptor_size=256, patch_size=49, mode='normal', sigma=1, rng=1)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.BRIEF.extract`** — method, defined; current `upstream-only`.
  - Signature: `(self, image, keypoints)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.CENSURE` — 2 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.CENSURE.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.CENSURE.detect`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.Cascade` — 11 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.Cascade.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.__new__`** — method, defined; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.__reduce__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.__setstate__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.detect_multi_scale`** — method, defined; current `upstream-only`.
  - Signature: `(self, img, scale_factor, step_ratio, min_size, max_size, min_neighbor_number=4, intersection_score_threshold=0.5)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.eps`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.features_number`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.stages_number`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.stumps_number`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.window_height`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.Cascade.window_width`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.ORB` — 4 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.ORB.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, downscale=1.2, n_scales=8, n_keypoints=500, fast_n=9, fast_threshold=0.08, harris_k=0.04)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.ORB.detect`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.ORB.detect_and_extract`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.ORB.extract`** — method, defined; current `upstream-only`.
  - Signature: `(self, image, keypoints, scales, orientations)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `feature.SIFT` — 5 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`feature.SIFT.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5, c_dog=0.013333333333333334, c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.deltas`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.detect`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.detect_and_extract`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`feature.SIFT.extract`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `filters.LPIFilter2D` — 2 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`filters.LPIFilter2D.__call__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, data)`.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark not registered.

- [ ] **`filters.LPIFilter2D.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Signature: `(self, impulse_response, **filter_params)`.
  - Evidence locators: parity [tests/test_lpi_filter_public.py](../tests/test_lpi_filter_public.py); benchmark not registered.

### `future.TrainableSegmenter` — 4 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`future.TrainableSegmenter.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, clf=None, features_func=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`future.TrainableSegmenter.compute_features`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`future.TrainableSegmenter.fit`** — method, defined; current `upstream-only`.
  - Signature: `(self, image, labels)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`future.TrainableSegmenter.predict`** — method, defined; current `upstream-only`.
  - Signature: `(self, image)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__new__`** — method, defined; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__reduce__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__setstate__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.find_costs`** — method, defined; current `upstream-only`.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.goal_reached`** — method, defined; current `upstream-only`.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.offsets`** — attribute, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP.traceback`** — method, defined; current `upstream-only`.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP_Connect` — 11 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP_Connect.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, /, *args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__new__`** — method, defined; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__reduce__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__setstate__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.create_connection`** — method, defined; current `upstream-only`.
  - Signature: `(self, id1, id2, tb1, tb2, cost1, cost2)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.find_costs`** — method, inherited; current `upstream-only`.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.goal_reached`** — method, inherited; current `upstream-only`.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.offsets`** — attribute, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Connect.traceback`** — method, inherited; current `upstream-only`.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP_Flexible` — 13 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP_Flexible.__init__`** — constructor, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__new__`** — method, defined; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__reduce__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__setstate__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.examine_neighbor`** — method, defined; current `upstream-only`.
  - Signature: `(self, index, new_index, offset_length)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.find_costs`** — method, inherited; current `upstream-only`.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.goal_reached`** — method, inherited; current `upstream-only`.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.offsets`** — attribute, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.traceback`** — method, inherited; current `upstream-only`.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.travel_cost`** — method, defined; current `upstream-only`.
  - Signature: `(self, old_cost, new_cost, offset_length)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Flexible.update_node`** — method, defined; current `upstream-only`.
  - Signature: `(self, index, new_index, offset_length)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.MCP_Geometric` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.MCP_Geometric.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__new__`** — method, defined; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__reduce__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__reduce_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__setstate__`** — method, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.__setstate_cython__`** — method, defined; current `upstream-only`.
  - Signature: `(self, __pyx_state)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.find_costs`** — method, inherited; current `upstream-only`.
  - Signature: `(self, starts, ends=None, find_all_ends=True, max_coverage=1.0, max_cumulative_cost=<DEPRECATED>, max_cost=<DEPRECATED>, *, max_step_cost=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.goal_reached`** — method, inherited; current `upstream-only`.
  - Signature: `(self, index, cumcost)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.offsets`** — attribute, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.MCP_Geometric.traceback`** — method, inherited; current `upstream-only`.
  - Signature: `(self, end)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `graph.RAG` — 46 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`graph.RAG.__contains__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__getitem__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, label_image=None, connectivity=1, data=None, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__iter__`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__len__`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__new__`** — staticmethod, inherited; current `upstream-only`.
  - Signature: `(cls, *args, backend=None, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.__str__`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_edge`** — method, defined; current `upstream-only`.
  - Signature: `(self, u, v, attr_dict=None, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_edges_from`** — method, inherited; current `upstream-only`.
  - Signature: `(self, ebunch_to_add, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_node`** — method, defined; current `upstream-only`.
  - Signature: `(self, n, attr_dict=None, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_nodes_from`** — method, inherited; current `upstream-only`.
  - Signature: `(self, nodes_for_adding, **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.add_weighted_edges_from`** — method, inherited; current `upstream-only`.
  - Signature: `(self, ebunch_to_add, weight='weight', **attr)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.adj`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.adjacency`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.clear`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.clear_edges`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.copy`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.degree`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.edge_subgraph`** — method, inherited; current `upstream-only`.
  - Signature: `(self, edges)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.edges`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.fresh_copy`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.get_edge_data`** — method, inherited; current `upstream-only`.
  - Signature: `(self, u, v, default=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.has_edge`** — method, inherited; current `upstream-only`.
  - Signature: `(self, u, v)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.has_node`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.is_directed`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.is_multigraph`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.merge_nodes`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst, weight_func=<function min_weight>, in_place=True, extra_arguments=None, extra_keywords=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.name`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.nbunch_iter`** — method, inherited; current `upstream-only`.
  - Signature: `(self, nbunch=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.neighbors`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.next_id`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.nodes`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.number_of_edges`** — method, inherited; current `upstream-only`.
  - Signature: `(self, u=None, v=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.number_of_nodes`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.order`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_edge`** — method, inherited; current `upstream-only`.
  - Signature: `(self, u, v)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_edges_from`** — method, inherited; current `upstream-only`.
  - Signature: `(self, ebunch)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_node`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.remove_nodes_from`** — method, inherited; current `upstream-only`.
  - Signature: `(self, nodes)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.size`** — method, inherited; current `upstream-only`.
  - Signature: `(self, weight=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.subgraph`** — method, inherited; current `upstream-only`.
  - Signature: `(self, nodes)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_directed`** — method, inherited; current `upstream-only`.
  - Signature: `(self, as_view=False)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_directed_class`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_undirected`** — method, inherited; current `upstream-only`.
  - Signature: `(self, as_view=False)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.to_undirected_class`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`graph.RAG.update`** — method, inherited; current `upstream-only`.
  - Signature: `(self, edges=None, nodes=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `io.ImageCollection` — 9 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`io.ImageCollection.__getitem__`** — method, defined; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, load_pattern, conserve_memory=True, load_func=None, **load_func_kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__iter__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__len__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.__str__`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.concatenate`** — method, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.conserve_memory`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.files`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.ImageCollection.reload`** — method, defined; current `upstream-only`.
  - Signature: `(self, n=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `io.MultiImage` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`io.MultiImage.__getitem__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, filename, conserve_memory=True, dtype=None, **imread_kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__iter__`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__len__`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.__str__`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.concatenate`** — method, inherited; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.conserve_memory`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.filename`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.files`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`io.MultiImage.reload`** — method, inherited; current `upstream-only`.
  - Signature: `(self, n=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.CircleModel` — 6 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.CircleModel.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, center, radius)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.params`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.predict_xy`** — method, defined; current `upstream-only`.
  - Signature: `(self, t, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.CircleModel.residuals`** — method, defined; current `upstream-only`.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.EllipseModel` — 6 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.EllipseModel.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, center, axis_lengths, theta)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.params`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.predict_xy`** — method, defined; current `upstream-only`.
  - Signature: `(self, t, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.EllipseModel.residuals`** — method, defined; current `upstream-only`.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.LineModelND` — 8 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.LineModelND.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, origin, direction)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.params`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.predict`** — method, defined; current `upstream-only`.
  - Signature: `(self, x, axis=0, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.predict_x`** — method, defined; current `upstream-only`.
  - Signature: `(self, y, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.predict_y`** — method, defined; current `upstream-only`.
  - Signature: `(self, x, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.LineModelND.residuals`** — method, defined; current `upstream-only`.
  - Signature: `(self, data, params=<DEPRECATED>)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `measure.RansacModelProtocol` — 6 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`measure.RansacModelProtocol.__class_getitem__`** — method, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, *args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.__init_subclass__`** — classmethod, inherited; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.__subclasshook__`** — classmethod, defined; current `upstream-only`.
  - Signature: `(other)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(*data)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`measure.RansacModelProtocol.residuals`** — method, defined; current `upstream-only`.
  - Signature: `(self, *data)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.AffineTransform` — 17 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.AffineTransform.__add__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__array__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__call__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Signature: `(self, matrix=None, *, scale=None, shear=None, rotation=None, translation=None, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__nice__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__repr__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.__str__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.dimensionality`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, src, dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Signature: `(dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.identity`** — classmethod, inherited; current `native-class-member-partial`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.inverse`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.rotation`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.scale`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.shear`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.AffineTransform.translation`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.EssentialMatrixTransform` — 8 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.EssentialMatrixTransform.__call__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, *, rotation=None, translation=None, matrix=None, dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.dimensionality`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.identity`** — classmethod, inherited; current `upstream-only`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.inverse`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.EssentialMatrixTransform.residuals`** — method, inherited; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.EuclideanTransform` — 15 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.EuclideanTransform.__add__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__array__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__call__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Signature: `(self, matrix=None, *, rotation=None, translation=None, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__nice__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__repr__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.__str__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.dimensionality`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Signature: `(src, dst) -> Union[Self, skimage._shared.utils.FailedEstimation]`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.identity`** — classmethod, inherited; current `native-class-member-partial`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.inverse`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.rotation`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.EuclideanTransform.translation`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.FundamentalMatrixTransform` — 8 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.FundamentalMatrixTransform.__call__`** — method, defined; current `upstream-only`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.__init__`** — constructor, inherited; current `upstream-only`.
  - Signature: `(self, matrix=None, *, dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.dimensionality`** — property, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.identity`** — classmethod, inherited; current `upstream-only`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.inverse`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.FundamentalMatrixTransform.residuals`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.PiecewiseAffineTransform` — 7 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.PiecewiseAffineTransform.__call__`** — method, defined; current `upstream-only`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.identity`** — classmethod, defined; current `upstream-only`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.inverse`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PiecewiseAffineTransform.residuals`** — method, inherited; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.PolynomialTransform` — 7 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.PolynomialTransform.__call__`** — method, defined; current `upstream-only`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self, params=None, *, dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst, order=2, weights=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(src, dst, order=2, weights=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.identity`** — classmethod, defined; current `upstream-only`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.inverse`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.PolynomialTransform.residuals`** — method, inherited; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

### `transform.ProjectiveTransform` — 13 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.ProjectiveTransform.__add__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__array__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__call__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__init__`** — constructor, inherited; current `native-class-member-partial`.
  - Signature: `(self, matrix=None, *, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__nice__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__repr__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.__str__`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.dimensionality`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, src, dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Signature: `(src, dst, weights=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.identity`** — classmethod, defined; current `native-class-member-partial`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.inverse`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.ProjectiveTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.SimilarityTransform` — 16 static members

Main: `native-class-partial`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.SimilarityTransform.__add__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, other)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__array__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, dtype=None, copy=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__call__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__init__`** — constructor, defined; current `native-class-member-partial`.
  - Signature: `(self, matrix=None, *, scale=None, rotation=None, translation=None, dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__nice__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__repr__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.__str__`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.dimensionality`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.estimate`** — method, defined; current `native-class-member-partial`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.from_estimate`** — classmethod, defined; current `native-class-member-partial`.
  - Signature: `(dst) -> Union[Self, skimage._shared.utils.FailedEstimation]`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.identity`** — classmethod, inherited; current `native-class-member-partial`.
  - Signature: `(dimensionality=None)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.inverse`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.residuals`** — method, inherited; current `native-class-member-partial`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.rotation`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.scale`** — property, defined; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

- [ ] **`transform.SimilarityTransform.translation`** — property, inherited; current `native-class-member-partial`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_classes_public.py](../tests/test_transform_classes_public.py); benchmark not registered.

### `transform.ThinPlateSplineTransform` — 5 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`transform.ThinPlateSplineTransform.__call__`** — method, defined; current `upstream-only`.
  - Signature: `(self, coords)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.__init__`** — constructor, defined; current `upstream-only`.
  - Signature: `(self)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.estimate`** — method, defined; current `upstream-only`.
  - Signature: `(self, src, dst)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.from_estimate`** — classmethod, defined; current `upstream-only`.
  - Signature: `(src, dst) -> Union[Self, skimage._shared.utils.FailedEstimation]`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`transform.ThinPlateSplineTransform.inverse`** — property, defined; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

### `util.FailedEstimationAccessError` — 13 static members

Main: `exception-metadata-provider`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`util.FailedEstimationAccessError.__getattribute__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, name, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__getstate__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__init__`** — constructor, inherited; current `exception-metadata-member`.
  - Signature: `(self, /, *args, **kwargs)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__new__`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__reduce__`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(self, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__repr__`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(self, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__setstate__`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.__str__`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(self, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.add_note`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.args`** — attribute, inherited; current `exception-metadata-member`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.name`** — attribute, inherited; current `exception-metadata-member`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.obj`** — attribute, inherited; current `exception-metadata-member`.
  - Signature: `None`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

- [ ] **`util.FailedEstimationAccessError.with_traceback`** — method, inherited; current `exception-metadata-member`.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity [tests/test_transform_exception_public.py](../tests/test_transform_exception_public.py); benchmark not registered.

### `util.PendingSkimage2Change` — 10 static members

Main: `upstream-only`. Close construction, fitted/unfitted state, properties, mutation/copy/serialization, inheritance, callbacks/RNG, inverse/composition and returned objects as applicable.

- [ ] **`util.PendingSkimage2Change.__getattribute__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, name, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__init__`** — constructor, inherited; current `upstream-only`.
  - Signature: `(self, /, *args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__new__`** — method, inherited; current `upstream-only`.
  - Signature: `(*args, **kwargs)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__reduce__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__repr__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__setstate__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.__str__`** — method, inherited; current `upstream-only`.
  - Signature: `(self, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.add_note`** — method, inherited; current `upstream-only`.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.args`** — attribute, inherited; current `upstream-only`.
  - Signature: `None`.
  - Evidence locators: parity not registered; benchmark not registered.

- [ ] **`util.PendingSkimage2Change.with_traceback`** — method, inherited; current `upstream-only`.
  - Signature: `(self, object, /)`.
  - Evidence locators: parity not registered; benchmark not registered.

## Runtime and whole-library closure

- [ ] Inventory runtime RegionProperties objects and every exposed attribute, including the still-unassessed property partitions; verify computation, cache invalidation and scalar/layout/state behavior.
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
4. Include the checklist delta in the same validated increment. If coverage did not change, record “reviewed, no coverage change” in the existing closeout; do not add a separate audit task or bookkeeping-only scientific campaign.
