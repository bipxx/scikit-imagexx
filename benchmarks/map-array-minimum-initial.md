# MapArray: first minimum-profile comparison

**Mixed result: 5 of 9 workloads faster; geometric mean 1.357390612×.** Four workloads are slower, including both signed sparse cases. This is the first original/current comparison for `util.map_array` on one Linux host at source `b08aef5fb21cb7bfe87ffa04c35767eab1727d8e`.

[Raw report](map-array-minimum-initial.json) preserves all 126 timing samples and original numeric text. Speedup is scikit-image median / C++ public-call median; values below 1 indicate a loss. Added time is C++ minus original.

| Workload | Input → output | Shape | Original ms | C++ ms | Speedup | Added ms |
|---|---|---|---:|---:|---:|---:|
| dense-u8-u16-64 | uint8 → uint16 | 64 × 64 | 0.040390549 | 0.009297905 | 4.344048512× | -0.031092644 |
| dense-u8-u16-256 | uint8 → uint16 | 256 × 256 | 0.182489704 | 0.066372000 | 2.749498338× | -0.116117704 |
| dense-u8-u16-1024 | uint8 → uint16 | 1024 × 1024 | 2.671608500 | 0.983376882 | 2.716769682× | -1.688231618 |
| signed64-sparse-f64-256 | int64 → float64 | 256 × 256 | 0.336146476 | 1.219551643 | 0.275631195× | +0.883405167 |
| signed64-sparse-f64-1024 | int64 → float64 | 1024 × 1024 | 7.197022500 | 21.057111000 | 0.341785846× | +13.860088500 |
| uint64-wide-c128-256 | uint64 → complex128 | 256 × 256 | 0.689529917 | 0.750491913 | 0.918770615× | +0.060961996 |
| uint64-wide-c128-1024 | uint64 → complex128 | 1024 × 1024 | 12.465582000 | 12.220805000 | 1.020029532× | -0.244777000 |
| strided-u16-u32-512 | uint16 → uint32 | 512 × 512 | 3.183608800 | 0.404463743 | 7.871184639× | -2.779145057 |
| strided-out-u8-f32-65536 | uint8 → float32 | 65536 | 0.180009533 | 0.259474850 | 0.693745592× | +0.079465317 |

| Cohort | Wins / cases | Geometric mean |
|---|---:|---:|
| dense | 3 / 3 | 3.189581106× |
| signed-sparse | 0 / 2 | 0.306931330× |
| strided-input | 1 / 1 | 7.871184639× |
| strided-output | 0 / 1 | 0.693745592× |
| wide | 1 / 2 | 0.968077043× |

Eight allocating calls have geometric mean 1.476192236× (5 wins). The preallocated strided-output call is 0.693745592×.

All four losses are retained:
- `signed64-sparse-f64-256`: 0.275631195×, +0.883405167 ms.
- `signed64-sparse-f64-1024`: 0.341785846×, +13.860088500 ms.
- `uint64-wide-c128-256`: 0.918770615×, +0.060961996 ms.
- `strided-out-u8-f32-65536`: 0.693745592×, +0.079465317 ms.

The largest penalty is the 1024² signed sparse workload: 7.197022500 ms becomes 21.057111000 ms (+13.860088500 ms). These results identify unresolved workloads; they do not establish which internal stage causes the losses.

## Measurement and correctness scope

Fixed seed 2026; seven paired rounds targeting 19 ms, one global worker, and all four reference pool limits set to one. The eight `out=None` calls include public output allocation and destruction. The ninth call reuses a strided output buffer, excludes output allocation, and releases the returned reference inside the timing primitive. Every call fully overwrites its logical output. There is no input/value-alias timing workload.

Dense inputs cover 64², 256² and 1024² with 257 mapping entries (a duplicate exercises last-key wins). Signed sparse and wide unsigned workloads each cover 256² and 1024² with six keys; wide keys include the high bit and maximum uint64. The strided input is a 512² view of a 1024² backing array with 1,025 keys. Preallocated output has 65,536 elements, byte stride 8 and 257 keys.

Before timing, fresh pinned scikit-image 0.26 and the C++ public API match an independent copied-value last-key lookup oracle in full output bytes, dtype, layout and output identity. Inputs, mappings, backing pixels and output gaps are checked unchanged. Genuine `map_array_apply` C returns are accepted for all nine cases while original Python and Cython forwarding are blocked; hooks are restored before timing. Exported image payloads contain hashes only: offline review authenticates the source prechecks and matching hashes rather than reconstructing full images.

Literal Python reshape/view/shape-setter/cast behavior remains in the public prelude; C++ owns lookup, value snapshots and destination writes. Broader protocol, warning, error and alias parity comes from the separate frozen 50-case suite. This report makes no per-stage causal claim, prior-C++ improvement claim or library-wide speed claim.

The producer passed 26 focused normal/scalar XML pairs (5,872 checks), color regression 3,696, matching 1,612, smoke 1,423 across 242 registered functions, plus Blur/Polygon/Euler guards (4 each) and nine integer-width guards. Source census is 239 image APIs with partial native coverage plus three metadata APIs. Full four-profile qualification remains **203 image APIs**; this minimum-host comparison grants no additional qualification.

## Provenance

- [Genuine run 40685511797](https://github.com/bipxx/scikit-imagexx/actions/runs/40685511797), isolated `measure-minimum` after successful `numpy-minimum`; 21 older timing steps skipped.
- [Retained artifact 12814229633](https://github.com/bipxx/scikit-imagexx/actions/runs/40685511797/artifacts/12814229633): `ci-performance-minimum`, 84,777,605 bytes, 78 members, downloaded once; every member CRC/SHA checked.
- Host: Linux 6.17.0-1022-azure x86_64, glibc 2.39; CPU model was not recorded. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0; AVX2 enabled. Report timestamp: 2026-10-28T12:17:58.367206+00:00.
- Exact source, core, SDK, product wheel, 13 runtime dependencies, original Python/Cython implementations and before/after runtime identities authenticated. Source manifest includes 1,717 committed blobs; product wheel has 157 members, SDK 71.

| Proof | SHA-256 |
|---|---|
| Raw report | `08f1b419efa26a6e6e92d7161a310e5498f4f352ffc0731f3731eec4c8678a3a` |
| Artifact ZIP | `b006fa18d41fdb350d0e754829b13bad9a1fe09687df84f807e4a1ff72e5ee8d` |
| Harness | `a6f25c1e7fe60ab9a86cb38406a2a49054a052d1b0e29dc555545a1cb021667b` |
| Prospective workload contract | `360e96314eb6e4ddbbab566acd23790ab8e8dfe38e7a1525242132f980dc17b8` |
| Original Python source | `aaca3e8ec753ab84d793a39e7aa71db991d5d39dd457cba96d11b4aba1f2cfe2` |
| Original compiled Cython primitive | `ec8b4ba7dc6b98f9e248b297ef194aae22396d96d145f9417dbe32858b22885e` |
| Installed native core | `cbbbba06720563d586f1acee6aa8beda93dc41053d05b99c1490701b7923dea0` |
| Canonical product source fingerprint | `39ea1323c45bdf9ceafe3cb48656d4c15718c342e81c40ebff1f9c5a5d8e3018` |
| Independent review | `457370fc23b0f667fe62851426b904b703c408f8f00a7e13da2d2c4754ecd08b` |
