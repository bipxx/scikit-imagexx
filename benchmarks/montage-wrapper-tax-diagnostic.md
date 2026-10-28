# Montage: same-core wrapper diagnostic

**Diagnostic only; no KEEP/DISCARD decision.** The three controls that failed the closed wrapper experiment have no positive median wrapper-route penalty on this diagnostic host. Incomplete uint8 is **−160.581 ns**, padded RGB uint8 **−215.364 ns**, and Fortran float64 **−848.600 ns**. The earlier whole-public-call regression cause remains unresolved.

[Complete raw reports, full journals, prospective contract, independent review and preparation/identity evidence](montage-wrapper-tax-diagnostic.json) retain **756 samples**, **21 paired rounds per case** and **1,532 timing-journal events**. Both routes use the same authenticated candidate core, input/fill/output pointers and descriptors, with worker count 1. Positive differences mean the wrapper was slower.

| Case | Wrapper−generic median ns | Range ns | Positive pairs/21 | Generic A−B null median ns | Null range ns |
|---|---:|---:|---:|---:|---:|
| gray-small-u8 | 7.123 | -15.749 to 26.179 | 18 | 2.204 | -18.144 to 18.454 |
| gray-incomplete-u8 | -160.581 | -280.876 to -93.204 | 0 | -18.724 | -145.654 to 71.274 |
| gray-dense-f32 | 179.787 | -4439.715 to 994.312 | 18 | -13.850 | -1821.460 to 695.403 |
| gray-large-u16 | 293.000 | -1594.435 to 2392.175 | 13 | -181.535 | -1558.065 to 799.845 |
| gray-mean-f64 | 2.300 | -2043.730 to 2650.860 | 11 | 87.560 | -1440.790 to 2351.310 |
| rgb-padded-u8 | -215.364 | -710.930 to 137.678 | 2 | -28.494 | -528.570 to 2516.552 |
| rgb-first-f32 | -45.923 | -1023.213 to 1130.887 | 10 | 22.077 | -611.380 to 699.040 |
| gray-strided-reverse-f32 | -462496.580 | -469964.160 to -448949.020 | 0 | -1010.880 | -6460.900 to 6391.760 |
| gray-fortran-f64 | -848.600 | -24876.567 to 11247.367 | 10 | -704.300 | -14927.967 to 5980.533 |

Incomplete uint8 is faster through the wrapper in 21/21 paired rounds; padded RGB in 19/21. Fortran has 10/21 positive differences, with a−704.300 ns null median and a−14,927.967 to 5,980.533 ns null range. Its small median difference does not separate a route gain from null variation. Every individual paired difference and original batch elapsed time remains in the raw evidence.

The eight noneligible controls compare the complete wrapper route against the identical generic function at the same loaded address. This includes branch, stack, PLT and calling-context effects. It does not isolate a single call instruction. The **strided float32 target is eligible for SIMD**: its−462,496.580ns median compares two copying algorithms and is **not a wrapper-tax estimate**.

The timing boundary contains fixed-loop prepared native calls and one common Python adapter crossing/GIL transition per batch. Fresh output allocation, output destruction and NumPy providers are outside this diagnostic. Oracle checks, journal writes and buffer checks are outside clocks. Randomized wrapper/generic and generic/generic null pairs are interleaved using seed 2026; all loop counts and label orders match the prospective contract.

The host was **AMD EPYC 7763 64-Core Processor**, pinned to permitted CPU0, with **CPython3.11.17**, genuine SDK 81/product 172 and 13 exact runtime dependencies. Preparation compiled only one diagnostic adapter TU on a separate job; no builds or tests ran on the timing host. No original scikit-image algorithm call was made. The prepared adapter and both exported functions belong to the same authenticated loaded core.

The closed [wrapper experiment](montage-wrapper-simd-same-host.md) measured allocation-inclusive public calls on AMD EPYC 9V45. Its Fortran failure was normalized-only despite a 1.203409x direct C++ improvement; original reference times drifted materially. This narrower, different-host diagnostic cannot establish the cause of those whole-call observations. The known extra stack/call work exists structurally, but removing it is not demonstrated here to fix the historical regressions. The [constant-copy](montage-strided-copy-same-host.md), [in-body SIMD](montage-stride2-simd-same-host.md) and wrapper DISCARD decisions remain unchanged.

All 1,532 journal events match the final report: each raw batch was saved before output validation, every completed batch was validated, and the complete journal hash matches. All raw JSON documents retain their original number text and terminal newline; JSONL/CPU/log byte members are base64 with SHA256 and lengths, and decode exactly to the original bytes. No giant prebuilt-product bundle is embedded.

- [Diagnostic run 40779297530](https://github.com/bipxx/scikit-imagexx/actions/runs/40779297530), controller `ca446f93313ef85ea249d580229bf3f853e70f8b`.
- [Genuine producer 40766081804](https://github.com/bipxx/scikit-imagexx/actions/runs/40766081804), source `61b4f79821b0c7fd0d224e2a12d4263d14308cb7`; this remains the private closed product, not current-main qualification.
- Raw diagnostic artifact `12855868535`, 37,410 bytes; SHA256 `077b712d37f5e779f2b858cbb8f28b49d8687cbe4d308ef2d817ac66c8aef1fc`.
- Preparation-log artifact `12856392183`, 5,840 bytes; SHA256 `4dd5247183eafeb216971605d528eb76eee9631a2b3e5049dac14dd4df32be5d`.
- Core SHA256 `4ae7f9622f3e702d0aa18e1342a67d0bddd4648d4cd1c046d8c70c039a257dc7`; adapter `67691f8b817dd786f4deafd4a132de022d041ea0af4b3e45efe566bf9759c9fd`.
- Journal SHA256 `6b8e38b0cf6b2acc27c938b3ac836d280af734a14ccea855fb78721302cf0e4b`; independent review `d8bb52ac113c241b7c21b8256f609ba35ee7b749a3d89c8857df427e0c302660`.
- Prospective contract SHA256 `fd88511cb90eab9e034d4fe4ffaa84766da77b99a6e3161141b075eccc9b4d0d`.

Root independently reproduced all 756 elapsed-time divisions, paired orders, 1,532 journal events and nine summaries. Root recomputation SHA256 `494f0d6fb94635bf4dbb2ec127b41020314f324ab6e3cc3fbe1ba3d01f885fa3` is preserved verbatim in the JSON publication.
