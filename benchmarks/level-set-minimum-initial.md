# Level-set generators: first one-thread comparison

All **8/8 cases beat scikit-image 0.26.0**, with **22.845615241x geometric-mean speedup** and no losses. Checkerboard gains average **59.673227301x**; disk gains average **8.746336663x**. Individual gains range from **6.516041121x to 219.969151116x**. These are whole public-call original/current results on one minimum Linux host; they do not establish a prior-C++ optimization gain or four-profile qualification.

The [unchanged raw report](level-set-minimum-initial.json) retains all **112 samples**: eight frozen default workloads, seven paired rounds per implementation, **one global worker**, and all four reference-library pools limited to one thread. Shape tuples are inputs; there are no uint8/float32/float64 input-image cohorts. All outputs are int8.

| API | Shape | Original ms | Current ms | Original/current | Added ms |
|---|---|---:|---:|---:|---:|
| checkerboard_level_set | 256 x 256 | 0.275587655 | 0.003472222 | 79.369252559x | -0.272115433 |
| checkerboard_level_set | 1024 x 1024 | 4.548570500 | 0.020678220 | 219.969151116x | -4.527892280 |
| checkerboard_level_set | 64 x 64 x 64 | 1.520544636 | 0.040752571 | 37.311624761x | -1.479792065 |
| checkerboard_level_set | 128 x 64 x 32 | 1.563628909 | 0.080329269 | 19.465245035x | -1.483299640 |
| disk_level_set | 256 x 256 | 0.346007750 | 0.045518368 | 7.601497337x | -0.300489382 |
| disk_level_set | 1024 x 1024 | 10.121899000 | 0.663942577 | 15.245142203x | -9.457956423 |
| disk_level_set | 64 x 64 x 64 | 1.639071000 | 0.211498577 | 7.749796825x | -1.427572423 |
| disk_level_set | 128 x 64 x 32 | 1.678167100 | 0.257543970 | 6.516041121x | -1.420623130 |

Each latency is the median of seven raw samples. Added milliseconds are current minus original latency; negative values indicate lower latency. Every assessed case is included. There are **no slower cases** in this cohort.

Calls use default arguments: checkerboard square size 5; disk integer center at each axis midpoint and radius `min(shape)*3/8`. Public default evaluation, argument normalization, admission, allocation, every output pixel and output destruction remain timed through the existing paired benchmark primitive. Native checkerboard generation uses scalar/SIMD row patterns; native disk generation preserves square-distance, square root, radius subtraction and strict comparison order. The measurement covers the complete implementation and does not isolate individual-stage or SIMD-only gains.

Before timers, fresh original results are authenticated against the existing original-first frozen metadata and SHA256. The harness then compares complete native/reference output bytes, dtype, shape, strides, ownership, writeability, contiguity and base kind, requires the actual C return and accepted native result, and forbids forwarding to the original whole API. Passive hooks are restored before timing. JSON exports hashes rather than full image byte payloads. Floating centers, protocol inputs, exceptional arithmetic and other source-defined provider partitions remain outside the timed cohort; their frozen parity checks remain separate.

Preparation passed **5,460 focused checks in 24 normal/scalar XML pairs**, including both 161-case public and 163-case pipeline level-set suites, plus the 51-case Profile-coordinate suites. It also passed **3,696 color regressions**, **1,612 full matching checks**, and **1,421 smoke checks for 241 registered functions** (238 image APIs plus three metadata APIs). Separate four Blur/four Polygon/four Euler/nine integer-width guards passed. The 20 older timing steps were skipped; only this eight-case benchmark ran in the isolated measurement job.

The sole retained minimum artifact authenticates **1,696 committed source blobs**, the complete **156-member product wheel**, **71 SDK members**, the sole installed core, **13 dependency wheels**, and identical official source/runtime/package/SDK checks before and after measurement. No producer wheel artifact was downloaded again. Retained private correctness evidence includes six native modes (7,571 checks per normal/scalar/sanitizer mode, 3,795 per portable mode), 80 exact frozen reference records, workers 1/4, and the separate frontend/pipeline checks. These are scoped evidence; full supported-profile qualification remains **203 image APIs**, and qualification of all 238 current image APIs is pending.

[Genuine Release run 40673515438](https://github.com/bipxx/scikit-imagexx/actions/runs/40673515438) measured exact source [`0a24ac663257815c77f2d488b5a281b646c918bf`](https://github.com/bipxx/scikit-imagexx/commit/0a24ac663257815c77f2d488b5a281b646c918bf), CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, and Linux x86_64. The report timestamp is **2026-10-28 10:47:53.656349 UTC**. CPU architecture is x86_64; the CPU model was not recorded, so no specific model is claimed.

[Retained artifact 12810250603](https://github.com/bipxx/scikit-imagexx/actions/runs/40673515438/artifacts/12810250603): **84,759,228 bytes**, **74 members**, SHA256 `4fcb261fe91ae9ecc514196774be51d1155ec90d07da0380ec319e04cdc6dee2`. Every member CRC, size and SHA was checked. Raw report SHA256 `1b2c29c206d745d2d3c6f2bca2a06f7305164730cb50013ea678a2dee2415f73`. Independent review SHA256 `88b91a92364516ec9ab88e6b755ecd6b16ed7fa04d96a734cebfc831f28ee285`. Harness SHA256 `1b5bc41d9b43783372ce7522d5cda85f0352ce483cb4e848fa312794f4f04ed4`. Frozen contract raw SHA256 `8fb0b31cf701c96c75730d618adebcae37118f243972901a570ce26efdf5a706`. Installed core SHA256 `5dee40531bb5c080da6ae68aebf090f7c320aa3638f91e819053e2937fe35994`.

The retained audit includes two corrected reviewer bookkeeping assertions: the contract raw SHA uses its committed CRLF bytes, while source fingerprints use canonical LF; each new API has three smoke nodes (surface, signature and one shape-carrier case). Original failed review receipts remain retained. Neither correction changes product source, frozen expectations, samples, speed metrics or the required 1,421-node smoke total.
