# N-D line: initial one-thread comparison

Prospective descriptive allocation-inclusive actual public comparison only. No timing or performance acceptance yet; preserve every loss. Source-safe bounded vector cases, not whole API/domain coverage.

| Case | Original ms | C++ ms | Original/current |
|---|---:|---:|---:|
| line-nd/empty | 0.011018235 | 0.002501674 | 4.404344082x |
| line-nd/unit | 0.012681904 | 0.003265507 | 3.883594710x |
| line-nd/small-descending | 0.014104193 | 0.003637161 | 3.877802884x |
| line-nd/long-integer-2d | 0.348743308 | 0.165693409 | 2.104750642x |
| line-nd/floating-3d | 0.027972042 | 0.013210730 | 2.117372885x |
| line-nd/integer-8d | 0.191946540 | 0.166386331 | 1.153619642x |
| line-nd/float16 | 0.010535425 | 0.003503715 | 3.006929475x |
| line-nd/float32 | 0.074532731 | 0.039917884 | 1.867151363x |
| line-nd/float64 | 0.282888596 | 0.106386023 | 2.659076712x |
| line-nd/reverse-3d | 0.200562392 | 0.123307128 | 1.626527161x |
| line-nd/readonly-unaligned | 0.095023655 | 0.042716139 | 2.224537532x |
| line-nd/int8-wrap | 0.012206945 | 0.003011525 | 4.053410205x |
