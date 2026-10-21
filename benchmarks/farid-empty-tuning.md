# Fixed empty-axis Farid experiment

Nine fixed calls: uint8/float32/float64, 512x512 contiguous/Fortran/reversed arrays, empty axes, one worker, eleven warmed allocation-inclusive samples. The metric is the geometric mean native median latency; lower is better. The recipe is [farid_empty_axes_tuning.py](farid_empty_axes_tuning.py).

| Source | Native geometric mean ms | Baseline / native | Faster than reference |
|---|---:|---:|---:|
| baseline `b6bb029` | 0.115560 | 1.00x | 5/9 |
| owned-zero-allocation `c0bc3f3` | 0.043790 | 2.64x | 4/9 |
| short-dispatch `3beea61` | 0.041399 | 2.79x | 8/9 |
| confirmation `3beea61` | 0.047133 | 2.45x | 6/9 |

The optimized path allocates a NumPy-owned zero buffer and avoids copying or reading an image when no derivative axes are requested. No convolution or image arithmetic is needed for that result. Mask erosion and multiplication still execute when requested. Input subclasses and unsupported metadata retain explicit compatibility dispatch. Ownership, dtype, layout, signed zeros, masks, endian storage and unused boundary options are parity-checked.

owned-zero-allocation: reference geometric-mean drift -1.2%; paired reference/native speedup improvement 2.61x.
short-dispatch: reference geometric-mean drift +1.2%; paired reference/native speedup improvement 2.82x.
confirmation: reference geometric-mean drift +7.1%; paired reference/native speedup improvement 2.62x.

The small incremental dispatch gain is not established as stable: confirmation is slower than the first optimized observations, while retaining a substantial improvement over the initial implementation. Allocation state and ordinary system load are uncontrolled; no affinity, frequency locking or hardware-counter profiling was used. Every raw sample, including slower cases, is retained in [JSON](farid-empty-tuning.json).

The [initial 243-case profile](farid-initial.md) retains its separate source fingerprint and all observations (241 faster, two empty-axis cases slower). Historical tuning/profile files are excluded from the [current-source aggregate](STATUS.md).
