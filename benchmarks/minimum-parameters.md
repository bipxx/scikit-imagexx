# Minimum threshold performance

Original scikit-image 0.26.0. Reference pools use one thread. Native calls run with explicit worker limits and fallback disabled. Exact output type, value, warnings and convergence exceptions are checked before timing. Every workload and raw sample is retained, including slowdowns and failed convergence.

| Dtype | Workload | Outcome | Workers | Original ms | C++ ms | Speedup | Preserved/current |
|---|---|---|---:|---:|---:|---:|---:|
| uint8 | bimodal | value | 1 | 1.9459 | 0.0822 | 23.68x | — |
| uint8 | fortran | value | 1 | 2.1879 | 0.1453 | 15.06x | — |
| uint8 | reversed | value | 1 | 2.0488 | 0.1467 | 13.96x | — |
| uint8 | stepped | value | 1 | 0.4239 | 0.0417 | 10.15x | — |
| uint8 | broadcast | value | 1 | 0.4293 | 0.0762 | 5.63x | — |
| uint8 | seven-bins | value | 1 | 1.9755 | 0.0884 | 22.36x | — |
| uint8 | 513-bins | value | 1 | 2.0561 | 0.0829 | 24.81x | — |
| uint8 | constant-failed | error | 1 | 0.7457 | 0.1373 | 5.43x | — |
| uint8 | iteration-limit | error | 1 | 0.4822 | 0.0981 | 4.92x | — |
| float32 | bimodal | value | 1 | 4.3932 | 0.6746 | 6.51x | — |
| float32 | fortran | value | 1 | 6.6693 | 1.2160 | 5.48x | — |
| float32 | reversed | value | 1 | 5.2129 | 0.7901 | 6.60x | — |
| float32 | stepped | value | 1 | 31.0886 | 0.5487 | 56.65x | — |
| float32 | broadcast | value | 1 | 31.2052 | 1.0165 | 30.70x | — |
| float32 | seven-bins | error | 1 | 1.8858 | 1.1511 | 1.64x | — |
| float32 | 513-bins | value | 1 | 19.4008 | 0.8078 | 24.02x | — |
| float32 | constant-failed | error | 1 | 1.7214 | 0.5833 | 2.95x | — |
| float32 | iteration-limit | error | 1 | 1.8288 | 0.7010 | 2.61x | — |
| float64 | bimodal | value | 1 | 6.5527 | 0.6539 | 10.02x | — |
| float64 | fortran | value | 1 | 6.1535 | 1.5080 | 4.08x | — |
| float64 | reversed | value | 1 | 5.7172 | 0.9335 | 6.12x | — |
| float64 | stepped | value | 1 | 12.3616 | 0.3839 | 32.20x | — |
| float64 | broadcast | value | 1 | 26.8318 | 1.0102 | 26.56x | — |
| float64 | seven-bins | error | 1 | 2.4087 | 1.1982 | 2.01x | — |
| float64 | 513-bins | value | 1 | 22.9043 | 0.8866 | 25.83x | — |
| float64 | constant-failed | error | 1 | 3.4928 | 0.7726 | 4.52x | — |
| float64 | iteration-limit | error | 1 | 3.8528 | 0.6996 | 5.51x | — |
| float32/int64 | bimodal-histogram | value | 1 | 5.1098 | 0.0618 | 82.72x | — |
| float32/int64 | random-histogram | value | 1 | 47.9687 | 0.4077 | 117.67x | — |
| float32/int64 | trimmed-histogram | value | 1 | 5.4527 | 0.0558 | 97.63x | — |
| float32/int64 | large-histogram | value | 1 | 22.3727 | 0.1838 | 121.70x | — |
| float32/float32 | bimodal-histogram | value | 1 | 5.8822 | 0.0613 | 96.01x | — |
| float32/float32 | random-histogram | value | 1 | 17.2155 | 0.1523 | 113.06x | — |
| float32/float32 | trimmed-histogram | value | 1 | 6.2881 | 0.0586 | 107.39x | — |
| float32/float32 | large-histogram | value | 1 | 25.3554 | 0.2600 | 97.52x | — |
| float32/float64 | bimodal-histogram | value | 1 | 4.8507 | 0.0486 | 99.71x | — |
| float32/float64 | random-histogram | value | 1 | 16.7623 | 0.1677 | 99.95x | — |
| float32/float64 | trimmed-histogram | value | 1 | 14.3423 | 0.1166 | 122.98x | — |
| float32/float64 | large-histogram | value | 1 | 30.5186 | 0.2700 | 113.04x | — |
