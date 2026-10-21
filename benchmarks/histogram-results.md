# Otsu input distributions

2048² uint8 pixels, one worker, AVX2, seven samples. Same process and shuffled timing order; allocation-inclusive Python calls.

| Input | Previous ms | Current ms | Old/current | scikit-image/current |
|---|---:|---:|---:|---:|
| random | 1.1378 | 1.0165 | 1.12x | 14.43x |
| constant | 1.8759 | 0.0768 | 24.43x | 5.01x |
| binary | 1.6825 | 1.2832 | 1.31x | 14.13x |
| skewed | 2.0621 | 1.6446 | 1.25x | 13.39x |
| late_outlier | 2.0529 | 1.8424 | 1.11x | 12.10x |
