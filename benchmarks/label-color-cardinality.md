# Label-color performance

Original scikit-image 0.26.0. Reference pools use one thread; native worker limits are explicit. Fallback is disabled. Values, dtype, metadata and warnings are checked before timing; RGB desaturation uses the documented HSV error budget. Every raw sample and slower workload is retained.

| Input | Workload | Workers | Original ms | Native ms | Original/native | Preserved/native |
|---|---|---:|---:|---:|---:|---:|
| uint64 | sparse-groups-31 | 1 | 0.5086 | 0.2107 | 2.4139x | 0.9717x |
| uint64 | sparse-groups-512 | 1 | 0.9697 | 0.4022 | 2.4109x | 0.9732x |
| uint64 | sparse-groups-513 | 1 | 0.9476 | 0.2421 | 3.9138x | 5.8296x |
| uint64 | sparse-groups-4096 | 1 | 2.3205 | 0.6308 | 3.6787x | 2.9718x |
| uint64 | sparse-groups-17157 | 1 | 4.3541 | 2.1835 | 1.9941x | 0.9659x |
| int64 | sparse-groups-31 | 1 | 0.5438 | 0.2122 | 2.5621x | 0.9577x |
| int64 | sparse-groups-512 | 1 | 1.0092 | 0.4439 | 2.2735x | 0.8925x |
| int64 | sparse-groups-513 | 1 | 1.0778 | 0.2644 | 4.0769x | 5.7719x |
| int64 | sparse-groups-4096 | 1 | 2.4338 | 0.6984 | 3.4847x | 2.8428x |
| int64 | sparse-groups-17157 | 1 | 4.5954 | 2.3240 | 1.9774x | 0.9543x |
| float64 | sparse-groups-31 | 1 | 0.5820 | 0.2376 | 2.4491x | 0.9879x |
| float64 | sparse-groups-512 | 1 | 1.0763 | 0.4183 | 2.5728x | 1.0948x |
| float64 | sparse-groups-513 | 1 | 1.0019 | 0.2923 | 3.4274x | 5.2582x |
| float64 | sparse-groups-4096 | 1 | 2.0821 | 0.5765 | 3.6118x | 3.1862x |
| float64 | sparse-groups-17157 | 1 | 4.1005 | 2.1028 | 1.9500x | 0.9607x |
