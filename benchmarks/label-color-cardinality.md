# Label-color performance

Original scikit-image 0.26.0. Reference pools use one thread; native worker limits are explicit. Fallback is disabled. Values, dtype, metadata and warnings are checked before timing; RGB desaturation uses the documented HSV error budget. Every raw sample and slower workload is retained.

| Input | Workload | Workers | Original ms | Native ms | Original/native | Preserved/native |
|---|---|---:|---:|---:|---:|---:|
| uint64 | sparse-groups-31 | 1 | 0.5401 | 0.2027 | 2.6652x | 3.9574x |
| uint64 | sparse-groups-512 | 1 | 1.0309 | 0.4063 | 2.5370x | 3.5559x |
| uint64 | sparse-groups-513 | 1 | 1.0320 | 1.4701 | 0.7020x | 0.9528x |
| uint64 | sparse-groups-4096 | 1 | 2.4794 | 1.9415 | 1.2771x | 0.9776x |
| uint64 | sparse-groups-17157 | 1 | 4.8547 | 2.0572 | 2.3598x | 1.0211x |
| int64 | sparse-groups-31 | 1 | 0.5716 | 0.2374 | 2.4079x | 3.5220x |
| int64 | sparse-groups-512 | 1 | 0.9865 | 0.4429 | 2.2271x | 3.3448x |
| int64 | sparse-groups-513 | 1 | 1.1568 | 1.5240 | 0.7591x | 0.9984x |
| int64 | sparse-groups-4096 | 1 | 2.2509 | 1.9111 | 1.1778x | 1.0594x |
| int64 | sparse-groups-17157 | 1 | 6.3831 | 2.1525 | 2.9654x | 1.1174x |
| float64 | sparse-groups-31 | 1 | 0.6281 | 0.2599 | 2.4163x | 3.4617x |
| float64 | sparse-groups-512 | 1 | 2.0749 | 0.5795 | 3.5807x | 3.2777x |
| float64 | sparse-groups-513 | 1 | 1.3773 | 1.5410 | 0.8937x | 1.0677x |
| float64 | sparse-groups-4096 | 1 | 2.9995 | 2.2091 | 1.3578x | 0.8916x |
| float64 | sparse-groups-17157 | 1 | 4.4152 | 2.1101 | 2.0925x | 1.0019x |
