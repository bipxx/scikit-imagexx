# MARK/LBP minimum-host initial comparison

Exact product source `16a5b50193125199ad26cf5a89acec5babf5290c`, controller `17a35a313877409c6c1e719b70a9c4d977b533db`, [run43727894422](https://github.com/bipxx/scikit-imagexx/actions/runs/43727894422). One Intel Xeon6975P-C Linux host, Python3.11.17, NumPy1.24.4, original scikit-image0.26.0. This is candidate evidence; main still ships271 partial native image paths.

One worker, allocation-inclusive public calls, seven samples per arm,100ms target and250ms warmup per arm:12cases,168accepted samples,24warmup groups (88actual warmup batches). All3wins and9losses are retained. Speedup is original median/current median; below1 is slower.

| API family / case | Original ms | Current ms | Speedup | Result |
|---|---:|---:|---:|---|
| mark / gray-u8-outer | 12.267197 | 38.155180 | 0.3215x | loss |
| mark / RGB-f32-thick | 7.505081 | 17.301998 | 0.4338x | loss |
| mark / gray-u16-F-inner | 8.402320 | 18.230412 | 0.4609x | loss |
| mark / RGB-f64-reverse-outline | 17.347328 | 53.867967 | 0.3220x | loss |
| mark / gray-f64-subpixel | 59.808517 | 8.553003 | 6.9927x | win |
| mark / RGB-f32-subpixel-outline | 60.371943 | 9.192905 | 6.5672x | win |
| mark / singleton-subpixel-version | 0.093236 | 0.011935 | 7.8119x | win |
| mark / half-promotion-provider | 11.842772 | 37.863672 | 0.3128x | loss |
| lbp / gray-u8-C | 0.211456 | 3.876004 | 0.0546x | loss |
| lbp / RGB-f32-F | 0.052467 | 2.045324 | 0.0257x | loss |
| lbp / RGB-f64-reverse-wrap | 0.115595 | 2.254755 | 0.0513x | loss |
| lbp / RGB-f64-C-large | 0.752779 | 36.468916 | 0.0206x | loss |

The three MARK subpixel cases are6.57–7.81x faster here. The other five MARK cases are0.313–0.461x as fast; allfour LBP cases are0.0206–0.0546x as fast (about18–48x slower). These results establish no library-wide speed superiority.

Both real first cohorts recorded one original/native call, source-enforced strict FP/caller32/C1 restoration and zero sampler/timing. Before/after identity checks agree with the source/runtime/core/SDK108 binding. Reports retain target control/status/MXCSR words; per-case raw32-byte payloads were not emitted. Helper32positive/6negative full32 controls are separate evidence. Untimed12pairs and sampler callback totals are source-derived/reconstructed, not direct profiler counters.

NumPy dtype/storage metadata and real palette transfer/allocation/copy are included; the cubic recurrence remains ordered serial. The historical half-promotion-provider case name does not imply that current half decoding is delegated. Raw report provider notes and native-return markers are retained. CPU flags do not identify the kernel chosen for each call.

Historical DDA64-case results retain their own source/runtime/host labels and are not rebound or combined into an80-case single-host claim. VI4 remains held. The prior controller containment failure43721720516 is preserved; it stopped before measurement by source-order evidence. The successor changed only output paths/authority and source controls; no completed timing experiment was repeated.
