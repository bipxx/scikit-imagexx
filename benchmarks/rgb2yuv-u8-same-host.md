# RGB2YUV packed uint8: controlled same-host result

**KEEP.** Fixed12 balanced direct gain is **1.267172459874x**, and reference-normalized gain is **1.267374128531x**. The two uint8 targets gain 3.780183955924x direct / 3.775544364284x normalized; the ten controls gain 1.018357541220x / 1.018802235791x. Both target cases beat both baseline passes in both candidate passes under both metrics, satisfying the frozen keep rule.

Balanced original/current scores from this same ABBA are **4.351957199724x baseline** and **5.515557963404x candidate**. All 12 cases beat original 0.26.0 in both candidate passes. These scores are not multiplied by the earlier cross-host 3.886 result.

Baseline source `e9c723705d2e1bd14b605363032e1bbb36199b77` (Release [40408342309](https://github.com/bipxx/scikit-imagexx/actions/runs/40408342309)) is compared with candidate `12d5e89d3fa3c069ed25fda8c56cbe92920570c6` (preparation [40415348603](https://github.com/bipxx/scikit-imagexx/actions/runs/40415348603)), kept and merged as `277b766`. Controlled workflow [40416508545](https://github.com/bipxx/scikit-imagexx/actions/runs/40416508545) uses controller `cdf5705df50c1b3d338c2b678f596c78222260fd`.

The [byte-exact compact archive](rgb2yuv-u8-same-host.zip) retains four complete reports, all **672 samples**, eight before/after identity logs, input hashes, CPU/platform evidence, offline install logs and authenticated artifact-selection receipts. No wheels or producer ZIPs are duplicated in Git.

| Case | Experiment role (raw group) | dtype.str | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added us | Original/candidate speedup | Repeat win D/N | Repeat loss D/N |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 256:native-u8 | target (control) | `\|u1` | 0.477502090 | 0.135188685 | 3.532115808x | 3.532537653x | -342.313406 | 13.179009414x | 1/1 | 0/0 |
| 256:native-f32 | control (control) | `<f4` | 0.052684897 | 0.051900585 | 1.015111809x | 1.020875827x | -0.784312 | 6.216205004x | 0/0 | 0/0 |
| 256:native-f64 | control (control) | `<f8` | 0.075461497 | 0.077859547 | 0.969200303x | 0.973298857x | +2.398050 | 4.936661664x | 0/0 | 1/1 |
| 256:native-f16 | control (target) | `<f2` | 0.111405800 | 0.109367797 | 1.018634403x | 1.016608773x | -2.038004 | 7.493506245x | 0/0 | 0/0 |
| 256:swapped-f32 | control (target) | `>f4` | 0.083692551 | 0.082160122 | 1.018651748x | 1.017819605x | -1.532430 | 5.677042331x | 1/1 | 0/0 |
| 256:swapped-f64 | control (target) | `>f8` | 0.139929494 | 0.143789305 | 0.973156484x | 0.958461198x | +3.859810 | 3.283145179x | 0/0 | 0/1 |
| 1024:native-u8 | target (control) | `\|u1` | 8.452082988 | 2.089165368 | 4.045674467x | 4.035267744x | -6362.917620 | 4.702100940x | 1/1 | 0/0 |
| 1024:native-f32 | control (control) | `<f4` | 0.794687436 | 0.753157558 | 1.055141022x | 1.068862101x | -41.529878 | 6.539596148x | 1/1 | 0/0 |
| 1024:native-f64 | control (control) | `<f8` | 1.319344556 | 1.341450526 | 0.983520846x | 0.979115305x | +22.105969 | 4.494748139x | 0/0 | 1/1 |
| 1024:native-f16 | control (target) | `<f2` | 2.309748091 | 1.835714820 | 1.258228166x | 1.266239508x | -474.033272 | 7.006832017x | 1/1 | 0/0 |
| 1024:swapped-f32 | control (target) | `>f4` | 1.329714132 | 1.426250816 | 0.932314370x | 0.938819020x | +96.536684 | 5.046396146x | 0/0 | 1/1 |
| 1024:swapped-f64 | control (target) | `>f8` | 2.613477490 | 2.635032409 | 0.991819866x | 0.982714360x | +21.554919 | 2.871331909x | 0/0 | 0/0 |

Each balanced latency is the geometric mean of the two per-pass seven-sample medians. Direct gain is baseline/candidate latency; normalized gain is candidate/reference speedup divided by baseline/reference speedup. Added microseconds are candidate minus baseline: positive values are penalties. The original/current column uses original divided by candidate latency. Repeat wins/losses require both candidate passes to beat/lose to both baseline passes. Raw harness groups remain unchanged: uint8 is an experiment target even though its raw group is `control`.

**Direct losses: 5; normalized losses: 5.** Both lists are: `256:native-f64`, `256:swapped-f64`, `1024:native-f64`, `1024:swapped-f32`, `1024:swapped-f64`. Repeat losses: 3 direct and 4 normalized. All penalties and repeat flags are shown above; unchanged-path variations are not attributed to the uint8 code change.

One fresh Ubuntu minimum-runtime host, **AMD EPYC 9V74**, runs baseline → candidate → candidate → baseline in separate processes, with both offline environments installed before measurement. CPython 3.11.17, NumPy 1.24.4 and SciPy 1.15.3; one global worker, seven warmed paired rounds. Input preparation is outside timing; public conversion/allocation/matrix/status/axis handling remains inside. All four passes use identical input bytes and the unchanged fixed12 harness. Thirteen dependency wheels match byte-for-byte. All eight identities match their authenticated source/wheel/package/core/SDK manifests before and after measurement.

The only production delta is `src/simd_color_avx2.cpp`: bounded four-byte loads convert uint8 RGB packets into packed binary64 values with the same `1/255` multiplication and unchanged multiply/add association. It avoids scalar conversion/stack staging; no FMA or reduction reassociation is introduced. Offset/sRGB, other dtypes, tails, workers and error policies retain their old paths. Native kept/candidate output and FP-flag streams match exactly across normal/scalar and sanitizer controls (25,214 controls / 8,404 calls per run); actual packed conversion instructions and no-FMA codegen were checked.

Two-profile normal/scalar components pass **7,540** storage/color checks with saved raw pairs exact to baseline. Genuine candidate Release passes **2,818 focused public checks**, **3,696 full 14-operation color regressions**, and **1,377 smoke checks** at **229 registered functions**. Preparation-only correctly skips all 11 standalone timing steps; this controlled ABBA is the performance evidence. Full supported-profile qualification remains **203 image APIs**, separate from the current registration count.

NumPy axis/metadata handling and Python floating-policy dispatch remain explicit providers. The thirteen other transform half routes, swapped-integer RGB, subclasses/extended types and print/log policies are outside this new storage scope. Existing tolerances, fixtures and all prior failure records are unchanged. This is one host and twelve fixed workloads, not universal performance or full-profile qualification.

Archive artifact `12696051905`: 29,995 bytes, 36 members, SHA-256 `65e4fddca06373ebef758088e1b30dd552ffc56b30601a835a1b45cfb61e6282`. Every member CRC/size/SHA is verified.

Harness canonical-LF SHA-256: `a621318a573081b070cd99ccb707ceb8f311c1346f70834d5c060a2962afd37b`.

Final review SHA-256: `d0aa00da5e5b4b8ae118470d25caaf48c46c4a1862f791e68fb91eedb935ebd6`. Contract SHA-256: `357c04cf36b8b82fa82a8f3f538cafc641a1ac645fac87fa1e5c1a9771c6481e`.

Baseline wheel SHA-256: `20acf35ff8fcd42f0f43a0679f094de4bb73f2db03e136c55e631c8318c090da`; core: `861391664b7d75e63baee5749aa6464f6ba7eebb47dddc3cf36e4a5231afea02`. Candidate wheel: `6039e402df42dc02a390d1f4921915762f52ae1f10d005e6ff87a31abffc0a9b`; core: `1fafca34613c9c8089ffbd8b0a5d58f386b4e0db23a4126cd773b96c906c3a22`. Baseline product fingerprint: `202d15819e4fafc50879e1b4a1117d0a1e4102e07fa79f2ee3e9cde2adad7c2c`; candidate: `39ae38bbcfbb01f35313e11e5ff36be29ab37be8e4601915136638e2e8390c9e`.
