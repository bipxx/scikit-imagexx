# Montage sustained timing: unresolved closure

**UNRESOLVED_MEASUREMENT — no source merge.** All fixed performance gates pass, but the unchanged per-metric null/worst-pair rules block acceptance. All 1,008 accepted samples, 144 warmup function groups, 72 returned records and complete calibration/batch journals are preserved in [the JSON report](montage-sustained-same-host.json).

[Run 40795200735](https://github.com/bipxx/scikit-imagexx/actions/runs/40795200735) measured unchanged baseline `4f69eba` and private byte-fill candidate `b5bd1df` in eight independent A B B A B A A B processes, CPU 0 on AMD EPYC 9V45 96-Core Processor, with one worker/reference pool. Every accepted batch was at least 100 ms (observed 100.160551–227.480228 ms), with at least 250 ms warmup/calibration per function. All actual loop counts, elapsed nanoseconds, rejected underlength probes and alternating function order passed frozen independent and root reviews. The original deadline and all strict gates were unchanged.

| Case | Direct A CPP / B CPP | Normalized B / A | Worst direct | Worst normalized | Candidate/original GM |
|---|---:|---:|---:|---:|---:|
| gray-small-u8 | 1.010790936× | 0.977474912× | 0.948525163× | 0.962926267× | 1.722254023× |
| gray-incomplete-u8 | 1.199923182× | 1.128340977× | 1.131450475× | 1.085803619× | 1.602376482× |
| gray-dense-f32 | 1.177026773× | 1.251300446× | 1.091406683× | 1.154470329× | 1.864401646× |
| gray-large-u16 | 1.312501234× | 1.330304119× | 1.205348599× | 1.278967934× | 1.694849980× |
| gray-mean-f64 | 1.218936810× | 1.229049350× | 1.158034732× | 1.200195830× | 1.322568174× |
| rgb-padded-u8 | 1.008550340× | 1.016224359× | 0.860400932× | 0.891322399× | 7.243503160× |
| rgb-first-f32 | 1.253398103× | 1.244538190× | 1.149901636× | 1.187630728× | 3.299331036× |
| gray-strided-reverse-f32 | 14.269883322× | 14.312237851× | 12.546149711× | 12.354722318× | 2.472812585× |
| gray-fortran-f64 | 1.278742188× | 1.251380795× | 1.140222852× | 0.695140488× | 1.231052060× |

Balanced direct GM is 1.553448629119×; normalized GM is 1.548231342462× for these nine workloads. Small uint8 retains a balanced normalized loss of 0.977474912×, within the unchanged fixed 0.97 floor. Aggregate gains do not override the failed uncertainty checks.

The unresolved metrics are small-uint8 **direct** (worst 0.948525163×), padded-RGB **direct and normalized** (0.860400932× / 0.891322399×), and Fortran float64 **normalized** (0.695140488×). Fortran direct worst gain 1.140222852× passes despite direct null variation. Its B3 original/current loss **0.980975710×** is retained. All ten fixed gates, six preserved original-win cohorts, target repetitions, identities, outputs/oracle hashes, layouts, caller-state checks and deadline pass.

| Case | Null pair | CPP earlier/later | Original earlier/later | Normalized later/earlier |
|---|---|---:|---:|---:|
| gray-small-u8 | B2/B3 | 0.980867761× | 0.993578435× | 0.987207176× |
| gray-small-u8 | A6/A7 | 1.037715072× | 1.017788117× | 1.019578686× |
| gray-incomplete-u8 | B2/B3 | 0.974032153× | 1.010023496× | 0.964365836× |
| gray-incomplete-u8 | A6/A7 | 1.002357402× | 1.014258832× | 0.988265884× |
| gray-dense-f32 | B2/B3 | 1.031760735× | 0.953598720× | 1.081965310× |
| gray-dense-f32 | A6/A7 | 0.996533802× | 1.017915773× | 0.978994361× |
| gray-large-u16 | B2/B3 | 0.965721078× | 0.963031849× | 1.002792461× |
| gray-large-u16 | A6/A7 | 1.004957529× | 1.012380282× | 0.992668019× |
| gray-mean-f64 | B2/B3 | 0.975435966× | 0.977979827× | 0.997398861× |
| gray-mean-f64 | A6/A7 | 0.999093812× | 1.003190380× | 0.995916460× |
| rgb-padded-u8 | B2/B3 | 0.954829795× | 0.957411479× | 0.997303474× |
| rgb-padded-u8 | A6/A7 | 0.908756303× | 0.994305271× | 0.913961064× |
| rgb-first-f32 | B2/B3 | 0.938353137× | 0.983932868× | 0.953675975× |
| rgb-first-f32 | A6/A7 | 1.070792893× | 1.000128297× | 1.070655532× |
| gray-strided-reverse-f32 | B2/B3 | 0.952301441× | 1.008928177× | 0.943874363× |
| gray-strided-reverse-f32 | A6/A7 | 0.982649605× | 0.998695209× | 0.983933432× |
| gray-fortran-f64 | B2/B3 | 1.124709088× | 1.560776524× | 0.720608666× |
| gray-fortran-f64 | A6/A7 | 0.892469528× | 1.389098170× | 0.642481249× |

Longer validated batches did not eliminate normalized-timing uncertainty on this host. Both C++ and original timings vary; these data do not identify a CPU, cache, allocator, scheduler or reference-only cause. No additional Montage campaign, variant, gate waiver or retry follows this closure. The earlier [row-owned v1](montage-row-owned-same-host.md) and [byte-fill v2](montage-row-owned-v2-same-host.md) outcomes remain separately closed UNRESOLVED_MEASUREMENT, with their original data intact.

The explicit sustained driver retains the frozen fixtures/oracle and full public allocation/destruction boundary. NumPy mean/rescale/axis/type preparation and unsupported provider boundaries remain explicit. Internal validation checks full output words and NumPy caller state; exported evidence is hashes/layouts, not complete output buffers or separate FP-register values. Source B stays private, and full four-profile qualification remains 241 APIs.

Result artifact `12863532173`, ZIP SHA `aeca5626a0202c7955d621eb20cb1e10521cf4aabcb6939dca9a19e95ff6864b`. Independent review SHA `2d49ad6019aaf8726ffa3fb5883593a53af6f23ff7026f07abec2f2d73a437e2`; batch-protocol review SHA `45648184dada00e3eeb812b4b182636201c03a0b4af48b1bf5add259c4efa0a7`; root recomputation SHA `38e2f30e85371607c9894a9c871f9208d38d5504266d5481ef53331f222d52c3`. All eight raw report objects and reviews retain original numeric text and terminal line endings; the complete original result ZIP is embedded byte-exactly as base64 with every member hash.
