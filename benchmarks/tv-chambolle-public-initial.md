TV-Chambolle has a verified mixed one-worker baseline against scikit-image 0.26.0. The geometric mean is faster on the recorded Windows host and slower on both recorded Linux hosts. This does not meet the goal of improving every operation.

| Runtime | Geometric mean speedup | Faster / 70 | Slower / 70 | Recorded CPU |
|---|---:|---:|---:|---|
| Windows | 1.1437x | 36 | 34 | AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD |
| Modern Linux | 0.8893x | 32 | 38 | AMD EPYC 7763 64-Core Processor |
| Minimum NumPy Linux | 0.8179x | 34 | 36 | AMD EPYC 9V45 96-Core Processor |

Speedup is original median time divided by C++ median time; values below 1 are slower. Each host ran the same 70 fixed finite whole-call cases with one native worker and seven paired rounds. Original and C++ callbacks ran in the same process, in shuffled order, with independently calibrated loop counts. Conversion, allocation and result destruction are included; fixture construction is excluded. These are observations on the recorded hosts, not a before/after patch experiment.

The [full-precision CSV](tv-chambolle-public-initial.csv) contains every one of the 210 rows and all 2,940 sample values. The [raw archive](tv-chambolle-public-initial-raw.zip) preserves the exact original compressed reports, including all 400 untimed validation observations per measurement profile, and the frozen controller sources. All 34/38/36 slower cases are retained. The archive records its pre-promotion export state; final measurement acceptance is recorded in the separate evidence below.

The measured native product is `f87120cda4e6ca9246c30f837479babe0bce0b92`, qualified in [run 40087742726](https://github.com/bipxx/scikit-imagexx/actions/runs/40087742726): 191 partial image paths, 108 wheel members, and the complete 62-member SDK across 17 families. Controller `20b78d5ff4c6182d64c57b76cc262ed62ac0294d` changes exactly the four reviewed controller paths while preserving all native source. These are not timings of the later Bregman 192-path product.

All six jobs in [measurement run 40103617685](https://github.com/bipxx/scikit-imagexx/actions/runs/40103617685) passed. Three fresh preparations checked normal/scalar execution at workers 1/4, all 400 fixed observations and the installed public pipeline. Each fresh measurement job consumed the completed three-profile barrier and immutable qualified product; it performed no build or pytest run. Full source, runtime, core, wheel RECORD, SDK, capsule and receipt closure were independently verified, followed by a separate root rehash and sample arithmetic review. See [measurement acceptance](../coverage/evidence/tv-chambolle-public-performance.json) and the [public parity scope](../coverage/evidence/public-wave12.json).

The earlier serialization and fresh-process policy failures remain retained as unaccepted runs 40094239260 and 40097514733; neither produced timings. No scientific observations or workload specifications were rewritten to repair those controller failures.

All 108 regressions were classified. Source and accepted-object inspection identify repeated integer division for noncontiguous input addressing as a concrete optimization candidate affecting 55 of those rows. Existing packed AVX2 instructions are present. A logical-stride cursor is being validated without changing arithmetic, validation order, providers or the fixed corpus; its speed benefit remains unmeasured. Contiguous fixed costs and other regressions remain open.
