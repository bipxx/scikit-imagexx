# Montage row-owned v1: closed unresolved measurement

**UNRESOLVED_MEASUREMENT. Candidate `6b22e64641a702c92873c376ff650cd8c8764d0d` remains private and unmerged.** Both fixed `gray-incomplete-u8` control floors fail: **0.862192099× direct / 0.872482075× normalized**. Its balanced C++ time is **15.983% longer** than baseline. The direct null ratios are inside the sealed band, so the direct loss is not dismissed as noise.

All other eight workloads improve, all nine beat original in each candidate pass, and global gains are **1.634644958× direct / 1.637134147× normalized**. None overrides a failed per-case floor. The normalized A6/A7 null is outside the band and worst normalized separation fails; the sealed status remains unresolved. No old gate is changed.

[Complete evidence](montage-row-owned-same-host.json) preserves all **1,008 samples**, **72 returned timing records**, eight raw reports, null/worst-pair facts, contract, source/SDK/runtime proofs, independent/root reviews and the original reviewer failure. The separate prospective v2 byte-fill correction is not part of this result and does not revise it.

| Case | A C++ ms GM | B C++ ms GM | Added ms | Direct | Normalized | Original/B GM | Worst direct | Worst normalized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.007179793 | 0.006622146 | -0.000557648 | 1.084209514× | 1.082736533× | 1.864865852× | 1.039337544× | 1.053685353× |
| gray-incomplete-u8 | 0.021702977 | 0.025171858 | 0.003468881 | 0.862192099× | 0.872482075× | 1.070122749× | 0.834290315× | 0.839174604× |
| gray-dense-f32 | 0.071652463 | 0.049368507 | -0.022283957 | 1.451380001× | 1.443434436× | 1.874683835× | 1.376269840× | 1.408564934× |
| gray-large-u16 | 0.129420353 | 0.080574344 | -0.048846009 | 1.606222858× | 1.582907684× | 1.940859306× | 1.479319007× | 1.486928824× |
| gray-mean-f64 | 0.267389925 | 0.180943362 | -0.086446563 | 1.477754819× | 1.501581160× | 1.395949923× | 1.419177134× | 1.476619669× |
| rgb-padded-u8 | 0.043312811 | 0.031106811 | -0.012206001 | 1.392389975× | 1.388543016× | 6.110388696× | 1.264092704× | 1.267606710× |
| rgb-first-f32 | 0.085393821 | 0.056104863 | -0.029288958 | 1.522039565× | 1.512392864× | 2.837964266× | 1.450585493× | 1.474911520× |
| gray-strided-reverse-f32 | 0.583751809 | 0.056819513 | -0.526932296 | 10.273791061× | 10.382260774× | 1.837199756× | 9.278740398× | 9.722034645× |
| gray-fortran-f64 | 0.926790508 | 0.779820693 | -0.146969815 | 1.188466165× | 1.195533495× | 1.079326911× | 1.162696143× | 1.143894660× |

Added time is candidate minus baseline. Direct and normalized baseline C++ losses contain `gray-incomplete-u8`; every other balanced result improves. All baseline-pass original losses and all per-pass original/CPP ratios are retained below and in the raw data.

| Fixed gate | Result |
|---|---|
| overall_direct | PASS |
| overall_normalized | PASS |
| target_each_candidate_direct | PASS |
| target_each_candidate_normalized | PASS |
| target_each_candidate_original | PASS |
| controls_balanced_direct | FAIL |
| controls_balanced_normalized | FAIL |
| preserved_original_each_candidate | PASS |
| rgb_first_each_candidate_direct | PASS |
| rgb_first_each_candidate_normalized | PASS |

`gray-incomplete-u8` direct B2/B3 and A6/A7 null ratios are **0.990084294 / 0.972147471**, both inside **[0.97, 1/0.97]**. Its direct worst A/B gain is **0.834290315×**. Normalized nulls are **0.977463931 / 0.959436620**; the latter is outside the band. Worst normalized gain **0.839174604×** fails the required 0.97 separation. The measured direct regression remains explicit despite the normalized variability classification.

| Case | Null pair | Direct earlier/later | Reference earlier/later | Normalized later/earlier |
|---|---|---:|---:|---:|
| gray-small-u8 | B2/B3 | 0.940317683 | 0.971871845 | 0.967532589 |
| gray-small-u8 | A6/A7 | 1.027984927 | 1.018346441 | 1.009464840 |
| gray-incomplete-u8 | B2/B3 | 0.990084294 | 1.012911334 | 0.977463931 |
| gray-incomplete-u8 | A6/A7 | 0.972147471 | 1.013248244 | 0.959436620 |
| gray-dense-f32 | B2/B3 | 0.988177869 | 1.010291198 | 0.978111926 |
| gray-dense-f32 | A6/A7 | 1.020287237 | 1.010458969 | 1.009726538 |
| gray-large-u16 | B2/B3 | 0.963780075 | 0.958869204 | 1.005121522 |
| gray-large-u16 | A6/A7 | 1.010567641 | 0.998082298 | 1.012509333 |
| gray-mean-f64 | B2/B3 | 0.966113671 | 0.952729747 | 1.014047976 |
| gray-mean-f64 | A6/A7 | 1.014689282 | 1.023938252 | 0.990967259 |
| rgb-padded-u8 | B2/B3 | 1.057503296 | 1.001158190 | 1.056279923 |
| rgb-padded-u8 | A6/A7 | 1.114822527 | 1.020155517 | 1.092796646 |
| rgb-first-f32 | B2/B3 | 0.979436744 | 0.984651152 | 0.994704309 |
| rgb-first-f32 | A6/A7 | 1.031219038 | 1.014171975 | 1.016808848 |
| gray-strided-reverse-f32 | B2/B3 | 0.930093199 | 1.015149510 | 0.916213021 |
| gray-strided-reverse-f32 | A6/A7 | 0.996368190 | 0.994386439 | 1.001992939 |
| gray-fortran-f64 | B2/B3 | 1.027631308 | 0.978342791 | 1.050379599 |
| gray-fortran-f64 | A6/A7 | 1.016613679 | 1.031429268 | 0.985635865 |

| Case | A1 original/C++ ms | B2 original/C++ ms | B3 original/C++ ms | A4 original/C++ ms | B5 original/C++ ms | A6 original/C++ ms | A7 original/C++ ms | B8 original/C++ ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gray-small-u8 | 0.012323528/0.007092727 | 0.012142228/0.006416988 | 0.012493651/0.006824277 | 0.012171764/0.007107807 | 0.012010720/0.006462389 | 0.012600187/0.007361102 | 0.012373183/0.007160710 | 0.012765227/0.006795384 |
| gray-incomplete-u8 | 0.026536913/0.021565639 | 0.026772575/0.024793299 | 0.026431312/0.025041604 | 0.026195722/0.021618964 | 0.026894073/0.025082784 | 0.027052683/0.021508301 | 0.026698969/0.022124525 | 0.027664960/0.025780356 |
| gray-dense-f32 | 0.092878023/0.072390780 | 0.092346326/0.048462196 | 0.091405653/0.049041977 | 0.092644132/0.070279420 | 0.090880551/0.048944549 | 0.093846504/0.072705584 | 0.092875126/0.071259917 | 0.095641962/0.051065146 |
| gray-large-u16 | 0.157133554/0.123833081 | 0.153599795/0.078516113 | 0.160188475/0.081466836 | 0.158799479/0.128928448 | 0.153901958/0.078717741 | 0.159259836/0.133258566 | 0.159565836/0.131865063 | 0.157941870/0.083709518 |
| gray-mean-f64 | 0.247329700/0.264109755 | 0.245573979/0.177533839 | 0.257758278/0.183760818 | 0.247348119/0.264043851 | 0.248869804/0.176602607 | 0.252799714/0.272725889 | 0.246889609/0.268777737 | 0.258394226/0.186054189 |
| rgb-padded-u8 | 0.189932787/0.044945245 | 0.189785453/0.032256013 | 0.189565900/0.030502045 | 0.190161474/0.042247090 | 0.189026303/0.030650261 | 0.193073958/0.045456432 | 0.189259338/0.040774590 | 0.191934000/0.031049037 |
| rgb-first-f32 | 0.162197468/0.085968831 | 0.156751288/0.054929619 | 0.159194744/0.056082865 | 0.158820522/0.084760505 | 0.159463349/0.055462369 | 0.161108317/0.086748321 | 0.158857000/0.084122110 | 0.161521000/0.057991832 |
| gray-strided-reverse-f32 | 0.101456688/0.579760600 | 0.103741712/0.052653968 | 0.102193530/0.056611497 | 0.102576856/0.582026200 | 0.103307083/0.055962180 | 0.104303273/0.585559174 | 0.104892091/0.587693565 | 0.108419811/0.062482684 |
| gray-fortran-f64 | 0.848099211/0.927615312 | 0.825950824/0.789749588 | 0.844234588/0.768514526 | 0.824381944/0.927878625 | 0.835735842/0.781796474 | 0.850311062/0.933494125 | 0.824400750/0.918238800 | 0.861201312/0.779370250 |

The order is **A B B A B A A B**, eight independent processes pinned to permitted CPU0 on **AMD EPYC 7763 64-Core Processor**, one worker/reference pool, seven unchanged rounds at 19 ms target. Baseline is genuine `4f69eba…`, candidate genuine `6b22e646…`; Python/binder, nine-case harness `93a0dc6c…` and workload contract `434d7540…` are unchanged. The prepared products have 247 image APIs +3 metadata,176 wheel members and SDK83, with 13 common dependency wheels. Source deltas and SDK objects/header-comment proof are preserved. No builds or tests ran on the timing host. Completion and fixed deadline passed.

C++ row ownership and overwritten-fill removal were a structural hypothesis; this whole-call result assigns no stage-level causal cost. The unchanged harness checks complete original/current/oracle bytes, output ownership/layout/input mutation and caller FP state internally. It exports hashes/layouts rather than full buffers or separate FP values; the publication does not claim otherwise.

The first frozen independent reviewer stopped at a descriptive metadata nesting alias. Its source, freeze, stderr and failure receipt are preserved. The successor only corrects that alias and versioned freeze path; the sealed gate fragment stays byte-identical (SHA256 `da1f9b8531a8694a64864fc55cd257ddf466b231bffe7424c2ee81367401d68b`). No result, threshold, source candidate, fixture or timing was changed or repeated.

This is a distinct closed v1 after the earlier [constant-copy](montage-strided-copy-same-host.md), [in-body SIMD](montage-stride2-simd-same-host.md) and [wrapper](montage-wrapper-simd-same-host.md) experiments. Their decisions remain unchanged. No whole-library speed or full qualification claim is made; the qualified baseline remains 241.

- [Timing run 40788962721](https://github.com/bipxx/scikit-imagexx/actions/runs/40788962721), controller `b79438dc67e07a001caed213286524be2b95de14`.
- Result artifact `12861315431`, **118,157 bytes /112 members**, SHA256 `6d3ebbe46b7a06e81f393548c5a9c222e2e462f50eb11dbb4a6f33efb1f80c73`.
- Independent successor review SHA256 `32be457bc247ecf5416dc8aa0218cde0cd8a0fd54265192360e9afcc5a191428`.
- Root recomputation SHA256 `2a612e586dd9086896ca5ff9a78dfb6c93e06c93d70d0fc54c5e433f07ca512c`.

All raw JSON documents retain original numeric text and terminal newlines. Journals/logs/reviewer programs are base64 with decoded length/SHA; all decode byte-exactly. No benchmark, fixture, original/native call or new timing ran during publication validation.
