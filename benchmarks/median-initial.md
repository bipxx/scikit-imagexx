# Median performance experiments

These exact-source trials remain unretained while the native median port is being improved. The initial source passes 8,656 scoped Windows cases and 4,460 minimum-NumPy cases in each SIMD mode, plus five native configurations. Its generic selection path is much too slow for long 1-D windows. The second source adds sliding heaps, bitonic SIMD through 32 taps and bounded footprint plans; it passes 9,296 scoped Windows and 5,100 minimum-NumPy cases in each mode, with native normal/dynamic/portable/ASan+UBSan normal/dynamic checks 2/2. Full rolling-source CI passes all three jobs: Linux 59,573 tests and Windows 59,571 plus two skips in each SIMD mode, with native normal/dynamic/portable checks 2/2 on both OSes. Later adapter and heap-layout changes have separate measurements and must not inherit these fingerprints.

All trials use 48 cases per OS at global worker limits one/four (96 rows/OS), seven randomized samples, seed 5227 and size 512. Reference pools are fixed at one; allocation, casting, dispatch and output handling are timed. All timed outputs match scikit-image 0.26.0 exactly with fallback disabled. The worker setting is a cap: small images remain serial below the activation threshold. Exact wheels, extracted packages, binary/source hashes and every raw sample are retained in the linked artifacts.

Initial one-worker geometric means are 4.219x Linux / 3.537x Windows over upstream, but 31-element 1-D windows are only 0.084–0.100x. Those severe regressions prevent retaining the initial source. The second source improves one-worker corpus means 1.602x/1.423x against that exact initial package, reaching 6.882x/5.223x over upstream. Its 5x5/3-D cases improve about 10–13x on Linux and 4–5x on Windows versus the initial package. Long 1-D improves about 8–10x but still trails upstream at 0.755–0.921x, and Windows tiny calls remain 0.811–0.860x. These adverse cases remain open targets.

The next candidate removes per-update integer remainder, uses compact heap indices and typed output stores, and avoids a redundant default-output allocation. Broader endian/output-protocol checks have been added; its measurements are pending. No timing sample below is discarded or reassigned to that candidate.

## Initial selection: `4af8806`

[Benchmark run](https://github.com/bipxx/scikit-imagexx/actions/runs/39808359531).

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | initial/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux raw | [JSON](median-initial-linux.json) | source `0394aea1d889af54ad43fc8dc437a197fe8ab90089ca65fc1496702817e0be86` | | | | | |
| Linux | uint8 | nearest | 1 | 28.64064 | 1.40678 | 20.359x | — |
| Linux | uint8 | nearest | 4 | 28.59742 | 0.81962 | 34.891x | — |
| Linux | uint8 | reflect | 1 | 28.54249 | 1.40506 | 20.314x | — |
| Linux | uint8 | reflect | 4 | 28.53959 | 0.82786 | 34.474x | — |
| Linux | uint8 | mirror | 1 | 28.54365 | 1.40018 | 20.386x | — |
| Linux | uint8 | mirror | 4 | 28.57702 | 0.82443 | 34.663x | — |
| Linux | uint8 | wrap | 1 | 28.58061 | 1.40403 | 20.356x | — |
| Linux | uint8 | wrap | 4 | 28.55200 | 0.82703 | 34.524x | — |
| Linux | uint8 | constant | 1 | 28.43783 | 1.40243 | 20.278x | — |
| Linux | uint8 | constant | 4 | 28.54235 | 0.82232 | 34.709x | — |
| Linux | uint8 | fortran | 1 | 28.85027 | 1.55889 | 18.507x | — |
| Linux | uint8 | fortran | 4 | 28.80533 | 0.97871 | 29.432x | — |
| Linux | uint8 | strided | 1 | 14.53520 | 0.77682 | 18.711x | — |
| Linux | uint8 | strided | 4 | 14.51286 | 0.77248 | 18.787x | — |
| Linux | uint8 | dense5 | 1 | 93.58740 | 91.77505 | 1.020x | — |
| Linux | uint8 | dense5 | 4 | 93.59410 | 27.03210 | 3.462x | — |
| Linux | uint8 | sparse7 | 1 | 28.79708 | 1.40167 | 20.545x | — |
| Linux | uint8 | sparse7 | 4 | 28.98816 | 0.84346 | 34.368x | — |
| Linux | uint8 | cross | 1 | 11.52499 | 0.91853 | 12.547x | — |
| Linux | uint8 | cross | 4 | 11.46399 | 0.60835 | 18.845x | — |
| Linux | uint8 | even | 1 | 7.65763 | 0.84141 | 9.101x | — |
| Linux | uint8 | even | 4 | 7.64869 | 0.59706 | 12.811x | — |
| Linux | uint8 | identity | 1 | 0.02188 | 0.01295 | 1.689x | — |
| Linux | uint8 | identity | 4 | 0.02523 | 0.01437 | 1.755x | — |
| Linux | uint8 | tiny | 1 | 0.01094 | 0.01508 | 0.726x | — |
| Linux | uint8 | tiny | 4 | 0.01104 | 0.01487 | 0.742x | — |
| Linux | uint8 | one-dimensional3 | 1 | 3.28876 | 0.36939 | 8.903x | — |
| Linux | uint8 | one-dimensional3 | 4 | 3.29926 | 0.39607 | 8.330x | — |
| Linux | uint8 | one-dimensional31 | 1 | 8.07050 | 91.20259 | 0.088x | — |
| Linux | uint8 | one-dimensional31 | 4 | 8.03008 | 28.23414 | 0.284x | — |
| Linux | uint8 | three-dimensional | 1 | 16.47638 | 16.11282 | 1.023x | — |
| Linux | uint8 | three-dimensional | 4 | 16.49312 | 16.13195 | 1.022x | — |
| Linux | float32 | nearest | 1 | 28.39447 | 1.93057 | 14.708x | — |
| Linux | float32 | nearest | 4 | 28.40271 | 0.92199 | 30.806x | — |
| Linux | float32 | reflect | 1 | 28.37814 | 1.92145 | 14.769x | — |
| Linux | float32 | reflect | 4 | 28.34990 | 0.94079 | 30.134x | — |
| Linux | float32 | mirror | 1 | 28.40312 | 1.92641 | 14.744x | — |
| Linux | float32 | mirror | 4 | 28.37649 | 0.94300 | 30.092x | — |
| Linux | float32 | wrap | 1 | 28.41068 | 1.92810 | 14.735x | — |
| Linux | float32 | wrap | 4 | 28.44169 | 0.94178 | 30.200x | — |
| Linux | float32 | constant | 1 | 28.27261 | 1.92848 | 14.661x | — |
| Linux | float32 | constant | 4 | 28.30217 | 0.93326 | 30.326x | — |
| Linux | float32 | fortran | 1 | 28.81940 | 2.55524 | 11.279x | — |
| Linux | float32 | fortran | 4 | 28.79002 | 1.52682 | 18.856x | — |
| Linux | float32 | strided | 1 | 14.42023 | 1.05222 | 13.705x | — |
| Linux | float32 | strided | 4 | 14.38756 | 1.05231 | 13.672x | — |
| Linux | float32 | dense5 | 1 | 93.27957 | 94.79224 | 0.984x | — |
| Linux | float32 | dense5 | 4 | 93.22298 | 27.04608 | 3.447x | — |
| Linux | float32 | sparse7 | 1 | 28.63956 | 1.93243 | 14.821x | — |
| Linux | float32 | sparse7 | 4 | 28.68959 | 0.93751 | 30.602x | — |
| Linux | float32 | cross | 1 | 11.37600 | 1.44387 | 7.879x | — |
| Linux | float32 | cross | 4 | 11.32652 | 0.67479 | 16.785x | — |
| Linux | float32 | even | 1 | 7.53552 | 1.35797 | 5.549x | — |
| Linux | float32 | even | 4 | 7.52838 | 0.66432 | 11.332x | — |
| Linux | float32 | identity | 1 | 0.05542 | 0.03357 | 1.651x | — |
| Linux | float32 | identity | 4 | 0.05759 | 0.03658 | 1.574x | — |
| Linux | float32 | tiny | 1 | 0.01098 | 0.01543 | 0.711x | — |
| Linux | float32 | tiny | 4 | 0.01091 | 0.01531 | 0.713x | — |
| Linux | float32 | one-dimensional3 | 1 | 3.71135 | 1.20701 | 3.075x | — |
| Linux | float32 | one-dimensional3 | 4 | 3.70585 | 0.66533 | 5.570x | — |
| Linux | float32 | one-dimensional31 | 1 | 9.54492 | 112.88527 | 0.085x | — |
| Linux | float32 | one-dimensional31 | 4 | 9.53914 | 32.18048 | 0.296x | — |
| Linux | float32 | three-dimensional | 1 | 16.44542 | 16.65499 | 0.987x | — |
| Linux | float32 | three-dimensional | 4 | 16.45230 | 16.63995 | 0.989x | — |
| Linux | float64 | nearest | 1 | 28.63625 | 3.75579 | 7.625x | — |
| Linux | float64 | nearest | 4 | 27.97752 | 0.98385 | 28.437x | — |
| Linux | float64 | reflect | 1 | 28.59202 | 3.71866 | 7.689x | — |
| Linux | float64 | reflect | 4 | 28.05355 | 0.94752 | 29.607x | — |
| Linux | float64 | mirror | 1 | 28.62609 | 3.79098 | 7.551x | — |
| Linux | float64 | mirror | 4 | 27.99869 | 0.94252 | 29.706x | — |
| Linux | float64 | wrap | 1 | 28.69866 | 3.79241 | 7.567x | — |
| Linux | float64 | wrap | 4 | 28.06227 | 0.94682 | 29.638x | — |
| Linux | float64 | constant | 1 | 28.55719 | 3.80352 | 7.508x | — |
| Linux | float64 | constant | 4 | 27.86364 | 0.96122 | 28.988x | — |
| Linux | float64 | fortran | 1 | 29.36260 | 5.77078 | 5.088x | — |
| Linux | float64 | fortran | 4 | 29.34165 | 4.57832 | 6.409x | — |
| Linux | float64 | strided | 1 | 14.20304 | 0.85904 | 16.534x | — |
| Linux | float64 | strided | 4 | 14.19405 | 0.85795 | 16.544x | — |
| Linux | float64 | dense5 | 1 | 93.46263 | 96.44587 | 0.969x | — |
| Linux | float64 | dense5 | 4 | 92.68988 | 27.11861 | 3.418x | — |
| Linux | float64 | sparse7 | 1 | 28.81370 | 3.72509 | 7.735x | — |
| Linux | float64 | sparse7 | 4 | 28.26427 | 0.95798 | 29.504x | — |
| Linux | float64 | cross | 1 | 11.60958 | 3.20765 | 3.619x | — |
| Linux | float64 | cross | 4 | 11.01144 | 0.69699 | 15.798x | — |
| Linux | float64 | even | 1 | 7.83899 | 3.12402 | 2.509x | — |
| Linux | float64 | even | 4 | 7.25458 | 0.66581 | 10.896x | — |
| Linux | float64 | identity | 1 | 0.09046 | 0.05846 | 1.547x | — |
| Linux | float64 | identity | 4 | 0.09196 | 0.05887 | 1.562x | — |
| Linux | float64 | tiny | 1 | 0.01104 | 0.01535 | 0.719x | — |
| Linux | float64 | tiny | 4 | 0.01104 | 0.01545 | 0.715x | — |
| Linux | float64 | one-dimensional3 | 1 | 3.82356 | 3.04629 | 1.255x | — |
| Linux | float64 | one-dimensional3 | 4 | 3.63927 | 0.66462 | 5.476x | — |
| Linux | float64 | one-dimensional31 | 1 | 10.09077 | 114.50990 | 0.088x | — |
| Linux | float64 | one-dimensional31 | 4 | 9.46989 | 32.20038 | 0.294x | — |
| Linux | float64 | three-dimensional | 1 | 16.33877 | 16.54205 | 0.988x | — |
| Linux | float64 | three-dimensional | 4 | 16.31383 | 16.53214 | 0.987x | — |
| Windows raw | [JSON](median-initial-windows.json) | source `0394aea1d889af54ad43fc8dc437a197fe8ab90089ca65fc1496702817e0be86` | | | | | |
| Windows | uint8 | nearest | 1 | 25.68750 | 1.87366 | 13.710x | — |
| Windows | uint8 | nearest | 4 | 25.60420 | 1.16580 | 21.963x | — |
| Windows | uint8 | reflect | 1 | 25.68640 | 1.88781 | 13.606x | — |
| Windows | uint8 | reflect | 4 | 25.88300 | 1.19729 | 21.618x | — |
| Windows | uint8 | mirror | 1 | 25.81040 | 1.89088 | 13.650x | — |
| Windows | uint8 | mirror | 4 | 25.54560 | 1.18603 | 21.539x | — |
| Windows | uint8 | wrap | 1 | 25.45670 | 1.89590 | 13.427x | — |
| Windows | uint8 | wrap | 4 | 25.26000 | 1.18213 | 21.368x | — |
| Windows | uint8 | constant | 1 | 25.61080 | 1.87746 | 13.641x | — |
| Windows | uint8 | constant | 4 | 25.69440 | 1.19759 | 21.455x | — |
| Windows | uint8 | fortran | 1 | 25.90730 | 2.11018 | 12.277x | — |
| Windows | uint8 | fortran | 4 | 25.65850 | 1.39633 | 18.376x | — |
| Windows | uint8 | strided | 1 | 12.98610 | 1.07154 | 12.119x | — |
| Windows | uint8 | strided | 4 | 13.61000 | 1.07233 | 12.692x | — |
| Windows | uint8 | dense5 | 1 | 86.96150 | 84.48290 | 1.029x | — |
| Windows | uint8 | dense5 | 4 | 85.71540 | 25.36340 | 3.379x | — |
| Windows | uint8 | sparse7 | 1 | 25.77830 | 1.84491 | 13.973x | — |
| Windows | uint8 | sparse7 | 4 | 25.79110 | 1.16474 | 22.143x | — |
| Windows | uint8 | cross | 1 | 10.30380 | 1.38381 | 7.446x | — |
| Windows | uint8 | cross | 4 | 10.23030 | 0.91734 | 11.152x | — |
| Windows | uint8 | even | 1 | 6.80265 | 1.32002 | 5.153x | — |
| Windows | uint8 | even | 4 | 6.76575 | 0.88826 | 7.617x | — |
| Windows | uint8 | identity | 1 | 0.01464 | 0.00683 | 2.142x | — |
| Windows | uint8 | identity | 4 | 0.01401 | 0.00655 | 2.138x | — |
| Windows | uint8 | tiny | 1 | 0.00848 | 0.01441 | 0.589x | — |
| Windows | uint8 | tiny | 4 | 0.00797 | 0.01244 | 0.641x | — |
| Windows | uint8 | one-dimensional3 | 1 | 3.64360 | 1.48411 | 2.455x | — |
| Windows | uint8 | one-dimensional3 | 4 | 3.63267 | 1.29342 | 2.809x | — |
| Windows | uint8 | one-dimensional31 | 1 | 8.39890 | 84.03890 | 0.100x | — |
| Windows | uint8 | one-dimensional31 | 4 | 8.38270 | 27.15890 | 0.309x | — |
| Windows | uint8 | three-dimensional | 1 | 15.15740 | 14.80080 | 1.024x | — |
| Windows | uint8 | three-dimensional | 4 | 14.94950 | 14.66680 | 1.019x | — |
| Windows | float32 | nearest | 1 | 25.93890 | 2.49346 | 10.403x | — |
| Windows | float32 | nearest | 4 | 25.41270 | 1.77753 | 14.297x | — |
| Windows | float32 | reflect | 1 | 26.01970 | 2.51722 | 10.337x | — |
| Windows | float32 | reflect | 4 | 26.01460 | 1.79107 | 14.525x | — |
| Windows | float32 | mirror | 1 | 26.24350 | 2.45868 | 10.674x | — |
| Windows | float32 | mirror | 4 | 26.31990 | 1.85100 | 14.219x | — |
| Windows | float32 | wrap | 1 | 25.80310 | 2.55194 | 10.111x | — |
| Windows | float32 | wrap | 4 | 25.98100 | 1.75880 | 14.772x | — |
| Windows | float32 | constant | 1 | 25.86890 | 2.51142 | 10.301x | — |
| Windows | float32 | constant | 4 | 25.98840 | 1.76557 | 14.720x | — |
| Windows | float32 | fortran | 1 | 26.37720 | 2.92680 | 9.012x | — |
| Windows | float32 | fortran | 4 | 26.64840 | 2.28095 | 11.683x | — |
| Windows | float32 | strided | 1 | 13.54260 | 1.51011 | 8.968x | — |
| Windows | float32 | strided | 4 | 13.21380 | 1.47212 | 8.976x | — |
| Windows | float32 | dense5 | 1 | 85.84860 | 84.52670 | 1.016x | — |
| Windows | float32 | dense5 | 4 | 86.55170 | 26.39320 | 3.279x | — |
| Windows | float32 | sparse7 | 1 | 27.12570 | 2.59906 | 10.437x | — |
| Windows | float32 | sparse7 | 4 | 26.44640 | 1.82782 | 14.469x | — |
| Windows | float32 | cross | 1 | 10.25060 | 2.04835 | 5.004x | — |
| Windows | float32 | cross | 4 | 10.56960 | 1.68826 | 6.261x | — |
| Windows | float32 | even | 1 | 6.72375 | 2.05040 | 3.279x | — |
| Windows | float32 | even | 4 | 6.71000 | 1.60545 | 4.180x | — |
| Windows | float32 | identity | 1 | 0.18332 | 0.17410 | 1.053x | — |
| Windows | float32 | identity | 4 | 0.19699 | 0.19413 | 1.015x | — |
| Windows | float32 | tiny | 1 | 0.00853 | 0.01422 | 0.600x | — |
| Windows | float32 | tiny | 4 | 0.00841 | 0.01362 | 0.617x | — |
| Windows | float32 | one-dimensional3 | 1 | 3.58720 | 1.99487 | 1.798x | — |
| Windows | float32 | one-dimensional3 | 4 | 3.51597 | 1.70060 | 2.067x | — |
| Windows | float32 | one-dimensional31 | 1 | 8.49560 | 101.31400 | 0.084x | — |
| Windows | float32 | one-dimensional31 | 4 | 8.83550 | 31.50640 | 0.280x | — |
| Windows | float32 | three-dimensional | 1 | 15.43020 | 14.91370 | 1.035x | — |
| Windows | float32 | three-dimensional | 4 | 15.24180 | 14.93570 | 1.020x | — |
| Windows | float64 | nearest | 1 | 26.20700 | 2.69882 | 9.711x | — |
| Windows | float64 | nearest | 4 | 26.15330 | 1.99478 | 13.111x | — |
| Windows | float64 | reflect | 1 | 26.39670 | 2.66922 | 9.889x | — |
| Windows | float64 | reflect | 4 | 25.99600 | 1.94683 | 13.353x | — |
| Windows | float64 | mirror | 1 | 25.87250 | 2.53088 | 10.223x | — |
| Windows | float64 | mirror | 4 | 25.90100 | 1.95990 | 13.215x | — |
| Windows | float64 | wrap | 1 | 25.85680 | 2.63792 | 9.802x | — |
| Windows | float64 | wrap | 4 | 25.34870 | 1.88505 | 13.447x | — |
| Windows | float64 | constant | 1 | 25.73400 | 2.52372 | 10.197x | — |
| Windows | float64 | constant | 4 | 26.21560 | 1.90153 | 13.787x | — |
| Windows | float64 | fortran | 1 | 26.03740 | 3.23575 | 8.047x | — |
| Windows | float64 | fortran | 4 | 26.19020 | 2.66778 | 9.817x | — |
| Windows | float64 | strided | 1 | 13.20650 | 1.69506 | 7.791x | — |
| Windows | float64 | strided | 4 | 13.25840 | 1.69777 | 7.809x | — |
| Windows | float64 | dense5 | 1 | 84.99130 | 83.63290 | 1.016x | — |
| Windows | float64 | dense5 | 4 | 84.18820 | 25.31720 | 3.325x | — |
| Windows | float64 | sparse7 | 1 | 25.82290 | 2.52460 | 10.229x | — |
| Windows | float64 | sparse7 | 4 | 25.23670 | 1.89610 | 13.310x | — |
| Windows | float64 | cross | 1 | 9.81830 | 2.18162 | 4.500x | — |
| Windows | float64 | cross | 4 | 9.76200 | 1.62932 | 5.991x | — |
| Windows | float64 | even | 1 | 6.40735 | 1.94236 | 3.299x | — |
| Windows | float64 | even | 4 | 6.40745 | 1.61564 | 3.966x | — |
| Windows | float64 | identity | 1 | 0.31018 | 0.29942 | 1.036x | — |
| Windows | float64 | identity | 4 | 0.30871 | 0.30253 | 1.020x | — |
| Windows | float64 | tiny | 1 | 0.00916 | 0.01333 | 0.687x | — |
| Windows | float64 | tiny | 4 | 0.00776 | 0.01259 | 0.616x | — |
| Windows | float64 | one-dimensional3 | 1 | 3.42935 | 1.96507 | 1.745x | — |
| Windows | float64 | one-dimensional3 | 4 | 3.45332 | 1.62932 | 2.119x | — |
| Windows | float64 | one-dimensional31 | 1 | 8.60960 | 100.47410 | 0.086x | — |
| Windows | float64 | one-dimensional31 | 4 | 8.34070 | 30.39560 | 0.274x | — |
| Windows | float64 | three-dimensional | 1 | 15.58460 | 14.71420 | 1.059x | — |
| Windows | float64 | three-dimensional | 4 | 15.28750 | 14.87950 | 1.027x | — |

## Rolling heaps and medium SIMD: `7b63b82`

[Benchmark run](https://github.com/bipxx/scikit-imagexx/actions/runs/39810758639).

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | initial/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux raw | [JSON](median-rolling-linux.json) | source `4b53f8b61e9eb25265e584d60c54de4d8c47afe9021de7b1e638d0b02a3115a6` | | | | | |
| Linux | uint8 | nearest | 1 | 32.78736 | 1.74075 | 18.835x | 1.018x |
| Linux | uint8 | nearest | 4 | 32.75290 | 1.02442 | 31.972x | 1.006x |
| Linux | uint8 | reflect | 1 | 32.76411 | 1.74739 | 18.750x | 1.001x |
| Linux | uint8 | reflect | 4 | 32.76769 | 1.00291 | 32.673x | 1.023x |
| Linux | uint8 | mirror | 1 | 32.67687 | 1.73343 | 18.851x | 1.008x |
| Linux | uint8 | mirror | 4 | 32.68006 | 1.02136 | 31.997x | 1.014x |
| Linux | uint8 | wrap | 1 | 32.84851 | 1.77504 | 18.506x | 1.013x |
| Linux | uint8 | wrap | 4 | 32.76536 | 1.00787 | 32.509x | 1.041x |
| Linux | uint8 | constant | 1 | 32.65278 | 1.73525 | 18.817x | 1.009x |
| Linux | uint8 | constant | 4 | 32.76051 | 0.99643 | 32.878x | 1.028x |
| Linux | uint8 | fortran | 1 | 33.13181 | 2.11964 | 15.631x | 1.025x |
| Linux | uint8 | fortran | 4 | 33.16531 | 1.39686 | 23.743x | 1.004x |
| Linux | uint8 | strided | 1 | 16.76516 | 0.97435 | 17.206x | 1.009x |
| Linux | uint8 | strided | 4 | 16.68633 | 0.96138 | 17.357x | 1.036x |
| Linux | uint8 | dense5 | 1 | 107.52188 | 7.95677 | 13.513x | 13.288x |
| Linux | uint8 | dense5 | 4 | 107.64773 | 4.03290 | 26.692x | 8.075x |
| Linux | uint8 | sparse7 | 1 | 32.99491 | 1.75142 | 18.839x | 1.016x |
| Linux | uint8 | sparse7 | 4 | 33.01253 | 1.01818 | 32.423x | 1.009x |
| Linux | uint8 | cross | 1 | 13.62601 | 1.18334 | 11.515x | 1.001x |
| Linux | uint8 | cross | 4 | 13.64920 | 0.73616 | 18.541x | 1.030x |
| Linux | uint8 | even | 1 | 9.24091 | 1.12190 | 8.237x | 1.012x |
| Linux | uint8 | even | 4 | 9.25217 | 0.73358 | 12.612x | 1.013x |
| Linux | uint8 | identity | 1 | 0.04545 | 0.02091 | 2.174x | 1.047x |
| Linux | uint8 | identity | 4 | 0.04310 | 0.02135 | 2.019x | 1.013x |
| Linux | uint8 | tiny | 1 | 0.01957 | 0.01879 | 1.042x | 1.597x |
| Linux | uint8 | tiny | 4 | 0.01998 | 0.01931 | 1.035x | 1.568x |
| Linux | uint8 | one-dimensional3 | 1 | 4.85000 | 3.17557 | 1.527x | 1.102x |
| Linux | uint8 | one-dimensional3 | 4 | 4.92884 | 2.85111 | 1.729x | 1.037x |
| Linux | uint8 | one-dimensional31 | 1 | 11.86510 | 14.16214 | 0.838x | 7.925x |
| Linux | uint8 | one-dimensional31 | 4 | 11.84492 | 7.28879 | 1.625x | 4.957x |
| Linux | uint8 | three-dimensional | 1 | 19.00512 | 1.78528 | 10.645x | 10.489x |
| Linux | uint8 | three-dimensional | 4 | 19.09302 | 1.77408 | 10.762x | 10.556x |
| Linux | float32 | nearest | 1 | 32.55494 | 2.37409 | 13.713x | 1.004x |
| Linux | float32 | nearest | 4 | 32.58321 | 1.22582 | 26.581x | 1.038x |
| Linux | float32 | reflect | 1 | 32.53366 | 2.34374 | 13.881x | 1.008x |
| Linux | float32 | reflect | 4 | 32.52660 | 1.24031 | 26.224x | 1.010x |
| Linux | float32 | mirror | 1 | 32.59388 | 2.37519 | 13.723x | 1.004x |
| Linux | float32 | mirror | 4 | 32.55171 | 1.24577 | 26.130x | 1.013x |
| Linux | float32 | wrap | 1 | 32.59323 | 2.36157 | 13.801x | 1.008x |
| Linux | float32 | wrap | 4 | 32.66524 | 1.24622 | 26.212x | 1.012x |
| Linux | float32 | constant | 1 | 32.54878 | 2.32079 | 14.025x | 1.006x |
| Linux | float32 | constant | 4 | 32.53598 | 1.21536 | 26.771x | 1.013x |
| Linux | float32 | fortran | 1 | 32.92701 | 2.97328 | 11.074x | 1.011x |
| Linux | float32 | fortran | 4 | 32.93857 | 1.85509 | 17.756x | 1.016x |
| Linux | float32 | strided | 1 | 16.60477 | 1.27463 | 13.027x | 1.009x |
| Linux | float32 | strided | 4 | 16.58308 | 1.27568 | 12.999x | 1.018x |
| Linux | float32 | dense5 | 1 | 107.67025 | 8.48922 | 12.683x | 12.895x |
| Linux | float32 | dense5 | 4 | 107.81133 | 4.21609 | 25.571x | 7.692x |
| Linux | float32 | sparse7 | 1 | 32.87725 | 2.34676 | 14.010x | 1.010x |
| Linux | float32 | sparse7 | 4 | 32.87896 | 1.21027 | 27.167x | 1.053x |
| Linux | float32 | cross | 1 | 13.49604 | 1.79708 | 7.510x | 1.012x |
| Linux | float32 | cross | 4 | 13.36777 | 0.98036 | 13.636x | 1.001x |
| Linux | float32 | even | 1 | 8.94493 | 1.69239 | 5.285x | 1.005x |
| Linux | float32 | even | 4 | 9.00187 | 0.95226 | 9.453x | 1.009x |
| Linux | float32 | identity | 1 | 0.09389 | 0.05573 | 1.685x | 0.971x |
| Linux | float32 | identity | 4 | 0.09367 | 0.05284 | 1.773x | 1.051x |
| Linux | float32 | tiny | 1 | 0.02073 | 0.01951 | 1.062x | 1.594x |
| Linux | float32 | tiny | 4 | 0.02189 | 0.02168 | 1.010x | 1.477x |
| Linux | float32 | one-dimensional3 | 1 | 4.43711 | 1.81593 | 2.443x | 0.925x |
| Linux | float32 | one-dimensional3 | 4 | 4.39790 | 0.98935 | 4.445x | 0.946x |
| Linux | float32 | one-dimensional31 | 1 | 11.47678 | 15.21108 | 0.755x | 8.631x |
| Linux | float32 | one-dimensional31 | 4 | 11.51957 | 4.91268 | 2.345x | 7.833x |
| Linux | float32 | three-dimensional | 1 | 19.08923 | 1.90545 | 10.018x | 10.244x |
| Linux | float32 | three-dimensional | 4 | 19.09962 | 1.87823 | 10.169x | 10.393x |
| Linux | float64 | nearest | 1 | 33.26738 | 5.04390 | 6.596x | 1.019x |
| Linux | float64 | nearest | 4 | 32.39881 | 1.27210 | 25.469x | 1.001x |
| Linux | float64 | reflect | 1 | 32.45850 | 1.99426 | 16.276x | 1.005x |
| Linux | float64 | reflect | 4 | 33.19089 | 3.73329 | 8.891x | 1.046x |
| Linux | float64 | mirror | 1 | 32.47837 | 1.99237 | 16.301x | 1.010x |
| Linux | float64 | mirror | 4 | 32.45396 | 1.32153 | 24.558x | 0.955x |
| Linux | float64 | wrap | 1 | 32.43194 | 2.02772 | 15.994x | 1.007x |
| Linux | float64 | wrap | 4 | 32.49489 | 1.30931 | 24.818x | 0.977x |
| Linux | float64 | constant | 1 | 33.12043 | 5.05852 | 6.547x | 1.006x |
| Linux | float64 | constant | 4 | 32.46997 | 1.30509 | 24.879x | 1.000x |
| Linux | float64 | fortran | 1 | 33.06076 | 6.29117 | 5.255x | 1.014x |
| Linux | float64 | fortran | 4 | 32.93967 | 5.05861 | 6.512x | 1.028x |
| Linux | float64 | strided | 1 | 16.62682 | 1.16214 | 14.307x | 1.009x |
| Linux | float64 | strided | 4 | 16.43801 | 1.16695 | 14.086x | 1.009x |
| Linux | float64 | dense5 | 1 | 108.20212 | 10.82979 | 9.991x | 10.361x |
| Linux | float64 | dense5 | 4 | 107.36695 | 4.35088 | 24.677x | 7.464x |
| Linux | float64 | sparse7 | 1 | 32.72269 | 2.05087 | 15.955x | 0.991x |
| Linux | float64 | sparse7 | 4 | 33.52182 | 3.83088 | 8.750x | 1.027x |
| Linux | float64 | cross | 1 | 13.18561 | 1.43735 | 9.174x | 1.009x |
| Linux | float64 | cross | 4 | 13.20019 | 1.01812 | 12.965x | 1.028x |
| Linux | float64 | even | 1 | 8.77304 | 1.39043 | 6.310x | 0.997x |
| Linux | float64 | even | 4 | 8.74951 | 0.99476 | 8.796x | 1.001x |
| Linux | float64 | identity | 1 | 0.15156 | 0.09336 | 1.623x | 0.972x |
| Linux | float64 | identity | 4 | 0.15251 | 0.09321 | 1.636x | 0.995x |
| Linux | float64 | tiny | 1 | 0.01920 | 0.01887 | 1.017x | 1.635x |
| Linux | float64 | tiny | 4 | 0.01816 | 0.01859 | 0.977x | 1.601x |
| Linux | float64 | one-dimensional3 | 1 | 4.94744 | 4.20252 | 1.177x | 1.010x |
| Linux | float64 | one-dimensional3 | 4 | 4.48640 | 1.03398 | 4.339x | 0.975x |
| Linux | float64 | one-dimensional31 | 1 | 11.62307 | 14.60400 | 0.796x | 8.919x |
| Linux | float64 | one-dimensional31 | 4 | 12.45003 | 7.37299 | 1.689x | 5.575x |
| Linux | float64 | three-dimensional | 1 | 18.97516 | 1.81436 | 10.458x | 10.659x |
| Linux | float64 | three-dimensional | 4 | 19.01496 | 1.81912 | 10.453x | 10.651x |
| Windows raw | [JSON](median-rolling-windows.json) | source `4b53f8b61e9eb25265e584d60c54de4d8c47afe9021de7b1e638d0b02a3115a6` | | | | | |
| Windows | uint8 | nearest | 1 | 23.69910 | 1.56969 | 15.098x | 1.072x |
| Windows | uint8 | nearest | 4 | 24.06060 | 0.96074 | 25.044x | 1.053x |
| Windows | uint8 | reflect | 1 | 24.24820 | 1.55630 | 15.581x | 1.072x |
| Windows | uint8 | reflect | 4 | 23.92430 | 0.96773 | 24.722x | 1.050x |
| Windows | uint8 | mirror | 1 | 24.11420 | 1.59114 | 15.155x | 1.057x |
| Windows | uint8 | mirror | 4 | 24.25020 | 0.97904 | 24.769x | 1.054x |
| Windows | uint8 | wrap | 1 | 23.36820 | 1.54039 | 15.170x | 1.092x |
| Windows | uint8 | wrap | 4 | 23.67310 | 0.96617 | 24.502x | 1.024x |
| Windows | uint8 | constant | 1 | 24.49750 | 1.58970 | 15.410x | 1.054x |
| Windows | uint8 | constant | 4 | 24.40020 | 0.97093 | 25.131x | 1.043x |
| Windows | uint8 | fortran | 1 | 24.53620 | 1.79035 | 13.705x | 1.059x |
| Windows | uint8 | fortran | 4 | 24.26130 | 1.16583 | 20.810x | 1.027x |
| Windows | uint8 | strided | 1 | 12.13080 | 0.87431 | 13.875x | 1.058x |
| Windows | uint8 | strided | 4 | 12.26560 | 0.87429 | 14.029x | 1.050x |
| Windows | uint8 | dense5 | 1 | 82.09530 | 17.78120 | 4.617x | 4.527x |
| Windows | uint8 | dense5 | 4 | 82.69220 | 7.66870 | 10.783x | 3.108x |
| Windows | uint8 | sparse7 | 1 | 25.25420 | 1.63976 | 15.401x | 1.077x |
| Windows | uint8 | sparse7 | 4 | 24.96910 | 0.99309 | 25.143x | 1.039x |
| Windows | uint8 | cross | 1 | 9.99290 | 1.17929 | 8.474x | 1.097x |
| Windows | uint8 | cross | 4 | 9.93330 | 0.79945 | 12.425x | 1.045x |
| Windows | uint8 | even | 1 | 6.55135 | 1.11647 | 5.868x | 1.086x |
| Windows | uint8 | even | 4 | 6.55895 | 0.74065 | 8.856x | 1.092x |
| Windows | uint8 | identity | 1 | 0.01260 | 0.00592 | 2.129x | 1.036x |
| Windows | uint8 | identity | 4 | 0.01259 | 0.00601 | 2.092x | 1.019x |
| Windows | uint8 | tiny | 1 | 0.00746 | 0.00868 | 0.860x | 1.312x |
| Windows | uint8 | tiny | 4 | 0.00742 | 0.00855 | 0.867x | 1.341x |
| Windows | uint8 | one-dimensional3 | 1 | 3.41495 | 1.30104 | 2.625x | 1.020x |
| Windows | uint8 | one-dimensional3 | 4 | 3.36262 | 1.11352 | 3.020x | 0.996x |
| Windows | uint8 | one-dimensional31 | 1 | 7.84100 | 8.51310 | 0.921x | 9.520x |
| Windows | uint8 | one-dimensional31 | 4 | 7.91220 | 3.43847 | 2.301x | 7.748x |
| Windows | uint8 | three-dimensional | 1 | 14.54650 | 3.19943 | 4.547x | 4.428x |
| Windows | uint8 | three-dimensional | 4 | 14.56350 | 3.19537 | 4.558x | 4.439x |
| Windows | float32 | nearest | 1 | 25.06920 | 2.29160 | 10.940x | 1.007x |
| Windows | float32 | nearest | 4 | 25.13820 | 1.58451 | 15.865x | 1.019x |
| Windows | float32 | reflect | 1 | 25.10330 | 2.29617 | 10.933x | 0.994x |
| Windows | float32 | reflect | 4 | 25.02060 | 1.57020 | 15.935x | 1.024x |
| Windows | float32 | mirror | 1 | 25.50720 | 2.31327 | 11.026x | 1.009x |
| Windows | float32 | mirror | 4 | 25.36020 | 1.60091 | 15.841x | 0.982x |
| Windows | float32 | wrap | 1 | 25.06760 | 2.31634 | 10.822x | 0.997x |
| Windows | float32 | wrap | 4 | 25.04480 | 1.59222 | 15.729x | 0.989x |
| Windows | float32 | constant | 1 | 25.02100 | 2.30044 | 10.877x | 1.000x |
| Windows | float32 | constant | 4 | 25.04500 | 1.55731 | 16.082x | 1.004x |
| Windows | float32 | fortran | 1 | 24.98660 | 2.70602 | 9.234x | 1.023x |
| Windows | float32 | fortran | 4 | 25.15000 | 1.98540 | 12.667x | 1.004x |
| Windows | float32 | strided | 1 | 12.54590 | 1.27905 | 9.809x | 1.028x |
| Windows | float32 | strided | 4 | 12.62820 | 1.38095 | 9.145x | 1.000x |
| Windows | float32 | dense5 | 1 | 83.23240 | 18.72410 | 4.445x | 4.395x |
| Windows | float32 | dense5 | 4 | 83.37650 | 8.37790 | 9.952x | 2.905x |
| Windows | float32 | sparse7 | 1 | 25.33480 | 2.30018 | 11.014x | 1.013x |
| Windows | float32 | sparse7 | 4 | 25.48570 | 1.58394 | 16.090x | 1.013x |
| Windows | float32 | cross | 1 | 9.63040 | 1.86904 | 5.153x | 1.021x |
| Windows | float32 | cross | 4 | 9.59630 | 1.39098 | 6.899x | 0.994x |
| Windows | float32 | even | 1 | 6.25905 | 1.80323 | 3.471x | 1.016x |
| Windows | float32 | even | 4 | 6.30980 | 1.38197 | 4.566x | 1.009x |
| Windows | float32 | identity | 1 | 0.17897 | 0.17145 | 1.044x | 1.004x |
| Windows | float32 | identity | 4 | 0.16728 | 0.15955 | 1.048x | 0.997x |
| Windows | float32 | tiny | 1 | 0.00747 | 0.00921 | 0.811x | 1.286x |
| Windows | float32 | tiny | 4 | 0.00748 | 0.00916 | 0.816x | 1.300x |
| Windows | float32 | one-dimensional3 | 1 | 3.34512 | 1.77716 | 1.882x | 1.018x |
| Windows | float32 | one-dimensional3 | 4 | 3.32247 | 1.51334 | 2.195x | 0.988x |
| Windows | float32 | one-dimensional31 | 1 | 8.12760 | 10.16880 | 0.799x | 9.663x |
| Windows | float32 | one-dimensional31 | 4 | 8.09130 | 3.95630 | 2.045x | 7.437x |
| Windows | float32 | three-dimensional | 1 | 14.72110 | 3.28777 | 4.478x | 4.348x |
| Windows | float32 | three-dimensional | 4 | 14.70240 | 3.31220 | 4.439x | 4.348x |
| Windows | float64 | nearest | 1 | 24.92220 | 2.30203 | 10.826x | 0.980x |
| Windows | float64 | nearest | 4 | 24.65650 | 1.64836 | 14.958x | 0.974x |
| Windows | float64 | reflect | 1 | 25.31290 | 2.29827 | 11.014x | 1.011x |
| Windows | float64 | reflect | 4 | 25.41140 | 1.67820 | 15.142x | 0.969x |
| Windows | float64 | mirror | 1 | 28.74240 | 3.12278 | 9.204x | 0.745x |
| Windows | float64 | mirror | 4 | 25.60740 | 1.74545 | 14.671x | 0.951x |
| Windows | float64 | wrap | 1 | 25.59040 | 2.39904 | 10.667x | 0.978x |
| Windows | float64 | wrap | 4 | 25.59570 | 1.74694 | 14.652x | 1.143x |
| Windows | float64 | constant | 1 | 25.98570 | 2.58210 | 10.064x | 0.915x |
| Windows | float64 | constant | 4 | 25.52540 | 1.74122 | 14.660x | 0.986x |
| Windows | float64 | fortran | 1 | 25.93270 | 3.10867 | 8.342x | 1.004x |
| Windows | float64 | fortran | 4 | 25.42610 | 2.37254 | 10.717x | 1.003x |
| Windows | float64 | strided | 1 | 12.61330 | 1.51621 | 8.319x | 0.968x |
| Windows | float64 | strided | 4 | 13.00190 | 1.45582 | 8.931x | 1.006x |
| Windows | float64 | dense5 | 1 | 84.29180 | 19.15680 | 4.400x | 4.343x |
| Windows | float64 | dense5 | 4 | 85.56410 | 15.61470 | 5.480x | 2.948x |
| Windows | float64 | sparse7 | 1 | 26.72090 | 2.41440 | 11.067x | 1.002x |
| Windows | float64 | sparse7 | 4 | 29.16190 | 2.00956 | 14.512x | 1.093x |
| Windows | float64 | cross | 1 | 10.98190 | 2.28913 | 4.797x | 1.056x |
| Windows | float64 | cross | 4 | 9.67930 | 1.54930 | 6.248x | 0.975x |
| Windows | float64 | even | 1 | 6.26740 | 1.82794 | 3.429x | 0.984x |
| Windows | float64 | even | 4 | 6.29010 | 1.57953 | 3.982x | 0.941x |
| Windows | float64 | identity | 1 | 0.29389 | 0.28621 | 1.027x | 0.992x |
| Windows | float64 | identity | 4 | 0.29207 | 0.29011 | 1.007x | 0.975x |
| Windows | float64 | tiny | 1 | 0.00754 | 0.00918 | 0.822x | 1.299x |
| Windows | float64 | tiny | 4 | 0.00748 | 0.00950 | 0.788x | 1.310x |
| Windows | float64 | one-dimensional3 | 1 | 3.35197 | 1.82393 | 1.838x | 0.980x |
| Windows | float64 | one-dimensional3 | 4 | 3.31208 | 1.46206 | 2.265x | 0.985x |
| Windows | float64 | one-dimensional31 | 1 | 8.04080 | 9.89210 | 0.813x | 9.829x |
| Windows | float64 | one-dimensional31 | 4 | 8.02790 | 3.81450 | 2.105x | 7.698x |
| Windows | float64 | three-dimensional | 1 | 14.45660 | 3.25815 | 4.437x | 4.354x |
| Windows | float64 | three-dimensional | 4 | 14.50830 | 3.23695 | 4.482x | 4.407x |


## Compact heaps and adapter: `6b0dd2a`

[Successful paired benchmark](https://github.com/bipxx/scikit-imagexx/actions/runs/39813349869). This source uses 32-bit heap positions, typed output stores, a counter instead of per-pixel remainder, and deferred default-output allocation. Scoped Windows SIMD/scalar checks pass 9,372 cases each; minimum-runtime checks pass 5,176 each, with five native configurations passing 2/2. These are separate-run measurements and cannot be treated as paired changes against the preceding rolling trial.

One-worker upstream/current geometric means are **8.193x Linux / 4.576x Windows**. The 31-element cases remain slower at **0.916–0.940x Linux** and **0.871–0.926x Windows**. Windows float32/float64 tiny cases are **0.979x/0.977x**. The streaming-input candidate is being measured separately; these results are preserved without claiming the remaining bottleneck is solved.

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | initial/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux raw | [JSON](median-compact-linux.json) | source `9c773062101107030765129a0127be80125e5df74eff15ef2e63f12a70fc440f` | | | | | |
| Linux | uint8 | nearest | 1 | 32.74123 | 1.70628 | 19.189x | 1.028x |
| Linux | uint8 | nearest | 4 | 32.77419 | 0.97968 | 33.454x | 1.040x |
| Linux | uint8 | reflect | 1 | 32.81472 | 1.70829 | 19.209x | 1.012x |
| Linux | uint8 | reflect | 4 | 32.69768 | 0.98813 | 33.090x | 1.021x |
| Linux | uint8 | mirror | 1 | 32.72166 | 1.72427 | 18.977x | 1.018x |
| Linux | uint8 | mirror | 4 | 32.75670 | 0.97979 | 33.432x | 1.058x |
| Linux | uint8 | wrap | 1 | 32.72518 | 1.71006 | 19.137x | 1.016x |
| Linux | uint8 | wrap | 4 | 32.66925 | 0.98100 | 33.302x | 1.039x |
| Linux | uint8 | constant | 1 | 32.70667 | 1.73583 | 18.842x | 1.007x |
| Linux | uint8 | constant | 4 | 32.73367 | 0.98306 | 33.298x | 1.038x |
| Linux | uint8 | fortran | 1 | 32.98131 | 2.08017 | 15.855x | 1.009x |
| Linux | uint8 | fortran | 4 | 32.83229 | 1.35607 | 24.211x | 1.035x |
| Linux | uint8 | strided | 1 | 16.65736 | 0.93536 | 17.808x | 1.019x |
| Linux | uint8 | strided | 4 | 16.56340 | 0.93500 | 17.715x | 1.026x |
| Linux | uint8 | dense5 | 1 | 107.62472 | 7.88657 | 13.647x | 13.412x |
| Linux | uint8 | dense5 | 4 | 107.61607 | 3.94639 | 27.270x | 8.207x |
| Linux | uint8 | sparse7 | 1 | 32.89817 | 1.69129 | 19.452x | 1.019x |
| Linux | uint8 | sparse7 | 4 | 32.91644 | 0.98961 | 33.262x | 1.039x |
| Linux | uint8 | cross | 1 | 13.52043 | 1.15225 | 11.734x | 1.019x |
| Linux | uint8 | cross | 4 | 13.61974 | 0.71179 | 19.135x | 1.046x |
| Linux | uint8 | even | 1 | 9.10840 | 1.05201 | 8.658x | 1.032x |
| Linux | uint8 | even | 4 | 9.10993 | 0.70420 | 12.937x | 1.037x |
| Linux | uint8 | identity | 1 | 0.03998 | 0.01914 | 2.089x | 1.038x |
| Linux | uint8 | identity | 4 | 0.03856 | 0.01817 | 2.122x | 1.049x |
| Linux | uint8 | tiny | 1 | 0.01772 | 0.01515 | 1.170x | 1.767x |
| Linux | uint8 | tiny | 4 | 0.01806 | 0.01535 | 1.177x | 1.726x |
| Linux | uint8 | one-dimensional3 | 1 | 3.93203 | 0.44975 | 8.743x | 1.078x |
| Linux | uint8 | one-dimensional3 | 4 | 4.77055 | 2.64841 | 1.801x | 1.017x |
| Linux | uint8 | one-dimensional31 | 1 | 11.42171 | 12.35858 | 0.924x | 9.232x |
| Linux | uint8 | one-dimensional31 | 4 | 11.45049 | 5.91058 | 1.937x | 5.984x |
| Linux | uint8 | three-dimensional | 1 | 19.05379 | 1.72909 | 11.020x | 10.819x |
| Linux | uint8 | three-dimensional | 4 | 19.09581 | 1.74289 | 10.956x | 10.712x |
| Linux | float32 | nearest | 1 | 32.45181 | 2.25008 | 14.423x | 1.027x |
| Linux | float32 | nearest | 4 | 32.55504 | 1.14122 | 28.527x | 1.072x |
| Linux | float32 | reflect | 1 | 32.39019 | 2.26207 | 14.319x | 1.026x |
| Linux | float32 | reflect | 4 | 32.45049 | 1.15480 | 28.101x | 1.060x |
| Linux | float32 | mirror | 1 | 32.49866 | 2.24415 | 14.482x | 1.034x |
| Linux | float32 | mirror | 4 | 32.49659 | 1.14895 | 28.284x | 1.062x |
| Linux | float32 | wrap | 1 | 32.49458 | 2.28042 | 14.249x | 1.022x |
| Linux | float32 | wrap | 4 | 32.45320 | 1.14971 | 28.227x | 1.070x |
| Linux | float32 | constant | 1 | 32.55423 | 2.26109 | 14.398x | 1.015x |
| Linux | float32 | constant | 4 | 32.48783 | 1.14064 | 28.482x | 1.078x |
| Linux | float32 | fortran | 1 | 33.02288 | 2.93997 | 11.232x | 1.019x |
| Linux | float32 | fortran | 4 | 32.99531 | 1.84175 | 17.915x | 1.029x |
| Linux | float32 | strided | 1 | 16.46200 | 1.23050 | 13.378x | 1.036x |
| Linux | float32 | strided | 4 | 16.46774 | 1.21535 | 13.550x | 1.045x |
| Linux | float32 | dense5 | 1 | 107.66702 | 8.45367 | 12.736x | 12.954x |
| Linux | float32 | dense5 | 4 | 107.74262 | 4.17577 | 25.802x | 7.758x |
| Linux | float32 | sparse7 | 1 | 32.81031 | 2.28174 | 14.380x | 1.027x |
| Linux | float32 | sparse7 | 4 | 32.81489 | 1.22546 | 26.778x | 1.062x |
| Linux | float32 | cross | 1 | 13.42684 | 1.69710 | 7.912x | 1.049x |
| Linux | float32 | cross | 4 | 13.44561 | 0.94329 | 14.254x | 1.088x |
| Linux | float32 | even | 1 | 8.98675 | 1.65010 | 5.446x | 1.038x |
| Linux | float32 | even | 4 | 8.97025 | 0.89612 | 10.010x | 1.092x |
| Linux | float32 | identity | 1 | 0.08681 | 0.05044 | 1.721x | 1.016x |
| Linux | float32 | identity | 4 | 0.08609 | 0.04986 | 1.726x | 1.006x |
| Linux | float32 | tiny | 1 | 0.02003 | 0.01720 | 1.165x | 1.660x |
| Linux | float32 | tiny | 4 | 0.01846 | 0.01616 | 1.142x | 1.733x |
| Linux | float32 | one-dimensional3 | 1 | 4.36113 | 1.71873 | 2.537x | 0.962x |
| Linux | float32 | one-dimensional3 | 4 | 4.37760 | 0.91431 | 4.788x | 1.063x |
| Linux | float32 | one-dimensional31 | 1 | 11.49050 | 12.22267 | 0.940x | 10.737x |
| Linux | float32 | one-dimensional31 | 4 | 11.51353 | 4.11111 | 2.801x | 9.370x |
| Linux | float32 | three-dimensional | 1 | 19.06132 | 1.83812 | 10.370x | 10.545x |
| Linux | float32 | three-dimensional | 4 | 19.08570 | 1.83675 | 10.391x | 10.583x |
| Linux | float64 | nearest | 1 | 32.88578 | 1.95342 | 16.835x | 2.138x |
| Linux | float64 | nearest | 4 | 32.27274 | 1.16544 | 27.692x | 1.116x |
| Linux | float64 | reflect | 1 | 32.37761 | 1.75246 | 18.476x | 1.116x |
| Linux | float64 | reflect | 4 | 32.94098 | 1.32197 | 24.918x | 2.474x |
| Linux | float64 | mirror | 1 | 32.24679 | 1.75224 | 18.403x | 1.112x |
| Linux | float64 | mirror | 4 | 32.30588 | 1.15707 | 27.920x | 1.090x |
| Linux | float64 | wrap | 1 | 32.29150 | 1.76829 | 18.261x | 1.121x |
| Linux | float64 | wrap | 4 | 32.40052 | 1.15780 | 27.984x | 1.100x |
| Linux | float64 | constant | 1 | 32.90497 | 1.96936 | 16.708x | 2.171x |
| Linux | float64 | constant | 4 | 32.30645 | 1.15807 | 27.897x | 1.114x |
| Linux | float64 | fortran | 1 | 32.66917 | 2.71350 | 12.040x | 1.079x |
| Linux | float64 | fortran | 4 | 32.62168 | 2.44517 | 13.341x | 1.764x |
| Linux | float64 | strided | 1 | 16.41082 | 1.01661 | 16.143x | 1.112x |
| Linux | float64 | strided | 4 | 16.39656 | 1.02337 | 16.022x | 1.101x |
| Linux | float64 | dense5 | 1 | 107.24833 | 7.92213 | 13.538x | 13.767x |
| Linux | float64 | dense5 | 4 | 107.96629 | 4.65525 | 23.192x | 7.217x |
| Linux | float64 | sparse7 | 1 | 32.56547 | 1.77610 | 18.335x | 1.099x |
| Linux | float64 | sparse7 | 4 | 32.69991 | 1.13891 | 28.711x | 1.105x |
| Linux | float64 | cross | 1 | 13.06867 | 1.20175 | 10.875x | 1.160x |
| Linux | float64 | cross | 4 | 13.07390 | 0.90041 | 14.520x | 1.185x |
| Linux | float64 | even | 1 | 9.25586 | 1.31422 | 7.043x | 2.733x |
| Linux | float64 | even | 4 | 8.72620 | 0.85040 | 10.261x | 1.198x |
| Linux | float64 | identity | 1 | 0.14050 | 0.08402 | 1.672x | 1.032x |
| Linux | float64 | identity | 4 | 0.14054 | 0.08640 | 1.627x | 0.989x |
| Linux | float64 | tiny | 1 | 0.01803 | 0.01690 | 1.067x | 1.725x |
| Linux | float64 | tiny | 4 | 0.01812 | 0.01609 | 1.126x | 1.679x |
| Linux | float64 | one-dimensional3 | 1 | 4.70418 | 1.17805 | 3.993x | 3.027x |
| Linux | float64 | one-dimensional3 | 4 | 4.43967 | 0.86056 | 5.159x | 1.194x |
| Linux | float64 | one-dimensional31 | 1 | 11.50320 | 12.55157 | 0.916x | 10.370x |
| Linux | float64 | one-dimensional31 | 4 | 12.11253 | 4.89918 | 2.472x | 8.083x |
| Linux | float64 | three-dimensional | 1 | 18.93244 | 1.72093 | 11.001x | 11.188x |
| Linux | float64 | three-dimensional | 4 | 18.87722 | 1.71601 | 11.001x | 11.194x |
| Windows raw | [JSON](median-compact-windows.json) | source `9c773062101107030765129a0127be80125e5df74eff15ef2e63f12a70fc440f` | | | | | |
| Windows | uint8 | nearest | 1 | 32.06000 | 2.85778 | 11.218x | 0.984x |
| Windows | uint8 | nearest | 4 | 32.18920 | 1.73506 | 18.552x | 1.003x |
| Windows | uint8 | reflect | 1 | 32.25400 | 2.91588 | 11.062x | 0.981x |
| Windows | uint8 | reflect | 4 | 32.23370 | 1.70405 | 18.916x | 1.012x |
| Windows | uint8 | mirror | 1 | 31.94260 | 2.90477 | 10.997x | 0.978x |
| Windows | uint8 | mirror | 4 | 32.24400 | 1.70550 | 18.906x | 1.003x |
| Windows | uint8 | wrap | 1 | 32.15440 | 2.92022 | 11.011x | 0.983x |
| Windows | uint8 | wrap | 4 | 33.31960 | 1.85620 | 17.950x | 1.066x |
| Windows | uint8 | constant | 1 | 32.52000 | 2.95950 | 10.988x | 1.081x |
| Windows | uint8 | constant | 4 | 32.19750 | 1.78325 | 18.056x | 1.014x |
| Windows | uint8 | fortran | 1 | 32.21580 | 3.32345 | 9.693x | 0.980x |
| Windows | uint8 | fortran | 4 | 32.28800 | 2.16708 | 14.899x | 0.995x |
| Windows | uint8 | strided | 1 | 16.35980 | 1.59407 | 10.263x | 1.019x |
| Windows | uint8 | strided | 4 | 16.31170 | 1.58159 | 10.313x | 1.038x |
| Windows | uint8 | dense5 | 1 | 114.89900 | 33.76990 | 3.402x | 3.178x |
| Windows | uint8 | dense5 | 4 | 110.63810 | 21.02140 | 5.263x | 2.756x |
| Windows | uint8 | sparse7 | 1 | 34.46520 | 2.94278 | 11.712x | 1.050x |
| Windows | uint8 | sparse7 | 4 | 34.70720 | 2.48006 | 13.995x | 1.003x |
| Windows | uint8 | cross | 1 | 14.76130 | 2.46884 | 5.979x | 0.992x |
| Windows | uint8 | cross | 4 | 14.95800 | 2.13328 | 7.012x | 1.001x |
| Windows | uint8 | even | 1 | 10.51820 | 2.81535 | 3.736x | 0.852x |
| Windows | uint8 | even | 4 | 10.46900 | 2.03457 | 5.146x | 1.032x |
| Windows | uint8 | identity | 1 | 0.04410 | 0.02127 | 2.074x | 1.053x |
| Windows | uint8 | identity | 4 | 0.04493 | 0.02074 | 2.167x | 1.029x |
| Windows | uint8 | tiny | 1 | 0.02172 | 0.01491 | 1.456x | 2.118x |
| Windows | uint8 | tiny | 4 | 0.01666 | 0.01516 | 1.099x | 1.759x |
| Windows | uint8 | one-dimensional3 | 1 | 5.13015 | 2.27497 | 2.255x | 1.054x |
| Windows | uint8 | one-dimensional3 | 4 | 5.41000 | 2.59623 | 2.084x | 0.962x |
| Windows | uint8 | one-dimensional31 | 1 | 11.46590 | 12.38460 | 0.926x | 9.296x |
| Windows | uint8 | one-dimensional31 | 4 | 12.11640 | 8.33230 | 1.454x | 7.720x |
| Windows | uint8 | three-dimensional | 1 | 19.58510 | 6.92850 | 2.827x | 2.768x |
| Windows | uint8 | three-dimensional | 4 | 19.82110 | 6.89595 | 2.874x | 2.826x |
| Windows | float32 | nearest | 1 | 32.97530 | 3.54273 | 9.308x | 1.099x |
| Windows | float32 | nearest | 4 | 35.35260 | 2.46707 | 14.330x | 1.451x |
| Windows | float32 | reflect | 1 | 33.92450 | 3.55127 | 9.553x | 1.023x |
| Windows | float32 | reflect | 4 | 33.59650 | 2.65876 | 12.636x | 1.394x |
| Windows | float32 | mirror | 1 | 32.42660 | 3.69037 | 8.787x | 1.082x |
| Windows | float32 | mirror | 4 | 33.25630 | 3.44475 | 9.654x | 1.003x |
| Windows | float32 | wrap | 1 | 36.54260 | 4.01748 | 9.096x | 1.059x |
| Windows | float32 | wrap | 4 | 35.20310 | 3.34565 | 10.522x | 1.119x |
| Windows | float32 | constant | 1 | 33.97440 | 3.80440 | 8.930x | 1.031x |
| Windows | float32 | constant | 4 | 36.54870 | 3.51242 | 10.406x | 1.069x |
| Windows | float32 | fortran | 1 | 35.06480 | 4.91870 | 7.129x | 1.043x |
| Windows | float32 | fortran | 4 | 34.42720 | 4.22405 | 8.150x | 1.031x |
| Windows | float32 | strided | 1 | 17.33050 | 1.98370 | 8.736x | 1.140x |
| Windows | float32 | strided | 4 | 16.72220 | 1.90953 | 8.757x | 1.128x |
| Windows | float32 | dense5 | 1 | 111.20090 | 34.27750 | 3.244x | 3.172x |
| Windows | float32 | dense5 | 4 | 107.65210 | 23.81240 | 4.521x | 2.433x |
| Windows | float32 | sparse7 | 1 | 32.51760 | 3.44787 | 9.431x | 1.061x |
| Windows | float32 | sparse7 | 4 | 32.34790 | 2.44704 | 13.219x | 1.103x |
| Windows | float32 | cross | 1 | 13.53840 | 2.83716 | 4.772x | 1.073x |
| Windows | float32 | cross | 4 | 13.50200 | 2.17792 | 6.199x | 1.133x |
| Windows | float32 | even | 1 | 9.15110 | 2.76988 | 3.304x | 1.071x |
| Windows | float32 | even | 4 | 9.10020 | 2.17000 | 4.194x | 1.133x |
| Windows | float32 | identity | 1 | 0.29802 | 0.27029 | 1.103x | 1.018x |
| Windows | float32 | identity | 4 | 0.30338 | 0.27496 | 1.103x | 1.020x |
| Windows | float32 | tiny | 1 | 0.01640 | 0.01676 | 0.979x | 1.709x |
| Windows | float32 | tiny | 4 | 0.01667 | 0.01663 | 1.002x | 1.710x |
| Windows | float32 | one-dimensional3 | 1 | 4.69343 | 2.84258 | 1.651x | 1.029x |
| Windows | float32 | one-dimensional3 | 4 | 4.70170 | 2.42538 | 1.939x | 1.067x |
| Windows | float32 | one-dimensional31 | 1 | 11.94830 | 13.71750 | 0.871x | 9.331x |
| Windows | float32 | one-dimensional31 | 4 | 11.83380 | 4.89445 | 2.418x | 7.695x |
| Windows | float32 | three-dimensional | 1 | 18.94340 | 5.99470 | 3.160x | 3.134x |
| Windows | float32 | three-dimensional | 4 | 19.01010 | 6.01280 | 3.162x | 3.132x |
| Windows | float64 | nearest | 1 | 32.10140 | 3.18110 | 10.091x | 1.104x |
| Windows | float64 | nearest | 4 | 32.12590 | 2.22680 | 14.427x | 1.234x |
| Windows | float64 | reflect | 1 | 32.10790 | 3.15572 | 10.174x | 1.113x |
| Windows | float64 | reflect | 4 | 32.11550 | 2.20308 | 14.578x | 1.216x |
| Windows | float64 | mirror | 1 | 31.99590 | 3.15468 | 10.142x | 1.097x |
| Windows | float64 | mirror | 4 | 32.06870 | 2.20700 | 14.530x | 1.228x |
| Windows | float64 | wrap | 1 | 32.12130 | 3.21042 | 10.005x | 1.089x |
| Windows | float64 | wrap | 4 | 34.44720 | 3.15838 | 10.907x | 0.961x |
| Windows | float64 | constant | 1 | 38.35660 | 3.29497 | 11.641x | 1.126x |
| Windows | float64 | constant | 4 | 32.59880 | 2.22610 | 14.644x | 1.232x |
| Windows | float64 | fortran | 1 | 33.66850 | 5.26810 | 6.391x | 1.054x |
| Windows | float64 | fortran | 4 | 33.73530 | 4.34763 | 7.759x | 1.111x |
| Windows | float64 | strided | 1 | 16.24070 | 2.03570 | 7.978x | 1.090x |
| Windows | float64 | strided | 4 | 16.20400 | 2.00621 | 8.077x | 1.120x |
| Windows | float64 | dense5 | 1 | 107.16800 | 33.95680 | 3.156x | 3.179x |
| Windows | float64 | dense5 | 4 | 109.01390 | 15.92640 | 6.845x | 2.035x |
| Windows | float64 | sparse7 | 1 | 32.29520 | 3.18800 | 10.130x | 1.101x |
| Windows | float64 | sparse7 | 4 | 32.31580 | 2.24410 | 14.400x | 1.228x |
| Windows | float64 | cross | 1 | 13.53620 | 2.62712 | 5.152x | 1.110x |
| Windows | float64 | cross | 4 | 13.67320 | 2.07568 | 6.587x | 1.228x |
| Windows | float64 | even | 1 | 9.09220 | 2.50768 | 3.626x | 1.133x |
| Windows | float64 | even | 4 | 9.32450 | 2.01423 | 4.629x | 1.241x |
| Windows | float64 | identity | 1 | 0.47074 | 0.44611 | 1.055x | 1.009x |
| Windows | float64 | identity | 4 | 0.47176 | 0.44121 | 1.069x | 1.016x |
| Windows | float64 | tiny | 1 | 0.01674 | 0.01713 | 0.977x | 1.801x |
| Windows | float64 | tiny | 4 | 0.01671 | 0.01687 | 0.990x | 1.775x |
| Windows | float64 | one-dimensional3 | 1 | 4.88983 | 2.51136 | 1.947x | 1.137x |
| Windows | float64 | one-dimensional3 | 4 | 4.86715 | 1.97062 | 2.470x | 1.231x |
| Windows | float64 | one-dimensional31 | 1 | 12.15280 | 13.87990 | 0.876x | 9.246x |
| Windows | float64 | one-dimensional31 | 4 | 12.17640 | 5.09155 | 2.391x | 7.621x |
| Windows | float64 | three-dimensional | 1 | 18.92080 | 5.94680 | 3.182x | 3.164x |
| Windows | float64 | three-dimensional | 4 | 18.98740 | 5.93845 | 3.197x | 3.180x |

## Streaming inputs: `256fd39`

[Paired benchmark run](https://github.com/bipxx/scikit-imagexx/actions/runs/39814966359), compared with exact compact source `6b0dd2a` from run `39813349869`. Product fingerprint: `159db3002e7a06e77642d38fb6edf3bced34ac7085f0b297ebce5e8f3134b838`. The candidate streams original 1-D inputs without constructing a full padded copy. Scoped validation passes 9,396 Windows and 5,200 minimum-NumPy cases per SIMD mode, plus all five native configurations (2/2 each). [Full streaming-source CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39814944869) passes all three jobs: Linux 59,673 and Windows 59,671 plus two skips per SIMD mode; native normal/dynamic/portable checks pass 2/2 on both platforms. These checks predate the signed-zero audit below.

Linux: one-worker geometric mean **8.553786x upstream/native**, **0.996603x compact/current**. [All raw samples](median-stream-linux.json).

Windows: one-worker geometric mean **4.739621x upstream/native**, **1.022931x compact/current**. [All raw samples](median-stream-windows.json).

The streaming change is broadly neutral against compact. Long 1-D uint8 improves 1.057x Linux / 1.068x Windows against compact, but Linux float32 is 0.974x. Long 1-D upstream/native ratios remain 0.840-0.910x Linux and 0.898-1.038x Windows; Linux tiny floating cases are 0.972-0.990x. These adverse samples remain open optimization targets.

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | compact/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | nearest | 1 | 29.35908 | 1.34606 | 21.811x | 1.004x |
| Linux | uint8 | nearest | 4 | 33.09054 | 1.05930 | 31.238x | 1.016x |
| Linux | uint8 | reflect | 1 | 29.03978 | 1.33844 | 21.697x | 1.015x |
| Linux | uint8 | reflect | 4 | 29.32822 | 1.02271 | 28.677x | 1.004x |
| Linux | uint8 | mirror | 1 | 29.51666 | 1.33490 | 22.112x | 0.992x |
| Linux | uint8 | mirror | 4 | 29.72956 | 1.06390 | 27.944x | 0.982x |
| Linux | uint8 | wrap | 1 | 32.93430 | 1.49897 | 21.971x | 1.009x |
| Linux | uint8 | wrap | 4 | 29.77663 | 1.04976 | 28.365x | 0.997x |
| Linux | uint8 | constant | 1 | 29.45611 | 1.34798 | 21.852x | 0.986x |
| Linux | uint8 | constant | 4 | 29.76370 | 1.01579 | 29.301x | 1.031x |
| Linux | uint8 | fortran | 1 | 29.24836 | 1.63422 | 17.897x | 1.008x |
| Linux | uint8 | fortran | 4 | 29.16777 | 1.27237 | 22.924x | 1.017x |
| Linux | uint8 | strided | 1 | 14.66544 | 0.70029 | 20.942x | 1.009x |
| Linux | uint8 | strided | 4 | 14.69715 | 0.70479 | 20.853x | 0.993x |
| Linux | uint8 | dense5 | 1 | 85.38487 | 7.70794 | 11.078x | 0.994x |
| Linux | uint8 | dense5 | 4 | 85.58872 | 3.64075 | 23.509x | 1.014x |
| Linux | uint8 | sparse7 | 1 | 29.70362 | 1.34197 | 22.134x | 0.995x |
| Linux | uint8 | sparse7 | 4 | 29.63325 | 1.00272 | 29.553x | 1.040x |
| Linux | uint8 | cross | 1 | 11.09873 | 0.87942 | 12.620x | 1.005x |
| Linux | uint8 | cross | 4 | 11.28574 | 0.87957 | 12.831x | 0.992x |
| Linux | uint8 | even | 1 | 7.29388 | 0.82331 | 8.859x | 1.013x |
| Linux | uint8 | even | 4 | 7.47731 | 0.81070 | 9.223x | 1.000x |
| Linux | uint8 | identity | 1 | 0.01857 | 0.00876 | 2.119x | 0.994x |
| Linux | uint8 | identity | 4 | 0.01834 | 0.00918 | 1.997x | 0.976x |
| Linux | uint8 | tiny | 1 | 0.00817 | 0.00788 | 1.037x | 0.999x |
| Linux | uint8 | tiny | 4 | 0.00830 | 0.00795 | 1.043x | 1.012x |
| Linux | uint8 | one-dimensional3 | 1 | 3.28651 | 0.75735 | 4.339x | 0.950x |
| Linux | uint8 | one-dimensional3 | 4 | 3.34710 | 0.97009 | 3.450x | 0.979x |
| Linux | uint8 | one-dimensional31 | 1 | 8.19417 | 9.00045 | 0.910x | 1.057x |
| Linux | uint8 | one-dimensional31 | 4 | 8.16402 | 3.41953 | 2.387x | 1.124x |
| Linux | uint8 | three-dimensional | 1 | 14.77587 | 1.57883 | 9.359x | 1.003x |
| Linux | uint8 | three-dimensional | 4 | 14.88762 | 1.56043 | 9.541x | 1.009x |
| Linux | float32 | nearest | 1 | 28.23946 | 1.47921 | 19.091x | 0.994x |
| Linux | float32 | nearest | 4 | 28.43594 | 1.22664 | 23.182x | 0.987x |
| Linux | float32 | reflect | 1 | 28.18729 | 1.46934 | 19.184x | 0.977x |
| Linux | float32 | reflect | 4 | 28.30993 | 1.19977 | 23.596x | 0.997x |
| Linux | float32 | mirror | 1 | 28.11013 | 1.45392 | 19.334x | 0.988x |
| Linux | float32 | mirror | 4 | 28.23577 | 1.18351 | 23.858x | 1.005x |
| Linux | float32 | wrap | 1 | 28.18596 | 1.47145 | 19.155x | 0.983x |
| Linux | float32 | wrap | 4 | 28.55903 | 1.22792 | 23.258x | 0.983x |
| Linux | float32 | constant | 1 | 28.11945 | 1.48079 | 18.989x | 0.990x |
| Linux | float32 | constant | 4 | 29.22198 | 1.21822 | 23.987x | 1.033x |
| Linux | float32 | fortran | 1 | 31.95981 | 1.91467 | 16.692x | 1.003x |
| Linux | float32 | fortran | 4 | 29.11865 | 1.57344 | 18.506x | 0.998x |
| Linux | float32 | strided | 1 | 14.31969 | 0.80199 | 17.855x | 1.008x |
| Linux | float32 | strided | 4 | 14.27873 | 0.79902 | 17.870x | 0.992x |
| Linux | float32 | dense5 | 1 | 81.54080 | 7.94915 | 10.258x | 1.002x |
| Linux | float32 | dense5 | 4 | 82.17158 | 4.01604 | 20.461x | 0.990x |
| Linux | float32 | sparse7 | 1 | 28.39370 | 1.47198 | 19.289x | 1.002x |
| Linux | float32 | sparse7 | 4 | 29.02865 | 1.26439 | 22.959x | 0.983x |
| Linux | float32 | cross | 1 | 10.98934 | 1.05738 | 10.393x | 0.979x |
| Linux | float32 | cross | 4 | 11.33051 | 1.03695 | 10.927x | 0.986x |
| Linux | float32 | even | 1 | 7.68961 | 1.01020 | 7.612x | 1.016x |
| Linux | float32 | even | 4 | 7.32810 | 1.02688 | 7.136x | 0.982x |
| Linux | float32 | identity | 1 | 0.07438 | 0.04398 | 1.691x | 1.015x |
| Linux | float32 | identity | 4 | 0.07344 | 0.04555 | 1.612x | 1.014x |
| Linux | float32 | tiny | 1 | 0.00862 | 0.00887 | 0.972x | 0.991x |
| Linux | float32 | tiny | 4 | 0.00864 | 0.00880 | 0.982x | 1.010x |
| Linux | float32 | one-dimensional3 | 1 | 3.12355 | 0.86146 | 3.626x | 0.985x |
| Linux | float32 | one-dimensional3 | 4 | 3.13726 | 0.96233 | 3.260x | 0.935x |
| Linux | float32 | one-dimensional31 | 1 | 8.12075 | 9.67167 | 0.840x | 0.973x |
| Linux | float32 | one-dimensional31 | 4 | 8.16917 | 3.35411 | 2.436x | 1.024x |
| Linux | float32 | three-dimensional | 1 | 14.47487 | 1.60468 | 9.020x | 0.988x |
| Linux | float32 | three-dimensional | 4 | 16.14943 | 1.72201 | 9.378x | 1.005x |
| Linux | float64 | nearest | 1 | 28.73120 | 1.59219 | 18.045x | 0.973x |
| Linux | float64 | nearest | 4 | 28.73061 | 1.26841 | 22.651x | 0.974x |
| Linux | float64 | reflect | 1 | 28.62308 | 1.61408 | 17.733x | 1.018x |
| Linux | float64 | reflect | 4 | 29.01052 | 1.28925 | 22.502x | 0.978x |
| Linux | float64 | mirror | 1 | 28.88635 | 1.67170 | 17.280x | 0.968x |
| Linux | float64 | mirror | 4 | 28.85663 | 1.32305 | 21.811x | 0.949x |
| Linux | float64 | wrap | 1 | 28.99662 | 1.62605 | 17.833x | 0.997x |
| Linux | float64 | wrap | 4 | 29.03024 | 1.29446 | 22.427x | 0.962x |
| Linux | float64 | constant | 1 | 28.62268 | 1.58766 | 18.028x | 0.974x |
| Linux | float64 | constant | 4 | 29.14655 | 1.30086 | 22.406x | 0.966x |
| Linux | float64 | fortran | 1 | 34.97071 | 2.57356 | 13.588x | 0.996x |
| Linux | float64 | fortran | 4 | 34.41953 | 2.13015 | 16.158x | 0.961x |
| Linux | float64 | strided | 1 | 16.08042 | 0.90892 | 17.692x | 0.985x |
| Linux | float64 | strided | 4 | 16.11087 | 0.92432 | 17.430x | 0.957x |
| Linux | float64 | dense5 | 1 | 82.96785 | 8.33888 | 9.950x | 0.990x |
| Linux | float64 | dense5 | 4 | 82.84926 | 4.20997 | 19.679x | 0.986x |
| Linux | float64 | sparse7 | 1 | 28.98031 | 1.62301 | 17.856x | 0.989x |
| Linux | float64 | sparse7 | 4 | 29.19251 | 1.32110 | 22.097x | 0.942x |
| Linux | float64 | cross | 1 | 10.73552 | 1.10345 | 9.729x | 0.977x |
| Linux | float64 | cross | 4 | 10.87501 | 1.08665 | 10.008x | 0.961x |
| Linux | float64 | even | 1 | 7.37750 | 1.02533 | 7.195x | 0.968x |
| Linux | float64 | even | 4 | 7.41942 | 1.02287 | 7.254x | 0.944x |
| Linux | float64 | identity | 1 | 0.24086 | 0.15805 | 1.524x | 0.999x |
| Linux | float64 | identity | 4 | 0.24472 | 0.15877 | 1.541x | 1.002x |
| Linux | float64 | tiny | 1 | 0.00848 | 0.00857 | 0.989x | 0.995x |
| Linux | float64 | tiny | 4 | 0.00851 | 0.00879 | 0.969x | 0.987x |
| Linux | float64 | one-dimensional3 | 1 | 3.55607 | 0.92517 | 3.844x | 1.029x |
| Linux | float64 | one-dimensional3 | 4 | 3.29104 | 0.97949 | 3.360x | 0.941x |
| Linux | float64 | one-dimensional31 | 1 | 8.50631 | 9.68831 | 0.878x | 1.053x |
| Linux | float64 | one-dimensional31 | 4 | 8.37119 | 3.55693 | 2.353x | 1.105x |
| Linux | float64 | three-dimensional | 1 | 14.33412 | 1.59207 | 9.003x | 1.008x |
| Linux | float64 | three-dimensional | 4 | 14.54126 | 1.59674 | 9.107x | 1.007x |
| Windows | uint8 | nearest | 1 | 43.48000 | 3.36780 | 12.911x | 1.036x |
| Windows | uint8 | nearest | 4 | 40.96950 | 2.19274 | 18.684x | 1.046x |
| Windows | uint8 | reflect | 1 | 40.96380 | 3.28483 | 12.471x | 1.020x |
| Windows | uint8 | reflect | 4 | 43.46100 | 2.79927 | 15.526x | 1.026x |
| Windows | uint8 | mirror | 1 | 44.99340 | 3.59017 | 12.532x | 1.063x |
| Windows | uint8 | mirror | 4 | 43.51570 | 2.82282 | 15.416x | 1.025x |
| Windows | uint8 | wrap | 1 | 44.27990 | 3.55037 | 12.472x | 0.975x |
| Windows | uint8 | wrap | 4 | 40.55810 | 2.83543 | 14.304x | 0.827x |
| Windows | uint8 | constant | 1 | 43.60900 | 3.39595 | 12.841x | 1.075x |
| Windows | uint8 | constant | 4 | 47.57980 | 2.92488 | 16.267x | 0.983x |
| Windows | uint8 | fortran | 1 | 41.97860 | 3.68962 | 11.377x | 1.045x |
| Windows | uint8 | fortran | 4 | 42.22780 | 3.21950 | 13.116x | 1.005x |
| Windows | uint8 | strided | 1 | 21.57370 | 1.88696 | 11.433x | 0.996x |
| Windows | uint8 | strided | 4 | 20.24200 | 1.72487 | 11.735x | 1.061x |
| Windows | uint8 | dense5 | 1 | 120.43280 | 31.54400 | 3.818x | 1.064x |
| Windows | uint8 | dense5 | 4 | 115.63340 | 19.28500 | 5.996x | 0.921x |
| Windows | uint8 | sparse7 | 1 | 41.26610 | 3.18728 | 12.947x | 1.060x |
| Windows | uint8 | sparse7 | 4 | 40.29740 | 2.07880 | 19.385x | 1.012x |
| Windows | uint8 | cross | 1 | 14.57610 | 2.52992 | 5.761x | 1.050x |
| Windows | uint8 | cross | 4 | 14.40070 | 1.76246 | 8.171x | 1.017x |
| Windows | uint8 | even | 1 | 9.47240 | 2.44530 | 3.874x | 1.010x |
| Windows | uint8 | even | 4 | 9.43910 | 1.69466 | 5.570x | 1.009x |
| Windows | uint8 | identity | 1 | 0.02831 | 0.01275 | 2.220x | 1.025x |
| Windows | uint8 | identity | 4 | 0.02792 | 0.01239 | 2.253x | 1.012x |
| Windows | uint8 | tiny | 1 | 0.01260 | 0.01191 | 1.057x | 1.064x |
| Windows | uint8 | tiny | 4 | 0.01276 | 0.01188 | 1.075x | 1.005x |
| Windows | uint8 | one-dimensional3 | 1 | 5.64390 | 3.18918 | 1.770x | 0.999x |
| Windows | uint8 | one-dimensional3 | 4 | 5.64965 | 2.78420 | 2.029x | 0.989x |
| Windows | uint8 | one-dimensional31 | 1 | 12.50100 | 12.04510 | 1.038x | 1.068x |
| Windows | uint8 | one-dimensional31 | 4 | 12.46120 | 4.79720 | 2.598x | 1.223x |
| Windows | uint8 | three-dimensional | 1 | 20.12680 | 5.47720 | 3.675x | 1.014x |
| Windows | uint8 | three-dimensional | 4 | 20.16880 | 5.48655 | 3.676x | 1.014x |
| Windows | float32 | nearest | 1 | 40.94970 | 3.94320 | 10.385x | 1.039x |
| Windows | float32 | nearest | 4 | 40.91800 | 2.92817 | 13.974x | 1.042x |
| Windows | float32 | reflect | 1 | 41.06720 | 3.93860 | 10.427x | 1.041x |
| Windows | float32 | reflect | 4 | 40.91140 | 2.94790 | 13.878x | 1.033x |
| Windows | float32 | mirror | 1 | 41.06200 | 3.94327 | 10.413x | 1.033x |
| Windows | float32 | mirror | 4 | 40.75290 | 3.01250 | 13.528x | 1.015x |
| Windows | float32 | wrap | 1 | 41.13580 | 3.99463 | 10.298x | 1.033x |
| Windows | float32 | wrap | 4 | 41.51810 | 3.46177 | 11.993x | 1.018x |
| Windows | float32 | constant | 1 | 41.60330 | 3.91347 | 10.631x | 1.036x |
| Windows | float32 | constant | 4 | 41.38410 | 3.07470 | 13.460x | 1.054x |
| Windows | float32 | fortran | 1 | 41.60030 | 4.77873 | 8.705x | 1.020x |
| Windows | float32 | fortran | 4 | 41.75930 | 4.09987 | 10.186x | 1.023x |
| Windows | float32 | strided | 1 | 21.03960 | 2.07277 | 10.150x | 1.031x |
| Windows | float32 | strided | 4 | 21.49470 | 2.16178 | 9.943x | 1.024x |
| Windows | float32 | dense5 | 1 | 119.66880 | 36.79470 | 3.252x | 0.943x |
| Windows | float32 | dense5 | 4 | 125.85030 | 29.80780 | 4.222x | 0.925x |
| Windows | float32 | sparse7 | 1 | 43.84510 | 4.22027 | 10.389x | 1.030x |
| Windows | float32 | sparse7 | 4 | 43.87680 | 3.77783 | 11.614x | 1.042x |
| Windows | float32 | cross | 1 | 15.86900 | 3.62377 | 4.379x | 1.027x |
| Windows | float32 | cross | 4 | 15.83620 | 3.36340 | 4.708x | 1.059x |
| Windows | float32 | even | 1 | 10.33870 | 3.39208 | 3.048x | 1.083x |
| Windows | float32 | even | 4 | 10.01300 | 3.39880 | 2.946x | 0.991x |
| Windows | float32 | identity | 1 | 0.48813 | 0.45879 | 1.064x | 1.000x |
| Windows | float32 | identity | 4 | 0.50796 | 0.49732 | 1.021x | 0.946x |
| Windows | float32 | tiny | 1 | 0.01432 | 0.01430 | 1.001x | 1.040x |
| Windows | float32 | tiny | 4 | 0.01339 | 0.01318 | 1.016x | 1.082x |
| Windows | float32 | one-dimensional3 | 1 | 4.69043 | 3.34623 | 1.402x | 1.016x |
| Windows | float32 | one-dimensional3 | 4 | 4.67773 | 3.29110 | 1.421x | 1.041x |
| Windows | float32 | one-dimensional31 | 1 | 11.63170 | 12.94880 | 0.898x | 0.992x |
| Windows | float32 | one-dimensional31 | 4 | 11.43410 | 8.18230 | 1.397x | 1.051x |
| Windows | float32 | three-dimensional | 1 | 20.54980 | 5.61005 | 3.663x | 1.069x |
| Windows | float32 | three-dimensional | 4 | 21.28720 | 5.70370 | 3.732x | 0.987x |
| Windows | float64 | nearest | 1 | 42.01280 | 4.31563 | 9.735x | 0.952x |
| Windows | float64 | nearest | 4 | 42.01440 | 3.95360 | 10.627x | 0.963x |
| Windows | float64 | reflect | 1 | 43.14320 | 4.34593 | 9.927x | 0.997x |
| Windows | float64 | reflect | 4 | 43.68530 | 3.88480 | 11.245x | 0.961x |
| Windows | float64 | mirror | 1 | 43.12840 | 4.47680 | 9.634x | 0.939x |
| Windows | float64 | mirror | 4 | 43.07630 | 3.83098 | 11.244x | 1.008x |
| Windows | float64 | wrap | 1 | 42.65030 | 4.36570 | 9.769x | 1.043x |
| Windows | float64 | wrap | 4 | 43.05170 | 3.79137 | 11.355x | 1.006x |
| Windows | float64 | constant | 1 | 43.43830 | 4.36330 | 9.955x | 0.993x |
| Windows | float64 | constant | 4 | 44.74010 | 3.77535 | 11.851x | 1.023x |
| Windows | float64 | fortran | 1 | 45.38920 | 5.50265 | 8.249x | 1.009x |
| Windows | float64 | fortran | 4 | 42.68190 | 5.20053 | 8.207x | 0.958x |
| Windows | float64 | strided | 1 | 20.89750 | 2.65982 | 7.857x | 1.008x |
| Windows | float64 | strided | 4 | 22.12480 | 2.66776 | 8.293x | 1.076x |
| Windows | float64 | dense5 | 1 | 122.15620 | 36.11680 | 3.382x | 1.064x |
| Windows | float64 | dense5 | 4 | 120.75120 | 31.79750 | 3.798x | 1.028x |
| Windows | float64 | sparse7 | 1 | 41.25790 | 4.04887 | 10.190x | 1.007x |
| Windows | float64 | sparse7 | 4 | 41.02580 | 3.00955 | 13.632x | 0.975x |
| Windows | float64 | cross | 1 | 15.15420 | 3.35665 | 4.515x | 1.022x |
| Windows | float64 | cross | 4 | 15.23300 | 2.70492 | 5.632x | 0.974x |
| Windows | float64 | even | 1 | 10.22920 | 3.32888 | 3.073x | 1.001x |
| Windows | float64 | even | 4 | 10.00400 | 2.65218 | 3.772x | 0.971x |
| Windows | float64 | identity | 1 | 0.73962 | 0.71593 | 1.033x | 0.998x |
| Windows | float64 | identity | 4 | 0.74119 | 0.71056 | 1.043x | 1.004x |
| Windows | float64 | tiny | 1 | 0.01346 | 0.01280 | 1.052x | 1.013x |
| Windows | float64 | tiny | 4 | 0.01254 | 0.01272 | 0.986x | 0.996x |
| Windows | float64 | one-dimensional3 | 1 | 4.93900 | 3.29533 | 1.499x | 1.015x |
| Windows | float64 | one-dimensional3 | 4 | 4.89240 | 2.65156 | 1.845x | 0.977x |
| Windows | float64 | one-dimensional31 | 1 | 11.71050 | 12.71910 | 0.921x | 1.062x |
| Windows | float64 | one-dimensional31 | 4 | 11.70540 | 4.94230 | 2.368x | 1.164x |
| Windows | float64 | three-dimensional | 1 | 20.12110 | 5.51475 | 3.649x | 1.006x |
| Windows | float64 | three-dimensional | 4 | 20.26330 | 5.54455 | 3.655x | 0.995x |

## Hole-based heap repair: `0638289`

[Paired run](https://github.com/bipxx/scikit-imagexx/actions/runs/39817335611), against the exact streaming package `256fd39` from run `39814966359`. Product fingerprint: `32c4d2efd95cc1208453fef641645d9a8f90791c4046ebe76d988585a3b09ca3`. Heap repair retains the changed item and writes one moved entry per hop. Local checks pass 7,449 Windows cases and 5,296 minimum-NumPy cases in each SIMD mode, all five native configurations (2/2), plus 471 namespace/inventory checks per runtime. This is an experiment, not a retained median port.

Linux: one-worker mean **8.044099x upstream/native**, **1.022302x streaming/current**. [Every raw sample](median-hole-linux.json).

Windows: one-worker mean **4.589960x upstream/native**, **1.001968x streaming/current**. [Every raw sample](median-hole-windows.json).

Long-window float32/64 improves roughly 5-6% against streaming on both OSes, but uint8 remains 0.931x upstream on Linux and long floating windows are 0.937-0.955x on Windows. Windows tiny float32 is 0.996x upstream / 0.966x streaming. A subsequent sign-bit audit also found that all previous median candidates choose different signed-zero representatives in fast 1-D paths. Timed inputs here do not contain negative zero; these measurements therefore do not validate signed-zero parity. A native tie-order correction and sign-aware benchmark cases are being checked separately.

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | streaming/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | nearest | 1 | 32.81614 | 1.74416 | 18.815x | 1.004x |
| Linux | uint8 | nearest | 4 | 32.67809 | 1.00797 | 32.420x | 1.004x |
| Linux | uint8 | reflect | 1 | 32.70358 | 1.74299 | 18.763x | 0.996x |
| Linux | uint8 | reflect | 4 | 32.70918 | 1.03126 | 31.718x | 0.989x |
| Linux | uint8 | mirror | 1 | 32.72135 | 1.74359 | 18.767x | 0.997x |
| Linux | uint8 | mirror | 4 | 32.76045 | 1.03179 | 31.751x | 1.013x |
| Linux | uint8 | wrap | 1 | 32.77877 | 1.78932 | 18.319x | 0.980x |
| Linux | uint8 | wrap | 4 | 32.68658 | 1.02595 | 31.860x | 0.980x |
| Linux | uint8 | constant | 1 | 32.67731 | 1.74010 | 18.779x | 0.989x |
| Linux | uint8 | constant | 4 | 32.77555 | 1.02280 | 32.045x | 0.983x |
| Linux | uint8 | fortran | 1 | 32.88888 | 2.11186 | 15.573x | 0.985x |
| Linux | uint8 | fortran | 4 | 33.06042 | 1.41576 | 23.352x | 0.980x |
| Linux | uint8 | strided | 1 | 16.65810 | 0.98512 | 16.910x | 0.996x |
| Linux | uint8 | strided | 4 | 16.73886 | 0.99540 | 16.816x | 0.984x |
| Linux | uint8 | dense5 | 1 | 107.65605 | 7.82999 | 13.749x | 1.010x |
| Linux | uint8 | dense5 | 4 | 107.65162 | 4.02704 | 26.732x | 0.992x |
| Linux | uint8 | sparse7 | 1 | 32.99011 | 1.75364 | 18.812x | 0.993x |
| Linux | uint8 | sparse7 | 4 | 33.03083 | 1.03957 | 31.774x | 1.005x |
| Linux | uint8 | cross | 1 | 13.69275 | 1.19449 | 11.463x | 1.004x |
| Linux | uint8 | cross | 4 | 13.72539 | 0.77854 | 17.630x | 0.997x |
| Linux | uint8 | even | 1 | 9.40003 | 1.09406 | 8.592x | 1.025x |
| Linux | uint8 | even | 4 | 9.31698 | 0.78648 | 11.846x | 0.924x |
| Linux | uint8 | identity | 1 | 0.04127 | 0.02044 | 2.019x | 0.982x |
| Linux | uint8 | identity | 4 | 0.03956 | 0.01998 | 1.980x | 1.011x |
| Linux | uint8 | tiny | 1 | 0.01813 | 0.01543 | 1.175x | 1.009x |
| Linux | uint8 | tiny | 4 | 0.01855 | 0.01528 | 1.214x | 1.050x |
| Linux | uint8 | one-dimensional3 | 1 | 3.95197 | 0.50486 | 7.828x | 1.004x |
| Linux | uint8 | one-dimensional3 | 4 | 4.02852 | 0.52884 | 7.618x | 1.010x |
| Linux | uint8 | one-dimensional31 | 1 | 10.14128 | 10.89667 | 0.931x | 0.999x |
| Linux | uint8 | one-dimensional31 | 4 | 10.13590 | 3.85254 | 2.631x | 0.987x |
| Linux | uint8 | three-dimensional | 1 | 19.03712 | 1.68582 | 11.292x | 1.042x |
| Linux | uint8 | three-dimensional | 4 | 19.04908 | 1.68700 | 11.292x | 1.039x |
| Linux | float32 | nearest | 1 | 32.52466 | 2.35675 | 13.801x | 1.033x |
| Linux | float32 | nearest | 4 | 32.57214 | 1.25496 | 25.955x | 1.062x |
| Linux | float32 | reflect | 1 | 32.49677 | 2.34365 | 13.866x | 1.041x |
| Linux | float32 | reflect | 4 | 32.54618 | 1.25263 | 25.982x | 1.048x |
| Linux | float32 | mirror | 1 | 32.48022 | 2.38329 | 13.628x | 1.029x |
| Linux | float32 | mirror | 4 | 32.53475 | 1.24663 | 26.098x | 1.056x |
| Linux | float32 | wrap | 1 | 32.52342 | 2.38536 | 13.635x | 1.027x |
| Linux | float32 | wrap | 4 | 32.56836 | 1.25644 | 25.921x | 1.054x |
| Linux | float32 | constant | 1 | 32.50045 | 2.33768 | 13.903x | 1.040x |
| Linux | float32 | constant | 4 | 32.50370 | 1.26736 | 25.647x | 1.046x |
| Linux | float32 | fortran | 1 | 32.96149 | 2.96282 | 11.125x | 1.031x |
| Linux | float32 | fortran | 4 | 32.91935 | 1.95774 | 16.815x | 1.000x |
| Linux | float32 | strided | 1 | 16.47592 | 1.27832 | 12.889x | 1.023x |
| Linux | float32 | strided | 4 | 16.58450 | 1.26256 | 13.136x | 1.048x |
| Linux | float32 | dense5 | 1 | 107.76764 | 8.44684 | 12.758x | 1.005x |
| Linux | float32 | dense5 | 4 | 107.61244 | 4.20585 | 25.586x | 1.005x |
| Linux | float32 | sparse7 | 1 | 32.73988 | 2.34588 | 13.956x | 1.039x |
| Linux | float32 | sparse7 | 4 | 32.74191 | 1.23134 | 26.590x | 1.043x |
| Linux | float32 | cross | 1 | 13.50065 | 1.78181 | 7.577x | 1.058x |
| Linux | float32 | cross | 4 | 13.52129 | 0.97927 | 13.807x | 1.054x |
| Linux | float32 | even | 1 | 9.27434 | 1.74207 | 5.324x | 1.052x |
| Linux | float32 | even | 4 | 9.22401 | 0.96652 | 9.543x | 1.034x |
| Linux | float32 | identity | 1 | 0.07992 | 0.04454 | 1.794x | 0.981x |
| Linux | float32 | identity | 4 | 0.08013 | 0.04320 | 1.855x | 1.017x |
| Linux | float32 | tiny | 1 | 0.01843 | 0.01610 | 1.145x | 1.017x |
| Linux | float32 | tiny | 4 | 0.01783 | 0.01708 | 1.044x | 0.966x |
| Linux | float32 | one-dimensional3 | 1 | 4.40167 | 1.80467 | 2.439x | 1.050x |
| Linux | float32 | one-dimensional3 | 4 | 4.46122 | 1.04034 | 4.288x | 1.040x |
| Linux | float32 | one-dimensional31 | 1 | 11.47846 | 11.47182 | 1.001x | 1.059x |
| Linux | float32 | one-dimensional31 | 4 | 11.43407 | 3.94747 | 2.897x | 1.025x |
| Linux | float32 | three-dimensional | 1 | 19.04376 | 1.84456 | 10.324x | 1.008x |
| Linux | float32 | three-dimensional | 4 | 19.00921 | 1.85612 | 10.241x | 1.005x |
| Linux | float64 | nearest | 1 | 32.39335 | 1.85570 | 17.456x | 1.045x |
| Linux | float64 | nearest | 4 | 32.41113 | 1.22315 | 26.498x | 1.009x |
| Linux | float64 | reflect | 1 | 32.40521 | 1.84461 | 17.568x | 1.046x |
| Linux | float64 | reflect | 4 | 32.33130 | 1.24040 | 26.065x | 1.015x |
| Linux | float64 | mirror | 1 | 32.32312 | 1.84668 | 17.503x | 1.045x |
| Linux | float64 | mirror | 4 | 32.38389 | 1.26585 | 25.583x | 1.004x |
| Linux | float64 | wrap | 1 | 32.35780 | 1.88900 | 17.130x | 1.033x |
| Linux | float64 | wrap | 4 | 32.41987 | 1.24196 | 26.104x | 1.015x |
| Linux | float64 | constant | 1 | 32.39705 | 1.85757 | 17.441x | 1.034x |
| Linux | float64 | constant | 4 | 32.35112 | 1.23246 | 26.249x | 1.013x |
| Linux | float64 | fortran | 1 | 32.65981 | 2.79375 | 11.690x | 1.018x |
| Linux | float64 | fortran | 4 | 32.57659 | 2.20333 | 14.785x | 1.005x |
| Linux | float64 | strided | 1 | 16.45822 | 1.04134 | 15.805x | 1.038x |
| Linux | float64 | strided | 4 | 16.45321 | 1.04993 | 15.671x | 1.033x |
| Linux | float64 | dense5 | 1 | 107.22905 | 8.03478 | 13.346x | 0.999x |
| Linux | float64 | dense5 | 4 | 107.28196 | 4.18903 | 25.610x | 1.000x |
| Linux | float64 | sparse7 | 1 | 32.67268 | 1.86114 | 17.555x | 1.046x |
| Linux | float64 | sparse7 | 4 | 32.70296 | 1.20780 | 27.076x | 1.041x |
| Linux | float64 | cross | 1 | 13.25007 | 1.30306 | 10.168x | 1.071x |
| Linux | float64 | cross | 4 | 13.12037 | 0.99324 | 13.210x | 1.008x |
| Linux | float64 | even | 1 | 8.85665 | 1.23810 | 7.153x | 1.060x |
| Linux | float64 | even | 4 | 8.78484 | 0.92617 | 9.485x | 1.042x |
| Linux | float64 | identity | 1 | 0.11601 | 0.06837 | 1.697x | 1.021x |
| Linux | float64 | identity | 4 | 0.11759 | 0.06846 | 1.718x | 1.017x |
| Linux | float64 | tiny | 1 | 0.01828 | 0.01730 | 1.057x | 0.975x |
| Linux | float64 | tiny | 4 | 0.01787 | 0.01760 | 1.015x | 0.950x |
| Linux | float64 | one-dimensional3 | 1 | 4.44714 | 1.16954 | 3.802x | 1.078x |
| Linux | float64 | one-dimensional3 | 4 | 4.45393 | 1.01117 | 4.405x | 0.985x |
| Linux | float64 | one-dimensional31 | 1 | 11.51511 | 11.77774 | 0.978x | 1.052x |
| Linux | float64 | one-dimensional31 | 4 | 11.49323 | 4.05299 | 2.836x | 1.015x |
| Linux | float64 | three-dimensional | 1 | 19.01017 | 1.73318 | 10.968x | 1.018x |
| Linux | float64 | three-dimensional | 4 | 18.93375 | 1.73887 | 10.889x | 1.009x |
| Windows | uint8 | nearest | 1 | 32.10150 | 2.87346 | 11.172x | 1.002x |
| Windows | uint8 | nearest | 4 | 32.23490 | 1.75356 | 18.383x | 0.983x |
| Windows | uint8 | reflect | 1 | 32.05760 | 2.84888 | 11.253x | 0.999x |
| Windows | uint8 | reflect | 4 | 32.10030 | 1.74606 | 18.384x | 0.990x |
| Windows | uint8 | mirror | 1 | 32.14270 | 2.85834 | 11.245x | 1.003x |
| Windows | uint8 | mirror | 4 | 32.01030 | 1.75047 | 18.287x | 0.991x |
| Windows | uint8 | wrap | 1 | 32.39420 | 2.91590 | 11.110x | 0.993x |
| Windows | uint8 | wrap | 4 | 32.16710 | 1.71901 | 18.713x | 0.996x |
| Windows | uint8 | constant | 1 | 32.06130 | 2.85496 | 11.230x | 1.010x |
| Windows | uint8 | constant | 4 | 32.30350 | 1.73811 | 18.585x | 0.980x |
| Windows | uint8 | fortran | 1 | 32.28830 | 3.25168 | 9.930x | 0.994x |
| Windows | uint8 | fortran | 4 | 32.33290 | 2.13094 | 15.173x | 0.980x |
| Windows | uint8 | strided | 1 | 16.39490 | 1.56608 | 10.469x | 0.992x |
| Windows | uint8 | strided | 4 | 16.33000 | 1.55967 | 10.470x | 0.993x |
| Windows | uint8 | dense5 | 1 | 106.55500 | 33.29700 | 3.200x | 1.003x |
| Windows | uint8 | dense5 | 4 | 106.65850 | 15.23140 | 7.003x | 1.005x |
| Windows | uint8 | sparse7 | 1 | 32.51150 | 2.87522 | 11.307x | 1.003x |
| Windows | uint8 | sparse7 | 4 | 32.34240 | 1.75779 | 18.400x | 0.986x |
| Windows | uint8 | cross | 1 | 13.55660 | 2.28520 | 5.932x | 1.011x |
| Windows | uint8 | cross | 4 | 13.55610 | 1.50270 | 9.021x | 0.974x |
| Windows | uint8 | even | 1 | 9.29540 | 2.21475 | 4.197x | 1.003x |
| Windows | uint8 | even | 4 | 9.27350 | 1.44207 | 6.431x | 0.984x |
| Windows | uint8 | identity | 1 | 0.04020 | 0.01851 | 2.172x | 0.987x |
| Windows | uint8 | identity | 4 | 0.04054 | 0.01874 | 2.163x | 0.975x |
| Windows | uint8 | tiny | 1 | 0.01688 | 0.01615 | 1.045x | 0.970x |
| Windows | uint8 | tiny | 4 | 0.01675 | 0.01594 | 1.051x | 0.965x |
| Windows | uint8 | one-dimensional3 | 1 | 5.08265 | 2.18900 | 2.322x | 0.993x |
| Windows | uint8 | one-dimensional3 | 4 | 5.09145 | 1.86909 | 2.724x | 0.992x |
| Windows | uint8 | one-dimensional31 | 1 | 11.27250 | 11.11760 | 1.014x | 1.060x |
| Windows | uint8 | one-dimensional31 | 4 | 11.29510 | 4.06053 | 2.782x | 1.071x |
| Windows | uint8 | three-dimensional | 1 | 18.74810 | 5.87020 | 3.194x | 1.007x |
| Windows | uint8 | three-dimensional | 4 | 18.72100 | 5.88680 | 3.180x | 0.994x |
| Windows | float32 | nearest | 1 | 31.92500 | 3.40030 | 9.389x | 1.037x |
| Windows | float32 | nearest | 4 | 31.95330 | 2.41586 | 13.226x | 0.988x |
| Windows | float32 | reflect | 1 | 31.95770 | 3.45530 | 9.249x | 1.014x |
| Windows | float32 | reflect | 4 | 31.91260 | 2.33974 | 13.639x | 1.027x |
| Windows | float32 | mirror | 1 | 31.98460 | 3.44255 | 9.291x | 1.023x |
| Windows | float32 | mirror | 4 | 32.05490 | 2.34402 | 13.675x | 1.018x |
| Windows | float32 | wrap | 1 | 31.99690 | 3.43525 | 9.314x | 1.028x |
| Windows | float32 | wrap | 4 | 32.07190 | 2.37316 | 13.514x | 1.037x |
| Windows | float32 | constant | 1 | 32.05780 | 3.43850 | 9.323x | 1.020x |
| Windows | float32 | constant | 4 | 31.96540 | 2.37817 | 13.441x | 1.004x |
| Windows | float32 | fortran | 1 | 32.56400 | 4.30700 | 7.561x | 1.019x |
| Windows | float32 | fortran | 4 | 32.46960 | 3.30560 | 9.823x | 1.017x |
| Windows | float32 | strided | 1 | 16.14150 | 1.74579 | 9.246x | 1.024x |
| Windows | float32 | strided | 4 | 16.21970 | 1.74069 | 9.318x | 1.027x |
| Windows | float32 | dense5 | 1 | 107.77690 | 33.86430 | 3.183x | 1.001x |
| Windows | float32 | dense5 | 4 | 107.30230 | 15.88890 | 6.753x | 1.005x |
| Windows | float32 | sparse7 | 1 | 32.31520 | 3.44678 | 9.375x | 1.024x |
| Windows | float32 | sparse7 | 4 | 32.32400 | 2.39234 | 13.511x | 1.018x |
| Windows | float32 | cross | 1 | 13.39890 | 2.87767 | 4.656x | 1.025x |
| Windows | float32 | cross | 4 | 13.49850 | 2.20002 | 6.136x | 0.991x |
| Windows | float32 | even | 1 | 9.10700 | 2.75002 | 3.312x | 1.039x |
| Windows | float32 | even | 4 | 9.15560 | 2.12323 | 4.312x | 1.030x |
| Windows | float32 | identity | 1 | 0.29828 | 0.27238 | 1.095x | 1.000x |
| Windows | float32 | identity | 4 | 0.29026 | 0.26655 | 1.089x | 1.001x |
| Windows | float32 | tiny | 1 | 0.01638 | 0.01644 | 0.996x | 0.966x |
| Windows | float32 | tiny | 4 | 0.01717 | 0.01647 | 1.043x | 1.080x |
| Windows | float32 | one-dimensional3 | 1 | 4.63373 | 2.67452 | 1.733x | 1.039x |
| Windows | float32 | one-dimensional3 | 4 | 4.63923 | 2.33666 | 1.985x | 1.015x |
| Windows | float32 | one-dimensional31 | 1 | 11.90680 | 12.71290 | 0.937x | 1.049x |
| Windows | float32 | one-dimensional31 | 4 | 11.79310 | 4.34983 | 2.711x | 1.058x |
| Windows | float32 | three-dimensional | 1 | 18.88700 | 6.00025 | 3.148x | 0.992x |
| Windows | float32 | three-dimensional | 4 | 18.84980 | 6.01860 | 3.132x | 0.991x |
| Windows | float64 | nearest | 1 | 32.27600 | 3.28938 | 9.812x | 0.963x |
| Windows | float64 | nearest | 4 | 32.02520 | 2.28098 | 14.040x | 0.974x |
| Windows | float64 | reflect | 1 | 32.00990 | 3.24070 | 9.877x | 0.976x |
| Windows | float64 | reflect | 4 | 32.11010 | 2.28553 | 14.049x | 0.982x |
| Windows | float64 | mirror | 1 | 32.11680 | 3.22978 | 9.944x | 0.978x |
| Windows | float64 | mirror | 4 | 32.10790 | 2.28717 | 14.038x | 0.979x |
| Windows | float64 | wrap | 1 | 32.15900 | 3.25672 | 9.875x | 0.974x |
| Windows | float64 | wrap | 4 | 32.06610 | 2.28698 | 14.021x | 0.973x |
| Windows | float64 | constant | 1 | 32.06810 | 3.26675 | 9.817x | 0.975x |
| Windows | float64 | constant | 4 | 32.07370 | 2.31160 | 13.875x | 0.995x |
| Windows | float64 | fortran | 1 | 33.79430 | 5.27705 | 6.404x | 0.986x |
| Windows | float64 | fortran | 4 | 33.79440 | 4.36597 | 7.740x | 0.992x |
| Windows | float64 | strided | 1 | 16.12960 | 2.01846 | 7.991x | 0.992x |
| Windows | float64 | strided | 4 | 16.29390 | 2.01303 | 8.094x | 0.980x |
| Windows | float64 | dense5 | 1 | 107.65790 | 33.69850 | 3.195x | 1.000x |
| Windows | float64 | dense5 | 4 | 107.31000 | 15.95730 | 6.725x | 0.984x |
| Windows | float64 | sparse7 | 1 | 32.25150 | 3.26308 | 9.884x | 0.973x |
| Windows | float64 | sparse7 | 4 | 32.34310 | 2.27590 | 14.211x | 1.016x |
| Windows | float64 | cross | 1 | 13.52670 | 2.71418 | 4.984x | 0.969x |
| Windows | float64 | cross | 4 | 13.36780 | 2.10077 | 6.363x | 0.969x |
| Windows | float64 | even | 1 | 9.17670 | 2.58324 | 3.552x | 0.975x |
| Windows | float64 | even | 4 | 9.16920 | 2.06367 | 4.443x | 0.981x |
| Windows | float64 | identity | 1 | 0.47211 | 0.44286 | 1.066x | 0.998x |
| Windows | float64 | identity | 4 | 0.46662 | 0.43728 | 1.067x | 0.998x |
| Windows | float64 | tiny | 1 | 0.01819 | 0.01816 | 1.002x | 0.987x |
| Windows | float64 | tiny | 4 | 0.01614 | 0.01672 | 0.965x | 1.003x |
| Windows | float64 | one-dimensional3 | 1 | 4.91180 | 2.59552 | 1.892x | 0.965x |
| Windows | float64 | one-dimensional3 | 4 | 4.80860 | 2.02098 | 2.379x | 0.977x |
| Windows | float64 | one-dimensional31 | 1 | 12.21200 | 12.78410 | 0.955x | 1.049x |
| Windows | float64 | one-dimensional31 | 4 | 12.33640 | 6.09130 | 2.025x | 1.211x |
| Windows | float64 | three-dimensional | 1 | 19.40310 | 6.05965 | 3.202x | 1.018x |
| Windows | float64 | three-dimensional | 4 | 19.03240 | 6.18905 | 3.075x | 1.057x |

## Native signed-zero insertion history: `68d8516`

[Paired benchmark](https://github.com/bipxx/scikit-imagexx/actions/runs/39819925274), against exact hole-repair source `0638289` from run `39817335611`. Product fingerprint: `c7d5e93fb11ee01823f6d75049729ae3496ad63e145642eed4fd66dbd67e6b1d`. This source adds native replay of the upstream mediator history for signed zeros. The 48 original cases retain seed 5227; four new signed-zero cases use independent seed 5228. There are 52 cases at worker caps one/four, 104 rows per OS. Local packaged-wheel checks pass 8,473 cases per runtime and SIMD mode, plus five native configurations (2/2 each); attribution is included in wheel and standalone install rules.

Linux: **6.965991x upstream/native over all 52 one-worker cases**; **8.504763x over the original 48-case corpus**. [All raw samples](median-ties-linux.json).

Windows: **3.957636x upstream/native over all 52 one-worker cases**; **4.623067x over the original 48-case corpus**. [All raw samples](median-ties-windows.json).

Current signed-zero results match reference values and sign bits. The older baseline differs in 37,364 signs for width three and about 98,000 signs for width 31, depending on OS. Those incorrect baseline cases are explicitly marked unmeasured, with mismatch counts retained; upstream/current timing remains available. The new replay path is only 0.58-0.76x upstream on Linux and 0.57-0.67x on Windows, so its implementation still needs optimization. Long ordinary 1-D windows and some tiny calls also remain adverse.

A subsequent N-D audit found that dense two-element footprints use the separable maximum filter, which keeps the later equal value; sparse extrema retain an earlier equal value. This source still fails that sign-bit case and remains unretained. Corrected source `7f1d317` adds 192 N-D signed-zero cases and a signed-zero two-tap benchmark; it passes 8,205 scoped Windows cases and 6,052 minimum-NumPy cases in each SIMD mode, plus all five native configurations (2/2 each). Its isolated measurements and full CI are pending.

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | hole/current | baseline parity |
|---|---|---|---:|---:|---:|---:|---:|---|
| Linux | uint8 | nearest | 1 | 28.60498 | 1.36013 | 21.031x | 1.003x | verified |
| Linux | uint8 | nearest | 4 | 28.59183 | 0.81572 | 35.051x | 0.980x | verified |
| Linux | uint8 | reflect | 1 | 28.60012 | 1.36206 | 20.998x | 1.001x | verified |
| Linux | uint8 | reflect | 4 | 28.55523 | 0.80222 | 35.595x | 0.996x | verified |
| Linux | uint8 | mirror | 1 | 29.03548 | 1.37427 | 21.128x | 1.006x | verified |
| Linux | uint8 | mirror | 4 | 28.61551 | 0.80482 | 35.555x | 0.988x | verified |
| Linux | uint8 | wrap | 1 | 28.64873 | 1.37717 | 20.803x | 0.998x | verified |
| Linux | uint8 | wrap | 4 | 28.59303 | 0.80483 | 35.527x | 0.995x | verified |
| Linux | uint8 | constant | 1 | 28.38969 | 1.35794 | 20.906x | 1.001x | verified |
| Linux | uint8 | constant | 4 | 28.45794 | 0.80955 | 35.153x | 0.994x | verified |
| Linux | uint8 | fortran | 1 | 28.79657 | 1.52032 | 18.941x | 0.999x | verified |
| Linux | uint8 | fortran | 4 | 28.82831 | 0.96538 | 29.862x | 0.986x | verified |
| Linux | uint8 | strided | 1 | 14.47022 | 0.74610 | 19.395x | 1.005x | verified |
| Linux | uint8 | strided | 4 | 14.45267 | 0.75106 | 19.243x | 0.999x | verified |
| Linux | uint8 | dense5 | 1 | 93.60268 | 6.51510 | 14.367x | 0.996x | verified |
| Linux | uint8 | dense5 | 4 | 93.58144 | 3.28787 | 28.463x | 0.993x | verified |
| Linux | uint8 | sparse7 | 1 | 28.82831 | 1.37209 | 21.011x | 0.999x | verified |
| Linux | uint8 | sparse7 | 4 | 28.83237 | 0.81183 | 35.515x | 0.990x | verified |
| Linux | uint8 | cross | 1 | 11.49441 | 0.88989 | 12.917x | 1.003x | verified |
| Linux | uint8 | cross | 4 | 11.49790 | 0.58848 | 19.538x | 1.008x | verified |
| Linux | uint8 | even | 1 | 7.66266 | 0.82275 | 9.313x | 1.002x | verified |
| Linux | uint8 | even | 4 | 7.63504 | 0.55858 | 13.669x | 1.000x | verified |
| Linux | uint8 | identity | 1 | 0.02418 | 0.01441 | 1.678x | 1.012x | verified |
| Linux | uint8 | identity | 4 | 0.02487 | 0.01472 | 1.689x | 1.026x | verified |
| Linux | uint8 | tiny | 1 | 0.01104 | 0.00984 | 1.122x | 1.019x | verified |
| Linux | uint8 | tiny | 4 | 0.01070 | 0.00988 | 1.083x | 1.016x | verified |
| Linux | uint8 | one-dimensional3 | 1 | 3.29692 | 0.36356 | 9.069x | 1.004x | verified |
| Linux | uint8 | one-dimensional3 | 4 | 3.28650 | 0.37522 | 8.759x | 1.017x | verified |
| Linux | uint8 | one-dimensional31 | 1 | 8.04462 | 8.74128 | 0.920x | 0.996x | verified |
| Linux | uint8 | one-dimensional31 | 4 | 8.05284 | 2.96503 | 2.716x | 0.986x | verified |
| Linux | uint8 | three-dimensional | 1 | 16.47688 | 1.41501 | 11.644x | 0.997x | verified |
| Linux | uint8 | three-dimensional | 4 | 16.48015 | 1.41405 | 11.655x | 0.999x | verified |
| Linux | float32 | nearest | 1 | 28.40413 | 1.86243 | 15.251x | 1.037x | verified |
| Linux | float32 | nearest | 4 | 28.38540 | 0.85569 | 33.172x | 1.042x | verified |
| Linux | float32 | reflect | 1 | 28.38594 | 1.86300 | 15.237x | 1.041x | verified |
| Linux | float32 | reflect | 4 | 28.38261 | 0.86674 | 32.747x | 1.035x | verified |
| Linux | float32 | mirror | 1 | 28.45248 | 1.86350 | 15.268x | 1.041x | verified |
| Linux | float32 | mirror | 4 | 28.42429 | 0.86217 | 32.968x | 1.040x | verified |
| Linux | float32 | wrap | 1 | 28.46041 | 1.87470 | 15.181x | 1.039x | verified |
| Linux | float32 | wrap | 4 | 28.46914 | 0.86634 | 32.861x | 1.043x | verified |
| Linux | float32 | constant | 1 | 28.38974 | 1.86096 | 15.255x | 1.038x | verified |
| Linux | float32 | constant | 4 | 28.30042 | 0.85966 | 32.921x | 1.041x | verified |
| Linux | float32 | fortran | 1 | 28.74409 | 2.47280 | 11.624x | 1.028x | verified |
| Linux | float32 | fortran | 4 | 28.71205 | 1.46458 | 19.604x | 1.024x | verified |
| Linux | float32 | strided | 1 | 14.40220 | 1.00640 | 14.311x | 1.036x | verified |
| Linux | float32 | strided | 4 | 14.40338 | 1.00833 | 14.284x | 1.036x | verified |
| Linux | float32 | dense5 | 1 | 93.25341 | 6.99773 | 13.326x | 1.011x | verified |
| Linux | float32 | dense5 | 4 | 93.33377 | 3.35294 | 27.836x | 1.013x | verified |
| Linux | float32 | sparse7 | 1 | 28.64400 | 1.86605 | 15.350x | 1.041x | verified |
| Linux | float32 | sparse7 | 4 | 28.64268 | 0.86046 | 33.288x | 1.041x | verified |
| Linux | float32 | cross | 1 | 11.33935 | 1.38622 | 8.180x | 1.057x | verified |
| Linux | float32 | cross | 4 | 11.33071 | 0.63918 | 17.727x | 1.047x | verified |
| Linux | float32 | even | 1 | 7.56116 | 1.32455 | 5.708x | 1.054x | verified |
| Linux | float32 | even | 4 | 7.54288 | 0.60901 | 12.385x | 1.052x | verified |
| Linux | float32 | identity | 1 | 0.05417 | 0.03344 | 1.620x | 1.000x | verified |
| Linux | float32 | identity | 4 | 0.05194 | 0.03122 | 1.664x | 1.005x | verified |
| Linux | float32 | tiny | 1 | 0.01071 | 0.01061 | 1.010x | 1.003x | verified |
| Linux | float32 | tiny | 4 | 0.01087 | 0.01071 | 1.015x | 0.997x | verified |
| Linux | float32 | one-dimensional3 | 1 | 3.72971 | 1.28279 | 2.908x | 1.060x | verified |
| Linux | float32 | one-dimensional3 | 4 | 3.72201 | 0.63763 | 5.837x | 1.060x | verified |
| Linux | float32 | one-dimensional31 | 1 | 9.57990 | 9.81926 | 0.976x | 0.991x | verified |
| Linux | float32 | one-dimensional31 | 4 | 9.56081 | 3.15250 | 3.033x | 0.992x | verified |
| Linux | float32 | three-dimensional | 1 | 16.46569 | 1.49045 | 11.047x | 1.015x | verified |
| Linux | float32 | three-dimensional | 4 | 16.47457 | 1.49456 | 11.023x | 1.013x | verified |
| Linux | float32 | signed-zero3 | 1 | 2.79868 | 4.74701 | 0.590x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float32 | signed-zero3 | 4 | 3.02928 | 4.75232 | 0.637x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float32 | signed-zero31 | 1 | 6.36974 | 8.41638 | 0.757x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float32 | signed-zero31 | 4 | 6.37105 | 8.40498 | 0.758x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float64 | nearest | 1 | 28.08720 | 1.46935 | 19.115x | 1.044x | verified |
| Linux | float64 | nearest | 4 | 28.03624 | 0.86048 | 32.582x | 1.074x | verified |
| Linux | float64 | reflect | 1 | 28.01518 | 1.46492 | 19.124x | 1.048x | verified |
| Linux | float64 | reflect | 4 | 28.05249 | 0.85716 | 32.727x | 1.076x | verified |
| Linux | float64 | mirror | 1 | 28.00823 | 1.47348 | 19.008x | 1.041x | verified |
| Linux | float64 | mirror | 4 | 28.04994 | 0.86300 | 32.503x | 1.075x | verified |
| Linux | float64 | wrap | 1 | 28.05470 | 1.47758 | 18.987x | 1.050x | verified |
| Linux | float64 | wrap | 4 | 28.06205 | 0.86411 | 32.475x | 1.079x | verified |
| Linux | float64 | constant | 1 | 27.93170 | 1.47735 | 18.907x | 1.044x | verified |
| Linux | float64 | constant | 4 | 27.90606 | 0.85659 | 32.578x | 1.076x | verified |
| Linux | float64 | fortran | 1 | 29.28764 | 3.28314 | 8.921x | 1.021x | verified |
| Linux | float64 | fortran | 4 | 29.29008 | 2.66040 | 11.010x | 1.032x | verified |
| Linux | float64 | strided | 1 | 14.23885 | 0.81633 | 17.443x | 1.043x | verified |
| Linux | float64 | strided | 4 | 14.21205 | 0.81794 | 17.375x | 1.041x | verified |
| Linux | float64 | dense5 | 1 | 92.78327 | 6.63697 | 13.980x | 1.015x | verified |
| Linux | float64 | dense5 | 4 | 92.70328 | 3.33616 | 27.787x | 1.025x | verified |
| Linux | float64 | sparse7 | 1 | 28.28664 | 1.46974 | 19.246x | 1.048x | verified |
| Linux | float64 | sparse7 | 4 | 28.27646 | 0.85541 | 33.056x | 1.083x | verified |
| Linux | float64 | cross | 1 | 11.06304 | 1.06035 | 10.433x | 1.001x | verified |
| Linux | float64 | cross | 4 | 11.03691 | 0.63030 | 17.511x | 1.106x | verified |
| Linux | float64 | even | 1 | 7.24283 | 0.92238 | 7.852x | 1.078x | verified |
| Linux | float64 | even | 4 | 7.24694 | 0.59870 | 12.104x | 1.114x | verified |
| Linux | float64 | identity | 1 | 0.09289 | 0.05800 | 1.602x | 0.997x | verified |
| Linux | float64 | identity | 4 | 0.08642 | 0.05392 | 1.603x | 0.999x | verified |
| Linux | float64 | tiny | 1 | 0.01070 | 0.01064 | 1.005x | 0.994x | verified |
| Linux | float64 | tiny | 4 | 0.01091 | 0.01069 | 1.021x | 0.996x | verified |
| Linux | float64 | one-dimensional3 | 1 | 3.64792 | 0.88216 | 4.135x | 1.052x | verified |
| Linux | float64 | one-dimensional3 | 4 | 3.64016 | 0.59133 | 6.156x | 1.105x | verified |
| Linux | float64 | one-dimensional31 | 1 | 9.49779 | 9.72508 | 0.977x | 1.018x | verified |
| Linux | float64 | one-dimensional31 | 4 | 9.48966 | 3.19674 | 2.969x | 1.024x | verified |
| Linux | float64 | three-dimensional | 1 | 16.33824 | 1.43158 | 11.413x | 1.016x | verified |
| Linux | float64 | three-dimensional | 4 | 16.34054 | 1.43024 | 11.425x | 1.012x | verified |
| Linux | float64 | signed-zero3 | 1 | 2.77007 | 4.74763 | 0.583x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float64 | signed-zero3 | 4 | 2.76610 | 4.74736 | 0.583x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float64 | signed-zero31 | 1 | 6.33947 | 10.14543 | 0.625x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float64 | signed-zero31 | 4 | 6.36046 | 10.18178 | 0.625x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | uint8 | nearest | 1 | 32.82710 | 2.68263 | 12.237x | 0.990x | verified |
| Windows | uint8 | nearest | 4 | 32.69620 | 1.87467 | 17.441x | 0.987x | verified |
| Windows | uint8 | reflect | 1 | 32.73660 | 2.61800 | 12.504x | 1.005x | verified |
| Windows | uint8 | reflect | 4 | 32.76720 | 1.87503 | 17.476x | 1.001x | verified |
| Windows | uint8 | mirror | 1 | 32.74440 | 2.58152 | 12.684x | 1.015x | verified |
| Windows | uint8 | mirror | 4 | 32.87530 | 1.86440 | 17.633x | 0.986x | verified |
| Windows | uint8 | wrap | 1 | 32.85090 | 2.64440 | 12.423x | 0.993x | verified |
| Windows | uint8 | wrap | 4 | 32.78890 | 1.83244 | 17.894x | 1.024x | verified |
| Windows | uint8 | constant | 1 | 32.89660 | 2.72236 | 12.084x | 0.996x | verified |
| Windows | uint8 | constant | 4 | 33.03680 | 1.87437 | 17.626x | 1.003x | verified |
| Windows | uint8 | fortran | 1 | 32.83130 | 2.89138 | 11.355x | 0.997x | verified |
| Windows | uint8 | fortran | 4 | 32.82720 | 2.15472 | 15.235x | 0.996x | verified |
| Windows | uint8 | strided | 1 | 16.50180 | 1.38739 | 11.894x | 1.008x | verified |
| Windows | uint8 | strided | 4 | 16.65790 | 1.39106 | 11.975x | 1.000x | verified |
| Windows | uint8 | dense5 | 1 | 93.47940 | 26.05570 | 3.588x | 1.044x | verified |
| Windows | uint8 | dense5 | 4 | 92.92900 | 13.21880 | 7.030x | 0.998x | verified |
| Windows | uint8 | sparse7 | 1 | 32.97480 | 2.62888 | 12.543x | 1.006x | verified |
| Windows | uint8 | sparse7 | 4 | 32.91520 | 1.86646 | 17.635x | 1.032x | verified |
| Windows | uint8 | cross | 1 | 11.99820 | 2.10087 | 5.711x | 1.013x | verified |
| Windows | uint8 | cross | 4 | 11.98810 | 1.66646 | 7.194x | 1.019x | verified |
| Windows | uint8 | even | 1 | 7.92340 | 2.08263 | 3.805x | 0.992x | verified |
| Windows | uint8 | even | 4 | 7.94400 | 1.60801 | 4.940x | 1.007x | verified |
| Windows | uint8 | identity | 1 | 0.02364 | 0.01025 | 2.306x | 0.996x | verified |
| Windows | uint8 | identity | 4 | 0.02336 | 0.01012 | 2.307x | 0.994x | verified |
| Windows | uint8 | tiny | 1 | 0.01046 | 0.01004 | 1.042x | 1.014x | verified |
| Windows | uint8 | tiny | 4 | 0.01062 | 0.01012 | 1.049x | 0.983x | verified |
| Windows | uint8 | one-dimensional3 | 1 | 4.79373 | 3.31058 | 1.448x | 0.994x | verified |
| Windows | uint8 | one-dimensional3 | 4 | 4.48520 | 2.97924 | 1.505x | 0.982x | verified |
| Windows | uint8 | one-dimensional31 | 1 | 10.27830 | 9.52320 | 1.079x | 1.002x | verified |
| Windows | uint8 | one-dimensional31 | 4 | 10.26240 | 4.07673 | 2.517x | 0.993x | verified |
| Windows | uint8 | three-dimensional | 1 | 16.41770 | 4.57680 | 3.587x | 1.056x | verified |
| Windows | uint8 | three-dimensional | 4 | 16.53280 | 4.59083 | 3.601x | 1.036x | verified |
| Windows | float32 | nearest | 1 | 33.19440 | 3.41730 | 9.714x | 0.996x | verified |
| Windows | float32 | nearest | 4 | 33.06570 | 2.76808 | 11.945x | 0.983x | verified |
| Windows | float32 | reflect | 1 | 33.24470 | 3.36117 | 9.891x | 1.000x | verified |
| Windows | float32 | reflect | 4 | 33.33220 | 2.70450 | 12.325x | 0.969x | verified |
| Windows | float32 | mirror | 1 | 33.07710 | 3.39655 | 9.738x | 1.009x | verified |
| Windows | float32 | mirror | 4 | 33.31490 | 2.72054 | 12.246x | 0.996x | verified |
| Windows | float32 | wrap | 1 | 33.31850 | 3.39895 | 9.803x | 1.004x | verified |
| Windows | float32 | wrap | 4 | 33.29030 | 2.70784 | 12.294x | 0.994x | verified |
| Windows | float32 | constant | 1 | 33.13370 | 3.39253 | 9.767x | 0.996x | verified |
| Windows | float32 | constant | 4 | 33.90300 | 3.10994 | 10.901x | 0.997x | verified |
| Windows | float32 | fortran | 1 | 33.67410 | 4.06013 | 8.294x | 1.004x | verified |
| Windows | float32 | fortran | 4 | 33.76170 | 3.32757 | 10.146x | 1.006x | verified |
| Windows | float32 | strided | 1 | 16.71420 | 1.61156 | 10.371x | 0.992x | verified |
| Windows | float32 | strided | 4 | 16.70740 | 1.60918 | 10.383x | 1.002x | verified |
| Windows | float32 | dense5 | 1 | 93.96650 | 26.93280 | 3.489x | 1.046x | verified |
| Windows | float32 | dense5 | 4 | 93.75060 | 14.09610 | 6.651x | 1.010x | verified |
| Windows | float32 | sparse7 | 1 | 33.07710 | 3.27700 | 10.094x | 1.027x | verified |
| Windows | float32 | sparse7 | 4 | 33.58830 | 2.64992 | 12.675x | 0.983x | verified |
| Windows | float32 | cross | 1 | 12.30860 | 2.77806 | 4.431x | 1.009x | verified |
| Windows | float32 | cross | 4 | 12.46590 | 2.37938 | 5.239x | 0.999x | verified |
| Windows | float32 | even | 1 | 7.98850 | 2.69536 | 2.964x | 1.002x | verified |
| Windows | float32 | even | 4 | 8.32210 | 2.33238 | 3.568x | 0.999x | verified |
| Windows | float32 | identity | 1 | 0.48256 | 0.47378 | 1.019x | 0.975x | verified |
| Windows | float32 | identity | 4 | 0.51225 | 0.49336 | 1.038x | 0.999x | verified |
| Windows | float32 | tiny | 1 | 0.01004 | 0.01017 | 0.987x | 1.005x | verified |
| Windows | float32 | tiny | 4 | 0.01052 | 0.01045 | 1.007x | 1.057x | verified |
| Windows | float32 | one-dimensional3 | 1 | 3.84513 | 2.56582 | 1.499x | 0.994x | verified |
| Windows | float32 | one-dimensional3 | 4 | 3.87923 | 2.33447 | 1.662x | 1.015x | verified |
| Windows | float32 | one-dimensional31 | 1 | 9.34860 | 9.48500 | 0.986x | 1.014x | verified |
| Windows | float32 | one-dimensional31 | 4 | 9.40020 | 3.50665 | 2.681x | 0.988x | verified |
| Windows | float32 | three-dimensional | 1 | 16.41210 | 4.68813 | 3.501x | 1.047x | verified |
| Windows | float32 | three-dimensional | 4 | 16.44100 | 4.60380 | 3.571x | 1.065x | verified |
| Windows | float32 | signed-zero3 | 1 | 2.99385 | 5.27535 | 0.568x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float32 | signed-zero3 | 4 | 3.02542 | 5.31210 | 0.570x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float32 | signed-zero31 | 1 | 6.65275 | 9.86270 | 0.675x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float32 | signed-zero31 | 4 | 6.63815 | 9.85480 | 0.674x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float64 | nearest | 1 | 33.19430 | 3.44293 | 9.641x | 1.032x | verified |
| Windows | float64 | nearest | 4 | 33.39030 | 2.75076 | 12.139x | 0.985x | verified |
| Windows | float64 | reflect | 1 | 33.08270 | 3.50320 | 9.444x | 0.986x | verified |
| Windows | float64 | reflect | 4 | 33.35680 | 2.75632 | 12.102x | 1.026x | verified |
| Windows | float64 | mirror | 1 | 33.33050 | 3.45858 | 9.637x | 0.990x | verified |
| Windows | float64 | mirror | 4 | 33.50140 | 2.75342 | 12.167x | 1.021x | verified |
| Windows | float64 | wrap | 1 | 33.38520 | 3.47157 | 9.617x | 0.992x | verified |
| Windows | float64 | wrap | 4 | 33.20920 | 2.82840 | 11.741x | 0.984x | verified |
| Windows | float64 | constant | 1 | 33.49030 | 3.51897 | 9.517x | 0.988x | verified |
| Windows | float64 | constant | 4 | 33.42620 | 2.82967 | 11.813x | 0.999x | verified |
| Windows | float64 | fortran | 1 | 34.71100 | 4.80190 | 7.229x | 1.028x | verified |
| Windows | float64 | fortran | 4 | 34.69870 | 4.16577 | 8.329x | 1.027x | verified |
| Windows | float64 | strided | 1 | 16.90270 | 2.28520 | 7.397x | 1.006x | verified |
| Windows | float64 | strided | 4 | 16.89500 | 2.28662 | 7.389x | 1.029x | verified |
| Windows | float64 | dense5 | 1 | 93.99130 | 27.40880 | 3.429x | 1.046x | verified |
| Windows | float64 | dense5 | 4 | 93.97800 | 14.50920 | 6.477x | 0.988x | verified |
| Windows | float64 | sparse7 | 1 | 33.73170 | 3.53300 | 9.548x | 1.025x | verified |
| Windows | float64 | sparse7 | 4 | 33.59960 | 3.06505 | 10.962x | 1.077x | verified |
| Windows | float64 | cross | 1 | 13.02190 | 3.04442 | 4.277x | 0.983x | verified |
| Windows | float64 | cross | 4 | 12.85330 | 2.81593 | 4.565x | 1.035x | verified |
| Windows | float64 | even | 1 | 8.66820 | 2.93260 | 2.956x | 1.014x | verified |
| Windows | float64 | even | 4 | 8.65300 | 2.57798 | 3.357x | 1.012x | verified |
| Windows | float64 | identity | 1 | 0.68179 | 0.64319 | 1.060x | 1.011x | verified |
| Windows | float64 | identity | 4 | 0.66940 | 0.65392 | 1.024x | 0.989x | verified |
| Windows | float64 | tiny | 1 | 0.01117 | 0.01168 | 0.957x | 0.984x | verified |
| Windows | float64 | tiny | 4 | 0.01214 | 0.01169 | 1.039x | 1.034x | verified |
| Windows | float64 | one-dimensional3 | 1 | 4.28343 | 3.07345 | 1.394x | 1.014x | verified |
| Windows | float64 | one-dimensional3 | 4 | 4.34873 | 2.84805 | 1.527x | 1.005x | verified |
| Windows | float64 | one-dimensional31 | 1 | 9.89450 | 10.39590 | 0.952x | 1.007x | verified |
| Windows | float64 | one-dimensional31 | 4 | 9.71020 | 4.17300 | 2.327x | 0.993x | verified |
| Windows | float64 | three-dimensional | 1 | 16.52970 | 4.59390 | 3.598x | 1.072x | verified |
| Windows | float64 | three-dimensional | 4 | 16.46650 | 4.58677 | 3.590x | 1.071x | verified |
| Windows | float64 | signed-zero3 | 1 | 3.25805 | 5.74605 | 0.567x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float64 | signed-zero3 | 4 | 3.20855 | 5.74295 | 0.559x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float64 | signed-zero31 | 1 | 6.83750 | 10.50650 | 0.651x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float64 | signed-zero31 | 4 | 6.93225 | 10.56150 | 0.656x | unmeasured | signed-zero-mismatch-unmeasured |

## Correct two-tap zero representatives: `7f1d317`

[Paired benchmark run](https://github.com/bipxx/scikit-imagexx/actions/runs/39820988391), against exact insertion-history source `68d8516` from run `39819925274`. Product fingerprint: `11b70cf84b4638d968db8bc27e41dbe063336a5caae42706e3c8365c2bec85e7`. The new dense two-tap path chooses the later equal sample, while sparse extrema retain their original order. Two new signed-zero pair cases bring the fixed corpus to 54 cases / 108 rows per OS. Every timed current result matches upstream values and sign bits, with fallback disabled. Local validation passes 8,205 Windows and 6,052 minimum-NumPy cases in each SIMD mode, plus all five native configurations (2/2 each).

Linux: **6.042353x upstream/native over all 54 one-worker cases**. [Every raw sample](median-zero-extrema-linux.json).

Windows: **4.227715x upstream/native over all 54 one-worker cases**. [Every raw sample](median-zero-extrema-windows.json).

The primary tuning metric is the lower OS geometric mean: **4.227715x** for this baseline. This does not imply every case is faster: signed-zero pair cases are only 0.260-0.307x upstream, signed-zero 1-D cases are 0.514-0.786x, and some ordinary long windows and Windows tiny calls remain adverse. The older pair baseline differs in 32,708 signs per case and is explicitly unmeasured there. The benchmark retains all incorrect-baseline counts and all current/reference samples. A centered-pointer trial is being measured separately, using the same 54 cases and metric.

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | prior/current | baseline parity |
|---|---|---|---:|---:|---:|---:|---:|---|
| Linux | uint8 | nearest | 1 | 33.08360 | 1.70569 | 19.396x | 0.997x | verified |
| Linux | uint8 | nearest | 4 | 32.76236 | 0.98515 | 33.256x | 0.977x | verified |
| Linux | uint8 | reflect | 1 | 32.68171 | 1.70208 | 19.201x | 0.993x | verified |
| Linux | uint8 | reflect | 4 | 32.69805 | 1.00037 | 32.686x | 0.969x | verified |
| Linux | uint8 | mirror | 1 | 32.74802 | 1.70966 | 19.155x | 0.987x | verified |
| Linux | uint8 | mirror | 4 | 32.68403 | 0.98648 | 33.132x | 0.998x | verified |
| Linux | uint8 | wrap | 1 | 32.72662 | 1.72270 | 18.997x | 0.981x | verified |
| Linux | uint8 | wrap | 4 | 32.73424 | 1.01160 | 32.359x | 0.965x | verified |
| Linux | uint8 | constant | 1 | 32.71903 | 1.70148 | 19.230x | 0.993x | verified |
| Linux | uint8 | constant | 4 | 32.73424 | 0.99163 | 33.011x | 0.978x | verified |
| Linux | uint8 | fortran | 1 | 32.97017 | 2.06989 | 15.928x | 0.991x | verified |
| Linux | uint8 | fortran | 4 | 32.88859 | 1.35885 | 24.203x | 0.997x | verified |
| Linux | uint8 | strided | 1 | 16.52670 | 0.93786 | 17.622x | 0.998x | verified |
| Linux | uint8 | strided | 4 | 16.59995 | 0.92804 | 17.887x | 1.002x | verified |
| Linux | uint8 | dense5 | 1 | 107.57091 | 7.84579 | 13.711x | 0.991x | verified |
| Linux | uint8 | dense5 | 4 | 107.43101 | 3.90911 | 27.482x | 1.002x | verified |
| Linux | uint8 | sparse7 | 1 | 32.93981 | 1.73183 | 19.020x | 0.988x | verified |
| Linux | uint8 | sparse7 | 4 | 33.02496 | 0.99653 | 33.140x | 0.980x | verified |
| Linux | uint8 | cross | 1 | 13.54735 | 1.14553 | 11.826x | 1.004x | verified |
| Linux | uint8 | cross | 4 | 13.58970 | 0.72621 | 18.713x | 0.988x | verified |
| Linux | uint8 | even | 1 | 9.21939 | 1.07249 | 8.596x | 0.998x | verified |
| Linux | uint8 | even | 4 | 9.17168 | 0.69267 | 13.241x | 0.984x | verified |
| Linux | uint8 | identity | 1 | 0.03789 | 0.01880 | 2.016x | 0.948x | verified |
| Linux | uint8 | identity | 4 | 0.03713 | 0.01774 | 2.093x | 0.972x | verified |
| Linux | uint8 | tiny | 1 | 0.01743 | 0.01535 | 1.135x | 0.956x | verified |
| Linux | uint8 | tiny | 4 | 0.01781 | 0.01570 | 1.134x | 0.933x | verified |
| Linux | uint8 | one-dimensional3 | 1 | 4.32707 | 0.47791 | 9.054x | 1.008x | verified |
| Linux | uint8 | one-dimensional3 | 4 | 4.34145 | 0.47756 | 9.091x | 0.968x | verified |
| Linux | uint8 | one-dimensional31 | 1 | 10.41180 | 10.24224 | 1.017x | 0.993x | verified |
| Linux | uint8 | one-dimensional31 | 4 | 10.41914 | 3.57179 | 2.917x | 0.999x | verified |
| Linux | uint8 | three-dimensional | 1 | 19.02998 | 1.75179 | 10.863x | 0.993x | verified |
| Linux | uint8 | three-dimensional | 4 | 19.07298 | 1.75266 | 10.882x | 1.000x | verified |
| Linux | float32 | nearest | 1 | 32.49322 | 2.17391 | 14.947x | 1.037x | verified |
| Linux | float32 | nearest | 4 | 32.46088 | 1.13129 | 28.694x | 1.036x | verified |
| Linux | float32 | reflect | 1 | 32.58912 | 2.17258 | 15.000x | 1.025x | verified |
| Linux | float32 | reflect | 4 | 32.52175 | 1.13022 | 28.775x | 1.029x | verified |
| Linux | float32 | mirror | 1 | 32.49041 | 2.15302 | 15.091x | 1.035x | verified |
| Linux | float32 | mirror | 4 | 32.48419 | 1.11985 | 29.008x | 1.031x | verified |
| Linux | float32 | wrap | 1 | 32.45278 | 2.15726 | 15.044x | 1.039x | verified |
| Linux | float32 | wrap | 4 | 32.53771 | 1.12301 | 28.974x | 1.037x | verified |
| Linux | float32 | constant | 1 | 32.43958 | 2.16426 | 14.989x | 1.035x | verified |
| Linux | float32 | constant | 4 | 32.41765 | 1.11886 | 28.974x | 1.015x | verified |
| Linux | float32 | fortran | 1 | 32.86392 | 2.75749 | 11.918x | 1.036x | verified |
| Linux | float32 | fortran | 4 | 32.84361 | 1.75952 | 18.666x | 1.025x | verified |
| Linux | float32 | strided | 1 | 16.45051 | 1.17404 | 14.012x | 1.035x | verified |
| Linux | float32 | strided | 4 | 16.45072 | 1.18648 | 13.865x | 1.029x | verified |
| Linux | float32 | dense5 | 1 | 107.62856 | 8.30707 | 12.956x | 1.002x | verified |
| Linux | float32 | dense5 | 4 | 107.62693 | 4.14230 | 25.982x | 0.988x | verified |
| Linux | float32 | sparse7 | 1 | 32.74802 | 2.18402 | 14.994x | 1.028x | verified |
| Linux | float32 | sparse7 | 4 | 32.72569 | 1.11677 | 29.304x | 1.043x | verified |
| Linux | float32 | cross | 1 | 13.40328 | 1.59829 | 8.386x | 1.050x | verified |
| Linux | float32 | cross | 4 | 13.38576 | 0.86655 | 15.447x | 1.032x | verified |
| Linux | float32 | even | 1 | 8.98143 | 1.55500 | 5.776x | 1.043x | verified |
| Linux | float32 | even | 4 | 8.93135 | 0.83225 | 10.732x | 1.047x | verified |
| Linux | float32 | identity | 1 | 0.08479 | 0.05038 | 1.683x | 1.007x | verified |
| Linux | float32 | identity | 4 | 0.08455 | 0.05087 | 1.662x | 0.989x | verified |
| Linux | float32 | tiny | 1 | 0.01798 | 0.01702 | 1.057x | 0.950x | verified |
| Linux | float32 | tiny | 4 | 0.01783 | 0.01676 | 1.063x | 0.969x | verified |
| Linux | float32 | one-dimensional3 | 1 | 4.35486 | 1.59324 | 2.733x | 1.053x | verified |
| Linux | float32 | one-dimensional3 | 4 | 4.38346 | 0.87183 | 5.028x | 1.017x | verified |
| Linux | float32 | one-dimensional31 | 1 | 11.37562 | 11.50605 | 0.989x | 1.006x | verified |
| Linux | float32 | one-dimensional31 | 4 | 11.36386 | 3.95396 | 2.874x | 1.006x | verified |
| Linux | float32 | three-dimensional | 1 | 19.04524 | 1.83115 | 10.401x | 0.995x | verified |
| Linux | float32 | three-dimensional | 4 | 19.05772 | 1.82342 | 10.452x | 1.000x | verified |
| Linux | float32 | signed-zero3 | 1 | 3.28951 | 5.39279 | 0.610x | 1.012x | verified |
| Linux | float32 | signed-zero3 | 4 | 3.27760 | 5.40831 | 0.606x | 1.011x | verified |
| Linux | float32 | signed-zero31 | 1 | 7.77782 | 9.89959 | 0.786x | 1.016x | verified |
| Linux | float32 | signed-zero31 | 4 | 7.79262 | 9.90492 | 0.787x | 1.015x | verified |
| Linux | float32 | signed-zero-pair | 1 | 1.89747 | 6.77864 | 0.280x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float32 | signed-zero-pair | 4 | 1.89602 | 3.44399 | 0.551x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float64 | nearest | 1 | 32.32018 | 1.83410 | 17.622x | 0.998x | verified |
| Linux | float64 | nearest | 4 | 32.30588 | 1.18832 | 27.186x | 0.986x | verified |
| Linux | float64 | reflect | 1 | 32.29225 | 1.83788 | 17.570x | 0.996x | verified |
| Linux | float64 | reflect | 4 | 32.35036 | 1.20367 | 26.876x | 0.987x | verified |
| Linux | float64 | mirror | 1 | 32.31024 | 1.83457 | 17.612x | 0.994x | verified |
| Linux | float64 | mirror | 4 | 32.25308 | 1.19502 | 26.990x | 0.987x | verified |
| Linux | float64 | wrap | 1 | 32.37110 | 1.85801 | 17.422x | 0.993x | verified |
| Linux | float64 | wrap | 4 | 32.29945 | 1.19190 | 27.099x | 0.980x | verified |
| Linux | float64 | constant | 1 | 32.35033 | 1.82147 | 17.761x | 1.002x | verified |
| Linux | float64 | constant | 4 | 32.36159 | 1.18075 | 27.408x | 1.004x | verified |
| Linux | float64 | fortran | 1 | 32.97794 | 3.81501 | 8.644x | 1.004x | verified |
| Linux | float64 | fortran | 4 | 33.07498 | 3.31357 | 9.982x | 0.998x | verified |
| Linux | float64 | strided | 1 | 16.40096 | 1.01862 | 16.101x | 1.002x | verified |
| Linux | float64 | strided | 4 | 16.42154 | 1.04623 | 15.696x | 0.977x | verified |
| Linux | float64 | dense5 | 1 | 107.18016 | 7.95254 | 13.477x | 0.996x | verified |
| Linux | float64 | dense5 | 4 | 107.14834 | 4.13387 | 25.920x | 1.001x | verified |
| Linux | float64 | sparse7 | 1 | 32.56794 | 1.85575 | 17.550x | 0.994x | verified |
| Linux | float64 | sparse7 | 4 | 32.54865 | 1.19645 | 27.204x | 0.980x | verified |
| Linux | float64 | cross | 1 | 12.99994 | 1.28313 | 10.131x | 0.989x | verified |
| Linux | float64 | cross | 4 | 13.07165 | 0.92938 | 14.065x | 0.960x | verified |
| Linux | float64 | even | 1 | 8.60573 | 1.20126 | 7.164x | 1.004x | verified |
| Linux | float64 | even | 4 | 8.74073 | 0.88923 | 9.830x | 0.966x | verified |
| Linux | float64 | identity | 1 | 0.13751 | 0.08382 | 1.641x | 0.991x | verified |
| Linux | float64 | identity | 4 | 0.13890 | 0.08528 | 1.629x | 0.969x | verified |
| Linux | float64 | tiny | 1 | 0.01778 | 0.01702 | 1.045x | 0.929x | verified |
| Linux | float64 | tiny | 4 | 0.01775 | 0.01687 | 1.052x | 0.959x | verified |
| Linux | float64 | one-dimensional3 | 1 | 4.41051 | 1.12809 | 3.910x | 1.010x | verified |
| Linux | float64 | one-dimensional3 | 4 | 4.41680 | 0.87304 | 5.059x | 1.015x | verified |
| Linux | float64 | one-dimensional31 | 1 | 11.50701 | 11.65634 | 0.987x | 0.973x | verified |
| Linux | float64 | one-dimensional31 | 4 | 11.55456 | 3.99583 | 2.892x | 0.986x | verified |
| Linux | float64 | three-dimensional | 1 | 18.98395 | 1.75822 | 10.797x | 1.003x | verified |
| Linux | float64 | three-dimensional | 4 | 18.99595 | 1.74851 | 10.864x | 1.014x | verified |
| Linux | float64 | signed-zero3 | 1 | 3.30762 | 5.33530 | 0.620x | 1.017x | verified |
| Linux | float64 | signed-zero3 | 4 | 3.31073 | 5.38080 | 0.615x | 1.054x | verified |
| Linux | float64 | signed-zero31 | 1 | 7.82940 | 10.00148 | 0.783x | 1.006x | verified |
| Linux | float64 | signed-zero31 | 4 | 7.81001 | 10.01445 | 0.780x | 1.006x | verified |
| Linux | float64 | signed-zero-pair | 1 | 1.93071 | 6.28343 | 0.307x | unmeasured | signed-zero-mismatch-unmeasured |
| Linux | float64 | signed-zero-pair | 4 | 1.94605 | 3.21920 | 0.605x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | uint8 | nearest | 1 | 24.88140 | 1.64253 | 15.148x | 1.006x | verified |
| Windows | uint8 | nearest | 4 | 24.46490 | 1.00673 | 24.301x | 0.997x | verified |
| Windows | uint8 | reflect | 1 | 24.56740 | 1.62250 | 15.142x | 1.018x | verified |
| Windows | uint8 | reflect | 4 | 25.11380 | 1.00532 | 24.981x | 1.019x | verified |
| Windows | uint8 | mirror | 1 | 24.27820 | 1.64410 | 14.767x | 0.985x | verified |
| Windows | uint8 | mirror | 4 | 24.27560 | 0.98781 | 24.575x | 1.007x | verified |
| Windows | uint8 | wrap | 1 | 24.54910 | 1.62976 | 15.063x | 1.000x | verified |
| Windows | uint8 | wrap | 4 | 24.49730 | 1.00861 | 24.288x | 0.999x | verified |
| Windows | uint8 | constant | 1 | 24.77370 | 1.62061 | 15.287x | 1.001x | verified |
| Windows | uint8 | constant | 4 | 24.68510 | 0.98595 | 25.037x | 1.010x | verified |
| Windows | uint8 | fortran | 1 | 24.98450 | 1.83366 | 13.625x | 1.002x | verified |
| Windows | uint8 | fortran | 4 | 24.91690 | 1.20655 | 20.651x | 0.994x | verified |
| Windows | uint8 | strided | 1 | 12.34350 | 0.88817 | 13.898x | 0.994x | verified |
| Windows | uint8 | strided | 4 | 12.41220 | 0.89709 | 13.836x | 0.996x | verified |
| Windows | uint8 | dense5 | 1 | 82.61480 | 18.64230 | 4.432x | 1.019x | verified |
| Windows | uint8 | dense5 | 4 | 82.74830 | 8.43380 | 9.812x | 0.974x | verified |
| Windows | uint8 | sparse7 | 1 | 24.70570 | 1.60601 | 15.383x | 1.017x | verified |
| Windows | uint8 | sparse7 | 4 | 24.62570 | 0.97917 | 25.150x | 1.012x | verified |
| Windows | uint8 | cross | 1 | 9.76780 | 1.17875 | 8.287x | 1.001x | verified |
| Windows | uint8 | cross | 4 | 9.71080 | 0.76345 | 12.720x | 1.009x | verified |
| Windows | uint8 | even | 1 | 6.33795 | 1.10178 | 5.752x | 1.017x | verified |
| Windows | uint8 | even | 4 | 6.41375 | 0.73561 | 8.719x | 1.016x | verified |
| Windows | uint8 | identity | 1 | 0.01381 | 0.00586 | 2.355x | 0.997x | verified |
| Windows | uint8 | identity | 4 | 0.01353 | 0.00567 | 2.384x | 1.005x | verified |
| Windows | uint8 | tiny | 1 | 0.00722 | 0.00807 | 0.894x | 1.002x | verified |
| Windows | uint8 | tiny | 4 | 0.00740 | 0.00828 | 0.894x | 0.997x | verified |
| Windows | uint8 | one-dimensional3 | 1 | 3.30877 | 1.26963 | 2.606x | 1.002x | verified |
| Windows | uint8 | one-dimensional3 | 4 | 3.32035 | 1.09425 | 3.034x | 0.999x | verified |
| Windows | uint8 | one-dimensional31 | 1 | 7.48050 | 7.38900 | 1.012x | 0.996x | verified |
| Windows | uint8 | one-dimensional31 | 4 | 7.54600 | 2.57086 | 2.935x | 1.000x | verified |
| Windows | uint8 | three-dimensional | 1 | 13.75720 | 3.08350 | 4.462x | 1.035x | verified |
| Windows | uint8 | three-dimensional | 4 | 14.44000 | 3.26210 | 4.427x | 1.040x | verified |
| Windows | float32 | nearest | 1 | 25.66720 | 2.20365 | 11.648x | 0.994x | verified |
| Windows | float32 | nearest | 4 | 24.67570 | 1.37556 | 17.939x | 0.995x | verified |
| Windows | float32 | reflect | 1 | 24.78520 | 2.05458 | 12.063x | 1.040x | verified |
| Windows | float32 | reflect | 4 | 24.52220 | 1.38356 | 17.724x | 1.002x | verified |
| Windows | float32 | mirror | 1 | 24.20250 | 2.04793 | 11.818x | 1.019x | verified |
| Windows | float32 | mirror | 4 | 24.56770 | 1.39531 | 17.607x | 0.993x | verified |
| Windows | float32 | wrap | 1 | 25.07380 | 2.11943 | 11.830x | 1.020x | verified |
| Windows | float32 | wrap | 4 | 24.82280 | 1.39974 | 17.734x | 1.007x | verified |
| Windows | float32 | constant | 1 | 24.90900 | 2.10443 | 11.836x | 1.028x | verified |
| Windows | float32 | constant | 4 | 24.73930 | 1.38944 | 17.805x | 0.999x | verified |
| Windows | float32 | fortran | 1 | 25.09170 | 2.49468 | 10.058x | 1.020x | verified |
| Windows | float32 | fortran | 4 | 25.01750 | 1.83220 | 13.654x | 0.989x | verified |
| Windows | float32 | strided | 1 | 12.50090 | 1.05285 | 11.873x | 1.031x | verified |
| Windows | float32 | strided | 4 | 12.39110 | 1.03607 | 11.960x | 1.028x | verified |
| Windows | float32 | dense5 | 1 | 82.63820 | 18.98790 | 4.352x | 1.019x | verified |
| Windows | float32 | dense5 | 4 | 84.18660 | 8.59440 | 9.796x | 0.999x | verified |
| Windows | float32 | sparse7 | 1 | 25.01920 | 2.08810 | 11.982x | 1.014x | verified |
| Windows | float32 | sparse7 | 4 | 24.48060 | 1.36785 | 17.897x | 0.986x | verified |
| Windows | float32 | cross | 1 | 9.45630 | 1.65056 | 5.729x | 1.037x | verified |
| Windows | float32 | cross | 4 | 9.49320 | 1.23251 | 7.702x | 1.006x | verified |
| Windows | float32 | even | 1 | 6.24790 | 1.58904 | 3.932x | 1.058x | verified |
| Windows | float32 | even | 4 | 6.35775 | 1.23936 | 5.130x | 1.026x | verified |
| Windows | float32 | identity | 1 | 0.16811 | 0.15914 | 1.056x | 1.019x | verified |
| Windows | float32 | identity | 4 | 0.18360 | 0.17496 | 1.049x | 0.997x | verified |
| Windows | float32 | tiny | 1 | 0.00728 | 0.00885 | 0.823x | 1.010x | verified |
| Windows | float32 | tiny | 4 | 0.00754 | 0.00914 | 0.825x | 0.997x | verified |
| Windows | float32 | one-dimensional3 | 1 | 3.33215 | 1.57711 | 2.113x | 1.036x | verified |
| Windows | float32 | one-dimensional3 | 4 | 3.30817 | 1.29085 | 2.563x | 1.041x | verified |
| Windows | float32 | one-dimensional31 | 1 | 8.07360 | 8.32920 | 0.969x | 1.000x | verified |
| Windows | float32 | one-dimensional31 | 4 | 7.93510 | 2.81054 | 2.823x | 0.999x | verified |
| Windows | float32 | three-dimensional | 1 | 14.38860 | 3.35193 | 4.293x | 1.013x | verified |
| Windows | float32 | three-dimensional | 4 | 14.40460 | 3.34375 | 4.308x | 1.026x | verified |
| Windows | float32 | signed-zero3 | 1 | 2.35683 | 4.58767 | 0.514x | 0.992x | verified |
| Windows | float32 | signed-zero3 | 4 | 2.36834 | 4.58347 | 0.517x | 0.985x | verified |
| Windows | float32 | signed-zero31 | 1 | 5.10370 | 7.81670 | 0.653x | 1.037x | verified |
| Windows | float32 | signed-zero31 | 4 | 5.30270 | 8.15490 | 0.650x | 1.012x | verified |
| Windows | float32 | signed-zero-pair | 1 | 1.44696 | 5.56910 | 0.260x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float32 | signed-zero-pair | 4 | 1.43902 | 3.48563 | 0.413x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float64 | nearest | 1 | 23.44190 | 1.89174 | 12.392x | 0.993x | verified |
| Windows | float64 | nearest | 4 | 24.13500 | 1.27688 | 18.902x | 1.000x | verified |
| Windows | float64 | reflect | 1 | 25.43140 | 2.00174 | 12.705x | 1.009x | verified |
| Windows | float64 | reflect | 4 | 24.15840 | 1.27366 | 18.968x | 0.998x | verified |
| Windows | float64 | mirror | 1 | 24.38220 | 1.93117 | 12.626x | 0.996x | verified |
| Windows | float64 | mirror | 4 | 24.51960 | 1.29281 | 18.966x | 1.012x | verified |
| Windows | float64 | wrap | 1 | 24.65050 | 1.93741 | 12.723x | 0.998x | verified |
| Windows | float64 | wrap | 4 | 23.81840 | 1.26552 | 18.821x | 0.995x | verified |
| Windows | float64 | constant | 1 | 24.07690 | 1.93122 | 12.467x | 0.990x | verified |
| Windows | float64 | constant | 4 | 23.86720 | 1.25052 | 19.086x | 1.005x | verified |
| Windows | float64 | fortran | 1 | 24.47740 | 3.29013 | 7.440x | 0.995x | verified |
| Windows | float64 | fortran | 4 | 24.49860 | 3.16323 | 7.745x | 0.992x | verified |
| Windows | float64 | strided | 1 | 12.36770 | 1.21297 | 10.196x | 0.998x | verified |
| Windows | float64 | strided | 4 | 11.80790 | 1.20985 | 9.760x | 1.026x | verified |
| Windows | float64 | dense5 | 1 | 81.23630 | 18.67070 | 4.351x | 1.023x | verified |
| Windows | float64 | dense5 | 4 | 81.47650 | 8.40830 | 9.690x | 1.011x | verified |
| Windows | float64 | sparse7 | 1 | 24.77670 | 1.95191 | 12.694x | 0.995x | verified |
| Windows | float64 | sparse7 | 4 | 24.55680 | 1.29755 | 18.926x | 1.015x | verified |
| Windows | float64 | cross | 1 | 9.52870 | 1.60022 | 5.955x | 1.018x | verified |
| Windows | float64 | cross | 4 | 9.57850 | 1.21533 | 7.881x | 0.987x | verified |
| Windows | float64 | even | 1 | 6.16860 | 1.45468 | 4.241x | 1.021x | verified |
| Windows | float64 | even | 4 | 6.17645 | 1.12312 | 5.499x | 0.982x | verified |
| Windows | float64 | identity | 1 | 0.29159 | 0.28109 | 1.037x | 1.021x | verified |
| Windows | float64 | identity | 4 | 0.29000 | 0.28274 | 1.026x | 1.007x | verified |
| Windows | float64 | tiny | 1 | 0.00731 | 0.00886 | 0.824x | 1.023x | verified |
| Windows | float64 | tiny | 4 | 0.00733 | 0.00869 | 0.843x | 1.015x | verified |
| Windows | float64 | one-dimensional3 | 1 | 3.25260 | 1.45720 | 2.232x | 0.995x | verified |
| Windows | float64 | one-dimensional3 | 4 | 3.26202 | 1.14276 | 2.855x | 1.007x | verified |
| Windows | float64 | one-dimensional31 | 1 | 8.27700 | 8.72000 | 0.949x | 1.013x | verified |
| Windows | float64 | one-dimensional31 | 4 | 8.14180 | 2.94588 | 2.764x | 1.008x | verified |
| Windows | float64 | three-dimensional | 1 | 14.43200 | 3.30990 | 4.360x | 1.014x | verified |
| Windows | float64 | three-dimensional | 4 | 14.41130 | 3.26622 | 4.412x | 1.026x | verified |
| Windows | float64 | signed-zero3 | 1 | 2.52280 | 4.88955 | 0.516x | 1.004x | verified |
| Windows | float64 | signed-zero3 | 4 | 2.55002 | 4.93330 | 0.517x | 1.009x | verified |
| Windows | float64 | signed-zero31 | 1 | 5.42330 | 8.56440 | 0.633x | 0.998x | verified |
| Windows | float64 | signed-zero31 | 4 | 5.44765 | 8.58030 | 0.635x | 1.000x | verified |
| Windows | float64 | signed-zero-pair | 1 | 1.69407 | 5.84485 | 0.290x | unmeasured | signed-zero-mismatch-unmeasured |
| Windows | float64 | signed-zero-pair | 4 | 1.69728 | 3.78005 | 0.449x | unmeasured | signed-zero-mismatch-unmeasured |

## Rejected centered-pointer trial: `f60755a`

[Paired run](https://github.com/bipxx/scikit-imagexx/actions/runs/39822039095) against correct source `7f1d317` from run `39820988391`. Product fingerprint: `c9593d74be2934f85c5ba986fc7466306ca73ec87c9dadaf0107ab4aad2d3d0a`. The trial combines index storage and caches centered heap/value pointers, preserving insertion and tie order. Scoped checks pass 8,205 Windows and 6,052 minimum-NumPy cases per SIMD mode; all five native configurations pass 2/2. The benchmark blob and ordered case identities match the 54-case baseline.

Linux: **6.102861x upstream/native**, **1.004086x paired baseline/current**, across all 54 one-worker cases. [All raw samples](median-centered-linux.json).

Windows: **3.808615x upstream/native**, **0.981089x paired baseline/current**, across all 54 one-worker cases. [All raw samples](median-centered-windows.json).

**Discarded:** the fixed minimum-OS metric is 3.808615x, below the 4.227715x baseline. Paired signed-zero replay improves roughly 2-5% on Linux and 3-8% on Windows, but the Windows corpus regresses 1.9% overall. Dense signed-zero pair cases remain only 0.25-0.30x upstream. All target gains and adverse controls are retained below. The next candidate restores the original index layout and moves signed-zero output-type dispatch outside the pixel loop.

| OS | dtype | layout | worker limit | upstream ms | native ms | upstream/native | baseline/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | nearest | 1 | 32.71146 | 1.74767 | 18.717x | 0.995x |
| Linux | uint8 | nearest | 4 | 32.68345 | 0.97338 | 33.577x | 0.991x |
| Linux | uint8 | reflect | 1 | 32.77232 | 1.71358 | 19.125x | 0.994x |
| Linux | uint8 | reflect | 4 | 32.74583 | 0.96696 | 33.865x | 1.009x |
| Linux | uint8 | mirror | 1 | 32.68357 | 1.69109 | 19.327x | 0.994x |
| Linux | uint8 | mirror | 4 | 32.74858 | 0.96849 | 33.814x | 0.998x |
| Linux | uint8 | wrap | 1 | 32.81253 | 1.70260 | 19.272x | 1.001x |
| Linux | uint8 | wrap | 4 | 32.81360 | 0.97823 | 33.544x | 0.999x |
| Linux | uint8 | constant | 1 | 32.69534 | 1.70121 | 19.219x | 0.999x |
| Linux | uint8 | constant | 4 | 32.65612 | 0.96947 | 33.684x | 0.992x |
| Linux | uint8 | fortran | 1 | 32.92687 | 2.05853 | 15.995x | 0.995x |
| Linux | uint8 | fortran | 4 | 32.85270 | 1.33848 | 24.545x | 1.006x |
| Linux | uint8 | strided | 1 | 16.57683 | 0.94131 | 17.610x | 0.997x |
| Linux | uint8 | strided | 4 | 16.54545 | 0.93932 | 17.614x | 0.993x |
| Linux | uint8 | dense5 | 1 | 107.57242 | 7.71774 | 13.938x | 1.011x |
| Linux | uint8 | dense5 | 4 | 107.58363 | 3.87096 | 27.792x | 0.996x |
| Linux | uint8 | sparse7 | 1 | 32.99150 | 1.70239 | 19.380x | 1.005x |
| Linux | uint8 | sparse7 | 4 | 33.04059 | 0.96455 | 34.255x | 1.003x |
| Linux | uint8 | cross | 1 | 13.68038 | 1.13686 | 12.033x | 0.991x |
| Linux | uint8 | cross | 4 | 13.65663 | 0.70415 | 19.394x | 0.998x |
| Linux | uint8 | even | 1 | 9.19126 | 1.05085 | 8.746x | 1.009x |
| Linux | uint8 | even | 4 | 9.19003 | 0.67028 | 13.711x | 0.992x |
| Linux | uint8 | identity | 1 | 0.03744 | 0.01798 | 2.083x | 0.983x |
| Linux | uint8 | identity | 4 | 0.03772 | 0.01678 | 2.247x | 1.070x |
| Linux | uint8 | tiny | 1 | 0.01741 | 0.01463 | 1.190x | 1.033x |
| Linux | uint8 | tiny | 4 | 0.01746 | 0.01454 | 1.201x | 1.045x |
| Linux | uint8 | one-dimensional3 | 1 | 3.92671 | 0.47409 | 8.283x | 0.974x |
| Linux | uint8 | one-dimensional3 | 4 | 3.96665 | 0.45225 | 8.771x | 0.990x |
| Linux | uint8 | one-dimensional31 | 1 | 9.95007 | 10.15008 | 0.980x | 1.009x |
| Linux | uint8 | one-dimensional31 | 4 | 9.91073 | 3.49876 | 2.833x | 1.012x |
| Linux | uint8 | three-dimensional | 1 | 19.00769 | 1.73958 | 10.927x | 1.004x |
| Linux | uint8 | three-dimensional | 4 | 19.12541 | 1.75066 | 10.925x | 1.004x |
| Linux | float32 | nearest | 1 | 32.49763 | 2.13909 | 15.192x | 1.004x |
| Linux | float32 | nearest | 4 | 32.54330 | 1.08844 | 29.899x | 1.009x |
| Linux | float32 | reflect | 1 | 32.55885 | 2.14663 | 15.167x | 0.999x |
| Linux | float32 | reflect | 4 | 32.48116 | 1.09436 | 29.680x | 0.998x |
| Linux | float32 | mirror | 1 | 32.45357 | 2.13292 | 15.216x | 1.001x |
| Linux | float32 | mirror | 4 | 32.53528 | 1.08506 | 29.985x | 0.999x |
| Linux | float32 | wrap | 1 | 32.54178 | 2.14881 | 15.144x | 1.009x |
| Linux | float32 | wrap | 4 | 32.62228 | 1.09904 | 29.683x | 0.994x |
| Linux | float32 | constant | 1 | 32.61805 | 2.13936 | 15.247x | 1.009x |
| Linux | float32 | constant | 4 | 32.49561 | 1.09832 | 29.587x | 1.002x |
| Linux | float32 | fortran | 1 | 32.82690 | 2.79314 | 11.753x | 0.989x |
| Linux | float32 | fortran | 4 | 32.70548 | 1.72773 | 18.930x | 1.010x |
| Linux | float32 | strided | 1 | 16.53726 | 1.17036 | 14.130x | 0.999x |
| Linux | float32 | strided | 4 | 16.45125 | 1.17418 | 14.011x | 0.996x |
| Linux | float32 | dense5 | 1 | 107.70956 | 8.25139 | 13.054x | 1.004x |
| Linux | float32 | dense5 | 4 | 107.82828 | 4.03993 | 26.691x | 1.008x |
| Linux | float32 | sparse7 | 1 | 32.85278 | 2.15448 | 15.249x | 0.999x |
| Linux | float32 | sparse7 | 4 | 32.84340 | 1.09562 | 29.977x | 0.997x |
| Linux | float32 | cross | 1 | 13.41344 | 1.58800 | 8.447x | 1.008x |
| Linux | float32 | cross | 4 | 13.50891 | 0.83143 | 16.248x | 1.003x |
| Linux | float32 | even | 1 | 8.93983 | 1.54109 | 5.801x | 1.003x |
| Linux | float32 | even | 4 | 8.92566 | 0.79543 | 11.221x | 1.006x |
| Linux | float32 | identity | 1 | 0.08258 | 0.04803 | 1.719x | 1.008x |
| Linux | float32 | identity | 4 | 0.08139 | 0.04858 | 1.675x | 0.998x |
| Linux | float32 | tiny | 1 | 0.01748 | 0.01585 | 1.103x | 1.020x |
| Linux | float32 | tiny | 4 | 0.01795 | 0.01605 | 1.119x | 1.000x |
| Linux | float32 | one-dimensional3 | 1 | 4.38220 | 1.53635 | 2.852x | 0.988x |
| Linux | float32 | one-dimensional3 | 4 | 4.40201 | 0.78833 | 5.584x | 1.012x |
| Linux | float32 | one-dimensional31 | 1 | 11.38321 | 11.50309 | 0.990x | 1.000x |
| Linux | float32 | one-dimensional31 | 4 | 11.41246 | 3.92688 | 2.906x | 0.996x |
| Linux | float32 | three-dimensional | 1 | 19.07767 | 1.76855 | 10.787x | 1.031x |
| Linux | float32 | three-dimensional | 4 | 19.14175 | 1.77759 | 10.768x | 1.030x |
| Linux | float32 | signed-zero3 | 1 | 3.24058 | 5.13019 | 0.632x | 1.053x |
| Linux | float32 | signed-zero3 | 4 | 3.28610 | 5.16105 | 0.637x | 1.049x |
| Linux | float32 | signed-zero31 | 1 | 7.83945 | 9.72230 | 0.806x | 1.017x |
| Linux | float32 | signed-zero31 | 4 | 7.82209 | 9.57137 | 0.817x | 1.033x |
| Linux | float32 | signed-zero-pair | 1 | 1.89552 | 6.69249 | 0.283x | 1.018x |
| Linux | float32 | signed-zero-pair | 4 | 1.88790 | 3.14378 | 0.601x | 1.019x |
| Linux | float64 | nearest | 1 | 32.24933 | 1.80006 | 17.916x | 1.013x |
| Linux | float64 | nearest | 4 | 32.35040 | 1.15210 | 28.079x | 1.002x |
| Linux | float64 | reflect | 1 | 32.33023 | 1.81960 | 17.768x | 1.000x |
| Linux | float64 | reflect | 4 | 32.39550 | 1.15175 | 28.127x | 1.003x |
| Linux | float64 | mirror | 1 | 32.28583 | 1.81119 | 17.826x | 1.007x |
| Linux | float64 | mirror | 4 | 32.38056 | 1.15255 | 28.095x | 0.996x |
| Linux | float64 | wrap | 1 | 32.30506 | 1.83292 | 17.625x | 1.002x |
| Linux | float64 | wrap | 4 | 32.45967 | 1.15216 | 28.173x | 1.002x |
| Linux | float64 | constant | 1 | 32.30135 | 1.80683 | 17.877x | 0.998x |
| Linux | float64 | constant | 4 | 32.24744 | 1.15566 | 27.904x | 0.994x |
| Linux | float64 | fortran | 1 | 32.93393 | 3.63390 | 9.063x | 1.017x |
| Linux | float64 | fortran | 4 | 32.95021 | 3.06115 | 10.764x | 1.045x |
| Linux | float64 | strided | 1 | 16.36904 | 1.00271 | 16.325x | 1.014x |
| Linux | float64 | strided | 4 | 16.36778 | 1.01307 | 16.157x | 0.994x |
| Linux | float64 | dense5 | 1 | 107.02725 | 8.15973 | 13.117x | 1.009x |
| Linux | float64 | dense5 | 4 | 107.21734 | 4.10850 | 26.096x | 1.001x |
| Linux | float64 | sparse7 | 1 | 32.60193 | 1.83370 | 17.779x | 0.986x |
| Linux | float64 | sparse7 | 4 | 32.53127 | 1.15502 | 28.165x | 1.003x |
| Linux | float64 | cross | 1 | 13.07224 | 1.25772 | 10.394x | 1.014x |
| Linux | float64 | cross | 4 | 13.06304 | 0.88636 | 14.738x | 1.009x |
| Linux | float64 | even | 1 | 8.56035 | 1.19537 | 7.161x | 1.007x |
| Linux | float64 | even | 4 | 8.66907 | 0.84829 | 10.219x | 0.999x |
| Linux | float64 | identity | 1 | 0.13875 | 0.08436 | 1.645x | 0.986x |
| Linux | float64 | identity | 4 | 0.14171 | 0.08420 | 1.683x | 1.008x |
| Linux | float64 | tiny | 1 | 0.01801 | 0.01665 | 1.081x | 0.972x |
| Linux | float64 | tiny | 4 | 0.01744 | 0.01609 | 1.084x | 0.971x |
| Linux | float64 | one-dimensional3 | 1 | 4.44805 | 1.12085 | 3.968x | 0.999x |
| Linux | float64 | one-dimensional3 | 4 | 4.43175 | 0.83194 | 5.327x | 1.008x |
| Linux | float64 | one-dimensional31 | 1 | 11.45149 | 11.23480 | 1.019x | 1.042x |
| Linux | float64 | one-dimensional31 | 4 | 11.45612 | 3.90133 | 2.936x | 1.019x |
| Linux | float64 | three-dimensional | 1 | 18.95696 | 1.78925 | 10.595x | 0.977x |
| Linux | float64 | three-dimensional | 4 | 18.94272 | 1.78249 | 10.627x | 0.980x |
| Linux | float64 | signed-zero3 | 1 | 3.32606 | 5.28213 | 0.630x | 1.023x |
| Linux | float64 | signed-zero3 | 4 | 3.34088 | 5.27741 | 0.633x | 1.023x |
| Linux | float64 | signed-zero31 | 1 | 7.91381 | 9.86929 | 0.802x | 1.019x |
| Linux | float64 | signed-zero31 | 4 | 7.90160 | 9.88136 | 0.800x | 1.022x |
| Linux | float64 | signed-zero-pair | 1 | 1.93237 | 6.41518 | 0.301x | 0.988x |
| Linux | float64 | signed-zero-pair | 4 | 1.94173 | 3.36385 | 0.577x | 0.971x |
| Windows | uint8 | nearest | 1 | 28.31780 | 2.32777 | 12.165x | 0.968x |
| Windows | uint8 | nearest | 4 | 28.31180 | 1.41848 | 19.959x | 0.984x |
| Windows | uint8 | reflect | 1 | 28.32370 | 2.31600 | 12.230x | 0.969x |
| Windows | uint8 | reflect | 4 | 28.24010 | 1.42104 | 19.873x | 0.985x |
| Windows | uint8 | mirror | 1 | 28.24310 | 2.34143 | 12.062x | 0.957x |
| Windows | uint8 | mirror | 4 | 28.22260 | 1.38743 | 20.342x | 0.993x |
| Windows | uint8 | wrap | 1 | 28.37290 | 2.31933 | 12.233x | 0.974x |
| Windows | uint8 | wrap | 4 | 28.26350 | 1.40971 | 20.049x | 0.990x |
| Windows | uint8 | constant | 1 | 28.18930 | 2.31567 | 12.173x | 0.966x |
| Windows | uint8 | constant | 4 | 28.16100 | 1.37686 | 20.453x | 1.012x |
| Windows | uint8 | fortran | 1 | 28.60220 | 2.54446 | 11.241x | 0.954x |
| Windows | uint8 | fortran | 4 | 28.59740 | 1.56228 | 18.305x | 0.990x |
| Windows | uint8 | strided | 1 | 14.67710 | 1.27566 | 11.505x | 0.973x |
| Windows | uint8 | strided | 4 | 14.41620 | 1.28707 | 11.201x | 0.961x |
| Windows | uint8 | dense5 | 1 | 93.72770 | 25.85860 | 3.625x | 0.997x |
| Windows | uint8 | dense5 | 4 | 94.56380 | 11.75140 | 8.047x | 0.999x |
| Windows | uint8 | sparse7 | 1 | 28.51910 | 2.33383 | 12.220x | 0.964x |
| Windows | uint8 | sparse7 | 4 | 28.49200 | 1.39642 | 20.404x | 0.999x |
| Windows | uint8 | cross | 1 | 11.47120 | 1.83307 | 6.258x | 0.958x |
| Windows | uint8 | cross | 4 | 11.48480 | 1.17331 | 9.788x | 0.998x |
| Windows | uint8 | even | 1 | 7.85660 | 1.71746 | 4.575x | 0.962x |
| Windows | uint8 | even | 4 | 7.83720 | 1.11678 | 7.018x | 0.995x |
| Windows | uint8 | identity | 1 | 0.02245 | 0.01074 | 2.091x | 0.986x |
| Windows | uint8 | identity | 4 | 0.02206 | 0.01067 | 2.066x | 0.999x |
| Windows | uint8 | tiny | 1 | 0.01169 | 0.01053 | 1.110x | 0.969x |
| Windows | uint8 | tiny | 4 | 0.01209 | 0.01081 | 1.119x | 0.978x |
| Windows | uint8 | one-dimensional3 | 1 | 4.51977 | 1.82690 | 2.474x | 0.990x |
| Windows | uint8 | one-dimensional3 | 4 | 4.58327 | 1.61429 | 2.839x | 0.995x |
| Windows | uint8 | one-dimensional31 | 1 | 9.33710 | 9.06600 | 1.030x | 1.000x |
| Windows | uint8 | one-dimensional31 | 4 | 9.32700 | 3.40808 | 2.737x | 1.019x |
| Windows | uint8 | three-dimensional | 1 | 16.54610 | 4.50223 | 3.675x | 0.994x |
| Windows | uint8 | three-dimensional | 4 | 16.55060 | 4.46870 | 3.704x | 1.005x |
| Windows | float32 | nearest | 1 | 28.43420 | 2.87843 | 9.878x | 0.977x |
| Windows | float32 | nearest | 4 | 28.42520 | 1.91310 | 14.858x | 0.993x |
| Windows | float32 | reflect | 1 | 28.40430 | 2.88837 | 9.834x | 0.974x |
| Windows | float32 | reflect | 4 | 28.42320 | 1.92563 | 14.760x | 0.996x |
| Windows | float32 | mirror | 1 | 28.42780 | 2.88480 | 9.854x | 0.972x |
| Windows | float32 | mirror | 4 | 28.50320 | 1.93403 | 14.738x | 0.987x |
| Windows | float32 | wrap | 1 | 28.83280 | 2.97693 | 9.685x | 0.975x |
| Windows | float32 | wrap | 4 | 28.84940 | 2.13808 | 13.493x | 0.980x |
| Windows | float32 | constant | 1 | 28.54680 | 2.89320 | 9.867x | 0.969x |
| Windows | float32 | constant | 4 | 28.47750 | 1.96501 | 14.492x | 0.978x |
| Windows | float32 | fortran | 1 | 29.02780 | 3.60997 | 8.041x | 0.976x |
| Windows | float32 | fortran | 4 | 28.99430 | 2.69254 | 10.768x | 0.976x |
| Windows | float32 | strided | 1 | 14.33480 | 1.48559 | 9.649x | 0.970x |
| Windows | float32 | strided | 4 | 14.33890 | 1.48780 | 9.638x | 0.990x |
| Windows | float32 | dense5 | 1 | 95.36750 | 26.59300 | 3.586x | 1.001x |
| Windows | float32 | dense5 | 4 | 95.36410 | 14.21500 | 6.709x | 1.000x |
| Windows | float32 | sparse7 | 1 | 29.45990 | 2.99377 | 9.840x | 0.989x |
| Windows | float32 | sparse7 | 4 | 28.84540 | 2.00633 | 14.377x | 0.964x |
| Windows | float32 | cross | 1 | 11.52020 | 2.38020 | 4.840x | 0.969x |
| Windows | float32 | cross | 4 | 11.54860 | 1.81091 | 6.377x | 0.978x |
| Windows | float32 | even | 1 | 7.89890 | 2.32367 | 3.399x | 0.945x |
| Windows | float32 | even | 4 | 7.85330 | 2.33356 | 3.365x | 0.733x |
| Windows | float32 | identity | 1 | 0.26438 | 0.25443 | 1.039x | 0.986x |
| Windows | float32 | identity | 4 | 0.24616 | 0.22953 | 1.072x | 1.048x |
| Windows | float32 | tiny | 1 | 0.01151 | 0.01163 | 0.990x | 1.000x |
| Windows | float32 | tiny | 4 | 0.01136 | 0.01201 | 0.946x | 0.966x |
| Windows | float32 | one-dimensional3 | 1 | 4.21177 | 2.25262 | 1.870x | 0.968x |
| Windows | float32 | one-dimensional3 | 4 | 4.21787 | 1.87981 | 2.244x | 0.970x |
| Windows | float32 | one-dimensional31 | 1 | 9.98700 | 10.36060 | 0.964x | 0.996x |
| Windows | float32 | one-dimensional31 | 4 | 10.07220 | 4.17067 | 2.415x | 1.304x |
| Windows | float32 | three-dimensional | 1 | 16.91410 | 4.57490 | 3.697x | 1.017x |
| Windows | float32 | three-dimensional | 4 | 17.15730 | 4.62605 | 3.709x | 1.009x |
| Windows | float32 | signed-zero3 | 1 | 3.24685 | 5.22045 | 0.622x | 1.033x |
| Windows | float32 | signed-zero3 | 4 | 3.25248 | 5.25120 | 0.619x | 1.042x |
| Windows | float32 | signed-zero31 | 1 | 6.78005 | 9.35980 | 0.724x | 1.055x |
| Windows | float32 | signed-zero31 | 4 | 6.92740 | 9.66140 | 0.717x | 1.035x |
| Windows | float32 | signed-zero-pair | 1 | 1.86594 | 7.40295 | 0.252x | 0.977x |
| Windows | float32 | signed-zero-pair | 4 | 1.85469 | 5.72185 | 0.324x | 0.971x |
| Windows | float64 | nearest | 1 | 29.65940 | 2.81906 | 10.521x | 0.940x |
| Windows | float64 | nearest | 4 | 29.28720 | 2.52192 | 11.613x | 0.942x |
| Windows | float64 | reflect | 1 | 29.58630 | 2.75768 | 10.729x | 0.972x |
| Windows | float64 | reflect | 4 | 28.85560 | 2.42885 | 11.880x | 0.995x |
| Windows | float64 | mirror | 1 | 29.04060 | 2.67808 | 10.844x | 0.975x |
| Windows | float64 | mirror | 4 | 29.42730 | 2.44117 | 12.055x | 0.993x |
| Windows | float64 | wrap | 1 | 28.78260 | 2.75826 | 10.435x | 0.960x |
| Windows | float64 | wrap | 4 | 29.07570 | 1.85951 | 15.636x | 1.021x |
| Windows | float64 | constant | 1 | 28.87950 | 2.78076 | 10.385x | 0.964x |
| Windows | float64 | constant | 4 | 28.80370 | 2.44140 | 11.798x | 0.990x |
| Windows | float64 | fortran | 1 | 29.58600 | 3.89817 | 7.590x | 0.967x |
| Windows | float64 | fortran | 4 | 29.35300 | 3.09318 | 9.490x | 0.955x |
| Windows | float64 | strided | 1 | 14.68960 | 1.76277 | 8.333x | 0.979x |
| Windows | float64 | strided | 4 | 14.98700 | 1.79796 | 8.336x | 0.949x |
| Windows | float64 | dense5 | 1 | 96.29300 | 26.28440 | 3.664x | 1.052x |
| Windows | float64 | dense5 | 4 | 96.39190 | 18.45960 | 5.222x | 1.005x |
| Windows | float64 | sparse7 | 1 | 29.07050 | 2.71064 | 10.725x | 0.948x |
| Windows | float64 | sparse7 | 4 | 29.65570 | 2.35549 | 12.590x | 0.840x |
| Windows | float64 | cross | 1 | 11.52840 | 2.14210 | 5.382x | 0.948x |
| Windows | float64 | cross | 4 | 11.51360 | 1.94331 | 5.925x | 0.813x |
| Windows | float64 | even | 1 | 8.26810 | 2.17337 | 3.804x | 0.984x |
| Windows | float64 | even | 4 | 8.04560 | 1.93436 | 4.159x | 1.005x |
| Windows | float64 | identity | 1 | 0.38504 | 0.36462 | 1.056x | 0.969x |
| Windows | float64 | identity | 4 | 0.37233 | 0.35198 | 1.058x | 1.001x |
| Windows | float64 | tiny | 1 | 0.01196 | 0.01239 | 0.965x | 0.948x |
| Windows | float64 | tiny | 4 | 0.01187 | 0.01195 | 0.993x | 1.010x |
| Windows | float64 | one-dimensional3 | 1 | 4.46820 | 2.07400 | 2.154x | 0.966x |
| Windows | float64 | one-dimensional3 | 4 | 4.38877 | 1.79463 | 2.446x | 1.056x |
| Windows | float64 | one-dimensional31 | 1 | 10.26560 | 10.43050 | 0.984x | 1.009x |
| Windows | float64 | one-dimensional31 | 4 | 10.27270 | 3.57895 | 2.870x | 1.005x |
| Windows | float64 | three-dimensional | 1 | 16.95910 | 4.46957 | 3.794x | 1.002x |
| Windows | float64 | three-dimensional | 4 | 16.94260 | 4.47600 | 3.785x | 1.003x |
| Windows | float64 | signed-zero3 | 1 | 3.48873 | 5.49490 | 0.635x | 1.035x |
| Windows | float64 | signed-zero3 | 4 | 3.69210 | 5.76315 | 0.641x | 1.001x |
| Windows | float64 | signed-zero31 | 1 | 6.96875 | 9.16750 | 0.760x | 1.079x |
| Windows | float64 | signed-zero31 | 4 | 7.48215 | 9.60960 | 0.779x | 1.062x |
| Windows | float64 | signed-zero-pair | 1 | 2.06527 | 7.33760 | 0.281x | 0.975x |
| Windows | float64 | signed-zero-pair | 4 | 2.01373 | 4.27805 | 0.471x | 1.048x |
