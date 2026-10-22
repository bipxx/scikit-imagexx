# Streaming adaptive mean thresholds

Measured source: `258c4a2`. This follows the [overflow-policy correction](adaptive-final.md); the paired baseline remains the exact [initial native port](adaptive.md). No cross-run ratio is used to claim an improvement over the correction.

The first mean-filter axis previously packed four independent lines into double buffers, ran SIMD rolling sums, then scattered results. Those lines already occupy adjacent memory lanes. The retained implementation loads them directly, converts float32 lanes to double, performs the same ordered initialization/update and division, and writes rounded output directly. Portable builds use the same streaming order. Other axes retain the packed path. The public signature and global optional worker limit are unchanged.

Windows passes 5,020 adaptive/API cases in each SIMD mode; Linux/NumPy 1.24 passes 2,958 adaptive cases in each mode. Native release, portable, ASan/UBSan and dynamic-team checks pass 2/2. Full candidate CI [39778472720](https://github.com/bipxx/scikit-imagexx/actions/runs/39778472720) is still running; its minimum-runtime job passes.

The unchanged 63-case corpus uses 512 by 513 images, uint8/float32/float64, windows 3/15, layouts, borders, offsets and tiny inputs. It runs one/four worker limits with seven randomized paired samples, 15 ms target batches and 20 ms cooldown. Parity runs before timing with fallback disabled and reference thread pools fixed to one. [Exact packages and all raw samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39778503119) are archived, including every adverse result.

| Host | Method | Workers | Original / current | Initial native / current |
|---|---|---:|---:|---:|---:|
| Linux | gaussian | 1 | 2.612x | 1.019x |
| Linux | gaussian | 4 | 3.165x | 1.022x |
| Linux | mean | 1 | 1.388x | 1.251x |
| Linux | mean | 4 | 1.696x | 1.187x |
| Linux | median | 1 | 6.752x | 1.015x |
| Linux | median | 4 | 10.633x | 1.020x |
| Windows | gaussian | 1 | 1.849x | 1.031x |
| Windows | gaussian | 4 | 2.158x | 1.021x |
| Windows | mean | 1 | 1.109x | 1.238x |
| Windows | mean | 4 | 1.393x | 1.157x |
| Windows | median | 1 | 5.853x | 1.011x |
| Windows | median | 4 | 8.766x | 1.005x |

The fixed one-worker mean metric improves on both OSes, so the streaming change is retained. Tiny API calls, Fortran normalization, the remaining packed axes and large-window generic median selection still need optimization; aggregate gains do not establish that every workload beats upstream. Gaussian/median measurements are included as controls, with all regressions retained below.

| Host | Dtype | Layout | Method | Window | Workers | Original ms | Current ms | Original / current | Initial / current |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Linux | uint8 | contiguous | gaussian | 3 | 1 | 1.82691 | 0.59988 | 3.045x | 1.010x |
| Linux | uint8 | contiguous | gaussian | 3 | 4 | 1.79540 | 0.42029 | 4.272x | 1.026x |
| Linux | uint8 | contiguous | gaussian | 15 | 1 | 3.50209 | 1.50050 | 2.334x | 1.013x |
| Linux | uint8 | contiguous | gaussian | 15 | 4 | 3.51076 | 0.89156 | 3.938x | 1.014x |
| Linux | uint8 | fortran | gaussian | 3 | 1 | 1.72859 | 0.84803 | 2.038x | 1.005x |
| Linux | uint8 | fortran | gaussian | 3 | 4 | 1.74520 | 0.70874 | 2.462x | 1.006x |
| Linux | uint8 | strided | gaussian | 3 | 1 | 0.88846 | 0.31844 | 2.790x | 1.009x |
| Linux | uint8 | strided | gaussian | 3 | 4 | 0.87411 | 0.30791 | 2.839x | 1.016x |
| Linux | uint8 | offset | gaussian | 3 | 1 | 1.76983 | 0.63735 | 2.777x | 1.006x |
| Linux | uint8 | offset | gaussian | 3 | 4 | 1.78724 | 0.48994 | 3.648x | 1.025x |
| Linux | uint8 | constant-border | gaussian | 3 | 1 | 1.77548 | 0.57804 | 3.072x | 1.000x |
| Linux | uint8 | constant-border | gaussian | 3 | 4 | 1.81644 | 0.41771 | 4.349x | 1.026x |
| Linux | uint8 | tiny | gaussian | 3 | 1 | 0.02686 | 0.01777 | 1.512x | 1.088x |
| Linux | uint8 | tiny | gaussian | 3 | 4 | 0.02674 | 0.01792 | 1.492x | 1.073x |
| Linux | uint8 | contiguous | mean | 3 | 1 | 1.82383 | 1.06064 | 1.720x | 1.266x |
| Linux | uint8 | contiguous | mean | 3 | 4 | 1.76987 | 0.74940 | 2.362x | 1.178x |
| Linux | uint8 | contiguous | mean | 15 | 1 | 1.75952 | 1.10169 | 1.597x | 1.196x |
| Linux | uint8 | contiguous | mean | 15 | 4 | 1.76293 | 0.76117 | 2.316x | 1.167x |
| Linux | uint8 | fortran | mean | 3 | 1 | 1.72945 | 1.29593 | 1.335x | 1.222x |
| Linux | uint8 | fortran | mean | 3 | 4 | 1.78496 | 1.04748 | 1.704x | 1.117x |
| Linux | uint8 | strided | mean | 3 | 1 | 0.83548 | 0.54323 | 1.538x | 1.243x |
| Linux | uint8 | strided | mean | 3 | 4 | 0.83533 | 0.53610 | 1.558x | 1.239x |
| Linux | uint8 | offset | mean | 3 | 1 | 1.75193 | 1.10975 | 1.579x | 1.232x |
| Linux | uint8 | offset | mean | 3 | 4 | 1.77368 | 0.81072 | 2.188x | 1.174x |
| Linux | uint8 | constant-border | mean | 3 | 1 | 1.75565 | 1.03103 | 1.703x | 1.255x |
| Linux | uint8 | constant-border | mean | 3 | 4 | 1.79974 | 0.75980 | 2.369x | 1.180x |
| Linux | uint8 | tiny | mean | 3 | 1 | 0.00851 | 0.01376 | 0.619x | 1.113x |
| Linux | uint8 | tiny | mean | 3 | 4 | 0.00832 | 0.01422 | 0.586x | 1.053x |
| Linux | uint8 | contiguous | median | 3 | 1 | 32.55242 | 2.33142 | 13.962x | 1.001x |
| Linux | uint8 | contiguous | median | 3 | 4 | 32.60355 | 1.36340 | 23.913x | 1.008x |
| Linux | uint8 | contiguous | median | 15 | 1 | 647.33283 | 574.20950 | 1.127x | 1.005x |
| Linux | uint8 | contiguous | median | 15 | 4 | 647.32128 | 179.35745 | 3.609x | 1.032x |
| Linux | uint8 | fortran | median | 3 | 1 | 34.18217 | 2.60646 | 13.114x | 1.002x |
| Linux | uint8 | fortran | median | 3 | 4 | 35.20290 | 1.69160 | 20.810x | 0.989x |
| Linux | uint8 | strided | median | 3 | 1 | 16.42812 | 1.24765 | 13.167x | 1.001x |
| Linux | uint8 | strided | median | 3 | 4 | 16.43006 | 1.23573 | 13.296x | 1.005x |
| Linux | uint8 | offset | median | 3 | 1 | 32.59605 | 2.39317 | 13.620x | 1.005x |
| Linux | uint8 | offset | median | 3 | 4 | 32.56042 | 1.45853 | 22.324x | 1.003x |
| Linux | uint8 | constant-border | median | 3 | 1 | 32.47410 | 2.29950 | 14.122x | 1.003x |
| Linux | uint8 | constant-border | median | 3 | 4 | 32.56690 | 1.39551 | 23.337x | 0.995x |
| Linux | uint8 | tiny | median | 3 | 1 | 0.01336 | 0.01407 | 0.950x | 1.087x |
| Linux | uint8 | tiny | median | 3 | 4 | 0.01241 | 0.01376 | 0.902x | 1.143x |
| Linux | float32 | contiguous | gaussian | 3 | 1 | 1.33372 | 0.29531 | 4.516x | 1.005x |
| Linux | float32 | contiguous | gaussian | 3 | 4 | 1.35121 | 0.28954 | 4.667x | 0.999x |
| Linux | float32 | contiguous | gaussian | 15 | 1 | 3.02043 | 1.29568 | 2.331x | 1.011x |
| Linux | float32 | contiguous | gaussian | 15 | 4 | 3.03466 | 0.74441 | 4.077x | 1.006x |
| Linux | float32 | fortran | gaussian | 3 | 1 | 1.28638 | 0.62204 | 2.068x | 1.014x |
| Linux | float32 | fortran | gaussian | 3 | 4 | 1.29854 | 0.62601 | 2.074x | 0.996x |
| Linux | float32 | strided | gaussian | 3 | 1 | 0.65173 | 0.22632 | 2.880x | 1.007x |
| Linux | float32 | strided | gaussian | 3 | 4 | 0.65168 | 0.22524 | 2.893x | 1.022x |
| Linux | float32 | offset | gaussian | 3 | 1 | 1.31534 | 0.31147 | 4.223x | 1.013x |
| Linux | float32 | offset | gaussian | 3 | 4 | 1.34225 | 0.34717 | 3.866x | 0.980x |
| Linux | float32 | constant-border | gaussian | 3 | 1 | 1.31427 | 0.28258 | 4.651x | 1.011x |
| Linux | float32 | constant-border | gaussian | 3 | 4 | 1.32850 | 0.27961 | 4.751x | 0.990x |
| Linux | float32 | tiny | gaussian | 3 | 1 | 0.02710 | 0.01623 | 1.670x | 1.091x |
| Linux | float32 | tiny | gaussian | 3 | 4 | 0.02698 | 0.01671 | 1.615x | 1.078x |
| Linux | float32 | contiguous | mean | 3 | 1 | 1.31274 | 0.71492 | 1.836x | 1.379x |
| Linux | float32 | contiguous | mean | 3 | 4 | 1.31787 | 0.56771 | 2.321x | 1.316x |
| Linux | float32 | contiguous | mean | 15 | 1 | 1.31642 | 0.75737 | 1.738x | 1.351x |
| Linux | float32 | contiguous | mean | 15 | 4 | 1.33459 | 0.57916 | 2.304x | 1.237x |
| Linux | float32 | fortran | mean | 3 | 1 | 1.27135 | 1.06825 | 1.190x | 1.235x |
| Linux | float32 | fortran | mean | 3 | 4 | 1.28595 | 0.91795 | 1.401x | 1.170x |
| Linux | float32 | strided | mean | 3 | 1 | 0.62722 | 0.42737 | 1.468x | 1.266x |
| Linux | float32 | strided | mean | 3 | 4 | 0.62754 | 0.42933 | 1.462x | 1.265x |
| Linux | float32 | offset | mean | 3 | 1 | 1.30972 | 0.73685 | 1.777x | 1.364x |
| Linux | float32 | offset | mean | 3 | 4 | 1.33616 | 0.61931 | 2.157x | 1.258x |
| Linux | float32 | constant-border | mean | 3 | 1 | 1.31931 | 0.71361 | 1.849x | 1.412x |
| Linux | float32 | constant-border | mean | 3 | 4 | 1.32578 | 0.56608 | 2.342x | 1.283x |
| Linux | float32 | tiny | mean | 3 | 1 | 0.00819 | 0.01276 | 0.641x | 1.086x |
| Linux | float32 | tiny | mean | 3 | 4 | 0.00805 | 0.01260 | 0.639x | 1.113x |
| Linux | float32 | contiguous | median | 3 | 1 | 32.34202 | 1.95066 | 16.580x | 1.003x |
| Linux | float32 | contiguous | median | 3 | 4 | 32.33513 | 1.12794 | 28.667x | 1.006x |
| Linux | float32 | contiguous | median | 15 | 1 | 646.54557 | 563.44682 | 1.147x | 1.002x |
| Linux | float32 | contiguous | median | 15 | 4 | 647.13722 | 178.31486 | 3.629x | 1.001x |
| Linux | float32 | fortran | median | 3 | 1 | 32.48408 | 2.27412 | 14.284x | 1.005x |
| Linux | float32 | fortran | median | 3 | 4 | 32.51526 | 1.46700 | 22.164x | 1.003x |
| Linux | float32 | strided | median | 3 | 1 | 16.32417 | 1.05740 | 15.438x | 1.010x |
| Linux | float32 | strided | median | 3 | 4 | 16.32414 | 1.06691 | 15.300x | 0.998x |
| Linux | float32 | offset | median | 3 | 1 | 32.26672 | 1.95338 | 16.518x | 1.005x |
| Linux | float32 | offset | median | 3 | 4 | 32.39379 | 1.16875 | 27.717x | 1.004x |
| Linux | float32 | constant-border | median | 3 | 1 | 32.35792 | 1.93161 | 16.752x | 1.006x |
| Linux | float32 | constant-border | median | 3 | 4 | 32.36459 | 1.11738 | 28.965x | 1.005x |
| Linux | float32 | tiny | median | 3 | 1 | 0.01245 | 0.01304 | 0.955x | 1.115x |
| Linux | float32 | tiny | median | 3 | 4 | 0.01217 | 0.01321 | 0.921x | 1.075x |
| Linux | float64 | contiguous | gaussian | 3 | 1 | 1.70228 | 0.50725 | 3.356x | 1.006x |
| Linux | float64 | contiguous | gaussian | 3 | 4 | 1.65737 | 0.33464 | 4.953x | 1.026x |
| Linux | float64 | contiguous | gaussian | 15 | 1 | 3.33172 | 1.39070 | 2.396x | 1.024x |
| Linux | float64 | contiguous | gaussian | 15 | 4 | 3.37423 | 0.79832 | 4.227x | 1.020x |
| Linux | float64 | fortran | gaussian | 3 | 1 | 1.61846 | 1.03685 | 1.561x | 1.006x |
| Linux | float64 | fortran | gaussian | 3 | 4 | 1.67071 | 0.91317 | 1.830x | 1.002x |
| Linux | float64 | strided | gaussian | 3 | 1 | 0.86763 | 0.35048 | 2.476x | 1.007x |
| Linux | float64 | strided | gaussian | 3 | 4 | 0.86466 | 0.34422 | 2.512x | 1.013x |
| Linux | float64 | offset | gaussian | 3 | 1 | 1.72126 | 0.57443 | 2.996x | 1.006x |
| Linux | float64 | offset | gaussian | 3 | 4 | 1.70816 | 0.40793 | 4.187x | 1.054x |
| Linux | float64 | constant-border | gaussian | 3 | 1 | 1.70273 | 0.48932 | 3.480x | 1.001x |
| Linux | float64 | constant-border | gaussian | 3 | 4 | 1.68804 | 0.32677 | 5.166x | 1.016x |
| Linux | float64 | tiny | gaussian | 3 | 1 | 0.02710 | 0.01651 | 1.641x | 1.062x |
| Linux | float64 | tiny | gaussian | 3 | 4 | 0.02691 | 0.01639 | 1.642x | 1.075x |
| Linux | float64 | contiguous | mean | 3 | 1 | 1.64945 | 0.92701 | 1.779x | 1.292x |
| Linux | float64 | contiguous | mean | 3 | 4 | 1.69207 | 0.67530 | 2.506x | 1.194x |
| Linux | float64 | contiguous | mean | 15 | 1 | 1.69445 | 0.97293 | 1.742x | 1.295x |
| Linux | float64 | contiguous | mean | 15 | 4 | 1.68894 | 0.67964 | 2.485x | 1.187x |
| Linux | float64 | fortran | mean | 3 | 1 | 1.62058 | 1.56173 | 1.038x | 1.158x |
| Linux | float64 | fortran | mean | 3 | 4 | 1.66290 | 1.32394 | 1.256x | 1.107x |
| Linux | float64 | strided | mean | 3 | 1 | 0.81197 | 0.56133 | 1.447x | 1.268x |
| Linux | float64 | strided | mean | 3 | 4 | 0.82485 | 0.58025 | 1.422x | 1.221x |
| Linux | float64 | offset | mean | 3 | 1 | 1.69578 | 1.01534 | 1.670x | 1.292x |
| Linux | float64 | offset | mean | 3 | 4 | 1.70179 | 0.72522 | 2.347x | 1.198x |
| Linux | float64 | constant-border | mean | 3 | 1 | 1.69086 | 0.94403 | 1.791x | 1.308x |
| Linux | float64 | constant-border | mean | 3 | 4 | 1.77619 | 0.66626 | 2.666x | 1.223x |
| Linux | float64 | tiny | mean | 3 | 1 | 0.00820 | 0.01286 | 0.637x | 1.103x |
| Linux | float64 | tiny | mean | 3 | 4 | 0.00795 | 0.01312 | 0.606x | 1.078x |
| Linux | float64 | contiguous | median | 3 | 1 | 32.47837 | 2.24331 | 14.478x | 1.001x |
| Linux | float64 | contiguous | median | 3 | 4 | 32.35458 | 1.30833 | 24.730x | 0.994x |
| Linux | float64 | contiguous | median | 15 | 1 | 646.81911 | 573.61555 | 1.128x | 1.005x |
| Linux | float64 | contiguous | median | 15 | 4 | 646.91708 | 179.59342 | 3.602x | 1.029x |
| Linux | float64 | fortran | median | 3 | 1 | 34.31048 | 2.82075 | 12.164x | 1.002x |
| Linux | float64 | fortran | median | 3 | 4 | 34.41633 | 1.90950 | 18.024x | 1.009x |
| Linux | float64 | strided | median | 3 | 1 | 16.34181 | 1.25608 | 13.010x | 0.999x |
| Linux | float64 | strided | median | 3 | 4 | 16.38291 | 1.25333 | 13.071x | 1.003x |
| Linux | float64 | offset | median | 3 | 1 | 32.28618 | 2.28885 | 14.106x | 1.002x |
| Linux | float64 | offset | median | 3 | 4 | 32.38419 | 1.35833 | 23.841x | 1.010x |
| Linux | float64 | constant-border | median | 3 | 1 | 32.51145 | 2.22955 | 14.582x | 1.006x |
| Linux | float64 | constant-border | median | 3 | 4 | 32.46426 | 1.29392 | 25.090x | 1.015x |
| Linux | float64 | tiny | median | 3 | 1 | 0.01221 | 0.01365 | 0.895x | 1.050x |
| Linux | float64 | tiny | median | 3 | 4 | 0.01268 | 0.01284 | 0.988x | 1.116x |
| Windows | uint8 | contiguous | gaussian | 3 | 1 | 2.31770 | 1.20294 | 1.927x | 1.007x |
| Windows | uint8 | contiguous | gaussian | 3 | 4 | 2.28420 | 1.05202 | 2.171x | 0.959x |
| Windows | uint8 | contiguous | gaussian | 15 | 1 | 3.82700 | 2.30862 | 1.658x | 1.040x |
| Windows | uint8 | contiguous | gaussian | 15 | 4 | 3.80853 | 1.41099 | 2.699x | 1.043x |
| Windows | uint8 | fortran | gaussian | 3 | 1 | 2.15258 | 1.38623 | 1.553x | 1.029x |
| Windows | uint8 | fortran | gaussian | 3 | 4 | 2.26328 | 1.15287 | 1.963x | 1.016x |
| Windows | uint8 | strided | gaussian | 3 | 1 | 1.24432 | 0.74804 | 1.663x | 1.042x |
| Windows | uint8 | strided | gaussian | 3 | 4 | 1.20735 | 0.73888 | 1.634x | 1.023x |
| Windows | uint8 | offset | gaussian | 3 | 1 | 2.23470 | 1.24083 | 1.801x | 1.012x |
| Windows | uint8 | offset | gaussian | 3 | 4 | 2.24622 | 0.96593 | 2.325x | 1.003x |
| Windows | uint8 | constant-border | gaussian | 3 | 1 | 2.19467 | 1.19424 | 1.838x | 1.009x |
| Windows | uint8 | constant-border | gaussian | 3 | 4 | 2.15550 | 0.93457 | 2.306x | 1.024x |
| Windows | uint8 | tiny | gaussian | 3 | 1 | 0.02566 | 0.01600 | 1.604x | 1.063x |
| Windows | uint8 | tiny | gaussian | 3 | 4 | 0.02287 | 0.01493 | 1.532x | 1.083x |
| Windows | uint8 | contiguous | mean | 3 | 1 | 2.28712 | 1.65470 | 1.382x | 1.229x |
| Windows | uint8 | contiguous | mean | 3 | 4 | 2.20543 | 1.21180 | 1.820x | 1.099x |
| Windows | uint8 | contiguous | mean | 15 | 1 | 2.20707 | 1.62977 | 1.354x | 1.249x |
| Windows | uint8 | contiguous | mean | 15 | 4 | 2.16660 | 1.24019 | 1.747x | 1.111x |
| Windows | uint8 | fortran | mean | 3 | 1 | 2.24457 | 1.84937 | 1.214x | 1.201x |
| Windows | uint8 | fortran | mean | 3 | 4 | 2.27905 | 1.49556 | 1.524x | 1.080x |
| Windows | uint8 | strided | mean | 3 | 1 | 1.18741 | 0.95408 | 1.245x | 1.198x |
| Windows | uint8 | strided | mean | 3 | 4 | 1.23792 | 0.92297 | 1.341x | 1.206x |
| Windows | uint8 | offset | mean | 3 | 1 | 2.21630 | 1.72826 | 1.282x | 1.200x |
| Windows | uint8 | offset | mean | 3 | 4 | 2.17710 | 1.24673 | 1.746x | 1.102x |
| Windows | uint8 | constant-border | mean | 3 | 1 | 2.11627 | 1.70781 | 1.239x | 1.155x |
| Windows | uint8 | constant-border | mean | 3 | 4 | 2.26708 | 1.18931 | 1.906x | 1.129x |
| Windows | uint8 | tiny | mean | 3 | 1 | 0.00739 | 0.01292 | 0.572x | 1.044x |
| Windows | uint8 | tiny | mean | 3 | 4 | 0.00712 | 0.01205 | 0.590x | 1.105x |
| Windows | uint8 | contiguous | median | 3 | 1 | 25.56280 | 2.29216 | 11.152x | 0.994x |
| Windows | uint8 | contiguous | median | 3 | 4 | 25.72450 | 1.50446 | 17.099x | 1.012x |
| Windows | uint8 | contiguous | median | 15 | 1 | 593.41680 | 658.85000 | 0.901x | 0.986x |
| Windows | uint8 | contiguous | median | 15 | 4 | 584.30340 | 203.92500 | 2.865x | 0.954x |
| Windows | uint8 | fortran | median | 3 | 1 | 26.30810 | 2.55560 | 10.294x | 1.011x |
| Windows | uint8 | fortran | median | 3 | 4 | 26.18970 | 1.80806 | 14.485x | 1.022x |
| Windows | uint8 | strided | median | 3 | 1 | 13.25090 | 1.36401 | 9.715x | 1.003x |
| Windows | uint8 | strided | median | 3 | 4 | 13.42200 | 1.36569 | 9.828x | 1.014x |
| Windows | uint8 | offset | median | 3 | 1 | 26.17410 | 2.39194 | 10.943x | 1.014x |
| Windows | uint8 | offset | median | 3 | 4 | 25.83350 | 1.63391 | 15.811x | 1.004x |
| Windows | uint8 | constant-border | median | 3 | 1 | 25.95750 | 2.38964 | 10.863x | 0.998x |
| Windows | uint8 | constant-border | median | 3 | 4 | 26.04340 | 1.60700 | 16.206x | 0.988x |
| Windows | uint8 | tiny | median | 3 | 1 | 0.01088 | 0.01230 | 0.885x | 1.093x |
| Windows | uint8 | tiny | median | 3 | 4 | 0.01039 | 0.01204 | 0.863x | 1.070x |
| Windows | float32 | contiguous | gaussian | 3 | 1 | 1.34481 | 0.51030 | 2.635x | 0.995x |
| Windows | float32 | contiguous | gaussian | 3 | 4 | 1.37914 | 0.42759 | 3.225x | 1.040x |
| Windows | float32 | contiguous | gaussian | 15 | 1 | 3.05890 | 1.22536 | 2.496x | 1.105x |
| Windows | float32 | contiguous | gaussian | 15 | 4 | 3.06878 | 0.89229 | 3.439x | 1.011x |
| Windows | float32 | fortran | gaussian | 3 | 1 | 1.38970 | 0.95452 | 1.456x | 0.991x |
| Windows | float32 | fortran | gaussian | 3 | 4 | 1.35200 | 0.88179 | 1.533x | 1.006x |
| Windows | float32 | strided | gaussian | 3 | 1 | 0.42421 | 0.36428 | 1.165x | 1.001x |
| Windows | float32 | strided | gaussian | 3 | 4 | 0.43169 | 0.36773 | 1.174x | 1.004x |
| Windows | float32 | offset | gaussian | 3 | 1 | 1.43043 | 0.51521 | 2.776x | 1.011x |
| Windows | float32 | offset | gaussian | 3 | 4 | 1.43592 | 0.47241 | 3.040x | 1.073x |
| Windows | float32 | constant-border | gaussian | 3 | 1 | 1.69370 | 0.55294 | 3.063x | 0.938x |
| Windows | float32 | constant-border | gaussian | 3 | 4 | 1.44731 | 0.45943 | 3.150x | 1.006x |
| Windows | float32 | tiny | gaussian | 3 | 1 | 0.02397 | 0.01474 | 1.626x | 1.067x |
| Windows | float32 | tiny | gaussian | 3 | 4 | 0.02217 | 0.01401 | 1.583x | 1.030x |
| Windows | float32 | contiguous | mean | 3 | 1 | 1.45754 | 1.03008 | 1.415x | 1.376x |
| Windows | float32 | contiguous | mean | 3 | 4 | 1.46759 | 0.73340 | 2.001x | 1.258x |
| Windows | float32 | contiguous | mean | 15 | 1 | 1.50901 | 1.05450 | 1.431x | 1.340x |
| Windows | float32 | contiguous | mean | 15 | 4 | 1.40272 | 0.69767 | 2.011x | 1.234x |
| Windows | float32 | fortran | mean | 3 | 1 | 1.33029 | 1.38410 | 0.961x | 1.282x |
| Windows | float32 | fortran | mean | 3 | 4 | 1.37147 | 1.08359 | 1.266x | 1.162x |
| Windows | float32 | strided | mean | 3 | 1 | 0.47409 | 0.61658 | 0.769x | 1.343x |
| Windows | float32 | strided | mean | 3 | 4 | 0.44033 | 0.69805 | 0.631x | 1.333x |
| Windows | float32 | offset | mean | 3 | 1 | 1.39256 | 1.00817 | 1.381x | 1.403x |
| Windows | float32 | offset | mean | 3 | 4 | 1.39740 | 0.71291 | 1.960x | 1.231x |
| Windows | float32 | constant-border | mean | 3 | 1 | 1.35496 | 0.97502 | 1.390x | 1.400x |
| Windows | float32 | constant-border | mean | 3 | 4 | 1.40306 | 0.66107 | 2.122x | 1.226x |
| Windows | float32 | tiny | mean | 3 | 1 | 0.00660 | 0.01067 | 0.619x | 1.111x |
| Windows | float32 | tiny | mean | 3 | 4 | 0.00720 | 0.01040 | 0.693x | 1.108x |
| Windows | float32 | contiguous | median | 3 | 1 | 25.57390 | 1.51947 | 16.831x | 0.991x |
| Windows | float32 | contiguous | median | 3 | 4 | 25.62060 | 0.94649 | 27.069x | 1.006x |
| Windows | float32 | contiguous | median | 15 | 1 | 592.17810 | 659.36510 | 0.898x | 0.989x |
| Windows | float32 | contiguous | median | 15 | 4 | 581.85160 | 198.66830 | 2.929x | 0.956x |
| Windows | float32 | fortran | median | 3 | 1 | 25.90690 | 1.90770 | 13.580x | 1.004x |
| Windows | float32 | fortran | median | 3 | 4 | 26.37760 | 1.40042 | 18.835x | 1.022x |
| Windows | float32 | strided | median | 3 | 1 | 12.94240 | 0.89972 | 14.385x | 1.000x |
| Windows | float32 | strided | median | 3 | 4 | 12.87690 | 0.90449 | 14.237x | 1.013x |
| Windows | float32 | offset | median | 3 | 1 | 25.14900 | 1.50846 | 16.672x | 1.002x |
| Windows | float32 | offset | median | 3 | 4 | 25.54860 | 0.95492 | 26.755x | 0.996x |
| Windows | float32 | constant-border | median | 3 | 1 | 25.71270 | 1.50920 | 17.037x | 0.987x |
| Windows | float32 | constant-border | median | 3 | 4 | 25.07790 | 0.90191 | 27.805x | 1.003x |
| Windows | float32 | tiny | median | 3 | 1 | 0.00978 | 0.01083 | 0.903x | 1.100x |
| Windows | float32 | tiny | median | 3 | 4 | 0.00976 | 0.01105 | 0.883x | 1.103x |
| Windows | float64 | contiguous | gaussian | 3 | 1 | 1.85432 | 0.79968 | 2.319x | 1.038x |
| Windows | float64 | contiguous | gaussian | 3 | 4 | 1.90287 | 0.65486 | 2.906x | 1.001x |
| Windows | float64 | contiguous | gaussian | 15 | 1 | 3.59808 | 1.91404 | 1.880x | 1.096x |
| Windows | float64 | contiguous | gaussian | 15 | 4 | 3.65223 | 1.14700 | 3.184x | 1.028x |
| Windows | float64 | fortran | gaussian | 3 | 1 | 1.75373 | 1.45130 | 1.208x | 1.041x |
| Windows | float64 | fortran | gaussian | 3 | 4 | 1.71213 | 1.35458 | 1.264x | 0.982x |
| Windows | float64 | strided | gaussian | 3 | 1 | 0.93037 | 0.67465 | 1.379x | 1.063x |
| Windows | float64 | strided | gaussian | 3 | 4 | 0.88934 | 0.65709 | 1.353x | 1.004x |
| Windows | float64 | offset | gaussian | 3 | 1 | 1.81561 | 0.82316 | 2.206x | 1.009x |
| Windows | float64 | offset | gaussian | 3 | 4 | 1.99007 | 0.67929 | 2.930x | 1.021x |
| Windows | float64 | constant-border | gaussian | 3 | 1 | 1.88897 | 0.81431 | 2.320x | 1.031x |
| Windows | float64 | constant-border | gaussian | 3 | 4 | 1.93462 | 0.64743 | 2.988x | 1.012x |
| Windows | float64 | tiny | gaussian | 3 | 1 | 0.02184 | 0.01364 | 1.601x | 1.090x |
| Windows | float64 | tiny | gaussian | 3 | 4 | 0.02171 | 0.01355 | 1.602x | 1.073x |
| Windows | float64 | contiguous | mean | 3 | 1 | 1.74385 | 1.27167 | 1.371x | 1.277x |
| Windows | float64 | contiguous | mean | 3 | 4 | 1.76671 | 0.89177 | 1.981x | 1.137x |
| Windows | float64 | contiguous | mean | 15 | 1 | 1.88547 | 1.31273 | 1.436x | 1.269x |
| Windows | float64 | contiguous | mean | 15 | 4 | 1.89805 | 0.93652 | 2.027x | 1.112x |
| Windows | float64 | fortran | mean | 3 | 1 | 1.82350 | 2.47764 | 0.736x | 1.142x |
| Windows | float64 | fortran | mean | 3 | 4 | 1.84758 | 2.08108 | 0.888x | 1.069x |
| Windows | float64 | strided | mean | 3 | 1 | 1.01763 | 0.93162 | 1.092x | 1.225x |
| Windows | float64 | strided | mean | 3 | 4 | 0.94186 | 0.91294 | 1.032x | 1.202x |
| Windows | float64 | offset | mean | 3 | 1 | 1.89463 | 1.31062 | 1.446x | 1.286x |
| Windows | float64 | offset | mean | 3 | 4 | 1.78912 | 0.92692 | 1.930x | 1.125x |
| Windows | float64 | constant-border | mean | 3 | 1 | 1.85577 | 1.26555 | 1.466x | 1.313x |
| Windows | float64 | constant-border | mean | 3 | 4 | 1.91350 | 0.90376 | 2.117x | 1.162x |
| Windows | float64 | tiny | mean | 3 | 1 | 0.00638 | 0.01111 | 0.574x | 1.050x |
| Windows | float64 | tiny | mean | 3 | 4 | 0.00658 | 0.01103 | 0.596x | 1.144x |
| Windows | float64 | contiguous | median | 3 | 1 | 25.46280 | 1.95771 | 13.006x | 0.999x |
| Windows | float64 | contiguous | median | 3 | 4 | 25.73080 | 1.27530 | 20.176x | 0.971x |
| Windows | float64 | contiguous | median | 15 | 1 | 576.20180 | 654.63260 | 0.880x | 0.987x |
| Windows | float64 | contiguous | median | 15 | 4 | 587.95800 | 211.18600 | 2.784x | 0.901x |
| Windows | float64 | fortran | median | 3 | 1 | 25.58540 | 2.66754 | 9.591x | 0.995x |
| Windows | float64 | fortran | median | 3 | 4 | 25.72130 | 1.90500 | 13.502x | 0.992x |
| Windows | float64 | strided | median | 3 | 1 | 12.59330 | 1.25249 | 10.055x | 0.996x |
| Windows | float64 | strided | median | 3 | 4 | 12.79880 | 1.23834 | 10.335x | 1.011x |
| Windows | float64 | offset | median | 3 | 1 | 24.75990 | 1.98887 | 12.449x | 0.997x |
| Windows | float64 | offset | median | 3 | 4 | 25.18410 | 1.24012 | 20.308x | 1.007x |
| Windows | float64 | constant-border | median | 3 | 1 | 25.21600 | 1.92424 | 13.104x | 1.003x |
| Windows | float64 | constant-border | median | 3 | 4 | 25.28800 | 1.22397 | 20.661x | 1.035x |
| Windows | float64 | tiny | median | 3 | 1 | 0.00990 | 0.01094 | 0.904x | 1.100x |
| Windows | float64 | tiny | median | 3 | 4 | 0.00977 | 0.01099 | 0.888x | 1.044x |
