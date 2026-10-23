# Hessian matrix measurements and optimization trials

The first source **e52dcf55b79431a08156802bd8b33d2c85e223d5** implements
both finite-difference and repeated Gaussian-derivative Hessians in C++17.
It is **not retained on main**: Gaussian derivatives failed the performance gate.

The isolated [initial measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39886134252)
completed against **scikit-image 0.26.0** on Linux and Windows. Exact source,
preserved binary hashes and seven samples per implementation per row were
verified. Timings include allocation, preparation and Python dispatch.

| One-worker geometric-mean speedup | Linux | Windows |
|---|---:|---:|
| Finite differences (120 cases) | 4.245316脳 | 3.683427脳 |
| Gaussian derivatives (120 cases) | 0.798795脳 | 0.662067脳 |

Finite differences are faster in all 120 cases per OS. Gaussian derivatives
are faster in 36/120 Linux and 30/120 Windows cases. Strided arrays are the
slowest Gaussian cases, around0.34脳 Linux and0.30脳 Windows. All 960 timed rows
have zero numerical absolute difference. This corpus does not prove parity
for every input: extended tiny-sigma tests later exposed symmetric-kernel
classification and signed-zero gaps in the initial implementation.

The fixed corpus is12real dtypes 脳2derivative modes 脳10cases: contiguous,
strided, Fortran, tiny, broadcast, zero sigma, anisotropic sigma, 3-D, 4-D, 5-D.
Ordinary images are256脳257, seed8311; worker caps1and4, seven samples,15ms
sample target. Reference thread pools are limited to one. The same corpus
will be retained throughout optimization; no adverse cases are removed.

The first optimization targets exactly-zero outer coefficients and associated
boundary mapping in long Gaussian-derivative kernels. Signed-zero repair and
SciPy's symmetric-before-antisymmetric epsilon classification are tested
explicitly. Revised performance results remain pending. No local timings
were run.

Unoptimized raw samples: [Linux](hessian-unoptimized-linux.json),
[Windows](hessian-unoptimized-windows.json). Exact packages, source archives,
validation logs and failed correction trials are retained in ignored build
checkpoints. The initial [full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39886134106)
is still running and will not establish retention for this failed trial.

## Zero-tail correction

Source **1172ed10fcce19855226cde30f38444f131514b0** removes exactly-zero
outer taps, repairs affected negative-zero outputs with the original ordered
pass, and matches SciPy's symmetric-before-antisymmetric epsilon classification.
Expanded tests also verify tiny-sigma behavior absent from the initial corpus.

The [second isolated run](https://github.com/bipxx/scikit-imagexx/actions/runs/39888217108)
passed on both platforms, with exact source/binary provenance and seven samples
verified. Fixed case-spec hashes and seeds match the first run. All 960 timed
rows retain zero numerical absolute difference.

| One-worker geometric-mean speedup | Linux | Windows |
|---|---:|---:|
| Finite differences | 4.243598× | 3.861404× |
| Gaussian derivatives | 2.163296× | 1.908508× |

Finite differences remain faster in all 120 cases per OS. Gaussian derivatives
are faster in 117/120 Linux and 110/120 Windows cases. The remaining Linux
slowdowns are float32/int8/int16 broadcast inputs (minimum 0.51785×). Windows
has those same slow broadcast cases (minimum 0.40158×) and several zero-sigma
copy cases. No cases were removed.

The next correction reuses exact results for byte-identical lines/inner blocks
and removes temporary gradients when all Gaussian passes are empty. Reuse is
restricted to complete preceding rows owned by the same worker partition.
Independent outputs, signed-zero patterns and worker parity are regression
tested. These revised measurements remain pending; this trial is not yet
retained on main.

Zero-tail raw samples: [Linux](hessian-zero-tail-linux.json),
[Windows](hessian-zero-tail-windows.json).
