# Symmetric image eigenvalue measurements

The two public entry points are `feature.hessian_matrix_eigvals` and
`feature.structure_tensor_eigenvalues`. Their C++ implementation uses the ordered
two-by-two formula and scaled double Jacobi iteration for larger matrices, with
AVX2 across independent matrices and optional global workers (default one).

The initial measured revision is **9555fda6b10363922c867c258ee17c3def025ce2**.
It is **not retained**: later probes found signed-zero changes in inactive SIMD
lanes and high-dimension LAPACK tie ordering. The raw results remain available
as evidence; these speedups do not establish parity for that revision.

| One-worker result | Linux | Windows |
|---|---:|---:|
| Hessian eigenvalues, geometric mean | 6.534536× | 5.462059× |
| Structure-tensor eigenvalues, geometric mean | 6.584984× | 5.472003× |
| Hessian eigenvalues, slowest case | 1.522426× | 1.394581× |
| Structure-tensor eigenvalues, slowest case | 1.527914× | 1.377323× |
| Faster one-worker cases per public API | 60/60 | 60/60 |

The fixed corpus contains 120 cases per OS: two public APIs, float32/64,
dimensions 1–5, and contiguous, strided, Fortran, tiny, broadcast and diagonal
inputs. Both worker caps 1 and 4 are measured, seven samples per implementation,
including allocation and boundary dispatch. Reference thread pools are limited
to one. The reference is **scikit-image 0.26.0**. Every timed row passes its
numerical and output-layout checks; these checks do not replace the broader
parity suite. Shapes and seed 9171 are fixed in `symmetric_eigen.py`.

The slowest cases use Fortran component fields for the two-by-two formula;
conversion to contiguous native fields contributes extra memory traffic.
Higher-dimension and unusual-input performance beyond this fixed corpus is
not established by these measurements.

Raw reports with all samples, source/binary hashes and adverse cases:
[Linux](symmetric-eigen-initial-ubuntu.json),
[Windows](symmetric-eigen-initial-windows.json).
The isolated [initial run](https://github.com/bipxx/scikit-imagexx/actions/runs/39870890783)
also preserves exact wheels and their source manifests.

Corrected source **33a0e570cd91f6702530ed609ad80a0cb56b069f** adds masked SIMD
rotations, LAPACK sorting and diagonal signed-zero compatibility with its
default 32-column reduction. Other LAPACK tuning remains unverified. Local
tests pass with SIMD enabled and disabled: Windows 936 passed / 4 rank-limit
skips, minimum NumPy 1.24 Linux 896 passed / 44 rank-limit skips. Standalone
C++ tests include dimension 64; installed-library consumers pass on both OSes.

The corrected-source [measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39872350846)
passed on both platforms. Exact source fingerprints, binary hashes and seven
samples per implementation were verified against the preserved packages.

| Corrected one-worker result | Linux | Windows |
|---|---:|---:|
| Hessian eigenvalues, geometric mean | 6.374628× | 5.176761× |
| Structure-tensor eigenvalues, geometric mean | 6.349021× | 4.725646× |
| Hessian eigenvalues, slowest case | 1.675500× | 1.432209× |
| Structure-tensor eigenvalues, slowest case | 1.662400× | 1.425715× |
| Faster one-worker cases per public API | 60/60 | 60/60 |

All 480 timed corrected rows pass parity. Float32 results in this fixed corpus
are exact; maximum float64 absolute error is 7.994e-15 on Linux and 1.066e-14 on
Windows. Raw corrected samples: [Linux](symmetric-eigen-corrected-ubuntu.json),
[Windows](symmetric-eigen-corrected-windows.json).

The intermediate masked source **97a3e4a760f643bd680de6638ba3530a5d34731f**
was also measured and is not retained because of high-dimension signed-zero
parity gaps: [Linux samples](symmetric-eigen-masked-control-ubuntu.json),
[Windows samples](symmetric-eigen-masked-control-windows.json),
[run](https://github.com/bipxx/scikit-imagexx/actions/runs/39871289601).
Each reported speedup compares that revision to upstream. These independent
runs do not establish a paired performance change between native revisions.

Retention and merge remain pending corrected-source
[full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39872352344).
