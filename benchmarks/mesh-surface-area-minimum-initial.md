# Mesh surface area: first minimum-profile Release comparison

All six measured cases are faster than scikit-image 0.26.0: geometric mean **2.228286562×**, range **2.037008052–2.365351524×**, with no losses. Every case records one accepted owned C++ call and zero error under the frozen geometric checker. All 84 raw samples are retained in the [byte-exact JSON](mesh-surface-area-minimum-initial.json).

Each workload is a unit-square grid with `z=.125*(x*x+2*y*y+x*y)` and two triangles per cell. Actual vertices use the stated float32/float64 dtype; faces use native `intp` (`<i8`). Geometry generation and the full `(verts, faces)` check occur before timing.

| Cells/axis | Vertex dtype | Vertices | Faces | Original ms | Current ms | Original/current | Current − original ms | Max error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | float32 | 4,225 | 8,192 | 0.924613545 | 0.390899000 | 2.365351524× | -0.533714545 | 0.0 |
| 64 | float64 | 4,225 | 8,192 | 0.982108600 | 0.457203500 | 2.148077607× | -0.524905100 | 0.0 |
| 128 | float32 | 16,641 | 32,768 | 3.589059667 | 1.546272333 | 2.321104497× | -2.042787333 | 0.0 |
| 128 | float64 | 16,641 | 32,768 | 3.960149000 | 1.790764000 | 2.211429870× | -2.169385000 | 0.0 |
| 256 | float32 | 66,049 | 131,072 | 14.036990000 | 6.091917500 | 2.304198965× | -7.945072500 | 0.0 |
| 256 | float64 | 66,049 | 131,072 | 15.380633000 | 7.550600000 | 2.037008052× | -7.830033000 | 0.0 |

These are allocation-inclusive public calls: NumPy gathers every face entry, prepares the native buffer, and applies the final `/2.0`; C++ computes edges, cross products, norms and the typed reduction. All six outputs are NumPy float64 scalars on the pinned NumPy 1.24.4 profile, including float32 inputs; raw scalar words match the original. Half, integer, complex and exceptional arithmetic retain explicit primitive-provider paths and are outside this timing cohort. Passive native-call callbacks are restored before timing.

The run uses one global worker and seven rounds per implementation on a fresh Linux x86_64 host, Python 3.11, NumPy 1.24.4 and SciPy 1.15.3. The CPU model was not captured. This is the first original/current comparison for these six meshes; it establishes no prior-C++ improvement or cross-host cumulative gain. The last complete profile qualification remains 203 image APIs.

Validation: 18 focused normal/scalar XML pairs total 4,064 passing checks, plus 3,696 color checks, the entire 1,612-check matching file, and 1,405 smoke checks for 236 registered APIs. Separate intake checks passed: four Polygon, four Euler and nine integer-width checks. The bounded private component passed 356 Mesh public checks and 88 shared checks across both pinned profiles and normal/scalar modes; all frozen routes, error bounds, metadata and state checks remained unchanged.

[Release run 40442613742](https://github.com/bipxx/scikit-imagexx/actions/runs/40442613742) measured source [`6d35344089bf3819bbead22ec134da7d22f7473f`](https://github.com/bipxx/scikit-imagexx/commit/6d35344089bf3819bbead22ec134da7d22f7473f). Review authenticated all 1,623 committed source blobs, all 61 artifact members, 14 wheels (13 runtime dependencies), the complete installed package/core/SDK, and identical before/after measurement identities. Only this new six-case timing ran; 16 earlier timing steps were skipped.

Raw JSON SHA-256: `25d8cd730a4dd3be691b997f0f5936f5ff2bf9e5bd890f49bc4bb592a845fc5c`. The sole retained performance artifact is `12706543928.zip` (84,631,278 bytes), SHA-256 `6dc2541b3f8dc9a136d2359de3cdea704b7a3af3e75148a63703bfe1bfac5efb`. Its 85 MB archive remains in ignored local evidence. Review SHA-256: `bb574789639014fdf49de3b8a04fdd8952205bb13e04910888481184ec0104fb`.
