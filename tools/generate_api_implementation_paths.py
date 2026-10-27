"""Generate a planning-only route for every pinned public API; no implementation claims.

Run normally to write the two planning artifacts, or --check to compare them without
writing.  This does not import skimage, execute kernels, or alter the live work queue.
Every API is explicitly registered below; an added inventory key fails closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "coverage/api.json"
JSON_PATH = ROOT / "coverage/API_IMPLEMENTATION_PATHS.json"
MD_PATH = ROOT / "coverage/API_IMPLEMENTATION_PATHS.md"
SPECS: dict[str, dict] = {}
FAMILIES: dict[str, dict] = {}

PRIMITIVES = {
    "array": "Typed strided buffers, dtype/range conversion, allocation, overlap and error admission; reuse current bindings and convert.cpp; preserve Python object protocol at the adapter.",
    "pointwise": "Existing scalar/SIMD color, contrast, alpha and distance formulas; prefer direct formulas or bounded LUTs, preserve branch decisions and saturation.",
    "reduce": "Existing typed reductions and integral scans; add stable grouped sums/moments with explicitly ordered or declared-tolerance arithmetic.",
    "histogram": "Existing contrast, threshold and rank histograms; shared dense/sparse counts, CDFs, sorted ranks and stable tie handling.",
    "neighborhood": "Existing Gaussian, sparse/separable convolution, stencil, median and flat morphology kernels; centralize halo, boundary and footprint handling.",
    "components": "Existing connected-component labels, component consumers and array maps; deterministic union-find/label numbering and component statistics.",
    "order_graph": "Stable sorted traversal, priority queues, adjacency and union-find for reconstruction, max trees and graph fronts; serial reference ordering first.",
    "geometry": "Rasterization, point-in-polygon, contour lookup tables and topology; reuse explicit integer recurrences and licensed upstream tables where appropriate.",
    "resample": "Existing spline/resample/resize and projective kernels; reusable coordinate mapping, interpolation, prefilter and clipping.",
    "small_solver": "Small QR/SVD/eigensystem/linear solves with license/profile-reviewed C++ dependency or existing native primitive; do not port all of SciPy.",
    "sparse_solver": "CSR/COO assembly plus license/profile-reviewed sparse direct/iterative C++ solver; freeze convergence, degeneracy and ordering contracts.",
    "fft": "License/profile-reviewed real/complex N-D FFT backend with explicit normalization, padding, cache and one-worker control; backend choice remains open.",
    "distance": "Exact separable Euclidean distance transform and nearest-site indices; spatial search for point sets; explicit tie ordering and anisotropic spacing.",
    "rng": "Pinned Generator/seed/state adapter and draw-order contract; a new C++ generator requires bit-generator/state/distribution parity or a separately declared noncompatible API.",
    "assets": "Pinned asset manifest with resource URL/path, hash, per-asset license, cache/offline/error behavior; image codecs and archive/NPY readers are explicit dependencies.",
    "provider": "Thin Python adapter for optional codecs, estimators, plotting, lazy arrays and object protocols; keep provider availability/errors honest and separate from native arithmetic.",
    "workers": "Existing process-global worker limit, default one; one budget across nested calls; no private pools or provider-thread oversubscription.",
}


def family(key, title, deps, wave, simd, workers, numeric, bench, frontier="Native C++ arithmetic target; Python retains documented array/protocol adaptation. Provider use is separately disclosed."):
    FAMILIES[key] = dict(title=title, dependencies=deps.split(), default_wave=wave,
                         simd=simd, workers=workers, numerical_policy=numeric,
                         benchmark_recipe=bench, native_provider_frontier=frontier)


family("point", "Pointwise formulas and conversions", "array pointwise workers", "W1",
       "Reuse vector lanes for independent pixels; scalar tails and exceptional branches remain equivalent.",
       "Partition disjoint pixel ranges above a measured threshold; default one and shared global budget.",
       "Exact dtype, clipping, integer rounding, branch and warning behavior; predeclare dtype-aware floating tolerances, preserving any existing bitwise contract.",
       "Contiguous and strided uint8/float32/float64 images; tiny/512x512/2048x2048; time public allocation and conversion as well as optional kernel-only diagnostic.")
family("reduce", "Reductions, histograms and moments", "array reduce histogram workers", "W1",
       "Vectorize sample transforms or independent accumulators only under the declared reduction policy.",
       "Thread local counts/partials with fixed merge where allowed; serial for exact ordered reductions.",
       "Exact counts, tie outcomes, dtype and ordering; predeclare error bounds for floating reductions and test cancellation/nonfinite inputs.",
       "Tiny and large arrays, repeated/unique distributions, sparse masks, strided layouts; retain allocation-inclusive timing.")
family("stencil", "Neighborhood and separable filters", "array neighborhood workers", "W2",
       "Reuse SIMD across independent output pixels and existing separable passes; handle borders/tails explicitly.",
       "Disjoint output tiles with read-only halos; avoid parallel writes to overlapping out buffers.",
       "Exact boundaries, footprint origins, dtype/out rules; freeze floating tolerance before comparison and retain strict scopes.",
       "2D/3D images with narrow/wide kernels, border-heavy tiny inputs, masks, channel axes and strided/output-buffer cases.")
family("rank", "Rolling histogram rank operations", "array histogram neighborhood workers", "W2",
       "Reuse existing rolling histogram SIMD/support scans; vectorize independent windows only when feedback admission permits.",
       "Independent scan strips with private histograms; certified feedback/overlap stays serial; global workers default one.",
       "Exact histogram counts, quantile/tie comparisons, integer division, signed-char shifts, masked centers and captured-bin stores; preserve frozen strict proofs.",
       "Reuse frozen rank benchmark cases including large windowed histograms, noise-filter regressions, percentile and bilateral thresholds; add uncovered contracts as new IDs.")
family("component", "Labels and component consumers", "array components histogram workers", "W2",
       "Vectorize predicates/count scans and remapping; union-find ordering remains explicit.",
       "Serial reference labeling first; independent component/statistic passes can partition after stable IDs exist.",
       "Exact labels, background, connectivity, output dtype, integer counts, ties and aliasing.",
       "Sparse/dense/noisy labels, all-background/all-connected, large label IDs, 2D/3D and strided outputs.")
family("footprint", "Footprint construction and decomposition", "array geometry neighborhood", "W1",
       "Vectorize coordinate predicates for large dense footprints; tiny metadata/sequences gain little.",
       "Usually serial; only large independent fills may use the global budget.",
       "Exact footprint cells, dtype, shape, decomposition sequence/order and iteration counts.",
       "Radius/shape sweeps, strict/non-strict boundaries, dense and sequence output; include tiny construction overhead.")
family("draw", "Raster coordinates and geometric masks", "array geometry workers", "W1",
       "Vectorize independent membership/coverage tests; ordered Bresenham/anti-alias recurrences stay scalar.",
       "Serial ordered coordinate emission; parallel independent rows only with stable output concatenation.",
       "Exact coordinate order, endpoint/clip rules and duplicate pixels; declared tolerance for anti-alias values.",
       "Short/long/degenerate primitives, clipped and unclipped output, small and megapixel masks; include coordinate allocation.")
family("geometry", "Contours, topology and shape geometry", "array geometry components small_solver workers", "W3",
       "Vectorize independent cell classification, geometry tests or triangle measures; traversal/topology resolution stays ordered.",
       "Parallel independent cells/components with stable stitching; serial global topology decisions.",
       "Exact topology, connectivity, tie/ordering contracts; predeclare coordinate/solver tolerances separately.",
       "Small degenerate and large structured/noisy 2D/3D geometry, open/closed boundaries and anisotropic spacing.")
family("resample", "Coordinate mapping and image resampling", "array resample neighborhood small_solver workers", "W3",
       "Reuse projective/spline SIMD for independent points/output pixels; scalar exceptional coordinates.",
       "Disjoint output tiles; sequential dependency between pyramid levels, one global worker budget.",
       "Exact shape/coordinate conventions, clipping, dtype, errors and branch choices; interpolation tolerance predeclared and strict historical profiles retained.",
       "Tiny/HD images and volumes, integer/float, contiguous/reversed inputs, affine/nonlinear maps, all supported interpolation/boundary modes.")
family("model", "Geometric model fitting and state", "array small_solver rng workers", "W3",
       "Vectorize batched transforms and residuals; tiny QR/SVD/iteration control need not use SIMD.",
       "Fit/control flow serial; parallel independent residual blocks or independent user-requested fits under global limit.",
       "Exact state transitions, failure objects, parameter conventions, rank/degeneracy decisions; prospective conditioning-aware solve/residual tolerance.",
       "Separate construction/fit/predict/residual calls for minimal, degenerate, ill-conditioned and large point sets; include object/public-call overhead.",
       "C++ small-solver/coordinate kernels plus own Python class state and descriptors; retaining NumPy/SciPy fit arithmetic is a provider frontier, not standalone C++ completion.")
family("tree", "Reconstruction and component trees", "array order_graph components neighborhood workers", "W2",
       "Vectorize admission, min/max and remapping scans; priority/parent propagation is data dependent.",
       "Serial exact queue/tree core first; independent preparation/output or separately proven component partitions only.",
       "Exact plateau/tie order, connectivity, parent/traverser consistency, integer dtype, offsets and seed-mask constraints.",
       "Plateaus, ramps, noisy/nested components, sparse markers, 2D/3D and dtype extremes; include tree construction versus reuse separately.")
family("distance", "Distance transforms and spatial queries", "array distance components workers", "W4",
       "Vectorize distance evaluation and independent line envelopes; nearest-site tie decisions explicit.",
       "Parallel independent transform lines or point-query batches; serial dimension passes and stable reductions.",
       "Exact selected-site/tie and empty-set contracts; spacing-aware floating tolerance predeclared.",
       "Sparse/dense binary masks and point clouds, anisotropic 2D/3D grids, equidistant sites and large distances.")
family("fft", "Fourier filters and spectral restoration", "array fft neighborhood workers", "W4",
       "Use reviewed FFT backend vectorization plus SIMD spectral multiplication; preserve normalization and real/complex conventions.",
       "FFT threads obey global budget; no nested pools; independent channels only when backend is single-threaded.",
       "Exact shapes, padding, phase/normalization and clipping; predeclare complex/real numerical tolerance and separate strict track when required.",
       "Odd/even and prime/composite 2D/3D shapes, real/complex inputs, cold versus warmed FFT plans, allocation-inclusive public calls.")
family("feature", "Feature pipelines and descriptors", "array neighborhood reduce geometry resample workers", "W4",
       "Vectorize gradients, response pixels and descriptor distance blocks; candidate ordering and suppression remain explicit.",
       "Independent scales/keypoints/blocks can partition under one global budget with stable output ordering.",
       "Exact candidate/descriptor shape, mask, ordering, tie and RNG decisions; declared tolerance for response and normalized descriptor floats.",
       "Blank/textured images, edge-heavy images, tiny and megapixel sizes, sparse/dense keypoints; report detect/extract/full pipeline separately.")
family("iterative", "Iterative dense image solvers", "array neighborhood reduce resample workers", "W4",
       "Vectorize independent stencil/update pixels subject to Jacobi/Gauss-Seidel update order; serial convergence decisions.",
       "Tile independent updates with barriers between iterations; serial or coloring only if proven equivalent for dependent updates.",
       "Freeze update order, stopping condition, dtype, callback and warning behavior; predeclare residual/output tolerance and test iteration count separately.",
       "Tiny/HD/3D images, flat/edge/noisy inputs, short/default/long iteration counts; record actual iteration/convergence behavior and total public latency.")
family("graph", "Graph fronts, cuts and sparse operators", "array order_graph components sparse_solver workers", "W5",
       "Vectorize edge weights/CSR preparation and batch residuals; heaps, hooks and mutable traversal are serial first.",
       "Independent edge construction/components/batch shortest paths only; mutable queues and callback event order remain serial.",
       "Exact graph/node/edge identities, mutation/callback order and tie outcomes; predeclare sparse-solver tolerance and convergence/warning policy.",
       "Small disconnected/degenerate and large grid/region graphs, varied degree/weights; separate graph build, solver and public wrapper costs.",
       "C++ graph/solver core; NetworkX/SciPy sparse/plotting object adapters remain explicit providers until independently replaced.")
family("segment", "Segmentation and ordered region algorithms", "array neighborhood components order_graph reduce rng workers", "W5",
       "Vectorize per-pixel features/distances; merges, heaps and topology-changing decisions retain exact order.",
       "Parallel independent preprocessing/local assignments; ordered merges/front propagation serial until proven equivalent.",
       "Exact labels, tie/order/connectivity, seed/RNG consumption and callbacks; declared tolerance for continuous energies without changing label branches silently.",
       "Flat/noisy/region-rich 2D/3D images, sparse/dense markers, anisotropic spacing and masks; include all preprocessing and post-labeling.")
family("random", "Random generators and stochastic helpers", "array rng neighborhood geometry workers", "W2",
       "Vectorize deterministic transforms after fixed-order random draws; do not swap RNG algorithm for speed.",
       "RNG state consumption serial unless a source-equivalent partition is proven; downstream independent image arithmetic may parallelize.",
       "Exact seed/Generator state, draw count/order and distribution algorithm where interchangeability requires it; distribution-only tests cannot prove seeded parity.",
       "Fixed saved RNG states, tiny/large output shapes and all distribution/mode branches; reset equivalent state outside each timed call.")
family("asset", "Bundled/downloaded data assets", "assets array provider", "P",
       "No image-algorithm SIMD claim; codec implementations may have separately reported acceleration.",
       "No native worker claim; controlled fetch/decode concurrency only with cache locking and provider contracts.",
       "Exact bytes/hashes and decoded shape/dtype/value/orientation/container contract; licenses/cache/offline/errors independently verified.",
       "Noncompute boundary: separate warm-cache decode/local file latency from cold authorized fetch; never compare network timing as C++ kernel speed.",
       "Asset manifest/cache and codec adapter first. This preserves API behavior but is not a native image algorithm or standalone C++ codec completion.")
family("io", "I/O and collection adapters", "assets array provider", "P",
       "No wrapper SIMD claim; native copy/parse kernels or codec SIMD measured separately.",
       "Keep stack/cache/event semantics serial; provider codec threads explicitly bounded when available.",
       "Exact filenames/order, frame/dtype/metadata, warnings, errors, lazy loading, mutation and file side effects.",
       "Noncompute boundary: local fixture round trips/parse/collection access; warm/cold-cache timing reported separately, no network in compute benchmarks.",
       "Preserve optional imageio/Pillow/tifffile-style providers behind adapters first; standalone C++ support requires explicit licensed codec/container selection and parity.")
family("ui", "Plotting and GUI adapters", "array geometry provider", "P",
       "No SIMD for GUI/event loop; image preprocessing can reuse native kernels.",
       "GUI state/event callbacks remain on the provider event thread; no parallel UI claim.",
       "Exact return artists/state, coordinate transforms, callback order, cancellation and provider missing/error behavior.",
       "Noncompute boundary: scripted headless artist/state/interaction fixtures; optional render latency separately, no native-kernel speed ratio.",
       "Matplotlib/GUI adapter is the intended compatibility boundary; native preprocessing is counted separately, not as replacement of the GUI provider.")
family("learn", "Optional estimators and learned models", "array reduce small_solver provider rng workers", "P",
       "Native feature normalization/batched distance or responsibility arithmetic can vectorize; arbitrary estimator methods are provider calls.",
       "Adapter respects estimator threading; future native batch prediction/EM uses global budget with deterministic reductions.",
       "Exact estimator identity/state, feature order, fit/predict calls, labels/errors and RNG behavior; tolerance separately specified for optional native solver.",
       "Noncompute adapter boundary initially: instrument estimator calls and payloads; benchmark feature extraction separately. A future C++ solver needs fit/predict timing on identical models.",
       "Keep scikit-learn/user-estimator adapter for public interchangeability. A standalone C++ GMM/EM or classifier backend is a separate explicitly scoped solver, not implied by the adapter.")
family("metadata", "Metadata, state and views", "array provider", "P",
       "No SIMD for type/state/view construction; vectorize only an explicit copy/fill when material.",
       "Serial metadata/state operations; no process-global worker benefit claimed.",
       "Exact identity/sharing, strides, readonly flags, dtype, signatures, inheritance, warnings/errors and state transitions.",
       "Noncompute boundary: descriptor/signature/state/alias fixtures; optional microbenchmarks only when overhead matters, not image-kernel speed claims.",
       "Own metadata or thin object-protocol wrapper is legitimate API behavior; Python/NumPy view/state machinery is not claimed as standalone C++ arithmetic.")


def add(module, family_id, rows, effort="M", wave=None):
    """Rows are name | concrete operation | distinguishing hazards | workload."""
    for line in rows.strip().splitlines():
        if not line.strip():
            continue
        name, algorithm, hazards, workload = (part.strip() for part in line.split("|"))
        api = module + "." + name
        if api in SPECS:
            raise ValueError("duplicate explicit plan: " + api)
        SPECS[api] = dict(family=family_id, algorithm=algorithm, hazards=hazards,
                          workload=workload, effort=effort,
                          wave=wave or FAMILIES[family_id]["default_wave"])


def override(api, **fields):
    SPECS[api].update(fields)


def alias(api, canonical, evidence):
    if api in SPECS:
        raise ValueError("duplicate alias: " + api)
    SPECS[api] = dict(alias=canonical, alias_evidence=evidence)


add("color", "point", """
combine_stains | Reuse native matrix product in stain-log space followed by exponential RGB reconstruction | arbitrary 3xK matrix, channel axis, clipping and promotion | K-stain channel arrays and extreme stain values
convert_colorspace | Extend existing named-space routing through the native RGB hub | case-insensitive names, unsupported-space errors, intermediate dtype/range | all supported source/destination pairs including identity
deltaE_cie76 | Reuse native Euclidean Lab difference with original promotion points | broadcast shapes, float32 intermediates, NaNs and extra channels | broadcast Lab tensors and near-zero differences
deltaE_ciede2000 | Reuse native hue-adjusted CIEDE2000 formula | hue wrap, zero chroma, branch ties, weighting and cancellation | hue-wrap/achromatic pairs plus dense Lab images
deltaE_ciede94 | Reuse native asymmetric CIE94 chroma/lightness weighting | reference/sample asymmetry, negative roundoff under square root, k parameters | swapped color pairs and textile parameter sets
deltaE_cmc | Reuse native CMC lightness/chroma/hue ellipse formula | low-lightness branch, hue sectors, zero chroma, kL/kC | sector-boundary colors and low-lightness pairs
gray2rgb | Reuse native channel replication | arbitrary channel_axis, dtype preservation and copy behavior | scalar-channel images and non-last channel insertion
gray2rgba | Reuse native RGB replication plus alpha broadcast/cast | scalar/array alpha, dtype scaling, broadcast errors | varied alpha shapes and integer/float images
hed2rgb | Compose native combine_stains with fixed HED inverse matrix | fixed matrix constants, channel_axis and clipping | HED basis vectors and mixed stain images
hsv2rgb | Reuse sector-based native HSV conversion | hue boundary/wrap behavior, saturation zero and range violations | six sector boundaries and grayscale HSV
lab2lch | Reuse native hypot/atan2 conversion preserving remaining channels | angle units/wrap, signed zero and extra channels | achromatic and quadrant-boundary Lab
lab2rgb | Compose existing Lab-to-XYZ and XYZ-to-sRGB kernels | illuminant/observer, negative XYZ warning, final clipping | all whitepoints and out-of-gamut Lab
lab2xyz | Reuse inverse piecewise Lab transform and whitepoint lookup | illuminant/observer validation and negative-Z warning contract | threshold-adjacent Lab and whitepoint grid
label2rgb | Reuse palette overlay and grouped-average native paths | bg_label/colors, negative labels, saturation, alpha, kind and channel_axis | sparse/large label IDs and overlay/average modes
lch2lab | Reuse native chroma times sine/cosine | radians, extra channels and dtype preservation | multi-turn hue values and zero chroma
luv2rgb | Compose native Luv-to-XYZ then XYZ-to-sRGB | zero lightness/chromaticity, final clip and channel_axis | black/near-black and out-of-gamut Luv
luv2xyz | Reuse native inverse Luv formula with whitepoint | illuminant/observer, zero denominators and lightness threshold | zero lightness and whitepoint branches
rgb2gray | Reuse native fixed luma dot product | integer scaling, 3-channel validation, dtype and channel_axis | integer/float RGB with non-last channels
rgb2hed | Compose native stain separation with fixed HED matrix | log floor, fixed constants, nonnegative clipping | black/white RGB and stain mixtures
rgb2hsv | Reuse native max/min sector conversion | equal-max ties, zero delta/value and hue wrapping | tied channels, black and six hue sectors
rgb2lab | Compose native sRGB-to-XYZ and XYZ-to-Lab | conversion range, illuminant/observer and promotion | low/high transfer-function branches and whitepoints
rgb2luv | Compose native sRGB-to-XYZ then XYZ-to-Luv | dark pixels and chromaticity denominator | black, primaries and near-black RGB
rgba2rgb | Reuse native alpha blend against background | alpha integer scaling, invalid background, channel axis | opaque/transparent/mixed alpha and broadcast background
separate_stains | Reuse native clamped log transform and matrix product | matrix K, log floor, float dtype and nonnegative output | 3xK matrices and black/negative input
xyz2lab | Reuse native whitepoint normalization and piecewise Lab transform | exact piecewise threshold, illuminant/observer and negative XYZ | threshold-adjacent values for every whitepoint
xyz2luv | Reuse native lightness and chromaticity transform | zero XYZ sum, whitepoint and threshold branch | black, cancelling XYZ and observer combinations
xyz2rgb | Reuse native inverse XYZ matrix and piecewise sRGB transfer | negative values, transfer threshold and final clipping | out-of-gamut and threshold-adjacent XYZ
rgb2xyz | Reuse native sRGB inverse transfer then XYZ matrix | integer normalization and transfer threshold | uint8/float RGB at transfer boundary
""", effort="S")
for space, description in {
    "yuv": "YUV luma/chroma matrix",
    "yiq": "YIQ luma/in-phase/quadrature matrix",
    "ypbpr": "YPbPr luma/color-difference matrix",
    "ycbcr": "YCbCr matrix with encoded offsets/ranges",
    "ydbdr": "YDbDr luma/chroma matrix",
    "rgbcie": "CIE RGB linear color matrix",
}.items():
    add("color", "point", f"rgb2{space} | Reuse native forward {description} | integer normalization, coefficient order, channel_axis and dtype | basis/negative RGB and strided channel layouts\n{space}2rgb | Reuse native inverse {description} | inverse offsets/scaling, out-of-range behavior, channel_axis | encoded basis/offset boundary colors", effort="S")
add("color", "metadata", """
xyz_tristimulus_values | Reuse owned illuminant/observer table and dtype cast | lookup keys, invalid illuminant/observer and returned dtype/copy | all table keys plus invalid keys
""", effort="XS")

ASSETS = {
    "astronaut": ("astronaut.png", "RGB astronaut image"),
    "brain": ("brain.tiff", "multipage brain volume with preserved axes"),
    "brick": ("brick.png", "as_gray conversion"),
    "camera": ("camera.png", "grayscale camera image"),
    "cell": ("cell.png", "grayscale cell image"),
    "cells3d": ("cells3d.tif", "multichannel z-stack axes/dtype"),
    "checkerboard": ("chessboard_GRAY.png", "grayscale checkerboard pixels"),
    "chelsea": ("chelsea.png", "RGB cat image"),
    "clock": ("clock_motion.png", "motion-blurred clock image"),
    "coffee": ("coffee.png", "RGB coffee image"),
    "coins": ("coins.png", "grayscale coins image"),
    "colorwheel": ("color.png", "color wheel channels/alpha contract"),
    "eagle": ("eagle.png", "eagle image decoded dtype/channels"),
    "grass": ("grass.png", "as_gray conversion"),
    "gravel": ("gravel.png", "as_gray conversion"),
    "horse": ("horse.png", "as_gray then img_as_bool conversion"),
    "hubble_deep_field": ("hubble_deep_field.jpg", "RGB JPEG decode contract"),
    "human_mitosis": ("mitosis.tif", "microscopy TIFF axes/dtype"),
    "immunohistochemistry": ("ihc.png", "RGB tissue image"),
    "kidney": ("kidney.tif", "multichannel volume axes/dtype"),
    "lfw_subset": ("lfw_subset.npy", "NPY array dtype/shape and byte order"),
    "lily": ("lily.tif", "multichannel microscopy axes/dtype"),
    "logo": ("logo.png", "logo alpha/channel contract"),
    "microaneurysms": ("microaneurysms.png", "retinal grayscale image"),
    "moon": ("moon.png", "grayscale moon image"),
    "nickel_solidification": ("solidification.tif", "time/volume frame ordering"),
    "page": ("page.png", "grayscale document image"),
    "palisades_of_vogt": ("palisades_of_vogt.tif", "TIFF shape/channel contract"),
    "protein_transport": ("protein_transport.tif", "time/volume frame ordering"),
    "retina": ("retina.jpg", "RGB JPEG decode contract"),
    "rocket": ("rocket.jpg", "RGB JPEG decode contract"),
    "shepp_logan_phantom": ("phantom.png", "as_gray phantom intensity conversion"),
    "skin": ("skin.jpg", "RGB JPEG decode contract"),
    "text": ("text.png", "grayscale texture image"),
    "stereo_motorcycle": ("motorcycle_left.png + motorcycle_right.png + motorcycle_disp.npz", "ordered left/right/disparity tuple including invalid disparity values"),
    "vortex": ("pivchallenge-B-B001_1.tif + pivchallenge-B-B001_2.tif", "ordered pair of frames"),
}
for name, (resource, semantics) in ASSETS.items():
    add("data", "asset", f"{name} | Manifest-backed fetch/decode of data/{resource}; preserve {semantics} | asset hash/license/cache; {semantics} | warm-cache exact fixture and offline/missing/corrupt resource", effort="S")
alias("data.cat", "data.chelsea", "Pinned skimage/data/_fetchers.py assigns cat = chelsea (0.26.0 local source).")
add("data", "asset", """
download_all | Enumerate pinned registry and fetch/cache every declared resource with hash validation | directory override, partial cache, offline failure, license and archive expansion | prepopulated/partial/offline cache fixtures
lbp_frontal_face_cascade_filename | Resolve manifest entry data/lbpcascade_frontalface_opencv.xml and return verified local path | path existence/type, cache/offline behavior and model license | cached XML path and missing-resource fixture
file_hash | Stream file bytes through standard cryptographic hash provider in bounded chunks | algorithm names, missing/unreadable file and exact hexadecimal result | empty/small/large local files for supported algorithms
""", effort="S")
add("data", "random", """
binary_blobs | Preserve upstream random point sampling then Gaussian smoothing and percentile threshold | RNG consumption, volume_fraction ties, n_dim, blob_size_fraction and boundary_mode | seeded 2D/3D fields at extreme volume fractions
""")

add("draw", "draw", """
bezier_curve | Port rational quadratic Bezier integer raster recurrence with weight | control-point/weight degeneracy, output order, clipping | straight/curved/degenerate weighted arcs
circle_perimeter | Port selected Bresenham or Andres integer circle recurrence | method-specific duplicates/order, radius zero and clipping | radius sweep and off-image circles
circle_perimeter_aa | Port weighted anti-aliased circle recurrence | coordinate duplicates/order, intensity weights and radius zero | tiny/large subpixel coverage circles
disk | Generate bounding-box pixels satisfying the radial interior test | center floats, strict radius edge and shape clipping | fractional centers and boundary radii
ellipse | Rotate bounding-box coordinates and test normalized quadratic interior | rotation, unequal radii, degenerate radii and clipping | rotated thin/large ellipses
ellipse_perimeter | Port ordered integer ellipse outline recurrence with orientation | orientation rounding, degeneracy, duplicates and clipping | axis-aligned/rotated outlines
ellipsoid | Evaluate scaled 3D quadratic field on exact spacing-derived grid | grid extents/spacing, boolean versus levelset output | anisotropic ellipsoids and degenerate dimensions
ellipsoid_stats | Direct volume formula plus reviewed elliptic-integral surface-area implementation | sphere/spheroid limits, axis ordering and special-function accuracy | spheres, oblate/prolate and triaxial cases
line | Port Bresenham integer line with exact endpoint order | octants, reversed endpoints, degenerate point | all octants and long diagonals
line_aa | Port anti-aliased line coverage recurrence | octant symmetry, duplicate/order and endpoint intensities | all octants and one-pixel segments
line_nd | Generate evenly spaced N-D coordinates with upstream safe integer rounding | endpoint flag, half ties, integer=False and reversed axes | fractional N-D endpoints and half-integer crossings
polygon | Scan bounding box with exact polygon inclusion/edge rule | self-intersection, concavity, edge inclusion and shape clipping | concave/self-touching/out-of-bounds polygons
polygon2mask | Reuse polygon fill coordinates and scatter into boolean mask | image_shape and polygon coordinate convention | fractional vertices and clipped polygons
polygon_perimeter | Clip polygon when requested then rasterize ordered edges | closure, duplicate vertices, clipping versus shape filtering | open inputs, repeated vertices and clipped outlines
rectangle | Build coordinate mesh between normalized start/end or extent bounds | end inclusivity, negative extent, shape clipping | reversed corners and partially clipped boxes
rectangle_perimeter | Construct expanded border polygon and reuse perimeter rasterizer | corner duplication, clip flag and off-image coordinates | tiny boxes and clipped outlines
set_color | Scatter blend color at supplied coordinates using native alpha arithmetic | in-place mutation, duplicate coordinates/order, alpha broadcasting and dtype | repeated coordinates, scalar/vector alpha and RGB arrays
""", effort="S")
add("draw", "random", """
random_shapes | Preserve seeded shape-choice/placement rejection loop and reuse raster kernels | num_trials, overlap, intensity_range, channel_axis and RNG draw order | seeded mixed-shape crowded and sparse canvases
""", effort="L")

add("exposure", "point", """
adjust_gamma | Reuse native power-law correction and uint8 LUT | nonnegative admission, gamma/gain, cast/round/saturation | uint8 LUT and floating extreme exponents
adjust_log | Reuse native log1p/expm1 correction with input range scaling | inv branch, negative input and integer output conversion | forward/inverse integer and floating ramps
adjust_sigmoid | Reuse native logistic correction or inverse mapping | gain/cutoff, inv branch, overflow and cast | near-cutoff ramps and extreme gain
rescale_intensity | Reuse native min/max selection and affine range remap | dtype/string/range options, constant images, NaNs and clipping | constant/nonfinite and explicit input/output ranges
""", effort="S")
add("exposure", "reduce", """
cumulative_distribution | Reuse histogram then exact-count cumulative normalization | integer versus float bins and empty/range behavior | sparse and dense histograms
equalize_hist | Reuse histogram/CDF interpolation with optional mask | masked CDF but full-image mapping, constant/empty mask and dtype | masked bimodal/constant images
histogram | Reuse typed dense integer and floating-bin histograms | source_range, normalize, channel_axis, bin edges and large integers | all range/channel modes and wide sparse integers
is_low_contrast | Reuse native percentile endpoints and dtype-range ratio | percentile interpolation, channel conversion and constant/nonfinite values | near-decision-threshold and strided images
match_histograms | Reuse sorted-value/count CDF matching and interpolation | unequal shapes, channel axis, repeated values and dtype | unequal-resolution and low/high-cardinality images
""", effort="S")
add("exposure", "stencil", """
equalize_adapthist | Audit existing CLAHE tile histograms/clipping/interpolation and extend uncovered cases | quantization, clip_limit, kernel_size, edge tiles and color conversion | tile-boundary/constant/noisy uint and float images
""", effort="M")

add("feature", "feature", """
BRIEF | Own descriptor class using Gaussian preprocessing and seeded sampled-pair intensity comparisons | mode/rng pair generation, patch border mask, boolean descriptors and extract state | fixed keypoints near borders and seeded normal/uniform pairs
CENSURE | Scale-space box/octagon/star center-surround responses plus extrema/line suppression | scale endpoints, filter normalization, subpixel/order and detect state | multiscale synthetic blobs and blank images
Cascade | Parse licensed LBP cascade XML; integral-image weak classifiers and multiscale window scanning | model format, stage threshold/eps, window ordering, detector state/pickle | pinned XML with empty/dense candidate windows
ORB | Compose image pyramid, FAST, Harris ranking, orientation and rotated BRIEF with own class state | keypoint ordering, scale mapping, border rejection and detect/extract consistency | detect/extract combined and separate on textured/blank images
SIFT | Gaussian/DoG octaves, subpixel extrema, orientation histograms and normalized descriptors | octave/sigma schedule, contrast/edge rejection, descriptor quantization and mutable fields | scale/rotation scenes and low-contrast/border keypoints
blob_dog | Compose Gaussian pyramid differences, N-D local extrema and overlap pruning | sigma_ratio, threshold_rel, border exclusion and pruning tie order | overlapping 2D/3D Gaussian blobs across scales
blob_doh | Use integral-image box Hessian determinant scale responses and overlap pruning | log_scale, approximate response normalization and pruning ties | small/large/overlapping blobs and scale endpoints
blob_log | Gaussian-Laplacian scale response plus N-D extrema and overlap pruning | scale normalization, log_scale, per-axis sigma and border rules | anisotropic 2D/3D blobs and threshold boundaries
canny | Gaussian smoothing, Sobel gradient, nonmaximum suppression and hysteresis connectivity | quantile thresholds, mask erosion, interpolation ties and boundary modes | edge angles, broken/connected edges and masked boundaries
corner_fast | Port FAST circle segment test with early rejection | n segment length, wraparound runs, strict threshold and border zeros | bright/dark corners at exact threshold
corner_moravec | Sliding shifted-patch squared-difference minima | window border, shift set and accumulation dtype | synthetic junctions and constant/strided images
corner_orientations | Intensity centroid moments in each masked keypoint patch | mask oddness, border points, zero moments and angular convention | symmetric/asymmetric patches and border keypoints
corner_peaks | Compose peak candidates with corner-specific connected-plateau suppression | indices=False, labels, num_peaks_per_label, distance metric and stable ties | flat corner plateaus and per-label quotas
corner_subpix | Fit local corner/edge intersection statistics with small linear solves | alpha hypothesis tests, border NaNs, singular patches and coordinate order | ideal/noisy/edge-only patches and degenerate points
daisy | Dense gradient orientation histograms sampled on concentric rings and normalized | custom sigmas/ring radii, normalization, visualize output and channel geometry | dense descriptor grids with every normalization
graycomatrix | Count quantized intensity pairs at rounded distance/angle offsets | levels bounds, angle rounding, symmetric counts, normalize and output dtype | small exact matrices and high-level sparse textures
graycoprops | Reduce GLCM probabilities to named contrast/correlation/energy statistics | zero-variance correlation, normalization and property validation | constant/diagonal/asymmetric cooccurrence matrices
haar_like_feature | Use integral-image rectangle inclusion-exclusion for supplied/enumerated Haar blocks | feature ordering/types, ROI offsets and signed output overflow | all feature families and custom coordinates
haar_like_feature_coord | Enumerate ordered rectangle tuples for each Haar feature family | combinatorial ordering, feature_type selection and empty geometry | tiny exact enumeration and larger windows
hog | Gradient channels, orientation bins, cell histograms and block normalization | orientation bin ties, L2-Hys clipping, channel max ties, visualize and flatten | gradients at bin edges and multichannel cell grids
local_binary_pattern | Bilinearly sample circular neighbors then encode default/ror/uniform/nri_uniform/var | interpolation, float equality, rotation tie mapping and output dtype | constant/rotated patterns and every method
match_descriptors | Blocked pairwise Hamming/Lp distances plus reciprocal and ratio filtering | metric selection, equal-distance tie, zero denominator and empty descriptors | binary/float descriptors with duplicates and outliers
match_template | FFT/direct correlation with integral sum/squared-sum normalization | zero variance, pad_input origin, boundary mode and dimensionality | constant templates, odd/even shapes and boundary peaks
multiblock_lbp | Reuse integral rectangle sums and compare eight blocks with center | bit order, equal sums, ROI boundaries and integer promotion | exact 3x3 block patterns and large integral values
multiscale_basic_features | Compose intensity, gradient magnitude and Hessian eigenvalue features at each Gaussian scale | scale/channel ordering, feature flags, sigma endpoints and workers semantics | channel/scale/feature combinations and non-last axes
peak_local_max | Neighborhood maximum mask, threshold/label filtering and ordered spatial suppression | plateau tie, exclude_border tuples, quotas, p_norm and inf distances | plateaus, per-label masks and close/equidistant peaks
""", effort="L")
add("feature", "stencil", """
corner_foerstner | Reuse structure tensor then determinant/trace and roundness ratios | zero trace, sigma, dtype and two-result contract | flat/isotropic/edge patches
corner_harris | Reuse structure tensor and selected Harris response formula | k versus eps method, negative determinant and promotion | corners/edges and both methods
corner_kitchen_rosenfeld | Reuse native first/second derivatives and curvature response | zero gradient, mode/cval and 2D admission | flat/ramp/corner images and borders
corner_shi_tomasi | Reuse native structure tensor smallest eigenvalue | symmetric eigenvalue order, tiny negative roundoff and sigma | rank-one/zero tensors and corners
hessian_matrix | Reuse native Gaussian-derivative or finite-difference Hessian paths | order rc/xy, warning default, sigma axes and boundary modes | derivative-mode/order combinations in 2D/3D
hessian_matrix_det | Reuse exact determinant or integral-box approximation | approximate branch dimensions, sigma cutoff and dtype | exact/approximate response across sigma ranges
shape_index | Reuse Hessian eigenvalues and native curvature ratio angle | zero curvatures, NaN convention and eigenvalue ordering | planes, saddles and spheres
structure_tensor | Reuse native directional gradients then Gaussian product smoothing | component ordering, sigma, rc/xy and boundary mode | 2D/3D oriented ramps and borders
""", effort="S")
add("feature", "point", """
hessian_matrix_eigvals | Reuse native symmetric eigenvalue kernel over Hessian components | 2D/3D tuple validation, order and repeated eigenvalues | repeated/ill-scaled symmetric tensors
structure_tensor_eigenvalues | Reuse native descending symmetric eigenvalue kernel | component count, tensor shape and repeated roots | isotropic/rank-deficient tensors
""", effort="S")
add("feature", "draw", """
draw_haar_like_feature | Reuse rectangle coordinate fill and alpha overlay for selected Haar blocks | random feature selection state, signed colors, ROI clipping and copy | overlapping positive/negative blocks and seeded subsets
draw_multiblock_lbp | Draw fixed nine-block LBP overlay through rectangle/alpha primitives | bit-to-neighbor orientation, center treatment and dtype/range | each individual bit and mixed block codes
""", effort="S")
add("feature", "learn", """
fisher_vector | Read fitted GMM parameters/responsibilities via adapter; native weighted mean/variance gradients and optional power/L2 normalization | model covariance convention, descriptor shape, alpha and zero normalization | fixed small/large fitted models and empty/degenerate descriptors
learn_gmm | Preserve optional GaussianMixture constructor/fit adapter; standalone alternative is C++ EM with reviewed covariance solvers | gm_args, model identity/state, initialization RNG, convergence warnings and covariance types | stub estimator call contract and fixed-seed fitted-model comparisons
""", effort="L")
add("feature", "ui", """
plot_matched_features | Preserve artist adapter; reuse image composition and keypoint coordinate offsets | alignment, only_matches, colors, axes identity and artist ordering | horizontal/vertical layouts and unmatched points
""", effort="S")

add("filters", "stencil", """
correlate_sparse | Audit native nonzero-kernel tap traversal and direct strided correlation | sparse zero taps, kernel center, N-D boundary modes and dtype | sparse/dense taps on 2D/3D strided arrays
difference_of_gaussians | Reuse two native Gaussian passes and subtraction | per-axis low/high sigma validation, channel_axis, mode and preserve dtype | narrow/wide and anisotropic sigma pairs
gaussian | Extend existing native separable Gaussian derivative-free passes | sigma axes, truncate radius, out aliasing, channel_axis and preserve_range | tiny/large sigma, all modes and out layouts
gabor | Reuse complex Gabor kernel then real/imag convolution | frequency/theta/bandwidth, explicit sigma, mode/cval and output dtype | oriented frequencies and complex response borders
laplace | Reuse native N-D Laplacian taps and mask erosion | ksize, dimensionality, mask border and dtype | impulses, constant images and masked boundaries
median | Audit native neighborhood selection and rank/ndimage behavior routing | behavior, even footprints, out overlap, mode/cval and dtype | odd/even irregular footprints and all supported dtypes
threshold_local | Reuse Gaussian/mean/median local filters; preserve generic callback adapter | method/param, odd block_size, mode and callback order/errors | all methods, anisotropic blocks and callback trace
threshold_niblack | Reuse local first/second moments then mean-minus-k-sigma | window_size, boundary convention, cancellation and dtype | constant/high-offset and varying-window images
threshold_sauvola | Reuse local moments and dtype-derived dynamic range | r default, window_size, constant images and overflow avoidance | explicit/default r and high-offset images
unsharp_mask | Reuse native Gaussian blur then image-plus-amount residual | radius, channel_axis, range/clipping and dtype | all preserve_range and strength branches
""", effort="S")
for name, kernel in {"sobel": "Sobel derivative/smoothing taps", "prewitt": "Prewitt derivative/box taps", "scharr": "Scharr derivative/smoothing taps", "farid": "Farid five-tap derivative/smoothing filters"}.items():
    add("filters", "stencil", f"{name} | Reuse native {kernel}; combine selected-axis squared responses | axis tuple, normalization, mask erosion and mode/cval | directional ramps and axis combinations\n{name}_h | Reuse native {kernel} along the horizontal-edge convention | exact row derivative orientation and 2D mask border | horizontal ramps/edges\n{name}_v | Reuse native {kernel} along the vertical-edge convention | exact column derivative orientation and 2D mask border | vertical ramps/edges", effort="S")
add("filters", "stencil", """
roberts | Reuse both native diagonal Roberts differences and normalized magnitude | 2D mask erosion, normalization and dtype | diagonal edges and masked boundaries
roberts_neg_diag | Reuse native negative-diagonal Roberts 2x2 stencil | tap orientation and mask border | negative diagonal ramps
roberts_pos_diag | Reuse native positive-diagonal Roberts 2x2 stencil | tap orientation and mask border | positive diagonal ramps
frangi | Audit existing native multiscale Hessian vesselness; retain measured precision/regression scopes | eigenvalue sorting/sign, gamma auto choice, scale deprecations and black_ridges | 2D/3D ridge polarity and repeated eigenvalues
hessian | Reuse native hybrid Hessian vesselness and final background handling | nonpositive response replacement, gamma and ridge polarity | flat/background and multiscale vessels
meijering | Reuse native Hessian eigenvalue combination and scale maximum | alpha default by dimension, sign and per-scale normalization | 2D/3D neurites and constant inputs
sato | Reuse native selected Hessian eigenvalue tubeness and scale maximum | eigenvalue sign/order, 2D/3D formula and polarity | bright/dark tubes across sigma
""", effort="M")
add("filters", "point", """
gabor_kernel | Reuse native sampled complex Gaussian-envelope sinusoid | support rounding, theta/sigma, offset and complex dtype | frequency/support extremes and complex64/complex128
""", effort="S")
add("filters", "reduce", """
rank_order | Reuse stable value sort/unique mapping to dense ranks | NaNs, signed zero, dtype preservation and return tuple | low/high-cardinality arrays and float special values
threshold_mean | Reuse native ordered mean reduction | accumulator promotion, empty input and nonfinite behavior | integer extrema and cancelling float arrays
threshold_otsu | Reuse histogram cumulative class weights/means and first maximizing split | hist override, bin centers, degenerate classes and first tie | multimodal/tied/constant histograms
threshold_yen | Reuse histogram cumulative squared-probability criterion | zeros/log domain, histogram validation and tie selection | sparse and symmetric histograms
threshold_triangle | Reuse peak-to-tail histogram distance geometry | tail reversal, zero extent, bin centers and tie | left/right skewed and constant histograms
threshold_isodata | Reuse cumulative class means and accepted midpoint scan | return_all, integer midpoint tolerance and empty split | multiple/zero fixed-point histogram solutions
threshold_minimum | Reuse iterative histogram smoothing and two-peak valley selection | plateau peak detection, max_num_iter error and valley ties | unimodal/noisy/bimodal distributions
threshold_li | Audit existing typed Li iteration; reuse frozen reference and private audit only after review | NaN/Inf admission, initial_guess callback, iteration callback/order and tolerance | captured difficult initialization/iteration cases and histogram shapes
threshold_multiotsu | Reuse native dynamic/multiclass histogram threshold search | class count, insufficient values, tie order and bin quantization | class/bin sweep and sparse histograms
""", effort="S")
add("filters", "component", """
apply_hysteresis_threshold | Reuse native low-mask labels and retain components touching high mask | strict comparison, broadcast low/high, connectivity and nonfinite input | weak bridges, isolated strong pixels and array thresholds
""", effort="S")
add("filters", "fft", """
LPIFilter2D | Own callable state/cache around sampled impulse response and padded FFT convolution | callable evaluation grid, parameter state, cache shape and real result | repeated same/different shapes and custom impulses
filter_forward | Reuse LPIFilter2D transfer function and FFT forward convolution | predefined_filter versus impulse_response, shape crop and normalization | impulse/delta and repeated filter use
filter_inverse | Reuse transfer function with magnitude-limited inverse | zero response, max_gain clipping and parameter precedence | near-zero spectral responses and gain sweep
wiener | Reuse LPI transfer function and conjugate-over-power-plus-K multiplier | K broadcasting/range, predefined filter and real/complex output | weak/zero transfer spectra and K sweep
butterworth | Construct N-D frequency-distance mask then FFT filter | squared_butterworth, high_pass, npad, odd/even grids and channels | low/high pass, order/cutoff sweep and channel axes
""", effort="M")
add("filters", "resample", """
window | Generate named 1D window through reviewed formulas/provider then radially resample to N-D | window_type parameter tuples, center/radius convention and warp_kwargs | named windows and odd/even anisotropic shapes
""", effort="M")
add("filters", "ui", """
try_all_threshold | Run owned threshold functions then preserve plotting/verbosity adapter | failed-method display, axes order, figsize and printed output | constant and multimodal images with headless artist checks
""", effort="S")

add("filters.rank", "rank", """
autolevel | Reuse rolling local min/max then full-range affine stretch | zero span, masked population and output integer cast | constant/two-level windows
autolevel_percentile | Reuse percentile-restricted endpoints then local affine stretch | p0/p1 boundaries, empty selected population and clipping | quantile-boundary windows
enhance_contrast | Reuse closest local min/max selection | equal-distance endpoint tie and masked centers | equidistant center values
enhance_contrast_percentile | Reuse closest selected-percentile endpoint | percentile cut bins and equal-distance ties | outlier-rich percentile windows
entropy | Reuse occupied-bin count entropy reduction | empty population, log base, float output and feedback stores | uniform/singleton/high-entropy windows
equalize | Reuse cumulative local histogram rank of center | inclusive cumulative bound, normalization and integer cast | tied intensities and empty masks
geometric_mean | Reuse log-domain occupied-bin weighted mean and inverse transform | zero bin handling, rounding and empty population | windows containing zero and wide intensity range
gradient | Reuse local occupied-bin max minus min | empty window and unsigned output | high dynamic range windows
gradient_percentile | Reuse percentile endpoint difference | p0/p1 selected-bin inclusivity | tied/boundary quantiles
majority | Reuse native lowest-bin maximal-count selection | equal-frequency tie, empty population and captured output dtype | multiway ties and masked windows
maximum | Reuse highest occupied local histogram bin | empty population, shifts and mask membership | shifted anisotropic footprints
mean | Reuse integer weighted count sum divided by population | signed integer division and output casting | nondivisible sums and masks
mean_bilateral | Reuse strict center-relative intensity-window weighted mean | strict g-s1<i<g+s0, empty set, shifts and feedback | exact bilateral interval endpoints
mean_percentile | Reuse mean over count-selected percentile population | fractional count cutoff and empty interval | repeated bins crossing p0/p1
median | Reuse rolling histogram median selection | even population tie convention and default footprint | even/odd masked populations
minimum | Reuse lowest occupied local histogram bin | empty population, shifts and mask membership | shifted anisotropic footprints
modal | Reuse local mode with source-qualified tie convention | equal maximum frequencies and empty population | tied modes and dtype limits
noise_filter | Reuse nearest occupied-neighbor intensity distance with center-excluded footprint | automatic center exclusion, absent neighbors and output cast | isolated impulses and existing slow sparse windows
otsu | Reuse ascending occupied-bin local Otsu criterion | strict floating winner comparisons, degenerate classes and empty population | flat/tied bimodal windows
percentile | Reuse cumulative count target selection for p0 | endpoint quantiles and discrete count tie | p0=0/1 and exact cumulative boundaries
pop | Reuse local masked population count | output width overflow/cast and empty footprint | huge populations and masks
pop_bilateral | Reuse strict center-relative interval population | s0/s1 conversion/order and interval endpoint exclusion | empty and endpoint-only intensity windows
pop_percentile | Reuse count in percentile-selected population | fractional cumulative cutoffs and p0/p1 order | repeated-bin count boundaries
subtract_mean | Reuse local mean subtraction with upstream midpoint scaling/offset | signed intermediate, integer division and wrap/store behavior | center above/below mean and integer extrema
subtract_mean_percentile | Reuse percentile mean subtraction and upstream scaling/offset | selected-population rounding and output cast | narrow selected populations and extrema
sum | Reuse weighted local histogram sum | overflow/store conversion and masks | high-intensity large windows
sum_bilateral | Reuse strict center-relative interval weighted sum | interval bounds, integer accumulation and out dtype | high-intensity selected bins
sum_percentile | Reuse percentile-selected weighted sum | partial-bin/count cutoff and output conversion | selected bins at exact cumulative fractions
threshold | Reuse center-versus-local-mean binary decision | equality direction, scaling and empty population | center exactly equal to mean
threshold_percentile | Reuse center-versus-selected-percentile binary decision | equality direction, p0 endpoint and output scaling | center exactly equal to percentile
windowed_histogram | Reuse local normalized histogram vector per center | n_bins truncation/inference, trailing axis, zero population and large output cost | small/large bin counts and retained slow large windows
""", effort="S")
for name in "autolevel enhance_contrast entropy equalize geometric_mean gradient majority maximum mean median minimum modal noise_filter otsu pop subtract_mean sum threshold windowed_histogram".split():
    alias("filters.rank.generic." + name, "filters.rank." + name,
          "python/skimagexx/_rank_namespace.py installs the same _rank callable; coverage.py RANK_IDENTITY_ALIASES and tests/contracts cover these explicit names.")
for name in "mean_bilateral pop_bilateral sum_bilateral".split():
    alias("filters.rank.bilateral." + name, "filters.rank." + name,
          "python/skimagexx/_rank_namespace.py installs the same _rank callable; coverage.py RANK_IDENTITY_ALIASES and rank alias contract cover these explicit names.")

add("future", "learn", """
TrainableSegmenter | Own estimator/feature-function state wrapper composing fit_segmenter and predict_segmenter | constructor defaults, fitted state, custom features_func and estimator identity | stub estimator lifecycle and image/feature pipelines
fit_segmenter | Select positive labeled pixels, flatten corresponding feature rows and call supplied estimator.fit | background label convention, sample order, label shape and estimator return | sparse labels and custom estimator call capture
predict_segmenter | Flatten feature rows, call supplied estimator.predict and reshape labels | feature axis, prediction dtype/shape and estimator error propagation | noncontiguous features and custom prediction outputs
""", effort="M")
add("future", "ui", """
manual_lasso_segmentation | Preserve interactive lasso adapter and polygon-mask generation | event ordering, multiple selections, alpha and return_all/cancel | scripted lasso selections and cancellation
manual_polygon_segmentation | Preserve polygon-vertex event adapter and raster masks | vertex closure, deletion/completion events and return_all | scripted concave polygons and cancellation
""", effort="M")

add("graph", "graph", """
MCP | Own Dijkstra-style grid cost front with predecessor storage and public state wrapper | offset ordering, starts/ends, max_cost, tie queue order and traceback state | weighted grids with multiple sources and unreachable targets
MCP_Connect | Extend shared MCP front with ordered meeting-front connection hooks | create_connection arguments/event order and retained path state | colliding multiple-source fronts
MCP_Flexible | Extend shared MCP front with Python travel/update/examine callbacks | callback order, exceptions, goal stopping and mutable subclass state | instrumented custom costs and early termination
MCP_Geometric | Extend MCP using edge length times endpoint cost interpolation | sampling spacing, diagonal lengths and cumulative tie order | anisotropic 2D/3D grids
RAG | Preserve NetworkX-compatible mutable graph adapter; native label adjacency and region statistics | 46 static inherited members, node/edge attrs, views, merges and copy semantics | adjacency/attribute/view mutation fixtures and label graph construction
central_pixel | Run partitioned sparse shortest paths and minimize total distance | disconnected graph, node mapping, partition_size and tie node | disconnected and symmetric grid graphs
cut_normalized | Build normalized graph Laplacian and recursive spectral threshold cuts | rng, eigensolver degeneracy, in_place mutation, num_cuts and label mapping | weakly connected and repeated-eigenvalue RAGs
cut_threshold | Remove threshold-qualified edges then connected-component relabeling | threshold equality, in_place and node-label mapping | threshold-equal edges and disconnected RAGs
merge_hierarchical | Stable priority queue of region edges with merge/weight callbacks | stale entries, rag_copy/in_place_merge and exact callback order | equal-weight merges and instrumented callbacks
pixel_graph | Enumerate neighbor pairs, calculate spacing/intensity edge weights and emit sparse graph | mask/node mapping, edge_function callbacks, sparse matrix/array and connectivity | sparse masks and custom edge functions
rag_boundary | Aggregate region-pair boundary edge values and counts | background/zero labels, connectivity and pair ordering | thin boundaries and triple junctions
rag_mean_color | Reuse native grouped color sums and label adjacency; compute mean-color edge distance/similarity | mode/sigma, channel count and node attributes | sparse labels, constant colors and mode variants
route_through_array | Construct MCP or MCP_Geometric and extract one path/cost | fully_connected, geometric flag, start/end validation and ties | axis/diagonal routes and unreachable endpoints
shortest_path | Dynamic programming along selected axis with bounded reach transitions | reach, axis, negative costs and output_indexlist ordering | narrow/wide reach and degenerate axes
""", effort="L")
add("graph", "ui", """
show_rag | Preserve plotting adapter using native region centroids/edge geometry | in_place graph position attrs, colormaps, axes and artist returns | fixed RAG artist positions and mutation checks
""", effort="M")

add("io", "io", """
ImageCollection | Own ordered filename/loader/cache collection wrapper around optional decode provider | glob/list ordering, conserve_memory, slices, reload and custom load_func | multi-pattern collections with instrumented lazy loader
MultiImage | Reuse collection wrapper with multipage image decoding and filename state | frame indexing, dtype option, conserve_memory and inherited methods | multipage TIFF/GIF-style local fixtures
concatenate_images | Stack collection arrays after compatible-shape validation using native copy path | loading order, mismatched shapes and dtype promotion | homogeneous/mixed-shape collections
imread | Preserve codec/plugin adapter, then own dtype/channel/as_gray postprocessing | plugin deprecation, URI/file objects, alpha/gray conversion and codec errors | PNG/JPEG/TIFF/NPY-supported provider matrix and local file objects
imread_collection | Construct ImageCollection through plugin-aware reader | load_pattern ordering, conserve_memory and deprecated plugin arguments | patterns, explicit lists and lazy load traces
imread_collection_wrapper | Wrap supplied imread callable into compatible collection reader | callable identity, kwargs forwarding and loader exceptions | stub reader call order and collection reload
imsave | Preserve encoder/plugin adapter with native range/contrast checks | check_contrast warning, dtype conversion, plugin kwargs and file side effects | lossless round trips and lossy-codec documented tolerances
load_sift | Parse text SIFT header/keypoints/descriptors into exact structured output | header counts, field order, malformed/truncated text and file objects | tiny hand-authored and large SIFT fixtures
load_surf | Parse SURF text records with exact coordinate/descriptor field layout | descriptor length/header, malformed records and dtype | hand-authored and large SURF fixtures
pop | Own process-local image stack removal with identical empty-stack behavior | LIFO identity, empty error and shared module state | push/pop interleavings
push | Own image stack insertion retaining exact object identity | accepted object types, aliasing and return contract | repeated references and mixed image objects
""", effort="S")

add("measure", "model", """
CircleModel | Own class state with centered algebraic circle least squares and geometric radial residuals | 0.26 constructor/from_estimate failure behavior, collinear points, negative radius and params | minimal/collinear/noisy circles and predict angles
EllipseModel | Normalized constrained ellipse least-squares solve; geometric closest-ellipse residual solve | conic rank/axis ordering, angle convention, degeneracy and solver convergence | near-circles, eccentric/degenerate ellipses and residual points
LineModelND | Center points and use leading singular vector for orthogonal line fit/residual projection | direction sign, rank failure, predict axis zero and dimensionality | collinear/noisy/high-dimensional points
ransac | Preserve seeded sample/control loop with model protocol; accelerate native model residual batches | rng draw order, validators, dynamic trial bound, initial_inliers and tie selection | fixed samples, failed fits, custom model callbacks and early stops
""", effort="L")
add("measure", "metadata", """
RansacModelProtocol | Preserve typing/protocol and factory/residual method contract | runtime protocol behavior, subclass hook and inherited typing members | conforming/nonconforming custom model classes
""", effort="S")
add("measure", "geometry", """
approximate_polygon | Port ordered Douglas-Peucker segment simplification | closed endpoints, tolerance equality, zero-length segments and order | open/closed/collinear/self-touching polylines
euler_number | Use connectivity-specific 2D/3D neighborhood configuration LUT sums | foreground/background dual connectivity, padding and 3D unsupported connectivity | holes/tunnels/cavities and border-touching objects
find_contours | Marching squares with interpolated crossings and stable segment stitching | saddle fully_connected choice, positive_orientation, NaNs/mask and open contours | ambiguous saddles and border/masked level sets
grid_points_in_poly | Reuse exact point-in-polygon evaluator on grid, retaining vertex/edge/interior codes | binarize false classification, coordinate orientation and edge ties | polygons passing through grid vertices/edges
marching_cubes | Port licensed Lewiner tables/topology rules and cell interpolation; offer classic method branch | ambiguous cells, gradient_direction, step_size, degenerates, mask and output ordering | topology fixtures, spheres and anisotropic volumes
mesh_surface_area | Gather triangle edges and reduce half cross-product norms | degenerate/repeated faces, index validity and floating accumulation | tiny/large meshes and zero-area triangles
perimeter | Reuse binary erosion/boundary classification and weighted configuration LUT | neighborhood=4/8, boundary padding and diagonal weights | exact small shapes and diagonal boundaries
perimeter_crofton | Count intercept configurations and apply 2/4-direction Crofton weights | direction option, border padding and binary conversion | rotated shapes and thin objects
points_in_poly | Port ray-crossing/winding-compatible inclusion tests with edge handling | vertex hits, horizontal edges, nonfinite points and self-crossings | points on/inside/outside polygon boundaries
subdivide_polygon | Apply degree-specific B-spline mask convolution to ordered vertices | degree support, closure and preserve_ends behavior | open/closed polygons and short vertex sequences
""", effort="M")
add("measure", "reduce", """
centroid | Compute zeroth/first raw moments then coordinate ratios | spacing, zero mass, signed/nonfinite weights and dimensionality | weighted 2D/3D impulses and zero mass
inertia_tensor | Reuse central second moments to form normalized symmetric inertia tensor | supplied mu, spacing, zero mass and component signs | anisotropic shapes and supplied moments
inertia_tensor_eigvals | Reuse supplied/computed tensor and symmetric eigenvalue solve | descending order, tiny-negative clipping and supplied T/mu | repeated/near-zero eigenvalues
intersection_coeff | Count boolean mask intersection divided by first-image support | optional mask and zero denominator | empty/disjoint/full-overlap masks
manders_coloc_coeff | Sum nonnegative image0 signal inside image1_mask over total signal | negativity, optional mask and zero total | sparse masks and zero/negative signals
manders_overlap_coeff | Normalize cross intensity product by energy product | range/shape validation, mask and zero energy | proportional/orthogonal/zero channels
moments | Reuse separable coordinate-power reductions for raw N-D moments | order, spacing, accumulator precision and dimensionality | high order, spacing and large-coordinate images
moments_central | Reuse raw/coordinate reductions about supplied or inferred centroid | center=None, order, deprecated kwargs and cancellation | large-offset objects and explicit centers
moments_coords | Accumulate coordinate monomials for point sets | coordinate orientation, empty points and order | integer/float point clouds
moments_coords_central | Center point coordinates then accumulate monomials | inferred center, empty points and floating cancellation | translated point clouds and explicit centers
moments_hu | Evaluate seven fixed Hu polynomial invariants from normalized moments | coefficient/order/sign and 2D input contract | reflected/rotated shapes and near-zero invariants
moments_normalized | Divide central moments by mass/spacing scale powers | low-order NaNs, zero mass, order and spacing | scaled shapes and zero mass
pearson_corr_coeff | Reuse stable masked centered sums plus reviewed p-value special function/provider | constant input warnings, sample count, mask and p-value tails | constant/near-perfect/uncorrelated channels
shannon_entropy | Count unique pixel symbols and reduce entropy in requested base | base validation, dtype/NaN equality and empty image | binary/multilevel and continuous-valued images
""", effort="S")
add("measure", "stencil", """
blur_effect | Box-blur each axis, compare sharp/blurred gradient changes and reduce per-axis score | h_size, channel conversion, reduce_func callback and zero gradients | sharp/blurred ramps and anisotropic images
""", effort="M")
add("measure", "reduce", """
block_reduce | Pad to block multiples then use native standard reducers over block axes | arbitrary func callback/provider, cval, dtype and func_kwargs | nondivisible shapes and built-in/custom reductions
""", effort="S")
add("measure", "resample", """
profile_line | Generate perpendicular line samples, reuse interpolation, then width reduction | inclusive endpoint, linewidth, order/mode and reduce_func=None/callback | fractional/reversed/zero-length profiles
""", effort="M")
override("measure.blur_effect",
         algorithm="Add source-ordered reflect uniform_filter1d; reuse float64 sparse correlation with source nonzero Sobel taps and one output cast; retain literal cropped NumPy reductions initially",
         hazards="channel-axis move/color conversion and printed errors; dtype/half behavior; h_size=1 sliding recurrence; positive tiny axes produce empty crop and NaN warnings; NumPy float64 epsilon promotion; final reducer receives the actual scalar list once",
         workload="positive 1D/2D/3D ramps, constants and textures; tiny extents <=3, odd/even/unit filter sizes, RGB channels, layouts and callback/error state; exclude zero spatial axes until source C safety is certified",
         simd="Vectorize independent uniform-filter lines while retaining each double running sum and entering-minus-leaving order; reuse sparse-correlation output-lane SIMD without reordering its source taps",
         workers="Use one global budget across independent lines or correlation output tiles; retain axis-list order and final callback order, with no nested pools or splitting a line recurrence",
         numerical_policy="Freeze exact typed uniform-filter/Sobel intermediates and public scalar/list metadata, words, warnings and callbacks before implementation; retain literal epsilon/clamp/cropped-sum/ratio promotion and explicit exceptional/provider frontiers",
         benchmark="One worker with upstream pool limits; positive 2D/3D uint8/float32/float64 images, several h_size values and edge density; prepare inputs outside the timer and include conversion, all filtering, allocations, crop reductions and final public reducer. Verify actual owned stages first and retain every slower case",
         frontier="Initial new owned work is the ordered moving uniform filter plus assessed sparse Sobel. Color/conversion routes require their own exact admission; irregular cropped sums, epsilon clamps, ratio and user reducer remain explicit NumPy providers. Copying the crop for a different reduction tree does not establish parity.")
override("measure.profile_line",
         algorithm="Keep literal float64 line/perpendicular coordinates; add a 2D order-0/1 typed sampler beside resample, then preserve flip/channel assembly and literal width reduction",
         hazards="inclusive endpoint and ceil length; default bool order; mode remapping; source C-order four-tap accumulation; integer half-away rounding/clipping before reduction; negative-stride None output; TypeError callback retry and mutation",
         workload="fractional/reversed/zero-length profiles, linewidth 0/1/many, RGB channels, raw sampled pixels, strided/endian storage, reducer callbacks and safe Python intake errors; exclude zero spatial axes and unsafe C coordinates from original capture",
         simd="Batch independent query coordinates across lanes; preserve axis-0 then axis-1 multiplication and ordered four-tap accumulation without FMA; reuse boundary/weight/cast helpers, not the separable zoom API",
         workers="Partition independent sampled output coordinates under the single global worker budget; keep coordinate construction, channel assembly and arbitrary reducer callbacks in source order",
         numerical_policy="Exact sampled pixel bytes and public result dtype/words/layout/events for the assessed order-0/1 contract, including integer rounding before reduction; higher-order/provider behavior remains explicit, with no post-observation tolerance fitting",
         benchmark="One worker and matching upstream pool limits; positive 2D/RGB images with uint8/float32/float64, long profiles and widths 1/9/33. Key workload by sampled output count and width; include public coordinate generation, allocation, sampling and reduction in timing, prepare input images outside; verify actual owned sampling before timing and retain every slower case",
         frontier="Initial native sampler covers order 0/1 ordinary uint8/bool/float32/float64 storage. Literal NumPy coordinates and width reducers remain providers; higher spline orders, unsupported storage and nonfinite sampled pixels/cval use disclosed primitive sampling. No source-unsafe empty-spatial reference calls or whole-profile forwarding credit.")
add("measure", "component", """
label | Extend existing deterministic equality-based union-find labels | signed/unsigned dtype, background, connectivity, return_num and numbering | adversarial equal-value components and strided 2D/3D
regionprops | Build own lazy region objects around component bounds/moments/geometry primitives | cache, spacing/offset, intensity channels, property aliases and extra_properties | every property on tiny canonical and irregular regions
regionprops_table | Reuse region objects/property kernels and flatten outputs into columns | separator, scalar/vector/object columns, empty labels and extra_properties | all property shapes and empty/noncontiguous labels
""", effort="L")
override("measure.label", effort="S")

add("metrics", "reduce", """
adapted_rand_error | Reuse sparse contingency counts and squared row/column totals | ignore_labels, supplied table, alpha and zero divisions | exact tiny segmentation partitions and sparse large labels
contingency_table | Stable pair-label counts into CSR sparse matrix/array adapter | ignore_labels, normalization, sparse_type and label index range | sparse high-ID labels and all-ignored pixels
mean_squared_error | Audit existing native dtype promotion and squared-difference reduction | integer promotion, strided/reversed layout and floating reduction order | retained reverse-stride cases and high-dynamic-range differences
normalized_mutual_information | Pad unequal shapes, joint histogram and entropy ratio | bins, zero entropy, shape padding and histogram edges | constant/independent/identical images and mismatched shape
normalized_root_mse | Reuse native MSE and selected norm denominator | euclidean/min-max/mean choices, zero denominator and dtype | constant/zero/shifted images for every normalization
peak_signal_noise_ratio | Reuse native MSE then dtype/range-aware logarithmic score | data_range inference/warnings, zero MSE and invalid range | exact matches and integer/float ranges
variation_of_information | Reuse normalized contingency counts and conditional entropy reductions | image versus table input, ignore_labels, zeros and result order | merge/split partitions with known conditional entropies
""", effort="S")
add("metrics", "distance", """
hausdorff_distance | Extract foreground points and compute bidirectional nearest-neighbor distances | standard versus modified mean, empty sets and max ties | sparse/dense/equidistant 2D/3D masks
hausdorff_pair | Reuse Hausdorff nearest-neighbor search and return attaining point pair | empty-set warning, coordinate order and tie selection | multiple maximizing pairs and empty masks
""", effort="M")
add("metrics", "stencil", """
structural_similarity | Reuse local mean/variance/covariance filters and SSIM formula; derive gradient/full map paths | data_range, sample covariance, channel axis, crop, gaussian weights and return tuple | constant/tiny images, channel layouts and gradient checks
""", effort="M")

add("morphology", "tree", """
area_closing | Reuse reviewed private inverted max-tree candidate; accumulate component area and reconstruct surviving minima | parent/tree_traverser reuse, inversion dtype, connectivity and threshold equality | nested dark plateaus with/without supplied tree
area_opening | Reuse reviewed private max-tree area attribute filter | parent/traverser consistency, equal-area threshold and stable levels | bright plateaus and supplied/recomputed trees
diameter_closing | Reuse private inverted max-tree bounding-box extent filter | closing output dtype, extent definition, connectivity and threshold equality | long thin dark structures and dtype extremes
diameter_opening | Reuse private max-tree component bounding-box extent filter | axis extents, supplied tree and threshold equality | thin bright structures and anisotropic boxes
max_tree | Reuse private deterministic level sorting, union-find parent construction and traversal candidate | plateau canonicalization, parent dtype, connectivity and total order | equal plateaus, NaNs if admitted, 2D/3D trees
max_tree_local_maxima | Reuse private max-tree leaf/plateau maxima labeling | supplied tree, label numbering and equal-level plateaus | multiple equal maxima and tree reuse
reconstruction | Review retained unbuilt draft, then stable queue/rank propagation constrained by mask | seed-mask ordering, dilation/erosion, footprint/offset and integer extremes | plateau/sparse-marker/3D frozen-reference cases
h_maxima | Reuse grayscale reconstruction of intensity-shifted seed then h-difference test | integer saturation, floating resolution correction, h=0 and footprint | maxima at exactly h and dtype limits
h_minima | Reuse erosion reconstruction of upward-shifted seed then h-difference test | integer saturation, floating correction, h=0 and minima threshold | minima at exactly h and dtype limits
local_maxima | Flood equal-valued plateaus and reject those touching larger neighbors | allow_borders, connectivity/footprint, indices and plateau tie | border and interior flat maxima
local_minima | Reuse plateau flood with reversed comparison | allow_borders, connectivity/footprint, indices and dtype extremes | border and interior flat minima
flood | Queue-based connected tolerance predicate from seed using shared offsets | seed bounds, tolerance promotion, connectivity versus footprint and NaN | exact/tolerant 2D/3D connected regions
flood_fill | Reuse flood mask then dtype-correct replacement into copy or original | in_place, new_value casting, overlap and return identity | typed replacements and in-place aliases
""", effort="M")
add("morphology", "footprint", """
ball | Audit native 3D radius predicate and supported sequence decomposition | strict_radius, dtype, approximation contract and sequence order | small exact and large decomposed balls
diamond | Reuse native 2D L1-radius predicate/decomposition | radius type, dtype and iteration counts | radius zero and decomposition equivalence
disk | Audit native 2D radial predicate and source-qualified decomposition | strict_radius, sequence/crosses option and approximate cells | radius sweep at strict boundaries
ellipse | Construct exact discrete ellipse footprint and optional crosses sequence | width/height semantics, dtype and decomposition equivalence | eccentric and zero-width/height cases
footprint_from_sequence | Apply repeated dilations to singleton seed to materialize sequence support | tuple validation, dimensionality, repetitions and origin/shape | asymmetric/even footprints and repeated sequence steps
footprint_rectangle | Reuse native all-ones N-D shape and sequence/separable decomposition | zero/negative shape, dtype, decomposition warnings and order | N-D odd/even shapes and decomposition variants
mirror_footprint | Reuse owned axis reversal for array or each footprint in sequence | view sharing, sequence iteration count and arbitrary rank | asymmetric footprints and sequence views
octagon | Build octagonal binary support and optional rectangle/diamond decomposition | m/n edge lengths, degenerate shapes and decomposition order | flat/degenerate/large octagons
octahedron | Generate 3D L1 ball and optional repeated small-footprint decomposition | radius/dtype and exact sequence support | radius zero and volume footprints
pad_footprint | Reuse native padding to odd shape with upstream end/start convention | pad_end, odd axes unchanged, dtype and sequence handling | mixed even/odd N-D footprints
star | Union axis-aligned and rotated-square supports using exact source shape convention | a=0/small sizes, parity and boundary cells | small exhaustive star footprints
""", effort="S")
add("morphology", "stencil", """
binary_closing | Reuse native binary dilation then erosion with binary border policy | deprecated API warnings, mode ignore/min/max, even footprint origin and out alias | border-touching holes and asymmetric footprints
binary_dilation | Reuse native boolean max stencil with binary footprint orientation | reflected versus gray footprint, mode and output dtype | asymmetric even footprints and border objects
binary_erosion | Reuse native boolean min stencil and neutral outside value | mode, even footprint anchor and out aliasing | all-true images and border erosion
binary_opening | Reuse native erosion then dilation using binary policy | sequence order, even footprints, warning and out buffers | thin components and border cases
black_tophat | Reuse native grayscale closing minus source | subtraction dtype/overflow, out overlap and mode/cval | dark small structures and integer extremes
closing | Audit native dilation-erosion composition | footprint reflection/padding, mode/cval, dtype and out alias | even/asymmetric footprints and retained dtype cases
dilation | Reuse native typed neighborhood maximum | NaN ordering, footprint sequences, reflected origin and modes | NaNs and asymmetric/even footprints
erosion | Reuse native typed neighborhood minimum | NaN ordering, cval/mode, footprint origin and out overlap | signed extremes and boundary modes
opening | Reuse native erosion-dilation composition | even footprint shift, sequence order, dtype and out | bright small structures and asymmetric footprints
white_tophat | Reuse source minus native opening | unsigned/signed subtraction, out alias and boundary behavior | bright structures and dtype extremes
""", effort="S")
add("morphology", "component", """
remove_small_holes | Extend native complement-label/count/fill consumer | boolean versus label input, max_size/deprecated area_threshold, connectivity and out | nested holes, threshold equality and warning paths
remove_small_objects | Extend native label/count/drop consumer | bool versus integer labels, max_size/deprecated min_size, single-label warning and out | sparse huge IDs and exact-size components
""", effort="S")
alias("morphology.label", "measure.label", "Both module sources import ._components.label; tests/test_components_public.py and test_connected_components_contracts.py assert identity.")
add("morphology", "geometry", """
convex_hull_image | Collect occupied boundary/offset points, compute convex hull and classify grid support | offset_coordinates, include_borders, tolerance, empty input and N-D hull dependency | concave/border/degenerate shapes and 2D/3D hulls
convex_hull_object | Label objects then reuse per-component convex_hull_image and combine masks | connectivity, component overlap and accepted dimensionality | touching/disconnected concave components
skeletonize | Port source LUT-based Zhang 2D and Lee 2D/3D thinning with exact deletion schedule | method/dimension defaults, topology, iteration order and boolean output | loops/branches/thin bridges and 3D topology
thin | Port two-subiteration Guo-Hall LUT thinning | max_num_iter, convergence, deletion order and border padding | loops, isolated pixels and truncated iterations
""", effort="L")
add("morphology", "distance", """
isotropic_closing | Compose exact-distance binary dilation then erosion | radius inequality, spacing, border convention and out | large-radius masks and anisotropic spacing
isotropic_dilation | Threshold distance transform of complement | radius equality, all-true/all-false input and spacing | equidistant boundary pixels and large radii
isotropic_erosion | Threshold foreground distance transform | strict versus nonstrict radius, outside-image convention and out | thin features and all-true images
isotropic_opening | Compose exact-distance erosion then dilation | intermediate border effects, radius and out | sparse/small objects and large radii
medial_axis | Distance transform then distance/cornerness/seeded-tie ordered topology thinning | RNG state, masked pixels, return_distance and tie order | symmetric shapes, holes and seeded ties
remove_objects_by_distance | Build object boundary points/spatial index and greedily remove by priority | priority ties, p_norm, spacing, min_distance and out dtype | equidistant neighboring objects and custom priority
""", effort="L")

add("registration", "iterative", """
optical_flow_ilk | Gaussian pyramid, repeated warp, local gradient tensor sums and small per-pixel Lucas-Kanade solves | radius/gaussian weighting, prefilter, dtype, singular tensor and warp count | translated textures, flat fields and 2D/3D motions
optical_flow_tvl1 | Coarse-to-fine warping with TV-L1 primal/dual updates | attachment/tightness, median prefilter, stopping tolerance and dtype | discontinuous flows and convergence edge cases
""", effort="L")
add("registration", "fft", """
phase_cross_correlation | FFT cross-power peak then localized upsampled DFT; separate masked normalized-correlation branch | normalization, space=fourier, overlap_ratio, disambiguation, peak ties and error outputs | integer/subpixel shifts, periodic ambiguity and masks
""", effort="L")

add("restoration", "point", """
ball_kernel | Construct spherical rolling-ball intensity kernel with outside-support infinities | radius/ndim, grid extents and center intensity | small/large N-D kernels
ellipsoid_kernel | Evaluate scaled ellipsoid cap on requested grid and intensity | shape parity, intensity scaling and outside-support infinities | anisotropic odd/even ellipsoids
""", effort="S")
add("restoration", "stencil", """
denoise_bilateral | Spatial Gaussian times intensity-distance LUT, normalize local weighted sums | bins quantization, sigma_color default, channels, mode and zero weights | constant/edge/color images and wide windows
denoise_nl_means | Direct patch-distance baseline then fast summed-distance shifts with Gaussian correction | fast_mode difference, sigma correction, patch/search borders and channels | repeated textures and patch/search size sweeps
rolling_ball | Min-plus neighborhood erosion using nonflat ball/ellipsoid height kernel | nansafe, kernel origin, radius, workers/num_threads warning and dtype | NaNs, bright background and wide kernels
""", effort="L")
add("restoration", "iterative", """
denoise_tv_bregman | Audit accepted split-Bregman native updates and existing cursor optimization | isotropic/anisotropic shrinkage, Gauss-Seidel order, eps, channels and dtype | frozen 70-case suite plus uncovered options
denoise_tv_chambolle | Audit accepted primal/dual divergence updates and merged single-admission optimization | energy stopping order, strided channels, weight/eps and provider calls | frozen one-worker cases with all slower observations retained
inpaint_biharmonic | Assemble biharmonic finite-difference sparse equations on mask and solve by region | boundary stencil, split_into_regions, empty/full mask, channels and solver failure | small exact holes and large/disconnected missing regions
richardson_lucy | Iterated PSF convolution, ratio and reversed-PSF correction with reusable direct/FFT paths | filter_epsilon, zero denominators, clip and PSF orientation | delta/asymmetric PSFs and short/long iterations
""", effort="L")
add("restoration", "fft", """
wiener | FFT PSF/regularizer and apply conjugate transfer over power-plus-balance regularization | is_real, supplied Fourier PSF/reg, normalization and clip | odd/even PSFs, complex mode and balance extremes
unsupervised_wiener | Reuse spectral Wiener conditional sampling with noise/prior precision chain | RNG draw sequence, user_params callbacks/burnin/threshold and returned chains | saved-state short chains and posterior convergence fixtures
""", effort="L")
add("restoration", "feature", """
denoise_wavelet | Reviewed wavelet filter-bank backend, threshold details, then inverse transform | wavelet/mode/levels, BayesShrink/VisuShrink, sigma scaling and YCbCr channel handling | multiple wavelets, odd sizes and color paths
estimate_sigma | Wavelet high-frequency detail coefficients and median absolute-deviation estimate | supported transform/backend, zero detail, channel averaging and dtype | known synthetic noise and constant/odd-shape inputs
""", effort="L")
add("restoration", "tree", """
unwrap_phase | Reliability-guided sorted-edge region merging with wrap offsets | 1D/2D/3D branches, masked arrays, wrap_around and RNG tie sequence | phase jumps, residues, masks and periodic boundaries
""", effort="L")
add("restoration", "metadata", """
calibrate_denoiser | Enumerate parameter grid and score J-invariant masked loss, returning configured callable | denoise_function identity/callbacks, approximate_loss, stride, extra_output and parameter order | instrumented denoiser with deterministic parameter grid
cycle_spin | Enumerate circular shifts, invoke supplied func, unshift and average | shift_steps/channel_axis, deprecated num_workers, callbacks and sum order | identity/custom denoisers and exhaustive shift grids
denoise_invariant | Build phase masks, interpolate held-out pixels, call supplied denoiser and combine predictions | masks generator, stride, denoiser_kwargs and callback/error order | fixed masks and instrumented identity/linear denoisers
""", effort="M", wave="W4")

add("segmentation", "iterative", """
active_contour | Sample native image/edge forces along snake; solve banded/cyclic internal-energy system then capped updates | boundary_condition, convergence history, endpoint constraints and interpolation | periodic/fixed/free snakes and near-convergence cases
chan_vese | Evolve level-set curvature and region-mean data force with source update schedule | init_level_set, energy history, stopping rule, dt and extended_output | constant/two-phase images and max-iteration boundaries
""", effort="L")
add("segmentation", "point", """
checkerboard_level_set | Fill alternating coordinate-block parity as binary level set | square_size and N-D shape | small exact parity grids and large volumes
disk_level_set | Evaluate center/radius radial inclusion on image grid | default center/radius, dimension and strict boundary | even/odd shapes and fractional centers
""", effort="S")
add("segmentation", "component", """
clear_border | Extend native component IDs touching border/mask and relabel background | buffer_size, mask precedence, bgval, source label dtype and out | border/mask-touching labels with aliases
find_boundaries | Compare neighbor labels and apply thick/inner/outer/subpixel construction | connectivity, background and doubled subpixel shape | triple junctions and thin labels
join_segmentations | Pair-label encode then stable sequential relabel with inverse mappings | dtype overflow, mapping objects, return_mapping and zero labels | sparse huge label pairs and mapping round trips
relabel_sequential | Reuse native sorted positive-ID compression and forward/inverse ArrayMap providers | offset, dtype growth, negative rejection and mapping provider state | huge sparse labels, zero and offset boundaries
""", effort="S")
add("segmentation", "distance", """
expand_labels | Reuse exact Euclidean nearest-label indices and threshold distance | equidistant-label tie, spacing, background and output label dtype | competing labels and anisotropic 2D/3D volumes
""", effort="M")
add("segmentation", "segment", """
felzenszwalb | Gaussian preprocessing, sorted neighbor edges and size/internal-difference union-find merges | edge-order ties, scale/min_size postpass and channel_axis | flat/equal-weight images and fine/coarse partitions
morphological_chan_vese | Alternating binary curvature morphology and region-energy threshold updates | initializer, smoothing cycle state, lambda weights and iter_callback | exact iteration traces on binary shape fixtures
morphological_geodesic_active_contour | Binary balloon/attachment updates plus alternating morphological smoothing | threshold=auto, balloon sign, initializer and callback order | weak-edge gaps and balloon pressure signs
quickshift | Kernel density in color/spatial space then nearest higher-density parent tree | rng jitter/ties, Lab conversion, max_dist, return_tree and channels | constant colors, density ties and tree outputs
random_walker | Build weighted grid Laplacian, partition seeds/unlabeled nodes and solve probabilities | inactive labels, beta/spacing, cg modes/preconditioners, prob_tol and full_prob | disconnected seeds, high contrast and convergence warnings
slic | Lab/spatial grid seeds, local k-means assignment/update and connectivity cleanup | mask seeding, spacing, slic_zero, start_label and cluster tie order | masked/anisotropic 2D/3D and low-contrast images
watershed | Stable priority flood from markers with optional compactness/line rules | entry-time ties, connectivity/offset, mask and marker generation | plateaus, touching fronts and compactness variants
""", effort="L")
add("segmentation", "stencil", """
inverse_gaussian_gradient | Reuse Gaussian gradient magnitude and inverse square-root edge-stop formula | alpha, sigma, dtype and boundaries | smooth/step edges and constant images
""", effort="S")
add("segmentation", "draw", """
mark_boundaries | Reuse find_boundaries and color overlay with optional outline | mode, background_label, subpixel upsampling, image dtype and color | inner/outer/subpixel boundaries and grayscale/RGB images
""", effort="S")
alias("segmentation.flood", "morphology.flood", "Pinned segmentation/_watershed.py imports morphology.flood; segmentation/__init__.pyi reexports that object.")
alias("segmentation.flood_fill", "morphology.flood_fill", "Pinned segmentation/_watershed.py imports morphology.flood_fill; segmentation/__init__.pyi reexports that object.")

add("transform", "model", """
AffineTransform | Audit owned partial class/native coordinate path; extend affine parameter decomposition and normalized least-squares fit | scale/shear/rotation conventions, dimension, inverse state, weights and failure descriptors | 2D/3D affine fits, singular matrices and class lifecycle
EssentialMatrixTransform | Reuse normalized eight-point solver then enforce essential singular values; own epipolar class state | rotation/translation validation, rank failure, epipolar residual and inverse | known camera poses and degenerate correspondences
EuclideanTransform | Audit owned partial class; rigid Procrustes/SVD fit with determinant correction and native coordinate calls | rotation representations, reflection handling, dimension and failure state | 2D/3D rigid fits and reflected/degenerate point sets
FundamentalMatrixTransform | Normalized eight-point SVD fit with rank-two projection and Sampson residuals | minimal points, normalization degeneracy, matrix scale and failure state | exact/noisy epipolar pairs and rank-deficient data
PiecewiseAffineTransform | Triangulate source/destination points then fit per-simplex affine maps and point location | Delaunay backend, hull-outside sentinel, simplex ties and inverse triangulation | boundary/outside points and degenerate triangulations
PolynomialTransform | Build polynomial design matrix and weighted normalized least-squares coefficients | monomial order, weights, rank deficiency and inverse support/error | order sweep and ill-conditioned coordinates
ProjectiveTransform | Audit owned partial class/native homography evaluation; extend normalized DLT/SVD fit and state | homogeneous zero denominator, dimension, composition order and failure object | singular/weighted fits and large batched point calls
SimilarityTransform | Audit owned partial class; centered Umeyama SVD scale/rotation/translation fit | reflection/rank handling, dimensionality and parameter decomposition | scaled/reflected/degenerate point sets
ThinPlateSplineTransform | Build radial-basis plus affine saddle-point system, solve weights and evaluate batched kernel | duplicate/collinear controls, singular solve, zero radius and inverse availability | minimal/degenerate controls and dense warp queries
estimate_transform | Route type string to owned transform class.from_estimate preserving failure contract | recognized aliases/type errors, kwargs, weights and returned class | every registered transform type and failed fits
""", effort="L")
add("transform", "resample", """
matrix_transform | Reuse existing private native homogeneous coordinate candidate and public wrapper after whole-product acceptance | dimensions, denominator zeros, dtype promotion and authenticated numerical profile | frozen matrix profiles including ill-conditioned coordinates
pyramid_expand | Reuse resize-up and Gaussian smoothing with exact scale-derived sigma | channel_axis, rounding, mode and preserve_range | odd/even sizes and per-axis channels
pyramid_gaussian | Lazy generator chaining native pyramid_reduce until shape/max_layer termination | yield count/order, original first level, channel_axis and lazy exceptions | tiny/odd shapes and finite/infinite max_layer
pyramid_laplacian | Lazy levels of image minus smoothed image then reduction | first/last level, residual dtype, sigma and lazy state | reconstruction-oriented pyramids and odd shapes
pyramid_reduce | Gaussian smooth then native resize-down at source-defined output shape | downscale validation, sigma default, shape rounding and channels | noninteger downscale and anisotropic inputs
rescale | Audit native scale-to-output-shape wrapper then resize core | channel_axis, scalar/vector scale, anti_aliasing and rounding | small/large/noninteger scale and channel axes
resize | Extend existing native spline/prefilter/antialias/clip implementation | order/bool restrictions, mode, range, output_shape, overlap and numerical profiles | frozen cases plus uncovered modes/orders/shapes
resize_local_mean | Exact area-overlap weights per axis followed by separable averaging | grid_mode pixel centers/extents, preserve_range and channel_axis | arbitrary rational scale ratios and integer-factor equivalence
rotate | Build centered inverse affine rotation and optional expanded bounds then reuse warp | default center, resize corner rounding, angle special cases and clipping | right angles, arbitrary angles and odd/even centers
swirl | Generate radial exponential swirl inverse coordinates then reuse warp | center, radius/strength, output_shape and coordinate order | identity/strong swirl and boundary center
warp | Dispatch matrix/owned transform or callable map; reuse N-D coordinate interpolation | inverse_map forms, map_args callbacks, channel handling, order/prefilter and range clip | matrix/callable/coordinate-array maps with modes and layouts
warp_coords | Generate ordered output grid and call coord_map with upstream reshape convention | shape 2D/3D meaning, dtype, callback order and returned shape | identity/nonlinear callback traces
warp_polar | Generate linear/log radial polar inverse map then reuse warp | center/radius defaults, angular endpoint, log origin and channel_axis | radial/angular patterns and all scaling modes
""", effort="M")
add("transform", "reduce", """
downscale_local_mean | Audit native padded block mean reduction | factors, cval, clip, dtype/range and output shape | nondivisible shapes and uint/float/strided blocks
integral_image | Audit native ordered N-D prefix scans with modular integer arithmetic | dtype choice, axis pass order, overflow and physical layout | cancellation and integer wrap on 2D/3D layouts
integrate | Reuse native integral-corner inclusion-exclusion queries | start/end normalization, negative indices, batched windows and corner order | tiny/large query batches and reverse layouts
""", effort="S")
add("transform", "geometry", """
frt2 | Implement finite Radon integer line sums on prime-sized square grid | supported size, modular line indexing and unsigned overflow | tiny exact prime grids and integer limits
ifrt2 | Invert finite Radon modular line sums with exact normalization | prime-size assumptions, division and signed/unsigned dtype | exact forward/inverse integer fixtures
hough_circle | Vote precomputed discrete perimeter offsets for each radius | normalize, full_output padding, radius cast and duplicate perimeter votes | exact single circles and dense edge masks
hough_circle_peaks | Peak-suppress each circle accumulator then stable cross-radius selection | normalize scoring, per-radius/total limits, x/y suppression ties | competing radii and equal peaks
hough_ellipse | Port triplet edge-point ellipse hypothesis/voting algorithm | accuracy bins, threshold, min/max size, output field order and ties | synthetic ellipses and dense edge clutter
hough_line | Vectorized rho computation per theta with exact integer accumulator votes | theta order/range, rho rounding, accumulator shape and overflow | horizontal/diagonal lines at bin boundaries
hough_line_peaks | Separable accumulator maximum suppression with angle wrap logic | min_distance/min_angle, threshold default, tie order and peak count | equal/symmetric peaks near angle boundaries
probabilistic_hough_line | Preserve seeded edge visitation, line votes, extension and pixel removal | RNG draw order, line_gap/length, threshold and endpoint order | crossing lines, gaps and fixed saved RNG state
radon | Rotate/sample image for each angle then sum projection rays | circle crop/pad/warning, center convention, preserve_range and angle order | delta/phantom projections at cardinal/arbitrary angles
iradon | FFT-filter projections then interpolate/backproject and scale | filter_name, circle masking, interpolation, output_size and angular normalization | analytic phantom and no-filter reconstructions
iradon_sart | Sequential projection updates with ray interpolation and relaxation | golden-angle order, projection_shifts, image alias, clip and dtype | one/multiple sweeps and shifted rays
order_angles_golden_ratio | Ordered nearest-angle selection around golden-ratio angular targets | ties, duplicates, angle periodicity and generator semantics | repeated/unsorted angle lists
""", effort="L")

add("util", "point", """
img_as_bool | Reuse native dtype-specific half-range threshold conversion | signed input, exact half ties, warning, copy and nonfinite values | dtype extrema and threshold-adjacent values
img_as_float | Reuse native integer-to-float scaling while retaining floating precision behavior | float16/float32 preservation, signed scaling and force_copy | all scalar dtypes and alias/copy cases
img_as_float32 | Reuse native normalized cast to float32 | integer endpoints, precision loss and force_copy | wide integers and float64 rounding boundaries
img_as_float64 | Reuse native normalized cast to float64 | signed minimum scaling and force_copy | integer extrema and strided arrays
img_as_int | Reuse native normalized signed-int16 conversion | input range error, clipping, rounding and warning | range endpoints and float half steps
img_as_ubyte | Reuse native normalized uint8 conversion | signed negative clipping, precision warning, rounding and copy | negative signed input and 255-scale ties
img_as_uint | Reuse native normalized uint16 conversion | input bit-depth scaling, signed negatives and rounding | uint8/uint64 and float threshold cases
invert | Reuse native dtype-dependent range complement | bool, signed integer, unsigned, signed_float and NaNs | dtype extrema and both floating conventions
compare_images | Reuse native difference/absolute blend and checkerboard coordinate selection | method, dtype promotion, n_tiles division and channel shape | exact small checkerboards and range extremes
""", effort="S")
add("util", "metadata", """
FailedEstimationAccessError | Preserve owned AttributeError-compatible exception provider and inherited metadata | args/name/obj, notes, pickle, traceback and string/state behavior | construct/raise/catch/pickle exception lifecycle
PendingSkimage2Change | Own warning subclass preserving PendingDeprecationWarning inheritance and metadata | filtering category, args/notes, pickle and traceback | warning filter and round-trip lifecycle
crop | Return slice view or requested-order copy after crop-width normalization | scalar/tuple widths, negative/large crops, order and memory sharing | asymmetric N-D crops and C/F/negative strides
dtype_limits | Reuse owned dtype metadata lookup with clip_negative | dtype recognition, signed range and unsupported dtype errors | every supported dtype and clip flag
lookfor | Preserve documentation-search adapter over pinned public namespace | search normalization, printed output and optional documentation source | fixed query hits and no-match output
regular_grid | Choose rounded per-axis strides and slices for approximately uniform sample count | n_points extremes, dimensionality and slice start/stop convention | anisotropic/tiny shapes and target counts
slice_along_axes | Normalize axes/slice pairs then return view or copy | duplicate/negative axes, bounds and sharing | reordered/negative axes and copy flags
view_as_blocks | Construct validated reshape/strided block view | divisibility, dimensionality, contiguous assumptions and write sharing | C/F/noncontiguous arrays and block mutation
view_as_windows | Construct rolling strided view with validated window/step | scalar/vector step, shape, strides and dangerous overlap semantics | overlapping windows and negative/zero invalid steps
""", effort="S")
add("util", "reduce", """
map_array | Reuse native dense/sparse label lookup and output casting | unsorted/duplicate input keys, missing keys, out shape/alias and large integers | sparse/wide keys and overlap cases
unique_rows | Stable/contract-ordered row deduplication using contiguous byte keys or typed comparisons | dtype, NaNs, row order and noncontiguous input | duplicate/noncontiguous rows with special values
""", effort="S")
add("util", "draw", """
label_points | Round coordinates with exact tie rule and scatter sequential point IDs | duplicate pixel overwrite, bounds, N-D output_shape and rounding | half-integer and coincident points
montage | Compute tile grid/padding, optional native rescale and fill before block copies | fill=mean, channel_axis, grid_shape, padding and dtype | incomplete grids and multichannel tiles
regular_seeds | Reuse regular_grid slices then assign ordered seed labels | dtype, label numbering and requested-count approximation | anisotropic grids and limited integer dtypes
""", effort="S")
add("util", "random", """
random_noise | Preserve Generator draws/distributions and native mode-specific transforms/clipping | gaussian/localvar/poisson/salt/pepper/s&p/speckle, signed range and RNG state | saved-state mode matrix and clipping/range edges
""", effort="L")
add("util", "metadata", """
apply_parallel | Preserve Dask/lazy-array overlap adapter or own bounded chunk executor for eager arrays | compute lazy/eager, chunks/depth/mode, channel_axis, callbacks and dtype inference | instrumented boundary-aware function on irregular chunks
""", effort="L")

# Family dependencies are capability menus, not universal blockers.  These
# smaller sets describe the minimum route; specific algorithms extend/replace
# them below.  The worker scheduler is optional unless explicitly requested.
MINIMUM_DEPS = {
    "point": "array pointwise", "reduce": "array reduce", "stencil": "array neighborhood",
    "rank": "array histogram neighborhood", "component": "array components",
    "footprint": "array geometry", "draw": "array geometry", "geometry": "array geometry",
    "resample": "array resample", "model": "array small_solver", "tree": "array order_graph",
    "distance": "array distance", "fft": "array fft", "feature": "array neighborhood",
    "iterative": "array neighborhood", "graph": "array order_graph",
    "segment": "array components order_graph", "random": "array rng",
    "asset": "assets provider", "io": "array provider", "ui": "provider",
    "learn": "array provider", "metadata": "array",
}
for spec in SPECS.values():
    if "alias" not in spec:
        spec["dependencies"] = MINIMUM_DEPS[spec["family"]].split()


def needs(names, deps):
    for api in names.split():
        override(api, dependencies=deps.split())


needs("graph.MCP graph.MCP_Connect graph.MCP_Flexible graph.MCP_Geometric graph.route_through_array graph.shortest_path graph.merge_hierarchical", "array order_graph")
needs("graph.cut_threshold graph.rag_boundary graph.rag_mean_color graph.RAG", "array components order_graph provider")
needs("graph.central_pixel", "array order_graph provider")
needs("graph.cut_normalized", "array order_graph sparse_solver rng provider")
needs("graph.pixel_graph", "array geometry provider")
needs("feature.match_template", "array neighborhood reduce")
needs("feature.match_descriptors", "array pointwise reduce")
needs("feature.graycomatrix feature.graycoprops", "array histogram reduce")
needs("feature.haar_like_feature feature.haar_like_feature_coord feature.multiblock_lbp", "array reduce geometry")
needs("feature.BRIEF feature.ORB", "array neighborhood geometry rng")
needs("feature.CENSURE feature.blob_doh", "array reduce neighborhood geometry")
needs("feature.SIFT feature.blob_dog feature.blob_log feature.daisy feature.multiscale_basic_features", "array neighborhood resample reduce")
needs("feature.Cascade", "array reduce assets")
needs("feature.corner_subpix", "array neighborhood small_solver")
needs("feature.corner_peaks feature.peak_local_max", "array neighborhood geometry")
needs("feature.draw_haar_like_feature", "array geometry pointwise rng")
needs("feature.fisher_vector", "array reduce pointwise provider")
needs("feature.learn_gmm future.TrainableSegmenter future.fit_segmenter future.predict_segmenter", "array provider")
needs("filters.rank_order exposure.histogram exposure.cumulative_distribution exposure.equalize_hist exposure.match_histograms filters.threshold_otsu filters.threshold_yen filters.threshold_triangle filters.threshold_isodata filters.threshold_minimum filters.threshold_multiotsu", "array histogram reduce")
needs("filters.threshold_li", "array reduce pointwise provider")
needs("filters.window", "array pointwise resample provider")
needs("measure.CircleModel measure.EllipseModel measure.LineModelND", "array small_solver")
needs("measure.ransac", "array rng provider")
needs("measure.RansacModelProtocol util.FailedEstimationAccessError util.PendingSkimage2Change util.lookfor", "provider")
needs("measure.inertia_tensor_eigvals", "array reduce small_solver")
needs("measure.pearson_corr_coeff", "array reduce provider")
needs("measure.euler_number measure.perimeter measure.perimeter_crofton", "array neighborhood geometry")
needs("measure.regionprops measure.regionprops_table", "array components reduce geometry provider")
needs("measure.block_reduce", "array reduce provider")
needs("metrics.contingency_table metrics.adapted_rand_error metrics.variation_of_information", "array histogram reduce provider")
needs("metrics.normalized_mutual_information measure.shannon_entropy", "array histogram reduce")
needs("morphology.convex_hull_image morphology.convex_hull_object", "array geometry components")
needs("morphology.skeletonize morphology.thin", "array neighborhood geometry")
needs("morphology.footprint_from_sequence", "array neighborhood")
needs("morphology.medial_axis", "array distance neighborhood rng")
needs("morphology.remove_objects_by_distance", "array distance components")
needs("registration.optical_flow_ilk", "array neighborhood resample small_solver")
needs("registration.optical_flow_tvl1", "array neighborhood resample reduce")
needs("restoration.inpaint_biharmonic", "array neighborhood components sparse_solver")
needs("restoration.richardson_lucy", "array neighborhood")
needs("restoration.denoise_wavelet restoration.estimate_sigma", "array neighborhood reduce provider")
needs("restoration.unsupervised_wiener", "array fft rng provider")
needs("restoration.unwrap_phase", "array order_graph rng")
needs("restoration.calibrate_denoiser restoration.cycle_spin restoration.denoise_invariant", "array neighborhood reduce provider")
needs("segmentation.active_contour", "array neighborhood resample small_solver")
needs("segmentation.chan_vese", "array neighborhood reduce")
needs("segmentation.morphological_chan_vese segmentation.morphological_geodesic_active_contour", "array neighborhood reduce provider")
needs("segmentation.random_walker", "array components sparse_solver")
needs("segmentation.quickshift", "array neighborhood pointwise order_graph rng")
needs("segmentation.slic", "array neighborhood components pointwise reduce")
needs("segmentation.join_segmentations", "array histogram components provider")
needs("segmentation.mark_boundaries", "array neighborhood pointwise resample")
needs("transform.matrix_transform", "array pointwise")
needs("transform.PiecewiseAffineTransform", "array geometry small_solver")
needs("transform.estimate_transform", "array provider")
needs("transform.frt2 transform.ifrt2 transform.order_angles_golden_ratio", "array reduce geometry")
needs("transform.hough_circle transform.hough_circle_peaks transform.hough_line transform.hough_line_peaks transform.hough_ellipse", "array geometry histogram")
needs("transform.probabilistic_hough_line", "array geometry histogram rng")
needs("transform.radon", "array resample reduce")
needs("transform.iradon", "array fft resample reduce")
needs("transform.iradon_sart", "array resample reduce geometry")
needs("util.map_array util.unique_rows", "array histogram")
needs("util.apply_parallel", "array provider")
needs("data.binary_blobs", "array rng neighborhood histogram")
needs("draw.random_shapes", "array rng geometry")
needs("draw.ellipsoid_stats", "pointwise provider")

# Source-backed peer audit corrections and explicit computational/provider edges.
override("feature.fisher_vector",
         algorithm="Read GMM probabilities/parameters through provider; compute mixture-weight, mean and variance gradients, concatenate K+2KD values, then optional power/L2 normalization",
         hazards="d_pi/d_mu/d_sigma scaling/order, covariance convention, alpha and zero normalization",
         workload="hand-computed small GMM with K+2KD output and large descriptor batches",
         benchmark_kind="compute-with-provider",
         benchmark="Time full public Fisher vector and separately native gradient/normalization core using fixed provider outputs; report predict_proba cost and keep model identical.",
         frontier="Own C++ Fisher-gradient/normalization arithmetic is the target; GMM predict_proba and model state remain an explicit estimator provider unless a separate C++ GMM solver is qualified.")
override("feature.match_template",
         algorithm="Start with direct correlation plus integral sum/squared-sum normalization; add an FFT backend only for measured large-template crossovers",
         optional_dependencies=["fft"])
override("morphology.reconstruction",
         hazards="seed-mask ordering, dilation/erosion, footprint/offset, integer extremes, signed-zero representative/provider decision and N-D output ownership")
override("metrics.adapted_rand_error",
         algorithm="Reuse contingency counts and self-pair-excluding n*(n-1) sums for joint and marginal totals; compute precision, recall and alpha-weighted error",
         hazards="ignore_labels, supplied table, alpha, singleton/all-singleton zero divisions and returned order",
         workload="exact tiny partitions, singleton/all-singleton cases and sparse large labels")
override("draw.ellipsoid_stats", effort="M",
         frontier="Direct native volume is simple; surface area needs licensed accurate elliptic-integral implementation or disclosed special-function provider. Adapter effort does not imply standalone completion.")
override("measure.pearson_corr_coeff",
         frontier="Native correlation reductions plus a separately qualified beta/t-distribution tail implementation or disclosed statistical p-value provider; provider retention does not complete standalone C++.")
override("restoration.denoise_wavelet",
         frontier="Initial PyWavelets adapter preserves the public wavelet catalogue; owned C++ filter-bank and coefficient/extension catalogue are a separate backend milestone, required for standalone native completion.")
override("restoration.estimate_sigma",
         frontier="Native MAD reduction can reuse explicit wavelet coefficients; a Python wavelet transform is a disclosed provider until a C++ filter-bank backend is qualified.")
override("util.apply_parallel",
         frontier="Dask lazy graph/object behavior remains a genuine provider contract. Owned eager C++ chunk arithmetic/executor does not claim replacement of arbitrary Python functions or distributed Dask.")
override("morphology.mirror_footprint",
         benchmark_kind="noncompute-boundary",
         benchmark="Noncompute boundary: check reversed-view strides, sharing, sequence structure and iteration counts; optional wrapper-overhead timing is not an image-kernel speed claim.",
         simd="No SIMD for axis-reversal view/sequence metadata; no pixel arithmetic is required by this route.",
         workers="Serial view/sequence construction; no worker acceleration claimed.",
         frontier="Owned metadata wrapper with NumPy-compatible reversed views and sequence handling; legitimate API behavior, not a native image kernel or standalone C++ array-provider completion.")
override("restoration.calibrate_denoiser", benchmark_kind="compute-with-provider",
         benchmark="Time parameter-search orchestration and native masked-loss core separately with a fixed deterministic denoiser; record denoiser calls and include its cost in end-to-end latency.")
override("restoration.cycle_spin", benchmark_kind="compute-with-provider",
         benchmark="Time shift/copy/reduction core and full fixed-func cycle-spin separately with one worker and fixed summation order; record callback invocation count.")
override("restoration.denoise_invariant", benchmark_kind="compute-with-provider",
         benchmark="Time mask/interpolation/combine core and full fixed-denoiser invocation separately with one worker; keep mask sequence fixed.")
for api in ("restoration.calibrate_denoiser", "restoration.cycle_spin", "restoration.denoise_invariant"):
    override(api, simd="Vectorize owned mask/interpolation/shift/reduction pixels; arbitrary denoiser callback remains a provider call.",
             workers="Default one; independent shifts/masks only if provider thread safety and original callback contract permit; share global budget.",
             frontier="Own array preparation/loss/reduction arithmetic; preserve user-supplied denoiser callable behavior. Python callable support is not a C++ denoiser implementation.")

# Frozen work is a review gate, not an instruction to execute while paused.
CHECKPOINTS = {
    "transform.matrix_transform": "Combined 199-path candidate remains unaccepted; ready metadata 1993c87 is not algorithm qualification. Review original/runtime and whole-product gate before implementation credit.",
    "morphology.max_tree": "Six max-tree APIs remain private candidate scope; combined qualification and original baseline/one-worker measurements are pending.",
    "morphology.max_tree_local_maxima": "Private max-tree candidate only; no accepted public-path increment.",
    "morphology.area_opening": "Private max-tree candidate only; whole-product acceptance and timings pending.",
    "morphology.area_closing": "Private candidate with closing dtype repair; retain original/reference cases and rerun genuine composed qualification only after resume.",
    "morphology.diameter_opening": "Private max-tree candidate only; whole-product acceptance and timings pending.",
    "morphology.diameter_closing": "Private candidate with closing dtype repair; no accepted public-path increment.",
    "morphology.reconstruction": "600a reference/draft retained and unbuilt; characterize/review before building after resume.",
    "filters.threshold_li": "1329/337 reference and private audit retained, not yet merged; accepted native-partial classification is unchanged.",
    "restoration.denoise_tv_chambolle": "af7b2af composed source is accepted; old-source adapter and new one-worker comparison remain pending. No speed claim for the merged single-admission change.",
    "metrics.mean_squared_error": "Reverse-stride parity is accepted; its speed remains unmeasured.",
    "metrics.normalized_root_mse": "Reverse-stride parity is accepted; its speed remains unmeasured.",
    "metrics.peak_signal_noise_ratio": "Reverse-stride parity is accepted; its speed remains unmeasured.",
}
for api in CHECKPOINTS:
    if api in SPECS:
        override(api, checkpoint=CHECKPOINTS[api])
for api in ("transform.matrix_transform", "morphology.max_tree", "morphology.max_tree_local_maxima", "morphology.area_opening", "morphology.area_closing", "morphology.diameter_opening", "morphology.diameter_closing", "filters.threshold_li", "restoration.denoise_tv_chambolle"):
    override(api, wave="W0")

WAVES = {
    "W0": "Review frozen ready/private checkpoints and close their existing qualification/performance gates; no new algorithm credit before acceptance.",
    "W1": "Cheap independent kernels, wrappers and selected existing native-partial contract closure; exact formulas/LUTs and already-owned kernels first.",
    "W2": "Shared neighborhood, histogram/order, connectivity and reconstruction primitives and their consumers.",
    "W3": "Geometry, resampling composition, small solvers and geometric class state/member closure.",
    "W4": "Distance/FFT/multiscale features, registration and restoration; choose reviewed backends before dependent optimization.",
    "W5": "Dependent graph, segmentation and specialized model/solver pipelines with exact event/tie/state semantics.",
    "P": "Independent provider/assets/metadata lane starts alongside W1; explicit adapter boundary is useful progress but not native arithmetic completion.",
}
EFFORTS = {
    "XS": "Small owned table/alias/metadata audit; excludes full downstream closure.",
    "S": "One bounded direct kernel, adapter or extension of an existing kernel; provider/back-end replacements excluded.",
    "M": "Several shared passes, nontrivial boundary semantics or a moderate solver/backend wrapper.",
    "L": "Stateful, iterative, stochastic, topology-sensitive or broad compatibility surface; split into reviewable contracts.",
    "XL": "Reserved for a separately scoped backend/catalogue replacement; not a calendar estimate.",
}
NONCOMPUTE_FAMILIES = {"asset", "io", "ui", "metadata", "learn"}
COMMON_POLICY = {
    "scope": "Planning proposals, not live execution state. See coverage/work_items.json and coverage/EXECUTION_PLAN.md for current assignments and authorization. This document changes no API, evidence or measured-speed status.",
    "existing_proofs": "Existing strict/bitwise/source-qualified contracts and frozen evidence remain unchanged. A new prospective policy cannot relax them or relabel old failures.",
    "structural": "Exact signature/default/return container, dtype/shape/layout, integer results, labels, tie and branch decisions, warning/error classes/messages/order, callbacks, mutation/alias/state and namespace identity where the public contract specifies them.",
    "floating": "Before observing candidate results, declare per-output dtype/conditioning-aware rtol/atol or ULP bounds from algorithm stability and application needs, plus nonfinite/signed-zero rules. Validate with untouched holdouts. A separate bitwise track is mandatory for strict existing scopes or contracts requiring exact arithmetic. Numerical allclose alone cannot excuse changed labels, convergence branches or errors.",
    "discovery": "Routes below are proposed engineering paths, not validated behavior. Inspect pinned 0.26.0 source and capture contract cases before implementing; ambiguity is an open contract question, not permission to guess.",
    "acceptance": "Freeze public original cases; scalar typed kernel and actual public wrapper; fallback disabled for claimed native scope; scalar/SIMD and one/multiple-global-worker checks; actual supported build/profile/SDK gates; isolated one-worker public-call comparisons with preserved regressions. Metadata/provider adapters use appropriate contract checks and do not receive image-kernel credit.",
    "benchmark": "Proposal, not measurement: use identical pinned inputs/options and predeclared parity policy; preserve source/package/runtime/CPU/dispatch/provider identities; xx.set_num_threads(1) and matching upstream BLAS/FFT/OpenMP limits. Warm appropriately, alternate/paired repeated runs, retain raw samples and every slower case. Run alone, without concurrent builds/tests or changing files used by a running benchmark. Report allocation-inclusive public latency; optional core-only diagnostics are labeled separately.",
    "providers": "An owned C++ objective cannot be satisfied by NumPy/SciPy image-algorithm forwarding. Python metadata, user callbacks, GUI, codec, Dask and estimator objects are explicit ecosystem boundaries. Any retained numerical provider is named; standalone C++ completion needs the relevant reviewed native backend, licenses, ABI/wheel/SDK profiles and its own parity/performance gates.",
    "parallelism": "Default one process-global worker budget; optional workers only for independent/proven-equivalent work, no nested uncontrolled pools. SIMD is a feasibility proposal; exact ordered or graph/event cores may stay serial. No universal speedup promise.",
    "priority": "Waves are relative dependency/effort priorities, not strict global barriers or a scope reduction. Family capabilities are menus; each row has separate mandatory route dependencies. Existing native-partial entries may need extensive uncovered-domain work despite a small next increment.",
}


def resolved_spec(api):
    spec = SPECS[api]
    if "alias" not in spec:
        return dict(spec), api
    target = spec["alias"]
    if target not in SPECS or "alias" in SPECS[target]:
        raise ValueError("alias must target an explicit canonical route: " + api)
    resolved = dict(SPECS[target])
    resolved.update(alias=target, alias_evidence=spec["alias_evidence"], effort="XS")
    return resolved, target


def next_action(entry, spec, canonical):
    api = entry["api"]
    status = entry["implementation"]
    if canonical != api:
        return f"Assert this namespace resolves to the canonical callable and preserves signature/visibility/fallback; run canonical cases through both paths. {spec['alias_evidence']}"
    if spec.get("checkpoint") and spec["wave"] == "W0" and status != "native-partial":
        return "Consult the live work ledger before acting on this historical checkpoint; close only a still-open named gate: " + spec["checkpoint"]
    if entry["kind"] == "class":
        prefix = "Audit and extend the owned partial class" if status == "native-class-partial" else "Capture constructor/state/member contracts and implement the first owned class slice"
        return f"{prefix}; first use {spec['workload']}. Enumerate the static-member appendix and dynamic fields before declaring class parity."
    if status == "native-partial":
        return f"Audit existing wrapper/kernel and recorded parity/benchmark references; extend uncovered {spec['hazards']}. Select optimization from retained measured regressions; do not reimplement the accepted core."
    if status in ("metadata", "exception-metadata-provider"):
        return f"Audit the owned metadata contract for {spec['hazards']}; add missing identity/state/error cases, without image-kernel completion credit."
    if spec["family"] == "asset":
        return f"Create the pinned resource/license/hash/cache fixture for {spec['algorithm']}; verify {spec['hazards']} before selecting or replacing a codec."
    if spec["family"] in {"io", "ui", "learn", "metadata"}:
        return f"Capture provider/state calls for {spec['workload']}; own the specified adapter and its {spec['hazards']} contract. Split any native numerical backend into a separately qualified step."
    return f"Freeze {spec['workload']} against pinned upstream, including {spec['hazards']}; implement the stated scalar route using only its mandatory primitives, then qualify acceleration."


def class_followup(entry):
    spec, _ = resolved_spec(entry["api"])
    members = [dict(api=m["api"], kind=m["kind"], signature=m.get("signature"),
                    defined_by=m.get("defined_by"), inherited=m.get("inherited", False),
                    current_implementation=m["implementation"],
                    current_execution_kind=m.get("execution_kind"),
                    existing_parity_reference=m.get("parity_test"),
                    existing_benchmark_reference=m.get("benchmark"))
               for m in entry.get("members", [])]
    return dict(class_api=entry["api"], current_implementation=entry["implementation"],
                static_member_count=len(members), static_members=members,
                first_slice=spec["algorithm"],
                member_action="Classify every listed constructor/method/property/inherited descriptor separately; capture signature, state before/after, errors, copies/views, serialization and array protocol where applicable. Split numerical methods from metadata/provider methods and give each numerical method its own public-call benchmark before claiming completion.",
                dynamic_scope="Inventory instance fields and runtime-generated attributes from successful/failed construction, fit/detect/extract, inverse, copy and reload as applicable; static discovery is not exhaustive.")


def make_document():
    raw = INVENTORY.read_bytes()
    inventory = json.loads(raw)
    entries = inventory["entries"]
    names = [e["api"] for e in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate inventory API")
    missing, extra = set(names) - set(SPECS), set(SPECS) - set(names)
    if missing or extra:
        raise ValueError(f"explicit route coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if len(entries) != 477 or len(inventory["modules"]) != 20:
        raise ValueError("pinned inventory scope changed; review explicit routes and counts, do not silently extend")
    records = []
    for entry in entries:
        api = entry["api"]
        spec, canonical = resolved_spec(api)
        fam = FAMILIES[spec["family"]]
        for dependency in spec["dependencies"]:
            if dependency not in PRIMITIVES:
                raise ValueError("dangling primitive: " + dependency)
        if spec["wave"] not in WAVES or spec["effort"] not in EFFORTS:
            raise ValueError("unknown priority: " + api)
        kind = spec.get("benchmark_kind", "noncompute-boundary" if spec["family"] in NONCOMPUTE_FAMILIES else "compute")
        algorithm = spec["algorithm"]
        if canonical != api:
            algorithm = f"Exact alias of {canonical}; reuse its route: {algorithm}. No extra algorithm credit."
        elif entry["implementation"] == "native-partial":
            algorithm = "Audit/extend existing native implementation: " + algorithm
        row = dict(api=api, kind=entry["kind"], signature=entry.get("signature"),
                   current_implementation=entry["implementation"],
                   current_execution_kind=entry.get("execution_kind"),
                   current_native_default_case_dtypes=entry.get("native_default_case_dtypes", []),
                   canonical_api=canonical, identity_alias=canonical != api,
                   family=spec["family"], wave=spec["wave"], next_increment_effort=spec["effort"],
                   easiest_credible_path=algorithm,
                   mandatory_route_dependencies=spec["dependencies"],
                   optional_route_dependencies=spec.get("optional_dependencies", []),
                   optional_acceleration_dependency="workers" if kind != "noncompute-boundary" else None,
                   next_concrete_action=next_action(entry, spec, canonical),
                   compatibility_hazards=spec["hazards"],
                   parity_cases_to_freeze=spec["workload"],
                   numerical_policy=spec.get("numerical_policy", fam["numerical_policy"]),
                   simd_strategy=spec.get("simd", fam["simd"]),
                   optional_global_workers_strategy=spec.get("workers", fam["workers"]),
                   benchmark_plan=dict(status="proposal-not-measured-results", kind=kind,
                                       specific_workload=spec["workload"],
                                       one_worker_recipe=spec.get("benchmark", fam["benchmark_recipe"]),
                                       existing_inventory_reference=entry.get("benchmark")),
                   native_provider_frontier=spec.get("frontier", fam["native_provider_frontier"]),
                   existing_inventory_parity_reference=entry.get("parity_test"),
                   evidence_interpretation="Inventory references locate existing scoped evidence; they do not prove this proposed expanded contract or benchmark passes.")
        if canonical != api:
            row["alias_identity_evidence"] = spec["alias_evidence"]
            row["benchmark_plan"]["alias_rule"] = "Run alias dispatch/identity checks; inherit canonical algorithm timings only after identity proof and report public-wrapper overhead separately. Never count alias as a new algorithm."
        if spec.get("checkpoint"):
            row["frozen_checkpoint_note"] = spec["checkpoint"]
        if entry["kind"] == "class":
            row["class_member_scope"] = f"Separate follow-up required for {len(entry.get('members', []))} static members and unassessed dynamic state."
        records.append(row)
    classes = [class_followup(e) for e in entries if e["kind"] == "class"]
    member_count = sum(c["static_member_count"] for c in classes)
    if len(classes) != 29 or member_count != 284:
        raise ValueError("static class inventory changed; review follow-up scope")
    counts = dict(api_entries=len(records), modules=len(inventory["modules"]),
                  current_implementation=dict(sorted(Counter(r["current_implementation"] for r in records).items())),
                  kinds=dict(sorted(Counter(r["kind"] for r in records).items())),
                  family=dict(sorted(Counter(r["family"] for r in records).items())),
                  wave=dict(sorted(Counter(r["wave"] for r in records).items())),
                  next_increment_effort=dict(sorted(Counter(r["next_increment_effort"] for r in records).items())),
                  benchmark_boundary=dict(sorted(Counter(r["benchmark_plan"]["kind"] for r in records).items())),
                  identity_aliases=sum(r["identity_alias"] for r in records),
                  canonical_routes=len({r["canonical_api"] for r in records}),
                  classes=len(classes), statically_discovered_class_members=member_count)
    families = {key: dict(title=value["title"], capability_menu=value["dependencies"],
                          **{k: v for k, v in value.items() if k not in ("title", "dependencies")})
                for key, value in FAMILIES.items()}
    for key, value in families.items():
        if not set(value["capability_menu"]) <= PRIMITIVES.keys():
            raise ValueError("dangling family capability: " + key)
    return dict(schema_version=1, artifact_kind="planning-only-per-api-routes",
                reference_version=inventory["reference_version"], source_inventory="coverage/api.json",
                source_inventory_sha256=hashlib.sha256(raw).hexdigest(),
                accepted_baseline_context="main 12fad34 at the historical planning review; authoritative classifications copied from api.json. Frozen checkpoints describe that review, not current execution state; consult work_items.json.",
                scope_statement="Exactly 477 inventoried public entries across 20 modules, plus a separate 29-class/284-static-member follow-up. Other namespaces, instance fields, runtime-generated attributes and future upstream additions remain outside the current static inventory; this is not an exhaustive entire-library completion claim.",
                algorithm_count_note="26 source-backed identity aliases are namespace routes, not additional algorithms. The remaining canonical routes also include wrappers/classes/assets and are not an algorithm-completion count.",
                common_policy=COMMON_POLICY, summary_counts=counts, wave_definitions=WAVES,
                effort_definitions=EFFORTS, shared_primitives=PRIMITIVES, families=families,
                entries=records, class_member_followup=classes,
                additional_scope_followup=[
                    "Inventory runtime RegionProperties objects returned by regionprops, ArrayMap-like mappings, generators, callback-created objects and class instance fields; exported static class discovery does not cover them.",
                    "Review other nested/plugin namespaces and callable/noncallable public symbols omitted from the pinned 477-key manifest before calling the full API exhaustive.",
                    "Track backend/catalogue replacements separately: codecs/containers, asset licensing/cache, FFT, sparse/small solvers, triangulation/hulls, wavelet catalogue, probability special functions and optional native GMM/classifier solvers.",
                    "Retain every existing proof, baseline package, raw timing/slower case, paused source work and benchmark-used file. Planning document generation must not trigger builds, scientific runs, cleanup, merges or dispatch.",
                ])


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(doc):
    summary = doc["summary_counts"]
    lines = [
        "# Easiest credible implementation path for every inventoried API", "",
        "These are proposed routes and acceptance work, not new implementations, measured speedups or a change to the authoritative inventory. Frozen checkpoints describe the historical planning review. Current execution state and authorization are in `work_items.json` and `EXECUTION_PLAN.md`.", "",
        doc["scope_statement"], "",
        f"Current classification: {summary['current_implementation'].get('native-partial', 0)} native-partial image entries, {summary['current_implementation'].get('upstream-only', 0)} upstream-only entries, {summary['current_implementation'].get('native-class-partial', 0)} native-class-partial entries, {summary['current_implementation'].get('metadata', 0)} metadata functions and {summary['current_implementation'].get('exception-metadata-provider', 0)} exception-metadata-provider. Partial means partial; listed test/benchmark references do not prove the expanded proposal. " + doc["algorithm_count_note"], "",
        "Generated from `coverage/api.json` (scikit-image " + doc["reference_version"] + ") by `python tools/generate_api_implementation_paths.py`; verify without writes using `python tools/generate_api_implementation_paths.py --check`. Inventory SHA-256: `" + doc["source_inventory_sha256"] + "`.", "",
        "## How to use the routes", "",
    ]
    for key in ("priority", "existing_proofs", "structural", "floating", "discovery", "acceptance", "benchmark", "providers", "parallelism"):
        lines.extend(["**" + key.replace("_", " ").capitalize() + ":** " + COMMON_POLICY[key], ""])
    lines.extend(["Effort describes the next reviewable increment, not whole-API completion or elapsed time. Asset/GUI/model/I/O adapter effort excludes replacing the full external backend. Waves can overlap; the independent P lane starts early. Exact alias rows share canonical algorithm work and get namespace/dispatch checks.", "", "## Priority and counts", "", "| Wave | Entries | Intended order |", "| --- | ---: | --- |"])
    for key, description in WAVES.items():
        lines.append(f"| {key} | {summary['wave'].get(key, 0)} | {cell(description)} |")
    lines.extend(["", "| Next increment effort | Entries | Meaning |", "| --- | ---: | --- |"])
    for key, description in EFFORTS.items():
        lines.append(f"| {key} | {summary['next_increment_effort'].get(key, 0)} | {cell(description)} |")
    lines.extend(["", "## Shared primitives", "", "These are reusable implementation capabilities. A row's **Deps** column is its minimum proposed route, while a family's capability menu is broader and is not a blanket blocker. Optional acceleration uses the single global worker budget; it need not precede a correct scalar implementation.", "", "| Primitive | Reuse / dependency decision |", "| --- | --- |"])
    for key, description in PRIMITIVES.items():
        lines.append(f"| `{key}` | {cell(description)} |")
    lines.extend(["", "## Family execution, numerical and benchmark policies", "", "Every API row inherits its named family's SIMD, optional-worker, numerical and baseline benchmark recipe; per-API overrides are printed in that row and expanded in JSON. A recipe is a proposed one-worker experiment, never a result. Common policy above applies even when a family summary is shorter.", "", "| Family (entries) | Capability menu | SIMD / optional workers | Numerical contract | One-worker benchmark or noncompute boundary | Native/provider frontier |", "| --- | --- | --- | --- | --- | --- |"])
    for key, fam in doc["families"].items():
        lines.append("| " + " | ".join(map(cell, [f"{key} ({summary['family'][key]})", ", ".join(fam["capability_menu"]), fam["simd"] + " " + fam["workers"], fam["numerical_policy"], fam["benchmark_recipe"], fam["native_provider_frontier"]])) + " |")
    lines.extend(["", "## All 477 public entry routes", "", "**Current** is copied from the inventory, not a completion claim. **Family** links each row logically to the policies above; **hazards/cases** identify API-specific parity and benchmark inputs. Existing reference strings locate prior scoped work only. The machine-readable artifact expands all inherited policies per row.", ""])
    for module in json.loads(INVENTORY.read_text(encoding="utf-8"))["modules"]:
        rows = [r for r in doc["entries"] if r["api"].rsplit(".", 1)[0] == module]
        lines.extend([f"### {module} ({len(rows)})", "", "| API | Current; priority / effort | Family; mandatory deps | Easiest credible route | Next concrete action | Compatibility hazards; parity / benchmark cases | Existing scoped references; policy overrides |", "| --- | --- | --- | --- | --- | --- | --- |"])
        for row in rows:
            spec, _ = resolved_spec(row["api"])
            refs = []
            if row["existing_inventory_parity_reference"]:
                refs.append("Parity ref: " + row["existing_inventory_parity_reference"])
            if row["benchmark_plan"]["existing_inventory_reference"]:
                refs.append("Benchmark ref: " + row["benchmark_plan"]["existing_inventory_reference"])
            if not refs:
                refs.append("No inventory parity/benchmark reference; proposed cases still need capture.")
            for key, label, field in (("simd", "SIMD", "simd_strategy"), ("workers", "Workers", "optional_global_workers_strategy"), ("frontier", "Frontier", "native_provider_frontier")):
                if key in spec:
                    refs.append(label + ": " + row[field])
            if "benchmark" in spec:
                refs.append("One-worker: " + row["benchmark_plan"]["one_worker_recipe"])
            if row.get("frozen_checkpoint_note"):
                refs.append("Checkpoint: " + row["frozen_checkpoint_note"])
            if row.get("class_member_scope"):
                refs.append(row["class_member_scope"])
            dependencies = row["family"] + "; " + ", ".join(row["mandatory_route_dependencies"])
            if row["optional_route_dependencies"]:
                dependencies += "; optional " + ", ".join(row["optional_route_dependencies"])
            values = ["`" + row["api"] + "`", row["current_implementation"] + "; " + row["wave"] + " / " + row["next_increment_effort"], dependencies, row["easiest_credible_path"], row["next_concrete_action"], row["compatibility_hazards"] + ". Cases: " + row["parity_cases_to_freeze"], " ".join(refs)]
            lines.append("| " + " | ".join(map(cell, values)) + " |")
        lines.append("")
    lines.extend(["## Separate class-member follow-up: 29 classes, 284 static members", "", "These members do not add 284 completed APIs to the 477-entry count. Every constructor/method/property/inherited descriptor needs separate state and semantic classification, with native arithmetic benchmarks where applicable. Availability and a signature are not parity. Dynamic instance fields and generated attributes remain unassessed. The JSON appendix preserves each member's signature, defining class, inherited flag, current implementation and existing references.", "", "| Class | Current | Static members | First class slice |", "| --- | --- | ---: | --- |"])
    for cls in doc["class_member_followup"]:
        lines.append("| " + " | ".join(map(cell, [cls["class_api"], cls["current_implementation"], cls["static_member_count"], cls["first_slice"]])) + " |")
    lines.extend(["", "All 284 statically discovered member paths are listed below for follow-up; their current classifications are copied, not upgraded.", "", "| Static member path | Kind; current implementation | State / numerical follow-up |", "| --- | --- | --- |"])
    for cls in doc["class_member_followup"]:
        for member in cls["static_members"]:
            kind = member["kind"]
            name = member["api"].rsplit(".", 1)[-1]
            if kind in {"property", "attribute"}:
                action = "Capture access before/after successful and failed operations; mutation, alias/view and error behavior; classify any derived numerical property."
            elif name in {"__init__", "__new__", "__setstate__", "__setstate_cython__", "__reduce__", "__reduce_cython__", "__getstate__"}:
                action = "Capture construction/serialization arguments, successful/failed state, identity and error behavior; do not infer numerical methods from this metadata."
            elif name in {"estimate", "from_estimate", "fit", "predict", "predict_x", "predict_y", "predict_xy", "residuals", "__call__", "detect", "extract", "detect_and_extract", "find_costs", "traceback", "detect_multi_scale", "compute_features"}:
                action = "Freeze this method's inputs, return/state/error contract; apply the class-specific route and benchmark its numerical core/public invocation separately."
            else:
                action = "Check inherited/owned method semantics, return identity, callbacks/mutation/order and failures; separate provider/metadata behavior from any numerical core."
            lines.append("| " + " | ".join(map(cell, ["`" + member["api"] + "`", kind + "; " + member["current_implementation"], action])) + " |")
    lines.extend(["", "## Scope still requiring discovery", ""])
    lines.extend("- " + text for text in doc["additional_scope_followup"])
    return "\n".join(lines) + "\n"


def validate(doc, markdown):
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    original = {e["api"]: e for e in inventory["entries"]}
    records = doc["entries"]
    actual = [r["api"] for r in records]
    if len(actual) != 477 or len(set(actual)) != 477 or set(actual) != set(original):
        raise ValueError("JSON does not cover exact 477 inventory APIs once")
    required = ("easiest_credible_path", "next_concrete_action", "compatibility_hazards", "parity_cases_to_freeze", "simd_strategy", "optional_global_workers_strategy", "native_provider_frontier", "numerical_policy")
    for row in records:
        source = original[row["api"]]
        for field, source_field in (("current_implementation", "implementation"), ("kind", "kind"), ("signature", "signature"), ("current_execution_kind", "execution_kind")):
            if row[field] != source.get(source_field):
                raise ValueError("inventory field drift: " + row["api"] + ":" + field)
        if any(not row[field] for field in required):
            raise ValueError("empty route content: " + row["api"])
        if row["family"] not in doc["families"] or not set(row["mandatory_route_dependencies"]) <= doc["shared_primitives"].keys():
            raise ValueError("dangling route dependency: " + row["api"])
        if not set(row["optional_route_dependencies"]) <= doc["shared_primitives"].keys():
            raise ValueError("dangling optional route dependency: " + row["api"])
        if row["canonical_api"] not in original:
            raise ValueError("dangling alias target: " + row["api"])
        if row["identity_alias"] and not row.get("alias_identity_evidence"):
            raise ValueError("alias without source evidence")
        if not row["benchmark_plan"]["one_worker_recipe"] or row["benchmark_plan"]["status"] != "proposal-not-measured-results":
            raise ValueError("missing or overstated benchmark plan")
    table_names = re.findall(r"^\| `([^`]+)` \|", markdown, re.MULTILINE)
    public_names = [name for name in table_names if name in original]
    if Counter(public_names) != Counter(original.keys()):
        raise ValueError("Markdown must contain exactly one primary row for every public API")
    original_members = {m["api"]: m for e in inventory["entries"] for m in e.get("members", [])}
    members = [m for cls in doc["class_member_followup"] for m in cls["static_members"]]
    if len(members) != 284 or Counter(m["api"] for m in members) != Counter(original_members.keys()):
        raise ValueError("class member scope mismatch")
    if Counter(name for name in table_names if name in original_members) != Counter(original_members.keys()):
        raise ValueError("Markdown class member scope mismatch")
    for member in members:
        source = original_members[member["api"]]
        if member["current_implementation"] != source["implementation"] or member["signature"] != source.get("signature"):
            raise ValueError("member classification/signature drift")
    return dict(api_entries=477, classes=29, static_members=284,
                modules=20, aliases=doc["summary_counts"]["identity_aliases"],
                families=len(doc["families"]), primitives=len(doc["shared_primitives"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate exact coverage and generated artifacts without writing.")
    args = parser.parse_args()
    doc = make_document()
    markdown = render_markdown(doc)
    counts = validate(doc, markdown)
    serialized = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    outputs = ((JSON_PATH, serialized), (MD_PATH, markdown))
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, content in outputs
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            print("Planning artifacts stale or missing: " + ", ".join(stale), file=sys.stderr)
            return 1
        existing_doc = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        validate(existing_doc, MD_PATH.read_text(encoding="utf-8"))
        print(json.dumps(dict(check="passed-no-writes", **counts, summary=doc["summary_counts"]), sort_keys=True))
    else:
        for path, content in outputs:
            path.write_text(content, encoding="utf-8", newline="\n")
        print(json.dumps(dict(generated=[str(path.relative_to(ROOT)) for path, _ in outputs], **counts), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
