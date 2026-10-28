"""Reviewed native dispatch entry points. Partial coverage is not full API parity."""
ROLLING_KERNEL_FUNCTIONS = ('restoration.ball_kernel', 'restoration.ellipsoid_kernel')
COORDS_MOMENTS_FUNCTIONS = ('measure.moments_coords', 'measure.moments_coords_central')
LEVEL_SET_FUNCTIONS = ('segmentation.checkerboard_level_set', 'segmentation.disk_level_set')
MOMENTS_FUNCTIONS = ('measure.moments', 'measure.moments_central', 'measure.centroid')
DERIVED_MOMENTS_FUNCTIONS = ('measure.moments_normalized', 'measure.moments_hu')
NATIVE_FUNCTIONS = {
    'restoration': ('denoise_tv_chambolle', 'denoise_tv_bregman', 'ball_kernel', 'ellipsoid_kernel'),
    'segmentation': ('clear_border','relabel_sequential','checkerboard_level_set','disk_level_set'),
    'filters.rank': ('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy','autolevel','equalize','enhance_contrast','threshold','geometric_mean','noise_filter','subtract_mean','windowed_histogram','autolevel_percentile','enhance_contrast_percentile','gradient_percentile','mean_percentile','pop_percentile','subtract_mean_percentile','threshold_percentile','sum_percentile','percentile','mean_bilateral','pop_bilateral','sum_bilateral','otsu','majority'),
    'filters.rank.bilateral': ('mean_bilateral','pop_bilateral','sum_bilateral'),
    'filters.rank.generic': ('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy','autolevel','equalize','enhance_contrast','threshold','geometric_mean','noise_filter','subtract_mean','windowed_histogram','otsu','majority'),
    'measure': ('approximate_polygon','blur_effect','profile_line','mesh_surface_area','points_in_poly','grid_points_in_poly','euler_number','perimeter','perimeter_crofton','block_reduce','shannon_entropy','moments_coords','moments_coords_central','moments','moments_central','centroid','inertia_tensor','inertia_tensor_eigvals','moments_normalized','moments_hu','label','intersection_coeff','manders_coloc_coeff','manders_overlap_coeff'),
    'metrics': ('mean_squared_error','normalized_root_mse','peak_signal_noise_ratio','normalized_mutual_information'),
    'feature': ('hessian_matrix_eigvals','structure_tensor_eigenvalues','structure_tensor','corner_harris','corner_shi_tomasi','corner_foerstner','corner_kitchen_rosenfeld','hessian_matrix','shape_index','hessian_matrix_det'),
    'transform': ('integral_image','integrate','downscale_local_mean','resize','rescale','matrix_transform'),
    'morphology': ('label','remove_small_objects','remove_small_holes','footprint_rectangle','diamond','disk','ball','mirror_footprint','pad_footprint',
                   'erosion','dilation','opening','closing','white_tophat','black_tophat',
                   'binary_erosion','binary_dilation','binary_opening','binary_closing',
                   'max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing','reconstruction','h_maxima','h_minima',
                   'ellipse','octagon','octahedron','star','footprint_from_sequence'),
    'exposure': ('adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
                 'histogram', 'cumulative_distribution', 'equalize_hist', 'equalize_adapthist','is_low_contrast','match_histograms'),
    'color': ('label2rgb','rgba2rgb','gray2rgba','rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
              'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
              'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb',
              'rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
              'lab2lch','lch2lab','xyz_tristimulus_values','xyz2luv','luv2xyz','rgb2luv','luv2rgb',
              'separate_stains','combine_stains','rgb2hed','hed2rgb','deltaE_cie76','convert_colorspace',
              'deltaE_ciede94','deltaE_ciede2000','deltaE_cmc'),
    'filters': ('window','filter_forward','filter_inverse','wiener','butterworth','frangi','hessian','meijering','sato','gabor','gabor_kernel','gaussian', 'difference_of_gaussians', 'unsharp_mask', 'sobel', 'sobel_h', 'sobel_v', 'prewitt', 'prewitt_h',
                'prewitt_v', 'scharr', 'scharr_h', 'scharr_v', 'threshold_mean', 'threshold_otsu',
                'threshold_yen', 'threshold_triangle', 'threshold_isodata', 'threshold_minimum', 'threshold_li', 'threshold_multiotsu',
                'threshold_niblack','threshold_sauvola','threshold_local','rank_order','apply_hysteresis_threshold','correlate_sparse','median','farid','farid_h','farid_v','laplace','roberts','roberts_pos_diag','roberts_neg_diag'),
    'util': ('montage', 'regular_seeds', 'unique_rows', 'label_points', 'map_array', 'compare_images', 'invert', 'img_as_float', 'img_as_float32', 'img_as_float64', 'img_as_ubyte',
             'img_as_uint', 'img_as_int', 'img_as_bool', 'dtype_limits'),
}


MAX_TREE_PATHS = tuple('morphology.'+name for name in (
    'max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing'))


def matrix_transform_contract():
    """Partial native arithmetic, separate from owned protocol providers."""
    return dict(api='transform.matrix_transform', implementation='native-partial',
        interfaces={'matrix-transform-math-v1':31,'matrix-transform-math-v2':63},
        kernel='Checked finite float64 homogeneous coordinates, rank2 coordinate tables with2..4 spatial columns; typed signed strides and original whole-copy/layout arithmetic profiles',
        providers='Owned NumPy constructor/ones/hstack/matmul stages preserve exceptional FP, dtype, allocation, callback and protocol semantics. No upstream matrix_transform or ProjectiveTransform algorithm forwarding',
        runtime='Exact loaded UMATH/OpenBLAS bytes, getter ownership, ILP64, unforced source-audited corename and vendor threads1; unknown/interposed runtime stays owned provider',
        actual_runtime_evidence=('Haswell','Zen'),
        prepared_only_runtime_evidence=('Cooperlake','SkylakeX'),
        wider_dtype_scope='Non-float64 matrices retain certified dyadic native scope; wider general casting remains unassessed',
        workers='Shared sole_core global setter, default1 and optional4; packed SIMD disabled by SKIMAGEXX_DISABLE_SIMD without changing fused arithmetic',
        performance='Unmeasured end-to-end public pipeline; homogeneous allocation fusion remains future work')



def moments_contract(api):
    """Owned contractions with explicitly retained preparation/formula providers."""
    if api not in MOMENTS_FUNCTIONS:
        raise KeyError(api)
    return dict(api=api, implementation='native-partial', entrypoints=('moments_project',),
        kernel='Per-axis C++ float32/float64/complex64/complex128 contractions; ascending reduction with SIMD across independent outputs',
        providers='NumPy image conversion, coordinate powers, result layout, literal owned raw-to-central analytical formulas and centroid division; no upstream moments/einsum/dot/matmul forwarding',
        workers='One shared process-global snapshot per projection; default1; optional OpenMP by work threshold; portable builds refuse parallel workers',
        simd='Scalar baseline; runtime AVX2 real contiguous load/broadcast paths and generic real/complex lanes; no FMA or horizontal reassociation',
        floating_errors='Per-axis native FP capture/restore; source dot remains silent, source bmm_einsum/matmul reports flags using its policy snapshot',
        numerical_policy='Frozen component epsilon and axis-length weighted cell scales; signed-mass conditioning bounded at8 with g*kappa<1/4; source structural zeros and nonfinite/warning/error/state behavior exact',
        scope='100 frozen specifications across three APIs; orders0..4, bounded1D..4D, standard image types/layouts and finite complex spacing/centers',
        gaps='Extended/replaced protocols, forged/stale storage, complex nonfinite parameters, ill-conditioned signed mass and print/log FP handling remain unassessed',
        public_parity_test='tests/test_moments_public.py', fully_native=False,
        component_evidence='Four genuine runtime/dispatch modes, 203 checks each plus unchanged families; not combined product qualification',
        performance_evidence=None)


def window_contract():
    """Owned sampling; window generation and coordinate math remain providers."""
    return dict(api='filters.window', implementation='native-partial', entrypoints=('window_sample',),
        kernel='Owned float64 order0/1 grid-constant sampling at source float32 coordinates; ordered endpoint weights and zero-weight taps',
        providers='SciPy signal.get_window(fftbins=False); source NumPy shape normalization, radial coordinate construction and clipping',
        workers='Existing process-global worker snapshot, default1; optional OpenMP above the existing parallel threshold; portable builds refuse parallel workers',
        simd='Scalar baseline and runtime AVX2 packed arithmetic; strict floating-point flags and no FMA',
        dtype_scope='Shape factory with no pixel input; provider and output float64, coordinates float32. Shared uint8/f32/f64 fixture labels supply shapes only',
        option_scope='Orders0/1; higher interpolation orders and mixed-empty shapes remain explicit unsupported partitions',
        numerical_policy='Prospective8*eps64 relative plus32*eps64*max(1, finite reference peak) absolute; exact masks, boundary zeros, metadata, warnings, errors and caller state',
        public_parity_test='tests/test_window_public.py', fully_native=False,
        component_evidence='89 frozen specifications, 179 nodes per runtime/dispatch mode; four genuine component modes, not combined product qualification',
        performance_evidence=None)


def butterworth_contract():
    """Owned Fourier stages; mask/protocol providers remain explicitly partial."""
    return dict(api='filters.butterworth', implementation='native-partial',
        entrypoints=('fft_r2c', 'fft_c2r', 'fft_c2c'),
        kernel='Pinned PocketFFT real/complex ND transforms; inverse spatial normalization; no Python FFT provider',
        providers='NumPy padding, frequency-grid/mask arithmetic, spectrum multiplication and cropping',
        workers='One process-global worker snapshot; default1; OpenMP actual-team execution; portable builds refuse parallel workers',
        simd='Scalar baseline and runtime-dispatched AVX2 leaf; GNU/Clang vectors; MSVC vectorization unassessed',
        buffers='Bounded ndarray-backed storage; native aligned C-contiguous prepared buffers; output allocation boundaries revalidated',
        dtype_scope='Builtin f16/f32/f64, integers/bool and c64/c128; extended FFT precision and other storage/protocol partitions remain explicit gaps',
        numerical_policy='Prospective8*eps*L relative plus32*eps*L*input_peak absolute, L=ceil(log2(max(2,padded spatial count))); exact metadata/nonfinite masks/warnings/errors',
        public_parity_test='tests/test_butterworth_public.py',
        component_evidence='Two runtime profiles, normal/scalar, 175 checks per mode; not whole-product qualification',
        performance_evidence=None, fully_native=False)


def native_functions():
    return tuple(f'{module}.{name}' for module, names in NATIVE_FUNCTIONS.items() for name in names)


RANK_CANONICAL_FUNCTIONS=tuple('filters.rank.'+name for name in NATIVE_FUNCTIONS['filters.rank'])
RANK_IDENTITY_ALIASES={module+'.'+name:'filters.rank.'+name
                       for module in ('filters.rank.generic','filters.rank.bilateral')
                       for name in NATIVE_FUNCTIONS[module]}


def rank_contract(api):
    """Execution facts and explicit limits, separate from measured evidence."""
    canonical=RANK_IDENTITY_ALIASES.get(api,api)
    if canonical not in RANK_CANONICAL_FUNCTIONS:raise KeyError(api)
    name=canonical.rsplit('.',1)[1]
    decisions=('otsu','majority')
    if name in decisions:
        return dict(version='rank-decisions-v1',canonical=canonical,
            identity_alias=api in RANK_IDENTITY_ALIASES,registered=api in native_functions(),
            implementation='cpp-rolling-histogram-runtime-simd',operation=29+decisions.index(name),
            capability=0x60000000,interface='rank_decisions_interface',
            arithmetic='Ascending occupied-bin Otsu with original strict floating comparisons, or lowest-bin majority tie; empty populations emit zero',
            providers='Own C++ histogram kernels; explicit dtype/allocation/index providers preserve captured buffers and signed-char conversion order; no upstream algorithm forwarding',
            workers='Sole process-global native worker limit, default one; optional parallel workers',
            buffers='Source-qualified disjoint 2D/3D prepared buffers and captured-bin-safe stores; unsupported overlap is rejected before writes',
            protocol_scope='Frozen502 cases/32 provider traces/256 independent oracles and qualified endian-provider supplements; unrepresented protocols remain unassessed',
            public_parity_test='tests/test_rank_decisions_public.py',
            frozen_contract='coverage/contracts/rank-decisions.json',
            public_contract='coverage/contracts/rank-decisions-public.json',
            parity_evidence=None,performance_evidence=None)
    bilateral=('mean_bilateral','pop_bilateral','sum_bilateral')
    if name in bilateral:
        return dict(version='rank-bilateral-v1',canonical=canonical,
            identity_alias=api in RANK_IDENTITY_ALIASES,registered=api in native_functions(),
            implementation='cpp-rolling-histogram-runtime-simd',operation=26+bilateral.index(name),
            capability=0x1c000000,interface='rank_bilateral_interface',
            arithmetic='Strict g-s1<i<g+s0 occupied-bin count/weighted sums, signed integer division and raw unsigned/bool stores; no reassociation',
            providers='Own C++ histogram/conversion foundations; NumPy metadata and structural output operations. Buffers/char shifts precede s0/s1 index hooks; no upstream image algorithm forwarding',
            workers='Sole process-global native worker limit, default one; optional parallel workers, serial certified feedback',
            buffers='Source-qualified 2D prepared buffers, captured-bin-safe radii, permanent masked centers and certified same-word feedback; unsafe scopes explicitly rejected',
            protocol_scope='Frozen708/96 plus680 zero-population and396 masked-center observations; unrepresented mutation/feedback scopes remain unassessed',
            public_parity_test='tests/test_rank_bilateral_public.py',
            frozen_contract='coverage/contracts/rank-bilateral.json',
            public_contract='coverage/contracts/rank-bilateral-public.json',
            masked_center_public_test='tests/test_rank_bilateral_masked_center_public.py',
            parity_evidence=None,performance_evidence=None)
    percentiles=('autolevel_percentile','enhance_contrast_percentile','gradient_percentile','mean_percentile',
                 'pop_percentile','subtract_mean_percentile','threshold_percentile','sum_percentile','percentile')
    if name in percentiles:
        return dict(version='rank-percentile-v1',canonical=canonical,identity_alias=False,
            registered=api in native_functions(),implementation='cpp-rolling-histogram-runtime-simd',
            operation=17+percentiles.index(name),capability=0x3fe0000,interface='rank_percentile_interface',
            arithmetic='Original strict/inclusive whole-bin CDF cutoffs, signed integer sums/division and exact half-residual expression; no interpolated quantiles or reassociation',
            providers='Own C++ kernels and conversion foundations; explicit NumPy metadata/allocation providers. Buffer/signed-char shifts precede PyFloat conversion; no upstream rank algorithm forwarding',
            workers='Sole process-global native worker limit, default one; optional parallel workers; NumPy provider pools external',
            simd='Typed rolling histogram updates and four-block AVX2 integer count/weighted-prefix transport; scalar-equivalent arithmetic',
            buffers='2D inputs, disjoint prepared buffers and qualified output strides; output reshape/error stages retained. Unsupported or unsafe scopes rejected before stores',
            protocol_scope='Frozen1544 cases,54parameter traces and explicit callback-error provenance guards; unrepresented mutation protocols remain unassessed',
            public_parity_test='tests/test_rank_percentile_public.py',frozen_contract='coverage/contracts/rank-percentile.json',
            public_contract='coverage/contracts/rank-percentile-public.json',parity_evidence=None,performance_evidence=None)
    statistics=('geometric_mean','noise_filter','subtract_mean','windowed_histogram')
    if name in statistics:
        vector=name=='windowed_histogram'
        return dict(version='rank-vector-v1' if vector else 'rank-statistics-v1',canonical=canonical,
            identity_alias=api in RANK_IDENTITY_ALIASES,registered=api in native_functions(),
            implementation='cpp-rolling-histogram-runtime-simd',operation=13+statistics.index(name),
            capability=0x10000 if vector else 0xe000,
            interface='rank_vector_interface' if vector else 'rank_statistics_interface',
            arithmetic={
                'geometric_mean':'Ascending occupied-bin count*log(bin+1), exp(weighted/pop)-1 and original integer rounding before all output stores',
                'noise_filter':'Nearest occupied lower/upper bins with original missing-side bounds; copied-footprint clearing preserves the original3D x/y center quirk',
                'subtract_mean':'Original floating weighted mean, half residual and mid-bin bias in the original expression order; undefined unsigned stores excluded',
                'windowed_histogram':'Count times reciprocal population, independent prepared histogram capacity and positive output channel depth; output dtype rounding retained',
            }[name],
            providers='Own C++ kernels; explicit NumPy preparation/allocation/index providers and literal source-ordered protocol metadata, without upstream rank algorithm forwarding',
            workers='Sole process-global native worker limit, default one; optional parallel workers; NumPy provider pools external',
            simd='Existing rolling histogram/extrema/sum transport and typed AVX2 vector-channel multiplication with scalar-compatible arithmetic order',
            buffers='Qualified disjoint prepared read/output domains; overlapping or unsafe scalar/vector storage rejected before writes',
            protocol_scope='Frozen995 cases plus explicit noise shape/copy/shift and vector premax/depth provider checks; unrepresented protocols remain unassessed',
            public_parity_test='tests/test_rank_statistics_public.py',
            frozen_contract='coverage/contracts/rank-statistics.json',
            public_contract='coverage/contracts/rank-statistics-public.json',
            parity_evidence=None,performance_evidence=None)
    if name in ('autolevel','equalize','enhance_contrast','threshold'):
        arithmetic={
            'autolevel':'Signed64 (bins-1)*(center-min)/(max-min), integer truncation toward zero before all output stores; unsigned narrowing is modular, floating stores retain the signed quotient',
            'equalize':'Inclusive ascending CDF; integer (bins-1)*count before double division by population',
            'enhance_contrast':'Local extrema; lower value wins equal-distance ties',
            'threshold':'Strict center > integer weighted sum / double population; outputs zero or one',
        }
        return dict(version='rank-histogram-v3',canonical=canonical,
            identity_alias=api in RANK_IDENTITY_ALIASES,registered=api in native_functions(),
            implementation='cpp-rolling-histogram-runtime-simd',
            operation=9+('autolevel','equalize','enhance_contrast','threshold').index(name),
            arithmetic=arithmetic[name],
            providers='Own C++ uint8/uint16 histogram kernels. NumPy owns preparation, dtype/error/warning semantics, structural output operations and staged stateful floating conversions; no upstream rank image algorithm forwarding',
            workers='one actual process-global C++ worker policy, default one; optional parallel workers, small workloads remain serial; NumPy/BLAS provider pools controlled separately',
            simd='Runtime AVX2 and portable loops; equalize has int64 blocked-prefix SIMD, other operations reuse extrema/uniform-run or sum SIMD paths',
            buffers='Qualified disjoint prepared image/mask/output read-write domains; overlapping prepared buffers remain excluded. Public R2 alias admission is unchanged; private alias capability480 does not admit R3 overlap',
            unsafe_buffers='Undersized masks/outputs and zero-depth outputs safely rejected; upstream undefined memory behavior excluded from parity',
            protocol_scope='Sealed ordinary ndarray/type/layout/metadata and defined rejection contracts; arbitrary unrepresented array/subclass protocols and callback interleavings are not newly claimed',
            capability=8191,public_parity_test='tests/test_rank_intensity_public.py',
            frozen_contract='coverage/contracts/rank-intensity.json',
            public_contract='coverage/contracts/rank-intensity-public.json',
            parity_evidence=None,performance_evidence=None)
    r2=canonical.rsplit('.',1)[1] in ('gradient','median','modal','entropy')
    return dict(version='rank-histogram-v2',canonical=canonical,
        identity_alias=api in RANK_IDENTITY_ALIASES,registered=api in native_functions(),
        implementation='cpp-rolling-histogram-runtime-simd',
        providers='NumPy astype after integer downcast warning hooks, signed maximum stage, owned dtype stages for stateful floating policies and reference-bearing comparisons; qualified R2 feedback uses own C++ histogram, historical plain uint8 compatibility fallback only when alias interface or operation bit is absent',
        workers='one actual process-global C++ worker policy, default one; optional four',
        buffers='Qualified rank-alias-v1 R2 feedback executes serially on the shared native core. Full-bin uint16 mixed-byte domains and zero invariants are explicit; unrepresented/unsafe domains remain pending and raise NotImplementedError before writes. Historical plain uint8 compatibility fallback remains only when native alias interface/bit is absent' if r2 else 'Defined safe v1 output aliases use serial upstream traversal order; unsafe buffer extents rejected',
        unsafe_buffers='Undersized masks/outputs and zero-depth outputs safely rejected; upstream undefined memory behavior excluded from parity',
        capability=511,public_parity_test='tests/test_rank_public.py',
        frozen_contract='coverage/contracts/rank-distribution.json' if r2 else 'coverage/contracts/rank-neighborhoods.json',
        parity_evidence=None,performance_evidence=None,
        **(dict(alias_interface='rank-alias-v1',alias_capability=1<<(5+('gradient','median','modal','entropy').index(name)),
                alias_public_test='tests/test_rank_alias_public.py',alias_frozen_contract='coverage/contracts/rank-aliases.json',
                alias_scope='Qualified accessed read/store domains only; active integral entropy, uint16/float feedback, and partial-bin uint16 mixed-word feedback excluded; arbitrary unrepresented protocols remain pending',
                alias_performance_evidence=None) if r2 else {}))


# Classes have state, constructors and members; they are never image-call rows.
# Populate only after the isolated public class contract is validated. The
# implementation/provider facts below describe candidates, not passing evidence.
NATIVE_CLASSES = {'filters': ('LPIFilter2D',), 'transform': ('ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform')}
CLASS_METADATA_PROVIDERS = {'util': ('FailedEstimationAccessError',)}
_CLASS_BASE_MEMBERS = ('__init__','__call__','__array__','__add__','__repr__','__nice__',
                       '__str__','identity','inverse','estimate','from_estimate',
                       'residuals','dimensionality')
_CLASS_EXTRA_MEMBERS = {
    'ProjectiveTransform': (),
    'AffineTransform': ('rotation','translation','scale','shear'),
    'EuclideanTransform': ('rotation','translation'),
    'SimilarityTransform': ('rotation','translation','scale'),
}
_CLASS_MEMBER_EXECUTION = {
    '__init__':'python-state-and-small-matrix-provider',
    '__call__':'cpp-simd-coordinates-or-numpy-policy-provider',
    '__array__':'numpy-view-or-dtype-cast',
    '__add__':'numpy-small-matrix-and-python-state',
    '__repr__':'python-metadata','__nice__':'python-metadata','__str__':'python-metadata',
    'identity':'numpy-matrix-and-python-state','inverse':'numpy-inverse-and-python-state',
    'estimate':'numpy-fit-provider-and-state-mutation',
    'from_estimate':'numpy-fit-provider-and-state-construction',
    'residuals':'cpp-coordinate-call-and-numpy-reduction',
    'dimensionality':'python-state-metadata','rotation':'python-numpy-small-matrix',
    'translation':'numpy-live-view','scale':'python-numpy-small-matrix',
    'shear':'python-small-matrix',
}


def native_classes():
    """Reviewed public class entry points, separate from image functions."""
    return tuple(f'{module}.{name}' for module,names in NATIVE_CLASSES.items() for name in names)


def class_metadata_providers():
    """Compatibility class aliases; these are not native class/image kernels."""
    return tuple(f'{module}.{name}' for module,names in CLASS_METADATA_PROVIDERS.items() for name in names)


def class_contract(api):
    """Class implementation facts; evidence paths do not prove execution/speed."""
    module,separator,name=api.rpartition('.')
    if api=='util.FailedEstimationAccessError':
        return dict(version='failed-estimation-exception-v1',registered=api in class_metadata_providers(),
            implementation='exception-metadata-provider',
            members={member:'python-attributeerror-inheritance' for member in ('__init__','__new__','__str__','__repr__','__reduce__','__setstate__','args','name','obj','add_note','with_traceback')},
            state=dict(args='inherited AttributeError constructor and pickle policy',attributes='standard exception name/obj and user attributes'),
            providers=dict(exception='Canonical skimage.util/skimage._shared exception metadata when installed; own AttributeError subclass when absent. No upstream image or transform class calls.'),
            workers=dict(exception='Python metadata; no C++ workers or BLAS execution'),
            compatibility_boundaries=['Canonical alias and native failure helpers share identity independent of algorithm fallback policy; unported compatible models retain the same catch type.'],
            public_parity_test='tests/test_transform_exception_public.py',frozen_contract=None,
            parity_evidence=None,performance_evidence=None)
    if api == 'filters.LPIFilter2D':
        return dict(version='lpi-filter-class-v1', registered=api in native_classes(),
            implementation='native-class-partial',
            members={'__init__':'owned-python-state', '__call__':'owned-cpp-fft-with-numpy-providers'},
            state=dict(cache='Public mutable complex transfer array; shape-only reuse; inverse/Wiener mutate near-zero cache entries',
                       parameters='Owned impulse callback and filter_params; source callback coordinates and kwargs'),
            providers=dict(fft='Owned C++ scalar/runtime AVX2 PocketFFT, real-input full-spectrum and complex transforms',
                           arrays='NumPy coordinate grids, casting, padding, products, reciprocal/gain/Wiener arithmetic, permutation, cropping and absolute value',
                           callback='User impulse_response and duck predefined_filter providers; no upstream LPI or Python FFT calls'),
            workers=dict(fft='Existing process-global worker snapshot, default1; optional OpenMP; portable builds refuse parallel workers'),
            compatibility_boundaries=['Frozen 80 source specifications on minimum and modern runtimes; extended FFT precision, other storage/protocol partitions and hostile callback mutation remain unassessed.'],
            public_parity_test='tests/test_lpi_filter_public.py', frozen_contract='tests/lpi_filter_cases.py',
            parity_evidence=None, performance_evidence=None)
    if module!='transform' or not separator or name not in _CLASS_EXTRA_MEMBERS:
        raise KeyError(api)
    members=_CLASS_BASE_MEMBERS+_CLASS_EXTRA_MEMBERS[name]
    return dict(version='geometric-class-v1',registered=api in native_classes(),
        implementation='native-class-partial',
        members={member:_CLASS_MEMBER_EXECUTION[member] for member in members},
        state=dict(params='mutable ndarray; constructor and __array__ preserve documented aliases',
                   user_attributes='supported',inverse='independent snapshot',
                   estimation='replaces params; preserves original failure/state behavior'),
        providers=dict(coordinates='C++ scalar/runtime AVX2 for default-policy real float64 work; NumPy for exceptional/dtype/protocol policies',
                       fitting='Own NumPy SVD/normalization/Umeyama; no upstream transform class calls',
                       matrix='NumPy inverse/composition/determinants; SciPy 3D Euler construction',
                       metadata='Own class/failure objects; pinned inspect.signature display shim'),
        workers=dict(coordinates='shared C++ process-global limit, default one; optional parallel workers',
                     math_providers='NumPy/SciPy thread pools are external; constrain BLAS/OMP separately'),
        compatibility_boundaries=[
            'FailedEstimationAccessError uses the canonical public util exception metadata provider; own subclass only when scikit-image is unavailable.',
            'FailedEstimation pickle round-trip preserves the pinned upstream RecursionError; valid fitted models pickle normally.',
            'Pinned return-annotation display mentions upstream spelling; executable annotations/failure objects are native.'],
        public_parity_test='tests/test_transform_classes_public.py',
        frozen_contract='tests/test_transform_class_contracts.py',
        parity_evidence=None,performance_evidence=None)


def native_dtypes(api):
    """Dtypes covered by the default native per-function parity/benchmark case."""
    # Only a descriptor-carrier projection: level sets consume image.shape,
    # never image pixels or an image dtype. This is not dtype coverage.
    if api in LEVEL_SET_FUNCTIONS:return ('uint8',)
    if api in ROLLING_KERNEL_FUNCTIONS:return ('float64',)
    if api == 'measure.approximate_polygon':return ('float64',)
    # These are actual prepared coordinate dtypes, not image-pixel parity.
    if api == 'util.label_points':return ('uint8','float32','float64')
    if api == 'util.map_array':return ('uint8',)
    if api == 'measure.moments_normalized':return ('float32','float64')
    if api in ('morphology.remove_small_objects','morphology.remove_small_holes','segmentation.relabel_sequential'):
        return ('uint8',)
    if api.startswith('filters.rank.'):
        # Shared descriptor smoke tests use these three sample constructors;
        # the frozen rank corpus separately covers full integer/half widths.
        return ('uint8','float32','float64')
    if api in ('feature.hessian_matrix_eigvals','feature.structure_tensor_eigenvalues'):return ('float32','float64')
    if api in ('filters.threshold_otsu', 'exposure.histogram',
               'exposure.cumulative_distribution', 'exposure.equalize_hist'):
        return ('uint8',)
    return ('uint8', 'float32', 'float64')


def execution_kind(api):
    """Keep constant dtype metadata separate from C++ image kernels in reports."""
    if api in native_classes():return 'stateful-class'
    return 'metadata' if api in ('util.dtype_limits','color.xyz_tristimulus_values','morphology.mirror_footprint') else 'native'


def needs_color_input(api):
    """Default parity/benchmark sample layout, reviewed alongside the registry."""
    return api.startswith('color.') and api not in ('color.label2rgb','color.gray2rgb','color.gray2rgba','color.xyz_tristimulus_values')


def default_call(api, image):
    """Arguments for the shared default parity/performance workload."""
    if api == 'measure.approximate_polygon':
        import numpy as np
        # Actual coordinate values: row position and the first image column.
        return (np.column_stack((np.arange(image.shape[0],dtype=np.float64),image[:,0])), .5), {}
    if api == 'restoration.ball_kernel':
        return ((min(image.shape)-1)/2, image.ndim), {}
    if api == 'restoration.ellipsoid_kernel':
        return (image.shape, 2.), {}
    if api == 'util.montage':
        # One real grayscale image batch; the dedicated benchmark owns tiling workloads.
        import numpy as np
        return (image[np.newaxis, ...],), {}
    if api == 'util.regular_seeds':
        # Geometry/output-dtype adapter: pixels are not constructor arguments.
        return (image.shape,min(64,max(1,image.size//16))), {'dtype':image.dtype}
    if api == 'util.label_points':
        import numpy as np
        # 2D geometry adapter: all dtypes share the top-left 256x256 window.
        # Coordinates stay 0..255, avoiding uint8 wrapping; output shape is full.
        coords=np.argwhere(image[:256,:256]).astype(image.dtype)
        return (coords,image.shape), {}
    if api == 'util.map_array':
        import numpy as np
        keys=np.arange(256,dtype=np.uint8)
        values=np.arange(256,dtype=np.uint16)*np.uint16(3)+np.uint16(1)
        return (image,keys,values), {}
    if api in LEVEL_SET_FUNCTIONS:
        return (image.shape,), {}
    if api == 'util.compare_images':
        return (image, image[::-1]), {}
    # Perimeter and Euler consume binary data; other API fixtures stay unchanged.
    if api in ('measure.perimeter', 'measure.euler_number'):
        binary = (image > (127 if image.dtype.kind == 'u' else .5)).astype(image.dtype)
        return (binary,), {}
    if api == 'measure.points_in_poly':
        import numpy as np
        points = np.empty((image.shape[0], 2), dtype=image.dtype)
        points.flat[:] = [image.flat[i % image.size] % 7 for i in range(points.size)]
        verts = np.array([[1, 1], [1, 5], [5, 5], [5, 1]], dtype=image.dtype)
        return (points, verts), {}
    if api == 'measure.grid_points_in_poly':
        import numpy as np
        rows, columns = image.shape
        verts = np.array([[0, 0], [0, max(1, columns-1)],
                          [max(1, rows-1), max(1, columns-1)], [max(1, rows-1), 0]], dtype=np.float64)
        return (image.shape, verts), {}
    if api == 'measure.profile_line':
        return (image, (0, 0), (image.shape[0]-1, image.shape[1]-1)), {}
    if api == 'measure.mesh_surface_area':
        import numpy as np
        verts = np.empty((3, 3), dtype=image.dtype)
        verts.flat[:] = [image.flat[i % image.size] for i in range(9)]
        return (verts, np.array([[0, 1, 2]], dtype=np.intp)), {}
    if api in COORDS_MOMENTS_FUNCTIONS:
        import numpy as np
        coords = np.empty((image.shape[0], 2), dtype=image.dtype)
        # Benchmarks consume the first 2*N seeded values. The modulo also
        # keeps the existing one-pixel generic smoke fixture meaningful.
        coords.flat[:] = [image.flat[i % image.size] for i in range(coords.size)]
        return (coords,), {}
    if api in DERIVED_MOMENTS_FUNCTIONS:
        import numpy as np
        # These APIs consume moment tensors, not the fixture image. Prepare
        # one owning 4x4 tensor before either parity calls or timed calls.
        moment = np.empty((4, 4), dtype=image.dtype)
        moment.flat[:] = [image.flat[i % image.size] for i in range(16)]
        if api == 'measure.moments_normalized':
            moment[0, 0] = 4
            moment[1, 0] = moment[0, 1] = 0
        return (moment,), {}
    if api == 'filters.window':
        return ('hann', image.shape), {}
    if api in ('filters.filter_forward', 'filters.filter_inverse', 'filters.wiener'):
        import numpy as np
        def impulse_response(r, c):
            return np.exp(-(r*r + c*c) / 2)
        return (image,), {'impulse_response': impulse_response}
    if api == 'measure.intersection_coeff':
        mask = image > (127 if str(image.dtype) == 'uint8' else .5)
        return (mask, mask[::-1]), {}
    if api == 'measure.manders_coloc_coeff':
        return (image, image > (127 if str(image.dtype) == 'uint8' else .5)), {}
    if api == 'measure.manders_overlap_coeff':
        return (image, image[::-1]), {}
    if api == 'morphology.ellipse':
        return (max(1, image.shape[-1]//16), max(1, image.shape[0]//16)), {'dtype': image.dtype}
    if api == 'morphology.octagon':
        side = max(1, max(image.shape)//32)
        return (side, side), {'dtype': image.dtype}
    if api == 'morphology.octahedron':
        return (max(1, max(image.shape)//64),), {'dtype': image.dtype}
    if api == 'morphology.star':
        return (max(1, max(image.shape)//32),), {'dtype': image.dtype}
    if api == 'morphology.footprint_from_sequence':
        import numpy as np
        return (((np.ones((3,)*image.ndim, dtype=image.dtype), max(1, max(image.shape)//32)),),), {}
    if api=='morphology.reconstruction':return (image,image),{}
    if api in ('morphology.h_maxima','morphology.h_minima'):return (image,1),{}
    if api=='morphology.remove_small_objects':
        # Random uint8 IDs have mixed counts around this threshold, so both
        # retained and removed IDs are represented without input conversion.
        return (image,),{'max_size':max(1,image.size//256)}
    if api=='morphology.remove_small_holes':return (image,),{'max_size':4}
    if api=='segmentation.clear_border':return (image,),{'buffer_size':1 if min(image.shape)>2 else 0}
    if api=='segmentation.relabel_sequential':return (image,),{'offset':3}
    if api.startswith('filters.rank.'):
        import numpy as np
        # Keep the shared smoke workload bounded; the dedicated vector corpus
        # separately tests default and wide channel depths.
        options={'n_bins':16} if api.rsplit('.',1)[1]=='windowed_histogram' else {}
        return (image,np.ones((3,)*image.ndim,np.uint8)),options
    if api.startswith('metrics.'):
        return (image,image[::-1]),{}
    if api in ('measure.label','morphology.label'):
        return ((image>(127 if str(image.dtype)=='uint8' else .5)).astype(image.dtype),),{}
    if api=='morphology.footprint_rectangle':return (image.shape,),{'dtype':image.dtype}
    if api in ('morphology.diamond','morphology.disk','morphology.ball'):
        return (max(1,max(image.shape)//16),),{'dtype':image.dtype}
    if api in ('feature.hessian_matrix_eigvals','feature.structure_tensor_eigenvalues'):
        return ([image,image*.2,image*.3],),{}
    if api=='filters.gabor':return (image,.2),{}
    if api=='filters.gabor_kernel':
        import numpy as np
        return (.2,),{'dtype':np.complex64 if image.dtype==np.float32 else np.complex128}
    if api=='exposure.match_histograms':return (image,image[...,::2]),{}
    if api=='transform.downscale_local_mean':return (image,2),{}
    if api=='transform.matrix_transform':
        import numpy as np
        return (np.column_stack((image.ravel(),image.ravel())),np.eye(3)),{}
    if api=='transform.resize':return (image,tuple(max(1,(n+1)//2) for n in image.shape)),{}
    if api=='transform.rescale':return (image,.5),{}
    if api=='transform.integrate':
        return (image,tuple(n//4 for n in image.shape),tuple(3*n//4 for n in image.shape)), {}
    if api=='filters.difference_of_gaussians':return (image,1.), {}
    if api=='filters.correlate_sparse':
        import numpy as np
        kernel=np.zeros((3,)*image.ndim);kernel.flat[0]=.25;kernel.flat[-1]=1
        return (image,kernel),{}
    if api=='filters.apply_hysteresis_threshold':return (image,63 if str(image.dtype)=='uint8' else .25,191 if str(image.dtype)=='uint8' else .75),{}
    if api=='color.rgba2rgb':
        import numpy as np
        return (np.concatenate((image,image[...,:1]),axis=-1),), {}
    if api == 'color.xyz_tristimulus_values':
        return (), {'illuminant':'D65','observer':'2','dtype':image.dtype}
    if api.startswith('color.deltaE_'):
        return (image,image[...,::-1]), {}
    if api == 'color.convert_colorspace':
        return (image,'RGB','HSV'), {}
    if api.startswith('morphology.binary_'):
        # Avoid timing the trivial all-foreground case for numeric inputs.
        return ((image > (127 if str(image.dtype)=='uint8' else .5)).astype(image.dtype),), {}
    if api in ('color.separate_stains','color.combine_stains'):
        from ._stain_matrices import MATRICES
        matrix=MATRICES['hed_from_rgb' if api.endswith('separate_stains') else 'rgb_from_hed']
        return (image,matrix), {}
    return (image,), {}


def moments_default_error(api, image, actual, expected):
    """Validate the frozen cell-specific default policy and return finite error."""
    if api in COORDS_MOMENTS_FUNCTIONS:
        from ._moments_coords_checks import default_error
        return default_error(api, image, actual, expected)
    if api == 'measure.inertia_tensor_eigvals':
        from ._inertia_checks import eigvals_default_error
        return eigvals_default_error(image, actual, expected)
    if api == 'measure.inertia_tensor':
        from ._inertia_checks import default_error
        return default_error(image, actual, expected)
    if api in DERIVED_MOMENTS_FUNCTIONS:
        from ._moments_derived_checks import derived_default_error
        return derived_default_error(api, image, expected, actual)
    from ._moments_checks import default_error
    return default_error(api, image, actual, expected)


def shannon_default_error(image, actual, expected):
    """Validate the frozen entropy-only policy after default base conversion."""
    from ._shannon_entropy_checks import default_error
    return default_error(image, actual, expected)


def blur_default_error(image, actual, expected):
    """Validate exact default Blur scalar words, including NaN words."""
    from ._blur_effect_checks import default_error
    return default_error(image, actual, expected)


def approximate_polygon_default_error(image, actual, expected):
    """Exact selected coordinate words, metadata, layout and fresh output."""
    import numpy as np
    for value in (actual, expected):
        assert type(value) is np.ndarray and value.dtype == np.dtype('float64')
        assert value.ndim == 2 and value.shape[1] == 2
        assert value.flags.owndata and value.flags.writeable and value.flags.aligned
        assert value.flags.c_contiguous and value.base is None
        assert not np.shares_memory(value, image)
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual is not expected and actual.shape == expected.shape
    assert (actual.strides, actual.flags.f_contiguous) == (expected.strides, expected.flags.f_contiguous)
    assert actual.tobytes() == expected.tobytes()
    return 0.


def rolling_kernel_default_error(api, shape, actual, expected):
    """Exact constructor geometry, Inf mask, storage and float64 words."""
    import numpy as np
    radius=(min(shape)-1)/2
    desired=((2*int(np.ceil(radius))+1,)*len(shape) if api=='restoration.ball_kernel'
             else tuple(2*max(dimension//2,1)+1 for dimension in shape))
    assert api in ROLLING_KERNEL_FUNCTIONS
    for value in (actual,expected):
        assert type(value) is np.ndarray and value.dtype==np.dtype('float64')
        assert value.dtype.metadata is None and value.shape==desired
    descriptor=lambda value:(value.strides,value.flags.owndata,value.flags.writeable,
                             value.flags.c_contiguous,value.flags.f_contiguous,type(value.base))
    assert descriptor(actual)==descriptor(expected)
    assert np.array_equal(np.isposinf(actual),np.isposinf(expected))
    assert actual.tobytes()==expected.tobytes()
    return 0.


def montage_default_error(image, actual, expected):
    """Exact single-image batch dtype, layout, fresh storage and logical words."""
    import numpy as np
    assert type(actual) is type(expected) is np.ndarray
    for value in (actual, expected):
        assert value.shape == image.shape and value.dtype == image.dtype
        assert value.dtype.metadata == image.dtype.metadata
        assert not np.shares_memory(value, image)
    descriptor = lambda value: (value.strides, value.flags.owndata, value.flags.writeable,
                                value.flags.c_contiguous, value.flags.f_contiguous, type(value.base))
    assert descriptor(actual) == descriptor(expected)
    assert actual.tobytes() == expected.tobytes()
    return 0.


def regular_seeds_default_error(shape, dtype, actual, expected):
    """Exact constructor shape, requested output dtype, storage and label words."""
    import numpy as np
    assert type(actual) is type(expected) is np.ndarray
    for value in (actual,expected):
        assert value.dtype==np.dtype(dtype) and value.dtype.metadata==np.dtype(dtype).metadata
        assert value.shape==tuple(shape)
    descriptor=lambda value:(value.strides,value.flags.owndata,value.flags.writeable,
                             value.flags.c_contiguous,value.flags.f_contiguous,type(value.base))
    assert descriptor(actual)==descriptor(expected)
    assert actual.tobytes()==expected.tobytes()
    return 0.


def unique_rows_default_error(image, actual, expected):
    """Exact byte-row output metadata/words; final NumPy gather stays literal."""
    import numpy as np
    assert type(actual) is type(expected) is np.ndarray
    for value in (actual,expected):
        assert value.dtype==image.dtype and value.dtype.metadata==image.dtype.metadata
        assert value.ndim==2 and value.shape[1]==image.shape[1]
    assert actual.shape==expected.shape and actual.shape[0]<=image.shape[0]
    descriptor=lambda value:(value.strides,value.flags.owndata,value.flags.writeable,
                             value.flags.c_contiguous,value.flags.f_contiguous,type(value.base))
    assert descriptor(actual)==descriptor(expected)
    assert actual.tobytes()==expected.tobytes()
    return 0.


def label_points_default_error(shape, actual, expected):
    """Exact uint64 label-mask words/storage for the actual output-shape arg."""
    import numpy as np
    assert type(actual) is type(expected) is np.ndarray
    for value in (actual,expected):
        assert value.dtype==np.dtype(np.uint64) and value.dtype.metadata is None
        assert value.shape==tuple(shape)
    descriptor=lambda value:(value.strides,value.flags.owndata,value.flags.writeable,
                             value.flags.c_contiguous,value.flags.f_contiguous,type(value.base))
    assert descriptor(actual)==descriptor(expected)
    assert actual.tobytes()==expected.tobytes()
    return 0.


def map_array_default_error(image, actual, expected):
    """Exact mapped output bytes/storage; default input is real uint8 pixels."""
    import numpy as np
    assert type(actual) is type(expected) is np.ndarray
    assert actual.shape==expected.shape==image.shape
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    descriptor=lambda value:(value.strides,value.flags.owndata,value.flags.writeable,
                             value.flags.c_contiguous,value.flags.f_contiguous,type(value.base))
    assert descriptor(actual)==descriptor(expected)
    assert actual.tobytes()==expected.tobytes()
    return 0.


def compare_default_error(image, actual, expected):
    """Validate exact default image-comparison metadata and words."""
    from ._compare_images_checks import default_error
    return default_error(image, actual, expected)


def level_set_default_error(shape, actual, expected):
    """Exact int8 words/storage for shape-only constructors, no input dtype."""
    import numpy as np
    if hasattr(shape,'shape'):shape=shape.shape
    assert type(actual) is type(expected)
    for value in (actual,expected):
        assert value.dtype==np.dtype(np.int8) and value.shape==tuple(shape)
    if isinstance(expected,np.ndarray):
        descriptor=lambda value:(value.strides,value.flags.owndata,value.flags.writeable,
                                 value.flags.c_contiguous,value.flags.f_contiguous,type(value.base))
        assert descriptor(actual)==descriptor(expected)
    assert actual.tobytes()==expected.tobytes()
    return 0.


def profile_default_error(image, actual, expected):
    """Validate exact profile output metadata and words without a tolerance."""
    from ._profile_line_checks import default_error
    return default_error(image, actual, expected)


def polygon_default_error(image, actual, expected):
    """Compare exact classifications and source raw bool label words."""
    from ._polygon_checks import default_error
    return default_error(image, actual, expected)
def mesh_default_error(call_input, actual, expected):
    """Validate the frozen mesh geometry envelope from the actual (verts,faces)."""
    from ._mesh_surface_area_checks import default_error
    return default_error(call_input, actual, expected)


def euler_default_error(image, actual, expected):
    """Validate the frozen exact histogram/LUT integer domain."""
    from ._euler_number_checks import default_error
    return default_error(image, actual, expected)


def perimeter_default_error(image, actual, expected):
    """Validate the frozen fifty-term perimeter weighted-sum policy."""
    from ._perimeter_checks import default_error
    return default_error(image, actual, expected)


def crofton_default_error(image, actual, expected):
    """Validate the frozen sixteen-term Crofton weighted-sum policy."""
    from ._perimeter_crofton_checks import default_error
    return default_error(image, actual, expected)


def block_default_error(image, actual, expected):
    """Validate the frozen per-block default-sum policy."""
    from ._block_reduce_checks import default_error
    return default_error(image, actual, expected)


def assert_default_parity(api, dtype, actual, expected, *, image=None):
    """Shared assertion; previous scalar budgets keep their existing checker."""
    if api == 'measure.approximate_polygon':
        approximate_polygon_default_error(image,actual,expected)
        return
    if api in ROLLING_KERNEL_FUNCTIONS:
        rolling_kernel_default_error(api,image.shape,actual,expected)
        return
    if api == 'util.montage':
        montage_default_error(image,actual,expected)
        return
    if api == 'util.regular_seeds':
        regular_seeds_default_error(image.shape,image.dtype,actual,expected)
        return
    if api == 'util.unique_rows':
        unique_rows_default_error(image,actual,expected)
        return
    if api == 'util.label_points':
        label_points_default_error(image.shape,actual,expected)
        return
    if api == 'util.map_array':
        map_array_default_error(image,actual,expected)
        return
    if api in LEVEL_SET_FUNCTIONS:
        level_set_default_error(image,actual,expected)
        return
    if api == 'util.compare_images':
        compare_default_error(image, actual, expected)
        return
    if api == 'measure.blur_effect':
        blur_default_error(image, actual, expected)
        return
    if api == 'measure.profile_line':
        profile_default_error(image, actual, expected)
        return
    if api in ('measure.points_in_poly', 'measure.grid_points_in_poly'):
        polygon_default_error(image, actual, expected)
        return
    if api == 'measure.mesh_surface_area':
        mesh_default_error(image, actual, expected)
        return
    if api == 'measure.euler_number':
        euler_default_error(image, actual, expected)
        return
    if api == 'measure.perimeter':
        perimeter_default_error(image, actual, expected)
        return
    if api == 'measure.perimeter_crofton':
        crofton_default_error(image, actual, expected)
        return
    if api == 'measure.block_reduce':
        block_default_error(image, actual, expected)
        return
    if api == 'measure.shannon_entropy':
        shannon_default_error(image, actual, expected)
        return
    if api in COORDS_MOMENTS_FUNCTIONS or api in MOMENTS_FUNCTIONS or api in DERIVED_MOMENTS_FUNCTIONS or api == 'measure.inertia_tensor' or api == 'measure.inertia_tensor_eigvals':
        moments_default_error(api, image, actual, expected)
        return
    import numpy as np
    rtol, atol = parity_tolerance(api, dtype, image=image, expected=expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def parity_tolerance(api, dtype, image=None, expected=None):
    """Floating budgets; Fourier filters use their frozen input/reference policies."""
    if api == 'measure.approximate_polygon':
        raise ValueError('Approximate polygon uses exact selected coordinate words and fresh layout; call assert_default_parity')
    if api in ROLLING_KERNEL_FUNCTIONS:
        raise ValueError('Rolling kernel constructors use exact Inf masks and words; call assert_default_parity')
    if api == 'util.montage':
        raise ValueError('Montage uses exact dtype, layout and words; call assert_default_parity')
    if api == 'util.regular_seeds':
        raise ValueError('Regular seeds use exact requested dtype, metadata and words; call assert_default_parity')
    if api == 'util.unique_rows':
        raise ValueError('Unique rows use exact metadata and words; call assert_default_parity')
    if api == 'util.label_points':
        raise ValueError('Label points use exact uint64 metadata and words; call assert_default_parity')
    if api == 'util.map_array':
        raise ValueError('Map arrays use exact metadata and words; call assert_default_parity')
    if api in LEVEL_SET_FUNCTIONS:
        raise ValueError('Level sets use exact int8 metadata and words; call assert_default_parity')
    if api == 'util.compare_images':
        raise ValueError('Comparisons use exact metadata and words; call assert_default_parity')
    if api == 'measure.blur_effect':
        raise ValueError('Blur uses exact scalar words; call assert_default_parity')
    if api == 'measure.profile_line':
        raise ValueError('Profiles use exact metadata and words; call assert_default_parity')
    if api in ('measure.points_in_poly', 'measure.grid_points_in_poly'):
        raise ValueError('Polygon classification uses exact bytes; call assert_default_parity')
    if api == 'measure.mesh_surface_area':
        raise ValueError('Mesh uses its geometric envelope; call assert_default_parity')
    if api == 'measure.euler_number':
        raise ValueError('Euler uses its exact integer domain; call assert_default_parity')
    if api == 'measure.perimeter':
        raise ValueError('Perimeter uses its histogram-specific bound; call assert_default_parity')
    if api == 'measure.perimeter_crofton':
        raise ValueError('Crofton uses its histogram-specific bound; call assert_default_parity')
    if api == 'measure.block_reduce':
        raise ValueError('Block sum uses per-block bounds; call assert_default_parity')
    if api == 'measure.shannon_entropy':
        raise ValueError('Shannon entropy uses its reference-specific bound; call assert_default_parity')
    if api in COORDS_MOMENTS_FUNCTIONS or api in MOMENTS_FUNCTIONS or api in DERIVED_MOMENTS_FUNCTIONS or api == 'measure.inertia_tensor' or api == 'measure.inertia_tensor_eigvals':
        raise ValueError('Moments use cell-specific bounds; call assert_default_parity')
    if api == 'filters.window':
        if expected is None:
            raise ValueError('Window tolerance requires the actual reference output')
        import numpy as np
        finite = np.abs(expected[np.isfinite(expected)])
        peak = max(1.0, float(finite.max()) if finite.size else 0.0)
        eps = np.finfo(np.float64).eps
        return 8 * eps, 32 * eps * peak
    if api in ('filters.filter_forward', 'filters.filter_inverse', 'filters.wiener'):
        if image is None or expected is None:
            raise ValueError('LPI tolerance requires the actual input and reference output')
        import numpy as np
        # Same prospective formula as the frozen public LPI contracts. This
        # scales with the reference output, never the candidate or its error.
        count = 1
        for size in image.shape:
            count *= 2 * int(size) - 1
        levels = max(1, (max(2, count) - 1).bit_length())
        eps = np.finfo(expected.real.dtype).eps
        peak = 0.0
        for value in (image, expected):
            finite = np.abs(value[np.isfinite(value)])
            if finite.size:
                peak = max(peak, float(finite.max()))
        return 32 * eps * levels, 64 * eps * levels * peak
    if api == 'filters.butterworth':
        if image is None:
            raise ValueError('Butterworth tolerance requires the actual default input image')
        import numpy as np
        # Frozen before candidate output: defaults have no padding/channel axis.
        # Full keyword contracts retain their independent padded-spatial policy.
        levels = max(1, (max(2, int(image.size)) - 1).bit_length())
        input_dtype = image.dtype
        single = (input_dtype.kind == 'f' and input_dtype.itemsize in (2, 4)
                  or input_dtype.kind == 'c' and input_dtype.itemsize == 8)
        eps = np.finfo(np.float32 if single else np.float64).eps
        finite = np.abs(image[np.isfinite(image)])
        peak = float(finite.max()) if finite.size else 0.0
        return 8 * eps * levels, 32 * eps * levels * peak
    if api in ('measure.intersection_coeff', 'measure.manders_coloc_coeff', 'measure.manders_overlap_coeff'):
        return (0., 0.)
    if api in ('morphology.ellipse', 'morphology.octagon', 'morphology.octahedron',
               'morphology.star', 'morphology.footprint_from_sequence'):
        return (0., 0.)
    if api in ('morphology.h_maxima','morphology.h_minima'):return (0.,0.)
    if api=='metrics.normalized_mutual_information':
        # Prospective entropy/ratio policy; exact counts, metadata and events
        # have separate assertions in the frozen214-case public contract.
        return (2e-12,2e-13)
    if api=='restoration.denoise_tv_bregman' or api in MAX_TREE_PATHS:return (0.,0.)
    if api in ('morphology.remove_small_objects','morphology.remove_small_holes','segmentation.clear_border','segmentation.relabel_sequential'):
        return (0.,0.)
    if api.startswith('filters.rank.'):return (0.,0.)
    if api.startswith('metrics.'):
        tolerance=3e-6 if str(dtype)=='float32' else 3e-13
        return tolerance,tolerance if api=='metrics.peak_signal_noise_ratio' else 0.
    if api=='transform.matrix_transform':return (0.,0.)
    if api in ('color.label2rgb','transform.integral_image','transform.integrate','filters.threshold_minimum','filters.threshold_multiotsu','filters.rank_order','filters.apply_hysteresis_threshold'):return (0.,0.)
    name = str(dtype)
    single = name == 'float32'
    if api=='color.deltaE_ciede2000' and single:
        # atan2 provider differences of a few float32 ULPs are amplified by
        # chroma in the hue term, especially when the final distance is small.
        # Float64 retains the base budget. Float32 also has independent
        # float64-reference accuracy checks on ordinary Lab colors.
        return (2e-6,2.5e-4)
    if api in ('color.xyz2lab','color.rgb2lab','color.xyz2luv','color.rgb2luv'):
        # Four epsilon units amplified by the 500*a scaling are about 2.4e-4.
        return (2e-6,2.5e-4) if single else (2e-13,2e-12)
    return (2e-6,2e-6) if single else (2e-13,2e-13)
