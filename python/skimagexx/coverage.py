"""Reviewed native dispatch entry points. Partial coverage is not full API parity."""
NATIVE_FUNCTIONS = {
    'restoration': ('denoise_tv_chambolle', 'denoise_tv_bregman'),
    'segmentation': ('clear_border','relabel_sequential'),
    'filters.rank': ('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy','autolevel','equalize','enhance_contrast','threshold','geometric_mean','noise_filter','subtract_mean','windowed_histogram','autolevel_percentile','enhance_contrast_percentile','gradient_percentile','mean_percentile','pop_percentile','subtract_mean_percentile','threshold_percentile','sum_percentile','percentile','mean_bilateral','pop_bilateral','sum_bilateral','otsu','majority'),
    'filters.rank.bilateral': ('mean_bilateral','pop_bilateral','sum_bilateral'),
    'filters.rank.generic': ('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy','autolevel','equalize','enhance_contrast','threshold','geometric_mean','noise_filter','subtract_mean','windowed_histogram','otsu','majority'),
    'measure': ('label',),
    'metrics': ('mean_squared_error','normalized_root_mse','peak_signal_noise_ratio','normalized_mutual_information'),
    'feature': ('hessian_matrix_eigvals','structure_tensor_eigenvalues','structure_tensor','corner_harris','corner_shi_tomasi','corner_foerstner','corner_kitchen_rosenfeld','hessian_matrix','shape_index','hessian_matrix_det'),
    'transform': ('integral_image','integrate','downscale_local_mean','resize','rescale','matrix_transform'),
    'morphology': ('label','remove_small_objects','remove_small_holes','footprint_rectangle','diamond','disk','ball','mirror_footprint','pad_footprint',
                   'erosion','dilation','opening','closing','white_tophat','black_tophat',
                   'binary_erosion','binary_dilation','binary_opening','binary_closing',
                   'max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing','reconstruction'),
    'exposure': ('adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
                 'histogram', 'cumulative_distribution', 'equalize_hist', 'equalize_adapthist','is_low_contrast','match_histograms'),
    'color': ('label2rgb','rgba2rgb','gray2rgba','rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
              'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
              'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb',
              'rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
              'lab2lch','lch2lab','xyz_tristimulus_values','xyz2luv','luv2xyz','rgb2luv','luv2rgb',
              'separate_stains','combine_stains','rgb2hed','hed2rgb','deltaE_cie76','convert_colorspace',
              'deltaE_ciede94','deltaE_ciede2000','deltaE_cmc'),
    'filters': ('frangi','hessian','meijering','sato','gabor','gabor_kernel','gaussian', 'difference_of_gaussians', 'unsharp_mask', 'sobel', 'sobel_h', 'sobel_v', 'prewitt', 'prewitt_h',
                'prewitt_v', 'scharr', 'scharr_h', 'scharr_v', 'threshold_mean', 'threshold_otsu',
                'threshold_yen', 'threshold_triangle', 'threshold_isodata', 'threshold_minimum', 'threshold_li', 'threshold_multiotsu',
                'threshold_niblack','threshold_sauvola','threshold_local','rank_order','apply_hysteresis_threshold','correlate_sparse','median','farid','farid_h','farid_v','laplace','roberts','roberts_pos_diag','roberts_neg_diag'),
    'util': ('invert', 'img_as_float', 'img_as_float32', 'img_as_float64', 'img_as_ubyte',
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
NATIVE_CLASSES = {'transform': ('ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform')}
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
    if api=='morphology.reconstruction':return (image,image),{}
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


def parity_tolerance(api, dtype):
    """Floating error budgets; CIE scaling amplifies cbrt/matrix roundoff."""
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
