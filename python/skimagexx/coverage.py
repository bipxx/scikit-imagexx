"""Reviewed native dispatch entry points. Partial coverage is not full API parity."""
NATIVE_FUNCTIONS = {
    'measure': ('label',),
    'metrics': ('mean_squared_error','normalized_root_mse','peak_signal_noise_ratio'),
    'feature': ('hessian_matrix_eigvals','structure_tensor_eigenvalues','structure_tensor','corner_harris','corner_shi_tomasi','corner_foerstner','corner_kitchen_rosenfeld','hessian_matrix','shape_index','hessian_matrix_det'),
    'transform': ('integral_image','integrate','downscale_local_mean','resize','rescale'),
    'morphology': ('label','footprint_rectangle','diamond','disk','ball','mirror_footprint','pad_footprint',
                   'erosion','dilation','opening','closing','white_tophat','black_tophat',
                   'binary_erosion','binary_dilation','binary_opening','binary_closing'),
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


def native_functions():
    return tuple(f'{module}.{name}' for module, names in NATIVE_FUNCTIONS.items() for name in names)


# Classes have state, constructors and members; they are never image-call rows.
# Populate only after the isolated public class contract is validated. The
# implementation/provider facts below describe candidates, not passing evidence.
NATIVE_CLASSES = {'transform': ('ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform')}
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


def class_contract(api):
    """Class implementation facts; evidence paths do not prove execution/speed."""
    module,separator,name=api.rpartition('.')
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
            'FailedEstimationAccessError has native identity; name, AttributeError hierarchy and message parity are separate contracts.',
            'FailedEstimation pickle round-trip preserves the pinned upstream RecursionError; valid fitted models pickle normally.',
            'Pinned return-annotation display mentions upstream spelling; executable annotations/failure objects are native.'],
        public_parity_test='tests/test_transform_classes_public.py',
        frozen_contract='tests/test_transform_class_contracts.py',
        parity_evidence=None,performance_evidence=None)


def native_dtypes(api):
    """Dtypes covered by the default native per-function parity/benchmark case."""
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
    if api.startswith('metrics.'):
        tolerance=3e-6 if str(dtype)=='float32' else 3e-13
        return tolerance,tolerance if api=='metrics.peak_signal_noise_ratio' else 0.
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
