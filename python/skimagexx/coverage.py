"""Reviewed native dispatch entry points. Partial coverage is not full API parity."""
NATIVE_FUNCTIONS = {
    'transform': ('integral_image',),
    'morphology': ('erosion','dilation','opening','closing','white_tophat','black_tophat',
                   'binary_erosion','binary_dilation','binary_opening','binary_closing'),
    'exposure': ('adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
                 'histogram', 'cumulative_distribution', 'equalize_hist'),
    'color': ('rgba2rgb','gray2rgba','rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
              'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
              'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb',
              'rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
              'lab2lch','lch2lab','xyz_tristimulus_values','xyz2luv','luv2xyz','rgb2luv','luv2rgb',
              'separate_stains','combine_stains','rgb2hed','hed2rgb','deltaE_cie76','convert_colorspace',
              'deltaE_ciede94','deltaE_ciede2000','deltaE_cmc'),
    'filters': ('gaussian', 'difference_of_gaussians', 'unsharp_mask', 'sobel', 'sobel_h', 'sobel_v', 'prewitt', 'prewitt_h',
                'prewitt_v', 'scharr', 'scharr_h', 'scharr_v', 'threshold_mean', 'threshold_otsu',
                'threshold_yen', 'threshold_triangle', 'threshold_isodata',
                'threshold_niblack','threshold_sauvola','farid','farid_h','farid_v','laplace','roberts','roberts_pos_diag','roberts_neg_diag'),
    'util': ('invert', 'img_as_float', 'img_as_float32', 'img_as_float64', 'img_as_ubyte',
             'img_as_uint', 'img_as_int', 'img_as_bool', 'dtype_limits'),
}


def native_functions():
    return tuple(f'{module}.{name}' for module, names in NATIVE_FUNCTIONS.items() for name in names)


def native_dtypes(api):
    """Dtypes covered by the default native per-function parity/benchmark case."""
    if api in ('filters.threshold_otsu', 'exposure.histogram',
               'exposure.cumulative_distribution', 'exposure.equalize_hist'):
        return ('uint8',)
    return ('uint8', 'float32', 'float64')


def execution_kind(api):
    """Keep constant dtype metadata separate from C++ image kernels in reports."""
    return 'metadata' if api in ('util.dtype_limits','color.xyz_tristimulus_values') else 'native'


def needs_color_input(api):
    """Default parity/benchmark sample layout, reviewed alongside the registry."""
    return api.startswith('color.') and api not in ('color.gray2rgb','color.gray2rgba','color.xyz_tristimulus_values')


def default_call(api, image):
    """Arguments for the shared default parity/performance workload."""
    if api=='filters.difference_of_gaussians':return (image,1.), {}
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
    if api=='transform.integral_image':return (0.,0.)
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
