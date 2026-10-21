"""Reviewed native dispatch entry points. Partial coverage is not full API parity."""
NATIVE_FUNCTIONS = {
    'morphology': ('erosion','dilation','opening','closing','white_tophat','black_tophat',
                   'binary_erosion','binary_dilation','binary_opening','binary_closing'),
    'exposure': ('adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
                 'histogram', 'cumulative_distribution', 'equalize_hist'),
    'color': ('rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
              'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
              'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb',
              'rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
              'lab2lch','lch2lab','xyz_tristimulus_values'),
    'filters': ('gaussian', 'sobel', 'sobel_h', 'sobel_v', 'prewitt', 'prewitt_h',
                'prewitt_v', 'scharr', 'scharr_h', 'scharr_v', 'threshold_mean', 'threshold_otsu'),
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
    return api.startswith('color.') and api not in ('color.gray2rgb','color.xyz_tristimulus_values')


def default_call(api, image):
    """Arguments for the shared default parity/performance workload."""
    if api == 'color.xyz_tristimulus_values':
        return (), {'illuminant':'D65','observer':'2','dtype':image.dtype}
    if api.startswith('morphology.binary_'):
        # Avoid timing the trivial all-foreground case for numeric inputs.
        return ((image > (127 if str(image.dtype)=='uint8' else .5)).astype(image.dtype),), {}
    return (image,), {}


def parity_tolerance(api, dtype):
    """Floating error budgets; CIE scaling amplifies cbrt/matrix roundoff."""
    name = str(dtype)
    single = name == 'float32'
    if api in ('color.xyz2lab','color.rgb2lab'):
        # Four epsilon units amplified by the 500*a scaling are about 2.4e-4.
        return (2e-6,2.5e-4) if single else (2e-13,2e-12)
    return (2e-6,2e-6) if single else (2e-13,2e-13)
