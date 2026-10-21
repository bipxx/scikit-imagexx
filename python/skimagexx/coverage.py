"""Reviewed native dispatch entry points. Partial coverage is not full API parity."""
NATIVE_FUNCTIONS = {
    'exposure': ('adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
                 'histogram', 'cumulative_distribution', 'equalize_hist'),
    'color': ('rgb2gray', 'gray2rgb'),
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
    return 'metadata' if api == 'util.dtype_limits' else 'native'
