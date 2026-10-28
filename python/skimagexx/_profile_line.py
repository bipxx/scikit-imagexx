"""Source-ordered profile coordinates and reducers with owned 0/1 sampling.

NumPy coordinates, channel stacking, flipping and reducers remain providers.
Unsupported storage, higher orders and exceptional sampling use the SciPy
map_coordinates primitive, never the upstream profile_line implementation.
"""
import numpy as np
from scipy import ndimage as ndi

from . import _core


def _validate_interpolation_order(image_dtype, order):
    if order is None:
        return 0 if image_dtype == bool else 1
    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the range 0-5.")
    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined with bool "
            "data type. Please set order to 0 or explicitly cast input image "
            "to another data type."
        )
    return order


def _line_profile_coordinates(src, dst, linewidth=1):
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)
    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    if type(linewidth) is int and 0 < linewidth <= 1_000_000:
        native_coordinates = _core.profile_coordinates(
            line_row, line_col, linewidth, row_width, col_width,
            _core.get_num_threads()
        )
        if native_coordinates is not None:
            return native_coordinates
    perp_rows = np.stack([
        np.linspace(row_i - row_width, row_i + row_width, linewidth)
        for row_i in line_row
    ])
    perp_cols = np.stack([
        np.linspace(col_i - col_width, col_i + col_width, linewidth)
        for col_i in line_col
    ])
    return np.stack([perp_rows, perp_cols])


def _sample_plane(image, coordinates, order, mode, cval, workers):
    native_pixels = None
    if (type(image) is np.ndarray and image.ndim == 2
            and image.dtype.char in '?Bfd'
            and isinstance(order, (int, np.integer)) and order in (0, 1)
            and type(mode) is str
            and mode in ('reflect', 'mirror', 'nearest', 'grid-constant', 'grid-wrap')
            and isinstance(cval, (int, float, np.integer, np.floating))):
        if 0 in image.shape:
            raise ValueError('profile_line cannot sample an empty image dimension')
        # These are the actual native buffers. NumPy handles caller layouts;
        # the binding validates the prepared storage after output allocation.
        prepared = np.require(image, dtype=image.dtype.newbyteorder('='),
                              requirements=('C', 'A', 'O'))
        native_pixels = _core.profile_sample(
            prepared, coordinates, order, mode, cval, workers
        )
    if native_pixels is not None:
        return native_pixels
    return ndi.map_coordinates(image, coordinates, prefilter=order > 1,
                               order=order, mode=mode, cval=cval)


def profile_line(image, src, dst, linewidth=1, order=None, mode='reflect',
                 cval=0.0, *, reduce_func=np.mean):
    """Return a profile with owned ordinary order-0/1 plane sampling."""
    order = _validate_interpolation_order(image.dtype, order)
    mode = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}.get(mode, mode)
    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
    workers = _core.get_num_threads()
    if image.ndim == 3:
        pixels = [
            _sample_plane(image[..., i], perp_lines, order, mode, cval, workers)
            for i in range(image.shape[2])
        ]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = _sample_plane(image, perp_lines, order, mode, cval, workers)
    pixels = np.flip(pixels, axis=1)
    if reduce_func is None:
        intensities = pixels
    else:
        try:
            intensities = reduce_func(pixels, axis=1)
        except TypeError:
            intensities = np.apply_along_axis(reduce_func, arr=pixels, axis=1)
    return intensities
