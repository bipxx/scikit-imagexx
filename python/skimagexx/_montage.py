"""Montage with literal preparation and owned prepared-buffer fill/copy."""
import numpy as np
from skimage._shared import utils
from . import _core

_INDEX_MAX = np.iinfo(np.intp).max
_PADDING_TYPES = (int, bool, np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)


@utils.channel_as_last_axis(multichannel_output=False)
def montage(arr_in, fill='mean', rescale_intensity=False, grid_shape=None,
            padding_width=0, *, channel_axis=None):
    # scikit-image0.26 _montage.py, BSD-3-Clause; see the retained upstream
    # license in third_party/scikit-image-LICENSE.txt. Preserve provider order:
    # rescaling can mutate the caller before a later fill or copy failure.
    if channel_axis is not None:
        arr_in = np.asarray(arr_in)
    else:
        arr_in = np.asarray(arr_in)[..., np.newaxis]
    if arr_in.ndim != 4:
        raise ValueError(
            'Input array has to be 3-dimensional for grayscale '
            'images, or 4-dimensional with a `channel_axis` '
            'specified.'
        )
    n_images, n_rows, n_cols, n_chan = arr_in.shape
    if grid_shape:
        ntiles_row, ntiles_col = (int(s) for s in grid_shape)
    else:
        ntiles_row = ntiles_col = int(np.ceil(np.sqrt(n_images)))
    if rescale_intensity:
        from skimage import exposure
        for i in range(n_images):
            arr_in[i] = exposure.rescale_intensity(arr_in[i])
    if fill == 'mean':
        fill = arr_in.mean(axis=(0, 1, 2))
    fill = np.atleast_1d(fill).astype(arr_in.dtype)
    n_pad = padding_width
    arr_out = np.empty(
        ((n_rows + n_pad) * ntiles_row + n_pad,
         (n_cols + n_pad) * ntiles_col + n_pad, n_chan), dtype=arr_in.dtype)
    native_result = False
    # User arithmetic protocols have already run in the literal allocation.
    # Refuse their tail here so slice construction/events remain literal too.
    if (type(n_pad) in _PADDING_TYPES
            and 0 <= n_pad <= _INDEX_MAX
            and 0 < ntiles_row <= _INDEX_MAX
            and 0 < ntiles_col <= _INDEX_MAX):
        native_result = _core.montage_apply(
            arr_in, fill, arr_out, (ntiles_row, ntiles_col, int(n_pad)),
            _core.get_num_threads())
    if not native_result:
        for idx_chan in range(n_chan):
            arr_out[..., idx_chan] = fill[idx_chan]
        slices_row = [
            slice(n_pad + (n_rows + n_pad) * n,
                  n_pad + (n_rows + n_pad) * n + n_rows)
            for n in range(ntiles_row)]
        slices_col = [
            slice(n_pad + (n_cols + n_pad) * n,
                  n_pad + (n_cols + n_pad) * n + n_cols)
            for n in range(ntiles_col)]
        for idx_image, image in enumerate(arr_in):
            idx_sr, idx_sc = divmod(idx_image, ntiles_col)
            arr_out[slices_row[idx_sr], slices_col[idx_sc], :] = image
    if channel_axis is not None:
        return arr_out
    return arr_out[..., 0]
