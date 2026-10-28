"""Owned finite image comparisons with literal conversion and protocol stages."""
import functools
from itertools import product
import numpy as np
from . import _core


def _rename_image_params(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, (value, param) in enumerate(
            zip(args, ["image0", "image1", "method", "n_tiles"])
        ):
            if param in kwargs:
                raise ValueError(
                    f"{param} passed both as positional and keyword argument."
                )
            else:
                kwargs[param] = value
        args = tuple()
        return func(*args, **kwargs)
    return wrapper


@_rename_image_params
def compare_images(image0, image1, *, method='diff', n_tiles=(8, 8)):
    if image1.shape != image0.shape:
        raise ValueError('Images must have the same shape.')
    from skimage.util.dtype import img_as_float
    img1 = img_as_float(image0)
    img2 = img_as_float(image1)
    native_comparison = None
    if method == 'diff':
        selected = 0
    elif method == 'blend':
        selected = 1
    elif method == 'checkerboard':
        selected = 2
        if img1.ndim != 2:
            raise ValueError(
                'Images must be 2-dimensional to be compared with the '
                'checkerboard method.'
            )
        shapex, shapey = img1.shape
        stepx = int(shapex / n_tiles[0])
        stepy = int(shapey / n_tiles[1])
        # Evaluate both index protocols once, in the source product's order.
        row_range, column_range = range(n_tiles[0]), range(n_tiles[1])
    else:
        raise ValueError(
            'Wrong value for `method`. '
            'Must be either "diff", "blend" or "checkerboard".'
        )
    if (type(img1) is np.ndarray and type(img2) is np.ndarray and img1.size
            and img1.dtype.char in 'fd' and img2.dtype.char in 'fd'
            and not (np.__version__ == '1.24.4' and img1.ndim == 0
                     and img1.dtype.char == img2.dtype.char == 'f' and selected == 1)):
        first = img1 if (img1.flags.c_contiguous and img1.dtype.isnative
                         and img1.flags.aligned and img1.flags.owndata) else np.require(
            img1, dtype=img1.dtype.newbyteorder('='), requirements=('C', 'A', 'O'))
        second = img2 if (img2.flags.c_contiguous and img2.dtype.isnative
                          and img2.flags.aligned and img2.flags.owndata) else np.require(
            img2, dtype=img2.dtype.newbyteorder('='), requirements=('C', 'A', 'O'))
        direct_output = (img1.flags.c_contiguous and img2.flags.c_contiguous
                         and img1.dtype.isnative and img2.dtype.isnative
                         and img1.dtype.metadata is None and img2.dtype.metadata is None)
        if selected == 2:
            if not direct_output:
                comparison = np.zeros_like(img1)
            if stepx > 0 and stepy > 0 and row_range.stop > 0 and column_range.stop > 0:
                grid = (shapex, shapey, stepx, stepy, row_range.stop, column_range.stop)
            else:
                grid = (shapex, shapey, 0, 0, 0, 0)
        else:
            if not direct_output:
                # NumPy's iterator allocates the ufunc's K-order layout without
                # evaluating subtraction, addition, absolute value or scaling.
                if selected == 0:
                    dtype = np.subtract.resolve_dtypes((img2.dtype, img1.dtype, None))[2]
                    dtype = np.absolute.resolve_dtypes((dtype, None))[1]
                else:
                    dtype = np.add.resolve_dtypes((img2.dtype, img1.dtype, None))[2]
                    dtype = np.multiply.resolve_dtypes((float, dtype, None))[2]
                with np.nditer([img2, img1, None], flags=['zerosize_ok'],
                               op_flags=[['readonly'], ['readonly'], ['writeonly', 'allocate']],
                               op_dtypes=[None, None, dtype], order='K') as iterator:
                    comparison = iterator.operands[2]
            grid = (0, 0, 0, 0, 0, 0)
        native_comparison = _core.compare_images(first, second, selected, grid, _core.get_num_threads())
        if native_comparison is not None:
            if direct_output:
                return native_comparison[()] if native_comparison.ndim == 0 else native_comparison
            comparison[...] = native_comparison
            return comparison[()] if comparison.ndim == 0 else comparison
    if selected == 0:
        comparison = np.abs(img2 - img1)
    elif selected == 1:
        comparison = 0.5 * (img2 + img1)
    else:
        mask = np.full((shapex, shapey), False)
        for i, j in product(row_range, column_range):
            if (i + j) % 2 == 0:
                mask[i * stepx : (i + 1) * stepx, j * stepy : (j + 1) * stepy] = True
        comparison = np.zeros_like(img1)
        comparison[mask] = img1[mask]
        comparison[~mask] = img2[~mask]
    return comparison
