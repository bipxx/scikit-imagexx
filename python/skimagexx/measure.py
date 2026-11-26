"""Native labels, partial colocalization and explicit compatibility APIs."""
from functools import wraps
import numpy as np
from . import _component_consumers, _components
from ._colocalization import intersection_coeff, manders_coloc_coeff, manders_overlap_coeff
from ._compat import module_attribute

__all__ = ['label', 'intersection_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

_label_fallback = getattr(_components.label, '__wrapped__', _components.label)
_dense_label = getattr(_components._core, 'components_label_dense', None)
_INDEX_DTYPE = np.dtype(np.intp)
_BOOL_LABEL_DTYPE = np.dtype(np.int32)
_INDEX_LIMIT = 1 << (8 * _INDEX_DTYPE.itemsize - 1)


@wraps(_label_fallback)
def label(label_image, background=None, return_num=False, connectivity=None):
    if (_dense_label is not None and type(label_image) is np.ndarray and label_image.ndim == 2
            and label_image.shape[0] > 1 and label_image.shape[1] > 1
            and (connectivity is None or type(connectivity) is int and 1 <= connectivity <= 2)
            and (background is None or type(background) in (int, bool))
            and type(return_num) is bool):
        descriptor = label_image.dtype
        if descriptor.kind == 'b':
            key = int(background == 1)
            dtype = _BOOL_LABEL_DTYPE if label_image.size < 2**31 - 2 else _INDEX_DTYPE
        elif (descriptor.kind in 'iu' and (background is None or -_INDEX_LIMIT <= background < _INDEX_LIMIT)
              or descriptor.char in 'fd'
              and background in (None, 0) and connectivity in (None, 2)):
            key = 0 if background is None else background
            dtype = descriptor if descriptor == _INDEX_DTYPE else _INDEX_DTYPE
        else:
            return _label_fallback(label_image, background, return_num, connectivity)
        result = _dense_label(label_image, key, 2 if connectivity is None else connectivity, dtype)
        if result is not None:
            return result if return_num else result[0]
    return _label_fallback(label_image, background, return_num, connectivity)


# Publish one callable even if the consumers imported the original first.
_components.label = _component_consumers._label = label


_GEOMETRIC_MODEL_EXPORTS = ('CircleModel', 'EllipseModel', 'RansacModelProtocol', 'ransac')


def __getattr__(name):
    if name in _GEOMETRIC_MODEL_EXPORTS:
        from . import _geometric_models
        value = getattr(_geometric_models, name)
        globals()[name] = value
        return value
    return module_attribute('measure', name)


from ._moments import centroid, inertia_tensor, moments, moments_central

__all__ += ['moments', 'moments_central', 'centroid', 'inertia_tensor']

from ._moments_derived import moments_hu, moments_normalized

__all__ += ['moments_normalized', 'moments_hu']


from ._moments import inertia_tensor_eigvals

__all__ += ['inertia_tensor_eigvals']


from ._moments import moments_coords, moments_coords_central

__all__ += ['moments_coords', 'moments_coords_central']


from ._shannon_entropy import shannon_entropy

__all__ += ['shannon_entropy']


from ._block_reduce import block_reduce

__all__ += ['block_reduce']


from ._perimeter_crofton import perimeter_crofton

__all__ += ['perimeter_crofton']


from ._perimeter import perimeter

__all__ += ['perimeter']


from ._euler_number import euler_number

__all__ += ['euler_number']

from ._polygon import points_in_poly, grid_points_in_poly

__all__ += ['points_in_poly', 'grid_points_in_poly']
from ._mesh_surface_area import mesh_surface_area

__all__ += ['mesh_surface_area']

from ._profile_line import profile_line

__all__ += ['profile_line']

from ._blur_effect import blur_effect

__all__ += ['blur_effect']

from ._approximate_polygon import approximate_polygon

__all__ += ['approximate_polygon']

from ._subdivide_polygon import subdivide_polygon

__all__ += ['subdivide_polygon']

from ._regionprops_state import regionprops

__all__ += ['regionprops']

from ._pearson import pearson_corr_coeff

__all__ += ['pearson_corr_coeff']


from ._find_contours import find_contours

__all__ += ['find_contours']

from ._regionprops_table import regionprops_table

__all__ += ['regionprops_table']

from ._line_model_nd import LineModelND
from ._marching_cubes import marching_cubes

__all__ += [
    'CircleModel', 'EllipseModel', 'LineModelND', 'RansacModelProtocol',
    'marching_cubes', 'ransac',
]
