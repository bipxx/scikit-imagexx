"""Bounded owned region geometry with explicit original descriptor providers.

Distinct backend subclass; full class arithmetic and literal type identity are
not claimed. Original scaffolding attribution: third_party/scikit-image-LICENSE.txt.
"""
import numpy as np
from . import _core


def __getattr__(name):
    if name in ('RegionProperties', '_Provider', '_cached', '_provider_regionprops'):
        from . import _regionprops_state_class
        value = getattr(_regionprops_state_class, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def regionprops(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None):
    # Resolve only missing lazy aliases; warm calls keep ordinary module globals.
    # Patched aliases and deletion remain visible without a repeated self-import.
    namespace = globals()
    if 'RegionProperties' not in namespace:
        __getattr__('RegionProperties')
    if '_provider_regionprops' not in namespace:
        __getattr__('_provider_regionprops')
    # Explicit bounded route. Unknown protocols/intensity/callback semantics are
    # still literal providers, never credited to the returned-state native leaf.
    if type(label_image) is not np.ndarray or intensity_image is not None or extra_properties is not None:
        return _provider_regionprops(label_image, intensity_image, cache, extra_properties=extra_properties, spacing=spacing, offset=offset)
    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')
    if not np.issubdtype(label_image.dtype, np.integer):
        if np.issubdtype(label_image.dtype, bool):
            raise TypeError('Non-integer image types are ambiguous: use skimage.measure.label to label the connected components of label_image, or label_image.astype(np.uint8) to interpret the True values as a single label.')
        raise TypeError('Non-integer label_image types are ambiguous')
    if offset is None:
        offset_arr = np.zeros((label_image.ndim,), dtype=int)
    else:
        offset_arr = np.asarray(offset)
        if offset_arr.ndim != 1 or offset_arr.size != label_image.ndim:
            raise ValueError('Offset should be an array-like of integers of shape (label_image.ndim,); ' + f'{offset} was provided.')
    if label_image.size == 0:
        # Both pinned SciPy find_objects implementations perform this reduction
        # before their native traversal. Retain NumPy's exact empty-array error.
        label_image.max()
    native_result = _core.regionprops_state_prepare(label_image)
    if native_result is None:
        return _provider_regionprops(label_image, intensity_image, cache, extra_properties=extra_properties, spacing=spacing, offset=offset)
    prepared, geometry = native_result
    result = []
    for index, (label, starts, stops) in enumerate(geometry):
        slices = tuple(slice(a, b) for a, b in zip(starts, stops))
        region = RegionProperties(slices, label, label_image, None, cache, spacing=spacing, offset=offset_arr)
        region._native_state = prepared
        region._native_index = index
        region._native_initial_label = region.label
        region._native_initial_slice = region.slice
        result.append(region)
    return result
