"""Bounded owned region geometry with explicit original descriptor providers.

Distinct backend subclass; full class arithmetic and literal type identity are
not claimed. Original scaffolding attribution: third_party/scikit-image-LICENSE.txt.
"""
import numpy as np
from skimage.measure._regionprops import RegionProperties as _Provider
from skimage.measure._regionprops import _cached
from skimage.measure._regionprops import regionprops as _provider_regionprops
from . import _core


class RegionProperties(_Provider):
    @property
    @_cached
    def image(self):
        prepared = self.__dict__.get('_native_state')
        if prepared is None:
            return _Provider.image.fget(self)
        output = np.empty_like(self._label_image[self.slice], dtype=bool, order='K', subok=False)
        native_result = _core.regionprops_state_image(prepared, self._native_index, self._label_image, output)
        if native_result is None:
            return _Provider.image.fget(self)
        return output

    @property
    @_cached
    def num_pixels(self):
        mask = self.image
        native_result = _core.regionprops_state_population(mask)
        if native_result is None:
            return np.sum(mask)
        return np.intp(native_result)

    @property
    def coords(self):
        prepared = self.__dict__.get('_native_state')
        if prepared is None:
            return _Provider.coords.fget(self)
        mask = self.image
        count = _core.regionprops_state_population(mask)
        if count is None:
            return _Provider.coords.fget(self)
        output = np.empty((count, self._ndim), dtype=np.intp, order='F')
        native_result = _core.regionprops_state_coords(prepared, self._native_index, self._label_image, mask, output)
        if native_result is None:
            return _Provider.coords.fget(self)
        # Literal provider scalar-promotion/ufunc tail; native emitted global
        # integer coordinates, the exact first addition in the original source.
        return output + self._offset


def regionprops(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None):
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
        result.append(region)
    return result
