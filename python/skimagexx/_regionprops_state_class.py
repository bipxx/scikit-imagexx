"""Lazy provider subclass for the bounded returned-region state bridge."""
import numpy as np
from skimage.measure._regionprops import RegionProperties as _Provider
from skimage.measure._regionprops import _cached
from skimage.measure._regionprops import regionprops as _provider_regionprops
from . import _core


class RegionProperties(_Provider):
    __module__ = 'skimagexx._regionprops_state'
    def _current_native_geometry(self):
        prepared = self.__dict__.get('_native_state')
        # Public writes do not invalidate upstream caches. Refuse only the
        # frozen native geometry; original property tails observe current state.
        if (prepared is not None and self.label is self._native_initial_label
                and self.slice is self._native_initial_slice):
            return prepared
        return None
    @property
    @_cached
    def image(self):
        prepared = self._current_native_geometry()
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
        prepared = self._current_native_geometry()
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
