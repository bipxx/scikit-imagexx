"""Lazy provider subclass for the bounded returned-region state bridge."""
import numpy as np
import warnings
from skimage.measure._regionprops import RegionProperties as _Provider
from skimage.measure._regionprops import _cached
from skimage.measure._regionprops import regionprops as _provider_regionprops
from . import _core


def _extent(first,denominator):
    # Same bounded FP-policy delivery convention as existing derived moments.
    # The operands are already evaluated in source order; providers reuse them.
    policy,handler=np.geterr(),np.geterrcall()
    if any(v not in ('ignore','warn','raise','call') for v in policy.values()):
        return first/denominator
    native_result=_core.regionprops_state_extent_f64(first,denominator)
    if native_result is None:return first/denominator
    value,status=native_result
    flags=((status&1)<<3)|((status&2)>>1)|((status&4)>>1)|((status&8)>>1)
    for bit,key,label in ((1,'divide','divide by zero'),(2,'over','overflow'),(4,'under','underflow'),(8,'invalid','invalid value')):
        if not flags&bit:continue
        action=policy[key]
        if action=='warn':warnings.warn(label+' encountered in scalar divide',RuntimeWarning,stacklevel=3)
        elif action=='raise':raise FloatingPointError(label+' encountered in scalar divide')
        elif action=='call':
            if handler is None:raise NameError(f'python callback specified for {label} (in  scalar divide) but no function found.')
            handler(label,flags)
    return value


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
    @_cached
    def area(self):
        if type(self) is not RegionProperties or '_native_state' not in self.__dict__:
            return _Provider.area.fget(self)
        mask=self.image
        count=_core.regionprops_state_population(mask) if type(mask) is np.ndarray else None
        if count is None:return np.sum(mask)*self._pixel_area
        pixel_area=self._pixel_area
        native_result=_core.regionprops_state_scaled_count_f64(count,pixel_area)
        if native_result is None:return np.intp(count)*pixel_area
        value,status=native_result
        # The admitted scaled-count domain has no overflow/underflow/invalid.
        assert status==0
        return value

    @property
    def area_bbox(self):
        if type(self) is not RegionProperties or '_native_state' not in self.__dict__:
            return _Provider.area_bbox.fget(self)
        result=_core.regionprops_state_area_bbox_get(self)
        if type(result) is tuple:
            count,pixel_area=result
            return count*pixel_area
        native_result=result
        return native_result

    @property
    def extent(self):
        if type(self) is not RegionProperties or '_native_state' not in self.__dict__:
            return _Provider.extent.fget(self)
        result=_core.regionprops_state_extent_get(self)
        if type(result) is tuple:
            first,denominator=result
            return _extent(first,denominator)
        native_result=result
        return native_result

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
