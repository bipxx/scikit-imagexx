"""Owned colocalization array stages with literal 0.26 observable ordering.

This private module is not public registration. Object/complex/mixed arithmetic,
unsupported floating reduction iterators and exceptional floating state are
explicit NumPy primitive providers, not native completion. The original three
colocalization functions are never called. The process worker policy is shared
with the sole installed/private core; this module does not create a worker pool.
"""
import warnings
from types import BuiltinFunctionType

import numpy as np

from . import _core


_PROFILES = {"1.24.4": 1, "2.5.4": 2}
_PROFILE = _PROFILES.get(np.__version__, 0)
_BOOL = np.dtype(bool)
_FLOATS = (np.dtype(np.float32), np.dtype(np.float64))
_ASARRAY = np.asarray if isinstance(np.asarray, BuiltinFunctionType) else None
_GETERR = np.geterr
_GETBUFSIZE = np.getbufsize
_GETERRCALL = np.geterrcall
_SUM = np.sum
_SQUARE = np.square
_MULTIPLY = np.multiply
_COUNT_NONZERO = np.count_nonzero
_LOGICAL_AND = np.logical_and
_FLAGS = ((1, "divide", "divide by zero"), (2, "over", "overflow"),
          (4, "under", "underflow"), (8, "invalid", "invalid value"))


class _Context:
    def __init__(self):
        self.guards = _core.ColocalizationGuards()
        self.profile = _PROFILE

    def check(self):
        self.guards.check()

    def retain(self, array):
        self.guards.add(array)
        return array

    def array(self, value):
        if type(value) is np.ndarray and np.asarray is _ASARRAY:
            return self.retain(value)
        # Reject a forged existing ndarray BEFORE NumPy gets a chance to read
        # it. For a foreign conversion protocol, check every prior live owner
        # after the callback and capture the returned array before any read.
        if isinstance(value, np.ndarray):
            self.retain(value)
        self.check()
        try:
            result = np.asarray(value)
        finally:
            self.check()
        return self.retain(result)

    def provider(self, function, *args):
        self.check()
        try:
            result = function(*args)
        finally:
            self.check()
        if isinstance(result, np.ndarray):
            return self.retain(result)
        return result

    def policy(self):
        policy, handler = np.geterr(), np.geterrcall()
        self.check()
        # print/log use NumPy's original C FILE*/first-error dispatch via the
        # literal primitive. Unknown versions and these policies are uncredited
        # provider frontiers; no homemade approximation of their behavior.
        admitted = self.profile and all(value in ("ignore", "warn", "raise", "call")
                                        for value in policy.values())
        return (policy, handler) if admitted else None

    def dispatch(self, flags, operation, snapshot):
        policy, handler = snapshot
        # Primary 1.24.4/2.5.4 HANDLEIT order and combined status argument.
        # The snapshot stays fixed across callbacks from THIS array operation.
        for flag, key, label in _FLAGS:
            if not flags & flag:
                continue
            method = policy[key]
            message = f"{label} encountered in {operation}"
            try:
                if method == "warn":
                    warnings.warn(message, RuntimeWarning, stacklevel=3)
                elif method == "raise":
                    raise FloatingPointError(message)
                elif method == "call":
                    if handler is None:
                        raise NameError(f"python callback specified for {label} (in  {operation}) but no function found.")
                    handler(label, flags)
            finally:
                # A warning hook/error callback can change root allocation,
                # shape, dtype, or policy. No next pixel stage precedes this.
                self.check()

    def binary(self, value, variable_name):
        array = self.array(value)
        if array.dtype == _BOOL:
            # Preserve noncanonical bool bytes: their truth is byte != 0.
            return array
        self.check()
        native = _core.coloc_binary(self.guards, array) if self.profile else None
        if native is None:
            # Keep the two comparisons and bitwise AND in source order.
            unequal_one = self.provider(lambda a: a != 1, array)
            unequal_zero = self.provider(lambda a: a != 0, array)
            invalid = self.provider(np.bitwise_and, unequal_one, unequal_zero)
            valid = not self.provider(np.any, invalid)
            if valid:
                return self.provider(lambda a: np.asarray(a, dtype=bool), array)
        else:
            valid, converted = native
            if valid:
                return self.retain(converted)
        raise ValueError(f"{variable_name} array is not of dtype boolean or contains values other than 0 and 1 so cannot be safely cast to boolean array.")

    def gather(self, array, mask):
        self.check()
        native = _core.coloc_gather(self.guards, array, mask) if self.profile else None
        return self.provider(lambda a, m: a[m], array, mask) if native is None else self.retain(native)

    def minimum_negative(self, array):
        self.check()
        native = _core.coloc_min_negative(self.guards, array) if self.profile else None
        if native is None:
            return self.provider(lambda a: a.min() < 0, array)
        return native

    def sum(self, array):
        snapshot = self.policy()
        native = None
        if snapshot is not None:
            # This is the current source-stage buffer size, not an imported or
            # cached value from before a user callback.
            buffer_elements = np.getbufsize()
            self.check()
            native = _core.coloc_sum(self.guards, array, self.profile, buffer_elements)
        if native is None:
            return self.provider(np.sum, array)
        value, status = native
        self.dispatch(status, "reduce", snapshot)
        return value

    def product(self, first, second=None):
        snapshot = self.policy()
        native = None
        if snapshot is not None:
            self.check()
            native = _core.coloc_product(self.guards, first, second)
        if native is None:
            return (self.provider(np.square, first) if second is None else
                    self.provider(np.multiply, first, second))
        value, status = native
        self.retain(value)
        self.dispatch(status, "square" if second is None else "multiply", snapshot)
        return value

    def sum_product(self, first, second=None):
        if first.size >= 128 and self.policy() is not None:
            buffer_elements = np.getbufsize()
            result = _core.coloc_product_sum(self.guards, first, second, self.profile, buffer_elements)
            if result is not None:
                return result
        return self.sum(self.product(first, second))

    def count(self, array):
        self.check()
        if not self.profile:
            return self.provider(np.count_nonzero, array)
        value = _core.coloc_count(self.guards, array)
        # With axis=None and no keepdims, the exact audited NumPy2 count route
        # returns np.intp; 1.24 returns Python int. The ensuing ratio therefore
        # has different public scalar ownership/type even with equal f64 bits.
        result = np.intp(value) if self.profile == 2 else value
        self.check()
        return result

    def joint_count(self, first, second):
        self.check()
        if not self.profile:
            return self.provider(np.count_nonzero,
                                 self.provider(np.logical_and, first, second))
        value = _core.coloc_joint_count(self.guards, first, second)
        result = np.intp(value) if self.profile == 2 else value
        self.check()
        return result


def _shape(*arrays):
    first = arrays[0]
    if not all(first.shape == array.shape for array in arrays[1:]):
        raise ValueError("Input images must have the same dimensions.")


def intersection_coeff(image0_mask, image1_mask, mask=None):
    if _PROFILE and np.asarray is _ASARRAY:
        counts = None
        if mask is None and np.count_nonzero is _COUNT_NONZERO and np.logical_and is _LOGICAL_AND:
            counts = _core.coloc_contiguous_intersection(image0_mask, image1_mask)
        if (counts is None and type(image0_mask) is np.ndarray and image0_mask.size >= 128
                and type(image1_mask) is np.ndarray and (mask is None or type(mask) is np.ndarray)):
            counts = _core.coloc_dense_intersection(image0_mask, image1_mask, mask)
        if counts is not None:
            total, joint = counts
            if total == 0:
                return 0
            return np.intp(joint) / np.intp(total) if _PROFILE == 2 else joint / total
    context = _Context()
    image0_mask = context.binary(image0_mask, "image0_mask")
    image1_mask = context.binary(image1_mask, "image1_mask")
    if mask is not None:
        mask = context.binary(mask, "mask")
        _shape(image0_mask, image1_mask, mask)
        image0_mask = context.gather(image0_mask, mask)
        image1_mask = context.gather(image1_mask, mask)
    else:
        _shape(image0_mask, image1_mask)
    nonzero_image0 = context.count(image0_mask)
    if nonzero_image0 == 0:
        return 0
    nonzero_joint = context.joint_count(image0_mask, image1_mask)
    return context.provider(lambda a, b: a / b, nonzero_joint, nonzero_image0)


def _dense_manders(image0, image1, mask, overlap):
    if (not _PROFILE or np.asarray is not _ASARRAY or np.geterr is not _GETERR
            or np.getbufsize is not _GETBUFSIZE or np.geterrcall is not _GETERRCALL
            or np.sum is not _SUM or np.square is not _SQUARE or np.multiply is not _MULTIPLY):
        return None
    if mask is None and type(image0) is np.ndarray and type(image1) is np.ndarray:
        result = _core.coloc_contiguous_manders(image0, image1, _PROFILE, np.getbufsize(), overlap)
        # False means the array layout or dtype needs the guarded route.
        # None requests source-ordered replay after speculative arithmetic.
        if result is not False:
            return result
    if (type(image0) is not np.ndarray or type(image1) is not np.ndarray
            or image0.dtype not in _FLOATS
            or image1.dtype != (image0.dtype if overlap else _BOOL)
            or image0.shape != image1.shape
            or (mask is not None and (type(mask) is not np.ndarray
                                      or mask.dtype != _BOOL or mask.shape != image0.shape))):
        return None
    if not all(method in ("ignore", "warn", "raise", "call") for method in np.geterr().values()):
        return None
    native = (_core.coloc_dense_manders_overlap if overlap else _core.coloc_dense_manders_coloc)
    values = native(image0, image1, mask, _PROFILE, np.getbufsize())
    if values is None:
        return None
    if overlap:
        square0, square1, numerator = values
        return numerator / ((square0 * square1) ** 0.5)
    total, joint = values
    return 0 if total == 0 else joint / total


def manders_coloc_coeff(image0, image1_mask, mask=None):
    dense = _dense_manders(image0, image1_mask, mask, False)
    if dense is not None:
        return dense
    context = _Context()
    image0 = context.array(image0)
    image1_mask = context.binary(image1_mask, "image1_mask")
    if mask is not None:
        mask = context.binary(mask, "mask")
        _shape(image0, image1_mask, mask)
        image0 = context.gather(image0, mask)
        image1_mask = context.gather(image1_mask, mask)
    else:
        _shape(image0, image1_mask)
    if context.minimum_negative(image0):
        raise ValueError("image contains negative values")
    total = context.sum(image0)
    if total == 0:
        return 0
    joint = context.sum_product(image0, image1_mask)
    return context.provider(lambda a, b: a / b, joint, total)


def manders_overlap_coeff(image0, image1, mask=None):
    dense = _dense_manders(image0, image1, mask, True)
    if dense is not None:
        return dense
    context = _Context()
    image0 = context.array(image0)
    image1 = context.array(image1)
    if mask is not None:
        mask = context.binary(mask, "mask")
        _shape(image0, image1, mask)
        image0 = context.gather(image0, mask)
        image1 = context.gather(image1, mask)
    else:
        _shape(image0, image1)
    if context.minimum_negative(image0):
        raise ValueError("image0 contains negative values")
    if context.minimum_negative(image1):
        raise ValueError("image1 contains negative values")
    square0 = context.sum_product(image0)
    square1 = context.sum_product(image1)
    product = context.provider(lambda a, b: a * b, square0, square1)
    denominator = context.provider(lambda a: a ** 0.5, product)
    numerator = context.sum_product(image0, image1)
    return context.provider(lambda a, b: a / b, numerator, denominator)
