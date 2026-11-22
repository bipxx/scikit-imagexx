"""Haar coordinates and values for bounded, validated integral-image windows."""

import operator
import sys

import numpy as np

from . import _core

_FAMILIES = ("type-2-x", "type-2-y", "type-3-x", "type-3-y", "type-4")
_KINDS = (_core._HaarKind.two_x, _core._HaarKind.two_y, _core._HaarKind.three_x,
          _core._HaarKind.three_y, _core._HaarKind.four)
_BLOCKS = (2, 2, 3, 3, 4)
_TILES = ((2, 1), (1, 2), (3, 1), (1, 3), (2, 2))
_EMPTY_VALUES = "need at least one array to concatenate"
_EMPTY_SELECTION = "not enough values to unpack (expected 2, got 0)"


def _selection(feature_type):
    if feature_type is None:
        return _FAMILIES
    if isinstance(feature_type, str):
        feature_type = (feature_type,)
    for name in feature_type:
        if name not in _FAMILIES:
            raise ValueError(f"The given feature type is unknown. Got {name} instead of one of {_FAMILIES}.")
    return tuple(feature_type)


def _dimensions(width, height):
    width, height = operator.index(width), operator.index(height)
    if any(n > sys.maxsize or n < -sys.maxsize - 1 for n in (width, height)):
        raise OverflowError("Python int too large to convert to C ssize_t")
    if width >= 2**31 - 1 or height >= 2**31 - 1:
        raise OverflowError("Haar image dimensions exceed native coordinate representation")
    return max(0, width), max(0, height)


def haar_like_feature_coord(width, height, feature_type=None):
    """Return owning object arrays containing Haar rectangles and feature types."""
    names = _selection(feature_type)
    if not names:
        raise ValueError(_EMPTY_SELECTION)
    width, height = _dimensions(width, height)
    return _core.haar_coordinates(width, height, [_KINDS[_FAMILIES.index(name)] for name in names], names)


def _validate_span(view):
    # Verify the actual strided read lies inside a known owning ndarray allocation.
    owner = view
    while type(owner.base) is np.ndarray:
        owner = owner.base
    if not owner.flags.owndata or not (owner.flags.c_contiguous or owner.flags.f_contiguous):
        raise ValueError("Haar integral image has an unverified backing buffer")
    if not view.size:
        return
    if view is owner:
        return
    start = view.__array_interface__["data"][0]
    base = owner.__array_interface__["data"][0]
    offsets = [(length - 1) * stride for length, stride in zip(view.shape, view.strides)]
    if start + sum(min(0, n) for n in offsets) < base or (
        start + sum(max(0, n) for n in offsets) + view.itemsize > base + owner.nbytes
    ):
        raise ValueError("Haar integral image view lies outside its backing allocation")


def _roi(int_image, r, c, width, height):
    if type(int_image) is not np.ndarray:
        raise TypeError("Haar integral image must be a plain NumPy ndarray")
    if int_image.ndim != 2:
        raise ValueError(f"Buffer has wrong number of dimensions (expected 2, got {int_image.ndim})")
    dtype = int_image.dtype
    if dtype.kind not in "buif" or (dtype.kind == "f" and dtype.itemsize not in (4, 8)):
        raise TypeError("No matching signature found")
    if not dtype.isnative:
        endian = "Big" if sys.byteorder == "little" else "Little"
        raise ValueError(f"{endian}-endian buffer not supported on {sys.byteorder}-endian compiler")
    r, c = operator.index(r), operator.index(c)
    view = int_image[r:r + height, c:c + width]
    _validate_span(view)
    return view


def _selected(coords, types):
    n = coords.shape[0]
    if n != types.shape[0]:
        raise ValueError("Inconsistent size between feature coordinatesand feature types.")
    if type(coords) is not np.ndarray or type(types) is not np.ndarray or types.ndim != 1:
        raise TypeError("Haar selection requires NumPy coordinate and one-dimensional type arrays")
    _validate_span(coords)
    _validate_span(types)
    groups = [np.flatnonzero(types == kind) for kind in _FAMILIES]
    count = sum(len(indices) for indices in groups)
    if not count:
        raise ValueError(_EMPTY_SELECTION)
    for indices in groups:
        if len(indices) and indices[-1] >= count:
            index = indices[np.searchsorted(indices, count)]
            raise IndexError(f"index {index} is out of bounds for axis 0 with size {count}")
    tokens, rows = [None] * count, [None] * count
    for family, indices in enumerate(groups):
        for index in indices:
            tokens[index] = _KINDS[family]
            boxes = coords[index]
            blocks = _BLOCKS[family]
            if len(boxes) < blocks:
                raise ValueError("Haar feature has too few rectangles")
            rectangles = []
            for i in range(blocks):
                (top, left), (bottom, right) = boxes[i]
                rectangles.append(tuple(map(operator.index, (top, left, bottom, right))))
            rows[index] = rectangles
    return tokens, rows


def haar_like_feature(int_image, r, c, width, height, feature_type=None, feature_coord=None):
    """Evaluate generated or selected Haar features on a relative integral-image ROI."""
    if feature_coord is None:
        families = [_FAMILIES.index(name) for name in _selection(feature_type)]
        if not families:
            raise ValueError(_EMPTY_VALUES)
        width, height = _dimensions(width, height)
        if not any(width >= _TILES[k][0] and height >= _TILES[k][1] for k in families):
            raise ValueError(_EMPTY_VALUES)
        view = _roi(int_image, r, c, width, height)
        if view.shape != (height, width):
            raise ValueError("Haar window lies outside the integral image ROI")
        return _core.haar_generated_values(view.copy(order="C"), [_KINDS[k] for k in families])

    width, height = _dimensions(width, height)
    kinds, rows = _selected(feature_coord, feature_type)
    view = _roi(int_image, r, c, width, height)
    return _core.haar_selected_values(view.copy(order="C"), kinds, rows)
