"""Watershed input preparation and the owned ordered flood."""

import numpy as np

from . import _core
from ._compat import fallback
from ._regular_seeds import regular_seeds


_REAL_SCALARS = {int, float, bool, np.bool_} | {
    np.dtype(code).type for code in 'bBhHiIlLqQefd'
}


def _canonical(array):
    if type(array) is not np.ndarray:
        return False
    dtype = array.dtype
    return (dtype.metadata is None and dtype.isnative
            and dtype.char in '?bBhHiIlLqQefd'
            and array.flags.c_contiguous and array.flags.aligned and array.flags.writeable)


def _neighborhood(image, connectivity, offset):
    from skimage.morphology._util import _offsets_to_raveled_neighbors, _validate_connectivity

    footprint, center = _validate_connectivity(image.ndim, connectivity, offset)
    padding = tuple(int(p) for p in center)
    if len(padding) != image.ndim:
        raise ValueError('number of dimensions in image shape, footprint and its center index does not match')
    if any(p < 0 for p in padding):
        raise ValueError("index can't contain negative values")
    # The flood relies on symmetric padding. Verify the actual populated
    # footprint when its nominal extent exceeds this padding.
    if any(f > 2 * p + 1 for f, p in zip(footprint.shape, padding)):
        if any(np.any(indices > 2 * p) for indices, p in zip(np.nonzero(footprint), padding)):
            raise ValueError('watershed offset does not pad the populated footprint')
    padded_shape = tuple(axis + 2 * p for axis, p in zip(image.shape, padding))
    neighbors = _offsets_to_raveled_neighbors(padded_shape, footprint, center)
    return padding, neighbors.tolist()


def watershed(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False):
    """Flood basins from marker labels using elevation and insertion order."""
    # Preparation follows scikit-image 0.26.0 _watershed.py (BSD-3-Clause,
    # third_party/scikit-image-LICENSE.txt). Marker and footprint providers
    # supply inputs to the C++ flood.
    canonical_markers = _canonical(markers) and markers.dtype.kind in 'biu'
    scalar_connectivity = connectivity is None or type(connectivity) in _REAL_SCALARS
    plain_offset = type(offset) in (tuple, list) and all(type(p) is int for p in offset)
    array_offset = _canonical(offset) and offset.ndim == 1 and offset.dtype.kind in 'iu'
    if (not _canonical(image) or image.ndim == 0
            or (markers is not None and not canonical_markers and type(markers) not in _REAL_SCALARS)
            or (mask is not None and not _canonical(mask))
            or (not scalar_connectivity and not _canonical(connectivity))
            or (offset is not None and not plain_offset and not array_offset)
            or type(compactness) not in _REAL_SCALARS
            or type(watershed_line) not in (bool, np.bool_)):
        return fallback('segmentation', 'watershed', image, markers, connectivity, offset,
                        mask, compactness, watershed_line)

    if mask is not None and mask.shape != image.shape:
        raise ValueError(f'`mask` (shape {mask.shape}) must have same shape as `image` (shape {image.shape})')
    if not canonical_markers or markers.shape != image.shape:
        _core.watershed_check_storage(image)
        if mask is None:
            selected = np.ones(image.shape, bool)
            pixel_count = image.size
        else:
            _core.watershed_check_storage(mask)
            selected = mask.astype(bool, copy=False)
            pixel_count = np.sum(selected)
        if markers is None:
            from scipy import ndimage as ndi
            from skimage.morphology import local_minima

            if not scalar_connectivity:
                _core.watershed_check_storage(connectivity)
            minima = local_minima(image, connectivity=connectivity) * selected
            footprint = ndi.generate_binary_structure(minima.ndim, connectivity)
            markers = ndi.label(minima, structure=footprint)[0]
        elif not canonical_markers:
            markers = regular_seeds(image.shape, int(markers / (pixel_count / image.size)))
            markers *= selected
        else:
            _core.watershed_check_storage(markers)
            markers = markers * selected
            if markers.shape != image.shape:
                raise ValueError(f'`markers` (shape {markers.shape}) must have same '
                                 f'shape as `image` (shape {image.shape})')

    padding = neighbors = None
    if offset is not None or not scalar_connectivity or (connectivity is not None and connectivity != 1):
        if not scalar_connectivity:
            _core.watershed_check_storage(connectivity)
        if array_offset:
            _core.watershed_check_storage(offset)
        padding, neighbors = _neighborhood(image, connectivity, offset)
    return _core.watershed(image, markers, mask, float(compactness), bool(watershed_line), padding, neighbors)
