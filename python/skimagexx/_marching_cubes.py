"""Native isosurface extraction with the scikit-image coordinate conventions."""

import numpy as np

from . import _core
from ._compat import fallback


def marching_cubes(
    volume,
    level=None,
    *,
    spacing=(1.0, 1.0, 1.0),
    gradient_direction='descent',
    step_size=1,
    allow_degenerate=True,
    method='lewiner',
    mask=None,
):
    """Extract an isosurface from a three-dimensional volume."""
    def provider():
        return fallback(
            'measure', 'marching_cubes', volume, level, spacing=spacing,
            gradient_direction=gradient_direction, step_size=step_size,
            allow_degenerate=allow_degenerate, method=method, mask=mask,
        )

    if method != 'lewiner' and method != 'lorensen':
        raise ValueError("method should be either 'lewiner' or 'lorensen'")
    if not isinstance(volume, np.ndarray) or volume.ndim != 3:
        raise ValueError('Input volume should be a 3D numpy array.')
    if min(volume.shape) < 2:
        raise ValueError('Input array must be at least 2x2x2.')
    if (
        type(volume) is not np.ndarray
        or volume.dtype.kind not in 'buif'
        or volume.dtype.metadata is not None
        or (mask is not None and (
            type(mask) is not np.ndarray
            or mask.dtype not in (np.dtype(bool), np.dtype('uint8'))
            or mask.dtype.metadata is not None
        ))
    ):
        return provider()

    if not volume.flags.owndata:
        try:
            _core.measure_marching_cubes_check_storage(volume)
        except ValueError:
            return provider()
    prepared = np.require(volume, np.float32, requirements=('C', 'A'))
    minimum, maximum = prepared.min(), prepared.max()
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        return provider()
    if level is None:
        iso = 0.5 * (minimum + maximum)
    else:
        iso = float(level)
        if iso < minimum or iso > maximum:
            raise ValueError('Surface level must be within volume data range.')
    if len(spacing) != 3:
        raise ValueError('`spacing` must consist of three floats.')
    step = int(step_size)
    if step < 1:
        raise ValueError('step_size must be at least one.')
    prepared_mask = None
    if mask is not None:
        if mask.shape != prepared.shape:
            raise ValueError('volume and mask must have the same shape.')
        if not mask.flags.owndata:
            try:
                _core.measure_marching_cubes_check_storage(mask)
            except ValueError:
                return provider()
        prepared_mask = np.ascontiguousarray(mask)
    if step > 2**31 - 1:
        return provider()
    vertices, faces, normals, values = _core.measure_marching_cubes(
        prepared, iso, step, method == 'lorensen', prepared_mask,
    )
    if not len(vertices):
        raise RuntimeError('No surface found at the given iso value.')

    vertices = vertices[:, ::-1]
    normals = normals[:, ::-1]
    if gradient_direction == 'descent':
        faces = faces[:, ::-1]
    elif gradient_direction != 'ascent':
        raise ValueError(
            f'Incorrect input {gradient_direction} in `gradient_direction`, see docstring.'
        )
    if spacing is not _UNIT_SPACING and not np.array_equal(spacing, (1, 1, 1)):
        vertices = vertices * np.r_[spacing]
    if not allow_degenerate:
        return _core.measure_marching_cubes_remove_degenerate(
            vertices.astype(np.float32), faces, normals, values,
        )
    return vertices, faces, normals, values


_UNIT_SPACING = marching_cubes.__kwdefaults__['spacing']
