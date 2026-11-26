"""Source-ordered seeded shape placement with native indexed painting."""

import warnings

import numpy as np

from . import _core
from ._compat import fallback
from ._draw_coordinates import ellipse
from ._draw_polygon import polygon


def _rectangle(row, column, height, width, lo, hi, rng):
    dx = lo + rng.integers(max(1, min(height - row, hi) - lo)) - 1
    dy = lo + rng.integers(max(1, min(width - column, hi) - lo)) - 1
    if dx and dy:
        rows = np.arange(row, row + dx + 1, dtype=np.intp)
        columns = np.arange(column, column + dy + 1, dtype=np.intp)
        points = rows.repeat(dy + 1), np.tile(columns, dx + 1)
    else:
        points = polygon([row, row + dx, row + dx, row], [column, column, column + dy, column + dy])
    return points, ('rectangle', ((row, row + dx + 1), (column, column + dy + 1)))


def _circle_offsets(radius):
    # Use the same division and squared-distance test as an unrotated disk.
    offsets = np.arange(1 - radius, radius, dtype=np.float64)
    squared = (offsets / radius) ** 2
    rr, cc = np.nonzero(squared[:, None] + squared < 1)
    return rr - (radius - 1), cc - (radius - 1)


def _circle(row, column, height, width, lo, hi, rng):
    if lo == 1 or hi == 1:
        raise ValueError('size must be > 1 for circles')
    minimum = lo // 2.0
    available = min(column, width - column, row, height - row, hi // 2.0) - minimum
    if available < 0:
        raise ArithmeticError('cannot fit shape to image')
    radius = int(minimum + rng.integers(max(1, available)))
    rr, cc = _circle_offsets(radius)
    points = rr + row, cc + column
    return points, ('circle', ((row - radius + 1, row + radius), (column - radius + 1, column + radius)))


def _triangle(row, column, height, width, lo, hi, rng):
    if lo == 1 or hi == 1:
        raise ValueError('dimension must be > 1 for triangles')
    available = min(width - column, row, hi) - lo
    side = lo + rng.integers(max(1, available)) - 1
    rise = int(np.ceil(_TRIANGLE_RISE * side))
    points = polygon([row, row - rise, row], [column, column + side // 2, column + side])
    return points, ('triangle', ((row - rise, row + 1), (column, column + side + 1)))


def _ellipse(row, column, height, width, lo, hi, rng):
    if lo == 1 or hi == 1:
        raise ValueError('size must be > 1 for ellipses')
    minimum = lo / 2.0
    available = min(column, width - column, row, height - row, hi / 2.0)
    if available < minimum:
        raise ArithmeticError('cannot fit shape to image')
    row_radius = rng.uniform(minimum, available + 1)
    column_radius = rng.uniform(minimum, available + 1)
    rotation = rng.uniform(-np.pi, np.pi)
    rr, cc = ellipse(row, column, row_radius, column_radius, shape=(height, width), rotation=rotation)
    return (rr, cc), ('ellipse', ((np.min(rr), np.max(rr) + 1), (np.min(cc), np.max(cc) + 1)))


_GENERATORS = dict(rectangle=_rectangle, circle=_circle, triangle=_triangle, ellipse=_ellipse)
_CHOICES = tuple(_GENERATORS.values())
_TRIANGLE_RISE = np.sqrt(3 / 4.0)


def _int_range(value):
    return type(value) is tuple and len(value) == 2 and type(value[0]) is int and type(value[1]) is int


def _ranges_supported(value, channels):
    if value is None:
        return True
    if channels == 1:
        return _int_range(value)
    return type(value) is tuple and len(value) in (1, channels) and all(_int_range(pair) for pair in value)


def random_shapes(image_shape, max_shapes, min_shapes=1, min_size=2, max_size=None, num_channels=3,
                  shape=None, intensity_range=None, allow_overlap=False, num_trials=100, rng=None, *, channel_axis=-1):
    """Generate bounded seeded shapes and placement-ordered bounding boxes."""
    channels = 1 if channel_axis is None else num_channels
    supported = (
        type(image_shape) is tuple and len(image_shape) == 2
        and type(image_shape[0]) is int and 1 <= image_shape[0] <= 256
        and type(image_shape[1]) is int and 1 <= image_shape[1] <= 256
        and type(max_shapes) is int and 0 <= max_shapes <= 128
        and type(min_shapes) is int and 0 <= min_shapes <= 128
        and type(min_size) is int and 1 <= min_size <= 256
        and (max_size is None or type(max_size) is int and 0 <= max_size <= 256 and (max_size == 0 or max_size >= min_size))
        and (channel_axis is None or type(channel_axis) is int and -64 <= channel_axis <= 64)
        and type(channels) is int and 1 <= channels <= 8
        and (shape is None or type(shape) is str)
        and type(allow_overlap) is bool and type(num_trials) is int and 0 <= num_trials <= 100
        and _ranges_supported(intensity_range, channels)
        and (type(rng) is int and 0 <= rng < 2**64
             or type(rng) is np.random.Generator and type(rng.bit_generator) is np.random.PCG64)
    )
    if not supported:
        return fallback('draw', 'random_shapes', image_shape, max_shapes, min_shapes=min_shapes, min_size=min_size,
                        max_size=max_size, num_channels=num_channels, shape=shape, intensity_range=intensity_range,
                        allow_overlap=allow_overlap, num_trials=num_trials, rng=rng, channel_axis=channel_axis)

    height, width = image_shape
    if min_size > height or min_size > width:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    hi = max_size or max(height, width)
    ranges = (intensity_range or (0, 254),) if channels == 1 else intensity_range or ((0, 254),)
    for lo, high in ranges:
        if not (0 <= lo <= 255 and 0 <= high <= 255):
            raise ValueError('Intensity range must lie within (0, 255) interval')
    if len(ranges) == 1:
        ranges = ranges * channels

    generator = rng if type(rng) is np.random.Generator else np.random.default_rng(rng)
    integers = generator.integers
    image = np.full((height, width, channels), 255, dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=np.uint8)
    labels = []
    count = integers(min_shapes, max_shapes + 1)
    colors = np.transpose([integers(lo, high + 1, size=count) for lo, high in ranges])
    colors = colors.astype(np.uint8, order='C')
    generate = _GENERATORS[shape] if count and shape is not None else None
    row_limit = max(1, height - min_size)
    column_limit = max(1, width - min_size)
    for index in range(count):
        place_shape = _CHOICES[integers(len(_CHOICES))] if generate is None else generate
        for _ in range(num_trials):
            column = integers(column_limit)
            row = integers(row_limit)
            try:
                (rr, cc), label = place_shape(row, column, height, width, min_size, hi, generator)
            except ArithmeticError:
                continue
            if _core.random_shapes_paint(image, occupied, rr, cc, colors[index], allow_overlap):
                labels.append(label)
                break
        else:
            warnings.warn('Could not fit any shapes to image, consider reducing the minimum dimension', stacklevel=2)

    if channel_axis is None:
        image = np.squeeze(image, axis=2)
    else:
        image = np.moveaxis(image, -1, channel_axis)
    return image, labels
