"""Four frozen full-call work items for the bounded convex hull object route."""

from functools import partial
import numpy as np


FIXED = (
    ("tiny-diagonal-1", "diagonal", (13, 17), {"connectivity": 1}),
    ("tiny-diagonal-2", "diagonal", (13, 17), {}),
    ("wide-separated", "separated", (128, 192), {"connectivity": 1}),
    ("overlapping-hulls", "opposite-corners-overlap", (96, 144), {}),
)


def fixture(recipe, shape):
    height, width = shape
    image = np.zeros(shape, dtype=bool)
    if recipe == "diagonal":
        image[1:5, 1] = image[4, 1:5] = image[5:10, 5] = image[9, 5:11] = True
    elif recipe == "separated":
        for row in range(10, height - 10, 25):
            for col in range(10, width - 10, 25):
                image[row:row + 8, col] = image[row + 7, col:col + 11] = True
    elif recipe == "opposite-corners-overlap":
        image[4:height - 4, 4] = image[height - 5, 4:width - 4] = True
        top = height // 3 + 2
        image[top, 8:width - 2] = image[top:height - 2, width - 3] = True
    else:
        raise ValueError(recipe)
    return image


def public_call_pairs(original, native):
    for name, recipe, shape, kwargs in FIXED:
        image = fixture(recipe, shape)
        yield name, partial(original, image, **kwargs), partial(native, image, **kwargs)
