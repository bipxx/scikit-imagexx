"""Matplotlib compatibility path for displaying descriptor correspondences."""

import numpy as np

from ._compat import fallback


def _plain_array(value):
    return type(value) is np.ndarray and value.dtype.metadata is None


def _pair_image(image0, image1, alignment):
    height = max(image0.shape[0], image1.shape[0])
    width = max(image0.shape[1], image1.shape[1])
    row_offset, col_offset = (0, width) if alignment == "horizontal" else (height, 0)
    shape = (height + row_offset, width + col_offset, *image0.shape[2:])
    if image0.shape == image1.shape:
        image = np.concatenate((image0, image1), axis=int(alignment == "horizontal"))
    else:
        image = np.zeros(shape, dtype=np.result_type(image0, image1))
        image[:image0.shape[0], :image0.shape[1]] = image0
        image[row_offset:row_offset + image1.shape[0], col_offset:col_offset + image1.shape[1]] = image1
    return image, row_offset, col_offset


def plot_matched_features(
    image0, image1, *, keypoints0, keypoints1, matches, ax,
    keypoints_color="k", matches_color=None, only_matches=False, alignment="horizontal",
):
    if not (
        all(map(_plain_array, (image0, image1, keypoints0, keypoints1, matches)))
        and image0.ndim in (2, 3) and image0.shape[2:] == image1.shape[2:]
        and keypoints0.ndim == keypoints1.ndim == matches.ndim == 2
        and keypoints0.shape[1] >= 2 and keypoints1.shape[1] >= 2
        and matches.shape[1] == 2 and matches.dtype.kind in "iu"
        and type(alignment) is str and alignment in ("horizontal", "vertical")
    ):
        return fallback(
            "feature", "plot_matched_features", image0, image1, keypoints0=keypoints0,
            keypoints1=keypoints1, matches=matches, ax=ax, keypoints_color=keypoints_color,
            matches_color=matches_color, only_matches=only_matches, alignment=alignment,
        )

    from matplotlib.colors import is_color_like
    from skimage.util import img_as_float

    image, dy, dx = _pair_image(img_as_float(image0), img_as_float(image1), alignment)
    # Use NumPy scalar offsets so small integer coordinate dtypes promote
    # exactly as in the upstream plotting path.
    dy, dx = np.intp(dy), np.intp(dx)
    y0, x0 = keypoints0[:, 0], keypoints0[:, 1]
    y1, x1 = keypoints1[:, 0] + dy, keypoints1[:, 1] + dx
    if not only_matches:
        ax.scatter(x0, y0, facecolors="none", edgecolors=keypoints_color)
        ax.scatter(x1, y1, facecolors="none", edgecolors=keypoints_color)
    ax.imshow(image, cmap="gray")
    ax.axis((0, image.shape[1], image.shape[0], 0))

    count = len(matches)
    if matches_color is None:
        colors = np.random.default_rng(seed=0).random((count, 3))
    elif is_color_like(matches_color):
        colors = [matches_color] * count
    elif hasattr(matches_color, "__len__") and len(matches_color) == count:
        colors = matches_color
    else:
        raise ValueError(
            "`matches_color` needs to be a single color or a sequence of length equal to the number of matches."
        )
    for i, (left, right) in enumerate(matches):
        ax.plot((x0[left], x1[right]), (y0[left], y1[right]), "-", color=colors[i])
