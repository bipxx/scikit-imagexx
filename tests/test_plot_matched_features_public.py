import inspect

from matplotlib.figure import Figure
import numpy as np
import pytest
from skimage.feature import plot_matched_features as original

from skimagexx import FallbackWarning, feature


@pytest.mark.parametrize("alignment", ("horizontal", "vertical"))
@pytest.mark.parametrize("only_matches", (False, True))
@pytest.mark.parametrize("matches_color", (None, "cyan", ["tab:red", "tab:blue"]))
def test_matched_feature_artists(alignment, only_matches, matches_color):
    candidate = feature.plot_matched_features
    assert inspect.signature(candidate) == inspect.signature(original)
    image0 = np.arange(35, dtype=np.uint8).reshape(5, 7)
    image1 = np.linspace(0, 1, 36, dtype=np.float32).reshape(6, 6)
    keypoints0 = np.array([[0, 2], [4, 6]], dtype=np.int16)
    keypoints1 = np.array([[1, 0], [3, 4]], dtype=np.float32)
    kwargs = dict(
        keypoints0=keypoints0, keypoints1=keypoints1,
        matches=np.array([[0, 1], [1, 0]]), matches_color=matches_color,
        alignment=alignment, only_matches=only_matches,
    )
    a, b = Figure().subplots(), Figure().subplots()
    assert candidate(image0, image1, ax=a, **kwargs) is original(image0, image1, ax=b, **kwargs) is None
    assert a.axis() == b.axis()
    assert len(a.collections) == len(b.collections)
    assert len(a.images) == len(b.images) == 1
    assert len(a.lines) == len(b.lines) == 2
    for left, right in zip(a.lines, b.lines, strict=True):
        np.testing.assert_array_equal(left.get_xdata(), right.get_xdata())
        np.testing.assert_array_equal(left.get_ydata(), right.get_ydata())
        np.testing.assert_array_equal(left.get_color(), right.get_color())
    for left, right in zip(a.collections, b.collections, strict=True):
        np.testing.assert_array_equal(left.get_offsets(), right.get_offsets())
        np.testing.assert_array_equal(left.get_edgecolors(), right.get_edgecolors())
    np.testing.assert_array_equal(a.images[0].get_array(), b.images[0].get_array())


def test_tagged_image_uses_compatibility_dispatch():
    image = np.ones((5, 5), dtype=np.dtype(np.float32, metadata={"test": True}))
    points = np.array([[1, 2]])
    with pytest.warns(FallbackWarning, match="plot_matched_features"):
        feature.plot_matched_features(
            image, image, keypoints0=points, keypoints1=points,
            matches=np.array([[0, 0]]), ax=Figure().subplots(),
        )
