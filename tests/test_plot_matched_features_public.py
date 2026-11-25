import importlib.abc
import importlib.util
import inspect
import sys
from types import ModuleType

import numpy as np
import pytest
from skimage.feature import plot_matched_features as original

from skimagexx import FallbackWarning, feature


try:
    HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
except ModuleNotFoundError:
    # A finder that blocks Matplotlib can raise instead of returning no spec.
    HAS_MATPLOTLIB = False


def _stub_axes_pair(monkeypatch):
    matplotlib = ModuleType("matplotlib")
    matplotlib.__path__ = []
    colors = ModuleType("matplotlib.colors")
    # The cases below use named colors and sequences of two named colors.
    colors.is_color_like = lambda value: isinstance(value, str)
    matplotlib.colors = colors
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", colors)

    class AxesRecorder:
        def __init__(self):
            self.calls = []

        def scatter(self, *args, **kwargs):
            self.calls.append(("scatter", args, kwargs))

        def imshow(self, *args, **kwargs):
            self.calls.append(("imshow", args, kwargs))

        def axis(self, *args, **kwargs):
            self.calls.append(("axis", args, kwargs))

        def plot(self, *args, **kwargs):
            self.calls.append(("plot", args, kwargs))

    return AxesRecorder(), AxesRecorder()


@pytest.fixture
def axes_pair(monkeypatch):
    if HAS_MATPLOTLIB:
        from matplotlib.figure import Figure

        return Figure().subplots(), Figure().subplots()
    return _stub_axes_pair(monkeypatch)


def _assert_axes_parity(a, b, *, match_count, only_matches=False):
    if hasattr(a, "calls"):
        scatter_calls = [] if only_matches else ["scatter", "scatter"]
        assert [name for name, _, _ in a.calls] == [
            *scatter_calls, "imshow", "axis", *(["plot"] * match_count),
        ]
        np.testing.assert_equal(a.calls, b.calls)
        return

    assert a.axis() == b.axis()
    assert len(a.collections) == len(b.collections) == (0 if only_matches else 2)
    assert len(a.images) == len(b.images) == 1
    assert len(a.lines) == len(b.lines) == match_count
    for left, right in zip(a.lines, b.lines, strict=True):
        np.testing.assert_array_equal(left.get_xdata(), right.get_xdata())
        np.testing.assert_array_equal(left.get_ydata(), right.get_ydata())
        np.testing.assert_array_equal(left.get_color(), right.get_color())
    for left, right in zip(a.collections, b.collections, strict=True):
        np.testing.assert_array_equal(left.get_offsets(), right.get_offsets())
        np.testing.assert_array_equal(left.get_edgecolors(), right.get_edgecolors())
    np.testing.assert_array_equal(a.images[0].get_array(), b.images[0].get_array())


@pytest.mark.parametrize("alignment", ("horizontal", "vertical"))
@pytest.mark.parametrize("only_matches", (False, True))
@pytest.mark.parametrize("matches_color", (None, "cyan", ["tab:red", "tab:blue"]))
def test_matched_feature_artists(axes_pair, alignment, only_matches, matches_color):
    _check_matched_feature_artists(axes_pair, alignment, only_matches, matches_color)


def _check_matched_feature_artists(axes_pair, alignment, only_matches, matches_color):
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
    a, b = axes_pair
    assert candidate(image0, image1, ax=a, **kwargs) is original(image0, image1, ax=b, **kwargs) is None
    _assert_axes_parity(a, b, match_count=2, only_matches=only_matches)


def test_tagged_image_uses_compatibility_dispatch(axes_pair):
    image = np.ones((5, 5), dtype=np.dtype(np.float32, metadata={"test": True}))
    points = np.array([[1, 2]])
    kwargs = dict(keypoints0=points, keypoints1=points, matches=np.array([[0, 0]]))
    a, b = axes_pair
    with pytest.warns(FallbackWarning, match="plot_matched_features"):
        assert feature.plot_matched_features(image, image, ax=a, **kwargs) is None
    assert original(image, image, ax=b, **kwargs) is None
    _assert_axes_parity(a, b, match_count=1)


def test_matched_feature_calls_with_matplotlib_blocked(monkeypatch):
    class BlockMatplotlib(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                raise ModuleNotFoundError("No module named 'matplotlib'")

    for name in tuple(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [BlockMatplotlib(), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError, match="No module named 'matplotlib'"):
        importlib.import_module("matplotlib")

    _check_matched_feature_artists(
        _stub_axes_pair(monkeypatch), "vertical", False, ["tab:red", "tab:blue"],
    )
