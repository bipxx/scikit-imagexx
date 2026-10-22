"""Nested import compatibility must retain explicit upstream execution policy."""
import importlib
import inspect
import sys

import numpy as np
import pytest
import skimagexx as xx


@pytest.fixture(autouse=True)
def strict():
    previous = xx.set_fallback_enabled(False)
    yield
    xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('path', ['filters.rank', 'filters.rank.generic',
                                  'color.colorconv', 'util.dtype'])
def test_nested_namespace_import_and_identity(path):
    native = importlib.import_module('skimagexx.' + path)
    assert native is importlib.import_module('skimagexx.' + path)
    parent, name = path.rsplit('.', 1)
    assert getattr(importlib.import_module('skimagexx.' + parent), name) is native
    assert native.__name__ == 'skimagexx.' + path
    assert native is not importlib.import_module('skimage.' + path)


def test_from_import_signature_and_policy():
    from skimagexx.filters.rank import mean
    from skimage.filters.rank import mean as original
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    footprint = np.ones((3, 3), dtype=bool)
    assert inspect.signature(mean) == inspect.signature(original)
    assert mean.__name__ == original.__name__
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        mean(image, footprint)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning, match='skimage.filters.rank.mean'):
        actual = mean(image, footprint)
    np.testing.assert_array_equal(actual, original(image, footprint))
    xx.set_fallback_enabled(False)
    with pytest.raises(NotImplementedError):
        mean(image, footprint)


def test_deep_import_and_native_precedence():
    from skimagexx.filters.rank.generic import maximum
    from skimage.filters.rank.generic import maximum as original
    assert inspect.signature(maximum) == inspect.signature(original)
    with pytest.raises(NotImplementedError):
        maximum(np.zeros((3, 3), np.uint8), np.ones((3, 3), bool))
    from skimagexx import filters
    from skimagexx.filters import gaussian
    assert gaussian is filters.gaussian
    assert gaussian.__module__ == 'skimagexx.filters'
    assert importlib.import_module('skimagexx._core') is xx._core
    # Treating a native file as a namespace must not alter its relative imports.
    np.testing.assert_array_equal(gaussian(np.zeros((3, 3))), np.zeros((3, 3)))


def test_classes_constants_and_discovery_keep_policy():
    namespace = importlib.import_module('skimagexx.color.colorconv')
    assert 'rgb2gray' in dir(namespace)
    with pytest.raises(NotImplementedError, match='upstream-only'):
        namespace.rgb_from_xyz
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = namespace.rgb_from_xyz
    from skimage.color.colorconv import rgb_from_xyz
    assert actual is rgb_from_xyz
    xx.set_fallback_enabled(False)
    with pytest.raises(NotImplementedError):
        xx.transform.AffineTransform


@pytest.mark.parametrize('path', ['filters.not_a_module', 'filters._median',
                                  'coverage.rank', '_core.rank'])
def test_unknown_private_and_native_paths_are_not_intercepted(path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('skimagexx.' + path)
    assert 'skimagexx.' + path not in sys.modules
