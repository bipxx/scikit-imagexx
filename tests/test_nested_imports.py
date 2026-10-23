"""Nested import compatibility must retain explicit upstream execution policy."""
import importlib
import inspect
import sys
import subprocess
import warnings

import numpy as np
import pytest
import skimagexx as xx


@pytest.fixture(autouse=True)
def strict():
    previous = xx.set_fallback_enabled(False)
    yield
    xx.set_fallback_enabled(previous)


def test_root_wildcard_keeps_upstream_modules_and_native_aliases():
    import skimage
    namespace = {}
    exec('from skimagexx import *', namespace)
    assert set(skimage.__all__) <= namespace.keys()
    assert namespace['__version__'] == xx.__version__
    assert namespace['threshold_li'] is xx.filters.threshold_li
    assert namespace['rank_order'] is xx.filters.rank_order
    assert namespace['label2rgb'] is xx.color.label2rgb
    for name in skimage.__all__:
        if name != '__version__':
            assert namespace[name] is getattr(xx, name)


def test_root_discovery_before_lazy_module_access():
    # A fresh process avoids earlier tests populating lazy module globals.
    result = subprocess.run([sys.executable, '-c',
        'import skimagexx as x; assert set(x.__all__) <= set(dir(x))'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('path,name,public', [
    ('color.colorconv', 'rgb2gray', 'color'),
    ('filters.thresholding', 'threshold_li', 'filters'),
    ('util.dtype', 'img_as_float32', 'util'),
    ('morphology.gray', 'erosion', 'morphology'),
    ('transform.integral', 'integral_image', 'transform'),
    ('filters.thresholding', 'histogram', 'exposure'),
])
def test_nested_native_alias_uses_the_public_wrapper(path, name, public):
    namespace = importlib.import_module('skimagexx.' + path)
    public_namespace = importlib.import_module('skimagexx.' + public)
    assert getattr(namespace, name) is getattr(public_namespace, name)


def test_imported_module_alias_uses_its_canonical_namespace():
    namespace = importlib.import_module('skimagexx.color.colorconv')
    assert namespace.dtype is importlib.import_module('skimagexx.util.dtype')


def test_native_wildcard_import_does_not_require_upstream():
    script = '''
import importlib.abc
import sys
class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'skimage' or fullname.startswith('skimage.'):
            raise ModuleNotFoundError('upstream is unavailable', name=fullname)
sys.meta_path.insert(0, NoUpstream())
namespace = {}
exec('from skimagexx import *', namespace)
assert 'threshold_li' in namespace and 'morphology' in namespace
assert not any(name == 'skimage' or name.startswith('skimage.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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
    from skimagexx import _rank
    assert mean is _rank.mean and mean is not original
    for policy in (False,True):
        xx.set_fallback_enabled(policy)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');actual=mean(image,footprint)
        assert not any(isinstance(w.message,xx.FallbackWarning) for w in notices)
        np.testing.assert_array_equal(actual,original(image,footprint))


def test_unported_rank_keeps_explicit_fallback_policy():
    from skimagexx.filters.rank import geometric_mean
    from skimage.filters.rank import geometric_mean as original
    image=np.arange(25,dtype=np.uint8).reshape(5,5);footprint=np.ones((3,3),bool)
    assert inspect.signature(geometric_mean)==inspect.signature(original)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):
        geometric_mean(image,footprint)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning,match='skimage.filters.rank.geometric_mean'):
        actual=geometric_mean(image,footprint)
    np.testing.assert_array_equal(actual,original(image,footprint))
    xx.set_fallback_enabled(False)
    with pytest.raises(NotImplementedError):geometric_mean(image,footprint)


def test_deep_import_and_native_precedence():
    from skimagexx.filters.rank.generic import maximum
    from skimage.filters.rank.generic import maximum as original
    assert inspect.signature(maximum) == inspect.signature(original)
    from skimagexx import _rank
    assert maximum is _rank.maximum and maximum is not original
    image=np.arange(9,dtype=np.uint8).reshape(3,3);footprint=np.ones((3,3),bool)
    for policy in (False,True):
        xx.set_fallback_enabled(policy)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');actual=maximum(image,footprint)
        assert not any(isinstance(w.message,xx.FallbackWarning) for w in notices)
        np.testing.assert_array_equal(actual,original(image,footprint))
    xx.set_fallback_enabled(False)
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
    from skimagexx import _geometric
    assert xx.transform.AffineTransform is _geometric.AffineTransform
    with pytest.raises(NotImplementedError):
        xx.transform.EssentialMatrixTransform


@pytest.mark.parametrize('path', ['filters.not_a_module', 'filters._median',
                                  'coverage.rank', '_core.rank'])
def test_unknown_private_and_native_paths_are_not_intercepted(path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('skimagexx.' + path)
    assert 'skimagexx.' + path not in sys.modules
