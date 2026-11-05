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
from skimagexx import _regionprops_state as owned
assert namespace['measure'].regionprops is owned.regionprops
assert 'skimagexx._regionprops_state_class' not in sys.modules
assert str(__import__('inspect').signature(owned.regionprops)) == '(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None)'
assert not any(name == 'skimage' or name.startswith('skimage.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_regionprops_lazy_provider_identity_and_lookup(monkeypatch):
    from skimagexx import _regionprops_state as owned
    from skimage.measure._regionprops import RegionProperties as provider
    cls = owned.RegionProperties
    assert cls is owned.RegionProperties and cls is not provider
    assert issubclass(cls, provider) and owned._Provider is provider
    assert (cls.__module__, cls.__qualname__) == ('skimagexx._regionprops_state', 'RegionProperties')
    assert xx.measure.regionprops is owned.regionprops
    calls = []
    sentinel = object()
    def tail(*args, **options):
        calls.append((args, options))
        return sentinel
    monkeypatch.setattr(owned, '_provider_regionprops', tail)
    labels = np.ones((2, 3), dtype=np.int16)
    intensity = np.ones_like(labels)
    assert owned.regionprops(labels, intensity) is sentinel
    assert len(calls) == 1 and calls[0][0][0] is labels and calls[0][0][1] is intensity


def test_regionprops_first_cold_lookup_and_warm_calls_do_not_self_import():
    script = """
import builtins
import sys
import numpy as np
import skimagexx as xx
from skimagexx import _regionprops_state as owned
import skimage.measure._regionprops as original
import scipy.ndimage as ndi
def denied(*args, **kwargs):
    raise AssertionError('original constructor algorithm was called')
original.regionprops = ndi.find_objects = denied
assert 'skimagexx._regionprops_state_class' not in sys.modules
assert 'RegionProperties' not in vars(owned) and '_provider_regionprops' not in vars(owned)
lookup = owned.__getattr__
calls = []
def tracked(name):
    calls.append(name)
    return lookup(name)
owned.__getattr__ = tracked
prior_import = builtins.__import__
def no_self_import(name, *args, **kwargs):
    assert name not in ('_regionprops_state', 'skimagexx._regionprops_state'), name
    return prior_import(name, *args, **kwargs)
builtins.__import__ = no_self_import
try:
    labels = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.int16)
    first = owned.regionprops(labels)
    assert calls == ['RegionProperties', '_provider_regionprops']
    assert first[0]._native_state is not None
    assert type(first[0]) is owned.RegionProperties
    calls.clear()
    second = owned.regionprops(labels)
    assert calls == [] and second[0]._native_state is not None
    assert type(second[0]) is type(first[0])
finally:
    builtins.__import__ = prior_import
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_regionprops_patched_aliases_and_deleted_lazy_aliases(monkeypatch):
    from skimagexx import _regionprops_state as owned
    labels = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.int16)
    cls = owned.RegionProperties
    provider = owned._provider_regionprops
    lookup = owned.__getattr__
    lookups = []
    def tracked(name):
        lookups.append(name)
        return lookup(name)
    monkeypatch.setattr(owned, '__getattr__', tracked)
    sentinel = object()
    calls = []
    def patched(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    monkeypatch.setattr(owned, '_provider_regionprops', patched)
    assert owned.regionprops(labels, labels) is sentinel and len(calls) == 1
    assert lookups == [] and calls[0][0][0] is labels and calls[0][0][1] is labels
    monkeypatch.delattr(owned, '_provider_regionprops')
    region = owned.regionprops(labels)[0]
    assert lookups == ['_provider_regionprops'] and owned._provider_regionprops is provider
    assert type(region) is cls and region._native_state is not None
    lookups.clear()
    monkeypatch.delattr(owned, 'RegionProperties')
    region = owned.regionprops(labels)[0]
    assert lookups == ['RegionProperties'] and type(region) is cls
    assert xx.measure.regionprops is owned.regionprops


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


def test_native_rank_and_unported_nested_filter_keep_explicit_policy(monkeypatch):
    from skimagexx.filters.rank import geometric_mean
    from skimage.filters.rank import geometric_mean as original
    from skimagexx import _rank
    image=np.arange(25,dtype=np.uint8).reshape(5,5);footprint=np.ones((3,3),bool)
    assert inspect.signature(geometric_mean)==inspect.signature(original)
    assert geometric_mean is _rank.geometric_mean and geometric_mean is not original
    for policy in (False,True):
        xx.set_fallback_enabled(policy)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');actual=geometric_mean(image,footprint)
        assert not any(isinstance(w.message,xx.FallbackWarning) for w in notices)
        np.testing.assert_array_equal(actual,original(image,footprint))
    canonical=importlib.import_module('skimagexx.filters.rank')
    generic=importlib.import_module('skimagexx.filters.rank.generic')
    original_rank=importlib.import_module('skimage.filters.rank')
    for name in ('majority','otsu'):
        native=getattr(canonical,name);reference=getattr(original_rank,name)
        assert native is getattr(generic,name) and native is getattr(_rank,name)
        assert native is not reference and native.__module__=='skimagexx._rank'
        assert inspect.signature(native)==inspect.signature(reference)
        for policy in (False,True):
            xx.set_fallback_enabled(policy)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual=native(image,footprint)
            assert not any(isinstance(w.message,xx.FallbackWarning) for w in notices)
            np.testing.assert_array_equal(actual,reference(image,footprint))
    # filter_forward is now native. Use the current upstream-only plotting
    # utility for policy checks, without making a plotting/numerical claim.
    namespace=importlib.import_module('skimagexx.filters')
    original_filters=importlib.import_module('skimage.filters')
    unported=namespace.try_all_threshold
    unported_original=original_filters.try_all_threshold
    from skimagexx.coverage import native_functions
    assert 'filters.try_all_threshold' not in native_functions()
    assert unported is not unported_original
    assert inspect.signature(unported)==inspect.signature(unported_original)
    calls=[];sentinel=object()
    def provider(value):
        calls.append(value)
        return sentinel
    monkeypatch.setattr(original_filters,'try_all_threshold',provider)
    xx.set_fallback_enabled(False)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):
        unported(image)
    assert not calls
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning,match=r'skimage\.filters\.try_all_threshold'):
        assert unported(image) is sentinel
    assert len(calls)==1 and calls[0] is image
    xx.set_fallback_enabled(False)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):
        unported(image)
    assert len(calls)==1


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
