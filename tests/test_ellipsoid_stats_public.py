"""Frozen public ellipsoid_stats behavior on both retained reference profiles."""
import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import types
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/ellipsoid_stats_originals_v1.json').read_bytes())
PROFILE = 'minimum' if np.__version__.startswith('1.') else 'modern'
ROWS = FIXTURE['profiles'][PROFILE]['rows']

if sys.platform == 'win32':
    overlay = (ROOT / 'tests/fixtures/ellipsoid_stats_windows_v2.json').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(overlay).hexdigest() == '901ad9987b738100ec234698e521843233668569075b9e3bab0a0944f7f443a9'
    precision = np.finfo(np.longdouble)
    profile_key = '/'.join((
        platform.system(), platform.machine().lower(),
        f'{sys.version_info.major}.{sys.version_info.minor}', np.__version__,
        version('scipy'), version('scikit-image'),
        f'{np.dtype(np.longdouble).itemsize}-{precision.nmant}-{precision.nexp}',
    ))
    replacements = json.loads(overlay)['profiles'][profile_key]
    ROWS = [replacements.get(row['id'], row) for row in ROWS]


def radius(kind, value):
    if kind in ('bool', 'np.bool_'):
        return (bool if kind == 'bool' else np.bool_)(value == 'True')
    if kind.startswith('np.'):
        return getattr(np, kind[3:])(value)
    if kind.startswith('array-'):
        number = float(value)
        return np.asarray({'array-0d': number, 'array-singleton': [number],
                           'array-vector': [number, number + 1]}[kind])
    return {'float': float, 'int': int, 'str': str, 'complex': complex}[kind](value)


@pytest.fixture(scope='module')
def product():
    extension = os.environ.get('ELLIPSOID_STATS_CORE')
    if extension:
        name = '_ellipsoid_stats_private'
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / 'python/skimagexx')]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + '._core', extension)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        package._core = core
        sys.modules[spec.name] = core
        frontend = importlib.import_module(name + '._ellipsoid_stats')
        compat = importlib.import_module(name + '._compat')
        return frontend.ellipsoid_stats, core, compat

    import skimagexx as xx
    frontend = importlib.import_module('skimagexx._ellipsoid_stats')
    compat = importlib.import_module('skimagexx._compat')
    assert xx.draw.ellipsoid_stats is frontend.ellipsoid_stats
    assert xx.draw.__all__.count('ellipsoid_stats') == 1
    assert xx.get_num_threads() == 1
    return xx.draw.ellipsoid_stats, xx._core, compat


def assert_value(observed, expected, field):
    assert f'{type(observed).__module__}.{type(observed).__qualname__}' == expected['kind']
    if 'shape' in expected:
        assert list(observed.shape) == expected['shape']
        assert observed.dtype.str == expected['dtype']
        np.testing.assert_allclose(observed.flat, [float.fromhex(word) for word in expected['values']], rtol=2e-14)
        return
    value = float.fromhex(expected['hex'])
    if math.isnan(value):
        assert math.isnan(observed)
    elif field == 0 or not math.isfinite(value):
        assert float(observed).hex() == expected['hex']
    elif type(observed) is np.float32:
        assert abs(observed - value) <= abs(value) * 3e-7
    else:
        assert abs(observed - value) <= abs(value) * 2e-14


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['id'])
def test_pinned_original(row, product):
    target, _, compat = product
    args = tuple(radius(*entry) for entry in row['args'])
    prior = compat.set_fallback_enabled(True)
    try:
        with warnings.catch_warnings(record=True) as logged:
            warnings.simplefilter('always')
            if 'error' in row:
                failure = row['error']
                cls = {'TypeError': TypeError, 'ValueError': ValueError,
                       'OverflowError': OverflowError, 'ZeroDivisionError': ZeroDivisionError}[failure['kind']]
                with pytest.raises(cls) as raised:
                    target(*args)
                assert str(raised.value) == failure['message']
            else:
                actual = target(*args)
                assert type(actual) is tuple and len(actual) == 2
                for field, (observed, expected) in enumerate(zip(actual, row['result'])):
                    assert_value(observed, expected, field)
        compatibility = any(kind.startswith('array-') or kind in ('str', 'complex') for kind, _ in row['args'])
        fallback_warnings = [warning for warning in logged if warning.category is compat.FallbackWarning]
        assert len(fallback_warnings) == compatibility
        assert [{'kind': item.category.__name__, 'message': str(item.message)} for item in logged
                if item.category is not compat.FallbackWarning] == row['warnings']
    finally:
        compat.set_fallback_enabled(prior)


def test_owned_entries_and_strict_fallback(product, monkeypatch):
    target, core, compat = product
    entries = {core._ellipsoid_stats_builtin, core._ellipsoid_stats_rg}
    assert str(inspect.signature(target)) == '(a, b, c)'
    assert all(inspect.isbuiltin(entry) and entry.__module__ == core.__name__ for entry in entries)
    calls = []

    def profile(frame, event, value):
        if event == 'c_return' and value in entries:
            calls.append(value.__name__)

    def refuse_provider(*args):
        raise AssertionError('bounded case delegated the integral')

    import scipy.special
    monkeypatch.setattr(scipy.special, 'elliprg', refuse_provider)
    old = sys.getprofile()
    previous = compat.set_fallback_enabled(False)
    try:
        sys.setprofile(profile)
        target(1.0, 2.0, 3.0)
        target(np.float32(1), np.float32(2), np.float32(3))
        assert calls == ['_ellipsoid_stats_builtin', '_ellipsoid_stats_builtin', '_ellipsoid_stats_rg']
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            target(np.array([1, 2]), 2.0, 3.0)
    finally:
        sys.setprofile(old)
        compat.set_fallback_enabled(previous)


def test_stock_and_native_public_pipeline(product):
    target, _, compat = product

    class Draw:
        ellipsoid_stats = staticmethod(target)

    from skimage import draw

    def pipeline(backend):
        return backend.ellipsoid_stats(1.0, 2.0, 3.0)

    previous = compat.set_fallback_enabled(False)
    try:
        np.testing.assert_allclose(pipeline(Draw), pipeline(draw), rtol=2e-14)
    finally:
        compat.set_fallback_enabled(previous)
