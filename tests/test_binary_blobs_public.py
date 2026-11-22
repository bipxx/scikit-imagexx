import inspect
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy
import skimage

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from binary_blobs_reference import cases, generator_state, observe, sha


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'coverage/contracts/binary-blobs-reference-v1.json'
REFERENCE = FIXTURE.read_bytes()
assert sha(REFERENCE.replace(b'\r\n', b'\n')) == '3a76afee05fd672ed033a8d6fbf6f0315093e4132c79e0d3a4a8f55c0c6b6fa6'
SAVED = {row['id']: row for row in json.loads(REFERENCE)['rows']}

PROFILE_FIXTURE = ROOT / 'coverage/contracts/binary-blobs-reference-profile-v2.json'
PROFILE_BYTES = PROFILE_FIXTURE.read_bytes().replace(b'\r\n', b'\n')
assert sha(PROFILE_BYTES) == '2702bcdb903187f24c0f1ad6c7ff2396c6785b8bea6e95dd1bfc9c46ddddc9af'
PROFILE_KEY = '/'.join((
    platform.system(), platform.machine().lower(),
    f'{sys.version_info.major}.{sys.version_info.minor}',
    np.__version__, scipy.__version__, skimage.__version__,
))
PROFILE_VALUES = json.loads(PROFILE_BYTES)['profiles'][PROFILE_KEY]
SAVED['default']['gaussian'][0]['output'] = PROFILE_VALUES['default_gaussian_sha256']
SAVED['rank-zero']['observed']['output']['type'] = PROFILE_VALUES['rank_zero_type_name']


@pytest.mark.parametrize('name,options,native', list(cases()), ids=list(SAVED))
def test_binary_blobs_frozen_public(name, options, native, monkeypatch):
    from skimage.data import binary_blobs as original
    from skimagexx._binary_blobs import binary_blobs as owned

    assert xx.data.binary_blobs is owned
    assert inspect.signature(owned) == inspect.signature(original)
    row = SAVED[name]
    assert native == row['native']
    gaussian, contrast = _core.gaussian_nd, _core.contrast
    assert all(inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
               for entry in (gaussian, contrast))
    gaussian_calls, contrast_calls = [], []

    def record_gaussian(mask, *args):
        result = gaussian(mask, *args)
        gaussian_calls.append(dict(input=sha(mask.tobytes()), output=sha(result[0].tobytes())))
        return result

    def record_contrast(field, positions):
        values = contrast(field, positions)
        percentile = 100 * (1 - options.get('volume_fraction', 0.5))
        expected = float(np.percentile(field, percentile))
        assert (values[0].hex(), values[1].hex()) == (expected.hex(), expected.hex())
        contrast_calls.append(positions)
        return values

    monkeypatch.setattr(_core, 'gaussian_nd', record_gaussian)
    monkeypatch.setattr(_core, 'contrast', record_contrast)
    previous = xx.set_fallback_enabled(False)
    try:
        if not native:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                xx.data.binary_blobs(**options)
            assert generator_state(options.get('rng')) == row['observed']['state_before']
            xx.set_fallback_enabled(True)
        actual = observe(xx.data.binary_blobs, options)
        fallback_notice = [FallbackWarning.__module__, FallbackWarning.__name__,
                           'skimage.data.binary_blobs executed by scikit-image fallback']
        assert actual['warnings'].count(fallback_notice) == int(not native)
        if not native:
            actual['warnings'].remove(fallback_notice)
        assert actual == row['observed']
    finally:
        xx.set_fallback_enabled(previous)
    assert len(contrast_calls) == int(native)
    assert gaussian_calls == (row['gaussian'] if native else [])


def test_binary_blobs_export_and_default_generator(monkeypatch):
    from skimage import data as original
    from skimagexx._binary_blobs import binary_blobs as owned

    exports = {}
    exec('from skimagexx.data import *', exports)
    assert exports['binary_blobs'] is xx.data.binary_blobs is owned
    default_rng = np.random.default_rng
    calls = []

    def observed_rng(seed):
        calls.append(seed)
        return default_rng(9912 if seed is None else seed)

    def pipeline(backend):
        return backend.data.binary_blobs(length=24, boundary_mode='wrap')

    monkeypatch.setattr(np.random, 'default_rng', observed_rng)
    previous = xx.set_fallback_enabled(False)
    try:
        marker = object()
        stub = SimpleNamespace(data=SimpleNamespace(binary_blobs=lambda **kw: marker))
        assert pipeline(stub) is marker
        np.testing.assert_array_equal(pipeline(xx), pipeline(SimpleNamespace(data=original)))
        assert calls == [None, None]
    finally:
        xx.set_fallback_enabled(previous)
