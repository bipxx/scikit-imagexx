import inspect
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import scipy
import skimage
import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from denoise_bilateral_reference import cases, load, observe, sha


ROOT = Path(__file__).resolve().parents[1]
RECIPES = load(ROOT / 'benchmarks/denoise_bilateral.py')
PROFILES = {
    ('Windows', '2.5.4', '1.18.1', '0.26.0'): ('windows', '6b3ada19f9b44651f525789a4dc19edb72e4bb3eef5645250653181e5d3801b5'),
    ('Linux', '1.24.4', '1.15.3', '0.26.0'): ('minimum', 'f20b967651f827c4bbd4c94100bb7f386ccd4d21eda990a56a2bdea5ce45451c'),
}
VERSION = (platform.system(), np.__version__, scipy.__version__, skimage.__version__)


@pytest.fixture(scope='module')
def oracle():
    if VERSION not in PROFILES:
        return None
    profile, expected = PROFILES[VERSION]
    path = ROOT / f'coverage/contracts/denoise-bilateral-{profile}-reference-v1.json'
    raw = path.read_bytes().replace(b'\r\n', b'\n')
    assert sha(raw) == expected
    return json.loads(raw)


@pytest.mark.parametrize('name,image,options,native', list(cases(RECIPES)), ids=lambda row: row if type(row) is str else None)
def test_frozen_bilateral(name, image, options, native, oracle, monkeypatch):
    from skimage.restoration import denoise_bilateral as upstream
    from skimagexx._denoise_bilateral import denoise_bilateral as owned

    assert xx.restoration.denoise_bilateral is owned
    assert inspect.signature(owned) == inspect.signature(upstream)
    expected = None
    if oracle is not None:
        expected = next(item for item in oracle['rows'] if item['id'] == name)
        assert expected['native'] == native
    native_entry = _core.denoise_bilateral_constant
    assert inspect.isbuiltin(native_entry) and native_entry.__module__ == 'skimagexx._core'
    observations = []

    def record(field, maximum, window, cval, color, spatial):
        color_sigma = options.get('sigma_color')
        if color_sigma is None:
            color_sigma = image.std()
        observations.append(dict(field_sha256=sha(field.tobytes()), maximum=float(maximum).hex(),
                                 window=window, bins=color.size, mode='constant', cval=float(cval).hex(),
                                 sigma_color=float(color_sigma).hex(),
                                 sigma_spatial=float(options.get('sigma_spatial', 1)).hex(),
                                 color=sha(color.tobytes()), spatial=sha(spatial.tobytes()),
                                 spatial_count=spatial.size))
        return native_entry(field, maximum, window, cval, color, spatial)

    monkeypatch.setattr(_core, 'denoise_bilateral_constant', record)
    was_enabled = xx.set_fallback_enabled(False)
    try:
        if not native:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                owned(image, **options)
            xx.set_fallback_enabled(True)
        actual = observe(owned, image, options, observations)
    finally:
        xx.set_fallback_enabled(was_enabled)
    if native:
        if expected is not None:
            assert actual == {key: value for key, value in expected.items() if key not in ('id', 'native')}
        else:
            ref = observe(upstream, image, options, [])
            if name.endswith('constant-identity'):
                assert not observations
            else:
                assert len(observations) >= 1
            assert actual == {**ref, 'prepared': actual['prepared']}
    else:
        fallback_warning = [FallbackWarning.__module__, FallbackWarning.__name__,
                            'skimage.restoration.denoise_bilateral executed by scikit-image fallback']
        assert actual['warnings'].count(fallback_warning) == 1 and not observations
        actual['warnings'].remove(fallback_warning)
        if expected is not None:
            assert actual == {key: value for key, value in expected.items() if key not in ('id', 'native', 'prepared')} | {'prepared': []}
        else:
            ref = observe(upstream, image, options, [])
            assert actual == ref


@pytest.mark.parametrize('spec', list(RECIPES.case_specs()), ids=lambda row: row['id'])
def test_bilateral_fixed_recipe(spec, oracle):
    from skimage.restoration import denoise_bilateral as upstream
    from skimagexx._denoise_bilateral import denoise_bilateral as owned

    (image,), options = RECIPES.make_case(spec)
    assert xx.restoration.denoise_bilateral is owned
    previous = xx.set_fallback_enabled(False)
    try:
        actual = observe(xx.restoration.denoise_bilateral, image, options, [])
    finally:
        xx.set_fallback_enabled(previous)
    if oracle is not None:
        expected = next(row for row in oracle['benchmark'] if row['id'] == spec['id'])
        assert actual == {key: value for key, value in expected.items() if key not in ('id', 'prepared')} | {'prepared': []}
    else:
        assert actual == observe(upstream, image, options, [])


def test_bilateral_hot_swap():
    from skimage import restoration as original

    (image,), options = RECIPES.make_case(dict(shape=(9, 13, 3), dtype='float64', kwargs={'channel_axis': -1}))

    def apply(backend):
        return backend.restoration.denoise_bilateral(image, **options)

    marker = object()
    assert apply(SimpleNamespace(restoration=SimpleNamespace(denoise_bilateral=lambda *a, **k: marker))) is marker
    before = xx.set_fallback_enabled(False)
    try:
        np.testing.assert_array_equal(apply(xx), apply(SimpleNamespace(restoration=original)))
    finally:
        xx.set_fallback_enabled(before)


_DESCRIPTORS = json.loads(
    (Path(__file__).with_name('fixtures') / 'denoise_bilateral_descriptors_v1.json').read_bytes()
)


def _metadata(dtype):
    return dict(dtype.metadata) if dtype.metadata is not None else None


@pytest.mark.parametrize(
    'row', _DESCRIPTORS['rows'],
    ids=lambda row: f"{row['dtype']}-{row['channels']}-{row['case']}",
)
def test_bilateral_exact_descriptors(row):
    from skimagexx._denoise_bilateral import denoise_bilateral as owned

    plain = np.dtype(row['dtype'])
    if row['case'] == 'equivalent':
        descriptor = plain.newbyteorder('>').newbyteorder('=')
        assert descriptor == plain and descriptor is not plain
    elif row['case'] == 'tagged':
        descriptor = np.dtype(plain, metadata=row['in_metadata'])
    else:
        descriptor = plain
    channels = row['channels']
    values = ((np.arange(7 * 9 * channels) % 31) / 32).reshape(row['shape']).astype(plain)
    image = values.view(descriptor).copy()
    assert _metadata(image.dtype) == row['in_metadata']
    before = image.tobytes()
    function = xx.restoration.denoise_bilateral
    assert function is owned
    entry = _core.denoise_bilateral_constant
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    calls = []

    def profile(frame, event, value):
        if event == 'c_return' and value is entry:
            calls.append(value)

    prior_fallback = xx.set_fallback_enabled(False)
    prior_profile = sys.getprofile()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        try:
            sys.setprofile(profile)
            result = function(image, win_size=5, sigma_color=0.125, bins=32,
                              channel_axis=None if channels == 1 else -1)
        finally:
            sys.setprofile(prior_profile)
            xx.set_fallback_enabled(prior_fallback)

    assert calls == [entry] and type(result) is np.ndarray and not np.shares_memory(result, image)
    assert image.tobytes() == before and _metadata(image.dtype) == row['in_metadata']
    assert row['status'] == 'success'
    actual = dict(
        out_metadata=_metadata(result.dtype), out_dtype=result.dtype.str,
        result_sha256=sha(result.tobytes()), shape=list(result.shape), strides=list(result.strides),
        owns=result.flags.owndata, base_type=type(result.base).__name__,
        base_metadata=_metadata(result.base.dtype) if type(result.base) is np.ndarray else None,
        captures=[[w.category.__name__, str(w.message)] for w in observed],
    )
    assert actual == {key: row[key] for key in actual}


def test_bilateral_tagged_unsupported_inputs():
    descriptor = np.dtype('float32', metadata={'units': 'normalized', 'sample': 9})
    image = np.zeros((7, 9), dtype=descriptor)
    foreign = image.astype(descriptor.newbyteorder('S'))
    previous = xx.set_fallback_enabled(False)
    function = xx.restoration.denoise_bilateral
    try:
        for value, options in ((foreign, {}), (image[:, ::2], {}),
                               (image, {'mode': 'reflect'})):
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                function(value, **options)
    finally:
        xx.set_fallback_enabled(previous)
