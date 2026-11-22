import inspect
import sys
import warnings

import numpy as np
import pytest

from random_noise_reference import make_call, observe, reference, reference_profile, snapshot, typename


PROFILE = reference_profile(sys.platform, np.__version__)


@pytest.mark.parametrize('row', reference(PROFILE), ids=lambda row: row['id'])
def test_random_noise_exact_public(row):
    import skimagexx as xx
    from skimagexx.util import random_noise

    previous = xx.set_fallback_enabled(False)
    try:
        observe(np, random_noise, row)
    finally:
        xx.set_fallback_enabled(previous)


def test_random_noise_native_dispatch():
    import skimage.util as original
    import skimagexx as xx
    from skimagexx import _core, util

    function = util.random_noise
    assert function.__module__ == 'skimagexx._random_noise'
    normal = _core.random_noise_apply_normal
    impulse = _core.random_noise_apply_impulse
    assert all(inspect.isbuiltin(entry) and entry.__module__ == _core.__name__ for entry in (normal, impulse))
    calls = []
    stock_code = original.random_noise.__code__

    def witness(frame, event, value):
        if event == 'c_return' and (value is normal or value is impulse):
            calls.append(value)
        if event == 'call' and frame.f_code is stock_code:
            calls.append('stock')

    rows = {row['id']: row for row in reference(PROFILE)}
    previous = xx.set_fallback_enabled(False)
    profiler = sys.getprofile()
    try:
        sys.setprofile(witness)
        observe(np, function, rows['speckle-signed-cached-caller'])
        observe(np, function, rows['sp-interior'])
        assert calls == [normal, impulse]
    finally:
        sys.setprofile(profiler)
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('case', ['localvar-default', 'localvar-post-draw-broadcast-error'])
def test_random_noise_fallback_hotswap(case):
    import skimage.util as original
    import skimagexx as xx
    from skimagexx import _core, util

    row = next(row for row in reference(PROFILE) if row['id'] == case)
    assert not row['native'] and row['rng_before'] != row['rng_after']
    args, kwargs = make_call(np, row)
    image, = args
    rng = kwargs['rng']
    assert snapshot(np, image)[0] == row['input']
    native = (_core.random_noise_apply_normal, _core.random_noise_apply_impulse)
    stock_code = original.random_noise.__code__
    calls = []

    def witness(frame, event, value):
        if event == 'call' and frame.f_code is stock_code:
            calls.append((frame.f_locals['image'], frame.f_locals['rng']))
        if event == 'c_return' and any(value is entry for entry in native):
            calls.append('native')

    prior = xx.set_fallback_enabled(False)
    profiler = sys.getprofile()
    try:
        sys.setprofile(witness)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                util.random_noise(*args, **kwargs)
        assert not calls and not observed and rng.bit_generator.state == row['rng_before']

        xx.set_fallback_enabled(True)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            if 'error' in row:
                with pytest.raises(ValueError) as caught:
                    util.random_noise(*args, **kwargs)
                error = caught.value
                assert dict(type=typename(error), message=str(error)) == row['error']
            else:
                result = util.random_noise(*args, **kwargs)
                fields, raw = snapshot(np, result)
                assert fields == row['output'] and raw.hex() == row['data_hex']
                assert not np.shares_memory(image, result)
        assert len(calls) == 1 and calls[0][0] is image and calls[0][1] is rng
        assert type(observed[0].message) is xx.FallbackWarning
        assert str(observed[0].message) == 'skimage.util.random_noise executed by scikit-image fallback'
        assert [dict(type=typename(w.message), message=str(w.message)) for w in observed[1:]] == row['warnings']
        assert rng.bit_generator.state == row['rng_after']
        assert snapshot(np, image)[0] == row['input']
    finally:
        sys.setprofile(profiler)
        xx.set_fallback_enabled(prior)
