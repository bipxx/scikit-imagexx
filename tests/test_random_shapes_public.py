import inspect
import sys
import warnings

import numpy as np
import pytest

from random_shapes_reference import observe, reference, reference_profile


PROFILE = reference_profile(sys.platform, np.__version__)


@pytest.mark.parametrize('row', reference(PROFILE), ids=lambda row: row['id'])
def test_random_shapes_exact_public(row):
    import skimagexx as xx
    from skimagexx.draw import random_shapes

    previous = xx.set_fallback_enabled(False)
    try:
        observe(np, random_shapes, row)
    finally:
        xx.set_fallback_enabled(previous)


def test_random_shapes_native_dispatch():
    import skimage.draw as original
    import skimagexx as xx
    from skimagexx import _core, draw

    function = draw.random_shapes
    assert function.__module__ == 'skimagexx._random_shapes'
    paint = _core.random_shapes_paint
    assert inspect.isbuiltin(paint) and paint.__module__ == _core.__name__
    calls = []
    stock_code = original.random_shapes.__code__

    def witness(frame, event, value):
        if event == 'c_return' and value is paint:
            calls.append(paint)
        if event == 'call' and frame.f_code is stock_code:
            calls.append('stock')

    row = next(row for row in reference(PROFILE) if row['id'] == 'invalid-channel-axis-after-paint')
    previous = xx.set_fallback_enabled(False)
    profiler = sys.getprofile()
    try:
        sys.setprofile(witness)
        observe(np, function, row)
        assert calls and all(call is paint for call in calls)
    finally:
        sys.setprofile(profiler)
        xx.set_fallback_enabled(previous)


def test_random_shapes_fallback_hotswap():
    import skimage.draw as original
    import skimagexx as xx
    from skimagexx import _core, draw

    row = next(row for row in reference(PROFILE) if row['id'] == 'rectangle-gray-passed-equivalent')
    paint = _core.random_shapes_paint
    stock_code = original.random_shapes.__code__
    calls = []

    def witness(frame, event, value):
        if event == 'call' and frame.f_code is stock_code:
            calls.append((frame.f_locals['image_shape'], frame.f_locals['rng']))
        if event == 'c_return' and value is paint:
            calls.append('native')

    def forwarded(image_shape, count, **options):
        # Stock reads the first two dimensions as scalars; the list forces delegation.
        shape = list(image_shape)
        rng = options['rng']
        previous = xx.set_fallback_enabled(False)
        try:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                with pytest.raises(NotImplementedError, match='fallback is disabled'):
                    draw.random_shapes(shape, count, **options)
            assert not calls and not observed and rng.bit_generator.state == row['rng_before']
            xx.set_fallback_enabled(True)
            with pytest.warns(xx.FallbackWarning, match='^skimage.draw.random_shapes executed by scikit-image fallback$'):
                result = draw.random_shapes(shape, count, **options)
            assert len(calls) == 1 and calls[0][0] is shape and calls[0][1] is rng
            return result
        finally:
            xx.set_fallback_enabled(previous)

    profiler = sys.getprofile()
    try:
        sys.setprofile(witness)
        observe(np, forwarded, row)
    finally:
        sys.setprofile(profiler)
