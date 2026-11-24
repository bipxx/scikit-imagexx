"""Private numeric feasibility. The stock Cython exporter is a separate gate."""

import inspect
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _corner_moravec
from skimagexx._compat import FallbackWarning
from fixtures.corner_moravec_v1 import ADMITTED, CONTRACT, assert_frozen_response, make_input


@pytest.fixture(params=(1, 4))
def candidate(request):
    previous_threads = xx.get_num_threads()
    previous_fallback = xx.set_fallback_enabled(False)
    try:
        xx.set_num_threads(request.param)
        assert xx.get_num_threads() == request.param
        target = xx.feature.corner_moravec
        assert target is _corner_moravec.corner_moravec
        assert inspect.isbuiltin(xx._core.corner_moravec_u8)
        assert str(inspect.signature(target)) == CONTRACT['signature']
        yield target
    finally:
        xx.set_num_threads(previous_threads)
        xx.set_fallback_enabled(previous_fallback)


@pytest.mark.parametrize('case', ADMITTED)
def test_frozen_numeric_and_public_memoryview(candidate, case):
    image, kwargs = make_input(case)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual = candidate(image, **kwargs)
    assert not notices
    assert_frozen_response(case, image, actual)


def test_repeat_and_c_readonly_view(candidate):
    image, _ = make_input('readonly')
    borrowed = image.view()
    assert not borrowed.flags.owndata and not borrowed.flags.writeable
    first = candidate(borrowed, 2)
    second = candidate(borrowed, window_size=2)
    owners = [assert_frozen_response('readonly', borrowed, result) for result in (first, second)]
    assert owners[0] is not owners[1] and not np.shares_memory(first, second)


def test_deferred_inputs_use_policy(candidate):
    image, _ = make_input('minimal-size')
    for deferred, width in (
        (image.astype(np.float64), 1), (image[:, ::-1], 1),
        (np.asfortranarray(image), 1), (image.view(np.dtype('uint8', metadata={'x': 1})), 1),
        (image, True), (image, np.int64(1)), (image, 0), (image, 1 << 100),
        (image[:4], 1), (image, 2), (np.zeros((5, 321), np.uint8), 1),
    ):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            candidate(deferred, width)


def test_fallback_delivers_original_arguments(candidate, monkeypatch):
    from skimage import feature

    seen = []
    output = object()

    def provider(image, window_size):
        seen.append((image, window_size))
        return output

    monkeypatch.setattr(feature, 'corner_moravec', provider)
    xx.set_fallback_enabled(True)
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    width = np.int64(1)
    with pytest.warns(FallbackWarning, match='skimage.feature.corner_moravec'):
        assert candidate(image, width) is output
    args, = seen
    assert args[0] is image and args[1] is width
