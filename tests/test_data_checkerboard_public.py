"""Public parity and fresh-ownership tests for data.checkerboard."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _data_checkerboard


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'coverage/contracts/data-checkerboard-v1.json').read_bytes())


def test_checkerboard_public_parity():
    from skimage import data as original

    rep = CONTRACT['representation']
    target = xx.data.checkerboard
    assert target is _data_checkerboard.checkerboard
    assert inspect.isbuiltin(xx._core.data_checkerboard)
    assert str(inspect.signature(target)) == CONTRACT['signature']

    previous = xx.set_fallback_enabled(False)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            first = target()
            second = target()
            first.flat[0] ^= 255
            third = target()
            with pytest.raises(TypeError):
                target(None)
            with pytest.raises(TypeError):
                target(image=None)
        assert not captured
    finally:
        xx.set_fallback_enabled(previous)

    for result in (first, second, third):
        assert type(result) is np.ndarray and result.dtype.str == rep['dtype']
        assert list(result.shape) == rep['shape'] and list(result.strides) == rep['strides']
        assert result.base is None and result.flags.owndata and result.flags.writeable
        assert result.flags.c_contiguous and result.flags.aligned
    assert not np.shares_memory(first, second) and not np.shares_memory(second, third)
    assert sha256(second.tobytes()).hexdigest() == sha256(third.tobytes()).hexdigest() == rep['decoded_sha256']
    assert sha256(first.tobytes()).hexdigest() != rep['decoded_sha256']

    expected = original.checkerboard()
    np.testing.assert_array_equal(second, expected)
    assert sha256(expected.tobytes()).hexdigest() == rep['decoded_sha256']
