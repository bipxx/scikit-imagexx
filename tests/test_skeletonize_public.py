"""Pinned stock observations, compiled ownership and compatibility routing."""

import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from skeletonize_cases import cases, snapshot

FIXTURE = Path(__file__).with_name('fixtures') / 'skeletonize_originals_v1.json'
DATA = FIXTURE.read_bytes()
assert hashlib.sha256(DATA).hexdigest() == '1d7dc129321b5690b16ea6e4fb60d379c01688b295515522bf68ed7ef45b032e'
BUNDLE = json.loads(DATA)
ROWS = {row['id']: row for row in BUNDLE['fixed'] + BUNDLE['controls']}
COMPAT = {'strided', 'fortran', 'empty', 'rank-one', 'invalid-method', 'zhang-3d'}


@pytest.fixture
def api():
    from skimagexx import _compat, _core, morphology
    from skimagexx._skeletonize import skeletonize
    from skimage.morphology import skeletonize as stock

    assert morphology.skeletonize is skeletonize
    assert inspect.signature(skeletonize) == inspect.signature(stock)
    assert inspect.isbuiltin(_core.skeletonize_zhang)
    previous = _compat.set_fallback_enabled(False)
    try:
        yield skeletonize, _core.skeletonize_zhang, _compat, stock
    finally:
        _compat.set_fallback_enabled(previous)


@pytest.mark.parametrize('name,image,method', list(cases(np)), ids=list(ROWS))
def test_frozen_public_contract(api, name, image, method, monkeypatch):
    public, entry, compat, stock = api
    row = ROWS[name]
    assert snapshot(image) == row['input'] and method == row['method']
    assert row['warnings'] == []
    events = []

    def profile(frame, event, value):
        if event == 'call' and frame.f_code is stock.__code__:
            raise AssertionError('unexpected original execution')
        if event in {'c_return', 'c_exception'} and value is entry:
            events.append(event)

    prior = sys.getprofile()
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        try:
            sys.setprofile(profile)
            if name in COMPAT:
                with pytest.raises(NotImplementedError, match='fallback is disabled'):
                    public(image, method=method)
            elif name == 'readonly':
                with pytest.raises(ValueError) as error:
                    public(image, method=method)
                assert str(error.value) == row['message']
            else:
                output = public(image, method=method)
        finally:
            sys.setprofile(prior)
    assert not notes
    assert events == (['c_exception'] if name == 'readonly' else ['c_return'])
    assert snapshot(image) == row['input']
    if name == 'readonly':
        return
    if name in COMPAT:
        from skimage import morphology

        expected = object()
        captured = []

        def provider(array, *, method=None):
            captured.append((array, method))
            return expected

        monkeypatch.setattr(morphology, 'skeletonize', provider)
        compat.set_fallback_enabled(True)
        with pytest.warns(compat.FallbackWarning, match='skimage.morphology.skeletonize'):
            assert public(image, method=method) is expected
        assert len(captured) == 1 and captured[0][0] is image and captured[0][1] == method
        return
    assert not np.shares_memory(output, image)
    assert snapshot(output) == row['output']
    if 'output_hex' in row:
        assert output.tobytes().hex() == row['output_hex']


@pytest.mark.parametrize('shape,method', [((5, 7), 'lee'), ((2, 5, 7), 'lee'), ((2, 5, 7), None)])
def test_lee_routes_without_claiming_native_work(api, monkeypatch, shape, method):
    from skimage import morphology

    public, _, compat, _ = api
    image = np.zeros(shape, dtype=bool)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(image, method=method)
    calls = []
    marker = object()

    def provider(array, *, method=None):
        calls.append((array, method))
        return marker

    monkeypatch.setattr(morphology, 'skeletonize', provider)
    compat.set_fallback_enabled(True)
    with pytest.warns(compat.FallbackWarning, match='skimage.morphology.skeletonize'):
        assert public(image, method=method) is marker
    assert len(calls) == 1 and calls[0][0] is image and calls[0][1] == method
    assert public(np.zeros((2, 3), dtype=bool)).flags.owndata and len(calls) == 1


def test_storage_refusal_does_not_cross_the_provider_boundary(api, monkeypatch):
    from skimage import morphology

    public, _, compat, _ = api
    # This is a small correctly sized buffer with an unrecognized owner.
    image = np.frombuffer(bytearray(9), dtype=bool).reshape(3, 3)
    compat.set_fallback_enabled(True)

    def unexpected(*args, **kwargs):
        pytest.fail('unverified owner passed to compatibility')

    monkeypatch.setattr(morphology, 'skeletonize', unexpected)
    with pytest.raises(ValueError, match='requires bounded input storage'):
        public(image)


def test_readonly_uint8_casts_to_independent_boolean_storage(api):
    public, _, _, _ = api
    image = np.array([[7, 0, 245, 3]], dtype=np.uint8)
    image.flags.writeable = False
    result = public(image, method='zhang')
    assert result.dtype == np.dtype(bool) and result.flags.owndata and result.flags.writeable
    assert result.tolist() == [[True, False, True, True]]


def test_live_traversal_keeps_survivors_across_deleted_runs(api):
    public, _, _, _ = api
    image = np.zeros((13, 67), dtype=bool)
    image[:6, :8] = True
    image[2:9, 11:16] = True
    image[7:, 57:] = True
    image[0, 65] = image[6, 30] = image[12, 0] = image[6, 66] = True
    before = snapshot(image)
    # Frozen 256-class table with simultaneous deletions per phase.
    expected = [
        [0, 65], [2, 3], [2, 4], [2, 5], [3, 1], [3, 2], [3, 14],
        [4, 13], [5, 13], [6, 12], [6, 30], [7, 12],
        [9, 60], [9, 61], [9, 62], [9, 63], [9, 64], [10, 58], [10, 59], [12, 0],
    ]
    assert np.argwhere(public(image, method='zhang')).tolist() == expected
    assert snapshot(image) == before
