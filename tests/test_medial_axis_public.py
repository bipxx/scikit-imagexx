"""Frozen profile fixture, owned compiled topology and compatibility boundary."""

import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from medial_axis_cases import FIXED, array_snapshot, cases

FIXTURE = Path(__file__).with_name('fixtures') / 'medial_axis_originals_v1.json'
RAW = FIXTURE.read_bytes()
assert hashlib.sha256(RAW).hexdigest() == '952e27f23474acd7517c55b3cfd01163334958d74bcf18a00ebbaf38a2c54c40'
FROZEN = json.loads(RAW)
ROWS = {row['case_id']: row for row in FROZEN['minimum']}
if sys.platform == 'win32' or np.__version__ != '1.24.4':
    ROWS.update(FROZEN['windows_overrides'])

NATIVE = {
    *(spec['id'] for spec in FIXED),
    'tie-even-rectangle-seed-0', 'tie-even-rectangle-seed-1',
    'generator-first', 'generator-next-masked', 'explicit-bit-generator',
    'one-pixel-zero', 'one-pixel-one', 'row-strip', 'column-strip', 'all-zero', 'all-foreground',
}


@pytest.fixture
def api():
    from skimagexx import _compat, _core, morphology
    from skimagexx._medial_axis import medial_axis
    from skimage.morphology import medial_axis as original

    assert morphology.medial_axis is medial_axis
    assert inspect.signature(medial_axis) == inspect.signature(original)
    entries = (_core.medial_axis_prepare, _core.medial_axis_ordered)
    assert all(inspect.isbuiltin(entry) for entry in entries)
    previous = _compat.set_fallback_enabled(False)
    try:
        yield medial_axis, entries, _compat, original
    finally:
        _compat.set_fallback_enabled(previous)


@pytest.mark.parametrize('name,image,mask,distances,rng,rng_label',
                         list(cases({'fixed': FIXED}, np)), ids=list(ROWS))
def test_frozen_public_contract(api, name, image, mask, distances, rng, rng_label, monkeypatch):
    from scipy import ndimage

    public, entries, compat, original = api
    row = ROWS[name]
    assert array_snapshot(image, small=True) == row['input']
    assert mask is None or array_snapshot(mask, small=True) == row['mask']
    assert distances is row['return_distance'] and rng_label == row['rng']
    if name not in NATIVE:
        from skimage import morphology

        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            public(image, mask, distances, rng=rng)
        seen = []
        value = object()

        def provider(data, mask=None, return_distance=False, *, rng=None):
            seen.append((data, mask, return_distance, rng))
            return value

        monkeypatch.setattr(morphology, 'medial_axis', provider)
        compat.set_fallback_enabled(True)
        with pytest.warns(compat.FallbackWarning, match='skimage.morphology.medial_axis'):
            assert public(image, mask, distances, rng=rng) is value
        assert len(seen) == 1 and all(x is y for x, y in zip(seen[0], (image, mask, distances, rng), strict=True))
        return

    generator = rng.bit_generator if type(rng) is np.random.Generator else rng
    if row['state_before'] is not None:
        generator.state = row['state_before']
    calls = []
    edt = ndimage.distance_transform_edt.__code__

    def profile(frame, event, value):
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('strict candidate entered the stock public body')
        if event == 'call' and frame.f_code is edt:
            calls.append('edt')
        if event == 'c_return' and any(value is entry for entry in entries):
            calls.append(value.__name__)

    prior = sys.getprofile()
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        try:
            sys.setprofile(profile)
            output = public(image, mask, distances, rng=rng)
        finally:
            sys.setprofile(prior)
    assert not notes
    assert calls == ['medial_axis_prepare', 'edt', 'medial_axis_ordered']
    if row['state_after'] is not None:
        assert generator.state == row['state_after']
    if distances:
        assert type(output) is tuple and len(output) == 2
        skeleton, distance = output
        assert array_snapshot(distance, small=True) == row['distance']
        assert not np.shares_memory(skeleton, distance) and not np.shares_memory(distance, image)
        if mask is not None:
            assert (distance[~mask] == 0).all()
    else:
        skeleton = output
    assert array_snapshot(skeleton, small=True) == row['skeleton']
    assert not np.shares_memory(skeleton, image)
    assert array_snapshot(image, small=True) == row['input']
    if mask is not None:
        assert array_snapshot(mask, small=True) == row['mask']
        np.testing.assert_array_equal(skeleton[~mask], image[~mask])


def test_replaced_edt_honors_hot_swap_without_running_a_surrogate(api, monkeypatch):
    from scipy import ndimage
    from skimage import morphology

    public, _, compat, _ = api
    image = np.eye(4, dtype=bool)
    compat.set_fallback_enabled(True)
    marker = object()
    calls = []

    def provider(arr, mask, return_distance, *, rng):
        calls.append((arr, mask, return_distance, rng))
        return marker

    def unexpected(*args, **kwargs):
        pytest.fail('a replacement EDT unexpectedly entered the native path')

    monkeypatch.setattr(morphology, 'medial_axis', provider)
    monkeypatch.setattr(ndimage, 'distance_transform_edt', unexpected)
    with pytest.warns(compat.FallbackWarning, match='skimage.morphology.medial_axis'):
        assert public(image, rng=13) is marker
    assert len(calls) == 1 and calls[0][0] is image and calls[0][1:] == (None, False, 13)


def test_direct_topology_refuses_invalid_priority_before_deletions(api):
    _, (_, ordered), _, _ = api
    image = np.ones((2, 2), dtype=bool)
    distance = np.ones(image.shape)
    with pytest.raises(ValueError, match='unique foreground priorities'):
        ordered(image, distance, np.array([0, 1, 1, 3], dtype=np.int64))
    assert image.all()


def test_readonly_c_input_returns_independent_writable_outputs(api):
    public, _, _, _ = api
    image = np.array([[False, True, True, False]], dtype=bool)
    image.flags.writeable = False
    result, distance = public(image, return_distance=True, rng=23)
    assert result.flags.owndata and result.flags.writeable
    assert distance.flags.owndata and distance.flags.writeable
    np.testing.assert_array_equal(result, image)
    np.testing.assert_array_equal(distance, [[0., 1., 1., 0.]])
