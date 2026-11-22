"""Stock-fixture parity at the find_contours public wrapper and native entry."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from find_contours_cases import TIMED, cases, input_snapshot, output_snapshot
from find_contours_comparison import assert_contours_equal

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/find_contours_originals_v1.json'
RAW = FIXTURE.read_bytes()
assert hashlib.sha256(RAW).hexdigest() == 'e9ed43707672f02f7b99f55fda355d345fadee5fb1514943003f16b4ed177b5a'
ROWS = json.loads(RAW)


def parameters():
    for name, image, options, native in cases(np):
        mask = options.get('mask')
        if mask is not None:
            # A later recipe makes the same mask read-only. Snapshot each
            # case's descriptor before pytest collects the next recipe.
            options = {**options, 'mask': mask.view()}
        yield name, image, options, native


@pytest.fixture
def api():
    from skimagexx import _core, _compat, measure
    from skimagexx._find_contours import find_contours as wrapper

    assert measure.find_contours is wrapper
    entry = _core.find_contours_finite
    assert inspect.isbuiltin(entry)
    previous = _compat.set_fallback_enabled(False)
    try:
        yield measure.find_contours, entry, _compat
    finally:
        _compat.set_fallback_enabled(previous)


@pytest.mark.parametrize('name,image,options,native', list(parameters()), ids=list(ROWS))
def test_frozen_public_and_compiled_routes(api, name, image, options, native):
    from skimage.measure import find_contours as original

    public, entry, _ = api
    row = ROWS[name]
    assert inspect.signature(public) == inspect.signature(original)
    assert input_snapshot(image) == row['input'] and native is row['native']
    if 'mask' in options:
        assert input_snapshot(options['mask']) == row['options']['mask']
    called = []

    def observe(frame, event, value):
        if event == 'c_return' and value is entry:
            called.append(value)
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('the strict candidate unexpectedly called original')

    prior = sys.getprofile()
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        try:
            sys.setprofile(observe)
            if native:
                output = public(image, **options)
            else:
                with pytest.raises(NotImplementedError, match='fallback is disabled'):
                    public(image, **options)
        finally:
            sys.setprofile(prior)
    assert not notes and called == [entry]
    assert input_snapshot(image) == row['input']
    if 'mask' in options:
        assert input_snapshot(options['mask']) == row['options']['mask']
    if not native:
        return
    observed = output_snapshot(output, hashlib.sha256())
    if name in TIMED:
        packed = json.dumps(observed['contours'], separators=(',', ':')).encode()
        observed = {'type':observed['type'],'count':observed['count'],'total_points':sum(len(c) for c in output),
                    'ordered_metadata_sha256':hashlib.sha256(packed).hexdigest()}
    assert observed == row['output']


def test_private_numpy_owners_and_hot_swap(api, monkeypatch):
    from skimage import measure as original

    public, _, compat = api
    image = np.array([[0., 1.], [1., 0.]])
    low = public(image, 0.5)
    reversed_views = public(image, 0.5, positive_orientation='high')
    assert len(low) == len(reversed_views) == 2
    assert all(a.flags.owndata and a.base is None for a in low)
    assert all(type(a.base) is np.ndarray and a.base.flags.owndata and a.strides == (-16, 8)
               for a in reversed_views)
    assert_contours_equal(reversed_views, [a[::-1] for a in low])
    for left, right in zip(low, reversed_views):
        assert not np.shares_memory(left, right)
    assert not np.shares_memory(low[0], low[1])

    seen = []
    marker = object()

    def provider(array, level, connectivity, orientation, *, mask):
        seen.append((array, level, connectivity, orientation, mask))
        return marker

    monkeypatch.setattr(original, 'find_contours', provider)
    compat.set_fallback_enabled(True)
    view = image[::-1]
    with pytest.warns(compat.FallbackWarning, match='skimage.measure.find_contours'):
        assert public(view, 0.5, 'high', 'high') is marker
    assert len(seen) == 1 and seen[0][0] is view and seen[0][1:] == (0.5, 'high', 'high', None)
    assert len(public(image, 0.5)) == 2 and len(seen) == 1


def test_storage_alignment_and_options_refuse_before_native_read(api):
    public, _, _ = api
    plain = np.zeros((2, 3))
    backing = bytearray(plain.nbytes + 1)
    misaligned = np.ndarray(plain.shape, dtype='float64', buffer=backing, offset=1)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(misaligned, 0.5)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(plain, np.float32(0.5))
    options = dict(level=0.5, mask=np.ones(plain.shape, 'bool')[:, ::-1])
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(plain, **options)


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'uint8'])
def test_equivalent_native_image_and_mask_descriptors(api, dtype):
    public, _, _ = api
    image = np.array([[0, 1], [1, 0]], dtype=dtype)
    mask = np.ones(image.shape, dtype=bool)
    # The generic descriptor test round-trips native byte order. Request a
    # distinct untagged descriptor for one-byte dtypes as well.
    image_dtype = np.dtype(dtype, copy=True).newbyteorder('>').newbyteorder('=')
    mask_dtype = np.dtype('bool', copy=True).newbyteorder('=')
    assert image_dtype == image.dtype and image_dtype is not image.dtype
    assert mask_dtype == mask.dtype and mask_dtype is not mask.dtype
    image_view, mask_view = image.view(image_dtype), mask.view(mask_dtype)
    options = {'level': 0.5, 'positive_orientation': 'high'}
    canonical = public(image, mask=mask, **options)
    equivalent = public(image_view, mask=mask_view, **options)
    assert_contours_equal(equivalent, canonical)
    assert output_snapshot(equivalent, hashlib.sha256()) == output_snapshot(canonical, hashlib.sha256())
    assert image_view.dtype is image_dtype and mask_view.dtype is mask_dtype


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'uint8', 'bool'])
def test_tagged_descriptors_follow_fallback_policy(api, monkeypatch, dtype):
    from skimage import measure as original

    public, entry, compat = api
    image = np.array([[0, 1], [1, 0]], dtype='float64' if dtype == 'bool' else dtype)
    mask = np.ones(image.shape, dtype=bool)
    tagged = np.dtype(dtype, metadata={'contour_descriptor': True})
    if dtype == 'bool':
        mask = mask.view(tagged)
    else:
        image = image.view(tagged)
    assert entry(image, 0.5, 'low', 'high', mask) is None
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(image, 0.5, 'low', 'high', mask=mask)

    marker = object()

    def provider(image_arg, level, connectivity, orientation, *, mask):
        assert image_arg is image and mask is mask_arg
        assert (level, connectivity, orientation) == (0.5, 'low', 'high')
        return marker

    mask_arg = mask
    monkeypatch.setattr(original, 'find_contours', provider)
    compat.set_fallback_enabled(True)
    with pytest.warns(compat.FallbackWarning, match='skimage.measure.find_contours'):
        assert public(image, 0.5, 'low', 'high', mask=mask) is marker


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_swapped_image_descriptor_stays_unsupported(api, dtype):
    public, entry, _ = api
    image = np.array([[0, 1], [1, 0]], dtype=np.dtype(dtype).newbyteorder('S'))
    assert not image.dtype.isnative
    assert entry(image, 0.5, 'low', 'low', None) is None
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(image, 0.5)


def test_contour_comparison_rejects_structural_and_coordinate_changes():
    first = np.array([[0., -0.], [1., 2.]])[::-1]
    second = np.array([[1., 3.], [1.5, 4.], [2., 4.5]])
    reference = [first, second]
    assert_contours_equal([first.copy(), second.copy()], reference)
    assert_contours_equal([], [])

    changed_zero = first.copy()
    changed_zero[1, 1] = 0.
    for changed, message in (
        (tuple(reference), 'result must be a list'),
        ([first], 'count differs'),
        ([first.tolist(), second], 'must be an ndarray'),
        ([second, first], 'shape differs'),
        ([first.astype('float32'), second], 'must use native float64'),
        ([first.view(np.dtype('float64', metadata={'tag': True})), second], 'must use an untagged dtype'),
        ([first[::-1], second], 'coordinate words differ'),
        ([changed_zero, second], 'coordinate words differ'),
    ):
        with pytest.raises(AssertionError, match=message):
            assert_contours_equal(changed, reference)
