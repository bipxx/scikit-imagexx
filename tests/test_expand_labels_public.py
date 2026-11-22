import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from expand_labels_reference import make_cases, sha, snapshot


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (ROOT / 'coverage/contracts/expand-labels-reference-v1.json').read_bytes()
assert sha(REFERENCE) == '445c534fab8a55acd6017031cc7ad3af20978c7e7b78f328f27d4ef74c81b2a2'
SAVED = {row['id']: row for row in json.loads(REFERENCE)['rows']}


@pytest.mark.parametrize('name,image,options,native', list(make_cases()), ids=list(SAVED))
def test_frozen_expand_labels(name, image, options, native, monkeypatch):
    from skimage.segmentation import expand_labels as original
    from skimagexx._expand_labels import expand_labels as owned

    entry = _core.expand_labels_integer
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    assert xx.segmentation.expand_labels is owned
    assert inspect.signature(owned) == inspect.signature(original)
    assert snapshot(image) == SAVED[name]['input']
    saved = SAVED[name]['output']
    assert native == SAVED[name]['native']
    entries = []

    def observe(*args):
        entries.append(args[0])
        return entry(*args)

    monkeypatch.setattr(_core, 'expand_labels_integer', observe)
    before = xx.set_fallback_enabled(False)
    try:
        if not native:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                owned(image, **options)
            xx.set_fallback_enabled(True)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            if 'error' in saved:
                with pytest.raises(Exception) as caught:
                    owned(image, **options)
                error = caught.value
                expected = saved['error']
                if name == 'wrong-distance-type' and int(np.__version__.split('.')[0]) == 1:
                    # NumPy 1 raises a built-in TypeError for this provider comparison.
                    with pytest.raises(TypeError) as current_original:
                        original(image, **options)
                    reference_error = current_original.value
                    expected = [type(reference_error).__module__, type(reference_error).__name__, str(reference_error)]
                assert [type(error).__module__, type(error).__name__, str(error)] == expected
            else:
                result = owned(image, **options)
                assert type(result) is np.ndarray
                assert snapshot(result) == saved['result']
                assert np.shares_memory(result, image) is saved['aliases']
        assert sum(item.category is FallbackWarning for item in notices) == int(not native)
        observed = [[item.category.__module__, item.category.__name__, str(item.message)]
                    for item in notices if item.category is not FallbackWarning]
        assert observed == saved['warnings']
    finally:
        xx.set_fallback_enabled(before)
    assert len(entries) == int(native)
    if native:
        assert entries[0] is image
    assert snapshot(image) == SAVED[name]['input']


def test_expand_labels_export_and_backend_swap():
    from skimage import segmentation as original
    from types import SimpleNamespace
    from skimagexx._expand_labels import expand_labels as owned

    exports = {}
    exec('from skimagexx.segmentation import *', exports)
    assert exports['expand_labels'] is xx.segmentation.expand_labels is owned
    labels = np.array([[0, 11, 0], [99, 0, 0], [0, 0, 5]], dtype=np.int64)

    def pipeline(backend):
        return backend.segmentation.expand_labels(labels, distance=1, spacing=[1, 1])

    marker = object()
    backend = SimpleNamespace(segmentation=SimpleNamespace(expand_labels=lambda *a, **k: marker))
    assert pipeline(backend) is marker
    before = xx.set_fallback_enabled(False)
    try:
        np.testing.assert_array_equal(pipeline(xx), pipeline(SimpleNamespace(segmentation=original)))
    finally:
        xx.set_fallback_enabled(before)


def test_tied_features_and_fractional_cutoffs():
    from skimage.segmentation import expand_labels as original

    rng = np.random.default_rng(83657)
    before = xx.set_fallback_enabled(False)
    try:
        # Exhaustive binary site placement with distinct nonzero label words.
        bits = 1 << np.arange(9).reshape(3, 3)
        for pattern in range(512):
            image = np.where(bits & pattern, bits, 0).astype(np.uint16)
            expected = original(image, distance=2, spacing=(1., 1.))
            np.testing.assert_array_equal(xx.segmentation.expand_labels(image, 2, (1., 1.)), expected)
        for case in range(600):
            rank = 1 + case % 3
            shape = tuple(rng.integers(1, 10, size=rank))
            labels = rng.integers(-3, 4, size=shape, dtype=np.int64)
            labels[rng.random(size=shape) < 0.63] = 0
            spacing = tuple(rng.choice([0.125, 0.7, 1.0, 1.3, 1.5, 4.], size=rank).tolist())
            radius = float(rng.uniform(0, 4))
            expected = original(labels, radius, spacing)
            np.testing.assert_array_equal(xx.segmentation.expand_labels(labels, radius, spacing), expected)
    finally:
        xx.set_fallback_enabled(before)
