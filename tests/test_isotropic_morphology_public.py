import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from isotropic_morphology_reference import OPERATIONS, cases, observe, sha, snapshot


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (ROOT / 'coverage/contracts/isotropic-morphology-reference-v1.json').read_bytes()
assert sha(REFERENCE.replace(b'\r\n', b'\n')) == '8150ee7a320f79ab355fe4357c7cd1af4222421c9620978f0ac3d497f997fd9a'
SAVED = {row['id']: row for row in json.loads(REFERENCE)['rows']}


@pytest.mark.parametrize('name,kind,image,options,out,native', list(cases()), ids=list(SAVED))
def test_frozen_isotropic_morphology(name, kind, image, options, out, native, monkeypatch):
    from skimage import morphology as original
    from skimagexx import _isotropic_morphology as owned

    operation = 'isotropic_' + kind
    target = getattr(xx.morphology, operation)
    assert target is getattr(owned, operation)
    assert inspect.signature(target) == inspect.signature(getattr(original, operation))
    entry = _core.isotropic_morphology_mask
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    assert snapshot(image) == SAVED[name]['input']
    assert native == SAVED[name]['native']
    entries = []

    def record(*args):
        entries.append(args[0])
        return entry(*args)

    monkeypatch.setattr(_core, 'isotropic_morphology_mask', record)
    previous = xx.set_fallback_enabled(False)
    try:
        if not native:
            pristine_out = None if out is None else snapshot(out)
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                target(image, out=out, **options)
            assert (None if out is None else snapshot(out)) == pristine_out
            assert snapshot(image) == SAVED[name]['input']
            xx.set_fallback_enabled(True)
        actual = observe(target, image, options, out)
        fallback_warning = [FallbackWarning.__module__, FallbackWarning.__name__,
                            f'skimage.morphology.{operation} executed by scikit-image fallback']
        assert actual['warnings'].count(fallback_warning) == int(not native)
        if not native:
            actual['warnings'].remove(fallback_warning)
        assert actual == SAVED[name]['result']
    finally:
        xx.set_fallback_enabled(previous)
    assert len(entries) == int(native)
    if native:
        assert entries[0] is image


def test_isotropic_exports_and_backend_swap():
    from types import SimpleNamespace

    from skimage import morphology as original
    from skimagexx import _isotropic_morphology as owned

    exports = {}
    exec('from skimagexx.morphology import *', exports)
    mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)

    def pipeline(backend, name):
        return getattr(backend.morphology, name)(mask, 1.25, spacing=(0.625, 1.75))

    previous = xx.set_fallback_enabled(False)
    try:
        marker = object()
        for operation in OPERATIONS:
            name = 'isotropic_' + operation
            assert exports[name] is getattr(xx.morphology, name) is getattr(owned, name)
            standin = SimpleNamespace(morphology=SimpleNamespace(**{name: lambda *a, **k: marker}))
            assert pipeline(standin, name) is marker
            np.testing.assert_array_equal(pipeline(xx, name), pipeline(SimpleNamespace(morphology=original), name))
    finally:
        xx.set_fallback_enabled(previous)


def test_isotropic_edge_cutoffs():
    from skimage import morphology as original

    rng = np.random.default_rng(17371)
    previous = xx.set_fallback_enabled(False)
    try:
        for case in range(120):
            rank = 1 + case % 3
            shape = tuple(rng.integers(1, 8, size=rank))
            mask = rng.random(shape) < rng.uniform(0., 1.)
            spacing = tuple(rng.choice([0.125, 0.7, 1., 1.3, 2., 7.], size=rank).tolist())
            offset = rng.integers(0, 4, size=rank) * spacing
            radius = np.sqrt(np.sum(offset ** 2))
            cut = np.nextafter(radius, [0., radius, np.inf][case % 3]).item()
            for operation in OPERATIONS:
                name = 'isotropic_' + operation
                expected = getattr(original, name)(mask, cut, spacing=spacing)
                np.testing.assert_array_equal(getattr(xx.morphology, name)(mask, cut, spacing=spacing), expected)
    finally:
        xx.set_fallback_enabled(previous)


def test_isotropic_noncanonical_boolean_storage():
    from skimage import morphology as original

    image = np.array([0, 2, 255, 0, 128, 0], dtype=np.uint8).view(np.bool_)
    previous = xx.set_fallback_enabled(False)
    try:
        for operation in OPERATIONS:
            name = 'isotropic_' + operation
            for radius in (0., 1., 2.):
                expected = getattr(original, name)(image, radius)
                result = getattr(xx.morphology, name)(image, radius)
                assert snapshot(result) == snapshot(expected)
    finally:
        xx.set_fallback_enabled(previous)
