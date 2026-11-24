"""Bounded installed-region-table parity against the retained stock observations."""

import inspect

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _regionprops_table
from skimagexx._compat import FallbackWarning
from fixtures.regionprops_table_v1 import CASE_IDS, CONTRACT, assert_frozen_table, make_input


@pytest.fixture(params=(1, 4))
def public_native(request):
    previous_workers = xx.get_num_threads()
    previous_fallback = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    try:
        assert xx.get_num_threads() == request.param
        function = xx.measure.regionprops_table
        assert function is _regionprops_table.regionprops_table
        assert str(inspect.signature(function)) == CONTRACT['signature']
        assert inspect.isbuiltin(xx._core.regionprops_table_default)
        yield function
    finally:
        xx.set_num_threads(previous_workers)
        xx.set_fallback_enabled(previous_fallback)


@pytest.mark.parametrize('case', CASE_IDS)
def test_frozen_default_table(case, public_native):
    image = make_input(case)
    assert_frozen_table(case, image, public_native(image))


@pytest.mark.parametrize('case', ('all-background', 'gapped-disconnected'))
def test_explicit_defaults_and_independent_storage(case, public_native):
    image = make_input(case)
    first = public_native(image)
    second = public_native(image, None, tuple(['label', 'bbox']), cache=True,
                           separator='-', extra_properties=None, spacing=None)
    assert first is not second
    bases_first = assert_frozen_table(case, image, first)
    bases_second = assert_frozen_table(case, image, second)
    assert all(a is not b and not np.shares_memory(a, b) for a in bases_first for b in bases_second)


def test_dtype_alias_and_strict_compatibility(public_native):
    image = make_input('gapped-disconnected')
    alias = image.dtype.newbyteorder('S').newbyteorder('=')
    assert_frozen_table('gapped-disconnected', image, public_native(image.view(alias)))
    for deferred in (image[:, ::-1], image.astype(np.uint16), image.astype(image.dtype.newbyteorder('S')),
                     image.view(np.dtype('int32', metadata={'x': 1})),
                     image - 8, image + 4096, np.zeros((1, 1025), np.int32)):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            public_native(deferred)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public_native(image, properties=('bbox', 'label'))
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public_native(image, cache=False)


def test_compatibility_keeps_original_arguments(public_native, monkeypatch):
    from skimage import measure

    image = make_input('one-pixel')
    returned = object()
    received = []

    def original(*args, **kwargs):
        received.append((args, kwargs))
        return returned

    monkeypatch.setattr(measure, 'regionprops_table', original)
    xx.set_fallback_enabled(True)
    properties = ['label']
    with pytest.warns(FallbackWarning, match='skimage.measure.regionprops_table'):
        assert public_native(image, properties=properties, separator='/') is returned
    (args, kwargs), = received
    assert args[0] is image and args[1] is None and args[2] is properties
    assert kwargs == dict(cache=True, separator='/', extra_properties=None, spacing=None)
