"""Private and installed front-end checks against the authenticated stock capture."""
import inspect
import sys
from types import ModuleType
import warnings

import numpy as np
import pytest

import skimagexx
from skimagexx import _core, transform
from skimagexx._compat import FallbackWarning

from pyramid_generator_cases import make_image, rows
from pyramid_spatial_cases import array_state, fixture


def snapshot(array):
    value = array_state(array)
    bases = []
    base = array.base
    while isinstance(base, np.ndarray):
        bases.append(array_state(base, with_bytes=False))
        base = base.base
    value.update(type=type(array).__module__ + '.' + type(array).__qualname__,
                 dtype_name=array.dtype.name, aligned=bool(array.flags.aligned),
                 base_chain=bases, terminal_base=type(base).__name__)
    return value


@pytest.fixture
def strict():
    prior = skimagexx.set_fallback_enabled(False)
    try:
        yield
    finally:
        skimagexx.set_fallback_enabled(prior)


FROZEN = list(rows())


@pytest.mark.parametrize('name', ['pyramid_gaussian', 'pyramid_laplacian'])
def test_public_generator_export(name, strict):
    from skimagexx import _pyramid_generators

    function = getattr(transform, name)
    assert function is getattr(_pyramid_generators, name)
    assert inspect.isgeneratorfunction(function)
    assert name in transform.__all__


@pytest.mark.parametrize('recipe, reference', FROZEN, ids=[row['case'] for _, row in FROZEN])
def test_frozen_pyramid_generator(recipe, reference, strict):
    image = make_image(np, recipe)
    assert snapshot(image) == reference['input']
    function = getattr(transform, reference['api'])
    returned = []
    native = (_core.gaussian_nd, _core.resize_zoom_typed)
    assert all(inspect.isbuiltin(entry) for entry in native)
    visited = []

    def witness(frame, event, arg):
        if event == 'c_return' and any(arg is entry for entry in native):
            visited.append(arg.__name__)

    old_profile = sys.getprofile()
    sys.setprofile(witness)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            gen = function(image, **recipe['options'])
            assert inspect.isgenerator(gen) and inspect.getgeneratorstate(gen) == 'GEN_CREATED'
            assert iter(gen) is gen and not notices and not visited
            if not reference['native']:
                with pytest.raises(NotImplementedError, match='fallback is disabled'):
                    next(gen)
                assert inspect.getgeneratorstate(gen) == 'GEN_CLOSED'
                assert not visited and not notices and snapshot(image) == reference['input']
                return
            marker = ValueError('probe marker')
            for step in reference['steps']:
                assert inspect.getgeneratorstate(gen) == step['state_before']
                visited.clear()
                notices.clear()
                action = step['action']
                actual = dict(action=action, state_before=inspect.getgeneratorstate(gen))
                try:
                    if action == 'send-non-none-before-start':
                        result = gen.send(1)
                    elif action == 'send-none':
                        result = gen.send(None)
                    elif action == 'close':
                        result = gen.close()
                    elif action == 'throw-marker':
                        result = gen.throw(marker)
                    elif action == 'mutate-input-before-first-next':
                        result = image.__setitem__((0, 0), 17)
                    elif action == 'mutate-last-yield':
                        last = returned[-1]
                        result = last.__setitem__((0, 0), last[0, 0] + 0.5)
                        actual.update(modified_layer=len(returned)-1, modified_output=snapshot(last))
                    else:
                        assert action in ('next', 'next-after-terminal', 'next-after-terminal-again')
                        result = next(gen)
                except StopIteration as end:
                    actual.update(result='stop', value=end.value)
                except Exception as error:
                    actual.update(result='error', error=dict(type=type(error).__qualname__, message=str(error),
                                                             is_throw_marker=error is marker))
                else:
                    if isinstance(result, np.ndarray):
                        actual.update(result='yield', array=snapshot(result), identical_input=result is image,
                                      shares_input=bool(np.shares_memory(result, image)),
                                      identical_prior=[result is prior for prior in returned],
                                      shares_prior=[bool(np.shares_memory(result, prior)) for prior in returned])
                        returned.append(result)
                    else:
                        actual.update(result='return', value=result)
                actual.update(state_after=inspect.getgeneratorstate(gen), input_after=snapshot(image),
                              warnings=[[note.category.__module__ + '.' + note.category.__qualname__, str(note.message)]
                                        for note in notices], compiled=list(visited))
                assert actual == step
    finally:
        sys.setprofile(old_profile)
    assert snapshot(image) == reference['final_input']
    assert [snapshot(out) for out in returned] == reference['final_outputs']


@pytest.mark.parametrize('name', ['pyramid_gaussian', 'pyramid_laplacian'])
def test_unsupported_generator_delegates_on_first_next(name, monkeypatch):
    providers = ModuleType('skimage.transform')
    image = fixture((5, 7), 'float32', np)
    visits = []

    def original(current, *args, **kwargs):
        visits.append((current, args, kwargs))
        yield current

    setattr(providers, name, original)
    monkeypatch.setitem(sys.modules, 'skimage.transform', providers)
    prior = skimagexx.set_fallback_enabled(False)
    try:
        fn = getattr(transform, name)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            gen = fn(image, downscale=1.25)
        assert not notices
        assert not visits and inspect.getgeneratorstate(gen) == 'GEN_CREATED'
        skimagexx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match=name):
            assert next(gen) is image
        assert len(visits) == 1 and visits[0][0] is image
        assert visits[0][1] == (-1, 1.25, None, 1, 'reflect', 0, False)
        assert visits[0][2] == {'channel_axis': None}
        with pytest.raises(StopIteration):
            next(gen)
        assert inspect.getgeneratorstate(gen) == 'GEN_CLOSED'
        with pytest.raises(StopIteration):
            next(gen)
        assert len(visits) == 1
    finally:
        skimagexx.set_fallback_enabled(prior)


def test_gaussian_delegates_reduction_on_the_live_edited_array(monkeypatch):
    providers = ModuleType('skimage.transform')
    image = fixture((5, 7), 'float32', np)
    seen = []

    def original(current, *args, **kwargs):
        seen.append(current)
        return current[:3, :4].copy()

    providers.pyramid_reduce = original
    monkeypatch.setitem(sys.modules, 'skimage.transform', providers)
    prior = skimagexx.set_fallback_enabled(True)
    try:
        gen = transform.pyramid_gaussian(image, max_layer=1)
        live = next(gen)
        assert live is image and not seen
        live[0, 0] = np.float32(131072)
        with pytest.warns(FallbackWarning, match='pyramid_reduce'):
            readback = next(gen)
        assert len(seen) == 1 and seen[0] is live
        assert readback.shape == (3, 4) and readback[0, 0] == 131072
        with pytest.raises(StopIteration):
            next(gen)
    finally:
        skimagexx.set_fallback_enabled(prior)
