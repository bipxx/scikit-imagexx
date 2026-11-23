"""Private and installed front-end checks against the authenticated stock capture."""
from base64 import urlsafe_b64encode
from hashlib import sha256
from importlib.metadata import distribution
import inspect
from pathlib import Path
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


def _same_host_minimum_laplacian(reference):
    """Resolve only the stock rows whose residual words vary across minimum hosts."""
    if (reference['case'] not in {
        'laplacian-rgb-u8-reflect-three-reductions',
        'laplacian-max-zero-unsigned-preserve',
        'laplacian-generator-throw',
    } or sys.platform != 'linux' or sys.version_info[:3] != (3, 11, 17)
            or np.__version__ != '1.24.4'):
        return None

    import scipy
    import skimage

    if (scipy.__version__, skimage.__version__) != ('1.15.3', '0.26.0'):
        return None

    from numpy.core import _multiarray_umath
    from scipy.ndimage import _filters, _interpolation, _nd_image, _ni_support
    from skimage import transform as original
    from skimage._shared import filters
    from skimage.transform import _warps, pyramids
    from skimage.util import dtype

    # Authenticate the actual loaded stock modules against the witnessed wheel
    # members and their installed RECORD, independent of CPU dispatch.
    providers = {
        'numpy': ((_multiarray_umath, 'db283346a443cde0c8cb4dcdde56490714d2c34e3ad16fe2c30ae8c20458fc86'),),
        'scipy': (
            (_filters, 'e821fbd5ae150dc9fdb616ae5a213504904e5416febcde421739f3fe52769c0c'),
            (_interpolation, '6656f865125b4ceadfdb575e74edbc1874d70342a1f613020d605d1af45b728e'),
            (_nd_image, '56d6740b800583fa54adba6594ef8562832f86d156435527c4a1ca8584653bbe'),
            (_ni_support, 'c1e60b92002969fd161b9e1d92cc499c5118d93a024fd3b75cd3f8778a69a453'),
        ),
        'scikit-image': (
            (pyramids, 'b3a784bae6957d3ee09b653ba0f63e21f2ab69d7aca15c1329c8c5a039047a65'),
            (_warps, '3749a6db262c7091285daea455c655cfc228478de3eb7bd78c4d0e4564e2e09e'),
            (filters, 'c906451edf7b3fc5f242e679a84915c414f5291497a1605b0507f63629924f42'),
            (dtype, 'd4f99f88966e4d0dbbe80a3c0b8acb3890644cc5fbfb707ba9b5a90e9119e627'),
        ),
    }
    for name, modules in providers.items():
        dist = distribution(name)
        records = {str(member): member for member in dist.files}
        site = Path(dist.locate_file('')).resolve()
        for module, digest in modules:
            path = Path(module.__file__).resolve()
            member = records[path.relative_to(site).as_posix()]
            raw = path.read_bytes()
            assert member.size == len(raw) and member.hash.mode == 'sha256'
            assert member.hash.value == urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
            assert sha256(raw).hexdigest() == digest
    assert reference['api'] == 'pyramid_laplacian' and reference['native']
    assert original.pyramid_laplacian is pyramids.pyramid_laplacian
    assert inspect.getsourcefile(original.pyramid_laplacian) == pyramids.__file__
    return original.pyramid_laplacian


def _with_live_payload(frozen, live):
    """Keep every array metadata field sourced from the frozen observation."""
    assert frozen.keys() == live.keys()
    return dict(frozen, **{key: live[key] for key in ('sha256', 'hex') if key in frozen})


def _check_generator_trace(function, recipe, reference, native, *, stock=False, live_outputs=None, stock_code=None):
    image = make_image(np, recipe)
    assert snapshot(image) == reference['input']
    returned = []
    visited = []

    def witness(frame, event, arg):
        if event == 'call' and frame.f_code is stock_code:
            raise AssertionError('the strict route entered the stock public generator')
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
                expected = dict(step, compiled=[]) if stock else step
                if step['result'] == 'yield' and (stock or live_outputs is not None):
                    assert actual['result'] == 'yield'
                    if stock:
                        live_array = actual['array']
                    else:
                        peer = live_outputs[len(returned) - 1]
                        assert result.tobytes() == peer.tobytes()
                        live_array = snapshot(peer)
                    expected = dict(expected, array=_with_live_payload(step['array'], live_array))
                assert actual == expected
    finally:
        sys.setprofile(old_profile)
    assert snapshot(image) == reference['final_input']
    expected_outputs = reference['final_outputs']
    if stock or live_outputs is not None:
        peers = returned if stock else live_outputs
        expected_outputs = [_with_live_payload(frozen, snapshot(peer))
                            for frozen, peer in zip(expected_outputs, peers, strict=True)]
    assert [snapshot(out) for out in returned] == expected_outputs
    return returned


@pytest.mark.parametrize('recipe, reference', FROZEN, ids=[row['case'] for _, row in FROZEN])
def test_frozen_pyramid_generator(recipe, reference, strict):
    native = (_core.gaussian_nd, _core.resize_zoom_typed)
    assert all(inspect.isbuiltin(entry) for entry in native)
    function = getattr(transform, reference['api'])
    stock = _same_host_minimum_laplacian(reference)
    if stock is None:
        _check_generator_trace(function, recipe, reference, native)
    else:
        live_outputs = _check_generator_trace(stock, recipe, reference, native, stock=True)
        _check_generator_trace(function, recipe, reference, native, live_outputs=live_outputs,
                               stock_code=stock.__code__)


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
