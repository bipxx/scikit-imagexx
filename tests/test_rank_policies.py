"""Stateful NumPy dtype-provider parity; no upstream rank forwarding."""
import importlib
import importlib.util
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
from skimage.filters import rank as reference

if os.environ.get('SKIMAGEXX_RANK_CANDIDATE'):
    spec = importlib.util.spec_from_file_location('skimagexx._rank_policy_candidate', os.environ['SKIMAGEXX_RANK_CANDIDATE'])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native._core = importlib.import_module(os.environ['SKIMAGEXX_RANK_BACKEND'])
else:
    native = importlib.import_module('skimagexx._rank')

NAMES = ('minimum', 'maximum', 'mean', 'sum', 'pop')


def record(array):
    return (array.dtype.str, array.shape, array.strides, array.tobytes(), array.flags.writeable)


def initialize_extended_padding(array):
    # x87 extended floats occupy ten meaningful bytes in sixteen-byte slots.
    # NumPy scalar assignment/fill may copy uninitialized stack padding. Give
    # each independently constructed input and callback write identical bytes;
    # keep full byte-for-byte assertions, including those padding bytes.
    component = array.real.dtype
    if component.kind=='f' and component.itemsize==16 and np.finfo(component).nmant==63:
        array.view(np.uint8).reshape(-1,16)[:,10:]=0


def make_arrays(dtype, stage, signal):
    image = np.full((3, 5), .5 if stage == 'image' else 3, dtype if stage == 'image' else np.uint8)
    mask = np.ones((3, 5), dtype if stage == 'mask' else np.uint8)
    footprint = np.ones((3, 3), np.complex64 if stage == 'footprint' else np.uint8)
    target = image if stage == 'image' else mask if stage == 'mask' else footprint
    if stage == 'footprint':
        target.flat[0] = complex(np.nan, 1)
    else:
        target.flat[0] = np.nan
        target.flat[1] = np.finfo(dtype).max
        if signal:
            unsigned = {np.float16: np.uint16, np.float32: np.uint32, np.float64: np.uint64}[dtype]
            word = {np.float16: 0x7c01, np.float32: 0x7f800001, np.float64: 0x7ff0000000000001}[dtype]
            target.view(unsigned).flat[0] = word
    out = np.full((3, 5), 77, np.uint8)
    return image, footprint, mask, out, target


def observe(provider, name, dtype, stage, policy, action, signal, capfd):
    image, fp, mask, out, target = make_arrays(dtype, stage, signal)
    events = []
    invoked = False
    original_policy = np.geterr()
    original_handler = np.geterrcall()

    def replacement(*arguments):
        events.append(('replacement', arguments, tuple(np.geterr().items())))

    def handle(*arguments):
        nonlocal invoked
        events.append(('initial', arguments, tuple(np.geterr().items())))
        if invoked:
            return
        invoked = True
        if action == 'fill':
            image.fill(0); fp.fill(0); mask.fill(0); out.fill(99)
        elif action == 'shape':
            target.shape = (1,) + target.shape
        elif action == 'dtype':
            target.dtype = np.uint8
        elif action == 'ignore':
            np.seterr(all='ignore')
        elif action == 'handler':
            np.seterrcall(replacement if policy == 'call' else Logger(replacement))
        elif action == 'reentrant':
            with np.errstate(all='ignore'):
                value = provider.sum(np.ones((3, 5), np.uint8), np.ones((1, 1), np.uint8))
                events.append(('nested', record(value)))
            np.seterr(over='ignore')

    class Logger:
        def __init__(self, callback):
            self.callback = callback
        def write(self, message):
            self.callback(message)

    np.seterrcall(handle if policy != 'log' else Logger(handle))
    np.seterr(all=policy)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                result = ('return', record(getattr(provider, name)(image, fp, mask=mask, out=out)))
            except Exception as error:
                result = ('error', type(error).__name__, str(error))
        state = (tuple(np.geterr().items()), tuple(events), tuple((type(w.message).__name__, str(w.message)) for w in notices))
        streams = capfd.readouterr()
        return result, state, tuple(record(a) for a in (image, fp, mask, out)), (streams.out, streams.err)
    finally:
        np.seterr(**original_policy)
        np.seterrcall(original_handler)


@pytest.fixture(params=[1, 4])
def workers(request):
    previous = native._core.get_num_threads()
    native._core.set_num_threads(request.param)
    yield request.param
    native._core.set_num_threads(previous)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('stage', ['image', 'mask', 'footprint'])
@pytest.mark.parametrize('policy', ['warn', 'raise', 'ignore', 'call', 'log', 'print'])
@pytest.mark.parametrize('signal', [False, True])
def test_dtype_policy_diagnostics_and_materialized_outputs(name, dtype, stage, policy, signal, workers, capfd):
    expected = observe(reference, name, dtype, stage, policy, 'none', signal, capfd)
    actual = observe(native, name, dtype, stage, policy, 'none', signal, capfd)
    assert actual == expected


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('stage', ['image', 'mask', 'footprint'])
@pytest.mark.parametrize('policy', ['call', 'log'])
@pytest.mark.parametrize('action', ['fill', 'shape', 'dtype', 'ignore', 'handler', 'reentrant'])
def test_callback_mutations_and_reentrant_policy(name, dtype, stage, policy, action, workers, capfd):
    expected = observe(reference, name, dtype, stage, policy, action, True, capfd)
    actual = observe(native, name, dtype, stage, policy, action, True, capfd)
    assert actual == expected

@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.longdouble, np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize('policy', ['call', 'log', 'print'])
@pytest.mark.parametrize('signal', [False, True])
def test_3d_footprint_provider_dtype_and_signal(name, dtype, policy, signal, workers, capfd):
    def execute(provider):
        image = np.full((3, 4, 5), 3, np.uint8)
        fp = np.ones((3, 3, 3), dtype)
        fp.flat[0] = np.nan
        initialize_extended_padding(fp)
        # Portable IEEE SNaN representations for half/single/double components.
        component = np.empty((), dtype).real.dtype
        words = {2: (np.uint16, 0x7c01), 4: (np.uint32, 0x7f800001), 8: (np.uint64, 0x7ff0000000000001)}
        if signal and component.itemsize in words:
            unsigned, word = words[component.itemsize]
            fp.view(unsigned).flat[0] = word
        calls = []
        class Logger:
            def write(self, message):
                calls.append(message)
        def callback(error, flags):
            calls.append((error, flags))
            # Comparison output is already materialized. Subsequent native
            # image reads must see this mutation, while the footprint must not.
            image.fill(7)
            fp.fill(0)
            initialize_extended_padding(fp)
        original = np.geterrcall()
        np.seterrcall(Logger() if policy == 'log' else callback)
        try:
            with np.errstate(all=policy), warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                result = getattr(provider, name)(image, fp)
            streams = capfd.readouterr()
            return record(result), record(image), record(fp), calls, [(type(w.message).__name__, str(w.message)) for w in notices], (streams.out, streams.err)
        finally:
            np.seterrcall(original)
    assert execute(native) == execute(reference)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('hook', ['precision-call', 'multiply-dtype', 'multiply-handler-error'])
def test_warning_hooks_capture_dtype_shape_and_dynamic_policy(name, dtype, hook, workers, capfd):
    def execute(provider):
        image = np.full((3, 5), .5, dtype)
        unsigned = {np.float16: np.uint16, np.float32: np.uint32, np.float64: np.uint64}[dtype]
        word = {np.float16: 0x7c01, np.float32: 0x7f800001, np.float64: 0x7ff0000000000001}[dtype]
        image.view(unsigned).flat[0] = word
        image.flat[1] = np.finfo(dtype).max
        events = []
        changed = False
        previous_handler = np.geterrcall()
        previous_policy = np.geterr()
        def callback(error, flags):
            events.append(('callback', error, flags))
            if hook == 'multiply-handler-error':
                raise LookupError('dtype callback sentinel')
        def warning_hook(message, category, filename, lineno, file=None, line=None):
            nonlocal changed
            events.append(('warning', category.__name__, str(message)))
            if changed:
                return
            if hook == 'precision-call' and category is UserWarning:
                changed = True
                np.seterrcall(callback)
                np.seterr(all='call')
            elif 'multiply' in str(message):
                changed = True
                if hook == 'multiply-dtype':
                    image.dtype = np.uint8
                else:
                    np.seterrcall(callback)
                    np.seterr(all='call')
        try:
            np.seterr(all='warn')
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.showwarning = warning_hook
                try:
                    value = ('return', record(getattr(provider, name)(image, np.ones((3, 3), np.uint8))))
                except Exception as error:
                    value = ('error', type(error).__name__, str(error))
            streams = capfd.readouterr()
            return value, events, record(image), tuple(np.geterr().items()), (streams.out, streams.err)
        finally:
            np.seterr(**previous_policy)
            np.seterrcall(previous_handler)
    assert execute(native) == execute(reference)

@pytest.mark.parametrize('name', NAMES)
def test_stateful_dtype_provider_never_forwards_rank(name, workers, monkeypatch):
    from skimage import util
    image = np.full((3, 5), .5, np.float16)
    image.flat[0] = np.nan
    fp = np.ones((3, 3), np.complex64)
    fp.flat[0] = complex(np.nan, 1)
    def handler(*arguments):
        pass
    original = np.geterrcall()
    np.seterrcall(handler)
    try:
        with np.errstate(all='call'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = getattr(reference, name)(image, fp)
            def forbidden(*args, **kwargs):
                raise AssertionError('upstream image algorithm forwarding')
            for operation in NAMES:
                monkeypatch.setattr(reference, operation, forbidden)
            monkeypatch.setattr(util, 'img_as_ubyte', forbidden)
            actual = getattr(native, name)(image, fp)
        np.testing.assert_array_equal(actual, expected)
    finally:
        np.seterrcall(original)
