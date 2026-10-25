"""Prospective source-ordered H floating addendum; no native admission.

The inherited observer is deliberately unchanged. Its outer all-warn context
records the call, while the supplied callable applies each case's explicit
policy to the actual original operation. No expected mask selects a route.
"""
import hashlib
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'coverage/contracts/h-extrema-floating-plan.json'


def cases():
    return json.loads(PLAN.read_bytes())['cases']


def validate_spec(spec):
    matches = [item for item in cases() if item['id'] == spec.get('id')]
    assert len(matches) == 1 and matches[0] == spec, 'not an exact frozen H floating case'


def fixture(spec, events):
    validate_spec(spec)
    import numpy as np
    dtype = np.dtype(spec['dtype'])
    family = spec['family']
    maxima = spec['function'] == 'h_maxima'
    if family == 'resolution-warning':
        # Literal old floating-real-limit/subnormal construction, including
        # the source's positive typed height. No new limiting approximation.
        unit = np.nextafter(dtype.type(0), dtype.type(1))
        values = [0, 4, 0, 8, 0] if maxima else [8, 4, 8, 0, 8]
        image = np.asarray([float(unit) * x for x in values], dtype=dtype)
        height = unit
    elif family == 'precision-warning':
        image = np.asarray([2**53, 2**53+1, 2**53+2, 2**53+4, 2**53+8], dtype=dtype)
        height = 1.5
    elif family in ('reconstruction-shape-warning', 'nonrepresentable-height'):
        values = [0, 4, 0, 8, 0] if maxima else [8, 4, 8, 0, 8]
        image = np.asarray(values, dtype=dtype)
        height = dtype.type(4)
        if family == 'nonrepresentable-height':
            height = 4.0 + spec['relation'] * 2.0**-23
            if spec['height_kind'] == 'numpy-float64':
                height = np.float64(height)
    elif family == 'wide-unsigned-comparison':
        values = [0, 2**54, 0, 2**55, 0] if maxima else [2**55, 2**54, 2**55, 0, 2**55]
        image = np.asarray(values, dtype=dtype)
        height = np.uint64(2**53 + spec['increment'])
    elif family == 'height-protocol':
        image = ((np.arange(35) * 7) % 13).reshape(5, 7).astype(dtype)

        class Height(float):
            def __gt__(self, other):
                events.append(['height_gt', _value(other)])
                return float.__gt__(self, other)

            def __eq__(self, other):
                events.append(['height_eq', _value(other)])
                if spec['mode'] == 'raise-at-zero-equality' and type(other) is int and other == 0:
                    raise LookupError('h-floating-zero-equality-stop')
                return float.__eq__(self, other)

            def __float__(self):
                events.append(['height_float'])
                return float.__float__(self)

            def __rsub__(self, other):
                events.append(['height_rsub', _value(other)])
                return other - float(self)

            def __radd__(self, other):
                events.append(['height_radd', _value(other)])
                return other + float(self)

        height = Height(1.5)
    else:
        raise AssertionError('unknown frozen H floating family')
    # All shapes/footprints are from source-certified old finite fixtures.
    assert type(image) is np.ndarray and image.ndim in (1, 2)
    assert image.size <= 35 and image.flags.c_contiguous and np.isfinite(image).all()
    return image, height, {}


def _value(value):
    # The exact inherited serializer is installed only during observe;
    # constructing a fixture never requires loading the original library.
    assert _SERIALIZER is not None
    return _SERIALIZER(value)


_SERIALIZER = None


def observe(spec, legacy, function=None):
    """Capture a real call with exact inherited intake/Cython certificates.

    Warning mutation hooks never resize or change dtype/owner/layout. The
    original warning operation and original numerical entry remain actual.
    ``function`` is reserved for a separately audited owned pipeline; this
    reference runner always omits it and therefore calls ORIGINAL directly.
    """
    import numpy as np
    global _SERIALIZER
    saved_fixture, saved_serializer = legacy.fixture, _SERIALIZER
    before_policy, before_handler, before_buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    old_original = legacy.ORIGINAL[spec['function']]
    old_reconstruction = legacy.SOURCE.grayreconstruct.reconstruction
    old_loop = legacy.R.SOURCE.reconstruction_loop
    actual_policy = None
    handler_events = []
    mutation_count = 0
    source_events = None

    def prepared(s, events):
        nonlocal source_events
        source_events = events
        return fixture(s, events)

    def actual(image, height, **kwargs):
        nonlocal actual_policy, mutation_count
        mode, family = spec.get('mode', 'record'), spec['family']

        def mutate():
            nonlocal mutation_count
            if mutation_count == 0:
                image.flat[-1] = image.flat[0]
                mutation_count += 1
                source_events.append(['image_after_callback', legacy.value(image)])

        def fp_handler(error, flag):
            handler_events.append(['numpy_callback', error, flag, legacy.value(image)])
            source_events.append(['numpy_callback', error, flag])
            if mode == 'call-mutate':
                mutate()

        prior_showwarning = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            handler_events.append(['warning_handler', category.__name__, str(message), legacy.value(image)])
            source_events.append(['warning_handler', category.__name__, str(message)])
            relevant = ((family == 'precision-warning' and issubclass(category, UserWarning)) or
                        (family == 'reconstruction-shape-warning' and issubclass(category, DeprecationWarning)))
            if mode == 'mutate' and relevant:
                mutate()
            prior_showwarning(message, category, filename, lineno, file=file, line=line)

        policy = mode if family == 'resolution-warning' else 'warn'
        if policy in ('call-record', 'call-mutate'):
            policy = 'call'
        with np.errstate(all=policy, call=fp_handler if policy == 'call' else np.geterrcall()):
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                if mode == 'error':
                    warnings.simplefilter('error', UserWarning if family == 'precision-warning' else DeprecationWarning)
                if family in ('precision-warning', 'reconstruction-shape-warning'):
                    warnings.showwarning = showwarning
                actual_policy = dict(errors=np.geterr(), handler='owned-recorder' if policy == 'call' else 'incoming')
                return (function or old_original)(image, height, **kwargs)

    legacy.fixture, _SERIALIZER = prepared, legacy.value
    try:
        row = legacy.observe(spec, actual)
    finally:
        legacy.fixture, _SERIALIZER = saved_fixture, saved_serializer
        assert np.geterr() == before_policy
        assert np.geterrcall() is before_handler
        assert np.getbufsize() == before_buffer
        assert legacy.ORIGINAL[spec['function']] is old_original
        assert legacy.SOURCE.grayreconstruct.reconstruction is old_reconstruction
        assert legacy.R.SOURCE.reconstruction_loop is old_loop
    row['actual_call_policy'] = actual_policy
    row['handler_events'] = handler_events
    row['finite_in_allocation_mutations'] = mutation_count
    row['observer_state_restored'] = True
    return row


def test_prospective_case_partition_and_source_pins():
    from collections import Counter
    plan = json.loads(PLAN.read_bytes())
    specs = cases()
    assert len(specs) == len({s['id'] for s in specs}) == 70
    assert Counter(s['family'] for s in specs) == dict(
        **{'resolution-warning': 30, 'precision-warning': 12, 'reconstruction-shape-warning': 12,
           'height-protocol': 4, 'nonrepresentable-height': 8, 'wide-unsigned-comparison': 4})
    assert plan['native_acceptance'] == 'pending' and plan['new_native_paths'] == 0
    import importlib.util
    spec=importlib.util.spec_from_file_location('h_case_source_provenance',ROOT/'tests/h_extrema_source_provenance.py')
    provenance=importlib.util.module_from_spec(spec);spec.loader.exec_module(provenance)
    for name, expected in plan['preserved_files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            provenance.authenticate_source(ROOT,name,expected)
        else:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for name, expected in plan['reference_files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            provenance.authenticate_source(ROOT,name,expected)
        else:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_unfrozen_input_is_rejected_before_fixture_operations():
    import pytest
    original = cases()[0]
    for key, value in (('id', 'unsafe/unregistered'), ('function', 'other'),
                       ('dtype', 'c16'), ('mode', 'unknown'), ('input_basis', 'modified')):
        altered = dict(original, **{key: value})
        with pytest.raises(AssertionError, match='exact frozen'):
            validate_spec(altered)
    with pytest.raises(AssertionError, match='exact frozen'):
        validate_spec({})


def test_frozen_original_addendum():
    import importlib.util
    path = ROOT / 'coverage/contracts/h-extrema-floating.json'
    if not path.exists():
        import pytest
        pytest.skip('prospective source freeze: original captures not authorized yet')
    spec = importlib.util.spec_from_file_location('h_floating_original_runner', ROOT / 'tests/h_extrema_floating_reference.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    legacy = runner.load('h_floating_frozen_legacy', 'test_h_extrema_contracts.py')
    _, authentication = runner.authenticate(legacy.R.profile())
    expected = json.loads(path.read_bytes())['profiles'][legacy.R.profile()]['observations']
    assert authentication['profile'] == legacy.R.profile()
    for s in cases():
        assert observe(s, legacy) == expected[s['id']]
