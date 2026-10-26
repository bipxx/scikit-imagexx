"""Original-first block reduction tests; the registered native API is still RED."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import pytest
if __package__:
    from .block_reduce_cases import API, STAGE, UNASSESSED, cases, fixture, error_limits
    from .test_moments_public import original_runtime, metadata
    from .test_moments_derived_public import observe, result_record
else:
    from block_reduce_cases import API, STAGE, UNASSESSED, cases, fixture, error_limits
    from test_moments_public import original_runtime, metadata
    from test_moments_derived_public import observe, result_record

CASES = cases()
PINS = {
    'skimage/measure/block.py': '6f87615de79a8817ab3e6169363b66e762afc119880723c8d54f98d12e60c2ce',
    'skimage/util/shape.py': 'fc9ab1bdc09fa464828b6ee3ce96e610d04606b90c13a3ed58f80138e65c876b',
}
NUMPY_PINS = {
    '1.24.4': ('numpy/core', '1f7d40d18bd6b1cfde33db24abcfc0ecbdd480ca828d772cd19d2997145080dd', '9cdb028ec03cf85ce05f5ef380a1bbc697d25dc426a67ca319ef15cb92e5618a'),
    '2.5.4': ('numpy/_core', '09b04d56271fa78c43f6aca66ec06c802b0b22b026ae2fa6153575459b6dab53', '724facf7e63c5b8fcc2fac14bf31a02cb048d9e28f516d86502ba5077a425476'),
}


def runtime():
    np, _, record = original_runtime()
    prefix, numeric, methods = NUMPY_PINS[np.__version__]
    pins = dict(PINS, **{prefix+'/fromnumeric.py': numeric, prefix+'/_methods.py': methods})
    sources = {}
    for name, expected in pins.items():
        dist = importlib.metadata.distribution('scikit-image' if name.startswith('skimage/') else 'numpy')
        raw = Path(dist.locate_file(name)).read_bytes()
        assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == expected
        sources[name] = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), canonical_lf_sha256=expected)
    record['block_sources'] = sources
    return np, importlib.import_module('skimage.measure.block'), record


def encode(value, np):
    if isinstance(value, (np.ndarray, np.generic)): return result_record(value)
    return value


def reference_case(spec, np, original):
    image, options, events = fixture(spec, np); captured = {}; prior = sys.getprofile()
    before = encode(image, np)
    def capture(frame, event, arg):
        if event == 'return' and frame.f_code is original.block_reduce.__code__:
            captured['blocked'] = frame.f_locals.get('blocked')
    sys.setprofile(capture)
    try: observed = observe(original.block_reduce, image, options, spec['policy'], np)
    finally: sys.setprofile(prior)
    return image, options, events, captured, observed, before


def validate(spec, captured, observed, np):
    result, error, _, checks = observed
    assert checks['state_unchanged']
    if spec['outcome'] != 'source': assert (error is None) == (spec['outcome'] == 'return'), (spec['id'], error)
    if error is not None: return None
    assert type(result) is np.ndarray or isinstance(result, np.generic)
    with np.errstate(all='ignore'): return error_limits(spec, captured.get('blocked'), result, np)


def compare(actual, expected, bound, np):
    assert metadata(actual) == metadata(expected)
    if expected.dtype.kind not in 'fc':
        np.testing.assert_array_equal(actual, expected); return
    components = ((actual.real, expected.real), (actual.imag, expected.imag)) if expected.dtype.kind == 'c' else ((actual, expected),)
    for a, e in components:
        for mask in (np.isnan, np.isposinf, np.isneginf): np.testing.assert_array_equal(mask(a), mask(e))
        finite = np.isfinite(e)
        assert np.all(np.abs(np.asarray(a[finite], dtype=np.float64)-np.asarray(e[finite], dtype=np.float64)) <= bound[finite])
        zeros = finite & (e == 0)
        np.testing.assert_array_equal(np.signbit(a[zeros]), np.signbit(e[zeros]))
        exact = finite & (bound == 0)
        np.testing.assert_array_equal(a[exact], e[exact])


@pytest.fixture(scope='module')
def reference_runtime(): return runtime()


@pytest.fixture(scope='module')
def worker_results(): return {}


def test_block_reduce_signature(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(), 'block_reduce remains upstream-only'
    assert inspect.signature(xx.measure.block_reduce) == inspect.signature(original.block_reduce)


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_block_reduce_owned(reference_runtime, spec, workers, worker_results, monkeypatch, tmp_path):
    np, original, _ = reference_runtime
    ref_image, ref_options, ref_events, captured, observed, before = reference_case(spec, np, original)
    expected, expected_error, expected_warnings, expected_checks = observed
    raw = dict(id=spec['id'], input=before, input_after=encode(ref_image, np),
               blocked=encode(captured.get('blocked'), np), result=encode(expected, np),
               error=expected_error, warnings=expected_warnings, checks=expected_checks, events=ref_events)
    (tmp_path/'original.json').write_text(json.dumps(raw)+'\n')
    bound = validate(spec, captured, observed, np)
    image, options, events = fixture(spec, np)
    import skimage.measure as upstream
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions() and xx.get_num_threads() == 1
    previous = xx.set_fallback_enabled(False); prior = sys.getprofile(); returns = []; forbidden = []
    def deny(*args, **kwargs): forbidden.append(True); raise AssertionError('upstream block reduction forwarding')
    def profile(frame, event, arg):
        if event == 'c_return' and getattr(arg, '__module__', None) == xx._core.__name__ and getattr(arg, '__name__', None) == STAGE:
            returns.append(STAGE)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(upstream, 'block_reduce', deny); patch.setattr(original, 'block_reduce', deny)
            if spec['route'] == 'native':
                numeric = importlib.import_module(NUMPY_PINS[np.__version__][0].replace('/', '.')+'.fromnumeric')
                patch.setattr(numeric, '_wrapreduction', deny)
                patch.setattr(np, 'sum', deny)
            sys.setprofile(profile)
            try: actual, error, warnings, checks = observe(xx.measure.block_reduce, image, options, spec['policy'], np)
            finally: sys.setprofile(prior)
        credit = len(returns) if spec['route'] == 'native' and error is None and actual.size else 0
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'], workers=workers, result=encode(actual, np),
            input_after=encode(image, np), error=error, warnings=warnings, checks=checks, events=events,
            actual_C_returns=returns, owned_sum_credit=credit, forbidden=forbidden))+'\n')
        assert not forbidden and error == expected_error and warnings == expected_warnings
        assert checks == expected_checks and events == ref_events
        if isinstance(image, np.ndarray):
            assert metadata(image) == metadata(ref_image) and image.tobytes(order='C') == ref_image.tobytes(order='C')
        else: assert image == ref_image
        if error is None:
            compare(actual, expected, bound, np)
            if spec['route'] == 'native' and actual.size: assert credit > 0
            if spec.get('kwargs') == 'out': assert actual is options['func_kwargs']['out'] and expected is ref_options['func_kwargs']['out']
        if spec['id'] in worker_results:
            prior_actual, prior_error, prior_warnings, prior_events = worker_results[spec['id']]
            assert (error, warnings, events) == (prior_error, prior_warnings, prior_events)
            if error is None:
                compare(actual, prior_actual, np.zeros(actual.shape, dtype=np.float64), np)
                if spec['route'] == 'native': assert actual.tobytes(order='C') == prior_actual.tobytes(order='C')
        else: worker_results[spec['id']] = (actual, error, warnings, events)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior); xx.set_num_threads(1); xx.set_fallback_enabled(previous)


def reference_only(output):
    assert sys.flags.optimize == 0 and not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    output.mkdir(); np, original, record = runtime()
    (output/'runtime.json').write_text(json.dumps(record, indent=2)+'\n')
    rows = []
    with (output/'journal.jsonl').open('x') as journal:
        for spec in CASES:
            image, options, events, captured, observed, before = reference_case(spec, np, original)
            result, error, warnings, checks = observed
            row = dict(id=spec['id'], spec=spec, input=before, input_after=encode(image, np),
                       blocked=encode(captured.get('blocked'), np), result=encode(result, np),
                       error=error, warnings=warnings, checks=checks, events=events)
            journal.write(json.dumps(row)+'\n'); journal.flush()
            bound = validate(spec, captured, observed, np)
            if error is None: compare(result, result, bound, np)
            rows.append(dict(id=spec['id'], error=error, warnings=warnings))
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    (output/'summary.json').write_text(json.dumps(dict(status='ORIGINAL_ONLY', cases=len(rows),
        returns=sum(r['error'] is None for r in rows), errors=sum(r['error'] is not None for r in rows),
        native_coverage=0, rows=rows, unassessed=UNASSESSED), indent=2)+'\n')
    print('original-only', len(rows), 'PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--reference-only', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); reference_only(args.output)
