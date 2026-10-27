"""Original-first Perimeter histogram/weighted-sum contracts; native remains RED."""
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
    from .perimeter_cases import API, STAGES, UNASSESSED, cases, fixture, error_limit
    from .test_moments_public import original_runtime, metadata
    from .test_moments_derived_public import observe, result_record
else:
    from perimeter_cases import API, STAGES, UNASSESSED, cases, fixture, error_limit
    from test_moments_public import original_runtime, metadata
    from test_moments_derived_public import observe, result_record

CASES = cases()
SOURCE = 'skimage/measure/_regionprops_utils.py'
SOURCE_SHA256 = 'b659f6b7d84b8ca5df77242ad33fa1d893d7e103e182c957cd7cb3016cd14f21'
SCIPY_PINS = {
    '1.15.3': {'scipy/ndimage/_morphology.py': '2c5f7580a25c1c85a80fd6ad87ff7e63b1e053041b03410baa0558be6d580160',
               'scipy/ndimage/_filters.py': 'e821fbd5ae150dc9fdb616ae5a213504904e5416febcde421739f3fe52769c0c'},
    '1.18.1': {'scipy/ndimage/_morphology.py': '5361c28977347a3bc4ba1240aea0b341a72a803498a5600506c9964a61eeb9ad',
               'scipy/ndimage/_filters.py': 'b90b80d80d038859966aad1b775431fec16f4d45b79150eb8ca385d769584223'},
}


def runtime():
    np, _, record = original_runtime()
    path = Path(importlib.metadata.distribution('scikit-image').locate_file(SOURCE))
    raw = path.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == SOURCE_SHA256
    record['perimeter_source'] = dict(path=str(path), raw_sha256=hashlib.sha256(raw).hexdigest(),
                                   canonical_lf_sha256=SOURCE_SHA256)
    backend = {}
    for name, expected in SCIPY_PINS[record['scipy']].items():
        raw = Path(importlib.metadata.distribution('scipy').locate_file(name)).read_bytes()
        assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == expected
        backend[name] = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), canonical_lf_sha256=expected)
    record['perimeter_backend_sources'] = backend
    return np, importlib.import_module('skimage.measure._regionprops_utils'), record


def encode(value, np):
    if isinstance(value, np.ndarray) and value.dtype.kind == 'O':
        assert all(type(item) is int for item in value.flat)
        return dict(metadata=metadata(value), values=value.tolist())
    if isinstance(value, (np.ndarray, np.generic)): return result_record(value)
    return value


def reference_case(spec, np, original):
    image, options, events = fixture(spec, np)
    captured = {}; prior = sys.getprofile()
    def capture(frame, event, arg):
        if event == 'return' and frame.f_code is original.perimeter.__code__:
            captured.update(h=frame.f_locals.get('perimeter_histogram'), coefs=frame.f_locals.get('perimeter_weights'), border=frame.f_locals.get('border_image'), codes=frame.f_locals.get('perimeter_image'))
    sys.setprofile(capture)
    try: observed = observe(original.perimeter, image, options, spec['policy'], np)
    finally: sys.setprofile(prior)
    return image, options, events, captured, observed


def validate(spec, captured, observed, np):
    result, error, messages, checks = observed
    assert all(checks.values()), spec['id']
    if spec['outcome'] != 'source':
        assert (error is None) == (spec['outcome'] == 'return'), (spec['id'], error)
    h = captured.get('h')
    if h is not None:
        assert type(h) is np.ndarray and h.ndim == 1 and 50 <= h.size <= 256 and h.dtype == np.dtype(np.intp)
        assert np.all(h >= 0)
    if error is not None: return None
    assert type(result) is np.float64 and np.isfinite(result) and result >= 0
    assert h.shape == (50,)
    return error_limit(h, captured['coefs'])


def compare(actual, expected, bound, np):
    assert type(actual) is type(expected) and metadata(actual) == metadata(expected)
    for mask in (np.isnan, np.isposinf, np.isneginf): assert bool(mask(actual)) == bool(mask(expected))
    if np.isfinite(expected):
        assert abs(float(actual)-float(expected)) <= bound
        if bound == 0: assert actual == expected and np.signbit(actual) == np.signbit(expected)


@pytest.fixture(scope='module')
def reference_runtime(): return runtime()


@pytest.fixture(scope='module')
def worker_results(): return {}


def test_perimeter_signature(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(), 'perimeter remains upstream-only'
    assert inspect.signature(xx.measure.perimeter) == inspect.signature(original.perimeter)


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_perimeter_owned(reference_runtime, spec, workers, worker_results, monkeypatch, tmp_path):
    np, original, _ = reference_runtime
    image, options, events, captured, observed = reference_case(spec, np, original)
    expected, error, messages, checks = observed
    row = dict(id=spec['id'], input=encode(image, np), histogram=encode(captured.get('h'), np),
               coefficients=encode(captured.get('coefs'), np), border=encode(captured.get('border'), np), codes=encode(captured.get('codes'), np), result=encode(expected, np), error=error,
               warnings=messages, checks=checks, events=events)
    (tmp_path/'original.json').write_text(json.dumps(row)+'\n')
    bound = validate(spec, captured, observed, np)
    expected_input, expected_events = encode(image, np), list(events)
    image, options, events = fixture(spec, np)
    import skimage.measure as upstream
    import scipy.ndimage as ndi
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions() and xx.get_num_threads() == 1
    call = xx.measure.perimeter
    old = xx.set_fallback_enabled(False); prior = sys.getprofile()
    returns = []; actual_local = {}; forbidden = []; neighborhood_events_at_counts = []
    def deny(*args, **kwargs):
        forbidden.append(True); raise AssertionError('upstream Perimeter/stencil/weighted reduction forwarding')
    def profile(frame, event, arg):
        if event == 'c_return' and getattr(arg, '__module__', None) == xx._core.__name__ and getattr(arg, '__name__', None) in STAGES:
            returns.append(arg.__name__)
            if arg.__name__ == STAGES[0]: neighborhood_events_at_counts.append(list(events))
        if event == 'return' and frame.f_code is call.__code__:
            actual_local['h'] = frame.f_locals.get('perimeter_histogram')
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(upstream, 'perimeter', deny)
            patch.setattr(original, 'perimeter', deny)
            patch.setattr(_compat, 'fallback', deny)
            for name in ('dot', 'matmul'): patch.setattr(np, name, deny)
            if captured.get('h') is not None:
                patch.setattr(ndi, 'binary_erosion', deny)
                patch.setattr(importlib.import_module('scipy.ndimage._morphology'), 'binary_erosion', deny)
                patch.setattr(ndi, 'convolve', deny)
                patch.setattr(importlib.import_module('scipy.ndimage._filters'), 'convolve', deny)
                patch.setattr(np, 'bincount', deny)
            sys.setprofile(profile)
            try: actual, actual_error, actual_messages, actual_checks = observe(call, image, options, spec['policy'], np)
            finally: sys.setprofile(prior)
        actual_h = actual_local.get('h')
        raw = dict(id=spec['id'], workers=workers, histogram=encode(actual_h, np), result=encode(actual, np),
                   error=actual_error, warnings=actual_messages, checks=actual_checks, events=events,
                   actual_C_returns=returns, count_credit=returns.count(STAGES[0]) if captured.get('h') is not None and image.size else 0,
                   weighted_credit=returns.count(STAGES[1]), neighborhood_events_at_counts=neighborhood_events_at_counts,
                   forbidden=forbidden)
        (tmp_path/'candidate.json').write_text(json.dumps(raw)+'\n')
        assert not forbidden and actual_error == error and actual_messages == messages
        assert actual_checks == checks and events == expected_events
        assert encode(image, np) == expected_input
        if captured.get('h') is None:
            assert actual_h is None
        else:
            assert metadata(actual_h) == metadata(captured['h'])
            np.testing.assert_array_equal(actual_h, captured['h'])
        if captured.get('h') is not None:
            assert returns.count(STAGES[0]) == 1
            assert neighborhood_events_at_counts == [expected_events]
        if captured.get('h') is None: assert not returns
        assert returns.count(STAGES[1]) == int(error is None)
        if STAGES[0] in returns and STAGES[1] in returns:
            assert returns.index(STAGES[0]) < returns.index(STAGES[1])
        if error is None: compare(actual, expected, bound, np)
        if spec['id'] in worker_results:
            previous, previous_h, previous_error, previous_messages, previous_events = worker_results[spec['id']]
            assert (actual_error, actual_messages, events) == (previous_error, previous_messages, previous_events)
            if actual_h is not None: np.testing.assert_array_equal(actual_h, previous_h)
            if error is None:
                compare(actual, previous, 0., np)
                assert actual.tobytes() == previous.tobytes()
        else: worker_results[spec['id']] = (actual, actual_h, actual_error, actual_messages, list(events))
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior); xx.set_num_threads(1); xx.set_fallback_enabled(old)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    assert args.reference_only and sys.flags.optimize == 0
    assert not any(name == 'skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    args.output.mkdir(); np, original, record = runtime()
    (args.output/'runtime.json').write_text(json.dumps(record, indent=2)+'\n')
    rows = []
    with (args.output/'journal.jsonl').open('x') as journal:
        for spec in CASES:
            image, options, events, captured, observed = reference_case(spec, np, original)
            expected, error, messages, checks = observed
            row = dict(id=spec['id'], spec=spec, input=encode(image, np), histogram=encode(captured.get('h'), np),
                       coefficients=encode(captured.get('coefs'), np), border=encode(captured.get('border'), np), codes=encode(captured.get('codes'), np), result=encode(expected, np), error=error,
                       warnings=messages, checks=checks, events=events)
            journal.write(json.dumps(row)+'\n'); journal.flush()
            bound = validate(spec, captured, observed, np)
            if error is None: compare(expected, expected, bound, np)
            rows.append(dict(id=spec['id'], error=error, warnings=messages, bound=bound))
    assert not any(name == 'skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),
        returns=sum(row['error'] is None for row in rows), errors=sum(row['error'] is not None for row in rows),
        native_calls=0, rows=rows, unassessed=UNASSESSED), indent=2)+'\n')
    print('original-only', len(rows), 'PASS', flush=True)


if __name__ == '__main__': main()
