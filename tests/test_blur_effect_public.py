"""Original-first blur-effect stages, metadata and reducer-protocol contracts."""
import argparse
import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import sys
import zlib

import pytest
if __package__:
    from .blur_effect_cases import API, STAGES, UNASSESSED, cases, fixture
    from .test_moments_public import original_runtime
    from .test_moments_derived_public import observe, result_record
else:
    from blur_effect_cases import API, STAGES, UNASSESSED, cases, fixture
    from test_moments_public import original_runtime
    from test_moments_derived_public import observe, result_record

CASES = cases()
SOURCE_SHA256 = '9a182f2d5ba039f428de223f664536ff3711555987121dbc3311217d58dcf280'
PYTHON_SOURCE_PINS = {
    'skimage.filters.edges': '315283a873217dadbe9886fce38039e99d53dc109823c183cef12f6f24b84091',
    'skimage.color.colorconv': 'fc5f49ac194c229325f21ddbe4c49f7df4307b9627dfbb5fcd2825133144fdfe',
    'skimage.util.dtype': 'd4f99f88966e4d0dbbe80a3c0b8acb3890644cc5fbfb707ba9b5a90e9119e627',
    'skimage._shared.utils': 'b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1',
}
SCIPY_SOURCE_PINS = {
    '1.15.3': {'scipy.ndimage._filters': 'e821fbd5ae150dc9fdb616ae5a213504904e5416febcde421739f3fe52769c0c',
               'scipy.ndimage._ni_support': 'c1e60b92002969fd161b9e1d92cc499c5118d93a024fd3b75cd3f8778a69a453'},
    '1.18.1': {'scipy.ndimage._filters': 'b90b80d80d038859966aad1b775431fec16f4d45b79150eb8ca385d769584223',
               'scipy.ndimage._ni_support': 'd49a55e974b2bcc3fafaba8a1c7101adcf3745a5640f3ca704794eebbdac7336'},
}
INTERMEDIATES = ('filt_im', 'im_sharp', 'im_blur', 'T', 'M1', 'M2')


def runtime():
    np, _, record = original_runtime()
    original = importlib.import_module('skimage.measure._blur_effect')
    path = Path(original.__file__)
    raw = path.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == SOURCE_SHA256
    additional = {}
    for name, pin in {**PYTHON_SOURCE_PINS, **SCIPY_SOURCE_PINS[record['scipy']]}.items():
        source_path = Path(importlib.import_module(name).__file__)
        source_bytes = source_path.read_bytes()
        assert hashlib.sha256(source_bytes.replace(b'\r\n', b'\n')).hexdigest() == pin, name
        additional[name] = dict(path=str(source_path), raw_sha256=hashlib.sha256(source_bytes).hexdigest(),
                                canonical_lf_sha256=pin)
    record.update(blur_source=dict(path=str(path), raw_sha256=hashlib.sha256(raw).hexdigest(),
                                   canonical_lf_sha256=SOURCE_SHA256),
                  additional_sources=additional,
                  policy='Exact stage/public metadata and words; only declared provider NaNs use masks')
    return np, original, record


def encode(value, np):
    if isinstance(value, (np.ndarray, np.generic)):
        return result_record(value)
    if isinstance(value, dict):
        return {key: encode(item, np) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return dict(container=type(value).__name__, items=[encode(item, np) for item in value])
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return 'callable:' + getattr(value, '__name__', type(value).__name__)


def captured_call(call, image, options, policy, np):
    lines, first = inspect.getsourcelines(call)
    append_lines = [first+i for i, line in enumerate(lines) if 'B.append(' in line]
    assert len(append_lines) == 1, 'literal per-axis append is the passive observation point'
    captured = []
    prior = sys.gettrace()
    def trace(frame, event, arg):
        if frame.f_code is call.__code__:
            if event == 'line' and frame.f_lineno == append_lines[0]:
                local = frame.f_locals
                captured.append(dict(axis=local['ax'], **{
                    name: encode(local[name], np) for name in INTERMEDIATES}))
            return trace
        return None
    stdout = io.StringIO()
    sys.settrace(trace)
    try:
        with contextlib.redirect_stdout(stdout):
            observed = observe(call, image, options, policy, np)
    finally:
        sys.settrace(prior)
    return captured, stdout.getvalue(), observed


def compare_records(actual, expected, nan_mask, np):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        if 'raw_c_zlib_base64' in expected:
            assert actual['metadata'] == expected['metadata'], 'stage/result metadata differs'
            a = zlib.decompress(base64.b64decode(actual['raw_c_zlib_base64']))
            e = zlib.decompress(base64.b64decode(expected['raw_c_zlib_base64']))
            dtype = np.dtype(expected['metadata'][2])
            if nan_mask and dtype.kind == 'f':
                av, ev = np.frombuffer(a, dtype=dtype), np.frombuffer(e, dtype=dtype)
                am, em = np.isnan(av), np.isnan(ev)
                assert np.array_equal(am, em), 'provider NaN masks differ'
                # Only NaN words may differ. Finite/Inf/signed-zero bytes stay exact.
                aw = np.frombuffer(a, dtype=np.uint8).reshape(-1, dtype.itemsize)
                ew = np.frombuffer(e, dtype=np.uint8).reshape(-1, dtype.itemsize)
                assert aw[~em].tobytes() == ew[~em].tobytes(), 'non-NaN words differ'
            else:
                assert a == e, 'stage/result words differ'
            return
        for key in expected:
            compare_records(actual[key], expected[key], nan_mask, np)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_records(a, e, nan_mask, np)
    else:
        assert actual == expected


def validate(spec, captured, observed, np):
    result, error, _, checks = observed
    assert checks == dict(state_unchanged=True, input_unchanged=True)
    assert (error is None) == (spec['outcome'] == 'return'), (spec['id'], error)
    assert [row['axis'] for row in captured] == list(range(len(captured)))
    if spec['route'] == 'native':
        assert captured, spec['id']
    if error is None:
        assert isinstance(result, (list, np.floating)), (spec['id'], type(result))
    for row in captured:
        shape = row['filt_im']['metadata'][4]
        assert all(size > 0 for size in shape)
        assert all(row[name]['metadata'][4] == shape for name in ('im_sharp', 'im_blur', 'T'))
        assert row['M1']['metadata'][4] == row['M2']['metadata'][4] == ()


@pytest.fixture(scope='module')
def reference_runtime():
    return runtime()


@pytest.fixture(scope='module')
def worker_results():
    return {}


def test_blur_effect_signature(reference_runtime):
    _, original, _ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(), 'blur_effect remains upstream-only'
    assert inspect.signature(xx.measure.blur_effect) == inspect.signature(original.blur_effect)


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_blur_effect_owned(reference_runtime, spec, workers, worker_results, monkeypatch, tmp_path):
    np, original, _ = reference_runtime
    image, options, events = fixture(spec, np)
    before = encode((image, options), np)
    captured, stdout, observed = captured_call(original.blur_effect, image, options, spec['policy'], np)
    expected, error, messages, checks = observed
    validate(spec, captured, observed, np)
    expected_events, expected_input = list(events), encode((image, options), np)
    row = dict(id=spec['id'], input_before=before, input_after=expected_input,
               stages=captured, result=encode(expected, np), error=error,
               warnings=messages, stdout=stdout, checks=checks, events=events)
    (tmp_path/'original.json').write_text(json.dumps(row)+'\n')

    image, options, events = fixture(spec, np)
    import skimage.measure as upstream
    import skimage.filters as original_filters
    import scipy.ndimage as ndi
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions() and xx.get_num_threads() == 1
    old = xx.set_fallback_enabled(False)
    prior = sys.getprofile()
    returns, forbidden = [], []
    def deny(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('upstream blur/Sobel/filter forwarding')
    def profile(frame, event, arg):
        if (event == 'c_return' and getattr(arg, '__module__', None) == xx._core.__name__
                and getattr(arg, '__name__', None) in STAGES):
            returns.append(arg.__name__)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(upstream, 'blur_effect', deny)
            patch.setattr(original, 'blur_effect', deny)
            patch.setattr(_compat, 'fallback', deny)
            patch.setattr(original_filters, 'sobel', deny)
            edges = importlib.import_module('skimage.filters.edges')
            patch.setattr(edges, 'sobel', deny)
            patch.setattr(edges, '_generic_edge_filter', deny)
            if spec['route'] == 'native':
                for namespace in (ndi, importlib.import_module('scipy.ndimage._filters')):
                    for name in ('uniform_filter1d', 'convolve', 'correlate'):
                        patch.setattr(namespace, name, deny)
                backend = importlib.import_module('scipy.ndimage._nd_image')
                for name in ('uniform_filter1d', 'correlate'):
                    patch.setattr(backend, name, deny)
            sys.setprofile(profile)
            try:
                actual_capture, actual_stdout, actual_observed = captured_call(
                    xx.measure.blur_effect, image, options, spec['policy'], np)
            finally:
                sys.setprofile(prior)
        actual, actual_error, actual_messages, actual_checks = actual_observed
        actual_row = dict(id=spec['id'], workers=workers, input_after=encode((image, options), np),
                          stages=actual_capture, result=encode(actual, np), error=actual_error,
                          warnings=actual_messages, stdout=actual_stdout, checks=actual_checks,
                          events=events, actual_C_returns=returns, forbidden=forbidden)
        (tmp_path/'candidate.json').write_text(json.dumps(actual_row)+'\n')
        assert not forbidden and actual_error == error and actual_messages == messages
        assert actual_stdout == stdout and actual_checks == checks
        assert events == expected_events and encode((image, options), np) == expected_input
        compare_records(actual_capture, captured, spec['nan_mask'], np)
        compare_records(encode(actual, np), encode(expected, np), spec['nan_mask'], np)
        if spec['route'] == 'native':
            assert returns == [STAGES[0], STAGES[1], STAGES[1]] * len(captured)
        elif spec['route'] == 'intake':
            assert not returns
        state = dict(result=encode(actual, np), stages=actual_capture, error=actual_error,
                     warnings=actual_messages, stdout=actual_stdout, events=list(events))
        if spec['id'] in worker_results:
            compare_records(state, worker_results[spec['id']], spec['nan_mask'], np)
        else:
            worker_results[spec['id']] = state
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.reference_only and sys.flags.optimize == 0
    assert not any(name == 'skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    args.output.mkdir()
    np, original, record = runtime()
    (args.output/'runtime.json').write_text(json.dumps(record, indent=2)+'\n')
    rows = []
    with (args.output/'journal.jsonl').open('x') as journal:
        for spec in CASES:
            image, options, events = fixture(spec, np)
            before = encode((image, options), np)
            captured, stdout, observed = captured_call(original.blur_effect, image, options, spec['policy'], np)
            result, error, messages, checks = observed
            row = dict(id=spec['id'], spec=spec, input_before=before,
                       input_after=encode((image, options), np), stages=captured,
                       result=encode(result, np), error=error, warnings=messages,
                       stdout=stdout, checks=checks, events=events)
            journal.write(json.dumps(row)+'\n')
            journal.flush()
            validate(spec, captured, observed, np)
            rows.append(dict(id=spec['id'], error=error, warnings=messages, stages=len(captured)))
    assert not any(name == 'skimagexx' or name.startswith('skimagexx.') for name in sys.modules)
    summary = dict(cases=len(rows), returns=sum(row['error'] is None for row in rows),
                   errors=sum(row['error'] is not None for row in rows), native_calls=0, rows=rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('original-only', len(rows), 'PASS', flush=True)


if __name__ == '__main__':
    main()
