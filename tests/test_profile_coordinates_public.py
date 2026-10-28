"""Exact additive coordinate contracts; reference-only mode never samples an image."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
import pytest
if __package__:
    from .profile_coordinates_cases import cases, fixture
else:
    from profile_coordinates_cases import cases, fixture

CASES = cases()
LINSPACE_SOURCE_SHA = {
    '1.24.4': '98a25cd1fa12874cc71d1e15cdee79fe0f3347899533a53289f7cfa8440ccb0d',
    '2.5.4': '15fdfd149704e43797cc4923990d64b74762deab81d1a77441cae96c8a9cb9a3',
}
HELPER_SOURCE_SHA = '930145182f2aa6ef60870c24e975f70be74f5cb57bc4ec128e542a6001440535'


def runtime():
    import numpy as np
    import scipy
    import skimage
    module = importlib.import_module('skimage.measure.profile')
    assert skimage.__version__ == '0.26.0'
    assert (np.__version__, scipy.__version__) in (('1.24.4', '1.15.3'), ('2.5.4', '1.18.1'))
    sources = dict(linspace=inspect.getsource(inspect.unwrap(np.linspace)),
                   helper=inspect.getsource(module._line_profile_coordinates))
    assert hashlib.sha256(sources['linspace'].encode()).hexdigest() == LINSPACE_SOURCE_SHA[np.__version__]
    assert hashlib.sha256(sources['helper'].encode()).hexdigest() == HELPER_SOURCE_SHA
    return np, module._line_profile_coordinates, dict(python=sys.version, numpy=np.__version__,
        scipy=scipy.__version__, skimage=skimage.__version__, linspace_source_sha256=LINSPACE_SOURCE_SHA[np.__version__],
        helper_source_sha256=HELPER_SOURCE_SHA, scope='New coordinate-only holdouts; no image/C sampler, no replacement of old46 references')


def encode(value, np):
    if isinstance(value, (np.ndarray, np.generic)):
        record = dict(type=type(value).__module__+'.'+type(value).__name__, dtype=value.dtype.str,
                      dtype_metadata=value.dtype.metadata, shape=list(value.shape), raw=value.tobytes().hex())
        if isinstance(value, np.ndarray):
            record.update(strides=list(value.strides), own=bool(value.flags.owndata),
                          c=bool(value.flags.c_contiguous), f=bool(value.flags.f_contiguous),
                          writeable=bool(value.flags.writeable), base_type=type(value.base).__name__)
        return record
    return value


def observe(call, spec, np):
    arguments, events, policy = fixture(spec, np)
    before = (np.geterr(), np.geterrcall(), np.getbufsize())
    captured = {}; calls = 0; branches = {}; scalar_types = {}; returns = []
    line_code = inspect.unwrap(np.linspace).__code__; helper_code = call.__code__
    def profile(frame, event, arg):
        nonlocal calls
        if event == 'call' and frame.f_code is line_code: calls += 1
        if event == 'return' and frame.f_code is line_code:
            local = frame.f_locals
            branch = 'zero-step' if bool(local.get('any_step_zero', False)) else 'ordinary' if local.get('div', 0) > 0 else 'num0/1'
            branches[branch] = branches.get(branch, 0)+1
            kinds = tuple(type(local.get(n)).__module__+'.'+type(local.get(n)).__name__ for n in ('start', 'stop', 'delta', 'step'))
            scalar_types[str(kinds)] = scalar_types.get(str(kinds), 0)+1
        if event == 'return' and frame.f_code is helper_code:
            captured.update({n: encode(frame.f_locals[n], np) for n in ('line_row', 'line_col', 'row_width', 'col_width') if n in frame.f_locals})
        if event == 'c_return' and (getattr(arg, '__module__', None) or '').startswith('skimagexx.'):
            returns.append(getattr(arg, '__name__', 'unknown'))
    prior = sys.getprofile(); result = error = None
    try:
        np.seterr(**policy)
        if 'call' in policy.values():
            np.seterrcall(lambda message, flag: events.append(dict(event='fp.call', message=message, flag=int(flag))))
        active = (np.geterr(), np.geterrcall(), np.getbufsize())
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always'); sys.setprofile(profile)
            try: result = call(*arguments)
            except Exception as exc:
                error = dict(type=type(exc).__module__+'.'+type(exc).__name__, message=str(exc))
            finally: sys.setprofile(prior)
        unchanged = (np.geterr(), np.geterrcall(), np.getbufsize()) == active
        assert unchanged
        return dict(id=spec['id'], spec=spec, result=encode(result, np), error=error,
            warnings=[dict(type=type(w.message).__module__+'.'+type(w.message).__name__, message=str(w.message)) for w in messages],
            events=events, prelude=captured, linspace_calls=calls, linspace_branches=branches,
            linspace_scalar_types=scalar_types, actual_C_returns=returns, state_unchanged=unchanged)
    finally:
        sys.setprofile(prior); np.seterr(**before[0]); np.seterrcall(before[1]); np.setbufsize(before[2])


@pytest.fixture(scope='module')
def reference_runtime(): return runtime()


@pytest.mark.parametrize('spec', CASES, ids=[r['id'] for r in CASES])
def test_profile_coordinate_words_and_protocol(reference_runtime, spec, tmp_path):
    np, original, _ = reference_runtime
    reference = observe(original, spec, np)
    (tmp_path/'original-coordinates.json').write_text(json.dumps(reference)+'\n')
    from skimagexx._profile_line import _line_profile_coordinates as candidate
    actual = observe(candidate, spec, np)
    (tmp_path/'candidate-coordinates.json').write_text(json.dumps(actual)+'\n')
    for name in ('result', 'error', 'warnings', 'events', 'prelude', 'state_unchanged'):
        assert actual[name] == reference[name], name


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    assert args.reference_only and sys.flags.optimize == 0
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    args.output.mkdir(); np, original, info = runtime()
    (args.output/'runtime.json').write_text(json.dumps(info, indent=2)+'\n')
    rows = []
    with (args.output/'journal.jsonl').open('x') as stream:
        for spec in CASES:
            row = observe(original, spec, np); rows.append(row)
            stream.write(json.dumps(row)+'\n'); stream.flush()
    assert len(rows) == 51 and not any(r['actual_C_returns'] for r in rows)
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    summary = dict(cases=len(rows), returns=sum(r['error'] is None for r in rows),
                   errors=sum(r['error'] is not None for r in rows), candidate_calls=0, image_sampling_calls=0,
                   linspace_calls={r['id']: r['linspace_calls'] for r in rows})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('original-only coordinate holdouts', len(rows), 'PASS', flush=True)


if __name__ == '__main__': main()
