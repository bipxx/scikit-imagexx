"""Actual line-AA registry/default/checker shipping routes from saved words."""
import base64
from contextlib import contextmanager
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import warnings
import zlib
import pytest
if __package__:
    from .test_line_aa_public import strict_owned
else:
    from test_line_aa_public import strict_owned

ROOT = Path(__file__).parents[1]
API = 'draw.line_aa'
CARRIERS = ('uint8', 'float32', 'float64')


@pytest.fixture(scope='module')
def saved_rows():
    # Immutable observations are parsed once per process, not once per carrier.
    return json.loads((ROOT/'coverage/contracts/line-aa.json').read_bytes())['original_observations']


def saved_output(saved_rows, np, case_id='line-aa/diagonal'):
    row = next(row for row in saved_rows[np.__version__] if row['id'] == case_id)
    assert row['error'] is None
    return tuple(np.frombuffer(zlib.decompress(base64.b64decode(a['raw_zlib_base64'])),
                              dtype=np.dtype(a['dtype'])).reshape(a['shape']).copy()
                 for a in row['result']['items'])


def test_line_aa_registry_public_export_and_shape_carrier_contract():
    import skimagexx as xx
    from skimagexx import _line_aa as owned
    from skimagexx.coverage import native_functions, native_dtypes, execution_kind, parity_tolerance
    assert API in native_functions() and execution_kind(API) == 'native'
    assert native_dtypes(API) == CARRIERS
    assert xx.draw.line_aa is owned.line_aa and 'line_aa' in xx.draw.__all__
    assert str(inspect.signature(xx.draw.line_aa)) == '(r0, c0, r1, c1)'
    with pytest.raises(ValueError, match='exact coordinate/weight words'):
        parity_tolerance(API, 'float64')


@pytest.mark.parametrize('dtype', CARRIERS)
@pytest.mark.parametrize('workers', (1, 4))
def test_line_aa_actual_default_pipeline_uses_saved_diagonal(dtype, workers, saved_rows, monkeypatch):
    import numpy as np
    from skimagexx.coverage import default_call, assert_default_parity
    # Tags/negative strides prove these are shape descriptors, not pixel inputs.
    carrier = np.arange(64, dtype=np.dtype(dtype)).reshape(8, 8)[:, ::-1]
    carrier = carrier.view(np.dtype(dtype, metadata={'role': 'shape carrier'}))
    before = (carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data)
    expected = saved_output(saved_rows, np)
    with strict_owned(monkeypatch) as (xx, owned):
        xx.set_num_threads(workers)
        args, options = default_call(API, carrier)
        assert args == (0, 0, 7, 7) and options == {}
        returns = []
        entry = owned._core.draw_line_aa
        previous = sys.getprofile()
        def profile(frame, event, value):
            if event == 'c_return' and value is entry: returns.append('draw_line_aa')
        sys.setprofile(profile)
        try:
            actual = xx.draw.line_aa(*args, **options)
        finally:
            sys.setprofile(previous)
        assert returns == ['draw_line_aa'] and xx.get_num_threads() == workers
        assert_default_parity(API, dtype, actual, expected, image=carrier)
    assert before == (carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data)


@pytest.mark.parametrize('checker', ('exact', 'shared'))
@pytest.mark.parametrize('fault', ('weight-ulp', 'weight-dtype', 'coordinate-dtype', 'view', 'container'))
def test_line_aa_existing_default_checker_rejects_corrupt_outputs(checker, fault, saved_rows):
    import numpy as np
    from skimagexx.coverage import draw_default_error, assert_default_parity
    carrier = np.zeros((8, 8), dtype=np.float64)
    expected = saved_output(saved_rows, np)
    actual = [a.copy() for a in expected]
    if fault == 'weight-ulp':
        index = int(np.flatnonzero(actual[2] < 1)[0])
        actual[2][index] = np.nextafter(actual[2][index], np.inf)
    elif fault == 'weight-dtype': actual[2] = actual[2].astype(np.float32)
    elif fault == 'coordinate-dtype': actual[0] = actual[0].astype(np.int32)
    elif fault == 'view': actual[0] = actual[0].view()
    value = actual if fault == 'container' else tuple(actual)
    with pytest.raises(AssertionError):
        if checker == 'exact': draw_default_error(API, carrier, value, expected)
        else: assert_default_parity(API, 'float64', value, expected, image=carrier)


@pytest.mark.parametrize('checker', ('exact', 'shared'))
def test_line_aa_existing_checker_rejects_alias_even_when_words_match(checker, saved_rows):
    import numpy as np
    from skimagexx.coverage import draw_default_error, assert_default_parity
    # Structural checker control: translate the cached one-pixel line from3,4
    # to3,3 so the two coordinate arrays have equal words without aliasing.
    expected = list(saved_output(saved_rows, np, 'line-aa/point'))
    expected[1][0] = expected[0][0]
    expected = tuple(expected)
    actual = (expected[0].copy(), expected[0].copy(), expected[2].copy())
    actual = (actual[0], actual[0], actual[2])
    assert all(a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()
               and a.flags.owndata and a.base is None for a, e in zip(actual, expected))
    with pytest.raises(AssertionError):
        if checker == 'exact': draw_default_error(API, np.zeros((8, 8)), actual, expected)
        else: assert_default_parity(API, 'float64', actual, expected, image=np.zeros((8, 8)))


@pytest.fixture(scope='module')
def coverage_benchmark():
    old = sys.path[:]
    sys.path.insert(0, str(ROOT/'benchmarks'))
    try:
        spec = importlib.util.spec_from_file_location('line_aa_actual_coverage_benchmark', ROOT/'benchmarks/coverage.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    assert Path(module.measure.__code__.co_filename).resolve() == (ROOT/'benchmarks/run.py').resolve()
    return module


@pytest.mark.parametrize('dtype', CARRIERS)
@pytest.mark.parametrize('fault', (None, 'weight-ulp'))
def test_line_aa_actual_generic_benchmark_precheck_selects_exact_route(dtype, fault, saved_rows,
                                                                    coverage_benchmark, monkeypatch, tmp_path):
    import numpy as np
    benchmark = coverage_benchmark
    expected = saved_output(saved_rows, np)
    reference_calls, checker_calls, measure_entries = [], [], []
    original_checker = benchmark.draw_default_error
    real_measure_code = benchmark.measure.__code__
    def cached_reference(*args, **options):
        assert args == (0, 0, 7, 7) and options == {}
        reference_calls.append(args)
        return tuple(a.copy() for a in expected)
    def exact_checker(api, image, actual, wanted):
        assert api == API and image.shape == (8, 8) and image.dtype.name == dtype
        before = image.tobytes()
        checker_calls.append(api)
        result = original_checker(api, image, actual, wanted)
        assert image.tobytes() == before
        return result
    class StopBeforeTimer(Exception): pass
    def trace(frame, event, value):
        if event == 'call' and frame.f_code is real_measure_code:
            measure_entries.append('real measure CALL before body')
            raise StopBeforeTimer
        return None
    output = tmp_path/'must-not-be-timing.json'
    with strict_owned(monkeypatch) as (xx, owned):
        public = importlib.import_module('skimage.draw')
        with monkeypatch.context() as patch, warnings.catch_warnings():
            patch.setattr(public, 'line_aa', cached_reference)
            patch.setattr(benchmark, 'native_dtypes', lambda api: (dtype,))
            patch.setattr(benchmark, 'draw_default_error', exact_checker)
            if fault:
                bad = [a.copy() for a in expected]
                index = int(np.flatnonzero(bad[2] < 1)[0])
                bad[2][index] = np.nextafter(bad[2][index], np.inf)
                patch.setattr(xx.draw, 'line_aa', lambda *args, **options: tuple(bad))
            patch.setattr(sys, 'argv', [str(ROOT/'benchmarks/coverage.py'), '--sizes', '8', '--threads', '1',
                                       '--repeat', '1', '--api', API, '--output', str(output)])
            previous = sys.gettrace()
            sys.settrace(trace)
            try:
                with pytest.raises(AssertionError if fault else StopBeforeTimer):
                    benchmark.main()
            finally:
                sys.settrace(previous)
    assert reference_calls == [(0, 0, 7, 7)] and checker_calls == [API]
    assert len(measure_entries) == (0 if fault else 1)
    assert not output.exists()
