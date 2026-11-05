"""Circle-AA defaults from translated saved arrays, with no original calls."""
import base64
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest
if __package__:
    from .test_circle_perimeter_aa_public import strict_owned
else:
    from test_circle_perimeter_aa_public import strict_owned

ROOT = Path(__file__).parents[1]
API = 'draw.circle_perimeter_aa'
CARRIERS = ('uint8', 'float32', 'float64')


@pytest.fixture(scope='module')
def saved_rows():
    return json.loads((ROOT / 'coverage/contracts/circle-perimeter-aa.json').read_bytes())['original_observations']


def translated_saved_output(saved_rows, np, case_id='circle-aa/radius-2'):
    row = next(r for r in saved_rows[np.__version__] if r['id'] == case_id)
    assert row['error'] is None
    arrays = [np.frombuffer(zlib.decompress(base64.b64decode(a['raw_zlib_base64'])),
                           dtype=np.dtype(a['dtype'])).reshape(a['shape']).copy()
              for a in row['result']['items']]
    # Translation is exact integer addition, with every point inside8x8.
    arrays[0] += 3
    arrays[1] += 3
    assert all(np.all((a >= 0) & (a < 8)) for a in arrays[:2])
    return tuple(arrays)


def test_circle_aa_default_registry_and_actual_public_export():
    import skimagexx as xx
    from skimagexx import _circle_perimeter_aa as owned
    from skimagexx.coverage import native_functions, native_dtypes, parity_tolerance
    assert API in native_functions() and native_dtypes(API) == CARRIERS
    assert xx.draw.circle_perimeter_aa is owned.circle_perimeter_aa
    assert 'circle_perimeter_aa' in xx.draw.__all__
    assert str(inspect.signature(xx.draw.circle_perimeter_aa)) == '(r, c, radius, shape=None)'
    with pytest.raises(ValueError, match='exact coordinate/weight words'):
        parity_tolerance(API, 'float64')


@pytest.mark.parametrize('dtype', CARRIERS)
@pytest.mark.parametrize('workers', (1, 4))
def test_circle_aa_default_saved_public_pipeline(dtype, workers, saved_rows, monkeypatch):
    import numpy as np
    from skimagexx.coverage import default_call, assert_default_parity
    carrier = np.arange(64, dtype=np.dtype(dtype)).reshape(8, 8)[:, ::-1]
    carrier = carrier.view(np.dtype(dtype, metadata={'role': 'descriptor carrier'}))
    before = (carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data)
    expected = translated_saved_output(saved_rows, np)
    with strict_owned(monkeypatch) as (xx, owned):
        xx.set_num_threads(workers)
        args, options = default_call(API, carrier)
        assert args == (3, 3, 2) and options == {'shape': (8, 8)}
        returns = []
        entry = owned._core.draw_circle_perimeter_aa
        previous = sys.getprofile()

        def profile(frame, event, value):
            if event == 'c_return' and value is entry:
                returns.append('draw_circle_perimeter_aa')

        sys.setprofile(profile)
        try:
            actual = xx.draw.circle_perimeter_aa(*args, **options)
        finally:
            sys.setprofile(previous)
        assert returns == ['draw_circle_perimeter_aa']
        assert_default_parity(API, dtype, actual, expected, image=carrier)
        assert before == (carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data)


@pytest.mark.parametrize('checker', ('exact', 'shared'))
@pytest.mark.parametrize('fault', ('weight-ulp', 'weight-dtype', 'coordinate-dtype',
                                  'view', 'container', 'missing'))
def test_circle_aa_existing_weight_checker_rejects_corrupt_defaults(checker, fault, saved_rows):
    import numpy as np
    from skimagexx.coverage import draw_default_error, assert_default_parity
    carrier = np.zeros((8, 8), dtype=np.float64)
    expected = translated_saved_output(saved_rows, np)
    actual = [a.copy() for a in expected]
    if fault == 'weight-ulp':
        index = int(np.flatnonzero(actual[2] < 1)[0])
        actual[2][index] = np.nextafter(actual[2][index], np.inf)
    elif fault == 'weight-dtype':
        actual[2] = actual[2].astype(np.float32)
    elif fault == 'coordinate-dtype':
        actual[0] = actual[0].astype(np.int32)
    elif fault == 'view':
        actual[0] = actual[0].view()
    elif fault == 'missing':
        actual = actual[:2]
    value = actual if fault == 'container' else tuple(actual)
    with pytest.raises(AssertionError):
        if checker == 'exact':
            draw_default_error(API, carrier, value, expected)
        else:
            assert_default_parity(API, 'float64', value, expected, image=carrier)


@pytest.mark.parametrize('checker', ('exact', 'shared'))
def test_circle_aa_existing_checker_rejects_alias_even_with_equal_words(checker, saved_rows):
    import numpy as np
    from skimagexx.coverage import draw_default_error, assert_default_parity
    expected = translated_saved_output(saved_rows, np, 'circle-aa/radius-0')
    assert expected[0].tobytes() == expected[1].tobytes()
    first = expected[0].copy()
    actual = (first, first, expected[2].copy())
    with pytest.raises(AssertionError):
        if checker == 'exact':
            draw_default_error(API, np.zeros((8, 8)), actual, expected)
        else:
            assert_default_parity(API, 'float64', actual, expected, image=np.zeros((8, 8)))


@pytest.mark.parametrize('index', (0, 1, 2))
def test_circle_aa_existing_recursive_output_checker_accepts_each_typed_array(index, saved_rows):
    import numpy as np
    from skimagexx.coverage import draw_default_error
    expected = translated_saved_output(saved_rows, np)
    assert draw_default_error(API, np.zeros((8, 8)), expected[index].copy(), expected[index]) == 0
