"""Advertised line-ND defaults reconstructed from exact saved source rows."""
import base64
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest
if __package__:
    from .test_line_nd_public import strict_owned
    from .draw_coordinates_observer import encode
else:
    from test_line_nd_public import strict_owned
    from draw_coordinates_observer import encode
ROOT = Path(__file__).parents[1]
API = 'draw.line_nd'
DTYPES = ('float16','float32','float64','int8','uint8','int64','uint64')
CASE_IDS = {d:'line-nd/dtype-'+d for d in DTYPES if d != 'int64'}
CASE_IDS['int64'] = 'line-nd/2d-end-False-int-True'


@pytest.fixture(scope='module')
def saved():
    return json.loads((ROOT/'coverage/contracts/line-nd.json').read_bytes())['original_observations']


def saved_output(saved, np, dtype):
    row = next(r for r in saved[np.__version__] if r['id'] == CASE_IDS[dtype])
    assert row['error'] is None
    items = row['result']['items']
    first = items[0]
    base = np.ndarray(first['base_shape'], dtype=np.dtype(first['dtype']), strides=first['base_strides'])
    integer = not dtype.startswith('float')
    result = tuple(base) if integer else tuple(base.T)
    for value, item in zip(result, items):
        value[:] = np.frombuffer(zlib.decompress(base64.b64decode(item['raw_zlib_base64'])), dtype=np.dtype(item['dtype']))
    assert encode(result, np) == row['result']
    return result


def test_line_nd_registered_public_default_scope():
    import skimagexx as xx
    from skimagexx import _line_nd as owned
    from skimagexx.coverage import native_functions, native_dtypes, execution_kind, parity_tolerance
    assert API in native_functions() and native_dtypes(API) == DTYPES and execution_kind(API) == 'native'
    assert xx.draw.line_nd is owned.line_nd and 'line_nd' in xx.draw.__all__
    assert str(inspect.signature(xx.draw.line_nd)) == '(start, stop, *, endpoint=False, integer=True)'
    with pytest.raises(ValueError, match='exact coordinate words'):
        parity_tolerance(API, 'float32')


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('workers', (1,4))
def test_line_nd_actual_default_uses_frozen_row(dtype, workers, saved, monkeypatch):
    import numpy as np
    from skimagexx.coverage import default_call, assert_default_parity
    carrier = np.arange(64).reshape(8,8)[:,::-1].astype(dtype)
    carrier = carrier.view(np.dtype(dtype, metadata={'role':'coordinate dtype selector'}))
    before = carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data
    expected = saved_output(saved, np, dtype)
    with strict_owned(monkeypatch, workers) as (xx, owned):
        args, options = default_call(API, carrier)
        assert all(a.dtype == np.dtype(dtype) and a.dtype.metadata is None for a in args)
        assert options == {'endpoint':False,'integer':not dtype.startswith('float')}
        entry, getter = owned._core.draw_line_nd, owned._core.get_num_threads
        returns, accepted, snapshots = [], [], []
        def snapshot():
            value = getter(); snapshots.append(value); return value
        def profile(frame,event,value):
            if event == 'c_return' and value is entry:returns.append('draw_line_nd')
            if event == 'return' and frame.f_code is owned.line_nd.__code__:accepted.append(frame.f_locals.get('native_result') is not None)
        previous = sys.getprofile()
        with monkeypatch.context() as patch:
            patch.setattr(owned._core,'get_num_threads',snapshot)
            sys.setprofile(profile)
            try:actual = xx.draw.line_nd(*args,**options)
            finally:sys.setprofile(previous)
        assert inspect.isbuiltin(entry) and returns == ['draw_line_nd'] and accepted == [True] and snapshots == [workers]
        assert_default_parity(API,dtype,actual,expected,image=carrier)
        assert encode(actual,np) == encode(expected,np)
        for a,b in zip(actual,expected):assert a is not b and a.base is not b.base
    assert before == (carrier.tobytes(),carrier.shape,carrier.strides,carrier.dtype,carrier.ctypes.data)


@pytest.mark.parametrize('checker', ('exact','shared'))
@pytest.mark.parametrize('dtype', ('float32','int64'))
@pytest.mark.parametrize('fault', ('word','owning-rows','container','missing','base-strides'))
def test_line_nd_default_checker_rejects_corruption(checker,dtype,fault,saved):
    import numpy as np
    from skimagexx.coverage import line_nd_default_error,assert_default_parity
    expected = saved_output(saved,np,dtype)
    actual = saved_output(saved,np,dtype)
    if fault == 'word':actual[0][0] += 1
    elif fault == 'owning-rows':actual = tuple(a.copy() for a in actual)
    elif fault == 'container':actual = list(actual)
    elif fault == 'missing':actual = actual[:-1]
    else:
        base = np.array(actual[0].base,copy=True,order='F' if dtype=='float32' else 'C')
        actual = tuple(base.T) if dtype=='float32' else tuple(base)
    with pytest.raises(AssertionError):
        if checker == 'exact':line_nd_default_error(np.zeros((8,8)),actual,expected)
        else:assert_default_parity(API,dtype,actual,expected,image=np.zeros((8,8)))


@pytest.mark.parametrize('dtype', ('int16','uint16','int32'))
def test_line_nd_unadvertised_default_dtype_is_explicit(dtype):
    import numpy as np
    from skimagexx.coverage import default_call
    with pytest.raises(ValueError, match='outside the frozen partition'):
        default_call(API,np.zeros((8,8),dtype=dtype))
