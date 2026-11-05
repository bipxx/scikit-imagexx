"""NumPy-only installed intake; keep separate from pure prebuild selectors."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT/'coverage/contracts/regionprops-coords-scaled-benchmark-v1.json').read_bytes())


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    loader = importlib.util.spec_from_file_location('coords_scaled_numpy_intake', ROOT/'benchmarks/regionprops_coords_scaled.py')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    def denied(*a, **kw):
        raise AssertionError('CLI/timer/regionprops calls forbidden in NumPy-only intake')
    monkeypatch.setattr(module, 'main', denied)
    monkeypatch.setattr(module, 'measure', denied, raising=False)
    if 'montage_sustained' in sys.modules:
        monkeypatch.setattr(sys.modules['montage_sustained'], 'measure', denied)
    assert not any(name.split('.')[0] in {'scipy','skimage','skimagexx'} for name in sys.modules)
    return module


@pytest.mark.parametrize('spec', CONTRACT['cases'], ids=lambda s:s['id'])
def test_coords_scaled_numpy_fixture_once(spec, benchmark):
    image, options = benchmark.fixture(spec, np)
    assert type(image) is np.ndarray and image.shape == tuple(spec['shape'])
    assert image.dtype == np.dtype(spec['dtype']) and image.dtype.isnative
    assert image.flags.owndata and image.base is None and image.flags.writeable and image.flags.aligned
    assert (image.flags.c_contiguous, image.flags.f_contiguous) == (spec['layout']=='C', spec['layout']=='F')
    assert options == dict(cache=spec['cache'], spacing=None if spec['spacing'] is None else tuple(spec['spacing']), offset=None if spec['offset'] is None else tuple(spec['offset']))
    assert type(options['cache']) is bool
    assert int(image.max()) == spec['max_label'] and int(image.min()) >= 0
    if spec['pattern'] == 'sparse':
        flat = image.ravel(order='C')
        positions = np.flatnonzero(flat)
        assert positions.size == 96 and positions.tolist() == list(range(0,image.size,257))
        assert flat[positions].tolist() == [(i//257)%8+1 for i in positions.tolist()]
        assert [(flat==label).sum().item() for label in range(1,9)] == [12]*8
    elif spec['pattern'] == 'checkerboard':
        assert set(np.unique(image).tolist()) == {1,2}
        assert np.count_nonzero(image==1) == np.count_nonzero(image==2) == image.size//2
        assert image[(0,)*image.ndim] == 1
        for axis,n in enumerate(image.shape):
            edge=[0]*image.ndim;edge[axis]=n-1
            assert image[tuple(edge)] == 1+(n-1)%2
    else:
        assert np.count_nonzero(image==1) == (3 if image.ndim==2 else 4)
        assert np.count_nonzero(image==2) == 1
    # Only finite source-domain bounds; no coordinates/property algorithm.
    spacing = (1.,)*image.ndim if spec['spacing'] is None else spec['spacing']
    offset = (0,)*image.ndim if spec['offset'] is None else spec['offset']
    assert all(abs((n-1)*step)+abs(shift) <= 1540 for n,step,shift in zip(image.shape,spacing,offset))


@pytest.mark.parametrize('dtype', ['float64','int64'])
def test_coords_scaled_numpy_checker_positive_owning_f(dtype, benchmark):
    expected = np.array([[-3,6],[-4,8]], dtype=dtype, order='F')
    actual = np.array(expected, copy=True, order='F')
    buffer = np.zeros((2,2),dtype='uint8')
    result = benchmark.checked_output(actual, expected, [buffer], np)
    assert result['arrays'][0]['raw'] == actual.tobytes(order='C').hex()
    assert result['arrays'][0]['f'] and not result['arrays'][0]['c']
    assert actual.flags.owndata and expected.flags.owndata


@pytest.mark.parametrize('fault', ['same-object','shared-storage','input-alias','wrong-dtype','wrong-shape','one-ULP','wrong-layout','view','external-nonowning','empty-self'])
def test_coords_scaled_numpy_checker_rejects_corruption(fault, benchmark):
    expected = np.array([[-3.5,6.],[-4.,8.]], dtype=np.float64, order='F')
    actual = np.array(expected, copy=True, order='F')
    buffers=[]
    if fault=='same-object':
        actual=expected
    elif fault=='shared-storage':
        actual=expected.view()
        assert actual is not expected and np.shares_memory(actual,expected)
    elif fault=='input-alias':
        buffers=[actual]
    elif fault=='wrong-dtype':
        actual=np.array(expected,dtype=np.float32,order='F')
    elif fault=='wrong-shape':
        actual=np.zeros((2,3),dtype=np.float64,order='F')
    elif fault=='one-ULP':
        actual[0,0]=np.nextafter(actual[0,0],np.float64(0.))
    elif fault=='wrong-layout':
        actual=np.array(expected,order='C')
    elif fault=='view':
        actual=actual.view()
        assert not np.shares_memory(actual,expected)
    elif fault=='external-nonowning':
        actual=np.ndarray((2,2),dtype=np.float64,buffer=bytearray(32),order='F')
        actual[:]=expected
        assert not actual.flags.owndata
    else:
        expected=np.empty((0,2),dtype=np.float64);actual=expected
        assert not np.shares_memory(actual,expected)
    with pytest.raises(AssertionError):
        benchmark.checked_output(actual,expected,buffers,np)
