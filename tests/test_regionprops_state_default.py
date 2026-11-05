"""Shipping defaults use saved original objects; no original algorithms run."""
import base64
import importlib
import inspect
import json
from pathlib import Path
import sys
import zlib
import pytest

ROOT = Path(__file__).parents[1]
API = 'measure.regionprops'
DTYPES = ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64')


@pytest.fixture(scope='module')
def saved_rows():
    return json.loads((ROOT / 'coverage/contracts/regionprops-state-v1.json').read_bytes())['original_observations']


@pytest.fixture
def saved_provider_factory(monkeypatch, saved_rows):
    import numpy as np
    from skimagexx import _regionprops_state as owned
    original = importlib.import_module('skimage.measure._regionprops')
    frozen = {}

    def decode(value):
        if not isinstance(value, dict):
            return value
        if value.get('type') == 'ndarray':
            source = np.frombuffer(zlib.decompress(base64.b64decode(value['raw_zlib_base64'])),
                                   dtype=np.dtype(value['dtype'])).reshape(value['shape'])
            result = np.empty(value['shape'], dtype=source.dtype,
                              order='F' if value['f'] and not value['c'] else 'C')
            result[...] = source
            assert result.strides == tuple(value['strides']) and result.flags.owndata
            return result
        if value.get('type') == 'slice':
            return slice(value['start'], value['stop'], value['step'])
        if 'container' in value:
            items = [decode(v) for v in value['items']]
            return tuple(items) if value['container'] == 'tuple' else items
        if 'raw' in value:
            return np.frombuffer(bytes.fromhex(value['raw']), dtype=np.dtype(value['dtype']))[0]
        raise AssertionError('Unexpected saved property encoding')

    def denied(*args, **kwargs):
        raise AssertionError('default test invoked an original constructor or numerical getter')

    monkeypatch.setattr(original, 'regionprops', denied)
    monkeypatch.setattr(importlib.import_module('skimage.measure'), 'regionprops', denied)
    monkeypatch.setattr(importlib.import_module('scipy.ndimage'), 'find_objects', denied)
    monkeypatch.setattr(owned, '_provider_regionprops', denied)
    for name in ('bbox', 'image', 'coords', 'num_pixels', 'area'):
        descriptor = getattr(original.RegionProperties, name)

        def replay(region, name=name, descriptor=descriptor):
            record = frozen.get(id(region))
            if record is not None:
                values, positions = record
                index = positions.get(name, 0)
                positions[name] = index + 1
                return values[name][index % len(values[name])]
            if name in ('image', 'coords', 'num_pixels'):
                return denied()
            # Area scalar arithmetic and bbox metadata remain declared providers.
            return descriptor.fget(region)

        monkeypatch.setattr(original.RegionProperties, name, property(replay))

    def make(dtype):
        case_id = ('regionprops-state/2d-cache-True' if dtype == 'int64'
                   else 'regionprops-state/dtype-' + dtype)
        row = next(r for r in saved_rows[np.__version__] if r['id'] == case_id)
        assert row['error'] is None
        result = []
        for source_region in row['regions']:
            steps = source_region['steps']
            obj = object.__new__(original.RegionProperties)
            object.__setattr__(obj, 'label', decode(steps[0]['result']))
            object.__setattr__(obj, 'slice', decode(steps[1]['result']))
            image = decode(steps[3]['result'])
            alias = next(a for a in row['array_aliases']
                         if a['left'] == [source_region['index'], 3]
                         and a['right'] == [source_region['index'], 7])
            assert alias['same_object'] and alias['shares_memory']
            values = dict(bbox=[decode(steps[2]['result'])], image=[image, image],
                          coords=[decode(steps[4]['result']), decode(steps[8]['result'])],
                          num_pixels=[decode(steps[5]['result'])], area=[decode(steps[6]['result'])])
            frozen[id(obj)] = (values, {})
            result.append(obj)
        return result

    return make


def test_regionprops_default_registry_and_public_export():
    import skimagexx as xx
    from skimagexx import _regionprops_state as owned
    from skimagexx.coverage import native_dtypes, native_functions, parity_tolerance
    assert API in native_functions() and native_dtypes(API) == DTYPES
    assert xx.measure.regionprops is owned.regionprops and 'regionprops' in xx.measure.__all__
    assert str(inspect.signature(xx.measure.regionprops)) == '(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None)'
    with pytest.raises(ValueError, match='exact property/storage'):
        parity_tolerance(API, 'int64')


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('workers', (1, 4))
def test_regionprops_default_saved_public_pipeline(dtype, workers, saved_provider_factory):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _compat, _regionprops_state as owned
    from skimagexx.coverage import assert_default_parity, default_call
    carrier = np.arange(24, dtype=np.dtype(dtype)).reshape(4, 6)[:, ::-1]
    carrier.setflags(write=False)
    before = (carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data)
    args, options = default_call(API, carrier)
    assert len(args) == 1 and args[0].shape == (3, 4) and args[0].dtype == carrier.dtype
    assert options == {'cache': True} and not np.shares_memory(args[0], carrier)
    expected = saved_provider_factory(dtype)
    entries = {getattr(owned._core, 'regionprops_state_' + name)
               for name in ('prepare', 'image', 'population', 'coords')}
    returns = []

    def profiler(frame, event, value):
        if event == 'c_return' and value in entries:
            returns.append(value)

    old_workers, old_fallback, old_profiler = xx.get_num_threads(), _compat._fallback_enabled, sys.getprofile()
    try:
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(False)
        sys.setprofile(profiler)
        actual = xx.measure.regionprops(*args, **options)
        assert_default_parity(API, dtype, actual, expected, image=carrier)
        assert set(returns) == entries
        assert before == (carrier.tobytes(), carrier.shape, carrier.strides, carrier.dtype, carrier.ctypes.data)
    finally:
        sys.setprofile(old_profiler)
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)


@pytest.mark.parametrize('checker', ('shared', 'exact'))
@pytest.mark.parametrize('fault', ('missing', 'container', 'wrong-class', 'label', 'image-word',
                                  'num-pixels', 'area-ulp', 'coords-dtype', 'coords-layout',
                                  'image-owner', 'cache-alias'))
def test_regionprops_default_checker_rejects_corruption(checker, fault, saved_provider_factory, monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _regionprops_state as owned
    from skimagexx.coverage import assert_default_parity, default_call, regionprops_default_error
    carrier = np.zeros((4, 6), dtype=np.int64)
    args, options = default_call(API, carrier)
    actual = xx.measure.regionprops(*args, **options)
    expected = saved_provider_factory('int64')
    region = actual[0]
    if fault == 'missing':
        actual = actual[:-1]
    elif fault == 'container':
        actual = tuple(actual)
    elif fault == 'wrong-class':
        actual[0] = expected[0]
    elif fault == 'label':
        region.label = 2
    elif fault == 'image-word':
        region.image[0, 0] ^= True
    elif fault == 'num-pixels':
        region._cache['num_pixels'] = np.intp(100)
    elif fault == 'area-ulp':
        region._cache['area'] = np.nextafter(np.float64(3), np.inf)
    elif fault in ('coords-dtype', 'coords-layout'):
        descriptor = owned.RegionProperties.coords

        def corrupt_coords(obj):
            value = descriptor.fget(obj)
            if obj is region:
                return value.astype(np.int32) if fault == 'coords-dtype' else value.copy(order='C')
            return value

        monkeypatch.setattr(owned.RegionProperties, 'coords', property(corrupt_coords))
    elif fault == 'image-owner':
        region._cache['image'] = region.image.view()
    else:
        descriptor = owned.RegionProperties.image
        monkeypatch.setattr(owned.RegionProperties, 'image',
                            property(lambda obj: descriptor.fget(obj).copy(order='K')))
    with pytest.raises(AssertionError):
        if checker == 'shared':
            assert_default_parity(API, 'int64', actual, expected, image=carrier)
        else:
            regionprops_default_error(carrier, actual, expected)
