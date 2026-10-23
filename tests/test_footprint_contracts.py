"""Frozen 0.26 footprint contracts; pending strict dispatch is not native proof.

Reference snapshots include exact container/scalar types, array values, layout,
ownership, input aliasing, warnings and errors. No performance is measured.
"""
import hashlib
import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import skimage
from skimage import morphology as reference

import skimagexx as xx
from skimagexx.coverage import native_functions


NAMES = ('footprint_rectangle', 'diamond', 'disk', 'ball',
         'mirror_footprint', 'pad_footprint')
CONTRACT_PATH = Path(__file__).resolve().parents[1] / 'coverage/contracts/footprints.json'
DTYPES = ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
          'uint64', 'int64', 'float16', 'float32', 'float64', 'complex64',
          'complex128', 'object', 'U3', 'S3', '>i2', 'structured', 'tagged')


def dtype_for(name):
    if name == 'structured':
        return np.dtype([('value', 'i2'), ('flag', '?')])
    if name == 'tagged':
        return np.dtype('uint8', metadata={'contract': 'footprint'})
    return np.dtype(name)


def corpus():
    cases = []

    def add(name, label, *args, **kwargs):
        cases.append((name + '/' + label, name, args, kwargs))

    for dtype in DTYPES:
        for name in NAMES[:4]:
            for decomposition in ([None, 'separable', 'sequence'] if name == 'footprint_rectangle'
                                  else [None, 'sequence', 'crosses'] if name == 'disk'
                                  else [None, 'sequence']):
                arg = (3, 5) if name == 'footprint_rectangle' else 3
                add(name, f'dtype={dtype}/decomposition={decomposition}', arg,
                    dtype=dtype_for(dtype), decomposition=decomposition)
    for shape in [(), (1,), (3,), (2, 4), (1, 5), (3, 5, 7), (0, 3),
                  (-1, 3), (2.5, 3), (True, 3), [3, 5], 3, None]:
        for decomposition in [None, 'sequence', 'separable', 'invalid']:
            add('footprint_rectangle', f'shape={shape!r}/decomposition={decomposition}',
                shape, decomposition=decomposition)
    for name in ['diamond', 'disk', 'ball']:
        for radius in [-2, -1, -.5, 0, .5, 1, 1.5, 2, 5, 10, True,
                       np.int64(3), np.float64(2), '2', None, np.nan, np.inf]:
            for decomposition in [None, 'sequence', 'crosses', 'invalid']:
                add(name, f'radius={type(radius).__name__}:{str(radius)}/decomposition={decomposition}',
                    radius, decomposition=decomposition)
        # The table boundary and out-of-table behavior are part of the contract.
        for radius in [250, 251]:
            add(name, f'table-radius={radius}', radius, decomposition='sequence')
        for dtype in ['uint8', 'bool', 'float32', 'U3']:
            add(name, f'invalid-dtype={dtype}', 2, dtype='not-a-dtype' if dtype == 'U3' else dtype,
                decomposition='invalid')
    for name in ['disk', 'ball']:
        for strict in [False, True, None, 0, 2]:
            for decomposition in [None, 'sequence', 'crosses']:
                add(name, f'strict={strict!r}/decomposition={decomposition}', 3,
                    strict_radius=strict, decomposition=decomposition)
    for name in NAMES[:4]:
        add(name, 'invalid-dtype', (3, 5) if name == 'footprint_rectangle' else 2,
            dtype='not-a-dtype')
    for name in NAMES[4:]:
        for dtype in DTYPES:
            for layout in ['C', 'F', 'reverse', 'readonly', 'broadcast']:
                image = np.ones((2, 3), dtype=dtype_for(dtype))
                if image.dtype.fields is None:
                    image[0, 0] = 0
                if layout == 'F': image = np.asfortranarray(image)
                elif layout == 'reverse': image = image[::-1, ::-1]
                elif layout == 'readonly': image.flags.writeable = False
                elif layout == 'broadcast': image = np.broadcast_to(image[:1], (2, 3))
                add(name, f'dtype={dtype}/layout={layout}', image)
        for shape in [(), (0,), (1,), (2,), (0, 2), (3, 5), (2, 4, 6), (1, 2, 3, 4)]:
            for pad_end in [False, True, None, 2]:
                options = {'pad_end': pad_end} if name == 'pad_footprint' else {}
                add(name, f'shape={shape}/pad_end={pad_end!r}', np.ones(shape), **options)
        a, b = np.arange(6).reshape(2, 3), np.ones((3, 1), dtype=bool)
        sequences = [(), ((a, 2),), [(a, np.int64(0)), (b, -2)],
                     ((a, True), (a, 3)), ((a, 1.5),), ((a,),), [[1, 2]],
                     (((a, 1), 2),), ((((a, 1),), 2),), None, 3, 'abc',
                     [1, 2], [[1, 0], [0, 1]]]
        for index, sequence in enumerate(sequences):
            add(name, f'sequence={index}', sequence)
    assert len({case[0] for case in cases}) == len(cases)
    return cases


CASES = corpus()


def arrays(value):
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in arrays(item)]
    return []


def describe(value, inputs):
    if isinstance(value, np.ndarray):
        # repr(list) is deterministic for these finite 0/1/coordinate fixtures,
        # including object, structured, Unicode and bytes dtypes.
        payload = repr(value.tolist()).encode('utf-8')
        return dict(kind='ndarray', type=type(value).__name__, dtype=value.dtype.str,
                    descr=repr(value.dtype.descr), metadata=repr(value.dtype.metadata),
                    shape=list(value.shape), strides=list(value.strides),
                    c=bool(value.flags.c_contiguous), f=bool(value.flags.f_contiguous),
                    owned=bool(value.flags.owndata), writeable=bool(value.flags.writeable),
                    values_sha256=hashlib.sha256(payload).hexdigest(),
                    input_identity=[value is a for a in inputs],
                    input_alias=[bool(np.shares_memory(value, a)) for a in inputs])
    if isinstance(value, (tuple, list)):
        return dict(kind=type(value).__name__, items=[describe(x, inputs) for x in value])
    return dict(kind=type(value).__name__, value=repr(value.item() if isinstance(value, np.generic) else value))


def observe(function, args, kwargs):
    inputs = arrays(args)
    before = [describe(a, []) for a in inputs]
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result = describe(function(*args, **kwargs), inputs)
            error = None
        except Exception as caught:
            result = None
            error = dict(type=type(caught).__name__, message=str(caught))
    assert [describe(a, []) for a in inputs] == before, 'footprint input was mutated'
    return dict(result=result, error=error,
                warnings=[dict(type=type(w.message).__name__, message=str(w.message))
                          for w in notices])


def snapshot():
    return {key: observe(getattr(reference, name), args, kwargs)
            for key, name, args, kwargs in CASES}


@pytest.fixture(scope='module')
def frozen():
    assert skimage.__version__ == '0.26.0', 'contracts require pinned original scikit-image'
    return json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', NAMES)
def test_frozen_signature(frozen, name):
    expected = frozen['functions'][name]['signature']
    assert str(inspect.signature(getattr(reference, name))) == expected
    assert str(inspect.signature(getattr(xx.morphology, name))) == expected


def test_fixed_corpus_manifest(frozen):
    baseline = frozen['snapshots']['numpy2']
    assert set(baseline) == {case[0] for case in CASES}
    assert hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest() == frozen['corpus_sha256']
    for profile, overrides in frozen['snapshot_overrides'].items():
        assert set(overrides) <= set(baseline), profile
    for digest, observation in frozen['observations'].items():
        assert hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest() == digest
    for name in NAMES:
        assert frozen['functions'][name]['case_count'] == sum(case[1] == name for case in CASES)


@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_original_frozen_contract(frozen, case):
    key, name, args, kwargs = case
    profile = 'numpy1' if np.__version__.split('.')[0] == '1' else 'numpy2'
    expected = frozen['snapshots']['numpy2'].copy()
    expected.update(frozen['snapshot_overrides'].get(profile, {}))
    actual = observe(getattr(reference, name), args, kwargs)
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    assert actual == frozen['observations'][expected[key]], (key, digest)


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('mode', ['compatibility', 'native'])
@pytest.mark.parametrize('case', CASES, ids=[c[0] for c in CASES])
def test_public_footprint_contract(case, mode, workers):
    key, name, args, kwargs = case
    previous_fallback = xx.set_fallback_enabled(mode == 'compatibility')
    previous_workers = xx.get_num_threads()
    xx.set_num_threads(workers)
    try:
        expected = observe(getattr(reference, name), args, kwargs)
        actual = observe(getattr(xx.morphology, name), args, kwargs)
        pending = 'morphology.' + name not in native_functions()
        if mode == 'native' and pending:
            assert actual['error'] is not None, 'unregistered API executed in strict mode'
            assert actual['error']['type'] == 'NotImplementedError'
            assert 'fallback is disabled' in actual['error']['message']
            assert not actual['warnings']
            pytest.xfail('native morphology.' + name + ' is pending; forwarding is not proof')
        fallback = [w for w in actual['warnings'] if w['type'] == 'FallbackWarning']
        assert len(fallback) == (1 if mode == 'compatibility' and pending else 0)
        actual['warnings'] = [w for w in actual['warnings'] if w['type'] != 'FallbackWarning']
        assert actual == expected, key
    finally:
        xx.set_fallback_enabled(previous_fallback)
        xx.set_num_threads(previous_workers)
