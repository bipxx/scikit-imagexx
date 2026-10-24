"""Original matrix_transform contracts; own class coverage is not function proof."""
import hashlib
from functools import lru_cache
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pytest
from skimage import transform as original
import skimage

API = 'transform.matrix_transform'
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/matrix-transform.json'
DTYPES = ('b', 'u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8', 'f2', 'f4', 'f8', 'c8', 'c16', 'O')


def cases():
    rows = []
    for dtype in DTYPES:
        for layout in ('C', 'F', 'reverse', 'step', 'readonly', 'unaligned', 'endian'):
            if dtype == 'O' and layout == 'unaligned': continue
            rows.append(dict(id=f'coordinates/{dtype}/{layout}', dtype=dtype, layout=layout, model='projective'))
    for dtype in DTYPES:
        for layout in ('C', 'F', 'reverse', 'step', 'readonly', 'unaligned', 'endian'):
            if dtype == 'O' and layout == 'unaligned': continue
            rows.append(dict(id=f'matrix/{dtype}/{layout}', matrix_dtype=dtype, matrix_layout=layout, model='translation'))
    for dimensions in (2, 3):
        for model in ('identity', 'translation', 'scale', 'projective', 'zero', 'negative-zero', 'near-zero'):
            for count in (0, 1, 7, 64):
                rows.append(dict(id=f'geometry/D{dimensions}/{model}/N{count}', dimensions=dimensions, n=count, model=model))
    for shape in ('empty-list', 'scalar', 'none', 'list2', 'list3', [0, 2], [0, 3], [2, 0], [2, 1], [2, 3], [2, 4], [2, 2, 2]):
        for matrix in ('identity', 'none', 'square4', 'nonsquare'):
            rows.append(dict(id=f'shape/{shape}/{matrix}', shape=shape, matrix_shape=matrix))
    for matrix in ('scalar', 'vector', 'square0', 'square1', 'square2', 'square5', 'rank3'):
        rows.append(dict(id='matrix-shape/' + matrix, matrix_shape=matrix))
    for kind in ('list', 'tuple', 'memoryview', 'matrix-subclass', 'array-protocol', 'array-fail'):
        for target in ('coords', 'matrix'):
            rows.append(dict(id=f'input/{target}/{kind}', input_target=target, input_kind=kind))
    for target in ('coords', 'matrix'):
        for failure in (False, True):
            rows.append(dict(id=f'object-scalar/{target}/fail{failure}', scalar_target=target, scalar_fail=failure))
    for hook in ('array', 'asarray', 'ones', 'hstack', 'where', 'finfo', 'eye'):
        for mode in ('observe', 'raise'):
            rows.append(dict(id=f'provider/{hook}/{mode}', provider=hook, provider_mode=mode,
                             matrix_shape='none' if hook == 'eye' else 'identity'))
    for mode in ('matrix-fail-before-coords', 'shape-error-before-coords', 'fill-coords-at-asarray', 'fill-matrix-at-array'):
        rows.append(dict(id='order/' + mode, order=mode))
    for kind in ('missing-matrix', 'missing-coords', 'extra-positional', 'channel-none', 'duplicate-coords', 'keyword-call'):
        rows.append(dict(id='call/' + kind, call=kind))
    for dtype in ('U2', 'S2', 'M8[D]', 'm8[D]', 'record'):
        for target in ('coords', 'matrix'):
            rows.append(dict(id=f'dtype-error/{target}/{dtype}', special_dtype=dtype, special_target=target))
    for dtype in ('i8', 'u8'):
        rows.append(dict(id='wide-integer/' + dtype, dtype=dtype, wide=True))
    rows.append(dict(id='shape/accepted-Nx4', shape=[2,4], matrix_shape='square5'))
    for target in ('coords','matrix'):
        rows.append(dict(id='dtype-metadata/'+target, dtype_metadata=target))
    assert len({s['id'] for s in rows}) == len(rows)
    return rows


class ContractError(RuntimeError):
    pass


ContractError.__module__ = 'matrix_transform_contract_fixture'


def scalar(value):
    if type(value) in (bool, int, str) or value is None:
        return value
    if type(value) is float:
        return dict(type='float', hex=value.hex())
    if type(value) is complex:
        return dict(type='complex', real=value.real.hex(), imag=value.imag.hex())
    if type(value) in (list,tuple): return [scalar(v) for v in value]
    if type(value) is dict: return {k:scalar(v) for k,v in value.items()}
    if isinstance(value,np.ndarray): return dict(type=type(value).__name__,shape=list(value.shape),dtype=value.dtype.str)
    if isinstance(value,type): return dict(type='class',module=value.__module__,name=value.__qualname__)
    if isinstance(value,np.dtype): return dict(type='dtype',str=value.str)
    if isinstance(value, np.generic):
        return dict(type=type(value).__name__, dtype=value.dtype.str, bytes=value.tobytes().hex())
    if isinstance(value, Coefficient):
        return dict(type='fixture-coefficient', name=value.name, value=value.value)
    return dict(type=type(value).__name__)


class Coefficient:
    def __init__(self, name, value, events, fail=False):
        self.name, self.value, self.events, self.fail = name, value, events, fail

    def __mul__(self, other):
        self.events.append(dict(operation='mul', name=self.name, other=scalar(other)))
        if self.fail:
            raise ContractError('coefficient-mul:' + self.name)
        return self.value * other

    def __rmul__(self, other):
        self.events.append(dict(operation='rmul', name=self.name, other=scalar(other)))
        if self.fail:
            raise ContractError('coefficient-rmul:' + self.name)
        return other * self.value


def array_observation(a):
    result = dict(dtype=a.dtype.str, shape=list(a.shape), strides=list(a.strides),
                  C=bool(a.flags.c_contiguous), F=bool(a.flags.f_contiguous),
                  owns=bool(a.flags.owndata), writable=bool(a.flags.writeable),
                  base_type=None if a.base is None else type(a.base).__name__,
                  dtype_descr=json.loads(json.dumps(a.dtype.descr)), dtype_metadata=None if a.dtype.metadata is None else dict(a.dtype.metadata))
    if a.dtype.hasobject:
        result['cells'] = [scalar(x) for x in a.flat]
    else:
        result['bytes'] = a.tobytes(order='C').hex()
    return result


def layout(a, name):
    if name == 'F':
        return np.asfortranarray(a)
    if name == 'reverse':
        return a[..., ::-1]
    if name == 'step':
        owner = np.empty((a.shape[0]*2,) + a.shape[1:], a.dtype); owner[::2] = a
        return owner[::2]
    if name == 'readonly':
        a.setflags(write=False)
    elif name == 'unaligned' and not a.dtype.hasobject:
        owner = np.zeros(a.nbytes+1, np.uint8); result = np.ndarray(a.shape, a.dtype, buffer=owner, offset=1)
        result[...] = a
        return result
    elif name == 'endian':
        return a.astype(a.dtype.newbyteorder('>' if sys.byteorder == 'little' else '<'))
    return a


def fixture(spec):
    events = []
    dimensions = spec.get('dimensions', 2); n = spec.get('n', 5)
    dtype = np.dtype(spec.get('dtype', 'f8'))
    values = ((np.arange(n*dimensions).reshape(n, dimensions)*3) % 17) / 4
    if dtype.kind in 'ifcO':
        values = values - 2
    if dtype.kind == 'c':
        values = values + 0.125j
    if spec.get('wide'):
        values = np.array([np.iinfo(dtype).max, 2**62, 1, 0, 2**63-2, 2**60, 17, 3, 5, 7], dtype=dtype).reshape(5, 2)
    coords = np.asarray(values, dtype=dtype)
    matrix = np.eye(dimensions+1); model = spec.get('model', 'identity')
    if model == 'translation':
        matrix[:-1, -1] = np.arange(1, dimensions+1)*2
    elif model == 'scale':
        for i in range(dimensions): matrix[i, i] = 2**i
    elif model in ('projective', 'near-zero'):
        matrix[-1, 0] = 0.125
        matrix[0, -1] = 2
        if model == 'near-zero': matrix[-1, -1] = 2**-40
    elif model in ('zero', 'negative-zero'):
        matrix[-1, :] = -0. if model == 'negative-zero' else 0.
    matrix = np.asarray(matrix, dtype=np.dtype(spec.get('matrix_dtype', 'f8')))
    shape = spec.get('shape')
    if isinstance(shape, list): coords = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    elif shape == 'empty-list': coords = []
    elif shape == 'scalar': coords = 2.
    elif shape == 'none': coords = None
    elif shape == 'list2': coords = [1., 2.]
    elif shape == 'list3': coords = [1., 2., 3.]
    mshape = spec.get('matrix_shape', 'identity')
    if mshape == 'none': matrix = None
    elif mshape == 'scalar': matrix = np.asarray(1.)
    elif mshape == 'vector': matrix = np.array([1., 2., 3.])
    elif mshape == 'nonsquare': matrix = np.ones((2, 3))
    elif mshape == 'rank3': matrix = np.ones((2, 2, 2))
    elif mshape.startswith('square'): matrix = np.eye(int(mshape[6:]))
    if spec.get('dtype_metadata')=='coords':coords=coords.view(np.dtype(coords.dtype,metadata={'unit':'coordinate','origin':1}))
    if spec.get('dtype_metadata')=='matrix':matrix=matrix.view(np.dtype(matrix.dtype,metadata={'unit':'coefficient','origin':2}))
    if isinstance(coords, np.ndarray): coords = layout(coords, spec.get('layout', 'C'))
    if isinstance(matrix, np.ndarray): matrix = layout(matrix, spec.get('matrix_layout', 'C'))
    target = spec.get('special_target')
    if target:
        a = coords if target == 'coords' else matrix; name = spec['special_dtype']
        if name == 'record': a = np.zeros(a.shape, dtype=[('x', 'f4'), ('y', 'i2')])
        elif name.startswith(('M', 'm')): a = np.zeros(a.shape, dtype=name)
        else: a = np.full(a.shape, '1', dtype=name)
        if target == 'coords': coords = a
        else: matrix = a
    target = spec.get('scalar_target')
    if target:
        a = np.asarray(coords if target == 'coords' else matrix, dtype=object)
        a.flat[0] = Coefficient(target+'0', 1.25, events, spec['scalar_fail'])
        if target == 'coords': coords = a
        else: matrix = a
    storage = (coords, matrix)
    class ArrayInput:
        def __init__(self, value, name, fail=False): self.value, self.name, self.fail = value, name, fail
        def __array__(self, *args, **kwargs):
            events.append(dict(operation='array-protocol', name=self.name,
                               args=[scalar(x) for x in args], kwargs={k:scalar(v) for k,v in kwargs.items()}))
            if self.fail: raise ContractError(self.name+'-array-fail')
            return self.value
    target = spec.get('input_target'); kind = spec.get('input_kind')
    if target:
        a = coords if target == 'coords' else matrix
        if kind == 'list': a = a.tolist()
        elif kind == 'tuple': a = tuple(tuple(row) for row in a.tolist())
        elif kind == 'memoryview': a = memoryview(a)
        elif kind == 'matrix-subclass':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); a = np.matrix(a)
        else: a = ArrayInput(a, target, kind == 'array-fail')
        if target == 'coords': coords = a
        else: matrix = a
    order = spec.get('order')
    if order == 'matrix-fail-before-coords':
        coords = ArrayInput(storage[0], 'coords', True); matrix = ArrayInput(storage[1], 'matrix', True)
    elif order == 'shape-error-before-coords':
        coords = ArrayInput(storage[0], 'coords', True); matrix = np.ones((2, 3)); storage = (storage[0], matrix)
    return coords, matrix, storage, events


def observe(function, spec):
    coords, matrix, storage, events = fixture(spec)
    before = [array_observation(a) if isinstance(a, np.ndarray) else scalar(a) for a in storage]
    passed_before=[array_observation(a) if isinstance(a,np.ndarray) else scalar(a) for a in (coords,matrix)]
    providers = {}
    names = ('array', 'asarray', 'ones', 'hstack', 'where', 'finfo', 'eye')
    hook = spec.get('provider'); order = spec.get('order')
    def provider(name, real):
        def call(*args, **kwargs):
            events.append(dict(operation='numpy-provider', name=name,
                               args=[scalar(a) for a in args],
                               kwargs={k:scalar(v) for k,v in kwargs.items()}))
            if name == hook and spec['provider_mode'] == 'raise': raise ContractError('numpy-'+name)
            if name == 'asarray' and order == 'fill-coords-at-asarray': storage[0].fill(0)
            if name == 'array' and order == 'fill-matrix-at-array': storage[1][0, -1] = 7
            return real(*args, **kwargs)
        return call
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            if hook or order in ('fill-coords-at-asarray', 'fill-matrix-at-array'):
                for name in names:
                    providers[name] = getattr(np, name); setattr(np, name, provider(name, providers[name]))
            call = spec.get('call')
            if call == 'missing-matrix': result = function(coords)
            elif call == 'missing-coords': result = function(matrix=matrix)
            elif call == 'extra-positional': result = function(coords, matrix, None)
            elif call == 'channel-none': result = function(coords, matrix, channel_axis=None)
            elif call == 'duplicate-coords': result = function(coords, matrix, coords=coords)
            elif call == 'keyword-call': result = function(coords=coords, matrix=matrix)
            else: result = function(coords, matrix)
            error = None
        except Exception as caught:
            result = None; error = dict(type=type(caught).__module__+'.'+type(caught).__qualname__, args=[scalar(a) for a in caught.args], message=str(caught))
        finally:
            for name, real in providers.items(): setattr(np, name, real)
    observed_events = list(events)
    observed = dict(error=error, warnings=[dict(type=type(w.message).__module__+'.'+type(w.message).__qualname__, message=str(w.message)) for w in notices], events=observed_events,
                    passed_before=passed_before, inputs_before=before, inputs_after=[array_observation(a) if isinstance(a,np.ndarray) else scalar(a) for a in storage])
    if isinstance(result, np.ndarray):
        observed['result'] = array_observation(result)
        observed['shares_input'] = [bool(np.shares_memory(result,a)) if isinstance(a,np.ndarray) else False for a in storage]
        observed['base_lifetime_copy'] = result.copy().tobytes().hex() if not result.dtype.hasobject else [scalar(a) for a in result.flat]
    else: observed['result'] = scalar(result)
    return observed


def profile():
    return platform.system().lower()+'-np'+np.__version__+'-py'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_facts():
    import skimage.transform._geometric as module
    names = ('matrix_transform', '_apply_homogeneous', '_append_homogeneous_dim')
    result = {name:inspect.getsource(getattr(module,name)) for name in names}
    for name in ('__init__', '_check_matrix', '_check_dims'):
        result['ProjectiveTransform.'+name] = inspect.getsource(getattr(module.ProjectiveTransform,name))
    return dict(signature=str(inspect.signature(original.matrix_transform)), numpy_copy_if_needed=module.NP_COPY_IF_NEEDED,
                source_utf8=result, source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in result.items()})


@lru_cache(maxsize=1)
def matrix_data():
    return json.loads(MATRIX.read_text(encoding='utf-8'))


@pytest.mark.parametrize('spec', cases(), ids=lambda s:s['id'])
def test_original_contract(spec):
    data = matrix_data()
    assert observe(original.matrix_transform, spec) == data['profiles'][profile()]['observations'][spec['id']]


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('spec', cases(), ids=lambda s:s['id'])
def test_native_pending_function_contract(spec, workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if API not in native_functions():
        pytest.xfail('native matrix_transform function pending; own ProjectiveTransform class is not function acceptance')
    previous = xx.set_fallback_enabled(False); old_workers = xx.get_num_threads()
    try:
        function = xx.transform.matrix_transform
        assert function is not original.matrix_transform and inspect.unwrap(function).__module__.startswith('skimagexx.')
        assert inspect.signature(function) == inspect.signature(original.matrix_transform)
        xx.set_num_threads(workers)
        assert observe(function, spec) == matrix_data()['profiles'][profile()]['observations'][spec['id']]
    finally:
        xx.set_num_threads(old_workers); xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('matrix,points,expected', [
    (np.eye(3), [[1,2],[-2,3]], [[1,2],[-2,3]]),
    ([[1,0,4],[0,1,-2],[0,0,1]], [[1,2],[-2,3]], [[5,0],[2,1]]),
    ([[2,0,0],[0,3,0],[0,0,1]], [[1,2],[-2,3]], [[2,6],[-4,9]]),
    ([[1,0,0],[0,1,0],[1,0,1]], [[0,1],[1,2],[3,8]], [[0,1],[0.5,1],[0.75,2]]),
    ([[1,0,0],[0,1,0],[0,0,0]], [[1,2]], [[2.**52,2.**53]]),
    ([[1,0,0,4],[0,1,0,-2],[0,0,1,0.5],[0,0,0,1]], [[1,2,3]], [[5,0,3.5]]),
])
def test_independent_small_definitions(matrix, points, expected):
    result = original.matrix_transform(points, matrix)
    assert result.dtype == np.dtype('f8')
    assert result.tobytes() == np.array(expected, dtype='f8').tobytes()
    assert result.flags.owndata and result.base is None


def test_independent_conversion_and_provider_order():
    row=observe(original.matrix_transform,dict(order='matrix-fail-before-coords'))
    assert [e['name'] for e in row['events']]==['matrix']
    assert row['error']['message']=='matrix-array-fail'
    row=observe(original.matrix_transform,dict(order='shape-error-before-coords'))
    assert not row['events'] and row['error']['type']=='builtins.ValueError'
    row=observe(original.matrix_transform,dict(provider='array',provider_mode='observe'))
    assert [e['name'] for e in row['events']]==['asarray','array','ones','hstack','finfo','where']


def test_independent_empty_and_float_promotion():
    result=original.matrix_transform(np.zeros((0,3),np.float32),np.eye(4,dtype=np.float32))
    assert result.shape==(0,3) and result.dtype==np.dtype('f8') and result.flags.owndata
    result=original.matrix_transform(np.array([[1,2]],np.float16),np.eye(3,dtype=np.float16))
    assert result.dtype==np.dtype('f8') and result.tobytes()==np.array([[1.,2.]],np.float64).tobytes()


def test_signature_source_and_inventory_integrity():
    data = matrix_data()
    assert skimage.__version__ == data['reference']['version'] == '0.26.0'
    assert str(inspect.signature(original.matrix_transform)) == data['signature'] == '(coords, matrix)'
    assert data['case_contract_sha256'] == digest(cases()) and data['cases'] == cases()
    current = source_facts(); recorded = data['profiles'][profile()]['source']
    assert current == recorded
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==data['test_source_canonical_lf_sha256']
    assert data['native_function_implementations'] == 0 and data['native_contract_status'] == 'pending'


def freeze_reference(output):
    observations = {spec['id']:observe(original.matrix_transform,spec) for spec in cases()}
    result = dict(profile=profile(), python=sys.version, numpy=np.__version__, skimage=skimage.__version__,
                  source=source_facts(), observations=observations)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze-reference', required=True)
    freeze_reference(parser.parse_args().freeze_reference)
