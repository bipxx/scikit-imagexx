"""Function proof on the actual installed public core; explicit private adapters remain separately labeled."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import skimagexx as xx

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load('_matrix_frozen_contract', ROOT/'tests/test_matrix_transform_contracts.py')
PRIVATE_BACKEND = os.environ.get('MATRIX_TRANSFORM_PRIVATE_BACKEND')
native = (load('skimagexx._matrix_transform_candidate', ROOT/'python/skimagexx/_matrix_transform.py')
          if PRIVATE_BACKEND else importlib.import_module('skimagexx._matrix_transform'))
BACKEND = (load('_matrix_math_core', Path(os.environ['MATRIX_TRANSFORM_PRIVATE_BACKEND']))
           if os.environ.get('MATRIX_TRANSFORM_PRIVATE_BACKEND') else xx._core)
# Explicit genuine private-module adapter; never a public registry/core proof.
if BACKEND is not xx._core:
    native._core = BACKEND
ENTRY = BACKEND.matrix_transform_run
ENTRY_V2 = BACKEND.matrix_transform_run_v2
assert inspect.isbuiltin(ENTRY_V2) and ENTRY_V2.__module__ == BACKEND.__name__
assert inspect.isbuiltin(ENTRY) and ENTRY.__module__ == BACKEND.__name__
assert native._core is BACKEND
# Exact prior376f9 certified-ID superset: preserving all old native dtype/
# shape admissions independently of gains in other domains.
OLD_CPP_IDS=frozenset((
    'call/keyword-call',
    'coordinates/b/C',
    'coordinates/b/F',
    'coordinates/b/endian',
    'coordinates/b/readonly',
    'coordinates/b/reverse',
    'coordinates/b/step',
    'coordinates/b/unaligned',
    'coordinates/f2/C',
    'coordinates/f2/F',
    'coordinates/f2/endian',
    'coordinates/f2/readonly',
    'coordinates/f2/reverse',
    'coordinates/f2/step',
    'coordinates/f2/unaligned',
    'coordinates/f4/C',
    'coordinates/f4/F',
    'coordinates/f4/endian',
    'coordinates/f4/readonly',
    'coordinates/f4/reverse',
    'coordinates/f4/step',
    'coordinates/f4/unaligned',
    'coordinates/f8/C',
    'coordinates/f8/F',
    'coordinates/f8/endian',
    'coordinates/f8/readonly',
    'coordinates/f8/reverse',
    'coordinates/f8/step',
    'coordinates/f8/unaligned',
    'coordinates/i1/C',
    'coordinates/i1/F',
    'coordinates/i1/endian',
    'coordinates/i1/readonly',
    'coordinates/i1/reverse',
    'coordinates/i1/step',
    'coordinates/i1/unaligned',
    'coordinates/i2/C',
    'coordinates/i2/F',
    'coordinates/i2/endian',
    'coordinates/i2/readonly',
    'coordinates/i2/reverse',
    'coordinates/i2/step',
    'coordinates/i2/unaligned',
    'coordinates/i4/C',
    'coordinates/i4/F',
    'coordinates/i4/endian',
    'coordinates/i4/readonly',
    'coordinates/i4/reverse',
    'coordinates/i4/step',
    'coordinates/i4/unaligned',
    'coordinates/i8/C',
    'coordinates/i8/F',
    'coordinates/i8/endian',
    'coordinates/i8/readonly',
    'coordinates/i8/reverse',
    'coordinates/i8/step',
    'coordinates/i8/unaligned',
    'coordinates/u1/C',
    'coordinates/u1/F',
    'coordinates/u1/endian',
    'coordinates/u1/readonly',
    'coordinates/u1/reverse',
    'coordinates/u1/step',
    'coordinates/u1/unaligned',
    'coordinates/u2/C',
    'coordinates/u2/F',
    'coordinates/u2/endian',
    'coordinates/u2/readonly',
    'coordinates/u2/reverse',
    'coordinates/u2/step',
    'coordinates/u2/unaligned',
    'coordinates/u4/C',
    'coordinates/u4/F',
    'coordinates/u4/endian',
    'coordinates/u4/readonly',
    'coordinates/u4/reverse',
    'coordinates/u4/step',
    'coordinates/u4/unaligned',
    'coordinates/u8/C',
    'coordinates/u8/F',
    'coordinates/u8/endian',
    'coordinates/u8/readonly',
    'coordinates/u8/reverse',
    'coordinates/u8/step',
    'coordinates/u8/unaligned',
    'dtype-metadata/coords',
    'dtype-metadata/matrix',
    'geometry/D2/identity/N1',
    'geometry/D2/identity/N64',
    'geometry/D2/identity/N7',
    'geometry/D2/projective/N1',
    'geometry/D2/projective/N64',
    'geometry/D2/projective/N7',
    'geometry/D2/scale/N1',
    'geometry/D2/scale/N64',
    'geometry/D2/scale/N7',
    'geometry/D2/translation/N1',
    'geometry/D2/translation/N64',
    'geometry/D2/translation/N7',
    'geometry/D2/zero/N1',
    'geometry/D2/zero/N64',
    'geometry/D2/zero/N7',
    'geometry/D3/identity/N1',
    'geometry/D3/identity/N64',
    'geometry/D3/identity/N7',
    'geometry/D3/projective/N1',
    'geometry/D3/projective/N64',
    'geometry/D3/projective/N7',
    'geometry/D3/scale/N1',
    'geometry/D3/scale/N64',
    'geometry/D3/scale/N7',
    'geometry/D3/translation/N1',
    'geometry/D3/translation/N64',
    'geometry/D3/translation/N7',
    'geometry/D3/zero/N1',
    'geometry/D3/zero/N64',
    'geometry/D3/zero/N7',
    'input/coords/array-protocol',
    'input/coords/list',
    'input/coords/matrix-subclass',
    'input/coords/memoryview',
    'input/coords/tuple',
    'input/matrix/array-protocol',
    'input/matrix/list',
    'input/matrix/matrix-subclass',
    'input/matrix/memoryview',
    'input/matrix/tuple',
    'matrix/b/C',
    'matrix/b/F',
    'matrix/b/endian',
    'matrix/b/readonly',
    'matrix/b/reverse',
    'matrix/b/step',
    'matrix/b/unaligned',
    'matrix/f2/C',
    'matrix/f2/F',
    'matrix/f2/endian',
    'matrix/f2/readonly',
    'matrix/f2/reverse',
    'matrix/f2/step',
    'matrix/f2/unaligned',
    'matrix/f4/C',
    'matrix/f4/F',
    'matrix/f4/endian',
    'matrix/f4/readonly',
    'matrix/f4/reverse',
    'matrix/f4/step',
    'matrix/f4/unaligned',
    'matrix/f8/C',
    'matrix/f8/F',
    'matrix/f8/endian',
    'matrix/f8/readonly',
    'matrix/f8/reverse',
    'matrix/f8/step',
    'matrix/f8/unaligned',
    'matrix/i1/C',
    'matrix/i1/F',
    'matrix/i1/endian',
    'matrix/i1/readonly',
    'matrix/i1/reverse',
    'matrix/i1/step',
    'matrix/i1/unaligned',
    'matrix/i2/C',
    'matrix/i2/F',
    'matrix/i2/endian',
    'matrix/i2/readonly',
    'matrix/i2/reverse',
    'matrix/i2/step',
    'matrix/i2/unaligned',
    'matrix/i4/C',
    'matrix/i4/F',
    'matrix/i4/endian',
    'matrix/i4/readonly',
    'matrix/i4/reverse',
    'matrix/i4/step',
    'matrix/i4/unaligned',
    'matrix/i8/C',
    'matrix/i8/F',
    'matrix/i8/endian',
    'matrix/i8/readonly',
    'matrix/i8/reverse',
    'matrix/i8/step',
    'matrix/i8/unaligned',
    'matrix/u1/C',
    'matrix/u1/F',
    'matrix/u1/endian',
    'matrix/u1/readonly',
    'matrix/u1/reverse',
    'matrix/u1/step',
    'matrix/u1/unaligned',
    'matrix/u2/C',
    'matrix/u2/F',
    'matrix/u2/endian',
    'matrix/u2/readonly',
    'matrix/u2/reverse',
    'matrix/u2/step',
    'matrix/u2/unaligned',
    'matrix/u4/C',
    'matrix/u4/F',
    'matrix/u4/endian',
    'matrix/u4/readonly',
    'matrix/u4/reverse',
    'matrix/u4/step',
    'matrix/u4/unaligned',
    'matrix/u8/C',
    'matrix/u8/F',
    'matrix/u8/endian',
    'matrix/u8/readonly',
    'matrix/u8/reverse',
    'matrix/u8/step',
    'matrix/u8/unaligned',
    'shape/[2, 3]/square4',
    'shape/accepted-Nx4',
    'shape/list2/identity',
    'shape/list2/none',
    'shape/list3/square4',
))
ROUTES=[]
GENERAL_PENDING = (
    dict(id='large-image-coordinates',coords=[[1024.25,512.5],[4095.75,2047.25]],
         matrix=[[1.,0.,320.],[0.,1.,-240.],[0.,0.,1.]]),
    dict(id='ordinary-affine-coefficients',coords=[[0.123456789,-0.34567891],[1234.567,2345.678]],
         matrix=[[0.9987623,-0.03456789,12.3456],[0.04567891,1.0012345,-7.6543],[0.,0.,1.]]),
    dict(id='ordinary-projective-coefficients',coords=[[31.125,15.875],[1023.75,511.25]],
         matrix=[[0.9937,0.0123,15.67],[-0.0234,1.0045,7.89],[0.0001234,-0.0002345,1.]]),
    dict(id='ordinary-3D-transform',coords=[[16.125,32.25,64.5],[-128.125,256.25,512.5]],
         matrix=[[1.000123,0.,0.,1.23],[0.,0.999876,0.,-2.34],[0.,0.,1.000345,3.45],
                 [0.0000123,-0.0000234,0.0000345,1.]]),
)
GENERAL_ROWS=[]
PIPELINE_GENERAL_ROWS=[]
LAST_ATTEMPTS=0
NONFINITE_ROWS=[]


@pytest.fixture(scope='session',autouse=True)
def retained_route_receipt():
    yield
    output=os.environ.get('MATRIX_TRANSFORM_PROOF_OUTPUT')
    if output:
        core=Path(BACKEND.__file__).resolve()
        Path(output).write_text(json.dumps(dict(schema='matrix-private-function-routes-v1',
            core=str(core),core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
            numpy=np.__version__,profile=contract.profile(),entry_owner=ENTRY.__module__,
            candidate_source_sha256=hashlib.sha256((ROOT/'python/skimagexx/_matrix_transform.py').read_bytes()).hexdigest(),
            rows=ROUTES,general_cpp_required_pending=GENERAL_ROWS,general_pipeline=PIPELINE_GENERAL_ROWS,nonfinite_provider=NONFINITE_ROWS,
            certificate='authenticated versioned finite F64 general math; broader runtime/protocol domains provider-only',
            private_backend_adapter=BACKEND is not xx._core,
            public_registration=not bool(PRIVATE_BACKEND),operation_timings=0),indent=2)+'\n')


@pytest.fixture(params=(1,4))
def workers(request):
    previous = BACKEND.get_num_threads(); fallback = xx.set_fallback_enabled(False)
    if request.param == 4 and not BACKEND.has_openmp:
        with pytest.raises(ValueError, match='OpenMP'):
            BACKEND.set_num_threads(4)
        xx.set_fallback_enabled(fallback)
        pytest.skip('portable worker4 rejected; not a parallel parity pass')
    BACKEND.set_num_threads(request.param)
    try: yield request.param
    finally: BACKEND.set_num_threads(previous); xx.set_fallback_enabled(fallback)


def guarded_observation(spec):
    """Count actual builtin calls and reject every upstream transform frame."""
    return guarded(lambda:contract.observe(native.matrix_transform,spec))


def guarded(call):
    global LAST_ATTEMPTS
    calls=[]
    before=BACKEND.matrix_transform_completed()
    previous=sys.getprofile()
    def guard(frame,event,arg):
        if event == 'c_call' and (arg is ENTRY or arg is ENTRY_V2):
            calls.append('real-checked-C++')
        if event == 'call' and frame.f_globals.get('__name__','').startswith('skimage.transform'):
            raise AssertionError('upstream transform algorithm/class forwarding: '+frame.f_code.co_name)
    sys.setprofile(guard)
    try: result=call()
    finally: sys.setprofile(previous)
    LAST_ATTEMPTS=len(calls)
    completed=BACKEND.matrix_transform_completed()-before
    assert completed in (0,1) and completed<=len(calls)
    return result,['real-checked-C++']*completed


def general_observation(function,spec):
    coords=np.array(spec['coords'],np.float64);matrix=np.array(spec['matrix'],np.float64)
    before=(coords.tobytes(),matrix.tobytes())
    result=function(coords,matrix)
    assert before==(coords.tobytes(),matrix.tobytes())
    return contract.array_observation(result)


@pytest.mark.parametrize('spec',GENERAL_PENDING,ids=lambda s:s['id'])
def test_ordinary_general_values_have_exact_real_cpp_parity(spec,workers):
    expected=general_observation(contract.original.matrix_transform,spec)
    actual,calls=guarded(lambda:general_observation(native.matrix_transform,spec))
    assert actual==expected and calls==['real-checked-C++']
    GENERAL_ROWS.append(dict(id=spec['id'],workers=workers,cpp_calls=len(calls),scope='actual-general-C++-private-bridge',
        input_sha256=contract.digest(spec),original=expected,actual=actual))


@pytest.mark.parametrize('spec',GENERAL_PENDING,ids=lambda s:s['id'])
def test_ordinary_general_values_require_real_cpp_execution(spec,workers):
    expected=general_observation(contract.original.matrix_transform,spec)
    actual,calls=guarded(lambda:general_observation(native.matrix_transform,spec))
    assert actual==expected
    assert calls==['real-checked-C++']


@pytest.mark.parametrize('spec',contract.cases(),ids=lambda s:s['id'])
def test_every_frozen_function_observation_on_actual_core(spec,workers):
    expected=contract.matrix_data()['profiles'][contract.profile()]['observations'][spec['id']]
    actual,calls=guarded_observation(spec)
    assert actual == expected, spec['id']
    assert len(calls)<=1
    ROUTES.append(dict(id=spec['id'],workers=workers,cpp_calls=len(calls),
                       entry_attempts=LAST_ATTEMPTS,
                       route='checked-C++' if calls else 'owned-NumPy/metadata',
                       error=actual['error'] is not None,observation_sha256=contract.digest(actual)))
    # Provider/protocol cases must not merely skip the frozen observations.
    if spec.get('provider') or spec.get('order') or spec.get('input_kind')=='array-fail':
        assert not calls
    if spec['id'] in OLD_CPP_IDS:
        assert calls==['real-checked-C++']


@pytest.mark.parametrize('spec',[
    dict(id='cert-identity',model='identity'),
    dict(id='cert-translation',model='translation'),
    dict(id='cert-scale',model='scale'),
    dict(id='cert-projective',model='projective'),
    dict(id='cert-zero',model='zero'),
    dict(id='cert-Nx3',dimensions=3,model='projective'),
    dict(id='cert-unaligned',layout='unaligned',model='translation'),
    dict(id='cert-reverse',layout='reverse',model='translation'),
    dict(id='general-near-zero',model='near-zero'),
    dict(id='general-negativezero',model='negative-zero'),
])
def test_certified_cases_execute_one_real_cpp_call(spec,workers):
    expected=contract.observe(contract.original.matrix_transform,spec)
    actual,calls=guarded_observation(spec)
    assert actual==expected and calls==['real-checked-C++']


@pytest.mark.parametrize('spec',[
    dict(id='provider-empty',n=0),
    dict(id='provider-object',dtype='O'),dict(id='provider-complex',dtype='c16'),
    dict(id='provider-hooks',provider='array',provider_mode='observe'),
])
def test_uncertified_values_and_hooks_use_owned_numpy_only(spec,workers):
    expected=contract.observe(contract.original.matrix_transform,spec)
    actual,calls=guarded_observation(spec)
    assert actual==expected and not calls


def test_backend_identity_and_default_policy_are_not_a_registration_claim():
    from skimagexx.coverage import native_functions
    assert (contract.API not in native_functions()) if PRIVATE_BACKEND else tuple(native_functions()).count(contract.API)==1
    assert native._core is BACKEND and native._checked_entry() is ENTRY
    assert inspect.isbuiltin(BACKEND.set_num_threads)
    assert inspect.isbuiltin(BACKEND.get_num_threads) and BACKEND.get_num_threads()==1
    assert inspect.signature(native.matrix_transform)==inspect.signature(contract.original.matrix_transform)
    assert native.matrix_transform is not contract.original.matrix_transform
    assert native.matrix_transform.__module__==('skimagexx._matrix_transform_candidate' if PRIVATE_BACKEND else 'skimagexx._matrix_transform')
    if not PRIVATE_BACKEND:
        assert native is importlib.import_module('skimagexx._matrix_transform') and BACKEND is xx._core
        assert native.matrix_transform is xx.transform.matrix_transform
        assert xx.set_num_threads is BACKEND.set_num_threads and xx.get_num_threads is BACKEND.get_num_threads


@pytest.mark.parametrize('policy',('ignore','raise','call'))
def test_nondefault_fp_policy_keeps_literal_owned_providers(policy,workers):
    image=np.array([[1.,2.],[3.,4.]]);matrix=np.eye(3)
    previous=np.seterr(all=policy)
    try:
        assert native._native(matrix,image) is None
        expected=contract.original.matrix_transform(image,matrix)
        actual=native.matrix_transform(image,matrix)
        assert actual.tobytes()==expected.tobytes()
    finally: np.seterr(**previous)


def test_unknown_arithmetic_runtime_and_missing_capability_keep_provider(monkeypatch,workers):
    image=np.array([[1.123,2000.234],[3000.345,4.456]]);matrix=np.eye(3)
    expected=contract.original.matrix_transform(image,matrix)
    monkeypatch.setattr(native,'_runtime_profile',lambda:None)
    actual,calls=guarded(lambda:native.matrix_transform(image,matrix))
    assert actual.tobytes()==expected.tobytes() and not calls


@pytest.mark.parametrize('attribute,value',(
    ('matrix_transform_interface','matrix-transform-math-v0'),
    ('matrix_transform_interface',None),
    ('matrix_transform_capabilities',30),
    ('matrix_transform_capabilities',None)))
def test_unqualified_real_bridge_metadata_keeps_owned_provider(monkeypatch,workers,attribute,value):
    image=np.array([[1.123,2000.234],[3000.345,4.456]]);matrix=np.eye(3)
    expected=contract.original.matrix_transform(image,matrix)
    monkeypatch.setattr(BACKEND,attribute,value)
    actual,calls=guarded(lambda:native.matrix_transform(image,matrix))
    assert actual.tobytes()==expected.tobytes() and not calls and LAST_ATTEMPTS==0


@pytest.mark.parametrize('target',('numpy','umath'))
def test_runtime_binary_origin_missing_fails_closed_without_image_call(monkeypatch,target):
    import importlib
    module=np if target=='numpy' else importlib.import_module(native._PREFIX+'._multiarray_umath')
    native._runtime_profile.cache_clear()
    monkeypatch.setattr(module,'__file__',None)
    try:assert native._runtime_profile() is None
    finally:native._runtime_profile.cache_clear()


@pytest.mark.parametrize('size',(16,32,1024,16384))
def test_unqualified_iterator_buffer_settings_keep_owned_provider(size,workers):
    previous=np.setbufsize(size)
    try:
        image=np.array([[1.123,2000.234],[3000.345,4.456]]);matrix=np.eye(3)
        expected=contract.original.matrix_transform(image,matrix)
        actual,calls=guarded(lambda:native.matrix_transform(image,matrix))
        assert actual.tobytes()==expected.tobytes() and not calls
    finally:np.setbufsize(previous)


def test_intake_callback_restoring_ones_preserves_actual_altered_padding(workers):
    real=np.ones;events=[]
    class Input:
        def __array__(self,*args,**kwargs):
            def replacement(*a,**k):
                events.append('ones');np.ones=real
                result=real(*a,**k);result.fill(2.);return result
            np.ones=replacement
            return np.array([[1.123,2.234],[3.345,4.456]])
    try:
        expected=contract.original.matrix_transform(Input(),np.eye(3));assert events==['ones']
        events.clear();actual,calls=guarded(lambda:native.matrix_transform(Input(),np.eye(3)))
        assert actual.tobytes()==expected.tobytes() and not calls and events==['ones']
    finally:np.ones=real


@pytest.mark.parametrize('dtype',('f2','>f2','f4','>f4','f8','>f8'))
def test_nonfinite_bit_preflight_has_no_floating_status_or_mutation(dtype):
    m=np.zeros((3,3),dtype=dtype)
    words=m.view(np.dtype('u'+str(m.itemsize)).newbyteorder(m.dtype.byteorder))
    before=m.tobytes()
    with np.errstate(all='raise'):
        assert not native._has_nonfinite_matrix(m)
    assert m.tobytes()==before
    words[0,0]={2:0x7c01,4:0x7f800001,8:0x7ff0000000000001}[m.itemsize]
    before=m.tobytes()
    with np.errstate(all='raise'):
        assert native._has_nonfinite_matrix(m)
    assert m.tobytes()==before


@pytest.mark.parametrize('dtype',('f2','>f2','f4','>f4','f8','>f8'))
def test_nonfinite_matrix_keeps_original_warning_events_without_extra_cast(dtype,workers):
    def observed(function):
        m=np.eye(3,dtype=dtype)
        u=np.dtype('u'+str(m.itemsize)).newbyteorder(m.dtype.byteorder)
        m.view(u)[0,0]={2:0x7c01,4:0x7f800001,8:0x7ff0000000000001}[m.itemsize]
        p=np.array([[1.123,2.234],[3.345,4.456]]);before=(p.tobytes(),m.tobytes())
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');result=function(p,m)
        assert before==(p.tobytes(),m.tobytes())
        return dict(result=contract.array_observation(result),warnings=[
            dict(category=type(w.message).__name__,message=str(w.message)) for w in notices])
    expected=observed(contract.original.matrix_transform)
    actual,calls=guarded(lambda:observed(native.matrix_transform))
    NONFINITE_ROWS.append(dict(dtype=dtype,workers=workers,original=expected,actual=actual,
                              cpp_completed=len(calls),entry_attempts=LAST_ATTEMPTS))
    # This tiny provider comparison retains every raw word, including NaNs.
    # It adds no general NaN-encoding promise to the finite native contract.
    assert actual==expected and not calls



def test_module_import_does_not_trust_an_already_replaced_source_provider(monkeypatch,workers):
    real=np.array;events=[]
    def replacement(*args,**kwargs):
        events.append('array');return real(*args,**kwargs)
    monkeypatch.setattr(np,'array',replacement)
    module=load('skimagexx._matrix_import_hook_probe',ROOT/'python/skimagexx/_matrix_transform.py')
    module._core=BACKEND
    image=real([[1.,2.]]);matrix=np.eye(3)
    events.clear();actual=module.matrix_transform(image,matrix)
    assert events==['array'] and actual.tobytes()==real([[1.,2.]]).tobytes()
    assert not module._pristine()


def test_replaced_dtype_factory_is_not_called_to_screen_admission(monkeypatch,workers):
    image=np.array([[1.,2.]]);matrix=np.eye(3);events=[]
    def forbidden(*args,**kwargs):
        events.append('dtype');raise AssertionError('dtype factory should not screen admission')
    monkeypatch.setattr(np,'dtype',forbidden)
    expected=contract.original.matrix_transform(image,matrix)
    actual=native.matrix_transform(image,matrix)
    assert actual.tobytes()==expected.tobytes() and not events
    assert not native._pristine()


def _general_cases():
    text=(ROOT/'tests/test_matrix_transform_native.cpp').read_text()
    spec=json.loads(text.split('// MATRIX_COHORTS_BEGIN\n/*\n')[1].split('\n*/')[0])
    for case in spec['cases']:
        for n in [0]+spec['rows']:
            for pl in spec['layouts']:
                for ml in spec['layouts']:
                    yield dict(case=case,n=n,pl=pl,ml=ml,extension=spec['row_extension'])


def _layout(a,name):
    if name=='C':return np.array(a,order='C')
    if name=='F':return np.array(a,order='F')
    if name=='reverse':return a[::-1].copy()[::-1]
    if name=='step':
        b=np.zeros((a.shape[0]*2,a.shape[1]*2));b[::2,::2]=a;return b[::2,::2]
    b=np.zeros(a.nbytes+1,np.uint8);c=np.ndarray(a.shape,dtype='f8',buffer=b,offset=1);c[:]=a;return c


@pytest.mark.parametrize('spec',list(_general_cases()),ids=lambda s:'%s/N%s/%s/%s'%(s['case']['id'],s['n'],s['pl'],s['ml']))
def test_all_4000_general_cohorts_end_to_end_actual_cpp(spec,workers):
    c=spec['case'];n=spec['n']
    coords=np.array(c['coords']) if n==0 else np.array([
        np.array(c['coords'][i%len(c['coords'])])+i*spec['extension'] for i in range(n)])
    coords=_layout(coords,spec['pl']);matrix=_layout(np.array(c['matrix'],dtype='f8'),spec['ml'])
    before=(coords.tobytes(),matrix.tobytes())
    expected=contract.array_observation(contract.original.matrix_transform(coords,matrix))
    actual,calls=guarded(lambda:contract.array_observation(native.matrix_transform(coords,matrix)))
    row=dict(id=c['id'],n=n,points_layout=spec['pl'],matrix_layout=spec['ml'],workers=workers,
             cpp_completed=len(calls),entry_attempts=LAST_ATTEMPTS,
             expected_sha256=contract.digest(expected),actual_sha256=contract.digest(actual))
    if actual!=expected or calls!=['real-checked-C++']:
        row.update(expected=expected,actual=actual)
    PIPELINE_GENERAL_ROWS.append(row)
    assert actual==expected and calls==['real-checked-C++']
    assert before==(coords.tobytes(),matrix.tobytes())


@pytest.mark.parametrize('key',native._FORCED_CPU)
def test_explicit_cpu_forcing_rejects_cached_certificate_without_algorithm(monkeypatch,key):
    # Metadata guard only: even a previously authenticated cache cannot bypass
    # a newly introduced forcing policy. No original/native image call here.
    native._runtime_profile()
    monkeypatch.setenv(key,'Haswell')
    assert native._runtime_profile() is None
    assert native._THREAD_QUERY is None and native._RUNTIME_CORENAME is None


def test_actual_loaded_runtime_certificate_records_real_unforced_owner():
    version=native._runtime_profile()
    assert version==np.__version__ and native._THREAD_QUERY()==1
    cert=native._authenticate_runtime()
    assert cert[1] in native._RUNTIME_CORES
    assert native._RUNTIME_CORENAME in native._RUNTIME_CORES[cert[1]]
    assert len(cert[-1])==2
    assert all(Path(p).is_file() and Path(p).stat().st_size==size for p,size,stamp in cert[-1])


@pytest.mark.parametrize('attribute,value',(
    ('matrix_transform_v2_interface',None),
    ('matrix_transform_v2_interface','matrix-transform-math-v1'),
    ('matrix_transform_v2_capabilities',31),
    ('matrix_transform_v2_capabilities',62),
    ('matrix_transform_v2_capabilities',None)))
def test_v2_negotiation_is_strict_and_legacy_entry_is_independent(monkeypatch,attribute,value):
    assert native._checked_entry(6) is ENTRY_V2
    monkeypatch.setattr(BACKEND,attribute,value)
    assert native._checked_entry(6) is None
    assert native._checked_entry() is ENTRY


def test_unqualified_v2_callable_cannot_impersonate_native_entry(monkeypatch):
    monkeypatch.setattr(BACKEND,'matrix_transform_run_v2',lambda *a:None)
    assert native._checked_entry(6) is None and native._checked_entry() is ENTRY


def test_wrong_loaded_symbol_owner_rejects_before_metadata_getter_calls(monkeypatch):
    cert=native._authenticate_runtime()
    assert cert is not None
    # Synthetic metadata rejection control, never arithmetic/runtime acceptance.
    library=cert[4]
    if sys.platform=='win32':
        actual_path=Path(cert[-1][1][0])
    else:
        actual_path=None
    monkeypatch.setattr(native,'_loaded_blas',lambda *a:(library,actual_path,lambda f:Path(BACKEND.__file__).resolve()))
    native._runtime_profile.cache_clear()
    try:assert native._runtime_profile() is None
    finally:native._runtime_profile.cache_clear()


def test_wrong_bundled_library_bytes_fail_closed(monkeypatch):
    pins={k:('0'*64,v[1]) for k,v in native._RUNTIME_PINS.items()}
    monkeypatch.setattr(native,'_RUNTIME_PINS',pins)
    native._runtime_profile.cache_clear()
    try:assert native._runtime_profile() is None
    finally:native._runtime_profile.cache_clear()


@pytest.mark.parametrize('name',('Unknown','Cooperlake-unqualified'))
def test_unknown_live_corename_fails_closed_with_otherwise_valid_metadata(monkeypatch,name):
    certificate=native._authenticate_runtime()
    assert certificate is not None
    fields=list(certificate);fields[2]=lambda:name.encode('ascii')
    # Explicit synthetic control; no mutation of actual vendor worker/coretype.
    monkeypatch.setattr(native,'_authenticate_runtime',lambda:tuple(fields))
    assert native._runtime_profile() is None


@pytest.mark.parametrize('version,core,expected',(
    ('1.24.4','Haswell',4),('1.24.4','Zen',4),
    ('1.24.4','Cooperlake',6),('2.5.4','SkylakeX',6),('2.5.4','Haswell',4)))
def test_audited_prepared_shape_selection_is_not_runtime_authentication(version,core,expected):
    h=np.ones((1,3));matrix=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],order='F')
    assert native._prepared_profile(h,matrix,version,core)==expected
    # Aligned matrixC instead reaches transposed GEMV for every audited core.
    assert native._prepared_profile(h,matrix.copy(order='C'),version,core)==3
    assert native._prepared_profile(np.ones((2,3)),matrix,version,core)==1


@pytest.mark.parametrize('name',('LD_PRELOAD','LD_AUDIT'))
def test_unknown_linker_interposition_rejects_ordinary_certificate(monkeypatch,name):
    # Never executed or loaded: explicit environmental admission metadata.
    monkeypatch.setenv(name,'/unqualified/not-loaded.so')
    assert native._runtime_profile() is None
    assert native._THREAD_QUERY is None and native._RUNTIME_CORENAME is None


@pytest.mark.parametrize('name',('LD_PRELOAD','LD_AUDIT'))
def test_empty_linker_variable_does_not_invent_an_interposer(monkeypatch,name):
    monkeypatch.setenv(name,'')
    assert native._runtime_profile()==np.__version__
