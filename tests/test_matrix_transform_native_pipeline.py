"""Private function proof on a genuine installed core, never public acceptance."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

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
native = load('skimagexx._matrix_transform_candidate', ROOT/'python/skimagexx/_matrix_transform.py')
ENTRY = xx._core.projective_apply_checked
assert inspect.isbuiltin(ENTRY) and ENTRY.__module__ == xx._core.__name__
assert native._core is xx._core
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


@pytest.fixture(scope='session',autouse=True)
def retained_route_receipt():
    yield
    output=os.environ.get('MATRIX_TRANSFORM_PROOF_OUTPUT')
    if output:
        core=Path(xx._core.__file__).resolve()
        Path(output).write_text(json.dumps(dict(schema='matrix-private-function-routes-v1',
            core=str(core),core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
            numpy=np.__version__,profile=contract.profile(),entry_owner=ENTRY.__module__,
            candidate_source_sha256=hashlib.sha256((ROOT/'python/skimagexx/_matrix_transform.py').read_bytes()).hexdigest(),
            rows=ROUTES,general_cpp_required_pending=GENERAL_ROWS,
            certificate='partial: abs<=16,multiples2^-20,2..4dimensions; ordinarygeneralnative math remains required',
            public_registration=False,operation_timings=0),indent=2)+'\n')


@pytest.fixture(params=(1,4))
def workers(request):
    previous = xx.get_num_threads(); fallback = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    try: yield request.param
    finally: xx.set_num_threads(previous); xx.set_fallback_enabled(fallback)


def guarded_observation(spec):
    """Count actual builtin calls and reject every upstream transform frame."""
    return guarded(lambda:contract.observe(native.matrix_transform,spec))


def guarded(call):
    calls=[]
    previous=sys.getprofile()
    def guard(frame,event,arg):
        if event == 'c_call' and arg is ENTRY:
            calls.append('real-checked-C++')
        if event == 'call' and frame.f_globals.get('__name__','').startswith('skimage.transform'):
            raise AssertionError('upstream transform algorithm/class forwarding: '+frame.f_code.co_name)
    sys.setprofile(guard)
    try: result=call()
    finally: sys.setprofile(previous)
    return result,calls


def general_observation(function,spec):
    coords=np.array(spec['coords'],np.float64);matrix=np.array(spec['matrix'],np.float64)
    before=(coords.tobytes(),matrix.tobytes())
    result=function(coords,matrix)
    assert before==(coords.tobytes(),matrix.tobytes())
    return contract.array_observation(result)


@pytest.mark.parametrize('spec',GENERAL_PENDING,ids=lambda s:s['id'])
def test_ordinary_general_values_keep_exact_owned_provider_parity(spec,workers):
    expected=general_observation(contract.original.matrix_transform,spec)
    actual,calls=guarded(lambda:general_observation(native.matrix_transform,spec))
    assert actual==expected and not calls
    GENERAL_ROWS.append(dict(id=spec['id'],workers=workers,cpp_calls=0,scope='ownedprovideronly/CPP-requiredPENDING',
        input_sha256=contract.digest(spec),original=expected,actual=actual))


@pytest.mark.parametrize('spec',GENERAL_PENDING,ids=lambda s:s['id'])
def test_ordinary_general_values_require_future_cpp_execution(spec,workers):
    expected=general_observation(contract.original.matrix_transform,spec)
    actual,calls=guarded(lambda:general_observation(native.matrix_transform,spec))
    assert actual==expected
    if not calls:
        pytest.xfail('CPP-required pending: current exact dyadic certificate excludes '+spec['id'])
    assert calls==['real-checked-C++']


@pytest.mark.parametrize('spec',contract.cases(),ids=lambda s:s['id'])
def test_every_frozen_function_observation_on_actual_core(spec,workers):
    expected=contract.matrix_data()['profiles'][contract.profile()]['observations'][spec['id']]
    actual,calls=guarded_observation(spec)
    assert actual == expected, spec['id']
    assert len(calls)<=1
    ROUTES.append(dict(id=spec['id'],workers=workers,cpp_calls=len(calls),
                       route='checked-C++' if calls else 'owned-NumPy/metadata',
                       error=actual['error'] is not None,observation_sha256=contract.digest(actual)))
    # Provider/protocol cases must not merely skip the frozen observations.
    if spec.get('provider') or spec.get('order') or spec.get('input_kind')=='array-fail':
        assert not calls
    if spec['id']=='geometry/D2/translation/N7':
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
])
def test_certified_cases_execute_one_real_cpp_call(spec,workers):
    expected=contract.observe(contract.original.matrix_transform,spec)
    actual,calls=guarded_observation(spec)
    assert actual==expected and calls==['real-checked-C++']


@pytest.mark.parametrize('spec',[
    dict(id='provider-empty',n=0),dict(id='provider-near-zero',model='near-zero'),
    dict(id='provider-object',dtype='O'),dict(id='provider-complex',dtype='c16'),
    dict(id='provider-negativezero',model='negative-zero'),
    dict(id='provider-hooks',provider='array',provider_mode='observe'),
])
def test_uncertified_values_and_hooks_use_owned_numpy_only(spec,workers):
    expected=contract.observe(contract.original.matrix_transform,spec)
    actual,calls=guarded_observation(spec)
    assert actual==expected and not calls


def test_backend_identity_and_default_policy_are_not_a_registration_claim():
    from skimagexx.coverage import native_functions
    assert contract.API not in native_functions()
    assert native._core is xx._core and native._checked_entry() is ENTRY
    assert xx.set_num_threads is xx._core.set_num_threads
    assert xx.get_num_threads is xx._core.get_num_threads and xx.get_num_threads()==1
    assert inspect.signature(native.matrix_transform)==inspect.signature(contract.original.matrix_transform)
    assert native.matrix_transform is not contract.original.matrix_transform
    assert native.matrix_transform.__module__=='skimagexx._matrix_transform_candidate'


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


def test_grid_certificate_boundaries_are_exact_not_tolerance_based():
    assert native._grid(np.array([[16.,-16.,2.**-20,0.]],dtype='f8'))
    for value in (np.nextafter(16.,np.inf),2.**-21,np.nextafter(1.,2.),-0.,np.inf,np.nan):
        assert not native._grid(np.array([[value]],dtype='f8'))


def test_module_import_does_not_trust_an_already_replaced_source_provider(monkeypatch,workers):
    real=np.array;events=[]
    def replacement(*args,**kwargs):
        events.append('array');return real(*args,**kwargs)
    monkeypatch.setattr(np,'array',replacement)
    module=load('skimagexx._matrix_import_hook_probe',ROOT/'python/skimagexx/_matrix_transform.py')
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
