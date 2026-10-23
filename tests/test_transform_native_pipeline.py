"""Private class adapter tests; public registration remains independently pending.

The unchanged frozen contract functions are collected here. Temporary namespace
entries expose only our candidate classes during each test, against the selected
actual compiled backend. This adapter is never public/performance evidence.
"""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import weakref
import gc
import numpy as np
import pytest
import skimagexx as xx

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

contract=load('transform_frozen_contract',ROOT/'tests/test_transform_class_contracts.py')
native=load('skimagexx._geometric',ROOT/'python/skimagexx/_geometric.py')
if os.environ.get('SKIMAGEXX_PROJECTIVE_BACKEND'):
    native._core=importlib.import_module(os.environ['SKIMAGEXX_PROJECTIVE_BACKEND'])
INITIAL_WORKERS=native._core.get_num_threads()
class_name=contract.class_name
transform_class=contract.transform_class
for _name in vars(contract):
    if _name.startswith('test_'):globals()[_name]=getattr(contract,_name)
del _name


@pytest.fixture(params=[1,4],autouse=True)
def private_adapter(request,monkeypatch):
    if os.environ.get('SKIMAGEXX_PROJECTIVE_PORTABLE') and request.param!=1:
        pytest.skip('portable build has no OpenMP support')
    previous=native._core.get_num_threads();native._core.set_num_threads(request.param)
    for name in contract.NAMES:
        monkeypatch.setitem(vars(xx.transform),name,getattr(native,name))
    yield
    native._core.set_num_threads(previous)


@pytest.mark.parametrize('dimensions',[2,3,4])
@pytest.mark.parametrize('layout',['c','f','reverse','broadcast','readonly','unaligned'])
def test_large_strided_coordinate_kernel(dimensions,layout):
    points=np.linspace(-2,2,150001*dimensions).reshape(-1,dimensions)
    if layout=='f':points=np.asfortranarray(points)
    elif layout=='reverse':points=points[::-1,::-1]
    elif layout=='broadcast':points=np.broadcast_to(points[:1],points.shape)
    elif layout=='readonly':points.flags.writeable=False
    elif layout=='unaligned':
        values=points;points=np.ndarray(values.shape,values.dtype,buffer=bytearray(values.nbytes+1),offset=1);points[...]=values
    matrix=np.eye(dimensions+1);matrix[:-1,-1]=.3;matrix[-1,0]=.01
    expected=native._apply_numpy(matrix,points)
    actual=native._core.projective_apply(points,matrix)
    assert actual.flags.owndata and actual.flags.c_contiguous
    assert not np.shares_memory(actual,points)
    np.testing.assert_allclose(actual,expected,rtol=5e-13,atol=5e-13)


def test_no_upstream_class_calls(monkeypatch):
    def reject(*args,**kwargs):raise AssertionError('upstream transform class called')
    for name in contract.NAMES:monkeypatch.setattr(contract.reference,name,reject)
    points=contract.points();destination=points+np.array([2.,-1.])
    for name in contract.NAMES:
        cls=getattr(native,name);model=cls();model(points)
        fitted=cls.from_estimate(points,destination)
        assert fitted and isinstance(fitted,cls)


def test_backend_guards_and_initial_workers():
    assert INITIAL_WORKERS==1
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):assert not native._core.has_avx2()
    if not os.environ.get('SKIMAGEXX_PROJECTIVE_BACKEND'):assert native._core is xx._core
    for points,matrix in [(np.ones(3),np.eye(3)),(np.ones((3,3)),np.eye(3)),(np.ones((3,2)),np.ones((3,4)))]:
        with pytest.raises(ValueError):native._core.projective_apply(points,matrix)


@pytest.mark.parametrize('policy',['warn','raise','ignore'])
def test_runtime_math_provider_warnings_errors(policy):
    matrix=np.array([[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]])
    points=np.array([[0.,0.],[np.inf,1.],[np.nan,2.]])
    with np.errstate(all=policy):
        contract.equivalent(contract.observe(lambda:contract.reference.ProjectiveTransform(matrix)(points)),
                            contract.observe(lambda:native.ProjectiveTransform(matrix)(points)))


def test_finite_overflow_replays_original_warning():
    matrix=np.eye(3);matrix[-1,0]=np.finfo(float).max
    points=np.array([[2.,1.],[3.,2.]])
    result,provider=native._core.projective_apply_checked(points,matrix)
    assert provider and np.all(np.isfinite(result))
    contract.equivalent(contract.observe(lambda:contract.reference.ProjectiveTransform(matrix)(points)),
                        contract.observe(lambda:native.ProjectiveTransform(matrix)(points)))


def layout(value):
    return (type(value).__name__,value.dtype.str,value.dtype.metadata,value.shape,value.strides,
            value.flags.owndata,value.flags.writeable,value.flags.c_contiguous,value.flags.f_contiguous,
            value.base is None)


@pytest.mark.parametrize('name',contract.NAMES)
@pytest.mark.parametrize('dimension',[2,3,4])
@pytest.mark.parametrize('empty',[False,True])
def test_public_array_layout_and_lifetime(name,dimension,empty):
    source=np.asfortranarray(np.arange((0 if empty else 7)*dimension,dtype=float).reshape(-1,dimension))
    matrices=[np.asfortranarray(np.eye(dimension+1)),np.eye(dimension+1)[::-1,::-1]]
    for matrix in matrices:
        expected=getattr(contract.reference,name)(matrix)
        actual=getattr(native,name)(matrix)
        for first,second in [(expected(source),actual(source)),
                             (expected.residuals(source,source),actual.residuals(source,source))]:
            assert layout(first)==layout(second)
            np.testing.assert_allclose(first,second,rtol=5e-11,atol=5e-11)
            assert not np.shares_memory(second,source) and not np.shares_memory(second,matrix)
        for dtype in [None,np.float32,np.float64]:
            a=expected.__array__(dtype=dtype,copy=False);b=actual.__array__(dtype=dtype,copy=False)
            assert layout(a)==layout(b)
            assert np.shares_memory(a,matrix)==np.shares_memory(b,matrix)
            assert (a is matrix)==(b is matrix)
    params=np.eye(dimension+1);model=getattr(native,name)(params)
    view=model.__array__();reference=weakref.ref(params)
    del model,params;gc.collect()
    assert reference() is view and view.shape==(dimension+1,dimension+1)


@pytest.mark.parametrize('name',['AffineTransform','EuclideanTransform','SimilarityTransform'])
@pytest.mark.parametrize('readonly',[False,True])
def test_translation_property_is_live_view(name,readonly):
    params=np.eye(3);params.flags.writeable=not readonly
    expected=getattr(contract.reference,name)(params);actual=getattr(native,name)(params)
    assert layout(expected.translation)==layout(actual.translation)
    assert np.shares_memory(actual.translation,params)
    if not readonly:
        actual.translation[0]=7
        assert actual.params[0,2]==7 and actual([[0.,0.]])[0,0]==7
    view=actual.translation;reference=weakref.ref(params)
    del actual,expected,params;gc.collect()
    assert reference() is not None and view.base is reference()


def test_native_failed_estimation_metadata_and_access():
    src=np.zeros((5,2));dst=np.ones((5,2))
    for name in contract.NAMES:
        expected=getattr(contract.reference,name).from_estimate(src,dst)
        actual=getattr(native,name).from_estimate(src,dst)
        assert type(actual) is native.FailedEstimation and not actual
        assert repr(actual)==repr(expected) and str(actual)==str(expected)
        for action in [lambda obj:obj(),lambda obj:obj.params]:
            _,ee,_=contract.observe(lambda:action(expected));_,ae,_=contract.observe(lambda:action(actual))
            # Native helper exceptions intentionally have native identity;
            # preserve the named AttributeError subtype and complete message.
            assert type(ae) is native.FailedEstimationAccessError
            assert type(ae).__name__==type(ee).__name__ and isinstance(ae,AttributeError)
            assert str(ae)==str(ee)
