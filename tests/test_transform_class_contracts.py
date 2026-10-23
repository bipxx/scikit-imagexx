"""Test-first contracts for four upstream-only transform classes.

Reference cases establish behavior, not native coverage. Each native case stays
visibly pending until strict class lookup returns a distinct native class.
"""
import inspect
import json
from pathlib import Path
import re
import warnings
import numpy as np
import pytest
from skimage import transform as reference
import skimagexx as xx


NAMES=('ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform')
MATRIX=Path(__file__).parents[1]/'coverage/contracts/transform-classes.json'


@pytest.fixture(params=NAMES)
def class_name(request):return request.param


@pytest.fixture(params=['reference','native'])
def transform_class(request,class_name):
    if request.param=='reference':
        yield getattr(reference,class_name)
        return
    previous=xx.set_fallback_enabled(False)
    try:
        try:cls=getattr(xx.transform,class_name)
        except NotImplementedError as error:
            assert 'upstream-only' in str(error)
            pytest.xfail(f'native transform.{class_name} class contract is pending')
        assert cls is not getattr(reference,class_name),'strict lookup leaked the upstream class'
        yield cls
    finally:xx.set_fallback_enabled(previous)


def observe(action):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=action(),None
        except Exception as caught:result,error=None,caught
    return result,error,[(type(w.message),str(w.message)) for w in notices]


def normalized(message):
    # Dynamic class modules and instance addresses are not numeric behavior.
    for name in NAMES:
        message=re.sub(r"(?:skimage|skimagexx)[A-Za-z0-9_.]*\."+name,name,message)
    return re.sub(r'0x[0-9a-fA-F]+','0xADDRESS',message)


def equivalent(expected,actual):
    er,ee,ew=expected;ar,ae,aw=actual
    assert [(t,normalized(m)) for t,m in aw]==[(t,normalized(m)) for t,m in ew]
    if ee is not None:
        assert type(ae) is type(ee)
        assert normalized(str(ae))==normalized(str(ee))
        return False
    if ae is not None:raise ae
    if isinstance(er,np.ndarray):
        assert isinstance(ar,np.ndarray) and ar.dtype==er.dtype and ar.shape==er.shape
        np.testing.assert_allclose(ar,er,rtol=5e-11,atol=5e-11,equal_nan=True)
    elif hasattr(er,'params'):
        assert type(ar).__name__==type(er).__name__
        np.testing.assert_allclose(ar.params,er.params,rtol=5e-11,atol=5e-11,equal_nan=True)
    elif type(er).__name__=='FailedEstimation':
        assert type(ar).__name__=='FailedEstimation' and not ar
        assert normalized(ar.message)==normalized(er.message)
    else:assert type(ar) is type(er) and ar==er
    return True


def points():return np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,-.5],[-1.,2.],[.3,.7]])


def known_matrix(name):
    if name=='ProjectiveTransform':return np.array([[1.1,.1,2.],[-.05,.9,-1.],[.01,-.02,1.]])
    if name=='AffineTransform':return np.array([[1.1,.1,2.],[-.05,.9,-1.],[0.,0.,1.]])
    c,s=np.cos(.2),np.sin(.2);scale=1.5 if name=='SimilarityTransform' else 1.
    return np.array([[scale*c,-scale*s,2.],[scale*s,scale*c,-1.],[0.,0.,1.]])


def mapped(matrix,coords):
    homogeneous=np.column_stack((coords,np.ones(len(coords))))@matrix.T
    return homogeneous[:,:-1]/homogeneous[:,-1:]


def test_declared_signatures_and_surface(transform_class,class_name):
    records=json.loads(MATRIX.read_text(encoding='utf-8'))['classes']
    row=next(row for row in records if row['name']==class_name)
    assert str(inspect.signature(transform_class))==row['constructor_signature']
    for name,contract in row['members'].items():
        member=getattr(transform_class,name)
        if contract['kind']=='property':assert isinstance(member,property) or inspect.isdatadescriptor(member)
        else:assert str(inspect.signature(member))==contract['signature']


@pytest.mark.parametrize('dimensionality',[None,1,2,3,4])
def test_identity_dimensions(transform_class,class_name,dimensionality):
    options={} if dimensionality is None else dict(dimensionality=dimensionality)
    expected=observe(lambda:getattr(reference,class_name)(**options))
    actual=observe(lambda:transform_class(**options))
    if equivalent(expected,actual):
        obj=actual[0];d=2 if dimensionality is None else dimensionality
        assert obj.dimensionality==d
        np.testing.assert_array_equal(obj.params,np.eye(d+1))
        assert obj.params.dtype==np.float64
        same=transform_class.identity(**options)
        assert type(same) is transform_class and same is not obj
        np.testing.assert_array_equal(same.params,obj.params)


@pytest.mark.parametrize('dtype',[np.uint8,np.int64,np.float16,np.float32,np.float64])
def test_matrix_alias_and_array_protocol(transform_class,dtype):
    matrix=np.eye(3,dtype=dtype);matrix[0,2]=2
    obj=transform_class(matrix=matrix)
    assert obj.params is matrix and np.shares_memory(np.asarray(obj),matrix)
    assert obj.__array__(copy=True) is matrix  # pinned reference ignores this flag
    converted=obj.__array__(dtype=np.float64,copy=False)
    assert converted.dtype==np.float64 and not np.shares_memory(converted,matrix)
    matrix[1,2]=3
    np.testing.assert_allclose(obj([[0.,0.]]),[[2.,3.]],rtol=0,atol=0)
    obj.params[0,2]=4;assert matrix[0,2]==4
    assert set(k for k in vars(obj) if not k.startswith('_'))=={'params'}
    obj.user_note={'id':7};assert vars(obj)['user_note']=={'id':7};del obj.user_note
    replacement=np.eye(4);obj.params=replacement
    assert obj.params is replacement and obj.dimensionality==3


@pytest.mark.parametrize('dtype',[np.int16,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(2,),(3,2),(0,2),(1,1,2),(3,3),()])
def test_point_dtype_shape_and_errors(transform_class,class_name,dtype,shape):
    coords=np.arange(int(np.prod(shape)) if shape else 1,dtype=dtype).reshape(shape)
    snapshot=coords.tobytes();matrix=known_matrix(class_name)
    expected=observe(lambda:getattr(reference,class_name)(matrix)(coords))
    actual=observe(lambda:transform_class(matrix.copy())(coords))
    equivalent(expected,actual)
    assert coords.tobytes()==snapshot


def test_inverse_composition_residuals_and_live_state(transform_class,class_name):
    matrix=known_matrix(class_name);model=transform_class(matrix.copy());src=points();dst=mapped(matrix,src)
    np.testing.assert_allclose(model(src),dst,rtol=5e-13,atol=5e-13)
    inverse=model.inverse;assert type(inverse) is transform_class
    assert not np.shares_memory(inverse.params,model.params)
    np.testing.assert_allclose(inverse(dst),src,rtol=5e-12,atol=5e-12)
    np.testing.assert_allclose(model.residuals(src,dst),np.zeros(len(src)),atol=5e-13)
    second=transform_class(np.array([[1.,0.,3.],[0.,1.,-2.],[0.,0.,1.]]))
    combined=model+second;assert type(combined) is transform_class
    np.testing.assert_allclose(combined(src),second(model(src)),rtol=5e-12,atol=5e-12)
    old_inverse=inverse.params.copy();model.params[0,2]+=1
    np.testing.assert_array_equal(inverse.params,old_inverse)
    np.testing.assert_allclose(model.inverse(model(src)),src,rtol=5e-12,atol=5e-12)
    reference_model=getattr(reference,class_name)(matrix)
    assert normalized(repr(transform_class(matrix)))==normalized(repr(reference_model))
    assert transform_class(matrix).__nice__()==reference_model.__nice__()
    np.testing.assert_array_equal(src,points())


def test_cross_type_composition_and_rejection(transform_class,class_name):
    other_name='AffineTransform' if class_name!='AffineTransform' else 'EuclideanTransform'
    # Using the reference other class for a native candidate must not silently
    # produce an upstream result; future native ports should accept their own
    # corresponding class through the native namespace.
    if transform_class is getattr(reference,class_name):other=getattr(reference,other_name)()
    else:other=getattr(xx.transform,other_name)()
    combined=transform_class()+other
    assert type(combined).__name__=='ProjectiveTransform'
    np.testing.assert_array_equal(combined.params,np.eye(3))
    expected=observe(lambda:getattr(reference,class_name)()+object())
    equivalent(expected,observe(lambda:transform_class()+object()))


def test_implicit_properties(transform_class,class_name):
    if class_name=='ProjectiveTransform':
        assert not hasattr(transform_class(),'rotation')
        return
    options=dict(rotation=.2,translation=(2.,-3.))
    if class_name=='AffineTransform':options.update(scale=(2.,3.),shear=.1)
    elif class_name=='SimilarityTransform':options['scale']=1.5
    expected=getattr(reference,class_name)(**options);actual=transform_class(**options)
    np.testing.assert_allclose(actual.params,expected.params,rtol=5e-13,atol=5e-13)
    for name in ('rotation','translation','scale','shear'):
        if hasattr(expected,name):
            np.testing.assert_allclose(getattr(actual,name),getattr(expected,name),rtol=5e-13,atol=5e-13)
    np.testing.assert_allclose(actual.translation,[2.,-3.])


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_estimate_success_state_and_constructor(transform_class,class_name,dtype):
    src=points().astype(dtype);dst=mapped(known_matrix(class_name),src)
    src_snapshot=src.copy();dst_snapshot=dst.copy()
    expected=getattr(reference,class_name)();actual=transform_class();before=actual.params
    er=observe(lambda:expected.estimate(src,dst));ar=observe(lambda:actual.estimate(src,dst))
    assert equivalent(er,ar) and ar[0] is True
    assert actual.params is not before
    np.testing.assert_array_equal(before,np.eye(3))
    np.testing.assert_allclose(actual.params,expected.params,rtol=5e-11,atol=5e-11)
    # Reference Umeyama fitting retains float32 centering precision even when
    # its final parameter matrix is float64. This is an independent fit check,
    # separate from the tighter native-versus-reference parameter comparison.
    fit_tolerance=2e-6 if dtype==np.float32 else 5e-11
    np.testing.assert_allclose(actual(src),dst,rtol=fit_tolerance,atol=fit_tolerance)
    equivalent(observe(lambda:getattr(reference,class_name).from_estimate(src,dst)),
               observe(lambda:transform_class.from_estimate(src,dst)))
    np.testing.assert_array_equal(src,src_snapshot);np.testing.assert_array_equal(dst,dst_snapshot)
    assert set(k for k in vars(actual) if not k.startswith('_'))=={'params'}


@pytest.mark.parametrize('case',['constant','collinear','insufficient','empty','mismatch','nan'])
def test_estimate_degenerate_warning_error_and_failure_state(transform_class,class_name,case):
    src=points();dst=src+np.array([2.,-1.])
    if case=='constant':src=np.zeros((5,2));dst=np.ones((5,2))
    elif case=='collinear':src=np.column_stack((np.arange(5.),np.zeros(5)));dst=src+1
    elif case=='insufficient':src=src[:1];dst=dst[:1]
    elif case=='empty':src=src[:0];dst=dst[:0]
    elif case=='mismatch':dst=dst[:-1]
    else:src[0,0]=np.nan
    er_model=getattr(reference,class_name)();ar_model=transform_class()
    er=observe(lambda:er_model.estimate(src,dst));ar=observe(lambda:ar_model.estimate(src,dst))
    equivalent(er,ar)
    np.testing.assert_allclose(ar_model.params,er_model.params,rtol=5e-11,atol=5e-11,equal_nan=True)
    equivalent(observe(lambda:getattr(reference,class_name).from_estimate(src,dst)),
               observe(lambda:transform_class.from_estimate(src,dst)))


@pytest.mark.parametrize('weights',[None,[1,1,1,1,1,1,1],[1,0,1,0,1,1,1],[-1,1,1,1,1,1,1],[1,2]])
def test_weighted_estimation_contract(transform_class,class_name,weights):
    src=points();dst=mapped(known_matrix(class_name),src);dst[-1]+=[.03,-.02]
    expected=getattr(reference,class_name)();actual=transform_class()
    er=observe(lambda:expected.estimate(src,dst,weights=weights))
    ar=observe(lambda:actual.estimate(src,dst,weights=weights))
    equivalent(er,ar)
    np.testing.assert_allclose(actual.params,expected.params,rtol=5e-11,atol=5e-11,equal_nan=True)


@pytest.mark.parametrize('case',['nonsquare','vector','dimension_mismatch','implicit_and_matrix','singular_inverse'])
def test_constructor_and_inverse_validation(transform_class,class_name,case):
    def run(cls):
        if case=='nonsquare':return cls(np.zeros((2,3)))
        if case=='vector':return cls(np.zeros(3))
        if case=='dimension_mismatch':return cls(np.eye(3),dimensionality=3)
        if case=='implicit_and_matrix':return cls(np.eye(3),translation=(1,2))
        return cls(np.zeros((3,3))).inverse
    equivalent(observe(lambda:run(getattr(reference,class_name))),observe(lambda:run(transform_class)))


@pytest.mark.parametrize('name',NAMES)
def test_upstream_only_class_is_rejected_in_strict_mode(name):
    previous=xx.set_fallback_enabled(False)
    try:
        candidate=vars(xx.transform).get(name)
        if candidate is None:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                with pytest.raises(NotImplementedError,match='upstream-only'):getattr(xx.transform,name)
            assert not notices
        else:assert candidate is not getattr(reference,name)
    finally:xx.set_fallback_enabled(previous)
