"""Actual public class contracts. No namespace adapter or alternate backend.

Frozen assertions are reused unchanged. Class/member/state checks are distinct
from callable image coverage; own hidden helper exception identity is a boundary.
"""
import gc
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings
import weakref

import numpy as np
import pytest
from skimage import transform as original
import skimagexx as xx
from skimagexx import _geometric as implementation
from skimagexx.coverage import native_classes, native_functions, class_contract

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_class_frozen_assertions',ROOT/'tests/test_transform_class_contracts.py')
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
class_name=contract.class_name
transform_class=contract.transform_class
for _name in vars(contract):
    if _name.startswith('test_'):globals()[_name]=getattr(contract,_name)
del _name


@pytest.fixture(params=[1,4],autouse=True)
def public_configuration(request):
    previous=xx.set_fallback_enabled(False);old_workers=xx.get_num_threads()
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(old_workers);xx.set_fallback_enabled(previous)


def descriptor(array):
    return (type(array),array.dtype.str,array.dtype.metadata,array.shape,array.strides,
            array.flags.c_contiguous,array.flags.f_contiguous,array.flags.writeable,
            array.flags.owndata,array.base is None)


def test_public_identity_registry_and_installed_backend():
    expected={f'transform.{name}' for name in contract.NAMES}
    assert set(native_classes())==expected and not expected&set(native_functions())
    assert implementation._core is xx._core
    assert Path(implementation.__file__).resolve().parent==Path(xx.__file__).resolve().parent
    assert not any(k.startswith('SKIMAGEXX_') and any(t in k for t in ('CANDIDATE','BACKEND','MODULE_DIR')) for k in os.environ)
    for name in contract.NAMES:
        candidate=getattr(xx.transform,name)
        assert candidate is getattr(implementation,name) and candidate is not getattr(original,name)
        assert name in xx.transform.__all__
        facts=class_contract('transform.'+name)
        assert facts['registered'] and facts['parity_evidence'] is facts['performance_evidence'] is None
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):assert not xx.has_avx2()


@pytest.mark.parametrize('name',contract.NAMES)
@pytest.mark.parametrize('layout',['c','f','reverse','readonly'])
def test_public_array_state_layout_lifetime_and_pickle(name,layout):
    matrix=contract.known_matrix(name)
    if layout=='f':matrix=np.asfortranarray(matrix)
    elif layout=='reverse':matrix=matrix[::-1,::-1]
    elif layout=='readonly':matrix.flags.writeable=False
    expected=getattr(original,name)(matrix);actual=getattr(xx.transform,name)(matrix)
    assert actual.params is matrix
    assert str(actual)==str(expected)
    points=contract.points()
    for array,reference in [(actual(points),expected(points)),
                            (actual.residuals(points,points),expected.residuals(points,points)),
                            (np.asarray(actual),np.asarray(expected)),
                            (actual.__array__(dtype=np.float32,copy=False),expected.__array__(dtype=np.float32,copy=False))]:
        assert descriptor(array)==descriptor(reference)
        np.testing.assert_allclose(array,reference,rtol=5e-11,atol=5e-11)
    actual.user_note={'source':'public class','ids':[1,2]}
    restored=pickle.loads(pickle.dumps(actual))
    wanted=pickle.loads(pickle.dumps(expected))
    assert type(restored) is getattr(xx.transform,name)
    assert restored.user_note==actual.user_note and restored.user_note is not actual.user_note
    np.testing.assert_array_equal(restored.params,wanted.params)
    assert descriptor(restored.params)==descriptor(wanted.params)
    np.testing.assert_allclose(restored(points),wanted(points),rtol=5e-11,atol=5e-11)
    assert not np.shares_memory(restored.params,actual.params)
    view=np.asarray(actual);reference=weakref.ref(matrix)
    del actual,expected,matrix;gc.collect()
    assert reference() is view


@pytest.mark.parametrize('name',contract.NAMES)
def test_own_failure_object_pickle_and_nominal_errors(name):
    points=np.zeros((5,2));destination=np.ones((5,2))
    expected=getattr(original,name).from_estimate(points,destination)
    actual=getattr(xx.transform,name).from_estimate(points,destination)
    assert type(actual) is implementation.FailedEstimation and not actual
    assert str(actual)==str(expected) and repr(actual)==repr(expected)
    # The pinned helper is not pickle-round-trip safe: unpickling probes an
    # attribute before message exists, recursively entering __getattr__. Keep
    # the observed failure separate from valid model pickle support.
    wanted_pickle=contract.observe(lambda:pickle.loads(pickle.dumps(expected)))
    actual_pickle=contract.observe(lambda:pickle.loads(pickle.dumps(actual)))
    assert isinstance(wanted_pickle[1],RecursionError)
    assert contract.equivalent(wanted_pickle,actual_pickle) is False
    for operation in [lambda obj:obj(),lambda obj:obj.params]:
        _,wanted,_=contract.observe(lambda:operation(expected))
        _,got,_=contract.observe(lambda:operation(actual))
        assert type(got) is implementation.FailedEstimationAccessError
        assert type(got).__name__==type(wanted).__name__ and isinstance(got,AttributeError)
        assert str(got)==str(wanted)
        assert type(got) is type(wanted), 'native and compatible results use the canonical public catch type'


def test_public_operations_never_call_original_classes(monkeypatch):
    def reject(*args,**kwargs):raise AssertionError('upstream transform class called')
    for name in contract.NAMES:monkeypatch.setattr(original,name,reject)
    points=contract.points();target=points+np.array([2.,-1.])
    for name in contract.NAMES:
        cls=getattr(xx.transform,name);obj=cls()
        obj(points);obj.inverse(points);obj.residuals(points,points)
        assert isinstance(obj+cls(),cls)
        assert isinstance(cls.from_estimate(points,target),cls)


def test_numpy_only_2d_path_without_skimage_or_scipy_imports():
    code=r'''
import importlib.abc,sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('skimage','scipy') or fullname.startswith(('skimage.','scipy.')):
            raise ModuleNotFoundError('blocked optional dependency: '+fullname,name=fullname)
sys.meta_path.insert(0,BlockOptional())
import numpy as np
import skimagexx as xx
xx.set_fallback_enabled(False)
points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,-.5]])
for name in ('ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform'):
    cls=getattr(xx.transform,name);obj=cls()
    np.testing.assert_allclose(obj(points),points)
    np.testing.assert_allclose(obj.inverse(points),points)
    fit=cls.from_estimate(points,points+[2.,-1.]);assert fit
    np.testing.assert_allclose(fit(points),points+[2.,-1.],rtol=5e-11,atol=5e-11)
assert not any(n=='skimage' or n.startswith('skimage.') or n=='scipy' or n.startswith('scipy.') for n in sys.modules)
failed=xx.transform.ProjectiveTransform.from_estimate(np.zeros((5,2)),np.ones((5,2)))
try:failed.params
except xx.util.FailedEstimationAccessError as error:
    from skimagexx import _geometric as g
    assert type(error) is g._StandaloneFailedEstimationAccessError
    assert g.FailedEstimation.error_cls is xx.util.FailedEstimationAccessError
else:raise AssertionError('standalone public exception alias did not catch')
assert not any(n=='skimage' or n.startswith('skimage.') or n=='scipy' or n.startswith('scipy.') for n in sys.modules)
try:xx.transform.EuclideanTransform(rotation=(.1,.2,.3))
except ModuleNotFoundError as error:assert error.name=='scipy'
else:raise AssertionError('3D Euler provider dependency was not exercised')
print('NUMPY_ONLY_2D_PUBLIC_CLASSES_PASS; SCIPY_3D_PROVIDER_EXPLICIT')
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path(xx.__file__).resolve().parent.parent)
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'NUMPY_ONLY_2D_PUBLIC_CLASSES_PASS' in result.stdout
