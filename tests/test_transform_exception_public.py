"""Public exception identity/catching and inherited Python exception behavior."""
import inspect
import os
import pickle
from pathlib import Path
import subprocess
import sys
import warnings
import numpy as np
import pytest
import skimage.util as original
from skimage._shared import utils as original_private
import skimagexx as xx
from skimagexx import _geometric as implementation


@pytest.fixture(params=[1,4],autouse=True)
def workers(request):
    previous=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(previous)


@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize('name',['ProjectiveTransform','AffineTransform','EuclideanTransform','SimilarityTransform'])
def test_failed_model_is_caught_by_public_exception_alias(fallback,name):
    assert original.FailedEstimationAccessError is original_private.FailedEstimationAccessError
    old=xx.set_fallback_enabled(fallback)
    try:
        failed=getattr(xx.transform,name).from_estimate(np.zeros((5,2)),np.ones((5,2)))
        caught=False
        try:failed.params
        except xx.util.FailedEstimationAccessError as error:
            assert type(error) is implementation.FailedEstimationAccessError
            caught=True
        assert caught
        assert xx.util.FailedEstimationAccessError is implementation.FailedEstimationAccessError
    finally:xx.set_fallback_enabled(old)


@pytest.mark.parametrize('args',[(),('message',),('a','b'),(42,),({'payload':1},)])
@pytest.mark.parametrize('keywords',[{},dict(name='params',obj='owner')])
def test_public_exception_constructor_state_and_pickle(args,keywords):
    native=xx.util.FailedEstimationAccessError;upstream=original.FailedEstimationAccessError
    actual=native(*args,**keywords);expected=upstream(*args,**keywords)
    assert native is implementation.FailedEstimationAccessError is upstream
    assert native.__bases__==upstream.__bases__==(AttributeError,)
    for error in [actual,pickle.loads(pickle.dumps(actual))]:
        assert type(error) is native and error.args==expected.args
        assert str(error)==str(expected) and repr(error)==repr(expected)
    assert actual.name==expected.name and actual.obj==expected.obj
    # AttributeError's pickle protocol intentionally does not retain name/obj.
    a=pickle.loads(pickle.dumps(actual));e=pickle.loads(pickle.dumps(expected))
    assert (a.name,a.obj)==(e.name,e.obj)
    actual.custom='state';assert pickle.loads(pickle.dumps(actual)).custom=='state'


def test_exception_signature_and_invalid_keyword_match_original():
    messages=[]
    for cls in [xx.util.FailedEstimationAccessError,original.FailedEstimationAccessError]:
        with pytest.raises(ValueError,match='no signature found'):inspect.signature(cls)
        with pytest.raises(TypeError) as result:cls(unknown=True)
        messages.append(str(result.value))
    assert messages[0]==messages[1]


@pytest.mark.parametrize('name',['EssentialMatrixTransform','FundamentalMatrixTransform','ThinPlateSplineTransform'])
def test_upstream_backed_failed_models_use_same_public_catch_type(name):
    old=xx.set_fallback_enabled(True)
    try:
        with pytest.warns(xx.FallbackWarning):cls=getattr(xx.transform,name)
        failed=cls.from_estimate(np.zeros((8,2)),np.ones((8,2)))
        caught=False
        try:failed.params
        except xx.util.FailedEstimationAccessError as error:
            assert type(error) is original.FailedEstimationAccessError
            caught=True
        assert caught
    finally:xx.set_fallback_enabled(old)


def test_own_only_alias_cannot_catch_compatibility_exception(monkeypatch):
    """Scoped regression demonstration, never an acceptance backend adapter."""
    old=xx.set_fallback_enabled(True)
    try:
        with pytest.warns(xx.FallbackWarning):cls=xx.transform.EssentialMatrixTransform
        failed=cls.from_estimate(np.zeros((8,2)),np.ones((8,2)))
        monkeypatch.setattr(xx.util,'FailedEstimationAccessError',implementation._StandaloneFailedEstimationAccessError)
        with pytest.raises(original.FailedEstimationAccessError):
            try:failed.params
            except xx.util.FailedEstimationAccessError:raise AssertionError('unrelated own alias caught upstream exception')
    finally:xx.set_fallback_enabled(old)


def test_resolver_caches_one_identity_and_exposes_no_kernel_claim():
    from skimagexx.coverage import native_classes,class_metadata_providers,class_contract
    assert len(native_classes())==4
    assert class_metadata_providers()==('util.FailedEstimationAccessError',)
    facts=class_contract(class_metadata_providers()[0])
    assert facts['implementation']=='exception-metadata-provider'
    assert facts['parity_evidence'] is facts['performance_evidence'] is None
    assert implementation._canonical_failure_error() is xx.util.FailedEstimationAccessError
    assert implementation.FailedEstimation.error_cls is xx.util.FailedEstimationAccessError
    assert 'FailedEstimationAccessError' in dir(xx.util)


def test_unrelated_optional_import_failure_is_not_hidden():
    code=r'''
import importlib.abc,sys
class BrokenImport(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='skimage':raise ModuleNotFoundError('unrelated dependency broken',name='unrelated_dependency')
sys.meta_path.insert(0,BrokenImport())
from skimagexx import _geometric as g
try:g._canonical_failure_error()
except ModuleNotFoundError as error:assert error.name=='unrelated_dependency'
else:raise AssertionError('unexpected import failure hidden')
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path(xx.__file__).resolve().parent.parent)
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
