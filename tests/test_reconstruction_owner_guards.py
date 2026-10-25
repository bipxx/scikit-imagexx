"""Benign owner metadata/properties only: never free or relocate an array."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('reconstruction_owner_pipeline',P/'test_reconstruction_native_pipeline.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
backend=M.backend

def data():return np.zeros((4,5),'f4'),np.ones((4,5),'f4'),np.ones((3,3),bool),np.array([1,1],np.uintp)

@pytest.mark.parametrize('role',('seed','mask','footprint','offset'))
def test_base_property_is_never_dispatched(backend,role):
    core,_=backend;args=list(data());index={'seed':0,'mask':1,'footprint':2,'offset':3}[role];calls=[]
    class BaseTrap(np.ndarray):
        @property
        def base(self):calls.append(True);raise AssertionError('subclass base property was called')
    args[index]=args[index].view(BaseTrap)
    owner,info=core.reconstruction_run(*args,0)
    assert not calls and np.all(owner[1:5,1:6]==0)
    assert info['parallel_flood'] is False

@pytest.mark.parametrize('entry',('reconstruction_ordered','reconstruction_negative_zero'))
@pytest.mark.parametrize('role',('seed','mask'))
def test_read_only_intake_uses_direct_base_slot(backend,entry,role):
    core,_=backend;a,b,_,_=data();calls=[]
    class Trap(np.ndarray):
        @property
        def base(self):calls.append(True);raise AssertionError('subclass base getter dispatched')
    if role=='seed':a=a.view(Trap)
    else:b=b.view(Trap)
    result=getattr(core,entry)(a,b,0) if entry=='reconstruction_ordered' else getattr(core,entry)(a,b)
    assert result is (entry=='reconstruction_ordered') and not calls

@pytest.mark.parametrize('role',('seed','mask','footprint','offset'))
@pytest.mark.parametrize('change',('shape','dtype','writeable'))
def test_root_owner_change_is_rejected_with_unchanged_view(backend,role,change):
    core,_=backend;a,b,fp,offset=data();args=[a,b,fp,offset];index={'seed':0,'mask':1,'footprint':2,'offset':3}[role]
    original=args[index];root=np.array(original.reshape(-1),copy=True);view=root.reshape(original.shape);args[index]=view
    before=(view.ctypes.data,view.shape,view.strides,view.dtype,view.nbytes);root_pointer=root.ctypes.data
    class MutateRoot:
        def __bool__(self):
            if change=='shape':root.shape=(1,root.size)
            elif change=='dtype':root.dtype=np.dtype('u1')
            else:root.flags.writeable=False
            assert root.ctypes.data==root_pointer
            return False
    core.reconstruction_counts(True)
    with pytest.raises(ValueError,match='metadata changed'):core.reconstruction_run(*args,0,MutateRoot())
    assert (view.ctypes.data,view.shape,view.strides,view.dtype,view.nbytes)==before
    assert core.reconstruction_counts(False)['run']==dict(attempted=1,completed=0)

@pytest.mark.parametrize('entry',('reconstruction_run','reconstruction_ordered','reconstruction_negative_zero'))
def test_nonndarray_storage_owner_is_explicitly_rejected(backend,entry):
    core,_=backend;a,b,fp,offset=data();external=np.frombuffer(bytearray(80),dtype='f4').reshape(4,5)
    with pytest.raises(ValueError,match='ndarray-backed storage'):
        if entry=='reconstruction_run':core.reconstruction_run(external,b,fp,offset,0)
        elif entry=='reconstruction_ordered':core.reconstruction_ordered(external,b,0)
        else:core.reconstruction_negative_zero(external,b)

@pytest.mark.parametrize('order',('C','F'))
def test_valid_nested_owning_array_views_remain_accepted(backend,order):
    core,_=backend;a,b,fp,offset=data();a=np.array(a,order=order).view();b=np.array(b,order=order).view()
    owner,info=core.reconstruction_run(a,b,fp,offset,0)
    assert np.all(owner==0) and owner.shape==(6,7) and owner.dtype==np.float32
