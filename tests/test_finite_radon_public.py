"""Own private/public integer invariants only; zero original reference credit."""
import importlib,inspect,os,sys,types
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def owned():
    directory=os.environ.get('FINITE_RADON_COMPONENT_DIR')
    if directory:
        sys.path.insert(0,directory);core=importlib.import_module('_finite_radon_core')
        p=types.ModuleType('finite_radon_component');p.__path__=[str(ROOT/'python/skimagexx')];sys.modules[p.__name__]=p;sys.modules[p.__name__+'._core']=core
        return importlib.import_module(p.__name__+'._finite_radon')
    return importlib.import_module('skimagexx._finite_radon')

@pytest.mark.parametrize('dtype',('bool','uint8','uint16','uint32','uint64','int8','int16','int32','int64'))
@pytest.mark.parametrize('workers',(1,4))
def test_own_frt_integer_coordinate_and_owner(dtype,workers,owned):
    core=owned._core;assert inspect.isbuiltin(core.frt2);core.set_num_threads(workers)
    a=np.array([[r*3+c for c in range(3)]for r in range(3)],dtype=dtype)[:,::-1];a.flags.writeable=False;before=a.tobytes()
    try:
        b=owned.frt2(a);assert b.dtype==np.uint32 and b.shape==(4,3)and b.flags.owndata and b.flags.c_contiguous and not np.shares_memory(a,b)
        for m in range(3):
            for c in range(3):assert int(b[m,c])==sum(int(a[r,(c+m*r)%3])for r in range(3))%(2**32)
        assert a.tobytes()==before
    finally:core.set_num_threads(1)

@pytest.mark.parametrize('dtype',('bool','uint8','uint16','uint32','uint64'))
@pytest.mark.parametrize('workers',(1,4))
def test_own_ifrt_unsigned_scalar_plan(dtype,workers,owned):
    core=owned._core;assert inspect.isbuiltin(core.ifrt2);core.set_num_threads(workers)
    a=np.array([[10,0],[0,0],[0,0]],dtype=dtype)
    try:
        out=owned.ifrt2(a);total=int(a[0,0])+int(a[0,1]);weak=np.__version__.split('.')[0]=='1'
        signed=dtype=='bool'and not weak
        for m in range(2):
            for c in range(2):
                value=(sum(int(a[r,(c-m*r)%2])for r in range(2))+int(a[2,m]))%(2**32)
                diff=value-total if signed else(value-total)%(2**(32 if weak else 64))
                assert out[m,c]==float(diff)/2
        assert out.dtype==np.float64 and out.flags.owndata and not np.shares_memory(a,out)
    finally:core.set_num_threads(1)

@pytest.mark.parametrize('dtype',('float16','float32','float64','complex64','complex128','object'))
def test_assessed_noninteger_public_domains(dtype,owned):
    result=owned.frt2(np.zeros((2,2),dtype=dtype));assert result.dtype==np.uint32 and result.shape==(3,2)
    with pytest.raises(Exception)as error:owned.ifrt2(np.zeros((3,2),dtype=dtype))
    assert type(error.value).__qualname__=='_UFuncOutputCastingError'

def test_shape_signature_and_signed_inverse_descriptor(owned):
    assert str(inspect.signature(owned.frt2))=='(a)'and str(inspect.signature(owned.ifrt2))=='(a)'
    with pytest.raises(ValueError,match='square'):owned.frt2(np.zeros((2,3),dtype='uint32'))
    with pytest.raises(ValueError,match='row x n column'):owned.ifrt2(np.zeros((2,2),dtype='uint32'))
    with pytest.raises(IndexError,match='size 0'):owned.ifrt2(np.empty((1,0),dtype='uint32'))
    with pytest.raises(Exception)as caught:owned.ifrt2(np.zeros((3,2),dtype='int32'))
    e=caught.value;assert type(e).__name__=='UFuncTypeError'or type(e).__name__=='_UFuncOutputCastingError'
    assert e.args[0]is np.add and e.args[1]=='same_kind'and e.args[3]==np.dtype('uint32')
