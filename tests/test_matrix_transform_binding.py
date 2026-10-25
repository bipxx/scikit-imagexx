"""Private real-bridge metadata/ownership guards, not public acceptance."""
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import numpy as np
import pytest

path=os.environ.get('MATRIX_TRANSFORM_PRIVATE_BACKEND')
if path:
    backend=sys.modules.get('_matrix_math_core')
    if backend is None:
        spec=importlib.util.spec_from_file_location('_matrix_math_core',path)
        backend=importlib.util.module_from_spec(spec);sys.modules['_matrix_math_core']=backend;spec.loader.exec_module(backend)
else:
    import skimagexx
    backend=skimagexx._core


def test_real_builtin_interface_single_worker_owner():
    assert inspect.isbuiltin(backend.matrix_transform_run)
    assert backend.matrix_transform_run.__module__==backend.__name__
    assert backend.matrix_transform_interface=='matrix-transform-math-v1'
    assert backend.matrix_transform_capabilities==31
    assert backend.get_num_threads()==1
    assert inspect.isbuiltin(backend.set_num_threads)


@pytest.mark.parametrize('dtype',('f4','i8','>f8','c16'))
def test_dtype_and_endian_reject_without_conversion(dtype):
    p=np.array([[1.,2.],[3.,4.]],dtype=dtype);m=np.eye(3);before=p.tobytes()
    with pytest.raises(TypeError,match='native float64'):
        backend.matrix_transform_run(p,m,1)
    assert p.tobytes()==before


def test_no_array_protocol_conversion():
    events=[]
    class Array:
        def __array__(self,*args,**kwargs):events.append('array');return np.ones((2,2))
    with pytest.raises(TypeError):backend.matrix_transform_run(Array(),np.eye(3),1)
    assert not events


def test_readonly_input_and_owned_writable_output():
    p=np.array([[1.123,2000.234],[3000.345,4.456]]);m=np.eye(3)
    p.flags.writeable=False;m.flags.writeable=False
    out,status=backend.matrix_transform_run(p,m,1)
    assert status==0 and out.tobytes()==p.tobytes()
    assert out.flags.owndata and out.flags.c_contiguous and out.flags.writeable
    assert not np.shares_memory(out,p) and not np.shares_memory(out,m)


def test_metadata_dtype_keeps_real_float64_storage():
    p=np.array([[1.123,2.234],[3.345,4.456]],dtype=np.dtype('f8',metadata={'unit':'x'}))
    out,status=backend.matrix_transform_run(p,np.eye(3),1)
    assert status==0 and out.tobytes()==p.tobytes() and out.dtype.metadata is None


def test_memoryview_of_numpy_owner_preserves_native_admission():
    p=np.array([[1.123,2.234],[3.345,4.456]])
    m=np.asarray(memoryview(np.eye(3)))
    assert isinstance(m.base,memoryview)
    out,status=backend.matrix_transform_run(p,m,1)
    assert status==0 and out.tobytes()==p.tobytes()


def test_untrusted_as_strided_owner_rejects_before_read():
    owner=np.array([1.])
    p=np.lib.stride_tricks.as_strided(owner,shape=(2,2),strides=(8000,8))
    out,status=backend.matrix_transform_run(p,np.eye(3),1)
    assert out is None and status==3 and owner.tolist()==[1.]


@pytest.mark.parametrize('profile,status',((0,1),(1000,1),(3,2),(4,2)))
def test_prewrite_status_keeps_inputs_unchanged(profile,status):
    p=np.array([[1.123,2.234],[3.345,4.456]]);m=np.eye(3);before=(p.tobytes(),m.tobytes())
    out,actual=backend.matrix_transform_run(p,m,profile)
    assert out is None and actual==status
    assert before==(p.tobytes(),m.tobytes())


def test_finite_range_rejects_whole_output():
    p=np.array([[1.123,2.234],[3.345,np.inf]]);before=p.tobytes()
    out,status=backend.matrix_transform_run(p,np.eye(3),1)
    assert out is None and status==6 and p.tobytes()==before


def test_numpy_valueerror_dimension_is_before_kernel():
    with pytest.raises(ValueError,match='two-dimensional'):
        backend.matrix_transform_run(np.array([1.,2.]),np.eye(3),1)


def test_portable_worker_policy_is_explicit():
    assert backend.get_num_threads()==1
    if not backend.has_openmp:
        with pytest.raises(ValueError,match='OpenMP'):backend.set_num_threads(4)
        assert backend.get_num_threads()==1
    else:
        backend.set_num_threads(4);assert backend.get_num_threads()==4
        backend.set_num_threads(1)



def test_strict_v2_is_owned_builtin_and_legacy_rejects_new_profile():
    assert backend.matrix_transform_interface=='matrix-transform-math-v1'
    assert backend.matrix_transform_capabilities==31
    assert backend.matrix_transform_v2_interface=='matrix-transform-math-v2'
    assert backend.matrix_transform_v2_capabilities==63
    assert inspect.isbuiltin(backend.matrix_transform_run_v2)
    assert backend.matrix_transform_run_v2.__module__==backend.__name__
    p=np.array([[1.125,2.25]]);m=np.eye(3)
    before=(backend.matrix_transform_completed(),backend.matrix_transform_v2_completed())
    out,status=backend.matrix_transform_run(p,m,6)
    assert out is None and status==1
    assert before==(backend.matrix_transform_completed(),backend.matrix_transform_v2_completed())
    out,status=backend.matrix_transform_run_v2(p,m,6)
    assert status==0 and out.tobytes()==p.tobytes()
    assert out.flags.owndata and out.flags.c_contiguous
    assert (backend.matrix_transform_completed(),backend.matrix_transform_v2_completed())==(before[0]+1,before[1]+1)


@pytest.mark.parametrize('profile,status',((0,1),(1000,1),(6,2)))
def test_v2_prewrite_rejection_keeps_counter_and_readonly_inputs(profile,status):
    p=np.array([[1.125,2.25],[3.5,4.75]]);m=np.eye(3)
    p.flags.writeable=False;m.flags.writeable=False
    before=(p.tobytes(),m.tobytes(),backend.matrix_transform_completed(),backend.matrix_transform_v2_completed())
    out,actual=backend.matrix_transform_run_v2(p,m,profile)
    assert out is None and actual==status
    assert before==(p.tobytes(),m.tobytes(),backend.matrix_transform_completed(),backend.matrix_transform_v2_completed())


@pytest.mark.parametrize('dtype',('f4','i8','>f8','c16'))
def test_v2_dtype_rejects_without_conversion_or_completion(dtype):
    p=np.array([[1.,2.]],dtype=dtype);before=(p.tobytes(),backend.matrix_transform_completed())
    with pytest.raises(TypeError,match='native float64'):
        backend.matrix_transform_run_v2(p,np.eye(3),6)
    assert before==(p.tobytes(),backend.matrix_transform_completed())


def test_v2_does_not_run_array_protocol():
    calls=[]
    class Coordinates:
        def __array__(self,*args,**kwargs):calls.append('array');return np.ones((1,2))
    with pytest.raises(TypeError):backend.matrix_transform_run_v2(Coordinates(),np.eye(3),6)
    assert not calls
