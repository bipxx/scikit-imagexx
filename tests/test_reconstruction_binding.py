"""Focused actual binding guards; inherited fixture keeps private scope explicit."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('reconstruction_pipeline_fixtures',P/'test_reconstruction_native_pipeline.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
backend=m.backend

def inputs():
    return np.zeros((4,5),'f4'),np.ones((4,5),'f4'),np.ones((3,3),bool),np.array([1,1],np.uintp)

@pytest.mark.parametrize('field',('seed','mask','footprint','offset'))
def test_scalar_conversion_cannot_invalidate_buffer(backend,field):
    core,_=backend;a,b,fp,off=inputs();target={'seed':a,'mask':b,'footprint':fp,'offset':off}[field]
    class Mutator:
        def __bool__(self):target.shape=(target.size,);return False
    before=core.reconstruction_counts(True)
    with pytest.raises(ValueError,match='metadata changed'):core.reconstruction_run(a,b,fp,off,0,Mutator())
    assert core.reconstruction_counts(False)['run']==dict(attempted=1,completed=0)

@pytest.mark.parametrize('bad',('list','dtype','strided','rank','endian'))
def test_offset_native_shape_type_admission(backend,bad):
    core,_=backend;a,b,fp,off=inputs()
    if bad=='list':off=off.tolist()
    if bad=='dtype':off=off.astype('f8')
    if bad=='strided':off=np.ones(4,np.uintp)[::2]
    if bad=='rank':off=off.reshape(1,2)
    if bad=='endian':off=off.astype(off.dtype.newbyteorder('>'))
    with pytest.raises((TypeError,ValueError)):core.reconstruction_run(a,b,fp,off,0)

@pytest.mark.parametrize('queue',(0,3,5))
def test_queue_output_owner_and_information(backend,queue):
    core,_=backend;a,b,fp,off=inputs();a[0,0]=.75
    owner,info=core.reconstruction_run(a,b,fp,off,0,None,queue)
    assert owner.flags.owndata and owner.shape==(6,7) and owner.dtype==np.float32
    assert np.all(owner[1:5,1:6]==np.float32(.75))
    assert info['parallel_flood'] is False and 1<=info['preprocessing_workers']<=core.get_num_threads()


def test_signed_zero_provider_partition_is_explicit(backend,monkeypatch):
    core,wrapper=backend;original=wrapper.np.ndarray.argsort;calls=[]
    # Spy the owned representative provider, not the native result or core.
    provider=wrapper._zero_provider
    def spy(*args):
        result=provider(*args)
        if result is not None:calls.append(result)
        return result
    monkeypatch.setattr(wrapper,'_zero_provider',spy)
    with pytest.warns(DeprecationWarning) if np.lib.NumpyVersion(np.__version__)>=np.lib.NumpyVersion('2.5.0') else __import__('contextlib').nullcontext():
        wrapper.reconstruction(np.zeros((4,5),'f4'),np.ones((4,5),'f4'))
    assert not calls
    a=np.zeros((4,5),'f4');a.flat[::2]=-0.;b=a.copy()
    with pytest.warns(DeprecationWarning) if np.lib.NumpyVersion(np.__version__)>=np.lib.NumpyVersion('2.5.0') else __import__('contextlib').nullcontext():
        out=wrapper.reconstruction(a,b)
    assert len(calls)==1 and core.reconstruction_counts(False)['run']['completed']>=1
