"""Owned binder runtime-ABI/policy checks; no original/provider image calls."""
import ctypes
import json
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
from test_finite_radon_public import owned


@pytest.fixture(params=(1,4))
def envelope(owned,request):
    from util_views_fenv import backend,FEnv
    libc,helper=backend();core=owned._core;old_workers=core.get_num_threads()
    def save():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    outer=save();policy=np.geterr(),np.geterrcall(),np.getbufsize();records=[]
    core.set_num_threads(request.param)
    yield owned,save,records
    np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2]);core.set_num_threads(old_workers);restore(outer)
    actual=save();assert bytes(actual)==bytes(outer)
    output=os.environ.get('FINITE_RADON_ABI_POLICY_RECORDS')
    if output:
        p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a')as stream:stream.write(json.dumps(dict(test=request.node.nodeid,workers=request.param,records=records,caller_saved=bytes(outer).hex(),caller_actual=bytes(actual).hex(),exact32=True))+'\n')


@pytest.mark.parametrize('policy',('ignore','warn','raise','call','log','print'))
def test_invalid_cast_runtime_policy_channels(policy,envelope,capfd):
    owned,save,records=envelope;events=[]
    class Log:
        def write(self,message):events.append(('log',message))
    handler=Log()if policy=='log'else lambda message,flag:events.append(('call',message,flag))
    np.seterrcall(handler);np.seterr(invalid=policy)
    a=np.array([[float('nan')]],dtype='float64');before=a.tobytes();hardware=bytes(save()).hex();error=None
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        try:result=owned.frt2(a)
        except FloatingPointError as exc:error=exc
    after=bytes(save()).hex();stderr=capfd.readouterr().err
    assert a.tobytes()==before and np.geterr()['invalid']==policy and np.geterrcall()is handler
    if policy=='raise':
        assert type(error)is FloatingPointError and error.args==('invalid value encountered in cast',)
    else:assert error is None and result.dtype==np.uint32 and result.shape==(2,1)
    assert [(n.category.__name__,str(n.message))for n in notices]==([('RuntimeWarning','invalid value encountered in cast')]*2 if policy=='warn'else[])
    assert events==([('call','invalid value',8)]*2 if policy=='call'else[('log','Warning: invalid value encountered in cast\n')]*2 if policy=='log'else[])
    assert stderr==('Warning: invalid value encountered in cast\n'*2 if policy=='print'else'')
    records.append(dict(policy=policy,before=hardware,after=after,error=None if error is None else list(error.args),events=events,stderr=stderr))


def test_complex_warning_precedes_each_invalid_cast(envelope):
    owned,save,records=envelope;np.seterr(invalid='warn');a=np.array([[complex(float('nan'),1)]],dtype='complex128');before=a.tobytes()
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always');result=owned.frt2(a)
    events=[(n.category.__name__,str(n.message))for n in notices]
    assert events==[('ComplexWarning','Casting complex values to real discards the imaginary part'),('RuntimeWarning','invalid value encountered in cast')]*2
    assert result.dtype==np.uint32 and a.tobytes()==before;records.append(dict(warnings=events))


def test_policy_callback_exception_and_next_phase_policy(envelope):
    owned,save,records=envelope;events=[];np.seterr(invalid='call')
    def handler(message,flag):
        events.append((message,flag));np.seterr(invalid='raise')
    np.seterrcall(handler);a=np.array([[float('nan')]],dtype='float64')
    with pytest.raises(FloatingPointError,match='invalid value encountered in cast'):owned.frt2(a)
    assert events==[('invalid value',8)]and np.geterr()['invalid']=='raise'
    sentinel=RuntimeError('ABI-handler-exception');np.seterr(invalid='call')
    def raises(message,flag):raise sentinel
    np.seterrcall(raises)
    with pytest.raises(RuntimeError)as error:owned.frt2(a)
    assert error.value is sentinel;records.append(dict(events=events,callback_exception_identity=True,next_phase_policy='raise'))


def test_current_descriptor_and_input_ownership(envelope):
    owned,save,records=envelope;meta={'units':'ABI-policy','source':'typed-metadata'};dtype=np.dtype('float32',metadata=meta)
    a=np.array([[0.,1.],[2.,3.]],dtype=dtype)[:,::-1];a.flags.writeable=False;before=a.tobytes();descriptor=a.dtype
    result=owned.frt2(a)
    assert result.dtype==np.dtype('uint32')and result.dtype.metadata is None
    assert result.flags.owndata and result.flags.c_contiguous and not np.shares_memory(result,a)
    assert a.tobytes()==before and a.dtype is descriptor and a.dtype.metadata==meta and not a.flags.writeable
    records.append(dict(input_metadata=dict(a.dtype.metadata),output_metadata=result.dtype.metadata,output_dtype=result.dtype.str))
