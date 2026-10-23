"""Integer _scale decides before warning, then casts its same ndarray after hooks."""
import importlib
import importlib.util
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage.filters import rank as reference
import skimagexx as xx

if os.environ.get('SKIMAGEXX_RANK_CANDIDATE'):
    spec=importlib.util.spec_from_file_location('skimagexx._rank_integer_candidate',os.environ['SKIMAGEXX_RANK_CANDIDATE'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
else:native=importlib.import_module('skimagexx._rank')

NAMES=('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy')
VALUE_CASES=[('image',np.int16,2),('image',np.int32,3),('image',np.int64,2),('image',np.uint32,3),('image',np.uint64,2),('mask',np.uint16,3)]


class WithoutAstype(np.ndarray):
    def astype(self,*args,**kwargs):
        raise AssertionError('img_as_ubyte must strip the input subclass before cast lookup')


def record(value):
    return dict(type=type(value).__name__,dtype=value.dtype.str,metadata=value.dtype.metadata,
        shape=value.shape,strides=value.strides,writeable=value.flags.writeable,owned=value.flags.owndata,
        c=value.flags.c_contiguous,f=value.flags.f_contiguous,bytes=value.tobytes().hex())


def observe(provider,name,stage,dtype,ndim,hook,action,subclass=False):
    shape=(5,7) if ndim==2 else (3,4,5)
    image=np.arange(int(np.prod(shape)),dtype=np.uint8).reshape(shape)%35
    mask=np.ones(shape,np.uint8)
    target=(np.arange(image.size)%35).astype(dtype).reshape(shape)
    if subclass:target=target.view(WithoutAstype)
    if stage=='image':image=target
    else:mask=target
    fp=np.ones((3,)*ndim,np.uint8)
    # A changed image shape must allocate its own result; never pass an output
    # smaller than the post-hook image into bounds-disabled original Cython.
    out=None if action in ('shape','dtype','nan-call','nan-raise','nan-handler-error') else np.full(shape,99,np.uint8)
    events=[];changed=False
    previous_policy=np.geterr();previous_handler=np.geterrcall()
    previous_format=warnings.formatwarning;previous_impl=warnings._showwarnmsg_impl
    previous_dtype=np.dtype

    def floating(message,flags):
        events.append(('floating',message,flags))
        if action=='nan-handler-error':raise ArithmeticError('integer cast callback sentinel')

    def consume(text):
        nonlocal changed
        events.append(('warning',str(text)))
        if changed or 'Downcasting' not in str(text):return
        changed=True
        if action=='zero':target.fill(0)
        elif action=='high':target.fill(511)
        elif action=='negative-or-wrap':target.fill(-3 if target.dtype.kind=='i' else 65535)
        elif action=='shape':target.shape=(1,)+target.shape
        elif action=='dtype':
            target.dtype={2:np.float16,4:np.float32,8:np.float64}[target.itemsize]
            target.fill(.5)
        elif action.startswith('nan-'):
            target.dtype=np.float64;target.fill(np.nan)
            np.seterrcall(floating);np.seterr(invalid='raise' if action=='nan-raise' else 'call')
        elif action=='hook-error':raise LookupError('integer downcast warning sentinel')
        elif action=='method':
            def dtype_factory(value,*args,**kwargs):
                # Intercept the next lookup only: this probes the cast target
                # after warning without replacing NumPy's later global API.
                np.dtype=previous_dtype
                events.append(('dtype-factory',repr(value)))
                return previous_dtype(value,*args,**kwargs)
            np.dtype=dtype_factory

    class Stream:
        def write(self,text):consume(text)
        def flush(self):pass

    try:
        np.seterr(all='warn')
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            if action=='warning-error':warnings.filterwarnings('error',message='Downcasting',category=UserWarning)
            warnings.formatwarning=lambda message,category,filename,lineno,line=None:f'{category.__name__}: {message}\n'
            if hook=='showwarning':
                warnings.showwarning=lambda message,category,filename,lineno,file=None,line=None:consume(str(message))
            elif hook=='implementation':warnings._showwarnmsg_impl=lambda message:consume(str(message.message))
            else:
                stream=Stream()
                warnings.showwarning=lambda message,category,filename,lineno,file=None,line=None:warnings._showwarning_orig(message,category,filename,lineno,file=stream,line=line)
            try:
                options=dict(mask=mask)
                if out is not None:options['out']=out
                value=getattr(provider,name)(image,fp,**options)
                outcome=('return',record(value),None if out is None else (value is out,bool(np.shares_memory(value,out))))
            except Exception as error:outcome=('error',type(error).__name__,str(error))
        return outcome,events,record(image),record(mask),None if out is None else record(out),tuple(np.geterr().items())
    finally:
        np.dtype=previous_dtype
        warnings.formatwarning=previous_format;warnings._showwarnmsg_impl=previous_impl
        np.seterr(**previous_policy);np.seterrcall(previous_handler)


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads()
    assert native._core is xx._core
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('stage,dtype,ndim',VALUE_CASES)
@pytest.mark.parametrize('hook',['showwarning','implementation','stream'])
@pytest.mark.parametrize('action',['zero','high','negative-or-wrap'])
def test_decision_before_warning_value_cast_after_warning(name,stage,dtype,ndim,hook,action):
    assert observe(native,name,stage,dtype,ndim,hook,action)==observe(reference,name,stage,dtype,ndim,hook,action)


META_CASES=[(np.int16,'shape',False),(np.int16,'shape',True),
    (np.uint32,'dtype',False),(np.uint32,'dtype',True),
    (np.int64,'nan-call',False),(np.uint64,'nan-raise',False),
    (np.int64,'nan-handler-error',False),(np.uint64,'method',False)]


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('stage',['image','mask'])
@pytest.mark.parametrize('hook',['showwarning','implementation','stream'])
@pytest.mark.parametrize('dtype,action,subclass',META_CASES)
def test_shape_dtype_method_capture_and_reentrant_fp_policy(name,stage,hook,dtype,action,subclass):
    ndim=3 if action in ('dtype','nan-raise','method') else 2
    assert observe(native,name,stage,dtype,ndim,hook,action,subclass)==observe(reference,name,stage,dtype,ndim,hook,action,subclass)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('stage',['image','mask'])
@pytest.mark.parametrize('hook',['showwarning','implementation','stream'])
@pytest.mark.parametrize('action',['hook-error','warning-error'])
def test_warning_error_precedes_cast_and_output_writes(name,stage,hook,action):
    assert observe(native,name,stage,np.int16,2,hook,action)==observe(reference,name,stage,np.int16,2,hook,action)


@pytest.mark.parametrize('name',NAMES)
def test_integer_warning_provider_does_not_forward_rank_algorithms(name,monkeypatch):
    expected=observe(reference,name,'image',np.int16,2,'implementation','high')
    def forbidden(*args,**kwargs):raise AssertionError('upstream rank forwarding')
    for operation in NAMES:monkeypatch.setattr(reference,operation,forbidden)
    assert observe(native,name,'image',np.int16,2,'implementation','high')==expected
