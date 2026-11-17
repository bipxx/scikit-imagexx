"""Current batched phases versus the retained general public path, untimed."""
import ctypes
import importlib
import inspect
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
fixture=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_execution.py'))['fixture']
FIXED=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp.py'))['case_specs']()


class GeneralArray(np.ndarray):
    """A plain subclass deliberately uses the existing protocol-capable path."""


@pytest.fixture(scope='module')
def owned():
    import skimagexx as xx
    module=importlib.import_module('skimagexx._draw_multiblock_lbp')
    assert xx.feature.draw_multiblock_lbp is module.draw_multiblock_lbp
    assert module._core is xx._core and inspect.isbuiltin(xx._core._lbp_apply)
    assert xx._core._lbp_apply.__module__=='skimagexx._core'
    return xx,module


@pytest.fixture(scope='module')
def fenv():
    from util_views_fenv import backend
    libc,helper=backend()
    def save():
        value=(ctypes.c_ubyte*32)()
        assert libc.fegetenv(ctypes.byref(value))==0
        return value
    def restore(value):
        assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    def hardware(value):
        raw=bytes(value)
        return raw[:2],raw[4:6],raw[28:32]
    return save,restore,hardware,libc


def call_record(module,image,args,options,save,hardware):
    outputs={};events=[];prior=sys.getprofile();error=None
    entries={name:getattr(module._core,name)for name in
             ('_lbp_prepare','_lbp_binary','_lbp_assign','_lbp_apply')}
    def profile(frame,event,value):
        if event=='c_return':
            for name,entry in entries.items():
                if value is entry:events.append(name)
    try:
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always')
            try:
                sys.setprofile(profile)
                outputs['value']=module.draw_multiblock_lbp(image,*args,**options)
            except Exception as exc:
                error=(type(exc),str(exc))
            finally:
                sys.setprofile(prior)
                outputs['hardware']=hardware(save())
        outputs['warnings']=[(type(w.message),str(w.message))for w in notices]
        outputs['error']=error;outputs['events']=events
        return outputs
    finally:
        sys.setprofile(prior)


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',FIXED+[
    dict(id='wrapped-overlap-f32',shape=[9,10,3],dtype='float32',layout='F',r=-7,c=-8,width=4,height=4,code=170,alpha=.3),
    dict(id='empty-native-f64',shape=[0,10,3],dtype='float64',layout='C',r=0,c=0,width=2,height=2,code=255,alpha=.5),
],ids=lambda spec:spec['id'])
def test_batched_matches_general_words_layout_natural_FP(spec,workers,owned,fenv):
    xx,module=owned;save,restore,hardware,_=fenv
    all_args,options=fixture(spec,np);image=all_args[0];args=all_args[1:]
    before=image.tobytes();process=save();old=xx.get_num_threads()
    policy=np.geterr(),np.geterrcall(),np.getbufsize()
    try:
        xx.set_num_threads(workers)
        caller=save()
        restore(caller)
        general=call_record(module,image.view(GeneralArray),args,options,save,hardware)
        restore(caller)
        batched=call_record(module,image,args,options,save,hardware)
        assert general['error'] is batched['error'] is None
        assert general['warnings']==batched['warnings']==[]
        assert general['hardware']==batched['hardware']
        a,b=general['value'],batched['value']
        assert type(a)is type(b)is np.ndarray and a.dtype==b.dtype
        assert a.shape==b.shape and a.strides==b.strides and a.tobytes()==b.tobytes()
        assert a.flags.owndata and b.flags.owndata and not np.shares_memory(b,image)
        assert batched['events'].count('_lbp_apply')==1
        assert batched['events'].count('_lbp_binary')==batched['events'].count('_lbp_assign')==0
        assert general['events'].count('_lbp_binary')==24
        assert image.tobytes()==before and xx.get_num_threads()==workers
        assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
    finally:
        xx.set_num_threads(old);restore(process)
        assert bytes(save())==bytes(process)


@pytest.mark.parametrize('shape',((7,8),(1,9),(0,8)))
@pytest.mark.parametrize('readonly',(False,True))
@pytest.mark.parametrize('workers',(1,4))
def test_direct_owned_uint8_prepare_matches_strided_general(shape,readonly,workers,owned,fenv):
    xx,module=owned;save,restore,hardware,_=fenv
    image=np.arange(shape[0]*shape[1],dtype=np.uint8).reshape(shape)
    if image.size:image.flat[-1]=255
    backing=np.empty((shape[0],shape[1]*2),dtype=np.uint8)
    strided=backing[:,::2];strided[:]=image
    if image.size:assert not strided.flags.c_contiguous
    # NumPy marks empty arrays contiguous regardless of strides; that case
    # checks the ordinary empty allocation/descriptor, rather than dispatch.
    image.flags.writeable=not readonly
    before=image.tobytes(),strided.tobytes()
    process=save();old=xx.get_num_threads();policy=np.geterr(),np.geterrcall(),np.getbufsize()
    try:
        xx.set_num_threads(workers);caller=save()
        restore(caller);expected=module._core._lbp_prepare(strided);expected_FP=hardware(save())
        restore(caller);actual=module._core._lbp_prepare(image);actual_FP=hardware(save())
        assert expected_FP==actual_FP
        assert type(actual)is np.ndarray and actual.dtype==expected.dtype==np.dtype('float64')
        assert actual.shape==expected.shape==shape+(3,)
        assert actual.strides==expected.strides and actual.tobytes()==expected.tobytes()
        assert actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable
        assert actual.base is None and not np.shares_memory(actual,image)
        assert (image.tobytes(),strided.tobytes())==before and xx.get_num_threads()==workers
        assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
        restore(caller);assert bytes(save())==bytes(caller)
    finally:
        xx.set_num_threads(old);restore(process);assert bytes(save())==bytes(process)


@pytest.mark.parametrize('action',('warn','raise','call'))
def test_caller_owned_color_resize_keeps_general_path(action,owned,fenv):
    xx,module=owned;save,restore,hardware,_=fenv
    process=save();policy=np.geterr(),np.geterrcall(),np.getbufsize()
    results=[]
    try:
        caller=save()
        for general in (True,False):
            image=np.zeros((5,6,3),dtype=np.float64)
            color=np.array([0.,1.,0.]);events=[]
            def callback(message,flags):
                events.append((message,flags))
                color.resize((6,),refcheck=False)
                color[:]=[1.,0.,1.,1.,0.,1.]
            np.seterr(invalid=action,over=action);np.seterrcall(callback)
            restore(caller)
            row=call_record(module,image.view(GeneralArray)if general else image,
                            (0,0,2,2),dict(alpha=float('inf'),color_less_block=color),save,hardware)
            row['callbacks']=events;row['color']=color.tobytes()
            results.append(row)
        a,b=results
        assert a['error']==b['error'] and a['warnings']==b['warnings']
        assert a['hardware']==b['hardware'] and a['callbacks']==b['callbacks']
        assert a['color']==b['color']
        assert '_lbp_apply'not in a['events']and '_lbp_apply'not in b['events']
        if a['error']is None:assert a['value'].tobytes()==b['value'].tobytes()
        assert xx.get_num_threads()==1
    finally:
        np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2]);restore(process)
        assert bytes(save())==bytes(process)


@pytest.mark.parametrize('dtype',('float32','float64'))
def test_batched_live_error_callback_can_change_rounding(dtype,owned,fenv):
    xx,module=owned;save,restore,hardware,libc=fenv
    process=save();policy=np.geterr(),np.geterrcall(),np.getbufsize();results=[]
    try:
        caller=save()
        for general in (True,False):
            image=np.ones((5,6,3),dtype=dtype);image[0,0,0]=np.inf
            events=[]
            def callback(message,flags):
                events.append((message,flags))
                # This is an explicit user callback side effect in a regression,
                # not seeded target entry state or a product FP repair.
                assert libc.fesetround(0x800)==0
            np.seterr(invalid='call');np.seterrcall(callback);restore(caller)
            row=call_record(module,image.view(GeneralArray)if general else image,
                            (0,0,2,2),dict(alpha=1.),save,hardware)
            row['callbacks']=events;results.append(row)
        a,b=results
        assert a['error']is b['error']is None and a['warnings']==b['warnings']==[]
        assert a['callbacks']==b['callbacks']and a['callbacks']
        assert a['hardware']==b['hardware']and a['value'].tobytes()==b['value'].tobytes()
        assert b['events'].count('_lbp_apply')==1
        assert (int.from_bytes(b['hardware'][0],'little')&0xc00)==0x800
        assert xx.get_num_threads()==1
    finally:
        np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2]);restore(process)
        assert bytes(save())==bytes(process)


@pytest.mark.parametrize('kind',('tagged','swapped','half','numpy-alpha','numpy-coordinate','extreme-f32-alpha'))
def test_outside_batched_domain_retains_general_entry(kind,owned,fenv):
    _,module=owned
    save,restore,_,_=fenv;process=save()
    image=np.zeros((5,6,3),dtype=np.float32);options={};args=(0,0,2,2)
    if kind=='tagged':image=image.astype(np.dtype('float32',metadata={'units':'linear'}))
    elif kind=='swapped':image=image.astype('>f4')
    elif kind=='half':image=image.astype('float16')
    elif kind=='numpy-alpha':options['alpha']=np.float32(.5)
    elif kind=='extreme-f32-alpha':options['alpha']=1e100
    else:args=(np.int32(0),0,2,2)
    events=[];entry=module._core._lbp_apply;prior=sys.getprofile()
    def profile(frame,event,value):
        if event=='c_call'and value is entry:events.append(value)
    try:
        try:
            sys.setprofile(profile);result=module.draw_multiblock_lbp(image,*args,**options)
        finally:sys.setprofile(prior)
        assert not events and not np.shares_memory(image,result)
    finally:
        restore(process);assert bytes(save())==bytes(process)
