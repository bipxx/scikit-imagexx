"""Pinned original-only percentile contracts; no native acceptance.

Only bounded, disjoint, source-qualified buffers and finite ordered percentile
domains reach unchecked Cython kernels. Provider failures are separate cases.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
import scipy
import skimage
from skimage.filters import rank as reference
from skimage.filters.rank import _percentile as percentile_module
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/rank-percentile.json'
NAMES=('autolevel_percentile','enhance_contrast_percentile','gradient_percentile','mean_percentile','pop_percentile','subtract_mean_percentile','threshold_percentile','sum_percentile','percentile')
SINGLE=('percentile','threshold_percentile')
spec=importlib.util.spec_from_file_location('rank_percentile_qualified_base',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


def cases():
    rows=[]
    for name in NAMES:
        for key,old,parameters in base.CASES:
            if old!='minimum' or parameters.get('ndim')!=2:continue
            if parameters.get('out') in ('image','view','overlap') or parameters.get('special')=='alias-converted':continue
            rows.append((name+'/'+key.split('/',1)[1],name,dict(parameters)))
        for token in ('default','zero','one','quarter','half','adjacent-low','adjacent-high','f32','np-scalar','array0','array1','bool','none','numeric-string','bad-string','list','tuple','object','complex'):
            rows.append((f'{name}/parameter={token}',name,dict(ndim=2,parameter=token)))
        if name not in SINGLE:
            for token in ('p1-bad-string','p1-object','p1-array1'):
                rows.append((f'{name}/parameter={token}',name,dict(ndim=2,parameter=token)))
        else:rows.append((f'{name}/unexpected-p1',name,dict(ndim=2,parameter='unexpected-p1')))
        for pattern in ('ties','ramp','single-bin','all-zero','gaps','center-low','center-high','zero-footprint','zero-mask','disconnected','even','shift','fractional-mean','endpoints'):
            for dtype in ('u1','u2'):
                rows.append((f'{name}/{dtype}/pattern={pattern}',name,dict(ndim=2,pattern=pattern,pattern_dtype=dtype)))
        for shape in ((2,3,4),(1,1,1),(),(3,),(0,3),(3,0),(2,2,2,2)):
            rows.append((f'{name}/dimensions={shape}',name,dict(ndim=2,shape=shape)))
        rows.append((name+'/missing-footprint',name,dict(omitted=True)))
        for output in ('tagged-u1','tagged-u2','tagged-f4','tagged-f8','owned-view'):
            rows.append((f'{name}/output-owner={output}',name,dict(ndim=2,output_owner=output)))
    assert len(rows)==len({key for key,_,_ in rows})
    return rows


CASES=cases()


def parameter_values(token):
    table={'default':(0.,1.),'zero':(0.,0.),'one':(1.,1.),'quarter':(.25,.75),'half':(.5,.5),'adjacent-low':(float(np.nextafter(0.,1.)),.75),'adjacent-high':(.25,float(np.nextafter(1.,0.))),'f32':(np.float32(.2),np.float32(.8)),'np-scalar':(np.float64(.25),np.float64(.75)),'array0':(np.array(.25),np.array(.75)),'array1':(np.array([.25]),np.array([.75])),'bool':(False,True),'numeric-string':('0.25','0.75'),'bad-string':('bad',.75),'list':([.25],.75),'tuple':((.25,),.75),'object':(SimpleNamespace(),.75),'complex':(np.complex64(.25+.1j),np.complex128(.75+.1j)),'p1-bad-string':(.25,'bad'),'p1-object':(.25,SimpleNamespace()),'p1-array1':(.25,np.array([.75])),'unexpected-p1':(.25,.75)}
    return table[token]


def make_case(name,parameters):
    if parameters.get('omitted'):
        image=np.arange(6,dtype=np.uint8).reshape(2,3);return (image,),{},dict(image=image)
    args,options,tracked=base.make_case(parameters)
    if 'parameter' in parameters:
        p0,p1=(None,.75) if parameters['parameter']=='none' else parameter_values(parameters['parameter']);options['p0']=p0
        if name not in SINGLE or parameters['parameter']=='unexpected-p1':options['p1']=p1
    if 'pattern' in parameters:
        dtype=np.dtype(parameters['pattern_dtype']);pattern=parameters['pattern']
        image=(np.arange(35).reshape(5,7)%31).astype(dtype);fp=np.ones((3,3),np.uint8);options=dict(p0=.25)
        if name not in SINGLE:options['p1']=.75
        if pattern=='ties':image=np.resize(np.array([0,0,2,2,4,4,6],dtype),image.size).reshape(image.shape)
        elif pattern=='single-bin':image.fill(7)
        elif pattern=='all-zero':image.fill(0)
        elif pattern=='gaps':image=np.resize(np.array([0,1,8,13,30],dtype),image.size).reshape(image.shape)
        elif pattern=='center-low':image.fill(20);image[2,3]=0
        elif pattern=='center-high':image.fill(2);image[2,3]=30
        elif pattern=='zero-footprint':fp.fill(0)
        elif pattern=='zero-mask':options['mask']=np.zeros(image.shape,np.uint8)
        elif pattern=='disconnected':fp.fill(0);fp[0,0]=fp[2,1]=1
        elif pattern=='even':fp=np.ones((2,4),np.uint8)
        elif pattern=='shift':options.update(shift_x=1,shift_y=-1)
        elif pattern=='fractional-mean':image=np.resize(np.array([0,1,2,4,7],dtype),image.size).reshape(image.shape)
        elif pattern=='endpoints':options['p0']=0.;options.update({} if name in SINGLE else dict(p1=1.))
        args=(image,fp);tracked=dict(image=image,footprint=fp,**{k:v for k,v in options.items() if k=='mask'})
    if 'output_owner' in parameters:
        token=parameters['output_owner']
        if token=='owned-view':
            owner=np.full(args[0].size+2,99,np.uint8);out=owner[1:-1].reshape(args[0].shape);tracked['out_owner']=owner
        else:out=np.full(args[0].shape,99,np.dtype(token[7:],metadata={'rank':'percentile-output'}))
        options['out']=out;tracked['out']=out
    return args,options,tracked


def array(value):
    if isinstance(value,np.ndarray):
        data=repr(value.tolist()).encode() if value.dtype.hasobject else value.tobytes(order='C')
        result=dict(type=type(value).__name__,dtype=value.dtype.str,descr=repr(value.dtype.descr),metadata=repr(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),owned=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),nbytes=value.nbytes,values_sha256=hashlib.sha256(data).hexdigest())
        if value.size<=64 and value.dtype.kind in 'buif' and np.isfinite(value).all():result['values']=value.tolist()
        return result
    if isinstance(value,(list,tuple)):return dict(type=type(value).__name__,items=[array(v) for v in value])
    return dict(type=type(value).__name__,value=repr(value.item() if isinstance(value,np.generic) else value))


def safety(name,args,options,tracked):
    assert all(v.nbytes<=8*1024*1024 for v in tracked.values() if isinstance(v,np.ndarray))
    assert not any(np.shares_memory(a,b) for a in (tracked.get('image'),tracked.get('mask'),tracked.get('footprint')) if isinstance(a,np.ndarray) for b in (tracked.get('out'),) if isinstance(b,np.ndarray))
    # Only finite ordered in-range percentiles can reach a kernel. Non-real
    # scalar/array/object forms below are known C-argument conversion failures,
    # or bounded NumPy __float__ providers; no arbitrary stateful hooks here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        converted=[]
        for key,default in (('p0',0.),('p1',1.)):
            value=options.get(key,default)
            if isinstance(value,(str,list,tuple,SimpleNamespace)):continue
            try:value=float(value)
            except (TypeError,ValueError):continue
            assert math.isfinite(value) and 0<=value<=1,key
            converted.append((key,value))
        if len(converted)==2:assert converted[0][1]<=converted[1][1]
    # The shared base corpus excludes short buffers and actual zero-channel
    # output. Multi-channel output is allocation-safe, then reshape may fail
    # after first-channel writes; freeze that partial mutation rather than
    # inventing an early shape error.
    image=args[0]
    if isinstance(image,np.ndarray) and image.ndim==2 and image.size:
        if image.dtype==bool:return # check_nD then explicit bool rejection precedes mask/buffer access
        out=options.get('out');mask=options.get('mask')
        if name=='subtract_mean_percentile' and image.dtype==np.uint16:
            target=image.dtype if out is None else out.dtype
            if target.kind=='u' and target.itemsize in (1,2):
                maximum=int(image.max());mid=(max(3,maximum)+1)//2
                assert 0<=math.trunc(mid-maximum*.5)
                assert math.trunc(mid+maximum*.5)<=np.iinfo(target).max,'undefined floating-to-unsigned percentile store excluded'
        if isinstance(out,np.ndarray) and out.ndim>=2:
            assert all(out.shape[k]>=image.shape[k] for k in range(2))
            if out.ndim==3:assert out.shape[2]>0
        if isinstance(mask,np.ndarray):assert mask.size>=image.size


def observe(function,name,parameters):
    args,options,tracked=make_case(name,parameters);safety(name,args,options,tracked)
    inputs=[v for v in tracked.values() if isinstance(v,np.ndarray)];before={k:array(v) for k,v in tracked.items()};pointers={k:v.ctypes.data for k,v in tracked.items() if isinstance(v,np.ndarray)}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            value=function(*args,**options);result=array(value)
            if isinstance(value,np.ndarray):result.update(identity=[value is a for a in inputs],alias=[bool(np.shares_memory(value,a)) for a in inputs])
            error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={k:array(v) for k,v in tracked.items()},pointer_stable={k:tracked[k].ctypes.data==ptr for k,ptr in pointers.items()})


def snapshot():
    assert skimage.__version__=='0.26.0'
    return {key:observe(getattr(reference,name),name,parameters) for key,name,parameters in CASES}


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text())


def test_integrity_safe_scope_and_native_pending(frozen):
    assert frozen['pinned_skimage']==skimage.__version__=='0.26.0'
    assert frozen['native_checks']==0 and frozen['native_pending']==len(CASES)
    assert frozen['canonical_cases']==len(CASES)==len({key for key,_,_ in CASES})
    assert text_sha(Path(__file__).with_name('test_rank_neighborhood_contracts.py'))==frozen['fixture_source_sha256']
    for observations in frozen['snapshots'].values():assert set(observations)=={key for key,_,_ in CASES}
    assert all(digest(value)==key for key,value in frozen['observations'].items())


def test_export_source_facts(frozen):
    assert not hasattr(reference,'equalize_percentile') and not hasattr(percentile_module,'equalize_percentile')
    assert callable(reference.percentile) and reference.percentile is percentile_module.percentile
    assert importlib.util.find_spec('skimage.filters.rank.percentile') is None
    assert list(percentile_module.__all__)==frozen['module_all']
    for label,module in (('canonical','skimage.filters.rank'),('private_module','skimage.filters.rank._percentile')):
        namespace={};exec('from '+module+' import *',namespace)
        assert {name:name in namespace and namespace[name] is getattr(reference,name) for name in NAMES}==frozen['star_observations'][label]
    for name in NAMES:
        function=getattr(reference,name)
        assert function is getattr(percentile_module,name)
        assert name in reference.__all__
        assert (name in percentile_module.__all__)==frozen['functions'][name]['module_in_all']
        assert str(inspect.signature(function))==frozen['functions'][name]['signature']
        assert hashlib.sha256(inspect.getsource(function).encode()).hexdigest()==frozen['functions'][name]['python_source_sha256']


@pytest.mark.parametrize('namespace',('canonical','private-module-identity'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_original_frozen_contract(case,namespace,frozen):
    key,name,parameters=case;provider=reference if namespace=='canonical' else percentile_module
    assert observe(getattr(provider,name),name,parameters)==frozen['observations'][frozen['snapshots'][profile()][key]],key


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_strict_pending_never_counts_forwarding(case,workers):
    _,name,parameters=case;policy=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        if 'filters.rank.'+name in native_functions():pytest.skip('native transition needs separately sealed actual public implementation proof')
        actual=observe(getattr(importlib.import_module('skimagexx.filters.rank'),name),name,parameters)
        assert actual['error']['type']=='NotImplementedError' and 'fallback is disabled' in actual['error']['message']
        assert actual['result'] is None and not actual['warnings'] and actual['before']==actual['after']
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(policy)


def neighborhoods(image,footprint,mask,shifts):
    center=tuple(n//2+int(s) for n,s in zip(footprint.shape,(shifts[1],shifts[0])))
    for position in np.ndindex(image.shape):
        values=[]
        for offset in np.ndindex(footprint.shape):
            point=tuple(p+i-c for p,i,c in zip(position,offset,center))
            if footprint[offset] and all(0<=i<n for i,n in zip(point,image.shape)) and mask[point]:values.append(int(image[point]))
        yield position,Counter(values)


def scalar_replay(name,image,footprint,mask,shifts,p0,p1,dtype):
    bins=max(3,int(image.max()))+1 if image.dtype==np.uint16 else 256;out=np.empty(image.shape,dtype)
    for position,counts in neighborhoods(image,footprint,mask,shifts):
        pop=sum(counts.values());g=int(image[position]);value=0
        if pop:
            def crossing(reverse=False,strict=False,target=0.):
                total=0;selected=0
                for i in range(bins-1,-1,-1) if reverse else range(bins):
                    total+=counts.get(i,0)
                    if total>target*pop if strict else total>=target*pop:selected=i;break
                return selected
            if name in ('autolevel_percentile','enhance_contrast_percentile','gradient_percentile'):
                strict=name!='gradient_percentile';lo=crossing(strict=strict,target=p0);hi=crossing(True,strict,1.-p1)
                if name=='autolevel_percentile':value=(bins-1)*(min(max(lo,g),hi)-lo)//(hi-lo) if hi>lo else hi-lo
                elif name=='gradient_percentile':value=hi-lo
                else:value=hi if hi-g<g-lo else lo
            elif name=='percentile':
                value=max(counts) if p0==1 else crossing(strict=True,target=p0)
            elif name=='threshold_percentile':value=(bins-1)*int(g>=crossing(target=p0))
            else:
                cumulative=0;n=0;total=0
                for i in range(bins):
                    cumulative+=counts.get(i,0)
                    if cumulative>=p0*pop and cumulative<=p1*pop:n+=counts.get(i,0);total+=counts.get(i,0)*i
                if name=='pop_percentile':value=n
                elif n:
                    if name=='sum_percentile':value=total
                    elif name=='mean_percentile':value=total//n
                    else:
                        # Cython's mean/n is signed integer division BEFORE
                        # floating half scaling; no -1 offset in this family.
                        value=(g-total//n)*.5+bins//2
                        assert value==float(Fraction(g-total//n,2)+bins//2)
        if np.dtype(dtype).kind in 'ui':
            if name=='subtract_mean_percentile':assert 0<=math.trunc(value)<=np.iinfo(dtype).max
            value=int(value)%(np.iinfo(dtype).max+1)
        out[position]=value
    return out


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('interval',((0.,1.),(.25,.75),(.5,.5),(1.,1.),(0.,0.)))
@pytest.mark.parametrize('output',('same','f4','f8'))
def test_independent_bin_scan_replay(name,dtype,kind,interval,output):
    image=(np.arange(35).reshape(5,7)%31).astype(dtype);fp=np.ones((2,4) if kind=='even' else (3,3),np.uint8);mask=np.ones(image.shape,np.uint8);shifts=(0,0)
    if kind=='disconnected':fp.fill(0);fp[0,0]=fp[-1,-1]=1
    elif kind=='zero':mask.fill(0)
    elif kind=='masked':mask.flat[::3]=0
    elif kind=='shift':shifts=(1,-1)
    p0,p1=interval;target=np.dtype(dtype if output=='same' else output)
    expected=scalar_replay(name,image,fp,mask,shifts,p0,p1,target)
    options=dict(mask=mask,shift_x=shifts[0],shift_y=shifts[1],p0=p0)
    if name not in SINGLE:options['p1']=p1
    if output!='same':options['out']=np.full(image.shape,99,target)
    before=fp.copy();actual=getattr(reference,name)(image,fp,**options)
    assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()
    np.testing.assert_array_equal(fp,before)


@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','masked','empty'))
@pytest.mark.parametrize('q',(Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2),Fraction(3,4),Fraction(7,8),Fraction(1)))
def test_sorted_order_statistic_oracle_without_histogram_scan(dtype,kind,q):
    image=(np.arange(35).reshape(5,7)%31).astype(dtype);fp=np.ones((2,4) if kind=='even' else (3,3),np.uint8);mask=np.ones(image.shape,np.uint8)
    if kind=='masked':mask.flat[::3]=0
    elif kind=='empty':mask.fill(0)
    bins=max(3,int(image.max()))+1 if dtype=='u2' else 256
    quantile=np.empty(image.shape,image.dtype);threshold=np.empty_like(quantile)
    for point,counts in neighborhoods(image,fp,mask,(0,0)):
        ordered=sorted(value for value,count in counts.items() for _ in range(count));pop=len(ordered)
        if not pop:quantile[point]=threshold[point]=0;continue
        quantile[point]=ordered[-1] if q==1 else ordered[int(q*pop)]
        cut=0 if q==0 else ordered[math.ceil(q*pop)-1]
        threshold[point]=(bins-1)*int(image[point]>=cut)
    actual=reference.percentile(image,fp,mask=mask,p0=float(q));other=reference.threshold_percentile(image,fp,mask=mask,p0=float(q))
    assert actual.tobytes()==quantile.tobytes() and other.tobytes()==threshold.tobytes()


class FloatEvent:
    def __init__(self,name,value,events,fail=False):self.name=name;self.value=value;self.events=events;self.fail=fail
    def __float__(self):
        self.events.append(self.name)
        if self.fail:raise ValueError(self.name+' conversion')
        return self.value


def parameter_observe(name,failure):
    events=[];image=np.arange(12,dtype=np.uint8).reshape(3,4);fp=np.ones((3,3),np.uint8);out=np.full(image.shape,99,np.int32 if failure=='bad-output' else np.uint8)
    options=dict(out=out,p0=FloatEvent('p0',.25,events,failure=='p0'))
    if name not in SINGLE:options['p1']=FloatEvent('p1',.75,events,failure=='p1')
    if failure=='bad-shift':options['shift_x']='bad'
    if failure=='bad-footprint':fp=None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:value=getattr(reference,name)(image,fp,**options);error=None
        except Exception as caught:value=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(events=events,error=error,output=array(out),result=None if value is None else array(value),warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices])


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('failure',('none','p0','p1','bad-output','bad-shift','bad-footprint'))
def test_parameter_provider_order_and_no_early_writes(name,failure,frozen):
    assert parameter_observe(name,failure)==frozen['parameter_events'][profile()][name+'/'+failure]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True);args=parser.parse_args()
    args.record.write_text(json.dumps(dict(profile=profile(),runtime=dict(python=os.sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__),snapshots=snapshot(),parameter_events={name+'/'+failure:parameter_observe(name,failure) for name in NAMES for failure in ('none','p0','p1','bad-output','bad-shift','bad-footprint')}),indent=2,allow_nan=False)+'\n')
