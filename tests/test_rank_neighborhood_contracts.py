"""Pinned rank contracts: original observations and strict pending are separate.

Only defined, allocation-safe buffer cases enter the corpus. Upstream Cython
disables bounds checks; undersized mask/output and zero-depth output buffers
are explicitly outside this frozen corpus, not invented ValueError contracts.
"""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import warnings
import numpy as np
import pytest
import skimage
from skimage.filters import rank as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

NAMES=('minimum','maximum','mean','sum','pop')
MATRIX=Path(__file__).parents[1]/'coverage/contracts/rank-neighborhoods.json'


def cases():
    result=[]
    def add(name,label,**spec):result.append((name+'/'+label,name,spec))
    for name in NAMES:
        for ndim in (2,3):
            for dtype in ('u1','u2','bool','i1','i2','i4','i8','u4','u8','f2','f4','f8','c8','O','S2','datetime64[D]','>u2','tagged'):
                add(name,f'{ndim}d/dtype={dtype}',ndim=ndim,dtype=dtype)
            for dtype in ('u1','u2'):
                for layout in ('F','reverse','step','broadcast','readonly','unaligned'):
                    add(name,f'{ndim}d/{dtype}/layout={layout}',ndim=ndim,dtype=dtype,layout=layout)
            for footprint in ('cross','asymmetric','even','oversized','zero','empty','wrongrank','none','list','sequence','negative-float','nan','strings','complex','readonly','F'):
                add(name,f'{ndim}d/footprint={footprint}',ndim=ndim,footprint=footprint)
            for mask in ('ones','zero','checker','negative-int16','float','range-error','nan','readonly','F','reverse','wrongrank','oversized'):
                add(name,f'{ndim}d/mask={mask}',ndim=ndim,mask=mask)
            for shift in ('xplus','xminus','yplus','zplus','bool','numpy-bool','float','array0','array1','invalid-string','too-large','negative-too-large','center-outside','ignored-z'):
                add(name,f'{ndim}d/shift={shift}',ndim=ndim,shift=shift)
            for out in ('u1','u2','f4','f8','bool','i1','i2','u4','i8','f2','c8','O','F','reverse','step','readonly','one-channel','two-channel','extra-dim','wrongrank','oversized','image','view','overlap'):
                add(name,f'{ndim}d/out={out}',ndim=ndim,out=out)
            for special in ('overflow-pop','overflow-sum','rounding','bins1023','bins1024','bins65535','bad-float','negative-float','nan-image','list-image','precedence','alias-converted'):
                add(name,f'{ndim}d/special={special}',ndim=ndim,special=special)
        for shape in ((),(3,),(2,2,2,2),(0,7),(3,0,5),(1,1),(1,1,1)):
            add(name,f'shape={shape}',shape=shape,ndim=3 if len(shape)==3 else 2)
    assert len({case[0] for case in result})==len(result)
    return result


CASES=cases()


def make_case(spec):
    ndim=spec.get('ndim',2);shape=spec.get('shape',(5,7) if ndim==2 else (3,4,5))
    dtype=spec.get('dtype','u1');dtype=np.dtype('u1',metadata={'rank':'contract'}) if dtype=='tagged' else np.dtype(dtype)
    a=(np.arange(int(np.prod(shape))).reshape(shape)%29).astype(dtype)
    if dtype.kind=='f':a=a/np.asarray(29,dtype)
    if dtype.kind=='i' and a.size:a.flat[0]=-3
    fp=np.ones((3,)*ndim,np.uint8);options={};owners={}
    special=spec.get('special')
    if special=='overflow-pop':a=np.ones((17,19) if ndim==2 else (5,17,19),np.uint8);fp=np.ones((17,17) if ndim==2 else (3,17,17),np.uint8)
    elif special=='overflow-sum':a=np.full(shape,255,np.uint8)
    elif special=='rounding':a=(np.arange(a.size).reshape(shape)%3).astype(np.uint8)
    elif special in ('bins1023','bins1024','bins65535'):a=np.full(shape,int(special[4:]),np.uint16)
    elif special in ('bad-float','negative-float','nan-image'):
        a=np.linspace(0,1,a.size,dtype=np.float32).reshape(shape)
        a.flat[0]={'bad-float':2.,'negative-float':-.5,'nan-image':np.nan}[special]
    elif special=='list-image':a=a.tolist()
    elif special=='precedence':a=np.ones(shape,bool);fp=None;options.update(mask=np.asarray('bad'),shift_x='bad')
    elif special=='alias-converted':a=np.linspace(0,1,a.size,dtype=np.float32).reshape(shape);options['out']=a
    layout=spec.get('layout')
    if layout=='F':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[(slice(None,None,-1),)*ndim]
    elif layout=='step':a=a[(slice(None,None,2),)*ndim]
    elif layout=='broadcast':a=np.broadcast_to(a[(slice(0,1),)+(slice(None),)*(ndim-1)],a.shape)
    elif layout=='readonly':a.flags.writeable=False
    elif layout=='unaligned':
        b=np.ndarray(a.shape,a.dtype,buffer=bytearray(a.nbytes+1),offset=1);b[...] = a;a=b
    kind=spec.get('footprint')
    if kind=='cross':
        fp.fill(0)
        for axis in range(ndim):index=[1]*ndim;index[axis]=slice(None);fp[tuple(index)]=1
    elif kind=='asymmetric':fp.fill(0);fp.flat[0]=fp.flat[-2]=1
    elif kind=='even':fp=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8)
    elif kind=='oversized':fp=np.ones((9,)*ndim,np.uint8)
    elif kind=='zero':fp.fill(0)
    elif kind=='empty':fp=np.empty((0,)+(3,)*(ndim-1),np.uint8)
    elif kind=='wrongrank':fp=np.ones((3,)*(5-ndim),np.uint8)
    elif kind=='none':fp=None
    elif kind=='list':fp=fp.tolist()
    elif kind=='sequence':fp=((fp,1),)
    elif kind=='negative-float':fp=np.linspace(-1,1,fp.size).reshape(fp.shape)
    elif kind=='nan':fp=fp.astype(float);fp.flat[0]=np.nan
    elif kind=='strings':fp=fp.astype('U1')
    elif kind=='complex':fp=fp.astype(np.complex64)
    elif kind=='readonly':fp.flags.writeable=False
    elif kind=='F':fp=np.asfortranarray(fp)
    mask_kind=spec.get('mask')
    if mask_kind:
        mask=np.ones(a.shape,np.uint8)
        if mask_kind=='zero':mask.fill(0)
        elif mask_kind=='checker':mask=(np.indices(a.shape).sum(axis=0)%2).astype(np.uint8)
        elif mask_kind=='negative-int16':mask=mask.astype(np.int16);mask.flat[::2]=-1
        elif mask_kind=='float':mask=mask.astype(float);mask.flat[::2]=.2
        elif mask_kind=='range-error':mask=mask.astype(float);mask.flat[0]=2
        elif mask_kind=='nan':mask=mask.astype(float);mask.flat[0]=np.nan
        elif mask_kind=='readonly':mask.flags.writeable=False
        elif mask_kind=='F':mask=np.asfortranarray(mask)
        elif mask_kind=='reverse':mask=mask[(slice(None,None,-1),)*ndim]
        elif mask_kind=='wrongrank':mask=np.ones((mask.size,),np.uint8)
        elif mask_kind=='oversized':mask=(np.arange(int(np.prod(tuple(s+2 for s in a.shape)))).reshape(tuple(s+2 for s in a.shape))%3==0).astype(np.uint8)
        options['mask']=mask
    shift=spec.get('shift')
    if shift:
        key='shift_y' if shift=='yplus' else 'shift_z' if shift in ('zplus','ignored-z') else 'shift_x'
        value={'xplus':1,'xminus':-1,'yplus':1,'zplus':1,'bool':True,'numpy-bool':np.bool_(True),'float':.5,'array0':np.asarray(0),'array1':np.asarray([0]),'invalid-string':'bad','too-large':128,'negative-too-large':-129,'center-outside':7,'ignored-z':'ignored'}[shift]
        options[key]=value
    out_kind=spec.get('out')
    if out_kind:
        if out_kind in ('image','view'):out=a if out_kind=='image' else a.view()
        elif out_kind=='overlap':
            owner=np.arange(a.size+1,dtype=np.uint8);a=owner[:-1].reshape(a.shape);out=owner[1:].reshape(a.shape);owners['overlap_owner']=owner
        elif out_kind=='step':
            owner=np.full(a.shape[:-1]+(a.shape[-1]*2,),99,np.uint8);out=owner[...,::2];owners['out_owner']=owner
        elif out_kind=='wrongrank':out=np.full((a.size,),99,np.uint8)
        elif out_kind in ('one-channel','two-channel','extra-dim'):
            extension=(2,) if out_kind=='two-channel' else (1,1) if out_kind=='extra-dim' else (1,)
            out=np.full(a.shape+extension,99,np.uint8)
        elif out_kind=='oversized':out=np.full(tuple(size+2 for size in a.shape),99,np.uint8)
        else:
            out=np.full(a.shape,99,np.uint8 if out_kind in ('F','reverse','readonly') else np.dtype(out_kind))
            if out_kind=='F':out=np.asfortranarray(out)
            elif out_kind=='reverse':out=out[(slice(None,None,-1),)*ndim]
            elif out_kind=='readonly':out.flags.writeable=False
        options['out']=out
    tracked={'image':a,'footprint':fp,**{k:v for k,v in options.items() if k in ('mask','out')},**owners}
    return (a,fp),options,tracked


def describe(value,inputs=()):
    if isinstance(value,np.ndarray):
        return dict(type=type(value).__name__,dtype=value.dtype.str,descr=repr(value.dtype.descr),metadata=repr(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),owned=bool(value.flags.owndata),writeable=bool(value.flags.writeable),values_sha256=hashlib.sha256(repr(value.tolist()).encode()).hexdigest(),identity=[value is item for item in inputs],alias=[bool(np.shares_memory(value,item)) for item in inputs])
    if isinstance(value,(tuple,list)):return dict(type=type(value).__name__,items=[describe(item) for item in value])
    return dict(type=type(value).__name__,value=repr(value.item() if isinstance(value,np.generic) else value))


def observe(function,spec):
    args,options,tracked=make_case(spec)
    inputs=[value for value in tracked.values() if isinstance(value,np.ndarray)]
    before={key:describe(value) for key,value in tracked.items()}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=describe(function(*args,**options),inputs),None
        except Exception as caught:result,error=None,dict(type=type(caught).__name__,message=str(caught))
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after={key:describe(value) for key,value in tracked.items()})


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def snapshot():
    assert skimage.__version__=='0.26.0'
    return {key:observe(getattr(reference,name),spec) for key,name,spec in CASES}


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name',NAMES)
def test_nested_signatures(frozen,name):
    expected=frozen['functions'][name]['signature']
    assert str(inspect.signature(getattr(reference,name)))==expected
    for path in ('skimagexx.filters.rank','skimagexx.filters.rank.generic'):
        assert str(inspect.signature(getattr(importlib.import_module(path),name)))==expected
    assert xx.filters.rank is importlib.import_module('skimagexx.filters.rank')


def test_corpus_integrity(frozen):
    expected=frozen['snapshots']['numpy2']
    assert set(expected)=={key for key,_,_ in CASES}
    assert digest(expected)==frozen['corpus_sha256']
    for key,value in frozen['observations'].items():assert digest(value)==key
    assert frozen['native_contract_checks']==0 and frozen['native_pending']==len(CASES)


@pytest.mark.parametrize('case',CASES,ids=[case[0] for case in CASES])
def test_original_frozen_contract(frozen,case):
    key,name,spec=case;profile='numpy1' if np.__version__.startswith('1.') else 'numpy2'
    expected=frozen['snapshots']['numpy2'].copy();expected.update(frozen['snapshot_overrides'].get(profile,{}))
    actual=observe(getattr(reference,name),spec)
    assert actual==frozen['observations'][expected[key]],(key,digest(actual))


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('case',CASES,ids=[case[0] for case in CASES])
def test_compatibility_is_separate_evidence(case,workers):
    _,name,spec=case;previous=xx.set_fallback_enabled(True);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        expected=observe(getattr(reference,name),spec)
        actual=observe(getattr(importlib.import_module('skimagexx.filters.rank'),name),spec)
        forwarding='filters.rank.'+name not in native_functions()
        fallback=[warning for warning in actual['warnings'] if warning['type']=='FallbackWarning']
        assert len(fallback)==int(forwarding)
        actual['warnings']=[warning for warning in actual['warnings'] if warning['type']!='FallbackWarning']
        assert actual==expected
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('case',CASES,ids=[case[0] for case in CASES])
@pytest.mark.parametrize('namespace',['filters.rank','filters.rank.generic'])
def test_strict_native_pending_is_not_implementation(case,namespace,workers):
    _,name,spec=case
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        function=getattr(importlib.import_module('skimagexx.'+namespace),name)
        if 'filters.rank.'+name not in native_functions():
            result=observe(function,spec)
            assert result['result'] is None and result['error']['type']=='NotImplementedError'
            assert 'fallback is disabled' in result['error']['message']
            assert not result['warnings'] and result['after']==result['before']
        else:assert observe(function,spec)==observe(getattr(reference,name),spec)
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


def neighborhood_oracle(name,image,footprint,mask,center,dtype):
    out=np.empty(image.shape,dtype)
    for position in np.ndindex(image.shape):
        values=[]
        for index in np.ndindex(footprint.shape):
            point=tuple(p+i-c for p,i,c in zip(position,index,center))
            if footprint[index] and all(0<=i<size for i,size in zip(point,image.shape)) and mask[point]:values.append(int(image[point]))
        if not values:value=0
        elif name=='minimum':value=min(values)
        elif name=='maximum':value=max(values)
        elif name=='sum':value=sum(values)
        elif name=='pop':value=len(values)
        else:value=sum(values)/len(values)
        if np.dtype(dtype).kind=='u':value=int(value)%(1<<(8*np.dtype(dtype).itemsize))
        out[position]=value
    return out


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('output',[None,np.float32,np.float64])
@pytest.mark.parametrize('pattern',['asymmetric','empty-neighborhood','fractional'])
def test_independent_border_mask_shift_definition(name,ndim,dtype,output,pattern):
    shape=(4,5) if ndim==2 else (3,4,5)
    image=(np.arange(int(np.prod(shape))).reshape(shape)%7).astype(dtype)
    fp=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8);mask=np.ones(shape,np.uint8)
    if pattern=='asymmetric':fp.flat[::3]=0;mask.flat[::4]=0
    elif pattern=='empty-neighborhood':mask.fill(0)
    shifts=(-1,1) if ndim==2 else (-1,1,0)
    options=dict(mask=mask,shift_x=shifts[0],shift_y=shifts[1],shift_z=0 if ndim==2 else shifts[2])
    dtype_out=dtype if output is None else output
    if output is not None:options['out']=np.full(shape,99,output)
    center=(fp.shape[0]//2+shifts[1],fp.shape[1]//2+shifts[0]) if ndim==2 else tuple(s//2+t for s,t in zip(fp.shape,shifts))
    # Keep the even 2D footprint center inside its two-row bounding box.
    if ndim==2:options['shift_y']=0;center=(fp.shape[0]//2,center[1])
    expected=neighborhood_oracle(name,image,fp,mask,center,dtype_out)
    actual=getattr(reference,name)(image,fp,**options)
    np.testing.assert_array_equal(actual,expected)




@pytest.mark.parametrize('name',NAMES)
def test_original_generic_alias_contract(frozen,name):
    original_generic=importlib.import_module('skimage.filters.rank.generic')
    alias=frozen['namespace_assessment']['selected_alias_contracts'][name]
    assert getattr(original_generic,name) is getattr(reference,name)
    assert alias['reference_identity'] and alias['externally_reachable_by_import']
    assert str(inspect.signature(getattr(original_generic,name)))==alias['signature']
    assert hashlib.sha256(inspect.getsource(getattr(original_generic,name)).encode()).hexdigest()==alias['reference_source_sha256']
    assert alias['inherits_canonical_case_count']==sum(key.startswith(name+'/') for key,_,_ in CASES)
    if 'filters.rank.'+name in native_functions():
        assert getattr(importlib.import_module('skimagexx.filters.rank.generic'),name) is getattr(importlib.import_module('skimagexx.filters.rank'),name)


def test_namespace_discovery_classification(frozen):
    original_generic=importlib.import_module('skimage.filters.rank.generic')
    assessment=frozen['namespace_assessment'];entries=assessment['discovered_uninventoried']
    discovered={'filters.rank.'+name for name in dir(reference) if not name.startswith('_') and name not in reference.__all__}
    discovered|={'filters.rank.generic.'+name for name in dir(original_generic) if not name.startswith('_')}
    assert discovered=={entry['path'] for entry in entries}
    assert assessment['declared_generic_exports']==list(original_generic.__all__)
    for entry in entries:
        assert entry['source_evidence'] and not entry['count_as_additional_algorithm']
        name=entry['path'].rsplit('.',1)[1]
        if entry['classification'] in ('explicit_public_export_alias','legacy_callable_alias'):
            assert getattr(original_generic,name) is getattr(reference,name)
            assert name in reference.__all__
            assert (name in original_generic.__all__)==(entry['classification']=='explicit_public_export_alias')
        else:
            assert entry['classification'] in ('imported_dependency_module','imported_dependency_callable','implementation_helper_module','legacy_module_namespace')
            assert name not in reference.__all__ and name not in original_generic.__all__


if __name__=='__main__':
    import argparse,sys
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);arguments=parser.parse_args()
    data=dict(python=sys.version,numpy=np.__version__,skimage=skimage.__version__,case_count=len(CASES),observations=snapshot(),source_hashes={name:hashlib.sha256(inspect.getsource(getattr(reference,name)).encode()).hexdigest() for name in NAMES})
    arguments.snapshot.write_text(json.dumps(data,sort_keys=True,indent=2),encoding='utf-8')
