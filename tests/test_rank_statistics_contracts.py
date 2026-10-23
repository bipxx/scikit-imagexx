"""Pinned rank-statistics reference contracts; strict pending is not native proof.

Cython disables bounds checks. Every executable vector case has a positive
channel count within the prepared histogram capacity and adequate buffers.
"""
import argparse
from collections import Counter
from decimal import Decimal,localcontext,ROUND_HALF_UP
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import operator
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
import scipy
import skimage
from skimage.filters import rank as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/rank-statistics.json'
NAMES=('geometric_mean','noise_filter','subtract_mean','windowed_histogram')
SCALAR=NAMES[:3]
spec=importlib.util.spec_from_file_location('rank_statistics_qualified_base',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


def cases():
    rows=[]
    for name in NAMES:
        for key,old,parameters in base.CASES:
            if old!='minimum' or parameters.get('out') in ('image','view','overlap') or parameters.get('special')=='alias-converted':continue
            if name=='windowed_histogram' and parameters.get('ndim')==2 and parameters.get('special')=='bins65535':continue
            rows.append((name+'/'+key.split('/',1)[1],name,dict(parameters)))
    for name in SCALAR:
        for ndim in (2,3):
            for pattern in ('flat-zero','flat-seven','geometric','fractional','low-center','high-center','zero-window','center-only','one-sided-low','one-sided-high'):
                rows.append((f'{name}/{ndim}d/statistic={pattern}',name,dict(ndim=ndim,statistic=pattern)))
            for output in ('u1','u2','f4','f8'):
                rows.append((f'{name}/{ndim}d/fractional/out={output}',name,dict(ndim=ndim,statistic='fractional',statistic_out=output)))
            rows.append((f'{name}/{ndim}d/singleton-u16-full-bin',name,dict(ndim=ndim,statistic='singleton-u16')))
    for token in ('none','one','four','full256','negative','float','string','bool','numpy-int','array0','array1'):
        rows.append((f'windowed_histogram/2d/n_bins={token}','windowed_histogram',dict(ndim=2,histogram=True,bins=token)))
    for output in ('u1','u2','f4','f8','bool','i1','i2','u4','i8','f2','c8','O','tagged','readonly','F','reverse','step','oversized','two-dimensional'):
        rows.append((f'windowed_histogram/2d/vector-out={output}','windowed_histogram',dict(ndim=2,histogram=True,vector_out=output,bins='four')))
    for token in ('zero','negative','float','string','too-deep'):
        rows.append((f'windowed_histogram/2d/ignored-n_bins={token}','windowed_histogram',dict(ndim=2,histogram=True,vector_out='f8',bins=token)))
    for dtype in ('u1','u2','f4','f8'):
        rows.append((f'windowed_histogram/2d/pop49/out={dtype}','windowed_histogram',dict(ndim=2,histogram=True,histogram_pattern='pop49',vector_out=dtype,bins='four')))
    for pattern in ('float-conversion','zero-window','singleton-u16'):
        rows.append((f'windowed_histogram/2d/histogram={pattern}','windowed_histogram',dict(ndim=2,histogram=True,histogram_pattern=pattern)))
    for name in NAMES:rows.append((name+'/missing-footprint',name,dict(ndim=2,omitted=True)))
    assert len({r[0] for r in rows})==len(rows)
    return rows


CASES=cases()


def make_case(parameters):
    if parameters.get('omitted'):
        image=np.arange(6,dtype=np.uint8).reshape(2,3);return (image,),{},dict(image=image)
    if 'statistic' not in parameters and not parameters.get('histogram'):return base.make_case(parameters)
    ndim=parameters['ndim'];shape=(5,7) if ndim==2 else (3,4,5);pattern=parameters.get('statistic',parameters.get('histogram_pattern','ramp'))
    image=(np.arange(math.prod(shape)).reshape(shape)%13).astype(np.uint8);footprint=np.ones((3,)*ndim,np.uint8);options={};owners={}
    if pattern=='flat-zero':image.fill(0)
    elif pattern=='flat-seven':image.fill(7)
    elif pattern=='geometric':image=np.resize(np.array([0,1,3,7,8,15],np.uint8),image.size).reshape(shape)
    elif pattern=='fractional':image=np.resize(np.array([0,0,1,2,4,7],np.uint8),image.size).reshape(shape);footprint=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8)
    elif pattern in ('low-center','high-center','one-sided-low','one-sided-high'):
        image.fill(8 if pattern in ('low-center','one-sided-low') else 20);image[tuple(n//2 for n in shape)]=1 if pattern in ('low-center','one-sided-low') else 250
    elif pattern=='zero-window':options['mask']=np.zeros(shape,np.uint8)
    elif pattern=='center-only':footprint.fill(0);footprint[(1,)*ndim]=1
    elif pattern=='singleton-u16':image=np.full((1,)*ndim,65535,np.uint16);footprint=np.ones((1,)*ndim,np.uint8)
    elif pattern=='float-conversion':image=np.resize(np.array([0.,.5,1.],np.float32),image.size).reshape(shape)
    elif pattern=='pop49':image=np.full((7,7),2,np.uint8);footprint=np.ones((7,7),np.uint8)
    output=parameters.get('statistic_out')
    if output:options['out']=np.full(image.shape,99,np.dtype(output))
    if parameters.get('histogram'):
        bins={'none':None,'one':1,'four':4,'full256':256,'negative':-1,'zero':0,'float':1.5,'string':'bad','bool':True,'numpy-int':np.int64(4),'array0':np.array(4),'array1':np.array([4]),'too-deep':512}
        if 'bins' in parameters:options['n_bins']=bins[parameters['bins']]
        output=parameters.get('vector_out')
        if output:
            special=output in ('readonly','F','reverse','step','oversized','two-dimensional')
            dtype=np.dtype('u1',metadata={'rank':'statistics-out'}) if output=='tagged' else np.dtype('f8' if special else output)
            extent=tuple(n+2 for n in image.shape) if output=='oversized' else image.shape
            out=np.full(extent+(4,),99,dtype)
            if output=='two-dimensional':out=np.full(image.shape,99,dtype)
            elif output=='readonly':out.flags.writeable=False
            elif output=='F':out=np.asfortranarray(out)
            elif output=='reverse':out=out[::-1]
            elif output=='step':
                owner=np.full(image.shape+(8,),99,dtype);out=owner[...,::2];owners['out_owner']=owner
            options['out']=out
    return (image,footprint),options,dict(image=image,footprint=footprint,**{k:v for k,v in options.items() if k in ('mask','out')},**owners)


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
    if name=='subtract_mean' and len(args)==2:
        image,fp=args;out=options.get('out')
        if isinstance(image,np.ndarray) and image.dtype==np.uint16 and image.size and image.ndim in (2,3) and isinstance(fp,np.ndarray) and fp.ndim==image.ndim and all(fp.shape):
            dtype=image.dtype if out is None else out.dtype
            if dtype.kind in 'ui':
                # Qualify the floating-to-unsigned domain before the C call.
                scalar_replay(name,image,fp,np.ones(image.shape,np.uint8),(0,0,0),dtype)
    if name!='windowed_histogram' or len(args)!=2:return
    image=args[0]
    if not isinstance(image,np.ndarray) or image.ndim!=2 or image.size==0:return # checked before kernel
    if image.dtype.kind not in 'buif':return # unsupported conversion fails before kernel
    capacity=max(3,int(image.max()))+1 if image.dtype==np.uint16 else 256
    supplied=options.get('out')
    if supplied is not None:
        if supplied.ndim==3:
            assert supplied.shape[0]>=image.shape[0] and supplied.shape[1]>=image.shape[1]
            assert 1<=supplied.shape[2]<=capacity
        return # other rank/reshape errors are checked before memoryview/kernel
    bins=options.get('n_bins')
    if bins is None:
        try:bins=int(image.max())+1
        except (TypeError,ValueError,OverflowError):return # default evaluation fails before kernel
    try:depth=operator.index(bins)
    except TypeError:
        if isinstance(bins,np.ndarray):assert bins.size==1 and int(bins.flat[0])==4
        return # allocation rejects non-index forms (or bounded singleton4)
    if depth<0:return # NumPy allocation rejects negative dimensions
    assert 1<=depth<=capacity,'unchecked vector output depth excluded'
    assert image.size*depth*8<=8*1024*1024,'unbounded vector allocation excluded'


def observe(function,name,parameters):
    args,options,tracked=make_case(parameters);safety(name,args,options,tracked)
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


def test_source_integrity_safety_and_explicit_native_pending(frozen):
    assert frozen['pinned_skimage']==skimage.__version__=='0.26.0'
    assert frozen['native_checks']==0 and frozen['native_pending']==len(CASES)
    assert len({key for key,_,_ in CASES})==len(CASES)==frozen['canonical_cases']
    assert text_sha(Path(__file__).with_name('test_rank_neighborhood_contracts.py'))==frozen['fixture_source_sha256']
    assert all(p.get('out') not in ('image','view','overlap') and p.get('special')!='alias-converted' for _,_,p in CASES)
    for source in frozen['snapshots'].values():assert set(source)=={key for key,_,_ in CASES}
    assert all(digest(value)==key for key,value in frozen['observations'].items())


@pytest.mark.parametrize('name',NAMES)
def test_original_signatures_and_identity_aliases(name,frozen):
    generic=importlib.import_module('skimage.filters.rank.generic');original=getattr(reference,name)
    assert original is getattr(generic,name)
    assert (name in reference.__all__)==frozen['functions'][name]['canonical_in_all']
    assert (name in generic.__all__)==frozen['functions'][name]['generic_in_all']
    assert str(inspect.signature(original))==frozen['functions'][name]['signature']
    assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==frozen['functions'][name]['python_source_sha256']


@pytest.mark.parametrize('namespace',('filters.rank','filters.rank.generic'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_original_frozen_contract(case,namespace,frozen):
    key,name,parameters=case;function=getattr(importlib.import_module('skimage.'+namespace),name)
    assert observe(function,name,parameters)==frozen['observations'][frozen['snapshots'][profile()][key]],key


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('namespace',('filters.rank','filters.rank.generic'))
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_strict_pending_never_counts_forwarding(case,namespace,workers):
    _,name,parameters=case;policy=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        if 'filters.rank.'+name in native_functions():pytest.skip('native transition requires separately sealed actual public implementation proof')
        actual=observe(getattr(importlib.import_module('skimagexx.'+namespace),name),name,parameters)
        assert actual['error']['type']=='NotImplementedError' and 'fallback is disabled' in actual['error']['message']
        assert actual['result'] is None and not actual['warnings'] and actual['before']==actual['after']
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(policy)


def oracle_fixture(ndim,dtype,kind):
    shape=(5,7) if ndim==2 else (3,4,5)
    image=(np.arange(math.prod(shape)).reshape(shape)%31).astype(dtype)
    footprint=np.ones(((2,4) if ndim==2 else (2,3,4)) if kind=='even' else (3,)*ndim,np.uint8)
    mask=np.ones(shape,np.uint8);shifts=(0,0,0)
    if kind=='disconnected':footprint.flat[1:-1]=0
    elif kind=='zero':mask.fill(0)
    elif kind=='masked':mask.flat[::3]=0
    elif kind=='shift':shifts=(1,-1,0)
    return image,footprint,mask,shifts


def neighborhoods(image,footprint,mask,shifts,noise=False):
    fp=(footprint>0).copy();ndim=image.ndim
    if noise:
        # Public3D wrapper removes [shift_y,shift_x,shift_z], while
        # the actual3D core centers use [shift_x,shift_y,shift_z].
        removed=(fp.shape[0]//2+shifts[1],fp.shape[1]//2+shifts[0])
        if ndim==3:removed+=(fp.shape[2]//2+shifts[2],)
        fp[removed]=False
    axes=(shifts[1],shifts[0]) if ndim==2 else shifts
    center=tuple(n//2+s for n,s in zip(fp.shape,axes))
    for position in np.ndindex(image.shape):
        values=[]
        for offset in np.ndindex(fp.shape):
            point=tuple(p+i-c for p,i,c in zip(position,offset,center))
            if fp[offset] and all(0<=i<n for i,n in zip(point,image.shape)) and mask[point]:values.append(int(image[point]))
        yield position,Counter(values)


def scalar_replay(name,image,footprint,mask,shifts,dtype):
    output=np.empty(image.shape,dtype);bins=max(3,int(image.max()))+1 if image.dtype==np.uint16 else 256
    for position,counts in neighborhoods(image,footprint,mask,shifts,name=='noise_filter'):
        pop=sum(counts.values());g=int(image[position])
        if name=='noise_filter':
            if counts.get(g,0):value=0
            else:
                lower=max((i for i in counts if i<=g),default=0)
                upper=min((i for i in counts if i>=g),default=bins-1)
                value=min(g-lower,upper-g)
        elif not pop:value=0
        elif name=='geometric_mean':
            total=0.
            for i in sorted(counts):total+=counts[i]*math.log(i+1)
            value=int((math.exp(total/pop)-1)+.5)
            with localcontext() as ctx:
                ctx.prec=80
                product=math.prod((i+1)**count for i,count in counts.items())
                exact=(Decimal(product)**(Decimal(1)/Decimal(pop)))-1
                assert value==int(exact.to_integral_value(rounding=ROUND_HALF_UP))
        else:
            total=sum(i*count for i,count in counts.items())
            value=((g-total/float(pop))/2.0+bins//2)-1.0
            rational=(Fraction(g)-Fraction(total,pop))/2+bins//2-1
            assert abs(value-float(rational))<=4*np.spacing(max(abs(float(rational)),1.))
        if np.dtype(dtype).kind in 'ui':
            value=int(value)
            if name=='subtract_mean':assert 0<=value<=np.iinfo(dtype).max
            else:value%=np.iinfo(dtype).max+1
        output[position]=value
    return output


@pytest.mark.parametrize('name',SCALAR)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','zero','masked','shift'))
@pytest.mark.parametrize('out_type',('same','f4','f8'))
def test_independent_arithmetic_and_ordered_replay(name,ndim,dtype,kind,out_type):
    image,fp,mask,shifts=oracle_fixture(ndim,dtype,kind);target=np.dtype(dtype if out_type=='same' else out_type)
    options=dict(mask=mask,shift_x=shifts[0],shift_y=shifts[1],shift_z=shifts[2])
    if out_type!='same':options['out']=np.full(image.shape,99,target)
    expected=scalar_replay(name,image,fp,mask,shifts,target)
    before=fp.copy();actual=getattr(reference,name)(image,fp,**options)
    assert actual.tobytes()==expected.tobytes() and actual.dtype==expected.dtype
    np.testing.assert_array_equal(fp,before)


@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','zero','masked'))
@pytest.mark.parametrize('depth',(4,32))
@pytest.mark.parametrize('out_type',('u1','u2','f4','f8'))
def test_histogram_independent_counts_and_exact_reciprocal_replay(dtype,kind,depth,out_type):
    image,fp,mask,shifts=oracle_fixture(2,dtype,kind)
    # For uint16 this fixture has max30, so actual capacity31, not32.
    effective_depth=min(depth,max(3,int(image.max()))+1) if dtype=='u2' else depth
    out=np.full(image.shape+(effective_depth,),99,np.dtype(out_type))
    actual=reference.windowed_histogram(image,fp,out=out,mask=mask,n_bins=effective_depth)
    expected=np.empty_like(out)
    for position,counts in neighborhoods(image,fp,mask,shifts):
        pop=sum(counts.values());scale=1.0/pop if pop else 0.
        for i in range(effective_depth):
            value=counts.get(i,0)*scale
            if pop:assert abs(value-float(Fraction(counts.get(i,0),pop)))<=2*np.spacing(max(value,1.))
            expected[position+(i,)]=int(value) if expected.dtype.kind in 'ui' else value
    assert actual is out and actual.tobytes()==expected.tobytes()


def test_histogram_integer_output_exposes_reciprocal_rounding():
    image=np.full((7,7),2,np.uint8);fp=np.ones((7,7),np.uint8)
    out=np.empty((7,7,4),np.uint8)
    value=reference.windowed_histogram(image,fp,out=out,n_bins=4)
    assert 49*(1.0/49)<1 and value[3,3,2]==0


def test_source_safe_singleton_full_uint16_histogram():
    image=np.array([[65535]],np.uint16);fp=np.ones((1,1),np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');out=reference.windowed_histogram(image,fp)
    assert out.shape==(1,1,65536) and out.dtype==np.float64 and out.nbytes==524288
    assert out[0,0,65535]==1 and np.count_nonzero(out)==1 and out.sum()==1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True);args=parser.parse_args()
    report=dict(profile=profile(),python=__import__('sys').version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,generic_python_sha256=text_sha(inspect.getsourcefile(reference.geometric_mean)),generic_extension_sha256=hashlib.sha256(Path(importlib.import_module('skimage.filters.rank.generic').generic_cy.__file__).read_bytes()).hexdigest(),snapshots=snapshot())
    args.record.parent.mkdir(parents=True,exist_ok=True);args.record.write_text(json.dumps(report,indent=2)+'\n');print(profile(),len(CASES),'reference cases recorded; native not exercised')
