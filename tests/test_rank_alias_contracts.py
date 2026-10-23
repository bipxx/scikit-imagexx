"""Owned, bounds-safe original rank overlap contracts; no candidate forwarding."""
import ctypes
import ctypes.util
import hashlib
from functools import lru_cache
import inspect
import json
import math
import os
from pathlib import Path
import struct
import warnings

import numpy as np
import pytest
import skimage
from skimage.filters import rank as reference

NAMES=('gradient','median','modal','entropy')
MATRIX=Path(__file__).parents[1]/'coverage/contracts/rank-aliases.json'
OUTPUTS=('u1','u2','bool','f4','f8')


def cases():
    rows=[]
    def add(name,label,**spec):rows.append((name+'/'+label,name,spec))
    for name in NAMES:
        for ndim in (2,3):
            for dtype in ('u1','u2'):
                for output in OUTPUTS:
                    for kind in ('view','offset','mask','both'):
                        if name=='entropy' and output not in ('f4','f8'):continue
                        if dtype=='u2' and output in ('f4','f8') and kind!='mask':continue
                        add(name,f'core/{ndim}d/{dtype}/{output}/{kind}',ndim=ndim,dtype=dtype,output=output,kind=kind)
            for dtype in ('u1','u2'):
                for layout in ('F','reverse','step'):
                    for kind in ('view','mask'):
                        for output in (OUTPUTS if name=='entropy' else ('u1','f8')):
                            add(name,f'copy/{ndim}d/{dtype}/{output}/{kind}/{layout}',ndim=ndim,dtype=dtype,output=output,kind=kind,layout=layout)
            if name!='entropy':
                for dtype in ('u1','u2'):
                    for kind in ('view','offset','mask','both'):
                        for variant in ('point','wide','disconnected','even-shifted','zero'):
                            add(name,f'traversal/{ndim}d/{dtype}/{kind}/{variant}',ndim=ndim,dtype=dtype,output=dtype,kind=kind,footprint=variant,mask_state='checker' if variant in ('point','disconnected') else 'ones')
            else:
                for dtype in ('u1','u2'):
                    for output in OUTPUTS:
                        for kind in ('view','mask'):
                            if dtype=='u2' and output in ('f4','f8') and kind=='view':continue
                            for invariant in ('zero-footprint','zero-mask'):
                                add(name,f'zero/{ndim}d/{dtype}/{output}/{kind}/{invariant}',ndim=ndim,dtype=dtype,output=output,kind=kind,footprint='zero' if invariant=='zero-footprint' else 'dense',mask_state='zero' if invariant=='zero-mask' else 'ones')
            for output in ('u1','f8'):
                for mask_dtype in ('u2','>u2','bool'):
                    add(name,f'mask-conversion/{ndim}d/{output}/{mask_dtype}',ndim=ndim,dtype='u2',output=output,kind='mask',mask_dtype=mask_dtype)
            for output in ('u1','f8'):
                add(name,f'endian-copy/{ndim}d/{output}',ndim=ndim,dtype='>u2',output=output,kind='view')
            for kind in ('view','mask'):
                for geometry in ('oversized','channels2'):
                    output='f8' if name=='entropy' else 'u1'
                    add(name,f'geometry/{ndim}d/{kind}/{geometry}',ndim=ndim,dtype='u1',output=output,kind=kind,geometry=geometry)
            for error in ('readonly-image','readonly-mask','readonly-out','F-out','step-out','bad-center','bad-shift','same-object','unsupported-out'):
                add(name,f'admission/{ndim}d/{error}',ndim=ndim,dtype='u1',output='f8' if name=='entropy' else 'u1',kind='mask' if error=='readonly-mask' else 'view',error=error)
            for shift in ('bool','array1'):
                add(name,f'shift/{ndim}d/{shift}',ndim=ndim,dtype='u1',output='f8' if name=='entropy' else 'u1',kind='mask',shift=shift)
            add(name,f'mask-identity/{ndim}d',ndim=ndim,dtype='u1',output='f8' if name=='entropy' else 'u1',kind='mask',mask_identity=name!='entropy')
            add(name,f'disjoint-common-owner/{ndim}d',ndim=ndim,dtype='u2',output='f8',kind='disjoint')
            add(name,f'population-drift/{ndim}d',ndim=ndim,shape=(7,7) if ndim==2 else (2,7,7),dtype='u1',output='f8' if name=='entropy' else 'u1',kind='mask',mask_state='first-three',output_offset=1)
    assert len({row[0] for row in rows})==len(rows)
    return rows


CASES=cases()


def shaped_view(owner,shape,dtype,offset=0,layout='C'):
    dtype=np.dtype(dtype)
    if layout=='step':
        wide=shape[:-1]+(shape[-1]*2,)
        return np.ndarray(wide,dtype,buffer=owner,offset=offset)[...,::2]
    result=np.ndarray(shape,dtype,buffer=owner,offset=offset,order='F' if layout=='F' else 'C')
    return result[(slice(None,None,-1),)*len(shape)] if layout=='reverse' else result


def byte_extent(array):
    low=high=int(array.ctypes.data)
    for size,stride in zip(array.shape,array.strides):
        low+=min(0,(size-1)*stride);high+=max(0,(size-1)*stride)
    return low,high+array.itemsize


def make_case(spec):
    ndim=spec['ndim'];shape=tuple(spec.get('shape',(5,7) if ndim==2 else (3,4,5)));count=math.prod(shape)
    dtype=np.dtype(spec['dtype']);output=np.dtype('i2' if spec.get('error')=='unsupported-out' else spec['output'])
    kind=spec['kind'];layout=spec.get('layout','C');geometry=spec.get('geometry')
    out_shape=tuple(s+2 for s in shape) if geometry=='oversized' else shape+(2,) if geometry=='channels2' else shape
    mask_dtype=np.dtype(spec.get('mask_dtype','u1'))
    image_bytes=count*dtype.itemsize*(2 if layout=='step' and kind=='view' else 1)
    mask_bytes=count*mask_dtype.itemsize*(2 if layout=='step' and kind=='mask' else 1)
    output_bytes=math.prod(out_shape)*output.itemsize*(2 if spec.get('error')=='step-out' else 1)
    offset=output.itemsize if kind=='offset' else ((image_bytes+7)//8)*8+16 if kind=='disjoint' else 0
    if 'output_offset' in spec:offset=spec['output_offset']*output.itemsize
    owner=np.full(max(image_bytes,mask_bytes,offset+output_bytes)+32,0x5a,np.uint8)
    image=shaped_view(owner,shape,dtype,layout=layout) if kind!='mask' else np.empty(shape,dtype)
    out=shaped_view(owner,out_shape,output,offset,layout='F' if spec.get('error')=='F-out' else 'step' if spec.get('error')=='step-out' else 'C')
    options={};mask=None
    if kind in ('mask','both'):
        mask=shaped_view(owner,shape,mask_dtype,layout=layout if kind=='mask' else 'C')
        mask[...] = 0 if spec.get('mask_state')=='zero' else (np.arange(count).reshape(shape)%3!=0)
        if spec.get('mask_state')=='first-three':
            mask.fill(0);mask.flat[0]=mask.flat[1]=mask.flat[shape[-1]]=1
    elif spec.get('mask_state'):
        mask=np.zeros(shape,np.uint8) if spec['mask_state']=='zero' else np.ones(shape,np.uint8)
        if spec['mask_state']=='checker':mask.flat[::3]=0
    image[...] = ((np.arange(count)*7+3)%23).reshape(shape)
    active_image=(kind!='mask' and layout=='C' and dtype.isnative)
    zero_invariant=spec.get('footprint')=='zero' or spec.get('mask_state')=='zero'
    full_bins=bool(dtype==np.uint16 and active_image and output.kind in 'ub' and output.itemsize!=2 and kind!='disjoint' and not zero_invariant)
    if full_bins:image.flat[-1]=65535
    fp=np.ones((3,)*ndim,np.uint8);variant=spec.get('footprint','dense')
    if variant=='point':fp=np.ones((1,)*ndim,np.uint8)
    elif variant=='wide':fp=np.ones((3,7) if ndim==2 else (3,3,7),np.uint8)
    elif variant=='disconnected':fp.fill(0);fp.flat[0]=fp.flat[-1]=1
    elif variant=='zero':fp.fill(0)
    elif variant=='even-shifted':
        fp=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8)
        options.update(shift_x=-1,shift_y=0 if ndim==2 else 1,shift_z=0)
    if spec.get('shift')=='bool':options['shift_x']=True
    elif spec.get('shift')=='array1':options['shift_x']=np.asarray([0])
    error=spec.get('error')
    if error=='readonly-image':image.flags.writeable=False
    elif error=='readonly-mask':mask.flags.writeable=False
    elif error=='readonly-out':out.flags.writeable=False
    elif error=='bad-center':options['shift_x']=7
    elif error=='bad-shift':options['shift_x']='bad'
    elif error=='same-object':out=image
    if spec.get('mask_identity'):out=mask
    options['out']=out
    if mask is not None:options['mask']=mask
    tracked=dict(image=image,footprint=fp,out=out,owner=owner)
    if mask is not None:tracked['mask']=mask
    # Each view has a real initialized allocation; no fabricated as_strided,
    # undersized buffer, zero-depth destination or unaligned wide pointer.
    lo,hi=byte_extent(owner)
    for key,array in tracked.items():
        assert array.size>0 and array.flags.aligned,(key,spec)
        if np.shares_memory(array,owner):
            a,b=byte_extent(array);assert lo<=a<=b<=hi,(key,spec,a,b,lo,hi)
    assert all(out.shape[i]>=image.shape[i] for i in range(ndim))
    if mask is not None:assert mask.size>=image.size and mask.ndim==ndim
    return (image,fp),options,tracked


def describe(array,arrays):
    return dict(dtype=array.dtype.str,metadata=repr(array.dtype.metadata),shape=list(array.shape),strides=list(array.strides),
        c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),aligned=bool(array.flags.aligned),owned=bool(array.flags.owndata),writeable=bool(array.flags.writeable),
        bytes_sha256=hashlib.sha256(array.tobytes()).hexdigest(),identity=[array is a for a in arrays],alias=[bool(np.shares_memory(array,a)) for a in arrays])


def observe(function,spec):
    # Every original invocation is preceded by an independently bounded replay
    # on a separate owner, including actual histogram-index admission checks.
    replay(function.__name__,spec)
    args,options,tracked=make_case(spec);arrays=list(tracked.values())
    before={key:describe(value,arrays) for key,value in tracked.items()}
    pointers={key:value.ctypes.data for key,value in tracked.items()}
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        try:result,error=describe(function(*args,**options),arrays),None
        except Exception as caught:result,error=None,dict(type=type(caught).__name__,message=str(caught))
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in messages],
        before=before,after={key:describe(value,arrays) for key,value in tracked.items()},pointer_stable={key:value.ctypes.data==pointers[key] for key,value in tracked.items()})


def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


_LOG_HANDLES=[]
@lru_cache(maxsize=1)
def source_log():
    if os.name=='nt':
        library=ctypes.CDLL('ucrtbase.dll');_LOG_HANDLES.append(library)
        function=library.log;function.argtypes=[ctypes.c_double];function.restype=ctypes.c_double
        return function
    library=ctypes.CDLL(ctypes.util.find_library('m'));process=ctypes.CDLL(None)
    _LOG_HANDLES.extend((library,process))
    if hasattr(process,'dlvsym'):
        process.dlvsym.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p];process.dlvsym.restype=ctypes.c_void_p
        address=process.dlvsym(library._handle,b'log',b'GLIBC_2.2.5')
        assert address,'pinned Linux original log symbol unavailable'
        return ctypes.CFUNCTYPE(ctypes.c_double,ctypes.c_double)(address)
    function=library.log;function.argtypes=[ctypes.c_double];function.restype=ctypes.c_double
    return function


def normalized_inputs(image,mask):
    if image.dtype not in (np.uint8,np.uint16):
        assert image.dtype.kind=='u' and image.itemsize==2 and image.max()<256
        image=image.astype(np.uint8)
    image=np.ascontiguousarray(image)
    if mask is not None:
        if mask.dtype.kind=='b':mask=mask.astype(np.uint8)*np.uint8(255)
        elif mask.dtype!=np.uint8:
            assert mask.dtype.kind=='u' and mask.itemsize==2 and mask.max()<256
            mask=mask.astype(np.uint8)
        mask=np.ascontiguousarray(mask)
    return image,mask


def replay(name,spec):return _replay(name,json.dumps(spec,sort_keys=True))


@lru_cache(maxsize=None)
def _replay(name,encoded):
    spec=json.loads(encoded);args,options,tracked=make_case(spec)
    early=bool(spec.get('error')) or spec.get('geometry')=='channels2' and spec['ndim']==3
    if spec.get('shift')=='array1':
        # Pinned NumPy1.24 permits this; NumPy2.5 raises before the Cython
        # kernel. Probe only the scalar protocol, never an image operation.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            try:int(np.asarray([0]))
            except TypeError:early=True
    if early:return dict(kernel=False,population_min=None,population_max=None,negative_population=[],negative_histograms=[],no_store=[],after=None)
    image,mask=normalized_inputs(args[0],options.get('mask'));out=options['out'];owner=tracked['owner'];ndim=image.ndim
    fp=args[1];shifts=[options.get('shift_'+axis,0) for axis in ('x','y','z')[:ndim]]
    shift_values=[int(np.asarray(value).reshape(-1)[0]) for value in shifts]
    centers=tuple(size//2+shift for size,shift in zip(fp.shape,shift_values[::-1] if ndim==2 else shift_values))
    active=[tuple(int(i)-c for i,c in zip(index,centers)) for index in np.ndindex(fp.shape) if fp[index]]
    active_set=set(active)
    enter={axis:[index for index in active if tuple(x+(1 if a==axis else 0) for a,x in enumerate(index)) not in active_set] for axis in range(ndim)}
    leave={axis:[index for index in active if tuple(x-(1 if a==axis else 0) for a,x in enumerate(index)) not in active_set] for axis in range(ndim)}
    bins=256 if image.dtype==np.uint8 else max(3,int(image.max()))+1
    feedback_image=np.shares_memory(image,out);feedback_mask=mask is not None and np.shares_memory(mask,out)
    if feedback_image and image.dtype==np.uint16:
        assert out.dtype.kind in 'ub','u16/float effective-type overlap excluded'
        assert out.dtype==np.uint16 or bins==65536 or not active or mask is not None and not np.any(mask),'mixed u16 byte feedback requires full histogram domain or zero invariant'
    if name=='entropy' and out.dtype.kind in 'ub':
        assert not (feedback_image or feedback_mask) or not active or mask is not None and not np.any(mask),'integral entropy feedback without zero/copy invariant excluded'
    counts={};pop=0;populations=[];negative_hist=[];no_store=[];emitted=[];probabilities=set();entropy_classes=[]
    def update(position,indices,sign,adjust=None):
        nonlocal pop
        adjust=(0,)*ndim if adjust is None else adjust
        for relative in indices:
            point=tuple(p+r+a for p,r,a in zip(position,relative,adjust))
            if any(i<0 or i>=size for i,size in zip(point,image.shape)):continue
            if mask is not None and not mask[point]:continue
            value=int(image[point]);assert 0<=value<bins,('unsafe replay histogram escape',name,spec,point,value,bins)
            count=counts.get(value,0)+sign
            if count:counts[value]=count
            else:counts.pop(value,None)
            pop+=sign
    def write(position):
        populations.append(pop)
        if any(count<0 for count in counts.values()):negative_hist.append(list(position))
        assert sum(counts.values())==pop
        store=True;value=0
        if pop:
            occupied=sorted(counts)
            if name=='gradient':value=occupied[-1]-occupied[0]
            elif name=='median':
                remaining=pop/2.;store=False
                for index in occupied:
                    remaining-=counts[index]
                    if remaining<0:value=index;store=True;break
            elif name=='modal':
                maximum=0
                for index in occupied:
                    if counts[index]>maximum:maximum=counts[index];value=index
            else:
                value=0.
                for index in occupied:
                    count=counts[index]
                    probability=count/float(pop)
                    if probability>0:
                        divisor=math.gcd(count,pop);probabilities.add((abs(count)//divisor,abs(pop)//divisor))
                        value-=probability*source_log()(probability)/0.6931471805599453
        if name=='entropy':
            entropy_classes.append('nan' if math.isnan(value) else 'positive-inf' if value==math.inf else 'negative-inf' if value==-math.inf else 'negative' if value<0 else 'zero' if value==0 else 'positive')
        if not store:no_store.append(list(position));return
        if out.dtype.kind in 'ub':
            if name=='entropy':assert math.isfinite(value) and 0<=value<256
            data=(int(value)&((1<<(out.itemsize*8))-1)).to_bytes(out.itemsize,byteorder='little' if np.little_endian else 'big')
        else:data=struct.pack('=f' if out.itemsize==4 else '=d',float(value))
        byte_offset=int(out.ctypes.data)-int(owner.ctypes.data)+sum(i*stride for i,stride in zip(position,out.strides))
        assert 0<=byte_offset<=owner.nbytes-out.itemsize
        owner[byte_offset:byte_offset+out.itemsize]=np.frombuffer(data,np.uint8)
        emitted.append(data.hex())
    planes=range(image.shape[0]) if ndim==3 else (None,)
    for plane in planes:
        counts={};pop=0
        position=(0,0) if ndim==2 else (plane,0,0)
        update(position,active,1);write(position)
        for row in range(image.shape[-2]):
            if row:
                position=(row,0 if row%2==0 else image.shape[-1]-1) if ndim==2 else (plane,row,0 if row%2==0 else image.shape[-1]-1)
                update(position,enter[ndim-2],1)
                adjust=tuple(-1 if axis==ndim-2 else 0 for axis in range(ndim))
                update(position,leave[ndim-2],-1,adjust);write(position)
            columns=range(1,image.shape[-1]) if row%2==0 else range(image.shape[-1]-2,-1,-1)
            for column in columns:
                position=(row,column) if ndim==2 else (plane,row,column)
                if row%2==0:
                    update(position,enter[ndim-1],1)
                    update(position,leave[ndim-1],-1,tuple(-1 if axis==ndim-1 else 0 for axis in range(ndim)))
                else:
                    update(position,leave[ndim-1],1)
                    update(position,enter[ndim-1],-1,tuple(1 if axis==ndim-1 else 0 for axis in range(ndim)))
                write(position)
    arrays=list(tracked.values())
    return dict(kernel=True,bins=bins,feedback_image=feedback_image,feedback_mask=feedback_mask,
        population_min=min(populations),population_max=max(populations),negative_population=[i for i,population in enumerate(populations) if population<0],
        negative_histograms=negative_hist,no_store=no_store,stores=len(emitted),emitted_sha256=hashlib.sha256(''.join(emitted).encode()).hexdigest(),
        probability_pairs=[list(pair) for pair in sorted(probabilities)],entropy_classes=entropy_classes,
        after={key:describe(value,arrays) for key,value in tracked.items()})


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text())


@pytest.mark.parametrize('case',CASES,ids=[row[0] for row in CASES])
def test_original_defined_alias_contract(case,frozen):
    key,name,spec=case
    assert skimage.__version__=='0.26.0'
    expected=frozen['observations'][frozen['snapshots'][profile()][key]]
    assert observe(getattr(reference,name),spec)==expected,key


@pytest.mark.parametrize('case',CASES,ids=[row[0] for row in CASES])
def test_independent_source_order_and_preserved_owner_state(case,frozen):
    key,name,spec=case;audit=replay(name,spec)
    original=frozen['observations'][frozen['snapshots'][profile()][key]]
    if audit['kernel']:assert audit['after']==original['after'],key
    else:assert original['error'] is not None and original['before']==original['after'],key


def test_source_only_scope_integrity(frozen):
    assert frozen['native_checks']==0 and frozen['native_pending']==len(CASES)
    for snapshot in frozen['snapshots'].values():assert set(snapshot)=={row[0] for row in CASES}
    for digest,value in frozen['observations'].items():assert hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()==digest
    for name in NAMES:
        assert hashlib.sha256(inspect.getsource(getattr(reference,name)).encode()).hexdigest()==frozen['functions'][name]['python_source_sha256']


@pytest.mark.parametrize('name',NAMES)
def test_source_signatures_and_generic_identity(name,frozen):
    from skimage.filters.rank import generic
    assert getattr(generic,name) is getattr(reference,name)
    assert str(inspect.signature(getattr(reference,name)))==frozen['functions'][name]['signature']


def test_negative_population_and_median_no_store_are_explicit(frozen):
    audits=frozen['audits'][profile()]
    for ndim in (2,3):
        median=audits[f'median/population-drift/{ndim}d']
        entropy=audits[f'entropy/population-drift/{ndim}d']
        assert median['population_min']<0 and len(median['no_store'])==8
        assert entropy['population_min']<0
    assert frozen['interface_proposal']['median_no_store']=='Preserve every existing destination byte when pop!=0 and the ascending nonzero-bin CDF never becomes negative.'
