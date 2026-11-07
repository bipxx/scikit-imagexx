"""Bounded prospective source recipes; original/native/capture budget is zero."""
from copy import deepcopy
from math import ceil,prod
API='restoration.rolling_ball'
SIGNATURE='(image, *, radius=100, kernel=None, nansafe=False, num_threads=<DEPRECATED>, workers=None)'
MAX_INPUT=64
MAX_KERNEL=201
MAX_PADDED=512
MAX_ADDITIONS=16384
MAX_NUMERICAL_BYTES=131072

def recipes():
    rows=[]
    def add(name,**updates):
        r=dict(id=name,shape=[3,4],dtype='float32',layout='C',pattern='ramp',radius=1,kernel=None,nansafe=False,options={},metadata=False,readonly=False,provider=None)
        r.update(updates);rows.append(r)
    for dtype in ('bool','uint8','uint16','uint32','uint64','int8','int16','int32','int64','float16','float32','float64'):
        add('dtype-'+dtype,dtype=dtype)
    for layout in ('F','reverse','stride2','transpose'):
        add('layout-'+layout,layout=layout)
    add('readonly',readonly=True)
    add('metadata',metadata=True)
    add('swapped-f32',dtype='>f4')
    add('shape-1d',shape=[7])
    add('shape-3d',shape=[2,3,4])
    add('shape-4d',shape=[2,2,2,2])
    add('shape-8d-radius0',shape=[1]*8,radius=0)
    add('empty-1d',shape=[0])
    add('empty-axis',shape=[0,3])
    add('singleton',shape=[1,1])
    add('default-radius-1d-only',shape=[3],radius='OMITTED')
    for radius in (0,.5,1.5,-.25,-1,'nan','inf','bad'):
        add('radius-'+str(radius),radius=radius,dtype='float64')
    for pattern in ('zeros','bright','step','signedzero','nan','allnan','inf','neginf'):
        add('values-'+pattern,pattern=pattern)
    add('nan-safe',pattern='nan',nansafe=True)
    add('allnan-safe',pattern='allnan',nansafe=True)
    for kernel in ('flat','even','asymmetric','wide','inf','nan','allinf','readonly','metadata','list','provider'):
        add('kernel-'+kernel,kernel=kernel)
    add('kernel-ignores-radius',kernel='flat',radius='bad')
    for workers in (None,0,1,2,4,'bad'):
        add('workers-'+str(workers),options={'workers':workers})
    for value in (None,0,1,2):
        add('deprecated-'+str(value),options={'num_threads':value})
    add('both-threads-conflict',options={'num_threads':1,'workers':2})
    add('both-threads-image-protocol',provider='raise',options={'num_threads':1,'workers':None})
    add('image-array-protocol',provider='array')
    add('image-refuses',provider='raise')
    add('image-list',provider='list')
    add('nansafe-truth',nansafe='probe')
    add('complex64-error',dtype='complex64')
    add('complex128-error',dtype='complex128')
    return deepcopy(rows)

def source_safe(spec):
    assert type(spec)is dict and spec in recipes(),'RECIPE_NOT_FROZEN'
    shape=spec['shape'];assert 1<=len(shape)<=8 and all(type(x)is int and 0<=x<=8 for x in shape)
    n=prod(shape);assert n<=MAX_INPUT
    kind=spec['kernel'];ndim=len(shape)
    if kind is not None:
        assert ndim==2
        ks={'even':[2,4],'wide':[5,7]}.get(kind,[3,3])
    elif spec['radius']in('nan','inf','bad')or spec['radius']==-1:
        # arange/ceil or empty-center indexing fails before unchecked Cython.
        ks=[1]*ndim
    else:
        radius=100 if spec['radius']=='OMITTED'else spec['radius']
        ks=[max(0,2*ceil(radius)+1)]*ndim
    k=prod(ks);padded=prod(s+2*(k//2)for s,k in zip(shape,ks));physical=max(1,n)*(2**len(shape) if spec['layout']=='stride2'else 1)
    additions=n*k
    # Conservative simultaneous named-numerical-array allowance, including
    # float64 kernel coordinate grids, all casts, padded storage/output, and
    # index/shape vectors. Excludes Python/import/allocator/process overhead.
    byte_upper=8*(4*physical+4*n+2*padded+(6*ndim+16)*k+32*ndim)
    assert k<=MAX_KERNEL and padded<=MAX_PADDED and additions<=MAX_ADDITIONS and byte_upper<=MAX_NUMERICAL_BYTES
    assert spec['options'].get('workers')in(None,0,1,2,4,'bad')
    return dict(input_elements=n,physical_elements=physical,kernel_shape=ks,kernel_elements=k,padded_elements=padded,addition_upper=additions,named_numerical_bytes_upper=byte_upper,workers_max=4,default_OpenMP_env_required=1)

def fixture(spec,np=None):
    source_safe(spec)
    # Explicit import request remains interceptable even if numpy is cached.
    if np is None:import numpy as np
    from crop_contract import ArrayProvider,BoolProbe
    events=[];shape=spec['shape'];n=prod(shape);dtype=np.dtype(spec['dtype'])
    if spec['metadata']:dtype=np.dtype(dtype,metadata={'units':'source-first','key':'rolling-ball'})
    values=np.arange(n,dtype=np.float64)%7
    a=values.astype(dtype).reshape(shape)
    if spec['pattern']=='zeros':a[...] = 0
    elif spec['pattern']=='bright':a[...] = 7
    elif spec['pattern']=='step':a.flat[:n//2]=0
    elif spec['pattern']=='signedzero':a.flat[0]=-0.0
    elif spec['pattern']=='nan':a.flat[n//2]=np.nan
    elif spec['pattern']=='allnan':a[...] = np.nan
    elif spec['pattern']=='inf':a.flat[n//2]=np.inf
    elif spec['pattern']=='neginf':a.flat[n//2]=-np.inf
    layout=spec['layout']
    if layout=='F':a=np.array(a,order='F')
    elif layout=='reverse':a=a[(slice(None,None,-1),)*a.ndim]
    elif layout=='transpose':a=a.T
    elif layout=='stride2':
        backing=np.zeros(tuple(2*s for s in shape),dtype=dtype);view=backing[(slice(None,None,2),)*len(shape)];view[...] = a;a=view
    if spec['readonly']:a.flags.writeable=False
    buffers={'image':a};image=a
    if spec['provider']in('array','raise'):image=ArrayProvider(a,events,error=spec['provider']=='raise')
    elif spec['provider']=='list':image=a.tolist()
    options=dict(spec['options']);radius=spec['radius']
    if radius!='OMITTED':options['radius']={'nan':float('nan'),'inf':float('inf')}.get(radius,radius)
    safe=spec['nansafe'];options['nansafe']=BoolProbe(True,events)if safe=='probe'else safe
    kind=spec['kernel']
    if kind is not None:
        ks=source_safe(spec)['kernel_shape'];kernel=np.zeros(ks,dtype=np.float64)
        if kind=='asymmetric':kernel[...] = np.arange(prod(ks)).reshape(ks)%3
        elif kind=='inf':kernel.flat[0]=np.inf
        elif kind=='nan':kernel.flat[0]=np.nan
        elif kind=='allinf':kernel[...] = np.inf
        if kind=='readonly':kernel.flags.writeable=False
        if kind=='metadata':kernel=kernel.astype(np.dtype('float64',metadata={'kernel':'typed'}))
        buffers['kernel']=kernel
        options['kernel']=kernel.tolist()if kind=='list'else ArrayProvider(kernel,events)if kind=='provider'else kernel
    return (image,),options,events,buffers

def reference_runtime(binding=None):
    raise SystemExit('ROLLING_BALL_PROSPECTIVE_UNBOUND: source safety/readiness and separate root GO required; original/native budget0')
