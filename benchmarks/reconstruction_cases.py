"""Deterministic reconstruction case contract; no controller or timer.

Construction is explicit. Import does not allocate fixtures, import skimagexx,
execute a reconstruction, load a product, or admit a native implementation.
"""
import hashlib
import itertools
import json
import math
import numpy as np

SCHEMA='reconstruction-benchmark-cases-v1'
WORKSPACE_CAP=256*1024*1024
DTYPES=('u1','u2','f4','f8')
METHODS=('dilation','erosion')
REFERENCE_COMMIT='600a403c1368b99e5d85200bc4f6eb6ed262ea0c'
REFERENCE_MATRIX_SHA256='621981325f5e6939d3fb4d1c94e52910b369ef3ca798c326bdb93322a12b504c'
PRIMARY_SOURCE_PINS={
 'grayreconstruct.py':'f87c127d396a7b6f58a7e2109b184d41d7db26d8021cca22c63ddfca7157114d',
 'grayreconstruct.pyx':'b77fed8bbd23c49215c21f474e4061ff181286d9bdf2ebd6b3ecf0c9ea103df9',
 'rank_order.py':'3546e2e6c3ed2935be5e6e9565cc27d05e69578e727da60378730426d1b2e279',
 'shared_utils.py':'b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1',
 'LICENSE.txt':'611d3207504dcb4a808df209d2e7c9b4f2e89fe690f84735c0d548160966b527'}


def specifications():
    rows=[]
    def add(group,dtype,method,shape,layout='C',footprint='dense',radius=1):
        rows.append(dict(id='/'.join((group,dtype,method,'x'.join(map(str,shape)),layout,footprint+'-r'+str(radius))),
            group=group,dtype=dtype,method=method,shape=list(shape),layout=layout,footprint=footprint,radius=radius,
            intended_future_measurement=group!='oracle',native_credit='explicit-NumPy-zero-representative-provider-plus-C++flood' if group=='signedzero' else 'pending-qualified-public',
            original_policy='all-warn',expected_error=False))
    for dtype,method,rank in itertools.product(DTYPES,METHODS,(2,3)):
        add('flood',dtype,method,(257,259) if rank==2 else (31,33,35))
    for dtype,method in itertools.product(DTYPES,METHODS):
        add('large-flood',dtype,method,(1024,1025))
        add('stepped-flood',dtype,method,(257,259),'step')
    for dtype,method,rank in itertools.product(('u2','f4'),METHODS,(2,3)):
        add('radius2-flood',dtype,method,(129,131) if rank==2 else (23,25,27),footprint='axial',radius=2)
    for i,dtype in enumerate(DTYPES):add('identity',dtype,METHODS[i%2],(257,259))
    for dtype,method in itertools.product(('f4','f8'),METHODS):add('signedzero',dtype,method,(65,67))
    for dtype,method,rank in itertools.product(DTYPES,METHODS,(2,3)):
        add('oracle',dtype,method,(5,7) if rank==2 else (4,5,6),footprint='axial')
    assert len(rows)==64 and sum(s['intended_future_measurement'] for s in rows)==48
    assert len({s['id'] for s in rows})==len(rows)
    return rows


def native_admission(*args,**kwargs):
    raise RuntimeError('native reconstruction benchmark admission pending genuine public200 product certificate; no private adapter accepted')


def _layout(array,kind):
    if kind=='C':return np.ascontiguousarray(array)
    if kind!='step':raise ValueError('undeclared storage layout')
    owner=np.zeros(tuple(2*s+1 for s in array.shape),array.dtype)
    result=owner[tuple(slice(1,2*s+1,2) for s in array.shape)];result[...]=array
    return result


def fixture(spec):
    # Exact declared specs only. Arbitrary shapes/dtypes/providers do not acquire
    # a source-safety certificate by resembling a supported measurement case.
    assert spec in specifications()
    shape=tuple(spec['shape']);dtype=np.dtype(spec['dtype']);n=math.prod(shape)
    indices=np.arange(n,dtype=np.uint64).reshape(shape)
    code=((indices*37+indices//shape[-1]*19)%251+1)
    if dtype.kind=='u':
        scale=257 if dtype.itemsize==2 else 1
        mask=(code*scale).astype(dtype);peak=dtype.type(65535 if scale==257 else 255)
    else:mask=(code.astype(dtype)/dtype.type(256));peak=dtype.type(1)
    seed=np.zeros(shape,dtype);center=tuple(s//2 for s in shape)
    seed[center]=peak;mask[center]=peak
    if spec['method']=='erosion':seed=peak-seed;mask=peak-mask
    if spec['group']=='identity':seed=mask
    elif spec['group']=='signedzero':
        seed=np.zeros(shape,dtype);seed.flat[::2]=-0.
        mask=np.zeros(shape,dtype);mask.flat[1::2]=-0.
        if spec['method']=='erosion':seed,mask=mask,seed
    seed=_layout(seed,spec['layout']);mask=seed if spec['group']=='identity' else _layout(mask,spec['layout'])
    rank=len(shape);radius=spec['radius'];fshape=(2*radius+1,)*rank
    if spec['footprint']=='dense':footprint=np.ones(fshape,bool)
    else:
        footprint=np.zeros(fshape,bool);center=(radius,)*rank
        for axis in range(rank):
            for offset in range(-radius,radius+1):
                point=list(center);point[axis]+=offset;footprint[tuple(point)]=True
    return seed,mask,dict(method=spec['method'],footprint=footprint)


def array_spec(a):
    owner=a
    while isinstance(owner.base,np.ndarray):owner=owner.base
    return dict(dtype=a.dtype.str,shape=list(a.shape),strides=list(a.strides),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),
        writable=bool(a.flags.writeable),own=bool(a.flags.owndata),owner_bytes=int(owner.nbytes),raw_bytes=int(a.nbytes),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def certificate(spec,seed,mask,options):
    assert spec in specifications() and type(seed) is type(mask) is np.ndarray
    assert seed.shape==mask.shape==tuple(spec['shape']) and seed.dtype==mask.dtype==np.dtype(spec['dtype'])
    assert 2<=seed.ndim<=3 and all(s>=4 for s in seed.shape)
    assert np.isfinite(seed).all() and np.isfinite(mask).all()
    assert options['method']==spec['method']
    assert np.all(seed<=mask) if spec['method']=='dilation' else np.all(seed>=mask)
    fp=options['footprint'];radius=spec['radius'];assert type(fp) is np.ndarray and fp.dtype==np.bool_ and fp.shape==(2*radius+1,)*seed.ndim
    assert fp[(radius,)*seed.ndim] and not any(s==0 for s in fp.shape)
    padded=tuple(s+f-1 for s,f in zip(seed.shape,fp.shape));plane=math.prod(padded)
    assert 2*plane < 2**31 and all(s<2**31 for s in padded)
    inputs=array_spec(seed)['owner_bytes']+(0 if seed is mask else array_spec(mask)['owner_bytes'])
    # Official Python/rank_order live-array accounting <=144 bytes/padded site
    # for 2 planes, int32 links/ranks, intp argsort, sorted values/booleans and
    # result materialization. Reserve another56/site for bounded certificate,
    # raw serialization/compression copies and allocation lifetime overlap.
    original=200*plane+inputs+16384+12*fp.size
    # Frozen reconstruction-v1 buffers: out W*P, mask W*N; size_t arrays
    # from_padded P +to_padded/queue/position3*N; heads <=65536, neighbors.
    width=4 if seed.dtype==np.float32 else 8
    heads=256 if seed.dtype==np.uint8 else 65536 if seed.dtype==np.uint16 else 33 if width==4 else 65
    native=(width+8)*plane+(width+24)*seed.size+8*heads+8*fp.size+inputs+16384
    if spec['group']=='signedzero':native+=80*plane # explicit NumPy representative workspace
    native+=64*plane # instrumentation/materialization margin, not kernel storage
    memory=max(original,native)
    assert memory<=WORKSPACE_CAP
    return dict(schema='source-reconstruction-entry-certificate-v1',case_id=spec['id'],padded_shape=list(padded),plane=plane,
        source_index_dtype='int32',original_workspace_bound_bytes=original,native_v1_planning_bound_bytes=native,workspace_bound_bytes=memory,workspace_cap_bytes=WORKSPACE_CAP,
        scope='Per-call data/workspace increment bound, not an absolute process RSS promise; runtime import/allocator baseline separate',
        proof='Finite same-shape supported plain arrays, seed/mask order, positive bounded spatial dimensions, centered odd finite footprint and padded rank0 halo. Pre-Cython prepared-array certificate is mandatory too.')


def case_manifest():
    rows=[]
    for spec in specifications():
        seed,mask,options=fixture(spec)
        rows.append(dict(spec=spec,seed=array_spec(seed),mask=array_spec(mask),footprint=array_spec(options['footprint']),
            seed_is_mask=seed is mask,certificate=certificate(spec,seed,mask,options)))
    return rows


def contract_hash(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
