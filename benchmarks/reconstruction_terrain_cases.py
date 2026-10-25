"""Prospective terrain supplement; no reconstruction, native admission or clocks.

Exact original48 functions are read-only interfaces. The byte-pinned certificate
is compiled unchanged into this separate case namespace; no historical global
is replaced and no original image algorithm is imported during construction.
"""
import ast
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/'benchmarks/reconstruction_cases.py'
BASE_SHA='f323f5c3d19a92e77620845d98a622b2b322441f8c02fedb92a80db5dded704a'
assert hashlib.sha256(BASE_PATH.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==BASE_SHA
_spec=importlib.util.spec_from_file_location('immutable_reconstruction_48',BASE_PATH)
BASE=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(BASE)
WORKSPACE_CAP=BASE.WORKSPACE_CAP
PRIMARY_SOURCE_PINS=BASE.PRIMARY_SOURCE_PINS
REFERENCE_MATRIX_SHA256=BASE.REFERENCE_MATRIX_SHA256
array_spec=BASE.array_spec
contract_hash=BASE.contract_hash


def specifications():
    rows=[]
    for group,dtype,method in itertools.product(('multiple-markers','plateau-barrier-corridor'),('u1','u2'),('dilation','erosion')):
        rows.append(dict(id='/'.join((group,dtype,method)),group=group,dtype=dtype,method=method,
            shape=[129,131] if group=='multiple-markers' else [17,19,21],
            layout=('C' if dtype=='u1' else 'step') if group=='multiple-markers' else ('F' if dtype=='u1' else 'step'),
            footprint='dense' if group=='multiple-markers' else 'axial',radius=1,
            intended_future_measurement=True,native_credit='pending-qualified-public',original_policy='all-warn',expected_error=False))
    for dtype,method,rank in itertools.product(('f4','f8'),('dilation','erosion'),(2,3)):
        rows.append(dict(id='/'.join(('high-cardinality',dtype,method,str(rank)+'d')),group='high-cardinality',dtype=dtype,method=method,
            shape=[257,259] if rank==2 else [31,33,35],layout=('C' if dtype=='f4' else 'F') if rank==2 else ('step' if dtype=='f4' else 'C'),
            footprint='dense' if rank==2 else 'axial',radius=1,
            intended_future_measurement=True,native_credit='pending-qualified-public',original_policy='all-warn',expected_error=False))
    assert len(rows)==16 and len({r['id'] for r in rows})==16
    return rows


def topology(spec):
    shape=spec['shape'];rank=len(shape)
    if spec['group']=='high-cardinality':
        return dict(markers=[tuple(max(1,s*f//4) for s in shape) for f in (1,2,3)],witness=None,cut_axis=None)
    left=(shape[0]//4,)+(tuple(s//2 for s in shape[1:]));right=(3*shape[0]//4,)+tuple(s//2 for s in shape[1:])
    witness=(3*shape[0]//4,)+tuple(s//4 for s in shape[1:])
    return dict(markers=[left,right],witness=witness,cut_axis=0,cut_index=shape[0]//2,
        corridor=tuple(s//2 for s in shape[1:]),right_level=112 if rank==2 else 96,
        cut_level=24 if rank==2 else 16,corridor_level=48 if rank==2 else 64)


def fixture(spec):
    assert spec in specifications()
    shape=tuple(spec['shape']);dtype=np.dtype(spec['dtype']);t=topology(spec);n=math.prod(shape)
    if spec['group']=='high-cardinality':
        # A serpentine bijection of1..N, represented exactly as dyadic floats.
        values=np.arange(1,n+1,dtype=np.uint64).reshape(shape)
        values[...,1::2,:]=values[...,1::2,::-1]
        if len(shape)==3:values[1::2]=values[1::2,::-1,::-1]
        denominator=1<<(n+1).bit_length()
        mask=(values.astype(dtype)/dtype.type(denominator));seed=np.zeros(shape,dtype);peak=dtype.type(1)
        for point in t['markers']:seed[point]=mask[point]
    else:
        code=np.full(shape,160 if len(shape)==2 else 176,dtype=np.uint16)
        cut=shape[0]//2
        if len(shape)==2:code[cut+1:]=144;code[1:4,1:4]=64
        code[cut]=t['cut_level'];code[(cut,)+t['corridor']]=t['corridor_level']
        left,right=t['markers'];code[left]=192;code[right]=t['right_level']
        scale=257 if dtype.itemsize==2 else 1
        mask=(code*scale).astype(dtype);seed=np.zeros(shape,dtype);peak=dtype.type(65535 if dtype.itemsize==2 else 255)
        seed[left]=dtype.type(192*scale);seed[right]=dtype.type(t['right_level']*scale)
    if spec['method']=='erosion':seed=peak-seed;mask=peak-mask
    if spec['layout']=='F':seed=np.asfortranarray(seed);mask=np.asfortranarray(mask)
    else:seed=BASE._layout(seed,spec['layout']);mask=BASE._layout(mask,spec['layout'])
    rank=len(shape);fp=np.ones((3,)*rank,bool) if spec['footprint']=='dense' else np.zeros((3,)*rank,bool)
    if spec['footprint']=='axial':
        for axis in range(rank):
            for step in range(3):point=[1]*rank;point[axis]=step;fp[tuple(point)]=True
    return seed,mask,dict(method=spec['method'],footprint=fp)


# Preserve the complete audited bounds/ownership/pad/index/workspace arithmetic.
_node=next(n for n in ast.parse(BASE_PATH.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='certificate')
exec(compile(ast.Module(body=[_node],type_ignores=[]),'<exact-immutable48-certificate>', 'exec'),globals())


def terrain_certificate(spec,seed,mask):
    t=topology(spec);peak=np.dtype(spec['dtype']).type(65535 if spec['dtype']=='u2' else 255 if spec['dtype']=='u1' else 1)
    # Convert erosion by an exact integer/dyadic complement for structural checks.
    a=seed if spec['method']=='dilation' else peak-seed
    b=mask if spec['method']=='dilation' else peak-mask
    assert np.all(b>0) and np.all(a>=0)
    nonzero=np.flatnonzero(a);points=[np.ravel_multi_index(p,a.shape) for p in t['markers']]
    assert set(map(int,nonzero))==set(map(int,points)) and len(points)>=2
    assert all(a[p]==b[p] for p in t['markers'])
    levels=int(np.unique(b).size)
    if spec['group']=='high-cardinality':
        assert levels==b.size and b.max()>a.max() and len(np.unique(a[tuple(np.asarray(t['markers']).T)]))==3
        witness='global initial maximum strictly below terrain maximum: reconstruction cannot equal mask'
    else:
        cut=b[t['cut_index']];scale=257 if spec['dtype']=='u2' else 1
        assert np.count_nonzero(cut==t['corridor_level']*scale)==1 and np.all(cut<=t['corridor_level']*scale)
        assert b[t['witness']]>a[t['markers'][1]]>cut.max()
        witness=dict(point=list(t['witness']),expected_output=float(a[t['markers'][1]]) if spec['method']=='dilation' else float(peak-a[t['markers'][1]]),proof='entire cut-plane bottleneck below right marker; uniform connected right plateau reaches exactly its marker level')
    return dict(marker_count=len(points),mask_unique_values=levels,seed_unique_values=int(np.unique(a).size),positive_connected_mask=True,
        expected_changed_elements=int(seed.size-len(points)),expected_result_differs_from_mask=True,nontrivial_output_witness=witness,
        topology={k:[list(p) for p in v] if k=='markers' else list(v) if isinstance(v,tuple) else v for k,v in t.items()})


def case_manifest():
    rows=[]
    for spec in specifications():
        seed,mask,kw=fixture(spec)
        rows.append(dict(spec=spec,seed=array_spec(seed),mask=array_spec(mask),footprint=array_spec(kw['footprint']),seed_is_mask=False,
            certificate=certificate(spec,seed,mask,kw),terrain=terrain_certificate(spec,seed,mask)))
    return rows


def native_admission(*args,**kwargs):
    raise RuntimeError('terrain native admission pending genuine qualified public200; original-only contract is not native acceptance')
