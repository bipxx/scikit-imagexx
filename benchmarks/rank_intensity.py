"""Fixed canonical rank calls, untimed aliases, exact artifacts and paired samples.

Local runs require --check-only. Reference-only checks do not exercise native
paths. Measurements require strict public registration and an artifact-fed
Actions host after all preparation profiles finish; no local timing is allowed.
"""
try:
    from .run import measure
    from .provenance import source_fingerprint
    from .resize import binary_hashes,python_hashes,source_revision,cpu_metadata,save,summarize
except ImportError:
    from run import measure
    from provenance import source_fingerprint
    from resize import binary_hashes,python_hashes,source_revision,cpu_metadata,save,summarize
import argparse
import inspect
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import traceback
import warnings
import zipfile
import numpy as np
import scipy
import skimage
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import ci_measurement_artifacts as artifacts
MANIFEST=ROOT/'coverage/contracts/rank-intensity-benchmark.json'
SOURCE_CONTRACT=ROOT/'coverage/contracts/rank-intensity.json'
NAMES=('autolevel','equalize','enhance_contrast','threshold')
CANONICAL=tuple('filters.rank.'+name for name in NAMES)
ALIASES=tuple('filters.rank.generic.'+name for name in NAMES)
APIS=CANONICAL+ALIASES
SEED=9419
CORPUS_VERSION='rank-intensity-benchmark-v1'


@dataclass(frozen=True)
class Case:
    name:str
    label:str
    fixture:dict
    timed:bool=True
    work_kind:str='rank-neighborhood'
    frozen_id:str|None=None


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source_text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def resolve(name,original=False,alias=False):
    module=('skimage' if original else 'skimagexx')+'.filters.rank'+('.generic' if alias else '')
    return getattr(importlib.import_module(module),name)


def profiles():
    shapes=[(17,19),(512,513),(7,11,13),(17,129,131)]
    patterns=['uniform','ramp','low-entropy','random'];footprints=['dense3','dense7','sparse','disconnected','shifted','even']
    outputs=['u1','u2','f4','f8']
    for s,shape in enumerate(shapes):
        for d,dtype in enumerate(['u1','u2']):
            for p,pattern in enumerate(patterns):
                yield dict(id=f'balanced/s{s}/d{d}/p{p}',shape=shape,dtype=dtype,pattern=pattern,
                           bins=256 if d==0 else [4,256,1024][(s+p)%3],footprint=footprints[(2*s+3*d+p)%6],
                           mask='sparse' if (s+p+d)%2 else 'none',layout='step' if (s+p//2+d)%2 else 'C',
                           out='default' if (s+d+p)%4==0 else outputs[(s+2*d+p)%4],out_layout='C')
    for ndim,shape in [(2,(17,19)),(3,(7,11,13))]:
        for d,dtype in enumerate(['u1','u2']):
            for f,footprint in enumerate(footprints):
                yield dict(id=f'footprint/{ndim}d/d{d}/f{f}',shape=shape,dtype=dtype,pattern='random',
                           bins=256 if d==0 else 1024,footprint=footprint,mask='sparse' if f%2 else 'none',
                           layout='C',out=outputs[(f+d)%4],out_layout='C')
    for ndim,shape in [(2,(17,19)),(3,(7,11,13))]:
        for b,bins in enumerate([4,256,1024,65536]):
            for p,pattern in enumerate(patterns):
                yield dict(id=f'bins/{ndim}d/b{bins}/p{p}',shape=shape,dtype='u2',pattern=pattern,bins=bins,
                           footprint='dense3' if p%2 else 'sparse',mask='sparse' if p%2 else 'none',
                           layout='step' if (b+p)%2 else 'C',out=outputs[p],out_layout='C')
    for ndim,shape in [(2,(17,19)),(3,(7,11,13))]:
        for d,dtype in enumerate(['u1','u2']):
            for o,out in enumerate(outputs):
                yield dict(id=f'output/{ndim}d/d{d}/o{o}',shape=shape,dtype=dtype,pattern='ramp',bins=256 if d==0 else 1024,
                           footprint='dense7',mask='sparse' if d else 'none',layout='C',out=out,
                           out_layout='step' if o%2 else 'C')


def corpus():
    manifest=json.loads(MANIFEST.read_text())
    fixtures=list(profiles())
    for name in NAMES:
        for profile in fixtures:
            successful=profile['out_layout']=='C'
            yield Case(name,profile['id'],profile,successful,'rank-neighborhood' if successful else 'defined-layout-error')
        for entry in manifest['selected_contracts']:
            if entry['name']==name:
                yield Case(name,'contract/'+entry['id'],entry['spec'],False,'contract-only',entry['id'])


def raster_fixture(spec):
    shape=tuple(spec['shape']);count=math.prod(shape);bins=spec['bins'];dtype=np.dtype(spec['dtype'])
    i=np.arange(count,dtype=np.uint64)
    if spec['pattern']=='uniform':values=np.full(count,bins-1,np.uint64)
    elif spec['pattern']=='ramp':values=i*(bins-1)//max(count-1,1)
    elif spec['pattern']=='low-entropy':values=np.array([0,bins//2,bins-1],np.uint64)[i%3]
    else:
        x=i+np.uint64(SEED)+np.uint64(0x9e3779b97f4a7c15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        values=(x^(x>>np.uint64(31)))%np.uint64(bins)
    a=values.astype(dtype).reshape(shape);a.flat[-1]=bins-1;tracked={}
    if spec['layout']=='step':
        owner=np.full(shape[:-1]+(shape[-1]*2,),17,dtype);owner[...,::2]=a;a=owner[...,::2];tracked['image_owner']=owner
    ndim=len(shape);kind=spec['footprint']
    fp=np.ones(((2,4) if ndim==2 else (2,3,4)) if kind=='even' else (7 if kind in ('dense7','sparse','disconnected') else 3,)*ndim,np.uint8)
    if kind=='sparse':fp.flat[::3]=0;fp.flat[0]=1
    elif kind=='disconnected':fp.fill(0);fp.flat[:2]=1;fp.flat[-2:]=1
    options={}
    if kind=='shifted':options.update(shift_x=1,shift_y=-1,shift_z=0)
    if spec['mask']=='sparse':options['mask']=((i*13+SEED)%7!=0).astype(np.uint8).reshape(shape)
    out=None if spec['out']=='default' else np.full(shape,99,np.dtype(spec['out']))
    if spec['out_layout']=='step':
        owner=np.full(shape[:-1]+(shape[-1]*2,),99,out.dtype);owner[...,::2]=out;out=owner[...,::2];tracked['out_owner']=owner
    if spec['out']!='default':options['out']=out
    tracked.update(image=a,footprint=fp)
    if 'out' in options:tracked['out']=out
    if 'mask' in options:tracked['mask']=options['mask']
    return (a,fp),options,tracked


def base_contract_fixture(spec):
    assert spec.get("out") not in ("image","view","overlap") and spec.get("special")!="alias-converted","R3 benchmark excludes all overlapping buffers"
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


def contract_fixture(parameters):
    if 'intensity' not in parameters and not parameters.get('omitted'):return base_contract_fixture(parameters)
    ndim=parameters['ndim'];shape=(5,7) if ndim==2 else (3,4,5);pattern=parameters.get('intensity')
    values=np.asarray([1,5,9],np.uint8)
    if pattern=='flat-zero':values=np.asarray([0],np.uint8)
    elif pattern=='flat-seven':values=np.asarray([7],np.uint8)
    elif pattern=='flat-u16':values=np.asarray([65535],np.uint16)
    elif pattern=='cdf-fraction':values=np.asarray([1,2,2,3,3,3,3],np.uint8)
    elif pattern=='u8-extremes':values=np.asarray([0,255,1,254,127,128],np.uint8)
    elif pattern=='u16-extremes':values=np.asarray([0,65535,1,65534,32767,32768],np.uint16)
    elif pattern=='u16-low-bins':values=np.asarray([0,1,2],np.uint16)
    image=np.resize(values,int(np.prod(shape))).reshape(shape)
    footprint=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8);options={}
    if pattern in ('center-below-window','center-above-window','masked-center'):
        image=np.resize(np.asarray([4,7,10],np.uint8),int(np.prod(shape))).reshape(shape)
        position=tuple(n//2 for n in shape);image[position]=0 if pattern=='center-below-window' else 255
        mask=np.ones(shape,np.uint8);mask[position]=0;options['mask']=mask
    elif pattern=='zero-window':options['mask']=np.zeros(shape,np.uint8)
    elif pattern=='disconnected':footprint.flat[1:-1]=0
    output=parameters.get('intensity_out')
    if output:
        dtype=np.dtype('u1',metadata={'rank':'intensity-out'}) if output=='tagged' else np.dtype(output)
        options['out']=np.full(shape,99,dtype=dtype)
    tracked={'image':image,**({} if parameters.get('omitted') else {'footprint':footprint}),**options}
    return ((image,) if parameters.get('omitted') else (image,footprint)),options,tracked

def make_case(case):return contract_fixture(case.fixture) if case.frozen_id else raster_fixture(case.fixture)


def describe(value,inputs=()):
    if isinstance(value,np.ndarray):
        return dict(type=type(value).__name__,dtype=value.dtype.str,descr=repr(value.dtype.descr),metadata=repr(value.dtype.metadata),
                    shape=list(value.shape),strides=list(value.strides),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),
                    owned=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),
                    values_sha256=hashlib.sha256((repr(value.tolist()).encode() if value.dtype.hasobject else value.tobytes(order='C'))).hexdigest(),
                    identity=[value is a for a in inputs],alias=[bool(np.shares_memory(value,a)) for a in inputs])
    if isinstance(value,(list,tuple)):return dict(type=type(value).__name__,items=[describe(v) for v in value])
    return dict(type=type(value).__name__,value=repr(value.item() if isinstance(value,np.generic) else value))


def spec(case):
    args,options,tracked=make_case(case)
    return dict(api='filters.rank.'+case.name,case=case.label,fixture=case.fixture,timed=case.timed,
                work_kind=case.work_kind,source_contract_id=case.frozen_id,
                positional_count=len(args),options={k:describe(v) for k,v in options.items()},
                input_contract={k:describe(v) for k,v in tracked.items()})


def observe_case(function,case):
    args,options,tracked=make_case(case);inputs=[a for a in tracked.values() if isinstance(a,np.ndarray)]
    before={k:describe(v) for k,v in tracked.items()};pointers={k:v.ctypes.data for k,v in tracked.items() if isinstance(v,np.ndarray)}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=describe(function(*args,**options),inputs),None
        except Exception as caught:result,error=None,dict(type=type(caught).__name__,message=str(caught))
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],
                before=before,after={k:describe(v) for k,v in tracked.items()},
                pointer_stable={k:tracked[k].ctypes.data==p for k,p in pointers.items()})


def check_aliases(workers,reference_only):
    records=[]
    for name in NAMES:
        original=resolve(name,True);alias=resolve(name,True,True)
        assert original is alias,'original generic alias identity changed'
        native=None
        if not reference_only:
            native=resolve(name);assert native is resolve(name,False,True),'native generic alias differs'
            assert native is not original and not getattr(native,'__module__','').startswith('skimage.'),'upstream-forwarded rank path'
        row=dict(canonical='filters.rank.'+name,alias='filters.rank.generic.'+name,reference_identity=True,
                 native_identity=None if reference_only else True,timed=False,algorithm_count=1,dispatch_checks=[])
        for ndim in [2,3]:
            case=Case(name,'alias-dispatch',dict(shape=(3,5) if ndim==2 else (3,4,5),dtype='u1' if ndim==2 else 'u2',
                pattern='ramp',bins=16,footprint='dense3',mask='none',layout='C',out='u1' if ndim==2 else 'u2',out_layout='C'),False)
            expected=observe_case(original,case);assert observe_case(alias,case)==expected
            if reference_only:row['dispatch_checks'].append(dict(ndim=ndim,reference=True))
            else:
                for count in workers:
                    xx.set_num_threads(count);assert xx.get_num_threads()==count
                    assert observe_case(native,case)==expected and observe_case(resolve(name,False,True),case)==expected
                    row['dispatch_checks'].append(dict(ndim=ndim,workers=count,passed=True))
        records.append(row)
    return records


def policy_observation(function,name,policy):
    case=Case(name,'policy/nan-image',dict(ndim=2,special='nan-image'),False,'dtype-policy','policy-only')
    previous=np.geterr();handler=np.geterrcall();events=[]
    def callback(message,flags):events.append([message,flags])
    class Logger:
        def write(self,message):events.append(message)
    saved=os.dup(2)
    try:
        np.seterrcall(Logger() if policy=='log' else callback);np.seterr(all=policy)
        with tempfile.TemporaryFile() as stream:
            sys.stderr.flush();os.dup2(stream.fileno(),2)
            try:result=observe_case(function,case);sys.stderr.flush()
            finally:os.dup2(saved,2)
            stream.seek(0);printed=stream.read().decode('utf-8',errors='replace')
        return dict(observation=result,events=events,printed=printed)
    finally:
        os.dup2(saved,2);os.close(saved);np.seterr(**previous);np.seterrcall(handler)


def check_policies(workers,reference_only):
    rows=[]
    for name in NAMES:
        for policy in ('warn','raise','ignore','call','log','print'):
            expected=policy_observation(resolve(name,True),name,policy)
            row=dict(api='filters.rank.'+name,policy=policy,timed=False,reference=expected)
            if not reference_only:
                row['actual_by_workers']={}
                for count in workers:
                    xx.set_num_threads(count)
                    actual=policy_observation(resolve(name),name,policy)
                    assert actual==expected,(name,policy,count,actual,expected)
                    row['actual_by_workers'][str(count)]=actual
            rows.append(row)
    return rows


def check_case(case,workers,reference_only,expectations,row):
    expected=observe_case(resolve(case.name,True),case)
    if case.frozen_id:
        profile=runtime_profile();frozen=expectations[case.frozen_id]['profiles'][profile]
        assert expected['error']==frozen['error'] and expected['warnings']==frozen['warnings'],'original differs from pinned1012 contract'
        if frozen['result'] is not None:
            for key,value in frozen['result'].items():
                actual_key='values_sha256' if key=='raw_bytes_sha256' else key
                assert expected['result'][actual_key]==value,('pinned original result differs',case.frozen_id,key)
    elif not case.timed:
        assert expected['error']==dict(type='ValueError',message='ndarray is not C-contiguous'),'defined strided-output admission changed'
    else:
        assert expected['error'] is None,'timing fixture must be a defined successful original call'
        for key in expected['before']:
            if key not in ('out','out_owner'):assert expected['before'][key]==expected['after'][key],'timed fixture mutated an input'
    row['reference']=expected
    if not reference_only:
        actual={}
        for count in workers:
            xx.set_num_threads(count);assert xx.get_num_threads()==count
            result=observe_case(resolve(case.name),case);assert result==expected,(count,result,expected)
            actual[str(count)]=result
        row['actual_by_workers']=actual


def runtime_profile():
    return 'minimum' if np.__version__.startswith('1.') else 'windows' if platform.system()=='Windows' else 'modern'


def policy_reference_rows(rows):
    return [{key:row[key] for key in ("api","policy","timed","reference")} for row in rows]


def contract_inventory(case):
    return json.loads(json.dumps(dict(name=case.name,label=case.label,fixture=case.fixture,timed=case.timed,work_kind=case.work_kind,frozen_id=case.frozen_id)))


def timing_callback(function,case):
    args,options,tracked=make_case(case)
    out=options.get('out')
    reset=None if out is None else out.copy(order='K')
    def call():
        # Caller-owned output reset is part of both public whole-call costs.
        # Default-output library allocation and returned-reference release are
        # included by the shared paired sampler.
        if out is not None:np.copyto(out,reset)
        return function(*args,**options)
    return call,tracked


def seal(capsule):
    path=capsule/'rank-intensity-artifacts.json';assert not path.exists(),'choose a fresh capsule'
    runtime=artifacts.freeze(ROOT,capsule/'wheels',capsule/'rank-intensity-runtime.json')
    archive=capsule/'source.zip';assert not archive.exists(),'choose a fresh source archive'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in runtime['source']['files_sha256']:z.write(ROOT/name,name)
    info=dict(schema='rank-intensity-exact-offline-v1',runtime_manifest_sha256=artifacts.sha(capsule/'rank-intensity-runtime.json'),
              corpus_manifest_sha256=artifacts.sha(MANIFEST),product_source=source_fingerprint(ROOT),source_archive_sha256=artifacts.sha(archive))
    path.write_text(json.dumps(info,indent=2)+'\n');return verify(capsule)


def verify(capsule):
    path=capsule/'rank-intensity-artifacts.json';info=json.loads(path.read_text())
    assert info['schema']=='rank-intensity-exact-offline-v1'
    assert info['runtime_manifest_sha256']==artifacts.sha(capsule/'rank-intensity-runtime.json'),'runtime manifest changed'
    assert info['corpus_manifest_sha256']==artifacts.sha(MANIFEST),'corpus manifest changed'
    assert info['product_source']==source_fingerprint(ROOT),'product source changed'
    assert info['source_archive_sha256']==artifacts.sha(capsule/'source.zip'),'source archive changed'
    runtime=artifacts.verify(ROOT,capsule/'wheels',capsule/'rank-intensity-runtime.json')
    assert {'include/skimagexx/image.hpp','include/skimagexx/rank.hpp'}<=set(runtime['runtime']['sdk_sha256']),'rank SDK required'
    with zipfile.ZipFile(capsule/'source.zip') as archive:
        expected=runtime['source']['files_sha256']
        assert set(archive.namelist())==set(expected),'complete source archive file set changed'
        assert all(hashlib.sha256(archive.read(n).replace(b'\r\n',b'\n')).hexdigest()==h for n,h in expected.items()),'source archive member changed'
    product_wheels=0
    for wheel in (capsule/'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            names=archive.namelist()
            if 'skimagexx/__init__.py' not in names:continue
            product_wheels+=1
            assert {n for n in names if n.startswith('include/skimagexx/') and not n.endswith('/')}=={
                p.relative_to(ROOT).as_posix() for p in (ROOT/'include/skimagexx').rglob('*') if p.is_file()},'complete SDK header set differs from source'
            assert {n.removeprefix('skimagexx/'):archive.read(n).replace(b'\r\n',b'\n') for n in names if n.startswith('skimagexx/') and n.endswith('.py')}=={
                p.relative_to(ROOT/'python/skimagexx').as_posix():p.read_bytes().replace(b'\r\n',b'\n') for p in (ROOT/'python/skimagexx').rglob('*.py')},'complete Python source differs from wheel'
            for name in names:
                if name.startswith('include/skimagexx/') and not name.endswith('/'):
                    assert archive.read(name).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),'SDK header bytes differ from source'
    assert product_wheels==1,'exactly one complete product wheel required'
    return dict(manifest_sha256=artifacts.sha(path),**info)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',default=[1]);parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--target-ms',type=float,default=15.);parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--capsule',type=Path);parser.add_argument('--prepare-capsule',action='store_true')
    args=parser.parse_args(argv)
    if min(args.threads)<=0 or max(args.threads)>1024 or 1 not in args.threads or len(set(args.threads))!=len(args.threads) or args.repeat<=0 or not math.isfinite(args.target_ms) or args.target_ms<=0:parser.error('distinct workers in1..1024/repeats/target-ms required; include worker1')
    if args.reference_only and (not args.check_only or args.capsule is not None):parser.error('--reference-only requires --check-only without a native capsule')
    if args.prepare_capsule and (not args.check_only or args.reference_only or args.capsule is None):parser.error('--prepare-capsule requires strict --check-only and --capsule')
    if not args.check_only and (args.output is None or args.capsule is None or os.environ.get('GITHUB_ACTIONS')!='true'):parser.error('measurements require an isolated Actions host, --output and --capsule; use --check-only locally')
    fingerprint=source_fingerprint(ROOT);manifest_hash=artifacts.sha(MANIFEST)
    harness_paths=['benchmarks/rank_intensity.py','benchmarks/resize.py','benchmarks/run.py','benchmarks/provenance.py','tools/ci_measurement_artifacts.py']
    harness={n:source_text_sha(ROOT/n) for n in harness_paths}
    harness_bytes={n:artifacts.sha(ROOT/n) for n in harness_paths}
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),**cpu_metadata(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,scikit_image=skimage.__version__,skimagexx=xx.__version__,avx2=xx.has_avx2(),source_revision=source_revision(ROOT),**fingerprint,harness_sha256=harness,harness_source_hash_schema="sha256-posix-lf-v1",harness_bytes_sha256=harness_bytes,corpus_manifest_sha256=manifest_hash,current_binary_sha256=binary_hashes(),current_python_sha256=python_hashes(),command=sys.argv if argv is None else list(argv),corpus_version=CORPUS_VERSION,seed=SEED,requested_workers=args.threads,primary_workers=1,algorithm_count=4,operation_ids=dict(zip(NAMES,range(9,13))),canonical_paths=list(CANONICAL),alias_paths=list(ALIASES),aliases_timed=False,fallback_enabled=False,check_only=args.check_only,reference_only=args.reference_only,validation_scope='reference-only; native image paths not exercised' if args.reference_only else 'strict actual public native canonical paths and untimed identity aliases',scope='Default-output public allocation included; supplied-output reset, public call and returned-reference release included on both backends. Input/footprint/mask fixture setup excluded. Error/metadata/alias/policy cases never timed.',repeat=args.repeat,target_ms=args.target_ms,cooldown_ms=20,sample_order_seed=81,execution_environment='github-actions' if os.environ.get('GITHUB_ACTIONS')=='true' else 'local-untimed',github_run_id=os.environ.get('GITHUB_RUN_ID'),github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),github_job=os.environ.get('GITHUB_JOB')),status='preflight',alias_checks=[],policy_checks=[],checks=[],results=[],untimed_cases=[],failures=[])
    extension=Path(importlib.import_module('skimage.filters.rank.generic').generic_cy.__file__).resolve()
    report['metadata']['reference_extension']=dict(path=str(extension),sha256=artifacts.sha(extension))
    save(report,args.output);previous_threads=xx.get_num_threads();previous_fallback=xx.set_fallback_enabled(False)
    try:
        if skimage.__version__!='0.26.0':raise RuntimeError('reference must be exactly scikit-image0.26.0')
        if not args.reference_only:
            missing=sorted(set(APIS)-set(native_functions()))
            if missing:raise RuntimeError('public native rank APIs are not registered: '+', '.join(missing))
            capabilities=getattr(xx._core,'rank_histogram_capabilities',0)
            if not isinstance(capabilities,int) or capabilities<0 or capabilities & 8191 != 8191:raise RuntimeError('rank-histogram-v3 compiled capability8191 required')
            assert previous_threads==1,'fresh public process must default to worker1'
        manifest=json.loads(MANIFEST.read_text())
        assert manifest['source_contract_hash_schema']=='sha256-posix-lf-v1'
        assert hashlib.sha256(SOURCE_CONTRACT.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['source_contract_sha256'],'pinned1012 source contract changed'
        for name,digest_value in manifest['fixture_source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest_value,'qualified fixture source changed'
        for name in NAMES:
            original=resolve(name,True);assert original is resolve(name,True,True),'original generic alias identity changed'
            facts=manifest['reference_functions'][name]
            assert str(inspect.signature(original))==facts['signature']
            assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==facts['python_source_sha256'],'original function source changed'
            if not args.reference_only:
                current=resolve(name);assert current is resolve(name,False,True),'native generic alias differs'
                assert current is not original and not getattr(current,'__module__','').startswith('skimage.'),'upstream-forwarded rank path'
                assert inspect.signature(current)==inspect.signature(original),'public rank signature differs'
        frozen=manifest['reference_profiles'].get(runtime_profile())
        if not args.reference_only:
            assert frozen is not None,'frozen original profile required before native checks'
            assert args.capsule is not None,'strict public checks require --capsule'
            report['metadata']['exact_artifacts']=seal(args.capsule) if args.prepare_capsule else verify(args.capsule)
        report['metadata']['intensity_formulas']={n:manifest['reference_functions'][n]['formula'] for n in NAMES}
        cases=list(corpus());inventory=[contract_inventory(c) for c in cases]
        assert inventory==manifest['case_inventory'],'fixed case IDs/fixtures changed'
        assert digest(inventory)==manifest['case_contract_sha256'],'fixed structural case contract changed'
        specs=[spec(c) for c in cases]
        if frozen:assert digest(specs)==frozen['case_spec_sha256'],'frozen actual input data/options changed'
        report['metadata'].update(profile=runtime_profile(),case_count=len(cases),timing_eligible_case_count=sum(c.timed for c in cases),case_spec_sha256=digest(specs),case_contract_sha256=digest(inventory),case_specs=specs)
        report['alias_checks']=check_aliases(args.threads,args.reference_only)
        report['policy_checks']=check_policies(args.threads,args.reference_only)
        if frozen:assert digest(policy_reference_rows(report['policy_checks']))==frozen['policy_sha256'],'frozen original error-policy observations changed'
        expectations={entry['id']:entry for entry in manifest['selected_contracts']}
        report['status']='checking';save(report,args.output)
        for index,case in enumerate(cases):
            row=dict(case_index=index,**specs[index])
            try:
                check_case(case,args.threads,args.reference_only,expectations,row)
                row['classification']='validation-error' if row['reference']['error'] else 'raster-workload' if case.timed else 'contract-metadata'
                if frozen:assert digest(row['reference'])==frozen['observations'][str(index)],'original observation differs from frozen profile'
                row['status']='passed'
            except Exception as error:
                row.update(status='failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc());report['failures'].append(row)
            report['checks'].append(row)
            if not case.timed:report['untimed_cases'].append(dict(case_index=index,reason='defined error/conversion/metadata source cases are never timed'))
            if (index+1)%32==0 or row['status']=='failed' or index+1==len(cases):save(report,args.output)
            print(f'CHECK {index+1}/{len(cases)} {case.name} {case.label}: {row["status"]}',flush=True)
        if report['failures']:report['status']='failed-parity';return 1
        def integrity():
            assert source_fingerprint(ROOT)==fingerprint and artifacts.sha(MANIFEST)==manifest_hash,'source/corpus changed'
            assert all(source_text_sha(ROOT/n)==h for n,h in harness.items()),'harness source changed'
            assert all(artifacts.sha(ROOT/n)==h for n,h in harness_bytes.items()),'harness bytes changed during run'
            assert binary_hashes()==report['metadata']['current_binary_sha256'] and python_hashes()==report['metadata']['current_python_sha256'],'installed package changed'
            if not args.reference_only:assert verify(args.capsule)==report['metadata']['exact_artifacts'],'artifact seal changed'
        integrity()
        if args.check_only:report['status']='complete-reference-check' if args.reference_only else 'complete-parity-check';return 0
        report['status']='measuring';rng=random.Random(81);orders=[]
        for _ in range(args.repeat):order=['scikit-image','current'];rng.shuffle(order);orders.append(order)
        report['metadata']['paired_sample_order']=orders
        for index,case in enumerate(cases):
            if not case.timed:continue
            for workers in args.threads:
                xx.set_num_threads(workers);left,left_arrays=timing_callback(resolve(case.name,True),case);right,right_arrays=timing_callback(resolve(case.name),case)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');timing=measure(left,right,args.repeat,args.target_ms)
                expected=report['checks'][index]['reference']['after']
                assert {k:describe(v) for k,v in left_arrays.items()}==expected and {k:describe(v) for k,v in right_arrays.items()}==expected,'timing changed workload state'
                row=dict(case_index=index,api='filters.rank.'+case.name,case=case.label,work_kind=case.work_kind,threads=workers,primary=workers==1,image_kernel_speedup_claim=False,ratio_scope='public canonical allocation/output-reset-inclusive call',slower_than_scikit_image=timing['speedup']<1,**timing)
                report['results'].append(row);save(report,args.output)
        integrity()
        report['summary_by_algorithm']={name:summarize([r for r in report['results'] if r['api']=='filters.rank.'+name],args.threads,sum(c.timed and c.name==name for c in cases)) for name in NAMES}
        report['status']='complete-measurement';return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'],exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));report['status']='failed';print(f'FAILED {type(error).__name__}: {error}',file=sys.stderr,flush=True);return 1
    finally:xx.set_num_threads(previous_threads);xx.set_fallback_enabled(previous_fallback);save(report,args.output)


if __name__=='__main__':raise SystemExit(main())
