"""Untimed, unaccepted discovery of actual default matrix BLAS arithmetic.

The hosted-only builder compiles a tiny genuine private module. Candidate
agreement never changes production admission or asserts public qualification.
``plan`` and ``verify`` are metadata-only and may run locally.
"""
from __future__ import annotations
import argparse
import ast
import base64
import csv
import ctypes
from datetime import datetime, timezone
from email.parser import BytesParser
import gzip
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import venv
import warnings
import zipfile
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'matrix-default-profile-discovery-v1'
SOURCE_REVISION = '52066265b89ceecf613c4a5f7d3b73c23b44d58d'
SOURCE_HASHES = {'include/skimagexx/matrix_transform.hpp': 'd94ba05b8cfa301c496f263a8cbf37586be5b5777281739cc684c177524f58ec', 'include/skimagexx/image.hpp': 'b0e283e0dc73d5aeefb90151f25fbe8b603758db74841189b8a37500da24e093', 'src/matrix_transform.cpp': '99df2df59bb6d90ae919ea423786e3f5bc1d7490aeea9c31c4671f886ce130b4', 'src/simd_matrix_transform_avx2.cpp': 'f920323c029df46b99e617bcc69dfb52123750641cec0c01bc90cf7ddbc5eabf', 'src/image.cpp': '17a224f0aa15be6591867d1a07b5b957ffdcec1983d072f19e48277eafec3452', 'src/simd_avx2.cpp': 'c4afd6790f2f8289eeaf41fd0b1663aa64bd9238a825b273ba48d94af1a7a4ab', 'src/simd.hpp': '848be89cc6dcfd0965dc9f18f905404ac8671fa4e33fb51b50372f07fde22141', 'src/execution.hpp': '029b02e920c962a2679e58ade6e201d16188b42885170a5c9a012fffe6a7331d', 'src/bind_matrix_transform.cpp': '0504ab1040e4e0467b872a151f8b3301a0e8777e2bb00cedd1c33f4b8be8582a', 'tests/test_matrix_transform_native.cpp': '466d7645d841dfbbdbaf9c880d72a73f458af528299e2962493ce63e32cc2eb0', 'tests/test_matrix_transform_contracts.py': 'a2bf6398cc758effe11531cc7aee1add7808903a5f2e098488d243cab2a7b3ea', 'coverage/contracts/matrix-transform.json': '6622a08fb4b420588d22325b3dc72fd053a7e4d689a4ed7ef88fe10fbc44e0f2', 'tools/public_qualification.py': 'feedb92ae0d87bb03d475145d8866b4ade093e1ab2fab7285689f151a3a4c8ed'}
PROFILES = {
    'windows': dict(python='3.13', config='Release'),
    'modern': dict(python='3.13', config='Release'),
    'minimum': dict(python='3.11', config='Release'),
    'debug': dict(python='3.13', config='Debug'),
}
SOURCE_HASHES['src/simd_math.hpp']='d5ac161aa4c38d09433b82dfe12abc36ee7660b7da8e82a02eceaeaba0785047'
# These declaration-only headers are included transitively by the exact shared
# simd.hpp foundation. Retain them even though this packet adds no new kernels.
SOURCE_HASHES.update({
    'include/skimagexx/convert.hpp':'829e727d1665b7fc2ad2ff90bc3b81bf5bc2318d7da07fcaab2df595ffe21016',
    'include/skimagexx/distance.hpp':'3a2733715aa86a371831a494f9af94399442234b510bb41909c7d0c1b0c893f5',
    'include/skimagexx/local_threshold.hpp':'019bbd11d69237e54f314bdc3c702f7ddc838970937bf09040f029a74d52e378',
})
SOURCE_HASHES.update({
    'python/skimagexx/_matrix_transform.py': 'deb219b0cfb0f38d04e53d13e09ebfc40196f29de97c1e3c4c3d4273e416e9b8',
    'tests/test_matrix_transform_native_pipeline.py': '7048150bb9de8b4569c8a54ed86e75fee0e1f8e617179f09e0903a4007b96322',
    'tests/test_matrix_transform_binding.py': 'd034027026c54a04c17247eb5af20fec19ab2d1ce756b7f11c64a3e90b642a9a',
})
FORBIDDEN = ('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES',
             'MKL_DEBUG_CPU_TYPE','MKL_SERVICE_FORCE_INTEL')
CONTROLLER_PATHS=('tools/matrix_transform_profiles.py','tests/test_matrix_transform_profiles.py',
                  '.github/workflows/matrix-transform-profiles.yml')
ENV_KEYS = (*FORBIDDEN,'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'BLIS_NUM_THREADS','OMP_WAIT_POLICY','SKIMAGEXX_DISABLE_SIMD','LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH')

# Accepted 192-path products provide ONLY genuine package/registry context.
# Matrix arithmetic always runs the separately compiled frozen520 private module.
CONTEXT_SOURCE = '3cdd3637ca78343edb2cb554b77269de5f5eaf20'
CONTEXT_RUN = 40098671228
CONTEXT = {
 'windows': (12562900552,'a6d724f148c181bc67d70e3cb3237760cd0d24bc66783dacf9bdfbd55056e4d1','904578774360e251fb484acc9b8b2459b6043f23689acc72f8c3d63e8ac89c1a','80d37e8c4b3563c2a7c869a0d886091b3f5026c05bebab60d9a7b262c75295bb'),
 'modern': (12563321173,'c7ce0cc96e136530a91cd352f6edb60fd407e06af10b2a1a89b9776d8a182ed3','8ec0e15fc4617c616129d86cffda66103ba77407a68bc4859427e1d6781c83e6','65e5d48949f6579ef02fbbaf9eaee9ebf80b159c7535db35b6e2c16418aeadab'),
 'minimum': (12562883719,'a559d90a8ec7188c59bdc2b731dc2e2df378c61f85c98a51a25aa48c882083b1','3bee573c9a8b14de45fb0803482398c939763b246e9eb0f27b0a387fc0b4391b','7ed0d849f97881259379baaae8e481adc3330b6633ca8aa44e08750b6f70ec24'),
}


def context_members(data, pins):
    """Authenticate the entire donor ZIP, then exact envelope and wheel bytes."""
    require(digest(data)==pins[1],'context artifact digest mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)),'duplicate context ZIP member')
        require(all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts and '\\' not in n for n in names),'context ZIP path escape')
        products=[n for n in names if n.endswith('product.json') and digest(archive.read(n))==pins[2]]
        wheels=[n for n in names if n.endswith('.whl') and digest(archive.read(n))==pins[3]]
        require(len(products)==len(wheels)==1,'exact accepted context product/wheel absent')
        return products[0],archive.read(products[0]),wheels[0],archive.read(wheels[0])


def verify_project_record(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=[n for n in archive.namelist() if not n.endswith('/')]
        require(len(names)==len(set(names)),'duplicate project wheel member')
        require(all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts and '\\' not in n for n in names),'project wheel path escape')
        records=[n for n in names if n.endswith('.dist-info/RECORD')]
        require(len(records)==1,'project wheel RECORD missing')
        rows=list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        require({r[0] for r in rows}==set(names) and len(rows)==len(names),'project RECORD closure differs')
        for name,hashed,size in rows:
            if name==records[0]:require(not hashed and not size,'RECORD self hash');continue
            raw=archive.read(name)
            want='sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
            require(hashed==want and size==str(len(raw)),'project RECORD bytes mismatch')
        return {n:dict(bytes=len(archive.read(n)),sha256=digest(archive.read(n))) for n in names}


def install_context(profile,out,journal):
    key='modern' if profile=='debug' else profile;pins=CONTEXT[key]
    token=os.environ.get('GH_TOKEN');require(token,'read-only Actions token required for exact accepted context')
    api='https://api.github.com/repos/bipxx/scikit-imagexx/actions/artifacts/'+str(pins[0])
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'}
    with urllib.request.urlopen(urllib.request.Request(api,headers=headers)) as response: metadata=json.load(response)
    require(metadata['workflow_run']['id']==CONTEXT_RUN and metadata['workflow_run']['head_sha']==CONTEXT_SOURCE,'context run/source differs')
    require(metadata['digest']=='sha256:'+pins[1] and not metadata['expired'],'context remote digest/expiry differs')
    # Never forward the Actions bearer token to the redirected blob host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(api+'/zip',headers=headers))
        raise ValueError('expected signed artifact redirect')
    except urllib.error.HTTPError as error:
        require(error.code in (301,302,303,307,308),'artifact redirect failed')
        url=error.headers['Location'];require(url.startswith('https://'),'unsafe artifact redirect')
    with urllib.request.urlopen(url) as response:data=response.read(512*1024*1024+1)
    require(len(data)<=512*1024*1024,'context artifact exceeded bounded download')
    product_name,product,wheel_name,wheel=context_members(data,pins)
    records=verify_project_record(wheel)
    folder=out/'context';folder.mkdir();path=folder/Path(wheel_name).name;path.write_bytes(wheel)
    (folder/'product.json').write_bytes(product)
    dump(out/'context.json',dict(artifact=metadata,product_member=product_name,wheel_member=wheel_name,
        wheel_sha256=digest(wheel),project_records=records,scope='Accepted package context only; donor core never supplies matrix arithmetic'))
    journal.run('install-context',[env_python(out/'runtime-env'),'-m','pip','install','--no-index','--no-deps',path],environment())


def verify_context_files(out,profile):
    out=Path(out);key='modern' if profile=='debug' else profile;pins=CONTEXT[key]
    files=[p for p in (out/'context').iterdir() if p.is_file()]
    wheels=[p for p in files if p.suffix=='.whl']
    require(len(files)==2 and len(wheels)==1,'context file closure differs')
    require(sha(wheels[0])==pins[3] and sha(out/'context/product.json')==pins[2],'context envelope/wheel pins differ')
    records=verify_project_record(wheels[0].read_bytes())
    context=json.loads((out/'context.json').read_text())
    require(context['project_records']==records and context['wheel_sha256']==pins[3],'context RECORD census differs')
    require(context['artifact']['id']==pins[0] and context['artifact']['digest']=='sha256:'+pins[1],'context artifact pin differs')
    require(context['artifact']['workflow_run']['id']==CONTEXT_RUN and context['artifact']['workflow_run']['head_sha']==CONTEXT_SOURCE,'context source/run differs')


def validate_pipeline(out,mode):
    suites=ET.parse(Path(out)/('pipeline-'+mode+'.xml')).getroot()
    counts={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','errors','failures','skipped')}
    require(counts==dict(tests=8894,errors=0,failures=0,skipped=0),'complete unchanged pipeline/direct guards failed')
    routes=json.loads((Path(out)/('pipeline-routes-'+mode+'.json')).read_text())
    require(len(routes['rows'])==748 and len(routes['general_pipeline'])==8000,'incomplete 374/4000 workers1/4 pipeline')
    require(all(r['cpp_completed']==1 and r['expected_sha256']==r['actual_sha256'] for r in routes['general_pipeline']),'general cohort was provider-only or differs')
    require({r['workers'] for r in routes['general_pipeline']}=={1,4},'missing global-worker pipeline')
    require(routes['public_registration'] is False and routes['operation_timings']==0,'false public/timing claim')
    return counts


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def dump(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def file_record(path):
    path=Path(path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))


def git(source,*args):
    return subprocess.check_output(['git','-C',str(source),*args])


def canonical(data):
    return data.replace(b'\r\n',b'\n')


def validate_source_records(records):
    require(set(records)==set(SOURCE_HASHES),'incomplete source closure')
    for name, expected in SOURCE_HASHES.items():
        require(records[name]==expected,'wrong frozen source: '+name)


def validate_controller_members(folder,records):
    require(set(records)==set(CONTROLLER_PATHS),'controller closure differs')
    for name,want in records.items():
        require(sha(Path(folder)/name)==want,'controller source bytes differ')


def source_state(source, revision, materialized=False):
    require(revision==SOURCE_REVISION,'only the reviewed exact source revision is permitted')
    require(git(source,'rev-parse',revision).decode().strip()==revision,'source revision mismatch')
    blobs={name:git(source,'show',revision+':'+name) for name in SOURCE_HASHES}
    hashes={name:digest(canonical(data)) for name,data in blobs.items()}
    validate_source_records(hashes)
    if materialized:
        require(git(source,'rev-parse','HEAD').decode().strip()==revision,'source checkout HEAD mismatch')
        require(not git(source,'status','--porcelain','--untracked-files=all').strip(),'source checkout dirty')
        validate_source_records({n:digest(canonical((source/n).read_bytes())) for n in SOURCE_HASHES})
    return dict(revision=revision,canonical_members=hashes,materialized=materialized),blobs


def literal(node):
    """Read pinned config literals without executing the source controller."""
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,(ast.List,ast.Tuple)):return [literal(n) for n in node.elts]
    if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
        result={}
        for item in node.keywords:
            if item.arg is None:result.update(literal(item.value))
            else:result[item.arg]=literal(item.value)
        return result
    raise ValueError('not a pinned config literal')


def pinned_config(blobs,profile):
    tree=ast.parse(blobs['tools/public_qualification.py'].decode())
    names={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('PROFILES','COMMON_RUNTIME','BUILD','RUNTIME_WHEELS'):names[name]=literal(node.value)
    require(set(names)=={'PROFILES','COMMON_RUNTIME','BUILD','RUNTIME_WHEELS'},'incomplete pinned dependency source')
    p=names['PROFILES'][profile]
    runtime=dict(names['COMMON_RUNTIME'],numpy=p['numpy'],scipy=p['scipy'],
        networkx='3.6.1' if profile=='minimum' else '3.7',
        tifffile='2026.3.3' if profile=='minimum' else '2026.10.15')
    if profile=='windows':runtime['colorama']='0.4.6'
    wheels=names['RUNTIME_WHEELS']['modern' if profile=='debug' else profile]
    require(set(wheels)==set(runtime),'pinned runtime closure mismatch')
    return dict(profile=p,runtime=runtime,runtime_wheels=wheels,
        build={n:names['BUILD'][n] for n in ('pybind11','cmake','ninja','packaging')},
        provenance='Exact frozen public_qualification.py config, authenticated runtime40024754860 versions/wheels; no original project wheel installed.')


def cohort_spec(data):
    value=json.loads(data.decode().split('// MATRIX_COHORTS_BEGIN\n/*\n')[1].split('\n*/')[0])
    require(value['schema']=='matrix-general-finite-cohorts-v2','wrong cohort version')
    require(value['include_literal_base'] is True and len(value['cases'])==10,'incomplete cohorts')
    require((1+len(value['rows']))*len(value['layouts'])**2*len(value['cases'])==4000,'not the complete4000 corpus')
    return value


def case_specs(spec):
    for case in spec['cases']:
        for n in [0]+spec['rows']:
            for p in spec['layouts']:
                for m in spec['layouts']:
                    yield dict(case=case,n=n,points_layout=p,matrix_layout=m,extension=spec['row_extension'])


def validate_default_environment(environ):
    for name in FORBIDDEN:
        require(name not in environ,'CPU dispatch override present: '+name)
    require(not environ.get('LD_PRELOAD') and not environ.get('LD_AUDIT'),'preloaded numerical/runtime code is not default dispatch')


def environment(mode=None):
    env=os.environ.copy();validate_default_environment(env)
    for name in list(env):
        if name.startswith('SKIMAGEXX_'):env.pop(name)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH='',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    if mode=='scalar':env['SKIMAGEXX_DISABLE_SIMD']='1'
    return env


class Journal:
    def __init__(self,out):self.out=Path(out);self.count=len(list(self.out.glob('command-*.json')))
    def run(self,label,argv,env=None):
        # A sequential child may have appended its own pytest command record.
        self.count=max(self.count,len(list(self.out.glob('command-*.json'))))
        self.count+=1;stem=self.out/('command-%03d-%s'%(self.count,label))
        record=dict(label=label,argv=list(map(str,argv)),utc=datetime.now(timezone.utc).isoformat(),
            environment={k:(env or os.environ).get(k) for k in ENV_KEYS})
        dump(stem.with_suffix('.json'),record)
        with stem.with_suffix('.stdout').open('wb') as stdout,stem.with_suffix('.stderr').open('wb') as stderr:
            r=subprocess.run(record['argv'],env=env,stdout=stdout,stderr=stderr)
        record['returncode']=r.returncode;dump(stem.with_suffix('.json'),record)
        require(r.returncode==0,'command failed; raw files retained: '+label)
        return stem.with_suffix('.stdout').read_bytes()


def norm_name(name):return re.sub('[-_.]+','-',name).lower()


def wheel_metadata(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist();require(len(names)==len(set(names)),'duplicate wheel members')
        for name in names:
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'wheel member escape')
        meta=next(n for n in names if n.endswith('.dist-info/METADATA'))
        m=BytesParser().parsebytes(z.read(meta));record=next(n for n in names if n.endswith('.dist-info/RECORD'))
        rows=list(csv.reader(io.StringIO(z.read(record).decode())))
        require({r[0] for r in rows}=={n for n in names if not n.endswith('/')},'wheel RECORD closure mismatch')
        for n,h,size in rows:
            if h:
                b=z.read(n);want='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('=')
                require(h==want and int(size)==len(b),'wheel RECORD bytes mismatch')
            else:require(n==record and not size,'unhashed wheel member')
        return dict(name=norm_name(m['Name']),version=m['Version'],filename=Path(path).name,sha256=sha(path))


def verify_wheelhouse(folder,pins,known=None):
    rows=[wheel_metadata(p) for p in sorted(Path(folder).glob('*.whl'))]
    require(len(rows)==len(pins) and {r['name'] for r in rows}==set(pins),'wheelhouse closure differs')
    for row in rows:
        require(row['version']==pins[row['name']],'wheel version differs')
        if known:
            require({k:row[k] for k in ('filename','sha256')}==known[row['name']],'original wheel bytes differ')
    return rows


def installed_records(wheels,site):
    """Authenticate loaded numerical source/native files to exact wheel bytes."""
    members={}
    for wheel in Path(wheels).glob('*.whl'):
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                if name.endswith('/'):continue
                require('.data/' not in name,'unexpected runtime wheel data relocation')
                b=z.read(name);path=Path(site)/name
                # Installer rewrites RECORD; entry point scripts are outside site.
                if name.endswith('.dist-info/RECORD'):continue
                require(path.is_file() and path.read_bytes()==b,'installed runtime file differs: '+name)
                members[str(path.resolve())]=digest(b)
    return members


def validate_library_origin(path,libdir,records):
    path=Path(path).resolve();libdir=Path(libdir).resolve()
    require(path.is_relative_to(libdir),'resolved BLAS has wrong origin')
    require(str(path) in records and sha(path)==records[str(path)],'resolved BLAS not authenticated to NumPy wheel')
    return file_record(path)


def pe_imports(path):
    data=Path(path).read_bytes();require(data[:2]==b'MZ','not PE')
    off=struct.unpack_from('<I',data,0x3c)[0];require(data[off:off+4]==b'PE\0\0','invalid PE')
    count=struct.unpack_from('<H',data,off+6)[0];size=struct.unpack_from('<H',data,off+20)[0];opt=off+24
    require(struct.unpack_from('<H',data,opt)[0]==0x20b,'not x64 PE32+')
    rva=struct.unpack_from('<I',data,opt+112+8)[0];sections=[]
    for i in range(count):
        p=opt+size+i*40;vs,va,rs,rp=struct.unpack_from('<IIII',data,p+8);sections.append((va,max(vs,rs),rp))
    def address(v):
        for base,length,raw in sections:
            if base<=v<base+length:return raw+v-base
        raise ValueError('PE RVA outside sections')
    p=address(rva);names=[]
    for _ in range(1000):
        row=struct.unpack_from('<IIIII',data,p);p+=20
        if not any(row):return names
        n=address(row[3]);end=data.index(b'\0',n);names.append(data[n:end].decode('ascii'))
    raise ValueError('PE imports unterminated')


def blas_facts(np,umath,records):
    libs=Path(np.__file__).parent.parent/'numpy.libs';symbols={}
    if os.name=='nt':
        imports=pe_imports(umath.__file__)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetModuleHandleW.argtypes=[ctypes.c_wchar_p];kernel.GetModuleHandleW.restype=ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32]
        matches=[]
        for name in imports:
            if 'openblas' not in name.lower():continue
            handle=kernel.GetModuleHandleW(name);require(handle,'imported BLAS module not loaded')
            buf=ctypes.create_unicode_buffer(32768);require(kernel.GetModuleFileNameW(handle,buf,len(buf))>0,'missing loaded DLL path')
            matches.append(Path(buf.value).resolve())
        require(len(matches)==1,'require one actual imported OpenBLAS DLL')
        library=ctypes.CDLL(str(matches[0]));origin=validate_library_origin(matches[0],libs,records)
        resolve=lambda f:matches[0]
        method='PE import basename + actual loaded module handle/path'
    else:
        # dlsym on UMATH searches its loaded dependency group; never open an
        # arbitrary bundled library and call that the reference dependency.
        library=ctypes.CDLL(umath.__file__)
        class Info(ctypes.Structure):
            _fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
        query=ctypes.CDLL(None).dladdr;query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];query.restype=ctypes.c_int
        def resolve(f):
            info=Info();require(query(ctypes.cast(f,ctypes.c_void_p),ctypes.byref(info))!=0,'dladdr failed')
            return Path(os.fsdecode(info.name)).resolve()
        origin=None;method='UMATH dependency-group dlsym + dladdr function ownership'
    result={}
    for base,integer in [('corename',False),('config',False),('num_threads',True),('parallel',True)]:
        options=['scipy_openblas_get_'+base+'64_','openblas_get_'+base+'64_',
                 'openblas_get_'+base,'scipy_openblas_get_'+base,'openblas_get_'+base+'_64_']
        found=None
        for name in options:
            try:found=getattr(library,name);break
            except AttributeError:continue
        require(found is not None,'missing actual BLAS metadata getter: '+base)
        actual=validate_library_origin(resolve(found),libs,records)
        if origin is None:origin=actual
        require(actual['path']==origin['path'],'BLAS getters resolve to different libraries')
        found.restype=ctypes.c_int if integer else ctypes.c_char_p;value=found()
        result[base]=int(value) if integer else value.decode('ascii')
        symbols[base]=dict(name=name,address=hex(ctypes.cast(found,ctypes.c_void_p).value),owner=actual['path'])
    require(result['num_threads']==1,'reference BLAS is not single-threaded')
    return dict(**result,library=origin,symbols=symbols,resolution=method,integer_abi='ILP64' if 'USE64BITINT' in result['config'] else 'LP64')


MODULE = r'''
#include <pybind11/pybind11.h>
#include "skimagexx/image.hpp"
namespace py=pybind11;
void bind_matrix_transform(py::module_&);
PYBIND11_MODULE(_matrix_math_core,m) {
 bind_matrix_transform(m);
 m.def("get_num_threads",&skimagexx::get_num_threads);
 m.def("set_num_threads",&skimagexx::set_num_threads);
 m.def("has_avx2",&skimagexx::has_avx2);
#ifdef _OPENMP
 m.attr("has_openmp")=true;
#else
 m.attr("has_openmp")=false;
#endif
#ifdef NDEBUG
 m.attr("gil_assertions")=false;
#else
 m.attr("gil_assertions")=true;
#endif
}
'''
CMAKE = r'''
cmake_minimum_required(VERSION 3.20)
project(matrix_default_discovery LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
add_library(matrix_runtime STATIC ${SOURCE}/src/matrix_transform.cpp ${SOURCE}/src/simd_matrix_transform_avx2.cpp ${SOURCE}/src/image.cpp ${SOURCE}/src/simd_avx2.cpp)
set_target_properties(matrix_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(matrix_runtime PUBLIC ${SOURCE}/include PRIVATE ${SOURCE}/src)
target_compile_definitions(matrix_runtime PUBLIC SKIMAGEXX_AVX2=1)
target_link_libraries(matrix_runtime PUBLIC OpenMP::OpenMP_CXX)
pybind11_add_module(_matrix_math_core NO_EXTRAS module.cpp ${SOURCE}/src/bind_matrix_transform.cpp)
target_link_libraries(_matrix_math_core PRIVATE matrix_runtime)
add_executable(matrix_native ${SOURCE}/tests/test_matrix_transform_native.cpp)
target_link_libraries(matrix_native PRIVATE matrix_runtime)
foreach(t matrix_runtime _matrix_math_core matrix_native)
 if(MSVC)
  target_compile_options(${t} PRIVATE /fp:strict /Gy /Gw)
  target_link_options(${t} PRIVATE /OPT:REF)
 else()
  target_compile_options(${t} PRIVATE -ffp-contract=off -fno-fast-math -march=x86-64 -mno-avx -mno-avx2 -mno-fma -ffunction-sections -fdata-sections)
  target_link_options(${t} PRIVATE -Wl,--gc-sections)
 endif()
endforeach()
if(MSVC)
 set_source_files_properties(${SOURCE}/src/simd_matrix_transform_avx2.cpp ${SOURCE}/src/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:strict")
else()
 set_source_files_properties(${SOURCE}/src/simd_matrix_transform_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
 set_source_files_properties(${SOURCE}/src/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
endif()
'''


def windows_generator(capabilities,instances):
    available={int(m.group(1)):g['name'] for g in capabilities['generators']
               if (m:=re.match(r'^Visual Studio (\d+) ',g['name']))}
    choices=[]
    for item in instances:
        major=int(item['installationVersion'].split('.')[0])
        if major in available and item.get('isComplete') is True and item.get('isLaunchable') is True:
            choices.append((tuple(map(int,item['installationVersion'].split('.'))),available[major],item['installationPath']))
    require(choices,'no matching installed Visual Studio C++ instance/generator')
    _,name,path=max(choices)
    return ['-G',name,'-A','x64','-DCMAKE_GENERATOR_INSTANCE='+path]


def hosted():
    require(os.environ.get('GITHUB_ACTIONS')=='true','native preparation/discovery is hosted-only; no local build/image trial')


def env_python(folder):return Path(folder)/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def cmake_path_definitions(source,python,pybind):
    # Only CMake's three path-valued definitions are encoded here. Raw process,
    # source and tool identity records retain the observed platform paths.
    # CMake4 FindPython EVAL otherwise treats Windows \a/\U as escapes.
    return ['-D'+name+'='+str(value).replace('\\','/') for name,value in
            [('SOURCE',source),('Python_EXECUTABLE',python),('pybind11_DIR',pybind)]]


def prepare(source,revision,profile,out):
    hosted();state,blobs=source_state(source,revision,True);validate_default_environment(os.environ)
    require('.'.join(map(str,sys.version_info[:2]))==PROFILES[profile]['python'],'wrong host Python')
    require((os.name=='nt')==(profile=='windows'),'wrong profile platform')
    require(platform.machine().lower() in ('amd64','x86_64'),'x64 required')
    out.mkdir(parents=True,exist_ok=False);journal=Journal(out);config=pinned_config(blobs,profile)
    dump(out/'request.json',dict(source_root=str(source),source_revision=revision,profile=profile,
         public_acceptance=False,workflow_run=os.environ.get('GITHUB_RUN_ID')))
    dump(out/'source.json',state);dump(out/'pins.json',config)
    require(not git(ROOT,'status','--porcelain','--untracked-files=all').strip(),'controller checkout dirty')
    controller_revision=git(ROOT,'rev-parse','HEAD').decode().strip()
    controller={n:git(ROOT,'show',controller_revision+':'+n) for n in CONTROLLER_PATHS}
    require(all(canonical((ROOT/n).read_bytes())==canonical(b) for n,b in controller.items()),'controller physical bytes differ')
    dump(out/'controller.json',dict(revision=controller_revision,raw_git_members={n:digest(b) for n,b in controller.items()}))
    for name,raw in blobs.items():
        p=out/'input-source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    for name,raw in controller.items():
        p=out/'input-controller'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    dump(out/'bootstrap.json',dict(python=sys.version,executable=file_record(sys.executable),
        base=file_record(getattr(sys,'_base_executable',sys.executable)),environment={k:os.environ.get(k) for k in ENV_KEYS}))
    for name in ('build-env','runtime-env'):
        venv.EnvBuilder(with_pip=True).create(out/name)
    for name,pins,known in [('build',config['build'],None),('runtime',config['runtime'],config['runtime_wheels'])]:
        folder=out/(name+'-wheels');folder.mkdir()
        py=env_python(out/(name+'-env'))
        journal.run('download-'+name,[py,'-m','pip','download','--only-binary=:all:','--no-deps','--dest',folder,*[n+'=='+v for n,v in pins.items()]],environment())
        rows=verify_wheelhouse(folder,pins,known);dump(out/(name+'-wheels.json'),rows)
        journal.run('install-'+name,[py,'-m','pip','install','--no-index','--no-deps','--find-links',folder,*[n+'=='+v for n,v in pins.items()]],environment())
    harness=out/'harness';harness.mkdir();(harness/'module.cpp').write_text(MODULE);(harness/'CMakeLists.txt').write_text(CMAKE)
    buildpy=env_python(out/'build-env');scripts=buildpy.parent
    cmake=scripts/('cmake.exe' if os.name=='nt' else 'cmake')
    cmakedir=journal.run('pybind-cmake-dir',[buildpy,'-m','pybind11','--cmakedir'],environment()).decode().strip()
    options=['-G','Ninja']
    if os.name=='nt':
        caps=json.loads(journal.run('cmake-capabilities',[cmake,'-E','capabilities'],environment()))
        vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
        instances=json.loads(journal.run('visual-studio-instances',[vswhere,'-all','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-format','json'],environment()))
        dump(out/'windows-toolchain-selection.json',dict(capabilities=caps,instances=instances))
        options=windows_generator(caps,instances)
    buildenv=environment();buildenv['PATH']=str(scripts)+os.pathsep+buildenv['PATH']
    journal.run('configure-private',[cmake,*options,'-S',harness,'-B',out/'native',
        *cmake_path_definitions(source,buildpy,cmakedir),
        '-DCMAKE_BUILD_TYPE='+PROFILES[profile]['config']],buildenv)
    journal.run('build-private',[cmake,'--build',out/'native','--config',PROFILES[profile]['config'],'--parallel','2'],buildenv)
    extension='.pyd' if os.name=='nt' else '.so'
    modules=list((out/'native').rglob('_matrix_math_core*'+extension));require(len(modules)==1,'require one real module')
    native=list((out/'native').rglob('matrix_native.exe' if os.name=='nt' else 'matrix_native'));require(len(native)==1,'require one native control')
    dump(out/'build.json',dict(module=file_record(modules[0]),native=file_record(native[0]),
        compiler_scope='Tiny seven-TU private closure, no full product or public registration',source=state))
    # Save real compiler/backend identity and disassembly without claiming that
    # empirical candidate agreement supplies the missing vendor source audit.
    cache=(out/'native'/'CMakeCache.txt').read_text(errors='replace')
    tools={str(cmake):file_record(cmake)}
    for line in cache.splitlines():
        if re.match(r'CMAKE_(CXX_COMPILER|LINKER|AR|RANLIB):FILEPATH=',line):
            p=Path(line.split('=',1)[1])
            if p.is_file():tools[str(p)]=file_record(p)
    dump(out/'build-tools.json',tools)
    if os.name!='nt':
        objdump=shutil.which('objdump');require(objdump,'objdump unavailable')
        dump(out/'disassembler.json',file_record(objdump))
        journal.run('module-disassembly',[objdump,'-d','-C',modules[0]],environment())
    else:
        # CMake records the selected actual VS instance; find its matching
        # installed dumpbin rather than assuming a Visual Studio version.
        instance=Path(options[-1].split('=',1)[1])
        bins=sorted(instance.glob('VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe'))
        require(bins,'selected VS instance has no x64 dumpbin')
        dump(out/'disassembler.json',file_record(bins[-1]))
        journal.run('module-disassembly',[bins[-1],'/DISASM',modules[0]],environment())
    py=env_python(out/'runtime-env')
    install_context(profile,out,journal)
    failures=[]
    for mode in ('normal','scalar'):
        for label,argv in (
            ('native-controls',[native[0]]),
            ('discover',[py,Path(__file__).resolve(),'_child','--profile',profile,'--source-root',source,
                '--source-revision',revision,'--output',out,'--mode',mode]),
            ('authenticated-pipeline',[py,Path(__file__).resolve(),'_pipeline','--profile',profile,'--source-root',source,
                '--source-revision',revision,'--output',out,'--mode',mode])):
            try:journal.run(label+'-'+mode,argv,environment(mode))
            except Exception as error:failures.append(dict(mode=mode,scope=label,error=str(error)))
    dump(out/'terminal-scopes.json',dict(failures=failures,all_commands_terminal=True))
    source_state(source,revision,True)
    require(not failures,'one or more private runtime scopes failed; all raw commands retained')


def pipeline(source,revision,profile,out,mode):
    """Require natural admission BEFORE running unchanged private tests."""
    hosted();source_state(source,revision,True);validate_default_environment(os.environ)
    import numpy as np
    import skimagexx as xx
    build=json.loads((out/'build.json').read_text());modulepath=Path(build['module']['path'])
    require(sha(modulepath)==build['module']['sha256'],'private module changed')
    backend=load_module(modulepath)
    require(backend.matrix_transform_v2_interface=='matrix-transform-math-v2' and backend.matrix_transform_v2_capabilities==63,'strict v2 interface missing')
    require(inspect.isbuiltin(backend.matrix_transform_run_v2) and backend.matrix_transform_run_v2.__module__==backend.__name__,'v2 is not owned builtin')
    require(backend.get_num_threads()==1 and xx._core is not backend,'one private worker owner / distinct genuine context required')
    site=Path(np.__file__).resolve().parent.parent
    require(Path(xx.__file__).resolve().parent==site/'skimagexx','package origin shadowed')
    context=json.loads((out/'context.json').read_text())
    for name,row in context['project_records'].items():
        if name.startswith('skimagexx/'):
            p=site/name
            require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'accepted package context bytes changed')
    spec=importlib.util.spec_from_file_location('skimagexx._matrix_runtime_probe',source/'python/skimagexx/_matrix_transform.py')
    shell=importlib.util.module_from_spec(spec);spec.loader.exec_module(shell);shell._core=backend
    version=shell._runtime_profile()
    metadata=dict(numpy=np.__version__,actual_corename=shell._RUNTIME_CORENAME,natural_runtime_certificate=version,
        source_revision=revision,private_module=file_record(modulepath),donor_core=file_record(xx._core.__file__),
        donor_package=file_record(xx.__file__),private_default_workers=backend.get_num_threads(),
        environment={k:os.environ.get(k) for k in ENV_KEYS},public_acceptance=False,
        scope='Actual default runtime, frozen strict v2 private module; genuine accepted donor package context only')
    dump(out/('pipeline-runtime-'+mode+'.json'),metadata)
    require(version==np.__version__ and shell._checked_entry(6) is backend.matrix_transform_run_v2,'actual runtime is unknown/interposed/unverifiable; provider-only results cannot pass')
    env=environment(mode);env['MATRIX_TRANSFORM_PRIVATE_BACKEND']=str(modulepath)
    env['MATRIX_TRANSFORM_PROOF_OUTPUT']=str(out/('pipeline-routes-'+mode+'.json'))
    Journal(out).run('full-pipeline-tests-'+mode,[sys.executable,'-m','pytest',
        source/'tests/test_matrix_transform_native_pipeline.py',source/'tests/test_matrix_transform_binding.py',
        '--capture=sys','-q','-o','cache_dir='+str(out/('pytest-cache-'+mode)),
        '--basetemp',out/('pytest-temp-'+mode),'--junitxml',out/('pipeline-'+mode+'.xml')],env)
    dump(out/('pipeline-counts-'+mode+'.json'),validate_pipeline(out,mode))


def layout(np,a,name):
    if name=='C':return np.array(a,order='C')
    if name=='F':return np.array(a,order='F')
    if name=='reverse':return a[::-1].copy()[::-1]
    if name=='step':
        b=np.zeros((a.shape[0]*2,a.shape[1]*2));b[::2,::2]=a;return b[::2,::2]
    require(name=='unaligned','unknown fixed layout')
    b=np.zeros(a.nbytes+1,np.uint8);c=np.ndarray(a.shape,dtype='f8',buffer=b,offset=1);c[:]=a;return c


def blas_layout(h,m):
    def flags(a):
        r,c=a.shape;s,t=a.strides
        return dict(shape=list(a.shape),strides=list(a.strides),aligned=bool(a.flags.aligned),
            C_blasable=t==8 and s%8==0 and s//8>=c,F_blasable=s==8 and t%8==0 and t//8>=r)
    a,b=flags(h),flags(m.T)
    return dict(padded_H=a,matrix_T=b,m=h.shape[0],k=h.shape[1],n=m.shape[0],alpha=1,beta=0,
        source_branch='GEMV-special' if h.shape[0]==1 else 'GEMM-or-source-copy/noblas',
        numpy_CBLAS_transpose_A='N' if a['C_blasable'] else 'T' if a['F_blasable'] else None,
        numpy_CBLAS_transpose_B='N' if b['C_blasable'] else 'T' if b['F_blasable'] else None,
        caution='Unaligned iterator buffers/copy order and vendor SMALL_MATRIX_OPT require source audit; no guessed internal layout.')


def load_module(path):
    spec=importlib.util.spec_from_file_location('_matrix_math_core',path)
    require(spec and spec.loader,'no native extension loader')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(inspect.isbuiltin(module.matrix_transform_run) and module.matrix_transform_run.__module__==module.__name__,'not a genuine owned builtin')
    require(module.matrix_transform_interface=='matrix-transform-math-v1' and module.matrix_transform_capabilities==31,'wrong real private interface')
    require(module.get_num_threads()==1 and module.has_openmp,'wrong default/shared worker policy')
    return module


def child(source,revision,profile,out,mode):
    hosted();validate_default_environment(os.environ);state,blobs=source_state(source,revision,True)
    pins=pinned_config(blobs,profile);build=json.loads((out/'build.json').read_text())
    modulepath=Path(build['module']['path']);require(sha(modulepath)==build['module']['sha256'],'module changed')
    # Runtime is installed in its own fresh venv, not source/python or a product facade.
    require(Path(sys.prefix).resolve()==(out/'runtime-env').resolve(),'not the exact runtime environment')
    for name,version in pins['runtime'].items():require(importlib.metadata.version(name)==version,'wrong runtime package: '+name)
    import numpy as np
    import skimage.transform._geometric as original
    site=Path(np.__file__).parent.parent
    records=installed_records(out/'runtime-wheels',site)
    prefix='numpy._core' if np.__version__.startswith('2.') else 'numpy.core'
    umath=importlib.import_module(prefix+'._multiarray_umath')
    require(str(Path(umath.__file__).resolve()) in records,'UMATH not authenticated to runtime wheel')
    reference=json.loads(blobs['coverage/contracts/matrix-transform.json'])
    key=('windows' if os.name=='nt' else 'linux')+'-np'+np.__version__+'-py'+'.'.join(map(str,sys.version_info[:2]))
    facts=reference['profiles'][key]['source']['source_utf8']
    for name,text in facts.items():
        f=getattr(original.ProjectiveTransform,name.split('.')[1]) if name.startswith('ProjectiveTransform.') else getattr(original,name)
        require(inspect.getsource(f)==text,'original source differs: '+name)
    blas=blas_facts(np,umath,records)
    capture=io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(capture):np.show_config()
    (out/('numpy-config-'+mode+'.stdout')).write_text(capture.getvalue(),encoding='utf-8')
    if Path('/proc/cpuinfo').is_file():
        (out/('cpuinfo-'+mode+'.stdout')).write_bytes(Path('/proc/cpuinfo').read_bytes())
    module=load_module(modulepath);require(module.gil_assertions==(profile=='debug'),'Debug GIL assertion scope mismatch')
    require(module.has_avx2()==(mode=='normal' and bool(umath.__cpu_features__.get('AVX2'))),'SIMD policy mismatch')
    spec=cohort_spec(blobs['tests/test_matrix_transform_native.cpp'])
    dump(out/('runtime-'+mode+'.json'),dict(schema=SCHEMA,python=sys.version,executable=file_record(sys.executable),
        base=file_record(getattr(sys,'_base_executable',sys.executable)),numpy=np.__version__,
        umath=file_record(umath.__file__),blas=blas,cpu_features=umath.__cpu_features__,platform=platform.uname()._asdict(),
        environment={k:os.environ.get(k) for k in ENV_KEYS},numpy_error_policy=np.geterr(),numpy_bufsize=np.getbufsize(),
        source=state,source_evidence='Frozen exact CPP and original0.26 source verified. Vendor-dispatch arithmetic audit remains pending.',
        private_genuine_CPP=True,public_acceptance=False,profile_admission='pending-source-tree-audit'))
    counts={str(i):dict(completed=0,exact=0,different=0,rejected=0) for i in range(1,6)}
    rows=0
    with gzip.open(out/('rows-'+mode+'.jsonl.gz'),'wt',encoding='utf-8') as stream:
        for workers in (1,4):
            module.set_num_threads(workers);require(module.get_num_threads()==workers,'global worker state differs')
            for item in case_specs(spec):
                case=item['case'];requested=item['n']
                points=np.array(case['coords']) if requested==0 else np.array([
                    np.array(case['coords'][i%len(case['coords'])])+i*item['extension'] for i in range(requested)])
                points=layout(np,points,item['points_layout']);matrix=layout(np,np.array(case['matrix'],dtype='f8'),item['matrix_layout'])
                before=(points.tobytes(),matrix.tobytes())
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter('always');expected=original.matrix_transform(points,matrix)
                h=np.hstack((points,np.ones((len(points),1))))
                row=dict(id=case['id'],requested_rows=requested,workers=workers,mode=mode,
                    points_layout=item['points_layout'],matrix_layout=item['matrix_layout'],
                    expected_words=expected.view('u8').tolist(),expected_dtype=expected.dtype.str,expected_shape=list(expected.shape),
                    warnings=[dict(category=type(w.message).__name__,message=str(w.message)) for w in notices],
                    source_operands=blas_layout(h,matrix),candidates=[])
                for candidate in range(1,6):
                    initial=module.matrix_transform_completed();actual,status=module.matrix_transform_run(points,matrix,candidate)
                    completed=module.matrix_transform_completed()-initial;require(completed==(status==0),'false native completion counter')
                    values=actual.view('u8').tolist() if status==0 else None
                    exact=values==row['expected_words'] if status==0 else False
                    count=counts[str(candidate)];count['completed']+=completed
                    count['exact' if exact else 'different' if status==0 else 'rejected']+=1
                    row['candidates'].append(dict(profile=candidate,status=status,completed=completed,
                        words=values,exact=exact,output=None if status else dict(dtype=actual.dtype.str,
                        shape=list(actual.shape),strides=list(actual.strides),owned=bool(actual.flags.owndata),
                        C=bool(actual.flags.c_contiguous)),admission='UNACCEPTED-arithmetic-candidate'))
                require(before==(points.tobytes(),matrix.tobytes()),'inputs mutated')
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');rows+=1
        module.set_num_threads(1)
    require(rows==8000,'incomplete fixed corpus')
    require(sha(modulepath)==build['module']['sha256'],'module mutated during discovery')
    require(installed_records(out/'runtime-wheels',site)==records,'runtime source/library bytes changed')
    require(blas_facts(np,umath,records)==blas,'loaded BLAS identity/configuration changed')
    dump(out/('runtime-members-'+mode+'.json'),records)
    dump(out/('result-'+mode+'.json'),dict(schema=SCHEMA,rows=rows,counts=counts,source=state,profile=profile,
        original_cases=4000,workers=[1,4],operation_timings=0,public_acceptance=False,
        profile_admission='pending-source-tree-audit',default_cpu=blas['corename'],
        decision='Candidate agreement is empirical research only; no production profile is selected or accepted.'))


def validate_discovery(result):
    require(result['schema']==SCHEMA and result['original_cases']==4000 and result['rows']==8000,'incomplete discovery cases')
    require(result['workers']==[1,4] and result['operation_timings']==0,'wrong workers/timing scope')
    require(result['public_acceptance'] is False and result['profile_admission']=='pending-source-tree-audit','discovery cannot be promoted to acceptance')
    validate_source_records(result['source']['canonical_members'])
    require(result['source']['revision']==SOURCE_REVISION,'source evidence revision differs')
    require(set(result['counts'])==set(map(str,range(1,6))),'incomplete candidate set')
    for count in result['counts'].values():
        require(set(count)=={'completed','exact','different','rejected'} and all(type(v) is int and v>=0 for v in count.values()),'invalid candidate counts')
        require(count['exact']+count['different']==count['completed'],'invalid completion accounting')
        require(count['completed']+count['rejected']==8000,'candidate rows omitted')


def row_key(row):
    return (row['id'],row['requested_rows'],row['points_layout'],row['matrix_layout'],row['workers'])


def validate_rows(path,result,spec,mode):
    expected={(i['case']['id'],i['n'],i['points_layout'],i['matrix_layout'],w)
              for i in case_specs(spec) for w in (1,4)}
    require(len(expected)==8000,'invalid fixed corpus identity closure')
    seen=set();counts={str(i):dict(completed=0,exact=0,different=0,rejected=0) for i in range(1,6)}
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);key=row_key(row)
            require(key in expected and key not in seen,'unexpected or duplicated scientific row');seen.add(key)
            require(row['mode']==mode,'wrong process mode')
            require([c['profile'] for c in row['candidates']]==list(range(1,6)),'candidate set differs')
            for c in row['candidates']:
                require(c['admission']=='UNACCEPTED-arithmetic-candidate','candidate silently accepted')
                require(type(c['status']) is int and 0<=c['status']<=7,'invalid native status')
                ok=c['status']==0;require(c['completed']==int(ok),'false native completion')
                exact=ok and c['words']==row['expected_words']
                require(c['exact'] is exact,'misreported raw numerical equality')
                require(ok or c['words'] is None,'rejection carries fabricated result')
                count=counts[str(c['profile'])];count['completed']+=int(ok)
                count['exact' if exact else 'different' if ok else 'rejected']+=1
    require(seen==expected,'scientific row closure differs')
    require(counts==result['counts'],'summary differs from retained raw rows')


def seal(out):
    """Copy only durable research evidence; never whole environments/caches."""
    artifact=out/'artifact';artifact.mkdir(exist_ok=False)
    source=Path(json.loads((out/'request.json').read_text())['source_root'])
    verify_context_files(out,json.loads((out/'request.json').read_text())['profile'])
    spec=cohort_spec(git(source,'show',SOURCE_REVISION+':tests/test_matrix_transform_native.cpp'))
    for mode in ('normal','scalar'):
        validate_pipeline(out,mode)
        result=json.loads((out/('result-'+mode+'.json')).read_text());validate_discovery(result)
        validate_rows(out/('rows-'+mode+'.jsonl.gz'),result,spec,mode)
    build=json.loads((out/'build.json').read_text())
    for key in ('module','native'):
        p=Path(build[key]['path']);require(sha(p)==build[key]['sha256'],'compiled bytes changed')
        shutil.copy2(p,artifact/p.name)
    for p in out.iterdir():
        if p.is_file() and (p.suffix in ('.json','.stdout','.stderr','.gz','.xml')):shutil.copy2(p,artifact/p.name)
    shutil.copytree(out/'context',artifact/'context')
    for folder in ('harness',):shutil.copytree(out/folder,artifact/folder)
    controller=json.loads((out/'controller.json').read_text())
    for name,want in controller['raw_git_members'].items():
        b=git(ROOT,'show',controller['revision']+':'+name);require(digest(b)==want,'controller Git bytes changed')
        p=artifact/'controller'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    # Source allowlist is copied from authenticated Git content, not a stale ZIP.
    state=json.loads((out/'source.json').read_text())
    for name in SOURCE_HASHES:
        p=artifact/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(git(source,'show',SOURCE_REVISION+':'+name))
    for name in ('CMakeCache.txt','compile_commands.json','build.ninja'):
        p=out/'native'/name
        if p.is_file():shutil.copy2(p,artifact/name)
    for p in (out/'native').rglob('*'):
        if p.is_file() and p.suffix in ('.o','.obj','.a','.lib'):
            dest=artifact/'objects'/p.relative_to(out/'native');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Actual Windows command/read tlogs; GNU dependency and command journals.
    if os.name=='nt':
        for p in (out/'native').rglob('*.tlog'):
            if p.is_file():
                dest=artifact/'tlogs'/p.relative_to(out/'native');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    else:
        ninja=env_python(out/'build-env').parent/'ninja'
        for name in ('commands','deps'):
            data=subprocess.check_output([str(ninja),'-C',str(out/'native'),'-t',name])
            (artifact/('ninja-'+name+'.txt')).write_bytes(data)
    records={p.relative_to(artifact).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in artifact.rglob('*') if p.is_file()}
    dump(artifact/'manifest.json',dict(schema=SCHEMA,records=records,source=state,public_acceptance=False,
        profile_admission='pending-source-tree-audit',operation_timings=0,
        retention='Exact private module/native controls/source/runtime-wheel identities/all scientific words/journals; environments and build caches stay ephemeral. Runtime wheels reference frozen known hashes, not duplicate bytes.',
        all_commands_terminal=True))
    return artifact


def verify(artifact):
    artifact=Path(artifact)
    m=json.loads((artifact/'manifest.json').read_text());require(m['public_acceptance'] is False and m['profile_admission']=='pending-source-tree-audit','fake acceptance')
    require(m['schema']==SCHEMA and m['operation_timings']==0 and m['all_commands_terminal'] is True,'wrong manifest scope')
    validate_source_records(m['source']['canonical_members'])
    require(not any(p.is_symlink() for p in artifact.rglob('*')),'artifact symlink not permitted')
    actual={p.relative_to(artifact).as_posix() for p in artifact.rglob('*') if p.is_file()}-{'manifest.json','verification.json'}
    require(actual==set(m['records']),'artifact closure differs')
    for name,row in m['records'].items():
        p=(artifact/name).resolve();require(p.is_relative_to(artifact.resolve()),'artifact path escape')
        require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'artifact raw bytes differ')
    validate_source_records({n:digest(canonical((artifact/'source'/n).read_bytes())) for n in SOURCE_HASHES})
    controller=json.loads((artifact/'controller.json').read_text())
    validate_controller_members(artifact/'controller',controller['raw_git_members'])
    verify_context_files(artifact,json.loads((artifact/'request.json').read_text())['profile'])
    spec=cohort_spec((artifact/'source/tests/test_matrix_transform_native.cpp').read_bytes())
    for mode in ('normal','scalar'):
        validate_pipeline(artifact,mode)
        result=json.loads((artifact/('result-'+mode+'.json')).read_text());validate_discovery(result)
        validate_rows(artifact/('rows-'+mode+'.jsonl.gz'),result,spec,mode)
    commands=[json.loads(p.read_text()) for p in artifact.glob('command-*.json')]
    require(commands and all(c.get('returncode')==0 for c in commands),'nonterminal/failed command')
    return dict(schema=SCHEMA,manifest_sha256=sha(artifact/'manifest.json'),records=len(actual),verified=True,
                public_acceptance=False,profile_admission='pending-source-tree-audit')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','all','verify','_child','_pipeline'])
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--source-revision',default=os.environ.get('MATRIX_SOURCE_REVISION'))
    parser.add_argument('--profile',choices=PROFILES,default='modern')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['normal','scalar'])
    args=parser.parse_args(argv);out=args.output.resolve();source=args.source_root.resolve()
    require(out.is_relative_to((ROOT/'build').resolve()),'output must be in owned ignored build directory')
    if args.stage=='verify':
        print(json.dumps(verify(out)));return
    if args.stage=='plan':
        state,blobs=source_state(source,args.source_revision)
        print(json.dumps(dict(schema=SCHEMA,source=state,pins=pinned_config(blobs,args.profile),
            cases=len(list(case_specs(cohort_spec(blobs['tests/test_matrix_transform_native.cpp'])))),
            public_acceptance=False,operation_timings=0,execution='plan only; all requires clean exact checkout and hosted grant context'),indent=2));return
    if args.stage in ('_child','_pipeline'):
        require(args.mode is not None,'missing explicit process mode')
        (child if args.stage=='_child' else pipeline)(source,args.source_revision,args.profile,out,args.mode);return
    request=dict(source_root=str(source),source_revision=args.source_revision,profile=args.profile,
                 public_acceptance=False,workflow_run=os.environ.get('GITHUB_RUN_ID'))
    try:
        prepare(source,args.source_revision,args.profile,out)
        dump(out/'request.json',request)
        artifact=seal(out);receipt=verify(artifact);dump(artifact/'verification.json',receipt);print(json.dumps(receipt))
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        dump(out/'failure.json',dict(request=request,type=type(error).__name__,message=str(error),public_acceptance=False))
        raise


if __name__=='__main__':main()
