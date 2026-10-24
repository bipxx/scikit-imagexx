"""TV-Chambolle reference corpus and future strictly gated public benchmark.

Phase1 is original-only/native0. A staged stencil capability is not admission
of the public denoiser. All330 original IDs remain untimed and immutable.
"""
import argparse
from dataclasses import dataclass
import gzip
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import traceback
import uuid
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import ci_measurement_artifacts as artifacts
from benchmarks.provenance import source_fingerprint
from benchmarks.run import measure
from benchmarks.resize import cpu_metadata,source_revision,summarize

MANIFEST=ROOT/'coverage/contracts/tv-chambolle-benchmark.json'
SOURCE_CONTRACT=ROOT/'coverage/contracts/tv-chambolle.json'
API='restoration.denoise_tv_chambolle'
VERSION='tv-chambolle-benchmark-v1'
LIMIT=8*1024*1024
_contracts=None


def contracts():
    global _contracts
    if _contracts is None:
        spec=importlib.util.spec_from_file_location('tv_benchmark_frozen_contract',ROOT/'tests/test_tv_chambolle_contracts.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_contracts=module
    return _contracts


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def profile():
    import numpy as np
    return 'windows' if os.name=='nt' else 'minimum' if np.__version__.split('.')[0]=='1' else 'modern'


def resolve(original=False):
    return importlib.import_module(('skimage' if original else 'skimagexx')+'.restoration').denoise_tv_chambolle


def public_admission(manifest):
    """Run before contract imports/fixtures/capsules/timers; no guessed cap."""
    import skimagexx as xx
    coverage=importlib.import_module('skimagexx.coverage')
    if API not in coverage.native_functions():raise RuntimeError('actual public TV native path is unregistered; Phase1 native0')
    accepted=manifest.get('accepted_public')
    if not isinstance(accepted,dict):raise RuntimeError('independently qualified whole public TV admission pending; stage-only capabilities are insufficient')
    products=manifest.get('accepted_products',{})
    if not all(isinstance(products.get(p),dict) for p in ('windows','modern','minimum')):raise RuntimeError('all3 independently qualified public TV product pins required before fixture admission')
    required={'wrapper_module','interface_attribute','interface','capability_attribute','capability','entries','qualified_source','qualified_proof_sha256'}
    assert set(accepted)==required and re.fullmatch('[0-9a-f]{40}',accepted['qualified_source']) and re.fullmatch('[0-9a-f]{64}',accepted['qualified_proof_sha256'])
    for pin in products.values():
        assert set(pin)=={'revision','qualification_sha256','wheel_sha256','core_sha256','source_sha256'}
        assert pin['revision']==accepted['qualified_source']
        assert all(re.fullmatch('[0-9a-f]{64}',pin[k]) for k in pin if k!='revision')
    core=xx._core;wrapper=importlib.import_module(accepted['wrapper_module']);function=resolve()
    assert artifacts.sha(Path(core.__file__))==products[profile()]['core_sha256']
    assert source_fingerprint(ROOT)['source_sha256']==products[profile()]['source_sha256']
    assert wrapper._core is core and getattr(wrapper,'denoise_tv_chambolle') is function
    assert function.__module__==accepted['wrapper_module'] and inspect.unwrap(function).__globals__.get('_core') is core
    assert function is not resolve(True) and inspect.signature(function)==inspect.signature(resolve(True))
    assert getattr(core,accepted['interface_attribute'],None)==accepted['interface']
    cap=getattr(core,accepted['capability_attribute'],None)
    assert type(cap) is int and type(accepted['capability']) is int and cap==accepted['capability']
    assert accepted['entries'] and len(set(accepted['entries']))==len(accepted['entries'])
    for name in accepted['entries']:
        entry=getattr(core,name,None);assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==core.__name__
    assert xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads
    assert xx.get_num_threads()==1
    for workers in (4,1):xx.set_num_threads(workers);assert core.get_num_threads()==workers
    assert xx.set_fallback_enabled(False) is False,'native benchmark requires caller fallback already disabled'
    assert not any(k.startswith('SKIMAGEXX_') and k!='SKIMAGEXX_DISABLE_SIMD' for k in os.environ),'private selectors forbidden'
    return accepted


@dataclass(frozen=True)
class Case:
    id:str
    fixture:dict
    timed:bool
    source_id:str|None=None


def timing_profiles():
    rows=[]
    for size,shape in [('tiny',[3,5]),('small',[33,35]),('raster',[257,259]),('volume',[9,33,35])]:
        for dtype in ('u1','f4','f8'):
            for stopping in ('one','two','fixed8','converge'):
                index=len(rows)
                rows.append(dict(id=f'{size}/{dtype}/{stopping}',shape=shape,dtype=dtype,layout=('C','F','step','readonly')[index%4],pattern='constant' if stopping=='converge' and dtype=='f4' else 'ramp',weight=0.1 if dtype=='u1' else 0.5,eps=0.0002 if stopping=='converge' else 0.,iterations={'one':1,'two':2,'fixed8':8,'converge':200}[stopping],cohort=size))
    for axis,shape in ((0,[3,51,53]),(-1,[51,53,3])):
        for dtype in ('u1','f4','f8'):
            for count in (1,8):rows.append(dict(id=f'channel/{axis}/{dtype}/i{count}',shape=shape,axis=axis,dtype=dtype,layout='step' if count==8 else 'C',pattern='ramp',weight=0.1,eps=0.,iterations=count,cohort='channels'))
    for layout in ('reverse','broadcast','endian','unaligned','tagged'):
        for dtype in ('f4','f8'):rows.append(dict(id=f'layout/{layout}/{dtype}',shape=[33,35],dtype=dtype,layout=layout,pattern='ramp',weight=0.5,eps=0.,iterations=8,cohort='layout'))
    assert len(rows)==70 and len({r['id'] for r in rows})==70
    return rows


def corpus():
    qualified=contracts()
    result=[Case('source/'+p['id'],dict(p),False,p['id']) for p in qualified.CASES]
    result.extend(Case('raster/'+p['id'],p,True) for p in timing_profiles())
    assert len(result)==400 and sum(c.timed for c in result)==70
    return tuple(result)


def inventory(case):
    return dict(id=case.id,api=API,fixture=case.fixture,timed=case.timed,source_id=case.source_id,classification='defined-finite-raster' if case.timed else 'frozen-reference-untimed',cohort=case.fixture.get('cohort','source-contract'))


def fixture(case):
    import numpy as np
    if not case.timed:return contracts().fixture(case.fixture)
    # Bounded workspace certificate covers image, conversion/output, p/g/d and
    # arithmetic temporaries. It is setup metadata, never an operation timer.
    shape=case.fixture['shape'];spatial=len(shape)-(1 if 'axis' in case.fixture else 0)
    assert math.prod(shape)*8*(4*spatial+7)<=LIMIT
    return contracts().fixture(case.fixture)


def case_spec(case):
    passed,image,owner,options,events=fixture(case)
    return dict(**inventory(case),input=contracts().array(image),owner=contracts().array(owner))


def observe(function,case):
    if not case.timed:return contracts().observe(function,case.fixture)
    # The same qualified observer captures return alias/base/strides/dtypes and
    # mutations as the frozen small corpus, without array snapshots in timing.
    return contracts().observe(function,case.fixture)


def comparison(value):return contracts().result_nan_encoding_contract(value)


def native_observe(function,case):
    """Detect exact original Python/Cython identities during untimed native calls."""
    original=resolve(True);module=importlib.import_module('skimage.restoration._denoise')
    forbidden={inspect.unwrap(original).__code__,module._denoise_tv_chambolle_nd.__code__}
    previous=sys.getprofile()
    def guard(frame,event,arg):
        if event=='call' and frame.f_code in forbidden:raise AssertionError('upstream TV algorithm forwarding detected')
        if event=='c_call' and (arg is original or arg is module._denoise_tv_chambolle_nd):raise AssertionError('upstream TV compiled forwarding detected')
        if previous is not None:previous(frame,event,arg)
    try:sys.setprofile(guard);return observe(function,case)
    finally:sys.setprofile(previous)


def timing_callback(function,case):
    assert case.timed and case.source_id is None
    passed,image,owner,options,events=fixture(case)
    assert not events and all(isinstance(v,(int,float)) for k,v in options.items() if k!='channel_axis')
    before=dict(image=contracts().array(image),owner=contracts().array(owner))
    def call():return function(passed,**options)
    return call,lambda:dict(image=contracts().array(image),owner=contracts().array(owner))==before


def save(value,path):
    if path is None:return
    path.parent.mkdir(parents=True,exist_ok=True);body=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
    path.write_bytes(gzip.compress(body,mtime=0) if str(path).endswith('.gz') else body)


def verify_record(wheel):
    import base64,csv,io
    with zipfile.ZipFile(wheel) as archive:
        names=[n for n in archive.namelist() if not n.endswith('/')];assert len(names)==len(set(names))
        record,=[n for n in names if n.endswith('.dist-info/RECORD')]
        rows=list(csv.reader(io.StringIO(archive.read(record).decode())));assert {r[0] for r in rows}==set(names) and len(rows)==len(names)
        for name,encoded,size in rows:
            if name==record:assert not encoded and not size
            else:
                body=archive.read(name);assert encoded=='sha256='+base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b'=').decode() and int(size)==len(body)


def capsule_tree(capsule):
    assert not any(p.is_symlink() for p in capsule.rglob('*'))
    return {p.relative_to(capsule).as_posix():artifacts.sha(p) for p in capsule.rglob('*') if p.is_file() and p.name!='tv-chambolle-artifacts.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def seal(capsule):
    manifest=json.loads(MANIFEST.read_text());public_admission(manifest)
    path=capsule/'tv-chambolle-artifacts.json';assert not path.exists()
    qualification=json.loads((capsule/'qualification.json').read_text());pin=manifest['accepted_products'][profile()]
    assert artifacts.sha(capsule/'qualification.json')==pin['qualification_sha256']
    assert qualification['passed'] is True and not qualification['failures'] and qualification['source']['revision']==pin['revision']
    assert API in qualification['expected_registry']['functions'] and qualification['profile']==profile()
    runtime=artifacts.freeze(ROOT,capsule/'wheels',capsule/'tv-chambolle-runtime.json')
    assert any(artifacts.sha(p)==pin['wheel_sha256'] for p in (capsule/'wheels').glob('*.whl'))
    with zipfile.ZipFile(capsule/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in runtime['source']['files_sha256']:archive.write(ROOT/name,name)
    info=dict(schema='tv-chambolle-exact-offline-v1',runtime_manifest_sha256=artifacts.sha(capsule/'tv-chambolle-runtime.json'),corpus_manifest_sha256=artifacts.sha(MANIFEST),product_source=source_fingerprint(ROOT),files_sha256=capsule_tree(capsule))
    path.write_text(json.dumps(info,indent=2)+'\n');return verify(capsule)


def verify(capsule):
    info=json.loads((capsule/'tv-chambolle-artifacts.json').read_text());assert info['schema']=='tv-chambolle-exact-offline-v1'
    assert info['files_sha256']==capsule_tree(capsule) and info['corpus_manifest_sha256']==artifacts.sha(MANIFEST) and info['product_source']==source_fingerprint(ROOT)
    assert info['runtime_manifest_sha256']==artifacts.sha(capsule/'tv-chambolle-runtime.json')
    runtime=artifacts.verify(ROOT,capsule/'wheels',capsule/'tv-chambolle-runtime.json');assert len(runtime['runtime']['sdk_sha256'])==59
    with zipfile.ZipFile(capsule/'source.zip') as archive:
        assert set(archive.namelist())==set(runtime['source']['files_sha256'])
        assert all(hashlib.sha256(archive.read(n).replace(b'\r\n',b'\n')).hexdigest()==h for n,h in runtime['source']['files_sha256'].items())
    for wheel in (capsule/'wheels').glob('*.whl'):
        verify_record(wheel)
        with zipfile.ZipFile(wheel) as archive:
            if 'skimagexx/__init__.py' not in archive.namelist():continue
            headers={n for n in archive.namelist() if n.startswith('include/skimagexx/') and not n.endswith('/')}
            assert headers=={p.relative_to(ROOT).as_posix() for p in (ROOT/'include/skimagexx').rglob('*') if p.is_file()}
            assert all(archive.read(n).replace(b'\r\n',b'\n')==(ROOT/n).read_bytes().replace(b'\r\n',b'\n') for n in headers)
            actual={n.removeprefix('skimagexx/'):archive.read(n).replace(b'\r\n',b'\n') for n in archive.namelist() if n.startswith('skimagexx/') and n.endswith('.py')}
            expected={p.relative_to(ROOT/'python/skimagexx').as_posix():p.read_bytes().replace(b'\r\n',b'\n') for p in (ROOT/'python/skimagexx').rglob('*.py')}
            assert actual==expected
    return dict(manifest_sha256=artifacts.sha(capsule/'tv-chambolle-artifacts.json'),**info)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',default=[1]);parser.add_argument('--repeat',type=int,default=7);parser.add_argument('--target-ms',type=float,default=15.)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--reference-only',action='store_true');parser.add_argument('--prepare-capsule',action='store_true');parser.add_argument('--capsule',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not args.threads or len(set(args.threads))!=len(args.threads) or 1 not in args.threads or min(args.threads)<1 or max(args.threads)>1024 or args.repeat!=7 or not math.isfinite(args.target_ms) or args.target_ms<=0:parser.error('distinct positive workers including1; seven rounds and positive target-ms required')
    if args.reference_only and (not args.check_only or args.capsule):parser.error('reference-only requires check-only without capsule')
    if args.prepare_capsule and (not args.check_only or args.reference_only or not args.capsule or set(args.threads)!={1,4}):parser.error('preparation requires check-only native capsule and workers1/4')
    if not args.check_only and (os.environ.get('GITHUB_ACTIONS')!='true' or args.capsule is None or args.output is None or args.threads!=[1]):parser.error('timing requires isolated Actions capsule/output, one native worker only')
    import skimagexx as xx
    import numpy as np
    import scipy,skimage
    previous=xx.get_num_threads();policy=xx.set_fallback_enabled(False)
    manifest=json.loads(MANIFEST.read_text());source=source_fingerprint(ROOT);manifest_sha=artifacts.sha(MANIFEST)
    report=dict(version=VERSION,status='preflight',scope='reference-only/native0' if args.reference_only else 'strict actual public TV',metadata=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,source_revision=source_revision(ROOT),source=source,workers=args.threads,primary_workers=1,repeat=7,check_only=args.check_only,reference_only=args.reference_only,session_id=str(uuid.uuid4()),cpu=cpu_metadata(),timing_scope='Whole public call including conversion/allocation/result destruction; fixture construction excluded; errors/protocols/NaN cases untimed'),checks=[],results=[],failures=[])
    save(report,args.output)
    try:
        assert skimage.__version__=='0.26.0'
        if not args.reference_only:public_admission(manifest)
        assert text_sha(SOURCE_CONTRACT)==manifest['source_contract_sha256']
        assert all(text_sha(ROOT/n)==h for n,h in manifest['fixture_source_sha256'].items())
        qualified=contracts();frozen=json.loads(SOURCE_CONTRACT.read_text())
        assert str(inspect.signature(resolve(True)))==frozen['signature'] and qualified.source_sha(resolve(True))==frozen['source_sha256']
        assert qualified.source_sha(importlib.import_module('skimage.restoration._denoise')._denoise_tv_chambolle_nd)==frozen['helper_sha256']
        cases=corpus();assert [inventory(c) for c in cases]==manifest['case_inventory'] and digest(manifest['case_inventory'])==manifest['case_contract_sha256']
        specs=[case_spec(c) for c in cases];expected_profile=manifest['reference_profiles'].get(profile())
        if expected_profile:assert digest(specs)==expected_profile['case_spec_sha256']
        report['metadata'].update(case_spec_sha256=digest(specs),case_contract_sha256=manifest['case_contract_sha256'],case_specs=specs,profile=profile())
        if not args.reference_only:
            assert expected_profile is not None and args.capsule is not None
            report['metadata']['artifacts']=seal(args.capsule) if args.prepare_capsule else verify(args.capsule)
        for index,case in enumerate(cases):
            row=dict(index=index,**inventory(case))
            try:
                expected=observe(resolve(True),case);row['reference']=expected
                if case.source_id:assert comparison(expected)==comparison(frozen['observations'][frozen['profiles'][profile()]['snapshots'][case.source_id]])
                if expected_profile:assert digest(comparison(expected))==expected_profile['observations'][str(index)]
                if case.timed:
                    assert expected['error'] is None and not expected['warnings'] and not expected['events']
                    assert expected['before']==expected['after'] and expected['pointer_stable']
                    arr=expected['result']['array'];values=np.frombuffer(bytes.fromhex(arr['bytes']),np.dtype(arr['dtype']));assert np.isfinite(values).all()
                row['actual_by_workers']={}
                if not args.reference_only:
                    for worker in args.threads:
                        xx.set_num_threads(worker);actual=native_observe(resolve(),case);row['actual_by_workers'][str(worker)]=actual
                        assert comparison(actual)==comparison(expected),'exact public observation mismatch'
                row['status']='passed'
            except Exception as error:
                row.update(status='failed',type=type(error).__name__,message=str(error),traceback=traceback.format_exc());report['failures'].append(row)
            report['checks'].append(row)
            if (index+1)%32==0 or row['status']=='failed':save(report,args.output)
        report['metadata'].update(case_count=len(cases),timing_eligible=70,native_checks=0 if args.reference_only else len(cases)*len(args.threads),algorithms=1,public_paths=[API],aliases=[])
        if report['failures']:report['status']='failed-parity';return 1
        def integrity():
            assert source_fingerprint(ROOT)==source and artifacts.sha(MANIFEST)==manifest_sha
            if not args.reference_only:assert verify(args.capsule)==report['metadata']['artifacts']
        integrity()
        if args.check_only:report['status']='complete-reference-check' if args.reference_only else 'complete-parity-check';return 0
        # Complete original+both-worker validation occurs before entering here.
        # The fresh measurement host performs no builds/pytest and onlyworker1.
        report['status']='measuring';xx.set_num_threads(1)
        for case in cases:
            if not case.timed:continue
            left,unchanged_left=timing_callback(resolve(True),case);right,unchanged_right=timing_callback(resolve(),case)
            with warnings.catch_warnings():warnings.simplefilter('ignore');timing=measure(left,right,7,args.target_ms)
            assert unchanged_left() and unchanged_right()
            report['results'].append(dict(id=case.id,cohort=case.fixture['cohort'],threads=1,slower_than_original=timing['speedup']<1,**timing));save(report,args.output)
        integrity();report['summary']=summarize(report['results'],[1],70);report['status']='complete-measurement';return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'],type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));report['status']='failed';return 1
    finally:xx.set_num_threads(previous);xx.set_fallback_enabled(policy);save(report,args.output)


if __name__=='__main__':raise SystemExit(main())
