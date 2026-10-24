"""TV-Chambolle corpus and qualified-artifact-only public benchmark.

Tracked acceptance pins remain NULL until a reviewed external certificate and
complete genuine products are supplied. Stage capabilities alone cannot admit
the denoiser. All330 original IDs and70 timing fixtures remain immutable.
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


def public_admission(manifest,qualified=None):
    """Actual public guard; certificate/product validation precedes fixtures."""
    import skimagexx as xx
    coverage=importlib.import_module('skimagexx.coverage')
    if API not in coverage.native_functions():raise RuntimeError('actual public TV native path is unregistered')
    if not isinstance(qualified,dict):raise RuntimeError('independently qualified public TV admission pending; static NULL pins and stage-only capabilities cannot admit it')
    core=xx._core;wrapper=importlib.import_module('skimagexx._tv_chambolle');function=resolve()
    assert artifacts.sha(Path(core.__file__))==qualified['core_sha256']
    assert source_fingerprint(ROOT)['source_sha256']==qualified['product_source_sha256']
    assert wrapper._core is core and wrapper.denoise_tv_chambolle is function
    assert inspect.isfunction(function) and function.__module__=='skimagexx._tv_chambolle'
    assert inspect.getmodule(function) is wrapper and function.__globals__.get('_core') is core
    assert function is not resolve(True) and inspect.signature(function)==inspect.signature(resolve(True))
    for attribute,version,capability,expected in (
        ('tv_chambolle_interface','tv-chambolle-stages-v1','tv_chambolle_capabilities',63),
        ('tv_chambolle_finite_interface','tv-chambolle-finite-v1','tv_chambolle_finite_capabilities',3)):
        assert getattr(core,attribute,None)==version
        cap=getattr(core,capability,None);assert type(cap) is int and cap==expected
    assert getattr(core,'tv_chambolle_finite_profiles',None)==('1.24.4','2.5.4')
    for name in ('tv_chambolle_divergence_into','tv_chambolle_gradient_into','tv_chambolle_norm_into','tv_chambolle_dual_into','tv_chambolle_finite_run'):
        entry=getattr(core,name,None);assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==core.__name__
    assert xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads
    assert xx.get_num_threads()==1
    for workers in (4,1):xx.set_num_threads(workers);assert core.get_num_threads()==workers
    assert xx.set_fallback_enabled(False) is False
    assert all(os.environ.get(name)=='1' for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')),'reference provider pool limits must be1'
    assert not any(k.startswith('SKIMAGEXX_') and k!='SKIMAGEXX_DISABLE_SIMD' for k in os.environ),'private selectors forbidden'
    return qualified


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
    previous=sys.getprofile();attempted=[]
    def guard(frame,event,arg):
        if event=='call' and frame.f_code in forbidden:
            attempted.append('python');raise AssertionError('upstream TV algorithm forwarding detected')
        if event=='c_call' and (arg is original or arg is module._denoise_tv_chambolle_nd):
            attempted.append('compiled');raise AssertionError('upstream TV compiled forwarding detected')
        if previous is not None:previous(frame,event,arg)
    try:sys.setprofile(guard);actual=observe(function,case)
    finally:sys.setprofile(previous)
    assert not attempted,'upstream TV forwarding detected (including caught attempts)'
    return actual


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


PROFILES=('windows','modern','minimum')
TV_SCOPE=dict(image_paths=191,metadata_paths=3,sdk_families=17,sdk_members=62)


def certificate(path,run=None,revision=None):
    value=json.loads(Path(path).read_text())
    assert set(value)=={'schema','status','qualified_run','qualified_source','profiles'}
    assert value['schema']=='tv-qualified-product-certificate-v1' and value['status']=='accepted-qualified-tv'
    assert isinstance(value['qualified_run'],str) and value['qualified_run'].isdigit()
    assert re.fullmatch('[0-9a-f]{40}',value['qualified_source'])
    if run is not None:assert value['qualified_run']==run
    if revision is not None:assert value['qualified_source']==revision
    assert set(value['profiles'])==set(PROFILES)
    for name,pin in value['profiles'].items():
        assert set(pin)=={'revision','qualification_sha256','product_sha256','wheel_sha256','source_sha256','core_sha256','product_source_sha256','scope','artifact_run','artifact_name','artifact_id','artifact_digest','python_version'}
        assert pin['revision']==value['qualified_source'] and pin['artifact_run']==value['qualified_run']
        assert pin['artifact_name']=='rank-statistics-product-'+name
        assert isinstance(pin['artifact_id'],str) and pin['artifact_id'].isdigit()
        assert re.fullmatch(r'3\.(11|13)\.[0-9]+',pin['python_version'])
        assert pin['python_version'].split('.')[1]==('11' if name=='minimum' else '13')
        for key in ('qualification_sha256','product_sha256','wheel_sha256','source_sha256','core_sha256','product_source_sha256','artifact_digest'):assert re.fullmatch('[0-9a-f]{64}',pin[key])
        assert set(pin['scope'])==set(TV_SCOPE)|{'sdk_family_names'}
        assert all(type(pin['scope'][k]) is int and pin['scope'][k]==v for k,v in TV_SCOPE.items())
        families=pin['scope']['sdk_family_names'];assert type(families) is list and len(families)==len(set(families))==17
        assert families[-3:]==['tv_chambolle','tv_reductions','tv_denoise']
    assert len({p['artifact_id'] for p in value['profiles'].values()})==3,'distinct profile artifacts required'
    return value


def controller_identity(qualified_root,cert):
    """Keep full qualified product source and reviewed controller source distinct."""
    qualified_root=Path(qualified_root).resolve(strict=True)
    assert qualified_root!=ROOT.resolve(), 'separate exact qualified-source checkout required'
    policy_path=ROOT/'coverage/contracts/tv-chambolle-controller.json'
    policy=json.loads(policy_path.read_text())
    allowed={'benchmarks/tv_chambolle.py','tests/test_tv_chambolle_benchmark.py',
             '.github/workflows/tv-chambolle-benchmark.yml','coverage/contracts/tv-chambolle-controller.json'}
    assert set(policy)=={'schema','qualified_source','allowed_controller_paths','product_source_sha256','native_source_members'}
    assert policy['schema']=='tv-qualified-controller-v1' and set(policy['allowed_controller_paths'])==allowed
    assert len(policy['allowed_controller_paths'])==len(allowed)
    assert policy['qualified_source']==cert['qualified_source']
    qualified=artifacts.source_state(qualified_root);controller=artifacts.source_state(ROOT)
    expected=os.environ['TV_CONTROLLER_SOURCE'];assert re.fullmatch('[0-9a-f]{40}',expected)
    assert controller['revision']==expected and controller['revision']!=qualified['revision']
    assert qualified['revision']==cert['qualified_source']
    before=qualified['files_sha256'];after=controller['files_sha256']
    delta={name:dict(qualified=before.get(name),controller=after.get(name))
           for name in sorted(set(before)|set(after)) if before.get(name)!=after.get(name)}
    assert set(delta)==allowed, 'controller changes outside exact reviewed four-path delta'
    def native_names(files):
        return {name for name in files if name in ('CMakeLists.txt','pyproject.toml')
                or name.startswith(('src/','include/skimagexx/')) and name.count('/')==(2 if name.startswith('include/') else 1)
                or name.startswith('python/skimagexx/') and name.count('/')==2 and name.endswith('.py')}
    names=native_names(before);assert names==native_names(after) and len(names)==policy['native_source_members']==255
    assert all(before[name]==after[name] for name in names)
    native=source_fingerprint(qualified_root);assert native==source_fingerprint(ROOT)
    assert native['source_sha256']==policy['product_source_sha256']
    assert all(pin['product_source_sha256']==native['source_sha256'] for pin in cert['profiles'].values())
    return dict(schema='tv-controller-source-separation-v1',qualified_source=qualified,
                controller_source=controller,delta=delta,native_source=native,
                policy_source_text_sha256=text_sha(policy_path))


def qualified_product(capsule,certificate_path,run=None,revision=None):
    """Standard-library artifact checks before any fixture or image operation."""
    from benchmarks import rank_statistics_compare as controller
    cert=certificate(certificate_path,run,revision);pin=cert['profiles'][profile()];product=capsule/'product'
    cfg=json.loads((ROOT/'coverage/contracts/rank-statistics-comparison.json').read_text())
    keys={'revision','qualification_sha256','product_sha256','wheel_sha256','source_sha256','scope','artifact_run','artifact_name'}
    cfg['accepted_products']={'before':{profile():{k:pin[k] for k in keys}}}
    verified=controller.verify_product(product,'before',profile(),cfg)
    q=json.loads((product/'qualification.json').read_text())
    assert API in q['expected_registry']['functions']
    qualified_root=Path(os.environ['TV_QUALIFIED_SOURCE_ROOT']).resolve(strict=True)
    controller_info=controller_identity(qualified_root,cert)
    assert q['source']==controller_info['qualified_source'],'complete qualified source index differs'
    assert source_fingerprint(ROOT)['source_sha256']==pin['product_source_sha256']
    provenance=json.loads((product/'product.json').read_text())['provenance']
    assert provenance['original_run']==cert['qualified_run'] and provenance['original_manifest_sha256']==pin['qualification_sha256']
    assert provenance['original_artifact']=='public-qualification-'+profile()
    verified_receipt=json.loads((product/'qualification-evidence/independent-verification.json').read_text())
    assert verified_receipt==dict(passed=True,manifest_sha256=pin['qualification_sha256'],records_rehashed=len(q['records']))
    package=json.loads((product/'qualification-evidence/package.json').read_text())
    assert package['wheel_sha256']==pin['wheel_sha256'] and len(package['members_sha256'])==108
    # Qualification must genuinely execute the installed public TV scope in both modes.
    for mode in ('normal','scalar'):
        row=q['counts'][mode+'-public-TV']
        assert row['tests']>=2148 and not any(row[k] for k in ('failures','errors','skipped'))
    runtime=artifacts.verify(qualified_root,product/'wheels',product/'runtime.json')
    assert len(runtime['runtime']['sdk_sha256'])==62
    cores={n:h for n,h in runtime['runtime']['packages_sha256']['skimagexx'].items() if n.startswith('_core.') and n.endswith(('.so','.pyd'))}
    assert list(cores.values())==[pin['core_sha256']]
    return dict(**pin,profile=profile(),certificate_sha256=artifacts.sha(certificate_path),verified_product=verified,controller=controller_info)


def preparation_reports(directory):
    rows={}
    for mode in ('normal','scalar'):
        path=directory/(mode+'.json');row=json.loads(path.read_text())
        assert row['status']=='complete-parity-check' and not row['failures'] and not row['results']
        m=row['metadata'];assert m['check_only'] is True and m['reference_only'] is False
        assert m['workers']==[1,4] and m['case_count']==400 and m['timing_eligible']==70 and m['native_checks']==800
        assert m['simd_mode']==mode and m['case_contract_sha256']==json.loads(MANIFEST.read_text())['case_contract_sha256']
        inventory=json.loads(MANIFEST.read_text())['case_inventory']
        assert len(row['checks'])==len(inventory)==400
        assert all(r['index']==i and {k:r[k] for k in spec}==spec for i,(r,spec) in enumerate(zip(row['checks'],inventory)))
        assert all(r['status']=='passed' and set(r['actual_by_workers'])=={'1','4'} for r in row['checks'])
        import xml.etree.ElementTree as ET
        suites=ET.parse(directory/('pipeline-'+mode+'.xml')).getroot().findall('testsuite')
        assert suites and sum(int(s.get('tests','0')) for s in suites)>=2148
        assert all(not int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped'))
        cases=[c for suite in suites for c in suite.findall('testcase')]
        assert len(cases)==sum(int(s.get('tests','0')) for s in suites) and len({c.get('name') for c in cases})==len(cases)
        defaults=[c for c in cases if c.get('name','').startswith('test_actual_default_finite_loop_once_per_spatial_channel[')]
        assert len(defaults)==632,'all316 default finite projections at workers1/4 required'
        compiled=0
        for case in defaults:
            props={p.get('name'):p.get('value') for p in case.findall('properties/property')}
            assert props.get('actual_default')=='True'
            loops=json.loads(props['actual_compiled_finite_loops']);stages=json.loads(props['actual_compiled_stage_calls'])
            if 'layout/tagged/' in case.get('name',''):
                assert not loops and stages.get('tv_chambolle_norm_into',0)>0 and stages.get('tv_chambolle_dual_into',0)>0
            else:assert loops and not stages;compiled+=1
        assert compiled==628,'tagged4 retain stages; other628 default rows must execute real finite loop'
        rows[mode]=row
    assert rows['normal']['metadata']['artifacts']==rows['scalar']['metadata']['artifacts']
    assert rows['normal']['metadata']['case_spec_sha256']==rows['scalar']['metadata']['case_spec_sha256']
    return rows


def seal_preparation(capsule,certificate_path,directory,receipt_output,run,revision):
    import shutil
    assert os.environ.get('GITHUB_ACTIONS')=='true'
    workflow_run=os.environ['GITHUB_RUN_ID'];attempt=os.environ['GITHUB_RUN_ATTEMPT']
    assert workflow_run.isdigit() and attempt.isdigit() and not receipt_output.exists()
    qualified=qualified_product(capsule,certificate_path,run,revision);public_admission(json.loads(MANIFEST.read_text()),qualified)
    rows=preparation_reports(directory)
    assert all(r['metadata']['artifacts']==qualified for r in rows.values())
    assert not (capsule/'tv-chambolle-artifacts.json').exists() and not (capsule/'checks').exists()
    shutil.copyfile(certificate_path,capsule/'tv-certificate.json');(capsule/'checks').mkdir()
    for name in ('normal.json','scalar.json','pipeline-normal.xml','pipeline-scalar.xml'):shutil.copyfile(directory/name,capsule/'checks'/name)
    info=dict(schema='tv-chambolle-qualified-offline-v3',corpus_manifest_sha256=artifacts.sha(MANIFEST),qualified=qualified,files_sha256=capsule_tree(capsule))
    path=capsule/'tv-chambolle-artifacts.json';path.write_text(json.dumps(info,indent=2)+'\n')
    receipt=dict(schema='tv-preparation-receipt-v2',status='ready',profile=profile(),workflow_run=workflow_run,workflow_attempt=attempt,qualified_run=run,qualified_source=revision,certificate_sha256=qualified['certificate_sha256'],product_sha256=qualified['product_sha256'],capsule_sha256=artifacts.sha(path),modes=['normal','scalar'],workers=[1,4],case_count=400,timed_count=70,native_checks_per_mode=800,pipeline_default_finite_required=True,controller_revision=qualified['controller']['controller_source']['revision'],controller_sha256=digest(qualified['controller']))
    receipt_output.parent.mkdir(parents=True,exist_ok=True);receipt_output.write_text(json.dumps(receipt,indent=2)+'\n')
    return verify(capsule)


def seal(*args):raise RuntimeError('use genuine qualified product plus complete normal/scalar preparation; no wheel build/seal shortcut')


def verify(capsule):
    info=json.loads((capsule/'tv-chambolle-artifacts.json').read_text());assert info['schema']=='tv-chambolle-qualified-offline-v3'
    assert info['files_sha256']==capsule_tree(capsule) and info['corpus_manifest_sha256']==artifacts.sha(MANIFEST)
    qualified=qualified_product(capsule,capsule/'tv-certificate.json')
    assert qualified==info['qualified']
    rows=preparation_reports(capsule/'checks');assert all(r['metadata']['artifacts']==qualified for r in rows.values())
    return dict(manifest_sha256=artifacts.sha(capsule/'tv-chambolle-artifacts.json'),**info)


def preparation_barrier(directory,capsule):
    assert os.environ.get('GITHUB_ACTIONS')=='true'
    cert=certificate(capsule/'tv-certificate.json');paths=list(directory.rglob('tv-receipt-*.json'))
    sealed_controller=json.loads((capsule/'tv-chambolle-artifacts.json').read_text())['qualified']['controller']
    assert sealed_controller['controller_source']['revision']==os.environ['TV_CONTROLLER_SOURCE']
    assert len(paths)==3,'all3 preparation receipts required before fixture/timer work'
    rows={}
    for path in paths:
        row=json.loads(path.read_text());name=row['profile'];assert name in PROFILES and name not in rows
        assert row['schema']=='tv-preparation-receipt-v2' and row['status']=='ready'
        assert row['workflow_run']==os.environ['GITHUB_RUN_ID'] and row['workflow_attempt']==os.environ['GITHUB_RUN_ATTEMPT'],'stale/reused preparation receipt'
        assert row['qualified_run']==cert['qualified_run'] and row['qualified_source']==cert['qualified_source']
        assert row['controller_revision']==sealed_controller['controller_source']['revision']
        assert row['controller_sha256']==digest(sealed_controller)
        assert row['certificate_sha256']==artifacts.sha(capsule/'tv-certificate.json')
        assert row['product_sha256']==cert['profiles'][name]['product_sha256']
        assert row['modes']==['normal','scalar'] and row['workers']==[1,4]
        assert row['case_count']==400 and row['timed_count']==70 and row['native_checks_per_mode']==800 and row['pipeline_default_finite_required'] is True
        assert re.fullmatch('[0-9a-f]{64}',row['capsule_sha256']);rows[name]=row
    assert rows[profile()]['capsule_sha256']==artifacts.sha(capsule/'tv-chambolle-artifacts.json')
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',default=[1]);parser.add_argument('--repeat',type=int,default=7);parser.add_argument('--target-ms',type=float,default=15.)
    parser.add_argument('--certificate',type=Path);parser.add_argument('--qualified-run');parser.add_argument('--qualified-source');parser.add_argument('--seal-preparation',action='store_true');parser.add_argument('--preparation-dir',type=Path);parser.add_argument('--receipt-output',type=Path);parser.add_argument('--barrier',type=Path)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--reference-only',action='store_true');parser.add_argument('--prepare-capsule',action='store_true');parser.add_argument('--capsule',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not args.threads or len(set(args.threads))!=len(args.threads) or 1 not in args.threads or min(args.threads)<1 or max(args.threads)>1024 or args.repeat!=7 or not math.isfinite(args.target_ms) or args.target_ms<=0:parser.error('distinct positive workers including1; seven rounds and positive target-ms required')
    if args.reference_only and (not args.check_only or args.capsule):parser.error('reference-only requires check-only without capsule')
    if args.prepare_capsule and (not args.check_only or args.reference_only or not args.capsule or set(args.threads)!={1,4}):parser.error('preparation requires check-only native capsule and workers1/4')
    if not args.check_only and (os.environ.get('GITHUB_ACTIONS')!='true' or args.capsule is None or args.output is None or args.threads!=[1]):parser.error('timing requires isolated Actions capsule/output, one native worker only')
    if args.seal_preparation:
        if not all((args.capsule,args.certificate,args.qualified_run,args.qualified_source,args.preparation_dir,args.receipt_output)) or not args.check_only:parser.error('complete qualified preparation seal inputs required')
        seal_preparation(args.capsule,args.certificate,args.preparation_dir,args.receipt_output,args.qualified_run,args.qualified_source);return 0
    if args.prepare_capsule and not all((args.certificate,args.qualified_run,args.qualified_source)):parser.error('explicit qualified artifact certificate/run/source required')
    if not args.reference_only and not args.prepare_capsule and args.capsule is None:parser.error('qualified sealed capsule required')
    if not args.check_only and args.barrier is None:parser.error('all3 fresh preparation barrier required')
    import skimagexx as xx
    import numpy as np
    import scipy,skimage
    previous=xx.get_num_threads();policy=xx.set_fallback_enabled(False)
    manifest=json.loads(MANIFEST.read_text());source=source_fingerprint(ROOT);manifest_sha=artifacts.sha(MANIFEST)
    report=dict(version=VERSION,status='preflight',scope='reference-only/native0' if args.reference_only else 'strict actual public TV',metadata=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,source_revision=source_revision(ROOT),source=source,workers=args.threads,primary_workers=1,repeat=7,check_only=args.check_only,reference_only=args.reference_only,simd_mode="scalar" if os.environ.get("SKIMAGEXX_DISABLE_SIMD")=="1" else "normal",session_id=str(uuid.uuid4()),process_pid=os.getpid(),paired_backend_process='same process: original and current callbacks',cpu=cpu_metadata(),timing_scope='Whole public call including conversion/allocation/result destruction; fixture construction excluded; errors/protocols/NaN cases untimed'),checks=[],results=[],failures=[])
    save(report,args.output)
    try:
        assert skimage.__version__=='0.26.0'
        if not args.reference_only:
            if args.prepare_capsule:qualified=qualified_product(args.capsule,args.certificate,args.qualified_run,args.qualified_source)
            else:
                if not args.check_only:report['metadata']['preparation_barrier']=preparation_barrier(args.barrier,args.capsule)
                sealed=verify(args.capsule);qualified=sealed['qualified']
            public_admission(manifest,qualified)
        assert text_sha(SOURCE_CONTRACT)==manifest['source_contract_sha256']
        assert all(text_sha(ROOT/n)==h for n,h in manifest['fixture_source_sha256'].items())
        contract_module=contracts();frozen=json.loads(SOURCE_CONTRACT.read_text())
        assert str(inspect.signature(resolve(True)))==frozen['signature'] and contract_module.source_sha(resolve(True))==frozen['source_sha256']
        assert contract_module.source_sha(importlib.import_module('skimage.restoration._denoise')._denoise_tv_chambolle_nd)==frozen['helper_sha256']
        cases=corpus();assert [inventory(c) for c in cases]==manifest['case_inventory'] and digest(manifest['case_inventory'])==manifest['case_contract_sha256']
        specs=[case_spec(c) for c in cases];expected_profile=manifest['reference_profiles'].get(profile())
        if expected_profile:assert digest(specs)==expected_profile['case_spec_sha256']
        report['metadata'].update(case_spec_sha256=digest(specs),case_contract_sha256=manifest['case_contract_sha256'],case_specs=specs,profile=profile())
        if not args.reference_only:
            assert expected_profile is not None and args.capsule is not None
            report['metadata']['artifacts']=qualified
            report['metadata']['controller']=qualified['controller']
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
            if not args.reference_only:
                current=qualified_product(args.capsule,args.certificate,args.qualified_run,args.qualified_source) if args.prepare_capsule else verify(args.capsule)['qualified']
                assert current==report['metadata']['artifacts']
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
