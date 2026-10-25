"""Public199 max-tree artifact admission. Standard-library only; no pin invention."""
from __future__ import annotations
import hashlib,importlib.util,json,os,re,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROFILES=('windows','modern','minimum')
RUNTIME_SOURCE_PINS={
    'src/max_tree.cpp':'1808c8581b465a96a5d5ee9cccd73ee269d17893877784933230173221f1e135',
    # a567 public header: authenticated comment-only evolution from private7d662.
    'include/skimagexx/max_tree.hpp':'60b33407a721e57b3629e671f0f822ab1dd344ccbaddfcf526a9e3f93ca9bd1c',
    'src/simd_max_tree_avx2.cpp':'2bdb72ade563581161038b8cf32260c6330e028e938590c7a382dc7c5fa92de4',
    'src/max_tree_stages.hpp':'8afefbbdfe06282b5524607c4a5a831ceaeddb59cf75047e33fbbaabee125184',
    'src/max_tree_stages.cpp':'d1961c7d0b4dd0940a765e76e7fb6f64aee8edfcfb3255c69b4e030f3b42d9e6',
    'src/bind_max_tree.cpp':'f4764b61180293ab98e474d68b40b214774ff367fd07e0ca4034c3705bbee2c5',
    # c98ec8e: closing allocation's canonical scalar dtype drops source tags.
    'python/skimagexx/_max_tree.py':'3519902c42187e3f79ebe1bf5e17bdf05b562518e773fee05d743f79160db3ae'}
ACCEPTED_PRODUCT_PINS=None
NAMES=('max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing')
FAMILIES=('components','component_consumers','metrics','footprint_dense','rank','rank_distribution',
          'rank_intensity','rank_alias','projective','rank_statistics','consumer_layout','rank_percentile',
          'rank_bilateral','rank_decisions','tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree','matrix_transform')
SCOPE=dict(image_paths=199,metadata_paths=3,sdk_families=20,sdk_members=65,sdk_family_names=list(FAMILIES))
PINS={'benchmarks/max_tree.py':'4b2419f98bd282585dbc041c090230653c421b38706f83ce3a6d7460a6547d13',
      'coverage/contracts/max-tree-benchmark.json':'09c9d36b548940d25dcd73cd9e59d5b5d8440bb4ae99ca1a0bdb404a8bb163bf',
      'coverage/contracts/max-tree.json':'45d5999d49f3b1a5cdb8c7ec5767de470e74b158f5fb489f288132722edc6365',
      'benchmarks/rank_statistics_compare.py':'286c616d3b502fe38f5db6d614aa699bab1d1b920635a2e185adfba5eac949bf',
      'tools/ci_measurement_artifacts.py':'104ba954d9e9b0cc809a21b8d37f59599b29777d88a7f5b818f7ad5565d5c6b7'}
PIN_FIELDS={'revision','qualification_sha256','product_sha256','wheel_sha256','source_sha256','scope','artifact_run','artifact_name'}


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return path


def frozen_sources():
    for name,value in PINS.items():require(text_sha(ROOT/name)==value,'frozen source changed: '+name)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def validator():
    frozen_sources();return module('_max_tree_complete_product_validator',ROOT/'benchmarks/rank_statistics_compare.py')


def certificate(path,expected_sha,controller_revision=None):
    require(re.fullmatch('[0-9a-f]{64}',expected_sha or '') and sha(path)==expected_sha,'exact external certificate SHA required')
    value=read(path)
    require(type(value) is dict and set(value)=={'schema','status','source','run','controller_revision','baseline','profiles','max_tree_runtime_source_sha256'},'certificate fields differ')
    require(value['schema']=='max-tree-public-product-v1' and value['status']=='root-accepted-qualified-public199','genuine public199 root certificate pending')
    require(isinstance(value['source'],str) and re.fullmatch('[0-9a-f]{40}',value['source']) and isinstance(value['run'],str) and value['run'].isdigit(),'exact genuine public source/run required')
    require(re.fullmatch('[0-9a-f]{40}',value['controller_revision'] or ''),'controller revision absent')
    if controller_revision is not None:require(value['controller_revision']==controller_revision,'certificate/controller source differs')
    require(value['baseline']=={name:PINS[name] for name in ('benchmarks/max_tree.py','coverage/contracts/max-tree-benchmark.json','coverage/contracts/max-tree.json')},'immutable351/252 source pins differ')
    require(value['max_tree_runtime_source_sha256']==RUNTIME_SOURCE_PINS,'frozen dc12/29/f650 max-tree runtime source differs')
    require(set(value['profiles'])==set(PROFILES),'all3 qualified profiles required')
    for profile,frame in value['profiles'].items():
        require(type(frame) is dict and set(frame)=={'pin','artifact','core_sha256','python_version'},'complete profile frame required')
        pin=frame['pin'];require(type(pin) is dict and set(pin)==PIN_FIELDS,'exact accepted product pin fields required')
        require(pin['revision']==value['source'] and pin['artifact_run']==value['run'] and pin['artifact_name']=='rank-statistics-product-'+profile,'product source/run/name differs')
        require(pin['scope']==SCOPE and all(type(pin['scope'][k]) is int for k in ('image_paths','metadata_paths','sdk_families','sdk_members')),'public199/SDK20 scope differs')
        for key in ('qualification_sha256','product_sha256','wheel_sha256','source_sha256'):require(re.fullmatch('[0-9a-f]{64}',pin[key] or ''),'product hash absent')
        require(re.fullmatch('[0-9a-f]{64}',frame['core_sha256'] or ''),'core hash absent')
        require(re.fullmatch(r'3\.'+('11' if profile=='minimum' else '13')+r'\.[0-9]+',frame['python_version'] or ''),'exact profile interpreter version required')
        artifact=frame['artifact'];require(type(artifact) is dict and set(artifact)=={'id','run','name','digest'},'raw artifact provenance absent')
        require(isinstance(artifact['id'],str) and artifact['id'].isdigit() and artifact['run']==value['run'] and artifact['name']==pin['artifact_name'] and re.fullmatch('[0-9a-f]{64}',artifact['digest'] or ''),'raw artifact source/name/digest differs')
    require(len({f['artifact']['id'] for f in value['profiles'].values()})==3,'distinct qualified profile artifacts required')
    return value


def controller_revision():
    result=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    require(subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--'],check=False).returncode==0,'tracked controller source changed')
    return result


def verify_product(path,profile,cert):
    """Complete authentic original product, before numerical/fixture imports."""
    require(profile in PROFILES,'wrong product profile');frame=cert['profiles'][profile];pin=frame['pin'];path=Path(path)
    c=validator();cfg=read(ROOT/'coverage/contracts/rank-statistics-comparison.json')
    cfg['accepted_products']={'before':{profile:pin}}
    result=c.verify_product(path,'before',profile,cfg)
    q=read(path/'qualification.json');package=read(path/'qualification-evidence/package.json')
    require(q['source']['revision']==cert['source'] and q['expected_registry']['image_count']==199 and q['expected_registry']['metadata_count']==3,'qualified registry counts differ')
    require(all(q['expected_registry']['functions'].count('morphology.'+n)==1 for n in NAMES),'qualified six max-tree paths missing')
    require(package['wheel_sha256']==pin['wheel_sha256'] and len(package['members_sha256'])==114,'whole114-member package required')
    require(read(path/'qualification-evidence/independent-verification.json')==dict(passed=True,manifest_sha256=pin['qualification_sha256'],records_rehashed=len(q['records'])),'independent qualification verification absent')
    product=read(path/'product.json');p=product['provenance']
    require(p['original_run']==cert['run'] and p['original_manifest_sha256']==pin['qualification_sha256'] and p['original_artifact']=='public-qualification-'+profile,'actual qualification provenance differs')
    for mode in ('normal','scalar'):
        row=q['counts'][mode+'-public-MaxTree']
        require(type(row['tests']) is int and row['tests']>=2*(351+233) and not any(row[k] for k in ('failures','errors','skipped')),'actual public-MaxTree scope incomplete/skipped')
    for name,h in RUNTIME_SOURCE_PINS.items():require(text_sha(path/'source'/name)==h,'qualified max-tree runtime source differs: '+name)
    runtime=read(path/'runtime.json')
    cores={name:value for name,value in runtime['runtime']['packages_sha256']['skimagexx'].items() if name.startswith('_core.') and name.endswith(('.pyd','.so'))}
    require(list(cores.values())==[frame['core_sha256']],'sole actual core hash differs')
    require(len(runtime['runtime']['sdk_sha256'])==65,'complete SDK65 required')
    return dict(result,core_sha256=frame['core_sha256'],public_group={m:q['counts'][m+'-public-MaxTree'] for m in ('normal','scalar')})


def verify_installed(path):
    frozen_sources();m=module('_max_tree_offline_runtime_validator',ROOT/'tools/ci_measurement_artifacts.py')
    runtime=read(Path(path)/'runtime.json')
    require(m.installed_state(Path(path)/'wheels',runtime['wheels'])==runtime['runtime'],'installed wheels/package/SDK/.libs/runtime differ')
    return runtime


def validate_receipt(row,certificate_sha,run,attempt):
    require(type(row) is dict and row.get('schema')=='max-tree-preparation-v1' and row.get('passed') is True,'failed/reference-only preparation')
    require(row.get('profile') in PROFILES and row.get('certificate_sha256')==certificate_sha and row.get('generation')==str(run) and row.get('attempt')==str(attempt),'stale/wrong-run certificate preparation')
    require(row.get('modes')==['normal','scalar'] and row.get('workers')==[1,4] and all(type(v) is int for v in row['workers']) and row.get('counts')=={'reference':351,'numeric':233,'metadata':118,'candidates':252},'partial original/native worker preparation')
    for name in ('report_sha256','product_sha256','controller_sha256'):require(isinstance(row.get(name),str) and re.fullmatch('[0-9a-f]{64}',row[name]),'preparation closure absent')
    return row


def barrier(paths,certificate_sha,run,attempt,cert):
    require(len(paths)==3,'ALL3 successful fresh preparation receipts required before fixtures/timers');rows={}
    for path in paths:
        row=validate_receipt(read(path),certificate_sha,run,attempt);profile=row['profile']
        require(profile not in rows,'duplicate preparation profile')
        require(row['product_sha256']==cert['profiles'][profile]['pin']['product_sha256'],'prepared product differs')
        report_path=Path(path).parent/'report.json';require(sha(report_path)==row['report_sha256'],'preparation report changed')
        rows[profile]=row
    require(set(rows)==set(PROFILES),'missing preparation profile');return rows


def claim(paths,certificate_sha,run,attempt,cert,profile,nonce):
    rows=barrier(paths,certificate_sha,run,attempt,cert)
    path=Path(os.environ['RUNNER_TEMP'])/('max-tree-'+str(run)+'-'+str(attempt)+'-'+profile+'.json')
    save(path,dict(schema='max-tree-measurement-claim-v1',profile=profile,generation=str(run),attempt=str(attempt),certificate_sha256=certificate_sha,
         receipt_sha256=[sha(p) for p in paths],controller_sha256=sha(ROOT/'benchmarks/max_tree_compare.py'),nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),controller_pid=os.getpid()))
    return path,rows


def retain_claim(claim,output):
    """Preserve raw bytes without moving/replacing the global once-only anchor."""
    claim=Path(claim);raw=claim.read_bytes();expected=hashlib.sha256(raw).hexdigest()
    target=Path(output)/'measurement-claim.json'
    with target.open('xb') as stream:stream.write(raw)
    require(target.read_bytes()==raw and sha(target)==sha(claim)==expected,'raw measurement claim changed during retention')
    return dict(anchor=str(claim.resolve()),path=target.name,sha256=expected,bytes=len(raw))


def fetch(frame,output):
    """Prepare host only. Stream exact outer ZIP to disk; never rebuild wheels."""
    require(os.environ.get('GITHUB_ACTIONS')=='true','download restricted to preparation host')
    output=Path(output);require(not output.exists(),'fresh product directory required');artifact=frame['artifact']
    run=json.loads(subprocess.check_output(['gh','api','repos/bipxx/scikit-imagexx/actions/runs/'+artifact['run']]))
    require(run['head_sha']==frame['pin']['revision'] and run['status']=='completed' and run['conclusion']=='success','genuine whole qualification run must finish successfully')
    meta=json.loads(subprocess.check_output(['gh','api','repos/bipxx/scikit-imagexx/actions/artifacts/'+artifact['id']]))
    require(meta['workflow_run']['head_sha']==frame['pin']['revision'] and str(meta['workflow_run']['id'])==artifact['run'] and meta['name']==artifact['name'] and not meta['expired'] and meta['digest']=='sha256:'+artifact['digest'],'actual artifact source/name/run/digest differs')
    require(type(meta['size_in_bytes']) is int and 0<meta['size_in_bytes']<=1600000000,'bounded product artifact size required')
    output.parent.mkdir(parents=True,exist_ok=True);archive=output.with_suffix('.outer.zip')
    with archive.open('xb') as stream:
        p=subprocess.run(['gh','api',meta['archive_download_url'].split('api.github.com/')[-1]],stdout=stream,check=True)
    require(archive.stat().st_size==meta['size_in_bytes'] and sha(archive)==artifact['digest'],'outer artifact bytes differ')
    output.mkdir();c=validator()
    with zipfile.ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        require(len(files)==len({i.filename for i in files}) and sum(i.file_size for i in files)<=2500000000,'duplicate/unbounded archive members')
        for info in files:
            c.safe_name(info.filename);require((info.external_attr>>16)&0o170000!=0o120000,'artifact symlink forbidden')
            target=output/info.filename;require(target.resolve().is_relative_to(output.resolve()),'artifact path escape')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('xb') as dest:
                for part in iter(lambda:source.read(1024*1024),b''):dest.write(part)
    return dict(raw_zip_sha256=sha(archive),bytes=archive.stat().st_size,members=len(files),CRC_checked=True)
