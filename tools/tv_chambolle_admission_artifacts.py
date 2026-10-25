"""TV single-admission product/common-input adapter. No numerical imports or rank mutation."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,inspect,io,json,os,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROFILES=('windows','modern','minimum');ROLES=('original','before','after')
BEFORE_PRODUCT='5e068a5a2267c3a917701d5348449972fad162a7'
BEFORE_ANCHOR='0d9945460edab972f3af0942eaffaf159f6d4198'
AFTER_PRODUCT='af7b2afb8f6cd0f30f408e3318631d5c9ffafbee'
TV_TUS={'src/tv_denoise.cpp','src/bind_tv_chambolle.cpp'}
PRIVATE_HEADER='src/tv_denoise_internal.hpp'
PRIVATE_HEADER_SHA256='dfb292a70d2abce10c917b3bc054548c7447ecef712456ddd8d2832fb3dd937e'
PRIVATE_HEADER_RAW={'dfb292a70d2abce10c917b3bc054548c7447ecef712456ddd8d2832fb3dd937e':837,
                    'b7a115329dcb1d9164cfb0721f94d665c6fb9ecc7f2c821be9e7724d27cd9dd8':853}
ALLOWED_DELTA=TV_TUS|{PRIVATE_HEADER,'tests/test_tv_denoise_admission_native.cpp','tools/public_qualification.py','tests/test_public_qualification.py'}
RANK_VALIDATOR_SHA256='286c616d3b502fe38f5db6d614aa699bab1d1b920635a2e185adfba5eac949bf'
BEFORE_BRIDGE_DIGEST='2031dd3557d78f6f4ea91b8b22caa75f81cadb5bc287c4ec5964c9573e3c16a3'
ACCEPTED_PRODUCTS=None  # Only an independently reviewed external certificate can admit products.
FROZEN_FACTORY_SHA256='35b62bff494661ec417cb894507717d0f9965d9f121cfbd8f257ae92c9bba09b'

def require(value,message):
    if not value:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return path
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def revision():return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
def rank_controller():
    # A new module object, never changing the historical controller's globals.
    path=ROOT/'benchmarks/rank_statistics_compare.py';require(text_sha(path)==RANK_VALIDATOR_SHA256,'historical rank validator changed');spec=importlib.util.spec_from_file_location('_tv_product_historical_validator',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate_experiment(e,controller_revision):
    required={'schema','status','controller_revision','before_product_source','intended_before_source','after_product_source','original_version','rounds','workers','case_count','timed_count','factory_sha256','manifest_sha256','root_after_admission','products','approved_after_source_sha256','before_source_equivalence'}
    require(type(e) is dict and set(e)==required,'experiment fields differ')
    require(digest(e['before_source_equivalence'])==BEFORE_BRIDGE_DIGEST,'before source equivalence witness changed')
    require(e.get('schema')=='tv-chambolle-admission-experiment-v1' and e.get('status')=='root-reviewed','root-reviewed TV experiment pending')
    require(e.get('controller_revision')==controller_revision and re.fullmatch('[0-9a-f]{40}',controller_revision or ''),'wrong frozen controller')
    require(e.get('before_product_source')==BEFORE_PRODUCT and e.get('intended_before_source')==BEFORE_ANCHOR and e.get('after_product_source')==AFTER_PRODUCT,'source labels may not be relabeled')
    require(e.get('original_version')=='0.26.0' and type(e.get('rounds')) is int and e['rounds']==7 and type(e.get('workers')) is int and e['workers']==1,'fixed original/round/worker contract differs')
    require(e.get('case_count')==400 and type(e['case_count']) is int and e.get('timed_count')==70 and type(e['timed_count']) is int,'complete frozen400/70 required')
    require(e.get('factory_sha256')==FROZEN_FACTORY_SHA256 and text_sha(ROOT/'benchmarks/tv_chambolle.py')==FROZEN_FACTORY_SHA256,'frozen TV factory/controller changed')
    require(e.get('manifest_sha256')==sha(ROOT/'coverage/contracts/tv-chambolle-benchmark.json'),'frozen400 manifest changed')
    require(isinstance(e.get('root_after_admission'),dict) and e['root_after_admission'].get('source')==AFTER_PRODUCT and e['root_after_admission'].get('passed') is True and re.fullmatch('[0-9a-f]{64}',e['root_after_admission'].get('sha256','')),'genuine AFTER four-profile root admission pending')
    require(set(e.get('products',{}))=={'before','after'},'both public products required')
    for role,source in (('before',BEFORE_PRODUCT),('after',AFTER_PRODUCT)):
        require(set(e['products'][role])==set(PROFILES),'all three product profiles required')
        for profile,frame in e['products'][role].items():
            require(isinstance(frame,dict) and set(frame)=={'pin','artifact','core_sha256'},'complete accepted product frame pending')
            pin=frame['pin'];c=rank_controller();c.accepted_pin({'accepted_products':{role:{profile:pin}}},role,profile)
            require(pin['revision']==source and pin['artifact_name']=='rank-statistics-product-'+profile,'wrong qualified backend source/artifact')
            scope=pin['scope'];require(all(type(scope[k]) is int and scope[k]==v for k,v in dict(image_paths=192,metadata_paths=3,sdk_families=18,sdk_members=63).items()),'exact192/SDK18 scope required')
            require(re.fullmatch('[0-9a-f]{64}',frame['core_sha256']),'owned core pin missing')
            a=frame['artifact'];require(set(a)=={'id','digest','run','name'} and str(a['id']).isdigit() and a['run']==pin['artifact_run'] and a['name']==pin['artifact_name'] and re.fullmatch('[0-9a-f]{64}',a['digest']),'raw artifact provenance absent')
    require(set(e.get('approved_after_source_sha256',{}))==ALLOWED_DELTA and all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for h in e['approved_after_source_sha256'].values()),'exact six-path source patch approval required')
    require(e['approved_after_source_sha256'][PRIVATE_HEADER]==PRIVATE_HEADER_SHA256,'exact reviewed private header pin required')
    return e

def verify_controller_checkout(expected_revision):
    require(revision()==expected_revision,'controller checkout revision differs')
    # Canonical Git text comparison works for Windows/WSL CRLF and hosted LF;
    # existing -text scientific files retain exact raw-byte Git comparison.
    require(subprocess.run(['git','-c','core.autocrlf=true','-C',str(ROOT),'diff','--quiet','HEAD','--'],check=False).returncode==0,'tracked controller/source checkout changed')
    return expected_revision


def load_experiment(path,expected_sha,controller_revision=None):
    require(re.fullmatch('[0-9a-f]{64}',expected_sha or '') and sha(path)==expected_sha,'exact external experiment SHA required')
    selected=controller_revision or revision();verify_controller_checkout(selected)
    return validate_experiment(read_json(path),selected)

def tv_common(identity,c=None):
    """Reversible fork of ONE common-input model; original rank stays strict."""
    c=c or rank_controller();source=inspect.getsource(c._common_build_identity)
    old="_below_build_root(source,facts['source_root'])=='src/rank.cpp'"
    new="_below_build_root(source,facts['source_root']) in TV_TUS"
    require(source.count(old)==1,'historical common model changed; review derivation')
    derived=source.replace(old,new).replace('approved_counts_source','approved_tv_admission_source').replace('counts_dependent','tv_admission_dependent').replace('actual src/rank.cpp translation unit absent','actual TV translation units absent')
    inverse=derived.replace(new,old).replace('approved_tv_admission_source','approved_counts_source').replace('tv_admission_dependent','counts_dependent').replace('actual TV translation units absent','actual src/rank.cpp translation unit absent')
    require(ast.dump(ast.parse(inverse),include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False),'non-reversible common model derivation')
    facts=identity['raw'];marked={c._below_build_root(u['source'],facts['source_root']) for u in facts['translation_units']} & TV_TUS
    require(marked==TV_TUS and sum(c._below_build_root(u['source'],facts['source_root']) in TV_TUS for u in facts['translation_units'])==2,'exactly two approved TV TU markers required')
    require(identity.get('common')==c._common_build_identity(identity) and identity.get('common_sha256')==c.digest(identity['common']),'raw full common identity was not authenticated')
    namespace=dict(vars(c),TV_TUS=TV_TUS);exec(compile(derived,'<separate-TV-common-input-model>','exec'),namespace)
    return namespace['_common_build_identity'](identity)


def header_projection(before_identity,after_identity,before_model,after_model,after_source,c=None):
    """Authenticate literal qualified bytes/edges, then project exactly one addition.

    Inputs and identity.common remain unchanged. The complete unprojected models
    and all raw header records remain auditable; only two used-header edges and
    their single global header entry are removed from a comparison copy.
    """
    c=c or rank_controller();require(before_model==tv_common(before_identity,c) and after_model==tv_common(after_identity,c),'projection models not bound to authenticated raw identities');facts=after_identity['raw']
    matches=[(p,r) for p,r in after_identity['file_records']['headers'].items()
             if c._below_build_root(p,facts['source_root'])==PRIVATE_HEADER]
    require(len(matches)==1,'exactly one qualified private header required')
    path,record=matches[0];raw=(Path(after_source)/PRIVATE_HEADER).read_bytes();raw_sha=hashlib.sha256(raw).hexdigest()
    require(raw_sha in PRIVATE_HEADER_RAW and len(raw)==PRIVATE_HEADER_RAW[raw_sha], 'private header is not exact reviewed LF/CRLF bytes')
    require(record['sha256']==raw_sha and record['bytes']==len(raw) and record['canonical_lf_sha256']==PRIVATE_HEADER_SHA256 and hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==PRIVATE_HEADER_SHA256,'compiled/private source header bytes differ')
    before_facts=before_identity['raw']
    require(not any(c._below_build_root(p,before_facts['source_root'])==PRIVATE_HEADER for p in before_identity['file_records']['headers']),'before unexpectedly uses private header')
    edges=[];normalized=[]
    for unit in facts['translation_units']:
        dependencies=[p for p in unit['dependencies'] if c._below_build_root(p,facts['source_root'])==PRIVATE_HEADER]
        if dependencies:
            relative=c._below_build_root(unit['source'],facts['source_root'])
            require(dependencies==[path] and relative in TV_TUS,'private header used by unapproved TU/duplicate or aliased edge')
            edges.append(relative);normalized.append(c._normalize_build_path(unit['source'],c._build_roots(facts)))
    require(len(edges)==2 and set(edges)==TV_TUS,'private header must have exactly two approved used-header edges')
    name=c._header_name(path,facts['header_roots']);require(name.startswith('project/'),'private header outside project roots')
    require(name not in before_model['used_headers'] and name in after_model['used_headers'],'private global header addition differs')
    require(after_model['used_headers'][name]==dict(path=c._normalize_build_path(path,c._build_roots(facts)),sha256=raw_sha,bytes=len(raw)),'private header full common record differs')
    projected=copy.deepcopy(after_model);projected['used_headers'].pop(name)
    removed=[]
    for unit in projected['translation_units']:
        if name in unit['used_headers']:
            require(unit['source'] in normalized and unit['used_headers'].count(name)==1,'unapproved normalized header edge')
            unit['used_headers'].remove(name);removed.append(unit['source'])
    require(sorted(removed)==sorted(normalized),'normalized header edge closure differs')
    require(projected==before_model,'compiler/TU/existingheader/flag/link change outside exact admission additions')
    return dict(schema='tv-admission-header-projection-v1',raw_before_common_sha256=digest(before_model),
                raw_after_common_sha256=digest(after_model),projected_common_sha256=digest(projected),
                private_header=PRIVATE_HEADER,canonical_lf_sha256=PRIVATE_HEADER_SHA256,
                actual_raw_sha256=raw_sha,actual_raw_bytes=len(raw),approved_added_used_header_edges=sorted(edges),
                actual_global_header_name=name,removed_global_entries=1,removed_edges=2)

def _runtime_member(n):
    return n in ('CMakeLists.txt','pyproject.toml') or n.startswith(('src/','include/skimagexx/')) and n.count('/')==(2 if n.startswith('include/') else 1) or n.startswith('python/skimagexx/') and n.count('/')==2 and n.endswith('.py')

def verify_pair(before,after,profile,e):
    c=rank_controller();cfg=read_json(ROOT/'coverage/contracts/rank-statistics-comparison.json');cfg['accepted_products']={r:{p:e['products'][r][p]['pin'] for p in PROFILES} for r in ('before','after')}
    result={};models={};identities={};products={'before':Path(before),'after':Path(after)}
    for role,path in products.items():
        result[role]=c.verify_product(path,role,profile,cfg)
        q=read_json(path/'qualification.json');identity=read_json(path/'build-identity.json');identities[role]=identity
        # Real old verifier binds all raw facts/evidence/source/wheel; no stub,
        # no rewritten identity.common and no accepted-pin change on disk.
        validated=c.verify_build_identity(path,profile,q,e['products'][role][profile]['pin']['revision'])
        models[role]=tv_common(identity,c)
        require(validated['sha256']==result[role]['build_identity_sha256'],'product identity differs')
    projection=header_projection(identities['before'],identities['after'],models['before'],models['after'],products['after']/'source',c)
    bp,ap=products['before']/'source',products['after']/'source';left=c.tree(bp);right=c.tree(ap)
    bridge=e['before_source_equivalence'];require(bridge.get('passed') is True and bridge.get('qualified_original_revision')==BEFORE_PRODUCT and bridge.get('before_source_anchor')==BEFORE_ANCHOR,'before source equivalence absent')
    members=bridge['runtime_and_installed_SDK_client_source_members'];families=e['products']['before'][profile]['pin']['scope']['sdk_family_names']
    required={n for n in left if _runtime_member(n) or n.startswith('tests/cmake_consumer/')}|{'tests/test_'+n+'_native.cpp' for n in families}
    require(set(members)==required,'incomplete before runtime/build/SDK source equivalence')
    for name,witness in members.items():require(text_sha(bp/name)==witness['sha256'],'before source bridge member differs: '+name)
    # Archive revisions stay original. Complete source changes outside runtime
    # are the authenticated 5e->0d bridge plus exact six reviewed paths.
    bridge_changed=set(bridge['qualification_source_full_git_delta_to_anchor']);allowed=bridge_changed|ALLOWED_DELTA
    changed={n for n in left|right if left.get(n)!=right.get(n)};require(changed<=allowed and TV_TUS<=changed,'source delta outside authenticated bridge/cursor packet')
    for name,h in e['approved_after_source_sha256'].items():require(text_sha(ap/name)==h,'approved cursor/plumbing file changed: '+name)
    for name in required-ALLOWED_DELTA:require(text_sha(bp/name)==text_sha(ap/name),'unapproved runtime/SDK change: '+name)
    for profile_pin in e['products']['after'].values():require(profile_pin['pin']['scope']==e['products']['before'][profile]['pin']['scope'],'scope mismatch')
    return dict(schema='tv-admission-common-input-v1',common_sha256=digest(models['before']),products=result,complete_source_changed=sorted(changed),approved_runtime_TUs=sorted(TV_TUS),header_projection=projection,before_source_label=BEFORE_PRODUCT,intended_before_anchor=BEFORE_ANCHOR,source_bridge_sha256=digest(bridge))

def fetch_product(frame,output):
    """Prepare-host only: authenticate original outer ZIP then safe extraction."""
    output=Path(output);require(not output.exists(),'fresh product output required');a=frame['artifact']
    metadata=subprocess.run(['gh','api','repos/bipxx/scikit-imagexx/actions/artifacts/'+str(a['id'])],capture_output=True,check=True);m=json.loads(metadata.stdout)
    require(m['workflow_run']['head_sha']==frame['pin']['revision'],'artifact source revision differs')
    require(m['name']==a['name'] and str(m['workflow_run']['id'])==a['run'] and m['digest']=='sha256:'+a['digest'],'actual artifact metadata differs')
    require(m['size_in_bytes']<=1600000000,'bounded product artifact exceeds RAM cap')
    proc=subprocess.Popen(['gh','api',m['archive_download_url'].split('api.github.com/')[-1]],stdout=subprocess.PIPE,stderr=subprocess.PIPE);buffer=io.BytesIO();h=hashlib.sha256();n=0
    for part in iter(lambda:proc.stdout.read(1024*1024),b''):n+=len(part);require(n<=1600000000,'artifact RAM cap');buffer.write(part);h.update(part)
    error=proc.stderr.read();require(proc.wait()==0,error.decode(errors='replace'));require(n==m['size_in_bytes'] and h.hexdigest()==a['digest'],'outer ZIP SHA differs');buffer.seek(0);output.mkdir(parents=True)
    with zipfile.ZipFile(buffer) as z:
        names=[i.filename for i in z.infolist() if not i.is_dir()];require(len(names)==len(set(names)),'duplicate artifact members')
        for name in names:
            c=rank_controller();c.safe_name(name);info=z.getinfo(name);require((info.external_attr>>16)&0o170000!=0o120000,'artifact symlink')
            target=output/name;require(target.resolve().is_relative_to(output.resolve()),'artifact escape');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    buffer.close();return dict(id=a['id'],raw_zip_sha256=h.hexdigest(),bytes=n,members=len(names),all_member_CRC_verified=True)

def validate_receipt(r,experiment_sha,generation,attempt):
    require(r.get('schema')=='tv-admission-preparation-v1' and r.get('passed') is True and r.get('profile') in PROFILES,'failed preparation')
    require(r.get('experiment_sha256')==experiment_sha and r.get('generation')==str(generation) and r.get('attempt')==str(attempt),'stale preparation')
    require(r.get('modes')==['normal','scalar'] and r.get('workers')==[1,4] and r.get('case_count')==400 and r.get('timed_count')==70,'partial/reference-only preparation')
    require(set(r.get('reports',{}))=={'normal','scalar'} and re.fullmatch('[0-9a-f]{64}',r.get('common_build_sha256','')),'preparation receipt closure absent')
    return r

def barrier(paths,experiment_sha,generation,attempt):
    require(len(paths)==3,'all three fresh preparations required');rows={}
    for path in paths:
        r=validate_receipt(read_json(path),experiment_sha,generation,attempt);require(r['profile'] not in rows,'duplicate profile');rows[r['profile']]=r
    require(set(rows)==set(PROFILES),'missing preparation profile');return rows
