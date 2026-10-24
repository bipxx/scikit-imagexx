"""Controlled R4 triads. Phase1 is reference-only/native0; products need root pins.

Private modules and failed65c are never baseline admission. The existing
rank_statistics.py corpus and timing_callback are the sole fixture/call source.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
import csv
from email.parser import BytesParser
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import secrets
import statistics
import subprocess
import sys
import traceback
import uuid
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/rank-statistics-comparison.json'
PROFILES=('windows','modern','minimum')
ROLES=('original','before','after')
BLOCKED_SOURCE='65c07410cde05cb5e60a9c8baf8f5b1953e2b7a3'


def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    path=Path(path)
    if os.name=='nt' and len(str(path.absolute()))>=248 and not str(path).startswith('\\\\?\\'):
        path=Path('\\\\?\\'+str(path.absolute()))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return path
def config(experiment=None,expected_sha=None):
    cfg=load_json(CONTRACT)
    if experiment is None:return cfg
    require(expected_sha and sha(experiment)==expected_sha,'external immutable experiment hash required')
    external=load_json(experiment)
    require(external.get('schema')=='rank-statistics-experiment-v1' and external.get('template_sha256')==text_sha(CONTRACT),'external experiment/template differs')
    cfg['accepted_products']=external['accepted_products'];cfg['experiment_sha256']=expected_sha;cfg['experiment_path']=str(Path(experiment).resolve())
    for role in ('before','after'):
        require(set(cfg['accepted_products'][role])==set(PROFILES),'all3 product profiles required')
        pins=[accepted_pin(cfg,role,p) for p in PROFILES]
        require(len({pin['revision'] for pin in pins})==1,'one qualified source per backend acrossall3 runtimes required')
    for profile in PROFILES:require(accepted_pin(cfg,'before',profile)['scope']==accepted_pin(cfg,'after',profile)['scope'],'cross-coverage before/after products forbidden')
    return cfg

def safe_name(name):
    path=PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,'unsafe artifact member')
    return path

def tree(path,exclude=()):
    result={}
    for item in Path(path).rglob('*'):
        require(not item.is_symlink(),'artifact symlink forbidden')
        if item.is_file() and item.relative_to(path).as_posix() not in exclude:
            result[item.relative_to(path).as_posix()]=sha(item)
    return result

def accepted_pin(cfg,role,profile):
    require(role in ('before','after') and profile in PROFILES,'invalid product selector')
    pin=cfg['accepted_products'][role][profile]
    require(isinstance(pin,dict),'independently qualified public product pending; Phase1 is native0')
    required={'revision','qualification_sha256','product_sha256','wheel_sha256','source_sha256','scope','artifact_run','artifact_name'}
    require(set(pin)==required,'accepted product pin fields differ')
    require(len(pin['revision'])==40 and pin['revision']!=BLOCKED_SOURCE,'failed65c/private source not admitted')
    require(all(isinstance(pin[k],str) and len(pin[k])==64 for k in ('qualification_sha256','product_sha256','wheel_sha256','source_sha256')),'invalid accepted product hash')
    scope=pin['scope'];require(set(scope)=={'image_paths','metadata_paths','sdk_families','sdk_members','sdk_family_names'},'coverage/SDK scope absent')
    require(all(type(scope[k]) is int and scope[k]>0 for k in ('image_paths','metadata_paths','sdk_families','sdk_members')),'coverage/SDK counts absent or invalid')
    require(type(scope['sdk_family_names']) is list and len(scope['sdk_family_names'])==len(set(scope['sdk_family_names']))==scope['sdk_families'],'SDK family identities absent/differ')
    require(isinstance(pin['artifact_run'],str) and pin['artifact_run'].isdigit() and isinstance(pin['artifact_name'],str) and pin['artifact_name'],'qualified run/artifact binding absent')
    return pin

def verify_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if not n.endswith('/')]
        require(len(names)==len(set(names)),'duplicate wheel member')
        for name in names:safe_name(name)
        records=[n for n in names if n.endswith('.dist-info/RECORD')];require(len(records)==1,'wheel RECORD missing/ambiguous')
        rows=list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        require(len(rows)==len(names) and {r[0] for r in rows}==set(names),'wheel RECORD file set differs')
        for name,encoded,size in rows:
            if not encoded:require(name==records[0],'unhashed wheel member');continue
            body=archive.read(name);expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b'=').decode()
            require(encoded==expected and str(len(body))==size,'wheel RECORD bytes differ')
        metadata=[n for n in names if n.endswith('.dist-info/METADATA')];require(len(metadata)==1,'wheel METADATA ambiguous')
        info=BytesParser().parsebytes(archive.read(metadata[0]))
        return dict(name=info['Name'].lower().replace('_','-'),version=info['Version'],sha256=sha(path),members={n:hashlib.sha256(archive.read(n)).hexdigest() for n in names})

BUILD_INPUT_SCHEMA='rank-statistics-build-input-v2'
BUILD_IDENTITY_SCHEMA='rank-statistics-build-identity-v2'
BUILD_FAMILIES={'project','python','numpy','pybind11','toolchain'}
BUILD_CONFIG_STRINGS={'compiler_id','compiler_version','generator','generator_version',
                      'target_arch','toolset','crt','build_type'}
BUILD_CONFIG_BOOLS={'openmp','avx2','lto'}
BUILD_FACT_KEYS={'schema','revision','profile','source_root','build_root','configuration',
                 'binaries','header_roots','translation_units','link_commands',
                 'link_inputs','evidence','product_wheel','core'}


def _absolute_build_path(value):
    require(type(value) is str and value and '\x00' not in value,'missing build path')
    normalized=value.replace('\\','/')
    require(normalized.startswith('/') or re.match(r'^[A-Za-z]:/',normalized),'build path must be absolute')
    require('..' not in normalized.split('/'),'parent traversal in build path')
    return normalized.rstrip('/') or '/'


def _below_build_root(value,root):
    value=_absolute_build_path(value);root=_absolute_build_path(root)
    left,right=(value.casefold(),root.casefold()) if re.match(r'^[A-Za-z]:/',root) or root.startswith('//') else (value,root)
    if left==right:return ''
    return value[len(root)+1:] if left.startswith(right.rstrip('/')+'/') else None


def _build_roots(facts):
    return sorted(((_absolute_build_path(facts['source_root']),'$SOURCE'),
                   (_absolute_build_path(facts['build_root']),'$BUILD')),key=lambda r:len(r[0]),reverse=True)


def _normalize_build_path(value,roots):
    for root,label in roots:
        relative=_below_build_root(value,root)
        if relative is not None:return label+('/'+relative if relative else '')
    return value


def _normalize_build_argument(value,roots):
    # Only actual path arguments under the two declared roots are normalized.
    # Flag names, order, values and unrelated compiler/system paths stay intact.
    normalized=value.replace('\\','/')
    if normalized.startswith('/') or re.match(r'^[A-Za-z]:/',normalized):
        result=_normalize_build_path(value,roots)
        if result!=value:return result
    for option in ('-isystem','-iquote','-include','-I','-L','/LIBPATH:','/IMPLIB:',
                   '/OUT:','/PDB:','/Fo','/Fd','/Fp','/Fi','/Fe','/I','-Wl,-rpath,'):
        if value.startswith(option) and len(value)>len(option):
            path=value[len(option):]
            if path.replace('\\','/').startswith('/') or re.match(r'^[A-Za-z]:/',path.replace('\\','/')):
                return option+_normalize_build_path(path,roots)
    if '=' in value:
        option,path=value.split('=',1)
        if path.replace('\\','/').startswith('/') or re.match(r'^[A-Za-z]:/',path.replace('\\','/')):
            return option+'='+_normalize_build_path(path,roots)
    return value


def _validate_build_facts(facts):
    require(type(facts) is dict and set(facts)==BUILD_FACT_KEYS and facts['schema']==BUILD_INPUT_SCHEMA,'incomplete build identity facts')
    require(re.fullmatch('[0-9a-f]{40}',facts['revision'] or '') and facts['profile'] in (*PROFILES,'debug'),'build identity source/profile absent')
    for key in ('source_root','build_root','product_wheel','core'):_absolute_build_path(facts[key])
    require(facts['source_root']!=facts['build_root'],'source/build roots must be distinct')
    require(all(_absolute_build_path(facts[k]) not in ('/',) and not re.fullmatch('[A-Za-z]:',_absolute_build_path(facts[k])) for k in ('source_root','build_root')),'filesystem root cannot normalize a build')
    configuration=facts['configuration']
    require(type(configuration) is dict and set(configuration)==BUILD_CONFIG_STRINGS|BUILD_CONFIG_BOOLS,'build configuration fields absent')
    require(all(type(configuration[k]) is str and configuration[k] for k in BUILD_CONFIG_STRINGS),'build configuration values absent')
    require(all(type(configuration[k]) is bool for k in BUILD_CONFIG_BOOLS),'build configuration booleans invalid')
    require(configuration['build_type'] in ('Release','Debug'),'unknown actual build type')
    family=configuration['compiler_id'];required={'GNU':{'compiler','backend','assembler','linker','link_driver','archiver','ranlib'},
                                               'MSVC':{'compiler','frontend','backend','linker','archiver'}}
    require(family in required,'unsupported compiler identity capture')
    require(type(facts['binaries']) is dict and required[family]<=facts['binaries'].keys(),'compiler/backend/linker binary facts absent')
    if family=='GNU' and configuration['lto']:
        require({'lto_backend','lto_wrapper','lto_plugin'}<=facts['binaries'].keys(),'executed GNU LTO tool identities absent')
    for role,path in facts['binaries'].items():
        require(re.fullmatch('[a-z][a-z0-9_-]*',role),'invalid binary role');_absolute_build_path(path)
    roots=facts['header_roots'];require(type(roots) is dict and set(roots)==BUILD_FAMILIES,'header families absent')
    for values in roots.values():
        require(type(values) is list and values and len(values)==len(set(values)),'header roots incomplete')
        for path in values:_absolute_build_path(path)
    require(type(facts['translation_units']) is list and facts['translation_units'],'actual translation units absent')
    require(type(facts['link_commands']) is list and facts['link_commands'],'actual link/archive commands absent')
    ids=set();outputs=set()
    for unit in facts['translation_units']:
        require(type(unit) is dict and set(unit)=={'source','target','argv','dependencies','outputs'},'translation unit facts incomplete')
        source=unit['source'];_absolute_build_path(source)
        require(_below_build_root(source,facts['source_root']) not in (None,''),'translation unit outside source root')
        require(type(unit['dependencies']) is list and unit['dependencies'],'actual used-header dependencies absent')
        for path in unit['dependencies']:_absolute_build_path(path)
    for command in [*facts['translation_units'],*facts['link_commands']]:
        if 'tool' in command:require(set(command)=={'target','tool','argv','outputs','inputs'} and command['tool'] in facts['binaries'],'link tool facts incomplete')
        target=command['target'];require(type(target) is str and target,'command target absent')
        identity=(target,command.get('source','link'),command.get('tool','compiler'));require(identity not in ids,'duplicate actual command');ids.add(identity)
        require(type(command['argv']) is list and command['argv'] and all(type(v) is str and v for v in command['argv']),'actual expanded argv absent')
        require(not any(v.startswith('@') or v.startswith(('-Wl,','-Wp,','-Wa,')) and any(p.startswith('@') for p in v.split(',')[1:]) for v in command['argv']),'unexpanded response file in actual command')
        require(not any(v.startswith(('-B','/B1','/B2','/Bx','/Bz')) for v in command['argv'][1:]),'unsupported compiler/backend override requires explicit capture support')
        for value in command['argv'][1:]:
            if value.startswith(('-fplugin=','-specs=','--specs=')):
                dependency=value.split('=',1)[1];_absolute_build_path(dependency)
                require(dependency in facts['binaries'].values(),'explicit compiler plugin/specs input is not pinned')
        _absolute_build_path(command['argv'][0])
        require(type(command['outputs']) is list and command['outputs'],'actual command outputs absent')
        for output in command['outputs']:
            require(_below_build_root(output,facts['build_root']) not in (None,''),'generated output outside declared build root');outputs.add(output)
    tools={r['tool'] for r in facts['link_commands']}
    require('archiver' in tools and bool(tools&{'compiler','linker'}),'module link and static archive facts required')
    executed_lto=any(v.startswith(('-flto','/GL','/LTCG')) for command in [*facts['translation_units'],*facts['link_commands']] for v in command['argv'])
    require(not executed_lto or configuration['lto'],'executed LTO cannot be reported as disabled')
    require(type(facts['link_inputs']) is list and facts['link_inputs'],'used link inputs absent')
    require(any(r.get('generated') is False for r in facts['link_inputs']),'external toolchain link input facts absent')
    for item in facts['link_inputs']:
        require(type(item) is dict and set(item)=={'path','generated'} and type(item['generated']) is bool,'link input facts invalid')
        _absolute_build_path(item['path'])
        if item['generated']:require(item['path'] in outputs,'generated link input lacks actual producing command')
    used_inputs=set()
    for command in facts['link_commands']:
        require(type(command['inputs']) is list and command['inputs'],'per-link dependency edges absent')
        for path in command['inputs']:_absolute_build_path(path);used_inputs.add(path)
    require(used_inputs=={r['path'] for r in facts['link_inputs']},'per-link/global input closure differs')
    evidence=facts['evidence'];require(type(evidence) is dict and {'configuration','compiler','commands','dependencies'}<=evidence.keys(),'raw build evidence absent')
    for role,paths in evidence.items():
        require(re.fullmatch('[a-z][a-z0-9_-]*',role) and type(paths) is list and paths,'raw evidence group incomplete')
        for path in paths:_absolute_build_path(path)


def _build_file_record(path,canonical=False):
    path=Path(path);require(path.is_file(),'actual build input missing: '+str(path))
    body=path.read_bytes();record=dict(path=str(path),resolved_path=str(path.resolve()),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    if canonical:record['canonical_lf_sha256']=hashlib.sha256(body.replace(b'\r\n',b'\n')).hexdigest()
    return record


def _header_name(path,roots):
    matches=[]
    for family,values in roots.items():
        for index,root in enumerate(values):
            relative=_below_build_root(path,root)
            if relative not in (None,''):matches.append((len(root),family+'/'+str(index)+'/'+relative))
    require(matches,'used dependency outside declared header roots: '+path)
    matches.sort(reverse=True);require(len(matches)==1 or matches[0][0]!=matches[1][0],'ambiguous used-header root')
    return matches[0][1]


def _record_valid(record,canonical=False):
    keys={'path','resolved_path','bytes','sha256'}|({'canonical_lf_sha256'} if canonical else set())
    require(type(record) is dict and set(record)==keys,'actual file record incomplete')
    _absolute_build_path(record['path']);_absolute_build_path(record['resolved_path'])
    require(type(record['bytes']) is int and record['bytes']>=0 and type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}',record['sha256']),'actual file hash/size absent')
    if canonical:require(type(record['canonical_lf_sha256']) is str and re.fullmatch('[0-9a-f]{64}',record['canonical_lf_sha256']),'canonical source/header hash absent')


def _build_evidence_paths(facts):
    return {'build-identity-evidence/'+role+'/'+str(index)+'-'+path.replace('\\','/').rsplit('/',1)[-1]:path
            for role,paths in facts['evidence'].items() for index,path in enumerate(paths)}


def _common_build_identity(identity):
    facts=identity['raw'];_validate_build_facts(facts);roots=_build_roots(facts)
    records=identity['file_records'];require(set(records)=={'binaries','sources','headers','outputs','link_inputs','product_wheel','core'},'build file records absent')
    require(set(records['binaries'])==set(facts['binaries']),'binary record set differs')
    for role,record in records['binaries'].items():
        _record_valid(record);require(record['path']==facts['binaries'][role] and record['bytes']>0,'binary path/bytes differ')
    common_binaries={role:dict(path=_normalize_build_path(r['resolved_path'],roots),sha256=r['sha256'],bytes=r['bytes']) for role,r in records['binaries'].items()}
    units=[];header_paths=set();source_paths=set();output_paths=set();counts_outputs=set()
    for unit in facts['translation_units']:
        source=unit['source'];source_paths.add(source);record=records['sources'][source];_record_valid(record,True);require(record['path']==source,'source record path differs')
        require(_absolute_build_path(unit['argv'][0])==_absolute_build_path(facts['binaries']['compiler']) or _absolute_build_path(unit['argv'][0])==_absolute_build_path(records['binaries']['compiler']['resolved_path']),'compile argv driver differs')
        dependencies=[]
        for path in unit['dependencies']:
            if _absolute_build_path(path)==_absolute_build_path(source):continue
            header_paths.add(path);dependencies.append(_header_name(path,facts['header_roots']))
        argv=[_normalize_build_argument(v,roots) for v in unit['argv']]
        fp=[v for v in argv if v.startswith(('/fp:','-ffp-','-ffast-math','-fno-fast-math','-funsafe-math','-fno-unsafe-math','-flto','-fno-lto','/GL'))]
        counts_source=_below_build_root(source,facts['source_root'])=='src/rank.cpp'
        if counts_source:counts_outputs.update(unit['outputs'])
        units.append(dict(target=unit['target'],source=_normalize_build_path(source,roots),argv=argv,used_headers=sorted(set(dependencies)),outputs=[_normalize_build_path(v,roots) for v in unit['outputs']],FP_LTO_flags=fp or ['compiler-default'],**(dict(approved_counts_source=True) if counts_source else dict(source_canonical_lf_sha256=record['canonical_lf_sha256']))))
        output_paths.update(unit['outputs'])
    require(counts_outputs,'actual src/rank.cpp translation unit absent')
    require(set(records['sources'])==source_paths and set(records['headers'])==header_paths,'source/used-header record closure differs')
    headers={};families=set()
    for path,record in records['headers'].items():
        _record_valid(record,True);require(record['path']==path,'header record path differs');name=_header_name(path,facts['header_roots']);families.add(name.split('/')[0]);require(name not in headers,'duplicate used-header identity')
        headers[name]=dict(path=_normalize_build_path(path,roots),sha256=record['sha256'],bytes=record['bytes'])
    require(families==BUILD_FAMILIES,'actual used-header family coverage incomplete')
    links=[]
    for command in facts['link_commands']:
        tool=command['tool'];driver=_absolute_build_path(command['argv'][0]);require(driver in {_absolute_build_path(facts['binaries'][tool]),_absolute_build_path(records['binaries'][tool]['resolved_path'])},'link argv tool differs')
        links.append(dict(target=command['target'],tool=tool,argv=[_normalize_build_argument(v,roots) for v in command['argv']],inputs=sorted(_normalize_build_path(v,roots) for v in command['inputs']),outputs=[_normalize_build_path(v,roots) for v in command['outputs']]));output_paths.update(command['outputs'])
    require(set(records['outputs'])==output_paths,'generated-output record closure differs')
    for path,record in records['outputs'].items():_record_valid(record);require(record['path']==path,'output record path differs')
    require(set(records['link_inputs'])=={r['path'] for r in facts['link_inputs']},'used link-input record closure differs')
    # Generated files are outputs, not common inputs. LTO objects may contain
    # random stamps, even with identical input source, flags and tools. Retain
    # their raw hashes per product; never mask build-env or external libraries.
    affected=set(counts_outputs)
    while True:
        expanded=affected|{p for c in facts['link_commands'] if affected.intersection(c['inputs']) for p in c['outputs']}
        if expanded==affected:break
        affected=expanded
    outputs={_normalize_build_path(path,roots):dict(traceable_generated_output=True,counts_dependent=path in affected) for path in records['outputs']}
    inputs=[]
    for item in facts['link_inputs']:
        record=records['link_inputs'][item['path']];_record_valid(record);require(record['path']==item['path'],'link-input record path differs')
        if item['generated']:require(record['sha256']==records['outputs'][item['path']]['sha256'],'generated input differs from produced output')
        inputs.append(dict(path=_normalize_build_path(item['path'],roots),generated=item['generated'],counts_dependent=item['generated'] and item['path'] in affected,**({} if item['generated'] else dict(sha256=record['sha256'],bytes=record['bytes']))))
    for name in ('product_wheel','core'):_record_valid(records[name]);require(records[name]['path']==facts[name],name+' actual path differs')
    return dict(schema=BUILD_IDENTITY_SCHEMA,configuration=facts['configuration'],binaries=common_binaries,
                header_roots={k:[_normalize_build_path(v,roots) for v in values] for k,values in facts['header_roots'].items()},
                used_headers=headers,translation_units=sorted(units,key=lambda u:(u['target'],u['source'])),
                link_commands=sorted(links,key=lambda u:(u['target'],u['tool'],u['argv'])),generated_outputs=outputs,link_inputs=sorted(inputs,key=lambda r:(r['path'],r['generated'])))


def capture_build_identity(facts_path,output):
    """Hash real build-host inputs, never execute compilers or numerical code."""
    facts=load_json(facts_path);_validate_build_facts(facts);output=Path(output)
    require(output.name=='build-identity.json' and not output.exists(),'fresh build-identity.json output required')
    output.parent.mkdir(parents=True,exist_ok=True)
    records=dict(binaries={k:_build_file_record(v) for k,v in facts['binaries'].items()},sources={},headers={},outputs={},link_inputs={},product_wheel=_build_file_record(facts['product_wheel']),core=_build_file_record(facts['core']))
    for unit in facts['translation_units']:
        records['sources'][unit['source']]=_build_file_record(unit['source'],True)
        for path in unit['dependencies']:
            if _absolute_build_path(path)!=_absolute_build_path(unit['source']):records['headers'][path]=_build_file_record(path,True)
    for command in [*facts['translation_units'],*facts['link_commands']]:
        for path in command['outputs']:records['outputs'][path]=_build_file_record(path)
    for item in facts['link_inputs']:records['link_inputs'][item['path']]=_build_file_record(item['path'])
    evidence={}
    for relative,path in _build_evidence_paths(facts).items():
        record=_build_file_record(path);safe_name(relative);target=output.parent/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(Path(path).read_bytes())
        evidence[relative]=dict(source_path=path,bytes=record['bytes'],sha256=record['sha256'])
    identity=dict(schema=BUILD_IDENTITY_SCHEMA,revision=facts['revision'],profile=facts['profile'],raw=facts,file_records=records,evidence_records=evidence)
    identity['common']=_common_build_identity(identity);identity['common_sha256']=digest(identity['common'])
    wheel=verify_wheel(facts['product_wheel']);core='skimagexx/'+Path(facts['core']).name
    require(wheel['name']=='scikit-imagexx' and wheel['members'].get(core)==records['core']['sha256'],'actual core differs from actual product wheel')
    write_new(output,identity);return dict(schema=BUILD_IDENTITY_SCHEMA,path=str(output),sha256=sha(output),common_sha256=identity['common_sha256'],source=facts['revision'],profile=facts['profile'])


def verify_build_identity(path,profile,qualification,revision,wheel=None):
    path=Path(path);filename='build-identity.json';record=qualification.get('records',{}).get(filename)
    require(qualification.get('schema')=='public-qualification-v2' and qualification.get('passed') is True and qualification.get('profile')==profile and qualification.get('source',{}).get('revision')==revision,'build identity lacks original successful source qualification')
    require(isinstance(record,dict) and (path/filename).is_file(),'qualified build identity missing; future producer capture required')
    require(sha(path/filename)==record.get('sha256') and (path/filename).stat().st_size==record.get('bytes'),'qualified build identity record differs')
    identity=load_json(path/filename)
    require(set(identity)=={'schema','revision','profile','raw','file_records','evidence_records','common','common_sha256'} and identity['schema']==BUILD_IDENTITY_SCHEMA,'build identity schema differs')
    _validate_build_facts(identity['raw'])
    require(identity['revision']==identity['raw']['revision']==revision and identity['profile']==identity['raw']['profile']==profile,'build identity source/profile differs')
    require(identity['raw']['configuration']['build_type']=='Release','attributable comparison requires actual Release build')
    common=_common_build_identity(identity);require(identity['common']==common and identity['common_sha256']==digest(common),'build identity common facts inconsistent')
    expected_evidence=_build_evidence_paths(identity['raw']);require(set(identity['evidence_records'])==set(expected_evidence),'raw build evidence closure differs')
    for name,evidence in identity['evidence_records'].items():
        safe_name(name);require(name.startswith('build-identity-evidence/'),'raw evidence path outside identity namespace')
        require(set(evidence)=={'source_path','bytes','sha256'} and evidence['source_path']==expected_evidence[name],'raw evidence source mapping differs')
        qualified=qualification['records'].get(name);require(isinstance(qualified,dict) and qualified.get('sha256')==evidence['sha256'] and qualified.get('bytes')==evidence['bytes'],'raw build evidence not bound by original qualification')
        require((path/name).is_file() and sha(path/name)==evidence['sha256'] and (path/name).stat().st_size==evidence['bytes'],'raw build evidence bytes differ')
    for source,record in identity['file_records']['sources'].items():
        relative=_below_build_root(source,identity['raw']['source_root']);safe_name(relative);require(text_sha(path/'source'/relative)==record['canonical_lf_sha256'],'compiled translation-unit source differs from qualified source')
    for header,record in identity['file_records']['headers'].items():
        relative=_below_build_root(header,identity['raw']['source_root'])
        if relative not in (None,'') and _below_build_root(header,identity['raw']['build_root']) is None:
            safe_name(relative);require(text_sha(path/'source'/relative)==record['canonical_lf_sha256'],'compiled project header differs from qualified source')
    if wheel is not None:
        require(identity['file_records']['product_wheel']['sha256']==wheel['sha256'],'build identity product wheel differs')
        core=identity['file_records']['core'];name=core['path'].replace('\\','/').rsplit('/',1)[-1];require(wheel['members'].get('skimagexx/'+name)==core['sha256'],'build identity core differs from qualified wheel')
    return dict(schema=BUILD_IDENTITY_SCHEMA,sha256=sha(path/filename),common_sha256=digest(common),common=common)


def verify_common_build(before,after,profile,cfg):
    identities=[]
    for role,path in (('before',before),('after',after)):
        pin=accepted_pin(cfg,role,profile);qualification=load_json(Path(path)/'qualification.json')
        identities.append(verify_build_identity(path,profile,qualification,pin['revision']))
    require(identities[0]['common']==identities[1]['common'],'before/after compiler, headers, used inputs or build flags differ')
    return dict(schema=BUILD_IDENTITY_SCHEMA,common_sha256=identities[0]['common_sha256'],products={r:i['sha256'] for r,i in zip(('before','after'),identities)})


def verify_product(path,role,profile,cfg):
    """Standard-library validation BEFORE importing fixtures or numerical code.

    Input format is a root-qualified complete product directory with wheels/,
    source.zip, extracted source/, qualification.json and product.json. No
    archive extraction, pip, compiler or acceptance invention happens here.
    """
    path=Path(path).resolve();pin=accepted_pin(cfg,role,profile)
    require(sha(path/'product.json')==pin['product_sha256'],'product seal differs from accepted pin')
    manifest=load_json(path/'product.json')
    require(manifest['schema']=='rank-statistics-comparison-product-v1' and manifest['revision']==pin['revision'] and manifest['profile']==profile,'wrong product source/runtime')
    require(manifest['files_sha256']==tree(path,('product.json',)),'complete product file set/bytes differ')
    require(sha(path/'qualification.json')==pin['qualification_sha256'],'qualification receipt differs')
    qualification=load_json(path/'qualification.json')
    require(qualification.get('passed') is True and qualification.get('source',{}).get('revision')==pin['revision'] and qualification.get('profile')==profile,'not the qualified actual public source')
    scope=pin['scope']
    require(qualification.get('schema')=='public-qualification-v2' and qualification.get('SDK_checks')==2*scope['sdk_families'] and qualification.get('no_timing') is True and not qualification.get('failures'),'failed/incomplete public+SDK qualification cannot admit product')
    expected=qualification.get('expected_registry',{})
    require(expected.get('image_count')==scope['image_paths'] and expected.get('metadata_count')==scope['metadata_paths']
            and expected.get('registered_count')==scope['image_paths']+scope['metadata_paths']
            and len(expected.get('functions',[]))==len(set(expected.get('functions',[])))==scope['image_paths']+scope['metadata_paths'],'qualified image/metadata registry scope differs')
    sdk_receipt=load_json(path/'sdk.json');sdk_record=qualification['records']['sdk.json']
    require(sha(path/'sdk.json')==sdk_record['sha256'] and (path/'sdk.json').stat().st_size==sdk_record['bytes'],'SDK qualification envelope record differs')
    require(sdk_receipt['families']==scope['sdk_family_names'] and sdk_receipt['checks']==2*scope['sdk_families'],'qualified SDK families differ')
    require(sha(path/'source.zip')==pin['source_sha256'],'source archive differs')
    source=path/'source'
    with zipfile.ZipFile(path/'source.zip') as archive:
        names=[n for n in archive.namelist() if not n.endswith('/')]
        require(len(names)==len(set(names)),'duplicate source ZIP member')
        for name in names:safe_name(name)
        require({n:hashlib.sha256(archive.read(n)).hexdigest() for n in names}==tree(source),'complete source ZIP/extracted bytes differ')
    for name,value in cfg['source_text_sha256'].items():require(text_sha(source/name)==value,'fixed R4 fixture/harness source differs: '+name)
    corpus=load_json(source/'coverage/contracts/rank-statistics-benchmark.json')
    require(corpus['case_contract_sha256']==cfg['case_contract_sha256'] and len(corpus['case_inventory'])==1379,'R4 corpus differs')
    wheel_rows={}
    for wheel in sorted((path/'wheels').glob('*.whl')):
        row=verify_wheel(wheel);require(row['name'] not in wheel_rows,'duplicate runtime distribution');wheel_rows[row['name']]=row
    require({'scikit-imagexx','scikit-image','numpy','scipy'}<=wheel_rows.keys(),'incomplete wheel closure')
    runtime=load_json(path/'runtime.json')
    require(runtime['schema']=='ci-offline-measurement-v1' and runtime['source']['revision']==pin['revision'],'runtime/source manifest differs')
    require({r['name']:dict(name=r['name'],version=r['version'],sha256=r['sha256']) for r in wheel_rows.values()}=={r['name']:r for r in runtime['wheels'].values()},'runtime wheel closure differs')
    require(sha(path/'runtime-lock.txt')==runtime['lock_sha256'],'runtime lock differs')
    product=wheel_rows['scikit-imagexx'];require(product['sha256']==pin['wheel_sha256'],'wrong accepted wheel')
    require(wheel_rows['scikit-image']['version']=='0.26.0' and wheel_rows['numpy']['version']==cfg['profiles'][profile]['numpy'] and wheel_rows['scipy']['version']==cfg['profiles'][profile]['scipy'],'wrong reference runtime')
    sdk={n:h for n,h in product['members'].items() if n.startswith(('include/skimagexx/','lib/cmake/skimagexx/','share/licenses/scikit-imagexx/')) or n in ('lib/skimagexx.lib','lib/libskimagexx.a')}
    require(len(sdk)==scope['sdk_members'] and sdk==runtime['runtime']['sdk_sha256']==sdk_receipt['SDK_sha256'],'complete configured SDK differs')
    with zipfile.ZipFile(next(w for w in (path/'wheels').glob('*.whl') if sha(w)==pin['wheel_sha256'])) as archive:
        headers={n for n in archive.namelist() if n.startswith('include/skimagexx/') and not n.endswith('/')}
        require(headers=={p.relative_to(source).as_posix() for p in (source/'include/skimagexx').rglob('*') if p.is_file()},'full source/header set differs')
        for name in headers:require(archive.read(name).replace(b'\r\n',b'\n')==(source/name).read_bytes().replace(b'\r\n',b'\n'),'source/header bytes differ')
        python={n.removeprefix('skimagexx/'):archive.read(n).replace(b'\r\n',b'\n') for n in archive.namelist() if n.startswith('skimagexx/') and n.endswith('.py')}
        require(python=={p.relative_to(source/'python/skimagexx').as_posix():p.read_bytes().replace(b'\r\n',b'\n') for p in (source/'python/skimagexx').rglob('*.py')},'complete Python source/wheel differs')
    build=verify_build_identity(path,profile,qualification,pin['revision'],product)
    return dict(role=role,profile=profile,revision=pin['revision'],product_sha256=pin['product_sha256'],wheel_sha256=pin['wheel_sha256'],qualification_sha256=pin['qualification_sha256'],scope=scope,artifact_run=pin['artifact_run'],artifact_name=pin['artifact_name'],runtime_sha256=sha(path/'runtime.json'),build_identity_sha256=build['sha256'],build_common_sha256=build['common_sha256'])

def canonical_source_bytes(path):
    body=Path(path).read_bytes()
    text_suffixes={'.cpp','.cc','.c','.h','.hpp','.hxx','.py','.pyi','.toml','.txt','.cmake','.json','.md','.rst','.yml','.yaml','.in','.csv','.ps1','.sh','.bat','.cmd'}
    return body.replace(b'\r\n',b'\n') if Path(path).suffix in text_suffixes or Path(path).name in ('.gitattributes','.gitignore','CMakeLists.txt') else body


def apply_counts_patch(before,patch):
    """Exact approved unified hunks, in memory, allowing only line offsets."""
    hunks=[];old=[];new=[];active=False
    for line in patch.splitlines(keepends=True):
        if line.startswith('@@'):
            if active:hunks.append((''.join(old),''.join(new)))
            active=True;old=[];new=[]
        elif active:
            require(line[:1] in (' ','+','-'),'unsupported counts patch directive')
            if line[0] in (' ','-'):old.append(line[1:])
            if line[0] in (' ','+'):new.append(line[1:])
    if active:hunks.append((''.join(old),''.join(new)))
    require(hunks,'counts patch missing')
    for previous,replacement in hunks:
        require(previous and before.count(previous)==1,'approved counts hunk absent/ambiguous')
        before=before.replace(previous,replacement,1)
    return before


def verify_pair_sources(before,after,profile,cfg):
    left=accepted_pin(cfg,'before',profile);right=accepted_pin(cfg,'after',profile)
    require(left['scope']==right['scope'],'cross-coverage before/after product scope')
    lq=load_json(Path(before)/'qualification.json');rq=load_json(Path(after)/'qualification.json')
    require(lq['expected_registry']==rq['expected_registry'],'different full public registries forbidden')
    common_build=verify_common_build(before,after,profile,cfg)
    directories=[Path(before)/'source',Path(after)/'source'];indices=[]
    for directory in directories:indices.append({p.relative_to(directory).as_posix():hashlib.sha256(canonical_source_bytes(p)).hexdigest() for p in directory.rglob('*') if p.is_file()})
    changed=sorted(n for n in set(indices[0])|set(indices[1]) if indices[0].get(n)!=indices[1].get(n))
    require('src/rank.cpp' in changed and set(changed)<=set(cfg['allowed_source_delta']),'source delta outside isolated counts packet')
    patch=cfg['counts_cpp_patch'];require(hashlib.sha256(patch.encode()).hexdigest()==cfg['counts_cpp_patch_sha256'],'approved counts patch changed')
    expected=apply_counts_patch(canonical_source_bytes(directories[0]/'src/rank.cpp').decode(),patch)
    require(expected.encode()==canonical_source_bytes(directories[1]/'src/rank.cpp'),'rank.cpp has changes beyond approved counts optimization')
    for name,value in cfg['counts_test_sha256'].items():
        if name in changed:require(text_sha(directories[1]/name)==value,'focused counts test delta differs')
    return dict(changed=changed,patch_sha256=cfg['counts_cpp_patch_sha256'],full_source_index_sha256=[digest(i) for i in indices],scope=left['scope'],build_identity=common_build)


def validate_preparation_report(report,cfg,profile):
    require(report.get('schema')=='rank-statistics-controlled-v1' and report.get('status')=='complete-parity-check'
            and report.get('check_only') is True and report.get('reference_only') is False
            and report.get('profile')==profile and not report.get('results') and not report.get('failures'), 'preparation report failed/partial/timed')
    require(report.get('contract_sha256')==text_sha(CONTRACT) and report.get('experiment_sha256')==cfg.get('experiment_sha256'),'preparation report contract differs')
    require(report.get('source_comparison',{}).get('patch_sha256')==cfg['counts_cpp_patch_sha256']
            and 'src/rank.cpp' in report['source_comparison'].get('changed',[])
            and set(report['source_comparison']['changed'])<=set(cfg['allowed_source_delta'])
            and report['source_comparison'].get('scope')==accepted_pin(cfg,'before',profile)['scope']==accepted_pin(cfg,'after',profile)['scope'],'isolated source attribution absent')
    require(len(report['source_comparison'].get('full_source_index_sha256',[]))==2 and all(isinstance(v,str) and len(v)==64 for v in report['source_comparison']['full_source_index_sha256']),'full source indices absent')
    build=report['source_comparison'].get('build_identity',{})
    require(build.get('schema')==BUILD_IDENTITY_SCHEMA and re.fullmatch('[0-9a-f]{64}',build.get('common_sha256','')) and set(build.get('products',{}))=={'before','after'} and all(re.fullmatch('[0-9a-f]{64}',v) for v in build['products'].values()),'common qualified build identity absent')
    require(set(report.get('checks',{}))=={'normal','scalar'} and set(report.get('worker_identities',{}))=={'normal','scalar'},'preparation modes missing')
    for mode in ('normal','scalar'):
        identities=report['worker_identities'][mode];require(set(identities)==set(ROLES),'backend identity missing')
        require(all(type(r.get('pid')) is int and r['pid']>0 for r in identities.values()) and len({r['pid'] for r in identities.values()})==3,'preparation processes not distinct')
        for role,identity in identities.items():
            require(identity['role']==role and identity['profile']==profile and type(identity['default_workers']) is int and identity['default_workers']==1
                    and identity['scalar']==(mode=='scalar') and identity['has_avx2']==(mode=='normal'),'wrong worker metadata/mode')
            product_role='before' if role=='original' else role;pin=accepted_pin(cfg,product_role,profile)
            require(identity['source']==pin['revision'] and identity['product_sha256']==pin['product_sha256'],'worker source/product not accepted')
        require(identities['before']['core_sha256']!=identities['after']['core_sha256'] and identities['before']['core_path']!=identities['after']['core_path'],'same native core cannot be before and after')
        require(set(report['checks'][mode])==set(ROLES),'backend parity missing')
        for role,value in report['checks'][mode].items():
            require(value['aliases']==cfg['algorithm_names'] and not value['mismatches'],'aliases/mismatches incomplete')
            require([v['workers'] for v in value['checks']]==[1,4] and all(type(v['workers']) is int for v in value['checks']),'preparation workers incomplete')
            for observations in value['checks']:
                require(len(observations['digests'])==len(observations['errors'])==len(observations['warnings'])==1379
                        and digest(observations['digests'])==cfg['expected_observation_vectors'][profile],'frozen observations incomplete/different')


def claim_measurement(receipts,profile,generation,attempt,claim_root,authentication=None):
    require(generation and attempt and claim_root,'measurement session provenance absent')
    key=digest(dict(profile=profile,generation=generation,attempt=attempt,receipts=[sha(p) for p in receipts]))
    path=Path(claim_root)/'rank-statistics-comparison-claims'/(key+'.json')
    return write_new(path,dict(profile=profile,generation=generation,attempt=attempt,receipt_hashes=[sha(p) for p in receipts],authentication=authentication))


def receipt_barrier(paths,cfg,generation,attempt):
    require(len(paths)==3,'all3 preparation receipts required before fixture imports/timers')
    receipts=[load_json(p) for p in paths]
    require({r.get('profile') for r in receipts}==set(PROFILES),'partial/duplicate preparation profile barrier')
    require(len({r.get('receipt_id') for r in receipts})==3,'preparation receipt reused')
    for receipt_path,row in zip(paths,receipts):
        require(row.get('schema')=='rank-statistics-comparison-preparation-v1' and row.get('native_admission') is True and row.get('reference_only') is False,'reference-only/private receipt cannot admit measurement')
        require(row.get('generation')==generation and row.get('attempt')==attempt and row.get('contract_sha256')==text_sha(CONTRACT) and row.get('experiment_sha256')==cfg.get('experiment_sha256'),'stale/reused preparation receipt')
        require(type(row.get('canonical_cases')) is int and type(row.get('alias_checks')) is int and all(type(w) is int for w in row.get('workers',[])) and all(type(i) is int for i in row.get('timed_ids',[])),'preparation counts/workers must be exact integers')
        require(row.get('modes')==['normal','scalar'] and row.get('workers')==[1,4] and row.get('canonical_cases')==1379 and row.get('alias_checks')==4 and row.get('timed_ids')==cfg['timed_ids'],'preparation scope incomplete')
        report_path=Path(receipt_path).parent/'report.json'
        require(sha(report_path)==row.get('report_sha256'),'preparation report seal differs')
        report=load_json(report_path);validate_preparation_report(report,cfg,row['profile'])
        for role in ('before','after'):
            pin=accepted_pin(cfg,role,row['profile']);require(row['products'][role]['product_sha256']==pin['product_sha256'],'prepared product differs')
            build=report['source_comparison']['build_identity'];sealed=row['products'][role]
            require(sealed.get('build_identity_sha256')==build['products'][role] and sealed.get('build_common_sha256')==build['common_sha256'],'prepared common build identity differs')
    return receipts

def seeded_triads(seed,rounds=7):
    require(rounds==7,'exact7 paired rounds required');rng=random.Random(seed);result=[]
    for _ in range(rounds):
        order=list(ROLES);rng.shuffle(order);result.append(order)
    return result

class SerialScheduler:
    def __init__(self):self.active=False
    def request(self,worker,command):
        require(not self.active,'overlapping backend operation forbidden');self.active=True
        try:return worker.request(command)
        finally:self.active=False


def harness(source):
    """Called only after admission metadata is authenticated."""
    source=Path(source);sys.path.insert(0,str(source/'benchmarks'));sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('controlled_r4_frozen_harness',source/'benchmarks/rank_statistics.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def early_native_admission():
    # Imports public package metadata only; no fixture/benchmark imports yet.
    import importlib
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    names=('geometric_mean','noise_filter','subtract_mean','windowed_histogram')
    registered=set(native_functions())
    require({prefix+n for prefix in ('filters.rank.','filters.rank.generic.') for n in names}<=registered,'R4 public paths absent')
    core=xx._core;xx.set_fallback_enabled(False)
    require(xx.get_num_threads()==1 and xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads,'sole core/default1 missing')
    for interface,bits in (('statistics',0xe000),('vector',0x10000)):
        capability=getattr(core,'rank_'+interface+'_capabilities',None)
        require(getattr(core,'rank_'+interface+'_interface',None)=='rank-'+interface+'-v1' and type(capability) is int and capability>=0 and capability&bits==bits,'R4 interface/capability missing')
    for name in ('rank_statistics_run','rank_windowed_histogram_run','rank_noise_footprint'):
        entry=getattr(core,name,None)
        require(inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==core.__name__,'owned R4 compiled entry missing')
    rank=importlib.import_module('skimagexx.filters.rank');generic=importlib.import_module('skimagexx.filters.rank.generic')
    for name in names:
        function=getattr(rank,name,None)
        require(callable(function) and function is getattr(generic,name,None),'actual canonical/generic identity differs')
        require(inspect.unwrap(function).__module__=='skimagexx._rank' and inspect.unwrap(function).__globals__.get('_core') is core,'forwarded/alternate-core R4 wrapper')
    return xx


@contextmanager
def no_forwarding(h):
    generic=__import__('skimage.filters.rank.generic',fromlist=['generic_cy']);cy=generic.generic_cy
    python=[h.resolve(n,True) for n in h.NAMES];codes={f.__code__ for f in python}
    fields={n:v for n,v in vars(cy).items() if callable(v) and any(part in n for part in ('geometric_mean','noise_filter','subtract_mean','windowed_hist'))}
    require(len(fields)>=4,'pinned original Cython entries missing')
    identities=tuple(fields.values());previous=sys.getprofile();hits=[]
    def forbidden(*a,**k):hits.append('Cython attribute');raise AssertionError('original rank algorithm forwarding')
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes or event=='c_call' and any(arg is value for value in identities):
            hits.append(event);raise AssertionError('original rank algorithm identity forwarding')
    try:
        for name in fields:setattr(cy,name,forbidden)
        sys.setprofile(profile);yield
    finally:
        sys.setprofile(previous)
        for name,value in fields.items():setattr(cy,name,value)
    require(not hits,'original algorithm forwarding observed')


def reference_characterization(output):
    """Every original observation, never a timer or native admission."""
    cfg=config();h=harness(ROOT);manifest=load_json(h.MANIFEST)
    for name in h.NAMES:
        function=h.resolve(name,True);facts=manifest['reference_functions'][name]
        require(str(inspect.signature(function))==facts['signature'] and hashlib.sha256(inspect.getsource(function).encode()).hexdigest()==facts['python_source_sha256'],'pinned original Python function differs')
    cases=list(h.corpus());require(len(cases)==1379,'fixed1379 differs')
    require(h.digest([h.inventory(c) for c in cases])==cfg['case_contract_sha256'],'case IDs/fixtures differ')
    frozen=manifest['reference_profiles'][h.runtime_profile()];rows=[];timed=[];failures=[]
    for index,case in enumerate(cases):
        actual=h.observe_case(h.resolve(case.name,True),case);value=h.digest(actual);eligible=case.timed and actual['error'] is None
        row=dict(case_index=index,id=h.inventory(case),observation_sha256=value,error=actual['error'],warnings=actual['warnings'],timing_eligible=eligible)
        if value!=frozen['observations'][str(index)]:row.update(actual=actual,expected_sha256=frozen['observations'][str(index)]);failures.append(row)
        rows.append(row)
        if eligible:timed.append(index)
    aliases=h.check_aliases([1],True)
    require(timed==cfg['timed_ids'] and len(timed)==376,'all376 timed IDs must remain unchanged')
    report=dict(schema='rank-statistics-comparison-reference-v1',reference_only=True,native_admission=False,barrier_eligible=False,performance_eligible=False,profile=h.runtime_profile(),status='failed' if failures else 'complete-reference-check',checks=rows,alias_checks=aliases,failures=failures,operation_timer_calls=0,contract_sha256=text_sha(CONTRACT),experiment_sha256=cfg.get('experiment_sha256'))
    write_new(output,report);require(not failures,'original observation differs; full failures retained');return report


def sample_admission(validated,measure,actions,workers,scalar,index,timed_ids):
    require(validated and measure and actions=='true','timers forbidden in check-only/Phase1')
    require(type(workers) is int and workers==1 and not scalar,'measurement must be normal/defaultworker1')
    require(index in timed_ids,'untimed/error case cannot become timed')


def startup_plan():
    nonces={role:secrets.token_hex(32) for role in ROLES}
    authentication=dict(controller_pid=os.getpid(),tool_sha256=sha(__file__),
                        startup_sha256={role:hashlib.sha256(value.encode()).hexdigest() for role,value in nonces.items()})
    return nonces,authentication


def startup_frame(authentication,role,profile,generation,attempt,launcher_pid,nonce):
    return dict(operation='authenticate-startup-v1',controller_pid=authentication['controller_pid'],
                launcher_pid=launcher_pid,role=role,profile=profile,generation=generation,attempt=attempt,
                tool_sha256=authentication['tool_sha256'],nonce=nonce)


def authenticate_startup(permit,claim,path,role,profile,generation,attempt,stream):
    """One-use inherited-stdio secret; no arbitrary ancestor traversal.

    The exclusive measurement claim binds controller/source and per-role secret
    hashes. The secret travels only on the controller-created stdin pipe. The
    exact declared Popen launcher may be the Windows venv redirector; the real
    worker must be its immediate child or the controller's direct child.
    """
    require(role in ROLES and not stream.isatty(),'direct interactive worker startup forbidden')
    authentication=claim.get('authentication')
    require(isinstance(authentication,dict) and set(authentication)=={'controller_pid','tool_sha256','startup_sha256'},'startup authentication absent')
    require(type(authentication['controller_pid']) is int and authentication['controller_pid']>0,'invalid controller identity')
    require({key:permit.get(key) for key in authentication}==authentication,'controller/source startup claim differs')
    hashes=authentication['startup_sha256']
    require(set(hashes)==set(ROLES) and len(set(hashes.values()))==3 and all(isinstance(v,str) and len(v)==64 for v in hashes.values()),'per-role startup hashes absent/duplicated')
    line=stream.readline();require(bool(line),'controller startup handshake missing')
    frame=json.loads(line)
    expected=dict(operation='authenticate-startup-v1',controller_pid=authentication['controller_pid'],role=role,
                  profile=profile,generation=generation,attempt=attempt,tool_sha256=sha(__file__))
    require({key:frame.get(key) for key in expected}==expected,'wrong controller/source/run startup handshake')
    nonce=frame.get('nonce');require(isinstance(nonce,str) and len(nonce)==64 and hashlib.sha256(nonce.encode('ascii')).hexdigest()==hashes[role],'startup secret differs')
    launcher=frame.get('launcher_pid');require(type(launcher) is int and launcher>0,'invalid startup launcher')
    parent=os.getppid();worker=os.getpid();controller=authentication['controller_pid']
    require((parent==controller and launcher==worker) or (parent==launcher and launcher!=worker),
            'worker is not the exact controller/launcher child')
    # Replaying a permit/frame cannot start another worker for the same role.
    startup_path=Path(path).with_name(Path(path).stem+'-'+role+'-stdio-startup.json')
    receipt=write_new(startup_path,dict(mechanism='one-use-inherited-stdio-v1',role=role,profile=profile,
        generation=generation,attempt=attempt,controller_pid=controller,launcher_pid=launcher,
        worker_pid=worker,worker_parent_pid=parent,tool_sha256=sha(__file__),nonce_sha256=hashes[role]))
    return dict(mechanism='one-use-inherited-stdio-v1',controller_pid=controller,launcher_pid=launcher,
                worker_pid=worker,worker_parent_pid=parent,startup_receipt_sha256=sha(receipt))


def worker_permit(path,profile,cfg,role=None,stream=None):
    require(path is not None and os.getenv('GITHUB_ACTIONS')=='true','direct worker timing without preparation permit forbidden')
    permit=load_json(path)
    require(permit['profile']==profile and permit['tool_sha256']==sha(__file__),'wrong controller/source permit')
    receipt_barrier([Path(p) for p in permit['receipts']],cfg,os.getenv('GITHUB_RUN_ID'),os.getenv('GITHUB_RUN_ATTEMPT'))
    require(sha(permit['claim'])==permit['claim_sha256'],'measurement claim differs')
    claim=load_json(permit['claim'])
    require(claim['profile']==profile and claim['generation']==os.getenv('GITHUB_RUN_ID') and claim['attempt']==os.getenv('GITHUB_RUN_ATTEMPT')
            and claim['receipt_hashes']==[sha(p) for p in permit['receipts']],'measurement claim receipt binding differs')
    return authenticate_startup(permit,claim,permit['claim'],role,profile,os.getenv('GITHUB_RUN_ID'),os.getenv('GITHUB_RUN_ATTEMPT'),stream or sys.stdin)


def worker_main(args):
    cfg=config(args.experiment,args.experiment_sha256);product_role='before' if args.role=='original' else args.role
    startup=worker_permit(args.permit,args.profile,cfg,args.role) if args.measure else None
    seal=verify_product(args.product,product_role,args.profile,cfg) # BEFORE numerical/fixture imports.
    early_native_admission() # mandatory BEFORE harness/contract fixture import.
    h=harness(Path(args.product)/'source');xx=h.xx;runtime=load_json(Path(args.product)/'runtime.json')
    # Full installed dependency, .libs, product and configured SDK against exact wheels.
    require(h.artifacts.installed_state(Path(args.product)/'wheels',runtime['wheels'])==runtime['runtime'],'installed runtime/package/SDK closure differs')
    require(Path(xx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'product outside ordinary worker prefix')
    core=xx._core;pkg=Path(xx.__file__).parent
    require(list(pkg.glob('*.pyd'))+list(pkg.glob('*.so'))==[Path(core.__file__)],'one actual ordinary-name core required')
    require(not any(k.startswith('SKIMAGEXX_') and k!='SKIMAGEXX_DISABLE_SIMD' for k in os.environ),'prototype/private selector forbidden')
    previous=xx.get_num_threads();xx.set_fallback_enabled(False);h.admit_native(previous)
    cases=list(h.corpus());require(len(cases)==1379 and h.digest([h.inventory(c) for c in cases])==cfg['case_contract_sha256'],'fixed corpus changed')
    identity=dict(role=args.role,pid=os.getpid(),source=seal['revision'],product_sha256=seal['product_sha256'],core_sha256=sha(core.__file__),core_path=str(Path(core.__file__).resolve()),python=sys.version,cpu=h.cpu_metadata(),profile=args.profile,scalar=bool(os.getenv('SKIMAGEXX_DISABLE_SIMD')),default_workers=previous,has_avx2=xx.has_avx2(),startup_authentication=startup)
    print(json.dumps(dict(ok=True,identity=identity)),flush=True);validated=False;callbacks={};expected_after={}
    for line in sys.stdin:
        request=json.loads(line)
        try:
            operation=request['operation']
            if operation=='validate':
                require(request['workers']==[1,4],'preflight workers1/4 mandatory');checks=[];mismatches=[]
                frozen=load_json(h.MANIFEST)['reference_profiles'][h.runtime_profile()]['observations']
                for workers in (1,4):
                    xx.set_num_threads(workers);observations=[];eligible=[]
                    for index,case in enumerate(cases):
                        function=h.resolve(case.name,args.role=='original')
                        if args.role=='original':observation=h.observe_case(function,case)
                        else:
                            with no_forwarding(h):observation=h.observe_case(function,case)
                        observations.append(observation);expected_after[index]=observation['after']
                        if case.timed and observation['error'] is None:eligible.append(index)
                        if h.digest(observation)!=frozen[str(index)]:mismatches.append(dict(case_index=index,workers=workers,actual=observation,expected_sha256=frozen[str(index)]))
                    require(eligible==cfg['timed_ids'],'all376 eligible source IDs must remain unchanged')
                    checks.append(dict(workers=workers,digests=[h.digest(v) for v in observations],errors=[v['error'] for v in observations],warnings=[v['warnings'] for v in observations]))
                xx.set_num_threads(1);aliases=[]
                for name in h.NAMES:
                    require(h.resolve(name,args.role=='original') is h.resolve(name,args.role=='original',alias=True),'canonical/alias identity differs')
                    index=next(i for i,c in enumerate(cases) if c.name==name);function=h.resolve(name,args.role=='original',alias=True)
                    if args.role=='original':value=h.observe_case(function,cases[index])
                    else:
                        with no_forwarding(h):value=h.observe_case(function,cases[index])
                    require(h.digest(value)==checks[0]['digests'][index],'alias observation differs');aliases.append(name)
                validated=True;reply=dict(checks=checks,aliases=aliases,mismatches=mismatches)
            elif operation in ('warm','sample'):
                index=request['case_index']
                sample_admission(validated,args.measure,os.getenv('GITHUB_ACTIONS'),xx.get_num_threads(),identity['scalar'],index,cfg['timed_ids'])
                if index not in callbacks:callbacks[index]=h.timing_callback(h.resolve(cases[index].name,args.role=='original'),cases[index])
                callback,tracked=callbacks[index]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if operation=='warm':result=callback();del result;reply=dict(warmed=True)
                    else:
                        loops=request['loops'];require(type(loops) is int and 1<=loops<=1000,'invalid sample loops')
                        # IPC, fixture setup, validation and serialization remain outside.
                        from time import perf_counter_ns,sleep
                        sleep(.02) # same cooldown as existing run.batch, outside timer.
                        start=perf_counter_ns()
                        for _ in range(loops):result=callback();del result
                        elapsed=perf_counter_ns()-start
                        reply=dict(ms=elapsed/loops/1e6,elapsed_ns=elapsed,loops=loops,identity=identity)
                require({k:h.qualified.array(v) for k,v in tracked.items()}==expected_after[index],'timed callback changed workload state')
                require(identity['cpu']==h.cpu_metadata(),'worker CPU identity changed')
            elif operation=='integrity':
                require(verify_product(args.product,product_role,args.profile,cfg)==seal,'capsule changed')
                require(h.artifacts.installed_state(Path(args.product)/'wheels',runtime['wheels'])==runtime['runtime'],'installed closure changed');reply=dict(seal=seal)
            elif operation=='close':print(json.dumps(dict(ok=True,result={})),flush=True);return
            else:raise RuntimeError('unknown worker request')
            print(json.dumps(dict(ok=True,result=reply)),flush=True)
        except BaseException as error:print(json.dumps(dict(ok=False,error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),operation=request,last_case=locals().get('index'),partial_checks=locals().get('checks'),last_observation=locals().get('observation'))),flush=True)


class Worker:
    def __init__(self,python,product,role,profile,output,measure=False,scalar=False,permit=None,experiment=None,experiment_sha256=None,startup_nonce=None):
        self.stderr=Path(output).open('x',encoding='utf-8');env=dict(os.environ)
        for key in list(env):
            if key.startswith('SKIMAGEXX_') or key in ('PYTHONPATH','PYTHONHOME'):env.pop(key)
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
        if scalar:env['SKIMAGEXX_DISABLE_SIMD']='1'
        command=[str(python),'-u',str(Path(__file__).resolve()),'worker','--profile',profile,'--product',str(Path(product).resolve()),'--role',role]
        if experiment:command+=['--experiment',str(experiment),'--experiment-sha256',experiment_sha256]
        if measure:command+=['--measure','--permit',str(permit)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,env=env)
        if measure:
            permit_data=load_json(permit)
            frame=startup_frame(permit_data,role,profile,os.getenv('GITHUB_RUN_ID'),os.getenv('GITHUB_RUN_ATTEMPT'),self.process.pid,startup_nonce)
            self.process.stdin.write(json.dumps(frame)+'\n');self.process.stdin.flush()
        hello=self._read();self.identity=hello['identity']
    def _read(self):
        line=self.process.stdout.readline();require(bool(line),'worker exited; retained stderr contains failure')
        reply=json.loads(line);require(reply.get('ok') is True,'worker failure: '+json.dumps(reply));return reply
    def request(self,command):
        self.process.stdin.write(json.dumps(command)+'\n');self.process.stdin.flush();return self._read()['result']
    def close(self):
        self.request(dict(operation='close'));require(self.process.wait()==0,'worker terminal failure');self.stderr.close()


def validate_workers(workers,scheduler):
    observations={role:scheduler.request(worker,dict(operation='validate',workers=[1,4])) for role,worker in workers.items()}
    reference=observations['original']
    if not all(value==reference and not value['mismatches'] for value in observations.values()):
        error=RuntimeError('full1379/4alias observations differ');error.observations=observations;raise error
    require(len(reference['checks'])==2 and all(len(r['digests'])==1379 for r in reference['checks']) and len(reference['aliases'])==4,'incomplete worker parity')
    require(reference['checks'][0]['digests']==reference['checks'][1]['digests'],'worker1/4 bits differ')
    return observations


def operation_profiles(profile):return [('original','before'),('before','before'),('after','after')]
def controller(args):
    cfg=config(args.experiment,args.experiment_sha256);require(args.output and not Path(args.output).exists(),'fresh output required; never reuse receipts/results')
    products={role:Path(getattr(args,role)) for role in ('before','after')}
    for role,path in products.items():verify_product(path,role,args.profile,cfg)
    source_comparison=verify_pair_sources(products['before'],products['after'],args.profile,cfg)
    measure=not args.check_only
    if measure:
        require(os.getenv('GITHUB_ACTIONS')=='true','no local operation timings')
        receipt_barrier(args.receipts,cfg,os.getenv('GITHUB_RUN_ID'),os.getenv('GITHUB_RUN_ATTEMPT'))
        require(args.repeat==7,'exact7 paired triads required')
    else:require(not args.receipts,'check-only cannot consume measurement barrier')
    output=Path(args.output);output.mkdir(parents=True);report=dict(schema='rank-statistics-controlled-v1',session_id=str(uuid.uuid4()),source_comparison=source_comparison,worker_identities={},status='preflight',profile=args.profile,check_only=args.check_only,reference_only=False,results=[],checks={},failures=[],contract_sha256=text_sha(CONTRACT),experiment_sha256=cfg.get('experiment_sha256'))
    # Exclusive local marker plus run/attempt binding rejects rerun/replay.
    permit=None;startup_nonces={}
    if measure:
        startup_nonces,authentication=startup_plan()
        claim=claim_measurement(args.receipts,args.profile,os.getenv('GITHUB_RUN_ID'),os.getenv('GITHUB_RUN_ATTEMPT'),os.getenv('RUNNER_TEMP'),authentication)
        write_new(output/'receipt-claim.json',dict(claim_path=str(claim),receipts={str(p):sha(p) for p in args.receipts}))
        permit=output/'worker-permit.json'
        write_new(permit,dict(profile=args.profile,**authentication,claim=str(claim),claim_sha256=sha(claim),receipts=[str(Path(p).resolve()) for p in args.receipts]))
    scheduler=SerialScheduler();workers={}
    try:
        modes=('normal','scalar') if args.check_only else ('normal',)
        for mode in modes:
            workers={}
            for role,product_role in operation_profiles(args.profile):
                workers[role]=Worker(getattr(args,product_role+'_python'),products[product_role],role,args.profile,output/(role+'-'+mode+'.stderr'),measure,mode=='scalar',permit,args.experiment,args.experiment_sha256,startup_nonces.get(role))
            require(len({w.identity['pid'] for w in workers.values()})==3,'separate ordinary worker processes required')
            require(workers['before'].identity['core_path']!=workers['after'].identity['core_path'] and workers['before'].identity['core_sha256']!=workers['after'].identity['core_sha256'],'before/after cores must be physically and byte-distinct')
            require(all(w.identity['cpu']==workers['original'].identity['cpu'] for w in workers.values()),'same-host CPU session required')
            report['worker_identities'][mode]={role:worker.identity for role,worker in workers.items()}
            # Full untimed workers1/4 validation completes before ANY warm/calibration/timing.
            # The fresh measurement job has no concurrent builds or correctness tests.
            report['checks'][mode]=validate_workers(workers,scheduler)
            if measure:
                report['triad_order']={}
                for index in cfg['timed_ids']:
                    orders=seeded_triads(cfg['seed']+index,7);report['triad_order'][str(index)]=orders
                    for worker in workers.values():scheduler.request(worker,dict(operation='warm',case_index=index))
                    first={role:scheduler.request(workers[role],dict(operation='sample',case_index=index,loops=1)) for role in ROLES}
                    loops={role:max(1,min(1000,int(15/max(first[role]['ms'],.001)))) for role in ROLES};samples={role:[] for role in ROLES}
                    raw=[]
                    for round_index,order in enumerate(orders):
                        for role in order:
                            sample=scheduler.request(workers[role],dict(operation='sample',case_index=index,loops=loops[role]));samples[role].append(sample['ms']);raw.append(dict(round=round_index,role=role,**sample))
                    medians={role:statistics.median(samples[role]) for role in ROLES}
                    report['results'].append(dict(case_index=index,inventory=cfg['case_inventory'][index],calibration=first,samples_ms=samples,raw_samples=raw,medians_ms=medians,before_speedup=medians['original']/medians['before'],after_speedup=medians['original']/medians['after'],improvement=medians['before']/medians['after'],slower_than_before=medians['after']>medians['before'],slower_than_original=medians['after']>medians['original']))
                    (output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            for worker in workers.values():scheduler.request(worker,dict(operation='integrity'));worker.close()
            workers={}
        require(verify_pair_sources(products['before'],products['after'],args.profile,cfg)==source_comparison,'source comparison changed')
        report['status']='complete-parity-check' if args.check_only else 'complete-controlled-measurement'
    except BaseException as error:report['status']='failed';report['failures'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),observations=getattr(error,'observations',None)));raise
    finally:
        # Never cancel/restart a running operation; close only at message boundaries.
        for worker in workers.values():
            if worker.process.poll() is None:worker.close()
        write_new(output/'report.json',report)
    if args.check_only:
        validate_preparation_report(report,cfg,args.profile)
        write_new(output/'preparation.json',dict(schema='rank-statistics-comparison-preparation-v1',receipt_id=str(uuid.uuid4()),generation=os.getenv('GITHUB_RUN_ID'),attempt=os.getenv('GITHUB_RUN_ATTEMPT'),profile=args.profile,contract_sha256=text_sha(CONTRACT),experiment_sha256=cfg.get('experiment_sha256'),report_sha256=sha(output/'report.json'),reference_only=False,native_admission=True,modes=['normal','scalar'],workers=[1,4],canonical_cases=1379,alias_checks=4,timed_ids=cfg['timed_ids'],products={role:verify_product(path,role,args.profile,cfg) for role,path in products.items()}))
    return report


def startup_probe(python,output,profile):
    output=Path(output);require(not output.exists(),'fresh probe output required');output.mkdir(parents=True)
    nonces,authentication=startup_plan();generation='metadata-only';attempt='1'
    claim=write_new(output/'claim.json',dict(authentication=authentication))
    permit=dict(profile=profile,**authentication,claim=str(claim));write_new(output/'permit.json',permit)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    with (output/'child.stderr').open('x') as error:
        process=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),'startup-probe-child',
            '--permit',str((output/'permit.json').resolve()),'--profile',profile],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=error,text=True,env=env)
        frame=startup_frame(authentication,'before',profile,generation,attempt,process.pid,nonces['before'])
        stdout,_=process.communicate(json.dumps(frame)+'\n')
    write_new(output/'result.json',dict(schema='rank-statistics-startup-probe-v1',probe_only=True,
        native_admission=False,barrier_eligible=False,operation_timer_calls=0,controller_pid=os.getpid(),
        popen_pid=process.pid,returncode=process.returncode,reply=json.loads(stdout) if stdout else None,
        interpreter=str(python),tool_sha256=sha(__file__)))
    require(process.returncode==0,'metadata startup child failed; stderr retained')
    return load_json(output/'result.json')


def startup_probe_child(args):
    permit=load_json(args.permit);claim=load_json(permit['claim'])
    require(permit['tool_sha256']==sha(__file__),'probe source differs')
    result=authenticate_startup(permit,claim,permit['claim'],'before',args.profile,'metadata-only','1',sys.stdin)
    print(json.dumps(dict(probe_only=True,native_admission=False,barrier_eligible=False,operation_timer_calls=0,
                         executable=sys.executable,base_executable=getattr(sys,'_base_executable',None),**result)),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','reference','worker','compare','verify-product','verify-pair','capture-build-identity','barrier','selectors','startup-probe','startup-probe-child'])
    p.add_argument('--check-only',action='store_true');p.add_argument('--reference-only',action='store_true');p.add_argument('--measure',action='store_true');p.add_argument('--profile',choices=PROFILES)
    for name in ('output','product','before','after','before-python','after-python','permit','experiment','python','facts'):p.add_argument('--'+name,type=Path)
    p.add_argument('--experiment-sha256');p.add_argument('--role',choices=ROLES);p.add_argument('--receipts',type=Path,nargs='*',default=[]);p.add_argument('--repeat',type=int,default=7)
    args=p.parse_args(argv)
    if args.mode=='capture-build-identity':
        require(args.facts and args.output and not args.measure,'actual build facts/output required; capture never measures');print(json.dumps(capture_build_identity(args.facts,args.output),indent=2));return
    if args.mode=='startup-probe':require(args.python and args.output and args.profile,'probe interpreter/output/profile required');startup_probe(args.python,args.output,args.profile);return
    if args.mode=='startup-probe-child':startup_probe_child(args);return
    if args.mode=='plan':print(json.dumps(dict(phase='Phase1 original-only/native0',products=config()['accepted_products'],canonical_cases=1379,aliases=4,timed_cases=376,worker_limits=[1,4],measurement_workers=[1],rounds=7,planned_scope=config()['planned_scope'],external_experiment_required=True,blocked_source=BLOCKED_SOURCE),indent=2));return
    if args.mode=='reference':require(args.reference_only and args.check_only and not args.measure and args.output,'reference requires --reference-only --check-only/output');reference_characterization(args.output);return
    require(not args.reference_only,'reference receipt cannot enter native/barrier/measurement')
    if args.mode=='worker':require(args.product and args.role and args.profile,'worker product/profile/role required');worker_main(args)
    elif args.mode=='verify-product':print(json.dumps(verify_product(args.product,args.role,args.profile,config(args.experiment,args.experiment_sha256)),indent=2))
    elif args.mode=='verify-pair':
        require(args.before and args.after and args.profile and not args.measure,'two qualified products/profile required')
        cfg=config(args.experiment,args.experiment_sha256)
        for role,path in (('before',args.before),('after',args.after)):verify_product(path,role,args.profile,cfg)
        print(json.dumps(verify_pair_sources(args.before,args.after,args.profile,cfg),indent=2))
    elif args.mode=='barrier':receipt_barrier(args.receipts,config(args.experiment,args.experiment_sha256),os.getenv('GITHUB_RUN_ID'),os.getenv('GITHUB_RUN_ATTEMPT'))
    elif args.mode=='selectors':
        cfg=config(args.experiment,args.experiment_sha256)
        for role in ('before','after'):
            pin=accepted_pin(cfg,role,args.profile);print(role+'_run='+pin['artifact_run']);print(role+'_artifact='+pin['artifact_name'])
    else:require(args.profile and args.before and args.after and args.before_python and args.after_python,'two complete products/ordinary interpreters required');controller(args)
if __name__=='__main__':main()
