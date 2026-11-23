"""Bind admitted one-day current306 transport on a fresh, exact Windows timing host."""

import argparse
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import os
import re
import runpy
import shutil
import stat
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
SOURCE = '5b94b20d30e572164a6473c8bb029e18d4a7e706'
TREE = '2ab7c8bd13965df3d48e565b1c6b779739d0974c'
CARRIER = 'dea051907025d8ecc92599a7f969fc5e1a0672b5'
PRODUCER_RUN = 45090847657
PRODUCER_JOB = 137419805469
SCOPE = 'current306-windows-timing-host-preparation-v1'
FOCUS = 'current306-windows-focused-receipts-v3'
TRANSPORT = 'current306-windows-timing-transport-v3'
FILE_PINS = {
    'run.py':'2089cc51a9b73aaa847783ee924d9b04f64df7052f97096c0049b47a755ca2db',
    'plan-v1.json':'cd0f2ef1f62d67561181ed3b7ff5ba62dd36a849fafe0fd7d778268d781adc31',
}
MEASUREMENT = '104ba954d9e9b0cc809a21b8d37f59599b29777d88a7f5b818f7ad5565d5c6b7'
THREADPOOLCTL = '12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b'
STOCK_TEMPLATE = '438b679850f916765ff668bb5f5dd7cbfa7a973c720757f4321a9416aeff1f31'
COMPLETE_SOURCE = 'dddc2f67d7a758f86af6131aa73f95b37da4a6c3bb59f0f608a02c6fb1bd714e'
RECEIPTS = {'measurement.json','runtime-lock.txt','terminal.json','producer-package.json',
            'producer-interpreter-provenance.json','producer-sdk.json',
            'public-normal.json','public-scalar.json','public-normal.xml','public-scalar.xml',
            'sdk-normal.xml','sdk-scalar.xml'}


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def preview(runner):
    assert {name:digest(runner/name) for name in FILE_PINS} == FILE_PINS
    fixed = runpy.run_path(str(runner/'run.py'))
    assert (fixed['HEAD'],fixed['TREE'],fixed['VERSIONS']['python']) == (SOURCE,TREE,'3.13.16')
    plan = read(runner/'plan-v1.json')
    assert plan['protocol']['accepted_total'] == 280
    assert not {'numpy','scipy','skimage','skimagexx','pytest','threadpoolctl'} & sys.modules.keys()
    return fixed,plan


def authorize(path, expected):
    assert digest(path) == expected
    grant = read(path)
    assert (grant['scope'],grant['authorized_by'],grant['source'],grant['tree'],grant['producer_carrier'],
            grant['producer_run'],grant['producer_job'],grant['launcher_sha256']) == (
        SCOPE,'/root',SOURCE,TREE,CARRIER,PRODUCER_RUN,PRODUCER_JOB,digest(Path(__file__)))
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_RUN_ATTEMPT') == '1'
    assert os.environ.get('GITHUB_REPOSITORY') == 'bipxx/scikit-imagexx'
    assert grant['timing_carrier'] == os.environ['GITHUB_SHA']
    assert os.name == 'nt' and sys.version.split()[0] == '3.13.16' and sys.maxsize == 2**63-1
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    assert grant['timing_authorized'] is False
    deadline = datetime.fromisoformat(grant['deadline_utc'])
    assert deadline.tzinfo is not None and deadline.utcoffset().total_seconds() == 0
    assert datetime.now(timezone.utc) < deadline
    assert set(grant['artifacts']) == {FOCUS,TRANSPORT}
    for name,evidence in grant['artifacts'].items():
        assert (evidence['name'],evidence['run'],evidence['job'],evidence['carrier'],evidence['conclusion']) == (
            name,PRODUCER_RUN,grant['producer_job'],CARRIER,'success')
        assert type(evidence['id']) is int and evidence['id'] > 0 and not evidence['expired']
        assert re.fullmatch('sha256:[0-9a-f]{64}',evidence['digest'])
        assert datetime.now(timezone.utc) < datetime.fromisoformat(evidence['expires_at'])
    assert grant['artifacts'][FOCUS]['id'] != grant['artifacts'][TRANSPORT]['id']
    return grant,deadline


def suite(path,count):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
    rows = root.findall('.//testcase')
    assert len(rows) == count and sum(int(s.attrib['tests']) for s in suites) == count
    assert not any(int(s.get(k,'0')) for s in suites for k in ('errors','failures','skipped'))
    assert not any(child.tag in ('error','failure','skipped') for row in rows for child in row)
    return rows


def validate_transport(grant, args, fixed, plan):
    proof = args.focus
    assert set(grant['files']) == RECEIPTS
    for name, expected in grant['files'].items():
        assert digest(proof/name) == expected, name
    for name in ('measurement.json','runtime-lock.txt','terminal.json'):
        assert digest(args.transport_receipts/name) == grant['files'][name]
    terminal = read(proof/'terminal.json')
    assert (terminal['status'],terminal['source'],terminal['tree'],terminal['profile'],
            terminal['modes'],terminal['sdk_checks'],terminal['complete_source_receipt_sha256']) == (
        'passed_focused_windows',SOURCE,TREE,'windows',['normal','scalar'],10,COMPLETE_SOURCE)
    assert terminal['timing_authorized'] is False and terminal['full_qualification'] is False
    product = read(proof/'producer-package.json')
    sdk = read(proof/'producer-sdk.json')
    assert product['source_revision'] == SOURCE and product['product_scope'] == 'current306' and product['record_verified']
    assert len(product['members_sha256']) == 282 and len(product['SDK_sha256']) == 125
    assert sdk['checks'] == 10 and sdk['families'] == list(fixed['FAMILIES'])
    assert sdk['SDK_sha256'] == product['SDK_sha256']
    public = [read(proof/f'public-{mode}.json') for mode in ('normal','scalar')]
    for mode, receipt in zip(('normal','scalar'),public,strict=True):
        assert (receipt['source'],receipt['tree'],receipt['mode'],receipt['core']) == (
            SOURCE,TREE,mode,product['sole_core'])
        assert receipt['pytest_exit'] == 0 and receipt['product'] == terminal['product']
        assert len(receipt['node_ids']) == len(set(receipt['node_ids'])) == terminal['public_checks'][mode] > 0
        assert {name for name,count in receipt['family_counts'].items() if count>0} == {
            f'tests/test_{family}_public.py' for family in fixed['FAMILIES']}
        suite(proof/f'public-{mode}.xml',len(receipt['node_ids']))
        sdk_rows = suite(proof/f'sdk-{mode}.xml',5)
        assert {row.get('name') for row in sdk_rows} == {f'installed_{name}_consumer' for name in fixed['FAMILIES']}
    assert public[0]['node_ids'] == public[1]['node_ids']
    manifest = read(proof/'measurement.json')
    assert manifest['schema'] == 'ci-offline-measurement-v1' and manifest['source']['revision'] == SOURCE
    assert manifest['lock_sha256'] == grant['files']['runtime-lock.txt']
    assert manifest['runtime']['sdk_sha256'] == sdk['SDK_sha256']
    assert manifest['runtime']['cache_tag'] == 'cpython-313' and manifest['runtime']['system'] == 'Windows'
    base = read(proof/'producer-interpreter-provenance.json')['base']
    assert base['version'] == manifest['runtime']['python'] == sys.version
    assert base['sha256'] == digest(sys.executable)
    assert fixed['loaded_python_dll']().parent == Path(sys.executable).resolve().parent
    wheels = manifest['wheels']
    assert len(wheels) == 15 and len({rec['name'] for rec in wheels.values()}) == 15
    assert {name:member['sha256'] for name,member in wheels.items()} == terminal['transport_sha256']
    assert {path.name for path in args.wheels.iterdir()} == set(wheels)
    by_distribution = {}
    for name, member in wheels.items():
        assert Path(name).name == name and name.endswith('.whl')
        assert digest(args.wheels/name) == member['sha256']
        by_distribution[member['name']] = (name,member)
    for name,sha in plan['stock_wheels'].items():
        assert by_distribution[name][1]['sha256'] == sha
    wheel_name, candidate_record = by_distribution['scikit-imagexx']
    assert candidate_record['sha256'] == terminal['product']['wheel'] == product['wheel_sha256']
    sorted_lock = [f"{name}=={r['version']} --hash=sha256:{r['sha256']}"
                   for name,(_,r) in sorted(by_distribution.items())]
    assert (proof/'runtime-lock.txt').read_text(encoding='utf8').splitlines() == sorted_lock
    return terminal,product,manifest,wheel_name,sorted_lock


def extract_wheel(wheel, product, terminal, candidate):
    want = product['members_sha256']
    with ZipFile(wheel) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        assert len(names) == len(set(n.casefold() for n in names)) == 282 and set(names) == set(want)
        record_name, = [name for name in names if name.endswith('.dist-info/RECORD')]
        assert want[record_name] == terminal['product']['record']
        core_name = product['sole_core']
        assert core_name == 'skimagexx/_core.cp313-win_amd64.pyd' and want[core_name] == terminal['product']['core']
        with archive.open(record_name) as stream:
            raw = stream.read(256*2**10+1)
        assert len(raw) <= 256*2**10 and sha256(raw).hexdigest() == want[record_name]
        rows = list(csv.reader(StringIO(raw.decode('utf8'),newline='')))
        index = {name:(code,size) for name,code,size in rows}
        assert len(rows) == len(index) == len(names) and set(index) == set(names) and index[record_name] == ('','')
        assert sum(entry.file_size for entry in entries) < 150*2**20
        candidate.mkdir()
        for entry in entries:
            name = entry.filename
            relative = PurePosixPath(name)
            assert relative.as_posix() == name and not relative.is_absolute() and '..' not in relative.parts
            assert '\\' not in name and ':' not in name and not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            target = candidate.joinpath(*relative.parts)
            assert target.resolve().is_relative_to(candidate)
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(entry) as stream, target.open('xb') as dest:
                shutil.copyfileobj(stream,dest,2**20)
            assert target.stat().st_size == entry.file_size and digest(target) == want[name], name
            if name != record_name:
                token = urlsafe_b64encode(bytes.fromhex(want[name])).decode().rstrip('=')
                assert index[name] == ('sha256='+token,str(entry.file_size)), name
    return record_name,core_name,names


def prepare(args,grant,deadline,fixed,plan):
    assert all(os.environ.get(name) == '1' for name in fixed['LIMITS'])
    assert not any(name.startswith('SKIMAGEXX_') or name in {
        'PYTHONPATH','PYTHONHOME','NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE','MKL_DEBUG_CPU_TYPE',
        'LD_PRELOAD','GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GH_TOKEN'} for name in os.environ)
    for name in ('runner','stock_template','source','focus','transport_receipts','wheels','scratch','evidence','threadpoolctl'):
        path = getattr(args,name)
        assert path is not None and path.is_absolute()
        setattr(args,name,path.resolve(strict=True))
    temp = Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
    assert args.scratch.parent == temp and args.scratch.name.startswith('current306-timing-')
    assert not args.evidence.is_relative_to(args.scratch) and not args.evidence.is_relative_to(args.source)
    assert args.wheels.is_relative_to(args.scratch) and args.transport_receipts.is_relative_to(args.scratch)
    disk_free = shutil.disk_usage(args.scratch).free
    assert disk_free >= 32*2**30
    candidate,stock,lock = (args.scratch/name for name in ('candidate','stock','stock-lock.txt'))
    binding_file, prep_file = (args.evidence/name for name in ('timing-binding.json','timing-prepared.json'))
    assert not any(p.exists() for p in (candidate,stock,lock,binding_file,prep_file))
    run_entry = args.runner/'run.py'
    helper = args.source/'tools/ci_measurement_artifacts.py'
    assert digest(helper) == MEASUREMENT
    assert digest(args.threadpoolctl) == THREADPOOLCTL
    assert digest(args.stock_template) == STOCK_TEMPLATE == grant['stock_template_sha256']
    git = Path(shutil.which('git.exe')).resolve(strict=True)
    assert fixed['source_entry'](dict(source_root=str(args.source),git=ref(git)),plan)==args.source
    terminal,product,manifest,wheel_name,locked = validate_transport(grant,args,fixed,plan)
    assert datetime.now(timezone.utc) < deadline
    lock.write_text('\n'.join(line for line in locked if not line.startswith('scikit-imagexx=='))+'\n',encoding='utf8')
    def checked(command, filename):
        with (args.evidence/filename).open('xb') as out:
            subprocess.run([str(item) for item in command],stdout=out,stderr=subprocess.STDOUT,check=True)
    checked([sys.executable,'-I','-B','-m','pip','--isolated','--disable-pip-version-check','install',
             '--no-index','--find-links',args.wheels,'--no-deps','--require-hashes','--no-cache-dir','--no-compile',
             '--target',stock,'-r',lock],'offline-install.log')
    record,core,names = extract_wheel(args.wheels/wheel_name,product,terminal,candidate)
    # Producer already ran freeze/verify. Check the very same manifest on the new signer and split sites.
    code = ('import runpy,sys; script,candidate,stock,wheels,manifest=sys.argv[1:]; '
            'sys.path[:0]=[candidate,stock]; sys.argv=[script,"verify","--wheel-dir",wheels,"--artifact",manifest]; '
            'runpy.run_path(script,run_name="__main__")')
    checked([sys.executable,'-I','-S','-B','-c',code,helper,candidate,stock,args.wheels,args.focus/'measurement.json'],
            'offline-verify.log')
    with (stock/'threadpoolctl.py').open('xb') as stream:
        stream.write(args.threadpoolctl.read_bytes())
    stock_data = read(args.stock_template)
    assert (stock_data['source'],stock_data['tree'],stock_data['scope']) == (SOURCE,TREE,fixed['SCOPE'])
    pools = []
    for name,item in stock_data['stock'].items():
        assert name in fixed['STOCK_MODULES'] and set(item['modules']) == fixed['STOCK_MODULES'][name]
        dist, = stock.glob(name.replace('-','_')+'-'+item['version']+'.dist-info')
        item.update(site=str(stock),record=ref(dist/'RECORD'))
        if 'openblas_member' in item:
            pool = ref(stock/item['openblas_member'])
            assert pool['sha256'] == manifest['runtime']['dependency_libraries_sha256'][name][
                Path(item['openblas_member']).name]
            pools.append(pool)
    assert len(pools) == 2
    # Record the canonical system VCOMP on this fresh host. The runner checks
    # which DLL was actually loaded after its authenticated scientific imports.
    system_vcomp = Path(os.environ['SystemRoot'])/'System32'/'vcomp140.dll'
    pools.append(ref(system_vcomp))
    assert len(pools) == 3
    # These paths remain hypotheses about actual process-loaded pools until fixed20 verifies them.
    binding = dict(scope=fixed['SCOPE'],source=SOURCE,tree=TREE,source_root=str(args.source),git=ref(git),
                   python=ref(sys.executable),python_dll=ref(fixed['loaded_python_dll']()),
                   producer=ref(args.focus/'terminal.json'),focused_public=ref(args.focus/'public-normal.json'),
                   relocated_sdk=ref(args.focus/'producer-sdk.json'),stock=stock_data['stock'],loaded_pools=pools,
                   threadpoolctl=ref(stock/'threadpoolctl.py'),
                   candidate=dict(site=str(candidate),record=ref(candidate/record),core=ref(candidate/core),
                                  distribution='scikit-imagexx',version='0.1.0',wheel_tag='cp313-cp313-win_amd64',
                                  wheel_sha256=product['wheel_sha256'],wheel_members=names))
    write(binding_file,binding)
    assert datetime.now(timezone.utc) < deadline
    checked([sys.executable,'-I','-S','-B',run_entry,'--check-binding',binding_file,'--binding-sha256',
             digest(binding_file),'--plan-sha256',FILE_PINS['plan-v1.json']],'fixed20-entry.log')
    write(prep_file,dict(source=SOURCE,tree=TREE,producer_artifacts=grant['artifacts'],binding=ref(binding_file),
                        entry=ref(args.evidence/'fixed20-entry.log'),offline_verify=ref(args.evidence/'offline-verify.log'),
                        candidate_core=binding['candidate']['core'],host=os.environ['COMPUTERNAME'],
                        scope=SCOPE,free_at_entry=disk_free,authority='preparation only',timing_authorized=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    for name in ('runner','stock-template','source','focus','transport-receipts','wheels','scratch','evidence','threadpoolctl','approval'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--approval-sha256')
    args = parser.parse_args()
    if not args.preview and not (args.approval and args.approval_sha256):
        print('REFUSED_NO_ROOT_PREPARATION_AUTHORITY',file=sys.stderr)
        return 2
    fixed,plan = preview(args.runner.resolve(strict=True))
    if args.preview:
        print(json.dumps(dict(source=SOURCE,tree=TREE,producer_run=PRODUCER_RUN,transport=TRANSPORT,
                              cases=sum(map(len,plan['cases'].values())),scientific_imports=0,timing_authorized=False)))
        return 0
    grant,deadline = authorize(args.approval,args.approval_sha256)
    prepare(args,grant,deadline,fixed,plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
