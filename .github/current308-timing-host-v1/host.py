"""One conditionally authorized current308 Windows preparation and timing job."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib import error, parse, request
from zipfile import ZipFile


SCOPE = 'current308-windows-hosted-fixed8-conditional-v1'
SOURCE = 'e44a720dd2c84eedbd7e75641829c6c4f76d9a6c'
TREE = 'c501bff41f16a4eecbde630a7d16bfa1884ceac8'
CARRIER = 'c8e9a8c071877df0e79a6b7c75228fdd0ec360cf'
RUN = 45108014478
JOB = 137473654127
REPO = 'bipxx/scikit-imagexx'
LIVE = {'current308-windows-focused-receipts-v1': (14788827471, 'sha256:ea5d8aad6eeff02a48dbe885eaf9d12dd7ca398835bdd8bc0a4daca14cafa257'), 'current308-windows-timing-transport-v1': (14788767654, 'sha256:bb0394ecaf2e3e8ce7531c2e169fd40dc10f51c24a58fbaeb4856408aabc73d4')}  # Bind only after root admits actual production and SDK receipts.
PINS = {
    'prepare': '803e12fc4eecf3d162a8a127920a7d507bb1ee0dc30ffaa23f8c185023617c48',
    'runner': '160427aeb7e330d418f975615a436001223373ccab800da14d5079606b10184e',
    'plan': '39f953f927d3e0a311f88e9b06ac85e0c8931be8b65ceb6d31af7cb143b01a1b',
    'stock_template': '91c31f4616b0ed78925a2d17ea6d02129b4e243d9a0c695662e58c1e29fd0176',
    'threadpoolctl': '12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b',
}
FLOOR = 32*2**30
TRANSPORT_BYTES = 86530626
ADMITTED = '872a7bc8baa6c2723214e5f31c13e8c6a59c5d5db26ca2da6947a4a2c5f215b8'
READBACK = 'ed0b9520cf989c6742cc174df1075faa0ca8bf06cc5f6bc96e9f55f430007617'


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def ref(path):
    return dict(path=str(path),sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def authorize(args):
    assert all((LIVE, TRANSPORT_BYTES, ADMITTED, READBACK)), "Unbound current308 compact and transport identities"
    assert digest(args.approval) == args.approval_sha256
    approval = read(args.approval)
    assert (approval['scope'],approval['authorized_by'],approval['intent'],approval['source'],
            approval['tree'],approval['producer_carrier'],approval['producer_run'],approval['producer_job']) == (
        SCOPE,'/root','one conditional preparation and timing run',SOURCE,TREE,CARRIER,RUN,JOB)
    assert approval['timing_carrier'] == os.environ['GITHUB_SHA']
    assert approval['github_actor'] == os.environ['GITHUB_ACTOR']
    assert os.environ['GITHUB_ACTIONS'] == 'true' and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    assert os.environ['GITHUB_REPOSITORY'] == REPO and os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch'
    event = read(os.environ['GITHUB_EVENT_PATH'])
    assert event['sender']['login'] == approval['github_actor']
    assert event['inputs']['root_authority_sha256'] == args.approval_sha256
    assert os.name == 'nt' and sys.maxsize == 2**63-1 and sys.version.split()[0] == '3.13.16'
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    assert approval['payload_sha256'] == PINS|{'host':digest(Path(__file__))}
    assert approval['contract'] == dict(exclusive=True,cpu=0,workers=1,cases=8,accepted_batches=112)
    deadline = datetime.fromisoformat(approval['deadline_utc'])
    assert deadline.tzinfo is not None and deadline.utcoffset().total_seconds() == 0
    assert datetime.now(timezone.utc) < deadline
    admitted = args.product_admission.resolve(strict=True)
    assert digest(admitted) == approval['root_product_sdk_admission_sha256'] == ADMITTED
    receipt = read(admitted)
    assert (receipt['source'], receipt['tree'], receipt['root_readback_sha256'],
            receipt['producer_run'], receipt['producer_job'], receipt['public_passes'],
            receipt['subtests'], receipt['SDK_passes']) == (SOURCE, TREE, READBACK, RUN, JOB, 122, 36, 4)
    assert receipt['product']['pins'] == approval['product']
    assert receipt['product']['member_count'] == 286 and receipt['product']['sdk_member_count'] == 127
    readback = args.compact_readback.resolve(strict=True)
    assert digest(readback) == READBACK
    compact = read(readback)
    assert compact['status'] == 'focused_windows_receipts_verified'
    assert (compact['run'], compact['job'], compact['carrier']) == (RUN, JOB, CARRIER)
    assert all(compact['compact_file_sha256'][name] == pin for name,pin in approval['files'].items())
    for name,(artifact,sha) in LIVE.items():
        assert (approval['artifacts'][name]['id'],approval['artifacts'][name]['digest']) == (artifact,sha)
    return approval,deadline


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def receive(approval,scratch,evidence,terminal):
    headers = {'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'}
    def api(path):
        req = request.Request(f'https://api.github.com/repos/{REPO}/actions/{path}',headers=headers)
        with request.urlopen(req,timeout=20) as reply:
            return json.load(reply)
    run,job = api(f'runs/{RUN}'),api(f'jobs/{JOB}')
    assert (run['id'],run['head_sha'],run['status'],run['conclusion']) == (RUN,CARRIER,'completed','success')
    assert (job['id'],job['run_id'],job['head_sha'],job['status'],job['conclusion']) == (JOB,RUN,CARRIER,'completed','success')
    observed = {}
    for name,(artifact,sha) in LIVE.items():
        info = api(f'artifacts/{artifact}')
        related = info['workflow_run']
        assert (info['id'],info['name'],info['digest'],info['expired']) == (artifact,name,sha,False)
        assert (related['id'],related['head_sha']) == (RUN,CARRIER)
        assert datetime.now(timezone.utc) < datetime.fromisoformat(info['expires_at'])
        observed[name] = dict(name=name,id=artifact,digest=sha,expired=False,expires_at=info['expires_at'],
                              run=RUN,job=JOB,carrier=CARRIER,conclusion='success',size=info['size_in_bytes'])
        assert observed[name] == approval['artifacts'][name]
    transport = observed['current308-windows-timing-transport-v1']
    assert transport['size'] == TRANSPORT_BYTES and TRANSPORT_BYTES <= 128 * 2**20
    uri = f"https://api.github.com/repos/{REPO}/actions/artifacts/{transport['id']}/zip"
    try:
        request.build_opener(NoRedirect()).open(request.Request(uri,headers=headers),timeout=20).close()
    except error.HTTPError as failure:
        assert failure.code in (301,302,303,307,308)
        url = failure.headers['Location']
    else:
        raise AssertionError('expected the selected GitHub artifact redirect')
    assert parse.urlsplit(url).scheme == 'https'
    with request.urlopen(url,timeout=120) as reply:
        raw = reply.read(TRANSPORT_BYTES+1)
    assert len(raw) == TRANSPORT_BYTES and 'sha256:'+sha256(raw).hexdigest() == transport['digest']
    wheeldir, receipts = (scratch/name for name in ('transport-wheels','transport-receipts'))
    expected = terminal['transport_sha256']|{name:approval['files'][name] for name in (
        'measurement.json','terminal.json')}
    assert len(expected) == 17
    with ZipFile(BytesIO(raw)) as z:
        entries = z.infolist()
        assert len(entries) == 17 and sum(item.file_size for item in entries) <= 128*2**20
        assert {PurePosixPath(item.filename).name for item in entries} == set(expected)
        wheeldir.mkdir()
        receipts.mkdir()
        for item in entries:
            member = PurePosixPath(item.filename)
            assert member.as_posix() == item.filename and not member.is_absolute() and '..' not in member.parts
            assert '\\' not in item.filename and ':' not in item.filename
            assert not item.is_dir() and not stat.S_ISLNK(item.external_attr>>16)
            name = member.name
            dest = (wheeldir if name.endswith('.whl') else receipts)/name
            with z.open(item) as src,dest.open('xb') as stream:
                shutil.copyfileobj(src,stream,2**20)
            assert dest.stat().st_size == item.file_size and digest(dest) == expected[name]
    write(evidence/'transport.json',dict(artifacts=observed,verified_zip_bytes=len(raw),
          outer_sha256=transport['digest'],members_sha256=expected,
          archive_member_names=[item.filename for item in entries],archive_retained=False,
          compact_transfer='Reused selected root-admitted compact files; no second download'))
    return wheeldir,receipts


def subprocess_exit(command, evidence, label, env):
    with (evidence/(label+'.stdout')).open('xb') as out,(evidence/(label+'.stderr')).open('xb') as errors:
        status = subprocess.run([str(item) for item in command],stdout=out,stderr=errors,env=env,check=False).returncode
    with (evidence/(label+'.exit')).open('x',encoding='ascii',newline='\n') as result:
        result.write(f'{status}\n')
    return status


def quiet(fixed, env):
    # A fresh Windows Actions job must have neither competing Windows work
    # nor an active WSL worker/VM. Do not start WSL to check.
    command = r"""
    $ErrorActionPreference='Stop'
    Get-CimInstance Win32_Process | Where-Object {
      $_.Name -match '^(python[\d.]*w?|pip[\d.]*|cl|link|cmake|ninja|ctest|clang.*|gcc|g\+\+|cc1plus|msbuild|java|gradle|rustc|gh|git|curl|wget|aria2c|robocopy|qupath.*|wsl|wslhost|wslg|wslrelay|vmmem|vmmemWSL)(\.exe)?$' -and
      $_.ProcessId -ne [int]$env:FIXED8_CONTROLLER_PID
    } | Select-Object ProcessId,Name | ConvertTo-Json -AsArray -Compress
    """
    shell = Path(shutil.which('pwsh.exe')).resolve(strict=True)
    env = env | {'FIXED8_CONTROLLER_PID':str(os.getpid())}
    proc = subprocess.run([shell,'-NoLogo','-NoProfile','-NonInteractive','-Command',command],
                          env=env,capture_output=True,check=True)
    rows = json.loads(proc.stdout) if proc.stdout.strip() else []
    api,handle,process,system = fixed['affinity']()
    import ctypes
    assert api.GetProcessAffinityMask(handle,ctypes.byref(process),ctypes.byref(system))
    assert process.value & 1 and system.value & 1
    wsl_count = sum(row['Name'].lower().removesuffix('.exe').startswith(('wsl','vmmem')) for row in rows)
    return dict(utc=datetime.now(timezone.utc).isoformat(),host=os.environ['COMPUTERNAME'],
                windows_competitors=len(rows)-wsl_count,wsl_competitors=wsl_count,
                blocking_processes=rows,process_mask=process.value,system_mask=system.value,
                logical_processors=os.cpu_count())


def run(args, approval, deadline):
    args.evidence = args.evidence.resolve(strict=True)
    args.source = args.source.resolve(strict=True)
    runner = args.runner.resolve(strict=True)
    for name,path in {'prepare':args.prepare,'runner':runner/'run.py','plan':runner/'plan-v1.json',
                      'stock_template':args.stock_template,'threadpoolctl':args.threadpoolctl}.items():
        assert digest(path) == PINS[name], name
    fixed = runpy.run_path(str(runner/'run.py'))
    plan = read(runner/'plan-v1.json')
    assert plan['protocol']['accepted_total'] == 112
    assert fixed['source_entry'](dict(git=ref(Path(shutil.which('git.exe')).resolve(strict=True)),
                                      source_root=str(args.source)),plan)==args.source
    for name,pin in approval['files'].items():
        assert Path(name).name == name and digest(args.focus/name) == pin
    assert approval['files'].keys() == runpy.run_path(str(args.prepare))['RECEIPTS']
    terminal = read(args.focus/'terminal.json')
    assert terminal['source'] == SOURCE and terminal['tree'] == TREE
    assert terminal['status'] == 'passed_focused_windows' and terminal['sdk_checks'] == 4
    assert terminal['product'] == approval['product']
    temp = Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
    assert args.evidence.is_relative_to(Path(os.environ['GITHUB_WORKSPACE']).resolve(strict=True))
    assert all(shutil.disk_usage(path).free >= FLOOR for path in (temp,args.source,args.evidence))
    scratch = Path(tempfile.mkdtemp(prefix='current308-timing-',dir=temp)).resolve(strict=True)
    env = {key:value for key,value in os.environ.items()
           if not key.startswith('SKIMAGEXX_') and key not in {
               'PYTHONPATH','PYTHONHOME','NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE','MKL_DEBUG_CPU_TYPE',
               'LD_PRELOAD','GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GH_TOKEN'}}
    env.update({name:'1' for name in fixed['LIMITS']})
    try:
        wheels,receipts = receive(approval,scratch,args.evidence,terminal)
        granted = dict(scope='current308-windows-timing-host-preparation-v1',authorized_by='/root',
                       source=SOURCE,tree=TREE,producer_carrier=CARRIER,producer_run=RUN,producer_job=JOB,
                       timing_carrier=approval['timing_carrier'],launcher_sha256=PINS['prepare'],
                       timing_authorized=False,deadline_utc=deadline.isoformat(),artifacts=approval['artifacts'],
                       stock_template_sha256=PINS['stock_template'],files=approval['files'])
        prep_grant = args.evidence/'preparation-authority.json'
        write(prep_grant,granted)
        assert datetime.now(timezone.utc) < deadline
        status = subprocess_exit([sys.executable,'-I','-S','-B',args.prepare,
            '--runner',runner,'--stock-template',args.stock_template,'--source',args.source,'--focus',args.focus,
            '--transport-receipts',receipts,'--wheels',wheels,'--scratch',scratch,'--evidence',args.evidence,
            '--threadpoolctl',args.threadpoolctl,'--approval',prep_grant,'--approval-sha256',digest(prep_grant)],
            args.evidence,'prepare',env)
        if status:
            return status
        prepared = read(args.evidence/'timing-prepared.json')
        binding_file = args.evidence/'timing-binding.json'
        assert prepared['binding'] == ref(binding_file) and prepared['host'] == os.environ['COMPUTERNAME']
        assert (prepared['source'],prepared['tree'],prepared['timing_authorized']) == (SOURCE,TREE,False)
        assert prepared['entry'] == ref(args.evidence/'fixed8-entry.log')
        assert prepared['offline_verify'] == ref(args.evidence/'offline-verify.log')
        entry = read(args.evidence/'fixed8-entry.log')
        verified = read(args.evidence/'offline-verify.log')
        binding = read(binding_file)
        assert verified['status'] == 'verify' and verified['revision'] == SOURCE and verified['wheel_count'] == 15
        assert (entry['source'],entry['tree'],entry['runtime'],entry['scientific_imports']) == (SOURCE,TREE,'3.13.16',0)
        assert entry['core_sha256'] == prepared['candidate_core']['sha256'] == approval['product']['core']
        assert binding['candidate']['record']['sha256'] == approval['product']['record']
        assert binding['candidate']['wheel_sha256'] == approval['product']['wheel']
        assert binding['python'] == ref(Path(sys.executable).resolve(strict=True))
        assert fixed['source_entry'](binding,plan) == args.source
        assert all(shutil.disk_usage(path).free >= FLOOR for path in (temp,args.source,args.evidence))
        result_dir = args.evidence/'fixed8'
        go_file = args.evidence/'root-timing-go.json'
        assert not result_dir.exists() and not go_file.exists()
        state = quiet(fixed,env)
        quiet_file = args.evidence/'quiet.json'
        write(quiet_file,state)
        assert state['windows_competitors'] == state['wsl_competitors'] == 0
        current = datetime.now(timezone.utc)
        assert current < deadline
        go = dict(authorized_by='/root',scope=fixed['SCOPE'],source=SOURCE,tree=TREE,exclusive=True,cpu=0,workers=1,
                  deadline_utc=min(deadline,current+timedelta(minutes=5)).isoformat(),
                  launcher=ref(runner/'run.py'),plan=ref(runner/'plan-v1.json'),binding=ref(binding_file),
                  quiet=ref(quiet_file),output=str(result_dir))
        # Authority comes only from the exact conditional job input validated
        # above, after verification and fresh live quiescence on this host.
        write(go_file,go)
        return subprocess_exit([sys.executable,'-I','-S','-B',runner/'run.py','--go',go_file,
                                '--go-sha256',digest(go_file)],args.evidence,'timing',env)
    finally:
        target = scratch.resolve(strict=True)
        assert target.parent == temp and target.name.startswith('current308-timing-')
        shutil.rmtree(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    for name in ('approval','product-admission','compact-readback','source','focus','prepare','runner',
                 'stock-template','threadpoolctl','evidence'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--approval-sha256')
    args = parser.parse_args()
    if args.preview:
        print(json.dumps(dict(source=SOURCE,tree=TREE,scope=SCOPE,producer_run=RUN,producer_job=JOB,
                              authorities_created=0,transfers=0,scientific_imports=0)))
        return 0
    if not (args.approval and args.approval_sha256):
        print('REFUSED_NO_ROOT_CONDITIONAL_TIMING_AUTHORITY',file=sys.stderr)
        return 2
    approval,deadline = authorize(args)
    return run(args,approval,deadline)


if __name__ == '__main__':
    raise SystemExit(main())
