import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

REPO = 'repos/bipxx/scikit-imagexx'
ROOT = Path.cwd()
PROOF = ROOT / 'proof'
PROOF.mkdir(exist_ok=False)
DATA = {
    'stock': dict(artifact=14385543897,run=44207792316,source='e27633bf9186bb7cab5c1d0b9fbf1584c94c6b81',
                  archive='6d66a2f094487c8630875b053eb63b72e3a1b81a222d86235246509c2e81f439'),
    'previous': dict(artifact=14396114399,run=44235086774,source='fc7d10f17382bc56c8b33c2fe5c418fa58f69921',
                    archive='9772c118ed563052665f32de272845257d3a22ba272d77064918585ded29e71e',
                    core='d5df5b90ecb3603cdfce9a721663e23d76e06cf0896f9315ea2386fdf3964250',
                    receipt='284a5c5982fe39cab9cefc2a4eb5ef1c1f198c2d832a7202e575ab4924fd5829'),
    'wide': dict(artifact=14399698237,run=44241778741,source='eb1f37e441b7d1262cb43f2205f39d3fec6c598c',
                archive='b8c60af05929952ebabd38a27bc37fd02c3a0a3146ae8515d2c8d80de62daeb4',
                core='17fc01c7a8d1753feebcf30699c26d85dfb15daf6bfea53d341cc495a785b2e3',
                receipt='7169208dc2dd6c9718b83aa4f5e9e82b95e448f079d67598a38b1a1a93994c2b'),
}
STOCK = json.loads((ROOT / 'diagnostic-f64-stock-pins.json').read_bytes())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def seal(path, obj):
    with path.open('x', encoding='utf-8') as out:
        json.dump(obj,out,indent=2)
        out.write(chr(10))

def request(part):
    return subprocess.check_output(['gh','api',REPO+'/'+part])

def artifact(meta):
    ident=meta['artifact']
    m=json.loads(request(f'actions/artifacts/{ident}'))
    assert m['id']==ident and m['workflow_run']['id']==meta['run'] and not m['expired']
    if 'core' in meta:
        assert m['workflow_run']['head_sha']=='517ae1fe54007f080fe269261f6c5c315cebb6d8'
    else:
        assert m['workflow_run']['head_sha']==meta['source']
    assert m['digest']=='sha256:'+meta['archive'] and 0<m['size_in_bytes']<128*1024**2
    body=request(f'actions/artifacts/{ident}/zip')
    assert len(body)==m['size_in_bytes'] and sha(body)==meta['archive']
    seal(PROOF/f'transfer-{ident}.json',dict(meta,archive_bytes=len(body),transient_outer_in_memory=True))
    z=zipfile.ZipFile(io.BytesIO(body))
    entries=z.infolist()
    assert len(entries)==len(set(x.filename for x in entries))
    for e in entries:
        q=PurePosixPath(e.filename)
        assert not q.is_absolute() and '..' not in q.parts and chr(92) not in e.filename
        assert not stat.S_ISLNK(e.external_attr>>16)
    return z

def ready_stock(scratch):
    wheel_root=scratch/'wheels'
    wheel_root.mkdir()
    with artifact(DATA['stock']) as z:
        for name,want in STOCK.items():
            n,=[t for t in z.namelist() if PurePosixPath(t).name==name]
            assert PurePosixPath(n).parent.name=='wheels'
            raw=z.read(n)
            assert sha(raw)==want
            (wheel_root/name).write_bytes(raw)
    site=scratch/'stock'
    argv=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile',
          '--target',str(site),*map(str,sorted(wheel_root.glob('*.whl')))]
    result=subprocess.run(argv,capture_output=True)
    (PROOF/'install.stdout').write_bytes(result.stdout)
    (PROOF/'install.stderr').write_bytes(result.stderr)
    seal(PROOF/'install.json',dict(argv=argv,exit=result.returncode))
    assert result.returncode==0
    for name in STOCK:
        (wheel_root/name).unlink()
    wheel_root.rmdir()
    return site

def ready_product(role,scratch):
    meta=DATA[role]
    target=scratch/role
    target.mkdir()
    with artifact(meta) as z:
        r=z.read('receipt.json')
        assert sha(r)==meta['receipt']
        claim=json.loads(r)
        assert claim['product_source']==meta['source'] and claim['core_sha256']==meta['core']
        assert claim['stock_dependencies_sha256']==STOCK and claim['timing_batches']==0
        assert all(claim['comparison'][x]['actual_exit']==0 for x in ('normal','disabled'))
        total=0
        for entry in z.infolist():
            if entry.is_dir() or not entry.filename.startswith('package/skimagexx/'):
                continue
            relative=PurePosixPath(entry.filename).relative_to('package')
            path=target.joinpath(*relative.parts)
            assert path.resolve().is_relative_to(target.resolve())
            total+=entry.file_size
            assert total<30*1024**2
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as out:
                out.write(z.read(entry))
    (scratch/(role+'-installed-receipt.json')).write_bytes(r)
    core,=(target/'skimagexx').glob('_core*.so')
    assert sha(core.read_bytes())==meta['core']
    return target


assert shutil.disk_usage(ROOT).free>=30*1024**3 and shutil.disk_usage(tempfile.gettempdir()).free>=30*1024**3
with tempfile.TemporaryDirectory(prefix='skimagexx-f64-abba-') as tmp:
    p=Path(tmp).resolve()
    stock=ready_stock(p)
    previous=ready_product('previous',p)
    wide=ready_product('wide',p)
    sr=p/'stock-receipt.json'
    seal(sr,dict(stock_root=str(stock),wheel_sha256=STOCK))
    cpu=min(os.sched_getaffinity(0))
    command=[sys.executable,'-B','-I',str(ROOT/'diagnostic-f64-same-host.py'),
             '--baseline-package',str(previous),'--wide-package',str(wide),
             '--baseline-receipt',str(p/'previous-installed-receipt.json'),
             '--wide-receipt',str(p/'wide-installed-receipt.json'),
             '--stock',str(stock),'--stock-receipt',str(sr),
             '--timer',str(ROOT/'benchmarks/run.py'),
             '--fixture',str(ROOT/'diagnostic-f64-coverage-le.bin'),
             '--cpu',str(cpu),'--out',str(PROOF/'measurement')]
    env={k:v for k,v in os.environ.items() if k not in ('GH_TOKEN','GITHUB_TOKEN')}
    run=subprocess.run(command,env=env)
    seal(PROOF/'measurement-command.json',dict(argv=command,actual_exit=run.returncode,
         upstream_calls=0,builds=0,tests=0,archive_files_saved=0))
    sys.exit(run.returncode)
