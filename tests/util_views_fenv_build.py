"""Build one test-only source97ff helper and seal actual emitted controls."""
import argparse
import ctypes
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from util_views_fenv import ABI,CONTEXT,ROOT,SOURCES,SOURCE_SHA256,sha,validate_controls


def dump(path,value):
    with path.open('x',encoding='utf-8',newline='\n')as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--control-python',action='append');args=parser.parse_args()
    assert sys.platform=='linux'and platform.machine()=='x86_64'and sys.byteorder=='little','Only Linux x86_64/glibc32 is admitted; Windows must refuse'
    assert sha(ROOT/SOURCES[0])==SOURCE_SHA256
    context=json.loads(CONTEXT.read_bytes())
    for name,pin in context['source_pins_raw_sha256'].items():assert sha(ROOT/name)==pin,name
    out=args.output.resolve();assert not out.exists(),'FENV output exists; never overwrite/retry unchanged build'
    compiler=Path(shutil.which('gcc')).resolve();backend=Path(subprocess.check_output([str(compiler),'-print-prog-name=cc1'],text=True).strip()).resolve()
    toolchain=dict(path=str(compiler),sha256=sha(compiler),backend=str(backend),backend_sha256=sha(backend),version=subprocess.check_output([str(compiler),'--version'],text=True),target=subprocess.check_output([str(compiler),'-dumpmachine'],text=True).strip())
    assert toolchain['target'].startswith('x86_64')
    out.mkdir(parents=True);(out/'headers').mkdir();(out/'source').mkdir()
    pins={name:sha(ROOT/name)for name in SOURCES}
    for name in SOURCES:shutil.copy2(ROOT/name,out/'source'/Path(name).name)
    shutil.copy2(CONTEXT,out/'source'/CONTEXT.name)
    dependencies=subprocess.check_output([str(compiler),'-M',str(ROOT/SOURCES[0])],text=True)
    (out/'compiler-dependencies.txt').write_text(dependencies,encoding='utf-8',newline='\n')
    headers=[]
    for p in shlex.split(dependencies.split(':',1)[1].replace('\\\n',' ')):
        path=Path(p)
        if path.name=='fenv.h':
            captured='headers/'+str(len(headers))+'-fenv.h';shutil.copy2(path,out/captured);headers.append(dict(original=str(path.resolve()),captured=captured))
    assert any(h['original'].endswith('/bits/fenv.h')for h in headers)
    library=out/'libutil_views_fenv_restore.so'
    argv=[str(compiler),'-std=c11','-O2','-fPIC','-shared','-fvisibility=hidden','-fno-lto','-fno-fast-math','-Wall','-Wextra','-Werror','-o',str(library),str(ROOT/SOURCES[0])]
    command=dict(argv=argv,compiler=toolchain,headers=headers,source_pins=pins)
    dump(out/'build-command.json',command)
    result=subprocess.run(argv,capture_output=True);(out/'compile.stdout').write_bytes(result.stdout);(out/'compile.stderr').write_bytes(result.stderr)
    dump(out/'compile-terminal.json',dict(exit_code=result.returncode))
    assert result.returncode==0,'FENV helper compile failed; outputs retained'
    library_hash=sha(library);controls=[]
    for i,python in enumerate(args.control_python or[sys.executable]):
        name='controls-'+str(i)+'.json';argv=[python,str(ROOT/'tests/util_views_fenv_probe.py'),'--library',str(library),'--library-sha256',library_hash,'--output',str(out/name)]
        dump(out/('probe-'+str(i)+'-command.json'),dict(argv=argv))
        result=subprocess.run(argv,capture_output=True);(out/('probe-'+str(i)+'.stdout')).write_bytes(result.stdout);(out/('probe-'+str(i)+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and(out/name).is_file(),'FENV actual control process failed; no ready receipt'
        validate_controls(json.loads((out/name).read_bytes()),library_hash);controls.append(name)
    libc=ctypes.CDLL(None);libc.gnu_get_libc_version.restype=ctypes.c_char_p
    record=dict(schema='util-views-fenv-build-v1',status='PASS_SOURCE_BUILT_FENV32',ABI=ABI,glibc=libc.gnu_get_libc_version().decode(),source_context_sha256=sha(CONTEXT),source_pins=pins,compiler=toolchain,headers=headers,controls=controls,files={p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    dump(out/'receipt.json',record)
    print(json.dumps(dict(status=record['status'],receipt=str(out/'receipt.json'),receipt_sha256=sha(out/'receipt.json'),library_sha256=library_hash,collector_directory=str(out))))
    return 0


if __name__=='__main__':raise SystemExit(main())
