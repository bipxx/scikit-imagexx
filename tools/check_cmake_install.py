"""Install and relocate a C++ build, then compile an independent consumer."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--config',default='Release')
    parser.add_argument('--cmake',default='cmake')
    parser.add_argument('--openmp',choices=['ON','OFF'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    build=args.build_dir.resolve();work=args.work_dir.resolve()
    if work.exists():parser.error('choose a fresh work directory to preserve previous results')
    if not (build/'CMakeCache.txt').is_file():parser.error('build directory has no CMake cache')
    if not work.is_relative_to((root/'build').resolve()):parser.error('work directory must be inside repository build/')
    home=[line.split('=',1)[1] for line in (build/'CMakeCache.txt').read_text(encoding='utf-8').splitlines()
          if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL=')]
    if len(home)!=1 or Path(home[0]).resolve()!=root:parser.error('build directory belongs to another source checkout')
    work.mkdir(parents=True)
    commands=[]
    def run(name,arguments,expect_success=True):
        command=[args.cmake,*map(str,arguments)];log=work/(name+'.log')
        with log.open('w',encoding='utf-8') as output:
            result=subprocess.run(command,cwd=root,stdout=output,stderr=subprocess.STDOUT)
        commands.append({'command':command,'exit_code':result.returncode,'log':log.name})
        (work/'commands.json').write_text(json.dumps(commands,indent=2)+'\n',encoding='utf-8')
        if expect_success and result.returncode:
            raise RuntimeError(f'{name} failed; inspect {log}')
        if not expect_success and not result.returncode:
            raise RuntimeError('incompatible package version unexpectedly accepted')
    prefix=work/'original-prefix';relocated=work/'relocated prefix'
    run('install',['--install',build,'--config',args.config,'--prefix',prefix])
    shutil.copytree(prefix,relocated)
    common=['-DCMAKE_BUILD_TYPE='+args.config,'-DCMAKE_PREFIX_PATH='+str(relocated),
        '-DSKIMAGEXX_EXPECT_PREFIX='+str(relocated),'-DSKIMAGEXX_EXPECT_OPENMP='+args.openmp,
        '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF','-DCMAKE_DISABLE_FIND_PACKAGE_Python=ON',
        '-DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON']
    if args.openmp=='OFF':common+=['-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON']
    consumer=work/'consumer-build'
    run('configure',['-S',root/'tests/cmake_consumer','-B',consumer,*common])
    run('build',['--build',consumer,'--config',args.config])
    ctest=Path(args.cmake).with_name('ctest'+Path(args.cmake).suffix) if Path(args.cmake).is_absolute() else 'ctest'
    with (work/'test.log').open('w',encoding='utf-8') as output:
        subprocess.run([str(ctest),'--test-dir',str(consumer),'-C',args.config,'--output-on-failure'],
            cwd=root,stdout=output,stderr=subprocess.STDOUT,check=True)
    run('version-rejection',['-S',root/'tests/cmake_consumer','-B',work/'wrong-version',*common,
        '-DSKIMAGEXX_FIND_VERSION=0.0'],expect_success=False)
    print(f'Installed, relocated C++ package passed ({args.openmp=}); logs: {work}')


if __name__=='__main__':main()
