"""Preserve a checked package for paired C++ measurements.

Build the wheel in the source checkout first, then supply that checkout and
wheel here. Python sources are checked against the wheel; the manifest records
the source revision/fingerprint and the exact extension binary hashes. To keep
an already measured installed package, use --installed-package and --measurement;
their product source fingerprint and measured binary hashes must agree.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--wheel',type=Path)
    mode.add_argument('--installed-package',type=Path)
    parser.add_argument('--measurement',type=Path,help='Benchmark JSON identifying the exact measured source and binaries')
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.source_root.resolve();output=args.output.resolve()
    def git(*arguments):return subprocess.check_output(['git','-C',str(root),*arguments],text=True,encoding='utf-8').strip()
    if args.installed_package and not args.measurement:
        parser.error('--installed-package requires --measurement')
    scope=('CMakeLists.txt','pyproject.toml','src','include/skimagexx','python/skimagexx')
    dirty=git('status','--porcelain','--',*scope) if args.installed_package else git('status','--porcelain')
    if dirty:parser.error('baseline product source must be clean' if args.installed_package else 'baseline source checkout must be clean')
    fingerprint=source_fingerprint(root)
    package=output/'_baseline'
    if any(path.exists() for path in (package,output/'baseline-source.json',output/'initial-measurement.json')):
        parser.error('choose a fresh output directory')
    members={}
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as archive:
            for name in archive.namelist():
                if not name.startswith('skimagexx/') or name.endswith('/'):continue
                relative=PurePosixPath(name).relative_to('skimagexx')
                members[Path(*relative.parts)]=archive.read(name)
    else:
        installed=args.installed_package.resolve()
        if not installed.is_dir():parser.error('installed package directory does not exist')
        for path in installed.rglob('*'):
            relative=path.relative_to(installed)
            if not path.is_file() or '__pycache__' in relative.parts or path.suffix=='.pyc':continue
            if not path.resolve().is_relative_to(installed):raise ValueError('installed member escapes package')
            members[relative]=path.read_bytes()
    for relative in members:
        if not (package/relative).resolve().is_relative_to(package):
            raise ValueError('member escapes baseline package')
    sources=list((root/'python/skimagexx').rglob('*.py'))
    expected_python={path.relative_to(root/'python/skimagexx') for path in sources}
    if {path for path in members if path.suffix=='.py'}!=expected_python:
        raise ValueError('package Python file set differs from checkout')
    for source in sources:
        relative=source.relative_to(root/'python/skimagexx')
        if relative not in members or source.read_bytes().replace(b'\r\n',b'\n')!=members[relative].replace(b'\r\n',b'\n'):
            raise ValueError(f'package Python source differs from checkout: {source.name}')
    binaries={path.name:hashlib.sha256(data).hexdigest() for path,data in members.items()
              if len(path.parts)==1 and path.suffix in ('.pyd','.so')}
    if not binaries:raise ValueError('package has no extension binary')
    measurement_data=None
    if args.measurement:
        measurement_data=args.measurement.read_bytes()
        measured=json.loads(measurement_data)['metadata']
        for key,value in fingerprint.items():
            if measured.get(key)!=value:raise ValueError(f'measured {key} differs from checkout')
        if measured.get('current_binary_sha256')!=binaries:
            raise ValueError('installed/wheel binaries differ from measured binaries')
    if source_fingerprint(root)!=fingerprint:raise ValueError('source changed during preservation')
    metadata={'revision':git('rev-parse','HEAD'),**fingerprint,
              'binary_sha256':binaries,
              'prepared_at_utc':datetime.now(timezone.utc).isoformat()}
    if measurement_data is not None:
        metadata.update(snapshot_kind='Exact measured package; product fingerprint, Python sources and binary hashes checked',
                        measurement_sha256=hashlib.sha256(measurement_data).hexdigest(),
                        measurement_report='initial-measurement.json')
    output.mkdir(parents=True,exist_ok=True)
    for relative,data in members.items():
        destination=package/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
    if measurement_data is not None:(output/'initial-measurement.json').write_bytes(measurement_data)
    (output/'baseline-source.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Preserved {metadata["revision"][:7]} with checked Python sources and {len(binaries)} binary hash(es).')


if __name__=='__main__':main()
