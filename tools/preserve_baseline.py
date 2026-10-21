"""Preserve a wheel built from a clean checkout for paired C++ measurements.

Build the wheel in the source checkout first, then supply that checkout and
wheel here. Python sources are checked against the wheel; the manifest records
the source revision/fingerprint and the exact extension binary hashes.
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
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.source_root.resolve();output=args.output.resolve()
    def git(*arguments):return subprocess.check_output(['git','-C',str(root),*arguments],text=True,encoding='utf-8').strip()
    if git('status','--porcelain'):parser.error('baseline source checkout must be clean')
    package=output/'_baseline'
    if package.exists() or (output/'baseline-source.json').exists():parser.error('choose a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('skimagexx/') or name.endswith('/'):continue
            relative=PurePosixPath(name).relative_to('skimagexx')
            destination=(package/Path(*relative.parts)).resolve()
            if not destination.is_relative_to(package):raise ValueError('wheel member escapes baseline package')
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(name))
    for source in (root/'python/skimagexx').rglob('*.py'):
        preserved=package/source.relative_to(root/'python/skimagexx')
        if source.read_bytes().replace(b'\r\n',b'\n')!=preserved.read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError(f'wheel Python source differs from checkout: {source.name}')
    binaries=[*package.glob('*.pyd'),*package.glob('*.so')]
    if not binaries:raise ValueError('wheel has no extension binary')
    metadata={'revision':git('rev-parse','HEAD'),**source_fingerprint(root),
              'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
              'prepared_at_utc':datetime.now(timezone.utc).isoformat()}
    (output/'baseline-source.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Preserved {metadata["revision"][:7]} with checked Python sources and {len(binaries)} binary hash(es).')


if __name__=='__main__':main()
