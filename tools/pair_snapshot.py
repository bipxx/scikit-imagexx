"""Validate an archived package and record hashes for paired benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,required=True)
    args=parser.parse_args()
    root=args.package_root
    source=json.loads((root/'baseline-source.json').read_text())
    package=root/'_baseline'
    binaries={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_core.*')}
    if binaries!=source['binary_sha256']:raise ValueError('archived extension hashes differ')
    python={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')}
    if not python:raise ValueError('archived Python package missing')
    target=root/'snapshot.json'
    snapshot={**source,'python_sha256':python}
    if target.exists() and json.loads(target.read_text())!=snapshot:raise ValueError('existing snapshot differs')
    target.write_text(json.dumps(snapshot,indent=2)+'\n')
    print('Verified archived binary hashes; recorded Python package hashes.')


if __name__=='__main__':main()
