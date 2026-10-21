"""Portable source fingerprints for the current benchmark report set."""
import hashlib
from pathlib import Path

SCHEMA='cpp-python-build-posix-lf-v2'


def source_fingerprint(root):
    root=Path(root)
    files=[root/'CMakeLists.txt',root/'pyproject.toml']
    for path in files:
        if not path.is_file():raise FileNotFoundError(path)
    files+=list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))
    files+=list((root/'python/skimagexx').glob('*.py'))
    digest=hashlib.sha256()
    for path in sorted((p for p in files if p.is_file()),key=lambda p:p.relative_to(root).as_posix()):
        name=path.relative_to(root).as_posix().encode('utf-8')
        data=path.read_bytes().replace(b'\r\n',b'\n')
        for value in (name,data):
            digest.update(len(value).to_bytes(8,'little'));digest.update(value)
    return {'source_sha256':digest.hexdigest(),'source_hash_schema':SCHEMA,
            'source_hash_scope':'CMakeLists.txt, pyproject.toml, src/*, include/skimagexx/*, python/skimagexx/*.py'}
