"""Read only the pinned failed qualification's selected wheel members."""

import argparse
from base64 import urlsafe_b64encode
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import sys
from urllib import error, parse, request
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
REPO = 'bipxx/scikit-imagexx'
PINS_SHA = '72e8ffabc97cdfbd450865af7f624fd3681787efddc78fce304dea123b2b7b6d'
HELPER_SHA = '1a54751980e8d5c06e31c46891f5a6112b5a6f64a07f494497fe93b98e912475'


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    assert digest(raw) == expected, path
    return raw


def record(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def verify_inner(raw, package, output):
    hashes = package['members_sha256']
    with ZipFile(BytesIO(raw)) as z:
        infos = z.infolist()
        names = [entry.filename for entry in infos]
        assert len(names) == len(set(names)) == len(hashes) == 272 and set(names) == set(hashes)
        name, = [n for n in names if n.endswith('.dist-info/RECORD')]
        payload = z.read(name)
        assert digest(payload) == hashes[name]
        rows = list(csv.reader(StringIO(payload.decode('utf-8'), newline='')))
        assert len(rows) == len(names)
        metadata = {row[0]: row[1:] for row in rows}
        assert set(metadata) == set(names) and metadata[name] == ['', '']
        output.mkdir()
        size = 0
        for member in infos:
            label = member.filename
            relative = PurePosixPath(label)
            assert not relative.is_absolute() and relative.parts and '..' not in relative.parts
            assert '\\' not in label and ':' not in label and not member.is_dir()
            assert not stat.S_ISLNK(member.external_attr >> 16)
            assert size + member.file_size <= 96 * 2**20
            content = payload if label == name else z.read(member)
            assert digest(content) == hashes[label] and len(content) == member.file_size, label
            if label != name:
                code = urlsafe_b64encode(bytes.fromhex(hashes[label])).decode().rstrip('=')
                assert metadata[label] == ['sha256=' + code, str(len(content))], label
            dest = output.joinpath(*relative.parts)
            dest.parent.mkdir(exist_ok=True, parents=True)
            with dest.open('xb') as file:
                file.write(content)
            size += len(content)
    return {'members': len(names), 'record_path': name, 'record_sha256': hashes[name], 'bytes': size}


def download(pins, helper, destination):
    assert os.environ['GITHUB_ACTIONS'] == 'true' and os.environ['GITHUB_REPOSITORY'] == REPO
    token = os.environ['GH_TOKEN']
    assert token
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28'}

    def api(suffix):
        req = request.Request(f'https://api.github.com/repos/{REPO}/actions/{suffix}', headers=headers)
        with request.urlopen(req, timeout=15) as res:
            return json.load(res)

    uri = f'artifacts/{pins["artifact"]}'
    info = api(uri)
    origin = info['workflow_run']
    assert (info['id'], info['name'], info['expired'], info['digest']) == (
        pins['artifact'], 'public-qualification-minimum', False, pins['outer_digest'])
    assert (origin['id'], origin['head_sha'], origin['head_branch']) == (pins['run'], pins['source'], pins['branch'])
    job = api(f'jobs/{pins["job"]}')
    assert (job['run_id'], job['head_sha'], job['status'], job['conclusion']) == (
        pins['run'], pins['source'], 'completed', 'failure')
    download = request.Request(f'https://api.github.com/repos/{REPO}/actions/{uri}/zip', headers=headers)
    try:
        request.build_opener(helper['NoRedirect']).open(download, timeout=15).close()
    except error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f'Artifact redirect HTTP {exc.code}') from None
        url = exc.headers.get('Location', '')
    else:
        raise RuntimeError('Expected signed blob redirect')
    assert parse.urlparse(url).scheme == 'https'
    transport = helper['RemoteZip'](url, info['size_in_bytes'])
    with transport, ZipFile(transport) as archive:
        entries = archive.namelist()
        for label, expected in (pins['pinned_outer'] | {pins['candidate_wheel_member']: pins['candidate_wheel']}).items():
            member = pins['namespace'] + label
            assert entries.count(member) == 1, label
            assert archive.getinfo(member).file_size == expected['bytes'], label

        def selected(name, expected):
            raw = archive.read(pins['namespace'] + name)
            assert record(raw) == expected, name
            return raw

        manifest, package = (json.loads(selected(name, pins['pinned_outer'][name]))
                             for name in ('manifest.json', 'package.json'))
        assert manifest['source']['revision'] == package['source_revision'] == pins['source']
        assert manifest['profile'] == 'minimum' and not manifest['passed'] and package['record_verified']
        assert manifest['records']['package.json'] == pins['pinned_outer']['package.json']
        assert manifest['records'][pins['candidate_wheel_member']] == pins['candidate_wheel']
        assert package['wheel_sha256'] == pins['candidate_wheel']['sha256']
        assert all(manifest['source']['files_sha256'][name] == pin['normalized_lf_sha256']
                   for name, pin in pins['source_paths'].items())
        raw = selected(pins['candidate_wheel_member'], pins['candidate_wheel'])
    member = package['sole_core']
    assert member == pins['candidate_core_member']
    assert package['members_sha256'][member] == pins['candidate_core_sha256']
    proof = verify_inner(raw, package, destination)
    return {**proof, 'artifact': info['id'], 'source': pins['source'], 'job': job['id'],
            'wheel_sha256': pins['candidate_wheel']['sha256'], 'core_member': member,
            'core_sha256': pins['candidate_core_sha256'], 'range_bytes': transport.bytes_read,
            'outer_zip_fully_verified': False, 'archives_retained': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--entry-only', action='store_true')
    args = parser.parse_args()
    pins = json.loads(pinned(HERE / 'pins.json', PINS_SHA))
    helper_path = HERE / 'range_helper.py'
    pinned(helper_path, HELPER_SHA)
    helper = runpy.run_path(str(helper_path))
    if args.entry_only:
        assert args.destination is None and args.proof is None
        print(json.dumps({'source': pins['source'], 'artifact': pins['artifact'], 'helper_sha256': HELPER_SHA}))
        raise SystemExit(0)
    assert sys.version_info[:3] == (3, 11, 17) and sys.flags.isolated and sys.dont_write_bytecode
    runpy.run_path(str(HERE / 'validate.py'))['authorize'](pins)
    assert args.destination and args.proof and not args.destination.exists() and not args.proof.exists()
    proof = download(pins, helper, args.destination)
    with args.proof.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')
