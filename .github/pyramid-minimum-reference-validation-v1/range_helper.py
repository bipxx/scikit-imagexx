"""Read selected terminal failure records without materializing the hosted ZIP."""
import argparse
import base64
import csv
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteZip(io.RawIOBase):
    def __init__(self, url, length):
        super().__init__()
        self.url, self.length, self.position = url, length, 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.length}[whence]
        self.position = base + offset
        if self.position < 0:
            raise OSError('negative remote ZIP offset')
        return self.position

    def read(self, size=-1):
        remaining = max(self.length - self.position, 0)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b''
        if size > 32 * 1024**2 or self.bytes_read + size > 160 * 1024**2:
            raise RuntimeError('diagnostic slice exceeds the retained-data limit')
        start, end = self.position, self.position + size - 1
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                assert response.status == 206, 'blob did not honor a bounded range'
                assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{self.length}'
                raw = response.read(size + 1)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Artifact blob range HTTP {error.code}') from None
        assert len(raw) == size
        self.position += size
        self.bytes_read += size
        return raw


def digest(raw):
    return sha256(raw).hexdigest()


def retain_product(out, raw, package, source):
    assert package['source_revision']==source and package['record_verified'] is True
    assert package['wheel_sha256']==digest(raw)
    pins=package['members_sha256']
    assert len(pins)==256
    assert package['sole_core'] in pins
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        names=wheel.namelist()
        record_name,=[name for name in names if name.endswith('.dist-info/RECORD')]
        record=wheel.read(record_name)
        rows=list(csv.reader(io.StringIO(record.decode('utf-8'),newline='')))
        assert len(names)==len(set(names))==len(rows)==len(pins)
        expected={name:(token,length) for name,token,length in rows}
        assert expected[record_name]==('','') and set(expected)==set(names)==set(pins)
        for name in names:
            path=PurePosixPath(name)
            assert path.parts and not path.is_absolute() and '..' not in path.parts and all(':' not in part for part in path.parts) and '\\' not in name
            payload=record if name==record_name else wheel.read(name)
            assert digest(payload)==pins[name]
            if name!=record_name:
                token='sha256='+base64.urlsafe_b64encode(bytes.fromhex(pins[name])).decode().rstrip('=')
                assert expected[name]==(token,str(len(payload)))
            dest=(out/'package').joinpath(*path.parts)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream:stream.write(payload)
    return dict(scope='wheel member and installed RECORD integrity only; qualification failed',
                wheel_archive_retained=False,verified_package_members=len(pins),
                wheel_derived_site='package',wheel_record_path='package/'+record_name,
                wheel_record_sha256=digest(record),core_member=package['sole_core'],
                core_sha256=pins[package['sole_core']],wheel_sha256=digest(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=int)
    parser.add_argument('profile', choices=('minimum','modern','windows','debug'))
    parser.add_argument('--retain-product',action='store_true')
    args = parser.parse_args()
    assert not args.retain_product or args.profile in ('minimum','windows')
    root = Path(__file__).resolve().parent
    repo = 'bipxx/scikit-imagexx'
    run = 45011599601
    api_path = f'repos/{repo}/actions/artifacts/{args.artifact}'
    metadata = json.loads(subprocess.check_output(['gh', 'api', api_path]))
    assert metadata['id'] == args.artifact and not metadata['expired']
    assert metadata['workflow_run']['id'] == run and metadata['name'] == 'public-qualification-' + args.profile
    token = subprocess.check_output(['gh', 'auth', 'token'], text=True).strip()
    req = urllib.request.Request('https://api.github.com/' + api_path + '/zip',
                                 headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=15).close()
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f'Artifact authorization HTTP {error.code}') from None
        url = error.headers.get('Location', '')
    else:
        raise RuntimeError('No signed artifact redirect')
    assert urllib.parse.urlparse(url).scheme == 'https'
    source = 'aef6527c8cf865d52ae6a572ea4d4d86bb51f7ff'
    requested = {'manifest.json', 'state.json', 'commands.json', 'build-identity.json', 'package.json', 'sdk.json',
                 'plan.json', 'sdk-build.log', 'sdk-normal.log', 'sdk-scalar.log',
                 'normal-public-post203.log', 'scalar-public-post203.log', 'normal-public-post203.xml', 'scalar-public-post203.xml',
                 'qualification-verify-' + args.profile + '.json'}
    archive = RemoteZip(url, metadata['size_in_bytes'])
    with archive, zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        manifests = [info for info in infos if info.filename.endswith('/' + args.profile + '/manifest.json')]
        if not manifests:
            manifests = [info for info in infos if info.filename == 'manifest.json']
        assert len(manifests) == 1, [info.filename for info in manifests]
        manifest_raw = z.read(manifests[0]); manifest = json.loads(manifest_raw)
        assert metadata['workflow_run']['head_sha'] == source
        assert manifest['source']['revision'] == source and manifest['profile'] == args.profile and not manifest['passed']
        wanted = {info.filename.rsplit('/', 1)[-1]: info for info in infos
                  if info.filename.rsplit('/', 1)[-1] in requested and not info.filename.startswith('tmp/')
                  and (info.filename.rsplit('/', 1)[0] == manifests[0].filename.rsplit('/', 1)[0] or
                       info.filename.endswith('qualification-verify-' + args.profile + '.json'))}
        failed_labels = {name for name, value in manifest['counts'].items()
                         if any(value.get(k, 0) for k in ('errors', 'failures', 'skipped'))}
        for label in failed_labels:
            for suffix in ('.xml', '.log'):
                key = label + suffix
                actual = manifests[0].filename.removesuffix('manifest.json') + key
                try:
                    wanted[key] = z.getinfo(actual)
                except KeyError:
                    pass
        out = root / f'hosted-{run}' / ('failed-range-' + args.profile)
        out.mkdir(exist_ok=False, parents=True)
        files = {}
        for name, info in wanted.items():
            assert info.file_size < 24 * 1024**2
            raw = z.read(info)
            record_name = info.filename.removeprefix(manifests[0].filename.removesuffix('manifest.json'))
            pin = manifest['records'].get(record_name)
            if pin:
                assert pin == {'bytes': len(raw), 'sha256': digest(raw)}, name
            with (out / name).open('xb') as f:
                f.write(raw)
            files[name] = digest(raw)
        if 'manifest.json' not in files:
            (out / 'manifest.json').write_bytes(manifest_raw)
            files['manifest.json'] = digest(manifest_raw)
        package_scope = None
        if args.retain_product and failed_labels == {'normal-shared','scalar-shared'}:
            member,=[name for name in manifest['records'] if name.startswith('product-wheels/scikit_imagexx-')
                     and name.endswith('.whl')]
            wheel=z.read(manifests[0].filename.removesuffix('manifest.json')+member)
            assert manifest['records'][member]=={'bytes':len(wheel),'sha256':digest(wheel)}
            package_scope=retain_product(out,wheel,json.loads((out/'package.json').read_bytes()),source)
        data = dict(run=run, artifact=args.artifact, profile=args.profile, revision=source,
                    individual_product_scope=package_scope,
                    registry=manifest.get('expected_registry', {}).get('image_count'), failures=manifest['failures'],
                    bad_groups=sorted(failed_labels), retained=files, bytes_read=archive.bytes_read,
                    listed_archive_sha256=metadata.get('digest'), whole_archive_verified=False)
        (out / 'root-range-readback.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
        print(json.dumps(data))


if __name__ == '__main__':
    main()
