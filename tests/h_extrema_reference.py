"""Source-authenticated additive reference capture; no native or timing path."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/h-extrema-plan.json'


def authenticate(profile):
    contract = json.loads(CONTRACT.read_bytes())
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-image')}
    assert versions == contract['source_audit'][profile]['versions']
    records = []
    for relative, expected in contract['source_audit'][profile]['files'].items():
        distribution = 'scikit-image' if relative.startswith('skimage/') else 'numpy'
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative))
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], str(path)
        records.append(dict(path=str(path), bytes=len(raw), sha256=expected['sha256']))
    for name, expected in contract['frozen_files'].items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
    return contract, dict(profile=profile, versions=versions, files=records, python=sys.version, executable=sys.executable)


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('windows', 'modern', 'minimum'), required=True)
    parser.add_argument('--suite', choices=('legacy', 'supplement'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    contract, authentication = authenticate(args.profile)
    legacy = load('h_extrema_exact_legacy', 'test_h_extrema_contracts.py')
    assert legacy.R.profile() == args.profile
    supplement = load('h_extrema_frozen_boundaries', 'test_h_extrema_boundaries.py')
    specs = legacy.cases() if args.suite == 'legacy' else supplement.cases()
    assert specs == contract[args.suite + '_cases']
    rows, failures = {}, []
    # Persist each completed observation in a separate append-only raw stream;
    # a failed certificate therefore cannot erase earlier genuine observations.
    journal = args.output.with_suffix('.jsonl')
    with journal.open('x') as stream:
        for spec in specs:
            row = legacy.observe(spec) if args.suite == 'legacy' else supplement.observe(spec, legacy)
            rows[spec['id']] = row
            failure = None
            if row['error'] and row['error']['type'] == 'AssertionError':
                failure = 'source certificate assertion'
            if spec.get('required_cython') and (row['source_unsafe'] or row['error'] or row['original_numeric_completed'] != 1):
                failure = 'required genuine original Cython witness missing'
            if failure:
                failures.append(dict(id=spec['id'], reason=failure))
            stream.write(json.dumps(dict(id=spec['id'], observation=row), allow_nan=False) + '\n')
            stream.flush()
            print(spec['id'], row['error'], row['original_numeric_completed'], flush=True)
    unsafe = [name for name, row in rows.items() if row['source_unsafe']]
    if args.suite == 'legacy':
        assert unsafe == contract['legacy_unsafe_ids']
    report = dict(schema='h-extrema-source-reference-profile-v1', suite=args.suite,
                  source_authentication=authentication, contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                  observations=rows, source_unsafe_ids=unsafe, certificate_failures=failures,
                  operation_timings=0, native_calls=0)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    assert not failures, failures
    print('CAPTURED', args.profile, args.suite, len(rows), 'observations;',
          sum(row['original_numeric_completed'] for row in rows.values()), 'Cython completions;',
          len(unsafe), 'blocked unsafe intakes; timings0', flush=True)


if __name__ == '__main__':
    main()
