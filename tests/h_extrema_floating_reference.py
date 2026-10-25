"""Original-only H floating captures; append raw observations before checks."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'coverage/contracts/h-extrema-floating-plan.json'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate(profile):
    plan = json.loads(PLAN.read_bytes())
    old = load('h_floating_unchanged_authentication', 'h_extrema_reference.py')
    _, authentication = old.authenticate(profile)
    for name, expected in {**plan['preserved_files'], **plan['reference_files']}.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            provenance=load('h_floating_source_provenance','h_extrema_source_provenance.py')
            evolution=provenance.authenticate_source(ROOT,name,expected)
            authentication.setdefault('floating_source_evolution',[]).append(evolution)
        else:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    helper = plan['warning_helpers'][profile]
    path = Path(importlib.metadata.distribution('scikit-image').locate_file(helper['relative']))
    raw = path.read_bytes()
    assert len(raw) == helper['bytes'] and hashlib.sha256(raw).hexdigest() == helper['sha256']
    authentication['warning_helper'] = dict(path=str(path), bytes=len(raw), sha256=helper['sha256'])
    return plan, authentication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('windows', 'modern', 'minimum'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix('.jsonl').exists()
    plan, authentication = authenticate(args.profile)
    module = load('h_floating_frozen_factory', 'test_h_extrema_floating_cases.py')
    legacy = load('h_floating_original_legacy', 'test_h_extrema_contracts.py')
    assert legacy.R.profile() == args.profile
    assert module.cases() == plan['cases']
    rows, failures = {}, []
    with args.output.with_suffix('.jsonl').open('x') as journal:
        for case in module.cases():
            observation = module.observe(case, legacy)
            rows[case['id']] = observation
            journal.write(json.dumps(dict(id=case['id'], observation=observation), allow_nan=False) + '\n')
            journal.flush()
            failure = None
            if observation['source_unsafe']:
                failure = 'unexpected source safety refusal'
            elif observation['error'] and observation['error']['type'] in ('AssertionError', 'SourceUnsafeBoundary'):
                failure = 'source certificate failure'
            elif case['family'] in ('nonrepresentable-height', 'wide-unsigned-comparison'):
                if observation['error'] or observation['original_numeric_completed'] != 1:
                    failure = 'required original Cython completion missing'
            if failure:
                failures.append(dict(id=case['id'], reason=failure))
            print(case['id'], observation['error'], observation['original_numeric_completed'], flush=True)
    report = dict(schema='h-extrema-floating-original-profile-v1', source_authentication=authentication,
                  plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(), observations=rows,
                  certificate_failures=failures, native_calls=0, operation_timings=0)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    assert not failures, failures
    print('CAPTURED', args.profile, len(rows), 'observations;',
          sum(r['original_numeric_completed'] for r in rows.values()), 'genuine Cython completions; native0/timing0')


if __name__ == '__main__':
    main()
