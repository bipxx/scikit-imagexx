"""Fixed scalar public-call recipe for the ellipsoid-stats sustained sampler."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / 'coverage/contracts/ellipsoid-stats-v1.json'


def fixed_cases():
    contract = json.loads(CONTRACT_PATH.read_bytes())
    refs = json.loads((ROOT / contract['reference']['fixture']).read_bytes())
    timed = [row for row in refs['profiles']['minimum']['rows'] if row['timed']]
    assert [row['id'] for row in timed] == contract['timing']['case_ids']
    for row in timed:
        assert all(kind == 'float' for kind, _ in row['args'])
        yield row['id'], tuple(float(value) for _, value in row['args'])


def public_call(target, args):
    return target(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(dict(cases=list(fixed_cases()), measurement_authority=None), indent=2))
