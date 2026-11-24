"""Four frozen full-public-call cases for subsequent medial-axis timing."""

import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def fixed_cases(np):
    source = runpy.run_path(str(ROOT / 'tests/medial_axis_cases.py'))
    for spec in source['FIXED']:
        image, mask = source['fixture'](spec, np)
        yield spec['id'], image, mask, spec['return_distance'], spec['seed']


def public_call(function, image, mask, return_distance, seed):
    return function(image, mask, return_distance, rng=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    contract = ROOT / 'coverage/contracts/medial-axis-v1.json'
    print(json.dumps(json.loads(contract.read_bytes())['timing'], indent=2))
