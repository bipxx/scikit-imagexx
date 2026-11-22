"""The fixed whole-public-call single-layer pyramid comparison cases."""

import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def fixed_cases(np):
    recipe = runpy.run_path(str(ROOT / 'tests/pyramid_spatial_cases.py'))
    for spec in recipe['FIXED']:
        yield spec['id'], spec['api'], recipe['fixture'](spec['shape'], spec['dtype'], np), spec['options']


def public_call(function, image, options):
    return function(image, **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    scope = ROOT / 'coverage/contracts/pyramid-spatial-v1.json'
    print(json.dumps(json.loads(scope.read_bytes())['timing'],indent=2))
