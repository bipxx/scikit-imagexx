"""Fixed full-public-call Zhang cases for a future one-worker comparison."""

import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def fixed_cases(np):
    module = runpy.run_path(str(ROOT / 'tests/skeletonize_cases.py'))
    return [(spec['id'], module['fixture'](spec, np), spec['method']) for spec in module['FIXED']]


def public_call(function, image, method):
    return function(image, method=method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    contract = ROOT / 'coverage/contracts/skeletonize-v1.json'
    print(json.dumps(json.loads(contract.read_bytes())['timing'], indent=2))
