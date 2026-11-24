"""Fixed allocation-inclusive FAST workloads; source-only plan."""

from pathlib import Path
import json
import runpy

ROOT = Path(__file__).resolve().parents[1]


def public_cases(np):
    recipe = runpy.run_path(str(ROOT / 'tests/corner_fast_cases.py'))
    cases = {name: (image, options) for name, image, options, _ in recipe['cases'](np)}
    return [(name, *cases[name]) for name in recipe['TIMED']]


def public_call(function, image, options):
    return function(image, **options)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    contract = json.loads((ROOT / 'coverage/contracts/corner-fast-v1.json').read_bytes())
    print(json.dumps(contract['timing'], indent=2))
