"""Fixed whole-call find_contours cases for a future one-worker speed gate."""

import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def fixed_cases(np):
    recipe = runpy.run_path(str(ROOT / 'tests/find_contours_cases.py'))
    wanted = recipe['TIMED']
    cases = [(name, image, options) for name, image, options, native in recipe['cases'](np)
             if native and name in wanted]
    assert tuple(name for name, _, _ in cases) == wanted
    return cases


def public_call(function, image, options):
    return function(image, **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(json.loads((ROOT/'coverage/contracts/find-contours-v1.json').read_bytes())['timing'], indent=2))
