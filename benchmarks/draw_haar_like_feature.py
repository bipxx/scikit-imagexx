"""Fixed Haar drawing recipes for the retained public-call sampler."""
import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]

def fixed_cases():
    recipe = runpy.run_path(str(ROOT/'tests/haar_feature_drawing_cases.py'))
    contract = json.loads((ROOT/'coverage/contracts/draw-haar-like-feature-v1.json').read_bytes())
    assert recipe['FAST_CASES'] == contract['timing']['case_ids']
    by_name = {case['id']:case for case in recipe['cases']()}
    return [by_name[name] for name in recipe['FAST_CASES']], recipe


def public_call(target,args,kwargs):
    return target(*args,**kwargs)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true',required=True)
    parser.parse_args()
    rows,_=fixed_cases()
    print(json.dumps({'cases':rows,'execution_authority':None},indent=2))
