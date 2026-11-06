"""Prospective unique execution locators; frozen arguments/method unchanged."""
import argparse,json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def plan():
    old=runpy.run_path(str(ROOT/'benchmarks/finite_radon_native.py'))['plan']()
    # Historical plans keyed three layouts only by n127. Preserve that source
    # key and make the future execution locator unique without changing cases.
    old['cases']=[dict(row,source_case_id=row['id'],id=row['id']+'/'+row['layout'])for row in old['cases']]
    old['native_acceptance']['inverse_error_prefix']='_core._finite_radon_inverse_prefix(a)'
    old['source_case_projection']='Drop source_case_id and restore id=source_case_id to recover exact frozen eight rows.'
    return old

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true');a=p.parse_args()
    if a.plan:print(json.dumps(plan()));return
    raise SystemExit('FINITE_RADON_UNBOUND_NO_TIMING')
if __name__=='__main__':main()
