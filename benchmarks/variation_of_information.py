"""Prospective full public VI calls; no source plan authorizes computation."""
import argparse
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/variation-of-information-benchmark-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def source_safe(spec):
    assert spec in case_specs()
    # Existing source-bounded pair fixture/geometry, no entropy computation.
    old=runpy.run_path(str(ROOT/'benchmarks/label_pair_metrics.py'))
    proxy=dict(spec,api='metrics.adapted_rand_error',normalize=True,alpha=.5)
    return old['source_safe'](proxy)


def fixture(spec,np):
    source_safe(spec);old=runpy.run_path(str(ROOT/'benchmarks/label_pair_metrics.py'))
    args,options=old['fixture'](dict(spec,api='metrics.adapted_rand_error',normalize=True,alpha=.5),np)
    options.pop('alpha');return args,options


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true');args=p.parse_args()
    data=json.loads(CONTRACT.read_bytes());bounds=[source_safe(s)for s in case_specs()]
    if args.plan:print(json.dumps(dict(status=data['status'],cases=case_specs(),bounds=bounds,method=data['method'],binding=data['binding']),indent=2));return
    raise SystemExit('VI_UNBOUND: genuine product/SDK/root cohort and separate timing GO required')


if __name__=='__main__':main()
