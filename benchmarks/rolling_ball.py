"""Prospective one-worker public rolling_ball workloads; execution stays UNBOUND."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def plan():
    d=json.loads((ROOT/'coverage/contracts/rolling-ball-v1.json').read_bytes())
    return dict(state='PROSPECTIVE_UNBOUND',api=d['api'],cases=d['benchmark_cases'],method=d['benchmark_method'],source=None,producer=None,runtime=None,core=None,SDK=None,root_GO=None)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',action='store_true');args=parser.parse_args()
    if args.plan:print(json.dumps(plan(),indent=2));return
    raise SystemExit('ROLLING_BALL_PROSPECTIVE_UNBOUND: no public-call original/native/sampler authority')
if __name__=='__main__':main()
