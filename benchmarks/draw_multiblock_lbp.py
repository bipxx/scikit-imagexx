"""Prospective allocation-inclusive one-thread visualizer; no measurement GO."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['benchmark_cases']


def case_bounds(row):
    import math
    assert row in case_specs()
    n=math.prod(row['shape']);assert n<=1048576
    assert row['code']in(0,85,170,255)and row['alpha']==0.5
    assert row['dtype']in('uint8','float32','float64')and row['layout']in('C','F','reverse')
    return dict(input_elements=n,max_output_elements=n*(3 if len(row['shape'])<3 else 1),source_updates=8,threads=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',action='store_true');args=parser.parse_args()
    if args.plan:
        data=json.loads(CONTRACT.read_bytes())
        print(json.dumps(dict(state='PROSPECTIVE_UNBOUND_NO_MEASUREMENT',cases=case_specs(),bounds=[case_bounds(r)for r in case_specs()],method=data['benchmark_method']),indent=2));return
    raise SystemExit('DRAW_MULTIBLOCK_LBP_UNBOUND: capture/native/helper/sampler/timing budget0; root gates required')


if __name__=='__main__':main()
