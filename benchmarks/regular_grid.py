"""Prospective serial one-thread public metadata benchmark; no target imports."""
import argparse
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/regular-grid-benchmark-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def case_bounds(row):
    shape=row['shape'];assert type(shape)is list and 1<=len(shape)<=8
    assert all(type(n)is int and 1<=n<=1048576 for n in shape)
    assert type(row['points'])is int and 1<=row['points']<=2**53
    assert row['carrier'] in ('tuple','readonly-reverse')
    product=math.prod(shape);assert product<=2**53
    return dict(metadata_elements=len(shape),shape_product=product,source_product_exact=True,
                source_root_evaluations_upper=len(shape)+1,image_elements_allocated=0)


def fixture(row,np):
    case_bounds(row);shape=tuple(row['shape'])
    if row['carrier']=='readonly-reverse':
        shape=np.array(shape[::-1],dtype=np.int64)[::-1];shape.flags.writeable=False
    return (shape,row['points']),{}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',action='store_true');args=parser.parse_args()
    contract=json.loads(CONTRACT.read_bytes());bounds=[case_bounds(row) for row in contract['cases']]
    if args.plan:
        print(json.dumps(dict(state=contract['state'],cases=contract['cases'],bounds=bounds,method=contract['method']),indent=2));return
    raise SystemExit('PROSPECTIVE_UNBOUND: genuine input and independent root benchmark GO required')


if __name__=='__main__':main()
