"""Prospective one-worker public join benchmark; refused before science imports."""
import argparse
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/join-segmentations-benchmark-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def case_bounds(spec):
    assert spec['dtype'] in ('uint8','uint16','int16','int32')
    assert spec['layout'] in ('C','F','reverse') and type(spec['mapping']) is bool
    assert len(spec['shape'])==2 and all(type(n)is int and 2<=n<=513 for n in spec['shape'])
    assert type(spec['left_count'])is int and 2<=spec['left_count']<=32
    assert type(spec['right_count'])is int and 2<=spec['right_count']<=8
    assert spec['left_scale'] in (1,7) and spec['right_scale'] in (1,7)
    count=math.prod(spec['shape']);factor=(spec['right_count']-1)*spec['right_scale']+1
    encoded=(spec['left_count']-1)*factor+spec['right_count']-1
    limit={'uint8':255,'uint16':65535,'int16':32767,'int32':2147483647}[spec['dtype']]
    assert encoded<limit and count<=262144
    return dict(input_elements_each=count,original_factor=factor,encoded_upper=encoded,
                distinct_pair_upper=spec['left_count']*spec['right_count'],maximum_dense_map_elements=spec['left_count']*spec['right_count']+1)


def fixture(spec,np):
    bounds=case_bounds(spec);n=bounds['input_elements_each']
    left=((np.arange(n,dtype=np.uint32)%spec['left_count'])*spec['left_scale']).astype(spec['dtype']).reshape(spec['shape'])
    right=(((np.arange(n,dtype=np.uint32)//spec['left_count'])%spec['right_count'])*spec['right_scale']).astype(spec['dtype']).reshape(spec['shape'])
    if spec['layout']=='F':left,right=np.asfortranarray(left),np.asfortranarray(right)
    elif spec['layout']=='reverse':left,right=left[:,::-1],right[:,::-1]
    return (left,right),dict(return_mapping=spec['mapping'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',action='store_true');args=parser.parse_args()
    contract=json.loads(CONTRACT.read_bytes());bounds=[case_bounds(s) for s in contract['cases']]
    if args.plan:
        print(json.dumps(dict(state=contract['state'],cases=contract['cases'],bounds=bounds,method=contract['method']),indent=2));return
    # No runnable measurement until genuine source/run/core/root admission is
    # bound through the existing admitted controller. This packet adds no new
    # dispatcher and must not import science or call original while unbound.
    raise SystemExit('PROSPECTIVE_UNBOUND: genuine input and independent root benchmark GO required')


if __name__=='__main__':main()
