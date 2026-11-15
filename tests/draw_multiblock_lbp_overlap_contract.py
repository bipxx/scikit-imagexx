"""One additive source-safe repeated-overlap recipe; no image algorithm."""
import argparse
import hashlib
import json
from pathlib import Path
import draw_multiblock_lbp_contract as legacy
ROOT=legacy.ROOT
CONTRACT=ROOT/'coverage/contracts/draw-multiblock-lbp-v2.json'


def added_case():
    row=next(s for s in legacy.cases() if s['id']==legacy.API+'/dtype-float32')
    return dict(row,id=legacy.API+'/v2-true-wrapped-overlap-f32',shape=[3,3,3],r=-3,c=-3,width=2,height=2,code=170,alpha=0.3)


def cases():return legacy.cases()+[added_case()]


def source_indices(row):
    # Literal source central_rect + int neighbor offset * extent, then basic
    # slice normalization. This establishes membership only, no image arithmetic.
    def patch(dr,dc):
        r=row['r']+row['height']+dr*row['height']
        c=row['c']+row['width']+dc*row['width']
        rr=range(*slice(r,r+row['height']).indices(row['shape'][0]))
        cc=range(*slice(c,c+row['width']).indices(row['shape'][1]))
        return [[a,b] for a in rr for b in cc]
    return dict(patches=[patch(*o) for o in legacy.OFFSETS],normalized_source_center=patch(0,0))


def source_safe(row):
    assert row==added_case(),'Only exact added overlap recipe'
    return dict(factory_seed_elements=168,factory_elements=27,max_output_elements=27,
                max_broadcast_temporary_elements=10000,source_neighbor_updates=8,original_call_budget=0)


def fixture(row,np):
    source_safe(row)
    seed=next(s for s in legacy.cases() if s['id']==legacy.API+'/dtype-float32')
    args,options,events,buffers=legacy.fixture(seed,np)
    # First27 C-order ramp values match the legacy recipe's exact ramp formula.
    image=buffers['image'].reshape(-1)[:27].reshape(3,3,3).copy()
    options=dict(options,lbp_code=row['code'],alpha=row['alpha'])
    return (image,row['r'],row['c'],row['width'],row['height']),options,events,dict(image=image)


def validate_intake(path):
    data=json.loads(Path(path).read_bytes());raw=CONTRACT.read_bytes();contract=json.loads(raw)
    assert set(data)=={'schema','mode','contract_sha256','source_files_sha256','original_calls'}
    assert data['schema']=='draw-multiblock-lbp-intake-v2' and data['mode']=='intake-only'
    assert type(data['original_calls']) is int and data['original_calls']==0
    assert data['contract_sha256']==hashlib.sha256(raw).hexdigest()
    assert data['source_files_sha256']==contract['source_freeze']
    for name,pin in data['source_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true');parser.add_argument('--intake-binding',type=Path)
    args=parser.parse_args()
    if args.plan:
        print(json.dumps(dict(status='SOURCE_ONLY_CAPTURE_BUDGET0',added_case=added_case(),indices=source_indices(added_case()),bounds=source_safe(added_case()))));return
    if args.intake_binding is None:raise SystemExit('DRAW_MULTIBLOCK_LBP_UNBOUND: source-only v2; original budget0')
    validate_intake(args.intake_binding)
    legacy.runtime_request()  # Existing real request path; intake tests intercept imports.


if __name__=='__main__':main()
