"""Prospective allocation-inclusive public mark_boundaries; no execution GO."""
import argparse
import json
import math


def case_specs():
    return [dict(id=name,shape=shape,rgb=rgb,dtype=dtype,layout=layout,mode=mode,outline=outline)
        for name,shape,rgb,dtype,layout,mode,outline in (
            ('gray-u8-outer',[512,513],False,'uint8','C','outer',False),
            ('RGB-f32-thick',[512,513],True,'float32','C','thick',False),
            ('gray-u16-F-inner',[512,513],False,'uint16','F','inner',False),
            ('RGB-f64-reverse-outline',[512,513],True,'float64','reverse','outer',True),
            ('gray-f64-subpixel',[64,65],False,'float64','C','subpixel',False),
            ('RGB-f32-subpixel-outline',[64,65],True,'float32','C','subpixel',True),
            ('singleton-subpixel-version',[1,1],False,'float64','C','subpixel',False),
            ('half-promotion-provider',[512,513],False,'float16','C','outer',False))]


def case_bounds(row):
    assert row in case_specs()
    n=math.prod(row['shape']);assert n<=263000
    out=math.prod(2*n-1 for n in row['shape'])if row['mode']=='subpixel'else n
    assert out*3<=4*1024*1024
    return dict(input_pixels=n,output_words=out*3,output_byte_bound=out*3*8,threads=1,source_resampler_required=row['mode']=='subpixel')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',action='store_true');args=parser.parse_args()
    if args.plan:
        print(json.dumps(dict(status='PROSPECTIVE_UNBOUND_NO_MEASUREMENT',cases=case_specs(),bounds=[case_bounds(s)for s in case_specs()],sampler='tools/montage_sustained.py:measure',repeat=7,target_ms=100,warmup_min_ms=250,threads=1,retain_all_losses=True,root_GO=None),indent=2));return
    raise SystemExit('UNBOUND: independent source safety/originalGO/native/genuine/public performance authority required before imports/calls')


if __name__=='__main__':main()
