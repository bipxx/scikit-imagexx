"""Frozen RDP workload parameters and independent selection oracle.

No scientific imports or original/native calls occur at import. The executable
public benchmark will be connected after registered parity passes.
"""
import json
import math
from pathlib import Path

CONTRACT=Path(__file__).resolve().parents[1]/'coverage/contracts/approximate-polygon-benchmark.json'


def case_specs():
    return json.loads(CONTRACT.read_bytes())['cases']


def points(spec):
    """Logical coordinates, independent of their subsequent storage layout."""
    n=spec['points'];kind=spec['geometry']
    if kind=='collinear':values=[(float(i),float(2*i+3))for i in range(n)]
    elif kind=='zigzag':values=[(float(i),1. if i%2 else -1.)for i in range(n)]
    elif kind=='smooth':
        values=[(i*(16.*math.pi/(n-1)),math.sin(i*(16.*math.pi/(n-1))))for i in range(n)]
    else:raise ValueError('Unknown frozen polygon geometry')
    if spec['layout']=='reverse-rows':values.reverse()
    return values


def fixture(spec,np):
    values=np.array(points(spec),dtype=np.float64)
    layout=spec['layout']
    if layout=='F':return np.asfortranarray(values)
    if layout=='reverse-rows':return values[::-1].copy()[::-1]
    if layout=='step-rows':
        owner=np.full((2*len(values),2),-911.,dtype=np.float64)
        owner[::2]=values
        return owner[::2]
    if layout=='step-columns':
        owner=np.full((len(values),4),-911.,dtype=np.float64)
        owner[:,::2]=values
        return owner[:,::2]
    assert layout=='C'
    return values


def selection_oracle(values,tolerance,trig=None):
    """Independent scalar RDP placement, with an explicit elementary provider.

    Keep source subtraction/multiply/add/sqrt order, strict projection and
    tolerance predicates, first maximum and left-first stack traversal. The
    provider returns sin(-atan2(dr,dc)),cos(-atan2(dr,dc)); timed Python can pass
    the exact NumPy scalar provider without sharing native selection logic.
    """
    if trig is None:
        def trig(dr,dc):
            angle=-math.atan2(dr,dc)
            return math.sin(angle),math.cos(angle)
    keep=[False]*len(values);keep[0]=keep[-1]=True
    stack=[(0,len(values)-1)]
    while stack:
        start,end=stack.pop();r0,c0=values[start];r1,c1=values[end]
        dr,dc=r1-r0,c1-c0;sine,cosine=trig(dr,dc)
        segment_dist=c0*sine+r0*cosine
        maximum=-math.inf;index=None;split=False
        for i in range(start+1,end):
            r,c=values[i];dr0,dc0=r-r0,c-c0;dr1,dc1=r-r1,c-c1
            p0=dr0*dr+dc0*dc;p1=(-dr1)*dr-dc1*dc
            if p0>0 and p1>0:distance=abs(r*cosine+c*sine-segment_dist)
            else:distance=min(math.sqrt(dc0*dc0+dr0*dr0),math.sqrt(dc1*dc1+dr1*dr1))
            split=split or distance>tolerance
            if distance>maximum:maximum,index=distance,i
        if split:
            assert index is not None
            keep[index]=True;stack.extend(((index,end),(start,index)))
    return [i for i,value in enumerate(keep)if value]


if __name__=='__main__':
    raise SystemExit('Prospective cohort: executable timing awaits registered parity and genuine product admission.')
