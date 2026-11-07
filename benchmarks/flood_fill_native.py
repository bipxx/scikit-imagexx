"""Allocation-inclusive public flood and fill; all execution authority external."""
import json
from math import prod
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/flood-fill-benchmark-v1.json'

def case_specs():return json.loads(CONTRACT.read_bytes())['routes']['flood-fill']['cases']

def case_bounds(spec):
    assert spec['api'] in ('morphology.flood','morphology.flood_fill')
    assert spec['dtype'] in ('uint8','float32') and spec['layout'] in ('C','F')
    assert spec['shape']==[257,257] and spec['tolerance'] in (None,1)
    return dict(elements=prod(spec['shape']),queue_entries=prod(spec['shape']),max_bytes=prod(n+2 for n in spec['shape'])*32)

def fixture(spec,np):
    case_bounds(spec)
    image=np.ones(spec['shape'],dtype=spec['dtype'],order=spec['layout'])
    image[::17,::19]=2
    args=(image,(128,128))
    if spec['api']=='morphology.flood_fill':args+= (7,)
    return args,dict(tolerance=spec['tolerance'])

def main():
    from flood_fill_execution import execute
    return execute('flood-fill',globals())

if __name__=='__main__':main()
