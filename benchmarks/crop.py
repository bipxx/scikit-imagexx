"""Prospective allocation-inclusive public crop workloads; no work at import."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/crop-benchmark-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def source_safe(spec):
    assert spec['dtype']in('uint8','int16','float32','float64','object')and spec['layout']in('C','F','reverse')
    assert 1<=len(spec['shape'])<=3 and all(type(n)is int and 3<=n<=513 for n in spec['shape'])
    assert math.prod(spec['shape'])<=263000 and type(spec['width'])is int and 0<=spec['width']<=2
    assert all(n>2*spec['width']for n in spec['shape'])and type(spec['copy'])is bool
    assert spec['payload_provider']==(spec['dtype']=='object')
    if spec['dtype']=='object':assert math.prod(spec['shape'])<=272
    return dict(input_elements=math.prod(spec['shape']),max_bytes=math.prod(spec['shape'])*8)


def fixture(spec,np):
    count=source_safe(spec)['input_elements']
    if spec['dtype']=='object':a=np.array(['crop-payload-'+str(i)for i in range(count)],dtype=object)
    else:a=np.arange(count,dtype=np.int64).astype(spec['dtype'])
    a=a.reshape(spec['shape'])
    if spec['layout']=='F':a=np.asfortranarray(a)
    elif spec['layout']=='reverse':a=a[...,::-1]
    return a,dict(copy=spec['copy'],order=spec['order'])


def main():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from next_util_execution import execute
    return execute('crop',globals())


if __name__=='__main__':main()
