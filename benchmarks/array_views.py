"""Prospective one-thread public borrowed views; no measurement authority."""
import argparse
import json
import math
from pathlib import Path


def case_specs():
    rows=[]
    for api in ('view_as_blocks','view_as_windows'):
        for layout,dtype in (('C','int16'),('F','float64'),('reverse','uint8'),('readonly','object')):
            rows.append(dict(id=api+'/'+layout,api='util.'+api,shape=[16,18] if dtype=='object' else [512,516],layout=layout,dtype=dtype,parameter=[2,3],step=1,copy=False,payload_provider=False,metadata_representation_provider='numpy.lib.stride_tricks.as_strided'))
    return rows


def case_bounds(row):
    assert row in case_specs() and row['copy'] is False
    n=math.prod(row['shape']);assert n<=264192
    if row['api'].endswith('blocks'):out=n
    else:out=math.prod(n-w+1 for n,w in zip(row['shape'],row['parameter']))*math.prod(row['parameter'])
    assert out<=1575936
    return dict(input_elements=n,logical_view_elements=out,payload_copies=0,threads=1)


def fixture(row,np):
    case_bounds(row);n=math.prod(row['shape'])
    if row['dtype']=='object':image=np.array(['view-payload-'+str(i)for i in range(n)],dtype=object)
    else:image=np.arange(n,dtype=np.int64).astype(row['dtype'])
    image=image.reshape(row['shape'])
    if row['layout']=='F':image=np.asfortranarray(image)
    elif row['layout']=='reverse':image=image[...,::-1]
    if row['layout']=='readonly':image.flags.writeable=False
    return (image,tuple(row['parameter'])),({}if row['api']=='util.view_as_blocks'else dict(step=row['step']))


def main():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from next_util_execution import execute
    return execute('views',globals())


if __name__=='__main__':main()
