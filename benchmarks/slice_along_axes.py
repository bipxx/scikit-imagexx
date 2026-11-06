"""Prospective allocation-inclusive public slicing; no execution before GO."""
import argparse
import json
import math
from pathlib import Path


def case_specs():
    return [dict(id=name,shape=shape,layout=layout,copy=copy,dtype=dtype,slices=slices,axes=axes)
        for name,shape,layout,copy,dtype,slices,axes in [
            ('view-C',[512,513],'C',False,'int16',[[32,480]],None),
            ('copy-C',[512,513],'C',True,'int16',[[32,480]],None),
            ('view-F',[512,513],'F',False,'float64',[[32,480],[16,496]],[0,1]),
            ('copy-F-C-order',[512,513],'F',True,'float64',[[32,480],[16,496]],[0,1]),
            ('reverse-view',[512,513],'reverse',False,'uint8',[[0,480]],None),
            ('reverse-copy',[512,513],'reverse',True,'uint8',[[0,480]],None),
            ('permuted-axes',[64,65,66],'C',False,'int32',[[1,60],[2,62]],[2,0]),
            ('negative-bound-copy',[512,513],'C',True,'int16',[[-480,-16]],[1])]]


def case_bounds(row):
    import math
    assert set(row)=={'id','shape','layout','copy','dtype','slices','axes'}
    assert row in case_specs()
    n=math.prod(row['shape']);assert n<=1048576
    return dict(input_elements=n,max_output_elements=n,source_slices=len(row['slices']),source_axes=len(row['shape']),threads=1)


def fixture(row,np):
    case_bounds(row);n=math.prod(row['shape'])
    if row['dtype']=='object':image=np.array(['view-payload-'+str(i)for i in range(n)],dtype=object)
    else:image=np.arange(n,dtype=np.int64).astype(row['dtype'])
    image=image.reshape(row['shape'])
    if row['layout']=='F':image=np.asfortranarray(image)
    elif row['layout']=='reverse':image=image[...,::-1]
    if row['layout']=='readonly':image.flags.writeable=False
    return (image,[tuple(pair)for pair in row['slices']]),dict(axes=row['axes'],copy=row['copy'])


def main():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from next_util_execution import execute
    return execute('slice',globals())


if __name__=='__main__':main()
