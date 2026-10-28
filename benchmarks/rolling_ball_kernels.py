"""Prospective constructor benchmark; ABI/reduction contract still under review.

No image or dtype grid stands in for constructor parameters. This draft cannot
run original/native constructors until the reviewed admission/oracle is frozen.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/rolling-ball-kernels-benchmark.json'
ADMISSION_READY=False


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def output_shape(spec):
    params=spec['parameters']
    if spec['api']=='restoration.ball_kernel':
        return (2*math.ceil(params['radius'])+1,)*params['ndim']
    if spec['api']=='restoration.ellipsoid_kernel':
        return tuple(2*max(dimension//2,1)+1 for dimension in params['shape'])
    raise ValueError('Unknown frozen constructor API')


def constructor_arguments(spec):
    params=spec['parameters']
    if spec['api']=='restoration.ball_kernel':return (params['radius'],params['ndim'])
    if spec['api']=='restoration.ellipsoid_kernel':return (list(params['shape']),params['intensity'])
    raise ValueError('Unknown frozen constructor API')


def source_axis_sum(terms):
    """Pinned NumPy float64 fast-axis order for this rank<=8 cohort."""
    assert 1<=len(terms)<=8
    if len(terms)<8:
        total=-0.0
        for value in terms:total+=value
    else:
        total=((terms[0]+terms[1])+(terms[2]+terms[3]))+((terms[4]+terms[5])+(terms[6]+terms[7]))
    return 0.0+total


def coordinate_oracle(spec):
    """Independent placement/arithmetic with the pinned binary64 reduction order.

    The source materializes both square roots for ball and applies its distance
    mask only afterwards. Ellipsoid normalizes before square/reduction, then
    subtracts from one, clips, roots, multiplies and applies the scaling mask.
    No original constructor, meshgrid or native function is called here.
    """
    shape=output_shape(spec);params=spec['parameters'];ball=spec['api']=='restoration.ball_kernel'
    axes=[(dimension-1)//2 for dimension in shape]
    raw=bytearray()
    for point in itertools.product(*(range(-axis,axis+1)for axis in axes)):
        if ball:
            squares=[float(coordinate)*float(coordinate)for coordinate in point]
            total=source_axis_sum(squares)
            distance=math.sqrt(total)
            value=math.sqrt(max(params['radius']**2-total,0.))
            if distance>params['radius']:value=math.inf
        else:
            normalized=[float(coordinate)/float(axis)for coordinate,axis in zip(point,axes)]
            total=source_axis_sum([coordinate*coordinate for coordinate in normalized])
            scaling=1.-total
            value=params['intensity']*math.sqrt(max(scaling,0.))
            if scaling<0.:value=math.inf
        raw.extend(struct.pack('=d',value))
    return bytes(raw),shape


def output_snapshot(array):
    return dict(dtype=array.dtype.str,dtype_metadata=array.dtype.metadata,shape=list(array.shape),strides=list(array.strides),own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),base_type=type(array.base).__name__,raw=array.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-rolling-ball-kernels.json'))
    parser.parse_args()
    if not ADMISSION_READY:
        raise SystemExit('Prospective harness: await approved native ABI, reduction and private packet. No constructor calls were made.')
    raise NotImplementedError('Owned entry names and native acceptance trace await ABI freeze')


if __name__=='__main__':main()
