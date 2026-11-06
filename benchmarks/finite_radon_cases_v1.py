"""Source-bounded successful integer benchmark inputs; no target calls."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT='coverage/contracts/finite-radon-benchmark-execution-v2.json'


def case_specs(api=None):
    rows=json.loads((ROOT/CONTRACT).read_bytes())['cases']
    return rows if api is None else [r for r in rows if r['api']==api]


def case_bounds(spec):
    assert spec in case_specs() and spec['api'] in ('frt2','ifrt2')
    n=spec['n'];assert type(n)is int and 1<=n<=257
    assert spec['dtype'] in ('uint8','uint16','uint32') and spec['layout'] in ('C','F','reverse','step')
    assert spec['maximum_value']==7
    rows=n+int(spec['api']=='ifrt2');item={'uint8':1,'uint16':2,'uint32':4}[spec['dtype']]
    count=rows*n;physical=count*(4 if spec['layout']=='step'else 1)
    output=n*(n+1)if spec['api']=='frt2'else n*n
    assert n*7+7<2**32 and count<=66306 and physical<=265224
    # Original: n axis0 sums, n*(n-1) additions each; frt adds n row sums.
    # Inverse adds last row, row sum, subtraction and division; upper bound is conservative.
    additions=n*n*(n-1)+n*(n-1)+n*n
    return dict(input_shape=[rows,n],input_elements=count,physical_elements=physical,
                input_bytes=physical*item,output_elements=output,output_bytes=output*(4 if spec['api']=='frt2'else 8),
                source_roll_calls=(n-1)**2,source_roll_copy_elements=n*(n-1)**2,
                addition_upper=additions,total_arithmetic_upper=additions+2*n*n,
                peak_live_array_bytes_upper=32*count+physical*item+output*16,
                source_accumulator_max=7*n,original_calls=0)


def fixture(spec,np):
    b=case_bounds(spec);shape=tuple(b['input_shape'])
    # Source input only; no frt/ifrt computation to generate inverse data.
    a=(np.arange(b['input_elements'],dtype=np.uint32)%8).astype(spec['dtype']).reshape(shape)
    if spec['layout']=='F':a=np.asfortranarray(a)
    elif spec['layout']=='reverse':a=a[:,::-1]
    elif spec['layout']=='step':
        owner=np.zeros((shape[0]*2,shape[1]*2),dtype=spec['dtype']);view=owner[::2,::2];view[...]=a;a=view
    return (a,),{}
