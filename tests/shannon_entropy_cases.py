"""Shannon entropy prospective symbols/base policy, before original outputs.

Counts are exact sorted UNIQUE frequencies, with all NaNs one symbol and
signed zeros equal. Entropy arithmetic is always binary64, independent of
image dtype. Reuse the existing frozen NMI entropy-only budget in nats:
2e-13+2e-12*abs(Hnats). For finite base conversion propagate through
1/abs(log(base)) and add one unit-roundoff*abs(Hbase); no denominator floor
or fitted base-dependent factor. Near1 explicitly uses1+2^-30. Empty and
constant Hnats0 remain exact; nonfinite outputs/warnings/errors/state exact.
"""
import math
API='measure.shannon_entropy'
STAGES=('shannon_counts','shannon_entropy_counts')
UNASSESSED=('forged/stale storage','arbitrary object comparison protocols','extended numeric dtypes','noncanonical bool storage','base protocols')

def cases():
    rows=[]
    def add(id,pattern='balanced',dtype='uint8',layout='C',base=2,unique='native',outcome='return',policy=None):
        rows.append(dict(id=id,pattern=pattern,dtype=dtype,layout=layout,base=base,unique=unique,outcome=outcome,policy=policy or {}))
    for id,pattern,dtype in [('empty','empty','uint8'),('scalar','scalar','float64'),('constant','constant','bool'),('balanced','balanced','uint8'),('skew','skew','int8'),('four','four','uint16'),('continuous','continuous','float32'),('wide-u64','u64','uint64'),('half','half','float16'),('symbols','symbols','float64')]:add(id,pattern,dtype)
    for layout in ('F','step','reverse','endian','readonly'):add('layout-'+layout,'four','float64',layout)
    for name,base in [('none',None),('e','e'),('10',10),('half',.5),('one-warn',1),('one-raise',1),('zero',0),('negative',-2),('nan','nan'),('inf','inf'),('near1','near1')]:
        add('base-'+name,pattern='constant'if name=='one-raise'else'balanced',base=base,
            outcome='error'if name in('one-raise','zero','negative')else'return',policy={'invalid':'raise'}if name=='one-raise'else{})
    add('complex','complex','complex128',unique='provider');add('object','object','object',unique='provider');add('unicode','unicode','U1',unique='provider');add('wide-i64','i64','int64')
    assert len(rows)==30 and len({r['id']for r in rows})==30
    return rows

def fixture(spec,np):
    p=spec['pattern'];dtype=spec['dtype']
    if p=='empty':image=np.empty((0,3),dtype=dtype)
    elif p=='scalar':image=np.array(-3.5,dtype=dtype)
    elif p=='constant':image=np.ones((3,5),dtype=dtype)
    elif p=='skew':image=np.array([-3]*7+[2],dtype=dtype)
    elif p=='four':image=np.array([0,0,1,1,2,2,3,3],dtype=dtype).reshape(2,4)
    elif p=='continuous':image=np.arange(257,dtype=dtype)/np.float32(16)
    elif p=='u64':image=np.array([2**63,2**63+1,2**63+1,2**63+1,2**63,2**64-1],dtype=dtype)
    elif p=='i64':image=np.array([-2**63,-2**63+1,2**63-1]*2,dtype=dtype)
    elif p=='half':image=np.array([0,.5,1,.5,1,1],dtype=dtype)
    elif p=='symbols':
        words=[0,1<<63,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0xfff8000000000007,0x7ff8000000000009,0x7ff0000000000000,0]
        image=np.array(words,dtype=np.uint64).view(np.float64).copy()
    elif p=='complex':image=np.array([0j,-0j,1+2j,1+2j,complex(math.nan,1),complex(2,math.nan)],dtype=dtype)
    elif p in('object','unicode'):image=np.array(['b','a','b','a','c'],dtype=dtype)
    else:image=np.array([0,0,0,0,1,1,1,1],dtype=dtype)
    layout=spec['layout']
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='step':
        base=np.zeros((image.shape[0]*2,image.shape[1]*2),dtype=image.dtype);base[::2,::2]=image;image=base[::2,::2]
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='readonly':image.flags.writeable=False
    base=spec['base'];base={'e':math.e,'nan':math.nan,'inf':math.inf,'near1':1+2**-30}.get(base,base)if isinstance(base,str)else base
    return image,dict(base=base)

def error_limit(base,nats,expected,np):
    if not np.isfinite(expected)or nats==0:return 0.
    denominator=1. if base is None else abs(math.log(base))
    if math.isinf(denominator):return 0.
    assert denominator>0 and math.isfinite(denominator)
    allowance=(2e-13+2e-12*abs(float(nats)))/denominator
    if base is not None:allowance+=float(np.finfo(np.float64).eps)/2*abs(float(expected))
    assert math.isfinite(allowance)and allowance>=0
    return allowance
