"""Literal SDK words projected from saved rows; no NumPy or reference calls."""
import base64
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[1]

def render():
    from flood_fill_contract import recipes
    data=json.loads((ROOT/'tests/fixtures/flood_fill_original_v1.json').read_bytes())
    rows=data['profiles']['minimum']['rows'];specs=recipes();calls=[]
    def vec(values):return '{'+','.join(str(v)for v in values)+'}'
    for row,spec in zip(rows,specs):
        if row['error']:continue
        a=row['before']['items'][0]['items'][0];result=row['result']
        dtype=a['dtype'];kind=dtype[1];size=int(dtype[2:]);T=('float'if size==4 else 'double')if kind=='f'else 'std::'+('u'if kind in'ub'else '')+'int'+str(size*8)+'_t'
        raw=zlib.decompress(base64.b64decode(a['raw_zlib_base64']));gold=zlib.decompress(base64.b64decode(result['raw_zlib_base64']))
        shape=spec['shape'];ndim=len(shape);foot=spec['footprint'];side=5 if foot=='large'else 3
        if foot:
            fp=[int(foot not in('cross','no-center','center-only'))or int(foot=='cross'and(i//side==1 or i%side==1))or int(foot=='no-center'and i in(1,5))or int(foot=='center-only'and i==4)for i in range(side**ndim)]
        else:
            conn=ndim if spec['connectivity']is None else max(1,spec['connectivity']);fp=[]
            for i in range(3**ndim):
                j=i;distance=0
                for axis in range(ndim):distance+=abs(j%3-1);j//=3
                fp.append(int(distance<=conn))
        seed=spec['seed']if isinstance(spec['seed'],list)else[spec['seed']]
        seed=[s%n if n else 0 for s,n in zip(seed,shape)]
        tol=spec['tolerance'];tol=1 if tol=='np-int8-one'else tol
        low=high='0'
        if tol is not None:
            # All admitted tolerance rows have int16 plateau seed2 or float32.
            tolerance=float('inf')if tol=='inf'else abs(tol)
            seedvalue=2
            if kind=='f':minimum=-3.4028234663852886e38 if size==4 else -1.7976931348623157e308;maximum=-minimum
            else:minimum=-(2**(size*8-1))if kind=='i'else 0;maximum=2**(size*8-(1 if kind=='i'else 0))-1
            low=str(max(minimum,seedvalue-tolerance));high=str(min(maximum,seedvalue+tolerance))
        value=spec['new_value'];value=44 if spec['id']=='flood_fill:new-value-out-of-range'else value
        if kind=='b':value=int(bool(value))
        calls.append('check<'+T+'>('+','.join([json.dumps(row['id']),vec(shape),vec(a['strides']),vec(seed),vec([side]*ndim),vec(fp),str(spec['layout']in('F','transpose')or sum(n>1 for n in shape)<=1).lower(),vec(raw),vec(gold),str(spec['api']=='flood_fill').lower(),str(value),str(tol is not None).lower(),'static_cast<'+T+'>('+low+')','static_cast<'+T+'>('+high+')'])+');')
    return '// Literal minimum saved return words; numeric casting policy is a Python provider.\n'+ '\n'.join(calls)+'\n'

if __name__=='__main__':
    (ROOT/'tests/fixtures/flood_fill_sdk_original_v1.inc').write_text(render(),encoding='utf8',newline='\n')
