"""Source/test-first display recipes; no original/native/helper execution."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json'
API='feature.draw_multiblock_lbp'
SIGNATURE='(image, r, c, width, height, lbp_code=0, color_greater_block=(1, 1, 1), color_less_block=(0, 0.69, 0.96), alpha=0.5)'
OFFSETS=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
UNASSESSED=('unbounded allocations/forged storage/concurrent mutation/arbitrary protocol code',
 'longdouble padding/platform rules and signaling-NaN/trapping/unmasked FP contexts',
 'full dtype/scalar-promotion/eventful FP parity remains unmeasured until rootGO',
 'Windows/default-int/intp and unsupported standalone SIMD/worker/fullSDK domains')


def cases():
    rows=[]
    def add(name,**kw):
        row=dict(id=API+'/'+name,shape=[9,10],dtype='uint8',layout='C',pattern='ramp',r=1,c=1,width=2,height=2,code=0,alpha=0.5,greater=[1,1,1],less=[0,0.69,0.96],carrier='array',policy={},quarantine=False)
        row.update(kw);rows.append(row)
    add('default-gray')
    add('RGB-uint8',shape=[9,10,3])
    for dtype in ('bool','uint16','uint32','uint64','int8','int16','int32','int64','float16','float32','float64','>u2','>f4'):
        add('dtype-'+dtype,dtype=dtype,shape=[7,8,3])
    add('signed-extrema',dtype='int64',pattern='extrema',shape=[7,8,3])
    add('unsigned-extrema',dtype='uint64',pattern='extrema',shape=[7,8,3])
    add('float-unscaled-range',dtype='float64',pattern='range',shape=[7,8,3])
    for i in range(8):add('bit-'+str(7-i),shape=[6,6,3],dtype='float64',pattern='zeros',r=0,c=0,code=1<<(7-i),alpha=1.,greater=[1,0,0],less=[0,0,1])
    for name,code in [('all-one',255),('negative-code',-1),('high-bit-only',256),('high-plus-low',513),('bool-code',True),('np-int64-code',5),('np-uint8-code',129)]:
        add(name,code=code,code_kind=name if name.startswith('np-')else'python')
    for name,alpha in [('alpha-zero',0.),('alpha-one',1.),('alpha-negative',-0.5),('alpha-over-one',1.5),('alpha-np-f32',0.3),('alpha-np-f64',0.3)]:
        add(name,shape=[7,8,3],dtype='float32',alpha=alpha,alpha_kind=name if name.startswith('alpha-np')else'python')
    for policy in ('warn','raise','call'):
        add('FP-inf-'+policy,dtype='float64',shape=[5,6,3],pattern='zeros',alpha='inf',policy={'invalid':policy,'over':policy})
    add('FP-nan',dtype='float64',shape=[5,6,3],alpha='nan')
    add('image-nan-alpha-one',dtype='float64',shape=[5,6,3],pattern='nan',alpha=1.)
    add('color-nan-alpha-zero',dtype='float64',shape=[5,6,3],greater=['nan',1,0],less=['nan',0,1],alpha=0.)
    for name,r,c in [('top-left-clipping',0,0),('bottom-right-clipping',7,8),('negative-wrapping',-2,-3),('wrapped-overlap',-7,-8),('outside-positive',32,32),('outside-negative',-32,-32)]:add(name,r=r,c=c)
    for name,width,height in [('zero-width',0,2),('zero-height',2,0),('negative-width',-2,2),('negative-height',2,-2),('bool-size',True,True),('float-width',2.,2),('float-height',2,2.)]:add(name,width=width,height=height)
    add('float-row',r=1.5)
    add('np-coordinate',geometry_kind='np-int32')
    add('np0d-width',geometry_kind='np0d-width')
    for channels in (1,2,4):add('channels-'+str(channels),shape=[5,6,channels])
    add('rank4',shape=[5,6,2,3])
    add('rank1',shape=[8])
    add('rank0',shape=[])
    add('empty-rows',shape=[0,6])
    add('empty-columns',shape=[5,0,3])
    for layout in ('F','reverse','readonly','strided'):add('layout-'+layout,shape=[7,8,3],dtype='float32',layout=layout)
    add('bad-greater',greater=['bad',0,1])
    add('unused-bad-less',less=['bad',0,1],code=255)
    add('scalar-color',greater=0.75,less=0.25)
    for n in (0,1,2,4):add('color-length-'+str(n),greater=[0.5]*n,less=[0.25]*n)
    add('matrix-color',greater=[[1,0,0],[0,1,0]],less=[[0,0,1],[1,0,1]])
    add('float-code',code=1.5)
    add('array-code',code=3,code_kind='array-two')
    add('alpha-vector',alpha=0.5,alpha_kind='vector-two')
    add('alpha-none',alpha=None)
    add('image-list',carrier='list')
    add('image-protocol',carrier='protocol')
    add('color-protocol',color_kind='protocol')
    add('alpha-protocol',alpha_kind='protocol')
    add('object-image',dtype='object')
    add('complex-image',dtype='complex64')
    add('missing-height',arity='missing-height')
    add('unknown-keyword',extra_keyword=True)
    return rows


def _flat(v):return [x for child in v for x in _flat(child)]if isinstance(v,list)else[v]


def source_safe(row):
    assert row in cases(),'Only exact frozen recipes'
    assert len(row['shape'])<=4 and all(type(n)is int and 0<=n<=10 for n in row['shape'])
    n=math.prod(row['shape']);assert n<=4096
    assert all(type(row[k])in(int,float,bool)and abs(row[k])<=32 for k in('r','c','width','height'))
    assert abs(row['code'])<=1024 and row['layout']in('C','F','reverse','readonly','strided')
    color_values=_flat(row['greater'])+_flat(row['less']);assert len(color_values)<=12
    assert all(v in('nan','bad')if isinstance(v,str)else type(v)in(int,float)and math.isfinite(v)and abs(v)<=2 for v in color_values)
    assert row['alpha']in(None,'nan','inf')or(type(row['alpha'])in(int,float)and abs(row['alpha'])<=2)
    # Basic NumPy slices never escape live owners. Frozen rank<=4 and axis<=10
    # bound every broadcast temporary by10**4 words, even error/empty branches.
    shape=row['shape']+[3]if len(row['shape'])<3 else row['shape']
    return dict(factory_elements=n,max_output_elements=math.prod(shape),max_broadcast_temporary_elements=10000,max_temporary_payload_bytes=3*10000*16,
                source_neighbor_updates=8,geometry_integer_abs_upper=128,controlled_protocols=True,original_call_budget=0,
                safety_basis='Small allocated ordinary arrays or bounded controlled providers; fixed8 NumPy basic slices/ufuncs, no unchecked coordinate/Cython descriptor helper calls')


class AlphaProtocol:
    def __init__(self,value,events):self.value,self.events=value,events
    def __rsub__(self,other):self.events.append('1-alpha');return other-self.value
    def __mul__(self,other):self.events.append('alpha*color');return self.value*other


def fixture(row,np):
    source_safe(row);events=[];buffers={}
    from draw_coordinates_contract import ArrayProvider
    shape=tuple(row['shape']);n=math.prod(shape)
    image=(np.arange(n,dtype=np.int64)%7).reshape(shape)
    if row['pattern']=='zeros':image=np.zeros(shape,dtype=np.int64)
    if row['pattern']=='range':image=image.astype(np.float64)-3.
    if row['dtype']=='object':image=np.array(['payload']*n,dtype=object).reshape(shape)
    else:image=image.astype(row['dtype'])
    if row['pattern']=='extrema':
        limits=np.iinfo(image.dtype);image.flat[0]=limits.min;image.flat[-1]=limits.max
    if row['pattern']=='nan':image.flat[0]=np.nan
    if row['layout']=='F':image=np.asfortranarray(image)
    elif row['layout']=='reverse':image=image[...,::-1]
    elif row['layout']=='readonly':image.flags.writeable=False
    elif row['layout']=='strided':
        backing=np.empty((shape[0]*2,)+shape[1:],dtype=image.dtype);backing[::2]=image;image=backing[::2]
    buffers['image']=image
    def number(value):return float(value)if value in('nan','inf')else value
    def color(value,name):
        if isinstance(value,list):value=[[number(x)for x in item]if isinstance(item,list)else number(item)for item in value]
        else:value=number(value)
        if row.get('color_kind')=='protocol':
            array=np.array(value,dtype=np.float64);buffers[name]=array;return ArrayProvider(array,events)
        return value
    alpha=number(row['alpha']);alpha_kind=row.get('alpha_kind')
    if alpha_kind=='alpha-np-f32':alpha=np.float32(alpha)
    elif alpha_kind=='alpha-np-f64':alpha=np.float64(alpha)
    elif alpha_kind=='vector-two':alpha=np.array([0.25,0.75])
    elif alpha_kind=='protocol':alpha=AlphaProtocol(alpha,events)
    code=row['code'];kind=row.get('code_kind')
    if kind=='np-int64-code':code=np.int64(code)
    elif kind=='np-uint8-code':code=np.uint8(code)
    elif kind=='array-two':code=np.array([code,code])
    r,c,width,height=(row[k]for k in('r','c','width','height'))
    if row.get('geometry_kind')=='np-int32':r,c=np.int32(r),np.int32(c)
    elif row.get('geometry_kind')=='np0d-width':width=np.array(width,dtype=np.int64)
    argument=image
    if row['carrier']=='list':argument=image.tolist()
    elif row['carrier']=='protocol':
        class ImageProtocol(ArrayProvider):
            @property
            def shape(self):self.events.append('image.shape');return self.array.shape
        argument=ImageProtocol(image,events)
    args=(argument,r,c,width,height)
    if row.get('arity')=='missing-height':args=args[:-1]
    options=dict(lbp_code=code,color_greater_block=color(row['greater'],'greater'),color_less_block=color(row['less'],'less'),alpha=alpha)
    if row.get('extra_keyword'):options['shape']=(5,6)
    return args,options,events,buffers


def encode(value,np):
    # Reuse admitted bounded crop/coordinate codecs; preserve object values,
    # never pointer payloads. Controlled protocol descriptors do not execute it.
    from crop_observer import encode as leaf
    from draw_coordinates_contract import ArrayProvider
    if isinstance(value,AlphaProtocol):return dict(controlled_protocol='AlphaProtocol',value=leaf(value.value,np))
    if isinstance(value,ArrayProvider):return dict(controlled_protocol=type(value).__name__,array=leaf(value.array,np))
    if isinstance(value,dict):return {key:encode(v,np)for key,v in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(v,np)for v in value])
    return leaf(value,np)


def validate_intake(path):
    data=json.loads(Path(path).read_bytes());raw=CONTRACT.read_bytes();contract=json.loads(raw)
    assert set(data)=={'schema','mode','contract_sha256','source_files_sha256','original_calls'}
    assert data['schema']=='draw-multiblock-lbp-intake-v1'and data['mode']=='intake-only'and type(data['original_calls'])is int and data['original_calls']==0
    assert data['contract_sha256']==hashlib.sha256(raw).hexdigest()and data['source_files_sha256']==contract['source_freeze']
    for name,pin in data['source_files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    return data


def runtime_request():
    import numpy as np
    import skimage.feature as feature
    assert str(inspect.signature(feature.draw_multiblock_lbp))==SIGNATURE
    return np,feature


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',action='store_true');parser.add_argument('--intake-binding',type=Path);args=parser.parse_args()
    if args.plan:
        print(json.dumps(dict(status='SOURCE_ONLY_CAPTURE_BUDGET0',cases=cases(),bounds=[source_safe(s)for s in cases()]),indent=2));return
    if args.intake_binding is None:raise SystemExit('DRAW_MULTIBLOCK_LBP_UNBOUND: source-only packet; original budget0')
    validate_intake(args.intake_binding);np,feature=runtime_request()
    print(json.dumps(dict(status='INTAKE_ONLY_NO_TARGET_CALLS',numpy=np.__version__,original_calls=0,helper_calls=0)))


if __name__=='__main__':main()
