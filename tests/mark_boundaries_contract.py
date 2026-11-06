"""Source-only recipes and bounded NumPy factories; no target dispatch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

API='segmentation.mark_boundaries'
SIGNATURE="(image, label_img, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)"
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/mark-boundaries-v1.json'
DTYPES=('bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64','float16','float32','float64','complex64','complex128','object','S3','big_endian_i16','tagged_i16')
LABELS=('bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64','float32','float64')
LAYOUTS=('C','F','transpose_spatial','reverse_rows','reverse_columns','reverse_channels','step_spatial')
MAX_INPUT=192;MAX_PHYSICAL=4096;MAX_OUTPUT=1024
SOURCE_FILES=('tests/mark_boundaries_contract.py','tests/test_mark_boundaries_intake.py','tests/test_mark_boundaries_reference.py','coverage/contracts/mark-boundaries-v1.json','benchmarks/mark_boundaries.py','tests/crop_contract.py','tests/crop_observer.py','tests/draw_coordinates_contract.py','tests/draw_coordinates_observer.py')


def color(values=(1,1,0),kind='tuple'):return dict(kind=kind,values=list(values))


def cases():
    rows=[]
    def add(name,**changes):
        row=dict(id='mark_boundaries/'+name,api=API,image_shape=[3,4],image_dtype='uint8',image_kind='ndarray',image_layout='C',image_pattern='ramp',label_shape=[3,4],label_dtype='uint8',label_layout='C',label_pattern='regions',readonly=False,color=color(),outline=None,mode='outer',background=0)
        row.update(changes);rows.append(row)
    add('default-gray')
    for mode in('thick','inner','outer','subpixel'):
        for rgb in(False,True):add(mode+('-RGB'if rgb else'-gray'),mode=mode,image_shape=[3,4,3]if rgb else[3,4])
    for dtype in DTYPES:
        for rgb in(False,True):add('image-'+dtype+('-RGB'if rgb else'-gray'),image_dtype=dtype,image_shape=[3,4,3]if rgb else[3,4])
    for dtype in LABELS:
        add('labels-'+dtype,label_dtype=dtype,mode='thick'if dtype.startswith('float')or dtype in('uint64','int64')else'outer')
    for layout in LAYOUTS[1:]:
        for rgb in(False,True):
            if layout=='reverse_channels'and not rgb:continue
            add('layout-'+layout+('-RGB'if rgb else'-gray'),image_shape=[3,4,3]if rgb else[3,4],image_layout=layout,label_layout=layout if layout!='reverse_channels'else'C')
    add('readonly',readonly=True,image_shape=[3,4,3]);add('signed-image-limits',image_dtype='int16',image_pattern='limits')
    add('signed-labels',label_dtype='int8',label_pattern='negative',mode='inner',background=-1)
    add('raw-bool-labels',label_dtype='bool',label_pattern='raw-bool')
    for mode in('outer','subpixel'):
        for outline in(color((0,0,0)),color((0.5,),kind='scalar'),color((1,0),kind='list')):
            add(mode+'-outline-'+str(len(rows)),mode=mode,outline=outline)
    for name,value in [('scalar',color((0.25,),kind='scalar')),('single',color((0.5,))),('short',color((1,0))),('four',color((1,0,0,1))),('array',color(kind='array')),('reverse-array',color(kind='reverse_array')),('provider',color(kind='provider')),('none',color((),kind='none')),('string',color((),kind='string')),('out-of-range',color((-1,2,0))),('nan',color(('nan',0,1))),('infinity',color(('inf',0,1))),('tiny',color(('tiny',0,1)))]:
        add('color-'+name,color=value,image_dtype='float32')
    for kind in('list','tuple','array_provider','array_provider_error','memoryview','subclass'):add('image-kind-'+kind,image_kind=kind)
    for kind in('invalid',None,7):add('mode-'+str(kind),mode=kind) # Source treats every non-special mode as thick.
    for value in(-1,2,2**70,1.5,None,'none'):add('background-'+str(value),background=value)
    for shape in([1,1],[1,4],[4,1],[2,2]):add('subpixel-size-'+str(shape),image_shape=shape,label_shape=shape,mode='subpixel')
    add('subpixel-F',mode='subpixel',image_layout='F',image_shape=[3,4,3]);add('subpixel-outline-readonly',mode='subpixel',outline=color((0,0,0)),readonly=True)
    add('subpixel-uint64-sentinel',mode='subpixel',label_dtype='uint64',label_pattern='maximum')
    for shape in([], [4], [0,4], [3,4,1], [3,4,4]):add('image-shape-'+str(shape),image_shape=shape,color=color((0.5,),kind='scalar')if len(shape)!=2 else color())
    add('RGBA-matching-color',image_shape=[3,4,4],color=color((1,0,0,1)))
    add('image-label-mismatch',label_shape=[2,4]);add('3D-spatial-no-outline',image_shape=[2,3,4,3],label_shape=[2,3,4],mode='thick')
    add('3D-spatial-outline-rank-error',image_shape=[2,3,4,3],label_shape=[2,3,4],mode='thick',outline=color((0,0,0)))
    add('subpixel-zero-image-before-zoom',image_shape=[0,4],mode='subpixel')
    for token in('nan','inf','negative-zero'):add('image-'+token,image_dtype='float32',image_pattern=token,mode='thick')
    return rows


def _color_safe(value):
    if value is None:return
    assert type(value)is dict and set(value)=={'kind','values'}
    assert value['kind']in('tuple','list','scalar','array','reverse_array','provider','none','string')
    assert type(value['values'])is list and len(value['values'])<=4
    if value['kind']=='scalar':assert len(value['values'])==1
    for v in value['values']:
        assert type(v)in(int,float,str)and(type(v)is not str and -2<=v<=2 or v in('nan','inf','tiny'))


def source_safe(spec):
    assert type(spec)is dict and set(spec)==set(cases()[0])
    assert spec['api']==API and type(spec['id'])is str and spec['id'].startswith('mark_boundaries/')
    for key in('image_shape','label_shape'):
        shape=spec[key];assert type(shape)is list and len(shape)<=4 and all(type(n)is int and 0<=n<=6 for n in shape)and math.prod(shape)<=MAX_INPUT
    assert 1<=len(spec['label_shape'])<=3 and all(spec['label_shape']),'quarantined empty/zeroD labels'
    assert spec['image_dtype']in DTYPES and spec['label_dtype']in LABELS
    assert spec['image_kind']in('ndarray','list','tuple','array_provider','array_provider_error','memoryview','subclass')
    for key in('image_layout','label_layout'):
        assert spec[key]in LAYOUTS
        if spec[key]in('transpose_spatial','reverse_columns','step_spatial'):assert len(spec['image_shape'if key=='image_layout'else'label_shape'])>=2
        if spec[key]=='reverse_channels':assert key=='image_layout'and len(spec['image_shape'])==3
    assert spec['image_pattern']in('ramp','limits','nan','inf','negative-zero')and spec['label_pattern']in('regions','negative','raw-bool','maximum')
    if spec['image_pattern']in('nan','inf','negative-zero'):assert spec['image_dtype']in('float32','float64')and spec['mode']!='subpixel'
    if spec['image_pattern']=='limits':assert spec['image_dtype']=='int16'
    if spec['label_pattern']=='negative':assert spec['label_dtype']=='int8'
    if spec['label_pattern']=='raw-bool':assert spec['label_dtype']=='bool'
    if spec['label_pattern']=='maximum':assert spec['mode']=='subpixel'and spec['label_dtype']=='uint64'
    assert not(spec['label_dtype']in('int64','uint64')and spec['mode']=='outer'),'quarantined wide outer double intermediate frontier'
    assert type(spec['readonly'])is bool and type(spec['mode'])in(str,int,type(None))and spec['mode']in('outer','inner','thick','subpixel','invalid',None,7)
    assert spec['background'] is None or type(spec['background'])in(int,float,str)
    if type(spec['background'])is int:assert abs(spec['background'])<=2**70
    elif type(spec['background'])is float:assert spec['background']==1.5
    elif type(spec['background'])is str:assert spec['background']=='none'
    _color_safe(spec['color']);_color_safe(spec['outline'])
    shape=spec['image_shape'];converted=shape+[3]if len(shape)==2 else shape
    spatial=converted[:-1];out=converted
    if spec['mode']=='subpixel'and all(spatial):out=[2*n-1 for n in spatial]+converted[-1:]
    physical=max(math.prod(shape),math.prod(spec['label_shape']))*4
    assert physical<=MAX_PHYSICAL and math.prod(out)<=MAX_OUTPUT
    return dict(max_input_elements=MAX_INPUT,physical_element_bound=physical,max_marked_elements=math.prod(out),max_marked_bytes=8*math.prod(out),subpixel=spec['mode']=='subpixel',subpixel_zero_divisor=spec['mode']=='subpixel'and any(n==0 for n in spatial),original_calls=0)


def fixture(spec,np):
    source_safe(spec)
    from crop_contract import fixture as carrier,ArrayProvider
    events=[];buffers={}
    def array(shape,dtype,layout,pattern,prefix):
        request=dict(id='crop/mark-carrier',api='util.crop',input_kind='ndarray',shape=shape,dtype=dtype,layout='C',readonly=False,width=0,copy=False,order='K')
        (a,_),_,_,owned=carrier(request,np);buffers.update({prefix+'_'+k:v for k,v in owned.items()})
        if prefix=='labels':
            values=np.arange(a.size,dtype=np.int64).reshape(a.shape)//2%4
            if pattern=='negative':values=values-1
            if pattern=='raw-bool':a.view(np.uint8)[...]=np.where(values==0,0,np.where(values==1,2,255)).astype(np.uint8)
            else:
                a[...]=values
                if pattern=='maximum':a.flat[-1]=np.iinfo(a.dtype).max
        elif pattern=='limits':a.flat[0]=np.iinfo(a.dtype).min;a.flat[-1]=np.iinfo(a.dtype).max
        elif pattern in('nan','inf','negative-zero'):a.flat[0]={'nan':float('nan'),'inf':float('inf'),'negative-zero':-0.0}[pattern]
        if layout=='F':a=np.asfortranarray(a)
        elif layout=='transpose_spatial':a=a.swapaxes(0,1)
        elif layout=='reverse_rows':a=a[::-1]
        elif layout=='reverse_columns':a=a[:,::-1]
        elif layout=='reverse_channels':a=a[...,::-1]
        elif layout=='step_spatial':
            a=np.repeat(np.repeat(a,2,axis=0),2,axis=1);buffers[prefix+'_expanded']=a;a=a[::2,::2]
        if spec['readonly']:a.flags.writeable=False
        buffers[prefix]=a;return a
    image=array(spec['image_shape'],spec['image_dtype'],spec['image_layout'],spec['image_pattern'],'image')
    labels=array(spec['label_shape'],spec['label_dtype'],spec['label_layout'],spec['label_pattern'],'labels')
    kind=spec['image_kind']
    if kind=='list':image=image.tolist()
    elif kind=='tuple':image=tuple(tuple(row)for row in image.tolist())
    elif kind in('array_provider','array_provider_error'):image=ArrayProvider(image,events,kind=='array_provider_error')
    elif kind=='memoryview':image=memoryview(image)
    elif kind=='subclass':
        class Carrier(np.ndarray):pass
        image=image.view(Carrier);buffers['subclass']=image
    def materialize(value,name):
        if value is None:return None
        kind=value['kind'];values=[{'nan':float('nan'),'inf':float('inf'),'tiny':2.0**-150}[x]if type(x)is str else x for x in value['values']]
        if kind=='none':return None
        if kind=='string':return 'red'
        if kind=='scalar':return values[0]
        if kind=='tuple':return tuple(values)
        if kind=='list':return values
        a=np.array(values,dtype=np.float64)
        if kind=='reverse_array':a=a[::-1];a.flags.writeable=False
        buffers[name]=a
        return ArrayProvider(a,events)if kind=='provider'else a
    options=dict(color=materialize(spec['color'],'color'),outline_color=materialize(spec['outline'],'outline'),mode=spec['mode'],background_label=spec['background'])
    events.clear();return (image,labels),options,events,buffers


def source_revision():
    git='git'if os.name=='nt'else'/mnt/c/Program Files/Git/cmd/git.exe'
    root=str(ROOT)if os.name=='nt'else str(ROOT).replace('/mnt/c/','C:/',1)
    return subprocess.check_output([git,'-C',root,'rev-parse','HEAD'],text=True).strip()


def validate_binding(path):
    data=json.loads(Path(path).read_bytes())
    assert set(data)=={'schema','status','source_revision','source_files_sha256','original_budget'}
    assert data['schema']=='mark-boundaries-intake-v1'and data['status']=='ROOT_BOUND_INTAKE_ONLY'and type(data['original_budget'])is int and data['original_budget']==0
    assert data['source_revision']==source_revision()and set(data['source_files_sha256'])==set(SOURCE_FILES)
    for name,pin in data['source_files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    return data


def intake_main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root-binding',required=True);args=parser.parse_args(argv);validate_binding(args.root_binding)
    import numpy as np  # Explicit REQUEST: cached modules cannot bypass refusal.
    import skimage.segmentation  # Tests refuse this request; never call a body.
    raise RuntimeError('INTAKE_ONLY_NO_ORIGINAL_CALL_AUTHORITY')


if __name__=='__main__':intake_main()
