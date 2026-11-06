"""Bounded source-first array view recipes; no target imports or calls."""
import math
APIS=('util.view_as_blocks','util.view_as_windows')
SIGNATURES={'util.view_as_blocks':'(arr_in, block_shape)','util.view_as_windows':'(arr_in, window_shape, step=1)'}
MAX_INPUT=256
MAX_OUTPUT=512
from crop_contract import DTYPES,ArrayProvider,InterfaceProvider,SequenceProbe,_materialize


def seq(*values,kind='tuple'):return {'container':kind,'items':list(values)}


def cases():
    rows=[]
    def add(api,name,**changes):
        row=dict(id=api.split('.')[-1]+'/'+name,api=api,shape=[4,6],dtype='int16',layout='C',readonly=False,input_kind='ndarray',parameter=seq(2,2),step=1,mutation=True)
        row.update(changes);rows.append(row)
    for api in APIS:
        add(api,'baseline')
        for layout in ('F','transpose','reverse_rows','reverse_columns','reverse_all','step_all'):add(api,'layout-'+layout,layout=layout)
        add(api,'readonly',readonly=True,mutation=False)
        for dtype in DTYPES:
            if dtype!='int16':add(api,'dtype-'+dtype,dtype=dtype,mutation=False)
        for kind in ('list','tuple','python_scalar','numpy_scalar','subclass','array_provider','array_provider_error','interface_provider','memoryview'):
            if api==APIS[0]and kind=='numpy_scalar':continue # scalar coercion ownership remains quarantined
            add(api,'input-'+kind,input_kind=kind,shape=[]if kind in ('python_scalar','numpy_scalar')else[4,6],parameter=seq()if kind in ('python_scalar','numpy_scalar')else seq(2,2),mutation=False)
        for name,shape,parameter in (('rank1',[6],seq(2)),('rank3',[2,4,6],seq(1,2,3)),('rank0',[],seq()),('empty',[0,6],seq(2,2)),('singleton',[1,6],seq(1,2))):add(api,name,shape=shape,parameter=parameter,mutation=False)
        for name,parameter in (('zero',seq(0,2)),('negative',seq(-1,2)),('rank-mismatch',seq(2)),('none',None)):
            add(api,'parameter-'+name,parameter=parameter,mutation=False)
    api=APIS[0]
    add(api,'non-tuple',parameter=seq(2,2,kind='list'),mutation=False)
    add(api,'non-divisible',parameter=seq(3,2),mutation=False)
    add(api,'invalid-parameter-before-provider',parameter=None,input_kind='array_provider_error',mutation=False)
    api=APIS[1]
    for name,parameter in (('scalar',2),('float-scalar',1.5),('bool-scalar',True),('list',seq(2,2,kind='list')),('numpy-scalar',{'np_scalar':'int64','value':2}),('sequence-probe',{'sequence_probe':[1,2]}),('too-large',seq(5,2))):add(api,'parameter-'+name,parameter=parameter,mutation=False)
    for name,step in (('scalar2',2),('axis2',seq(2,3)),('zero',0),('negative',-1),('axis-zero',seq(0,1)),('float',1.5),('rank-mismatch',seq(1))):add(api,'step-'+name,step=step,mutation=False)
    add(api,'input-error-before-parameter',input_kind='array_provider_error',parameter=None,step=0,mutation=False)
    return rows


def literal(value):
    if type(value)is dict:
        if set(value)=={'container','items'}:
            assert value['container']in('tuple','list')and type(value['items'])is list and len(value['items'])<=3
            return tuple(literal(v)for v in value['items'])
        if set(value)=={'np_scalar','value'}:assert value['np_scalar']=='int64';return literal(value['value'])
        if value=={'sequence_probe':[1,2]}:return (1,2)
        raise AssertionError('quarantined parameter protocol')
    assert value is None or type(value)in(int,float,bool)
    if type(value)is float:assert math.isfinite(value)and value in(1.5,2.0)
    if type(value)is int:assert -1<=value<=8
    return value


def source_safe(spec):
    assert set(spec)=={'id','api','shape','dtype','layout','readonly','input_kind','parameter','step','mutation'}
    assert spec['api']in APIS and type(spec['shape'])is list and len(spec['shape'])<=3
    assert all(type(v)is int and 0<=v<=6 for v in spec['shape'])and math.prod(spec['shape'])<=MAX_INPUT
    assert spec['dtype']in DTYPES and spec['layout']in('C','F','transpose','reverse_rows','reverse_columns','reverse_all','step_all')
    assert spec['input_kind']in('ndarray','subclass','list','tuple','python_scalar','numpy_scalar','array_provider','array_provider_error','interface_provider','memoryview')
    assert all(type(spec[k])is bool for k in('readonly','mutation'))
    shape=tuple(spec['shape']);shape=shape[::-1]if spec['layout']=='transpose'else shape
    if spec['layout']in('reverse_rows','reverse_columns','step_all','transpose'):assert len(shape)==2
    p=literal(spec['parameter']);step=literal(spec['step']);ndim=len(shape);api=spec['api']
    # Non-array inputs fail before view construction (blocks reads ndim;
    # windows rejects isinstance). Blocks memoryview has bounded buffer metadata;
    # NumPy scalar metadata may proceed too, so it is separately quarantined.
    assert not(spec['api']==APIS[0]and spec['input_kind']=='numpy_scalar'),'quarantined scalar conversion ownership'
    proceeds=('ndarray','subclass','memoryview')if api==APIS[0]else('ndarray','subclass')
    if spec['input_kind']not in proceeds:return dict(kind='pre_view_error',reason='input type/metadata gate',input_elements=math.prod(shape),output_elements=0)
    if api==APIS[0]:
        if not(type(spec['parameter'])is dict and spec['parameter'].get('container')=='tuple'):return dict(kind='pre_view_error',reason='block tuple gate',input_elements=math.prod(shape),output_elements=0)
        # Float/bool/object block dimensions are deliberately quarantined even
        # if a runtime could reject them: do not speculate at as_strided.
        assert all(type(v)is int for v in p),'quarantined block dimension type'
        if any(v<=0 for v in p)or len(p)!=ndim or any(n%b for n,b in zip(shape,p)):return dict(kind='pre_view_error',reason='block positivity/rank/divisibility gate',input_elements=math.prod(shape),output_elements=0)
        out=tuple(n//b for n,b in zip(shape,p))+p
        proof='i*b+j covers exactly input axis0..n-1; original signed strides retained'
    else:
        if p is None:return dict(kind='pre_view_error',reason='window len gate',input_elements=math.prod(shape),output_elements=0)
        p=tuple(p)if type(p)is tuple else(p,)*ndim
        if len(p)!=ndim:return dict(kind='pre_view_error',reason='window rank gate',input_elements=math.prod(shape),output_elements=0)
        assert all(type(v)in(int,float,bool)for v in p)
        p=tuple(int(v)for v in p) # source casts into arr_shape integer dtype
        if any(v<1 or v>n for n,v in zip(shape,p)):return dict(kind='pre_view_error',reason='window size gate',input_elements=math.prod(shape),output_elements=0)
        if type(step)is not tuple:
            if step<1:return dict(kind='pre_view_error',reason='scalar step gate',input_elements=math.prod(shape),output_elements=0)
            step=(step,)*ndim
        if len(step)!=ndim or any(type(v)is float or v==0 for v in step):return dict(kind='pre_view_error',reason='step rank/basic slice index gate',input_elements=math.prod(shape),output_elements=0)
        assert all(type(v)is int and v>0 for v in step),'quarantined sequence negative step'
        out=tuple((n-w)//st+1 for n,w,st in zip(shape,p,step))+p
        proof='0<=i*step+j<=floor((n-w)/step)*step+w-1<=n-1; signed input strides retained'
    assert math.prod(out)<=MAX_OUTPUT
    return dict(kind='bounded_view',input_elements=math.prod(shape),output_elements=math.prod(out),output_shape=list(out),index_proof=proof)


def fixture(spec,np):
    bound=source_safe(spec)
    from crop_contract import fixture as crop_fixture
    request=dict(id='crop/array-view-carrier',api='util.crop',input_kind=spec['input_kind'],shape=spec['shape'],dtype=spec['dtype'],layout=spec['layout'],readonly=spec['readonly'],width=0,copy=False,order='K')
    (array,_),_,events,buffers=crop_fixture(request,np)
    parameter=_materialize(spec['parameter'],np,events);step=_materialize(spec['step'],np,events)
    events.clear()
    return (array,parameter),{}if spec['api']==APIS[0]else{'step':step},events,buffers,bound
