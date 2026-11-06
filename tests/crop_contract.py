"""Prospective crop recipes; constructing recipes executes no array operations."""
import math
API='util.crop'
SIGNATURE="(ar, crop_width, copy=False, order='K')"
MAX_INPUT_ELEMENTS=256
MAX_PHYSICAL_ELEMENTS=4096
DTYPES=('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex64','complex128','S3','U2','V3','packed','object','datetime64[D]','timedelta64[D]','tagged_i16','big_endian_i16')
INPUTS=('ndarray','list','tuple','python_scalar','numpy_scalar','subclass','array_provider','array_provider_error','interface_provider','memoryview')
LAYOUTS=('C','F','transpose','reverse_rows','reverse_columns','reverse_all','step_all','permute3')


def seq(*values,kind='tuple'):return {'container':kind,'items':list(values)}


def cases():
    rows=[]
    def add(name,**changes):
        row=dict(id='crop/'+name,api=API,input_kind='ndarray',shape=[4,5],dtype='int16',layout='C',readonly=False,width=1,copy=False,order='K')
        row.update(changes);rows.append(row)
    widths={'zero':0,'scalar':1,'single-tuple':seq(1),'single-list':seq(1,kind='list'),'pair-tuple':seq(1,2),'pair-list':seq(1,2,kind='list'),'broadcast-pair':seq(seq(1,2)),'axis-pairs':seq(seq(0,1),seq(2,0)),'axis-lists':seq(seq(0,1,kind='list'),seq(2,0,kind='list'),kind='list'),'bool':True,'numpy-int64':{'np_scalar':'int64','value':1},'numpy-uint64':{'np_scalar':'uint64','value':1},'numpy-width-vector':{'width_array':[1,2]},'negative':-1,'negative-pair':seq(-2,1),'mixed-outside':seq(seq(-100,100),seq(1,-1)),'large':2**70,'large-pair':seq(2**70,-2**70)}
    for name,width in widths.items():add('width-'+name,width=width)
    for copy in (False,True):
        for order in ('K','C','F','A','invalid',None,0,'c'):add('copy-'+str(copy)+'-order-'+str(order),copy=copy,order=order)
    for name,shape in (('rank1',[7]),('rank3',[3,4,5]),('rank4',[2,3,2,2]),('empty-leading',[0,5]),('empty-trailing',[4,0])):add(name,shape=shape,width=0)
    add('zeroD-view',shape=[],width=0);add('zeroD-copy',shape=[],width=1,copy=True)
    for layout in LAYOUTS[1:]:add('layout-'+layout,layout=layout,shape=[2,3,4]if layout=='permute3'else[4,5],width=seq(seq(0,1))if layout=='permute3'else seq(seq(0,1),seq(1,0)))
    for order in ('K','C','F','A'):add('copy-permuted3-'+order,layout='permute3',shape=[2,3,4],width=0,copy=True,order=order)
    add('readonly-view',readonly=True);add('readonly-copy',readonly=True,copy=True)
    for dtype in DTYPES:
        if dtype!='int16':add('dtype-'+dtype,dtype=dtype,width=0)
    for dtype in ('packed','object','tagged_i16','complex128'):add('copy-dtype-'+dtype,dtype=dtype,width=0,copy=True)
    add('zeroD-object-view',shape=[],dtype='object',width=0)
    add('zeroD-packed-view',shape=[],dtype='packed',width=0)
    for kind in INPUTS[1:]:add('input-'+kind,input_kind=kind,shape=[]if kind in('python_scalar','numpy_scalar')else[4,5],width=0)
    for name,width in (('none',None),('float',1.5),('empty',seq()),('flat-length3',seq(1,2,3)),('pair-length1',seq(seq(1))),('wrong-axes',seq(seq(1,1),seq(1,1),seq(1,1))),('float-index',seq(seq(1.5,0))),('literal-dict',{'literal_dict':True})):
        add('width-error-'+name,width=width)
    for truth in (False,True,'raise'):add('copy-truth-'+str(truth),copy={'bool_probe':truth})
    add('width-sequence-probe',width={'sequence_probe':[1,2]})
    add('width-index-probe',width=seq(seq({'index_probe':1},0)))
    add('width-index-raises',width=seq(seq({'index_probe':'raise'},0)))
    add('coercion-error-before-width',input_kind='array_provider_error',width=None)
    add('object-distinct-copy',input_kind='distinct_objects',dtype='object',shape=[2,2],width=0,copy=True)
    add('zeroD-distinct-object',input_kind='distinct_objects',dtype='object',shape=[],width=0)
    return rows


def _literal_safe(value,depth=0):
    assert depth<=4
    if value is None or type(value)is bool:return
    if type(value)is int:assert abs(value)<=2**70;return
    if type(value)is float:assert math.isfinite(value)and abs(value)<=100;return
    assert type(value)is dict
    if set(value)=={'container','items'}:
        assert value['container']in('tuple','list')and type(value['items'])is list and len(value['items'])<=8
        for item in value['items']:_literal_safe(item,depth+1)
    elif set(value)=={'np_scalar','value'}:
        assert value['np_scalar']in('int64','uint64')and type(value['value'])is int
        assert (-(2**63)<=value['value']<2**63)if value['np_scalar']=='int64'else(0<=value['value']<2**64)
    elif set(value)=={'width_array'}:assert type(value['width_array'])is list and len(value['width_array'])<=8 and all(type(x)is int and abs(x)<=100 for x in value['width_array'])
    elif set(value)=={'sequence_probe'}:assert value['sequence_probe']==[1,2]
    elif set(value)=={'index_probe'}:assert value['index_probe']in(1,'raise')
    else:assert value=={'literal_dict':True}


def source_safe(spec):
    assert type(spec)is dict
    assert set(spec)=={'id','api','input_kind','shape','dtype','layout','readonly','width','copy','order'}
    assert type(spec['api'])is str and spec['api']==API and type(spec['id'])is str and spec['id'].startswith('crop/')
    shape=spec['shape'];assert type(shape)is list and len(shape)<=4 and all(type(n)is int and 0<=n<=8 for n in shape)
    count=math.prod(shape);assert count<=MAX_INPUT_ELEMENTS
    assert all(type(spec[key])is str for key in ('dtype','input_kind','layout'))
    assert spec['dtype']in DTYPES and spec['input_kind']in INPUTS+('distinct_objects',) and spec['layout']in LAYOUTS
    assert type(spec['readonly'])is bool
    if spec['layout']=='permute3':assert len(shape)==3
    elif spec['layout']!='C':assert len(shape)==2
    if spec['input_kind']in('python_scalar','numpy_scalar'):assert shape==[]
    if spec['input_kind']in('list','tuple'):assert len(shape)==2 and spec['dtype']=='int16'
    if spec['input_kind']=='memoryview':assert spec['dtype']=='int16'and spec['layout']=='C'
    if spec['input_kind']=='distinct_objects':assert spec['dtype']=='object'and spec['layout']=='C'
    physical=count*(2**len(shape)if spec['layout']=='step_all'else 1);assert physical<=MAX_PHYSICAL_ELEMENTS
    _literal_safe(spec['width'])
    assert type(spec['copy'])is bool or(type(spec['copy'])is dict and set(spec['copy'])=={'bool_probe'}and(type(spec['copy']['bool_probe'])is bool or spec['copy']['bool_probe']=='raise'and type(spec['copy']['bool_probe'])is str))
    assert type(spec['order'])in(str,int,type(None))and spec['order']in('K','C','F','A','invalid',None,0,'c')
    # Every ordinary step1 slice is within its original array; clipping negative
    # and arbitrarily large Python integer bounds cannot enlarge an allocation.
    return dict(logical_input_elements=count,physical_input_elements=physical,max_result_elements=count,max_payload_bytes=16*physical,max_object_references=physical if spec['dtype']=='object'else 0,width_container_depth_bound=4,original_calls=0)


class IndexProbe:
    def __init__(self,value,events):self.value,self.events=value,events
    def __index__(self):
        self.events.append({'event':'index','value':self.value})
        if self.value=='raise':raise TypeError('crop index probe refused')
        return self.value


class BoolProbe:
    def __init__(self,value,events):self.value,self.events=value,events
    def __bool__(self):
        self.events.append({'event':'copy.bool','value':self.value})
        if self.value=='raise':raise RuntimeError('crop copy truth probe refused')
        return self.value


class SequenceProbe:
    def __init__(self,items,events):self.items,self.events=items,events
    def __len__(self):self.events.append({'event':'width.len'});return len(self.items)
    def __getitem__(self,index):self.events.append({'event':'width.getitem','index':index});return self.items[index]


class ArrayProvider:
    def __init__(self,array,events,error=False):self.array,self.events,self.error=array,events,error
    def __array__(self,dtype=None,copy=None):
        self.events.append({'event':'array.__array__','dtype':None if dtype is None else str(dtype),'copy':copy})
        if self.error:raise RuntimeError('crop array provider refused')
        return self.array


class InterfaceProvider:
    def __init__(self,array,events):self.array,self.events=array,events
    @property
    def __array_interface__(self):
        self.events.append({'event':'array.__array_interface__'})
        return dict(self.array.__array_interface__) # Living owner retained; no forged pointer/span.


def _materialize(value,np,events):
    if type(value)is not dict:return value
    if 'container'in value:
        items=[_materialize(x,np,events)for x in value['items']]
        return tuple(items)if value['container']=='tuple'else items
    if 'np_scalar'in value:return np.dtype(value['np_scalar']).type(value['value'])
    if 'width_array'in value:return np.array(value['width_array'],dtype=np.int64)
    if 'sequence_probe'in value:return SequenceProbe(tuple(value['sequence_probe']),events)
    if 'index_probe'in value:return IndexProbe(value['index_probe'],events)
    if 'bool_probe'in value:return BoolProbe(value['bool_probe'],events)
    assert value=={'literal_dict':True};return {}


def fixture(spec,np):
    """Future bounded NumPy materialization only; never call crop or another API."""
    bound=source_safe(spec);events=[];dtype=spec['dtype'];count=bound['logical_input_elements']
    dt=np.dtype([('x','i2'),('y','i4')])if dtype=='packed'else np.dtype('i2',metadata={'crop':'tag'})if dtype=='tagged_i16'else np.dtype('>i2')if dtype=='big_endian_i16'else np.dtype(dtype)
    if dtype=='packed':
        array=np.zeros(count,dtype=dt);array['x']=np.arange(count,dtype=np.int16);array['y']=np.arange(count,dtype=np.int32)
    elif dtype in('S3','U2','V3','object'):
        array=np.zeros(count,dtype=dt)
        for i in range(count):array[i]=('x','y',None,3)[i%4]if dtype=='object'else b'xy'if dtype in('S3','V3')else'\u00e9'
        if spec['input_kind']=='distinct_objects':
            for i in range(count):array[i]=''.join(('crop','-payload-',str(i%2),'-owned'))
    else:array=np.arange(count,dtype=np.int64).astype(dt)
    array=array.reshape(tuple(spec['shape']));buffers={'owner':array};layout=spec['layout']
    if layout=='F':array=np.asfortranarray(array)
    elif layout=='transpose':array=array.T
    elif layout=='permute3':array=array.transpose(2,0,1)
    elif layout=='reverse_rows':array=array[::-1]
    elif layout=='reverse_columns':array=array[:,::-1]
    elif layout=='reverse_all':array=array[tuple(slice(None,None,-1)for _ in array.shape)]
    elif layout=='step_all':
        for axis in range(array.ndim):array=np.repeat(array,2,axis=axis)
        buffers['expanded']=array;array=array[tuple(slice(None,None,2)for _ in array.shape)]
    if spec['readonly']:array.flags.writeable=False
    buffers['input_array']=array;kind=spec['input_kind'];ar=array
    if kind=='list':ar=array.tolist()
    elif kind=='tuple':ar=tuple(tuple(row)for row in array.tolist())
    elif kind=='python_scalar':ar=3
    elif kind=='numpy_scalar':ar=np.int16(3)
    elif kind=='array_provider':ar=ArrayProvider(array,events)
    elif kind=='array_provider_error':ar=ArrayProvider(array,events,True)
    elif kind=='interface_provider':ar=InterfaceProvider(array,events)
    elif kind=='memoryview':ar=memoryview(array)
    elif kind=='subclass':
        class TraceArray(np.ndarray):
            def __array_finalize__(self,parent):self.events=events
            def __getitem__(self,index):
                events.append({'event':'subclass.getitem','index_type':type(index).__name__})
                return np.ndarray.__getitem__(self,index)
        ar=array.view(TraceArray);buffers['subclass']=ar
    width=_materialize(spec['width'],np,events);copy=_materialize(spec['copy'],np,events)
    if isinstance(width,np.ndarray):buffers['width_array']=width
    events.clear()
    return (ar,width),{'copy':copy,'order':spec['order']},events,buffers
