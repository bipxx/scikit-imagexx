"""Bounded source-first recipes; no target imports or implicit execution."""
from crop_contract import DTYPES,INPUTS,LAYOUTS,fixture as crop_fixture,source_safe as crop_safe
API='util.slice_along_axes'
SIGNATURE='(image, slices, axes=None, copy=False)'
SOURCE_SHA256='1a75717aa0236c350b51cdbb79d8787411d7151bec2dcd335a474823b2be1bac'
MAX_ELEMENTS=256
MAX_PHYSICAL_ELEMENTS=4096


def seq(*items,kind='list'):return dict(container=kind,items=list(items))
def pair(a,b):return seq(a,b,kind='tuple')


def cases():
    rows=[]
    def add(name,**changes):
        row=dict(id='slice_along_axes/'+name,api=API,input_kind='ndarray',shape=[4,5],dtype='int16',layout='C',readonly=False,slices=seq(pair(1,3)),axes=None,copy=False,mutation=False,policy={})
        row.update(changes);rows.append(row)
    add('ordinary-default')
    add('two-axis',slices=seq(pair(1,3),pair(0,4)))
    add('explicit-axis1',slices=seq(pair(1,4)),axes=seq(1))
    add('axis-order',slices=seq(pair(0,3),pair(1,3)),axes=seq(1,0))
    add('full-view',slices=seq(pair(0,4),pair(0,5)))
    add('empty-result',slices=seq(pair(1,1)))
    add('upper-equality',slices=seq(pair(4,4)))
    for name,value in [('list',seq()),('tuple',seq(kind='tuple')),('none',None),('false',False),('zero',0),('string','')]:
        add('empty-slices-'+name,slices=value,axes=2**70,copy={'bool_probe':'raise'})
    add('empty-array-slices',slices={'array':'int64','items':[]})
    for name,value in [('true',True),('int',1),('string','ab'),('flat-pair',seq(0,2)),('pair-short',seq(seq(1))),('pair-long',seq(seq(0,1,2))),('pair-none',seq(None)),('ambiguous-array',{'array':'int64','items':[[0,2],[1,3]]})]:add('bad-slices-'+name,slices=value)
    for name,axes in [('tuple',seq(0,kind='tuple')),('range',{'axis_range':[0,1]}),('negative',seq(-1)),('outside',seq(2)),('huge',seq(2**70)),('bool-false',seq(False)),('bool-true',seq(True)),('scalar-int',0),('scalar-numpy',{'np_scalar':'int64','value':0}),('scalar-float',0.),('string','0'),('empty',seq()),('float-index',seq(0.)),('unhashable',seq(seq(0))),('numpy-order',{'array':'int64','items':[0]}),('numpy-float',{'array':'float64','items':[0.]})]:add('axes-'+name,axes=axes)
    for name,axes in [('duplicate',seq(0,0)),('duplicate-negative',seq(-1,-1)),('duplicate-bool',seq(False,0))]:add('axes-'+name,slices=seq(pair(0,2),pair(0,2)),axes=axes)
    add('too-many-default',slices=seq(pair(0,1),pair(0,1),pair(0,1)))
    for name,a,b in [('negative-start',-1,4),('negative-stop',0,-1),('negative-both',-3,-1),('overnegative',-9,-5),('wrapped-order',-1,0),('ordering',3,1),('stop-outside',0,6),('huge-stop',0,2**70),('huge-start',2**70,2**70+1),('huge-negative',-(2**70),-(2**70)+2),('none-start',None,2),('none-stop',0,None),('float',.5,2.5),('numpy-int',{'np_scalar':'int64','value':1},{'np_scalar':'int64','value':3}),('numpy-uint',{'np_scalar':'uint64','value':1},{'np_scalar':'uint64','value':3})]:add('bounds-'+name,slices=seq(pair(a,b)))
    add('zero-axis',shape=[0,5],slices=seq(pair(0,0)))
    add('zero-axis-negative',shape=[0,5],slices=seq(pair(-1,0)))
    for policy in ('warn','call','raise'):add('zero-axis-numpy-'+policy,shape=[0,5],slices=seq(pair({'np_scalar':'int64','value':-1},0)),policy={'divide':policy})
    for shape in ([],[7],[2,3,4],[2,3,2,2]):add('rank-'+str(len(shape)),shape=shape,slices=seq()if not shape else seq(pair(0,1)))
    add('zeroD-nonempty',shape=[])
    for layout in LAYOUTS[1:]:
        for copy in (False,True):add('layout-'+layout+'-'+str(copy),layout=layout,shape=[2,3,4]if layout=='permute3'else[4,5],slices=seq(pair(0,1)),copy=copy)
    for copy in (False,True):add('readonly-'+str(copy),readonly=True,copy=copy)
    for dtype in DTYPES:
        if dtype!='int16':add('dtype-'+dtype,dtype=dtype,slices=seq(pair(0,2)))
    for dtype in ('object','packed','complex128','tagged_i16','S3','U2','V3'):add('copy-dtype-'+dtype,dtype=dtype,slices=seq(pair(0,2)),copy=True)
    for kind in INPUTS[1:]+('distinct_objects',):
        shape=[]if kind in('python_scalar','numpy_scalar')else[4,5]
        dtype='object'if kind=='distinct_objects'else'int16'
        add('input-'+kind,input_kind=kind,shape=shape,dtype=dtype,slices=seq(pair(0,1)))
        add('identity-'+kind,input_kind=kind,shape=shape,dtype=dtype,slices=seq(),copy=True)
    for copy in (False,True,'raise'):add('copy-truth-'+str(copy),copy={'bool_probe':copy})
    add('copy-array-ambiguous',copy={'array':'int64','items':[1,0]})
    add('copy-error-after-getitem',slices=seq(pair(.5,2.5)),copy={'bool_probe':'raise'})
    add('identity-none-image',input_kind='python_none',slices=seq(),copy={'bool_probe':'raise'},axes=2**70)
    add('nonempty-none-image',input_kind='python_none')
    add('slice-sequence',slices={'sequence':'slices','items':[pair(0,2)]})
    add('axes-sequence',axes={'sequence':'axes','items':[0]})
    for phase in (None,'compare','mod','index'):
        add('bound-probe-'+str(phase),slices=seq(pair({'bound_probe':-1 if phase=='mod'else 1,'failure':phase},4)))
    for phase in (None,'hash','compare','index'):add('axis-probe-'+str(phase),axes=seq({'axis_probe':0,'failure':phase}))
    add('axis-probe-duplicate',slices=seq(pair(0,2),pair(0,2)),axes=seq({'axis_probe':0,'failure':None},{'axis_probe':0,'failure':None}))
    for result in ('view','owned','list','raise'):
        for copy in (False,True):add('image-probe-'+result+'-'+str(copy),input_kind='image_probe',image_result=result,copy=copy)
    for copy in (False,True):add('mutation-'+str(copy),copy=copy,mutation=True)
    add('mutation-readonly',readonly=True,mutation=True)
    return rows


def _literal(value,depth=0):
    assert depth<=4
    if value is None or type(value)is bool:return
    if type(value)is int:assert abs(value)<=2**70+1;return
    if type(value)is float:assert -64<=value<=64;return
    if type(value)is str:assert value in ('','ab','0');return
    assert type(value)is dict
    if set(value)=={'container','items'}:
        assert value['container']in('list','tuple')and type(value['items'])is list and len(value['items'])<=8
        for item in value['items']:_literal(item,depth+1)
    elif set(value)=={'np_scalar','value'}:
        assert value['np_scalar']in('int64','uint64','float64')and type(value['value'])in(int,float)
        if value['np_scalar']!='float64':assert type(value['value'])is int
        assert (-2**63<=value['value']<2**63)if value['np_scalar']=='int64'else(0<=value['value']<2**64)if value['np_scalar']=='uint64'else(-64<=value['value']<=64)
    elif set(value)=={'array','items'}:
        assert value['array']in('int64','float64')and type(value['items'])is list and len(value['items'])<=8
        if any(type(item)is list for item in value['items']):assert all(type(item)is list and len(item)==len(value['items'][0])for item in value['items'])
        for item in value['items']:
            if type(item)is list:
                assert len(item)<=8
                for element in item:assert type(element)in(int,float)and abs(element)<=64
            else:assert type(item)in(int,float)and abs(item)<=64
    elif set(value)=={'axis_range'}:assert value['axis_range']==[0,1]
    elif set(value)=={'sequence','items'}:
        assert value['sequence']in('axes','slices')and type(value['items'])is list and len(value['items'])<=4
        for item in value['items']:_literal(item,depth+1)
    elif set(value)=={'bound_probe','failure'}:assert type(value['bound_probe'])is int and value['bound_probe']in(-1,1)and value['failure']in(None,'compare','mod','index')
    elif set(value)=={'axis_probe','failure'}:assert type(value['axis_probe'])is int and value['axis_probe']==0 and value['failure']in(None,'hash','compare','index')
    else:assert set(value)=={'bool_probe'}and(type(value['bool_probe'])is bool or type(value['bool_probe'])is str and value['bool_probe']=='raise')


def _base(spec):
    return dict(id='crop/slice-payload',api='util.crop',input_kind='ndarray'if spec['input_kind']in('image_probe','python_none')else spec['input_kind'],shape=spec['shape'],dtype=spec['dtype'],layout=spec['layout'],readonly=spec['readonly'],width=0,copy=False,order='K')


def source_safe(spec):
    assert type(spec)is dict
    fields={'id','api','input_kind','shape','dtype','layout','readonly','slices','axes','copy','mutation','policy'}
    assert set(spec)==fields|({'image_result'}if spec.get('input_kind')=='image_probe'else set())
    assert spec['api']==API and type(spec['id'])is str and spec['id'].startswith('slice_along_axes/')
    assert spec['input_kind']in INPUTS+('distinct_objects','image_probe','python_none')
    if spec['input_kind']=='image_probe':assert spec['image_result']in('view','owned','list','raise')and spec['shape']==[4,5]and spec['dtype']=='int16'and spec['layout']=='C'
    if spec['input_kind']=='python_none':assert spec['shape']==[4,5]and spec['dtype']=='int16'and spec['layout']=='C'
    bound=crop_safe(_base(spec))
    for key in ('slices','axes','copy'):_literal(spec[key])
    assert type(spec['mutation'])is bool
    assert spec['policy']=={}or spec['policy']in({'divide':'warn'},{'divide':'call'},{'divide':'raise'})
    if spec['mutation']:assert spec['dtype']=='int16'and spec['input_kind']=='ndarray'and spec['shape']==[4,5]and spec['layout']=='C'and spec['slices']==seq(pair(1,3))and spec['axes']is None
    if spec['dtype']=='object':assert bound['physical_input_elements']+2*bound['logical_input_elements']<=512
    return {**bound,'bound_magnitude':2**70+1,'axis_sequence_length_max':8,'basic_step_one_only':True,'source_negative_modulo_zero_is_error_or_NumPy_FP':True,'original_calls':0}


class SequenceProbe:
    def __init__(self,label,items,events):self.label,self.items,self.events=label,tuple(items),events
    def __len__(self):self.events.append({'event':self.label+'.len'});return len(self.items)
    def __getitem__(self,index):self.events.append({'event':self.label+'.getitem','index':index});return self.items[index]
    def __repr__(self):return 'SequenceProbe('+self.label+','+repr(self.items)+')'


class BoundProbe:
    def __init__(self,value,failure,events):self.value,self.failure,self.events=value,failure,events
    def compare(self,other,operation):
        self.events.append({'event':'bound.'+operation,'value':self.value})
        if self.failure=='compare':raise RuntimeError('slice bound comparison refused')
        other=other.value if isinstance(other,BoundProbe)else other
        return self.value<other if operation=='lt'else self.value>other
    def __lt__(self,other):return self.compare(other,'lt')
    def __gt__(self,other):return self.compare(other,'gt')
    def __mod__(self,other):
        self.events.append({'event':'bound.mod','value':self.value,'divisor':other})
        if self.failure=='mod':raise RuntimeError('slice bound modulo refused')
        return self.value%other
    def __index__(self):
        self.events.append({'event':'bound.index','value':self.value})
        if self.failure=='index':raise TypeError('slice bound index refused')
        return self.value
    def __repr__(self):return 'BoundProbe('+repr(self.value)+','+repr(self.failure)+')'


class AxisProbe:
    def __init__(self,value,failure,events):self.value,self.failure,self.events=value,failure,events
    def __hash__(self):
        self.events.append({'event':'axis.hash','value':self.value})
        if self.failure=='hash':raise RuntimeError('slice axis hash refused')
        return hash(self.value)
    def __eq__(self,other):self.events.append({'event':'axis.eq'});return self.value==(other.value if isinstance(other,AxisProbe)else other)
    def compare(self,other,operation):
        self.events.append({'event':'axis.'+operation,'value':self.value})
        if self.failure=='compare':raise RuntimeError('slice axis comparison refused')
        return self.value>=other if operation=='ge'else self.value<other
    def __ge__(self,other):return self.compare(other,'ge')
    def __lt__(self,other):return self.compare(other,'lt')
    def __index__(self):
        self.events.append({'event':'axis.index','value':self.value})
        if self.failure=='index':raise TypeError('slice axis index refused')
        return self.value
    def __repr__(self):return 'AxisProbe('+repr(self.value)+','+repr(self.failure)+')'


class CopyProbe:
    def __init__(self,value,events):self.value,self.events=value,events
    def __bool__(self):
        self.events.append({'event':'copy.bool','value':self.value})
        if self.value=='raise':raise RuntimeError('slice copy truth refused')
        return self.value


class ImageProbe:
    def __init__(self,array,result,events):self.array,self.result,self.events=array,result,events
    @property
    def ndim(self):self.events.append({'event':'image.ndim'});return self.array.ndim
    @property
    def shape(self):self.events.append({'event':'image.shape'});return self.array.shape
    def __getitem__(self,index):
        self.events.append({'event':'image.getitem'})
        if self.result=='raise':raise RuntimeError('slice image indexing refused')
        if self.result=='owned':return self.array.copy(order='F')
        if self.result=='list':return [1,2]
        return self.array[index]


def _materialize(value,np,events):
    if type(value)is not dict:return value
    if 'container'in value:
        items=[_materialize(v,np,events)for v in value['items']];return tuple(items)if value['container']=='tuple'else items
    if 'np_scalar'in value:return np.dtype(value['np_scalar']).type(value['value'])
    if 'array'in value:return np.array(value['items'],dtype=value['array'])
    if 'axis_range'in value:return range(*value['axis_range'])
    if 'sequence'in value:return SequenceProbe(value['sequence'],[_materialize(v,np,events)for v in value['items']],events)
    if 'bound_probe'in value:return BoundProbe(value['bound_probe'],value['failure'],events)
    if 'axis_probe'in value:return AxisProbe(value['axis_probe'],value['failure'],events)
    return CopyProbe(value['bool_probe'],events)


def fixture(spec,np):
    source_safe(spec);args,_,events,buffers=crop_fixture(_base(spec),np);image=args[0]
    if spec['input_kind']=='python_none':image=None
    if spec['input_kind']=='image_probe':image=ImageProbe(image,spec['image_result'],events)
    if spec['input_kind']=='subclass':
        # Extend the admitted crop subclass with an observable copy phase.
        parent=image
        class TraceSliceArray(type(parent)):
            def copy(self,*args,**kwargs):events.append({'event':'subclass.copy','args':list(args),'kwargs':kwargs});return np.ndarray.copy(self,*args,**kwargs)
        image=parent.view(TraceSliceArray);buffers['subclass']=image
    limits=_materialize(spec['slices'],np,events);axes=_materialize(spec['axes'],np,events);copy=_materialize(spec['copy'],np,events)
    events.clear();return (image,limits),dict(axes=axes,copy=copy),events,buffers
