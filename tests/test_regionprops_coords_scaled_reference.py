"""Real coords_scaled fixtures/controller; prospective intakes never call originals."""
import hashlib
import json
import math
from pathlib import Path
import warnings
import pytest
from regionprops_state_observer import encode, scientific_profile, reference_runtime, SOURCE_SHA, UTILS_SHA
ROOT=Path(__file__).parents[1]
PROPOSAL=ROOT/'coverage/contracts/regionprops-coords-scaled-prospective-v1.json'
REFERENCE=ROOT/'coverage/contracts/regionprops-coords-scaled-reference-v1.json'
CASES=json.loads(PROPOSAL.read_bytes())['cases']
BY_ID={s['id']:s for s in CASES}


def source_safe(spec):
    # Exact finite recipes are the external boundary; internal steps consume them.
    assert type(spec)is dict and spec==BY_ID[spec['id']]
    shape=spec['shape'];ndim=len(shape)
    assert shape in ([4,5],[4,4,4])and math.prod(shape)<=64
    assert spec['dtype']in ('int16','int32','int64','uint8','uint16')and spec['order']in ('C','F')and type(spec['cache'])is bool
    assert spec['selected_region_index']==0 and spec['property']=='coords_scaled'
    assert spec['spacing_kind']in ('default','tuple-f64','scalar-f64','tuple-int','array-f32')
    assert spec['offset_kind']in ('default','tuple-int','array-f32')
    if spec['spacing_kind']=='default':assert spec['spacing']is None
    elif spec['spacing_kind']=='scalar-f64':assert type(spec['spacing'])is float
    else:assert type(spec['spacing'])is list and len(spec['spacing'])==ndim
    if spec['offset_kind']=='default':assert spec['offset']is None
    else:assert type(spec['offset'])is list and len(spec['offset'])==ndim and all(type(v)in (int,float)and math.isfinite(v)and abs(v)<=4 for v in spec['offset'])
    for point in spec['label_1_points']+[spec['label_2_point']]:
        assert len(point)==ndim and all(type(v)is int and 0<=v<n for v,n in zip(point,shape))
    primed=False;reads=items=0
    for action,value in spec['sequence']:
        if action in ('attr','item'):
            assert value=='coords_scaled';reads+=1;items+=action=='item'
        elif action=='prime':assert value=='image'and spec['cache'];primed=True
        elif action=='set-label-pixel':
            point,pixel=value;assert len(point)==ndim and all(type(v)is int and 0<=v<n for v,n in zip(point,shape))and pixel==0
        elif action=='set-primed-mask':assert primed and value==[[0,0],False]and ndim==2
        else:assert action=='set-public-slice'and primed and value==[[2,4],[2,4]]and ndim==2
    assert reads==3 and items==1
    for value in ([]if spec['spacing']is None else spec['spacing']if type(spec['spacing'])is list else[spec['spacing']]):
        assert type(value)in (int,float)and math.isfinite(value)and abs(value)<=3
    return {'max_input_elements':64,'max_label':2,'regions':2,'max_population':4,'max_output_elements':12,'coords_scaled_reads':3,'getitem_dispatches':1}


def fixture(spec,np):
    bounds=source_safe(spec)
    image=np.zeros(tuple(spec['shape']),dtype=spec['dtype'],order=spec['order'])
    for point in spec['label_1_points']:image[tuple(point)]=1
    image[tuple(spec['label_2_point'])]=2
    def parameter(value,kind):
        if kind=='default':return None
        if kind=='scalar-f64':return value
        if kind=='array-f32':return np.array(value,dtype=np.float32)
        return tuple(value)
    options=dict(cache=spec['cache'],spacing=parameter(spec['spacing'],spec['spacing_kind']),offset=parameter(spec['offset'],spec['offset_kind']))
    buffers={'labels':image}
    for key in ('spacing','offset'):
        if isinstance(options[key],np.ndarray):buffers[key]=options[key]
    return image,options,buffers,bounds


def observe(call,spec,np):
    image,options,buffers,bounds=fixture(spec,np)
    before=encode({'labels':image,'options':options},np);expected_image=image.copy()
    descriptors={k:encode(a,np)for k,a in buffers.items()};pointers={k:a.ctypes.data for k,a in buffers.items()}
    old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    phase=['constructor'];events=[];steps=[];arrays=[];regions=None;error=None;region_type=None
    primed=None;expected_primed=None;primed_pointer=None;primed_descriptor=None;expected_slice=None
    def handler(message,flags):events.append({'phase':phase[0],'message':message,'flags':flags})
    with np.errstate(**spec['FP_policy'],call=handler):
        policy=np.geterr().copy()
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always')
            try:regions=call(image,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
        constructor_warnings=[[type(x.message).__module__,type(x.message).__name__,str(x.message)]for x in notices]
        if error is None:
            assert type(regions)is list and len(regions)==bounds['regions']
            region=regions[spec['selected_region_index']];region_type=[type(region).__module__,type(region).__qualname__];expected_slice=region.slice
            for index,(action,argument)in enumerate(spec['sequence']):
                phase[0]=index;value=None;failure=None
                with warnings.catch_warnings(record=True)as notices:
                    warnings.simplefilter('always')
                    try:
                        if action=='attr':value=getattr(region,argument)
                        elif action=='item':value=region[argument]
                        elif action=='prime':
                            primed=getattr(region,argument);assert type(primed)is np.ndarray
                            expected_primed=primed.copy();primed_pointer=primed.ctypes.data;primed_descriptor=encode(primed,np)
                            value={'primed':'image','type':[type(primed).__module__,type(primed).__qualname__],'shape':list(primed.shape),'dtype':primed.dtype.str}
                        elif action=='set-label-pixel':
                            point,pixel=argument;image[tuple(point)]=pixel;expected_image[tuple(point)]=pixel
                        elif action=='set-primed-mask':
                            point,pixel=argument;primed[tuple(point)]=pixel;expected_primed[tuple(point)]=pixel
                        else:
                            assert action=='set-public-slice';expected_slice=tuple(slice(a,b)for a,b in argument);region.slice=expected_slice
                    except Exception as exc:failure=[type(exc).__module__,type(exc).__name__,str(exc)]
                if isinstance(value,np.ndarray):
                    assert type(value)is np.ndarray and value.ndim==2 and value.shape[1]==image.ndim and value.size<=bounds['max_output_elements']and not value.dtype.hasobject and value.dtype.itemsize<=8
                    arrays.append((index,value))
                steps.append({'action':action,'argument':argument,'result':encode(value,np),'error':failure,'warnings':[[type(x.message).__module__,type(x.message).__name__,str(x.message)]for x in notices]})
            assert region.slice==expected_slice
        stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    labels_after=encode(image,np);label_descriptor={k:v for k,v in labels_after.items()if k not in ('raw_sha256','raw_zlib_base64')}
    old_label_descriptor={k:v for k,v in descriptors['labels'].items()if k not in ('raw_sha256','raw_zlib_base64')}
    checks={'input_matches_declared_writes':bool(np.array_equal(image,expected_image)),'label_descriptor_unchanged':label_descriptor==old_label_descriptor,'all_input_pointers_stable':all(a.ctypes.data==pointers[k]for k,a in buffers.items()),'parameter_buffers_unchanged':all(encode(a,np)==descriptors[k]for k,a in buffers.items()if k!='labels'),'policy_unchanged':stable,'FP_state_restored':np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer}
    if primed is not None:
        after=encode(primed,np);checks['primed_mask_matches_declared_writes']=bool(np.array_equal(primed,expected_primed));checks['primed_mask_descriptor_pointer_stable']=primed.ctypes.data==primed_pointer and {k:v for k,v in after.items()if k not in ('raw_sha256','raw_zlib_base64')}=={k:v for k,v in primed_descriptor.items()if k not in ('raw_sha256','raw_zlib_base64')}
    assert all(checks.values())
    row={'id':spec['id'],'api':'measure.regionprops','property':'coords_scaled','input_before':before,'input_after':encode({'labels':image,'options':options},np),'constructor_error':error,'constructor_warnings':constructor_warnings,'returned_count':None if regions is None else len(regions),'region_type':region_type,'steps':steps,'FP_events':events,'output_pair_aliases':[{'first':i,'second':j,'same_object':a is b,'shares_memory':bool(np.shares_memory(a,b)),'same_base':a.base is b.base}for i,a in arrays for j,b in arrays],'output_input_aliases':[{'step':i,'input':k,'same_object':a is b,'shares_memory':bool(np.shares_memory(a,b))}for i,a in arrays for k,b in buffers.items()],'output_primed_mask_aliases':[]if primed is None else[{'step':i,'same_object':a is primed,'shares_memory':bool(np.shares_memory(a,primed))}for i,a in arrays],'post_sequence_outputs':[{'step':i,'value':encode(a,np)}for i,a in arrays],'checks':checks}
    json.dumps(row,allow_nan=False);return row


def test_coords_scaled_prospective_scope_and_source():
    data=json.loads(REFERENCE.read_bytes());assert data['property']=='coords_scaled'and data['prospective_contract_sha256']==hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
    assert len(CASES)==len(BY_ID)==12
    for spec in CASES:source_safe(spec)
    for name,pin in data['source_pins'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert data['budget']=={'top_level_regionprops_calls':24,'coords_scaled_fget_reads':72,'explicit_image_primes':6,'canonical_getitem_dispatches':24,'internal_RegionProperties_init_calls':48,'image_wrapper_reads_including_dependencies':78}


@pytest.mark.parametrize('spec',CASES,ids=lambda s:s['id'])
def test_coords_scaled_controller_intake_originals_refused(spec):
    np=scientific_profile();calls=[]
    def refused(image,**options):calls.append(1);raise RuntimeError('coords-scaled-original-refused')
    row=observe(refused,spec,np)
    assert calls==[1]and row['constructor_error']==['builtins','RuntimeError','coords-scaled-original-refused']and row['steps']==[]and row['returned_count']is None
    assert row['input_before']['labels']['shape']==spec['shape']and row['input_before']['labels']['dtype']==np.dtype(spec['dtype']).str
    assert row['input_before']['labels']['f'if spec['order']=='F'else'c']
    assert row['input_before']==row['input_after']and all(row['checks'].values())



def test_coords_scaled_saved_sequences_and_accounting():
    data=json.loads(REFERENCE.read_bytes());assert set(data['original_observations'])=={'1.24.4','2.5.4'}
    assert data['original_observations']['1.24.4']==data['original_observations']['2.5.4']
    for rows in data['original_observations'].values():
        assert [row['id']for row in rows]==[spec['id']for spec in CASES]
        for spec,row in zip(CASES,rows):
            assert row['constructor_error']is None and row['constructor_warnings']==[]and row['returned_count']==2
            assert len(row['steps'])==len(spec['sequence'])and row['FP_events']==[]and all(row['checks'].values())
            assert [(step['action'],step['argument'])for step in row['steps']]==[tuple(x)for x in spec['sequence']]
            assert all(step['error']is None and step['warnings']==[]for step in row['steps'])
            counts=row['call_accounting']['actual_source_entries'];primes=sum(a=='prime'for a,v in spec['sequence'])
            assert counts['regionprops']==1 and counts['RegionProperties.__init__']==2 and counts['coords_scaled']==3 and counts['getitem']==1 and counts['image_wrapper']==3+primes
            assert counts['image_compute']==(1 if spec['cache']else 3)
            reads=[step['result']for step in row['steps']if step['action']in ('attr','item')]
            assert len(reads)==3 and reads[0]==reads[1]==reads[2]
            dtype='<i8'if spec['id'].endswith('/integer-spacing3d')else'<f8'
            for value in reads:
                n,ndim=value['shape'];assert ndim==len(spec['shape'])and n<=4 and value['dtype']==dtype and value['dtype_metadata']is None
                assert value['own']and value['base_type']=='NoneType'and value['writeable']and value['aligned']and value['f']
                assert value['strides']==([8,n*8]if n else[0,0])and value['c']==(n==0)
            for pair in row['output_pair_aliases']:
                diagonal=pair['first']==pair['second'];assert pair['same_object']==diagonal and pair['same_base']
                assert pair['shares_memory']==(diagonal and reads[0]['shape'][0]!=0)
            assert all(not x['same_object']and not x['shares_memory']for x in row['output_input_aliases']+row['output_primed_mask_aliases'])
            assert [x['value']for x in row['post_sequence_outputs']]==reads


def test_coords_scaled_saved_payload_words_and_declared_mutations():
    import base64
    import struct
    import zlib
    data=json.loads(REFERENCE.read_bytes());formats={('i',2):'h',('i',4):'i',('i',8):'q',('u',1):'B',('u',2):'H',('f',4):'f',('f',8):'d'}
    def decode(value):
        raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']));assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
        dtype=value['dtype'];width=int(dtype[2:]);size=math.prod(value['shape']);assert len(raw)==size*width
        return struct.unpack('<'+formats[(dtype[1],width)]*size,raw)
    def walk(value):
        if isinstance(value,dict):
            if 'raw_zlib_base64'in value:decode(value)
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(data['original_observations'])
    for rows in data['original_observations'].values():
        for spec,row in zip(CASES,rows):
            before=list(decode(row['input_before']['labels']));expected=before.copy();shape=spec['shape']
            for action,arg in spec['sequence']:
                if action=='set-label-pixel':
                    point,pixel=arg;flat=0
                    for index,extent in zip(point,shape):flat=flat*extent+index
                    expected[flat]=pixel
            assert list(decode(row['input_after']['labels']))==expected
            assert row['input_before']['options']==row['input_after']['options']
            for step in row['steps']:
                if step['action']in ('attr','item'):assert all(math.isfinite(v)and abs(v)<=20 for v in decode(step['result']))
