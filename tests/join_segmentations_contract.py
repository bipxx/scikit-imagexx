"""Prospective bounded join inputs. No target imports or implicit execution."""
API='segmentation.join_segmentations'
SIGNATURE='(s1, s2, return_mapping: bool = False)'
MAX_ELEMENTS=192
INTEGER_DTYPES=('int8','uint8','int16','uint16','int32','uint32','int64','uint64')


def cases():
    rows=[]
    def add(name,**extra):
        row=dict(id='join_segmentations/'+name,api=API,shape=[2,4],left_dtype='int16',right_dtype='int16',
            left=[0,0,1,1,2,2,3,3],right=[0,1,1,0,0,1,1,0],layout='C',flag=True,map_protocols=False)
        row.update(extra);rows.append(row)
    for dtype in INTEGER_DTYPES:
        add(dtype+'-labels',left_dtype=dtype,right_dtype=dtype,flag=False,omit_flag=dtype=='int8')
        add(dtype+'-maps',left_dtype=dtype,right_dtype=dtype,map_protocols=dtype=='int16')
    add('all-zero',left=[0]*8,right=[0]*8)
    add('no-zero',left=[1,1,3,3,7,7,1,3],right=[2,3,2,3,2,3,2,3])
    add('sparse-small',left=[0,7,7,0,3,3,7,0],right=[0,7,0,7,0,7,0,7])
    add('shape-mismatch',right_shape=[4,2])
    for layout in ('F','reverse','step','transpose','readonly-C','readonly-F'):
        add(layout,layout=layout,left_dtype='int32',right_dtype='int32')
    add('rank-zero',shape=[],left=[2],right=[3])
    add('rank-three',shape=[2,2,2],flag=False,left_dtype='uint32',right_dtype='uint32')
    add('dtype-metadata',metadata=True)
    for name,a,b in [('mixed-u8-i16','uint8','int16'),('mixed-i32-u16','int32','uint16'),('mixed-u64-i64','uint64','int64')]:
        add(name,left_dtype=a,right_dtype=b)
    for side in ('left','right'):
        add('negative-'+side,**{side:[-1,0,1,1,2,2,3,3]})
        add('float-'+side,**{side+'_dtype':'float64'})
        add('bool-'+side,**{side+'_dtype':'bool'})
    add('empty',shape=[0,3],left=[],right=[])
    add('plain-list-left',plain_list=True)
    for name,flag in [('none',None),('zero',0),('one',1),('empty-string',''),('truthy-string','yes')]:add('flag-'+name,flag=flag)
    add('flag-ambiguous',flag_array=True)
    add('same-object',alias='same',right=[0,0,1,1,2,2,3,3])
    add('overlapping-inputs',alias='overlap')
    add('broadcast-readonly',layout='broadcast')
    add('nonnative-endian',left_dtype='>i2',right_dtype='>i2')
    return rows


def source_bounds(spec):
    import math
    assert spec['api']==API and spec['layout']in('C','F','reverse','step','transpose','readonly-C','readonly-F','broadcast')
    assert len(spec['shape'])<=3 and all(type(n)is int and 0<=n<=8 for n in spec['shape'])
    n=math.prod(spec['shape']);assert n<=MAX_ELEMENTS
    right_shape=spec.get('right_shape',spec['shape']);assert len(right_shape)<=3 and all(type(v)is int and 0<=v<=8 for v in right_shape)and math.prod(right_shape)==n
    assert len(spec['left'])==len(spec['right'])==n
    assert all(type(v)is int and -1<=v<=7 for side in('left','right')for v in spec[side])
    assert spec['left_dtype']in(*INTEGER_DTYPES,'float64','bool','>i2') and spec['right_dtype']in(*INTEGER_DTYPES,'float64','bool','>i2')
    assert spec['flag']is None or(type(spec['flag'])in(bool,int,str)and spec['flag']in(False,True,0,1,'','yes'))
    assert type(spec['map_protocols'])is bool
    for key in('omit_flag','metadata','plain_list','flag_array'):
        if key in spec:assert type(spec[key])is bool
    if spec['layout']=='broadcast' or spec.get('alias')=='overlap':assert spec['shape']==[2,4]
    if spec['map_protocols']:assert spec['id']=='join_segmentations/int16-maps'and spec['layout']=='C'
    assert spec.get('alias')in(None,'same','overlap')
    # Negative/type/empty/shape/readonly/endian failures are recorded as they
    # occur. No mathematical join is used as an output oracle.
    return dict(input_elements_each=n,source_label_upper=7,source_factor_upper=8,
                source_encoded_upper=63,max_map_entries=64,max_dense_map_elements=65,
                input_output_Cython_lengths_equal=True,map_key_value_lengths_equal=True)


def fixture(spec,np):
    bounds=source_bounds(spec)
    def make(side):
        dtype=np.dtype(spec[side+'_dtype'],metadata={'join':side})if spec.get('metadata')else np.dtype(spec[side+'_dtype'])
        return np.array(spec[side],dtype=dtype).reshape(spec.get(side+'_shape',spec['shape']))
    left,right=make('left'),make('right');buffers={}
    layout=spec['layout']
    def arrange(array,side):
        if layout in('F','readonly-F'):array=np.asfortranarray(array)
        elif layout=='reverse':array=array[...,::-1]
        elif layout=='step':
            storage=np.repeat(array,2,axis=-1);buffers[side+'_storage']=storage;array=storage[...,::2]
        elif layout=='transpose':array=array.T
        elif layout=='broadcast':array=np.broadcast_to(np.array([0,1,2,3],dtype=array.dtype),(2,4))
        if layout.startswith('readonly'):array.flags.writeable=False
        return array
    left,right=arrange(left,'left'),arrange(right,'right')
    if spec.get('alias')=='same':right=left
    elif spec.get('alias')=='overlap':
        storage=np.array([0,1,2,3,0,1,2,3,0,1],dtype=np.int16).reshape(2,5)
        buffers['shared_storage']=storage;left,right=storage[:,:4],storage[:,1:]
    buffers.update(left=left,right=right)
    arguments=(left.tolist()if spec.get('plain_list')else left,right)
    flag=spec['flag']
    if spec.get('flag_array'):flag=np.array([1,0],dtype=np.int64);buffers['flag']=flag
    return arguments,{}if spec.get('omit_flag')else{'return_mapping':flag},buffers,bounds
