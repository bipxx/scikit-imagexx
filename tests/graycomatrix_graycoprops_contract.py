"""Bounded source-first GLCM recipes; factories need explicit NumPy, target requests refuse."""
import copy
import math

PROPERTIES=('contrast','dissimilarity','homogeneity','energy','correlation','ASM','mean','variance','std','entropy')
INTEGER_DTYPES=('uint8','int8','uint16','int16','uint32','int32','uint64','int64')
DTYPE_BYTES={'bool':1,'uint8':1,'int8':1,'uint16':2,'int16':2,'uint32':4,'int32':4,'uint64':8,'int64':8,'float16':2,'float32':4,'float64':8,'complex64':8,'complex128':16,'object':8,'U1':4,'>u2':2,'>f8':8}


def recipes():
 rows=[]
 def matrix(name,**changes):
  value=dict(id='graycomatrix/'+name,api='feature.graycomatrix',shape=[3,4],dtype='uint8',layout='C',levels=4,distances=[1.0],angles=[0.0],symmetric=False,normed=False,pattern='ordinary',metadata=False,readonly=False,input_kind='array',source_partition='ordinary integer counts',prospective_observation='UNOBSERVED')
  value.update(changes);rows.append(value)
 for dtype in INTEGER_DTYPES:matrix(dtype,dtype=dtype)
 for dtype in ('bool','float16','float32','float64','complex64','complex128','object','U1'):matrix('type-'+dtype,dtype=dtype,source_partition='validation/fused dtype error or cast protocol, outcome unobserved')
 matrix('default-u8',shape=[1,1],levels=None)
 matrix('default-i8',shape=[1,1],dtype='int8',levels=None)
 matrix('default-normalized',shape=[1,1],levels=None,normed=True)
 for layout in ('F','transpose','reverse','zero'):matrix('layout-'+layout,layout=layout)
 matrix('readonly',readonly=True,source_partition='nonconst Cython buffer request, outcome unobserved')
 matrix('metadata',metadata=True)
 matrix('swapped-u16',dtype='>u2',source_partition='non-native-endian fused buffer, outcome unobserved')
 matrix('image-list',input_kind='list')
 for symmetric,normed in ((True,False),(False,True),(True,True)):matrix('flags-'+str(int(symmetric))+str(int(normed)),symmetric=symmetric,normed=normed)
 for name,distances,angles in [('zero',[0.0],[0.0]),('negative',[-1.0],[0.0]),('half',[0.5,1.5],[0.0]),('negative-half',[-0.5,-1.5],[0.0]),('near-half',[0.49999999999999994,0.5000000000000001],[0.0]),('axes',[1.0],[0.0,math.pi/2,math.pi,-math.pi/2]),('multi',[1.0,2.0],[math.pi/4,-math.pi/4])]:matrix('offset-'+name,distances=distances,angles=angles)
 for name,changes in [('rank1',dict(shape=[4])),('rank3',dict(shape=[1,3,4])),('empty',dict(shape=[0,4])),('negative-image',dict(dtype='int16',pattern='negative')),('missing-levels',dict(dtype='uint16',levels=None)),('zero-levels',dict(levels=0)),('negative-levels',dict(levels=-1)),('float-levels',dict(levels=4.5)),('small-levels',dict(levels=2)),('distance-scalar',dict(distance_shape=[])),('angle-rank2',dict(angle_shape=[1,1])),('distance-empty',dict(distances=[])),('angle-empty',dict(angles=[]))]:matrix('error-'+name,source_partition='source validation/error ordering, outcome unobserved',**changes)
 def props(name,**changes):
  value=dict(id='graycoprops/'+name,api='feature.graycoprops',shape=[3,3,2,2],dtype='uint32',layout='C',prop='contrast',pattern='asymmetric',metadata=False,readonly=False,input_kind='array',source_partition='normalized property reduction',prospective_observation='UNOBSERVED')
  value.update(changes);rows.append(value)
 for prop in PROPERTIES:props(prop,prop=prop)
 for prop in ('contrast','correlation','ASM','entropy'):props('zero-'+prop,prop=prop,pattern='zero')
 for prop in ('correlation','variance','std','entropy'):props('diagonal-'+prop,prop=prop,pattern='diagonal')
 for dtype in ('float16','float32','float64','complex128','object','>f8'):props('type-'+dtype,dtype=dtype,source_partition='astypefloat64 conversion/warning/reduction, outcome unobserved')
 for layout in ('F','transpose','reverse','zero'):props('layout-'+layout,layout=layout,prop='correlation')
 props('readonly',readonly=True);props('metadata',dtype='float64',metadata=True)
 for pattern in ('negative','nan','inf'):
  for prop in ('entropy','correlation'):props(pattern+'-'+prop,pattern=pattern,dtype='float64',prop=prop,source_partition='nonfinite/negative normalize+log/sqrt warning policy, outcome unobserved')
 for name,changes in [('rank3',dict(shape=[3,3,2])),('nonsquare',dict(shape=[2,3,1,1])),('empty-levels',dict(shape=[0,0,1,1])),('empty-distance',dict(shape=[3,3,0,1])),('empty-angle',dict(shape=[3,3,1,0])),('list',dict(input_kind='list')),('invalid-property',dict(prop='CONTRAST')),('none-property',dict(prop=None))]:props('error-'+name,source_partition='source validation/property error ordering, outcome unobserved',**changes)
 return rows


def bounds(spec):
 shape=spec['shape'];pixels=math.prod(shape);width=DTYPE_BYTES[spec['dtype']]
 if spec['api']=='feature.graycomatrix':
  levels=256 if spec['levels']is None and spec['dtype']in('uint8','int8')else 4 if not isinstance(spec['levels'],int)or spec['levels']<=0 else spec['levels']
  d,a=len(spec['distances']),len(spec['angles']);cells=levels*levels*d*a
  return dict(input_elements=pixels,physical_elements=pixels,output_cells=cells,pair_visit_ceiling=pixels*d*a,offset_abs_ceiling=5,named_array_allowance_bytes=cells*32+pixels*max(width,8)*3+16*(d+a)+8*d*a+4096,work_unit_ceiling=pixels*d*a+cells*6+2*d*a)
 levels=shape[0]if shape else 1;planes=math.prod(shape[2:])if len(shape)==4 else 1
 return dict(input_elements=pixels,physical_elements=pixels,output_cells=planes,pair_visit_ceiling=0,offset_abs_ceiling=0,named_array_allowance_bytes=pixels*128+levels*levels*32+planes*128+4096,work_unit_ceiling=pixels*32+levels*levels*4)


def validate_recipe(spec):
 originals={r['id']:r for r in recipes()}
 assert spec['id']in originals and spec==originals[spec['id']], 'Not a frozen source-safe recipe'
 assert len(spec['shape'])in(1,2,3,4)and all(type(x)is int and 0<=x<=4 for x in spec['shape'])
 b=bounds(spec);assert b['input_elements']<=36 and b['output_cells']<=65536 and b['pair_visit_ceiling']<=144
 assert b['named_array_allowance_bytes']<=2200000 and b['work_unit_ceiling']<=400000
 if spec['api']=='feature.graycomatrix':
  assert all(math.isfinite(x)and abs(x)<=4 for x in spec['distances']+spec['angles'])
  assert len(spec['distances'])<=3 and len(spec['angles'])<=4
 return b


def make_inputs(spec,np):
 """Metadata/factory only; no target/helper algorithm. Actual NumPy use is a later gate."""
 validate_recipe(spec);shape=spec['shape'];size=math.prod(shape)
 values=[(i*3+i//4)%4 if spec['api']=='feature.graycomatrix'else (i*5+i//3)%7 for i in range(size)]
 if spec['pattern']=='zero':values=[0]*size
 elif spec['pattern']=='diagonal':
  values=[1 if (i//math.prod(shape[2:]))//shape[1]==(i//math.prod(shape[2:]))%shape[1]else 0 for i in range(size)]
 elif values and spec['pattern']in('negative','nan','inf'):values[0]={'negative':-1,'nan':float('nan'),'inf':float('inf')}[spec['pattern']]
 dtype=np.dtype(spec['dtype'],metadata={'source':'GLCM-contract'})if spec['metadata']else np.dtype(spec['dtype'])
 array=np.array(values,dtype=dtype).reshape(tuple(shape));buffers=[array]
 if spec['layout']=='F':array=array.copy(order='F')
 elif spec['layout']=='transpose':array=array.transpose().copy().transpose()
 elif spec['layout']=='reverse':array=array[(slice(None),)*(len(shape)-1)+(slice(None,None,-1),)]
 elif spec['layout']=='zero':array=np.broadcast_to(array[(slice(0,1),)+(slice(None),)*(len(shape)-1)],tuple(shape))
 if spec['readonly']:array.setflags(write=False)
 buffers.append(array);value=array.tolist()if spec['input_kind']=='list'else array
 if spec['api']=='feature.graycomatrix':
  distance=np.array(spec['distances'],dtype='float64');angle=np.array(spec['angles'],dtype='float64')
  if 'distance_shape'in spec:distance=distance.reshape(tuple(spec['distance_shape']))
  if 'angle_shape'in spec:angle=angle.reshape(tuple(spec['angle_shape']))
  buffers.extend([distance,angle]);return (value,distance,angle),dict(levels=spec['levels'],symmetric=spec['symmetric'],normed=spec['normed']),buffers
 return (value,),dict(prop=spec['prop']),buffers


def refuse_original_request(spec,resolve):
 validate_recipe(spec)
 raise RuntimeError('UNBOUND source-first packet: no original/native/helper body authority')
