"""Source-bounded prospective IGG cases; no original/native/helper authority."""
from copy import deepcopy
from math import prod,isfinite
API='segmentation.inverse_gaussian_gradient'
SIGNATURE='(image, alpha=100.0, sigma=5.0)'
MAX_INPUT=64
MAX_OUTPUT=128
MAX_RADIUS=20
MAX_WORK=65536
MAX_NUMERICAL_BYTES=131072

def recipes():
 rows=[]
 def add(name,**kw):
  d=dict(id=name,shape=[3,4],dtype='float32',layout='C',pattern='ramp',alpha=100.0,alpha_kind='python',sigma=1.0,sigma_kind='python',metadata=False,readonly=False,provider=None)
  d.update(kw);rows.append(d)
 for dtype in('bool','uint8','uint16','uint32','uint64','int8','int16','int32','int64','float16','float32','float64','complex64','complex128'):add('dtype-'+dtype,dtype=dtype)
 for layout in('F','reverse','stride2','transpose'):add('layout-'+layout,layout=layout)
 for pattern in('zeros','constant','step','bright','signedzero','nan','inf','neginf'):add('values-'+pattern,pattern=pattern)
 for name,shape in [('rank0',[]),('1d',[7]),('3d',[3,4,5]),('4d',[2,2,2,2]),('8d',[1]*8),('singleton',[1,1])]:add(name,shape=shape)
 add('default-sigma',sigma='OMITTED')
 add('empty-sigma0',shape=[0,3],sigma=0)
 add('rank0-ignores-bad-sigma',shape=[],sigma='bad')
 for sigma in(0,-1,1e-16,1e-15,1.0000000001e-15,.5,5,'nan','inf','bad',[0,1],[.5,1.5],[1],[1,2,3],[],['nan',1]):add('sigma-'+str(sigma),sigma=sigma)
 add('sigma-np-f32',sigma=.75,sigma_kind='np-f32')
 add('sigma-array0d',sigma=.75,sigma_kind='array')
 add('sigma-array',sigma=[.5,1.5],sigma_kind='array')
 add('sigma-array-rank2',sigma=[[1,2]],sigma_kind='array')
 for alpha in(0,1,-1,-100,.5,'nan','inf','neginf','bad','complex'):add('alpha-'+str(alpha),alpha=alpha)
 for kind in('np-f32','np-i16','row','column','leading','bad-shape','list','provider'):add('alpha-'+kind,alpha=2,alpha_kind=kind)
 add('readonly',readonly=True)
 add('metadata',metadata=True)
 add('swapped',dtype='>f4')
 add('image-array-protocol',provider='array')
 add('image-provider-refuses',provider='raise')
 add('image-list',provider='list')
 return deepcopy(rows)

def broadcast_shape(first,second):
 out=[]
 for i in range(1,max(len(first),len(second))+1):
  a=first[-i]if i<=len(first)else 1;b=second[-i]if i<=len(second)else 1
  if a!=b and a!=1 and b!=1:raise ValueError('incompatible frozen broadcast')
  out.insert(0,b if a==1 else a)
 return out

def source_safe(spec):
 assert type(spec)is dict and spec in recipes(),'RECIPE_NOT_FROZEN'
 shape=spec['shape'];rank=len(shape);assert rank<=8 and all(type(x)is int and 0<=x<=7 for x in shape)
 count=prod(shape);assert count<=MAX_INPUT
 physical=count*(2**rank if spec['layout']=='stride2'else 1)
 alpha_shape={'row':[1,4],'column':[3,1],'leading':[2,1,1],'bad-shape':[2,3],'list':[4],'provider':[1,4]}.get(spec['alpha_kind'],[])
 try:out_shape=broadcast_shape(shape,alpha_shape);out_count=prod(out_shape);broadcast_error=False
 except ValueError:out_shape=None;out_count=0;broadcast_error=True
 assert out_count<=MAX_OUTPUT
 sigma=5 if spec['sigma']=='OMITTED'else spec['sigma'];sigmas=sigma if isinstance(sigma,list)else[sigma]*rank
 early_sigma_error=len(sigmas)!=rank or any(isinstance(x,(list,str))and x!='nan'for x in sigmas)
 radii=[];active=[]
 for value in sigmas:
  if isinstance(value,(int,float)):
   assert not(isfinite(value)and value>5)
   active.append(value>1e-15);radii.append(int(4*value+.5)if value>1e-15 else 0)
  else:active.append(False);radii.append(0)
 if early_sigma_error:radii=[0]*rank;active=[False]*rank
 # Zero/negative/NaN sigma filters are skipped, not absolute-valued/clamped.
 widths=[2*r+1 for r,yes in zip(radii,active)if yes];axes=len(widths)
 components=2 if spec['dtype'].startswith('complex')else 1
 filter_ops=components*count*rank*sum(widths)
 coefficient_ops=rank*sum(widths)*16+rank*rank*32
 pointwise=2*rank*count+count+4*out_count
 work=filter_ops+coefficient_ops+pointwise
 line_elements=max([(count//length)*(length+2*r)for length,r,yes in zip(shape,radii,active)if length>0 and yes]or[0])
 # NI_AllocateLineBuffer clampslines to actualN/axislength beforemalloc;
 # input halos plusoutputline buffers and coefficientarrays are counted here.
 bytes_upper=16*(physical+12*count+8*out_count+prod(alpha_shape))+8*(2*line_elements+rank*sum(widths)*8+32*rank)+4096
 assert max(radii or[0])<=MAX_RADIUS and work<=MAX_WORK and bytes_upper<=MAX_NUMERICAL_BYTES
 return dict(input_elements=count,physical_elements=physical,output_shape_upper=out_shape,output_elements_upper=out_count,alpha_broadcast_error=broadcast_error,radii=radii,line_elements_upper=line_elements,filter_ops_upper=filter_ops,coefficient_ops_upper=coefficient_ops,work_upper=work,named_numerical_bytes_upper=bytes_upper,early_sigma_error=early_sigma_error)

def fixture(spec,np=None):
 source_safe(spec)
 if np is None:import numpy as np
 from crop_contract import ArrayProvider
 events=[];shape=spec['shape'];n=prod(shape);dtype=np.dtype(spec['dtype'])
 if spec['metadata']:dtype=np.dtype(dtype,metadata={'units':'IGG','source':'prospective'})
 image=(np.arange(n,dtype=np.float64)%7).astype(dtype).reshape(shape)
 pattern=spec['pattern']
 if pattern=='zeros':image[...] = 0
 elif pattern in('constant','bright'):image[...] = 2 if pattern=='constant'else 7
 elif pattern=='step':image.flat[:n//2]=0
 elif pattern=='signedzero':image.flat[0]=-0.0
 elif pattern=='nan':image.flat[n//2]=np.nan
 elif pattern=='inf':image.flat[n//2]=np.inf
 elif pattern=='neginf':image.flat[n//2]=-np.inf
 if spec['layout']=='F':image=np.array(image,order='F')
 elif spec['layout']=='reverse':image=image[(slice(None,None,-1),)*image.ndim]
 elif spec['layout']=='transpose':image=image.T
 elif spec['layout']=='stride2':
  backing=np.zeros(tuple(2*s for s in shape),dtype=dtype);view=backing[(slice(None,None,2),)*len(shape)];view[...]=image;image=view
 if spec['readonly']:image.flags.writeable=False
 buffers={'image':image};argument=image
 if spec['provider']in('array','raise'):argument=ArrayProvider(image,events,error=spec['provider']=='raise')
 elif spec['provider']=='list':argument=image.tolist()
 def scalar(value):
  if value=='nan':return float('nan')
  if value=='inf':return float('inf')
  if value=='neginf':return-float('inf')
  if value=='complex':return 1+2j
  return value
 alpha=scalar(spec['alpha']);kind=spec['alpha_kind']
 if kind=='np-f32':alpha=np.float32(alpha)
 elif kind=='np-i16':alpha=np.int16(alpha)
 elif kind in('row','column','leading','bad-shape','list','provider'):
  ashape={'row':(1,4),'column':(3,1),'leading':(2,1,1),'bad-shape':(2,3),'list':(4,),'provider':(1,4)}[kind]
  value=np.full(ashape,alpha,dtype=np.float32);buffers['alpha']=value
  alpha=value.tolist()if kind=='list'else ArrayProvider(value,events)if kind=='provider'else value
 options={'alpha':alpha};sigma=spec['sigma']
 if sigma!='OMITTED':
  sigma=[scalar(x)for x in sigma]if isinstance(sigma,list)else scalar(sigma)
  if spec['sigma_kind']=='np-f32':sigma=np.float32(sigma)
  elif spec['sigma_kind']=='array':sigma=np.asarray(sigma,dtype=np.float64);buffers['sigma']=sigma
  options['sigma']=sigma
 return(argument,),options,events,buffers

def reference_runtime(binding=None):
 raise SystemExit('IGG_PROSPECTIVE_UNBOUND: source/storage/readiness and separate root GO required; budgets0')
