"""NumPy-only factory verification; no target/provider resolver."""
import hashlib,math,json
from inverse_gaussian_gradient_contract import recipes,source_safe

def validate_factory(spec,args,options,events,buffers,np):
 b=source_safe(spec);a=buffers['image'];n=b['input_elements'];shape=tuple(reversed(spec['shape']))if spec['layout']=='transpose'else tuple(spec['shape'])
 assert type(a)is np.ndarray and a.shape==shape and a.size==n and events==[]
 assert a.dtype.str==np.dtype(spec['dtype']).str and a.dtype.metadata==({'units':'IGG','source':'prospective'}if spec['metadata']else None)
 assert a.flags.writeable is(not spec['readonly'])
 root=a
 while isinstance(root.base,np.ndarray):root=root.base
 assert root.size<=b['physical_elements']
 if n:
  p=a.__array_interface__['data'][0];base=root.__array_interface__['data'][0];lo=sum(min(0,(d-1)*s)for d,s in zip(shape,a.strides));hi=sum(max(0,(d-1)*s)for d,s in zip(shape,a.strides))+a.itemsize
  assert base<=p+lo<=p+hi<=base+root.nbytes
 for i,value in enumerate(a.flat):
  if spec['layout']=='reverse':source=n-1-i
  elif spec['layout']=='transpose':
   coords=[];v=i
   for d in reversed(shape):coords.append(v%d);v//=d
   source=0
   for c,d in zip(coords,spec['shape']):source=source*d+c
  else:source=i
  expected=source%7;pattern=spec['pattern']
  if pattern=='zeros'or pattern=='step'and source<n//2:expected=0
  elif pattern=='constant':expected=2
  elif pattern=='bright':expected=7
  elif pattern=='signedzero'and source==0:expected=-0.0
  elif pattern=='nan'and source==n//2:expected=float('nan')
  elif pattern=='inf'and source==n//2:expected=float('inf')
  elif pattern=='neginf'and source==n//2:expected=-float('inf')
  if a.dtype.kind=='b':expected=bool(expected)
  value=value.item();part=value.real if isinstance(value,complex)else value
  if isinstance(expected,float)and math.isnan(expected):assert math.isnan(part)
  else:assert value==expected
  if pattern=='signedzero'and source==0:assert math.copysign(1,part)==-1
 assert len(args)==1
 if spec['provider']in('array','raise'):assert args[0].array is a and args[0].events is events and args[0].error==(spec['provider']=='raise')
 elif spec['provider']=='list':assert args[0]==a.tolist()
 else:assert args[0]is a
 def literal(value):
  if isinstance(value,list):return [literal(v)for v in value]
  return {'nan':float('nan'),'inf':float('inf'),'neginf':-float('inf'),'complex':1+2j}.get(value,value)if isinstance(value,str)else value
 def equal(actual,expected):
  if isinstance(expected,list):assert isinstance(actual,list)and len(actual)==len(expected);[equal(a,e)for a,e in zip(actual,expected)]
  elif isinstance(expected,float)and math.isnan(expected):assert math.isnan(actual)
  else:assert actual==expected
 alpha=options['alpha'];kind=spec['alpha_kind']
 if kind in ('row','column','leading','bad-shape','list','provider'):
  value=buffers['alpha'];shape={'row':(1,4),'column':(3,1),'leading':(2,1,1),'bad-shape':(2,3),'list':(4,),'provider':(1,4)}[kind]
  assert value.shape==shape and value.dtype==np.dtype('float32')
  for x in value.flat:equal(x.item(),literal(spec['alpha']))
  if kind=='list':equal(alpha,value.tolist())
  elif kind=='provider':assert alpha.array is value and alpha.events is events
  else:assert alpha is value
 else:
  equal(alpha.item()if isinstance(alpha,np.generic)else alpha,literal(spec['alpha']))
  if kind=='np-f32':assert type(alpha)is np.float32
  elif kind=='np-i16':assert type(alpha)is np.int16
  else:assert type(alpha)is type(literal(spec['alpha']))
 if spec['sigma']=='OMITTED':assert 'sigma'not in options
 else:
  sigma=options['sigma'];kind=spec['sigma_kind']
  if kind=='array':assert sigma is buffers['sigma']and sigma.dtype==np.dtype('float64');equal(sigma.tolist(),literal(spec['sigma']))
  else:
   equal(sigma.item()if isinstance(sigma,np.generic)else sigma,literal(spec['sigma']))
   if kind=='np-f32':assert type(sigma)is np.float32
 assert events==[]
 descriptors={k:dict(shape=list(v.shape),dtype=v.dtype.str,strides=list(v.strides),metadata=None if v.dtype.metadata is None else dict(v.dtype.metadata),bytes=v.nbytes,word_bytes_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest())for k,v in buffers.items()}
 # Logical payload codec is existing transport, not an algorithm oracle.
 from inverse_gaussian_gradient_observer import encode
 diagnostics=dict(bounds=b,buffers=descriptors,options=encode(options,np),scope='Input words/descriptors only; output unobserved')
 json.dumps(diagnostics,allow_nan=False)
 return diagnostics
