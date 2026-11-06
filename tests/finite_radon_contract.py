"""Source-safe prospective finite Radon recipes. Original budget is zero."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/finite-radon-v1.json'
UPSTREAM_SHA='2659f407ee34e8b25c9e94a49d0ba2ef10fe8a0234f92ccee7b84698290171cc'
DTYPES=('bool','uint8','uint16','uint32','uint64','int8','int16','int32','int64','float16','float32','float64','complex64','complex128','object','S3','U2','datetime64[D]','timedelta64[D]')
def cases():
 rows=[]
 for api in ('frt2','ifrt2'):
  def add(name,**kw):
   d=dict(id='transform.'+api+'/'+name,api=api,n=3,dtype='uint32',layout='C',input='ndarray',pattern='ramp',readonly=False);d.update(kw);rows.append(d)
  add('default')
  for n in (0,1,2,4,5,7):add('size-'+str(n),n=n)
  for dtype in DTYPES:
   if dtype!='uint32':add('dtype-'+dtype,dtype=dtype)
  for layout in ('F','transpose','reverse_rows','reverse_columns','step','unaligned','swapped'):add('layout-'+layout,layout=layout)
  for pattern in ('zero','max','negative','u32-overflow','u64-overflow','float-fraction','float-nan','float-inf'):
   dtype='uint64'if pattern=='u64-overflow'else'int64'if pattern=='negative'else'float64'if pattern.startswith('float-')else'uint32';add('pattern-'+pattern,pattern=pattern,dtype=dtype)
  add('readonly',readonly=True)
  for kind in ('list','tuple','scalar','numpy_scalar','memoryview','array_provider','interface_provider','subclass','copy_error'):add('input-'+kind,input=kind)
  for shape in ([3],[2,3],[2,2,2],[]):add('shape-'+str(shape),shape=shape)
 return rows

def source_safe(row):
 assert row in cases() and type(row['n'])is int and 0<=row['n']<=7
 shape=row.get('shape',[row['n']+(row['api']=='ifrt2'),row['n']]);size=1
 for n in shape:assert type(n)is int and 0<=n<=8;size*=n
 assert size<=64
 n=row['n'];return dict(input_elements=size,max_physical_elements=max(1,size*4+1),max_output_elements=n*(n+1),max_shift_copy_elements=n,max_original_additions=max(1,n*n*(n+1)),target_calls=0,original_budget=0)

def fixture(row,np):
 from crop_contract import ArrayProvider,InterfaceProvider
 source_safe(row);n=row['n'];shape=tuple(row.get('shape',[n+(row['api']=='ifrt2'),n]));size=1
 for d in shape:size*=d
 dtype=np.dtype(row['dtype']);pattern=row['pattern'];values=[]
 for k in range(size):
  x=k%7
  if pattern=='zero':x=0
  elif pattern=='max':x=int(np.iinfo(dtype).max)if dtype.kind in'iu'else 1
  elif pattern=='negative':x=-x-1
  elif pattern=='u32-overflow':x=2**32-1
  elif pattern=='u64-overflow':x=2**64-1
  elif pattern=='float-fraction':x=x/2-1.5
  elif pattern=='float-nan':x=float('nan')if k==0 else x
  elif pattern=='float-inf':x=float('inf')if k==0 else x
  values.append(x)
 if dtype.kind=='M':values=['2000-01-01']*size
 a=np.array(values,dtype=dtype).reshape(shape).copy();layout=row['layout'];events=[]
 if layout=='F':a=np.asfortranarray(a)
 elif layout=='transpose':a=a.T
 elif layout=='reverse_rows'and a.ndim:a=a[::-1]
 elif layout=='reverse_columns'and a.ndim==2:a=a[:,::-1]
 elif layout=='step':
  owner=np.zeros(tuple(d*2 for d in shape),dtype=dtype);view=owner[tuple(slice(None,None,2)for d in shape)];view[...]=a;a=view
 elif layout=='unaligned':
  owner=np.zeros(max(1,size*dtype.itemsize+1),dtype=np.uint8);view=np.ndarray(shape,dtype=dtype,buffer=owner,offset=1);view[...]=a;a=view
 elif layout=='swapped':a=a.astype(dtype.newbyteorder('>'))
 if row['readonly']:a.flags.writeable=False
 buffers={'image':a};kind=row['input'];arg=a
 if kind=='list':arg=a.tolist()
 elif kind=='tuple':arg=tuple(a.tolist())
 elif kind=='scalar':arg=3
 elif kind=='numpy_scalar':arg=np.uint32(3)
 elif kind=='memoryview':arg=memoryview(a)
 elif kind=='array_provider':arg=ArrayProvider(a,events)
 elif kind=='interface_provider':arg=InterfaceProvider(a,events)
 elif kind in('subclass','copy_error'):
  class Controlled(np.ndarray):
   def copy(self,*args,**kwargs):
    events.append({'event':'copy','args':list(args),'kwargs':kwargs})
    if kind=='copy_error':raise RuntimeError('finite Radon controlled copy refusal')
    return np.asarray(self).copy(*args,**kwargs)
  arg=a.view(Controlled)
 return (arg,),{},events,buffers

def validate_intake(path):
 d=json.loads(Path(path).read_bytes());raw=CONTRACT.read_bytes();c=json.loads(raw)
 assert set(d)=={'schema','mode','contract_sha256','source_files_sha256','original_calls'}
 assert d['schema']=='finite-radon-intake-v1'and d['mode']=='intake-only'and type(d['original_calls'])is int and d['original_calls']==0
 assert d['contract_sha256']==hashlib.sha256(raw).hexdigest()and d['source_files_sha256']==c['source_freeze']
 for n,p in d['source_files_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==p,n
 return d

def runtime_request():
 import numpy
 import skimage.transform
 raise AssertionError('UNAUTHORIZED_TARGET_RESOLUTION')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true');p.add_argument('--intake-binding',type=Path);a=p.parse_args()
 if a.plan:print(json.dumps(dict(status='SOURCE_ONLY_BUDGET0',recipes=cases(),bounds=[source_safe(s)for s in cases()])));return
 if a.intake_binding is None:raise SystemExit('FINITE_RADON_UNBOUND originalbudget0')
 validate_intake(a.intake_binding);runtime_request()
if __name__=='__main__':main()
