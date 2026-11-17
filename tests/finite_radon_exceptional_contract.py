"""Four prospective width5 exceptional-cast witnesses; no capture authority."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT='coverage/contracts/finite-radon-exceptional-v1.json'


def cases():return json.loads((ROOT/CONTRACT).read_bytes())['recipes']


def source_safe(row):
 assert row in cases() and row['n']==5 and row['dtype']=='float64'and row['layout']=='C'
 assert row['input']=='ndarray'and row['pattern']in('float-nan','float-inf')and row['readonly']is False
 count=(5+int(row['api']=='ifrt2'))*5
 return dict(input_elements=count,physical_elements=count,physical_bytes=count*8,max_output_elements=30,max_output_bytes=240,max_original_additions=150,max_total_arithmetic=200,max_shift_copy_elements=5,max_roll_calls=16,original_budget=0)


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
