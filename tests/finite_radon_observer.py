"""One supplied call; no target resolver, no reference authorization."""
import json,warnings
from crop_observer import encode
from finite_radon_contract import fixture

def error_value(v,np):
 if isinstance(v,np.ufunc):return dict(type='numpy.ufunc',name=v.__name__,nin=v.nin,nout=v.nout)
 if isinstance(v,np.dtype):return dict(type='numpy.dtype',str=v.str,descr=encode(v.descr,np),metadata=None if v.metadata is None else dict(v.metadata))
 if isinstance(v,(tuple,list)):return dict(container=type(v).__name__,items=[error_value(x,np)for x in v])
 if isinstance(v,dict):return{k:error_value(x,np)for k,x in v.items()}
 return encode(v,np)

def observe(call,spec,np):
 args,options,events,buffers=fixture(spec,np);before=encode((args,options),np);result=None;error=None
 policy=np.geterr();handler=np.geterrcall();buffer=np.getbufsize()
 with warnings.catch_warnings(record=True)as caught:
  warnings.simplefilter('always')
  try:result=call(*args,**options)
  except Exception as e:error=dict(module=type(e).__module__,name=type(e).__name__,qualname=type(e).__qualname__,message=str(e),args=error_value(e.args,np))
 aliases={}
 if isinstance(result,np.ndarray):
  for n,a in buffers.items():aliases[n]=dict(identity=result is a,base_is_input=result.base is a,shares_memory=bool(np.shares_memory(result,a)))
 row=dict(id=spec['id'],before=before,after=encode((args,options),np),result=None if error else encode(result,np),error=error,warnings=[dict(category=w.category.__name__,message=str(w.message))for w in caught],events=events,aliases=aliases,numpy_before=dict(policy=policy,handler_type=type(handler).__name__,buffer=buffer),numpy_after=dict(policy=np.geterr(),handler_type=type(np.geterrcall()).__name__,buffer=np.getbufsize()),supplied_call_attempts=1)
 json.dumps(row,allow_nan=False);return row
