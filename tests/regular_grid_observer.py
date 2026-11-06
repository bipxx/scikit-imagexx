"""Slice/scalar words and public state observer; no original forwarding/import."""
import warnings
import ctypes
import sys


def encode(value,np):
    kind=[type(value).__module__,type(value).__qualname__]
    if type(value)is slice:return dict(type=kind,start=encode(value.start,np),stop=encode(value.stop,np),step=encode(value.step,np))
    if type(value)in(tuple,list):return dict(type=kind,items=[encode(x,np)for x in value])
    if isinstance(value,np.ndarray):
        result=dict(type=kind,dtype=value.dtype.str,metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),
            shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),
            c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),aligned=bool(value.flags.aligned))
        if value.dtype.kind=='O':result['items']=[encode(x,np)for x in value.flat]
        else:result['raw']=value.tobytes(order='C').hex()
        return result
    if isinstance(value,np.generic):return dict(type=kind,dtype=value.dtype.str,raw=value.tobytes().hex())
    if type(value)is float:return dict(type=kind,hex=value.hex())
    if value is None or type(value)in(bool,int,str):return dict(type=kind,value=value)
    raise TypeError('Unassessed observer object: '+repr(kind))


def observe(call,spec,np):
    from regular_grid_contract import fixture
    args,options,events=fixture(spec,np);before=encode(args,np)
    old_policy=np.geterr();old_handler=np.geterrcall();old_buffer=np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    np.seterrcall(handler);result=error=None
    # Existing Linux fenv mask reused from boundary tests; capture only, no reset.
    libc=ctypes.CDLL(None)if sys.platform=='linux'else None
    def fenv():
        if libc is None or not all(hasattr(libc,n)for n in('fetestexcept','fegetround')):return None
        return dict(flags=libc.fetestexcept(0x3d),rounding=libc.fegetround())
    try:
        with np.errstate(**spec['policy']),warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always')
            applied_policy=np.geterr().copy()
            fenv_before=fenv()
            try:result=call(*args,**options)
            except Exception as exc:
                fenv_after=fenv();error=dict(type=type(exc).__name__,message=str(exc))
            else:fenv_after=fenv()
            state_during=dict(policy=np.geterr(),handler_unchanged=np.geterrcall()is handler,buffer=np.getbufsize())
        return dict(id=spec['id'],before=before,after=encode(args,np),result=None if error else encode(result,np),error=error,
            events=events,warnings=[dict(category=type(w.message).__name__,message=str(w.message))for w in caught],
            fenv_before=fenv_before,fenv_after=fenv_after,state_during=state_during,checks=dict(policy_unchanged=state_during['policy']==applied_policy,buffer_unchanged=state_during['buffer']==old_buffer))
    finally:np.seterr(**old_policy);np.seterrcall(old_handler);np.setbufsize(old_buffer)
