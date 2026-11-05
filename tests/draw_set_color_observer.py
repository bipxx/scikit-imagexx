"""Exact in-place state observer; original execution is controller-owned."""
import warnings
from draw_coordinates_observer import encode,scientific_profile
from draw_set_color_contract import fixture

def observe(call,spec,np):
    args,kw=fixture(spec,np);image=args[0]
    before=encode((args,kw),np);pointer=image.ctypes.data
    descriptors=(image.shape,image.strides,image.dtype,image.flags.writeable,image.flags.owndata)
    arrays=[a for a in (image,*args[1],args[2],kw['alpha'])if type(a)is np.ndarray]
    aliases=[[bool(np.shares_memory(a,b)),a is b]for a in arrays for b in arrays]
    old=np.geterr(),np.geterrcall(),np.getbufsize();events=[]
    def callback(message,flags):events.append([message,flags])
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(all='warn',call=callback):
            policy=np.geterr();error=None;result=None
            try:result=call(*args,**kw)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            stable=np.geterr()==policy and np.geterrcall()is callback and np.getbufsize()==old[2]
    checks=dict(pointer=image.ctypes.data==pointer,descriptors=(image.shape,image.strides,image.dtype,image.flags.writeable,image.flags.owndata)==descriptors,policy=stable,restored=np.geterr()==old[0]and np.geterrcall()is old[1]and np.getbufsize()==old[2])
    assert all(checks.values())
    return dict(id=spec['id'],before=before,after=encode((args,kw),np),result=encode(result,np),error=error,warnings=[[type(x.message).__module__,type(x.message).__name__,str(x.message)]for x in notices],FP_events=events,aliases=aliases,checks=checks)
