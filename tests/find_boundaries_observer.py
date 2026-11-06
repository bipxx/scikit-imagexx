"""Raw observer reusing the established encoder; no implicit captures."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import sys
import warnings
if __package__:
    from .draw_coordinates_observer import encode,scientific_profile
    from .find_boundaries_contract import fixture,SIGNATURE
else:
    from draw_coordinates_observer import encode,scientific_profile
    from find_boundaries_contract import fixture,SIGNATURE

SOURCE_SHA256='fbdbbf36d8813c09d00583d235063acb0911d842cd94628d482d5c5685818131'
DEPENDENCY_SOURCE_PINS={
    'skimage.morphology.gray':'19a82264276da07e8ff780fae8b092543965b5c88f34db4d16536a8d1f1e302e',
    'skimage.util.shape':'fc9ab1bdc09fa464828b6ee3ce96e610d04606b90c13a3ed58f80138e65c876b',
}
SCIPY_SOURCE_PINS={
    '1.15.3':'2c5f7580a25c1c85a80fd6ad87ff7e63b1e053041b03410baa0558be6d580160',
    '1.18.1':'5361c28977347a3bc4ba1240aea0b341a72a803498a5600506c9964a61eeb9ad',
}


def reference_runtime():
    np=scientific_profile();module=importlib.import_module('skimage.segmentation.boundaries')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA256
    pins=dict(DEPENDENCY_SOURCE_PINS)
    pins['scipy.ndimage._morphology']=SCIPY_SOURCE_PINS[importlib.metadata.version('scipy')]
    sources={}
    for name,pin in pins.items():
        dependency=importlib.import_module(name);path=Path(dependency.__file__)
        assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
        sources[name]=dict(path=str(path),canonical_lf_sha256=pin)
    original=module.find_boundaries
    assert str(inspect.signature(original))==SIGNATURE
    assert importlib.import_module('skimage.segmentation').find_boundaries is original
    return np,original,dict(python=sys.version,numpy=np.__version__,source_path=str(module.__file__),source_canonical_lf_sha256=SOURCE_SHA256,dependencies=sources)


@contextmanager
def without_original_activity(original):
    module=importlib.import_module('skimage.segmentation.boundaries')
    targets=[original,module._find_boundaries_subpixel,module.dilation,module.erosion,module.ndi.generate_binary_structure]
    codes={fn.__code__ for fn in targets}
    for fn in targets:
        while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__;codes.add(fn.__code__)
    previous=sys.getprofile()
    def guard(frame,event,value):
        if event=='call'and frame.f_code in codes:raise AssertionError('Original find-boundaries dependency forbidden during fixture intake')
    sys.setprofile(guard)
    try:yield
    finally:sys.setprofile(previous)


def observe(call,spec,np):
    args,options,buffers=fixture(spec,np)
    before=encode((args,options,buffers),np);pointers={key:value.ctypes.data for key,value in buffers.items()}
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();events=[];result=None;error=None
    def callback(message,flags):events.append(dict(message=message,flags=int(flags)))
    try:
        np.seterrcall(callback);np.seterr(all='warn')
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always')
            try:result=call(*args,**options)
            except Exception as exc:error=dict(type=type(exc).__name__,module=type(exc).__module__,message=str(exc))
        row=dict(id=spec['id'],before=before,after=encode((args,options,buffers),np),result=encode(result,np),error=error,
                 warnings=[dict(category=type(w.message).__name__,message=str(w.message))for w in notices],FP_events=events,
                 output_input_aliases={}if not isinstance(result,np.ndarray)else{key:bool(np.shares_memory(result,value))for key,value in buffers.items()},
                 checks=dict(input_addresses_stable=pointers=={key:value.ctypes.data for key,value in buffers.items()}))
    finally:np.seterr(**policy);np.seterrcall(handler);np.setbufsize(buffer)
    row['checks']['policy_restored']=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
    return row
