"""Cold public dtype-factory substitution regression; no operation timings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import skimagexx as xx


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('behavior',['observe','raise'])
def test_cold_public_scalar_admission_never_calls_python_dtype_factory(workers,behavior):
    code=r'''
import json,sys,warnings
import numpy as np
from skimage import morphology as original
import skimagexx as xx
native=xx.morphology
xx.set_fallback_enabled(False);xx.set_num_threads(int(sys.argv[1]))
canonical=np.dtype;behavior=sys.argv[2];checks=0
types=[('default',None),('None',None),('bool',bool),('int',int),('float',float),('complex',complex),
       ('np.bool_',np.bool_),('np.uint8',np.uint8),('np.int8',np.int8),('np.uint16',np.uint16),
       ('np.int16',np.int16),('np.uint32',np.uint32),('np.int32',np.int32),('np.uint64',np.uint64),
       ('np.int64',np.int64),('np.float16',np.float16),('np.float32',np.float32),('np.float64',np.float64),
       ('np.complex64',np.complex64),('np.complex128',np.complex128),
       ('descriptor',canonical('>i2',metadata={'caller':'descriptor'})),('string','uint8')]
for label,dtype in types:
 for decomposition in [None,'sequence','separable']:
  observations=[]
  for namespace in [original,native]:
   events=[]
   def factory(*args,**kwargs):
    events.append(repr(args))
    if behavior=='raise':raise RuntimeError('caller replaced numpy.dtype factory')
    return canonical(*args,**kwargs)
   options=dict(decomposition=decomposition)
   if label!='default':options['dtype']=dtype
   try:
    np.dtype=factory
    with warnings.catch_warnings(record=True) as notices:
     warnings.simplefilter('always')
     try:value,error=namespace.footprint_rectangle((3,5),**options),None
     except Exception as caught:value,error=None,caught
   finally:np.dtype=canonical
   assert not events,(label,decomposition,namespace.__name__,events)
   assert error is None,(label,decomposition,namespace.__name__,repr(error))
   elements=value if isinstance(value,tuple) else ((value,1),)
   observations.append((tuple((a.dtype.str,a.dtype.metadata,a.shape,a.strides,a.flags.owndata,a.flags.writeable,a.tobytes(),n) for a,n in elements),[(type(w.message),str(w.message)) for w in notices]))
  assert observations[0]==observations[1],(label,decomposition)
  checks+=1
assert xx.get_num_threads()==int(sys.argv[1])
print(json.dumps(dict(status='passed-cold-public-dtype-factory',cases=checks,workers=xx.get_num_threads(),behavior=behavior)))
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path(xx.__file__).resolve().parent.parent)
    result=subprocess.run([sys.executable,'-c',code,str(workers),behavior],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report==dict(status='passed-cold-public-dtype-factory',cases=66,workers=workers,behavior=behavior)
