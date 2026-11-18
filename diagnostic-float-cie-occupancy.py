"""Private profiling of specifically retained float-color timing gaps."""
from importlib import import_module
import json
from pathlib import Path
import sys

report_path=Path(sys.argv[1])
source=Path.cwd()
sys.path.insert(0,str(source/'benchmarks'))
from run import measure
from skimagexx import _core, color, set_fallback_enabled, set_num_threads
import numpy as np

set_fallback_enabled(False)
set_num_threads(1)
targets={('color.rgb2lab','float32'),('color.rgb2lab','float64'),
         ('color.xyz2lab','float32'),('color.xyz2lab','float64'),('color.rgb2luv','float32')}
basis=json.loads(report_path.read_bytes())
eligible=[(r['api'],r['dtype']) for r in basis['results']
          if r['speedup']<1 and (r['api'],r['dtype']) in targets]
introspect=import_module('numpy.lib.introspect')
dispatch=introspect.opt_func_info(func_name='^(cbrt|power)$')
rng=np.random.default_rng(218)
white=color.xyz_tristimulus_values(illuminant='D65',observer='2')
report=[]
for api,dtype in eligible:
    name=api.rsplit('.',1)[1]
    dtype=np.dtype(dtype)
    uv=color._luv_reference(tuple(white),dtype.str) if name=='rgb2luv' else None
    forward_color=name in ('rgb2lab','rgb2luv')
    cases={'linear':np.full((64,64,3),.01 if forward_color else .003,dtype=dtype),
           'both_non_linear':rng.uniform(.25,.8,size=(64,64,3)).astype(dtype)}
    if forward_color:
        cases['only_rgb_decode']=np.full((64,64,3),.07,dtype=dtype)
    else:
        cases['mixed']=np.where(rng.integers(0,2,size=(64,64,3)),.003,.48).astype(dtype)
    for label,payload in cases.items():
        op=color._PERCEPTUAL[name]
        def public():
            return getattr(color,name)(payload)
        def direct():
            return _core.perceptual_transform(payload,op,white,uv)
        actual,negative,unsupported=direct()
        expected=public()
        assert not negative and not unsupported and actual.tobytes()==expected.tobytes()
        timing=measure(public,direct,7,15)
        report.append(dict(api=api,dtype=str(dtype),case=label,public_first=True,
                           comparison='exact own public and direct allocation-inclusive calls',
                           numpy_dispatch=dispatch,**timing))
Path(sys.argv[2]).write_text(json.dumps(dict(scope='changed input branch occupancy; no release timing evidence',rows=report),indent=2)+'\n',encoding='utf-8')
