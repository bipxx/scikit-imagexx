"""Small cross-platform diagnostics for inverse-Luv rounding failures."""
import json
import math
import os
import subprocess
import sys
import numpy as np
from skimage import color
import skimagexx as xx

xx.set_fallback_enabled(False)
rgb=np.random.default_rng(71).random((519,521,3)).astype(np.float32)
luv=color.rgb2luv(rgb)
expected_xyz=color.luv2xyz(luv)
actual_xyz=xx.color.luv2xyz(luv)
expected_rgb=color.luv2rgb(luv)
actual_rgb=xx.color.luv2rgb(luv)
curve=(luv[...,0]+np.float32(16))/np.float32(116)
power=np.power(curve,np.float32(3))
double_cube=(curve.astype(np.float64)**3).astype(np.float32)
multiply_cube=(curve*curve)*curve
nonlinear=luv[...,0]>np.float32(7.999625)
y=actual_xyz[...,1]
diagnostics={'numpy':np.__version__,'avx2':xx.has_avx2(),'simd_disabled':os.environ.get('SKIMAGEXX_DISABLE_SIMD'),
             'xyz_max_error':np.max(np.abs(actual_xyz-expected_xyz)).item(),
             'xyz_y_different':np.count_nonzero(actual_xyz[...,1]!=expected_xyz[...,1]).item(),
             'rgb_max_error':np.max(np.abs(actual_rgb-expected_rgb)).item(),
             'rgb_outside':np.count_nonzero(np.abs(actual_rgb-expected_rgb)>2e-6*(1+np.abs(expected_rgb))).item(),
             'native_y_vs_power':np.count_nonzero((y!=power)&nonlinear).item(),
             'native_y_vs_multiply':np.count_nonzero((y!=multiply_cube)&nonlinear).item(),
             'native_y_vs_double_cube':np.count_nonzero((y!=double_cube)&nonlinear).item(),
             'numpy_power_vs_multiply':np.count_nonzero((power!=multiply_cube)&nonlinear).item()}
print(json.dumps(diagnostics),flush=True)
if os.environ.get('SKIMAGEXX_DISABLE_SIMD')!='1':
    env=os.environ.copy(); env['SKIMAGEXX_DISABLE_SIMD']='1'
    subprocess.run([sys.executable,__file__],env=env,check=True)
