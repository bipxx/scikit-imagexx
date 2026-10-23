"""Run untimed ridge parity with NumPy advanced-vector dispatch disabled."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    if not work.is_relative_to(root/'build') or work.exists():
        parser.error('choose a fresh work directory inside repository build/')
    work.mkdir(parents=True)
    core = np.core if np.__version__.startswith('1.') else np._core
    features = core._multiarray_umath.__cpu_features__
    disabled = {key for key in core._multiarray_umath.__cpu_dispatch__
                if features.get(key, False) and
                (key in ('AVX2', 'FMA3', 'X86_V3', 'X86_V4') or key.startswith('AVX512'))}
    disabled.update(filter(None, os.environ.get('NPY_DISABLE_CPU_FEATURES', '').split(',')))
    base_env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=','.join(sorted(disabled)))
    metadata_code = """import json,numpy as np,skimagexx as xx
core=np.core if np.__version__.startswith('1.') else np._core
features=core._multiarray_umath.__cpu_features__
try:
 from numpy.lib.introspect import opt_func_info
 dispatch=opt_func_info(func_name='^exp$')
except ImportError:dispatch=None
# NumPy can retain hardware capability bits for uncompiled target groups.
# Validate the compiled targets and, when exposed, the actual exp loop.
targets=core._multiarray_umath.__cpu_dispatch__
assert not any(features.get(key,False) for key in targets if key in ('AVX2','FMA3','X86_V3','X86_V4') or key.startswith('AVX512'))
if dispatch is not None:assert dispatch['exp']['ff']['current'].startswith('baseline')
print(json.dumps(dict(numpy=np.__version__,numpy_cpu_features=features,numpy_dispatch_targets=core._multiarray_umath.__cpu_dispatch__,numpy_exp_dispatch=dispatch,cpp_avx2=xx.has_avx2(),cpp_fma=xx._core.frangi_has_fma(),float_exp_uses_numpy=xx._core.frangi_float_exp_uses_numpy())))
"""
    for mode in ('auto', 'scalar'):
        env = base_env.copy()
        if mode == 'scalar':
            env['SKIMAGEXX_DISABLE_SIMD'] = '1'
        else:
            env.pop('SKIMAGEXX_DISABLE_SIMD', None)
        with (work/(mode+'-metadata.json')).open('w', encoding='utf-8') as log, (work/(mode+'-metadata.stderr.log')).open('w', encoding='utf-8') as errors:
            subprocess.run([sys.executable, '-c', metadata_code], cwd=root, env=env,
                           stdout=log, stderr=errors, check=True, text=True)
        with (work/(mode+'-tests.log')).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_frangi.py'],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('NumPy generic dispatch / C++ '+mode+' parity passed; logs: '+str(work))


if __name__ == '__main__':
    main()
