"""LBP hooks for the admitted public-call envelope and Montage entry guard."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/draw-multiblock-lbp-benchmark-execution-v7.json'


def boundary():
    path=ROOT/'benchmarks/mark_boundaries_execution.py'
    spec=importlib.util.spec_from_file_location('_lbp_public_boundary',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def historical_bytes(root,name):return boundary().historical_bytes(root,name,context_path=CONTEXT)


def fixture(spec,np):
    import math
    image=(np.arange(math.prod(spec['shape']),dtype=np.int64)%7).reshape(spec['shape']).astype(spec['dtype'])
    if spec['layout']=='F':image=np.asfortranarray(image)
    elif spec['layout']=='reverse':image=image[:,::-1]
    return (image,spec['r'],spec['c'],spec['width'],spec['height']),dict(lbp_code=spec['code'],alpha=spec['alpha'])


def native_requirements(spec):
    # The C++ preparation captures workers once internally. There is no Python
    # getter call inside this public function; pre/post global checks remain1.
    return dict(get_num_threads=0,_lbp_prepare=1,_lbp_binary=24,_lbp_assign=8 if len(spec['shape'])==2 else 9)


def execute(module):return boundary().execute(module,context_path=CONTEXT)
