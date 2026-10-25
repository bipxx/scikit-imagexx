"""Genuine installed public reconstruction gate; private adapters forbidden.

The frozen numerical/provider/binding assertions are reused unchanged. Only
this fixture selects the actual installed sole core and public wrapper.
"""
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
PIPELINE=load('reconstruction_public_fixed_pipeline','test_reconstruction_native_pipeline.py')
BINDING=load('reconstruction_public_fixed_binding','test_reconstruction_binding.py')
OWNERS=load('reconstruction_public_fixed_owners','test_reconstruction_owner_guards.py')
OFFSETS=load('reconstruction_public_fixed_offsets','test_reconstruction_offset_guards.py')

@pytest.fixture(scope='module')
def backend():
    assert 'SKIMAGEXX_RECONSTRUCTION_CORE' not in os.environ
    assert 'SKIMAGEXX_EXPECT_OPENMP_OFF' not in os.environ
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    wrapper=importlib.import_module('skimagexx._reconstruction');morphology=importlib.import_module('skimagexx.morphology');core=xx._core
    assert core.__name__=='skimagexx._core' and wrapper._core is core
    assert Path(core.__file__).resolve().parent==Path(xx.__file__).resolve().parent==Path(wrapper.__file__).resolve().parent
    assert core.get_num_threads()==1 and getattr(core,'has_openmp',None) is True
    assert core.reconstruction_interface=='reconstruction-v1' and type(core.reconstruction_capabilities) is int and core.reconstruction_capabilities==3
    assert core.reconstruction_parallel_flood is False
    assert tuple(native_functions()).count('morphology.reconstruction')==1
    assert morphology.reconstruction is wrapper.reconstruction and morphology.__all__.count('reconstruction')==1
    assert wrapper.reconstruction.__module__=='skimagexx._reconstruction'
    assert str(inspect.signature(wrapper.reconstruction))=="(seed, mask, method='dilation', footprint=None, offset=None)"
    for name in ('reconstruction_run','reconstruction_ordered','reconstruction_negative_zero','reconstruction_counts'):
        entry=getattr(core,name);assert inspect.isbuiltin(entry) and entry.__module__==core.__name__ and entry.__name__==name
    assert type(getattr(core,'_consumer_gil_assertions',None)) is bool
    previous=xx.set_fallback_enabled(False);assert xx.set_fallback_enabled(False) is False
    try:yield core,wrapper
    finally:core.set_num_threads(1);xx.set_fallback_enabled(previous)

# Reuse exact function objects/marks: no changed expected value, tolerance,
# source-unsafe boundary, side-effect assertion, or private source replacement.
test_actual_public_pipeline=PIPELINE.test_actual_private_pipeline
test_actual_large_queue_and_workers=PIPELINE.test_actual_large_queue_and_workers
for name in vars(BINDING):
    if name.startswith('test_'):globals()[name]=getattr(BINDING,name)
for name in vars(OWNERS):
    if name.startswith('test_'):globals()[name]=getattr(OWNERS,name)
for name in vars(OFFSETS):
    if name.startswith('test_'):globals()[name]=getattr(OFFSETS,name)


def test_actual_public_no_original_algorithm_forwarding(backend):
    core,wrapper=backend;original=PIPELINE.CONTRACT.ORIGINAL;attempts=[];previous=sys.getprofile()
    def reject(frame,event,arg):
        if event=='call' and frame.f_code is original.__code__:attempts.append(True);raise RuntimeError('controlled upstream-original denial')
    try:
        sys.setprofile(reject)
        with pytest.raises(RuntimeError,match='controlled upstream-original denial'):original(None,None)
    finally:sys.setprofile(previous)
    assert attempts==[True]
