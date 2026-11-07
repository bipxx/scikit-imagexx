"""Ordinary installed IGG namespace/default/one-core plumbing; genuine gate only."""
import ast
import importlib
import inspect
from pathlib import Path
import runpy
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]


def test_IGG_shared_source():
    cmake=(ROOT/'CMakeLists.txt').read_text();binding=(ROOT/'src/bindings.cpp').read_text()
    assert cmake.count('target_sources(skimagexx PRIVATE src/inverse_gaussian_gradient.cpp)')==1
    assert 'src/bind_inverse_gaussian_gradient.cpp'in cmake
    assert cmake.count('set_source_files_properties(src/inverse_gaussian_gradient.cpp src/bind_inverse_gaussian_gradient.cpp')==2
    assert binding.count('void bind_inverse_gaussian_gradient(pybind11::module_&);')==binding.count('    bind_inverse_gaussian_gradient(m);')==1
    text=(ROOT/'python/skimagexx/segmentation.py').read_text();exports=[]
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='__all__'for t in node.targets):exports=ast.literal_eval(node.value)
        if isinstance(node,ast.AugAssign)and isinstance(node.target,ast.Name)and node.target.id=='__all__':exports+=ast.literal_eval(node.value)
    assert 'inverse_gaussian_gradient'in exports and 'from ._inverse_gaussian_gradient import inverse_gaussian_gradient'in text
    coverage=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'))
    assert 'segmentation.inverse_gaussian_gradient'in coverage['native_functions']()
    assert coverage['native_dtypes']('segmentation.inverse_gaussian_gradient')==('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64')
    assert 'fully_native'in coverage['inverse_gaussian_gradient_contract']()and not coverage['inverse_gaussian_gradient_contract']()['fully_native']
    assert 'installed_inverse_gaussian_gradient_consumer'in(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()
    assert (ROOT/'tests/cmake_consumer/inverse_gaussian_gradient_consumer.cpp').read_text()=='#include "../test_inverse_gaussian_gradient_sdk.cpp"\n'


def test_IGG_installed_default_single_core_and_hot_swap(monkeypatch):
    import numpy as np
    import skimagexx as xx
    import skimage.segmentation as original
    from skimagexx.coverage import default_call,assert_default_parity
    owned=importlib.import_module('skimagexx._inverse_gaussian_gradient')
    assert owned._core is xx._core
    for name in ('inverse_gaussian_magnitude','inverse_gaussian_finish','get_num_threads','set_num_threads'):
        entry=getattr(xx._core,name);assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    exports={};exec('from skimagexx.segmentation import *',exports)
    assert exports['inverse_gaussian_gradient']is xx.segmentation.inverse_gaussian_gradient is owned.inverse_gaussian_gradient
    assert xx.get_num_threads()==1
    image=np.arange(9,dtype=np.float32).reshape(3,3);before=image.tobytes()
    args,options=default_call('segmentation.inverse_gaussian_gradient',image)
    assert args==(image,)and options=={'alpha':100.0,'sigma':1.0}
    def pipeline(backend):return backend.segmentation.inverse_gaussian_gradient(*args,**options)
    marker=object();requests=[]
    def substituted(*a,**k):requests.append((a,k));return marker
    assert pipeline(SimpleNamespace(segmentation=SimpleNamespace(inverse_gaussian_gradient=substituted)))is marker
    assert requests[0][0][0]is image and requests[0][1]==options
    expected=original.inverse_gaussian_gradient(*args,**options)
    old=xx.set_fallback_enabled(False)
    try:
        calls=[];getter=xx._core.get_num_threads
        def counted():calls.append(True);return getter()
        with monkeypatch.context()as scope:
            scope.setattr(xx._core,'get_num_threads',counted)
            actual=pipeline(xx)
        assert calls==[True]
        assert_default_parity('segmentation.inverse_gaussian_gradient',image.dtype,actual,expected,image=image)
        assert image.tobytes()==before and not np.shares_memory(actual,image)and xx.get_num_threads()==1
    finally:xx.set_fallback_enabled(old)
