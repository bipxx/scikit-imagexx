"""Enabled single-core rolling-ball plumbing; genuine execution is a release gate."""
import ast
import importlib
import inspect
from types import SimpleNamespace
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def test_rolling_ball_shared_source():
    cmake=(ROOT/'CMakeLists.txt').read_text();bindings=(ROOT/'src/bindings.cpp').read_text()
    assert cmake.count('target_sources(skimagexx PRIVATE src/rolling_ball.cpp)')==1
    assert 'src/bind_rolling_ball.cpp'in cmake
    assert cmake.count('set_source_files_properties(src/rolling_ball.cpp src/bind_rolling_ball.cpp')==2
    assert bindings.count('void bind_rolling_ball(pybind11::module_&);')==bindings.count('    bind_rolling_ball(m);')==1
    public=(ROOT/'python/skimagexx/restoration.py').read_text()
    assert 'from ._rolling_ball import rolling_ball'in public
    tree=ast.parse(public);exports=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='__all__'for t in n.targets))
    assert 'rolling_ball'in exports
    coverage=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'))
    assert 'restoration.rolling_ball'in coverage['native_functions']()
    assert coverage['native_dtypes']('restoration.rolling_ball')==('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64')
    q=(ROOT/'tools/public_qualification.py').read_text()
    assert "selected['public-rolling-ball-saved-linux']"in q and "row['tests']==81"in q
    assert "'tests/fixtures/rolling_ball_sdk_v1.bin'"in q
    assert 'installed_rolling_ball_consumer'in(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()
    for source,dest,name in (('tests/test_rolling_ball_sdk.cpp','tests/cmake_consumer/rolling_ball_consumer.cpp','rolling_ball_float_client'),('tests/test_rolling_ball_sdk_dtypes.cpp','tests/cmake_consumer/rolling_ball_storage_consumer.cpp','rolling_ball_storage_client')):
        expected=(ROOT/source).read_text().replace('int main(int argc,char** argv)','int '+name+'(int argc,char** argv)').replace('}catch(const std::exception& e){std::cerr','return 0;}catch(const std::exception& e){std::cerr')
        current=(ROOT/dest).read_text();assert current.startswith(expected)
        if name.endswith('storage_client'):assert current==expected
    assert 'rolling_ball_storage_client(2,storage_args)'in(ROOT/'tests/cmake_consumer/rolling_ball_consumer.cpp').read_text()


def test_rolling_ball_single_core_star_default_and_hot_swap():
    import numpy as np
    import skimagexx as xx
    from skimagexx.coverage import default_call,assert_default_parity
    owned=importlib.import_module('skimagexx._rolling_ball')
    assert owned._core is xx._core
    for name in ('rolling_ball_apply','ball_kernel_apply','get_num_threads','set_num_threads'):
        value=getattr(xx._core,name)
        assert inspect.isbuiltin(value)and value.__module__=='skimagexx._core'
    exports={};exec('from skimagexx.restoration import *',exports)
    assert exports['rolling_ball']is xx.restoration.rolling_ball is owned.rolling_ball
    assert xx.get_num_threads()==1
    fallback=xx.set_fallback_enabled(False)
    try:
        image=np.arange(9,dtype=np.float32).reshape(3,3);before=image.tobytes()
        args,options=default_call('restoration.rolling_ball',image)
        assert args[0]is image and options=={'radius':1}
        expected=xx.restoration.rolling_ball(*args,**options)
        def pipeline(backend):return backend.restoration.rolling_ball(*args,**options)
        marker=object();requests=[]
        def substituted(*a,**k):requests.append((a,k));return marker
        assert pipeline(SimpleNamespace(restoration=SimpleNamespace(rolling_ball=substituted)))is marker
        assert requests[0][0][0]is image and requests[0][1]==options
        actual=pipeline(importlib.import_module('skimagexx'))
        assert_default_parity('restoration.rolling_ball',image.dtype,actual,expected,image=image)
        assert image.tobytes()==before and xx.get_num_threads()==1
    finally:xx.set_fallback_enabled(fallback)
