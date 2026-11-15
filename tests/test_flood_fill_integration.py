"""Current single-core flood integration; saved algorithm proof is separate."""
import ast
import inspect
import importlib
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_flood_shared_source():
    c=(ROOT/'CMakeLists.txt').read_text();b=(ROOT/'src/bindings.cpp').read_text()
    assert 'target_sources(skimagexx PRIVATE src/flood_fill.cpp)'in c and 'src/bind_flood_fill.cpp'in c
    assert b.count('bind_flood_fill(m);')==1
    for n in ('morphology','segmentation'):
        source=(ROOT/('python/skimagexx/'+n+'.py')).read_text()
        assert 'from ._flood_fill import flood, flood_fill'in source and "__all__ += ['flood','flood_fill']"in source
    q=(ROOT/'tools/public_qualification.py').read_text()
    assert "selected['public-flood-saved-linux']"in q and "row['tests']==100"in q
    assert 'FLOOD_FILL_CALLER_RECORD_DIR'in q and "'tests/fixtures/flood_fill_sdk_original_v1.inc'"in q
    assert 'installed_flood_fill_consumer'in(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()
    coverage=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'))
    assert all(a in coverage['native_functions']()for a in coverage['FLOOD_FILL_FUNCTIONS'])
    assert 'image.min full-image'in coverage['flood_fill_contract']()['providers']


def test_flood_single_core_alias_star_default_and_pipeline():
    import numpy as np
    import skimagexx as xx
    from skimagexx.coverage import default_call,FLOOD_FILL_FUNCTIONS,assert_default_parity
    owned=importlib.import_module('skimagexx._flood_fill');assert owned._core is xx._core
    for name in ('flood_mask','flood_assign','crop_copy_into','get_num_threads','set_num_threads'):
        assert inspect.isbuiltin(getattr(xx._core,name))and getattr(xx._core,name).__module__=='skimagexx._core'
    assert xx.get_num_threads()==1
    for module in ('morphology','segmentation'):
        values={};exec('from skimagexx.'+module+' import *',values)
        assert values['flood']is xx.morphology.flood is xx.segmentation.flood is owned.flood
        assert values['flood_fill']is xx.morphology.flood_fill is xx.segmentation.flood_fill is owned.flood_fill
    before_fallback=xx.set_fallback_enabled(False)
    try:
        for api in FLOOD_FILL_FUNCTIONS:
            image=np.array([[1,1],[1,2]],dtype=np.uint8);before=image.copy();args,options=default_call(api,image)
            def pipeline(backend):return getattr(getattr(backend,api.split('.')[0]),api.split('.')[1])(*args,**options)
            actual=pipeline(xx)
            if api.endswith('.flood'):
                # Source returns a bool view cropped from padded uint8 flags.
                flags=np.zeros((4,4),dtype=np.uint8)
                flags[1:3,1:3]=[[1,1],[1,0]]
                expected=flags[1:3,1:3].view(bool)
                assert expected.strides==(4,1)and not expected.flags.owndata
                assert np.shares_memory(expected,flags)and not np.shares_memory(expected,image)
            else:expected=np.array([[1,1],[1,2]],dtype=np.uint8)
            assert_default_parity(api,image.dtype,actual,expected,image=image)
            assert image.tobytes()==before.tobytes()and xx.get_num_threads()==1
    finally:xx.set_fallback_enabled(before_fallback)
