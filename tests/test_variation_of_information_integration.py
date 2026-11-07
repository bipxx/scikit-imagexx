"""Public registration/hot-swap metadata plus one retained saved operation."""
import ast
import base64
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import zlib
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_vi_shared_source():
    cmake=(ROOT/'CMakeLists.txt').read_text();bindings=(ROOT/'src/bindings.cpp').read_text()
    assert 'target_sources(skimagexx PRIVATE src/variation_of_information.cpp)'in cmake
    assert 'src/bind_variation_of_information.cpp'in cmake and bindings.count('bind_variation_of_information(m);')==1
    namespace=ast.parse((ROOT/'python/skimagexx/metrics.py').read_bytes())
    names=next(ast.literal_eval(n.value)for n in namespace.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='__all__'for t in n.targets))
    assert names.count('variation_of_information')==1
    consumer=(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()
    assert 'installed_variation_of_information_consumer'in consumer
    assert 'variation_of_information_enabled_consumer.cpp'in consumer
    qualifier=(ROOT/'tools/public_qualification.py').read_text()
    assert "selected['public-VI-saved-linux']"in qualifier and "row['tests']==40"in qualifier
    assert "row['tests']==224"in qualifier and "row['tests']==18"in qualifier
    assert "'tests/fixtures/variation_of_information_saved_v1.hpp'"in qualifier
    assert "'tests/variation_of_information_current_counts_controls.hpp'"in qualifier


def test_vi_public_star_registry_and_single_core():
    import skimagexx as xx
    from skimagexx.coverage import native_functions,native_dtypes
    owned=importlib.import_module('skimagexx._variation_of_information');namespace={}
    exec('from skimagexx.metrics import *',namespace)
    assert xx.metrics.variation_of_information is namespace['variation_of_information']is owned.variation_of_information
    assert owned._core is xx._core and owned.contingency_table is xx.metrics.contingency_table
    assert native_functions().count('metrics.variation_of_information')==1
    assert native_dtypes('metrics.variation_of_information')==('int8','uint8','int16','uint16','int32','uint32','int64','uint64')
    assert inspect.isbuiltin(xx._core.variation_of_information_csr)and xx._core.variation_of_information_csr.__module__=='skimagexx._core'
    assert xx.get_num_threads()==1


def test_vi_swappable_public_saved_perfect_and_strict_checker():
    import numpy as np
    import skimagexx as xx
    from skimagexx.coverage import default_call,assert_default_parity,variation_of_information_default_error
    from label_pair_metrics_contract import cases,fixture
    row=next(x for x in json.loads((ROOT/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())['original_observations'][np.__version__]if x['id']=='variation_of_information/perfect')
    spec=next(x for x in cases()if x['id']==row['id']);args,options=fixture(spec,np)
    def pipeline(backend):return backend.metrics.variation_of_information(*args,**options)
    actual=pipeline(xx);expected_raw=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
    assert hashlib.sha256(expected_raw).hexdigest()==row['result']['raw_sha256']and actual.tobytes()==expected_raw
    expected=np.frombuffer(expected_raw,np.float64).copy()
    assert variation_of_information_default_error(actual,expected)==0.
    assert_default_parity('metrics.variation_of_information',np.dtype('int16'),actual,expected)
    for dtype in('uint8','float32','float64'):
        image=np.zeros((2,3),dtype=dtype);values,kwargs=default_call('metrics.variation_of_information',image)
        assert len(values)==2 and values[0].dtype==values[1].dtype==np.dtype(dtype)and kwargs=={}
    changed=expected.copy();changed[0]=1.
    with pytest.raises(AssertionError):variation_of_information_default_error(changed,expected)
    assert xx.get_num_threads()==1
