"""Shared public aliases, saved tiny defaults and meaningful whole-output negatives."""
import ast
import importlib
import inspect
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
DATA=json.loads((ROOT/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())
APIS=('metrics.adapted_rand_error','metrics.contingency_table')


def test_label_pair_default_source_scope_without_science():
    source=(ROOT/'python/skimagexx/coverage.py').read_text();ast.parse(source)
    assert "labels=np.array([[0,1,1],[2,2,2]],dtype=image.dtype)"in source
    assert "np.array([[0,2,2],[1,1,2]],dtype=image.dtype)"in source
    assert 'label_pair_default_error(api,actual,expected);return'in source
    for name in ('test_api.py','test_dtype_aliases.py','test_dtype_metadata.py'):
        text=(ROOT/'tests'/name).read_text();assert "('metrics.adapted_rand_error','metrics.contingency_table')"in text
    assert 'label_pair_default_error(api,actual,expected)'in(ROOT/'benchmarks/coverage.py').read_text()
    assert 'third_party/scipy-sparse-LICENSE.txt'in(ROOT/'CMakeLists.txt').read_text()


@pytest.mark.parametrize('path',('python/skimagexx/_label_pair_metrics.py','tests/test_label_pair_rand_public.py'))
def test_ARE_metadata_context_rejects_tampered_source(path):
    from contingency_table_context import predecessor_bytes
    raw=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
    assert predecessor_bytes(ROOT,path)
    with pytest.raises(AssertionError):predecessor_bytes(ROOT,path,source=raw+b'\n# tampered\n')


@pytest.mark.parametrize('api',APIS)
def test_label_pair_public_default_saved_words_and_alias(api,monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx.coverage import default_call,assert_default_parity
    from label_pair_metrics_observer import observation
    public=importlib.import_module('skimagexx.metrics');name=api.split('.')[1]
    module=importlib.import_module('skimagexx._label_pair_metrics'if name=='adapted_rand_error'else'skimagexx._contingency_table')
    original=importlib.import_module('skimage.metrics')
    def deny(*a,**k):raise AssertionError('Original metric forbidden')
    for n in ('adapted_rand_error','contingency_table','variation_of_information'):monkeypatch.setattr(original,n,deny)
    assert getattr(public,name)is getattr(module,name)and public.__all__.count(name)==1
    image=np.empty((9,11),np.int16);arguments,options=default_call(api,image)
    before=[a.tobytes()for a in arguments];result=getattr(public,name)(*arguments,**options)
    case='adapted_rand_error/perfect'if name=='adapted_rand_error'else'contingency_table/duplicates'
    row=next(r for r in DATA['original_observations'][np.__version__]if r['id']==case)
    assert observation(result,np)==row['result']
    second=getattr(public,name)(*arguments,**options);assert_default_parity(api,image.dtype,result,second,image=image)
    assert [a.tobytes()for a in arguments]==before


@pytest.mark.parametrize('fault',('signedzero','dtype','container'))
def test_label_pair_scalar_checker_negative(fault):
    import numpy as np
    from skimagexx.coverage import label_pair_default_error
    expected=(np.float64(0),np.float64(1),np.float64(1));actual=expected
    if fault=='signedzero':actual=(np.float64(-0.),*expected[1:])
    elif fault=='dtype':actual=tuple(np.float32(a)for a in expected)
    else:actual=list(expected)
    with pytest.raises(AssertionError):label_pair_default_error(APIS[0],actual,expected)


@pytest.mark.parametrize('fault',('identity','word','cache','dtype'))
def test_label_pair_CSR_checker_negative(fault):
    import numpy as np
    from skimagexx._contingency_table import contingency_table
    from skimagexx.coverage import label_pair_default_error
    truth=np.array([[0,1,1],[2,2,2]],np.int16);test=np.array([[0,2,2],[1,1,2]],np.int16)
    expected=contingency_table(truth,test);actual=contingency_table(truth,test)
    if fault=='identity':actual=expected
    elif fault=='word':actual.data[0]+=1
    elif fault=='cache':actual.has_sorted_indices=False
    else:actual.data=actual.data.astype(np.float32)
    with pytest.raises(AssertionError):label_pair_default_error(APIS[1],actual,expected)
