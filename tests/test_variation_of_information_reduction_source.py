"""Pure changed-source gates; no numerical or scientific imports."""
import hashlib
import json
import pytest
from variation_of_information_reduction_contract import ROOT, CONTEXT, historical_vi_source


def test_exact_source_and_unchanged_metric_dependency():
    context=json.loads(CONTEXT.read_bytes())
    for name,pin in context['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    assert hashlib.sha256((ROOT/'src/metrics.cpp').read_bytes()).hexdigest()==context['metrics_cpp_unchanged_raw_sha256']
    historical_vi_source((ROOT/'src/variation_of_information.cpp').read_bytes())


@pytest.mark.parametrize('mutate',(lambda b:b+b' ',lambda b:b.replace(b'std::fma',b'std::pow',1)))
def test_projection_rejects_changed_source(mutate):
    with pytest.raises(AssertionError):historical_vi_source(mutate((ROOT/'src/variation_of_information.cpp').read_bytes()))


def test_reused_pairwise_graph_is_exact():
    source=(ROOT/'src/variation_of_information.cpp').read_text()
    metric=(ROOT/'src/metrics.cpp').read_text()
    begin='template<class T,class F> T pairwise('
    graph=metric[metric.index(begin):metric.index('template<class T> MetricSums merge(')]
    assert source.count(graph)==1
    assert 'if(n<=128)return pairwise<double>'in source
    assert 'auto half=n/2;half-=half%8;'in source
    assert 'return get(0)+reduction_tree(n-1,'in source


def test_dot_and_boundary_witnesses_are_generic():
    source=(ROOT/'src/variation_of_information.cpp').read_text()
    assert 'const auto packet=n-n%16;'in source
    assert 'lanes[j]=std::fma(left(i+j),right(i+j),lanes[j]);'in source
    assert 'result+=source_dot_product(left(i),right(i))'in source
    assert 'get_num_threads()'in source and 'case_id'not in source
    control=(ROOT/'tests/test_variation_of_information_reduction_native.cpp').read_text()
    assert '127,128,129,255,256,257'in control


def test_saved_profile_rows_and_fp_are_preserved():
    saved=json.loads((ROOT/'tests/data/variation_of_information_large_saved_v1.json').read_bytes())
    assert set(saved['profiles'])=={'1.24.4','2.5.4'}
    for version,profile in saved['profiles'].items():
        rows=profile['rows'];assert len(rows)==4 and len({r['id']for r in rows})==4
        assert all(r['error']is None and not r['warnings']and not r['events']for r in rows)
        assert all(r['target_immediate']['before']['mxcsr']==(8098 if version=='1.24.4'else 8096)for r in rows)
        assert all(r['caller_restore']['exact32']for r in rows)
