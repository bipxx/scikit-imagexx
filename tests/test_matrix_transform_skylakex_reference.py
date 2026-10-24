"""Inert source/admission/census guards; no numerical dependency imports."""
import ast,hashlib,importlib.util,json,sys,types
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('skx_original_controller',ROOT/'tools/matrix_transform_skylakex_reference.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
FIELDS=dict(P=192,Q=384,R=10688,unroll_m=16,unroll_n=2,unroll_mn=32)


def test_live_schema_and_every_required_boundary_are_included():
    for k in (3,4,5):
        values=set(c.boundary_sizes(FIELDS,k))
        assert set(range(1,27))<=values
        for field in ('P','R'):
            for center in (FIELDS[field],2*FIELDS[field]):
                assert {center-1,center,center+1}<=values
        assert {15,16,17,31,32,33,47,48,49,63,64,65}<=values
        cut=1_000_000//(k*k);assert {cut-1,cut,cut+1}<=values


@pytest.mark.parametrize('key,value',(('P',4),('Q',4),('R',0),('unroll_n',4),('unroll_m',8),('unroll_mn',16),('P',True)))
def test_unqualified_live_fields_fail_closed(key,value):
    with pytest.raises(ValueError):c.validate_fields(dict(FIELDS,**{key:value}))


def test_complete_large_boundary_set_is_rejected_not_trimmed():
    fields=dict(FIELDS,R=65536)
    with pytest.raises(ValueError,match='no case filtering'):c.boundary_sizes(fields,3)


def synthetic_source():
    spec=dict(schema='matrix-general-finite-cohorts-v2',include_literal_base=True,cases=[dict(id=str(i)) for i in range(10)],rows=list(range(1,16)),layouts=['C','F','reverse','step','unaligned'],row_extension=0.03125)
    return ('// MATRIX_COHORTS_BEGIN\n/*\n'+json.dumps(spec)+'\n*/').encode(),b'{}'


def test_exact_old_source_pins_and_full4000_inventory(monkeypatch):
    cpp,matrix=synthetic_source();monkeypatch.setattr(c,'COHORT_SHA',c.H(cpp));monkeypatch.setattr(c,'MATRIX_SHA',c.H(matrix))
    spec,_=c.source_inputs(cpp,matrix)
    rows=[row for row in c.case_specs(spec,FIELDS) if row['family']=='legacy4000']
    assert len(rows)==4000 and len({(r['case']['id'],r['n'],r['pl'],r['ml']) for r in rows})==4000
    with pytest.raises(ValueError,match='source differs'):c.source_inputs(cpp+b'\n',matrix)
    with pytest.raises(ValueError,match='374'):c.source_inputs(cpp,b'{"changed":true}')


def test_boundary_data_are_separate_fixed_finite_definitions():
    rows=c.adversarial_cases();assert len(rows)==12 and len({r['id'] for r in rows})==12
    for row in rows:
        d=len(row['coords'][0]);assert d in (2,3,4)
        assert len(row['matrix'])==d+1 and all(len(a)==d+1 for a in row['matrix'])
    assert any(float.fromhex('0x1p-53') in row['matrix'][0] for row in rows)
    assert c.COHORT_SHA=='466d7645d841dfbbdbaf9c880d72a73f458af528299e2962493ce63e32cc2eb0'
    assert c.MATRIX_SHA=='6622a08fb4b420588d22325b3dc72fd053a7e4d689a4ed7ef88fe10fbc44e0f2'


def env():return dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')


@pytest.mark.parametrize('name',c.FORBIDDEN)
def test_any_CPU_forcing_key_even_empty_refused(name):
    with pytest.raises(ValueError,match='forcing'):c.validate_environment(dict(env(),**{name:''}))


@pytest.mark.parametrize('name',('LD_PRELOAD','LD_AUDIT','MATRIX_TRANSFORM_PRIVATE_BACKEND','SKIMAGEXX_MAX_TREE_CORE'))
def test_interposition_or_private_selectors_are_not_sanitized(name):
    with pytest.raises(ValueError):c.validate_environment(dict(env(),**{name:'unknown'}))


def test_read_span_requires_actual_owner_page_and_image_bounds():
    query=dict(allocation_base=1000,state=0x1000,protect=4,type=0x1000000,region_base=1100,region_size=100)
    c.readable_image_span(1000,1000,1120,24,query)
    for q,address,size in ((dict(query,allocation_base=900),1120,24),(dict(query,protect=0x104),1120,24),(dict(query,state=0),1120,24),(dict(query,type=0x20000),1120,24),(dict(query,protect=0x10),1120,24),(query,1190,24),(query,900,24),(query,1120,0)):
        with pytest.raises(ValueError):c.readable_image_span(1000,1000,address,size,q)


@pytest.mark.parametrize('index,value',((0,'1.24.4'),(1,'0'*64),(2,b'Haswell'),(2,b'Cooperlake'),(3,4),(3,True),(4,'0'*64)))
def test_unknown_or_wrong_library_actual_runtime_refused(index,value):
    facts=['2.5.4',c.BLAS_SHA,b'SkylakeX',1,c.UMATH_SHA];facts[index]=value
    with pytest.raises(ValueError):c.runtime_scope(*facts)


def test_raw_buffer_storage_is_lossless_deduplicated_portable_and_bounded(tmp_path,monkeypatch):
    array=types.SimpleNamespace(nbytes=8,tobytes=lambda **kw:b'inert123',dtype=types.SimpleNamespace(str='<f8',metadata=None),shape=(1,),strides=(8,),ctypes=types.SimpleNamespace(data=4096),flags=types.SimpleNamespace(aligned=True,c_contiguous=True,f_contiguous=True,owndata=True,writeable=True))
    store=c.Buffers(tmp_path);first=store.array(array);second=store.array(array)
    assert first==second and len(store.records)==1
    row=store.records[first['buffer_sha256']];assert row['path']=='arrays/'+first['buffer_sha256']+'.gz'
    assert (tmp_path/row['path']).exists()
    (tmp_path/row['path']).write_bytes(b'corrupt')
    with pytest.raises(Exception):c.Buffers(tmp_path).array(array)
    other=tmp_path/'other';other.mkdir();monkeypatch.setattr(c,'MAX_BUFFER_RETAINED',0)
    with pytest.raises(ValueError,match='never omit cases'):c.Buffers(other).array(array)


def test_unchanged_original_call_trace_captures_actual_operands_without_copy():
    marker=object();params=object()
    def helper(matrix,points):
        points_h=marker
        new_points_h=marker
        return marker
    def public(points,matrix):return helper(matrix,points)
    code=helper.__code__;result,captured=c.capture_original(public,helper,marker,params)
    assert result is marker and captured['matrix'] is params
    assert captured['points_h'] is captured['new_points_h'] is marker
    assert helper.__code__ is code and sys.gettrace() is None


def test_actual_matmul_source_node_is_required_without_execution():
    source='def _apply_homogeneous(matrix, points):\n    new_points_h = points_h @ matrix.T\n    return new_points_h\n'
    c.source_matmul_node(source)
    with pytest.raises(ValueError):c.source_matmul_node(source.replace('matrix.T','matrix'))


def test_no_native_algorithm_build_or_timer_request_and_no_guard_change():
    path=ROOT/'tools/matrix_transform_skylakex_reference.py';tree=ast.parse(path.read_text())
    forbidden={'matrix_transform_run','matrix_transform_run_v2','set_num_threads','seterr','setbufsize','perf_counter','timeit','cache_clear'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','compile') for n in ast.walk(tree))
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x in ast.unparse(n) for x in ('numpy','skimage')) for n in tree.body)
    prepare=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'))
    assert prepare.index('fresh hosted preparation only')<prepare.index('fetch_artifact(')
    refs=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='references'))
    assert refs.index('live_fields(')<refs.index('case_specs(')<refs.index('capture_original(')
    assert 'native_acceptance=False' in refs and 'public_acceptance=False' in refs


def test_workflow_manual_no_install_compile_or_hidden_forcing():
    text=(ROOT/'.github/workflows/matrix-transform-skylakex-reference.yml').read_text()
    for token in ('workflow_dispatch:',"python-version: '3.13.16'",'if: always()','actions: read','source-inputs/**','reports/**'):
        assert token in text
    for token in ('pip install','cmake','ninja','OPENBLAS_CORETYPE:','gh workflow run','genuine-prefix/**'):
        assert token not in text
