"""AST/source-only checks: no scientific imports, fixtures, cores or timers."""
import ast
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).parents[1]


def test_half_stain_selector_preserves_legacy_module_and_history():
    contract = json.loads((ROOT/'coverage/contracts/combine-stains-half-benchmark.json').read_bytes())
    archived = (ROOT/'coverage/contracts/combine-stains-legacy-benchmark-source-v1.json').read_bytes()
    assert hashlib.sha256(archived).hexdigest() == '03c0890c5064660c0ddcb0d8a4b0990a5586ebbf2c66a6361faf48dbd38972aa'
    baseline = json.loads(archived)
    assert baseline['source_commit'] == 'cff831a1f7f4df65764d4a405ec00eab4efa57c0'
    assert baseline['source_path'] == 'benchmarks/stains.py'
    assert baseline['before_commit'] == '13ce8f08c7981ae4d62aa1140504c8e15b67dc96'
    assert baseline['encoding'] == 'utf-8' and baseline['newline'] == 'LF'
    source = baseline['source_text'].encode('utf-8')
    assert b'\r' not in source
    assert hashlib.sha256(source).hexdigest() == baseline['source_sha256'] == contract['legacy_stains_sha256']
    # AST dumps contain interpreter schema fields. Parse both immutable baseline
    # and reduced candidate here, so their equality tests code, not Python version.
    original = ast.parse(source)
    original_main = next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    tree = ast.parse((ROOT/'benchmarks/stains.py').read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name == '_half_main')]
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    kept, arguments, branches = [], 0, 0
    for node in main.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr == 'add_argument' and isinstance(node.value.args[0],ast.Constant) and node.value.args[0].value in ('--half-only','--deadline-utc'):
            arguments += 1
        elif isinstance(node,ast.If) and isinstance(node.test,ast.Attribute) and isinstance(node.test.value,ast.Name) and node.test.value.id == 'args' and node.test.attr == 'half_only':
            assert len(node.body) == 1 and isinstance(node.body[0],ast.Return)
            assert isinstance(node.body[0].value,ast.Call) and node.body[0].value.func.id == '_half_main'
            branches += 1
        else:
            kept.append(node)
    assert arguments == 2 and branches == 1
    main.body = kept
    assert ast.dump(main,include_attributes=False) == ast.dump(original_main,include_attributes=False)
    assert ast.dump(tree,include_attributes=False) == ast.dump(original,include_attributes=False)
    assert hashlib.sha256((ROOT/'benchmarks/stain-parameters.json').read_bytes()).hexdigest() == contract['legacy_report_sha256']


def test_half_stain_frozen_cases_policy_and_bounds():
    contract = json.loads((ROOT/'coverage/contracts/combine-stains-half-benchmark.json').read_bytes())
    assert hashlib.sha256((ROOT/'coverage/contracts/combine-stains-half.json').read_bytes()).hexdigest() == contract['reference_contract_sha256']
    assert contract['size'] == 512 and contract['threads'] == 1 and contract['repeat'] == 7
    assert contract['raw_samples'] == 56 and contract['warmup_groups'] == 8
    assert contract['target_ms'] == 100 and contract['warmup_min_ms'] == 250
    assert contract['numerical_policy']['rtol'] == contract['numerical_policy']['atol'] == 2e-6
    assert contract['max_input_elements'] == 512*512*3
    assert len(contract['cases']) == len({c['id'] for c in contract['cases']}) == 4
    assert [(c['matrix_dtype'],c['layout'],len(c['matrix'][0])) for c in contract['cases']] == [('float32','C',3),('float64','C',3),('float32','C',7),('float32','reverse',1)]
    for case in contract['cases']:
        assert case['api'] == 'color.combine_stains'
        matrix = case['matrix']
        assert len(matrix) == 3 and len({len(row) for row in matrix}) == 1
        assert all(abs(v) <= 1 for row in matrix for v in row)
        assert sum(map(len,matrix)) <= contract['max_matrix_elements'] == 21
        assert 512*512*len(matrix[0]) <= contract['max_output_elements'] == 2000000
    assert 3 * contract['dataset']['max_abs_input'] * 14 == contract['dataset']['max_abs_exponent'] == 31.5
    for name,pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == pin


def test_half_stain_sampler_witness_state_and_hardware_source():
    contract = json.loads((ROOT/'coverage/contracts/combine-stains-half-benchmark.json').read_bytes())
    for filename,key in (('tools/montage_sustained.py','method_sha256'),('benchmarks/run.py','legacy_method_sha256')):
        source = (ROOT/filename).read_text(encoding='utf-8')
        method = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'measure')
        text = ''.join(source.splitlines(keepends=True)[method.lineno-1:method.end_lineno])
        assert hashlib.sha256(text.encode()).hexdigest() == contract[key]
    source = (ROOT/'benchmarks/stains.py').read_text(encoding='utf-8')
    half = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == '_half_main')
    text = ast.get_source_segment(source,half)
    required = ["from montage_sustained import measure as sustained_measure", "args.repeat,100,journal=journal", 'deadline=deadline',
        "args.size == 512 and args.threads == [1] and args.repeat == 7", "calls == ['stain_transform']", "casts == [np.dtype(np.float16).str]",
        'actual_C_returns=calls,half_normalizations=casts,native_accepted=True',
        'np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)',
        'atol *= max(1.,float(np.max(np.sum(np.abs(matrix),axis=0))))',
        'contract_path.read_bytes() == contract_raw', 'reference_path.read_bytes() == reference_raw',
        'not args.output.exists() and not journal.exists()', 'snapshot((image,matrix)) == input_before',
        "'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'", 'original_output_dtype=expected.dtype.name']
    assert all(item in text for item in required)
    template = ast.parse((ROOT/'benchmarks/line_aa.py').read_text(encoding='utf-8'))
    old = next(n for n in template.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    blocks = []
    for node in (half,old):
        index = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'cpu_model' for t in n.targets))
        blocks.append([ast.dump(n,include_attributes=False) for n in node.body[index:index+4]])
    assert blocks[0] == blocks[1]
