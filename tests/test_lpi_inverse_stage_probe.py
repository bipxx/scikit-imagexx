"""Inert observer/authentication tests; no scientific imports or execution."""
import importlib.util, json, sys, types, zipfile, io
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lpi_probe', ROOT / 'tools/lpi_inverse_stage_probe.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


@pytest.mark.parametrize('candidate', (False, True))
@pytest.mark.parametrize('fail', (False, True))
def test_observer_calls_existing_operations_once_and_restores_hooks(tmp_path, candidate, fail):
    events = []; records = {}
    class Token:
        def __rtruediv__(self, value): events.append('reciprocal'); return Token()
        def __setitem__(self, key, value): events.append('clip')
        def __mul__(self, other): events.append('product'); return Token()
    def real_fft(*args):
        events.append('ifft')
        if fail: raise RuntimeError('original FFT failure')
        return Token()
    tail = '_fft(G * F, False)' if candidate else 'fft.ifftn(G * F)'
    text = ("def _prepare():\n    events.append('prepare')\n    return F0, G0\n"
        "def _min_limit(x):\n    events.append('floor')\n    mask='floor-mask'\n"
        "def filter_inverse():\n    F, G = _prepare()\n    _min_limit(F)\n    F = 1 / F\n"
        "    mask='gain-mask'\n    F[mask] = F\n    return " + tail + "\n")
    source = tmp_path / 'unchanged.py'; source.write_text(text)
    module = types.ModuleType('inert_lpi'); module.__file__ = str(source)
    module.__dict__.update(events=events, F0=Token(), G0=Token(), _fft=real_fft, fft=types.SimpleNamespace(ifftn=real_fft))
    exec(compile(text, str(source), 'exec'), module.__dict__)
    code = module.filter_inverse.__code__; core = types.SimpleNamespace(__name__='inert_core') if candidate else None
    def capture(name, value):
        assert name not in records; assert value is not None; records[name] = value
    if fail:
        with pytest.raises(RuntimeError, match='original FFT failure'):
            p.observe(module, module.filter_inverse, capture, [], core)
    else:
        result = p.observe(module, module.filter_inverse, capture, [], core)
        assert records['IFFT'] is records['crop'] is result and set(records) == set(p.STAGES)
    assert events == ['prepare', 'floor', 'reciprocal', 'clip', 'product', 'ifft']
    assert module.filter_inverse.__code__ is code and source.read_text() == text
    assert module._fft is real_fft and module.fft.ifftn is real_fft
    assert sys.gettrace() is None and sys.getprofile() is None


def test_raw_snapshot_preserves_bytes_metadata_and_input(tmp_path):
    class Array:
        dtype = types.SimpleNamespace(str='<f8'); shape = (2,); strides = (8,); base = None
        flags = types.SimpleNamespace(owndata=True, writeable=False)
        def tobytes(self, order): assert order == 'C'; return b'unchanged words'
    array = Array(); row = p.capture_array(tmp_path, 'original-F', array)
    assert (tmp_path / row['file']).read_bytes() == array.tobytes('C')
    assert row['shape'] == [2] and row['dtype'] == '<f8' and row['writeable'] is False
    assert json.loads((tmp_path / 'original-F.json').read_text()) == row


def test_transport_reuses_old_function_in_isolated_new_donor_namespace():
    before = (ROOT / 'tools/matrix_runtime_certificate_probe.py').read_bytes()
    helper = p.transport()
    old = p.load('historical_probe_unchanged', ROOT / 'tools/matrix_runtime_certificate_probe.py')
    assert helper.ARTIFACT_ID == 12805154787 and old.ARTIFACT_ID == 12569527308
    assert helper.fetch_artifact.__code__.co_code == old.fetch_artifact.__code__.co_code
    assert (ROOT / 'tools/matrix_runtime_certificate_probe.py').read_bytes() == before


@pytest.mark.parametrize('fault', ('source', 'wheel', 'sdk', 'runtime'))
def test_prepare_rejects_wrong_or_incomplete_donor_before_extracting_runtime(tmp_path, fault):
    state = dict(source=dict(revision=p.DONOR_SOURCE), runtime_wheels=dict.fromkeys(map(str, range(14))))
    package = dict(source_revision=p.DONOR_SOURCE, wheel_sha256=p.WHEEL_SHA, source_archive_sha256=p.SOURCE_SHA,
        members_sha256=dict.fromkeys(map(str, range(155))), SDK_sha256=dict.fromkeys(map(str, range(71))))
    if fault == 'source': state['source']['revision'] = 'wrong'
    if fault == 'wheel': package['wheel_sha256'] = 'wrong'
    if fault == 'sdk': package['SDK_sha256'].pop('0')
    if fault == 'runtime': state['runtime_wheels'].pop('0')
    body = io.BytesIO()
    with zipfile.ZipFile(body, 'w') as z:
        z.writestr(p.PREFIX + 'state.json', json.dumps(state)); z.writestr(p.PREFIX + 'package.json', json.dumps(package))
    with pytest.raises(ValueError): p.prepare(body.getvalue(), tmp_path, object())
    assert not list((tmp_path / 'genuine-prefix').iterdir())


def test_plan_is_inert_and_does_not_prepare_a_local_runtime(capsys):
    out = ROOT / 'build/unit-plan-does-not-exist'
    assert not out.exists() and p.main(['plan', '--output', str(out)]) == 0 and not out.exists()
    record = json.loads(capsys.readouterr().out)
    assert record['cases'] == 4 and record['modes'] == ['normal', 'scalar']
    assert record['new_host'] is True and record['public_acceptance'] is False


def test_intermediate_type_mismatch_is_diagnostic_not_a_new_allowance(tmp_path):
    stages = dict(original={}, candidate={})
    for role in stages:
        for name in p.STAGES:
            file = role + '-' + name + '.bin'; (tmp_path / file).write_bytes(b'logical words')
            stages[role][name] = dict(dtype='<f8' if role == 'original' else '>f8', shape=[1], file=file)
    result = p.compare_stages(tmp_path, stages, object())
    assert result['first_bitwise_divergence'] == 'F'
    assert all(not r['same_dtype_shape'] for r in result['stages'])
    assert result['intermediate_differences_are_diagnostics_not_new_allowances'] is True
