"""Inert authorization/diagnostic tests; synthetic modules never compute science or touch FP."""
import builtins
import datetime
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import struct
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'tools/diagnostics/weighted_contraction'
DRIVER = DIRECTORY/'run_retained_hue_witness_v1.py'
PINS = {
    'retained_hue_graph_v2.cpp': 'dd8dd59f31f0d9a51bcf6f95f735a4f3965010a0e6486320c9c5989781720fef',
    'run_retained_hue_witness_v1.py': '9ec911d0c04e88c9178e02a3d0b7fb4fa0fb9438777f86bf78c9e92fe6df8e94',
    'entry-UNBOUND-v2.json': '83f31987b65ee379b55e05bb4b9347d9f089f32019daa074bd6f5a8b67f0867f',
    'input-UNBOUND-v2.json': 'd1096586c2c091e416ea7142229e65296bc05ccede893193808f69d7705d4a56',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value)+'\n', encoding='utf-8')
    return {'path': str(path), 'sha256': digest(path)}


def load_driver():
    spec = importlib.util.spec_from_file_location('weighted_witness_under_test', DRIVER)
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__builtins__'] = dict(vars(builtins))
    spec.loader.exec_module(module)
    return module


def binding(tmp_path, monkeypatch):
    module = load_driver()
    variants = {}
    for role in ('baseline', 'strict'):
        path = tmp_path/(role+'.so')
        path.write_bytes(('INERT '+role).encode())
        variants[role] = {'path': str(path), 'sha256': digest(path)}
    original = tmp_path/'delta_e.py'
    original.write_text('def get_dH2(first, second, channel_axis=0):\n'
                        '    return observe(first, second, channel_axis)\n', encoding='utf-8')
    helper = tmp_path/'fenv.json'
    helper.write_text('{}', encoding='utf-8')
    entry = {'status': 'PASS_WEIGHTED_HUE_ENTRY_AND_CODEGEN_V1',
             'source_revision': 'a'*40, 'production_epoch': module.EPOCH,
             'runtime_versions': dict(module.VERSIONS), 'os_version': '26.04', 'glibc': '2.43',
             'baseline_has_contraction': True, 'strict_has_contraction': False,
             'sole_change_is_contract_off': True, 'baseline_already_contract_off': False,
             'cpp_sha256': module.CPP_SHA256, 'variants': variants}
    now = datetime.datetime.now(datetime.timezone.utc)
    pins = dict(module.SOURCE_PINS)
    for name in ('tests/util_views_fenv.py', 'coverage/contracts/util-views-fenv-v1.json'):
        pins[name] = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    go = {'status': module.STATUS, 'production_epoch': module.EPOCH,
          'source_revision': 'a'*40, 'driver_sha256': digest(DRIVER),
          'maximum_invocations': 1, 'retries': 0, 'caps': dict(module.CAPS),
          'issued_utc': (now-datetime.timedelta(seconds=5)).isoformat(),
          'deadline_utc': (now+datetime.timedelta(minutes=5)).isoformat(),
          'entry': write_json(tmp_path/'entry.json', entry),
          'cpp_source': {'path': str(DIRECTORY/'retained_hue_graph_v2.cpp'),
                         'sha256': module.CPP_SHA256},
          'source_root': str(ROOT), 'canonical_source_pins': pins, 'variants': variants,
          'original_source': {'path': str(original), 'sha256': digest(original)},
          'fenv_receipt': {'path': str(helper), 'sha256': digest(helper)},
          'output': str(tmp_path/'out')}
    for name in ('NPY_DISABLE_CPU_FEATURES', 'SKIMAGEXX_DISABLE_SIMD'):
        monkeypatch.delenv(name, raising=False)
    return module, go, entry


def invoke(module, go, tmp_path, monkeypatch):
    path = tmp_path/'go.json'
    pin = write_json(path, go)
    monkeypatch.setattr(sys, 'argv', [str(DRIVER), '--binding', str(path),
                                    '--binding-sha256', pin['sha256']])
    module.main()


def test_imported_sources_are_exact_frozen_bytes():
    for name, expected in PINS.items():
        assert digest(DIRECTORY/name) == expected
    assert 'visibility("default")' in (DIRECTORY/'retained_hue_graph_v2.cpp').read_text()
    for name in ('entry-UNBOUND-v2.json', 'input-UNBOUND-v2.json'):
        record = json.loads((DIRECTORY/name).read_text())
        assert record['status'].startswith('UNBOUND')


@pytest.mark.parametrize('fault,match', [
    ('unbound', 'UNBOUND witness'), ('expired', 'not current'),
    ('budget', 'budgets differ'), ('cpp', 'Primitive CPP differs'),
    ('runtime', 'Host/runtime'), ('already-off', 'sole-change'),
    ('not-sole-change', 'sole-change'), ('mask', 'CPU masks'),
    ('driver', 'Witness source differs'), ('entry-pin', 'Receipt pin differs'),
])
def test_rejections_precede_output_allocation_and_scientific_import(tmp_path, monkeypatch, fault, match):
    module, go, entry = binding(tmp_path, monkeypatch)
    if fault == 'unbound':
        go['status'] = 'UNBOUND'
    elif fault == 'expired':
        go['issued_utc'] = '2020-01-01T00:00:00+00:00'
        go['deadline_utc'] = '2020-01-01T00:10:00+00:00'
    elif fault == 'budget':
        go['caps']['original'] = 2
    elif fault == 'cpp':
        go['cpp_source']['sha256'] = '0'*64
    elif fault == 'runtime':
        entry['runtime_versions']['numpy'] = '2.0.0'
    elif fault == 'already-off':
        entry['baseline_already_contract_off'] = True
    elif fault == 'not-sole-change':
        entry['sole_change_is_contract_off'] = False
    elif fault == 'mask':
        monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES', 'AVX2')
    elif fault == 'driver':
        go['driver_sha256'] = '0'*64
    if fault in ('runtime', 'already-off', 'not-sole-change'):
        go['entry'] = write_json(tmp_path/'entry.json', entry)
    elif fault == 'entry-pin':
        go['entry']['sha256'] = '0'*64
    attempted = []
    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name.split('.')[0] in {'numpy', 'scipy', 'skimage', 'skimagexx', 'ctypes'}:
            attempted.append(name)
            raise AssertionError('Unexpected body/import: '+name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(module.__dict__['__builtins__'], '__import__', deny)
    with pytest.raises(RuntimeError, match=match):
        invoke(module, go, tmp_path, monkeypatch)
    assert attempted == []
    assert not Path(go['output']).exists()


class Scalar(float):
    shape = ()
    dtype = 'float64'


class Lab:
    shape = (3,)
    strides = (8,)

    def __init__(self, pointer):
        self.flags = types.SimpleNamespace(writeable=True)
        self.__array_interface__ = {'data': (pointer, False)}

    def __getitem__(self, index):
        return (50.0, 1.0, 1.0)[index]

    def tobytes(self):
        return struct.pack('<3d', 50.0, 1.0, 1.0)


class Colors:
    def __init__(self, pointer=101):
        self.pointer = pointer

    def __mul__(self, value):
        assert value == 200
        return self

    def __sub__(self, value):
        assert value == 100
        return self

    def astype(self, dtype):
        assert dtype is Scalar
        return self

    def __getitem__(self, value):
        if isinstance(value, slice):
            assert value == slice(None, None, -1)
            return Colors(202)
        assert value == 33
        return Lab(self.pointer)

    def copy(self):
        return Colors(self.pointer)


class Env:
    def __init__(self, raw=None):
        self.raw = bytearray(32 if raw is None else raw)

    @classmethod
    def from_buffer_copy(cls, raw):
        return cls(raw)

    def __bytes__(self):
        return bytes(self.raw)

    @property
    def control(self):
        return struct.unpack_from('<H', self.raw, 0)[0]

    @property
    def tags(self):
        return struct.unpack_from('<H', self.raw, 8)[0]

    @property
    def mxcsr(self):
        return struct.unpack_from('<I', self.raw, 28)[0]


def fake_runtime(module, monkeypatch, go, *, snapshot_fault=None, restore_fault=False,
                 original_fault=False, baseline_value=1.0):
    """All data below are synthetic constants/counters; no floating API is called."""
    initial = bytearray(32)
    struct.pack_into('<H', initial, 0, 0x37f)
    struct.pack_into('<H', initial, 8, 0xffff)
    struct.pack_into('<I', initial, 12, 0x12345678)
    struct.pack_into('<I', initial, 28, 0x1f80)
    state = types.SimpleNamespace(raw=bytes(initial), initial=bytes(initial), calls=[],
                                  saves=0, restores=0, backend=0)

    def observe(role):
        state.calls.append(role)
        raw = bytearray(state.raw)
        struct.pack_into('<H', raw, 4, 0x20)
        struct.pack_into('<I', raw, 28, 0x1fa0)
        state.raw = bytes(raw)
        if role == 'original' and original_fault:
            raise ValueError('first-target-fault')
        return Scalar(0.0) if role == 'original' else baseline_value if role == 'baseline' else 0.0

    def save(value):
        state.saves += 1
        if state.saves == snapshot_fault:
            raise RuntimeError('post-snapshot-fault')
        value.raw[:] = state.raw
        return 0

    def restore(value, length):
        state.restores += 1
        assert length == 32
        if restore_fault:
            return -3
        state.raw = bytes(value)
        return 0

    def backend():
        state.backend += 1
        return types.SimpleNamespace(fegetenv=save), types.SimpleNamespace(reference_fenv_restore=restore)

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.FEnv = Env
            module.backend = backend

    real_spec = importlib.util.spec_from_file_location

    def spec(name, path, *args, **kwargs):
        if name == 'weighted_fenv':
            return importlib.util.spec_from_loader(name, Loader())
        return real_spec(name, path, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'spec_from_file_location', spec)
    numpy = types.ModuleType('numpy')
    numpy.__version__ = '1.24.4'
    numpy.float64 = Scalar
    numpy.dtype = lambda value: 'float64'
    numpy.random = types.SimpleNamespace(default_rng=lambda seed: types.SimpleNamespace(
        random=lambda shape: Colors() if seed == 32 and shape == (67, 3) else None))
    numpy.geterr = lambda: {'over': 'warn', 'under': 'ignore', 'divide': 'warn', 'invalid': 'warn'}
    numpy.geterrcall = lambda: None
    numpy.getbufsize = lambda: 8192
    original = types.ModuleType('skimage.color.delta_e')
    original.__file__ = go['original_source']['path']
    exec(compile(Path(original.__file__).read_text(), original.__file__, 'exec'),
         {'observe': lambda *args: observe('original')}, original.__dict__)
    skimage = types.ModuleType('skimage')
    skimage.__path__ = []
    color = types.ModuleType('skimage.color')
    color.__path__ = []
    color.delta_e = original
    skimage.color = color
    ctypes = types.ModuleType('ctypes')
    ctypes.c_double = object()
    ctypes.byref = lambda value: value
    ctypes.sizeof = lambda value: 32

    class Function:
        def __init__(self, role):
            self.role = role

        def __call__(self, *args):
            assert args == (1.0, 1.0, 1.0, 1.0)
            return observe(self.role)

    ctypes.CDLL = lambda path: types.SimpleNamespace(weighted_hue_graph_v1=Function(
        'baseline' if path == go['variants']['baseline']['path'] else 'strict'))
    fake_importlib = types.SimpleNamespace(util=importlib.util,
        metadata=types.SimpleNamespace(version=lambda name: {
            'scipy': '1.15.3', 'scikit-image': '0.26.0'}[name]))
    imports = {'numpy': numpy, 'skimage': skimage, 'skimage.color': color,
               'ctypes': ctypes, 'platform': types.SimpleNamespace(python_version=lambda: '3.11.17'),
               'importlib.metadata': fake_importlib}
    real_import = builtins.__import__

    def inert_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in imports:
            return imports[name]
        return real_import(name, globals, locals, fromlist, level)

    # Only this loaded driver's builtins see the stubs; pytest/plugins remain untouched.
    monkeypatch.setitem(module.__dict__['__builtins__'], '__import__', inert_import)
    monkeypatch.setenv('UTIL_VIEWS_FENV_MODE', 'source-built-v1')
    monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT', go['fenv_receipt']['path'])
    monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT_SHA256', go['fenv_receipt']['sha256'])
    return state


@pytest.mark.parametrize('contraction,baseline_value,expected', [
    (True, 1.0, 'COMPLETE_CONTRACTION_WITNESS'),
    (False, 1.0, 'COMPLETE_INCONCLUSIVE_NUMERICAL_OBSERVATION'),
    (False, 0.0, 'COMPLETE_INCONCLUSIVE_NUMERICAL_OBSERVATION'),
])
def test_three_fixed_observations_and_full32_even_without_contraction(
        tmp_path, monkeypatch, contraction, baseline_value, expected):
    module, go, entry = binding(tmp_path, monkeypatch)
    entry['baseline_has_contraction'] = contraction
    go['entry'] = write_json(tmp_path/'entry.json', entry)
    state = fake_runtime(module, monkeypatch, go, baseline_value=baseline_value)
    invoke(module, go, tmp_path, monkeypatch)
    result = json.loads((Path(go['output'])/'result-v1.json').read_text())
    assert result['status'] == expected and result['completed']
    assert state.calls == ['original', 'baseline', 'strict']
    assert state.saves == 10 and state.restores == 3 and state.backend == 1
    assert result['counts'] == module.CAPS
    assert result['last_restore_verified'] and state.raw == state.initial
    assert set(result['outputs']) == {'original', 'baseline', 'strict'}
    for frame in result['FP'].values():
        assert frame['before'] == frame['restored'] == state.initial.hex()
        assert frame['after'] != frame['before']


def test_returned_output_survives_post_target_snapshot_failure(tmp_path, monkeypatch):
    module, go, _ = binding(tmp_path, monkeypatch)
    state = fake_runtime(module, monkeypatch, go, snapshot_fault=3)
    with pytest.raises(RuntimeError, match='post-snapshot-fault'):
        invoke(module, go, tmp_path, monkeypatch)
    result = json.loads((Path(go['output'])/'result-v1.json').read_text())
    assert result['outputs']['original']['raw64'] == struct.pack('<d', 0.0).hex()
    assert result['primary']['message'] == 'post-snapshot-fault'
    assert result['last_restore_verified'] and not result['completed']
    assert state.calls == ['original'] and state.saves == 4 and state.restores == 1


def test_first_target_error_survives_secondary_snapshot_failure(tmp_path, monkeypatch):
    module, go, _ = binding(tmp_path, monkeypatch)
    state = fake_runtime(module, monkeypatch, go, original_fault=True, snapshot_fault=3)
    with pytest.raises(ValueError, match='first-target-fault'):
        invoke(module, go, tmp_path, monkeypatch)
    result = json.loads((Path(go['output'])/'result-v1.json').read_text())
    assert result['primary']['message'] == 'first-target-fault'
    assert any(row['message'] == 'post-snapshot-fault' for row in result['secondary'])
    assert result['last_restore_verified'] and state.calls == ['original']


def test_restore_failure_is_not_claimed_as_complete(tmp_path, monkeypatch):
    module, go, _ = binding(tmp_path, monkeypatch)
    state = fake_runtime(module, monkeypatch, go, restore_fault=True)
    with pytest.raises(RuntimeError, match='Full32 restore failed'):
        invoke(module, go, tmp_path, monkeypatch)
    result = json.loads((Path(go['output'])/'result-v1.json').read_text())
    assert result['last_restore_verified'] is False and not result['completed']
    assert result['outputs']['original']['raw64'] is not None
    assert state.calls == ['original'] and state.restores == 1


def test_writer_failure_emits_postrestore_outputs_and_actual_counts(tmp_path, monkeypatch, capsys):
    module, go, _ = binding(tmp_path, monkeypatch)
    state = fake_runtime(module, monkeypatch, go)
    real_open = Path.open
    target = Path(go['output'])/'result-v1.json'

    def fail_result(path, *args, **kwargs):
        if path == target:
            raise OSError('result-writer-fault')
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', fail_result)
    with pytest.raises(OSError, match='result-writer-fault'):
        invoke(module, go, tmp_path, monkeypatch)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['status'] == 'RESULT_WRITE_FAILED'
    assert result['primary']['message'] == 'result-writer-fault'
    assert result['counts'] == module.CAPS and result['last_restore_verified']
    assert set(result['outputs']) == {'original', 'baseline', 'strict'}
    assert state.raw == state.initial and state.saves == 10 and state.restores == 3
