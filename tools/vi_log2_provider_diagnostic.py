"""Bounded provider words only; import is inert and execution requires a root GO."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/vi-log2-provider-v2.json'
FIXTURE = ROOT / 'coverage/contracts/vi-log2-input-words-v1.json'
AUTHORITY = 'ROOT_AUTHORIZED_ONE_LOG2_PROVIDER_EXPERIMENT'
SOURCE = '16a5b50193125199ad26cf5a89acec5babf5290c'
LIMITS = dict(numpy_log2_array_calls_max=4, scalar_log2_values_max=128,
              scalar_log2_values_expected=112, VI_original_calls=0, VI_native_calls=0,
              timed_batches=0, retries=0, feature_children_max=2,
              FP_save_max=24, FP_restore_max=16, F1_ABI_queries_max=2,
              F1_glibc_queries_max=2, dlvsym_queries_max=1, dladdr_queries_max=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def inputs():
    contract = json.loads(CONTRACT.read_bytes())
    assert sha(FIXTURE) == contract['input_sha256']
    data = json.loads(FIXTURE.read_bytes())
    assert data['total_words'] == 112 and data['total_payload_bytes'] == 896
    raw = []
    for i, vector in enumerate(data['vectors']):
        assert vector['index'] == i and vector['dtype'] == '<f8'
        assert vector['shape'] == [56] and vector['stride_bytes'] == 8
        words = vector['words_uint64_hex']
        assert len(words) == 56
        assert all(len(w) == 16 and all(c in '0123456789abcdef' for c in w) for w in words)
        # Validate positive normal finite<1 using integer fields, without FP calls.
        assert all(0 < (int(w, 16) >> 52) < 1023 for w in words)
        payload = b''.join(int(w, 16).to_bytes(8, 'little') for w in words)
        assert hashlib.sha256(payload).hexdigest() == vector['payload_sha256']
        raw.append(payload)
    assert len(raw) == 2
    assert hashlib.sha256(b''.join(raw)).hexdigest() == data['concatenated_payload_sha256']
    return raw


def feature_map(np):
    module = np.core._multiarray_umath
    return dict(baseline=list(module.__cpu_baseline__), dispatch=list(module.__cpu_dispatch__),
                enabled={k: bool(v) for k, v in module.__cpu_features__.items()},
                numpy_version=np.__version__, numpy_binary=str(Path(module.__file__).resolve()))


def disabled_targets(features):
    assert 'AVX512_SKX' in features['dispatch'] and features['enabled'].get('AVX512_SKX') is True
    targets = sorted(n for n in features['dispatch'] if n.startswith('AVX512') and features['enabled'].get(n))
    assert 'AVX512_SKX' in targets and not set(targets) & set(features['baseline'])
    return targets


def check_intervention(base, disabled, targets):
    assert disabled['baseline'] == base['baseline'] and disabled['dispatch'] == base['dispatch']
    assert disabled['numpy_version'] == base['numpy_version']
    assert disabled['numpy_binary'] == base['numpy_binary']
    assert set(disabled['enabled']) == set(base['enabled'])
    assert targets == disabled_targets(base)
    for name, enabled in base['enabled'].items():
        assert disabled['enabled'][name] == (False if name in targets else enabled), name


def cpu_record():
    text = Path('/proc/cpuinfo').read_text(encoding='utf-8').split('\n\n', 1)[0]
    cpu = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
    stable = {k.strip(): v.strip() for k, v in cpu.items()}
    return {k: stable[k] for k in ('vendor_id','model name','cpu family','model','stepping','flags')}


def check_go(go, *, runner_hash, now):
    contract = json.loads(CONTRACT.read_bytes())
    assert go['status'] == AUTHORITY and go['limits'] == LIMITS
    assert go['runner_sha256'] == runner_hash and go['contract_sha256'] == sha(CONTRACT)
    assert go['input_sha256'] == sha(FIXTURE) == contract['input_sha256']
    assert go['source_revision'] == SOURCE and go['mode'] in ('private-baseline', 'hosted-feature')
    created, deadline = (datetime.fromisoformat(go[k]) for k in ('created_utc', 'deadline_utc'))
    assert created.utcoffset() is not None and created.utcoffset().total_seconds() == 0
    assert deadline.utcoffset() is not None and deadline.utcoffset().total_seconds() == 0
    assert 0 < (deadline-created).total_seconds() <= 7200 and created <= now < deadline
    assert Path(go['output_root']).is_absolute() and Path(go['source_root']).is_absolute()
    assert go['core_sha256'] == contract['core_sha256']
    for name, pin in contract['prepared_dependencies'].items():
        assert sha(Path(go['source_root']) / name) == pin, name
    assert sys.platform == 'linux' and platform.machine() == 'x86_64' and sys.byteorder == 'little'
    assert not sys.flags.optimize and sys.executable == go['interpreter']
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'))
    assert all(not os.environ.get(k) for k in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','OPENBLAS_CORETYPE','SKIMAGEXX_DISABLE_SIMD'))
    if go['mode'] == 'hosted-feature':
        assert sys.version == contract['genuine_python'] and go['runtime_sha256'] == contract['genuine_runtime_sha256']
        assert sha(go['runtime_path']) == go['runtime_sha256']
        runtime = json.loads(Path(go['runtime_path']).read_bytes())
        assert runtime['source']['revision'] == SOURCE and len(runtime['runtime']['sdk_sha256']) == 108
        assert runtime['runtime']['python'] == sys.version
        loader = contract['hosted_loader_environment']
        assert sys.base_prefix == go['interpreter_base_prefix'] == loader['base_prefix']
        allowed = sys.base_prefix + '/lib'
        assert allowed == go['allowed_LD_LIBRARY_PATH'] == loader['LD_LIBRARY_PATH']
        assert os.environ.get('LD_LIBRARY_PATH') == allowed
        assert os.environ.get('UTIL_VIEWS_FENV_MODE') == 'source-built-v1'
        assert sha(go['helper_binding_path']) == go['helper_binding_sha256']
        helper = json.loads(Path(go['helper_binding_path']).read_bytes())
        assert all(os.environ.get(k) == v for k, v in helper['env'].items())
    else:
        assert not os.environ.get('LD_LIBRARY_PATH')
        prior = go['private_binding_path']
        assert sha(prior) == contract['private_binding_sha256'] == go['private_binding_sha256']
        binding = json.loads(Path(prior).read_bytes())
        assert binding['kind'] == 'PRIVATE_LOCAL_31114_RETAINED_16CORE_NOT_GENUINE'
        assert binding['source_revision'] == SOURCE and binding['core']['sha256'] == go['core_sha256']
        assert sys.version == binding['python_version'] and sys.executable == binding['interpreter']
        assert os.environ.get('UTIL_VIEWS_FENV_MODE') == 'private-fixed-v1'
        assert os.environ.get('UTIL_VIEWS_FENV_HELPER') == binding['helper']['path']
        assert not os.environ.get('UTIL_VIEWS_FENV_RECEIPT')
    return contract


def semantic(raw):
    return dict(control=int.from_bytes(raw[:2], 'little'), status=int.from_bytes(raw[4:6], 'little'),
                mxcsr=int.from_bytes(raw[28:32], 'little'))


def seed_state(raw, expected):
    assert len(raw) == 32 and raw[8:10] == b'\xff\xff'
    value = bytearray(raw)
    for key, offset, width in (('control',0,2),('status',4,2),('mxcsr',28,4)):
        value[offset:offset+width] = expected[key].to_bytes(width, 'little')
    return bytes(value)


def observe(state, name, function, save, restore, seed, caller):
    """Snapshot immediately after the primitive, restore before any value codec."""
    row = dict(provider=name, before=restore(seed))
    state.append(row)
    try:
        row['value'] = function()
    finally:
        row['after'] = save()
        row['caller_restored'] = restore(caller)


def scalar_provider():
    import ctypes
    libm = ctypes.CDLL('libm.so.6')
    loader = ctypes.CDLL(None)
    loader.dlvsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    loader.dlvsym.restype = ctypes.c_void_p
    address = loader.dlvsym(libm._handle, b'log2', b'GLIBC_2.29')
    assert address, 'Required core-version scalar symbol missing; no fallback'
    class Info(ctypes.Structure):
        _fields_ = [('filename',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('symbol_address',ctypes.c_void_p)]
    loader.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
    loader.dladdr.restype = ctypes.c_int
    info = Info()
    assert loader.dladdr(address, ctypes.byref(info)) != 0
    path = Path(os.fsdecode(info.filename)).resolve()
    assert path.is_file() and path.name.startswith('libm.so')
    record = dict(library=str(path), sha256=sha(path), bytes=path.stat().st_size,
                  symbol='log2', version='GLIBC_2.29', address=hex(address),
                  relative_offset=hex(address-info.base), dladdr_symbol=os.fsdecode(info.symbol) if info.symbol else None,
                  resolution='dlvsym of the version imported by the retained VI core; not Python math.log2')
    return ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(address), record, libm


def encode_observations(rows, np):
    encoded = []
    for row in rows:
        record = {k: row[k].hex() for k in ('before','after','caller_restored') if k in row}
        record.update(provider=row['provider'], before_semantic=semantic(row['before']),
                      after_semantic=semantic(row['after']) if 'after' in row else None)
        if 'value' in row:
            value = row['value']
            if row['provider'].startswith('numpy'):
                assert type(value) is np.ndarray and value.dtype.str == '<f8'
                assert value.shape == (56,) and value.strides == (8,) and value.nbytes == 448
                assert value.flags.owndata and value.flags.c_contiguous
                payload = value.tobytes()
            else:
                assert len(value) == 56
                payload = b''.join(struct.pack('<d', v) for v in value)
            record.update(dtype='<f8', shape=[56], raw_hex=payload.hex(),
                          words_uint64_hex=[f'{int.from_bytes(payload[i:i+8], "little"):016x}' for i in range(0,448,8)])
        encoded.append(record)
    return encoded


def child(go, go_path, go_hash, role):
    contract = check_go(go, runner_hash=sha(__file__), now=datetime.now(timezone.utc))
    targets = go.get('disabled_targets', []) if role == 'disabled' else []
    assert os.environ.get('NPY_DISABLE_CPU_FEATURES', '') == ','.join(targets)
    assert role in ('base','disabled') and (role == 'base' or go['mode'] == 'hosted-feature')
    # Reuse the admitted F1 backend and controls, without importing the VI driver.
    path = Path(go['source_root']) / 'tests/util_views_fenv.py'
    spec = importlib.util.spec_from_file_location('_provider_existing_F1', path)
    fenv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fenv)
    libc, helper = fenv.backend()
    import ctypes
    counts = dict(numpy_log2_array_calls=0, scalar_log2_values=0, FP_save=0, FP_restore=0)
    def save():
        counts['FP_save'] += 1
        assert counts['FP_save'] <= (15 if role == 'base' else 9)
        buf = (ctypes.c_uint32*8)()
        assert libc.fegetenv(ctypes.byref(buf)) == 0
        return bytes(buf)
    def restore(raw):
        counts['FP_restore'] += 1
        assert counts['FP_restore'] <= (10 if role == 'base' else 6)
        buf = (ctypes.c_uint32*8).from_buffer_copy(raw)
        assert helper.reference_fenv_restore(ctypes.byref(buf), 32) == 0
        actual = save()
        assert actual == raw, 'Full caller32 restoration failed'
        return actual
    caller = save()
    rows, record, primary, np, notices = [], dict(role=role, mode=go['mode'], counts=counts,
        loader_environment=dict(base_prefix=sys.base_prefix, LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH',''),
                                allowed_LD_LIBRARY_PATH=go.get('allowed_LD_LIBRARY_PATH',''))), None, None, []
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            import numpy as np
            assert np.__version__ == '1.24.4'
            features = feature_map(np)
            record.update(features=features, cpu=cpu_record(), numpy_policy=dict(errors=np.geterr(), buffer=np.getbufsize()))
            if go['mode'] == 'private-baseline':
                assert record['cpu']['model name'] == contract['private_CPU_model']
            if go['mode'] == 'hosted-feature':
                base = go['base_features']
                if role == 'base':
                    assert features == base and disabled_targets(features) == go['disabled_targets']
                else:
                    check_intervention(base, features, targets)
            assert not notices, 'Import/feature-disable warning; zero primitives required'
            raw = inputs()
            arrays = [np.frombuffer(p, dtype='<f8').copy() for p in raw]
            seed = seed_state(caller, contract['incoming_FP_semantic'])
            scalar, scalar_identity, retained_libm = scalar_provider() if role == 'base' else (None, None, None)
            record['scalar_identity'] = scalar_identity
            scalar_inputs = [[ctypes.c_double.from_buffer_copy(p[i:i+8]) for i in range(0,448,8)] for p in raw]
            for i, array in enumerate(arrays):
                def numpy_once():
                    counts['numpy_log2_array_calls'] += 1
                    assert counts['numpy_log2_array_calls'] <= 2
                    return np.log2(array)
                observe(rows, 'numpy-'+str(i), numpy_once, save, restore, seed, caller)
                if scalar is not None:
                    def scalar_once():
                        values = []
                        for value in scalar_inputs[i]:
                            counts['scalar_log2_values'] += 1
                            assert counts['scalar_log2_values'] <= 112
                            values.append(scalar(value))
                        return values
                    observe(rows, 'core-scalar-'+str(i), scalar_once, save, restore, seed, caller)
            assert not notices
            assert record['numpy_policy'] == dict(errors=np.geterr(), buffer=np.getbufsize())
            assert all(a.tobytes() == p for a,p in zip(arrays,raw))
    except BaseException as error:
        primary = error
        record['failure'] = dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        # FP snapshots are raw bytes; only now encode values, after restoring caller.
        try:
            restore(caller)
            record.update(observations=encode_observations(rows, np), caller_raw32=caller.hex(),
                          full32_restored_before_codec=True, final_writer_restore_pending=True,
                          warnings=[dict(category=n.category.__name__,message=str(n.message)) for n in notices])
            try:
                assert sha(go_path) == go_hash
                check_go(go, runner_hash=sha(__file__), now=datetime.now(timezone.utc))
                record['postbatch_authority_unchanged'] = True
            except BaseException as error:
                record['authority_failure'] = dict(type=type(error).__name__, error=str(error))
                if primary is None:
                    primary = error
            record['status'] = 'PRIMITIVE_WORDS_RETAINED' if primary is None else 'FAILURE_NO_RETRY'
            write_json(Path(go['output_root']) / (role+'.json'), record)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note('Provider record/restoration error: '+str(error))
        finally:
            try:
                restore(caller)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Final caller32 restore failed: '+str(error))
    if primary is not None:
        raise primary
    expected = dict(numpy_log2_array_calls=2, scalar_log2_values=112 if role == 'base' else 0,
                    FP_save=15 if role == 'base' else 9, FP_restore=10 if role == 'base' else 6)
    assert counts == expected
    print(json.dumps(dict(status='CHILD_PASS_WITH_FINAL32_RESTORE', role=role, counts=counts)))


def word_differences(left, right):
    assert len(left) == len(right) == 56
    return [dict(index=i, left=a, right=b) for i,(a,b) in enumerate(zip(left,right)) if a != b]


def campaign(go, go_path, go_hash):
    check_go(go, runner_hash=sha(__file__), now=datetime.now(timezone.utc))
    assert not os.environ.get('NPY_DISABLE_CPU_FEATURES')
    folder = Path(go['output_root'])
    folder.mkdir(exist_ok=False)
    roles = ('base','disabled') if go['mode'] == 'hosted-feature' else ('base',)
    outcomes = []
    for role in roles:
        command = [sys.executable, str(Path(__file__).resolve()), '--go',str(go_path), '--go-sha256',go_hash,'--child',role]
        env = dict(os.environ)
        if role == 'disabled':
            env['NPY_DISABLE_CPU_FEATURES'] = ','.join(go['disabled_targets'])
        write_json(folder / (role+'-command.json'), dict(argv=command, environment_delta={} if role == 'base' else {'NPY_DISABLE_CPU_FEATURES':env['NPY_DISABLE_CPU_FEATURES']}))
        with (folder/(role+'.stdout')).open('xb') as out, (folder/(role+'.stderr')).open('xb') as err:
            done = subprocess.run(command, env=env, stdout=out, stderr=err)
        write_json(folder/(role+'-exit.json'), dict(exit_code=done.returncode, no_retry=True))
        assert done.returncode == 0, 'Stop dependent child; preserve failed provider outcome'
        terminal = json.loads((folder/(role+'.stdout')).read_bytes())
        assert terminal['status'] == 'CHILD_PASS_WITH_FINAL32_RESTORE' and terminal['role'] == role
        assert terminal['counts'] == dict(numpy_log2_array_calls=2,scalar_log2_values=112 if role == 'base' else 0,
                                         FP_save=15 if role == 'base' else 9,FP_restore=10 if role == 'base' else 6)
        outcome = json.loads((folder/(role+'.json')).read_bytes())
        outcome['actual_final_terminal'] = terminal
        outcomes.append(outcome)
    rows = {v['provider']:v for v in outcomes[0]['observations']}
    comparisons = dict(numpy_vs_core_scalar=[word_differences(rows['numpy-'+str(i)]['words_uint64_hex'], rows['core-scalar-'+str(i)]['words_uint64_hex']) for i in range(2)])
    if len(outcomes) == 2:
        assert outcomes[0]['cpu'] == outcomes[1]['cpu'], 'Host changed between children'
        off = {v['provider']:v for v in outcomes[1]['observations']}
        comparisons['base_vs_AVX512_disabled'] = [word_differences(rows['numpy-'+str(i)]['words_uint64_hex'], off['numpy-'+str(i)]['words_uint64_hex']) for i in range(2)]
        comparisons['disabled_vs_core_scalar'] = [word_differences(off['numpy-'+str(i)]['words_uint64_hex'], rows['core-scalar-'+str(i)]['words_uint64_hex']) for i in range(2)]
    assert sha(go_path) == go_hash
    write_json(folder/'summary.json', dict(status='PROVIDER_WORD_EXPERIMENT_COMPLETE_NOT_VI_PARITY',mode=go['mode'], comparisons=comparisons,
               actual_final_terminals=[v['actual_final_terminal'] for v in outcomes],
               limits=LIMITS, VI_original_calls=0,VI_native_calls=0,timed_batches=0,retries=0,
               per_call_kernel_selection_proved=False,old_9V45_cause_proved=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go', type=Path)
    parser.add_argument('--go-sha256')
    parser.add_argument('--child', choices=('base','disabled'))
    parser.add_argument('--feature-gate', type=Path)
    args = parser.parse_args()
    if args.feature_gate:
        assert os.environ.get('DIAGNOSTIC_AUTHORITY') == AUTHORITY
        assert not os.environ.get('NPY_DISABLE_CPU_FEATURES')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            import numpy as np
            record = feature_map(np)
        assert not notices and record['numpy_version'] == '1.24.4'
        eligible = 'AVX512_SKX' in record['dispatch'] and record['enabled'].get('AVX512_SKX') is True
        record.update(eligible=eligible, disabled_targets=disabled_targets(record) if eligible else [],
                      numpy_log2_array_calls=0,scalar_log2_values=0,VI_original_calls=0,VI_native_calls=0)
        write_json(args.feature_gate, record)
        return
    assert args.go is not None and args.go_sha256 is not None
    assert sha(args.go) == args.go_sha256
    go = json.loads(args.go.read_bytes())
    if args.child:
        child(go, args.go, args.go_sha256, args.child)
    else:
        campaign(go, args.go, args.go_sha256)


if __name__ == '__main__':
    main()
