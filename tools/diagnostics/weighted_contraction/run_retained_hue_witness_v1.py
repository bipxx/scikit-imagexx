"""One source-bound primitive observation; compilation and entry belong to root."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

STATUS = 'ROOT_GO_WEIGHTED_HUE_CONTRACTION_V1'
EPOCH = 'a5f3aa9748b77540690189067fb1178385bb7500'
CPP_SHA256 = 'dd8dd59f31f0d9a51bcf6f95f735a4f3965010a0e6486320c9c5989781720fef'
SOURCE_PINS = {
    'src/distance_metrics.hpp': '63ae09db1a66f8493440cb2e11d6d6078462c000673b99ec3e6afa5f6a199293',
    'tests/test_weighted_distance.py': '81433215024975b06e4dd2253bf18039d2ceb70a444ff70ce3d45b43a3eca118',
    'CMakeLists.txt': '06ead19d2848e3d83fd3041a03cdb7ea6600abaf82faa8112497487ec3535fcc',
}
VERSIONS = {'python': '3.11.17', 'numpy': '1.24.4', 'scipy': '1.15.3', 'skimage': '0.26.0'}
CAPS = {'original': 1, 'baseline': 1, 'strict': 1, 'backend': 1,
        'fegetenv': 10, 'restore': 3, 'public_color': 0, 'timing': 0}
ROLES = ('original', 'baseline', 'strict')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_json(reference):
    path = Path(reference['path'])
    require(path.is_absolute() and sha(path) == reference['sha256'], 'Receipt pin differs')
    return json.loads(path.read_text(encoding='utf-8'))


def error_record(error):
    return {'type': type(error).__name__, 'message': str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--binding-sha256', required=True)
    args = parser.parse_args()
    go = bound_json({'path': args.binding, 'sha256': args.binding_sha256})
    require(__debug__ and go['status'] == STATUS, 'UNBOUND witness')
    require(go['driver_sha256'] == sha(__file__) and go['production_epoch'] == EPOCH,
            'Witness source differs')
    require(go['maximum_invocations'] == 1 and go['retries'] == 0 and go['caps'] == CAPS,
            'Witness budgets differ')
    now = datetime.datetime.now(datetime.timezone.utc)
    issued = datetime.datetime.fromisoformat(go['issued_utc'])
    deadline = datetime.datetime.fromisoformat(go['deadline_utc'])
    require(issued <= now < deadline and (deadline-issued).total_seconds() <= 1800,
            'Witness GO is not current')
    require(not any(os.environ.get(name) for name in
                    ('NPY_DISABLE_CPU_FEATURES', 'SKIMAGEXX_DISABLE_SIMD')),
            'CPU masks are forbidden')
    entry = bound_json(go['entry'])
    require(entry['status'] == 'PASS_WEIGHTED_HUE_ENTRY_AND_CODEGEN_V1'
            and entry['source_revision'] == go['source_revision']
            and entry['production_epoch'] == EPOCH, 'Entry/source scope differs')
    require(entry['runtime_versions'] == VERSIONS and entry['os_version'] == '26.04'
            and entry['glibc'] == '2.43', 'Host/runtime is outside this witness')
    require(type(entry['baseline_has_contraction']) is bool
            and type(entry['strict_has_contraction']) is bool, 'Codegen observation is absent')
    require(entry['sole_change_is_contract_off'] is True
            and entry['baseline_already_contract_off'] is False,
            'There is no sole-change contraction comparison')
    require(entry['variants'] == go['variants'], 'Compiled variant binding differs')
    require(go['cpp_source']['sha256'] == CPP_SHA256
            and sha(go['cpp_source']['path']) == CPP_SHA256
            and entry['cpp_sha256'] == CPP_SHA256, 'Primitive CPP differs')
    root = Path(go['source_root'])
    require(root.is_absolute() and sys.byteorder == 'little', 'Source/word ABI differs')
    require({'tests/util_views_fenv.py', 'coverage/contracts/util-views-fenv-v1.json'}
            <= set(go['canonical_source_pins']), 'Executed FENV source/context is unbound')
    for name, digest in SOURCE_PINS.items():
        require(go['canonical_source_pins'].get(name) == digest, 'Scalar source epoch differs')
    for name, digest in go['canonical_source_pins'].items():
        raw = (root/name).read_bytes().replace(b'\r\n', b'\n')
        require(hashlib.sha256(raw).hexdigest() == digest, 'Source pin differs: '+name)
    for variant in go['variants'].values():
        require(Path(variant['path']).is_absolute() and sha(variant['path']) == variant['sha256'],
                'Primitive library differs')
    out = Path(go['output'])
    require(out.is_absolute() and not out.exists(), 'Fresh output is required')
    out.mkdir(parents=True)
    (out/'claim.json').write_text(json.dumps({'status': 'CLAIMED_ONCE',
        'binding_sha256': args.binding_sha256, 'maximum_invocations': 1, 'retries': 0})+'\n',
        encoding='utf-8')
    counts = dict.fromkeys(CAPS, 0)
    primary = None
    secondary = []
    outputs = {}
    frames = {}
    outer_raw = None
    last_restore = None
    input_record = None
    identity = None
    software_before = software_after = None

    def remember(error):
        nonlocal primary
        if primary is None:
            primary = error
        else:
            secondary.append(error_record(error))

    try:
        # Every scientific import, fixture operation, and library load is AFTER GO.
        import ctypes
        import importlib.metadata
        import platform
        import numpy as np
        from skimage.color import delta_e
        require(platform.python_version() == VERSIONS['python']
                and np.__version__ == VERSIONS['numpy']
                and importlib.metadata.version('scipy') == VERSIONS['scipy']
                and importlib.metadata.version('scikit-image') == VERSIONS['skimage'],
                'Loaded runtime differs')
        require('skimagexx' not in sys.modules, 'No project package belongs to this primitive study')
        original_path = Path(delta_e.__file__).resolve()
        require(str(original_path) == str(Path(go['original_source']['path']).resolve())
                and sha(original_path) == go['original_source']['sha256'],
                'Pristine original source differs')
        original = delta_e.get_dH2
        require(Path(original.__code__.co_filename).resolve() == original_path,
                'Original callable ownership differs')
        spec = importlib.util.spec_from_file_location('weighted_fenv', root/'tests/util_views_fenv.py')
        fenv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fenv)
        require(os.environ.get('UTIL_VIEWS_FENV_MODE') == 'source-built-v1'
                and os.environ.get('UTIL_VIEWS_FENV_RECEIPT') == go['fenv_receipt']['path']
                and os.environ.get('UTIL_VIEWS_FENV_RECEIPT_SHA256') == go['fenv_receipt']['sha256'],
                'FENV receipt binding differs')
        counts['backend'] += 1
        libc, helper = fenv.backend()
        require(ctypes.sizeof(fenv.FEnv) == 32, 'Full32 layout differs')
        libraries = {}
        functions = {}
        for role in ('baseline', 'strict'):
            libraries[role] = ctypes.CDLL(go['variants'][role]['path'])
            function = libraries[role].weighted_hue_graph_v1
            function.argtypes = [ctypes.c_double]*4
            function.restype = ctypes.c_double
            functions[role] = function
        # Exactly the existing float64 sample branch, one predetermined reverse-pair center.
        colors = (np.random.default_rng(32).random((67, 3))*200-100).astype(np.float64)
        reversed_colors = colors[::-1].copy()
        first, second = colors[33], reversed_colors[33]
        first.flags.writeable = second.flags.writeable = False
        first_raw, second_raw = first.tobytes(), second.tobytes()
        require(first_raw == second_raw and len(first_raw) == 24, 'Fixed center-pair words differ')
        arguments = tuple(float(value) for value in (first[1], first[2], second[1], second[2]))
        input_record = {'factory': 'seed32/float64/(67,3)/row33/reverse-copy',
                        'first_raw64': first_raw.hex(), 'second_raw64': second_raw.hex(),
                        'shape': list(first.shape), 'strides': list(first.strides),
                        'first_pointer': first.__array_interface__['data'][0],
                        'second_pointer': second.__array_interface__['data'][0]}
        software_before = {'errors': np.geterr(), 'error_call_id': id(np.geterrcall()),
                           'buffer': np.getbufsize()}
        identity = {'entry': go['entry'], 'source_revision': go['source_revision'],
                    'original_source': go['original_source'], 'variants': go['variants'],
                    'fenv_receipt': go['fenv_receipt'], 'runtime_versions': VERSIONS,
                    'codegen': {'baseline_has_contraction': entry['baseline_has_contraction'],
                                'strict_has_contraction': entry['strict_has_contraction']},
                    'FP_scope': 'after imports/setup/loads through the last restore-readback; '
                                'no across-postmetadata/serialization/finish claim'}

        def save():
            counts['fegetenv'] += 1
            value = fenv.FEnv()
            require(libc.fegetenv(ctypes.byref(value)) == 0, 'fegetenv failed')
            return bytes(value)

        def restore(raw):
            counts['restore'] += 1
            value = fenv.FEnv.from_buffer_copy(raw)
            require(helper.reference_fenv_restore(ctypes.byref(value), 32) == 0,
                    'Full32 restore failed')

        outer_raw = save()
        initial = fenv.FEnv.from_buffer_copy(outer_raw)
        require(initial.tags == 0xffff and initial.control & 0x3f == 0x3f
                and initial.mxcsr & 0x1f80 == 0x1f80 and initial.mxcsr & 0xffff0000 == 0,
                'Natural caller32 is outside the existing helper restore contract')
        for role in ROLES:
            frame = {}
            frames[role] = frame
            try:
                before = save()
                frame['before'] = before.hex()
                require(before == outer_raw, 'Natural caller32 differs before '+role)
                counts[role] += 1
                if role == 'original':
                    outputs[role] = original(first, second, channel_axis=0)
                else:
                    outputs[role] = functions[role](*arguments)
            except BaseException as error:
                remember(error)
            finally:
                try:
                    after = save()
                    frame['after'] = after.hex()
                    prior = fenv.FEnv.from_buffer_copy(outer_raw)
                    actual = fenv.FEnv.from_buffer_copy(after)
                    require(prior.control == actual.control
                            and (prior.mxcsr & ~0x3f) == (actual.mxcsr & ~0x3f),
                            'Target changed natural FP controls')
                except BaseException as error:
                    remember(error)
                try:
                    restore(outer_raw)
                except BaseException as error:
                    remember(error)
                try:
                    restored = save()
                    frame['restored'] = restored.hex()
                    last_restore = restored == outer_raw
                    require(last_restore, 'Caller32 restoration differs')
                except BaseException as error:
                    last_restore = False
                    remember(error)
            if primary is not None:
                break
        # Numeric differences are OBSERVATIONS, never an early failure before strict.
        require(counts == CAPS and set(outputs) == set(ROLES) and last_restore is True,
                'Completed-pair call/FP/caller evidence is incomplete')
        require(first.tobytes() == first_raw and second.tobytes() == second_raw,
                'Fixture input mutated')
        software_after = {'errors': np.geterr(), 'error_call_id': id(np.geterrcall()),
                          'buffer': np.getbufsize()}
        require(software_before == software_after, 'Natural NumPy software context changed')
        require(delta_e.get_dH2 is original and sha(original_path) == go['original_source']['sha256'],
                'Original callable/source changed')
        for variant in go['variants'].values():
            require(sha(variant['path']) == variant['sha256'], 'Compiled library changed')
    except BaseException as error:
        remember(error)

    # This terminal is built AFTER the last restoration. No further FENV queries.
    words = {}
    for role, value in outputs.items():
        try:
            words[role] = {'raw64': None, 'type': type(value).__name__,
                           'shape': list(getattr(value, 'shape', ())),
                           'dtype': str(getattr(value, 'dtype', 'Cdouble'))}
            words[role]['raw64'] = struct.pack('<d', float(value)).hex()
            if role == 'original':
                require(type(value) is np.float64 and value.shape == ()
                        and value.dtype == np.dtype('float64'), 'Original output metadata differs')
            else:
                require(type(value) is float, 'Cdouble output metadata differs')
        except BaseException as error:
            remember(error)
    completed = (counts == CAPS and set(words) == set(ROLES) and last_restore is True
                 and all(row['raw64'] is not None for row in words.values()))
    supported = (completed and entry['baseline_has_contraction'] is True
                 and entry['strict_has_contraction'] is False
                 and words['baseline']['raw64'] != words['original']['raw64']
                 and words['strict']['raw64'] == words['original']['raw64'])
    record = {'status': ('FAILED_PROTOCOL' if primary is not None else
                        'COMPLETE_CONTRACTION_WITNESS' if supported else
                        'COMPLETE_INCONCLUSIVE_NUMERICAL_OBSERVATION'),
              'counts': counts, 'caps': CAPS, 'completed': completed,
              'contraction_supported_on_this_host': bool(supported and primary is None),
              'input': input_record, 'outputs': words, 'FP': frames,
              'outer32': None if outer_raw is None else outer_raw.hex(),
              'last_restore_verified': last_restore, 'identity': identity,
              'software_before': software_before, 'software_after': software_after,
              'primary': None if primary is None else error_record(primary),
              'secondary': secondary, 'old_CI_executed_cause_proven': False,
              'scope': 'Primitive-only new-host observation; no public parity/release/timing credit'}
    try:
        with (out/'result-v1.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    except BaseException as error:
        remember(error)
        record.update(status='RESULT_WRITE_FAILED', primary=error_record(primary),
                      secondary=secondary)
        print(json.dumps(record), flush=True)
    if primary is not None:
        raise primary
    print(json.dumps({'status': record['status'], 'counts': counts,
                      'result': str(out/'result-v1.json')}), flush=True)


if __name__ == '__main__':
    main()
