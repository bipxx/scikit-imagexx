"""One root-GO-gated VI value diagnostic around the unchanged cohort probe.

No target, factory, helper or sampler body runs at import. The wrapper never
calls a target itself: it retains the single existing probe call's values.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

CONTEXT = 'coverage/contracts/vi-cohort-value-diagnostic-v3.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_words(value, *, output):
    assert type(value).__module__ == 'numpy' and type(value).__name__ == 'ndarray'
    assert not value.dtype.hasobject, 'Object pointer payloads are outside this diagnostic'
    assert value.nbytes <= (32 if output else 131072)
    if output:
        assert tuple(value.shape) == (2,)
    raw = value.tobytes()
    assert len(raw) == value.nbytes
    result = dict(type=['numpy', 'ndarray'], dtype=value.dtype.str,
        dtype_descr=value.dtype.descr,
        dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),
        shape=list(value.shape), strides=list(value.strides), nbytes=len(raw),
        payload_sha256=hashlib.sha256(raw).hexdigest())
    if hasattr(value, 'flags'):
        result['flags'] = {name: bool(getattr(value.flags, name)) for name in
                           ('owndata', 'writeable', 'c_contiguous', 'f_contiguous')}
    if hasattr(value, 'base'):
        result['base_type'] = None if value.base is None else [type(value.base).__module__, type(value.base).__name__]
    if output:
        result['raw_hex'] = raw.hex()
    return result


def frame_values(c, record):
    # Called only after old main unwinds. Preserve its collected state before
    # array codecs, which may themselves affect the caller's FP environment.
    record['value_frame_found'] = True
    if 'spec' in c:
        record['case'] = dict(c['spec'])
    record['strict_trace'] = {}
    for key in ('reference_before', 'native_before', 'reference_FP', 'native_FP',
                'returns', 'requirements', 'accepted', 'suffix'):
        if key in c:
            value = c[key]
            record['strict_trace'][key] = dict(value) if isinstance(value, dict) else list(value)
    if 'caller' in c:
        record['driver_caller_raw32'] = bytes(c['caller']).hex()
    if 'before' in c:
        record['input_before'] = c['before']
    record['execute_identity'] = {key: c[key] for key in
        ('api', 'index', 'binding_pin', 'go_pin', 'source_before') if key in c}
    if 'b' in c:
        record['execute_binding_identity'] = {key: c['b'][key] for key in
            ('source_revision', 'producer', 'runtime_sha256', 'core_sha256',
             'installed_site', 'interpreter', 'numpy', 'product_kind') if key in c['b']}
    # These are already computed warning records, not a second observation.
    for key in ('reference_notices', 'native_notices'):
        if key in c:
            record[key] = [dict(category=w.category.__name__, message=str(w.message)) for w in c[key]]
    if 'args' in c:
        assert len(c['args']) == 2
        record['input_after'] = [array_words(a, output=False) for a in c['args']]
    for key in ('expected', 'actual'):
        if key in c:
            record[key] = array_words(c[key], output=True)


def runtime_cpu_evidence():
    """Read metadata only; never import NumPy or invoke a CPU/kernel probe."""
    result = dict(dispatch_scope='Loaded NumPy capability/enabled-feature metadata, not selected per-call kernel identity.')
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        with cpuinfo.open(encoding='utf-8') as stream:
            first = stream.read(1048576).split('\n\n', 1)[0]
        fields = {name.strip(): value.strip() for name, value in
                  (line.split(':', 1) for line in first.splitlines() if ':' in line)}
        result['cpu'] = {name: fields[name] for name in fields if name in
                         ('vendor_id', 'model name', 'cpu family', 'model', 'stepping', 'flags')}
    result['numpy_dispatch'] = {}
    for name in ('numpy.core._multiarray_umath', 'numpy._core._multiarray_umath'):
        module = sys.modules.get(name)
        if module is not None:
            data = vars(module)
            result['numpy_dispatch'][name] = {key: dict(data[key]) if isinstance(data[key], dict) else list(data[key]) for key in
                ('__cpu_features__', '__cpu_baseline__', '__cpu_dispatch__') if key in data}
    return result


def traceback_values(error, execute_path, probe_path, record):
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    execute = [f for f in frames if f.f_code.co_name == 'execute'
               and Path(f.f_code.co_filename).resolve() == Path(execute_path).resolve()]
    probe = [f for f in frames if f.f_code.co_name == 'main'
             and Path(f.f_code.co_filename).resolve() == Path(probe_path).resolve()]
    # Preserve all already-collected FP/count evidence BEFORE dtype/array codecs.
    if probe:
        values = probe[-1].f_locals
        if 'counts' in values:
            record['probe_counts'] = dict(values['counts'])
        if 'restored' in values:
            record['probe_restored'] = values['restored']
    if not execute:
        record['value_frame_found'] = False
        return
    assert len(execute) == 1
    record['value_frame_source'] = 'traceback'
    frame_values(execute[0].f_locals, record)


def write_record(path, value):
    # Exclusive create: retain a partially written file if the writer fails.
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run_diagnostic(probe, argv, save, restore, output, execute_path, terminal_path,
                   final_check=lambda: None):
    """Call old main once; encode retained values only after its finally."""
    old_path, old_argv = list(sys.path), sys.argv
    old_trace, old_profile = sys.gettrace(), sys.getprofile()
    saved = save()
    record = dict(schema='vi-cohort-value-diagnostic-v2', own_before_raw32=bytes(saved).hex(),
                  target_calls_added_by_wrapper=0, sampler_body_calls_authorized=0,
                  secondary_errors=[])
    primary = primary_tb = None
    validated_locals = []
    try:
        try:
            sys.argv = argv
            validator = probe.validate_native_entry
            def retain_validated(*args, **kwargs):
                values = validator(*args, **kwargs)
                validated_locals.append(dict(values))
                return values
            probe.validate_native_entry = retain_validated
            try:
                probe.main()
            finally:
                probe.validate_native_entry = validator
        except BaseException as error:
            primary, primary_tb = error, error.__traceback__
            after_old_probe = bytes(save()).hex()
            record.update(status='FAILED_OLD_PROBE_NO_RETRY', primary_type=type(error).__name__,
                primary_error=str(error), primary_traceback=''.join(traceback.format_exception(error)),
                after_old_probe_raw32=after_old_probe)
            try:
                traceback_values(error, execute_path, probe.__file__, record)
                if not record['value_frame_found'] and validated_locals:
                    assert len(validated_locals) == 1
                    record['value_frame_source'] = 'validated_sampler_entry'
                    frame_values(validated_locals[0], record)
            except BaseException as secondary:
                record['secondary_errors'].append(dict(phase='value-codec', type=type(secondary).__name__, error=str(secondary)))
        else:
            # Old main returns normally only after its mandatory sampler stop+C1.
            after_old_probe = bytes(save()).hex()
            record['after_old_probe_raw32'] = after_old_probe
            terminal = json.loads(Path(terminal_path).read_bytes())
            assert terminal['status'] == 'PASS_REAL_CLI_STOP_BEFORE_BODY'
            assert terminal['counts'] == dict(original=1, native=1)
            assert terminal['sampler_body_calls'] == terminal['timed_batches'] == 0
            assert terminal['outer_state_restored'] is terminal['poststop_source_authority_unchanged'] is True
            record.update(status='PASS_OLD_PROBE_SAMPLER_STOP_NO_TIMING', terminal=terminal,
                          probe_counts=dict(terminal['counts']), probe_restored=terminal['outer_state_restored'])
            assert len(validated_locals) == 1
            record['value_frame_source'] = 'validated_sampler_entry'
            try:
                frame_values(validated_locals[0], record)
            except BaseException as secondary:
                record['secondary_errors'].append(dict(phase='value-codec', type=type(secondary).__name__, error=str(secondary)))
                primary, primary_tb = secondary, secondary.__traceback__
        try:
            record['runtime_cpu_evidence'] = runtime_cpu_evidence()
        except BaseException as secondary:
            record['secondary_errors'].append(dict(phase='CPU-metadata', type=type(secondary).__name__, error=str(secondary)))
            if primary is None:
                primary, primary_tb = secondary, secondary.__traceback__
        try:
            final_check()
        except BaseException as secondary:
            record['secondary_errors'].append(dict(phase='final-authority', type=type(secondary).__name__, error=str(secondary)))
            if primary is None:
                primary, primary_tb = secondary, secondary.__traceback__
    except BaseException as error:
        if primary is None:
            primary, primary_tb = error, error.__traceback__
        record.setdefault('status', 'FAILED_DIAGNOSTIC_SETUP_OR_SUCCESS_TAIL')
        record.setdefault('primary_type', type(primary).__name__)
        record.setdefault('primary_error', str(primary))
        if validated_locals and not record.get('value_frame_found'):
            try:
                assert len(validated_locals) == 1
                record['value_frame_source'] = 'validated_sampler_entry'
                frame_values(validated_locals[0], record)
            except BaseException as secondary:
                record['secondary_errors'].append(dict(phase='value-codec', type=type(secondary).__name__, error=str(secondary)))
    finally:
        sys.path[:] = old_path
        sys.argv = old_argv
        sys.settrace(old_trace)
        sys.setprofile(old_profile)
        try:
            restore(saved)
            actual = bytes(save())
            record.update(own_restored_raw32=actual.hex(), own_full32_restored=actual == bytes(saved))
            assert record['own_full32_restored']
        except BaseException as secondary:
            record['secondary_errors'].append(dict(phase='own-restore', type=type(secondary).__name__, error=str(secondary)))
            if primary is None:
                primary, primary_tb = secondary, secondary.__traceback__
    try:
        write_record(output, record)
    except BaseException as secondary:
        # No target retry; retain the primary and already captured FP/count data.
        fallback = dict(record)
        fallback['writer_error'] = dict(type=type(secondary).__name__, error=str(secondary))
        try:
            write_record(Path(output).with_suffix('.failure.json'), fallback)
        except BaseException as fallback_error:
            if primary is not None:
                primary.add_note('Diagnostic writer and fallback failed: ' + type(fallback_error).__name__)
            else:
                primary, primary_tb = fallback_error, fallback_error.__traceback__
        if primary is None:
            primary, primary_tb = secondary, secondary.__traceback__
    finally:
        try:
            restore(saved)
            assert bytes(save()) == bytes(saved)
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note('Final diagnostic FP restoration failed: ' + type(cleanup_error).__name__)
            else:
                primary, primary_tb = cleanup_error, cleanup_error.__traceback__
    if primary is not None:
        raise primary.with_traceback(primary_tb)
    return record


def validate_authority(args, controller):
    assert digest(args.diagnostic_go) == args.diagnostic_go_sha256
    go = json.loads(args.diagnostic_go.read_bytes())
    assert go['status'] == 'ROOT_AUTHORIZED_ONE_VI_VALUE_DIAGNOSTIC', 'UNBOUND diagnostic'
    assert go['wrapper_sha256'] == digest(__file__)
    assert go['binding_sha256'] == digest(args.binding)
    assert go['cohort_GO_sha256'] == digest(args.root_go)
    assert go['maximum_original_topcalls'] == go['maximum_native_topcalls'] == 1
    assert go['api'] == 'metrics.variation_of_information'
    assert go['sampler_body_calls'] == go['timed_batches'] == 0
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go['deadline_utc'])
    assert Path(go['output_path']).resolve() == args.output.resolve()
    assert not args.output.exists() and not args.output.with_suffix('.failure.json').exists()
    assert args.output.parent.is_dir()
    context_raw = (controller / CONTEXT).read_bytes()
    assert go['context_sha256'] == hashlib.sha256(context_raw).hexdigest()
    context = json.loads(context_raw)
    for name, pin in context['controller_dependencies'].items():
        assert digest(controller / name) == pin
    binding = json.loads(args.binding.read_bytes())
    root = Path(binding['source_root']).resolve()
    assert binding['source_revision'] == go['source_revision'] == context['qualified_source']
    for name, pin in context['prepared_dependencies'].items():
        assert digest(root / name) == pin
    return binding, root, context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binding', 'root-go', 'diagnostic-go', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--diagnostic-go-sha256', required=True)
    args = parser.parse_args()
    controller = Path(__file__).resolve().parents[1]
    binding, root, context = validate_authority(args, controller)
    # Existing source-bound F1 controller, no new helper implementation/build.
    path = root / 'benchmarks/next_util_execution.py'
    spec = importlib.util.spec_from_file_location('_vi_diagnostic_common', path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    actual_context = json.loads((root / context['driver_context']).read_bytes())
    save, restore, _ = common.fp_controller(root, actual_context)
    spec = importlib.util.spec_from_file_location('_vi_unchanged_probe', controller / context['probe'])
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    argv = [str(controller / context['probe']), '--binding', str(args.binding),
            '--root-go', str(args.root_go), '--route', 'vi']
    return run_diagnostic(probe, argv, save, restore, args.output,
        root / 'benchmarks/variation_of_information_execution.py',
        Path(binding['output_root']) / 'first-cohort/vi/terminal.json',
        final_check=lambda: validate_authority(args, controller))


if __name__ == '__main__':
    main()
