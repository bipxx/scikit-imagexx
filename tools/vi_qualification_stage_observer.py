"""One untimed case0 public-boundary observation; no authority at import/default."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
import warnings

STATUS = 'ROOT_GO_VI_CASE0_PUBLIC_STAGE_OBSERVER_V2'
CASE = dict(id='metrics.variation_of_information/128x128/8/C',
            api='metrics.variation_of_information', shape=[128, 128], labels=8,
            dtype='int64', layout='C', ignore=[0])
CAPS = dict(original=1, native=1, fixtures=1, original_ufunc=2, native_ufunc=2,
            CSR_boundary=1, tables_return=1, xlogx_return=2,
            copied_bytes=4194304, FP_save=11, FP_restore=5,
            F1_backend=1, F1_ABI_query=1, F1_glibc_query=1,
            software_getters=12, software_setters=3, worker_getters=4,
            worker_restores=1, fallback_setters=2, alias_queries=5,
            sampler=0, timing=0, warmup=0, retries=0, CPU_masks=0)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def check_go(go, pin, now):
    require(go is not None, 'UNBOUND: root hosted entry and fresh one-pair GO required')
    require(go.get('status') == STATUS and go.get('observer_sha256') == pin, 'Wrong observer authority/source')
    require(go.get('case') == CASE and go.get('workers') == 1 and go.get('caps') == CAPS, 'Wrong exact case/budget')
    require(go.get('source_revision') == '8ac153c6659a3056e3e6ac18063213af4642a768', 'Wrong failed-product source')
    require(go.get('producer_run') == 43817382620 and go.get('instrumented_diagnostic_only') is True, 'No qualification authority')
    require(go.get('hosted_entry', {}).get('path') and len(go['hosted_entry'].get('sha256', '')) == 64, 'Hosted entry unbound')
    require(go.get('runtime_binding', {}).get('path') and len(go['runtime_binding'].get('sha256', '')) == 64, 'Runtime/product/helper/capacity binding unbound')
    require(isinstance(go.get('output'), str) and Path(go['output']).is_absolute(), 'Evidence output unbound')
    issued = datetime.fromisoformat(go['issued_utc']); deadline = datetime.fromisoformat(go['deadline_utc'])
    require(issued.tzinfo is not None and deadline.tzinfo is not None and issued <= now < deadline,
            'GO expired/not current')
    require((deadline - issued).total_seconds() <= 3600, 'One short-lived campaign only')


def load_pinned(reference, name):
    path = Path(reference['path']); raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Pinned source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class BoundaryObserver:
    """Copies actual arguments/return-frame buffers, never recomputes a stage."""
    def __init__(self, ctx, record, counts):
        self.ctx, self.record, self.counts = ctx, record, counts
        self.phase = 'setup'
        self.true_ufunc = ctx['np'].log2
        self.builtin = ctx['core'].variation_of_information_csr
        self.codes = {ctx['upstream']._vi_tables.__code__: '_vi_tables',
                      ctx['upstream']._xlogx.__code__: '_xlogx'}

    def bump(self, key, amount=1):
        require(self.counts[key] + amount <= CAPS[key], 'Observer budget exceeded: ' + key)
        self.counts[key] += amount

    def array(self, value, *, retain=True):
        require(type(value).__module__ == 'numpy' and not value.dtype.hasobject, 'Plain numeric buffer required')
        require(value.nbytes <= 131072, 'One case0 buffer exceeds source bound')
        self.bump('copied_bytes', value.nbytes)
        raw = memoryview(value).tobytes()
        require(len(raw) == value.nbytes, 'Buffer metadata/byte length differs')
        result = dict(type=[type(value).__module__, type(value).__name__], dtype=value.dtype.str,
                      shape=list(value.shape), strides=list(value.strides), nbytes=len(raw))
        if hasattr(value, 'flags'):
            result['flags'] = {n: bool(getattr(value.flags, n)) for n in ('owndata', 'writeable', 'c_contiguous', 'f_contiguous', 'aligned')}
        if retain: result['_raw_bytes'] = raw
        else: result['sha256'] = hashlib.sha256(raw).hexdigest()
        return result

    def sparse(self, value):
        require(value.format in ('csr', 'csc'), 'Only actual CSR/CSC buffers are observed')
        return dict(format=value.format, shape=list(value.shape),
                    data=self.array(value.data), indices=self.array(value.indices), indptr=self.array(value.indptr))

    def capture(self, destination, key, operation):
        try: destination[key] = operation()
        except Exception as error:
            destination[key] = dict(unavailable=type(error).__name__, error=str(error))
            self.record['secondary'].append(dict(stage=self.phase + ':' + key, type=type(error).__name__, error=str(error)))

    def log2(self, values):
        key = self.phase + '_ufunc'
        require(key in ('original_ufunc', 'native_ufunc'), 'Ufunc outside the two target phases')
        self.bump(key)
        event = dict(phase=self.phase, ordinal=self.counts[key], returned=False)
        self.record['ufunc'].append(event)
        self.capture(event, 'input', lambda: self.array(values))
        # Saved current public ufunc is called ONCE. No arithmetic/F1/clock in this callback.
        result = self.true_ufunc(values)
        event['returned'] = True
        self.capture(event, 'output', lambda: self.array(result))
        return result

    def csr(self, *args, **kwargs):
        self.bump('CSR_boundary')
        require(not kwargs and len(args) == 7, 'Pinned frontend CSR call shape changed')
        event = dict(rows=args[3], columns=args[4], workers=args[5],
                     callback_is_current_np_log2=args[6] is self.ctx['np'].log2,
                     delegates_saved_builtin=True)
        self.record['native_CSR'].append(event)
        for name, value in zip(('data', 'indices', 'indptr'), args[:3]):
            self.capture(event, name, lambda value=value: self.array(value))
        return self.builtin(*args, **kwargs)

    def core_proxy(self):
        observer = self
        class CoreBoundary:
            def __getattr__(self, name):
                return observer.csr if name == 'variation_of_information_csr' else getattr(observer.ctx['core'], name)
        return CoreBoundary()

    def trace(self, frame, event, value):
        kind = self.codes.get(frame.f_code)
        if kind is None: return None
        frame.f_trace_lines = False; frame.f_trace_opcodes = False
        if event != 'return': return self.trace
        self.bump('tables_return' if kind == '_vi_tables' else 'xlogx_return')
        row = dict(function=kind, return_captured=False)
        self.record['original_frames'].append(row)
        local = frame.f_locals
        if kind == '_xlogx':
            for name in ('x', 'y'):
                if name in local: self.capture(row, name, lambda name=name: self.sparse(local[name]))
            if 'z' in local: self.capture(row, 'z', lambda: self.array(local['z']))
            if 'nz' in local: self.capture(row, 'nz', lambda: [self.array(v) for v in local['nz']])
            row['return_is_y'] = value is local.get('y')
            row['return_captured'] = row['return_is_y'] and 'data' in row.get('y', {})
        else:
            if 'pxy' in local: self.capture(row, 'pxy', lambda: self.sparse(local['pxy']))
            for name in ('px', 'py', 'hxgy', 'hygx'):
                if name in local: self.capture(row, name, lambda name=name: self.array(local[name]))
            for name in ('px_inv', 'py_inv'):
                if name in local:
                    self.capture(row, name, lambda name=name: dict(data=self.array(local[name].data), offsets=self.array(local[name].offsets)))
            if value is not None:
                self.capture(row, 'return', lambda: [self.array(v) for v in value])
                row['return_captured'] = isinstance(row.get('return'), list) and len(row['return']) == 2 and all('_raw_bytes' in v for v in row['return'])
        return self.trace


def encode_raw(value):
    """Run only after target snapshots/caller restoration; no scientific codec."""
    if isinstance(value, list): return [encode_raw(v) for v in value]
    if isinstance(value, dict):
        result = {k: encode_raw(v) for k, v in value.items() if k != '_raw_bytes'}
        if '_raw_bytes' in value:
            raw = value['_raw_bytes']; result.update(raw_hex=raw.hex(), payload_sha256=hashlib.sha256(raw).hexdigest())
        return result
    return value


def cleanup_all(actions, remember):
    for stage, action in actions:
        try: action()
        except BaseException as error: remember(error, stage)


def final_closeout(ctx, record, counts, outputs, outer, restore, remember, primary_state, output, record_written):
    """One existing final restore, then publish terminal facts before any rethrow."""
    attempted = outer is not None
    verified = False
    try:
        if attempted:
            actual = restore(outer)
            require(actual == bytes(outer), 'Final caller32 differs')
            verified = True
    except BaseException as error:
        remember(error, 'final-writer-restore')
    completed = all(name in outputs for name in ('expected', 'actual'))
    bounded = all(type(value) is int and 0 <= value <= CAPS[name] for name, value in counts.items())
    exact = counts['FP_save'] == 11 and counts['FP_restore'] == 5
    try:
        require(bounded, 'Observer terminal counters exceeded literal bounds')
        if completed: require(exact, 'Complete two-returned-target path requires exact11/5 FP counts')
    except BaseException as error:
        remember(error, 'terminal-counter-validation')
    primary = primary_state()
    pending = None if completed else ('No observer outer snapshot' if not attempted else 'One or both targets did not return; exact11/5 not claimed')
    ctx['_observer_terminal'] = dict(
        schema='vi-case0-stage-observer-terminal-v2',
        status='OBSERVER_FAILURE_RETAINED' if primary is not None else ('INSTRUMENTED_PAIR_TERMINAL_NOT_QUALIFICATION' if completed else 'PARTIAL_OBSERVER_TERMINAL'),
        completed_two_returned_targets=completed, counts=dict(counts), bounds_passed=bounded,
        full_FP_counts_passed=exact if completed else None, pending_reason=pending,
        final_restore_attempted=attempted, final_restore_verified=verified,
        last_restore_observation=dict(record.get('last_restore_observation', {})) if attempted else None,
        primary=None if primary is None else dict(type=type(primary).__name__, error=str(primary), traceback=''.join(traceback.format_exception(primary))),
        secondary=list(record['secondary']), outputs=encode_raw(record['outputs']),
        raw_record_path=str(output), raw_record_written=record_written,
        scope='Observed final restore/counters inside observer, before entry.finish; no across-finish or pre-import FP claim')
    return ctx['_observer_terminal']


def run_pair(ctx, go):
    """Root's separately authenticated hosted entry supplies only pinned objects."""
    require(ctx['status'] == 'ROOT_AUTHENTICATED_FAILED8AC_STAGE_ENTRY_V1', 'Hosted entry did not authenticate scope')
    np, core, owned, upstream = (ctx[k] for k in ('np', 'core', 'owned', 'upstream'))
    xx, compat, fixture = (ctx[k] for k in ('xx', 'compat', 'fixture'))
    reference, native = ctx['skimage'].metrics.variation_of_information, xx.metrics.variation_of_information
    require(reference is upstream.variation_of_information and native is owned.variation_of_information,
            'Public function identity changed')
    require(owned._core is core is xx._core and inspect.isbuiltin(core.variation_of_information_csr), 'Native ownership changed')
    require(type(np.log2) is np.ufunc and sys.getprofile() is None and sys.gettrace() is None, 'Unexpected provider/instrumentation')
    record = dict(status='STARTED_INSTRUMENTED_PAIR_NOT_QUALIFICATION', identity=ctx['identity'],
                  case=CASE, workers=1, primary=None, secondary=[], ufunc=[], native_CSR=[], original_frames=[],
                  outputs={}, journal=[], gates={}, timing=False,
                  instrumentation='Temporary frontend _core forwarder, public np.log2 observer, native ownership profile and original two-code return trace')
    counts = {k: 0 for k in CAPS if k not in ('F1_ABI_query', 'F1_glibc_query')}
    observer = BoundaryObserver(ctx, record, counts)
    primary = primary_tb = None
    outer = caller = None; outputs = {}; old_profile = sys.getprofile(); old_trace = sys.gettrace()
    old_core = owned._core; true_ufunc = np.log2
    old_workers = old_fallback = software = None
    libc = helper = None
    record_written = False

    def remember(error, stage):
        nonlocal primary, primary_tb
        if primary is None: primary, primary_tb = error, error.__traceback__
        else: record['secondary'].append(dict(stage=stage, type=type(error).__name__, error=str(error)))

    def uninstrument():
        cleanup_all((('trace-restore', lambda: sys.settrace(old_trace)),
                     ('profile-restore', lambda: sys.setprofile(old_profile)),
                     ('core-global-restore', lambda: setattr(owned, '_core', old_core)),
                     ('log2-global-restore', lambda: setattr(np, 'log2', true_ufunc))), remember)

    def save():
        observer.bump('FP_save'); state = ctx['fenv'].FEnv()
        require(libc.fegetenv(ctx['ctypes'].byref(state)) == 0, 'FP snapshot failed')
        return state

    def restore(state):
        observer.bump('FP_restore')
        expected = bytes(state)
        record['last_restore_observation'] = dict(attempt=counts['FP_restore'], expected_raw32=expected.hex(), actual_raw32=None, exact=False)
        require(helper.reference_fenv_restore(ctx['ctypes'].byref(state), 32) == 0, 'F1 restore failed')
        actual = bytes(save())
        record['last_restore_observation'].update(actual_raw32=actual.hex(), exact=actual == expected)
        require(actual == expected, 'Caller32 differs'); return actual

    def semantic(state):
        return {n: getattr(state, n) for n in ('control', 'status', 'mxcsr')}

    def policy():
        observer.bump('software_getters', 3); return np.geterr(), np.geterrcall(), np.getbufsize()

    def workers():
        observer.bump('worker_getters'); return core.get_num_threads()

    def same_policy(current):
        return current[0] == software[0] and current[1] is software[1] and current[2] == software[2]

    try:
        observer.bump('F1_backend'); libc, helper = ctx['fenv'].backend()
        ctx['_observer_backend'] = (libc, helper)
        # Existing backend configures one ABI and one glibc query; these are source bounds.
        record['F1_metadata_bounds_not_measured'] = dict(ABI_query=1, glibc_query=1)
        outer = save(); record['outer_before_raw32'] = bytes(outer).hex()
        old_workers = workers(); require(old_workers == 1, 'Default worker count must already be one')
        old_fallback = compat._fallback_enabled
        observer.bump('fallback_setters'); xx.set_fallback_enabled(False)
        software = policy()
        require(software[0] == go['software_policy']['errors'] and software[1] is None and software[2] == go['software_policy']['buffer'], 'Software policy differs')
        spec = fixture.case_specs()[0]; require(spec == CASE, 'Exact frozen case0 differs')
        fixture.case_bounds(spec); observer.bump('fixtures'); args, options = fixture.fixture(spec, np)
        require(options == dict(ignore_labels=[0]), 'Unexpected normalized fixture options')
        record['inputs_before'] = [observer.array(v, retain=False) for v in args]
        requirements = fixture.native_requirements(spec)
        require(sum(requirements.values()) == 11, 'Native return closure changed')
        entries = {name: getattr(core, name) for name in requirements}
        require(all(inspect.isbuiltin(v) and v.__module__ == core.__name__ for v in entries.values()), 'Native builtin identity differs')
        phase = ['reference']; calls = dict(original=0, native=0); returned=[]; accepted=[]; suffix=[]
        record.update(counts=calls, requirements=requirements, returned=returned, accepted=accepted, suffix=suffix)
        guard = ctx['ownership_profile'](reference, native, owned.contingency_table, compat.fallback,
                                         entries, phase, calls, returned, accepted, suffix)
        def profile(frame, event, value):
            guard(frame, event, value)
            require(calls['original'] <= 1 and calls['native'] <= 1, 'Unexpected extra public target')
        caller = save(); record['natural_caller_raw32'] = bytes(caller).hex()
        for role, call in (('original', reference), ('native', native)):
            require(datetime.now(timezone.utc) < datetime.fromisoformat(go['deadline_utc']), 'Pair deadline expired')
            observer.phase = role; phase[0] = 'reference' if role == 'original' else 'native'
            np.log2 = observer.log2
            try:
                if role == 'original': sys.settrace(observer.trace)
                else: owned._core = observer.core_proxy()
                # Reuse the unchanged native structural/provider guard. It wraps the
                # observer, which delegates the saved true ufunc once per batch.
                manager = ctx['candidate_providers'](np, ctx['sparse']) if role == 'native' else ctx['nullcontext']()
                with manager, warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter('always'); restore(caller)
                    observer.bump(role); record['journal'].append(dict(role=role, event='entered'))
                    snapshots = []
                    record[role + '_FP'] = {}
                    def phase_env():
                        state = save(); snapshots.append(state)
                        name = 'before' if len(snapshots) == 1 else 'after'
                        record[role + '_FP'][name] = semantic(state)
                        record[role + '_FP'][name + '_raw32'] = bytes(state).hex()
                        return state
                    try:
                        value, before, after = ctx['call_once'](call, args, options, phase_env, profile, outputs,
                                                             'expected' if role == 'original' else 'actual', record['secondary'])
                    except BaseException as error:
                        remember(error, role + '-target-or-FP'); raise
                    record['journal'].append(dict(role=role, event='returned'))
                    record[role + '_warnings'] = [dict(category=v.category.__name__, message=str(v.message)) for v in notices]
            except BaseException as error:
                if error is not primary: remember(error, role)
                raise
            finally:
                uninstrument()
            if primary is not None: raise primary.with_traceback(primary_tb)
            if role == 'original':
                record['reference_policy_unchanged'] = same_policy(policy())
                record['reference_workers'] = workers()
                require(not record['original_warnings'] and record['reference_policy_unchanged'] and record['reference_workers'] == 1, 'Original changed policy/workers/warnings')
        record.update(counts=calls, requirements=requirements, returned=returned, accepted=accepted, suffix=suffix)
        record['inputs_after'] = [observer.array(v, retain=False) for v in args]
        record['late_policy_unchanged'] = same_policy(policy()); record['late_workers'] = workers()
        for name, value in outputs.items(): observer.capture(record['outputs'], name, lambda value=value: observer.array(value))
        expected, actual = outputs['expected'], outputs['actual']
        def aliases(a, b):
            observer.bump('alias_queries'); return bool(np.shares_memory(a, b))
        alias_rows = [aliases(expected, actual)] + [aliases(v, a) for v in (expected, actual) for a in args]
        expected_raw = record['outputs']['expected'].get('_raw_bytes'); actual_raw = record['outputs']['actual'].get('_raw_bytes')
        gates = dict(strict_words=(isinstance(expected_raw, bytes) and len(expected_raw) == 16 and expected_raw == actual_raw
                                  and expected.dtype.str == actual.dtype.str == '<f8' and list(expected.shape) == list(actual.shape) == [2]),
            FP=(record['original_FP']['before'] == record['native_FP']['before'] and record['original_FP']['after'] == record['native_FP']['after']),
            caller_entry=(record['original_FP']['before_raw32'] == record['native_FP']['before_raw32'] == bytes(caller).hex()),
            targets=(calls == dict(original=1, native=1)), returns=({k:returned.count(k) for k in requirements} == requirements and len(returned) == 11),
            ownership=(accepted == suffix == [True]), provider=(counts['original_ufunc'] == counts['native_ufunc'] == 2),
            inputs=(record['inputs_before'] == record['inputs_after']), outputs_owned=(expected.flags.owndata and actual.flags.owndata and not any(alias_rows)),
            policy=(record['late_policy_unchanged'] and record['late_workers'] == 1),
            identity=(owned._core is old_core and np.log2 is true_ufunc and upstream.variation_of_information is reference and owned.variation_of_information is native),
            warnings=(not record['original_warnings'] and not record['native_warnings']), diagnostic_capture=(not record['secondary']))
        record['gates'] = gates
        for name, passed in gates.items(): require(passed, 'Strict pair gate failed: ' + name)
    except BaseException as error:
        if error is not primary: remember(error, 'pair')
    finally:
        uninstrument()
        if caller is not None:
            try: record['caller_restored_raw32'] = restore(caller).hex()
            except BaseException as error: remember(error, 'caller-restore')
        # Capture each returned output even if a post-FP/context/assertion failed.
        for name, value in outputs.items():
            if name not in record['outputs']: observer.capture(record['outputs'], name, lambda value=value: observer.array(value))
        cleanup_all((
            ('worker', lambda: (observer.bump('worker_restores'), core.set_num_threads(old_workers)) if old_workers is not None else None),
            ('fallback', lambda: (observer.bump('fallback_setters'), xx.set_fallback_enabled(old_fallback)) if old_fallback is not None else None),
            ('errors', lambda: (observer.bump('software_setters'), np.seterr(**software[0])) if software is not None else None),
            ('handler', lambda: (observer.bump('software_setters'), np.seterrcall(software[1])) if software is not None else None),
            ('buffer', lambda: (observer.bump('software_setters'), np.setbufsize(software[2])) if software is not None else None)), remember)
        try:
            if software is not None:
                record['global_restored'] = same_policy(policy()) and workers() == old_workers and compat._fallback_enabled is old_fallback
                require(record['global_restored'], 'Global restoration failed')
            ctx['post_identity']()
            record['post_identity'] = True
        except BaseException as error: remember(error, 'post-identity/global-restore')
        try:
            if outer is not None: record['outer_restored_raw32'] = restore(outer).hex()
        except BaseException as error: remember(error, 'outer-restore')
        if primary is not None:
            record['primary'] = dict(type=type(primary).__name__, error=str(primary), traceback=''.join(traceback.format_exception(primary)))
        record['observed_counts_before_final_restore'] = dict(counts)
        record['final_writer_restore_pending'] = outer is not None
        record['status'] = 'STRICT_PAIR_FAILURE_RETAINED_NO_RETRY' if primary else 'INSTRUMENTED_PAIR_MATCH_NOT_QUALIFICATION'
        record['remaining_uncertainty'] = 'Native entropy vectors/products/final dot operands are unavailable. Equal CSR/provider buffers would not identify the first later differing operation.'
        try:
            with Path(go['output']).open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(encode_raw(record), stream, indent=2); stream.flush(); os.fsync(stream.fileno())
            record_written = True
        except BaseException as error: remember(error, 'writer')
        finally:
            final_closeout(ctx, record, counts, outputs, outer, restore, remember,
                           lambda: primary, go['output'], record_written)
    if primary is not None: raise primary.with_traceback(primary_tb)
    return ctx['_observer_terminal']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go', type=Path); parser.add_argument('--go-sha256')
    cli = parser.parse_args(argv)
    require(cli.go is not None and cli.go_sha256 is not None, 'UNBOUND: no hosted entry, imports or target calls')
    raw = cli.go.read_bytes(); require(hashlib.sha256(raw).hexdigest() == cli.go_sha256, 'External GO pin differs')
    go = json.loads(raw)
    check_go(go, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datetime.now(timezone.utc))
    require(sys.flags.isolated and sys.flags.no_site and not sys.flags.optimize, 'Use -I -S without -O')
    require(not Path(go['output']).exists(), 'Partial output exists; no overwrite/restart')
    runtime_raw = Path(go['runtime_binding']['path']).read_bytes()
    require(hashlib.sha256(runtime_raw).hexdigest() == go['runtime_binding']['sha256'], 'Root runtime binding differs')
    entry = load_pinned(go['hosted_entry'], '_root_bound_vi_stage_entry')
    # Root must review/authorize this future entry, including import/process-outer
    # restoration, source/runtime/core/F1/Git/CPU/capacity and post-identity bounds.
    ctx = entry.prepare(go, json.loads(runtime_raw))
    primary = primary_tb = None; terminal = None
    try:
        terminal = run_pair(ctx, go)
    except BaseException as error:
        primary, primary_tb = error, error.__traceback__
        if '_observer_terminal' not in ctx:
            ctx['_observer_terminal'] = dict(schema='vi-case0-stage-observer-terminal-v2',
                status='OBSERVER_ENTRY_VALIDATION_FAILURE', completed_two_returned_targets=False,
                counts={}, bounds_passed=None, full_FP_counts_passed=None,
                pending_reason='run_pair failed before its caller/cleanup envelope was entered',
                final_restore_attempted=False, final_restore_verified=False, last_restore_observation=None,
                primary=dict(type=type(error).__name__, error=str(error), traceback=''.join(traceback.format_exception(error))),
                secondary=[], outputs={}, raw_record_path=go['output'], raw_record_written=False,
                scope='Before observer caller envelope; no FP restoration claim')
    finally:
        try: entry.finish(ctx)
        except BaseException as error:
            if primary is None: primary, primary_tb = error, error.__traceback__
            else: primary.add_note('Hosted entry finalization also failed: ' + str(error))
    if primary is not None: raise primary.with_traceback(primary_tb)
    print(json.dumps(terminal, indent=2))


if __name__ == '__main__':
    main()
