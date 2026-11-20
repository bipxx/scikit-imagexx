"""Bounded family protocol observer; caller supplies target and per-action envelope.

No target resolution, authority, replay, or numerical implementation lives here.
"""
import inspect
import traceback
import warnings

from draw_coordinates_observer import encode
from order_angles_golden_ratio_contract import case_bounds, fixture


def error_record(error, np):
    return dict(module=type(error).__module__, name=type(error).__name__,
                qualname=type(error).__qualname__, args=encode(error.args, np),
                message=str(error))


def np_state(np):
    handler = np.geterrcall()
    return dict(policy=np.geterr(), buffer=np.getbufsize(),
                handler_type=[type(handler).__module__, type(handler).__qualname__],
                handler_id=id(handler))


def observe(call, spec, np, envelope, events=None):
    """One construction and bounded resumes. EntryRefused propagates, never retries.

    envelope(action, kind) returns (value, Exception-or-None, immediate-FP-record),
    restoring its actual outer caller after the immediate outgoing snapshot.
    JSON/encoding occurs outside that target interval. BaseException refusals
    must propagate. No source or scalar word is normalized.
    """
    bounds = case_bounds(spec)
    args, options = fixture(spec, np)
    incoming = np_state(np)
    rows = [] if events is None else events
    assert not rows
    generator = None
    yields = 0
    resumes = 0
    mutation_done = False

    def step(kind, action):
        before = encode((args, options), np)
        policy = np_state(np)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            value, error, fp = envelope(action, kind)
        # The action has consumed authority. Retain its immediate hardware and
        # caller-restoration proof before ANY post-call codec/policy operation.
        row = dict(action=kind, phase='consumed', consumed=True, immediate_fp=fp,input_before=before,numpy_before=policy,
                   value_returned=error is None,
                   target_error_type=None if error is None else [type(error).__module__, type(error).__qualname__],
                   warning_types=[[type(w.message).__module__,type(w.message).__qualname__]for w in notices])
        rows.append(row)
        row['failure_stage']='input-after-encoding'
        row.update(input_before=before, input_after=encode((args, options), np), numpy_before=policy)
        row['failure_stage']='numpy-state'
        row['numpy_after']=np_state(np)
        row['failure_stage']='error-encoding'
        row['error']=None if error is None else error_record(error, np)
        row['failure_stage']='warning-encoding'
        row['warnings']=[dict(module=type(w.message).__module__,qualname=type(w.message).__qualname__,message=str(w.message))for w in notices]
        row['failure_stage']='value-alias-encoding'
        if kind == 'construct' and error is None:
            row['generator'] = dict(type=[type(value).__module__, type(value).__qualname__],
                                    is_generator=inspect.isgenerator(value),
                                    state=inspect.getgeneratorstate(value))
        else:
            row['value'] = None if error else encode(value, np)
            row['value_type'] = None if error else [type(value).__module__, type(value).__qualname__]
            row['value_is_theta'] = bool(args and value is args[0])
            row['value_input_alias'] = None
            if error is None and args and isinstance(args[0], np.ndarray) and isinstance(value, (np.ndarray, np.generic)):
                row['value_input_alias'] = dict(identity=value is args[0], base_is_theta=getattr(value, 'base', None) is args[0], shares_memory=bool(np.shares_memory(value, args[0])))
            row['generator_state'] = None if generator is None else inspect.getgeneratorstate(generator)
        row['failure_stage']='numpy-policy-assertion'
        assert policy == row['numpy_after'], 'TARGET_CHANGED_NUMPY_STATE'
        row['phase']='observed';row['failure_stage']=None
        return value, error

    primary_failure = False
    try:
        generator, error = step('construct', lambda: call(*args, **options))
        if error is not None:
            return finish(spec, rows, yields, resumes)
        assert inspect.isgenerator(generator), 'PUBLIC_RESULT_NOT_GENERATOR'
        if spec['protocol'] == 'close-created':
            step('close-created', generator.close)
        else:
            while resumes < bounds['resume_attempts_upper']:
                mutation = spec['mutation']
                if mutation and not mutation_done and yields == mutation['after_yields']:
                    step('caller-mutation', lambda: args[0].__setitem__(mutation['index'], mutation['value']))
                    mutation_done = True
                protocol = spec['protocol']
                if protocol == 'non-none-send-created':
                    action, name = lambda: generator.send(1), 'send-non-none-created'
                elif protocol == 'send-none-exhaust':
                    action, name = lambda: generator.send(None), 'send-none'
                elif protocol == 'send-token-after-first' and yields:
                    action, name = lambda: generator.send(123), 'send-token'
                else:
                    action, name = lambda: next(generator), 'next'
                resumes += 1
                _, error = step(name, action)
                if error is not None:
                    break
                yields += 1
                assert yields <= bounds['successful_yields_upper']
                if protocol == 'prefix-close':
                    step('prefix-close', generator.close)
                    break
            else:
                raise AssertionError('GENERATOR_EXCEEDED_FIXED_RESUME_BOUND')
        return finish(spec, rows, yields, resumes)
    except BaseException as error:
        primary_failure = True
        rows.append(dict(action='observer-failure',phase='failed',stage=rows[-1].get('failure_stage')if rows else'pre-action',
                         type=type(error).__qualname__,message=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if inspect.isgenerator(generator) and inspect.getgeneratorstate(generator) != 'GEN_CLOSED':
                step('cleanup-close', generator.close)
            assert np_state(np) == incoming
        except BaseException as error:
            if not primary_failure:raise
            rows.append(dict(action='observer-cleanup-failure',phase='failed',type=type(error).__qualname__,
                             message=str(error),traceback=traceback.format_exc()))


def finish(spec, rows, yields, resumes):
    return dict(id=spec['id'], actions=rows, yields=yields, resume_attempts=resumes)


def counters(row):
    actions = row['actions']
    construction = next(r for r in actions if r['action'] == 'construct')
    return dict(api_attempts=1, construction_requests=1,
                successful_constructions=int(construction['value_returned']),
                next_requests=sum(r['action']=='next' for r in actions),
                send_none_requests=sum(r['action']=='send-none' for r in actions),
                send_token_requests=sum(r['action']=='send-token' for r in actions),
                send_non_none_created_requests=sum(r['action']=='send-non-none-created' for r in actions),
                close_requests=sum(r['action'] in ('close-created','prefix-close','cleanup-close') for r in actions),
                resume_requests=row['resume_attempts'], yielded_values=row['yields'],
                mutations=sum(r['action']=='caller-mutation' for r in actions),
                action_errors=sum(r.get('error') is not None for r in actions))
