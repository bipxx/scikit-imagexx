"""Metadata-only regression: retain actual/error/event evidence before assertions."""
import ast
import io
import json
from pathlib import Path

SOURCE = Path(__file__).with_name('test_li_u16_pipeline.py')


def test_raw_failed_observation_is_flushed_losslessly():
    tree = ast.parse(SOURCE.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_observation')
    namespace = dict(json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    class Journal(io.StringIO):
        flushed = False
        def flush(self): self.flushed = True
    journal = Journal()
    actual = dict(error=dict(type='RuntimeError', message='retained failure'), result=None)
    case = dict(id='synthetic-only', proposed_route='solver')
    events = [dict(event='c_exception', name='run')]
    namespace['write_observation'](journal, 1, case, actual, [], events, [])
    assert journal.flushed
    row = json.loads(journal.getvalue())
    assert row['observation'] == actual and row['compiled_events'] == events
    assert row['compiled_completions'] == [] and row['case'] == case


def test_journal_precedes_completion_and_parity_assertions():
    tree = ast.parse(SOURCE.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_installed_product')
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'write_observation']
    assertions = [n for n in ast.walk(node) if isinstance(n, ast.Assert) and 'completions' in ast.unparse(n)]
    assert len(calls) == len(assertions) == 1 and calls[0].lineno < assertions[0].lineno
    assert "int(route == 'solver')" in ast.unparse(assertions[0])


def _replay_compiled_events(events):
    from types import SimpleNamespace
    tree = ast.parse(SOURCE.read_bytes())
    audit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_installed_product')
    guard = next(n for n in ast.walk(audit) if isinstance(n, ast.FunctionDef) and n.name == 'guard')
    namespace = dict(forbidden=set(), attempts=[], compiled_events=[], completions=[])
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    for record in events:
        owner = None
        if record['owner_module'] is not None:
            owner = type('RecordedOwner', (), {'__module__': record['owner_module']})()
        argument = SimpleNamespace(__module__=record['module'], __name__=record['name'],
                                   __qualname__=record.get('qualname', record['name']), __self__=owner)
        namespace['guard'](None, record['event'], argument)
    return namespace['completions']


def test_real_c_return_metadata_accepts_callable_and_bound_owner_modules():
    # Actual40183231914 modern event shape; no native/import/image execution.
    for owner in ('pybind11_builtins', 'skimagexx._core'):
        module = 'skimagexx._core' if owner == 'pybind11_builtins' else None
        events = [dict(event=e, module=module, name='run', owner_module=owner)
                  for e in ('c_call', 'c_return')]
        assert _replay_compiled_events(events) == ['run']


def test_only_real_named_core_returns_count_as_solver_completion():
    base = dict(event='c_return', module='skimagexx._core', name='run', owner_module='pybind11_builtins')
    for change in (dict(event='c_call'), dict(event='c_exception'), dict(name='li_prepare'),
                   dict(module='unrelated')):
        assert _replay_compiled_events([dict(base, **change)]) == []
