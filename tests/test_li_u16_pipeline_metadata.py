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
