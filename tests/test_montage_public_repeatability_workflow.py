"""Stdlib-only guards for the prospective full-public repeatability diagnostic."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/montage-public-repeatability.yml'
CONTRACT = ROOT / 'coverage/contracts/montage-public-repeatability.json'


def launcher_source():
    text = WORKFLOW.read_text()
    start = text.index('          cat > results/launch.py <<\'PY\'\n')
    body = text[start:].splitlines()[1:]
    end = body.index('          PY')
    return '\n'.join(line[10:] for line in body[:end]) + '\n'


class RepeatabilityWorkflowTests(unittest.TestCase):
    def test_frozen_products_order_and_sample_budget(self):
        c = json.loads(CONTRACT.read_bytes())
        self.assertEqual(c['order'], ['A', 'B', 'B', 'A', 'B', 'A', 'A', 'B'])
        self.assertEqual(c['same_product_null_pairs'], [[2, 3], [6, 7]])
        self.assertEqual(len(set(c['case_ids'])), 9)
        self.assertEqual((c['independent_subprocesses'], c['workers'], c['repeat'], c['target_ms']), (8, 1, 7, 19))
        self.assertEqual(c['raw_samples'], 8 * 9 * 7 * 2)
        self.assertEqual(c['samples_per_pass'], 126)
        self.assertEqual(c['products']['A']['revision'], '390dff892ad13c1f9b8cee0e76dce9a016f14c01')
        self.assertEqual(c['products']['B']['revision'], '61b4f79821b0c7fd0d224e2a12d4263d14308cb7')
        for product in c['products'].values():
            self.assertIn(str(product['run']), WORKFLOW.read_text())
            self.assertIn(str(product['artifact']), WORKFLOW.read_text())
            self.assertIn(product['sha256'], WORKFLOW.read_text())

    def test_inline_python_is_syntactically_valid(self):
        lines = WORKFLOW.read_text().splitlines()
        blocks = []
        for i, line in enumerate(lines):
            if line.strip().endswith("python - <<'PY'"):
                end = lines.index('          PY', i + 1)
                blocks.append('\n'.join(row[10:] for row in lines[i + 1:end]))
        self.assertGreaterEqual(len(blocks), 4)
        for block in blocks:
            ast.parse(block)
        ast.parse(launcher_source())

    def test_affinity_delegation_and_failure_retention_structure(self):
        source = launcher_source()
        tree = ast.parse(source)
        callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'retain_measure')
        delegates = [n for n in ast.walk(callback) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'delegate']
        self.assertEqual(len(delegates), 1)
        self.assertEqual([type(n).__name__ for n in delegates[0].args], ['Starred'])
        self.assertEqual(delegates[0].keywords[0].arg, None)
        self.assertLess(source.index('os.sched_setaffinity'), source.index('import run'))
        self.assertLess(source.index("process-before.json"), source.index('runpy.run_path'))
        self.assertIn("status='PENDING_FINAL_HARNESS_VALIDATION'", source)
        self.assertIn('stream.flush()', source)
        self.assertIn('finally:', source)
        self.assertIn('raise\n', source)
        self.assertNotIn('run.batch=', source)
        text = WORKFLOW.read_text()
        self.assertIn('1-baseline 2-candidate 3-candidate 4-baseline 5-candidate 6-baseline 7-baseline 8-candidate', text)
        self.assertIn('exit "$failed"', text)
        self.assertIn('if: always()', text)
        self.assertNotIn('g++', text)
        self.assertNotIn('pytest', text.replace("  PYTEST_ADDOPTS: ''", '').replace("  PYTEST_PLUGINS: ''", '').replace("  PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1'", '').lower())
        self.assertNotIn("status='KEEP'", text)
        self.assertNotIn("status='DISCARD'", text)

    def test_actual_launcher_success_and_partial_failure_without_science(self):
        source = launcher_source()
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                product = root / 'product'
                (product / 'benchmarks').mkdir(parents=True)
                run_file = product / 'benchmarks/run.py'
                run_file.write_bytes(b'# inert run module\n')
                c = json.loads(CONTRACT.read_bytes())
                c['measure_sha256'] = hashlib.sha256(run_file.read_bytes()).hexdigest()
                (root / 'dispatch-contract.json').write_text(json.dumps(c))
                output = root / '1-baseline'
                module = types.ModuleType('run')
                module.__file__ = str(run_file)
                calls = []
                timing = {'samples_ms': [[1] * 7, [2] * 7], 'loops': [3, 4]}

                def measure(*args, **kwargs):
                    calls.append((args, kwargs))
                    if fail and len(calls) == 2:
                        raise ValueError('retained first attempt')
                    return timing

                module.measure = measure
                reference, native = object(), object()

                def harness(path, run_name):
                    self.assertEqual(run_name, '__main__')
                    self.assertTrue((output / 'process-before.json').is_file())
                    for _ in range(9):
                        self.assertIs(module.measure(reference, native, 7, 19), timing)
                    (output / 'montage.json').write_text('{}')

                argv = ['launch.py', str(product), str(output), '7', '1-baseline', 'baseline']
                old_path = sys.path[:]
                try:
                    with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'run': module}), patch.object(runpy, 'run_path', harness), patch('os.sched_setaffinity', create=True) as affinity, patch('os.sched_getaffinity', return_value={7}, create=True):
                        if fail:
                            with self.assertRaisesRegex(ValueError, 'retained first attempt'):
                                exec(compile(source, '<inert-launcher>', 'exec'), {})
                        else:
                            exec(compile(source, '<inert-launcher>', 'exec'), {})
                        affinity.assert_called_once_with(0, {7})
                finally:
                    sys.path[:] = old_path
                final = json.loads((output / 'process-after.json').read_bytes())
                journal = [json.loads(line) for line in (output / 'returned-measures.jsonl').read_text().splitlines()]
                self.assertEqual(final['status'], 'FAILED' if fail else 'COMPLETE')
                self.assertEqual(final['completed_measures'], 1 if fail else 9)
                self.assertEqual(len(journal), 1 if fail else 9)
                self.assertEqual(len(calls), 2 if fail else 9)
                self.assertTrue(all(args == (reference, native, 7, 19) and kwargs == {} for args, kwargs in calls))
                self.assertEqual(journal[0]['timing'], timing)
                self.assertEqual([row['case_id'] for row in journal], c['case_ids'][:len(journal)])
                self.assertEqual(final['affinity_after'], [7])
                if fail:
                    self.assertEqual(final['failure']['type'], 'ValueError')
                    self.assertIn('retained first attempt', final['failure']['traceback'])


if __name__ == '__main__':
    unittest.main()
